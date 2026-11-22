"""R-peak detectors, polarity handling and local R refinement for 200 Hz ECG.

All detectors return integer sample indices into the *input* signal.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import neurokit2 as nk
from scipy.signal import butter, sosfiltfilt

# NeuroKit methods that ecg_peaks accepts and that work at 200 Hz.
NK_METHODS = [
    "neurokit",
    "pantompkins1985",
    "hamilton2002",
    "christov2004",
    "elgendi2010",
    "engzeemod2012",
    "rodrigues2021",
]
ALL_METHODS = NK_METHODS + ["xqrs"]
DEFAULT_METHODS = ["neurokit", "pantompkins1985", "hamilton2002", "christov2004", "xqrs"]

# ecg_clean has its own preprocessing only for some methods; the rest use NeuroKit's default.
_NK_CLEAN_METHOD = {
    "neurokit": "neurokit",
    "pantompkins1985": "pantompkins1985",
    "hamilton2002": "hamilton2002",
    "elgendi2010": "elgendi2010",
    "engzeemod2012": "engzeemod2012",
}


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth bandpass (second-order sections)."""
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


# --------------------------------------------------------------------------- polarity
def estimate_polarity(ecg: np.ndarray, fs: float, win_ms: float = 60.0) -> tuple[int, float]:
    """Estimate dominant QRS polarity independently of any R detector.

    QRS candidates are taken from the energy of the 5-20 Hz signal (polarity-blind),
    then at each candidate the sign of the largest deviation in the 0.5-40 Hz signal
    is checked. Returns (+1 or -1, fraction of candidates that are negative).
    """
    qrs_band = bandpass(ecg, fs, 5.0, 20.0)
    energy = np.convolve(qrs_band**2, np.ones(int(0.08 * fs)) / int(0.08 * fs), mode="same")
    # candidate = local energy maxima above a robust threshold, at least 250 ms apart
    from scipy.signal import find_peaks

    thr = np.median(energy) + 3 * np.median(np.abs(energy - np.median(energy)))
    cand, _ = find_peaks(energy, height=thr, distance=int(0.25 * fs))
    if cand.size < 5:
        return 1, 0.0
    broad = bandpass(ecg, fs, 0.5, 40.0)
    half = int(win_ms / 1000 * fs)
    signs = []
    for c in cand:
        seg = broad[max(c - half, 0): c + half + 1]
        if seg.size == 0:
            continue
        seg = seg - np.median(seg)
        signs.append(np.sign(seg[np.argmax(np.abs(seg))]))
    neg_frac = float(np.mean(np.asarray(signs) < 0))
    return (-1 if neg_frac > 0.5 else 1), neg_frac


# --------------------------------------------------------------------------- refinement
def refine_peaks(
    ecg: np.ndarray, peaks: np.ndarray, fs: float, polarity: int = 1, win_ms: float = 50.0,
    mode: str = "polarity",
) -> np.ndarray:
    """Move each detection to the R extremum within +-win_ms.

    mode="polarity": max of polarity*signal (use after polarity estimation);
    mode="absmax":   largest |deviation| from the local median (handles mixed polarity, e.g. PVCs).
    """
    if peaks.size == 0:
        return peaks
    half = int(win_ms / 1000 * fs)
    out = np.empty_like(peaks)
    for i, p in enumerate(peaks):
        lo, hi = max(p - half, 0), min(p + half + 1, len(ecg))
        seg = ecg[lo:hi]
        if mode == "absmax":
            seg = np.abs(seg - np.median(seg))
        else:
            seg = polarity * seg
        out[i] = lo + int(np.argmax(seg))
    # merge detections that collapsed onto the same/near sample (< 200 ms)
    out = np.unique(out)
    if out.size > 1:
        keep = np.concatenate([[True], np.diff(out) > int(0.2 * fs)])
        out = out[keep]
    return out


# --------------------------------------------------------------------------- detectors
@dataclass
class DetectionResult:
    method: str
    peaks: np.ndarray
    runtime_s: float
    error: str | None = None
    extra: dict = field(default_factory=dict)


def _run_single(x: np.ndarray, fs: float, method: str, preprocessing: str) -> np.ndarray:
    if method == "xqrs":
        from wfdb import processing

        # XQRS does its own bandpass internally; feed it the provided signal.
        xq = processing.XQRS(sig=np.asarray(x, dtype=float), fs=fs)
        xq.detect(verbose=False)
        return np.asarray(xq.qrs_inds, dtype=int)

    if preprocessing == "recommended":
        x_in = nk.ecg_clean(x, sampling_rate=fs, method=_NK_CLEAN_METHOD.get(method, "neurokit"))
    else:  # "shared": signal already filtered by the caller
        x_in = x
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, info = nk.ecg_peaks(x_in, sampling_rate=fs, method=method, correct_artifacts=False)
    return np.asarray(info.get("ECG_R_Peaks", []), dtype=int)


def run_detector(
    ecg_raw: np.ndarray,
    fs: float,
    method: str,
    *,
    preprocessing: str = "shared",
    shared_band: tuple[float, float] = (0.5, 40.0),
    fix_polarity: bool = False,
    refine: str | None = None,
    polarity: int | None = None,
) -> DetectionResult:
    """Run one detector.

    preprocessing: "recommended" -> NeuroKit method-specific ecg_clean on the raw signal;
                   "shared"      -> every method gets the same bandpass(shared_band) signal.
    fix_polarity:  invert the signal before detection if dominant QRS is negative.
    refine:        None | "polarity" | "absmax" -- snap peaks to the R extremum of the
                   0.5-40 Hz signal (+-50 ms).
    """
    t0 = time.perf_counter()
    try:
        if polarity is None:
            polarity = estimate_polarity(ecg_raw, fs)[0] if (fix_polarity or refine) else 1
        sign = polarity if fix_polarity else 1
        if preprocessing == "shared" and method.startswith("christov"):
            # Christov's adaptive thresholds break on a narrow 5-20 Hz input and on NeuroKit's
            # 0.5 Hz high-pass cleaning (thousands of false beats on these records), but work on a
            # 0.5-40 Hz band, so it always gets that.
            x = bandpass(ecg_raw, fs, 0.5, 40.0) * sign
        elif preprocessing == "shared":
            x = bandpass(ecg_raw, fs, *shared_band) * sign
        else:
            x = ecg_raw * sign
        peaks = _run_single(x, fs, method, preprocessing)
        if refine:
            ref_sig = bandpass(ecg_raw, fs, 0.5, 40.0)
            peaks = refine_peaks(ref_sig, peaks, fs, polarity=polarity, mode=refine)
        peaks = peaks[(peaks >= 0) & (peaks < len(ecg_raw))]
        return DetectionResult(method, np.sort(peaks), time.perf_counter() - t0,
                               extra={"polarity": int(polarity)})
    except Exception as exc:  # keep the benchmark running if one detector fails
        return DetectionResult(method, np.array([], dtype=int), time.perf_counter() - t0, error=repr(exc))
