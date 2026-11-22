from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import neurokit2 as nk
from scipy.signal import butter, sosfiltfilt

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

_NK_CLEAN_METHOD = {
    "neurokit": "neurokit",
    "pantompkins1985": "pantompkins1985",
    "hamilton2002": "hamilton2002",
    "elgendi2010": "elgendi2010",
    "engzeemod2012": "engzeemod2012",
}


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def estimate_polarity(ecg: np.ndarray, fs: float, win_ms: float = 60.0) -> tuple[int, float]:
    qrs_band = bandpass(ecg, fs, 5.0, 20.0)
    energy = np.convolve(qrs_band**2, np.ones(int(0.08 * fs)) / int(0.08 * fs), mode="same")
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


def refine_peaks(
    ecg: np.ndarray, peaks: np.ndarray, fs: float, polarity: int = 1, win_ms: float = 50.0,
    mode: str = "polarity",
) -> np.ndarray:
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
    out = np.unique(out)
    if out.size > 1:
        keep = np.concatenate([[True], np.diff(out) > int(0.2 * fs)])
        out = out[keep]
    return out


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

        xq = processing.XQRS(sig=np.asarray(x, dtype=float), fs=fs)
        xq.detect(verbose=False)
        return np.asarray(xq.qrs_inds, dtype=int)

    if preprocessing == "recommended":
        x_in = nk.ecg_clean(x, sampling_rate=fs, method=_NK_CLEAN_METHOD.get(method, "neurokit"))
    else:
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
    t0 = time.perf_counter()
    try:
        if polarity is None:
            polarity = estimate_polarity(ecg_raw, fs)[0] if (fix_polarity or refine) else 1
        sign = polarity if fix_polarity else 1
        if preprocessing == "shared" and method.startswith("christov"):
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
    except Exception as exc:
        return DetectionResult(method, np.array([], dtype=int), time.perf_counter() - t0, error=repr(exc))
