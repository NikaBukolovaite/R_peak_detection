"""Elgendi's blocks-of-interest R-peak detector.

Method: M. Elgendi, "Fast QRS detection with an optimized knowledge-based method",
PLOS ONE 8 (2013) e73557. Settings (9-25 Hz band, 0.2 s and 0.82 s moving averages, beta 0.08,
final move to the largest absolute value within 0.13275 s) are the ones used by
Wolf et al. 2026, Sci Rep 16:15748 [5], where Elgendi was the best of 17 detectors.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def elgendi_blocks(x: np.ndarray, fs: float, band: tuple[float, float] = (9.0, 25.0), order: int = 3,
                   qrs_win_s: float = 0.2, beat_win_s: float = 0.82, beta: float = 0.08,
                   snap_s: float = 0.13275) -> np.ndarray:
    """Return R-peak sample indices found by the blocks-of-interest method."""
    x = np.asarray(x, dtype=float)
    b, a = butter(order, [band[0] / (0.5 * fs), band[1] / (0.5 * fs)], btype="band")
    y = filtfilt(b, a, x)
    energy = y ** 2

    n_qrs, n_beat = int(qrs_win_s * fs), int(beat_win_s * fs)
    ma_qrs = np.convolve(energy, np.ones(n_qrs), mode="same") / n_qrs      # QRS-length moving average
    ma_beat = np.convolve(energy, np.ones(n_beat), mode="same") / n_beat   # beat-length moving average
    inside = ma_qrs > ma_beat + beta * np.mean(energy)                      # blocks of interest

    # one candidate per block: the highest point of the filtered signal inside the block
    edges = np.diff(np.concatenate([[0], inside.astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    candidates = [s + int(np.argmax(y[s:t])) for s, t in zip(starts, stops)]

    # move every candidate to the largest absolute value of the input nearby
    half = int(snap_s * fs)
    magnitude = np.abs(x)
    peaks = {c - half + int(np.argmax(magnitude[c - half:c + half]))
             for c in candidates if c - half >= 0 and c + half < len(x)}
    return np.array(sorted(peaks), dtype=int)
