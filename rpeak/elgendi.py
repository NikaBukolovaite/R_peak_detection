from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def elgendi_blocks(x: np.ndarray, fs: float, band: tuple[float, float] = (9.0, 25.0), order: int = 3,
                   qrs_win_s: float = 0.2, beat_win_s: float = 0.82, beta: float = 0.08,
                   snap_s: float = 0.13275) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    b, a = butter(order, [band[0] / (0.5 * fs), band[1] / (0.5 * fs)], btype="band")
    y = filtfilt(b, a, x)
    energy = y ** 2

    n_qrs, n_beat = int(qrs_win_s * fs), int(beat_win_s * fs)
    ma_qrs = np.convolve(energy, np.ones(n_qrs), mode="same") / n_qrs
    ma_beat = np.convolve(energy, np.ones(n_beat), mode="same") / n_beat
    inside = ma_qrs > ma_beat + beta * np.mean(energy)

    edges = np.diff(np.concatenate([[0], inside.astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    candidates = [s + int(np.argmax(y[s:t])) for s, t in zip(starts, stops)]

    half = int(snap_s * fs)
    magnitude = np.abs(x)
    peaks = {c - half + int(np.argmax(magnitude[c - half:c + half]))
             for c in candidates if c - half >= 0 and c + half < len(x)}
    return np.array(sorted(peaks), dtype=int)
