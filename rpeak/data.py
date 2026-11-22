"""Loading the recordings.

Expected layout of the data folder (the data are not part of this repository):

    data/records.csv            record, quality, groups
    data/<record>.npy           single-lead ECG at 200 Hz: int16 in microvolts, or float in millivolts
    data/<record>.baseline.json {"baseline": [sample indices of the R-peak marks stored with the recording]}
    data/<record>.vote.json     optional, {"vote": [...]}: the vote the poster used (see README)

quality is clean, noisy or pacemaker. groups is a space-separated list of problem groups
(t_wave, small_peak, negative_qrs, missed_beats, unclear), possibly empty.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

FS = 200


def load_signal(path: Path) -> np.ndarray:
    x = np.load(path)
    if np.issubdtype(x.dtype, np.integer):     # microvolts
        return x.astype(float) * 0.001
    return x.astype(float)                       # millivolts


def review_copy(sig: np.ndarray) -> np.ndarray:
    """The 16-bit copy of a recording that the evaluation steps were run on.

    Median removed, rounded to 1 microvolt and clipped at +-32.767 mV (only recording 1716089,
    with a saturated segment, reaches the clip). The detectors ran on the recording as read; the
    vote position, beat shapes, Elgendi and the figures used this copy. Keeping both reproduces
    the poster numbers exactly.
    """
    return np.clip(np.round((sig - np.median(sig)) * 1000), -32767, 32767) * 0.001


def load_records(data_dir: str | Path) -> dict[str, dict]:
    data_dir = Path(data_dir)
    out = {}
    with open(data_dir / "records.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rec = row["record"]
            base = json.loads((data_dir / f"{rec}.baseline.json").read_text(encoding="utf-8"))
            out[rec] = {
                "sig": load_signal(data_dir / f"{rec}.npy"),
                "baseline": np.sort(np.asarray(base["baseline"], dtype=int)),
                "quality": row["quality"],
                "groups": row["groups"].split(),
            }
    return out
