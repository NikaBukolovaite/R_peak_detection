from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

FS = 200


def load_signal(path: Path) -> np.ndarray:
    x = np.load(path)
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(float) * 0.001
    return x.astype(float)


def review_copy(sig: np.ndarray) -> np.ndarray:
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
