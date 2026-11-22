from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import neurokit2 as nk
import numpy as np

from rpeak.data import load_records, review_copy
from rpeak.detectors import bandpass, estimate_polarity, refine_peaks, run_detector
from rpeak.elgendi import elgendi_blocks
from rpeak.metrics import consensus_peaks

FS = 200.0
BAND = (5.0, 20.0)
METHODS = ["neurokit", "pantompkins1985", "hamilton2002", "christov2004", "xqrs"]
VOTERS = ["neurokit", "pantompkins1985", "hamilton2002", "xqrs"]
VOTE_TOL_MS = 150.0


def detect_all(ecg: np.ndarray) -> dict[str, np.ndarray]:
    polarity = estimate_polarity(ecg, FS)[0]
    marks = {}
    for suffix, fix, refine in (("", False, None), ("+fix", True, "polarity")):
        for m in METHODS:
            r = run_detector(ecg, FS, m, preprocessing="shared", shared_band=BAND,
                             fix_polarity=fix, refine=refine, polarity=polarity)
            if r.error:
                print(f"    {m + suffix} failed: {r.error}")
            marks[m + suffix] = r.peaks

    x = nk.signal_filter(ecg, sampling_rate=FS, method="butterworth", lowcut=0.5, highcut=10, order=5)
    _, info = nk.ecg_peaks(x, sampling_rate=FS, method="neurokit", correct_artifacts=False)
    marks["neurokit_0.5-10"] = np.asarray(info["ECG_R_Peaks"], dtype=int)

    ev = review_copy(ecg)
    ev_polarity = estimate_polarity(ev, FS)[0]
    narrow = bandpass(ev, FS, *BAND)
    marks["elgendi"] = elgendi_blocks(narrow, FS)
    fixed = elgendi_blocks(narrow * ev_polarity, FS)
    marks["elgendi+fix"] = refine_peaks(bandpass(ev, FS, 0.5, 40.0), fixed, FS, polarity=ev_polarity, mode="polarity")
    return marks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    (args.out / "marks").mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")

    for rec, d in load_records(args.data).items():
        print(f"{rec}: {len(d['sig']) / FS / 60:.1f} min")
        marks = detect_all(d["sig"])
        vote, _ = consensus_peaks({m: marks[m] for m in VOTERS}, FS, VOTE_TOL_MS)
        stored = args.data / f"{rec}.vote.json"
        if stored.exists():
            marks["vote_recomputed"] = vote
            marks["vote"] = np.asarray(json.loads(stored.read_text(encoding="utf-8"))["vote"], dtype=int)
        else:
            marks["vote"] = vote
        marks["baseline"] = d["baseline"]
        out = {m: [int(v) for v in np.sort(p)] for m, p in marks.items()}
        (args.out / "marks" / f"{rec}.json").write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
