"""Step 2: every number on the poster, from the marks of step 1.

    python evaluate.py --data data --marks results/marks --out results

Reference: the vote of step 1, each beat moved to the largest deflection of the 0.5-40 Hz signal
within 100 ms (the dominant QRS extremum, whatever its polarity). A beat counts as found when a mark
lies within 150 ms of it (one mark per beat). Missed beats are counted out of the vote beats, extra
marks out of the marks of the method.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from rpeak.data import load_records, review_copy
from rpeak.detectors import bandpass

FS = 200
T150, T75, T25 = 30, 15, 5            # tolerances in samples
PRE, POST = int(0.2 * FS), int(0.4 * FS)


def tol(ms: float) -> int:
    return int(ms / 1000 * FS)


def match(ref, det, t):
    """One-to-one matching within t samples, closest pairs first. Returns sorted ref, det and matched indices."""
    ref = np.sort(np.asarray(ref)); det = np.sort(np.asarray(det)); pairs = []; j0 = 0
    for i, x in enumerate(ref):
        while j0 < len(det) and det[j0] < x - t:
            j0 += 1
        j = j0
        while j < len(det) and det[j] <= x + t:
            pairs.append((abs(det[j] - x), i, j)); j += 1
    pairs.sort(); used_r, used_d, mr, md = set(), set(), [], []
    for _, i, j in pairs:
        if i not in used_r and j not in used_d:
            used_r.add(i); used_d.add(j); mr.append(i); md.append(j)
    return ref, det, np.array(mr, int), np.array(md, int)


def vote_at_extremum(broad, vote):
    h = int(0.1 * FS); out = []
    for p in vote:
        lo, hi = max(p - h, 0), min(p + h + 1, len(broad)); seg = broad[lo:hi]
        out.append(lo + int(np.argmax(np.abs(seg - np.median(seg)))))
    e = np.unique(out)
    keep = np.concatenate([[True], np.diff(e) > int(0.2 * FS)]) if e.size > 1 else np.array([True])
    return e[keep]


def beat_groups(broad, p, threshold=0.2):
    """Cut each beat from 200 ms before to 400 ms after its mark and group the beats by shape."""
    p = np.sort(np.asarray(p)); p = p[(p - PRE >= 0) & (p + POST < len(broad))]
    W = np.stack([broad[q - PRE:q + POST] for q in p])
    W = W - W.mean(1, keepdims=True); W /= (W.std(1, keepdims=True) + 1e-9)
    if len(W) < 3:
        return p, np.ones(len(W), int)
    return p, fcluster(linkage(W, method="average", metric="correlation"), t=threshold, criterion="distance")


def scores(recs, ref, marks, m, t=T150):
    f1s, TP, FN, FP, offs, per = [], 0, 0, 0, [], {}
    for k in recs:
        r, d, mr, md = match(ref[k], marks[k][m], t)
        tp = len(mr); fn = len(r) - tp; fp = len(d) - tp
        f1 = 2 * tp / (2 * tp + fp + fn); f1s.append(f1); per[k] = dict(f1=f1, missed=fn, extra=fp, n=len(d))
        TP += tp; FN += fn; FP += fp; offs += list((d[md] - r[mr]) * 1000 / FS)
    offs = np.asarray(offs)
    return dict(mean_f1=float(np.mean(f1s)), missed_pct=100 * FN / (TP + FN), extra_pct=100 * FP / (TP + FP),
                within25_pct=100 * float(np.mean(np.abs(offs) <= 25)), per_record=per)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--marks", type=Path, default=Path("results/marks"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    recs = load_records(args.data)
    marks = {k: json.loads((args.marks / f"{k}.json").read_text(encoding="utf-8")) for k in recs}
    broad = {k: bandpass(review_copy(d["sig"]), FS, 0.5, 40.0) for k, d in recs.items()}
    ref = {k: vote_at_extremum(broad[k], marks[k]["vote"]) for k in recs}
    for k in recs:
        marks[k]["vote_at_extremum"] = ref[k].tolist()
    out = {}

    # ---------------------------------------------------------------- data (Section 2)
    mins = [len(d["sig"]) / FS / 60 for d in recs.values()]
    out["data"] = dict(records=len(recs), minutes_min=min(mins), minutes_max=max(mins), hours=sum(mins) / 60,
                       vote_beats=int(sum(len(r) for r in ref.values())),
                       baseline_marks=int(sum(len(d["baseline"]) for d in recs.values())),
                       quality={q: sum(d["quality"] == q for d in recs.values()) for q in ("clean", "noisy", "pacemaker")})

    # ---------------------------------------------------------------- Table 2: baseline errors by group
    rows = {}
    for k, d in recs.items():
        r, det, mr, md = match(ref[k], d["baseline"], T150)
        rows[k] = dict(vote=len(r), marks=len(det), missed=len(r) - len(mr), extra=len(det) - len(md))
    groups = {"All": list(recs), "Clean": [k for k in recs if recs[k]["quality"] == "clean"],
              "Noisy": [k for k in recs if recs[k]["quality"] == "noisy"],
              "Pacemaker": [k for k in recs if recs[k]["quality"] == "pacemaker"]}
    for g, name in (("t_wave", "T-wave errors"), ("small_peak", "Small-peak errors"), ("negative_qrs", "Negative QRS")):
        groups[name] = [k for k in recs if g in recs[k]["groups"]]
    t2 = {}
    for name, ks in groups.items():
        v = sum(rows[k]["vote"] for k in ks); c = sum(rows[k]["marks"] for k in ks)
        t2[name] = dict(records=len(ks), missed_pct=100 * sum(rows[k]["missed"] for k in ks) / v,
                        extra_pct=100 * sum(rows[k]["extra"] for k in ks) / c)
    err = np.array([100 * (r["missed"] + r["extra"]) / r["vote"] for r in rows.values()])
    out["table2"] = dict(groups=t2, missed=sum(r["missed"] for r in rows.values()),
                         extra=sum(r["extra"] for r in rows.values()),
                         per_record_error_pct=dict(median=float(np.median(err)), q1=float(np.percentile(err, 25)),
                                                   q3=float(np.percentile(err, 75)), max=float(err.max()),
                                                   records_above_5pct=int(np.sum(err > 5))))

    # ---------------------------------------------------------------- Table 3: methods vs the vote
    methods = ["xqrs+fix", "pantompkins1985+fix", "hamilton2002+fix", "christov2004+fix", "neurokit",
               "neurokit_0.5-10", "elgendi+fix", "baseline",
               "xqrs", "pantompkins1985", "hamilton2002", "christov2004", "neurokit+fix", "elgendi"]
    out["table3"] = {m: scores(recs, ref, marks, m) for m in methods}

    # ---------------------------------------------------------------- Table 1: example recordings
    t1 = {}
    for k in ["1704502.531", "1715213.477", "1652786.026", "1716089.465", "1633509.831", "1640106.330",
              "1640137.809", "1640159.572"]:
        if k in recs:
            t1[k] = {m: {x: out["table3"][m]["per_record"][k][x] for x in ("missed", "extra")}
                     for m in ("baseline", "xqrs+fix")}
            t1[k]["vote_beats"] = rows[k]["vote"]
    out["table1"] = t1

    # ---------------------------------------------------------------- tolerance (conclusions)
    out["tolerance_mean_f1"] = {m: {ms: scores(recs, ref, marks, m, tol(ms))["mean_f1"] for ms in (150, 75, 25)}
                                for m in ("xqrs+fix", "neurokit", "neurokit+fix")}

    # ---------------------------------------------------------------- mark position details
    k = "1715135.145"
    if k in recs:
        med = {}
        for m in ("hamilton2002", "hamilton2002+fix"):
            r, d, mr, md = match(ref[k], marks[k][m], T150)
            med[m] = float(np.median((d[md] - r[mr]) * 1000 / FS))
        out["hamilton_1715135_median_offset_ms"] = med

    # ---------------------------------------------------------------- Fig. 3: beat-shape groups
    outside = {m: [0, 0] for m in ("baseline", "vote_at_extremum", "xqrs+fix")}
    per_rec = {}
    for k in recs:
        per_rec[k] = {}
        for m in outside:
            _, lab = beat_groups(broad[k], marks[k][m])
            n_out = int(len(lab) - np.bincount(lab).max())
            outside[m][0] += n_out; outside[m][1] += len(lab)
            per_rec[k][m] = dict(beats=len(lab), outside=n_out, outside_pct=100 * n_out / len(lab))
    extra_out = extra_all = 0
    for k, d in recs.items():
        r, det, mr, md = match(ref[k], d["baseline"], T150)
        is_extra = np.ones(len(det), bool); is_extra[md] = False
        p, lab = beat_groups(broad[k], det)
        main = np.argmax(np.bincount(lab)); kept = np.isin(det, p)
        extra_all += int(is_extra[kept].sum()); extra_out += int(np.sum(is_extra[kept] & (lab != main)))
    out["fig3"] = dict(outside_main_group_pct={m: 100 * a / b for m, (a, b) in outside.items()},
                       baseline_extra_marks_outside=[extra_out, extra_all], per_record=per_rec)

    # ---------------------------------------------------------------- report
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "poster_numbers.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    D = out["data"]; T2 = out["table2"]; P = T2["per_record_error_pct"]
    print(f"Data: {D['records']} recordings, {D['minutes_min']:.1f}-{D['minutes_max']:.1f} min each, "
          f"{D['hours']:.1f} h, {D['vote_beats']} vote beats, {D['baseline_marks']} baseline marks, {D['quality']}")
    print("\nTable 2. Baseline method vs the vote (150 ms)")
    for g, v in T2["groups"].items():
        print(f"  {g:18s} n={v['records']:2d}  missed {v['missed_pct']:5.1f} %  extra {v['extra_pct']:5.1f} %")
    print(f"  per recording: median {P['median']:.1f} % (IQR {P['q1']:.1f}-{P['q3']:.1f} %, max {P['max']:.0f} %), "
          f"{P['records_above_5pct']} recordings above 5 %")
    print("\nTable 3. Methods vs the vote (150 ms)")
    for m, s in out["table3"].items():
        print(f"  {m:20s} F1 {s['mean_f1']:.3f}  missed {s['missed_pct']:5.2f} %  extra {s['extra_pct']:5.2f} %"
              f"  within 25 ms {s['within25_pct']:5.1f} %")
    e = out["table3"]["elgendi+fix"]["per_record"]
    worst = min(e, key=lambda k: e[k]["f1"])
    print(f"  Elgendi with fix: worst {worst} F1 {e[worst]['f1']:.3f}, {e[worst]['n']} marks for "
          f"{rows[worst]['vote']} beats, {e[worst]['missed']} beats missed; other recordings F1 at least "
          f"{min(v['f1'] for k, v in e.items() if k != worst):.3f}")
    print("\nTable 1. Example recordings (baseline method | XQRS with fix)")
    for k, v in out["table1"].items():
        print(f"  {k}: missed {v['baseline']['missed']}, extra {v['baseline']['extra']} | "
              f"missed {v['xqrs+fix']['missed']}, extra {v['xqrs+fix']['extra']}  (vote beats {v['vote_beats']})")
    print("\nTolerance, mean F1 at 150 / 75 / 25 ms")
    for m, v in out["tolerance_mean_f1"].items():
        print(f"  {m:14s} {v[150]:.3f} / {v[75]:.3f} / {v[25]:.3f}")
    if "hamilton_1715135_median_offset_ms" in out:
        print("\n1715135 Hamilton median offset (ms):", {m: round(v) for m, v in out["hamilton_1715135_median_offset_ms"].items()})
    F = out["fig3"]
    print("\nFig. 3. Share of marks outside the main beat-shape group:",
          {m: round(v, 1) for m, v in F["outside_main_group_pct"].items()})
    print(f"  baseline extra marks outside the main group: {F['baseline_extra_marks_outside'][0]} of "
          f"{F['baseline_extra_marks_outside'][1]}")
    for k in ("1652786.026", "1645756.288"):
        if k in recs:
            print(f"  {k}:", {m: (v["beats"], v["outside"], round(v["outside_pct"], 1)) for m, v in F["per_record"][k].items()})


if __name__ == "__main__":
    main()
