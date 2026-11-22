from __future__ import annotations

import numpy as np


def match_peaks(ref: np.ndarray, det: np.ndarray, tol: int) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(ref, dtype=int)
    det = np.asarray(det, dtype=int)
    if ref.size == 0 or det.size == 0:
        return np.array([], int), np.array([], int)
    pairs = []
    j0 = 0
    for i, r in enumerate(ref):
        while j0 < det.size and det[j0] < r - tol:
            j0 += 1
        j = j0
        while j < det.size and det[j] <= r + tol:
            pairs.append((abs(det[j] - r), i, j))
            j += 1
    pairs.sort()
    used_r, used_d, mr, md = set(), set(), [], []
    for _, i, j in pairs:
        if i in used_r or j in used_d:
            continue
        used_r.add(i)
        used_d.add(j)
        mr.append(i)
        md.append(j)
    order = np.argsort(mr)
    return np.asarray(mr, int)[order], np.asarray(md, int)[order]


def detection_metrics(ref: np.ndarray, det: np.ndarray, fs: float, tol_ms: float = 75.0) -> dict:
    tol = int(round(tol_ms / 1000 * fs))
    mr, md = match_peaks(ref, det, tol)
    tp = mr.size
    fn = len(ref) - tp
    fp = len(det) - tp
    se = tp / (tp + fn) if tp + fn else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * se * ppv / (se + ppv) if (se + ppv) and not np.isnan(se + ppv) else np.nan
    signed_ms = (np.asarray(det)[md] - np.asarray(ref)[mr]) * 1000 / fs if tp else np.array([])
    err_ms = np.abs(signed_ms)
    return {
        "offset_median_ms": float(np.median(signed_ms)) if tp else np.nan,
        "TP": tp, "FN": fn, "FP": fp,
        "Se": se, "PPV": ppv, "F1": f1,
        "loc_err_median_ms": float(np.median(err_ms)) if tp else np.nan,
        "loc_err_p95_ms": float(np.percentile(err_ms, 95)) if tp else np.nan,
        "within_25ms": float(np.mean(err_ms <= 25)) if tp else np.nan,
        "within_50ms": float(np.mean(err_ms <= 50)) if tp else np.nan,
    }


def consensus_peaks(results: dict[str, np.ndarray], fs: float, tol_ms: float = 75.0,
                    min_votes: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    clusters, votes = all_clusters(results, fs, tol_ms)
    n = len(results)
    if min_votes is None:
        min_votes = n // 2 + 1
    keep = votes >= min_votes
    return clusters[keep], votes[keep]


def all_clusters(results: dict[str, np.ndarray], fs: float, tol_ms: float = 75.0
                 ) -> tuple[np.ndarray, np.ndarray]:
    tol = tol_ms / 1000 * fs
    pts = []
    for m, p in results.items():
        pts.extend((int(x), m) for x in p)
    if not pts:
        return np.array([], int), np.array([], int)
    pts.sort()
    clusters, votes = [], []
    cur = [pts[0]]
    for x, m in pts[1:]:
        if x - cur[-1][0] <= tol and m not in {mm for _, mm in cur}:
            cur.append((x, m))
        else:
            clusters.append(int(np.median([c[0] for c in cur])))
            votes.append(len(cur))
            cur = [(x, m)]
    clusters.append(int(np.median([c[0] for c in cur])))
    votes.append(len(cur))
    return np.asarray(clusters), np.asarray(votes)


def rr_flags(peaks: np.ndarray, fs: float, min_rr_ms: float = 250.0, gap_factor: float = 1.8) -> dict:
    if len(peaks) < 3:
        return {"n_peaks": len(peaks), "mean_hr_bpm": np.nan, "rr_too_short": 0, "rr_gaps": 0,
                "short_idx": np.array([], int), "gap_idx": np.array([], int)}
    rr = np.diff(peaks) * 1000 / fs
    k = 9
    pad = np.pad(rr, k // 2, mode="edge")
    local_med = np.array([np.median(pad[i:i + k]) for i in range(len(rr))])
    short = np.where(rr < min_rr_ms)[0]
    gaps = np.where(rr > gap_factor * local_med)[0]
    return {
        "n_peaks": len(peaks),
        "mean_hr_bpm": float(60000 / np.median(rr)),
        "rr_too_short": int(short.size),
        "rr_gaps": int(gaps.size),
        "short_idx": peaks[short + 1],
        "gap_idx": peaks[gaps],
    }


def parse_intervals(raw) -> list[tuple[int, int]]:
    out = []
    if not raw:
        return out
    for item in raw:
        try:
            if isinstance(item, dict):
                s = item.get("start", item.get("from", item.get("begin")))
                e = item.get("end", item.get("to", item.get("stop")))
            else:
                s, e = item[0], item[1]
            out.append((int(s), int(e)))
        except Exception:
            continue
    return out


def in_intervals(peaks: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(len(peaks), bool)
    for s, e in intervals:
        mask |= (peaks >= s) & (peaks <= e)
    return mask
