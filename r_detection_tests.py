"""R dantelio detektavimo testai (viena EKG derivacija).

Kiekvienam įrašui paleidžia kelis NeuroKit2 detektorius dviem variantais:
  std     - ecg_clean (metodui skirtas valymas) + ecg_peaks
  siaura  - detekcija ant siauros juostos (8-20 Hz) signalo, tada R vieta
            patikslinama originaliame (0.5-40 Hz) signale, ±50 ms lange

Jei tikros R vietos žinomos (sintetiniai įrašai) - lygina su jomis,
jei ne (tikri įrašai) - su metodų sutarimu (R, kurį rado >= pusė paleidimų).

Rezultatai:
  rezultatai/suvestine.csv              - matai kiekvienam įrašui ir metodui
  rezultatai/<įrašas>/<įrašas>_XXXs.png - 10 s langai, kuriuose kažkas suklydo

Paleidimas:
  python r_detection_tests.py --sintetiniai
  python r_detection_tests.py --failai duomenys/*.csv --fs 250
"""
import argparse
import glob
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

METHODS = ["neurokit", "pantompkins1985", "hamilton2002", "elgendi2010",
           "engzeemod2012", "christov2004", "kalidas2017", "martinez2004",
           "rodrigues2021", "promac"]
# metodai, kuriems ecg_clean turi savo valymo būdą
CLEAN_METHODS = {"neurokit", "pantompkins1985", "hamilton2002",
                 "elgendi2010", "engzeemod2012"}

TOL_S = 0.150        # R laikomas rastu, jei ±150 ms nuo tikro (ANSI/AAMI EC57)
REFINE_S = 0.050     # tikslinimo langas
NARROW_BAND = (8.0, 20.0)
WINDOW_S = 10.0

COL = {"signal": "#2a78d6", "true": "#0b0b0b", "tp": "#0ca30c",
       "fp": "#d03b3b", "fn": "#eda100", "ink": "#52514e", "grid": "#e1e0d9"}


# ---------- nuskaitymas ----------

def load_record(path):
    """Grąžina (signalas, fs arba None). Pakeisti, kai bus Zive nuskaitymo algoritmas."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(float).ravel(), None
    df = pd.read_csv(path, sep=None, engine="python")
    num = df.select_dtypes("number")
    return num.iloc[:, -1].to_numpy(float), None


# ---------- filtrai ir detekcija ----------

def bandpass(x, fs, lo, hi, order=2):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)  # nulinė fazė - R nepasislenka ant šlaito


def refine_peaks(peaks, ref, fs):
    """Perkelia kiekvieną R į ryškiausią ref signalo tašką ±REFINE_S lange.

    Poliarumas nustatomas visam įrašui (kad nepagautų gilaus S vietoj R)."""
    if len(peaks) == 0:
        return peaks
    w = int(REFINE_S * fs)
    win = [ref[max(p - w, 0):p + w + 1] for p in peaks]
    extremes = np.array([s[np.argmax(np.abs(s))] for s in win])
    sign = 1.0 if np.median(extremes) >= 0 else -1.0
    out = np.array([max(p - w, 0) + int(np.argmax(sign * s)) for p, s in zip(peaks, win)])
    # du aptikimai gali susitraukti į tą patį R - paliekam vieną
    out = np.unique(out)
    keep = [out[0]]
    for p in out[1:]:
        if p - keep[-1] < int(0.2 * fs):
            if sign * ref[p] > sign * ref[keep[-1]]:
                keep[-1] = p
        else:
            keep.append(p)
    return np.array(keep)


def detect(sig, fs, method, variant):
    if variant == "std":
        clean = nk.ecg_clean(sig, sampling_rate=fs,
                             method=method if method in CLEAN_METHODS else "neurokit")
        _, info = nk.ecg_peaks(clean, sampling_rate=fs, method=method)
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    narrow = bandpass(sig, fs, *NARROW_BAND)
    _, info = nk.ecg_peaks(narrow, sampling_rate=fs, method=method)
    ref = bandpass(sig, fs, 0.5, 40.0)
    return refine_peaks(np.asarray(info["ECG_R_Peaks"], dtype=int), ref, fs)


# ---------- vertinimas ----------

def match(true, det, fs):
    """Vienas su vienu, artimiausi pirmiau. Grąžina (TP porų sąrašas, FP idx, FN idx)."""
    tol = int(TOL_S * fs)
    pairs = []
    for i, d in enumerate(det):
        j = np.searchsorted(true, d)
        for jj in (j - 1, j):
            if 0 <= jj < len(true) and abs(true[jj] - d) <= tol:
                pairs.append((abs(true[jj] - d), jj, i))
    pairs.sort()
    used_t, used_d, tp = set(), set(), []
    for _, jj, i in pairs:
        if jj not in used_t and i not in used_d:
            used_t.add(jj); used_d.add(i); tp.append((jj, i))
    fp = [i for i in range(len(det)) if i not in used_d]
    fn = [j for j in range(len(true)) if j not in used_t]
    return tp, fp, fn


def consensus(all_peaks, fs, min_votes):
    pts = sorted((p, k) for k, peaks in enumerate(all_peaks) for p in peaks)
    out, cluster = [], []
    for p, k in pts + [(np.inf, -1)]:
        if cluster and p - cluster[0][0] > 0.075 * fs:
            if len({kk for _, kk in cluster}) >= min_votes:
                out.append(int(np.median([pp for pp, _ in cluster])))
            cluster = []
        cluster.append((p, k))
    return np.array(out, dtype=int)


# ---------- vaizdai ----------

def plot_window(name, sig, fs, t0, ref, ref_label, runs, out_path):
    a, b = int(t0 * fs), min(int((t0 + WINDOW_S) * fs), len(sig))
    t = np.arange(a, b) / fs
    labels = list(runs)
    variants = ["std", "siaura"]
    n = len(labels) // 2
    fig, axes = plt.subplots(3, 1, figsize=(15, 3 + 0.5 * n), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, n * 0.28, n * 0.28]})
    ax = axes[0]
    ax.plot(t, sig[a:b], color=COL["signal"], lw=1)
    r_in = ref[(ref >= a) & (ref < b)]
    ax.plot(r_in / fs, sig[r_in], "o", ms=5, mfc="none", mec=COL["true"], label=ref_label)
    ax.set_title(f"{name}   {t0:.0f}–{t0 + WINDOW_S:.0f} s", loc="left", color=COL["ink"])
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    for ax, var in zip(axes[1:], variants):
        rows = [k for k in labels if k.endswith("|" + var)]
        for y, key in enumerate(rows):
            det, tp, fp, fn = runs[key]
            tp_d = np.array([det[i] for _, i in tp], dtype=int)
            fp_d = np.array([det[i] for i in fp], dtype=int)
            fn_t = np.array([ref[j] for j in fn], dtype=int)
            for arr, mk, c, lab in ((tp_d, "|", COL["tp"], "rastas (TP)"),
                                    (fp_d, "x", COL["fp"], "klaidingas (FP)"),
                                    (fn_t, "v", COL["fn"], "praleistas (FN)")):
                sel = arr[(arr >= a) & (arr < b)]
                ax.plot(sel / fs, np.full(sel.size, y), mk, color=c, ms=8, mew=2,
                        label=lab if y == 0 else None)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([k.split("|")[0] for k in rows], fontsize=8)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.invert_yaxis()
        ax.set_ylabel("std" if var == "std" else "siaura\n+tikslin.", color=COL["ink"])
        ax.grid(axis="x", color=COL["grid"], lw=0.6)
    axes[1].legend(loc="upper right", bbox_to_anchor=(1, 1.35), ncol=3, frameon=False, fontsize=8)
    axes[-1].set_xlabel("laikas, s")
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------- pagrindinis ciklas ----------

def run_record(name, sig, fs, true_r, kinds, methods, out_dir, all_windows):
    runs, rows, errors = {}, [], {}
    for method in methods:
        for var in ("std", "siaura"):
            t0 = time.perf_counter()
            try:
                det = detect(sig, fs, method, var)
                errors[(method, var)] = None
            except Exception as e:  # kai kurie metodai lūžta ant keistų signalų
                det = np.array([], dtype=int)
                errors[(method, var)] = f"{type(e).__name__}: {e}"[:120]
            runs[f"{method}|{var}"] = [det, time.perf_counter() - t0]

    if true_r is None:
        ref = consensus([r[0] for r in runs.values()], fs, min_votes=len(runs) // 2)
        ref_label, ref_kinds = "sutarimas", None
    else:
        ref, ref_label, ref_kinds = np.asarray(true_r), "tikras R", kinds

    for key, (det, secs) in runs.items():
        tp, fp, fn = match(ref, det, fs)
        runs[key] = (det, tp, fp, fn)
        method, var = key.split("|")
        n_tp = len(tp)
        se = n_tp / len(ref) if len(ref) else np.nan
        ppv = n_tp / len(det) if len(det) else np.nan
        off = [abs(int(det[i]) - int(ref[j])) / fs * 1000 for j, i in tp]
        row = {"irasas": name, "metodas": method, "variantas": var,
               "etalonas": ref_label, "R_etalone": len(ref), "rasta": len(det),
               "TP": n_tp, "FP": len(fp), "FN": len(fn),
               "Se": round(se, 4), "PPV": round(ppv, 4),
               "F1": round(2 * se * ppv / (se + ppv), 4) if n_tp else 0.0,
               "vid_poslinkis_ms": round(float(np.mean(off)), 1) if off else np.nan,
               "laikas_s": round(secs, 2), "klaida": errors[(method, var)]}
        if ref_kinds is not None:
            for k in ("S", "V"):
                row[f"FN_{k}"] = int(sum(ref_kinds[j] == k for j in fn))
        rows.append(row)

    rec_dir = os.path.join(out_dir, name)
    os.makedirs(rec_dir, exist_ok=True)
    # vaizdai tik langams su FP/FN (poslinkis matomas lentelėje)
    for t0 in np.arange(0, len(sig) / fs - 1, WINDOW_S):
        a, b = int(t0 * fs), int((t0 + WINDOW_S) * fs)
        bad = all_windows
        for det, tp, fp, fn in runs.values():
            if bad:
                break
            bad = any(a <= det[i] < b for i in fp) or any(a <= ref[j] < b for j in fn)
        if bad:
            plot_window(name, sig, fs, t0, ref, ref_label, runs,
                        os.path.join(rec_dir, f"{name}_{int(t0):04d}s.png"))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sintetiniai", action="store_true", help="naudoti sintetinius įrašus")
    ap.add_argument("--failai", nargs="*", default=[], help="įrašų failai (.csv/.txt/.npy)")
    ap.add_argument("--fs", type=float, default=None, help="diskretizavimo dažnis, Hz")
    ap.add_argument("--metodai", nargs="*", default=METHODS)
    ap.add_argument("--visi-langai", action="store_true", help="saugoti ir langus be klaidų")
    ap.add_argument("--isvestis", default="rezultatai")
    args = ap.parse_args()

    records = []
    if args.sintetiniai:
        from synth_ecg import make_cases
        for name, (sig, fs, r, k) in make_cases().items():
            records.append((name, sig, fs, r, k))
    for pattern in args.failai:
        for path in sorted(glob.glob(pattern)):
            sig, fs = load_record(path)
            fs = fs or args.fs
            if fs is None:
                ap.error(f"{path}: nežinomas fs, nurodykite --fs")
            records.append((os.path.splitext(os.path.basename(path))[0], sig, fs, None, None))
    if not records:
        ap.error("nėra įrašų: --sintetiniai arba --failai")

    os.makedirs(args.isvestis, exist_ok=True)
    rows = []
    for name, sig, fs, r, k in records:
        print(f"{name} ({len(sig) / fs:.0f} s, {fs:g} Hz)")
        rows += run_record(name, sig, fs, r, k, args.metodai, args.isvestis, args.visi_langai)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.isvestis, "suvestine.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("\nF1 (std / siaura):")
    pv = df.pivot_table(index="irasas", columns=["variantas", "metodas"], values="F1")
    print(pv.round(3).to_string())
    print("\nVidurkis per įrašus:")
    print(df.groupby(["metodas", "variantas"])[["Se", "PPV", "F1", "vid_poslinkis_ms", "laikas_s"]]
          .mean().round(3).sort_values("F1", ascending=False).to_string())
    failed = df[df["klaida"].notna()]
    if len(failed):
        print("\nLūžę paleidimai:")
        print(failed[["irasas", "metodas", "variantas", "klaida"]].to_string(index=False))


if __name__ == "__main__":
    main()
