"""Sintetiniai vienos derivacijos EKG įrašai su žinomomis R vietomis.

Kiekvienas atvejis atkartoja vieną R detektavimo klaidos tipą
(aukštas T, S/V ekstrasistolės, artefaktai, žema QRS amplitudė, judesys,
apverstas poliarumas). Naudojama, kol nėra tikrų įrašų.

Dūžis modeliuojamas Gauso bangų suma (P, Q, R, S, T) laike apie R viršūnę.
"""
import numpy as np

# (centras s, amplitudė mV, plotis s)
NORMAL = {"P": (-0.20, 0.12, 0.025), "Q": (-0.030, -0.12, 0.008),
          "R": (0.0, 1.00, 0.010), "S": (0.030, -0.25, 0.008),
          "T": (0.28, 0.30, 0.050)}
# V: platus QRS, be P, priešingos krypties T
PVC = {"Q": (-0.05, -0.10, 0.015), "R": (0.0, 1.30, 0.028),
       "S": (0.07, -0.45, 0.025), "T": (0.33, -0.45, 0.070)}


def _add_beat(x, t, r_time, waves, scale=1.0):
    for c, a, w in waves.values():
        x += scale * a * np.exp(-((t - (r_time + c)) ** 2) / (2 * w ** 2))


def _make(duration, fs, rng, hr=70.0, waves=NORMAL, p_svpb=0.0, p_pvc=0.0,
          r_scale=1.0):
    t = np.arange(int(duration * fs)) / fs
    x = np.zeros_like(t)
    rr0 = 60.0 / hr
    r_times, kinds = [], []
    tr = 0.6
    while tr < duration - 0.6:
        u = rng.random()
        if u < p_svpb:
            kind, tr = "S", r_times[-1] + 0.62 * rr0 if r_times else tr
        elif u < p_svpb + p_pvc:
            kind, tr = "V", r_times[-1] + 0.60 * rr0 if r_times else tr
        else:
            kind = "N"
        if tr >= duration - 0.6:
            break
        if kind == "V":
            _add_beat(x, t, tr, PVC, r_scale)
        else:
            w = dict(waves)
            if kind == "S":  # ankstyvas dūžis, P kitokia forma
                w["P"] = (-0.16, -0.08, 0.02)
            _add_beat(x, t, tr, w, r_scale)
        r_times.append(tr)
        kinds.append(kind)
        # po ekstrasistolės kompensacinė pauzė, kitaip įprastas RR su HRV
        tr += (1.35 if kind == "V" else 1.0) * rr0 * (1 + 0.04 * rng.standard_normal())
    r_idx = np.round(np.array(r_times) * fs).astype(int)
    return t, x, r_idx, np.array(kinds)


def _wander(t, rng, amp=0.15):
    return amp * (np.sin(2 * np.pi * 0.25 * t + rng.random() * 6)
                  + 0.5 * np.sin(2 * np.pi * 0.07 * t + rng.random() * 6))


def make_cases(duration=60.0, fs=250, seed=0):
    """Grąžina dict: vardas -> (signalas, fs, tikros R indeksai, dūžių tipai)."""
    rng = np.random.default_rng(seed)
    cases = {}

    t, x, r, k = _make(duration, fs, rng)
    cases["01_normalus"] = (x + 0.01 * rng.standard_normal(x.size), r, k)

    tall_t = dict(NORMAL, T=(0.26, 1.15, 0.035))
    t, x, r, k = _make(duration, fs, rng, waves=tall_t)
    cases["02_aukstas_T"] = (x + _wander(t, rng, 0.05) + 0.015 * rng.standard_normal(x.size), r, k)

    t, x, r, k = _make(duration, fs, rng, p_svpb=0.15)
    cases["03_S_ekstrasistoles"] = (x + 0.015 * rng.standard_normal(x.size), r, k)

    t, x, r, k = _make(duration, fs, rng, p_pvc=0.15)
    cases["04_V_ekstrasistoles"] = (x + _wander(t, rng, 0.08) + 0.015 * rng.standard_normal(x.size), r, k)

    t, x, r, k = _make(duration, fs, rng)
    spikes = np.zeros_like(x)
    for i in rng.choice(x.size - 10, size=int(duration / 2), replace=False):
        spikes[i:i + 2] += rng.choice([-1, 1]) * rng.uniform(0.8, 1.6)
    cases["05_impulsiukai"] = (x + spikes + 0.015 * rng.standard_normal(x.size), r, k)

    t, x, r, k = _make(duration, fs, rng, r_scale=0.35)
    emg = 0.06 * rng.standard_normal(x.size)
    cases["06_zema_QRS_triuksmas"] = (x + emg + _wander(t, rng, 0.2), r, k)

    t, x, r, k = _make(duration, fs, rng)
    motion = np.zeros_like(x)
    for start in rng.uniform(2, duration - 6, size=4):
        a, b = int(start * fs), int((start + rng.uniform(1.5, 3.5)) * fs)
        n = b - a
        burst = (0.8 * np.sin(2 * np.pi * rng.uniform(1, 3) * np.arange(n) / fs)
                 + 0.35 * rng.standard_normal(n))
        motion[a:b] += burst * np.hanning(n)
    cases["07_judesio_artefaktai"] = (x + motion + _wander(t, rng, 0.1) + 0.015 * rng.standard_normal(x.size), r, k)

    t, x, r, k = _make(duration, fs, rng, p_pvc=0.05)
    cases["08_apverstas"] = (-x + 0.015 * rng.standard_normal(x.size), r, k)

    return {name: (sig, fs, r, k) for name, (sig, r, k) in cases.items()}
