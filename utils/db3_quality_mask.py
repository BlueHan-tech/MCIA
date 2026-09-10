"""DB3 two-layer quality masks: hard dropouts plus WC-BQD (2026-09-10 adopted).

Layer 1 (hard zero): channels that are exactly zero across all train-repetition
(1/3/4) active samples are dropped for the whole recording.

Layer 2 (WC-BQD, within-channel baseline quality detector): each channel-second
is judged only against that channel's own train-repetition baseline (bottom-up,
per-contaminant).  It replaced the Grönlund (2005) MQP cross-channel outlier
layer, which confused normal inter-channel amplitude differences (agonist
activation, quiet upper-arm electrodes) with poor contact: healthy-DB2 floor
14.5% at p>0.20, per-channel flag rate vs channel RMS Pearson 0.95.

Tests (constants pre-declared during 2026-09-09/10 validation):
  T1 plateau  >=10 identical samples at the second's P99.5 amplitude extremes
               (clipping/saturation) OR >=100 identical samples (50 ms flatline
               dropout); amplitude gate exempts benign quantization duplicates.
  T2 low energy  log-RMS robust z < -3.5 vs the channel's own train-repetition
               distribution (electrode fell off / dead segment).
  T3 burst       robust z > 5.0 AND >=80% of the second's power below 20 Hz
               (motion-artifact band; genuine contraction is broadband).

Literature anchors: bottom-up strategy and contaminant taxonomy - Farago et al.
2023 (IEEE Rev Biomed Eng); within-channel baseline thresholds - Hodges & Bui
1996 (Electroencephalogr Clin Neurophysiol 101:511-519) and the 2023 onset
detection review (J Neuroeng Rehabil 20); <20 Hz motion-artifact band - De Luca
et al. 2010 (J Biomech 43:157-163); median/MAD robust z and the 3.5 rule -
Leys et al. 2013 (J Exp Soc Psychol 49:764-766).

Validation summary (run 06_diagnostics/wcbqd_validation_20260909): healthy
held-out floor 0.15%/0.09% (S01-S10/S11-S20), activation-tracking Pearson
indistinguishable from 0, DB3 held-out flag rate 4.82% (MQP reference 18.58%),
per-subject hard-zero consistency 100% (S06/S07 Ch9/Ch10), downstream S05 dev
A/B: B-vs-A RMSE gap +4.15% -> +0.30%.
"""
from __future__ import annotations

import numpy as np

# ---- pre-declared WC-BQD constants (see module docstring) ----
KAPPA_LOW = 3.5
KAPPA_HIGH = 5.0
LOW_FREQ_HZ = 20.0
POWER_FRAC = 0.80
PLATEAU_RUN = 10          # short plateau counts only at amplitude extremes
PLATEAU_HARD = 100        # 50 ms flatline counts regardless of amplitude
PLATEAU_DEV_PCTL = 99.5   # amplitude-extreme gate within the second
TRAIN_REPS = (1, 3, 4)
LOG_EPS = 1e-12
MAD_FLOOR = 1e-3
MIN_TRAIN_SECONDS = 4


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _plateau_stats(sec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per channel: longest identical-value run (in SAMPLES) and its value.

    L 个相同采样点产生 L-1 对相等标记；此处返回 ends-starts+1，使计数
    语义与文档一致（PLATEAU_HARD=100 即 100 个采样点 = 50 ms）。
    2026-09-10 修复：此前返回相等对数（差一），阈值实际多要一个采样点。
    """
    n, c = sec.shape
    runlen = np.zeros(c, dtype=np.int64)
    runval = np.zeros(c, dtype=np.float64)
    padded = np.zeros(n + 1, dtype=bool)
    for col in range(c):
        padded[1:-1] = sec[1:, col] == sec[:-1, col]
        d = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        if len(starts):
            k = int(np.argmax(ends - starts))
            runlen[col] = ends[k] - starts[k] + 1
            runval[col] = sec[starts[k], col]
    return runlen, runval


def second_features(sec: np.ndarray) -> dict:
    """Per-second per-channel features and the baseline-free T1 verdict."""
    x = sec - sec.mean(axis=0, keepdims=True)
    rms = np.sqrt((x ** 2).mean(axis=0))
    logrms = np.log(np.maximum(rms, LOG_EPS))
    runlen, runval = _plateau_stats(sec)
    sec_mean = sec.mean(axis=0, keepdims=True)
    dev_scale = np.percentile(np.abs(sec - sec_mean), PLATEAU_DEV_PCTL, axis=0)
    t1 = (runlen >= PLATEAU_HARD) | (
        (runlen >= PLATEAU_RUN) & (np.abs(runval - sec_mean[0]) >= dev_scale)
    )
    spec = np.abs(np.fft.rfft(x, axis=0)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / 2000.0)
    low = spec[freqs < LOW_FREQ_HZ].sum(axis=0)
    tot = spec.sum(axis=0)
    lowfrac = np.where(tot > 1e-24, low / np.maximum(tot, 1e-24), 1.0)
    return {"logrms": logrms, "t1": t1, "lowfrac": lowfrac}


def _fit_baselines(logrms_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(logrms_train, axis=0)
    mad = 1.4826 * np.median(np.abs(logrms_train - med), axis=0)
    return med, np.maximum(mad, MAD_FLOOR)


def _wcbqd_flags(feats: dict, med: np.ndarray, mad: np.ndarray) -> np.ndarray:
    z = (feats["logrms"] - med) / mad
    t2 = z < -KAPPA_LOW
    t3 = (z > KAPPA_HIGH) & (feats["lowfrac"] >= POWER_FRAC)
    return feats["t1"] | t2 | t3


def db3_quality_mask(raw: np.ndarray, labels: np.ndarray, repetitions: np.ndarray,
                     target_fs: int, action_ids: tuple[int, ...] = tuple(range(1, 49))) -> tuple[np.ndarray, dict]:
    """Return a 200-Hz keep mask and auditable two-layer mask metadata.

    Hard channels are fitted only from train repetitions 1/3/4.  WC-BQD
    baselines are likewise fitted from train-repetition active seconds and
    evaluated independently on every active second.  If a subject has fewer
    than MIN_TRAIN_SECONDS train seconds, T2/T3 are skipped (baseline-free T1
    still applies) and the metadata records the fallback.
    """
    raw = np.asarray(raw, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    repetitions = np.asarray(repetitions).reshape(-1)
    n = min(len(raw), len(labels), len(repetitions))
    raw, labels, repetitions = raw[:n], labels[:n], repetitions[:n]
    valid_action = np.isin(labels, action_ids)
    train = valid_action & np.isin(repetitions, TRAIN_REPS)
    hard = np.all(raw[train] == 0.0, axis=0) if train.any() else np.zeros(raw.shape[1], dtype=bool)
    raw_keep = np.ones_like(raw, dtype=bool)
    raw_keep[:, hard] = False
    centered = raw - raw.mean(axis=0, keepdims=True)

    seconds = []          # (cursor, is_train, features)
    for start, end in _runs(valid_action):
        for cursor in range(start, end - 2000 + 1, 2000):
            if np.all(repetitions[cursor:cursor + 2000] == repetitions[cursor]):
                is_train = int(repetitions[cursor]) in TRAIN_REPS
                seconds.append(
                    (cursor, is_train, second_features(centered[cursor:cursor + 2000]))
                )

    logrms_train = [f["logrms"] for _, is_train, f in seconds if is_train]
    baselines = None
    fallback = None
    if len(logrms_train) >= MIN_TRAIN_SECONDS:
        baselines = _fit_baselines(np.stack(logrms_train))
    else:
        fallback = (f"only {len(logrms_train)} train seconds; "
                    "T2/T3 skipped, baseline-free T1 only")

    t_counts = {"t1": 0, "t2_t3": 0}
    for cursor, _, feats in seconds:
        if baselines is not None:
            flagged = _wcbqd_flags(feats, baselines[0], baselines[1])
        else:
            flagged = feats["t1"]
        if flagged.any():
            raw_keep[cursor:cursor + 2000, flagged] = False
            t_counts["t1"] += int((feats["t1"] & flagged).sum())
            t_counts["t2_t3"] += int((~feats["t1"] & flagged).sum())

    factor = 2000 // int(target_fs)
    usable = (len(raw_keep) // factor) * factor
    down_keep = raw_keep[:usable].reshape(-1, factor, raw.shape[1]).all(axis=1)
    return down_keep.astype(np.float32), {
        "hard_channels_one_based": (np.flatnonzero(hard) + 1).astype(int).tolist(),
        "mask_rule": "hard_zero_train_1_3_4 OR wcbqd_v0.1_plateau_lowenergy_lowfreqburst",
        "wcbqd_constants": {
            "kappa_low": KAPPA_LOW, "kappa_high": KAPPA_HIGH,
            "low_freq_hz": LOW_FREQ_HZ, "power_frac": POWER_FRAC,
            "plateau_run": PLATEAU_RUN, "plateau_hard": PLATEAU_HARD,
            "plateau_dev_pctl": PLATEAU_DEV_PCTL,
        },
        "wcbqd_one_second_segments": len(seconds),
        "wcbqd_train_seconds": len(logrms_train),
        "wcbqd_baseline_fallback": fallback,
        "wcbqd_channel_second_alerts": t_counts["t1"] + t_counts["t2_t3"],
        "wcbqd_t1_alerts": t_counts["t1"],
        "wcbqd_t2_t3_alerts": t_counts["t2_t3"],
    }
