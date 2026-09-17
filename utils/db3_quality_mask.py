"""DB3 双层质量掩码：硬掉线与 WC-BQD 的并集（2026-09-10 采用）。

第一层（硬零）：若某通道在训练 repetition（1/3/4）的全部动作采样点上严格为零，
则在整段记录中丢弃该通道。

第二层（WC-BQD，通道内基线质量检测器）：每个“通道-秒”只与该通道自身在训练
repetition 上的基线比较，并按污染类型自底向上判定。它取代了 Grönlund（2005）
MQP 的跨通道离群层；后者会把正常的通道间幅值差异（主动肌激活、安静的上臂
电极）误判为接触不良：健康 DB2 的误报下限为 14.5%（p>0.20），逐通道标记率
与通道 RMS 的 Pearson 相关系数为 0.95。

检测项（常量在 2026-09-09/10 验证期间预先声明）：
  T1 平台：该秒 P99.5 幅值极端处出现不少于 10 个相同采样点（裁剪/饱和），或
           出现不少于 100 个相同采样点（50 ms 平直掉线）；幅值门可豁免无害的
           量化重复值。
  T2 低能量：相对该通道训练 repetition 分布，log-RMS 的稳健 z 值小于 -3.5
           （电极脱落/失活片段）。
  T3 突发：稳健 z 值大于 5.0，且该秒至少 80% 的功率低于 20 Hz（运动伪迹频带；
           真实肌肉收缩为宽带信号）。

文献依据：Farago 等（2023，IEEE Rev Biomed Eng）的自底向上策略与污染类型；
Hodges 和 Bui（1996，Electroencephalogr Clin Neurophysiol 101:511-519）及 2023 年
起始检测综述（J Neuroeng Rehabil 20）的通道内基线阈值；De Luca 等（2010，
J Biomech 43:157-163）的低于 20 Hz 运动伪迹频带；Leys 等（2013，J Exp Soc
Psychol 49:764-766）的中位数/MAD 稳健 z 值与 3.5 规则。

验证摘要（run 06_diagnostics/wcbqd_validation_20260909）：健康留出集误报下限为
0.15%/0.09%（S01-S10/S11-S20）；对激活水平的跟踪相关与 0 无显著差别；DB3
留出集标记率为 4.82%（MQP 参照为 18.58%）；逐被试硬零一致性为 100%
（S06/S07 的 Ch9/Ch10）；下游 S05 开发 A/B 中，B 相对 A 的 RMSE 差距由
+4.15% 缩小至 +0.30%。
"""
from __future__ import annotations

import numpy as np

# ---- 预先声明的 WC-BQD 常量（见模块说明）----
KAPPA_LOW = 3.5
KAPPA_HIGH = 5.0
LOW_FREQ_HZ = 20.0
POWER_FRAC = 0.80
PLATEAU_RUN = 10          # 短平台仅在幅值极端处计入
PLATEAU_HARD = 100        # 不论幅值如何，50 ms 平直段均计入
PLATEAU_DEV_PCTL = 99.5   # 单秒内的幅值极端门限
TRAIN_REPS = (1, 3, 4)
LOG_EPS = 1e-12
MAD_FLOOR = 1e-3
MIN_TRAIN_SECONDS = 4


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _plateau_stats(sec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐通道返回最长相同值连续段的采样点数及其取值。

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
    """返回逐秒、逐通道特征以及无需基线的 T1 判定。"""
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
    """返回 200 Hz 保留掩码和可审计的双层掩码元数据。

    硬掉线通道仅根据训练 repetition 1/3/4 拟合。WC-BQD 基线同样只由训练
    repetition 的动作秒拟合，并在每个动作秒上独立评估。若某被试的训练数据少于
    MIN_TRAIN_SECONDS 秒，则跳过 T2/T3（仍应用无需基线的 T1），并在元数据中记录
    回退原因。
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

    seconds = []          # （游标、是否训练数据、特征）
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
        fallback = (f"仅有 {len(logrms_train)} 个训练秒；"
                    "已跳过 T2/T3，仅使用无需基线的 T1")

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
