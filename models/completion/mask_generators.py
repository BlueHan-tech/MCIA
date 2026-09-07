"""
掩码生成器集合 (Mask Generators)

【通道分组】(NinaPro DB2/DB3, 12通道 sEMG)
  屈肌组 (Flexor):     Ch1-4, Ch9  -> idx [0,1,2,3,8]  (5通道)
  伸肌组 (Extensor):   Ch5-8, Ch10 -> idx [4,5,6,7,9]  (5通道)
  大臂组 (Upper Arm):  Ch11, Ch12  -> idx [10,11]       (2通道)

本模块提供两类策略：
  1) GroupWiseMaskGenerator + AdaptiveCurriculumScheduler  [LEGACY]
     5 级难度的线性课程学习，用于消融对照 / 老 sanity 脚本。
  2) ScenarioMixMaskGenerator  [RECOMMENDED]
     4 场景退化启发混合分布 (S1/S2/S3/S4)，全程 patch-aligned 时间块，
     每组存活通道硬下限保证信息论可解。
"""
from typing import Dict, List, Optional

import numpy as np
import torch


class AdaptiveCurriculumScheduler:
    """5级课程调度器"""

    def __init__(self, total_epochs=100, warmup_epochs=20):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

    def get_difficulty(self, epoch):
        if epoch < self.warmup_epochs:
            return 0.2 * (epoch / self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return 0.2 + 0.8 * progress

    def get_phase_name(self, epoch):
        difficulty = self.get_difficulty(epoch)
        return self.get_status_description(difficulty)

    def get_status_description(self, difficulty):
        if difficulty < 0.2:
            return f"Lv1: 入门 (Basics)      | D={difficulty:.2f}"
        elif difficulty < 0.4:
            return f"Lv2: 进阶 (Intermediate)| D={difficulty:.2f}"
        elif difficulty < 0.6:
            return f"Lv3: 混合 (Mixed High)  | D={difficulty:.2f}"
        elif difficulty < 0.8:
            return f"Lv4: 强化 (Advanced)    | D={difficulty:.2f}"
        else:
            return f"Lv5: 极限 (Survival Mixed)| D={difficulty:.2f}"


class GroupWiseMaskGenerator:
    """
    5级难度分组掩码生成器

    通道分组（索引从0开始）：
    - 屈肌组 (Flexor): Ch1-4, Ch9 -> idx [0,1,2,3,8]
    - 伸肌组 (Extensor): Ch5-8, Ch10 -> idx [4,5,6,7,9]
    - 大臂组 (Upper Arm): Ch11-12 -> idx [10,11]
    """
    GROUP_FLEXOR = [0, 1, 2, 3, 8]
    GROUP_EXTENSOR = [4, 5, 6, 7, 9]
    GROUP_UPPER_ARM = [10, 11]
    
    def __init__(self, n_channels=12, time_steps=128):
        self.n_channels = n_channels
        self.time_steps = time_steps

    def _apply_block_mask(self, mask_row, difficulty):
        """时间轴掩码（沿时间挖坑）"""
        T = mask_row.shape[0]
        num_blocks = int(3 + 5 * difficulty)
        min_len = int(10 + 20 * difficulty)
        max_len = int(30 + 50 * difficulty)
        
        for _ in range(num_blocks):
            length = np.random.randint(min_len, max_len + 1)
            if T - length > 0:
                start = np.random.randint(0, T - length)
                mask_row[start : start + length] = 0.0
        return mask_row

    def _get_level_params(self, difficulty):
        """
        根据难度返回各组命中数量和空间掩码概率

        数量增长: Flex/Ext 1→3, Upper 0→2
        概率增长: 5%→80% 线性阶梯
        """
        n_hits_flex = int(1.2 + 2.0 * difficulty)
        n_hits_ext = int(1.2 + 2.0 * difficulty)
        n_hits_upper = int(0.0 + 2.0 * difficulty)

        n_hits_flex = max(1, min(n_hits_flex, len(self.GROUP_FLEXOR)))
        n_hits_ext = max(1, min(n_hits_ext, len(self.GROUP_EXTENSOR)))
        n_hits_upper = max(0, min(n_hits_upper, len(self.GROUP_UPPER_ARM)))

        if difficulty < 0.2:
            spatial_prob = 0.05
        elif difficulty < 0.4:
            spatial_prob = 0.20
        elif difficulty < 0.6:
            spatial_prob = 0.40
        elif difficulty < 0.8:
            spatial_prob = 0.60
        else:
            spatial_prob = 0.80

        return n_hits_flex, n_hits_ext, n_hits_upper, spatial_prob

    def generate_mask(self, emg_batch, difficulty):
        if emg_batch.dim() == 3:
            if emg_batch.shape[2] == self.n_channels:
                B, T, C = emg_batch.shape
                dims_btc = True
            else:
                B, C, T = emg_batch.shape
                dims_btc = False
        else:
            raise ValueError("Input must be 3D tensor")
            
        device = emg_batch.device
        mask = torch.ones((B, C, T), device=device)

        n_flex, n_ext, n_upper, kill_prob = self._get_level_params(difficulty)

        for b in range(B):
            victims_flex = np.random.choice(self.GROUP_FLEXOR, n_flex, replace=False)
            victims_ext = np.random.choice(self.GROUP_EXTENSOR, n_ext, replace=False)
            
            if n_upper > 0:
                victims_upper = np.random.choice(self.GROUP_UPPER_ARM, n_upper, replace=False)
                all_victims = np.concatenate([victims_flex, victims_ext, victims_upper])
            else:
                all_victims = np.concatenate([victims_flex, victims_ext])
            
            for ch in all_victims:
                if np.random.rand() < kill_prob:
                    mask[b, ch, :] = 0.0
                else:
                    mask[b, ch, :] = self._apply_block_mask(mask[b, ch, :], difficulty)

        if dims_btc:
            mask = mask.permute(0, 2, 1)
        return mask

    def generate_batch_masks(self, batch_size, n_channels=12, time_steps=128,
                            device='cpu', difficulty_level=0.5):
        dummy = torch.zeros((batch_size, n_channels, time_steps), device=device)
        mask = self.generate_mask(dummy, difficulty_level)
        return mask


# ============================================================================
# ScenarioMixMaskGenerator  [推荐]
# ============================================================================

_DEFAULT_GROUP_INDICES: Dict[str, List[int]] = {
    'flexor':    [0, 1, 2, 3, 8],
    'extensor':  [4, 5, 6, 7, 9],
    'upper_arm': [10, 11],
}

_DEFAULT_MIN_ALIVE: Dict[str, int] = {
    'flexor': 3,
    'extensor': 3,
    'upper_arm': 1,
}

_DEFAULT_WEIGHTS: Dict[str, float] = {
    's1_transient':   0.20,
    's2_one_channel': 0.40,
    's3_two_channels': 0.40,
}

_DEFAULT_PARAMS: Dict[str, Dict] = {
    's1': {'n_blocks': [1, 2], 'block_len_patches': [2, 4]},
    's2': {'n_blocks': [1, 2], 'block_len_patches': [2, 4]},
    's3': {'n_blocks': [1, 2], 'block_len_patches': [2, 4]},
}

_SCENARIO_ALIAS = {
    's1_transient': 's1',
    's2_one_channel': 's2',
    's3_two_channels': 's3',
}


class ScenarioMixMaskGenerator:
    """4-场景退化启发混合掩码生成器

    ─ Scenarios ──────────────────────────────────────────────────────────────
      S1 scattered       : 每通道独立 Bernoulli 整通道 kill（组内保留硬约束）
                           + 存活通道 patch 对齐时间块
      S2 adjacent_cluster: flex/ext 组内按 Ch 编号挑 2–3 个相邻通道整条 kill
                           + 其余通道弱时间块
      S3 cross_group     : 恰好 1 flex + 1 ext + 1 upper 整条 kill
                           + 其余通道中等时间块
      S4 heavy_time      : 不杀整通道，每通道独立 patch 对齐时间块

    ─ 契约保证（防数据泄露） ──────────────────────────────────────────────────
      [C1] 输出 mask ∈ {0.0, 1.0} 严格布尔浮点，无中间值
      [C2] 时间块起点 ∈ {0, P, 2P, ...}，长度 ∈ {k*P | k ≥ 2}
           → sample-level 与 patch-level mask ratio 严格相等
           → 下游 mcia_core.derive_patch_time_mask 的 min-pool 派生零信息损失
      [C3] 每组存活通道数 ≥ min_alive_per_group[group]
           → 每个样本都存在"可解"的跨通道推断路径
    """

    def __init__(
        self,
        n_channels: int = 12,
        time_steps: int = 256,
        patch_size: int = 8,
        group_indices: Optional[Dict[str, List[int]]] = None,
        min_alive_per_group: Optional[Dict[str, int]] = None,
        scenario_weights: Optional[Dict[str, float]] = None,
        scenario_params: Optional[Dict[str, Dict]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        assert time_steps % patch_size == 0, \
            f"time_steps ({time_steps}) must be divisible by patch_size ({patch_size})"
        self.n_channels = n_channels
        self.time_steps = time_steps
        self.patch_size = patch_size
        self.num_patches = time_steps // patch_size

        self.group_indices = dict(group_indices) if group_indices else dict(_DEFAULT_GROUP_INDICES)
        self.min_alive = dict(min_alive_per_group) if min_alive_per_group else dict(_DEFAULT_MIN_ALIVE)
        self.weights = dict(scenario_weights) if scenario_weights else dict(_DEFAULT_WEIGHTS)
        # 允许用户只覆盖部分参数
        self.params = {k: dict(v) for k, v in _DEFAULT_PARAMS.items()}
        if scenario_params:
            for k, v in scenario_params.items():
                if k in self.params:
                    self.params[k].update(v)
                else:
                    self.params[k] = dict(v)
        self.scenario_weights = self.weights
        self.scenario_params = self.params

        self.rng = rng if rng is not None else np.random.default_rng()

    # ---------- full-channel victim sampling ----------

    def _sample_victim_channels(self, count: int) -> set[int]:
        if count not in (1, 2):
            raise ValueError(f"Expected one or two full-channel victims, got {count}")
        candidates = np.arange(self.n_channels)
        for _ in range(256):
            victims = set(int(value) for value in self.rng.choice(candidates, size=count, replace=False))
            valid = True
            for gname, indices in self.group_indices.items():
                minimum = int(self.min_alive.get(gname, 0))
                dead = sum(channel in victims for channel in indices)
                if len(indices) - dead < minimum:
                    valid = False
                    break
            if valid:
                return victims
        raise RuntimeError("Unable to sample full-channel victims under min_alive constraints")

    # ---------- patch 对齐时间块 ----------

    def _sample_patch_aligned_blocks(self, n_blocks_range, block_len_patches_range) -> np.ndarray:
        """采样 patch 对齐时间块掩码 shape=(T,), 0=缺失, 1=可见。"""
        T, P, N = self.time_steps, self.patch_size, self.num_patches
        n_blocks = int(self.rng.integers(n_blocks_range[0], n_blocks_range[1] + 1))
        mask = np.ones(T, dtype=np.float32)
        for _ in range(n_blocks):
            len_p = int(self.rng.integers(block_len_patches_range[0],
                                          block_len_patches_range[1] + 1))
            len_p = min(len_p, N)
            max_start = N - len_p
            if max_start < 0:
                continue
            start_p = int(self.rng.integers(0, max_start + 1))
            s = start_p * P
            e = s + len_p * P
            mask[s:e] = 0.0
        return mask

    # ---------- 场景采样 ----------

    def _sample_scenario_id(self) -> str:
        names = list(self.weights.keys())
        ws = np.array([self.weights[n] for n in names], dtype=np.float64)
        total = ws.sum()
        if total <= 0:
            raise ValueError(f"scenario_weights sum to {total}, must be positive")
        ws = ws / total
        idx = int(self.rng.choice(len(names), p=ws))
        return _SCENARIO_ALIAS.get(names[idx], names[idx])

    # ---------- 各场景 ----------

    def _apply_temporal_blocks(self, mask_ct: np.ndarray, params: Dict,
                               killed: set[int]) -> np.ndarray:
        for channel in range(self.n_channels):
            if channel in killed:
                mask_ct[channel, :] = 0.0
            else:
                mask_ct[channel, :] *= self._sample_patch_aligned_blocks(
                    params['n_blocks'], params['block_len_patches'])
        return mask_ct

    def _apply_s1(self, mask_ct: np.ndarray, params: Dict) -> np.ndarray:
        return self._apply_temporal_blocks(mask_ct, params, set())

    def _apply_s2(self, mask_ct: np.ndarray, params: Dict) -> np.ndarray:
        return self._apply_temporal_blocks(mask_ct, params, self._sample_victim_channels(1))

    def _apply_s3(self, mask_ct: np.ndarray, params: Dict) -> np.ndarray:
        return self._apply_temporal_blocks(mask_ct, params, self._sample_victim_channels(2))

    def _dispatch(self, mask_ct: np.ndarray, scenario: str) -> np.ndarray:
        scn = _SCENARIO_ALIAS.get(scenario, scenario)
        if scn == 's1':
            return self._apply_s1(mask_ct, self.params['s1'])
        if scn == 's2':
            return self._apply_s2(mask_ct, self.params['s2'])
        if scn == 's3':
            return self._apply_s3(mask_ct, self.params['s3'])
        raise ValueError(f"Unknown scenario: {scenario}")

    def generate_batch_masks(
        self,
        batch_size: int,
        n_channels: Optional[int] = None,
        time_steps: Optional[int] = None,
        device: str = 'cpu',
        scenario: Optional[str] = None,
        difficulty_level: Optional[float] = None,
    ) -> torch.Tensor:
        del difficulty_level
        channels = n_channels if n_channels is not None else self.n_channels
        steps = time_steps if time_steps is not None else self.time_steps
        assert channels == self.n_channels, f"n_channels mismatch: {channels} vs {self.n_channels}"
        assert steps == self.time_steps, f"time_steps mismatch: {steps} vs {self.time_steps}"
        mask = np.ones((batch_size, channels, steps), dtype=np.float32)
        for index in range(batch_size):
            selected = scenario if scenario is not None else self._sample_scenario_id()
            self._dispatch(mask[index], selected)
        return torch.from_numpy(mask).to(device)

    def generate_mask(
        self,
        emg_batch: torch.Tensor,
        difficulty: Optional[float] = None,
        scenario: Optional[str] = None,
    ) -> torch.Tensor:
        del difficulty
        if emg_batch.dim() != 3:
            raise ValueError("Input must be 3D tensor")
        if emg_batch.shape[2] == self.n_channels:
            batch_size, steps, channels = emg_batch.shape
            btc = True
        else:
            batch_size, channels, steps = emg_batch.shape
            btc = False
        mask = self.generate_batch_masks(
            batch_size, channels, steps, device=str(emg_batch.device), scenario=scenario
        )
        return mask.permute(0, 2, 1) if btc else mask


def _run_self_check(n_trials: int = 100, batch_size: int = 32, seed: int = 0) -> None:
    generator = ScenarioMixMaskGenerator(
        n_channels=12, time_steps=256, patch_size=8, rng=np.random.default_rng(seed)
    )
    for scenario, expected_dead in (('s1', 0), ('s2', 1), ('s3', 2)):
        mask = generator.generate_batch_masks(batch_size * n_trials, scenario=scenario)
        alive = mask.amax(dim=-1) > 0.5
        dead = (alive == 0).sum(dim=1)
        assert int(dead.min()) == expected_dead == int(dead.max())
        for group, indices in generator.group_indices.items():
            assert int(alive[:, indices].sum(dim=1).min()) >= generator.min_alive[group]
        print(f"{scenario}: exact_full_channel_loss={expected_dead}")
    counts = {'s1': 0, 's2': 0, 's3': 0}
    for _ in range(10000):
        counts[generator._sample_scenario_id()] += 1
    print(f"sampling={counts}")
    print("All three-scenario contracts satisfied")


if __name__ == '__main__':
    _run_self_check()
