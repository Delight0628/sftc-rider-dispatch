"""hybrid/replay_buffer.py — off-policy 经验回放（边级信用回填）

文档约定：docs/hybrid_implementation.md §4.1

Transition（每个决策 epoch 一条）：
  obs       : obs_pack_t（build_all_pairs_set_obs 输出 dict）
  edges     : List[(i, j, order_id)]      该 epoch 被匹配的边
  utility   : List[float]                 episode 末回填的实现效用（与 edges 对齐）
  obs_boots : List[obs_pack]              λ-return 阶梯各级 bootstrap obs（exp-c）
  boot_w    : List[float]                 阶梯 λ 质量权重（和为 1）
  boot_g    : List[float]                 各级 γ^gap（episode 已终止的级为 0）
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

import numpy as np


def pad_collate(obs_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """把变长 N 的 obs_pack 列表 pad 成 batch 张量（K 固定）。"""
    b = len(obs_list)
    n_max = max(o["rider_feat"].shape[0] for o in obs_list)
    k = obs_list[0]["cand_feat"].shape[1]

    rider_feat = np.zeros((b, n_max, obs_list[0]["rider_feat"].shape[1]), np.float32)
    rider_mask = np.zeros((b, n_max), np.float32)
    cand_feat = np.zeros((b, n_max, k, obs_list[0]["cand_feat"].shape[2]), np.float32)
    cand_mask = np.zeros((b, n_max, k), np.float32)
    edge_feat = np.zeros((b, n_max, k, obs_list[0]["edge_feat"].shape[2]), np.float32)
    global_feat = np.zeros((b, obs_list[0]["global_feat"].shape[0]), np.float32)

    for bi, o in enumerate(obs_list):
        n = o["rider_feat"].shape[0]
        rider_feat[bi, :n] = o["rider_feat"]
        rider_mask[bi, :n] = o["rider_mask"]
        cand_feat[bi, :n] = o["cand_feat"]
        cand_mask[bi, :n] = o["cand_mask"]
        edge_feat[bi, :n] = o["edge_feat"]
        global_feat[bi] = o["global_feat"]

    return {
        "rider_feat": rider_feat,
        "rider_mask": rider_mask,
        "cand_feat": cand_feat,
        "cand_mask": cand_mask,
        "edge_feat": edge_feat,
        "global_feat": global_feat,
    }


class ReplayBuffer:
    """定容 FIFO 回放；sample 返回 pad 后的 obs/batch 边索引/目标分量。"""

    def __init__(self, capacity: int = 10000, seed: int = 0):
        self.capacity = int(capacity)
        self._data: List[Dict[str, Any]] = []
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._data)

    def add_episode(self, transitions: List[Dict[str, Any]]) -> None:
        for t in transitions:
            self._data.append(t)
        if len(self._data) > self.capacity:
            self._data = self._data[-self.capacity:]

    def sample(self, batch_size: int, recent_frac: float = 0.0,
               recent_window: int = 0) -> Dict[str, Any]:
        """采样 batch，展开为逐边训练样本。

        recent_frac > 0 时为 recent 优先混合采样：recent_frac 比例从最近
        recent_window 条中均匀抽（无放回），其余从全 buffer 抽（有放回），
        抑制 off-policy staleness（旧行为策略样本混入过多 → q 目标噪声）。
        recent_frac=0 时退化为全 buffer 均匀无放回采样（原行为）。

        返回：
          obs       : pad_collate(obs_t)
          obs_boots : List[pad_collate]   λ-return 阶梯各级 bootstrap obs（exp-c）
          edge_batch/edge_i/edge_j : [E] 每条训练边的 (batch_idx, rider_i, slot_j)
          utility   : [E] 回填的实现效用
          boot_w    : [E,L] 阶梯 λ 质量权重   boot_g : [E,L] 各级 γ^gap
        """
        n = min(batch_size, len(self._data))
        if recent_frac > 0 and recent_window > 0 and len(self._data) > 1:
            n_recent = min(int(round(n * recent_frac)), n)
            lo = max(0, len(self._data) - int(recent_window))
            idx = self._rng.sample(range(lo, len(self._data)),
                                   k=min(n_recent, len(self._data) - lo))
            if len(idx) < n:
                idx += self._rng.choices(range(len(self._data)),
                                         k=n - len(idx))
            self._rng.shuffle(idx)
        else:
            idx = self._rng.sample(range(len(self._data)), k=n)
        L = len(self._data[idx[0]]["boot_w"])
        obs_list = []
        boot_lists: List[List[Any]] = [[] for _ in range(L)]
        e_b, e_i, e_j, e_u, e_w, e_g = [], [], [], [], [], []
        for bi, di in enumerate(idx):
            tr = self._data[di]
            obs_list.append(tr["obs"])
            for k in range(L):
                boot_lists[k].append(tr["obs_boots"][k])
            for (i, j, _oid), u in zip(tr["edges"], tr["utility"]):
                e_b.append(bi)
                e_i.append(i)
                e_j.append(j)
                e_u.append(u)
                e_w.append(tr["boot_w"])
                e_g.append(tr["boot_g"])
        return {
            "obs": pad_collate(obs_list),
            "obs_boots": [pad_collate(bl) for bl in boot_lists],
            "edge_batch": np.asarray(e_b, np.int32),
            "edge_i": np.asarray(e_i, np.int32),
            "edge_j": np.asarray(e_j, np.int32),
            "utility": np.asarray(e_u, np.float32),
            "boot_w": np.asarray(e_w, np.float32),
            "boot_g": np.asarray(e_g, np.float32),
        }
