"""hybrid/evaluate.py — 对拍评估（neural / linear / 启发式基线）

文档约定：docs/hybrid_implementation.md §6
- summary 键名与 evaluation_delivery.py 一致：
  completion_rate / on_time_rate / avg_tardiness / makespan /
  mean_utilization / distance_per_order / episode_score
- 有效性铁律：neural ≥ linear > nearest/edd/fifo（同订单集、同 seed、同 episode 数）
- 评估路径 100% numpy（NpEdgeValueNet），不要求评估机装 TF。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from environments.delivery_config import calculate_delivery_episode_score
from environments.delivery_env import DeliveryEnv
from environments.hybrid_dispatch import HybridDispatcher
from hybrid.collect import _make_batch_scorer
from hybrid.scorers import LinearEdgeScorer


def summarize_episodes(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """把 get_final_stats() 行聚合成 summary（键名对齐 evaluation_delivery.py）。"""

    def _mean(key: str) -> float:
        vals = [float(r.get(key, 0) or 0) for r in rows]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "completion_rate": _mean("completion_rate"),
        "on_time_rate": _mean("on_time_rate"),
        "avg_tardiness": _mean("avg_tardiness"),
        "total_tardiness": _mean("total_tardiness"),
        "makespan": _mean("makespan"),
        "mean_utilization": _mean("mean_utilization"),
        "distance_per_order": _mean("distance_per_order"),
        "episode_score": _mean("episode_score"),
        "total_orders": _mean("total_orders"),
        "total_parts": _mean("total_parts"),
    }


def _heuristic_actions(env: DeliveryEnv, baseline: str,
                       rng: np.random.RandomState) -> Dict[str, list]:
    """逐骑手独立启发式（沿用 evaluation_delivery._pick_by_baseline 口径）。"""
    actions: Dict[str, list] = {}
    for agent in env.agents:
        info = env.infos.get(agent, {})
        cands = info.get("candidates_map") or []
        mask = info.get("action_mask")
        legal = []
        for c in cands:
            a = int(c["action"])
            if mask is not None and a < len(mask) and mask[a] < 0.5:
                continue
            legal.append(c)
        if not legal:
            actions[agent] = [0]
            continue
        if baseline == "random":
            pick = legal[int(rng.randint(0, len(legal)))]
        elif baseline == "fifo":
            pick = sorted(legal, key=lambda c: (c.get("ready_time", 0.0), c.get("part_id", 0)))[0]
        elif baseline == "nearest":
            pick = sorted(legal, key=lambda c: (c.get("to_pickup_time", 1e9), c.get("part_id", 0)))[0]
        elif baseline == "edd":
            pick = sorted(legal, key=lambda c: (c.get("slack", 1e9), c.get("part_id", 0)))[0]
        elif baseline == "idle":
            pick = legal[0]
        else:
            raise ValueError(f"unknown baseline: {baseline}")
        actions[agent] = [int(pick["action"])]
    return actions


def evaluate_episode(env_config: Dict[str, Any], scorer_kind: str,
                     scorer_artifact: Optional[str] = None, seed: int = 0,
                     hidden: int = 64, num_heads: int = 4,
                     max_steps: int = 800) -> Dict[str, Any]:
    """单 episode 贪心评估（temp=0，无探索）。

    scorer_kind: linear | neural(npz 路径) | 启发式基线名
    （nearest / edd / fifo / random / idle，逐骑手独立选择）。
    """
    env = DeliveryEnv(env_config)
    env.reset(seed=seed)

    heuristics = ("nearest", "edd", "fifo", "random", "idle")
    dispatcher = None
    rng = np.random.RandomState(seed)
    if scorer_kind not in heuristics:
        # neural 评估时 artifact 为 npz 路径，_make_batch_scorer 认 "npz"
        kind = "npz" if scorer_kind == "neural" else scorer_kind
        batch_scorer = _make_batch_scorer(kind, scorer_artifact,
                                          hidden=hidden, num_heads=num_heads)
        dispatcher = HybridDispatcher(env, batch_scorer=batch_scorer)

    steps = 0
    for _ in range(max_steps):
        if dispatcher is not None:
            actions = dispatcher.act(temp=0.0, rng=rng)
        else:
            actions = _heuristic_actions(env, scorer_kind, rng)
        _, _, terms, truncs, _ = env.step(actions)
        steps += 1
        if all(terms.values()) or all(truncs.values()):
            break

    stats = env.sim.get_final_stats()
    stats["steps"] = steps
    stats["episode_score"] = float(
        calculate_delivery_episode_score(stats, config=env.config))
    return stats


def evaluate_all(env_config: Dict[str, Any],
                 neural_artifact: Optional[str] = None,
                 episodes: int = 3, seed: int = 0,
                 hidden: int = 64, num_heads: int = 4,
                 baselines: Tuple[str, ...] = ("edd", "nearest", "fifo"),
                 include_linear: bool = True) -> Dict[str, Dict[str, float]]:
    """neural vs linear vs 启发式的同 seed 对拍。

    返回 {scorer_name: summary}。scorer 集合：neural（给了 artifact 时）、
    linear（include_linear）、baselines。所有 episode 用相同 seed 序列（seed+ep），
    保证同一订单流上对拍（铁律前提）。
    """
    scorers: List[Tuple[str, str, Optional[str]]] = []
    if neural_artifact is not None:
        scorers.append(("neural", "neural", neural_artifact))
    if include_linear:
        scorers.append(("linear", "linear", None))
    for b in baselines:
        scorers.append((b, b, None))

    out: Dict[str, Dict[str, float]] = {}
    for name, kind, artifact in scorers:
        rows = [
            evaluate_episode(env_config, kind, artifact, seed=seed + ep,
                             hidden=hidden, num_heads=num_heads)
            for ep in range(episodes)
        ]
        out[name] = summarize_episodes(rows)
    return out
