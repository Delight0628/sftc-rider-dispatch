"""hybrid/collect.py — episode 采集（行为策略 + 边级信用回填 + n-step 链接）

文档约定：docs/hybrid_implementation.md §4.1/§4.2/§4.3
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from environments.delivery_env import DeliveryEnv
from environments.hybrid_dispatch import EdgeScorer, HybridDispatcher


def _make_batch_scorer(kind: str, artifact: Optional[str],
                       hidden: int, num_heads: int):
    """行为策略打分器：linear（手调先验）或 npz 路径的神经 numpy 前向。

    采集 worker 不依赖 TF：neural 走 NpEdgeValueNet。
    """
    if kind == "linear":
        from hybrid.scorers import LinearEdgeScorer
        return LinearEdgeScorer()
    from hybrid.edge_value_net import NpEdgeValueNet

    class _NpBatchScorer:
        def __init__(self, path):
            self.net = NpEdgeValueNet(path, hidden=hidden, num_heads=num_heads)

        def score_edges(self, obs_pack):
            return self.net.forward(obs_pack)

    return _NpBatchScorer(artifact)


def collect_episode(env_config: Dict[str, Any],
                    scorer_kind: str = "linear",
                    scorer_artifact: Optional[str] = None,
                    temp: float = 0.0,
                    seed: int = 0,
                    n_step: int = 5,
                    gamma: float = 0.99,
                    hidden: int = 64,
                    num_heads: int = 4,
                    max_steps: int = 800,
                    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """跑一个 episode，返回 (transitions, final_stats)。

    transitions 结构见 replay_buffer.py 模块注释。行为策略：
    - temp=0：贪心（全部合法边参与匹配）；
    - temp>0：每骑手每轮按 softmax(q/temp) 采样一条候选边后贪心匹配。
    """
    env = DeliveryEnv(env_config)
    env.reset(seed=seed)
    batch_scorer = _make_batch_scorer(scorer_kind, scorer_artifact, hidden, num_heads)
    dispatcher = HybridDispatcher(env, batch_scorer=batch_scorer)
    rng = np.random.RandomState(seed)

    traces: List[Dict[str, Any]] = []
    steps = 0
    for _ in range(max_steps):
        actions, trace = dispatcher.act(temp=temp, rng=rng, record=True)
        _, _, terms, truncs, _ = env.step(actions)
        traces.append(trace)
        steps += 1
        if all(terms.values()) or all(truncs.values()):
            break
    done = all(terms.values())

    stats = env.sim.get_final_stats()
    transitions = build_transitions(traces, env.sim, n_step=n_step,
                                    gamma=gamma, done=done)
    return transitions, stats


def build_transitions(traces: List[Dict[str, Any]], sim,
                      n_step: int = 5, gamma: float = 0.99,
                      done: bool = True) -> List[Dict[str, Any]]:
    """traces → transitions：回填实现效用 + n-step bootstrap 链接。

    效用公式复用 EdgeScorer.realized_utility（口径唯一，delivery_config 配置）：
      取餐/送达两段 shaping（exp-a）：
        w_pick·(1 - 取餐等待/pick_norm) + (1-w_pick)·(1 - w_tard·迟到/60 - w_dist·距离/20)
      未履约：已取餐按取餐段得分 + 送达段 -1；未取餐 -1
    """
    order_by_id = {o.order_id: o for o in sim.orders}
    sim_end = float(getattr(sim, "_max_sim_time", 0.0) or 0.0) or None

    T = len(traces)
    # order_id → (t, edge_pos_in_trace) 索引；同单只记首次派入
    assign_at: Dict[int, Tuple[int, int]] = {}
    for t, tr in enumerate(traces):
        for e_pos, (_i, _j, oid, _dist) in enumerate(tr["matched"]):
            assign_at.setdefault(oid, (t, e_pos))

    # 每条的 utility 先初始化 0，再按订单回填
    utilities: List[List[float]] = [
        [0.0] * len(tr["matched"]) for tr in traces
    ]
    for oid, (t, e_pos) in assign_at.items():
        order = order_by_id.get(oid)
        if order is None:
            continue
        dist = traces[t]["matched"][e_pos][3]
        utilities[t][e_pos] = EdgeScorer.realized_utility(order, dist, sim_end=sim_end)

    transitions: List[Dict[str, Any]] = []
    for t, tr in enumerate(traces):
        if not tr["matched"]:
            continue  # 无匹配 epoch 不产生训练边
        t_boot = min(t + n_step, T - 1)
        boot = float(gamma ** n_step) if (t + n_step < T or not done) else 0.0
        edges = [(int(i), int(j), int(oid)) for (i, j, oid, _d) in tr["matched"]]
        transitions.append({
            "obs": tr["obs"],
            "edges": edges,
            "utility": utilities[t],
            "obs_boot": traces[t_boot]["obs"],
            "boot": boot,
        })
    return transitions


# ---- 并行采集 worker（顶层函数，可 pickle；worker 内不依赖 TF）----
def collect_episode_worker(payload: Dict[str, Any]):
    """ProcessPoolExecutor 入口；payload 键同 collect_episode 参数。"""
    return collect_episode(**payload)
