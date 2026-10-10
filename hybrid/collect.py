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
                    lam: float = 0.0,
                    hidden: int = 64,
                    num_heads: int = 4,
                    max_steps: int = 800,
                    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """跑一个 episode，返回 (transitions, final_stats)。

    transitions 结构见 replay_buffer.py 模块注释。行为策略：
    - temp=0：贪心（全部合法边参与匹配）；
    - temp>0：每骑手每轮按 softmax(q/temp) 采样一条候选边后贪心匹配。
    - lam>0：TD(λ) 截断混合 bootstrap（exp-c）；lam=0 退化为单阶 n-step。
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
                                    gamma=gamma, done=done, lam=lam)
    return transitions, stats


def build_transitions(traces: List[Dict[str, Any]], sim,
                      n_step: int = 5, gamma: float = 0.99,
                      done: bool = True, lam: float = 0.0) -> List[Dict[str, Any]]:
    """traces → transitions：回填实现效用 + λ-return 混合 bootstrap（exp-c）。

    效用公式复用 EdgeScorer.realized_utility（口径唯一，delivery_config 配置）：
      履约：  1 - w_tard·max(0,迟到)/60 - w_dist·距离/20
      未履约：-1

    bootstrap 阶梯 ladder = [1,2,4,...,n_step]，λ 质量分配（截断几何聚合）：
      w_k = λ^{n_{k-1}} − λ^{n_k}（k<L），w_L = λ^{n_{L-1}}（尾部质量归最后阶）
      λ→0 退化单阶（1-step），λ→1 退化纯 n-step。目标：
      y = u + Σ_k w_k·γ^{gap_k}·V(s_{t+n_k})，episode 终止的阶 gap 项记 0。
    """
    order_by_id = {o.order_id: o for o in sim.orders}

    T = len(traces)
    # ---- λ-return 阶梯与质量分配 ----
    if lam and lam > 0:
        ladder = [n for n in (1, 2, 4, 8, 16, 32, 64) if n <= n_step]
        if not ladder or ladder[-1] != n_step:
            ladder.append(int(n_step))
        ws: List[float] = []
        prev = 0
        for nn in ladder[:-1]:
            ws.append(float(lam ** prev - lam ** nn))
            prev = nn
        ws.append(float(lam ** prev))
    else:
        ladder = [int(n_step)]
        ws = [1.0]

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
        utilities[t][e_pos] = EdgeScorer.realized_utility(order, dist)

    transitions: List[Dict[str, Any]] = []
    for t, tr in enumerate(traces):
        if not tr["matched"]:
            continue  # 无匹配 epoch 不产生训练边
        obs_boots: List[Any] = []
        boot_g: List[float] = []
        for nn in ladder:
            t_boot = min(t + nn, T - 1)
            obs_boots.append(traces[t_boot]["obs"])
            if t + nn < T or not done:
                boot_g.append(float(gamma ** (t_boot - t)))
            else:
                boot_g.append(0.0)  # episode 已终止，无 bootstrap
        edges = [(int(i), int(j), int(oid)) for (i, j, oid, _d) in tr["matched"]]
        transitions.append({
            "obs": tr["obs"],
            "edges": edges,
            "utility": utilities[t],
            "obs_boots": obs_boots,
            "boot_w": list(ws),
            "boot_g": boot_g,
        })
    return transitions


# ---- 并行采集 worker（顶层函数，可 pickle；worker 内不依赖 TF）----
def collect_episode_worker(payload: Dict[str, Any]):
    """ProcessPoolExecutor 入口；payload 键同 collect_episode 参数。"""
    return collect_episode(**payload)
