"""
方案 B：集合观测导出
====================
从 DeliverySim 抽出结构化 set 观测，供 PPONetworkSet / 无 TF 单测使用。

输出 dict（每个 agent 一份）：
  rider_feat  [N, D_r]
  rider_mask  [N]
  cand_feat   [N, K, D_c]
  cand_mask   [N, K]
  global_feat [D_g]
  self_index  int   # 本 agent 在骑手轴上的下标
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

# 特征维度（与 plan_scheme_b §3.3 对齐）
RIDER_FEAT_DIM = 8
CAND_FEAT_DIM = 12
GLOBAL_FEAT_DIM = 5


def _compressed(x: float, norm: float) -> float:
    if norm <= 0:
        norm = 1.0
    v = max(0.0, float(x)) / float(norm)
    return v / (1.0 + v)


def build_set_obs(sim, agent_id: str) -> Dict[str, Any]:
    """构造单 agent 的 set 观测。sim 为 DeliverySim 实例。"""
    rider_name = agent_id.replace("agent_", "")
    if rider_name not in sim.riders:
        raise KeyError(f"unknown agent {agent_id}")
    now = float(sim.current_time)
    names = list(sim.rider_names)
    self_idx = names.index(rider_name)
    n = len(names)
    k_max = int(sim._obs_cfg.get("num_candidate_orders", 10))
    slack_norm = float(sim._obs_cfg.get("slack_time_norm", 120.0))
    route_norm = float(sim._obs_cfg.get("total_remaining_time_norm", 120.0))
    op_norm = float(sim._obs_cfg.get("max_op_duration_norm", 60.0))
    grid = float(sim._geo.get("grid_size", 20.0))

    # ---- 骑手 set ----
    rider_feat = np.zeros((n, RIDER_FEAT_DIM), dtype=np.float32)
    rider_mask = np.zeros((n,), dtype=np.float32)
    for i, name in enumerate(names):
        r = sim.riders[name]
        offline = sim._rider_is_offline(r, now)
        pos, free_t = r.projected_free()
        free_delay = max(0.0, (free_t - now) if free_t > 0 else 0.0)
        rider_feat[i] = [
            len(r.carry) / max(1, r.capacity),
            r.busy_time / max(1.0, now),
            1.0 if offline else 0.0,
            float(r.speed) / max(1e-6, float(sim._geo.get("speed_norm", 0.6))),
            float(r.capacity) / 6.0,
            _compressed(free_delay, 60.0),
            float(pos[0]) / grid,
            float(pos[1]) / grid,
        ]
        rider_mask[i] = 0.0 if offline else 1.0

    # ---- 本骑手候选 set ----
    cand_feat = np.zeros((n, k_max, CAND_FEAT_DIM), dtype=np.float32)
    cand_mask = np.zeros((n, k_max), dtype=np.float32)
    # 全局：候选仅针对 self 骑手（分布式执行）；其余骑手候选置 0 但 mask=0
    me = sim.riders[rider_name]
    cands = sim._get_candidate_orders(rider_name)
    pos, free_t = me.projected_free()
    start = max(now, free_t if free_t > 0 else now)
    for c in cands:
        j = int(c["index"])
        if j >= k_max:
            continue
        o = c["order"]
        est = sim._estimate_route(me, o, pos, start)
        slack = float(o.due_date - est["deliver_time"])
        due_rel = float(o.due_date - now)
        # 到达序：池内按 ready_time 排名（0=最早）
        rank = 0.0
        if sim.pool:
            order_sorted = sorted(sim.pool, key=lambda x: (x.ready_time, x.order_id))
            for ri, oo in enumerate(order_sorted):
                if oo is o:
                    rank = ri / max(1, len(order_sorted))
                    break
        # 偏好/willingness 不入模（原表无骑手×订单意愿字段；偏好由并行双塔侧处理）
        cand_feat[self_idx, j] = [
            1.0,
            float(np.clip(due_rel / route_norm, -1.0, 3.0)),  # [1] due 相对（紧迫）
            _compressed(est["leg1_time"], op_norm),
            _compressed(est["total_time"], route_norm),
            sim._pickup_congestion(o),
            (float(o.priority) - 1.0) / 2.0,
            1.0 if o.priority == 1 else 0.0,
            float(getattr(o, "type_id", 0)) / 4.0,
            float(np.clip(1.0 - slack / slack_norm, 0.0, 1.0)),
            float(np.clip(slack / slack_norm, -3.0, 3.0)),
            float(np.clip(due_rel / route_norm, -1.0, 3.0)),
            rank,
        ]
        cand_mask[self_idx, j] = 1.0

    # ---- 全局 ----
    total = max(1, len(sim.orders))
    busy = sum(1 for r in sim.riders.values() if len(r.carry) >= r.capacity)
    global_feat = np.asarray([
        min(1.0, now / max(1.0, sim._simulation_time)),
        len(sim.pool) / total,
        busy / max(1, n),
        _compressed(len(sim.pool), 20.0),
        n / 12.0,
    ], dtype=np.float32)

    return {
        "rider_feat": rider_feat,
        "rider_mask": rider_mask,
        "cand_feat": cand_feat,
        "cand_mask": cand_mask,
        "global_feat": global_feat,
        "self_index": self_idx,
        "agent_id": agent_id,
    }


def build_global_set_state(sim) -> Dict[str, Any]:
    """集中式 Critic 用：全体骑手 + 池摘要 + 全局。"""
    names = list(sim.rider_names)
    n = len(names)
    now = float(sim.current_time)
    grid = float(sim._geo.get("grid_size", 20.0))
    rider_feat = np.zeros((n, RIDER_FEAT_DIM), dtype=np.float32)
    rider_mask = np.zeros((n,), dtype=np.float32)
    for i, name in enumerate(names):
        r = sim.riders[name]
        offline = sim._rider_is_offline(r, now)
        pos, free_t = r.projected_free()
        free_delay = max(0.0, (free_t - now) if free_t > 0 else 0.0)
        rider_feat[i] = [
            len(r.carry) / max(1, r.capacity),
            r.busy_time / max(1.0, now),
            1.0 if offline else 0.0,
            float(r.speed) / max(1e-6, float(sim._geo.get("speed_norm", 0.6))),
            float(r.capacity) / 6.0,
            _compressed(free_delay, 60.0),
            float(pos[0]) / grid,
            float(pos[1]) / grid,
        ]
        rider_mask[i] = 0.0 if offline else 1.0

    # 池 set（截断到 32，防爆）
    k_pool = 32
    pool_feat = np.zeros((k_pool, 6), dtype=np.float32)
    pool_mask = np.zeros((k_pool,), dtype=np.float32)
    slack_norm = float(sim._obs_cfg.get("slack_time_norm", 120.0))
    for j, o in enumerate(sim.pool[:k_pool]):
        slack = float(o.due_date - now)
        pool_feat[j] = [
            _compressed(o.weight, 10.0),
            (float(o.priority) - 1.0) / 2.0,
            float(np.clip(slack / slack_norm, -3.0, 3.0)),
            float(np.clip(1.0 - slack / slack_norm, 0.0, 1.0)),
            float(o.due_date - now) / 120.0,
            j / max(1, min(len(sim.pool), k_pool)),
        ]
        pool_mask[j] = 1.0

    total = max(1, len(sim.orders))
    global_feat = np.asarray([
        min(1.0, now / max(1.0, sim._simulation_time)),
        len(sim.delivered) / total,
        len(sim.pool) / total,
        _compressed(len(sim.pool), 20.0),
        n / 12.0,
    ], dtype=np.float32)

    return {
        "rider_feat": rider_feat,
        "rider_mask": rider_mask,
        "pool_feat": pool_feat,
        "pool_mask": pool_mask,
        "global_feat": global_feat,
    }


# =============================================================================
# 集中式打分观测（hybrid 主干）：全对候选矩阵
# 约定见 docs/hybrid_implementation.md §2
# =============================================================================
EDGE_FEAT_DIM = 16  # 12 基础 + exp-b 4 项（拥堵ETA/送达簇/竞争/疲劳）


def build_all_pairs_set_obs(sim) -> Dict[str, Any]:
    """集中式边打分的全对候选观测（hybrid 主干）。

    与 build_set_obs 的差异：cand 轴对**每个骑手**都填充（非仅 self 行），
    并额外给出 cand_order_ids / cand_actions / edge_feat，供匹配→动作映射
    与经验回放对齐（docs/hybrid_implementation.md §2.2/§2.3）。

    返回 dict：
      rider_feat [N,8]  rider_mask [N]  global_feat [5]
      cand_feat [N,K,12]  cand_mask [N,K]（1=该槽有效且动作合法，对齐 action_mask）
      edge_feat [N,K,12]（与 EdgeScorer.build_features 逐维一致）
      cand_order_ids [N,K] int32（-1=空槽）  cand_actions [N,K] int32（0=无）
      rider_names List[str]
    """
    from environments.hybrid_dispatch import EdgeScorer  # 局部导入，避免模块耦合

    names = list(sim.rider_names)
    n = len(names)
    now = float(sim.current_time)
    grid = float(sim._geo.get("grid_size", 20.0))
    k_max = int(sim._obs_cfg.get("num_candidate_orders", 10))
    slack_norm = float(sim._obs_cfg.get("slack_time_norm", 120.0))
    route_norm = float(sim._obs_cfg.get("total_remaining_time_norm", 120.0))
    op_norm = float(sim._obs_cfg.get("max_op_duration_norm", 60.0))

    rider_feat = np.zeros((n, RIDER_FEAT_DIM), dtype=np.float32)
    rider_mask = np.zeros((n,), dtype=np.float32)
    cand_feat = np.zeros((n, k_max, CAND_FEAT_DIM), dtype=np.float32)
    cand_mask = np.zeros((n, k_max), dtype=np.float32)
    edge_feat = np.zeros((n, k_max, EDGE_FEAT_DIM), dtype=np.float32)
    cand_order_ids = np.full((n, k_max), -1, dtype=np.int64)
    cand_actions = np.zeros((n, k_max), dtype=np.int32)

    # 池内 ready 排名（rank 特征）
    rank_of = {}
    if sim.pool:
        order_sorted = sorted(sim.pool, key=lambda x: (x.ready_time, x.order_id))
        for ri, oo in enumerate(order_sorted):
            rank_of[id(oo)] = ri / max(1, len(order_sorted))

    for i, name in enumerate(names):
        r = sim.riders[name]
        offline = sim._rider_is_offline(r, now)
        pos, free_t = r.projected_free()
        free_delay = max(0.0, (free_t - now) if free_t > 0 else 0.0)
        rider_feat[i] = [
            len(r.carry) / max(1, r.capacity),
            r.busy_time / max(1.0, now),
            1.0 if offline else 0.0,
            float(r.speed) / max(1e-6, float(sim._geo.get("speed_norm", 0.6))),
            float(r.capacity) / 6.0,
            _compressed(free_delay, 60.0),
            float(pos[0]) / grid,
            float(pos[1]) / grid,
        ]
        rider_mask[i] = 0.0 if offline else 1.0
        if offline:
            continue
        can_take = len(r.carry) < r.capacity
        start = max(now, free_t if free_t > 0 else now)
        for c in sim._get_candidate_orders(name):
            j = int(c["index"])
            if j >= k_max:
                continue
            o = c["order"]
            est = sim._estimate_route(r, o, pos, start)
            slack = float(o.due_date - est["deliver_time"])
            due_rel = float(o.due_date - now)
            cand_feat[i, j] = [
                1.0,
                float(np.clip(due_rel / route_norm, -1.0, 3.0)),
                _compressed(est["leg1_time"], op_norm),
                _compressed(est["total_time"], route_norm),
                sim._pickup_congestion(o),
                (float(o.priority) - 1.0) / 2.0,
                1.0 if o.priority == 1 else 0.0,
                float(getattr(o, "type_id", 0)) / 4.0,
                float(np.clip(1.0 - slack / slack_norm, 0.0, 1.0)),
                float(np.clip(slack / slack_norm, -3.0, 3.0)),
                float(np.clip(due_rel / route_norm, -1.0, 3.0)),
                rank_of.get(id(o), 0.0),
            ]
            cand_mask[i, j] = 1.0 if can_take else 0.0
            cand_order_ids[i, j] = int(o.order_id)
            cand_actions[i, j] = j + 1  # 动作号 = 候选 index + 1（delivery_env 约定）
            edge_feat[i, j] = EdgeScorer.build_features(sim, r, o, pos, start, now)

    total = max(1, len(sim.orders))
    busy = sum(1 for r in sim.riders.values() if len(r.carry) >= r.capacity)
    global_feat = np.asarray([
        min(1.0, now / max(1.0, sim._simulation_time)),
        len(sim.pool) / total,
        busy / max(1, n),
        _compressed(len(sim.pool), 20.0),
        n / 12.0,
    ], dtype=np.float32)

    return {
        "rider_feat": rider_feat,
        "rider_mask": rider_mask,
        "cand_feat": cand_feat,
        "cand_mask": cand_mask,
        "edge_feat": edge_feat,
        "cand_order_ids": cand_order_ids,
        "cand_actions": cand_actions,
        "global_feat": global_feat,
        "rider_names": names,
    }
