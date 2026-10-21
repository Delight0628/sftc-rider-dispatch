"""sim_stream — DeliveryEnv 仿真帧流（Live 大屏数据源）

跑一个真实数据 episode，逐决策帧捕获：
  骑手位置/执行链、订单池、候选边（紧迫 q + 双塔偏好 w + 融合分）、
  派单结果、事件 ticker、KPI、Gantt——供大屏渲染与回放。

融合契约（project.md §3）：
  本仓 hybrid 输出紧迫维度边效用 q(i,j)；双塔（王新春侧）输出偏好分
  willingness ∈ [0,1]（契约 upstream_candidates: {骑手: [{order_id, willingness}]}）。
  外部加权融合：fused = w_pref * pref + w_urg * urg(q 归一化)，
  融合后每骑手订单 list 即大屏最终输出语义。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from environments.delivery_config import (HYBRID_DISPATCH_CONFIG,
                                          generate_mock_upstream_candidates)
from environments.delivery_env import DeliveryEnv
from environments.hybrid_dispatch import HybridDispatcher
from environments.real_data_loader import (load_real_training_pool,
                                           sample_order_window)

DEFAULT_CSV = r"d:\rider-dispatch-mappo\rl_hybrid_dispatch_data_20261008.csv"
DEFAULT_MODEL = (r"d:\rider-dispatch-mappo\runs\shanghai_hybrid_1017"
                 r"\models\checkpoints\edge_net_best.npz")
HIDDEN = 128
NUM_HEADS = 4


class _RecordingScorer:
    """包一层 NpEdgeValueNet：forward 时留存 q 矩阵供帧捕获。"""

    def __init__(self, path: str):
        from hybrid.edge_value_net import NpEdgeValueNet
        self.net = NpEdgeValueNet(path, hidden=HIDDEN, num_heads=NUM_HEADS)
        self.last_q: Optional[np.ndarray] = None

    def score_edges(self, obs_pack: Dict[str, Any]) -> np.ndarray:
        q = np.asarray(self.net.forward(obs_pack), dtype=np.float64)
        self.last_q = q
        return q


def _norm01(vals: List[float]) -> List[float]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def build_env_config(pool: List[Dict[str, Any]], riders: Dict[str, Any],
                     geo: Dict[str, Any], upstream: Dict[str, List[Dict[str, Any]]],
                     window_size: int = 40, seed: int = 20261020,
                     sim_time_cap: float = 480.0) -> Dict[str, Any]:
    """真实订单窗口 + 双塔候选注入的 episode 配置（resolve_env_config 口径）。"""
    from hybrid.collect import resolve_env_config
    cfg = {
        "scenario": "delivery",
        "training_mode": True,
        "real_order_pool": pool,
        "episode_order_size": window_size,
        "riders": riders,
        "geo_config": geo,
        "candidate_source": "upstream",          # 双塔精排候选驱动
        "upstream_order_by": "urgency",          # 会议口径：紧迫性定序
        "upstream_candidates": upstream,          # 王新春侧契约
        "deterministic_candidates": True,
        "MAX_SIM_STEPS": 400,
    }
    return resolve_env_config(cfg, seed=seed)


def capture_episode(csv_path: str = DEFAULT_CSV,
                    model_path: Optional[str] = DEFAULT_MODEL,
                    city: str = "上海市",
                    window_size: int = 40,
                    seed: int = 20261020,
                    max_steps: int = 400,
                    w_pref: float = 0.5,
                    progress_cb=None) -> Dict[str, Any]:
    """跑一个 episode，返回大屏帧包 {meta, frames, summary}。"""
    t0 = time.time()
    pool_cfg = load_real_training_pool(csv_path, rider_path=csv_path,
                                       max_orders=2000, max_riders=5, city=city)
    pool = pool_cfg.get("order_pool") or []
    riders = pool_cfg.get("riders") or {}
    geo = pool_cfg.get("geo_config") or {}
    names = list(riders.keys())

    # 双塔偏好源：mock 替身（契约与真实上游一致，替换点就在这里）
    window = sample_order_window(pool, window_size, seed=seed)
    upstream = generate_mock_upstream_candidates(window, names, top_k=10, seed=seed)
    pref_index = {nm: {int(e["order_id"]): float(e["willingness"])
                       for e in upstream.get(nm, [])} for nm in names}

    env_config = build_env_config(pool, riders, geo, upstream,
                                  window_size=window_size, seed=seed)
    env = DeliveryEnv(env_config)
    env.reset(seed=seed)
    sim = env.sim

    if model_path:
        try:
            batch_scorer = _RecordingScorer(model_path)
        except Exception:
            batch_scorer = None
    else:
        batch_scorer = None
    dispatcher = HybridDispatcher(env, batch_scorer=batch_scorer)
    rng = np.random.RandomState(seed)

    order_by_id = {o.order_id: o for o in sim.orders}
    frames: List[Dict[str, Any]] = []
    w_urg = 1.0 - w_pref

    def pref_of(nm: str, oid: int) -> Tuple[Optional[float], str]:
        """偏好分（契约口 _willingness_for：上游命中→双塔分；未覆盖→属性兜底）。"""
        hit = sim._upstream_index.get(nm, {}).get(int(oid))
        o = order_by_id.get(oid)
        if o is None:
            return None, "none"
        if hit is not None:
            return float(hit), "dual_tower"
        return float(sim._willingness_for(nm, o)), "fallback"

    def snapshot(matched: List[Tuple[int, int, int, float]],
                 trace_obs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        now = float(sim.current_time)
        # ---- 骑手 ----
        riders_snap = []
        for nm in names:
            r = sim.riders[nm]
            pos, free_t = r.projected_free()
            riders_snap.append({
                "name": nm,
                "x": round(float(r.position[0]), 3),
                "y": round(float(r.position[1]), 3),
                "load": len(r.carry),
                "cap": r.capacity,
                "offline": bool(r.offline),
                "delivered": int(r.delivered_count),
                "dist": round(float(r.travel_distance), 2),
                "carry_ids": [o.order_id for o in r.carry],
                "free_pos": [round(float(pos[0]), 3), round(float(pos[1]), 3)],
                "free_t": round(float(free_t), 2),
            })
        # ---- 订单（窗口内全部）----
        orders_snap = []
        for o in sim.orders:
            orders_snap.append({
                "id": int(o.order_id),
                "px": round(float(o.pickup[0]), 3), "py": round(float(o.pickup[1]), 3),
                "dx": round(float(o.dropoff[0]), 3), "dy": round(float(o.dropoff[1]), 3),
                "state": o.state,
                "ready": round(float(o.ready_time), 1),
                "due": round(float(o.due_date), 1),
                "slack": round(float(o.due_date - (o.planned_deliver_time or now)), 1),
                "rider": o.assigned_rider,
                "on_time": (o.actual_deliver_time is not None
                            and o.actual_deliver_time <= o.due_date),
            })
        # ---- 候选边（q x w 融合）----
        cands_snap = []
        if trace_obs is not None and batch_scorer is not None and batch_scorer.last_q is not None:
            q = batch_scorer.last_q
            oid_m = trace_obs["cand_order_ids"]
            act_m = trace_obs["cand_actions"]
            msk = trace_obs["cand_mask"]
            rnames = trace_obs["rider_names"]
            matched_set = {(m[2]) for m in matched}
            for i, nm in enumerate(rnames):
                row_q = [float(q[i, j]) for j in range(q.shape[1])
                         if oid_m[i, j] >= 0 and msk[i, j] > 0.5]
                row_norm = _norm01(row_q)
                k = 0
                for j in range(q.shape[1]):
                    oid = int(oid_m[i, j])
                    if oid < 0 or msk[i, j] < 0.5:
                        continue
                    o = order_by_id.get(oid)
                    if o is None:
                        continue
                    urg = row_norm[k]; k += 1
                    pref_v, pref_src = pref_of(nm, oid)
                    fused = (w_pref * pref_v + w_urg * urg) if pref_v is not None else urg
                    cands_snap.append({
                        "rider": nm, "oid": oid,
                        "q": round(float(q[i, j]), 4), "urg": round(urg, 4),
                        "pref": (round(pref_v, 4) if pref_v is not None else None),
                        "pref_src": pref_src,
                        "fused": round(float(fused), 4),
                        "slack": round(float(o.due_date - sim.current_time), 1),
                        "matched": oid in matched_set,
                    })
        # ---- 待派池队列（满载/未覆盖单兜底展示，rider=null 表示未指派）----
        covered = {c["oid"] for c in cands_snap}
        pend = sorted([x for x in sim.orders if x.state == "pool"
                       and x.order_id not in covered],
                      key=lambda x: x.due_date)
        for o in pend[:10]:
            urg_q = float(np.clip(1.0 - (o.due_date - sim.current_time) / 120.0, -1.0, 1.0))
            cands_snap.append({
                "rider": None, "oid": int(o.order_id),
                "q": None, "urg": round(urg_q, 4),
                "pref": None, "pref_src": "queue",
                "fused": round(urg_q, 4),
                "slack": round(float(o.due_date - sim.current_time), 1),
                "matched": False,
            })
        # ---- KPI（进行中）----
        delivered = len(sim.delivered)
        total = len(sim.orders)
        on_time = int(sim.stats.get("on_time_count", 0))
        tard = float(sim.stats.get("total_tardiness", 0.0))
        rider_utils = [(r.busy_time / max(now, 1e-6)) for r in sim.riders.values()]
        kpi = {
            "t": round(now, 2),
            "delivered": delivered, "total": total,
            "completion": round(delivered / max(total, 1), 4),
            "on_time_rate": round(on_time / max(delivered, 1), 4),
            "avg_tardiness": round(tard / max(delivered, 1), 2),
            "utilization": round(float(np.mean(rider_utils)) if rider_utils else 0.0, 4),
            "distance": round(float(sim.stats.get("total_distance", 0.0)), 1),
            "online_n": sum(1 for r in sim.riders.values() if not r.offline),
        }
        return {
            "t": round(now, 2),
            "riders": riders_snap,
            "orders": orders_snap,
            "cands": sorted(cands_snap, key=lambda c: (-c["fused"], c["rider"]))[:60],
            "kpi": kpi,
        }

    # ---- 主循环：决策帧 ----
    for step_i in range(int(max_steps)):
        actions, trace = dispatcher.act(temp=0.0, rng=rng, record=True)
        matched = trace.get("matched") or []
        obs = trace.get("obs")
        snap = snapshot(matched, obs)
        snap["step"] = step_i
        snap["matched"] = [{"rider": (obs or {}).get("rider_names", names)[m[0]],
                            "oid": int(m[2]), "dist": round(float(m[3]), 2)}
                           for m in matched]
        frames.append(snap)
        if progress_cb is not None:
            progress_cb(step_i, snap)
        _, _, terms, truncs, _ = env.step(actions)
        if all(terms.values()) or all(truncs.values()):
            break

    # ---- 事件 & Gantt & 汇总 ----
    fin = sim.get_final_stats()
    from environments.delivery_config import calculate_delivery_episode_score
    summary = {
        "episode_score": round(float(calculate_delivery_episode_score(fin, config=env.config)), 4),
        "completion_rate": round(float(fin.get("completion_rate", 0)), 4),
        "on_time_rate": round(float(fin.get("on_time_rate", 0)), 4),
        "avg_tardiness": round(float(fin.get("avg_tardiness", 0)), 2),
        "mean_utilization": round(float(fin.get("mean_utilization", 0)), 4),
        "distance_per_order": round(float(fin.get("distance_per_order", 0)), 2),
        "total_orders": int(fin.get("total_orders", 0)),
        "delivered": int(fin.get("total_parts", 0)),
    }
    return {
        "meta": {
            "csv": csv_path.split("\\")[-1], "city": city,
            "model": (model_path or "linear").split("\\")[-1],
            "window_size": window_size, "seed": seed,
            "w_pref": w_pref, "w_urg": round(w_urg, 4),
            "n_frames": len(frames),
            "elapsed_sec": round(time.time() - t0, 1),
            "dual_tower_source": "generate_mock_upstream_candidates（契约同真实上游）",
        },
        "frames": frames,
        "gantt": list(sim.gantt_chart_history),
        "events": list(sim.event_timeline),
        "summary": summary,
    }
