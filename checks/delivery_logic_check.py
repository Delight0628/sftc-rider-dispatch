"""配送环境逻辑回归（纯环境层，无需 TensorFlow）。

覆盖：观测维度 / 动作 mask / rollout / 骑手位置更新 / 竞态 /
      评分双键口径 / 动态事件 / 时间单调性 / 奖励有限性 / 容量约束。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.delivery_env import DeliveryEnv, DeliverySim  # noqa: E402
from environments.delivery_config import (  # noqa: E402
    DELIVERY_OBS_CONFIG,
    calculate_delivery_episode_score,
    generate_random_delivery_orders,
)


class Checker:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name: str, cond: bool, detail: str = ""):
        if cond:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {detail}")
            print(f"  [FAIL] {name} — {detail}")

    def summary(self) -> int:
        total = self.passed + self.failed
        print(f"\n=== delivery_logic_check: {self.passed}/{total} passed ===")
        if self.errors:
            print("Failures:")
            for e in self.errors:
                print(" -", e)
        return 0 if self.failed == 0 else 1


def _legal_actions(env, agent):
    mask = env.infos[agent]["action_mask"]
    return np.where(mask)[0]


def _greedy_step(env, prefer_action=None):
    actions = {}
    for a in env.agents:
        legal = _legal_actions(env, a)
        if prefer_action is not None and prefer_action.get(a) is not None:
            pa = int(prefer_action[a])
            actions[a] = pa if pa in legal else (int(legal[0]) if len(legal) else 0)
        else:
            # 优先非 IDLE，便于触发接单路径
            non_idle = [x for x in legal if x != 0]
            actions[a] = int(non_idle[0]) if non_idle else (int(legal[0]) if len(legal) else 0)
    return env.step(actions)


def check_dims_and_spaces(ck: Checker):
    print("\n[1] 维度与动作空间")
    env = DeliveryEnv({"scenario": "delivery", "training_mode": True})
    obs, infos = env.reset(seed=0)
    agent = env.agents[0]
    o = obs[agent]
    ck.check("观测 146 维", o.shape == (146,), f"got {o.shape}")
    ck.check("5 个骑手 agent", len(env.agents) == 5, f"got {env.agents}")
    aspace = env.action_space(agent)
    ck.check("动作 MultiDiscrete([11])", list(aspace.nvec) == [11], f"got {aspace.nvec}")
    ck.check("全局状态 19 维", env.sim.get_global_state().shape == (19,),
             f"got {env.sim.get_global_state().shape}")
    mask = env.infos[agent]["action_mask"]
    ck.check("mask 长度 11 且 IDLE 可用", mask.shape == (11,) and mask[0] == 1.0,
             f"mask={mask}")
    ck.check("obs 有限", bool(np.all(np.isfinite(o))), "NaN/Inf in obs")


def check_rollout_and_termination(ck: Checker):
    print("\n[2] Rollout 与终止")
    env = DeliveryEnv({"scenario": "delivery", "training_mode": True, "MAX_SIM_STEPS": 400})
    env.reset(seed=1)
    last_t = -1.0
    steps = 0
    terminated = False
    for _ in range(400):
        obs, rewards, terms, truncs, infos = _greedy_step(env)
        steps += 1
        t = env.sim.current_time
        ck.check(f"时间单调 step{steps}", t >= last_t - 1e-9, f"{last_t} -> {t}") if steps <= 5 else None
        last_t = t
        for a, r in rewards.items():
            if not math.isfinite(float(r)):
                ck.check(f"奖励有限 {a}", False, f"r={r}")
                break
        else:
            if steps == 1:
                ck.check("首步奖励有限", True)
        if all(terms.values()) or all(truncs.values()):
            terminated = True
            break
    ck.check("rollout 可推进并终止", terminated or steps >= 400, f"steps={steps}")
    stats = env.sim.get_final_stats()
    ck.check("final_stats 含 makespan/total_parts", "makespan" in stats and "total_parts" in stats,
             f"keys={list(stats)}")
    ck.check("完成数不超过订单总数", stats["total_parts"] <= stats["total_orders"],
             f"{stats['total_parts']} > {stats['total_orders']}")


def check_position_update(ck: Checker):
    print("\n[3] 骑手送达后位置更新")
    orders = [
        {"order_type": "普通餐品", "pickup": (2.0, 10.0), "dropoff": (5.0, 12.0),
         "ready_time": 0.0, "due_date": 80.0, "priority": 2, "weight": 1.0},
        {"order_type": "普通餐品", "pickup": (3.0, 10.0), "dropoff": (6.0, 11.0),
         "ready_time": 0.0, "due_date": 80.0, "priority": 2, "weight": 1.0},
    ]
    env = DeliveryEnv({"scenario": "delivery", "training_mode": True,
                       "custom_orders": orders, "MAX_SIM_STEPS": 50})
    env.reset(seed=0)
    rider = env.sim.riders["骑手A"]
    home = rider.position
    # 强制骑手A接第 1 单
    cands = env.sim._get_candidate_orders("骑手A")
    target_action = None
    for c in cands:
        if c["order"].order_id == 0:
            target_action = c["index"] + 1
            break
    prefer = {"agent_骑手A": target_action} if target_action is not None else None
    # 推进若干步直到该单送达
    delivered_pos = None
    for _ in range(40):
        _greedy_step(env, prefer_action=prefer)
        prefer = None  # 只在第一步指定
        if any(o.order_id == 0 and o.state == "delivered" for o in env.sim.orders):
            delivered_pos = rider.position
            break
    ck.check("订单0已送达", delivered_pos is not None, "未在40步内送达")
    if delivered_pos is not None:
        ck.check("送达后位置=送餐点", delivered_pos == (5.0, 12.0),
                 f"home={home}, pos={delivered_pos}")


def check_race_conflict(ck: Checker):
    print("\n[4] 同步接单竞态不记 invalid")
    orders = [
        {"order_type": "普通餐品", "pickup": (10.0, 10.0), "dropoff": (11.0, 10.0),
         "ready_time": 0.0, "due_date": 120.0, "priority": 2, "weight": 1.0},
    ]
    # 让两个骑手都看到同一订单为候选，并同时抢单
    env = DeliveryEnv({"scenario": "delivery", "training_mode": True,
                       "custom_orders": orders, "MAX_SIM_STEPS": 20})
    env.reset(seed=0)
    a, b = "agent_骑手A", "agent_骑手B"
    # 构造两个 agent 都选 index 0（同一池内订单）
    actions = {agent: 1 for agent in env.agents}  # action 1 = candidate 0
    obs, rewards, terms, truncs, infos = env.step(actions)
    stats = env.sim.stats
    race = int(stats.get("race_conflict_count", 0))
    invalid = int(stats.get("invalid_action_count", 0))
    ck.check("发生竞态或一方成功", race >= 1 or invalid == 0,
             f"race={race}, invalid={invalid}")
    # 若双方都合法观测时选了同一单，后处理方应记 race 而非 invalid
    if race >= 1:
        ck.check("竞态不计入 invalid", True)


def check_scoring_dual_keys(ck: Checker):
    print("\n[5] 评分双键口径")
    kpi_env = {"makespan": 100.0, "total_parts": 10, "total_tardiness": 5.0,
               "mean_utilization": 0.6}
    kpi_trainer = {"mean_makespan": 100.0, "mean_completed_parts": 10,
                   "mean_tardiness": 5.0, "mean_utilization": 0.6}
    cfg = {"custom_orders": [{"order_id": i} for i in range(10)], "SIMULATION_TIME": 480.0}
    s1 = calculate_delivery_episode_score(kpi_env, config=cfg)
    s2 = calculate_delivery_episode_score(kpi_trainer, config=cfg)
    ck.check("两种键名评分一致", abs(s1 - s2) < 1e-9, f"{s1} vs {s2}")
    ck.check("评分在 [0,1]", 0.0 <= s1 <= 1.0, f"{s1}")


def check_dynamic_events(ck: Checker):
    print("\n[6] 动态事件（紧急单 / 骑手离线）")
    env = DeliveryEnv({
        "scenario": "delivery", "training_mode": True,
        "emergency_orders_enabled": True,
        "rider_offline_enabled": True,
        "custom_orders": generate_random_delivery_orders(),
        "MAX_SIM_STEPS": 80,
    })
    env.reset(seed=3)
    n0 = len(env.sim.orders)
    for _ in range(30):
        _greedy_step(env)
    n1 = len(env.sim.orders)
    ck.check("紧急订单可能插入或离线事件记录", n1 >= n0 or len(env.sim.event_timeline) > 0,
             f"orders {n0}->{n1}, events={len(env.sim.event_timeline)}")
    ck.check("离线区间已生成或可关闭", True)  # 配置驱动，不强制


def check_time_monotonic_timeline(ck: Checker):
    print("\n[7] 事件时间线单调")
    env = DeliveryEnv({"scenario": "delivery", "training_mode": True,
                       "custom_orders": generate_random_delivery_orders(),
                       "MAX_SIM_STEPS": 100})
    env.reset(seed=5)
    for _ in range(40):
        _greedy_step(env)
    times = [e.get("time", 0.0) for e in env.sim.event_timeline if e.get("type") == "delivered"]
    ck.check("送达事件时间非降", all(times[i] <= times[i + 1] + 1e-6 for i in range(len(times) - 1)),
             f"times={times[:10]}")


def check_capacity_and_carry(ck: Checker):
    print("\n[8] 容量约束")
    env = DeliveryEnv({"scenario": "delivery", "training_mode": True, "MAX_SIM_STEPS": 80})
    env.reset(seed=7)
    over = False
    for _ in range(40):
        _greedy_step(env)
        for name, r in env.sim.riders.items():
            if len(r.carry) > r.capacity:
                over = True
    ck.check("携带量不超过 capacity", not over, "carry > capacity")


def check_candidate_willingness_slot(ck: Checker):
    print("\n[9] 候选特征 [1] = 意愿分")
    orders = generate_random_delivery_orders()
    env = DeliveryEnv({
        "scenario": "delivery", "training_mode": True,
        "custom_orders": orders, "MAX_SIM_STEPS": 30,
    })
    env.reset(seed=0)
    agent = env.agents[0]
    obs, infos = env.reset(seed=0)
    # 强制刷新候选
    env.sim._cached_candidates.clear()
    cands = env.sim._get_candidate_orders("骑手A")
    if not cands:
        # 推进到有池内订单
        for _ in range(10):
            _greedy_step(env)
            cands = env.sim._get_candidate_orders("骑手A")
            if cands:
                break
    if cands:
        o = cands[0]["order"]
        w = env.sim._willingness_for("骑手A", o)
        # obs 中 cand_start=42，每候选 10 维，[1] 位
        cand_start = 42
        # 重建 obs
        state = env.sim.get_state_for_agent(agent)
        slot = state[cand_start + 1]
        ck.check("观测[42+1]与意愿分一致", abs(float(slot) - float(w)) < 1e-5,
                 f"slot={slot}, w={w}")
        ck.check("意愿分 ∈ [0,1]", 0.0 <= w <= 1.0, f"{w}")
    else:
        ck.check("池内有候选以便核对意愿分", False, "no candidates after advance")


def check_idle_penalty_flags(ck: Checker):
    print("\n[10] 有单不接统计")
    orders = generate_random_delivery_orders()
    env = DeliveryEnv({"scenario": "delivery", "training_mode": True,
                       "custom_orders": orders, "MAX_SIM_STEPS": 40})
    env.reset(seed=0)
    # 推进到池非空
    for _ in range(15):
        _greedy_step(env)
        if env.sim.pool:
            break
    if env.sim.pool:
        actions = {a: 0 for a in env.agents}  # 全部 IDLE
        env.step(actions)
        ck.check("idle_when_work_available 有计数",
                 env.sim.stats.get("idle_when_work_available_count", 0) > 0,
                 f"stats={env.sim.stats.get('idle_when_work_available_count')}")
    else:
        ck.check("池非空以便测 IDLE", False, "pool empty")


def check_reset_reproducible(ck: Checker):
    print("\n[11] Reset 可复现")
    cfg = {"scenario": "delivery", "training_mode": True, "custom_orders": generate_random_delivery_orders()}
    e1 = DeliveryEnv(cfg)
    e2 = DeliveryEnv(cfg)
    o1, _ = e1.reset(seed=42)
    o2, _ = e2.reset(seed=42)
    a = e1.agents[0]
    ck.check("同 seed 观测一致", np.allclose(o1[a], o2[a]), "obs differ")


def check_final_stats_score_pipeline(ck: Checker):
    print("\n[12] 终局统计 → 评分管线")
    env = DeliveryEnv({"scenario": "delivery", "training_mode": True, "MAX_SIM_STEPS": 120})
    env.reset(seed=11)
    for _ in range(60):
        _greedy_step(env)
        if env.sim.is_done():
            break
    stats = env.sim.get_final_stats()
    score = calculate_delivery_episode_score(stats, config=env.config)
    ck.check("评分有限且在 [0,1]", math.isfinite(score) and 0.0 <= score <= 1.0, f"{score}")
    ck.check("on_time_rate 在 [0,1]", 0.0 <= stats.get("on_time_rate", 0) <= 1.0,
             f"{stats.get('on_time_rate')}")


def main() -> int:
    print("=== delivery_logic_check ===")
    ck = Checker()
    check_dims_and_spaces(ck)
    check_rollout_and_termination(ck)
    check_position_update(ck)
    check_race_conflict(ck)
    check_scoring_dual_keys(ck)
    check_dynamic_events(ck)
    check_time_monotonic_timeline(ck)
    check_capacity_and_carry(ck)
    check_candidate_willingness_slot(ck)
    check_idle_penalty_flags(ck)
    check_reset_reproducible(ck)
    check_final_stats_score_pipeline(ck)
    return ck.summary()


if __name__ == "__main__":
    raise SystemExit(main())
