"""场景分发集成校验：工厂不受影响 + delivery 分发 + 上游意愿分逐单核对。

纯环境层，无需 TensorFlow。工厂场景依赖 simpy，若缺失会跳过工厂部分并说明。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.w_factory_env import make_parallel_env  # noqa: E402
from environments.delivery_env import DeliveryEnv  # noqa: E402
from environments.delivery_config import (  # noqa: E402
    generate_random_delivery_orders,
    generate_mock_upstream_candidates,
)


class Checker:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []

    def check(self, name: str, cond: bool, detail: str = ""):
        if cond:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {detail}")
            print(f"  [FAIL] {name} — {detail}")

    def skip(self, name: str, detail: str = ""):
        self.skipped += 1
        print(f"  [SKIP] {name} — {detail}")

    def summary(self) -> int:
        total = self.passed + self.failed
        print(f"\n=== dispatch_integration_check: {self.passed}/{total} passed"
              + (f", {self.skipped} skipped" if self.skipped else "") + " ===")
        if self.errors:
            print("Failures:")
            for e in self.errors:
                print(" -", e)
        return 0 if self.failed == 0 else 1


def check_factory_unaffected(ck: Checker):
    print("\n[1] 工厂场景未受影响")
    try:
        env = make_parallel_env({"scenario": "factory"})
    except Exception as e:
        ck.skip("工厂环境创建", f"依赖不可用: {e}")
        return
    try:
        obs, infos = env.reset()
        agent = env.agents[0]
        o = obs[agent]
        ck.check("工厂观测 146 维", o.shape == (146,), f"got {o.shape}")
        aspace = env.action_space(agent)
        # 原工厂为 MultiDiscrete([11,11]) 双头
        nvec = list(aspace.nvec) if hasattr(aspace, "nvec") else [aspace.n]
        ck.check("工厂动作双头 [11,11]", nvec == [11, 11], f"got {nvec}")
        gs = infos[agent].get("global_state")
        if gs is not None:
            ck.check("工厂全局状态 19 维", np.asarray(gs).shape == (19,), f"{np.asarray(gs).shape}")
        else:
            ck.check("工厂 infos 含 global_state", False, "missing")
    except Exception as e:
        ck.check("工厂 reset/接口", False, str(e))


def check_delivery_dispatch(ck: Checker):
    print("\n[2] make_parallel_env → delivery 分发")
    env = make_parallel_env({"scenario": "delivery", "training_mode": True})
    obs, infos = env.reset(seed=0)
    agent = env.agents[0]
    ck.check("delivery 观测 146", obs[agent].shape == (146,), f"{obs[agent].shape}")
    ck.check("delivery 5 agents", len(env.agents) == 5, f"{env.agents}")
    ck.check("meta.scenario=delivery", env.obs_meta.get("scenario") == "delivery",
             f"{env.obs_meta.get('scenario')}")
    mask = infos[agent]["action_mask"]
    ck.check("delivery mask 11", mask.shape == (11,), f"{mask.shape}")
    ck.check("infos 有 candidates_map", "candidates_map" in infos[agent], f"{list(infos[agent])}")


def check_upstream_willingness_per_order(ck: Checker):
    print("\n[3] 上游意愿分逐单核对")
    # 全部 t=0 就绪，保证 reset 后池内有单、候选非空
    orders = []
    for i in range(8):
        ang = 2 * np.pi * i / 8
        orders.append({
            "order_id": i,
            "order_type": "普通餐品",
            "pickup": (10 + 3 * np.cos(ang), 10 + 3 * np.sin(ang)),
            "dropoff": (10 + 5 * np.cos(ang), 10 + 5 * np.sin(ang)),
            "ready_time": 0.0,
            "due_date": 50.0 + 4 * i,
            "priority": 2,
            "weight": 1.0,
        })
    RIDERS = ["骑手A", "骑手B", "骑手C", "骑手D", "骑手E"]
    mock = generate_mock_upstream_candidates(orders, RIDERS, top_k=10, seed=9)

    env = DeliveryEnv({
        "scenario": "delivery", "training_mode": True,
        "custom_orders": orders,
        "candidate_source": "upstream",
        "upstream_order_by": "urgency",
        "upstream_candidates": mock,
    })
    env.reset(seed=0)
    env.sim._cached_candidates.clear()
    agent = "agent_骑手A"
    infos = env._build_infos()
    cands = infos[agent]["candidates_map"]
    upstream_map = {int(e["order_id"]): float(e["willingness"]) for e in mock.get("骑手A", [])}
    checked = 0
    mismatch = []
    for c in cands:
        oid = int(c["part_id"])
        if oid in upstream_map:
            checked += 1
            if abs(c["willingness"] - upstream_map[oid]) > 1e-5:
                mismatch.append((oid, c["willingness"], upstream_map[oid]))
    ck.check("候选非空", len(cands) > 0, f"n={len(cands)}, pool={len(env.sim.pool)}")
    ck.check("上游覆盖单意愿一致", checked > 0 and not mismatch,
             f"checked={checked}, mismatch={mismatch[:3]}, upstream={list(upstream_map)[:5]}")
    # 观测槽位
    state = env.sim.get_state_for_agent(agent)
    cand_start = 42
    slots_ok = True
    for c in cands:
        slot = float(state[cand_start + c["queue_index"] * 10 + 1])
        if abs(slot - c["willingness"]) > 1e-5:
            slots_ok = False
            break
    ck.check("观测槽位与 candidates_map 一致", slots_ok, "slot mismatch")


def check_factory_default_no_delivery_keys(ck: Checker):
    print("\n[4] 默认工厂不注入 delivery 专属键")
    try:
        env = make_parallel_env({})  # 默认 factory
    except Exception as e:
        ck.skip("默认工厂创建", str(e))
        return
    # 工厂环境不应有 obs_meta.scenario=delivery
    obs, infos = env.reset()
    meta = infos[env.agents[0]].get("obs_meta") or {}
    ck.check("默认场景非 delivery", meta.get("scenario") != "delivery",
             f"scenario={meta.get('scenario')}")


def check_score_config_shared(ck: Checker):
    print("\n[5] 评分/配置共享面")
    from environments.delivery_config import calculate_delivery_episode_score
    from environments.w_factory_config import calculate_episode_score  # noqa: F401
    s = calculate_delivery_episode_score(
        {"makespan": 50, "total_parts": 5, "total_tardiness": 0, "mean_utilization": 0.5},
        config={"custom_orders": [0, 1, 2, 3, 4], "SIMULATION_TIME": 480},
    )
    ck.check("配送评分可计算", 0.0 <= s <= 1.0, f"{s}")


def check_action_mask_consistency(ck: Checker):
    print("\n[6] mask 与 candidates_map 一致")
    env = DeliveryEnv({"scenario": "delivery", "training_mode": True})
    env.reset(seed=1)
    infos = env._build_infos()
    for agent in env.agents:
        mask = infos[agent]["action_mask"]
        cands = infos[agent]["candidates_map"]
        for c in cands:
            if mask[c["action"]] != 1.0 and len(env.sim.riders[agent.replace("agent_", "")].carry) < \
                    env.sim.riders[agent.replace("agent_", "")].capacity:
                ck.check(f"{agent} mask 对齐", False, f"action {c['action']} mask=0")
                break
        else:
            continue
        break
    else:
        ck.check("mask 与 candidates_map 对齐", True)


def main() -> int:
    print("=== dispatch_integration_check ===")
    ck = Checker()
    check_factory_unaffected(ck)
    check_delivery_dispatch(ck)
    check_upstream_willingness_per_order(ck)
    check_factory_default_no_delivery_keys(ck)
    check_score_config_shared(ck)
    check_action_mask_consistency(ck)
    return ck.summary()


if __name__ == "__main__":
    raise SystemExit(main())
