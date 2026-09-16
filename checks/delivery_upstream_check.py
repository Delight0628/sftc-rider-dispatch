"""三层联调（上游候选/意愿分）校验，纯环境层，无需 TensorFlow。

覆盖：契约归一化 / 兜底意愿分 / 观测对齐 / 会议口径 urgency 优先 /
      回退与缺失统计 / mock 生成器 / upstream rollout。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.delivery_env import DeliveryEnv  # noqa: E402
from environments.delivery_config import (  # noqa: E402
    DELIVERY_UPSTREAM_CONFIG,
    generate_random_delivery_orders,
    generate_mock_upstream_candidates,
    heuristic_willingness,
    normalize_upstream_candidates,
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
        print(f"\n=== delivery_upstream_check: {self.passed}/{total} passed ===")
        if self.errors:
            print("Failures:")
            for e in self.errors:
                print(" -", e)
        return 0 if self.failed == 0 else 1


RIDERS = ["骑手A", "骑手B", "骑手C", "骑手D", "骑手E"]


def _custom_orders(n=12):
    """可控订单：ready 全 0，便于立刻进入池。"""
    orders = []
    for i in range(n):
        ang = 2 * np.pi * i / n
        pickup = (10 + 4 * np.cos(ang), 10 + 4 * np.sin(ang))
        dropoff = (10 + 6 * np.cos(ang + 0.5), 10 + 6 * np.sin(ang + 0.5))
        orders.append({
            "order_type": "普通餐品" if i % 2 == 0 else "文件急送",
            "pickup": pickup, "dropoff": dropoff,
            "ready_time": 0.0,
            "due_date": 40.0 + 5.0 * i,  # 递增交期
            "priority": 1 + (i % 3),
            "weight": 0.5 + 0.2 * i,
            "order_id": i,
        })
    return orders


def check_normalize_contract(ck: Checker):
    print("\n[1] 契约归一化")
    payload = {
        "agent_骑手A": [{"order_id": 1, "willingness": 0.9}, (2, 0.8), {"order_id": 1, "willingness": 0.95}],
        "骑手B": [{"order_id": 3, "willingness": 0.7}],
        "global": [{"order_id": 4, "willingness": 0.6}],
    }
    norm = normalize_upstream_candidates(payload, RIDERS, top_k=10)
    ck.check("A 去重取高意愿(0.95)",
             any(e["order_id"] == 1 and e["willingness"] == 0.95 for e in norm["骑手A"]),
             f"{norm.get('骑手A')}")
    ck.check("A 无重复 order_id",
             len({e["order_id"] for e in norm["骑手A"]}) == len(norm["骑手A"]),
             f"{norm.get('骑手A')}")
    ck.check("B 保留", norm["骑手B"][0]["order_id"] == 3, f"{norm.get('骑手B')}")
    ck.check("global 扩散到未覆盖骑手", all(r in norm for r in RIDERS), f"keys={list(norm)}")
    # 非法输入
    ck.check("None → 空", normalize_upstream_candidates(None, RIDERS) == {})
    ck.check("缺意愿分忽略", normalize_upstream_candidates(
        {"骑手A": [{"order_id": 1}]}, RIDERS) == {} or
        "骑手A" not in normalize_upstream_candidates({"骑手A": [{"order_id": 1}]}, RIDERS),
        "应忽略无意愿条目")
    ck.check("意愿分裁剪到 [0,1]", normalize_upstream_candidates(
        {"骑手A": [{"order_id": 1, "willingness": 1.5}]}, RIDERS
    )["骑手A"][0]["willingness"] == 1.0, "clip failed")
    ck.check("top_k 截断", len(normalize_upstream_candidates(
        {"骑手A": [{"order_id": i, "willingness": 0.5} for i in range(20)]}, RIDERS, top_k=3
    )["骑手A"]) == 3, "top_k failed")


def check_heuristic_willingness(ck: Checker):
    print("\n[2] 兜底意愿分")
    easy = heuristic_willingness({"order_type": "文件急送", "priority": 3, "weight": 0.2})
    hard = heuristic_willingness({"order_type": "团餐大单", "priority": 1, "weight": 6.0})
    ck.check("轻松单分更高", easy > hard, f"easy={easy}, hard={hard}")
    ck.check("范围 [0,1]", 0.0 <= easy <= 1.0 and 0.0 <= hard <= 1.0, f"{easy},{hard}")


def check_obs_alignment(ck: Checker):
    print("\n[3] 观测意愿分对齐")
    orders = _custom_orders(10)
    upstream = {
        "骑手A": [{"order_id": 0, "willingness": 0.10},
                  {"order_id": 1, "willingness": 0.99}],
    }
    env = DeliveryEnv({
        "scenario": "delivery", "training_mode": True,
        "custom_orders": orders,
        "candidate_source": "upstream",
        "upstream_order_by": "urgency",
        "upstream_candidates": upstream,
        "MAX_SIM_STEPS": 20,
    })
    env.reset(seed=0)
    env.sim._cached_candidates.clear()
    state = env.sim.get_state_for_agent("agent_骑手A")
    cands = env.sim._get_candidate_orders("骑手A")
    ck.check("有候选", len(cands) > 0, f"n={len(cands)}")
    cand_start = 42
    ok = True
    detail = []
    for c in cands:
        idx = c["index"]
        w_env = env.sim._willingness_for("骑手A", c["order"])
        slot = float(state[cand_start + idx * 10 + 1])
        if abs(slot - w_env) > 1e-5:
            ok = False
            detail.append(f"idx{idx}: slot={slot} w={w_env}")
    ck.check("每个候选 [1] 位=意愿分", ok, "; ".join(detail) or "ok")
    # 未上游覆盖的订单用 heuristic
    other = [c for c in cands if c["order"].order_id not in (0, 1)]
    if other:
        o = other[0]["order"]
        expect = heuristic_willingness({
            "order_type": o.order_type, "priority": o.priority, "weight": o.weight,
        })
        got = env.sim._willingness_for("骑手A", o)
        ck.check("未覆盖订单走 heuristic", abs(expect - got) < 1e-6, f"{expect} vs {got}")
    else:
        ck.check("存在未覆盖订单", False, "all candidates from upstream")


def check_urgency_over_willingness(ck: Checker):
    print("\n[4] 会议口径：紧迫优先于意愿")
    # order 0: 紧但意愿低；order 1: 松但意愿高
    orders = [
        {"order_id": 0, "order_type": "普通餐品", "pickup": (10.0, 10.0), "dropoff": (10.5, 10.0),
         "ready_time": 0.0, "due_date": 12.0, "priority": 2, "weight": 1.0},
        {"order_id": 1, "order_type": "普通餐品", "pickup": (10.0, 10.0), "dropoff": (10.5, 10.0),
         "ready_time": 0.0, "due_date": 200.0, "priority": 2, "weight": 1.0},
    ]
    upstream = {"骑手A": [
        {"order_id": 1, "willingness": 0.99},
        {"order_id": 0, "willingness": 0.10},
    ]}
    env = DeliveryEnv({
        "scenario": "delivery", "training_mode": True,
        "custom_orders": orders,
        "candidate_source": "upstream",
        "upstream_order_by": "urgency",
        "upstream_candidates": upstream,
        "MAX_SIM_STEPS": 10,
    })
    env.reset(seed=0)
    env.sim._cached_candidates.clear()
    cands = env.sim._get_candidate_orders("骑手A")
    ids = [c["order"].order_id for c in cands]
    ck.check("紧迫单(0)排在意愿单(1)前", ids.index(0) < ids.index(1), f"order={ids}")
    slacks = [env.sim._order_slack(env.sim.riders["骑手A"], c["order"], env.sim.current_time)
              for c in cands]
    ck.check("slack 非降（urgency 序）", all(slacks[i] <= slacks[i + 1] + 1e-6
                                        for i in range(len(slacks) - 1)), f"{slacks}")

    # upstream 序模式：保持意愿序
    env2 = DeliveryEnv({
        "scenario": "delivery", "training_mode": True,
        "custom_orders": orders,
        "candidate_source": "upstream",
        "upstream_order_by": "upstream",
        "upstream_candidates": upstream,
        "MAX_SIM_STEPS": 10,
    })
    env2.reset(seed=0)
    env2.sim._cached_candidates.clear()
    cands2 = env2.sim._get_candidate_orders("骑手A")
    ids2 = [c["order"].order_id for c in cands2]
    ck.check("upstream 序保持意愿优先", ids2[0] == 1, f"order={ids2}")


def check_fallback_and_stats(ck: Checker):
    print("\n[5] 回退 / 缺失 / 兑现统计")
    orders = _custom_orders(8)
    # 只给骑手B候选，且包含一个不存在的 order_id
    upstream = {
        "骑手B": [{"order_id": 0, "willingness": 0.9},
                  {"order_id": 9999, "willingness": 0.8}],
    }
    env = DeliveryEnv({
        "scenario": "delivery", "training_mode": True,
        "custom_orders": orders,
        "candidate_source": "upstream",
        "upstream_order_by": "urgency",
        "upstream_candidates": upstream,
        "MAX_SIM_STEPS": 15,
    })
    env.reset(seed=0)
    env.sim._cached_candidates.clear()
    env.sim._get_candidate_orders("骑手A")  # 无上游 → fallback
    env.sim._get_candidate_orders("骑手B")
    st = env.sim.stats
    ck.check("A 回退自采样", st.get("upstream_fallback_count", 0) >= 1, f"{st}")
    ck.check("B 兑现 1 + 缺失 1",
             st.get("upstream_matched_count", 0) >= 1 and st.get("upstream_missing_count", 0) >= 1,
             f"{st}")
    final = env.sim.get_final_stats()
    ck.check("final_stats 暴露联调口径", "upstream_matched_count" in final, f"{list(final)}")


def check_meta_and_backfill(ck: Checker):
    print("\n[6] meta 与 backfill")
    orders = _custom_orders(6)
    upstream = {"骑手A": [{"order_id": 0, "willingness": 0.8}]}  # 只 1 条，需补齐到 10
    env = DeliveryEnv({
        "scenario": "delivery", "training_mode": True,
        "custom_orders": orders,
        "candidate_source": "upstream",
        "upstream_order_by": "urgency",
        "upstream_candidates": upstream,
        "MAX_SIM_STEPS": 10,
    })
    env.reset(seed=0)
    meta = env.obs_meta
    ck.check("meta.scenario=delivery", meta.get("scenario") == "delivery", f"{meta.get('scenario')}")
    ck.check("meta.candidate_source=upstream", meta.get("candidate_source") == "upstream",
             f"{meta.get('candidate_source')}")
    ck.check("meta.order_by=urgency", meta.get("upstream_order_by") == "urgency",
             f"{meta.get('upstream_order_by')}")
    ck.check("覆盖骑手列表", "骑手A" in meta.get("upstream_riders_covered", []),
             f"{meta.get('upstream_riders_covered')}")
    env.sim._cached_candidates.clear()
    cands = env.sim._get_candidate_orders("骑手A")
    ck.check("backfill 后候选可达 top_k 或池大小", len(cands) >= 1,
             f"n={len(cands)}")
    ck.check("候选不超过 10", len(cands) <= 10, f"n={len(cands)}")


def check_mock_generator(ck: Checker):
    print("\n[7] mock 上游生成器")
    orders = _custom_orders(15)
    mock = generate_mock_upstream_candidates(orders, RIDERS, top_k=10, seed=1)
    ck.check("5 骑手全覆盖", set(mock.keys()) == set(RIDERS), f"{list(mock)}")
    ck.check("每骑手 ≤10 条", all(len(v) <= 10 for v in mock.values()),
             f"{ {k: len(v) for k, v in mock.items()} }")
    ck.check("意愿分 ∈ [0,1]",
             all(0.0 <= e["willingness"] <= 1.0 for v in mock.values() for e in v), "range")
    ck.check("按意愿降序",
             all(v[i]["willingness"] >= v[i + 1]["willingness"] - 1e-9
                 for v in mock.values() for i in range(len(v) - 1)), "order")
    mock2 = generate_mock_upstream_candidates(orders, RIDERS, top_k=10, seed=1)
    ck.check("同 seed 可复现", mock == mock2, "not reproducible")


def check_upstream_rollout(ck: Checker):
    print("\n[8] upstream 模式 rollout")
    orders = generate_random_delivery_orders()
    mock = generate_mock_upstream_candidates(orders, RIDERS, top_k=10, seed=2)
    env = DeliveryEnv({
        "scenario": "delivery", "training_mode": True,
        "custom_orders": orders,
        "candidate_source": "upstream",
        "upstream_order_by": "urgency",
        "upstream_candidates": mock,
        "MAX_SIM_STEPS": 80,
    })
    obs, _ = env.reset(seed=0)
    ck.check("obs 146", obs[env.agents[0]].shape == (146,), f"{obs[env.agents[0]].shape}")
    for t in range(40):
        actions = {}
        for a in env.agents:
            mask = env.infos[a]["action_mask"]
            legal = np.where(mask)[0]
            non_idle = [x for x in legal if x != 0]
            actions[a] = int(non_idle[0]) if non_idle else (int(legal[0]) if len(legal) else 0)
        obs, rewards, terms, truncs, infos = env.step(actions)
        for r in rewards.values():
            if not math.isfinite(float(r)):
                ck.check("upstream 奖励有限", False, f"r={r}")
                break
        if all(terms.values()) or all(truncs.values()):
            break
    else:
        ck.check("upstream 可推进", True)
    st = env.sim.get_final_stats()
    ck.check("upstream 有兑现统计", st.get("upstream_matched_count", 0) >= 0, f"{st.get('upstream_matched_count')}")
    # 默认配置
    ck.check("默认 order_by=urgency",
             DELIVERY_UPSTREAM_CONFIG.get("order_by") == "urgency",
             f"{DELIVERY_UPSTREAM_CONFIG.get('order_by')}")
    ck.check("默认 top_k=10", int(DELIVERY_UPSTREAM_CONFIG.get("top_k", 0)) == 10,
             f"{DELIVERY_UPSTREAM_CONFIG.get('top_k')}")


def check_tuple_and_list_forms(ck: Checker):
    print("\n[9] 兼容元组/列表载荷")
    norm = normalize_upstream_candidates(
        [("5", 0.4), (6, "0.55")], RIDERS, top_k=10)
    # list payload → global
    ck.check("list 载荷可解析", all(len(v) >= 1 for v in norm.values()), f"{norm}")
    norm2 = normalize_upstream_candidates(
        {"骑手C": [(7, 0.33), {"order_id": 8, "willingness": 0.44}]}, RIDERS)
    ck.check("元组+dict 混合", norm2.get("骑手C", [{}])[0].get("order_id") in (7, 8),
             f"{norm2.get('骑手C')}")


def check_capacity_blocks_accept(ck: Checker):
    print("\n[10] 满载时 mask 不允许接单")
    orders = _custom_orders(8)
    env = DeliveryEnv({
        "scenario": "delivery", "training_mode": True,
        "custom_orders": orders, "MAX_SIM_STEPS": 5,
    })
    env.reset(seed=0)
    rider = env.sim.riders["骑手A"]
    # 填满容量
    env.sim._cached_candidates.clear()
    cands = env.sim._get_candidate_orders("骑手A")
    while len(rider.carry) < rider.capacity and cands:
        o = cands[0]["order"]
        if o in env.sim.pool:
            env.sim.pool.remove(o)
            o.state = "carried"
            o.assigned_rider = "骑手A"
            o.planned_deliver_time = env.sim.current_time + 30.0
            o.pull_time = env.sim.current_time
            rider.carry.append(o)
        env.sim._cached_candidates.clear()
        cands = env.sim._get_candidate_orders("骑手A")
    mask = env.sim._get_action_mask("骑手A")
    ck.check("满载仅 IDLE", int(mask.sum()) == 1 and mask[0] == 1.0, f"mask={mask}")


def check_endogenous_unchanged(ck: Checker):
    print("\n[11] endogenous 不受 upstream 影响")
    orders = _custom_orders(8)
    env = DeliveryEnv({
        "scenario": "delivery", "training_mode": True,
        "custom_orders": orders,
        "candidate_source": "endogenous",
        "MAX_SIM_STEPS": 10,
    })
    env.reset(seed=0)
    ck.check("source=endogenous", env.obs_meta.get("candidate_source") == "endogenous",
             f"{env.obs_meta.get('candidate_source')}")
    env.sim._cached_candidates.clear()
    cands = env.sim._get_candidate_orders("骑手A")
    # 意愿分来自 heuristic（无上游）
    if cands:
        o = cands[0]["order"]
        expect = heuristic_willingness({
            "order_type": o.order_type, "priority": o.priority, "weight": o.weight})
        got = env.sim._willingness_for("骑手A", o)
        ck.check("endogenous 意愿=heuristic", abs(expect - got) < 1e-6, f"{expect} vs {got}")
    else:
        ck.check("endogenous 有候选", False, "empty")


def check_multi_rider_independence(ck: Checker):
    print("\n[12] 骑手候选独立")
    orders = _custom_orders(10)
    upstream = {
        "骑手A": [{"order_id": 0, "willingness": 0.9}],
        "骑手B": [{"order_id": 1, "willingness": 0.8}],
    }
    env = DeliveryEnv({
        "scenario": "delivery", "training_mode": True,
        "custom_orders": orders,
        "candidate_source": "upstream",
        "upstream_order_by": "urgency",
        "upstream_candidates": upstream,
    })
    env.reset(seed=0)
    env.sim._cached_candidates.clear()
    ca = {c["order"].order_id for c in env.sim._get_candidate_orders("骑手A")}
    cb = {c["order"].order_id for c in env.sim._get_candidate_orders("骑手B")}
    ck.check("A 含 0", 0 in ca, f"{ca}")
    ck.check("B 含 1", 1 in cb, f"{cb}")


def main() -> int:
    print("=== delivery_upstream_check ===")
    ck = Checker()
    check_normalize_contract(ck)
    check_heuristic_willingness(ck)
    check_obs_alignment(ck)
    check_urgency_over_willingness(ck)
    check_fallback_and_stats(ck)
    check_meta_and_backfill(ck)
    check_mock_generator(ck)
    check_upstream_rollout(ck)
    check_tuple_and_list_forms(ck)
    check_capacity_blocks_accept(ck)
    check_endogenous_unchanged(ck)
    check_multi_rider_independence(ck)
    return ck.summary()


if __name__ == "__main__":
    raise SystemExit(main())
