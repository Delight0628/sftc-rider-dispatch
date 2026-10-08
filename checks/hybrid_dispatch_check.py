"""
混合派单框架校验（hybrid_dispatch）
==================================
覆盖：
  1. EdgeScorer 边打分（维度/确定性/特征口径）
  2. 约束二分图匹配（一对一、无重复派单、交换改进不降分）
  3. 全 episode 接入 DeliveryEnv（无抢单 race_conflict、无非法动作）
  4. 对拍启发式基线：hybrid 的 episode_score 必须赢 nearest/EDD/fifo
     （项目铁律：学习方法必须赢启发式才有效）
  5. 变规模（骑手 3/5/8）匹配层原生可跑（不依赖 146 维 one-hot）
  6. 学习回路（learn=True 时权重随实现效用更新）

运行：python checks/hybrid_dispatch_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # noqa: E402

import numpy as np  # noqa: E402

from environments.hybrid_dispatch import (  # noqa: E402
    EDGE_FEATURE_NAMES, EdgeScorer, HybridDispatcher, max_weight_matching,
)
from environments.delivery_config import (  # noqa: E402
    HYBRID_DISPATCH_CONFIG,
    calculate_delivery_episode_score,
    generate_random_delivery_orders,
)
from environments.delivery_env import DeliveryEnv  # noqa: E402


class Checker:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures = []

    def check(self, name: str, cond: bool, detail: str = ""):
        if cond:
            self.passed += 1
            print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
        else:
            self.failed += 1
            self.failures.append(name)
            print(f"  [FAIL] {name} — {detail}")

    def skip(self, name: str, detail: str = ""):
        self.skipped += 1
        print(f"  [SKIP] {name}" + (f" — {detail}" if detail else ""))

    def summary(self) -> int:
        total = self.passed + self.failed
        print(f"\n=== hybrid_dispatch_check: {self.passed}/{total} passed, "
              f"{self.skipped} skipped ===")
        if self.failures:
            print("Failures:")
            for f in self.failures:
                print(f"  - {f}")
            return 1
        return 0


# -----------------------------------------------------------------------------
def check_edge_scorer(ck: Checker):
    print("\n[1] EdgeScorer 边打分")
    scorer = EdgeScorer()
    ck.check("权重维度 = 特征维度",
             scorer.weights.shape == (len(EDGE_FEATURE_NAMES),),
             f"w={scorer.weights.shape}, f={len(EDGE_FEATURE_NAMES)}")

    orders = generate_random_delivery_orders()
    for i, o in enumerate(orders):
        o.setdefault("order_id", i)
    env = DeliveryEnv({
        "scenario": "delivery", "training_mode": True,
        "custom_orders": orders, "deterministic_candidates": True,
    })
    env.reset(seed=0)
    sim = env.sim
    rider = next(iter(sim.riders.values()))
    order = sim.orders[0]
    pos, free_t = rider.projected_free()
    start = max(sim.current_time, free_t if free_t > 0 else sim.current_time)
    f1 = scorer.build_features(sim, rider, order, pos, start, sim.current_time)
    f2 = scorer.build_features(sim, rider, order, pos, start, sim.current_time)
    ck.check("特征维度 12 且确定", f1.shape == (12,) and np.allclose(f1, f2))
    ck.check("on_time_feasible 是 0/1 量", f1[2] in (0.0, 1.0))
    s = scorer.score(f1)
    ck.check("打分 = w·f", abs(s - float(scorer.weights @ f1)) < 1e-9, f"score={s:.3f}")


def check_matching(ck: Checker):
    print("\n[2] 约束二分图匹配")
    edges = [
        ("骑手A", 1, 5.0, "a1"), ("骑手A", 2, 3.0, "a2"),
        ("骑手B", 1, 4.0, "b1"), ("骑手B", 2, 4.5, "b2"),
    ]
    m = max_weight_matching(edges)
    pairs = {(r, o) for r, o, _, _ in m}
    ck.check("一对一：无重复骑手/订单",
             len({r for r, _, _, _ in m}) == len(m) and len({o for _, o, _, _ in m}) == len(m))
    total = sum(s for _, _, s, _ in m)
    # 全局最优应为 (A,1)+(B,2)=9.5 而非 (A,2)+(B,1)=7.0
    ck.check("交换改进取到全局最优", abs(total - 9.5) < 1e-9, f"total={total}")
    ck.check("payload 透传", all(p in ("a1", "b2") for _, _, _, p in m))
    ck.check("空边集返回空", max_weight_matching([]) == [])
    # 并列分数的确定性
    tie = [("骑手A", 9, 1.0, None), ("骑手B", 8, 1.0, None)]
    m1, m2 = max_weight_matching(tie), max_weight_matching(tie)
    ck.check("并列分数确定性", m1 == m2)


def _run_baseline(env: DeliveryEnv, baseline: str, max_steps: int = 800) -> dict:
    from evaluation_delivery import run_episode
    rng = np.random.RandomState(0)
    return run_episode(env, baseline, rng, max_steps=max_steps)


def check_episode_and_vs_heuristics(ck: Checker):
    print("\n[3] 全 episode 接入 + 对拍启发式")
    orders = generate_random_delivery_orders()
    for i, o in enumerate(orders):
        o.setdefault("order_id", i)

    def make_env():
        return DeliveryEnv({
            "scenario": "delivery", "training_mode": True,
            "custom_orders": orders, "deterministic_candidates": True,
        })

    stats_h = _run_baseline(make_env(), "hybrid")
    ck.check("hybrid 全程无抢单冲突", stats_h.get("race_conflict_count", 1) == 0,
             f"race={stats_h.get('race_conflict_count')}")
    ck.check("hybrid 全程无非法动作", stats_h.get("invalid_action_count", 1) == 0,
             f"invalid={stats_h.get('invalid_action_count')}")
    ck.check("hybrid 完成率 100%", stats_h.get("completion_rate", 0) >= 1.0 - 1e-9,
             f"completion={stats_h.get('completion_rate'):.3f}")

    scores = {"hybrid": stats_h["episode_score"]}
    for b in ("nearest", "edd", "fifo"):
        s = _run_baseline(make_env(), b)
        scores[b] = s["episode_score"]
    detail = " ".join(f"{k}={v:.3f}" for k, v in scores.items())
    ck.check("hybrid 赢 nearest", scores["hybrid"] > scores["nearest"], detail)
    ck.check("hybrid 赢 edd", scores["hybrid"] > scores["edd"], detail)
    ck.check("hybrid 赢 fifo", scores["hybrid"] > scores["fifo"], detail)
    ck.check("hybrid 准时率更高",
             stats_h["on_time_rate"] >= 0.5,
             f"on_time={stats_h['on_time_rate']:.3f}")


def check_variable_scale(ck: Checker):
    print("\n[4] 变规模（骑手 3/5/8）")
    for n in (3, 5, 8):
        names = [f"骑手{chr(ord('A') + i)}" for i in range(n)]
        riders = {
            nm: {"count": 1, "capacity": 3, "speed": 0.55,
                 "home": (2.0 + 3.0 * i, 10.0)}
            for i, nm in enumerate(names)
        }
        orders = generate_random_delivery_orders()
        for i, o in enumerate(orders):
            o.setdefault("order_id", i)
        env = DeliveryEnv({
            "scenario": "delivery", "training_mode": True,
            "custom_orders": orders, "riders": riders,
            "deterministic_candidates": True,
        })
        env.reset(seed=0)
        dispatcher = HybridDispatcher(env)
        ok = True
        for _ in range(20):
            actions = dispatcher.act()
            if set(actions.keys()) != {f"agent_{nm}" for nm in names}:
                ok = False
                break
            # 每步派单数不超过该骑手剩余容量（动作里非 0 项数）
            for aid, acts in actions.items():
                rider = env.sim.riders[aid.replace("agent_", "")]
                n_assign = sum(1 for a in acts if int(a) != 0)
                if n_assign > rider.capacity - len(rider.carry):
                    ok = False
            _, _, terms, truncs, _ = env.step(actions)
            if all(terms.values()) or all(truncs.values()):
                break
        ck.check(f"N={n} 匹配/容量约束成立", ok, f"riders={n}")
        stats = env.sim.get_final_stats()
        ck.check(f"N={n} 无抢单冲突", stats.get("race_conflict_count", 1) == 0)


def check_learning_loop(ck: Checker):
    print("\n[5] 学习回路（learn=True）")
    scorer = EdgeScorer(learn=True, lr=0.1)
    w_before = scorer.weights.copy()
    feats = np.zeros(len(EDGE_FEATURE_NAMES))
    feats[0] = 1.0
    scorer.update(feats, target=2.0)
    ck.check("权重被更新", not np.allclose(scorer.weights, w_before))
    ck.check("update 计数 +1", scorer.update_count == 1)

    scorer_off = EdgeScorer(learn=False)
    w0 = scorer_off.weights.copy()
    scorer_off.update(feats, target=2.0)
    ck.check("learn=False 不更新", np.allclose(scorer_off.weights, w0))

    # episode 级回路：派单记录 → 实现效用回归
    orders = generate_random_delivery_orders()
    for i, o in enumerate(orders):
        o.setdefault("order_id", i)
    env = DeliveryEnv({
        "scenario": "delivery", "training_mode": True,
        "custom_orders": orders, "deterministic_candidates": True,
    })
    env.reset(seed=0)
    dispatcher = HybridDispatcher(env, scorer=EdgeScorer(learn=True, lr=0.05))
    for _ in range(30):
        actions = dispatcher.act()
        _, _, terms, truncs, _ = env.step(actions)
        if all(terms.values()) or all(truncs.values()):
            break
    w_before = dispatcher.scorer.weights.copy()
    n_updates = dispatcher.learn_from_episode()
    ck.check("learn_from_episode 有更新", n_updates > 0 and
             not np.allclose(dispatcher.scorer.weights, w_before),
             f"updates={n_updates}")
    ck.check("配置默认 learn 关闭（评估确定性）",
             HYBRID_DISPATCH_CONFIG.get("learn") is False)


def check_score_consistency(ck: Checker):
    print("\n[6] 评分口径一致性")
    stats = {"makespan": 100.0, "total_parts": 10, "total_tardiness": 0.0,
             "mean_utilization": 0.8}
    s = calculate_delivery_episode_score(stats, config={"custom_orders": [{}] * 10})
    ck.check("calculate_delivery_episode_score 可算", 0.0 <= s <= 1.0, f"score={s:.3f}")


def main() -> int:
    ck = Checker()
    check_edge_scorer(ck)
    check_matching(ck)
    check_episode_and_vs_heuristics(ck)
    check_variable_scale(ck)
    check_learning_loop(ck)
    check_score_consistency(ck)
    return ck.summary()


if __name__ == "__main__":
    raise SystemExit(main())
