"""真实订单池训练链路校验（无 TensorFlow）。

覆盖：池加载 / 窗口采样 / 时间 rebase / env rollout / mock 上游 / 骑手注入。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.real_data_loader import (  # noqa: E402
    load_real_training_pool,
    sample_order_window,
)
from environments.delivery_config import generate_mock_upstream_candidates  # noqa: E402
from environments.delivery_env import DeliveryEnv  # noqa: E402


class Checker:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name, cond, detail=""):
        if cond:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {detail}")
            print(f"  [FAIL] {name} — {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n=== real_pool_train_check: {self.passed}/{total} passed ===")
        for e in self.errors:
            print(" -", e)
        return 0 if self.failed == 0 else 1


ORDERS = ROOT / "data/real_incoming/orders_guoluo_20260915_ready.csv"
RIDERS = ROOT / "data/real_incoming/riders_guoluo_20260915_full.csv"


def main():
    print("=== real_pool_train_check ===")
    ck = Checker()
    if not ORDERS.exists():
        ck.check("样本存在", False, str(ORDERS))
        return ck.summary()

    print("\n[1] 订单池加载")
    pool_cfg = load_real_training_pool(ORDERS, RIDERS, max_orders=2000, max_riders=5)
    pool = pool_cfg.get("order_pool") or []
    ck.check("池非空", len(pool) > 0, f"n={len(pool)}")
    ck.check("含 riders", bool(pool_cfg.get("riders")), str(list((pool_cfg.get("riders") or {}))))
    ck.check("含 geo_config", bool(pool_cfg.get("geo_config")), str(pool_cfg.get("geo_config")))
    ready = [float(o["ready_time"]) for o in pool]
    ck.check("池按 ready 排序", all(ready[i] <= ready[i + 1] + 1e-6 for i in range(len(ready) - 1)))

    print("\n[2] 窗口采样")
    w1 = sample_order_window(pool, 80, seed=1, rebase_time=True)
    w2 = sample_order_window(pool, 80, seed=2, rebase_time=True)
    ck.check("窗口长度 80", len(w1) == 80, str(len(w1)))
    r1 = [o["ready_time"] for o in w1]
    ck.check("rebase 后起点≈0", min(r1) < 1.0, f"min={min(r1)}")
    ck.check("due>ready", all(o["due_date"] > o["ready_time"] for o in w1))
    ids1 = {o["order_id"] for o in w1}
    ids2 = {o["order_id"] for o in w2}
    ck.check("不同 seed 窗口可不同", ids1 != ids2 or len(pool) <= 80, f"{len(ids1 & ids2)} overlap")
    full = sample_order_window(pool, 10 ** 6, seed=0)
    ck.check("size>池 → 整池", len(full) == len(pool), f"{len(full)} vs {len(pool)}")

    print("\n[3] env + mock 上游（模拟 trainer 回合配置）")
    riders = pool_cfg["riders"]
    geo = pool_cfg["geo_config"]
    orders_ep = sample_order_window(pool, 60, seed=3, rebase_time=True)
    upstream = generate_mock_upstream_candidates(
        orders_ep, list(riders.keys()), top_k=10, seed=3)
    env = DeliveryEnv({
        "scenario": "delivery",
        "training_mode": True,
        "custom_orders": orders_ep,
        "riders": riders,
        "geo_config": geo,
        "candidate_source": "upstream",
        "upstream_order_by": "urgency",
        "upstream_candidates": upstream,
        "MAX_SIM_STEPS": 80,
    })
    obs, infos = env.reset(seed=0)
    a = env.agents[0]
    ck.check("obs 146", obs[a].shape == (146,), str(obs[a].shape))
    ck.check("5 agents", len(env.agents) == 5, str(env.agents))
    ck.check("meta source upstream", env.obs_meta.get("candidate_source") == "upstream")
    for _ in range(20):
        actions = {}
        for ag in env.agents:
            mask = env.infos[ag]["action_mask"]
            legal = np.where(mask)[0]
            non_idle = [x for x in legal if x != 0]
            actions[ag] = int(non_idle[0]) if non_idle else 0
        obs, rewards, terms, truncs, infos = env.step(actions)
        if all(terms.values()) or all(truncs.values()):
            break
    st = env.sim.get_final_stats()
    ck.check("final_stats 有完成率", "completion_rate" in st)
    ck.check("upstream 有兑现统计键", "upstream_matched_count" in st)

    print("\n[4] CLI 参数存在性（源码静态检查；MAPPO 已归档，主干=hybrid）")
    src = (ROOT / "hybrid_train.py").read_text(encoding="utf-8")
    ck.check("--real-orders 在训练入口", "--real-orders" in src)
    ck.check("--real-riders 在训练入口", "--real-riders" in src)
    ck.check("load_real_training_pool 被调用", "load_real_training_pool" in src)
    ck.check("--episode-order-size 在训练入口", "--episode-order-size" in src)
    tr = (ROOT / "hybrid/trainer.py").read_text(encoding="utf-8")
    ck.check("trainer 使用 ProcessPoolExecutor 并行采集", "ProcessPoolExecutor" in tr)
    ck.check("trainer 透传 env_config（含 real_order_pool）", "env_config" in tr)
    ck.check("trainer 软更新目标网络", "target_soft_tau" in tr)
    ev = (ROOT / "evaluation_delivery.py").read_text(encoding="utf-8")
    ck.check("评估入口支持 --hybrid-scorer", "--hybrid-scorer" in ev)

    return ck.summary()


if __name__ == "__main__":
    raise SystemExit(main())
