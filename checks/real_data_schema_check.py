"""真实数据映射/导入/KPI 选择校验（纯环境层）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.real_data_schema import (  # noqa: E402
    ORDER_CORE_MAP, RIDER_CORE_MAP, KPI_SPEC, ROADSHOW_KPI_KEYS,
    TRAINING_DIAG_KPI_KEYS, summarize_schema,
)
from environments.real_data_loader import (  # noqa: E402
    extract_order_core, filter_training_orders, GeoProjector,
    build_custom_orders, build_rider_configs, load_real_episode_config,
    offline_kpis_from_real, load_rows,
)
from environments.delivery_env import DeliveryEnv  # noqa: E402
from checks.make_realistic_sample import make_orders, make_riders, write_csv  # noqa: E402
import numpy as np  # noqa: E402


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
        print(f"\n=== real_data_schema_check: {self.passed}/{total} passed ===")
        for e in self.errors:
            print(" -", e)
        return 0 if self.failed == 0 else 1


def main():
    print("=== real_data_schema_check ===")
    ck = Checker()

    print("\n[1] 字段映射规模")
    ck.check("订单核心映射 ≥ 30", len(ORDER_CORE_MAP) >= 30, str(len(ORDER_CORE_MAP)))
    ck.check("骑士核心映射 ≥ 10", len(RIDER_CORE_MAP) >= 10, str(len(RIDER_CORE_MAP)))
    ck.check("KPI ≥ 10 且分层", len(KPI_SPEC) >= 10 and all("tier" in k for k in KPI_SPEC),
             str(len(KPI_SPEC)))
    ck.check("路演 KPI 子集存在", len(ROADSHOW_KPI_KEYS) >= 5 and
             all(any(k["key"] == x for k in KPI_SPEC) for x in ROADSHOW_KPI_KEYS),
             str(ROADSHOW_KPI_KEYS))
    ck.check("训练诊断 KPI 子集存在", len(TRAINING_DIAG_KPI_KEYS) >= 3)
    sm = summarize_schema()
    ck.check("schema 摘要可序列化", sm["order_core_mapped"] >= 30)

    print("\n[2] 样本生成与导入")
    tmp = ROOT / ".tmp" / "real_sample"
    write_csv(tmp / "sample_orders.csv", make_orders(40, seed=3))
    write_csv(tmp / "sample_riders.csv", make_riders(5, seed=3))
    cfg = load_real_episode_config(tmp / "sample_orders.csv", tmp / "sample_riders.csv",
                                   max_orders=30, max_riders=5)
    orders = cfg.get("custom_orders") or []
    ck.check("custom_orders 非空", len(orders) > 0, f"n={len(orders)}")
    ck.check("订单含 pickup/dropoff/ready/due",
             all(k in orders[0] for k in ("pickup", "dropoff", "ready_time", "due_date", "order_id")),
             str(list(orders[0])[:12]))
    ck.check("ready 非负", all(o["ready_time"] >= 0 for o in orders))
    ck.check("due > ready", all(o["due_date"] > o["ready_time"] for o in orders))
    ck.check("riders 注入 5 人", cfg.get("riders") and len(cfg["riders"]) == 5,
             str(list((cfg.get("riders") or {}).keys())))
    ck.check("数据源元信息", "data_source" in cfg and cfg["data_source"]["n_orders"] == len(orders))

    print("\n[3] 环境可跑真实样本")
    env = DeliveryEnv(cfg)
    obs, infos = env.reset(seed=0)
    a = env.agents[0]
    ck.check("obs 146", obs[a].shape == (146,), str(obs[a].shape))
    ck.check("5 agents", len(env.agents) == 5, str(env.agents))
    for _ in range(15):
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
    for key in ("completion_rate", "on_time_rate", "distance_per_order",
                "late_gt_5m_rate", "late_gt_15m_rate", "avg_tardiness"):
        ck.check(f"final_stats 含 {key}", key in st, str(sorted(st)[:20]))

    print("\n[4] 离线 KPI（样本标注）")
    cores = [extract_order_core(r) for r in load_rows(tmp / "sample_orders.csv")]
    cores = filter_training_orders(cores, completed_only=True)
    proj = GeoProjector.from_points(
        [(c["pickup_lng"], c["pickup_lat"]) for c in cores] +
        [(c["dropoff_lng"], c["dropoff_lat"]) for c in cores])
    env_orders = build_custom_orders(cores, proj, max_orders=40)
    off = offline_kpis_from_real(env_orders)
    ck.check("离线准时率 ∈ [0,1]", 0.0 <= off["offline_on_time_rate"] <= 1.0, str(off))
    ck.check("离线单均计提 > 0", off["avg_trade_amount"] > 0, str(off))

    print("\n[5] 骑手配置字段")
    riders = cfg["riders"]
    sample = next(iter(riders.values()))
    ck.check("capacity/speed/home", all(k in sample for k in ("capacity", "speed", "home")))
    ck.check("_real 标签保留", "_real" in sample and "vehicle_type" in sample["_real"])

    return ck.summary()


if __name__ == "__main__":
    raise SystemExit(main())
