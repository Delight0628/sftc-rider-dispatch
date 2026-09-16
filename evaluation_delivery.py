"""
配送场景启发式基线评估
======================
在 evaluation.py（工厂专用）已移除后，为 delivery 场景提供可对比的规则基线：

- idle      ：有单就接（最近/最紧迫候选的第 1 个），否则 IDLE
- fifo      ：优先接 ready_time 最早 / order_id 最小的合法候选
- nearest   ：优先接 to_pickup_time 最短的合法候选
- edd       ：优先接 slack 最小（最紧迫）的合法候选
- random    ：合法动作均匀随机（对照）

用法：
  python evaluation_delivery.py --baseline edd --episodes 5
  python evaluation_delivery.py --baseline all --episodes 3 --candidate-source upstream
  python evaluation_delivery.py --model path/to/weights.h5   # 需 TF，可选

输出：KPI 汇总（完成率/准时率/tardiness/makespan/利用率/评分）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.delivery_env import DeliveryEnv  # noqa: E402
from environments.delivery_config import (  # noqa: E402
    calculate_delivery_episode_score,
    generate_random_delivery_orders,
    generate_mock_upstream_candidates,
)

RIDER_NAMES = ["骑手A", "骑手B", "骑手C", "骑手D", "骑手E"]


def _pick_by_baseline(infos_agent: dict, baseline: str, rng: np.random.RandomState) -> int:
    cands = infos_agent.get("candidates_map") or []
    mask = infos_agent.get("action_mask")
    if mask is not None:
        legal = [int(a) for a in np.where(mask)[0]]
    else:
        legal = [0] + [c["action"] for c in cands]
    if not legal:
        return 0
    if 0 not in legal and baseline == "idle":
        return int(legal[0])
    if not cands:
        return 0 if 0 in legal else int(legal[0])

    # 仅在 mask 允许的候选中选择
    usable = [c for c in cands if int(c["action"]) in legal and int(c["action"]) != 0]
    if not usable:
        return 0 if 0 in legal else int(legal[0])

    if baseline == "random":
        return int(legal[int(rng.randint(0, len(legal)))])

    if baseline == "fifo":
        usable.sort(key=lambda c: (c.get("ready_time", 0.0), c.get("part_id", 0)))
    elif baseline == "nearest":
        usable.sort(key=lambda c: (c.get("to_pickup_time", 1e9), c.get("part_id", 0)))
    elif baseline == "edd":
        usable.sort(key=lambda c: (c.get("slack", 1e9), c.get("part_id", 0)))
    elif baseline == "idle":
        # 有单就接：取第 1 个可用候选（候选本身已按 urgency 或上游序排列）
        pass
    else:
        raise ValueError(f"unknown baseline: {baseline}")
    return int(usable[0]["action"])


def run_episode(env: DeliveryEnv, baseline: str, rng: np.random.RandomState,
                max_steps: int = 800) -> Dict:
    obs, infos = env.reset()
    steps = 0
    for _ in range(max_steps):
        actions = {}
        for agent in env.agents:
            actions[agent] = _pick_by_baseline(env.infos[agent], baseline, rng)
        obs, rewards, terms, truncs, infos = env.step(actions)
        steps += 1
        if all(terms.values()) or all(truncs.values()):
            break
    stats = env.sim.get_final_stats()
    stats["steps"] = steps
    stats["episode_score"] = float(calculate_delivery_episode_score(stats, config=env.config))
    return stats


def evaluate_baseline(baseline: str, episodes: int = 5, seed: int = 0,
                      candidate_source: str = "endogenous",
                      upstream_order_by: str = "urgency",
                      use_fixed_orders: bool = True) -> Dict:
    rng = np.random.RandomState(seed)
    rows: List[Dict] = []
    for ep in range(episodes):
        ep_seed = seed + ep
        np.random.seed(ep_seed)
        if use_fixed_orders:
            orders = generate_random_delivery_orders()
            for i, o in enumerate(orders):
                o.setdefault("order_id", i)
            upstream = None
            if candidate_source == "upstream":
                upstream = generate_mock_upstream_candidates(
                    orders, RIDER_NAMES, top_k=10, seed=ep_seed)
            cfg = {
                "scenario": "delivery",
                "training_mode": True,
                "custom_orders": orders,
                "candidate_source": candidate_source,
                "upstream_order_by": upstream_order_by,
            }
            if upstream is not None:
                cfg["upstream_candidates"] = upstream
        else:
            cfg = {
                "scenario": "delivery",
                "training_mode": True,
                "candidate_source": candidate_source,
                "upstream_order_by": upstream_order_by,
            }
        env = DeliveryEnv(cfg)
        env.reset(seed=ep_seed)
        stats = run_episode(env, baseline, rng)
        rows.append(stats)

    def _mean(key: str) -> float:
        vals = [float(r.get(key, 0) or 0) for r in rows]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "baseline": baseline,
        "episodes": episodes,
        "candidate_source": candidate_source,
        "upstream_order_by": upstream_order_by,
        "completion_rate": _mean("completion_rate"),
        "on_time_rate": _mean("on_time_rate"),
        "total_parts": _mean("total_parts"),
        "total_orders": _mean("total_orders"),
        "total_tardiness": _mean("total_tardiness"),
        "makespan": _mean("makespan"),
        "mean_utilization": _mean("mean_utilization"),
        "episode_score": _mean("episode_score"),
        "upstream_matched_count": _mean("upstream_matched_count"),
        "upstream_fallback_count": _mean("upstream_fallback_count"),
    }
    return summary


def try_model_baseline(model_path: str, episodes: int = 3, seed: int = 0) -> Optional[Dict]:
    """可选：加载 TF 模型做贪心评估。环境无 TF 时返回 None。"""
    try:
        import tensorflow as tf  # noqa: F401
    except Exception:
        print("ℹ️ 未安装 TensorFlow，跳过模型评估")
        return None
    print(f"⚠️ 模型评估入口已预留：{model_path}（需接入 ppo_network 加载逻辑）")
    return None


def main():
    parser = argparse.ArgumentParser(description="配送场景启发式基线评估")
    parser.add_argument("--baseline", type=str, default="all",
                        choices=["all", "idle", "fifo", "nearest", "edd", "random"],
                        help="基线策略")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-source", type=str, default="endogenous",
                        choices=["endogenous", "upstream"])
    parser.add_argument("--upstream-order-by", type=str, default="urgency",
                        choices=["urgency", "upstream"])
    parser.add_argument("--json-out", type=str, default="",
                        help="可选：汇总结果写入 JSON 文件")
    parser.add_argument("--model", type=str, default="",
                        help="可选：模型权重路径（需 TF）")
    args = parser.parse_args()

    baselines = ["idle", "fifo", "nearest", "edd", "random"] if args.baseline == "all" else [args.baseline]
    results = []
    print("=" * 72)
    print(f"配送启发式基线 | source={args.candidate_source} order_by={args.upstream_order_by} "
          f"episodes={args.episodes}")
    print("=" * 72)
    for b in baselines:
        print(f"\n>>> baseline={b}")
        s = evaluate_baseline(
            b, episodes=args.episodes, seed=args.seed,
            candidate_source=args.candidate_source,
            upstream_order_by=args.upstream_order_by,
        )
        results.append(s)
        print(f"  completion={s['completion_rate']:.3f}  on_time={s['on_time_rate']:.3f}  "
              f"tardiness={s['total_tardiness']:.1f}  makespan={s['makespan']:.1f}  "
              f"util={s['mean_utilization']:.3f}  score={s['episode_score']:.3f}")

    if args.model:
        try_model_baseline(args.model, episodes=args.episodes, seed=args.seed)

    print("\n" + "=" * 72)
    print(f"{'baseline':10s} {'compl':>7s} {'ontime':>7s} {'tardy':>8s} {'span':>8s} {'util':>7s} {'score':>7s}")
    for s in results:
        print(f"{s['baseline']:10s} {s['completion_rate']:7.3f} {s['on_time_rate']:7.3f} "
              f"{s['total_tardiness']:8.1f} {s['makespan']:8.1f} {s['mean_utilization']:7.3f} "
              f"{s['episode_score']:7.3f}")
    print("=" * 72)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"已写入 {args.json_out}")


if __name__ == "__main__":
    main()
