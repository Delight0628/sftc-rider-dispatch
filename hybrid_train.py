"""hybrid_train.py — Hybrid 派单训练 CLI

文档约定：docs/hybrid_implementation.md §7
参数面继承原 ppo_marl_train.py（--models-dir/--logs-dir/--scenario/--real-orders 等），
新增 --episodes（max_iters）/ --seed / --num-parallel-workers / --init-linear。

使用：
  python hybrid_train.py --scenario delivery --episodes 200 --seed 0
  python hybrid_train.py --scenario delivery --real-orders 订单.xlsx --real-riders 骑手.xlsx
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

# TF 种子设置放在 Trainer 之前（防止 import 时即初始化 GPU 导致问题）
try:
    import tensorflow as tf
    _HAS_TF = True
except Exception:
    tf = None
    _HAS_TF = False


def _setup_dirs(models_dir: str | None, logs_dir: str | None) -> tuple:
    if models_dir is None:
        models_dir = os.path.join(os.getcwd(), "models", "hybrid")
    if logs_dir is None:
        logs_dir = os.path.join(os.getcwd(), "logs", "hybrid")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    return models_dir, logs_dir


def main():
    parser = argparse.ArgumentParser(description="Hybrid 派单训练入口")
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--logs-dir", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="delivery",
                        choices=["factory", "delivery"])
    parser.add_argument("--candidate-source", type=str, default="endogenous",
                        choices=["endogenous", "upstream"])
    parser.add_argument("--upstream-order-by", type=str, default="urgency",
                        choices=["urgency", "upstream"])
    parser.add_argument("--real-orders", type=str, default="",
                        help="真实订单样本路径（xlsx/csv/json）")
    parser.add_argument("--real-riders", type=str, default="",
                        help="真实骑手样本路径（可与订单同 xlsx 的骑手 sheet）")
    parser.add_argument("--episode-order-size", type=int, default=40,
                        help="每 episode 从真实订单池采样的订单数")
    parser.add_argument("--real-max-pool", type=int, default=2000)
    # Hybrid 新增参数
    parser.add_argument("--episodes", type=int, default=200,
                        help="训练迭代轮数（max_iters，每轮采集+更新）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-parallel-workers", type=int, default=2,
                        help="并行采集 worker 数（Windows 建议 2-4）")
    parser.add_argument("--init-linear", action="store_true", default=True,
                        help="Warm-start：末层 edge_feat 切片初始化为线性先验（默认开）")
    parser.add_argument("--no-init-linear", action="store_false", dest="init_linear",
                        help="关闭 warm-start，随机初始化")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="同 --episodes，兼容旧命名")
    cli_args, _ = parser.parse_known_args()

    scenario = cli_args.scenario.lower()
    candidate_source = cli_args.candidate_source.lower()
    upstream_order_by = cli_args.upstream_order_by.lower()

    if scenario != "delivery" and candidate_source == "upstream":
        print("⚠️ --candidate-source=upstream 仅对 delivery 生效，已回退 endogenous")
        candidate_source = "endogenous"
    if scenario != "delivery" and (cli_args.real_orders or cli_args.real_riders):
        print("⚠️ --real-orders/--real-riders 仅对 delivery 生效，已忽略")
        cli_args.real_orders = ""
        cli_args.real_riders = ""

    models_dir, logs_dir = _setup_dirs(cli_args.models_dir, cli_args.logs_dir)
    seed = int(cli_args.seed)
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TF:
        tf.random.set_seed(seed)

    # 环境配置
    env_config: dict = {
        "scenario": scenario,
        "candidate_source": candidate_source,
        "upstream_order_by": upstream_order_by,
    }
    if scenario == "delivery" and cli_args.real_orders:
        from environments.real_data_loader import load_real_training_pool
        pool_cfg = load_real_training_pool(
            cli_args.real_orders,
            rider_path=cli_args.real_riders or None,
            max_orders=int(cli_args.real_max_pool),
            max_riders=5,
        )
        env_config["real_order_pool"] = pool_cfg.get("order_pool") or []
        env_config["episode_order_size"] = int(cli_args.episode_order_size)
        if pool_cfg.get("riders"):
            env_config["riders"] = pool_cfg["riders"]
        if pool_cfg.get("geo_config"):
            env_config["geo_config"] = pool_cfg["geo_config"]
        env_config["data_source"] = pool_cfg.get("data_source")
        print(f"📦 已加载真实订单池: {len(env_config['real_order_pool'])} 单 | "
              f"骑手 {len(env_config.get('riders') or {})}")
    else:
        env_config["use_fixed_orders"] = True

    print("=" * 72)
    print(f"🚚 Hybrid 派单训练 | scenario={scenario} | seed={seed}")
    print(f"   models={models_dir} | logs={logs_dir}")
    print(f"   workers={cli_args.num_parallel_workers} | init_linear={cli_args.init_linear}")
    if cli_args.real_orders:
        print(f"   真实订单: {cli_args.real_orders}")
    print("=" * 72)

    if not _HAS_TF:
        print("❌ TensorFlow 未安装，无法启动训练。请安装 TF 2.15：")
        print("   pip install tensorflow==2.15.0")
        sys.exit(1)

    from environments.delivery_config import HYBRID_TRAINING_CONFIG
    from hybrid.trainer import HybridTrainer

    trainer = HybridTrainer(
        env_config=env_config,
        cfg=HYBRID_TRAINING_CONFIG,
        models_dir=models_dir,
        logs_dir=logs_dir,
        seed=seed,
        num_parallel_workers=cli_args.num_parallel_workers,
        init_linear=cli_args.init_linear,
    )

    max_iters = cli_args.max_iters if cli_args.max_iters is not None else cli_args.episodes
    results = trainer.train(max_iters=max_iters)

    # 训练结束保存 last（供续训）
    trainer._save_ckpt("last")
    print(f"💾 最终模型已保存到 {models_dir}")
    if results:
        print(f"\n🎉 训练完成 | best_score={results['best_score']:.4f} "
              f"@ iter {results['best_iter']} | elapsed={results['elapsed']:.1f}s")


if __name__ == "__main__":
    main()
