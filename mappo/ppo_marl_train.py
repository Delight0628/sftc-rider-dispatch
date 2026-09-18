"""
MAPPO多智能体强化学习训练入口
=================================
基于MAPPO算法的工厂调度系统训练入口

模块组织（自底向上）：
┌─────────────────────────────────────┐
│  ppo_marl_train.py (本文件)         │  ← 训练入口
├─────────────────────────────────────┤
│  ppo_trainer.py                     │  ← 训练器主类
├──────────┬──────────┬───────────────┤
│ ppo_     │ ppo_     │ ppo_          │
│ buffer.py│ network  │ worker.py     │  ← 核心组件
└──────────┴──────────┴───────────────┘

使用方式：
    python mappo/ppo_marl_train.py [--models-dir DIR] [--logs-dir DIR]
"""

import os
# 训练模式默认使用随机初始化，提高探索能力
os.environ.setdefault('DETERMINISTIC_INIT', '0')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 🔧 强制worker子进程使用CPU，避免多进程GPU资源竞争导致BrokenProcessPool
# 主进程（训练器）仍使用GPU进行模型更新，子进程（采样）用CPU
os.environ['FORCE_WORKER_CPU'] = '1'

import sys
import random
import numpy as np
import tensorflow as tf
import argparse
import multiprocessing

# 添加环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from environments.w_factory_config import *
from mappo.ppo_trainer import SimplePPOTrainer


def main():
    """
    训练主入口
    
    执行流程：
    1. 解析命令行参数（模型/日志目录）
    2. 设置随机种子
    3. 加载训练配置
    4. 创建训练器实例
    5. 启动自适应训练循环
    6. 输出训练结果
    """
    print(f"✨ 训练进程PID: {os.getpid()}")

    # 解析外部传入的目录参数（由 auto_train.py 传入）
    parser = argparse.ArgumentParser(description="MAPPO 训练入口")
    parser.add_argument("--models-dir", type=str, default=None, help="用于保存训练模型的根目录（由auto_train传入）")
    parser.add_argument("--logs-dir", type=str, default=None, help="用于保存TensorBoard日志的根目录（由auto_train传入）")
    parser.add_argument("--scenario", type=str, default="factory", choices=["factory", "delivery"],
                        help="训练场景：factory=工厂生产调度，delivery=运力商圈配送调度（比赛）")
    parser.add_argument("--candidate-source", type=str, default="endogenous",
                        choices=["endogenous", "upstream"],
                        help="候选订单来源：endogenous=环境自采样；upstream=采用上游（双塔+W&D）精排候选"
                             "（未注入真实上游时自动使用 mock 精排，用于三层联调）")
    parser.add_argument("--upstream-order-by", type=str, default="urgency",
                        choices=["urgency", "upstream"],
                        help="上游候选用途口径：urgency=按时间紧迫性排序（会议口径，默认）；"
                             "upstream=保持上游意愿序")
    parser.add_argument("--real-orders", type=str, default="",
                        help="真实订单样本路径（xlsx/csv/json，如果洛 dwd 导出）；"
                             "设置后 episode 从该订单池滑动窗口采样")
    parser.add_argument("--real-riders", type=str, default="",
                        help="真实骑手样本路径（可与订单同一 xlsx 的骑手 sheet）")
    parser.add_argument("--episode-order-size", type=int, default=80,
                        help="每个训练 episode 从真实订单池采样的订单数（默认 80）")
    parser.add_argument("--real-max-pool", type=int, default=2000,
                        help="加载真实订单池时的最大订单数（完成单过滤后）")
    cli_args, _ = parser.parse_known_args()
    scenario = cli_args.scenario.lower()
    candidate_source = cli_args.candidate_source.lower()
    upstream_order_by = cli_args.upstream_order_by.lower()
    if scenario != "delivery" and candidate_source == "upstream":
        print("⚠️ --candidate-source=upstream 仅对 delivery 场景生效，已回退 endogenous")
        candidate_source = "endogenous"
    if scenario != "delivery" and (cli_args.real_orders or cli_args.real_riders):
        print("⚠️ --real-orders/--real-riders 仅对 delivery 场景生效，已忽略")
        cli_args.real_orders = ""
        cli_args.real_riders = ""

    # 场景配置选择（delivery 使用运力商圈配送配置）
    if scenario == "delivery":
        from environments.delivery_config import (
            DELIVERY_TRAINING_FLOW_CONFIG, DELIVERY_REWARD_CONFIG, RIDERS
        )
        flow_config = DELIVERY_TRAINING_FLOW_CONFIG
        reward_config_for_print = DELIVERY_REWARD_CONFIG
    else:
        flow_config = TRAINING_FLOW_CONFIG
        reward_config_for_print = REWARD_CONFIG

    # 设置随机种子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    try:
        # 从配置文件获取训练参数
        max_episodes = flow_config["general_params"]["max_episodes"]
        steps_per_episode = flow_config["general_params"]["steps_per_episode"]
        eval_frequency = flow_config["general_params"]["eval_frequency"]

        print("=" * 80)
        print(f"🚚 场景: {scenario}" + ("（运力评估商圈智能体-比赛）" if scenario == "delivery" else "（W工厂生产调度）"))
        foundation_criteria = flow_config["foundation_phase"]["graduation_criteria"]
        generalization_criteria = flow_config["generalization_phase"]["completion_criteria"]
        foundation_mixing = flow_config["foundation_phase"]["multi_task_mixing"]
        generalization_mixing = flow_config["generalization_phase"]["multi_task_mixing"]
        dynamic_events = flow_config["generalization_phase"].get("dynamic_events", {})

        print(f"\n📚阶段一：基础能力训练（随机订单泛化）")
        print(f"   策略: 随机订单 + {int(foundation_mixing.get('base_worker_fraction', 0)*100)}% worker使用基础订单锚点")
        print(f"   目标: 综合评分 > {foundation_criteria['target_score']:.2f}, "
              f"完成率 > {foundation_criteria['min_completion_rate']:.0f}%, "
              f"延期 < {foundation_criteria['tardiness_threshold']:.0f}min, "
              f"连续{foundation_criteria['target_consistency']}次")

        print(f"\n🚀阶段二：动态事件鲁棒性训练（动态事件鲁棒性）")
        if scenario == "delivery":
            print(f"   策略: 随机订单 + 动态事件（骑手离线{'✓' if dynamic_events.get('rider_offline_enabled') else '✗'}、紧急订单{'✓' if dynamic_events.get('emergency_orders_enabled') else '✗'}）")
        else:
            print(f"   策略: 随机订单 + 动态事件（设备故障{'✓' if dynamic_events.get('equipment_failure_enabled') else '✗'}、紧急插单{'✓' if dynamic_events.get('emergency_orders_enabled') else '✗'}）")
        print(f"        + {int(generalization_mixing.get('base_worker_fraction', 0)*100)}% worker使用基础订单锚点")
        print(f"   目标: 综合评分 > {generalization_criteria['target_score']:.2f}, "
              f"完成率 > {generalization_criteria['min_completion_rate']:.0f}%, "
              f"连续{generalization_criteria['target_consistency']}次")

        print(f"📊 轮数上限: {max_episodes}轮")
        print("=" * 80)
        print("🔧 核心配置:")
        if scenario == "delivery":
            print("  骑手:")
            for rider, config in RIDERS.items():
                print(f"    - {rider}: 携带上限={config['capacity']}, 速度={config['speed']}km/min")
        else:
            print("  工作站:")
            for station, config in WORKSTATIONS.items():
                print(f"    - {station}: 数量={config['count']}, 容量={config['capacity']}")

        print("  奖励系统:")
        for key, value in reward_config_for_print.items():
            print(f"    - {key}: {value}")

        cl_config = flow_config["foundation_phase"].get("curriculum_learning", {"enabled": False})
        dynamic_events_cfg = flow_config["generalization_phase"].get("dynamic_events", {})

        print("  启用/禁用模块:")
        print(f"    - 课程学习: {'启用' if cl_config.get('enabled', False) else '禁用'}")
        if scenario == "delivery":
            print(f"    - 骑手离线: {'启用' if dynamic_events_cfg.get('rider_offline_enabled', False) else '禁用'}")
        else:
            print(f"    - 设备故障: {'启用' if dynamic_events_cfg.get('equipment_failure_enabled', False) else '禁用'}")
        print(f"    - 紧急插单: {'启用' if dynamic_events_cfg.get('emergency_orders_enabled', False) else '禁用'}")
        if scenario == "delivery":
            if candidate_source == "upstream":
                print(f"    - 候选来源: upstream（上游双塔+W&D 精排候选；未注入真实上游时用 mock 精排）")
            else:
                print(f"    - 候选来源: endogenous（环境自采样 EDD5+最近3+随机2）")
            print(f"    - 候选用途口径: {upstream_order_by}"
                  + ("（时间紧迫性优先，会议口径）" if upstream_order_by == "urgency" else "（保持上游意愿序）"))
            if cli_args.real_orders:
                print(f"    - 真实订单池: {cli_args.real_orders}")
                print(f"    - 骑手样本: {cli_args.real_riders or '（默认 RIDERS）'}")
                print(f"    - 每回合采样: {cli_args.episode_order_size} 单（池上限 {cli_args.real_max_pool}）")
        print("-" * 40)

        env_config = {
            'scenario': scenario,
            'candidate_source': candidate_source,
            'upstream_order_by': upstream_order_by,
        }
        if scenario == "delivery" and cli_args.real_orders:
            from environments.real_data_loader import load_real_training_pool
            pool_cfg = load_real_training_pool(
                cli_args.real_orders,
                rider_path=cli_args.real_riders or None,
                max_orders=int(cli_args.real_max_pool),
                max_riders=5,
            )
            env_config['real_order_pool'] = pool_cfg.get('order_pool') or []
            env_config['episode_order_size'] = int(cli_args.episode_order_size)
            if pool_cfg.get('riders'):
                env_config['riders'] = pool_cfg['riders']
            if pool_cfg.get('geo_config'):
                env_config['geo_config'] = pool_cfg['geo_config']
            env_config['data_source'] = pool_cfg.get('data_source')
            print(f"📦 已加载真实订单池: {len(env_config['real_order_pool'])} 单 | "
                  f"骑手 {len(env_config.get('riders') or {})} | "
                  f"网格 {env_config.get('geo_config')}")

        trainer = SimplePPOTrainer(
            initial_lr=LEARNING_RATE_CONFIG["initial_lr"],
            total_train_episodes=max_episodes,
            steps_per_episode=steps_per_episode,
            training_targets=None,
            models_root_dir=cli_args.models_dir,
            logs_root_dir=cli_args.logs_dir,
            env_config=env_config,
        )
        
        # 启动自适应训练：系统将根据性能自动决定何时停止
        results = trainer.train(
            max_episodes=max_episodes,
            steps_per_episode=steps_per_episode,
            eval_frequency=eval_frequency,
            adaptive_mode=True
        )
        
        if results:
            print("\n🎉 自适应训练成功完成！")
            print(f"📊 实际训练轮数: {len(trainer.iteration_times)}")
            if scenario == "delivery":
                total_target = trainer._get_target_parts(None)
                unit_label = "单"
            else:
                total_target = get_total_parts_count()
                unit_label = "个零件"
            final_completion_rate = (results['best_kpi'].get('mean_completed_parts', 0) / total_target) * 100 if total_target > 0 else 0
            print(f"🎯 最终目标达成: {trainer.adaptive_state['target_achieved_count']}次连续达标 (基于最终阶段分数)")

            best_episode_final = trainer.best_episode_dual_objective if trainer.best_episode_dual_objective != -1 else trainer.final_stage_best_episode
            print(f"📈 历史最佳性能 (双重标准，第 {best_episode_final} 回合): {final_completion_rate:.1f}% ({results['best_kpi'].get('mean_completed_parts', 0):.1f}{unit_label})")
        else:
            print("\n❌ 训练失败")
            
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置多进程启动方法为'spawn'，避免TensorFlow的fork不安全问题
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()

