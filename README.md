<div align="center">

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🛵  骑手派单调度 · 多智能体强化学习系统  🤖                 ║
║                                                               ║
║     双塔召回 → Wide&Deep 精排 → MAPPO 全局调度               ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

# Rider Dispatch MAPPO

**基于多智能体强化学习（MAPPO）的即时配送骑手派单调度系统**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-1.24+-4CAF50?style=for-the-badge&logo=python&logoColor=white)](https://pettingzoo.farama.org/)
[![SimPy](https://img.shields.io/badge/SimPy-4.0+-E91E63?style=for-the-badge)](https://simpy.readthedocs.io/)

---

🛵 **最小化送达时间** · ⏰ **最小化超时率** · 📈 **最大化骑手利用率** · 🛡️ **动态鲁棒性**

</div>

---

## 项目概述

本项目是三层级联调度架构中的 **MAPPO 调度层**，负责在时间约束下进行全局骑手派单决策。

### 三层架构

| 层级 | 模型 | 职责 |
|------|------|------|
| 召回 | 双塔模型 | 骑手-订单粗排，输出候选集 |
| 精排 | Wide & Deep | 接单意愿精排，输出 Top-K 候选 |
| **调度** | **MAPPO（本仓库）** | **时间紧迫性优先的全局派单** |

### 核心设计

- **场景映射**：骑手=智能体，订单=工件，配送路线=动态两段工艺（取餐→送餐）
- **观测空间**：146 维层次化状态（骑手状态 + 全局宏观 + 队列摘要 + 候选特征 + 未来订单）
- **动作空间**：MultiDiscrete，每个骑手从候选池选单或 IDLE
- **奖励体系**：送达奖励 + 准时奖励 + 超时 Huber 惩罚 + 终局分数 bonus
- **上游接入**：支持上游（双塔+W&D）精排候选注入，意愿分编码在观测特征中

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配送场景训练（mock 上游候选）
python mappo/ppo_marl_train.py --scenario delivery --candidate-source upstream

# 对比基线（环境自采样候选）
python mappo/ppo_marl_train.py --scenario delivery --candidate-source endogenous

# 自动化训练
python auto_train.py "<实验名>" --scenario delivery --candidate-source upstream
```

## 项目结构

```
rider-dispatch-mappo/
├── environments/
│   ├── delivery_config.py      # 配送配置：骑手/地理/订单生成/奖励/上游接口
│   ├── delivery_env.py         # 配送仿真环境：DeliverySim + DeliveryEnv
│   ├── w_factory_env.py        # 环境工厂（场景分发入口）
│   └── w_factory_config.py     # 共享基础配置
├── mappo/
│   ├── ppo_marl_train.py       # 训练入口（支持 --scenario delivery）
│   ├── ppo_trainer.py          # MAPPO 训练器
│   ├── ppo_network.py          # Actor-Critic 网络
│   ├── ppo_worker.py           # 并行采样 Worker
│   ├── ppo_buffer.py           # 经验缓冲
│   └── sampling_utils.py       # 候选采样工具
├── auto_train.py               # 自动化训练流水线
├── update.md                   # 改造日志与进度记录
└── requirements.txt
```

## 上游候选数据契约

```python
# 上游（双塔+W&D）注入格式
config['upstream_candidates'] = {
    "骑手A": [
        {"order_id": 12, "willingness": 0.93},
        {"order_id": 7,  "willingness": 0.81},
    ],
    # 每骑手最多 top_k（默认 10）条
}
```

- 意愿分编码在候选特征 `[1]` 位，146 维观测布局不变
- 排序口径：`urgency`（时间紧迫性优先，默认）/ `upstream`（保持上游意愿序）
- 容错：未覆盖骑手回退自采样；候选不足用最紧迫订单补齐

## 验证

```bash
# 环境逻辑回归（纯环境层，无需 TF）
python .tmp/delivery_logic_check.py

# 三层联调接口校验
python .tmp/delivery_upstream_check.py

# 场景分发集成校验
python .tmp/dispatch_integration_check.py
```

## License

MIT
