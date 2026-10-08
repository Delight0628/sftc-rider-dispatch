<div align="center">

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🛵  骑手派单调度 · Hybrid 学习打分 + 约束匹配  🤖           ║
║                                                               ║
║     双塔召回 → Wide&Deep 精排 → Hybrid 全局调度              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

# Rider Dispatch Hybrid

**基于「学习边效用 + 约束二分图匹配 + 滚动重优化」的即时配送骑手派单调度系统**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-1.24+-4CAF50?style=for-the-badge&logo=python&logoColor=white)](https://pettingzoo.farama.org/)
[![SimPy](https://img.shields.io/badge/SimPy-4.0+-E91E63?style=for-the-badge)](https://simpy.readthedocs.io/)

---

🛵 **最小化送达时间** · ⏰ **最小化超时率** · 📈 **最大化骑手利用率** · 🛡️ **动态鲁棒性**

</div>

---

## 项目概述

本项目是三层级联调度架构中的 **Hybrid 调度层**，负责在时间约束下进行全局骑手派单决策。

### 三层架构

| 层级 | 模型 | 职责 |
|------|------|------|
| 召回 | 双塔模型 | 骑手-订单粗排，输出候选集 |
| 精排 | Wide & Deep | 接单意愿精排，输出 Top-K 候选 |
| **调度** | **Hybrid（本仓库）** | **时间紧迫性优先的全局派单** |

### Hybrid 核心设计

- **范式**：学习边效用 `q(骑手, 订单)` + 约束二分图匹配 + 滚动重优化
  - 与 MAPPO/CTDE 对比：集中决策替代分散策略，匹配层保证 feasibility，边级信用替代全局 reward 均摊
  - 理论依据：`docs/research_dispatch_algorithm_survey.md` §5；工业验证：DiDi KDD 2018/2019/2022、Meituan SCDN
- **编码**：集合编码（DeepSets + Cross-Attention），N/K 可变，零 one-hot 宽度约束
- **学习**：off-policy n-step TD（fitted-Q 式），γ=0.99，n=5，目标网络软更新
- **匹配**：贪心 + 交换改进的加权二分图匹配，带容量 b-matching（同骑手多单 FIFO）
- **状态**：rider_set `[N,8]` + cand_set `[N,K,12]` + global `[5]` + edge_feat `[N,K,12]`
- **动作**：匹配结果映射为候选下标动作（经 `env.step` 落地）
- **探索**：温度 τ=0.5 softmax 边采样 + 贪心匹配
- **上游接入**：支持上游（双塔+W&D）精排候选注入；意愿分不进策略，仅作候选元信息

> **历史**：原 MAPPO/CTDE 栈已归档至 `archive/mappo/`（冻结对照）；2026-10-08 起 Hybrid 为唯一训练框架。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# Hybrid 训练（mock 联调）
python hybrid_train.py --scenario delivery --episodes 50

# Hybrid 训练（真实果洛样本）
python hybrid_train.py \
  --real-orders "骑手派单仿真样本_果洛藏族自治州_20260915_全量.xlsx" \
  --real-riders "骑手派单仿真样本_果洛藏族自治州_20260915_全量.xlsx" \
  --episode-order-size 40 \
  --episodes 200

# 对拍评估
python evaluation_delivery.py --baseline all --episodes 5
```

## 项目结构

```
rider-dispatch-mappo/
├── environments/
│   ├── delivery_config.py      # 配送配置：骑手/地理/订单/奖励/评分/HYBRID_*_CONFIG
│   ├── delivery_env.py         # 配送仿真环境：DeliverySim + DeliveryEnv
│   ├── hybrid_dispatch.py      # 边打分 + 约束匹配 + 滚动重优化（推理主干，零 TF）
│   ├── set_obs.py              # 集合观测导出（含 build_all_pairs_set_obs）
│   ├── real_data_schema.py     # 真实数仓字段映射 + 调度 KPI 筛选
│   ├── real_data_loader.py     # 订单/骑士样本 → custom_orders
│   ├── w_factory_env.py        # 环境工厂（场景分发入口）
│   └── w_factory_config.py     # 共享基础配置
├── hybrid/
│   ├── set_encoder.py          # 集合编码算子（TF + numpy 双实现）
│   ├── edge_value_net.py       # 边效用网络（TF/Keras + npz 导出降级）
│   ├── scorers.py              # Linear / Neural EdgeScorer（统一接口）
│   ├── replay_buffer.py        # off-policy 经验回放（边级信用回填）
│   ├── collect.py              # episode 采集（温度采样行为策略，可并行）
│   ├── trainer.py              # n-step TD 训练循环 + early-stop 回滚
│   └── evaluate.py             # 对拍评估（neural/linear/启发式同口径）
├── hybrid_train.py             # 训练入口 CLI
├── archive/mappo/              # MAPPO/CTDE 历史栈（冻结，仅作对照）
├── evaluation_delivery.py      # 基线对拍（--baseline hybrid --hybrid-scorer …）
├── docs/
│   ├── research_dispatch_algorithm_survey.md   # 算法调研与路线对比
│   ├── hybrid_implementation.md                # 本框架实现文档
│   └── plan_scheme_b_set_encoding.md           # 方案 B 集合编码设计
├── supervisor/                 # 监督器、备份、启动脚本
├── runs/                       # 训练产物（模型 + 日志 + metrics.jsonl）
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

- 意愿分不进观测/策略：仅存在于 `candidates_map[i]['willingness']`；候选特征 `[1]` 位为 `due_rel`（紧迫维度）
- 排序口径：`urgency`（时间紧迫性优先，默认）/ `upstream`（保持上游意愿序）
- 容错：未覆盖骑手回退自采样；候选不足用最紧迫订单补齐

## 验证

```bash
# 纯环境层回归（无需 TensorFlow）
python checks/delivery_logic_check.py
python checks/delivery_upstream_check.py
python checks/dispatch_integration_check.py

# Hybrid 基础校验（含 scorer parity / 对拍 / TF 降级）
python checks/hybrid_dispatch_check.py
python checks/check_hybrid_training.py

# 启发式基线对比（idle/fifo/nearest/edd/random/hybrid）
python evaluation_delivery.py --baseline all --episodes 5

# 真实样本（业务方导出或本地同构样本）
python evaluation_delivery.py --baseline all --episodes 5 \
  --real-orders data/real_incoming/orders_guoluo_20260915_ready.csv \
  --real-riders data/real_incoming/riders_guoluo_20260915_full.csv
```

## License

MIT
