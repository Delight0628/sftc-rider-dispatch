# update.md — 运力商圈智能体比赛改造日志

> 本文件是本次比赛代码改造的唯一进度文档，**每次对话必须维护更新**。
> 项目路径：`D:\rider-dispatch-mappo`（原工业车间调度 MAPPO → 外卖骑手派单调度）

---

## 1. 背景与比赛信息（来源：2026-09-14 听记「双塔模型与MAPPO调度算法讨论」）

- **参赛项目**：运力评估商圈智能体（可包装成智能体：外接 LLM + 提示词，把模型当可调度的工具）
- **三层级联架构**（三个模型互补，已确立为技术路线）：
  1. **双塔模型**（召回/粗排）：骑手塔 + 订单塔，分类/连续特征 → Embedding → MLP → 向量，内积/Cosine 相似度；正样本=接单，负样本=曝光未接；训练出带权重模型文件（PyTorch）
  2. **Wide & Deep**（精排）：对召回的 ~100 单做精细排序，输出接单意愿得分；解决双塔特征交叉不足的问题
  3. **MAPPO**（调度层，**本仓库负责**）：多智能体近端策略优化，解决时间约束下的全局派单调度
- **MAPPO 场景映射**（听记原文要点）：
  - 骑手 = 机器（agent），订单 = 工件
  - 原场景：5 台机器、每个工件走的机器数固定、流程标准化
  - 新场景：**每个骑手对应的单量和地址不相同**（"两边都变成动的"），订单有取餐点/送餐点/截止时间（交期），骑手状态（位置/负载）实时变动
  - 调度语义：原来"先做 A 还是先做 B/C" → 现在"先把单派给骑士 A 还是骑士 B"
  - 时间紧迫性优先于意愿排序（例：12:30 出发 2 点前送完 5 单，先送最紧迫的）
  - 上游漏斗：双塔召回 ~100 → W&D 精排出「最想接的 10 个」→ MAPPO 决定时间关系与派单
- **数据策略**：强化学习不挑数据真实性 → **先用随机模拟数据跑通逻辑**（分布合理即可），后续接真实数据微调
- **数据字段需求**（业务方提供）：骑士 ID、订单链（取餐点/送餐点）、预期/实际送达时间、订单重量/体积、用户特征、企业渠道等
- **算力**：当前无 GPU；堡垒机纯 CPU 效率低；短期 CPU 小规模验证，需商定租用算力云平台
- **关键时间节点**：
  - **2026-10-14 前提交参赛模板文档**（会议纪要可直接转）
  - **2026 年 11 月中旬决赛线下路演 + Demo 展示**
  - 每周四进度同步会（算法改造进展 + 数据准备情况）

## 2. 会议要求对照（第 4 次对话复核结论）

| 会议要求（听记原意） | 现状 | 证据 |
|---|---|---|
| 骑手=机器、订单=工件 | ✅ 已满足 | `delivery_env.py` 场景映射 |
| 两边都变成动的（订单路线/骑手单量动态） | ✅ 已满足 | 坐标行程估算 + carry/capacity/projected_free |
| 先派给骑士 A 还是 B（多智能体决策） | ✅ 已满足 | 5 agent 共享待派池抢单 |
| 时间紧迫性优先（12:30→2点先送最紧迫） | ✅ 已满足 | slack/time_pressure + Huber；upstream 默认 `urgency` |
| 上游取出 10 个最想接的单，MAPPO 管时间关系 | ✅ 已满足 | `candidate_source=upstream` + 候选特征[1]=意愿分 |
| 「可能多加一层向量」 | ⚠️ 工程取舍 | **不扩网络**，复用 146 维布局，意愿分占原 `remaining_ops`（恒 2 无信息）槽位；网络/BC 教师零改动 |
| 模拟随机数据先跑通 | ✅ 已满足 | `generate_random_delivery_orders` |
| 输出带权重模型 + 派单序列 | ⚠️ 部分 | 环境可训；完整收敛待算力；可视化/甘特图未做 |
| 启发式基线对比 | ✅ 第 4 次补全 | `evaluation_delivery.py`（idle/fifo/nearest/edd/random） |
| 智能体包装（LLM+tool+提示词） | ⏳ 未开始 | 决赛 Demo 需要 |
| 10-14 模板 / 11 月路演 Demo | ⏳ 未开始 | |
| 完整训练收敛验证 | ⏳ 阻塞于算力 | |

**总判**：MAPPO **环境层与三层联调接口已满足会议改造要求**；算法收敛、智能体包装、模板文档仍是赛程缺口。

## 3. 原项目结构与改造后仓库

### 3.1 仍保留的工厂共享面（网络/训练栈复用依赖）

| 模块 | 文件 | 说明 |
|---|---|---|
| 环境工厂 | `environments/w_factory_env.py` | `make_parallel_env` 按 `scenario` 分发 factory/delivery |
| 共享配置 | `environments/w_factory_config.py` | PPO 网络配置、评分、评估配置等 |
| MAPPO | `mappo/ppo_network.py` 等 | Actor/Critic、Trainer、Worker 零逻辑分叉复用 |

### 3.2 已删除（commit `910895f`，仓库聚焦骑手派单）

`evaluation.py`、`debug_marl_behavior.py`、`app/`、`plotting.py`、`log_parser.py` 等工厂专用演示/评估文件已移除。

### 3.3 配送场景核心

| 文件 | 说明 |
|---|---|
| `environments/delivery_config.py` | 骑手/地理/订单生成/奖励/评分 + 上游契约与 mock |
| `environments/delivery_env.py` | DeliverySim + DeliveryEnv（事件驱动，146 维观测） |
| `evaluation_delivery.py` | **第 4 次新增**：配送启发式基线评估 |
| `update.md` / `README.md` | 进度与说明 |

## 4. 改造设计（骑手派单调度）

### 4.1 场景映射

| 工业 | 运力配送 | 实现要点 |
|---|---|---|
| 工件 Part | 配送订单 | 取餐点/送餐点坐标、ready_time、due_date、优先级、品类 |
| 机器/工作站 | 骑手（agent） | 位置、capacity、速度、繁忙率、离线 |
| 工艺路线（固定5站） | 订单路线（动态2段） | 去取餐→取餐→去送餐→送达，按坐标/速度计时 |
| 站点队列 | 待派单池（共享） | endogenous：EDD5+最近3+随机2；upstream：上游集合 |
| 交期 due_date | 承诺送达时间 | 奖励/time_pressure/slack |
| 设备故障 | 骑手离线（二期） | mtbf/mttr |
| 紧急插单 | 紧急订单（二期） | 紧截止动态到达 |

### 4.2 关键设计决策

1. **观测布局保持 146 维**（8+4+30+100+4），仅重定义特征语义 → `ppo_network.py` 与 BC 教师索引**零改动复用**
2. **5 骑手 = 5 agent**，one-hot / 全局 19 维与工厂对齐
3. **动作 MultiDiscrete([11])**（单头，RIDERS.count=1）：0=IDLE，1-10=接候选
4. **事件推进**：step 推进到下一决策相关事件（用 `carry[0]` 最早完成，而非链尾）
5. **奖励**：送达+80、准时+80、迟到 Huber、负 slack、无效-0.5、有单不接-1.0、全送完+500、终局 bonus
6. **`final_stats` 键名兼容**工厂（makespan/mean_utilization/total_parts/total_tardiness）
7. **模拟数据**：订单/骑手/坐标全随机（合理分布）
8. **场景切换**：`make_parallel_env` 按 `scenario=='delivery'` 分发
9. **上游精排候选**（会议核心落地）：
   - `candidate_source='upstream'`：候选集合来自上游，不再自采样全池
   - 候选 10 维特征 **[1] 位** = 上游意愿分（原 remaining_ops 恒 2）
   - 默认 `upstream_order_by='urgency'`：**时间紧迫性优先于意愿排序**
   - 兜底 `heuristic_willingness`；不足 backfill；失效计入 `upstream_missing_count`
   - `generate_mock_upstream_candidates` 作为真实上游替身

### 4.3 文件改造清单

| 文件 | 改动 | 状态 |
|---|---|---|
| `environments/delivery_config.py` | 配置 + 上游契约/mock/兜底/评分 | ✅ |
| `environments/delivery_env.py` | DeliverySim/Env + 上游接入 + 意愿分观测 | ✅ |
| `environments/w_factory_env.py` | `make_parallel_env` 场景分发 | ✅ |
| `mappo/ppo_trainer.py` | scenario 注入 + 上游候选按回合透传 | ✅ |
| `mappo/ppo_marl_train.py` | `--scenario` / `--candidate-source` / `--upstream-order-by` | ✅ |
| `auto_train.py` | 场景透传；非工厂跳过工厂专用评估 | ✅ |
| `checks/delivery_logic_check.py` | 环境逻辑回归 | ✅ **31/31（第 4 次重建）** |
| `checks/delivery_upstream_check.py` | 三层联调接口 | ✅ **41/41（第 4 次重建）** |
| `checks/dispatch_integration_check.py` | 场景分发集成 | ✅ **14/14（第 4 次重建）** |
| `evaluation_delivery.py` | 配送启发式基线 | ✅ **第 4 次新增** |
| `evaluation.py` / app 演示 / 甘特图 | 工厂版已删；配送可视化未做 | ⏳ |
| 真实双塔/W&D 上游接入 | 替换 mock（契约已固定） | ⏳ 等业务方口径 |

## 5. 进度记录（倒序）

### 2026-09-15（第 4 次对话）—— 听记复核 + 证据链重建 + 配送基线

**一、听记复核**

再次读取听记（taskUuid `763275696434...375f35`，摘要 + 逐字稿 341 段），对照代码与本文档：

- 核心 MAPPO 改造（动态输入、多智能体派单、紧迫优先、上游 10 单）**已落地**
- 发现文档债：项目路径仍写 `D:\MARL_FOR_W_Factory`；声称的 18/34/12 校验脚本已丢失；`evaluation.py` 已删但文档仍写「待适配」
- 「多加一层向量」为会议 tentative 表述，实现上选择**不扩维、槽位复用**，需在周四同步时主动说明取舍

**二、本次改动**

1. **重建可复现校验**（本机 Python 3.14 + gymnasium/pettingzoo/numpy，纯环境层，迁入 `checks/` 以便入库）：
   - `checks/delivery_logic_check.py` → **31/31**
   - `checks/delivery_upstream_check.py` → **41/41**
   - `checks/dispatch_integration_check.py` → **14/14**（工厂 146 维/双头 `[11,11]` 未受影响）
2. **新增** `evaluation_delivery.py`：idle / fifo / nearest / edd / random 五基线，支持 `--candidate-source upstream` 与 JSON 导出
3. **修订本文档**：路径、仓库清理后结构、会议要求对照表、真实校验数字

**三、基线试跑（2 episode，endogenous）**

| baseline | completion | on_time | tardiness | score |
|---|---|---|---|---|
| idle/fifo/nearest/edd | 1.000 | 0.206 | 721.0 | 0.539 |
| random | 1.000 | 0.382 | 411.9 | 0.595 |

说明：endogenous 候选已按 slack 紧迫序排列，四类贪心基线在小样本上会收敛到同一首候选；random 反而略好，提示**容量约束下的探索价值**，也说明必须等 MAPPO 训练结果才有说服力对比。建议后续加「全局最近骑手指派」类基线。

**四、本机冒烟**

`DeliveryEnv` import/reset/legal-step/upstream 模式均通过；`obs=(146,)`，`action=MultiDiscrete([11])`，upstream 覆盖 5 骑手。

### 2026-09-15（第 3 次对话）—— 三层联调接口补全

- 听记要求核对：原候选全部环境自采样，未接上游意愿分 → 本次补全
- `delivery_config`：`DELIVERY_UPSTREAM_CONFIG` / `normalize_upstream_candidates` / `heuristic_willingness` / `generate_mock_upstream_candidates`
- `delivery_env`：endogenous/upstream 双模式、[1] 位意愿分、urgency/upstream 排序、回退统计
- `ppo_trainer` / `ppo_marl_train` / `auto_train`：上游候选按回合透传与 CLI
- 当时校验 18+34+12（脚本后续在仓库清理中丢失，第 4 次已重建）

### 2026-09-15（第 2 次对话）—— 逻辑校准与缺陷修复

修复 8 处（详见第 7 节）：评分双键、送达后位置、竞态 vs invalid、池内 slack 估算、事件用 `carry[0]`、Huber 公式、trainer 场景感知、事件时间线排序。

### 2026-09-15（第 1 次对话）—— 核心环境改造

- 新增 `delivery_config.py` + `delivery_env.py`
- `make_parallel_env` 场景分发；trainer `--scenario`
- 随机策略 rollout + 5 回合短训冒烟通过

## 6. 待办与风险

- [ ] **周四同步会前**：远程算力跑完整训练（100–200 回合），看奖励曲线与准时率
  - `python mappo/ppo_marl_train.py --scenario delivery --candidate-source upstream`
  - 对比：`--candidate-source endogenous`
  - 自动化：`python auto_train.py "<实验名>" --scenario delivery --candidate-source upstream`
- [ ] 用 `evaluation_delivery.py` 出 MAPPO vs 启发式对比表（需训练好的权重；模型加载入口已预留）
- [ ] 增强基线：全局最近骑手指派 / 插入启发式，避免贪心首候选塌缩
- [ ] 与业务方对齐真实数据字段，替换 `generate_mock_upstream_candidates`
- [ ] upstream vs endogenous A/B（验证「上游精排 + 紧迫调度」）
- [ ] 算力平台选型与租赁
- [ ] 10-14 模板文档：三层架构 + 本仓库定位 + 演示截图
- [ ] Demo：骑手派单可视化（甘特图/地图）+ 智能体包装（LLM + 提示词调用模型）
- [ ] 风险：MAPPO 超参需按配送场景重调；收敛时间待实测
- [ ] 风险：候选 [1] 位语义变更后，旧配送权重需重训或短程微调（工厂权重不受影响）
- [x] 校验脚本可复现（第 4 次重建）
- [x] 配送启发式基线脚本（第 4 次新增）
- [x] `.gitignore` 白名单含 `evaluation_delivery.py`

## 7. 逻辑校准明细（第 2 次对话，供回溯）

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| 1 | `calculate_delivery_episode_score` | 只认 `makespan/total_parts`，trainer 传 `mean_*` → 评分失真 | 双键兼容 |
| 2 | `_advance_to_next_epoch` | 骑手送达后位置不更新 | `position = dropoff` |
| 3 | `step_with_actions` | 同步抢单被抢先记 invalid | 区分 race_conflicts |
| 4 | `get_rewards` | 池内 slack 硬编码 20min | 按订单自身取送路线 |
| 5 | `_next_event_time` | 用 `carry[-1]` 导致链中送达无决策点 | 改用 `carry[0]` |
| 6 | `get_rewards` | Huber 小段线性 vs 工厂二次 | 对齐 `0.5t²` |
| 7 | `_advance_to_next_epoch` | 跨骑手送达事件时间乱序 | 按 (时刻, 骑手名) 排序 |
| 8 | `ppo_trainer.py` | 阶段二回退工厂口径；`_get_base_parts_count` 返回 42 | 场景感知 |

## 8. 三层联调接口契约（供业务方/双塔+W&D 对接）

### 8.1 数据契约（上游 → MAPPO）

```python
upstream_candidates = {
    "agent_骑手A": [                      # key 支持 "骑手A" 或 "agent_骑手A"
        {"order_id": 12, "willingness": 0.93},   # 意愿分 ∈ [0,1]
        {"order_id": 7,  "willingness": 0.81},
        # 每骑手最多 top_k（默认 10）条；亦可 (order_id, willingness) 元组
    ],
    "骑手B": [...],
    # 或 "global": [...] 对所有骑手生效后按专属候选优先合并
}
```

- 注入：`config['upstream_candidates']`
- 容错：order_id 必须与 env 一致；缺意愿分忽略；非法返回空
- 未覆盖骑手 → 回退自采样；不足 backfill；失效 → `upstream_missing_count`

### 8.2 观测承载

候选 10 维的 **[1] 位** = `upstream_willingness`（工厂为 remaining_ops）。
→ 146 维、网络结构、BC 教师索引（只用 [0][3][8]）**全部不变**。

### 8.3 排序口径

| `upstream_order_by` | 含义 |
|---|---|
| `urgency`（默认） | 候选来自上游，**按 slack 升序**（会议口径） |
| `upstream` | 保持上游意愿序（A/B 对比） |

### 8.4 验证命令

```bash
# 纯环境层，无需 TensorFlow
set PYTHONPATH=D:\rider-dispatch-mappo
python checks/delivery_logic_check.py         # 31/31
python checks/delivery_upstream_check.py      # 41/41
python checks/dispatch_integration_check.py   # 14/14

# 启发式基线
python evaluation_delivery.py --baseline all --episodes 5 --candidate-source upstream
```
