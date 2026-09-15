# update.md — 运力商圈智能体比赛改造日志

> 本文件是本次比赛代码改造的唯一进度文档，**每次对话必须维护更新**。
> 项目：`D:\MARL_FOR_W_Factory`（原工业车间调度 MAPPO → 外卖骑手派单调度）

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
- **数据策略**：强化学习不挑数据真实性 → **先用随机模拟数据跑通逻辑**（分布合理即可），后续接真实数据微调
- **数据字段需求**（业务方提供）：骑士 ID、订单链（取餐点/送餐点）、预期/实际送达时间、订单重量/体积、用户特征、企业渠道等
- **算力**：当前无 GPU；堡垒机纯 CPU 效率低；短期 CPU 小规模验证，需商定租用算力云平台
- **关键时间节点**：
  - **2026-10-14 前提交参赛模板文档**（会议纪要可直接转）
  - **2026 年 11 月中旬决赛线下路演 + Demo 展示**
  - 每周四进度同步会（下次：算法改造进展 + 数据准备情况）

## 2. 原项目结构（工业车间调度，改造前）

| 模块 | 文件 | 说明 |
|---|---|---|
| 环境 | `environments/w_factory_env.py` | WFactorySim（SimPy 仿真，~2190 行）+ WFactoryEnv（PettingZoo） |
| 配置 | `environments/w_factory_config.py` | 唯一配置源：工作站/工艺路线/订单/奖励/PPO超参 |
| MAPPO | `mappo/ppo_network.py` | Actor(146→1024/512/256→2头×11) + 集中式 Critic(19+onehot5→V) |
| 训练 | `mappo/ppo_trainer.py` / `ppo_worker.py` / `ppo_marl_train.py` | 两阶段训练、4 worker 并行采集、GAE、启发式 BC 教师 |
| 评估 | `evaluation.py` 等 | 启发式基线（FIFO/EDD/SPT/CR）+ 甘特图 |

- 观测 146 维 = 自身8(one-hot5+容量+繁忙率+故障) + 全局4 + 队列摘要30 + 候选工件10×10 + 未来订单4
- 动作 MultiDiscrete([11,11])：0=IDLE，1-10=候选工件（EDD5+SPT3+随机2 采样）
- MAPPO 栈与环境耦合面很窄：`make_parallel_env(config)` + PettingZoo 接口 + `infos['global_state'/'action_mask']` + `env.sim.get_final_stats()`

## 3. 改造设计（骑手派单调度）

### 3.1 场景映射

| 工业 | 运力配送 | 实现要点 |
|---|---|---|
| 工件 Part | 配送订单 | 取餐点/送餐点坐标、ready_time、due_date（承诺送达）、优先级、品类 |
| 机器/工作站 | 骑手（agent） | 位置、最大携带单量（capacity）、速度、繁忙率、休息/离线 |
| 工艺路线（固定5站） | 订单路线（动态2段） | 去取餐点→取餐→去送餐点→送达，路程按坐标距离/速度计算 |
| 站点队列 | 待派单池（共享） | 候选按骑手视角采样：slack最近5 + 取餐距离最近3 + 随机2 |
| 交期 due_date | 承诺送达时间 | 奖励、time_pressure、slack 均按送达截止计算 |
| 设备故障 | 骑手离线/休息（二期） | mtbf/mttr 机制沿用 |
| 紧急插单 | 紧急订单（二期） | 紧截止时间订单动态到达 |

### 3.2 关键设计决策

1. **观测布局保持 146 维结构不变**（8+4+30+100+4），仅重定义特征语义 —— 让 `ppo_network.py`（含启发式 BC 教师的索引偏移）**零改动复用**
2. **5 个骑手 = 5 个 agent**，与原 5 工作站对齐（one-hot 宽度、全局状态 19 维均不变）
3. **动作 MultiDiscrete([11])**（单头）：0=IDLE（不接单），1-10=从候选池接单；骑手携带量未满时可继续接单（模拟真实骑手批量带单）
4. **事件推进**：step 内推进到下一个"决策相关事件"（骑手空闲/订单到达），保证每步都有有效决策，避免空转步
5. **奖励语义迁移**：送达+80、准时+80、迟到 Huber 惩罚、负 slack 持续惩罚、无效动作-0.5、有单不接-1.0、全部送完+500、终局分数 bonus（复用 REWARD_CONFIG 数值结构）
6. **`final_stats` 键名兼容**（makespan/mean_utilization/total_parts/total_tardiness），训练器/KPI 日志无需改
7. **模拟数据生成**：订单/骑手/坐标全随机（合理分布），听记确认 RL 可先用模拟数据
8. **场景切换**：`make_parallel_env(config)` 按 `config['scenario']=='delivery'` 分发，工厂场景完全不受影响
9. **上游精排候选接入（第 3 次对话新增，会议结论落地）**：听记明确"上游取出 10 个最想接的单给到调度层，你的算法立马起作用"，即 MAPPO 的输入应是**上游候选集合 + 意愿分**，而 MAPPO 负责"时间关系"。落地方式：
   - `candidate_source='upstream'` 时，候选集合直接取上游（双塔+W&D）精排结果，不再自行从全池采样
   - 候选 10 维特征的 **[1] 位**（配送场景"剩余段数"恒为 2、无信息量）复用为**上游精排意愿分** → 146 维布局与网络/BC 教师索引**保持不变**
   - 排序遵循会议口径：**时间紧迫性优先于意愿排序**（`upstream_order_by='urgency'`，默认）；同时保留 `'upstream'` 档用于 A/B 对比
   - 上游未覆盖时兜底：按订单属性估意愿分（`heuristic_willingness`）；候选不足用池内最紧迫订单补齐；失效订单计入 `upstream_missing_count`
   - 提供 `generate_mock_upstream_candidates` 作为真实上游接入前的替身，可**立即开始三层联调训练**

### 3.3 文件改造清单

| 文件 | 改动 | 状态 |
|---|---|---|
| `environments/delivery_config.py` | 新增：骑手/地理/订单生成/配送奖励/评分配置 + **三层联调上游接口（契约/归一化/mock 生成器/兜底意愿分）** | ✅ 完成 |
| `environments/delivery_env.py` | 新增：DeliverySim + DeliveryEnv + 工厂函数 + **上游候选接入 + 意愿分观测特征** | ✅ 完成 |
| `environments/w_factory_env.py` | `make_parallel_env` 增加 scenario 分发 | ✅ 完成 |
| `mappo/ppo_trainer.py` | `SimplePPOTrainer(env_config=...)`；订单生成按场景分发；scenario 注入 worker/评估环境；**上游候选按回合生成/透传（训练与评估同口径）** | ✅ 完成 |
| `mappo/ppo_marl_train.py` | `--scenario delivery` + **`--candidate-source` / `--upstream-order-by`** | ✅ 完成 |
| `auto_train.py` | **`--scenario` / `--candidate-source` 透传；非工厂场景跳过工厂专用 evaluation/debug** | ✅ 完成（第 3 次对话） |
| `environments/delivery_env.py` 单元冒烟 | 维度/动作/奖励/终止 全链路 | ✅ 通过 |
| `train_delivery_smoke.py` 短训冒烟 | 5 回合小步数真实训练 | ✅ 通过 |
| 逻辑校准与缺陷修复（第 2 次对话） | 评分键名/骑手位置/竞态/事件推进/Huber 等 8 处 | ✅ 完成（18/18 环境级校验通过） |
| **三层联调接口（第 3 次对话）** | **上游候选/意愿分接入 + 会议口径落地 + 3 套校验（18 + 34 + 12 全绿）** | ✅ 完成 |
| `evaluation.py` / 甘特图 / app 演示 | 配送场景适配 | ⏳ 未开始（下阶段） |
| 真实双塔/W&D 上游接入 | 用真实精排结果替换 mock（数据契约已固定） | ⏳ 等业务方特征口径 |

## 4. 进度记录（倒序，每次对话追加）

### 2026-09-15（第 3 次对话）—— 听记要求核对 + 三层联调接口补全 + git 检查点

**一、听记核对结论（需求对照）**

读取听记（`7632756964344...375f35`，09-14「双塔模型与MAPPo调度算法讨论」，逐字稿 341 段）与当前代码逐条比对：

| 会议要求 | 现状判定 |
|---|---|
| 骑手=机器、订单=工件，场景迁移 | ✅ 已满足（delivery 环境） |
| 骑手单量/地址动态、位置负载实时变动 | ✅ 已满足（RiderState 位置/携带/占用） |
| "先派给骑士A还是骑士B"的调度决策 | ✅ 已满足（5 骑手 agent 共享池抢单） |
| 时间紧迫性优先（12:30 出发 2 点前送完，先送最紧迫的） | ✅ 已满足（slack/time_pressure + 迟到/Huber 惩罚） |
| 模拟数据先跑通逻辑 | ✅ 已满足 |
| **上游漏斗接入：召回 100 → W&D 精排 10 → MAPPO 调度** | ❌ **缺口（本次补全）**：原实现候选全部由环境自采样，完全未接上游意愿分，无法体现"意愿 vs 紧迫"权衡 |
| 输出带权重模型 + 派单/执行序列 | ⚠️ 模型已可产出；执行序列可视化（evaluation/甘特图/app）未适配 |
| 训练收敛验证、算力 | ⏳ 待远程算力 |

**二、本次改造（补缺口）**

1. `delivery_config.py`：新增第 5.5 节三层联调配置与工具 —— `DELIVERY_UPSTREAM_CONFIG`（数据契约）、`normalize_upstream_candidates`（契约校验/归一化）、`heuristic_willingness`（兜底意愿分）、`generate_mock_upstream_candidates`（mock 上游）
2. `delivery_env.py`：候选来源双模式（`endogenous`/`upstream`）、候选特征 [1] 位承载上游意愿分、排序口径 urgency/upstream 可切换、上游覆盖/失效/回退三类统计、`candidates_map` 与 `final_stats` 暴露联调口径
3. `ppo_trainer.py`：配送场景按回合生成/透传上游候选（mock 或外部注入），worker 与评估环境同口径
4. `ppo_marl_train.py`：新增 `--candidate-source`、`--upstream-order-by`；`auto_train.py` 透传场景参数并跳过工厂专用 downstream 脚本（防止产出错误口径的对比数据）

**三、验证（本机隔离 venv，纯环境层，未跑 TensorFlow）**

- `.tmp/delivery_logic_check.py`：**18/18 通过**（原有回归，确认零回归）
- `.tmp/delivery_upstream_check.py`：**34/34 通过**（契约/兜底/观测对齐/会议口径/回退统计/mock 生成器/rollout）
- `.tmp/dispatch_integration_check.py`：**12/12 通过**（工厂场景 146 维/双头动作空间未受影响 + delivery 分发 + 上游意愿分逐单核对一致）
- 关键结论：构造用例证明**"紧急但意愿低（0.10）"的单排在"宽松但意愿高（0.99）"之前**，即时间紧迫性优先于意愿排序 ✅

**四、git 检查点（按要求改动前先暂存）**

- 暂存点：`stash@{0}`（message=`checkpoint: 配送场景改造(第2次对话后) 2026-09-15 13:15`）+ 标签 `checkpoint-delivery-20260915`，含 4 个改动文件 + 3 个未跟踪新文件
- ⚠️ 过程中沙箱中断导致 `.git/refs` 目录丢失（仓库一度不可识别），已用 `git update-ref` 重建 `refs/heads/main`、扫描对象库找回 stash 提交并 `git stash store` 复原引用；**工作区文件无任何丢失**

### 2026-09-15（第 2 次对话）—— 逻辑校准与缺陷修复
- 全链路代码审查（delivery_config / delivery_env / make_parallel_env 分发 / trainer scenario 注入 / worker / BC 教师索引对齐），确认观测布局与 ppo_network BC 教师索引（exists=0/leg1=3/congestion=4/tpress=8/slack=9，cand_start=42）零改动复用成立
- **修复 8 处逻辑缺陷**（详见 6 节）：
  1. 评分函数 KPI 键名不兼容（阻断级）：配送评分只认 makespan/total_parts，训练循环传的是 mean_makespan/mean_completed_parts → 评分恒虚高、毕业判定失效
  2. 骑手送达后位置不更新（阻断级）：候选距离/拥堵/观测全部基于过期位置
  3. 并行接单竞态误判为无效动作：同步决策被同伴抢先不计 invalid 惩罚，新增 race_conflicts 统计
  4. 池内订单 slack 惩罚用硬编码 20min 估算 → 改为订单自身取送路线（参考速度）
  5. `_next_event_time` 用 FIFO 链最晚送达（carry[-1]）→ 链中间订单送达无决策点；改用 carry[0]
  6. Huber 迟到惩罚小段公式与工厂不一致（线性 vs 二次）
  7. trainer 阶段二动态事件采样回退默认值混入工厂口径；`_get_base_parts_count` 对 delivery 返回工厂零件数 42（sqrt 阈值缩放基数错）
  8. 跨骑手送达事件写入 event_timeline 不按时间序（影响可视化）；train() 一处日志硬编码工厂配置
- **环境级逻辑验证 18/18 通过**（`.tmp/delivery_logic_check.py`，纯环境层无 TF）：维度/mask/rollout/位置更新/竞态/评分双口径/动态事件/时间单调性
- 未跑训练（本机约束），训练验证待远程算力

### 2026-09-15（第 1 次对话）
- 读取听记（摘要 + 全部转写 7 页），确认改造需求与时间节点
- 梳理原代码结构与 MAPPO 栈耦合面
- 创建本文件
- **完成核心改造**（见 3.3 状态列）：
  - 新增配送场景环境（`delivery_config.py` + `delivery_env.py`，~1100 行）：事件驱动仿真、146 维观测、10 候选接单、配送奖励体系、终局评分
  - `make_parallel_env` 场景分发；训练器 `--scenario` 支持；worker/评估环境注入
  - **冒烟验证通过**：随机策略 rollout（46 单全部送达、维度 146、mask/奖励正常）+ 5 回合短训（loss 正常下降、KPI 正常产出、模型保存成功）
- 遗留事项见第 5 节

## 5. 待办与风险

- [ ] **周四同步会前**：远程算力跑一轮完整 CPU/GPU 训练（如 100-200 回合），看奖励曲线与准时率是否收敛
  - 命令（三层联调用 mock 上游）：`python mappo/ppo_marl_train.py --scenario delivery --candidate-source upstream`
  - 对比跑（环境自采样）：`python mappo/ppo_marl_train.py --scenario delivery --candidate-source endogenous`
  - 或走自动化：`python auto_train.py "<实验名>" --scenario delivery --candidate-source upstream`
- [ ] evaluation.py 启发式基线（FIFO/最近骑手/最早截止）适配配送场景，出对比数据（auto_train 已对非工厂场景跳过工厂专用评估，避免错误口径）
- [x] `.gitignore` 白名单补充 `environments/delivery_env.py`、`environments/delivery_config.py`、`update.md`（第 3 次对话已完成）
- [ ] 与业务方对齐真实数据字段口径（骑士ID/订单链/送达时间/特征），设计数据导入接口替换随机生成器
  - MAPPO 侧契约已固定：`config['upstream_candidates'] = {"骑手X":[{"order_id":..,"willingness":..}]}`，见 `DELIVERY_UPSTREAM_CONFIG` 注释；接真实双塔/W&D 时只需替换 `generate_mock_upstream_candidates`
- [ ] upstream 模式与 endogenous 模式的 A/B 对比（验证"上游精排 + 紧迫性调度"是否优于纯自采样候选）
- [ ] 算力平台选型与租赁（决赛前需规模化训练）
- [ ] 10-14 模板文档：三层架构说明 + 本仓库定位 + 演示截图
- [ ] Demo：骑手派单可视化（甘特图/地图轨迹）+ 智能体包装（LLM + 提示词调用模型）
- [ ] 风险：MAPPO 超参对配送场景可能需要重调（状态空间语义变了，含新增的意愿分特征）；训练收敛时间待实测
- [ ] 风险：观测候选特征 [1] 位语义由常量改为意愿分 —— 若沿用**旧配送模型权重**继续训练，建议重新训练或先做短程微调后再评估（工厂场景观测语义未变，不受影响）

## 6. 逻辑校准明细（第 2 次对话，供回溯）

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| 1 | `delivery_config.calculate_delivery_episode_score` | 只认 `makespan/total_parts/total_tardiness`，trainer 传 `mean_*` 键 → 评分恒失真，两阶段毕业判定失效 | 双键兼容（与工厂口径一致） |
| 2 | `DeliverySim._advance_to_next_epoch` | 骑手 position 初始化后永不更新，候选距离/拥堵/观测基于过期位置 | 送达结算时 `position = dropoff` |
| 3 | `DeliverySim.step_with_actions` | 同步决策下后处理骑手接单被同伴抢先 → 误记 invalid_action 惩罚（观测时该动作合法） | 区分 race_conflicts 与 invalid；新增统计键 |
| 4 | `DeliverySim.get_rewards` | 池内订单 slack 用硬编码 `est_route=20.0` | 按订单自身取送路线 + 参考速度估算 |
| 5 | `DeliverySim._next_event_time` | 用 `carry[-1]`（FIFO 链最晚送达）→ 链中间订单送达无决策点，容量释放/位置更新被推迟到整链结束 | 改用 `carry[0]`（最早完成） |
| 6 | `DeliverySim.get_rewards` | Huber 小迟到段线性 `δt`，工厂为二次 `0.5t²` | 对齐工厂公式 |
| 7 | `DeliverySim._advance_to_next_epoch` | 跨骑手送达事件按骑手字典序写 event_timeline，时间乱序（影响可视化） | 按 (送达时刻, 骑手名) 排序写入 |
| 8 | `ppo_trainer.py` | 阶段二动态事件采样回退到工厂 `EQUIPMENT_FAILURE/EMERGENCY_ORDERS`；`_get_base_parts_count` 对 delivery 返回 42（工厂零件数）；delivery 评分未注入本回合实际订单分母；train() 日志硬编码工厂配置 | 全部改为场景感知；评分分母取 `_last_episode_config['custom_orders']` |

## 7. 三层联调接口契约与验证（第 3 次对话新增，供业务方/双塔+W&D 侧对接）

### 7.1 数据契约（上游 → MAPPO）

```python
# MAPPO 消费的上游精排结果（双塔召回 + Wide&Deep 精排）
upstream_candidates = {
    "agent_骑手A": [                      # key 支持 "骑手A" 或 "agent_骑手A"
        {"order_id": 12, "willingness": 0.93},   # 意愿分 ∈ [0,1]
        {"order_id": 7,  "willingness": 0.81},
        # 每骑手最多 top_k（默认 10）条
    ],
    "骑手B": [...],
}
```

- 注入方式：`config['upstream_candidates']`（训练：`SimplePPOTrainer(env_config={...})`；CLI：mock 上游自动生成）
- 容错：`order_id` 必须与 env 侧订单一致；元组 `(order_id, willingness)` 亦可；缺意愿分的条目忽略；非法输入返回空
- 未覆盖骑手 → 回退环境自采样；上游候选不足 → 池内最紧迫订单补齐；失效订单 → `upstream_missing_count`

### 7.2 观测特征承载（零改动复用网络的关键）

候选 10 维特征的 **[1] 位**：工厂为 `remaining_ops`，配送原为常量 `remaining_legs=1.0`（2 段取送恒定、无信息量），现复用为 **`upstream_willingness`**。
→ 146 维布局、`ppo_network` 结构、BC 启发式教师索引（只用 [0][3][8]）**全部保持不变**。

### 7.3 排序口径（会议结论）

| `upstream_order_by` | 含义 | 用途 |
|---|---|---|
| `urgency`（默认） | 候选集合来自上游，**按时间紧迫性（slack 升序）排列** | 会议口径：时间紧迫性优先于意愿排序 |
| `upstream` | 保持上游意愿序 | A/B 对比实验 |

### 7.4 验证脚本（本机隔离 venv 运行，无需 TensorFlow）

```bash
python .venv2/Scripts/python.exe .tmp/delivery_logic_check.py        # 18/18 回归
python .venv2/Scripts/python.exe .tmp/delivery_upstream_check.py     # 34/34 三层联调接口
python .venv2/Scripts/python.exe .tmp/dispatch_integration_check.py  # 12/12 场景分发集成
```

