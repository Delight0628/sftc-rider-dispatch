# update.md — 运力商圈智能体比赛改造日志

> 本文件是本次比赛代码改造的唯一进度文档，**每次对话必须维护更新**。
> 项目路径：`D:\rider-dispatch-mappo`（工业车间 MAPPO → 外卖骑手派单调度）
> 最近同步：2026-09-17（果洛全量样本接入 + 真实数据链路）

---

## 1. 背景与比赛信息

来源：2026-09-14 听记「双塔模型与MAPPO调度算法讨论」；2026-09-17 钉钉群「顶尖算法团队」。

- **参赛项目**：运力评估商圈智能体（模型可包装为智能体工具：外接 LLM + 提示词）
- **三层级联架构**（本仓库负责第 3 层）：
  1. **双塔**（召回/粗排）：骑手塔 + 订单塔 → 向量相似度；正样本=接单
  2. **Wide & Deep**（精排）：输出接单意愿分
  3. **MAPPO**（调度）：时间约束下多骑手全局派单与执行顺序
- **MAPPO 场景映射**：
  - 骑手 = 机器（agent），订单 = 工件
  - 原：工件走固定机器数；新：**订单路线与骑手单量两边都动**
  - 「先做 A/B/C」→「先把单派给骑士 A 还是 B」
  - **时间紧迫性优先于意愿排序**
  - 上游给出「最想接的 ~10 单」→ MAPPO 决定时间关系
- **数据**：先模拟跑通，再接真实宽表/维表
- **算力**：当前无 GPU；完整训练待租用算力平台
- **节点**：2026-10-14 模板文档；2026-11 中旬决赛路演 + Demo；每周四同步

## 2. 当前仓库结构（与磁盘一致）

```
rider-dispatch-mappo/
├── environments/
│   ├── delivery_config.py       # 配送配置/奖励/评分/上游契约/mock
│   ├── delivery_env.py          # DeliverySim + DeliveryEnv（146 维观测）
│   ├── real_data_schema.py      # 真实表字段映射 + KPI 分层筛选
│   ├── real_data_loader.py      # 样本(xlsx/csv/json) → custom_orders/riders
│   ├── w_factory_env.py         # make_parallel_env 场景分发
│   └── w_factory_config.py      # 网络/评分等共享配置
├── mappo/                       # Actor-Critic / Trainer / Worker（零分叉复用）
├── checks/
│   ├── delivery_logic_check.py         # 31/31
│   ├── delivery_upstream_check.py      # 41/41
│   ├── dispatch_integration_check.py   # 14/14
│   ├── real_data_schema_check.py       # 24/24
│   └── make_realistic_sample.py        # 无库权限时生成同构样本
├── evaluation_delivery.py       # 启发式基线 + 真实样本评估
├── auto_train.py
├── data/schema/                 # 833/71 字段中文对照（入库）
├── data/real_incoming/          # 果洛样本 CSV（本地，不入库）
└── update.md / README.md
```

已删除（`910895f`）：工厂 `evaluation.py`、`app/`、`plotting.py` 等。

关键 commit：
- `e8b0279` 配送环境 + 上游候选
- `910895f` 聚焦骑手派单，删工厂演示
- `b9cc082` 真实数仓字段映射 + KPI 筛选 + 样本导入

## 3. 会议 / 群消息要求对照

| 要求 | 现状 | 证据 |
|---|---|---|
| 骑手=机器、订单=工件 | ✅ | `delivery_env.py` |
| 订单路线/骑手单量动态（两边都动） | ✅ | 坐标行程 + carry/capacity/projected_free |
| 先派给 A 还是 B | ✅ | 5 agent 共享池抢单 |
| 时间紧迫性优先 | ✅ | slack/time_pressure + Huber；upstream 默认 `urgency` |
| 上游 ~10 单 + 意愿分 | ✅ | `candidate_source=upstream`，候选[1]=willingness |
| 「多加一层向量」 | ⚠️ 取舍 | 不扩 146 维；意愿分占用恒 2 的 remaining_ops 槽 |
| 模拟数据先跑通 | ✅ | `generate_random_delivery_orders` |
| 真实订单/骑士标签作训练源 | ✅ 链路通 | schema/loader + 果洛全量样本 |
| 调度效果可度量 | ✅ | KPI 分层 + `evaluation_delivery` 输出 |
| 启发式基线对比 | ✅ | idle/fifo/nearest/edd/random |
| 完整训练收敛 | ⏳ | 阻塞算力 |
| 智能体包装 / 模板 / Demo 可视化 | ⏳ | 决赛前 |

**总判**：环境层、三层联调、真实数据导入与基线评估已齐；训练收敛、LLM 包装、路演材料仍是缺口。

## 4. 核心设计

### 4.1 场景映射

| 工业 | 配送 | 要点 |
|---|---|---|
| 工件 | 订单 | pickup/dropoff、ready、due、优先级、重量 |
| 机器 | 骑手 agent | 位置、capacity、速度、繁忙/离线 |
| 固定工艺路线 | 动态两段取送 | 按坐标/速度计时 |
| 站点队列 | 共享待派池 | endogenous：EDD5+近3+随机2；upstream：上游集合 |
| 交期 | 承诺送达 | 奖励 / time_pressure / slack |

### 4.2 关键决策

1. **观测 146 维布局不变**（8+4+30+100+4）→ 网络与 BC 教师零改动  
2. **5 骑手 = 5 agent**；动作 `MultiDiscrete([11])`：0=IDLE，1-10=接候选  
3. **事件推进**用 `carry[0]`（链上最早完成），避免中段送达无决策点  
4. **奖励**：送达+80、准时+80、迟到 Huber、负 slack、无效-0.5、有单不接-1.0、全送完+500、终局 bonus  
5. **`final_stats` 与工厂键兼容**，另增 `distance_per_order` / `avg_tardiness` / `late_gt_{5,15,30}m_rate`  
6. **`config['riders']` / `geo_config` 可注入**（真实样本）  
7. **上游候选**：urgency 优先；兜底 heuristic；backfill；missing/fallback 统计  
8. **真实样本**：秒/毫秒时间戳自适应；经纬度 P5–P95 自适应网格；同点订单黄金角抖动

### 4.3 三层联调契约（上游 → MAPPO）

```python
config["upstream_candidates"] = {
    "agent_骑手A": [{"order_id": 12, "willingness": 0.93}, ...],  # ≤top_k=10
    "骑手B": [...],
    # 或 "global": [...] 合并到未覆盖骑手
}
config["candidate_source"] = "upstream"      # | endogenous
config["upstream_order_by"] = "urgency"      # | upstream（A/B）
```

- 候选 10 维特征 **[1] 位 = willingness**（工厂 remaining_ops）  
- order_id 必须与 env 订单一致；缺意愿分忽略  
- 未覆盖骑手回退自采样；池内不足 backfill  

## 5. 真实数据与 KPI

### 5.1 数仓表（群内确认）

| 表 | 规模 | 用途 |
|---|---|---|
| `dts.dwd_fact_order_whole` | 833 字段，分区 edt/emn | 订单时空、链路时刻、准时/超时、计提 |
| `dw.dim_rider_single` | 71 字段，在职骑手 | 等级/车型/距离偏好/站点 |

字段对照：`data/schema/*字段中文对照.txt`。

### 5.2 特征工程收敛

- **订单塔 / W&D item**：取送坐标、ready/due 偏移、重量、距离、service/dispatch/delivery 类型、城市/站点、天气、小时、预估时长  
- **骑手塔**：level、work_type、vehicle_type、extra_flag 位、team、city/station、在职时长  
- **双塔样本**：正=完单（status=1 + rider_ucode）；负=同商圈同小时未派给该骑手（曝光日志到位后改硬负）

### 5.3 调度 KPI 筛选（`real_data_schema.KPI_SPEC`）

| 层级 | 指标 |
|---|---|
| 主指标（路演） | 完成率、准时率、总超时分钟 |
| 效率 | makespan、骑手利用率、单均里程 |
| 诊断 | 超时>5/15/30 分桶、接起/到店/配送时长、接单超时率 |
| 业务代理 | 单均骑士计提、超时赔付率 |

不进主表：营销/渠道枚举等弱相关数百列。

### 5.4 果洛全量样本（2026-09-15）

来源：`骑手派单仿真样本_果洛藏族自治州_20260915_全量.xlsx`（小新导出；500 行截断版作废）。

| 项 | 值 |
|---|---|
| 订单 | 1193 行；完成 1032 / 接起取消 160 / 未接 1 |
| 骑手 | 200 行；当日 22 完单骑手中 21 可 ucode 关联 |
| 清洗 | 仅 09-15 完成单 → `data/real_incoming/orders_guoluo_20260915_ready.csv`（1032） |
| 真实准时率 | ≈61.6% |
| 特征 | 秒级时间戳；同点单约 87%；跨数百 km → 自适应 80km 网格 |

**基线（120 单窗 × 2 episode × 5 骑手）**

| baseline | compl | ontime | late15 | km/order | score |
|---|---|---|---|---|---|
| **nearest** | 1.00 | **0.683** | **0.317** | **16.7** | **0.491** |
| fifo | 1.00 | 0.600 | 0.358 | 25.9 | 0.490 |
| random | 0.76 | 0.439 | 0.561 | 37.0 | 0.403 |
| idle/edd | 0.57 | 0.412 | 0.588 | 51.2 | 0.326 |

MAPPO 对标：完成率不掉前提下，逼近/超过 nearest 的准时与里程。

## 6. 改造清单（状态）

| 模块 | 状态 |
|---|---|
| `delivery_config.py` / `delivery_env.py` | ✅ 环境 + 上游 + 真实 riders/geo 注入 |
| `real_data_schema.py` / `real_data_loader.py` | ✅ 字段映射、xlsx/csv、自适应投影、同点抖动 |
| `evaluation_delivery.py` | ✅ 五基线 + `--real-orders/--real-riders` + KPI 输出 |
| `ppo_trainer.py` / `ppo_marl_train.py` / `auto_train.py` | ✅ scenario + candidate-source 透传 |
| checks 四套 | ✅ 31+41+14+24 全绿 |
| 真实双塔/W&D 替换 mock | ⏳ 契约已固定 |
| MAPPO 完整训练 | ⏳ 待算力 |
| 配送可视化 / LLM 智能体包装 | ⏳ 决赛 Demo |

## 7. 进度记录（倒序）

### 2026-09-17 · 第 6 次 — 果洛全量样本接入
- 接入全量 xlsx（1193/200），弃用 500 行截断版  
- 清洗当日完成单 1032；跑五基线；nearest 最优（见 §5.4）  
- loader：xlsx 双 sheet、秒级时间、P5–P95 网格、同点抖动  

### 2026-09-17 · 第 5 次 — 群消息数据源 + KPI 筛选
- 对齐 `dwd_fact_order_whole` / `dim_rider_single`  
- 新增 schema/loader；KPI 分层；evaluation 支持真实样本  
- env 支持 riders/geo 注入；final_stats 补 KPI 字段  
- commit `b9cc082`  

### 2026-09-15 · 第 4 次 — 听记复核 + 证据链重建
- 发现文档路径过时、校验脚本丢失、evaluation 已删仍写待适配  
- 重建 checks（31/41/14）；新增 `evaluation_delivery.py`  

### 2026-09-15 · 第 3 次 — 三层联调接口
- 上游候选/意愿分接入；urgency 优先；CLI `--candidate-source`  

### 2026-09-15 · 第 2 次 — 逻辑校准（8 处）
见 §9。  

### 2026-09-15 · 第 1 次 — 核心环境
- `delivery_config/env`；`make_parallel_env` 分发；短训冒烟  

## 8. 待办与风险

- [ ] 远程算力完整训练（100–200 回合）  
  - `python mappo/ppo_marl_train.py --scenario delivery --candidate-source upstream`  
  - 对比 endogenous；或 `auto_train.py`  
- [ ] MAPPO 权重 vs nearest/fifo 对比表（模型加载入口已预留）  
- [ ] 可选：更贴近业务的全局指派基线（当前贪心首候选在稀疏区完成率偏低）  
- [ ] 双塔/W&D 真实输出替换 `generate_mock_upstream_candidates`  
- [ ] 曝光日志 → 双塔硬负样本  
- [ ] due 口径与业务确认（当前优先 `loc_assessment_time`）  
- [ ] 10-14 模板文档；11 月 Demo（甘特图/地图 + LLM 工具包装）  
- [ ] 风险：配送超参需重调；候选[1]语义变更后旧权重需重训  
- [x] 真实字段映射 / 导入 / KPI / 果洛样本基线  
- [x] 校验脚本可复现并入库  

## 9. 逻辑校准明细（第 2 次，供回溯）

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| 1 | `calculate_delivery_episode_score` | 只认 makespan/total_parts，trainer 传 mean_* | 双键兼容 |
| 2 | `_advance_to_next_epoch` | 送达后位置不更新 | `position=dropoff` |
| 3 | `step_with_actions` | 抢单竞态记 invalid | 区分 race_conflicts |
| 4 | `get_rewards` | 池内 slack 硬编码 20min | 按订单自身路线 |
| 5 | `_next_event_time` | `carry[-1]` 无中段决策点 | 改 `carry[0]` |
| 6 | `get_rewards` | Huber 小段线性 | 对齐 `0.5t²` |
| 7 | `_advance_to_next_epoch` | 送达事件时间乱序 | 按 (时刻,骑手) 排序 |
| 8 | `ppo_trainer.py` | 阶段二回退工厂口径 | 场景感知 |

## 10. 验证命令

```bash
set PYTHONPATH=D:\rider-dispatch-mappo

python checks/delivery_logic_check.py         # 31/31
python checks/delivery_upstream_check.py      # 41/41
python checks/dispatch_integration_check.py   # 14/14
python checks/real_data_schema_check.py       # 24/24

# 果洛真实样本基线
python evaluation_delivery.py --baseline all --episodes 2 \
  --real-orders data/real_incoming/orders_guoluo_20260915_ready.csv \
  --real-riders data/real_incoming/riders_guoluo_20260915_full.csv

# 训练（需算力）
python mappo/ppo_marl_train.py --scenario delivery --candidate-source upstream
```
