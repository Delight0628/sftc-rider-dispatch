# project.md — Rider Dispatch MAPPO（项目总图）

> **给后续对话中的 Agent**：开工前先读本文件；训练目的、观测约束、数据口径、展示形态变更后**必须回写本节**。  
> （原名 `AGENTS.md`，2026-09-30 起改为 `project.md`，**每次会话维护本文件**。）  
> 工作目录：`D:\rider-dispatch-mappo`（对应远端 `/gemini/code/sftc-rider-dispatch`）。

---

## 0. 项目全景（一眼看懂）

```text
┌─────────────────────────────────────────────────────────────────┐
│  数据层  果洛 xlsx（1193单+200骑手）→ real_data_loader → 订单池   │
│          只用 ex-ante 状态；无「骑手×订单意愿」字段（已核实）      │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  仿真层  DeliverySim / DeliveryEnv                               │
│          取→送两段、共享池、事件推进、due/slack 奖励              │
│          观测：146 维（N=5）∥ set 观测（N 可变，方案 B）          │
└───────────────┬─────────────────────────────┬───────────────────┘
                ▼                             ▼
┌──────────────────────────┐   ┌──────────────────────────────────┐
│ 训练层 MAPPO + CTDE      │   │ 评估层 基线/KPI/离线对照          │
│ Actor: 分布式 per 骑手   │   │ 完成率·准时·单均迟到·util·里程   │
│ Critic: 集中式全局 V     │   │ evaluation_delivery + checks      │
│ PPO clip + GAE（未改）   │   └──────────────────────────────────┘
└───────────────┬──────────┘
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  输出  每骑手「紧迫维度」订单 list                               │
│  ════════════════════════════════════════════════                │
│  外部融合（非本仓）：双塔偏好 list  ⊕  MAPPO 紧迫 list            │
│                 → 实时订单 list → **Live 调度大屏**（最终展示）   │
└─────────────────────────────────────────────────────────────────┘
```

**MAPPO / CTDE 主体未改**：仍是多智能体 Actor-Critic、集中训练分散执行、PPO+GAE。  
变的是：① 不再融合「骑手偏好/意愿」进策略（并行双塔侧负责）；② 方案 B 可选变长观测。

---

## 1. 项目一句话

用 **MAPPO（CTDE）** 学出骑手派单策略：**多接、准点、少闲置**；只做**时间紧迫性**维度，偏好由并行双塔处理，结果外部加权，最终上 **Live 调度大屏**。

---

## 2. 业务问题

即时配送 / 同城运力：多个骑手 + 动态订单池，如何派单使得

| 目标 | 说明 |
|------|------|
| 早送达 | 最小化送达时长 / makespan |
| 少超时 | 相对订单承诺 `due_date` |
| 骑手不闲 | 最大化利用率 |
| 动态稳 | 骑手离线、紧急单、运力扰动下仍可完成 |

**不是**「端到端聊天/大模型」，而是 **调度决策层**。

---

## 3. 系统位置（并行双模型 + 外部融合）

```text
                    ┌─ 双塔（王新春侧）：骑手偏好维度 → 每骑手订单 list
业务输入 ──────────┤
                    └─ MAPPO（本仓库）：时间紧迫性维度 → 每骑手订单 list
                              ↓
              外部加权融合 → 每骑手实时订单 list（交付给调度/前端）
```

- **不是**串行下游，**不消费**双塔 `willingness`；两模型 **并行独立** 产出，在外部加权融合。
- 原表 **无**「骑手×订单」意愿字段（笛卡尔积需事后算）；偏好 **不进** MAPPO 观测/策略。
- 本仓库只做 **时间紧迫性** 派单（due/slack/time_pressure）；偏好归双塔。
- 历史（09-14/09-30）「上游→MAPPO 级联 + 意愿特征」口径 **已作废**；`upstream_candidates.willingness` 仅作实验注入位，**不入策略**。

---

## 4. 训练形式化（MAPPO / CTDE 在学什么 —— 架构主体不变）

| 维度 | 定义 |
|------|------|
| 范式 | **CTDE**：Actor 分布式（每骑手），Critic 集中式（全局状态） |
| 优化 | **MAPPO** = Multi-Agent PPO（clip + GAE）— **未改动** |
| 智能体 | 每骑手一个 Actor；**N 不是永远 5**（见下） |
| 状态 146 维 | 自身 8 + 全局 4 + 池摘要 30 + 候选 10×10 + 未来订单 4（**N=5 约定**） |
| 候选特征 [1] | **due_rel**（紧迫）；**无意愿/willingness** |
| 动作 | 每骑手：候选 10 选 1 或 IDLE |
| 奖励 | 送达 + 准时 − 超时 Huber − 闲置/无效 + 终局 bonus |

### 骑手数 N（回答「每一步还是五个骑手吗」）

| 场景 | N | 说明 |
|------|---|------|
| 146 维训练（现行默认） | **5** | one-hot/全局维写死；`max_riders=5` 切片 |
| 方案 B set 路径 | **可变**（已测 3/5/8） | mask 聚合，不限一人一码 |
| 业务站点 | 每站 N 在线骑手动态进出 | 复制权重到各站；P2 多尺度 N 采样 |
| 果洛样本 | 名单 200，当日完单 ~22 | 不是每 episode 都 5 个「真」骑手 ID |

**结论**：仿真决策步里在线骑手 = 当前 env 配置的 agent 数；**146 路径锁 5，set 路径 N 可变**。不是业务永远只有 5 人。

### 观测布局（勿随意改维）

```text
146 = 自身 8 + 全局 4 + 池摘要 30 + 候选 10×10 + 未来订单 4
      ↑ one-hot 5 绑定骑手数；改 N 会牵动网络 / BC 教师 / 146 约定
```

**骑手数 = 5 的原因（历史折衷）**

- 从工厂 5 工位对齐迁来；one-hot / 全局维写死宽度。
- 真实果洛有 ~200 骑手，仿真只注入 **前 5**（`real_data_loader.build_rider_configs(max_riders=5)`）。
- **业务上骑手应可变**；正确演进是 padding+mask 或集合编码（须重训）。未升级前 **不要改 max_riders**。

---

## 5. 两阶段训练在练什么

| 阶段 | 数据 | 目的 |
|------|------|------|
| **Foundation** | 随机单 / 果洛真实单 + mock 或上游候选 | 按时接单、少超时、送得完 |
| **Generalization** | + 骑手离线 / 紧急订单 | **鲁棒性**，不背样本 |

- 混训：`multi_task_mixing.base_worker_fraction`（当前 0.40），部分 worker 跑 BASE 锚点任务。
- 真实 xlsx：吃 **真实坐标、时间、完成分布**，不是理想 mock。
- Early-stop：基础连续 8 次、泛化连续 10 次达标可提前结束（`target_score` 等在 `DELIVERY_TRAINING_FLOW_CONFIG`）。

---

## 6. 评价口径（看训练是否有效）

**Score（0–1）** `calculate_delivery_episode_score`：

| 子项 | 权重 | 现行口径 |
|------|------|----------|
| 完成率 | 40% | completed / target |
| 准时 | 35% | **1 − 单均迟到/60min**（已按真实单修正；勿退回 `总延期/sim_time`） |
| makespan | 15% | 相对仿真窗 |
| 利用率 | 10% | 均值 |

**有效信号**：完成率↑、延期↓、score 尖峰↑、actor loss 有更新、entropy 不塌。  
**无效陷阱**：只比 reward 绝对值（真单/假单量纲不同）；「少送单换零延期」在旧评分下虚高。

分阶段看：BASE score 通常 > REAL；REAL 均分低不等于没学到。

---

## 7. 数据与路径

| 资产 | 路径 |
|------|------|
| 果洛全量样本 | `骑手派单仿真样本_果洛藏族自治州_20260915_全量.xlsx`（订单 1193 + 骑手 200） |
| 清洗 CSV | `data/real_incoming/orders_guoluo_*.csv` 等 |
| 字段映射 | `environments/real_data_schema.py` |
| 加载 | `environments/real_data_loader.py` → `custom_orders` + `riders` |
| 远端代码 | `/gemini/code/sftc-rider-dispatch`（**产物写 `runs/`，勿写 `/quota`**） |
| 本机备份 | `training_backup\` / `training_logs\` |

**训练必须带真实样本**（业务向）：

```bash
--real-orders "$XLSX" --real-riders "$XLSX" --episode-order-size 40
```

不加 `--real-orders` = 随机 mock，仅适合联调，**不算有效业务训练**。

**订单窗口采样口径（ready + due）**：`sample_order_window`  
- 窗口内 **ready 升序**（入池因果）  
- 选窗 **按 due/slack 紧迫度加权**（`prefer_urgent_window=True`），不能只看 ready  

---

## 7.5 最终展示：Live 调度大屏（定稿）

比赛/路演最终形态是 **实时调度大屏**，不是静态 PPT：

| 模块 | 内容 |
|------|------|
| 地图/站点 | 当前站点骑手位置、订单取送点、热力/拥堵 |
| 实时派单流 | 每骑手当前候选 list（融合后）、接单/送达事件 ticker |
| 双模型融合条 | 偏好分（双塔）vs 紧迫分（MAPPO）与融合权重 |
| KPI 面板 | 完成率 / 准时率 / 单均迟到 / 利用率 / 在线 N |
| 回放 | 拖动时间轴看策略决策序列（Gantt/时间线） |

实现要点：复用 `DeliveryEnv` 仿真或线上回调 → `supervisor` 指标流 → WebSocket/轮询前端；**输出语义 = 每骑手实时订单 list（已与双塔加权）**。  
源码落点建议：`supervisor/live_dashboard/`（后端）+ 大屏前端；与 `local_dashboard` 训练面板分开。

---

## 8. 运行 / 云环境约束（virtai / Gemini）

| 事实 | 约束 |
|------|------|
| 容器配额 | 以 **cgroup** 为准（勿信 `nproc`/`free` 的宿主机数） |
| 产物 | **与代码同路径** `runs/<stamp>_*`，跨实例可活 |
| `/quota` | 实例级，关机/重调度即清空 |
| ORION_TASK_IDLE_TIME | 约 1h 无**用户交互**会杀任务；后台 keepalive **无效**，需网页端偶发操作 |
| BrokenProcessPool | 多为 idle 杀 worker 的**结果**，不是 OOM（以 dmesg + 日志时序为准） |
| SSH | 密码易失败；优先网页终端 + JupyterLab 下载；启动用 `nohup` + `PYTHONUNBUFFERED=1` |
| worker | `num_parallel_workers` 建议 ≤ cgroup CPU 配额；`start_train_remote.sh` 会按核数写回，防覆盖需直接启 python |

推荐启动（网页终端）：

```bash
cd /gemini/code/sftc-rider-dispatch
# workers=5 写入 w_factory_config 后：
nohup /root/miniconda3/bin/python -u mappo/ppo_marl_train.py \
  --scenario delivery --candidate-source upstream \
  --real-orders "骑手派单仿真样本_果洛藏族自治州_20260915_全量.xlsx" \
  --real-riders "骑手派单仿真样本_果洛藏族自治州_20260915_全量.xlsx" \
  --episode-order-size 40 \
  --models-dir runs/$RUN/models --logs-dir runs/$RUN/logs \
  > runs/$RUN/train_full.log 2>&1 &
```

---

## 9. 代码地图

```text
environments/delivery_config.py   # 配送真理源：RIDERS/OBS/奖励/评分/上游契约
environments/delivery_env.py      # DeliverySim + DeliveryEnv（146 维观测）
environments/real_data_*.py       # 真实样本 schema/loader
mappo/ppo_marl_train.py           # CLI 入口（--scenario delivery / --real-*）
mappo/ppo_trainer.py              # MAPPO 训练循环 + 指标/回合日志
mappo/ppo_network.py              # Actor-Critic（146 入参）
supervisor/                       # 监督器、备份、启动脚本
runs/                             # 远端训练产物（代码同路径）
```

勿把工厂 `validate_config` 的「砂光机」横幅当配送诊断（delivery 已跳过，见 `ppo_trainer`）。

---

## 10. 已知限制 / 路线

| 限制 | 影响 | 路线 |
|------|------|------|
| 骑手固定 5 | 果洛 200 人只能切片 | 变长观测 padding+mask → 重训 |
| 真单 score 曾被延期打崩 | 策略「少送」 | 已改单均迟到；勿回退 |
| Idle 杀任务 | 长训易断 | 平台非交互作业 / 保持网页活跃 |
| 未 resume | 断点续训靠重跑 | 后续加 checkpoint resume |

---

## 11. Agent 协作约定

1. **每次对话先读 `project.md`**（原 AGENTS.md）；改训练目标、观测维、评分、数据源、启动方式、展示形态后 **立即更新本文件**。
2. 汇报结论区分：**mock 联调** vs **真实样本训练**；注明 run 目录与 episode 数。
3. 报资源用 **cgroup** 数字，不写宿主机 96 核/503G。
4. 不提交密钥；SSH 密码只走环境变量 `TRAIN_SSH_PASSWORD`。
5. 模型/日志只落 `runs/` 或本机 `training_backup/`，避免 `/quota`。

---

## 12. 变更日志（维护节）

| 日期 | 变更 |
|------|------|
| 2026-09-30 | **AGENTS.md → project.md**；全景图；确认 MAPPO/CTDE 未改；定稿 Live 调度大屏；订单采样 ready+due；去偏好（并行双塔） |
| 2026-09-28 | 删除过时 `update.md`/截断样本与本地缓存；口径以本文件为准；评分改单均迟到；episode_order_size=40 |
| 2026-09-27 | 产物改代码路径 runs/；监督器/备份；确认 ORION idle 为断训主因 |
| 2026-09-25 | 每回合完整日志 + metrics.jsonl；本机面板 |
| 2026-09-17 | 果洛全量 xlsx 接入；上游候选契约 |
