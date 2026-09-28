# AGENTS.md — Rider Dispatch MAPPO

> **给后续对话中的 Agent**：开工前先读本文件；训练目的、观测约束、数据口径变更后**必须回写本节**。  
> 工作目录：`D:\rider-dispatch-mappo`（对应远端 `/gemini/code/sftc-rider-dispatch`）。

---

## 1. 项目一句话

在「双塔召回 → Wide&Deep 精排」给出的候选上，用 **多智能体 PPO（MAPPO）** 学出骑手派单策略：**多接、准点、少闲置，换人换急单也不崩**。

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

## 3. 系统位置（三层级联）

```text
双塔召回（想不想接）
  → Wide&Deep 精排（意愿 Top-K）
    → MAPPO（本仓库）：时间紧迫性优先的全局派单
```

- MAPPO **不替代**推荐；消费上游 `upstream_candidates`（`order_id` + `willingness`）。
- 会议口径（2026-09-14）：上游管 **意愿**，本层管 **时间关系**（urgency 优先于意愿序）。
- 契约见 `environments/delivery_config.py` §5.5；未注入真上游时用 mock 精排联调。

---

## 4. 训练形式化（MAPPO 在学什么）

| 维度 | 定义 |
|------|------|
| 智能体 | 每骑手一个 Actor（当前 **5** 人） |
| 状态 **146 维** | 自身 8 + 全局 4 + 池摘要 30 + 候选 10×10 + 未来订单 4 |
| 自身 8 维 | **骑手 one-hot(5)** + 载负荷 + 繁忙率 + 离线 |
| 全局态 | `1 + 2 + 5×3 + 1 = 19`（含骑手 one-hot 片段，见 env） |
| 动作 | `MultiDiscrete`：从候选 10 单选 1，或 IDLE |
| 环境 | SimPy：取餐 → 送餐两段动态工艺，共享待派池 |
| 奖励 | 送达 + 准时 − 超时 Huber − 闲置/无效动作 + 终局分数 bonus |
| 优化 | MAPPO（Actor-Critic，PPO clip + GAE） |

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

1. **每次对话先读本文件**；改训练目标、观测维、评分、数据源、启动方式后 **立即更新本文件**。
2. 汇报结论区分：**mock 联调** vs **真实样本训练**；注明 run 目录与 episode 数。
3. 报资源用 **cgroup** 数字，不写宿主机 96 核/503G。
4. 不提交密钥；SSH 密码只走环境变量 `TRAIN_SSH_PASSWORD`。
5. 模型/日志只落 `runs/` 或本机 `training_backup/`，避免 `/quota`。

---

## 12. 变更日志（维护节）

| 日期 | 变更 |
|------|------|
| 2026-09-28 | 建立 AGENTS.md；评分改单均迟到；真实单 episode_order_size=40；仿真窗自适应 |
| 2026-09-27 | 产物改代码路径 runs/；监督器/备份；确认 ORION idle 为断训主因 |
| 2026-09-25 | 每回合完整日志 + metrics.jsonl；本机面板 |
| 2026-09-17 | 果洛全量 xlsx 接入；上游候选契约 |
