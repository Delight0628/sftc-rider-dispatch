# project.md — Rider Dispatch Hybrid（项目总图）

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
│  仿真层  DeliverySim / DeliveryEnv（未改）                        │
│          取→送两段、共享池、事件推进、due/slack 奖励              │
│          set 观测：rider[N,8] × cand[N,K,12] × global[5]         │
└───────────────┬─────────────────────────────┬───────────────────┘
                ▼                             ▼
┌──────────────────────────┐   ┌──────────────────────────────────┐
│ 训练层 Hybrid（本仓主干） │   │ 评估层 基线/KPI/离线对照          │
│ ① 边效用网络 q(骑手,订单) │   │ 完成率·准时·单均迟到·util·里程   │
│    集合编码+注意力(TF)    │   │ evaluation_delivery + checks      │
│ ② off-policy n-step TD    │   │ hybrid/evaluate 对拍矩阵          │
│ ③ 约束二分图匹配(不可微)  │   │ 铁律：必须赢 nearest/EDD/fifo    │
│ ④ 滚动重优化(事件驱动)    │   └──────────────────────────────────┘
└───────────────┬──────────┘
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  输出  每骑手「紧迫维度」订单 list                               │
│  ════════════════════════════════════════════════                │
│  外部融合（非本仓）：双塔偏好 list  ⊕  本仓紧迫 list              │
│                 → 实时订单 list → **Live 调度大屏**（最终展示）   │
└─────────────────────────────────────────────────────────────────┘
```

**范式（2026-10-08 起）**：**Hybrid = 学习边效用 + 约束二分图匹配 + 滚动重优化**（DiDi KDD 2018/2019/2022、Meituan SCDN 工业范式）。  
原 MAPPO/CTDE 栈已归档至 `archive/mappo/`（冻结，仅作历史对照，不再维护）。  
理论依据与路线对比见 `docs/research_dispatch_algorithm_survey.md`；实现细节见 `docs/hybrid_implementation.md`。

---

## 1. 项目一句话

用 **Hybrid 派单框架**（边效用网络 + off-policy TD + 二分图匹配）学出骑手派单策略：**多接、准点、少闲置**；只做**时间紧迫性**维度，偏好由并行双塔处理，结果外部加权，最终上 **Live 调度大屏**。

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
                    └─ Hybrid（本仓库）：时间紧迫性维度 → 每骑手订单 list
                              ↓
              外部加权融合 → 每骑手实时订单 list（交付给调度/前端）
```

- **不是**串行下游，**不消费**双塔 `willingness`；两模型 **并行独立** 产出，在外部加权融合。
- 原表 **无**「骑手×订单」意愿字段（笛卡尔积需事后算）；偏好 **不进** 本仓观测/打分。
- 本仓库只做 **时间紧迫性** 派单（due/slack/time_pressure）；偏好归双塔。
- 历史（09-14/09-30）「上游→策略级联 + 意愿特征」口径 **已作废**；`upstream_candidates.willingness` 仅作实验注入位，**不入打分**。

---

## 4. 训练形式化（Hybrid 在学什么）

| 维度 | 定义 |
|------|------|
| 范式 | **单智能体 RL（组合动作）+ 组合解码**：学习边效用，匹配层保证可行性 |
| 学习信号 | **off-policy n-step TD**（fitted-Q 式 MSE，n=16（1013 起），γ=0.99，目标网络软更新 τ=0.005） |
| 学的东西 | 边效用 `q(骑手i, 订单j | 全局上下文)`（不是策略 logits） |
| 决策 | 每个决策 epoch：边打分 → 约束二分图匹配（带容量分轮）→ 滚动重优化 |
| 状态 | set 观测：rider_feat `[N,8]` + cand_feat `[N,K,12]` + global_feat `[5]` + edge_feat `[N,K,12]` |
| 动作 | 匹配结果 → 每骑手候选下标动作（经 `env.step` 落地，FIFO carry） |
| 信用分配 | 边级：订单履约的 realized_utility 回填到派单时刻的边（非全局均摊） |
| 探索 | 温度 τ=0.5 softmax 边采样 + 贪心匹配（行为策略） |

### 骑手数 N

| 场景 | N | 说明 |
|------|---|------|
| Hybrid 主干 | **可变**（已测 3/5/8） | 集合编码 + mask 聚合，无 one-hot 宽度约束 |
| 业务站点 | 每站 N 在线骑手动态进出 | 同一边效用网络直接泛化 |
| 果洛样本 | 名单 200，当日完单 ~22 | episode 窗口注入 |

### 观测布局

- **主干（hybrid）**：set 观测，N/K 可变，mask float32（1=有效），见 `environments/set_obs.py`。
- **146 维固定观测**：随 MAPPO 归档，仅 `DeliveryEnv` 保留产出能力（历史对照），**不再用于训练**。

---

## 5. 训练在练什么

| 阶段 | 数据 | 目的 |
|------|------|------|
| **Foundation** | 随机单 / 果洛真实单窗口 | 按时接单、少超时、送得完 |
| **鲁棒性** | + 骑手离线 / 紧急订单（域随机化） | 不背样本 |

- 真实 xlsx：吃 **真实坐标、时间、完成分布**，不是理想 mock。
- Early-stop：对拍评估连续 2 轮不赢 linear hybrid 基线 → 回滚最近赢的 checkpoint。
- 评估铁律：**neural ≥ linear hybrid > 全部启发式（nearest/EDD/fifo）** 才算有效。

### 5.1 训练路线、基准与实测（2026-10-09 ~ 10-10 四轮迭代）

**参照基准（对拍矩阵，同口径同解码器）**：

| 基准 | 定义 | 来源 |
|------|------|------|
| edd | 最早送达优先启发式派单 | 经典规则基线，不学习 |
| linear | 线性边效用 + **同一**贪心匹配解码器 | 相同边特征拟合线性 q(i,j)，隔离「学习打分」本身的贡献 |
| neural | EdgeScorer（MLP）+ 同一解码器 | 训练主体 |

**目标定义**：neural eval 得分 ≥ linear，逼近上限。得分 = 边效用 `realized_utility = 1 − w_tard·tard/60 − w_dist·dist/20`（未履约记 −1）；果洛真实数据 + 固定 eval seed（seed_base=90000）保证轮间严格可比。

**四轮实测轨迹**（每轮 400 iters、12 ep/iter、6 workers、果洛真实单）：

| 轮次 | 方案 | eval 均值 | best | ≥linear | 结论 |
|------|------|-----------|------|---------|------|
| 1010 | 机制修正（lr 下限/init 回滚兜底/固定 eval seed/recent 采样） | 0.645 | **0.6726**@90 | 8/40 | 基线确立 |
| 1011 | 放宽保护（patience 4/grace 3）自 1010 best 续训 | 0.597 ✗ | — | 0/40 | TD 噪声主导，放宽方向错误 |
| 1012 | 收紧+降噪（lr 5e-4/updates 4/patience 2/best_score 继承） | 0.658 | 0.6726(继承) | 10/40 | +0.06，方向有效 |
| 1013 | 容量+目标（hidden 64→128/n_step 8→16/updates 2） | **0.665** | **0.6726**@80 | **13/40** | 优化侧到顶 |

**核心结论**：
1. TD 更新噪声主导训练净效应；0.64-0.67 高分带 = **回滚钉住效应**（保护机制防变坏有效，但高分靠钉住幸运快照而非学习爬升）。
2. best 连续两轮钉死 0.6726（容量翻倍+目标加倍+噪声减半仍不破）→ 上限来自**学习目标/特征本身**（realized_utility 口径、12 维边特征信息量），非模型容量或优化侧。
3. 边际收益 +0.06 → +0.007 → 0，优化侧（lr/容量/采样/保护）手段穷尽，收尾。

**下阶段（范式级，三分支实验）**：① realized_utility 加分段 shaping（取餐/送达中间奖励）；② 边特征工程（ETA 拥堵/骑手疲劳/订单簇）；③ n-step 换 eligibility/λ-return。
实施路线：三个 git 分支各实现一项 → 逐一上服务器完整训练（400 iters）实时盯盘 → 三方向交叉对比 → **融合最佳组合再训练一轮** → 全部产物归档本地。
详细记录：`docs/draft_hybrid_tuning_1011.md`。

### 5.2 三分支范式实验与融合（2026-10-11，guoluo_hybrid_1014{a,b,c}）

协议同 5.1（400 iters × 12 ep、6 workers、seed 0、固定 eval seed），1013 配置基座，唯一变量 = 各分支范式改动。分支代码：`exp-a-shaping` / `exp-b-features` / `exp-c-lambda`（校验 39/37/36 全绿）。

| 分支 | 改动 | best | 均值 | ≥linear | 0.673触及 | 判定 |
|------|------|------|------|---------|-----------|------|
| exp-a | realized_utility 取餐/送达分段 shaping（w_pick=0.3） | 0.6726@60 | 0.653 | 2/40 | 1 | 目标侧无增益 |
| exp-b | 边特征 12→16 维（拥堵ETA/送达簇/取餐竞争/骑手疲劳） | **0.6726@30** | 0.660 | **21/40** | **18** | **最强方向** |
| exp-c | n-step→TD(λ) 截断混合 bootstrap（λ=0.8） | 0.6726@150 | — | — | 1 | ~209 中断* |

*exp-c 训练至 iter 233 时实例到租用时长被回收，断点产物已归档本地 `runs/guoluo_hybrid_1014c/`。

**上限归因闭环**：0.6726 在 1010/1013/exp-a/exp-b/exp-c/exp-fusion 六个 run 中被精确复现——优化侧（容量/噪声）、目标侧（shaping/λ-return）、特征侧（16 维）及组合均不破。上限由**学习目标口径（realized_utility）与任务可辨识度**决定。
**增益方向**：特征工程显著提升高位带占领频率（≥linear 13/40→21/40、0.673 触及 10→18 次、触顶 @30）。

**融合终局（1015f，exp-fusion = exp-b × exp-a，400/400 完赛）**：best 0.6726（第六次复现）、均值 0.639、≥linear 11/40、0.673 触及 10 次——**未超越 exp-b 单独**，shaping 叠加 16 维特征反而稀释特征增益（目标侧/输入侧互相干扰）。
**最终最佳方案 = exp-b 单独（16 维边特征 + 原 realized_utility 口径）**，分支 `exp-b-features`，权重 `runs/guoluo_hybrid_1014b/checkpoints/edge_net_best.npz`。全部产物（1014a/1014b/1014c/1015f）已归档本地。

### 5.3 新数据接入、口径订正与真实订单流路线（2026-10-13 ~ 10-14）

> ⚠️ **口径订正（最重要）**：§5.1/§5.2 全部 run（1009-1015f）**及 1016** 的分数均为**合成订单口径**——
> `real_order_pool` 无消费者 bug（真实订单池透传 env_config 却无消费点，DeliveryEnv 只认 `custom_orders`，
> episode 退回合成随机订单）。2026-10-14 修复：`hybrid/collect.py::resolve_env_config`（池→
> `sample_order_window` 窗口注入），collect/evaluate 双侧接线，回归 31/31。
> **横向对比一律以 1017 起的真实口径为准**，§5.1/§5.2 的 0.6726 等数字仅作历史记录。详见 `docs/draft_hybrid_tuning_1011.md` §9。

**新数据**：王新春交付 `rl_hybrid_dispatch_data_20261008.csv`（2026-10-08 快照，30000 单×72 列、全国 20+ 城、
真实 expect_time 口径）；接入时修复 3 坑（`\\N` 空值短路 due 链 / csv.Sniffer 失败 / 跨城坐标拉爆投影→`--real-city` 单城过滤）。

| 轮次 | 数据/口径 | best | 均值 | ≥linear | 结论 |
|------|-----------|------|------|---------|------|
| 1016 | 上海新数据×16维，**合成订单口径** | 0.8259@40 | 0.810 | 11/40 | 数字作废（L=0.820/edd=0.595 为合成基线，暴露 bug 的诊断证据） |
| 1017 | 同协议，**真实订单流**（修复后重训） | **0.5616@370** | 0.541 | **37/40** | 真实口径基线 L=0.531/edd=0.496 |

**1017 判定**（400/400 完赛、11282s、实时盯盘全程）：真实订单流下 neural 稳定超越 linear（37/40、均值 +0.030），
16 维特征方案价值成立；best 在 iter 370 末期仍突破（0.55+ 触及 8 次），temp=0.1 低探索期有爬升空间。

**验证集独立评估**（`rl_train20_20261008.csv` = 全量 20% 随机子集 random_state=42，6000 单；上海 401 单、
20 ep × 40 单/ep）：neural（1016 权重）**0.6683** vs linear 0.6209（**+0.0474**、17/20 ep 胜）、
单均迟到 −7.9min、距离 −3%；完整结果 `runs/eval_verify_1016/eval_train20_results.json`。
注：验证集与训练池**同源非独立 held-out**（口径如实标注）；该验证用 1016 权重（合成口径训练），
**1017 权重复测是下一步行动①**。

### 5.4 下阶段计划与下一步具体行动（2026-10-20 制定）

**总体目标**：在真实订单流口径下，把 neural 对 linear 的稳定优势（1017: 37/40、+0.030）放大到
可上线水平，并完成生产候选替换评估。**唯一已验证有效方向 = 特征工程**（5.2 归因 + 5.3 真实口径复现）。

| # | 行动 | 具体做什么 | 验收标准 |
|---|------|-----------|---------|
| ① | **1017 权重验证集复测**（立即） | 用 `runs/shanghai_hybrid_1017/models/checkpoints/edge_net_best.npz` 替换 eval_verify 的 MODEL_NPZ，在 rl_train20 上海 401 单上重跑 20 ep 对拍 | neural 对 linear 差值 ≥ +0.04，复现 37/40 优势；与 1016 权重（0.6683）对比记录 |
| ② | **验证集多城扩展** | 去掉 `--real-city` 限制或分城跑（北京/深圳/广州…），验证 16 维特征跨城泛化 | ≥3 城对拍 neural 仍超 linear |
| ③ | **边特征工程深化**（主攻方向） | 在 16 维基础上加：骑手历史准时率、订单时空簇密度、区域运力缺口 | 1017 真实口径 best 0.5616 突破 0.58 |
| ④ | **学习目标口径调优** | realized_utility 尝试分段 shaping（5.2 exp-a 无增益但当时是合成口径，真实口径下需复验） | 均值 ≥ 0.55 且 ≥linear 不低于 37/40 |
| ⑤ | **生产候选替换评估** | 1017 best 权重导出 `edge_net_best.npz` + 评估报告，交付双塔融合侧做 A/B | 融合后调度 KPI 不降 |
| ⑥ | **产物与文档归档** | runs/ 全量备份本机 `training_backup\`；draft §9 订正同步 push GitHub | git 与远端一致 |

**执行顺序**：① → ③（并行②）→ ④ → ⑤ → ⑥。
**原则**：一切对比以 1017 起真实口径为准；特征工程优先，优化侧（lr/容量/保护）已验证到顶不再投入。

---

## 6. 评价口径（看训练是否有效）

**Score（0–1）** `calculate_delivery_episode_score`：

| 子项 | 权重 | 现行口径 |
|------|------|----------|
| 完成率 | 40% | completed / target |
| 准时 | 35% | **1 − 单均迟到/60min**（已按真实单修正；勿退回 `总延期/sim_time`） |
| makespan | 15% | 相对仿真窗 |
| 利用率 | 10% | 均值 |

**有效信号**：完成率↑、延期↓、score↑、TD loss 下降不 NaN、对拍矩阵中 neural 名次前移。  
**无效陷阱**：只比 reward 绝对值（真单/假单量纲不同）；「少送单换零延期」在旧评分下虚高。

分阶段看：mock score 通常 > 真实样本；真实均分低不等于没学到。

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
| 双模型融合条 | 偏好分（双塔）vs 紧迫分（本仓 hybrid）与融合权重 |
| KPI 面板 | 完成率 / 准时率 / 单均迟到 / 利用率 / 在线 N |
| 回放 | 拖动时间轴看策略决策序列（Gantt/时间线） |

实现要点：复用 `DeliveryEnv` 仿真或线上回调 → `supervisor` 指标流 → WebSocket/轮询前端；**输出语义 = 每骑手实时订单 list（已与双塔加权）**。  
源码落点建议：`supervisor/live_dashboard/`（后端）+ 大屏前端；与 `local_dashboard` 训练面板分开。

**已实现（2026-10-21，`supervisor/live_dashboard/`）**：

```bash
python supervisor/live_dashboard/server.py --port 8788   # 打开 http://127.0.0.1:8788
```

- `sim_stream.py`：真实数据 episode 帧流（`resolve_env_config` 口径 + 1017 best 权重 16 维边效用），
  逐决策帧捕获骑手/订单/候选边/派单链路/KPI/甘特/事件；**融合公式 fused = w_p·偏好 + w_u·紧迫(q 归一)**
  （偏好源 `_willingness_for` 契约口：双塔命中→willingness，未覆盖→属性兜底，帧内标注 pref_src）。
- `server.py`：HTTP API（`/api/pack` 数据、`/api/regen` 换 seed 重推演）+ HTML 数据注入（单文件可离线复看）。
- `static/index.html`：「暗夜调度指挥台」大屏——KPI 作战面板 / GEO 态势图（骑手脉冲点+取送点+派单链路流光动画）/
  双模型融合控制台（w 滑杆实时重排+偏好来源统计）/ 融合派单流（双色融合条 TOP 候选+满载时待派池队列兜底）/
  事件 ticker / 底部甘特回放（时间轴拖动+播放/倍速，空格/方向键快捷键）。
- 双塔（王新春侧）接入点：`generate_mock_upstream_candidates`（契约 `{骑手:[{order_id,willingness}]}`），
  真实上游直接替换该数据源即可，其余链路零改动。

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
| worker | `--num-parallel-workers` 建议 ≤ cgroup CPU 配额（start_train_remote.sh 自动探测） |

推荐启动（网页终端）：

```bash
cd /gemini/code/sftc-rider-dispatch
bash supervisor/start_train_remote.sh
# 或手动：
nohup /root/miniconda3/bin/python -u hybrid_train.py \
  --real-orders "骑手派单仿真样本_果洛藏族自治州_20260915_全量.xlsx" \
  --real-riders "骑手派单仿真样本_果洛藏族自治州_20260915_全量.xlsx" \
  --episode-order-size 40 \
  --models-dir runs/$RUN/models --logs-dir runs/$RUN/logs \
  > runs/$RUN/train_full.log 2>&1 &
```

---

## 9. 代码地图

```text
environments/delivery_config.py   # 配送真理源：RIDERS/OBS/奖励/评分/HYBRID_*_CONFIG
environments/delivery_env.py      # DeliverySim + DeliveryEnv（仿真层，未改）
environments/hybrid_dispatch.py   # 边打分+二分图匹配+滚动重优化（推理主干，零 TF 依赖）
environments/set_obs.py           # set 观测导出（含 build_all_pairs_set_obs）
environments/real_data_*.py       # 真实样本 schema/loader
hybrid/set_encoder.py             # 集合编码算子（TF + numpy 双实现）
hybrid/edge_value_net.py          # 边效用网络（TF/Keras + npz 导出降级）
hybrid/scorers.py                 # Linear / Neural EdgeScorer（统一接口）
hybrid/replay_buffer.py           # off-policy 经验回放（边级信用回填）
hybrid/collect.py                 # episode 采集（温度采样行为策略，可并行）
hybrid/trainer.py                 # n-step TD 训练循环 + early-stop 回滚
hybrid/evaluate.py                # 对拍评估（neural/linear/启发式同口径）
hybrid_train.py                   # CLI 入口（参数面对齐历史 ppo_marl_train.py）
evaluation_delivery.py            # 基线对拍（--baseline hybrid --hybrid-scorer …）
archive/mappo/                    # MAPPO/CTDE 历史栈（冻结，勿维护）
supervisor/                       # 监督器、备份、启动脚本
runs/                             # 远端训练产物（代码同路径）
```

勿把工厂 `validate_config` 的「砂光机」横幅当配送诊断（delivery 已跳过）。

---

## 10. 已知限制 / 路线

| 限制 | 影响 | 路线 |
|------|------|------|
| 匹配层不可微 | 不能端到端 policy gradient | 已采用 fitted-Q（边效用回归） |
| 真单 score 曾被延期打崩 | 策略「少送」 | 已改单均迟到；勿回退 |
| Idle 杀任务 | 长训易断 | 平台非交互作业 / 保持网页活跃 |
| 未 resume | 断点续训靠重跑 | 后续加 checkpoint resume |
| 合单/再平衡未建模 | 一单一人 | P4 分层（上层是否等待/合单） |

---

## 11. Agent 协作约定

1. **每次对话先读 `project.md`**；改训练目标、观测维、评分、数据源、启动方式、展示形态后 **立即更新本文件**。
2. 汇报结论区分：**mock 联调** vs **真实样本训练**；注明 run 目录与 episode 数。
3. 报资源用 **cgroup** 数字，不写宿主机 96 核/503G。
4. 不提交密钥；SSH 密码只走环境变量 `TRAIN_SSH_PASSWORD`。
5. 模型/日志只落 `runs/` 或本机 `training_backup/`，避免 `/quota`。
6. `archive/` 下代码冻结，只读不改；新功能一律在 `hybrid/` 与 `environments/` 演进。

---

## 12. 变更日志（维护节）

| 日期 | 变更 |
|------|------|
| 2026-10-21 | **Live 调度大屏落地（§7.5 实现）**：`supervisor/live_dashboard/`（sim_stream 帧流引擎 + server HTTP API + 「暗夜调度指挥台」单文件前端）；输出语义=每骑手融合订单 list（fused = w_p·双塔偏好 + w_u·紧迫 q，契约 `upstream_candidates` 可直接换真实上游）；真实数据+1017 权重推演 78 帧验证：完成率 95%、KPI/地图/融合滑杆/派单流/事件/甘特回放全交互通过 |
| 2026-10-20 | **§5.3 新数据真实口径路线 + §5.4 下阶段计划回写**：整理 1016（合成口径作废）→ 1017（真实口径 best 0.5616@370、≥linear 37/40）完整路线与 rl_train20 验证集独立评估（1016 权重 0.6683 vs linear 0.6209）；制定 6 项行动计划（1017 权重复测→特征工程深化→多城扩展→shaping 复验→生产候选→归档），执行顺序 ①→③(∥②)→④→⑤→⑥；git 清理实验分支后与远程同步（仅 main） |
| 2026-10-14 | **修复 real_order_pool 无消费者 bug 并重训 shanghai_hybrid_1017**：hybrid 链路把真实订单池透传 env_config 却无消费者（DeliveryEnv 只认 custom_orders），episode 退回合成订单；新增 `resolve_env_config`（池→sample_order_window 窗口注入），collect/evaluate 双侧接线，回归 31/31。诊断坐实：1016 的 eval 基线 L=0.820/edd=0.595 为合成口径，修复后 L=0.531/edd=0.496（真实订单流）。1017 同协议 400/400 完赛（11282s）：**best 0.5616@370**、均值 0.541、**≥linear 37/40**（L 均值 0.531）；产物 `runs/shanghai_hybrid_1017/` 归档，仪表盘并入 11 runs。**1016 及之前 run 的分数为合成订单口径，横向对比须以 1017 起为准** |
| 2026-10-13 | **接入王新春新数据 rl_hybrid_dispatch_data_20261008.csv（30000 单×72 列、全国 20+ 城、真实 expect_time 口径）并完成上海口径训练 shanghai_hybrid_1016**：特征核查满足要求；修复 `\\N` 空值短路 due 链（`_first_due` epoch 量级校验）、csv.Sniffer 失败（.csv 直读）、跨城坐标拉爆投影（`--real-city` 单城过滤）3 坑；exp-b 16 维方案 400/400 完赛，best **0.8259@40**、0.826 平台 10 次、均值 0.810、**≥linear 11/40**（linear 0.820/edd 0.595），形态与果洛一致→**方案泛化成立但上限形态不变**；产物归档 `runs/shanghai_hybrid_1016/`，仪表盘并入 10 runs；详见 `docs/draft_hybrid_tuning_1011.md` §8 |
| 2026-10-11 | **三分支范式实验并入 §5.2**：exp-a shaping 无增益 / exp-b 16 维特征最强（21/40、18 次触顶）/ exp-c λ-return 中断于 iter 233（断点产物已归档）；0.6726 六 run 复现 → 上限归因闭环；融合 exp-fusion（1015f，b×a）完赛未超越 exp-b 单独（0.639/11/40）→ **最终最佳 = exp-b 16 维特征**；全部产物归档本地 |
| 2026-10-10 | **四轮训练迭代（1010-1013）全过程并入 §5.1**：机制修正→放宽失败→收紧+降噪→容量+目标；best 钉死 0.6726，判定优化侧到顶，上限在学习目标/特征（realized_utility 口径、12 维边特征）；下阶段三分支范式实验（shaping / 边特征 / λ-return）→交叉对比→融合复训；详见 `docs/draft_hybrid_tuning_1011.md` |
| 2026-10-08 | **MAPPO/CTDE 完全替换为 Hybrid 派单框架**（边效用网络+off-policy TD+约束匹配+滚动重优化）；mappo/ 与 auto_train.py 归档至 archive/；训练入口改 `hybrid_train.py`；观测主干改 set 观测（N 可变）；依据 `docs/research_dispatch_algorithm_survey.md` |
| 2026-10-08 | **Hybrid 训练链路完成并全量校验通过**：hybrid/{edge_value_net,scorers,replay_buffer,collect,evaluate,trainer}.py + hybrid_train.py；checks 30/30、27/27、22/22、14/14；mock 冒烟 neural 0.9097≥linear 0.9095≫启发式 0.666；修复 warm-start ±x 双通道/目标网络硬拷贝/npz 键名/order_id int64 四个关键 bug |
| 2026-09-30 | **AGENTS.md → project.md**；全景图；定稿 Live 调度大屏；订单采样 ready+due；去偏好（并行双塔） |
| 2026-09-28 | 删除过时 `update.md`/截断样本与本地缓存；口径以本文件为准；评分改单均迟到；episode_order_size=40 |
| 2026-09-27 | 产物改代码路径 runs/；监督器/备份；确认 ORION idle 为断训主因 |
| 2026-09-25 | 每回合完整日志 + metrics.jsonl；本机面板 |
| 2026-09-17 | 果洛全量 xlsx 接入；上游候选契约 |
