# 方案 B 实现计划：集合编码（DeepSets / Attention）变长骑手·订单

> 状态：**实现中**（代码见 `mappo/set_encoder.py`、`mappo/ppo_network_set.py`、`environments/set_obs.py`）。  
> 对照：`docs/design_variable_riders_orders.md`（草稿）；定稿后回写 `AGENTS.md`。  
> 原则：**不建模路径互扰**；订单保留 **due / 到达序**；旧 146 维路径**并行保留**。

---

## 0. 与 09-30 融合会的关系

| 会议结论 | 对方案 B 的含义 |
|---|---|
| 上游管「骑手接受维度」，调度管「时间紧迫性」 | 集合编码输入里 **willingness 与 slack/due 分槽**，不合成单一分数 |
| 融合：紧迫优先 + 少量偏好自由度 | Actor 对候选打分时 **urgency 特征进主干**；willingness 可进特征或轻量 bias（系数可学、默认小） |
| 「权重相加」未拍板 | **不在 MAPPO 外再做一层线性加权**；融合发生在策略网络内部（避免双算匹配、冗余） |

---

## 1. 目标

1. 骑手数 **N 可变**（站点内进出、上下线），不再绑死 one-hot(5)  
2. 订单池大小任意，动作仍是 **每骑手：候选 Top-K 选 1 或 IDLE**  
3. 观测对 N、K **排列不变**（set），可 zero-shot 到新站点人数  
4. 训练仍为 MAPPO（CTDE：分布式 Actor + 集中式 Critic）

**非目标（本阶段）**：路径互扰、跨站点全局优化、双塔/W&D 重训。

---

## 2. 问题拆解（相对 W-Factory / 现 146 维）

| 维度 | 现状（146） | 方案 B |
|---|---|---|
| 骑手身份 | one-hot(5) 写死宽度 | **属性向量 + Set 编码**，无身份维 |
| 骑手集合 | 全局态 `5×3` 拼死 | 对在线骑手 **mask 后聚合** |
| 订单 | 候选 10×10 扁平 | **订单候选 set** + 全局池 set |
| 时间序 | slack 特征 | 额外 **due 相对 / 到达 rank**（有序信息进特征，集合仍无序） |
| 动作 | MultiDiscrete([11]) | 保持每骑手动作维，**agent 集合可变** |

---

## 3. 架构

```text
                    ┌─────────────────────────────┐
  rider_i 属性 ──►  │ φ_r (MLP) → h_r_i            │
  其他骑手属性 ──►  │ DeepSets: ρ_r( Σ φ_r ) → g_r │──┐
                    └─────────────────────────────┘  │
                                                     ▼
  候选订单_j 特征 ► φ_o (MLP) ──► cross-attn(r_i, O) ──► Actor logits (K+1)
  池摘要 / 全局  ──► φ_g ──────────────────────────────┘
                                                     │
  所有骑手 h ─────► DeepSets g_all + 池 set ──────────► Critic V(s)
```

### 3.1 模块

| 模块 | 职责 |
|---|---|
| `environments/set_obs.py` | 从 `DeliverySim` 导出结构化 dict（骑手 set / 候选 set / 池摘要 / mask） |
| `mappo/set_encoder.py` | `phi` MLP、masked DeepSets、multi-head cross-attention |
| `mappo/ppo_network_set.py` | `PPONetworkSet`：Actor（每骑手）、Critic（全局集合） |
| `delivery_env` | `obs_mode='set'` 时返回 dict；默认仍 146 向量 |

### 3.2 张量约定（batch 维 B 可后加，先支持单 env / 步）

```text
rider_feat     : [N, D_r]     # 载荷、繁忙、离线、速度、capacity、home 相对坐标…
rider_mask     : [N]          # 1=在线可决策
cand_feat      : [N, K, D_c]  # slack, time_pressure, to_pickup, willingness, due_rel, rank, …
cand_mask      : [N, K]       # 有效候选
global_feat    : [D_g]        # 时间进度、池占比、运力饱和…
```

### 3.3 特征清单（ex-ante 优先）

**骑手 `D_r`**：`load, busy_ratio, offline, speed, capacity_norm, free_delay, x, y`  
**候选 `D_c`**：`exists, willingness, to_pickup, total_route, congestion, priority, is_urgent, type, time_pressure, slack, due_rel, arrive_rank_norm`  
**全局 `D_g`**：`time_prog, pool_ratio, saturation, pool_len, n_online_norm`

> **禁止**：`finish_time`、`is_timeliness`、`trade_amount` 等事后字段进网络输入。

---

## 4. 动作与训练

- 每骑手动作空间不变：`Discrete(K+1)`（0=IDLE）；mask 掉无效  
- 多骑手并行：`MultiDiscrete` 或 dict of Discrete；**N 可变 → agent 列表可变**  
- Critic：`g_all ⊕ g_pool ⊕ global` → V  
- 仍用 PPO clip + GAE；优势按 agent 计算  
- BC 教师（旧 146 索引）**不迁移到 set 路径**；set 路径从随机/规则教师冷启动

---

## 5. 里程碑

| 阶段 | 交付 | 验收 |
|---|---|---|
| **P0 本迭代** | set_obs + set_encoder + PPONetworkSet + env `obs_mode='set'` + 单测 | ✅ `checks/set_encoding_check.py` **25/25**（N=3/5/8；padding 无关；146 在 N=5 仍可用） |
| **P1** | trainer 接 `--obs-mode set`；worker 返回 dict obs；短训冒烟 | 5–20 episode loss 有限 |
| **P2** | 多尺度 N 采样（3–12）；站点切片数据 | N 变化下 eval 不崩 |
| **P3** | 与 146 路径 A/B；写回 AGENTS | KPI 不显著变差或更好 |

---

## 6. 风险与回退

| 风险 | 缓解 |
|---|---|
| 训练不稳 / 慢 | 保留 146 路径；set 仅 delivery `obs_mode` 开关 |
| 无 TF 本机只做结构单测 | `checks/set_encoding_check.py` 用 numpy 测聚合/mask；TF 构图在远端验 |
| 观测语义变 | 候选仍 10 维语义超集；K 不变 |
| 10-14 节点 | P0/P1 可并行进远端；路演仍可用 146 模型 |

---

## 7. 接口草图

```python
# env
env = DeliveryEnv({"scenario": "delivery", "obs_mode": "set", "max_riders": 8, "riders": ...})
obs, infos = env.reset()
# obs[agent] = {"rider_feat": ..., "cand_feat": ..., "global_feat": ..., "masks": {...}}

# net
net = PPONetworkSet(rider_dim=D_r, cand_dim=D_c, global_dim=D_g,
                    num_candidates=10, action_space=Discrete(11))
```

---

## 8. 变更记录

| 日期 | 说明 |
|------|------|
| 2026-09-30 | 初版计划；对齐 09-30「紧迫优先 + 少量偏好」融合口径 |
| 2026-09-30 | P0 落地：`mappo/set_encoder.py`、`mappo/ppo_network_set.py`、`environments/set_obs.py`；`DeliveryEnv.get_set_observation` / `obs_mode='set'`；`checks/set_encoding_check.py` 25/25 |
