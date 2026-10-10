# hybrid_implementation.md — Hybrid 派单框架实现文档

> 本文档是 Hybrid 派单框架的**技术实现约定**（网络结构 / 训练目标 / 张量形状 / 配置），与代码严格一致。
> 调研与路线论证见 `research_dispatch_algorithm_survey.md`；项目总图见 `project.md`。
> 代码注释中引用的维度、公式、参数名以本文档为准。

## 1. 总体架构

```
事件驱动决策点（DeliveryEnv 每 step = 一个 epoch）
  │
  ├─ 观测构造 build_all_pairs_set_obs(sim)
  │    rider_feat [N,8]  rider_mask [N]
  │    cand_feat  [N,K,12] cand_mask [N,K]   ← 全对候选（每骑手 top-K）
  │    global_feat [5]
  │
  ├─ 边效用网络 EdgeValueNet（hybrid/edge_value_net.py）
  │    h_i   = RiderSetEncoder(rider_feat, rider_mask, global_feat)   [N,D]
  │    e_ij  = OrderCandEncoder(query=h_i, cand_feat, cand_mask)      [N,K,D]
  │    q_ij  = MLP([h_i, e_ij, h_i⊙e_ij, edge_feat_ij])               [N,K]
  │
  ├─ 约束匹配 max_weight_matching（贪心 + 交换改进，environments/hybrid_dispatch.py）
  │    按容量分轮求解 = 带容量 b-matching；一对一、无抢单
  │
  └─ 动作落地 env.step({agent: [cand_action,...]})；未派订单留池滚动重优化
```

**范式定位**：单智能体 RL（组合动作空间）+ 组合解码器。不是 CTDE（无分散策略）；匹配层不可微，因此学习目标是 **边效用回归（fitted-Q 式 TD）**，不是 policy gradient。

## 2. 观测与特征约定

### 2.1 rider_feat `[N, 8]`（`environments/set_obs.py`，未改）
`[载负荷, busy率, 离线, speed/0.6, capacity/6, 压缩free_delay(60), x/grid, y/grid]`；`rider_mask`：1=在线可决策，离线=0。

### 2.2 cand_feat `[N, K, 12]`（K=`num_candidate_orders`=10）
`[exists, due_rel, to_pickup压缩, total_route压缩, congestion, priority, is_urgent, type_id/4, time_pressure, slack, due_rel(重复), rank]`。
- `build_set_obs`（已有）：仅 self 行非零（分布式视角）。
- **`build_all_pairs_set_obs`（新增）**：**每行都填**（集中式打分视角），`cand_mask[i,j]=1` 表示该槽有效且动作合法（对齐 action_mask）；返回额外键 `cand_order_ids [N,K] int32`（-1=空槽）与 `cand_actions [N,K] int32`（该槽对应的 env 动作号，0=无）。

### 2.3 edge_feat `[N, K, 12]`（与 `EdgeScorer.build_features` 逐维一致）
`[bias=1, order_urgency, on_time_feasible, late_norm, to_pickup_norm, route_norm, rider_load, free_delay_norm, priority_norm, is_urgent, congestion, slack_norm]`。
作用：① 可解释锚点；② warm-start 载体（见 §3.3）；③ 网络输出头的显式输入。

### 2.4 global_feat `[5]`
`[时间进度, 池占比, 运力饱和, 压缩池长(20), n/12]`。

## 3. 边效用网络（hybrid/edge_value_net.py）

### 3.1 结构（TF/Keras，隐藏维 D=64）
```
rider_emb = PhiMLP(rider_feat)                       # [B,N,64]
set_ctx   = MaskedDeepSets(rider_emb, rider_mask)    # [B,64]
h_i       = MLP([rider_emb, set_ctx[:,None,:], global_proj])  # [B,N,64]
e_ij      = CrossAttention(query=h_i, keys/values=φ(cand_feat), mask=cand_mask)  # [B,N,K,64]
q_ij      = MLP([h_i[:,:,None,:]broadcast, e_ij, h_i⊙e_ij, edge_feat]) → [B,N,K]
```
padding 处输出由 `cand_mask` 屏蔽，不进入匹配。

### 3.2 numpy 降级
`save_npz(path)` 导出全部权重；`np_forward(...)` 纯 numpy 前向（供无 TF 环境的推理/单测）。结构对齐：逐层 `relu(x@W+b)`，cross-attention 用 `np_cross_attention`。

### 3.3 Warm-start（起点不劣于线性先验）
输出头为两层 `q_head([3D+E]→D, relu) + q_out(D→1)`。由于 relu 截断负值（edge_feat 含负，如 slack_norm∈[-3,3]），采用 **±x 双通道**初始化：
- `q_head` kernel 仅 edge_feat 切片非零：第 d 维写入列 d（+1）与列 E+d（−1） → `qh[d]=relu(+x_d)`，`qh[E+d]=relu(−x_d)`；
- `q_out` kernel 前 2E 维为 `[w, −w]`，其余置零 → `q = Σ w_d·relu(x_d) − w_d·relu(−x_d) = Σ w_d·x_d`。
要求 `hidden ≥ 2E`（64 ≥ 24 ✓）。**零训练时 q ≡ 线性先验分**，NeuralEdgeScorer 初始行为与 LinearEdgeScorer 一致（parity 校验见 checks/check_hybrid_training.py [5]）。

### 3.4 Scorer 接口（hybrid/scorers.py）
```python
class BaseEdgeScorer:
    def score_edges(self, obs_pack) -> np.ndarray  # [N,K] 打分，padding 处 -inf
    def learnable(self) -> bool
class LinearEdgeScorer(BaseEdgeScorer)   # 包装 environments.hybrid_dispatch.EdgeScorer
class NeuralEdgeScorer(BaseEdgeScorer)   # 包装 EdgeValueNet（在线网络）；TF 缺失时抛错由调用方回退
```
`obs_pack` = `build_all_pairs_set_obs` 输出 + `edge_feat`。

## 4. 训练（hybrid/trainer.py）

### 4.1 经验结构（hybrid/replay_buffer.py）
每个决策 epoch 存一条 transition：
```
obs_pack_t, matched {(i,j): (order_id, edge_feat)}, obs_pack_{t+1}, done
```
**边级延迟信用回填**：episode 内维护 `order_id → (t, edge)`；订单履约/超时/episode 结束时按
`realized_utility = 1 - w_tard·max(0,迟到)/60 - w_dist·距离/20`（未履约 = -1）
回填为该边的即时回报 `r_e`。与 `EdgeScorer.realized_utility` 公式一致（delivery_config.HYBRID_DISPATCH_CONFIG）。

### 4.2 TD 目标（n-step，默认 n=5，γ=0.99）
对边 e=(i,j) 在时刻 t 被匹配：
```
y_e = Σ_{k=0}^{n-1} γ^k · r̃_{t+k}  +  γ^n · max_{e'∈候选(i,·)} q_target(e', obs_{t+n})
```
其中 `r̃` 为该边回填回报（未回填的 epoch 记 0）；若 episode 在 n 步内结束则 bootstrap 截断（done）。
**Loss**：`MSE(q_online(e, obs_t), y_e)`，仅对被匹配的边计算（匹配层当环境；广义贪心）。
目标网络软更新：`θ_target ← τ·θ_online + (1-τ)·θ_target`，τ=0.005。

### 4.3 行为策略（hybrid/collect.py）
温度采样：对每骑手边分按 τ_temp=0.5 做 softmax 采样一行 → 集合送入贪心匹配（保证可行性）。
探索随训练线性退火 τ: 0.5→0.1（`temp_anneal_episodes`）。

### 4.4 训练循环
```
init ckpt: warm-start / 续训起点权重显式留存（edge_net_init.* + 内存副本，回滚兜底目标）
for iter:
  并行采集 collect_episodes(n_episodes_per_iter, 行为策略)   # ProcessPoolExecutor，失败降级串行
  buffer.add(...)
  for _ in range(updates_per_iter):
      batch = buffer.sample(batch_size, recent_frac, recent_window)   # recent 优先混合采样
      td_update(batch)
      target_soft_update()
  if iter % eval_every == 0:
      metrics = evaluate_all(neural, linear, heuristics)       # hybrid/evaluate.py，固定 seed 集
      if neural < linear 连续 2 轮:
          回滚 best_ckpt（best 未触发时回滚 init ckpt）; lr = max(lr_min, lr*0.5)
      if 达标(可配置 target_score): 保存 best 并可 early-stop
```
产物：`<models_dir>/checkpoints/edge_net_{init,best,last}.npz` + `.weights.h5`、`edge_net_eval.npz`、`metrics.jsonl`、TensorBoard。

## 5. 配置（delivery_config.py）

```python
HYBRID_TRAINING_CONFIG = {
    "lr": 5e-4, "lr_min": 1e-5, "gamma": 0.99, "n_step": 8,
    "temp_start": 0.5, "temp_end": 0.1, "temp_anneal_episodes": 200,
    "buffer_size": 10000, "batch_size": 96,
    "recent_frac": 0.5, "recent_window": 2000,
    "target_soft_tau": 0.005,
    "hidden_dim": 64, "num_heads": 4,
    "episodes_per_iter": 4, "updates_per_iter": 4, "eval_every": 10,
    "eval_episodes": 3, "eval_seed_base": 90000,
    "rollback_patience": 2, "rollback_grace_evals": 0,
    "resume_best_score": -1e9,
    "max_grad_norm": 5.0,
}
```

## 6. 对拍与有效性铁律

- 评估统一走 `hybrid/evaluate.py` / `evaluation_delivery.py`，summary 键名一致：
  `completion_rate / on_time_rate / avg_tardiness / makespan / mean_utilization / distance_per_order / episode_score`。
- **铁律**：`neural ≥ linear hybrid > nearest/edd/fifo`（同订单集、同 seed、同 episode 数）。
- 汇报必须区分 mock 联调 vs 真实样本训练，注明 episode 数与 seed。

## 7. CLI（hybrid_train.py）

```
python hybrid_train.py [--episodes 200] [--seed 0]
  [--real-orders PATH] [--real-riders PATH] [--episode-order-size 40]
  [--num-parallel-workers N] [--models-dir DIR] [--logs-dir DIR]
  [--init-linear | --no-init-linear]
```
- `--episodes` = 训练迭代轮数（max_iters，每轮采集 `episodes_per_iter` 条 episode）。
- linear 基线不作为 `--scorer` 选项暴露：它不可训练，且每次评估已由
  `evaluate_all(include_linear=True)` 自动对拍（§6 铁律）。
- `--real-*` 缺省 = mock（仅联调，不算业务训练）。
- 评估入口：`evaluation_delivery.py --baseline hybrid --hybrid-scorer <npz路径|linear>`。

## 8. 风险与降级

| 风险 | 降级 |
|---|---|
| TF 不可用 | NeuralEdgeScorer 导入失败 → LinearEdgeScorer + warning；npz numpy 推理 |
| 并行采集失败 | 捕获 BrokenProcessPool → 单进程串行 |
| TD 发散 | loss NaN → 回滚 best + lr 减半 + 冻结 target 10 轮 |
| neural 不赢 linear | 连续 `rollback_patience` 轮不赢 → 回滚 best（未触发时回滚 init）+ lr 减半至下限 `lr_min`；`rollback_grace_evals` 轮宽限期可跳过回滚（默认关）；续训时 `resume_best_score` 继承 best 基线（低于基线不刷新 best，回滚锁定高点权重） |
