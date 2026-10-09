# 草稿：Hybrid 训练保护机制调参（2026-10-09/10，待确认后并入 project.md）

> 状态：草稿。按项目惯例，方案先记录于此，待用户确认后再更新 project.md。

## 1. 背景：两轮真实训练的观察

### 首轮（guoluo_hybrid_1009，200 iters，旧机制）
- neural 全程 0.555-0.660，0/20 轮赢 linear（0.701）
- 机制缺陷三连：lr 无下限减半锁死（≈4e-6）、best 从未保存致回滚空操作、eval seed 漂移不可比

### 次轮（guoluo_hybrid_1010，400 iters，修正机制：lr_min=1e-5 / init 回滚兜底 / 固定 eval seed / n_step=8 / recent 采样）
- **best 0.6726 @ iter 90**，final 0.666 > linear 0.662，8/40 轮 ≥ linear，40/40 轮赢启发式（0.542）
- 修复项全部按设计工作（lr 下限钳制日志、init 回滚生效、loss 降至 0.2-0.3）
- **新瓶颈**：30-iter 循环震荡（0.666→0.649→0.641→回滚到 best），回滚保护"防变坏"有效但**锁住上限**——
  权重每离开 best 邻域即被拉回，无法持续爬升突破 0.6726

## 2. 本轮改进（已实现）

| 配置 | 旧 | 新 | 依据 |
|---|---|---|---|
| `rollback_patience` | 2 | 4 | 30-iter 循环 = 2 轮即回滚；放宽到 4 轮（40 iters）给学习留空间 |
| `lr_min` | 1e-5 | 5e-5 | 钳制后步子过小（学习≈冻结）；提高下限保持续学习率 |
| `rollback_grace_evals` | （无） | 3 | 新 best 后 3 轮宽限期跳过回滚/减半，允许逃离 best 邻域探更高点 |

实现：`hybrid/trainer.py`（grace_left 状态机）、`environments/delivery_config.py`（配置）；
`docs/hybrid_implementation.md` §5/§8 已同步。

## 3. 本轮训练方案

- **起点**：预置 `edge_net_last.npz` = 1010 的 best（0.6726 权重）续训（新 run 目录，不混口径）
- 温度：temp_start=0.3 → 0.1（续训精炼为主，保留少量探索）
- 400 iters × 12 ep（6 workers × 2），B1.small
- run 目录：`runs/guoluo_hybrid_1011/`
- **终止判据（自主迭代）**：连续 2 轮训练 best 无提升 → 判定当前手段耗尽，停止并总结

## 4. 待确认事项

- [ ] 本调参方案并入 project.md（变更日志 + 配置表）
- [ ] 若 0.6726 突破：是否将该版本权重作为生产候选替换 linear 上线评估
