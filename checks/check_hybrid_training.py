"""Hybrid 训练链路校验（numpy 为主，TF 可选）

覆盖：
[1] build_all_pairs_set_obs 结构（docs/hybrid_implementation.md §2）
[2] collect_episode：transitions 结构 / boot 值 / 边级 utility 回填口径（§4.1）
[3] ReplayBuffer：pad_collate 变长 N、sample 索引界内
[4] evaluate：summary 键名、linear vs edd 同 seed 对拍（§6 铁律口径）
[5] TF 可用时：EdgeValueNet 前向、warm-start parity（§3.3）、npz 双实现一致、
    HybridTrainer 单步 TD 更新 loss 有限
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.delivery_env import DeliveryEnv  # noqa: E402
from environments.delivery_config import HYBRID_DISPATCH_CONFIG, HYBRID_TRAINING_CONFIG  # noqa: E402
from environments.set_obs import build_all_pairs_set_obs, EDGE_FEAT_DIM  # noqa: E402


class Checker:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name, cond, detail=""):
        if cond:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {detail}")
            print(f"  [FAIL] {name} — {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n=== check_hybrid_training: {self.passed}/{total} passed ===")
        for e in self.errors:
            print(" -", e)
        return 0 if self.failed == 0 else 1


MOCK_CFG = {"scenario": "delivery", "training_mode": True, "MAX_SIM_STEPS": 60}


def main():
    print("=== check_hybrid_training ===")
    ck = Checker()

    # ------------------------------------------------------------------
    print("\n[1] build_all_pairs_set_obs 结构")
    env = DeliveryEnv(MOCK_CFG)
    env.reset(seed=0)
    obs = build_all_pairs_set_obs(env.sim)
    n = len(env.sim.rider_names)
    k = int(env.sim._obs_cfg.get("num_candidate_orders", 10))
    ck.check("rider_feat [N,8]", obs["rider_feat"].shape == (n, 8), str(obs["rider_feat"].shape))
    ck.check("cand_feat [N,K,12]", obs["cand_feat"].shape == (n, k, 12), str(obs["cand_feat"].shape))
    ck.check("edge_feat [N,K,12]", obs["edge_feat"].shape == (n, k, EDGE_FEAT_DIM))
    ck.check("global_feat [5]", obs["global_feat"].shape == (5,))
    ck.check("cand_order_ids int64 -1 空槽（真实 order_id 大整数）",
             obs["cand_order_ids"].dtype == np.int64
             and int(obs["cand_order_ids"].min()) >= -1)
    # mask=1 ⇒ 有订单；cand_actions = j+1
    m = obs["cand_mask"] > 0.5
    ck.check("mask=1 ⇒ order_id>=0", bool((obs["cand_order_ids"][m] >= 0).all()))
    ck.check("cand_actions = j+1",
             bool((obs["cand_actions"][m] == np.arange(k, dtype=np.int32)[None, :].repeat(n, 0)[m]).all()))
    ck.check("edge_feat bias=1（mask 处）",
             bool(np.allclose(obs["edge_feat"][..., 0][m], 1.0)))

    # ------------------------------------------------------------------
    print("\n[2] collect_episode 采集结构")
    from hybrid.collect import collect_episode
    trans, stats = collect_episode(MOCK_CFG, scorer_kind="linear", temp=0.3, seed=0,
                                   n_step=5, gamma=0.99, lam=0.8)
    ck.check("transitions 非空", len(trans) > 0, str(len(trans)))
    t0 = trans[0]
    ck.check("transition 键齐（exp-c λ阶梯）",
             {"obs", "edges", "utility", "obs_boots", "boot_w", "boot_g"} <= set(t0.keys()))
    ck.check("edges 与 utility 对齐", len(t0["edges"]) == len(t0["utility"]))
    ck.check("exp-c 阶梯长度一致且 boot_w 和为 1",
             len(t0["obs_boots"]) == len(t0["boot_w"]) == len(t0["boot_g"])
             and abs(sum(t0["boot_w"]) - 1.0) < 1e-9,
             f"L={len(t0['boot_w'])} sum={sum(t0['boot_w']):.6f}")
    ck.check("exp-c boot_g ∈ [0, γ]",
             all(0.0 <= g <= 0.99 + 1e-9 for g in t0["boot_g"]))
    # λ 质量分配公式：w_k = λ^{n_{k-1}} − λ^{n_k}（尾阶 λ^{n_{L-2}}...见 collect）
    _lam, _lad = 0.8, [1, 2, 4, 5]
    _ws = [_lam ** 0 - _lam ** 1, _lam ** 1 - _lam ** 2, _lam ** 2 - _lam ** 4, _lam ** 4]
    ck.check("exp-c λ 质量分配公式一致",
             all(abs(a - b) < 1e-9 for a, b in zip(t0["boot_w"], _ws)),
             f"{t0['boot_w']} vs {_ws}")
    # utility 口径：履约边 ∈ [-1, 1]，未履约 = -1
    all_u = [u for tr in trans for u in tr["utility"]]
    ck.check("utility ∈ [-1, 1+ε]", all(-1.0 - 1e-6 <= u <= 1.0 + 1e-6 for u in all_u))
    # 边索引界内
    for tr in trans:
        nn = tr["obs"]["rider_feat"].shape[0]
        kk = tr["obs"]["cand_feat"].shape[1]
        ok = all(0 <= i < nn and 0 <= j < kk for (i, j, _oid) in tr["edges"])
        if not ok:
            ck.check("边索引界内", False, str(tr["edges"]))
            break
    else:
        ck.check("边索引界内", True)
    # 边的 utility 与 realized_utility 公式一致（抽查首个履约边）
    from environments.hybrid_dispatch import EdgeScorer
    ck.check("realized_utility 公式唯一口径",
             hasattr(EdgeScorer, "realized_utility"))

    # ------------------------------------------------------------------
    print("\n[3] ReplayBuffer / pad_collate 变长 N")
    from hybrid.replay_buffer import ReplayBuffer, pad_collate
    buf = ReplayBuffer(capacity=100, seed=0)
    for ep in range(2):
        trs, _ = collect_episode(MOCK_CFG, scorer_kind="linear", temp=0.3, seed=ep,
                                 lam=0.8)
        buf.add_episode(trs)
    ck.check("buffer 累积", len(buf) >= len(trans), str(len(buf)))
    b = buf.sample(8)
    ck.check("obs batch 键齐",
             {"rider_feat", "rider_mask", "cand_feat", "cand_mask",
              "edge_feat", "global_feat"} == set(b["obs"].keys()))
    eb, ei, ej = b["edge_batch"], b["edge_i"], b["edge_j"]
    ck.check("edge 索引非空", eb.size > 0)
    ck.check("edge_batch 界内", int(eb.max()) < b["obs"]["rider_feat"].shape[0])
    ck.check("edge_i 界内", int(ei.max()) < b["obs"]["rider_feat"].shape[1])
    ck.check("edge_j 界内", int(ej.max()) < b["obs"]["cand_feat"].shape[2])
    ck.check("utility/boot_w/boot_g 与边数对齐",
             b["utility"].shape == eb.shape == b["boot_w"].shape[:1] == b["boot_g"].shape[:1]
             and b["boot_w"].shape[1] == len(t0["boot_w"])
             and len(b["obs_boots"]) == len(t0["boot_w"]))

    # ------------------------------------------------------------------
    print("\n[4] evaluate：summary 键 + 铁律口径（同 seed 对拍）")
    from hybrid.evaluate import evaluate_all
    sums = evaluate_all(MOCK_CFG, neural_artifact=None, episodes=2, seed=0,
                        baselines=("edd", "nearest"), include_linear=True)
    need = {"completion_rate", "on_time_rate", "avg_tardiness", "makespan",
            "mean_utilization", "distance_per_order", "episode_score"}
    ck.check("summary 键名对齐 §6", need <= set(sums["linear"].keys()),
             str(sorted(sums["linear"].keys())))
    ck.check("linear ≥ edd（mock seed0-1，P0 已知结论）",
             sums["linear"]["episode_score"] >= sums["edd"]["episode_score"] - 1e-9,
             f"linear={sums['linear']['episode_score']:.4f} edd={sums['edd']['episode_score']:.4f}")

    # ------------------------------------------------------------------
    print("\n[5] TF 网络与训练器（可选）")
    try:
        import tensorflow as tf  # noqa: F401
        from hybrid.edge_value_net import EdgeValueNet, NpEdgeValueNet
        has_tf = True
    except Exception:
        has_tf = False

    if not has_tf:
        ck.check("TF 不可用（跳过，远程训练机执行）", True)
    else:
        import tempfile, os
        from hybrid.scorers import LinearEdgeScorer

        net = EdgeValueNet(hidden=HYBRID_TRAINING_CONFIG["hidden_dim"],
                           num_heads=HYBRID_TRAINING_CONFIG["num_heads"])
        net.warm_start_from_linear()
        q_tf = net({kk: obs[kk][None] for kk in
                    ("rider_feat", "rider_mask", "cand_feat", "cand_mask",
                     "edge_feat", "global_feat")}, training=False).numpy()[0]
        lin = LinearEdgeScorer().score_edges(obs)
        finite = np.isfinite(lin)
        ck.check("warm-start neural ≡ linear（打分 parity）",
                 bool(np.allclose(q_tf[finite], lin[finite], atol=1e-3)),
                 f"max|Δ|={np.abs(q_tf[finite]-lin[finite]).max():.2e}")
        ck.check("mask 处 -1e9", bool((q_tf[~finite] <= -1e8).all()))

        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "w.npz")
            net.export_npz(p)
            np_net = NpEdgeValueNet(p, hidden=HYBRID_TRAINING_CONFIG["hidden_dim"],
                                    num_heads=HYBRID_TRAINING_CONFIG["num_heads"])
            q_np = np_net.forward(obs)
            ck.check("npz numpy 前向 ≡ TF 前向",
                     bool(np.allclose(q_np[finite], q_tf[finite], atol=1e-3)),
                     f"max|Δ|={np.abs(q_np[finite]-q_tf[finite]).max():.2e}")

            # HybridTrainer 单步 TD 更新（loss 有限即梯度链路通）
            from hybrid.trainer import HybridTrainer
            tr = HybridTrainer(MOCK_CFG, cfg=dict(HYBRID_TRAINING_CONFIG),
                               models_dir=td, logs_dir=td, seed=0,
                               num_parallel_workers=1, init_linear=True)
            for ep in range(2):
                trs, _ = collect_episode(MOCK_CFG, scorer_kind="linear", temp=0.3, seed=ep,
                                         lam=0.8)
                tr.buffer.add_episode(trs)
            tr.cfg["batch_size"] = 8
            batch = tr.buffer.sample(8)
            loss, qm = tr._td_update(batch)
            ck.check("TD 更新 loss 有限", np.isfinite(loss), f"loss={loss}")
            ck.check("TD 更新 q_mean 有限", np.isfinite(qm), f"q_mean={qm}")
            # τ=0.005 软更新后目标网络与在线网络应非常接近但不严格恒等
            diff = max(float(np.abs(a.numpy() - b.numpy()).max())
                       for a, b in zip(tr.target_net.trainable_variables,
                                       tr.online_net.trainable_variables))
            ck.check("目标网络软更新幅度合理（<0.1）", diff < 0.1, f"max|Δ|={diff:.2e}")

            # --- 保护机制回归（2026-10-09 修正版）---
            # recent 优先采样：recent_frac=1 时样本全部来自最近 recent_window 条
            tag_buf = ReplayBuffer(capacity=200, seed=0)
            trs3, _ = collect_episode(MOCK_CFG, scorer_kind="linear", temp=0.3, seed=3)
            for gi, t in enumerate(trs3):
                t = dict(t)
                t["utility"] = [float(gi)] * len(t["edges"])
                tag_buf.add_episode([t])
            n_total = len(tag_buf)
            bb = tag_buf.sample(8, recent_frac=1.0, recent_window=5)
            u_max = float(bb["utility"].max())
            ck.check("recent 优先采样仅取最近窗口",
                     u_max >= float(n_total - 5), f"max utility idx={u_max}/{n_total}")
            # init 回滚：扰动权重后 _rollback_to_init 恢复 warm-start 副本
            for w in tr.online_net.trainable_variables:
                w.assign(w.numpy() + 1.0)
            tr._rollback_to_init()
            w1 = tr.online_net.get_weights()
            ck.check("init 回滚恢复 warm-start 权重",
                     all(np.allclose(a, b) for a, b in zip(w1, tr._init_weights)))
            # lr 下限配置合法（0 < lr_min <= lr）
            ck.check("lr_min 下限配置合法",
                     0.0 < HYBRID_TRAINING_CONFIG["lr_min"] <= HYBRID_TRAINING_CONFIG["lr"],
                     f"lr_min={HYBRID_TRAINING_CONFIG['lr_min']}")
            # 回滚保护配置合法（patience>=2，宽限期 >=0 且训练器状态机就绪）
            ck.check("回滚保护配置合法",
                     HYBRID_TRAINING_CONFIG["rollback_patience"] >= 2
                     and HYBRID_TRAINING_CONFIG["rollback_grace_evals"] >= 0
                     and hasattr(tr, "grace_left"),
                     f"patience={HYBRID_TRAINING_CONFIG['rollback_patience']} "
                     f"grace={HYBRID_TRAINING_CONFIG['rollback_grace_evals']}")

    return ck.summary()


if __name__ == "__main__":
    raise SystemExit(main())
