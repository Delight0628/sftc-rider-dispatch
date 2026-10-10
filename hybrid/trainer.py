"""hybrid/trainer.py — HybridTrainer（off-policy n-step TD + 温度采样采集）

文档约定：docs/hybrid_implementation.md §4
- 范式：fitted-Q 式边效用回归（MSE，仅对被匹配边）
- 目标网络软更新；梯度裁剪；回滚 early-stop
- 采集 100% numpy（NpEdgeValueNet），训练机要求 TF
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from environments.delivery_config import HYBRID_TRAINING_CONFIG
from hybrid.collect import collect_episode_worker
from hybrid.replay_buffer import ReplayBuffer

try:
    import tensorflow as tf
    _HAS_TF = True
except Exception:  # pragma: no cover
    tf = None
    _HAS_TF = False

if _HAS_TF:
    from hybrid.edge_value_net import EdgeValueNet, NEG_INF


def _safe_makedirs(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


class HybridTrainer:
    """训练器：在线网络 + 目标网络 + replay buffer + 并行采集 + TD 更新。"""

    def __init__(self,
                 env_config: Dict[str, Any],
                 cfg: Optional[Dict[str, Any]] = None,
                 models_dir: Optional[str] = None,
                 logs_dir: Optional[str] = None,
                 seed: int = 0,
                 num_parallel_workers: int = 2,
                 init_linear: bool = True):
        if not _HAS_TF:
            raise RuntimeError("TensorFlow 未安装，无法训练。请确保 TF 2.15 可用。")
        self.env_config = dict(env_config)
        self.cfg = dict(cfg or HYBRID_TRAINING_CONFIG)
        self.seed = int(seed)
        self.num_parallel_workers = max(1, int(num_parallel_workers))
        self.init_linear = bool(init_linear)

        self.models_dir = models_dir or os.path.join(os.getcwd(), "models", "hybrid")
        self.logs_dir = logs_dir or os.path.join(os.getcwd(), "logs", "hybrid")
        _safe_makedirs(self.models_dir)
        _safe_makedirs(self.logs_dir)
        _safe_makedirs(os.path.join(self.models_dir, "checkpoints"))

        self.buffer = ReplayBuffer(capacity=self.cfg["buffer_size"], seed=self.seed)
        hidden = self.cfg["hidden_dim"]
        heads = self.cfg["num_heads"]

        self.online_net = EdgeValueNet(hidden=hidden, num_heads=heads)
        self.target_net = EdgeValueNet(hidden=hidden, num_heads=heads)
        # 建权重（dummy forward）
        self._build_net(self.online_net)
        self._build_net(self.target_net)

        if self.init_linear:
            self.online_net.warm_start_from_linear()
            print("✅ Warm-start：在线网络已初始化为线性先验")
        # 目标网络 = 在线网络硬拷贝（bootstrap 口径一致；不能只同步 warm-start，
        # 否则前层是两份独立随机初始化，训练解锁前层后 bootstrap 目标即错）
        self.target_net.set_weights(self.online_net.get_weights())

        # 断点续训：存在 edge_net_last 时加载（优先于 warm-start）
        last_npz = os.path.join(self.models_dir, "checkpoints", "edge_net_last.npz")
        if os.path.exists(last_npz):
            self.online_net.load_npz(last_npz)
            self.target_net.load_npz(last_npz)
            print(f"♻️ 断点续训：已加载 {last_npz}")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg["lr"])

        # init ckpt：warm-start / 续训起点权重显式留存，作为回滚兜底目标
        # （否则 best 从未触发时回滚是空操作，坏策略一路训到底）
        self._init_weights = [np.asarray(w).copy()
                              for w in self.online_net.get_weights()]
        self._save_ckpt("init")

        # 回滚 / best 状态（resume_best_score：续训时继承上一轮 best 基线，
        # 低于基线的 eval 不刷新 best，回滚目标锁定为高点权重）
        self.best_score = float(self.cfg.get("resume_best_score", -1e9))
        self.best_ckpt_base: Optional[str] = None
        self.poor_streak = 0
        self.grace_left = 0          # 新 best 后宽限 eval 轮数（期内跳过回滚保护）
        self.iter_count = 0

        # 日志
        self.metrics_path = os.path.join(self.logs_dir, "metrics.jsonl")
        self.tb_writer = tf.summary.create_file_writer(
            os.path.join(self.logs_dir, "train")) if _HAS_TF else None

    @staticmethod
    def _build_net(net: "EdgeValueNet") -> None:
        dummy = {
            "rider_feat": np.zeros((1, 1, 8), np.float32),
            "rider_mask": np.ones((1, 1), np.float32),
            "cand_feat": np.zeros((1, 1, 1, 12), np.float32),
            "cand_mask": np.ones((1, 1, 1), np.float32),
            "edge_feat": np.zeros((1, 1, 1, 12), np.float32),
            "global_feat": np.zeros((1, 5), np.float32),
        }
        net(dummy, training=False)

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------
    def train(self, max_iters: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if max_iters is None:
            max_iters = self.cfg.get("max_iters", 1000)
        max_iters = int(max_iters)
        temp_start = float(self.cfg["temp_start"])
        temp_end = float(self.cfg["temp_end"])
        temp_anneal = int(self.cfg["temp_anneal_episodes"])
        eval_every = int(self.cfg["eval_every"])
        rollback_patience = int(self.cfg["rollback_patience"])
        target_score = float(self.cfg.get("target_score", 0.85))
        early_stop_iters = int(self.cfg.get("early_stop_iters", 0))  # 0=禁用
        best_iter = 0

        print(f"🚀 HybridTrainer 启动 | max_iters={max_iters} | workers={self.num_parallel_workers}")
        t0 = time.time()
        for iter_idx in range(1, max_iters + 1):
            self.iter_count = iter_idx
            temp = max(temp_end, temp_start - (temp_start - temp_end)
                       * min(iter_idx, temp_anneal) / max(1, temp_anneal))

            # ---- 采集 ----
            collect_results = self._collect_iter(temp)
            n_trans = sum(len(t[0]) for t in collect_results)
            print(f"  iter {iter_idx:03d} | temp={temp:.3f} | 采集 {len(collect_results)} ep | "
                  f"transitions={n_trans} | buffer={len(self.buffer)}")

            # ---- 更新 ----
            loss_val, q_mean = self._update_iter()
            print(f"  iter {iter_idx:03d} | loss={loss_val:.4f} | q_mean={q_mean:.4f}")

            # ---- 日志 ----
            self._log_scalar(iter_idx, "loss", loss_val)
            self._log_scalar(iter_idx, "q_mean", q_mean)
            self._log_scalar(iter_idx, "temp", temp)
            self._log_scalar(iter_idx, "buffer_size", len(self.buffer))
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "iter": iter_idx, "temp": temp, "loss": loss_val,
                    "q_mean": q_mean, "transitions": n_trans,
                    "buffer": len(self.buffer),
                }, ensure_ascii=False, default=float) + "\n")

            # ---- 评估 ----
            eval_summaries = None
            if iter_idx % eval_every == 0 or iter_idx == max_iters:
                eval_summaries = self._eval_and_log(iter_idx)
                neural_score = eval_summaries.get("neural", {}).get("episode_score", -1.0)
                linear_score = eval_summaries.get("linear", {}).get("episode_score", -1.0)

                # 回滚检查
                if neural_score >= linear_score:
                    self.poor_streak = 0
                    if neural_score > self.best_score:
                        self.best_score = neural_score
                        self.best_ckpt_base = self._save_ckpt("best")
                        best_iter = iter_idx
                        self.grace_left = int(self.cfg.get("rollback_grace_evals", 0))
                        print(f"  🏆 新 best neural={neural_score:.4f} @ iter {iter_idx}"
                              f"（宽限 {self.grace_left} 轮）")
                else:
                    self.poor_streak += 1
                    print(f"  ⚠️ neural({neural_score:.4f}) < linear({linear_score:.4f}) "
                          f"连续 {self.poor_streak} 轮")
                    if self.poor_streak >= rollback_patience:
                        if self.grace_left > 0:
                            # 宽限期：新 best 后允许逃离其邻域，不回滚不减半
                            print(f"  🕊️ 宽限期（剩 {self.grace_left} 轮），跳过回滚保护")
                            self.poor_streak = 0
                        else:
                            # 回滚目标：best 优先，best 从未触发时回滚 init（兜底）
                            if self.best_ckpt_base is not None:
                                print(f"  🔙 回滚到 best ckpt ({self.best_ckpt_base})")
                                self._rollback(self.best_ckpt_base)
                            else:
                                print("  🔙 best 未触发，回滚到 init ckpt（warm-start 权重）")
                                self._rollback_to_init()
                            # lr 减半（下限保护：低于 lr_min 不再降，防学习冻结）
                            lr_min = float(self.cfg.get("lr_min", 1e-5))
                            cur_lr = float(self.optimizer.learning_rate)
                            new_lr = max(lr_min, cur_lr * 0.5)
                            if new_lr < cur_lr:
                                self.optimizer.learning_rate.assign(new_lr)
                                print(f"  📉 lr 减半 → {new_lr:.2e}")
                            else:
                                print(f"  ⛔ lr 已达下限 {lr_min:.2e}，保持不变")
                            self.poor_streak = 0
                if self.grace_left > 0:
                    self.grace_left -= 1

                # early-stop
                if early_stop_iters > 0 and (iter_idx - best_iter) >= early_stop_iters:
                    print(f"⏹️ Early stop：已连续 {early_stop_iters} 轮无提升")
                    break
                if neural_score >= target_score and self.poor_streak == 0:
                    print(f"✅ 达到 target_score={target_score:.3f}，保存最终模型")
                    self._save_ckpt("final")
                    break

        elapsed = time.time() - t0
        print(f"🎉 训练结束 | iters={self.iter_count} | best_score={self.best_score:.4f} "
              f"@ iter {best_iter} | elapsed={elapsed:.1f}s")
        return {
            "best_score": self.best_score,
            "best_iter": best_iter,
            "iters": self.iter_count,
            "elapsed": elapsed,
        }

    # ------------------------------------------------------------------
    # 采集
    # ------------------------------------------------------------------
    def _collect_iter(self, temp: float) -> List[Tuple[List[Dict], Dict]]:
        """并行采集 episodes_per_iter 条 episode。"""
        n_eps = int(self.cfg["episodes_per_iter"])
        hidden = self.cfg["hidden_dim"]
        heads = self.cfg["num_heads"]
        n_step = self.cfg["n_step"]
        gamma = self.cfg["gamma"]
        lam = float(self.cfg.get("lam", 0.0))

        # 导出当前在线网络为临时 npz（worker 纯 numpy 加载）
        temp_npz = None
        if self.online_net is not None:
            fd, temp_npz = tempfile.mkstemp(suffix=".npz", prefix="edge_net_temp_",
                                            dir=self.models_dir)
            os.close(fd)
            self.online_net.export_npz(temp_npz)

        payloads = [
            {
                "env_config": self.env_config,
                "scorer_kind": "npz" if temp_npz else "linear",
                "scorer_artifact": temp_npz,
                "temp": temp,
                "seed": self.seed + self.iter_count * 1000 + i,
                "n_step": n_step,
                "gamma": gamma,
                "lam": lam,
                "hidden": hidden,
                "num_heads": heads,
                "max_steps": 800,
            }
            for i in range(n_eps)
        ]

        results: List[Tuple[List[Dict], Dict]] = []
        try:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=self.num_parallel_workers,
                                     mp_context=ctx) as ex:
                futures = [ex.submit(collect_episode_worker, p) for p in payloads]
                results = [f.result(timeout=300) for f in futures]
        except Exception as e:
            print(f"⚠️ 并行采集失败，降级串行: {e}")
            traceback.print_exc()
            results = [collect_episode_worker(p) for p in payloads]
        finally:
            if temp_npz and os.path.exists(temp_npz):
                os.remove(temp_npz)

        for transitions, _stats in results:
            if transitions:
                self.buffer.add_episode(transitions)
        return results

    # ------------------------------------------------------------------
    # TD 更新
    # ------------------------------------------------------------------
    def _update_iter(self) -> Tuple[float, float]:
        """updates_per_iter 次 mini-batch TD 更新。"""
        updates = int(self.cfg["updates_per_iter"])
        batch_size = int(self.cfg["batch_size"])
        if len(self.buffer) < batch_size:
            return float("nan"), float("nan")

        total_loss = 0.0
        total_q = 0.0
        recent_frac = float(self.cfg.get("recent_frac", 0.0))
        recent_window = int(self.cfg.get("recent_window", 0))
        for _ in range(updates):
            batch = self.buffer.sample(batch_size, recent_frac=recent_frac,
                                       recent_window=recent_window)
            loss, qm = self._td_update(batch)
            total_loss += loss
            total_q += qm
        return total_loss / updates, total_q / updates

    def _td_update(self, batch: Dict[str, Any]) -> Tuple[float, float]:
        obs = batch["obs"]               # pad_collate dict
        edge_batch = tf.constant(batch["edge_batch"], dtype=tf.int32)
        edge_i = tf.constant(batch["edge_i"], dtype=tf.int32)
        edge_j = tf.constant(batch["edge_j"], dtype=tf.int32)
        utility = tf.constant(batch["utility"], dtype=tf.float32)
        boot_w = tf.constant(batch["boot_w"], dtype=tf.float32)      # [E,L]
        boot_g = tf.constant(batch["boot_g"], dtype=tf.float32)      # [E,L]

        indices = tf.stack([edge_batch, edge_i, edge_j], axis=1)
        bi_indices = tf.stack([edge_batch, edge_i], axis=1)

        with tf.GradientTape() as tape:
            q_online = self.online_net(obs, training=True)            # [B,N,K]
            q_pred = tf.gather_nd(q_online, indices)                  # [E]

            # λ-return 混合 bootstrap（exp-c）：
            # y = u + Σ_k w_k·γ^gap_k·max_q(s_{t+n_k})，各级目标网络前向
            y = utility
            for k, obs_k in enumerate(batch["obs_boots"]):
                q_t = self.target_net(obs_k, training=False)          # [B,N,K]
                boot_mask = tf.cast(obs_k["cand_mask"], q_t.dtype)    # [B,N,K]
                masked_q = q_t * boot_mask + (1.0 - boot_mask) * float(NEG_INF)
                max_q = tf.reduce_max(masked_q, axis=-1)              # [B,N]
                has_cand = tf.reduce_max(boot_mask, axis=-1)          # [B,N]
                max_q_bi = tf.gather_nd(max_q, bi_indices)
                has_cand_e = tf.gather_nd(has_cand, bi_indices)
                max_q_bi = tf.where(tf.cast(has_cand_e, tf.bool), max_q_bi, 0.0)
                y = y + boot_w[:, k] * boot_g[:, k] * max_q_bi
            loss = tf.reduce_mean(tf.square(q_pred - tf.stop_gradient(y)))

        grads = tape.gradient(loss, self.online_net.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, float(self.cfg["max_grad_norm"]))
        self.optimizer.apply_gradients(zip(grads, self.online_net.trainable_variables))

        # 软更新目标网络
        tau = float(self.cfg["target_soft_tau"])
        for t_var, o_var in zip(self.target_net.trainable_variables,
                                self.online_net.trainable_variables):
            t_var.assign(tau * o_var + (1.0 - tau) * t_var)

        return float(loss.numpy()), float(tf.reduce_mean(q_pred).numpy())

    # ------------------------------------------------------------------
    # 评估与日志
    # ------------------------------------------------------------------
    def _eval_and_log(self, iter_idx: int) -> Dict[str, Dict[str, float]]:
        from hybrid.evaluate import evaluate_all

        # 导出当前网络供评估（纯 numpy）
        eval_npz = os.path.join(self.models_dir, "edge_net_eval.npz")
        self.online_net.export_npz(eval_npz)

        summaries = evaluate_all(
            self.env_config,
            neural_artifact=eval_npz,
            episodes=int(self.cfg["eval_episodes"]),
            # 固定 seed 集：每轮 eval 同一订单流，跨 iter 曲线可比（消方差）
            seed=self.seed + int(self.cfg.get("eval_seed_base", 90000)),
            hidden=self.cfg["hidden_dim"],
            num_heads=self.cfg["num_heads"],
            baselines=("edd", "nearest", "fifo"),
            include_linear=True,
        )

        print(f"  📊 eval @ iter {iter_idx}")
        for name, s in summaries.items():
            sc = s.get("episode_score", 0.0)
            comp = s.get("completion_rate", 0.0)
            ot = s.get("on_time_rate", 0.0)
            print(f"    {name:10s} score={sc:.4f} comp={comp:.3f} on_time={ot:.3f}")
            self._log_scalar(iter_idx, f"eval/{name}_score", sc)
            self._log_scalar(iter_idx, f"eval/{name}_completion", comp)
            self._log_scalar(iter_idx, f"eval/{name}_on_time", ot)

        # 写入 metrics.jsonl
        row = {"iter": iter_idx, "eval": summaries}
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
        return summaries

    def _log_scalar(self, step: int, name: str, value: float) -> None:
        if self.tb_writer is None or math.isnan(value):
            return
        with self.tb_writer.as_default():
            tf.summary.scalar(name, float(value), step=step)

    # ------------------------------------------------------------------
    # checkpoint / 回滚
    # ------------------------------------------------------------------
    def _save_ckpt(self, tag: str) -> str:
        """保存 npz（numpy 推理）+ TF 权重。文件名兼容 Keras2/3（.weights.h5）。"""
        base = os.path.join(self.models_dir, "checkpoints", f"edge_net_{tag}")
        self.online_net.export_npz(f"{base}.npz")
        self.online_net.save_weights(f"{base}.weights.h5")
        return base

    def _rollback(self, ckpt_base: str) -> None:
        npz_path = f"{ckpt_base}.npz"
        if os.path.exists(npz_path):
            self.online_net.load_npz(npz_path)
            self.target_net.load_npz(npz_path)
        else:
            h5_path = f"{ckpt_base}.weights.h5"
            if os.path.exists(h5_path):
                self.online_net.load_weights(h5_path)
                self.target_net.load_weights(h5_path)

    def _rollback_to_init(self) -> None:
        """回滚到 warm-start / 续训起点权重（内存副本 + edge_net_init ckpt）。"""
        self.online_net.set_weights(self._init_weights)
        self.target_net.set_weights(self._init_weights)
