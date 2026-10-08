"""Hybrid 派单框架：神经边效用网络
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from environments.delivery_config import HYBRID_DISPATCH_CONFIG, HYBRID_TRAINING_CONFIG

try:
    import tensorflow as tf
    _HAS_TF = True
except Exception:  # pragma: no cover
    tf = None
    _HAS_TF = False

NEG_INF = -1e9


# =============================================================================
# 1. TensorFlow 模型
# =============================================================================
if _HAS_TF:

    class EdgeValueNet(tf.keras.Model):
        """q(骑手 i, 候选 j) 边效用网络。

        结构（§3.1）：
          rider_emb = PhiMLP(rider_feat)                      [B,N,D]
          set_ctx   = rho(masked_mean(rider_emb, mask))       [B,D]
          h_i       = MLP([rider_emb, set_ctx, global_proj])  [B,N,D]
          keys0     = PhiMLP(cand_feat)                       [B,N,K,D]
          ctx_i     = MultiHeadAttention(query=h_i, keys0, mask)  [B,N,D]
          e_ij      = keys0 + ctx_i[:,:,None,:]               [B,N,K,D]
          q_ij      = MLP([h_i, e_ij, h_i⊙e_ij, edge_feat])   [B,N,K]
        padding 处输出 -1e9（cand_mask 屏蔽）。
        """

        def __init__(self, hidden: int = 64, num_heads: int = 4,
                         name: str = "edge_value_net"):
            super().__init__(name=name)
            self.hidden = hidden
            self.num_heads = num_heads
            self.dk = max(4, hidden // max(1, num_heads))

            self.rider_phi = [
                tf.keras.layers.Dense(hidden, activation="relu", name="rider_phi_d0"),
                tf.keras.layers.Dense(hidden, activation="relu", name="rider_phi_d1"),
            ]
            self.rider_rho = tf.keras.layers.Dense(hidden, activation="relu", name="rider_rho")
            self.global_proj = tf.keras.layers.Dense(hidden, activation="relu", name="global_proj")
            self.h_head = tf.keras.layers.Dense(hidden, activation="relu", name="h_head")
            self.cand_phi = [
                tf.keras.layers.Dense(hidden, activation="relu", name="cand_phi_d0"),
                tf.keras.layers.Dense(hidden, activation="relu", name="cand_phi_d1"),
            ]
            self.attn_wq = tf.keras.layers.Dense(hidden, name="attn_wq")
            self.attn_wk = tf.keras.layers.Dense(hidden, name="attn_wk")
            self.attn_wv = tf.keras.layers.Dense(hidden, name="attn_wv")
            self.q_head = tf.keras.layers.Dense(hidden, activation="relu", name="q_head_d0")
            self.q_out = tf.keras.layers.Dense(1, name="q_out")

        # ---- 前向 ----
        def call(self, inputs: Dict[str, Any], training=False):
            rider_feat = inputs["rider_feat"]      # [B,N,8]
            rider_mask = inputs["rider_mask"]      # [B,N]
            cand_feat = inputs["cand_feat"]        # [B,N,K,12]
            cand_mask = inputs["cand_mask"]        # [B,N,K]
            edge_feat = inputs["edge_feat"]        # [B,N,K,12]
            global_feat = inputs["global_feat"]    # [B,5]

            b = tf.shape(rider_feat)[0]
            n = tf.shape(rider_feat)[1]
            k = tf.shape(cand_feat)[2]
            d = self.hidden

            # 骑手嵌入 + 集合上下文
            remb = rider_feat
            for layer in self.rider_phi:
                remb = layer(remb, training=training)                     # [B,N,D]
            m = tf.cast(rider_mask, remb.dtype)[..., None]
            cnt = tf.clip_by_value(tf.reduce_sum(m, axis=1), 1e-6, 1e9)
            set_ctx = self.rider_rho(tf.reduce_sum(remb * m, axis=1) / cnt,
                                     training=training)                   # [B,D]
            gctx = self.global_proj(global_feat, training=training)       # [B,D]
            h_in = tf.concat([
                remb,
                tf.broadcast_to(set_ctx[:, None, :], [b, n, d]),
                tf.broadcast_to(gctx[:, None, :], [b, n, d]),
            ], axis=-1)
            h_i = self.h_head(h_in, training=training)                    # [B,N,D]

            # 候选嵌入 + 多头注意力上下文
            keys0 = cand_feat
            for layer in self.cand_phi:
                keys0 = layer(keys0, training=training)                   # [B,N,K,D]
            ctx = self._cross_attn(h_i, keys0, cand_mask)                 # [B,N,D]
            e_ij = keys0 + tf.expand_dims(ctx, axis=2)                    # [B,N,K,D]

            h_b = tf.broadcast_to(h_i[:, :, None, :], [b, n, k, d])
            q_in = tf.concat([h_b, e_ij, h_b * e_ij, edge_feat], axis=-1)
            qh = self.q_head(q_in, training=training)
            q = tf.squeeze(self.q_out(qh, training=training), axis=-1)    # [B,N,K]
            cm = tf.cast(cand_mask, q.dtype)
            return q * cm + (1.0 - cm) * NEG_INF

        def _cross_attn(self, h_i, keys0, cand_mask):
            """query h_i [B,N,D] 对 keys0 [B,N,K,D] 做多头注意力 → [B,N,D]。"""
            b = tf.shape(h_i)[0]
            n = tf.shape(h_i)[1]
            k = tf.shape(keys0)[2]
            hh, dk = self.num_heads, self.dk
            q = self.attn_wq(h_i)                    # [B,N,D]
            kk = self.attn_wk(keys0)                 # [B,N,K,D]
            vv = self.attn_wv(keys0)                 # [B,N,K,D]
            q = tf.reshape(q, [b, n, hh, dk])
            kk = tf.reshape(kk, [b, n, k, hh, dk])
            vv = tf.reshape(vv, [b, n, k, hh, dk])
            # scores: [B,N,H,K]
            scores = tf.einsum("bnhd,bnkhd->bnhk", q, kk) / math.sqrt(float(dk))
            mask = tf.cast(cand_mask, scores.dtype)[:, :, None, :]        # [B,N,1,K]
            scores = scores * mask + (1.0 - mask) * NEG_INF
            attn = tf.nn.softmax(scores, axis=-1)
            out = tf.einsum("bnhk,bnkhd->bnhd", attn, vv)                 # [B,N,H,dk]
            return tf.reshape(out, [b, n, hh * dk])

        # ---- warm-start：末层 edge_feat 切片 ← 线性先验，其余置零（§3.3）----
        def warm_start_from_linear(self, edge_weights: np.ndarray = None):
            if edge_weights is None:
                edge_weights = np.asarray(HYBRID_DISPATCH_CONFIG["edge_weights"],
                                          dtype=np.float32)
            if not self.q_out.built:
                # Keras 惰性建权重：Dense 只依赖最后一维，最小 dummy 前向即可 build
                dummy = {
                    "rider_feat": np.zeros((1, 1, 8), np.float32),
                    "rider_mask": np.ones((1, 1), np.float32),
                    "cand_feat": np.zeros((1, 1, 1, 12), np.float32),
                    "cand_mask": np.ones((1, 1, 1), np.float32),
                    "edge_feat": np.zeros((1, 1, 1, 12), np.float32),
                    "global_feat": np.zeros((1, 5), np.float32),
                }
                self(dummy, training=False)

            # 头结构为 q_head([3D+E]→D, relu) + q_out(D→1)。
            # relu 会截断负值（edge_feat 含负），故用 ±x 双通道：
            #   qh[d] = relu(+x_d)，qh[E+d] = relu(-x_d)
            #   q = Σ w_d·qh[d] - w_d·qh[E+d] = Σ w_d·x_d ≡ 线性先验
            E = len(edge_weights)
            assert 2 * E <= self.hidden, "hidden 需 ≥ 2×edge_feat 维以承载 ±x 通道"
            kh, bh = self.q_head.get_weights()          # [3D+E, D], [D]
            kh = np.zeros_like(kh)
            bh = np.zeros_like(bh)
            base = 3 * self.hidden                       # edge_feat 切片起点
            for d in range(E):
                kh[base + d, d] = 1.0                    # +x_d
                kh[base + d, E + d] = -1.0               # -x_d
            self.q_head.set_weights([kh, bh])
            ko, bo = self.q_out.get_weights()            # [D, 1], [1]
            ko = np.zeros_like(ko)
            bo = np.zeros_like(bo)
            ko[:E, 0] = edge_weights
            ko[E:2 * E, 0] = -edge_weights
            self.q_out.set_weights([ko, bo])

        # ---- npz 导出 / 加载（numpy 推理降级，§3.2）----
        @staticmethod
        def _npz_key(layer_name: str, w_name: str) -> str:
            # TF2.15 中 w.name 自带模型名前缀（edge_value_net/rider_phi_d0/kernel:0），
            # 取最后一段 + 层名，键格式 "rider_phi_d0/kernel:0"（与 NpEdgeValueNet 对齐）
            return f"{layer_name}/{w_name.split('/')[-1]}"

        def export_npz(self, path: str):
            weights = {}
            for layer in self.layers:
                for w in layer.weights:
                    weights[self._npz_key(layer.name, w.name)] = w.numpy()
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            np.savez(path, **weights)

        def load_npz(self, path: str):
            data = np.load(path)
            for layer in self.layers:
                for w in layer.weights:
                    key = self._npz_key(layer.name, w.name)
                    if key in data:
                        w.assign(data[key])
            return self


# =============================================================================
# 2. NumPy 前向（无 TF 推理降级 / 并行 worker 行为策略）
# =============================================================================
class NpEdgeValueNet:
    """与 EdgeValueNet 逐层对齐的 numpy 前向。从 npz 加载。"""

    def __init__(self, npz_path: str = None, weights: Dict[str, np.ndarray] = None,
                 hidden: int = 64, num_heads: int = 4):
        if weights is None:
            data = np.load(npz_path)
            weights = {k: data[k] for k in data.files}
        self.w = weights
        self.hidden = hidden
        self.num_heads = num_heads
        self.dk = max(4, hidden // max(1, num_heads))

    @staticmethod
    def _dense(x, w, b=None, relu=False):
        y = x @ w
        if b is not None:
            y = y + b
        return np.maximum(y, 0.0) if relu else y

    def _get(self, layer, suffix="kernel:0"):
        return self.w[f"{layer}/{suffix}"]

    def forward(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """obs 为 build_all_pairs_set_obs 输出（无 batch 轴）→ 返回 [N,K] 打分。"""
        rf = obs["rider_feat"][None].astype(np.float64)   # [1,N,8]
        rm = obs["rider_mask"][None].astype(np.float64)   # [1,N]
        cf = obs["cand_feat"][None].astype(np.float64)    # [1,N,K,12]
        cm = obs["cand_mask"][None].astype(np.float64)    # [1,N,K]
        ef = obs["edge_feat"][None].astype(np.float64)    # [1,N,K,12]
        gf = obs["global_feat"][None].astype(np.float64)  # [1,5]
        B, N = rf.shape[0], rf.shape[1]
        K = cf.shape[2]
        D, H, dk = self.hidden, self.num_heads, self.dk

        remb = self._dense(rf, self._get("rider_phi_d0"), self._get("rider_phi_d0", "bias:0"), relu=True)
        remb = self._dense(remb, self._get("rider_phi_d1"), self._get("rider_phi_d1", "bias:0"), relu=True)
        m = rm[..., None]
        cnt = np.clip(m.sum(axis=1), 1e-6, None)
        set_ctx = self._dense((remb * m).sum(axis=1) / cnt,
                              self._get("rider_rho"), self._get("rider_rho", "bias:0"), relu=True)
        gctx = self._dense(gf, self._get("global_proj"), self._get("global_proj", "bias:0"), relu=True)
        h_in = np.concatenate([
            remb,
            np.broadcast_to(set_ctx[:, None, :], (B, N, D)),
            np.broadcast_to(gctx[:, None, :], (B, N, D)),
        ], axis=-1)
        h_i = self._dense(h_in, self._get("h_head"), self._get("h_head", "bias:0"), relu=True)

        keys0 = self._dense(cf, self._get("cand_phi_d0"), self._get("cand_phi_d0", "bias:0"), relu=True)
        keys0 = self._dense(keys0, self._get("cand_phi_d1"), self._get("cand_phi_d1", "bias:0"), relu=True)

        # 多头注意力（与 _cross_attn 对齐）
        q = self._dense(h_i, self._get("attn_wq"), self._get("attn_wq", "bias:0"))
        kk = self._dense(keys0, self._get("attn_wk"), self._get("attn_wk", "bias:0"))
        vv = self._dense(keys0, self._get("attn_wv"), self._get("attn_wv", "bias:0"))
        q = q.reshape(B, N, H, dk)
        kk = kk.reshape(B, N, K, H, dk)
        vv = vv.reshape(B, N, K, H, dk)
        scores = np.einsum("bnhd,bnkhd->bnhk", q, kk) / math.sqrt(float(dk))
        mask = cm[:, :, None, :]
        scores = scores * mask + (1.0 - mask) * NEG_INF
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores) * mask
        attn = exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)
        ctx = np.einsum("bnhk,bnkhd->bnhd", attn, vv).reshape(B, N, H * dk)

        e_ij = keys0 + ctx[:, :, None, :]
        h_b = np.broadcast_to(h_i[:, :, None, :], (B, N, K, D))
        q_in = np.concatenate([h_b, e_ij, h_b * e_ij, ef], axis=-1)
        qh = self._dense(q_in, self._get("q_head_d0"), self._get("q_head_d0", "bias:0"), relu=True)
        q = self._dense(qh, self._get("q_out"), self._get("q_out", "bias:0"))[..., 0]
        return (q * cm + (1.0 - cm) * NEG_INF)[0]
