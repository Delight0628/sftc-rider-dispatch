"""
集合编码（方案 B）：DeepSets + Cross-Attention
==============================================
骑手/订单变长输入的共用算子。TF 用于训练；numpy 供无 TF 单测与调试。

约定：
- 骑手: [N, D_r] + mask [N]
- 候选: [N, K, D_c] + mask [N, K]
- 所有聚合必须 **mask 加权**，禁止 padding 污染统计量
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
    _HAS_TF = True
except Exception:  # pragma: no cover - 本机可无 TF
    tf = None
    _HAS_TF = False


# =============================================================================
# 1. NumPy 参考实现（单测 / 无 TF）
# =============================================================================
def np_phi(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """单层 MLP：x [..., D] @ w [D, H] + b [H] → relu"""
    h = x @ w + b
    return np.maximum(h, 0.0)


def np_masked_mean(x: np.ndarray, mask: np.ndarray, axis: int = 0) -> np.ndarray:
    """mask 加权均值。x [..., D]，mask 与 x 的前导维对齐（去掉最后一维）。"""
    m = mask.astype(np.float32)
    while m.ndim < x.ndim - 1:
        m = m[..., None]
    m = m if m.shape[-1] == x.shape[-1] or m.shape[-1] == 1 else m
    # 广播到 x
    if m.shape != x.shape:
        m = np.broadcast_to(m if m.shape[-1] == 1 else m[..., None] if m.ndim == x.ndim - 1 else m,
                            x.shape) if False else _expand_mask(mask, x)
    else:
        m = m.astype(np.float32)
    s = (x * m).sum(axis=axis)
    c = np.clip(m.sum(axis=axis), 1e-6, None)
    return s / c


def _expand_mask(mask: np.ndarray, x: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)
    # mask 形状应为 x.shape[:-1] 或可广播
    while m.ndim < x.ndim - 1:
        m = np.expand_dims(m, -1)
    if m.ndim == x.ndim - 1:
        m = np.repeat(m[..., None], x.shape[-1], axis=-1)
    return np.broadcast_to(m, x.shape).astype(np.float32)


def np_deepsets(x: np.ndarray, mask: np.ndarray,
                w1: np.ndarray, b1: np.ndarray,
                w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """DeepSets：ρ( mean( φ(x_i) ) )，x [N, D]，mask [N]。"""
    h = np_phi(x, w1, b1)                 # [N, H]
    m = np.broadcast_to(mask.astype(np.float32)[:, None], h.shape)
    pooled = (h * m).sum(axis=0) / np.clip(m.sum(axis=0), 1e-6, None)  # [H]
    return np.maximum(pooled @ w2 + b2, 0.0)  # [H2]


def np_cross_attention(query: np.ndarray,
                       keys: np.ndarray,
                       values: np.ndarray,
                       mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """单头 cross-attn。query [Dq] 或 [1,Dq]；keys/values [K,D]；mask [K]。

    返回 (context [Dv], attn [K])。
    """
    q = np.atleast_2d(query).astype(np.float32)          # [1, Dq]
    k = keys.astype(np.float32)                          # [K, Dq]
    v = values.astype(np.float32)                        # [K, Dv]
    scale = 1.0 / np.sqrt(max(1, k.shape[-1]))
    logits = (q @ k.T) * scale                           # [1, K]
    m = mask.astype(np.float32)[None, :]
    logits = np.where(m > 0.5, logits, -1e9)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    e = e * m
    attn = e / np.clip(e.sum(axis=-1, keepdims=True), 1e-6, None)
    ctx = attn @ v                                       # [1, Dv]
    return ctx[0], attn[0]


# =============================================================================
# 2. TensorFlow 模块
# =============================================================================
if _HAS_TF:

    class PhiMLP(tf.keras.layers.Layer):
        def __init__(self, hidden: Tuple[int, ...] = (64, 64), name: str = "phi"):
            super().__init__(name=name)
            self.layers = [
                tf.keras.layers.Dense(h, activation="relu", name=f"{name}_d{i}")
                for i, h in enumerate(hidden)
            ]

        def call(self, x, training=False):
            h = x
            for layer in self.layers:
                h = layer(h, training=training)
            return h

    class MaskedDeepSets(tf.keras.layers.Layer):
        """ρ( masked_mean( φ(set) ) )。输入 [B, N, D] 或 [N, D]。"""

        def __init__(self, phi_hidden=(64, 64), rho_hidden=(64,), name="deepsets"):
            super().__init__(name=name)
            self.phi = PhiMLP(phi_hidden, name=f"{name}_phi")
            self.rho_layers = [
                tf.keras.layers.Dense(h, activation="relu", name=f"{name}_rho{i}")
                for i, h in enumerate(rho_hidden)
            ]

        def call(self, x, mask, training=False):
            # x: [..., N, D], mask: [..., N]
            h = self.phi(x, training=training)             # [..., N, H]
            m = tf.cast(mask, h.dtype)[..., None]          # [..., N, 1]
            s = tf.reduce_sum(h * m, axis=-2)              # [..., H]
            c = tf.clip_by_value(tf.reduce_sum(m, axis=-2), 1e-6, 1e9)
            pooled = s / c
            out = pooled
            for layer in self.rho_layers:
                out = layer(out, training=training)
            return out

    class CrossAttention(tf.keras.layers.Layer):
        """query [B, Dq] × keys [B, K, Dk] → context [B, Dv]。"""

        def __init__(self, dim: int, num_heads: int = 4, name="xattn"):
            super().__init__(name=name)
            self.mha = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=max(4, dim // max(1, num_heads)),
                name=f"{name}_mha",
            )

        def call(self, query, keys, values, key_mask, training=False):
            # query: [B, 1, D] 或 [B, D]；keys/values: [B, K, D]
            q = query if query.ndim == 3 else tf.expand_dims(query, axis=1)
            # MHA: attention_mask 为 bool，True=keep
            am = tf.cast(key_mask, tf.bool) if key_mask is not None else None
            ctx = self.mha(q, keys, values, attention_mask=am, training=training)
            return tf.squeeze(ctx, axis=1)                 # [B, D]

    class RiderSetEncoder(tf.keras.layers.Layer):
        def __init__(self, out_dim=64, name="rider_set"):
            super().__init__(name=name)
            self.ds = MaskedDeepSets((64, 64), (out_dim,), name=f"{name}_ds")
            self.out_dim = out_dim

        def call(self, rider_feat, rider_mask, training=False):
            return self.ds(rider_feat, rider_mask, training=training)

    class OrderCandEncoder(tf.keras.layers.Layer):
        """每骑手候选订单集合 → 上下文向量。cand: [B, N, K, Dc]"""

        def __init__(self, dim=64, num_heads=4, name="cand_enc"):
            super().__init__(name=name)
            self.phi = PhiMLP((64, dim), name=f"{name}_phi")
            self.attn = CrossAttention(dim, num_heads, name=f"{name}_attn")
            self.dim = dim

        def call(self, query, cand_feat, cand_mask, training=False):
            # query: [B, N, Dr'] ；cand: [B, N, K, Dc]；mask: [B, N, K]
            b = tf.shape(cand_feat)[0]
            n = tf.shape(cand_feat)[1]
            k = tf.shape(cand_feat)[2]
            flat_c = tf.reshape(cand_feat, [b * n, k, tf.shape(cand_feat)[-1]])
            flat_m = tf.reshape(cand_mask, [b * n, k])
            keys = self.phi(flat_c, training=training)     # [B*N, K, dim]
            q = tf.reshape(query, [b * n, -1])
            ctx = self.attn(q, keys, keys, flat_m, training=training)  # [B*N, dim]
            return tf.reshape(ctx, [b, n, self.dim])
