"""
方案 B：集合编码 Actor-Critic（MAPPO / CTDE）
==========================================
- Actor：每骑手「self 属性 ⊕ DeepSets(骑手) ⊕ CrossAttn(候选)」→ K+1 logits
- Critic：DeepSets(骑手) ⊕ DeepSets(池) ⊕ global → V

与旧 PPONetwork 并行；不共享权重。训练栈可在 obs_mode='set' 时选用。
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    import tensorflow as tf
    from mappo.set_encoder import (
        MaskedDeepSets, PhiMLP, OrderCandEncoder, _HAS_TF,
    )
except Exception as e:  # pragma: no cover
    _HAS_TF = False
    _IMPORT_ERR = e


if _HAS_TF:

    class PPONetworkSet:
        """变长骑手/订单的 MAPPO 网络（方案 B）。"""

        def __init__(
            self,
            rider_dim: int,
            cand_dim: int,
            global_dim: int,
            num_candidates: int,
            action_dim: int,
            hidden: int = 128,
            lr: Any = 1e-4,
            num_heads: int = 4,
        ):
            self.rider_dim = int(rider_dim)
            self.cand_dim = int(cand_dim)
            self.global_dim = int(global_dim)
            self.num_candidates = int(num_candidates)
            self.action_dim = int(action_dim)  # 含 IDLE
            self.hidden = int(hidden)
            self._build()
            if lr is not None:
                self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr * 0.5)
            else:
                self.actor_optimizer = None
                self.critic_optimizer = None

        def _build(self):
            # ---------- Actor ----------
            # 每骑手独立前向：输入其 self 行 + 全体骑手集合 + 其候选
            self._self_proj = tf.keras.layers.Dense(self.hidden, activation="relu")
            self._rider_set = MaskedDeepSets((self.hidden, self.hidden), (self.hidden,))
            self._cand_enc = OrderCandEncoder(dim=self.hidden, num_heads=4)
            self._global_proj = tf.keras.layers.Dense(self.hidden, activation="relu")
            self._fuse = tf.keras.Sequential([
                tf.keras.layers.Dense(self.hidden * 2, activation="relu"),
                tf.keras.layers.Dense(self.hidden, activation="relu"),
            ])
            self._logits = tf.keras.layers.Dense(self.action_dim, activation=None)
            # 可学习偏好 bias（会议：紧迫优先，偏好只让一小部分自由度）
            self._pref_bias_scale = tf.Variable(0.1, trainable=True, name="pref_bias_scale")

            # ---------- Critic ----------
            self._c_rider = MaskedDeepSets((self.hidden, self.hidden), (self.hidden,))
            self._c_pool = MaskedDeepSets((self.hidden, self.hidden), (self.hidden,))
            self._c_global = tf.keras.layers.Dense(self.hidden, activation="relu")
            self._c_head = tf.keras.Sequential([
                tf.keras.layers.Dense(self.hidden * 2, activation="relu"),
                tf.keras.layers.Dense(1, activation=None),
            ])

        def actor_forward(self, rider_feat, rider_mask, self_index,
                          cand_feat, cand_mask, global_feat, training=False):
            """
            rider_feat: [B, N, Dr]  cand_feat: [B, N, K, Dc]
            self_index: [B] int
            返回 logits: [B, A]
            """
            g = self._global_proj(global_feat, training=training)          # [B, H]
            rs = self._rider_set(rider_feat, rider_mask, training=training)  # [B, H]
            # self 行
            b = tf.shape(rider_feat)[0]
            idx = tf.cast(self_index, tf.int32)
            batch_ix = tf.range(b)
            gather_ix = tf.stack([batch_ix, idx], axis=1)
            self_row = tf.gather_nd(rider_feat, gather_ix)                 # [B, Dr]
            s = self._self_proj(self_row, training=training)               # [B, H]
            # 候选上下文：仅取 self 的候选行 [B,K,D] / [B,K]
            gather_self = tf.stack([batch_ix, idx], axis=1)
            cand_self = tf.gather_nd(cand_feat, gather_self)
            mask_self = tf.gather_nd(cand_mask, gather_self)
            q = tf.concat([s, rs, g], axis=-1)                             # [B, 3H]
            q = self._fuse(q, training=training)                           # [B, H]
            ctx = self._cand_enc(q[:, None, :], cand_self[:, None, :, :],
                                 mask_self[:, None, :], training=training)
            ctx = ctx[:, 0, :]                                             # [B, H]
            fused = tf.concat([q, ctx], axis=-1)
            # 轻量偏好 bias：候选 willingness 均值 → 非 IDLE 维（会议：少量偏好自由度）
            w = cand_self[:, :, 1]
            wm = tf.cast(mask_self, w.dtype)
            w_mean = tf.reduce_sum(w * wm, axis=-1) / tf.clip_by_value(
                tf.reduce_sum(wm, axis=-1), 1e-6, 1e9)
            logits = self._logits(fused, training=training)
            pref = self._pref_bias_scale * (w_mean * 2.0 - 1.0)             # [B]
            pref = tf.expand_dims(pref, -1)                                # [B,1]
            zeros = tf.zeros_like(logits[:, :1])
            logits = tf.concat([zeros, logits[:, 1:] + pref], axis=-1)
            return logits

        def critic_forward(self, rider_feat, rider_mask,
                           pool_feat, pool_mask, global_feat, training=False):
            rs = self._c_rider(rider_feat, rider_mask, training=training)
            ps = self._c_pool(pool_feat, pool_mask, training=training)
            g = self._c_global(global_feat, training=training)
            h = tf.concat([rs, ps, g], axis=-1)
            v = self._c_head(h, training=training)
            return tf.squeeze(v, axis=-1)

else:
    class PPONetworkSet:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError(f"TensorFlow unavailable: {_IMPORT_ERR}")
