"""方案 B 集合编码 / set 观测校验（numpy 为主，TF 可选）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.delivery_env import DeliveryEnv  # noqa: E402
from environments.set_obs import (  # noqa: E402
    build_set_obs, build_global_set_state, RIDER_FEAT_DIM, CAND_FEAT_DIM, GLOBAL_FEAT_DIM,
)
from mappo.set_encoder import (  # noqa: E402
    np_deepsets, np_cross_attention, np_masked_mean, _HAS_TF,
)


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
        print(f"\n=== set_encoding_check: {self.passed}/{total} passed ===")
        for e in self.errors:
            print(" -", e)
        return 0 if self.failed == 0 else 1


def _mk_riders(n):
    riders = {}
    for i in range(n):
        name = f"骑手{chr(ord('A') + i)}"
        riders[name] = {
            "count": 1, "capacity": 3, "speed": 0.55,
            "home": (2.0 + 3 * i, 10.0), "_real": {},
        }
    return riders


def main():
    print("=== set_encoding_check ===")
    ck = Checker()

    print("\n[1] numpy 集合算子")
    rng = np.random.RandomState(0)
    x = rng.randn(7, 5).astype(np.float32)
    mask = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    w1, b1 = rng.randn(5, 4).astype(np.float32) * 0.1, np.zeros(4, np.float32)
    w2, b2 = rng.randn(4, 3).astype(np.float32) * 0.1, np.zeros(3, np.float32)
    g = np_deepsets(x, mask, w1, b1, w2, b2)
    ck.check("DeepSets 输出 [3]", g.shape == (3,), str(g.shape))
    # padding 不应改变聚合
    x2 = np.concatenate([x[:3], rng.randn(4, 5).astype(np.float32) * 10], axis=0)
    g2 = np_deepsets(x2, mask, w1, b1, w2, b2)
    ck.check("padding 无关", np.allclose(g, g2, atol=1e-5), f"{g} vs {g2}")

    keys = rng.randn(10, 8).astype(np.float32)
    vals = rng.randn(10, 6).astype(np.float32)
    q = rng.randn(8).astype(np.float32)
    am = np.zeros(10, np.float32)
    am[:4] = 1.0
    ctx, attn = np_cross_attention(q, keys, vals, am)
    ck.check("attn ctx 维度", ctx.shape == (6,), str(ctx.shape))
    ck.check("attn 对 mask 外≈0", float(attn[4:].sum()) < 1e-5, f"{attn}")
    ck.check("attn 和为 1", abs(float(attn.sum()) - 1.0) < 1e-4, f"{attn.sum()}")

    print("\n[2] set 观测导出（N=3 / 5 / 8）")
    for n in (3, 5, 8):
        env = DeliveryEnv({
            "scenario": "delivery", "training_mode": True,
            "riders": _mk_riders(n), "obs_mode": "set",
            "MAX_SIM_STEPS": 30,
        })
        env.reset(seed=0)
        agent = env.agents[0]
        so = env.get_set_observation(agent)
        ck.check(f"N={n} rider_feat", so["rider_feat"].shape == (n, RIDER_FEAT_DIM),
                 str(so["rider_feat"].shape))
        ck.check(f"N={n} cand_feat", so["cand_feat"].shape[0] == n and so["cand_feat"].shape[2] == CAND_FEAT_DIM,
                 str(so["cand_feat"].shape))
        ck.check(f"N={n} global", so["global_feat"].shape == (GLOBAL_FEAT_DIM,),
                 str(so["global_feat"].shape))
        ck.check(f"N={n} self_index", so["self_index"] == 0, str(so["self_index"]))
        gs = build_global_set_state(env.sim)
        ck.check(f"N={n} global set rider", gs["rider_feat"].shape == (n, RIDER_FEAT_DIM))
        flat = env.sim.get_state_for_agent(agent)
        if n == 5:
            ck.check("N=5 时 146 obs 仍可用", flat.shape == (146,), str(flat.shape))
        else:
            # one-hot 宽度随 N 变，146 约定仅绑 N=5；set 路径不吃该约束
            ck.check(f"N={n} flat 可序列化（非 146 亦可）", flat.ndim == 1 and flat.size > 0,
                     str(flat.shape))

    print("\n[3] infos 附带 set_obs")
    env = DeliveryEnv({"scenario": "delivery", "training_mode": True, "obs_mode": "set"})
    env.reset(seed=1)
    ck.check("infos.set_obs", "set_obs" in env.infos[env.agents[0]])

    print("\n[4] TF 网络（可选）")
    if _HAS_TF:
        from mappo.ppo_network_set import PPONetworkSet
        import tensorflow as tf
        net = PPONetworkSet(
            rider_dim=RIDER_FEAT_DIM, cand_dim=CAND_FEAT_DIM,
            global_dim=GLOBAL_FEAT_DIM, num_candidates=10, action_dim=11, lr=1e-4,
        )
        B, N, K = 2, 5, 10
        rf = tf.random.normal([B, N, RIDER_FEAT_DIM])
        rm = tf.ones([B, N])
        cf = tf.random.normal([B, N, K, CAND_FEAT_DIM])
        cm = tf.ones([B, N, K])
        gf = tf.random.normal([B, GLOBAL_FEAT_DIM])
        si = tf.constant([0, 1])
        logits = net.actor_forward(rf, rm, si, cf, cm, gf)
        ck.check("actor logits [B,11]", tuple(logits.shape) == (B, 11), str(logits.shape))
        pf = tf.random.normal([B, 32, 6])
        pm = tf.ones([B, 32])
        v = net.critic_forward(rf, rm, pf, pm, gf)
        ck.check("critic V [B]", tuple(v.shape) == (B,), str(v.shape))
    else:
        ck.check("TF 不可用（跳过构图）", True)

    return ck.summary()


if __name__ == "__main__":
    raise SystemExit(main())
