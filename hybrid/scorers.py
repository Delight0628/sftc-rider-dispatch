"""hybrid/scorers.py — 边打分器统一接口（Linear / Neural）

文档约定：docs/hybrid_implementation.md §3.4
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from environments.hybrid_dispatch import EdgeScorer as _LinearEdgeScorer


class BaseEdgeScorer:
    """边打分器接口：集中式，从 obs_pack（build_all_pairs_set_obs 输出）生成边分。"""

    def score_edges(self, obs_pack: Dict[str, Any]) -> np.ndarray:
        """返回 [N,K] 打分矩阵；padding / 非法位置用 -inf 标记。

        obs_pack 包含 keys: rider_feat, rider_mask, cand_feat, cand_mask,
        edge_feat, global_feat, cand_order_ids, cand_actions, rider_names。
        """
        raise NotImplementedError

    def learnable(self) -> bool:
        return False

    def learn(self, *args, **kwargs) -> int:
        """学习更新步数。"""
        return 0


class LinearEdgeScorer(BaseEdgeScorer):
    """包装现有的手调线性 EdgeScorer（零 TF 依赖，永远可用）。"""

    def __init__(self, weights: np.ndarray = None):
        self._scorer = _LinearEdgeScorer(weights=weights, learn=False)

    def score_edges(self, obs_pack: Dict[str, Any]) -> np.ndarray:
        edge_feat = obs_pack["edge_feat"]    # [N,K,12]
        cand_mask = obs_pack["cand_mask"]    # [N,K]
        scores = np.einsum("nkd,d->nk", edge_feat.astype(np.float64),
                           self._scorer.weights)
        return np.where(cand_mask > 0.5, scores, -1e9).astype(np.float32)


class NeuralEdgeScorer(BaseEdgeScorer):
    """包装 TF EdgeValueNet，TF 不可用时抛异常（由调用方捕获回退）。

    维护在线网络 + 目标网络（惰性构建）。npz 加载降级路径。
    """

    def __init__(self, npz_path: str = None, hidden: int = 64, num_heads: int = 4,
                 target_npz_path: str = None):
        self.npz_path = npz_path
        self._np_net = None
        self._tf_net = None
        self._tf_target = None
        self.hidden = hidden
        self.num_heads = num_heads
        self.target_npz_path = target_npz_path

        # 如果有 npz 路径，先加载 numpy 网络（推理用）
        if npz_path is not None:
            from hybrid.edge_value_net import NpEdgeValueNet
            self._np_net = NpEdgeValueNet(npz_path, hidden=hidden, num_heads=num_heads)

        # 尝试加载 TF 网络（训练用）
        self._try_tf()

    @staticmethod
    def _build_tf_net(net):
        """Keras 惰性建权重：Dense 只依赖最后一维，用最小 dummy 前向即可 build。"""
        dummy = {
            "rider_feat": np.zeros((1, 1, 8), np.float32),
            "rider_mask": np.ones((1, 1), np.float32),
            "cand_feat": np.zeros((1, 1, 1, 12), np.float32),
            "cand_mask": np.ones((1, 1, 1), np.float32),
            "edge_feat": np.zeros((1, 1, 1, 16), np.float32),
            "global_feat": np.zeros((1, 5), np.float32),
        }
        net(dummy, training=False)

    def _try_tf(self):
        try:
            from hybrid.edge_value_net import EdgeValueNet, _HAS_TF
            if not _HAS_TF:
                return
            self._tf_net = EdgeValueNet(hidden=self.hidden, num_heads=self.num_heads)
            if self.npz_path is not None:
                self._build_tf_net(self._tf_net)  # 先建权重，load_npz 才生效
                self._tf_net.load_npz(self.npz_path)
            if self.target_npz_path is not None:
                self._tf_target = EdgeValueNet(hidden=self.hidden, num_heads=self.num_heads)
                self._build_tf_net(self._tf_target)
                self._tf_target.load_npz(self.target_npz_path)
        except Exception:
            self._tf_net = None
            self._tf_target = None

    def score_edges(self, obs_pack: Dict[str, Any]) -> np.ndarray:
        """优先 numpy 推理（TF 可选，避免 worker 进程 TF 串行）。"""
        if self._np_net is not None:
            return self._np_net.forward(obs_pack)
        if self._tf_net is None:
            raise RuntimeError(
                "NeuralEdgeScorer: TF 不可用且 npz 未加载；请确保 npz_path 有效或 TF 已安装")
        # TF 推理（加 batch 轴）
        batch = {k: obs_pack[k][None] for k in
                 ("rider_feat", "rider_mask", "cand_feat", "cand_mask",
                  "edge_feat", "global_feat")}
        q = self._tf_net(batch, training=False).numpy()[0]
        return q

    def learnable(self) -> bool:
        return self._tf_net is not None

    def tf_net(self):
        """供训练器访问在线网络。"""
        return self._tf_net

    def target_net(self):
        if self._tf_target is None:
            raise RuntimeError("目标网络未初始化")
        return self._tf_target

    def sync_target(self, tau: float = 0.005):
        """软更新目标网络。"""
        if self._tf_target is None or self._tf_net is None:
            return
        for t_var, o_var in zip(self._tf_target.trainable_variables,
                                  self._tf_net.trainable_variables):
            t_var.assign(tau * o_var + (1.0 - tau) * t_var)

    def save(self, path: str, target_path: str = None):
        """保存 npz（numpy 推理）+ h5（TF 完整状态）。"""
        if self._tf_net is not None:
            self._tf_net.export_npz(path)
            if target_path and self._tf_target is not None:
                self._tf_target.export_npz(target_path)
        elif self._np_net is not None:
            # 回退：npz 已在构造函数中加载，但无法反保存（无 TF 状态）
            import shutil
            shutil.copy(self.npz_path, path)
