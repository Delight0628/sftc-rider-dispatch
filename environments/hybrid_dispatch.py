"""
混合派单框架：学习边效用 + 约束二分图匹配 + 滚动重优化
====================================================
落地 docs/research_dispatch_algorithm_survey.md §5 推荐的工业派单范式
（DiDi KDD 2018/2019/2022、Meituan SCDN 已验证路线）：

    事件驱动决策点（env 每步 = 一个决策 epoch）
      → [学习的边效用]  q(骑手 i, 订单 j | 全局上下文)   EdgeScorer
      → [约束匹配层]    加权二分图匹配（一人一单/一单一人）
                        按容量分轮求解 = 带容量 b-matching
      → [滚动重优化]    未派订单留在池中，下一 epoch 重建边集重解

与 MAPPO 路线的关键差异（对应调研 §4a 缺点逐条消解）：
- 变规模原生：边打分对任意骑手数 × 订单数直接工作，不依赖 146 维/one-hot；
- 无抢单冲突：匹配层保证一单至多派给一人（race_conflict ≡ 0）；
- 可行性保证：动作只取 action_mask 合法候选，接单数 ≤ 剩余容量；
- 信用分配到边级：效用是 (骑手,订单) 对的函数，学习目标可下沉到边。

接口约束（与 delivery_env.py 对齐）：
- 只消费 infos['candidates_map'] / ['action_mask']（外部接口）+ sim 骑手状态
  （同包内部接口，与 set_obs.py 同级用法）；
- 动作格式：{agent_id: [cand_action, ...]}，cand_action 即 candidates_map[i]['action']，
  同骑手多动作按 FIFO 顺序落地（step_with_actions 逐个执行）；
- 「时间紧迫性」维度 only：不含 willingness（偏好由并行双塔侧处理，project.md §3）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .delivery_config import HYBRID_DISPATCH_CONFIG

# 边特征名（可学习权重与之一一对应）
EDGE_FEATURE_NAMES = (
    "bias",              # 1.0 截距
    "order_urgency",     # 订单紧迫度：越接近/超过 due 越大（EDD 分量，骑手无关）
    "on_time_feasible",  # 预计准时可达（slack >= 0）
    "late_norm",         # 预计迟到量压缩归一（越大越差）
    "to_pickup_norm",    # 首段取餐耗时压缩归一（nearest 分量）
    "route_norm",        # 全程耗时压缩归一
    "rider_load",        # 骑手载负荷（含本轮已虚拟派入），负载均衡
    "free_delay_norm",   # 骑手预计空闲延迟（快空闲者优先）
    "priority_norm",     # 订单优先级
    "is_urgent",         # 是否紧急单
    "congestion",        # 取餐点拥堵
    "slack_norm",        # 配对安全余量（同等条件偏好更稳的配对）
)


# =============================================================================
# 1. 边效用打分器（线性可学习；手调先验初始化，支持在线更新）
# =============================================================================
class EdgeScorer:
    """q(骑手 i, 订单 j | 上下文) = w · f(i, j)。

    - 默认权重 = 手调先验（紧迫分量 + 配对质量分量分解，见 module docstring）；
    - learn=True 时按 delta 规则在线更新（contextual bandit 式效用回归），
      为后续 off-policy TD / 反事实基线预留的最小学习回路；
    - 打分与动作选择解耦，天然支持"学习打分 + 匹配"分离训练。
    """

    def __init__(self, weights: Optional[np.ndarray] = None,
                 learn: bool = False, lr: Optional[float] = None):
        cfg = HYBRID_DISPATCH_CONFIG
        if weights is None:
            weights = np.asarray(cfg["edge_weights"], dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64).copy()
        if self.weights.shape != (len(EDGE_FEATURE_NAMES),):
            raise ValueError(
                f"edge weights 维度应为 {len(EDGE_FEATURE_NAMES)}，得到 {self.weights.shape}")
        self.learn = bool(learn)
        self.lr = float(lr if lr is not None else cfg.get("learn_lr", 0.01))
        self.update_count = 0

    # ---- 特征构造 ----
    @staticmethod
    def build_features(sim, rider, order, start_pos: Tuple[float, float],
                       start_time: float, now: float,
                       virt_load: int = 0) -> np.ndarray:
        """边特征向量（12 维，与 EDGE_FEATURE_NAMES 对齐）。

        start_pos / start_time / virt_load 为"虚拟执行链"投影：
        同一决策步为骑手连派多单时，后一单的特征按前单完成后的状态估算。
        """
        cfg = HYBRID_DISPATCH_CONFIG
        slack_norm = cfg["slack_norm"]
        est = sim._estimate_route(rider, order, start_pos, start_time)
        slack = order.due_date - est["deliver_time"]
        due_rel = order.due_date - now
        compressed = sim._compressed
        feats = np.asarray([
            1.0,
            float(np.clip(1.0 - due_rel / slack_norm, -1.0, 2.0)),      # 越紧迫越大
            1.0 if slack >= 0.0 else 0.0,
            compressed(max(0.0, -slack), 60.0),                          # 迟到量
            compressed(est["leg1_time"], 60.0),
            compressed(est["total_time"], 120.0),
            (len(rider.carry) + virt_load) / max(1, rider.capacity),
            compressed(max(0.0, start_time - now), 60.0),
            (order.priority - 1) / 2.0,
            1.0 if order.priority == 1 else 0.0,
            sim._pickup_congestion(order),
            float(np.clip(slack / slack_norm, -3.0, 3.0)),
        ], dtype=np.float64)
        return feats

    def score(self, feats: np.ndarray) -> float:
        return float(self.weights @ feats)

    # ---- 学习回路（最小实现：效用回归 delta 规则）----
    def update(self, feats: np.ndarray, target: float) -> None:
        """用实现效用回归边效用：w += lr * (target - w·f) * f。"""
        if not self.learn:
            return
        pred = self.score(feats)
        self.weights = self.weights + self.lr * (target - pred) * feats
        self.update_count += 1

    @staticmethod
    def realized_utility(order, distance: float,
                         sim_end: Optional[float] = None) -> float:
        """边的实现效用（学习目标）：取餐/送达两段 shaping（exp-a）。

        取餐段 u_pick    = 1 − ready→取到餐等待/pick_norm（及时取餐）
        送达段 u_deliver = 1 − w_tard·迟到/60 − w_dist·距离/20（原口径不变）
        组合             = w_pick·u_pick + (1−w_pick)·u_deliver
        未履约：已取餐按取餐段得分 + 送达段记 −1；未取餐整体 −1。
        """
        cfg = HYBRID_DISPATCH_CONFIG
        w_pick = float(cfg.get("w_pick", 0.3))
        pick_norm = float(cfg.get("pick_norm", 60.0))

        pick_time = order.planned_pickup_time
        picked_up = (pick_time is not None
                     and (sim_end is None or pick_time <= sim_end))
        u_pick = 1.0 - max(0.0, (pick_time or order.ready_time) - order.ready_time) / pick_norm

        if order.actual_deliver_time is None:
            if not picked_up:
                return -1.0  # 未取餐未履约
            return float(w_pick * u_pick + (1.0 - w_pick) * (-1.0))
        tard = max(0.0, order.actual_deliver_time - order.due_date)
        u_deliver = float(1.0
                          - cfg["w_tard"] * (tard / 60.0)
                          - cfg["w_dist"] * (distance / 20.0))
        return float(w_pick * u_pick + (1.0 - w_pick) * u_deliver)


# =============================================================================
# 2. 约束二分图匹配（贪心 + 交换改进，无第三方依赖）
# =============================================================================
def max_weight_matching(edges: List[Tuple[str, int, float, Any]],
                        ) -> List[Tuple[str, int, float, Any]]:
    """一对一二分图匹配：edges = [(rider, order_id, score, payload), ...]。

    贪心按分数降序构造 + 两两交换改进至局部最优（规模：骑手≤~20 × 候选≤10，
    每轮 ≤ 200 边，交换 O(E²) 足够快）。依赖 scipy 时可换匈牙利，此处保持零依赖、
    确定性（并列按 (rider, order_id) 平铺打破）。
    """
    if not edges:
        return []
    keyed = {(r, o): (s, p) for r, o, s, p in edges}
    ordered = sorted(edges, key=lambda e: (-e[2], e[0], e[1]))
    matched_r: Dict[str, Tuple[int, float, Any]] = {}
    matched_o: Dict[int, Tuple[str, float, Any]] = {}
    for r, o, s, p in ordered:
        if r not in matched_r and o not in matched_o:
            matched_r[r] = (o, s, p)
            matched_o[o] = (r, s, p)

    # 两两交换改进（2-opt for assignment）
    improved = True
    while improved:
        improved = False
        riders = sorted(matched_r.keys())
        for i in range(len(riders)):
            for k in range(i + 1, len(riders)):
                r1, r2 = riders[i], riders[k]
                o1, s1, _ = matched_r[r1]
                o2, s2, _ = matched_r[r2]
                alt1 = keyed.get((r1, o2))
                alt2 = keyed.get((r2, o1))
                if alt1 is None or alt2 is None:
                    continue
                if alt1[0] + alt2[0] > s1 + s2 + 1e-9:
                    matched_r[r1] = (o2, alt1[0], alt1[1])
                    matched_r[r2] = (o1, alt2[0], alt2[1])
                    matched_o[o1] = (r2, alt2[0], alt2[1])
                    matched_o[o2] = (r1, alt1[0], alt1[1])
                    improved = True
    return [(r, o, s, p) for r, (o, s, p) in sorted(matched_r.items())]


# =============================================================================
# 3. 滚动重优化派单器（每个决策 epoch 重建边集并求匹配）
# =============================================================================
class HybridDispatcher:
    """事件驱动滚动派单：act() 每步从当前状态重解一次匹配。

    - 带容量 b-matching：按轮求解（每轮一人至多一单），轮内特征按虚拟执行链
      投影更新，轮数 = 最大剩余容量 → 同骑手多单的先后顺序即 FIFO 落地顺序；
    - 未派订单留在池中，下一 epoch 自动重入（滚动重优化）；
    - learn=True 时 episode 结束调用 learn_from_episode() 回归实现效用。
    """

    def __init__(self, env, scorer: Optional[EdgeScorer] = None,
                 config: Optional[Dict[str, Any]] = None,
                 batch_scorer: Optional[Any] = None):
        self.env = env
        self.cfg = dict(HYBRID_DISPATCH_CONFIG)
        if config:
            self.cfg.update(config)
        self.scorer = scorer or EdgeScorer(
            learn=bool(self.cfg.get("learn", False)),
            lr=self.cfg.get("learn_lr"))
        # 批式打分器（hybrid.scorers.BaseEdgeScorer）；设置后 act() 走集中式打分路径
        self.batch_scorer = batch_scorer
        self._records: List[Tuple[np.ndarray, str, int, float]] = []
        self._order_by_id = {o.order_id: o for o in env.sim.orders}

    # ---- 单个决策 epoch 的联合动作 ----
    def act(self, temp: float = 0.0, rng: Optional[np.random.RandomState] = None,
            record: bool = False):
        """集中式/分布式统一入口。

        - batch_scorer 为空：逐边打分（EdgeScorer），支持虚拟执行链投影（原 P0 路径）；
        - batch_scorer 非空：集中式 obs_pack 打分（build_all_pairs_set_obs），
          temp>0 时按温度对边分做 softmax 采样（探索），匹配轮内静态 q；
        - record=True：返回 (actions, trace)，trace 含 obs_pack 与 matched 边，
          供 off-policy 回放（hybrid/collect.py）。
        """
        if self.batch_scorer is not None:
            return self._act_batch(temp=temp, rng=rng, record=record)
        return self._act_legacy(record=record)

    def _act_batch(self, temp: float = 0.0,
                   rng: Optional[np.random.RandomState] = None,
                   record: bool = False):
        from environments.set_obs import build_all_pairs_set_obs
        sim = self.env.sim
        obs = build_all_pairs_set_obs(sim)
        q = self.batch_scorer.score_edges(obs)          # [N,K]，非法 -inf
        names: List[str] = obs["rider_names"]
        cand_oid = obs["cand_order_ids"]                # [N,K] int32，-1 空
        cand_act = obs["cand_actions"]                  # [N,K] int32，0 无
        cand_mask = obs["cand_mask"]                    # [N,K]

        actions: Dict[str, List[int]] = {aid: [] for aid in self.env.agents}
        taken_orders: set = set()
        virt_load: Dict[str, int] = {}
        matched: List[Tuple[int, int, int, float]] = []  # (i, j, order_id, distance)
        if rng is None:
            rng = np.random.RandomState(0)

        max_rounds = min(
            max((sim.riders[nm].capacity - len(sim.riders[nm].carry) for nm in names),
                default=0),
            int(self.cfg.get("max_assign_per_step", 8)))

        for round_idx in range(max_rounds):
            edges: List[Tuple[str, int, float, Any]] = []
            for i, nm in enumerate(names):
                rider = sim.riders[nm]
                if rider.capacity - len(rider.carry) - virt_load.get(nm, 0) <= 0:
                    continue
                row_valid = False
                # 探索：按温度对合法边做 softmax 采样一个候选（行为策略，§4.3）
                sampled_j = -1
                if temp > 0.0:
                    legal = [j for j in range(q.shape[1])
                             if cand_mask[i, j] > 0.5 and int(cand_oid[i, j]) >= 0
                             and int(cand_oid[i, j]) not in taken_orders]
                    if legal:
                        logits = np.asarray([q[i, j] for j in legal], dtype=np.float64)
                        logits = logits / max(temp, 1e-6)
                        logits -= logits.max()
                        p = np.exp(logits)
                        p /= p.sum()
                        sampled_j = legal[int(rng.choice(len(legal), p=p))]
                for j in range(q.shape[1]):
                    oid = int(cand_oid[i, j])
                    if oid < 0 or cand_mask[i, j] < 0.5 or oid in taken_orders:
                        continue
                    if temp > 0.0 and j != sampled_j:
                        continue  # 探索期每骑手每轮只放行被采样边
                    row_valid = True
                    edges.append((nm, oid, float(q[i, j]), (i, j)))
                if not row_valid:
                    continue

            for nm, oid, score, (i, j) in max_weight_matching(edges):
                rider = sim.riders[nm]
                order = self._order_by_id.get(oid)
                if order is None or order.state != "pool":
                    continue
                aid = f"agent_{nm}"
                actions[aid].append(int(cand_act[i, j]))
                taken_orders.add(oid)
                pos, free_t = rider.projected_free()
                start = max(sim.current_time, free_t if free_t > 0 else sim.current_time)
                est = sim._estimate_route(rider, order, pos, start)
                virt_load[nm] = virt_load.get(nm, 0) + 1
                matched.append((i, j, oid, float(est["distance"])))

        for aid in self.env.agents:
            if not actions[aid]:
                actions[aid] = [0]

        if record:
            trace = {"obs": obs, "matched": matched}
            return actions, trace
        return actions

    def _act_legacy(self, record: bool = False) -> Dict[str, Any]:
        sim = self.env.sim
        infos = self.env.infos
        now = sim.current_time
        actions: Dict[str, List[int]] = {aid: [] for aid in self.env.agents}
        taken_orders: set = set()
        # 虚拟执行链：(pos, free_time, virt_load)，随本轮派入更新
        virt: Dict[str, Tuple[Tuple[float, float], float, int]] = {}
        agent_to_rider = {aid: aid.replace("agent_", "") for aid in self.env.agents}

        max_rounds = 0
        for aid in self.env.agents:
            rider = sim.riders[agent_to_rider[aid]]
            free_slots = rider.capacity - len(rider.carry)
            max_rounds = max(max_rounds, free_slots)
        max_rounds = min(max_rounds, int(self.cfg.get("max_assign_per_step", 8)))

        for _ in range(max_rounds):
            edges: List[Tuple[str, int, float, Any]] = []
            for aid in self.env.agents:
                rider_name = agent_to_rider[aid]
                rider = sim.riders[rider_name]
                info = infos.get(aid, {})
                mask = info.get("action_mask")
                cands = info.get("candidates_map") or []
                if not cands:
                    continue
                if rider.capacity - len(rider.carry) <= virt.get(rider_name, (None, None, 0))[2]:
                    continue  # 剩余容量已被本轮虚拟派入占满
                pos, free_t, vload = virt.get(
                    rider_name, (*rider.projected_free(), 0))
                start = max(now, free_t if free_t > 0 else now)
                for c in cands:
                    act = int(c["action"])
                    if mask is not None and act < len(mask) and mask[act] < 0.5:
                        continue
                    oid = int(c["part_id"])
                    if oid in taken_orders:
                        continue
                    order = self._order_by_id.get(oid)
                    if order is None or order.state != "pool":
                        continue
                    feats = self.scorer.build_features(
                        sim, rider, order, pos, start, now, virt_load=vload)
                    score = self.scorer.score(feats)
                    edges.append((rider_name, oid, score, (aid, act, feats, order)))

            for rider_name, oid, score, (aid, act, feats, order) in max_weight_matching(edges):
                actions[aid].append(act)
                taken_orders.add(oid)
                rider = sim.riders[rider_name]
                pos, free_t, vload = virt.get(rider_name, (*rider.projected_free(), 0))
                start = max(now, free_t if free_t > 0 else now)
                est = sim._estimate_route(rider, order, pos, start)
                virt[rider_name] = (order.dropoff, est["deliver_time"], vload + 1)
                self._records.append((feats, rider_name, oid, est["distance"]))

        # 无单可派的骑手显式 IDLE（保持动作字典覆盖全部 agent）
        for aid in self.env.agents:
            if not actions[aid]:
                actions[aid] = [0]
        return actions

    # ---- episode 级学习回路：实现效用回归 ----
    def learn_from_episode(self) -> int:
        if not self.scorer.learn:
            return 0
        n = 0
        for feats, _rider_name, oid, dist in self._records:
            order = self._order_by_id.get(oid)
            if order is None:
                continue
            target = self.scorer.realized_utility(
                order, dist, sim_end=float(getattr(self.env.sim, "_max_sim_time", 0.0) or 0.0) or None)
            self.scorer.update(feats, target)
            n += 1
        self._records.clear()
        return n
