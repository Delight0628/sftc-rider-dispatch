"""
运力商圈配送调度系统 - 仿真环境核心
====================================
比赛项目「运力评估商圈智能体」MAPPO 调度层环境

场景：多骑手（agent）从共享待派单池接单，FIFO 依次完成
"去取餐点 → 取餐 → 去送餐点 → 送达"，在承诺送达时间（due_date）约束下
最大化准时率/完成率并压缩总时长。

设计约束（与工厂环境 w_factory_env.py 严格对齐，MAPPO 训练栈零改动复用）：
- 观测 146 维布局：自身8(one-hot5+载负荷+繁忙率+离线) + 全局4 + 池摘要30
  + 候选订单10×10 + 未来订单4
- 动作 MultiDiscrete([11])：0=IDLE（不接单），1-10=接候选订单
- 全局状态 19 维：时间1 + 进度2 + 骑手状态5×3 + 平均利用率1
- final_stats 键名与工厂环境兼容（makespan/total_parts/total_tardiness/mean_utilization）
- infos 提供 global_state / action_mask / candidates_map / queue_snapshot / obs_meta

三层级联联调（双塔召回 → Wide&Deep 精排 → MAPPO 调度）：
- candidate_source='endogenous'（默认）：环境自采样候选（EDD5+最近3+随机2）
- candidate_source='upstream'：候选集合与意愿分来自上游精排
  config['upstream_candidates'] = {"骑手A": [{"order_id":1,"willingness":0.9}, ...], ...}
  （契约见 delivery_config.DELIVERY_UPSTREAM_CONFIG；可用
    generate_mock_upstream_candidates 生成 mock 上游做联调）
- 候选 10 维特征的 [1] 位承载"上游精排意愿分"（配送场景剩余段数恒为 2，原
  位置无信息量），故 146 维布局与网络/BC 教师索引保持不变
- 排序遵循会议口径"时间紧迫性优先于意愿排序"（upstream_order_by='urgency'）
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .delivery_config import (
    RIDERS, DELIVERY_GEO_CONFIG, ORDER_TYPES, ORDER_TYPE_LIST, ORDER_TYPE_ID,
    DELIVERY_BASE_ORDERS, DELIVERY_OBS_CONFIG, DELIVERY_ACTION_CONFIG,
    DELIVERY_REWARD_CONFIG, RIDER_OFFLINE_CONFIG, EMERGENCY_DELIVERY_ORDERS,
    DELIVERY_SIMULATION_TIME, DELIVERY_TIMEOUT_MULTIPLIER,
    DELIVERY_RIDER_RANDOMIZATION, DELIVERY_UPSTREAM_CONFIG,
    generate_random_delivery_orders, calculate_delivery_episode_score,
    normalize_upstream_candidates, heuristic_willingness,
)
from .w_factory_config import ENHANCED_OBS_CONFIG, REWARD_CONFIG

SILENT_MODE = True


# =============================================================================
# 1. 数据结构
# =============================================================================
class DeliveryOrder:
    """配送订单：一次取送（2 段行程）"""

    __slots__ = ("order_id", "order_type", "type_id", "pickup", "dropoff",
                 "ready_time", "due_date", "priority", "weight", "state",
                 "assigned_rider", "planned_deliver_time", "actual_deliver_time",
                 "planned_distance", "pull_time", "slack_at_pull")

    def __init__(self, order_id: int, order_type: str, pickup: Tuple[float, float],
                 dropoff: Tuple[float, float], ready_time: float, due_date: float,
                 priority: int, weight: float):
        self.order_id = order_id
        self.order_type = order_type
        self.type_id = ORDER_TYPE_ID.get(order_type, 0)
        self.pickup = tuple(pickup)
        self.dropoff = tuple(dropoff)
        self.ready_time = float(ready_time)
        self.due_date = float(due_date)
        self.priority = int(priority)
        self.weight = float(weight)
        # state: future -> pool -> carried -> delivered
        self.state = "future"
        self.assigned_rider: Optional[str] = None
        self.planned_deliver_time: Optional[float] = None
        self.actual_deliver_time: Optional[float] = None
        self.planned_distance: float = 0.0
        self.pull_time: Optional[float] = None
        self.slack_at_pull: Optional[float] = None

    def __repr__(self):
        return (f"Order({self.order_id},{self.order_type},due={self.due_date:.0f},"
                f"state={self.state})")


class RiderState:
    """骑手运行时状态（含执行链投影）"""

    __slots__ = ("name", "capacity", "speed", "position", "home",
                 "carry", "busy_time", "last_busy_update", "offline",
                 "offline_intervals", "offline_idx", "delivered_count",
                 "travel_distance", "idle_time")

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.capacity = int(config.get("capacity", 3))
        self.speed = float(config.get("speed", 0.55))
        self.home = tuple(config.get("home", (10.0, 10.0)))
        self.position = self.home
        self.carry: List[DeliveryOrder] = []
        self.busy_time = 0.0
        self.last_busy_update = 0.0
        self.offline = False
        self.offline_intervals: List[Tuple[float, float]] = []
        self.offline_idx = 0
        self.delivered_count = 0
        self.travel_distance = 0.0
        self.idle_time = 0.0

    # ---- 执行链投影：骑手完成当前携带队列后的位置与空闲时刻 ----
    def projected_free(self) -> Tuple[Tuple[float, float], float]:
        """返回 (预计空闲位置, 预计空闲时刻)。无携带单时即当前位置/当前即刻。"""
        if not self.carry:
            return self.position, -1.0  # -1 表示"立即可用"
        last = self.carry[-1]
        return last.dropoff, float(last.planned_deliver_time)

    def update_busy_time(self, now: float):
        """携带非空期间累计忙碌时间（事件推进时调用）"""
        if self.carry:
            self.busy_time += max(0.0, now - self.last_busy_update)
        self.last_busy_update = now


def _travel_time(p1: Tuple[float, float], p2: Tuple[float, float],
                 speed: float, geo: Dict[str, Any]) -> float:
    """两点行程时间（分钟）"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if str(geo.get("distance_metric", "euclidean")) == "manhattan":
        dist = abs(dx) + abs(dy)
    else:
        dist = float(np.hypot(dx, dy))
    return dist / max(1e-6, speed), dist


# =============================================================================
# 2. 配送仿真核心
# =============================================================================
class DeliverySim:
    """运力商圈配送调度仿真（事件驱动，替代 SimPy 的轻量实现）"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._training_mode = bool(self.config.get('training_mode', False))

        try:
            self._simulation_time = float(self.config.get('SIMULATION_TIME', DELIVERY_SIMULATION_TIME))
        except Exception:
            self._simulation_time = float(DELIVERY_SIMULATION_TIME)
        try:
            mult = float(self.config.get('SIMULATION_TIMEOUT_MULTIPLIER', DELIVERY_TIMEOUT_MULTIPLIER))
        except Exception:
            mult = float(DELIVERY_TIMEOUT_MULTIPLIER)
        self._max_sim_time = float(self.config.get('MAX_SIM_TIME', self._simulation_time * mult))

        self._geo = dict(DELIVERY_GEO_CONFIG)
        _geo_c = self.config.get('geo_config') or {}
        if isinstance(_geo_c, dict):
            self._geo.update(_geo_c)
        self._obs_cfg = dict(DELIVERY_OBS_CONFIG)
        self._reward_cfg = dict(DELIVERY_REWARD_CONFIG)

        # 骑手配置：默认模块 RIDERS；真实数据可经 config['riders'] 注入
        # （网络 146 维假设 5 骑手 one-hot，注入时请保持 5 名）
        self._riders_cfg = dict(RIDERS)
        _riders_c = self.config.get('riders')
        if isinstance(_riders_c, dict) and _riders_c:
            self._riders_cfg = {str(k): dict(v) for k, v in _riders_c.items()}

        # 智能体 = 骑手（与工厂 5 工作站对齐）
        self.agents = [f"agent_{name}" for name in self._riders_cfg.keys()]
        self.rider_names = list(self._riders_cfg.keys())

        # 动态事件（兼容工厂键名）
        self._rider_offline_enabled = bool(self.config.get('rider_offline_enabled', False)) or \
            bool(self.config.get('equipment_failure_enabled', False))
        if 'disable_failures' in self.config:
            self._rider_offline_enabled = not bool(self.config.get('disable_failures'))
        self._emergency_orders_enabled = bool(self.config.get('emergency_orders_enabled', False))

        self._offline_cfg = dict(RIDER_OFFLINE_CONFIG)
        _c = self.config.get('equipment_failure_config', {})
        if _c:
            self._offline_cfg.update(_c)
        self._emergency_cfg = dict(EMERGENCY_DELIVERY_ORDERS)
        _c = self.config.get('emergency_orders_config', {})
        if _c:
            self._emergency_cfg.update(_c)

        self._deterministic_candidates = bool(self.config.get('deterministic_candidates', False))
        self._randomize_env = bool(self.config.get('randomize_env', False))

        # ---- 三层级联联调：上游（双塔召回 + Wide&Deep 精排）候选与意愿分 ----
        # candidate_source='upstream' 时直接采用上游候选集合，排序遵循会议口径
        # （时间紧迫性优先）；'endogenous' 保持环境自采样（EDD5 + 最近3 + 随机2）
        self._upstream_cfg = dict(DELIVERY_UPSTREAM_CONFIG)
        _up = self.config.get('upstream_config') or {}
        if isinstance(_up, dict):
            self._upstream_cfg.update(_up)
        self._candidate_source = str(
            self.config.get('candidate_source', self._upstream_cfg.get('candidate_source', 'endogenous'))
        ).lower()
        self._upstream_order_by = str(
            self.config.get('upstream_order_by', self._upstream_cfg.get('order_by', 'urgency'))
        ).lower()
        self._upstream_top_k = int(self._upstream_cfg.get('top_k', 10))
        self._willingness_clip = tuple(self._upstream_cfg.get('willingness_clip', (0.0, 1.0)))
        self._willingness_jitter = float(self._upstream_cfg.get('willingness_jitter', 0.0) or 0.0)
        self._backfill_from_pool = bool(self._upstream_cfg.get('backfill_from_pool', True))
        self._upstream_norm = normalize_upstream_candidates(
            self.config.get('upstream_candidates'), self.rider_names, self._upstream_top_k)
        self._upstream_index = {
            r: {int(e["order_id"]): float(e["willingness"]) for e in entries}
            for r, entries in self._upstream_norm.items()
        }
        self._upstream_order = {
            r: [int(e["order_id"]) for e in entries] for r, entries in self._upstream_norm.items()
        }

        # 状态容器
        self.orders: List[DeliveryOrder] = []
        self.pool: List[DeliveryOrder] = []          # 已到达未接
        self.future: List[DeliveryOrder] = []        # 未到达
        self.delivered: List[DeliveryOrder] = []
        self.riders: Dict[str, RiderState] = {}
        self.current_time = 0.0
        self.simulation_ended = False
        self._last_reward_time = 0.0

        self._cached_candidates: Dict[str, List[Dict[str, Any]]] = {}
        self._candidate_action_start = 1
        self._candidate_action_end = int(self._obs_cfg.get("num_candidate_orders", 10))

        # 统计
        self.stats: Dict[str, Any] = {}
        self.event_timeline: List[Dict[str, Any]] = []
        self.gantt_chart_history: List[Dict[str, Any]] = []

        self.reset()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def reset(self):
        self.current_time = 0.0
        self.simulation_ended = False
        self._last_reward_time = 0.0
        self.orders = []
        self.pool = []
        self.future = []
        self.delivered = []
        self.riders = {}
        self._cached_candidates.clear()
        self.event_timeline = []
        self.gantt_chart_history = []

        # ---- 骑手 ----
        for name, rcfg in self._riders_cfg.items():
            rider = RiderState(name, rcfg)
            if self._randomize_env:
                jit = DELIVERY_RIDER_RANDOMIZATION
                pos = (float(np.clip(rider.home[0] + np.random.uniform(-jit["position_jitter_km"], jit["position_jitter_km"]), 0, self._geo["grid_size"])),
                       float(np.clip(rider.home[1] + np.random.uniform(-jit["position_jitter_km"], jit["position_jitter_km"]), 0, self._geo["grid_size"])))
                rider.position = pos
                rider.home = pos
                rider.speed = max(0.2, rider.speed + np.random.uniform(-jit["speed_jitter"], jit["speed_jitter"]))
            self.riders[name] = rider

        # ---- 订单 ----
        custom_orders = self.config.get('custom_orders')
        if custom_orders is not None:
            order_cfgs = list(custom_orders)
        else:
            orders_scale = float(self.config.get('orders_scale', 1.0))
            order_cfgs = generate_random_delivery_orders(orders_scale=orders_scale)

        for i, oc in enumerate(order_cfgs):
            self._append_order(i, oc)

        # randomize_env：订单轻微抖动（时效/就绪时刻）
        if self._randomize_env and custom_orders is not None:
            for o in self.orders:
                o.due_date = max(o.ready_time + 10.0, o.due_date + np.random.uniform(-8.0, 8.0))
                o.ready_time = max(0.0, o.ready_time + np.random.uniform(-5.0, 5.0))

        # ---- 紧急订单（阶段二动态事件，预生成到达流） ----
        if self._emergency_orders_enabled:
            self._generate_emergency_orders(start_id=10000)

        # ---- 骑手离线计划（阶段二动态事件） ----
        if self._rider_offline_enabled:
            self._generate_offline_intervals()

        # 分桶：future / pool
        for o in self.orders:
            if o.ready_time <= self.current_time:
                o.state = "pool"
                self.pool.append(o)
            else:
                o.state = "future"
                self.future.append(o)
        self.future.sort(key=lambda o: o.ready_time)

        # 空订单集保护：直接结束，避免无事件可推进
        if not self.orders:
            self.simulation_ended = True

        # ---- 统计 ----
        self.stats = {
            'completed_orders': 0,
            'makespan': 0.0,
            'total_tardiness': 0.0,
            'max_tardiness': 0.0,
            'total_parts': 0,
            'total_orders': len(self.orders),
            'on_time_count': 0,
            'total_distance': 0.0,
            'idle_when_work_available_count': 0,
            'invalid_action_count': 0,
            'race_conflict_count': 0,
            'emergency_orders_inserted_count': max(0, len(self.orders) - len(order_cfgs)),
            # 三层联调（上游精排候选）统计
            'upstream_matched_count': 0,      # 上游候选在待派池中兑现的次数（人次）
            'upstream_missing_count': 0,      # 上游候选不在池中（已送达/未到达）的次数
            'upstream_fallback_count': 0,     # 该骑手无上游候选 → 回退自采样的次数
        }

    def _append_order(self, idx: int, oc: Dict[str, Any]):
        order = DeliveryOrder(
            order_id=int(oc.get('order_id', idx)),
            order_type=str(oc.get('order_type', ORDER_TYPE_LIST[0])),
            pickup=tuple(oc.get('pickup', (10.0, 10.0))),
            dropoff=tuple(oc.get('dropoff', (10.0, 10.0))),
            ready_time=float(oc.get('ready_time', 0.0)),
            due_date=float(oc.get('due_date', 240.0)),
            priority=int(oc.get('priority', 2)),
            weight=float(oc.get('weight', 1.0)),
        )
        self.orders.append(order)

    def _generate_emergency_orders(self, start_id: int):
        """按泊松到达流生成紧急订单（紧时效）"""
        rate_per_min = float(self._emergency_cfg.get('arrival_rate', 0.15)) / 60.0
        due_reduction = float(self._emergency_cfg.get('due_date_reduction', 0.5))
        boost = int(self._emergency_cfg.get('priority_boost', 1))
        horizon = self._simulation_time
        t = 0.0
        idx = 0
        while t < horizon:
            gap = float(np.random.exponential(1.0 / max(1e-9, rate_per_min)))
            t += gap
            if t >= horizon:
                break
            geo = self._geo
            grid = float(geo["grid_size"])
            pickup = (float(np.random.uniform(0, grid)), float(np.random.uniform(0, grid)))
            dropoff = (float(np.random.uniform(0, grid)), float(np.random.uniform(0, grid)))
            tt, dist = _travel_time(pickup, dropoff, 0.55, geo)
            route = tt + geo.get("pickup_service_time", 3.0) + geo.get("dropoff_service_time", 2.0)
            buffer_base = 35.0
            due = t + buffer_base * (1.0 - min(0.8, due_reduction)) + route
            oc = {
                'order_id': start_id + idx, 'order_type': '文件急送',
                'pickup': pickup, 'dropoff': dropoff, 'ready_time': t,
                'due_date': due, 'priority': max(1, 2 - boost), 'weight': 0.5,
            }
            self._append_order(start_id + idx, oc)
            self.event_timeline.append({
                "type": "emergency_order", "time": round(t, 2),
                "order_id": start_id + idx,
            })
            idx += 1

    def _generate_offline_intervals(self):
        """为每个骑手预生成离线区间（mtbf/mttr）"""
        cfg = self._offline_cfg
        prob_per_min = float(cfg.get('failure_probability', 0.03)) / 60.0
        mttr = float(cfg.get('mttr_minutes', 20.0))
        horizon = self._simulation_time
        for rider in self.riders.values():
            t = 0.0
            while t < horizon:
                if np.random.random() < prob_per_min * min(horizon - t, 60.0):
                    start = t + float(np.random.uniform(0, 60.0))
                    dur = max(5.0, float(np.random.exponential(mttr)))
                    if start < horizon:
                        rider.offline_intervals.append((start, min(horizon, start + dur)))
                        self.event_timeline.append({
                            "type": "rider_offline", "rider": rider.name,
                            "start": round(start, 2), "end": round(min(horizon, start + dur), 2),
                        })
                        t = start + dur
                        continue
                t += 60.0
            rider.offline_idx = 0

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------
    def is_done(self) -> bool:
        if self.simulation_ended:
            return True
        if self.current_time >= self._max_sim_time:
            return True
        return len(self.delivered) >= len(self.orders) and len(self.orders) > 0

    def _rider_is_offline(self, rider: RiderState, t: float) -> bool:
        while rider.offline_idx < len(rider.offline_intervals):
            s, e = rider.offline_intervals[rider.offline_idx]
            if t < s:
                return False
            if t <= e:
                return True
            rider.offline_idx += 1
        return False

    def _estimate_route(self, rider: RiderState, order: DeliveryOrder,
                        start_pos: Tuple[float, float], start_time: float) -> Dict[str, float]:
        """估算骑手从 start_pos/start_time 出发完成订单的路线（含服务时间）"""
        geo = self._geo
        t1, d1 = _travel_time(start_pos, order.pickup, rider.speed, geo)
        pickup_done = start_time + t1 + float(geo.get("pickup_service_time", 3.0))
        t2, d2 = _travel_time(order.pickup, order.dropoff, rider.speed, geo)
        deliver_time = pickup_done + t2 + float(geo.get("dropoff_service_time", 2.0))
        return {
            "leg1_time": t1, "leg2_time": t2,
            "total_time": deliver_time - start_time,
            "deliver_time": deliver_time,
            "distance": d1 + d2,
        }

    def _order_slack(self, rider: RiderState, order: DeliveryOrder, now: float) -> float:
        """订单相对某骑手的 slack：due - 预计完成时刻（负=预计迟到）"""
        pos, free_t = rider.projected_free()
        start = max(now, free_t if free_t > 0 else now)
        est = self._estimate_route(rider, order, pos, start)
        return order.due_date - est["deliver_time"]

    def _pickup_congestion(self, order: DeliveryOrder) -> float:
        """取餐点区域拥堵度 ∈ [0,1]：半径内待派订单密度 + 空闲骑手竞争"""
        radius = float(self._geo.get("congestion_radius_km", 2.0))
        same_zone_pool = 0
        for other in self.pool:
            if other is order:
                continue
            if np.hypot(other.pickup[0] - order.pickup[0], other.pickup[1] - order.pickup[1]) <= radius:
                same_zone_pool += 1
        riders_near = 0
        for r in self.riders.values():
            if np.hypot(r.position[0] - order.pickup[0], r.position[1] - order.pickup[1]) <= radius:
                riders_near += 1
        supply = max(1, riders_near)
        congestion = same_zone_pool / (supply + same_zone_pool)
        return float(np.clip(congestion, 0.0, 1.0))

    def _compressed(self, x: float, norm: float) -> float:
        """非负压缩归一化 y=(x/n)/(1+x/n)"""
        if norm <= 0:
            norm = 1.0
        v = max(0.0, x) / norm
        return v / (1.0 + v)

    def _willingness_for(self, rider_name: str, order: "DeliveryOrder") -> float:
        """候选订单的"接单意愿分" ∈ [0,1]（对应上游 Wide&Deep 精排输出）。

        三层联调语义：意愿分由上游（双塔召回 + W&D 精排）给出；若上游未覆盖
        该订单，则按订单属性做兜底估计（保证无上游数据时训练/评估仍可进行）。
        """
        v = self._upstream_index.get(rider_name, {}).get(int(order.order_id), None)
        if v is None:
            v = heuristic_willingness({
                "order_type": order.order_type,
                "priority": order.priority,
                "weight": order.weight,
            })
            if self._willingness_jitter > 0.0:
                v = float(np.clip(
                    v + np.random.uniform(-self._willingness_jitter, self._willingness_jitter), 0.0, 1.0))
        lo, hi = self._willingness_clip
        return float(np.clip(v, lo, hi))

    # ---- 候选订单采样 ----
    # endogenous：EDD5 + 最近3 + 随机2（环境自采样，原逻辑）
    # upstream   ：候选集合来自上游精排，排序按会议口径（时间紧迫性优先）
    def _get_candidate_orders(self, rider_name: str) -> List[Dict[str, Any]]:
        if rider_name in self._cached_candidates:
            return self._cached_candidates[rider_name]

        rider = self.riders[rider_name]
        now = self.current_time
        n_urgent = int(self._obs_cfg.get("num_urgent_candidates", 5))
        n_near = int(self._obs_cfg.get("num_near_candidates", 3))
        n_rand = int(self._obs_cfg.get("num_random_candidates", 2))
        k_max = int(self._obs_cfg.get("num_candidate_orders", 10))

        candidates: List[Dict[str, Any]] = []
        if self.pool:
            pos, free_t = rider.projected_free()
            start = max(now, free_t if free_t > 0 else now)
            scored = []
            for o in self.pool:
                slack = self._order_slack(rider, o, now)
                est = self._estimate_route(rider, o, pos, start)
                scored.append((o, slack, est["leg1_time"]))
            # EDD 类比：slack 最小（最紧急）优先 —— 时间紧迫性口径
            by_slack = sorted(scored, key=lambda x: (x[1], x[0].order_id))
            by_id = {o.order_id: (o, slack) for o, slack, _ in scored}

            up_ids = self._upstream_order.get(rider_name) if self._candidate_source == 'upstream' else None
            if up_ids:
                # ---- 上游精排候选：集合来自上游（最可能接的单），时间紧迫性决定顺序 ----
                picked, missed = [], 0
                for oid in up_ids:
                    hit = by_id.get(int(oid))
                    if hit is None:
                        missed += 1        # 已送达 / 未到达 / 不在池中
                        continue
                    if hit[0] not in picked:
                        picked.append(hit[0])
                self.stats['upstream_matched_count'] += len(picked)
                self.stats['upstream_missing_count'] += missed

                if self._upstream_order_by == 'upstream':
                    ordered = list(picked)                     # 保持上游意愿序
                else:
                    order_key = {o.order_id: i for i, (o, _, _) in enumerate(by_slack)}
                    ordered = sorted(picked, key=lambda o: order_key[o.order_id])

                if self._backfill_from_pool and len(ordered) < k_max:
                    chosen = set(ordered)
                    for o, _, _ in by_slack:
                        if len(ordered) >= k_max:
                            break
                        if o not in chosen:
                            ordered.append(o)
                            chosen.add(o)
                candidates = ordered[:k_max]
            else:
                if self._candidate_source == 'upstream':
                    # 该骑手上游无候选（或上游候选全部不可用）→ 回退自采样
                    self.stats['upstream_fallback_count'] += 1
                for o, slack, leg1 in by_slack[:n_urgent]:
                    candidates.append(o)
                # SPT 类比：取餐点最近（首段最短）优先
                by_near = sorted(scored, key=lambda x: (x[2], x[0].order_id))
                for o, slack, leg1 in by_near[:n_near]:
                    if o not in candidates:
                        candidates.append(o)
                # 随机多样性
                rest = [o for o, _, _ in scored if o not in candidates]
                if rest:
                    if self._deterministic_candidates:
                        pick = rest[:n_rand]
                    else:
                        ids = np.random.choice(len(rest), size=min(n_rand, len(rest)), replace=False)
                        pick = [rest[int(i)] for i in ids]
                    candidates.extend(pick)
                candidates = candidates[:k_max]

        result = [{"order": c, "index": i} for i, c in enumerate(candidates)]
        self._cached_candidates[rider_name] = result
        return result


    def _get_action_mask(self, rider_name: str) -> np.ndarray:
        k_max = int(self._obs_cfg.get("num_candidate_orders", 10))
        mask = np.zeros((1 + k_max,), dtype=np.float32)
        mask[0] = 1.0  # IDLE 恒可用
        rider = self.riders[rider_name]
        if len(rider.carry) < rider.capacity and not self._rider_is_offline(rider, self.current_time):
            cands = self._get_candidate_orders(rider_name)
            for c in cands:
                a = self._candidate_action_start + c["index"]
                if a < mask.shape[0]:
                    mask[a] = 1.0
        return mask

    # ------------------------------------------------------------------
    # 观测构建（146 维，布局与工厂环境对齐）
    # ------------------------------------------------------------------
    def get_state_for_agent(self, agent_id: str) -> np.ndarray:
        rider_name = agent_id.replace("agent_", "")
        rider = self.riders[rider_name]
        now = self.current_time
        geo = self._geo
        obs_cfg = self._obs_cfg

        feats: List[float] = []

        # ---- [1] 自身特征 (8) ----
        one_hot = np.zeros((len(self.rider_names),), dtype=np.float32)
        one_hot[self.rider_names.index(rider_name)] = 1.0
        feats.extend(one_hot.tolist())
        feats.append(len(rider.carry) / max(1, rider.capacity))            # 载负荷
        feats.append(rider.busy_time / max(1.0, now))                       # 繁忙率
        feats.append(1.0 if self._rider_is_offline(rider, now) else 0.0)   # 离线

        # ---- [2] 全局特征 (4) ----
        total = max(1, len(self.orders))
        busy_riders = sum(1 for r in self.riders.values() if len(r.carry) >= r.capacity)
        feats.append(min(1.0, now / max(1.0, self._simulation_time)))       # 时间进度
        feats.append(len(self.pool) / total)                                # 待派池占比
        feats.append(busy_riders / max(1, len(self.riders)))                # 运力饱和度
        feats.append(self._compressed(len(self.pool), float(obs_cfg.get("pool_len_norm", 20.0))))

        # ---- [3] 待派池摘要 (6 特征 × 5 统计 = 30) ----
        pool_feats = np.zeros((int(obs_cfg.get("pool_summary_features", 6))
                               * int(obs_cfg.get("pool_summary_stats", 5)),), dtype=np.float32)
        if self.pool:
            pos, free_t = rider.projected_free()
            start = max(now, free_t if free_t > 0 else now)
            cols = []
            for o in self.pool:
                est = self._estimate_route(rider, o, pos, start)
                cols.append([
                    self._compressed(est["leg1_time"], float(obs_cfg.get("max_op_duration_norm", 60.0))),   # 到取餐点耗时
                    self._compressed(o.weight, 10.0),                                                          # 重量
                    self._compressed(est["total_time"], float(obs_cfg.get("total_remaining_time_norm", 120.0))),  # 全程耗时
                    self._pickup_congestion(o),                                                                # 取餐区拥堵
                    (o.priority - 1) / 2.0,                                                                    # 优先级
                    1.0 if o.priority == 1 else 0.0,                                                           # 是否紧急
                ])
            arr = np.asarray(cols, dtype=np.float32)
            pool_feats = np.concatenate([
                arr.min(axis=0), arr.max(axis=0), arr.mean(axis=0),
                arr.std(axis=0), np.median(arr, axis=0),
            ])
        feats.extend(pool_feats.tolist())

        # ---- [4] 候选订单 (10 × 10) ----
        slack_norm = float(obs_cfg.get("slack_time_norm", 120.0))
        n_types = max(1, len(ORDER_TYPE_LIST))
        cands = self._get_candidate_orders(rider_name)
        cand_block = np.zeros((int(obs_cfg.get("num_candidate_orders", 10)),
                               int(obs_cfg.get("candidate_feature_dim", 10))), dtype=np.float32)
        pos, free_t = rider.projected_free()
        start = max(now, free_t if free_t > 0 else now)
        for c in cands:
            o: DeliveryOrder = c["order"]
            est = self._estimate_route(rider, o, pos, start)
            slack = o.due_date - est["deliver_time"]
            row = [
                1.0,                                                                       # exists
                self._willingness_for(rider_name, o),                                      # [1] 上游精排意愿分
                self._compressed(est["total_time"], float(obs_cfg.get("total_remaining_time_norm", 120.0))),
                self._compressed(est["leg1_time"], float(obs_cfg.get("max_op_duration_norm", 60.0))),  # ≈opdur
                self._pickup_congestion(o),                                                # ≈downstream_congestion
                (o.priority - 1) / 2.0,
                1.0 if o.priority == 1 else 0.0,                                           # 紧急单
                o.type_id / n_types,                                                       # 品类
                float(np.clip(1.0 - slack / slack_norm, 0.0, 1.0)),                        # time_pressure
                float(np.clip(slack / slack_norm, -3.0, 3.0)),                             # slack
            ]
            cand_block[c["index"]] = row
        feats.extend(cand_block.reshape(-1).tolist())

        # ---- [5] 未来订单摘要 (4) ----
        fut = np.zeros((4,), dtype=np.float32)
        if self.future:
            gaps = [f.ready_time - now for f in self.future]
            urgent_frac = sum(1 for f in self.future if f.priority == 1) / len(self.future)
            fut = np.asarray([
                self._compressed(len(self.future), 20.0),
                self._compressed(max(0.0, min(gaps)), 120.0),
                urgent_frac,
                self._compressed(float(np.mean([f.weight for f in self.future])), 10.0),
            ], dtype=np.float32)
        feats.extend(fut.tolist())

        return np.asarray(feats, dtype=np.float32)

    def get_global_state(self) -> np.ndarray:
        """全局状态 19 维（集中式 Critic 输入，布局与工厂一致）"""
        now = self.current_time
        total = max(1, len(self.orders))
        g = [
            min(1.0, now / max(1.0, self._simulation_time)),   # 时间
            len(self.delivered) / total,                        # 完成率
            len(self.pool) / total,                             # 待派率
        ]
        for name in self.rider_names:
            r = self.riders[name]
            g.extend([
                len(r.carry) / max(1, r.capacity),
                r.busy_time / max(1.0, now),
                1.0 if self._rider_is_offline(r, now) else 0.0,
            ])
        utils = [r.busy_time / max(1.0, now) for r in self.riders.values()]
        g.append(float(np.mean(utils)) if utils else 0.0)
        return np.asarray(g, dtype=np.float32)

    # ------------------------------------------------------------------
    # 动作执行与事件推进
    # ------------------------------------------------------------------
    def step_with_actions(self, actions: Dict[str, Any]) -> Dict[str, float]:
        decision_time = self.current_time
        action_context: Dict[str, Dict[str, Any]] = {}

        for agent_id, agent_action in actions.items():
            rider_name = agent_id.replace("agent_", "")
            rider = self.riders[rider_name]
            if not isinstance(agent_action, (list, np.ndarray)):
                agent_action = [agent_action]

            ctx = {
                "action": agent_action,
                "decision_time": decision_time,
                "started_orders": [],
                "invalid_attempts": 0,
                "race_conflicts": 0,       # 同步决策被同伴抢先（观测时合法，不应惩罚）
                "idle_when_available": False,
                "assigned_rider": rider_name,
            }
            action_context[agent_id] = ctx

            offline = self._rider_is_offline(rider, decision_time)
            has_capacity = len(rider.carry) < rider.capacity
            cands = self._get_candidate_orders(rider_name)
            can_pull = (not offline) and has_capacity and len(cands) > 0

            for act in agent_action:
                act = int(act)
                if act <= 0:
                    if can_pull:
                        ctx["idle_when_available"] = True
                        self.stats['idle_when_work_available_count'] += 1
                    continue
                cand_idx = act - self._candidate_action_start
                target = None
                race_conflict = False
                if cand_idx < len(cands):
                    o = cands[cand_idx]["order"]
                    if o.state == "pool":
                        target = o
                    elif o.state == "carried":
                        # 同步决策下被更早处理的同伴接走：观测时该动作合法，记竞态而非无效
                        race_conflict = True
                if target is not None and not can_pull:
                    # 超容量/离线导致的失败仍属无效动作
                    target = None
                if target is None:
                    if race_conflict:
                        ctx["race_conflicts"] += 1
                        self.stats['race_conflict_count'] = self.stats.get('race_conflict_count', 0) + 1
                    else:
                        ctx["invalid_attempts"] += 1
                        self.stats['invalid_action_count'] += 1
                    continue

                # ---- 接单：从池中移除，追加到骑手执行链 ----
                self.pool.remove(target)
                target.state = "carried"
                target.assigned_rider = rider_name
                pos, free_t = rider.projected_free()
                start = max(decision_time, free_t if free_t > 0 else decision_time)
                est = self._estimate_route(rider, target, pos, start)
                target.planned_deliver_time = est["deliver_time"]
                target.planned_distance = est["distance"]
                target.pull_time = decision_time
                target.slack_at_pull = target.due_date - est["deliver_time"]
                rider.carry.append(target)
                rider.travel_distance += est["distance"]
                ctx["started_orders"].append({
                    "order_id": target.order_id,
                    "slack": target.slack_at_pull,
                })
                self.gantt_chart_history.append({
                    "rider": rider_name, "order_id": target.order_id,
                    "start": round(start, 2), "end": round(est["deliver_time"], 2),
                    "pickup": target.pickup, "dropoff": target.dropoff,
                })
                # 接单后容量变化影响后续动作有效性
                has_capacity = len(rider.carry) < rider.capacity
                can_pull = (not offline) and has_capacity and len(cands) > 0

        # ---- 推进到下一个决策相关事件 ----
        self._advance_to_next_epoch()
        self._cached_candidates.clear()

        # ---- 奖励 ----
        rewards = self.get_rewards(action_context)
        return rewards

    def _next_event_time(self) -> float:
        """下一个决策相关事件时刻：订单到达 / 骑手队列清空 / 离线边界 / 仿真结束"""
        now = self.current_time
        t_next = self._max_sim_time
        if self.future:
            t_next = min(t_next, self.future[0].ready_time)
        for r in self.riders.values():
            if r.carry:
                # FIFO 链上最早完成的订单（carry[0]）才是下一个事件：
                # 若用 carry[-1]（最晚），链中间订单送达时不会产生决策点，
                # 容量释放/位置更新/送达奖励都会被推迟到整链结束
                t_next = min(t_next, float(r.carry[0].planned_deliver_time))
            while r.offline_idx < len(r.offline_intervals):
                s, e = r.offline_intervals[r.offline_idx]
                if s > now:
                    t_next = min(t_next, s)
                    break
                if e > now:
                    t_next = min(t_next, e)
                    break
                r.offline_idx += 1
        return t_next

    def _advance_to_next_epoch(self):
        """执行事件并推进到下一个决策时刻（时间严格前进）"""
        now = self.current_time
        t_target = self._next_event_time()
        if t_target <= now:
            t_target = now + 1e-6
        t_target = min(t_target, self._max_sim_time)

        # 忙碌时长入账（至目标时刻）
        for r in self.riders.values():
            r.update_busy_time(t_target)

        # 订单到达
        while self.future and self.future[0].ready_time <= t_target:
            o = self.future.pop(0)
            o.state = "pool"
            self.pool.append(o)

        # 送达结算（骑行队列按计划时刻完成；跨骑手按送达时刻排序，保证事件线时间单调）
        pending_deliveries = []
        for r in self.riders.values():
            for o in r.carry:
                if o.planned_deliver_time <= t_target:
                    pending_deliveries.append((r, o))
        pending_deliveries.sort(key=lambda x: (x[1].planned_deliver_time, x[0].name))
        for r, o in pending_deliveries:
            r.carry.remove(o)
            o.state = "delivered"
            o.actual_deliver_time = o.planned_deliver_time
            self.delivered.append(o)
            r.delivered_count += 1
            # 送达后骑手位置更新为送达点（影响后续接单距离/拥堵/观测）
            r.position = o.dropoff
            lateness = max(0.0, o.actual_deliver_time - o.due_date)
            self.stats['total_tardiness'] += lateness
            self.stats['max_tardiness'] = max(self.stats['max_tardiness'], lateness)
            if lateness <= 0.0:
                self.stats['on_time_count'] += 1
            self.stats['makespan'] = max(self.stats['makespan'], o.actual_deliver_time)
            self.stats['total_distance'] += o.planned_distance
            self.event_timeline.append({
                "type": "delivered", "rider": r.name, "order_id": o.order_id,
                "time": round(o.actual_deliver_time, 2),
                "lateness": round(lateness, 2),
            })

        self.current_time = t_target
        self.stats['completed_orders'] = len(self.delivered)
        self.stats['total_parts'] = len(self.delivered)

        if self.current_time >= self._max_sim_time:
            self.simulation_ended = True

    # ------------------------------------------------------------------
    # 奖励（配送语义，键名沿用工厂配置）
    # ------------------------------------------------------------------
    def get_rewards(self, action_context: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        cfg = self._reward_cfg
        rewards = {agent_id: 0.0 for agent_id in self.agents}
        n_riders = max(1, len(self.agents))
        now = self.current_time
        slack_norm = float(self._obs_cfg.get("slack_time_norm", 120.0))

        # 1) 本步送达结算（接单骑手得奖励）
        newly = [o for o in self.delivered if o.pull_time is not None
                 and o.actual_deliver_time is not None
                 and o.actual_deliver_time > (self._last_reward_time or -1.0)
                 and o.actual_deliver_time <= now]
        for o in newly:
            agent_id = f"agent_{o.assigned_rider}" if o.assigned_rider else None
            if agent_id is None or agent_id not in rewards:
                continue
            rewards[agent_id] += float(cfg.get("order_completion_reward", 80.0))
            if o.actual_deliver_time <= o.due_date:
                rewards[agent_id] += float(cfg.get("on_time_completion_reward", 80.0))
            else:
                lateness_norm = (o.actual_deliver_time - o.due_date) / slack_norm
                if cfg.get("use_huber_tardiness", True):
                    # Huber：小段二次 0.5t²，大段线性 δ(t-0.5δ)（与工厂公式一致）
                    delta = float(cfg.get("tardiness_huber_delta_norm", 0.3))
                    ax = abs(lateness_norm)
                    if ax <= delta:
                        huber_val = 0.5 * (lateness_norm ** 2)
                    else:
                        huber_val = delta * (ax - 0.5 * delta)
                    pen = float(cfg.get("tardiness_penalty_scaler", -4.0)) * huber_val
                else:
                    pen = float(cfg.get("tardiness_penalty_scaler", -4.0)) * lateness_norm
                rewards[agent_id] += pen
        self._last_reward_time = now

        # 2) 负 slack 持续惩罚（池内 + 携带中订单的时间压力）
        coeff = float(cfg.get("slack_time_penalty_coeff", -0.03))
        if coeff != 0.0:
            tanh_scale = float(cfg.get("slack_penalty_tanh_scale", 60.0))
            max_abs = float(cfg.get("slack_penalty_max_abs", 50.0))
            pressure_team = 0.0
            # 池内订单：按订单自身取送路线估算（参考速度下界，独立于具体骑手）
            ref_speed = float(self._geo.get("speed_norm", 0.55))
            for o in self.pool:
                t1, _ = _travel_time(o.pickup, o.dropoff, ref_speed, self._geo)
                est_route = t1 + float(self._geo.get("pickup_service_time", 3.0)) + float(self._geo.get("dropoff_service_time", 2.0))
                slack = o.due_date - now - est_route
                if slack < 0:
                    pressure_team += float(np.tanh(-slack / tanh_scale))
            for r in self.riders.values():
                for o in r.carry:
                    slack = o.due_date - float(o.planned_deliver_time)
                    if slack < 0:
                        pressure_team += float(np.tanh(-slack / tanh_scale))
            if pressure_team > 0:
                per = float(np.clip(coeff * pressure_team, -max_abs, 0.0)) / n_riders
                for agent_id in rewards:
                    rewards[agent_id] += per

        # 3) 池积压惩罚
        wip_coeff = float(cfg.get("wip_penalty_coeff", -0.01))
        if wip_coeff != 0.0 and self.pool:
            per = wip_coeff * len(self.pool) / n_riders
            for agent_id in rewards:
                rewards[agent_id] += per

        # 4) 动作质量（无效 / 有单不接）
        for agent_id, ctx in action_context.items():
            if agent_id not in rewards:
                continue
            rewards[agent_id] += float(cfg.get("invalid_action_penalty", -0.5)) * ctx["invalid_attempts"]
            if ctx.get("idle_when_available", False):
                rewards[agent_id] += float(cfg.get("idle_when_work_available_penalty", -1.0))

        # 5) 全部送达团队奖励
        if len(self.delivered) >= len(self.orders) and len(self.orders) > 0:
            bonus = float(cfg.get("final_all_orders_completion_bonus", 500.0)) / n_riders
            for agent_id in rewards:
                rewards[agent_id] += bonus

        return rewards

    # ------------------------------------------------------------------
    # 终局统计（键名与工厂环境兼容）
    # ------------------------------------------------------------------
    def get_final_stats(self) -> Dict[str, Any]:
        total = len(self.orders)
        delivered = len(self.delivered)
        now = max(self.current_time, 1e-6)
        rider_utils = {name: (r.busy_time / now) for name, r in self.riders.items()}
        mean_util = float(np.mean(list(rider_utils.values()))) if rider_utils else 0.0

        on_time = self.stats.get('on_time_count', 0)
        delivery_times = [o.actual_deliver_time - o.ready_time for o in self.delivered
                          if o.actual_deliver_time is not None]

        result = {
            'makespan': float(self.stats.get('makespan', 0.0) or (self.current_time if delivered else 0.0)),
            'total_parts': int(delivered),
            'total_orders': int(total),
            'total_tardiness': float(self.stats.get('total_tardiness', 0.0)),
            'max_tardiness': float(self.stats.get('max_tardiness', 0.0)),
            'avg_tardiness': float(self.stats.get('total_tardiness', 0.0) / delivered) if delivered > 0 else 0.0,
            'mean_utilization': mean_util,
            'equipment_utilization': rider_utils,
            'on_time_rate': (on_time / delivered) if delivered > 0 else 0.0,
            'on_time_count': int(on_time),
            'avg_delivery_time': float(np.mean(delivery_times)) if delivery_times else 0.0,
            'total_distance': float(self.stats.get('total_distance', 0.0)),
            'distance_per_order': (float(self.stats.get('total_distance', 0.0)) / delivered) if delivered > 0 else 0.0,
            'completion_rate': (delivered / total) if total > 0 else 0.0,
            'undelivered_count': int(total - delivered),
            'idle_when_work_available_count': int(self.stats.get('idle_when_work_available_count', 0)),
            'invalid_action_count': int(self.stats.get('invalid_action_count', 0)),
            'race_conflict_count': int(self.stats.get('race_conflict_count', 0)),
            'emergency_orders_inserted_count': int(self.stats.get('emergency_orders_inserted_count', 0)),
            # 超时分桶（对齐数仓 ol_fin_late_* 口径）
            'late_gt_5m_rate': 0.0,
            'late_gt_15m_rate': 0.0,
            'late_gt_30m_rate': 0.0,
            # 三层联调（上游精排候选）口径与兑现情况
            'candidate_source': self._candidate_source,
            'upstream_order_by': self._upstream_order_by,
            'upstream_riders_covered': len(self._upstream_order),
            'upstream_matched_count': int(self.stats.get('upstream_matched_count', 0)),
            'upstream_missing_count': int(self.stats.get('upstream_missing_count', 0)),
            'upstream_fallback_count': int(self.stats.get('upstream_fallback_count', 0)),
            'rider_delivered': {name: r.delivered_count for name, r in self.riders.items()},
            'gantt_chart_history': self.gantt_chart_history,
            'event_timeline': list(self.event_timeline),
        }
        if delivered > 0:
            late5 = late15 = late30 = 0
            for o in self.delivered:
                if o.actual_deliver_time is None:
                    continue
                late = max(0.0, o.actual_deliver_time - o.due_date)
                if late > 5:
                    late5 += 1
                if late > 15:
                    late15 += 1
                if late > 30:
                    late30 += 1
            result['late_gt_5m_rate'] = late5 / delivered
            result['late_gt_15m_rate'] = late15 / delivered
            result['late_gt_30m_rate'] = late30 / delivered
        return result

    # 兼容工厂训练器对 env.sim 的个别探测
    _trigger_strategy_reset = False


# =============================================================================
# 3. PettingZoo 多智能体环境接口
# =============================================================================
class DeliveryEnv(ParallelEnv):
    """运力商圈配送多智能体强化学习环境 - 基于PettingZoo（接口与 WFactoryEnv 对齐）"""

    metadata = {"render_modes": ["human"], "name": "delivery_dispatch_v1"}

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config if config else {}
        self.sim = DeliverySim(self.config)
        self.agents = self.sim.agents
        self.possible_agents = self.sim.agents

        _num_candidates = int(self.sim._obs_cfg.get("num_candidate_orders", 10))
        _expected = 1 + _num_candidates
        _configured = DELIVERY_ACTION_CONFIG.get("action_space_size", _expected)
        if _configured != _expected:
            raise ValueError(
                f"动作空间大小配置不一致: 配置为{_configured}, 应为{_expected} (1 + num_candidate_orders)")

        self._setup_spaces()

        global_state_dim = 1 + 2 + len(self.sim.rider_names) * 3 + 1
        self.global_state_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                 shape=(global_state_dim,), dtype=np.float32)

        self.max_steps = int(self.config.get("MAX_SIM_STEPS", 800))
        self.step_count = 0
        self.render_mode = None

        self._terminal_bonus_given = False
        self._terminal_score_baseline_ema = float(
            DELIVERY_REWARD_CONFIG.get('terminal_score_bonus_baseline_value', 0.0))

        self.obs_meta = {
            'agent_feature_names': ['rider_id_one_hot', 'carry_load_norm', 'busy_ratio', 'is_offline'],
            'global_feature_names': ['time_progress', 'pool_ratio', 'capacity_saturation', 'pool_len_norm'],
            'queue_summary_feature_names': [
                'to_pickup_time', 'weight', 'total_route_time',
                'pickup_zone_congestion', 'priority', 'is_urgent'],
            'queue_summary_stat_names': ['min', 'max', 'mean', 'std', 'median'],
            'candidate_feature_names': [
                'exists', 'upstream_willingness', 'total_route_time', 'to_pickup_time',
                'pickup_zone_congestion', 'priority', 'is_urgent', 'order_type',
                'time_pressure', 'slack'],
            'normalization_constants': {
                'max_op_duration_norm': DELIVERY_OBS_CONFIG.get('max_op_duration_norm', 60.0),
                'total_remaining_time_norm': DELIVERY_OBS_CONFIG.get('total_remaining_time_norm', 120.0),
                'slack_time_norm': DELIVERY_OBS_CONFIG.get('slack_time_norm', 120.0),
            },
            'num_stations': len(self.sim.rider_names),
            'multi_discrete_num_heads': getattr(self, '_multi_discrete_num_heads', None),
            'multi_discrete_action_dim': getattr(self, '_multi_discrete_action_dim', None),
            'multi_discrete_heads_equal_dim': True,
            'action_names': DELIVERY_ACTION_CONFIG.get('action_names'),
            'candidate_action_start': self.sim._candidate_action_start,
            'candidate_action_end': self.sim._candidate_action_end,
            'scenario': 'delivery',
            # 三层联调：候选来源与上游口径（供 app/联调核对）
            'candidate_source': self.sim._candidate_source,
            'upstream_order_by': self.sim._upstream_order_by,
            'upstream_top_k': int(self.sim._upstream_top_k),
            'upstream_riders_covered': sorted(self.sim._upstream_order.keys()),
        }

    def observation_space(self, agent: str = None):
        return self._observation_spaces[agent]

    def action_space(self, agent: str = None):
        return self._action_spaces[agent]

    def _get_obs_shape(self) -> Tuple[int,]:
        temp_sim = DeliverySim(self.config)
        temp_sim.reset()
        return temp_sim.get_state_for_agent(temp_sim.agents[0]).shape

    def _setup_spaces(self):
        obs_shape = self._get_obs_shape()
        self._observation_spaces = {
            agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
            for agent in self.agents
        }
        action_size = 1 + int(self.sim._obs_cfg.get("num_candidate_orders", 10))
        max_heads = max(1, max(int(v.get("count", 1)) for v in self.sim._riders_cfg.values()))
        self._multi_discrete_num_heads = max_heads
        self._multi_discrete_action_dim = action_size
        self._action_spaces = {
            agent: gym.spaces.MultiDiscrete([action_size] * max_heads)
            for agent in self.agents
        }

    # ------------------------------------------------------------------
    def _build_infos(self) -> Dict[str, Dict[str, Any]]:
        infos = {agent: {} for agent in self.agents}
        global_state = self.sim.get_global_state()
        for agent_id in self.agents:
            rider_name = agent_id.replace("agent_", "")
            infos[agent_id]['global_state'] = global_state
            infos[agent_id]['obs_meta'] = self.obs_meta

            cands = self.sim._get_candidate_orders(rider_name)
            rider_obj = self.sim.riders[rider_name]
            _infos_cands = []
            for c in cands:
                _o = c["order"]
                _slack = self.sim._order_slack(rider_obj, _o, self.sim.current_time)
                _pos, _free = rider_obj.projected_free()
                _start = max(self.sim.current_time, _free if _free > 0 else self.sim.current_time)
                _est = self.sim._estimate_route(rider_obj, _o, _pos, _start)
                _infos_cands.append({
                    'action': self.sim._candidate_action_start + c["index"],
                    'queue_index': c["index"],
                    'part_id': _o.order_id,
                    'order_type': _o.order_type,
                    'due_date': float(_o.due_date),
                    'ready_time': float(_o.ready_time),          # 供 FIFO 基线
                    'to_pickup_time': float(_est["leg1_time"]),  # 供最近骑手/SPT 基线
                    'priority': int(_o.priority),
                    'willingness': self.sim._willingness_for(rider_name, _o),
                    'slack': _slack,
                })
            infos[agent_id]['candidates_map'] = _infos_cands
            rider = self.sim.riders[rider_name]
            infos[agent_id]['queue_snapshot'] = [
                {'queue_index': i, 'part_id': o.order_id,
                 'slack': float(o.due_date - (o.planned_deliver_time or self.sim.current_time)),
                 'proc_time': float(max(0.0, (o.planned_deliver_time or self.sim.current_time) - self.sim.current_time))}
                for i, o in enumerate(rider.carry)
            ]
            infos[agent_id]['action_mask'] = self.sim._get_action_mask(rider_name)
        return infos

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.sim.reset()
        self.step_count = 0
        self.agents = self.possible_agents[:]
        self._terminal_bonus_given = False

        observations = {agent: self.sim.get_state_for_agent(agent) for agent in self.agents}
        infos = self._build_infos()
        self.infos = infos
        return observations, infos

    def step(self, actions: Dict[str, Any]):
        self.step_count += 1

        rewards = self.sim.step_with_actions(actions)
        observations = {agent: self.sim.get_state_for_agent(agent) for agent in self.agents}
        terminations = {agent: self.sim.is_done() for agent in self.agents}
        truncations = {agent: self.step_count >= self.max_steps for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        episode_ended = bool(any(terminations.values()) or any(truncations.values()))
        extra_infos = self._build_infos()
        if episode_ended:
            final_stats = self.sim.get_final_stats()
            episode_score = float(calculate_delivery_episode_score(final_stats, config=self.config))
            for agent_id in self.agents:
                infos[agent_id]["final_stats"] = final_stats
                infos[agent_id]["episode_score"] = episode_score

            # 终局分数 bonus（复用工厂 env 的逻辑与键名）
            if (not self._terminal_bonus_given) and bool(
                    DELIVERY_REWARD_CONFIG.get('terminal_score_bonus_enabled', False)):
                baseline_mode = str(DELIVERY_REWARD_CONFIG.get('terminal_score_bonus_baseline_mode', 'ema'))
                if baseline_mode == 'none':
                    baseline = 0.0
                elif baseline_mode == 'fixed':
                    baseline = float(DELIVERY_REWARD_CONFIG.get('terminal_score_bonus_baseline_value', 0.0))
                else:
                    baseline = float(self._terminal_score_baseline_ema)

                delta = float(episode_score) - float(baseline)
                if bool(DELIVERY_REWARD_CONFIG.get('terminal_score_bonus_positive_only', False)):
                    if delta < 0.0:
                        delta = 0.0
                clip_abs = float(DELIVERY_REWARD_CONFIG.get('terminal_score_bonus_clip_delta_abs', 0.0))
                if clip_abs > 0.0:
                    delta = float(np.clip(delta, -clip_abs, clip_abs))

                scale = float(DELIVERY_REWARD_CONFIG.get('terminal_score_bonus_scale', 0.0))
                bonus_total = float(scale) * float(delta)
                if bonus_total != 0.0:
                    per_agent = float(bonus_total) / float(max(1, len(rewards)))
                    for agent_id in rewards:
                        rewards[agent_id] += per_agent
                    for agent_id in infos:
                        infos[agent_id]['terminal_score_bonus'] = float(per_agent)
                        infos[agent_id]['episode_score_baseline'] = float(baseline)
                        infos[agent_id]['episode_score_delta'] = float(delta)

                if baseline_mode == 'ema':
                    alpha = float(np.clip(
                        float(DELIVERY_REWARD_CONFIG.get('terminal_score_bonus_ema_alpha', 0.05)), 0.0, 1.0))
                    self._terminal_score_baseline_ema = (
                        (1.0 - alpha) * float(self._terminal_score_baseline_ema) + alpha * float(episode_score))
                self._terminal_bonus_given = True

        for agent_id in self.agents:
            base = infos[agent_id]
            base.update(extra_infos[agent_id])

        self.infos = infos
        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    def render(self, mode="human"):
        self.render_mode = mode
        if mode == "human":
            print(f"仿真时间: {self.sim.current_time:.1f}")
            print(f"已送达: {len(self.sim.delivered)}/{len(self.sim.orders)} 待派池: {len(self.sim.pool)}")
            for name, r in self.sim.riders.items():
                print(f"{name}: 携带={len(r.carry)}/{r.capacity}, 位置=({r.position[0]:.1f},{r.position[1]:.1f})")

    def close(self):
        pass


# =============================================================================
# 4. 环境工厂函数
# =============================================================================
def make_parallel_delivery_env(config: Dict[str, Any] = None):
    """创建配送调度 PettingZoo 环境（仅主进程打印日志）"""
    import os
    try:
        import multiprocessing as _mp
        is_main_process = (_mp.current_process().name == 'MainProcess')
    except Exception:
        is_main_process = True

    if is_main_process and not SILENT_MODE:
        print("创建运力商圈配送调度环境 (scenario=delivery)")

    env_config = dict(config or {})
    env_config['scenario'] = 'delivery'
    return DeliveryEnv(env_config)
