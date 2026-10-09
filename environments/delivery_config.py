"""
运力商圈配送调度系统 - 配送场景全局配置
==========================================
比赛项目「运力评估商圈智能体」MAPPO 调度层配置（唯一真理来源）

场景映射（工业车间 → 外卖/同城配送）：
- 工件 Part        -> 配送订单（取餐点/送餐点坐标、承诺送达时间）
- 机器/工作站       -> 骑手（位置、最大携带单量、速度）
- 工艺路线（固定站序）-> 订单路线（去取餐点→取餐→去送餐点→送达，随订单动态变化）
- 站点队列          -> 共享待派单池（所有骑手从池中接单）
- 设备故障          -> 骑手离线/休息
- 紧急插单          -> 紧急订单（紧截止时间）

时间单位：分钟；距离单位：公里。
数据策略：先用随机模拟数据验证算法逻辑（强化学习对数据真实性容忍度高），
后续通过 custom_orders / import 接入真实业务数据。

三层级联联调（比赛架构：双塔召回 → Wide&Deep 精排 → MAPPO 调度）：
- 上游（双塔+W&D）产出「骑手 × 候选订单 × 接单意愿分」→ 见第 5.5 节数据契约
  （DELIVERY_UPSTREAM_CONFIG / normalize_upstream_candidates /
   generate_mock_upstream_candidates）
- 本仓库（MAPPO）消费上游候选与意愿分，在时间紧迫性约束下决定派单与执行顺序
  （会议口径：时间紧迫性优先于意愿排序）
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# =============================================================================
# 1. 基础仿真参数
# =============================================================================
DELIVERY_SIMULATION_TIME = 480.0      # 单次仿真时长（分钟）：8 小时午高峰+晚高峰前
DELIVERY_TIME_UNIT = "minutes"
DELIVERY_TIMEOUT_MULTIPLIER = 1.5     # 超时上限倍数

# =============================================================================
# 2. 骑手配置（Riders）—— 对应原 WORKSTATIONS（5 个 agent）
# =============================================================================
# count：并发"头"数（每骑手 1，保持单头动作空间）
# capacity：最大同时携带订单数（接单后 FIFO 依次执行）
# speed：骑行速度 km/min
RIDERS = {
    "骑手A": {"count": 1, "capacity": 3, "speed": 0.55, "home": (2.0, 10.0)},
    "骑手B": {"count": 1, "capacity": 3, "speed": 0.55, "home": (10.0, 2.0)},
    "骑手C": {"count": 1, "capacity": 3, "speed": 0.50, "home": (18.0, 10.0)},
    "骑手D": {"count": 1, "capacity": 3, "speed": 0.50, "home": (10.0, 18.0)},
    "骑手E": {"count": 1, "capacity": 2, "speed": 0.60, "home": (10.0, 10.0)},
}

# =============================================================================
# 3. 地理与配送参数
# =============================================================================
DELIVERY_GEO_CONFIG = {
    "grid_size": 20.0,                # 地图边长（公里），坐标范围 [0, grid_size]
    "distance_metric": "euclidean",   # euclidean | manhattan
    "speed_norm": 0.6,                # 速度归一化基准（km/min）
    "distance_norm": 20.0,            # 距离归一化基准（公里）
    "route_time_norm": 60.0,          # 单段行程时间归一化基准（分钟）
    "total_route_time_norm": 120.0,   # 订单全流程时间归一化基准（分钟）
    "pickup_service_time": 3.0,       # 取餐耗时（分钟）
    "dropoff_service_time": 2.0,      # 送达耗时（分钟）
    # 商圈（取餐点聚集区），随机生成取餐点时 60% 从商圈采样、40% 全图均匀
    "business_zones": [
        {"center": (4.0, 4.0), "radius": 3.0},
        {"center": (16.0, 16.0), "radius": 3.0},
        {"center": (10.0, 10.0), "radius": 2.5},
    ],
    "business_zone_weight": 0.6,
    # 取餐点区域拥堵：统计半径内其他骑手/待派订单密度
    "congestion_radius_km": 2.0,
}

# 订单品类（对应原产品类型）：影响承诺时效与优先级
ORDER_TYPES = {
    "普通餐品":   {"time_buffer": 45.0, "priority": 2},
    "文件急送":   {"time_buffer": 30.0, "priority": 1},
    "生鲜冷链":   {"time_buffer": 25.0, "priority": 1},
    "团餐大单":   {"time_buffer": 60.0, "priority": 3},
}
ORDER_TYPE_LIST = tuple(ORDER_TYPES.keys())
ORDER_TYPE_ID = {name: idx for idx, name in enumerate(ORDER_TYPE_LIST)}

# =============================================================================
# 4. 基础订单模板（模拟数据；真实数据通过 custom_orders 注入）
# =============================================================================
# 字段：order_type / pickup (x,y) / dropoff (x,y) / ready_time / due_date /
#       priority / weight(kg)
DELIVERY_BASE_ORDERS = [
    {"order_type": "普通餐品", "pickup": (4.5, 3.8), "dropoff": (9.0, 6.5),  "ready_time": 0.0,  "priority": 2, "weight": 1.2},
    {"order_type": "普通餐品", "pickup": (3.5, 4.6), "dropoff": (5.5, 12.0), "ready_time": 5.0,  "priority": 2, "weight": 0.8},
    {"order_type": "文件急送", "pickup": (16.2, 15.5), "dropoff": (10.5, 8.0), "ready_time": 10.0, "priority": 1, "weight": 0.3},
    {"order_type": "生鲜冷链", "pickup": (10.8, 9.2), "dropoff": (15.0, 13.5), "ready_time": 15.0, "priority": 1, "weight": 2.5},
    {"order_type": "团餐大单", "pickup": (9.5, 10.5), "dropoff": (3.0, 16.0), "ready_time": 20.0, "priority": 3, "weight": 6.0},
    {"order_type": "普通餐品", "pickup": (17.0, 17.2), "dropoff": (13.5, 10.0), "ready_time": 25.0, "priority": 2, "weight": 1.0},
    {"order_type": "普通餐品", "pickup": (4.0, 5.0),  "dropoff": (12.0, 3.5), "ready_time": 30.0, "priority": 2, "weight": 1.5},
    {"order_type": "文件急送", "pickup": (15.5, 16.5), "dropoff": (18.5, 6.0), "ready_time": 35.0, "priority": 1, "weight": 0.2},
]

# =============================================================================
# 5. 随机订单生成（泛化训练；强化学习先用合理分布的模拟数据）
# =============================================================================
DELIVERY_RANDOM_ORDERS_CONFIG = {
    "min_orders": 12,
    "max_orders": 20,
    "priority_weights": [0.25, 0.5, 0.25],      # P1/P2/P3
    "type_weights": [0.55, 0.2, 0.15, 0.1],     # 普通/文件/生鲜/团餐
    "ready_time_range": (0.0, 120.0),           # 订单就绪（到达）时间
    "due_buffer_range": (25.0, 60.0),           # due = ready + buffer + 行程时间
    "weight_range": (0.2, 4.0),
    # orders_scale（课程学习缩放）作用于订单数量
}

# 随机骑手扰动（randomize_env=True 时启用）
DELIVERY_RIDER_RANDOMIZATION = {
    "position_jitter_km": 3.0,    # 初始位置抖动
    "speed_jitter": 0.08,         # 速度抖动（km/min）
}

# 骑手离线/休息事件（对应设备故障，阶段二启用）
RIDER_OFFLINE_CONFIG = {
    "mtbf_hours": 12,              # 平均离线间隔（小时）
    "mttr_minutes": 20,            # 平均离线时长（分钟）
    "failure_probability": 0.03,   # 每小时离线概率
}

# 紧急订单（对应紧急插单，阶段二启用）
EMERGENCY_DELIVERY_ORDERS = {
    "arrival_rate": 0.15,          # 每小时紧急订单到达率
    "priority_boost": 1,           # 优先级提升
    "due_date_reduction": 0.5,     # 承诺时效缩短比例
}

# =============================================================================
# 5.5 三层级联联调：上游（双塔召回 + Wide&Deep 精排）→ MAPPO 调度
# =============================================================================
# 会议共识（2026-09-14 听记）：
#   「已经给你取出了 10 个，他是最想能接的，给他接了这 10 个，然后你的算法就立马
#     起作用了」——上游负责"想不想接"（意愿排序），MAPPO 负责"时间关系"
#   （"你这里排出来了 100 个，只是说他想接的程度，但实际还要考虑时间关系"）。
#
# 【上游 → MAPPO 数据契约】（双塔+W&D 侧按此产出，本仓库消费）
#   upstream_candidates = {
#       "agent_骑手A": [                      # key 支持 "骑手A" 或 "agent_骑手A"
#           {"order_id": 12, "willingness": 0.93},   # 意愿分 ∈ [0,1]，建议降序
#           {"order_id": 7,  "willingness": 0.81},
#           ...                                       # 每骑手最多 top_k 条
#       ],
#       "骑手B": [...],
#   }
#   兼容写法：元素可为 dict{order_id, willingness} 或 (order_id, willingness) 元组；
#            整个 payload 也可直接传 list（视为对所有骑手生效的全局候选）。
#   注意：缺少意愿分的条目会被忽略（无法参与意愿排序），不做臆造打分。
#   约定：
#   - order_id 必须与 env 侧订单 id 一致（custom_orders/生成器的 order_id）
#   - 未覆盖的骑手 → 自动回退 endogenous 自采样
#   - 上游给出但不在待派池（已送达/未到达）的订单 → 忽略并计入 upstream_missing_count
#   - 上游候选不足 top_k，且 backfill_from_pool=True → 用池内最紧迫订单补齐
DELIVERY_UPSTREAM_CONFIG = {
    "candidate_source": "endogenous",   # endogenous=本环境自采样 | upstream=采用上游精排候选
    "top_k": 10,                        # 候选数（= 动作 1..10）
    "order_by": "urgency",              # urgency=按时间紧迫性排序（会议口径）| upstream=保持上游意愿序
    "willingness_clip": (0.0, 1.0),
    "backfill_from_pool": True,
    "willingness_jitter": 0.0,          # >0 时给"兜底意愿分"加扰动（用于鲁棒性/泛化训练）
}


# =============================================================================
# 6. 观测/动作空间配置（与工厂场景布局严格对齐，复用 MAPPO 网络）
# =============================================================================
# 观测 146 维 = 自身8 + 全局4 + 池摘要30 + 候选10×10 + 未来订单4
DELIVERY_OBS_CONFIG = {
    "num_candidate_orders": 10,        # 候选订单数（动作 1-10）
    "num_urgent_candidates": 5,        # slack 最小（最早截止优先，EDD 类比）
    "num_near_candidates": 3,          # 取餐点最近（SPT 类比）
    "num_random_candidates": 2,        # 随机多样性
    "candidate_feature_dim": 10,
    "pool_summary_features": 6,
    "pool_summary_stats": 5,           # min/max/mean/std/median
    "future_arrival_features": 4,
    "max_op_duration_norm": 60.0,      # 单段行程归一化
    "total_remaining_time_norm": 120.0,
    "slack_time_norm": 120.0,          # slack 归一化基准（分钟）
    "pool_len_norm": 20.0,             # 池长归一化
    "use_compressed_norm": True,
}

# 候选订单 10 维特征（索引必须与 ppo_network BC 教师对齐；教师只用 [0][3][8]）：
# [0]exists [1]due_rel(紧迫：相对截止) [2]total_route_time [3]to_pickup_time(≈opdur)
# [4]pickup_zone_congestion [5]priority [6]is_urgent [7]order_type_id
# [8]time_pressure [9]slack
# 【架构 2026-09-30 更正】MAPPO 与双塔**并行**，不在策略内融合骑手偏好/意愿。
# 原表无「骑手×订单」意愿字段（小新核实）；偏好由双塔侧独立算，结果在外部加权融合。
# 原 [1] willingness 槽改为 due_rel。
DELIVERY_ACTION_CONFIG = {
    "action_names": ["IDLE"] + [f"CANDIDATE_{i}" for i in range(1, 11)],
}

# =============================================================================
# 7. 奖励配置（配送语义；键名与工厂 REWARD_CONFIG 对齐，终局 bonus 逻辑复用）
# =============================================================================
DELIVERY_REWARD_CONFIG = {
    # 任务完成
    "order_completion_reward": 80.0,          # 单均送达
    "final_all_orders_completion_bonus": 500.0,
    # 时间质量
    "on_time_completion_reward": 80.0,
    "tardiness_penalty_scaler": -4.0,
    "use_huber_tardiness": True,
    "tardiness_huber_delta_norm": 0.8,
    # 过程塑形
    "progress_shaping_coeff": 0.1,
    "unnecessary_idle_penalty": -1.0,         # 有可接订单且携带未满却不接单
    "invalid_action_penalty": -0.5,           # 接不存在的候选/超携带上限
    "slack_time_penalty_coeff": -0.03,
    "slack_penalty_tanh_scale": 60.0,
    "slack_penalty_max_abs": 50.0,
    # slack 分段迟交惩罚
    "slack_based_tardiness_enabled": True,
    "slack_tardiness_step_penalty": -0.8,
    "slack_tardiness_overdue_penalty": -3.0,
    "slack_tardiness_threshold": 0.0,
    "slack_tardiness_normalize_scale": 60.0,
    "wip_penalty_coeff": -0.01,               # 池积压惩罚
    "idle_penalty_coeff": -0.005,
    # 终局分数 bonus（DeliveryEnv.step 复用同一逻辑）
    "terminal_score_bonus_enabled": True,
    "terminal_score_bonus_scale": 50.0,
    "terminal_score_bonus_clip_delta_abs": 0.2,
    "terminal_score_bonus_baseline_mode": "fixed",
    "terminal_score_bonus_baseline_value": 0.55,
    "terminal_score_bonus_ema_alpha": 0.05,
    "terminal_score_bonus_positive_only": True,
    "idle_when_work_available_penalty": -1.0,
}

# =============================================================================
# 8. 训练流程（配送场景两阶段）
# =============================================================================
DELIVERY_TRAINING_FLOW_CONFIG = {
    "foundation_phase": {
        "graduation_criteria": {
            "target_score": 0.70,
            "target_consistency": 8,
            "tardiness_threshold": 300.0,
            "min_completion_rate": 95.0,
        },
        "multi_task_mixing": {"enabled": True, "base_worker_fraction": 0.40},
        # 配送场景暂不启用课程学习（订单规模由随机生成器控制）
        "curriculum_learning": {"enabled": False, "stages": []},
    },
    "generalization_phase": {
        "completion_criteria": {
            "target_score": 0.60,
            "target_consistency": 10,
            "min_completion_rate": 85.0,
        },
        "multi_task_mixing": {"enabled": True, "base_worker_fraction": 0.40},
        "dynamic_events": {
            "rider_offline_enabled": True,       # 骑手离线
            "emergency_orders_enabled": True,    # 紧急订单
        },
    },
    "general_params": {
        "max_episodes": 1000,
        "steps_per_episode": 800,
        "eval_frequency": 1,
        "early_stop_patience": 100,
        "performance_window": 15,
    },
}


# =============================================================================
# 9. 随机订单生成器
# =============================================================================
def generate_random_delivery_orders(orders_scale: float = 1.0,
                                    config: Dict[str, Any] = None,
                                    geo_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    生成随机配送订单（模拟数据，用于泛化训练）。

    分布设计（保证合理范围内随机，听记结论：RL 不挑数据真实性）：
    - 订单数：min~max 随机，受 orders_scale 缩放
    - 取餐点：60% 商圈聚集 + 40% 全图均匀（模拟真实商圈结构）
    - 送餐点：全图均匀
    - due_date = ready_time + 品类时效缓冲 + 估算行程时间（保证可完成性）
    """
    import random

    cfg = dict(DELIVERY_RANDOM_ORDERS_CONFIG)
    if config:
        cfg.update(config)
    geo = dict(DELIVERY_GEO_CONFIG)
    if geo_config:
        geo.update(geo_config)

    grid = float(geo["grid_size"])
    zones = geo.get("business_zones", [])
    zone_w = float(geo.get("business_zone_weight", 0.5))

    def _sample_pickup(rng: "np.random.RandomState"):
        if zones and rng.random() < zone_w:
            z = zones[int(rng.randint(0, len(zones)))]
            r = float(z["radius"]) * np.sqrt(rng.random())
            theta = 2 * np.pi * rng.random()
            x = float(np.clip(z["center"][0] + r * np.cos(theta), 0.0, grid))
            y = float(np.clip(z["center"][1] + r * np.sin(theta), 0.0, grid))
        else:
            x = float(rng.random() * grid)
            y = float(rng.random() * grid)
        return (x, y)

    def _sample_dropoff(rng: "np.random.RandomState"):
        return (float(rng.random() * grid), float(rng.random() * grid))

    num_orders = int(round(np.random.randint(cfg["min_orders"], cfg["max_orders"] + 1) * float(orders_scale)))
    num_orders = max(1, num_orders)

    speed = float(geo.get("speed_norm", 0.55))
    p_w = cfg["priority_weights"]
    t_w = cfg["type_weights"]

    orders = []
    for _ in range(num_orders):
        type_name = ORDER_TYPE_LIST[int(np.random.choice(len(ORDER_TYPE_LIST), p=t_w))]
        type_cfg = ORDER_TYPES[type_name]
        priority = int(np.random.choice([1, 2, 3], p=p_w / np.sum(p_w)))
        pickup = _sample_pickup(np.random)
        dropoff = _sample_dropoff(np.random)
        ready = float(np.random.uniform(*cfg["ready_time_range"]))
        dist = float(np.hypot(dropoff[0] - pickup[0], dropoff[1] - pickup[1]))
        route_time = dist / speed + geo.get("pickup_service_time", 3.0) + geo.get("dropoff_service_time", 2.0)
        buffer_min, buffer_max = cfg["due_buffer_range"]
        due = ready + float(np.random.uniform(buffer_min, buffer_max)) + route_time
        weight = float(np.random.uniform(*cfg["weight_range"]))
        orders.append({
            "order_type": type_name,
            "pickup": (round(pickup[0], 3), round(pickup[1], 3)),
            "dropoff": (round(dropoff[0], 3), round(dropoff[1], 3)),
            "ready_time": round(ready, 2),
            "due_date": round(due, 2),
            "priority": priority,
            "weight": round(weight, 2),
        })
    orders.sort(key=lambda o: o["ready_time"])
    return orders


def get_total_orders_count(orders: List[Dict[str, Any]] = None) -> int:
    """获取总订单数（评分用）"""
    if orders is not None:
        return len(orders)
    return len(DELIVERY_BASE_ORDERS)


# =============================================================================
# 10. 上游精排（双塔 + Wide&Deep）接口工具
# =============================================================================
def heuristic_willingness(order: Dict[str, Any]) -> float:
    """兜底"接单意愿分" ∈ [0,1]（上游未覆盖该订单时使用）。

    直觉来自会议原话「其实我最喜欢送的肯定这些文件那些东西最舒服」：
    - 时效缓冲越宽松 → 越从容 → 分越高
    - 优先级越紧急（P1）→ 越不情愿接 → 分越低
    - 重量越重 → 越不情愿 → 分越低
    """
    tname = str(order.get("order_type", ORDER_TYPE_LIST[0]))
    tcfg = ORDER_TYPES.get(tname, {"time_buffer": 45.0, "priority": 2})
    buffer_norm = float(np.clip((float(tcfg.get("time_buffer", 45.0)) - 20.0) / 40.0, 0.0, 1.0))
    pr_norm = (3 - max(1, min(3, int(order.get("priority", 2))))) / 2.0
    w_norm = float(np.clip(float(order.get("weight", 1.0)) / 6.0, 0.0, 1.0))
    score = 0.5 * (1.0 - pr_norm) + 0.3 * buffer_norm + 0.2 * (1.0 - w_norm)
    return float(np.clip(score, 0.0, 1.0))


def normalize_upstream_candidates(payload: Any, rider_names: List[str] = None,
                                  top_k: int = None) -> Dict[str, List[Dict[str, Any]]]:
    """校验并归一化上游精排候选 → {骑手名: [{"order_id":.., "willingness":..}, ...]}

    见 DELIVERY_UPSTREAM_CONFIG 上方数据契约。容错点：
    - payload 为 None / 非法类型 → 返回 {}
    - 允许 dict / list 两种形态；list 视为全局候选（对所有骑手生效）
    - key 可带或不带 `agent_` 前缀；`global`/`all` 视为全局候选
    - value 元素可为 dict / (order_id, willingness) 元组 / 纯 order_id
    - 按 order_id 去重（保留最大意愿分）→ 意愿降序 → 截断 top_k
    """
    cfg = dict(DELIVERY_UPSTREAM_CONFIG)
    if top_k is None:
        top_k = int(cfg.get("top_k", 10))
    top_k = max(1, int(top_k))
    lo, hi = cfg.get("willingness_clip", (0.0, 1.0))

    def _entry(item: Any) -> Optional[Dict[str, Any]]:
        oid, wil = None, None
        if isinstance(item, dict):
            oid = item.get("order_id", item.get("orderId", item.get("id")))
            wil = item.get("willingness", item.get("score", item.get("willingness_score")))
        elif isinstance(item, (list, tuple)) and len(item) >= 1:
            oid = item[0]
            wil = item[1] if len(item) >= 2 else None
        elif isinstance(item, (int, np.integer)):
            oid = int(item)
        if oid is None:
            return None
        try:
            oid = int(oid)
        except Exception:
            return None
        if wil is None:
            return None
        try:
            wil = float(np.clip(float(wil), lo, hi))
        except Exception:
            return None
        return {"order_id": oid, "willingness": wil}

    def _clean(seq: Any) -> List[Dict[str, Any]]:
        if isinstance(seq, dict):
            seq = [seq]
        if not isinstance(seq, (list, tuple)):
            return []
        best: Dict[int, Dict[str, Any]] = {}
        for item in seq:
            ent = _entry(item)
            if ent is None:
                continue
            old = best.get(ent["order_id"])
            if old is None or ent["willingness"] > old["willingness"]:
                best[ent["order_id"]] = ent
        ordered = sorted(best.values(), key=lambda e: (-e["willingness"], e["order_id"]))
        return ordered[:top_k]

    if payload is None or isinstance(payload, str):
        return {}
    if isinstance(payload, (list, tuple)):
        payload = {"global": list(payload)}
    if not isinstance(payload, dict):
        return {}

    riders = list(rider_names) if rider_names else []
    global_entries: List[Dict[str, Any]] = []
    result: Dict[str, List[Dict[str, Any]]] = {}

    for key, seq in payload.items():
        k = str(key)
        if k.lower() in ("global", "all", "*"):
            global_entries.extend(_clean(seq))
            continue
        name = k[6:] if k.startswith("agent_") else k
        entries = _clean(seq)
        if entries:
            result[name] = entries

    if global_entries:
        merged = _clean(global_entries)
        for name in riders:
            if name in result:
                # 骑手专属候选优先，全局候选补齐
                result[name] = _clean(list(result[name]) + list(merged))
            else:
                result[name] = list(merged)

    return result


def generate_mock_upstream_candidates(orders: List[Dict[str, Any]],
                                      rider_names: List[str],
                                      top_k: int = 10,
                                      seed: int = None,
                                      noise: float = 0.18) -> Dict[str, List[Dict[str, Any]]]:
    """Mock 上游（双塔召回 + Wide&Deep 精排）输出 —— 三层联调的替身。

    在业务方真实特征口径到位前，用「订单属性意愿 + 骑手取餐点邻近度 + 噪声」
    模拟精排打分，使 MAPPO 可以立刻在 upstream 模式下训练/验证；
    真实上游接入时只需把本函数替换为真实精排结果（数据契约完全一致）。

    score = 0.55 * heuristic_willingness(order)
          + 0.30 * (1 - 归一化取餐距离)
          + 0.15 * 噪声（每个骑手独立，模拟个性化偏好）
    """
    rng = np.random.RandomState(seed)
    grid = float(DELIVERY_GEO_CONFIG.get("grid_size", 20.0))
    result: Dict[str, List[Dict[str, Any]]] = {}
    payload: Dict[str, List[Dict[str, Any]]] = {}
    for name in rider_names:
        home = RIDERS.get(name, {}).get("home", (grid / 2.0, grid / 2.0))
        scored = []
        for o in orders:
            oid = int(o.get("order_id", len(scored)))
            base = heuristic_willingness(o)
            pickup = o.get("pickup", (grid / 2.0, grid / 2.0))
            dist = float(np.hypot(pickup[0] - home[0], pickup[1] - home[1]))
            near = 1.0 - float(np.clip(dist / (grid * 0.5), 0.0, 1.0))
            score = 0.55 * base + 0.30 * near + 0.15 * float(rng.random())
            scored.append({"order_id": oid, "willingness": round(float(np.clip(score, 0.0, 1.0)), 4)})
        scored.sort(key=lambda e: (-e["willingness"], e["order_id"]))
        payload[f"agent_{name}"] = scored[:max(1, int(top_k))]
    # 复用同一套校验/归一化逻辑，保证与真实上游走同一条路径
    for name, entries in normalize_upstream_candidates(payload, rider_names, top_k).items():
        result[name] = entries
    return result


# =============================================================================
# 11. 评分函数（配送语义，输出与工厂 calculate_episode_score 同量纲 [0,1]）
# =============================================================================
def calculate_delivery_episode_score(kpi_results: Dict[str, float], config: Dict = None) -> float:
    """
    配送场景综合评分：
    - 完成率 40%：送达订单数 / 总订单数
    - 准时率 35%：1 - 归一化总迟到
    - makespan 15%：总时长越短越好
    - 骑手利用率 10%
    """
    config = config or {}
    sim_time = float(config.get('SIMULATION_TIME', DELIVERY_SIMULATION_TIME))

    # 适配 `get_final_stats` 与 `quick_kpi_evaluation` 两种 key（与工厂侧口径一致）：
    # 环境终局统计: makespan / total_parts / total_tardiness
    # 训练器聚合:   mean_makespan / mean_completed_parts / mean_tardiness
    makespan = float(kpi_results.get('makespan', kpi_results.get('mean_makespan', 0)) or 0)
    completed = int(kpi_results.get('total_parts', kpi_results.get('mean_completed_parts', 0)) or 0)
    utilization = float(kpi_results.get('mean_utilization', 0) or 0)
    tardiness = float(kpi_results.get('total_tardiness', kpi_results.get('mean_tardiness', 0)) or 0)

    if 'custom_orders' in config:
        target = len(config['custom_orders'])
    elif 'orders_scale' in config:
        base_count = (DELIVERY_RANDOM_ORDERS_CONFIG["min_orders"] + DELIVERY_RANDOM_ORDERS_CONFIG["max_orders"]) / 2.0
        target = int(round(base_count * float(config.get('orders_scale', 1.0))))
    else:
        target = get_total_orders_count()

    if target <= 0:
        return 0.0

    completion_score = min(1.0, max(0.0, completed / target))

    # 真实果洛等大区：总 tardiness 可达数千分钟，用 sim_time 做分母会被打成 0。
    # 改为「单均迟到」：avg_late = total_tardiness / max(完成或目标单量)
    # 60 分钟单均迟到 → 归零；15 分钟 → 0.75。与池大小无关。
    n_for_avg = max(int(completed), int(target), 1)
    avg_late = float(tardiness) / float(n_for_avg)
    tardiness_score = max(0.0, 1.0 - avg_late / 60.0)

    # makespan 也按「相对仿真窗口」并防止真实长序列被一票否决
    makespan_score = max(0.0, 1.0 - makespan / max(sim_time * 1.2, 1.0))
    utilization_score = min(1.0, max(0.0, utilization))

    return float(
        completion_score * 0.40 +
        tardiness_score * 0.35 +
        makespan_score * 0.15 +
        utilization_score * 0.10
    )


# =============================================================================
# 12. 混合派单框架（学习边效用 + 约束二分图匹配 + 滚动重优化）
#     对应 docs/research_dispatch_algorithm_survey.md §5 推荐架构
# =============================================================================
HYBRID_DISPATCH_CONFIG = {
    # 边特征 12 维权重（EdgeScorer.EDGE_FEATURE_NAMES 对齐）：
    # bias, order_urgency, on_time_feasible, late_norm(-), to_pickup_norm(-),
    # route_norm(-), rider_load(-), free_delay_norm(-), priority_norm,
    # is_urgent, congestion(-), slack_norm
    "edge_weights": [
        0.0,    # bias
        1.5,    # order_urgency   ：紧迫单优先（EDD 分量）
        2.0,    # on_time_feasible：预计可达准时，强正权
        -2.0,   # late_norm       ：预计迟到重罚
        -0.8,   # to_pickup_norm  ：nearest 分量
        -0.5,   # route_norm      ：全程耗时代价
        -0.6,   # rider_load      ：负载均衡
        -0.7,   # free_delay_norm ：快空闲骑手优先
        0.6,    # priority_norm
        0.8,    # is_urgent
        -0.3,   # congestion
        0.4,    # slack_norm      ：同等条件偏好安全余量
    ],
    "slack_norm": 120.0,          # 紧迫/余量归一基准（分钟），与 obs slack_time_norm 一致
    "w_tard": 2.0,                # 学习目标中迟到项权重（realized_utility）
    "w_dist": 0.5,                # 学习目标中里程项权重
    "learn": False,               # 默认关闭在线更新（评估确定性）；实验打开
    "learn_lr": 0.01,
    "max_assign_per_step": 8,     # 单决策步单骑手最大连派单数（容量 b-matching 轮数上限）
}


# Hybrid 训练配置（docs/hybrid_implementation.md §5；与代码严格一致）
HYBRID_TRAINING_CONFIG = {
    "lr": 1e-3,
    "lr_min": 5e-5,               # 保护机制 lr 下限（减半不得低于此值，防学习冻结）
    "gamma": 0.99,
    "n_step": 8,                  # n-step TD 步数（缓解履约延迟奖励）
    "temp_start": 0.5,            # 行为策略 softmax 温度（探索）
    "temp_end": 0.1,
    "temp_anneal_episodes": 200,  # 温度线性退火区间
    "buffer_size": 10000,
    "batch_size": 64,
    "recent_frac": 0.5,           # recent 优先采样比例（抑 off-policy staleness）
    "recent_window": 2000,        # recent 采样窗口（最近 N 条）
    "target_soft_tau": 0.005,     # 目标网络软更新
    "hidden_dim": 64,
    "num_heads": 4,
    "episodes_per_iter": 4,       # 每次迭代的采集 episode 数
    "updates_per_iter": 8,        # 每次迭代的 mini-batch 更新数
    "eval_every": 10,             # 每隔多少迭代做一次对拍评估
    "eval_episodes": 3,
    "eval_seed_base": 90000,      # 固定 eval seed 集基址（跨 iter 同场景对拍，消方差）
    "rollback_patience": 4,       # neural 连续不赢 linear 的轮数 → 回滚
    "rollback_grace_evals": 3,    # 新 best 后宽限 eval 轮数（期内跳过回滚保护，允许逃离 best 邻域）
    "max_grad_norm": 5.0,
}
