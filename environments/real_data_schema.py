"""
真实数仓字段映射与调度 KPI 筛选
================================
来源：钉钉群「顶尖算法团队」2026-09-17
- dts.dwd_fact_order_whole  订单明细宽表（833 字段，月级上亿行，分区 edt/emn）
- dw.dim_rider_single       骑士维表（71 字段，每 ucode 在职 rider）

业务方结论：两表可覆盖 80%+ 训练需求，需做特征工程；调度效果指标由算法侧筛选。

本模块只做「字段语义层」：把数仓列名映射到 env / 双塔 / W&D / KPI，
不直接连库；真实样本通过 real_data_loader 转成 custom_orders。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# =============================================================================
# 1. 订单表 → 环境订单（custom_orders）核心字段
# =============================================================================
# 优先级从左到右：首选列缺失时回退次选。
ORDER_CORE_MAP: Dict[str, Tuple[str, ...]] = {
    # 标识
    "order_id": ("order_id", "sf_bill_id", "out_order_id"),
    # 取餐点（店铺）
    "pickup_lng": ("shop_lng", "arriveshop_lng", "pickup_lng"),
    "pickup_lat": ("shop_lat", "arriveshop_lat", "pickup_lat"),
    # 送餐点（用户）
    "dropoff_lng": ("user_lng", "finish_lng"),
    "dropoff_lat": ("user_lat", "finish_lat"),
    # 就绪/推单时刻（订单进入可派池）
    "ready_epoch_ms": ("push_time", "order_push_time", "list_time", "order_time",
                       "create_time", "start_time", "user_order_time"),
    # 承诺/考核送达时刻（due）
    "due_epoch_ms": ("loc_assessment_time", "latest_delivery_time", "expect_time",
                     "shop_expect_time", "external_expect_time", "finish_time"),
    # 实际链路时刻（回放/标注用，仿真不消费）
    "distribute_epoch_ms": ("distribute_time", "first_dispatch_time", "order_disp_time"),
    "confirm_epoch_ms": ("confirm_time", "order_received_time", "virtual_accept_time"),
    "arriveshop_epoch_ms": ("arriveshop_time",),
    "pickup_epoch_ms": ("pickup_time", "fetch_time", "first_pickup_time"),
    "finish_epoch_ms": ("finish_time", "end_time"),
    # 重量体积
    "weight_gram": ("weight_gram",),
    "volume_litre": ("volume_litre",),
    # 状态 / 完单骑士
    "da_order_status": ("da_order_status", "order_status"),
    "rider_ucode": ("rider_ucode_finished", "rider_ucode_accept", "rider_ucode", "ucode"),
    "rider_id": ("rider_id_finished", "rider_id"),
    # 业务标签
    "service_mode": ("service_mode", "for_type"),          # 1驻店 / 2商圈
    "dispatch_type": ("dispatch_type",),                   # 1商圈 2驻店中餐 3驻店快餐
    "delivery_type": ("delivery_type",),                   # 0预约 1立即 2限时
    "product_level": ("product_level",),
    "business_type": ("business_type",),
    "city_id": ("city_id", "super_city_id"),
    "business_station_id": ("business_station_id", "source_station_id"),
    "resource_station_id": ("resource_station_id", "distribution_station_id"),
    # 距离（米或 km，见单位表）
    "distance_km": ("distance_km",),
    "total_distance_m": ("total_distance", "delivery_distance_meter",
                         "delivery_distance_meter_finished"),
    "pickup_distance_m": ("pickup_distance",),
    "arriveshop_distance_m": ("arriveshop_distance",),
    # 时长（分钟，优先 *_minutes）
    "confirm_duration_min": ("confirm_order_duration_minutes",
                             "confirm_order_duration_minutes_no_take"),
    "arrive_shop_duration_min": ("arrive_shop_duration_minutes", "arrive_shop_duration"),
    "wait_duration_min": ("wait_duration_minutes", "wait_duration"),
    "distribution_duration_min": ("distribution_duration_minutes", "distribution_duration"),
    "estimated_delivery_time_min": ("estimated_delivery_time", "estimated_time_for_strategy"),
    # 骑士/环境标签
    "new_level": ("new_level",),
    "rider_label": ("rider_label",),
    "rider_net": ("rider_net",),
    "rider_vehicle_type": ("rider_vehicle_type",),
    "weather_level": ("weather_level", "original_weather_level", "external_weather_level"),
    # 业务结果（KPI/特征）
    "is_timeliness": ("is_timeliness", "is_timeliness_new", "is_timeliness_ele"),
    "is_finished": ("is_finished", "finished_status"),
    "ol_fin_no_late": ("ol_fin_no_late",),
    "ol_fin_late_o5m": ("ol_fin_late_o5m",),
    "ol_fin_late_o15m": ("ol_fin_late_o15m",),
    "ol_fin_late_o30m": ("ol_fin_late_o30m",),
    "is_overtime_accept": ("is_overtime_accept",),
    "is_overtime_payment": ("is_overtime_payment",),
    "trade_amount": ("trade_amount", "trade_amount_finished"),
    "basic_fee": ("basic_fee",),
}

# 单位说明：distance_km 为 km（向下取整）；total_distance 等多为米
DISTANCE_UNIT = {
    "distance_km": "km",
    "total_distance_m": "m",
    "pickup_distance_m": "m",
    "arriveshop_distance_m": "m",
}

# =============================================================================
# 2. 骑士维表 → 环境骑手 / 双塔 User Tower
# =============================================================================
RIDER_CORE_MAP: Dict[str, Tuple[str, ...]] = {
    "rider_id": ("rider_id",),
    "ucode": ("ucode",),
    "main_id": ("main_id",),
    "cityid": ("cityid", "super_city_id"),
    "city": ("city", "super_city"),
    "station_id": ("station_id", "lc_id"),
    "level": ("level",),
    "work_type": ("work_type",),                 # 工作性质/骑士类型
    "hire_type": ("hire_type",),
    "work_status": ("work_status",),
    "account_status": ("account_status",),
    "vehicle_type": ("vehicle_type",),
    "extra_flag": ("extra_flag",),               # 距离偏好/先锋/冲锋
    "rider_flag": ("rider_flag",),
    "team_id": ("team_id",),
    "transport_type": ("transport_type",),       # 1组织送 0非组织送
    "rider_credit": ("rider_credit",),
    "first_finish_time": ("first_finish_time",),
    "register_time": ("register_time", "first_register_time"),
}

# extra_flag 位语义（维表注释）
EXTRA_FLAG_BITS = {
    "normal": 0,
    "pioneer": 1,          # 先锋
    "sprint": 4096,        # 冲锋
    "org_short": 32,       # 组织化-短距离
    "org_mid": 64,
    "org_long": 128,
}

# =============================================================================
# 3. 特征工程清单（双塔 / Wide&Deep / MAPPO 观测）
# =============================================================================
# 订单塔 / W&D item 侧（从 833 字段收敛，避免过拟合与泄漏）
ORDER_FEATURE_SPEC: List[Dict[str, str]] = [
    {"name": "order_id", "role": "id", "source": "dwd_fact_order_whole.order_id"},
    {"name": "pickup_xy", "role": "geo", "source": "shop_lng/shop_lat"},
    {"name": "dropoff_xy", "role": "geo", "source": "user_lng/user_lat"},
    {"name": "ready_offset_min", "role": "time", "source": "push_time-order_t0"},
    {"name": "due_offset_min", "role": "time", "source": "loc_assessment_time-order_t0"},
    {"name": "promise_window_min", "role": "time", "source": "due-ready"},
    {"name": "weight_kg", "role": "dense", "source": "weight_gram/1000"},
    {"name": "volume_l", "role": "dense", "source": "volume_litre"},
    {"name": "distance_km", "role": "dense", "source": "distance_km|total_distance/1000"},
    {"name": "service_mode", "role": "cat", "source": "service_mode"},
    {"name": "dispatch_type", "role": "cat", "source": "dispatch_type"},
    {"name": "delivery_type", "role": "cat", "source": "delivery_type"},
    {"name": "product_level", "role": "cat", "source": "product_level"},
    {"name": "city_id", "role": "cat", "source": "city_id"},
    {"name": "business_station_id", "role": "cat", "source": "business_station_id"},
    {"name": "weather_level", "role": "cat", "source": "weather_level"},
    {"name": "order_hour", "role": "cat", "source": "hour(order_time)"},
    {"name": "est_delivery_min", "role": "dense", "source": "estimated_delivery_time"},
]

# 骑手塔 / User Tower（从 71 字段收敛）
RIDER_FEATURE_SPEC: List[Dict[str, str]] = [
    {"name": "rider_id", "role": "id", "source": "dim_rider_single.rider_id"},
    {"name": "ucode", "role": "id", "source": "dim_rider_single.ucode"},
    {"name": "level", "role": "cat", "source": "level"},
    {"name": "work_type", "role": "cat", "source": "work_type"},
    {"name": "vehicle_type", "role": "cat", "source": "vehicle_type"},
    {"name": "extra_flag_bits", "role": "multi_hot", "source": "extra_flag"},
    {"name": "team_id", "role": "cat", "source": "team_id"},
    {"name": "transport_type", "role": "cat", "source": "transport_type"},
    {"name": "city_id", "role": "cat", "source": "cityid"},
    {"name": "station_id", "role": "cat", "source": "station_id"},
    {"name": "tenure_days", "role": "dense", "source": "first_finish_time-register_time"},
]

# 正负样本构造（双塔训练）
# 正：da_order_status=1 且 rider_ucode 非空（完单）
# 负：曝光未接 —— 需 expose_time 存在且无 confirm；当前宽表无完整曝光日志时，
#     用「同商圈同小时、未派给该骑手的完单」做代理负样本（训练侧再精化）
DUAL_TOWER_SAMPLE_RULES = {
    "positive": "da_order_status==1 and rider_ucode not null",
    "negative_proxy": "same city/station & hour, order not finished by this rider",
    "hard_negative": "expose_time not null and confirm_time is null (若有曝光表)",
}

# =============================================================================
# 4. 调度效果 KPI 筛选（从 833 字段收敛到路演/训练可解释指标）
# =============================================================================
# 分层：主指标（路演必讲） / 过程诊断 / 效率 / 业务价值代理
KPI_SPEC: List[Dict[str, str]] = [
    # ---- 主指标 ----
    {
        "key": "completion_rate",
        "tier": "primary",
        "label": "完成率",
        "formula": "delivered / total_orders",
        "table_fields": "da_order_status / is_finished",
        "why": "运力能否把单做完，路演第一眼",
    },
    {
        "key": "on_time_rate",
        "tier": "primary",
        "label": "准时率",
        "formula": "on_time_count / delivered（仿真）或 AVG(is_timeliness)（离线）",
        "table_fields": "is_timeliness / is_timeliness_new / ol_fin_no_late",
        "why": "会议口径：时间关系是 MAPPO 核心责任",
    },
    {
        "key": "total_tardiness",
        "tier": "primary",
        "label": "总超时分钟",
        "formula": "sum(max(0, finish-due))",
        "table_fields": "finish_time - loc_assessment_time；分桶 ol_fin_late_*",
        "why": "比准时率更连续，可作训练/对比曲线",
    },
    # ---- 过程诊断 ----
    {
        "key": "late_bucket_rates",
        "tier": "diagnostic",
        "label": "超时分桶率",
        "formula": "AVG(ol_fin_late_o5m/o15m/o30m)",
        "table_fields": "ol_fin_late_o5m / o15m / o30m / o50m",
        "why": "区分「轻微超时」与「严重超时」，指导奖励塑形",
    },
    {
        "key": "accept_duration",
        "tier": "diagnostic",
        "label": "接起时长(分)",
        "formula": "AVG(confirm_order_duration_minutes)",
        "table_fields": "confirm_order_duration_minutes",
        "why": "派单是否被骑手及时接受",
    },
    {
        "key": "arrive_shop_duration",
        "tier": "diagnostic",
        "label": "到店时长(分)",
        "formula": "AVG(arrive_shop_duration_minutes)",
        "table_fields": "arrive_shop_duration_minutes",
        "why": "第一段路径质量（对应 env leg1）",
    },
    {
        "key": "distribution_duration",
        "tier": "diagnostic",
        "label": "配送时长(分)",
        "formula": "AVG(distribution_duration_minutes)",
        "table_fields": "distribution_duration_minutes",
        "why": "第二段路径质量（对应 env leg2）",
    },
    {
        "key": "overtime_accept_rate",
        "tier": "diagnostic",
        "label": "接单超时率",
        "formula": "AVG(is_overtime_accept)",
        "table_fields": "is_overtime_accept",
        "why": "调度窗口是否过紧",
    },
    # ---- 效率 ----
    {
        "key": "makespan",
        "tier": "efficiency",
        "label": "总完工时长",
        "formula": "max(finish) - t0",
        "table_fields": "finish_time / create_time",
        "why": "批次清空速度",
    },
    {
        "key": "mean_utilization",
        "tier": "efficiency",
        "label": "骑手利用率",
        "formula": "mean(busy_time / horizon)",
        "table_fields": "仿真内生；离线可用 distribution_duration 聚合代理",
        "why": "运力是否吃满",
    },
    {
        "key": "distance_per_order",
        "tier": "efficiency",
        "label": "单均里程(km)",
        "formula": "mean(distance_km | total_distance/1000)",
        "table_fields": "distance_km / total_distance",
        "why": "路径规划是否绕路",
    },
    # ---- 业务价值代理 ----
    {
        "key": "commission_per_order",
        "tier": "business",
        "label": "单均骑士计提",
        "formula": "AVG(trade_amount)",
        "table_fields": "trade_amount / trade_amount_finished",
        "why": "调度结果与骑士收益的对齐度（Demo 可讲）",
    },
    {
        "key": "overtime_payment_rate",
        "tier": "business",
        "label": "超时赔付率",
        "formula": "AVG(is_overtime_payment)",
        "table_fields": "is_overtime_payment",
        "why": "超时的直接成本",
    },
]

# 路演主表只保留 primary + efficiency 中的关键项
ROADSHOW_KPI_KEYS = [
    "completion_rate", "on_time_rate", "total_tardiness",
    "makespan", "mean_utilization", "distance_per_order",
]

# 训练/日志建议同时打的诊断项
TRAINING_DIAG_KPI_KEYS = [
    "late_bucket_rates", "accept_duration", "arrive_shop_duration",
    "distribution_duration", "overtime_accept_rate",
]


def kpi_by_tier(tier: str) -> List[Dict[str, str]]:
    return [k for k in KPI_SPEC if k.get("tier") == tier]


def summarize_schema() -> Dict[str, object]:
    return {
        "order_table": "dts.dwd_fact_order_whole",
        "order_fields_total": 833,
        "order_core_mapped": len(ORDER_CORE_MAP),
        "rider_table": "dw.dim_rider_single",
        "rider_fields_total": 71,
        "rider_core_mapped": len(RIDER_CORE_MAP),
        "kpi_count": len(KPI_SPEC),
        "roadshow_kpis": ROADSHOW_KPI_KEYS,
        "training_diag_kpis": TRAINING_DIAG_KPI_KEYS,
    }
