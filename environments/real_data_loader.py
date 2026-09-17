"""
真实数仓样本 → 配送环境配置
===========================
把 dwd_fact_order_whole / dim_rider_single 的导出样本（CSV/JSON 行）
转成 DeliveryEnv 可消费的 custom_orders 与骑手配置。

坐标：经纬度 → 以样本包围盒中心为原点的公里平面（近似，足够仿真）。
时间：epoch ms → 相对回合起点的分钟。
权限：本模块不查库；请用业务方导出的小样本（一天/一商圈）。
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .real_data_schema import ORDER_CORE_MAP, RIDER_CORE_MAP, DISTANCE_UNIT
from .delivery_config import ORDER_TYPE_LIST

EARTH_KM_PER_DEG = 111.0  # 粗略


def _first(row: Dict[str, Any], candidates: Sequence[str]) -> Any:
    for c in candidates:
        if c in row and row[c] not in (None, "", "NULL", "null"):
            return row[c]
    return None


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    try:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    f = _to_float(v, None)
    if f is None:
        return default
    try:
        return int(f)
    except (TypeError, ValueError):
        return default


def _epoch_min(v: Any) -> Optional[float]:
    """epoch 秒/毫秒 → 分钟绝对值；已是分钟量级则原样返回。"""
    f = _to_float(v, None)
    if f is None:
        return None
    # ms (13位) / s (10位)
    if f > 1e12:
        return f / 1000.0 / 60.0
    if f > 1e9:
        return f / 60.0
    return f


def load_rows(path: str | Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """读 CSV（自动嗅探分隔符）或 JSONL/JSON 数组。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
        rows = data if isinstance(data, list) else data.get("rows", [])
    else:
        sample = text[:4096]
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        rows = list(csv.DictReader(text.splitlines(), dialect=dialect))
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    return rows


class GeoProjector:
    """经纬度 → 本地公里平面；仿真坐标原点=样本均值中心。"""

    def __init__(self, center_lng: float, center_lat: float, grid_size: float = 20.0):
        self.center_lng = float(center_lng)
        self.center_lat = float(center_lat)
        self.grid_size = float(grid_size)
        self._kx = EARTH_KM_PER_DEG * math.cos(math.radians(self.center_lat))
        self._ky = EARTH_KM_PER_DEG

    @classmethod
    def from_points(cls, points: Iterable[Tuple[float, float]], grid_size: float = 20.0):
        pts = [(float(a), float(b)) for a, b in points if a is not None and b is not None]
        if not pts:
            return cls(0.0, 0.0, grid_size)
        lng = sum(p[0] for p in pts) / len(pts)
        lat = sum(p[1] for p in pts) / len(pts)
        return cls(lng, lat, grid_size)

    def project(self, lng: float, lat: float) -> Tuple[float, float]:
        x = (float(lng) - self.center_lng) * self._kx + self.grid_size / 2.0
        y = (float(lat) - self.center_lat) * self._ky + self.grid_size / 2.0
        # 轻微夹紧，避免极端点把地图撑爆
        x = max(-5.0, min(self.grid_size + 5.0, x))
        y = max(-5.0, min(self.grid_size + 5.0, y))
        return (round(x, 3), round(y, 3))


def _map_order_type(row: Dict[str, Any]) -> str:
    """业务类型 → 环境品类（影响 due 缓冲与优先级）。"""
    service = _to_int(_first(row, ORDER_CORE_MAP["service_mode"]), None)
    delivery = _to_int(_first(row, ORDER_CORE_MAP["delivery_type"]), None)
    level = _to_int(_first(row, ORDER_CORE_MAP["product_level"]), None)
    weight = (_to_float(_first(row, ORDER_CORE_MAP["weight_gram"]), 1000.0) or 1000.0) / 1000.0
    if weight >= 4.0:
        return "团餐大单"
    if delivery == 2:  # 限时达
        return "文件急送"
    if service == 2 and (level or 2) <= 2:
        return "生鲜冷链" if weight >= 2.0 else "普通餐品"
    return ORDER_TYPE_LIST[min(3, max(0, (level or 2) - 1))]


def _map_priority(row: Dict[str, Any], order_type: str) -> int:
    delivery = _to_int(_first(row, ORDER_CORE_MAP["delivery_type"]), None)
    if delivery == 2:
        return 1
    if order_type in ("文件急送", "生鲜冷链"):
        return 1
    if order_type == "团餐大单":
        return 3
    return 2


def extract_order_core(row: Dict[str, Any]) -> Dict[str, Any]:
    """一行宽表 → 中间语义 dict（绝对分钟时刻 + 经纬度 + kg）。"""
    out: Dict[str, Any] = {"raw": row}
    for key, cands in ORDER_CORE_MAP.items():
        out[key] = _first(row, cands)
    out["ready_min"] = _epoch_min(out.get("ready_epoch_ms"))
    out["due_min"] = _epoch_min(out.get("due_epoch_ms"))
    out["finish_min"] = _epoch_min(out.get("finish_epoch_ms"))
    out["pickup_lng"] = _to_float(out.get("pickup_lng"))
    out["pickup_lat"] = _to_float(out.get("pickup_lat"))
    out["dropoff_lng"] = _to_float(out.get("dropoff_lng"))
    out["dropoff_lat"] = _to_float(out.get("dropoff_lat"))
    grams = _to_float(out.get("weight_gram"), None)
    out["weight_kg"] = (grams / 1000.0) if grams is not None else 1.0
    dist_km = _to_float(out.get("distance_km"), None)
    total_m = _to_float(out.get("total_distance_m"), None)
    if dist_km is None and total_m is not None:
        dist_km = total_m / 1000.0
    out["distance_km"] = dist_km
    return out


def filter_training_orders(cores: List[Dict[str, Any]],
                           completed_only: bool = True) -> List[Dict[str, Any]]:
    """默认只要已完成单（有真实 due/finish 可校准分布）。"""
    kept = []
    for c in cores:
        status = _to_int(c.get("da_order_status"), None)
        if completed_only and status not in (None, 1):
            # status 缺失时保留（可能是已过滤样本）；明确非完成则丢
            if status is not None and status != 1:
                continue
        if c.get("pickup_lng") is None or c.get("dropoff_lng") is None:
            continue
        kept.append(c)
    return kept


def build_custom_orders(cores: List[Dict[str, Any]],
                        projector: GeoProjector,
                        max_orders: int = 80,
                        time_origin: Optional[float] = None) -> List[Dict[str, Any]]:
    """中间语义 → env.custom_orders（相对分钟）。"""
    valid = [c for c in cores if c.get("ready_min") is not None]
    if not valid:
        return []
    if time_origin is None:
        time_origin = min(c["ready_min"] for c in valid)
    # 按就绪时间排序后截断，保证是一个连续时间窗
    valid.sort(key=lambda c: c["ready_min"])
    valid = valid[:max_orders]

    orders: List[Dict[str, Any]] = []
    for i, c in enumerate(valid):
        ready = max(0.0, float(c["ready_min"]) - float(time_origin))
        due_abs = c.get("due_min")
        if due_abs is None:
            # 回退：就绪 + 品类缓冲 + 估算行程
            pickup = projector.project(c["pickup_lng"], c["pickup_lat"])
            dropoff = projector.project(c["dropoff_lng"], c["dropoff_lat"])
            route = math.hypot(dropoff[0] - pickup[0], dropoff[1] - pickup[1]) / 0.55 + 5.0
            due = ready + 40.0 + route
        else:
            due = max(ready + 10.0, float(due_abs) - float(time_origin))

        order_type = _map_order_type(c.get("raw") or {})
        oid = c.get("order_id")
        try:
            oid = int(oid) if oid is not None else i
        except (TypeError, ValueError):
            oid = i
        orders.append({
            "order_id": oid,
            "order_type": order_type,
            "pickup": projector.project(c["pickup_lng"], c["pickup_lat"]),
            "dropoff": projector.project(c["dropoff_lng"], c["dropoff_lat"]),
            "ready_time": round(ready, 2),
            "due_date": round(due, 2),
            "priority": _map_priority(c.get("raw") or {}, order_type),
            "weight": round(float(c.get("weight_kg") or 1.0), 2),
            # 保留真实链路标注，便于离线 KPI / 双塔样本
            "_real": {
                "rider_ucode": c.get("rider_ucode"),
                "rider_id": c.get("rider_id"),
                "distance_km": c.get("distance_km"),
                "is_timeliness": c.get("is_timeliness"),
                "confirm_duration_min": _to_float(c.get("confirm_duration_min")),
                "arrive_shop_duration_min": _to_float(c.get("arrive_shop_duration_min")),
                "distribution_duration_min": _to_float(c.get("distribution_duration_min")),
                "trade_amount": _to_float(c.get("trade_amount")),
            },
        })
    return orders


def build_rider_configs(rider_rows: List[Dict[str, Any]],
                        projector: GeoProjector,
                        max_riders: int = 5,
                        default_capacity: int = 3) -> Dict[str, Dict[str, Any]]:
    """骑士维表 → RIDERS 风格配置（home 用城市均值投影点附近随机散布）。"""
    riders: Dict[str, Dict[str, Any]] = {}
    for i, row in enumerate(rider_rows[:max_riders]):
        rid = _first(row, RIDER_CORE_MAP["rider_id"]) or _first(row, RIDER_CORE_MAP["ucode"]) or i
        name = f"骑手{chr(ord('A') + i)}"
        # 维表无实时坐标；用投影中心附近的确定性偏置代表常驻区域
        # （真实轨迹接入前的占位；后续可用历史 finish_lng/lat 均值替换）
        ang = 2 * math.pi * (i + 0.5) / max(1, max_riders)
        radius = 3.0 + (i % 3)
        home = (
            round(projector.grid_size / 2 + radius * math.cos(ang), 2),
            round(projector.grid_size / 2 + radius * math.sin(ang), 2),
        )
        vehicle = _to_int(_first(row, RIDER_CORE_MAP["vehicle_type"]), 1)
        # 摩托/汽车略快
        speed = 0.60 if vehicle in (3, 5, 6) else 0.55
        riders[name] = {
            "count": 1,
            "capacity": default_capacity,
            "speed": speed,
            "home": home,
            "_real": {
                "rider_id": _first(row, RIDER_CORE_MAP["rider_id"]),
                "ucode": _first(row, RIDER_CORE_MAP["ucode"]),
                "level": _first(row, RIDER_CORE_MAP["level"]),
                "work_type": _first(row, RIDER_CORE_MAP["work_type"]),
                "vehicle_type": vehicle,
                "extra_flag": _first(row, RIDER_CORE_MAP["extra_flag"]),
                "team_id": _first(row, RIDER_CORE_MAP["team_id"]),
                "cityid": _first(row, RIDER_CORE_MAP["cityid"]),
            },
        }
    return riders


def load_real_episode_config(order_path: str | Path,
                             rider_path: Optional[str | Path] = None,
                             max_orders: int = 60,
                             max_riders: int = 5,
                             completed_only: bool = True) -> Dict[str, Any]:
    """一站式：样本文件 → DeliveryEnv(config=...)。

    返回键：
      scenario / custom_orders / riders / geo_grid_size / data_source
    """
    order_rows = load_rows(order_path, max_rows=max(max_orders * 3, 200))
    cores = [extract_order_core(r) for r in order_rows]
    cores = filter_training_orders(cores, completed_only=completed_only)

    points = []
    for c in cores:
        if c.get("pickup_lng") is not None:
            points.append((c["pickup_lng"], c["pickup_lat"]))
        if c.get("dropoff_lng") is not None:
            points.append((c["dropoff_lng"], c["dropoff_lat"]))
    projector = GeoProjector.from_points(points, grid_size=20.0)
    orders = build_custom_orders(cores, projector, max_orders=max_orders)

    riders = None
    if rider_path:
        rider_rows = load_rows(rider_path, max_rows=max_riders * 2)
        riders = build_rider_configs(rider_rows, projector, max_riders=max_riders)

    cfg: Dict[str, Any] = {
        "scenario": "delivery",
        "training_mode": True,
        "custom_orders": orders,
        "data_source": {
            "order_path": str(order_path),
            "rider_path": str(rider_path) if rider_path else None,
            "n_orders": len(orders),
            "n_riders": len(riders) if riders else None,
            "geo_center": (projector.center_lng, projector.center_lat),
            "geo_grid_size": projector.grid_size,
        },
    }
    if riders:
        cfg["riders"] = riders
        # geo 配置覆盖地图尺寸（与投影一致）
        cfg["geo_config"] = {"grid_size": projector.grid_size}
    return cfg


def offline_kpis_from_real(orders: List[Dict[str, Any]]) -> Dict[str, float]:
    """用样本自带真实标注算离线基线 KPI（不跑仿真）。"""
    finished = [o for o in orders if (o.get("_real") or {}).get("is_timeliness") is not None
                or (o.get("_real") or {}).get("rider_ucode")]
    n = len(orders) or 1
    on_time = 0
    trade = []
    dist = []
    confirm = []
    for o in orders:
        real = o.get("_real") or {}
        it = real.get("is_timeliness")
        if it in (1, "1", 1.0, True):
            on_time += 1
        if real.get("trade_amount") is not None:
            trade.append(float(real["trade_amount"]))
        if real.get("distance_km") is not None:
            dist.append(float(real["distance_km"]))
        if real.get("confirm_duration_min") is not None:
            confirm.append(float(real["confirm_duration_min"]))
    return {
        "n_orders": float(len(orders)),
        "labeled_finished": float(len(finished)),
        "offline_on_time_rate": on_time / n,
        "avg_trade_amount": (sum(trade) / len(trade)) if trade else 0.0,
        "avg_distance_km": (sum(dist) / len(dist)) if dist else 0.0,
        "avg_confirm_min": (sum(confirm) / len(confirm)) if confirm else 0.0,
    }
