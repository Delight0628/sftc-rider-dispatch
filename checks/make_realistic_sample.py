"""
生成贴近真实宽表列名的本地样本（无库权限时联调用）。
用法：
  python checks/make_realistic_sample.py --out data/samples
产出：
  sample_orders.csv / sample_riders.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


def make_orders(n: int, seed: int = 7):
    rng = random.Random(seed)
    # 深圳/华南附近
    base_lng, base_lat = 114.05, 22.55
    t0 = 1_789_600_000_000  # ms
    rows = []
    for i in range(n):
        ang = rng.uniform(0, 2 * math.pi)
        r1 = rng.uniform(0.01, 0.04)
        r2 = rng.uniform(0.01, 0.05)
        shop_lng = base_lng + r1 * math.cos(ang)
        shop_lat = base_lat + r1 * math.sin(ang)
        user_lng = base_lng + r2 * math.cos(ang + rng.uniform(-0.5, 0.5))
        user_lat = base_lat + r2 * math.sin(ang + rng.uniform(-0.5, 0.5))
        push = t0 + int(rng.uniform(0, 3 * 3600 * 1000))
        promise = int(rng.uniform(25, 55) * 60 * 1000)
        arrive_shop = push + int(rng.uniform(3, 12) * 60 * 1000)
        confirm = push + int(rng.uniform(0.5, 4) * 60 * 1000)
        pickup = arrive_shop + int(rng.uniform(2, 8) * 60 * 1000)
        dist_km = max(1, int(rng.uniform(1, 8)))
        finish = pickup + int((dist_km * 2.5 + rng.uniform(5, 15)) * 60 * 1000)
        due = push + promise
        late = max(0, finish - due)
        rows.append({
            "order_id": 10_000_000 + i,
            "sf_bill_id": f"SF{i:08d}",
            "shop_lng": round(shop_lng, 6),
            "shop_lat": round(shop_lat, 6),
            "user_lng": round(user_lng, 6),
            "user_lat": round(user_lat, 6),
            "push_time": push,
            "order_time": push - 30_000,
            "create_time": push - 60_000,
            "loc_assessment_time": due,
            "latest_delivery_time": due + 120_000,
            "expect_time": due,
            "distribute_time": push + 10_000,
            "confirm_time": confirm,
            "arriveshop_time": arrive_shop,
            "pickup_time": pickup,
            "finish_time": finish,
            "weight_gram": round(rng.uniform(200, 3500), 1),
            "volume_litre": round(rng.uniform(0.2, 3.0), 2),
            "da_order_status": 1,
            "rider_ucode_finished": f"UC{rng.randint(1000, 1004)}",
            "rider_id": rng.randint(20000, 20004),
            "service_mode": 2,
            "dispatch_type": 1,
            "delivery_type": rng.choice([1, 1, 1, 2]),
            "product_level": rng.choice([1, 2, 2, 3]),
            "city_id": 755,
            "business_station_id": rng.randint(100, 104),
            "distance_km": dist_km,
            "total_distance": dist_km * 1000 + rng.randint(0, 400),
            "confirm_order_duration_minutes": round((confirm - push) / 60000.0, 2),
            "arrive_shop_duration_minutes": round((arrive_shop - confirm) / 60000.0, 2),
            "distribution_duration_minutes": round((finish - pickup) / 60000.0, 2),
            "estimated_delivery_time": rng.randint(30, 50),
            "new_level": rng.randint(1, 5),
            "rider_label": rng.choice([0, 1, 2, 4]),
            "rider_net": rng.choice([1, 2, 3, 4]),
            "weather_level": "0",
            "is_timeliness": 1 if late == 0 else 0,
            "is_finished": 1,
            "ol_fin_no_late": 1 if late == 0 else 0,
            "ol_fin_late_o5m": 1 if late > 5 * 60 * 1000 else 0,
            "ol_fin_late_o15m": 1 if late > 15 * 60 * 1000 else 0,
            "ol_fin_late_o30m": 1 if late > 30 * 60 * 1000 else 0,
            "is_overtime_accept": 0,
            "is_overtime_payment": 1 if late > 15 * 60 * 1000 else 0,
            "trade_amount": round(rng.uniform(4, 18), 2),
            "basic_fee": round(rng.uniform(3, 10), 2),
        })
    return rows


def make_riders(n: int = 5, seed: int = 7):
    rng = random.Random(seed + 1)
    rows = []
    for i in range(n):
        rows.append({
            "rider_id": 20000 + i,
            "ucode": f"UC{1000 + i}",
            "main_id": 30000 + i,
            "cityid": 755,
            "city": "深圳市",
            "station_id": 100 + i,
            "level": rng.randint(1, 5),
            "work_type": rng.choice([1, 2]),
            "hire_type": "全职" if i % 2 == 0 else "众包",
            "work_status": 1,
            "account_status": 1,
            "vehicle_type": rng.choice([1, 1, 2, 3]),
            "extra_flag": rng.choice([0, 1, 32, 64, 128, 4096]),
            "rider_flag": 0,
            "team_id": rng.randint(1, 4),
            "transport_type": rng.choice([0, 1]),
            "rider_credit": rng.randint(50, 100),
            "first_finish_time": 1_700_000_000_000,
            "register_time": 1_690_000_000_000,
        })
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/samples")
    ap.add_argument("--orders", type=int, default=80)
    ap.add_argument("--riders", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    op = out / "sample_orders.csv"
    rp = out / "sample_riders.csv"
    write_csv(op, make_orders(args.orders, args.seed))
    write_csv(rp, make_riders(args.riders, args.seed))
    print(f"wrote {op} ({args.orders} rows)")
    print(f"wrote {rp} ({args.riders} rows)")


if __name__ == "__main__":
    main()
