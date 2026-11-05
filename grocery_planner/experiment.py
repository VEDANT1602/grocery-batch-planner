"""Train chronologically, calibrate a buffer, compare held-out dispatch waves."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import ortools

from .data import FEATURES, generate, load, matrix
from .planner import Order, Shopper, candidates, fifo_plan, greedy_plan, plan_wave, trip_seconds, validate_plan


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(plan, orders, shoppers, actual):
    omap, smap = {o.order_id: o for o in orders}, {s.shopper_id: s for s in shoppers}
    ontime, late, minutes = 0, 0, 0.
    for b in plan:
        observed = [replace(omap[x], predicted_minutes=actual[x]) for x in b.order_ids]
        seconds = trip_seconds(observed, smap[b.shopper_id], 0)
        minutes += seconds / 60
        for o in observed:
            if seconds <= o.deadline_minutes * 60:
                ontime += 1
            else:
                late += 1
    return {"assigned": ontime + late, "on_time": ontime, "late": late,
            "unassigned": len(orders) - ontime - late, "realized_shopper_minutes": minutes}


def run(output=None):
    result_dir = Path(output) if output else ROOT / "results"
    data_path = ROOT / "data/orders.csv"
    # On the reference Windows environment, importing both native libraries
    # into one process stalled the solver. Keep the contract explicit and the
    # model stage bounded in its own process on every platform.
    subprocess.run([sys.executable, "-m", "grocery_planner.train"], cwd=ROOT,
                   check=True, timeout=120, capture_output=True, text=True)
    rows = load(data_path)
    train = [r for r in rows if r["day"] < 50]
    calibration = [r for r in rows if 50 <= r["day"] < 65]
    test = [r for r in rows if r["day"] >= 65]
    model_path = ROOT / "models/shopping-duration.json"
    training = json.loads((ROOT / "models/training-metrics.json").read_text())
    # Reserve against aggregate positive error for at most three orders. This is
    # an engineering buffer, not a conformal joint-coverage guarantee.
    residual90 = training["positive_residual_q90_minutes"]
    buffer = training["trip_buffer_minutes"]
    predictions = json.loads((ROOT / "models/heldout-predictions.json").read_text())
    rng = np.random.default_rng(703)
    records, input_waves = [], []
    for day in range(65, 90):
        daily = [r for r in test if r["day"] == day]
        # 24 orders from one day's held-out labels; only predecision fields and
        # independent generated promise windows are sent to the planner.
        chosen = rng.choice(len(daily), size=24, replace=False)
        wave_rows = [daily[int(i)] for i in chosen]
        orders = [Order(r["order_id"], f"store-{int(r['store_code'])}", int(r["items"]),
                        predictions[r["order_id"]], float(rng.uniform(25, 68)),
                        r["distance_km"], i) for i, r in enumerate(wave_rows)]
        shoppers = [Shopper(f"s{s}-{i}", f"store-{s}", cap, pace)
                    for s in range(3) for i, (cap, pace) in enumerate(((43, 1.0), (57, 1.12)))]
        actual = {r["order_id"]: r["duration_minutes"] for r in wave_rows}
        input_waves.append({"day": day,
                            "orders": [o.__dict__ for o in orders],
                            "shoppers": [s.__dict__ for s in shoppers],
                            "realized_duration_minutes": actual})
        pool = candidates(orders, shoppers, buffer)
        optimized, meta = plan_wave(orders, shoppers, buffer, .2)
        unbuffered, unbuffered_meta = plan_wave(orders, shoppers, 0, .2)
        forced_fallback, fallback_meta = plan_wave(orders, shoppers, buffer, 0)
        policies = {"fifo": fifo_plan(orders, shoppers, buffer), "greedy": greedy_plan(pool),
                    "cp_sat": optimized, "cp_sat_unbuffered": unbuffered,
                    "zero_budget_fallback": forced_fallback}
        for name, plan in policies.items():
            validate_plan(plan, orders, shoppers, 0 if name == "cp_sat_unbuffered" else buffer)
            records.append({"day": day, "policy": name, "orders": len(orders),
                            "score": score(plan, orders, shoppers, actual),
                            "plan": [{"shopper_id": b.shopper_id, "order_ids": list(b.order_ids),
                                      "predicted_seconds": b.predicted_seconds} for b in plan],
                            "runtime": meta if name == "cp_sat" else unbuffered_meta if name == "cp_sat_unbuffered" else fallback_meta if name == "zero_budget_fallback" else None})
    totals = {}
    for name in ("fifo", "greedy", "cp_sat", "cp_sat_unbuffered", "zero_budget_fallback"):
        group = [r for r in records if r["policy"] == name]
        keys = group[0]["score"].keys()
        totals[name] = {k: sum(r["score"][k] for r in group) for k in keys}
        totals[name]["on_time_fraction_all_orders"] = totals[name]["on_time"] / 600
        totals[name]["shopper_minutes_per_on_time_order"] = (totals[name]["realized_shopper_minutes"] / totals[name]["on_time"]
                                                            if totals[name]["on_time"] else None)
    # Paired resampling uses the dispatch day as unit; this is simulator
    # uncertainty across this small fixture, never a field causal confidence interval.
    diffs = np.array([next(r["score"]["on_time"] for r in records if r["day"] == d and r["policy"] == "cp_sat")
                      - next(r["score"]["on_time"] for r in records if r["day"] == d and r["policy"] == "greedy") for d in range(65, 90)])
    bootstrap = np.random.default_rng(42).choice(diffs, (2000, len(diffs)), replace=True).mean(axis=1)
    timings = [r["runtime"]["total_seconds"] * 1000 for r in records if r["policy"] == "cp_sat"]
    wave_path = ROOT / "data/dispatch-waves.json"
    write_json(wave_path, input_waves)
    metadata = {"data": {"rows": len(rows), "train_days": [0,49], "calibration_days": [50,64],
                         "test_days": [65,89], "train_rows": len(train), "calibration_rows": len(calibration),
                         "test_rows": len(test), "dispatch_waves": 25, "dispatch_orders": 600},
                "model": {"xgboost_test_mae_minutes": training["xgboost_test_mae_minutes"],
                          "store_mean_test_mae_minutes": training["store_mean_test_mae_minutes"],
                          "positive_residual_q90_minutes": residual90, "trip_buffer_minutes": buffer,
                          "features": FEATURES},
                "policies": totals,
                "cp_sat": {"status_counts": dict(Counter(r["runtime"]["solver_status"] for r in records if r["policy"] == "cp_sat")),
                           "fallback_waves": sum(r["runtime"]["fallback_used"] for r in records if r["policy"] == "cp_sat"),
                           "total_latency_median_ms": float(np.median(timings)), "total_latency_p95_ms": float(np.quantile(timings,.95)),
                           "search_budget_seconds": .2,
                           "on_time_orders_per_wave_minus_greedy": float(diffs.mean()),
                           "paired_day_bootstrap_95_interval": np.quantile(bootstrap,[.025,.975]).tolist()},
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                "xgboost": training["xgboost_version"], "ortools": ortools.__version__, "numpy": np.__version__,
                                "cpu_only": True},
                "sha256": {"data/orders.csv": sha(data_path), "models/shopping-duration.json": sha(model_path),
                           "data/dispatch-waves.json": sha(wave_path),
                           "models/heldout-predictions.json": sha(ROOT / "models/heldout-predictions.json"),
                           **{f"grocery_planner/{name}.py": sha(ROOT / "grocery_planner" / f"{name}.py")
                              for name in ("data", "train", "planner", "experiment")}}}
    write_json(result_dir / "wave-results.json", records)
    write_json(result_dir / "metrics.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
