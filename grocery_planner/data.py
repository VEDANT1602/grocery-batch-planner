"""Seeded fictional orders: no customer, retailer or employer data."""
import csv
from pathlib import Path
import numpy as np

FEATURES = ["items", "produce_fraction", "frozen_fraction", "distinct_aisles", "congestion", "store_code"]


def generate(path, seed=1602):
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["order_id", "day"] + FEATURES + ["distance_km", "duration_minutes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for day in range(90):
            for index in range(60):
                items = int(rng.integers(4, 29))
                produce, frozen = rng.uniform(0, .7), rng.uniform(0, .3)
                aisles = int(rng.integers(2, min(items, 12) + 1))
                congestion = float(rng.uniform(.1, .95))
                store = int(rng.integers(0, 3))
                duration = (3 + .42 * items + .58 * aisles + 4.2 * produce
                            + 1.9 * frozen + 6 * congestion ** 2 + .8 * store
                            + .16 * items * congestion + rng.normal(0, 1.3))
                row = {"order_id": f"d{day:02d}-{index:02d}", "day": day,
                       "items": items, "produce_fraction": round(float(produce), 6),
                       "frozen_fraction": round(float(frozen), 6), "distinct_aisles": aisles,
                       "congestion": round(congestion, 6), "store_code": store,
                       "distance_km": round(float(rng.uniform(.5, 5)), 6),
                       "duration_minutes": round(float(max(2, duration)), 6)}
                writer.writerow(row)
    return path


def load(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [{k: (v if k == "order_id" else float(v)) for k, v in row.items()}
                for row in csv.DictReader(handle)]


def matrix(rows):
    return np.asarray([[r[f] for f in FEATURES] for r in rows], dtype=np.float32)
