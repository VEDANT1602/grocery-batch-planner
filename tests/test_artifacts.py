import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import numpy as np

from grocery_planner.data import generate, load, matrix
from grocery_planner.planner import Batch, Order, Shopper, validate_plan

ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def test_committed_artifact_hashes_match(self):
        metrics = json.loads((ROOT / "results/metrics.json").read_text())
        for name, expected in metrics["sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)

    def test_fictional_dataset_generation_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(generate(Path(tmp) / "orders.csv").read_bytes(), (ROOT / "data/orders.csv").read_bytes())

    def test_model_loads_and_reproduces_heldout_mae(self):
        code = """from grocery_planner.data import load,matrix
from xgboost import XGBRegressor
import numpy as np
rows=[r for r in load('data/orders.csv') if r['day']>=65]
model=XGBRegressor()
model.load_model('models/shopping-duration.json')
prediction=np.maximum(.1,model.predict(matrix(rows)))
print(float(np.mean(abs(prediction-[r['duration_minutes'] for r in rows]))))
"""
        process = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True,
                                 capture_output=True, text=True, timeout=60)
        metrics = json.loads((ROOT / "results/metrics.json").read_text())
        self.assertAlmostEqual(float(process.stdout.strip()), metrics["model"]["xgboost_test_mae_minutes"], places=6)

    def test_report_arithmetic_and_fallback_equivalence(self):
        records = json.loads((ROOT / "results/wave-results.json").read_text())
        metrics = json.loads((ROOT / "results/metrics.json").read_text())
        self.assertEqual(len(records), 125)
        for policy, totals in metrics["policies"].items():
            selected = [r for r in records if r["policy"] == policy]
            self.assertEqual(sum(r["score"]["on_time"] for r in selected), totals["on_time"])
            self.assertEqual(totals["assigned"], totals["on_time"] + totals["late"])
            self.assertEqual(600, totals["assigned"] + totals["unassigned"])
        for day in range(65, 90):
            greedy = next(r for r in records if r["day"] == day and r["policy"] == "greedy")
            fallback = next(r for r in records if r["day"] == day and r["policy"] == "zero_budget_fallback")
            self.assertEqual(greedy["score"], fallback["score"])
            self.assertEqual(greedy["plan"], fallback["plan"])

    def test_all_published_assignments_pass_independent_replay_validation(self):
        inputs = {r["day"]: r for r in json.loads((ROOT / "data/dispatch-waves.json").read_text())}
        records = json.loads((ROOT / "results/wave-results.json").read_text())
        buffer = json.loads((ROOT / "results/metrics.json").read_text())["model"]["trip_buffer_minutes"]
        for r in records:
            wave = inputs[r["day"]]
            orders = [Order(**o) for o in wave["orders"]]
            shoppers = [Shopper(**s) for s in wave["shoppers"]]
            batches = [Batch(b["shopper_id"], tuple(b["order_ids"]), b["predicted_seconds"]) for b in r["plan"]]
            self.assertTrue(validate_plan(batches, orders, shoppers, 0 if r["policy"] == "cp_sat_unbuffered" else buffer))

    def test_published_outcomes_replay_from_realized_labels(self):
        # Independent replay uses the simulator's documented equation, not the
        # stored predicted trip seconds or experiment.score implementation.
        inputs = {r["day"]: r for r in json.loads((ROOT / "data/dispatch-waves.json").read_text())}
        records = json.loads((ROOT / "results/wave-results.json").read_text())
        for record in records:
            wave = inputs[record["day"]]
            orders = {o["order_id"]: o for o in wave["orders"]}
            shoppers = {s["shopper_id"]: s for s in wave["shoppers"]}
            ontime, late, realized_minutes = 0, 0, 0.
            for batch in record["plan"]:
                ids = batch["order_ids"]
                share = .9 if len(ids) == 1 else .68
                minutes = (3 + share * sum(wave["realized_duration_minutes"][i] for i in ids)
                           * shoppers[batch["shopper_id"]]["pace"]
                           + 1.4 * sum(orders[i]["distance_km"] for i in ids) + 1.2 * len(ids))
                seconds = math.ceil(minutes * 60)
                realized_minutes += seconds / 60
                for order_id in ids:
                    if seconds <= orders[order_id]["deadline_minutes"] * 60:
                        ontime += 1
                    else:
                        late += 1
            with self.subTest(day=record["day"], policy=record["policy"]):
                self.assertEqual(record["score"]["on_time"], ontime)
                self.assertEqual(record["score"]["late"], late)
                self.assertAlmostEqual(record["score"]["realized_shopper_minutes"], realized_minutes, places=8)


if __name__ == "__main__":
    unittest.main()
