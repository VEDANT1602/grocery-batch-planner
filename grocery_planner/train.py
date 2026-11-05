"""Model stage: intentionally isolated from the native optimization runtime."""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import xgboost as xgb

from .data import FEATURES, generate, load, matrix

ROOT = Path(__file__).resolve().parents[1]


def run():
    data_path = ROOT / "data/orders.csv"
    generate(data_path)
    rows = load(data_path)
    train = [r for r in rows if r["day"] < 50]
    calibration = [r for r in rows if 50 <= r["day"] < 65]
    test = [r for r in rows if r["day"] >= 65]
    model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=.07,
                            subsample=1., colsample_bytree=1., tree_method="hist",
                            objective="reg:squarederror", random_state=42, n_jobs=1)
    model.fit(matrix(train), np.array([r["duration_minutes"] for r in train]))
    model_path = ROOT / "models/shopping-duration.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    store_mean = {s: float(np.mean([r["duration_minutes"] for r in train if r["store_code"] == s]))
                  for s in range(3)}
    pred_test = np.maximum(.1, model.predict(matrix(test)))
    pred_cal = model.predict(matrix(calibration))
    residual90 = max(0., float(np.quantile(np.array([r["duration_minutes"] for r in calibration]) - pred_cal, .9)))
    summary = {"xgboost_test_mae_minutes": float(np.mean(np.abs(pred_test - [r["duration_minutes"] for r in test]))),
               "store_mean_test_mae_minutes": float(np.mean([abs(store_mean[int(r["store_code"])] - r["duration_minutes"]) for r in test])),
               "positive_residual_q90_minutes": residual90, "trip_buffer_minutes": round(3 * residual90, 6),
               "features": FEATURES, "xgboost_version": xgb.__version__, "python_version": platform.python_version(),
               "sha256": {"data/orders.csv": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                          "models/shopping-duration.json": hashlib.sha256(model_path.read_bytes()).hexdigest()}}
    predictions = {r["order_id"]: float(p) for r, p in zip(test, pred_test)}
    for filename, value in (("training-metrics.json", summary), ("heldout-predictions.json", predictions)):
        (model_path.parent / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
