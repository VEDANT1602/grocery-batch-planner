# Grocery Batch Planner

On day 65, six shoppers have 24 orders to cover across three stores. Buffered greedy dispatches 12 orders on time. CP-SAT fits 13 into the same wave, with every assigned order meeting its deadline. Compare the `greedy` and `cp_sat` entries for day 65 to see which orders share a shopper. [Assignments and realized outcomes](results/wave-results.json)

## How the planner picks a batch

The duration model trains on days 0–49. Residuals from days 50–64 set a buffer on predicted time. A shopper can take up to three orders from the same store if the items fit and the batch can meet its predicted deadlines.

Greedy supplies a feasible starting assignment. CP-SAT has 200 ms to search for an assignment that covers more orders, using predicted shopper minutes to break ties. Before dispatch, a separate check recomputes capacity, order uniqueness, batch size, duration and deadlines. If the search returns no usable assignment, the planner can still use the checked greedy plan. [Planner](grocery_planner/planner.py) · [Feasibility tests](tests/test_planner.py)

## Across the 25 held-out waves

Reference run: Python 3.11.9, Windows CPU, XGBoost 3.1.2, OR-Tools 9.10.4067. The dataset contains **5,400** fictional orders: 3,000 training, 900 calibration and 1,500 held out. Assignment evaluation uses **25 held-out daily waves / 600 orders**.

The fitted model achieves **1.103 min MAE**, versus **4.284 min** for a training-only store-mean control. This measures individual shopping-duration prediction, not customer delivery-time accuracy.

| Policy | Assigned | On time | Late | Left for a later wave | Shopper min / on-time order |
|---|---:|---:|---:|---:|---:|
| FIFO, buffered | 239 | 239 | 0 | 361 | 20.336 |
| Greedy, buffered | 275 | 275 | 0 | 325 | 18.360 |
| CP-SAT, buffered | 284 | 284 | 0 | 316 | 18.144 |
| CP-SAT, no buffer | 315 | 302 | 13 | 285 | 18.554 |
| Forced zero-search fallback | 275 | 275 | 0 | 325 | 18.360 |

Buffered optimization serves **9 additional on-time orders** over buffered greedy in these waves. The paired day-bootstrap interval for the difference is **0.16–0.56 on-time orders per wave** (2,000 resamples). This interval describes this simulator fixture; it is not a field treatment effect.

A second dispatch wave would make the cost of deferral visible. This experiment ends after the first assignment in each day, so a pending order has no observed eventual delivery time. Dropping the buffer got more orders out on time overall, at the cost of 13 missed promises.

All 25 buffered searches returned `OPTIMAL`; none needed fallback during the normal run. A separate zero-budget run exercises fallback on all 25 waves. Observed full planning-call latency was **17.9 ms median / 29.4 ms p95** on this machine. These are 25 small local calls, including candidate creation and verification, after predictions are available. They exclude model inference, loading, networking and training and are not a service SLO. The 200 ms limit applies to solver search, not to the full request.

## Run it

Use Python 3.11 in a virtual environment:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m grocery_planner.experiment --output runs/local
```

The experiment generates the seeded dataset, trains the real model in a separate process, loads held-out predictions, solves the waves and writes metrics plus every assignment. Training has a 120-second subprocess timeout. A separate process keeps the model and optimization native runtimes apart: importing both in one process stalled the reference Windows solver. This separation is an execution boundary, not a model-serving claim.

To repeat only the model stage:

```bash
python -m grocery_planner.train
```

## Inspect the evidence

| Artifact | Purpose |
|---|---|
| [Case study](docs/case-study.md) | Problem, design choices, interpretation and next experiment |
| [Data card](docs/data-card.md) | Generator, temporal split and what the simulation omits |
| [Operations contract](docs/operations.md) | Input limits, failure behavior and deployment gates |
| [Raw metrics](results/metrics.json) | Full-precision results, environment and SHA-256 digests |
| [Wave results](results/wave-results.json) | All 125 policy-wave outcomes and assignments |
| [Dispatch inputs](data/dispatch-waves.json) | Predecision orders, shoppers and separately named realized labels |
| [Training orders](data/orders.csv) | The 5,400-row generated dataset |
| [Trained model](models/shopping-duration.json) | Loadable XGBoost model |
| [Planner](grocery_planner/planner.py) | Enumeration, bounded optimization, fallback and independent verifier |
| [Tests](tests/) | Exhaustive small-case optimality, invalid inputs, fallback, replay and artifact checks |

## Boundaries that matter

This is one dispatch wave, not a marketplace simulator. Each shopper starts at one store and can take one batch. Travel distance is an additive approximation, not a road-network route. Unassigned orders remain pending; the experiment does not simulate their eventual completion. The data-generating shopping equation is simple and stationary, so its predictive accuracy will be optimistic compared with unfamiliar stores or disrupted operations. The buffer is three times the calibration 90th-percentile positive residual; it is not a joint conformal coverage guarantee. No real deployment, live load test, A/B trial, retailer endorsement or historical employer result is asserted.

## References and authorship

[Project provenance and data sources](docs/provenance.md). Technical references include [OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver), [solver limits](https://developers.google.com/optimization/cp/cp_tasks), [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html), and ECC’s [verification workflow](https://github.com/affaan-m/ECC/blob/b38be6a7f6350964d59328b52f0624e1d1d552d1/skills/verification-loop/SKILL.md). The [Instacart role](https://www.instacart.careers/job?gh_jid=8240305) helped select lifecycle and decision-evaluation themes; this project has no affiliation with Instacart. No ECC code, retailer dataset or resume text was copied.
