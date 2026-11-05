# Grocery Batch Planner

**A good duration model still needs a feasible assignment and a way to fail safely.**

This independent portfolio experiment connects an actual XGBoost shopping-duration model to an OR-Tools CP-SAT batch planner. It compares decisions against FIFO and a competitive greedy policy, then rechecks every assignment outside the solver. Everything runs on a CPU with fictional orders; there are no paid services or retailer records.

## The decision I wanted to make

At one planning instant, 24 orders compete for six shoppers across three stores. Which same-store batches should leave now, given item capacity and promised deadlines? An accurate prediction alone cannot answer that question. Picking the next order can consume the shopper who would have covered several other orders, and a solver result can still be unusable if the application trusts stale inputs or cached durations.

My approach is **predict → reserve → assign → verify**:

1. Fit duration on days 0–49; calibrate an engineering time buffer on days 50–64.
2. Enumerate only batches of up to three same-store orders that fit capacity and predicted deadlines.
3. Search for the maximum number of assigned orders, using total predicted shopper time to break ties. Give the search 200 ms and seed it with a feasible greedy solution.
4. Independently recompute store, capacity, uniqueness, batch size, duration and deadline constraints before returning a plan. If the search yields no usable incumbent, return the checked fallback.

The reserve is a deliberate policy choice. I also run the same optimizer without it so the reader can see the throughput–lateness tradeoff.

## What the actual run shows

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

The unbuffered policy serves more orders on time overall, but misses 13 promises. The buffered policy misses none in this fixture and defers more orders. I would not call either policy a universal winner without a business loss function for lateness, deferral and shopper effort.

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

Original implementation and fictional data by Vedant Thaker with AI implementation assistance; see [provenance](docs/provenance.md). Conceptual references: [OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver), [solver limits](https://developers.google.com/optimization/cp/cp_tasks), [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html), and ECC’s [verification workflow](https://github.com/affaan-m/ECC/blob/b38be6a7f6350964d59328b52f0624e1d1d552d1/skills/verification-loop/SKILL.md). The [Instacart role](https://www.instacart.careers/job?gh_jid=8240305) helped select lifecycle and decision-evaluation themes; this project has no affiliation with Instacart. No ECC code, retailer dataset or resume text was copied.
