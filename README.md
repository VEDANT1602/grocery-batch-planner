# Grocery Batch Planner

Can a batch planner get more grocery orders out on time than a greedy assignment? I gave both policies the same six shoppers, the same orders and the same time buffer. On the first held-out day, greedy handled twelve of the 24 orders on time. CP-SAT handled thirteen. The assignment log shows how: shopper `s0-1` took the three-order batch that greedy had given to `s0-0`, freeing `s0-0` to pick two other orders. [Day 65 assignments](results/wave-results.json)

## From shopping time to a dispatch plan

I fit XGBoost on the first 50 days of fictional orders to predict how long an individual basket would take. I used the next 15 days to estimate how much those predictions could run short. Three times the 90th percentile of positive residuals gave me about five minutes of slack per trip. On the held-out orders the model missed individual shopping times by 1.103 minutes on average in absolute terms. Using each store’s training average missed by 4.284 minutes. [Data card](docs/data-card.md) · [Model and policy metrics](results/metrics.json)

Then I let each shopper take up to three orders from one store, subject to basket capacity and predicted deadlines. Greedy picked a feasible plan. CP-SAT got 200 ms to look for one that served more orders, with shorter predicted shopper time breaking ties. A separate verifier recomputed the constraints before accepting the plan. When I set the search budget to zero, all 25 waves fell back to the greedy assignment. [Planner](grocery_planner/planner.py) · [Feasibility tests](tests/test_planner.py)

## Where the nine extra orders came from

Across 600 orders in 25 held-out waves, buffered CP-SAT got **284** out on time, buffered greedy **275**, and buffered FIFO **239**. The optimizer found one extra on-time order in nine waves and tied greedy in the other sixteen. Both policies used the same buffer, so that comparison tells me what the search bought in this simulation. The paired day-bootstrap interval was 0.16 to 0.56 additional on-time orders per wave across 2,000 resamples. [All 125 policy-wave outcomes](results/wave-results.json)

Turning the buffer off answered a different question. CP-SAT then assigned 315 orders. Of those, 302 were on time and 13 were late. With the buffer on, all 284 assignments met their deadlines, and 316 orders waited. Whether that’s preferable depends on what happens to the orders left behind.

All 25 buffered searches reached `OPTIMAL`. On the reference Windows CPU, the full planning call took 17.9 ms at the median and 29.4 ms at p95 across those waves. The clock started after predictions were available. It covered candidate generation, search and verification. [Environment and timing summary](results/metrics.json) · [Case study](docs/case-study.md)

## Notes to self

What I’d do differently is keep the clock running for a second dispatch. The current simulation stops after the first assignment of the day. It treats an order that waits five more minutes the same as one that misses its promise entirely. I’d record the original deadline and eventual completion time for every deferred order before choosing a buffer for a store.

## Run it

Use Python 3.11 in a virtual environment.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m grocery_planner.experiment --output runs/local
```

The experiment generates 5,400 seeded fictional orders (3,000 training, 900 calibration and 1,500 held out), fits XGBoost, loads held-out predictions, solves the dispatch waves and writes the assignments. I kept training in a separate process with a 120-second timeout after the reference Windows solver stalled when both native runtimes were loaded together. The reference versions were Python 3.11.9, XGBoost 3.1.2 and OR-Tools 9.10.4067. To repeat just the model stage, run `python -m grocery_planner.train`.

## Data and implementation

[Training orders](data/orders.csv) · [Dispatch inputs](data/dispatch-waves.json) · [Trained model](models/shopping-duration.json) · [Operations contract](docs/operations.md) · [Tests](tests/)

Each shopper starts at one store and takes one batch. Travel is an additive approximation, and the shopping-time generator uses the same equation throughout. The buffer is an engineering allowance calibrated on this fixture. A store with different traffic or shopping patterns would need its own evaluation.

## References

[Project provenance and data sources](docs/provenance.md). Technical references include [OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver), [solver limits](https://developers.google.com/optimization/cp/cp_tasks), [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html), and ECC’s [verification workflow](https://github.com/affaan-m/ECC/blob/b38be6a7f6350964d59328b52f0624e1d1d552d1/skills/verification-loop/SKILL.md). The [Instacart role](https://www.instacart.careers/job?gh_jid=8240305) helped select lifecycle and decision-evaluation themes; this project has no affiliation with Instacart. No ECC code, retailer dataset or resume text was copied.
