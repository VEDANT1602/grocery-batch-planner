# Execution and deployment contract

## Implemented boundary

The Python API is a local planning component. It accepts up to 40 orders and 12 shoppers per wave. Partitioning, authentication and queue ownership belong upstream. Orders need unique nonempty IDs, a store, a positive integer item count, a nonnegative integer acceptance index, finite predicted duration, a finite remaining promise window and finite nonnegative distance. The supported numeric domain is at most 240 predicted minutes, 480 remaining minutes, 100 km and a shopper pace no greater than 3.

A shopper can leave once, with no more than three orders from the same store and no more items than capacity. Every order can appear at most once. Predicted completion seconds round upward; remaining deadline seconds round downward. The verifier recomputes duration, so corrupted cached metadata does not pass validation.

`solver_seconds` limits native search only. Candidate creation, Python model construction, verification and library overhead are outside that limit. Cardinality limits reduce accidental enumeration blowups, but a caller that needs a hard wall-clock deadline must isolate the entire planning call in a killable worker and manage request cancellation. This implementation does not promise a hard end-to-end deadline.

## Failure behavior

| Condition | Behavior |
|---|---|
| Invalid IDs, shape, numeric values, reserve or budget | Raise `ValueError` before solver search |
| Empty wave, no shoppers, no feasible candidate | Valid empty plan |
| Zero search budget | Skip search, return independently checked greedy fallback |
| Solver has no feasible incumbent within budget | Checked greedy fallback |
| Solver incumbent has a worse objective than fallback | Keep fallback |
| Plan violates the verifier | Raise; never accept that plan |
| Model stage fails or exceeds 120 seconds | Experiment fails; no fabricated predictions or results |

Solver status, fallback use, search time, candidate count and total call time are recorded per optimized wave. A fallback is a feasible prediction-based plan, not a guarantee about realized travel or lateness.

## Gates before a live dispatch service

Version the feature contract and model together; measure preprocessing and inference in the serving environment. Add an outer worker deadline, request IDs, duplicate-dispatch prevention and a transactional commit check against live shopper/order state. Shadow the current policy and report on-time completion, deferral, shopper effort, feasibility rejection, fallback rate and feature freshness. Alert on drift and sustained fallback, with an explicit owner and rollback target. Container, cloud rollout, live dashboards and operational support are design proposals here; they were not deployed or measured.
