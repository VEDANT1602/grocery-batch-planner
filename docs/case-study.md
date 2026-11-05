# From a shopping-time prediction to a dispatch decision

## Topic: fulfillment decisions

Fulfillment joins several decisions that are easy to evaluate separately and harder to connect: predicting shopping time, forming a batch, selecting a shopper and protecting a promise. This project narrows the problem to one planning wave. That keeps the decision and its failure modes inspectable.

## The familiar shortcut

FIFO is transparent. A short-trip greedy rule is usually stronger: build the largest feasible batch, then prefer less time. Both can make locally sensible choices that block other combinations. A model leaderboard would miss that assignment interaction. The opposite shortcut—trusting an optimizer because its result is mathematically optimal—misses the possibility that the application has constructed or interpreted the wrong plan.

## How I approached it

I separated four contracts. The model estimates individual shopping duration using predecision basket and store fields. A calibrated reserve represents one explicit tolerance for prediction error. The optimizer chooses only from feasible candidates. A verifier recomputes constraints using the original wave inputs rather than solver variables or cached candidate metadata.

The solver's objective is lexicographic: maximize the count assigned in this wave, then minimize predicted trip seconds. A computed coefficient makes one additional assigned order worth more than the maximum possible total-time difference. This avoids silently trading an order for a few seconds. Greedy provides both a solution hint and a checked fallback; a solver incumbent cannot displace it with a worse objective.

I held out the final 25 days and fixed the policy before evaluating those waves. No realized shopping time enters candidate creation or optimization. The realized label is used afterward to score lateness in an explicitly defined trip simulator. The buffer is calibrated only on the middle 15 days.

## What changed in the experiment

The fitted duration model reduces held-out individual MAE from 4.284 to 1.103 minutes compared with a training-only store mean. Across 600 dispatch orders, buffered CP-SAT assigns 284 on time compared with 275 for buffered greedy and 239 for buffered FIFO. That is a modest decision gain over a credible heuristic rather than a claim that optimization replaces all simpler methods.

The more informative result is the safety tradeoff. Removing the buffer raises on-time assignments to 302 but also creates 13 late assignments. Buffering yields 284 on time and no late assignments in these waves, while deferring 316 orders. The system cannot choose a product policy from those counts alone: it needs an agreed cost for a missed promise versus a delayed assignment. I preserved that unresolved decision instead of selecting a flattering headline metric.

## Reliability work that changed the design

The reference Windows environment stalled when XGBoost and the OR-Tools native runtime were loaded into one process. The solver worked when isolated. I therefore made training and prediction an explicit subprocess stage with a finite timeout, and kept the planning process free of the XGBoost import. This is an operational boundary worth testing on the deployment platform; it is not proof of the native-library root cause.

I also made fallback observable rather than hoping a deadline would trigger it. A zero-search experiment runs all 25 waves through the greedy fallback, and its plans match greedy exactly. Tests corrupt stored duration, repeat shoppers and orders, cross store boundaries, exceed capacity and send nonfinite budgets. Each failure is rejected before a plan can be accepted.

## What I would do next

Before a real launch I would replay delayed and out-of-order events, learn per-store prediction drift, add shopper arrival times and road travel, and define the cost of deferral. I would compare policies across predeclared demand and disruption regimes, then use a shadow service before any randomized field evaluation. Production readiness would require authenticated inputs, an outer request deadline, measured model inference, immutable model-feature versions and a dispatch commit protocol. None of those deployment claims is included in the CPU results here.
