"""Bounded set packing with an independently checked greedy fallback.

The planner receives predictions, never realized shopping times. A batch is a
single trip, at one store, for at most three orders. It is not a route solver.
"""
from dataclasses import dataclass
from itertools import combinations
import math
from time import perf_counter

from ortools.sat.python import cp_model


def number(value, name, lower=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < lower:
        raise ValueError(f"{name} must be finite and >= {lower}")
    return float(value)


@dataclass(frozen=True)
class Order:
    order_id: str
    store: str
    items: int
    predicted_minutes: float
    deadline_minutes: float
    distance_km: float
    accepted_index: int


@dataclass(frozen=True)
class Shopper:
    shopper_id: str
    store: str
    capacity_items: int
    pace: float = 1.0


@dataclass(frozen=True)
class Batch:
    shopper_id: str
    order_ids: tuple[str, ...]
    predicted_seconds: int


def validate_inputs(orders, shoppers):
    if len(orders) > 40 or len(shoppers) > 12:
        raise ValueError("one wave is limited to 40 orders and 12 shoppers; partition upstream")
    for rows, attr in ((orders, "order_id"), (shoppers, "shopper_id")):
        ids = [getattr(r, attr) for r in rows]
        if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{attr} must be unique nonempty strings")
    for o in orders:
        if not isinstance(o.store, str) or not o.store:
            raise ValueError("store must be nonempty")
        if isinstance(o.items, bool) or not isinstance(o.items, int) or o.items <= 0:
            raise ValueError("items must be a positive integer")
        if isinstance(o.accepted_index, bool) or not isinstance(o.accepted_index, int) or o.accepted_index < 0:
            raise ValueError("accepted_index must be a nonnegative integer")
        number(o.predicted_minutes, "predicted_minutes", 0.01)
        number(o.deadline_minutes, "deadline_minutes", 0.01)
        number(o.distance_km, "distance_km")
        if o.predicted_minutes > 240 or o.deadline_minutes > 480 or o.distance_km > 100:
            raise ValueError("order exceeds the declared single-wave numeric domain")
    for s in shoppers:
        if not isinstance(s.store, str) or not s.store:
            raise ValueError("store must be nonempty")
        if isinstance(s.capacity_items, bool) or not isinstance(s.capacity_items, int) or s.capacity_items <= 0:
            raise ValueError("capacity_items must be a positive integer")
        number(s.pace, "pace", 0.01)
        if s.pace > 3:
            raise ValueError("pace must be <= 3")


def trip_seconds(orders, shopper, buffer_minutes=0.0):
    """Explicit simulation approximation, rounded upward to integer seconds.

    A multi-order trip shares the shopping effort but incurs a stop per order.
    Distance is additive: this intentionally does not claim actual route quality.
    """
    number(buffer_minutes, "buffer_minutes")
    share = 0.9 if len(orders) == 1 else 0.68
    minutes = 3 + share * sum(o.predicted_minutes for o in orders) * shopper.pace
    minutes += 1.4 * sum(o.distance_km for o in orders) + 1.2 * len(orders)
    minutes += buffer_minutes
    return math.ceil(minutes * 60)


def feasible_batch(orders, shopper, buffer_minutes=0.0):
    if not 1 <= len(orders) <= 3 or any(o.store != shopper.store for o in orders):
        return False
    return (sum(o.items for o in orders) <= shopper.capacity_items
            and trip_seconds(orders, shopper, buffer_minutes)
            <= math.floor(min(o.deadline_minutes for o in orders) * 60))


def candidates(orders, shoppers, buffer_minutes=0.0):
    validate_inputs(orders, shoppers)
    number(buffer_minutes, "buffer_minutes")
    result = []
    for s in shoppers:
        local = sorted((o for o in orders if o.store == s.store), key=lambda o: o.order_id)
        for size in (1, 2, 3):
            for subset in combinations(local, size):
                if feasible_batch(subset, s, buffer_minutes):
                    result.append(Batch(s.shopper_id, tuple(o.order_id for o in subset),
                                        trip_seconds(subset, s, buffer_minutes)))
    return result


def validate_plan(plan, orders, shoppers, buffer_minutes=0.0):
    """Recompute every constraint; do not trust solver/fallback cached costs."""
    validate_inputs(orders, shoppers)
    number(buffer_minutes, "buffer_minutes")
    omap = {o.order_id: o for o in orders}
    smap = {s.shopper_id: s for s in shoppers}
    seen_orders, seen_shoppers = set(), set()
    for b in plan:
        if b.shopper_id not in smap or b.shopper_id in seen_shoppers:
            raise ValueError("unknown or repeated shopper")
        if not b.order_ids or len(set(b.order_ids)) != len(b.order_ids):
            raise ValueError("empty batch or repeated order within batch")
        if any(x not in omap or x in seen_orders for x in b.order_ids):
            raise ValueError("unknown or repeated order")
        subset = [omap[x] for x in b.order_ids]
        shopper = smap[b.shopper_id]
        if not feasible_batch(subset, shopper, buffer_minutes):
            raise ValueError("batch violates store, capacity, size or predicted deadline")
        if isinstance(b.predicted_seconds, bool) or b.predicted_seconds != trip_seconds(subset, shopper, buffer_minutes):
            raise ValueError("cached duration differs from independent recomputation")
        seen_orders.update(b.order_ids)
        seen_shoppers.add(b.shopper_id)
    return True


def greedy_plan(pool):
    """Max orders first, then short trips; no solver dependency for selection."""
    chosen, orders, shoppers = [], set(), set()
    for b in sorted(pool, key=lambda b: (-len(b.order_ids), b.predicted_seconds, b.shopper_id, b.order_ids)):
        if b.shopper_id not in shoppers and not orders.intersection(b.order_ids):
            chosen.append(b)
            orders.update(b.order_ids)
            shoppers.add(b.shopper_id)
    return chosen


def fifo_plan(orders, shoppers, buffer_minutes=0.0):
    validate_inputs(orders, shoppers)
    number(buffer_minutes, "buffer_minutes")
    used, result = set(), []
    for s in sorted(shoppers, key=lambda s: s.shopper_id):
        subset = []
        for o in sorted(orders, key=lambda o: (o.accepted_index, o.order_id)):
            if o.order_id not in used and feasible_batch(subset + [o], s, buffer_minutes):
                subset.append(o)
        if subset:
            result.append(Batch(s.shopper_id, tuple(o.order_id for o in subset),
                                trip_seconds(subset, s, buffer_minutes)))
            used.update(o.order_id for o in subset)
    validate_plan(result, orders, shoppers, buffer_minutes)
    return result


def plan_wave(orders, shoppers, buffer_minutes=0.0, solver_seconds=0.2):
    """The budget limits solver search, not total request latency.

    Candidate enumeration and independent validation run outside that budget.
    Zero budget deliberately skips solver search and exercises the fallback.
    """
    number(solver_seconds, "solver_seconds")
    started = perf_counter()
    pool = candidates(orders, shoppers, buffer_minutes)
    fallback = greedy_plan(pool)
    validate_plan(fallback, orders, shoppers, buffer_minutes)
    meta = {"candidate_count": len(pool), "solver_status": "SKIPPED", "fallback_used": True,
            "search_seconds": 0.0, "search_limit_seconds": solver_seconds}
    plan = fallback
    if pool and solver_seconds > 0:
        model = cp_model.CpModel()
        variables = [model.new_bool_var(f"batch_{i}") for i in range(len(pool))]
        for s in shoppers:
            model.add(sum(v for v, b in zip(variables, pool) if b.shopper_id == s.shopper_id) <= 1)
        for o in orders:
            model.add(sum(v for v, b in zip(variables, pool) if o.order_id in b.order_ids) <= 1)
        # Lexicographic objective: one additional order beats every possible
        # difference in total trip seconds across the wave.
        large = 1 + sum(max((b.predicted_seconds for b in pool if b.shopper_id == s.shopper_id), default=0)
                        for s in shoppers)
        model.maximize(sum(v * (large * len(b.order_ids) - b.predicted_seconds)
                           for v, b in zip(variables, pool)))
        hint = {(b.shopper_id, b.order_ids) for b in fallback}
        for v, b in zip(variables, pool):
            model.add_hint(v, int((b.shopper_id, b.order_ids) in hint))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = solver_seconds
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 42
        status = solver.solve(model)
        meta.update(solver_status=solver.status_name(status), search_seconds=solver.wall_time)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            proposal = [b for b, v in zip(pool, variables) if solver.value(v)]
            validate_plan(proposal, orders, shoppers, buffer_minutes)
            # A valid incumbent is allowed only if no worse than our fallback.
            score = lambda p: (sum(len(b.order_ids) for b in p), -sum(b.predicted_seconds for b in p))
            if score(proposal) >= score(fallback):
                plan = proposal
                meta["fallback_used"] = False
    validate_plan(plan, orders, shoppers, buffer_minutes)
    meta["total_seconds"] = perf_counter() - started
    return plan, meta
