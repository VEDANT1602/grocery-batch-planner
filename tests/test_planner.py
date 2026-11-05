from dataclasses import replace
from itertools import combinations
import math
import unittest

from grocery_planner.planner import (
    Batch, Order, Shopper, candidates, feasible_batch, fifo_plan,
    greedy_plan, plan_wave, trip_seconds, validate_inputs, validate_plan,
)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.orders = [Order("a", "s", 7, 8., 50., 1., 0),
                       Order("b", "s", 10, 9., 50., 1., 1),
                       Order("c", "s", 13, 10., 50., 1., 2),
                       Order("d", "s", 8, 6., 50., 1., 3)]
        self.shoppers = [Shopper("x", "s", 24), Shopper("y", "s", 24)]

    def test_solver_matches_exhaustive_small_instance(self):
        pool = candidates(self.orders, self.shoppers, 2.)
        valid = [[]]
        for size in (1, 2):
            for subset in combinations(pool, size):
                try:
                    validate_plan(subset, self.orders, self.shoppers, 2.)
                    valid.append(subset)
                except ValueError:
                    pass
        score = lambda p: (sum(len(b.order_ids) for b in p), -sum(b.predicted_seconds for b in p))
        expected = max(map(score, valid))
        plan, meta = plan_wave(self.orders, self.shoppers, 2., 1.)
        self.assertEqual(meta["solver_status"], "OPTIMAL")
        self.assertEqual(score(plan), expected)

    def test_zero_budget_is_valid_deterministic_fallback(self):
        plan, meta = plan_wave(self.orders, self.shoppers, 2., 0)
        self.assertEqual(plan, greedy_plan(candidates(self.orders, self.shoppers, 2.)))
        self.assertTrue(meta["fallback_used"])
        self.assertEqual(meta["solver_status"], "SKIPPED")
        self.assertTrue(validate_plan(plan, self.orders, self.shoppers, 2.))

    def test_tiny_search_budget_never_returns_worse_than_fallback(self):
        plan, _ = plan_wave(self.orders, self.shoppers, 2., 1e-8)
        fallback = greedy_plan(candidates(self.orders, self.shoppers, 2.))
        score = lambda p: (sum(len(b.order_ids) for b in p), -sum(b.predicted_seconds for b in p))
        self.assertGreaterEqual(score(plan), score(fallback))
        self.assertTrue(validate_plan(plan, self.orders, self.shoppers, 2.))

    def test_unknown_and_duplicate_entities_rejected(self):
        plan, _ = plan_wave(self.orders, self.shoppers)
        for corrupted in ([plan[0], plan[0]], [replace(plan[0], shopper_id="missing")],
                          [replace(plan[0], order_ids=("missing",))],
                          [replace(plan[0], order_ids=("a", "a"))]):
            with self.subTest(corrupted=corrupted), self.assertRaises(ValueError):
                validate_plan(corrupted, self.orders, self.shoppers)

    def test_verifier_recomputes_cached_seconds(self):
        plan, _ = plan_wave(self.orders, self.shoppers)
        with self.assertRaises(ValueError):
            validate_plan([replace(plan[0], predicted_seconds=0)], self.orders, self.shoppers)

    def test_store_capacity_size_and_deadline_are_independent_guards(self):
        order = self.orders[0]
        self.assertFalse(feasible_batch([replace(order, store="other")], self.shoppers[0]))
        self.assertFalse(feasible_batch([replace(order, items=25)], self.shoppers[0]))
        self.assertFalse(feasible_batch(self.orders, Shopper("z", "s", 100)))
        self.assertFalse(feasible_batch([replace(order, deadline_minutes=.5)], self.shoppers[0]))

    def test_rounding_cannot_admit_fractional_second_overrun(self):
        order = self.orders[0]
        duration = trip_seconds([order], self.shoppers[0])
        exact = replace(order, deadline_minutes=duration / 60)
        too_short = replace(order, deadline_minutes=(duration - .1) / 60)
        self.assertTrue(feasible_batch([exact], self.shoppers[0]))
        self.assertFalse(feasible_batch([too_short], self.shoppers[0]))

    def test_no_orders_no_shoppers_and_impossible_orders_are_valid_empty(self):
        for orders, shoppers in (([], self.shoppers), (self.orders, []),
                                  ([replace(self.orders[0], items=1000)], self.shoppers)):
            with self.subTest(orders=orders, shoppers=shoppers):
                plan, _ = plan_wave(orders, shoppers)
                self.assertEqual(plan, [])
                self.assertTrue(validate_plan(plan, orders, shoppers))

    def test_fallback_and_fifo_cannot_cross_store(self):
        orders = [replace(o, store="other") for o in self.orders]
        self.assertEqual(fifo_plan(orders, self.shoppers), [])
        self.assertEqual(plan_wave(orders, self.shoppers, solver_seconds=0)[0], [])

    def test_all_public_planning_paths_reject_nonfinite_parameters(self):
        for invalid in (math.nan, math.inf, -math.inf, -1, True, "0.2"):
            for fn in (lambda: plan_wave(self.orders, self.shoppers, solver_seconds=invalid),
                       lambda: plan_wave(self.orders, self.shoppers, buffer_minutes=invalid),
                       lambda: fifo_plan(self.orders, self.shoppers, buffer_minutes=invalid),
                       lambda: validate_plan([], self.orders, self.shoppers, buffer_minutes=invalid)):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    fn()

    def test_untrusted_numeric_fields_rejected_before_solver(self):
        for field in ("predicted_minutes", "deadline_minutes", "distance_km"):
            for invalid in (math.nan, math.inf, -1, True, "10"):
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    plan_wave([replace(self.orders[0], **{field: invalid})], self.shoppers)
        for field, invalid in (("items", 2.5), ("items", True), ("accepted_index", -1), ("accepted_index", True)):
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                validate_inputs([replace(self.orders[0], **{field: invalid})], self.shoppers)

    def test_duplicate_input_ids_rejected(self):
        for orders, shoppers in ((self.orders * 2, self.shoppers), (self.orders, self.shoppers * 2)):
            with self.assertRaises(ValueError):
                plan_wave(orders, shoppers)

    def test_partition_and_numeric_domain_limits_fail_before_enumeration(self):
        orders = [replace(self.orders[0], order_id=str(i)) for i in range(41)]
        with self.assertRaises(ValueError):
            plan_wave(orders, self.shoppers)
        for field, value in (("predicted_minutes", 241), ("deadline_minutes", 481), ("distance_km", 101)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                plan_wave([replace(self.orders[0], **{field: value})], self.shoppers)


if __name__ == "__main__":
    unittest.main()
