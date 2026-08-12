"""问题一精确抽样程序的回归与边界测试。"""

from __future__ import annotations

import math
import unittest

from problem1 import (
    acceptable_defective_count,
    binomial_probabilities,
    build_population_sensitivity_rows,
    classify_observation,
    compute_thresholds,
    cumulative_tails,
    hypergeometric_probabilities,
    search_thresholds,
)


class DistributionTests(unittest.TestCase):
    def test_binomial_probabilities_are_normalised(self) -> None:
        probabilities = binomial_probabilities(22, 0.10)
        self.assertAlmostEqual(math.fsum(probabilities), 1.0, places=14)
        self.assertAlmostEqual(probabilities[0], 0.9**22, places=14)

    def test_hypergeometric_known_small_case(self) -> None:
        probabilities = hypergeometric_probabilities(10, 2, 3)
        expected = [56 / 120, 56 / 120, 8 / 120, 0.0]
        for actual, target in zip(probabilities, expected):
            self.assertAlmostEqual(actual, target, places=14)

    def test_cumulative_tails_include_endpoint(self) -> None:
        left, right = cumulative_tails([0.2, 0.3, 0.5])
        self.assertEqual(left, [0.2, 0.5, 1.0])
        self.assertAlmostEqual(right[0], 1.0)
        self.assertAlmostEqual(right[1], 0.8)
        self.assertAlmostEqual(right[2], 0.5)


class ThresholdTests(unittest.TestCase):
    def test_default_minimum_sample_sizes(self) -> None:
        _, minimums = search_thresholds(50)
        self.assertEqual(minimums["first_reject_sample_size"], 2)
        self.assertEqual(minimums["first_accept_sample_size"], 22)
        self.assertEqual(minimums["first_both_sample_size"], 22)

    def test_default_n22_cutoffs_and_boundary_probabilities(self) -> None:
        result = compute_thresholds(22)
        self.assertEqual(result.accept_cutoff, 0)
        self.assertEqual(result.reject_cutoff, 6)
        self.assertLessEqual(result.accept_tail_at_cutoff, 0.10)
        self.assertGreater(result.accept_tail_after_cutoff, 0.10)
        self.assertLessEqual(result.reject_tail_at_cutoff, 0.05)
        self.assertGreater(result.reject_tail_before_cutoff, 0.05)
        self.assertTrue(result.accept_boundary_maximal_ok)
        self.assertTrue(result.reject_boundary_minimal_ok)
        self.assertTrue(result.disjoint_regions_ok)

    def test_first_extreme_rejection_occurs_at_n2(self) -> None:
        result_n1 = compute_thresholds(1)
        result_n2 = compute_thresholds(2)
        self.assertIsNone(result_n1.reject_cutoff)
        self.assertEqual(result_n2.reject_cutoff, 2)

    def test_hypergeometric_full_census_is_exact(self) -> None:
        population_size = 100
        result = compute_thresholds(
            population_size,
            model="hypergeometric",
            population_size=population_size,
        )
        self.assertEqual(result.acceptable_defectives, 10)
        self.assertEqual(result.first_unacceptable_defectives, 11)
        self.assertEqual(result.accept_cutoff, 10)
        self.assertEqual(result.reject_cutoff, 11)
        self.assertIsNone(result.inconclusive_low)
        self.assertIsNone(result.inconclusive_high)

    def test_observation_classification_has_three_states(self) -> None:
        result = compute_thresholds(22)
        accepted = classify_observation(result, 0)
        self.assertEqual(accepted["decision"], "接收")
        self.assertAlmostEqual(
            accepted["accept_left_tail_probability"],
            0.9**22,
            places=14,
        )
        self.assertEqual(classify_observation(result, 3)["decision"], "证据不足")
        self.assertEqual(classify_observation(result, 6)["decision"], "拒收")

    def test_decimal_floor_for_finite_population_boundary(self) -> None:
        self.assertEqual(acceptable_defective_count(23, 0.10), 2)

    def test_population_sensitivity_approaches_binomial_case(self) -> None:
        rows = build_population_sensitivity_rows(
            [25, 1000],
            0.10,
            0.95,
            0.90,
            50,
            22,
        )
        self.assertLessEqual(rows[0]["first_both_sample_size"], 22)
        self.assertEqual(rows[1]["first_both_sample_size"], 22)

    def test_invalid_sample_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_thresholds(101, model="hypergeometric", population_size=100)
        with self.assertRaises(ValueError):
            compute_thresholds(0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
