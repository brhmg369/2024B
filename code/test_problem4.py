import math
import unittest

from programs import q4_bayesian_model as q4


class TestQuestion4Sensitivity(unittest.TestCase):
    def test_default_sample_sizes_include_large_sample_limits(self) -> None:
        self.assertEqual(q4.DEFAULT_SAMPLE_SIZES, (40, 100, 200, 1000, 10000))

    def test_sample_size_must_make_all_defect_counts_integer(self) -> None:
        self.assertTrue(q4.is_valid_common_sample_size(40))
        self.assertTrue(q4.is_valid_common_sample_size(10000))
        self.assertFalse(q4.is_valid_common_sample_size(41))

    def test_beta_posterior_statistics_match_closed_form_moments(self) -> None:
        row = q4.posterior_statistics(0.10, 40)
        alpha = 5
        beta = 37
        expected_mean = alpha / (alpha + beta)
        expected_std = math.sqrt(
            alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
        )

        self.assertEqual(row["k"], 4)
        self.assertEqual(row["alpha"], alpha)
        self.assertEqual(row["beta"], beta)
        self.assertAlmostEqual(row["posterior_mean"], expected_mean)
        self.assertAlmostEqual(row["posterior_std"], expected_std)
        self.assertLess(row["ci95_lower"], row["posterior_mean"])
        self.assertGreater(row["ci95_upper"], row["posterior_mean"])

    def test_deterministic_q2_best_tracks_ties(self) -> None:
        best = q4.deterministic_q2_best_by_case()

        self.assertIn("1101", best[3])
        self.assertIn("1111", best[3])


if __name__ == "__main__":
    unittest.main()
