from __future__ import annotations

import unittest

import programs.q2_decision_model as q2


class BeliefStateTests(unittest.TestCase):
    def test_canonicalize_returns_probability_vector(self) -> None:
        state = q2.canonicalize({(q2.GOOD, q2.BAD): 2.0, (q2.BAD, q2.GOOD): 1.0})
        self.assertAlmostEqual(sum(state), 1.0, places=10)
        self.assertTrue(all(value >= 0.0 for value in state))

    def test_action_transition_probabilities_close(self) -> None:
        params = q2.CASES[0]
        bought = q2.buy_transition(q2.START_STATE, params, part=1, inspect=0)
        self.assertAlmostEqual(sum(prob for prob, _ in bought.transitions), 1.0)
        state = bought.transitions[0][1]
        inspected = q2.inspect_transition(state, params, part=1)
        self.assertAlmostEqual(sum(prob for prob, _ in inspected.transitions), 1.0)


class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        q2.ROUND_DIGITS = 6
        cls.solutions = [q2.solve_case(params) for params in q2.CASES]

    def test_all_cases_converge_with_small_bellman_residual(self) -> None:
        for solution in self.solutions:
            with self.subTest(case=solution["params"].case):
                self.assertTrue(solution["converged"])
                self.assertLess(solution["bellman_residual"], 1e-8)

    def test_expected_costs_match_reference_run(self) -> None:
        expected = (37.077779, 44.0, 39.346664, 41.25, 40.55, 34.32133)
        for solution, target in zip(self.solutions, expected):
            value = solution["values"][q2.START_STATE]
            self.assertAlmostEqual(value, target, places=5)

    def test_belief_mdp_dominates_restricted_fixed_policy(self) -> None:
        fixed = (37.077778, 44.0, 39.411111, 41.25, 40.55, 34.32133)
        for solution, baseline in zip(self.solutions, fixed):
            value = solution["values"][q2.START_STATE]
            self.assertLessEqual(value, baseline + 2e-6)

    def test_case3_uses_state_dependent_improvement(self) -> None:
        traced = q2.trace_initial_policy(self.solutions[2])
        self.assertEqual(traced["initial_component_action_1"], "buy_p1_notest")
        self.assertEqual(traced["initial_component_action_2"], "buy_p2_notest")
        self.assertEqual(traced["first_assembly_action"], "assemble_test_disassemble")
        self.assertEqual(traced["after_first_defect_action"], "inspect_p1")


if __name__ == "__main__":
    unittest.main()
