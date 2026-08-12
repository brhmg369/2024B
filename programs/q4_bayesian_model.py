"""Question 4 Bayesian defect-rate uncertainty model.

This module reuses the production logic of Q2/Q3 without changing their source
files.  Defect rates are treated as Beta posterior random variables generated
from the modelling assumption n in {40, 100, 200, 1000, 10000} and
k = n * p_hat.

The inner production expectation is analytical.  Posterior sampling is only
used for parameter uncertainty, and the same posterior scenarios are shared by
all strategies under the same experiment setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import argparse
import csv
import json
import math
import random
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from programs import q2_decision_model as q2
from programs import q3_decision_model as q3


SCENARIOS = 5000
BASE_SEED = 4024
Q2_STRATEGY_BITS = 7
Q3_STRATEGY_BITS = 16
NOMINAL_RATES = (0.05, 0.10, 0.20)
PRIORS = {
    "uniform": (1.0, 1.0),
    "jeffreys": (0.5, 0.5),
}
DEFAULT_SAMPLE_SIZES = (40, 100, 200, 1000, 10000)
CRITICAL_SEARCH_GRID = (
    40,
    50,
    60,
    80,
    100,
    120,
    140,
    160,
    180,
    200,
    250,
    300,
    400,
    500,
    750,
    1000,
    1500,
    2000,
    3000,
    5000,
    7500,
    10000,
)


@dataclass(frozen=True)
class PosteriorSpec:
    nominal_rate: float
    n: int
    k: int
    alpha: float
    beta: float


@dataclass(frozen=True)
class ScenarioSummary:
    mean: float
    sd: float
    q05: float
    feasible: int


@dataclass(frozen=True)
class GroupScenario:
    cost: np.ndarray | None
    good_prob: np.ndarray | None
    feasible: bool
    reason: str = ""


def strategy_code(bits: tuple[int, ...]) -> str:
    return "".join(str(bit) for bit in bits)


def bits_from_code(code: str) -> tuple[int, ...]:
    bits = tuple(int(char) for char in code.strip())
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("strategy code must be binary")
    return bits


def iter_bits(width: int):
    return product((0, 1), repeat=width)


def posterior_spec(rate: float, n: int, prior: str = "uniform") -> PosteriorSpec:
    raw_k = n * rate
    k = int(round(raw_k))
    if abs(raw_k - k) > 1e-9:
        raise ValueError(f"n={n} cannot represent rate={rate:g} with integer k")
    pseudo_a, pseudo_b = PRIORS[prior]
    return PosteriorSpec(rate, n, k, k + pseudo_a, n - k + pseudo_b)


def is_valid_common_sample_size(n: int) -> bool:
    try:
        for rate in NOMINAL_RATES:
            posterior_spec(rate, n)
    except ValueError:
        return False
    return True


def beta_ppf(probability: float, alpha: float, beta: float) -> float:
    try:
        from scipy.stats import beta as beta_distribution

        return float(beta_distribution.ppf(probability, alpha, beta))
    except Exception:
        mean = alpha / (alpha + beta)
        variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
        z = 1.959963984540054
        return min(1.0, max(0.0, mean + z * math.sqrt(variance) * (-1 if probability < 0.5 else 1)))


def posterior_statistics(rate: float, n: int, prior: str = "uniform") -> dict:
    spec = posterior_spec(rate, n, prior)
    total = spec.alpha + spec.beta
    mean = spec.alpha / total
    variance = spec.alpha * spec.beta / (total * total * (total + 1))
    std = math.sqrt(variance)
    return {
        "n": n,
        "nominal_rate": rate,
        "k": spec.k,
        "alpha": spec.alpha,
        "beta": spec.beta,
        "posterior_mean": mean,
        "posterior_std": std,
        "ci95_lower": beta_ppf(0.025, spec.alpha, spec.beta),
        "ci95_upper": beta_ppf(0.975, spec.alpha, spec.beta),
        "mean_minus_nominal": mean - rate,
    }


def build_posterior_statistics_rows(sample_sizes: list[int], prior: str = "uniform") -> list[dict]:
    rows: list[dict] = []
    for n in sample_sizes:
        for rate in NOMINAL_RATES:
            row = posterior_statistics(rate, n, prior)
            rows.append(
                {
                    "n": row["n"],
                    "nominal_rate": row["nominal_rate"],
                    "k": row["k"],
                    "posterior": f"Beta({row['alpha']},{row['beta']})",
                    "posterior_mean": finite_round(row["posterior_mean"], 10),
                    "posterior_std": finite_round(row["posterior_std"], 10),
                    "ci95_lower": finite_round(row["ci95_lower"], 10),
                    "ci95_upper": finite_round(row["ci95_upper"], 10),
                    "mean_minus_nominal": finite_round(row["mean_minus_nominal"], 10),
                }
            )
    return rows


def sample_beta(
    rate: float,
    n: int,
    size: int,
    rng: np.random.Generator,
    prior: str = "uniform",
) -> np.ndarray:
    spec = posterior_spec(rate, n, prior)
    return rng.beta(spec.alpha, spec.beta, size=size)


def summarize_profit(values: np.ndarray | None, risk: bool = True) -> ScenarioSummary:
    if values is None or not np.all(np.isfinite(values)):
        return ScenarioSummary(-math.inf, math.inf, -math.inf, 0)
    mean = float(np.mean(values))
    if not risk:
        return ScenarioSummary(mean, math.nan, math.nan, 1)
    return ScenarioSummary(
        mean=mean,
        sd=float(np.std(values, ddof=0)),
        q05=float(np.quantile(values, 0.05)),
        feasible=1,
    )


def finite_round(value: float, digits: int = 6) -> float | str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return ""
    return round(value, digits)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def q2_case_scenarios(
    params: q2.CaseParams,
    n: int,
    size: int,
    seed: int,
    prior: str = "uniform",
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + params.case * 97)
    return {
        "p1": sample_beta(params.p1, n, size, rng, prior),
        "p2": sample_beta(params.p2, n, size, rng, prior),
        "pf": sample_beta(params.pf, n, size, rng, prior),
    }


Q2_STATUS_NONE = "N"
Q2_STATUS_KNOWN_GOOD = "KG"
Q2_STATUS_UNKNOWN_GOOD = "UG"
Q2_STATUS_UNKNOWN_BAD = "UB"
Q2_PHASE_INITIAL = 0
Q2_PHASE_RECOVERY = 1


def _q2_part_params(params: q2.CaseParams, part: int) -> tuple[float, float, float]:
    if part == 0:
        return params.p1, params.c1, params.t1
    return params.p2, params.c2, params.t2


def _q2_next_recovered(actual_good: bool, known_good: bool) -> str:
    if actual_good and known_good:
        return Q2_STATUS_KNOWN_GOOD
    if actual_good:
        return Q2_STATUS_UNKNOWN_GOOD
    return Q2_STATUS_UNKNOWN_BAD


def _q2_seven_part_options(
    strategy: tuple[int, ...],
    params: q2.CaseParams,
    part: int,
    status: str,
    p1: np.ndarray,
    p2: np.ndarray,
    pf: np.ndarray,
):
    """Options for a part entering assembly: list of (prob, cost, good, known)."""
    p, c, t = _q2_part_params(params, part)
    ones = np.ones_like(p1)
    if status == Q2_STATUS_NONE:
        if strategy[part]:
            return [(ones, (c + t) / (1.0 - p), True, True)]
        return [
            (1.0 - p, np.full_like(p1, c), True, False),
            (p, np.full_like(p1, c), False, False),
        ]
    if status == Q2_STATUS_KNOWN_GOOD:
        return [(ones, np.zeros_like(p1), True, True)]
    inspect = strategy[4 + part]
    if status == Q2_STATUS_UNKNOWN_GOOD:
        if inspect:
            return [(ones, np.full_like(p1, t), True, True)]
        return [(ones, np.zeros_like(p1), True, False)]
    # UNKNOWN_BAD
    if inspect:
        options = []
        for prob, cost, good, known in _q2_seven_part_options(
            strategy, params, part, Q2_STATUS_NONE, p1, p2, pf
        ):
            options.append((prob, cost + t, good, known))
        return options
    return [(ones, np.zeros_like(p1), False, False)]


def _q2_seven_state_terms(
    strategy: tuple[int, ...],
    params: q2.CaseParams,
    state: tuple[str, str, int],
    p1: np.ndarray,
    p2: np.ndarray,
    pf: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[str, str, int], np.ndarray]]:
    status1, status2, phase = state
    product_inspection = strategy[2] if phase == Q2_PHASE_INITIAL else strategy[6]
    constant = np.zeros_like(p1)
    transitions: dict[tuple[str, str, int], np.ndarray] = {}
    ones = np.ones_like(p1)

    for prob1, cost1, good1, known1 in _q2_seven_part_options(
        strategy, params, 0, status1, p1, p2, pf
    ):
        for prob2, cost2, good2, known2 in _q2_seven_part_options(
            strategy, params, 1, status2, p1, p2, pf
        ):
            base_prob = prob1 * prob2
            base_cost = cost1 + cost2 + params.ca + product_inspection * params.tf
            if good1 and good2:
                quality_outcomes = [(1.0 - pf, True), (pf, False)]
            else:
                quality_outcomes = [(ones, False)]
            for quality_prob, product_good in quality_outcomes:
                prob = base_prob * quality_prob
                constant = constant + prob * base_cost
                if product_good:
                    continue
                extra = (
                    np.zeros_like(p1)
                    if product_inspection
                    else np.full_like(p1, params.exchange_loss)
                )
                if strategy[3]:
                    extra = extra + params.disassemble_cost
                    next_state = (
                        _q2_next_recovered(good1, known1),
                        _q2_next_recovered(good2, known2),
                        Q2_PHASE_RECOVERY,
                    )
                else:
                    next_state = (Q2_STATUS_NONE, Q2_STATUS_NONE, Q2_PHASE_INITIAL)
                constant = constant + prob * extra
                transitions[next_state] = transitions.get(next_state, 0.0) + prob
    return constant, transitions


def _q2_seven_reachable_states(
    strategy: tuple[int, ...],
    params: q2.CaseParams,
) -> list[tuple[str, str, int]]:
    p1 = np.array([params.p1], dtype=float)
    p2 = np.array([params.p2], dtype=float)
    pf = np.array([params.pf], dtype=float)
    start = (Q2_STATUS_NONE, Q2_STATUS_NONE, Q2_PHASE_INITIAL)
    seen = {start}
    ordered = [start]
    cursor = 0
    while cursor < len(ordered):
        state = ordered[cursor]
        cursor += 1
        _, transitions = _q2_seven_state_terms(strategy, params, state, p1, p2, pf)
        for next_state, prob in transitions.items():
            if prob[0] <= 0 or next_state in seen:
                continue
            seen.add(next_state)
            ordered.append(next_state)
    return ordered


def _q2_seven_scalar_cost(
    strategy: tuple[int, ...],
    params: q2.CaseParams,
) -> float | None:
    states = _q2_seven_reachable_states(strategy, params)
    index = {state: i for i, state in enumerate(states)}
    n = len(states)
    p1 = np.array([params.p1], dtype=float)
    p2 = np.array([params.p2], dtype=float)
    pf = np.array([params.pf], dtype=float)
    matrix = np.zeros((n, n), dtype=float)
    rhs = np.zeros(n, dtype=float)
    for i, state in enumerate(states):
        matrix[i, i] = 1.0
        constant, transitions = _q2_seven_state_terms(
            strategy, params, state, p1, p2, pf
        )
        rhs[i] = float(constant[0])
        for next_state, prob in transitions.items():
            matrix[i, index[next_state]] -= float(prob[0])
    try:
        values = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    cost = float(values[index[(Q2_STATUS_NONE, Q2_STATUS_NONE, Q2_PHASE_INITIAL)]])
    if not math.isfinite(cost) or cost < 0:
        return None
    return cost


def _q2_seven_profit_vector(
    strategy: tuple[int, ...],
    params: q2.CaseParams,
    scenarios: dict[str, np.ndarray],
) -> np.ndarray | None:
    p1 = scenarios["p1"]
    p2 = scenarios["p2"]
    pf = scenarios["pf"]
    states = _q2_seven_reachable_states(strategy, params)
    index = {state: i for i, state in enumerate(states)}
    n = len(states)
    size = len(p1)
    matrix = np.zeros((size, n, n), dtype=float)
    rhs = np.zeros((size, n, 1), dtype=float)
    for i, state in enumerate(states):
        matrix[:, i, i] = 1.0
        constant, transitions = _q2_seven_state_terms(
            strategy, params, state, p1, p2, pf
        )
        rhs[:, i, 0] = constant
        for next_state, prob in transitions.items():
            matrix[:, i, index[next_state]] -= prob
    try:
        values = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    cost = values[:, index[(Q2_STATUS_NONE, Q2_STATUS_NONE, Q2_PHASE_INITIAL)], 0]
    if not np.all(np.isfinite(cost)):
        return None
    return params.sale - cost


def q2_profit_scenarios(
    strategy: tuple[int, ...],
    params: q2.CaseParams,
    scenarios: dict[str, np.ndarray],
) -> np.ndarray | None:
    if len(strategy) != Q2_STRATEGY_BITS:
        raise ValueError("Q2 strategy must have 7 bits")
    if _q2_seven_scalar_cost(strategy, params) is None:
        return None
    return _q2_seven_profit_vector(strategy, params, scenarios)


def q2_fixed_profit(
    strategy: tuple[int, ...],
    params: q2.CaseParams,
) -> ScenarioSummary:
    scenarios = {
        "p1": np.array([params.p1], dtype=float),
        "p2": np.array([params.p2], dtype=float),
        "pf": np.array([params.pf], dtype=float),
    }
    return summarize_profit(q2_profit_scenarios(strategy, params, scenarios))


def deterministic_q2_best_by_case(tolerance: float = 1e-9) -> dict[int, tuple[str, ...]]:
    best: dict[int, tuple[str, ...]] = {}
    for params in q2.CASES:
        rows = []
        for strategy in iter_bits(Q2_STRATEGY_BITS):
            summary = q2_fixed_profit(strategy, params)
            rows.append((summary.mean, strategy_code(strategy)))
        best_profit = max(value for value, _ in rows)
        best[params.case] = tuple(
            code for value, code in rows if abs(value - best_profit) <= tolerance
        )
    return best


def evaluate_q2_case(
    params: q2.CaseParams,
    n: int,
    size: int,
    seed: int,
    prior: str = "uniform",
) -> tuple[list[dict], dict]:
    scenarios = q2_case_scenarios(params, n, size, seed, prior)
    rows: list[dict] = []

    for strategy in iter_bits(Q2_STRATEGY_BITS):
        fixed = q2_fixed_profit(strategy, params)
        bayes = summarize_profit(q2_profit_scenarios(strategy, params, scenarios))
        rows.append(
            {
                "case": params.case,
                "n": n,
                "strategy": strategy_code(strategy),
                "feasible_fixed": fixed.feasible,
                "fixed_expected_profit": finite_round(fixed.mean),
                "feasible_bayes": bayes.feasible,
                "bayes_expected_profit": finite_round(bayes.mean),
                "bayes_profit_sd": finite_round(bayes.sd),
                "bayes_profit_q05": finite_round(bayes.q05),
            }
        )

    fixed_best = max(
        (row for row in rows if row["feasible_fixed"]),
        key=lambda row: row["fixed_expected_profit"],
    )
    bayes_best = max(
        (row for row in rows if row["feasible_bayes"]),
        key=lambda row: row["bayes_expected_profit"],
    )
    best = {
        "case": params.case,
        "n": n,
        "fixed_best_strategy": fixed_best["strategy"],
        "fixed_best_expected_profit": fixed_best["fixed_expected_profit"],
        "bayes_best_strategy": bayes_best["strategy"],
        "bayes_best_expected_profit": bayes_best["bayes_expected_profit"],
        "bayes_profit_sd": bayes_best["bayes_profit_sd"],
        "bayes_profit_q05": bayes_best["bayes_profit_q05"],
        "matches_fixed_best": int(fixed_best["strategy"] == bayes_best["strategy"]),
        "strategy_changed": int(fixed_best["strategy"] != bayes_best["strategy"]),
    }
    return rows, best


def q3_scenarios(
    n: int,
    size: int,
    seed: int,
    prior: str = "uniform",
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 3009)
    return {
        "parts": np.vstack(
            [sample_beta(rate, n, size, rng, prior) for rate in q3.TABLE2.part_defect]
        ),
        "semis": np.vstack(
            [sample_beta(rate, n, size, rng, prior) for rate in q3.TABLE2.semi_defect]
        ),
        "final": sample_beta(q3.TABLE2.final_defect, n, size, rng, prior),
    }


def group_code(strategy: tuple[int, ...], group: int) -> int:
    if group == 0:
        part_bits = strategy[0:3]
        return part_bits[0] + 2 * part_bits[1] + 4 * part_bits[2] + 8 * strategy[8] + 16 * strategy[11]
    if group == 1:
        part_bits = strategy[3:6]
        return part_bits[0] + 2 * part_bits[1] + 4 * part_bits[2] + 8 * strategy[9] + 16 * strategy[12]
    part_bits = strategy[6:8]
    return part_bits[0] + 2 * part_bits[1] + 4 * strategy[10] + 8 * strategy[13]


def build_group_option(
    group: int,
    code: int,
    scenarios: dict[str, np.ndarray],
) -> GroupScenario:
    if group in (0, 1):
        local_count = 3
        semi_test = (code >> 3) & 1
        semi_disassemble = (code >> 4) & 1
    else:
        local_count = 2
        semi_test = (code >> 2) & 1
        semi_disassemble = (code >> 3) & 1

    part_indices = q3.PART_GROUPS[group]
    part_cost = np.zeros(scenarios["final"].shape, dtype=float)
    children_good_prob = np.ones(scenarios["final"].shape, dtype=float)
    all_children_inspected = True

    for local, part in enumerate(part_indices):
        tested = (code >> local) & 1
        p = scenarios["parts"][part]
        if tested:
            part_cost += (q3.TABLE2.part_price[part] + q3.TABLE2.part_test[part]) / (1.0 - p)
        else:
            all_children_inspected = False
            part_cost += q3.TABLE2.part_price[part]
            children_good_prob *= 1.0 - p

    p_semi = scenarios["semis"][group]
    success = children_good_prob * (1.0 - p_semi)

    if semi_test:
        if semi_disassemble:
            if not all_children_inspected:
                return GroupScenario(None, None, False, "semi disassembly can loop forever with uninspected children")
            semi_cycle = (
                q3.TABLE2.semi_assembly[group]
                + q3.TABLE2.semi_test[group]
                + p_semi * q3.TABLE2.semi_disassemble[group]
            ) / (1.0 - p_semi)
            return GroupScenario(part_cost + semi_cycle, np.ones_like(p_semi), True)

        attempt_cost = (
            part_cost
            + q3.TABLE2.semi_assembly[group]
            + q3.TABLE2.semi_test[group]
        )
        return GroupScenario(attempt_cost / success, np.ones_like(p_semi), True)

    return GroupScenario(
        part_cost + q3.TABLE2.semi_assembly[group],
        success,
        True,
    )


def build_group_cache(scenarios: dict[str, np.ndarray]) -> list[dict[int, GroupScenario]]:
    return [
        {code: build_group_option(0, code, scenarios) for code in range(32)},
        {code: build_group_option(1, code, scenarios) for code in range(32)},
        {code: build_group_option(2, code, scenarios) for code in range(16)},
    ]


def q3_profit_scenarios(
    strategy: tuple[int, ...],
    scenarios: dict[str, np.ndarray],
    group_cache: list[dict[int, GroupScenario]] | None = None,
) -> np.ndarray | None:
    bits = q3.normalize_strategy(strategy)
    cache = group_cache or build_group_cache(scenarios)
    groups = [cache[idx][group_code(bits, idx)] for idx in range(3)]
    if any(not group.feasible for group in groups):
        return None

    semi_cost = groups[0].cost + groups[1].cost + groups[2].cost
    semi_good_prob = groups[0].good_prob * groups[1].good_prob * groups[2].good_prob

    final_test = bits[14]
    final_disassemble = bits[15]
    p_final = scenarios["final"]
    product_test_cost = final_test * q3.TABLE2.final_test
    exchange_if_defect = 0.0 if final_test else q3.TABLE2.exchange_loss

    if final_disassemble:
        if not (bits[8] and bits[9] and bits[10]):
            return None
        cycle = (
            q3.TABLE2.final_assembly
            + product_test_cost
            + p_final * (exchange_if_defect + q3.TABLE2.final_disassemble)
        ) / (1.0 - p_final)
        cost = semi_cost + cycle
    else:
        success = semi_good_prob * (1.0 - p_final)
        attempt_cost = (
            semi_cost
            + q3.TABLE2.final_assembly
            + product_test_cost
            + (1.0 - success) * exchange_if_defect
        )
        cost = attempt_cost / success

    return q3.TABLE2.sale_price - cost


def q3_fixed_summary(strategy: tuple[int, ...]) -> ScenarioSummary:
    scenarios = {
        "parts": np.array(q3.TABLE2.part_defect, dtype=float).reshape(8, 1),
        "semis": np.array(q3.TABLE2.semi_defect, dtype=float).reshape(3, 1),
        "final": np.array([q3.TABLE2.final_defect], dtype=float),
    }
    return summarize_profit(q3_profit_scenarios(strategy, scenarios))


def deterministic_q3_best(tolerance: float = 1e-9) -> tuple[str, ...]:
    best_profit = -math.inf
    rows: list[tuple[float, str]] = []
    scenarios = {
        "parts": np.array(q3.TABLE2.part_defect, dtype=float).reshape(8, 1),
        "semis": np.array(q3.TABLE2.semi_defect, dtype=float).reshape(3, 1),
        "final": np.array([q3.TABLE2.final_defect], dtype=float),
    }
    group_cache = build_group_cache(scenarios)
    for bits in iter_bits(Q3_STRATEGY_BITS):
        summary = q3_eval_summary(bits, scenarios, group_cache, risk=False)
        if summary.mean > best_profit:
            best_profit = summary.mean
        rows.append((summary.mean, strategy_code(bits)))
    return tuple(code for value, code in rows if abs(value - best_profit) <= tolerance)


def q3_eval_summary(
    strategy: tuple[int, ...],
    scenarios: dict[str, np.ndarray],
    group_cache: list[dict[int, GroupScenario]],
    risk: bool = False,
) -> ScenarioSummary:
    return summarize_profit(q3_profit_scenarios(strategy, scenarios, group_cache), risk=risk)


def q3_enumerate_bayes(
    scenarios: dict[str, np.ndarray],
    group_cache: list[dict[int, GroupScenario]],
    fixed_scenarios: dict[str, np.ndarray],
    fixed_group_cache: list[dict[int, GroupScenario]],
) -> list[dict]:
    rows: list[dict] = []
    p_final = scenarios["final"]
    for code0, group0 in group_cache[0].items():
        for code1, group1 in group_cache[1].items():
            for code2, group2 in group_cache[2].items():
                base_bits = [0] * Q3_STRATEGY_BITS
                for local, part in enumerate(q3.PART_GROUPS[0]):
                    base_bits[part] = (code0 >> local) & 1
                base_bits[8] = (code0 >> 3) & 1
                base_bits[11] = (code0 >> 4) & 1
                for local, part in enumerate(q3.PART_GROUPS[1]):
                    base_bits[part] = (code1 >> local) & 1
                base_bits[9] = (code1 >> 3) & 1
                base_bits[12] = (code1 >> 4) & 1
                for local, part in enumerate(q3.PART_GROUPS[2]):
                    base_bits[part] = (code2 >> local) & 1
                base_bits[10] = (code2 >> 2) & 1
                base_bits[13] = (code2 >> 3) & 1

                groups_feasible = group0.feasible and group1.feasible and group2.feasible
                if groups_feasible:
                    semi_cost = group0.cost + group1.cost + group2.cost
                    semi_good = group0.good_prob * group1.good_prob * group2.good_prob

                for final_test in (0, 1):
                    for final_disassemble in (0, 1):
                        bits = list(base_bits)
                        bits[14] = final_test
                        bits[15] = final_disassemble
                        bits_tuple = tuple(bits)
                        fixed = q3_eval_summary(
                            bits_tuple,
                            fixed_scenarios,
                            fixed_group_cache,
                            risk=False,
                        )

                        profit_values: np.ndarray | None
                        if not groups_feasible:
                            profit_values = None
                        elif final_disassemble and not (bits[8] and bits[9] and bits[10]):
                            profit_values = None
                        elif final_disassemble:
                            product_test_cost = final_test * q3.TABLE2.final_test
                            exchange_if_defect = 0.0 if final_test else q3.TABLE2.exchange_loss
                            cycle = (
                                q3.TABLE2.final_assembly
                                + product_test_cost
                                + p_final
                                * (exchange_if_defect + q3.TABLE2.final_disassemble)
                            ) / (1.0 - p_final)
                            cost = semi_cost + cycle
                            profit_values = q3.TABLE2.sale_price - cost
                        else:
                            product_test_cost = final_test * q3.TABLE2.final_test
                            exchange_if_defect = 0.0 if final_test else q3.TABLE2.exchange_loss
                            success = semi_good * (1.0 - p_final)
                            attempt_cost = (
                                semi_cost
                                + q3.TABLE2.final_assembly
                                + product_test_cost
                                + (1.0 - success) * exchange_if_defect
                            )
                            cost = attempt_cost / success
                            profit_values = q3.TABLE2.sale_price - cost

                        bayes = summarize_profit(profit_values, risk=False)
                        rows.append(
                            {
                                "strategy": strategy_code(bits_tuple),
                                "feasible_fixed": fixed.feasible,
                                "fixed_expected_profit": finite_round(fixed.mean),
                                "feasible_bayes": bayes.feasible,
                                "bayes_expected_profit": finite_round(bayes.mean),
                                "bayes_profit_sd": "",
                                "bayes_profit_q05": "",
                            }
                        )
    return rows


def add_q3_risk_metrics(
    row: dict,
    scenarios: dict[str, np.ndarray],
    group_cache: list[dict[int, GroupScenario]],
) -> dict:
    summary = q3_eval_summary(bits_from_code(row["strategy"]), scenarios, group_cache, risk=True)
    return {
        **row,
        "bayes_profit_sd": finite_round(summary.sd),
        "bayes_profit_q05": finite_round(summary.q05),
    }


def q3_top_rows(
    rows: list[dict],
    scenarios: dict[str, np.ndarray],
    group_cache: list[dict[int, GroupScenario]],
    limit: int = 10,
) -> list[dict]:
    feasible = [row for row in rows if row["feasible_bayes"]]
    ranked = sorted(feasible, key=lambda row: row["bayes_expected_profit"], reverse=True)
    return [
        {"rank": idx + 1, **add_q3_risk_metrics(row, scenarios, group_cache)}
        for idx, row in enumerate(ranked[:limit])
    ]


def random_q3_strategy(rng: random.Random) -> tuple[int, ...]:
    return tuple(rng.randrange(2) for _ in range(Q3_STRATEGY_BITS))


def tournament_select(
    population: list[tuple[int, ...]],
    fitness: dict[tuple[int, ...], float],
    rng: random.Random,
    tournament_size: int = 3,
) -> tuple[int, ...]:
    competitors = rng.sample(population, tournament_size)
    return max(competitors, key=lambda item: fitness[item])


def crossover(
    a: tuple[int, ...],
    b: tuple[int, ...],
    rng: random.Random,
    rate: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if rng.random() >= rate:
        return a, b
    point = rng.randrange(1, Q3_STRATEGY_BITS)
    return a[:point] + b[point:], b[:point] + a[point:]


def mutate(
    strategy: tuple[int, ...],
    rng: random.Random,
    rate: float,
) -> tuple[int, ...]:
    return tuple(1 - bit if rng.random() < rate else bit for bit in strategy)


def q3_run_ga(
    scenarios: dict[str, np.ndarray],
    group_cache: list[dict[int, GroupScenario]],
    shared_cache: dict[tuple[int, ...], ScenarioSummary],
    seed: int,
    population_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    elite_size: int,
) -> dict:
    rng = random.Random(seed)

    def fit(strategy: tuple[int, ...]) -> float:
        if strategy not in shared_cache:
            shared_cache[strategy] = q3_eval_summary(strategy, scenarios, group_cache, risk=False)
        return shared_cache[strategy].mean

    population = [random_q3_strategy(rng) for _ in range(population_size)]
    best_strategy = population[0]
    best_profit = fit(best_strategy)

    for generation in range(generations + 1):
        fitness = {strategy: fit(strategy) for strategy in population}
        ranked = sorted(population, key=lambda item: fitness[item], reverse=True)
        if fitness[ranked[0]] > best_profit:
            best_strategy = ranked[0]
            best_profit = fitness[ranked[0]]
        if generation == generations:
            break

        next_population = ranked[:elite_size]
        while len(next_population) < population_size:
            parent_a = tournament_select(population, fitness, rng)
            parent_b = tournament_select(population, fitness, rng)
            child_a, child_b = crossover(parent_a, parent_b, rng, crossover_rate)
            next_population.append(mutate(child_a, rng, mutation_rate))
            if len(next_population) < population_size:
                next_population.append(mutate(child_b, rng, mutation_rate))
        population = next_population

    summary = shared_cache[best_strategy]
    if math.isnan(summary.sd) or math.isnan(summary.q05):
        summary = q3_eval_summary(best_strategy, scenarios, group_cache, risk=True)
        shared_cache[best_strategy] = summary
    return {
        "seed": seed,
        "best_strategy": strategy_code(best_strategy),
        "best_expected_profit": finite_round(summary.mean),
        "best_profit_sd": finite_round(summary.sd),
        "best_profit_q05": finite_round(summary.q05),
        "evaluated_count_so_far": len(shared_cache),
    }


def decode_q2_strategy(code: str) -> dict[str, str]:
    bits = bits_from_code(code)
    if len(bits) != Q2_STRATEGY_BITS:
        raise ValueError("Q2 strategy must have 7 bits")
    labels = [
        ("part_1_first_inspection", "inspect" if bits[0] else "no_inspection"),
        ("part_2_first_inspection", "inspect" if bits[1] else "no_inspection"),
        ("final_inspection", "inspect" if bits[2] else "no_inspection"),
        ("defective_final_disassemble", "disassemble" if bits[3] else "scrap"),
        ("part_1_recovered_inspection", "inspect" if bits[4] else "no_inspection"),
        ("part_2_recovered_inspection", "inspect" if bits[5] else "no_inspection"),
        ("reassembly_final_inspection", "inspect" if bits[6] else "no_inspection"),
    ]
    return dict(labels)


def build_posterior_specs(n: int, prior: str = "uniform") -> list[dict]:
    return [
        {
            "nominal_rate": rate,
            "n": n,
            "k": posterior_spec(rate, n, prior).k,
            "posterior": f"Beta({posterior_spec(rate, n, prior).alpha:g},{posterior_spec(rate, n, prior).beta:g})",
        }
        for rate in NOMINAL_RATES
    ]


def find_first_stable_n(
    rows: list[dict],
    deterministic_strategies: tuple[str, ...],
    *,
    problem: str,
    case: int | str = "",
) -> int | None:
    selected = [
        row
        for row in rows
        if row["problem"] == problem
        and str(row.get("case", "")) == str(case)
    ]
    selected.sort(key=lambda row: int(row["n"]))
    for idx, row in enumerate(selected):
        if row["best_strategy"] not in deterministic_strategies:
            continue
        if all(item["best_strategy"] in deterministic_strategies for item in selected[idx:]):
            return int(row["n"])
    return None


def build_critical_rows(
    q2_rows: list[dict],
    q3_rows: list[dict],
    q2_fixed_best: dict[int, tuple[str, ...]],
    q3_fixed_best: tuple[str, ...],
) -> list[dict]:
    rows = []
    for case in sorted(q2_fixed_best):
        threshold = find_first_stable_n(
            q2_rows,
            q2_fixed_best[case],
            problem="q2",
            case=case,
        )
        rows.append(
            {
                "problem": "q2",
                "case": case,
                "deterministic_best_strategy": "|".join(q2_fixed_best[case]),
                "first_stable_n": "" if threshold is None else threshold,
            }
        )
    threshold = find_first_stable_n(
        q3_rows,
        q3_fixed_best,
        problem="q3",
        case="",
    )
    rows.append(
        {
            "problem": "q3",
            "case": "",
            "deterministic_best_strategy": "|".join(q3_fixed_best),
            "first_stable_n": "" if threshold is None else threshold,
        }
    )
    return rows


def solve_q4(
    scenarios_count: int,
    sample_sizes: list[int],
    ga_runs: int,
    population_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    elite_size: int,
    output_dir: Path,
    prior: str = "uniform",
) -> dict:
    started = time.perf_counter()
    main_n = sample_sizes[0]
    invalid_n = [n for n in sample_sizes if not is_valid_common_sample_size(n)]
    if invalid_n:
        raise ValueError(f"invalid sample sizes for all nominal rates: {invalid_n}")

    q2_all_rows: list[dict] = []
    q2_best_rows: list[dict] = []
    q2_sensitivity_rows: list[dict] = []
    q3_sensitivity_rows: list[dict] = []
    posterior_rows = build_posterior_statistics_rows(sample_sizes, prior)
    q2_fixed_best = deterministic_q2_best_by_case()
    q3_fixed_best = deterministic_q3_best()

    for n in sample_sizes:
        for params in q2.CASES:
            rows, best = evaluate_q2_case(
                params, n, scenarios_count, BASE_SEED + n, prior
            )
            if n == main_n:
                q2_all_rows.extend(rows)
                q2_best_rows.append(best)
            q2_sensitivity_rows.append(
                {
                    "problem": "q2",
                    "case": params.case,
                    "n": n,
                    "best_strategy": best["bayes_best_strategy"],
                    "best_expected_profit": best["bayes_best_expected_profit"],
                    "profit_sd": best["bayes_profit_sd"],
                    "profit_q05": best["bayes_profit_q05"],
                    "deterministic_best_strategy": "|".join(q2_fixed_best[params.case]),
                    "matches_deterministic_best": int(
                        best["bayes_best_strategy"] in q2_fixed_best[params.case]
                    ),
                    "changed_from_n40": "",
                }
            )

    q3_main_scenarios = q3_scenarios(
        main_n, scenarios_count, BASE_SEED + main_n, prior
    )
    q3_main_cache = build_group_cache(q3_main_scenarios)
    q3_fixed_scenarios = {
        "parts": np.array(q3.TABLE2.part_defect, dtype=float).reshape(8, 1),
        "semis": np.array(q3.TABLE2.semi_defect, dtype=float).reshape(3, 1),
        "final": np.array([q3.TABLE2.final_defect], dtype=float),
    }
    q3_fixed_cache = build_group_cache(q3_fixed_scenarios)
    q3_rows = q3_enumerate_bayes(
        q3_main_scenarios,
        q3_main_cache,
        q3_fixed_scenarios,
        q3_fixed_cache,
    )
    q3_top10 = q3_top_rows(q3_rows, q3_main_scenarios, q3_main_cache, 10)
    exact_best = q3_top10[0]
    all_ga_rows: list[dict] = []
    main_ga_best: dict | None = None
    main_ga_hit_count = 0

    n40_strategy = exact_best["strategy"]
    for n in sample_sizes:
        if n == main_n:
            scenarios = q3_main_scenarios
            group_cache = q3_main_cache
            rows = q3_rows
            best = exact_best
        else:
            scenarios = q3_scenarios(n, scenarios_count, BASE_SEED + n, prior)
            group_cache = build_group_cache(scenarios)
            rows = q3_enumerate_bayes(
                scenarios,
                group_cache,
                q3_fixed_scenarios,
                q3_fixed_cache,
            )
            best = q3_top_rows(rows, scenarios, group_cache, 1)[0]
        shared_cache: dict[tuple[int, ...], ScenarioSummary] = {}
        ga_rows = [
            {
                "n": n,
                **q3_run_ga(
                    scenarios,
                    group_cache,
                    shared_cache,
                    seed=BASE_SEED + n + run,
                    population_size=population_size,
                    generations=generations,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate,
                    elite_size=elite_size,
                ),
            }
            for run in range(ga_runs)
        ]
        all_ga_rows.extend(ga_rows)
        ga_best = max(ga_rows, key=lambda row: row["best_expected_profit"])
        ga_hit_count = sum(
            1
            for row in ga_rows
            if row["best_strategy"] == best["strategy"]
            and abs(row["best_expected_profit"] - best["bayes_expected_profit"]) <= 1e-7
        )
        if n == main_n:
            main_ga_best = ga_best
            main_ga_hit_count = ga_hit_count
        q3_sensitivity_rows.append(
            {
                "problem": "q3",
                "case": "",
                "n": n,
                "ga_best_strategy": ga_best["best_strategy"],
                "exact_best_strategy": best["strategy"],
                "ga_matches_exact": int(ga_best["best_strategy"] == best["strategy"]),
                "ga_hit_count": ga_hit_count,
                "ga_hit_rate": ga_hit_count / ga_runs,
                "best_strategy": best["strategy"],
                "best_expected_profit": best["bayes_expected_profit"],
                "profit_sd": best["bayes_profit_sd"],
                "profit_q05": best["bayes_profit_q05"],
                "deterministic_best_strategy": "|".join(q3_fixed_best),
                "matches_deterministic_best": int(best["strategy"] in q3_fixed_best),
                "changed_from_n40": int(best["strategy"] != n40_strategy),
            }
        )

    q2_n40_best = {
        row["case"]: row["best_strategy"]
        for row in q2_sensitivity_rows
        if row["problem"] == "q2" and row["n"] == main_n
    }
    for row in q2_sensitivity_rows:
        row["changed_from_n40"] = int(row["best_strategy"] != q2_n40_best[row["case"]])

    critical_sample_sizes = sorted(
        {
            n
            for n in list(sample_sizes) + list(CRITICAL_SEARCH_GRID)
            if n >= min(sample_sizes) and is_valid_common_sample_size(n)
        }
    )
    q2_critical_rows: list[dict] = []
    q3_critical_rows: list[dict] = []
    completed_q3_critical = {int(row["n"]) for row in q3_sensitivity_rows}

    for n in critical_sample_sizes:
        if n not in {int(row["n"]) for row in q2_sensitivity_rows}:
            for params in q2.CASES:
                _, best = evaluate_q2_case(
                    params, n, scenarios_count, BASE_SEED + n, prior
                )
                q2_critical_rows.append(
                    {
                        "problem": "q2",
                        "case": params.case,
                        "n": n,
                        "best_strategy": best["bayes_best_strategy"],
                    }
                )

        if n not in completed_q3_critical:
            scenarios = q3_scenarios(n, scenarios_count, BASE_SEED + n, prior)
            group_cache = build_group_cache(scenarios)
            rows = q3_enumerate_bayes(
                scenarios,
                group_cache,
                q3_fixed_scenarios,
                q3_fixed_cache,
            )
            best = q3_top_rows(rows, scenarios, group_cache, 1)[0]
            q3_critical_rows.append(
                {
                    "problem": "q3",
                    "case": "",
                    "n": n,
                    "best_strategy": best["strategy"],
                }
            )

    q2_threshold_input = [
        {
            "problem": row["problem"],
            "case": row["case"],
            "n": row["n"],
            "best_strategy": row["best_strategy"],
        }
        for row in q2_sensitivity_rows
    ] + q2_critical_rows
    q3_threshold_input = [
        {
            "problem": row["problem"],
            "case": row["case"],
            "n": row["n"],
            "best_strategy": row["best_strategy"],
        }
        for row in q3_sensitivity_rows
    ] + q3_critical_rows
    critical_rows = build_critical_rows(
        q2_threshold_input,
        q3_threshold_input,
        q2_fixed_best,
        q3_fixed_best,
    )

    fixed_q3_best = max(
        (row for row in q3_rows if row["feasible_fixed"]),
        key=lambda row: row["fixed_expected_profit"],
    )
    if main_ga_best is None:
        raise RuntimeError("main GA result was not computed")

    summary = {
        "method": "Beta posterior scenarios + analytical production expectation + GA + exact enumeration",
        "posterior_assumption": (
            f"Table rates are independent sample defect rates with {prior} prior; "
            "n is the evidence-strength parameter and n=40 is the representative small-sample scenario."
        ),
        "prior": prior,
        "scenario_count": scenarios_count,
        "main_sample_size": main_n,
        "sample_sizes": sample_sizes,
        "posterior_specs_n40": build_posterior_specs(main_n, prior),
        "posterior_statistics": posterior_rows,
        "q2_7bit_note": "Q2 output enumerates the 7-variable fixed-strategy class (x1,x2,y,z,r1,r2,yr) consistent with the Q2 fixed baseline; the repository's formal Q2 model remains a state-dependent belief MDP.",
        "q2_best": q2_best_rows,
        "q2_critical_sample_sizes": [
            row for row in critical_rows if row["problem"] == "q2"
        ],
        "q3_fixed_best_strategy": fixed_q3_best["strategy"],
        "q3_fixed_best_expected_profit": fixed_q3_best["fixed_expected_profit"],
        "q3_ga_best_strategy": main_ga_best["best_strategy"],
        "q3_ga_best_expected_profit": main_ga_best["best_expected_profit"],
        "q3_exact_best_strategy": exact_best["strategy"],
        "q3_exact_best_expected_profit": exact_best["bayes_expected_profit"],
        "q3_exact_best_profit_sd": exact_best["bayes_profit_sd"],
        "q3_exact_best_profit_q05": exact_best["bayes_profit_q05"],
        "q3_ga_matches_exact": int(main_ga_best["best_strategy"] == exact_best["strategy"]),
        "q3_ga_hit_count": main_ga_hit_count,
        "q3_ga_hit_rate": main_ga_hit_count / ga_runs,
        "q3_decoded_exact_best": q3.decode_strategy(exact_best["strategy"]),
        "q3_sensitivity": q3_sensitivity_rows,
        "q3_critical_sample_size": next(
            (
                row["first_stable_n"]
                for row in critical_rows
                if row["problem"] == "q3"
            ),
            "",
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    write_csv(output_dir / "q4_posterior_stats.csv", posterior_rows)
    write_csv(output_dir / "q4_q2_policy_results.csv", q2_all_rows)
    write_csv(output_dir / "q4_q2_best_policies.csv", q2_best_rows)
    write_csv(output_dir / "q4_q3_ga_runs.csv", all_ga_rows)
    write_csv(output_dir / "q4_q3_exact_all_strategies.csv", q3_rows)
    write_csv(output_dir / "q4_q3_top10_strategies.csv", q3_top10)
    write_csv(output_dir / "q4_sensitivity.csv", q2_sensitivity_rows + q3_sensitivity_rows)
    write_csv(output_dir / "q4_critical_sample_sizes.csv", critical_rows)
    write_json(output_dir / "q4_summary.json", summary)

    return {
        "summary": summary,
        "q2_best": q2_best_rows,
        "q3_top10": q3_top10,
        "sensitivity": q2_sensitivity_rows + q3_sensitivity_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Question 4 Bayesian uncertainty model.")
    parser.add_argument("--scenarios", type=int, default=SCENARIOS)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=list(DEFAULT_SAMPLE_SIZES))
    parser.add_argument("--ga-runs", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.02)
    parser.add_argument("--elite-size", type=int, default=5)
    parser.add_argument(
        "--prior",
        choices=sorted(PRIORS),
        default="uniform",
        help="Beta prior pseudo-counts: uniform Beta(1,1) or Jeffreys Beta(1/2,1/2)",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = (args.output_dir or project_root / "programs" / "results").resolve()
    result = solve_q4(
        scenarios_count=args.scenarios,
        sample_sizes=args.sample_sizes,
        ga_runs=args.ga_runs,
        population_size=args.population_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elite_size=args.elite_size,
        output_dir=output_dir,
        prior=args.prior,
    )

    summary = result["summary"]
    print("Question 4 completed")
    print(f"scenario_count: {summary['scenario_count']}")
    print(f"main_sample_size: {summary['main_sample_size']}")
    print(f"q3_ga_best_strategy: {summary['q3_ga_best_strategy']}")
    print(f"q3_exact_best_strategy: {summary['q3_exact_best_strategy']}")
    print(f"q3_exact_best_expected_profit: {summary['q3_exact_best_expected_profit']}")
    print(f"q3_exact_best_profit_sd: {summary['q3_exact_best_profit_sd']}")
    print(f"q3_exact_best_profit_q05: {summary['q3_exact_best_profit_q05']}")
    print(f"q3_ga_hit_rate: {summary['q3_ga_hit_rate']}")
    print("Q2 best by case:")
    for row in result["q2_best"]:
        print(
            f"case {row['case']}: fixed={row['fixed_best_strategy']} "
            f"bayes={row['bayes_best_strategy']} "
            f"profit={row['bayes_best_expected_profit']}"
        )


if __name__ == "__main__":
    main()
