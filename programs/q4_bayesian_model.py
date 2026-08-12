"""Question 4 Bayesian defect-rate uncertainty model.

This module reuses the production logic of Q2/Q3 without changing their source
files.  Defect rates are treated as Beta posterior random variables generated
from the modelling assumption n in {40, 100, 200} and k = n * p_hat.

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
Q2_STRATEGY_BITS = 4
Q3_STRATEGY_BITS = 16


@dataclass(frozen=True)
class PosteriorSpec:
    nominal_rate: float
    n: int
    k: int
    alpha: int
    beta: int


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


def posterior_spec(rate: float, n: int) -> PosteriorSpec:
    raw_k = n * rate
    k = int(round(raw_k))
    if abs(raw_k - k) > 1e-9:
        raise ValueError(f"n={n} cannot represent rate={rate:g} with integer k")
    return PosteriorSpec(rate, n, k, k + 1, n - k + 1)


def sample_beta(rate: float, n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    spec = posterior_spec(rate, n)
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
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
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
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + params.case * 97)
    return {
        "p1": sample_beta(params.p1, n, size, rng),
        "p2": sample_beta(params.p2, n, size, rng),
        "pf": sample_beta(params.pf, n, size, rng),
    }


def q2_profit_scenarios(
    strategy: tuple[int, ...],
    params: q2.CaseParams,
    scenarios: dict[str, np.ndarray],
) -> np.ndarray | None:
    if len(strategy) != Q2_STRATEGY_BITS:
        raise ValueError("Q2 strategy must have 4 bits")

    x1, x2, y, z = strategy
    p1 = scenarios["p1"]
    p2 = scenarios["p2"]
    pf = scenarios["pf"]

    part1_cost = (params.c1 + params.t1) / (1.0 - p1) if x1 else params.c1
    part2_cost = (params.c2 + params.t2) / (1.0 - p2) if x2 else params.c2
    q1 = np.ones_like(p1) if x1 else (1.0 - p1)
    q2_good = np.ones_like(p2) if x2 else (1.0 - p2)

    if z and (not x1 or not x2):
        return None

    base = part1_cost + part2_cost
    product_test_cost = y * params.tf
    exchange_if_defect = 0.0 if y else params.exchange_loss

    if z:
        cycle_cost = (
            params.ca
            + product_test_cost
            + pf * (exchange_if_defect + params.disassemble_cost)
        ) / (1.0 - pf)
        cost = base + cycle_cost
    else:
        success = q1 * q2_good * (1.0 - pf)
        attempt_cost = (
            base
            + params.ca
            + product_test_cost
            + (1.0 - success) * exchange_if_defect
        )
        cost = attempt_cost / success

    return params.sale - cost


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


def evaluate_q2_case(
    params: q2.CaseParams,
    n: int,
    size: int,
    seed: int,
) -> tuple[list[dict], dict]:
    scenarios = q2_case_scenarios(params, n, size, seed)
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
        "strategy_changed": int(fixed_best["strategy"] != bayes_best["strategy"]),
    }
    return rows, best


def q3_scenarios(n: int, size: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 3009)
    return {
        "parts": np.vstack(
            [sample_beta(rate, n, size, rng) for rate in q3.TABLE2.part_defect]
        ),
        "semis": np.vstack(
            [sample_beta(rate, n, size, rng) for rate in q3.TABLE2.semi_defect]
        ),
        "final": sample_beta(q3.TABLE2.final_defect, n, size, rng),
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
    labels = [
        ("part_1_inspection", "inspect" if bits[0] else "no_inspection"),
        ("part_2_inspection", "inspect" if bits[1] else "no_inspection"),
        ("finished_product_inspection", "inspect" if bits[2] else "no_inspection"),
        ("defective_finished_product", "disassemble" if bits[3] else "scrap"),
    ]
    return dict(labels)


def build_posterior_specs(n: int) -> list[dict]:
    return [
        {
            "nominal_rate": rate,
            "n": n,
            "k": posterior_spec(rate, n).k,
            "posterior": f"Beta({posterior_spec(rate, n).alpha},{posterior_spec(rate, n).beta})",
        }
        for rate in (0.05, 0.10, 0.20)
    ]


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
) -> dict:
    started = time.perf_counter()
    main_n = sample_sizes[0]

    q2_all_rows: list[dict] = []
    q2_best_rows: list[dict] = []
    q2_sensitivity_rows: list[dict] = []

    for n in sample_sizes:
        for params in q2.CASES:
            rows, best = evaluate_q2_case(params, n, scenarios_count, BASE_SEED + n)
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
                    "changed_from_n40": "",
                }
            )

    q3_main_scenarios = q3_scenarios(main_n, scenarios_count, BASE_SEED + main_n)
    q3_main_cache = build_group_cache(q3_main_scenarios)
    q3_fixed_scenarios = {
        "parts": np.array(q3.TABLE2.part_defect, dtype=float).reshape(8, 1),
        "semis": np.array(q3.TABLE2.semi_defect, dtype=float).reshape(3, 1),
        "final": np.array([q3.TABLE2.final_defect], dtype=float),
    }
    q3_fixed_cache = build_group_cache(q3_fixed_scenarios)
    shared_cache: dict[tuple[int, ...], ScenarioSummary] = {}
    ga_rows = [
        q3_run_ga(
            q3_main_scenarios,
            q3_main_cache,
            shared_cache,
            seed=BASE_SEED + run,
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
        )
        for run in range(ga_runs)
    ]

    q3_rows = q3_enumerate_bayes(
        q3_main_scenarios,
        q3_main_cache,
        q3_fixed_scenarios,
        q3_fixed_cache,
    )
    q3_top10 = q3_top_rows(q3_rows, q3_main_scenarios, q3_main_cache, 10)
    exact_best = q3_top10[0]
    ga_best = max(ga_rows, key=lambda row: row["best_expected_profit"])
    ga_hit_count = sum(
        1
        for row in ga_rows
        if row["best_strategy"] == exact_best["strategy"]
        and abs(row["best_expected_profit"] - exact_best["bayes_expected_profit"]) <= 1e-7
    )

    q3_sensitivity_rows: list[dict] = []
    n40_strategy = exact_best["strategy"]
    for n in sample_sizes:
        if n == main_n:
            best = exact_best
        else:
            scenarios = q3_scenarios(n, scenarios_count, BASE_SEED + n)
            group_cache = build_group_cache(scenarios)
            rows = q3_enumerate_bayes(
                scenarios,
                group_cache,
                q3_fixed_scenarios,
                q3_fixed_cache,
            )
            best = q3_top_rows(rows, scenarios, group_cache, 1)[0]
        q3_sensitivity_rows.append(
            {
                "problem": "q3",
                "case": "",
                "n": n,
                "best_strategy": best["strategy"],
                "best_expected_profit": best["bayes_expected_profit"],
                "profit_sd": best["bayes_profit_sd"],
                "profit_q05": best["bayes_profit_q05"],
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

    fixed_q3_best = max(
        (row for row in q3_rows if row["feasible_fixed"]),
        key=lambda row: row["fixed_expected_profit"],
    )

    summary = {
        "method": "Beta posterior scenarios + analytical production expectation + GA + exact enumeration",
        "posterior_assumption": "Table rates are independent sample defect rates with uniform prior; n=40 is the main modelling assumption.",
        "scenario_count": scenarios_count,
        "main_sample_size": main_n,
        "posterior_specs_n40": build_posterior_specs(main_n),
        "q2_fixed16_note": "Q2 output enumerates the requested fixed 4-bit restricted strategies; the repository's formal Q2 model remains a state-dependent belief MDP.",
        "q2_best": q2_best_rows,
        "q3_fixed_best_strategy": fixed_q3_best["strategy"],
        "q3_fixed_best_expected_profit": fixed_q3_best["fixed_expected_profit"],
        "q3_ga_best_strategy": ga_best["best_strategy"],
        "q3_ga_best_expected_profit": ga_best["best_expected_profit"],
        "q3_exact_best_strategy": exact_best["strategy"],
        "q3_exact_best_expected_profit": exact_best["bayes_expected_profit"],
        "q3_exact_best_profit_sd": exact_best["bayes_profit_sd"],
        "q3_exact_best_profit_q05": exact_best["bayes_profit_q05"],
        "q3_ga_matches_exact": int(ga_best["best_strategy"] == exact_best["strategy"]),
        "q3_ga_hit_count": ga_hit_count,
        "q3_ga_hit_rate": ga_hit_count / ga_runs,
        "q3_decoded_exact_best": q3.decode_strategy(exact_best["strategy"]),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    write_csv(output_dir / "q4_q2_policy_results.csv", q2_all_rows)
    write_csv(output_dir / "q4_q2_best_policies.csv", q2_best_rows)
    write_csv(output_dir / "q4_q3_ga_runs.csv", ga_rows)
    write_csv(output_dir / "q4_q3_exact_all_strategies.csv", q3_rows)
    write_csv(output_dir / "q4_q3_top10_strategies.csv", q3_top10)
    write_csv(output_dir / "q4_sensitivity.csv", q2_sensitivity_rows + q3_sensitivity_rows)
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
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[40, 100, 200])
    parser.add_argument("--ga-runs", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.02)
    parser.add_argument("--elite-size", type=int, default=5)
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
