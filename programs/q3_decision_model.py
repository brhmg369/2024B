"""Question 3 analytical expectation, GA search, and full enumeration.

The model evaluates a fixed 16-bit production strategy with a finite Markov
reward process.  State variables preserve the actual quality of recovered
parts and semi-finished products; disassembly never re-samples defect rates.

Objective: maximize expected profit for finally delivering one qualified
finished product.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import argparse
import csv
import math
import random
import time

import numpy as np


NONE = 0
KNOWN_GOOD = 1
UNKNOWN_GOOD = 2
UNKNOWN_BAD = 3

STATUS_LABELS = {
    NONE: "N",
    KNOWN_GOOD: "KG",
    UNKNOWN_GOOD: "UG",
    UNKNOWN_BAD: "UB",
}

PART_COUNT = 8
SEMI_COUNT = 3
STATE_SIZE = PART_COUNT + SEMI_COUNT
SEMI_OFFSET = PART_COUNT

PART_GROUPS = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7),
)

STRATEGY_BITS = 16
INF_COST = 1e100


@dataclass(frozen=True)
class Q3Params:
    part_defect: tuple[float, ...]
    part_price: tuple[float, ...]
    part_test: tuple[float, ...]
    semi_defect: tuple[float, ...]
    semi_assembly: tuple[float, ...]
    semi_test: tuple[float, ...]
    semi_disassemble: tuple[float, ...]
    final_defect: float
    final_assembly: float
    final_test: float
    final_disassemble: float
    sale_price: float
    exchange_loss: float


TABLE2 = Q3Params(
    part_defect=(0.10,) * 8,
    part_price=(2, 8, 12, 2, 8, 12, 8, 12),
    part_test=(1, 1, 2, 1, 1, 2, 1, 2),
    semi_defect=(0.10, 0.10, 0.10),
    semi_assembly=(8, 8, 8),
    semi_test=(4, 4, 4),
    semi_disassemble=(6, 6, 6),
    final_defect=0.10,
    final_assembly=8,
    final_test=6,
    final_disassemble=10,
    sale_price=200,
    exchange_loss=40,
)


START_STATE = (NONE,) * STATE_SIZE


@dataclass(frozen=True)
class Step:
    cost: float
    transitions: tuple[tuple[float, tuple[int, ...]], ...]
    terminal_prob: float = 0.0
    operation: str = ""


@dataclass(frozen=True)
class PolicyEvaluation:
    strategy: tuple[int, ...]
    feasible: int
    expected_cost: float
    expected_profit: float
    num_states: int
    infeasible_reason: str = ""


def normalize_strategy(strategy: tuple[int, ...] | list[int] | str) -> tuple[int, ...]:
    if isinstance(strategy, str):
        bits = tuple(int(ch) for ch in strategy.strip())
    else:
        bits = tuple(int(x) for x in strategy)
    if len(bits) != STRATEGY_BITS or any(bit not in (0, 1) for bit in bits):
        raise ValueError("strategy must contain exactly 16 binary values")
    return bits


def strategy_to_code(strategy: tuple[int, ...] | list[int] | str) -> str:
    return "".join(str(bit) for bit in normalize_strategy(strategy))


def part_test_flag(strategy: tuple[int, ...], part: int) -> int:
    return strategy[part]


def semi_test_flag(strategy: tuple[int, ...], semi: int) -> int:
    return strategy[8 + semi]


def semi_disassemble_flag(strategy: tuple[int, ...], semi: int) -> int:
    return strategy[11 + semi]


def final_test_flag(strategy: tuple[int, ...]) -> int:
    return strategy[14]


def final_disassemble_flag(strategy: tuple[int, ...]) -> int:
    return strategy[15]


def actual_good(status: int) -> bool:
    return status in (KNOWN_GOOD, UNKNOWN_GOOD)


def set_status(state: tuple[int, ...], index: int, status: int) -> tuple[int, ...]:
    updated = list(state)
    updated[index] = status
    return tuple(updated)


def set_many(state: tuple[int, ...], pairs: list[tuple[int, int]]) -> tuple[int, ...]:
    updated = list(state)
    for index, status in pairs:
        updated[index] = status
    return tuple(updated)


def reset_all() -> tuple[int, ...]:
    return START_STATE


def reset_semi_and_children(state: tuple[int, ...], semi: int) -> tuple[int, ...]:
    pairs = [(SEMI_OFFSET + semi, NONE)]
    pairs.extend((part, NONE) for part in PART_GROUPS[semi])
    return set_many(state, pairs)


def semi_children_ready(state: tuple[int, ...], semi: int) -> bool:
    return all(state[part] != NONE for part in PART_GROUPS[semi])


def aggregate_transitions(
    cost: float,
    transitions: list[tuple[float, tuple[int, ...]]],
    terminal_prob: float = 0.0,
    operation: str = "",
) -> Step:
    by_state: dict[tuple[int, ...], float] = {}
    for prob, next_state in transitions:
        if prob <= 0:
            continue
        by_state[next_state] = by_state.get(next_state, 0.0) + prob
    return Step(
        cost=cost,
        transitions=tuple((prob, state) for state, prob in by_state.items()),
        terminal_prob=terminal_prob,
        operation=operation,
    )


def acquire_part_step(
    state: tuple[int, ...],
    strategy: tuple[int, ...],
    params: Q3Params,
    part: int,
) -> Step:
    p = params.part_defect[part]
    price = params.part_price[part]
    test = params.part_test[part]

    if part_test_flag(strategy, part):
        # Repeated purchase and inspection until a qualified part is obtained.
        cost = (price + test) / (1.0 - p)
        next_state = set_status(state, part, KNOWN_GOOD)
        return aggregate_transitions(cost, [(1.0, next_state)], operation=f"buy_test_part_{part + 1}")

    good_state = set_status(state, part, UNKNOWN_GOOD)
    bad_state = set_status(state, part, UNKNOWN_BAD)
    return aggregate_transitions(
        price,
        [(1.0 - p, good_state), (p, bad_state)],
        operation=f"buy_no_test_part_{part + 1}",
    )


def inspect_part_step(
    state: tuple[int, ...],
    params: Q3Params,
    part: int,
) -> Step:
    status = state[part]
    cost = params.part_test[part]
    if status == UNKNOWN_GOOD:
        next_state = set_status(state, part, KNOWN_GOOD)
    elif status == UNKNOWN_BAD:
        next_state = set_status(state, part, NONE)
    else:
        raise ValueError("inspect_part_step called for a non-unknown part")
    return aggregate_transitions(cost, [(1.0, next_state)], operation=f"inspect_part_{part + 1}")


def assemble_semi_step(
    state: tuple[int, ...],
    params: Q3Params,
    semi: int,
) -> Step:
    children = PART_GROUPS[semi]
    cost = params.semi_assembly[semi]
    semi_index = SEMI_OFFSET + semi

    if all(actual_good(state[part]) for part in children):
        good_state = set_status(state, semi_index, UNKNOWN_GOOD)
        bad_state = set_status(state, semi_index, UNKNOWN_BAD)
        p = params.semi_defect[semi]
        transitions = [(1.0 - p, good_state), (p, bad_state)]
    else:
        transitions = [(1.0, set_status(state, semi_index, UNKNOWN_BAD))]

    return aggregate_transitions(cost, transitions, operation=f"assemble_semi_{semi + 1}")


def inspect_semi_step(
    state: tuple[int, ...],
    strategy: tuple[int, ...],
    params: Q3Params,
    semi: int,
) -> Step:
    semi_index = SEMI_OFFSET + semi
    status = state[semi_index]
    cost = params.semi_test[semi]

    if status == UNKNOWN_GOOD:
        next_state = set_status(state, semi_index, KNOWN_GOOD)
        return aggregate_transitions(cost, [(1.0, next_state)], operation=f"inspect_semi_{semi + 1}_good")

    if status != UNKNOWN_BAD:
        raise ValueError("inspect_semi_step called for a non-unknown semi")

    if semi_disassemble_flag(strategy, semi):
        cost += params.semi_disassemble[semi]
        next_state = set_status(state, semi_index, NONE)
        operation = f"inspect_semi_{semi + 1}_bad_disassemble"
    else:
        next_state = reset_semi_and_children(state, semi)
        operation = f"inspect_semi_{semi + 1}_bad_scrap"

    return aggregate_transitions(cost, [(1.0, next_state)], operation=operation)


def assemble_final_step(
    state: tuple[int, ...],
    strategy: tuple[int, ...],
    params: Q3Params,
) -> Step:
    semi_statuses = state[SEMI_OFFSET:SEMI_OFFSET + SEMI_COUNT]
    all_good = all(actual_good(status) for status in semi_statuses)

    base_cost = params.final_assembly + final_test_flag(strategy) * params.final_test
    bad_extra = 0.0 if final_test_flag(strategy) else params.exchange_loss
    if final_disassemble_flag(strategy):
        bad_extra += params.final_disassemble
        bad_state = state
        operation = "assemble_final_bad_disassemble"
    else:
        bad_state = reset_all()
        operation = "assemble_final_bad_scrap"

    if all_good:
        p_bad = params.final_defect
        cost = base_cost + p_bad * bad_extra
        return aggregate_transitions(
            cost,
            [(p_bad, bad_state)],
            terminal_prob=1.0 - p_bad,
            operation=operation,
        )

    cost = base_cost + bad_extra
    return aggregate_transitions(
        cost,
        [(1.0, bad_state)],
        terminal_prob=0.0,
        operation=operation,
    )


def policy_step(
    state: tuple[int, ...],
    strategy: tuple[int, ...],
    params: Q3Params = TABLE2,
) -> Step:
    """Return the next Markov reward step under a fixed 16-bit strategy."""
    for semi in range(SEMI_COUNT):
        semi_index = SEMI_OFFSET + semi
        semi_status = state[semi_index]

        if semi_status != NONE:
            if semi_test_flag(strategy, semi) and semi_status in (UNKNOWN_GOOD, UNKNOWN_BAD):
                return inspect_semi_step(state, strategy, params, semi)
            continue

        for part in PART_GROUPS[semi]:
            part_status = state[part]
            if part_status == NONE:
                return acquire_part_step(state, strategy, params, part)
            if part_test_flag(strategy, part) and part_status in (UNKNOWN_GOOD, UNKNOWN_BAD):
                return inspect_part_step(state, params, part)

        if semi_children_ready(state, semi):
            return assemble_semi_step(state, params, semi)

    return assemble_final_step(state, strategy, params)


def discover_states(
    strategy: tuple[int, ...],
    params: Q3Params = TABLE2,
    max_states: int = 100000,
) -> list[tuple[int, ...]]:
    states = [START_STATE]
    seen = {START_STATE}
    cursor = 0
    while cursor < len(states):
        state = states[cursor]
        cursor += 1
        step = policy_step(state, strategy, params)
        for prob, next_state in step.transitions:
            if prob <= 0 or next_state in seen:
                continue
            if len(states) >= max_states:
                raise RuntimeError("reachable state limit exceeded")
            seen.add(next_state)
            states.append(next_state)
    return states


def evaluate_strategy(
    strategy: tuple[int, ...] | list[int] | str,
    params: Q3Params = TABLE2,
) -> PolicyEvaluation:
    bits = normalize_strategy(strategy)

    try:
        states = discover_states(bits, params)
        index = {state: i for i, state in enumerate(states)}
        n = len(states)
        matrix = np.eye(n, dtype=float)
        rhs = np.zeros(n, dtype=float)

        for state in states:
            row = index[state]
            step = policy_step(state, bits, params)
            rhs[row] = step.cost
            for prob, next_state in step.transitions:
                matrix[row, index[next_state]] -= prob

        values = np.linalg.solve(matrix, rhs)
        cost = float(values[index[START_STATE]])

        if not math.isfinite(cost) or cost < -1e-7 or cost > INF_COST / 10:
            raise np.linalg.LinAlgError("non-finite expected cost")

        return PolicyEvaluation(
            strategy=bits,
            feasible=1,
            expected_cost=cost,
            expected_profit=params.sale_price - cost,
            num_states=n,
        )

    except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
        return PolicyEvaluation(
            strategy=bits,
            feasible=0,
            expected_cost=math.inf,
            expected_profit=-math.inf,
            num_states=0,
            infeasible_reason=str(exc) or "singular Markov reward equations",
        )


def random_strategy(rng: random.Random) -> tuple[int, ...]:
    return tuple(rng.randrange(2) for _ in range(STRATEGY_BITS))


def tournament_select(
    population: list[tuple[int, ...]],
    fitness: dict[tuple[int, ...], float],
    rng: random.Random,
    tournament_size: int = 3,
) -> tuple[int, ...]:
    competitors = rng.sample(population, tournament_size)
    return max(competitors, key=lambda item: fitness[item])


def crossover(
    parent_a: tuple[int, ...],
    parent_b: tuple[int, ...],
    rng: random.Random,
    crossover_rate: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if rng.random() >= crossover_rate:
        return parent_a, parent_b
    point = rng.randrange(1, STRATEGY_BITS)
    child_a = parent_a[:point] + parent_b[point:]
    child_b = parent_b[:point] + parent_a[point:]
    return child_a, child_b


def mutate(
    strategy: tuple[int, ...],
    rng: random.Random,
    mutation_rate: float,
) -> tuple[int, ...]:
    return tuple(1 - bit if rng.random() < mutation_rate else bit for bit in strategy)


def run_ga(
    params: Q3Params = TABLE2,
    seed: int = 2024,
    population_size: int = 100,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.02,
    elite_size: int = 5,
) -> dict:
    rng = random.Random(seed)
    cache: dict[tuple[int, ...], PolicyEvaluation] = {}

    def fit(strategy: tuple[int, ...]) -> float:
        if strategy not in cache:
            cache[strategy] = evaluate_strategy(strategy, params)
        return cache[strategy].expected_profit

    population = [random_strategy(rng) for _ in range(population_size)]
    best_strategy = population[0]
    best_profit = fit(best_strategy)
    history = []

    for generation in range(generations + 1):
        fitness = {strategy: fit(strategy) for strategy in population}
        ranked = sorted(population, key=lambda item: fitness[item], reverse=True)

        if fitness[ranked[0]] > best_profit:
            best_strategy = ranked[0]
            best_profit = fitness[ranked[0]]

        history.append(
            {
                "generation": generation,
                "best_profit": best_profit,
                "current_best_profit": fitness[ranked[0]],
                "unique_evaluated": len(cache),
            }
        )

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

    best_eval = cache.get(best_strategy) or evaluate_strategy(best_strategy, params)
    return {
        "seed": seed,
        "best_strategy": best_strategy,
        "best_profit": best_eval.expected_profit,
        "best_cost": best_eval.expected_cost,
        "best_feasible": best_eval.feasible,
        "best_num_states": best_eval.num_states,
        "history": history,
        "evaluated_count": len(cache),
    }


def enumerate_all_strategies(params: Q3Params = TABLE2) -> list[PolicyEvaluation]:
    rows: list[PolicyEvaluation] = []
    for bits in product((0, 1), repeat=STRATEGY_BITS):
        rows.append(evaluate_strategy(bits, params))
    return rows


def top_feasible(evaluations: list[PolicyEvaluation], n: int = 10) -> list[PolicyEvaluation]:
    feasible = [row for row in evaluations if row.feasible]
    return sorted(feasible, key=lambda row: row.expected_profit, reverse=True)[:n]


def decode_strategy(strategy: tuple[int, ...] | list[int] | str) -> dict[str, str]:
    bits = normalize_strategy(strategy)
    decoded: dict[str, str] = {}
    for part in range(PART_COUNT):
        decoded[f"part_{part + 1}_inspection"] = "inspect" if bits[part] else "no_inspection"
    for semi in range(SEMI_COUNT):
        decoded[f"semi_{semi + 1}_inspection"] = "inspect" if bits[8 + semi] else "no_inspection"
    for semi in range(SEMI_COUNT):
        decoded[f"defective_semi_{semi + 1}"] = "disassemble" if bits[11 + semi] else "scrap"
    decoded["finished_product_inspection"] = "inspect" if bits[14] else "no_inspection"
    decoded["defective_finished_product"] = "disassemble" if bits[15] else "scrap"
    return decoded


def state_to_label(state: tuple[int, ...]) -> str:
    part_labels = ",".join(STATUS_LABELS[state[i]] for i in range(PART_COUNT))
    semi_labels = ",".join(STATUS_LABELS[state[SEMI_OFFSET + j]] for j in range(SEMI_COUNT))
    return f"parts=[{part_labels}];semis=[{semi_labels}]"


def monte_carlo_check(
    strategy: tuple[int, ...] | list[int] | str,
    params: Q3Params = TABLE2,
    trials: int = 5000,
    seed: int = 2024,
    max_steps: int = 10000,
) -> dict:
    bits = normalize_strategy(strategy)
    rng = random.Random(seed)
    costs = []
    failures = 0

    for _ in range(trials):
        state = list(START_STATE)
        total_cost = 0.0
        delivered = False

        for _step in range(max_steps):
            progressed = False

            for semi in range(SEMI_COUNT):
                semi_index = SEMI_OFFSET + semi
                semi_status = state[semi_index]

                if semi_status != NONE:
                    if semi_test_flag(bits, semi) and semi_status in (UNKNOWN_GOOD, UNKNOWN_BAD):
                        total_cost += params.semi_test[semi]
                        if semi_status == UNKNOWN_GOOD:
                            state[semi_index] = KNOWN_GOOD
                        elif semi_disassemble_flag(bits, semi):
                            total_cost += params.semi_disassemble[semi]
                            state[semi_index] = NONE
                        else:
                            state[semi_index] = NONE
                            for part in PART_GROUPS[semi]:
                                state[part] = NONE
                        progressed = True
                    if progressed:
                        break
                    continue

                for part in PART_GROUPS[semi]:
                    part_status = state[part]
                    if part_status == NONE:
                        if part_test_flag(bits, part):
                            while True:
                                total_cost += params.part_price[part] + params.part_test[part]
                                if rng.random() >= params.part_defect[part]:
                                    state[part] = KNOWN_GOOD
                                    break
                        else:
                            total_cost += params.part_price[part]
                            state[part] = (
                                UNKNOWN_GOOD
                                if rng.random() >= params.part_defect[part]
                                else UNKNOWN_BAD
                            )
                        progressed = True
                        break

                    if part_test_flag(bits, part) and part_status in (UNKNOWN_GOOD, UNKNOWN_BAD):
                        total_cost += params.part_test[part]
                        state[part] = KNOWN_GOOD if part_status == UNKNOWN_GOOD else NONE
                        progressed = True
                        break

                if progressed:
                    break

                if all(state[part] != NONE for part in PART_GROUPS[semi]):
                    total_cost += params.semi_assembly[semi]
                    if all(actual_good(state[part]) for part in PART_GROUPS[semi]):
                        state[semi_index] = (
                            UNKNOWN_GOOD
                            if rng.random() >= params.semi_defect[semi]
                            else UNKNOWN_BAD
                        )
                    else:
                        state[semi_index] = UNKNOWN_BAD
                    progressed = True
                    break

            if progressed:
                continue

            total_cost += params.final_assembly + final_test_flag(bits) * params.final_test
            all_semis_good = all(
                actual_good(state[SEMI_OFFSET + semi])
                for semi in range(SEMI_COUNT)
            )
            final_good = all_semis_good and rng.random() >= params.final_defect

            if final_good:
                delivered = True
                break

            if not final_test_flag(bits):
                total_cost += params.exchange_loss
            if final_disassemble_flag(bits):
                total_cost += params.final_disassemble
            else:
                state = list(START_STATE)

        if delivered:
            costs.append(total_cost)
        else:
            failures += 1

    mean_cost = math.inf if not costs else sum(costs) / len(costs)
    return {
        "trials": trials,
        "completed_trials": len(costs),
        "failed_trials": failures,
        "mc_expected_cost": mean_cost,
        "mc_expected_profit": params.sale_price - mean_cost if costs else -math.inf,
    }


def evaluation_to_row(evaluation: PolicyEvaluation, rank: int | None = None) -> dict:
    row = {
        "strategy": strategy_to_code(evaluation.strategy),
        "feasible": evaluation.feasible,
        "expected_cost": "inf" if not evaluation.feasible else round(evaluation.expected_cost, 6),
        "expected_profit": "-inf" if not evaluation.feasible else round(evaluation.expected_profit, 6),
        "num_states": evaluation.num_states,
        "infeasible_reason": evaluation.infeasible_reason,
    }
    if rank is not None:
        row = {"rank": rank, **row}
    return row


def ga_run_to_row(run: dict, exact_best: PolicyEvaluation | None = None) -> dict:
    row = {
        "seed": run["seed"],
        "best_strategy": strategy_to_code(run["best_strategy"]),
        "best_cost": round(run["best_cost"], 6),
        "best_profit": round(run["best_profit"], 6),
        "best_feasible": run["best_feasible"],
        "best_num_states": run["best_num_states"],
        "evaluated_count": run["evaluated_count"],
    }
    if exact_best is not None:
        row["hit_global_best"] = int(
            run["best_strategy"] == exact_best.strategy
            and abs(run["best_profit"] - exact_best.expected_profit) <= 1e-7
        )
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


def run_problem3(
    ga_runs: int = 20,
    enumerate_exact: bool = True,
    population_size: int = 100,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.02,
    elite_size: int = 5,
) -> dict:
    started = time.perf_counter()
    out_dir = Path(__file__).resolve().parent / "results"

    ga_results = [
        run_ga(
            seed=2024 + run,
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
        )
        for run in range(ga_runs)
    ]
    ga_best = max(ga_results, key=lambda run: run["best_profit"])

    exact_rows: list[PolicyEvaluation] = []
    exact_best: PolicyEvaluation | None = None
    top10: list[PolicyEvaluation] = []
    if enumerate_exact:
        exact_rows = enumerate_all_strategies(TABLE2)
        top10 = top_feasible(exact_rows, 10)
        exact_best = top10[0]

    write_csv(
        out_dir / "q3_ga_runs.csv",
        [ga_run_to_row(run, exact_best) for run in ga_results],
    )

    if exact_rows:
        write_csv(
            out_dir / "q3_exact_all_strategies.csv",
            [evaluation_to_row(row) for row in exact_rows],
        )
        write_csv(
            out_dir / "q3_top10_strategies.csv",
            [evaluation_to_row(row, rank=i + 1) for i, row in enumerate(top10)],
        )

    exact_hit_count = ""
    exact_hit_rate = ""
    if exact_best is not None:
        exact_hit_count = sum(
            1
            for run in ga_results
            if run["best_strategy"] == exact_best.strategy
            and abs(run["best_profit"] - exact_best.expected_profit) <= 1e-7
        )
        exact_hit_rate = exact_hit_count / ga_runs

    mc_target = exact_best.strategy if exact_best is not None else ga_best["best_strategy"]
    analytic_target = evaluate_strategy(mc_target)
    mc = monte_carlo_check(mc_target, trials=20000, seed=777)

    summary = {
        "ga_runs": ga_runs,
        "ga_best_strategy": strategy_to_code(ga_best["best_strategy"]),
        "ga_best_expected_cost": round(ga_best["best_cost"], 6),
        "ga_best_expected_profit": round(ga_best["best_profit"], 6),
        "exact_enumeration": int(enumerate_exact),
        "exact_best_strategy": "" if exact_best is None else strategy_to_code(exact_best.strategy),
        "exact_best_expected_cost": "" if exact_best is None else round(exact_best.expected_cost, 6),
        "exact_best_expected_profit": "" if exact_best is None else round(exact_best.expected_profit, 6),
        "ga_matches_exact_best": "" if exact_best is None else int(ga_best["best_strategy"] == exact_best.strategy),
        "ga_global_hit_count": exact_hit_count,
        "ga_global_hit_rate": exact_hit_rate,
        "monte_carlo_checked_strategy": strategy_to_code(mc_target),
        "analytic_cost_for_mc_strategy": round(analytic_target.expected_cost, 6),
        "mc_expected_cost": round(mc["mc_expected_cost"], 6),
        "mc_completed_trials": mc["completed_trials"],
        "mc_failed_trials": mc["failed_trials"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_summary(out_dir / "q3_summary.txt", summary)

    return {
        "ga_results": ga_results,
        "ga_best": ga_best,
        "exact_rows": exact_rows,
        "exact_best": exact_best,
        "top10": top10,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Question 3 with analytical MRP, GA, and enumeration.")
    parser.add_argument("--ga-runs", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.02)
    parser.add_argument("--elite-size", type=int, default=5)
    parser.add_argument("--skip-enumeration", action="store_true")
    args = parser.parse_args()

    result = run_problem3(
        ga_runs=args.ga_runs,
        enumerate_exact=not args.skip_enumeration,
        population_size=args.population_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elite_size=args.elite_size,
    )

    summary = result["summary"]
    print("Question 3 completed")
    for key, value in summary.items():
        print(f"{key}: {value}")

    exact_best = result["exact_best"]
    if exact_best is not None:
        print("\nTop 10 exact strategies:")
        for rank, row in enumerate(result["top10"], start=1):
            print(
                f"{rank:2d}. {strategy_to_code(row.strategy)} "
                f"profit={row.expected_profit:.6f} cost={row.expected_cost:.6f}"
            )


if __name__ == "__main__":
    main()
