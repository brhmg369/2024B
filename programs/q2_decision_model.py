"""Question 2 belief-state MDP model.

The second question is modeled as a Markov decision process on the firm's
information state.  A state is a belief distribution over the actual qualities
of the two currently held parts.  This avoids the earlier shortcut of assigning
a standalone recovered-part value: recovered parts are either inspected, or they
go directly back into assembly.

Cost unit: expected total cost to finally deliver one qualified product.
Profit unit: sale price minus expected cost.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import argparse
import csv
import json
import math


NONE = "N"
GOOD = "G"
BAD = "B"

QUALITIES = (NONE, GOOD, BAD)
COMBOS = tuple(product(QUALITIES, repeat=2))
COMBO_INDEX = {combo: i for i, combo in enumerate(COMBOS)}

ROUND_DIGITS = 6
PROB_TOL = 1e-12
VALUE_TOL = 1e-10
MAX_ITERATIONS = 20000
MAX_STATES = 100000


@dataclass(frozen=True)
class CaseParams:
    case: int
    p1: float
    c1: float
    t1: float
    p2: float
    c2: float
    t2: float
    pf: float
    ca: float
    tf: float
    sale: float
    exchange_loss: float
    disassemble_cost: float


@dataclass(frozen=True)
class Action:
    name: str
    kind: str
    priority: int
    part: int | None = None
    inspect_part: int | None = None
    product_test: int | None = None
    disassemble: int | None = None


@dataclass(frozen=True)
class ActionEval:
    cost: float
    terminal_prob: float
    transitions: tuple[tuple[float, tuple[float, ...]], ...]


CASES = [
    CaseParams(1, 0.10, 4, 2, 0.10, 18, 3, 0.10, 6, 3, 56, 6, 5),
    CaseParams(2, 0.20, 4, 2, 0.20, 18, 3, 0.20, 6, 3, 56, 6, 5),
    CaseParams(3, 0.10, 4, 2, 0.10, 18, 3, 0.10, 6, 3, 56, 30, 5),
    CaseParams(4, 0.20, 4, 1, 0.20, 18, 1, 0.20, 6, 2, 56, 30, 5),
    CaseParams(5, 0.10, 4, 8, 0.20, 18, 1, 0.10, 6, 2, 56, 10, 5),
    CaseParams(6, 0.05, 4, 2, 0.05, 18, 3, 0.05, 6, 3, 56, 10, 40),
]


START_STATE = tuple(
    1.0 if combo == (NONE, NONE) else 0.0
    for combo in COMBOS
)


def part_params(params: CaseParams, part: int) -> tuple[float, float, float]:
    if part == 1:
        return params.p1, params.c1, params.t1
    return params.p2, params.c2, params.t2


def canonicalize(prob_by_combo: dict[tuple[str, str], float]) -> tuple[float, ...]:
    values = [max(0.0, prob_by_combo.get(combo, 0.0)) for combo in COMBOS]
    total = sum(values)
    if total <= PROB_TOL:
        raise ValueError("empty belief state")

    values = [value / total for value in values]
    rounded = [0.0 if value < PROB_TOL else round(value, ROUND_DIGITS) for value in values]
    rounded_total = sum(rounded)

    if rounded_total <= PROB_TOL:
        max_index = max(range(len(values)), key=lambda i: values[i])
        rounded = [0.0 for _ in values]
        rounded[max_index] = 1.0
        return tuple(rounded)

    residual = round(1.0 - rounded_total, ROUND_DIGITS)
    if abs(residual) > 0:
        max_index = max(range(len(rounded)), key=lambda i: rounded[i])
        rounded[max_index] = round(rounded[max_index] + residual, ROUND_DIGITS)

    rounded = [0.0 if abs(value) < PROB_TOL else value for value in rounded]
    return tuple(rounded)


def iter_positive(state: tuple[float, ...]):
    for combo, prob in zip(COMBOS, state):
        if prob > PROB_TOL:
            yield combo, prob


def set_part(combo: tuple[str, str], part: int, quality: str) -> tuple[str, str]:
    if part == 1:
        return quality, combo[1]
    return combo[0], quality


def marginal_prob(state: tuple[float, ...], part: int, quality: str) -> float:
    idx = part - 1
    return sum(prob for combo, prob in iter_positive(state) if combo[idx] == quality)


def part_missing(state: tuple[float, ...], part: int) -> bool:
    return marginal_prob(state, part, NONE) >= 1.0 - 1e-9


def part_present(state: tuple[float, ...], part: int) -> bool:
    return marginal_prob(state, part, NONE) <= 1e-9


def both_parts_present(state: tuple[float, ...]) -> bool:
    return part_present(state, 1) and part_present(state, 2)


def condition_state(
    state: tuple[float, ...],
    predicate,
    transform=lambda combo: combo,
) -> tuple[float, tuple[float, ...] | None]:
    event_prob = 0.0
    out: dict[tuple[str, str], float] = {}

    for combo, prob in iter_positive(state):
        if not predicate(combo):
            continue
        event_prob += prob
        next_combo = transform(combo)
        out[next_combo] = out.get(next_combo, 0.0) + prob

    if event_prob <= PROB_TOL:
        return 0.0, None
    return event_prob, canonicalize(out)


def buy_transition(
    state: tuple[float, ...],
    params: CaseParams,
    part: int,
    inspect: int,
) -> ActionEval:
    p, price, test_cost = part_params(params, part)

    if inspect:
        cost = (price + test_cost) / (1.0 - p)
        out = {set_part(combo, part, GOOD): prob for combo, prob in iter_positive(state)}
        return ActionEval(cost, 0.0, ((1.0, canonicalize(out)),))

    out: dict[tuple[str, str], float] = {}
    for combo, prob in iter_positive(state):
        good_combo = set_part(combo, part, GOOD)
        bad_combo = set_part(combo, part, BAD)
        out[good_combo] = out.get(good_combo, 0.0) + prob * (1.0 - p)
        out[bad_combo] = out.get(bad_combo, 0.0) + prob * p
    return ActionEval(price, 0.0, ((1.0, canonicalize(out)),))


def inspect_transition(
    state: tuple[float, ...],
    params: CaseParams,
    part: int,
) -> ActionEval:
    _, _, test_cost = part_params(params, part)
    idx = part - 1
    transitions: list[tuple[float, tuple[float, ...]]] = []

    good_prob, good_state = condition_state(
        state,
        lambda combo: combo[idx] == GOOD,
    )
    if good_state is not None:
        transitions.append((good_prob, good_state))

    bad_prob, bad_state = condition_state(
        state,
        lambda combo: combo[idx] == BAD,
        lambda combo: set_part(combo, part, NONE),
    )
    if bad_state is not None:
        transitions.append((bad_prob, bad_state))

    return ActionEval(test_cost, 0.0, tuple(transitions))


def defect_probability_for_combo(combo: tuple[str, str], params: CaseParams) -> float:
    if combo == (GOOD, GOOD):
        return params.pf
    return 1.0


def assemble_transition(
    state: tuple[float, ...],
    params: CaseParams,
    product_test: int,
    disassemble: int,
) -> ActionEval:
    success_prob = state[COMBO_INDEX[(GOOD, GOOD)]] * (1.0 - params.pf)
    defect_prob = 1.0 - success_prob
    extra_if_defect = (0.0 if product_test else params.exchange_loss)
    if disassemble:
        extra_if_defect += params.disassemble_cost

    cost = params.ca + product_test * params.tf + defect_prob * extra_if_defect

    if defect_prob <= PROB_TOL:
        return ActionEval(cost, 1.0, tuple())

    if not disassemble:
        return ActionEval(cost, success_prob, ((defect_prob, START_STATE),))

    posterior: dict[tuple[str, str], float] = {}
    for combo, prob in iter_positive(state):
        combo_defect_prob = defect_probability_for_combo(combo, params)
        weight = prob * combo_defect_prob
        if weight <= PROB_TOL:
            continue
        posterior[combo] = posterior.get(combo, 0.0) + weight

    return ActionEval(cost, success_prob, ((defect_prob, canonicalize(posterior)),))


def available_actions(state: tuple[float, ...]) -> list[Action]:
    actions: list[Action] = []

    for part in (1, 2):
        if part_missing(state, part):
            actions.append(
                Action(
                    name=f"buy_p{part}_test",
                    kind="buy",
                    priority=10 + part,
                    part=part,
                    inspect_part=1,
                )
            )
            actions.append(
                Action(
                    name=f"buy_p{part}_notest",
                    kind="buy",
                    priority=20 + part,
                    part=part,
                    inspect_part=0,
                )
            )

    for part in (1, 2):
        if part_present(state, part) and marginal_prob(state, part, BAD) > PROB_TOL:
            actions.append(
                Action(
                    name=f"inspect_p{part}",
                    kind="inspect",
                    priority=30 + part,
                    part=part,
                )
            )

    if both_parts_present(state):
        actions.extend(
            [
                Action(
                    name="assemble_notest_scrap",
                    kind="assemble",
                    priority=40,
                    product_test=0,
                    disassemble=0,
                ),
                Action(
                    name="assemble_notest_disassemble",
                    kind="assemble",
                    priority=41,
                    product_test=0,
                    disassemble=1,
                ),
                Action(
                    name="assemble_test_scrap",
                    kind="assemble",
                    priority=42,
                    product_test=1,
                    disassemble=0,
                ),
                Action(
                    name="assemble_test_disassemble",
                    kind="assemble",
                    priority=43,
                    product_test=1,
                    disassemble=1,
                ),
            ]
        )

    return actions


def evaluate_action(
    state: tuple[float, ...],
    action: Action,
    params: CaseParams,
) -> ActionEval:
    if action.kind == "buy":
        assert action.part is not None and action.inspect_part is not None
        return buy_transition(state, params, action.part, action.inspect_part)

    if action.kind == "inspect":
        assert action.part is not None
        return inspect_transition(state, params, action.part)

    if action.kind == "assemble":
        assert action.product_test is not None and action.disassemble is not None
        return assemble_transition(state, params, action.product_test, action.disassemble)

    raise ValueError(f"unknown action kind: {action.kind}")


def discover_states(params: CaseParams) -> list[tuple[float, ...]]:
    seen = {START_STATE}
    ordered = [START_STATE]
    queue: deque[tuple[float, ...]] = deque([START_STATE])

    while queue:
        state = queue.popleft()
        for action in available_actions(state):
            action_eval = evaluate_action(state, action, params)
            for prob, next_state in action_eval.transitions:
                if prob <= PROB_TOL or next_state in seen:
                    continue
                if len(ordered) >= MAX_STATES:
                    raise RuntimeError(
                        f"state limit exceeded for case {params.case}; "
                        f"lower ROUND_DIGITS or raise MAX_STATES"
                    )
                seen.add(next_state)
                ordered.append(next_state)
                queue.append(next_state)

    return ordered


def q_value(
    action_eval: ActionEval,
    values: dict[tuple[float, ...], float],
) -> float:
    return action_eval.cost + sum(
        prob * values[next_state]
        for prob, next_state in action_eval.transitions
    )


def select_best_action(
    state: tuple[float, ...],
    params: CaseParams,
    values: dict[tuple[float, ...], float],
) -> tuple[Action, ActionEval, float]:
    candidates = []
    for action in available_actions(state):
        action_eval = evaluate_action(state, action, params)
        candidates.append((action, action_eval, q_value(action_eval, values)))

    best_value = min(value for _, _, value in candidates)
    tied = [
        item for item in candidates
        if item[2] <= best_value + 1e-9
    ]
    return min(tied, key=lambda item: item[0].priority)


def validate_case_params(params: CaseParams) -> None:
    for name in ("p1", "p2", "pf"):
        value = getattr(params, name)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")
    for name in (
        "c1",
        "t1",
        "c2",
        "t2",
        "ca",
        "tf",
        "sale",
        "exchange_loss",
        "disassemble_cost",
    ):
        if getattr(params, name) < 0.0:
            raise ValueError(f"{name} must be non-negative")


def bellman_residual(
    states: list[tuple[float, ...]],
    params: CaseParams,
    values: dict[tuple[float, ...], float],
) -> float:
    return max(
        abs(select_best_action(state, params, values)[2] - values[state])
        for state in states
    )


def solve_case(params: CaseParams) -> dict:
    validate_case_params(params)
    states = discover_states(params)
    values = {state: 0.0 for state in states}
    delta = math.inf
    iterations = 0
    residual_history: list[float] = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        delta = 0.0
        for state in states:
            _, _, best_value = select_best_action(state, params, values)
            delta = max(delta, abs(best_value - values[state]))
            values[state] = best_value
        residual_history.append(delta)
        iterations = iteration
        if delta < VALUE_TOL:
            break

    converged = delta < VALUE_TOL
    policy: dict[tuple[float, ...], tuple[Action, ActionEval, float]] = {}
    for state in states:
        policy[state] = select_best_action(state, params, values)

    return {
        "params": params,
        "states": states,
        "values": values,
        "policy": policy,
        "iterations": iterations,
        "delta": delta,
        "bellman_residual": bellman_residual(states, params, values),
        "residual_history": residual_history,
        "converged": converged,
    }


def state_label(state: tuple[float, ...]) -> str:
    parts = [
        f"{combo[0]}{combo[1]}:{prob:.6g}"
        for combo, prob in iter_positive(state)
    ]
    return ";".join(parts)


def state_features(state: tuple[float, ...]) -> dict[str, float]:
    p_gg = state[COMBO_INDEX[(GOOD, GOOD)]]
    return {
        "p_NN": state[COMBO_INDEX[(NONE, NONE)]],
        "p_NG": state[COMBO_INDEX[(NONE, GOOD)]],
        "p_NB": state[COMBO_INDEX[(NONE, BAD)]],
        "p_GN": state[COMBO_INDEX[(GOOD, NONE)]],
        "p_GG": p_gg,
        "p_GB": state[COMBO_INDEX[(GOOD, BAD)]],
        "p_BN": state[COMBO_INDEX[(BAD, NONE)]],
        "p_BG": state[COMBO_INDEX[(BAD, GOOD)]],
        "p_BB": state[COMBO_INDEX[(BAD, BAD)]],
        "part1_missing_prob": marginal_prob(state, 1, NONE),
        "part1_good_prob": marginal_prob(state, 1, GOOD),
        "part1_bad_prob": marginal_prob(state, 1, BAD),
        "part2_missing_prob": marginal_prob(state, 2, NONE),
        "part2_good_prob": marginal_prob(state, 2, GOOD),
        "part2_bad_prob": marginal_prob(state, 2, BAD),
    }


def defect_prob_if_assemble(state: tuple[float, ...], params: CaseParams) -> str:
    if not both_parts_present(state):
        return ""
    return f"{1.0 - state[COMBO_INDEX[(GOOD, GOOD)]] * (1.0 - params.pf):.10f}"


def is_self_loop_if_repeated(
    state: tuple[float, ...],
    action_eval: ActionEval,
) -> int:
    if action_eval.terminal_prob > PROB_TOL:
        return 0
    if len(action_eval.transitions) != 1:
        return 0
    prob, next_state = action_eval.transitions[0]
    return int(prob >= 1.0 - 1e-9 and next_state == state)


def rounded(value: float | str, digits: int = 6) -> float | str:
    if isinstance(value, str):
        return value
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return round(value, digits)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_state_policy_rows(solution: dict) -> list[dict]:
    params: CaseParams = solution["params"]
    states = solution["states"]
    values = solution["values"]
    policy = solution["policy"]
    state_index = {state: i for i, state in enumerate(states)}

    rows = []
    for state in states:
        action, action_eval, best_value = policy[state]
        row = {
            "case": params.case,
            "state_id": state_index[state],
            "state": state_label(state),
            **{key: rounded(value, 10) for key, value in state_features(state).items()},
            "defect_prob_if_assemble": defect_prob_if_assemble(state, params),
            "best_action": action.name,
            "best_action_kind": action.kind,
            "best_action_part": "" if action.part is None else action.part,
            "best_product_test": "" if action.product_test is None else action.product_test,
            "best_disassemble": "" if action.disassemble is None else action.disassemble,
            "one_step_cost": rounded(action_eval.cost, 6),
            "terminal_prob": rounded(action_eval.terminal_prob, 10),
            "continuation_prob": rounded(
                sum(prob for prob, _ in action_eval.transitions),
                10,
            ),
            "self_loop_if_repeated": is_self_loop_if_repeated(state, action_eval),
            "value": rounded(best_value, 6),
        }
        rows.append(row)
    return rows


def build_action_value_rows(solution: dict) -> list[dict]:
    params: CaseParams = solution["params"]
    states = solution["states"]
    values = solution["values"]
    policy = solution["policy"]
    state_index = {state: i for i, state in enumerate(states)}

    rows = []
    for state in states:
        best_action, _, best_value = policy[state]
        for action in available_actions(state):
            action_eval = evaluate_action(state, action, params)
            value = q_value(action_eval, values)
            row = {
                "case": params.case,
                "state_id": state_index[state],
                "state": state_label(state),
                **{key: rounded(feature, 10) for key, feature in state_features(state).items()},
                "action": action.name,
                "action_kind": action.kind,
                "part": "" if action.part is None else action.part,
                "part_inspection": "" if action.inspect_part is None else action.inspect_part,
                "product_test": "" if action.product_test is None else action.product_test,
                "disassemble": "" if action.disassemble is None else action.disassemble,
                "one_step_cost": rounded(action_eval.cost, 6),
                "terminal_prob": rounded(action_eval.terminal_prob, 10),
                "continuation_prob": rounded(
                    sum(prob for prob, _ in action_eval.transitions),
                    10,
                ),
                "self_loop_if_repeated": is_self_loop_if_repeated(state, action_eval),
                "q_value": rounded(value, 6),
                "is_optimal": int(abs(value - best_value) <= 1e-7),
                "chosen_by_tiebreak": int(action.name == best_action.name),
            }
            rows.append(row)
    return rows


def state_after_buying_both_without_tests(params: CaseParams) -> tuple[float, ...]:
    first = buy_transition(START_STATE, params, 1, inspect=0).transitions[0][1]
    second = buy_transition(first, params, 2, inspect=0).transitions[0][1]
    return second


def state_both_known_good() -> tuple[float, ...]:
    return canonicalize({(GOOD, GOOD): 1.0})


def state_after_defect_from_unknown_both(params: CaseParams) -> tuple[float, ...] | None:
    both_unknown = state_after_buying_both_without_tests(params)
    action_eval = assemble_transition(both_unknown, params, product_test=1, disassemble=1)
    if not action_eval.transitions:
        return None
    return action_eval.transitions[0][1]


def lookup_action(
    solution: dict,
    state: tuple[float, ...] | None,
) -> str:
    if state is None:
        return ""
    item = solution["policy"].get(state)
    if item is None:
        return ""
    return item[0].name


def trace_initial_policy(solution: dict) -> dict[str, str]:
    """Trace the on-policy path through purchases to the first assembly."""

    state = START_STATE
    component_actions: list[str] = []
    first_assembly_action = ""
    after_first_defect_action = ""

    for _ in range(12):
        action, action_eval, _ = solution["policy"][state]
        if action.kind == "assemble":
            first_assembly_action = action.name
            if action_eval.transitions:
                defect_state = action_eval.transitions[0][1]
                after_first_defect_action = lookup_action(solution, defect_state)
            else:
                after_first_defect_action = "terminal"
            break

        component_actions.append(action.name)
        if len(action_eval.transitions) != 1:
            after_first_defect_action = "branch_before_assembly"
            break
        state = action_eval.transitions[0][1]

    return {
        "initial_component_action_1": (
            component_actions[0] if component_actions else ""
        ),
        "initial_component_action_2": (
            component_actions[1] if len(component_actions) > 1 else ""
        ),
        "first_assembly_action": first_assembly_action,
        "after_first_defect_action": after_first_defect_action,
    }


def build_best_rows(solutions: list[dict]) -> list[dict]:
    rows = []
    for solution in solutions:
        params: CaseParams = solution["params"]
        start_action, _, start_value = solution["policy"][START_STATE]
        both_unknown = state_after_buying_both_without_tests(params)
        known_good = state_both_known_good()
        defect_unknown = state_after_defect_from_unknown_both(params)
        traced = trace_initial_policy(solution)

        rows.append(
            {
                "case": params.case,
                "num_states": len(solution["states"]),
                "iterations": solution["iterations"],
                "bellman_delta": f"{solution['delta']:.3e}",
                "bellman_residual": f"{solution['bellman_residual']:.3e}",
                "converged": int(solution["converged"]),
                **traced,
                "start_action": start_action.name,
                "both_new_uninspected_state_action": lookup_action(solution, both_unknown),
                "both_known_good_state_action": lookup_action(solution, known_good),
                "after_defect_from_uninspected_parts_action": lookup_action(solution, defect_unknown),
                "expected_cost": rounded(start_value, 6),
                "expected_profit": rounded(params.sale - start_value, 6),
            }
        )
    return rows


def build_convergence_rows(solutions: list[dict]) -> list[dict]:
    rows = []
    for solution in solutions:
        params: CaseParams = solution["params"]
        rows.extend(
            {
                "case": params.case,
                "iteration": iteration,
                "update_delta": f"{delta:.12e}",
            }
            for iteration, delta in enumerate(
                solution["residual_history"],
                start=1,
            )
        )
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="2024 年国赛 B 题问题二：联合信念状态 MDP。"
    )
    parser.add_argument("--output-dir", type=Path, help="CSV/JSON 输出目录。")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    out_dir = (args.output_dir or project_root / "programs" / "results").resolve()
    solutions = [solve_case(params) for params in CASES]

    state_policy_rows: list[dict] = []
    action_value_rows: list[dict] = []
    for solution in solutions:
        state_policy_rows.extend(build_state_policy_rows(solution))
        action_value_rows.extend(build_action_value_rows(solution))

    best_rows = build_best_rows(solutions)
    convergence_rows = build_convergence_rows(solutions)

    write_csv(out_dir / "q2_policy_results.csv", action_value_rows)
    write_csv(out_dir / "q2_state_policy.csv", state_policy_rows)
    write_csv(out_dir / "q2_best_policies.csv", best_rows)
    write_csv(out_dir / "q2_convergence.csv", convergence_rows)
    write_json(
        out_dir / "q2_summary.json",
        {
            "problem": "2024 年国赛 B 题问题二",
            "method": "联合信念状态马尔可夫决策过程与 Gauss-Seidel 值迭代",
            "solver": {
                "belief_round_digits": ROUND_DIGITS,
                "probability_tolerance": PROB_TOL,
                "value_tolerance": VALUE_TOL,
                "maximum_iterations": MAX_ITERATIONS,
            },
            "cases": best_rows,
            "checks": {
                "all_converged": all(solution["converged"] for solution in solutions),
                "maximum_bellman_residual": max(
                    solution["bellman_residual"] for solution in solutions
                ),
            },
            "outputs": {
                "best_policies": "programs/results/q2_best_policies.csv",
                "state_policy": "programs/results/q2_state_policy.csv",
                "action_values": "programs/results/q2_policy_results.csv",
                "convergence": "programs/results/q2_convergence.csv",
            },
        },
    )

    print("Best MDP policies by case:")
    for row in best_rows:
        print(
            f"case {row['case']}: cost={float(row['expected_cost']):.4f}, "
            f"profit={float(row['expected_profit']):.4f}, "
            f"start={row['start_action']}, "
            f"both_unknown={row['both_new_uninspected_state_action']}, "
            f"known_good={row['both_known_good_state_action']}"
        )


if __name__ == "__main__":
    main()
