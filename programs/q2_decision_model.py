"""Question 2 decision model with recovered-part inspection modes.

Part policy modes:
0. first_inspect: inspect the part before the first assembly; after disassembly
   this part is known qualified, so it is not inspected again.
1. never_inspect: do not inspect before assembly, and do not inspect after
   disassembly.
2. inspect_after_recovery: do not inspect before assembly, but inspect this
   recovered part after disassembly.

Workflow used for the handoff:
1. Enumerate 3 x 3 part policies.
2. Compute the initial finished-product defect probability q.
3. Decide finished-product inspection by the local rule t_f < q L.
4. Enumerate disassembly z = 0, 1.
5. Compute total expected cost and expected profit by a finite-state recursion.

Cost unit: expected total cost to finally deliver one qualified finished product.
Profit unit: sale price minus expected cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import csv
import math


NONE = "none"
KNOWN_GOOD = "known_good"
RECOVERED_GOOD = "recovered_good"
RECOVERED_BAD = "recovered_bad"

PART_MODES = {
    0: "first_inspect",
    1: "never_inspect",
    2: "inspect_after_recovery",
}

STATUSES = [NONE, KNOWN_GOOD, RECOVERED_GOOD, RECOVERED_BAD]
STATES = [(s1, s2) for s1 in STATUSES for s2 in STATUSES]
STATE_INDEX = {state: i for i, state in enumerate(STATES)}


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


CASES = [
    CaseParams(1, 0.10, 4, 2, 0.10, 18, 3, 0.10, 6, 3, 56, 6, 5),
    CaseParams(2, 0.20, 4, 2, 0.20, 18, 3, 0.20, 6, 3, 56, 6, 5),
    CaseParams(3, 0.10, 4, 2, 0.10, 18, 3, 0.10, 6, 3, 56, 30, 5),
    CaseParams(4, 0.20, 4, 1, 0.20, 18, 1, 0.20, 6, 2, 56, 30, 5),
    CaseParams(5, 0.10, 4, 8, 0.20, 18, 1, 0.10, 6, 2, 56, 10, 5),
    CaseParams(6, 0.05, 4, 2, 0.05, 18, 3, 0.05, 6, 3, 56, 10, 40),
]


def solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve Ax=b with Gaussian elimination and partial pivoting."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("singular system")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]

        pivot_value = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= pivot_value

        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if abs(factor) < 1e-15:
                continue
            for j in range(col, n + 1):
                a[row][j] -= factor * a[col][j]

    return [a[i][n] for i in range(n)]


def part_params(params: CaseParams, part: int) -> tuple[float, float, float]:
    if part == 1:
        return params.p1, params.c1, params.t1
    return params.p2, params.c2, params.t2


def expected_inspected_new_cost(price: float, test_cost: float, defect_rate: float) -> float:
    return (price + test_cost) / (1 - defect_rate)


def acquire_new_options(params: CaseParams, part: int, mode: int):
    """Return options: probability, cost, actual_good, known_good."""
    p, c, t = part_params(params, part)
    if mode == 0:
        return [(1.0, expected_inspected_new_cost(c, t, p), True, True)]
    return [
        (1 - p, c, True, False),
        (p, c, False, False),
    ]


def use_component_options(params: CaseParams, part: int, mode: int, status: str):
    """Return options: probability, cost, actual_good, known_good."""
    p, _, t = part_params(params, part)

    if status == NONE:
        return acquire_new_options(params, part, mode)

    if status == KNOWN_GOOD:
        return [(1.0, 0.0, True, True)]

    if status == RECOVERED_GOOD:
        if mode == 2:
            return [(1.0, t, True, True)]
        return [(1.0, 0.0, True, False)]

    if status == RECOVERED_BAD:
        if mode == 2:
            # Pay recovery inspection, discard the bad part, then acquire a new
            # part according to the initial mode. Mode 2 means new parts are not
            # inspected before assembly.
            return [
                (prob, t + cost, good, known)
                for prob, cost, good, known in acquire_new_options(params, part, mode)
            ]
        return [(1.0, 0.0, False, False)]

    raise ValueError(f"unknown status: {status}")


def recovered_status(actual_good: bool, known_good: bool) -> str:
    if actual_good and known_good:
        return KNOWN_GOOD
    if actual_good:
        return RECOVERED_GOOD
    return RECOVERED_BAD


def initial_quality(params: CaseParams, mode1: int, mode2: int) -> tuple[float, float, float]:
    u1 = 1.0 if mode1 == 0 else 1 - params.p1
    u2 = 1.0 if mode2 == 0 else 1 - params.p2
    g = u1 * u2 * (1 - params.pf)
    return u1, u2, g


def state_terms(
    params: CaseParams,
    mode1: int,
    mode2: int,
    y: int,
    z: int,
    state: tuple[str, str],
) -> tuple[float, dict[tuple[str, str], float]]:
    """Return one state's constant cost and transition probabilities."""
    constant = 0.0
    transitions: dict[tuple[str, str], float] = {}
    status1, status2 = state

    for prob1, cost1, good1, known1 in use_component_options(params, 1, mode1, status1):
        for prob2, cost2, good2, known2 in use_component_options(params, 2, mode2, status2):
            base_prob = prob1 * prob2
            base_cost = cost1 + cost2 + params.ca + y * params.tf

            if good1 and good2:
                outcomes = [(1 - params.pf, True), (params.pf, False)]
            else:
                outcomes = [(1.0, False)]

            for quality_prob, product_good in outcomes:
                prob = base_prob * quality_prob
                if prob == 0:
                    continue

                constant += prob * base_cost
                if product_good:
                    continue

                extra = 0.0 if y else params.exchange_loss
                if z:
                    extra += params.disassemble_cost
                    next_state = (
                        recovered_status(good1, known1),
                        recovered_status(good2, known2),
                    )
                else:
                    next_state = (NONE, NONE)

                constant += prob * extra
                transitions[next_state] = transitions.get(next_state, 0.0) + prob

    return constant, transitions


def reachable_states(params: CaseParams, mode1: int, mode2: int, y: int, z: int):
    start = (NONE, NONE)
    seen = {start}
    ordered = [start]
    cursor = 0

    while cursor < len(ordered):
        state = ordered[cursor]
        cursor += 1
        _, transitions = state_terms(params, mode1, mode2, y, z, state)
        for next_state, prob in transitions.items():
            if prob <= 0 or next_state in seen:
                continue
            seen.add(next_state)
            ordered.append(next_state)

    return ordered


def build_equations(params: CaseParams, mode1: int, mode2: int, y: int, z: int):
    states = reachable_states(params, mode1, mode2, y, z)
    index = {state: i for i, state in enumerate(states)}
    n = len(states)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    rhs = [0.0 for _ in range(n)]

    for i in range(n):
        matrix[i][i] = 1.0

    for state in states:
        row = index[state]
        constant, transitions = state_terms(params, mode1, mode2, y, z, state)
        rhs[row] = constant
        for next_state, prob in transitions.items():
            matrix[row][index[next_state]] -= prob

    return matrix, rhs, states


def evaluate_strategy(params: CaseParams, mode1: int, mode2: int, z: int) -> dict:
    _, _, g_initial = initial_quality(params, mode1, mode2)
    q_initial = 1 - g_initial

    y = 1 if params.tf < q_initial * params.exchange_loss else 0
    product_inspection_margin = q_initial * params.exchange_loss - params.tf

    matrix, rhs, states = build_equations(params, mode1, mode2, y, z)
    try:
        values = solve_linear_system(matrix, rhs)
        expected_cost = values[states.index((NONE, NONE))]
        expected_profit = params.sale - expected_cost
        feasible = 1
    except ValueError:
        expected_cost = math.inf
        expected_profit = -math.inf
        feasible = 0

    return {
        "case": params.case,
        "part1_mode": PART_MODES[mode1],
        "part2_mode": PART_MODES[mode2],
        "part1_mode_id": mode1,
        "part2_mode_id": mode2,
        "product_defect_prob_q": round(q_initial, 6),
        "qL_minus_tf": round(product_inspection_margin, 6),
        "inspect_product_by_rule": y,
        "disassemble": z,
        "feasible": feasible,
        "expected_cost": "inf" if not feasible else round(expected_cost, 6),
        "expected_profit": "-inf" if not feasible else round(expected_profit, 6),
    }


def evaluate_all() -> list[dict]:
    rows = []
    for params in CASES:
        for mode1, mode2 in product([0, 1, 2], repeat=2):
            for z in [0, 1]:
                rows.append(evaluate_strategy(params, mode1, mode2, z))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_key(row: dict):
    profit = -math.inf if row["expected_profit"] == "-inf" else float(row["expected_profit"])
    return (row["case"], row["feasible"] == 0, -profit)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    rows = evaluate_all()
    rows.sort(key=sort_key)

    best_rows = []
    for case_id in sorted({row["case"] for row in rows}):
        case_rows = [
            row for row in rows
            if row["case"] == case_id and row["feasible"]
        ]
        best_rows.append(max(case_rows, key=lambda row: float(row["expected_profit"])))

    write_csv(out_dir / "q2_policy_results.csv", rows)
    write_csv(out_dir / "q2_best_policies.csv", best_rows)

    print("Best policies by case:")
    for row in best_rows:
        print(
            f"case {row['case']}: "
            f"part1={row['part1_mode']}, part2={row['part2_mode']}, "
            f"y={row['inspect_product_by_rule']}, z={row['disassemble']}, "
            f"q={row['product_defect_prob_q']:.4f}, "
            f"cost={float(row['expected_cost']):.4f}, "
            f"profit={float(row['expected_profit']):.4f}"
        )


if __name__ == "__main__":
    main()
