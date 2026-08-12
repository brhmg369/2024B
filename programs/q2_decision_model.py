"""Question 2 Markov expected-cost model.

Decision variables:
    x1, x2: inspect part 1/2 before the first assembly.
    y: inspect the first finished product.
    z: disassemble defective finished products.
    r1, r2: inspect recovered part 1/2 after disassembly.
    yr: inspect finished products assembled after disassembly.

The model deliberately does not use a standalone recovered-part value V. In the
problem statement, only defective finished products can be scrapped or
disassembled. Recovered parts must repeat the part-inspection and assembly
steps: inspect and discard only detected defective parts, or do not inspect and
send the part directly to assembly.

Cost unit: expected total cost to finally deliver one qualified finished
product. Profit unit: sale price minus expected cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import csv
import math


NONE = "none"
KNOWN_GOOD = "known_good"
UNKNOWN_GOOD = "unknown_good"
UNKNOWN_BAD = "unknown_bad"

STATUSES = [NONE, KNOWN_GOOD, UNKNOWN_GOOD, UNKNOWN_BAD]
PHASE_INITIAL = 0
PHASE_RECOVERY = 1
PHASES = [PHASE_INITIAL, PHASE_RECOVERY]
STATES = [
    (s1, s2, phase)
    for s1 in STATUSES
    for s2 in STATUSES
    for phase in PHASES
]


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
class Policy:
    x1: int
    x2: int
    y: int
    z: int
    r1: int
    r2: int
    yr: int

    def key(self) -> tuple[int, int, int, int, int, int, int]:
        return self.x1, self.x2, self.y, self.z, self.r1, self.r2, self.yr


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


def first_inspection(policy: Policy, part: int) -> int:
    return policy.x1 if part == 1 else policy.x2


def recovered_inspection(policy: Policy, part: int) -> int:
    return policy.r1 if part == 1 else policy.r2


def expected_inspected_new_cost(price: float, test_cost: float, defect_rate: float) -> float:
    return (price + test_cost) / (1 - defect_rate)


def acquire_new_part_options(params: CaseParams, policy: Policy, part: int):
    """Return options for a missing part: probability, cost, actual_good, known_good.

    A newly purchased part is governed by x_i. If x_i=1, buying and inspection
    are repeated until a qualified part is obtained.
    """
    p, c, t = part_params(params, part)
    if first_inspection(policy, part):
        return [(1.0, expected_inspected_new_cost(c, t, p), True, True)]
    return [
        (1 - p, c, True, False),
        (p, c, False, False),
    ]


def part_use_options(params: CaseParams, policy: Policy, part: int, status: str):
    """Return options when a part enters the current assembly attempt.

    Options are probability, cost, actual_good, known_good_before_assembly.
    """
    _, _, test_cost = part_params(params, part)

    if status == NONE:
        return acquire_new_part_options(params, policy, part)

    if status == KNOWN_GOOD:
        return [(1.0, 0.0, True, True)]

    if status == UNKNOWN_GOOD:
        if recovered_inspection(policy, part):
            return [(1.0, test_cost, True, True)]
        return [(1.0, 0.0, True, False)]

    if status == UNKNOWN_BAD:
        if recovered_inspection(policy, part):
            # Only detected defective parts can be discarded. After discarding,
            # the process returns to step (1) for this part and obtains a new
            # part according to the first-stage inspection rule x_i.
            return [
                (prob, test_cost + cost, actual_good, known)
                for prob, cost, actual_good, known
                in acquire_new_part_options(params, policy, part)
            ]
        return [(1.0, 0.0, False, False)]

    raise ValueError(f"unknown part status: {status}")


def next_recovered_status(actual_good: bool, known_good: bool) -> str:
    if actual_good and known_good:
        return KNOWN_GOOD
    if actual_good:
        return UNKNOWN_GOOD
    return UNKNOWN_BAD


def initial_defect_probability(params: CaseParams, policy: Policy) -> float:
    u1 = 1.0 if policy.x1 else 1 - params.p1
    u2 = 1.0 if policy.x2 else 1 - params.p2
    return 1 - u1 * u2 * (1 - params.pf)


def state_terms(
    params: CaseParams,
    policy: Policy,
    state: tuple[str, str, int],
) -> tuple[float, dict[tuple[str, str, int], float]]:
    """Return one state's constant cost and transition probabilities."""
    status1, status2, phase = state
    product_inspection = policy.y if phase == PHASE_INITIAL else policy.yr

    constant = 0.0
    transitions: dict[tuple[str, str, int], float] = {}

    for prob1, cost1, good1, known1 in part_use_options(params, policy, 1, status1):
        for prob2, cost2, good2, known2 in part_use_options(params, policy, 2, status2):
            base_prob = prob1 * prob2
            base_cost = cost1 + cost2 + params.ca + product_inspection * params.tf

            if good1 and good2:
                quality_outcomes = [(1 - params.pf, True), (params.pf, False)]
            else:
                quality_outcomes = [(1.0, False)]

            for quality_prob, product_good in quality_outcomes:
                prob = base_prob * quality_prob
                if prob == 0:
                    continue

                constant += prob * base_cost
                if product_good:
                    continue

                extra = 0.0 if product_inspection else params.exchange_loss
                if policy.z:
                    extra += params.disassemble_cost
                    next_state = (
                        next_recovered_status(good1, known1),
                        next_recovered_status(good2, known2),
                        PHASE_RECOVERY,
                    )
                else:
                    next_state = (NONE, NONE, PHASE_INITIAL)

                constant += prob * extra
                transitions[next_state] = transitions.get(next_state, 0.0) + prob

    return constant, transitions


def reachable_states(params: CaseParams, policy: Policy):
    start = (NONE, NONE, PHASE_INITIAL)
    seen = {start}
    ordered = [start]
    cursor = 0

    while cursor < len(ordered):
        state = ordered[cursor]
        cursor += 1
        _, transitions = state_terms(params, policy, state)
        for next_state, prob in transitions.items():
            if prob <= 0 or next_state in seen:
                continue
            seen.add(next_state)
            ordered.append(next_state)

    return ordered


def build_equations(params: CaseParams, policy: Policy):
    states = reachable_states(params, policy)
    index = {state: i for i, state in enumerate(states)}
    n = len(states)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    rhs = [0.0 for _ in range(n)]

    for i in range(n):
        matrix[i][i] = 1.0

    for state in states:
        row = index[state]
        constant, transitions = state_terms(params, policy, state)
        rhs[row] = constant
        for next_state, prob in transitions.items():
            matrix[row][index[next_state]] -= prob

    return matrix, rhs, states


def is_dominated_reinspection(policy: Policy) -> bool:
    """Detect reinspection of a part already known qualified before assembly."""
    return bool(policy.z and ((policy.x1 and policy.r1) or (policy.x2 and policy.r2)))


def evaluate_policy(params: CaseParams, policy: Policy) -> dict:
    q_initial = initial_defect_probability(params, policy)
    ql_minus_tf = q_initial * params.exchange_loss - params.tf

    dominated = is_dominated_reinspection(policy)
    matrix, rhs, states = build_equations(params, policy)
    try:
        values = solve_linear_system(matrix, rhs)
        expected_cost = values[states.index((NONE, NONE, PHASE_INITIAL))]
        expected_profit = params.sale - expected_cost
        feasible = 1
        infeasible_reason = ""
    except ValueError:
        expected_cost = math.inf
        expected_profit = -math.inf
        feasible = 0
        infeasible_reason = "infinite loop or singular expectation system"

    return {
        "case": params.case,
        "x1": policy.x1,
        "x2": policy.x2,
        "y": policy.y,
        "z": policy.z,
        "r1": policy.r1,
        "r2": policy.r2,
        "yr": policy.yr,
        "initial_product_defect_prob_q": round(q_initial, 6),
        "qL_minus_tf": round(ql_minus_tf, 6),
        "feasible": feasible,
        "dominated_reinspection": int(dominated),
        "infeasible_reason": infeasible_reason,
        "expected_cost": "inf" if not feasible else round(expected_cost, 6),
        "expected_profit": "-inf" if not feasible else round(expected_profit, 6),
    }


def enumerate_policies():
    for values in product([0, 1], repeat=7):
        yield Policy(*values)


def evaluate_all() -> list[dict]:
    rows = []
    for params in CASES:
        for policy in enumerate_policies():
            rows.append(evaluate_policy(params, policy))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def profit_value(row: dict) -> float:
    if row["expected_profit"] == "-inf":
        return -math.inf
    return float(row["expected_profit"])


def sort_key(row: dict):
    return (
        row["case"],
        row["feasible"] == 0,
        row["dominated_reinspection"] == 1,
        -profit_value(row),
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    rows = evaluate_all()
    rows.sort(key=sort_key)

    best_rows = []
    for case_id in sorted({row["case"] for row in rows}):
        case_rows = [
            row for row in rows
            if row["case"] == case_id
            and row["feasible"]
            and not row["dominated_reinspection"]
        ]
        best_rows.append(max(case_rows, key=profit_value))

    write_csv(out_dir / "q2_policy_results.csv", rows)
    write_csv(out_dir / "q2_best_policies.csv", best_rows)

    print("Best non-dominated feasible policies by case:")
    for row in best_rows:
        print(
            f"case {row['case']}: "
            f"x1={row['x1']}, x2={row['x2']}, y={row['y']}, z={row['z']}, "
            f"r1={row['r1']}, r2={row['r2']}, yr={row['yr']}, "
            f"cost={float(row['expected_cost']):.4f}, "
            f"profit={float(row['expected_profit']):.4f}"
        )


if __name__ == "__main__":
    main()
