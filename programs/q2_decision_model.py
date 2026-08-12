"""Question 2 compressed decision model.

Workflow used for the handoff:
1. Enumerate four part-inspection policies (x1, x2).
2. Compute the corresponding finished-product defect probability q.
3. Decide finished-product inspection by the local rule t_f < q L.
4. Enumerate disassembly z = 0, 1.
5. Compute total expected cost and expected profit, then compare globally.

Cost unit: expected total cost to finally deliver one qualified finished product.
Profit unit: sale price minus expected cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import csv


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


def inspected_part_cost(price: float, test_cost: float, defect_rate: float) -> float:
    """Expected cost of obtaining one qualified inspected part."""
    return (price + test_cost) / (1 - defect_rate)


def part_entry_cost_and_quality(
    inspect: int,
    price: float,
    test_cost: float,
    defect_rate: float,
) -> tuple[float, float]:
    """Return expected entry cost K_i and qualified probability u_i."""
    if inspect:
        return inspected_part_cost(price, test_cost, defect_rate), 1.0
    return price, 1 - defect_rate


def recovered_part_value(
    inspect: int,
    entry_cost: float,
    entry_good_prob: float,
    product_good_prob: float,
    product_bad_prob: float,
) -> float:
    """Approximate value of a part recovered from a defective product.

    If a part was inspected before assembly, it is certainly qualified and keeps
    its full replacement value. If not, use the conditional probability that it
    is actually qualified given that the finished product is defective.
    """
    if inspect:
        return entry_cost
    if product_bad_prob <= 0:
        return 0.0
    conditional_good_prob = (entry_good_prob - product_good_prob) / product_bad_prob
    return max(0.0, conditional_good_prob * entry_cost)


def evaluate_strategy(params: CaseParams, x1: int, x2: int, z: int) -> dict:
    k1, u1 = part_entry_cost_and_quality(x1, params.c1, params.t1, params.p1)
    k2, u2 = part_entry_cost_and_quality(x2, params.c2, params.t2, params.p2)

    good_prob = u1 * u2 * (1 - params.pf)
    bad_prob = 1 - good_prob

    # Local dominance rule for finished-product inspection.
    y = 1 if params.tf < bad_prob * params.exchange_loss else 0
    product_inspection_margin = bad_prob * params.exchange_loss - params.tf

    base_cost = k1 + k2 + params.ca + y * params.tf

    v1 = recovered_part_value(x1, k1, u1, good_prob, bad_prob)
    v2 = recovered_part_value(x2, k2, u2, good_prob, bad_prob)
    recovery_value = v1 + v2
    disassembly_margin = recovery_value - params.disassemble_cost

    # If z = 1, the defective-product branch pays disassembly cost and recovers
    # parts with expected value V. If z = 0, this term is omitted.
    defective_branch_cost = (1 - y) * params.exchange_loss
    if z:
        defective_branch_cost += params.disassemble_cost - recovery_value

    expected_cost = (base_cost + bad_prob * defective_branch_cost) / good_prob
    expected_profit = params.sale - expected_cost

    return {
        "case": params.case,
        "inspect_part1": x1,
        "inspect_part2": x2,
        "product_defect_prob_q": round(bad_prob, 6),
        "qL_minus_tf": round(product_inspection_margin, 6),
        "inspect_product_by_rule": y,
        "disassemble": z,
        "recovery_value_V": round(recovery_value, 6),
        "V_minus_d": round(disassembly_margin, 6),
        "expected_cost": round(expected_cost, 6),
        "expected_profit": round(expected_profit, 6),
    }


def evaluate_all() -> list[dict]:
    rows = []
    for params in CASES:
        for x1, x2 in product([0, 1], repeat=2):
            for z in [0, 1]:
                rows.append(evaluate_strategy(params, x1, x2, z))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    rows = evaluate_all()
    rows.sort(key=lambda row: (row["case"], -row["expected_profit"]))

    best_rows = []
    for case_id in sorted({row["case"] for row in rows}):
        case_rows = [row for row in rows if row["case"] == case_id]
        best_rows.append(max(case_rows, key=lambda row: row["expected_profit"]))

    write_csv(out_dir / "q2_policy_results.csv", rows)
    write_csv(out_dir / "q2_best_policies.csv", best_rows)

    print("Best policies by case:")
    for row in best_rows:
        print(
            f"case {row['case']}: "
            f"x1={row['inspect_part1']}, x2={row['inspect_part2']}, "
            f"y={row['inspect_product_by_rule']}, z={row['disassemble']}, "
            f"q={row['product_defect_prob_q']:.4f}, "
            f"V={row['recovery_value_V']:.4f}, "
            f"cost={row['expected_cost']:.4f}, "
            f"profit={row['expected_profit']:.4f}"
        )


if __name__ == "__main__":
    main()
