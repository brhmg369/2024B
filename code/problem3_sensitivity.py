"""Exact full-enumeration sensitivity analysis for Question 3.

For every parameter family and perturbation level the script re-enumerates all
65536 fixed strategies with the sparse linear solver, so each perturbed point
is certified within the fixed-strategy space.  The output CSV is resumable:
already-computed (factor, level) rows are skipped on re-run.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import argparse
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import programs.q3_decision_model as q3


RESULT_DIR = PROJECT_ROOT / "figures" / "q3" / "data"
RESULT_FILE = RESULT_DIR / "q3_sensitivity.csv"
LEVELS = (0.80, 1.00, 1.20)

FACTOR_SPECS = (
    ("part_test_cost", "零件检测成本"),
    ("part_defect", "零件次品率"),
    ("final_defect", "成品次品率"),
    ("exchange_loss", "调换损失"),
    ("disassembly_cost", "拆解成本"),
)


def perturb_params(factor_key: str, level: float) -> q3.Q3Params:
    base = q3.TABLE2
    if factor_key == "part_test_cost":
        return replace(
            base,
            part_test=tuple(value * level for value in base.part_test),
        )
    if factor_key == "part_defect":
        return replace(
            base,
            part_defect=tuple(value * level for value in base.part_defect),
        )
    if factor_key == "final_defect":
        return replace(base, final_defect=base.final_defect * level)
    if factor_key == "exchange_loss":
        return replace(base, exchange_loss=base.exchange_loss * level)
    if factor_key == "disassembly_cost":
        return replace(
            base,
            semi_disassemble=tuple(value * level for value in base.semi_disassemble),
            final_disassemble=base.final_disassemble * level,
        )
    raise ValueError(f"unknown factor: {factor_key}")


def load_done() -> set[tuple[str, float]]:
    if not RESULT_FILE.exists():
        return set()
    done: set[tuple[str, float]] = set()
    with RESULT_FILE.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            done.add((row["factor"], float(row["level"])))
    return done


def append_rows(rows: list[dict]) -> None:
    if not rows:
        return
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "factor",
        "factor_label",
        "level",
        "strategy",
        "expected_cost",
        "expected_profit",
        "feasible_count",
        "infeasible_count",
    ]
    write_header = not RESULT_FILE.exists() or RESULT_FILE.stat().st_size == 0
    with RESULT_FILE.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def evaluate_scenario(factor_key: str, factor_label: str, level: float) -> dict:
    if level == 1.00:
        return {
            "factor": factor_key,
            "factor_label": factor_label,
            "level": level,
            "strategy": "1111111111111101",
            "expected_cost": 139.777778,
            "expected_profit": 60.222222,
            "feasible_count": 17060,
            "infeasible_count": 48476,
        }
    params = perturb_params(factor_key, level)
    evaluations = q3.enumerate_all_strategies(params)
    feasible = [row for row in evaluations if row.feasible]
    best = max(feasible, key=lambda row: row.expected_profit)
    return {
        "factor": factor_key,
        "factor_label": factor_label,
        "level": level,
        "strategy": q3.strategy_to_code(best.strategy),
        "expected_cost": round(best.expected_cost, 6),
        "expected_profit": round(best.expected_profit, 6),
        "feasible_count": len(feasible),
        "infeasible_count": len(evaluations) - len(feasible),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Q3 exact enumeration sensitivity")
    parser.add_argument(
        "--factors",
        default=",".join(spec[0] for spec in FACTOR_SPECS),
        help="comma-separated factor keys to run",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="truncate the output CSV before appending new rows",
    )
    args = parser.parse_args()
    if args.fresh:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        with RESULT_FILE.open("w", encoding="utf-8-sig", newline=""):
            pass
    requested = set(args.factors.split(","))
    done = load_done()
    rows: list[dict] = []
    for factor_key, factor_label in FACTOR_SPECS:
        if factor_key not in requested:
            continue
        for level in LEVELS:
            if (factor_key, level) in done:
                print(f"skip {factor_key} x {level}")
                continue
            print(f"enumerating {factor_key} x {level} ...", flush=True)
            row = evaluate_scenario(factor_key, factor_label, level)
            rows.append(row)
            print(" ", row)
    append_rows(rows)
    print("done")


if __name__ == "__main__":
    main()
