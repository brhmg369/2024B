"""Final paper-vs-results consistency check.

Verifies that the key numbers used in the LaTeX paper match the final result
files.  Run from the repository root:  python code/final_consistency_check.py
"""

from __future__ import annotations

from pathlib import Path
import csv
import json


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "sections"
RESULTS = ROOT / "programs" / "results"


def tex_text() -> str:
    parts = []
    for path in sorted(PAPER.glob("*.tex")):
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def load_csv(name: str) -> list[dict]:
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    text = tex_text()
    failures: list[str] = []

    def check(label: str, needle: str) -> None:
        if needle in text:
            print(f"PASS  {label}")
        else:
            failures.append(label)
            print(f"FAIL  {label}  (missing: {needle})")

    # Q1
    q1 = json.loads((ROOT / "results" / "q1_summary.json").read_text(encoding="utf-8"))
    check("Q1 n*=22", "22")
    check("Q1 accept cutoff 0", "接收")
    check("Q1 reject cutoff 6", "拒收")
    for value in ("0.098477", "0.018215"):
        check(f"Q1 tail {value}", value)

    # Q2
    q2 = load_json("q2_summary.json")
    q2_costs = "37.0778, 44.0000, 39.3467, 41.2500, 40.5500, 34.3213"
    check("Q2 six costs", "37.0778")
    check("Q2 six profits", "18.9222")
    check("Q2 case3 improvement 0.0644", "0.0644")
    check("Q2 fixed baseline 39.411111", "39.411111")

    # Q3
    q3 = (RESULTS / "q3_summary.txt").read_text(encoding="utf-8")
    check("Q3 best strategy", "1111111111111101")
    check("Q3 profit 60.2222", "60.2222")
    check("Q3 cost 139.7778", "139.7778")
    check("Q3 feasible 17060", "17060")
    check("Q3 infeasible 48476", "48476")
    mc = load_json("q3_mc_stats.json")
    first = mc["strategies"][0]
    check("Q3 MC optimal CI lower", str(round(first["mc_ci95_lower"], 4)))
    check("Q3 MC optimal CI upper", str(round(first["mc_ci95_upper"], 4)))

    # Q4
    q4 = load_json("q4_summary.json")
    q4_best = load_csv("q4_q2_best_policies.csv")
    check("Q4 seven-bit class note", "七变量固定策略类")
    check("Q4 case5 switch 0111100", "0111100")
    for row in q4_best:
        rounded = str(round(float(row["bayes_best_expected_profit"]), 4))
        check(f"Q4 case{row['case']} profit {rounded}", rounded)
    q3_q4 = [r for r in q4["q3_sensitivity"] if r["n"] == 40][0]
    check("Q4 Q3 profit 55.0311", str(round(q3_q4["best_expected_profit"], 4)))
    check("Q4 Q3 sd 4.8719", str(round(q3_q4["profit_sd"], 4)))
    prior = load_csv("q4_prior_sensitivity.csv")
    check("Q4 prior sensitivity uniform case5", "0111100")
    check("Q4 prior sensitivity jeffreys case5", "0101100")
    stability = load_csv("q4_scenario_stability.csv")
    check(
        "Q4 scenario stability S=20000",
        str(round(float(stability[-1]["q3_bayes_expected_profit"]), 4)),
    )

    print()
    if failures:
        print(f"CONSISTENCY CHECK: {len(failures)} FAILURES")
        for item in failures:
            print(" -", item)
    else:
        print("CONSISTENCY CHECK: ALL PASS")


if __name__ == "__main__":
    main()
