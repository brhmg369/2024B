"""Monte Carlo cross-checks for Question 3 with statistical error scales.

Runs 20000 simulations for the optimal strategy, a near-optimal strategy and a
recycle-heavy feasible strategy, then writes analytic vs simulated cost, the
simulation standard deviation, standard error and 95% confidence interval to
``programs/results/q3_mc_stats.json``.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import programs.q3_decision_model as q3


OUT_FILE = PROJECT_ROOT / "programs" / "results" / "q3_mc_stats.json"
TRIALS = 20000
SEED = 777
STRATEGIES = (
    ("optimal", "1111111111111101"),
    ("near_optimal", "1111111111111001"),
    ("recycle_heavy", "1111111111101101"),
)


def main() -> None:
    rows = []
    for label, code in STRATEGIES:
        analytic = q3.evaluate_strategy(code)
        mc = q3.monte_carlo_check(code, trials=TRIALS, seed=SEED)
        rows.append(
            {
                "strategy_label": label,
                "strategy": code,
                "analytic_expected_cost": round(analytic.expected_cost, 6),
                "mc_mean_cost": round(mc["mc_expected_cost"], 6),
                "mc_cost_std": round(mc["mc_expected_cost_std"], 6),
                "mc_standard_error": round(mc["mc_standard_error"], 6),
                "mc_ci95_lower": round(mc["mc_ci95_lower"], 6),
                "mc_ci95_upper": round(mc["mc_ci95_upper"], 6),
                "completed_trials": mc["completed_trials"],
                "failed_trials": mc["failed_trials"],
                "analytic_in_ci": (
                    1
                    if mc["mc_ci95_lower"] <= analytic.expected_cost <= mc["mc_ci95_upper"]
                    else 0
                ),
            }
        )
        print(rows[-1])
    OUT_FILE.write_text(
        json.dumps(
            {"trials": TRIALS, "seed": SEED, "strategies": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("written to", OUT_FILE)


if __name__ == "__main__":
    main()
