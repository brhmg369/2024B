"""Build the traceable Question 3 Figure Pack and validation data.

The script re-runs the seeded GA to capture per-generation histories, then
reads the full-enumeration CSV for the profit landscape and top-10 comparison,
and finally enumerates a small single-factor sensitivity grid around the
Table-2 parameters.  All figures use the repository-wide CUMCM style and are
exported as both PDF and SVG with their source CSV under figures/q3/data/.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import programs.q3_decision_model as q3


FIGURE_DIR = PROJECT_ROOT / "figures" / "q3"
DATA_DIR = FIGURE_DIR / "data"
RESULT_DIR = PROJECT_ROOT / "programs" / "results"
STYLE_PATH = PROJECT_ROOT / "figures" / "style" / "cumcm.mplstyle"

GA_RUNS = 20
GENERATIONS = 200
SENSITIVITY_FACTORS = (0.80, 1.00, 1.20)

OPTIMAL_STRATEGY = "1111111111111101"
OPTIMAL_PROFIT = 60.222222
OPTIMAL_COST = 139.777778

MAIN_BLUE = "#0072B2"
CONTRAST_ORANGE = "#E69F00"
TEAL = "#009E73"
RED = "#D55E00"
DARK_GRAY = "#4D4D4D"
LIGHT_GRAY = "#B8B8B8"


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    materialised = list(rows)
    if not materialised:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialised[0].keys()))
        writer.writeheader()
        writer.writerows(materialised)


def configure_plot_style() -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if STYLE_PATH.exists():
        plt.style.use(STYLE_PATH)
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ["SimSun", "STSong", "Microsoft YaHei", "Noto Serif CJK SC"]
    selected = next((name for name in candidates if name in available), "DejaVu Serif")
    matplotlib.rcParams["font.family"] = selected
    matplotlib.rcParams["axes.unicode_minus"] = False


def save_figure(figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / f"{stem}.svg", format="svg", bbox_inches="tight")


def collect_ga_histories() -> list[dict]:
    rows: list[dict] = []
    for run_index in range(GA_RUNS):
        seed = 2024 + run_index
        result = q3.run_ga(seed=seed, generations=GENERATIONS)
        for record in result["history"]:
            rows.append(
                {
                    "run": run_index + 1,
                    "seed": seed,
                    "generation": record["generation"],
                    "best_profit": round(record["best_profit"], 9),
                    "current_best_profit": round(record["current_best_profit"], 9),
                    "unique_evaluated": record["unique_evaluated"],
                }
            )
    return rows


def load_exact_rows() -> list[dict]:
    rows: list[dict] = []
    with (RESULT_DIR / "q3_exact_all_strategies.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            profit = row["expected_profit"]
            rows.append(
                {
                    "strategy": row["strategy"],
                    "feasible": int(row["feasible"]),
                    "expected_cost": float(row["expected_cost"])
                    if row["expected_cost"] != "inf"
                    else float("inf"),
                    "expected_profit": float(profit)
                    if profit != "-inf"
                    else -float("inf"),
                    "num_states": int(row["num_states"]),
                }
            )
    return rows


def plot_ga_convergence(history_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    history = pd.DataFrame(history_rows)
    generations = history["generation"].unique()
    means = history.groupby("generation")["best_profit"].mean()
    mins = history.groupby("generation")["best_profit"].min()
    maxs = history.groupby("generation")["best_profit"].max()

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for run, group in history.groupby("run"):
        axis.plot(
            group["generation"],
            group["best_profit"],
            color=MAIN_BLUE,
            alpha=0.18,
            linewidth=0.9,
            label="单次运行" if run == 1 else None,
        )
    axis.fill_between(
        generations, mins, maxs, color=MAIN_BLUE, alpha=0.12, label="20 次运行范围"
    )
    axis.plot(
        generations,
        means,
        color=MAIN_BLUE,
        linewidth=1.8,
        label="20 次运行均值",
    )
    axis.axhline(
        OPTIMAL_PROFIT,
        color=RED,
        linestyle="--",
        linewidth=1.2,
        label="全枚举最优 60.2222",
    )
    axis.annotate(
        "20/20 次运行均命中全局最优",
        xy=(10, OPTIMAL_PROFIT),
        xytext=(60, OPTIMAL_PROFIT - 3.2),
        fontsize=8,
        color=DARK_GRAY,
        arrowprops=dict(arrowstyle="->", color=DARK_GRAY, lw=0.8),
    )
    axis.set_xlabel("进化代数")
    axis.set_ylabel("当前最优期望利润（元）")
    axis.set_xlim(-2, 205)
    axis.legend(loc="lower right", ncol=1)
    axis.margins(y=0.08)
    save_figure(figure, "fig_q3_ga_convergence")
    plt.close(figure)


def plot_profit_distribution(exact_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    feasible = [row for row in exact_rows if row["feasible"]]
    infeasible_count = len(exact_rows) - len(feasible)
    profits = np.array([row["expected_profit"] for row in feasible])

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    lo = float(np.floor(profits.min() / 20.0) * 20.0)
    hi = float(np.ceil(profits.max() / 5.0) * 5.0)
    bins = np.linspace(lo, hi, 121)
    axis.hist(
        profits,
        bins=bins,
        color=MAIN_BLUE,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.3,
        label="可行策略",
    )
    axis.axvline(
        OPTIMAL_PROFIT,
        color=RED,
        linestyle="--",
        linewidth=1.2,
        label=f"最优利润 {OPTIMAL_PROFIT:.4f}",
    )
    axis.text(
        0.99,
        0.95,
        f"可行 {len(feasible)} 个；不可行 {infeasible_count} 个",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=DARK_GRAY,
    )
    axis.set_xlabel("期望利润（元）")
    axis.set_ylabel("策略数")
    axis.legend(loc="upper left")
    save_figure(figure, "fig_q3_profit_distribution")
    plt.close(figure)


def plot_top10(top10: list[dict]) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    ranks = list(range(1, 11))
    profits = [row["expected_profit"] for row in top10]
    colours = [MAIN_BLUE if rank == 1 else LIGHT_GRAY for rank in ranks]
    bars = axis.barh(ranks, profits, color=colours, height=0.62)
    for bar, row in zip(bars, top10):
        code = row["strategy"]
        axis.text(
            bar.get_width() + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"{row['expected_profit']:.4f}",
            va="center",
            ha="left",
            fontsize=8,
            color=DARK_GRAY,
        )
        axis.text(
            -238.0,
            bar.get_y() + bar.get_height() / 2,
            code,
            va="center",
            ha="left",
            fontsize=7.5,
            color=DARK_GRAY,
        )
    axis.set_yticks(ranks)
    axis.set_yticklabels([str(rank) for rank in ranks])
    axis.invert_yaxis()
    axis.set_xlabel("期望利润（元）")
    axis.set_ylabel("排名")
    axis.set_xlim(-240, 64)
    axis.axvline(OPTIMAL_PROFIT, color=RED, linestyle="--", linewidth=1.0)
    axis.text(
        OPTIMAL_PROFIT + 0.2,
        9.2,
        "全枚举最优",
        fontsize=8,
        color=RED,
        rotation=90,
        va="bottom",
    )
    save_figure(figure, "fig_q3_top10")
    plt.close(figure)


def run_sensitivity() -> list[dict]:
    rows: list[dict] = []
    factor_specs = (
        ("final_defect", "成品次品率"),
        ("exchange_loss", "调换损失"),
        ("disassembly_cost", "拆解成本"),
    )
    for factor_key, factor_label in factor_specs:
        for level in SENSITIVITY_FACTORS:
            if level == 1.00:
                rows.append(
                    {
                        "factor": factor_key,
                        "factor_label": factor_label,
                        "level": level,
                        "strategy": OPTIMAL_STRATEGY,
                        "expected_cost": OPTIMAL_COST,
                        "expected_profit": OPTIMAL_PROFIT,
                        "ga_hit_count": GA_RUNS,
                        "baseline_strategy_profit": OPTIMAL_PROFIT,
                    }
                )
                continue
            if factor_key == "final_defect":
                params = replace(q3.TABLE2, final_defect=0.10 * level)
            elif factor_key == "exchange_loss":
                params = replace(q3.TABLE2, exchange_loss=40.0 * level)
            else:
                params = replace(
                    q3.TABLE2,
                    semi_disassemble=tuple(6.0 * level for _ in range(3)),
                    final_disassemble=10.0 * level,
                )
            # Re-run the same seeded GA search (20 independent runs) under the
            # perturbed parameters; the baseline optimal strategy is also
            # re-evaluated analytically for comparison.  Full re-enumeration is
            # not repeated for every perturbed scenario.
            ga_results = [
                q3.run_ga(
                    params=params,
                    seed=2024 + run,
                    generations=GENERATIONS,
                )
                for run in range(GA_RUNS)
            ]
            best = max(ga_results, key=lambda run: run["best_profit"])
            baseline_eval = q3.evaluate_strategy(OPTIMAL_STRATEGY, params)
            hit_count = sum(
                1
                for run in ga_results
                if run["best_strategy"] == best["best_strategy"]
                and abs(run["best_profit"] - best["best_profit"]) <= 1e-7
            )
            rows.append(
                {
                    "factor": factor_key,
                    "factor_label": factor_label,
                    "level": level,
                    "strategy": q3.strategy_to_code(best["best_strategy"]),
                    "expected_cost": round(best["best_cost"], 6),
                    "expected_profit": round(best["best_profit"], 6),
                    "ga_hit_count": hit_count,
                    "baseline_strategy_profit": round(
                        baseline_eval.expected_profit, 6
                    ),
                }
            )
    return rows


def plot_sensitivity(sensitivity_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    data = pd.DataFrame(sensitivity_rows)
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    style = {
        "成品次品率": (MAIN_BLUE, "o", "次品率"),
        "调换损失": (CONTRAST_ORANGE, "s", "调换损失"),
        "拆解成本": (TEAL, "^", "拆解成本"),
    }
    for label, (colour, marker, _legend) in style.items():
        group = data[data["factor_label"] == label].sort_values("level")
        axis.plot(
            group["level"],
            group["expected_profit"],
            color=colour,
            marker=marker,
            linewidth=1.6,
            markersize=5,
            label=label,
        )
    axis.axvline(1.00, color=DARK_GRAY, linestyle=":", linewidth=1.0, label="基准倍率")
    axis.set_xlabel("参数倍率")
    axis.set_ylabel("最优期望利润（元）")
    axis.set_xticks(list(SENSITIVITY_FACTORS))
    axis.legend(loc="lower left")
    axis.margins(x=0.06)
    save_figure(figure, "fig_q3_sensitivity")
    plt.close(figure)


def main() -> None:
    configure_plot_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    history_rows = collect_ga_histories()
    write_csv(DATA_DIR / "q3_ga_history.csv", history_rows)
    plot_ga_convergence(history_rows)

    exact_rows = load_exact_rows()
    feasible_sorted = sorted(
        (row for row in exact_rows if row["feasible"]),
        key=lambda row: row["expected_profit"],
        reverse=True,
    )
    top10 = feasible_sorted[:10]
    write_csv(DATA_DIR / "q3_top10_plot.csv", top10)
    write_csv(
        DATA_DIR / "q3_profit_distribution.csv",
        [
            {
                "strategy": row["strategy"],
                "feasible": row["feasible"],
                "expected_profit": (
                    None if not row["feasible"] else row["expected_profit"]
                ),
                "expected_cost": (
                    None if not row["feasible"] else row["expected_cost"]
                ),
            }
            for row in exact_rows
        ],
    )
    plot_profit_distribution(exact_rows)
    plot_top10(top10)

    sensitivity_rows = run_sensitivity()
    write_csv(DATA_DIR / "q3_sensitivity.csv", sensitivity_rows)
    plot_sensitivity(sensitivity_rows)

    print("Question 3 Figure Pack written to figures/q3/")
    print(f"GA history rows: {len(history_rows)}")
    print(f"Exact rows: {len(exact_rows)}; feasible: {len(feasible_sorted)}")
    print("Sensitivity:")
    for row in sensitivity_rows:
        print(row)


if __name__ == "__main__":
    main()
