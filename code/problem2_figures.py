"""Build the traceable Figure Pack and validation data for question 2."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable
import csv
import math
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import programs.q2_decision_model as q2


FIGURE_DIR = PROJECT_ROOT / "figures" / "q2"
DATA_DIR = FIGURE_DIR / "data"
RESULT_DIR = PROJECT_ROOT / "results"
STYLE_PATH = PROJECT_ROOT / "figures" / "style" / "cumcm.mplstyle"

SENSITIVITY_FACTORS = (0.80, 0.90, 1.00, 1.10, 1.20)
ROUNDING_DIGITS = (4, 5, 6, 7)

# Restricted seven-variable policy baseline from repository commit 53166ac.
FIXED_POLICY_BASELINE = {
    1: 37.077778,
    2: 44.000000,
    3: 39.411111,
    4: 41.250000,
    5: 40.550000,
    6: 34.321330,
}

ACTION_LABELS = {
    "buy_p1_test": "购检部件1",
    "buy_p2_test": "购检部件2",
    "buy_p1_notest": "购部件1不检",
    "buy_p2_notest": "购部件2不检",
    "inspect_p1": "检回收部件1",
    "inspect_p2": "检回收部件2",
    "assemble_notest_scrap": "装配\n成品不检/报废",
    "assemble_notest_disassemble": "装配\n成品不检/拆解",
    "assemble_test_scrap": "装配\n成品检/报废",
    "assemble_test_disassemble": "装配\n成品检/拆解",
    "terminal": "完成交付",
}

ACTION_GROUPS = {
    "buy_p1_test": "购入并检测",
    "buy_p2_test": "购入并检测",
    "buy_p1_notest": "购入不检测",
    "buy_p2_notest": "购入不检测",
    "inspect_p1": "检测回收件",
    "inspect_p2": "检测回收件",
    "assemble_notest_disassemble": "不检成品并拆解",
    "assemble_test_disassemble": "检测成品并拆解",
    "assemble_notest_scrap": "不检成品并报废",
    "assemble_test_scrap": "检测成品并报废",
    "terminal": "完成交付",
}

GROUP_ORDER = (
    "购入并检测",
    "购入不检测",
    "检测回收件",
    "不检成品并拆解",
    "检测成品并拆解",
    "不检成品并报废",
    "检测成品并报废",
    "完成交付",
)

GROUP_COLOURS = {
    "购入并检测": "#0072B2",
    "购入不检测": "#56B4E9",
    "检测回收件": "#E69F00",
    "不检成品并拆解": "#009E73",
    "检测成品并拆解": "#CC79A7",
    "不检成品并报废": "#4D4D4D",
    "检测成品并报废": "#D55E00",
    "完成交付": "#B8B8B8",
}


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


def expected_cost(solution: dict) -> float:
    return float(solution["values"][q2.START_STATE])


def policy_signature(solution: dict) -> str:
    traced = q2.trace_initial_policy(solution)
    return ">".join(
        traced[key]
        for key in (
            "initial_component_action_1",
            "initial_component_action_2",
            "first_assembly_action",
            "after_first_defect_action",
        )
    )


def build_policy_path_rows(solutions: list[dict]) -> list[dict]:
    stages = (
        ("initial_component_action_1", "首次购件动作"),
        ("initial_component_action_2", "第二购件动作"),
        ("first_assembly_action", "首次装配动作"),
        ("after_first_defect_action", "首次缺陷后动作"),
    )
    rows = []
    for solution in solutions:
        case = solution["params"].case
        traced = q2.trace_initial_policy(solution)
        for stage_order, (field, stage_label) in enumerate(stages, start=1):
            action = traced[field]
            rows.append(
                {
                    "case": case,
                    "stage_order": stage_order,
                    "stage": stage_label,
                    "action": action,
                    "action_label": ACTION_LABELS[action].replace("\n", " / "),
                    "action_group": ACTION_GROUPS[action],
                }
            )
    return rows


def scale_case(params: q2.CaseParams, parameter: str, factor: float) -> q2.CaseParams:
    if parameter == "defect_rates":
        return replace(
            params,
            p1=min(params.p1 * factor, 1.0 - 1e-9),
            p2=min(params.p2 * factor, 1.0 - 1e-9),
            pf=min(params.pf * factor, 1.0 - 1e-9),
        )
    if parameter == "exchange_loss":
        return replace(params, exchange_loss=params.exchange_loss * factor)
    if parameter == "disassemble_cost":
        return replace(params, disassemble_cost=params.disassemble_cost * factor)
    raise ValueError(parameter)


def build_sensitivity_rows(
    baseline_solutions: list[dict],
) -> list[dict]:
    baseline_by_case = {
        solution["params"].case: solution for solution in baseline_solutions
    }
    rows = []
    for parameter in ("defect_rates", "exchange_loss", "disassemble_cost"):
        for params in q2.CASES:
            baseline = baseline_by_case[params.case]
            baseline_signature = policy_signature(baseline)
            baseline_profit = params.sale - expected_cost(baseline)
            for factor in SENSITIVITY_FACTORS:
                varied = scale_case(params, parameter, factor)
                solution = q2.solve_case(varied)
                cost = expected_cost(solution)
                signature = policy_signature(solution)
                rows.append(
                    {
                        "parameter": parameter,
                        "case": params.case,
                        "factor": f"{factor:.2f}",
                        "expected_cost": f"{cost:.9f}",
                        "expected_profit": f"{params.sale - cost:.9f}",
                        "profit_change_from_baseline": (
                            f"{params.sale - cost - baseline_profit:.9f}"
                        ),
                        "policy_signature": signature,
                        "policy_changed": int(signature != baseline_signature),
                        "num_states": len(solution["states"]),
                        "iterations": solution["iterations"],
                        "bellman_residual": f"{solution['bellman_residual']:.12e}",
                    }
                )
    return rows


def build_rounding_rows() -> list[dict]:
    original_digits = q2.ROUND_DIGITS
    raw_rows = []
    try:
        for digits in ROUNDING_DIGITS:
            q2.ROUND_DIGITS = digits
            for params in q2.CASES:
                solution = q2.solve_case(params)
                raw_rows.append(
                    {
                        "case": params.case,
                        "round_digits": digits,
                        "expected_cost": expected_cost(solution),
                        "num_states": len(solution["states"]),
                        "iterations": solution["iterations"],
                        "bellman_residual": solution["bellman_residual"],
                    }
                )
    finally:
        q2.ROUND_DIGITS = original_digits

    reference = {
        row["case"]: row["expected_cost"]
        for row in raw_rows
        if row["round_digits"] == max(ROUNDING_DIGITS)
    }
    return [
        {
            **row,
            "absolute_difference_from_7_digits": abs(
                row["expected_cost"] - reference[row["case"]]
            ),
        }
        for row in raw_rows
    ]


def build_fixed_policy_comparison(solutions: list[dict]) -> list[dict]:
    rows = []
    for solution in solutions:
        case = solution["params"].case
        mdp_cost = expected_cost(solution)
        fixed_cost = FIXED_POLICY_BASELINE[case]
        saving = fixed_cost - mdp_cost
        if abs(saving) < 2e-6:
            saving = 0.0
        rows.append(
            {
                "case": case,
                "fixed_policy_expected_cost": f"{fixed_cost:.9f}",
                "belief_mdp_expected_cost": f"{mdp_cost:.9f}",
                "cost_saving": f"{saving:.9f}",
                "relative_saving_percent": f"{100.0 * saving / fixed_cost:.6f}",
                "fixed_baseline_source": "git commit 53166ac",
            }
        )
    return rows


def plot_policy_path(rows: list[dict]) -> None:
    configure_plot_style()
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cases = sorted({int(row["case"]) for row in rows})
    stages = sorted(
        {(int(row["stage_order"]), row["stage"]) for row in rows},
        key=lambda item: item[0],
    )
    stage_labels = [label for _, label in stages]
    lookup = {(int(row["stage_order"]), int(row["case"])): row for row in rows}
    used_groups = [group for group in GROUP_ORDER if any(row["action_group"] == group for row in rows)]
    group_index = {group: idx for idx, group in enumerate(used_groups)}
    matrix = np.array(
        [
            [group_index[lookup[(stage_order, case)]["action_group"]] for case in cases]
            for stage_order, _ in stages
        ]
    )

    figure, axis = plt.subplots(figsize=(8.2, 4.2))
    axis.imshow(
        matrix,
        cmap=ListedColormap([GROUP_COLOURS[group] for group in used_groups]),
        vmin=-0.5,
        vmax=len(used_groups) - 0.5,
        aspect="auto",
    )
    axis.set_xticks(range(len(cases)), [f"情形 {case}" for case in cases])
    axis.set_yticks(range(len(stage_labels)), stage_labels)
    axis.set_xticks(np.arange(-0.5, len(cases), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(stage_labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.4)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.grid(which="major", visible=False)
    for row_index, (stage_order, _) in enumerate(stages):
        for column_index, case in enumerate(cases):
            action = lookup[(stage_order, case)]["action"]
            axis.text(
                column_index,
                row_index,
                ACTION_LABELS[action],
                ha="center",
                va="center",
                color="white" if ACTION_GROUPS[action] not in {"购入不检测", "完成交付"} else "black",
                fontsize=7.5,
                fontweight="bold" if row_index >= 2 else "normal",
            )
    handles = [Patch(facecolor=GROUP_COLOURS[group], label=group) for group in used_groups]
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout()
    save_figure(figure, "fig_q2_policy_path")
    plt.close(figure)


def plot_validation(
    convergence_rows: list[dict],
    rounding_rows: list[dict],
) -> None:
    configure_plot_style()
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    colours = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#4D4D4D")
    markers = ("o", "s", "^", "D", "v", "P")
    linestyles = ("-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1)))
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

    for case, colour, marker, linestyle in zip(range(1, 7), colours, markers, linestyles):
        subset = [row for row in convergence_rows if int(row["case"]) == case]
        axes[0].plot(
            [int(row["iteration"]) for row in subset],
            [float(row["update_delta"]) for row in subset],
            color=colour,
            marker=marker,
            markevery=max(1, len(subset) // 5),
            linestyle=linestyle,
            label=f"情形 {case}",
        )
        rounded = [row for row in rounding_rows if int(row["case"]) == case]
        axes[1].plot(
            [int(row["round_digits"]) for row in rounded],
            [1e4 * float(row["absolute_difference_from_7_digits"]) for row in rounded],
            color=colour,
            marker=marker,
            linestyle=linestyle,
            label=f"情形 {case}",
        )

    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0e"))
    axes[0].axhline(q2.VALUE_TOL, color="#4D4D4D", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("值迭代轮次")
    axes[0].set_ylabel("最大更新量")
    axes[0].set_title("(a) 值迭代收敛")
    axes[1].set_xlabel("信念概率保留小数位")
    axes[1].set_ylabel("相对 7 位结果的绝对差（0.0001 元）")
    axes[1].set_xticks(ROUNDING_DIGITS)
    axes[1].set_title("(b) 信念离散精度检验")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=6)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, "fig_q2_validation")
    plt.close(figure)


def plot_sensitivity(rows: list[dict]) -> None:
    configure_plot_style()
    import matplotlib.pyplot as plt

    panels = (
        ("defect_rates", "(a) 三类次品率同比扰动"),
        ("exchange_loss", "(b) 调换损失扰动"),
        ("disassemble_cost", "(c) 拆解成本扰动"),
    )
    colours = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#4D4D4D")
    markers = ("o", "s", "^", "D", "v", "P")
    linestyles = ("-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1)))
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.1), sharey=True)
    for axis, (parameter, title) in zip(axes, panels):
        for case, colour, marker, linestyle in zip(range(1, 7), colours, markers, linestyles):
            subset = [
                row
                for row in rows
                if row["parameter"] == parameter and int(row["case"]) == case
            ]
            subset.sort(key=lambda row: float(row["factor"]))
            axis.plot(
                [float(row["factor"]) for row in subset],
                [float(row["expected_profit"]) for row in subset],
                color=colour,
                marker=marker,
                linestyle=linestyle,
                label=f"情形 {case}",
            )
        axis.axvline(1.0, color="#4D4D4D", linestyle="--", linewidth=1.0)
        axis.set_xlabel("参数倍率")
        axis.set_title(title)
        axis.set_xticks(SENSITIVITY_FACTORS)
    axes[0].set_ylabel("期望利润（元/件）")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=6)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(figure, "fig_q2_sensitivity")
    plt.close(figure)


def main() -> int:
    q2.ROUND_DIGITS = 6
    baseline_solutions = [q2.solve_case(params) for params in q2.CASES]
    policy_rows = build_policy_path_rows(baseline_solutions)
    convergence_rows = q2.build_convergence_rows(baseline_solutions)
    sensitivity_rows = build_sensitivity_rows(baseline_solutions)
    rounding_rows = build_rounding_rows()
    comparison_rows = build_fixed_policy_comparison(baseline_solutions)

    write_csv(DATA_DIR / "q2_policy_path.csv", policy_rows)
    write_csv(DATA_DIR / "q2_convergence.csv", convergence_rows)
    write_csv(DATA_DIR / "q2_sensitivity.csv", sensitivity_rows)
    write_csv(DATA_DIR / "q2_rounding_validation.csv", rounding_rows)
    write_csv(DATA_DIR / "q2_fixed_policy_comparison.csv", comparison_rows)
    write_csv(RESULT_DIR / "q2_fixed_policy_comparison.csv", comparison_rows)

    plot_policy_path(policy_rows)
    plot_validation(convergence_rows, rounding_rows)
    plot_sensitivity(sensitivity_rows)
    print(f"Question 2 Figure Pack written to {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
