"""Question 4 figure pack: posterior densities and sample-size convergence.

Panel (a) shows the Beta(1,1)-prior posteriors of the three nominal defect
rates at n=40; panel (b) shows the Q3 posterior expected profit and its
one-standard-deviation band converging to the point-estimate result as the
evidence sample size grows.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import programs.q4_bayesian_model as q4

FIGURE_DIR = PROJECT_ROOT / "figures" / "q4"
DATA_DIR = FIGURE_DIR / "data"
RESULTS_DIR = PROJECT_ROOT / "programs" / "results"
STYLE_PATH = PROJECT_ROOT / "figures" / "style" / "cumcm.mplstyle"


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


def main() -> None:
    from scipy.stats import beta as beta_dist
    import matplotlib.pyplot as plt

    configure_plot_style()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    summary = json.loads((RESULTS_DIR / "q4_summary.json").read_text(encoding="utf-8"))
    specs = {
        row["nominal_rate"]: q4.posterior_spec(row["nominal_rate"], row["n"], "uniform")
        for row in summary["posterior_specs_n40"]
    }
    convergence = [
        row
        for row in summary["q3_sensitivity"]
        if row["problem"] == "q3"
    ]
    convergence.sort(key=lambda row: row["n"])
    point_estimate = 60.222222

    figure, (axis_a, axis_b) = plt.subplots(1, 2, figsize=(9.8, 3.9))

    x = np.linspace(0.0, 0.46, 600)
    colours = {"0.05": "#0072B2", "0.10": "#E69F00", "0.20": "#009E73"}
    for rate in (0.05, 0.10, 0.20):
        spec = specs[rate]
        y = beta_dist.pdf(x, spec.alpha, spec.beta)
        colour = colours[f"{rate:.2f}"]
        axis_a.plot(x, y, color=colour, linewidth=1.6, label=f"名义值 {rate:.0%}")
        axis_a.axvline(rate, color=colour, linestyle=":", linewidth=0.9)
    axis_a.set_xlabel("次品率")
    axis_a.set_ylabel("后验密度")
    axis_a.set_xlim(0, 0.46)
    axis_a.legend(loc="upper right", fontsize=8)

    ns = [row["n"] for row in convergence]
    profits = [row["best_expected_profit"] for row in convergence]
    sds = [row["profit_sd"] for row in convergence]
    ns_arr = np.asarray(ns, dtype=float)
    profits_arr = np.asarray(profits, dtype=float)
    sds_arr = np.asarray(sds, dtype=float)
    axis_b.fill_between(
        ns_arr, profits_arr - sds_arr, profits_arr + sds_arr,
        color="#0072B2", alpha=0.15, label="均值 ± 标准差",
    )
    axis_b.plot(ns_arr, profits_arr, color="#0072B2", marker="o", linewidth=1.6, label="后验期望利润")
    axis_b.axhline(point_estimate, color="#D55E00", linestyle="--", linewidth=1.1, label="点估计口径 60.2222")
    axis_b.set_xscale("log")
    axis_b.set_xticks(ns_arr)
    axis_b.set_xticklabels([str(n) for n in ns])
    axis_b.set_xlabel("抽样样本量 $n$（对数坐标）")
    axis_b.set_ylabel("后验期望利润（元）")
    axis_b.legend(loc="lower right", fontsize=8)

    figure.tight_layout()
    save_figure(figure, "fig_q4_posterior_convergence")
    plt.close(figure)

    # traceable data
    with (DATA_DIR / "q4_convergence.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["n", "profit", "sd", "q05"])
        writer.writeheader()
        for row in convergence:
            writer.writerow({
                "n": row["n"],
                "profit": row["best_expected_profit"],
                "sd": row["profit_sd"],
                "q05": row["profit_q05"],
            })
    with (DATA_DIR / "q4_posterior_densities.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["p", "rate_0.05", "rate_0.10", "rate_0.20"])
        writer.writeheader()
        for p in x:
            writer.writerow({
                "p": round(float(p), 6),
                "rate_0.05": round(float(beta_dist.pdf(p, specs[0.05].alpha, specs[0.05].beta)), 6),
                "rate_0.10": round(float(beta_dist.pdf(p, specs[0.10].alpha, specs[0.10].beta)), 6),
                "rate_0.20": round(float(beta_dist.pdf(p, specs[0.20].alpha, specs[0.20].beta)), 6),
            })
    print("fig_q4_posterior_convergence.pdf/.svg + data written to figures/q4/")


if __name__ == "__main__":
    main()
