"""Render the Question 3 production-structure schematic (non-data figure).

The figure shows the two-stage assembly structure of Table 2 together with the
sixteen decision bits of the strategy code.  A matching ``fig_q3_structure.drawio``
source is kept under ``figures/q3/`` for editable re-layout; this script only
provides the deterministic vector export used by the paper when the DrawIO CLI
is not available.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FIGURE_DIR = PROJECT_ROOT / "figures" / "q3"
STYLE_PATH = PROJECT_ROOT / "figures" / "style" / "cumcm.mplstyle"

PART_BLUE = "#56B4E9"
SEMI_TEAL = "#009E73"
FINAL_BLUE = "#0072B2"
ARROW_GRAY = "#4D4D4D"
DECISION_GRAY = "#4D4D4D"
BORDER_GRAY = "#4D4D4D"


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


def draw_box(axis, x, y, width, height, text, fill, fontsize=9, text_color="white"):
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=0.9,
        edgecolor=BORDER_GRAY,
        facecolor=fill,
    )
    axis.add_patch(box)
    axis.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        zorder=5,
    )


def draw_arrow(axis, x1, y1, x2, y2, color=ARROW_GRAY, lw=1.2):
    axis.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=10,
            shrinkA=2,
            shrinkB=2,
        ),
        zorder=3,
    )


def main() -> None:
    import matplotlib.pyplot as plt

    configure_plot_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9.5, 5.4))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 6.0)
    axis.axis("off")

    part_x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    part_y = 4.7
    for index, x in enumerate(part_x, start=1):
        draw_box(axis, x, part_y, 0.78, 0.58, f"零配件{index}", PART_BLUE, fontsize=8)
        axis.text(
            x,
            part_y - 0.62,
            f"$x_{index}$：检测",
            ha="center",
            va="top",
            fontsize=7,
            color=DECISION_GRAY,
        )

    semi_groups = ((1, 2, 3), (4, 5, 6), (7, 8))
    semi_x = [2.0, 5.0, 7.5]
    semi_y = 2.8
    for group, x in zip(semi_groups, semi_x):
        members = "、".join(str(i) for i in group)
        draw_box(axis, x, semi_y, 1.7, 0.62, f"半成品（{members}）", SEMI_TEAL, fontsize=8)
        for part in group:
            draw_arrow(axis, part_x[part - 1], part_y - 0.30, x, semi_y + 0.32)

    axis.text(
        0.35,
        semi_y - 0.62,
        "半成品检测 $x_9,x_{10},x_{11}$；\n不合格半成品拆解 $x_{12},x_{13},x_{14}$",
        ha="left",
        va="top",
        fontsize=8,
        color=DECISION_GRAY,
    )

    final_x = 4.5
    final_y = 1.0
    draw_box(axis, final_x, final_y, 2.0, 0.66, "成品", FINAL_BLUE, fontsize=9)
    for x in semi_x:
        draw_arrow(axis, x, semi_y - 0.32, final_x, final_y + 0.34)

    axis.text(
        final_x + 2.35,
        final_y + 0.35,
        "成品检测 $x_{15}$；\n不合格成品拆解 $x_{16}$",
        ha="left",
        va="center",
        fontsize=8,
        color=DECISION_GRAY,
    )

    axis.text(
        0.35,
        5.65,
        "质量传递：任一下级输入不合格则上级必不合格；仅当下级全部合格时，"
        "上级按给定次品率产生装配缺陷。",
        ha="left",
        va="top",
        fontsize=8,
        color=DECISION_GRAY,
    )

    figure.savefig(FIGURE_DIR / "fig_q3_structure.pdf", format="pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / "fig_q3_structure.svg", format="svg", bbox_inches="tight")
    plt.close(figure)
    print("fig_q3_structure.pdf/.svg written to figures/q3/")


if __name__ == "__main__":
    main()
