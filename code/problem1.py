"""2024 年国赛 B 题问题一：精确抽样检测方案。

主模型为固定样本量的精确二项单侧检验；给出总体量 N 时，同时计算
无放回抽样的超几何有限总体修正。程序不会把逐次重复查看固定样本
检验结果解释为序贯检验。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_FLOOR
from itertools import accumulate
from pathlib import Path
from typing import Iterable, Sequence


FLOAT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ThresholdResult:
    """某一固定样本量下的三态判定边界及校核量。"""

    model: str
    population_size: int | None
    sample_size: int
    nominal_defect_rate: float
    acceptable_defectives: int | None
    first_unacceptable_defectives: int | None
    accept_confidence: float
    reject_confidence: float
    accept_alpha: float
    reject_alpha: float
    accept_cutoff: int | None
    reject_cutoff: int | None
    accept_tail_at_cutoff: float | None
    accept_tail_after_cutoff: float | None
    reject_tail_at_cutoff: float | None
    reject_tail_before_cutoff: float | None
    inconclusive_low: int | None
    inconclusive_high: int | None
    accept_constraint_ok: bool
    reject_constraint_ok: bool
    accept_boundary_maximal_ok: bool
    reject_boundary_minimal_ok: bool
    disjoint_regions_ok: bool

    @property
    def has_accept_region(self) -> bool:
        return self.accept_cutoff is not None

    @property
    def has_reject_region(self) -> bool:
        return self.reject_cutoff is not None

    @property
    def has_both_regions(self) -> bool:
        return (
            self.has_accept_region
            and self.has_reject_region
            and self.disjoint_regions_ok
        )

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row.update(
            {
                "has_accept_region": self.has_accept_region,
                "has_reject_region": self.has_reject_region,
                "has_both_regions": self.has_both_regions,
            }
        )
        return row


def _validate_probability(value: float, name: str, *, open_interval: bool = True) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数。")
    if open_interval and not 0.0 < value < 1.0:
        raise ValueError(f"{name} 必须严格位于 (0, 1)。")
    if not open_interval and not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} 必须位于 [0, 1]。")


def _validate_integer(value: int, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必须是不小于 {minimum} 的整数。")


def _log_combination(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _normalise_log_probabilities(log_probabilities: Sequence[float]) -> list[float]:
    finite_logs = [value for value in log_probabilities if math.isfinite(value)]
    if not finite_logs:
        raise ArithmeticError("概率支持集为空。")
    shift = max(finite_logs)
    weights = [
        math.exp(value - shift) if math.isfinite(value) else 0.0
        for value in log_probabilities
    ]
    total = math.fsum(weights)
    if total <= 0.0 or not math.isfinite(total):
        raise ArithmeticError("概率归一化失败。")
    probabilities = [weight / total for weight in weights]
    if abs(math.fsum(probabilities) - 1.0) > 1e-10:
        raise ArithmeticError("概率和未通过归一化校验。")
    return probabilities


def binomial_probabilities(sample_size: int, defect_rate: float) -> list[float]:
    """返回 X~Bin(n,p) 在 0..n 上的概率。"""

    _validate_integer(sample_size, "sample_size", minimum=0)
    _validate_probability(defect_rate, "defect_rate", open_interval=False)
    probabilities = [0.0] * (sample_size + 1)
    if defect_rate == 0.0:
        probabilities[0] = 1.0
        return probabilities
    if defect_rate == 1.0:
        probabilities[sample_size] = 1.0
        return probabilities

    log_p = math.log(defect_rate)
    log_q = math.log1p(-defect_rate)
    logs = [
        _log_combination(sample_size, defects)
        + defects * log_p
        + (sample_size - defects) * log_q
        for defects in range(sample_size + 1)
    ]
    return _normalise_log_probabilities(logs)


def hypergeometric_probabilities(
    population_size: int,
    defective_count: int,
    sample_size: int,
) -> list[float]:
    """返回 X~Hypergeom(N,D,n) 在 0..n 上的概率。"""

    _validate_integer(population_size, "population_size", minimum=1)
    _validate_integer(defective_count, "defective_count", minimum=0)
    _validate_integer(sample_size, "sample_size", minimum=0)
    if defective_count > population_size:
        raise ValueError("defective_count 不能超过 population_size。")
    if sample_size > population_size:
        raise ValueError("sample_size 不能超过 population_size。")

    lower = max(0, sample_size - (population_size - defective_count))
    upper = min(sample_size, defective_count)
    denominator = _log_combination(population_size, sample_size)
    logs = [-math.inf] * (sample_size + 1)
    for defects in range(lower, upper + 1):
        logs[defects] = (
            _log_combination(defective_count, defects)
            + _log_combination(
                population_size - defective_count,
                sample_size - defects,
            )
            - denominator
        )
    return _normalise_log_probabilities(logs)


def cumulative_tails(probabilities: Sequence[float]) -> tuple[list[float], list[float]]:
    """返回包含端点的左尾 CDF 和右尾 SF: P(X>=x)。"""

    if not probabilities:
        raise ValueError("概率序列不能为空。")
    if any(value < -FLOAT_TOLERANCE for value in probabilities):
        raise ValueError("概率不能为负。")
    total = math.fsum(probabilities)
    if abs(total - 1.0) > 1e-10:
        raise ValueError("输入概率序列之和必须为 1。")

    left = list(accumulate(probabilities))
    right = [0.0] * len(probabilities)
    running = 0.0
    for index in range(len(probabilities) - 1, -1, -1):
        running += probabilities[index]
        right[index] = running
    left[-1] = 1.0
    right[0] = 1.0
    return left, right


def acceptable_defective_count(population_size: int, nominal_rate: float) -> int:
    """计算不使总体次品率超过标称值的最大整数次品数。"""

    _validate_integer(population_size, "population_size", minimum=1)
    _validate_probability(nominal_rate, "nominal_rate")
    product = Decimal(population_size) * Decimal(str(nominal_rate))
    return int(product.to_integral_value(rounding=ROUND_FLOOR))


def _find_cutoffs(
    accept_probabilities: Sequence[float],
    reject_probabilities: Sequence[float],
    accept_alpha: float,
    reject_alpha: float,
) -> tuple[
    int | None,
    int | None,
    list[float],
    list[float],
]:
    if len(accept_probabilities) != len(reject_probabilities):
        raise ValueError("接收和拒收分布必须定义在相同样本空间。")
    accept_cdf, _ = cumulative_tails(accept_probabilities)
    _, reject_sf = cumulative_tails(reject_probabilities)

    accept_cutoff = None
    for defects, probability in enumerate(accept_cdf):
        if probability <= accept_alpha + FLOAT_TOLERANCE:
            accept_cutoff = defects
        else:
            break

    reject_cutoff = None
    for defects, probability in enumerate(reject_sf):
        if probability <= reject_alpha + FLOAT_TOLERANCE:
            reject_cutoff = defects
            break

    return accept_cutoff, reject_cutoff, accept_cdf, reject_sf


def compute_thresholds(
    sample_size: int,
    nominal_rate: float = 0.10,
    reject_confidence: float = 0.95,
    accept_confidence: float = 0.90,
    *,
    model: str = "binomial",
    population_size: int | None = None,
) -> ThresholdResult:
    """计算固定样本量的接收、拒收临界值。"""

    _validate_integer(sample_size, "sample_size", minimum=1)
    _validate_probability(nominal_rate, "nominal_rate")
    _validate_probability(reject_confidence, "reject_confidence")
    _validate_probability(accept_confidence, "accept_confidence")
    accept_alpha = 1.0 - accept_confidence
    reject_alpha = 1.0 - reject_confidence

    d0: int | None = None
    d1: int | None = None
    if model == "binomial":
        if population_size is not None:
            raise ValueError("二项模型不使用 population_size。")
        probabilities = binomial_probabilities(sample_size, nominal_rate)
        accept_probabilities = probabilities
        reject_probabilities = probabilities
    elif model == "hypergeometric":
        if population_size is None:
            raise ValueError("超几何模型必须给出 population_size。")
        _validate_integer(population_size, "population_size", minimum=1)
        if sample_size > population_size:
            raise ValueError("sample_size 不能超过 population_size。")
        d0 = acceptable_defective_count(population_size, nominal_rate)
        d1 = d0 + 1
        reject_probabilities = hypergeometric_probabilities(
            population_size,
            d0,
            sample_size,
        )
        accept_probabilities = hypergeometric_probabilities(
            population_size,
            d1,
            sample_size,
        )
    else:
        raise ValueError("model 只能是 'binomial' 或 'hypergeometric'。")

    accept_cutoff, reject_cutoff, accept_cdf, reject_sf = _find_cutoffs(
        accept_probabilities,
        reject_probabilities,
        accept_alpha,
        reject_alpha,
    )

    accept_at = accept_cdf[accept_cutoff] if accept_cutoff is not None else None
    accept_after = (
        accept_cdf[accept_cutoff + 1]
        if accept_cutoff is not None and accept_cutoff < sample_size
        else None
    )
    reject_at = reject_sf[reject_cutoff] if reject_cutoff is not None else None
    reject_before = (
        reject_sf[reject_cutoff - 1]
        if reject_cutoff is not None and reject_cutoff > 0
        else None
    )

    accept_ok = accept_at is None or accept_at <= accept_alpha + FLOAT_TOLERANCE
    reject_ok = reject_at is None or reject_at <= reject_alpha + FLOAT_TOLERANCE
    accept_maximal_ok = (
        accept_cdf[0] > accept_alpha - FLOAT_TOLERANCE
        if accept_cutoff is None
        else accept_after is None or accept_after > accept_alpha - FLOAT_TOLERANCE
    )
    reject_minimal_ok = (
        reject_sf[-1] > reject_alpha - FLOAT_TOLERANCE
        if reject_cutoff is None
        else reject_before is None or reject_before > reject_alpha - FLOAT_TOLERANCE
    )
    disjoint_ok = (
        accept_cutoff is None
        or reject_cutoff is None
        or accept_cutoff < reject_cutoff
    )
    if not (
        accept_ok
        and reject_ok
        and accept_maximal_ok
        and reject_minimal_ok
        and disjoint_ok
    ):
        raise ArithmeticError("临界值未通过尾概率或判定域校验。")

    inconclusive_low = 0 if accept_cutoff is None else accept_cutoff + 1
    inconclusive_high = sample_size if reject_cutoff is None else reject_cutoff - 1
    if inconclusive_low > inconclusive_high:
        inconclusive_low = None
        inconclusive_high = None

    return ThresholdResult(
        model=model,
        population_size=population_size,
        sample_size=sample_size,
        nominal_defect_rate=nominal_rate,
        acceptable_defectives=d0,
        first_unacceptable_defectives=d1,
        accept_confidence=accept_confidence,
        reject_confidence=reject_confidence,
        accept_alpha=accept_alpha,
        reject_alpha=reject_alpha,
        accept_cutoff=accept_cutoff,
        reject_cutoff=reject_cutoff,
        accept_tail_at_cutoff=accept_at,
        accept_tail_after_cutoff=accept_after,
        reject_tail_at_cutoff=reject_at,
        reject_tail_before_cutoff=reject_before,
        inconclusive_low=inconclusive_low,
        inconclusive_high=inconclusive_high,
        accept_constraint_ok=accept_ok,
        reject_constraint_ok=reject_ok,
        accept_boundary_maximal_ok=accept_maximal_ok,
        reject_boundary_minimal_ok=reject_minimal_ok,
        disjoint_regions_ok=disjoint_ok,
    )


def search_thresholds(
    maximum_sample_size: int,
    nominal_rate: float = 0.10,
    reject_confidence: float = 0.95,
    accept_confidence: float = 0.90,
    *,
    model: str = "binomial",
    population_size: int | None = None,
) -> tuple[list[ThresholdResult], dict[str, int | None]]:
    """枚举固定 n，并返回三种最小样本量口径。"""

    _validate_integer(maximum_sample_size, "maximum_sample_size", minimum=1)
    if model == "hypergeometric":
        if population_size is None:
            raise ValueError("超几何搜索必须给出 population_size。")
        maximum_sample_size = min(maximum_sample_size, population_size)

    rows: list[ThresholdResult] = []
    first_accept = None
    first_reject = None
    first_both = None
    for sample_size in range(1, maximum_sample_size + 1):
        result = compute_thresholds(
            sample_size,
            nominal_rate,
            reject_confidence,
            accept_confidence,
            model=model,
            population_size=population_size,
        )
        rows.append(result)
        if first_accept is None and result.has_accept_region:
            first_accept = sample_size
        if first_reject is None and result.has_reject_region:
            first_reject = sample_size
        if first_both is None and result.has_both_regions:
            first_both = sample_size

    minimums = {
        "first_accept_sample_size": first_accept,
        "first_reject_sample_size": first_reject,
        "first_both_sample_size": first_both,
    }
    return rows, minimums


def classify_observation(result: ThresholdResult, observed_defects: int) -> dict[str, object]:
    """按已计算的固定样本边界判定观测结果。"""

    _validate_integer(observed_defects, "observed_defects", minimum=0)
    if observed_defects > result.sample_size:
        raise ValueError("observed_defects 不能超过 sample_size。")

    accept = result.accept_cutoff is not None and observed_defects <= result.accept_cutoff
    reject = result.reject_cutoff is not None and observed_defects >= result.reject_cutoff
    if accept and reject:
        raise ArithmeticError("观测值同时落入接收域和拒收域。")
    if accept:
        decision = "接收"
    elif reject:
        decision = "拒收"
    else:
        decision = "证据不足"

    if result.model == "binomial":
        accept_probabilities = binomial_probabilities(
            result.sample_size,
            result.nominal_defect_rate,
        )
        reject_probabilities = accept_probabilities
    else:
        if (
            result.population_size is None
            or result.acceptable_defectives is None
            or result.first_unacceptable_defectives is None
        ):
            raise ArithmeticError("超几何阈值缺少有限总体边界参数。")
        accept_probabilities = hypergeometric_probabilities(
            result.population_size,
            result.first_unacceptable_defectives,
            result.sample_size,
        )
        reject_probabilities = hypergeometric_probabilities(
            result.population_size,
            result.acceptable_defectives,
            result.sample_size,
        )
    accept_cdf, _ = cumulative_tails(accept_probabilities)
    _, reject_sf = cumulative_tails(reject_probabilities)
    return {
        "sample_size": result.sample_size,
        "observed_defects": observed_defects,
        "observed_defect_rate": observed_defects / result.sample_size,
        "decision": decision,
        "accept_cutoff": result.accept_cutoff,
        "reject_cutoff": result.reject_cutoff,
        "accept_left_tail_probability": accept_cdf[observed_defects],
        "reject_right_tail_probability": reject_sf[observed_defects],
    }


def _select_row(rows: Sequence[ThresholdResult], sample_size: int) -> ThresholdResult:
    if sample_size < 1 or sample_size > len(rows):
        raise ValueError("指定 sample_size 不在已计算范围内。")
    result = rows[sample_size - 1]
    if result.sample_size != sample_size:
        raise ArithmeticError("阈值表的样本量索引不一致。")
    return result


def build_sensitivity_rows(
    nominal_rate: float,
    reject_confidence: float,
    accept_confidence: float,
    maximum_sample_size: int,
    *,
    population_size: int | None = None,
) -> list[dict[str, object]]:
    """生成单因素信度灵敏度数据。"""

    reject_grid = sorted({0.90, 0.925, reject_confidence, 0.975, 0.99})
    accept_grid = sorted({0.80, 0.85, accept_confidence, 0.95, 0.99})
    model_specs = [("binomial", None)]
    if population_size is not None:
        model_specs.append(("hypergeometric", population_size))

    rows: list[dict[str, object]] = []
    for model, current_population_size in model_specs:
        for confidence in reject_grid:
            _, minimums = search_thresholds(
                maximum_sample_size,
                nominal_rate,
                confidence,
                accept_confidence,
                model=model,
                population_size=current_population_size,
            )
            rows.append(
                {
                    "model": model,
                    "population_size": current_population_size,
                    "varied_parameter": "reject_confidence",
                    "varied_confidence": confidence,
                    "fixed_other_confidence": accept_confidence,
                    **minimums,
                }
            )
        for confidence in accept_grid:
            _, minimums = search_thresholds(
                maximum_sample_size,
                nominal_rate,
                reject_confidence,
                confidence,
                model=model,
                population_size=current_population_size,
            )
            rows.append(
                {
                    "model": model,
                    "population_size": current_population_size,
                    "varied_parameter": "accept_confidence",
                    "varied_confidence": confidence,
                    "fixed_other_confidence": reject_confidence,
                    **minimums,
                }
            )
    return rows


def build_population_sensitivity_rows(
    population_sizes: Iterable[int],
    nominal_rate: float,
    reject_confidence: float,
    accept_confidence: float,
    maximum_sample_size: int,
    binomial_first_both: int | None,
) -> list[dict[str, object]]:
    """生成有限总体量 N 对超几何方案影响的数据。"""

    rows: list[dict[str, object]] = []
    for population_size in sorted(set(population_sizes)):
        _validate_integer(population_size, "population_size", minimum=1)
        thresholds, minimums = search_thresholds(
            min(maximum_sample_size, population_size),
            nominal_rate,
            reject_confidence,
            accept_confidence,
            model="hypergeometric",
            population_size=population_size,
        )
        first_both = minimums["first_both_sample_size"]
        selected = None if first_both is None else _select_row(thresholds, first_both)
        rows.append(
            {
                "model": "hypergeometric",
                "population_size": population_size,
                "acceptable_defectives": acceptable_defective_count(
                    population_size,
                    nominal_rate,
                ),
                **minimums,
                "accept_cutoff_at_first_both": (
                    None if selected is None else selected.accept_cutoff
                ),
                "reject_cutoff_at_first_both": (
                    None if selected is None else selected.reject_cutoff
                ),
                "binomial_first_both_sample_size": binomial_first_both,
                "difference_from_binomial": (
                    None
                    if first_both is None or binomial_first_both is None
                    else first_both - binomial_first_both
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialised = list(rows)
    if not materialised:
        raise ValueError("没有可写入 CSV 的数据。")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialised[0].keys()))
        writer.writeheader()
        writer.writerows(materialised)


def _configure_plot_style() -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    style_path = (
        Path(__file__).resolve().parents[1]
        / "figures"
        / "style"
        / "cumcm.mplstyle"
    )
    if style_path.exists():
        plt.style.use(style_path)

    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ["SimSun", "STSong", "Microsoft YaHei", "Noto Serif CJK SC"]
    selected = next((name for name in candidates if name in available), "DejaVu Serif")
    matplotlib.rcParams["font.family"] = selected
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42


def _save_figure(figure, pdf_path: Path) -> None:
    """Save a formal line figure as traceable PDF and SVG outputs."""

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, format="pdf", bbox_inches="tight")
    figure.savefig(pdf_path.with_suffix(".svg"), format="svg", bbox_inches="tight")


def plot_decision_boundaries(
    model_rows: dict[str, Sequence[ThresholdResult]],
    figure_path: Path,
) -> None:
    """绘制样本量与判定临界次品数的关系。"""

    _configure_plot_style()
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    colours = {"binomial": "#0072B2", "hypergeometric": "#D55E00"}
    model_names = {"binomial": "二项", "hypergeometric": "超几何"}
    for model, rows in model_rows.items():
        sample_sizes = [row.sample_size for row in rows]
        accept_values = [
            math.nan if row.accept_cutoff is None else row.accept_cutoff for row in rows
        ]
        reject_values = [
            math.nan if row.reject_cutoff is None else row.reject_cutoff for row in rows
        ]
        colour = colours[model]
        axis.plot(
            sample_sizes,
            accept_values,
            color=colour,
            linestyle="-",
            linewidth=1.8,
            label=f"{model_names[model]}：接收上界",
        )
        axis.plot(
            sample_sizes,
            reject_values,
            color=colour,
            linestyle="--",
            linewidth=1.8,
            label=f"{model_names[model]}：拒收下界",
        )
    axis.set_xlabel("固定抽样数 $n$")
    axis.set_ylabel("样本次品数临界值")
    axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    _save_figure(figure, figure_path)
    plt.close(figure)


def plot_confidence_sensitivity(
    rows: Sequence[dict[str, object]],
    figure_path: Path,
) -> None:
    """绘制信度变化与首次同时存在双判定域的样本量。"""

    _configure_plot_style()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    parameter_names = ["reject_confidence", "accept_confidence"]
    x_labels = ["拒收信度", "接收信度"]
    minimum_fields = ["first_reject_sample_size", "first_accept_sample_size"]
    y_labels = ["首次存在拒收域的 $n$", "首次存在接收域的 $n$"]
    models = sorted({str(row["model"]) for row in rows})
    colours = {"binomial": "#0072B2", "hypergeometric": "#D55E00"}
    labels = {"binomial": "二项", "hypergeometric": "超几何"}
    for axis, parameter, x_label, minimum_field, y_label in zip(
        axes,
        parameter_names,
        x_labels,
        minimum_fields,
        y_labels,
    ):
        for model in models:
            subset = [
                row
                for row in rows
                if row["model"] == model and row["varied_parameter"] == parameter
            ]
            subset.sort(key=lambda row: float(row["varied_confidence"]))
            x_values = [float(row["varied_confidence"]) for row in subset]
            y_values = [
                math.nan
                if row[minimum_field] is None
                else int(row[minimum_field])
                for row in subset
            ]
            axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.8,
                color=colours[model],
                label=labels[model],
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, figure_path)
    plt.close(figure)


def plot_population_sensitivity(
    rows: Sequence[dict[str, object]],
    figure_path: Path,
) -> None:
    """绘制有限总体量与超几何最小统一抽样数。"""

    _configure_plot_style()
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: int(row["population_size"]))
    x_values = [int(row["population_size"]) for row in ordered]
    y_values = [
        math.nan
        if row["first_both_sample_size"] is None
        else int(row["first_both_sample_size"])
        for row in ordered
    ]
    binomial_value = ordered[0]["binomial_first_both_sample_size"]

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    axis.plot(
        x_values,
        y_values,
        color="#D55E00",
        marker="o",
        linewidth=1.8,
        label="超几何有限总体",
    )
    if binomial_value is not None:
        axis.axhline(
            int(binomial_value),
            color="#0072B2",
            linestyle="--",
            linewidth=1.8,
            label="二项极限",
        )
    axis.set_xscale("log")
    axis.set_xlabel("有限总体量 $N$（对数坐标）")
    axis.set_ylabel("首次同时存在接收域与拒收域的 $n$")
    axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, figure_path)
    plt.close(figure)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="2024 年国赛 B 题问题一：二项主模型与超几何有限总体校核。"
    )
    parser.add_argument("--defect-rate", type=float, default=0.10, help="标称次品率。")
    parser.add_argument(
        "--reject-confidence",
        type=float,
        default=0.95,
        help="认定超过标称值并拒收的信度。",
    )
    parser.add_argument(
        "--accept-confidence",
        type=float,
        default=0.90,
        help="认定不超过标称值并接收的信度。",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        help="有限总体量 N；给出后增加超几何无放回校核。",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="要重点输出的固定样本量；默认使用二项模型首次双域样本量。",
    )
    parser.add_argument(
        "--observed-defects",
        type=int,
        help="指定样本中的观测次品数，用于直接判定。",
    )
    parser.add_argument(
        "--max-sample-size",
        type=int,
        default=200,
        help="阈值搜索上限，默认 200；超几何模式不超过 N。",
    )
    parser.add_argument(
        "--population-grid",
        type=int,
        nargs="+",
        default=[25, 50, 100, 200, 500, 1000, 5000],
        help="用于有限总体灵敏度分析的一组 N。",
    )
    parser.add_argument("--output-dir", type=Path, help="CSV/JSON 输出目录。")
    parser.add_argument("--figure-dir", type=Path, help="PDF/SVG Figure Pack 输出目录。")
    parser.add_argument("--no-plots", action="store_true", help="不生成 PDF/SVG 图。")
    return parser


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def main(argv: Sequence[str] | None = None) -> int:
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = (args.output_dir or project_root / "results").resolve()
    figure_dir = (args.figure_dir or project_root / "figures" / "q1").resolve()
    figure_data_dir = figure_dir / "data"

    _validate_probability(args.defect_rate, "defect_rate")
    _validate_probability(args.reject_confidence, "reject_confidence")
    _validate_probability(args.accept_confidence, "accept_confidence")
    _validate_integer(args.max_sample_size, "max_sample_size", minimum=1)
    if args.population_size is not None:
        _validate_integer(args.population_size, "population_size", minimum=1)
    if args.sample_size is not None:
        _validate_integer(args.sample_size, "sample_size", minimum=1)

    search_limit = args.max_sample_size
    if args.sample_size is not None:
        search_limit = max(search_limit, args.sample_size)

    binomial_rows, binomial_minimums = search_thresholds(
        search_limit,
        args.defect_rate,
        args.reject_confidence,
        args.accept_confidence,
        model="binomial",
    )
    model_rows: dict[str, Sequence[ThresholdResult]] = {"binomial": binomial_rows}
    model_minimums: dict[str, dict[str, int | None]] = {
        "binomial": binomial_minimums
    }

    if args.population_size is not None:
        hyper_limit = min(search_limit, args.population_size)
        hyper_rows, hyper_minimums = search_thresholds(
            hyper_limit,
            args.defect_rate,
            args.reject_confidence,
            args.accept_confidence,
            model="hypergeometric",
            population_size=args.population_size,
        )
        model_rows["hypergeometric"] = hyper_rows
        model_minimums["hypergeometric"] = hyper_minimums

    selection_model = "hypergeometric" if args.population_size is not None else "binomial"
    default_selected_n = model_minimums[selection_model]["first_both_sample_size"]
    if args.sample_size is None and default_selected_n is None:
        raise ValueError(
            "搜索上限内未找到同时存在接收域和拒收域的样本量；请增大 --max-sample-size。"
        )
    selected_n = args.sample_size or int(default_selected_n)
    if args.population_size is not None and selected_n > args.population_size:
        raise ValueError("sample_size 不能超过 population_size。")
    selected_rows: dict[str, ThresholdResult] = {
        model: _select_row(rows, selected_n) for model, rows in model_rows.items()
    }

    threshold_rows = [
        result.to_row()
        for model in ("binomial", "hypergeometric")
        if model in model_rows
        for result in model_rows[model]
    ]
    sensitivity_rows = build_sensitivity_rows(
        args.defect_rate,
        args.reject_confidence,
        args.accept_confidence,
        search_limit,
        population_size=args.population_size,
    )
    population_grid = set(args.population_grid)
    if args.population_size is not None:
        population_grid.add(args.population_size)
    population_sensitivity_rows = build_population_sensitivity_rows(
        population_grid,
        args.defect_rate,
        args.reject_confidence,
        args.accept_confidence,
        search_limit,
        binomial_minimums["first_both_sample_size"],
    )

    threshold_path = output_dir / "q1_thresholds.csv"
    sensitivity_path = output_dir / "q1_sensitivity.csv"
    population_sensitivity_path = output_dir / "q1_population_sensitivity.csv"
    summary_path = output_dir / "q1_summary.json"
    _write_csv(threshold_path, threshold_rows)
    _write_csv(sensitivity_path, sensitivity_rows)
    _write_csv(population_sensitivity_path, population_sensitivity_rows)
    _write_csv(figure_data_dir / "q1_thresholds.csv", threshold_rows)
    _write_csv(figure_data_dir / "q1_sensitivity.csv", sensitivity_rows)
    _write_csv(
        figure_data_dir / "q1_population_sensitivity.csv",
        population_sensitivity_rows,
    )

    observed_results = None
    if args.observed_defects is not None:
        observed_results = {
            model: classify_observation(result, args.observed_defects)
            for model, result in selected_rows.items()
        }

    all_enumerated_thresholds_valid = all(
        result.accept_constraint_ok
        and result.reject_constraint_ok
        and result.accept_boundary_maximal_ok
        and result.reject_boundary_minimal_ok
        and result.disjoint_regions_ok
        for rows in model_rows.values()
        for result in rows
    )
    selected_constraint_checks = {
        model: {
            "accept_tail_probability_ok": result.accept_constraint_ok,
            "reject_tail_probability_ok": result.reject_constraint_ok,
            "accept_cutoff_is_maximal": result.accept_boundary_maximal_ok,
            "reject_cutoff_is_minimal": result.reject_boundary_minimal_ok,
            "decision_regions_disjoint": result.disjoint_regions_ok,
        }
        for model, result in selected_rows.items()
    }

    summary = {
        "problem": "2024 年国赛 B 题问题一",
        "method": "固定样本精确单侧检验；二项主模型，超几何有限总体修正",
        "parameters": {
            "nominal_defect_rate": args.defect_rate,
            "reject_confidence": args.reject_confidence,
            "accept_confidence": args.accept_confidence,
            "population_size": args.population_size,
            "search_limit": search_limit,
            "selected_sample_size": selected_n,
        },
        "scope_note": (
            "题面未给出 N，正式数值采用二项模型。"
            if args.population_size is None
            else "已按给定 N 同时计算超几何无放回有限总体修正。"
        ),
        "minimum_sample_sizes": model_minimums,
        "selected_thresholds": {
            model: result.to_row() for model, result in selected_rows.items()
        },
        "observed_results": observed_results,
        "constraint_checks": {
            "all_enumerated_thresholds_valid": all_enumerated_thresholds_valid,
            "selected_sample": selected_constraint_checks,
        },
        "warnings": [
            "首次可拒收样本量只表示存在极端拒收结果，不表示具有指定检验功效。",
            "阈值适用于预先固定样本量的一次检验，不得直接用于逐样本反复窥探。",
            "若要设计功效样本量或真正序贯检验，还需给出备择次品率或错误率分配。",
        ],
        "outputs": {
            "thresholds_csv": _portable_path(threshold_path, project_root),
            "sensitivity_csv": _portable_path(sensitivity_path, project_root),
            "population_sensitivity_csv": _portable_path(
                population_sensitivity_path,
                project_root,
            ),
            "decision_boundaries_pdf": _portable_path(
                figure_dir / "q1_decision_boundaries.pdf",
                project_root,
            ),
            "confidence_sensitivity_pdf": _portable_path(
                figure_dir / "q1_confidence_sensitivity.pdf",
                project_root,
            ),
            "population_sensitivity_pdf": _portable_path(
                figure_dir / "q1_population_sensitivity.pdf",
                project_root,
            ),
            "figure_data_dir": _portable_path(figure_data_dir, project_root),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if not args.no_plots:
        plot_decision_boundaries(
            model_rows,
            figure_dir / "q1_decision_boundaries.pdf",
        )
        plot_confidence_sensitivity(
            sensitivity_rows,
            figure_dir / "q1_confidence_sensitivity.pdf",
        )
        plot_population_sensitivity(
            population_sensitivity_rows,
            figure_dir / "q1_population_sensitivity.pdf",
        )

    print("问题一固定样本抽检方案已计算完成。")
    for model, minimums in model_minimums.items():
        label = "二项主模型" if model == "binomial" else "超几何有限总体模型"
        selected = selected_rows[model]
        print(
            f"{label}: 首次接收 n={minimums['first_accept_sample_size']}, "
            f"首次拒收 n={minimums['first_reject_sample_size']}, "
            f"首次双域 n={minimums['first_both_sample_size']}。"
        )
        print(
            f"  指定 n={selected.sample_size}: 接收 x<={selected.accept_cutoff}, "
            f"拒收 x>={selected.reject_cutoff}, "
            f"其余为证据不足。"
        )
    if observed_results is not None:
        for model, result in observed_results.items():
            print(f"{model} 对 x={args.observed_defects} 的判定：{result['decision']}。")
    print(f"结果目录：{output_dir}")
    if not args.no_plots:
        print(f"图表目录：{figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
