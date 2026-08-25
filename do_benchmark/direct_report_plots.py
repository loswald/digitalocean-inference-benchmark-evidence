"""Publication plots for the direct DigitalOcean benchmark.

Every figure uses matched estimands, explicit sample sizes, and visible
censoring. Chronological lines are used only inside one exact controller run;
figures never connect categorical workload shapes, pool unmatched regimes into
a ranking, or treat a missing cell as zero.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#7A7A7A",
    "light": "#D9E2EA",
}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _short_endpoint(value: str) -> str:
    replacements = {
        "nvidia-nemotron-3-super-120b": "nemotron-3-super-120b",
        "nemotron-3-ultra-550b": "nemotron-3-ultra-550b",
        "arcee-trinity-large-thinking": "arcee-trinity-thinking",
        "deepseek-v4-flash-0731": "deepseek-v4-flash-0731",
    }
    return replacements.get(value, value)


def _save(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")


def _setup() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )
    return plt


def _endpoint_grid(plt: Any, endpoints: Sequence[str], *, height: float = 3.0):
    columns = 3
    row_count = math.ceil(len(endpoints) / columns)
    fig, axes = plt.subplots(
        row_count,
        columns,
        figsize=(12.2, max(4.4, height * row_count)),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.78 if row_count == 1 else 0.90,
        bottom=0.25 if row_count == 1 else 0.12,
        hspace=0.46,
        wspace=0.28,
    )
    for axis in list(axes.flat)[len(endpoints) :]:
        axis.set_visible(False)
    return fig, axes


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _rank_interval(values: Sequence[float], probability: float) -> tuple[float, float]:
    """Approximate request-level 95% quantile interval using order-statistic ranks."""

    ordered = sorted(values)
    if len(ordered) < 2:
        value = ordered[0] if ordered else math.nan
        return value, value
    centre = probability * (len(ordered) - 1)
    rank_se = math.sqrt(len(ordered) * probability * (1 - probability))
    lower = max(0, math.floor(centre - 1.96 * rank_se))
    upper = min(len(ordered) - 1, math.ceil(centre + 1.96 * rank_se))
    return ordered[lower], ordered[upper]


def _binned_quantile_series(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_field: str,
    y_field: str,
    boundaries: Sequence[float],
    minimum_bin_size: int = 6,
) -> list[dict[str, float | int]]:
    points: list[dict[str, float | int]] = []
    for lower, upper in zip(boundaries, boundaries[1:]):
        members = [
            row
            for row in rows
            if (x := _number(row.get(x_field))) is not None
            and (y := _number(row.get(y_field))) is not None
            and x > lower
            and x <= upper
            and y > 0
        ]
        if len(members) < minimum_bin_size:
            continue
        x_values = [float(row[x_field]) for row in members]
        y_values = [float(row[y_field]) for row in members]
        p50 = _quantile(y_values, 0.50)
        p95 = _quantile(y_values, 0.95)
        p50_low, p50_high = _rank_interval(y_values, 0.50)
        p95_low, p95_high = _rank_interval(y_values, 0.95)
        points.append(
            {
                "x": _quantile(x_values, 0.50),
                "p50": p50,
                "p50_low": p50_low,
                "p50_high": p50_high,
                "p95": p95,
                "p95_low": p95_low,
                "p95_high": p95_high,
                "n": len(members),
            }
        )
    return points


def _draw_binned_quantiles(
    axis: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    x_field: str,
    y_field: str,
    boundaries: Sequence[float],
    color: str,
) -> None:
    points = _binned_quantile_series(
        rows,
        x_field=x_field,
        y_field=y_field,
        boundaries=boundaries,
    )
    if not points:
        return
    x = [float(point["x"]) for point in points]
    p50 = [float(point["p50"]) for point in points]
    p95 = [float(point["p95"]) for point in points]
    axis.errorbar(
        x,
        p50,
        yerr=[
            [value - float(point["p50_low"]) for value, point in zip(p50, points)],
            [float(point["p50_high"]) - value for value, point in zip(p50, points)],
        ],
        color=color,
        linewidth=1.5,
        marker="o",
        markersize=3.5,
        capsize=1.5,
        zorder=4,
    )
    axis.errorbar(
        x,
        p95,
        yerr=[
            [value - float(point["p95_low"]) for value, point in zip(p95, points)],
            [float(point["p95_high"]) - value for value, point in zip(p95, points)],
        ],
        color=color,
        linewidth=1.0,
        linestyle="--",
        marker="^",
        markersize=3,
        capsize=1.2,
        alpha=0.9,
        zorder=4,
    )


def _coverage_plot(
    plt: Any,
    path: Path,
    endpoints: Sequence[str],
    dimensions: Sequence[str],
    coverage: Sequence[Mapping[str, Any]],
) -> bool:
    statuses = [
        "completed",
        "unsupported",
        "operational_failure",
        "degraded",
        "inconclusive",
        "skipped",
        "untested",
    ]
    colors = [
        PALETTE["green"],
        PALETTE["purple"],
        PALETTE["red"],
        PALETTE["orange"],
        "#F0C419",
        PALETTE["gray"],
        PALETTE["light"],
    ]
    status_index = {name: index for index, name in enumerate(statuses)}
    status_symbols = {
        "completed": "C",
        "unsupported": "U",
        "operational_failure": "F",
        "degraded": "D",
        "inconclusive": "?",
        "skipped": "S",
        "untested": "·",
    }
    lookup = {
        (str(row.get("endpoint_id")), str(row.get("coverage_dimension"))): str(
            row.get("status") or "untested"
        )
        for row in coverage
    }
    if not lookup:
        return False
    matrix = [
        [
                status_index.get(lookup.get((endpoint, dimension), "untested"), 6)
            for dimension in dimensions
        ]
        for endpoint in endpoints
    ]
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(13.2, 8.0))
    ax.imshow(matrix, aspect="auto", cmap=ListedColormap(colors), vmin=-0.5, vmax=6.5)
    ax.set_xticks(
        range(len(dimensions)),
        [value.replace("_", " ") for value in dimensions],
        rotation=42,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(
        range(len(endpoints)), [_short_endpoint(value) for value in endpoints]
    )
    ax.grid(False)
    counts = Counter(lookup.values())
    for endpoint_index, endpoint in enumerate(endpoints):
        for dimension_index, dimension in enumerate(dimensions):
            status = lookup.get((endpoint, dimension), "untested")
            foreground = (
                "white"
                if status in {"completed", "unsupported", "operational_failure"}
                else "#17212B"
            )
            ax.text(
                dimension_index,
                endpoint_index,
                status_symbols.get(status, "?"),
                ha="center",
                va="center",
                color=foreground,
                fontsize=7.2,
                fontweight="bold",
            )
    ax.set_title(
        "Evidence completion matrix - status is not endpoint quality",
        loc="left",
        pad=14,
    )
    ax.text(
        0,
        1.01,
        "Each cell describes whether the planned experiment is resolved; capability outcome is shown separately.",
        transform=ax.transAxes,
        color="#4E5965",
        fontsize=9,
    )
    ax.legend(
        handles=[
            Patch(
                color=color,
                label=f"{status_symbols[status]} = {status} (n={counts.get(status, 0)})",
            )
            for status, color in zip(statuses, colors)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
    )
    _save(fig, path)
    plt.close(fig)
    return True


def _ttft_plot(
    plt: Any,
    path: Path,
    endpoints: Sequence[str],
    requests: Sequence[Mapping[str, Any]],
) -> bool:
    rows = [
        row
        for row in requests
        if row.get("reconciliation_status") == "matched"
        and not row.get("multi_choice")
        and (_number(row.get("input_tokens")) or 0) > 0
        and (_number(row.get("ttft_seconds")) or 0) > 0
    ]
    shown = [
        endpoint
        for endpoint in endpoints
        if any(row.get("endpoint_id") == endpoint for row in rows)
    ]
    if not shown:
        return False
    state_styles = {
        "cache_miss_observed": (PALETTE["blue"], "Observed cache miss"),
        "cache_hit_observed": (PALETTE["green"], "Observed cache hit"),
        "not_reported_unknown": (PALETTE["gray"], "Cache state not reported"),
    }
    token_boundaries = (
        0,
        64,
        256,
        1_024,
        4_096,
        16_384,
        65_536,
        262_144,
        1_048_576,
        math.inf,
    )
    fig, axes = _endpoint_grid(plt, shown, height=2.8)
    fig.subplots_adjust(bottom=0.18)
    for endpoint, axis in zip(shown, axes.flat):
        endpoint_rows = [row for row in rows if row.get("endpoint_id") == endpoint]
        for state, (color, _) in state_styles.items():
            values = [
                row for row in endpoint_rows if str(row.get("cache_state")) == state
            ]
            if values:
                axis.scatter(
                    [float(row["input_tokens"]) for row in values],
                    [float(row["ttft_seconds"]) for row in values],
                    s=16,
                    alpha=0.55,
                    color=color,
                    edgecolors="none",
                )
                _draw_binned_quantiles(
                    axis,
                    values,
                    x_field="input_tokens",
                    y_field="ttft_seconds",
                    boundaries=token_boundaries,
                    color=color,
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(
            f"{_short_endpoint(endpoint)}  (n={len(endpoint_rows)})", loc="left"
        )
    fig.suptitle(
        "Streaming TTFT versus server-reported input length, stratified by cache state",
        y=0.975,
    )
    label_y = 0.15 if len(shown) <= 3 else 0.09
    fig.supxlabel("server-reported prompt tokens (log scale)", y=label_y)
    fig.supylabel("time to first streamed content event, seconds (log scale)", x=0.018)
    present_states = {
        str(row.get("cache_state")) for row in rows if row.get("cache_state")
    }
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=label)
        for state, (color, label) in state_styles.items()
        if state in present_states
    ]
    handles.extend(
        [
            plt.Line2D(
                [],
                [],
                marker="o",
                color="#17212B",
                label="binned p50 + approx. 95% rank interval",
            ),
            plt.Line2D(
                [],
                [],
                marker="^",
                linestyle="--",
                color="#17212B",
                label="binned p95 + approx. 95% rank interval",
            ),
        ]
    )
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005))
    _save(fig, path)
    plt.close(fig)
    return True


def _output_proxy_plot(
    plt: Any,
    path: Path,
    endpoints: Sequence[str],
    requests: Sequence[Mapping[str, Any]],
) -> bool:
    from matplotlib.ticker import LogLocator, NullFormatter

    rows = [
        row
        for row in requests
        if row.get("reconciliation_status") == "matched"
        and not row.get("multi_choice")
        and (_number(row.get("output_tokens")) or 0) > 0
        and (_number(row.get("post_ttft_output_tokens_per_second_proxy")) or 0) > 0
    ]
    shown = [
        endpoint
        for endpoint in endpoints
        if any(row.get("endpoint_id") == endpoint for row in rows)
    ]
    if not shown:
        return False

    def load_class(row: Mapping[str, Any]) -> str:
        phase = str(row.get("phase") or "").casefold()
        if "baseline" in phase or "low_load" in phase or "paired_low" in phase:
            return "low"
        if row.get("source_kind") in {"direct_aimd", "direct_soak"}:
            return "loaded"
        return "other"

    styles = {
        "low": (PALETTE["green"], "Low load"),
        "loaded": (PALETTE["orange"], "AIMD / soak load"),
        "other": (PALETTE["gray"], "Other measured request"),
    }
    token_boundaries = (
        0,
        16,
        64,
        256,
        1_024,
        4_096,
        16_384,
        65_536,
        262_144,
        math.inf,
    )
    fig, axes = _endpoint_grid(plt, shown, height=2.8)
    fig.subplots_adjust(bottom=0.18)
    for endpoint, axis in zip(shown, axes.flat):
        endpoint_rows = [row for row in rows if row.get("endpoint_id") == endpoint]
        for kind, (color, _) in styles.items():
            values = [row for row in endpoint_rows if load_class(row) == kind]
            if values:
                axis.scatter(
                    [float(row["output_tokens"]) for row in values],
                    [
                        float(row["post_ttft_output_tokens_per_second_proxy"])
                        for row in values
                    ],
                    s=16,
                    alpha=0.50,
                    color=color,
                    edgecolors="none",
                )
                _draw_binned_quantiles(
                    axis,
                    values,
                    x_field="output_tokens",
                    y_field="post_ttft_output_tokens_per_second_proxy",
                    boundaries=token_boundaries,
                    color=color,
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        # Narrow endpoint-specific ranges otherwise cause Matplotlib to label
        # every 2x/4x/6x log subdivision.  Those overlapping labels made the
        # old figure harder to read than the data.  Keep only powers of ten as
        # labelled major ticks and retain unlabelled minor grid positions.
        for axis_dimension in (axis.xaxis, axis.yaxis):
            axis_dimension.set_major_locator(LogLocator(base=10, numticks=5))
            axis_dimension.set_minor_locator(
                LogLocator(base=10, subs=(2, 3, 4, 5, 6, 7, 8, 9), numticks=12)
            )
            axis_dimension.set_minor_formatter(NullFormatter())
        axis.set_title(
            f"{_short_endpoint(endpoint)}  (n={len(endpoint_rows)})", loc="left"
        )
    fig.suptitle(
        "Billed completion-token service-output proxy versus realized output length",
        y=0.975,
    )
    label_y = 0.15 if len(shown) <= 3 else 0.09
    fig.supxlabel("server-reported billed completion tokens (log scale)", y=label_y)
    fig.supylabel("tokens / (request end - streamed TTFT), log scale", x=0.018)
    present_load_classes = {load_class(row) for row in rows}
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=label)
        for kind, (color, label) in styles.items()
        if kind in present_load_classes
    ]
    handles.extend(
        [
            plt.Line2D(
                [],
                [],
                marker="o",
                color="#17212B",
                label="binned p50 + approx. 95% rank interval",
            ),
            plt.Line2D(
                [],
                [],
                marker="^",
                linestyle="--",
                color="#17212B",
                label="binned p95 + approx. 95% rank interval",
            ),
        ]
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.005),
    )
    _save(fig, path)
    plt.close(fig)
    return True


def _audit_plot(
    plt: Any,
    path: Path,
    audit_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> bool:
    legacy = sorted(
        value
        for row in audit_rows
        if (value := _number(row.get("legacy_sse_chunk_span_output_tps_proxy")))
        is not None
        and value > 0
    )
    corrected = sorted(
        value
        for row in audit_rows
        if (value := _number(row.get("post_ttft_output_tps_proxy"))) is not None
        and value > 0
    )
    corrected_one_second = sorted(
        value
        for row in audit_rows
        if (value := _number(row.get("post_ttft_output_tps_proxy"))) is not None
        and value > 0
        and (request_seconds := _number(row.get("request_seconds"))) is not None
        and (ttft_seconds := _number(row.get("streamed_ttft_seconds"))) is not None
        and request_seconds - ttft_seconds >= 1.0
    )
    if not legacy and not corrected:
        return False
    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1.45, 1]}
    )
    for values, color, label in (
        (legacy, PALETTE["red"], "Legacy SSE-chunk-span proxy (invalid as decode TPS)"),
        (corrected, PALETTE["blue"], "Corrected post-TTFT end-to-end proxy"),
    ):
        if values:
            probabilities = [(index + 1) / len(values) for index in range(len(values))]
            axes[0].plot(
                values,
                probabilities,
                color=color,
                linewidth=2,
                label=f"{label}; n={len(values):,}",
            )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("reported/proxy billed completion tokens per second (log scale)")
    axes[0].set_ylabel("empirical cumulative fraction")
    axes[0].set_ylim(0, 1.01)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title("Same requests, different timing denominator", loc="left")
    axes[1].axis("off")
    lines = [
        "Why the old extremes were not decoder measurements",
        "",
        f"Legacy observations: {int(summary.get('legacy_sse_proxy_observations') or 0):,}",
        f"Legacy >=1,000 tokens/s proxy: {int(summary.get('legacy_sse_proxy_at_least_1000') or 0):,}",
        f"Legacy >=10,000 tokens/s proxy: {int(summary.get('legacy_sse_proxy_at_least_10000') or 0):,}",
        f"Legacy >=100,000 tokens/s proxy: {int(summary.get('legacy_sse_proxy_at_least_100000') or 0):,}",
        f"Legacy maximum: {float(summary.get('legacy_sse_proxy_max') or 0):,.0f} tokens/s proxy",
        "",
        f"Corrected observations: {int(summary.get('corrected_post_ttft_proxy_observations') or 0):,}",
        f"Sub-100 ms intervals explicitly censored: {int(summary.get('sub_100ms_post_ttft_intervals_censored') or 0):,}",
        f"Corrected median: {float(summary.get('corrected_post_ttft_proxy_median') or 0):,.2f} tokens/s proxy",
        f"Corrected maximum retained: {float(summary.get('corrected_post_ttft_proxy_max') or 0):,.2f} tokens/s proxy",
        *(
            [
                "",
                "One-second denominator sensitivity audit:",
                f"n={len(corrected_one_second):,}; median={statistics.median(corrected_one_second):,.2f}",
                (
                    "p99="
                    f"{corrected_one_second[max(0, math.ceil(0.99 * len(corrected_one_second)) - 1)]:,.2f}; "
                    f"maximum={max(corrected_one_second):,.2f} tokens/s proxy"
                ),
            ]
            if corrected_one_second
            else []
        ),
        "",
        "SSE events are transport chunks, not tokens.",
        "No values were silently trimmed or winsorized.",
        "Sensitivity values are an audit, not a pooled endpoint comparison.",
    ]
    axes[1].text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=8.8,
        linespacing=1.28,
    )
    fig.suptitle(
        "Metric audit: legacy chunk timing versus corrected client-observed proxy",
        y=1.02,
    )
    _save(fig, path)
    plt.close(fig)
    return True


def _capacity_plots(
    plt: Any,
    chart_dir: Path,
    endpoints: Sequence[str],
    epochs: Sequence[Mapping[str, Any]],
) -> list[Path]:
    from matplotlib.ticker import LogLocator, NullFormatter

    rows = [
        row
        for row in epochs
        if row.get("source_kind") == "direct_aimd"
        and (_number(row.get("offered_rpm")) or 0) > 0
        and _number(row.get("achieved_rpm")) is not None
    ]
    paths: list[Path] = []
    for shape in sorted({str(row.get("shape") or "unspecified") for row in rows}):
        members = [
            row for row in rows if str(row.get("shape") or "unspecified") == shape
        ]
        shown = [
            endpoint
            for endpoint in endpoints
            if any(row.get("endpoint_id") == endpoint for row in members)
        ]
        if not shown:
            continue
        fig, axes = _endpoint_grid(plt, shown, height=2.65)
        maxima: list[float] = []
        for row in members:
            maxima.extend([float(row["offered_rpm"]), float(row["achieved_rpm"])])
        shared_max = max(maxima, default=1.0) * 1.12
        positive = [value for value in maxima if value > 0]
        shared_min = max(0.5, min(positive, default=1.0) / 1.5)
        for endpoint, axis in zip(shown, axes.flat):
            values = [row for row in members if row.get("endpoint_id") == endpoint]
            for row in values:
                valid = bool(row.get("valid_for_capacity"))
                healthy = row.get("healthy") is True
                color = (
                    PALETTE["green"]
                    if valid and healthy
                    else (PALETTE["red"] if valid else PALETTE["gray"])
                )
                marker = "o" if valid and healthy else ("X" if valid else "D")
                axis.scatter(
                    float(row["offered_rpm"]),
                    float(row["achieved_rpm"]),
                    marker=marker,
                    color=color,
                    s=28,
                    alpha=0.72,
                )
            axis.plot(
                [shared_min, shared_max],
                [shared_min, shared_max],
                linestyle="--",
                linewidth=0.8,
                color="#9AA5B1",
            )
            # Orders-of-magnitude differences across endpoint quotas made a
            # shared linear scale collapse most models into the origin.  A
            # shared log offered axis plus a symlog achieved axis preserves
            # cross-endpoint comparability while keeping zero-goodput epochs.
            axis.set_xscale("log")
            axis.set_yscale("symlog", linthresh=1.0, linscale=0.7)
            axis.set_xlim(shared_min, shared_max)
            axis.set_ylim(0, shared_max)
            axis.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
            axis.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=12))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_title(
                f"{_short_endpoint(endpoint)}  (epochs={len(values)})", loc="left"
            )
        fig.suptitle(
            f"Open-loop capacity points: {shape.replace('_', ' ')} (matched workload only)",
            y=0.975,
        )
        label_y = 0.15 if len(shown) <= 3 else 0.045
        fig.supxlabel("offered requests per minute (log scale)", y=label_y)
        fig.supylabel("successful requests per minute (symlog; zero retained)", x=0.018)
        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=PALETTE["green"],
                label="healthy valid epoch",
            ),
            plt.Line2D(
                [],
                [],
                marker="X",
                linestyle="",
                color=PALETTE["red"],
                label="unhealthy valid epoch",
            ),
            plt.Line2D(
                [],
                [],
                marker="D",
                linestyle="",
                color=PALETTE["gray"],
                label="censored / invalid epoch",
            ),
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.015)
        )
        target = chart_dir / f"capacity-{shape.replace('_', '-')}-matched-points.png"
        _save(fig, target)
        plt.close(fig)
        paths.append(target)
    return paths


def _controller_step_plots(
    plt: Any,
    chart_dir: Path,
    endpoints: Sequence[str],
    epochs: Sequence[Mapping[str, Any]],
) -> list[Path]:
    rows = [
        row
        for row in epochs
        if row.get("source_kind") == "direct_aimd"
        and _number(row.get("offered_rpm")) is not None
        and _number(row.get("achieved_rpm")) is not None
    ]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (str(row.get("source_id")), str(row.get("shape") or "unspecified"))
        ].append(row)
    paths: list[Path] = []
    for (source_id, shape), members in sorted(groups.items()):
        shown = [
            endpoint
            for endpoint in endpoints
            if any(row.get("endpoint_id") == endpoint for row in members)
        ]
        if not shown:
            continue
        fig, axes = _endpoint_grid(plt, shown, height=2.75)
        for endpoint, axis in zip(shown, axes.flat):
            values = sorted(
                [row for row in members if row.get("endpoint_id") == endpoint],
                key=lambda row: (
                    _number(row.get("sequence")) is None,
                    _number(row.get("sequence")) or 0,
                    str(row.get("started_at") or ""),
                ),
            )
            x = list(range(1, len(values) + 1))
            offered = [float(row.get("offered_rpm") or 0) for row in values]
            achieved = [float(row.get("achieved_rpm") or 0) for row in values]
            axis.step(
                x,
                offered,
                where="mid",
                color=PALETTE["blue"],
                linewidth=1.2,
                label="offered",
            )
            axis.plot(
                x,
                achieved,
                color="#273442",
                linewidth=1.0,
                marker=".",
                markersize=3,
                label="achieved",
            )
            for index, row in enumerate(values, start=1):
                phase = str(row.get("phase") or "").casefold()
                if row.get("valid_for_capacity") is not True:
                    color, marker = PALETTE["gray"], "D"
                elif row.get("healthy") is True:
                    color, marker = PALETTE["green"], "o"
                else:
                    color, marker = PALETTE["red"], "X"
                if phase in {"confirmation", "confirm"}:
                    marker = "s"
                elif "recovery" in phase or "fallback" in phase:
                    marker = "v"
                axis.scatter(
                    index,
                    float(row.get("achieved_rpm") or 0),
                    color=color,
                    marker=marker,
                    s=22,
                    zorder=3,
                )
            axis.set_yscale("symlog", linthresh=1.0, linscale=0.7)
            axis.set_ylim(bottom=0)
            axis.set_title(
                f"{_short_endpoint(endpoint)}  (epochs={len(values)})", loc="left"
            )
            axis.grid(True, axis="y", alpha=0.18)
        fig.suptitle(
            f"Chronological AIMD controller: {shape.replace('_', ' ')}\n{source_id}",
            y=0.985,
        )
        fig.supxlabel(
            "epoch order inside this exact run",
            y=0.15 if len(shown) <= 3 else 0.045,
        )
        fig.supylabel("requests per minute (symlog; zero retained)", x=0.018)
        handles = [
            plt.Line2D([], [], color=PALETTE["blue"], label="offered arrival rate"),
            plt.Line2D([], [], color="#273442", label="achieved successful rate"),
            plt.Line2D(
                [],
                [],
                marker="s",
                linestyle="",
                color=PALETTE["green"],
                label="healthy confirmation",
            ),
            plt.Line2D(
                [],
                [],
                marker="X",
                linestyle="",
                color=PALETTE["red"],
                label="unhealthy valid epoch",
            ),
            plt.Line2D(
                [],
                [],
                marker="v",
                linestyle="",
                color=PALETTE["green"],
                label="recovery/fallback",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.01),
            fontsize=7.5,
        )
        source_slug = "".join(
            character if character.isalnum() else "-"
            for character in source_id.casefold()
        ).strip("-")
        target = chart_dir / (
            f"aimd-controller-{shape.replace('_', '-')}-{source_slug[:40]}.png"
        )
        _save(fig, target)
        plt.close(fig)
        paths.append(target)
    return paths


def _soak_plot(
    plt: Any,
    path: Path,
    endpoints: Sequence[str],
    blocks: Sequence[Mapping[str, Any]],
) -> bool:
    if not blocks:
        return False
    shapes = sorted({str(row.get("shape") or "unspecified") for row in blocks})
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in blocks:
        grouped[
            (str(row.get("endpoint_id")), str(row.get("shape") or "unspecified"))
        ].append(row)
    success_matrix: list[list[float]] = []
    variability_matrix: list[list[float]] = []
    for endpoint in endpoints:
        success_row: list[float] = []
        variation_row: list[float] = []
        for shape in shapes:
            cell = grouped.get((endpoint, shape), [])
            success = [
                v for row in cell if (v := _number(row.get("success_rate"))) is not None
            ]
            output = [
                v
                for row in cell
                if (v := _number(row.get("effective_output_tpm"))) is not None and v > 0
            ]
            success_row.append(min(success) if success else math.nan)
            variation_row.append(
                max(output) / min(output) if len(output) >= 2 else math.nan
            )
        success_matrix.append(success_row)
        variability_matrix.append(variation_row)
    has_variability = any(
        math.isfinite(value) for row in variability_matrix for value in row
    )
    panel_count = 2 if has_variability else 1
    fig, axes_value = plt.subplots(
        1,
        panel_count,
        figsize=((12.2, 6.5) if has_variability else (7.6, 6.5)),
        constrained_layout=True,
    )
    axes = [axes_value] if panel_count == 1 else list(axes_value)
    first = axes[0].imshow(success_matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    panels: list[tuple[Any, str]] = [(axes[0], "Worst 30-second block success rate")]
    second = None
    if has_variability:
        second = axes[1].imshow(
            variability_matrix, aspect="auto", vmin=1, vmax=3, cmap="magma_r"
        )
        panels.append((axes[1], "Max/min output goodput across 4 blocks"))
    for panel_index, (axis, title) in enumerate(panels):
        axis.set_xticks(
            range(len(shapes)),
            [s.replace("_", " ") for s in shapes],
            rotation=34,
            ha="right",
        )
        axis.set_yticks(range(len(endpoints)), [_short_endpoint(v) for v in endpoints])
        axis.set_title(title, loc="left")
        axis.grid(False)
        matrix = success_matrix if panel_index == 0 else variability_matrix
        for row_index, values in enumerate(matrix):
            for column_index, value in enumerate(values):
                if not math.isfinite(value):
                    label = "—"
                    color = "#536171"
                elif panel_index == 0:
                    label = f"{value:.0%}"
                    color = "white" if value <= 0.25 or value >= 0.78 else "#17212b"
                else:
                    label = f"{value:.2f}×"
                    color = "white" if value >= 2.1 else "#17212b"
                axis.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=color,
                    fontweight="semibold" if math.isfinite(value) else "normal",
                )
    fig.colorbar(first, ax=axes[0], fraction=0.04, label="success fraction")
    if second is not None:
        fig.colorbar(
            second, ax=axes[1], fraction=0.04, label="fold variation (clipped at 3x)"
        )
    fig.suptitle(
        "Two-minute soak stability: four predeclared 30-second analysis blocks", y=1.02
    )
    fig.text(
        0.5,
        -0.015,
        "— = no valid four-block estimate; blanks are never imputed as success or failure",
        ha="center",
        fontsize=8,
        color="#536171",
    )
    _save(fig, path)
    plt.close(fig)
    return True


def _capability_plot(
    plt: Any,
    path: Path,
    endpoints: Sequence[str],
    evidence: Sequence[Mapping[str, Any]],
) -> bool:
    if not evidence:
        return False
    dimensions = sorted({str(row.get("capability_dimension")) for row in evidence})
    lookup = {
        (str(row.get("endpoint_id")), str(row.get("capability_dimension"))): row
        for row in evidence
    }
    transport_values = [
        "documented_unavailable",
        "inconclusive",
        "observed_transport_degraded",
        "observed_supported",
    ]
    functional_values = ["not_scored", "failed", "degraded", "passed"]
    transport_colors = [
        PALETTE["purple"],
        PALETTE["gray"],
        PALETTE["orange"],
        PALETTE["green"],
    ]
    functional_colors = [
        PALETTE["light"],
        PALETTE["red"],
        PALETTE["orange"],
        PALETTE["green"],
    ]
    from matplotlib.colors import ListedColormap

    transport = [
        [
            transport_values.index(
                str(lookup.get((e, d), {}).get("transport_status") or "inconclusive")
            )
            for d in dimensions
        ]
        for e in endpoints
    ]
    functional = [
        [
            functional_values.index(
                str(lookup.get((e, d), {}).get("functional_status") or "not_scored")
            )
            for d in dimensions
        ]
        for e in endpoints
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 7.0), constrained_layout=True)
    axes[0].imshow(
        transport,
        aspect="auto",
        cmap=ListedColormap(transport_colors),
        vmin=-0.5,
        vmax=3.5,
    )
    axes[1].imshow(
        functional,
        aspect="auto",
        cmap=ListedColormap(functional_colors),
        vmin=-0.5,
        vmax=3.5,
    )
    for axis, title in zip(
        axes,
        ("Valid-call transport support", "Functional correctness on 2xx responses"),
    ):
        axis.set_xticks(
            range(len(dimensions)),
            [d.replace("_", " ") for d in dimensions],
            rotation=45,
            ha="right",
            fontsize=7.5,
        )
        axis.set_yticks(range(len(endpoints)), [_short_endpoint(e) for e in endpoints])
        axis.set_title(title, loc="left")
        axis.grid(False)
    fig.suptitle(
        "Capability evidence separates transport, answer correctness, and malformed-input validation",
        y=1.02,
    )
    _save(fig, path)
    plt.close(fig)
    return True


def _context_plot(
    plt: Any,
    path: Path,
    endpoints: Sequence[str],
    requests: Sequence[Mapping[str, Any]],
) -> bool:
    rows = [
        row
        for row in requests
        if row.get("workload") in {"long_context_retrieval", "context_boundary"}
        and (
            _number(row.get("requested_input_tokens"))
            or _number(row.get("estimated_target_input_tokens"))
            or 0
        )
        > 0
    ]
    shown = [
        endpoint
        for endpoint in endpoints
        if any(row.get("endpoint_id") == endpoint for row in rows)
    ]
    if not shown:
        return False
    fig, axes = _endpoint_grid(plt, shown, height=2.45)
    for endpoint, axis in zip(shown, axes.flat):
        values = [row for row in rows if row.get("endpoint_id") == endpoint]
        for row in values:
            x = _number(row.get("requested_input_tokens")) or _number(
                row.get("estimated_target_input_tokens")
            )
            if row.get("transport_success"):
                y = 1.0 if row.get("functional_valid") is not False else 0.55
                color = PALETTE["green"] if y == 1.0 else PALETTE["orange"]
                marker = "o" if y == 1.0 else "^"
            elif (
                str(row.get("coverage_classification"))
                == "explicit_context_limit_rejection"
            ):
                y, color, marker = 0.15, PALETTE["purple"], "X"
            else:
                y, color, marker = 0.0, PALETTE["gray"], "D"
            axis.scatter(x, y, color=color, marker=marker, s=24, alpha=0.72)
        axis.set_xscale("log")
        axis.set_ylim(-0.1, 1.1)
        axis.set_yticks(
            [0, 0.55, 1], ["inconclusive", "retrieval fail", "retrieval pass"]
        )
        axis.set_title(f"{_short_endpoint(endpoint)}  (n={len(values)})", loc="left")
    fig.suptitle(
        "Long-context acceptance and retrieval outcome (concurrent timing is not compared)",
        y=0.975,
    )
    fig.supxlabel(
        "planned input tokens (log scale)",
        y=0.15 if len(shown) <= 3 else 0.045,
    )
    _save(fig, path)
    plt.close(fig)
    return True


def _quality_plot(
    plt: Any,
    path: Path,
    pairs: Sequence[Mapping[str, Any]],
) -> bool:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pair_counts: Counter[tuple[str, str, str]] = Counter()
    for row in pairs:
        value = _number(row.get("paired_quality_delta_near_minus_low"))
        block_id = row.get("analysis_block_id")
        if (
            value is not None
            and block_id is not None
            and row.get("exact_request_payload_hash_match") is not False
        ):
            key = (
                str(row.get("source_id")),
                str(row.get("endpoint_id")),
                str(row.get("shape")),
            )
            grouped[key][str(block_id)].append(value)
            pair_counts[key] += 1
    estimates: list[tuple[str, str, str, float, float, float, int, int]] = []
    critical_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    for (source_id, endpoint, shape), by_block in grouped.items():
        block_means = [statistics.mean(values) for values in by_block.values()]
        estimate = statistics.mean(block_means)
        if len(block_means) >= 2:
            critical = critical_95.get(len(block_means), 1.96)
            half = (
                critical * statistics.stdev(block_means) / math.sqrt(len(block_means))
            )
        else:
            half = math.nan
        estimates.append(
            (
                source_id,
                endpoint,
                shape,
                estimate,
                estimate - half,
                estimate + half,
                len(block_means),
                pair_counts[(source_id, endpoint, shape)],
            )
        )
    if not estimates:
        return False
    shape_order = [
        shape
        for shape in ("short_short", "input32k_short", "short_long", "mixed")
        if any(item[2] == shape for item in estimates)
    ]
    shape_order.extend(sorted({item[2] for item in estimates} - set(shape_order)))
    columns = 2
    rows = math.ceil(len(shape_order) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.0, max(5.8, 4.1 * rows)),
        squeeze=False,
    )
    finite_bounds = [
        bound
        for item in estimates
        for bound in (item[4], item[5])
        if math.isfinite(bound)
    ]
    maximum = max((abs(value) for value in finite_bounds), default=1.0)
    x_limit = max(0.25, maximum * 1.22)
    for shape, axis in zip(shape_order, axes.flat):
        values = sorted(
            [item for item in estimates if item[2] == shape],
            key=lambda item: (_short_endpoint(item[1]), item[0]),
        )
        labels: list[str] = []
        duplicate_endpoints = Counter(item[1] for item in values)
        for position, (
            source_id,
            endpoint,
            _,
            estimate,
            low,
            high,
            n_blocks,
            n_pairs,
        ) in enumerate(values):
            label = _short_endpoint(endpoint)
            if duplicate_endpoints[endpoint] > 1:
                label = f"{label} | {source_id[-8:]}"
            labels.append(label)
            color = PALETTE["green"] if estimate >= 0 else PALETTE["orange"]
            if math.isfinite(low) and math.isfinite(high):
                axis.errorbar(
                    estimate,
                    position,
                    xerr=[[estimate - low], [high - estimate]],
                    fmt="o",
                    color=color,
                    capsize=2,
                )
            else:
                axis.scatter(estimate, position, color=color)
            axis.text(
                x_limit * 0.98,
                position,
                f"{n_blocks}b/{n_pairs}p",
                va="center",
                ha="right",
                fontsize=7,
                color="#4E5965",
            )
        axis.axvline(0, color="#273442", linewidth=1)
        axis.set_xlim(-x_limit, x_limit)
        axis.set_yticks(range(len(values)), labels, fontsize=7.3)
        axis.set_title(shape.replace("_", " "), loc="left")
        axis.grid(True, axis="x", alpha=0.18)
        axis.grid(False, axis="y")
    for axis in axes.flat[len(shape_order) :]:
        axis.set_visible(False)
    fig.suptitle(
        "Paired quality change under load: exact endpoint and workload cells",
        y=0.985,
    )
    fig.supxlabel(
        "near-load minus matched low-load quality score; bars are 95% t intervals across block means",
        y=0.035,
    )
    fig.text(
        0.5,
        0.012,
        "Right-edge labels report independent analysis blocks (b) and exact matched pairs (p).",
        ha="center",
        fontsize=8,
        color="#4E5965",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95), h_pad=1.25, w_pad=1.2)
    _save(fig, path)
    plt.close(fig)
    return True


def _cost_performance_plot(
    plt: Any,
    path: Path,
    endpoints: Sequence[str],
    epochs: Sequence[Mapping[str, Any]],
) -> bool:
    grouped_epochs: dict[tuple[str, str, str, float], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in epochs:
        phase = str(row.get("phase") or "").casefold()
        offered_rps = _number(row.get("offered_rps"))
        if (
            row.get("source_kind") != "direct_aimd"
            or phase not in {"confirmation", "confirm"}
            or row.get("healthy") is not True
            or row.get("valid_for_capacity") is not True
            or offered_rps is None
        ):
            continue
        grouped_epochs[
            (
                str(row.get("source_id")),
                str(row.get("endpoint_id")),
                str(row.get("shape") or "unspecified"),
                offered_rps,
            )
        ].append(row)
    cells: list[tuple[str, str, str, float, float, float, float]] = []
    critical_95 = {3: 4.303, 4: 3.182, 5: 2.776}
    for (source_id, endpoint, shape, offered_rps), members in grouped_epochs.items():
        if len(members) < 3:
            continue
        outputs: list[float] = []
        realized_tokens = 0.0
        cost = 0.0
        complete_cost = True
        for row in members:
            elapsed = _number(row.get("elapsed_seconds"))
            output_tpm = _number(row.get("effective_output_tpm"))
            epoch_cost = _number(row.get("estimated_cost_usd"))
            if not elapsed or output_tpm is None or output_tpm <= 0:
                continue
            outputs.append(output_tpm)
            realized_tokens += output_tpm * elapsed / 60
            if epoch_cost is None:
                complete_cost = False
            else:
                cost += epoch_cost
        if len(outputs) < 3 or realized_tokens <= 0 or not complete_cost:
            continue
        estimate = statistics.mean(outputs)
        critical = critical_95.get(len(outputs), 1.96)
        half = critical * statistics.stdev(outputs) / math.sqrt(len(outputs))
        cells.append(
            (
                source_id,
                endpoint,
                shape,
                offered_rps,
                cost * 1_000_000 / realized_tokens,
                estimate,
                half,
            )
        )
    shapes = [
        shape
        for shape in ("short_short", "input32k_short", "short_long", "mixed")
        if any(row[2] == shape for row in cells)
    ]
    if not shapes:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.3), squeeze=False)
    endpoint_colors = {
        endpoint: plt.cm.tab20(index % 20) for index, endpoint in enumerate(endpoints)
    }
    handles: dict[str, Any] = {}
    for shape, axis in zip(shapes, axes.flat):
        values = [row for row in cells if row[2] == shape]
        for _, endpoint, _, offered_rps, cost, output, half in values:
            handle = axis.errorbar(
                cost,
                output,
                yerr=half,
                fmt="o",
                color=endpoint_colors[endpoint],
                markersize=4,
                capsize=2,
                alpha=0.85,
            )
            handles.setdefault(endpoint, handle)
            axis.annotate(
                f"{offered_rps * 60:,.0f}",
                (cost, output),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(shape.replace("_", " "), loc="left")
        axis.set_xlabel("estimated $ / million successful output tokens")
        axis.set_ylabel("effective output tokens/minute")
        axis.grid(True, which="both", alpha=0.18)
    for axis in axes.flat[len(shapes) :]:
        axis.set_visible(False)
    fig.legend(
        list(handles.values()),
        [_short_endpoint(endpoint) for endpoint in handles],
        loc="lower center",
        ncol=3,
        fontsize=7,
        frameon=False,
    )
    fig.suptitle(
        "Exact confirmed-rate cells: cost versus achieved output goodput",
        y=0.98,
    )
    fig.text(
        0.5,
        0.105,
        "Number beside each point = healthy offered RPM; bars = 95% t interval across confirmation epochs",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    _save(fig, path)
    plt.close(fig)
    return True


def build_public_plots(
    output_directory: Path,
    *,
    endpoints: Sequence[str],
    dimensions: Sequence[str],
    requests: Sequence[Mapping[str, Any]],
    epochs: Sequence[Mapping[str, Any]],
    coverage_matrix: Sequence[Mapping[str, Any]],
    soak_blocks: Sequence[Mapping[str, Any]],
    quality_pairs: Sequence[Mapping[str, Any]],
    capability_evidence: Sequence[Mapping[str, Any]],
    metric_audit_rows: Sequence[Mapping[str, Any]],
    metric_audit_summary: Mapping[str, Any],
) -> list[str]:
    """Render the non-empty publication suite and return relative paths."""

    plt = _setup()
    chart_dir = Path(output_directory) / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    for stale in chart_dir.iterdir():
        if stale.is_file() and stale.suffix.casefold() in {".png", ".jpg", ".jpeg"}:
            stale.unlink()
    paths: list[Path] = []

    def add(filename: str, builder: Any, *args: Any) -> None:
        target = chart_dir / filename
        if builder(plt, target, *args):
            paths.append(target)

    add(
        "coverage-status-matrix.png",
        _coverage_plot,
        endpoints,
        dimensions,
        coverage_matrix,
    )
    add("ttft-input-cache-strata.png", _ttft_plot, endpoints, requests)
    add("output-post-ttft-proxy.png", _output_proxy_plot, endpoints, requests)
    add(
        "metric-outlier-audit.png", _audit_plot, metric_audit_rows, metric_audit_summary
    )
    paths.extend(_capacity_plots(plt, chart_dir, endpoints, epochs))
    paths.extend(_controller_step_plots(plt, chart_dir, endpoints, epochs))
    add("soak-four-block-stability.png", _soak_plot, endpoints, soak_blocks)
    add(
        "capability-transport-functional-matrix.png",
        _capability_plot,
        endpoints,
        capability_evidence,
    )
    add("context-acceptance-retrieval.png", _context_plot, endpoints, requests)
    add("paired-quality-load-effect.png", _quality_plot, quality_pairs)
    add("matched-cost-performance.png", _cost_performance_plot, endpoints, epochs)
    return [path.relative_to(output_directory).as_posix() for path in paths]


__all__ = ["build_public_plots"]
