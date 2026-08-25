"""Build the concise DigitalOcean inference engineering encyclopedia.

The encyclopedia consumes only the sanitized, derived CSV bundle.  It avoids
heterogeneous endpoint-wide averages, never connects AIMD epochs into loops,
and labels censored or unverified capacity explicitly.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Flowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .core import DIGITALOCEAN_HOSTED_MODEL_IDS
from .direct_report import EXPECTED_ENDPOINT_IDS


REPORT_ENDPOINT_IDS = DIGITALOCEAN_HOSTED_MODEL_IDS
EXCLUDED_PARTNER_MODEL_ID = "arcee-trinity-large-thinking"


NAVY = "#0B1F33"
BLUE = "#1976D2"
TEAL = "#00897B"
AMBER = "#D97706"
RED = "#C2413A"
GRAY = "#687583"
LIGHT = "#F4F7FA"
MID = "#D8E0E8"
WHITE = "#FFFFFF"

SHAPES = (
    ("short_short", "Short input / short output"),
    ("input32k_short", "32K input / short output"),
    ("short_long", "Short input / long output"),
    ("mixed", "Heterogeneous mixed"),
)

SHORT_NAMES = {
    "arcee-trinity-large-thinking": "Arcee Trinity",
    "deepseek-v4-flash-0731": "DeepSeek V4 Flash 0731",
    "gemma-4-31B-it": "Gemma 4 31B IT",
    "glm-5.2": "GLM 5.2",
    "kimi-k3": "Kimi K3",
    "minimax-m2.5": "MiniMax M2.5",
    "mimo-v2.5-pro": "MiMo V2.5 Pro",
    "nemotron-3-ultra-550b": "Nemotron 3 Ultra",
    "nvidia-nemotron-3-super-120b": "Nemotron 3 Super",
    "openai-gpt-oss-120b": "GPT-OSS 120B",
    "qwen3.5-397b-a17b": "Qwen 3.5 397B",
    "qwen3.8-max": "Qwen 3.8 Max",
}

CAPABILITY_ORDER = (
    "streaming",
    "vision",
    "tools",
    "parallel_tool_calls",
    "structured_output",
    "temperature",
    "top_p",
    "stop",
    "seed",
    "logprobs",
)


@dataclass(frozen=True)
class Bundle:
    root: Path
    rows: Mapping[str, tuple[Mapping[str, str], ...]]
    analysis: Mapping[str, Any]


class _FooterCanvas(canvas.Canvas):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pages: list[dict[str, Any]] = []

    def showPage(self) -> None:  # noqa: N802
        self._pages.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        total = len(self._pages)
        for state in self._pages:
            self.__dict__.update(state)
            self.saveState()
            self.setStrokeColor(colors.HexColor(MID))
            self.setLineWidth(0.4)
            self.line(16 * mm, 13 * mm, A4[0] - 16 * mm, 13 * mm)
            self.setFillColor(colors.HexColor(GRAY))
            self.setFont("Helvetica", 7.2)
            self.drawString(
                16 * mm, 8.8 * mm, "DigitalOcean inference engineering encyclopedia"
            )
            self.drawRightString(
                A4[0] - 16 * mm,
                8.8 * mm,
                f"{self._pageNumber} / {total}",
            )
            self.restoreState()
            super().showPage()
        super().save()


def _read_csv(path: Path) -> tuple[Mapping[str, str], ...]:
    if not path.is_file():
        return ()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def load_bundle(root: Path) -> Bundle:
    root = Path(root)
    names = (
        "endpoint-inventory.csv",
        "endpoint-summary.csv",
        "capacity-summary.csv",
        "soak-cell-summary.csv",
        "soak-block-summary.csv",
        "quality-pair-summary.csv",
        "coverage-matrix.csv",
        "capability-evidence.csv",
        "observed-limits.csv",
        "metric-audit.csv",
        "endpoint-workload-metrics.csv",
        "normalized-requests.csv",
    )
    rows = {name: _read_csv(root / name) for name in names}
    analysis_path = root / "analysis.json"
    compressed_path = root / "analysis.json.gz"
    if analysis_path.is_file():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    elif compressed_path.is_file():
        with gzip.open(compressed_path, "rt", encoding="utf-8") as handle:
            analysis = json.load(handle)
    else:
        raise FileNotFoundError(
            "analysis.json or analysis.json.gz is required for the encyclopedia"
        )
    inventory_ids = {row.get("endpoint_id") for row in rows["endpoint-inventory.csv"]}
    if inventory_ids != set(EXPECTED_ENDPOINT_IDS):
        raise ValueError("endpoint inventory does not match the frozen 12-endpoint set")
    return Bundle(root=root, rows=rows, analysis=analysis)


def _num(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _integer(value: Any) -> int | None:
    number = _num(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _truth(value: Any) -> bool:
    return value is True or str(value).casefold() == "true"


def _ci(value: Any) -> tuple[float, float] | None:
    if value in (None, ""):
        return None
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if isinstance(parsed, list) and len(parsed) == 2:
        low, high = (_num(parsed[0]), _num(parsed[1]))
    elif isinstance(parsed, Mapping):
        low, high = (_num(parsed.get("ci95_low")), _num(parsed.get("ci95_high")))
    else:
        return None
    return (low, high) if low is not None and high is not None else None


def _fmt(value: Any, digits: int = 1) -> str:
    number = _num(value)
    if number is None:
        return "—"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.1f}K"
    if abs(number) >= 100:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def _money(value: Any) -> str:
    number = _num(value)
    return "—" if number is None else f"${number:,.3f}"


def _pct(value: Any) -> str:
    number = _num(value)
    return "—" if number is None else f"{number * 100:.1f}%"


def _short(endpoint: str) -> str:
    return SHORT_NAMES.get(endpoint, endpoint)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=31,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#DCEBFA"),
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor(NAVY),
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor(BLUE),
            spaceBefore=7,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=12.4,
            textColor=colors.HexColor(NAVY),
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.1,
            leading=9.5,
            textColor=colors.HexColor(GRAY),
        ),
        "table": ParagraphStyle(
            "Table",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.1,
            leading=8.9,
            textColor=colors.HexColor(NAVY),
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=8.6,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "kpi": ParagraphStyle(
            "KPI",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=18,
            textColor=colors.HexColor(NAVY),
            alignment=TA_CENTER,
        ),
        "kpi_label": ParagraphStyle(
            "KPILabel",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=6.8,
            leading=8.2,
            textColor=colors.HexColor(GRAY),
            alignment=TA_CENTER,
        ),
        "right": ParagraphStyle(
            "Right",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.1,
            leading=8.9,
            textColor=colors.HexColor(NAVY),
            alignment=TA_RIGHT,
        ),
    }


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    safe = str(text if text not in (None, "") else "—")
    safe = safe.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe, style)


def _table(
    rows: Sequence[Sequence[Any]],
    styles: Mapping[str, ParagraphStyle],
    widths: Sequence[float] | None = None,
    *,
    header: bool = True,
) -> Table:
    rendered: list[list[Any]] = []
    for row_index, row in enumerate(rows):
        rendered.append(
            [
                value
                if isinstance(value, Flowable)
                else _p(
                    value,
                    styles["table_head"]
                    if header and row_index == 0
                    else styles["table"],
                )
                for value in row
            ]
        )
    table = Table(
        rendered, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT"
    )
    commands: list[tuple[Any, ...]] = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(MID)),
        (
            "ROWBACKGROUNDS",
            (0, 1 if header else 0),
            (-1, -1),
            [colors.white, colors.HexColor(LIGHT)],
        ),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)))
    table.setStyle(TableStyle(commands))
    return table


def _kpis(
    items: Sequence[tuple[str, str]], styles: Mapping[str, ParagraphStyle]
) -> Table:
    cells = [
        [_p(value, styles["kpi"]), Spacer(1, 1.2 * mm), _p(label, styles["kpi_label"])]
        for value, label in items
    ]
    table = Table([cells], colWidths=[(A4[0] - 34 * mm) / len(cells)] * len(cells))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(LIGHT)),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(MID)),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor(MID)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def _callout(
    title: str,
    text: str,
    styles: Mapping[str, ParagraphStyle],
    *,
    warning: bool = False,
) -> Table:
    border = AMBER if warning else BLUE
    background = "#FFF7E8" if warning else "#EDF6FF"
    table = Table(
        [[_p(title, styles["h2"])], [_p(text, styles["body"])]],
        colWidths=[A4[0] - 34 * mm],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(background)),
                ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(border)),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, 0), 6),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 6),
            ]
        )
    )
    return table


def _access_incident(bundle: Bundle) -> Mapping[str, Any] | None:
    for row in reversed(tuple(bundle.analysis.get("data_sources", ()))):
        if str(row.get("source_id", "")).startswith("do-matched-closure-"):
            return row
    return None


def _chart_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.edgecolor": MID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E7ECF1",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "text.color": NAVY,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
        }
    )


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    return path


def _best_capacity_rows(bundle: Bundle) -> dict[tuple[str, str], Mapping[str, str]]:
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in bundle.rows["capacity-summary.csv"]:
        endpoint, shape = row.get("endpoint_id"), row.get("shape")
        if endpoint in REPORT_ENDPOINT_IDS and shape in {key for key, _ in SHAPES}:
            grouped[(str(endpoint), str(shape))].append(row)
    selected: dict[tuple[str, str], Mapping[str, str]] = {}
    for key, rows in grouped.items():
        selected[key] = max(
            rows,
            key=lambda row: (
                _integer(row.get("candidate_rate_confirmation_epoch_count")) or 0,
                _num(row.get("capacity_lower_bound_rps")) or -1,
                _num(row.get("highest_observed_healthy_rps")) or -1,
            ),
        )
    return selected


def _best_soak_rows(bundle: Bundle) -> dict[tuple[str, str], Mapping[str, str]]:
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in bundle.rows["soak-cell-summary.csv"]:
        if not _truth(row.get("scientifically_complete")):
            continue
        key = (str(row.get("endpoint_id")), str(row.get("shape")))
        grouped[key].append(row)
    selected: dict[tuple[str, str], Mapping[str, str]] = {}
    for key, rows in grouped.items():
        passes = [row for row in rows if _truth(row.get("soak_acceptance_pass"))]
        candidates = passes or rows
        selected[key] = max(
            candidates, key=lambda row: _num(row.get("candidate_rate_rps")) or -1
        )
    return selected


def _human_rate(value: float, _position: int | None = None) -> str:
    return _fmt(value, 0)


def _configure_log_x(ax: Any) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.xaxis.set_major_formatter(FuncFormatter(_human_rate))
    ax.xaxis.set_minor_formatter(NullFormatter())


def build_charts(bundle: Bundle, output: Path) -> list[Path]:
    _chart_style()
    output.mkdir(parents=True, exist_ok=True)
    charts: list[Path] = []
    capacity = _best_capacity_rows(bundle)
    soak = _best_soak_rows(bundle)

    fig, ax = plt.subplots(figsize=(10, 6.3))
    dimensions = sorted(
        {str(row["coverage_dimension"]) for row in bundle.rows["coverage-matrix.csv"]}
    )
    dimension_labels = [dimension.replace("_", " ") for dimension in dimensions]
    statuses = ("completed", "unsupported", "inconclusive")
    palette = {"completed": TEAL, "unsupported": GRAY, "inconclusive": AMBER}
    left = [0] * len(dimensions)
    for status in statuses:
        values = [
            sum(
                row.get("coverage_dimension") == dimension
                and row.get("status") == status
                for row in bundle.rows["coverage-matrix.csv"]
            )
            for dimension in dimensions
        ]
        ax.barh(
            dimension_labels,
            values,
            left=left,
            color=palette[status],
            label=status.title(),
        )
        left = [a + b for a, b in zip(left, values)]
    ax.set_xlim(0, len(REPORT_ENDPOINT_IDS))
    ax.set_xlabel(f"DigitalOcean-hosted endpoints (of {len(REPORT_ENDPOINT_IDS)})")
    ax.set_title("Evidence status by required benchmark dimension", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    charts.append(_save(fig, output / "coverage-by-dimension.png"))

    fig, axes = plt.subplots(1, 4, figsize=(14, 7.2), sharey=True)
    for ax, (shape, label) in zip(axes, SHAPES):
        values = []
        for endpoint in REPORT_ENDPOINT_IDS:
            row = capacity.get((endpoint, shape), {})
            rpm = _num(row.get("capacity_lower_bound_rpm"))
            values.append(rpm)
        y = list(range(len(REPORT_ENDPOINT_IDS)))
        ax.scatter(
            [value or math.nan for value in values], y, color=BLUE, s=34, zorder=3
        )
        for yi, value in zip(y, values):
            if value is not None:
                ax.text(value, yi, f"  {_fmt(value, 0)}", va="center", fontsize=7.2)
        _configure_log_x(ax)
        ax.set_title(label)
        ax.set_xlabel("offered RPM\n(confirmed lower bound)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(REPORT_ENDPOINT_IDS)))
    axes[0].set_yticklabels([_short(endpoint) for endpoint in REPORT_ENDPOINT_IDS])
    fig.suptitle(
        "AIMD healthy offered-rate lower bounds — matched workload cells",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(wspace=0.2, left=0.2)
    charts.append(_save(fig, output / "aimd-capacity-lower-bounds.png"))

    fig, axes = plt.subplots(1, 4, figsize=(14, 7.2), sharey=True)
    for ax, (shape, label) in zip(axes, SHAPES):
        for yi, endpoint in enumerate(REPORT_ENDPOINT_IDS):
            row = soak.get((endpoint, shape))
            if not row:
                continue
            estimate = _num(
                row.get("arrival_cohort_successful_rpm_including_drain_block_mean")
            )
            interval = _ci(
                row.get(
                    "arrival_cohort_successful_rpm_including_drain_block_mean_ci95_student_t"
                )
            )
            if estimate is None:
                continue
            color = TEAL if _truth(row.get("soak_acceptance_pass")) else AMBER
            if interval:
                ax.plot(interval, [yi, yi], color=color, lw=2)
            ax.scatter([estimate], [yi], color=color, s=32, zorder=3)
        _configure_log_x(ax)
        ax.set_title(label)
        ax.set_xlabel("successful RPM\n(2-minute block mean, 95% CI)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(REPORT_ENDPOINT_IDS)))
    axes[0].set_yticklabels([_short(endpoint) for endpoint in REPORT_ENDPOINT_IDS])
    fig.suptitle(
        "Two-minute achieved goodput at the tested candidate rate",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.01,
        "Teal = predeclared composite passed; amber = measured but composite did not pass. Intervals use four contiguous 30-second blocks.",
        fontsize=8,
        color=GRAY,
    )
    fig.subplots_adjust(wspace=0.2, left=0.2, bottom=0.12)
    charts.append(_save(fig, output / "two-minute-achieved-rpm.png"))

    for metric, ci_field, title, filename in (
        (
            "effective_input_tpm",
            "effective_input_tpm_ci95",
            "Effective input throughput at the confirmed AIMD cell",
            "effective-input-tpm.png",
        ),
        (
            "effective_output_tpm",
            "effective_output_tpm_ci95",
            "Effective output throughput at the confirmed AIMD cell",
            "effective-output-tpm.png",
        ),
        (
            "ttft_p50_seconds",
            "ttft_p50_seconds_ci95",
            "Median time to first streamed token at the confirmed AIMD cell",
            "ttft-at-confirmed-cell.png",
        ),
    ):
        fig, axes = plt.subplots(1, 4, figsize=(14, 7.2), sharey=True)
        for ax, (shape, label) in zip(axes, SHAPES):
            for yi, endpoint in enumerate(REPORT_ENDPOINT_IDS):
                row = capacity.get((endpoint, shape))
                if not row:
                    continue
                estimate = _num(row.get(metric))
                interval = _ci(row.get(ci_field))
                if estimate is None or estimate <= 0:
                    continue
                if interval and interval[0] > 0:
                    ax.plot(interval, [yi, yi], color=BLUE, lw=2)
                ax.scatter([estimate], [yi], color=BLUE, s=30, zorder=3)
            if metric == "ttft_p50_seconds":
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            else:
                _configure_log_x(ax)
            ax.set_title(label)
            ax.set_xlabel("seconds" if "seconds" in metric else "tokens/minute")
            if metric == "ttft_p50_seconds":
                ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda value, _pos: f"{value:.1f}")
                )
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.grid(axis="y", visible=False)
        axes[0].set_yticks(range(len(REPORT_ENDPOINT_IDS)))
        axes[0].set_yticklabels(
            [_short(endpoint) for endpoint in REPORT_ENDPOINT_IDS]
        )
        fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
        fig.subplots_adjust(wspace=0.2, left=0.2)
        charts.append(_save(fig, output / filename))

    inventory = {
        row["endpoint_id"]: row for row in bundle.rows["endpoint-inventory.csv"]
    }
    limits = defaultdict(list)
    for row in bundle.rows["observed-limits.csv"]:
        if row.get("dimension") == "prompt context window":
            limits[str(row.get("endpoint_id"))].append(row)
    fig, ax = plt.subplots(figsize=(10, 6.3))
    y = list(range(len(REPORT_ENDPOINT_IDS)))
    for yi, endpoint in enumerate(REPORT_ENDPOINT_IDS):
        documented = _num(inventory[endpoint].get("context_window"))
        functional = max(
            (
                _num(row.get("maximum_functionally_valid_input_tokens")) or 0
                for row in limits.get(endpoint, [])
            ),
            default=0,
        )
        if documented:
            ax.scatter(documented, yi, marker="|", s=170, linewidth=2.5, color=GRAY)
        if functional:
            ax.scatter(functional, yi, marker="o", s=42, color=TEAL)
            if documented:
                ax.plot(
                    [min(functional, documented), max(functional, documented)],
                    [yi, yi],
                    color=MID,
                    lw=1.5,
                )
    _configure_log_x(ax)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(endpoint) for endpoint in REPORT_ENDPOINT_IDS])
    ax.set_xlabel("Tokens (log scale)")
    ax.set_title(
        "Documented context versus highest retrieval-valid prompt observed", loc="left"
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], marker="|", s=170, linewidth=2.5, color=GRAY, label="Documented")
    ax.scatter([], [], marker="o", s=42, color=TEAL, label="Retrieval-valid observed")
    ax.legend(frameon=False, loc="lower right")
    charts.append(_save(fig, output / "context-documentation-vs-observed.png"))

    audit = Counter(
        row.get("classification") or "unknown"
        for row in bundle.rows["metric-audit.csv"]
    )
    labels = ["valid_ordinary", "valid_extreme_keep_and_flag", "invalid_or_censored"]
    fig, ax = plt.subplots(figsize=(8.5, 3.1))
    values = [audit.get(label, 0) for label in labels]
    display = ["Qualified ordinary", "Qualified extreme", "Invalid / censored"]
    ax.barh(display, values, color=[TEAL, BLUE, GRAY])
    for yi, value in enumerate(values):
        ax.text(value, yi, f"  {value:,}", va="center")
    ax.set_xlabel("Request observations")
    ax.set_title("Timing-metric qualification audit", loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", visible=False)
    charts.append(_save(fig, output / "timing-metric-audit.png"))
    return charts


def _inventory(bundle: Bundle) -> dict[str, Mapping[str, str]]:
    return {
        str(row["endpoint_id"]): row for row in bundle.rows["endpoint-inventory.csv"]
    }


def _endpoint_summary(bundle: Bundle) -> dict[str, Mapping[str, str]]:
    return {str(row["endpoint_id"]): row for row in bundle.rows["endpoint-summary.csv"]}


def _coverage_counts(bundle: Bundle, endpoint: str | None = None) -> Counter[str]:
    rows = bundle.rows["coverage-matrix.csv"]
    if endpoint:
        rows = tuple(row for row in rows if row.get("endpoint_id") == endpoint)
    return Counter(str(row.get("status")) for row in rows)


def _portfolio_coverage_counts(bundle: Bundle) -> Counter[str]:
    return Counter(
        str(row.get("status"))
        for row in bundle.rows["coverage-matrix.csv"]
        if row.get("endpoint_id") in REPORT_ENDPOINT_IDS
    )


def _capability_rows(bundle: Bundle, endpoint: str) -> list[Mapping[str, str]]:
    rows = [
        row
        for row in bundle.rows["capability-evidence.csv"]
        if row.get("endpoint_id") == endpoint
    ]
    rank = {name: index for index, name in enumerate(CAPABILITY_ORDER)}
    rows.sort(
        key=lambda row: (
            rank.get(str(row.get("capability_dimension")), 999),
            str(row.get("capability_dimension")),
        )
    )
    return rows


def _limit_rows(bundle: Bundle, endpoint: str) -> list[Mapping[str, str]]:
    return [
        row
        for row in bundle.rows["observed-limits.csv"]
        if row.get("endpoint_id") == endpoint
    ]


def _cover(styles: Mapping[str, ParagraphStyle], bundle: Bundle) -> Flowable:
    coverage = _portfolio_coverage_counts(bundle)
    cost = bundle.analysis.get("cost_summary", {})
    matrix = tuple(
        row
        for row in bundle.rows["coverage-matrix.csv"]
        if row.get("endpoint_id") in REPORT_ENDPOINT_IDS
    )
    complete = coverage.get("completed", 0) + coverage.get("unsupported", 0)
    request_rows = sum(
        _integer(row.get("request_count")) or 0
        for row in bundle.rows["endpoint-summary.csv"]
        if row.get("endpoint_id") in REPORT_ENDPOINT_IDS
    )
    body = [
        Spacer(1, 28 * mm),
        _p("DIGITALOCEAN PUBLIC INFERENCE", styles["subtitle"]),
        Spacer(1, 3 * mm),
        _p("Engineering encyclopedia", styles["title"]),
        Spacer(1, 4 * mm),
        _p(
            "A request-level technical benchmark of 11 DigitalOcean-hosted endpoints across latency, throughput, context, output, tools, structured output, vision, quality, overload, and recovery. A historical partner-model mistake is isolated in an incident appendix and excluded from production comparisons.",
            styles["subtitle"],
        ),
        Spacer(1, 18 * mm),
        _kpis(
            [
                (f"{len(REPORT_ENDPOINT_IDS)}", "DO-HOSTED ENDPOINTS"),
                (f"{request_rows:,}", "NORMALIZED REQUESTS"),
                (f"{complete}/{len(matrix)}", "CONCLUSIVE COVERAGE CELLS"),
                (
                    _money(cost.get("conservative_campaign_exposure_usd")),
                    "CONSERVATIVE EXPOSURE",
                ),
            ],
            styles,
        ),
        Spacer(1, 18 * mm),
        _p(
            "How to read this report: every number is tied to one endpoint and one workload. Intervals describe the measured sample—not a provider guarantee. An arrow or ‘lower bound’ means the endpoint did not fail before the test stopped. ‘Inconclusive’ is never silently converted to supported or unsupported.",
            styles["subtitle"],
        ),
    ]
    table = Table([[body]], colWidths=[A4[0] - 34 * mm], rowHeights=[A4[1] - 44 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(NAVY)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 14 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14 * mm),
            ]
        )
    )
    return table


def _executive_pages(
    styles: Mapping[str, ParagraphStyle], bundle: Bundle, charts: Sequence[Path]
) -> list[Flowable]:
    coverage = _portfolio_coverage_counts(bundle)
    request_rows = sum(
        _integer(row.get("request_count")) or 0
        for row in bundle.rows["endpoint-summary.csv"]
        if row.get("endpoint_id") in REPORT_ENDPOINT_IDS
    )
    epoch_rows = sum(
        _integer(row.get("epoch_count")) or 0
        for row in bundle.rows["endpoint-summary.csv"]
        if row.get("endpoint_id") in REPORT_ENDPOINT_IDS
    )
    chart = {path.stem: path for path in charts}
    access_incident = _access_incident(bundle)
    story: list[Flowable] = [PageBreak(), _p("1. What the evidence says", styles["h1"])]
    story.extend(
        [
            _p(
                "The completed runs show that these endpoints can serve substantial workloads, but they are not interchangeable and should not be operated from one global RPM number. Throughput depends strongly on whether the bottleneck is request rate, prompt prefill, decoding, or a heterogeneous mix. The measured cells are engineering anchors—not current production authorization or contractual quotas.",
                styles["body"],
            ),
            _callout(
                "Inference balance gate · restored, verify before load",
                (
                    "The interrupted closure wave encountered HTTP 403 while the Serverless Inference prepaid balance was depleted. After the owner replenished the balance, the same inference credential again returned HTTP 200 from /v1/models. The separate account-control endpoint still returns 403 and is not used as an inference-readiness gate. Run two cheap serial DO-hosted marker controls before each new load wave."
                    if access_incident
                    else "No current-access closure receipt is present in this bundle. Validate account and inference access immediately before production use."
                ),
                styles,
                warning=True,
            ),
            Spacer(1, 5 * mm),
            _kpis(
                [
                    (f"{request_rows:,}", "REQUEST OBSERVATIONS"),
                    (f"{epoch_rows:,}", "LOAD EPOCHS"),
                    (f"{coverage.get('completed', 0)}", "COMPLETED CELLS"),
                    (f"{coverage.get('inconclusive', 0)}", "INCONCLUSIVE CELLS"),
                ],
                styles,
            ),
            Spacer(1, 5 * mm),
            _p("Production decision rules", styles["h2"]),
            _table(
                [
                    ["Observed condition", "Engineering action"],
                    [
                        "HTTP 429",
                        "Honor reset/retry hints, multiplicatively decrease offered load, then additively recover.",
                    ],
                    [
                        "HTTP 5xx or timeout",
                        "Retry with bounded exponential backoff and jitter; count the failed attempt in reliability and cost.",
                    ],
                    [
                        "HTTP 400 validation error",
                        "Do not retry unchanged. Correct the exact parameter or payload state.",
                    ],
                    [
                        "HTTP 401/403 on controls",
                        "Stop the lane. Treat it as credential/account access failure, not a model capability result.",
                    ],
                    [
                        "AIMD lower bound only",
                        "Treat the number as a tested floor, not a quota or sustainable guarantee.",
                    ],
                    [
                        "Two-minute cell measured but composite failed",
                        "Use the transport/latency measurements, but do not promote the cell as production-verified without resolving the failed criterion.",
                    ],
                    [
                        "Capability inconclusive",
                        "Feature-gate it. Do not infer support from model-family marketing or a generic 2xx response.",
                    ],
                ],
                styles,
                [45 * mm, 124 * mm],
            ),
            Spacer(1, 5 * mm),
            _p("Attribution boundary", styles["h2"]),
            _p(
                "These measurements characterize DigitalOcean’s serving route plus the named model. No identical external-provider control was used, so a quality or latency difference cannot be assigned uniquely to the base model, quantization, serving stack, or routing layer.",
                styles["body"],
            ),
            PageBreak(),
            _p("2. Coverage and measurement quality", styles["h1"]),
            Image(str(chart["coverage-by-dimension"]), width=176 * mm, height=111 * mm),
            Spacer(1, 4 * mm),
            _p(
                "Completed means the planned cell produced interpretable evidence. Unsupported requires documentation or an exact evidence-backed rejection. Inconclusive includes transport failures, missing prerequisites, or insufficient statistical confirmation; it is not a euphemism for failure.",
                styles["body"],
            ),
            Image(str(chart["timing-metric-audit"]), width=170 * mm, height=62 * mm),
            _p(
                "The timing audit removes buffered TTFT, sub-100 ms unstable post-TTFT denominators, multi-choice aggregate usage, and other invalid observations before any tokens/second summary. Qualified extremes remain visible and flagged; nothing is winsorized or silently clipped.",
                styles["body"],
            ),
        ]
    )
    for number, stem, heading, explanation in (
        (
            "3",
            "aimd-capacity-lower-bounds",
            "AIMD capacity map",
            "Each dot is the highest endpoint/workload offered rate with the required separated healthy confirmations. It is a lower bound when no overload knee was reached.",
        ),
        (
            "4",
            "two-minute-achieved-rpm",
            "Two-minute goodput",
            "Dots and intervals summarize four predeclared 30-second blocks. The interval is exploratory because adjacent blocks may be serially correlated.",
        ),
        (
            "5",
            "effective-input-tpm",
            "Input-token goodput",
            "Successful prompt tokens divided by complete elapsed wall time. This is end-to-end input goodput—not direct server-side prefill speed.",
        ),
        (
            "6",
            "effective-output-tpm",
            "Output-token goodput",
            "Successfully generated completion tokens divided by complete elapsed wall time, including queueing and drain.",
        ),
        (
            "7",
            "ttft-at-confirmed-cell",
            "Time to first streamed token",
            "TTFT is reported only when streaming made the first content event observable. Buffered responses are censored rather than mislabeled.",
        ),
        (
            "8",
            "context-documentation-vs-observed",
            "Context envelope",
            "Retrieval-valid means a marker embedded in synthetic long context was recovered correctly. Mere HTTP acceptance is not counted as functional long-context success.",
        ),
    ):
        story.extend(
            [
                PageBreak(),
                _p(f"{number}. {heading}", styles["h1"]),
                Image(
                    str(chart[stem]),
                    width=181 * mm,
                    height=94 * mm if number == "8" else 101 * mm,
                ),
                Spacer(1, 4 * mm),
                _p(explanation, styles["body"]),
            ]
        )
    return story


def _capability_table(
    bundle: Bundle, endpoint: str, styles: Mapping[str, ParagraphStyle]
) -> Table:
    rows = [
        [
            "Capability / parameter",
            "Transport",
            "Functional",
            "Valid n",
            "Malformed validation",
        ]
    ]
    for row in _capability_rows(bundle, endpoint)[:18]:
        rows.append(
            [
                str(row.get("capability_dimension") or "—").replace("_", " "),
                str(row.get("transport_status") or "—").replace("_", " "),
                str(row.get("functional_status") or "—").replace("_", " "),
                row.get("valid_probe_attempt_count") or "0",
                str(row.get("malformed_validation_status") or "—").replace("_", " "),
            ]
        )
    return _table(rows, styles, [42 * mm, 34 * mm, 31 * mm, 17 * mm, 45 * mm])


def _operating_table(
    bundle: Bundle, endpoint: str, styles: Mapping[str, ParagraphStyle]
) -> Table:
    capacity = _best_capacity_rows(bundle)
    soak = _best_soak_rows(bundle)
    rows = [
        [
            "Workload",
            "AIMD healthy offered",
            "Epochs",
            "2-min achieved RPM (95% CI)",
            "2-min status",
        ]
    ]
    for shape, label in SHAPES:
        aimd = capacity.get((endpoint, shape), {})
        soak_row = soak.get((endpoint, shape), {})
        lower = _num(aimd.get("capacity_lower_bound_rpm"))
        confirmations = _integer(aimd.get("candidate_rate_confirmation_epoch_count"))
        achieved = _num(
            soak_row.get("arrival_cohort_successful_rpm_including_drain_block_mean")
        )
        interval = _ci(
            soak_row.get(
                "arrival_cohort_successful_rpm_including_drain_block_mean_ci95_student_t"
            )
        )
        if achieved is not None and interval:
            achieved_text = (
                f"{_fmt(achieved, 1)} [{_fmt(interval[0], 1)}, {_fmt(interval[1], 1)}]"
            )
        else:
            achieved_text = "—"
        if not soak_row:
            soak_status = "not complete"
        elif _truth(soak_row.get("soak_acceptance_pass")):
            soak_status = "composite passed"
        else:
            soak_status = "measured; composite failed"
        rows.append(
            [
                label,
                f"≥ {_fmt(lower, 0)} RPM" if lower is not None else "—",
                str(confirmations or "—"),
                achieved_text,
                soak_status,
            ]
        )
    return _table(rows, styles, [38 * mm, 32 * mm, 15 * mm, 48 * mm, 37 * mm])


def _limits_table(
    bundle: Bundle, endpoint: str, styles: Mapping[str, ParagraphStyle]
) -> Table:
    rows = [["Envelope", "Documented", "Observed functional / realized", "Censoring"]]
    for row in _limit_rows(bundle, endpoint):
        dimension = str(row.get("dimension") or "—")
        observed = (
            row.get("maximum_functionally_valid_input_tokens")
            or row.get("maximum_realized_output_tokens")
            or row.get("observed_value")
            or "—"
        )
        rows.append(
            [
                dimension,
                _fmt(row.get("documented_value"), 0),
                _fmt(observed, 0),
                str(
                    row.get("boundary_censoring")
                    or ("exact" if _truth(row.get("boundary_exact")) else "—")
                ).replace("_", " "),
            ]
        )
    return _table(rows[:8], styles, [49 * mm, 32 * mm, 48 * mm, 39 * mm])


def _endpoint_pages(
    styles: Mapping[str, ParagraphStyle], bundle: Bundle
) -> list[Flowable]:
    inventory = _inventory(bundle)
    summaries = _endpoint_summary(bundle)
    story: list[Flowable] = []
    for index, endpoint in enumerate(REPORT_ENDPOINT_IDS, 1):
        inv = inventory[endpoint]
        summary = summaries.get(endpoint, {})
        documented = json.loads(inv.get("documented_capabilities") or "{}")
        coverage = _coverage_counts(bundle, endpoint)
        modalities = ", ".join(documented.get("modalities") or []) or "not documented"
        story.extend(
            [
                PageBreak(),
                _p(f"Endpoint {index:02d} · {_short(endpoint)}", styles["h1"]),
                _p(endpoint, styles["small"]),
                Spacer(1, 3 * mm),
                _kpis(
                    [
                        (_fmt(summary.get("request_count"), 0), "REQUESTS"),
                        (_fmt(summary.get("epoch_count"), 0), "LOAD EPOCHS"),
                        (_fmt(summary.get("error_count"), 0), "ERRORS"),
                        (
                            f"{coverage.get('completed', 0) + coverage.get('unsupported', 0)}/16",
                            "CONCLUSIVE DIMENSIONS",
                        ),
                    ],
                    styles,
                ),
                Spacer(1, 5 * mm),
                _p("Inventory", styles["h2"]),
                _table(
                    [
                        [
                            "Provider",
                            "API",
                            "Documented context",
                            "Documented output",
                            "Input $/M",
                            "Output $/M",
                        ],
                        [
                            inv.get("provider") or "—",
                            f"{inv.get('api_surface') or '—'} / {inv.get('api_version') or '—'}",
                            _fmt(inv.get("context_window"), 0),
                            _fmt(inv.get("max_output_tokens"), 0),
                            _money(inv.get("input_usd_per_million")),
                            _money(inv.get("output_usd_per_million")),
                        ],
                    ],
                    styles,
                    [27 * mm, 41 * mm, 29 * mm, 29 * mm, 22 * mm, 22 * mm],
                ),
                Spacer(1, 3 * mm),
                _p(
                    f"Documented modalities: {modalities}. Documented tools: {documented.get('tools')}. Structured output: {documented.get('structured_output')}. Prompt caching: {documented.get('prompt_caching')}.",
                    styles["body"],
                ),
                _p("Measured operating anchors", styles["h2"]),
                _operating_table(bundle, endpoint, styles),
                Spacer(1, 4 * mm),
                _p(
                    "Interpretation: the AIMD column is the highest confirmed healthy offered-rate lower bound in the matched workload. The two-minute column is achieved goodput, including drain. Neither is a contractual quota or a diurnal guarantee.",
                    styles["small"],
                ),
                PageBreak(),
                _p(f"{_short(endpoint)} · capability and limits", styles["h1"]),
                _p("Capability and validation evidence", styles["h2"]),
                _capability_table(bundle, endpoint, styles),
                Spacer(1, 4 * mm),
                _p("Context and output envelope", styles["h2"]),
                _limits_table(bundle, endpoint, styles),
                Spacer(1, 4 * mm),
                _p("Deployment guidance", styles["h2"]),
                _p(
                    "Current gate: /v1/models access recovered after the prepaid balance was replenished; pass two fresh serial streamed controls before deployment or another load wave. Then feature-gate from observed evidence, not the family name. Start below the relevant measured anchor, keep a separate concurrency ceiling, and use open-loop AIMD so rising latency cannot hide offered load. Back off on 429; retry bounded 5xx/timeouts; never retry an unchanged validation error. Re-run the exact profile after a model version, region, quota, or serving-stack change.",
                    styles["body"],
                ),
            ]
        )
    return story


def _partner_model_incident_page(
    styles: Mapping[str, ParagraphStyle], bundle: Bundle
) -> list[Flowable]:
    summary = _endpoint_summary(bundle).get(EXCLUDED_PARTNER_MODEL_ID, {})
    rows = [
        row
        for row in bundle.rows["normalized-requests.csv"]
        if row.get("endpoint_id") == EXCLUDED_PARTNER_MODEL_ID
    ]
    attributed_cost = sum(_num(row.get("estimated_cost_usd")) or 0.0 for row in rows)
    return [
        PageBreak(),
        _p("Incident appendix · excluded partner model", styles["h1"]),
        _callout(
            "Arcee Trinity is not in the hosted-only production scope",
            "DigitalOcean's current documentation lists Arcee Trinity in a separate Arcee partner-model section, before the DigitalOcean-Hosted Models table. Startup-program credits exclude third-party inference hosted outside DigitalOcean infrastructure. The benchmark should therefore not have included this endpoint under a credits-only instruction.",
            styles,
            warning=True,
        ),
        Spacer(1, 5 * mm),
        _kpis(
            [
                (_fmt(summary.get("request_count"), 0), "HISTORICAL ROWS"),
                (_money(attributed_cost), "TOKEN-ATTRIBUTED ESTIMATE"),
                ("EXCLUDED", "PRODUCTION COMPARISONS"),
            ],
            styles,
        ),
        Spacer(1, 5 * mm),
        _p(
            "The rows remain in the immutable evidence bundle for cost reconciliation and forensic reproducibility. They are excluded from every chart, portfolio KPI, endpoint recommendation, and future spend-bearing default in this encyclopedia. The code now uses a documented hosted-model allowlist and rejects an explicit Arcee selection before any request can be sent.",
            styles["body"],
        ),
        _p(
            "The request ledger alone cannot prove which exact balance line item caused the prepaid account to reach zero. However, this partner-model usage is the campaign's identified passthrough exposure and is the most plausible reason credits did not absorb all inference charges.",
            styles["body"],
        ),
    ]


def _method_pages(
    styles: Mapping[str, ParagraphStyle], bundle: Bundle
) -> list[Flowable]:
    cost = bundle.analysis.get("cost_summary", {})
    return [
        PageBreak(),
        _p("9. Measurement contract", styles["h1"]),
        _table(
            [
                ["Term", "Operational definition"],
                [
                    "Offered RPM",
                    "Arrivals scheduled by the open-loop generator per minute. Slow responses do not reduce this denominator.",
                ],
                [
                    "Achieved RPM",
                    "Completed requests divided by full elapsed wall time.",
                ],
                [
                    "Effective input TPM",
                    "Prompt tokens on successful requests divided by full elapsed wall-clock minutes.",
                ],
                [
                    "Effective output TPM",
                    "Completion tokens on successful requests divided by full elapsed wall-clock minutes.",
                ],
                [
                    "TTFT",
                    "Time from send to first streamed content event. Buffered responses are censored.",
                ],
                [
                    "Post-TTFT output proxy",
                    "Billed completion tokens divided by request time minus streamed TTFT. It is not server-internal decoder speed.",
                ],
                [
                    "AIMD lower bound",
                    "Highest offered rate with the required separated healthy confirmation epochs; right-censored if no overload knee was found.",
                ],
                [
                    "Two-minute interval",
                    "Student-t interval over four contiguous 30-second blocks. Exploratory because serial correlation is not modeled.",
                ],
                [
                    "Quality-adjusted goodput",
                    "Goodput multiplied only by scores from deterministic task checks; unscored outputs are not presumed correct.",
                ],
            ],
            styles,
            [43 * mm, 126 * mm],
        ),
        Spacer(1, 5 * mm),
        _p("Outlier policy", styles["h2"]),
        _p(
            "No timing value is trimmed, winsorized, or silently clipped. Impossible or unstable denominators are excluded by a deterministic validity rule and retained in metric-audit.csv. Qualified extremes remain in the raw derived table and are flagged. Endpoint charts compare matched workload cells only.",
            styles["body"],
        ),
        _p("Statistical unit", styles["h2"]),
        _p(
            "Serial baselines use independent request IDs. AIMD capacity uses epochs. Two-minute stability uses the four predeclared blocks. Paired quality uses pair IDs. Output tokens are never treated as independent observations. Sparse p99 values are suppressed rather than decorated with false precision.",
            styles["body"],
        ),
        PageBreak(),
        _p("10. Cost, reproducibility, and reuse", styles["h1"]),
        _kpis(
            [
                (
                    _money(cost.get("request_attributed_estimated_cost_usd")),
                    "REQUEST-ATTRIBUTED ESTIMATE",
                ),
                (
                    _money(cost.get("conservative_campaign_exposure_usd")),
                    "AUTHORITATIVE EXPOSURE",
                ),
                (_money(cost.get("cost_cap_usd")), "CAMPAIGN CAP"),
            ],
            styles,
        ),
        Spacer(1, 5 * mm),
        _p(
            "The two cost figures overlap and must not be added. Request-attributed cost uses reported token usage and includes the quarantined $9.486 historical partner-model estimate. Conservative exposure retains worst-case reservations for failed, timed-out, or usage-incomplete requests and is the budget guard.",
            styles["body"],
        ),
        _p("Reproduce a provider run", styles["h2"]),
        _table(
            [
                ["Step", "Required artifact"],
                [
                    "1",
                    "Freeze exact endpoint IDs, API surface, region, pricing, limits, and capability documentation.",
                ],
                [
                    "2",
                    "Generate the provider-neutral workload matrix and immutable request identities.",
                ],
                [
                    "3",
                    "Run low-load baselines, open-loop AIMD, two-minute candidate soaks, capability probes, and context boundaries in isolated lanes.",
                ],
                [
                    "4",
                    "Persist request, epoch, block, reservation, and cost receipts without prompts, outputs, credentials, or raw headers in the public bundle.",
                ],
                [
                    "5",
                    "Normalize with the shared metric schema; qualify timing; compute intervals at the correct sampling unit; label censoring.",
                ],
                [
                    "6",
                    "Render matched charts and endpoint profiles; run schema, matrix, sample, unit, CI, secret, and visual gates before publication.",
                ],
            ],
            styles,
            [13 * mm, 156 * mm],
        ),
        Spacer(1, 5 * mm),
        _p("What this report does not prove", styles["h2"]),
        _p(
            "It does not prove current account availability, 24-hour or regional stability, contractual quotas, server-internal prefill or decoder speed, or causal differences versus another serving provider. It provides auditable tested anchors, failure distributions, feature evidence, context behavior, and reusable experiment code from the completed measurement periods.",
            styles["body"],
        ),
    ]


def build_pdf(artifact_dir: Path, output_pdf: Path) -> Path:
    bundle = load_bundle(artifact_dir)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    charts = build_charts(bundle, output_pdf.parent / "encyclopedia-charts")
    styles = _styles()
    story: list[Flowable] = [_cover(styles, bundle)]
    story.extend(_executive_pages(styles, bundle, charts))
    story.extend(_endpoint_pages(styles, bundle))
    story.extend(_partner_model_incident_page(styles, bundle))
    story.extend(_method_pages(styles, bundle))
    document = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="DigitalOcean inference engineering encyclopedia",
        author="Sqwish Labs",
        subject="Technical operating envelopes for DigitalOcean hosted inference",
    )
    document.build(story, canvasmaker=_FooterCanvas)
    return output_pdf
