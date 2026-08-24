"""Build the clean public PDF for the direct DigitalOcean benchmark.

The builder consumes only the public analysis bundle: ``analysis.json``, a
small allow-list of normalized/derived CSV files, and images under ``charts/``.
It never reads provider-native raw request journals, prompts, model responses,
raw HTTP headers, credentials, private checkpoints, or execution logs.

``build_story`` deliberately stops before rendering so layout and content
contracts can be tested without creating a PDF.  ``build_pdf`` is the sole
rendering entry point.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import html
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    CondPageBreak,
    Flowable,
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .direct_report import (
    EXPECTED_ENDPOINT_IDS,
    PUBLIC_SAFETY_SCAN_SCHEMA,
    validate_public_analysis_contract,
)


EXPECTED_ENDPOINT_COUNT = len(EXPECTED_ENDPOINT_IDS)
PUBLIC_CSV_FILES = (
    "normalized-requests.csv",
    "normalized-epochs.csv",
    "endpoint-summary.csv",
    "endpoint-workload-metrics.csv",
    "capacity-summary.csv",
    "soak-cell-summary.csv",
    "soak-block-summary.csv",
    "quality-pair-summary.csv",
    "recovery-summary.csv",
    "coverage-ledger.csv",
    "coverage-matrix.csv",
    "scope-exclusions.csv",
    "observed-limits.csv",
    "metric-audit.csv",
    "cache-state-metrics.csv",
    "capability-evidence.csv",
)
PUBLIC_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

DO_BLUE = colors.HexColor("#0080FF")
NAVY = colors.HexColor("#071B33")
CYAN = colors.HexColor("#00B3E6")
INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#5B6875")
GRID = colors.HexColor("#CBD7E3")
LIGHT = colors.HexColor("#F4F8FC")
PALE_BLUE = colors.HexColor("#EAF5FF")
PALE_CYAN = colors.HexColor("#E8FAFD")
PALE_AMBER = colors.HexColor("#FFF6E6")
GREEN = colors.HexColor("#158553")
AMBER = colors.HexColor("#B96400")
RED = colors.HexColor("#B42318")

_ENDPOINT_KEYS = (
    "endpoint_id",
    "model_id",
    "exact_model_id",
    "endpoint",
    "model",
    "id",
)
_OPERATIONAL_HISTORY_TERMS = re.compile(
    r"\b(?:incident|checkpoint|migration|repair receipt|temporal|workflow|worker|"
    r"tmux|deployment commit|debug log|internal audit|credential|secret|raw header|"
    r"request body|model output|prompt text)\b",
    flags=re.IGNORECASE,
)
_PRIVATE_PATH = re.compile(
    r"(?:[A-Za-z]:\\(?:Users|home|private)\\\S+|/(?:home|private|tmp)/\S+)",
    flags=re.IGNORECASE,
)
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)
_SECRET_ASSIGNMENT = re.compile(
    r"\b(?:api[_ -]?key|token|password|secret|authorization)\s*[:=]\s*\S+",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class PublicReportInputs:
    """Parsed, public-only inputs for the report."""

    artifact_dir: Path
    analysis: Mapping[str, Any]
    csvs: Mapping[str, tuple[Mapping[str, str], ...]]
    charts: tuple[Path, ...]
    mode: str
    draft_watermark: bool


class NumberedCanvas(canvas.Canvas):
    """Add a restrained footer after the final page count is known."""

    def __init__(
        self, *args: Any, draft_watermark: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._saved_states: list[dict[str, Any]] = []
        self._draft_watermark = draft_watermark

    def showPage(self) -> None:  # noqa: N802 - ReportLab API spelling
        self._saved_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        page_count = len(self._saved_states)
        for state in self._saved_states:
            self.__dict__.update(state)
            self.saveState()
            if self._draft_watermark:
                # Keep the incomplete-release status visible on every page
                # without obscuring tables or charts.  The earlier diagonal
                # watermark crossed the data and made the report materially
                # harder to read.
                banner_height = 8 * mm
                self.setFillColor(PALE_AMBER)
                self.rect(
                    0,
                    A4[1] - banner_height,
                    A4[0],
                    banner_height,
                    stroke=0,
                    fill=1,
                )
                self.setFillColor(AMBER)
                self.setFont("Helvetica-Bold", 8)
                self.drawCentredString(
                    A4[0] / 2,
                    A4[1] - 5.2 * mm,
                    "DRAFT - INCOMPLETE EVIDENCE - NOT FOR PUBLICATION",
                )
            self.setStrokeColor(GRID)
            self.setLineWidth(0.45)
            self.line(17 * mm, 14 * mm, A4[0] - 17 * mm, 14 * mm)
            self.setFillColor(MUTED)
            self.setFont("Helvetica", 7.5)
            self.drawString(
                17 * mm,
                9.5 * mm,
                "DigitalOcean public inference endpoint benchmark",
            )
            self.drawRightString(
                A4[0] - 17 * mm,
                9.5 * mm,
                f"Page {self._pageNumber} of {page_count}",
            )
            self.restoreState()
            super().showPage()
        super().save()


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DirectReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=32,
            textColor=colors.white,
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "DirectReportSubtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=12,
            leading=17,
            textColor=colors.HexColor("#DDEEFF"),
            spaceAfter=7,
        ),
        "h1": ParagraphStyle(
            "DirectReportH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "DirectReportH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=DO_BLUE,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "DirectReportH3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=NAVY,
            spaceBefore=6,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "DirectReportBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12.7,
            textColor=INK,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "DirectReportSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
            spaceAfter=4,
        ),
        "table": ParagraphStyle(
            "DirectReportTable",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.1,
            leading=9.3,
            textColor=INK,
        ),
        "table_header": ParagraphStyle(
            "DirectReportTableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=9,
            textColor=colors.white,
        ),
        "callout_title": ParagraphStyle(
            "DirectReportCalloutTitle",
            fontName="Helvetica-Bold",
            fontSize=9.2,
            leading=12,
            textColor=NAVY,
            spaceAfter=3,
        ),
        "callout": ParagraphStyle(
            "DirectReportCallout",
            fontName="Helvetica",
            fontSize=8.3,
            leading=11.5,
            textColor=INK,
        ),
        "chart_title": ParagraphStyle(
            "DirectReportChartTitle",
            fontName="Helvetica-Bold",
            fontSize=9.2,
            leading=12,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "chart_caption": ParagraphStyle(
            "DirectReportChartCaption",
            fontName="Helvetica",
            fontSize=7.2,
            leading=9.5,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=7,
        ),
    }


def _plain_text(value: Any, *, limit: int = 700) -> str:
    """Return HTML-safe public prose with common secret/path shapes removed."""

    if value is None:
        return "Not reported"
    raw = str(value)
    raw = raw.replace("\u2010", "-").replace("\u2011", "-")
    raw = raw.replace("\u2012", "-").replace("\u2013", "-").replace("\u2014", "-")
    raw = _BEARER.sub("[redacted]", raw)
    raw = _SECRET_ASSIGNMENT.sub("[redacted]", raw)
    raw = _PRIVATE_PATH.sub("[private path removed]", raw)
    raw = " ".join(raw.split())
    if len(raw) > limit:
        raw = raw[: limit - 3].rstrip() + "..."
    return html.escape(raw, quote=False)


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).casefold()).strip("-")


def _first(row: Mapping[str, Any] | None, *keys: str) -> Any:
    if not row:
        return None
    for key in keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _path_value(row: Mapping[str, Any] | None, *paths: str) -> Any:
    if not row:
        return None
    for path in paths:
        current: Any = row
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                current = None
                break
            current = current[part]
        if current not in (None, "", [], {}):
            return current
    return None


def _endpoint_id(row: Mapping[str, Any] | None) -> str | None:
    value = _first(row, *_ENDPOINT_KEYS)
    return str(value) if value not in (None, "") else None


def _as_rows(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        rows: list[Mapping[str, Any]] = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                copy = dict(item)
                if _endpoint_id(copy) is None:
                    copy["endpoint_id"] = str(key)
                rows.append(copy)
        return rows
    return []


def _read_csv(path: Path) -> tuple[Mapping[str, str], ...]:
    if not path.is_file():
        return ()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return tuple(dict(row) for row in csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_manifest(root: Path, required_paths: Sequence[str]) -> None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("final build requires manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version")
        != "digitalocean_direct_public_bundle_manifest_v1"
    ):
        raise ValueError("final build requires the frozen public manifest schema")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("public manifest files must be a list")
    by_path: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("public manifest entry must be an object")
        relative = str(entry.get("path") or "")
        candidate = Path(relative)
        if (
            not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or relative in by_path
        ):
            raise ValueError("public manifest contains an unsafe or duplicate path")
        by_path[relative] = entry
    for relative in required_paths:
        entry = by_path.get(relative)
        path = (root / relative).resolve()
        if entry is None or root not in path.parents or not path.is_file():
            raise ValueError("final build manifest is missing a required artifact")
        if int(entry.get("bytes") or -1) != path.stat().st_size:
            raise ValueError("final build artifact size disagrees with manifest")
        if str(entry.get("sha256") or "") != _sha256(path):
            raise ValueError("final build artifact hash disagrees with manifest")


def _coverage_matrix_signature(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[tuple[str, str, str, int, int, int, bool], ...]:
    signature: list[tuple[str, str, str, int, int, int, bool]] = []
    for row in rows:
        try:
            planned = int(str(row.get("planned_cell_or_epoch_count")))
            observed = int(str(row.get("observed_attempt_count")))
            exclusion_count = int(str(row.get("explicit_untested_subtest_count")))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "coverage matrix contains an invalid sample count"
            ) from error
        raw_has_exclusions = row.get("has_explicit_scope_exclusions")
        if isinstance(raw_has_exclusions, bool):
            has_exclusions = raw_has_exclusions
        elif str(raw_has_exclusions).casefold() in {"true", "false"}:
            has_exclusions = str(raw_has_exclusions).casefold() == "true"
        else:
            raise ValueError("coverage matrix contains an invalid exclusion flag")
        signature.append(
            (
                str(row.get("endpoint_id") or ""),
                str(row.get("coverage_dimension") or ""),
                str(row.get("status") or ""),
                planned,
                observed,
                exclusion_count,
                has_exclusions,
            )
        )
    return tuple(signature)


def _scope_exclusion_signature(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[tuple[str, ...], ...]:
    keys = (
        "schema_version",
        "source_kind",
        "source_id",
        "source_manifest_sha256",
        "endpoint_id",
        "scope_exclusion_id",
        "measurement_label",
        "coverage_dimension",
        "status",
        "reason",
        "claim_policy",
    )
    return tuple(sorted(tuple(str(row.get(key) or "") for key in keys) for row in rows))


def _verify_scope_exclusion_ledger(
    exclusions: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    matrix: Sequence[Mapping[str, Any]],
) -> None:
    ledger_by_key = {
        (
            str(row.get("source_id") or ""),
            str(row.get("endpoint_id") or ""),
            str(row.get("scope_exclusion_id") or ""),
        ): row
        for row in ledger
        if row.get("evidence_scope") == "manifest_scope_exclusion"
        and row.get("status") == "untested"
        and row.get("claim_policy") == "explicitly_excluded_not_tested"
    }
    exclusion_keys = {
        (
            str(row.get("source_id") or ""),
            str(row.get("endpoint_id") or ""),
            str(row.get("scope_exclusion_id") or ""),
        )
        for row in exclusions
    }
    if set(ledger_by_key) != exclusion_keys:
        raise ValueError(
            "coverage-ledger.csv omits or alters explicit scope exclusions"
        )
    for exclusion in exclusions:
        key = (
            str(exclusion.get("source_id") or ""),
            str(exclusion.get("endpoint_id") or ""),
            str(exclusion.get("scope_exclusion_id") or ""),
        )
        ledger_row = ledger_by_key[key]
        expected_pairs = {
            "coverage_dimension": exclusion.get("coverage_dimension"),
            "measurement_label": exclusion.get("measurement_label"),
            "exclusion_reason": exclusion.get("reason"),
            "source_manifest_sha256": exclusion.get("source_manifest_sha256"),
            "scope_exclusion_schema_version": exclusion.get("schema_version"),
        }
        if any(
            str(ledger_row.get(field) or "") != str(expected or "")
            for field, expected in expected_pairs.items()
        ) or any(
            int(str(ledger_row.get(field) or 0)) != 0
            for field in (
                "planned_attempt_count",
                "observed_attempt_count",
                "conclusive_attempt_count",
            )
        ):
            raise ValueError(
                "coverage-ledger.csv alters explicit scope exclusion state"
            )
    matrix_rows = {
        (
            str(row.get("endpoint_id") or ""),
            str(row.get("coverage_dimension") or ""),
        ): row
        for row in matrix
    }
    exclusion_counts = Counter(
        (
            str(exclusion.get("endpoint_id") or ""),
            str(exclusion.get("coverage_dimension") or ""),
        )
        for exclusion in exclusions
    )
    for pair, count in exclusion_counts.items():
        matrix_row = matrix_rows.get(pair)
        if matrix_row is None:
            raise ValueError("coverage matrix omits an explicit scope exclusion")
        try:
            declared_count = int(
                str(matrix_row.get("explicit_untested_subtest_count") or 0)
            )
        except ValueError as error:
            raise ValueError(
                "coverage matrix has an invalid exclusion count"
            ) from error
        has_exclusions = (
            str(matrix_row.get("has_explicit_scope_exclusions")).casefold() == "true"
        )
        if declared_count != count or not has_exclusions:
            raise ValueError(
                "coverage matrix alters explicit scope-exclusion accounting"
            )
        if (
            int(str(matrix_row.get("planned_cell_or_epoch_count") or 0)) == 0
            and str(matrix_row.get("status") or "") != "untested"
        ):
            raise ValueError("exclusion-only coverage dimension must remain untested")


def load_public_inputs(
    artifact_dir: str | Path, *, mode: str = "draft"
) -> PublicReportInputs:
    """Load only the explicitly public inputs used by the PDF."""

    if mode not in {"draft", "final"}:
        raise ValueError("mode must be 'draft' or 'final'")
    root = Path(artifact_dir).resolve()
    analysis_path = root / "analysis.json"
    compressed_analysis_path = root / "analysis.json.gz"
    if analysis_path.is_file():
        analysis_text = analysis_path.read_text(encoding="utf-8")
    elif compressed_analysis_path.is_file():
        with gzip.open(
            compressed_analysis_path, "rt", encoding="utf-8"
        ) as analysis_handle:
            analysis_text = analysis_handle.read()
    else:
        raise FileNotFoundError(
            "Required public analysis file not found: expected analysis.json "
            "or analysis.json.gz"
        )
    analysis = json.loads(analysis_text)
    if not isinstance(analysis, Mapping):
        raise ValueError("analysis.json must contain one JSON object")
    validate_public_analysis_contract(analysis, require_complete=mode == "final")
    csvs = {name: _read_csv(root / name) for name in PUBLIC_CSV_FILES}
    endpoint_json = _as_rows(analysis.get("endpoint_summaries"))
    endpoint_csv = csvs.get("endpoint-summary.csv", ())
    if [str(row.get("endpoint_id") or "") for row in endpoint_json] != [
        str(row.get("endpoint_id") or "") for row in endpoint_csv
    ]:
        raise ValueError(
            "endpoint-summary.csv endpoint order disagrees with analysis.json"
        )
    for json_row, csv_row in zip(endpoint_json, endpoint_csv, strict=True):
        try:
            csv_count = int(str(csv_row.get("request_count") or ""))
            csv_cost = float(str(csv_row.get("estimated_cost_usd") or ""))
        except ValueError as error:
            raise ValueError(
                "endpoint-summary.csv has invalid cost accounting"
            ) from error
        json_count = int(json_row.get("request_count") or 0)
        json_cost = float(json_row.get("estimated_cost_usd") or 0.0)
        if csv_count != json_count or not math.isclose(
            csv_cost, json_cost, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError(
                "endpoint-summary.csv cost accounting disagrees with analysis.json"
            )
    matrix_json = _as_rows(analysis.get("coverage_matrix"))
    matrix_csv = csvs.get("coverage-matrix.csv", ())
    if _coverage_matrix_signature(matrix_json) != _coverage_matrix_signature(
        matrix_csv
    ):
        raise ValueError("coverage-matrix.csv disagrees with analysis.json")
    exclusions_json = _as_rows(analysis.get("scope_exclusions"))
    exclusions_csv = csvs.get("scope-exclusions.csv", ())
    if _scope_exclusion_signature(exclusions_json) != _scope_exclusion_signature(
        exclusions_csv
    ):
        raise ValueError("scope-exclusions.csv disagrees with analysis.json")
    _verify_scope_exclusion_ledger(
        exclusions_json,
        csvs.get("coverage-ledger.csv", ()),
        matrix_csv,
    )
    if mode == "final":
        missing_csvs = [
            name for name in PUBLIC_CSV_FILES if not (root / name).is_file()
        ]
        if missing_csvs:
            raise ValueError("final build is missing required derived CSV files")
        safety_path = root / "public-safety-scan.json"
        if not safety_path.is_file():
            raise ValueError("final build requires public-safety-scan.json")
        safety = json.loads(safety_path.read_text(encoding="utf-8"))
        if (
            not isinstance(safety, Mapping)
            or safety.get("schema_version") != PUBLIC_SAFETY_SCAN_SCHEMA
            or safety.get("passed") is not True
        ):
            raise ValueError("final build requires a passing public safety scan")
    output_files = analysis.get("output_files")
    declared = (
        output_files.get("charts", ()) if isinstance(output_files, Mapping) else ()
    )
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
        raise ValueError("analysis output_files.charts must be a list")
    charts: list[Path] = []
    for raw in declared:
        relative = Path(str(raw))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "charts"
            or relative.suffix.casefold() not in PUBLIC_IMAGE_SUFFIXES
        ):
            raise ValueError("analysis declares an unsafe chart path")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("analysis declares a missing chart")
        charts.append(path)
    if mode == "final":
        _verify_manifest(
            root,
            [
                "analysis.json",
                "public-safety-scan.json",
                *PUBLIC_CSV_FILES,
                *(str(Path(raw)).replace("\\", "/") for raw in declared),
            ],
        )
    return PublicReportInputs(
        root,
        analysis,
        csvs,
        tuple(charts),
        mode,
        mode == "draft",
    )


def _format_number(value: Any, *, digits: int = 2) -> str:
    if value in (None, ""):
        return "Not reported"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _plain_text(value)
    if not math.isfinite(number):
        return "Not reported"
    if number == 0:
        return "0"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:,.{digits}f}".rstrip("0").rstrip(".")


def _format_percent(value: Any) -> str:
    if value in (None, ""):
        return "Not reported"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _plain_text(value)
    if 0 <= number <= 1:
        number *= 100
    return f"{number:.1f}%"


def _format_money(value: Any) -> str:
    if value in (None, ""):
        return "Not reported"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _plain_text(value)
    if not math.isfinite(number):
        return "Not reported"
    return f"${number:,.6f}" if abs(number) < 1 else f"${number:,.2f}"


def _display_value(label: str, value: Any) -> str:
    lower = label.casefold()
    if "rate" in lower and "rpm" not in lower and "rps" not in lower:
        return _format_percent(value)
    if "%" in label or "success" in lower or "error" in lower or "quality" in lower:
        return _format_percent(value)
    if any(token in lower for token in ("cost", "price", "spend", "exposure")):
        return _format_money(value)
    if any(
        token in lower
        for token in ("rpm", "rps", "tpm", "seconds", "tokens", "concurrency")
    ):
        return _format_number(value)
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return _plain_text(value)


def _humanize_machine_value(value: Any) -> str:
    """Render persisted machine-state labels as compact engineering language."""
    if value in (None, ""):
        return "Not reported"
    text = str(value)
    labels = {
        "input32k_short": "32K input / short output",
        "short_long": "Short input / long output",
        "short_short": "Short input / short output",
        "mixed": "Heterogeneous mixed",
        "confirmed_bracketed_interval": "Confirmed bracket",
        "confirmed_right_censored_lower_bound": "Confirmed lower bound (right-censored)",
        "unconfirmed_healthy_observation_only": "Exploratory healthy observation",
        "baseline_transport_gate_failed": "Baseline transport gate failed",
        "complete": "Complete",
        "accepted_and_retrieval_valid": "Accepted; retrieval passed",
        "accepted_and_retrieval_valid_through": "Accepted; retrieval passed through",
        "accepted_and_retrieval_valid_through_boundary": "Accepted; retrieval passed through boundary",
        "transport_accepted_but_retrieval_not_verified": "Accepted; retrieval not verified",
        "realized_generation_observed": "Realized generation observed",
        "observed_functional": "Functional in observed tests",
        "observed_rejected_or_unsupported": "Rejected or unsupported in observed tests",
        "untested_or_inconclusive": "Untested or inconclusive",
        "inconclusive": "Inconclusive",
    }
    if text in labels:
        return labels[text]
    return text.replace("_", " ").strip().capitalize()


def _paragraph(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_plain_text(value), style)


def _table(
    rows: Sequence[Sequence[Any]],
    widths: Sequence[float],
    styles: Mapping[str, ParagraphStyle],
    *,
    repeat_header: bool = True,
) -> Table:
    normalized: list[list[Flowable]] = []
    for row_index, row in enumerate(rows):
        style = (
            styles["table_header"]
            if repeat_header and row_index == 0
            else styles["table"]
        )
        normalized.append([_paragraph(value, style) for value in row])
    result = Table(
        normalized,
        colWidths=list(widths),
        repeatRows=1 if repeat_header else 0,
        hAlign="LEFT",
    )
    commands: list[tuple[Any, ...]] = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if repeat_header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ]
        )
    else:
        commands.append(("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT]))
    result.setStyle(TableStyle(commands))
    return result


def _kv_table(
    values: Sequence[tuple[str, Any]],
    styles: Mapping[str, ParagraphStyle],
) -> Table:
    rows = [[label, _display_value(label, value)] for label, value in values]
    return _table(rows, [54 * mm, 120 * mm], styles, repeat_header=False)


def _section(
    styles: Mapping[str, ParagraphStyle], number: str, title: str
) -> list[Flowable]:
    return [
        CondPageBreak(32 * mm),
        Paragraph(f"{_plain_text(number)}. {_plain_text(title)}", styles["h1"]),
        HRFlowable(width="100%", thickness=1.1, color=DO_BLUE, spaceAfter=8),
    ]


def _callout(
    styles: Mapping[str, ParagraphStyle],
    title: str,
    simple: str,
    rigorous: str,
    *,
    background: colors.Color = PALE_BLUE,
) -> Table:
    contents = [
        Paragraph(_plain_text(title), styles["callout_title"]),
        Paragraph(f"<b>Plain language:</b> {_plain_text(simple)}", styles["callout"]),
        Spacer(1, 2),
        Paragraph(
            f"<b>Technical definition:</b> {_plain_text(rigorous)}", styles["callout"]
        ),
    ]
    result = Table([[contents]], colWidths=[174 * mm], hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.7, DO_BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return result


def _missing(
    styles: Mapping[str, ParagraphStyle],
    what: str,
    reason: str = "not supplied in the public analysis bundle",
) -> Table:
    return _callout(
        styles,
        f"{what}: inconclusive",
        "There is not enough public evidence to make this comparison.",
        f"The value is explicitly unavailable because it was {reason}. Missing is never interpreted as zero or as a pass.",
        background=PALE_AMBER,
    )


def _matching(
    endpoint: str, rows: Iterable[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    target = _slug(endpoint)
    return [row for row in rows if _slug(_endpoint_id(row) or "") == target]


def _endpoint_records(inputs: PublicReportInputs) -> list[Mapping[str, Any]]:
    analysis = inputs.analysis
    inventory = _as_rows(analysis.get("endpoint_inventory"))
    candidate_rows: list[Mapping[str, Any]] = list(inventory)
    for key in ("endpoint_summaries", "capacity_summaries", "observed_limits"):
        candidate_rows.extend(_as_rows(analysis.get(key)))
    for filename in (
        "endpoint-summary.csv",
        "endpoint-workload-metrics.csv",
        "capacity-summary.csv",
        "coverage-ledger.csv",
        "observed-limits.csv",
    ):
        candidate_rows.extend(inputs.csvs.get(filename, ()))

    by_slug: dict[str, Mapping[str, Any]] = {}
    ordered: list[str] = []
    for row in candidate_rows:
        endpoint = _endpoint_id(row)
        if not endpoint:
            continue
        key = _slug(endpoint)
        if not key:
            continue
        if key not in by_slug:
            by_slug[key] = row
            ordered.append(key)
        elif row in inventory:
            by_slug[key] = row

    records = [by_slug[key] for key in ordered]
    while len(records) < EXPECTED_ENDPOINT_COUNT:
        slot = len(records) + 1
        records.append(
            {
                "endpoint_id": f"Endpoint evidence not supplied - slot {slot}",
                "_placeholder": True,
            }
        )
    return records


def _summary_for(endpoint: str, analysis: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = _matching(endpoint, _as_rows(analysis.get("endpoint_summaries")))
    return matches[0] if matches else {}


def _coalesce_rows(
    endpoint: str,
    analysis_value: Any,
    csv_rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    rows = _matching(endpoint, _as_rows(analysis_value))
    # The JSON analysis is authoritative and preserves nested CI objects. CSV
    # is a fallback transport, not a second sample source to concatenate.
    if not rows:
        rows = _matching(endpoint, csv_rows)
    seen: set[str] = set()
    result: list[Mapping[str, Any]] = []
    for row in rows:
        digest = json.dumps(dict(row), sort_keys=True, default=str)
        if digest not in seen:
            result.append(row)
            seen.add(digest)
    return result


def _coverage_rows_for_endpoint(
    inputs: PublicReportInputs, endpoint: str
) -> list[Mapping[str, Any]]:
    rows = _matching(endpoint, inputs.csvs.get("coverage-matrix.csv", ()))
    if not rows:
        rows = _matching(endpoint, inputs.csvs.get("coverage-ledger.csv", ()))
    return rows


def _scope_exclusions_for_endpoint(
    inputs: PublicReportInputs, endpoint: str
) -> list[Mapping[str, Any]]:
    return _matching(endpoint, inputs.csvs.get("scope-exclusions.csv", ()))


def _scope_exclusion_table(
    rows: Sequence[Mapping[str, Any]],
    styles: Mapping[str, ParagraphStyle],
    *,
    include_endpoint_count: bool,
) -> Table | None:
    if not rows:
        return None
    table_rows: list[list[Any]] = [
        [
            "Explicitly untested measurement",
            "Coverage dimension",
            "Applies to" if include_endpoint_count else "State",
            "Reason",
        ]
    ]
    grouped: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (
            str(
                row.get("measurement_label")
                or row.get("scope_exclusion_id")
                or "Untitled exclusion"
            ),
            str(row.get("coverage_dimension") or "Not mapped"),
            str(row.get("reason") or "No reason supplied"),
        )
        endpoint = _endpoint_id(row)
        if endpoint:
            grouped[key].add(endpoint)
    for (label, dimension, reason), endpoints in sorted(grouped.items()):
        applies = (
            f"{len(endpoints)} endpoint{'s' if len(endpoints) != 1 else ''}"
            if include_endpoint_count
            else "Untested"
        )
        table_rows.append([label, dimension, applies, reason])
    return _table(table_rows, [45 * mm, 34 * mm, 24 * mm, 71 * mm], styles)


def _ci_text(row: Mapping[str, Any], metric: str) -> str:
    direct = _first(row, f"{metric}_ci95", f"{metric}_95ci", f"{metric}_ci")
    if isinstance(direct, Mapping):
        lower = _first(direct, "ci95_low", "lower", "low", "lo", "lower_bound")
        upper = _first(direct, "ci95_high", "upper", "high", "hi", "upper_bound")
    elif isinstance(direct, (list, tuple)) and len(direct) >= 2:
        lower, upper = direct[0], direct[1]
    else:
        lower = _first(
            row,
            f"{metric}_ci95_lower",
            f"{metric}_ci_lower",
            f"{metric}_lower",
        )
        upper = _first(
            row,
            f"{metric}_ci95_upper",
            f"{metric}_ci_upper",
            f"{metric}_upper",
        )
    if lower in (None, "") or upper in (None, ""):
        return "Exploratory / CI unavailable"
    return f"{_format_number(lower)} to {_format_number(upper)}"


def _metric_row_value(row: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    return _first(row, *aliases)


def _capacity_table(
    rows: Sequence[Mapping[str, Any]], styles: Mapping[str, ParagraphStyle]
) -> Table | None:
    if not rows:
        return None
    table_rows: list[list[Any]] = [
        [
            "Workload",
            "Run / source",
            "Evidence",
            "Confirmed healthy offered RPM",
            "Achieved RPM 95% CI",
            "Input TPM 95% CI",
            "Output TPM 95% CI",
        ]
    ]
    for row in rows[:18]:
        workload = _metric_row_value(
            row, ("workload", "workload_id", "load_shape", "shape", "task_family")
        )
        evidence = _metric_row_value(
            row,
            (
                "capacity_claim",
                "evidence_status",
                "capacity_status",
                "status",
                "qualification",
            ),
        )
        rpm = _metric_row_value(
            row,
            (
                "confirmed_healthy_offered_rpm",
                "capacity_lower_bound_rpm",
            ),
        )
        table_rows.append(
            [
                _humanize_machine_value(workload),
                _first(row, "source_id", "run_id", "campaign_id") or "Not reported",
                _humanize_machine_value(evidence or "Exploratory"),
                _format_number(rpm),
                _ci_text(row, "achieved_rpm"),
                _ci_text(row, "effective_input_tpm"),
                _ci_text(row, "effective_output_tpm"),
            ]
        )
    return _table(
        table_rows,
        [25 * mm, 27 * mm, 25 * mm, 23 * mm, 22 * mm, 22 * mm, 30 * mm],
        styles,
    )


def _range_text(value: Any, *, suffix: str = "") -> str:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        low, high = value[0], value[1]
        if low == high:
            return f"{_format_number(low)}{suffix}"
        return f"{_format_number(low)}-{_format_number(high)}{suffix}"
    return "Not observed"


def _capacity_contract_table(
    rows: Sequence[Mapping[str, Any]], styles: Mapping[str, ParagraphStyle]
) -> Table | None:
    qualified = [
        row
        for row in rows
        if int(float(row.get("candidate_rate_confirmation_epoch_count") or 0)) >= 3
    ]
    if not qualified:
        return None
    table_rows: list[list[Any]] = [
        [
            "Workload / run",
            "Epochs / rows",
            "Realized prompt tokens",
            "Realized output tokens",
            "Stream mode",
            "Concurrency ceiling",
            "Epoch duration",
        ]
    ]
    for row in qualified[:18]:
        stream_modes = row.get("candidate_stream_modes")
        stream_text = (
            ", ".join(str(value) for value in stream_modes)
            if isinstance(stream_modes, Sequence)
            and not isinstance(stream_modes, (str, bytes, bytearray))
            and stream_modes
            else "Not recorded"
        )
        table_rows.append(
            [
                f"{_humanize_machine_value(row.get('shape'))}\n{row.get('source_id')}",
                (
                    f"{int(float(row.get('candidate_rate_confirmation_epoch_count') or 0))} / "
                    f"{int(float(row.get('candidate_request_row_count') or 0))}"
                ),
                _range_text(row.get("candidate_realized_input_tokens_range")),
                _range_text(row.get("candidate_realized_output_tokens_range")),
                stream_text,
                _range_text(row.get("candidate_concurrency_ceiling_range")),
                _range_text(
                    row.get("candidate_epoch_duration_seconds_range"), suffix=" s"
                ),
            ]
        )
    return _table(
        table_rows,
        [34 * mm, 21 * mm, 27 * mm, 27 * mm, 24 * mm, 22 * mm, 20 * mm],
        styles,
    )


def _capacity_latency_table(
    rows: Sequence[Mapping[str, Any]], styles: Mapping[str, ParagraphStyle]
) -> Table | None:
    qualified = [
        row
        for row in rows
        if int(float(row.get("candidate_rate_confirmation_epoch_count") or 0)) >= 3
    ]
    if not qualified:
        return None
    table_rows: list[list[Any]] = [
        [
            "Workload / run",
            "Success / scheduled",
            "429 / 5xx / timeout",
            "TTFT p50 seconds (95% epoch CI)",
            "Latency p95 seconds (95% epoch CI)",
        ]
    ]
    for row in qualified[:18]:
        table_rows.append(
            [
                f"{_humanize_machine_value(row.get('shape'))}\n{row.get('source_id')}",
                (
                    f"{int(float(row.get('candidate_successful_request_count') or 0))} / "
                    f"{int(float(row.get('candidate_scheduled_request_count') or 0))}"
                ),
                (
                    f"{int(float(row.get('candidate_rate_limit_count') or 0))} / "
                    f"{int(float(row.get('candidate_server_error_count') or 0))} / "
                    f"{int(float(row.get('candidate_timeout_count') or 0))}"
                ),
                (
                    f"{_format_number(row.get('ttft_p50_seconds'))} "
                    f"({_ci_text(row, 'ttft_p50_seconds')})"
                ),
                (
                    f"{_format_number(row.get('latency_p95_seconds'))} "
                    f"({_ci_text(row, 'latency_p95_seconds')})"
                ),
            ]
        )
    return _table(
        table_rows,
        [38 * mm, 29 * mm, 29 * mm, 39 * mm, 39 * mm],
        styles,
    )


def _soak_table(
    rows: Sequence[Mapping[str, Any]], styles: Mapping[str, ParagraphStyle]
) -> Table | None:
    if not rows:
        return None
    table_rows: list[list[Any]] = [
        [
            "Workload",
            "Offered arrival rate",
            "State / acceptance",
            "Drain-inclusive cohort RPM mean (95% block t interval)",
            "Drain-inclusive input TPM",
            "Drain-inclusive output TPM",
        ]
    ]
    for row in rows[:8]:
        interval = row.get(
            "arrival_cohort_successful_rpm_including_drain_block_mean_ci95_student_t"
        )
        if isinstance(interval, (list, tuple)) and len(interval) == 2:
            interval_text = (
                f"{_format_number(interval[0])} to {_format_number(interval[1])}"
            )
        else:
            interval_text = "Exploratory / CI unavailable"
        mean_value = _format_number(
            row.get("arrival_cohort_successful_rpm_including_drain_block_mean")
        )
        table_rows.append(
            [
                _humanize_machine_value(row.get("shape")),
                (
                    f"{_format_number(row.get('two_minute_soak_verified_rps'))} RPS"
                    if row.get("two_minute_soak_verified_rps") is not None
                    else "Not soak-verified"
                ),
                (
                    f"{_humanize_machine_value(row.get('status'))}; pass"
                    if row.get("soak_acceptance_pass") is True
                    else (
                        f"{_humanize_machine_value(row.get('status'))}; measured fail"
                        if row.get("soak_acceptance_pass") is False
                        else f"{_humanize_machine_value(row.get('status'))}; inconclusive"
                    )
                ),
                f"{mean_value} ({interval_text})",
                _format_number(
                    row.get(
                        "arrival_cohort_effective_input_tpm_including_drain_block_mean"
                    )
                ),
                _format_number(
                    row.get(
                        "arrival_cohort_effective_output_tpm_including_drain_block_mean"
                    )
                ),
            ]
        )
    return _table(
        table_rows,
        [29 * mm, 27 * mm, 31 * mm, 43 * mm, 22 * mm, 22 * mm],
        styles,
    )


def _recovery_table(
    rows: Sequence[Mapping[str, Any]], styles: Mapping[str, ParagraphStyle]
) -> Table | None:
    if not rows:
        return None
    table_rows: list[list[Any]] = [
        [
            "Workload",
            "Target / realized RPS",
            "Recovery decision",
            "Success rate (95% request CI)",
            "TTFT / latency p95 seconds",
        ]
    ]
    for row in rows[:12]:
        success_interval = row.get("success_rate_ci95_wilson")
        if isinstance(success_interval, (list, tuple)) and len(success_interval) >= 2:
            success_text = (
                f"{_format_percent(row.get('success_rate'))} "
                f"({_format_percent(success_interval[0])} to "
                f"{_format_percent(success_interval[1])})"
            )
        else:
            success_text = _format_percent(row.get("success_rate"))
        table_rows.append(
            [
                _humanize_machine_value(row.get("shape")),
                (
                    f"{_format_number(row.get('offered_rps_target'))} / "
                    f"{_format_number(row.get('offered_rps_realized_schedule'))}"
                ),
                (
                    "Pass"
                    if row.get("predeclared_recovery_pass") is True
                    else "Measured fail"
                ),
                success_text,
                (
                    f"{_format_number(row.get('ttft_p95_seconds'))} / "
                    f"{_format_number(row.get('latency_p95_seconds'))}"
                ),
            ]
        )
    return _table(
        table_rows,
        [33 * mm, 32 * mm, 28 * mm, 47 * mm, 34 * mm],
        styles,
    )


def _outlier_traceability_table(
    rows: Sequence[Mapping[str, Any]], styles: Mapping[str, ParagraphStyle]
) -> Table | None:
    qualified = [
        row
        for row in rows
        if _numeric_value(row.get("post_ttft_output_tps_proxy")) is not None
        and str(row.get("classification") or "").startswith("valid_")
    ]
    qualified.sort(
        key=lambda row: _numeric_value(row.get("post_ttft_output_tps_proxy")) or 0,
        reverse=True,
    )
    if not qualified:
        return None
    table_rows: list[list[Any]] = [
        [
            "Endpoint / workload",
            "Output tokens",
            "TTFT s",
            "Total s",
            "Post-TTFT denominator s",
            "Cache state",
            "Qualified proxy tokens/s",
            "Classification",
        ]
    ]
    for row in qualified[:10]:
        total = _numeric_value(row.get("request_seconds"))
        ttft = _numeric_value(row.get("streamed_ttft_seconds"))
        denominator = total - ttft if total is not None and ttft is not None else None
        table_rows.append(
            [
                f"{row.get('endpoint_id')}\n{_humanize_machine_value(row.get('workload'))}",
                _format_number(row.get("output_tokens")),
                _format_number(ttft),
                _format_number(total),
                _format_number(denominator),
                _humanize_machine_value(row.get("cache_state")),
                _format_number(row.get("post_ttft_output_tps_proxy")),
                _humanize_machine_value(row.get("classification")),
            ]
        )
    return _table(
        table_rows,
        [36 * mm, 17 * mm, 16 * mm, 16 * mm, 23 * mm, 24 * mm, 21 * mm, 21 * mm],
        styles,
    )


def _limits_table(
    rows: Sequence[Mapping[str, Any]], styles: Mapping[str, ParagraphStyle]
) -> Table | None:
    if not rows:
        return None
    table_rows: list[list[Any]] = [
        [
            "Limit or capability",
            "Documented",
            "Max accepted / supported",
            "Max retrieval-valid / realized",
            "First explicit rejection",
            "Censoring / finding",
        ]
    ]
    for row in rows[:20]:
        dimension = str(
            _first(row, "dimension", "capability", "limit_name", "parameter", "test")
            or "Not reported"
        )
        if dimension == "combined prompt + requested output":
            accepted = row.get("maximum_accepted_combined_target_tokens")
            valid = row.get("maximum_functionally_valid_combined_target_tokens")
            rejected = row.get("minimum_rejected_estimated_combined_target_tokens")
        elif dimension == "prompt context window":
            accepted = row.get("maximum_accepted_input_tokens")
            valid = row.get("maximum_functionally_valid_input_tokens")
            rejected = row.get("minimum_rejected_estimated_input_tokens")
        elif dimension == "output limit":
            accepted = row.get("maximum_accepted_requested_output_target")
            valid = row.get("maximum_realized_output_tokens")
            rejected = row.get("minimum_rejected_requested_output_target")
        else:
            accepted = _first(row, "observed_value", "observed_limit", "observed")
            valid = row.get("finding")
            rejected = None
        table_rows.append(
            [
                dimension,
                _first(
                    row,
                    "documented_value",
                    "documented_limit",
                    "documented",
                    "advertised",
                )
                or "Not reported",
                accepted if accepted is not None else "Not observed",
                valid if valid is not None else "Not verified",
                rejected if rejected is not None else "Not observed",
                _humanize_machine_value(
                    _first(row, "boundary_censoring", "finding", "status", "result")
                    or "Inconclusive"
                ),
            ]
        )
    return _table(
        table_rows,
        [30 * mm, 27 * mm, 28 * mm, 31 * mm, 27 * mm, 31 * mm],
        styles,
    )


def _coverage_table(
    rows: Sequence[Mapping[str, Any]], styles: Mapping[str, ParagraphStyle]
) -> Table | None:
    if not rows:
        return None
    statuses = Counter(
        str(
            _first(row, "coverage_status", "status", "result") or "unreported"
        ).casefold()
        for row in rows
    )
    normalized = {
        "completed": sum(
            value
            for key, value in statuses.items()
            if key in {"completed", "complete", "passed", "supported"}
        ),
        "inconclusive": sum(
            value
            for key, value in statuses.items()
            if "inconclusive" in key or "partial" in key
        ),
        "unsupported": sum(
            value
            for key, value in statuses.items()
            if "unsupported" in key or "not_supported" in key
        ),
        "untested": sum(
            value
            for key, value in statuses.items()
            if key in {"untested", "not_run", "skipped", "planned"}
        ),
    }
    table_rows: list[list[Any]] = [["Coverage state", "Cells", "Meaning"]]
    meanings = {
        "completed": "A planned cell produced usable evidence.",
        "inconclusive": "The attempt ran, but evidence was insufficient for a claim.",
        "unsupported": "The API or endpoint rejected the capability as unsupported.",
        "untested": "No result exists; this is not a pass or a zero.",
    }
    for key in ("completed", "inconclusive", "unsupported", "untested"):
        table_rows.append([key.title(), normalized[key], meanings[key]])
    return _table(table_rows, [38 * mm, 22 * mm, 114 * mm], styles)


def _capability_evidence_table(
    rows: Sequence[Mapping[str, Any]], styles: Mapping[str, ParagraphStyle]
) -> Table | None:
    if not rows:
        return None
    output: list[list[Any]] = [
        [
            "Capability dimension",
            "Valid-call transport",
            "Functional result",
            "Malformed validation",
            "n",
        ]
    ]
    for row in sorted(
        rows, key=lambda value: str(value.get("capability_dimension") or "")
    ):
        valid_n = int(float(row.get("valid_probe_attempt_count") or 0))
        valid_2xx = int(float(row.get("valid_probe_2xx_count") or 0))
        scored_n = int(float(row.get("functional_scored_count") or 0))
        passed_n = int(float(row.get("functional_pass_count") or 0))
        malformed_n = int(float(row.get("malformed_validation_attempt_count") or 0))
        output.append(
            [
                _humanize_machine_value(row.get("capability_dimension")),
                f"{_humanize_machine_value(row.get('transport_status'))} ({valid_2xx}/{valid_n} 2xx)",
                f"{_humanize_machine_value(row.get('functional_status'))} ({passed_n}/{scored_n})",
                f"{_humanize_machine_value(row.get('malformed_validation_status'))} (n={malformed_n})",
                valid_n + malformed_n,
            ]
        )
    return _table(
        output,
        [34 * mm, 42 * mm, 36 * mm, 45 * mm, 17 * mm],
        styles,
    )


def _extract_limitations(value: Any, *, endpoint: str | None = None) -> list[str]:
    values: list[Any]
    if isinstance(value, list):
        values = value
    elif isinstance(value, Mapping):
        values = list(value.values())
    elif value in (None, ""):
        values = []
    else:
        values = [value]
    result: list[str] = []
    endpoint_slug = _slug(endpoint or "")
    for item in values:
        if isinstance(item, Mapping):
            scope = _first(item, "endpoint_id", "model_id", "endpoint", "scope")
            if (
                endpoint
                and scope
                and _slug(scope) not in {endpoint_slug, "all", "global"}
            ):
                continue
            text_value = _first(
                item, "limitation", "statement", "description", "impact", "reason"
            )
        else:
            text_value = item
        if text_value in (None, ""):
            continue
        text_value = str(text_value)
        if _OPERATIONAL_HISTORY_TERMS.search(text_value):
            continue
        result.append(text_value)
    return result


def _definition_cards(
    styles: Mapping[str, ParagraphStyle], analysis: Mapping[str, Any]
) -> list[Flowable]:
    defaults = [
        (
            "Time to first token (TTFT)",
            "How long a streaming caller waits before any answer text arrives.",
            "TTFT is the monotonic-clock interval from request dispatch to receipt of the first non-empty response token. It includes network, queue, and end-to-end prefill time; it is not a direct server-side prefill measurement.",
        ),
        (
            "End-to-end latency",
            "How long the whole request takes from send to completion or terminal failure.",
            "Latency is terminal timestamp minus dispatch timestamp. Percentiles use independent requests or epochs as the sampling unit and include failures in reliability summaries.",
        ),
        (
            "Aggregate output goodput",
            "How many successfully billed answer tokens finish per wall-clock second or minute.",
            "Headline output goodput is successful server-reported completion tokens divided by the complete epoch, soak block, or source-active wall-clock interval. It includes reasoning tokens when the provider bills and reports them.",
        ),
        (
            "Post-TTFT service-output proxy",
            "A conservative per-request view of output delivery after streaming begins.",
            "Billed completion tokens / (request end - streamed TTFT). This includes network and response buffering and is not direct decoder speed. Sub-100 ms intervals are timing-unstable and are explicitly censored from this rate but retained in aggregate goodput. The first-to-last SSE-event span is audit-only because one event can contain many tokens.",
        ),
        (
            "Cache stratum",
            "Whether DigitalOcean explicitly reported reused prompt tokens, a miss, or no cache counter.",
            "TTFT is never pooled across observed cache hits, observed misses, and unknown cache state. Prompt tokens / TTFT is headline-eligible only for an explicit cache miss. Missing counters are not treated as zero.",
        ),
        (
            "Effective input and output TPM",
            "How many useful prompt and answer tokens the endpoint successfully handles per minute.",
            "Effective input TPM = successful server-reported prompt tokens / elapsed minutes. Effective output TPM = successful server-reported completion tokens / elapsed minutes. Offered and failed tokens are reported separately.",
        ),
        (
            "Goodput",
            "Throughput that actually produced usable answers.",
            "Goodput counts successful, non-truncated requests that pass the workload validity rule. Quality-adjusted goodput additionally requires deterministic task correctness or a preregistered quality threshold.",
        ),
        (
            "AIMD capacity sweep",
            "The test raises load while healthy and backs off sharply after congestion.",
            "Additive increase, multiplicative decrease uses open-loop arrivals with a separate concurrency ceiling. Three separated healthy confirmation epochs establish a confirmed healthy offered arrival rate, not completed or sustained capacity. Achieved drain-inclusive goodput remains a separate interval; an observed maximum without confirmations is exploratory.",
        ),
        (
            "95% confidence interval",
            "A range showing how uncertain the estimate is across independent repetitions.",
            "Intervals use independent requests or load epochs, never individual output tokens. Sparse tail percentiles are marked exploratory; p99 is omitted unless roughly 1,000 relevant observations support it.",
        ),
        (
            "Buffered response",
            "A non-streaming request returns the whole answer at once.",
            "Buffered calls expose full-response latency but not token-level TTFT. Their response-arrival timestamp is censored from TTFT, prefill, and post-TTFT per-sequence curves.",
        ),
    ]
    supplied = analysis.get("metric_definitions")
    cards: list[tuple[str, str, str]] = []
    if isinstance(supplied, list):
        for item in supplied:
            if not isinstance(item, Mapping):
                continue
            name = _first(item, "name", "metric", "term")
            simple = _first(item, "plain_language", "simple", "meaning")
            exact = _first(item, "technical_definition", "definition", "formula")
            if (
                name
                and simple
                and exact
                and not _OPERATIONAL_HISTORY_TERMS.search(str(exact))
            ):
                cards.append((str(name), str(simple), str(exact)))
    elif isinstance(supplied, Mapping):
        for name, item in supplied.items():
            if isinstance(item, Mapping):
                simple = _first(item, "plain_language", "simple", "meaning")
                exact = _first(item, "technical_definition", "definition", "formula")
                if (
                    simple
                    and exact
                    and not _OPERATIONAL_HISTORY_TERMS.search(str(exact))
                ):
                    cards.append((str(name), str(simple), str(exact)))
    if not cards:
        cards = defaults
    return [
        _callout(styles, name, simple, exact, background=PALE_CYAN)
        for name, simple, exact in cards[:12]
    ]


def _chart_title(path: Path) -> str:
    words = path.stem.replace("_", " ").replace("-", " ").split()
    return " ".join(
        word.upper()
        if word.casefold() in {"rpm", "rps", "tpm", "ttft", "aimd"}
        else word.capitalize()
        for word in words
    )


def _chart_block(
    path: Path,
    styles: Mapping[str, ParagraphStyle],
    *,
    max_height: float = 112 * mm,
) -> Flowable:
    try:
        with PILImage.open(path) as source:
            width, height = source.size
        if width <= 0 or height <= 0:
            raise ValueError("image has no dimensions")
        scale = min((174 * mm) / width, max_height / height)
        image = RLImage(str(path), width=width * scale, height=height * scale)
        stem = path.stem.casefold()
        captions = {
            "metric-outlier-audit": (
                "The red distribution is the rejected first-to-last SSE-event-span "
                "calculation. The blue distribution uses request end minus streamed "
                "TTFT. All valid extremes remain; invalid observations are censored "
                "with a reason rather than trimmed."
            ),
            "ttft-input-cache-strata": (
                "Each point is one streamed, single-choice request. Log axes preserve "
                "the full context and latency ranges. Buffered calls are excluded; "
                "cache hit, miss, and unknown strata are never pooled."
            ),
            "output-post-ttft-proxy": (
                "The y-axis is billed completion tokens divided by request end minus "
                "streamed TTFT. It is an end-to-end delivery proxy, not server decode "
                "speed. Multi-choice responses are excluded from this per-sequence view."
            ),
            "soak-four-block-stability": (
                "Each cell uses four fixed 30-second analysis_block_id units. Missing "
                "cells are not zero; they mean the block estimand was unavailable."
            ),
            "capability-transport-functional-matrix": (
                "Valid-call transport support and task correctness are separate. "
                "Malformed-input rejection is reported in the capability evidence "
                "table and never used to infer lack of valid-call support."
            ),
            "paired-quality-load-effect": (
                "Only exact-payload matched low-load/near-load pairs enter the figure. "
                "Pair deltas are averaged within each predeclared soak analysis block; "
                "the Student-t interval uses those block means and reports both counts."
            ),
            "matched-cost-performance": (
                "Every point is one exact source, endpoint, workload shape, and confirmed "
                "healthy offered rate with at least three confirmation epochs. Bars use "
                "the epoch output-TPM values. Phases, offered rates, and runs are never pooled."
            ),
        }
        caption = captions.get(
            stem,
            "Intervals and evidence labels come from the public analysis bundle. "
            "Blank or missing cells mean untested, censored, or inconclusive - not zero.",
        )
        if stem.startswith("aimd-controller-"):
            caption = (
                "Chronology is connected only within one exact source run and workload. "
                "The blue step is offered arrival load; the dark line is drain-inclusive "
                "successful throughput. Markers identify confirmations, congestion, and "
                "recovery/fallback without connecting different endpoints or workloads."
            )
        return KeepTogether(
            [
                Paragraph(_plain_text(_chart_title(path)), styles["chart_title"]),
                image,
                Paragraph(_plain_text(caption), styles["chart_caption"]),
            ]
        )
    except Exception as error:  # chart corruption must not erase the report
        return _missing(
            styles,
            f"Chart {_chart_title(path)}",
            f"unreadable ({type(error).__name__})",
        )


_OVERVIEW_SHAPES: tuple[tuple[str, str], ...] = (
    ("short_short", "Short / short"),
    ("input32k_short", "32K / short"),
    ("short_long", "Short / long"),
    ("mixed", "Mixed"),
)


def _numeric_value(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _overview_capacity_cell(
    endpoint: str,
    shape: str,
    capacity_rows: Sequence[Mapping[str, Any]],
    soak_rows: Sequence[Mapping[str, Any]],
) -> str:
    matched_soaks = [
        row
        for row in soak_rows
        if _endpoint_id(row) == endpoint
        and str(_first(row, "shape", "workload", "load_shape") or "") == shape
        and row.get("scientifically_complete") is True
    ]
    passed_rates = [
        value * 60.0
        for row in matched_soaks
        if row.get("soak_acceptance_pass") is True
        and (value := _numeric_value(row.get("candidate_rate_rps"))) is not None
    ]
    failed_rates = [
        value * 60.0
        for row in matched_soaks
        if row.get("soak_acceptance_pass") is False
        and (value := _numeric_value(row.get("candidate_rate_rps"))) is not None
    ]
    if passed_rates:
        minimum_pass = min(passed_rates)
        maximum_pass = max(passed_rates)
        text = (
            f"2-min healthy offered {minimum_pass:,.0f} RPM"
            if math.isclose(minimum_pass, maximum_pass)
            else (
                f"2-min healthy offered {minimum_pass:,.0f}-{maximum_pass:,.0f} "
                "RPM (runs)"
            )
        )
        higher_failures = [rate for rate in failed_rates if rate > max(passed_rates)]
        if higher_failures:
            text += f"; fail {min(higher_failures):,.0f}"
        return text
    if failed_rates:
        return f"2-min offered-rate fail {min(failed_rates):,.0f} RPM"

    matched_capacity = [
        row
        for row in capacity_rows
        if _endpoint_id(row) == endpoint
        and str(_first(row, "shape", "workload", "load_shape") or "") == shape
    ]
    confirmed = [
        row
        for row in matched_capacity
        if (_numeric_value(row.get("candidate_rate_confirmation_epoch_count")) or 0)
        >= 3
        and _numeric_value(
            row.get("confirmed_healthy_offered_rpm")
            or row.get("capacity_lower_bound_rpm")
        )
        is not None
    ]
    if confirmed:
        lower_values = [
            _numeric_value(
                row.get("confirmed_healthy_offered_rpm")
                or row.get("capacity_lower_bound_rpm")
            )
            or 0.0
            for row in confirmed
        ]
        lower = max(lower_values)
        if not math.isclose(min(lower_values), lower):
            return (
                f"Run-specific healthy {min(lower_values):,.0f}-{lower:,.0f} RPM; "
                "do not pool"
            )
        upper_values = [
            value
            for row in confirmed
            if (
                value := _numeric_value(
                    row.get("confirmed_healthy_offered_upper_rpm")
                    or row.get("capacity_upper_bound_rpm")
                )
            )
            is not None
            and value >= lower
        ]
        if upper_values:
            return f"Healthy offered {lower:,.0f}-{min(upper_values):,.0f} RPM"
        return f"Healthy offered >= {lower:,.0f} RPM"

    exploratory = [
        value
        for row in matched_capacity
        if (value := _numeric_value(row.get("highest_observed_healthy_rpm")))
        is not None
    ]
    if exploratory:
        return f"Exploratory healthy {max(exploratory):,.0f} RPM"
    return "No qualified rate"


def _overview_capacity_matrix(
    analysis: Mapping[str, Any],
    styles: Mapping[str, ParagraphStyle],
) -> Table | None:
    inventory = analysis.get("endpoint_inventory")
    if not isinstance(inventory, Sequence) or isinstance(
        inventory, (str, bytes, bytearray)
    ):
        return None
    capacity_rows = _as_rows(analysis.get("capacity_summaries"))
    soak_rows = _as_rows(analysis.get("soak_summaries"))
    endpoints = [
        endpoint
        for row in inventory
        if isinstance(row, Mapping) and (endpoint := _endpoint_id(row))
    ]
    if not endpoints:
        return None
    rows: list[list[Any]] = [["Endpoint", *[label for _, label in _OVERVIEW_SHAPES]]]
    for endpoint in endpoints:
        rows.append(
            [
                endpoint,
                *[
                    _overview_capacity_cell(endpoint, shape, capacity_rows, soak_rows)
                    for shape, _ in _OVERVIEW_SHAPES
                ],
            ]
        )
    return _table(
        rows,
        [45 * mm, 32.25 * mm, 32.25 * mm, 32.25 * mm, 32.25 * mm],
        styles,
    )


def _disposition_cell(
    endpoint: str,
    shape: str,
    capacity_rows: Sequence[Mapping[str, Any]],
    soak_rows: Sequence[Mapping[str, Any]],
) -> str:
    matched_soaks = [
        row
        for row in soak_rows
        if _endpoint_id(row) == endpoint and str(row.get("shape") or "") == shape
    ]
    passes = [
        row
        for row in matched_soaks
        if row.get("scientifically_complete") is True
        and row.get("soak_acceptance_pass") is True
    ]
    if passes:
        offered = max(
            (
                (_numeric_value(row.get("candidate_rate_rps")) or 0) * 60
                for row in passes
            ),
            default=0,
        )
        return f"2-min soak passed\n{offered:,.0f} offered RPM"
    measured_failures = [
        row
        for row in matched_soaks
        if row.get("scientifically_complete") is True
        and row.get("soak_acceptance_pass") is False
    ]
    if measured_failures:
        rate = min(
            (_numeric_value(row.get("candidate_rate_rps")) or 0) * 60
            for row in measured_failures
        )
        return f"Restricted\n2-min fail at {rate:,.0f} RPM"
    if any(
        row.get("status") == "baseline_transport_gate_failed" for row in matched_soaks
    ):
        return "Transport impaired\nbaseline gate"
    matched_capacity = [
        row
        for row in capacity_rows
        if _endpoint_id(row) == endpoint and str(row.get("shape") or "") == shape
    ]
    if any(
        int(float(row.get("candidate_rate_confirmation_epoch_count") or 0)) >= 3
        for row in matched_capacity
    ):
        return "Exploratory only\nshort AIMD"
    return "Insufficient evidence"


def _disposition_matrix(
    analysis: Mapping[str, Any], styles: Mapping[str, ParagraphStyle]
) -> Table | None:
    inventory = _as_rows(analysis.get("endpoint_inventory"))
    if not inventory:
        return None
    capacity_rows = _as_rows(analysis.get("capacity_summaries"))
    soak_rows = _as_rows(analysis.get("soak_summaries"))
    rows: list[list[Any]] = [["Endpoint", *[label for _, label in _OVERVIEW_SHAPES]]]
    for record in inventory:
        endpoint = _endpoint_id(record)
        if not endpoint:
            continue
        rows.append(
            [
                endpoint,
                *[
                    _disposition_cell(endpoint, shape, capacity_rows, soak_rows)
                    for shape, _ in _OVERVIEW_SHAPES
                ],
            ]
        )
    return _table(
        rows,
        [45 * mm, 32.25 * mm, 32.25 * mm, 32.25 * mm, 32.25 * mm],
        styles,
    )


def _executive_summary(
    styles: Mapping[str, ParagraphStyle],
    inputs: PublicReportInputs,
    endpoint_count: int,
) -> list[Flowable]:
    analysis = inputs.analysis
    coverage = analysis.get("coverage_summary")
    if not isinstance(coverage, Mapping):
        coverage = {}
    total_requests = _first(
        analysis,
        "request_count",
        "total_requests",
        "successful_and_failed_requests",
    )
    if total_requests is None:
        total_requests = len(inputs.csvs.get("normalized-requests.csv", ())) or None
    completed = _first(
        coverage,
        "completed_or_evidence_backed_unsupported_cells",
        "completed",
        "completed_cells",
        "complete",
    )
    planned = _first(
        coverage,
        "required_endpoint_dimension_cells",
        "planned",
        "planned_cells",
        "total_cells",
    )
    generated_at = _first(
        analysis, "generated_at", "campaign_completed_at", "analysis_time"
    )
    cost_summary = analysis.get("cost_summary")
    if not isinstance(cost_summary, Mapping):
        cost_summary = {}

    values = [
        ("Exact endpoints represented", endpoint_count),
        ("Request-level observations", total_requests),
        (
            "Request-attributed estimated cost",
            cost_summary.get("request_attributed_estimated_cost_usd"),
        ),
        (
            "Conservative campaign exposure",
            cost_summary.get("conservative_campaign_exposure_usd"),
        ),
        ("Campaign cost cap", cost_summary.get("cost_cap_usd")),
        (
            "Billing/credit HTTP 402 latch observed",
            cost_summary.get("billing_credit_http_402_latched"),
        ),
        ("Completed coverage cells", completed),
        ("Planned coverage cells", planned),
        ("Analysis generated", generated_at),
        ("Analysis schema", _first(analysis, "schema_version") or "Not reported"),
    ]
    result: list[Flowable] = []
    result.extend(_section(styles, "1", "Executive conclusion"))
    result.append(
        Paragraph(
            "This report maps the measured operating envelope of the DigitalOcean-hosted endpoints in the supplied public bundle. It separates what was observed from what was documented, distinguishes service reliability from task quality, and labels every missing or statistically weak region explicitly. The endpoint pages are the decision surface: each uses the same inventory, coverage, capacity, limits, quality, cost, and limitations structure.",
            styles["body"],
        )
    )
    result.append(
        _callout(
            styles,
            "What can this report support?",
            "It can guide workload-specific starting limits and identify where more measurement is required.",
            "An AIMD capacity claim requires three explicit healthy confirmation epochs at one exact matched-cell rate, and is reported as a bracket or a right-censored lower bound. A sustained claim additionally requires the independent soak blocks. The report does not convert sparse observations into an SLA.",
        )
    )
    result.append(Spacer(1, 5))
    result.append(_kv_table(values, styles))
    result.append(Paragraph("Decision rule", styles["h2"]))
    result.append(
        Paragraph(
            "Treat an AIMD number only as a confirmed healthy offered arrival rate for the exact endpoint, workload shape, streaming mode, token lengths, epoch duration, and concurrency regime shown. It is not completed goodput. Use the separately reported achieved-RPM interval and require a passing drain-inclusive soak before calling a rate sustained. This report does not manufacture operational headroom; choose it from your risk target, retain adaptive backoff, and rerun the matched cell when conditions change.",
            styles["body"],
        )
    )
    metric_audit = analysis.get("metric_audit_summary")
    if isinstance(metric_audit, Mapping):
        legacy_max = _first(metric_audit, "legacy_sse_proxy_max")
        corrected_max = _first(metric_audit, "corrected_post_ttft_proxy_max")
        result.append(Paragraph("Metric correction", styles["h2"]))
        result.append(
            _callout(
                styles,
                "The old six-digit token-rate outliers were a timing-definition error",
                "They divided completion tokens by the gap between streamed chunks; one chunk may contain many tokens.",
                (
                    f"The legacy maximum was {_format_number(legacy_max)} TPS. It is now audit-only. "
                    f"The corrected post-TTFT proxy retains its full qualified range (maximum {_format_number(corrected_max)} TPS), "
                    "while aggregate successful billed-token goodput is the headline throughput. Buffered TTFT, multi-choice per-sequence rates, and timing-unstable sub-100 ms intervals are explicitly censored. The 100 ms rule is a post-hoc measurement correction and does not remove requests from reliability, cost, token, quality, or aggregate-goodput accounting. No qualified extreme is silently trimmed."
                ),
                background=PALE_AMBER,
            )
        )
        result.append(CondPageBreak(75 * mm))
        result.append(Paragraph("Largest qualified timing observations", styles["h3"]))
        result.append(
            Paragraph(
                "These are the largest observations that survive the declared timing rules. They are shown row by row so a reader can audit the token count and denominator; none is silently clipped or winsorized.",
                styles["small"],
            )
        )
        outlier_table = _outlier_traceability_table(
            inputs.csvs.get("metric-audit.csv", ()), styles
        )
        result.append(
            outlier_table or _missing(styles, "Qualified timing-outlier traceability")
        )
    result.append(PageBreak())
    result.append(Paragraph("Measured capacity navigation", styles["h2"]))
    result.append(
        Paragraph(
            "This compact matrix is a navigation aid, not a provider ranking. "
            "A two-minute pass is shown before AIMD evidence; otherwise a rate is "
            "labelled as a healthy offered-rate bracket/lower bound or as exploratory. Exact token "
            "counts, concurrency, achieved-rate intervals, latency, TPM, failures, "
            "and recovery evidence remain in the matched endpoint profile. When "
            "qualified runs disagree, the observed range across runs is shown rather "
            "than silently selecting the fastest run.",
            styles["small"],
        )
    )
    overview_matrix = _overview_capacity_matrix(analysis, styles)
    result.append(
        overview_matrix or _missing(styles, "Measured capacity navigation matrix")
    )
    result.append(PageBreak())
    result.append(
        Paragraph("Engineering disposition by measured workload", styles["h2"])
    )
    result.append(
        Paragraph(
            "A two-minute pass means only that the exact soak passed at the shown offered load; it is not a production recommendation or SLA. Restricted means the soak produced a measured failure. Transport impaired means the low-load gate failed. Exploratory means only short AIMD confirmation exists. Apply only to the exact contract in the endpoint profile and retain retry/backoff.",
            styles["small"],
        )
    )
    result.append(
        _disposition_matrix(analysis, styles)
        or _missing(styles, "Engineering disposition matrix")
    )
    return result


def _methodology_section(
    styles: Mapping[str, ParagraphStyle], inputs: PublicReportInputs
) -> list[Flowable]:
    analysis = inputs.analysis
    result: list[Flowable] = []
    result.extend(_section(styles, "2", "Definitions and measurement method"))
    result.append(
        Paragraph(
            "Each term is explained twice: first as an engineering intuition, then as the precise calculation used to support comparisons.",
            styles["body"],
        )
    )
    for card in _definition_cards(styles, analysis):
        result.extend([card, Spacer(1, 4)])

    result.append(Paragraph("Statistical methodology", styles["h2"]))
    result.append(
        Paragraph(
            "Latency and request metrics treat independent requests as observations; saturation capacity treats independent load epochs as observations. Endpoint order and comparable task instances should be randomized or interleaved, while each saturation sweep remains isolated. Confidence intervals describe sampling uncertainty in the measured regime, not future provider guarantees. Service errors, timeouts, refusals, malformed responses, truncations, and retries remain in the denominator appropriate to the metric.",
            styles["body"],
        )
    )
    methodology = analysis.get("statistical_methodology")
    if isinstance(methodology, Mapping):
        methodology_labels = {
            "confidence_level": "Confidence level",
            "bootstrap_replicates": "Bootstrap resamples",
            "bootstrap_seed": "Reproducibility seed",
            "serial_sampling_unit": "Low-load sampling unit",
            "load_sampling_unit": "AIMD sampling unit",
            "soak_sampling_unit": "Soak sampling unit",
            "soak_quality_sampling_unit": "Paired-quality sampling unit",
            "soak_recovery_sampling_unit": "Recovery sampling unit",
            "success_interval_serial": "Success-rate interval",
            "continuous_metric_interval_serial": "Low-load continuous-metric interval",
            "load_intervals": "AIMD interval",
            "p99_minimum_observations": "Minimum observations before reporting p99",
            "output_tokens_are_not_independent_samples": "Output tokens treated as independent samples",
            "soak_capacity_policy": "Soak interpretation",
            "soak_block_dependence_note": "Soak-block dependence caveat",
        }
        methodology_order = (
            "confidence_level",
            "bootstrap_replicates",
            "bootstrap_seed",
            "serial_sampling_unit",
            "load_sampling_unit",
            "soak_sampling_unit",
            "soak_quality_sampling_unit",
            "soak_recovery_sampling_unit",
            "success_interval_serial",
            "continuous_metric_interval_serial",
            "load_intervals",
            "p99_minimum_observations",
            "output_tokens_are_not_independent_samples",
            "soak_capacity_policy",
            "soak_block_dependence_note",
        )
        for key in methodology_order:
            if key not in methodology:
                continue
            value = methodology[key]
            if key == "confidence_level":
                rendered = _format_percent(value)
            elif key == "output_tokens_are_not_independent_samples":
                rendered = "No" if bool(value) else "Yes"
            else:
                rendered = _plain_text(value)
            result.append(
                Paragraph(
                    f"- <b>{methodology_labels[key]}:</b> {rendered}",
                    styles["body"],
                )
            )
    else:
        for item in _extract_limitations(methodology)[:10]:
            result.append(Paragraph(f"- {_plain_text(item)}", styles["body"]))
    result.append(
        _callout(
            styles,
            "End-to-end prefill proxy",
            "TTFT shows the caller's total wait before generation begins.",
            "Unless the service exposes authenticated server timing, prompt-processing speed is inferred only as an end-to-end proxy from TTFT versus realized server input tokens. It must not be described as direct server-side prefill throughput.",
            background=PALE_AMBER,
        )
    )
    return result


def _cross_endpoint_section(
    styles: Mapping[str, ParagraphStyle],
    inputs: PublicReportInputs,
    endpoints: Sequence[Mapping[str, Any]],
) -> list[Flowable]:
    analysis = inputs.analysis
    result: list[Flowable] = []
    result.extend(_section(styles, "3", "Cross-endpoint inventory and coverage"))
    inventory_rows: list[list[Any]] = [
        ["Endpoint", "Version / region", "Context / output", "Capabilities"]
    ]
    for record in endpoints:
        endpoint = _endpoint_id(record) or "Not reported"
        if record.get("_placeholder"):
            inventory_rows.append(
                [endpoint, "Not reported", "Not reported", "Evidence missing"]
            )
            continue
        version_region = (
            " / ".join(
                str(value)
                for value in (
                    _first(record, "model_version", "version", "revision"),
                    _first(record, "region", "deployment_region", "server_region"),
                )
                if value not in (None, "")
            )
            or "Not reported"
        )
        limits = (
            " / ".join(
                str(value)
                for value in (
                    _first(
                        record,
                        "context_window",
                        "context_window_tokens",
                        "context_limit",
                        "max_context_tokens",
                    ),
                    _first(
                        record,
                        "max_output_tokens",
                        "output_limit_tokens",
                        "max_completion_tokens",
                    ),
                )
                if value not in (None, "")
            )
            or "Not reported"
        )
        caps: list[str] = []
        for label, keys, documented_name in (
            ("stream", ("streaming", "supports_streaming"), "streaming"),
            ("tools", ("tool_calling", "supports_tools", "tools"), "tool_calling"),
            (
                "JSON",
                ("structured_output", "supports_structured_output"),
                "structured_output",
            ),
            ("vision", ("vision", "supports_vision", "image_input"), "vision"),
            ("cache", (), "prompt_caching"),
            ("reasoning", (), "reasoning"),
        ):
            value = _first(record, *keys) if keys else None
            if value is None:
                value = _documented_capability(record, documented_name)
            if value is True or str(value).casefold() in {"true", "yes", "supported"}:
                caps.append(label)
        if _documented_capability(record, "vision") is False:
            caps.append("text only")
        inventory_rows.append(
            [endpoint, version_region, limits, ", ".join(caps) or "Not reported"]
        )
    result.append(_table(inventory_rows, [51 * mm, 42 * mm, 38 * mm, 43 * mm], styles))

    coverage = analysis.get("coverage_summary")
    if isinstance(coverage, Mapping):
        status_counts = coverage.get("status_counts")
        if not isinstance(status_counts, Mapping):
            status_counts = {}
        coverage_values = [
            (
                "Frozen endpoint x dimension cells",
                _first(
                    coverage,
                    "required_endpoint_dimension_cells",
                    "planned_cells",
                    "planned",
                    "total_cells",
                ),
            ),
            (
                "Completed or evidence-backed unsupported",
                _first(
                    coverage,
                    "completed_or_evidence_backed_unsupported_cells",
                    "completed_cells",
                    "completed",
                    "complete",
                ),
            ),
            ("Completed", status_counts.get("completed")),
            ("Unsupported", status_counts.get("unsupported")),
            ("Inconclusive", status_counts.get("inconclusive")),
            (
                "Untested / skipped",
                int(status_counts.get("untested") or 0)
                + int(status_counts.get("skipped") or 0),
            ),
        ]
        result.extend(
            [
                Paragraph("Coverage accounting", styles["h2"]),
                _kv_table(coverage_values, styles),
            ]
        )
    else:
        result.extend(
            [
                Paragraph("Coverage accounting", styles["h2"]),
                _missing(styles, "Coverage summary"),
            ]
        )

    cost_summary = analysis.get("cost_summary")
    if isinstance(cost_summary, Mapping):
        source_stages = _as_rows(cost_summary.get("source_stages"))
        result.append(Paragraph("Time and cost ledger", styles["h2"]))
        result.append(
            _kv_table(
                [
                    (
                        "Request-attributed estimated cost",
                        cost_summary.get("request_attributed_estimated_cost_usd"),
                    ),
                    (
                        "Conservative campaign exposure",
                        cost_summary.get("conservative_campaign_exposure_usd"),
                    ),
                    ("Campaign cost cap", cost_summary.get("cost_cap_usd")),
                    (
                        "Billing/credit HTTP 402 latch observed",
                        cost_summary.get("billing_credit_http_402_latched"),
                    ),
                    (
                        "Carried prior at first reconciled stage",
                        cost_summary.get("initial_carried_conservative_exposure_usd"),
                    ),
                    (
                        "Requests with attributed cost",
                        f"{_format_number(cost_summary.get('cost_attributed_request_count'))} / "
                        f"{_format_number((cost_summary.get('cost_attributed_request_count') or 0) + (cost_summary.get('cost_unattributed_request_count') or 0))}",
                    ),
                ],
                styles,
            )
        )
        if source_stages:
            stage_labels = {
                "direct_aimd": "AIMD + carried breadth reconciliation",
                "direct_soak": "Independent two-minute soak",
            }

            def stage_label(stage: Mapping[str, Any]) -> str:
                source_id = str(stage.get("source_id") or "").casefold()
                if "context" in source_id:
                    return "Context envelope"
                if "capability" in source_id:
                    return "Capability envelope"
                return stage_labels.get(
                    str(stage.get("source_kind") or ""),
                    _humanize_machine_value(stage.get("source_kind")),
                )

            def compact_time(value: Any) -> str:
                text = str(value or "")
                if not text:
                    return "Not reported"
                try:
                    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                except ValueError:
                    return "Not reported"
                if parsed.tzinfo is None:
                    return "Not reported"
                return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

            ledger_rows: list[list[Any]] = [
                ["Stage", "UTC window", "Prior", "Final", "Increment"]
            ]
            for stage in source_stages:
                started = compact_time(stage.get("started_at"))
                ended = compact_time(stage.get("ended_at"))
                ledger_rows.append(
                    [
                        stage_label(stage),
                        f"{started} to {ended}",
                        _format_money(stage.get("prior_conservative_exposure_usd")),
                        _format_money(
                            stage.get("cumulative_conservative_exposure_usd")
                        ),
                        _format_money(
                            stage.get("incremental_conservative_exposure_usd")
                        ),
                    ]
                )
            result.append(
                _table(
                    ledger_rows,
                    [38 * mm, 54 * mm, 27 * mm, 27 * mm, 28 * mm],
                    styles,
                )
            )
        result.append(
            Paragraph(
                _plain_text(
                    cost_summary.get("interpretation")
                    or "Request-attributed estimated cost and conservative campaign exposure use different accounting bases and are not interchangeable."
                ),
                styles["small"],
            )
        )
        if cost_summary.get("billing_credit_http_402_latched") is True:
            result.append(
                Paragraph(
                    "A billing/credit HTTP 402 latch was observed in the terminal "
                    "context-envelope stage. Later context cells were therefore "
                    "censored or left inconclusive; the latch is not evidence that "
                    "a model capability is unsupported.",
                    styles["small"],
                )
            )
    scope_exclusions = _as_rows(analysis.get("scope_exclusions"))
    result.append(Paragraph("Explicit measurement exclusions", styles["h2"]))
    if scope_exclusions:
        result.append(
            Paragraph(
                "These subtests were excluded by the capability campaign manifest. "
                "They remain untested in the coverage ledger and cannot be interpreted "
                "as supported, unsupported, completed, or implicitly covered by a nearby test.",
                styles["body"],
            )
        )
        exclusion_table = _scope_exclusion_table(
            scope_exclusions,
            styles,
            include_endpoint_count=True,
        )
        if exclusion_table is not None:
            result.append(exclusion_table)
    else:
        result.append(
            Paragraph(
                "No source manifest supplied an explicit measurement exclusion.",
                styles["body"],
            )
        )
    return result


def _charts_section(
    styles: Mapping[str, ParagraphStyle],
    inputs: PublicReportInputs,
    endpoint_slugs: set[str],
) -> list[Flowable]:
    result: list[Flowable] = []
    result.extend(
        _section(styles, "4", "Capacity, latency, quality, and cost comparisons")
    )
    result.append(
        Paragraph(
            "Every figure is a derived public view of the same endpoint and workload identities used in the tables. Categorical workload shapes are not connected as if they formed a continuous curve. Token axes should use realized server-reported token counts; requested anchors are labels, not substitutes for realized usage.",
            styles["body"],
        )
    )
    cross_charts = [
        path
        for path in inputs.charts
        if not any(slug and slug in _slug(path.stem) for slug in endpoint_slugs)
    ]
    if not cross_charts:
        result.append(_missing(styles, "Cross-endpoint charts"))
    else:
        for index, path in enumerate(cross_charts):
            if index:
                result.append(PageBreak())
            result.append(_chart_block(path, styles, max_height=205 * mm))
    return result


def _documented_capabilities(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("documented_capabilities")
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def _documented_capability(record: Mapping[str, Any], name: str) -> Any:
    capabilities = _documented_capabilities(record)
    if name == "vision":
        modalities = capabilities.get("modalities")
        if isinstance(modalities, Sequence) and not isinstance(
            modalities, (str, bytes, bytearray)
        ):
            return any(
                str(value).casefold() in {"image", "vision"} for value in modalities
            )
        return None
    aliases = {
        "tool_calling": ("tools", "tool_calling"),
        "structured_output": ("structured_output", "json_schema"),
        "streaming": ("streaming",),
        "prompt_caching": ("prompt_caching",),
        "reasoning": ("reasoning",),
    }
    return _first(capabilities, *aliases.get(name, (name,)))


def _record_or_documented_capability(
    record: Mapping[str, Any], name: str, *keys: str
) -> Any:
    explicit = _first(record, *keys)
    return explicit if explicit is not None else _documented_capability(record, name)


def _endpoint_inventory_values(
    record: Mapping[str, Any], summary: Mapping[str, Any]
) -> list[tuple[str, Any]]:
    return [
        ("Exact endpoint / model ID", _endpoint_id(record)),
        (
            "Display name (if exposed)",
            _first(record, "display_name", "name", "model_name"),
        ),
        (
            "Version / revision (if exposed)",
            _first(record, "model_version", "version", "revision"),
        ),
        (
            "Serving region (if exposed)",
            _first(record, "region", "deployment_region", "server_region")
            or _first(summary, "region", "server_region"),
        ),
        (
            "API version / surface",
            _first(record, "api_version", "api_surface", "endpoint_type"),
        ),
        (
            "Input price per million tokens",
            _first(
                record,
                "input_usd_per_million",
                "input_price_per_million_tokens",
                "input_price_mtok",
                "input_price",
            ),
        ),
        (
            "Output price per million tokens",
            _first(
                record,
                "output_usd_per_million",
                "output_price_per_million_tokens",
                "output_price_mtok",
                "output_price",
            ),
        ),
        (
            "Request-attributed estimated cost (this endpoint)",
            _path_value(
                summary,
                "cost.estimated_spend_usd",
                "cost.total_cost_usd",
                "estimated_cost_usd",
                "estimated_spend_usd",
                "cost_usd",
            ),
        ),
        (
            "Documented context tokens",
            _first(
                record,
                "context_window",
                "context_window_tokens",
                "context_limit",
                "max_context_tokens",
            ),
        ),
        (
            "Documented output tokens",
            _first(
                record,
                "max_output_tokens",
                "output_limit_tokens",
                "max_completion_tokens",
            ),
        ),
        (
            "Documented streaming",
            _record_or_documented_capability(
                record, "streaming", "streaming", "supports_streaming"
            ),
        ),
        (
            "Documented tool calling",
            _record_or_documented_capability(
                record, "tool_calling", "tool_calling", "supports_tools", "tools"
            ),
        ),
        (
            "Documented structured output",
            _record_or_documented_capability(
                record,
                "structured_output",
                "structured_output",
                "supports_structured_output",
            ),
        ),
        (
            "Documented vision input",
            _record_or_documented_capability(
                record, "vision", "vision", "supports_vision", "image_input"
            ),
        ),
    ]


def _quality_table(
    rows: Sequence[Mapping[str, Any]], styles: Mapping[str, ParagraphStyle]
) -> Table | None:
    matched = [
        row
        for row in rows
        if _first(row, "quality_pass_rate") not in (None, "")
        and int(float(_first(row, "quality_scored_count") or 0)) > 0
        and int(float(_first(row, "epoch_count") or 0)) == 0
    ]
    if not matched:
        return None
    preferred_workloads = (
        "short_short",
        "input32k_short",
        "short_long",
        "mixed",
        "long_context_retrieval",
        "reasoning",
        "coding_executable",
        "structured_json",
        "summarization",
        "tool_call_exact",
        "vision",
        "controlled_output",
    )
    priority = {name: index for index, name in enumerate(preferred_workloads)}
    matched = sorted(
        matched,
        key=lambda row: (
            priority.get(str(_first(row, "workload", "workload_id") or ""), 10_000),
            str(_first(row, "workload", "workload_id") or ""),
            str(row.get("source_id") or ""),
            str(row.get("phase") or ""),
            float(row.get("offered_rps") or -1),
            str(row.get("task_id") or ""),
        ),
    )
    table_rows: list[list[Any]] = [
        [
            "Exact task / workload",
            "Run / phase / offered load",
            "Scored requests",
            "Pass rate",
            "95% request CI",
        ]
    ]
    for row in matched[:12]:
        interval = _first(row, "quality_pass_rate_ci95")
        low = _first(interval, "ci95_low") if isinstance(interval, Mapping) else None
        high = _first(interval, "ci95_high") if isinstance(interval, Mapping) else None
        interval_text = (
            f"{_format_percent(low)} to {_format_percent(high)}"
            if low not in (None, "") and high not in (None, "")
            else "CI unavailable"
        )
        table_rows.append(
            [
                (
                    f"{_humanize_machine_value(_first(row, 'workload', 'workload_id'))}"
                    f"\n{row.get('task_id') or 'task id not recorded'}"
                ),
                (
                    f"{row.get('source_id') or 'source not recorded'}\n"
                    f"{_humanize_machine_value(row.get('phase') or 'low load / capability')}"
                    + (
                        f" @ {_format_number(row.get('offered_rps'))} RPS"
                        if row.get("offered_rps") is not None
                        else ""
                    )
                ),
                _first(row, "quality_scored_count") or 0,
                _format_percent(_first(row, "quality_pass_rate")),
                interval_text,
            ]
        )
    return _table(
        table_rows,
        [44 * mm, 52 * mm, 23 * mm, 23 * mm, 32 * mm],
        styles,
    )


def _paired_quality_table(
    rows: Sequence[Mapping[str, Any]], styles: Mapping[str, ParagraphStyle]
) -> Table | None:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pair_counts: Counter[tuple[str, str]] = Counter()
    pass_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        value = _numeric_value(row.get("paired_quality_delta_near_minus_low"))
        block_id = row.get("analysis_block_id")
        if value is None or block_id is None:
            continue
        key = (str(row.get("source_id")), str(row.get("shape")))
        grouped[key][str(block_id)].append(value)
        pair_counts[key] += 1
        pass_counts[key] += row.get("predeclared_quality_acceptance_pass") is True
    if not grouped:
        return None
    critical_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    table_rows: list[list[Any]] = [
        [
            "Workload / run",
            "Blocks / pairs",
            "Mean near-minus-low score",
            "95% t interval across block means",
            "Pair acceptance",
        ]
    ]
    for key, by_block in sorted(grouped.items()):
        source_id, shape = key
        block_means = [statistics.fmean(values) for values in by_block.values()]
        estimate = statistics.fmean(block_means)
        if len(block_means) >= 2:
            critical = critical_95.get(len(block_means), 1.96)
            half = (
                critical * statistics.stdev(block_means) / math.sqrt(len(block_means))
            )
            interval_text = (
                f"{_format_number(estimate - half)} to "
                f"{_format_number(estimate + half)}"
            )
        else:
            interval_text = "CI unavailable"
        table_rows.append(
            [
                f"{_humanize_machine_value(shape)}\n{source_id}",
                f"{len(block_means)} / {pair_counts[key]}",
                _format_number(estimate),
                interval_text,
                f"{pass_counts[key]} / {pair_counts[key]}",
            ]
        )
    return _table(
        table_rows,
        [42 * mm, 26 * mm, 34 * mm, 43 * mm, 29 * mm],
        styles,
    )


def _endpoint_section(
    styles: Mapping[str, ParagraphStyle],
    inputs: PublicReportInputs,
    record: Mapping[str, Any],
    index: int,
) -> list[Flowable]:
    analysis = inputs.analysis
    endpoint = _endpoint_id(record) or f"Unreported endpoint {index}"
    summary = _summary_for(endpoint, analysis)
    workload_rows = _coalesce_rows(
        endpoint,
        analysis.get("workload_summaries"),
        inputs.csvs.get("endpoint-workload-metrics.csv", ()),
    )
    capacity_rows = _coalesce_rows(
        endpoint,
        analysis.get("capacity_summaries"),
        inputs.csvs.get("capacity-summary.csv", ()),
    )
    if not capacity_rows:
        capacity_rows = workload_rows
    soak_rows = _coalesce_rows(
        endpoint,
        analysis.get("soak_summaries"),
        inputs.csvs.get("soak-cell-summary.csv", ()),
    )
    soak_quality_rows = _coalesce_rows(
        endpoint,
        analysis.get("soak_quality_summaries"),
        inputs.csvs.get("quality-pair-summary.csv", ()),
    )
    recovery_rows = _coalesce_rows(
        endpoint,
        analysis.get("soak_recovery_summaries"),
        inputs.csvs.get("recovery-summary.csv", ()),
    )
    coverage_rows = _coverage_rows_for_endpoint(inputs, endpoint)
    capability_rows = _matching(
        endpoint, inputs.csvs.get("capability-evidence.csv", ())
    )
    scope_exclusions = _scope_exclusions_for_endpoint(inputs, endpoint)
    limit_rows = _coalesce_rows(
        endpoint,
        analysis.get("observed_limits"),
        inputs.csvs.get("observed-limits.csv", ()),
    )
    limitations = _extract_limitations(
        _first(summary, "limitations", "caveats", "unresolved_questions"),
        endpoint=endpoint,
    )

    result: list[Flowable] = [PageBreak()]
    if index == 1:
        result.extend(
            _section(styles, "5", "Endpoint-by-endpoint engineering profiles")
        )
        result.append(
            Paragraph(
                "Each profile uses the same inventory, coverage, capacity, soak, limit, and quality structure so engineers can compare endpoints directly.",
                styles["body"],
            )
        )
    result.extend(_section(styles, f"5.{index}", endpoint))
    if record.get("_placeholder"):
        result.append(
            _missing(
                styles,
                "Endpoint profile",
                "absent from endpoint_inventory and all endpoint-keyed public summary tables",
            )
        )

    result.append(Paragraph("Inventory", styles["h2"]))
    result.append(_kv_table(_endpoint_inventory_values(record, summary), styles))

    result.append(Paragraph("Coverage", styles["h2"]))
    coverage_table = _coverage_table(coverage_rows, styles)
    result.append(coverage_table or _missing(styles, "Endpoint coverage ledger"))
    if scope_exclusions:
        result.append(
            Paragraph("Explicitly untested capability subtests", styles["h2"])
        )
        result.append(
            Paragraph(
                "The following manifest exclusions are untested for this endpoint. "
                "The broad endpoint-by-dimension matrix may be complete from separate "
                "evidence, but that never upgrades these named zero-attempt subtests.",
                styles["small"],
            )
        )
        exclusion_table = _scope_exclusion_table(
            scope_exclusions,
            styles,
            include_endpoint_count=False,
        )
        if exclusion_table is not None:
            result.append(exclusion_table)

    result.append(Paragraph("Capability outcome", styles["h2"]))
    result.append(
        Paragraph(
            "Transport support answers whether valid calls were served. Functional "
            "correctness scores the returned task result. Malformed validation asks "
            "whether deliberately invalid calls were rejected correctly. A generic "
            "4xx on a malformed probe is never used to claim that valid calls are "
            "unsupported.",
            styles["small"],
        )
    )
    result.append(
        _capability_evidence_table(capability_rows, styles)
        or _missing(styles, "Separated capability evidence")
    )

    result.append(Paragraph("Capacity and latency by workload", styles["h2"]))
    result.append(
        Paragraph(
            "A confirmed healthy offered rate requires three explicit healthy confirmation epochs at the same arrival rate. It describes the load sent, not the rate completed. The achieved-RPM and TPM intervals use drain-inclusive epoch outcomes. Short AIMD confirmations are never relabelled as sustained capacity; that requires the separate drain-inclusive soak evidence.",
            styles["small"],
        )
    )
    capacity_table = _capacity_table(capacity_rows, styles)
    result.append(capacity_table or _missing(styles, "Capacity evidence"))
    result.append(CondPageBreak(70 * mm))
    result.append(Paragraph("Exact confirmed-cell contract", styles["h3"]))
    result.append(
        _capacity_contract_table(capacity_rows, styles)
        or _missing(styles, "Confirmed-cell workload contract")
    )
    result.append(CondPageBreak(60 * mm))
    result.append(Paragraph("Latency and service outcome", styles["h3"]))
    result.append(
        _capacity_latency_table(capacity_rows, styles)
        or _missing(styles, "Matched latency and service outcome")
    )

    result.append(Paragraph("Independent two-minute soak", styles["h2"]))
    result.append(
        Paragraph(
            "Each displayed offered rate applies only to the exact endpoint, workload recipe, and observed two-minute arrival interval. Headline RPM/TPM divide each arrival cohort by its full time through the last completion, including drain. Four contiguous analysis blocks use analysis_block_id as the exploratory interval unit; serial correlation is not modelled. This is not a longer-duration or time-of-day SLA.",
            styles["small"],
        )
    )
    soak_table = _soak_table(soak_rows, styles)
    result.append(soak_table or _missing(styles, "Two-minute soak evidence"))
    result.append(CondPageBreak(55 * mm))
    result.append(Paragraph("Post-overload recovery", styles["h3"]))
    result.append(
        _recovery_table(recovery_rows, styles)
        or _missing(styles, "Post-overload recovery evidence")
    )

    result.append(Paragraph("Observed versus documented limits", styles["h2"]))
    limits_table = _limits_table(limit_rows, styles)
    result.append(limits_table or _missing(styles, "Limit mapping"))

    result.append(Paragraph("Low-load and capability-task quality", styles["h2"]))
    quality_table = _quality_table(workload_rows, styles)
    if quality_table is None:
        result.append(_missing(styles, "Quality evidence"))
    else:
        result.append(quality_table)
        result.append(
            Paragraph(
                "Every displayed percentage is one exact source, task, phase, offered rate, stream mode, and token-target stratum; AIMD phases and rates are never pooled. Service failure and task failure are separate: an unavailable or throttled response reduces service goodput, while an HTTP-successful but incorrect answer reduces quality-adjusted goodput.",
                styles["small"],
            )
        )
    result.append(CondPageBreak(60 * mm))
    result.append(Paragraph("Paired quality change under load", styles["h2"]))
    result.append(
        Paragraph(
            "Exact-payload low-load and near-load pairs are averaged within their predeclared analysis block. The displayed Student-t interval then uses the independent analysis_block_id means as its units; both block and pair counts are shown.",
            styles["small"],
        )
    )
    result.append(
        _paired_quality_table(soak_quality_rows, styles)
        or _missing(styles, "Paired quality under load")
    )

    endpoint_charts = [
        path for path in inputs.charts if _slug(endpoint) in _slug(path.stem)
    ]
    if endpoint_charts:
        result.append(Paragraph("Endpoint figures", styles["h2"]))
        for path in endpoint_charts[:6]:
            result.extend(
                [CondPageBreak(80 * mm), _chart_block(path, styles, max_height=92 * mm)]
            )

    if limitations:
        result.append(Paragraph("Endpoint-specific limitations", styles["h2"]))
        for item in limitations[:8]:
            result.append(Paragraph(f"- {_plain_text(item)}", styles["body"]))
    return result


def _final_sections(
    styles: Mapping[str, ParagraphStyle], inputs: PublicReportInputs
) -> list[Flowable]:
    analysis = inputs.analysis
    result: list[Flowable] = [PageBreak()]
    result.extend(_section(styles, "6", "Study-wide limitations"))
    limitations = _extract_limitations(analysis.get("limitations"))
    if limitations:
        for item in limitations:
            result.append(Paragraph(f"- {_plain_text(item)}", styles["body"]))
    else:
        result.append(
            Paragraph(
                "No study-wide limitation list was supplied. At minimum, confidence intervals apply only to the measured sampling units and conditions; sparse tails do not establish p99; time-of-day observations do not establish seasonality or an SLA; and untested cells remain unknown.",
                styles["body"],
            )
        )

    result.append(Paragraph("Public artifact map", styles["h2"]))
    file_rows: list[list[Any]] = [["Artifact", "Present", "Purpose"]]
    purposes = {
        "normalized-requests.csv": "Sanitized request-level metrics; no prompt or response text.",
        "normalized-epochs.csv": "Independent load-epoch results used for capacity intervals.",
        "endpoint-summary.csv": "One-row engineering summary per exact endpoint.",
        "endpoint-workload-metrics.csv": "Workload-specific latency, throughput, reliability, and quality.",
        "capacity-summary.csv": "AIMD confirmation points, brackets, and right-censoring; no automatic headroom.",
        "soak-cell-summary.csv": "Exact two-minute soak result per endpoint and workload; separate from AIMD.",
        "soak-block-summary.csv": "Four predeclared 30-second analysis_block_id units per complete soak cell.",
        "quality-pair-summary.csv": "Exact-payload low-load versus near-load quality-pair evidence.",
        "recovery-summary.csv": "Post-soak recovery phases and predeclared pass/failure reasons.",
        "coverage-ledger.csv": "Every planned cell and its terminal evidence state.",
        "coverage-matrix.csv": "Compact endpoint-by-dimension coverage view.",
        "scope-exclusions.csv": "Manifest-declared subtests that remain explicitly untested.",
        "observed-limits.csv": "Observed versus documented boundary results.",
        "metric-audit.csv": "Request-traceable legacy versus corrected timing metrics; no trimming.",
        "cache-state-metrics.csv": "TTFT and prefill proxies split by observed hit, miss, or unknown cache state.",
        "capability-evidence.csv": "Transport support, functional correctness, and malformed-input validation kept separate.",
    }
    for name in PUBLIC_CSV_FILES:
        file_rows.append(
            [name, "Yes" if inputs.csvs.get(name) else "No / empty", purposes[name]]
        )
    file_rows.append(
        [
            "charts/",
            "Yes" if inputs.charts else "No / empty",
            "Derived, publication-safe figures used in this PDF.",
        ]
    )
    result.append(_table(file_rows, [48 * mm, 25 * mm, 101 * mm], styles))
    result.append(
        _callout(
            styles,
            "Public-data boundary",
            "This PDF contains performance evidence, not test content or account details.",
            f"The builder reads only analysis.json (or its gzip-compressed equivalent), {len(PUBLIC_CSV_FILES)} named derived CSVs, and declared chart images. Final builds require the matching manifest and a passing public-bundle safety scan.",
            background=PALE_CYAN,
        )
    )
    return result


def build_story(
    artifact_dir: str | Path,
    *,
    title: str = "DigitalOcean Inference Endpoint Technical Benchmark",
    subtitle: str = "Public engineering report - heterogeneous workloads, operating envelopes, and uncertainty",
    mode: str = "draft",
) -> list[Flowable]:
    """Return the complete ReportLab story without producing a PDF."""

    inputs = load_public_inputs(artifact_dir, mode=mode)
    styles = _styles()
    endpoints = _endpoint_records(inputs)
    actual_endpoints = [row for row in endpoints if not row.get("_placeholder")]
    endpoint_slugs = {_slug(_endpoint_id(row) or "") for row in actual_endpoints}

    story: list[Flowable] = []
    cover = Table(
        [
            [
                [
                    Spacer(1, 23 * mm),
                    Paragraph("PUBLIC TECHNICAL BENCHMARK", styles["subtitle"]),
                    Spacer(1, 7 * mm),
                    Paragraph(_plain_text(title), styles["title"]),
                    Paragraph(_plain_text(subtitle), styles["subtitle"]),
                    (
                        Paragraph(
                            (
                                "DRAFT - COMPLETE COVERAGE - NOT FOR PUBLICATION"
                                if _path_value(
                                    inputs.analysis,
                                    "coverage_summary.is_100_percent",
                                )
                                is True
                                else "DRAFT - INCOMPLETE COVERAGE - NOT FOR PUBLICATION"
                            ),
                            styles["subtitle"],
                        )
                        if inputs.draft_watermark
                        else Spacer(1, 0)
                    ),
                    Spacer(1, 17 * mm),
                    Paragraph(
                        _plain_text(
                            f"{len(actual_endpoints)} reported endpoints | "
                            f"{EXPECTED_ENDPOINT_COUNT} endpoint profiles | "
                            "capacity, latency, limits, quality, cost, and coverage"
                        ),
                        styles["subtitle"],
                    ),
                    Spacer(1, 34 * mm),
                    Paragraph(
                        "Observed facts, calculated estimates, and interpretations are labelled separately. Missing evidence is shown explicitly.",
                        styles["subtitle"],
                    ),
                ]
            ]
        ],
        colWidths=[174 * mm],
        rowHeights=[242 * mm],
    )
    cover.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 14 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.extend([cover, PageBreak()])
    story.extend(_executive_summary(styles, inputs, len(actual_endpoints)))
    story.extend(_methodology_section(styles, inputs))
    story.extend(_cross_endpoint_section(styles, inputs, endpoints))
    story.extend(_charts_section(styles, inputs, endpoint_slugs))

    for index, record in enumerate(endpoints, start=1):
        story.extend(_endpoint_section(styles, inputs, record, index))
    story.extend(_final_sections(styles, inputs))
    return story


def build_pdf(
    artifact_dir: str | Path,
    output: str | Path,
    *,
    title: str = "DigitalOcean Inference Endpoint Technical Benchmark",
    subtitle: str = "Public engineering report - heterogeneous workloads, operating envelopes, and uncertainty",
    mode: str = "draft",
) -> Path:
    """Render the public report and return its resolved output path."""

    inputs = load_public_inputs(artifact_dir, mode=mode)
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title=title,
        author="DigitalOcean public inference endpoint benchmark",
        subject="Endpoint performance envelopes across heterogeneous workloads",
    )
    document.build(
        build_story(artifact_dir, title=title, subtitle=subtitle, mode=mode),
        canvasmaker=lambda *args, **kwargs: NumberedCanvas(
            *args, draft_watermark=inputs.draft_watermark, **kwargs
        ),
    )
    return output_path


__all__ = [
    "EXPECTED_ENDPOINT_COUNT",
    "PUBLIC_CSV_FILES",
    "PublicReportInputs",
    "build_pdf",
    "build_story",
    "load_public_inputs",
]
