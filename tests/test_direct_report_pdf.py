from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


pytest.importorskip("reportlab")

from do_benchmark.direct_report import (
    EXPECTED_ENDPOINT_IDS,
    REQUIRED_COVERAGE_DIMENSIONS,
    SCHEMA_VERSION,
    DirectReportError,
)
from do_benchmark.direct_report_pdf import (
    PUBLIC_CSV_FILES,
    _coverage_rows_for_endpoint,
    _disposition_cell,
    _endpoint_inventory_values,
    _overview_capacity_cell,
    build_story,
    load_public_inputs,
)


def _story_text(value: object) -> str:
    parts: list[str] = []

    def visit(item: object) -> None:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(str(text))
        cells = getattr(item, "_cellvalues", None)
        if cells is not None:
            visit(cells)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return " ".join(parts)


def test_overview_uses_narrow_evidence_labels_without_pooled_ceiling() -> None:
    capacity_rows = [
        {
            "endpoint_id": EXPECTED_ENDPOINT_IDS[0],
            "shape": "short_short",
            "candidate_rate_confirmation_epoch_count": 3,
            "confirmed_healthy_offered_rpm": 60,
        },
        {
            "endpoint_id": EXPECTED_ENDPOINT_IDS[0],
            "shape": "short_short",
            "candidate_rate_confirmation_epoch_count": 3,
            "confirmed_healthy_offered_rpm": 600,
        },
    ]
    assert (
        _overview_capacity_cell(
            EXPECTED_ENDPOINT_IDS[0], "short_short", capacity_rows, []
        )
        == "Run-specific healthy 60-600 RPM; do not pool"
    )

    soak_rows = [
        {
            "endpoint_id": EXPECTED_ENDPOINT_IDS[0],
            "shape": "short_short",
            "scientifically_complete": True,
            "soak_acceptance_pass": True,
            "candidate_rate_rps": 1.0,
        }
    ]
    assert (
        _disposition_cell(EXPECTED_ENDPOINT_IDS[0], "short_short", [], soak_rows)
        == "2-min soak passed\n60 offered RPM"
    )


def test_pdf_story_uses_only_public_bundle_and_has_twelve_profiles(
    tmp_path: Path,
) -> None:
    coverage_matrix = [
        {
            "endpoint_id": endpoint,
            "coverage_dimension": dimension,
            "status": "completed" if index < 96 else "untested",
            "planned_cell_or_epoch_count": 1 if index < 96 else 0,
            "observed_attempt_count": 1 if index < 96 else 0,
            "explicit_untested_subtest_count": 0,
            "has_explicit_scope_exclusions": False,
        }
        for index, (endpoint, dimension) in enumerate(
            (endpoint, dimension)
            for endpoint in EXPECTED_ENDPOINT_IDS
            for dimension in REQUIRED_COVERAGE_DIMENSIONS
        )
    ]
    scope_exclusion = {
        "schema_version": "digitalocean_direct_scope_exclusion_v1",
        "source_kind": "direct_breadth",
        "source_id": "capability",
        "source_manifest_sha256": "a" * 64,
        "endpoint_id": EXPECTED_ENDPOINT_IDS[0],
        "scope_exclusion_id": "adaptive_tool_over_limit_followups",
        "measurement_label": "Adaptive tool +1 boundary discovery",
        "coverage_dimension": "tool_calling",
        "status": "untested",
        "reason": "not part of the fixed capability plan",
        "claim_policy": "explicitly_excluded_not_tested",
    }
    excluded_matrix_row = next(
        row
        for row in coverage_matrix
        if row["endpoint_id"] == scope_exclusion["endpoint_id"]
        and row["coverage_dimension"] == scope_exclusion["coverage_dimension"]
    )
    assert excluded_matrix_row["status"] == "completed"
    excluded_matrix_row["explicit_untested_subtest_count"] = 1
    excluded_matrix_row["has_explicit_scope_exclusions"] = True
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "endpoint_inventory": [
            {
                "endpoint_id": endpoint,
                "context_window": 128_000,
                "max_output_tokens": 32_000,
                "input_usd_per_million": "0.10",
                "output_usd_per_million": "0.50",
                "server_region": "not exposed",
                "api_version": "v1",
                "api_surface": "chat_completions",
                "documented_capabilities": json.dumps(
                    {
                        "modalities": ["text"],
                        "tools": True,
                        "structured_output": True,
                    }
                ),
            }
            for endpoint in EXPECTED_ENDPOINT_IDS
        ],
        "endpoint_summaries": [
            {
                "endpoint_id": endpoint,
                "request_count": 20,
                "epoch_count": 0,
                "estimated_cost_usd": 0.2,
                "heterogeneous_metrics_omitted": True,
                "aggregate_policy": "counts_and_cost_only",
            }
            for endpoint in EXPECTED_ENDPOINT_IDS
        ],
        "cost_summary": {
            "schema_version": "digitalocean_public_cost_summary_v1",
            "request_attributed_estimated_cost_usd": 2.4,
            "cost_attributed_request_count": 240,
            "cost_unattributed_request_count": 0,
            "request_cost_attribution_complete": True,
            "conservative_campaign_exposure_usd": 3.1,
            "conservative_exposure_source_id": "aimd",
            "conservative_exposure_receipt_schema_version": "do_direct_summary_v1",
            "conservative_exposure_receipt_sha256": "a" * 64,
            "cost_cap_usd": 200.0,
            "cost_cap_revision_count": 0,
            "cost_cap_history_usd": [200.0],
            "initial_carried_conservative_exposure_usd": 0.7,
            "estimand_relationship": "overlapping_non_additive",
            "billing_credit_http_402_latched": False,
            "http_402_latched_source_ids": [],
            "source_stages": [
                {
                    "source_kind": "direct_aimd",
                    "source_id": "aimd",
                    "started_at": "2026-08-23T14:00:00+01:00",
                    "ended_at": "2026-08-23T17:00:00+01:00",
                    "prior_conservative_exposure_usd": 0.7,
                    "cumulative_conservative_exposure_usd": 3.1,
                    "incremental_conservative_exposure_usd": 2.4,
                    "cost_cap_usd": 200.0,
                    "cost_basis": "portable_reconciliation",
                    "reconciliation_schema_version": "do_direct_aimd_reconciliation_v1",
                    "reconciliation_sha256": "b" * 64,
                    "summary_schema_version": "do_direct_summary_v1",
                    "summary_sha256": "a" * 64,
                    "terminal_status": "complete_right_censored",
                    "http_402_latched": False,
                }
            ],
            "interpretation": "The accounting bases overlap and are not additive.",
        },
        "capacity_summaries": [],
        "workload_summaries": [],
        "observed_limits": [],
        "coverage_summary": {
            "required_endpoint_count": 12,
            "required_dimension_count": 16,
            "required_endpoint_dimension_cells": 192,
            "completed_or_evidence_backed_unsupported_cells": 96,
            "coverage_fraction": 96 / 192,
            "is_100_percent": False,
            "status_counts": {
                "completed": 96,
                "untested": 96,
            },
        },
        "coverage_matrix": coverage_matrix,
        "scope_exclusions": [scope_exclusion],
        "request_reconciliation": {"all_requests_reconciled": True},
        "statistical_methodology": {
            "confidence_level": 0.95,
            "serial_sampling_unit": "request_id",
            "load_sampling_unit": "epoch_id",
            "bootstrap_replicates": 100,
        },
        "limitations": ["Unmeasured cells remain inconclusive."],
        "output_files": {"charts": []},
    }
    (tmp_path / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    for name in PUBLIC_CSV_FILES:
        with (tmp_path / name).open("w", encoding="utf-8", newline="") as handle:
            if name == "endpoint-summary.csv":
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "endpoint_id",
                        "request_count",
                        "estimated_cost_usd",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    {
                        "endpoint_id": row["endpoint_id"],
                        "request_count": row["request_count"],
                        "estimated_cost_usd": row["estimated_cost_usd"],
                    }
                    for row in analysis["endpoint_summaries"]
                )
                continue
            if name == "coverage-matrix.csv":
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "endpoint_id",
                        "coverage_dimension",
                        "status",
                        "planned_cell_or_epoch_count",
                        "observed_attempt_count",
                        "explicit_untested_subtest_count",
                        "has_explicit_scope_exclusions",
                    ],
                )
                writer.writeheader()
                writer.writerows(coverage_matrix)
                continue
            if name == "scope-exclusions.csv":
                writer = csv.DictWriter(handle, fieldnames=sorted(scope_exclusion))
                writer.writeheader()
                writer.writerow(scope_exclusion)
                continue
            if name == "coverage-ledger.csv":
                ledger_row = {
                    "source_kind": "direct_breadth",
                    "source_id": "capability",
                    "endpoint_id": EXPECTED_ENDPOINT_IDS[0],
                    "coverage_dimension": "tool_calling",
                    "workload": "adaptive_tool_over_limit_followups",
                    "cell_or_epoch_id": (
                        "scope-exclusion:adaptive_tool_over_limit_followups"
                    ),
                    "planned_attempt_count": 0,
                    "observed_attempt_count": 0,
                    "conclusive_attempt_count": 0,
                    "status": "untested",
                    "evidence_scope": "manifest_scope_exclusion",
                    "scope_exclusion_id": "adaptive_tool_over_limit_followups",
                    "claim_policy": "explicitly_excluded_not_tested",
                    "measurement_label": "Adaptive tool +1 boundary discovery",
                    "exclusion_reason": "not part of the fixed capability plan",
                    "scope_exclusion_schema_version": (
                        "digitalocean_direct_scope_exclusion_v1"
                    ),
                    "source_manifest_sha256": "a" * 64,
                }
                writer = csv.DictWriter(handle, fieldnames=sorted(ledger_row))
                writer.writeheader()
                writer.writerow(ledger_row)
                continue
            writer = csv.DictWriter(handle, fieldnames=["endpoint_id", "status"])
            writer.writeheader()
    (tmp_path / "README.md").write_text("stale v1 report", encoding="utf-8")
    (tmp_path / "stale-v1.pdf").write_bytes(b"stale")
    (tmp_path / "charts").mkdir()
    (tmp_path / "charts" / "stale-v1.png").write_bytes(b"not an image")
    inputs = load_public_inputs(tmp_path)
    assert len(inputs.analysis["endpoint_inventory"]) == 12
    assert inputs.charts == ()
    story = build_story(tmp_path)
    text = _story_text(story)
    for endpoint in EXPECTED_ENDPOINT_IDS:
        assert endpoint in text
    assert "worker" not in text.casefold()
    assert "checkpoint" not in text.casefold()
    assert "draft - incomplete coverage" in text.casefold()
    assert "confidence level" in text.casefold()
    assert "bootstrap resamples" in text.casefold()
    assert "low-load sampling unit" in text.casefold()
    assert "aimd sampling unit" in text.casefold()
    assert "request-attributed estimated cost" in text.casefold()
    assert "conservative campaign exposure" in text.casefold()
    assert "$3.10" in text
    assert "time and cost ledger" in text.casefold()
    assert "not additive" in text.casefold()
    assert "2026-08-23 13:00 UTC" in text
    assert "adaptive tool +1 boundary discovery" in text.casefold()
    assert "explicitly untested capability subtests" in text.casefold()
    assert {path.name for path in tmp_path.glob("*.pdf")} == {"stale-v1.pdf"}
    inventory_values = dict(
        _endpoint_inventory_values(
            analysis["endpoint_inventory"][0], analysis["endpoint_summaries"][0]
        )
    )
    assert inventory_values["Documented context tokens"] == 128_000
    assert inventory_values["Input price per million tokens"] == "0.10"
    assert inventory_values["Output price per million tokens"] == "0.50"
    assert inventory_values["Serving region (if exposed)"] == "not exposed"
    assert inventory_values["Documented tool calling"] is True
    assert inventory_values["Documented structured output"] is True
    assert inventory_values["Documented vision input"] is False
    assert inventory_values["Request-attributed estimated cost (this endpoint)"] == 0.2
    matrix_rows = _coverage_rows_for_endpoint(inputs, EXPECTED_ENDPOINT_IDS[0])
    assert len(matrix_rows) == len(REQUIRED_COVERAGE_DIMENSIONS)

    endpoint_csv_path = tmp_path / "endpoint-summary.csv"
    original_endpoint_csv = endpoint_csv_path.read_text(encoding="utf-8")
    endpoint_csv_path.write_text(
        original_endpoint_csv.replace(",0.2\n", ",999.0\n", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cost accounting disagrees"):
        load_public_inputs(tmp_path)
    endpoint_csv_path.write_text(original_endpoint_csv, encoding="utf-8")

    with pytest.raises(DirectReportError, match="100% completed"):
        load_public_inputs(tmp_path, mode="final")
    for row in coverage_matrix:
        row["status"] = "completed"
        row["explicit_untested_subtest_count"] = 0
        row["has_explicit_scope_exclusions"] = False
    analysis["scope_exclusions"] = []
    analysis["coverage_summary"].update(
        {
            "completed_or_evidence_backed_unsupported_cells": 192,
            "coverage_fraction": 1.0,
            "is_100_percent": True,
            "status_counts": {"completed": 192},
        }
    )
    analysis["coverage_matrix"] = coverage_matrix
    (tmp_path / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    (tmp_path / "scope-exclusions.csv").write_text("endpoint_id\n", encoding="utf-8")
    (tmp_path / "coverage-ledger.csv").write_text("endpoint_id\n", encoding="utf-8")
    with (tmp_path / "coverage-matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "endpoint_id",
                "coverage_dimension",
                "status",
                "planned_cell_or_epoch_count",
                "observed_attempt_count",
                "explicit_untested_subtest_count",
                "has_explicit_scope_exclusions",
            ],
        )
        writer.writeheader()
        writer.writerows(coverage_matrix)
    complete_draft = load_public_inputs(tmp_path, mode="draft")
    assert complete_draft.draft_watermark is True
    complete_story = build_story(tmp_path, mode="draft")
    complete_text = _story_text(complete_story)
    assert "draft - complete coverage - not for publication" in complete_text.casefold()
    with pytest.raises(ValueError, match="public-safety-scan"):
        load_public_inputs(tmp_path, mode="final")
