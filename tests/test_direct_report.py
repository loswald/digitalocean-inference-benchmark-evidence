from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from do_benchmark.core import MODEL_BY_ID

from do_benchmark.direct_report import (
    DEEPSEEK_ENDPOINT_ID,
    EXPECTED_ENDPOINT_IDS,
    KIMI_UNDOCUMENTED_CONTEXT_PROBE_ANCHOR,
    REQUIRED_COVERAGE_DIMENSIONS,
    DirectReportError,
    _breadth_cost_summary_required,
    _build_cost_summary,
    _epoch_units_from_requests,
    _source_cost_ledger_fields,
    analyze_and_write,
    build_capability_evidence,
    build_metric_audit,
    build_coverage,
    build_observed_limits,
    build_capacity_summary,
    load_breadth_directory,
    load_breadth_scope_exclusions,
    normalize_epoch,
    normalize_request,
    reconcile_request_rows,
    scan_public_bundle_safety,
    summarize_group,
    validate_public_analysis_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_public_safety_scanner_reads_only_canonical_compressed_analysis(
    tmp_path: Path,
) -> None:
    with gzip.open(tmp_path / "analysis.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"safe": "derived metric"}, handle)
    assert scan_public_bundle_safety(tmp_path)["passed"] is True

    with gzip.open(tmp_path / "other.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"safe": "derived metric"}, handle)
    scan = scan_public_bundle_safety(tmp_path)
    assert scan["passed"] is False
    assert {finding["rule"] for finding in scan["findings"]} == {
        "unapproved_compressed_file"
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _breadth_record(
    *,
    cell_id: str,
    family: str,
    input_tokens: int = 128,
    output_tokens: int = 32,
) -> dict:
    return {
        "schema_version": "digitalocean_inference_benchmark_record_v1",
        "cell_id": cell_id,
        "model_id": DEEPSEEK_ENDPOINT_ID,
        "task_id": cell_id,
        "task_family": family,
        "context_bucket": str(input_tokens),
        "output_bucket": "short",
        "started_at": "2026-08-23T12:00:00+00:00",
        "ended_at": "2026-08-23T12:00:02+00:00",
        "status": "success",
        "http_status": 200,
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "timing": {
            "request_seconds": 2.0,
            "ttft_seconds": 0.5,
            "generation_seconds": 1.0,
            "output_tokens_per_second": float(output_tokens),
        },
        "quality_score": 1.0,
        "score_kind": "exact_text",
        "estimated_cost_usd": 0.0001,
        "task_metadata": {"planned_input_tokens": input_tokens},
        "response": {"text": "TOP SECRET MODEL OUTPUT"},
        "response_headers": {"authorization": "Bearer secret"},
        "error": "must not be copied",
    }


def test_normalization_rejects_sse_chunk_span_as_decode_rate() -> None:
    raw = {
        **_breadth_record(
            cell_id="legacy-six-digit-rate", family="short_short", output_tokens=3
        ),
        "timing": {
            "request_seconds": 0.693,
            "ttft_seconds": 0.664,
            "generation_seconds": 0.00001026,
            "output_tokens_per_second": 292_397.6608187134,
        },
        "stream": {"event_count": 1, "first_event_kind": "content"},
        "usage": {
            "prompt_tokens": 32,
            "completion_tokens": 3,
            "cache_read_input_tokens": 0,
        },
    }
    normalized = normalize_request(
        raw, source_kind="direct_aimd", source_id="legacy-audit"
    )
    assert normalized[
        "legacy_sse_chunk_span_output_tokens_per_second_proxy"
    ] == pytest.approx(292_397.6608187134)
    assert normalized["legacy_sse_span_headline_eligible"] is False
    assert normalized["post_ttft_output_tokens_per_second_proxy"] is None
    assert normalized["output_tokens_per_second"] is None
    assert (
        "post_ttft_interval_below_100ms_unstable_rate"
        in normalized["timing_metric_invalidity_reasons"]
    )


def test_buffered_ttft_and_multi_choice_per_sequence_metrics_are_censored() -> None:
    buffered = {
        **_breadth_record(cell_id="buffered", family="parameter_validation"),
        "bindings": {"stream": False, "n": 1},
        "timing": {
            "request_seconds": 1.25,
            "ttft_seconds": 1.25,
            "generation_seconds": 0.0,
        },
        "stream_observation": {
            "event_count": 1,
            "first_event_kind": "buffered_response",
        },
    }
    normalized_buffered = normalize_request(
        buffered, source_kind="direct_breadth", source_id="capability"
    )
    assert normalized_buffered["request_seconds"] == 1.25
    assert normalized_buffered["ttft_seconds"] is None
    assert normalized_buffered["output_tokens_per_second"] is None
    assert (
        "not_streamed_ttft_unobservable"
        in normalized_buffered["timing_metric_invalidity_reasons"]
    )

    multi = {
        **_breadth_record(cell_id="multi", family="parameter_validation"),
        "bindings": {"stream": True, "n": 3},
        "stream_observation": {"event_count": 20, "first_event_kind": "content"},
    }
    normalized_multi = normalize_request(
        multi, source_kind="direct_breadth", source_id="capability"
    )
    assert normalized_multi["choice_count"] == 3
    assert normalized_multi["multi_choice"] is True
    assert normalized_multi["output_tokens_per_second"] is None
    assert (
        "multiple_choices_aggregate_usage_not_per_sequence"
        in normalized_multi["timing_metric_invalidity_reasons"]
    )


def test_cache_state_is_explicit_and_unknown_never_becomes_a_miss() -> None:
    base = {
        **_breadth_record(cell_id="cache", family="long_context_retrieval"),
        "stream": {"event_count": 20, "first_event_kind": "content"},
    }
    miss = normalize_request(
        {
            **base,
            "usage": {
                "prompt_tokens": 1_000,
                "completion_tokens": 32,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        },
        source_kind="direct_aimd",
        source_id="miss",
    )
    hit = normalize_request(
        {
            **base,
            "usage": {
                "prompt_tokens": 1_000,
                "completion_tokens": 32,
                "prompt_tokens_details": {"cached_tokens": 900},
            },
        },
        source_kind="direct_aimd",
        source_id="hit",
    )
    unknown = normalize_request(base, source_kind="direct_aimd", source_id="unknown")
    assert miss["cache_state"] == "cache_miss_observed"
    assert miss["prefill_proxy_tokens_per_second"] == 2_000
    assert hit["cache_state"] == "cache_hit_observed"
    assert hit["prefill_proxy_tokens_per_second"] is None
    assert unknown["cache_state"] == "not_reported_unknown"
    assert unknown["prefill_proxy_tokens_per_second"] is None


def test_metric_audit_retains_extremes_and_counts_invalid_legacy_tail() -> None:
    requests = [
        {
            "request_id": "max-legacy",
            "endpoint_id": DEEPSEEK_ENDPOINT_ID,
            "legacy_sse_chunk_span_output_tokens_per_second_proxy": 292_454.0,
            "post_ttft_output_tokens_per_second_proxy": 30.8,
            "stream_mode": "streamed",
            "choice_count": 1,
            "multi_choice": False,
            "stream_event_count": 1,
            "output_tokens": 3,
            "ttft_seconds": 0.2,
            "timing_metric_audit_classification": "valid_ordinary",
        },
        {
            "request_id": "corrected-extreme",
            "endpoint_id": DEEPSEEK_ENDPOINT_ID,
            "legacy_sse_chunk_span_output_tokens_per_second_proxy": 10_000.0,
            "post_ttft_output_tokens_per_second_proxy": 3_415.78,
            "stream_mode": "streamed",
            "choice_count": 1,
            "multi_choice": False,
            "stream_event_count": 65,
            "output_tokens": 10_020,
            "ttft_seconds": 1.0,
            "timing_metric_audit_classification": "valid_extreme_keep_and_flag",
        },
        {
            "request_id": "buffered-multi",
            "endpoint_id": "qwen3.8-max",
            "legacy_sse_chunk_span_output_tokens_per_second_proxy": 1_000.0,
            "post_ttft_output_tokens_per_second_proxy": None,
            "stream_mode": "buffered_nonstream",
            "choice_count": 3,
            "multi_choice": True,
            "stream_event_count": 1,
            "output_tokens": 12,
            "ttft_seconds": None,
            "timing_metric_audit_classification": "invalid_or_censored",
        },
    ]
    rows, summary = build_metric_audit(requests)
    assert len(rows) == 3
    assert summary["legacy_sse_proxy_at_least_1000"] == 3
    assert summary["legacy_sse_proxy_at_least_10000"] == 2
    assert summary["legacy_sse_proxy_at_least_100000"] == 1
    assert summary["legacy_sse_proxy_max"] == 292_454.0
    assert summary["corrected_post_ttft_proxy_median"] == 30.8
    assert summary["corrected_post_ttft_proxy_max"] == 3_415.78
    assert summary["buffered_nonstream_ttft_censored"] == 1
    assert summary["multi_choice_per_sequence_excluded"] == 1
    assert all(row["trimmed_or_winsorized"] is False for row in rows)


def test_capability_support_correctness_and_malformed_validation_are_separate() -> None:
    evidence = build_capability_evidence(
        [
            {
                "endpoint_id": "qwen3.8-max",
                "workload": "vision",
                "capability_dimension": "vision",
                "malformed_validation_probe": False,
                "provider_send_attempted": True,
                "transport_success": True,
                "quality_scored": True,
                "functional_valid": False,
            },
            {
                "endpoint_id": "qwen3.8-max",
                "workload": "vision",
                "capability_dimension": "vision",
                "malformed_validation_probe": True,
                "provider_send_attempted": True,
                "transport_success": False,
                "http_status": 400,
                "quality_scored": False,
            },
        ]
    )[0]
    assert evidence["transport_status"] == "observed_supported"
    assert evidence["functional_status"] == "failed"
    assert evidence["malformed_validation_status"] == "correct_rejection_observed"


def test_cost_ledger_uses_portable_reconciliation_and_separates_estimands(
    tmp_path: Path,
) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "do_direct_summary_v1",
                "status": "complete_right_censored",
                "all_models_complete": True,
                "http_402_latched": False,
                "started_at": "2026-08-23T14:33:14+00:00",
                "ended_at": "2026-08-23T17:31:46+00:00",
                "prior_cost_usd": 21.0,
                "conservative_exposure_usd": 38.0,
                "max_cost_usd": 200.0,
            }
        ),
        encoding="utf-8",
    )
    reconciliation_body = {
        "schema_version": "do_direct_aimd_reconciliation_v1",
        "settlement": {
            "reconciled_prior_exposure_usd": 53.0,
            "reconciled_cumulative_exposure_usd": 72.0,
        },
    }
    reconciliation = {
        **reconciliation_body,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                reconciliation_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
    }
    (tmp_path / "reconciliation-portable.json").write_text(
        json.dumps(reconciliation), encoding="utf-8"
    )
    fields = _source_cost_ledger_fields(
        tmp_path,
        expected_source_kind="direct_aimd",
        required=True,
        prefer_portable_reconciliation=True,
    )
    assert fields["prior_conservative_exposure_usd"] == 53.0
    assert fields["cumulative_conservative_exposure_usd"] == 72.0
    assert fields["cost_basis"] == "portable_reconciliation"

    cost = _build_cost_summary(
        [{"source_kind": "direct_aimd", "source_id": "aimd", **fields}],
        [
            {"estimated_cost_usd": 3.25, "cost_attributed": True},
            {"estimated_cost_usd": 4.75, "cost_attributed": True},
        ],
        [
            {"request_count": 1, "estimated_cost_usd": 3.25},
            {"request_count": 1, "estimated_cost_usd": 4.75},
        ],
    )
    assert cost["request_attributed_estimated_cost_usd"] == 8.0
    assert cost["conservative_campaign_exposure_usd"] == 72.0
    assert cost["cost_cap_usd"] == 200.0
    assert cost["initial_carried_conservative_exposure_usd"] == 53.0
    assert cost["source_stages"][0]["incremental_conservative_exposure_usd"] == 19.0

    revised_cap_stage = {
        **fields,
        "started_at": "2026-08-23T18:00:00+00:00",
        "ended_at": "2026-08-23T19:00:00+00:00",
        "prior_conservative_exposure_usd": 72.0,
        "cumulative_conservative_exposure_usd": 73.0,
        "cost_cap_usd": 400.0,
    }
    revised = _build_cost_summary(
        [
            {"source_kind": "direct_aimd", "source_id": "aimd", **fields},
            {
                "source_kind": "direct_aimd",
                "source_id": "fresh-aimd",
                **revised_cap_stage,
            },
        ],
        [{"estimated_cost_usd": 1.0, "cost_attributed": True}],
        [{"request_count": 1, "estimated_cost_usd": 1.0}],
    )
    assert revised["cost_cap_usd"] == 400.0
    assert revised["cost_cap_revision_count"] == 1
    assert revised["cost_cap_history_usd"] == [200.0, 400.0]

    lower_cap_stage = {**revised_cap_stage, "cost_cap_usd": 100.0}
    with pytest.raises(DirectReportError, match="cap decreased"):
        _build_cost_summary(
            [
                {"source_kind": "direct_aimd", "source_id": "aimd", **fields},
                {
                    "source_kind": "direct_aimd",
                    "source_id": "bad-cap",
                    **lower_cap_stage,
                },
            ],
            [{"estimated_cost_usd": 1.0, "cost_attributed": True}],
            [{"request_count": 1, "estimated_cost_usd": 1.0}],
        )

    partial = _build_cost_summary(
        [{"source_kind": "direct_aimd", "source_id": "aimd", **fields}],
        [
            {"estimated_cost_usd": 1.0, "cost_attributed": True},
            {"estimated_cost_usd": None, "cost_attributed": False},
        ],
        [{"request_count": 2, "estimated_cost_usd": 1.0}],
    )
    assert partial["cost_attributed_request_count"] == 1
    assert partial["cost_unattributed_request_count"] == 1
    assert partial["request_cost_attribution_complete"] is False

    broken_stage = {
        **fields,
        "started_at": "2026-08-23T18:00:00+00:00",
        "ended_at": "2026-08-23T19:00:00+00:00",
        "prior_conservative_exposure_usd": 71.0,
        "cumulative_conservative_exposure_usd": 73.0,
    }
    with pytest.raises(DirectReportError, match="not exposure-contiguous"):
        _build_cost_summary(
            [
                {"source_kind": "direct_aimd", "source_id": "aimd", **fields},
                {"source_kind": "direct_soak", "source_id": "soak", **broken_stage},
            ],
            [{"estimated_cost_usd": 1.0, "cost_attributed": True}],
            [{"request_count": 1, "estimated_cost_usd": 1.0}],
        )

    reconciliation["receipt_sha256"] = "0" * 64
    (tmp_path / "reconciliation-portable.json").write_text(
        json.dumps(reconciliation), encoding="utf-8"
    )
    with pytest.raises(DirectReportError, match="internal hash mismatch"):
        _source_cost_ledger_fields(
            tmp_path,
            expected_source_kind="direct_aimd",
            required=True,
            prefer_portable_reconciliation=True,
        )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    summary["schema_version"] = "unknown_summary"
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(DirectReportError, match="unrecognized terminal summary schema"):
        _source_cost_ledger_fields(
            tmp_path,
            expected_source_kind="direct_aimd",
            required=True,
        )


def test_current_frozen_aimd_contract_does_not_require_legacy_reconciliation(
    tmp_path: Path,
) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "do_direct_summary_v1",
                "status": "complete_right_censored",
                "all_models_complete": True,
                "http_402_latched": False,
                "started_at": "2026-08-24T04:36:15+00:00",
                "ended_at": "2026-08-24T06:48:14+00:00",
                "prior_cost_usd": 179.0,
                "conservative_exposure_usd": 198.0,
                "max_cost_usd": 385.0,
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "model_specs": [
            asdict(MODEL_BY_ID[model_id]) for model_id in EXPECTED_ENDPOINT_IDS
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    fields = _source_cost_ledger_fields(
        tmp_path,
        expected_source_kind="direct_aimd",
        required=True,
        prefer_portable_reconciliation=True,
    )
    assert fields["cost_basis"] == "source_terminal_summary_current_frozen_contract"
    assert fields["prior_conservative_exposure_usd"] == 179.0
    assert fields["cumulative_conservative_exposure_usd"] == 198.0

    manifest["model_specs"][0]["input_usd_per_million"] += 0.01
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DirectReportError, match="portable reconciliation is required"):
        _source_cost_ledger_fields(
            tmp_path,
            expected_source_kind="direct_aimd",
            required=True,
            prefer_portable_reconciliation=True,
        )


def test_cost_bearing_breadth_manifest_requires_terminal_summary(
    tmp_path: Path,
) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": "do_direct_context_manifest_v3"}),
        encoding="utf-8",
    )
    assert _breadth_cost_summary_required(tmp_path) is True
    with pytest.raises(DirectReportError, match="terminal summary.json is required"):
        _source_cost_ledger_fields(
            tmp_path,
            expected_source_kind="direct_breadth",
            required=True,
        )


def test_normalizer_allowlists_public_fields_and_rejects_old_deepseek() -> None:
    normalized = normalize_request(
        _breadth_record(cell_id="one", family="short_exact"),
        source_kind="direct_breadth",
        source_id="breadth",
    )
    encoded = json.dumps(normalized)
    assert "TOP SECRET" not in encoded
    assert "Bearer" not in encoded
    assert "must not be copied" not in encoded
    assert normalized["endpoint_id"] == DEEPSEEK_ENDPOINT_ID
    missing_cost = _breadth_record(cell_id="missing-cost", family="short_exact")
    missing_cost.pop("estimated_cost_usd")
    normalized_missing_cost = normalize_request(
        missing_cost, source_kind="direct_breadth", source_id="breadth"
    )
    assert normalized_missing_cost["estimated_cost_usd"] is None
    assert normalized_missing_cost["cost_attributed"] is False
    bad = _breadth_record(cell_id="bad", family="short_exact")
    bad["model_id"] = "deepseek-4-flash"
    with pytest.raises(DirectReportError, match="only deepseek-v4-flash-0731"):
        normalize_request(bad, source_kind="direct_breadth", source_id="breadth")


def test_exact_direct_aimd_epoch_schema_is_normalized_without_semantic_loss() -> None:
    row = normalize_epoch(
        {
            "schema_version": "do_direct_epoch_v1",
            "campaign_id": "campaign",
            "epoch_id": "epoch",
            "model_id": DEEPSEEK_ENDPOINT_ID,
            "shape": "long_short",
            "phase": "discover",
            "offered_rps_target": 1.5,
            "offered_rps_realized_schedule": 1.45,
            "elapsed_seconds_including_drain": 22.0,
            "scheduled_requests": 30,
            "provider_send_attempts": 30,
            "completed_requests": 30,
            "successes": 29,
            "quality_passes": 28,
            "http_429": 1,
            "http_5xx": 0,
            "timeouts": 0,
            "achieved_rpm": 81.8,
            "successful_rpm": 79.1,
            "effective_input_tpm": 250_000.0,
            "effective_output_tpm": 5_000.0,
            "quality_adjusted_input_tpm": 240_000.0,
            "quality_adjusted_output_tpm": 4_800.0,
            "schedule_lag_p95_seconds": 0.02,
            "max_observed_concurrency": 17,
            "accounted_cost_usd": 0.5,
            "healthy": False,
            "health_reasons": ["rate_limit_rate_above_0.01"],
            "valid_for_capacity": True,
        },
        source_kind="direct_aimd",
        source_id="aimd",
    )
    assert row["offered_rps"] == 1.45
    assert row["offered_rps_target"] == 1.5
    assert row["offered_rps_realized_schedule"] == 1.45
    assert row["scheduled_count"] == 30
    assert row["completed_count"] == 30
    assert row["success_count"] == 29
    assert row["completed_rpm"] == 81.8
    assert row["achieved_rpm"] == 79.1
    assert row["p95_arrival_lag_seconds"] == 0.02
    assert row["peak_concurrency"] == 17
    assert row["estimated_cost_usd"] == 0.5


def test_direct_aimd_low_rate_metrics_use_full_arrival_window() -> None:
    row = normalize_epoch(
        {
            "schema_version": "do_direct_epoch_v1",
            "campaign_id": "campaign",
            "epoch_id": "low-rate",
            "model_id": DEEPSEEK_ENDPOINT_ID,
            "shape": "mixed",
            "phase": "confirmation",
            "arrival_mode": "open_loop",
            "offered_rps_target": 0.133,
            "offered_rps_realized_schedule": 0.2,
            "epoch_seconds": 5.0,
            "elapsed_seconds_including_drain": 0.5,
            "scheduled_requests": 1,
            "completed_requests": 1,
            "successes": 1,
            "quality_passes": 1,
            # Raw runner values used the invalid 0.5-second denominator.
            "achieved_rpm": 120.0,
            "successful_rpm": 120.0,
            "effective_input_tpm": 1200.0,
            "effective_output_tpm": 120.0,
            "goodput_rpm": 120.0,
            "healthy": True,
            "valid_for_capacity": True,
        },
        source_kind="direct_aimd",
        source_id="aimd",
    )
    assert row["drain_elapsed_seconds_observed"] == 0.5
    assert row["elapsed_seconds"] == 5.0
    assert row["offered_rps"] == 0.2
    assert row["offered_rpm"] == 12.0
    assert row["offered_rpm_target"] == pytest.approx(7.98)
    assert row["completed_rpm"] == 12.0
    assert row["achieved_rpm"] == 12.0
    assert row["effective_input_tpm"] == 120.0
    assert row["effective_output_tpm"] == 12.0
    assert row["aggregate_output_goodput_tokens_per_second"] == 0.2


def test_coverage_maps_direct_shapes_quality_phases_and_breadth_families() -> None:
    plans = [
        {
            "source_kind": "direct_breadth",
            "source_id": "capabilities",
            "endpoint_id": DEEPSEEK_ENDPOINT_ID,
            "cell_id": "parameter-temperature",
            "workload": "parameter_temperature",
            "shape": None,
            "planned_attempt_count": 1,
        },
        {
            "source_kind": "direct_breadth",
            "source_id": "capabilities",
            "endpoint_id": DEEPSEEK_ENDPOINT_ID,
            "cell_id": "capability-tools",
            "workload": "capability_tool_calling",
            "shape": None,
            "planned_attempt_count": 1,
        },
    ]
    requests = [
        {
            "source_id": "capabilities",
            "cell_id": plan["cell_id"],
            "transport_success": True,
            "http_status": 200,
            "status": "success",
        }
        for plan in plans
    ]
    epochs = [
        {
            "source_kind": "direct_aimd",
            "source_id": "aimd",
            "endpoint_id": DEEPSEEK_ENDPOINT_ID,
            "epoch_id": "baseline-32k",
            "shape": "input32k_short",
            "workload": "input32k_short",
            "phase": "serial_baseline",
            "scheduled_count": 4,
            "completed_count": 4,
            "valid_for_capacity": True,
        },
        {
            "source_kind": "direct_aimd",
            "source_id": "aimd",
            "endpoint_id": DEEPSEEK_ENDPOINT_ID,
            "epoch_id": "confirmation-long-output",
            "shape": "short_long",
            "workload": "short_long",
            "phase": "confirmation",
            "scheduled_count": 8,
            "completed_count": 8,
            "valid_for_capacity": True,
        },
        {
            "source_kind": "direct_aimd",
            "source_id": "aimd",
            "endpoint_id": DEEPSEEK_ENDPOINT_ID,
            "epoch_id": "mixed-screen",
            "shape": "mixed",
            "workload": "mixed",
            "phase": "fixed_screen",
            "scheduled_count": 8,
            "completed_count": 8,
            "valid_for_capacity": True,
        },
    ]
    ledger, _, _ = build_coverage(plans, requests, epochs)
    dimensions = {
        (row["cell_or_epoch_id"], row["coverage_dimension"]) for row in ledger
    }
    assert ("parameter-temperature", "parameter_validation") in dimensions
    assert ("capability-tools", "tool_calling") in dimensions
    assert ("baseline-32k", "aimd_long_short") in dimensions
    assert ("baseline-32k", "quality_low_load") in dimensions
    assert ("confirmation-long-output", "aimd_short_long") in dimensions
    assert (
        "confirmation-long-output",
        "quality_near_saturation",
    ) in dimensions
    assert ("mixed-screen", "aimd_mixed") in dimensions
    mixed = next(row for row in ledger if row["cell_or_epoch_id"] == "mixed-screen")
    assert mixed["status"] == "completed"
    assert mixed["evidence_scope"] == "exploratory_fixed_screen"


def test_quality_coverage_requires_request_level_scores() -> None:
    endpoint = DEEPSEEK_ENDPOINT_ID
    epochs = [
        {
            "source_kind": "direct_aimd",
            "source_id": "aimd",
            "endpoint_id": endpoint,
            "epoch_id": "baseline",
            "shape": "short_short",
            "workload": "short_short",
            "phase": "serial_baseline",
            "scheduled_count": 1,
            "completed_count": 1,
            "valid_for_capacity": True,
        },
        {
            "source_kind": "direct_aimd",
            "source_id": "aimd",
            "endpoint_id": endpoint,
            "epoch_id": "confirmation",
            "shape": "short_short",
            "workload": "short_short",
            "phase": "confirmation",
            "scheduled_count": 1,
            "completed_count": 1,
            "valid_for_capacity": True,
        },
    ]
    requests = [
        {
            "source_kind": "direct_aimd",
            "source_id": "aimd",
            "endpoint_id": endpoint,
            "epoch_id": epoch_id,
            "quality_scored": False,
            "functional_valid": None,
        }
        for epoch_id in ("baseline", "confirmation")
    ]
    ledger, matrix, _ = build_coverage([], requests, epochs)
    quality_rows = [
        row
        for row in ledger
        if row["coverage_dimension"] in {"quality_low_load", "quality_near_saturation"}
    ]
    assert len(quality_rows) == 2
    assert {row["status"] for row in quality_rows} == {"inconclusive"}
    matrix_by_dimension = {
        row["coverage_dimension"]: row["status"]
        for row in matrix
        if row["endpoint_id"] == endpoint
    }
    assert matrix_by_dimension["quality_low_load"] == "inconclusive"
    assert matrix_by_dimension["quality_near_saturation"] == "inconclusive"


def test_exact_matched_control_retry_supersedes_old_unpaired_probe() -> None:
    endpoint = DEEPSEEK_ENDPOINT_ID
    plans = [
        {
            "source_kind": "direct_breadth",
            "source_id": "do-direct-capability-20260823",
            "endpoint_id": endpoint,
            "cell_id": "old",
            "probe_id": "temperature--0.01",
            "task_id": "cap-temperature--0.01",
            "workload": "parameter_validation",
            "shape": "capability_envelope",
            "planned_attempt_count": 1,
        },
        {
            "source_kind": "direct_breadth",
            "source_id": "do-matched-closure-r2",
            "endpoint_id": endpoint,
            "cell_id": "new",
            "probe_id": "temperature--0.01",
            "task_id": "cap-temperature--0.01",
            "workload": "parameter_validation",
            "shape": "matched_control_closure",
            "planned_attempt_count": 1,
        },
    ]
    requests = [
        {
            "source_kind": "direct_breadth",
            "source_id": plans[0]["source_id"],
            "endpoint_id": endpoint,
            "cell_id": "old",
            "coverage_conclusive": False,
            "transport_success": False,
        },
        {
            "source_kind": "direct_breadth",
            "source_id": plans[1]["source_id"],
            "endpoint_id": endpoint,
            "cell_id": "new",
            "coverage_conclusive": True,
            "coverage_classification": "matched_control_rejection",
            "transport_success": False,
        },
    ]
    ledger, matrix, _ = build_coverage(plans, requests, [])
    old = next(row for row in ledger if row["cell_or_epoch_id"] == "old")
    new = next(row for row in ledger if row["cell_or_epoch_id"] == "new")
    assert old["status"] == "superseded"
    assert new["status"] == "unsupported"
    status = next(
        row["status"]
        for row in matrix
        if row["endpoint_id"] == endpoint
        and row["coverage_dimension"] == "parameter_validation"
    )
    assert status == "unsupported"


def test_zero_quality_score_is_a_scored_failure_not_missing_evidence() -> None:
    raw = _breadth_record(cell_id="quality-zero", family="short_short")
    raw.update(
        {
            "epoch_id": "baseline",
            "shape": "short_short",
            "quality_score": 0.0,
        }
    )
    request = normalize_request(
        raw,
        source_kind="direct_aimd",
        source_id="aimd",
    )
    assert request["quality_scored"] is True
    assert request["quality_score"] == 0.0
    assert request["functional_valid"] is False

    epoch = {
        "source_kind": "direct_aimd",
        "source_id": "aimd",
        "endpoint_id": DEEPSEEK_ENDPOINT_ID,
        "epoch_id": "baseline",
        "shape": "short_short",
        "workload": "short_short",
        "phase": "serial_baseline",
        "scheduled_count": 1,
        "completed_count": 1,
        "valid_for_capacity": True,
    }
    ledger, _, _ = build_coverage([], [request], [epoch])
    quality_row = next(
        row for row in ledger if row["coverage_dimension"] == "quality_low_load"
    )
    assert quality_row["status"] == "completed"
    assert quality_row["conclusive_attempt_count"] == 1

    summary = summarize_group(
        [request],
        [],
        seed=11,
        bootstrap_replicates=20,
    )
    assert summary["quality_scored_count"] == 1
    assert summary["quality_pass_rate"] == 0.0
    assert summary["quality_pass_rate_ci95"]["estimate"] == 0.0


def test_load_points_match_epoch_unit_ci_estimands() -> None:
    requests = [
        {
            "transport_success": success,
            "scientific_success": success,
            "goodput_success": success,
            "quality_scored": False,
            "ttft_seconds": ttft,
            "request_seconds": latency,
            "output_tokens_per_second": decode,
            "input_tokens": 10,
            "output_tokens": 2,
            "estimated_cost_usd": 0.0,
            "started_at": float(index),
            "ended_at": float(index) + latency,
        }
        for index, (success, ttft, latency, decode) in enumerate(
            (
                (True, 0.1, 0.8, 30.0),
                (True, 0.2, 1.0, 25.0),
                (False, 5.0, 6.0, 1.0),
            )
        )
    ]
    epochs = [
        {
            "success_rate": 1.0,
            "elapsed_seconds": 1.0,
            "ttft_p50_seconds": 0.15,
            "ttft_p90_seconds": 0.2,
            "ttft_p95_seconds": 0.2,
            "latency_p50_seconds": 0.9,
            "latency_p90_seconds": 1.0,
            "latency_p95_seconds": 1.0,
            "aggregate_output_goodput_tokens_per_second": 27.5,
        },
        {
            "success_rate": 0.0,
            "elapsed_seconds": 6.0,
            "ttft_p50_seconds": 5.0,
            "ttft_p90_seconds": 5.0,
            "ttft_p95_seconds": 5.0,
            "latency_p50_seconds": 6.0,
            "latency_p90_seconds": 6.0,
            "latency_p95_seconds": 6.0,
            "aggregate_output_goodput_tokens_per_second": 1.0,
        },
    ]
    summary = summarize_group(
        requests,
        epochs,
        seed=7,
        bootstrap_replicates=20,
    )
    for point_key, ci_key in (
        ("transport_success_rate", "transport_success_rate_ci95"),
        ("ttft_p50_seconds", "ttft_p50_ci95"),
        ("latency_p50_seconds", "latency_p50_ci95"),
        (
            "aggregate_output_goodput_tps_epoch_p50",
            "aggregate_output_goodput_tps_epoch_p50_ci95",
        ),
    ):
        assert summary[point_key] == pytest.approx(summary[ci_key]["estimate"])
        assert summary[ci_key]["sampling_unit"] == "epoch_id"
    assert summary["post_ttft_output_tps_proxy_p50"] is None
    assert summary["post_ttft_output_tps_proxy_observation_count"] == 0
    assert summary["post_ttft_output_tps_proxy_p50_ci95"]["estimate"] is None
    assert (
        summary["post_ttft_output_tps_proxy_p50_ci95"]["sampling_unit"] == "request_id"
    )
    assert summary["aggregate_output_goodput_epoch_observation_count"] == 2


def test_serial_post_ttft_proxy_never_becomes_epoch_goodput() -> None:
    requests = [
        {
            "transport_success": True,
            "scientific_success": True,
            "goodput_success": True,
            "quality_scored": False,
            "request_seconds": 1.0,
            "post_ttft_output_tokens_per_second_proxy": value,
            "input_tokens": 10,
            "output_tokens": 2,
            "estimated_cost_usd": 0.0,
            "started_at": float(index),
            "ended_at": float(index) + 1.0,
        }
        for index, value in enumerate((20.0, 30.0, 40.0))
    ]
    summary = summarize_group(
        requests,
        [],
        seed=8,
        bootstrap_replicates=20,
    )
    assert summary["post_ttft_output_tps_proxy_p50"] == 30.0
    assert summary["post_ttft_output_tps_proxy_observation_count"] == 3
    assert (
        summary["post_ttft_output_tps_proxy_p50_ci95"]["sampling_unit"] == "request_id"
    )
    assert summary["aggregate_output_goodput_tps_epoch_p50"] is None
    assert summary["aggregate_output_goodput_epoch_observation_count"] == 0
    assert (
        summary["aggregate_output_goodput_tps_epoch_p50_ci95"]["sampling_unit"]
        == "epoch_id"
    )


def test_analysis_combines_breadth_and_aimd(tmp_path: Path) -> None:
    breadth = tmp_path / "breadth-direct"
    plan_rows = []
    records = []
    for index, family in enumerate(
        ("short_exact", "long_context_retrieval", "controlled_output")
    ):
        cell_id = f"cell-{index}"
        plan_rows.append(
            {
                "cell_id": cell_id,
                "model_id": DEEPSEEK_ENDPOINT_ID,
                "task": {
                    "task_id": cell_id,
                    "family": family,
                    "context_bucket": "32768" if "context" in family else "short",
                    "output_bucket": "4096" if "output" in family else "short",
                    "requires_vision": False,
                },
            }
        )
        records.append(
            _breadth_record(
                cell_id=cell_id,
                family=family,
                input_tokens=32_768 if "context" in family else 128,
                output_tokens=1_024 if "output" in family else 32,
            )
        )
    _write_jsonl(breadth / "plan.jsonl", plan_rows)
    (breadth / "manifest.json").write_text(
        json.dumps(
            {
                "models": [DEEPSEEK_ENDPOINT_ID],
                "scope_exclusions": {
                    "adaptive_tool_over_limit_followups": (
                        "not part of this fixed plan; report as untested rather than inferred"
                    ),
                    "conditional_retry_backoff_followups": (
                        "a separate recovery experiment, not part of this no-retry lane"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    records.append(
        _breadth_record(
            cell_id="orphan-breadth",
            family="long_context_retrieval",
            input_tokens=999_999,
        )
    )
    _write_jsonl(breadth / "records.jsonl", records)

    aimd = tmp_path / "aimd-direct"
    request_rows = []
    epoch_rows = []
    for index, (rate, healthy) in enumerate(
        ((1.0, True), (1.0, True), (1.0, True), (2.0, False), (2.0, False))
    ):
        epoch_id = f"epoch-{index}"
        request_rows.append(
            {
                **_breadth_record(cell_id=f"aimd-{index}", family="aimd_short_short"),
                "schema_version": "do_direct_request_v1",
                "request_id": f"request-{index}",
                "epoch_id": epoch_id,
                "shape": "short_short",
                "phase": "confirm" if rate == 1.0 else "discover",
                "offered_rps": rate,
            }
        )
        epoch_rows.append(
            {
                "schema_version": "do_direct_epoch_v1",
                "epoch_id": epoch_id,
                "model_id": DEEPSEEK_ENDPOINT_ID,
                "shape": "short_short",
                "phase": "confirm" if rate == 1.0 else "discover",
                "sequence": index,
                "offered_rps": rate,
                "elapsed_seconds": 20.0,
                "scheduled_count": 20,
                "completed_count": 20,
                "success_count": 20 if healthy else 18,
                "success_rate": 1.0 if healthy else 0.9,
                "achieved_rpm": 60.0 if healthy else 54.0,
                "effective_input_tpm": 10_000.0,
                "effective_output_tpm": 2_000.0,
                "ttft_p50_seconds": 0.4,
                "latency_p95_seconds": 2.0,
                "healthy": healthy,
                "valid_for_capacity": True,
            }
        )
    _write_jsonl(aimd / "requests.jsonl", request_rows)
    _write_jsonl(aimd / "epochs.jsonl", epoch_rows)
    (aimd / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "do_direct_summary_v1",
                "status": "complete_right_censored",
                "all_models_complete": True,
                "http_402_latched": False,
                "started_at": "2026-08-23T14:00:00+00:00",
                "ended_at": "2026-08-23T15:00:00+00:00",
                "prior_cost_usd": 0.0,
                "conservative_exposure_usd": 0.001,
                "max_cost_usd": 200.0,
            }
        ),
        encoding="utf-8",
    )
    reconciliation_body = {
        "schema_version": "do_direct_aimd_reconciliation_v1",
        "settlement": {
            "reconciled_prior_exposure_usd": 0.0,
            "reconciled_cumulative_exposure_usd": 0.001,
        },
    }
    (aimd / "reconciliation-portable.json").write_text(
        json.dumps(
            {
                **reconciliation_body,
                "receipt_sha256": hashlib.sha256(
                    json.dumps(
                        reconciliation_body,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("utf-8")
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "public-analysis"
    analysis = analyze_and_write(
        breadth_directories=[breadth],
        aimd_directories=[aimd],
        endpoint_freeze=ROOT / "config" / "endpoint-freeze.json",
        output_directory=output,
        seed=7,
        bootstrap_replicates=50,
    )
    assert len(analysis["endpoint_inventory"]) == len(EXPECTED_ENDPOINT_IDS)
    assert [row["endpoint_id"] for row in analysis["endpoint_inventory"]] == list(
        EXPECTED_ENDPOINT_IDS
    )
    assert not analysis["coverage_summary"]["is_100_percent"]
    capacity = next(
        row
        for row in analysis["capacity_summaries"]
        if row["endpoint_id"] == DEEPSEEK_ENDPOINT_ID and row["shape"] == "short_short"
    )
    assert capacity["confirmed_healthy_offered_rps"] == 1.0
    assert "sustainable_confirmed_rps" not in capacity
    assert "sustainable_rpm" not in capacity
    assert capacity["capacity_lower_bound_rps"] == 1.0
    assert capacity["capacity_upper_bound_rps"] == 2.0
    assert capacity["right_censored"] is False
    assert "recommended_rpm" not in capacity
    assert "recommended_headroom_rps" not in capacity
    assert capacity["saturation_knee_rps"] == 2.0
    assert capacity["achieved_rpm_ci95"]["sampling_unit"] == "epoch_id"
    breadth_summary = next(
        row
        for row in analysis["workload_summaries"]
        if row["source_kind"] == "direct_breadth" and row["workload"] == "short_exact"
    )
    assert breadth_summary["sampling_unit"] == "request_id"
    assert breadth_summary["transport_success_interval_method"] == "request_wilson"
    for row in analysis["workload_summaries"]:
        for point_key, ci_key, sampling_unit in (
            (
                "transport_success_rate",
                "transport_success_rate_ci95",
                row["sampling_unit"],
            ),
            ("ttft_p50_seconds", "ttft_p50_ci95", row["sampling_unit"]),
            ("latency_p50_seconds", "latency_p50_ci95", row["sampling_unit"]),
            (
                "post_ttft_output_tps_proxy_p50",
                "post_ttft_output_tps_proxy_p50_ci95",
                "request_id",
            ),
            (
                "aggregate_output_goodput_tps_epoch_p50",
                "aggregate_output_goodput_tps_epoch_p50_ci95",
                "epoch_id",
            ),
        ):
            if row[point_key] is not None:
                assert row[point_key] == pytest.approx(row[ci_key]["estimate"])
            assert row[ci_key]["sampling_unit"] == sampling_unit
    assert len(analysis["endpoint_summaries"]) == len(EXPECTED_ENDPOINT_IDS)
    assert analysis["request_reconciliation"]["matched_request_rows"] == 8
    assert analysis["request_reconciliation"]["orphan_request_rows"] == 1
    assert analysis["request_reconciliation"]["all_requests_reconciled"] is False
    assert analysis["request_reconciliation"]["matched_policy_counts"] == {
        "legacy_id_endpoint_only": 3,
        "persisted_epoch_id_and_endpoint": 5,
    }
    normalized_text = (output / "normalized-requests.jsonl").read_text(encoding="utf-8")
    assert "orphan-breadth" not in normalized_text
    orphan_text = (output / "orphan-requests.jsonl").read_text(encoding="utf-8")
    assert "orphan-breadth" in orphan_text
    context_limit = next(
        row
        for row in analysis["observed_limits"]
        if row["endpoint_id"] == DEEPSEEK_ENDPOINT_ID
        and row["dimension"] == "prompt context window"
    )
    assert context_limit["maximum_accepted_input_tokens"] == 32_768
    assert (output / "coverage-matrix.csv").is_file()
    assert (output / "scope-exclusions.csv").is_file()
    assert (output / "scope-exclusions.jsonl").is_file()
    assert {row["scope_exclusion_id"] for row in analysis["scope_exclusions"]} == {
        "adaptive_tool_over_limit_followups",
        "conditional_retry_backoff_followups",
    }
    exclusion_ledger = [
        json.loads(line)
        for line in (output / "coverage-ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and "manifest_scope_exclusion" in line
    ]
    assert len(exclusion_ledger) == 2
    assert {row["status"] for row in exclusion_ledger} == {"untested"}
    assert {row["coverage_dimension"] for row in exclusion_ledger} == {
        "tool_calling",
        "post_overload_recovery",
    }
    matrix_by_dimension = {
        row["coverage_dimension"]: row["status"]
        for row in analysis["coverage_matrix"]
        if row["endpoint_id"] == DEEPSEEK_ENDPOINT_ID
    }
    assert matrix_by_dimension["tool_calling"] == "untested"
    assert matrix_by_dimension["post_overload_recovery"] == "untested"
    spoofed_scope = json.loads(json.dumps(analysis))
    next(
        row
        for row in spoofed_scope["coverage_matrix"]
        if row["endpoint_id"] == DEEPSEEK_ENDPOINT_ID
        and row["coverage_dimension"] == "tool_calling"
    )["explicit_untested_subtest_count"] = 0
    with pytest.raises(DirectReportError, match="scope-exclusion"):
        validate_public_analysis_contract(spoofed_scope)
    assert (output / "normalized-requests.jsonl").is_file()
    assert (output / "charts" / "capacity-short-short-matched-points.png").is_file()
    public_bytes = (output / "analysis.json").read_text(encoding="utf-8")
    assert "TOP SECRET MODEL OUTPUT" not in public_bytes
    assert "Bearer secret" not in public_bytes
    scan = json.loads((output / "public-safety-scan.json").read_text(encoding="utf-8"))
    assert scan["passed"] is True
    assert analysis["public_bundle_safety"]["input_policy"] == (
        "allowlisted derived fields only"
    )


def test_scope_exclusion_loader_is_source_bound_and_never_upgrades_coverage(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "capability"
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "models": [DEEPSEEK_ENDPOINT_ID, "qwen3.8-max"],
                "scope_exclusions": {
                    "adaptive_tool_over_limit_followups": "fixed plan omitted +1 followups",
                    "conditional_retry_backoff_followups": "no retries in this lane",
                },
            }
        ),
        encoding="utf-8",
    )
    rows = load_breadth_scope_exclusions(directory)
    assert len(rows) == 4
    assert {row["status"] for row in rows} == {"untested"}
    assert {row["claim_policy"] for row in rows} == {"explicitly_excluded_not_tested"}
    assert {row["source_id"] for row in rows} == {"capability"}
    assert all(len(row["source_manifest_sha256"]) == 64 for row in rows)

    ledger, matrix, _ = build_coverage([], [], [], rows)
    assert len(ledger) == 4
    assert {row["evidence_scope"] for row in ledger} == {"manifest_scope_exclusion"}
    affected = {
        (row["endpoint_id"], row["coverage_dimension"]): row["status"]
        for row in matrix
        if row["endpoint_id"] in {DEEPSEEK_ENDPOINT_ID, "qwen3.8-max"}
        and row["coverage_dimension"]
        in {
            "tool_calling",
            "post_overload_recovery",
        }
    }
    assert affected == {
        (DEEPSEEK_ENDPOINT_ID, "tool_calling"): "untested",
        (DEEPSEEK_ENDPOINT_ID, "post_overload_recovery"): "untested",
        ("qwen3.8-max", "tool_calling"): "untested",
        ("qwen3.8-max", "post_overload_recovery"): "untested",
    }


def test_missing_endpoint_rows_are_explicitly_untested(tmp_path: Path) -> None:
    breadth = tmp_path / "breadth"
    _write_jsonl(
        breadth / "plan.jsonl",
        [
            {
                "cell_id": "only",
                "model_id": DEEPSEEK_ENDPOINT_ID,
                "task": {"task_id": "only", "family": "short_exact"},
            }
        ],
    )
    _write_jsonl(
        breadth / "records.jsonl",
        [_breadth_record(cell_id="only", family="short_exact")],
    )
    output = tmp_path / "analysis"
    analysis = analyze_and_write(
        breadth_directories=[breadth],
        aimd_directories=[],
        endpoint_freeze=ROOT / "config" / "endpoint-freeze.json",
        output_directory=output,
        bootstrap_replicates=10,
    )
    assert analysis["coverage_summary"]["status_counts"]["untested"] > 0
    rows = (output / "coverage-matrix.csv").read_text(encoding="utf-8")
    assert "arcee-trinity-large-thinking" not in rows
    assert "untested" in rows
    assert analysis["publication_status"] == "draft_incomplete_coverage"
    with pytest.raises(DirectReportError, match="100% completed"):
        validate_public_analysis_contract(analysis, require_complete=True)
    bad_schema = json.loads(json.dumps(analysis))
    bad_schema["schema_version"] = "stale-v1"
    with pytest.raises(DirectReportError, match="schema_version"):
        validate_public_analysis_contract(bad_schema)
    bad_unit = json.loads(json.dumps(analysis))
    bad_unit["workload_summaries"][0]["sampling_unit"] = "token"
    with pytest.raises(DirectReportError, match="sampling_unit"):
        validate_public_analysis_contract(bad_unit)
    bad_global = json.loads(json.dumps(analysis))
    bad_global["endpoint_summaries"][0]["effective_input_tpm"] = 123.0
    with pytest.raises(DirectReportError, match="heterogeneous aggregate forbidden"):
        validate_public_analysis_contract(bad_global)
    bad_ci = json.loads(json.dumps(analysis))
    bad_ci["workload_summaries"][0]["transport_success_rate_ci95"]["ci95_low"] = 1.1
    with pytest.raises(DirectReportError, match="CI must contain"):
        validate_public_analysis_contract(bad_ci)
    spoofed = json.loads(json.dumps(analysis))
    spoofed["coverage_summary"].update(
        {
            "completed_or_evidence_backed_unsupported_cells": (
                len(EXPECTED_ENDPOINT_IDS) * len(REQUIRED_COVERAGE_DIMENSIONS)
            ),
            "coverage_fraction": 1.0,
            "is_100_percent": True,
            "status_counts": {
                "completed": (
                    len(EXPECTED_ENDPOINT_IDS) * len(REQUIRED_COVERAGE_DIMENSIONS)
                )
            },
        }
    )
    spoofed["workload_summaries"] = []
    spoofed["capacity_summaries"] = []
    with pytest.raises(DirectReportError, match="coverage_matrix"):
        validate_public_analysis_contract(spoofed, require_complete=True)
    for ci_key, point_key in (
        ("ttft_p50_ci95", "ttft_p50_seconds"),
        ("latency_p50_ci95", "latency_p50_seconds"),
        (
            "post_ttft_output_tps_proxy_p50_ci95",
            "post_ttft_output_tps_proxy_p50",
        ),
        (
            "aggregate_output_goodput_tps_epoch_p50_ci95",
            "aggregate_output_goodput_tps_epoch_p50",
        ),
    ):
        mismatched = json.loads(json.dumps(analysis))
        row = next(
            (
                candidate
                for candidate in mismatched["workload_summaries"]
                if candidate[point_key] is not None
            ),
            None,
        )
        if row is None:
            continue
        point = float(row[point_key])
        row[ci_key].update(
            {
                "estimate": point + 10.0,
                "ci95_low": point + 9.0,
                "ci95_high": point + 11.0,
            }
        )
        with pytest.raises(DirectReportError, match="reported point estimate"):
            validate_public_analysis_contract(mismatched)


def test_analysis_quarantines_historical_partner_rows(tmp_path: Path) -> None:
    breadth = tmp_path / "breadth-with-partner"
    hosted = _breadth_record(cell_id="hosted", family="short_exact")
    partner = _breadth_record(cell_id="partner", family="short_exact")
    partner["model_id"] = "arcee-trinity-large-thinking"
    _write_jsonl(
        breadth / "plan.jsonl",
        [
            {
                "cell_id": "hosted",
                "model_id": DEEPSEEK_ENDPOINT_ID,
                "task": {"task_id": "hosted", "family": "short_exact"},
            },
            {
                "cell_id": "partner",
                "model_id": "arcee-trinity-large-thinking",
                "task": {"task_id": "partner", "family": "short_exact"},
            },
        ],
    )
    _write_jsonl(breadth / "records.jsonl", [hosted, partner])
    (breadth / "manifest.json").write_text(
        json.dumps(
            {
                "models": [DEEPSEEK_ENDPOINT_ID, "arcee-trinity-large-thinking"],
                "scope_exclusions": {},
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_and_write(
        breadth_directories=[breadth],
        aimd_directories=[],
        endpoint_freeze=ROOT / "config" / "endpoint-freeze.json",
        output_directory=tmp_path / "public",
        bootstrap_replicates=10,
    )

    assert all(
        row["endpoint_id"] != "arcee-trinity-large-thinking"
        for row in analysis["endpoint_summaries"]
    )
    quarantine = next(
        row
        for row in analysis["data_sources"]
        if row["source_kind"] == "scope_quarantine"
    )
    assert quarantine["quarantined_rows"]["requests"] == 1
    assert quarantine["request_attributed_estimated_cost_usd"] == pytest.approx(
        partner["estimated_cost_usd"]
    )


def test_capacity_uses_explicit_confirmations_and_never_invents_headroom() -> None:
    common = {
        "source_kind": "direct_aimd",
        "source_id": "campaign",
        "endpoint_id": DEEPSEEK_ENDPOINT_ID,
        "shape": "short_short",
        "valid_for_capacity": True,
        "achieved_rpm": 60.0,
        "completed_rpm": 60.0,
        "effective_input_tpm": 1_000.0,
        "effective_output_tpm": 100.0,
        "ttft_p50_seconds": 0.2,
        "latency_p95_seconds": 0.7,
    }
    # Healthy discovery epochs alone never become a confirmed-capacity claim.
    exploratory = build_capacity_summary(
        [
            {
                **common,
                "epoch_id": f"discover-{index}",
                "phase": "additive_aimd",
                "sequence": index,
                "offered_rps": 4.0,
                "healthy": True,
            }
            for index in range(4)
        ],
        seed=1,
        bootstrap_replicates=10,
    )[0]
    assert exploratory["confirmed_healthy_offered_rps"] is None
    assert exploratory["capacity_claim"] == "unconfirmed_healthy_observation_only"
    assert exploratory["achieved_rpm"] is None

    confirmed_epochs = [
        {
            **common,
            "epoch_id": f"confirmation-{index}",
            "phase": "confirmation",
            "sequence": index,
            "offered_rps": 2.0,
            "healthy": True,
        }
        for index in range(3)
    ]
    right_censored = build_capacity_summary(
        confirmed_epochs,
        seed=1,
        bootstrap_replicates=10,
    )[0]
    assert right_censored["capacity_claim"] == "confirmed_right_censored_lower_bound"
    assert right_censored["capacity_lower_bound_rps"] == 2.0
    assert right_censored["confirmed_healthy_offered_rps"] == 2.0
    assert right_censored["confirmed_healthy_offered_rpm"] == 120.0
    assert right_censored["capacity_metric_kind"] == (
        "healthy_realized_offered_arrival_rate_over_short_confirmation_epochs; "
        "not_drain_inclusive_completed_goodput_or_sustained_capacity"
    )
    assert right_censored["capacity_upper_bound_rps"] is None
    assert right_censored["right_censored"] is True
    assert "recommended_rpm" not in right_censored
    assert "recommended_headroom_rps" not in right_censored


def test_final_analysis_mode_fails_closed_on_incomplete_coverage(
    tmp_path: Path,
) -> None:
    breadth = tmp_path / "breadth"
    _write_jsonl(
        breadth / "plan.jsonl",
        [
            {
                "cell_id": "only",
                "model_id": DEEPSEEK_ENDPOINT_ID,
                "task": {"task_id": "only", "family": "short_exact"},
            }
        ],
    )
    _write_jsonl(
        breadth / "records.jsonl",
        [_breadth_record(cell_id="only", family="short_exact")],
    )
    with pytest.raises(DirectReportError, match="100% completed"):
        analyze_and_write(
            breadth_directories=[breadth],
            aimd_directories=[],
            endpoint_freeze=ROOT / "config" / "endpoint-freeze.json",
            output_directory=tmp_path / "final",
            bootstrap_replicates=5,
            publication_mode="final",
        )


def test_empty_metric_charts_are_suppressed_instead_of_rendered_as_empty_panels(
    tmp_path: Path,
) -> None:
    breadth = tmp_path / "breadth"
    _write_jsonl(
        breadth / "plan.jsonl",
        [
            {
                "cell_id": "failed",
                "model_id": DEEPSEEK_ENDPOINT_ID,
                "task": {"task_id": "failed", "family": "short_exact"},
            }
        ],
    )
    failed = _breadth_record(cell_id="failed", family="short_exact")
    failed.update({"status": "timeout", "http_status": None, "usage": {}})
    failed.pop("timing")
    _write_jsonl(breadth / "records.jsonl", [failed])
    output = tmp_path / "analysis"
    analysis = analyze_and_write(
        breadth_directories=[breadth],
        aimd_directories=[],
        endpoint_freeze=ROOT / "config" / "endpoint-freeze.json",
        output_directory=output,
        bootstrap_replicates=5,
    )
    assert analysis["output_files"]["charts"] == ["charts/coverage-status-matrix.png"]
    assert sorted(path.name for path in (output / "charts").iterdir()) == [
        "coverage-status-matrix.png"
    ]


def test_direct_context_schema_reconciles_estimates_and_server_token_axis(
    tmp_path: Path,
) -> None:
    context = tmp_path / "direct-context"
    plan_rows = [
        {
            "schema_version": "do_direct_context_plan_v1",
            "request_id": "context-accepted",
            "probe_id": "percentage-probe",
            "model_id": DEEPSEEK_ENDPOINT_ID,
            "estimated_target_prompt_tokens": 32_768,
            "requested_max_output_tokens": 32,
            "coverage_tags": ["advertised_context_percentage_0.25"],
            "context_window_anchor_source": "documented_model_page",
        },
        {
            "schema_version": "do_direct_context_plan_v1",
            "request_id": "context-limit",
            "probe_id": "boundary-probe",
            "model_id": DEEPSEEK_ENDPOINT_ID,
            "estimated_target_prompt_tokens": 131_072,
            "requested_max_output_tokens": 32,
            "coverage_tags": ["advertised_context_prompt_estimate_upper"],
            "context_window_anchor_source": "documented_model_page",
        },
        {
            "schema_version": "do_direct_context_plan_v1",
            "request_id": "context-retrieval-failed",
            "probe_id": "percentage-probe-failed",
            "model_id": DEEPSEEK_ENDPOINT_ID,
            "estimated_target_prompt_tokens": 65_536,
            "requested_max_output_tokens": 32,
            "coverage_tags": ["advertised_context_percentage_0.50"],
            "context_window_anchor_source": "documented_model_page",
        },
        {
            "schema_version": "do_direct_context_plan_v1",
            "request_id": "context-generic-400",
            "probe_id": "generic-client-error",
            "model_id": DEEPSEEK_ENDPOINT_ID,
            "estimated_target_prompt_tokens": 140_000,
            "requested_max_output_tokens": 32,
            "coverage_tags": ["observed_transition_bisection"],
            "context_window_anchor_source": "documented_model_page",
        },
        {
            "schema_version": "do_direct_context_plan_v1",
            "request_id": "combined-accepted",
            "probe_id": "combined-lower",
            "model_id": DEEPSEEK_ENDPOINT_ID,
            "estimated_target_prompt_tokens": 48_000,
            "requested_max_output_tokens": 4_096,
            "coverage_tags": ["advertised_context_combined_estimate_lower"],
            "context_window_anchor_source": "documented_model_page",
        },
        {
            "schema_version": "do_direct_context_plan_v1",
            "request_id": "combined-limit",
            "probe_id": "combined-upper",
            "model_id": DEEPSEEK_ENDPOINT_ID,
            "estimated_target_prompt_tokens": 62_000,
            "requested_max_output_tokens": 4_096,
            "coverage_tags": ["advertised_context_combined_estimate_upper"],
            "context_window_anchor_source": "documented_model_page",
        },
    ]
    common = {
        "schema_version": "do_direct_context_request_v1",
        "model_id": DEEPSEEK_ENDPOINT_ID,
        "started_at": "2026-08-23T12:00:00+00:00",
        "ended_at": "2026-08-23T12:00:02+00:00",
        "requested_max_output_tokens": 32,
        "context_window_anchor_source": "documented_model_page",
        "timing": {"request_seconds": 2.0, "ttft_seconds": 0.5},
        "estimated_cost_usd": 0.001,
    }
    records = [
        {
            **common,
            "request_id": "context-accepted",
            "probe_id": "percentage-probe",
            "estimated_target_prompt_tokens": 32_768,
            "coverage_tags": ["advertised_context_percentage_0.25"],
            "status": "success",
            "http_status": 200,
            "transport_success": True,
            "scientific_success": True,
            "functional_valid": True,
            "coverage_classification": "accepted",
            "coverage_conclusive": True,
            "usage": {
                "prompt_tokens": 32_100,
                "completion_tokens": 8,
                "total_tokens": 32_108,
            },
            "actual_prompt_tokens_x_axis": 32_100,
            "planning_error_tokens": -668,
            "planning_absolute_error_tokens": 668,
            "planning_tolerance_tokens": 1_638,
            "planning_within_tolerance": True,
            "retrieval_correct": True,
            "quality_score": 1.0,
        },
        {
            **common,
            "request_id": "context-limit",
            "probe_id": "boundary-probe",
            "estimated_target_prompt_tokens": 131_072,
            "coverage_tags": ["advertised_context_prompt_estimate_upper"],
            "status": "explicit_context_limit_rejection",
            "http_status": 400,
            "transport_success": False,
            "scientific_success": False,
            "functional_valid": False,
            "coverage_classification": "explicit_context_limit_rejection",
            "coverage_conclusive": True,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "actual_prompt_tokens_x_axis": None,
            "retrieval_correct": False,
        },
        {
            **common,
            "request_id": "context-retrieval-failed",
            "probe_id": "percentage-probe-failed",
            "estimated_target_prompt_tokens": 65_536,
            "coverage_tags": ["advertised_context_percentage_0.50"],
            "status": "success",
            "http_status": 200,
            "coverage_classification": "accepted",
            "coverage_conclusive": True,
            "usage": {
                "prompt_tokens": 65_500,
                "completion_tokens": 8,
                "total_tokens": 65_508,
            },
            "actual_prompt_tokens_x_axis": 65_500,
            "retrieval_correct": False,
        },
        {
            **common,
            "request_id": "context-generic-400",
            "probe_id": "generic-client-error",
            "estimated_target_prompt_tokens": 140_000,
            "coverage_tags": ["observed_transition_bisection"],
            "status": "other_4xx_inconclusive",
            "http_status": 400,
            "transport_success": False,
            "scientific_success": False,
            "functional_valid": False,
            "coverage_classification": "other_4xx_inconclusive",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "actual_prompt_tokens_x_axis": None,
            "retrieval_correct": False,
        },
        {
            **common,
            "request_id": "combined-accepted",
            "probe_id": "combined-lower",
            "estimated_target_prompt_tokens": 48_000,
            "requested_max_output_tokens": 4_096,
            "coverage_tags": ["advertised_context_combined_estimate_lower"],
            "status": "success",
            "http_status": 200,
            "coverage_classification": "accepted",
            "coverage_conclusive": True,
            "usage": {
                "prompt_tokens": 47_500,
                "completion_tokens": 8,
                "total_tokens": 47_508,
            },
            "actual_prompt_tokens_x_axis": 47_500,
            "retrieval_correct": True,
        },
        {
            **common,
            "request_id": "combined-limit",
            "probe_id": "combined-upper",
            "estimated_target_prompt_tokens": 62_000,
            "requested_max_output_tokens": 4_096,
            "coverage_tags": ["advertised_context_combined_estimate_upper"],
            "status": "explicit_context_limit_rejection",
            "http_status": 400,
            "coverage_classification": "explicit_context_limit_rejection",
            "coverage_conclusive": True,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "actual_prompt_tokens_x_axis": None,
            "retrieval_correct": False,
        },
    ]
    _write_jsonl(context / "plan.jsonl", plan_rows)
    _write_jsonl(context / "requests.jsonl", records)
    output = tmp_path / "context-analysis"
    analysis = analyze_and_write(
        breadth_directories=[context],
        aimd_directories=[],
        endpoint_freeze=ROOT / "config" / "endpoint-freeze.json",
        output_directory=output,
        bootstrap_replicates=5,
    )
    normalized = {
        row["request_id"]: row
        for row in map(
            json.loads,
            (output / "normalized-requests.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
        if row["source_id"] == "direct-context"
    }
    accepted = normalized["context-accepted"]
    assert accepted["cell_id"] == "context-accepted"
    assert accepted["workload"] == "long_context_retrieval"
    assert accepted["requested_input_tokens"] == 32_768
    assert accepted["estimated_target_input_tokens"] == 32_768
    assert accepted["server_reported_input_tokens"] == 32_100
    assert accepted["input_tokens"] == 32_100
    assert accepted["planning_error_tokens"] == -668
    assert accepted["retrieval_correct"] is True
    assert accepted["functional_valid"] is True
    assert accepted["quality_score"] == 1.0
    assert accepted["goodput_success"] is True
    retrieval_failed = normalized["context-retrieval-failed"]
    assert retrieval_failed["transport_success"] is True
    assert retrieval_failed["scientific_success"] is True
    assert retrieval_failed["functional_valid"] is False
    assert retrieval_failed["quality_score"] == 0.0
    assert retrieval_failed["goodput_success"] is False
    assert normalized["context-limit"]["workload"] == "context_boundary"

    ledger = {
        row["cell_or_epoch_id"]: row
        for row in map(
            json.loads,
            (output / "coverage-ledger.jsonl").read_text(encoding="utf-8").splitlines(),
        )
        if row["source_id"] == "direct-context"
    }
    assert ledger["context-accepted"]["coverage_dimension"] == "input_context"
    assert ledger["context-limit"]["status"] == "completed"
    assert ledger["context-generic-400"]["status"] == "inconclusive"

    context_limit = next(
        row
        for row in analysis["observed_limits"]
        if row["endpoint_id"] == DEEPSEEK_ENDPOINT_ID
        and row["dimension"] == "prompt context window"
    )
    assert context_limit["maximum_accepted_input_tokens"] == 65_500
    assert context_limit["maximum_functionally_valid_input_tokens"] == 32_100
    assert context_limit["minimum_rejected_estimated_input_tokens"] == 131_072
    assert "minimum_rejected_requested_input_tokens" not in context_limit
    assert context_limit["boundary_censoring"] == "interval_censored"
    assert context_limit["boundary_interval_censored"] is True
    assert context_limit["boundary_exact"] is False
    assert context_limit["context_window_anchor_source"] == "documented_model_page"
    combined_limit = next(
        row
        for row in analysis["observed_limits"]
        if row["endpoint_id"] == DEEPSEEK_ENDPOINT_ID
        and row["dimension"] == "combined prompt + requested output"
        and row["requested_output_target"] == 4_096
    )
    assert combined_limit["maximum_accepted_input_tokens"] == 47_500
    assert combined_limit["maximum_functionally_valid_input_tokens"] == 47_500
    assert combined_limit["minimum_rejected_estimated_input_tokens"] == 62_000
    assert combined_limit["maximum_accepted_combined_target_tokens"] == 51_596
    assert combined_limit["maximum_functionally_valid_combined_target_tokens"] == 51_596
    assert combined_limit["minimum_rejected_estimated_combined_target_tokens"] == 66_096
    assert combined_limit["boundary_censoring"] == "interval_censored"
    assert combined_limit["boundary_exact"] is False
    assert not any(
        row["dimension"] == "context window" for row in analysis["observed_limits"]
    )
    assert (
        json.loads((output / "public-safety-scan.json").read_text())["passed"] is True
    )


def test_kimi_context_probe_anchor_is_not_reported_as_documented() -> None:
    row = normalize_request(
        {
            "schema_version": "do_direct_context_request_v1",
            "request_id": "kimi-anchor",
            "model_id": "kimi-k3",
            "status": "success",
            "http_status": 200,
            "estimated_target_prompt_tokens": 65_000,
            "requested_max_output_tokens": 32,
            "coverage_tags": ["undocumented_probe_anchor_prompt_estimate_lower"],
            "context_window_anchor_source": "undocumented_probe_anchor",
            "coverage_classification": "accepted",
            "coverage_conclusive": True,
            "usage": {
                "prompt_tokens": 64_500,
                "completion_tokens": 4,
                "total_tokens": 64_504,
            },
            "actual_prompt_tokens_x_axis": 64_500,
            "retrieval_correct": True,
        },
        source_kind="direct_breadth",
        source_id="direct-context",
    )
    observed = next(
        item
        for item in build_observed_limits(
            [row], [{"endpoint_id": "kimi-k3", "context_window": 65_536}]
        )
        if item["endpoint_id"] == "kimi-k3"
        and item["dimension"] == "prompt context window"
    )
    assert observed["documented_value"] is None
    assert observed["documentation_status"] == "undocumented_probe_anchor_value"
    assert observed["context_window_anchor_source"] == "undocumented_probe_anchor"
    assert observed["context_window_probe_anchor_value"] == (
        KIMI_UNDOCUMENTED_CONTEXT_PROBE_ANCHOR
    )
    assert observed["boundary_censoring"] == "right_censored"


def test_kimi_undocumented_probe_anchor_is_an_endpoint_invariant() -> None:
    observed = next(
        item
        for item in build_observed_limits(
            [],
            [
                {
                    "endpoint_id": "kimi-k3",
                    # Even a mistakenly populated inventory value must never
                    # turn the frozen 65,536 planning anchor into documentation.
                    "context_window": KIMI_UNDOCUMENTED_CONTEXT_PROBE_ANCHOR,
                }
            ],
        )
        if item["endpoint_id"] == "kimi-k3"
        and item["dimension"] == "prompt context window"
    )
    assert observed["documented_value"] is None
    assert observed["documentation_status"] == "undocumented_probe_anchor_value"
    assert observed["context_window_anchor_source"] == "undocumented_probe_anchor"
    assert observed["context_window_probe_anchor_value"] == 65_536


def test_nonmonotonic_context_outcomes_are_inconclusive() -> None:
    rows = [
        {
            "endpoint_id": DEEPSEEK_ENDPOINT_ID,
            "workload": "context_boundary",
            "transport_success": True,
            "functional_valid": True,
            "goodput_success": True,
            "input_tokens": 950,
            # The client estimate remains below the rejected estimate, while
            # the accepted server coordinate crosses it. Either crossing must
            # make the mixed-coordinate interval inconclusive.
            "requested_input_tokens": 850,
            "requested_output_target": 32,
            "coverage_tags": ["observed_prompt_transition_bisection"],
            "context_window_anchor_source": "advertised_official_documentation",
        },
        {
            "endpoint_id": DEEPSEEK_ENDPOINT_ID,
            "workload": "context_boundary",
            "transport_success": False,
            "functional_valid": False,
            "goodput_success": False,
            "input_tokens": None,
            "requested_input_tokens": 900,
            "requested_output_target": 32,
            "coverage_classification": "explicit_context_limit_rejection",
            "coverage_tags": ["observed_prompt_transition_bisection"],
            "context_window_anchor_source": "advertised_official_documentation",
        },
    ]
    observed = next(
        item
        for item in build_observed_limits(
            rows,
            [{"endpoint_id": DEEPSEEK_ENDPOINT_ID, "context_window": 1_048_576}],
        )
        if item["endpoint_id"] == DEEPSEEK_ENDPOINT_ID
        and item["dimension"] == "prompt context window"
    )
    assert observed["boundary_censoring"] == "nonmonotonic_inconclusive"
    assert observed["boundary_interval_censored"] is False
    assert observed["boundary_monotonic"] is False
    assert observed["observed_value"] is None
    assert observed["finding"] == "nonmonotonic_context_outcomes_inconclusive"


def test_transport_failure_terminal_zero_tokens_remain_unknown() -> None:
    row = normalize_request(
        {
            "request_id": "rejected-zero-placeholders",
            "model_id": DEEPSEEK_ENDPOINT_ID,
            "status": "error",
            "http_status": 400,
            # A contradictory source flag cannot upgrade an explicit rejection.
            "transport_success": True,
            "coverage_classification": "explicit_context_limit_rejection",
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "realized_input_tokens": 0,
            "realized_output_tokens": 0,
        },
        source_kind="direct_breadth",
        source_id="context",
    )
    assert row["transport_success"] is False
    assert row["server_reported_input_tokens"] is None
    assert row["input_tokens"] is None
    assert row["output_tokens"] is None
    assert row["prefill_proxy_tokens_per_second"] is None


def test_strict_plan_contract_reconciliation_and_authoritative_lane(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "strict-capability"
    directory.mkdir()
    shared_hashes = {
        "request_identity_sha256": "1" * 64,
        "rendered_payload_sha256": "2" * 64,
        "scorer_contract_sha256": "3" * 64,
        "model_contract_sha256": "4" * 64,
        "documentation_contract_sha256": "5" * 64,
    }
    plan_row = {
        "request_id": "strict-combined",
        "cell_id": "strict-combined",
        "model_id": DEEPSEEK_ENDPOINT_ID,
        "workload_id": "context_boundary",
        "requested_max_output_tokens": 4_096,
        "coverage_tags": ["advertised_context_combined_estimate_lower"],
        **shared_hashes,
    }
    plan_text = json.dumps(plan_row, sort_keys=True) + "\n"
    (directory / "plan.jsonl").write_text(plan_text, encoding="utf-8", newline="\n")
    plan_sha256 = hashlib.sha256(plan_text.encode("utf-8")).hexdigest()
    (directory / "manifest.json").write_text(
        json.dumps({"plan_sha256": plan_sha256}), encoding="utf-8"
    )
    # Request metadata intentionally omits the lane tags, workload, and output
    # target. The exact identity/hash contract proves which plan cell ran, so
    # the authoritative plan supplies those report semantics.
    request_row = {
        "schema_version": "do_direct_context_request_v1",
        "request_id": "strict-combined",
        "model_id": DEEPSEEK_ENDPOINT_ID,
        "status": "success",
        "http_status": 200,
        "transport_success": True,
        "scientific_success": True,
        "functional_valid": True,
        "usage": {
            "prompt_tokens": 48_000,
            "completion_tokens": 8,
            "total_tokens": 48_008,
        },
        "estimated_target_prompt_tokens": 48_500,
        "campaign_plan_sha256": plan_sha256,
        **shared_hashes,
    }
    _write_jsonl(directory / "records.jsonl", [request_row])
    plans, requests = load_breadth_directory(directory)
    matched, orphaned = reconcile_request_rows(plans, requests, [])
    assert not orphaned
    assert len(matched) == 1
    row = matched[0]
    assert row["reconciliation_policy"] == "strict_plan_contract_hashes"
    assert row["workload"] == "context_boundary"
    assert row["workload_provenance"] == "authoritative_plan"
    assert row["coverage_tags"] == ["advertised_context_combined_estimate_lower"]
    assert row["requested_output_target"] == 4_096
    limits = [
        item
        for item in build_observed_limits(
            matched,
            [{"endpoint_id": DEEPSEEK_ENDPOINT_ID, "context_window": 1_048_576}],
        )
        if item["endpoint_id"] == DEEPSEEK_ENDPOINT_ID
    ]
    assert any(
        item["dimension"] == "combined prompt + requested output"
        and item["requested_output_target"] == 4_096
        for item in limits
    )
    prompt = next(
        item for item in limits if item["dimension"] == "prompt context window"
    )
    assert prompt["maximum_accepted_input_tokens"] is None

    bad = dict(requests[0])
    bad["request_identity_sha256"] = "f" * 64
    matched, orphaned = reconcile_request_rows(plans, [bad], [])
    assert not matched
    assert orphaned[0]["orphan_reason"] == "request_identity_sha256_mismatch"

    missing = dict(requests[0])
    missing["request_identity_sha256"] = None
    matched, orphaned = reconcile_request_rows(plans, [missing], [])
    assert not matched
    assert orphaned[0]["orphan_reason"] == "missing_request_identity_sha256"


def test_legacy_reconciliation_is_explicitly_labeled() -> None:
    plan = {
        "source_kind": "direct_breadth",
        "source_id": "legacy",
        "cell_id": "cell",
        "endpoint_id": DEEPSEEK_ENDPOINT_ID,
        "workload": "short_exact",
        "coverage_tags": [],
        "requested_output_target": None,
    }
    request = {
        "source_kind": "direct_breadth",
        "source_id": "legacy",
        "cell_id": "cell",
        "endpoint_id": DEEPSEEK_ENDPOINT_ID,
        "workload": "short_exact",
        "workload_provenance": "request_declared",
        "coverage_tags": [],
        "requested_output_target": None,
    }
    matched, orphaned = reconcile_request_rows([plan], [request], [])
    assert not orphaned
    assert matched[0]["reconciliation_policy"] == "legacy_id_endpoint_only"


def test_epoch_units_do_not_collapse_identical_ids_across_sources() -> None:
    common = {
        "schema_version": "digitalocean_public_epoch_v1",
        "source_kind": "direct_aimd",
        "epoch_id": "epoch-0",
        "endpoint_id": DEEPSEEK_ENDPOINT_ID,
        "workload": "short_short",
        "shape": "short_short",
    }
    epochs = _epoch_units_from_requests(
        [],
        [
            {**common, "source_id": "campaign-a", "offered_rps": 1.0},
            {**common, "source_id": "campaign-b", "offered_rps": 2.0},
        ],
    )
    assert len(epochs) == 2
    assert {(row["source_id"], row["offered_rps"]) for row in epochs} == {
        ("campaign-a", 1.0),
        ("campaign-b", 2.0),
    }


def test_duplicate_normalized_plan_keys_fail_closed(tmp_path: Path) -> None:
    breadth = tmp_path / "duplicate-plan"
    plan = {
        "cell_id": "duplicate",
        "model_id": DEEPSEEK_ENDPOINT_ID,
        "task": {"task_id": "duplicate", "family": "short_exact"},
    }
    _write_jsonl(breadth / "plan.jsonl", [plan, plan])
    _write_jsonl(
        breadth / "records.jsonl",
        [_breadth_record(cell_id="duplicate", family="short_exact")],
    )
    with pytest.raises(DirectReportError, match="duplicate .* plan rows"):
        analyze_and_write(
            breadth_directories=[breadth],
            aimd_directories=[],
            endpoint_freeze=ROOT / "config" / "endpoint-freeze.json",
            output_directory=tmp_path / "analysis",
            bootstrap_replicates=5,
        )
