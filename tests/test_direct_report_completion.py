from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from do_benchmark.core import canonical_json, stable_hash
from do_benchmark.direct_completion import attempt_request_id
from do_benchmark.direct_report import (
    DirectReportError,
    EXPECTED_ENDPOINT_IDS,
    REQUIRED_COVERAGE_DIMENSIONS,
    _build_cost_summary,
    _source_cost_ledger_fields,
    analyze_and_write,
    build_coverage,
    load_completion_directory,
    reconcile_request_rows,
)


MODEL_ID = "deepseek-v4-flash-0731"
SOURCE_REQUEST_ID = "context-source-request"
SEMANTIC_ID = "do-completion-probe-test"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _completion_fixture(
    directory: Path, *, with_wave: bool = False, with_censor: bool = False
) -> dict[str, Any]:
    if with_censor and not with_wave:
        raise ValueError("a censor fixture requires a soak wave")
    directory.mkdir()
    attempt_ids = [attempt_request_id(SEMANTIC_ID, index) for index in range(2)]
    unresolved_cell = f"{MODEL_ID}:short_short"
    censored_cell = f"{MODEL_ID}:mixed"
    unresolved_cells = (
        [unresolved_cell, censored_cell]
        if with_censor
        else ([unresolved_cell] if with_wave else [])
    )
    plan = {
        "schema_version": "do_direct_completion_plan_v1",
        "source_contract": {
            "source_exposures_usd": {
                "soak": 180.0,
                "context": 190.0,
                "capability": 195.0,
            },
            "soak_summary_sha256": "1" * 64,
            "context_summary_sha256": "2" * 64,
            "context_plan_sha256": "3" * 64,
            "capability_summary_sha256": "4" * 64,
            "capability_plan_sha256": "5" * 64,
        },
        "models": [MODEL_ID],
        "seed": 20260824,
        "prior_cost_usd": 200.0,
        "max_cost_usd": 400.0,
        "launch_stop_cost_usd": 385.0,
        "drain_reserve_usd": 15.0,
        "duration_hours": 6.0,
        "absolute_hard_deadline": None,
        "send_reserve_minutes": 5.0,
        "max_attempts": 2,
        "retry_backoff_seconds": 2.0,
        "rate_ladder": [0.75],
        "output_token_anchors": [256],
        "initial_unresolved_soak_cells": unresolved_cells,
        "probes": [
            {
                "semantic_id": SEMANTIC_ID,
                "lane": "context_retry",
                "model_id": MODEL_ID,
                "source_request_id": SOURCE_REQUEST_ID,
                "source_probe_id": "context-probe",
                "task_id": "context-task",
                "task_family": "direct_context_retrieval",
                "requested_max_output_tokens": 32,
                "request_payload_sha256": "6" * 64,
                "group_id": None,
                "group_size": 1,
                "estimated_tokens": 1_056,
                "attempt_request_ids": attempt_ids,
            }
        ],
    }
    plan_sha256 = hashlib.sha256(canonical_json(plan).encode("utf-8")).hexdigest()
    campaign_id = f"do-completion-{plan_sha256[:20]}"
    manifest = {
        "schema_version": "do_direct_completion_manifest_v1",
        "campaign_id": campaign_id,
        "plan_sha256": plan_sha256,
        "created_at": "2026-08-24T10:00:00+00:00",
        "planned_semantic_probes": 1,
        "planned_attempt_slots": 2,
        "planned_descending_soak_cells": len(unresolved_cells),
        "largest_single_probe_reservation_usd": 0.5,
        "launch_gate_passes": True,
        "hard_cap_usd": 400.0,
        "launch_stop_cost_usd": 385.0,
        "drain_reserve_usd": 15.0,
    }
    requests = [
        {
            "schema_version": "do_direct_completion_request_v1",
            "campaign_id": campaign_id,
            "plan_sha256": plan_sha256,
            "request_id": attempt_ids[0],
            "semantic_id": SEMANTIC_ID,
            "attempt_index": 0,
            "provider": "digitalocean-serverless-inference",
            "model_id": MODEL_ID,
            "shape": "context_retry",
            "phase": "completion_probe",
            "source_request_id": SOURCE_REQUEST_ID,
            "source_probe_id": "context-probe",
            "provider_send_attempted": True,
            "started_at": "2026-08-24T10:01:00+00:00",
            "ended_at": "2026-08-24T10:01:01+00:00",
            "status": "error",
            "coverage_classification": "transient_provider_failure",
            "coverage_conclusive": False,
            "http_status": 503,
            "error_type": "ReadTimeout",
            "usage": {},
            "timing": {"request_seconds": 1.0},
            "quality_score": 0.0,
            "score_kind": "retrieval",
            "requested_max_output_tokens": 32,
            "worst_case_reserved_cost_usd": 0.1,
            "reserved_prompt_tokens": 1_024,
            "estimated_cost_usd": None,
            "accounted_cost_usd": 0.1,
            "retryable": True,
        },
        {
            "schema_version": "do_direct_completion_request_v1",
            "campaign_id": campaign_id,
            "plan_sha256": plan_sha256,
            "request_id": attempt_ids[1],
            "semantic_id": SEMANTIC_ID,
            "attempt_index": 1,
            "provider": "digitalocean-serverless-inference",
            "model_id": MODEL_ID,
            "shape": "context_retry",
            "phase": "completion_probe",
            "source_request_id": SOURCE_REQUEST_ID,
            "source_probe_id": "context-probe",
            "provider_send_attempted": True,
            "started_at": "2026-08-24T10:01:03+00:00",
            "ended_at": "2026-08-24T10:01:04+00:00",
            "status": "success",
            "http_status": 200,
            "coverage_conclusive": True,
            "scientific_success": True,
            "functional_valid": True,
            "quality_score": 1.0,
            "score_kind": "retrieval",
            "usage": {
                "prompt_tokens": 1_000,
                "completion_tokens": 8,
                "total_tokens": 1_008,
            },
            "timing": {
                "request_seconds": 1.0,
                "ttft_seconds": 0.2,
            },
            "stream": {"event_count": 2, "first_event_kind": "content"},
            "requested_max_output_tokens": 32,
            "worst_case_reserved_cost_usd": 0.1,
            "reserved_prompt_tokens": 1_024,
            "estimated_cost_usd": 0.01,
            "accounted_cost_usd": 0.01,
            "retryable": False,
        },
    ]
    outcomes = [
        {
            "schema_version": "do_direct_completion_probe_outcome_v1",
            "campaign_id": campaign_id,
            "semantic_id": SEMANTIC_ID,
            "lane": "context_retry",
            "model_id": MODEL_ID,
            "source_request_id": SOURCE_REQUEST_ID,
            "source_probe_id": "context-probe",
            "status": "success",
            "coverage_conclusive": True,
            "functional_valid": True,
            "final_request_id": attempt_ids[1],
            "completed_at": "2026-08-24T10:01:04+00:00",
        }
    ]
    cumulative_cost = 201.0 if with_wave else 200.11
    summary = {
        "schema_version": "do_direct_completion_summary_v1",
        "campaign_id": campaign_id,
        "plan_sha256": plan_sha256,
        "status": "incomplete_or_censored" if with_censor else "complete",
        "started_at": "2026-08-24T10:00:00+00:00",
        "ended_at": "2026-08-24T10:10:00+00:00",
        "send_cutoff": "2026-08-24T15:55:00+00:00",
        "hard_deadline": "2026-08-24T16:00:00+00:00",
        "planned_semantic_probes": 1,
        "terminal_probe_outcomes": 1,
        "conclusive_probe_outcomes": 1,
        "request_rows": 2,
        "outlier_audit_rows": 2,
        "soak_waves": 1 if with_wave else 0,
        "initial_unresolved_soak_cells": unresolved_cells,
        "remaining_unresolved_soak_cells": [censored_cell] if with_censor else [],
        "censored_soak_cells": [censored_cell] if with_censor else [],
        "prior_cost_usd": 200.0,
        "conservative_exposure_usd": cumulative_cost,
        "max_cost_usd": 400.0,
        "launch_stop_cost_usd": 385.0,
        "drain_reserve_usd": 15.0,
    }
    _write_json(directory / "plan.json", plan)
    _write_json(directory / "manifest.json", manifest)
    _write_json(directory / "summary.json", summary)
    _write_jsonl(directory / "requests.jsonl", requests)
    _write_jsonl(directory / "probe-outcomes.jsonl", outcomes)
    if with_wave:
        wave_id = stable_hash(
            {
                "campaign_id": campaign_id,
                "wave_index": 0,
                "multiplier": 0.75,
                "cells": sorted(unresolved_cells),
            },
            prefix="do-completion-soak-wave-",
        )
        _write_jsonl(
            directory / "soak-waves.jsonl",
            [
                {
                    "schema_version": "do_direct_completion_soak_wave_v1",
                    "campaign_id": campaign_id,
                    "wave_id": wave_id,
                    "wave_index": 0,
                    "candidate_rate_multiplier": 0.75,
                    "soak_campaign_id": "nested-soak-campaign",
                    "soak_plan_sha256": "7" * 64,
                    "soak_artifact_relative_path": (
                        "wave-0-0.75-eligible-123456789abc" if with_censor else None
                    ),
                    "attempted_cells": [unresolved_cell],
                    "censored_cells": [censored_cell] if with_censor else [],
                    "passed_cells": [unresolved_cell],
                    "unresolved_after": [censored_cell] if with_censor else [],
                    "conservative_exposure_usd": cumulative_cost,
                    "ended_at": "2026-08-24T10:09:00+00:00",
                }
            ],
        )
        if with_censor:
            censor_id = stable_hash(
                {
                    "campaign_id": campaign_id,
                    "endpoint_shape": censored_cell,
                    "first_blocked_wave_index": 0,
                    "candidate_rate_multiplier": 0.75,
                    "blocked_status": "blocked_no_valid_aimd_candidate",
                    "blocked_reason": "three_separated_confirmations_not_attested",
                },
                prefix="do-completion-soak-censor-",
            )
            _write_jsonl(
                directory / "soak-censors.jsonl",
                [
                    {
                        "schema_version": "do_direct_completion_soak_censor_v1",
                        "campaign_id": campaign_id,
                        "censor_id": censor_id,
                        "endpoint_shape": censored_cell,
                        "model_id": MODEL_ID,
                        "shape": "mixed",
                        "wave_index": 0,
                        "candidate_rate_multiplier": 0.75,
                        "status": "censored_ineligible_aimd_prerequisite",
                        "blocked_status": "blocked_no_valid_aimd_candidate",
                        "blocked_reason": "three_separated_confirmations_not_attested",
                    }
                ],
            )
    return {
        "campaign_id": campaign_id,
        "plan_sha256": plan_sha256,
        "attempt_ids": attempt_ids,
        "unresolved_cell": unresolved_cell,
        "censored_cell": censored_cell if with_censor else None,
    }


def test_completion_loader_recomputes_plan_identity(tmp_path: Path) -> None:
    completion = tmp_path / "completion"
    identity = _completion_fixture(completion)

    loaded = load_completion_directory(completion)

    assert loaded["campaign_id"] == identity["campaign_id"]
    assert loaded["plan_sha256"] == identity["plan_sha256"]
    manifest = json.loads((completion / "manifest.json").read_text(encoding="utf-8"))
    manifest["plan_sha256"] = "0" * 64
    _write_json(completion / "manifest.json", manifest)
    with pytest.raises(DirectReportError, match="recomputed plan identity"):
        load_completion_directory(completion)


def test_completion_retries_remain_physical_but_final_is_only_coverage_attempt(
    tmp_path: Path,
) -> None:
    completion = tmp_path / "completion"
    _completion_fixture(completion)
    loaded = load_completion_directory(completion)

    assert len(loaded["requests"]) == 2
    assert [row["semantic_coverage_attempt"] for row in loaded["requests"]] == [
        False,
        True,
    ]
    assert sum(not row["transport_success"] for row in loaded["requests"]) == 1
    assert sum(
        float(row["estimated_cost_usd"]) for row in loaded["requests"]
    ) == pytest.approx(0.11)

    matched, orphaned = reconcile_request_rows(loaded["plans"], loaded["requests"], [])
    assert len(matched) == 2
    assert not orphaned
    ledger, matrix, summary = build_coverage(loaded["plans"], matched, [])
    row = next(item for item in ledger if item["source_kind"] == "direct_completion")
    assert row["physical_attempt_count"] == 2
    assert row["nonfinal_physical_attempt_count"] == 1
    assert row["observed_attempt_count"] == 1
    assert row["conclusive_attempt_count"] == 1
    assert row["status"] == "completed"
    required_cells = len(EXPECTED_ENDPOINT_IDS) * len(REQUIRED_COVERAGE_DIMENSIONS)
    assert len(matrix) == required_cells
    assert summary["required_endpoint_dimension_cells"] == required_cells

    unresolved_plan = copy.deepcopy(loaded["plans"])
    unresolved_plan[0]["terminal_outcome_status"] = "unknown_prior_reservation"
    unresolved_requests = copy.deepcopy(matched)
    for request in unresolved_requests:
        request["semantic_coverage_attempt"] = False
    ledger, _, _ = build_coverage(unresolved_plan, unresolved_requests, [])
    row = next(item for item in ledger if item["source_kind"] == "direct_completion")
    assert row["status"] == "inconclusive"
    assert row["physical_attempt_count"] == 2


def test_completion_supersedes_only_after_conclusive_final_attempt(
    tmp_path: Path,
) -> None:
    completion = tmp_path / "completion"
    _completion_fixture(completion)
    loaded = load_completion_directory(completion)
    source_plan = {
        "source_kind": "direct_breadth",
        "source_id": "context",
        "cell_id": SOURCE_REQUEST_ID,
        "endpoint_id": MODEL_ID,
        "workload": "direct_context_retrieval",
        "shape": "context_boundary",
        "planned_attempt_count": 1,
    }
    source_request = {
        "source_kind": "direct_breadth",
        "source_id": "context",
        "request_id": SOURCE_REQUEST_ID,
        "cell_id": SOURCE_REQUEST_ID,
        "endpoint_id": MODEL_ID,
        "status": "error",
        "coverage_conclusive": False,
    }

    ledger, _, _ = build_coverage(
        [source_plan, *loaded["plans"]],
        [source_request, *loaded["requests"]],
        [],
    )
    original = next(row for row in ledger if row["source_id"] == "context")
    replacement = next(
        row for row in ledger if row["source_kind"] == "direct_completion"
    )
    assert original["status"] == "superseded"
    assert original["superseded_by_request_id"] == loaded["requests"][1]["request_id"]
    assert replacement["supersession_status"] == "applied_to_inconclusive_source"

    capability_plans = copy.deepcopy(loaded["plans"])
    capability_plans[0]["shape"] = "capability_retry"
    capability_plans[0]["workload"] = "direct_capability_parameter"
    parameter_source_plan = {
        **source_plan,
        "workload": "parameter_interactions",
        "shape": "capability_envelope",
    }
    ledger, _, _ = build_coverage(
        [parameter_source_plan, *capability_plans],
        [source_request, *loaded["requests"]],
        [],
    )
    replacement = next(
        row for row in ledger if row["source_kind"] == "direct_completion"
    )
    assert replacement["coverage_dimension"] == "parameter_interactions"
    assert replacement["retry_task_family_inferred_dimension"] == (
        "parameter_validation"
    )
    assert replacement["supersession_dimension_policy"] == (
        "inherited_from_exact_endpoint_and_source_request_id"
    )

    unresolved_requests = copy.deepcopy(loaded["requests"])
    unresolved_requests[1]["coverage_conclusive"] = False
    unresolved_requests[1]["transport_success"] = False
    unresolved_requests[1]["scientific_success"] = False
    ledger, _, _ = build_coverage(
        [source_plan, *loaded["plans"]],
        [source_request, *unresolved_requests],
        [],
    )
    original = next(row for row in ledger if row["source_id"] == "context")
    replacement = next(
        row for row in ledger if row["source_kind"] == "direct_completion"
    )
    assert original["status"] == "inconclusive"
    assert replacement["status"] == "inconclusive"
    assert replacement["supersession_status"].endswith("inconclusive")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "unknown", "unrecognized terminal summary schema"),
        ("status", "running", "cost summary is not terminal"),
    ],
)
def test_completion_cost_receipt_requires_known_schema_and_terminal_state(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    completion = tmp_path / "completion"
    _completion_fixture(completion)
    fields = _source_cost_ledger_fields(
        completion,
        expected_source_kind="direct_completion",
        required=True,
    )
    assert fields["summary_schema_version"] == "do_direct_completion_summary_v1"
    assert fields["terminal_status"] == "complete"

    summary = json.loads((completion / "summary.json").read_text(encoding="utf-8"))
    summary[field] = value
    _write_json(completion / "summary.json", summary)
    with pytest.raises(DirectReportError, match=message):
        _source_cost_ledger_fields(
            completion,
            expected_source_kind="direct_completion",
            required=True,
        )


def test_nested_soak_wave_is_loaded_but_parent_is_only_cost_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = tmp_path / "completion"
    identity = _completion_fixture(completion, with_wave=True)
    seen_overrides: list[str | None] = []

    def fake_load_soak(
        path: Path,
        *,
        source_id_override: str | None = None,
        allow_incomplete_terminal: bool = False,
    ) -> dict[str, Any]:
        seen_overrides.append(source_id_override)
        assert path.name == "wave-0-0.75"
        assert allow_incomplete_terminal is True
        plan_cells = [{"model_id": MODEL_ID, "shape": "short_short"}]
        return {
            "source_kind": "direct_soak",
            "source_id": source_id_override,
            "campaign_id": "nested-soak-campaign",
            "plan_sha256": "7" * 64,
            "plan_cells": plan_cells,
            "cell_rows": [
                {
                    "model_id": MODEL_ID,
                    "shape": "short_short",
                    "scientifically_complete": True,
                    "two_minute_observed_acceptance_pass": True,
                    "post_soak_recovery_predeclared_pass": True,
                }
            ],
            "summary": {"conservative_exposure_usd": 201.0},
        }

    monkeypatch.setattr(
        "do_benchmark.direct_report.load_soak_directory", fake_load_soak
    )
    loaded = load_completion_directory(completion)
    assert len(loaded["nested_soaks"]) == 1
    assert seen_overrides == ["completion-soak-wave-0"]

    completion_fields = _source_cost_ledger_fields(
        completion,
        expected_source_kind="direct_completion",
        required=True,
    )
    sources = [
        {
            "source_kind": "direct_completion",
            "source_id": "completion",
            "cost_summary_required": True,
            **completion_fields,
        },
        {
            "source_kind": "direct_soak",
            "source_id": "completion-soak-wave-0",
            "cost_summary_required": False,
            "cost_stage_policy": "parent_completion_summary_is_single_stage",
        },
    ]
    requests = [
        {"estimated_cost_usd": 0.11, "cost_attributed": True},
        {"estimated_cost_usd": 0.89, "cost_attributed": True},
    ]
    endpoint_summaries = [{"request_count": 2, "estimated_cost_usd": 1.0}]
    cost = _build_cost_summary(sources, requests, endpoint_summaries)
    assert [stage["source_id"] for stage in cost["source_stages"]] == ["completion"]
    assert cost["conservative_campaign_exposure_usd"] == 201.0
    assert identity["unresolved_cell"].endswith(":short_short")


def test_completion_loader_accepts_only_eligible_subwave_and_preserves_censor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = tmp_path / "completion"
    identity = _completion_fixture(completion, with_wave=True, with_censor=True)

    def fake_load_soak(
        path: Path,
        *,
        source_id_override: str | None = None,
        allow_incomplete_terminal: bool = False,
    ) -> dict[str, Any]:
        assert path.name == "wave-0-0.75-eligible-123456789abc"
        assert allow_incomplete_terminal is True
        plan_cells = [{"model_id": MODEL_ID, "shape": "short_short"}]
        return {
            "source_kind": "direct_soak",
            "source_id": source_id_override,
            "campaign_id": "nested-soak-campaign",
            "plan_sha256": "7" * 64,
            "plan_cells": plan_cells,
            "cell_rows": [
                {
                    "model_id": MODEL_ID,
                    "shape": "short_short",
                    "scientifically_complete": True,
                    "two_minute_observed_acceptance_pass": True,
                    "post_soak_recovery_predeclared_pass": True,
                }
            ],
            "summary": {"conservative_exposure_usd": 201.0},
        }

    monkeypatch.setattr(
        "do_benchmark.direct_report.load_soak_directory", fake_load_soak
    )
    loaded = load_completion_directory(completion)

    assert len(loaded["nested_soaks"]) == 1
    assert len(loaded["soak_censors"]) == 1
    assert loaded["soak_censors"][0]["endpoint_shape"] == identity["censored_cell"]
    assert loaded["summary"]["remaining_unresolved_soak_cells"] == [
        identity["censored_cell"]
    ]


def test_completion_loader_rejects_censor_hidden_as_attempted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = tmp_path / "completion"
    identity = _completion_fixture(completion, with_wave=True, with_censor=True)
    waves_path = completion / "soak-waves.jsonl"
    wave = json.loads(waves_path.read_text(encoding="utf-8"))
    wave["attempted_cells"].append(identity["censored_cell"])
    _write_jsonl(waves_path, [wave])
    monkeypatch.setattr(
        "do_benchmark.direct_report.load_soak_directory", lambda *args, **kwargs: {}
    )

    with pytest.raises(DirectReportError, match="soak-wave lineage"):
        load_completion_directory(completion)


def test_completion_directory_flows_through_public_pipeline_and_safety_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = tmp_path / "completion"
    _completion_fixture(completion)
    monkeypatch.setattr(
        "do_benchmark.direct_report._plot_bundle", lambda *args, **kwargs: []
    )

    analysis = analyze_and_write(
        breadth_directories=[],
        aimd_directories=[],
        soak_directories=[],
        completion_directories=[completion],
        endpoint_freeze=Path(__file__).parents[1] / "config" / "endpoint-freeze.json",
        output_directory=tmp_path / "public",
        bootstrap_replicates=10,
        publication_mode="draft",
    )

    assert analysis["contract_gate"]["passed"] is True
    assert analysis["coverage_summary"]["required_endpoint_dimension_cells"] == (
        len(EXPECTED_ENDPOINT_IDS) * len(REQUIRED_COVERAGE_DIMENSIONS)
    )
    assert analysis["public_bundle_safety"]["scanner"].endswith("safety_scan_v1")
    assert [
        stage["summary_schema_version"]
        for stage in analysis["cost_summary"]["source_stages"]
    ] == ["do_direct_completion_summary_v1"]
