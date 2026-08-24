from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from do_benchmark.direct_report import (
    DEEPSEEK_ENDPOINT_ID,
    DirectReportError,
    _validate_soak_phase_against_requests,
    analyze_and_write,
    load_soak_directory,
)
from do_benchmark.core import canonical_json
from do_benchmark.direct_aimd import wilson_interval


ROOT = Path(__file__).resolve().parents[1]
CELL_ID = "do-soak-cell-test"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _mutate_json(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _mutate_jsonl(path: Path, mutate) -> None:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mutate(rows)
    _write_jsonl(path, rows)


def _mutate_cell_and_summary(directory: Path, key: str, value) -> None:
    _mutate_jsonl(
        directory / "cells.jsonl", lambda rows: rows[0].__setitem__(key, value)
    )
    _mutate_json(
        directory / "summary.json",
        lambda summary: summary["cells"][0].__setitem__(key, value),
    )


def _soak_fixture(
    directory: Path,
    *,
    mismatched_pair_hash: bool = False,
    expanded_model_inventory: bool = False,
) -> Path:
    directory.mkdir()
    plan_cell = {
        "cell_id": CELL_ID,
        "model_id": DEEPSEEK_ENDPOINT_ID,
        "shape": "short_short",
        "status": "ready",
        "candidate_rate_rps": 1.0,
        "recovery_rate_rps": 0.5,
        "low_load_requests": 4,
        "soak_requests": 4,
        "recovery_requests": 1,
        "total_requests": 9,
        "soak_block_request_counts": [1, 1, 1, 1],
        "workload_contract": {"task_recipe_version": "test-v1"},
        "candidate_evidence": {
            "source_evidence_level": "single_valid_healthy_epoch_exploratory",
            "confirmation_epoch_ids": [],
        },
    }
    plan_identity = {
        "schema_version": "do_direct_soak_plan_v1",
        "provider_adapter": "digitalocean-openai-compatible-streaming",
        "models": (
            [DEEPSEEK_ENDPOINT_ID, "kimi-k3"]
            if expanded_model_inventory
            else [DEEPSEEK_ENDPOINT_ID]
        ),
        "soak_seconds": 120.0,
        "analysis_block_seconds": 30.0,
        "analysis_block_count": 4,
        "quality_pairs_per_cell": 4,
        "recovery_seconds": 30.0,
        "cells": [plan_cell],
    }
    if expanded_model_inventory:
        plan_identity["selected_cells"] = [f"{DEEPSEEK_ENDPOINT_ID}:short_short"]
    plan_sha = hashlib.sha256(canonical_json(plan_identity).encode("utf-8")).hexdigest()
    campaign_id = f"do-soak-{plan_sha[:20]}"
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "do_direct_soak_campaign_v1",
                "campaign_id": campaign_id,
                "plan_sha256": plan_sha,
                "claim_scope": "exact two-minute observation",
            }
        ),
        encoding="utf-8",
    )
    (directory / "plan.json").write_text(
        json.dumps({**plan_identity, "plan_sha256": plan_sha}),
        encoding="utf-8",
    )
    phase_ids = {
        "paired_low_load": "phase-low",
        "two_minute_soak": "phase-soak",
        "post_soak_recovery": "phase-recovery",
    }
    phase_counts = {
        "paired_low_load": 4,
        "two_minute_soak": 4,
        "post_soak_recovery": 1,
    }
    phases = []
    for phase, phase_id in phase_ids.items():
        count = phase_counts[phase]
        offered_target = (
            1.0
            if phase == "two_minute_soak"
            else (0.5 if phase == "post_soak_recovery" else None)
        )
        offered_realized = (
            count / 120.0
            if phase == "two_minute_soak"
            else (count / 30.0 if phase == "post_soak_recovery" else None)
        )
        phases.append(
            {
                "schema_version": "do_direct_soak_phase_v1",
                "campaign_id": campaign_id,
                "plan_sha256": plan_sha,
                "phase_id": phase_id,
                "cell_id": CELL_ID,
                "provider": "digitalocean-serverless-inference",
                "model_id": DEEPSEEK_ENDPOINT_ID,
                "shape": "short_short",
                "phase": phase,
                "status": "complete",
                "scheduled_requests": count,
                "completed_request_rows": count,
                "provider_send_attempts": count,
                "successes": count,
                "quality_passes": count,
                "success_rate": 1.0,
                "quality_pass_rate": 1.0,
                "success_rate_ci95_wilson": wilson_interval(count, count),
                "quality_pass_rate_ci95_wilson": wilson_interval(count, count),
                "elapsed_seconds_including_drain": 30.0,
                "successful_rpm": count * 2.0,
                "successful_rows_with_complete_input_usage": count,
                "successful_rows_with_complete_output_usage": count,
                "input_usage_complete_for_all_successes": True,
                "output_usage_complete_for_all_successes": True,
                "effective_input_tpm": count * 20.0,
                "effective_output_tpm": count * 4.0,
                "ttft_p50_seconds": 0.2,
                "ttft_p95_seconds": 0.2,
                "ttft_p95_ci95_dkw_seconds": [0.2, 0.2],
                "latency_p50_seconds": 1.0,
                "latency_p95_seconds": 1.0,
                "latency_p95_ci95_dkw_seconds": [1.0, 1.0],
                "offered_rps_target": offered_target,
                "offered_rps_realized_schedule": offered_realized,
                "http_429": 0,
                "http_5xx": 0,
                "timeouts": 0,
            }
        )
    _write_jsonl(directory / "phases.jsonl", phases)

    blocks = []
    for index in range(4):
        pair_id = f"pair-{index}"
        blocks.append(
            {
                "schema_version": "do_direct_soak_analysis_block_v1",
                "campaign_id": campaign_id,
                "plan_sha256": plan_sha,
                "analysis_block_id": f"block-{index}",
                "analysis_block_index": index,
                "analysis_block_seconds": 30.0,
                "cell_id": CELL_ID,
                "provider": "digitalocean-serverless-inference",
                "model_id": DEEPSEEK_ENDPOINT_ID,
                "shape": "short_short",
                "phase": "two_minute_soak",
                "candidate_rate_rps": 1.0,
                "scheduled_requests": 1,
                "completed_request_rows": 1,
                "successes": 1,
                "quality_passes": 1,
                "success_rate": 1.0,
                "quality_pass_rate": 1.0,
                "success_rate_ci95_wilson": wilson_interval(1, 1),
                "quality_pass_rate_ci95_wilson": wilson_interval(1, 1),
                "predeclared_acceptance_pass": True,
                "acceptance_reasons": [],
                "offered_rps_realized_schedule": 1 / 30.0,
                "successful_rpm_per_predeclared_window": 2.0,
                "successful_rows_with_complete_input_usage": 1,
                "successful_rows_with_complete_output_usage": 1,
                "input_usage_complete_for_all_successes": True,
                "output_usage_complete_for_all_successes": True,
                "effective_input_tpm_per_predeclared_window": 20.0,
                "effective_output_tpm_per_predeclared_window": 4.0,
                "arrival_cohort_elapsed_seconds_including_drain": 30.0,
                "arrival_cohort_successful_rpm_including_drain": 2.0,
                "arrival_cohort_effective_input_tpm_including_drain": 20.0,
                "arrival_cohort_effective_output_tpm_including_drain": 4.0,
                "ttft_p50_seconds": 0.2,
                "ttft_p95_seconds": 0.2,
                "ttft_p95_ci95_dkw_seconds": [0.2, 0.2],
                "latency_p50_seconds": 1.0,
                "latency_p95_seconds": 1.0,
                "latency_p95_ci95_dkw_seconds": [1.0, 1.0],
                "schedule_lag_p95_seconds": 0.0,
                "queue_growth_late_minus_early_median_seconds": 0.0,
                "http_429": 0,
                "http_5xx": 0,
                "timeouts": 0,
                "quality_pair_count": 1,
                "quality_pairs": [
                    {
                        "quality_pair_id": pair_id,
                        "quality_pair_index": index,
                        "task_family": "direct_short_exact",
                        "complete": True,
                        "exact_payload_hash_match": True,
                        "low_load_quality_score": 1.0,
                        "near_load_quality_score": 1.0,
                        "low_load_quality_pass": True,
                        "near_load_quality_pass": True,
                        "quality_delta_near_minus_low": 0.0,
                    }
                ],
            }
        )
    _write_jsonl(directory / "analysis-blocks.jsonl", blocks)

    requests = []
    pairs = []
    for index in range(4):
        pair_id = f"pair-{index}"
        payload_hash = f"{index + 1:064x}"
        low_id = f"low-{index}"
        near_id = f"near-{index}"
        for role, request_id, phase in (
            ("low_load", low_id, "paired_low_load"),
            ("near_load", near_id, "two_minute_soak"),
        ):
            request_hash = payload_hash
            if mismatched_pair_hash and index == 0 and role == "near_load":
                request_hash = "f" * 64
            requests.append(
                {
                    "schema_version": "do_direct_soak_request_v1",
                    "campaign_id": campaign_id,
                    "plan_sha256": plan_sha,
                    "request_id": request_id,
                    "phase_id": phase_ids[phase],
                    "cell_id": CELL_ID,
                    "provider": "digitalocean-serverless-inference",
                    "endpoint_id": DEEPSEEK_ENDPOINT_ID,
                    "model_id": DEEPSEEK_ENDPOINT_ID,
                    "shape": "short_short",
                    "phase": phase,
                    "task_family": "direct_short_exact",
                    "task_id": "short-exact",
                    "status": "success",
                    "http_status": 200,
                    "provider_send_attempted": True,
                    "request_payload_sha256": request_hash,
                    "quality_pair_id": pair_id,
                    "quality_pair_index": index,
                    "quality_pair_role": role,
                    "quality_score": 1.0,
                    "score_kind": "exact_text",
                    "input_usage_complete": True,
                    "output_usage_complete": True,
                    "requested_max_output_tokens": 64,
                    "accounted_cost_usd": 0.001,
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                    "timing": {"request_seconds": 1.0, "ttft_seconds": 0.2},
                    "load": {"schedule_lag_seconds": 0.0},
                    "started_at": f"2026-08-23T18:00:0{index}+00:00",
                    "ended_at": f"2026-08-23T18:00:1{index}+00:00",
                    "workload_tags": {
                        "load_phase": phase,
                        "analysis_block_index": index if role == "near_load" else None,
                        "paired_analysis_block_index": index,
                    },
                }
            )
        pairs.append(
            {
                "schema_version": "do_direct_soak_quality_pair_v1",
                "campaign_id": campaign_id,
                "plan_sha256": plan_sha,
                "quality_pair_id": pair_id,
                "quality_pair_index": index,
                "cell_id": CELL_ID,
                "provider": "digitalocean-serverless-inference",
                "model_id": DEEPSEEK_ENDPOINT_ID,
                "shape": "short_short",
                "analysis_block_index": index,
                "status": "complete",
                "low_load_request_id": low_id,
                "near_load_request_id": near_id,
                "exact_request_payload_hash_match": True,
                "low_load_success": True,
                "near_load_success": True,
                "low_load_quality_score": 1.0,
                "near_load_quality_score": 1.0,
                "paired_quality_delta_near_minus_low": 0.0,
                "paired_latency_ratio_near_over_low": 1.0,
                "predeclared_quality_acceptance_pass": True,
                "quality_acceptance_reasons": [],
            }
        )
    requests.append(
        {
            "schema_version": "do_direct_soak_request_v1",
            "campaign_id": campaign_id,
            "plan_sha256": plan_sha,
            "request_id": "recovery-0",
            "phase_id": phase_ids["post_soak_recovery"],
            "cell_id": CELL_ID,
            "provider": "digitalocean-serverless-inference",
            "endpoint_id": DEEPSEEK_ENDPOINT_ID,
            "model_id": DEEPSEEK_ENDPOINT_ID,
            "shape": "short_short",
            "phase": "post_soak_recovery",
            "task_family": "direct_short_exact",
            "status": "success",
            "http_status": 200,
            "provider_send_attempted": True,
            "quality_score": 1.0,
            "input_usage_complete": True,
            "output_usage_complete": True,
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            "timing": {"request_seconds": 1.0, "ttft_seconds": 0.2},
            "load": {"schedule_lag_seconds": 0.0},
            "started_at": "2026-08-23T18:03:00+00:00",
            "ended_at": "2026-08-23T18:03:01+00:00",
            "workload_tags": {"load_phase": "post_soak_recovery"},
        }
    )
    _write_jsonl(directory / "requests.jsonl", requests)
    _write_jsonl(directory / "quality-pairs.jsonl", pairs)
    cell = {
        "schema_version": "do_direct_soak_cell_v1",
        "campaign_id": campaign_id,
        "plan_sha256": plan_sha,
        "cell_id": CELL_ID,
        "provider": "digitalocean-serverless-inference",
        "model_id": DEEPSEEK_ENDPOINT_ID,
        "shape": "short_short",
        "status": "complete",
        "execution_complete": True,
        "scientifically_complete": True,
        "candidate_rate_rps": 1.0,
        "source_aimd_evidence": plan_cell["candidate_evidence"],
        "analysis_block_count": 4,
        "quality_pair_count": 4,
        "two_minute_observed_acceptance_pass": True,
        "successful_rpm_block_mean": 2.0,
        "successful_rpm_block_mean_ci95_student_t": [2.0, 2.0],
        "input_usage_complete_for_all_blocks": True,
        "output_usage_complete_for_all_blocks": True,
        "successful_rows_with_complete_input_usage": 4,
        "successful_rows_with_complete_output_usage": 4,
        "effective_input_tpm_block_mean": 20.0,
        "effective_input_tpm_block_mean_ci95_student_t": [20.0, 20.0],
        "effective_output_tpm_block_mean": 4.0,
        "effective_output_tpm_block_mean_ci95_student_t": [4.0, 4.0],
        "paired_quality_delta_mean": 0.0,
        "paired_quality_delta_mean_ci95_student_t": [0.0, 0.0],
        "post_soak_recovery_predeclared_pass": True,
        "post_soak_recovery_acceptance_reasons": [],
        "post_soak_recovery_quality_delta_from_low_load": 0.0,
        "post_soak_recovery_success_rate": 1.0,
        "post_soak_recovery_quality_pass_rate": 1.0,
        "post_soak_recovery_ttft_p95_seconds": 0.2,
        "post_soak_recovery_target_rps": 0.5,
        "post_soak_recovery_realized_schedule_rps": 1 / 30.0,
        "workload_contract": plan_cell["workload_contract"],
        "capacity_generalization": "none",
        "claim_scope": "exact two-minute observation",
        "block_ci_note": "four contiguous blocks; serial correlation not modelled",
    }
    _write_jsonl(directory / "cells.jsonl", [cell])
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "do_direct_soak_summary_v1",
                "campaign_id": campaign_id,
                "plan_sha256": plan_sha,
                "status": "complete",
                "execution_complete": True,
                "scientifically_complete": True,
                "target_cells": 1,
                "terminal_cells": 1,
                "request_rows": 9,
                "analysis_block_rows": 4,
                "quality_pair_rows": 4,
                "http_402_latched": False,
                "started_at": "2026-08-23T18:00:00+00:00",
                "ended_at": "2026-08-23T18:03:00+00:00",
                "prior_cost_usd": 0.0,
                "conservative_exposure_usd": 0.001,
                "max_cost_usd": 200.0,
                "cells": [cell],
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_report_ingests_soak_blocks_as_distinct_analysis_block_units(
    tmp_path: Path,
) -> None:
    evidence = load_soak_directory(_soak_fixture(tmp_path / "soak"))
    assert len(evidence["block_summaries"]) == 4
    assert {row["analysis_block_id"] for row in evidence["block_summaries"]} == {
        "block-0",
        "block-1",
        "block-2",
        "block-3",
    }
    assert {row["sampling_unit"] for row in evidence["block_summaries"]} == {
        "analysis_block_id"
    }
    assert evidence["soak_summaries"][0]["capacity_claim"] == (
        "exact_two_minute_soak_pass"
    )
    assert (
        evidence["soak_summaries"][0][
            "arrival_cohort_successful_rpm_including_drain_block_mean"
        ]
        == 2.0
    )
    assert evidence["soak_summaries"][0]["headline_goodput_denominator"] == (
        "arrival_cohort_elapsed_seconds_including_drain"
    )
    assert "sustainable_confirmed_rps" not in evidence["soak_summaries"][0]


def test_soak_accepts_identity_bound_selected_subset_inventory(tmp_path: Path) -> None:
    evidence = load_soak_directory(
        _soak_fixture(tmp_path / "soak", expanded_model_inventory=True)
    )
    assert len(evidence["cell_rows"]) == 1
    assert evidence["cell_rows"][0]["model_id"] == DEEPSEEK_ENDPOINT_ID


def test_soak_accepts_deadline_terminal_with_explicitly_incomplete_cells(
    tmp_path: Path,
) -> None:
    directory = _soak_fixture(tmp_path / "soak")
    for name in (
        "requests.jsonl",
        "phases.jsonl",
        "analysis-blocks.jsonl",
        "quality-pairs.jsonl",
        "cells.jsonl",
    ):
        (directory / name).write_text("", encoding="utf-8")
    _mutate_json(
        directory / "summary.json",
        lambda summary: summary.update(
            {
                "status": "incomplete",
                "execution_complete": False,
                "scientifically_complete": False,
                "terminal_cells": 0,
                "request_rows": 0,
                "analysis_block_rows": 0,
                "quality_pair_rows": 0,
                "cells": [
                    {
                        "cell_id": CELL_ID,
                        "model_id": DEEPSEEK_ENDPOINT_ID,
                        "shape": "short_short",
                        "status": "skipped_campaign_deadline",
                        "execution_complete": False,
                        "scientifically_complete": False,
                    }
                ],
            }
        ),
    )
    with pytest.raises(DirectReportError, match="cell journal"):
        load_soak_directory(directory)
    evidence = load_soak_directory(directory, allow_incomplete_terminal=True)
    assert len(evidence["plan_cells"]) == 1
    assert evidence["cell_rows"] == []


def test_soak_incomplete_phase_accepts_only_explicit_local_censor_rows() -> None:
    elapsed_seconds = 24.0
    success = {
        "status": "success",
        "provider_send_attempted": True,
        "quality_score": 0.0,
        "input_usage_complete": True,
        "output_usage_complete": True,
        "usage": {"prompt_tokens": 100, "completion_tokens": 200},
        "timing": {"ttft_seconds": 0.5, "request_seconds": 24.0},
    }
    skipped = {
        "status": "skipped_send_deadline",
        "provider_send_attempted": False,
    }
    requests = [success, skipped, skipped, skipped]
    phase = {
        "phase_id": "phase-incomplete",
        "status": "incomplete",
        "scheduled_requests": 4,
        "completed_request_rows": 4,
        "provider_send_attempts": 1,
        "successes": 1,
        "quality_passes": 0,
        "success_rate": 0.25,
        "success_rate_ci95_wilson": wilson_interval(1, 4),
        "quality_pass_rate": 0.0,
        "quality_pass_rate_ci95_wilson": wilson_interval(0, 1),
        "elapsed_seconds_including_drain": elapsed_seconds,
        "successful_rows_with_complete_input_usage": 1,
        "successful_rows_with_complete_output_usage": 1,
        "input_usage_complete_for_all_successes": True,
        "output_usage_complete_for_all_successes": True,
        "successful_rpm": 1 / (elapsed_seconds / 60.0),
        "effective_input_tpm": 100 / (elapsed_seconds / 60.0),
        "effective_output_tpm": 200 / (elapsed_seconds / 60.0),
        "ttft_p50_seconds": 0.5,
        "ttft_p95_seconds": 0.5,
        "ttft_p95_ci95_dkw_seconds": [0.5, 0.5],
        "latency_p50_seconds": 24.0,
        "latency_p95_seconds": 24.0,
        "latency_p95_ci95_dkw_seconds": [24.0, 24.0],
        "http_429": 0,
        "http_5xx": 0,
        "timeouts": 0,
    }
    with pytest.raises(DirectReportError, match="not complete"):
        _validate_soak_phase_against_requests(phase, requests, expected_scheduled=4)
    _validate_soak_phase_against_requests(
        phase, requests, expected_scheduled=4, allow_incomplete=True
    )

    requests[-1] = {
        "status": "mystery_local_skip",
        "provider_send_attempted": False,
    }
    with pytest.raises(DirectReportError, match="invalid incomplete-send cohort"):
        _validate_soak_phase_against_requests(
            phase, requests, expected_scheduled=4, allow_incomplete=True
        )


def test_soak_pair_requires_matching_exact_payload_hash(tmp_path: Path) -> None:
    directory = _soak_fixture(tmp_path / "soak", mismatched_pair_hash=True)
    with pytest.raises(DirectReportError, match="payload hashes"):
        load_soak_directory(directory)


def test_soak_rejects_plan_mutation_without_recomputed_identity(tmp_path: Path) -> None:
    directory = _soak_fixture(tmp_path / "soak")
    _mutate_json(
        directory / "plan.json",
        lambda plan: plan["cells"][0].__setitem__("candidate_rate_rps", 999.0),
    )
    with pytest.raises(DirectReportError, match="recomputed plan identities"):
        load_soak_directory(directory)


def test_soak_requires_terminal_summary(tmp_path: Path) -> None:
    directory = _soak_fixture(tmp_path / "soak")
    (directory / "summary.json").unlink()
    with pytest.raises(DirectReportError, match="terminal summary.json"):
        load_soak_directory(directory)


def test_soak_rejects_phase_send_count_not_backed_by_raw_requests(
    tmp_path: Path,
) -> None:
    directory = _soak_fixture(tmp_path / "soak")
    _mutate_jsonl(
        directory / "phases.jsonl",
        lambda rows: rows[0].__setitem__("provider_send_attempts", 0),
    )
    with pytest.raises(DirectReportError, match="provider_send_attempts"):
        load_soak_directory(directory)


def test_soak_rejects_block_counts_not_backed_by_raw_requests(tmp_path: Path) -> None:
    directory = _soak_fixture(tmp_path / "soak")

    def mutate(rows: list[dict]) -> None:
        rows[0]["scheduled_requests"] = 2
        rows[0]["completed_request_rows"] = 2
        rows[0]["successes"] = 2

    _mutate_jsonl(directory / "analysis-blocks.jsonl", mutate)
    with pytest.raises(DirectReportError, match="scheduled_requests"):
        load_soak_directory(directory)


def test_soak_rejects_tampered_block_metric(tmp_path: Path) -> None:
    directory = _soak_fixture(tmp_path / "soak")
    _mutate_jsonl(
        directory / "analysis-blocks.jsonl",
        lambda rows: rows[0].__setitem__(
            "effective_input_tpm_per_predeclared_window", 999.0
        ),
    )
    with pytest.raises(DirectReportError, match="effective_input_tpm"):
        load_soak_directory(directory)


def test_soak_rejects_pair_scores_not_backed_by_raw_requests(tmp_path: Path) -> None:
    directory = _soak_fixture(tmp_path / "soak")

    def mutate(rows: list[dict]) -> None:
        rows[0]["near_load_quality_score"] = 0.0
        rows[0]["paired_quality_delta_near_minus_low"] = -1.0

    _mutate_jsonl(directory / "quality-pairs.jsonl", mutate)
    with pytest.raises(DirectReportError, match="near_load_quality_score"):
        load_soak_directory(directory)


def test_soak_rejects_ghost_quality_pair_metadata_on_recovery_request(
    tmp_path: Path,
) -> None:
    directory = _soak_fixture(tmp_path / "soak")

    def mutate(rows: list[dict]) -> None:
        recovery = next(row for row in rows if row["phase"] == "post_soak_recovery")
        recovery["quality_pair_id"] = "ghost"
        recovery["quality_pair_role"] = "low_load"
        recovery["quality_pair_index"] = 99

    _mutate_jsonl(directory / "requests.jsonl", mutate)
    with pytest.raises(DirectReportError, match="quality-pair identity"):
        load_soak_directory(directory)


def test_soak_rejects_request_pair_id_absent_from_pair_journal(tmp_path: Path) -> None:
    directory = _soak_fixture(tmp_path / "soak")

    def mutate(rows: list[dict]) -> None:
        low = next(
            row
            for row in rows
            if row.get("quality_pair_role") == "low_load"
            and row.get("quality_pair_index") == 0
        )
        low["quality_pair_id"] = "ghost"

    _mutate_jsonl(directory / "requests.jsonl", mutate)
    with pytest.raises(DirectReportError, match="quality pair"):
        load_soak_directory(directory)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("successful_rpm_block_mean", 999.0),
        ("successful_rpm_block_mean_ci95_student_t", [998.0, 1000.0]),
        ("paired_quality_delta_mean", -1.0),
        ("paired_quality_delta_mean_ci95_student_t", [-2.0, 0.0]),
    ],
)
def test_soak_rejects_tampered_cell_block_or_pair_statistics(
    tmp_path: Path, key: str, value
) -> None:
    directory = _soak_fixture(tmp_path / "soak")
    _mutate_cell_and_summary(directory, key, value)
    with pytest.raises(DirectReportError, match=key):
        load_soak_directory(directory)


def test_analysis_emits_separate_soak_outputs_and_exact_coverage(
    tmp_path: Path,
) -> None:
    soak = _soak_fixture(tmp_path / "soak")
    output = tmp_path / "public"
    analysis = analyze_and_write(
        breadth_directories=[],
        aimd_directories=[],
        soak_directories=[soak],
        endpoint_freeze=ROOT / "config" / "endpoint-freeze.json",
        output_directory=output,
        seed=7,
        bootstrap_replicates=5,
    )
    for name in (
        "soak-cell-summary.csv",
        "soak-block-summary.csv",
        "quality-pair-summary.csv",
        "recovery-summary.csv",
    ):
        assert (output / name).is_file()
    assert analysis["statistical_methodology"]["soak_sampling_unit"] == (
        "analysis_block_id"
    )
    assert analysis["soak_summaries"][0]["block_ci_sampling_unit"] == (
        "analysis_block_id"
    )
    assert analysis["soak_summaries"][0]["paired_quality_ci_sampling_unit"] == (
        "quality_pair_id"
    )
    assert (
        analysis["soak_recovery_summaries"][0][
            "within_phase_binomial_interval_sampling_unit"
        ]
        == "request_id"
    )
    assert analysis["capacity_summaries"] == []
    assert not any(
        chart.startswith("charts/aimd-") for chart in analysis["output_files"]["charts"]
    )
    matrix = {
        row["coverage_dimension"]: row["status"]
        for row in analysis["coverage_matrix"]
        if row["endpoint_id"] == DEEPSEEK_ENDPOINT_ID
    }
    assert matrix["low_load_baseline"] == "completed"
    assert matrix["post_overload_recovery"] == "completed"
    assert matrix["quality_low_load"] == "completed"
    assert matrix["quality_near_saturation"] == "completed"
    assert matrix["aimd_short_short"] == "untested"
