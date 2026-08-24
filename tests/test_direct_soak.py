from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from do_benchmark.core import MODEL_BY_ID, MODEL_SPECS, StreamResult
from do_benchmark.direct_soak import (
    DirectSoakCampaign,
    SHAPES,
    SoakConfig,
    SoakPreflightError,
    default_model_ids,
    load_aimd_candidates,
)
from do_benchmark.direct_aimd import DirectAIMDCampaign, DirectConfig
from do_benchmark.direct_completion import _reconcile_invalidated_soak_exposure


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _aimd_fixture(
    root: Path,
    *,
    model_ids: tuple[str, ...],
    rate_rps: float = 100.0,
    blocked: tuple[str, str] | None = None,
    source_prior_cost_usd: float = 0.0,
    realized_rps: float | None = None,
) -> None:
    campaign_id = "do-direct-synthetic-complete"
    manifest = {
        "schema_version": "do_direct_campaign_v1",
        "campaign_id": campaign_id,
        "models": list(model_ids),
        "model_specs": [asdict(MODEL_BY_ID[model_id]) for model_id in model_ids],
        "shapes": list(SHAPES),
        "aimd_shapes": list(SHAPES),
        "prior_cost_usd": source_prior_cost_usd,
        "input_tokens": 32_000,
        "long_output_words": 2,
        "short_max_output_tokens": 16,
        "long_max_output_tokens": 32,
        "mixed_max_output_tokens": 32,
    }
    summary_models: list[dict] = []
    epochs: list[dict] = []
    requests: list[dict] = []
    reservations: list[dict] = []
    for model_id in model_ids:
        shape_rows: list[dict] = []
        for shape in SHAPES:
            lineage: list[str] = []

            def add_epoch(
                phase: str, ordinal: int, offered: float, scheduled: int = 1
            ) -> str:
                epoch_id = f"epoch-{model_id}-{shape}-{phase}-{ordinal}"
                lineage.append(epoch_id)
                serial = phase in {"serial_baseline", "confirmation_separator_serial"}
                realized = offered if realized_rps is None else realized_rps
                epoch_seconds = 1.0 if serial else scheduled / realized
                epochs.append(
                    {
                        "schema_version": "do_direct_epoch_v1",
                        "campaign_id": campaign_id,
                        "epoch_id": epoch_id,
                        "model_id": model_id,
                        "shape": shape,
                        "phase": phase,
                        "offered_rps_target": offered,
                        "offered_rps_realized_schedule": scheduled / epoch_seconds,
                        "epoch_seconds": epoch_seconds,
                        "arrival_mode": "serial" if serial else "open_loop",
                        "scheduled_requests": scheduled,
                        "valid_for_capacity": True,
                        "healthy": True,
                    }
                )
                for request_index in range(scheduled):
                    spec = MODEL_BY_ID[model_id]
                    actual_cost = (
                        64 * spec.input_usd_per_million
                        + 8 * spec.output_usd_per_million
                    ) / 1_000_000
                    max_output_tokens = (
                        16 if shape in {"short_short", "input32k_short"} else 32
                    )
                    requests.append(
                        {
                            "schema_version": "do_direct_request_v1",
                            "campaign_id": campaign_id,
                            "request_id": f"request-{epoch_id}-{request_index}",
                            "epoch_id": epoch_id,
                            "model_id": model_id,
                            "shape": shape,
                            "provider_send_attempted": True,
                            "status": "success",
                            "http_status": 200,
                            "requested_max_output_tokens": max_output_tokens,
                            "worst_case_reserved_cost_usd": 0.01,
                            "reserved_prompt_tokens": 100,
                            "usage_reported": True,
                            "estimated_cost_usd": actual_cost,
                            "accounted_cost_usd": actual_cost,
                            "usage": {
                                "prompt_tokens": 64,
                                "completion_tokens": 8,
                                "total_tokens": 72,
                            },
                        }
                    )
                    reservations.append(
                        {
                            "schema_version": "do_direct_reservation_v1",
                            "campaign_id": campaign_id,
                            "request_id": f"request-{epoch_id}-{request_index}",
                            "epoch_id": epoch_id,
                            "model_id": model_id,
                            "shape": shape,
                            "reserved_cost_usd": 0.01,
                            "reserved_prompt_tokens": 100,
                            "max_output_tokens": max_output_tokens,
                        }
                    )
                return epoch_id

            baseline = add_epoch("serial_baseline", 0, 1.0)
            confirmations = []
            for index in range(3):
                confirmations.append(add_epoch("confirmation", index, rate_rps, 2))
                if index < 2:
                    add_epoch("confirmation_separator_serial", index, 1.0)
            is_blocked = blocked == (model_id, shape)
            shape_rows.append(
                {
                    "model_id": model_id,
                    "shape": shape,
                    "status": "incomplete" if is_blocked else "complete_right_censored",
                    "candidate_confirmed_healthy_rps": rate_rps,
                    "confirmation_target_rps": rate_rps,
                    "candidate_confirmed_three_separated_epochs": not is_blocked,
                    "epoch_ids": lineage,
                    "baseline_epoch_id": baseline,
                    "confirmation_epoch_ids": confirmations,
                }
            )
        summary_models.append(
            {
                "model_id": model_id,
                "status": "complete_right_censored",
                "shapes": shape_rows,
            }
        )
    summary = {
        "schema_version": "do_direct_summary_v1",
        "campaign_id": campaign_id,
        "status": "complete_right_censored",
        "all_models_complete": True,
        "prior_cost_usd": source_prior_cost_usd,
        "conservative_exposure_usd": source_prior_cost_usd
        + sum(float(row["accounted_cost_usd"]) for row in requests),
        "models": summary_models,
    }
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "summary.json", summary)
    _write_jsonl(root / "epochs.jsonl", epochs)
    _write_jsonl(root / "requests.jsonl", requests)
    _write_jsonl(root / "reservations.jsonl", reservations)


def _fake_result(task) -> StreamResult:
    expected = task.expected
    kind = expected["kind"]
    text = ""
    tool_calls = []
    if kind == "exact_text":
        text = str(expected["value"])
    elif kind == "controlled_words":
        count = int(expected["count"])
        text = " ".join(["azure"] * (count - 1) + [str(expected["marker"])])
    elif kind == "json_exact":
        text = json.dumps(expected["value"])
    elif kind == "tool_exact":
        tool_calls = [expected["value"]]
    else:  # pragma: no cover - fixed direct workload universe
        raise AssertionError(kind)
    return StreamResult(
        status_code=200,
        response_headers={
            "x-request-id": "private-provider-request-id",
            "cf-ray": "private-edge-id",
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "999",
        },
        text=text,
        reasoning_text="private reasoning trace",
        tool_calls=tool_calls,
        usage={"prompt_tokens": 64, "completion_tokens": 8, "total_tokens": 72},
        finish_reason="stop",
        request_seconds=0.001,
        headers_seconds=0.0002,
        ttft_seconds=0.0005,
        generation_seconds=0.0005,
        stream_seconds=0.0008,
        event_count=2,
        first_event_kind="content",
    )


def _fast_config(
    aimd_dir: Path, output_dir: Path, *, prior: float | None = None
) -> SoakConfig:
    if prior is None:
        prior = float(
            json.loads((aimd_dir / "summary.json").read_text())[
                "conservative_exposure_usd"
            ]
        )
    return SoakConfig(
        aimd_dir=aimd_dir,
        output_dir=output_dir,
        model_ids=("deepseek-v4-flash-0731",),
        soak_seconds=0.08,
        analysis_block_seconds=0.02,
        analysis_block_count=4,
        concurrency_ceiling=4,
        quality_pairs_per_cell=4,
        recovery_seconds=0.02,
        recovery_rate_fraction=0.5,
        request_timeout_seconds=1.0,
        max_cost_usd=200.0,
        prior_cost_usd=prior,
        stop_launch_at=datetime.now(timezone.utc) + timedelta(minutes=2),
        hard_campaign_deadline=datetime.now(timezone.utc) + timedelta(minutes=3),
    )


def test_candidate_loader_requires_three_separated_receipted_confirmations(
    tmp_path,
) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(
        source,
        model_ids=("deepseek-v4-flash-0731",),
        blocked=("deepseek-v4-flash-0731", "short_long"),
    )
    decisions, provenance = load_aimd_candidates(source, ("deepseek-v4-flash-0731",))
    assert len(decisions) == 4
    blocked = [decision for decision in decisions if decision.status == "blocked"]
    assert [(decision.model_id, decision.shape) for decision in blocked] == [
        ("deepseek-v4-flash-0731", "short_long")
    ]
    assert blocked[0].reason == "aimd_shape_not_complete"
    assert set(provenance["artifact_sha256"]) == {
        "manifest",
        "summary",
        "epochs",
        "requests",
        "reservations",
    }
    assert all(len(value) == 64 for value in provenance["artifact_sha256"].values())


def test_incomplete_aimd_plan_fails_closed_before_executor(tmp_path) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(
        source,
        model_ids=("deepseek-v4-flash-0731",),
        blocked=("deepseek-v4-flash-0731", "mixed"),
    )
    campaign = DirectSoakCampaign(_fast_config(source, tmp_path / "soak"))
    assert campaign.preflight["target_cell_count"] == 4
    assert campaign.preflight["ready_cell_count"] == 3
    assert campaign.preflight["complete_coverage_ready"] is False

    calls = 0

    async def executor(model_id, task, max_output_tokens):
        nonlocal calls
        calls += 1
        return _fake_result(task)

    with pytest.raises(SoakPreflightError):
        asyncio.run(campaign.run(executor))
    assert calls == 0


def test_all_failure_ceiling_can_exceed_cap_when_inflight_gate_fits(tmp_path) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(
        source,
        model_ids=("deepseek-v4-flash-0731",),
        rate_rps=3_000_000.0,
    )
    config = _fast_config(source, tmp_path / "soak", prior=25.33)
    campaign = DirectSoakCampaign(config)
    assert campaign.preflight["total_all_failure_reservation_ceiling_usd"] > 200.0
    assert campaign.preflight["all_failure_ceiling_may_exceed_cap"] is True
    assert campaign.preflight["launch_gate_exposure_usd"] < 200.0
    assert campaign.preflight["budget_passes"] is True
    assert campaign.preflight["passes"] is True


def test_soak_is_sanitized_paired_blocked_and_resume_deduplicated(tmp_path) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",), rate_rps=100.0)
    output = tmp_path / "soak"
    config = _fast_config(source, output)
    calls = 0

    async def executor(model_id, task, max_output_tokens):
        nonlocal calls
        assert model_id == "deepseek-v4-flash-0731"
        assert max_output_tokens > 0
        calls += 1
        return _fake_result(task)

    first = asyncio.run(DirectSoakCampaign(config).run(executor))
    assert first["status"] == "complete"
    assert len(first["cells"]) == 4
    assert all(cell["status"] == "complete" for cell in first["cells"])
    assert all(cell["post_soak_recovery_predeclared_pass"] for cell in first["cells"])
    assert all(cell["post_soak_recovery_target_rps"] == 50.0 for cell in first["cells"])
    assert all(
        cell["post_soak_recovery_realized_schedule_rps"] == 50.0
        for cell in first["cells"]
    )
    assert len((output / "analysis-blocks.jsonl").read_text().splitlines()) == 16
    assert len((output / "quality-pairs.jsonl").read_text().splitlines()) == 17
    first_calls = calls
    assert first_calls == 53

    request_text = (output / "requests.jsonl").read_text(encoding="utf-8")
    for forbidden in (
        "private-provider-request-id",
        "private-edge-id",
        "private reasoning trace",
        "LOAD-OK",
        "CRITICAL NEEDLE",
    ):
        assert forbidden not in request_text
    for line in request_text.splitlines():
        row = json.loads(line)
        assert "messages" not in row
        assert "response" not in row
        assert "response_headers" not in row
        assert len(row["request_payload_sha256"]) == 64
        assert len(row["response_sha256"]) == 64

    pair_rows = [
        json.loads(line)
        for line in (output / "quality-pairs.jsonl").read_text().splitlines()
    ]
    assert all(row["exact_request_payload_hash_match"] for row in pair_rows)
    assert all(row["paired_quality_delta_near_minus_low"] == 0.0 for row in pair_rows)
    block_rows = [
        json.loads(line)
        for line in (output / "analysis-blocks.jsonl").read_text().splitlines()
    ]
    assert {row["analysis_block_index"] for row in block_rows} == {0, 1, 2, 3}
    assert all(row["predeclared_acceptance_pass"] for row in block_rows)

    second = asyncio.run(DirectSoakCampaign(config).run(executor))
    assert second["status"] == "complete"
    assert calls == first_calls
    assert second["request_rows"] == first["request_rows"]


def test_completion_nonce_preserves_exact_quality_pair_identity(tmp_path) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",), rate_rps=100.0)
    output = tmp_path / "soak"
    selected_cells = tuple(f"deepseek-v4-flash-0731:{shape}" for shape in SHAPES)
    config = replace(
        _fast_config(source, output),
        completion_attempt_label="closure-wave-1",
        selected_cells=selected_cells,
    )

    async def executor(model_id, task, max_output_tokens):
        return _fake_result(task)

    summary = asyncio.run(DirectSoakCampaign(config).run(executor))
    assert summary["scientifically_complete"] is True
    request_rows = [
        json.loads(line)
        for line in (output / "requests.jsonl").read_text().splitlines()
    ]
    pair_rows: dict[str, list[dict]] = {}
    for row in request_rows:
        pair_id = row.get("quality_pair_id")
        if pair_id:
            pair_rows.setdefault(pair_id, []).append(row)
    assert pair_rows
    assert all(
        len(rows) == 2
        and rows[0]["request_payload_sha256"] == rows[1]["request_payload_sha256"]
        for rows in pair_rows.values()
    )
    recorded_pairs = [
        json.loads(line)
        for line in (output / "quality-pairs.jsonl").read_text().splitlines()
    ]
    assert all(row["exact_request_payload_hash_match"] for row in recorded_pairs)
    assert all(row["predeclared_quality_acceptance_pass"] for row in recorded_pairs)
    reconciled = _reconcile_invalidated_soak_exposure(
        output,
        expected_selected_cells=selected_cells,
        expected_attempt_label="closure-wave-1",
        expected_prior_cost_usd=config.prior_cost_usd,
    )
    assert reconciled == pytest.approx(summary["conservative_exposure_usd"])


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 64},
        {"prompt_tokens": 64, "completion_tokens": 0, "total_tokens": 64},
    ],
)
def test_incomplete_usage_counter_retains_full_reservation(
    tmp_path, usage: dict[str, int]
) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",), rate_rps=100.0)
    campaign = DirectSoakCampaign(_fast_config(source, tmp_path / "soak"))
    cell = next(cell for cell in campaign.cell_plans if cell.shape == "short_short")
    task = campaign._task(cell, "paired_low_load", 0, 0, True)
    result = _fake_result(task)
    result.usage = usage
    item = campaign._schedule(cell, "paired_low_load")[0]
    base = campaign._base_request_row(
        cell=cell,
        phase="paired_low_load",
        phase_id="phase",
        request_id="request",
        task=task,
        item=item,
        started_at="2026-08-23T00:00:00+00:00",
        ended_at="2026-08-23T00:00:01+00:00",
        reserved_cost_usd=0.123,
        reserved_prompt_tokens=100,
        schedule_lag_seconds=0.0,
        observed_concurrency=1,
    )
    row = campaign._success_row(
        base=base,
        task=task,
        result=result,
        spec=next(spec for spec in MODEL_SPECS if spec.model_id == cell.model_id),
    )
    assert row["usage"] == usage
    assert row["input_usage_complete"] is True
    assert row["output_usage_complete"] is False
    assert row["usage_complete_for_settlement"] is False
    assert row["estimated_cost_usd"] is None
    assert row["accounted_cost_usd"] == 0.123
    assert row["timing"]["output_tokens_per_second"] is None


def test_prompt_only_success_censors_output_tpm_at_every_summary_level(
    tmp_path,
) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",), rate_rps=100.0)
    output = tmp_path / "soak"

    async def executor(model_id, task, max_output_tokens):
        result = _fake_result(task)
        result.usage = {
            "prompt_tokens": 64,
            "completion_tokens": 0,
            "total_tokens": 64,
        }
        return result

    summary = asyncio.run(
        DirectSoakCampaign(_fast_config(source, output)).run(executor)
    )
    assert summary["status"] == "complete"
    request_rows = [
        json.loads(line)
        for line in (output / "requests.jsonl").read_text().splitlines()
    ]
    assert all(row["input_usage_complete"] is True for row in request_rows)
    assert all(row["output_usage_complete"] is False for row in request_rows)
    assert all(row["usage_complete_for_settlement"] is False for row in request_rows)
    assert all(row["estimated_cost_usd"] is None for row in request_rows)
    assert all(
        row["accounted_cost_usd"] == row["worst_case_reserved_cost_usd"]
        for row in request_rows
    )
    phase_rows = [
        json.loads(line) for line in (output / "phases.jsonl").read_text().splitlines()
    ]
    assert all(
        row["output_usage_complete_for_all_successes"] is False
        and row["effective_output_tpm"] is None
        for row in phase_rows
    )
    block_rows = [
        json.loads(line)
        for line in (output / "analysis-blocks.jsonl").read_text().splitlines()
    ]
    assert all(
        row["output_usage_complete_for_all_successes"] is False
        and row["effective_output_tpm_per_predeclared_window"] is None
        and row["arrival_cohort_effective_output_tpm_including_drain"] is None
        for row in block_rows
    )
    assert all(
        cell["output_usage_complete_for_all_blocks"] is False
        and cell["effective_output_tpm_block_mean"] is None
        and cell["effective_output_tpm_block_mean_ci95_student_t"] is None
        for cell in summary["cells"]
    )


def test_transport_gate_is_execution_terminal_but_science_incomplete(tmp_path) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",), rate_rps=100.0)

    async def executor(model_id, task, max_output_tokens):
        raise RuntimeError("synthetic transport failure")

    summary = asyncio.run(
        DirectSoakCampaign(_fast_config(source, tmp_path / "soak")).run(executor)
    )
    assert summary["status"] == "execution_complete_science_incomplete"
    assert summary["execution_complete"] is True
    assert summary["scientifically_complete"] is False
    assert all(cell["execution_complete"] is True for cell in summary["cells"])
    assert all(cell["scientifically_complete"] is False for cell in summary["cells"])
    assert all(
        cell["status"] == "baseline_transport_gate_failed" for cell in summary["cells"]
    )


def test_plan_only_all_12_has_no_credential_dependency(tmp_path) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(source, model_ids=default_model_ids(), rate_rps=1.0)
    output = tmp_path / "plan"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run-digitalocean-direct-soak.py"
    )
    environment = dict(os.environ)
    for name in ("DIGITALOCEAN_API_KEY", "DIGITALOCEAN_TOKEN", "DO_API_TOKEN"):
        environment.pop(name, None)
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--aimd-dir",
            str(source),
            "--output-dir",
            str(output),
            "--prior-cost-usd",
            "25.33",
            "--duration-minutes",
            "180",
            "--drain-minutes",
            "5",
            "--plan-only",
        ],
        cwd=script.parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["credentials_loaded"] is False
    assert result["billable_requests_sent"] == 0
    assert result["preflight"]["target_cell_count"] == 48
    assert result["preflight"]["ready_cell_count"] == 48
    assert result["preflight"]["planned_request_count"] > 0
    assert result["preflight"]["passes"] is True


def test_block_acceptance_fails_on_exact_pair_quality_regression(tmp_path) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",), rate_rps=100.0)
    output = tmp_path / "soak"
    seen: dict[str, int] = {}

    async def executor(model_id, task, max_output_tokens):
        result = _fake_result(task)
        seen[task.task_id] = seen.get(task.task_id, 0) + 1
        # Every exact pair is first called at serial low load and then repeated
        # once as the tagged near-load arrival. Corrupt only the second copy.
        if seen[task.task_id] == 2:
            result.text = "definitely-wrong"
            result.tool_calls = []
        return result

    summary = asyncio.run(
        DirectSoakCampaign(_fast_config(source, output)).run(executor)
    )
    assert summary["scientifically_complete"] is True
    blocks = [
        json.loads(line)
        for line in (output / "analysis-blocks.jsonl").read_text().splitlines()
    ]
    assert blocks
    assert all(
        pair["exact_payload_hash_match"]
        for block in blocks
        for pair in block["quality_pairs"]
    )
    assert any(
        "paired_quality_regression_near_vs_low" in block["acceptance_reasons"]
        for block in blocks
    )
    assert any(block["predeclared_acceptance_pass"] is False for block in blocks)
    assert any(
        cell["two_minute_observed_acceptance_pass"] is False
        for cell in summary["cells"]
    )


def test_candidate_uses_realized_schedule_and_global_arrival_clock(tmp_path) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(
        source,
        model_ids=("deepseek-v4-flash-0731",),
        rate_rps=125.0,
        realized_rps=100.0,
    )
    campaign = DirectSoakCampaign(_fast_config(source, tmp_path / "soak"))
    cell = next(cell for cell in campaign.cell_plans if cell.shape == "short_short")
    assert cell.candidate_evidence["source_target_rate_rps"] == 125.0
    assert cell.candidate_rate_rps == 100.0
    schedule = campaign._schedule(cell, "two_minute_soak")
    assert [item["scheduled_offset_seconds"] for item in schedule] == [
        index / 100.0 for index in range(8)
    ]
    assert [item["analysis_block_index"] for item in schedule] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert cell.soak_block_request_counts == (2, 2, 2, 2)
    assert cell.soak_realized_schedule_rps == 100.0


def test_zero_to_zero_quality_pair_fails_predeclared_acceptance(tmp_path) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",), rate_rps=100.0)

    async def executor(model_id, task, max_output_tokens):
        result = _fake_result(task)
        result.text = "wrong"
        result.tool_calls = []
        return result

    output = tmp_path / "soak"
    summary = asyncio.run(
        DirectSoakCampaign(_fast_config(source, output)).run(executor)
    )
    blocks = [
        json.loads(line)
        for line in (output / "analysis-blocks.jsonl").read_text().splitlines()
    ]
    assert summary["scientifically_complete"] is True
    assert all(block["predeclared_acceptance_pass"] is False for block in blocks)
    assert all(
        "paired_low_load_quality_failure" in block["acceptance_reasons"]
        and "paired_near_load_quality_failure" in block["acceptance_reasons"]
        for block in blocks
    )


def test_mixed_quality_pairs_cover_all_five_task_families(tmp_path) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",), rate_rps=100.0)
    output = tmp_path / "soak"

    async def executor(model_id, task, max_output_tokens):
        return _fake_result(task)

    asyncio.run(DirectSoakCampaign(_fast_config(source, output)).run(executor))
    requests = [
        json.loads(line)
        for line in (output / "requests.jsonl").read_text().splitlines()
    ]
    families = {
        row["task_family"]
        for row in requests
        if row["shape"] == "mixed" and row["quality_pair_role"] == "low_load"
    }
    assert families == {
        "direct_short_exact",
        "direct_mixed_context4k",
        "direct_mixed_output512",
        "direct_mixed_structured",
        "direct_mixed_tool",
    }


def test_source_exposure_is_a_total_floor_not_an_additive_reset(tmp_path) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(
        source,
        model_ids=("deepseek-v4-flash-0731",),
        source_prior_cost_usd=25.33,
    )
    exposure = float(
        json.loads((source / "summary.json").read_text())["conservative_exposure_usd"]
    )
    with pytest.raises(SoakPreflightError, match="below the reconciled cumulative"):
        DirectSoakCampaign(
            _fast_config(source, tmp_path / "low", prior=exposure - 1e-6)
        )
    campaign = DirectSoakCampaign(
        _fast_config(source, tmp_path / "exact", prior=exposure)
    )
    assert campaign.budget.exposure_usd == pytest.approx(exposure)
    assert campaign.preflight["prior_cost_usd"] == pytest.approx(exposure)


def test_conditional_prior_basis_requires_explicit_plan_bound_acceptance(
    tmp_path, monkeypatch
) -> None:
    source_dir = tmp_path / "aimd"
    _aimd_fixture(source_dir, model_ids=("deepseek-v4-flash-0731",))
    decisions, source = load_aimd_candidates(source_dir, ("deepseek-v4-flash-0731",))
    source["reconciliation"] = {
        "schema_version": "test",
        "policy_version": "test",
        "receipt_sha256": "a" * 64,
        "performance_evidence_preserved": True,
        "prior_exposure_basis_status": (
            "conditional on the hash-bound runner's 4,096-token source default; "
            "the historical invocation receipt omitted the numeric CLI value"
        ),
        "prior_exposure_basis_is_conditional": True,
    }
    monkeypatch.setattr(
        "do_benchmark.direct_soak.load_aimd_candidates",
        lambda *args, **kwargs: (decisions, source),
    )
    rejected = DirectSoakCampaign(_fast_config(source_dir, tmp_path / "not-accepted"))
    assert rejected.preflight["complete_coverage_ready"] is True
    assert rejected.preflight["numeric_budget_passes"] is True
    assert rejected.preflight["budget_passes"] is False
    assert rejected.preflight["passes"] is False
    assert (
        rejected.preflight["conditional_prior_exposure_basis_explicitly_accepted"]
        is False
    )

    accepted = DirectSoakCampaign(
        replace(
            _fast_config(source_dir, tmp_path / "accepted"),
            accept_conditional_prior_exposure_basis=True,
        )
    )
    assert accepted.preflight["budget_passes"] is True
    assert accepted.preflight["passes"] is True
    assert accepted.plan_identity["accept_conditional_prior_exposure_basis"] is True


@pytest.mark.parametrize(
    ("artifact", "mutation", "message"),
    [
        (
            "epochs.jsonl",
            lambda row: {**row, "schema_version": "unknown_epoch_v999"},
            "unsupported AIMD epoch schema",
        ),
        (
            "manifest.json",
            lambda row: {
                **row,
                "model_specs": [
                    {
                        **row["model_specs"][0],
                        "input_usd_per_million": 999.0,
                    }
                ],
            },
            "model contract does not match",
        ),
    ],
)
def test_source_schema_and_model_contract_fail_closed(
    tmp_path, artifact, mutation, message
) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",))
    path = source / artifact
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0] = mutation(rows[0])
        _write_jsonl(path, rows)
    else:
        _write_json(path, mutation(json.loads(path.read_text())))
    with pytest.raises(SoakPreflightError, match=message):
        DirectSoakCampaign(_fast_config(source, tmp_path / "soak"))


def test_resume_rejects_corrupted_request_identity(tmp_path) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",), rate_rps=100.0)
    output = tmp_path / "soak"
    config = _fast_config(source, output)

    async def executor(model_id, task, max_output_tokens):
        return _fake_result(task)

    asyncio.run(DirectSoakCampaign(config).run(executor))
    path = output / "requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["request_payload_sha256"] = "0" * 64
    _write_jsonl(path, rows)
    with pytest.raises(SoakPreflightError, match="resume request identity mismatch"):
        DirectSoakCampaign(config)


def test_resume_rejects_tampered_terminal_cost_accounting(tmp_path) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",), rate_rps=100.0)
    output = tmp_path / "soak"
    config = _fast_config(source, output)

    async def executor(model_id, task, max_output_tokens):
        return _fake_result(task)

    asyncio.run(DirectSoakCampaign(config).run(executor))
    path = output / "requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["accounted_cost_usd"] = 0.0
    _write_jsonl(path, rows)
    with pytest.raises(
        SoakPreflightError, match="terminal cost accounting identity mismatch"
    ):
        DirectSoakCampaign(config)


def test_source_requires_complete_campaign_and_attempt_reservations(tmp_path) -> None:
    source = tmp_path / "incomplete"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",))
    summary = json.loads((source / "summary.json").read_text())
    summary["status"] = "incomplete"
    summary["all_models_complete"] = False
    _write_json(source / "summary.json", summary)
    with pytest.raises(SoakPreflightError, match="not scientifically complete"):
        load_aimd_candidates(source, ("deepseek-v4-flash-0731",))

    source = tmp_path / "missing-reservation"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",))
    reservations = [
        json.loads(line)
        for line in (source / "reservations.jsonl").read_text().splitlines()
    ]
    _write_jsonl(source / "reservations.jsonl", reservations[1:])
    with pytest.raises(SoakPreflightError, match="lacks its pre-send reservation"):
        load_aimd_candidates(source, ("deepseek-v4-flash-0731",))

    source = tmp_path / "tampered-source-cost"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",))
    requests = [
        json.loads(line)
        for line in (source / "requests.jsonl").read_text().splitlines()
    ]
    removed_cost = float(requests[0]["accounted_cost_usd"])
    requests[0]["accounted_cost_usd"] = 0.0
    requests[0]["estimated_cost_usd"] = 0.0
    _write_jsonl(source / "requests.jsonl", requests)
    summary = json.loads((source / "summary.json").read_text())
    summary["conservative_exposure_usd"] -= removed_cost
    _write_json(source / "summary.json", summary)
    with pytest.raises(SoakPreflightError, match="terminal cost accounting"):
        load_aimd_candidates(source, ("deepseek-v4-flash-0731",))


def test_recovery_acceptance_requires_deterministic_quality(tmp_path) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",), rate_rps=100.0)
    campaign = DirectSoakCampaign(_fast_config(source, tmp_path / "soak"))
    recovery_task_ids = {
        campaign._task_for_schedule(cell, "post_soak_recovery", item).task_id
        for cell in campaign.cell_plans
        if cell.status == "ready"
        for item in campaign._schedule(cell, "post_soak_recovery")
    }

    async def executor(model_id, task, max_output_tokens):
        result = _fake_result(task)
        if task.task_id in recovery_task_ids:
            result.text = "wrong"
            result.tool_calls = []
        return result

    summary = asyncio.run(campaign.run(executor))
    assert summary["scientifically_complete"] is True
    assert all(
        cell["post_soak_recovery_predeclared_pass"] is False
        for cell in summary["cells"]
    )
    assert all(
        "recovery_deterministic_quality_pass_rate_below_1.0"
        in cell["post_soak_recovery_acceptance_reasons"]
        for cell in summary["cells"]
    )


def test_output_directory_lease_prevents_duplicate_sends_and_stale_replay(
    tmp_path,
) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",), rate_rps=100.0)
    config = _fast_config(source, tmp_path / "soak")
    first = DirectSoakCampaign(config)
    second = DirectSoakCampaign(config)
    calls = 0

    async def executor(model_id, task, max_output_tokens):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.002)
        return _fake_result(task)

    async def race():
        return await asyncio.gather(
            first.run(executor), second.run(executor), return_exceptions=True
        )

    outcomes = asyncio.run(race())
    assert sum(isinstance(outcome, SoakPreflightError) for outcome in outcomes) == 1
    assert calls == 53
    # The object that lost the lease was constructed from an empty snapshot.
    # Its retry must reload under the lease and observe all terminal IDs.
    loser = first if isinstance(outcomes[0], SoakPreflightError) else second
    resumed = asyncio.run(loser.run(executor))
    assert resumed["scientifically_complete"] is True
    assert calls == 53


def test_hard_deadline_is_required_with_send_cutoff_and_bounds_run(tmp_path) -> None:
    source = tmp_path / "aimd"
    _aimd_fixture(source, model_ids=("deepseek-v4-flash-0731",), rate_rps=100.0)
    base = _fast_config(source, tmp_path / "invalid")
    with pytest.raises(ValueError, match="must be supplied together"):
        DirectSoakCampaign(
            replace(
                base,
                stop_launch_at=None,
                hard_campaign_deadline=datetime.now(timezone.utc)
                + timedelta(milliseconds=15),
            )
        )

    now = datetime.now(timezone.utc)
    bounded = replace(
        base,
        output_dir=tmp_path / "bounded",
        stop_launch_at=now + timedelta(milliseconds=5),
        hard_campaign_deadline=now + timedelta(milliseconds=15),
    )

    async def never_returns(model_id, task, max_output_tokens):
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    campaign = DirectSoakCampaign(bounded)
    started = time.perf_counter()
    summary = asyncio.run(campaign.run(never_returns))
    elapsed = time.perf_counter() - started
    assert elapsed < 0.08
    assert summary["scientifically_complete"] is False


def test_cli_requires_explicit_total_prior_cost(tmp_path) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run-digitalocean-direct-soak.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--aimd-dir",
            str(tmp_path / "unused"),
            "--output-dir",
            str(tmp_path / "unused-output"),
            "--duration-minutes",
            "1",
            "--drain-minutes",
            "1",
            "--plan-only",
        ],
        cwd=script.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--prior-cost-usd" in completed.stderr


def test_loader_accepts_artifacts_emitted_by_direct_aimd_runner(tmp_path) -> None:
    source = tmp_path / "real-direct-aimd"
    config = DirectConfig(
        output_dir=source,
        model_ids=("deepseek-v4-flash-0731",),
        epoch_seconds=0.001,
        concurrency_ceiling=4,
        initial_rps=0.1,
        additive_step_rps=0.1,
        maximum_rps=0.1,
        input_initial_rps=0.1,
        input_additive_step_rps=0.1,
        input_maximum_rps=0.1,
        output_initial_rps=0.1,
        output_additive_step_rps=0.1,
        output_maximum_rps=0.1,
        mixed_initial_rps=0.1,
        mixed_additive_step_rps=0.1,
        mixed_maximum_rps=0.1,
        rapid_bracket_epochs=1,
        heavy_rapid_bracket_epochs=1,
        additive_aimd_epochs=1,
        baseline_samples=1,
        input_tokens=32,
        long_output_words=2,
        short_max_output_tokens=16,
        long_max_output_tokens=32,
        mixed_max_output_tokens=32,
        request_timeout_seconds=1.0,
        max_cost_usd=200.0,
    )

    async def executor(model_id, task, max_output_tokens):
        return _fake_result(task)

    direct_summary = asyncio.run(DirectAIMDCampaign(config).run(executor))
    decisions, provenance = load_aimd_candidates(source, ("deepseek-v4-flash-0731",))
    assert direct_summary["all_models_complete"] is True
    assert len(decisions) == 4
    assert all(decision.status == "ready" for decision in decisions)
    assert all(
        decision.candidate.rate_rps <= decision.candidate.source_target_rate_rps
        for decision in decisions
        if decision.candidate is not None
    )
    assert all(
        decision.candidate is not None and decision.candidate.rate_rps == 0.1
        for decision in decisions
    )
    assert provenance["source_cumulative_exposure_usd"] == pytest.approx(
        direct_summary["conservative_exposure_usd"]
    )
