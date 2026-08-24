from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from do_benchmark.core import MODEL_BY_ID, StreamResult
from do_benchmark.direct_aimd import (
    BudgetLedger,
    ControllerState,
    DirectAIMDCampaign,
    DirectConfig,
    additive_aimd_transition,
    assess_epoch,
    default_model_ids,
    preflight_worst_case_cost,
    rapid_bracket_transition,
    sanitized_failure_row,
    sanitized_success_row,
)


def test_rapid_bracket_requires_two_consecutive_bad_epochs() -> None:
    state = ControllerState(offered_rps=4.0, best_healthy_rps=2.0)
    first = rapid_bracket_transition(
        state,
        healthy=False,
        additive_step_rps=1.0,
        maximum_rps=32.0,
    )
    assert first.offered_rps == 4.0
    assert first.consecutive_unhealthy == 1
    assert first.saturation_rps is None

    second = rapid_bracket_transition(
        first,
        healthy=False,
        additive_step_rps=1.0,
        maximum_rps=32.0,
    )
    assert second.offered_rps == 2.0
    assert second.saturation_rps == 4.0


def test_additive_aimd_is_additive_up_and_half_down() -> None:
    healthy = additive_aimd_transition(
        ControllerState(offered_rps=7.0, best_healthy_rps=5.0),
        healthy=True,
        additive_step_rps=2.0,
        maximum_rps=32.0,
    )
    assert healthy.offered_rps == 9.0
    assert healthy.best_healthy_rps == 7.0

    unhealthy = additive_aimd_transition(
        healthy,
        healthy=False,
        additive_step_rps=2.0,
        maximum_rps=32.0,
    )
    assert unhealthy.offered_rps == 4.5


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
    else:  # pragma: no cover - every direct task currently uses one of the above
        raise AssertionError(kind)
    return StreamResult(
        status_code=200,
        response_headers={
            "x-request-id": "provider-secret-request-id",
            "cf-ray": "provider-edge-id",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
            "retry-after": None,
        },
        text=text,
        reasoning_text="private reasoning",
        tool_calls=tool_calls,
        usage={"prompt_tokens": 128, "completion_tokens": 8, "total_tokens": 136},
        finish_reason="stop",
        request_seconds=0.02,
        headers_seconds=0.005,
        ttft_seconds=0.01,
        generation_seconds=0.01,
        stream_seconds=0.015,
        event_count=2,
        first_event_kind="content",
    )


def test_direct_campaign_is_sanitized_and_resume_deduplicated(tmp_path) -> None:
    calls = 0

    async def executor(model_id, task, max_output_tokens):
        nonlocal calls
        assert model_id == "deepseek-v4-flash-0731"
        assert max_output_tokens > 0
        assert task.messages[0]["content"].startswith("UNCACHED-")
        assert task.metadata["cache_intent"] == "uncached_randomized_prefix"
        calls += 1
        return _fake_result(task)

    config = DirectConfig(
        output_dir=tmp_path,
        model_ids=("deepseek-v4-flash-0731",),
        epoch_seconds=0.001,
        concurrency_ceiling=4,
        initial_rps=1.0,
        additive_step_rps=1.0,
        maximum_rps=2.0,
        rapid_bracket_epochs=1,
        additive_aimd_epochs=1,
        baseline_samples=1,
        heavy_rapid_bracket_epochs=1,
        output_initial_rps=1.0,
        output_additive_step_rps=1.0,
        output_maximum_rps=2.0,
        mixed_initial_rps=1.0,
        mixed_additive_step_rps=1.0,
        mixed_maximum_rps=2.0,
        input_tokens=128,
        long_output_words=8,
        short_max_output_tokens=16,
        long_max_output_tokens=32,
        mixed_max_output_tokens=32,
        max_cost_usd=200.0,
        stop_launch_at=datetime.now(timezone.utc) + timedelta(minutes=2),
    )
    first = asyncio.run(DirectAIMDCampaign(config).run(executor))
    assert first["request_rows"] > 0
    assert first["outlier_audit_rows"] == first["request_rows"]
    first_call_count = calls
    assert first_call_count > 0

    request_text = (tmp_path / "requests.jsonl").read_text(encoding="utf-8")
    assert "provider-secret-request-id" not in request_text
    assert "provider-edge-id" not in request_text
    assert "private reasoning" not in request_text
    assert "LOAD-OK" not in request_text
    for line in request_text.splitlines():
        row = json.loads(line)
        assert "response" not in row
        assert "response_headers" not in row
        assert len(row["request_payload_sha256"]) == 64
        assert len(row["response_sha256"]) == 64
        assert row["timing"]["event_count"] == 2
        assert row["timing"]["sse_chunk_span_output_tokens_per_second_proxy"] is None
    audit_rows = [
        json.loads(line)
        for line in (tmp_path / "outlier-audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(audit_rows) == first["request_rows"]
    assert all(row["request_id"] for row in audit_rows)

    second = asyncio.run(DirectAIMDCampaign(config).run(executor))
    assert calls == first_call_count
    assert second["request_rows"] == first["request_rows"]
    assert second["all_models_complete"] is True
    assert second["models"][0]["status"] == "complete_right_censored"
    assert all(
        shape["saturation_definition_met"] is False
        for shape in second["models"][0]["shapes"]
        if shape["status"] == "complete_right_censored"
    )


def test_failure_row_never_serializes_provider_body() -> None:
    from do_benchmark.core import ProviderHTTPError
    from do_benchmark.direct_aimd import make_task

    task = make_task(
        shape="short_short",
        ordinal=1,
        input_tokens=128,
        long_output_words=8,
    )
    error = ProviderHTTPError(429, "sensitive provider response body", "2")
    row = sanitized_failure_row(
        campaign_id="campaign",
        request_id="request",
        epoch_id="epoch",
        model_id="deepseek-v4-flash-0731",
        shape="short_short",
        phase="rapid_bracket",
        task=task,
        max_output_tokens=16,
        error=error,
        reserved_cost_usd=0.1,
        reserved_prompt_tokens=100,
        started_at="2026-08-23T00:00:00+00:00",
        ended_at="2026-08-23T00:00:01+00:00",
        elapsed_seconds=1.0,
        scheduled_offset_seconds=0.0,
        schedule_lag_seconds=0.0,
        concurrency_ceiling=8,
    )
    encoded = json.dumps(row)
    assert "sensitive provider response body" not in encoded
    assert row["http_status"] == 429
    assert row["retry_after_seconds"] == 2.0
    assert row["accounted_cost_usd"] == 0.1


def test_prompt_only_success_retains_full_reservation() -> None:
    from do_benchmark.direct_aimd import make_task

    task = make_task(
        shape="short_short",
        ordinal=1,
        input_tokens=128,
        long_output_words=8,
    )
    result = _fake_result(task)
    result.usage = {
        "prompt_tokens": 128,
        "completion_tokens": 0,
        "total_tokens": 128,
    }
    row = sanitized_success_row(
        campaign_id="campaign",
        request_id="request",
        epoch_id="epoch",
        model_id="deepseek-v4-flash-0731",
        shape="short_short",
        phase="rapid_bracket",
        task=task,
        max_output_tokens=16,
        result=result,
        spec=MODEL_BY_ID["deepseek-v4-flash-0731"],
        reserved_cost_usd=0.1,
        reserved_prompt_tokens=256,
        started_at="2026-08-23T00:00:00+00:00",
        ended_at="2026-08-23T00:00:01+00:00",
        scheduled_offset_seconds=0.0,
        schedule_lag_seconds=0.0,
        concurrency_ceiling=8,
    )
    assert row["usage_reported"] is True
    assert row["input_usage_complete"] is True
    assert row["output_usage_complete"] is False
    assert row["usage_complete_for_settlement"] is False
    assert row["estimated_cost_usd"] is None
    assert row["accounted_cost_usd"] == 0.1
    assert row["timing"]["output_tokens_per_second"] is None


def test_resume_ledger_repairs_legacy_prompt_only_under_settlement(tmp_path) -> None:
    ledger = BudgetLedger(
        path=tmp_path / "reservations.jsonl",
        max_cost_usd=200.0,
        prior_cost_usd=0.0,
        terminal_rows={
            "request": {
                "request_id": "request",
                "provider_send_attempted": True,
                "status": "success",
                "usage": {"prompt_tokens": 128, "completion_tokens": 0},
                "worst_case_reserved_cost_usd": 0.1,
                "accounted_cost_usd": 0.00001,
            }
        },
    )
    ledger.reservations["request"] = {
        "request_id": "request",
        "reserved_cost_usd": 0.1,
    }
    assert ledger.exposure_usd == 0.1


def test_default_all_model_preflight_fits_after_prior_cost(tmp_path) -> None:
    config = DirectConfig(
        output_dir=tmp_path,
        model_ids=default_model_ids(),
        prior_cost_usd=21.48,
    )
    preflight = preflight_worst_case_cost(config)
    assert preflight["passes"] is True
    assert preflight["total_worst_case_exposure_usd"] < 200.0
    assert preflight["remaining_margin_usd"] > 20.0

    too_little_room = DirectConfig(
        output_dir=tmp_path / "blocked",
        model_ids=default_model_ids(),
        prior_cost_usd=150.0,
    )
    try:
        DirectAIMDCampaign(too_little_room)
    except ValueError as error:
        assert "worst-case reservation exceeds" in str(error)
    else:  # pragma: no cover
        raise AssertionError("over-budget full schedule was not rejected")


def test_local_nonsend_invalidates_epoch_for_capacity() -> None:
    row = {
        "status": "skipped_cost_cap",
        "provider_send_attempted": False,
        "http_status": None,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "timing": {"request_seconds": 0.0, "ttft_seconds": None},
        "quality_score": 0.0,
        "load": {"schedule_lag_seconds": 0.0},
        "accounted_cost_usd": 0.0,
    }
    epoch = assess_epoch(
        campaign_id="campaign",
        epoch_id="epoch",
        model_id="deepseek-v4-flash-0731",
        shape="short_short",
        phase="rapid_bracket",
        offered_rps=1.0,
        epoch_seconds=5.0,
        scheduled_requests=1,
        rows=[row],
        elapsed_seconds=5.0,
        baseline_ttft_p95=None,
        baseline_latency_p95=None,
        baseline_quality_rate=None,
        max_observed_concurrency=0,
    )
    assert epoch["valid_for_capacity"] is False
    assert epoch["healthy"] is False
    assert "local_nonsend_or_unknown_outcome_present" in epoch["health_reasons"]


def test_http_402_is_a_billing_latch_not_provider_capacity() -> None:
    row = {
        "status": "error",
        "provider_send_attempted": True,
        "http_status": 402,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "timing": {"request_seconds": 0.1, "ttft_seconds": None},
        "quality_score": 0.0,
        "load": {"schedule_lag_seconds": 0.0},
        "accounted_cost_usd": 0.1,
    }
    epoch = assess_epoch(
        campaign_id="campaign",
        epoch_id="epoch",
        model_id="deepseek-v4-flash-0731",
        shape="short_short",
        phase="rapid_bracket",
        offered_rps=1.0,
        epoch_seconds=5.0,
        scheduled_requests=1,
        rows=[row],
        elapsed_seconds=5.0,
        baseline_ttft_p95=None,
        baseline_latency_p95=None,
        baseline_quality_rate=None,
        max_observed_concurrency=1,
    )
    assert epoch["valid_for_capacity"] is False
    assert epoch["healthy"] is False
    assert "billing_or_credit_latch_present" in epoch["health_reasons"]


def test_full_stream_has_hard_wall_clock_timeout(tmp_path) -> None:
    async def slow_executor(model_id, task, max_output_tokens):
        await asyncio.sleep(1.0)
        return _fake_result(task)

    config = DirectConfig(
        output_dir=tmp_path,
        model_ids=("deepseek-v4-flash-0731",),
        epoch_seconds=0.001,
        rapid_bracket_epochs=1,
        heavy_rapid_bracket_epochs=1,
        additive_aimd_epochs=1,
        baseline_samples=1,
        request_timeout_seconds=0.01,
        stop_launch_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    campaign = DirectAIMDCampaign(config)
    epoch = asyncio.run(
        campaign._run_epoch(
            slow_executor,
            model_id="deepseek-v4-flash-0731",
            shape="short_short",
            phase="serial_baseline",
            ordinal=0,
            offered_rps=1.0,
            scheduled_requests=1,
            serial=True,
            baseline=None,
        )
    )
    assert epoch["timeouts"] == 1
    row = next(iter(campaign.request_rows.values()))
    assert row["error_type"] == "TimeoutError"
    assert row["timing"]["request_seconds"] < 0.2


def test_reconstructed_epoch_elapsed_includes_schedule_lag(tmp_path) -> None:
    config = DirectConfig(
        output_dir=tmp_path,
        model_ids=("deepseek-v4-flash-0731",),
        epoch_seconds=5.0,
        rapid_bracket_epochs=1,
        heavy_rapid_bracket_epochs=1,
        additive_aimd_epochs=1,
    )
    first = DirectAIMDCampaign(config)
    epoch_id = first._epoch_id(
        model_id="deepseek-v4-flash-0731",
        shape="short_short",
        phase="rapid_bracket",
        ordinal=0,
        offered_rps=0.4,
    )
    rows = []
    for index, (offset, lag) in enumerate(((0.0, 1.0), (2.5, 3.0))):
        rows.append(
            {
                "schema_version": "do_direct_request_v1",
                "campaign_id": first.campaign_id,
                "request_id": first._request_id(epoch_id, index),
                "epoch_id": epoch_id,
                "model_id": "deepseek-v4-flash-0731",
                "shape": "short_short",
                "phase": "rapid_bracket",
                "status": "success",
                "provider_send_attempted": True,
                "http_status": 200,
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                "timing": {"request_seconds": 2.0, "ttft_seconds": 1.0},
                "quality_score": 1.0,
                "accounted_cost_usd": 0.0,
                "load": {
                    "scheduled_offset_seconds": offset,
                    "schedule_lag_seconds": lag,
                },
            }
        )
    (tmp_path / "requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    resumed = DirectAIMDCampaign(config)

    async def forbidden_executor(model_id, task, max_output_tokens):  # pragma: no cover
        raise AssertionError("terminal request IDs must not be resent")

    epoch = asyncio.run(
        resumed._run_epoch(
            forbidden_executor,
            model_id="deepseek-v4-flash-0731",
            shape="short_short",
            phase="rapid_bracket",
            ordinal=0,
            offered_rps=0.4,
            scheduled_requests=2,
            serial=False,
            baseline=None,
        )
    )
    assert epoch["reconstructed_from_terminal_request_rows"] is True
    assert epoch["elapsed_seconds_including_drain"] == 7.5
    assert epoch["valid_for_capacity"] is True


def test_tested_overload_is_strictly_above_candidate(tmp_path) -> None:
    async def executor(model_id, task, max_output_tokens):
        return _fake_result(task)

    config = DirectConfig(
        output_dir=tmp_path,
        model_ids=("deepseek-v4-flash-0731",),
        epoch_seconds=0.001,
        initial_rps=1.0,
        additive_step_rps=1.0,
        maximum_rps=8.0,
        rapid_bracket_epochs=1,
        heavy_rapid_bracket_epochs=1,
        additive_aimd_epochs=1,
        baseline_samples=1,
        input_tokens=128,
        long_output_words=8,
        short_max_output_tokens=16,
        long_max_output_tokens=32,
        mixed_max_output_tokens=32,
    )
    result = asyncio.run(
        DirectAIMDCampaign(config)._run_aimd_shape(
            executor,
            model_id="deepseek-v4-flash-0731",
            shape="short_short",
        )
    )
    assert result["status"] == "complete_right_censored"
    assert result["overload_tested"] is True
    assert result["overload_offered_rps"] > result["candidate_confirmed_healthy_rps"]
