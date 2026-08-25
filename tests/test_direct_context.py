from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import do_benchmark.direct_context as direct_context
from do_benchmark.core import (
    DIGITALOCEAN_HOSTED_MODEL_IDS,
    MODEL_SPECS,
    ProviderHTTPError,
    StreamResult,
)
from do_benchmark.direct_context import (
    AccountQuotaGovernor,
    CONTEXT_PERCENTAGES,
    ContextPreflightError,
    ContextConfig,
    DirectContextCampaign,
    OutputDirectoryLease,
    build_context_probes,
    build_retrieval_task,
    classify_failure,
    conservative_reservation,
)
from do_benchmark.direct_report import load_breadth_directory, reconcile_request_rows


def _fake_result(task, *, prompt_tokens: int | None = None) -> StreamResult:
    target = int(task.metadata.get("estimated_target_prompt_tokens") or 64)
    prompt_tokens = prompt_tokens or target
    return StreamResult(
        status_code=200,
        response_headers={
            "x-request-id": "private-provider-request-id",
            "cf-ray": "private-edge-id",
            "x-ratelimit-limit-requests": "1000000000",
            "x-ratelimit-remaining-requests": "999999999",
            "x-ratelimit-reset-requests": "0",
            "x-ratelimit-limit-tokens-per-minute": "1000000000000",
            "x-ratelimit-remaining-tokens-per-minute": "999999999900",
            "x-ratelimit-reset-tokens-per-minute": "0",
            "x-ratelimit-limit-tokens-per-day": "1000000000",
            "x-ratelimit-remaining-tokens-per-day": "999999900",
            "x-ratelimit-reset-tokens-per-day": "0",
        },
        text=str(task.expected["value"]),
        reasoning_text="private reasoning trace",
        tool_calls=[],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 4,
            "total_tokens": prompt_tokens + 4,
        },
        finish_reason="stop",
        request_seconds=0.01,
        headers_seconds=0.002,
        ttft_seconds=0.004,
        generation_seconds=0.006,
        stream_seconds=0.008,
        event_count=2,
        first_event_kind="content",
    )


def _config(tmp_path, **overrides) -> ContextConfig:
    values = {
        "output_dir": tmp_path,
        "model_ids": ("openai-gpt-oss-120b",),
        "per_model_concurrency": 1,
        "model_parallelism": 12,
        "global_concurrency": 12,
        "fallback_account_rpm": 1_000_000_000.0,
        "fallback_account_tpm": 1_000_000_000_000.0,
        "quota_utilization_fraction": 1.0,
        "request_timeout_seconds": 1.0,
        "max_cost_usd": 200.0,
        "max_payload_bytes": 16_384,
        "max_bisection_rounds": 0,
    }
    values.update(overrides)
    return ContextConfig(**values)


def test_fixed_design_covers_all_hosted_11_models_and_required_anchors() -> None:
    model_ids = DIGITALOCEAN_HOSTED_MODEL_IDS
    probes = build_context_probes(model_ids)
    assert len(model_ids) == 11
    assert len(probes) == 15 * 11
    for model_id in model_ids:
        rows = [probe for probe in probes if probe.model_id == model_id]
        observed_percentages = {
            percentage for probe in rows for percentage in probe.anchor_percentages
        }
        tags = {tag for probe in rows for tag in probe.coverage_tags}
        assert observed_percentages == set(CONTEXT_PERCENTAGES)
        prefix = (
            "undocumented_probe_anchor"
            if model_id == "kimi-k3"
            else "advertised_context"
        )
        assert {
            f"{prefix}_prompt_estimate_lower",
            f"{prefix}_prompt_estimate_center",
            f"{prefix}_prompt_estimate_upper",
            f"{prefix}_combined_estimate_lower",
            f"{prefix}_combined_estimate_center",
            f"{prefix}_combined_estimate_upper",
        }.issubset(tags)


def test_all_model_context_and_price_contracts_match_frozen_inventory() -> None:
    freeze = json.loads(direct_context.ENDPOINT_FREEZE_PATH.read_text(encoding="utf-8"))
    frozen = {row["model_id"]: row for row in freeze["endpoints"]}
    assert set(frozen).issuperset(spec.model_id for spec in MODEL_SPECS)
    for spec in MODEL_SPECS:
        row = frozen[spec.model_id]
        if spec.model_id == "kimi-k3":
            assert row["context_window"] is None
            assert spec.context_window == 65_536
        else:
            assert spec.context_window == int(row["context_window"])
        assert spec.input_usd_per_million == pytest.approx(
            float(row["input_usd_per_million"])
        )
        assert spec.output_usd_per_million == pytest.approx(
            float(row["output_usd_per_million"])
        )
    qwen = [
        probe
        for probe in build_context_probes(("qwen3.8-max",))
        if 0.99 in probe.anchor_percentages
    ]
    assert len(qwen) == 1
    assert qwen[0].estimated_target_prompt_tokens == 990_000


def test_kimi_anchor_is_never_labelled_documented() -> None:
    probes = build_context_probes(("kimi-k3",))
    rows = [probe.sanitized_plan_row() for probe in probes]
    assert all(
        row["context_window_anchor_source"] == "undocumented_probe_anchor"
        for row in rows
    )
    assert not any(
        str(tag).startswith("documented_")
        for row in rows
        for tag in row["coverage_tags"]
    )
    non_kimi = build_context_probes(("deepseek-v4-flash-0731",))[0]
    assert non_kimi.sanitized_plan_row()["context_window_anchor_source"] == (
        "advertised_official_documentation"
    )


def test_reduced_plan_knobs_preserve_endpoint_selection() -> None:
    probes = build_context_probes(
        ("deepseek-v4-flash-0731", "kimi-k3"),
        percentages=(0.5,),
        include_prompt_boundary_triplet=False,
        include_combined_boundary_triplet=False,
    )
    assert len(probes) == 2
    assert {probe.model_id for probe in probes} == {
        "deepseek-v4-flash-0731",
        "kimi-k3",
    }


def test_near_million_token_probe_is_generated_under_exact_byte_cap() -> None:
    probe = next(
        item
        for item in build_context_probes(("deepseek-v4-flash-0731",))
        if 0.99 in item.anchor_percentages
    )
    task, planning = build_retrieval_task(
        probe, chars_per_token=4.0, max_payload_bytes=8 * 1024 * 1024
    )
    assert probe.estimated_target_prompt_tokens > 1_000_000
    assert planning["request_payload_bytes"] <= 8 * 1024 * 1024
    assert len(task.messages[0]["content"]) > 3_500_000


def test_campaign_sanitizes_content_and_deduplicates_resume(tmp_path) -> None:
    calls = 0

    async def executor(model_id, task, max_output_tokens):
        nonlocal calls
        assert model_id == "openai-gpt-oss-120b"
        assert max_output_tokens >= 1
        calls += 1
        return _fake_result(task)

    config = _config(tmp_path)
    first = asyncio.run(DirectContextCampaign(config).run(executor))
    assert first["execution_complete"] is True
    assert first["scientifically_complete"] is True
    assert calls == 15
    text = (tmp_path / "requests.jsonl").read_text(encoding="utf-8")
    assert "private-provider-request-id" not in text
    assert "private-edge-id" not in text
    assert "private reasoning trace" not in text
    assert "NEEDLE-" not in text
    assert "cobalt" not in text
    for line in text.splitlines():
        row = json.loads(line)
        assert "response" not in row
        assert "response_headers" not in row
        assert "error" not in row
        assert len(row["request_payload_sha256"]) == 64
        assert len(row["response_sha256"]) == 64
        assert row["actual_prompt_tokens_x_axis"] > 0
        assert "estimated_target_prompt_tokens" in row
        assert "target_prompt_tokens" not in row
        assert row["planning_within_tolerance"] is True
        assert row["latency_measurement_scope"] == "concurrent_context_probe"
        assert row["latency_comparison_eligible"] is False
        assert row["timing"] == {}
        assert row["concurrent_timing_diagnostic"]["request_seconds"] == 0.01

    second = asyncio.run(DirectContextCampaign(config).run(executor))
    assert calls == 15
    assert second["terminal_rows"] == 15


def test_generic_4xx_is_inconclusive_and_body_is_not_serialized(tmp_path) -> None:
    async def executor(model_id, task, max_output_tokens):
        raise ProviderHTTPError(400, "sensitive provider response body")

    summary = asyncio.run(DirectContextCampaign(_config(tmp_path)).run(executor))
    model = summary["models"]["openai-gpt-oss-120b"]
    assert model["conclusive_rows"] == 0
    assert model["outcomes"] == {"other_4xx_inconclusive": 15}
    assert model["execution_complete"] is True
    assert model["scientifically_complete"] is False
    text = (tmp_path / "requests.jsonl").read_text(encoding="utf-8")
    assert "sensitive provider response body" not in text


def test_only_allowlisted_context_reason_makes_400_or_413_conclusive() -> None:
    classification, conclusive, category, fingerprint = classify_failure(
        ProviderHTTPError(400, "maximum context length exceeded: private detail")
    )
    assert classification == "explicit_context_limit_rejection"
    assert conclusive is True
    assert category == "allowlisted_context_or_token_limit_reason"
    assert len(fingerprint) == 64

    classification, conclusive, category, _ = classify_failure(
        ProviderHTTPError(400, "input is too long")
    )
    assert classification == "other_4xx_inconclusive"
    assert conclusive is False
    assert category == "generic_client_rejection"

    classification, conclusive, category, fingerprint = classify_failure(
        ProviderHTTPError(413, "private payload body")
    )
    assert classification == "http_413_payload_size_inconclusive"
    assert conclusive is False
    assert category == "http_413_without_context_limit_reason"
    assert len(fingerprint) == 64

    classification, conclusive, category, _ = classify_failure(
        ProviderHTTPError(413, "input exceeds request-body limit")
    )
    assert classification == "http_413_payload_size_inconclusive"
    assert conclusive is False
    assert category == "http_413_without_context_limit_reason"

    classification, conclusive, category, fingerprint = classify_failure(
        ProviderHTTPError(413, "maximum context length exceeded")
    )
    assert classification == "explicit_context_limit_rejection"
    assert conclusive is True
    assert category == "http_413_allowlisted_context_or_token_limit_reason"
    assert len(fingerprint) == 64


def test_hard_wall_clock_timeout_is_terminal_but_inconclusive(tmp_path) -> None:
    async def executor(model_id, task, max_output_tokens):
        await asyncio.sleep(1.0)
        return _fake_result(task)

    config = _config(tmp_path, request_timeout_seconds=0.01)
    summary = asyncio.run(DirectContextCampaign(config).run(executor))
    model = summary["models"]["openai-gpt-oss-120b"]
    assert model["execution_complete"] is True
    assert model["scientifically_complete"] is False
    assert model["conclusive_rows"] == 0
    assert model["outcomes"] == {"timed_out_inconclusive": 15}


def test_partial_usage_retains_reservation_and_cannot_anchor_refinement(
    tmp_path,
) -> None:
    async def executor(model_id, task, max_output_tokens):
        result = _fake_result(task)
        result.usage = {"completion_tokens": 4, "total_tokens": 4}
        return result

    summary = asyncio.run(
        DirectContextCampaign(_config(tmp_path, max_bisection_rounds=2)).run(executor)
    )
    assert summary["execution_complete"] is True
    assert summary["scientifically_complete"] is False
    assert summary["total_plan_rows"] == 15
    model = summary["models"]["openai-gpt-oss-120b"]
    assert model["highest_accepted_actual_prompt_tokens"] is None
    assert model["boundary_observation"]["classification"] == (
        "unobserved_or_inconclusive"
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["usage_complete_for_settlement"] is False for row in rows)
    assert all(row["actual_prompt_tokens_x_axis"] is None for row in rows)
    assert all(row["estimated_cost_usd"] is None for row in rows)
    assert all(
        row["accounted_cost_usd"] == row["worst_case_reserved_cost_usd"] for row in rows
    )


def test_zero_prompt_usage_retains_reservation(tmp_path) -> None:
    async def executor(model_id, task, max_output_tokens):
        result = _fake_result(task)
        result.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 4,
            "total_tokens": 4,
        }
        return result

    summary = asyncio.run(DirectContextCampaign(_config(tmp_path)).run(executor))
    assert summary["scientifically_complete"] is False
    rows = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["usage_complete_for_settlement"] is False for row in rows)
    assert all(
        row["accounted_cost_usd"] == row["worst_case_reserved_cost_usd"] for row in rows
    )


def test_zero_completion_usage_retains_reservation(tmp_path) -> None:
    async def executor(model_id, task, max_output_tokens):
        result = _fake_result(task)
        result.usage = {
            "prompt_tokens": 64,
            "completion_tokens": 0,
            "total_tokens": 64,
        }
        return result

    asyncio.run(DirectContextCampaign(_config(tmp_path)).run(executor))
    rows = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows
    assert all(row["usage_complete_for_settlement"] is False for row in rows)
    assert all(row["estimated_cost_usd"] is None for row in rows)
    assert all(
        row["accounted_cost_usd"] == row["worst_case_reserved_cost_usd"] for row in rows
    )


def test_model_lanes_overlap_but_each_model_chain_is_sequential(tmp_path) -> None:
    active_total = 0
    maximum_active_total = 0
    active_by_model: dict[str, int] = {}
    maximum_by_model: dict[str, int] = {}

    async def executor(model_id, task, max_output_tokens):
        nonlocal active_total, maximum_active_total
        active_total += 1
        active_by_model[model_id] = active_by_model.get(model_id, 0) + 1
        maximum_active_total = max(maximum_active_total, active_total)
        maximum_by_model[model_id] = max(
            maximum_by_model.get(model_id, 0), active_by_model[model_id]
        )
        await asyncio.sleep(0.001)
        active_by_model[model_id] -= 1
        active_total -= 1
        return _fake_result(task)

    config = _config(
        tmp_path,
        model_ids=("openai-gpt-oss-120b", "gemma-4-31B-it"),
    )
    asyncio.run(DirectContextCampaign(config).run(executor))
    assert maximum_active_total >= 2
    assert maximum_by_model == {
        "openai-gpt-oss-120b": 1,
        "gemma-4-31B-it": 1,
    }


def test_transition_bracket_adds_deterministic_bisection_rows(tmp_path) -> None:
    async def executor(model_id, task, max_output_tokens):
        target = int(task.metadata["estimated_target_prompt_tokens"])
        threshold = 100_000 if max_output_tokens == 32 else 110_000
        if target >= threshold:
            raise ProviderHTTPError(400, "maximum context length exceeded")
        return _fake_result(task)

    config = _config(tmp_path, max_bisection_rounds=2)
    first = asyncio.run(DirectContextCampaign(config).run(executor))
    assert first["total_plan_rows"] == 19
    boundary = first["models"]["openai-gpt-oss-120b"]["boundary_observation"]
    assert boundary["classification"] == "interval_censored_mixed_coordinates"
    assert boundary["accepted_lower_actual_prompt_tokens"] is not None
    assert boundary["rejected_upper_actual_prompt_tokens"] is None
    assert boundary["exact_boundary_identified"] is False
    combined = first["models"]["openai-gpt-oss-120b"]["combined_boundary_observation"]
    assert combined["classification"] == "interval_censored_mixed_coordinates"
    assert combined["requested_max_output_tokens"] == 4_096
    plan = [
        json.loads(line) for line in (tmp_path / "plan.jsonl").read_text().splitlines()
    ]
    adaptive = [
        row for row in plan if "observed_transition_bisection" in row["coverage_tags"]
    ]
    assert len(adaptive) == 4
    assert (
        sum(
            "observed_prompt_transition_bisection" in row["coverage_tags"]
            for row in adaptive
        )
        == 2
    )
    assert (
        sum(
            "observed_combined_transition_bisection" in row["coverage_tags"]
            for row in adaptive
        )
        == 2
    )
    ids = [row["request_id"] for row in adaptive]
    second = asyncio.run(DirectContextCampaign(config).run(executor))
    assert second["total_plan_rows"] == 19
    plan_again = [
        json.loads(line) for line in (tmp_path / "plan.jsonl").read_text().splitlines()
    ]
    assert [
        row["request_id"] for row in plan_again if row.get("adaptive_round") is not None
    ] == ids


def test_cost_preflight_fails_before_any_provider_send(tmp_path) -> None:
    with pytest.raises(ValueError, match="cannot fit"):
        DirectContextCampaign(_config(tmp_path, max_cost_usd=0.000001))


def test_preflight_sums_worst_simultaneously_runnable_model_reservations(
    tmp_path,
) -> None:
    config = ContextConfig(
        output_dir=tmp_path,
        model_ids=DIGITALOCEAN_HOSTED_MODEL_IDS,
        prior_cost_usd=21.438454073,
        max_cost_usd=200.0,
        max_bisection_rounds=8,
    )
    campaign = DirectContextCampaign(config)
    assert campaign.all_requests_settlement_ceiling_usd > (
        config.max_cost_usd - config.prior_cost_usd
    )
    assert campaign.max_inflight_reservation_usd < (
        config.max_cost_usd - config.prior_cost_usd
    )
    assert campaign.max_inflight_reservation_usd > 20.0


@pytest.mark.parametrize("contract_change", ["price", "context"])
def test_model_contract_change_invalidates_resume(
    tmp_path, monkeypatch, contract_change: str
) -> None:
    config = _config(tmp_path)
    original = direct_context.MODEL_BY_ID["openai-gpt-oss-120b"]
    original_campaign = DirectContextCampaign(config)
    original_identity = config.identity_payload()
    original_request_ids = {
        probe.request_id for probe in original_campaign.fixed_probes
    }
    replacement = (
        replace(original, input_usd_per_million=original.input_usd_per_million + 0.01)
        if contract_change == "price"
        else replace(original, context_window=int(original.context_window or 0) + 1_000)
    )
    monkeypatch.setitem(direct_context.MODEL_BY_ID, "openai-gpt-oss-120b", replacement)
    with pytest.raises(ValueError, match="frozen endpoint contract mismatch"):
        DirectContextCampaign(config)
    assert config.identity_payload() != original_identity
    changed = build_context_probes(("openai-gpt-oss-120b",))
    assert {probe.request_id for probe in changed} != original_request_ids


def test_plan_and_requests_reconcile_under_strict_hash_contract(tmp_path) -> None:
    async def executor(model_id, task, max_output_tokens):
        return _fake_result(task)

    asyncio.run(DirectContextCampaign(_config(tmp_path)).run(executor))
    plans, requests = load_breadth_directory(tmp_path)
    matched, orphaned = reconcile_request_rows(plans, requests, [])
    assert not orphaned
    assert len(matched) == len(plans) == 15
    assert {row["reconciliation_policy"] for row in matched} == {
        "strict_plan_contract_hashes"
    }


def test_process_lease_blocks_overlapping_context_runner(tmp_path) -> None:
    calls = 0

    async def executor(model_id, task, max_output_tokens):
        nonlocal calls
        calls += 1
        return _fake_result(task)

    campaign = DirectContextCampaign(_config(tmp_path))
    with OutputDirectoryLease(campaign.execution_lease_path):
        with pytest.raises(ContextPreflightError, match="another process holds"):
            asyncio.run(campaign.run(executor))
    assert calls == 0


def test_unknown_prior_reservation_is_never_replayed(tmp_path) -> None:
    config = _config(tmp_path)
    campaign = DirectContextCampaign(config)
    reserved_probe = campaign.fixed_probes[0]
    reserved_cost, reserved_tokens = conservative_reservation(reserved_probe, config)
    reservation = {
        "schema_version": direct_context.RESERVATION_SCHEMA,
        "campaign_id": campaign.campaign_id,
        "request_id": reserved_probe.request_id,
        "model_id": reserved_probe.model_id,
        "reserved_at": "2026-08-23T00:00:00+00:00",
        "reserved_cost_usd": reserved_cost,
        "reserved_prompt_tokens": reserved_tokens,
        "requested_max_output_tokens": reserved_probe.requested_max_output_tokens,
    }
    campaign.reservations_path.write_text(
        json.dumps(reservation, sort_keys=True) + "\n", encoding="utf-8"
    )
    sent_ids: list[str] = []

    async def executor(model_id, task, max_output_tokens):
        sent_ids.append(task.task_id)
        return _fake_result(task)

    summary = asyncio.run(DirectContextCampaign(config).run(executor))
    assert reserved_probe.probe_id not in sent_ids
    assert len(sent_ids) == 14
    assert (
        summary["models"][reserved_probe.model_id]["outcomes"][
            "unknown_prior_reservation"
        ]
        == 1
    )
    assert summary["conservative_exposure_usd"] >= reserved_cost


def test_deadline_and_http_402_latch_stop_new_provider_sends(tmp_path) -> None:
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    deadline_calls = 0

    async def deadline_executor(model_id, task, max_output_tokens):
        nonlocal deadline_calls
        deadline_calls += 1
        return _fake_result(task)

    deadline_summary = asyncio.run(
        DirectContextCampaign(_config(tmp_path / "deadline", stop_launch_at=past)).run(
            deadline_executor
        )
    )
    assert deadline_calls == 0
    assert deadline_summary["models"]["openai-gpt-oss-120b"]["outcomes"] == {
        "skipped_deadline": 15
    }

    billing_calls = 0

    async def billing_executor(model_id, task, max_output_tokens):
        nonlocal billing_calls
        billing_calls += 1
        raise ProviderHTTPError(402, "private billing detail")

    billing_summary = asyncio.run(
        DirectContextCampaign(
            _config(tmp_path / "billing", per_model_concurrency=1)
        ).run(billing_executor)
    )
    outcomes = billing_summary["models"]["openai-gpt-oss-120b"]["outcomes"]
    assert billing_calls == 1
    assert outcomes == {"account_blocked_402": 1, "skipped_http_402_latch": 14}
    assert billing_summary["http_402_latched"] is True


def test_plan_preflight_reports_honest_cost_and_timeout_bounds(tmp_path) -> None:
    config = ContextConfig(
        output_dir=tmp_path,
        model_ids=DIGITALOCEAN_HOSTED_MODEL_IDS,
        per_model_concurrency=1,
        request_timeout_seconds=180.0,
    )
    campaign = DirectContextCampaign(config)
    assert campaign.full_plan_guaranteed_to_fit_budget is False
    assert campaign.parallel_timeout_only_projection_seconds == 7_380
    assert campaign.first_calibration_header_projection_seconds == 5_760
    assert campaign.serialized_configured_timeout_sum_seconds == 61_380
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["maximum_total_requests"] == 341
    assert manifest["full_plan_guaranteed_to_fit_budget"] is False
    assert manifest["parallel_timeout_only_projection_seconds"] == 7_380
    assert manifest["first_calibration_header_timeout_projection_seconds"] == 5_760
    assert manifest["serialized_configured_timeout_sum_seconds"] == 61_380
    assert "not a wall-clock hard bound" in manifest["timeout_bound_contract"]
    assert manifest["quota_governor_wait_upper_bound_seconds"] is None


class _FakeClock:
    def __init__(self) -> None:
        self.monotonic_value = 100.0
        self.epoch_value = 1_000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.monotonic_value

    def epoch(self) -> float:
        return self.epoch_value

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.monotonic_value += delay
        self.epoch_value += delay


def test_account_governor_uses_quota_headers_and_global_429_aimd() -> None:
    config = _config(
        Path("unused"),
        fallback_account_rpm=120.0,
        fallback_account_tpm=500_000.0,
        quota_utilization_fraction=0.8,
    )
    clock = _FakeClock()
    governor = AccountQuotaGovernor(
        config,
        monotonic=clock.monotonic,
        epoch_time=clock.epoch,
        sleeper=clock.sleep,
    )

    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        signals = await governor.observe(
            headers={
                "x-ratelimit-limit-requests": "600",
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-reset-requests": "1010",
                "x-ratelimit-limit-tokens-per-minute": "2000000",
                "x-ratelimit-remaining-tokens-per-minute": "0",
                "x-ratelimit-reset-tokens-per-minute": "1008",
                "x-ratelimit-limit-tokens-per-day": "9000000",
                "x-ratelimit-remaining-tokens-per-day": "8000000",
                "x-ratelimit-reset-tokens-per-day": "0",
                "retry-after": "5",
            },
            http_status=429,
        )
        admission = await governor.acquire(
            estimated_tokens=1_000,
            stop_launch_at=datetime.fromtimestamp(2_000, timezone.utc),
        )
        assert admission is not None
        return signals, admission

    signals, admission = asyncio.run(scenario())
    assert governor.bootstrap_ready is True
    assert signals["rate_limit_reset_requests_epoch_seconds"] == 1010.0
    assert clock.sleeps == [10.0]
    assert admission["effective_request_rate_per_minute"] == pytest.approx(240.0)
    assert admission["effective_token_rate_per_minute"] == pytest.approx(800_000.0)
    assert admission["congestion_factor"] == 0.5
    snapshot = governor.snapshot()
    assert snapshot["http_429_observations"] == 1
    assert snapshot["quota_scope"] == "per_account"


def test_large_probe_is_paced_not_rejected_by_fallback_tpm() -> None:
    config = _config(
        Path("unused-large"),
        fallback_account_rpm=120.0,
        fallback_account_tpm=500_000.0,
        quota_utilization_fraction=0.8,
    )
    clock = _FakeClock()
    governor = AccountQuotaGovernor(
        config,
        monotonic=clock.monotonic,
        epoch_time=clock.epoch,
        sleeper=clock.sleep,
    )

    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        first = await governor.acquire(
            estimated_tokens=1_000_000,
            stop_launch_at=datetime.fromtimestamp(5_000, timezone.utc),
        )
        second = await governor.acquire(
            estimated_tokens=1,
            stop_launch_at=datetime.fromtimestamp(5_000, timezone.utc),
        )
        assert first is not None and second is not None
        return first, second

    first, second = asyncio.run(scenario())
    assert first["open_loop_wait_seconds"] == 0.0
    assert second["open_loop_wait_seconds"] == pytest.approx(150.0)
    assert clock.sleeps == [150.0]


def test_scheduler_contract_is_hash_bound_and_latency_scope_is_explicit(
    tmp_path,
) -> None:
    first = DirectContextCampaign(
        _config(tmp_path / "first", model_parallelism=2, global_concurrency=2)
    )
    second = DirectContextCampaign(
        _config(tmp_path / "second", model_parallelism=1, global_concurrency=1)
    )
    assert first.campaign_plan_sha256 != second.campaign_plan_sha256
    assert first.latency_measurement_scope == "concurrent_context_probe"
    assert second.latency_measurement_scope == "isolated_single_inflight"
    manifest = json.loads((tmp_path / "first" / "manifest.json").read_text())
    assert manifest["account_quota_governor"]["quota_scope"] == "per_account"
    assert manifest["model_parallelism"] == 2
    assert manifest["global_concurrency"] == 2


def test_provider_send_cutoff_is_hash_bound(tmp_path) -> None:
    first_deadline = datetime(2026, 8, 23, 22, 30, tzinfo=timezone.utc)
    second_deadline = datetime(2026, 8, 23, 22, 31, tzinfo=timezone.utc)
    first_config = _config(tmp_path / "first", stop_launch_at=first_deadline)
    second_config = _config(tmp_path / "second", stop_launch_at=second_deadline)
    first = DirectContextCampaign(first_config)
    second = DirectContextCampaign(second_config)

    assert first_config.identity_payload()["stop_launch_at"] == (
        "2026-08-23T22:30:00+00:00"
    )
    assert first.campaign_plan_sha256 != second.campaign_plan_sha256
    with pytest.raises(ContextPreflightError, match="plan fixed prefix"):
        DirectContextCampaign(
            _config(tmp_path / "first", stop_launch_at=second_deadline)
        )


def test_headerless_429_enforces_global_reduced_rate_cooldown() -> None:
    config = _config(
        Path("unused-headerless-429"),
        fallback_account_rpm=120.0,
        fallback_account_tpm=500_000.0,
        quota_utilization_fraction=0.8,
    )
    clock = _FakeClock()
    governor = AccountQuotaGovernor(
        config,
        monotonic=clock.monotonic,
        epoch_time=clock.epoch,
        sleeper=clock.sleep,
    )

    async def scenario() -> dict[str, object]:
        await governor.observe(headers={}, http_status=429)
        admission = await governor.acquire(
            estimated_tokens=1,
            stop_launch_at=datetime.fromtimestamp(2_000, timezone.utc),
        )
        assert admission is not None
        return admission

    admission = asyncio.run(scenario())
    # 120 RPM * 0.8 utilization * 0.5 multiplicative decrease = 48 RPM.
    assert clock.sleeps == [pytest.approx(60.0 / 48.0)]
    assert admission["open_loop_wait_seconds"] == pytest.approx(60.0 / 48.0)
    assert admission["congestion_factor"] == 0.5


def test_429_with_full_account_quota_is_endpoint_pressure_not_global_congestion() -> None:
    config = _config(
        Path("unused-endpoint-pressure-429"),
        fallback_account_rpm=120.0,
        fallback_account_tpm=500_000.0,
        quota_utilization_fraction=0.8,
    )
    clock = _FakeClock()
    governor = AccountQuotaGovernor(
        config,
        monotonic=clock.monotonic,
        epoch_time=clock.epoch,
        sleeper=clock.sleep,
    )

    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        signals = await governor.observe(
            headers={
                "x-ratelimit-limit-requests": "4500",
                "x-ratelimit-remaining-requests": "4499",
                "x-ratelimit-reset-requests": "0",
                "x-ratelimit-limit-tokens-per-minute": "3500000",
                "x-ratelimit-remaining-tokens-per-minute": "3500000",
                "x-ratelimit-reset-tokens-per-minute": "0",
                "x-ratelimit-limit-tokens-per-day": "10000000000",
                "x-ratelimit-remaining-tokens-per-day": "1000000000",
                "x-ratelimit-reset-tokens-per-day": "0",
            },
            http_status=429,
        )
        admission = await governor.acquire(
            estimated_tokens=1,
            stop_launch_at=datetime.fromtimestamp(2_000, timezone.utc),
        )
        assert admission is not None
        return signals, admission

    signals, admission = asyncio.run(scenario())
    assert signals["account_quota_congestion_evidence"] is False
    assert signals["http_429_scope_classification"] == (
        "endpoint_pressure_with_account_quota_remaining"
    )
    assert admission["congestion_factor"] == 1.0
    assert clock.sleeps == []
    snapshot = governor.snapshot()
    assert snapshot["http_429_observations"] == 1
    assert snapshot["endpoint_pressure_429_observations"] == 1


@pytest.mark.parametrize(
    "field",
    (
        "fallback_account_rpm",
        "fallback_account_tpm",
        "quota_utilization_fraction",
        "governor_multiplicative_decrease",
        "governor_additive_increase_fraction",
        "governor_minimum_congestion_factor",
    ),
)
def test_governor_contract_rejects_nonfinite_values(tmp_path, field: str) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        DirectContextCampaign(_config(tmp_path, **{field: float("nan")}))


def test_isolated_context_run_keeps_latency_in_comparison_field(tmp_path) -> None:
    async def executor(model_id, task, max_output_tokens):
        return _fake_result(task)

    asyncio.run(
        DirectContextCampaign(
            _config(tmp_path, model_parallelism=1, global_concurrency=1)
        ).run(executor)
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl").read_text().splitlines()
    ]
    assert rows
    assert all(row["latency_comparison_eligible"] is True for row in rows)
    assert all(row["timing"]["request_seconds"] == 0.01 for row in rows)
    assert all(row["concurrent_timing_diagnostic"] is None for row in rows)


def test_429_is_congestion_and_never_a_context_rejection() -> None:
    classification, conclusive, category, fingerprint = classify_failure(
        ProviderHTTPError(
            429,
            "private rate response",
            "2",
            {
                "x-ratelimit-limit-requests": "120",
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-reset-requests": "1234",
            },
        )
    )
    assert classification == "rate_limited"
    assert conclusive is False
    assert category == "rate_limit"
    assert len(fingerprint) == 64


def test_bootstrap_is_serial_until_request_and_tpm_limits_are_observed(
    tmp_path,
) -> None:
    models = (
        "openai-gpt-oss-120b",
        "gemma-4-31B-it",
        "deepseek-v4-flash-0731",
    )
    calls: list[str] = []
    active = 0
    maximum_active = 0

    async def executor(model_id, task, max_output_tokens):
        nonlocal active, maximum_active
        calls.append(model_id)
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.001)
        active -= 1
        result = _fake_result(task)
        if len(calls) == 1:
            # Request header alone is not enough to safely release token-heavy
            # cross-endpoint probes.
            result.response_headers.pop("x-ratelimit-limit-tokens-per-minute")
        return result

    summary = asyncio.run(
        DirectContextCampaign(
            _config(
                tmp_path,
                model_ids=models,
                model_parallelism=3,
                global_concurrency=2,
            )
        ).run(executor)
    )
    assert calls[:2] == list(models[:2])
    assert maximum_active == 2
    assert summary["quota_governor"]["bootstrap_ready"] is True
    assert summary["latency_measurement_scope"] == "concurrent_context_probe"
