from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import itertools
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

import do_benchmark.direct_capability as capability_module
from do_benchmark.core import MODEL_BY_ID, MODEL_SPECS, ProviderHTTPError, StreamResult
from do_benchmark.direct_capability import (
    CapabilityPreflightError,
    DOCUMENTED_STRUCTURED_OUTPUT_MODELS,
    DOCUMENTED_TOOL_MODELS,
    DOCUMENTED_VISION_MODELS,
    OutputDirectoryLease,
    PAIRWISE_FACTORS,
    PAIRWISE_ROWS,
    TOOL_ARGUMENT_BYTE_ANCHORS,
    TOOL_COUNT_ANCHORS,
    TOOL_MALFORMED_CASES,
    TOOL_NESTING_DEPTH_ANCHORS,
    TOOL_REQUIRED_OPTIONAL_MODES,
    TOOL_SCHEMA_BYTE_ANCHORS,
    CapabilityConfig,
    DirectCapabilityCampaign,
    _classification,
    _conservative_cost,
    _render_payload,
    _score_capability_result,
    build_capability_cells,
)
from do_benchmark.direct_report import (
    DEEPSEEK_ENDPOINT_ID,
    PUBLIC_SAFETY_SCAN_SCHEMA,
    analyze_and_write,
)


ROOT = Path(__file__).resolve().parents[1]


def _model_cells(model_id: str):
    return [
        cell
        for cell in build_capability_cells(tuple(spec.model_id for spec in MODEL_SPECS))
        if cell.model_id == model_id
    ]


def test_default_design_covers_every_model_and_fits_budget() -> None:
    model_ids = tuple(spec.model_id for spec in MODEL_SPECS)
    cells = build_capability_cells(model_ids)
    assert len(model_ids) == 12
    assert len(cells) == 1_260
    assert sum(cell.provider_send_expected for cell in cells) == 1_248
    assert len(cells) < 1_300
    planned_reservation = sum(
        _conservative_cost(MODEL_BY_ID[cell.model_id], cell)[0]
        for cell in cells
        if cell.provider_send_expected
    )
    assert planned_reservation == pytest.approx(54.747827242)
    assert planned_reservation < 200

    sample = next(cell for cell in cells if cell.provider_send_expected)
    payload_bytes = len(
        capability_module.canonical_json(
            _render_payload(sample.model_id, sample.task, sample.max_output_tokens)
        ).encode("utf-8")
    )
    assert _conservative_cost(MODEL_BY_ID[sample.model_id], sample)[1] >= (
        payload_bytes + 512
    )

    for model_id in model_ids:
        model_cells = [cell for cell in cells if cell.model_id == model_id]
        expected_cells = 114 if model_id in DOCUMENTED_VISION_MODELS else 96
        if model_id in DOCUMENTED_TOOL_MODELS:
            expected_cells += 24
        assert len(model_cells) == expected_cells
        states: dict[str, set[str]] = {}
        for cell in model_cells:
            states.setdefault(cell.dimension, set()).add(cell.state)
        assert states["temperature"] == {
            "-0.01",
            "0.0",
            "0.5",
            "1.0",
            "1.5",
            "2.0",
            "2.01",
        }
        assert states["top_p"] == {
            "-0.01",
            "0.0",
            "0.25",
            "0.5",
            "0.75",
            "1.0",
            "1.01",
        }
        assert states["top_logprobs"] == {"-1", "0", "5", "10", "15", "20", "21"}
        assert states["max_completion_tokens"] == {"medium"}
        assert states["max_tokens"] == {"small", "medium", "high", "just_over"}
        assert states["tools"] == {"required", "named"}
        assert states["parallel_tool_calls"] == {"false", "true"}
        assert states["stop"] == {"present"}
        assert states["seed"] == {"42", "42-replicate-2"}
        assert states["caching_option"] == {"documented_unavailable"}
        assert "one_small_png" in states["vision"]
        smoke = [
            cell for cell in model_cells if "capability_smoke" in cell.coverage_tags
        ]
        assert len(smoke) == 1
        cache = next(cell for cell in model_cells if cell.dimension == "caching_option")
        assert cache.provider_send_expected is False
        assert cache.local_terminal_status == "documented_unavailable"

        if model_id in DOCUMENTED_TOOL_MODELS:
            assert states["tool_count"] == {str(value) for value in TOOL_COUNT_ANCHORS}
            assert states["tool_schema_bytes"] == {
                str(value) for value in TOOL_SCHEMA_BYTE_ANCHORS
            }
            assert states["tool_nesting_depth"] == {
                str(value) for value in TOOL_NESTING_DEPTH_ANCHORS
            }
            assert states["tool_argument_bytes"] == {
                str(value) for value in TOOL_ARGUMENT_BYTE_ANCHORS
            }
            assert states["tool_required_optional"] == set(TOOL_REQUIRED_OPTIONAL_MODES)
            assert states["tool_malformed_schema"] == set(TOOL_MALFORMED_CASES)
        if model_id in DOCUMENTED_STRUCTURED_OUTPUT_MODELS:
            assert states["response_format"] == {"json_object", "json_schema"}

        penalties = [
            cell.bindings
            for cell in model_cells
            if cell.dimension == "parameter_interaction_penalties"
        ]
        anchors = {-2.01, -2.0, -1.0, 0.0, 1.0, 2.0, 2.01}
        assert {row["frequency_penalty"] for row in penalties} == anchors
        assert {row["presence_penalty"] for row in penalties} == anchors
        extended = [
            cell.bindings
            for cell in model_cells
            if cell.dimension == "parameter_interaction_extended"
        ]
        assert {row["logit_bias_value"] for row in extended} == {
            -101,
            -100,
            -50,
            0,
            50,
            100,
            101,
        }
        assert {row["n"] for row in extended} == {0, 1, 4, 8, 12, 16, 17}
        assert {row["reasoning_effort"] for row in extended} == {
            None,
            "none",
            "low",
            "medium",
            "high",
        }
        assert {row["user"] for row in extended} == {None, "capability-envelope"}

        pairwise_bindings = [
            cell.bindings for cell in model_cells if cell.dimension == "pairwise_core"
        ]
        assert len(PAIRWISE_FACTORS) == 17
        for factor, levels in PAIRWISE_FACTORS.items():
            assert {row[factor] for row in pairwise_bindings} == set(levels)
        triple_bindings = [
            cell.bindings
            for cell in model_cells
            if cell.dimension == "parameter_interaction_temperature_top_p_output"
        ]
        assert len(triple_bindings) == 6
        assert all(
            {"temperature", "top_p", "output_tokens"} == set(row)
            for row in triple_bindings
        )

        seed_cells = [cell for cell in model_cells if cell.dimension == "seed"]
        assert len(seed_cells) == 2
        assert _render_payload(
            model_id, seed_cells[0].task, seed_cells[0].max_output_tokens
        ) == _render_payload(
            model_id, seed_cells[1].task, seed_cells[1].max_output_tokens
        )

    expected_vision_states = {
        "one_small_png",
        "two_small_png",
        "four_small_png",
        "eight_small_png",
        "one_large_png",
        "one_2048_square_png",
        "one_small_jpeg",
        "one_small_webp",
        "one_wide_aspect_png",
        "byte_anchor_16kb",
        "byte_anchor_256kb",
        "byte_anchor_1mb",
        "byte_anchor_4mb",
        "one_png_mixed_text_8k",
        "concurrency_4_a",
        "concurrency_4_b",
        "concurrency_4_c",
        "concurrency_4_d",
        "malformed_data_uri",
    }
    for model_id in DOCUMENTED_VISION_MODELS:
        assert {
            cell.state
            for cell in cells
            if cell.model_id == model_id and cell.dimension == "vision"
        } == expected_vision_states


def test_pairwise_rows_cover_every_pair_of_factor_states() -> None:
    rows = list(PAIRWISE_ROWS)
    assert len(PAIRWISE_FACTORS) == 17
    assert len(rows) == 40
    names = list(PAIRWISE_FACTORS)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            observed = {(row[left], row[right]) for row in rows}
            expected = set(
                itertools.product(PAIRWISE_FACTORS[left], PAIRWISE_FACTORS[right])
            )
            assert observed == expected


def test_documented_tool_envelope_renders_exact_frozen_byte_anchors() -> None:
    cells = _model_cells("qwen3.8-max")
    schema_cells = [cell for cell in cells if cell.dimension == "tool_schema_bytes"]
    assert len(schema_cells) == len(TOOL_SCHEMA_BYTE_ANCHORS)
    for cell in schema_cells:
        assert len(
            capability_module.canonical_json(cell.task.tools).encode("utf-8")
        ) == int(cell.state)

    argument_cells = [cell for cell in cells if cell.dimension == "tool_argument_bytes"]
    assert len(argument_cells) == len(TOOL_ARGUMENT_BYTE_ANCHORS)
    for cell in argument_cells:
        arguments = cell.task.expected["value"]["arguments"]
        assert len(capability_module.canonical_json(arguments).encode("utf-8")) == int(
            cell.state
        )
        assert cell.max_output_tokens == int(cell.state) + 128

    assert not [
        cell
        for cell in _model_cells("deepseek-v4-flash-0731")
        if cell.dimension.startswith("tool_")
    ]


def test_multi_image_tasks_score_every_distinct_semantic_signal_without_leak() -> None:
    task = next(
        cell.task
        for cell in _model_cells("qwen3.8-max")
        if cell.dimension == "vision" and cell.state == "eight_small_png"
    )
    content = task.messages[0]["content"]
    prompt = content[0]["text"].casefold()
    for colour in ("red", "green", "blue", "yellow"):
        assert colour not in prompt
    image_urls = [item["image_url"]["url"] for item in content[1:]]
    assert len(image_urls) == 8
    assert len(set(image_urls)) == 8
    expected_groups = str(task.expected["value"]).split(" | ")
    assert len(expected_groups) == 8
    correct = _fake_result(task)
    assert _score_capability_result(task, correct)["quality_score"] == 1.0
    wrong = replace(correct, text=" | ".join(reversed(expected_groups)))
    assert _score_capability_result(task, wrong)["quality_score"] == 0.0


def test_noisy_byte_anchor_retains_quadrant_signal() -> None:
    task = next(
        cell.task
        for cell in _model_cells("qwen3.8-max")
        if cell.dimension == "vision" and cell.state == "byte_anchor_256kb"
    )
    data_uri = task.messages[0]["content"][1]["image_url"]["url"]
    image = Image.open(io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1]))).convert(
        "RGB"
    )
    width, height = image.size
    samples = [
        image.getpixel((width // 4, height // 4)),
        image.getpixel((3 * width // 4, height // 4)),
        image.getpixel((width // 4, 3 * height // 4)),
        image.getpixel((3 * width // 4, 3 * height // 4)),
    ]
    red, green, blue, yellow = samples
    assert red[0] > 180 and red[1] < 80 and red[2] < 80
    assert green[1] > 180 and green[0] < 80 and green[2] < 80
    assert blue[2] > 180 and blue[0] < 80 and blue[1] < 80
    assert yellow[0] > 180 and yellow[1] > 180 and yellow[2] < 80


def test_preflight_rejects_cumulative_budget_or_insufficient_interaction_slots(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="full capability plan cannot fit"):
        DirectCapabilityCampaign(
            replace(_config(tmp_path / "budget"), prior_cost_usd=199.999)
        )
    with pytest.raises(ValueError, match="largest planned interaction"):
        DirectCapabilityCampaign(
            CapabilityConfig(
                output_dir=tmp_path / "concurrency",
                model_ids=("qwen3.8-max",),
                max_workers=2,
                per_model_concurrency=2,
            )
        )


def _fake_result(task) -> StreamResult:
    expected_calls = task.metadata.get("expected_tool_calls")
    if isinstance(expected_calls, list):
        text = ""
        tool_calls = expected_calls
    elif task.expected["kind"] == "json_exact":
        text = json.dumps(task.expected["value"])
        tool_calls = []
    elif task.expected["kind"] == "tool_exact":
        text = ""
        tool_calls = [task.expected["value"]]
    else:
        text = str(task.expected["value"])
        tool_calls = []
    return StreamResult(
        status_code=200,
        response_headers={
            "x-request-id": "private-provider-request-id",
            "cf-ray": "private-edge-id",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
        },
        text=text,
        reasoning_text="private reasoning trace",
        tool_calls=tool_calls,
        usage={"prompt_tokens": 64, "completion_tokens": 8, "total_tokens": 72},
        finish_reason="stop",
        request_seconds=0.01,
        headers_seconds=0.002,
        ttft_seconds=0.004,
        generation_seconds=0.006,
        stream_seconds=0.008,
        event_count=2,
        first_event_kind="content",
    )


def _config(tmp_path: Path) -> CapabilityConfig:
    return CapabilityConfig(
        output_dir=tmp_path,
        model_ids=("openai-gpt-oss-120b",),
        max_workers=16,
        per_model_concurrency=8,
        request_timeout_seconds=1.0,
        max_cost_usd=200.0,
        stop_launch_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )


def test_parallel_tool_score_distinguishes_enabled_and_disabled_contracts() -> None:
    enabled_task = next(
        cell.task
        for cell in _model_cells("openai-gpt-oss-120b")
        if cell.dimension == "parallel_tool_calls" and cell.state == "true"
    )
    complete = _fake_result(enabled_task)
    assert _score_capability_result(enabled_task, complete)["quality_score"] == 1.0
    assert (
        _score_capability_result(
            enabled_task, replace(complete, tool_calls=complete.tool_calls[:1])
        )["quality_score"]
        == 0.0
    )
    wrong_argument = replace(
        complete,
        tool_calls=[
            complete.tool_calls[0],
            {"name": "lookup_backup", "arguments": {"station": "wrong"}},
        ],
    )
    assert (
        _score_capability_result(enabled_task, wrong_argument)["quality_score"] == 0.0
    )

    disabled_task = next(
        cell.task
        for cell in _model_cells("openai-gpt-oss-120b")
        if cell.dimension == "parallel_tool_calls" and cell.state == "false"
    )
    single = _fake_result(disabled_task)
    score = _score_capability_result(disabled_task, single)
    assert score["quality_score"] == 1.0
    assert score["score_kind"] == "parallel_disabled_single_call_exact"
    assert len(single.tool_calls) == 1


def test_campaign_sanitizes_rows_and_deduplicates_resume(tmp_path: Path) -> None:
    calls = 0

    async def executor(model_id, task, max_output_tokens):
        nonlocal calls
        assert model_id == "openai-gpt-oss-120b"
        assert max_output_tokens >= 1
        calls += 1
        return _fake_result(task)

    config = _config(tmp_path)
    expected_provider_calls = sum(
        cell.provider_send_expected for cell in DirectCapabilityCampaign(config).cells
    )
    first = asyncio.run(DirectCapabilityCampaign(config).run(executor))
    assert first["planned_requests"] == 96
    assert first["terminal_coverage_complete"] is True
    assert calls == expected_provider_calls == 95

    request_text = (tmp_path / "records.jsonl").read_text(encoding="utf-8")
    plan_text = (tmp_path / "plan.jsonl").read_text(encoding="utf-8")
    assert "private-provider-request-id" not in request_text
    assert "private-edge-id" not in request_text
    assert "private reasoning trace" not in request_text
    assert "CAP-OK" not in request_text
    assert "CAP-OK" not in plan_text
    assert "data:image" not in plan_text
    rows = [json.loads(line) for line in request_text.splitlines()]
    for row in rows:
        assert "response" not in row
        assert "response_headers" not in row
        assert "error" not in row
        assert len(row["request_payload_sha256"]) == 64
        if row["provider_send_attempted"] and row["status"] == "accepted":
            assert len(row["response_sha256"]) == 64

    seed_rows = [row for row in rows if row["dimension"] == "seed"]
    assert len(seed_rows) == 2
    assert (
        seed_rows[0]["request_payload_sha256"] == seed_rows[1]["request_payload_sha256"]
    )

    second = asyncio.run(DirectCapabilityCampaign(config).run(executor))
    assert calls == expected_provider_calls
    assert second["terminal_rows"] == 96


def test_output_directory_lease_blocks_duplicate_sends_and_stale_replay(
    tmp_path: Path,
) -> None:
    calls = 0

    async def executor(model_id, task, max_output_tokens):
        nonlocal calls
        calls += 1
        return _fake_result(task)

    config = _config(tmp_path)
    first = DirectCapabilityCampaign(config)
    stale = DirectCapabilityCampaign(config)
    with OutputDirectoryLease(first.execution_lease_path):
        with pytest.raises(CapabilityPreflightError, match="execution lease"):
            asyncio.run(first.run(executor))
    assert calls == 0

    first_summary = asyncio.run(first.run(executor))
    expected_calls = first_summary["provider_attempts"]
    assert expected_calls == 95
    stale_summary = asyncio.run(stale.run(executor))
    assert calls == expected_calls
    assert stale_summary["terminal_coverage_complete"] is True


def test_http_402_latch_is_reconstructed_before_resume_sends(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = DirectCapabilityCampaign(config)
    cell = next(cell for cell in first.cells if cell.provider_send_expected)
    reserved_cost, reserved_tokens = _conservative_cost(
        MODEL_BY_ID[cell.model_id], cell
    )
    assert asyncio.run(
        first.budget.reserve(
            campaign_id=first.campaign_id,
            request_id=cell.request_id,
            epoch_id="capability-envelope",
            model_id=cell.model_id,
            shape=cell.dimension,
            reserved_cost_usd=reserved_cost,
            reserved_prompt_tokens=reserved_tokens,
            max_output_tokens=cell.max_output_tokens,
        )
    )
    row = {
        **first._base_row(cell),
        "provider_send_attempted": True,
        "status": "account_blocked_402",
        "coverage_classification": "account_blocked_402",
        "coverage_conclusive": False,
        "http_status": 402,
        "usage": {},
        "worst_case_reserved_cost_usd": reserved_cost,
        "reserved_prompt_tokens": reserved_tokens,
        "estimated_cost_usd": None,
        "accounted_cost_usd": reserved_cost,
    }
    asyncio.run(first._append(row))

    resumed = DirectCapabilityCampaign(config)
    assert resumed.account_blocked_402 is True
    calls = 0

    async def executor(model_id, task, max_output_tokens):
        nonlocal calls
        calls += 1
        return _fake_result(task)

    summary = asyncio.run(resumed.run(executor))
    assert calls == 0
    assert summary["http_402_latched"] is True
    assert summary["terminal_coverage_complete"] is True


def test_duplicate_or_foreign_resume_rows_fail_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    campaign = DirectCapabilityCampaign(config)
    cell = campaign.cells[0]
    row = {**campaign._base_row(cell), "request_id": cell.request_id}
    encoded = json.dumps(row, sort_keys=True) + "\n"
    (tmp_path / "records.jsonl").write_text(encoded + encoded, encoding="utf-8")
    with pytest.raises(CapabilityPreflightError, match="duplicate request_id"):
        DirectCapabilityCampaign(config)


def test_resume_rejects_understated_incomplete_usage_cost(tmp_path: Path) -> None:
    config = _config(tmp_path)
    campaign = DirectCapabilityCampaign(config)
    cell = next(cell for cell in campaign.cells if cell.provider_send_expected)
    reserved_cost, reserved_tokens = _conservative_cost(
        MODEL_BY_ID[cell.model_id], cell
    )
    assert asyncio.run(
        campaign.budget.reserve(
            campaign_id=campaign.campaign_id,
            request_id=cell.request_id,
            epoch_id="capability-envelope",
            model_id=cell.model_id,
            shape=cell.dimension,
            reserved_cost_usd=reserved_cost,
            reserved_prompt_tokens=reserved_tokens,
            max_output_tokens=cell.max_output_tokens,
        )
    )
    asyncio.run(
        campaign.requests_journal.append(
            {
                **campaign._base_row(cell),
                "provider_send_attempted": True,
                "status": "accepted",
                "http_status": 200,
                "usage": {"prompt_tokens": 64, "completion_tokens": 0},
                "worst_case_reserved_cost_usd": reserved_cost,
                "reserved_prompt_tokens": reserved_tokens,
                "accounted_cost_usd": 0.0,
            }
        )
    )
    with pytest.raises(CapabilityPreflightError, match="incomplete-usage request cost"):
        DirectCapabilityCampaign(config)


def test_cli_requires_explicit_prior_and_send_window(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run-digitalocean-direct-capability.py"
    )
    base = [sys.executable, str(script), "--output-dir", str(tmp_path), "--plan-only"]
    missing_both = subprocess.run(
        base,
        cwd=script.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing_both.returncode == 2
    assert "--prior-cost-usd" in missing_both.stderr

    missing_window = subprocess.run(
        [*base, "--prior-cost-usd", "0"],
        cwd=script.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing_window.returncode == 2
    assert "--stop-launch-at" in missing_window.stderr


def test_campaign_identity_binds_exact_plan_and_science_contract(
    tmp_path: Path,
) -> None:
    campaign = DirectCapabilityCampaign(_config(tmp_path))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    plan_bytes = (tmp_path / "plan.jsonl").read_bytes()
    assert hashlib.sha256(plan_bytes).hexdigest() == campaign.plan_sha256
    assert manifest["plan_sha256"] == campaign.plan_sha256
    assert manifest["campaign_identity_sha256"] == campaign.campaign_identity_sha256
    assert manifest["documentation_contract"]["artifact_sha256"]
    assert manifest["reservation_contract_version"]
    assert set(manifest["runner_source_sha256"]) == {
        "do_benchmark/direct_capability.py",
        "do_benchmark/core.py",
        "do_benchmark/direct_aimd.py",
        "scripts/run-digitalocean-direct-capability.py",
    }
    assert manifest["planned_worst_case_reservation_usd"] == pytest.approx(
        campaign.planned_reservation_usd
    )
    assert len(manifest["pairwise_factors"]) == 17
    for cell in campaign.cells:
        assert len(cell.rendered_payload_sha256) == 64
        assert len(cell.scorer_contract_sha256) == 64
        assert len(cell.model_contract_sha256) == 64
        assert len(cell.documentation_contract_sha256) == 64
        assert len(cell.request_identity_sha256) == 64
        assert cell.request_id.endswith(cell.request_identity_sha256[:20])

    (tmp_path / "plan.jsonl").write_bytes(plan_bytes + b"\n")
    with pytest.raises(RuntimeError, match="exact hash"):
        DirectCapabilityCampaign(_config(tmp_path))


def test_request_identity_changes_with_scorer_and_model_contract(monkeypatch) -> None:
    model_id = "openai-gpt-oss-120b"

    def smoke_id() -> str:
        return next(
            cell.request_id
            for cell in build_capability_cells((model_id,))
            if cell.probe_id == "capability-smoke"
        )

    baseline = smoke_id()
    with monkeypatch.context() as patch:
        patch.setattr(
            capability_module,
            "SCORER_CONTRACT_VERSION",
            "direct_capability_scorer_contract_changed_for_test",
        )
        assert smoke_id() != baseline
    with monkeypatch.context() as patch:
        spec = MODEL_BY_ID[model_id]
        patch.setitem(
            MODEL_BY_ID,
            model_id,
            replace(spec, input_usd_per_million=spec.input_usd_per_million + 0.001),
        )
        assert smoke_id() != baseline


def test_incomplete_usage_retains_worst_case_reservation(tmp_path: Path) -> None:
    async def executor(model_id, task, max_output_tokens):
        return replace(_fake_result(task), usage={"prompt_tokens": 64})

    asyncio.run(DirectCapabilityCampaign(_config(tmp_path)).run(executor))
    rows = [
        json.loads(line)
        for line in (tmp_path / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    accepted = [row for row in rows if row["provider_send_attempted"]]
    assert accepted
    assert all(row["prompt_usage_present"] is True for row in accepted)
    assert all(row["completion_usage_present"] is False for row in accepted)
    assert all(row["usage_complete_for_settlement"] is False for row in accepted)
    assert all(row["estimated_cost_usd"] is None for row in accepted)
    assert all(
        row["accounted_cost_usd"] == row["worst_case_reserved_cost_usd"]
        for row in accepted
    )


def test_explicit_zero_completion_usage_retains_worst_case_reservation(
    tmp_path: Path,
) -> None:
    async def executor(model_id, task, max_output_tokens):
        return replace(
            _fake_result(task),
            usage={"prompt_tokens": 64, "completion_tokens": 0},
        )

    asyncio.run(DirectCapabilityCampaign(_config(tmp_path)).run(executor))
    rows = [
        json.loads(line)
        for line in (tmp_path / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    accepted = [row for row in rows if row["provider_send_attempted"]]
    assert accepted
    assert all(row["usage_complete_for_settlement"] is False for row in accepted)
    assert all(row["estimated_cost_usd"] is None for row in accepted)
    assert all(
        row["accounted_cost_usd"] == row["worst_case_reserved_cost_usd"]
        for row in accepted
    )


def test_only_explicit_validation_statuses_are_conclusive_rejections(
    tmp_path: Path,
) -> None:
    assert (
        _classification(ProviderHTTPError(400, "body")) == "client_error_inconclusive"
    )
    assert (
        _classification(ProviderHTTPError(400, "unsupported parameter: top_p"))
        == "rejected_or_unsupported"
    )
    assert (
        _classification(
            ProviderHTTPError(
                422,
                json.dumps(
                    {
                        "detail": [
                            {
                                "loc": ["body", "temperature"],
                                "msg": "must be less than or equal to 2",
                                "input": 9,
                            }
                        ]
                    }
                ),
            )
        )
        == "rejected_or_unsupported"
    )
    # An echoed request parameter outside an allowlisted reason field must not
    # combine with a generic error message to manufacture conclusive evidence.
    echoed = json.dumps(
        {"error": {"message": "invalid input"}, "request": {"temperature": 9}}
    )
    assert (
        _classification(ProviderHTTPError(400, echoed)) == "client_error_inconclusive"
    )
    for status in (401, 403, 404, 405, 408, 409, 415):
        assert (
            _classification(ProviderHTTPError(status, "body"))
            == "client_error_inconclusive"
        )
    assert _classification(ProviderHTTPError(429, "body")) == "rate_limited"
    assert _classification(ProviderHTTPError(503, "body")) == "provider_error"

    async def executor(model_id, task, max_output_tokens):
        raise ProviderHTTPError(
            400, "unsupported parameter top_p; sensitive provider detail"
        )

    summary = asyncio.run(DirectCapabilityCampaign(_config(tmp_path)).run(executor))
    assert summary["models"]["openai-gpt-oss-120b"]["conclusive_cells"] == 96
    text = (tmp_path / "records.jsonl").read_text(encoding="utf-8")
    assert "sensitive provider detail" not in text
    assert text.count('"coverage_classification": "rejected_or_unsupported"') == 95
    assert text.count('"provider_reason_sha256":') == 95
    assert (
        text.count(
            '"provider_reason_category": "explicit_unsupported_parameter_or_capability"'
        )
        == 95
    )
    assert text.count('"coverage_classification": "documented_unavailable"') == 1


def test_hard_wall_clock_timeout_is_terminal_and_inconclusive(tmp_path: Path) -> None:
    async def executor(model_id, task, max_output_tokens):
        await asyncio.sleep(1.0)
        return _fake_result(task)

    config = replace(_config(tmp_path), request_timeout_seconds=0.01)
    summary = asyncio.run(DirectCapabilityCampaign(config).run(executor))
    assert summary["terminal_coverage_complete"] is True
    assert summary["models"]["openai-gpt-oss-120b"]["outcomes"] == {
        "documented_unavailable": 1,
        "timed_out": 95,
    }
    assert summary["models"]["openai-gpt-oss-120b"]["conclusive_cells"] == 1


def test_orphaned_reservation_is_never_replayed(tmp_path: Path) -> None:
    calls = 0

    async def executor(model_id, task, max_output_tokens):
        nonlocal calls
        calls += 1
        return _fake_result(task)

    config = _config(tmp_path)
    first_campaign = DirectCapabilityCampaign(config)
    orphan = next(cell for cell in first_campaign.cells if cell.provider_send_expected)
    reserved_cost, reserved_tokens = _conservative_cost(
        MODEL_BY_ID[orphan.model_id], orphan
    )
    assert asyncio.run(
        first_campaign.budget.reserve(
            campaign_id=first_campaign.campaign_id,
            request_id=orphan.request_id,
            epoch_id="capability-envelope",
            model_id=orphan.model_id,
            shape=orphan.dimension,
            reserved_cost_usd=reserved_cost,
            reserved_prompt_tokens=reserved_tokens,
            max_output_tokens=orphan.max_output_tokens,
        )
    )

    summary = asyncio.run(DirectCapabilityCampaign(config).run(executor))
    assert calls == 94
    assert summary["terminal_coverage_complete"] is True
    rows = [
        json.loads(line)
        for line in (tmp_path / "records.jsonl").read_text().splitlines()
    ]
    orphan_row = next(row for row in rows if row["request_id"] == orphan.request_id)
    assert orphan_row["status"] == "unknown_prior_reservation"
    assert orphan_row["provider_send_attempted"] is True


def test_capability_campaign_reconciles_into_public_report_dimensions(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "capability"

    async def executor(model_id, task, max_output_tokens):
        return _fake_result(task)

    config = replace(
        _config(campaign_dir),
        model_ids=(DEEPSEEK_ENDPOINT_ID,),
    )
    asyncio.run(DirectCapabilityCampaign(config).run(executor))
    report_dir = tmp_path / "report"
    analysis = analyze_and_write(
        breadth_directories=[campaign_dir],
        aimd_directories=[],
        endpoint_freeze=ROOT / "config" / "endpoint-freeze.json",
        output_directory=report_dir,
        seed=7,
        bootstrap_replicates=5,
    )
    assert analysis["coverage_summary"]
    endpoint_rows = [
        row
        for row in map(
            json.loads,
            (report_dir / "coverage-ledger.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
        if row["endpoint_id"] == DEEPSEEK_ENDPOINT_ID
    ]
    dimensions = {row["coverage_dimension"] for row in endpoint_rows}
    assert {
        "capability_smoke",
        "parameter_validation",
        "parameter_interactions",
        "output_length",
        "vision",
        "tool_calling",
        "structured_output",
    } <= dimensions
    measurement_rows = [
        row
        for row in endpoint_rows
        if row.get("evidence_scope") != "manifest_scope_exclusion"
    ]
    exclusion_rows = [
        row
        for row in endpoint_rows
        if row.get("evidence_scope") == "manifest_scope_exclusion"
    ]
    assert all(row["observed_attempt_count"] == 1 for row in measurement_rows)
    assert exclusion_rows
    assert all(row["observed_attempt_count"] == 0 for row in exclusion_rows)
    tool_matrix = next(
        row
        for row in analysis["coverage_matrix"]
        if row["endpoint_id"] == DEEPSEEK_ENDPOINT_ID
        and row["coverage_dimension"] == "tool_calling"
    )
    assert tool_matrix["status"] == "completed"
    assert tool_matrix["explicit_untested_subtest_count"] == 1
    assert tool_matrix["has_explicit_scope_exclusions"] is True
    scan = json.loads(
        (report_dir / "public-safety-scan.json").read_text(encoding="utf-8")
    )
    assert scan["passed"] is True
    assert scan["schema_version"] == PUBLIC_SAFETY_SCAN_SCHEMA
    assert analysis["public_bundle_safety"]["scan_receipt"] == (
        "public-safety-scan.json"
    )
    assert not (report_dir / "publication-scan.json").exists()
