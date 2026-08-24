from __future__ import annotations

import asyncio
import base64
from types import MethodType

from do_benchmark.core import (
    BenchmarkTask,
    BenchmarkRunner,
    ModelSpec,
    StreamResult,
    build_plan,
    parse_token_usage,
    quadrant_png_data_uri,
    score_result,
    wilson_interval,
)


def _result(*, text: str = "", tool_calls=None) -> StreamResult:
    return StreamResult(
        status_code=200,
        response_headers={},
        text=text,
        reasoning_text="",
        tool_calls=tool_calls or [],
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        finish_reason="stop",
        request_seconds=1.0,
        headers_seconds=0.1,
        ttft_seconds=0.2,
        generation_seconds=0.5,
        stream_seconds=0.9,
        event_count=1,
        first_event_kind="content",
    )


def _task(task_id: str, *, vision: bool = False) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        family="exact",
        context_bucket="short",
        output_bucket="short",
        messages=[{"role": "user", "content": "test"}],
        expected={"kind": "exact_text", "value": "yes"},
        requires_vision=vision,
    )


def test_quadrant_image_is_a_real_png() -> None:
    uri = quadrant_png_data_uri(8)
    encoded = uri.split(",", 1)[1]
    assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")


def test_usage_parser_preserves_presence_and_rejects_inexact_counters() -> None:
    assert parse_token_usage({"completion_tokens": 0}) == {"completion_tokens": 0}
    assert parse_token_usage({"prompt_tokens": 12, "completion_tokens": 0}) == {
        "prompt_tokens": 12,
        "completion_tokens": 0,
    }
    assert (
        parse_token_usage(
            {
                "prompt_tokens": 1.5,
                "completion_tokens": "2",
                "total_tokens": None,
                "ignored": 4,
            }
        )
        == {}
    )
    assert parse_token_usage(
        {"prompt_tokens": True, "completion_tokens": -1, "total_tokens": 12.0}
    ) == {"total_tokens": 12}


def test_balanced_plan_is_deterministic_and_filters_vision() -> None:
    text_model = ModelSpec("text", 1, 1, 1000)
    vision_model = ModelSpec("vision", 1, 1, 1000, vision=True)
    tasks = [_task("plain"), _task("image", vision=True)]
    first = build_plan(
        model_specs=[text_model, vision_model], tasks=tasks, repeats=2, seed=7
    )
    second = build_plan(
        model_specs=[text_model, vision_model], tasks=tasks, repeats=2, seed=7
    )
    assert [cell.cell_id for cell in first] == [cell.cell_id for cell in second]
    assert len(first) == 6
    assert all(cell.model_id == "vision" for cell in first if cell.task.requires_vision)


def test_tool_score_requires_exact_name_and_arguments() -> None:
    expected = {"name": "publish", "arguments": {"ids": ["a", "b"], "count": 2}}
    task = BenchmarkTask(
        task_id="tool",
        family="tool_call_exact",
        context_bucket="short",
        output_bucket="tool",
        messages=[],
        expected={"kind": "tool_exact", "value": expected},
    )
    assert score_result(task, _result(tool_calls=[expected]))["quality_score"] == 1
    reordered_list = {"name": "publish", "arguments": {"ids": ["b", "a"], "count": 2}}
    assert (
        score_result(task, _result(tool_calls=[reordered_list]))["quality_score"] == 0
    )


def test_controlled_output_scores_length_and_sentinel() -> None:
    task = BenchmarkTask(
        task_id="output",
        family="controlled_output",
        context_bucket="short",
        output_bucket="4",
        messages=[],
        expected={"kind": "controlled_words", "count": 4, "marker": "FIN"},
    )
    assert (
        score_result(task, _result(text="azure azure azure FIN"))["quality_score"] == 1
    )
    assert score_result(task, _result(text="azure azure FIN"))["quality_score"] == 0.75


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(7, 10)
    assert low < 0.7 < high


def test_serial_by_model_continues_after_one_lane_cell_is_cancelled(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "do_benchmark.core.digitalocean_credentials",
        lambda: {"api_key": "test", "api_base": "https://example.invalid"},
    )
    tasks = [_task("first"), _task("second")]
    cells = build_plan(
        model_specs=[ModelSpec("model-a", 1, 1, 1_000)],
        tasks=tasks,
        repeats=1,
        seed=7,
    )
    runner = BenchmarkRunner(
        output_dir=tmp_path,
        max_workers=1,
        per_model_max_concurrency=1,
        initial_concurrency=1,
        grow_after=1,
        request_timeout_seconds=1,
        total_request_timeout_seconds=1,
        safety_max_output_tokens=8,
        max_cost_usd=1,
    )
    observed: list[str] = []

    async def fake_run_cell(self, client, cell) -> None:
        del self, client
        observed.append(cell.task.task_id)
        if len(observed) == 1:
            raise asyncio.CancelledError

    runner._run_cell = MethodType(fake_run_cell, runner)
    summary = asyncio.run(runner.run(cells, serial_by_model=True))

    assert sorted(observed) == ["first", "second"]
    assert len(summary["unjournaled_errors"]) == 1
