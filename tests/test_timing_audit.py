from __future__ import annotations

from do_benchmark.core import StreamResult, parse_token_usage
from do_benchmark.timing_audit import audit_row, timing_evidence


def _result(
    *,
    completion_tokens: int = 128,
    request_seconds: float = 2.0,
    ttft_seconds: float | None = 0.5,
    generation_seconds: float | None = 1.0,
    event_count: int = 16,
    usage_extra: dict[str, object] | None = None,
) -> StreamResult:
    usage: dict[str, object] = {
        "prompt_tokens": 1_000,
        "completion_tokens": completion_tokens,
        "total_tokens": 1_000 + completion_tokens,
    }
    usage.update(usage_extra or {})
    return StreamResult(
        status_code=200,
        response_headers={},
        text="ok",
        reasoning_text="",
        tool_calls=[],
        usage=usage,  # type: ignore[arg-type]
        finish_reason="stop",
        request_seconds=request_seconds,
        headers_seconds=0.1,
        ttft_seconds=ttft_seconds,
        generation_seconds=generation_seconds,
        stream_seconds=max(0.0, request_seconds - 0.1),
        event_count=event_count,
        first_event_kind="content",
    )


def test_batched_single_chunk_does_not_become_fake_decode_tps() -> None:
    evidence = timing_evidence(
        _result(completion_tokens=3, generation_seconds=0.0000103, event_count=1)
    )
    assert evidence["sse_chunk_span_output_tokens_per_second_proxy"] is None
    assert evidence["exploratory_sse_chunk_span_output_tokens_per_second_proxy"] is None
    assert evidence["output_tokens_per_second"] == 2.0
    assert (
        "fewer_than_eight_content_events"
        in evidence["sse_chunk_span_invalidity_reasons"]
    )


def test_sse_proxy_requires_strict_events_tokens_and_span() -> None:
    assert (
        timing_evidence(_result(event_count=7))[
            "sse_chunk_span_output_tokens_per_second_proxy"
        ]
        is None
    )
    assert (
        timing_evidence(_result(completion_tokens=31))[
            "sse_chunk_span_output_tokens_per_second_proxy"
        ]
        is None
    )
    assert (
        timing_evidence(_result(generation_seconds=0.049))[
            "sse_chunk_span_output_tokens_per_second_proxy"
        ]
        is None
    )
    valid = timing_evidence(
        _result(completion_tokens=128, event_count=8, generation_seconds=0.05)
    )
    assert valid["sse_chunk_span_output_tokens_per_second_proxy"] == 2_560.0
    assert valid["sse_chunk_span_endpoint_comparison_eligible"] is True


def test_cache_unknown_and_hit_suppress_prefill_headline() -> None:
    unknown = timing_evidence(_result())
    assert unknown["prompt_tokens_per_second_to_first_token"] is None
    assert unknown["exploratory_prompt_tokens_per_second_to_first_token"] == 2_000.0
    hit = timing_evidence(
        _result(usage_extra={"prompt_tokens_details": {"cached_tokens": 900}})
    )
    assert hit["cache_observation"]["observed_state"] == "cache_hit_observed"
    assert hit["prompt_tokens_per_second_to_first_token"] is None


def test_observed_cache_miss_allows_prefill_proxy_and_nested_counter_is_preserved() -> (
    None
):
    parsed = parse_token_usage(
        {"prompt_tokens": 1_000, "prompt_tokens_details": {"cached_tokens": 0}}
    )
    assert parsed["cached_tokens"] == 0
    miss = timing_evidence(
        _result(usage_extra={"prompt_tokens_details": {"cached_tokens": 0}}),
        intended_cache_state="uncached_randomized_prefix",
    )
    assert miss["cache_observation"]["observed_state"] == "cache_miss_observed"
    assert miss["prompt_tokens_per_second_to_first_token"] == 2_000.0


def test_inconsistent_monotonic_clock_censors_rates() -> None:
    evidence = timing_evidence(
        _result(), monotonic_started_ns=5_000, monotonic_ended_ns=4_000
    )
    assert evidence["output_tokens_per_second"] is None
    assert "invalid_monotonic_timestamp_pair" in evidence["timing_invalidity_reasons"]


def test_multi_sequence_and_nonstreaming_usage_never_mix_into_per_sequence_rates() -> (
    None
):
    multi = timing_evidence(_result(), sequence_count=16)
    assert multi["output_tokens_per_second"] is None
    assert "aggregate_usage_for_multiple_sequences" in multi["usage_invalidity_reasons"]
    nonstream = timing_evidence(_result(), streaming=False)
    assert nonstream["ttft_seconds"] == 0.5  # raw transport value is preserved
    assert nonstream["output_tokens_per_second"] is None
    assert (
        "nonstreaming_response_has_no_observed_ttft"
        in nonstream["usage_invalidity_reasons"]
    )


def test_valid_extreme_is_retained_flagged_and_request_traceable() -> None:
    result = _result(
        completion_tokens=2_000,
        request_seconds=1.1,
        ttft_seconds=0.1,
        generation_seconds=0.5,
        event_count=20,
        usage_extra={"cache_read_input_tokens": 0},
    )
    timing = timing_evidence(result)
    assert timing["output_tokens_per_second"] == 2_000.0
    assert timing["metric_audit_classification"] == "valid_extreme_keep_and_flag"
    row = audit_row(
        {
            "request_id": "request-123",
            "campaign_id": "campaign-1",
            "model_id": "deepseek-v4-flash-0731",
            "shape": "short_long",
            "phase": "aimd",
            "status": "success",
            "usage": result.usage,
            "timing": timing,
        }
    )
    assert row["request_id"] == "request-123"
    assert row["classification"] == "valid_extreme_keep_and_flag"
    assert row["trimmed_or_winsorized"] is False
