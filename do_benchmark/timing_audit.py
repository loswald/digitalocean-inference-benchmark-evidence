"""Conservative timing metrics and request-level outlier evidence.

Server streaming APIs expose content *events*, not one timestamp per decoded
token.  A provider may batch hundreds of tokens into one or two SSE events, so
``last_event - first_event`` is not a decode duration unless the event span is
both measurable and internally consistent.  This module keeps the raw
observations, computes a conservative post-TTFT end-to-end proxy, and labels
every invalid or extreme measurement instead of silently trimming it.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from do_benchmark.core import StreamResult, parse_token_usage


MIN_MEASURABLE_INTERVAL_SECONDS = 1e-4
EXTREME_OUTPUT_TPS = 1_000.0
EXTREME_PREFILL_PROXY_TPS = 1_000_000.0
EXTREME_LOW_TTFT_SECONDS = 0.001


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_interval(value: Any) -> float | None:
    parsed = _finite(value)
    if parsed is None or parsed < MIN_MEASURABLE_INTERVAL_SECONDS:
        return None
    return parsed


def cache_observation(
    usage_value: Any, *, intended_state: str | None = None
) -> dict[str, Any]:
    """Return explicit cache state without inferring a miss from absent fields."""

    usage = parse_token_usage(usage_value)
    present = any(
        key in usage
        for key in (
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "cached_tokens",
        )
    )
    cache_read = max(
        int(usage.get("cache_read_input_tokens") or 0),
        int(usage.get("cached_tokens") or 0),
    )
    return {
        "intended_state": intended_state,
        "observed_state": (
            "cache_hit_observed"
            if cache_read > 0
            else ("cache_miss_observed" if present else "not_reported_unknown")
        ),
        "cache_counters_reported": present,
        "cache_read_tokens": cache_read if present else None,
        "cache_creation_input_tokens": (
            int(usage.get("cache_creation_input_tokens") or 0) if present else None
        ),
        "prefill_comparability": (
            "stratify_by_observed_cache_state; do_not_pool_unknown_with_hit_or_miss"
        ),
    }


def timing_evidence(
    result: StreamResult,
    *,
    monotonic_started_ns: int | None = None,
    monotonic_ended_ns: int | None = None,
    intended_cache_state: str | None = None,
    sequence_count: int = 1,
    streaming: bool = True,
) -> dict[str, Any]:
    """Return raw observations, validity reasons, and conservative proxies.

    ``output_tokens_per_second`` intentionally means completion tokens divided
    by the end-to-end interval after TTFT.  The event-span calculation is
    retained separately and explicitly labelled as an SSE-chunk-span proxy.
    It is never represented as direct server decode throughput.
    """

    usage = parse_token_usage(result.usage)
    cache = cache_observation(result.usage, intended_state=intended_cache_state)
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    request_seconds = _finite(result.request_seconds)
    headers_seconds = _finite(result.headers_seconds)
    ttft_seconds = _finite(result.ttft_seconds)
    generation_seconds = _finite(result.generation_seconds)
    stream_seconds = _finite(result.stream_seconds)
    invalid_timing_reasons: list[str] = []

    if request_seconds is None or request_seconds < 0:
        invalid_timing_reasons.append("missing_or_negative_request_seconds")
    if headers_seconds is None or headers_seconds < 0:
        invalid_timing_reasons.append("missing_or_negative_headers_seconds")
    if (
        request_seconds is not None
        and headers_seconds is not None
        and headers_seconds > request_seconds + 1e-9
    ):
        invalid_timing_reasons.append("headers_after_request_end")
    if ttft_seconds is not None and ttft_seconds < 0:
        invalid_timing_reasons.append("negative_ttft_seconds")
    if (
        request_seconds is not None
        and ttft_seconds is not None
        and ttft_seconds > request_seconds + 1e-9
    ):
        invalid_timing_reasons.append("ttft_after_request_end")
    if generation_seconds is not None and generation_seconds < 0:
        invalid_timing_reasons.append("negative_sse_event_span")
    if (
        request_seconds is not None
        and generation_seconds is not None
        and generation_seconds > request_seconds + 1e-9
    ):
        invalid_timing_reasons.append("sse_event_span_exceeds_request")
    if stream_seconds is not None and stream_seconds < 0:
        invalid_timing_reasons.append("negative_stream_seconds")
    if (
        request_seconds is not None
        and stream_seconds is not None
        and stream_seconds > request_seconds + 1e-9
    ):
        invalid_timing_reasons.append("stream_seconds_exceeds_request")

    monotonic_elapsed_seconds: float | None = None
    if monotonic_started_ns is not None or monotonic_ended_ns is not None:
        if (
            not isinstance(monotonic_started_ns, int)
            or not isinstance(monotonic_ended_ns, int)
            or monotonic_ended_ns < monotonic_started_ns
        ):
            invalid_timing_reasons.append("invalid_monotonic_timestamp_pair")
        else:
            monotonic_elapsed_seconds = (
                monotonic_ended_ns - monotonic_started_ns
            ) / 1_000_000_000
            if request_seconds is not None and abs(
                monotonic_elapsed_seconds - request_seconds
            ) > max(0.05, 0.10 * max(request_seconds, 1e-9)):
                invalid_timing_reasons.append(
                    "adapter_request_duration_disagrees_with_outer_monotonic_clock"
                )

    usage_invalid_reasons: list[str] = []
    if prompt_tokens is None or prompt_tokens <= 0:
        usage_invalid_reasons.append("missing_or_nonpositive_prompt_tokens")
    if completion_tokens is None or completion_tokens <= 0:
        usage_invalid_reasons.append("missing_or_nonpositive_completion_tokens")
    if sequence_count != 1:
        usage_invalid_reasons.append("aggregate_usage_for_multiple_sequences")
    if not streaming:
        usage_invalid_reasons.append("nonstreaming_response_has_no_observed_ttft")

    post_ttft_seconds: float | None = None
    if request_seconds is not None and ttft_seconds is not None:
        post_ttft_seconds = request_seconds - ttft_seconds
    valid_post_ttft = _positive_interval(post_ttft_seconds)
    output_tps = (
        completion_tokens / valid_post_ttft
        if completion_tokens is not None
        and completion_tokens > 0
        and valid_post_ttft is not None
        and not invalid_timing_reasons
        and sequence_count == 1
        and streaming
        else None
    )
    prefill_interval = _positive_interval(ttft_seconds)
    exploratory_prefill_proxy = (
        prompt_tokens / prefill_interval
        if prompt_tokens is not None
        and prompt_tokens > 0
        and prefill_interval is not None
        and not invalid_timing_reasons
        and sequence_count == 1
        and streaming
        else None
    )
    prefill_proxy = (
        exploratory_prefill_proxy
        if cache["observed_state"] == "cache_miss_observed"
        else None
    )

    sse_invalid_reasons: list[str] = []
    if result.event_count < 8:
        sse_invalid_reasons.append("fewer_than_eight_content_events")
    measurable_generation = _positive_interval(generation_seconds)
    if measurable_generation is None or measurable_generation < 0.05:
        sse_invalid_reasons.append("missing_or_event_span_below_50ms")
    if completion_tokens is None or completion_tokens <= 0:
        sse_invalid_reasons.append("missing_or_nonpositive_completion_tokens")
    elif completion_tokens < 32:
        sse_invalid_reasons.append("fewer_than_32_completion_tokens")
    if invalid_timing_reasons:
        sse_invalid_reasons.append("inconsistent_request_timing")
    if sequence_count != 1:
        sse_invalid_reasons.append("aggregate_usage_for_multiple_sequences")
    if not streaming:
        sse_invalid_reasons.append("nonstreaming_response_has_no_sse_token_timing")
    sse_tps = (
        completion_tokens / measurable_generation
        if completion_tokens is not None
        and completion_tokens > 0
        and measurable_generation is not None
        and result.event_count >= 8
        and completion_tokens >= 32
        and measurable_generation >= 0.05
        and not invalid_timing_reasons
        and sequence_count == 1
        and streaming
        else None
    )
    exploratory_sse_tps = (
        completion_tokens / generation_seconds
        if completion_tokens is not None
        and completion_tokens > 0
        and generation_seconds is not None
        and generation_seconds > 0
        and result.event_count >= 2
        and not invalid_timing_reasons
        and sequence_count == 1
        and streaming
        else None
    )

    extreme_triggers: list[str] = []
    if output_tps is not None and output_tps >= EXTREME_OUTPUT_TPS:
        extreme_triggers.append("post_ttft_output_tps_at_least_1000")
    if (
        exploratory_prefill_proxy is not None
        and exploratory_prefill_proxy >= EXTREME_PREFILL_PROXY_TPS
    ):
        extreme_triggers.append("prefill_proxy_tps_at_least_1000000")
    if ttft_seconds is not None and 0 <= ttft_seconds <= EXTREME_LOW_TTFT_SECONDS:
        extreme_triggers.append("ttft_at_most_1ms")
    if sse_tps is not None and sse_tps >= EXTREME_OUTPUT_TPS:
        extreme_triggers.append("sse_chunk_span_proxy_at_least_1000")

    if invalid_timing_reasons or usage_invalid_reasons or sse_invalid_reasons:
        audit_classification = "invalid_or_partially_censored"
    elif extreme_triggers:
        audit_classification = "valid_extreme_keep_and_flag"
    else:
        audit_classification = "valid_ordinary"

    derived_monotonic: dict[str, int | None] = {
        "request_started_ns": monotonic_started_ns,
        "request_ended_ns": monotonic_ended_ns,
        "headers_observed_ns": None,
        "first_content_event_ns": None,
        "last_content_event_ns": None,
    }
    if isinstance(monotonic_started_ns, int):
        if headers_seconds is not None and headers_seconds >= 0:
            derived_monotonic["headers_observed_ns"] = monotonic_started_ns + round(
                headers_seconds * 1_000_000_000
            )
        if ttft_seconds is not None and ttft_seconds >= 0:
            first_ns = monotonic_started_ns + round(ttft_seconds * 1_000_000_000)
            derived_monotonic["first_content_event_ns"] = first_ns
            if generation_seconds is not None and generation_seconds >= 0:
                derived_monotonic["last_content_event_ns"] = first_ns + round(
                    generation_seconds * 1_000_000_000
                )

    return {
        "request_seconds": request_seconds,
        "headers_seconds": headers_seconds,
        "ttft_seconds": ttft_seconds,
        "generation_seconds": generation_seconds,
        "stream_seconds": stream_seconds,
        "event_count": int(result.event_count),
        "sequence_count": sequence_count,
        "streaming": streaming,
        "first_event_kind": result.first_event_kind,
        "monotonic_timestamps_ns": derived_monotonic,
        "outer_monotonic_elapsed_seconds": monotonic_elapsed_seconds,
        "post_ttft_seconds": post_ttft_seconds,
        "output_tokens_per_second": output_tps,
        "output_tokens_per_second_metric_kind": (
            "completion_tokens_over_request_minus_ttft_end_to_end_proxy"
        ),
        "sse_chunk_span_output_tokens_per_second_proxy": sse_tps,
        "exploratory_sse_chunk_span_output_tokens_per_second_proxy": exploratory_sse_tps,
        "sse_chunk_span_metric_kind": (
            "completion_tokens_over_first_to_last_sse_content_event_span; "
            "not direct decode throughput"
        ),
        "sse_chunk_span_valid": not sse_invalid_reasons,
        "sse_chunk_span_endpoint_comparison_eligible": (
            not sse_invalid_reasons
            and completion_tokens is not None
            and completion_tokens >= 128
        ),
        "sse_chunk_span_invalidity_reasons": sse_invalid_reasons,
        "prompt_tokens_per_second_to_first_token": prefill_proxy,
        "exploratory_prompt_tokens_per_second_to_first_token": exploratory_prefill_proxy,
        "prefill_metric_kind": "end_to_end_proxy_not_server_compute",
        "prefill_headline_eligible": prefill_proxy is not None,
        "cache_observation": cache,
        "timing_invalidity_reasons": invalid_timing_reasons,
        "usage_invalidity_reasons": usage_invalid_reasons,
        "metric_audit_classification": audit_classification,
        "extreme_metric_triggers": extreme_triggers,
    }


def audit_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project a sanitized request record into a traceable outlier row."""

    timing = record.get("timing") if isinstance(record.get("timing"), Mapping) else {}
    usage = parse_token_usage(record.get("usage"))
    cache_counters_present = any(
        key in usage
        for key in (
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "cached_tokens",
        )
    )
    cache_read_tokens = max(
        int(usage.get("cache_read_input_tokens") or 0),
        int(usage.get("cached_tokens") or 0),
    )
    cache_state = (
        "cache_hit_observed"
        if cache_read_tokens > 0
        else (
            "cache_miss_observed" if cache_counters_present else "not_reported_unknown"
        )
    )
    invalid = list(timing.get("timing_invalidity_reasons") or [])
    invalid.extend(timing.get("usage_invalidity_reasons") or [])
    invalid.extend(timing.get("sse_chunk_span_invalidity_reasons") or [])
    triggers = list(timing.get("extreme_metric_triggers") or [])
    classification = str(
        timing.get("metric_audit_classification")
        or (
            "not_applicable_failure"
            if record.get("status") != "success"
            else "unaudited"
        )
    )
    return {
        "schema_version": "do_metric_outlier_audit_v1",
        "request_id": record.get("request_id"),
        "campaign_id": record.get("campaign_id"),
        "model_id": record.get("model_id"),
        "shape": record.get("shape"),
        "phase": record.get("phase"),
        "status": record.get("status"),
        "http_status": record.get("http_status"),
        "usage": usage,
        "cache_state": cache_state,
        "cache_read_tokens": cache_read_tokens if cache_counters_present else None,
        "event_count": timing.get(
            "event_count", (record.get("stream") or {}).get("event_count")
        ),
        "sequence_count": timing.get("sequence_count"),
        "streaming": timing.get("streaming"),
        "ttft_seconds": timing.get("ttft_seconds"),
        "request_seconds": timing.get("request_seconds"),
        "post_ttft_seconds": timing.get("post_ttft_seconds"),
        "generation_seconds": timing.get("generation_seconds"),
        "output_tokens_per_second": timing.get("output_tokens_per_second"),
        "output_tokens_per_second_metric_kind": timing.get(
            "output_tokens_per_second_metric_kind"
        ),
        "sse_chunk_span_output_tokens_per_second_proxy": timing.get(
            "sse_chunk_span_output_tokens_per_second_proxy"
        ),
        "prompt_tokens_per_second_to_first_token": timing.get(
            "prompt_tokens_per_second_to_first_token"
        ),
        "monotonic_timestamps_ns": timing.get("monotonic_timestamps_ns"),
        "classification": classification,
        "invalidity_reasons": sorted(set(str(item) for item in invalid)),
        "extreme_metric_triggers": sorted(set(str(item) for item in triggers)),
        "trimmed_or_winsorized": False,
        "audit_contract": (
            "Every request is retained. Valid extremes are kept and flagged; invalid "
            "rates are censored to null, never silently trimmed or replaced."
        ),
    }
