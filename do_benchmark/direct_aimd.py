"""Direct, resumable DigitalOcean saturation benchmark.

The runner deliberately has one process, one endpoint active at a time, and
three append-only JSONL journals.  Request arrivals are open-loop: every
arrival is scheduled from the epoch clock before it waits for the independent
concurrency semaphore.  This prevents slow calls from silently reducing the
offered load (coordinated omission).

Only hashes and measurements are persisted.  Prompts, model output, response
bodies, credentials, and raw response headers never enter these journals.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

import httpx

from do_benchmark.core import (
    BenchmarkTask,
    JsonlJournal,
    MODEL_BY_ID,
    DIGITALOCEAN_HOSTED_MODEL_IDS,
    ModelSpec,
    StreamResult,
    require_digitalocean_hosted_models,
    _context_task,
    _controlled_output_task,
    canonical_json,
    percentile,
    parse_token_usage,
    score_result,
    stable_hash,
    stream_chat_completion,
    utc_now,
)
from do_benchmark.credentials import digitalocean_credentials
from do_benchmark.timing_audit import audit_row, timing_evidence


REQUEST_SCHEMA = "do_direct_request_v1"
EPOCH_SCHEMA = "do_direct_epoch_v1"
RESERVATION_SCHEMA = "do_direct_reservation_v1"
MANIFEST_SCHEMA = "do_direct_campaign_v1"
SHAPES = ("short_short", "input32k_short", "short_long", "mixed")
AIMD_SHAPES = frozenset(SHAPES)
AIMD_TASK_RECIPE_VERSION = "direct-aimd-uncached-prefix-v2"
AIMD_METRIC_RECIPE_VERSION = "conservative-timing-cache-aware-v2"

RequestExecutor = Callable[[str, BenchmarkTask, int], Awaitable[StreamResult]]


@dataclass(frozen=True)
class DirectConfig:
    output_dir: Path
    model_ids: tuple[str, ...]
    seed: int = 20260823
    epoch_seconds: float = 5.0
    concurrency_ceiling: int = 128
    initial_rps: float = 2.0
    additive_step_rps: float = 1.0
    maximum_rps: float = 32.0
    input_initial_rps: float = 0.4
    input_additive_step_rps: float = 0.4
    input_maximum_rps: float = 2.4
    rapid_bracket_epochs: int = 5
    heavy_rapid_bracket_epochs: int = 3
    additive_aimd_epochs: int = 1
    baseline_samples: int = 1
    output_initial_rps: float = 0.4
    output_additive_step_rps: float = 0.2
    output_maximum_rps: float = 1.6
    mixed_initial_rps: float = 0.4
    mixed_additive_step_rps: float = 0.4
    mixed_maximum_rps: float = 3.2
    input_tokens: int = 32_000
    long_output_words: int = 1_024
    short_max_output_tokens: int = 64
    long_max_output_tokens: int = 2_048
    mixed_max_output_tokens: int = 1_024
    request_timeout_seconds: float = 120.0
    max_cost_usd: float = 200.0
    prior_cost_usd: float = 0.0
    stop_launch_at: datetime | None = None

    def validate(self) -> None:
        unknown = sorted(set(self.model_ids) - MODEL_BY_ID.keys())
        if unknown:
            raise ValueError(f"unknown DigitalOcean models: {', '.join(unknown)}")
        require_digitalocean_hosted_models(self.model_ids)
        if not self.model_ids:
            raise ValueError("at least one model is required")
        if self.epoch_seconds <= 0:
            raise ValueError("epoch_seconds must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.concurrency_ceiling < 1:
            raise ValueError("concurrency_ceiling must be positive")
        if self.initial_rps <= 0 or self.additive_step_rps <= 0:
            raise ValueError("initial and additive rates must be positive")
        if self.maximum_rps < self.initial_rps:
            raise ValueError("maximum_rps must be >= initial_rps")
        if self.input_initial_rps <= 0 or self.input_additive_step_rps <= 0:
            raise ValueError("input AIMD rates must be positive")
        if self.input_maximum_rps < self.input_initial_rps:
            raise ValueError("input_maximum_rps must be >= input_initial_rps")
        if min(self.rapid_bracket_epochs, self.heavy_rapid_bracket_epochs) < 1:
            raise ValueError("rapid bracket epoch counts must be positive")
        if self.additive_aimd_epochs < 1:
            raise ValueError("invalid AIMD epoch counts")
        if self.baseline_samples < 1:
            raise ValueError("baseline_samples must be positive")
        for label, initial, step, maximum in (
            ("short", self.initial_rps, self.additive_step_rps, self.maximum_rps),
            (
                "input",
                self.input_initial_rps,
                self.input_additive_step_rps,
                self.input_maximum_rps,
            ),
            (
                "output",
                self.output_initial_rps,
                self.output_additive_step_rps,
                self.output_maximum_rps,
            ),
            (
                "mixed",
                self.mixed_initial_rps,
                self.mixed_additive_step_rps,
                self.mixed_maximum_rps,
            ),
        ):
            if initial <= 0 or step <= 0 or maximum < initial:
                raise ValueError(f"invalid {label} AIMD rates")
        if self.input_tokens < 1 or self.long_output_words < 1:
            raise ValueError("input/output targets must be positive")
        if (
            min(
                self.short_max_output_tokens,
                self.long_max_output_tokens,
                self.mixed_max_output_tokens,
            )
            < 1
        ):
            raise ValueError("output token ceilings must be positive")
        if self.max_cost_usd <= 0 or self.prior_cost_usd < 0:
            raise ValueError("invalid cost envelope")
        if self.prior_cost_usd > self.max_cost_usd:
            raise ValueError("prior cost already exceeds the campaign cap")
        if self.stop_launch_at is not None and self.stop_launch_at.tzinfo is None:
            raise ValueError("stop_launch_at must be timezone-aware")

    def identity_payload(self) -> dict[str, Any]:
        """Configuration that must stay identical when resuming a directory."""

        return {
            "schema_version": MANIFEST_SCHEMA,
            "task_recipe_version": AIMD_TASK_RECIPE_VERSION,
            "metric_recipe_version": AIMD_METRIC_RECIPE_VERSION,
            "models": list(self.model_ids),
            "seed": self.seed,
            "epoch_seconds": self.epoch_seconds,
            "concurrency_ceiling": self.concurrency_ceiling,
            "initial_rps": self.initial_rps,
            "additive_step_rps": self.additive_step_rps,
            "maximum_rps": self.maximum_rps,
            "input_initial_rps": self.input_initial_rps,
            "input_additive_step_rps": self.input_additive_step_rps,
            "input_maximum_rps": self.input_maximum_rps,
            "rapid_bracket_epochs": self.rapid_bracket_epochs,
            "heavy_rapid_bracket_epochs": self.heavy_rapid_bracket_epochs,
            "additive_aimd_epochs": self.additive_aimd_epochs,
            "baseline_samples": self.baseline_samples,
            "output_initial_rps": self.output_initial_rps,
            "output_additive_step_rps": self.output_additive_step_rps,
            "output_maximum_rps": self.output_maximum_rps,
            "mixed_initial_rps": self.mixed_initial_rps,
            "mixed_additive_step_rps": self.mixed_additive_step_rps,
            "mixed_maximum_rps": self.mixed_maximum_rps,
            "input_tokens": self.input_tokens,
            "long_output_words": self.long_output_words,
            "short_max_output_tokens": self.short_max_output_tokens,
            "long_max_output_tokens": self.long_max_output_tokens,
            "mixed_max_output_tokens": self.mixed_max_output_tokens,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_cost_usd": self.max_cost_usd,
            "prior_cost_usd": self.prior_cost_usd,
        }


@dataclass
class ControllerState:
    offered_rps: float
    best_healthy_rps: float = 0.0
    consecutive_unhealthy: int = 0
    saturation_rps: float | None = None


def rapid_bracket_transition(
    state: ControllerState,
    *,
    healthy: bool,
    additive_step_rps: float,
    maximum_rps: float,
    minimum_rps: float = 0.25,
) -> ControllerState:
    """Double healthy bracket rates; confirm congestion before 0.5 decrease."""

    if healthy:
        return ControllerState(
            offered_rps=min(
                maximum_rps,
                max(state.offered_rps * 2.0, state.offered_rps + additive_step_rps),
            ),
            best_healthy_rps=max(state.best_healthy_rps, state.offered_rps),
            consecutive_unhealthy=0,
            saturation_rps=state.saturation_rps,
        )
    streak = state.consecutive_unhealthy + 1
    if streak < 2:
        # A single bad epoch can be transient. Repeat the exact offered rate so
        # "saturation" always means two consecutive breaches.
        return ControllerState(
            offered_rps=state.offered_rps,
            best_healthy_rps=state.best_healthy_rps,
            consecutive_unhealthy=streak,
            saturation_rps=state.saturation_rps,
        )
    return ControllerState(
        offered_rps=max(minimum_rps, state.offered_rps * 0.5),
        best_healthy_rps=state.best_healthy_rps,
        consecutive_unhealthy=streak,
        saturation_rps=state.saturation_rps or state.offered_rps,
    )


def additive_aimd_transition(
    state: ControllerState,
    *,
    healthy: bool,
    additive_step_rps: float,
    maximum_rps: float,
    minimum_rps: float = 0.25,
) -> ControllerState:
    """Classic additive-increase / 0.5 multiplicative-decrease transition."""

    if healthy:
        return ControllerState(
            offered_rps=min(maximum_rps, state.offered_rps + additive_step_rps),
            best_healthy_rps=max(state.best_healthy_rps, state.offered_rps),
            consecutive_unhealthy=0,
            saturation_rps=state.saturation_rps,
        )
    streak = state.consecutive_unhealthy + 1
    return ControllerState(
        offered_rps=max(minimum_rps, state.offered_rps * 0.5),
        best_healthy_rps=state.best_healthy_rps,
        consecutive_unhealthy=streak,
        saturation_rps=(
            state.saturation_rps
            if state.saturation_rps is not None or streak < 2
            else state.offered_rps
        ),
    )


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path, identity_key: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"torn JSONL journal {path}:{line_number}: {exc}"
                ) from exc
            identity = record.get(identity_key)
            if identity:
                records[str(identity)] = record
    return records


def _task_payload(task: BenchmarkTask, max_output_tokens: int) -> dict[str, Any]:
    return {
        "messages": task.messages,
        "tools": task.tools,
        "tool_choice": task.tool_choice,
        "response_format": task.response_format,
        "parameters": task.parameters,
        "max_tokens": max_output_tokens,
        "stream": True,
        "temperature": 0,
    }


def conservative_request_cost(
    spec: ModelSpec,
    task: BenchmarkTask,
    max_output_tokens: int,
) -> tuple[float, int]:
    """Return a deliberately padded pre-send token-cost reservation."""

    payload_bytes = len(
        canonical_json(_task_payload(task, max_output_tokens)).encode("utf-8")
    )
    planned = int(task.metadata.get("planned_input_tokens") or 0)
    # All generated requests are ASCII. One token per UTF-8 byte is a strict
    # tokenizer-independent content bound; 512 additional tokens cover chat
    # framing that the provider may add outside the serialized request body.
    # Server-reported usage replaces this reservation after success.
    prompt_token_reserve = max(payload_bytes + 512, math.ceil(planned * 1.5))
    cost = (
        prompt_token_reserve * spec.input_usd_per_million
        + max_output_tokens * spec.output_usd_per_million
    ) / 1_000_000
    return cost, prompt_token_reserve


def _scheduled_arrivals(rate: float, seconds: float) -> int:
    return max(1, math.floor(rate * seconds))


def _maximum_aimd_arrivals(
    *,
    initial_rps: float,
    additive_step_rps: float,
    maximum_rps: float,
    rapid_bracket_epochs: int,
    additive_aimd_epochs: int,
    epoch_seconds: float,
    baseline_samples: int,
) -> int:
    """Upper-bound arrivals over every possible controller path.

    The all-healthy path maximizes rapid-bracket and additive rates. We then
    include three maximum-rate confirmations, two serial separators, one full
    maximum-rate overload even when the actual run may be right-censored, and
    one half-maximum recovery. This intentionally overstates the live plan.
    """

    rate = initial_rps
    total = baseline_samples
    for _ in range(rapid_bracket_epochs):
        total += _scheduled_arrivals(rate, epoch_seconds)
        rate = min(maximum_rps, max(rate * 2.0, rate + additive_step_rps))
    for _ in range(additive_aimd_epochs):
        total += _scheduled_arrivals(rate, epoch_seconds)
        rate = min(maximum_rps, rate + additive_step_rps)
    total += 3 * _scheduled_arrivals(maximum_rps, epoch_seconds)
    total += 2  # serial confirmation separators
    total += _scheduled_arrivals(maximum_rps, epoch_seconds)  # overload upper bound
    total += _scheduled_arrivals(maximum_rps * 0.5, epoch_seconds)
    return total


def preflight_worst_case_cost(config: DirectConfig) -> dict[str, Any]:
    """Conservative full-schedule reservation before any credential is loaded."""

    short_requests = _maximum_aimd_arrivals(
        initial_rps=config.initial_rps,
        additive_step_rps=config.additive_step_rps,
        maximum_rps=config.maximum_rps,
        rapid_bracket_epochs=config.rapid_bracket_epochs,
        additive_aimd_epochs=config.additive_aimd_epochs,
        epoch_seconds=config.epoch_seconds,
        baseline_samples=config.baseline_samples,
    )
    input_requests = _maximum_aimd_arrivals(
        initial_rps=config.input_initial_rps,
        additive_step_rps=config.input_additive_step_rps,
        maximum_rps=config.input_maximum_rps,
        rapid_bracket_epochs=config.rapid_bracket_epochs,
        additive_aimd_epochs=config.additive_aimd_epochs,
        epoch_seconds=config.epoch_seconds,
        baseline_samples=config.baseline_samples,
    )
    output_requests = _maximum_aimd_arrivals(
        initial_rps=config.output_initial_rps,
        additive_step_rps=config.output_additive_step_rps,
        maximum_rps=config.output_maximum_rps,
        rapid_bracket_epochs=config.heavy_rapid_bracket_epochs,
        additive_aimd_epochs=config.additive_aimd_epochs,
        epoch_seconds=config.epoch_seconds,
        baseline_samples=config.baseline_samples,
    )
    mixed_requests = _maximum_aimd_arrivals(
        initial_rps=config.mixed_initial_rps,
        additive_step_rps=config.mixed_additive_step_rps,
        maximum_rps=config.mixed_maximum_rps,
        rapid_bracket_epochs=config.heavy_rapid_bracket_epochs,
        additive_aimd_epochs=config.additive_aimd_epochs,
        epoch_seconds=config.epoch_seconds,
        baseline_samples=max(config.baseline_samples, 5),
    )
    per_model: dict[str, dict[str, Any]] = {}
    total = 0.0
    for model_id in config.model_ids:
        spec = MODEL_BY_ID[model_id]
        short_cost, _ = conservative_request_cost(
            spec,
            make_task(
                shape="short_short",
                ordinal=999_999_999_990,
                input_tokens=config.input_tokens,
                long_output_words=config.long_output_words,
            ),
            config.short_max_output_tokens,
        )
        input_cost, _ = conservative_request_cost(
            spec,
            make_task(
                shape="input32k_short",
                ordinal=999_999_999_991,
                input_tokens=config.input_tokens,
                long_output_words=config.long_output_words,
            ),
            config.short_max_output_tokens,
        )
        output_cost, _ = conservative_request_cost(
            spec,
            make_task(
                shape="short_long",
                ordinal=999_999_999_992,
                input_tokens=config.input_tokens,
                long_output_words=config.long_output_words,
            ),
            config.long_max_output_tokens,
        )
        mixed_cost = max(
            conservative_request_cost(
                spec,
                make_task(
                    shape="mixed",
                    ordinal=999_999_999_995 + selector,
                    input_tokens=config.input_tokens,
                    long_output_words=config.long_output_words,
                ),
                config.mixed_max_output_tokens,
            )[0]
            for selector in range(5)
        )
        shapes = {
            "short_short": short_requests * short_cost,
            "input32k_short": input_requests * input_cost,
            "short_long": output_requests * output_cost,
            "mixed": mixed_requests * mixed_cost,
        }
        model_total = sum(shapes.values())
        per_model[model_id] = {
            "maximum_scheduled_requests": {
                "short_short": short_requests,
                "input32k_short": input_requests,
                "short_long": output_requests,
                "mixed": mixed_requests,
            },
            "worst_case_reserved_cost_usd": model_total,
            "shape_cost_usd": shapes,
        }
        total += model_total
    total_with_prior = config.prior_cost_usd + total
    return {
        "method": (
            "all-healthy maximum-rate path plus maximum-rate overload upper bound; "
            "each request uses the padded pre-send token reservation"
        ),
        "prior_cost_usd": config.prior_cost_usd,
        "new_campaign_worst_case_reserved_cost_usd": total,
        "total_worst_case_exposure_usd": total_with_prior,
        "max_cost_usd": config.max_cost_usd,
        "remaining_margin_usd": config.max_cost_usd - total_with_prior,
        "passes": total_with_prior <= config.max_cost_usd + 1e-12,
        "per_model": per_model,
    }


def _short_task(ordinal: int) -> BenchmarkTask:
    nonce = stable_hash({"ordinal": ordinal}, prefix="n-")
    return BenchmarkTask(
        task_id=f"direct-short-{ordinal}",
        family="direct_short_exact",
        context_bucket="short",
        output_bucket="short",
        messages=[
            {
                "role": "user",
                "content": f"Ignore batch marker {nonce}. Return only the exact text LOAD-OK",
            }
        ],
        expected={"kind": "exact_text", "value": "LOAD-OK"},
        metadata={"cache_variant": nonce, "planned_input_tokens": 24},
    )


def _structured_mixed_task(ordinal: int) -> BenchmarkTask:
    marker = f"M{ordinal:06d}"
    return BenchmarkTask(
        task_id=f"direct-mixed-json-{ordinal}",
        family="direct_mixed_structured",
        context_bucket="short",
        output_bucket="short",
        messages=[
            {
                "role": "user",
                "content": (
                    "Return only JSON with keys marker, primes, and ok. "
                    f"marker must be {marker}; primes must be [2,3,5]; ok must be true."
                ),
            }
        ],
        expected={
            "kind": "json_exact",
            "value": {"marker": marker, "primes": [2, 3, 5], "ok": True},
        },
        response_format={"type": "json_object"},
        metadata={"planned_input_tokens": 48},
    )


def _tool_mixed_task(ordinal: int) -> BenchmarkTask:
    location = f"station-{ordinal:06d}"
    arguments = {"location": location, "unit": "celsius"}
    return BenchmarkTask(
        task_id=f"direct-mixed-tool-{ordinal}",
        family="direct_mixed_tool",
        context_bucket="short_tool",
        output_bucket="short",
        messages=[
            {
                "role": "user",
                "content": f"Use lookup_temperature for {location} in celsius.",
            }
        ],
        expected={
            "kind": "tool_exact",
            "value": {"name": "lookup_temperature", "arguments": arguments},
        },
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_temperature",
                    "description": "Look up synthetic station temperature.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location", "unit"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "lookup_temperature"}},
        metadata={"planned_input_tokens": 180},
    )


def make_task(
    *,
    shape: str,
    ordinal: int,
    input_tokens: int,
    long_output_words: int,
) -> BenchmarkTask:
    if shape == "short_short":
        task = _short_task(ordinal)
    elif shape == "input32k_short":
        task = _context_task(input_tokens, ordinal, chars_per_token=4.0)
        task.family = "direct_input32k_short"
    elif shape == "short_long":
        task = _controlled_output_task(long_output_words, ordinal)
        task.family = "direct_short_long"
    elif shape != "mixed":
        raise ValueError(f"unknown shape: {shape}")
    else:
        selector = ordinal % 5
        if selector == 0:
            task = _short_task(ordinal)
        elif selector == 1:
            task = _context_task(4_096, ordinal, chars_per_token=4.0)
            task.family = "direct_mixed_context4k"
        elif selector == 2:
            task = _controlled_output_task(512, ordinal)
            task.family = "direct_mixed_output512"
        elif selector == 3:
            task = _structured_mixed_task(ordinal)
        else:
            task = _tool_mixed_task(ordinal)

    # DigitalOcean open-source routes cache shared prompt prefixes
    # automatically.  AIMD is intended to measure uncached offered load, so a
    # request-unique nonce is deliberately the first prompt token.  Cache
    # counters remain evidence and are never inferred when absent.
    nonce = stable_hash({"shape": shape, "ordinal": ordinal}, prefix="UNCACHED-")
    first = task.messages[0]
    if isinstance(first.get("content"), str):
        first["content"] = f"{nonce} {first['content']}"
    task.metadata = {
        **task.metadata,
        "cache_intent": "uncached_randomized_prefix",
        "early_nonce_sha256": hashlib.sha256(nonce.encode("utf-8")).hexdigest(),
    }
    return task


def max_output_tokens_for_shape(config: DirectConfig, shape: str) -> int:
    if shape in {"short_short", "input32k_short"}:
        return config.short_max_output_tokens
    if shape == "short_long":
        return config.long_max_output_tokens
    return config.mixed_max_output_tokens


def _parsed_nonnegative_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def sanitized_header_signals(headers: Mapping[str, Any]) -> dict[str, Any]:
    request_id = headers.get("x-request-id")
    cf_ray = headers.get("cf-ray")
    return {
        "request_id_sha256": _sha256_text(str(request_id)) if request_id else None,
        "edge_id_sha256": _sha256_text(str(cf_ray)) if cf_ray else None,
        "rate_limit_limit_requests": _parsed_nonnegative_float(
            headers.get("x-ratelimit-limit-requests")
        ),
        "rate_limit_remaining_requests": _parsed_nonnegative_float(
            headers.get("x-ratelimit-remaining-requests")
        ),
        "retry_after_seconds": _parsed_nonnegative_float(headers.get("retry-after")),
    }


def _safe_rate(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def sanitized_success_row(
    *,
    campaign_id: str,
    request_id: str,
    epoch_id: str,
    model_id: str,
    shape: str,
    phase: str,
    task: BenchmarkTask,
    max_output_tokens: int,
    result: StreamResult,
    spec: ModelSpec,
    reserved_cost_usd: float,
    reserved_prompt_tokens: int,
    started_at: str,
    ended_at: str,
    scheduled_offset_seconds: float,
    schedule_lag_seconds: float,
    concurrency_ceiling: int,
    monotonic_started_ns: int | None = None,
    monotonic_ended_ns: int | None = None,
) -> dict[str, Any]:
    usage = parse_token_usage(result.usage)
    input_usage_complete = usage.get("prompt_tokens", 0) > 0
    output_usage_complete = usage.get("completion_tokens", 0) > 0
    usage_reported = input_usage_complete or output_usage_complete
    usage_complete_for_settlement = input_usage_complete and output_usage_complete
    actual_cost = (
        (
            usage["prompt_tokens"] * spec.input_usd_per_million
            + usage["completion_tokens"] * spec.output_usd_per_million
        )
        / 1_000_000
        if usage_complete_for_settlement
        else None
    )
    accounted_cost = actual_cost if actual_cost is not None else reserved_cost_usd
    quality = score_result(task, result)
    response_fingerprint = {
        "text": result.text,
        "reasoning": result.reasoning_text,
        "tool_calls": result.tool_calls,
    }
    measured_timing = timing_evidence(
        result,
        monotonic_started_ns=monotonic_started_ns,
        monotonic_ended_ns=monotonic_ended_ns,
        intended_cache_state=str(task.metadata.get("cache_intent") or "unknown"),
        sequence_count=int(task.parameters.get("n") or 1),
        streaming=task.parameters.get("stream") is not False,
    )
    return {
        "schema_version": REQUEST_SCHEMA,
        "campaign_id": campaign_id,
        "request_id": request_id,
        "epoch_id": epoch_id,
        "provider": "digitalocean-serverless-inference",
        "model_id": model_id,
        "shape": shape,
        "phase": phase,
        "task_id": task.task_id,
        "task_family": task.family,
        "request_payload_sha256": _sha256_text(
            canonical_json(_task_payload(task, max_output_tokens))
        ),
        "request_payload_bytes": len(
            canonical_json(_task_payload(task, max_output_tokens)).encode("utf-8")
        ),
        "response_sha256": _sha256_text(canonical_json(response_fingerprint)),
        "response_text_bytes": len(result.text.encode("utf-8")),
        "reasoning_text_bytes": len(result.reasoning_text.encode("utf-8")),
        "tool_call_count": len(result.tool_calls),
        "requested_max_output_tokens": max_output_tokens,
        "provider_send_attempted": True,
        "started_at": started_at,
        "ended_at": ended_at,
        "status": "success",
        "http_status": result.status_code,
        "finish_reason": result.finish_reason,
        "usage": usage,
        "usage_reported": usage_reported,
        "input_usage_complete": input_usage_complete,
        "output_usage_complete": output_usage_complete,
        "usage_complete_for_settlement": usage_complete_for_settlement,
        "timing": measured_timing,
        "stream": {
            "event_count": result.event_count,
            "first_event_kind": result.first_event_kind,
        },
        "header_signals": sanitized_header_signals(result.response_headers),
        "quality_score": float(quality["quality_score"]),
        "score_kind": str(quality["score_kind"]),
        "worst_case_reserved_cost_usd": reserved_cost_usd,
        "reserved_prompt_tokens": reserved_prompt_tokens,
        "estimated_cost_usd": (actual_cost),
        "accounted_cost_usd": accounted_cost,
        "load": {
            "arrival_mode": "open_loop",
            "scheduled_offset_seconds": scheduled_offset_seconds,
            "schedule_lag_seconds": schedule_lag_seconds,
            "concurrency_ceiling": concurrency_ceiling,
        },
    }


def sanitized_failure_row(
    *,
    campaign_id: str,
    request_id: str,
    epoch_id: str,
    model_id: str,
    shape: str,
    phase: str,
    task: BenchmarkTask,
    max_output_tokens: int,
    error: BaseException,
    reserved_cost_usd: float,
    reserved_prompt_tokens: int,
    started_at: str,
    ended_at: str,
    elapsed_seconds: float,
    scheduled_offset_seconds: float,
    schedule_lag_seconds: float,
    concurrency_ceiling: int,
    status: str = "error",
) -> dict[str, Any]:
    status_code = getattr(error, "status_code", None)
    return {
        "schema_version": REQUEST_SCHEMA,
        "campaign_id": campaign_id,
        "request_id": request_id,
        "epoch_id": epoch_id,
        "provider": "digitalocean-serverless-inference",
        "model_id": model_id,
        "shape": shape,
        "phase": phase,
        "task_id": task.task_id,
        "task_family": task.family,
        "request_payload_sha256": _sha256_text(
            canonical_json(_task_payload(task, max_output_tokens))
        ),
        "request_payload_bytes": len(
            canonical_json(_task_payload(task, max_output_tokens)).encode("utf-8")
        ),
        "requested_max_output_tokens": max_output_tokens,
        "provider_send_attempted": True,
        "started_at": started_at,
        "ended_at": ended_at,
        "status": status,
        "http_status": status_code if isinstance(status_code, int) else None,
        "error_type": type(error).__name__,
        "retry_after_seconds": _parsed_nonnegative_float(
            getattr(error, "retry_after", None)
        ),
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "timing": {"request_seconds": elapsed_seconds, "ttft_seconds": None},
        "quality_score": 0.0,
        "score_kind": str(task.expected.get("kind") or "unknown"),
        "worst_case_reserved_cost_usd": reserved_cost_usd,
        "reserved_prompt_tokens": reserved_prompt_tokens,
        # Failed or interrupted calls can have been partially billed. Retain
        # the full preflight reservation instead of assuming they were free.
        "estimated_cost_usd": None,
        "accounted_cost_usd": reserved_cost_usd,
        "load": {
            "arrival_mode": "open_loop",
            "scheduled_offset_seconds": scheduled_offset_seconds,
            "schedule_lag_seconds": schedule_lag_seconds,
            "concurrency_ceiling": concurrency_ceiling,
        },
    }


class BudgetLedger:
    """Crash-safe conservative budget using reservations before provider sends."""

    def __init__(
        self,
        *,
        path: Path,
        max_cost_usd: float,
        prior_cost_usd: float,
        terminal_rows: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.journal = JsonlJournal(path)
        self.max_cost_usd = max_cost_usd
        self.prior_cost_usd = prior_cost_usd
        self.reservations = _read_jsonl(path, "request_id")
        self.terminal_rows = terminal_rows
        self._lock = asyncio.Lock()

    def request_exposure(self, request_id: str) -> float:
        row = self.terminal_rows.get(request_id)
        if row is not None:
            accounted = float(row.get("accounted_cost_usd") or 0.0)
            usage = row.get("usage")
            prompt_tokens = (
                int(usage.get("prompt_tokens") or 0)
                if isinstance(usage, Mapping)
                else 0
            )
            completion_tokens = (
                int(usage.get("completion_tokens") or 0)
                if isinstance(usage, Mapping)
                else 0
            )
            if row.get("provider_send_attempted") is True and (
                prompt_tokens <= 0 or completion_tokens <= 0
            ):
                reservation = self.reservations.get(request_id)
                reserved = (
                    float(reservation.get("reserved_cost_usd") or 0.0)
                    if reservation is not None
                    else float(row.get("worst_case_reserved_cost_usd") or 0.0)
                )
                return max(accounted, reserved)
            return accounted
        reservation = self.reservations.get(request_id)
        if reservation is not None:
            return float(reservation.get("reserved_cost_usd") or 0.0)
        return 0.0

    @property
    def exposure_usd(self) -> float:
        identities = set(self.reservations) | set(self.terminal_rows)
        return self.prior_cost_usd + sum(
            self.request_exposure(item) for item in identities
        )

    async def reserve(
        self,
        *,
        campaign_id: str,
        request_id: str,
        epoch_id: str,
        model_id: str,
        shape: str,
        reserved_cost_usd: float,
        reserved_prompt_tokens: int,
        max_output_tokens: int,
    ) -> bool:
        async with self._lock:
            if request_id in self.reservations or request_id in self.terminal_rows:
                return False
            if self.exposure_usd + reserved_cost_usd > self.max_cost_usd + 1e-12:
                return False
            row = {
                "schema_version": RESERVATION_SCHEMA,
                "campaign_id": campaign_id,
                "request_id": request_id,
                "epoch_id": epoch_id,
                "model_id": model_id,
                "shape": shape,
                "reserved_at": utc_now(),
                "reserved_cost_usd": reserved_cost_usd,
                "reserved_prompt_tokens": reserved_prompt_tokens,
                "max_output_tokens": max_output_tokens,
            }
            await self.journal.append(row)
            self.reservations[request_id] = row
            return True

    async def settle(self, request_id: str, row: Mapping[str, Any]) -> None:
        async with self._lock:
            self.terminal_rows[request_id] = dict(row)


def _reconstructed_epoch_elapsed(
    rows: Sequence[Mapping[str, Any]],
    *,
    serial: bool,
    epoch_seconds: float,
) -> float:
    """Rebuild wall time from durable rows without hiding queue or drain time."""

    if serial:
        return max(
            1e-9,
            sum(
                float(row.get("load", {}).get("schedule_lag_seconds") or 0.0)
                + float(row.get("timing", {}).get("request_seconds") or 0.0)
                for row in rows
            ),
        )
    return max(
        epoch_seconds,
        max(
            (
                float(row.get("load", {}).get("scheduled_offset_seconds") or 0.0)
                + float(row.get("load", {}).get("schedule_lag_seconds") or 0.0)
                + float(row.get("timing", {}).get("request_seconds") or 0.0)
                for row in rows
            ),
            default=0.0,
        ),
    )


def assess_epoch(
    *,
    campaign_id: str,
    epoch_id: str,
    model_id: str,
    shape: str,
    phase: str,
    offered_rps: float,
    epoch_seconds: float,
    scheduled_requests: int,
    rows: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
    baseline_ttft_p95: float | None,
    baseline_latency_p95: float | None,
    baseline_quality_rate: float | None,
    max_observed_concurrency: int,
    valid_for_capacity: bool = True,
    extra_health_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    successes = [row for row in rows if row.get("status") == "success"]
    success_count = len(successes)
    quality_passes = sum(
        float(row.get("quality_score") or 0.0) >= 0.999999 for row in successes
    )
    http_429 = sum(row.get("http_status") == 429 for row in rows)
    http_5xx = sum(
        isinstance(row.get("http_status"), int) and int(row["http_status"]) >= 500
        for row in rows
    )
    timeout_types = {
        "ReadTimeout",
        "ConnectTimeout",
        "PoolTimeout",
        "TimeoutException",
        "TimeoutError",
    }
    timeouts = sum(str(row.get("error_type")) in timeout_types for row in rows)
    ttfts = [
        float(row["timing"]["ttft_seconds"])
        for row in successes
        if row.get("timing", {}).get("ttft_seconds") is not None
    ]
    latencies = [
        float(row["timing"]["request_seconds"])
        for row in successes
        if row.get("timing", {}).get("request_seconds") is not None
    ]
    lags = [
        float(row.get("load", {}).get("schedule_lag_seconds") or 0.0) for row in rows
    ]
    prompt_tokens = sum(
        int(row.get("usage", {}).get("prompt_tokens") or 0) for row in successes
    )
    output_tokens = sum(
        int(row.get("usage", {}).get("completion_tokens") or 0) for row in successes
    )
    provider_attempts = [row for row in rows if row.get("provider_send_attempted")]
    offered_input_token_reserve = sum(
        int(row.get("reserved_prompt_tokens") or 0) for row in provider_attempts
    )
    offered_output_token_ceiling = sum(
        int(row.get("requested_max_output_tokens") or 0) for row in provider_attempts
    )
    quality_successes = [
        row for row in successes if float(row.get("quality_score") or 0.0) >= 0.999999
    ]
    quality_input_tokens = sum(
        int(row.get("usage", {}).get("prompt_tokens") or 0) for row in quality_successes
    )
    quality_output_tokens = sum(
        int(row.get("usage", {}).get("completion_tokens") or 0)
        for row in quality_successes
    )
    total = len(rows)
    invalid_local_statuses = {
        "skipped_cost_cap",
        "skipped_deadline",
        "skipped_http_402_latch",
        "not_launched_interruption",
        "unknown_interrupted",
        "unknown_prior_reservation",
        "unknown_cancelled",
        "local_runner_error",
    }
    local_or_unknown_rows = [
        row
        for row in rows
        if not row.get("provider_send_attempted")
        or str(row.get("status")) in invalid_local_statuses
    ]
    billing_latch_rows = [row for row in rows if row.get("http_status") == 402]
    if local_or_unknown_rows or billing_latch_rows or total != scheduled_requests:
        valid_for_capacity = False
    success_rate = success_count / total if total else 0.0
    quality_rate = quality_passes / success_count if success_count else 0.0
    ttft_p95 = percentile(ttfts, 0.95)
    latency_p95 = percentile(latencies, 0.95)
    lag_p95 = percentile(lags, 0.95)
    reasons = list(extra_health_reasons)
    if total != scheduled_requests:
        reasons.append("terminal_rows_do_not_match_scheduled_arrivals")
    if success_rate < 0.99:
        reasons.append("success_rate_below_0.99")
    if total and (http_5xx + timeouts) / total > 0.01:
        reasons.append("combined_timeout_and_5xx_rate_above_0.01")
    if total and http_429 / total > 0.01:
        reasons.append("rate_limit_rate_above_0.01")
    if baseline_ttft_p95 is not None and ttft_p95 is not None:
        if ttft_p95 > 2.0 * baseline_ttft_p95:
            reasons.append("ttft_p95_above_2x_serial_baseline")
    if baseline_latency_p95 is not None and latency_p95 is not None:
        if latency_p95 > 2.0 * baseline_latency_p95:
            reasons.append("latency_p95_above_2x_serial_baseline")
    if offered_rps > 0 and lags:
        midpoint = max(1, len(lags) // 2)
        early = percentile(lags[:midpoint], 0.5) or 0.0
        late = percentile(lags[midpoint:], 0.5) or 0.0
        queue_growth_limit = max(0.25, 1.0 / offered_rps)
        if late - early > queue_growth_limit:
            reasons.append("sustained_arrival_queue_growth")
    if baseline_quality_rate is not None and successes:
        if quality_rate + 1e-12 < max(0.0, baseline_quality_rate - 0.05):
            reasons.append("quality_pass_rate_dropped_more_than_0.05_from_baseline")
    if not valid_for_capacity:
        reasons.append("epoch_invalid_for_capacity")
    if local_or_unknown_rows:
        reasons.append("local_nonsend_or_unknown_outcome_present")
    if billing_latch_rows:
        reasons.append("billing_or_credit_latch_present")
    # Preserve order but avoid duplicate reason labels.
    reasons = list(dict.fromkeys(reasons))
    elapsed_minutes = elapsed_seconds / 60.0 if elapsed_seconds > 0 else 0.0
    offered_arrival_rps = (
        scheduled_requests / epoch_seconds if epoch_seconds > 0 else 0.0
    )
    return {
        "schema_version": EPOCH_SCHEMA,
        "campaign_id": campaign_id,
        "epoch_id": epoch_id,
        "provider": "digitalocean-serverless-inference",
        "model_id": model_id,
        "shape": shape,
        "phase": phase,
        "offered_rps_target": offered_rps,
        "offered_rps_realized_schedule": offered_arrival_rps,
        "epoch_seconds": epoch_seconds,
        "elapsed_seconds_including_drain": elapsed_seconds,
        "scheduled_requests": scheduled_requests,
        "provider_send_attempts": len(provider_attempts),
        "completed_requests": total,
        "successes": success_count,
        "quality_passes": quality_passes,
        "http_429": http_429,
        "http_5xx": http_5xx,
        "timeouts": timeouts,
        "success_rate": success_rate,
        "success_rate_ci95_wilson": wilson_interval(success_count, total),
        "quality_pass_rate": quality_rate,
        "quality_pass_rate_ci95_wilson": wilson_interval(quality_passes, success_count),
        "achieved_rpm": total / elapsed_minutes if elapsed_minutes else 0.0,
        "successful_rpm": success_count / elapsed_minutes if elapsed_minutes else 0.0,
        "offered_input_tpm_conservative": (
            offered_input_token_reserve / elapsed_minutes if elapsed_minutes else 0.0
        ),
        "offered_output_token_ceiling_tpm": (
            offered_output_token_ceiling / elapsed_minutes if elapsed_minutes else 0.0
        ),
        "accepted_input_tpm": prompt_tokens / elapsed_minutes
        if elapsed_minutes
        else 0.0,
        "accepted_output_tpm": output_tokens / elapsed_minutes
        if elapsed_minutes
        else 0.0,
        "effective_input_tpm": prompt_tokens / elapsed_minutes
        if elapsed_minutes
        else 0.0,
        "effective_output_tpm": output_tokens / elapsed_minutes
        if elapsed_minutes
        else 0.0,
        "quality_adjusted_input_tpm": (
            quality_input_tokens / elapsed_minutes if elapsed_minutes else 0.0
        ),
        "quality_adjusted_output_tpm": (
            quality_output_tokens / elapsed_minutes if elapsed_minutes else 0.0
        ),
        "goodput_rpm": quality_passes / elapsed_minutes if elapsed_minutes else 0.0,
        "accounted_cost_usd": sum(
            float(row.get("accounted_cost_usd") or 0.0) for row in rows
        ),
        "http_status_distribution": {
            str(status): sum(row.get("http_status") == status for row in rows)
            for status in sorted(
                {
                    row.get("http_status")
                    for row in rows
                    if row.get("http_status") is not None
                }
            )
        },
        "request_status_distribution": {
            status: sum(str(row.get("status")) == status for row in rows)
            for status in sorted({str(row.get("status")) for row in rows})
        },
        "ttft_p50_seconds": percentile(ttfts, 0.50),
        "ttft_p90_seconds": percentile(ttfts, 0.90),
        "ttft_p95_seconds": ttft_p95,
        "latency_p50_seconds": percentile(latencies, 0.50),
        "latency_p90_seconds": percentile(latencies, 0.90),
        "latency_p95_seconds": latency_p95,
        "schedule_lag_p95_seconds": lag_p95,
        "max_observed_concurrency": max_observed_concurrency,
        "valid_for_capacity": valid_for_capacity,
        "healthy": not reasons,
        "health_reasons": reasons,
        "healthy_definition": {
            "success_rate_min": 0.99,
            "combined_timeout_5xx_rate_max": 0.01,
            "rate_limit_rate_max": 0.01,
            "ttft_p95_vs_serial_baseline_max": 2.0,
            "latency_p95_vs_serial_baseline_max": 2.0,
            "quality_pass_rate_drop_from_baseline_max": 0.05,
            "queue_growth_late_minus_early_median_seconds_max": (
                max(0.25, 1.0 / offered_rps) if offered_rps > 0 else None
            ),
        },
    }


class DirectAIMDCampaign:
    def __init__(self, config: DirectConfig) -> None:
        config.validate()
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.requests_path = self.output_dir / "requests.jsonl"
        self.epochs_path = self.output_dir / "epochs.jsonl"
        self.reservations_path = self.output_dir / "reservations.jsonl"
        self.outlier_audit_path = self.output_dir / "outlier-audit.jsonl"
        self.requests_journal = JsonlJournal(self.requests_path)
        self.epochs_journal = JsonlJournal(self.epochs_path)
        self.outlier_audit_journal = JsonlJournal(self.outlier_audit_path)
        self.request_rows = _read_jsonl(self.requests_path, "request_id")
        self.epoch_rows = _read_jsonl(self.epochs_path, "epoch_id")
        self.outlier_audit_rows = _read_jsonl(self.outlier_audit_path, "request_id")
        self.budget = BudgetLedger(
            path=self.reservations_path,
            max_cost_usd=config.max_cost_usd,
            prior_cost_usd=config.prior_cost_usd,
            terminal_rows=self.request_rows,
        )
        self.campaign_id = stable_hash(config.identity_payload(), prefix="do-direct-")
        self.preflight = preflight_worst_case_cost(config)
        if not self.preflight["passes"]:
            raise ValueError(
                "full direct campaign worst-case reservation exceeds the cumulative cap: "
                f"{self.preflight['total_worst_case_exposure_usd']:.6f} > "
                f"{self.config.max_cost_usd:.6f} USD"
            )
        self.account_blocked_402 = False
        self._write_or_validate_manifest()

    def _write_or_validate_manifest(self) -> None:
        path = self.output_dir / "manifest.json"
        identity = self.config.identity_payload()
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("campaign_id") != self.campaign_id:
                raise RuntimeError(
                    "output directory belongs to a different direct campaign configuration"
                )
            return
        order = list(self.config.model_ids)
        random.Random(self.config.seed).shuffle(order)
        manifest = {
            **identity,
            "campaign_id": self.campaign_id,
            "created_at": utc_now(),
            "model_order": order,
            "model_specs": [asdict(MODEL_BY_ID[model_id]) for model_id in order],
            "shapes": list(SHAPES),
            "aimd_shapes": sorted(AIMD_SHAPES),
            "preflight_worst_case_cost": self.preflight,
            "saturation_definition": "two consecutive valid unhealthy epochs",
            "aimd_confirmation_definition": (
                "highest observed healthy load followed by three healthy confirmation "
                "epochs separated by serial low-load sentinels; this is not sustained "
                "capacity without an independent soak"
            ),
            "arrival_contract": (
                "monotonic-clock open-loop arrivals; independent concurrency ceiling; "
                "all scheduled arrivals retained while calls queue"
            ),
            "sanitization": (
                "no credential, prompt, output, response body, or raw header is persisted; "
                "request and response content are represented only by SHA-256"
            ),
            "measurement_note": (
                "SSE content-event span is not server decode time. Headline per-request output "
                "rate is the conservative completion/(request-TTFT) end-to-end proxy. Prefill "
                "proxy is headline-eligible only for an observed cache miss. All invalid and "
                "extreme rows remain request-addressable in outlier-audit.jsonl."
            ),
            "default_nominal_measured_load_time_minutes": (
                len(self.config.model_ids)
                * (
                    2
                    * (
                        self.config.rapid_bracket_epochs
                        + self.config.additive_aimd_epochs
                        + 3
                        + 1
                        + 1
                    )
                    * self.config.epoch_seconds
                    + 2
                    * (
                        self.config.heavy_rapid_bracket_epochs
                        + self.config.additive_aimd_epochs
                        + 3
                        + 1
                        + 1
                    )
                    * self.config.epoch_seconds
                )
                / 60.0
            ),
        }
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _deadline_reached(self) -> bool:
        cutoff = self.config.stop_launch_at
        return cutoff is not None and datetime.now(timezone.utc) >= cutoff.astimezone(
            timezone.utc
        )

    def _epoch_id(
        self,
        *,
        model_id: str,
        shape: str,
        phase: str,
        ordinal: int,
        offered_rps: float,
    ) -> str:
        return stable_hash(
            {
                "campaign_id": self.campaign_id,
                "model_id": model_id,
                "shape": shape,
                "phase": phase,
                "ordinal": ordinal,
                "offered_rps": round(offered_rps, 9),
            },
            prefix="do-epoch-",
        )

    @staticmethod
    def _request_id(epoch_id: str, index: int) -> str:
        return stable_hash({"epoch_id": epoch_id, "index": index}, prefix="do-request-")

    def _task_ordinal(self, epoch_id: str, index: int) -> int:
        digest = hashlib.sha256(
            f"{self.config.seed}:{epoch_id}:{index}".encode()
        ).hexdigest()
        return int(digest[:10], 16)

    def _make_request_task(
        self, *, epoch_id: str, index: int, shape: str
    ) -> BenchmarkTask:
        ordinal = self._task_ordinal(epoch_id, index)
        if shape == "mixed":
            # Consecutive arrivals cycle deterministically through all five
            # heterogeneous task families. The mixed serial baseline has at
            # least five samples, so even a low-rate run observes every family.
            ordinal = ordinal - ordinal % 5 + index % 5
        return make_task(
            shape=shape,
            ordinal=ordinal,
            input_tokens=self.config.input_tokens,
            long_output_words=self.config.long_output_words,
        )

    async def _append_request(self, row: dict[str, Any]) -> None:
        request_id = str(row["request_id"])
        if request_id in self.request_rows:
            return
        await self.requests_journal.append(row)
        self.request_rows[request_id] = row
        await self.budget.settle(request_id, row)
        if request_id not in self.outlier_audit_rows:
            projected = audit_row(row)
            await self.outlier_audit_journal.append(projected)
            self.outlier_audit_rows[request_id] = projected

    async def _reconcile_outlier_audit(self) -> None:
        for request_id, row in self.request_rows.items():
            if request_id in self.outlier_audit_rows:
                continue
            projected = audit_row(row)
            await self.outlier_audit_journal.append(projected)
            self.outlier_audit_rows[request_id] = projected

    async def _append_unlaunched_row(
        self,
        *,
        request_id: str,
        epoch_id: str,
        model_id: str,
        shape: str,
        phase: str,
        task: BenchmarkTask,
        max_output_tokens: int,
        scheduled_offset: float,
        reason: str,
    ) -> dict[str, Any]:
        reservation = self.budget.reservations.get(request_id)
        reserved_cost = (
            float(reservation.get("reserved_cost_usd") or 0.0) if reservation else 0.0
        )
        reserved_tokens = (
            int(reservation.get("reserved_prompt_tokens") or 0) if reservation else 0
        )
        row = {
            "schema_version": REQUEST_SCHEMA,
            "campaign_id": self.campaign_id,
            "request_id": request_id,
            "epoch_id": epoch_id,
            "provider": "digitalocean-serverless-inference",
            "model_id": model_id,
            "shape": shape,
            "phase": phase,
            "task_id": task.task_id,
            "task_family": task.family,
            "request_payload_sha256": _sha256_text(
                canonical_json(_task_payload(task, max_output_tokens))
            ),
            "request_payload_bytes": len(
                canonical_json(_task_payload(task, max_output_tokens)).encode("utf-8")
            ),
            "started_at": utc_now(),
            "ended_at": utc_now(),
            "status": reason,
            "requested_max_output_tokens": max_output_tokens,
            "provider_send_attempted": reason
            in {
                "unknown_interrupted",
                "unknown_prior_reservation",
            },
            "http_status": None,
            "error_type": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "timing": {"request_seconds": 0.0, "ttft_seconds": None},
            "quality_score": 0.0,
            "score_kind": str(task.expected.get("kind") or "unknown"),
            "worst_case_reserved_cost_usd": reserved_cost,
            "reserved_prompt_tokens": reserved_tokens,
            "estimated_cost_usd": None,
            "accounted_cost_usd": reserved_cost,
            "load": {
                "arrival_mode": "open_loop",
                "scheduled_offset_seconds": scheduled_offset,
                "schedule_lag_seconds": 0.0,
                "concurrency_ceiling": self.config.concurrency_ceiling,
            },
        }
        await self._append_request(row)
        return row

    async def _run_epoch(
        self,
        executor: RequestExecutor,
        *,
        model_id: str,
        shape: str,
        phase: str,
        ordinal: int,
        offered_rps: float,
        scheduled_requests: int | None,
        serial: bool,
        baseline: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        epoch_id = self._epoch_id(
            model_id=model_id,
            shape=shape,
            phase=phase,
            ordinal=ordinal,
            offered_rps=offered_rps,
        )
        existing_epoch = self.epoch_rows.get(epoch_id)
        if existing_epoch is not None:
            return existing_epoch
        count = scheduled_requests
        if count is None:
            count = max(1, math.floor(offered_rps * self.config.epoch_seconds))
        request_ids = [self._request_id(epoch_id, index) for index in range(count)]
        # Build every deterministic payload before starting the epoch clock.
        # In particular, constructing hundreds of unique 32K prompts inside
        # arrival coroutines would block the event loop and manufacture client
        # schedule lag that is unrelated to the provider.
        request_tasks = [
            self._make_request_task(epoch_id=epoch_id, index=index, shape=shape)
            for index in range(count)
        ]
        terminal_count = sum(
            request_id in self.request_rows for request_id in request_ids
        )
        unsettled_count = sum(
            request_id in self.budget.reservations
            and request_id not in self.request_rows
            for request_id in request_ids
        )
        max_output_tokens = max_output_tokens_for_shape(self.config, shape)
        if unsettled_count or 0 < terminal_count < count:
            # Do not combine arrivals from before and after a process restart
            # into a fake continuous load epoch. Missing reserved calls become
            # unknown; missing unreserved calls are explicitly not launched.
            rows: list[dict[str, Any]] = []
            for index, request_id in enumerate(request_ids):
                if request_id in self.request_rows:
                    rows.append(self.request_rows[request_id])
                    continue
                task = request_tasks[index]
                reason = (
                    "unknown_interrupted"
                    if request_id in self.budget.reservations
                    else "not_launched_interruption"
                )
                rows.append(
                    await self._append_unlaunched_row(
                        request_id=request_id,
                        epoch_id=epoch_id,
                        model_id=model_id,
                        shape=shape,
                        phase=phase,
                        task=task,
                        max_output_tokens=max_output_tokens,
                        scheduled_offset=(0.0 if serial else index / offered_rps),
                        reason=reason,
                    )
                )
            elapsed = _reconstructed_epoch_elapsed(
                rows,
                serial=serial,
                epoch_seconds=self.config.epoch_seconds,
            )
            summary = assess_epoch(
                campaign_id=self.campaign_id,
                epoch_id=epoch_id,
                model_id=model_id,
                shape=shape,
                phase=phase,
                offered_rps=offered_rps,
                epoch_seconds=(elapsed if serial else self.config.epoch_seconds),
                scheduled_requests=count,
                rows=rows,
                elapsed_seconds=elapsed,
                baseline_ttft_p95=(baseline or {}).get("ttft_p95_seconds"),
                baseline_latency_p95=(baseline or {}).get("latency_p95_seconds"),
                baseline_quality_rate=(baseline or {}).get("quality_pass_rate"),
                max_observed_concurrency=0,
                valid_for_capacity=False,
                extra_health_reasons=("process_restart_split_epoch",),
            )
            await self.epochs_journal.append(summary)
            self.epoch_rows[epoch_id] = summary
            return summary

        if terminal_count == count:
            rows = [self.request_rows[request_id] for request_id in request_ids]
            elapsed = _reconstructed_epoch_elapsed(
                rows,
                serial=serial,
                epoch_seconds=self.config.epoch_seconds,
            )
            summary = assess_epoch(
                campaign_id=self.campaign_id,
                epoch_id=epoch_id,
                model_id=model_id,
                shape=shape,
                phase=phase,
                offered_rps=offered_rps,
                epoch_seconds=(elapsed if serial else self.config.epoch_seconds),
                scheduled_requests=count,
                rows=rows,
                elapsed_seconds=max(elapsed, 1e-9),
                baseline_ttft_p95=(baseline or {}).get("ttft_p95_seconds"),
                baseline_latency_p95=(baseline or {}).get("latency_p95_seconds"),
                baseline_quality_rate=(baseline or {}).get("quality_pass_rate"),
                max_observed_concurrency=max(
                    (
                        int(row.get("load", {}).get("observed_concurrency") or 0)
                        for row in rows
                    ),
                    default=0,
                ),
            )
            summary["arrival_mode"] = "serial" if serial else "open_loop"
            summary["reconstructed_from_terminal_request_rows"] = True
            summary["conservative_exposure_usd_after_epoch"] = self.budget.exposure_usd
            await self.epochs_journal.append(summary)
            self.epoch_rows[epoch_id] = summary
            return summary

        semaphore = asyncio.Semaphore(1 if serial else self.config.concurrency_ceiling)
        rows: list[dict[str, Any]] = []
        rows_lock = asyncio.Lock()
        active = 0
        max_active = 0
        active_lock = asyncio.Lock()
        epoch_started_perf = time.perf_counter()

        async def one(index: int, scheduled_offset: float) -> None:
            nonlocal active, max_active
            request_id = request_ids[index]
            existing = self.request_rows.get(request_id)
            if existing is not None:
                async with rows_lock:
                    rows.append(existing)
                return
            task = request_tasks[index]
            if not serial:
                delay = epoch_started_perf + scheduled_offset - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            scheduled_at = (
                time.perf_counter() if serial else epoch_started_perf + scheduled_offset
            )
            async with semaphore:
                queue_admitted_perf = time.perf_counter()
                schedule_lag = max(0.0, queue_admitted_perf - scheduled_at)
                if self._deadline_reached():
                    row = await self._append_unlaunched_row(
                        request_id=request_id,
                        epoch_id=epoch_id,
                        model_id=model_id,
                        shape=shape,
                        phase=phase,
                        task=task,
                        max_output_tokens=max_output_tokens,
                        scheduled_offset=scheduled_offset,
                        reason="skipped_deadline",
                    )
                    async with rows_lock:
                        rows.append(row)
                    return
                if self.account_blocked_402:
                    row = await self._append_unlaunched_row(
                        request_id=request_id,
                        epoch_id=epoch_id,
                        model_id=model_id,
                        shape=shape,
                        phase=phase,
                        task=task,
                        max_output_tokens=max_output_tokens,
                        scheduled_offset=scheduled_offset,
                        reason="skipped_http_402_latch",
                    )
                    async with rows_lock:
                        rows.append(row)
                    return
                spec = MODEL_BY_ID[model_id]
                reserved_cost, reserved_prompt_tokens = conservative_request_cost(
                    spec, task, max_output_tokens
                )
                reserved = await self.budget.reserve(
                    campaign_id=self.campaign_id,
                    request_id=request_id,
                    epoch_id=epoch_id,
                    model_id=model_id,
                    shape=shape,
                    reserved_cost_usd=reserved_cost,
                    reserved_prompt_tokens=reserved_prompt_tokens,
                    max_output_tokens=max_output_tokens,
                )
                if not reserved:
                    # A pre-existing reservation is never resent. In a newly
                    # budget-exhausted cell there is no reservation and no call.
                    reason = (
                        "unknown_prior_reservation"
                        if request_id in self.budget.reservations
                        else "skipped_cost_cap"
                    )
                    row = await self._append_unlaunched_row(
                        request_id=request_id,
                        epoch_id=epoch_id,
                        model_id=model_id,
                        shape=shape,
                        phase=phase,
                        task=task,
                        max_output_tokens=max_output_tokens,
                        scheduled_offset=scheduled_offset,
                        reason=reason,
                    )
                    async with rows_lock:
                        rows.append(row)
                    return
                provider_started_perf = time.perf_counter()
                provider_started_ns = time.perf_counter_ns()
                schedule_lag = max(0.0, provider_started_perf - scheduled_at)
                started_at = utc_now()
                async with active_lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    # This wraps the complete streaming lifecycle, including
                    # response drain. It is independent of client-library
                    # phase timeouts and therefore supplies a hard wall clock.
                    result = await asyncio.wait_for(
                        executor(model_id, task, max_output_tokens),
                        timeout=self.config.request_timeout_seconds,
                    )
                    provider_ended_ns = time.perf_counter_ns()
                    row = sanitized_success_row(
                        campaign_id=self.campaign_id,
                        request_id=request_id,
                        epoch_id=epoch_id,
                        model_id=model_id,
                        shape=shape,
                        phase=phase,
                        task=task,
                        max_output_tokens=max_output_tokens,
                        result=result,
                        spec=spec,
                        reserved_cost_usd=reserved_cost,
                        reserved_prompt_tokens=reserved_prompt_tokens,
                        started_at=started_at,
                        ended_at=utc_now(),
                        scheduled_offset_seconds=scheduled_offset,
                        schedule_lag_seconds=schedule_lag,
                        concurrency_ceiling=(
                            1 if serial else self.config.concurrency_ceiling
                        ),
                        monotonic_started_ns=provider_started_ns,
                        monotonic_ended_ns=provider_ended_ns,
                    )
                except asyncio.CancelledError as error:
                    row = sanitized_failure_row(
                        campaign_id=self.campaign_id,
                        request_id=request_id,
                        epoch_id=epoch_id,
                        model_id=model_id,
                        shape=shape,
                        phase=phase,
                        task=task,
                        max_output_tokens=max_output_tokens,
                        error=error,
                        reserved_cost_usd=reserved_cost,
                        reserved_prompt_tokens=reserved_prompt_tokens,
                        started_at=started_at,
                        ended_at=utc_now(),
                        elapsed_seconds=time.perf_counter() - provider_started_perf,
                        scheduled_offset_seconds=scheduled_offset,
                        schedule_lag_seconds=schedule_lag,
                        concurrency_ceiling=(
                            1 if serial else self.config.concurrency_ceiling
                        ),
                        status="unknown_cancelled",
                    )
                    await self._append_request(row)
                    raise
                except Exception as error:
                    if getattr(error, "status_code", None) == 402:
                        self.account_blocked_402 = True
                    row = sanitized_failure_row(
                        campaign_id=self.campaign_id,
                        request_id=request_id,
                        epoch_id=epoch_id,
                        model_id=model_id,
                        shape=shape,
                        phase=phase,
                        task=task,
                        max_output_tokens=max_output_tokens,
                        error=error,
                        reserved_cost_usd=reserved_cost,
                        reserved_prompt_tokens=reserved_prompt_tokens,
                        started_at=started_at,
                        ended_at=utc_now(),
                        elapsed_seconds=time.perf_counter() - provider_started_perf,
                        scheduled_offset_seconds=scheduled_offset,
                        schedule_lag_seconds=schedule_lag,
                        concurrency_ceiling=(
                            1 if serial else self.config.concurrency_ceiling
                        ),
                    )
                finally:
                    async with active_lock:
                        active -= 1
                await self._append_request(row)
                async with rows_lock:
                    rows.append(row)

        if serial:
            for index in range(count):
                await one(index, 0.0)
        else:
            await asyncio.gather(
                *(one(index, index / offered_rps) for index in range(count))
            )
        observed_drain_elapsed = time.perf_counter() - epoch_started_perf
        # A low-rate epoch can contain a single arrival at offset zero.  Its
        # service time is not the epoch's achieved-throughput denominator: the
        # predeclared open-loop arrival window still consumed epoch_seconds.
        # Use the longer of arrival window and drain, while retaining the raw
        # drain observation for audit.
        elapsed = (
            observed_drain_elapsed
            if serial
            else max(observed_drain_elapsed, self.config.epoch_seconds)
        )
        rows.sort(
            key=lambda item: float(
                item.get("load", {}).get("scheduled_offset_seconds") or 0.0
            )
        )
        summary = assess_epoch(
            campaign_id=self.campaign_id,
            epoch_id=epoch_id,
            model_id=model_id,
            shape=shape,
            phase=phase,
            offered_rps=offered_rps,
            epoch_seconds=(elapsed if serial else self.config.epoch_seconds),
            scheduled_requests=count,
            rows=rows,
            elapsed_seconds=elapsed,
            baseline_ttft_p95=(baseline or {}).get("ttft_p95_seconds"),
            baseline_latency_p95=(baseline or {}).get("latency_p95_seconds"),
            baseline_quality_rate=(baseline or {}).get("quality_pass_rate"),
            max_observed_concurrency=max_active,
        )
        summary["arrival_mode"] = "serial" if serial else "open_loop"
        summary["request_drain_elapsed_seconds_observed"] = observed_drain_elapsed
        summary["conservative_exposure_usd_after_epoch"] = self.budget.exposure_usd
        await self.epochs_journal.append(summary)
        self.epoch_rows[epoch_id] = summary
        return summary

    async def _baseline(
        self,
        executor: RequestExecutor,
        *,
        model_id: str,
        shape: str,
    ) -> dict[str, Any]:
        samples = (
            max(self.config.baseline_samples, 5)
            if shape == "mixed"
            else self.config.baseline_samples
        )
        return await self._run_epoch(
            executor,
            model_id=model_id,
            shape=shape,
            phase="serial_baseline",
            ordinal=0,
            offered_rps=1.0,
            scheduled_requests=samples,
            serial=True,
            baseline=None,
        )

    async def _run_aimd_shape(
        self,
        executor: RequestExecutor,
        *,
        model_id: str,
        shape: str,
    ) -> dict[str, Any]:
        baseline = await self._baseline(executor, model_id=model_id, shape=shape)
        if shape == "input32k_short":
            initial_rps = self.config.input_initial_rps
            additive_step_rps = self.config.input_additive_step_rps
            maximum_rps = self.config.input_maximum_rps
            bracket_epochs = self.config.rapid_bracket_epochs
        elif shape == "short_long":
            initial_rps = self.config.output_initial_rps
            additive_step_rps = self.config.output_additive_step_rps
            maximum_rps = self.config.output_maximum_rps
            bracket_epochs = self.config.heavy_rapid_bracket_epochs
        elif shape == "mixed":
            initial_rps = self.config.mixed_initial_rps
            additive_step_rps = self.config.mixed_additive_step_rps
            maximum_rps = self.config.mixed_maximum_rps
            bracket_epochs = self.config.heavy_rapid_bracket_epochs
        elif shape == "short_short":
            initial_rps = self.config.initial_rps
            additive_step_rps = self.config.additive_step_rps
            maximum_rps = self.config.maximum_rps
            bracket_epochs = self.config.rapid_bracket_epochs
        else:  # pragma: no cover - validated caller surface
            raise ValueError(f"unknown AIMD shape: {shape}")
        state = ControllerState(offered_rps=initial_rps)
        epochs: list[dict[str, Any]] = [baseline]
        local_stop = self._epoch_hit_local_stop(baseline)
        ordinal = 0
        valid_brackets = 0
        while valid_brackets < bracket_epochs and not local_stop:
            if self._deadline_reached() or self.account_blocked_402:
                break
            epoch = await self._run_epoch(
                executor,
                model_id=model_id,
                shape=shape,
                phase="rapid_bracket",
                ordinal=ordinal,
                offered_rps=state.offered_rps,
                scheduled_requests=None,
                serial=False,
                baseline=baseline,
            )
            ordinal += 1
            epochs.append(epoch)
            if not epoch.get("valid_for_capacity"):
                local_stop = self._epoch_hit_local_stop(epoch)
                continue
            valid_brackets += 1
            state = rapid_bracket_transition(
                state,
                healthy=bool(epoch.get("healthy")),
                additive_step_rps=additive_step_rps,
                maximum_rps=maximum_rps,
                minimum_rps=initial_rps * 0.5,
            )
            if state.saturation_rps is not None:
                break

        valid_aimd = 0
        while valid_aimd < self.config.additive_aimd_epochs and not local_stop:
            if self._deadline_reached() or self.account_blocked_402:
                break
            epoch = await self._run_epoch(
                executor,
                model_id=model_id,
                shape=shape,
                phase="additive_aimd",
                ordinal=ordinal,
                offered_rps=state.offered_rps,
                scheduled_requests=None,
                serial=False,
                baseline=baseline,
            )
            ordinal += 1
            epochs.append(epoch)
            if not epoch.get("valid_for_capacity"):
                local_stop = self._epoch_hit_local_stop(epoch)
                continue
            valid_aimd += 1
            state = additive_aimd_transition(
                state,
                healthy=bool(epoch.get("healthy")),
                additive_step_rps=additive_step_rps,
                maximum_rps=maximum_rps,
                minimum_rps=initial_rps * 0.5,
            )

        candidate = state.best_healthy_rps or initial_rps
        candidate_basis = (
            "highest_observed_healthy"
            if state.best_healthy_rps > 0
            else "initial_rate_fallback_no_healthy_epoch"
        )
        confirmation_epochs: list[dict[str, Any]] = []
        for confirmation_index in range(3):
            if local_stop or self._deadline_reached() or self.account_blocked_402:
                break
            confirmation = await self._run_epoch(
                executor,
                model_id=model_id,
                shape=shape,
                phase="confirmation",
                ordinal=confirmation_index,
                offered_rps=candidate,
                scheduled_requests=None,
                serial=False,
                baseline=baseline,
            )
            epochs.append(confirmation)
            confirmation_epochs.append(confirmation)
            local_stop = self._epoch_hit_local_stop(confirmation)
            if (
                confirmation_index < 2
                and not local_stop
                and not self._deadline_reached()
            ):
                gap = await self._run_epoch(
                    executor,
                    model_id=model_id,
                    shape=shape,
                    phase="confirmation_separator_serial",
                    ordinal=confirmation_index,
                    offered_rps=1.0,
                    scheduled_requests=1,
                    serial=True,
                    baseline=baseline,
                )
                epochs.append(gap)
                local_stop = self._epoch_hit_local_stop(gap)

        overload_rate: float | None = None
        if candidate < maximum_rps - 1e-12:
            proposed = state.saturation_rps or max(
                candidate + additive_step_rps, candidate * 2.0
            )
            proposed = min(maximum_rps, proposed)
            if proposed > candidate + 1e-12:
                overload_rate = proposed
        overload: dict[str, Any] | None = None
        recovery: dict[str, Any] | None = None
        if (
            overload_rate is not None
            and not local_stop
            and not self._deadline_reached()
            and not self.account_blocked_402
        ):
            overload = await self._run_epoch(
                executor,
                model_id=model_id,
                shape=shape,
                phase="post_confirmation_overload",
                ordinal=0,
                offered_rps=overload_rate,
                scheduled_requests=None,
                serial=False,
                baseline=baseline,
            )
            epochs.append(overload)
        if (
            overload is not None
            and not local_stop
            and not self._deadline_reached()
            and not self.account_blocked_402
        ):
            recovery = await self._run_epoch(
                executor,
                model_id=model_id,
                shape=shape,
                phase="post_overload_recovery",
                ordinal=0,
                offered_rps=candidate * 0.5,
                scheduled_requests=None,
                serial=False,
                baseline=baseline,
            )
            epochs.append(recovery)

        valid_confirmations = [
            epoch for epoch in confirmation_epochs if epoch.get("valid_for_capacity")
        ]
        confirmed = len(valid_confirmations) == 3 and all(
            bool(epoch.get("healthy")) for epoch in valid_confirmations
        )
        all_epoch_rows_valid = bool(epochs) and all(
            bool(epoch.get("valid_for_capacity")) for epoch in epochs
        )
        if len(valid_confirmations) != 3 or not all_epoch_rows_valid:
            shape_status = "incomplete"
        elif overload_rate is None:
            shape_status = "complete_right_censored"
        elif overload is None or recovery is None:
            shape_status = "incomplete"
        elif state.saturation_rps is None:
            shape_status = "complete_right_censored"
        else:
            shape_status = "complete"
        return {
            "model_id": model_id,
            "shape": shape,
            "status": shape_status,
            "highest_observed_healthy_rps": state.best_healthy_rps or None,
            "confirmation_target_rps": candidate,
            "candidate_confirmed_healthy_rps": candidate if confirmed else None,
            "candidate_basis": candidate_basis,
            "candidate_confirmed_three_separated_epochs": confirmed,
            "saturation_rps": state.saturation_rps,
            "saturation_definition_met": state.saturation_rps is not None,
            "right_censored_without_saturation": state.saturation_rps is None,
            "configured_maximum_rps": maximum_rps,
            "overload_tested": overload is not None,
            "overload_offered_rps": overload_rate,
            "overload_untested_reason": (
                "candidate_at_configured_maximum_rps"
                if overload_rate is None and candidate >= maximum_rps - 1e-12
                else (
                    "no_strictly_higher_rate_inside_configured_envelope"
                    if overload_rate is None
                    else None
                )
            ),
            "production_rate_recommendation": None,
            "production_rate_recommendation_reason": (
                "AIMD confirmation alone does not establish sustained production "
                "headroom; use a matched two-minute soak or report unverified."
            ),
            "post_overload_recovery_healthy": (
                bool(recovery.get("healthy")) if recovery is not None else None
            ),
            "epoch_ids": [str(epoch["epoch_id"]) for epoch in epochs],
        }

    @staticmethod
    def _epoch_hit_local_stop(epoch: Mapping[str, Any]) -> bool:
        statuses = epoch.get("request_status_distribution") or {}
        return any(
            int(statuses.get(status) or 0) > 0
            for status in (
                "skipped_cost_cap",
                "skipped_deadline",
                "skipped_http_402_latch",
            )
        )

    async def _run_with_executor(self, executor: RequestExecutor) -> dict[str, Any]:
        await self._reconcile_outlier_audit()
        started_at = utc_now()
        model_order = list(self.config.model_ids)
        random.Random(self.config.seed).shuffle(model_order)
        results: list[dict[str, Any]] = []
        for model_id in model_order:
            if self._deadline_reached() or self.account_blocked_402:
                results.append(
                    {
                        "model_id": model_id,
                        "status": (
                            "skipped_deadline"
                            if self._deadline_reached()
                            else "skipped_http_402_latch"
                        ),
                    }
                )
                continue
            shape_order = list(SHAPES)
            shape_rng = random.Random(
                self.config.seed
                + int(hashlib.sha256(model_id.encode()).hexdigest()[:8], 16)
            )
            shape_rng.shuffle(shape_order)
            shape_results = []
            for shape in shape_order:
                if self._deadline_reached() or self.account_blocked_402:
                    shape_results.append(
                        {"shape": shape, "status": "skipped_campaign_latch"}
                    )
                else:
                    shape_results.append(
                        await self._run_aimd_shape(
                            executor, model_id=model_id, shape=shape
                        )
                    )
            terminal_shape_statuses = {"complete", "complete_right_censored"}
            model_complete = len(shape_results) == len(SHAPES) and all(
                result.get("status") in terminal_shape_statuses
                for result in shape_results
            )
            model_right_censored = model_complete and any(
                result.get("status") == "complete_right_censored"
                for result in shape_results
            )
            results.append(
                {
                    "model_id": model_id,
                    "status": (
                        "complete_right_censored"
                        if model_right_censored
                        else ("complete" if model_complete else "incomplete")
                    ),
                    "endpoint_isolation": "no other endpoint active during these shapes",
                    "shape_order": shape_order,
                    "shapes": shape_results,
                }
            )
        terminal_model_statuses = {"complete", "complete_right_censored"}
        all_models_complete = bool(results) and all(
            result.get("status") in terminal_model_statuses for result in results
        )
        campaign_right_censored = all_models_complete and any(
            result.get("status") == "complete_right_censored" for result in results
        )
        summary = {
            "schema_version": "do_direct_summary_v1",
            "status": (
                "complete_right_censored"
                if campaign_right_censored
                else ("complete" if all_models_complete else "incomplete")
            ),
            "campaign_id": self.campaign_id,
            "started_at": started_at,
            "ended_at": utc_now(),
            "model_order": model_order,
            "endpoint_isolation": True,
            "max_cost_usd": self.config.max_cost_usd,
            "prior_cost_usd": self.config.prior_cost_usd,
            "conservative_exposure_usd": self.budget.exposure_usd,
            "http_402_latched": self.account_blocked_402,
            "preflight_worst_case_cost": self.preflight,
            "request_rows": len(self.request_rows),
            "outlier_audit_rows": len(self.outlier_audit_rows),
            "epoch_rows": len(self.epoch_rows),
            "models": results,
            "all_models_complete": all_models_complete,
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary

    async def run(self, executor: RequestExecutor | None = None) -> dict[str, Any]:
        if executor is not None:
            return await self._run_with_executor(executor)
        credentials = digitalocean_credentials()
        limits = httpx.Limits(
            max_connections=self.config.concurrency_ceiling,
            max_keepalive_connections=self.config.concurrency_ceiling,
        )
        timeout = httpx.Timeout(
            self.config.request_timeout_seconds,
            connect=min(30.0, self.config.request_timeout_seconds),
            read=self.config.request_timeout_seconds,
        )
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

            async def live_executor(
                model_id: str,
                task: BenchmarkTask,
                max_output_tokens: int,
            ) -> StreamResult:
                return await stream_chat_completion(
                    client,
                    api_base=credentials["api_base"],
                    api_key=credentials["api_key"],
                    model_id=model_id,
                    task=task,
                    safety_max_output_tokens=max_output_tokens,
                )

            return await self._run_with_executor(live_executor)


def default_model_ids() -> tuple[str, ...]:
    return DIGITALOCEAN_HOSTED_MODEL_IDS
