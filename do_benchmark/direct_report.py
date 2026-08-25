"""Offline, provider-safe analysis for the direct DigitalOcean benchmark.

The module deliberately accepts only already-recorded, secret-free evidence. It
normalizes the rapid breadth journal, direct open-loop AIMD journals, terminal
completion retries, and soak waves into one public schema.
Prompt text, model output, response bodies, credentials, and raw HTTP headers
are never copied to an output file.

Independent requests are the sampling unit for serial/breadth measurements.
For load experiments, each epoch is reduced to one unit before bootstrap
resampling; individual requests or output tokens are not treated as independent
load observations.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import (
    DIGITALOCEAN_HOSTED_MODEL_IDS,
    MODEL_BY_ID,
    canonical_json,
    parse_token_usage,
    percentile,
)
from .statistics import (
    bootstrap_interval,
    deterministic_seed,
    mean,
    nearest_rank,
    wilson_interval,
)


SCHEMA_VERSION = "digitalocean_direct_public_analysis_v3"
NORMALIZED_REQUEST_SCHEMA = "digitalocean_public_request_v2"
NORMALIZED_EPOCH_SCHEMA = "digitalocean_public_epoch_v2"
PUBLIC_SAFETY_SCAN_SCHEMA = "digitalocean_direct_public_safety_scan_v1"
SCOPE_EXCLUSION_SCHEMA = "digitalocean_direct_scope_exclusion_v1"
SOAK_MANIFEST_SCHEMA = "do_direct_soak_campaign_v1"
SOAK_PLAN_SCHEMA = "do_direct_soak_plan_v1"
SOAK_REQUEST_SCHEMA = "do_direct_soak_request_v1"
SOAK_PHASE_SCHEMA = "do_direct_soak_phase_v1"
SOAK_BLOCK_SCHEMA = "do_direct_soak_analysis_block_v1"
SOAK_QUALITY_PAIR_SCHEMA = "do_direct_soak_quality_pair_v1"
SOAK_CELL_SCHEMA = "do_direct_soak_cell_v1"
SOAK_SUMMARY_SCHEMA = "do_direct_soak_summary_v1"
COMPLETION_PLAN_SCHEMA = "do_direct_completion_plan_v1"
COMPLETION_MANIFEST_SCHEMA = "do_direct_completion_manifest_v1"
COMPLETION_REQUEST_SCHEMA = "do_direct_completion_request_v1"
COMPLETION_OUTCOME_SCHEMA = "do_direct_completion_probe_outcome_v1"
COMPLETION_SUMMARY_SCHEMA = "do_direct_completion_summary_v1"
COMPLETION_SOAK_WAVE_SCHEMA = "do_direct_completion_soak_wave_v1"
COMPLETION_SOAK_CENSOR_SCHEMA = "do_direct_completion_soak_censor_v1"
MATCHED_CLOSURE_PLAN_SCHEMA = "do_matched_closure_plan_v1"
MATCHED_CLOSURE_MANIFEST_SCHEMA = "do_matched_closure_manifest_v1"
MATCHED_CLOSURE_REQUEST_SCHEMA = "do_matched_closure_request_v1"
MATCHED_CLOSURE_OUTCOME_SCHEMA = "do_matched_closure_outcome_v1"
MATCHED_CLOSURE_SUMMARY_SCHEMA = "do_matched_closure_summary_v1"

COMPLETION_LANES = frozenset(
    {"capability_retry", "context_retry", "realized_output", "cache_observation"}
)
COMPLETION_SUPERSESSION_LANES = frozenset({"capability_retry", "context_retry"})
COMPLETION_TERMINAL_STATUSES = frozenset({"complete", "incomplete_or_censored"})

SOAK_TIMEOUT_ERROR_TYPES = frozenset(
    {"ReadTimeout", "ConnectTimeout", "PoolTimeout", "TimeoutException", "TimeoutError"}
)

SOAK_PHASES = (
    "paired_low_load",
    "two_minute_soak",
    "post_soak_recovery",
)

EXPECTED_ENDPOINT_IDS = DIGITALOCEAN_HOSTED_MODEL_IDS
EXPECTED_ENDPOINT_SET = frozenset(EXPECTED_ENDPOINT_IDS)
HISTORICAL_PARTNER_ENDPOINT_IDS = frozenset({"arcee-trinity-large-thinking"})
KNOWN_EVIDENCE_ENDPOINT_SET = EXPECTED_ENDPOINT_SET | HISTORICAL_PARTNER_ENDPOINT_IDS
DEEPSEEK_ENDPOINT_ID = "deepseek-v4-flash-0731"
KIMI_ENDPOINT_ID = "kimi-k3"
KIMI_UNDOCUMENTED_CONTEXT_PROBE_ANCHOR = 65_536

STRICT_REQUEST_CONTRACT_HASH_FIELDS = (
    "request_identity_sha256",
    "rendered_payload_sha256",
    "request_payload_sha256",
    "scorer_contract_sha256",
    "model_contract_sha256",
    "documentation_contract_sha256",
    "campaign_plan_sha256",
)
REJECTED_REQUEST_CLASSIFICATIONS = frozenset(
    {
        "explicit_context_limit_rejection",
        "rejected_or_unsupported",
        "documented_unavailable",
        "matched_control_rejection",
    }
)

REQUIRED_COVERAGE_DIMENSIONS = (
    "capability_smoke",
    "low_load_baseline",
    "input_context",
    "output_length",
    "parameter_validation",
    "parameter_interactions",
    "aimd_short_short",
    "aimd_long_short",
    "aimd_short_long",
    "aimd_mixed",
    "post_overload_recovery",
    "quality_low_load",
    "quality_near_saturation",
    "tool_calling",
    "structured_output",
    "vision",
)

_CAPABILITY_SCOPE_EXCLUSION_CONTRACT: dict[str, tuple[str, str]] = {
    "adaptive_tool_over_limit_followups": (
        "Adaptive tool +1 boundary discovery",
        "tool_calling",
    ),
    "conditional_retry_backoff_followups": (
        "Conditional HTTP 429/5xx retry-backoff behavior",
        "post_overload_recovery",
    ),
}

METRIC_DEFINITIONS = {
    "offered_rpm": (
        "Requests scheduled into an open-loop offered window, divided by that "
        "window in minutes. Slow responses do not reduce offered load."
    ),
    "achieved_rpm": (
        "Successful requests completed per elapsed wall-clock minute for the "
        "reported group or load epoch."
    ),
    "effective_input_tpm": (
        "Server-reported prompt tokens from scientifically usable responses, "
        "divided by elapsed wall-clock minutes."
    ),
    "effective_output_tpm": (
        "Server-reported generated tokens from scientifically usable responses, "
        "divided by elapsed wall-clock minutes."
    ),
    "ttft_seconds": (
        "Client-observed time from dispatch to the first streamed content or "
        "reasoning event. It includes network, queueing, and prefill time."
    ),
    "prefill_proxy": (
        "Prompt tokens divided by streamed TTFT, reported only when the service "
        "explicitly reports a cache miss. This is an end-to-end proxy, not direct "
        "server-side prefill speed. Buffered responses and unknown cache state are "
        "censored."
    ),
    "post_ttft_output_tokens_per_second_proxy": (
        "Server-reported billed completion tokens divided by request duration minus "
        "streamed TTFT. This conservative client-observed service-output proxy includes "
        "network and buffering effects; it is not direct decoder speed. Intervals shorter "
        "than 100 ms are timing-unstable and are explicitly censored from this per-request "
        "rate while remaining in aggregate billed-token goodput."
    ),
    "aggregate_output_goodput": (
        "Successful billed completion tokens divided by the complete epoch, soak block, "
        "or source-active wall-clock interval. This is the headline throughput measure."
    ),
    "legacy_sse_chunk_span_proxy": (
        "Completion tokens divided by the time between first and last content SSE "
        "events. Retained only in the metric audit because SSE events may contain many "
        "tokens; it is never labelled or plotted as decode throughput."
    ),
    "goodput_rpm": (
        "Requests per minute that completed with authoritative token usage and, "
        "when a deterministic task score applies, passed that score."
    ),
    "quality_adjusted_output_tpm": (
        "Generated output TPM multiplied request-by-request by deterministic "
        "quality score. Unscored requests do not silently receive full credit."
    ),
    "confidence_interval": (
        "A 95% uncertainty interval. Serial results resample requests; load "
        "results first reduce each epoch to one independent unit and resample epochs."
    ),
    "p99": (
        "The 99th percentile is reported only for at least 1,000 relevant "
        "observations; otherwise it is explicitly suppressed."
    ),
}

LIMITATIONS = (
    "This is an observed operating envelope for the tested account, client "
    "region, time period, request mix, and API version; it is not a universal SLA.",
    "Prompt tokens divided by TTFT is an end-to-end prefill proxy because the "
    "service does not expose direct server-side prefill timing.",
    "A request limit being accepted is different from the model actually "
    "realizing that many output tokens.",
    "Capacity is confirmed only when at least three valid healthy epochs exist "
    "at the same offered rate. Otherwise the highest healthy rate is exploratory.",
    "p99 is suppressed below 1,000 relevant observations; sparse p95 values are "
    "labelled exploratory.",
    "Quality comparisons are endpoint observations, not causal DigitalOcean "
    "failures, unless a matched external-provider control is supplied separately.",
    "Streaming events are not token timestamps. The legacy first-to-last SSE-event "
    "rate is audit-only; valid endpoint comparisons use aggregate wall-clock goodput "
    "or the explicitly labelled post-TTFT end-to-end proxy.",
    "Buffered responses do not expose token-level TTFT. Their full-response arrival "
    "times are retained as latency but censored from TTFT and prefill summaries.",
)

_DENIED_PUBLIC_KEY = re.compile(
    r"(?i)(?:^|_)(?:authorization|api_?key|access_?token|refresh_?token|"
    r"password|secret|cookie|set_?cookie|prompt|messages?|response_?body|"
    r"request_?body|raw_?headers?|request_?headers?|response_?headers?|"
    r"model_?output|reasoning_?text|hidden_?trace|private_?path)(?:$|_)"
)
_CREDENTIAL_VALUE = re.compile(
    r"(?i)(?:dop_v1_[a-z0-9_-]{8,}|bearer\s+[a-z0-9._~+/-]{8,}|"
    r"sk-[a-z0-9_-]{8,}|xox[baprs]-[a-z0-9-]{8,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)
_PRIVATE_PATH_VALUE = re.compile(
    r"(?i)(?:\b[A-Z]:[\\/](?:Users|home|private|tmp)[\\/]|"
    r"(?:^|[\s\"'])/(?:home|users|root|tmp|var/tmp)/|file://|"
    r"\\\\[^\\\s]+\\)"
)
_PUBLIC_SUFFIXES = {
    ".json",
    ".jsonl",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".gz",
    ".md",
}


class DirectReportError(ValueError):
    """Raised when public analysis inputs violate the frozen endpoint contract."""


def scan_public_bundle_safety(directory: Path) -> dict[str, Any]:
    """Fail closed on secret-bearing fields, values, paths, or file types.

    This is a narrowly named safety/sanitization check, not a scientific or
    publication-readiness review. Findings name only a location and rule; they
    never echo the matched value.
    """

    root = Path(directory)
    findings: list[dict[str, str]] = []

    def finding(pointer: str, rule: str) -> None:
        findings.append({"pointer": pointer, "rule": rule})

    def visit(value: Any, pointer: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, nested in value.items():
                key = str(raw_key)
                child = f"{pointer}/{key}" if pointer else key
                if _DENIED_PUBLIC_KEY.search(key):
                    finding(child, "denied_field_name")
                visit(nested, child)
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, nested in enumerate(value):
                visit(nested, f"{pointer}/{index}")
            return
        if isinstance(value, str):
            if _CREDENTIAL_VALUE.search(value):
                finding(pointer, "credential_pattern")
            if _PRIVATE_PATH_VALUE.search(value):
                finding(pointer, "private_path_pattern")

    if not root.is_dir() or root.is_symlink():
        raise DirectReportError("public output must be a regular directory")
    scanned = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            finding(relative, "symlink_not_allowed")
            continue
        if not path.is_file():
            continue
        if path.suffix.casefold() not in _PUBLIC_SUFFIXES:
            finding(relative, "unapproved_file_type")
            continue
        if path.suffix.casefold() == ".gz" and path.name != "analysis.json.gz":
            finding(relative, "unapproved_compressed_file")
            continue
        scanned += 1
        if path.suffix.casefold() in {".png", ".jpg", ".jpeg"}:
            continue
        try:
            if path.suffix.casefold() == ".gz":
                with gzip.open(path, "rt", encoding="utf-8") as compressed:
                    text = compressed.read()
            else:
                text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            finding(relative, "not_strict_utf8_text")
            continue
        if _CREDENTIAL_VALUE.search(text):
            finding(relative, "credential_pattern")
        if _PRIVATE_PATH_VALUE.search(text):
            finding(relative, "private_path_pattern")
        try:
            if path.suffix.casefold() in {".json", ".gz"}:
                visit(json.loads(text), relative)
            elif path.suffix.casefold() == ".jsonl":
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if line.strip():
                        visit(json.loads(line), f"{relative}:{line_number}")
            elif path.suffix.casefold() == ".csv":
                for row_number, row in enumerate(
                    csv.DictReader(text.splitlines()), start=2
                ):
                    visit(row, f"{relative}:{row_number}")
            else:
                visit(text, relative)
        except (ValueError, csv.Error):
            finding(relative, "invalid_structured_text")
    unique = {(row["pointer"], row["rule"]): row for row in findings}
    ordered = [unique[key] for key in sorted(unique)]
    return {
        "schema_version": PUBLIC_SAFETY_SCAN_SCHEMA,
        "passed": not ordered,
        "scanned_file_count": scanned,
        "finding_count": len(ordered),
        "findings": ordered,
    }


# Backwards-compatible import for callers outside the public build. New code
# and receipts use the narrower name above so a sanitization pass is never
# mistaken for a scientific publication-readiness decision.
scan_public_bundle = scan_public_bundle_safety


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer(value: Any) -> int | None:
    parsed = _number(value)
    if parsed is None or parsed < 0 or not parsed.is_integer():
        return None
    return int(parsed)


def _signed_integer(value: Any) -> int | None:
    parsed = _number(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_non_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _timestamp(value: Any) -> float | None:
    parsed = _number(value)
    if parsed is not None:
        return parsed
    text = _text(value)
    if text is None:
        return None
    try:
        normalized = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def _iso_time(value: Any) -> str | None:
    parsed = _timestamp(value)
    if parsed is None:
        return None
    return datetime.fromtimestamp(parsed, timezone.utc).isoformat()


def _stable_id(prefix: str, *parts: Any) -> str:
    body = "\0".join(str(part) for part in parts)
    return prefix + hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


def _require_endpoint(value: Any) -> str:
    endpoint = str(value or "")
    if endpoint.casefold().startswith("deepseek") and endpoint != DEEPSEEK_ENDPOINT_ID:
        raise DirectReportError(
            "only deepseek-v4-flash-0731 is admissible; found " + repr(endpoint)
        )
    if endpoint not in KNOWN_EVIDENCE_ENDPOINT_SET:
        raise DirectReportError(f"unexpected DigitalOcean endpoint {endpoint!r}")
    return endpoint


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DirectReportError(
                    f"invalid JSONL in {path.name} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise DirectReportError(
                    f"expected object in {path.name} at line {line_number}"
                )
            rows.append(value)
    return rows


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _request_identity(row: Mapping[str, Any], source_id: str) -> str:
    return str(
        row.get("request_id")
        or row.get("attempt_id")
        or row.get("cell_id")
        or _stable_id(
            "req-",
            source_id,
            row.get("model_id") or row.get("endpoint_id"),
            row.get("started_at") or row.get("sent_unix"),
            row.get("task_id") or row.get("workload_id"),
        )
    )


def _success_flags(row: Mapping[str, Any]) -> tuple[bool, bool, bool, bool | None]:
    status = str(row.get("status") or row.get("attempt_state") or "").casefold()
    http_status = _integer(row.get("http_status"))
    transport = row.get("transport_success")
    if not isinstance(transport, bool):
        transport = status in {"success", "completed", "ok"} and (
            http_status is None or 200 <= http_status < 300
        )
    classification = str(row.get("coverage_classification") or "")
    if classification in REJECTED_REQUEST_CLASSIFICATIONS or status in {
        "explicit_context_limit_rejection",
        "rejected_or_unsupported",
        "documented_unavailable",
    }:
        transport = False
    usage = _mapping(row.get("usage"))
    prompt_tokens = _integer(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or row.get("actual_prompt_tokens_x_axis")
        or row.get("server_prompt_tokens")
        or row.get("realized_input_tokens")
    )
    completion_tokens = _integer(
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or row.get("realized_output_tokens")
    )
    usage_ok = row.get("usage_gate_passed")
    if not isinstance(usage_ok, bool):
        usage_ok = bool(
            transport and prompt_tokens is not None and completion_tokens is not None
        )
    scientific = row.get("scientific_success")
    if not isinstance(scientific, bool):
        scientific = bool(transport and usage_ok)
    score = _number(row.get("quality_score"))
    functional = row.get("functional_valid")
    if not isinstance(functional, bool):
        retrieval_correct = row.get("retrieval_correct")
        if isinstance(retrieval_correct, bool):
            functional = retrieval_correct
        else:
            functional = None if score is None else score >= 0.999999
    goodput = bool(scientific and functional is not False)
    return bool(transport), bool(scientific), goodput, functional


def _requested_input(row: Mapping[str, Any]) -> int | None:
    metadata = _mapping(row.get("task_metadata"))
    return _integer(
        row.get("requested_input_tokens")
        or row.get("target_input_tokens")
        or row.get("estimated_target_prompt_tokens")
        or row.get("target_prompt_tokens")
        or metadata.get("planned_input_tokens")
        or metadata.get("estimated_target_prompt_tokens")
        or metadata.get("target_prompt_tokens")
        or (
            row.get("context_bucket")
            if str(row.get("context_bucket", "")).isdigit()
            else None
        )
    )


def _requested_output(row: Mapping[str, Any]) -> tuple[int | None, str | None]:
    metadata = _mapping(row.get("task_metadata"))
    explicit = _integer(
        row.get("requested_output_tokens")
        or row.get("target_output_tokens")
        or row.get("requested_max_output_tokens")
        or metadata.get("requested_output_tokens")
    )
    if explicit is not None:
        return explicit, "tokens"
    words = _integer(metadata.get("planned_output_words"))
    if words is None and str(row.get("task_family")) == "controlled_output":
        words = _integer(row.get("output_bucket"))
    return (words, "words") if words is not None else (None, None)


def _coverage_tags(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = row.get("coverage_tags")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value if _text(item) is not None)


def _context_workload(row: Mapping[str, Any]) -> str | None:
    """Infer a stable report workload from a direct-context evidence row."""

    schema = str(row.get("schema_version") or "")
    tags = _coverage_tags(row)
    is_context = schema.startswith("do_direct_context_") or bool(
        tags
        and any(
            marker in tag.casefold()
            for tag in tags
            for marker in ("percentage_", "context", "combined", "probe_anchor")
        )
    )
    if not is_context:
        return None
    if any(
        marker in tag.casefold()
        for tag in tags
        for marker in (
            "boundary",
            "estimate_lower",
            "estimate_center",
            "estimate_upper",
            "transition",
            "bisection",
        )
    ):
        return "context_boundary"
    return "long_context_retrieval"


def _row_coverage_conclusive(row: Mapping[str, Any]) -> bool:
    explicit = row.get("coverage_conclusive")
    if isinstance(explicit, bool):
        return explicit
    if row.get("transport_success") is True:
        return True
    classification = str(row.get("coverage_classification") or "")
    if classification in {
        "documented_unavailable",
        "explicit_context_limit_rejection",
        "matched_control_rejection",
        "rejected_or_unsupported",
    }:
        return True
    return False


def _row_evidence_backed_unsupported(row: Mapping[str, Any]) -> bool:
    classification = str(row.get("coverage_classification") or "")
    if classification:
        return classification in {
            "rejected_or_unsupported",
            "documented_unavailable",
            "matched_control_rejection",
        }
    return False


def _usage_counter(usage: Mapping[str, Any], *keys: str) -> int | None:
    """Return a present non-negative counter without turning zero into missing."""

    for key in keys:
        if key in usage:
            value = _integer(usage.get(key))
            if value is not None:
                return value
    return None


def _cache_observation(
    row: Mapping[str, Any], usage: Mapping[str, Any]
) -> dict[str, Any]:
    timing = _mapping(row.get("timing"))
    explicit = _mapping(timing.get("cache_observation"))
    explicit_state = _text(explicit.get("observed_state"))
    if explicit_state in {
        "cache_hit_observed",
        "cache_miss_observed",
        "not_reported_unknown",
    }:
        return {
            "state": explicit_state,
            "counters_reported": bool(explicit.get("cache_counters_reported")),
            "read_tokens": _integer(explicit.get("cache_read_tokens")),
            "creation_tokens": _integer(explicit.get("cache_creation_input_tokens")),
        }

    details = _mapping(usage.get("prompt_tokens_details"))
    read_candidates = [
        _usage_counter(usage, "cache_read_input_tokens", "cached_tokens"),
        _usage_counter(details, "cached_tokens"),
    ]
    creation_candidates = [
        _usage_counter(
            usage,
            "cache_creation_input_tokens",
            "cache_created_input_tokens",
        )
    ]
    present = any(value is not None for value in read_candidates + creation_candidates)
    read = max((value for value in read_candidates if value is not None), default=None)
    creation = max(
        (value for value in creation_candidates if value is not None), default=None
    )
    return {
        "state": (
            "cache_hit_observed"
            if read is not None and read > 0
            else ("cache_miss_observed" if present else "not_reported_unknown")
        ),
        "counters_reported": present,
        "read_tokens": read,
        "creation_tokens": creation,
    }


def _stream_observation(
    row: Mapping[str, Any], *, source_kind: str
) -> tuple[str, int | None, str | None]:
    """Return (mode, content-event count, first-event kind)."""

    timing = _mapping(row.get("timing"))
    stream = _mapping(row.get("stream"))
    stream_observation = _mapping(row.get("stream_observation"))
    bindings = _mapping(row.get("bindings"))
    first_kind = _text(
        timing.get("first_event_kind")
        or stream.get("first_event_kind")
        or stream_observation.get("first_event_kind")
    )
    event_count = _integer(
        _first_non_none(
            timing.get("event_count"),
            stream.get("event_count"),
            stream_observation.get("event_count"),
        )
    )
    stream_binding = bindings.get("stream") if "stream" in bindings else None
    if stream_binding is False or first_kind == "buffered_response":
        mode = "buffered_nonstream"
    elif stream_binding is True or source_kind in {
        "direct_breadth",
        "direct_aimd",
        "direct_soak",
        "direct_context",
    }:
        mode = "streamed"
    elif event_count is not None and event_count >= 2:
        mode = "streamed"
    else:
        mode = "unknown"
    return mode, event_count, first_kind


def _choice_count(row: Mapping[str, Any]) -> int:
    bindings = _mapping(row.get("bindings"))
    explicit = _integer(
        _first_non_none(
            row.get("choice_count"),
            row.get("n"),
            bindings.get("n"),
        )
    )
    return explicit if explicit is not None and explicit > 0 else 1


def _validation_probe(row: Mapping[str, Any]) -> bool:
    if _text(row.get("dimension")) is None:
        return False
    text = " ".join(
        str(value or "").casefold()
        for value in (
            row.get("probe_id"),
            row.get("state"),
            row.get("design_role"),
            row.get("coverage_classification"),
        )
    )
    markers = (
        "malformed",
        "invalid",
        "outside",
        "below_min",
        "above_max",
        "just_over",
        "over_limit",
        "unsupported_format",
    )
    return any(marker in text for marker in markers)


def normalize_request(
    row: Mapping[str, Any], *, source_kind: str, source_id: str
) -> dict[str, Any]:
    """Return an allowlisted public request row from any admitted journal."""

    endpoint = _require_endpoint(row.get("model_id") or row.get("endpoint_id"))
    timing = _mapping(row.get("timing"))
    usage = _mapping(row.get("usage"))
    transport, scientific, goodput, functional = _success_flags(row)
    input_tokens = _integer(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or row.get("actual_prompt_tokens_x_axis")
        or row.get("server_prompt_tokens")
        or row.get("realized_input_tokens")
    )
    output_tokens = _integer(
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or row.get("realized_output_tokens")
    )
    # A rejected or otherwise transport-failed call has no trustworthy actual
    # token coordinate. Several source journals intentionally use terminal zero
    # placeholders; those are accounting sentinels, not server-reported usage.
    rejection_classification = str(row.get("coverage_classification") or "")
    rejected = (
        rejection_classification in REJECTED_REQUEST_CLASSIFICATIONS
        or str(row.get("status") or "").casefold() in REJECTED_REQUEST_CLASSIFICATIONS
    )
    if not transport or rejected:
        input_tokens = None
        output_tokens = None
    requested_output, requested_output_unit = _requested_output(row)
    request_seconds = _number(
        timing.get("request_seconds")
        or timing.get("end_to_end_seconds")
        or row.get("latency_seconds")
        or row.get("elapsed_seconds")
    )
    generation_seconds = _number(
        timing.get("generation_seconds") or row.get("generation_seconds")
    )
    # Historical runners labelled the first-to-last content-event span as
    # generation time. SSE events are batched transport chunks, not token
    # timestamps, so the resulting rate is retained only for a transparent
    # compatibility audit and is never used as decoder throughput.
    legacy_sse_span_tps = _number(
        timing.get("output_tokens_per_second")
        or timing.get("sse_chunk_span_output_tokens_per_second_proxy")
        or row.get("output_tokens_per_second")
    )
    if (
        legacy_sse_span_tps is None
        and output_tokens is not None
        and generation_seconds not in {None, 0}
    ):
        legacy_sse_span_tps = output_tokens / float(generation_seconds)
    raw_ttft = _number(
        timing.get("ttft_seconds")
        or timing.get("time_to_first_token_seconds")
        or row.get("ttft_seconds")
    )
    stream_mode, event_count, first_event_kind = _stream_observation(
        row, source_kind=source_kind
    )
    choice_count = _choice_count(row)
    multi_choice = choice_count > 1
    # Buffered completion APIs reveal only full-response latency. Reporting
    # that timestamp as TTFT would make TTFT and prefill curves meaningless.
    ttft = raw_ttft if stream_mode == "streamed" else None
    post_ttft_seconds = (
        request_seconds - ttft
        if request_seconds is not None and ttft is not None
        else None
    )
    timing_invalidity_reasons: list[str] = []
    if stream_mode != "streamed":
        timing_invalidity_reasons.append("not_streamed_ttft_unobservable")
    if multi_choice:
        timing_invalidity_reasons.append(
            "multiple_choices_aggregate_usage_not_per_sequence"
        )
    if (
        request_seconds is not None
        and raw_ttft is not None
        and raw_ttft > request_seconds + 1e-9
    ):
        timing_invalidity_reasons.append("ttft_after_request_end")
    if post_ttft_seconds is not None and post_ttft_seconds < 0.1:
        timing_invalidity_reasons.append("post_ttft_interval_below_100ms_unstable_rate")
    post_ttft_output_tps = (
        output_tokens / post_ttft_seconds
        if output_tokens is not None
        and output_tokens > 0
        and post_ttft_seconds is not None
        and post_ttft_seconds >= 0.1
        and not timing_invalidity_reasons
        else None
    )
    cache = _cache_observation(row, usage)
    prefill_proxy = (
        input_tokens / ttft
        if input_tokens is not None
        and input_tokens > 0
        and ttft is not None
        and ttft >= 1e-4
        and cache["state"] == "cache_miss_observed"
        and not multi_choice
        else None
    )
    started = _iso_time(
        row.get("started_at")
        or row.get("sent_at")
        or row.get("sent_unix")
        or row.get("arrival_unix")
    )
    ended = _iso_time(
        row.get("ended_at") or row.get("finished_at") or row.get("finished_unix")
    )
    quality = _number(row.get("quality_score"))
    if quality is None and isinstance(row.get("retrieval_correct"), bool):
        quality = float(bool(row.get("retrieval_correct")))
    status = str(row.get("status") or row.get("attempt_state") or "unknown")
    http_status = _integer(row.get("http_status"))
    error_type = _text(row.get("error_type") or row.get("classification"))
    timeout = bool(
        "timeout" in str(error_type or "").casefold() or http_status in {408, 504}
    )
    phase = _text(row.get("phase") or row.get("stage"))
    shape = _text(row.get("shape") or row.get("shape_id"))
    load = _mapping(row.get("load"))
    task_family = _text(row.get("task_family"))
    declared_workload = _text(task_family or row.get("workload_id"))
    workload = _text(declared_workload or _context_workload(row) or shape)
    if workload is None:
        workload = "unspecified"
    if shape is None and source_kind == "direct_aimd":
        shape = workload
    if source_kind == "direct_aimd" and shape is not None:
        workload = shape
    raw_cost = _first_non_none(
        row.get("estimated_cost_usd"),
        row.get("accounted_cost_usd"),
        row.get("settled_cost_usd"),
        row.get("cost_usd"),
    )
    estimated_cost = _number(raw_cost)
    if raw_cost is not None and estimated_cost is None:
        raise DirectReportError("request cost must be a finite numeric value")
    return {
        "schema_version": NORMALIZED_REQUEST_SCHEMA,
        "source_kind": source_kind,
        "source_id": source_id,
        "run_id": _text(row.get("run_id")) or source_id,
        "request_id": _request_identity(row, source_id),
        "semantic_id": _text(row.get("semantic_id")),
        "attempt_index": _integer(row.get("attempt_index")),
        "source_request_id": _text(row.get("source_request_id")),
        "supersedes_request_id": _text(row.get("supersedes_request_id")),
        "semantic_final_request_id": _text(row.get("semantic_final_request_id")),
        "semantic_coverage_attempt": (
            row.get("semantic_coverage_attempt")
            if isinstance(row.get("semantic_coverage_attempt"), bool)
            else None
        ),
        "cell_id": _text(
            row.get("cell_id")
            or (
                row.get("request_id") or row.get("probe_id")
                if source_kind == "direct_breadth"
                else None
            )
        ),
        "epoch_id": _text(
            row.get("epoch_id")
            or row.get("science_epoch_id")
            or row.get("physical_epoch_id")
        ),
        "endpoint_id": endpoint,
        "workload": workload,
        "workload_provenance": (
            "request_declared" if declared_workload is not None else "report_inferred"
        ),
        "task_family": task_family,
        "shape": shape,
        "phase": phase,
        "task_id": _text(row.get("task_id")),
        "block_id": _integer(row.get("block_id")),
        "repeat_index": _integer(row.get("repeat_index")),
        "started_at": started,
        "ended_at": ended,
        "status": status,
        "http_status": http_status,
        "error_type": error_type,
        "timeout": timeout,
        "rate_limited": http_status == 429,
        "server_error": bool(http_status is not None and 500 <= http_status < 600),
        "transport_success": transport,
        "scientific_success": scientific,
        "goodput_success": goodput,
        "functional_valid": functional,
        "quality_score": quality,
        "quality_scored": quality is not None,
        "score_kind": _text(row.get("score_kind") or row.get("score_basis")),
        "finish_reason": _text(row.get("finish_reason")),
        "requested_input_tokens": _requested_input(row),
        "estimated_target_input_tokens": _integer(
            row.get("estimated_target_prompt_tokens")
            or row.get("target_prompt_tokens")
            or row.get("target_input_tokens")
        ),
        "server_reported_input_tokens": input_tokens,
        "requested_output_target": requested_output,
        "requested_output_unit": requested_output_unit,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "request_seconds": request_seconds,
        "http_response_start_seconds": _number(timing.get("headers_seconds")),
        "ttft_seconds": ttft,
        "raw_first_response_event_seconds": raw_ttft,
        "generation_seconds": generation_seconds,
        "stream_seconds": _number(timing.get("stream_seconds")),
        "stream_mode": stream_mode,
        "stream_event_count": event_count,
        "stream_first_event_kind": first_event_kind,
        "choice_count": choice_count,
        "multi_choice": multi_choice,
        "mean_inter_token_seconds_proxy": _number(
            timing.get("mean_inter_token_seconds_proxy")
            or row.get("mean_inter_token_seconds_proxy")
        ),
        "post_ttft_seconds": post_ttft_seconds,
        "post_ttft_output_tokens_per_second_proxy": post_ttft_output_tps,
        # Backwards-compatible public column now carries the corrected proxy,
        # never the old SSE-event-span calculation.
        "output_tokens_per_second": post_ttft_output_tps,
        "output_tokens_per_second_metric_kind": (
            "billed_completion_tokens_over_request_minus_streamed_ttft_proxy"
            if post_ttft_output_tps is not None
            else None
        ),
        "legacy_sse_chunk_span_output_tokens_per_second_proxy": legacy_sse_span_tps,
        "legacy_sse_span_headline_eligible": False,
        "timing_metric_invalidity_reasons": timing_invalidity_reasons,
        "timing_metric_audit_classification": (
            "valid_extreme_keep_and_flag"
            if post_ttft_output_tps is not None and post_ttft_output_tps >= 1_000
            else (
                "invalid_or_censored"
                if timing_invalidity_reasons or post_ttft_output_tps is None
                else "valid_ordinary"
            )
        ),
        "cache_state": cache["state"],
        "cache_counters_reported": cache["counters_reported"],
        "cache_read_tokens": cache["read_tokens"],
        "cache_creation_tokens": cache["creation_tokens"],
        "prefill_proxy_tokens_per_second": prefill_proxy,
        "prefill_headline_eligible": prefill_proxy is not None,
        "offered_rps": _number(row.get("offered_rps") or load.get("offered_rps")),
        "concurrency_ceiling": _integer(
            row.get("concurrency_ceiling")
            or load.get("concurrency_ceiling")
            or row.get("aimd_limit_before")
        ),
        "arrival_lag_seconds": _number(
            row.get("arrival_lag_seconds")
            or row.get("launch_lag_seconds")
            or load.get("schedule_lag_seconds")
        ),
        "estimated_cost_usd": estimated_cost,
        "cost_attributed": estimated_cost is not None,
        "request_hash": _text(
            row.get("request_hash")
            or row.get("payload_hash")
            or row.get("request_payload_sha256")
        ),
        "request_payload_sha256": _text(
            row.get("request_payload_sha256")
            or row.get("payload_hash")
            or row.get("request_hash")
        ),
        "rendered_payload_sha256": _text(row.get("rendered_payload_sha256")),
        "request_identity_sha256": _text(row.get("request_identity_sha256")),
        "scorer_contract_sha256": _text(row.get("scorer_contract_sha256")),
        "model_contract_sha256": _text(row.get("model_contract_sha256")),
        "documentation_contract_sha256": _text(
            row.get("documentation_contract_sha256")
        ),
        "campaign_plan_sha256": _text(
            row.get("campaign_plan_sha256") or row.get("plan_sha256")
        ),
        "response_hash": _text(row.get("response_hash") or row.get("response_sha256")),
        "coverage_tags": list(_coverage_tags(row)),
        "coverage_classification": _text(row.get("coverage_classification")),
        "capability_status": _text(row.get("capability_status")),
        "coverage_conclusive": (
            row.get("coverage_conclusive")
            if isinstance(row.get("coverage_conclusive"), bool)
            else None
        ),
        "provider_send_attempted": (
            row.get("provider_send_attempted")
            if isinstance(row.get("provider_send_attempted"), bool)
            else None
        ),
        "documentation_status": _text(row.get("documentation_status")),
        "capability_dimension": _text(row.get("dimension")),
        "capability_state": _text(row.get("state")),
        "capability_probe_id": _text(row.get("probe_id")),
        "capability_design_role": _text(row.get("design_role")),
        "malformed_validation_probe": _validation_probe(row),
        "retrieval_correct": (
            row.get("retrieval_correct")
            if isinstance(row.get("retrieval_correct"), bool)
            else None
        ),
        "planning_error_tokens": _signed_integer(row.get("planning_error_tokens")),
        "planning_absolute_error_tokens": _integer(
            row.get("planning_absolute_error_tokens")
        ),
        "planning_tolerance_tokens": _integer(row.get("planning_tolerance_tokens")),
        "planning_within_tolerance": (
            row.get("planning_within_tolerance")
            if isinstance(row.get("planning_within_tolerance"), bool)
            else None
        ),
        "context_window_anchor_source": _text(row.get("context_window_anchor_source")),
    }


def normalize_epoch(
    row: Mapping[str, Any], *, source_kind: str, source_id: str
) -> dict[str, Any]:
    endpoint = _require_endpoint(row.get("model_id") or row.get("endpoint_id"))
    counts = _mapping(row.get("counts"))
    ttft = _mapping(row.get("ttft"))
    latency = _mapping(row.get("latency"))
    raw_elapsed = _number(
        row.get("elapsed_seconds")
        or row.get("epoch_elapsed_seconds")
        or row.get("elapsed_seconds_including_drain")
        or row.get("offered_window_seconds")
    )
    offered_window = _number(
        row.get("epoch_seconds") or row.get("offered_window_seconds")
    )
    # The original direct runner stopped its live wall clock when the final
    # request drained.  At low rates, an epoch containing one early arrival
    # could therefore report one request divided by sub-second service time,
    # even though that request represented a fixed five-second arrival window.
    # Reconstruct the actual cohort wall time here.  This preserves drain time
    # when it is longer and preserves the predeclared arrival window otherwise.
    direct_window_correction = (
        source_kind == "direct_aimd"
        and raw_elapsed is not None
        and raw_elapsed > 0
        and offered_window is not None
        and offered_window > 0
    )
    elapsed = (
        max(raw_elapsed, offered_window) if direct_window_correction else raw_elapsed
    )
    scheduled = _integer(
        _first_non_none(
            row.get("scheduled"),
            row.get("scheduled_count"),
            row.get("scheduled_requests"),
            row.get("offered_arrivals"),
            counts.get("scheduled"),
        )
    )
    completed = _integer(
        _first_non_none(
            row.get("completed"),
            row.get("completed_count"),
            row.get("completed_requests"),
            row.get("terminal_count"),
            counts.get("completed"),
        )
    )
    successes = _integer(
        _first_non_none(
            row.get("success"),
            row.get("success_count"),
            row.get("successes"),
            row.get("successful_count"),
            counts.get("success"),
        )
    )
    quality_passes = _integer(
        _first_non_none(
            row.get("quality_pass_count"),
            row.get("quality_passes"),
            counts.get("quality_pass"),
        )
    )
    success_rate = _number(row.get("success_rate"))
    if success_rate is None and scheduled not in {None, 0} and successes is not None:
        success_rate = successes / scheduled
    offered_target = _number(row.get("offered_rps_target") or row.get("offered_rps"))
    offered_realized = _number(
        row.get("offered_rps_realized_schedule") or row.get("offered_rps_realized")
    )
    offered_observed = (
        offered_realized
        if source_kind == "direct_aimd" and offered_realized is not None
        else offered_target
    )
    completed_rpm = None
    successful_rpm = None
    if direct_window_correction and elapsed and completed is not None:
        completed_rpm = completed * 60 / elapsed
    else:
        completed_rpm = _number(row.get("achieved_rpm") or row.get("completed_rpm"))
    if direct_window_correction and elapsed and successes is not None:
        successful_rpm = successes * 60 / elapsed
    else:
        successful_rpm = _number(
            row.get("successful_rpm") or row.get("achieved_successful_rpm")
        )
    if successful_rpm is None and elapsed and successes is not None:
        successful_rpm = successes * 60 / elapsed
    if completed_rpm is None and elapsed and completed is not None:
        completed_rpm = completed * 60 / elapsed
    raw_effective_input_tpm = _number(
        row.get("effective_input_tpm") or row.get("successful_input_tpm")
    )
    raw_effective_output_tpm = _number(
        row.get("effective_output_tpm") or row.get("successful_output_tpm")
    )
    rate_scale = (
        raw_elapsed / elapsed
        if direct_window_correction
        and raw_elapsed is not None
        and elapsed is not None
        and elapsed > 0
        else 1.0
    )

    def corrected_direct_rate(value: Any) -> float | None:
        number = _number(value)
        return number * rate_scale if number is not None else None

    effective_input_tpm = corrected_direct_rate(raw_effective_input_tpm)
    effective_output_tpm = corrected_direct_rate(raw_effective_output_tpm)
    return {
        "schema_version": NORMALIZED_EPOCH_SCHEMA,
        "source_kind": source_kind,
        "source_id": source_id,
        "run_id": _text(row.get("run_id")) or source_id,
        "epoch_id": str(
            row.get("epoch_id")
            or row.get("science_epoch_id")
            or _stable_id(
                "epoch-",
                source_id,
                endpoint,
                row.get("shape") or row.get("shape_id"),
                row.get("started_at") or row.get("campaign_hour"),
            )
        ),
        "endpoint_id": endpoint,
        "workload": _text(row.get("workload") or row.get("workload_id")),
        "shape": _text(row.get("shape") or row.get("shape_id")),
        "phase": _text(row.get("phase") or row.get("stage")),
        "sequence": _integer(
            row.get("sequence") or row.get("epoch_index") or row.get("ordinal")
        ),
        "started_at": _iso_time(
            row.get("started_at") or row.get("started_unix") or row.get("sent_unix")
        ),
        "ended_at": _iso_time(row.get("ended_at") or row.get("ended_unix")),
        "elapsed_seconds": elapsed,
        "drain_elapsed_seconds_observed": raw_elapsed,
        "offered_window_seconds": offered_window or elapsed,
        "offered_rps": offered_observed,
        "offered_rps_target": offered_target,
        "offered_rps_realized_schedule": offered_realized,
        "concurrency_ceiling": _integer(row.get("concurrency_ceiling")),
        "peak_concurrency": _integer(
            row.get("peak_concurrency")
            or row.get("maximum_in_flight")
            or row.get("max_observed_concurrency")
        ),
        "scheduled_count": scheduled,
        "completed_count": completed,
        "success_count": successes,
        "quality_pass_count": quality_passes,
        "rate_limit_count": _integer(
            row.get("rate_limit_count")
            or row.get("http_429")
            or counts.get("rate_limited")
        )
        or 0,
        "timeout_count": _integer(row.get("timeout_count") or counts.get("timeout"))
        or 0,
        "server_error_count": _integer(
            row.get("server_error_count")
            or row.get("http_5xx")
            or counts.get("server_error")
        )
        or 0,
        "other_error_count": _integer(
            row.get("other_error_count") or counts.get("other_error")
        )
        or 0,
        "success_rate": success_rate,
        "offered_rpm": (
            offered_observed * 60 if offered_observed is not None else None
        ),
        "offered_rpm_target": (
            offered_target * 60 if offered_target is not None else None
        ),
        "offered_realized_schedule_rpm": (
            offered_realized * 60 if offered_realized is not None else None
        ),
        "completed_rpm": completed_rpm,
        "achieved_rpm": successful_rpm,
        "successful_rpm": successful_rpm,
        "effective_input_tpm": effective_input_tpm,
        "effective_output_tpm": effective_output_tpm,
        "aggregate_output_goodput_tokens_per_second": (
            effective_output_tpm / 60 if effective_output_tpm is not None else None
        ),
        "offered_input_tpm_conservative": _number(
            corrected_direct_rate(row.get("offered_input_tpm_conservative"))
        ),
        "offered_output_token_ceiling_tpm": _number(
            corrected_direct_rate(row.get("offered_output_token_ceiling_tpm"))
        ),
        "accepted_input_tpm": corrected_direct_rate(row.get("accepted_input_tpm")),
        "accepted_output_tpm": corrected_direct_rate(row.get("accepted_output_tpm")),
        "quality_adjusted_input_tpm": corrected_direct_rate(
            row.get("quality_adjusted_input_tpm")
        ),
        "goodput_rpm": (
            quality_passes * 60 / elapsed
            if direct_window_correction and quality_passes is not None and elapsed
            else corrected_direct_rate(row.get("goodput_rpm"))
        ),
        "quality_adjusted_output_tpm": corrected_direct_rate(
            row.get("quality_adjusted_output_tpm")
        ),
        "ttft_p50_seconds": _number(row.get("ttft_p50_seconds") or ttft.get("p50")),
        "ttft_p90_seconds": _number(row.get("ttft_p90_seconds") or ttft.get("p90")),
        "ttft_p95_seconds": _number(row.get("ttft_p95_seconds") or ttft.get("p95")),
        "latency_p50_seconds": _number(
            row.get("latency_p50_seconds") or latency.get("p50")
        ),
        "latency_p90_seconds": _number(
            row.get("latency_p90_seconds") or latency.get("p90")
        ),
        "latency_p95_seconds": _number(
            row.get("latency_p95_seconds") or latency.get("p95")
        ),
        # Compatibility alias: epoch TPS is aggregate successful billed-token
        # goodput over epoch wall time, never an SSE content-event span.
        "output_tokens_per_second": (
            effective_output_tpm / 60 if effective_output_tpm is not None else None
        ),
        "output_tokens_per_second_metric_kind": (
            "aggregate_successful_completion_tokens_over_epoch_wall_clock"
            if effective_output_tpm is not None
            else None
        ),
        "mean_arrival_lag_seconds": _number(row.get("mean_arrival_lag_seconds")),
        "p95_arrival_lag_seconds": _number(
            row.get("p95_arrival_lag_seconds") or row.get("schedule_lag_p95_seconds")
        ),
        "healthy": row.get("healthy") if isinstance(row.get("healthy"), bool) else None,
        "health_reasons": ";".join(str(v) for v in row.get("health_reasons", ()))
        if isinstance(row.get("health_reasons"), Sequence)
        and not isinstance(row.get("health_reasons"), (str, bytes))
        else _text(row.get("health_reasons")),
        "valid_for_capacity": bool(row.get("valid_for_capacity", True)),
        "estimated_cost_usd": _number(
            _first_non_none(
                row.get("estimated_cost_usd"),
                row.get("accounted_cost_usd"),
                row.get("cost_usd"),
            )
        ),
    }


def _normalize_plan_row(row: Mapping[str, Any], *, source_id: str) -> dict[str, Any]:
    task = _mapping(row.get("task"))
    endpoint = _require_endpoint(row.get("model_id") or row.get("endpoint_id"))
    workload = (
        _text(task.get("family") or row.get("workload_id") or _context_workload(row))
        or "unspecified"
    )
    cell_id = row.get("cell_id") or row.get("request_id") or row.get("probe_id")
    return {
        "source_kind": "direct_breadth",
        "source_id": source_id,
        "cell_id": str(cell_id or _stable_id("cell-", source_id, endpoint, workload)),
        "endpoint_id": endpoint,
        "workload": workload,
        "shape": _text(
            row.get("shape")
            or task.get("shape")
            or ("context_boundary" if _context_workload(row) else None)
        ),
        "task_id": _text(
            task.get("task_id") or row.get("task_id") or row.get("probe_id")
        ),
        "probe_id": _text(row.get("probe_id")),
        "planned_attempt_count": 1,
        "context_bucket": _text(
            task.get("context_bucket")
            or row.get("estimated_target_prompt_tokens")
            or row.get("target_prompt_tokens")
        ),
        "output_bucket": _text(task.get("output_bucket")),
        "requires_vision": bool(task.get("requires_vision")),
        "coverage_tags": list(_coverage_tags(row)),
        "estimated_target_input_tokens": _integer(
            row.get("estimated_target_prompt_tokens") or row.get("target_prompt_tokens")
        ),
        "requested_output_target": _integer(row.get("requested_max_output_tokens")),
        "context_window_anchor_source": _text(row.get("context_window_anchor_source")),
        "request_payload_sha256": _text(
            row.get("request_payload_sha256")
            or row.get("payload_hash")
            or row.get("request_hash")
        ),
        "rendered_payload_sha256": _text(row.get("rendered_payload_sha256")),
        "request_identity_sha256": _text(row.get("request_identity_sha256")),
        "scorer_contract_sha256": _text(row.get("scorer_contract_sha256")),
        "model_contract_sha256": _text(row.get("model_contract_sha256")),
        "documentation_contract_sha256": _text(
            row.get("documentation_contract_sha256")
        ),
        "campaign_plan_sha256": _text(row.get("campaign_plan_sha256")),
    }


def load_breadth_directory(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    directory = Path(path)
    source_id = directory.name
    plan_path = directory / "plan.jsonl"
    if not plan_path.is_file() and (directory / "plan.jsonl.gz").is_file():
        plan_path = directory / "plan.jsonl.gz"
    plans = [
        _normalize_plan_row(row, source_id=source_id) for row in _read_jsonl(plan_path)
    ]
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        manifest_plan_hash = (
            _text(manifest.get("plan_sha256"))
            if isinstance(manifest, Mapping)
            else None
        )
        if manifest_plan_hash is not None:
            if plan_path.suffix == ".gz":
                raise DirectReportError(
                    "cannot validate a manifest plan hash against compressed plan.jsonl.gz"
                )
            actual_plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            if actual_plan_hash != manifest_plan_hash:
                raise DirectReportError(
                    "breadth plan.jsonl does not match manifest plan_sha256"
                )
            for plan in plans:
                existing = _text(plan.get("campaign_plan_sha256"))
                if existing is not None and existing != manifest_plan_hash:
                    raise DirectReportError(
                        "plan row campaign_plan_sha256 disagrees with manifest"
                    )
                plan["campaign_plan_sha256"] = manifest_plan_hash
    request_path = directory / "requests.jsonl"
    if not request_path.is_file():
        request_path = directory / "records.jsonl"
    requests = [
        normalize_request(row, source_kind="direct_breadth", source_id=source_id)
        for row in _read_jsonl(request_path)
    ]
    return plans, requests


def load_breadth_scope_exclusions(path: Path) -> list[dict[str, Any]]:
    """Load explicit untested scope from a direct breadth manifest.

    A manifest exclusion is evidence that a named subtest was deliberately not
    run. It is never converted into capability support, rejection, or a
    completed coverage cell.
    """

    directory = Path(path)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise DirectReportError("breadth manifest must contain one JSON object")
    raw_exclusions = manifest.get("scope_exclusions")
    if raw_exclusions is None:
        return []
    if not isinstance(raw_exclusions, Mapping):
        raise DirectReportError("breadth manifest scope_exclusions must be an object")
    raw_models = manifest.get("models")
    if not isinstance(raw_models, Sequence) or isinstance(
        raw_models, (str, bytes, bytearray)
    ):
        raise DirectReportError(
            "breadth manifest with scope exclusions must declare endpoint models"
        )
    endpoints = [_require_endpoint(item) for item in raw_models]
    if len(endpoints) != len(set(endpoints)):
        raise DirectReportError("breadth manifest scope endpoint models must be unique")
    manifest_sha256 = _sha256(manifest_path)
    source_id = directory.name
    rows: list[dict[str, Any]] = []
    for raw_exclusion_id, raw_reason in sorted(
        raw_exclusions.items(), key=lambda item: str(item[0])
    ):
        if not isinstance(raw_exclusion_id, str) or not isinstance(raw_reason, str):
            raise DirectReportError(
                "breadth manifest scope exclusions require string IDs and reasons"
            )
        exclusion_id = _text(raw_exclusion_id)
        reason = _text(raw_reason)
        if exclusion_id is None or reason is None:
            raise DirectReportError(
                "breadth manifest scope exclusions require non-empty IDs and reasons"
            )
        label, dimension = _CAPABILITY_SCOPE_EXCLUSION_CONTRACT.get(
            exclusion_id,
            (exclusion_id.replace("_", " ").strip().capitalize(), "capability_smoke"),
        )
        for endpoint in endpoints:
            rows.append(
                {
                    "schema_version": SCOPE_EXCLUSION_SCHEMA,
                    "source_kind": "direct_breadth",
                    "source_id": source_id,
                    "source_manifest_sha256": manifest_sha256,
                    "endpoint_id": endpoint,
                    "scope_exclusion_id": exclusion_id,
                    "measurement_label": label,
                    "coverage_dimension": dimension,
                    "status": "untested",
                    "reason": reason,
                    "claim_policy": "explicitly_excluded_not_tested",
                }
            )
    return rows


def load_aimd_directory(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    directory = Path(path)
    source_id = directory.name
    request_path = directory / "requests.jsonl"
    if not request_path.is_file():
        request_path = directory / "records.jsonl"
    requests = [
        normalize_request(row, source_kind="direct_aimd", source_id=source_id)
        for row in _read_jsonl(request_path)
    ]
    epochs = []
    for sequence, row in enumerate(_read_jsonl(directory / "epochs.jsonl")):
        normalized = normalize_epoch(
            row, source_kind="direct_aimd", source_id=source_id
        )
        if normalized.get("sequence") is None:
            normalized["sequence"] = sequence
        epochs.append(normalized)
    return requests, epochs


def _soak_rows_by_id(
    path: Path,
    *,
    identity: str,
    schema: str,
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise DirectReportError(f"direct soak is missing required {path.name}")
    output: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if row.get("schema_version") != schema:
            raise DirectReportError(f"direct soak {path.name} has an invalid schema")
        row_id = _text(row.get(identity))
        if row_id is None or row_id in output:
            raise DirectReportError(
                f"direct soak {path.name} has a missing or duplicate {identity}"
            )
        output[row_id] = row
    return output


def _soak_identity(
    row: Mapping[str, Any],
    *,
    campaign_id: str,
    plan_sha256: str,
    cell: Mapping[str, Any],
    label: str,
) -> None:
    expected = {
        "campaign_id": campaign_id,
        "plan_sha256": plan_sha256,
        "cell_id": cell.get("cell_id"),
        "model_id": cell.get("model_id"),
        "shape": cell.get("shape"),
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise DirectReportError(
                f"direct soak {label} {key} does not match its plan cell"
            )
    endpoint = row.get("endpoint_id")
    if endpoint is not None and endpoint != cell.get("model_id"):
        raise DirectReportError(
            f"direct soak {label} endpoint_id does not match its plan cell"
        )
    if row.get("provider") not in {None, "digitalocean-serverless-inference"}:
        raise DirectReportError(f"direct soak {label} has an unexpected provider")


def _soak_count(row: Mapping[str, Any], key: str) -> int:
    value = _integer(row.get(key))
    if value is None:
        raise DirectReportError(f"direct soak {key} must be a non-negative integer")
    return value


def _soak_values_match(actual: Any, expected: Any) -> bool:
    """Compare persisted soak derivations without hiding material drift."""

    if expected is None or isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        parsed = _number(actual)
        return parsed is not None and math.isclose(
            parsed,
            float(expected),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
    if isinstance(expected, Sequence) and not isinstance(
        expected, (str, bytes, bytearray)
    ):
        return (
            isinstance(actual, Sequence)
            and not isinstance(actual, (str, bytes, bytearray))
            and len(actual) == len(expected)
            and all(
                _soak_values_match(observed, wanted)
                for observed, wanted in zip(actual, expected)
            )
        )
    return actual == expected


def _require_soak_value(
    row: Mapping[str, Any], key: str, expected: Any, *, label: str
) -> None:
    if not _soak_values_match(row.get(key), expected):
        raise DirectReportError(
            f"direct soak {label} {key} does not reconcile to raw evidence"
        )


def _soak_wilson_interval(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _soak_t_mean_ci95(values: Sequence[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    critical = {2: 12.706, 3: 4.303, 4: 3.182}.get(len(values), 1.96)
    estimate = statistics.fmean(values)
    radius = critical * statistics.stdev(values) / math.sqrt(len(values))
    return [estimate - radius, estimate + radius]


def _soak_dkw_quantile_ci95(
    values: Sequence[float], quantile: float
) -> list[float] | None:
    if not values:
        return None
    epsilon = math.sqrt(math.log(2.0 / 0.05) / (2.0 * len(values)))
    lower = percentile(values, max(0.0, quantile - epsilon))
    upper = percentile(values, min(1.0, quantile + epsilon))
    if lower is None or upper is None:
        return None
    return [lower, upper]


def _soak_quality_pass(row: Mapping[str, Any]) -> bool:
    score = _number(row.get("quality_score"))
    return row.get("status") == "success" and score is not None and score >= 0.999999


def _validate_soak_phase_against_requests(
    row: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    *,
    expected_scheduled: int,
    allow_incomplete: bool = False,
) -> None:
    """Bind a terminal phase receipt to its exact raw request cohort."""

    label = f"phase {row.get('phase_id')}"
    provider_attempts = sum(
        value.get("provider_send_attempted") is True for value in requests
    )
    successes = [value for value in requests if value.get("status") == "success"]
    quality_passes = sum(_soak_quality_pass(value) for value in successes)
    phase_status = row.get("status")
    incomplete = phase_status == "incomplete"
    if phase_status != "complete" and not (allow_incomplete and incomplete):
        raise DirectReportError(f"direct soak {label} is not complete")
    for key, expected in (
        ("scheduled_requests", expected_scheduled),
        ("completed_request_rows", len(requests)),
        ("provider_send_attempts", provider_attempts),
        ("successes", len(successes)),
        ("quality_passes", quality_passes),
    ):
        _require_soak_value(row, key, expected, label=label)
    if len(requests) != expected_scheduled:
        raise DirectReportError(
            f"direct soak {label} does not contain the complete terminal request cohort"
        )
    if not incomplete and provider_attempts != expected_scheduled:
        raise DirectReportError(
            f"direct soak {label} does not contain the complete provider-send cohort"
        )
    if incomplete:
        unsent = [
            value for value in requests if value.get("provider_send_attempted") is False
        ]
        allowed_unsent_statuses = {
            "skipped_campaign_deadline",
            "skipped_cost_cap",
            "skipped_hard_campaign_deadline",
            "skipped_http_402_latch",
            "skipped_send_deadline",
        }
        if (
            not unsent
            or provider_attempts >= expected_scheduled
            or any(
                value.get("status") not in allowed_unsent_statuses for value in unsent
            )
        ):
            raise DirectReportError(
                f"direct soak {label} has an invalid incomplete-send cohort"
            )
    _require_soak_value(
        row,
        "success_rate",
        len(successes) / len(requests) if requests else 0.0,
        label=label,
    )
    _require_soak_value(
        row,
        "success_rate_ci95_wilson",
        _soak_wilson_interval(len(successes), len(requests)),
        label=label,
    )
    _require_soak_value(
        row,
        "quality_pass_rate",
        quality_passes / len(successes) if successes else 0.0,
        label=label,
    )
    _require_soak_value(
        row,
        "quality_pass_rate_ci95_wilson",
        _soak_wilson_interval(quality_passes, len(successes)),
        label=label,
    )
    elapsed_seconds = _number(row.get("elapsed_seconds_including_drain"))
    if elapsed_seconds is None or elapsed_seconds <= 0:
        raise DirectReportError(f"direct soak {label} has invalid elapsed time")
    elapsed_minutes = elapsed_seconds / 60.0
    input_complete_count = sum(
        value.get("input_usage_complete") is True for value in successes
    )
    output_complete_count = sum(
        value.get("output_usage_complete") is True for value in successes
    )
    input_complete = input_complete_count == len(successes)
    output_complete = output_complete_count == len(successes)
    prompt_tokens = sum(
        int(parse_token_usage(value.get("usage")).get("prompt_tokens") or 0)
        for value in successes
        if value.get("input_usage_complete") is True
    )
    output_tokens = sum(
        int(parse_token_usage(value.get("usage")).get("completion_tokens") or 0)
        for value in successes
        if value.get("output_usage_complete") is True
    )
    ttfts = [
        float(value["timing"]["ttft_seconds"])
        for value in successes
        if _number(_mapping(value.get("timing")).get("ttft_seconds")) is not None
    ]
    latencies = [
        float(value["timing"]["request_seconds"])
        for value in successes
        if _number(_mapping(value.get("timing")).get("request_seconds")) is not None
    ]
    for key, expected in (
        ("successful_rows_with_complete_input_usage", input_complete_count),
        ("successful_rows_with_complete_output_usage", output_complete_count),
        ("input_usage_complete_for_all_successes", input_complete),
        ("output_usage_complete_for_all_successes", output_complete),
        ("successful_rpm", len(successes) / elapsed_minutes),
        (
            "effective_input_tpm",
            prompt_tokens / elapsed_minutes if input_complete else None,
        ),
        (
            "effective_output_tpm",
            output_tokens / elapsed_minutes if output_complete else None,
        ),
        ("ttft_p50_seconds", percentile(ttfts, 0.50)),
        ("ttft_p95_seconds", percentile(ttfts, 0.95)),
        ("ttft_p95_ci95_dkw_seconds", _soak_dkw_quantile_ci95(ttfts, 0.95)),
        ("latency_p50_seconds", percentile(latencies, 0.50)),
        ("latency_p95_seconds", percentile(latencies, 0.95)),
        ("latency_p95_ci95_dkw_seconds", _soak_dkw_quantile_ci95(latencies, 0.95)),
        ("http_429", sum(value.get("http_status") == 429 for value in requests)),
        (
            "http_5xx",
            sum(
                isinstance(value.get("http_status"), int)
                and int(value["http_status"]) >= 500
                for value in requests
            ),
        ),
        (
            "timeouts",
            sum(
                str(value.get("error_type")) in SOAK_TIMEOUT_ERROR_TYPES
                for value in requests
            ),
        ),
    ):
        _require_soak_value(row, key, expected, label=label)


def _validate_soak_block_against_requests(
    row: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    *,
    expected_scheduled: int,
    analysis_block_seconds: float,
) -> None:
    """Recompute every public block statistic from its exact arrival cohort."""

    label = f"analysis block {row.get('analysis_block_id')}"
    successes = [value for value in requests if value.get("status") == "success"]
    provider_attempts = sum(
        value.get("provider_send_attempted") is True for value in requests
    )
    quality_passes = sum(_soak_quality_pass(value) for value in successes)
    if len(requests) != expected_scheduled or provider_attempts != expected_scheduled:
        raise DirectReportError(
            f"direct soak {label} does not contain the complete provider-send cohort"
        )
    for key, expected in (
        ("analysis_block_seconds", analysis_block_seconds),
        ("scheduled_requests", expected_scheduled),
        ("completed_request_rows", len(requests)),
        ("successes", len(successes)),
        ("quality_passes", quality_passes),
    ):
        _require_soak_value(row, key, expected, label=label)
    _require_soak_value(
        row,
        "success_rate",
        len(successes) / len(requests) if requests else 0.0,
        label=label,
    )
    _require_soak_value(
        row,
        "success_rate_ci95_wilson",
        _soak_wilson_interval(len(successes), len(requests)),
        label=label,
    )
    _require_soak_value(
        row,
        "quality_pass_rate",
        quality_passes / len(successes) if successes else 0.0,
        label=label,
    )
    _require_soak_value(
        row,
        "quality_pass_rate_ci95_wilson",
        _soak_wilson_interval(quality_passes, len(successes)),
        label=label,
    )
    interval_minutes = analysis_block_seconds / 60.0
    _require_soak_value(
        row,
        "offered_rps_realized_schedule",
        expected_scheduled / analysis_block_seconds,
        label=label,
    )
    _require_soak_value(
        row,
        "successful_rpm_per_predeclared_window",
        len(successes) / interval_minutes,
        label=label,
    )
    block_index = _integer(row.get("analysis_block_index"))
    if block_index is None or block_index < 0:
        raise DirectReportError(f"direct soak {label} has an invalid block index")
    cohort_drain_seconds = max(
        analysis_block_seconds,
        max(
            (
                float(
                    _mapping(value.get("load")).get("scheduled_offset_seconds") or 0.0
                )
                - block_index * analysis_block_seconds
                + float(_mapping(value.get("load")).get("schedule_lag_seconds") or 0.0)
                + float(_mapping(value.get("timing")).get("request_seconds") or 0.0)
                for value in requests
            ),
            default=0.0,
        ),
    )
    drain_minutes = cohort_drain_seconds / 60.0
    _require_soak_value(
        row,
        "arrival_cohort_elapsed_seconds_including_drain",
        cohort_drain_seconds,
        label=label,
    )
    _require_soak_value(
        row,
        "arrival_cohort_successful_rpm_including_drain",
        len(successes) / drain_minutes,
        label=label,
    )
    input_complete_count = sum(
        value.get("input_usage_complete") is True for value in successes
    )
    output_complete_count = sum(
        value.get("output_usage_complete") is True for value in successes
    )
    input_complete = input_complete_count == len(successes)
    output_complete = output_complete_count == len(successes)
    prompt_tokens = sum(
        int(parse_token_usage(value.get("usage")).get("prompt_tokens") or 0)
        for value in successes
        if value.get("input_usage_complete") is True
    )
    output_tokens = sum(
        int(parse_token_usage(value.get("usage")).get("completion_tokens") or 0)
        for value in successes
        if value.get("output_usage_complete") is True
    )
    for key, expected in (
        ("successful_rows_with_complete_input_usage", input_complete_count),
        ("successful_rows_with_complete_output_usage", output_complete_count),
        ("input_usage_complete_for_all_successes", input_complete),
        ("output_usage_complete_for_all_successes", output_complete),
        (
            "effective_input_tpm_per_predeclared_window",
            prompt_tokens / interval_minutes if input_complete else None,
        ),
        (
            "effective_output_tpm_per_predeclared_window",
            output_tokens / interval_minutes if output_complete else None,
        ),
        (
            "arrival_cohort_effective_input_tpm_including_drain",
            prompt_tokens / drain_minutes if input_complete else None,
        ),
        (
            "arrival_cohort_effective_output_tpm_including_drain",
            output_tokens / drain_minutes if output_complete else None,
        ),
    ):
        _require_soak_value(row, key, expected, label=label)
    ttfts = [
        float(value["timing"]["ttft_seconds"])
        for value in successes
        if _number(_mapping(value.get("timing")).get("ttft_seconds")) is not None
    ]
    latencies = [
        float(value["timing"]["request_seconds"])
        for value in successes
        if _number(_mapping(value.get("timing")).get("request_seconds")) is not None
    ]
    for key, expected in (
        ("ttft_p50_seconds", percentile(ttfts, 0.50)),
        ("ttft_p95_seconds", percentile(ttfts, 0.95)),
        ("ttft_p95_ci95_dkw_seconds", _soak_dkw_quantile_ci95(ttfts, 0.95)),
        ("latency_p50_seconds", percentile(latencies, 0.50)),
        ("latency_p95_seconds", percentile(latencies, 0.95)),
        ("latency_p95_ci95_dkw_seconds", _soak_dkw_quantile_ci95(latencies, 0.95)),
        ("http_429", sum(value.get("http_status") == 429 for value in requests)),
        (
            "http_5xx",
            sum(
                isinstance(value.get("http_status"), int)
                and int(value["http_status"]) >= 500
                for value in requests
            ),
        ),
        (
            "timeouts",
            sum(
                str(value.get("error_type")) in SOAK_TIMEOUT_ERROR_TYPES
                for value in requests
            ),
        ),
    ):
        _require_soak_value(row, key, expected, label=label)


def _soak_phase_summary(row: Mapping[str, Any], *, source_id: str) -> dict[str, Any]:
    return {
        "schema_version": "digitalocean_public_soak_phase_summary_v1",
        "source_kind": "direct_soak",
        "source_id": source_id,
        "sampling_unit": "phase_id",
        "within_phase_binomial_interval_sampling_unit": "request_id",
        "phase_id": row.get("phase_id"),
        "cell_id": row.get("cell_id"),
        "endpoint_id": row.get("model_id"),
        "shape": row.get("shape"),
        "phase": row.get("phase"),
        "status": row.get("status"),
        "scheduled_requests": row.get("scheduled_requests"),
        "completed_request_rows": row.get("completed_request_rows"),
        "provider_send_attempts": row.get("provider_send_attempts"),
        "successes": row.get("successes"),
        "success_rate": row.get("success_rate"),
        "success_rate_ci95_wilson": row.get("success_rate_ci95_wilson"),
        "quality_passes": row.get("quality_passes"),
        "quality_pass_rate": row.get("quality_pass_rate"),
        "quality_pass_rate_ci95_wilson": row.get("quality_pass_rate_ci95_wilson"),
        "offered_rps_target": row.get("offered_rps_target"),
        "offered_rps_realized_schedule": row.get("offered_rps_realized_schedule"),
        "successful_rpm": row.get("successful_rpm"),
        "effective_input_tpm": row.get("effective_input_tpm"),
        "effective_output_tpm": row.get("effective_output_tpm"),
        "ttft_p50_seconds": row.get("ttft_p50_seconds"),
        "ttft_p95_seconds": row.get("ttft_p95_seconds"),
        "ttft_p95_ci95_dkw_seconds": row.get("ttft_p95_ci95_dkw_seconds"),
        "latency_p50_seconds": row.get("latency_p50_seconds"),
        "latency_p95_seconds": row.get("latency_p95_seconds"),
        "latency_p95_ci95_dkw_seconds": row.get("latency_p95_ci95_dkw_seconds"),
        "claim_scope": row.get("claim_scope"),
    }


def load_soak_directory(
    path: Path,
    *,
    source_id_override: str | None = None,
    allow_incomplete_terminal: bool = False,
) -> dict[str, Any]:
    """Strictly load one terminal direct-soak evidence directory.

    Soak blocks remain ``analysis_block_id`` sampling units.  They are never
    converted to AIMD confirmation epochs, and quality is reconciled through
    exact low/near request pairs instead of treating all soak arrivals as
    independently scored observations.
    """

    directory = Path(path)
    source_id = _text(source_id_override) or directory.name
    manifest_path = directory / "manifest.json"
    plan_path = directory / "plan.json"
    summary_path = directory / "summary.json"
    if (
        not manifest_path.is_file()
        or not plan_path.is_file()
        or not summary_path.is_file()
    ):
        raise DirectReportError(
            "direct soak requires manifest.json, plan.json, and terminal summary.json"
        )
    manifest = _read_json(manifest_path)
    plan = _read_json(plan_path)
    summary = _read_json(summary_path)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != SOAK_MANIFEST_SCHEMA
    ):
        raise DirectReportError("direct soak manifest schema is invalid")
    if not isinstance(plan, Mapping) or plan.get("schema_version") != SOAK_PLAN_SCHEMA:
        raise DirectReportError("direct soak plan schema is invalid")
    if (
        not isinstance(summary, Mapping)
        or summary.get("schema_version") != SOAK_SUMMARY_SCHEMA
    ):
        raise DirectReportError("direct soak terminal summary schema is invalid")
    campaign_id = _text(manifest.get("campaign_id"))
    plan_sha256 = _text(plan.get("plan_sha256"))
    plan_identity = dict(plan)
    plan_identity.pop("plan_sha256", None)
    computed_plan_sha256 = hashlib.sha256(
        canonical_json(plan_identity).encode("utf-8")
    ).hexdigest()
    computed_campaign_id = f"do-soak-{computed_plan_sha256[:20]}"
    if (
        campaign_id is None
        or plan_sha256 is None
        or not re.fullmatch(r"[0-9a-f]{64}", plan_sha256)
        or plan_sha256 != computed_plan_sha256
        or campaign_id != computed_campaign_id
        or manifest.get("plan_sha256") != plan_sha256
    ):
        raise DirectReportError(
            "direct soak manifest, campaign, and recomputed plan identities disagree"
        )
    analysis_block_count = _integer(plan.get("analysis_block_count"))
    analysis_block_seconds = _number(plan.get("analysis_block_seconds"))
    soak_seconds = _number(plan.get("soak_seconds"))
    if (
        analysis_block_count != 4
        or analysis_block_seconds is None
        or analysis_block_seconds <= 0
        or soak_seconds is None
        or not math.isclose(
            soak_seconds,
            analysis_block_count * analysis_block_seconds,
            rel_tol=0,
            abs_tol=1e-10,
        )
    ):
        raise DirectReportError(
            "direct soak plan has an invalid four-block time contract"
        )
    raw_cells = plan.get("cells")
    if not isinstance(raw_cells, Sequence) or isinstance(
        raw_cells, (str, bytes, bytearray)
    ):
        raise DirectReportError("direct soak plan cells must be a list")
    plan_cells: dict[str, dict[str, Any]] = {}
    plan_cell_pairs: set[tuple[str, str]] = set()
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, Mapping):
            raise DirectReportError("direct soak plan cell must be an object")
        cell = dict(raw_cell)
        cell_id = _text(cell.get("cell_id"))
        endpoint = _require_endpoint(cell.get("model_id"))
        shape = _text(cell.get("shape"))
        if (
            cell_id is None
            or cell_id in plan_cells
            or shape not in {"short_short", "input32k_short", "short_long", "mixed"}
            or (endpoint, str(shape)) in plan_cell_pairs
        ):
            raise DirectReportError("direct soak plan has a duplicate or invalid cell")
        if cell.get("status") not in {"ready", "blocked"}:
            raise DirectReportError("direct soak plan cell status is invalid")
        counts = {
            count_key: _soak_count(cell, count_key)
            for count_key in (
                "low_load_requests",
                "soak_requests",
                "recovery_requests",
                "total_requests",
            )
        }
        raw_block_counts = cell.get("soak_block_request_counts")
        if not isinstance(raw_block_counts, Sequence) or isinstance(
            raw_block_counts, (str, bytes, bytearray)
        ):
            raise DirectReportError("direct soak plan cell lacks block request counts")
        block_counts = [_integer(value) for value in raw_block_counts]
        if (
            len(block_counts) != analysis_block_count
            or any(value is None for value in block_counts)
            or sum(int(value) for value in block_counts if value is not None)
            != counts["soak_requests"]
            or counts["total_requests"]
            != counts["low_load_requests"]
            + counts["soak_requests"]
            + counts["recovery_requests"]
        ):
            raise DirectReportError(
                "direct soak plan cell request-count algebra is invalid"
            )
        cell["soak_block_request_counts"] = [int(value) for value in block_counts]
        plan_cells[cell_id] = cell
        plan_cell_pairs.add((endpoint, str(shape)))

    raw_models = plan.get("models")
    if not isinstance(raw_models, Sequence) or isinstance(
        raw_models, (str, bytes, bytearray)
    ):
        raise DirectReportError("direct soak plan models must be a list")
    models = [_require_endpoint(value) for value in raw_models]
    cell_models = {str(cell["model_id"]) for cell in plan_cells.values()}
    selected_cells = plan.get("selected_cells")
    if len(models) != len(set(models)) or not cell_models.issubset(set(models)):
        raise DirectReportError("direct soak plan model inventory is inconsistent")
    if selected_cells is None:
        if set(models) != cell_models:
            raise DirectReportError("direct soak plan model inventory is inconsistent")
    else:
        if not isinstance(selected_cells, Sequence) or isinstance(
            selected_cells, (str, bytes, bytearray)
        ):
            raise DirectReportError("direct soak selected cells must be a list")
        selected_pairs: list[tuple[str, str]] = []
        for raw_selected in selected_cells:
            selected = _text(raw_selected)
            if selected is None or ":" not in selected:
                raise DirectReportError("direct soak selected cell is invalid")
            raw_endpoint, raw_shape = selected.rsplit(":", 1)
            endpoint = _require_endpoint(raw_endpoint)
            if raw_shape not in {
                "short_short",
                "input32k_short",
                "short_long",
                "mixed",
            }:
                raise DirectReportError("direct soak selected cell is invalid")
            selected_pairs.append((endpoint, raw_shape))
        if (
            len(selected_pairs) != len(set(selected_pairs))
            or set(selected_pairs) != plan_cell_pairs
        ):
            raise DirectReportError(
                "direct soak selected cells do not match the science plan"
            )

    raw_requests = _soak_rows_by_id(
        directory / "requests.jsonl", identity="request_id", schema=SOAK_REQUEST_SCHEMA
    )
    raw_phases = _soak_rows_by_id(
        directory / "phases.jsonl", identity="phase_id", schema=SOAK_PHASE_SCHEMA
    )
    raw_blocks = _soak_rows_by_id(
        directory / "analysis-blocks.jsonl",
        identity="analysis_block_id",
        schema=SOAK_BLOCK_SCHEMA,
    )
    raw_pairs = _soak_rows_by_id(
        directory / "quality-pairs.jsonl",
        identity="quality_pair_id",
        schema=SOAK_QUALITY_PAIR_SCHEMA,
    )
    raw_cell_rows = _soak_rows_by_id(
        directory / "cells.jsonl", identity="cell_id", schema=SOAK_CELL_SCHEMA
    )
    if not set(raw_cell_rows).issubset(plan_cells) or (
        not allow_incomplete_terminal and set(raw_cell_rows) != set(plan_cells)
    ):
        raise DirectReportError("direct soak cell journal does not match plan cells")

    phases_by_cell_and_name: dict[tuple[str, str], Mapping[str, Any]] = {}
    phase_by_id: dict[str, Mapping[str, Any]] = {}
    phase_summaries: list[dict[str, Any]] = []
    for phase_id, row in raw_phases.items():
        cell_id = _text(row.get("cell_id"))
        cell = plan_cells.get(str(cell_id))
        phase = _text(row.get("phase"))
        if cell is None or phase not in SOAK_PHASES:
            raise DirectReportError("direct soak phase is outside the science plan")
        _soak_identity(
            row,
            campaign_id=campaign_id,
            plan_sha256=plan_sha256,
            cell=cell,
            label="phase",
        )
        key = (str(cell_id), str(phase))
        if key in phases_by_cell_and_name:
            raise DirectReportError("direct soak has duplicate cell/phase evidence")
        scheduled = _soak_count(row, "scheduled_requests")
        completed = _soak_count(row, "completed_request_rows")
        attempts = _soak_count(row, "provider_send_attempts")
        successes = _soak_count(row, "successes")
        if successes > completed or attempts > scheduled or completed != scheduled:
            raise DirectReportError("direct soak phase sample counts are inconsistent")
        phases_by_cell_and_name[key] = row
        phase_by_id[phase_id] = row
        phase_summaries.append(_soak_phase_summary(row, source_id=source_id))

    block_by_cell_and_index: dict[tuple[str, int], Mapping[str, Any]] = {}
    block_summaries: list[dict[str, Any]] = []
    normalized_epochs: list[dict[str, Any]] = []
    for phase_id, row in raw_phases.items():
        if row.get("phase") == "two_minute_soak":
            continue
        phase_copy = dict(row)
        phase_copy["epoch_id"] = phase_id
        phase_copy["workload"] = f"direct_soak_{row.get('phase')}"
        phase_copy["valid_for_capacity"] = False
        phase_copy["completed_requests"] = row.get("completed_request_rows")
        phase_copy["success_count"] = row.get("successes")
        phase_copy["quality_pass_count"] = row.get("quality_passes")
        phase_copy["timeout_count"] = row.get("timeouts")
        normalized = normalize_epoch(
            phase_copy, source_kind="direct_soak", source_id=source_id
        )
        normalized["sampling_unit"] = "phase_id"
        normalized["phase_id"] = phase_id
        normalized_epochs.append(normalized)
    for block_id, row in raw_blocks.items():
        cell_id = _text(row.get("cell_id"))
        cell = plan_cells.get(str(cell_id))
        block_index = _integer(row.get("analysis_block_index"))
        if cell is None or block_index not in {0, 1, 2, 3}:
            raise DirectReportError(
                "direct soak analysis block is outside the science plan"
            )
        _soak_identity(
            row,
            campaign_id=campaign_id,
            plan_sha256=plan_sha256,
            cell=cell,
            label="analysis block",
        )
        if row.get("phase") != "two_minute_soak":
            raise DirectReportError("direct soak analysis block has an invalid phase")
        key = (str(cell_id), int(block_index))
        if key in block_by_cell_and_index:
            raise DirectReportError("direct soak has duplicate cell/block evidence")
        scheduled = _soak_count(row, "scheduled_requests")
        completed = _soak_count(row, "completed_request_rows")
        successes = _soak_count(row, "successes")
        if completed != scheduled or successes > completed:
            raise DirectReportError("direct soak block sample counts are inconsistent")
        block_by_cell_and_index[key] = row
        normalized_source = dict(row)
        normalized_source.update(
            {
                "epoch_id": block_id,
                "workload": "direct_two_minute_soak",
                "completed_requests": row.get("completed_request_rows"),
                "success_count": row.get("successes"),
                "quality_pass_count": row.get("quality_passes"),
                "timeout_count": row.get("timeouts"),
                "offered_rps_target": row.get("candidate_rate_rps"),
                "elapsed_seconds": row.get(
                    "arrival_cohort_elapsed_seconds_including_drain"
                ),
                "effective_input_tpm": row.get(
                    "arrival_cohort_effective_input_tpm_including_drain"
                ),
                "effective_output_tpm": row.get(
                    "arrival_cohort_effective_output_tpm_including_drain"
                ),
                "successful_rpm": row.get(
                    "arrival_cohort_successful_rpm_including_drain"
                ),
                "healthy": row.get("predeclared_acceptance_pass"),
                "health_reasons": row.get("acceptance_reasons"),
                "valid_for_capacity": False,
            }
        )
        normalized = normalize_epoch(
            normalized_source, source_kind="direct_soak", source_id=source_id
        )
        normalized["sampling_unit"] = "analysis_block_id"
        normalized["analysis_block_id"] = block_id
        normalized["analysis_block_index"] = block_index
        normalized_epochs.append(normalized)
        block_summaries.append(
            {
                "schema_version": "digitalocean_public_soak_block_summary_v1",
                "source_kind": "direct_soak",
                "source_id": source_id,
                "sampling_unit": "analysis_block_id",
                "analysis_block_id": block_id,
                "analysis_block_index": block_index,
                "cell_id": cell_id,
                "endpoint_id": row.get("model_id"),
                "shape": row.get("shape"),
                "candidate_rate_rps": row.get("candidate_rate_rps"),
                "scheduled_requests": scheduled,
                "completed_request_rows": completed,
                "successes": successes,
                "success_rate": row.get("success_rate"),
                "success_rate_ci95_wilson": row.get("success_rate_ci95_wilson"),
                "quality_passes": row.get("quality_passes"),
                "quality_pass_rate": row.get("quality_pass_rate"),
                "quality_pass_rate_ci95_wilson": row.get(
                    "quality_pass_rate_ci95_wilson"
                ),
                "predeclared_acceptance_pass": row.get("predeclared_acceptance_pass"),
                "acceptance_reasons": row.get("acceptance_reasons"),
                "offered_rps_realized_schedule": row.get(
                    "offered_rps_realized_schedule"
                ),
                "arrival_window_successful_rpm_accounting": row.get(
                    "successful_rpm_per_predeclared_window"
                ),
                "arrival_window_effective_input_tpm_accounting": row.get(
                    "effective_input_tpm_per_predeclared_window"
                ),
                "arrival_window_effective_output_tpm_accounting": row.get(
                    "effective_output_tpm_per_predeclared_window"
                ),
                "arrival_cohort_elapsed_seconds_including_drain": row.get(
                    "arrival_cohort_elapsed_seconds_including_drain"
                ),
                "arrival_cohort_successful_rpm_including_drain": row.get(
                    "arrival_cohort_successful_rpm_including_drain"
                ),
                "arrival_cohort_effective_input_tpm_including_drain": row.get(
                    "arrival_cohort_effective_input_tpm_including_drain"
                ),
                "arrival_cohort_effective_output_tpm_including_drain": row.get(
                    "arrival_cohort_effective_output_tpm_including_drain"
                ),
                "ttft_p50_seconds": row.get("ttft_p50_seconds"),
                "ttft_p95_seconds": row.get("ttft_p95_seconds"),
                "ttft_p95_ci95_dkw_seconds": row.get("ttft_p95_ci95_dkw_seconds"),
                "latency_p50_seconds": row.get("latency_p50_seconds"),
                "latency_p95_seconds": row.get("latency_p95_seconds"),
                "latency_p95_ci95_dkw_seconds": row.get("latency_p95_ci95_dkw_seconds"),
                "claim_scope": row.get("claim_scope"),
            }
        )

    normalized_requests: list[dict[str, Any]] = []
    requests_by_pair_role: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    requests_by_phase_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    requests_by_cell_and_block: dict[tuple[str, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for request_id, row in raw_requests.items():
        cell_id = _text(row.get("cell_id"))
        cell = plan_cells.get(str(cell_id))
        phase = _text(row.get("phase"))
        phase_id = _text(row.get("phase_id"))
        if cell is None or phase not in SOAK_PHASES or phase_id not in phase_by_id:
            raise DirectReportError("direct soak request is outside a persisted phase")
        _soak_identity(
            row,
            campaign_id=campaign_id,
            plan_sha256=plan_sha256,
            cell=cell,
            label="request",
        )
        if (
            phase_by_id[str(phase_id)].get("cell_id") != cell_id
            or phase_by_id[str(phase_id)].get("phase") != phase
        ):
            raise DirectReportError(
                "direct soak request phase identity is inconsistent"
            )
        if not isinstance(row.get("provider_send_attempted"), bool):
            raise DirectReportError("direct soak request lacks a provider-send flag")
        status = _text(row.get("status"))
        if (
            status is None
            or "unknown" in status.casefold()
            or row.get("http_status") == 402
        ):
            raise DirectReportError(
                "direct soak request has an inadmissible terminal status"
            )
        if phase == "two_minute_soak":
            tags = _mapping(row.get("workload_tags"))
            block_index = _integer(tags.get("analysis_block_index"))
            block = block_by_cell_and_index.get((str(cell_id), int(block_index or 0)))
            if block_index not in {0, 1, 2, 3} or block is None:
                raise DirectReportError(
                    "direct soak request does not map to an exact analysis block"
                )
            sampling_parent_id = str(block.get("analysis_block_id"))
            sampling_unit = "analysis_block_id"
            requests_by_cell_and_block[(str(cell_id), int(block_index))].append(row)
        else:
            block_index = None
            sampling_parent_id = str(phase_id)
            sampling_unit = "phase_id"
        request_copy = dict(row)
        request_copy["epoch_id"] = sampling_parent_id
        normalized = normalize_request(
            request_copy, source_kind="direct_soak", source_id=source_id
        )
        normalized.update(
            {
                "phase_id": phase_id,
                "analysis_block_id": (
                    sampling_parent_id if sampling_unit == "analysis_block_id" else None
                ),
                "analysis_block_index": block_index,
                "sampling_unit": sampling_unit,
                "quality_pair_id": _text(row.get("quality_pair_id")),
                "quality_pair_index": _integer(row.get("quality_pair_index")),
                "quality_pair_role": _text(row.get("quality_pair_role")),
            }
        )
        normalized_requests.append(normalized)
        requests_by_phase_id[str(phase_id)].append(row)
        pair_id = _text(row.get("quality_pair_id"))
        pair_role = _text(row.get("quality_pair_role"))
        pair_index = _integer(row.get("quality_pair_index"))
        if pair_id is not None or pair_role is not None or pair_index is not None:
            if (
                pair_id is None
                or pair_role not in {"low_load", "near_load"}
                or pair_index is None
                or pair_index >= _soak_count(cell, "low_load_requests")
            ):
                raise DirectReportError(
                    "direct soak request has an incomplete quality-pair identity"
                )
            tags = _mapping(row.get("workload_tags"))
            expected_pair_block = (
                3 if cell.get("shape") == "mixed" and pair_index == 4 else pair_index
            )
            if (
                _integer(tags.get("paired_analysis_block_index")) != expected_pair_block
                or (pair_role == "low_load" and phase != "paired_low_load")
                or (pair_role == "near_load" and phase != "two_minute_soak")
                or (
                    pair_role == "near_load"
                    and _integer(tags.get("analysis_block_index"))
                    != expected_pair_block
                )
            ):
                raise DirectReportError(
                    "direct soak request quality-pair role, phase, or block is inconsistent"
                )
            cell_result = raw_cell_rows.get(str(cell_id))
            if cell_result is None:
                if pair_id in raw_pairs:
                    requests_by_pair_role[(pair_id, pair_role)].append(row)
                # A deadline-censored cell can persist provisional pair IDs before
                # it has enough evidence to write a pair or terminal cell row.
                # Preserve the physical request for cost/reliability accounting,
                # but do not manufacture pair evidence.
            elif cell_result.get("status") == "baseline_transport_gate_failed":
                if pair_role != "low_load":
                    raise DirectReportError(
                        "baseline-gated soak cell contains impossible pair evidence"
                    )
                # These low-load requests were assigned provisional pair IDs before
                # the transport gate stopped the soak. They are baseline evidence,
                # not completed quality pairs, and therefore have no pair journal.
            else:
                requests_by_pair_role[(pair_id, pair_role)].append(row)

    expected_phase_counts = {
        "paired_low_load": "low_load_requests",
        "two_minute_soak": "soak_requests",
        "post_soak_recovery": "recovery_requests",
    }
    for phase_id, row in raw_phases.items():
        cell = plan_cells[str(row["cell_id"])]
        count_key = expected_phase_counts[str(row["phase"])]
        _validate_soak_phase_against_requests(
            row,
            requests_by_phase_id.get(phase_id, ()),
            expected_scheduled=_soak_count(cell, count_key),
            allow_incomplete=allow_incomplete_terminal,
        )
    for (cell_id, block_index), row in block_by_cell_and_index.items():
        plan_cell = plan_cells[cell_id]
        _require_soak_value(
            row,
            "candidate_rate_rps",
            plan_cell.get("candidate_rate_rps"),
            label=f"analysis block {row.get('analysis_block_id')}",
        )
        _validate_soak_block_against_requests(
            row,
            requests_by_cell_and_block.get((cell_id, block_index), ()),
            expected_scheduled=int(plan_cell["soak_block_request_counts"][block_index]),
            analysis_block_seconds=float(analysis_block_seconds),
        )

    quality_summaries: list[dict[str, Any]] = []
    pairs_by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pair_id, row in raw_pairs.items():
        cell_id = _text(row.get("cell_id"))
        cell = plan_cells.get(str(cell_id))
        block_index = _integer(row.get("analysis_block_index"))
        pair_index = _integer(row.get("quality_pair_index"))
        expected_block_index = (
            3
            if cell is not None and cell.get("shape") == "mixed" and pair_index == 4
            else pair_index
        )
        if (
            cell is None
            or block_index not in {0, 1, 2, 3}
            or pair_index is None
            or pair_index >= _soak_count(cell, "low_load_requests")
            or block_index != expected_block_index
        ):
            raise DirectReportError(
                "direct soak quality pair is outside the science plan"
            )
        _soak_identity(
            row,
            campaign_id=campaign_id,
            plan_sha256=plan_sha256,
            cell=cell,
            label="quality pair",
        )
        low = requests_by_pair_role.get((pair_id, "low_load"), [])
        near = requests_by_pair_role.get((pair_id, "near_load"), [])
        if len(low) != 1 or len(near) != 1:
            raise DirectReportError(
                "direct soak quality pair requires exactly one low-load and one near-load request"
            )
        if (
            low[0].get("request_id") != row.get("low_load_request_id")
            or near[0].get("request_id") != row.get("near_load_request_id")
            or low[0].get("cell_id") != cell_id
            or near[0].get("cell_id") != cell_id
            or _mapping(near[0].get("workload_tags")).get("analysis_block_index")
            != block_index
            or _mapping(low[0].get("workload_tags")).get("paired_analysis_block_index")
            != block_index
            or _mapping(near[0].get("workload_tags")).get("paired_analysis_block_index")
            != block_index
            or _integer(low[0].get("quality_pair_index")) != pair_index
            or _integer(near[0].get("quality_pair_index")) != pair_index
        ):
            raise DirectReportError(
                "direct soak quality pair request identities disagree"
            )
        low_hash = _text(low[0].get("request_payload_sha256"))
        near_hash = _text(near[0].get("request_payload_sha256"))
        if (
            row.get("exact_request_payload_hash_match") is not True
            or low_hash is None
            or low_hash != near_hash
        ):
            raise DirectReportError(
                "direct soak quality pair payload hashes do not match"
            )
        block = block_by_cell_and_index.get((str(cell_id), int(block_index)))
        if block is None:
            raise DirectReportError("direct soak quality pair lacks its analysis block")
        low_success = low[0].get("status") == "success"
        near_success = near[0].get("status") == "success"
        low_score = _number(low[0].get("quality_score"))
        near_score = _number(near[0].get("quality_score"))
        quality_delta = (
            near_score - low_score
            if low_score is not None and near_score is not None
            else None
        )
        low_pass = low_success and low_score is not None and low_score >= 0.999999
        near_pass = near_success and near_score is not None and near_score >= 0.999999
        expected_reasons: list[str] = []
        if not low_pass:
            expected_reasons.append("paired_low_load_quality_failure")
        if not near_pass:
            expected_reasons.append("paired_near_load_quality_failure")
        low_latency = _number(_mapping(low[0].get("timing")).get("request_seconds"))
        near_latency = _number(_mapping(near[0].get("timing")).get("request_seconds"))
        expected_latency_ratio = (
            near_latency / low_latency
            if low_latency is not None and low_latency > 0 and near_latency is not None
            else None
        )
        for key, expected in (
            ("status", "complete"),
            ("low_load_success", low_success),
            ("near_load_success", near_success),
            ("low_load_quality_score", low[0].get("quality_score")),
            ("near_load_quality_score", near[0].get("quality_score")),
            ("paired_quality_delta_near_minus_low", quality_delta),
            ("paired_latency_ratio_near_over_low", expected_latency_ratio),
            ("predeclared_quality_acceptance_pass", not expected_reasons),
            ("quality_acceptance_reasons", expected_reasons),
        ):
            _require_soak_value(row, key, expected, label=f"quality pair {pair_id}")
        pairs_by_cell[str(cell_id)].append(row)
        quality_summaries.append(
            {
                "schema_version": "digitalocean_public_soak_quality_pair_summary_v1",
                "source_kind": "direct_soak",
                "source_id": source_id,
                "sampling_unit": "quality_pair_id",
                "quality_pair_id": pair_id,
                "cell_id": cell_id,
                "endpoint_id": row.get("model_id"),
                "shape": row.get("shape"),
                "analysis_block_id": block.get("analysis_block_id"),
                "analysis_block_index": block_index,
                "status": row.get("status"),
                "exact_request_payload_hash_match": True,
                "low_load_success": row.get("low_load_success"),
                "near_load_success": row.get("near_load_success"),
                "low_load_quality_score": row.get("low_load_quality_score"),
                "near_load_quality_score": row.get("near_load_quality_score"),
                "paired_quality_delta_near_minus_low": row.get(
                    "paired_quality_delta_near_minus_low"
                ),
                "predeclared_quality_acceptance_pass": row.get(
                    "predeclared_quality_acceptance_pass"
                ),
                "quality_acceptance_reasons": row.get("quality_acceptance_reasons"),
                "claim_scope": row.get("claim_scope"),
            }
        )

    expected_pair_role_keys = {
        (pair_id, role) for pair_id in raw_pairs for role in ("low_load", "near_load")
    }
    if set(requests_by_pair_role) != expected_pair_role_keys:
        raise DirectReportError(
            "direct soak request pair identities do not exactly match the pair journal"
        )

    pairs_by_cell_and_block: dict[tuple[str, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for cell_id, rows in pairs_by_cell.items():
        pair_indices = [_integer(row.get("quality_pair_index")) for row in rows]
        if len(pair_indices) != len(set(pair_indices)):
            raise DirectReportError("direct soak has duplicate quality-pair indices")
        for row in rows:
            block_index = _integer(row.get("analysis_block_index"))
            if block_index is None:
                raise DirectReportError(
                    "direct soak quality pair lacks its block index"
                )
            pairs_by_cell_and_block[(cell_id, block_index)].append(row)

    for (cell_id, block_index), block in block_by_cell_and_index.items():
        block_requests = sorted(
            requests_by_cell_and_block[(cell_id, block_index)],
            key=lambda row: (
                float(_mapping(row.get("load")).get("scheduled_offset_seconds") or 0.0),
                str(row.get("request_id") or ""),
            ),
        )
        successes = [row for row in block_requests if row.get("status") == "success"]
        lags = [
            float(_mapping(row.get("load")).get("schedule_lag_seconds") or 0.0)
            for row in block_requests
        ]
        midpoint = max(1, len(lags) // 2)
        queue_growth = (percentile(lags[midpoint:], 0.5) or 0.0) - (
            percentile(lags[:midpoint], 0.5) or 0.0
        )
        _require_soak_value(
            block,
            "schedule_lag_p95_seconds",
            percentile(lags, 0.95),
            label=f"analysis block {block.get('analysis_block_id')}",
        )
        _require_soak_value(
            block,
            "queue_growth_late_minus_early_median_seconds",
            queue_growth,
            label=f"analysis block {block.get('analysis_block_id')}",
        )
        reasons: list[str] = []
        total = len(block_requests)
        if (
            total
            and (
                sum(
                    isinstance(row.get("http_status"), int)
                    and int(row["http_status"]) >= 500
                    for row in block_requests
                )
                + sum(
                    str(row.get("error_type")) in SOAK_TIMEOUT_ERROR_TYPES
                    for row in block_requests
                )
            )
            / total
            > 0.01
        ):
            reasons.append("combined_timeout_5xx_rate_above_0.01")
        if (
            total
            and sum(row.get("http_status") == 429 for row in block_requests) / total
            > 0.01
        ):
            reasons.append("rate_limit_rate_above_0.01")
        if total and len(successes) / total < 0.99:
            reasons.insert(0, "success_rate_below_0.99")
        baseline = phases_by_cell_and_name[(cell_id, "paired_low_load")]
        block_ttft = _number(block.get("ttft_p95_seconds"))
        baseline_ttft = _number(baseline.get("ttft_p95_seconds"))
        if (
            block_ttft is not None
            and baseline_ttft is not None
            and block_ttft > 2 * baseline_ttft
        ):
            reasons.append("ttft_p95_above_2x_paired_low_load_phase")
        block_latency = _number(block.get("latency_p95_seconds"))
        baseline_latency = _number(baseline.get("latency_p95_seconds"))
        if (
            block_latency is not None
            and baseline_latency is not None
            and block_latency > 2 * baseline_latency
        ):
            reasons.append("latency_p95_above_2x_paired_low_load_phase")
        candidate_rate = _number(block.get("candidate_rate_rps"))
        if candidate_rate is None or candidate_rate <= 0:
            raise DirectReportError(
                "direct soak analysis block has invalid candidate rate"
            )
        if queue_growth > max(0.25, 1.0 / candidate_rate):
            reasons.append("arrival_queue_growth")
        pair_details: list[dict[str, Any]] = []
        block_pairs = sorted(
            pairs_by_cell_and_block.get((cell_id, block_index), ()),
            key=lambda row: int(row["quality_pair_index"]),
        )
        for pair in block_pairs:
            pair_id = str(pair["quality_pair_id"])
            low = requests_by_pair_role[(pair_id, "low_load")][0]
            near = requests_by_pair_role[(pair_id, "near_load")][0]
            low_score = _number(low.get("quality_score"))
            near_score = _number(near.get("quality_score"))
            delta = (
                near_score - low_score
                if low_score is not None and near_score is not None
                else None
            )
            low_pass = _soak_quality_pass(low)
            near_pass = _soak_quality_pass(near)
            if not low_pass:
                reasons.append("paired_low_load_quality_failure")
            if not near_pass:
                reasons.append("paired_near_load_quality_failure")
            if delta is not None and delta < -1e-12:
                reasons.append("paired_quality_regression_near_vs_low")
            pair_details.append(
                {
                    "quality_pair_id": pair_id,
                    "quality_pair_index": pair.get("quality_pair_index"),
                    "task_family": low.get("task_family"),
                    "complete": True,
                    "exact_payload_hash_match": True,
                    "low_load_quality_score": low_score,
                    "near_load_quality_score": near_score,
                    "low_load_quality_pass": low_pass,
                    "near_load_quality_pass": near_pass,
                    "quality_delta_near_minus_low": delta,
                }
            )
        reasons = list(dict.fromkeys(reasons))
        block_label = f"analysis block {block.get('analysis_block_id')}"
        for key, expected in (
            ("quality_pair_count", len(pair_details)),
            ("quality_pairs", pair_details),
            ("predeclared_acceptance_pass", not reasons),
            ("acceptance_reasons", reasons),
        ):
            _require_soak_value(block, key, expected, label=block_label)

    soak_summaries: list[dict[str, Any]] = []
    recovery_summaries: list[dict[str, Any]] = []
    for cell_id, row in raw_cell_rows.items():
        plan_cell = plan_cells[cell_id]
        _soak_identity(
            row,
            campaign_id=campaign_id,
            plan_sha256=plan_sha256,
            cell=plan_cell,
            label="cell",
        )
        if row.get("execution_complete") is not True:
            raise DirectReportError("direct soak cell is not execution-complete")
        cell_blocks = [
            value
            for (key_cell, _), value in block_by_cell_and_index.items()
            if key_cell == cell_id
        ]
        cell_pairs = pairs_by_cell.get(cell_id, [])
        cell_phases = {
            phase: value
            for (key_cell, phase), value in phases_by_cell_and_name.items()
            if key_cell == cell_id
        }
        expected_pairs = _soak_count(plan_cell, "low_load_requests")
        drain_inclusive_successful_rpms: list[float] = []
        drain_inclusive_input_tpms: list[float] = []
        drain_inclusive_output_tpms: list[float] = []
        input_complete = False
        output_complete = False
        if row.get("scientifically_complete") is True:
            if (
                set(cell_phases) != set(SOAK_PHASES)
                or len(cell_blocks) != 4
                or len(cell_pairs) != expected_pairs
                or {_integer(pair.get("quality_pair_index")) for pair in cell_pairs}
                != set(range(expected_pairs))
            ):
                raise DirectReportError(
                    "scientifically complete soak cell lacks exact evidence"
                )
            ordered_blocks = sorted(
                cell_blocks, key=lambda block: int(block["analysis_block_index"])
            )
            ordered_pairs = sorted(
                cell_pairs, key=lambda pair: int(pair["quality_pair_index"])
            )
            successful_rpms = [
                float(block["successful_rpm_per_predeclared_window"])
                for block in ordered_blocks
            ]
            drain_inclusive_successful_rpms = [
                float(block["arrival_cohort_successful_rpm_including_drain"])
                for block in ordered_blocks
            ]
            input_complete = all(
                block.get("input_usage_complete_for_all_successes") is True
                for block in ordered_blocks
            )
            output_complete = all(
                block.get("output_usage_complete_for_all_successes") is True
                for block in ordered_blocks
            )
            input_tpms = [
                float(block["effective_input_tpm_per_predeclared_window"])
                for block in ordered_blocks
                if block.get("effective_input_tpm_per_predeclared_window") is not None
            ]
            output_tpms = [
                float(block["effective_output_tpm_per_predeclared_window"])
                for block in ordered_blocks
                if block.get("effective_output_tpm_per_predeclared_window") is not None
            ]
            drain_inclusive_input_tpms = [
                float(block["arrival_cohort_effective_input_tpm_including_drain"])
                for block in ordered_blocks
                if block.get("arrival_cohort_effective_input_tpm_including_drain")
                is not None
            ]
            drain_inclusive_output_tpms = [
                float(block["arrival_cohort_effective_output_tpm_including_drain"])
                for block in ordered_blocks
                if block.get("arrival_cohort_effective_output_tpm_including_drain")
                is not None
            ]
            quality_deltas = [
                float(pair["paired_quality_delta_near_minus_low"])
                for pair in ordered_pairs
                if pair.get("paired_quality_delta_near_minus_low") is not None
            ]
            cell_label = f"cell {cell_id}"
            for key, expected in (
                ("status", "complete"),
                ("candidate_rate_rps", plan_cell.get("candidate_rate_rps")),
                ("source_aimd_evidence", plan_cell.get("candidate_evidence")),
                ("analysis_block_count", len(ordered_blocks)),
                ("quality_pair_count", len(ordered_pairs)),
                (
                    "two_minute_observed_acceptance_pass",
                    all(
                        block.get("predeclared_acceptance_pass") is True
                        for block in ordered_blocks
                    ),
                ),
                ("successful_rpm_block_mean", statistics.fmean(successful_rpms)),
                (
                    "successful_rpm_block_mean_ci95_student_t",
                    _soak_t_mean_ci95(successful_rpms),
                ),
                ("input_usage_complete_for_all_blocks", input_complete),
                ("output_usage_complete_for_all_blocks", output_complete),
                (
                    "successful_rows_with_complete_input_usage",
                    sum(
                        int(block.get("successful_rows_with_complete_input_usage") or 0)
                        for block in ordered_blocks
                    ),
                ),
                (
                    "successful_rows_with_complete_output_usage",
                    sum(
                        int(
                            block.get("successful_rows_with_complete_output_usage") or 0
                        )
                        for block in ordered_blocks
                    ),
                ),
                (
                    "effective_input_tpm_block_mean",
                    statistics.fmean(input_tpms)
                    if input_complete and input_tpms
                    else (0.0 if input_complete else None),
                ),
                (
                    "effective_input_tpm_block_mean_ci95_student_t",
                    _soak_t_mean_ci95(input_tpms) if input_complete else None,
                ),
                (
                    "effective_output_tpm_block_mean",
                    statistics.fmean(output_tpms)
                    if output_complete and output_tpms
                    else (0.0 if output_complete else None),
                ),
                (
                    "effective_output_tpm_block_mean_ci95_student_t",
                    _soak_t_mean_ci95(output_tpms) if output_complete else None,
                ),
                (
                    "paired_quality_delta_mean",
                    statistics.fmean(quality_deltas) if quality_deltas else None,
                ),
                (
                    "paired_quality_delta_mean_ci95_student_t",
                    _soak_t_mean_ci95(quality_deltas),
                ),
                ("workload_contract", plan_cell.get("workload_contract")),
            ):
                _require_soak_value(row, key, expected, label=cell_label)

            baseline = cell_phases["paired_low_load"]
            recovery = cell_phases["post_soak_recovery"]
            recovery_reasons: list[str] = []
            if float(recovery.get("success_rate") or 0.0) < 0.99:
                recovery_reasons.append("recovery_success_rate_below_0.99")
            recovery_quality = float(recovery.get("quality_pass_rate") or 0.0)
            baseline_quality = float(baseline.get("quality_pass_rate") or 0.0)
            if recovery_quality < 0.999999:
                recovery_reasons.append(
                    "recovery_deterministic_quality_pass_rate_below_1.0"
                )
            if baseline_quality - recovery_quality > 0.05 + 1e-12:
                recovery_reasons.append(
                    "recovery_quality_drop_from_low_load_above_0.05"
                )
            recovery_total = int(recovery.get("completed_request_rows") or 0)
            if (
                recovery_total
                and (
                    int(recovery.get("http_5xx") or 0)
                    + int(recovery.get("timeouts") or 0)
                )
                / recovery_total
                > 0.01
            ):
                recovery_reasons.append("recovery_timeout_5xx_rate_above_0.01")
            if (
                recovery_total
                and int(recovery.get("http_429") or 0) / recovery_total > 0.01
            ):
                recovery_reasons.append("recovery_rate_limit_rate_above_0.01")
            recovery_ttft = _number(recovery.get("ttft_p95_seconds"))
            baseline_ttft = _number(baseline.get("ttft_p95_seconds"))
            if (
                recovery_ttft is not None
                and baseline_ttft is not None
                and recovery_ttft > 2 * baseline_ttft
            ):
                recovery_reasons.append("recovery_ttft_p95_above_2x_low_load")
            recovery_latency = _number(recovery.get("latency_p95_seconds"))
            baseline_latency = _number(baseline.get("latency_p95_seconds"))
            if (
                recovery_latency is not None
                and baseline_latency is not None
                and recovery_latency > 2 * baseline_latency
            ):
                recovery_reasons.append("recovery_latency_p95_above_2x_low_load")
            for key, expected in (
                ("post_soak_recovery_success_rate", recovery.get("success_rate")),
                (
                    "post_soak_recovery_quality_pass_rate",
                    recovery.get("quality_pass_rate"),
                ),
                (
                    "post_soak_recovery_ttft_p95_seconds",
                    recovery.get("ttft_p95_seconds"),
                ),
                (
                    "post_soak_recovery_target_rps",
                    recovery.get("offered_rps_target"),
                ),
                (
                    "post_soak_recovery_realized_schedule_rps",
                    recovery.get("offered_rps_realized_schedule"),
                ),
                (
                    "post_soak_recovery_predeclared_pass",
                    not recovery_reasons,
                ),
                (
                    "post_soak_recovery_acceptance_reasons",
                    recovery_reasons,
                ),
                (
                    "post_soak_recovery_quality_delta_from_low_load",
                    recovery_quality - baseline_quality,
                ),
            ):
                _require_soak_value(row, key, expected, label=cell_label)
        elif row.get("status") == "baseline_transport_gate_failed":
            if set(cell_phases) != {"paired_low_load"} or cell_blocks or cell_pairs:
                raise DirectReportError(
                    "transport-gated soak cell contains impossible later evidence"
                )
            baseline = cell_phases["paired_low_load"]
            for key, expected in (
                ("baseline_success_rate", baseline.get("success_rate")),
                ("provider_send_attempted", True),
            ):
                _require_soak_value(
                    row,
                    key,
                    expected,
                    label=f"baseline-gated cell {cell_id}",
                )
        else:
            raise DirectReportError(
                "execution-complete direct soak cell has an unsupported terminal state"
            )
        source_evidence = _mapping(plan_cell.get("candidate_evidence"))
        source_level = _text(source_evidence.get("source_evidence_level"))
        soak_summaries.append(
            {
                "schema_version": "digitalocean_public_soak_cell_summary_v1",
                "source_kind": "direct_soak",
                "source_id": source_id,
                "sampling_unit": "cell_id_with_four_analysis_block_ids",
                "block_ci_sampling_unit": "analysis_block_id",
                "paired_quality_ci_sampling_unit": "quality_pair_id",
                "cell_id": cell_id,
                "endpoint_id": row.get("model_id"),
                "shape": row.get("shape"),
                "status": row.get("status"),
                "execution_complete": True,
                "scientifically_complete": row.get("scientifically_complete") is True,
                "candidate_rate_rps": row.get("candidate_rate_rps"),
                "two_minute_soak_observed_rps": (
                    row.get("candidate_rate_rps")
                    if row.get("scientifically_complete") is True
                    else None
                ),
                "two_minute_soak_verified_rps": (
                    row.get("candidate_rate_rps")
                    if row.get("scientifically_complete") is True
                    and row.get("two_minute_observed_acceptance_pass") is True
                    else None
                ),
                "soak_acceptance_pass": row.get("two_minute_observed_acceptance_pass"),
                "soak_block_count": len(cell_blocks),
                "quality_pair_count": len(cell_pairs),
                "arrival_window_successful_rpm_block_mean_accounting": row.get(
                    "successful_rpm_block_mean"
                ),
                "arrival_window_successful_rpm_block_mean_ci95_student_t_accounting": row.get(
                    "successful_rpm_block_mean_ci95_student_t"
                ),
                "arrival_cohort_successful_rpm_including_drain_block_mean": (
                    statistics.fmean(drain_inclusive_successful_rpms)
                    if drain_inclusive_successful_rpms
                    else None
                ),
                "arrival_cohort_successful_rpm_including_drain_block_mean_ci95_student_t": (
                    _soak_t_mean_ci95(drain_inclusive_successful_rpms)
                    if drain_inclusive_successful_rpms
                    else None
                ),
                "arrival_cohort_effective_input_tpm_including_drain_block_mean": (
                    statistics.fmean(drain_inclusive_input_tpms)
                    if input_complete and drain_inclusive_input_tpms
                    else (0.0 if input_complete else None)
                ),
                "arrival_cohort_effective_input_tpm_including_drain_block_mean_ci95_student_t": (
                    _soak_t_mean_ci95(drain_inclusive_input_tpms)
                    if input_complete
                    else None
                ),
                "arrival_cohort_effective_output_tpm_including_drain_block_mean": (
                    statistics.fmean(drain_inclusive_output_tpms)
                    if output_complete and drain_inclusive_output_tpms
                    else (0.0 if output_complete else None)
                ),
                "arrival_cohort_effective_output_tpm_including_drain_block_mean_ci95_student_t": (
                    _soak_t_mean_ci95(drain_inclusive_output_tpms)
                    if output_complete
                    else None
                ),
                "headline_goodput_denominator": (
                    "arrival_cohort_elapsed_seconds_including_drain"
                    if row.get("scientifically_complete") is True
                    else None
                ),
                "successful_rpm_block_mean": row.get("successful_rpm_block_mean"),
                "successful_rpm_block_mean_ci95_student_t": row.get(
                    "successful_rpm_block_mean_ci95_student_t"
                ),
                "effective_input_tpm_block_mean": row.get(
                    "effective_input_tpm_block_mean"
                ),
                "effective_input_tpm_block_mean_ci95_student_t": row.get(
                    "effective_input_tpm_block_mean_ci95_student_t"
                ),
                "effective_output_tpm_block_mean": row.get(
                    "effective_output_tpm_block_mean"
                ),
                "effective_output_tpm_block_mean_ci95_student_t": row.get(
                    "effective_output_tpm_block_mean_ci95_student_t"
                ),
                "paired_quality_delta_mean": row.get("paired_quality_delta_mean"),
                "paired_quality_delta_mean_ci95_student_t": row.get(
                    "paired_quality_delta_mean_ci95_student_t"
                ),
                "source_aimd_evidence_level": source_level,
                "source_aimd_confirmation_epoch_count": len(
                    source_evidence.get("confirmation_epoch_ids", ())
                )
                if isinstance(source_evidence.get("confirmation_epoch_ids"), Sequence)
                and not isinstance(
                    source_evidence.get("confirmation_epoch_ids"),
                    (str, bytes, bytearray),
                )
                else 0,
                "capacity_claim": (
                    "exact_two_minute_soak_pass"
                    if row.get("scientifically_complete") is True
                    and row.get("two_minute_observed_acceptance_pass") is True
                    else (
                        "exact_two_minute_soak_measured_fail"
                        if row.get("scientifically_complete") is True
                        else "not_soak_verified"
                    )
                ),
                "capacity_generalization": row.get("capacity_generalization"),
                "claim_scope": row.get("claim_scope"),
                "block_ci_note": row.get("block_ci_note"),
            }
        )
        recovery = cell_phases.get("post_soak_recovery")
        if recovery is not None:
            recovery_summaries.append(
                {
                    **_soak_phase_summary(recovery, source_id=source_id),
                    "schema_version": "digitalocean_public_soak_recovery_summary_v1",
                    "predeclared_recovery_pass": row.get(
                        "post_soak_recovery_predeclared_pass"
                    ),
                    "recovery_acceptance_reasons": row.get(
                        "post_soak_recovery_acceptance_reasons"
                    ),
                    "quality_delta_from_low_load": row.get(
                        "post_soak_recovery_quality_delta_from_low_load"
                    ),
                    "claim_scope": row.get("claim_scope"),
                }
            )

    raw_summary_cells = summary.get("cells")
    if not isinstance(raw_summary_cells, Sequence) or isinstance(
        raw_summary_cells, (str, bytes, bytearray)
    ):
        raise DirectReportError("direct soak terminal summary cells are invalid")
    summary_cells: dict[str, Mapping[str, Any]] = {}
    for value in raw_summary_cells:
        if not isinstance(value, Mapping):
            raise DirectReportError("direct soak terminal summary cell is invalid")
        cell_id = _text(value.get("cell_id"))
        if cell_id is None or cell_id in summary_cells:
            raise DirectReportError("direct soak terminal summary has duplicate cells")
        plan_cell = plan_cells.get(str(cell_id))
        if (
            plan_cell is None
            or value.get("model_id") != plan_cell.get("model_id")
            or value.get("shape") != plan_cell.get("shape")
        ):
            raise DirectReportError(
                "direct soak terminal summary cell is outside the science plan"
            )
        summary_cells[cell_id] = value
    execution_complete = summary.get("execution_complete") is True
    expected_science_complete = bool(
        execution_complete
        and len(raw_cell_rows) == len(plan_cells)
        and all(
            value.get("scientifically_complete") is True
            for value in raw_cell_rows.values()
        )
    )
    expected_summary_status = (
        "complete"
        if expected_science_complete
        else (
            "execution_complete_science_incomplete"
            if execution_complete
            else "incomplete"
        )
    )
    if (
        summary.get("campaign_id") != campaign_id
        or summary.get("plan_sha256") != plan_sha256
        or (not allow_incomplete_terminal and not execution_complete)
        or summary.get("scientifically_complete") is not expected_science_complete
        or summary.get("status") != expected_summary_status
        or summary.get("http_402_latched") is not False
        or _integer(summary.get("target_cells")) != len(plan_cells)
        or _integer(summary.get("terminal_cells")) != len(raw_cell_rows)
        or _integer(summary.get("request_rows")) != len(raw_requests)
        or _integer(summary.get("analysis_block_rows")) != len(raw_blocks)
        or _integer(summary.get("quality_pair_rows")) != len(raw_pairs)
        or set(summary_cells)
        != (set(plan_cells) if allow_incomplete_terminal else set(raw_cell_rows))
        or any(
            canonical_json(dict(summary_cells[cell_id]))
            != canonical_json(dict(raw_cell_rows[cell_id]))
            for cell_id in raw_cell_rows
        )
    ):
        raise DirectReportError("direct soak terminal summary does not reconcile")

    return {
        "source_kind": "direct_soak",
        "source_id": source_id,
        "campaign_id": campaign_id,
        "plan_sha256": plan_sha256,
        "source_manifest_sha256": _sha256(manifest_path),
        "manifest": manifest,
        "plan": plan,
        "plan_cells": list(plan_cells.values()),
        "requests": normalized_requests,
        "epochs": normalized_epochs,
        "phase_summaries": phase_summaries,
        "soak_summaries": soak_summaries,
        "block_summaries": block_summaries,
        "quality_summaries": quality_summaries,
        "recovery_summaries": recovery_summaries,
        "cell_rows": list(raw_cell_rows.values()),
        "summary": summary,
    }


def _completion_attempt_request_id(semantic_id: str, attempt_index: int) -> str:
    digest = hashlib.sha256(
        canonical_json(
            {"semantic_id": semantic_id, "attempt_index": attempt_index}
        ).encode("utf-8")
    ).hexdigest()[:20]
    return f"do-completion-request-{digest}"


def _completion_string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise DirectReportError(f"direct completion {label} must be a list")
    output = [_text(item) for item in value]
    if any(item is None for item in output) or len(output) != len(set(output)):
        raise DirectReportError(
            f"direct completion {label} must contain unique non-empty strings"
        )
    return [str(item) for item in output]


def _completion_terminal_window(summary: Mapping[str, Any]) -> tuple[str, str]:
    started = _iso_time(summary.get("started_at"))
    ended = _iso_time(summary.get("ended_at"))
    if started is None or ended is None or str(ended) < str(started):
        raise DirectReportError("direct completion terminal window is invalid")
    return started, ended


def load_matched_closure_directory(path: Path) -> dict[str, Any]:
    """Load one terminal matched-control closure campaign.

    Physical control requests remain in the cost and reliability accounting, but
    only the final probe request inherits the semantic capability outcome.  This
    prevents a healthy control from being mistaken for evidence that the tested
    parameter or capability itself worked.
    """

    directory = Path(path)
    source_id = directory.name
    manifest_path = directory / "manifest.json"
    plan_path = directory / "plan.jsonl"
    attempts_path = directory / "attempts.jsonl"
    outcomes_path = directory / "records.jsonl"
    summary_path = directory / "summary.json"
    required = (
        manifest_path,
        plan_path,
        attempts_path,
        outcomes_path,
        summary_path,
    )
    if any(not candidate.is_file() for candidate in required):
        raise DirectReportError(
            "matched closure requires manifest.json, plan.jsonl, attempts.jsonl, "
            "records.jsonl, and terminal summary.json"
        )
    try:
        manifest = _read_json(manifest_path)
        summary = _read_json(summary_path)
        raw_plans = _read_jsonl(plan_path)
        raw_attempts = _read_jsonl(attempts_path)
        raw_outcomes = _read_jsonl(outcomes_path)
    except (OSError, ValueError) as error:
        raise DirectReportError("matched closure contains invalid JSON") from error
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != MATCHED_CLOSURE_MANIFEST_SCHEMA
        or not isinstance(summary, Mapping)
        or summary.get("schema_version") != MATCHED_CLOSURE_SUMMARY_SCHEMA
    ):
        raise DirectReportError("matched closure metadata schema is invalid")
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    campaign_id = _text(manifest.get("campaign_id"))
    if (
        campaign_id is None
        or manifest.get("plan_sha256") != plan_sha256
        or summary.get("campaign_id") != campaign_id
        or summary.get("plan_sha256") != plan_sha256
        or summary.get("status") not in COMPLETION_TERMINAL_STATUSES
    ):
        raise DirectReportError("matched closure identity or terminal status disagrees")

    plans_by_id: dict[str, dict[str, Any]] = {}
    normalized_plans: list[dict[str, Any]] = []
    for value in raw_plans:
        row = _mapping(value)
        cell_id = _text(row.get("cell_id"))
        endpoint = _require_endpoint(row.get("model_id") or row.get("endpoint_id"))
        probe_id = _text(row.get("probe_id"))
        workload = _text(row.get("workload_id"))
        task = _mapping(row.get("task"))
        task_id = _text(task.get("task_id"))
        requested_output = _integer(row.get("requested_max_output_tokens"))
        if (
            row.get("schema_version") != MATCHED_CLOSURE_PLAN_SCHEMA
            or cell_id is None
            or cell_id in plans_by_id
            or probe_id is None
            or workload is None
            or task_id is None
            or requested_output is None
            or requested_output <= 0
        ):
            raise DirectReportError("matched closure plan row is invalid")
        plans_by_id[cell_id] = dict(row)
        normalized_plans.append(
            {
                "source_kind": "direct_completion",
                "source_id": source_id,
                "cell_id": cell_id,
                "endpoint_id": endpoint,
                "probe_id": probe_id,
                "workload": workload,
                "shape": _text(row.get("shape")) or "matched_control_closure",
                "phase": _text(row.get("phase")),
                "task_id": task_id,
                "planned_attempt_count": 1,
                "physical_attempt_slot_count": _integer(manifest.get("max_attempts")),
                "request_payload_sha256": _text(row.get("rendered_payload_sha256")),
                "campaign_plan_sha256": plan_sha256,
                "requested_output_target": requested_output,
                "requested_output_unit": "tokens",
            }
        )
    if _integer(manifest.get("planned_cells")) != len(plans_by_id) or _integer(
        summary.get("planned_cells")
    ) != len(plans_by_id):
        raise DirectReportError("matched closure plan counts do not reconcile")

    outcomes: dict[str, dict[str, Any]] = {}
    for value in raw_outcomes:
        row = _mapping(value)
        cell_id = _text(row.get("cell_id"))
        plan = plans_by_id.get(str(cell_id))
        if (
            row.get("schema_version") != MATCHED_CLOSURE_OUTCOME_SCHEMA
            or cell_id is None
            or cell_id in outcomes
            or plan is None
            or row.get("campaign_id") != campaign_id
            or row.get("plan_sha256") != plan_sha256
            or row.get("model_id") != plan.get("model_id")
            or row.get("probe_id") != plan.get("probe_id")
        ):
            raise DirectReportError("matched closure outcome row is inconsistent")
        outcomes[cell_id] = dict(row)

    if _integer(summary.get("terminal_cells")) != len(outcomes) or _integer(
        summary.get("conclusive_cells")
    ) != sum(row.get("coverage_conclusive") is True for row in outcomes.values()):
        raise DirectReportError("matched closure outcome counts do not reconcile")
    if summary.get("status") == "complete" and len(outcomes) != len(plans_by_id):
        raise DirectReportError("matched closure claims complete with missing outcomes")

    for plan_row in normalized_plans:
        outcome = outcomes.get(str(plan_row["cell_id"]), {})
        plan_row["semantic_final_request_id"] = _text(
            outcome.get("semantic_final_request_id")
        )
        plan_row["terminal_outcome_status"] = _text(outcome.get("status"))
        plan_row["terminal_coverage_classification"] = _text(
            outcome.get("coverage_classification")
        )

    normalized_requests: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    for value in raw_attempts:
        row = dict(_mapping(value))
        request_id = _text(row.get("request_id"))
        cell_id = _text(row.get("cell_id"))
        plan = plans_by_id.get(str(cell_id))
        outcome = outcomes.get(str(cell_id), {})
        if (
            row.get("schema_version") != MATCHED_CLOSURE_REQUEST_SCHEMA
            or request_id is None
            or request_id in request_ids
            or plan is None
            or row.get("campaign_id") != campaign_id
            or row.get("plan_sha256") != plan_sha256
            or row.get("model_id") != plan.get("model_id")
            or row.get("probe_id") != plan.get("probe_id")
            or row.get("provider_send_attempted") is not True
        ):
            raise DirectReportError("matched closure request row is inconsistent")
        request_ids.add(request_id)
        final_request_id = _text(outcome.get("semantic_final_request_id"))
        is_semantic = request_id == final_request_id
        row.update(
            {
                "semantic_id": cell_id,
                "semantic_final_request_id": final_request_id,
                "semantic_coverage_attempt": is_semantic,
                "task_id": _text(_mapping(plan.get("task")).get("task_id")),
                "coverage_tags": plan.get("coverage_tags") or [],
                "rendered_payload_sha256": plan.get("rendered_payload_sha256"),
                "request_identity_sha256": plan.get("request_identity_sha256"),
            }
        )
        if is_semantic:
            for key in (
                "coverage_classification",
                "coverage_conclusive",
                "capability_status",
                "functional_valid",
                "quality_score",
                "score_kind",
                "finish_reason",
            ):
                row[key] = outcome.get(key)
        normalized = normalize_request(
            row,
            source_kind="direct_completion",
            source_id=source_id,
        )
        normalized_requests.append(normalized)

    if _integer(summary.get("provider_attempts")) != len(normalized_requests):
        raise DirectReportError("matched closure request count does not reconcile")
    return {
        "source_id": source_id,
        "campaign_id": campaign_id,
        "source_manifest_sha256": _sha256(manifest_path),
        "summary": dict(summary),
        "plans": normalized_plans,
        "requests": normalized_requests,
        "outcomes": list(outcomes.values()),
    }


def load_completion_directory(path: Path) -> dict[str, Any]:
    """Strictly load one terminal direct-completion evidence directory.

    All request journal rows are retained as physical attempts.  Exactly the
    request named by each probe outcome's ``final_request_id`` is eligible as
    that probe's semantic coverage attempt.  Nested terminal soak waves are
    loaded through the ordinary strict soak loader, but their cumulative cost
    receipts remain subordinate to the parent completion receipt.
    """

    directory = Path(path)
    source_id = directory.name
    plan_path = directory / "plan.json"
    manifest_path = directory / "manifest.json"
    summary_path = directory / "summary.json"
    requests_path = directory / "requests.jsonl"
    outcomes_path = directory / "probe-outcomes.jsonl"
    required = (plan_path, manifest_path, requests_path, outcomes_path, summary_path)
    if any(not candidate.is_file() for candidate in required):
        raise DirectReportError(
            "direct completion requires plan.json, manifest.json, requests.jsonl, "
            "probe-outcomes.jsonl, and terminal summary.json"
        )
    try:
        plan = _read_json(plan_path)
        manifest = _read_json(manifest_path)
        summary = _read_json(summary_path)
    except (OSError, ValueError) as error:
        raise DirectReportError(
            "direct completion has invalid JSON metadata"
        ) from error
    if (
        not isinstance(plan, Mapping)
        or plan.get("schema_version") != COMPLETION_PLAN_SCHEMA
    ):
        raise DirectReportError("direct completion plan schema is invalid")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != COMPLETION_MANIFEST_SCHEMA
    ):
        raise DirectReportError("direct completion manifest schema is invalid")
    if (
        not isinstance(summary, Mapping)
        or summary.get("schema_version") != COMPLETION_SUMMARY_SCHEMA
    ):
        raise DirectReportError("direct completion terminal summary schema is invalid")

    plan_sha256 = hashlib.sha256(canonical_json(dict(plan)).encode("utf-8")).hexdigest()
    campaign_id = f"do-completion-{plan_sha256[:20]}"
    if (
        manifest.get("campaign_id") != campaign_id
        or manifest.get("plan_sha256") != plan_sha256
        or summary.get("campaign_id") != campaign_id
        or summary.get("plan_sha256") != plan_sha256
    ):
        raise DirectReportError(
            "direct completion manifest, summary, and recomputed plan identity disagree"
        )

    models = _completion_string_list(plan.get("models"), label="plan models")
    endpoints = [_require_endpoint(model_id) for model_id in models]
    if endpoints != models:
        raise DirectReportError("direct completion model identity is not canonical")
    max_attempts = _integer(plan.get("max_attempts"))
    if max_attempts is None or max_attempts < 1:
        raise DirectReportError("direct completion max_attempts is invalid")
    raw_probes = plan.get("probes")
    if not isinstance(raw_probes, Sequence) or isinstance(
        raw_probes, (str, bytes, bytearray)
    ):
        raise DirectReportError("direct completion plan probes must be a list")
    probes: dict[str, dict[str, Any]] = {}
    attempt_to_semantic: dict[str, str] = {}
    normalized_plans: list[dict[str, Any]] = []
    for raw_probe in raw_probes:
        if not isinstance(raw_probe, Mapping):
            raise DirectReportError("direct completion plan probe must be an object")
        probe = dict(raw_probe)
        semantic_id = _text(probe.get("semantic_id"))
        lane = _text(probe.get("lane"))
        endpoint = _require_endpoint(probe.get("model_id"))
        source_request_id = _text(probe.get("source_request_id"))
        source_probe_id = _text(probe.get("source_probe_id"))
        task_family = _text(probe.get("task_family"))
        task_id = _text(probe.get("task_id"))
        payload_sha256 = _text(probe.get("request_payload_sha256"))
        requested_output = _integer(probe.get("requested_max_output_tokens"))
        if (
            semantic_id is None
            or semantic_id in probes
            or lane not in COMPLETION_LANES
            or endpoint not in endpoints
            or source_probe_id is None
            or task_family is None
            or task_id is None
            or requested_output in {None, 0}
            or not re.fullmatch(r"[0-9a-f]{64}", str(payload_sha256 or ""))
        ):
            raise DirectReportError("direct completion plan has an invalid probe")
        if (lane in COMPLETION_SUPERSESSION_LANES) is not (
            source_request_id is not None
        ):
            raise DirectReportError(
                "direct completion retry probes require exactly one source_request_id"
            )
        attempt_ids = _completion_string_list(
            probe.get("attempt_request_ids"),
            label=f"attempt IDs for {semantic_id}",
        )
        expected_attempt_ids = [
            _completion_attempt_request_id(semantic_id, index)
            for index in range(max_attempts)
        ]
        if attempt_ids != expected_attempt_ids:
            raise DirectReportError(
                "direct completion attempt IDs disagree with the frozen plan identity"
            )
        if any(request_id in attempt_to_semantic for request_id in attempt_ids):
            raise DirectReportError(
                "direct completion attempt IDs are not globally unique"
            )
        for request_id in attempt_ids:
            attempt_to_semantic[request_id] = semantic_id
        probes[semantic_id] = probe
        normalized_plans.append(
            {
                "source_kind": "direct_completion",
                "source_id": source_id,
                "cell_id": semantic_id,
                "endpoint_id": endpoint,
                "workload": task_family,
                "shape": lane,
                "task_id": task_id,
                "planned_attempt_count": 1,
                "physical_attempt_slot_count": len(attempt_ids),
                "source_request_id": source_request_id,
                "supersedes_request_id": (
                    source_request_id if lane in COMPLETION_SUPERSESSION_LANES else None
                ),
                "request_payload_sha256": payload_sha256,
                "campaign_plan_sha256": plan_sha256,
                "requested_output_target": requested_output,
                "requested_output_unit": "tokens",
            }
        )

    initial_unresolved = _completion_string_list(
        plan.get("initial_unresolved_soak_cells"),
        label="initial unresolved soak cells",
    )
    if (
        _integer(manifest.get("planned_semantic_probes")) != len(probes)
        or _integer(manifest.get("planned_attempt_slots")) != len(probes) * max_attempts
        or _integer(manifest.get("planned_descending_soak_cells"))
        != len(initial_unresolved)
        or manifest.get("launch_gate_passes") is not True
    ):
        raise DirectReportError(
            "direct completion manifest counts do not match the plan"
        )

    raw_requests_list = _read_jsonl(requests_path)
    raw_requests: dict[str, dict[str, Any]] = {}
    for row in raw_requests_list:
        request_id = _text(row.get("request_id"))
        semantic_id = _text(row.get("semantic_id"))
        probe = probes.get(str(semantic_id))
        attempt_index = _integer(row.get("attempt_index"))
        accounted_cost = _number(row.get("accounted_cost_usd"))
        estimated_cost = _number(row.get("estimated_cost_usd"))
        if (
            row.get("schema_version") != COMPLETION_REQUEST_SCHEMA
            or request_id is None
            or request_id in raw_requests
            or probe is None
            or attempt_index is None
            or attempt_index >= max_attempts
            or request_id
            != _completion_attempt_request_id(str(semantic_id), attempt_index)
            or attempt_to_semantic.get(request_id) != semantic_id
            or row.get("campaign_id") != campaign_id
            or row.get("plan_sha256") != plan_sha256
            or row.get("model_id") != probe.get("model_id")
            or row.get("shape") != probe.get("lane")
            or row.get("source_request_id") != probe.get("source_request_id")
            or row.get("source_probe_id") != probe.get("source_probe_id")
            or row.get("provider_send_attempted") is not True
            or row.get("status") not in {"success", "error"}
            or accounted_cost is None
            or accounted_cost < 0
            or (
                row.get("estimated_cost_usd") is not None
                and (estimated_cost is None or estimated_cost < 0)
            )
        ):
            raise DirectReportError("direct completion request journal is inconsistent")
        raw_requests[request_id] = row

    raw_outcomes_list = _read_jsonl(outcomes_path)
    outcomes: dict[str, dict[str, Any]] = {}
    for row in raw_outcomes_list:
        semantic_id = _text(row.get("semantic_id"))
        probe = probes.get(str(semantic_id))
        final_request_id = _text(row.get("final_request_id"))
        if (
            row.get("schema_version") != COMPLETION_OUTCOME_SCHEMA
            or semantic_id is None
            or semantic_id in outcomes
            or probe is None
            or row.get("campaign_id") != campaign_id
            or row.get("lane") != probe.get("lane")
            or row.get("model_id") != probe.get("model_id")
            or row.get("source_request_id") != probe.get("source_request_id")
            or row.get("source_probe_id") != probe.get("source_probe_id")
        ):
            raise DirectReportError("direct completion outcome journal is inconsistent")
        planned_attempt_ids = set(probe.get("attempt_request_ids") or ())
        if final_request_id is not None and final_request_id not in planned_attempt_ids:
            raise DirectReportError(
                "direct completion final_request_id is outside its semantic plan"
            )
        final_request = raw_requests.get(str(final_request_id))
        if final_request is None:
            if final_request_id is not None and row.get("coverage_conclusive") is True:
                raise DirectReportError(
                    "direct completion cannot claim conclusive coverage without a "
                    "physical terminal request"
                )
        elif (
            final_request.get("semantic_id") != semantic_id
            or row.get("status") != final_request.get("status")
            or row.get("coverage_conclusive")
            != final_request.get("coverage_conclusive")
            or row.get("functional_valid") != final_request.get("functional_valid")
        ):
            raise DirectReportError(
                "direct completion outcome disagrees with its final physical request"
            )
        outcomes[semantic_id] = row

    normalized_requests: list[dict[str, Any]] = []
    for request_id, row in raw_requests.items():
        semantic_id = str(row["semantic_id"])
        probe = probes[semantic_id]
        outcome = outcomes.get(semantic_id, {})
        final_request_id = _text(outcome.get("final_request_id"))
        source_request_id = _text(probe.get("source_request_id"))
        public_row = dict(row)
        public_row.update(
            {
                "semantic_final_request_id": final_request_id,
                "semantic_coverage_attempt": request_id == final_request_id,
                "supersedes_request_id": (
                    source_request_id
                    if probe.get("lane") in COMPLETION_SUPERSESSION_LANES
                    else None
                ),
            }
        )
        normalized = normalize_request(
            public_row,
            source_kind="direct_completion",
            source_id=source_id,
        )
        normalized["cell_id"] = semantic_id
        normalized["physical_retry_attempt"] = int(row["attempt_index"]) > 0
        normalized["retryable"] = row.get("retryable") is True
        normalized_requests.append(normalized)

    for plan_row in normalized_plans:
        outcome = outcomes.get(str(plan_row["cell_id"]), {})
        plan_row["semantic_final_request_id"] = _text(outcome.get("final_request_id"))
        plan_row["terminal_outcome_status"] = _text(outcome.get("status"))

    remaining_unresolved = _completion_string_list(
        summary.get("remaining_unresolved_soak_cells"),
        label="remaining unresolved soak cells",
    )
    expected_status = (
        "complete"
        if len(outcomes) == len(probes) and not remaining_unresolved
        else "incomplete_or_censored"
    )
    _completion_terminal_window(summary)
    conclusive_outcomes = sum(
        row.get("coverage_conclusive") is True for row in outcomes.values()
    )
    prior_cost = _number(summary.get("prior_cost_usd"))
    cumulative_cost = _number(summary.get("conservative_exposure_usd"))
    max_cost = _number(summary.get("max_cost_usd"))
    launch_stop_cost = _number(summary.get("launch_stop_cost_usd"))
    drain_reserve = _number(summary.get("drain_reserve_usd"))
    plan_prior_cost = _number(plan.get("prior_cost_usd"))
    plan_max_cost = _number(plan.get("max_cost_usd"))
    plan_launch_stop_cost = _number(plan.get("launch_stop_cost_usd"))
    plan_drain_reserve = _number(plan.get("drain_reserve_usd"))
    manifest_max_cost = _number(manifest.get("hard_cap_usd"))
    manifest_launch_stop_cost = _number(manifest.get("launch_stop_cost_usd"))
    manifest_drain_reserve = _number(manifest.get("drain_reserve_usd"))
    if (
        summary.get("status") not in COMPLETION_TERMINAL_STATUSES
        or summary.get("status") != expected_status
        or _integer(summary.get("planned_semantic_probes")) != len(probes)
        or _integer(summary.get("terminal_probe_outcomes")) != len(outcomes)
        or _integer(summary.get("conclusive_probe_outcomes")) != conclusive_outcomes
        or _integer(summary.get("request_rows")) != len(raw_requests)
        or _integer(summary.get("outlier_audit_rows")) != len(raw_requests)
        or prior_cost is None
        or cumulative_cost is None
        or max_cost is None
        or launch_stop_cost is None
        or drain_reserve is None
        or plan_prior_cost is None
        or plan_max_cost is None
        or plan_launch_stop_cost is None
        or plan_drain_reserve is None
        or manifest_max_cost is None
        or manifest_launch_stop_cost is None
        or manifest_drain_reserve is None
        or not 0 <= prior_cost <= cumulative_cost <= launch_stop_cost <= max_cost
        or not math.isclose(
            drain_reserve, max_cost - launch_stop_cost, rel_tol=0, abs_tol=1e-9
        )
        or not math.isclose(plan_prior_cost, prior_cost, rel_tol=0, abs_tol=1e-9)
        or not math.isclose(plan_max_cost, max_cost, rel_tol=0, abs_tol=1e-9)
        or not math.isclose(
            plan_launch_stop_cost,
            launch_stop_cost,
            rel_tol=0,
            abs_tol=1e-9,
        )
        or not math.isclose(plan_drain_reserve, drain_reserve, rel_tol=0, abs_tol=1e-9)
        or not math.isclose(manifest_max_cost, max_cost, rel_tol=0, abs_tol=1e-9)
        or not math.isclose(
            manifest_launch_stop_cost,
            launch_stop_cost,
            rel_tol=0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            manifest_drain_reserve, drain_reserve, rel_tol=0, abs_tol=1e-9
        )
    ):
        raise DirectReportError("direct completion terminal summary does not reconcile")

    wave_rows = _read_jsonl(directory / "soak-waves.jsonl")
    censor_rows = _read_jsonl(directory / "soak-censors.jsonl")
    censors: dict[str, dict[str, Any]] = {}
    censored_endpoint_shapes: set[str] = set()
    for row in censor_rows:
        censor_id = _text(row.get("censor_id"))
        endpoint_shape = _text(row.get("endpoint_shape"))
        if (
            row.get("schema_version") != COMPLETION_SOAK_CENSOR_SCHEMA
            or row.get("campaign_id") != campaign_id
            or censor_id is None
            or censor_id in censors
            or endpoint_shape is None
            or endpoint_shape not in set(initial_unresolved)
            or row.get("status") != "censored_ineligible_aimd_prerequisite"
            or not _text(row.get("blocked_status"))
            or not _text(row.get("blocked_reason"))
        ):
            raise DirectReportError(
                "direct completion soak-censor journal is inconsistent"
            )
        censors[censor_id] = row
        censored_endpoint_shapes.add(endpoint_shape)
    waves: dict[str, dict[str, Any]] = {}
    wave_indices: set[int] = set()
    rate_ladder = plan.get("rate_ladder")
    if not isinstance(rate_ladder, Sequence) or isinstance(
        rate_ladder, (str, bytes, bytearray)
    ):
        raise DirectReportError("direct completion rate ladder must be a list")
    nested_soaks: list[dict[str, Any]] = []
    nested_directories: list[Path] = []
    last_unresolved = initial_unresolved
    last_exposure = prior_cost
    for row in wave_rows:
        wave_id = _text(row.get("wave_id"))
        wave_index = _integer(row.get("wave_index"))
        multiplier = _number(row.get("candidate_rate_multiplier"))
        if (
            row.get("schema_version") != COMPLETION_SOAK_WAVE_SCHEMA
            or row.get("campaign_id") != campaign_id
            or wave_id is None
            or wave_id in waves
            or wave_index is None
            or wave_index in wave_indices
            or wave_index >= len(rate_ladder)
            or multiplier is None
            or not math.isclose(
                multiplier,
                float(rate_ladder[wave_index]),
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise DirectReportError(
                "direct completion soak-wave journal is inconsistent"
            )
        expected_wave_digest = hashlib.sha256(
            canonical_json(
                {
                    "campaign_id": campaign_id,
                    "wave_index": wave_index,
                    "multiplier": multiplier,
                    "cells": sorted(last_unresolved),
                }
            ).encode("utf-8")
        ).hexdigest()[:20]
        if wave_id != f"do-completion-soak-wave-{expected_wave_digest}":
            raise DirectReportError(
                "direct completion soak-wave identity is inconsistent"
            )
        attempted = _completion_string_list(
            row.get("attempted_cells"), label=f"wave {wave_index} attempted cells"
        )
        censored = _completion_string_list(
            row.get("censored_cells") or (),
            label=f"wave {wave_index} censored cells",
        )
        passed = _completion_string_list(
            row.get("passed_cells"), label=f"wave {wave_index} passed cells"
        )
        unresolved = _completion_string_list(
            row.get("unresolved_after"), label=f"wave {wave_index} unresolved cells"
        )
        if (
            not set(passed).issubset(attempted)
            or set(attempted).intersection(censored)
            or set(attempted).union(censored) != set(last_unresolved)
            or not set(censored).issubset(censored_endpoint_shapes)
            or set(unresolved) != set(last_unresolved) - set(passed)
        ):
            raise DirectReportError(
                "direct completion soak-wave lineage is inconsistent"
            )
        artifact_name = _text(row.get("soak_artifact_relative_path"))
        if artifact_name is None:
            artifact_name = f"wave-{wave_index}-{multiplier:g}"
        if Path(artifact_name).name != artifact_name or not re.fullmatch(
            r"wave-[0-9]+-[0-9.]+(?:-eligible-[0-9a-f]{12})?", artifact_name
        ):
            raise DirectReportError(
                "direct completion soak-wave artifact path is unsafe"
            )
        wave_directory = directory / "soak-waves" / artifact_name
        nested = load_soak_directory(
            wave_directory,
            source_id_override=f"{source_id}-soak-wave-{wave_index}",
            allow_incomplete_terminal=True,
        )
        nested_cells = {
            f"{cell.get('model_id')}:{cell.get('shape')}"
            for cell in nested.get("plan_cells", ())
        }
        nested_passed = {
            f"{cell.get('model_id')}:{cell.get('shape')}"
            for cell in nested.get("cell_rows", ())
            if cell.get("scientifically_complete") is True
            and cell.get("two_minute_observed_acceptance_pass") is True
            and cell.get("post_soak_recovery_predeclared_pass") is True
        }
        nested_exposure = _number(
            _mapping(nested.get("summary")).get("conservative_exposure_usd")
        )
        row_exposure = _number(row.get("conservative_exposure_usd"))
        if (
            nested.get("campaign_id") != row.get("soak_campaign_id")
            or nested.get("plan_sha256") != row.get("soak_plan_sha256")
            or nested_cells != set(attempted)
            or nested_passed != set(passed)
            or nested_exposure is None
            or row_exposure is None
            or not math.isclose(nested_exposure, row_exposure, rel_tol=0, abs_tol=1e-9)
        ):
            raise DirectReportError(
                "direct completion soak-wave receipt does not match nested evidence"
            )
        nested["parent_completion_source_id"] = source_id
        nested["parent_completion_campaign_id"] = campaign_id
        nested["completion_wave_id"] = wave_id
        nested["completion_wave_index"] = wave_index
        nested["artifact_directory"] = wave_directory
        nested_soaks.append(nested)
        nested_directories.append(wave_directory.resolve())
        waves[wave_id] = row
        wave_indices.add(wave_index)
        last_unresolved = unresolved
        last_exposure = row_exposure

    if wave_indices and wave_indices != set(range(len(wave_indices))):
        raise DirectReportError(
            "direct completion soak-wave indices must be contiguous"
        )
    summary_initial_unresolved = _completion_string_list(
        summary.get("initial_unresolved_soak_cells"),
        label="summary initial unresolved soak cells",
    )
    summary_censored = _completion_string_list(
        summary.get("censored_soak_cells") or (),
        label="summary censored soak cells",
    )
    if (
        _integer(summary.get("soak_waves")) != len(waves)
        or set(summary_initial_unresolved) != set(initial_unresolved)
        or set(remaining_unresolved) != set(last_unresolved)
        or set(summary_censored) != censored_endpoint_shapes
        or (
            bool(waves)
            and not math.isclose(
                cumulative_cost, float(last_exposure), rel_tol=0, abs_tol=1e-9
            )
        )
    ):
        raise DirectReportError(
            "direct completion soak-wave summary does not reconcile"
        )

    source_contract = plan.get("source_contract")
    required_source_hashes = {
        "soak_summary_sha256",
        "context_summary_sha256",
        "context_plan_sha256",
        "capability_summary_sha256",
        "capability_plan_sha256",
    }
    source_exposures = _mapping(_mapping(source_contract).get("source_exposures_usd"))
    if (
        not isinstance(source_contract, Mapping)
        or not required_source_hashes.issubset(source_contract)
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(source_contract.get(key) or ""))
            for key in required_source_hashes
        )
        or set(source_exposures) != {"soak", "context", "capability"}
        or any(
            _number(source_exposures.get(key)) is None
            or float(source_exposures[key]) < 0
            or float(source_exposures[key]) > prior_cost
            for key in source_exposures
        )
    ):
        raise DirectReportError("direct completion source contract is invalid")

    return {
        "source_kind": "direct_completion",
        "source_id": source_id,
        "campaign_id": campaign_id,
        "plan_sha256": plan_sha256,
        "source_manifest_sha256": _sha256(manifest_path),
        "manifest": manifest,
        "plan": plan,
        "plans": normalized_plans,
        "requests": normalized_requests,
        "outcomes": list(outcomes.values()),
        "soak_censors": list(censors.values()),
        "summary": summary,
        "nested_soaks": nested_soaks,
        "nested_soak_directories": nested_directories,
    }


def _duration_seconds(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Return summed source-active wall clock, excluding inter-campaign idle gaps.

    A single min-to-max interval across multiple campaigns can include hours of
    dead time and make goodput meaningless. Within each source (or explicit
    epoch), overlapping requests share one active interval; independent source
    intervals are then summed.
    """

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("source_id") or "unknown-source"),
                str(row.get("epoch_id") or "source-active"),
            )
        ].append(row)
    total = 0.0
    any_duration = False
    for members in groups.values():
        starts = [_timestamp(row.get("started_at")) for row in members]
        ends = [_timestamp(row.get("ended_at")) for row in members]
        valid_starts = [value for value in starts if value is not None]
        valid_ends = [value for value in ends if value is not None]
        if valid_starts and valid_ends and max(valid_ends) > min(valid_starts):
            total += max(valid_ends) - min(valid_starts)
            any_duration = True
            continue
        services = [
            value
            for row in members
            if (value := _number(row.get("request_seconds"))) is not None and value > 0
        ]
        if services:
            total += sum(services)
            any_duration = True
    return total if any_duration and total > 0 else None


def _bootstrap_metric(
    values: Sequence[float], *, seed: int, replicates: int, statistic: str = "median"
) -> dict[str, Any]:
    fn = mean if statistic == "mean" else lambda sample: nearest_rank(sample, 0.50)
    return bootstrap_interval(values, fn, seed=seed, replicates=replicates)


def _epoch_units_from_requests(
    rows: Sequence[Mapping[str, Any]], epoch_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    def epoch_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
        return (
            str(row.get("source_kind")),
            str(row.get("source_id")),
            str(row.get("epoch_id")),
        )

    existing = {epoch_key(row): dict(row) for row in epoch_rows}
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        epoch_id = _text(row.get("epoch_id"))
        if epoch_id:
            grouped[epoch_key(row)].append(row)
    for key, members in grouped.items():
        if key in existing:
            continue
        epoch_id = key[2]
        duration = _duration_seconds(members)
        total = len(members)
        successes = sum(bool(row.get("scientific_success")) for row in members)
        quality_pass = sum(bool(row.get("goodput_success")) for row in members)
        successful_input = sum(
            int(row.get("input_tokens") or 0)
            for row in members
            if row.get("scientific_success")
        )
        successful_output = sum(
            int(row.get("output_tokens") or 0)
            for row in members
            if row.get("scientific_success")
        )
        first = members[0]
        existing[key] = {
            "schema_version": NORMALIZED_EPOCH_SCHEMA,
            "source_kind": first.get("source_kind"),
            "source_id": first.get("source_id"),
            "run_id": first.get("run_id"),
            "epoch_id": epoch_id,
            "endpoint_id": first.get("endpoint_id"),
            "workload": first.get("workload"),
            "shape": first.get("shape"),
            "phase": first.get("phase"),
            "sequence": None,
            "started_at": min(
                (str(row["started_at"]) for row in members if row.get("started_at")),
                default=None,
            ),
            "ended_at": max(
                (str(row["ended_at"]) for row in members if row.get("ended_at")),
                default=None,
            ),
            "elapsed_seconds": duration,
            "offered_window_seconds": duration,
            "offered_rps": _number(first.get("offered_rps")),
            "offered_rpm": (
                _number(first.get("offered_rps")) * 60
                if _number(first.get("offered_rps")) is not None
                else None
            ),
            "concurrency_ceiling": _integer(first.get("concurrency_ceiling")),
            "peak_concurrency": None,
            "scheduled_count": total,
            "completed_count": total,
            "success_count": successes,
            "quality_pass_count": quality_pass,
            "rate_limit_count": sum(bool(row.get("rate_limited")) for row in members),
            "timeout_count": sum(bool(row.get("timeout")) for row in members),
            "server_error_count": sum(bool(row.get("server_error")) for row in members),
            "other_error_count": sum(
                not row.get("transport_success") for row in members
            ),
            "success_rate": successes / total if total else None,
            "achieved_rpm": successes * 60 / duration if duration else None,
            "effective_input_tpm": successful_input * 60 / duration
            if duration
            else None,
            "effective_output_tpm": successful_output * 60 / duration
            if duration
            else None,
            "aggregate_output_goodput_tokens_per_second": successful_output / duration
            if duration
            else None,
            "goodput_rpm": quality_pass * 60 / duration if duration else None,
            "quality_adjusted_output_tpm": None,
            "ttft_p50_seconds": nearest_rank(
                [row.get("ttft_seconds") for row in members], 0.50
            ),
            "ttft_p95_seconds": nearest_rank(
                [row.get("ttft_seconds") for row in members], 0.95
            ),
            "latency_p50_seconds": nearest_rank(
                [row.get("request_seconds") for row in members], 0.50
            ),
            "latency_p95_seconds": nearest_rank(
                [row.get("request_seconds") for row in members], 0.95
            ),
            "output_tokens_per_second": successful_output / duration
            if duration
            else None,
            "output_tokens_per_second_metric_kind": (
                "aggregate_successful_completion_tokens_over_epoch_wall_clock"
                if duration
                else None
            ),
            "healthy": None,
            "health_reasons": None,
            "valid_for_capacity": False,
            "estimated_cost_usd": sum(
                float(row.get("estimated_cost_usd") or 0) for row in members
            ),
        }
    return list(existing.values())


def _rate_interval(
    rows: Sequence[Mapping[str, Any]],
    epochs: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> tuple[dict[str, Any], str]:
    if epochs:
        values = [
            float(value)
            for row in epochs
            if (value := _number(row.get("success_rate"))) is not None
        ]
        return (
            _bootstrap_metric(
                values, seed=seed, replicates=replicates, statistic="mean"
            ),
            "epoch_percentile_bootstrap",
        )
    successes = sum(bool(row.get("transport_success")) for row in rows)
    return wilson_interval(successes, len(rows)), "request_wilson"


def _metric_interval(
    rows: Sequence[Mapping[str, Any]],
    epochs: Sequence[Mapping[str, Any]],
    *,
    request_key: str,
    epoch_key: str,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if epochs:
        values = [
            float(value)
            for row in epochs
            if (value := _number(row.get(epoch_key))) is not None
        ]
        result = _bootstrap_metric(values, seed=seed, replicates=replicates)
        result["sampling_unit"] = "epoch_id"
        return result
    values = [
        float(value)
        for row in rows
        if (value := _number(row.get(request_key))) is not None
    ]
    result = _bootstrap_metric(values, seed=seed, replicates=replicates)
    result["sampling_unit"] = "request_id"
    return result


def summarize_group(
    rows: Sequence[Mapping[str, Any]],
    epochs: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    total = len(rows)
    transport = sum(bool(row.get("transport_success")) for row in rows)
    scientific = sum(bool(row.get("scientific_success")) for row in rows)
    good = sum(bool(row.get("goodput_success")) for row in rows)
    quality_rows = [row for row in rows if row.get("quality_scored")]
    quality_passes = sum(bool(row.get("functional_valid")) for row in quality_rows)
    elapsed = _duration_seconds(rows)
    if epochs:
        epoch_elapsed = sum(
            float(value)
            for row in epochs
            if (value := _number(row.get("elapsed_seconds"))) is not None
        )
        if epoch_elapsed > 0:
            elapsed = epoch_elapsed
    minutes = elapsed / 60 if elapsed and elapsed > 0 else None
    successful_rows = [row for row in rows if row.get("scientific_success")]
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in successful_rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in successful_rows)
    quality_weighted_tokens = sum(
        int(row.get("output_tokens") or 0) * float(row.get("quality_score") or 0)
        for row in successful_rows
        if row.get("quality_scored")
    )
    ttft_values = [
        float(value)
        for row in rows
        if (value := _number(row.get("ttft_seconds"))) is not None
    ]
    latency_values = [
        float(value)
        for row in rows
        if (value := _number(row.get("request_seconds"))) is not None
    ]
    tps_values = [
        float(value)
        for row in rows
        if (value := _number(row.get("post_ttft_output_tokens_per_second_proxy")))
        is not None
    ]
    cache_state_counts = Counter(
        str(row.get("cache_state") or "not_reported_unknown") for row in rows
    )
    success_interval, success_method = _rate_interval(
        rows,
        epochs,
        seed=deterministic_seed(seed, "success"),
        replicates=bootstrap_replicates,
    )
    success_interval = dict(success_interval)
    success_interval["sampling_unit"] = "epoch_id" if epochs else "request_id"
    success_interval["interval_method"] = success_method
    success_interval.setdefault("n_units", int(success_interval.get("n") or 0))
    success_interval["group_unit_count"] = len(epochs) if epochs else total
    quality_interval = wilson_interval(quality_passes, len(quality_rows))
    quality_interval["sampling_unit"] = "request_id"
    quality_interval["interval_method"] = "request_wilson"
    quality_interval.setdefault("n_units", int(quality_interval.get("n") or 0))
    quality_interval["group_unit_count"] = len(quality_rows)
    ttft_interval = _metric_interval(
        rows,
        epochs,
        request_key="ttft_seconds",
        epoch_key="ttft_p50_seconds",
        seed=deterministic_seed(seed, "ttft"),
        replicates=bootstrap_replicates,
    )
    latency_interval = _metric_interval(
        rows,
        epochs,
        request_key="request_seconds",
        epoch_key="latency_p50_seconds",
        seed=deterministic_seed(seed, "latency"),
        replicates=bootstrap_replicates,
    )
    post_ttft_output_tps_interval = _metric_interval(
        rows,
        (),
        request_key="post_ttft_output_tokens_per_second_proxy",
        epoch_key="unused",
        seed=deterministic_seed(seed, "post-ttft-output-tps"),
        replicates=bootstrap_replicates,
    )
    aggregate_output_goodput_interval = _metric_interval(
        (),
        epochs,
        request_key="unused",
        epoch_key="aggregate_output_goodput_tokens_per_second",
        seed=deterministic_seed(seed, "aggregate-output-goodput-tps"),
        replicates=bootstrap_replicates,
    )
    # This estimand exists only at the epoch level.  Preserve that unit even
    # for an empty interval so downstream code cannot mistake a missing epoch
    # metric for a request-level observation.
    aggregate_output_goodput_interval["sampling_unit"] = "epoch_id"

    def epoch_median(key: str) -> float | None:
        return nearest_rank(
            [row.get(key) for row in epochs],
            0.50,
        )

    load_group = bool(epochs)
    transport_success_rate = (
        _number(success_interval.get("estimate"))
        if load_group
        else (transport / total if total else None)
    )
    ttft_p50 = (
        _number(ttft_interval.get("estimate"))
        if load_group
        else nearest_rank(ttft_values, 0.50)
    )
    latency_p50 = (
        _number(latency_interval.get("estimate"))
        if load_group
        else nearest_rank(latency_values, 0.50)
    )
    post_ttft_output_tps_p50 = nearest_rank(tps_values, 0.50)
    epoch_goodput_values = [
        float(value)
        for row in epochs
        if (value := _number(row.get("aggregate_output_goodput_tokens_per_second")))
        is not None
    ]
    epoch_goodput_pairs = [
        (float(value), float(epoch_elapsed))
        for row in epochs
        if (value := _number(row.get("aggregate_output_goodput_tokens_per_second")))
        is not None
        and (epoch_elapsed := _number(row.get("elapsed_seconds"))) is not None
        and epoch_elapsed > 0
    ]
    aggregate_output_goodput = (
        sum(value * epoch_elapsed for value, epoch_elapsed in epoch_goodput_pairs)
        / sum(epoch_elapsed for _, epoch_elapsed in epoch_goodput_pairs)
        if epoch_goodput_pairs
        else (output_tokens / elapsed if elapsed else None)
    )
    tail_unit_count = len(epochs) if load_group else len(latency_values)
    return {
        "request_count": total,
        "epoch_count": len(epochs),
        "sampling_unit": "epoch_id" if epochs else "request_id",
        "transport_success_count": transport,
        "scientific_success_count": scientific,
        "goodput_count": good,
        "error_count": total - transport,
        "rate_limit_count": sum(bool(row.get("rate_limited")) for row in rows),
        "timeout_count": sum(bool(row.get("timeout")) for row in rows),
        "server_error_count": sum(bool(row.get("server_error")) for row in rows),
        "http_status_distribution": json.dumps(
            dict(
                sorted(
                    Counter(
                        str(row.get("http_status") or "none") for row in rows
                    ).items()
                )
            ),
            sort_keys=True,
        ),
        "transport_success_rate": transport_success_rate,
        "transport_success_rate_ci95_low": success_interval.get("ci95_low"),
        "transport_success_rate_ci95_high": success_interval.get("ci95_high"),
        "transport_success_rate_ci95": success_interval,
        "transport_success_interval_method": success_method,
        "quality_scored_count": len(quality_rows),
        "quality_pass_count": quality_passes,
        "quality_pass_rate": quality_passes / len(quality_rows)
        if quality_rows
        else None,
        "quality_pass_rate_ci95_low": quality_interval.get("ci95_low"),
        "quality_pass_rate_ci95_high": quality_interval.get("ci95_high"),
        "quality_pass_rate_ci95": quality_interval,
        "elapsed_seconds": elapsed,
        "elapsed_basis": (
            "sum_of_epoch_elapsed_seconds"
            if epochs
            else "sum_of_source_or_epoch_active_wall_clock_intervals"
        ),
        "offered_rpm_max": max(
            (
                float(row["offered_rpm"])
                for row in epochs
                if _number(row.get("offered_rpm")) is not None
            ),
            default=None,
        ),
        "achieved_rpm": scientific / minutes if minutes else None,
        "goodput_rpm": good / minutes if minutes else None,
        "effective_input_tpm": input_tokens / minutes if minutes else None,
        "effective_output_tpm": output_tokens / minutes if minutes else None,
        "aggregate_output_goodput_tokens_per_second": aggregate_output_goodput,
        "aggregate_output_goodput_tps_epoch_p50": _number(
            aggregate_output_goodput_interval.get("estimate")
        ),
        "aggregate_output_goodput_tps_epoch_p50_ci95": (
            aggregate_output_goodput_interval
        ),
        "aggregate_output_goodput_epoch_observation_count": len(epoch_goodput_values),
        "quality_adjusted_output_tpm_scored_only": (
            quality_weighted_tokens / minutes if minutes else None
        ),
        "quality_scored_output_fraction": (
            sum(int(row.get("output_tokens") or 0) for row in quality_rows)
            / output_tokens
            if output_tokens
            else None
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft_p50_seconds": ttft_p50,
        "ttft_p90_seconds": (
            epoch_median("ttft_p90_seconds")
            if load_group
            else nearest_rank(ttft_values, 0.90)
        ),
        "ttft_p95_seconds": (
            epoch_median("ttft_p95_seconds")
            if load_group
            else nearest_rank(ttft_values, 0.95)
        ),
        "ttft_p99_seconds": nearest_rank(ttft_values, 0.99)
        if not load_group and len(ttft_values) >= 1_000
        else None,
        "latency_p50_seconds": latency_p50,
        "latency_p90_seconds": (
            epoch_median("latency_p90_seconds")
            if load_group
            else nearest_rank(latency_values, 0.90)
        ),
        "latency_p95_seconds": (
            epoch_median("latency_p95_seconds")
            if load_group
            else nearest_rank(latency_values, 0.95)
        ),
        "latency_p99_seconds": nearest_rank(latency_values, 0.99)
        if not load_group and len(latency_values) >= 1_000
        else None,
        "post_ttft_output_tps_proxy_p50": post_ttft_output_tps_p50,
        "post_ttft_output_tps_proxy_p05": nearest_rank(tps_values, 0.05),
        "post_ttft_output_tps_proxy_observation_count": len(tps_values),
        "ttft_observation_count": len(ttft_values),
        "buffered_ttft_censored_count": sum(
            row.get("stream_mode") == "buffered_nonstream" for row in rows
        ),
        "multi_choice_per_sequence_curve_excluded_count": sum(
            bool(row.get("multi_choice")) for row in rows
        ),
        "cache_hit_observed_count": cache_state_counts["cache_hit_observed"],
        "cache_miss_observed_count": cache_state_counts["cache_miss_observed"],
        "cache_state_unknown_count": cache_state_counts["not_reported_unknown"],
        "ttft_p50_ci95": ttft_interval,
        "latency_p50_ci95": latency_interval,
        "post_ttft_output_tps_proxy_p50_ci95": post_ttft_output_tps_interval,
        "estimated_cost_usd": sum(
            float(row.get("estimated_cost_usd") or 0) for row in rows
        ),
        "cost_per_successful_request_usd": (
            sum(float(row.get("estimated_cost_usd") or 0) for row in rows) / scientific
            if scientific
            else None
        ),
        "cost_per_million_effective_tokens_usd": (
            sum(float(row.get("estimated_cost_usd") or 0) for row in rows)
            * 1_000_000
            / (input_tokens + output_tokens)
            if input_tokens + output_tokens
            else None
        ),
        "p95_is_sparse": tail_unit_count < 60,
        "p99_qualified": not load_group and len(latency_values) >= 1_000,
        "p99_suppression_reason": (
            None
            if not load_group and len(latency_values) >= 1_000
            else (
                "load_tail_requires_independent_epoch_units_and_is_not_reported_as_request_p99"
                if load_group
                else "fewer_than_1000_independent_requests"
            )
        ),
    }


def _group_rows(
    requests: Sequence[Mapping[str, Any]],
    epochs: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_replicates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_key = tuple[
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
    ]
    epoch_groups: dict[group_key, list[Mapping[str, Any]]] = defaultdict(list)
    for row in epochs:
        if row.get("source_kind") == "direct_soak":
            continue
        epoch_groups[
            (
                str(row.get("source_kind")),
                str(row.get("source_id")),
                str(row.get("endpoint_id")),
                str(row.get("workload") or row.get("shape") or "unspecified"),
                str(row.get("shape") or "none"),
                str(row.get("phase") or "unreported"),
                str(row.get("offered_rps") or "unreported"),
                str(row.get("concurrency_ceiling") or "unreported"),
                "unreported",
                "unreported",
                "unreported",
                "unreported",
            )
        ].append(row)
    request_groups: dict[group_key, list[Mapping[str, Any]]] = defaultdict(list)
    for row in requests:
        if row.get("source_kind") == "direct_soak":
            continue
        request_groups[
            (
                str(row.get("source_kind")),
                str(row.get("source_id")),
                str(row.get("endpoint_id")),
                str(row.get("workload") or "unspecified"),
                str(row.get("shape") or "none"),
                str(row.get("phase") or "unreported"),
                str(row.get("offered_rps") or "unreported"),
                str(row.get("concurrency_ceiling") or "unreported"),
                str(row.get("stream_mode") or "unreported"),
                str(row.get("requested_input_tokens") or "unreported"),
                str(row.get("requested_output_target") or "unreported"),
                str(row.get("task_id") or "unreported"),
            )
        ].append(row)
    workload_summaries: list[dict[str, Any]] = []
    for key in sorted(set(request_groups) | set(epoch_groups)):
        (
            source_kind,
            source_id,
            endpoint,
            workload,
            shape,
            phase,
            offered_rps,
            concurrency_ceiling,
            stream_mode,
            requested_input_tokens,
            requested_output_target,
            task_id,
        ) = key
        row_epochs = epoch_groups.get(key, [])
        row_requests = request_groups.get(key, [])
        workload_summaries.append(
            {
                "source_kind": source_kind,
                "source_id": source_id,
                "endpoint_id": endpoint,
                "workload": workload,
                "shape": None if shape == "none" else shape,
                "phase": None if phase == "unreported" else phase,
                "offered_rps": (
                    None if offered_rps == "unreported" else float(offered_rps)
                ),
                "concurrency_ceiling": (
                    None
                    if concurrency_ceiling == "unreported"
                    else int(float(concurrency_ceiling))
                ),
                "stream_mode": (None if stream_mode == "unreported" else stream_mode),
                "requested_input_tokens": (
                    None
                    if requested_input_tokens == "unreported"
                    else int(float(requested_input_tokens))
                ),
                "requested_output_target": (
                    None
                    if requested_output_target == "unreported"
                    else int(float(requested_output_target))
                ),
                "task_id": None if task_id == "unreported" else task_id,
                "load_regime_key": json.dumps(
                    {
                        "source_id": source_id,
                        "phase": None if phase == "unreported" else phase,
                        "offered_rps": (
                            None if offered_rps == "unreported" else offered_rps
                        ),
                        "concurrency_ceiling": (
                            None
                            if concurrency_ceiling == "unreported"
                            else concurrency_ceiling
                        ),
                        "stream_mode": (
                            None if stream_mode == "unreported" else stream_mode
                        ),
                        "requested_input_tokens": (
                            None
                            if requested_input_tokens == "unreported"
                            else requested_input_tokens
                        ),
                        "requested_output_target": (
                            None
                            if requested_output_target == "unreported"
                            else requested_output_target
                        ),
                        "task_id": None if task_id == "unreported" else task_id,
                    },
                    sort_keys=True,
                ),
                **summarize_group(
                    row_requests,
                    row_epochs,
                    seed=deterministic_seed(
                        seed,
                        source_kind,
                        source_id,
                        endpoint,
                        workload,
                        shape,
                        phase,
                        offered_rps,
                        concurrency_ceiling,
                        stream_mode,
                        requested_input_tokens,
                        requested_output_target,
                        task_id,
                    ),
                    bootstrap_replicates=bootstrap_replicates,
                ),
            }
        )
    endpoint_summaries: list[dict[str, Any]] = []
    for endpoint in EXPECTED_ENDPOINT_IDS:
        endpoint_requests = [
            row for row in requests if row.get("endpoint_id") == endpoint
        ]
        endpoint_epochs = [row for row in epochs if row.get("endpoint_id") == endpoint]
        endpoint_summaries.append(
            {
                "endpoint_id": endpoint,
                "request_count": len(endpoint_requests),
                "epoch_count": len(endpoint_epochs),
                "transport_success_count": sum(
                    bool(row.get("transport_success")) for row in endpoint_requests
                ),
                "scientific_success_count": sum(
                    bool(row.get("scientific_success")) for row in endpoint_requests
                ),
                "error_count": sum(
                    not bool(row.get("transport_success")) for row in endpoint_requests
                ),
                "rate_limit_count": sum(
                    bool(row.get("rate_limited")) for row in endpoint_requests
                ),
                "timeout_count": sum(
                    bool(row.get("timeout")) for row in endpoint_requests
                ),
                "server_error_count": sum(
                    bool(row.get("server_error")) for row in endpoint_requests
                ),
                "estimated_cost_usd": sum(
                    float(row.get("estimated_cost_usd") or 0)
                    for row in endpoint_requests
                ),
                "source_kind_count": len(
                    {str(row.get("source_kind")) for row in endpoint_requests}
                ),
                "workload_cell_count": len(
                    {
                        (
                            str(row.get("source_kind")),
                            str(row.get("workload")),
                            str(row.get("shape")),
                        )
                        for row in endpoint_requests
                    }
                ),
                "heterogeneous_metrics_omitted": True,
                "aggregate_policy": (
                    "counts_and_cost_only; RPM, TPM, latency, decode, and quality "
                    "remain in matched workload cells"
                ),
            }
        )
    return endpoint_summaries, workload_summaries


def _coverage_dimension(workload: str, shape: str | None, phase: str | None) -> str:
    value = (
        " ".join(filter(None, (workload, shape, phase))).casefold().replace("-", "_")
    )
    if "recovery" in value:
        return "post_overload_recovery"
    if "near" in value and "quality" in value:
        return "quality_near_saturation"
    if "quality" in value:
        return "quality_low_load"
    if "short_short" in value or "short / short" in value:
        return "aimd_short_short"
    if "input32k_short" in value or "long_short" in value or "long / short" in value:
        return "aimd_long_short"
    if "short_long" in value or "short / long" in value:
        return "aimd_short_long"
    if "mixed" in value:
        return "aimd_mixed"
    if "vision" in value or "image" in value:
        return "vision"
    if "tool" in value:
        return "tool_calling"
    if "structured" in value or "json" in value:
        return "structured_output"
    if "parameter" in value and ("interaction" in value or "pairwise" in value):
        return "parameter_interactions"
    if "parameter" in value:
        return "parameter_validation"
    if "context" in value or "retrieval" in value:
        return "input_context"
    if "output" in value or "decode" in value:
        return "output_length"
    if "smoke" in value or "capability" in value:
        return "capability_smoke"
    return "low_load_baseline"


def _coverage_status(rows: Sequence[Mapping[str, Any]], planned: int) -> str:
    if not rows:
        return "untested"
    skipped = sum(str(row.get("status", "")).startswith("skipped") for row in rows)
    conclusive = [row for row in rows if _row_coverage_conclusive(row)]
    unsupported = [row for row in rows if _row_evidence_backed_unsupported(row)]
    if skipped == len(rows):
        return "skipped"
    if len(rows) < planned or len(conclusive) < len(rows):
        return "inconclusive"
    if unsupported and len(unsupported) == len(rows):
        return "unsupported"
    return "completed"


def build_coverage(
    plans: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    epochs: Sequence[Mapping[str, Any]],
    scope_exclusions: Sequence[Mapping[str, Any]] = (),
    soak_sources: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    requests_by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    requests_by_epoch: dict[tuple[str, str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in requests:
        if row.get("cell_id"):
            requests_by_cell[
                (str(row.get("source_id")), str(row.get("cell_id")))
            ].append(row)
        if row.get("epoch_id"):
            requests_by_epoch[
                (
                    str(row.get("source_kind")),
                    str(row.get("source_id")),
                    str(row.get("epoch_id")),
                )
            ].append(row)
    ledger: list[dict[str, Any]] = []
    for plan in plans:
        physical_rows = requests_by_cell.get(
            (str(plan["source_id"]), str(plan["cell_id"])), []
        )
        rows = (
            [
                row
                for row in physical_rows
                if row.get("semantic_coverage_attempt") is True
            ]
            if plan.get("source_kind") == "direct_completion"
            else physical_rows
        )
        planned = int(plan.get("planned_attempt_count") or 1)
        coverage_status = _coverage_status(rows, planned)
        if (
            plan.get("source_kind") == "direct_completion"
            and not rows
            and plan.get("terminal_outcome_status") is not None
        ):
            terminal_classification = str(
                plan.get("terminal_coverage_classification") or ""
            )
            if terminal_classification == "matched_control_repeated_provider_failure":
                coverage_status = "operational_failure"
            else:
                coverage_status = (
                    "skipped"
                    if str(plan.get("terminal_outcome_status")).startswith("skipped")
                    else "inconclusive"
                )
        coverage_row = {
            "source_kind": plan["source_kind"],
            "source_id": plan["source_id"],
            "endpoint_id": plan["endpoint_id"],
            "coverage_dimension": _coverage_dimension(
                str(plan["workload"]), plan.get("shape"), None
            ),
            "workload": plan["workload"],
            "cell_or_epoch_id": plan["cell_id"],
            "probe_id": plan.get("probe_id"),
            "task_id": plan.get("task_id"),
            "planned_attempt_count": planned,
            "observed_attempt_count": len(rows),
            "conclusive_attempt_count": sum(
                _row_coverage_conclusive(row) for row in rows
            ),
            "status": coverage_status,
        }
        if plan.get("source_kind") == "direct_completion":
            coverage_row.update(
                {
                    "completion_lane": plan.get("shape"),
                    "physical_attempt_count": len(physical_rows),
                    "nonfinal_physical_attempt_count": len(physical_rows) - len(rows),
                    "semantic_attempt_policy": "declared_final_request_id_only",
                    "semantic_final_request_id": plan.get("semantic_final_request_id"),
                    "terminal_coverage_classification": plan.get(
                        "terminal_coverage_classification"
                    ),
                    "supersedes_request_id": plan.get("supersedes_request_id"),
                }
            )
        ledger.append(coverage_row)
    for epoch in epochs:
        if epoch.get("source_kind") == "direct_soak":
            continue
        phase = str(epoch.get("phase") or "").casefold()
        scheduled = int(epoch.get("scheduled_count") or 0)
        completed = int(epoch.get("completed_count") or 0)
        epoch_status = (
            "completed"
            if epoch.get("valid_for_capacity") and completed == scheduled
            else "inconclusive"
        )
        base_row = {
            "source_kind": epoch["source_kind"],
            "source_id": epoch["source_id"],
            "endpoint_id": epoch["endpoint_id"],
            "workload": epoch.get("workload") or epoch.get("shape"),
            "cell_or_epoch_id": epoch["epoch_id"],
            "planned_attempt_count": scheduled,
            "observed_attempt_count": completed,
            "conclusive_attempt_count": completed,
            "status": epoch_status,
            "evidence_scope": (
                "exploratory_fixed_screen"
                if phase == "fixed_screen"
                else "capacity_epoch"
            ),
        }
        ledger.append(
            {
                **base_row,
                "coverage_dimension": _coverage_dimension(
                    str(epoch.get("workload") or ""),
                    _text(epoch.get("shape")),
                    phase,
                ),
            }
        )
        if phase in {"serial_baseline", "confirmation", "confirm"}:
            quality_rows = requests_by_epoch.get(
                (
                    str(epoch.get("source_kind")),
                    str(epoch.get("source_id")),
                    str(epoch.get("epoch_id")),
                ),
                [],
            )
            quality_scored_rows = [
                row
                for row in quality_rows
                if row.get("quality_scored") is True
                and isinstance(row.get("functional_valid"), bool)
            ]
            quality_status = (
                "completed"
                if scheduled > 0
                and completed == scheduled
                and len(quality_rows) == scheduled
                and len(quality_scored_rows) == scheduled
                else "inconclusive"
            )
            quality_dimension = (
                "quality_low_load"
                if phase == "serial_baseline"
                else "quality_near_saturation"
            )
            ledger.append(
                {
                    **base_row,
                    "coverage_dimension": quality_dimension,
                    "planned_attempt_count": scheduled,
                    "observed_attempt_count": len(quality_rows),
                    "conclusive_attempt_count": len(quality_scored_rows),
                    "status": quality_status,
                    "evidence_scope": (
                        "paired_quality_low_load_epoch"
                        if phase == "serial_baseline"
                        else "paired_quality_near_saturation_epoch"
                    ),
                }
            )
    for source in soak_sources:
        source_id = str(source.get("source_id") or "")
        plan_cells = {
            str(row.get("cell_id")): row
            for row in source.get("plan_cells", ())
            if isinstance(row, Mapping) and row.get("cell_id") is not None
        }
        phase_summaries = [
            row for row in source.get("phase_summaries", ()) if isinstance(row, Mapping)
        ]
        phases = {
            (str(row.get("cell_id")), str(row.get("phase"))): row
            for row in phase_summaries
        }
        pairs_by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in source.get("quality_summaries", ()):
            if isinstance(row, Mapping):
                pairs_by_cell[str(row.get("cell_id"))].append(row)
        cell_rows = {
            str(row.get("cell_id")): row
            for row in source.get("cell_rows", ())
            if isinstance(row, Mapping) and row.get("cell_id") is not None
        }
        for cell_id, plan_cell in plan_cells.items():
            endpoint = _require_endpoint(plan_cell.get("model_id"))
            shape = str(plan_cell.get("shape") or "")
            low_expected = int(plan_cell.get("low_load_requests") or 0)
            low = phases.get((cell_id, "paired_low_load"))
            low_observed = int(low.get("completed_request_rows") or 0) if low else 0
            low_complete = bool(
                low
                and low_expected > 0
                and low.get("status") == "complete"
                and int(low.get("scheduled_requests") or 0) == low_expected
                and low_observed == low_expected
                and int(low.get("provider_send_attempts") or 0) == low_expected
            )
            ledger.append(
                {
                    "source_kind": "direct_soak",
                    "source_id": source_id,
                    "endpoint_id": endpoint,
                    "coverage_dimension": "low_load_baseline",
                    "workload": shape,
                    "cell_or_epoch_id": (
                        str(low.get("phase_id"))
                        if low
                        else f"{cell_id}:paired_low_load"
                    ),
                    "planned_attempt_count": low_expected,
                    "observed_attempt_count": low_observed,
                    "conclusive_attempt_count": low_observed if low_complete else 0,
                    "status": "completed" if low_complete else "inconclusive",
                    "evidence_scope": "exact_two_minute_soak_low_load_phase",
                    "soak_cell_id": cell_id,
                }
            )

            recovery_expected = int(plan_cell.get("recovery_requests") or 0)
            recovery = phases.get((cell_id, "post_soak_recovery"))
            recovery_observed = (
                int(recovery.get("completed_request_rows") or 0) if recovery else 0
            )
            cell_result = cell_rows.get(cell_id, {})
            recovery_result_recorded = isinstance(
                cell_result.get("post_soak_recovery_predeclared_pass"), bool
            )
            recovery_complete = bool(
                recovery
                and recovery_expected > 0
                and recovery.get("status") == "complete"
                and int(recovery.get("scheduled_requests") or 0) == recovery_expected
                and recovery_observed == recovery_expected
                and int(recovery.get("provider_send_attempts") or 0)
                == recovery_expected
                and recovery_result_recorded
            )
            ledger.append(
                {
                    "source_kind": "direct_soak",
                    "source_id": source_id,
                    "endpoint_id": endpoint,
                    "coverage_dimension": "post_overload_recovery",
                    "workload": shape,
                    "cell_or_epoch_id": (
                        str(recovery.get("phase_id"))
                        if recovery
                        else f"{cell_id}:post_soak_recovery"
                    ),
                    "planned_attempt_count": recovery_expected,
                    "observed_attempt_count": recovery_observed,
                    "conclusive_attempt_count": (
                        recovery_observed if recovery_complete else 0
                    ),
                    "status": "completed" if recovery_complete else "inconclusive",
                    "evidence_scope": "exact_post_soak_recovery_phase",
                    "soak_cell_id": cell_id,
                    "predeclared_recovery_pass": cell_result.get(
                        "post_soak_recovery_predeclared_pass"
                    ),
                    "recovery_acceptance_reasons": cell_result.get(
                        "post_soak_recovery_acceptance_reasons"
                    ),
                }
            )

            pair_rows = pairs_by_cell.get(cell_id, [])
            conclusive_pairs = [
                row
                for row in pair_rows
                if row.get("exact_request_payload_hash_match") is True
                and isinstance(row.get("low_load_success"), bool)
                and isinstance(row.get("near_load_success"), bool)
                and _number(row.get("low_load_quality_score")) is not None
                and _number(row.get("near_load_quality_score")) is not None
                and isinstance(row.get("predeclared_quality_acceptance_pass"), bool)
            ]
            pairs_complete = bool(
                low_expected > 0
                and len(pair_rows) == low_expected
                and len(conclusive_pairs) == low_expected
            )
            for dimension, role in (
                ("quality_low_load", "low_load"),
                ("quality_near_saturation", "near_load"),
            ):
                ledger.append(
                    {
                        "source_kind": "direct_soak",
                        "source_id": source_id,
                        "endpoint_id": endpoint,
                        "coverage_dimension": dimension,
                        "workload": shape,
                        "cell_or_epoch_id": f"{cell_id}:quality_pairs:{role}",
                        "planned_attempt_count": low_expected,
                        "observed_attempt_count": len(pair_rows),
                        "conclusive_attempt_count": len(conclusive_pairs),
                        "status": "completed" if pairs_complete else "inconclusive",
                        "evidence_scope": f"exact_two_minute_soak_quality_{role}_pairs",
                        "soak_cell_id": cell_id,
                        "sampling_unit": "quality_pair_id",
                    }
                )
    for exclusion in scope_exclusions:
        endpoint = _require_endpoint(exclusion.get("endpoint_id"))
        exclusion_id = _text(exclusion.get("scope_exclusion_id"))
        dimension = _text(exclusion.get("coverage_dimension"))
        reason = _text(exclusion.get("reason"))
        if exclusion_id is None or dimension not in REQUIRED_COVERAGE_DIMENSIONS:
            raise DirectReportError("scope exclusion has an invalid coverage contract")
        if reason is None or exclusion.get("status") != "untested":
            raise DirectReportError("scope exclusion must remain explicitly untested")
        ledger.append(
            {
                "source_kind": str(exclusion.get("source_kind") or "direct_breadth"),
                "source_id": str(exclusion.get("source_id") or ""),
                "endpoint_id": endpoint,
                "coverage_dimension": dimension,
                "workload": exclusion_id,
                "cell_or_epoch_id": f"scope-exclusion:{exclusion_id}",
                "planned_attempt_count": 0,
                "observed_attempt_count": 0,
                "conclusive_attempt_count": 0,
                "status": "untested",
                "evidence_scope": "manifest_scope_exclusion",
                "scope_exclusion_id": exclusion_id,
                "measurement_label": str(
                    exclusion.get("measurement_label") or exclusion_id
                ),
                "exclusion_reason": reason,
                "claim_policy": "explicitly_excluded_not_tested",
                "scope_exclusion_schema_version": exclusion.get("schema_version"),
                "source_manifest_sha256": exclusion.get("source_manifest_sha256"),
            }
        )
    for replacement in (
        row
        for row in ledger
        if row.get("source_kind") == "direct_completion"
        and row.get("supersedes_request_id") is not None
    ):
        if replacement.get("status") not in {"completed", "unsupported"}:
            replacement["supersession_status"] = (
                "not_applied_final_semantic_attempt_inconclusive"
            )
            continue
        source_request_id = str(replacement["supersedes_request_id"])
        candidates = [
            row
            for row in ledger
            if row.get("source_kind") == "direct_breadth"
            and str(row.get("cell_or_epoch_id")) == source_request_id
            and row.get("endpoint_id") == replacement.get("endpoint_id")
        ]
        if len(candidates) > 1:
            raise DirectReportError(
                "completion supersession matches multiple source coverage cells"
            )
        if not candidates:
            replacement["supersession_status"] = "source_cell_not_loaded"
            continue
        target = candidates[0]
        target_dimension = str(target.get("coverage_dimension"))
        completion_lane = replacement.get("completion_lane")
        allowed_dimensions_by_lane = {
            "capability_retry": {
                "output_length",
                "parameter_interactions",
                "parameter_validation",
                "structured_output",
                "tool_calling",
            },
            "context_retry": {"input_context"},
        }
        if target_dimension not in allowed_dimensions_by_lane.get(
            str(completion_lane), set()
        ):
            raise DirectReportError(
                "completion supersession lane is inconsistent with the exact source "
                "coverage dimension"
            )
        inferred_dimension = replacement.get("coverage_dimension")
        replacement.update(
            {
                "coverage_dimension": target_dimension,
                "retry_task_family_inferred_dimension": inferred_dimension,
                "supersession_dimension_policy": (
                    "inherited_from_exact_endpoint_and_source_request_id"
                ),
            }
        )
        if target.get("status") == "inconclusive":
            target.update(
                {
                    "status": "superseded",
                    "superseded_by_source_id": replacement.get("source_id"),
                    "superseded_by_request_id": replacement.get(
                        "semantic_final_request_id"
                    ),
                    "supersession_policy": (
                        "conclusive_declared_final_completion_attempt"
                    ),
                }
            )
            replacement["supersession_status"] = "applied_to_inconclusive_source"
        else:
            replacement["supersession_status"] = (
                f"not_applied_source_status_{target.get('status')}"
            )

    # A matched-control closure row is a stronger measurement of the same exact
    # capability state than the old unpaired 4xx/timeout row. Preserve the old
    # row, but do not let it poison endpoint-level experimental coverage after
    # the exact state has been rerun conclusively.
    matched_replacements = [
        row
        for row in ledger
        if (
            row.get("completion_lane") == "matched_control_closure"
            or str(row.get("source_id", "")).startswith("do-matched-closure-")
        )
        and row.get("status") in {"completed", "unsupported"}
        and row.get("probe_id") is not None
    ]
    for replacement in matched_replacements:
        for target in ledger:
            if target is replacement or target.get("status") != "inconclusive":
                continue
            if (
                target.get("endpoint_id") == replacement.get("endpoint_id")
                and target.get("coverage_dimension")
                == replacement.get("coverage_dimension")
                and target.get("probe_id") == replacement.get("probe_id")
            ):
                target.update(
                    {
                        "status": "superseded",
                        "superseded_by_source_id": replacement.get("source_id"),
                        "superseded_by_cell_id": replacement.get("cell_or_epoch_id"),
                        "supersession_policy": (
                            "exact_endpoint_dimension_probe_matched_control"
                        ),
                    }
                )

    # Load, recovery, and paired-quality cells are repeated measurements. A
    # failed replicate remains a failure observation, but it does not mean the
    # experiment was never completed once another exact endpoint/workload
    # replicate is complete. Grouping by workload prevents a short/short pass
    # from standing in for a long-context or mixed-load experiment.
    replicated_dimensions = {
        "low_load_baseline",
        "post_overload_recovery",
        "quality_low_load",
        "quality_near_saturation",
    }
    replicated_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in ledger:
        if row.get("coverage_dimension") not in replicated_dimensions:
            continue
        replicated_groups[
            (
                str(row.get("endpoint_id")),
                str(row.get("coverage_dimension")),
                str(row.get("workload")),
            )
        ].append(row)
    for members in replicated_groups.values():
        completed = [row for row in members if row.get("status") == "completed"]
        if not completed:
            continue
        for row in members:
            if row.get("status") != "inconclusive":
                continue
            row.update(
                {
                    "status": "replicate_failure_observed",
                    "completed_replicate_source_ids": sorted(
                        {str(item.get("source_id")) for item in completed}
                    ),
                    "coverage_policy": (
                        "exact endpoint/dimension/workload completion exists; "
                        "retain failed replicate without relabelling it successful"
                    ),
                }
            )

    # Repeated soak attempts are replicates, not an all-or-nothing chain. Once
    # one complete two-minute cell exists for the same endpoint, workload and
    # coverage dimension, a transport-gated replicate remains visible as a
    # failure observation but no longer means the experiment was never done.
    soak_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        if row.get("source_kind") != "direct_soak":
            continue
        soak_groups[
            (
                str(row.get("endpoint_id")),
                str(row.get("coverage_dimension")),
                str(row.get("workload")),
            )
        ].append(row)
    for members in soak_groups.values():
        completed = [row for row in members if row.get("status") == "completed"]
        if not completed:
            continue
        for row in members:
            if row.get("status") == "inconclusive":
                row.update(
                    {
                        "status": "replicate_failure_observed",
                        "completed_replicate_source_ids": sorted(
                            {str(item.get("source_id")) for item in completed}
                        ),
                        "coverage_policy": (
                            "complete_replication_exists; retain failed replicate "
                            "without relabelling it successful"
                        ),
                    }
                )
    matrix: list[dict[str, Any]] = []
    for endpoint in EXPECTED_ENDPOINT_IDS:
        for dimension in REQUIRED_COVERAGE_DIMENSIONS:
            cells = [
                row
                for row in ledger
                if row["endpoint_id"] == endpoint
                and row["coverage_dimension"] == dimension
            ]
            exclusions = [
                row
                for row in cells
                if row.get("evidence_scope") == "manifest_scope_exclusion"
            ]
            measurement_cells = [
                row
                for row in cells
                if row.get("evidence_scope") != "manifest_scope_exclusion"
            ]
            statuses = Counter(str(row["status"]) for row in measurement_cells)
            has_scope_exclusion = bool(exclusions)
            if not measurement_cells and has_scope_exclusion:
                status = "untested"
            elif statuses.get("completed"):
                status = "completed"
            elif statuses.get("unsupported"):
                status = "unsupported"
            elif statuses.get("operational_failure"):
                status = "operational_failure"
            elif statuses.get("inconclusive"):
                status = "inconclusive"
            elif statuses.get("skipped"):
                status = "skipped"
            else:
                status = "untested"
            matrix.append(
                {
                    "endpoint_id": endpoint,
                    "coverage_dimension": dimension,
                    "status": status,
                    "planned_cell_or_epoch_count": len(measurement_cells),
                    "observed_attempt_count": sum(
                        int(row.get("observed_attempt_count") or 0)
                        for row in measurement_cells
                    ),
                    "completed_subcell_count": statuses.get("completed", 0),
                    "unsupported_subcell_count": statuses.get("unsupported", 0),
                    "operational_failure_subcell_count": statuses.get(
                        "operational_failure", 0
                    ),
                    "inconclusive_subcell_count": statuses.get("inconclusive", 0),
                    "skipped_subcell_count": statuses.get("skipped", 0),
                    "superseded_subcell_count": statuses.get("superseded", 0),
                    "replicate_failure_subcell_count": statuses.get(
                        "replicate_failure_observed", 0
                    ),
                    "explicit_untested_subtest_count": len(exclusions),
                    "has_explicit_scope_exclusions": has_scope_exclusion,
                }
            )
    resolved = sum(
        row["status"] in {"completed", "unsupported", "operational_failure"}
        for row in matrix
    )
    return (
        ledger,
        matrix,
        {
            "required_endpoint_count": len(EXPECTED_ENDPOINT_IDS),
            "required_dimension_count": len(REQUIRED_COVERAGE_DIMENSIONS),
            "required_endpoint_dimension_cells": len(matrix),
            "resolved_experiment_cells": resolved,
            # Legacy key retained for existing consumers. Its value now means
            # resolved experimental coverage, as the explicit claim below says.
            "completed_or_evidence_backed_unsupported_cells": resolved,
            "coverage_fraction": resolved / len(matrix),
            "is_100_percent": resolved == len(matrix),
            "status_counts": dict(
                sorted(Counter(row["status"] for row in matrix).items())
            ),
            "matrix_scope": "broad_endpoint_by_dimension",
            "explicit_scope_exclusion_count": len(scope_exclusions),
            "coverage_claim": (
                "Broad endpoint-by-dimension coverage. A completed exact subcell "
                "establishes that the dimension was exercised; unsupported, "
                "operational-failure, inconclusive, skipped, superseded, "
                "failed-replicate, and named zero-attempt subtests remain "
                "separately counted and are not implied successful."
            ),
        },
    )


def build_capacity_summary(
    epochs: Sequence[Mapping[str, Any]],
    *,
    requests: Sequence[Mapping[str, Any]] = (),
    seed: int,
    bootstrap_replicates: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in epochs:
        if row.get("source_kind") == "direct_aimd":
            groups[
                (
                    str(row.get("source_id")),
                    str(row["endpoint_id"]),
                    str(row.get("shape") or "unspecified"),
                )
            ].append(row)
    output: list[dict[str, Any]] = []
    for (source_id, endpoint, shape), rows in sorted(groups.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                _integer(row.get("sequence")) is None,
                _integer(row.get("sequence")) or 0,
                _timestamp(row.get("started_at")) or 0,
                str(row.get("epoch_id")),
            ),
        )
        valid = [row for row in ordered if row.get("valid_for_capacity")]
        healthy = [row for row in valid if row.get("healthy") is True]
        by_rate: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
        for row in healthy:
            rate = _number(row.get("offered_rps"))
            if rate is not None:
                by_rate[rate].append(row)
        confirmation_by_rate: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
        for row in valid:
            phase = str(row.get("phase") or "").casefold()
            rate = _number(row.get("offered_rps"))
            if (
                rate is not None
                and phase in {"confirmation", "confirm"}
                and row.get("healthy") is True
            ):
                confirmation_by_rate[rate].append(row)
        confirmed_rates = [
            rate for rate, values in confirmation_by_rate.items() if len(values) >= 3
        ]
        confirmed_rate = max(confirmed_rates, default=None)
        highest = max(by_rate, default=None)
        knee = None
        for left, right in zip(valid, valid[1:]):
            if left.get("healthy") is False and right.get("healthy") is False:
                rates = [
                    value
                    for row in (left, right)
                    if (value := _number(row.get("offered_rps"))) is not None
                ]
                if rates:
                    knee = min(rates)
                    break
        # The frozen protocol defines saturation as two consecutive unhealthy
        # valid epochs. A lone unhealthy point remains visible in the epoch plot
        # but cannot manufacture an upper capacity bracket.
        bracket_upper = (
            knee
            if confirmed_rate is not None and knee is not None and knee > confirmed_rate
            else None
        )
        right_censored = bool(confirmed_rate is not None and bracket_upper is None)
        candidate_epochs = (
            confirmation_by_rate.get(confirmed_rate, [])
            if confirmed_rate is not None
            else []
        )
        candidate_epoch_ids = {
            str(row.get("epoch_id"))
            for row in candidate_epochs
            if row.get("epoch_id") is not None
        }
        candidate_requests = [
            row
            for row in requests
            if str(row.get("source_id")) == source_id
            and str(row.get("endpoint_id")) == endpoint
            and str(row.get("epoch_id")) in candidate_epoch_ids
        ]
        candidate_successful_requests = [
            row for row in candidate_requests if row.get("scientific_success") is True
        ]

        def numeric_range(
            source_rows: Sequence[Mapping[str, Any]], key: str
        ) -> list[float] | None:
            values = [
                float(value)
                for row in source_rows
                if (value := _number(row.get(key))) is not None
            ]
            return [min(values), max(values)] if values else None

        def interval(key: str) -> dict[str, Any]:
            values = [
                float(value)
                for row in candidate_epochs
                if (value := _number(row.get(key))) is not None
            ]
            result = _bootstrap_metric(
                values,
                seed=deterministic_seed(
                    seed, source_id, endpoint, shape, confirmed_rate, key
                ),
                replicates=bootstrap_replicates,
                statistic="mean",
            )
            result["sampling_unit"] = "epoch_id"
            return result

        achieved_interval = interval("achieved_rpm")
        completed_interval = interval("completed_rpm")
        input_interval = interval("effective_input_tpm")
        output_interval = interval("effective_output_tpm")
        ttft_interval = interval("ttft_p50_seconds")
        latency_interval = interval("latency_p95_seconds")
        output.append(
            {
                "source_id": source_id,
                "endpoint_id": endpoint,
                "shape": shape,
                "epoch_count": len(rows),
                "valid_epoch_count": len(valid),
                "healthy_epoch_count": len(healthy),
                "tested_min_offered_rps": min(
                    (
                        value
                        for row in rows
                        if (value := _number(row.get("offered_rps"))) is not None
                    ),
                    default=None,
                ),
                "tested_max_offered_rps": max(
                    (
                        value
                        for row in rows
                        if (value := _number(row.get("offered_rps"))) is not None
                    ),
                    default=None,
                ),
                "highest_observed_healthy_rps": highest,
                "confirmed_healthy_offered_rps": confirmed_rate,
                "highest_observed_healthy_rpm": (
                    highest * 60 if highest is not None else None
                ),
                "confirmed_healthy_offered_rpm": (
                    confirmed_rate * 60 if confirmed_rate is not None else None
                ),
                "candidate_rate_confirmation_epoch_count": len(candidate_epochs),
                "candidate_confirmation_epoch_ids": sorted(candidate_epoch_ids),
                "candidate_epoch_duration_seconds_range": numeric_range(
                    candidate_epochs, "elapsed_seconds"
                ),
                "candidate_concurrency_ceiling_range": numeric_range(
                    candidate_epochs, "concurrency_ceiling"
                ),
                "candidate_scheduled_request_count": sum(
                    int(row.get("scheduled_count") or 0) for row in candidate_epochs
                ),
                "candidate_completed_request_count": sum(
                    int(row.get("completed_count") or 0) for row in candidate_epochs
                ),
                "candidate_successful_request_count": sum(
                    int(row.get("success_count") or 0) for row in candidate_epochs
                ),
                "candidate_rate_limit_count": sum(
                    int(row.get("rate_limit_count") or 0) for row in candidate_epochs
                ),
                "candidate_timeout_count": sum(
                    int(row.get("timeout_count") or 0) for row in candidate_epochs
                ),
                "candidate_server_error_count": sum(
                    int(row.get("server_error_count") or 0) for row in candidate_epochs
                ),
                "candidate_request_row_count": len(candidate_requests),
                "candidate_realized_input_tokens_range": numeric_range(
                    candidate_successful_requests, "input_tokens"
                ),
                "candidate_realized_output_tokens_range": numeric_range(
                    candidate_successful_requests, "output_tokens"
                ),
                "candidate_requested_input_tokens_range": numeric_range(
                    candidate_requests, "requested_input_tokens"
                ),
                "candidate_requested_output_target_range": numeric_range(
                    candidate_requests, "requested_output_target"
                ),
                "candidate_stream_modes": sorted(
                    {
                        str(row.get("stream_mode"))
                        for row in candidate_requests
                        if row.get("stream_mode") is not None
                    }
                ),
                "saturation_knee_rps": knee,
                "capacity_lower_bound_rps": confirmed_rate,
                "capacity_upper_bound_rps": bracket_upper,
                "capacity_lower_bound_rpm": (
                    confirmed_rate * 60 if confirmed_rate is not None else None
                ),
                "capacity_upper_bound_rpm": (
                    bracket_upper * 60 if bracket_upper is not None else None
                ),
                "confirmed_healthy_offered_upper_rps": bracket_upper,
                "confirmed_healthy_offered_upper_rpm": (
                    bracket_upper * 60 if bracket_upper is not None else None
                ),
                "capacity_metric_kind": (
                    "healthy_realized_offered_arrival_rate_over_short_confirmation_epochs; "
                    "not_drain_inclusive_completed_goodput_or_sustained_capacity"
                ),
                "right_censored": right_censored,
                "capacity_claim": (
                    "confirmed_right_censored_lower_bound"
                    if right_censored
                    else (
                        "confirmed_bracketed_interval"
                        if confirmed_rate is not None
                        else (
                            "unconfirmed_healthy_observation_only"
                            if highest is not None
                            else "censored_no_valid_healthy_epoch"
                        )
                    )
                ),
                "achieved_rpm": achieved_interval.get("estimate"),
                "completed_rpm": completed_interval.get("estimate"),
                "effective_input_tpm": input_interval.get("estimate"),
                "effective_output_tpm": output_interval.get("estimate"),
                "ttft_p50_seconds": ttft_interval.get("estimate"),
                "latency_p95_seconds": latency_interval.get("estimate"),
                "achieved_rpm_ci95": achieved_interval,
                "completed_rpm_ci95": completed_interval,
                "effective_input_tpm_ci95": input_interval,
                "effective_output_tpm_ci95": output_interval,
                "ttft_p50_seconds_ci95": ttft_interval,
                "latency_p95_seconds_ci95": latency_interval,
            }
        )
    return output


_HETEROGENEOUS_ENDPOINT_METRICS = frozenset(
    {
        "transport_success_rate",
        "quality_pass_rate",
        "achieved_rpm",
        "goodput_rpm",
        "effective_input_tpm",
        "effective_output_tpm",
        "quality_adjusted_output_tpm_scored_only",
        "ttft_p50_seconds",
        "latency_p50_seconds",
        "latency_p95_seconds",
        "post_ttft_output_tps_proxy_p50",
        "aggregate_output_goodput_tps_epoch_p50",
    }
)
_CAPACITY_CI_FIELDS = {
    "achieved_rpm_ci95": "achieved_rpm",
    "completed_rpm_ci95": "completed_rpm",
    "effective_input_tpm_ci95": "effective_input_tpm",
    "effective_output_tpm_ci95": "effective_output_tpm",
    "ttft_p50_seconds_ci95": "ttft_p50_seconds",
    "latency_p95_seconds_ci95": "latency_p95_seconds",
}
_CI_POINT_UNSET = object()


def _validate_ci(
    value: Any,
    *,
    pointer: str,
    expected_unit: str,
    require_estimate: bool,
    point_estimate: Any = _CI_POINT_UNSET,
    errors: list[str],
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{pointer}: missing CI object")
        return
    if value.get("sampling_unit") != expected_unit:
        errors.append(f"{pointer}: sampling_unit must be {expected_unit}")
    estimate = _number(value.get("estimate"))
    low = _number(value.get("ci95_low"))
    high = _number(value.get("ci95_high"))
    n_units = _integer(value.get("n_units"))
    if n_units is None:
        errors.append(f"{pointer}: n_units must be a non-negative integer")
    if estimate is None:
        if require_estimate:
            errors.append(f"{pointer}: estimate and bounds are required")
        if low is not None or high is not None:
            errors.append(f"{pointer}: null estimate cannot have CI bounds")
        if (
            point_estimate is not _CI_POINT_UNSET
            and _number(point_estimate) is not None
        ):
            errors.append(
                f"{pointer}: CI estimate must equal the reported point estimate"
            )
        return
    if low is None or high is None:
        errors.append(f"{pointer}: estimate requires both CI bounds")
    elif not low - 1e-12 <= estimate <= high + 1e-12:
        errors.append(f"{pointer}: CI must contain its estimate")
    if not n_units:
        errors.append(f"{pointer}: a reported estimate requires n_units > 0")
    if point_estimate is not _CI_POINT_UNSET:
        point = _number(point_estimate)
        if point is None or not math.isclose(
            point,
            estimate,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append(
                f"{pointer}: CI estimate must equal the reported point estimate"
            )


def validate_public_analysis_contract(
    analysis: Mapping[str, Any], *, require_complete: bool = False
) -> dict[str, Any]:
    """Validate schema, sampling units, units, intervals, and final readiness.

    The validator deliberately does not bless the scientific conclusions. It
    proves that the public tables obey the frozen analysis contract and that a
    final build is not made from incomplete coverage or orphaned evidence.
    """

    errors: list[str] = []
    if analysis.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    inventory = _as_analysis_rows(analysis.get("endpoint_inventory"))
    inventory_ids = [str(row.get("endpoint_id") or "") for row in inventory]
    if inventory_ids != list(EXPECTED_ENDPOINT_IDS):
        errors.append(
            "endpoint_inventory must contain the exact hosted endpoint IDs in order"
        )
    endpoint_summaries = _as_analysis_rows(analysis.get("endpoint_summaries"))
    endpoint_ids = [str(row.get("endpoint_id") or "") for row in endpoint_summaries]
    if endpoint_ids != list(EXPECTED_ENDPOINT_IDS):
        errors.append(
            "endpoint_summaries must contain the exact hosted endpoint IDs in order"
        )
    for index, row in enumerate(endpoint_summaries):
        for key in _HETEROGENEOUS_ENDPOINT_METRICS:
            if row.get(key) is not None:
                errors.append(
                    f"endpoint_summaries[{index}].{key}: heterogeneous aggregate forbidden"
                )
        if row.get("heterogeneous_metrics_omitted") is not True:
            errors.append(
                f"endpoint_summaries[{index}]: heterogeneous_metrics_omitted must be true"
            )

    cost_summary = _mapping(analysis.get("cost_summary"))
    if not cost_summary and require_complete:
        errors.append("final build requires verified cost accounting")
    if cost_summary:
        if cost_summary.get("schema_version") != "digitalocean_public_cost_summary_v1":
            errors.append("cost_summary.schema_version is invalid")
        attributed = _number(cost_summary.get("request_attributed_estimated_cost_usd"))
        conservative = _number(cost_summary.get("conservative_campaign_exposure_usd"))
        cost_cap = _number(cost_summary.get("cost_cap_usd"))
        if attributed is None or attributed < 0:
            errors.append(
                "cost_summary.request_attributed_estimated_cost_usd must be nonnegative"
            )
        endpoint_attributed = sum(
            _number(row.get("estimated_cost_usd")) or 0.0 for row in endpoint_summaries
        )
        endpoint_request_count = sum(
            _integer(row.get("request_count")) or 0 for row in endpoint_summaries
        )
        if attributed is not None and not math.isclose(
            attributed, endpoint_attributed, rel_tol=0, abs_tol=1e-9
        ):
            errors.append(
                "cost_summary attributed total disagrees with endpoint summaries"
            )
        attributed_count = _integer(cost_summary.get("cost_attributed_request_count"))
        unattributed_count = _integer(
            cost_summary.get("cost_unattributed_request_count")
        )
        if (
            attributed_count is None
            or unattributed_count is None
            or attributed_count + unattributed_count != endpoint_request_count
        ):
            errors.append(
                "cost_summary request attribution counts disagree with endpoint summaries"
            )
        attribution_complete = cost_summary.get("request_cost_attribution_complete")
        if not isinstance(attribution_complete, bool) or attribution_complete != (
            unattributed_count == 0
        ):
            errors.append("cost_summary request attribution completeness is invalid")
        if require_complete and attribution_complete is not True:
            errors.append("final build requires complete request cost attribution")
        if cost_summary.get("estimand_relationship") != "overlapping_non_additive":
            errors.append("cost_summary estimand relationship is invalid")
        reported_402 = cost_summary.get("billing_credit_http_402_latched")
        if not isinstance(reported_402, bool):
            errors.append("cost_summary HTTP 402 latch aggregate must be boolean")
        stage_rows = _as_analysis_rows(cost_summary.get("source_stages"))
        if conservative is not None and conservative < 0:
            errors.append(
                "cost_summary.conservative_campaign_exposure_usd must be nonnegative"
            )
        if cost_cap is not None and cost_cap <= 0:
            errors.append("cost_summary.cost_cap_usd must be positive")
        if stage_rows and conservative is None:
            errors.append(
                "cost_summary with source stages requires conservative campaign exposure"
            )
        if stage_rows and cost_cap is None:
            errors.append("cost_summary with source stages requires a cost cap")
        if require_complete and not stage_rows:
            errors.append("final build requires a verified cost source chain")
        if (
            conservative is not None
            and cost_cap is not None
            and conservative > cost_cap + 1e-9
        ):
            errors.append("cost_summary conservative exposure exceeds the cost cap")
        source_ids: list[str] = []
        latched_source_ids: list[str] = []
        parsed_windows: list[tuple[datetime, datetime]] = []
        observed_cap_history: list[float] = []
        observed_cap_revision_count = 0
        terminal_status_allowlist = {
            "do_direct_summary_v1": {"complete_right_censored", "complete"},
            "do_direct_soak_summary_v1": {
                "execution_complete_science_incomplete",
                "complete",
            },
            "do_direct_capability_summary_v3": {"terminal_coverage_complete"},
            "do_direct_context_summary_v3": {
                "execution_complete_scientifically_incomplete",
                "complete",
            },
            "do_direct_completion_summary_v1": {
                "complete",
                "incomplete_or_censored",
            },
            "do_matched_closure_summary_v1": {
                "complete",
                "incomplete_or_censored",
            },
        }
        for index, stage in enumerate(stage_rows):
            pointer = f"cost_summary.source_stages[{index}]"
            source_id = str(stage.get("source_id") or "")
            source_ids.append(source_id)
            prior = _number(stage.get("prior_conservative_exposure_usd"))
            cumulative = _number(stage.get("cumulative_conservative_exposure_usd"))
            incremental = _number(stage.get("incremental_conservative_exposure_usd"))
            stage_cap = _number(stage.get("cost_cap_usd"))
            if (
                prior is None
                or prior < 0
                or cumulative is None
                or cumulative < prior
                or stage_cap is None
                or cumulative > stage_cap + 1e-9
            ):
                errors.append(f"{pointer}: prior/cumulative exposure is invalid")
            elif incremental is None or not math.isclose(
                incremental,
                cumulative - prior,
                rel_tol=0,
                abs_tol=1e-9,
            ):
                errors.append(f"{pointer}: incremental exposure is inconsistent")
            revision_from = _number(stage.get("cost_cap_revision_from_usd"))
            revision_to = _number(stage.get("cost_cap_revision_to_usd"))
            if stage_cap is not None:
                if not observed_cap_history:
                    observed_cap_history.append(stage_cap)
                    if revision_from is not None or revision_to is not None:
                        errors.append(
                            f"{pointer}: initial stage must not declare a cap revision"
                        )
                else:
                    previous_cap = observed_cap_history[-1]
                    if stage_cap < previous_cap - 1e-9:
                        errors.append(f"{pointer}: stage cap decreased")
                    cap_changed = not math.isclose(
                        stage_cap,
                        previous_cap,
                        rel_tol=0,
                        abs_tol=1e-9,
                    )
                    if cap_changed:
                        observed_cap_revision_count += 1
                        observed_cap_history.append(stage_cap)
                        if (
                            revision_from is None
                            or revision_to is None
                            or not math.isclose(
                                revision_from,
                                previous_cap,
                                rel_tol=0,
                                abs_tol=1e-9,
                            )
                            or not math.isclose(
                                revision_to,
                                stage_cap,
                                rel_tol=0,
                                abs_tol=1e-9,
                            )
                        ):
                            errors.append(
                                f"{pointer}: cap revision receipt is inconsistent"
                            )
                    elif revision_from is not None or revision_to is not None:
                        errors.append(
                            f"{pointer}: unchanged cap must not declare a revision"
                        )
            summary_sha = str(stage.get("summary_sha256") or "")
            if not re.fullmatch(r"[0-9a-f]{64}", summary_sha):
                errors.append(f"{pointer}.summary_sha256 is invalid")
            summary_schema = _text(stage.get("summary_schema_version"))
            terminal_status = _text(stage.get("terminal_status"))
            if summary_schema not in terminal_status_allowlist:
                errors.append(f"{pointer}.summary_schema_version is required")
            elif terminal_status not in terminal_status_allowlist[summary_schema]:
                errors.append(f"{pointer}.terminal_status is invalid")
            if stage.get("http_402_latched") is True:
                latched_source_ids.append(source_id)
            elif stage.get("http_402_latched") is not False:
                errors.append(f"{pointer}.http_402_latched must be boolean")
            if stage.get("cost_basis") == "portable_reconciliation":
                if stage.get(
                    "reconciliation_schema_version"
                ) != "do_direct_aimd_reconciliation_v1" or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(stage.get("reconciliation_sha256") or ""),
                ):
                    errors.append(
                        f"{pointer}: portable reconciliation proof is invalid"
                    )
            started_text = _text(stage.get("started_at"))
            ended_text = _text(stage.get("ended_at"))
            try:
                started = datetime.fromisoformat(
                    str(started_text).replace("Z", "+00:00")
                )
                ended = datetime.fromisoformat(str(ended_text).replace("Z", "+00:00"))
                if started.tzinfo is None or ended.tzinfo is None or ended < started:
                    raise ValueError
                parsed_windows.append(
                    (
                        started.astimezone(timezone.utc),
                        ended.astimezone(timezone.utc),
                    )
                )
            except (TypeError, ValueError):
                errors.append(f"{pointer}: UTC stage window is invalid")
        if len(source_ids) != len(set(source_ids)):
            errors.append("cost_summary source stage IDs must be unique")
        declared_revision_count = _integer(cost_summary.get("cost_cap_revision_count"))
        if declared_revision_count != observed_cap_revision_count:
            errors.append("cost_summary cap revision count is inconsistent")
        declared_cap_history = cost_summary.get("cost_cap_history_usd")
        if (
            not isinstance(declared_cap_history, Sequence)
            or isinstance(declared_cap_history, (str, bytes, bytearray))
            or len(declared_cap_history) != len(observed_cap_history)
            or any(
                _number(declared) is None
                or not math.isclose(
                    float(_number(declared)),
                    observed,
                    rel_tol=0,
                    abs_tol=1e-9,
                )
                for declared, observed in zip(
                    declared_cap_history,
                    observed_cap_history,
                    strict=True,
                )
            )
        ):
            errors.append("cost_summary cap history is inconsistent")
        declared_latched_ids = cost_summary.get("http_402_latched_source_ids")
        if (
            not isinstance(declared_latched_ids, Sequence)
            or isinstance(declared_latched_ids, (str, bytes, bytearray))
            or list(declared_latched_ids) != latched_source_ids
            or reported_402 is not bool(latched_source_ids)
        ):
            errors.append("cost_summary HTTP 402 latch aggregation is inconsistent")
        for index in range(1, min(len(stage_rows), len(parsed_windows))):
            previous = stage_rows[index - 1]
            current = stage_rows[index]
            previous_cumulative = _number(
                previous.get("cumulative_conservative_exposure_usd")
            )
            current_prior = _number(current.get("prior_conservative_exposure_usd"))
            if (
                previous_cumulative is None
                or current_prior is None
                or not math.isclose(
                    previous_cumulative,
                    current_prior,
                    rel_tol=0,
                    abs_tol=1e-9,
                )
            ):
                errors.append("cost_summary source stages are not exposure-contiguous")
            if parsed_windows[index][0] < parsed_windows[index - 1][1]:
                errors.append("cost_summary source stage windows overlap")
        if stage_rows:
            final_stage = stage_rows[-1]
            final_source_id = str(final_stage.get("source_id") or "")
            final_exposure = _number(
                final_stage.get("cumulative_conservative_exposure_usd")
            )
            final_schema = _text(final_stage.get("summary_schema_version"))
            final_sha = str(final_stage.get("summary_sha256") or "")
            final_cap = _number(final_stage.get("cost_cap_usd"))
            if cost_summary.get("conservative_exposure_source_id") != final_source_id:
                errors.append(
                    "cost_summary selected exposure source is not final stage"
                )
            if (
                conservative is None
                or final_exposure is None
                or not math.isclose(
                    conservative, final_exposure, rel_tol=0, abs_tol=1e-9
                )
            ):
                errors.append("cost_summary exposure disagrees with final stage")
            if (
                cost_summary.get("conservative_exposure_receipt_schema_version")
                != final_schema
                or cost_summary.get("conservative_exposure_receipt_sha256") != final_sha
            ):
                errors.append("cost_summary terminal receipt identity is inconsistent")
            if (
                cost_cap is None
                or final_cap is None
                or not math.isclose(cost_cap, final_cap, rel_tol=0, abs_tol=1e-9)
            ):
                errors.append("cost_summary cap disagrees with final stage")

    methodology = _mapping(analysis.get("statistical_methodology"))
    if not math.isclose(
        float(_number(methodology.get("confidence_level")) or -1),
        0.95,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        errors.append("statistical_methodology.confidence_level must be 0.95")
    if methodology.get("serial_sampling_unit") != "request_id":
        errors.append("statistical_methodology.serial_sampling_unit must be request_id")
    if methodology.get("load_sampling_unit") != "epoch_id":
        errors.append("statistical_methodology.load_sampling_unit must be epoch_id")
    if _integer(methodology.get("bootstrap_replicates")) in {None, 0}:
        errors.append("statistical_methodology.bootstrap_replicates must be positive")

    workload_rows = _as_analysis_rows(analysis.get("workload_summaries"))
    for index, row in enumerate(workload_rows):
        pointer = f"workload_summaries[{index}]"
        request_count = _integer(row.get("request_count"))
        epoch_count = _integer(row.get("epoch_count"))
        if request_count is None or epoch_count is None:
            errors.append(f"{pointer}: request_count and epoch_count must be integers")
            continue
        expected_unit = "epoch_id" if epoch_count else "request_id"
        if row.get("sampling_unit") != expected_unit:
            errors.append(f"{pointer}.sampling_unit must be {expected_unit}")
        _validate_ci(
            row.get("transport_success_rate_ci95"),
            pointer=f"{pointer}.transport_success_rate_ci95",
            expected_unit=expected_unit,
            require_estimate=(epoch_count if epoch_count else request_count) > 0,
            point_estimate=row.get("transport_success_rate"),
            errors=errors,
        )
        quality_count = _integer(row.get("quality_scored_count")) or 0
        _validate_ci(
            row.get("quality_pass_rate_ci95"),
            pointer=f"{pointer}.quality_pass_rate_ci95",
            expected_unit="request_id",
            require_estimate=quality_count > 0,
            point_estimate=row.get("quality_pass_rate"),
            errors=errors,
        )
        for ci_key, point_key, metric_unit in (
            ("ttft_p50_ci95", "ttft_p50_seconds", expected_unit),
            ("latency_p50_ci95", "latency_p50_seconds", expected_unit),
            (
                "post_ttft_output_tps_proxy_p50_ci95",
                "post_ttft_output_tps_proxy_p50",
                "request_id",
            ),
            (
                "aggregate_output_goodput_tps_epoch_p50_ci95",
                "aggregate_output_goodput_tps_epoch_p50",
                "epoch_id",
            ),
        ):
            point_value = row.get(point_key)
            _validate_ci(
                row.get(ci_key),
                pointer=f"{pointer}.{ci_key}",
                expected_unit=metric_unit,
                require_estimate=_number(point_value) is not None,
                point_estimate=point_value,
                errors=errors,
            )
        proxy_count = _integer(row.get("post_ttft_output_tps_proxy_observation_count"))
        if proxy_count is None or proxy_count < 0:
            errors.append(
                f"{pointer}.post_ttft_output_tps_proxy_observation_count: "
                "must be a non-negative integer"
            )
        elif (_number(row.get("post_ttft_output_tps_proxy_p50")) is not None) != (
            proxy_count > 0
        ):
            errors.append(
                f"{pointer}: post-TTFT proxy estimate and observation count disagree"
            )
        aggregate_count = _integer(
            row.get("aggregate_output_goodput_epoch_observation_count")
        )
        if aggregate_count is None or aggregate_count < 0:
            errors.append(
                f"{pointer}.aggregate_output_goodput_epoch_observation_count: "
                "must be a non-negative integer"
            )
        elif (
            _number(row.get("aggregate_output_goodput_tps_epoch_p50")) is not None
        ) != (aggregate_count > 0):
            errors.append(
                f"{pointer}: aggregate epoch-goodput estimate and observation count "
                "disagree"
            )
        for key in (
            "offered_rpm_max",
            "achieved_rpm",
            "goodput_rpm",
            "effective_input_tpm",
            "effective_output_tpm",
            "quality_adjusted_output_tpm_scored_only",
            "ttft_p50_seconds",
            "latency_p50_seconds",
            "post_ttft_output_tps_proxy_p50",
            "aggregate_output_goodput_tps_epoch_p50",
            "estimated_cost_usd",
        ):
            value = row.get(key)
            parsed = _number(value)
            if value is not None and (parsed is None or parsed < 0):
                errors.append(f"{pointer}.{key}: must be a finite non-negative value")

    capacity_rows = _as_analysis_rows(analysis.get("capacity_summaries"))
    for index, row in enumerate(capacity_rows):
        pointer = f"capacity_summaries[{index}]"
        if "recommended_headroom_rps" in row or "recommended_rpm" in row:
            errors.append(f"{pointer}: automatic headroom/recommended RPM is forbidden")
        if "sustainable_confirmed_rps" in row or "sustainable_rpm" in row:
            errors.append(
                f"{pointer}: AIMD confirmations must not be labelled sustainable"
            )
        confirmed = _number(row.get("confirmed_healthy_offered_rps"))
        confirmed_rpm = _number(row.get("confirmed_healthy_offered_rpm"))
        upper = _number(row.get("capacity_upper_bound_rps"))
        right_censored = row.get("right_censored")
        confirmations = (
            _integer(row.get("candidate_rate_confirmation_epoch_count")) or 0
        )
        if confirmed is None:
            if confirmed_rpm is not None or upper is not None or right_censored is True:
                errors.append(
                    f"{pointer}: unconfirmed row cannot report a capacity bound"
                )
        else:
            if confirmations < 3:
                errors.append(
                    f"{pointer}: confirmed capacity requires three confirmations"
                )
            if confirmed_rpm is None or not math.isclose(
                confirmed_rpm, confirmed * 60, rel_tol=0, abs_tol=1e-9
            ):
                errors.append(
                    f"{pointer}: confirmed healthy offered RPM must equal RPS x 60"
                )
            if right_censored is True and upper is not None:
                errors.append(
                    f"{pointer}: right-censored capacity cannot have an upper bound"
                )
            if right_censored is False and (upper is None or upper <= confirmed):
                errors.append(
                    f"{pointer}: bracketed capacity needs a strict upper bound"
                )
            if _number(row.get("achieved_rpm")) is None:
                errors.append(f"{pointer}: confirmed capacity requires achieved_rpm")
        for key, point_key in _CAPACITY_CI_FIELDS.items():
            _validate_ci(
                row.get(key),
                pointer=f"{pointer}.{key}",
                expected_unit="epoch_id",
                require_estimate=_number(row.get(point_key)) is not None,
                point_estimate=row.get(point_key),
                errors=errors,
            )

    has_soak_source = any(
        row.get("source_kind") == "direct_soak"
        for row in _as_analysis_rows(analysis.get("data_sources"))
    )
    soak_rows = _as_analysis_rows(analysis.get("soak_summaries"))
    soak_block_rows = _as_analysis_rows(analysis.get("soak_block_summaries"))
    soak_quality_rows = _as_analysis_rows(analysis.get("soak_quality_summaries"))
    soak_recovery_rows = _as_analysis_rows(analysis.get("soak_recovery_summaries"))
    if has_soak_source:
        if methodology.get("soak_sampling_unit") != "analysis_block_id":
            errors.append(
                "statistical_methodology.soak_sampling_unit must be analysis_block_id"
            )
        if methodology.get("soak_quality_sampling_unit") != "quality_pair_id":
            errors.append(
                "statistical_methodology.soak_quality_sampling_unit must be quality_pair_id"
            )
        if methodology.get("soak_recovery_sampling_unit") != "phase_id":
            errors.append(
                "statistical_methodology.soak_recovery_sampling_unit must be phase_id"
            )
        if (
            methodology.get("soak_recovery_within_phase_binomial_sampling_unit")
            != "request_id"
        ):
            errors.append(
                "statistical_methodology soak recovery request intervals must use request_id"
            )
        if not soak_rows or not soak_block_rows:
            errors.append(
                "direct soak source requires cell and analysis-block summaries"
            )
    block_keys: list[tuple[str, str]] = []
    for index, row in enumerate(soak_block_rows):
        pointer = f"soak_block_summaries[{index}]"
        if row.get("schema_version") != "digitalocean_public_soak_block_summary_v1":
            errors.append(f"{pointer}.schema_version is invalid")
        if row.get("sampling_unit") != "analysis_block_id":
            errors.append(f"{pointer}.sampling_unit must be analysis_block_id")
        block_id = _text(row.get("analysis_block_id"))
        source_id = _text(row.get("source_id"))
        if block_id is None or source_id is None:
            errors.append(f"{pointer}: source_id and analysis_block_id are required")
        else:
            block_keys.append((source_id, block_id))
        drain_elapsed = _number(
            row.get("arrival_cohort_elapsed_seconds_including_drain")
        )
        drain_rpm = _number(row.get("arrival_cohort_successful_rpm_including_drain"))
        if drain_elapsed is None or drain_elapsed <= 0 or drain_rpm is None:
            errors.append(
                f"{pointer}: drain-inclusive cohort elapsed time and RPM are required"
            )
    if len(block_keys) != len(set(block_keys)):
        errors.append("soak_block_summaries contains duplicate analysis_block_id units")
    for index, row in enumerate(soak_rows):
        pointer = f"soak_summaries[{index}]"
        if row.get("schema_version") != "digitalocean_public_soak_cell_summary_v1":
            errors.append(f"{pointer}.schema_version is invalid")
        if row.get("block_ci_sampling_unit") != "analysis_block_id":
            errors.append(f"{pointer}.block_ci_sampling_unit must be analysis_block_id")
        if row.get("paired_quality_ci_sampling_unit") != "quality_pair_id":
            errors.append(
                f"{pointer}.paired_quality_ci_sampling_unit must be quality_pair_id"
            )
        claim = row.get("capacity_claim")
        if claim not in {
            "exact_two_minute_soak_pass",
            "exact_two_minute_soak_measured_fail",
            "not_soak_verified",
        }:
            errors.append(
                f"{pointer}.capacity_claim must remain a two-minute observation"
            )
        if any(
            key in row for key in ("sustainable_confirmed_rps", "aimd_confirmed_rps")
        ):
            errors.append(
                f"{pointer}: soak evidence cannot be relabelled as AIMD confirmation"
            )
        block_count = _integer(row.get("soak_block_count"))
        if row.get("scientifically_complete") is True and block_count != 4:
            errors.append(f"{pointer}: complete soak requires four analysis blocks")
        if row.get("scientifically_complete") is True:
            if row.get("headline_goodput_denominator") != (
                "arrival_cohort_elapsed_seconds_including_drain"
            ):
                errors.append(
                    f"{pointer}: headline soak goodput must be drain-inclusive"
                )
            if (
                _number(
                    row.get("arrival_cohort_successful_rpm_including_drain_block_mean")
                )
                is None
            ):
                errors.append(f"{pointer}: drain-inclusive soak RPM mean is required")
        if row.get("scientifically_complete") is True and claim == "not_soak_verified":
            errors.append(
                f"{pointer}: complete soak must report its measured pass/fail"
            )
        if (
            row.get("scientifically_complete") is not True
            and claim != "not_soak_verified"
        ):
            errors.append(
                f"{pointer}: incomplete soak cannot claim a measured pass/fail"
            )
    pair_keys: list[tuple[str, str]] = []
    for index, row in enumerate(soak_quality_rows):
        pointer = f"soak_quality_summaries[{index}]"
        if row.get("sampling_unit") != "quality_pair_id":
            errors.append(f"{pointer}.sampling_unit must be quality_pair_id")
        pair_id = _text(row.get("quality_pair_id"))
        source_id = _text(row.get("source_id"))
        if pair_id is None or source_id is None:
            errors.append(f"{pointer}: source_id and quality_pair_id are required")
        else:
            pair_keys.append((source_id, pair_id))
        if row.get("exact_request_payload_hash_match") is not True:
            errors.append(f"{pointer}: exact payload hash match is required")
    if len(pair_keys) != len(set(pair_keys)):
        errors.append("soak_quality_summaries contains duplicate quality_pair_id units")
    for index, row in enumerate(soak_recovery_rows):
        if row.get("sampling_unit") != "phase_id":
            errors.append(
                f"soak_recovery_summaries[{index}].sampling_unit must be phase_id"
            )
        if row.get("within_phase_binomial_interval_sampling_unit") != "request_id":
            errors.append(
                f"soak_recovery_summaries[{index}] request Wilson intervals must use request_id"
            )

    coverage = _mapping(analysis.get("coverage_summary"))
    matrix_rows = _as_analysis_rows(analysis.get("coverage_matrix"))
    expected_pairs = [
        (endpoint, dimension)
        for endpoint in EXPECTED_ENDPOINT_IDS
        for dimension in REQUIRED_COVERAGE_DIMENSIONS
    ]
    actual_pairs = [
        (
            str(row.get("endpoint_id") or ""),
            str(row.get("coverage_dimension") or ""),
        )
        for row in matrix_rows
    ]
    if actual_pairs != expected_pairs:
        errors.append(
            "coverage_matrix must contain each frozen endpoint/dimension pair exactly once in order"
        )
    allowed_coverage_statuses = {
        "completed",
        "unsupported",
        "operational_failure",
        "inconclusive",
        "skipped",
        "untested",
    }
    for index, row in enumerate(matrix_rows):
        status = str(row.get("status") or "")
        if status not in allowed_coverage_statuses:
            errors.append(f"coverage_matrix[{index}].status is invalid")
        for key in (
            "planned_cell_or_epoch_count",
            "observed_attempt_count",
            "completed_subcell_count",
            "unsupported_subcell_count",
            "operational_failure_subcell_count",
            "inconclusive_subcell_count",
            "skipped_subcell_count",
            "superseded_subcell_count",
            "replicate_failure_subcell_count",
        ):
            if _integer(row.get(key)) is None:
                errors.append(
                    f"coverage_matrix[{index}].{key} must be a non-negative integer"
                )
        if _integer(row.get("explicit_untested_subtest_count")) is None:
            errors.append(
                f"coverage_matrix[{index}].explicit_untested_subtest_count must be a non-negative integer"
            )
        if not isinstance(row.get("has_explicit_scope_exclusions"), bool):
            errors.append(
                f"coverage_matrix[{index}].has_explicit_scope_exclusions must be boolean"
            )
    matrix_status_counts = Counter(str(row.get("status") or "") for row in matrix_rows)
    matrix_completed = sum(
        status in {"completed", "unsupported", "operational_failure"}
        for status in (str(row.get("status") or "") for row in matrix_rows)
    )
    required = _integer(coverage.get("required_endpoint_dimension_cells"))
    completed = _integer(coverage.get("completed_or_evidence_backed_unsupported_cells"))
    resolved = _integer(coverage.get("resolved_experiment_cells"))
    fraction = _number(coverage.get("coverage_fraction"))
    frozen_required = len(expected_pairs)
    if (
        _integer(coverage.get("required_endpoint_count")) != len(EXPECTED_ENDPOINT_IDS)
        or _integer(coverage.get("required_dimension_count"))
        != len(REQUIRED_COVERAGE_DIMENSIONS)
        or required != frozen_required
    ):
        errors.append(
            "coverage_summary must declare the frozen hosted endpoint/dimension matrix"
        )
    if completed is None or required in {None, 0}:
        errors.append("coverage_summary completed/required counts are invalid")
    elif completed != matrix_completed:
        errors.append("coverage_summary completed count disagrees with coverage_matrix")
    elif resolved != matrix_completed:
        errors.append("coverage_summary resolved count disagrees with coverage_matrix")
    elif fraction is None or not math.isclose(
        fraction, matrix_completed / frozen_required, rel_tol=0, abs_tol=1e-12
    ):
        errors.append(
            "coverage_summary.coverage_fraction disagrees with coverage_matrix"
        )
    elif bool(coverage.get("is_100_percent")) != (matrix_completed == frozen_required):
        errors.append("coverage_summary.is_100_percent disagrees with coverage_matrix")
    declared_status_counts = coverage.get("status_counts")
    if not isinstance(declared_status_counts, Mapping) or {
        str(key): _integer(value) for key, value in declared_status_counts.items()
    } != dict(sorted(matrix_status_counts.items())):
        errors.append("coverage_summary.status_counts disagrees with coverage_matrix")

    raw_scope_exclusions = analysis.get("scope_exclusions", ())
    if not isinstance(raw_scope_exclusions, Sequence) or isinstance(
        raw_scope_exclusions, (str, bytes, bytearray)
    ):
        errors.append("scope_exclusions must be a list")
        scope_exclusion_rows: list[Mapping[str, Any]] = []
    else:
        scope_exclusion_rows = _as_analysis_rows(raw_scope_exclusions)
        if len(scope_exclusion_rows) != len(raw_scope_exclusions):
            errors.append("scope_exclusions entries must be objects")
    scope_keys: list[tuple[str, str, str]] = []
    for index, row in enumerate(scope_exclusion_rows):
        pointer = f"scope_exclusions[{index}]"
        endpoint = str(row.get("endpoint_id") or "")
        exclusion_id = str(row.get("scope_exclusion_id") or "")
        dimension = str(row.get("coverage_dimension") or "")
        source_id = str(row.get("source_id") or "")
        if row.get("schema_version") != SCOPE_EXCLUSION_SCHEMA:
            errors.append(f"{pointer}.schema_version is invalid")
        if endpoint not in EXPECTED_ENDPOINT_SET:
            errors.append(f"{pointer}.endpoint_id is outside the frozen inventory")
        if not exclusion_id or not source_id:
            errors.append(f"{pointer}: source and exclusion IDs are required")
        if dimension not in REQUIRED_COVERAGE_DIMENSIONS:
            errors.append(f"{pointer}.coverage_dimension is invalid")
        if row.get("status") != "untested":
            errors.append(f"{pointer}.status must remain untested")
        if row.get("claim_policy") != "explicitly_excluded_not_tested":
            errors.append(f"{pointer}.claim_policy is invalid")
        if (
            _text(row.get("measurement_label")) is None
            or _text(row.get("reason")) is None
        ):
            errors.append(f"{pointer}: measurement label and reason are required")
        manifest_hash = str(row.get("source_manifest_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", manifest_hash):
            errors.append(f"{pointer}.source_manifest_sha256 is invalid")
        scope_keys.append((source_id, endpoint, exclusion_id))
    if len(scope_keys) != len(set(scope_keys)):
        errors.append(
            "scope_exclusions contains duplicate source/endpoint/exclusion rows"
        )
    matrix_by_pair = {
        (
            str(row.get("endpoint_id") or ""),
            str(row.get("coverage_dimension") or ""),
        ): row
        for row in matrix_rows
    }
    exclusions_by_pair = Counter(
        (
            str(row.get("endpoint_id") or ""),
            str(row.get("coverage_dimension") or ""),
        )
        for row in scope_exclusion_rows
    )
    for pair, matrix_row in matrix_by_pair.items():
        expected_count = exclusions_by_pair[pair]
        if (
            _integer(matrix_row.get("explicit_untested_subtest_count"))
            != expected_count
        ):
            errors.append(
                "coverage_matrix explicit scope-exclusion count disagrees with ledger"
            )
        if matrix_row.get("has_explicit_scope_exclusions") is not bool(expected_count):
            errors.append(
                "coverage_matrix explicit scope-exclusion flag disagrees with ledger"
            )
        if (
            expected_count
            and _integer(matrix_row.get("planned_cell_or_epoch_count")) == 0
            and matrix_row.get("status") != "untested"
        ):
            errors.append("exclusion-only coverage dimension must remain untested")

    reconciliation = _mapping(analysis.get("request_reconciliation"))
    if require_complete and reconciliation.get("all_requests_reconciled") is not True:
        errors.append(
            "final build requires every request to reconcile to frozen evidence"
        )
    if require_complete and (
        coverage.get("is_100_percent") is not True
        or matrix_completed != frozen_required
    ):
        errors.append("final build requires 100% completed/evidence-backed coverage")
    if errors:
        raise DirectReportError("public analysis contract failed: " + "; ".join(errors))
    return {
        "schema_version": "digitalocean_direct_public_contract_gate_v1",
        "passed": True,
        "require_complete": require_complete,
        "coverage_complete": coverage.get("is_100_percent") is True,
        "orphan_free": reconciliation.get("all_requests_reconciled") is True,
        "workload_summary_count": len(workload_rows),
        "capacity_summary_count": len(capacity_rows),
        "scope_exclusion_count": len(scope_exclusion_rows),
    }


def _as_analysis_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def build_metric_audit(
    requests: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a traceable, no-trimming audit of legacy and corrected timing rates."""

    rows: list[dict[str, Any]] = []
    legacy_values: list[float] = []
    corrected_values: list[float] = []
    for request in requests:
        legacy = _number(
            request.get("legacy_sse_chunk_span_output_tokens_per_second_proxy")
        )
        corrected = _number(request.get("post_ttft_output_tokens_per_second_proxy"))
        if legacy is None and corrected is None and request.get("ttft_seconds") is None:
            continue
        if legacy is not None:
            legacy_values.append(legacy)
        if corrected is not None:
            corrected_values.append(corrected)
        reasons = request.get("timing_metric_invalidity_reasons")
        if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes)):
            reasons = ()
        rows.append(
            {
                "request_id": request.get("request_id"),
                "endpoint_id": request.get("endpoint_id"),
                "source_kind": request.get("source_kind"),
                "workload": request.get("workload"),
                "stream_mode": request.get("stream_mode"),
                "choice_count": request.get("choice_count"),
                "input_tokens": request.get("input_tokens"),
                "output_tokens": request.get("output_tokens"),
                "request_seconds": request.get("request_seconds"),
                "streamed_ttft_seconds": request.get("ttft_seconds"),
                "sse_content_event_count": request.get("stream_event_count"),
                "legacy_sse_chunk_span_seconds": request.get("generation_seconds"),
                "legacy_sse_chunk_span_output_tps_proxy": legacy,
                "post_ttft_output_tps_proxy": corrected,
                "corrected_to_legacy_ratio": (
                    corrected / legacy
                    if corrected is not None and legacy not in {None, 0}
                    else None
                ),
                "cache_state": request.get("cache_state"),
                "classification": request.get("timing_metric_audit_classification"),
                "invalidity_reasons": ";".join(sorted(str(v) for v in reasons)),
                "trimmed_or_winsorized": False,
            }
        )

    def q(values: Sequence[float], probability: float) -> float | None:
        return nearest_rank(values, probability) if values else None

    summary = {
        "request_rows": len(requests),
        "audit_rows": len(rows),
        "legacy_sse_proxy_observations": len(legacy_values),
        "legacy_sse_proxy_at_least_1000": sum(v >= 1_000 for v in legacy_values),
        "legacy_sse_proxy_at_least_10000": sum(v >= 10_000 for v in legacy_values),
        "legacy_sse_proxy_at_least_100000": sum(v >= 100_000 for v in legacy_values),
        "legacy_sse_proxy_max": max(legacy_values, default=None),
        "corrected_post_ttft_proxy_observations": len(corrected_values),
        "corrected_post_ttft_proxy_median": q(corrected_values, 0.50),
        "corrected_post_ttft_proxy_p99_exploratory": q(corrected_values, 0.99),
        "corrected_post_ttft_proxy_p999_exploratory": q(corrected_values, 0.999),
        "corrected_post_ttft_proxy_max": max(corrected_values, default=None),
        "buffered_nonstream_ttft_censored": sum(
            row.get("stream_mode") == "buffered_nonstream" for row in requests
        ),
        "multi_choice_per_sequence_excluded": sum(
            bool(row.get("multi_choice")) for row in requests
        ),
        "sub_100ms_post_ttft_intervals_censored": sum(
            "post_ttft_interval_below_100ms_unstable_rate"
            in set(row.get("timing_metric_invalidity_reasons") or ())
            for row in requests
        ),
        "event_count_differs_from_completion_tokens": sum(
            _integer(row.get("stream_event_count")) is not None
            and _integer(row.get("output_tokens")) is not None
            and _integer(row.get("stream_event_count"))
            != _integer(row.get("output_tokens"))
            for row in requests
        ),
        "policy": (
            "No observation is silently trimmed or winsorized. The legacy SSE-event "
            "span is audit-only. Qualified corrected extremes remain visible; "
            "unobservable or timing-unstable metrics are null with a reason, while "
            "their billed tokens remain in aggregate goodput."
        ),
    }
    return rows, summary


def build_cache_state_summaries(
    requests: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_replicates: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in requests:
        if row.get("multi_choice"):
            continue
        groups[
            (
                str(row.get("endpoint_id")),
                str(row.get("workload") or "unspecified"),
                str(row.get("cache_state") or "not_reported_unknown"),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (endpoint, workload, cache_state), members in sorted(groups.items()):
        ttft = [
            float(value)
            for row in members
            if (value := _number(row.get("ttft_seconds"))) is not None
        ]
        prefill = [
            float(value)
            for row in members
            if (value := _number(row.get("prefill_proxy_tokens_per_second")))
            is not None
        ]
        latency = [
            float(value)
            for row in members
            if (value := _number(row.get("request_seconds"))) is not None
        ]
        ttft_ci = _bootstrap_metric(
            ttft,
            seed=deterministic_seed(seed, endpoint, workload, cache_state, "ttft"),
            replicates=bootstrap_replicates,
        )
        ttft_ci["sampling_unit"] = "request_id"
        output.append(
            {
                "endpoint_id": endpoint,
                "workload": workload,
                "cache_state": cache_state,
                "request_count": len(members),
                "ttft_observation_count": len(ttft),
                "ttft_p50_seconds": nearest_rank(ttft, 0.50),
                "ttft_p95_seconds": nearest_rank(ttft, 0.95),
                "ttft_p50_ci95": ttft_ci,
                "latency_observation_count": len(latency),
                "latency_p50_seconds": nearest_rank(latency, 0.50),
                "prefill_proxy_observation_count": len(prefill),
                "prefill_proxy_p50_tokens_per_second": nearest_rank(prefill, 0.50),
                "prefill_headline_eligible": (
                    cache_state == "cache_miss_observed" and bool(prefill)
                ),
                "sampling_unit": "request_id",
            }
        )
    return output


def build_capability_evidence(
    requests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Separate transport support, task correctness, and validation behavior."""

    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in requests:
        dimension = _text(row.get("capability_dimension"))
        if dimension is None:
            continue
        groups[
            (
                str(row.get("endpoint_id")),
                str(row.get("workload") or "capability"),
                dimension,
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (endpoint, workload, dimension), members in sorted(groups.items()):
        validation = [row for row in members if row.get("malformed_validation_probe")]
        valid = [row for row in members if not row.get("malformed_validation_probe")]
        valid_attempted = [
            row for row in valid if row.get("provider_send_attempted") is not False
        ]
        valid_2xx = [row for row in valid_attempted if row.get("transport_success")]
        scored = [row for row in valid_2xx if row.get("quality_scored")]
        functional_passes = sum(bool(row.get("functional_valid")) for row in scored)
        documented_unavailable = sum(
            str(row.get("coverage_classification")) == "documented_unavailable"
            for row in valid
        )
        validation_attempted = [
            row for row in validation if row.get("provider_send_attempted") is not False
        ]
        validation_4xx = sum(
            (status := _integer(row.get("http_status"))) is not None
            and 400 <= status < 500
            for row in validation_attempted
        )
        validation_2xx = sum(
            bool(row.get("transport_success")) for row in validation_attempted
        )
        if valid_2xx:
            transport_status = "observed_supported"
        elif documented_unavailable and not valid_attempted:
            transport_status = "documented_unavailable"
        elif valid_attempted:
            transport_status = "observed_transport_degraded"
        else:
            transport_status = "inconclusive"
        if not scored:
            functional_status = "not_scored"
        elif functional_passes == len(scored):
            functional_status = "passed"
        elif functional_passes == 0:
            functional_status = "failed"
        else:
            functional_status = "degraded"
        if not validation_attempted:
            validation_status = "not_tested"
        elif validation_4xx == len(validation_attempted):
            validation_status = "correct_rejection_observed"
        elif validation_2xx:
            validation_status = "invalid_input_accepted"
        elif validation_4xx:
            validation_status = "inconsistent"
        else:
            validation_status = "inconclusive"
        transport_ci = wilson_interval(len(valid_2xx), len(valid_attempted))
        functional_ci = wilson_interval(functional_passes, len(scored))
        output.append(
            {
                "endpoint_id": endpoint,
                "workload": workload,
                "capability_dimension": dimension,
                "transport_status": transport_status,
                "valid_probe_attempt_count": len(valid_attempted),
                "valid_probe_2xx_count": len(valid_2xx),
                "valid_probe_2xx_rate_ci95": transport_ci,
                "functional_status": functional_status,
                "functional_scored_count": len(scored),
                "functional_pass_count": functional_passes,
                "functional_pass_rate_ci95": functional_ci,
                "malformed_validation_status": validation_status,
                "malformed_validation_attempt_count": len(validation_attempted),
                "malformed_validation_4xx_count": validation_4xx,
                "malformed_validation_2xx_count": validation_2xx,
                "documented_unavailable_count": documented_unavailable,
                "sampling_unit": "request_id",
            }
        )
    return output


def build_observed_limits(
    requests: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    inventory_by_endpoint = {str(row.get("endpoint_id")): row for row in inventory}
    for endpoint in EXPECTED_ENDPOINT_IDS:
        documented = inventory_by_endpoint.get(endpoint, {})
        raw_capabilities = documented.get("documented_capabilities")
        if isinstance(raw_capabilities, str):
            try:
                documented_capabilities = _mapping(json.loads(raw_capabilities))
            except ValueError:
                documented_capabilities = {}
        else:
            documented_capabilities = _mapping(raw_capabilities)
        rows = [row for row in requests if row.get("endpoint_id") == endpoint]
        context_rows = [
            row
            for row in rows
            if any(
                word in str(row.get("workload", "")).casefold()
                for word in ("context", "retrieval")
            )
        ]
        combined_context_rows = [
            row
            for row in context_rows
            if any("combined" in tag.casefold() for tag in _coverage_tags(row))
        ]
        prompt_context_rows = [
            row for row in context_rows if row not in combined_context_rows
        ]
        output_rows = [
            row
            for row in rows
            if any(
                word in str(row.get("workload", "")).casefold()
                for word in ("output", "decode")
            )
        ]
        accepted_output = [row for row in output_rows if row.get("transport_success")]
        rejected_output = [
            row for row in output_rows if _row_evidence_backed_unsupported(row)
        ]
        observed_capabilities: dict[str, str] = {}
        for capability, markers in {
            "vision": ("vision", "image"),
            "tool_calling": ("tool",),
            "structured_output": ("structured", "json"),
        }.items():
            relevant = [
                row
                for row in rows
                if any(
                    marker in str(row.get("workload", "")).casefold()
                    for marker in markers
                )
            ]
            if any(
                row.get("transport_success") and row.get("goodput_success")
                for row in relevant
            ):
                observed_capabilities[capability] = "observed_functional"
            elif any(_row_evidence_backed_unsupported(row) for row in relevant):
                observed_capabilities[capability] = "observed_rejected_or_unsupported"
            elif relevant:
                observed_capabilities[capability] = "inconclusive"
            else:
                observed_capabilities[capability] = "untested"
        maximum_requested_output = max(
            (
                int(row["requested_output_target"])
                for row in accepted_output
                if row.get("requested_output_target") is not None
            ),
            default=None,
        )
        maximum_realized_output = max(
            (
                int(row["output_tokens"])
                for row in accepted_output
                if row.get("output_tokens") is not None
            ),
            default=None,
        )
        minimum_rejected_output = min(
            (
                int(row["requested_output_target"])
                for row in rejected_output
                if row.get("requested_output_target") is not None
            ),
            default=None,
        )

        def context_limit_row(
            lane_rows: Sequence[Mapping[str, Any]],
            *,
            dimension: str,
            requested_output_target: int | None,
        ) -> dict[str, Any]:
            accepted = [row for row in lane_rows if row.get("transport_success")]
            valid = [
                row
                for row in accepted
                if row.get("functional_valid") is True or row.get("goodput_success")
            ]
            rejected = [
                row
                for row in lane_rows
                if row.get("coverage_classification")
                == "explicit_context_limit_rejection"
            ]
            maximum_accepted_input = max(
                (
                    int(row["input_tokens"])
                    for row in accepted
                    if row.get("input_tokens") is not None
                ),
                default=None,
            )
            maximum_valid_input = max(
                (
                    int(row["input_tokens"])
                    for row in valid
                    if row.get("input_tokens") is not None
                ),
                default=None,
            )
            minimum_rejected_input = min(
                (
                    int(row["requested_input_tokens"])
                    for row in rejected
                    if row.get("requested_input_tokens") is not None
                ),
                default=None,
            )
            maximum_accepted_estimated_input = max(
                (
                    int(row["requested_input_tokens"])
                    for row in accepted
                    if row.get("requested_input_tokens") is not None
                ),
                default=None,
            )
            nonmonotonic = bool(
                minimum_rejected_input is not None
                and (
                    (
                        maximum_accepted_estimated_input is not None
                        and maximum_accepted_estimated_input >= minimum_rejected_input
                    )
                    or (
                        maximum_accepted_input is not None
                        and maximum_accepted_input >= minimum_rejected_input
                    )
                )
            )
            maximum_accepted_combined = (
                maximum_accepted_input + requested_output_target
                if maximum_accepted_input is not None
                and requested_output_target is not None
                else None
            )
            maximum_valid_combined = (
                maximum_valid_input + requested_output_target
                if maximum_valid_input is not None
                and requested_output_target is not None
                else None
            )
            minimum_rejected_combined = (
                minimum_rejected_input + requested_output_target
                if minimum_rejected_input is not None
                and requested_output_target is not None
                else None
            )
            has_accepted_lower_bound = maximum_accepted_input is not None
            has_rejected_upper_bound = minimum_rejected_input is not None
            if nonmonotonic:
                boundary_censoring = "nonmonotonic_inconclusive"
            elif has_accepted_lower_bound and has_rejected_upper_bound:
                boundary_censoring = "interval_censored"
            elif has_accepted_lower_bound:
                boundary_censoring = "right_censored"
            elif has_rejected_upper_bound:
                boundary_censoring = "left_censored"
            else:
                boundary_censoring = "unobserved"
            anchor_sources = sorted(
                {
                    str(row["context_window_anchor_source"])
                    for row in lane_rows
                    if row.get("context_window_anchor_source")
                }
            )
            kimi_undocumented_anchor = endpoint == KIMI_ENDPOINT_ID
            undocumented_anchor = (
                kimi_undocumented_anchor
                or "undocumented_probe_anchor" in anchor_sources
            )
            reported_anchor_source: str | list[str]
            if kimi_undocumented_anchor:
                reported_anchor_source = "undocumented_probe_anchor"
            else:
                reported_anchor_source = (
                    anchor_sources[0] if len(anchor_sources) == 1 else anchor_sources
                )
            combined = dimension == "combined prompt + requested output"
            return {
                "endpoint_id": endpoint,
                "dimension": dimension,
                "documented_value": (
                    None if undocumented_anchor else documented.get("context_window")
                ),
                "context_window_anchor_source": reported_anchor_source,
                "context_window_probe_anchor_value": (
                    KIMI_UNDOCUMENTED_CONTEXT_PROBE_ANCHOR
                    if kimi_undocumented_anchor
                    else None
                ),
                "documentation_status": (
                    "undocumented_probe_anchor_value"
                    if undocumented_anchor
                    else "documented_or_inventory_unavailable"
                ),
                "observed_value": (
                    None
                    if nonmonotonic
                    else (
                        maximum_accepted_combined
                        if combined
                        else maximum_accepted_input
                    )
                ),
                "finding": (
                    "nonmonotonic_context_outcomes_inconclusive"
                    if nonmonotonic
                    else (
                        "accepted_and_retrieval_valid_through_"
                        + str(
                            maximum_valid_combined if combined else maximum_valid_input
                        )
                        if maximum_valid_input is not None
                        else (
                            "transport_accepted_but_retrieval_not_verified"
                            if maximum_accepted_input is not None
                            else "untested_or_inconclusive"
                        )
                    )
                ),
                "requested_output_target": requested_output_target,
                "requested_output_target_unit": (
                    "tokens" if requested_output_target is not None else None
                ),
                "maximum_accepted_input_tokens": maximum_accepted_input,
                "maximum_accepted_estimated_input_tokens": (
                    maximum_accepted_estimated_input
                ),
                "maximum_functionally_valid_input_tokens": maximum_valid_input,
                "minimum_rejected_estimated_input_tokens": minimum_rejected_input,
                "maximum_accepted_combined_target_tokens": maximum_accepted_combined,
                "maximum_functionally_valid_combined_target_tokens": maximum_valid_combined,
                "minimum_rejected_estimated_combined_target_tokens": minimum_rejected_combined,
                "boundary_censoring": boundary_censoring,
                "boundary_interval_censored": (
                    boundary_censoring == "interval_censored"
                ),
                "boundary_exact": False,
                "boundary_monotonic": not nonmonotonic,
            }

        prompt_targets = {
            int(row["requested_output_target"])
            for row in prompt_context_rows
            if row.get("requested_output_target") is not None
        }
        prompt_output_target = (
            next(iter(prompt_targets)) if len(prompt_targets) == 1 else None
        )
        output.append(
            context_limit_row(
                prompt_context_rows,
                dimension="prompt context window",
                requested_output_target=prompt_output_target,
            )
        )
        combined_targets = sorted(
            {
                int(row["requested_output_target"])
                for row in combined_context_rows
                if row.get("requested_output_target") is not None
            }
        )
        for target in combined_targets:
            output.append(
                context_limit_row(
                    [
                        row
                        for row in combined_context_rows
                        if row.get("requested_output_target") == target
                    ],
                    dimension="combined prompt + requested output",
                    requested_output_target=target,
                )
            )
        output.append(
            {
                "endpoint_id": endpoint,
                "dimension": "output limit",
                "documented_value": documented.get("max_output_tokens"),
                "observed_value": maximum_realized_output,
                "finding": (
                    "realized_generation_observed"
                    if maximum_realized_output is not None
                    else "realized_limit_unverified"
                ),
                "maximum_accepted_requested_output_target": maximum_requested_output,
                "requested_output_target_unit": next(
                    (
                        row.get("requested_output_unit")
                        for row in accepted_output
                        if row.get("requested_output_unit")
                    ),
                    None,
                ),
                "maximum_realized_output_tokens": maximum_realized_output,
                "minimum_rejected_requested_output_target": minimum_rejected_output,
            }
        )
        documented_values = {
            "vision": (
                "image" in json.dumps(documented_capabilities).casefold()
                or "vision" in json.dumps(documented_capabilities).casefold()
            ),
            "tool_calling": documented_capabilities.get("tools"),
            "structured_output": documented_capabilities.get("structured_output"),
        }
        for capability in ("vision", "tool_calling", "structured_output"):
            output.append(
                {
                    "endpoint_id": endpoint,
                    "dimension": capability.replace("_", " "),
                    "documented_value": documented_values.get(capability),
                    "observed_value": observed_capabilities[capability],
                    "finding": observed_capabilities[capability],
                }
            )
    return output


def load_endpoint_inventory(path: Path) -> list[dict[str, Any]]:
    value = _read_json(Path(path))
    endpoints = value.get("endpoints") if isinstance(value, Mapping) else None
    if not isinstance(endpoints, list):
        raise DirectReportError("endpoint freeze is missing endpoints")
    by_id = {
        str(row.get("model_id")): row
        for row in endpoints
        if isinstance(row, Mapping)
        and str(row.get("model_id")) in EXPECTED_ENDPOINT_SET
    }
    missing = sorted(EXPECTED_ENDPOINT_SET - set(by_id))
    if missing:
        raise DirectReportError(
            "endpoint freeze missing required IDs: " + ", ".join(missing)
        )
    return [
        {
            "endpoint_id": endpoint,
            "api_surface": by_id[endpoint].get("api_surface"),
            "provider": by_id[endpoint].get("provider"),
            "context_window": by_id[endpoint].get("context_window"),
            "max_output_tokens": by_id[endpoint].get("max_output_tokens"),
            "input_usd_per_million": by_id[endpoint].get("input_usd_per_million"),
            "output_usd_per_million": by_id[endpoint].get("output_usd_per_million"),
            "documented_capabilities": json.dumps(
                by_id[endpoint].get("documented_capabilities"), sort_keys=True
            ),
            "api_version": value.get("api_version"),
            "server_region": value.get("server_region"),
            "freeze_sha256": value.get("source_sha256"),
        }
        for endpoint in EXPECTED_ENDPOINT_IDS
    ]


def _flatten(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _flatten(row.get(key)) for key in fields})


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plot_bundle(
    output_directory: Path,
    requests: Sequence[Mapping[str, Any]],
    epochs: Sequence[Mapping[str, Any]],
    coverage_matrix: Sequence[Mapping[str, Any]],
    *,
    soak_blocks: Sequence[Mapping[str, Any]] = (),
    quality_pairs: Sequence[Mapping[str, Any]] = (),
    capability_evidence: Sequence[Mapping[str, Any]] = (),
    metric_audit_rows: Sequence[Mapping[str, Any]] = (),
    metric_audit_summary: Mapping[str, Any] | None = None,
) -> list[str]:
    """Render only the matched-estimand, non-empty publication chart suite."""

    from .direct_report_plots import build_public_plots

    return build_public_plots(
        output_directory,
        endpoints=EXPECTED_ENDPOINT_IDS,
        dimensions=REQUIRED_COVERAGE_DIMENSIONS,
        requests=requests,
        epochs=epochs,
        coverage_matrix=coverage_matrix,
        soak_blocks=soak_blocks,
        quality_pairs=quality_pairs,
        capability_evidence=capability_evidence,
        metric_audit_rows=metric_audit_rows,
        metric_audit_summary=metric_audit_summary or {},
    )


def reconcile_request_rows(
    plans: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    epochs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate evidence-bearing requests from unmatched audit rows.

    Breadth and completion requests must reconcile to an exact plan cell. Load
    requests must reconcile to an explicitly persisted epoch; an inferred epoch
    cannot make an otherwise orphaned request eligible for scientific summaries.
    """

    plan_by_key = {
        (
            str(row.get("source_kind")),
            str(row.get("source_id")),
            str(row.get("cell_id")),
        ): row
        for row in plans
        if row.get("cell_id") is not None
    }
    epoch_by_key = {
        (
            str(row.get("source_kind")),
            str(row.get("source_id")),
            str(row.get("epoch_id")),
        ): row
        for row in epochs
        if row.get("epoch_id") is not None
    }
    matched: list[dict[str, Any]] = []
    orphaned: list[dict[str, Any]] = []
    for request in requests:
        row = dict(request)
        source_kind = str(row.get("source_kind"))
        source_id = str(row.get("source_id"))
        parent: Mapping[str, Any] | None = None
        reason: str | None = None
        if source_kind in {"direct_breadth", "direct_completion"}:
            cell_id = row.get("cell_id")
            if cell_id is None:
                reason = "missing_plan_cell_id"
            else:
                parent = plan_by_key.get((source_kind, source_id, str(cell_id)))
                if parent is None:
                    reason = "unmatched_plan_cell_id"
        else:
            epoch_id = row.get("epoch_id")
            if epoch_id is None:
                reason = "missing_persisted_epoch_id"
            else:
                parent = epoch_by_key.get((source_kind, source_id, str(epoch_id)))
                if parent is None:
                    reason = "unmatched_persisted_epoch_id"
        if parent is not None and parent.get("endpoint_id") != row.get("endpoint_id"):
            reason = "parent_endpoint_mismatch"
            parent = None
        if parent is not None and source_kind in {
            "direct_breadth",
            "direct_completion",
        }:
            advertised_hashes = {
                field: _text(parent.get(field))
                for field in STRICT_REQUEST_CONTRACT_HASH_FIELDS
                if _text(parent.get(field)) is not None
            }
            strict_identity = advertised_hashes.get("request_identity_sha256")
            matching_legacy_hashes = 0
            for field, expected in advertised_hashes.items():
                observed = _text(row.get(field))
                if strict_identity is not None and observed is None:
                    reason = f"missing_{field}"
                    parent = None
                    break
                if observed is not None and observed != expected:
                    reason = f"{field}_mismatch"
                    parent = None
                    break
                if observed is not None:
                    matching_legacy_hashes += 1
            if parent is not None:
                row["reconciliation_policy"] = (
                    "strict_plan_contract_hashes"
                    if strict_identity is not None
                    else (
                        "legacy_id_endpoint_with_matching_available_hashes"
                        if matching_legacy_hashes
                        else "legacy_id_endpoint_only"
                    )
                )

                plan_tags = tuple(sorted(set(_coverage_tags(parent))))
                request_tags = tuple(sorted(set(_coverage_tags(row))))
                if plan_tags and request_tags and plan_tags != request_tags:
                    reason = "plan_coverage_tags_mismatch"
                    parent = None
                elif plan_tags:
                    row["coverage_tags"] = list(plan_tags)

            if parent is not None:
                plan_workload = _text(parent.get("workload"))
                if plan_workload == "unspecified":
                    plan_workload = None
                request_workload = _text(row.get("workload"))
                if plan_workload is not None:
                    workload_is_declared = (
                        row.get("workload_provenance") == "request_declared"
                        if row.get("workload_provenance") is not None
                        else request_workload not in {None, "unspecified"}
                    )
                    if not workload_is_declared:
                        row["workload"] = plan_workload
                        row["task_family"] = plan_workload
                        row["workload_provenance"] = "authoritative_plan"
                    elif request_workload != plan_workload:
                        reason = "plan_workload_mismatch"
                        parent = None

            if parent is not None:
                plan_output = _integer(parent.get("requested_output_target"))
                request_output = _integer(row.get("requested_output_target"))
                if plan_output is not None:
                    if request_output is None:
                        row["requested_output_target"] = plan_output
                        row["requested_output_unit"] = "tokens"
                    elif request_output != plan_output:
                        reason = "plan_requested_output_target_mismatch"
                        parent = None
        elif parent is not None:
            row["reconciliation_policy"] = "persisted_epoch_id_and_endpoint"
        if parent is None:
            row["reconciliation_status"] = "orphaned"
            row["orphan_reason"] = reason or "unmatched_parent"
            orphaned.append(row)
        else:
            row["reconciliation_status"] = "matched"
            matched.append(row)
    return matched, orphaned


def _source_cost_ledger_fields(
    path: Path,
    *,
    expected_source_kind: str | None = None,
    required: bool = False,
    prefer_portable_reconciliation: bool = False,
) -> dict[str, Any]:
    """Load only the allowlisted timing and cost fields from a source receipt.

    Source summaries may contain private execution details, so the public analysis
    never copies them wholesale.  The AIMD portable reconciliation is authoritative
    when present because it reprices the carried prior and terminal reservations on
    the same frozen basis as the later campaigns.
    """

    source = Path(path)
    summary_path = source / "summary.json" if source.is_dir() else None
    if summary_path is None or not summary_path.is_file():
        if required:
            raise DirectReportError(f"{source.name}: terminal summary.json is required")
        return {}
    try:
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DirectReportError(
            f"{source.name}: invalid terminal summary.json"
        ) from error
    if not isinstance(loaded, Mapping):
        raise DirectReportError(f"{source.name}: terminal summary must be an object")
    summary = loaded
    has_cost_ledger = any(
        summary.get(key) is not None
        for key in ("prior_cost_usd", "conservative_exposure_usd", "max_cost_usd")
    )
    if not required and not has_cost_ledger:
        return {}

    schema = _text(summary.get("schema_version"))
    expected_schemas = {
        "direct_aimd": {"do_direct_summary_v1"},
        "direct_soak": {"do_direct_soak_summary_v1"},
        "direct_completion": {
            "do_direct_completion_summary_v1",
            "do_matched_closure_summary_v1",
        },
        "direct_breadth": {
            "do_direct_capability_summary_v3",
            "do_direct_context_summary_v3",
        },
    }
    if (
        expected_source_kind in expected_schemas
        and schema not in expected_schemas[expected_source_kind]
    ):
        raise DirectReportError(
            f"{source.name}: unrecognized terminal summary schema {schema!r}"
        )
    terminal = False
    terminal_status = _text(summary.get("status"))
    if schema == "do_direct_summary_v1":
        terminal = summary.get("all_models_complete") is True
    elif schema == "do_direct_soak_summary_v1":
        terminal = summary.get("execution_complete") is True
    elif schema == "do_direct_capability_summary_v3":
        terminal = summary.get("terminal_coverage_complete") is True
        terminal_status = terminal_status or "terminal_coverage_complete"
    elif schema == "do_direct_context_summary_v3":
        terminal = summary.get("execution_complete") is True
    elif schema == "do_direct_completion_summary_v1":
        terminal = terminal_status in COMPLETION_TERMINAL_STATUSES
    elif schema == "do_matched_closure_summary_v1":
        terminal = terminal_status in COMPLETION_TERMINAL_STATUSES
    if not terminal:
        raise DirectReportError(f"{source.name}: cost summary is not terminal")

    def normalized_utc(value: Any, *, field: str) -> str:
        text = _text(value)
        if text is None:
            raise DirectReportError(f"{source.name}: {field} is required")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as error:
            raise DirectReportError(
                f"{source.name}: {field} is not an ISO-8601 timestamp"
            ) from error
        if parsed.tzinfo is None:
            raise DirectReportError(f"{source.name}: {field} must include a timezone")
        return parsed.astimezone(timezone.utc).isoformat()

    fields: dict[str, Any] = {
        "started_at": normalized_utc(summary.get("started_at"), field="started_at"),
        "ended_at": normalized_utc(summary.get("ended_at"), field="ended_at"),
        "prior_conservative_exposure_usd": _number(summary.get("prior_cost_usd")),
        "cumulative_conservative_exposure_usd": _number(
            summary.get("conservative_exposure_usd")
        ),
        "cost_cap_usd": _number(summary.get("max_cost_usd")),
        "cost_basis": "source_terminal_summary",
        "summary_schema_version": schema,
        "summary_sha256": _sha256(summary_path),
        "terminal_status": terminal_status or "execution_complete",
        "http_402_latched": summary.get("http_402_latched") is True,
    }

    if prefer_portable_reconciliation and source.is_dir():
        reconciliation_path = source / "reconciliation-portable.json"
        if not reconciliation_path.is_file():
            manifest_path = source / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise DirectReportError(
                    f"{source.name}: portable reconciliation is required"
                ) from error
            rows = (
                manifest.get("model_specs") if isinstance(manifest, Mapping) else None
            )
            contracts: dict[str, Mapping[str, Any]] = {}
            if isinstance(rows, list):
                for value in rows:
                    if not isinstance(value, Mapping):
                        contracts = {}
                        break
                    model_id = _text(value.get("model_id"))
                    if model_id is None or model_id in contracts:
                        contracts = {}
                        break
                    contracts[model_id] = value
            # Historical AIMD manifests may include the now-quarantined Arcee
            # endpoint.  They remain admissible only for exact cumulative-cost
            # reconciliation; hosted performance estimands were filtered above.
            current_contract = set(contracts) in {
                EXPECTED_ENDPOINT_SET,
                KNOWN_EVIDENCE_ENDPOINT_SET,
            }
            for model_id, observed in contracts.items():
                expected = MODEL_BY_ID.get(model_id)
                current_contract = (
                    current_contract
                    and expected is not None
                    and all(
                        (
                            _number(observed.get("input_usd_per_million"))
                            == expected.input_usd_per_million,
                            _number(observed.get("output_usd_per_million"))
                            == expected.output_usd_per_million,
                            _integer(observed.get("context_window"))
                            == expected.context_window,
                            observed.get("vision") is expected.vision,
                            observed.get("tool_calling") is expected.tool_calling,
                            observed.get("primary") is expected.primary,
                        )
                    )
                )
            if not current_contract:
                raise DirectReportError(
                    f"{source.name}: portable reconciliation is required"
                )
            fields.update(
                {
                    "cost_basis": "source_terminal_summary_current_frozen_contract",
                    "model_contract_attestation": (
                        "source_manifest_exactly_matches_current_frozen_contract"
                    ),
                    "source_manifest_sha256": _sha256(manifest_path),
                }
            )
            reconciliation_path = None
        if reconciliation_path is None:
            pass
        else:
            try:
                reconciliation = json.loads(
                    reconciliation_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as error:
                raise DirectReportError(
                    f"{source.name}: invalid portable reconciliation"
                ) from error
            reconciliation = _mapping(reconciliation)
            if (
                reconciliation.get("schema_version")
                != "do_direct_aimd_reconciliation_v1"
            ):
                raise DirectReportError(
                    f"{source.name}: unrecognized portable reconciliation schema"
                )
            reconciliation_body = dict(reconciliation)
            internal_sha = reconciliation_body.pop("receipt_sha256", None)
            reconstructed_sha = hashlib.sha256(
                canonical_json(reconciliation_body).encode("utf-8")
            ).hexdigest()
            if internal_sha != reconstructed_sha:
                raise DirectReportError(
                    f"{source.name}: portable reconciliation internal hash mismatch"
                )
            settlement = _mapping(reconciliation.get("settlement"))
            prior = _number(settlement.get("reconciled_prior_exposure_usd"))
            cumulative = _number(settlement.get("reconciled_cumulative_exposure_usd"))
            if prior is None or cumulative is None:
                raise DirectReportError(
                    f"{source.name}: portable reconciliation lacks settled exposure"
                )
            fields.update(
                {
                    "prior_conservative_exposure_usd": prior,
                    "cumulative_conservative_exposure_usd": cumulative,
                    "cost_basis": "portable_reconciliation",
                    "reconciliation_schema_version": reconciliation.get(
                        "schema_version"
                    ),
                    "reconciliation_sha256": _sha256(reconciliation_path),
                }
            )

    prior = _number(fields.get("prior_conservative_exposure_usd"))
    cumulative = _number(fields.get("cumulative_conservative_exposure_usd"))
    cap = _number(fields.get("cost_cap_usd"))
    if (
        prior is None
        or cumulative is None
        or cap is None
        or prior < 0
        or cumulative < prior
        or cumulative > cap + 1e-9
    ):
        raise DirectReportError(f"{source.name}: invalid terminal exposure chain")
    return fields


def _build_cost_summary(
    sources: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    endpoint_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a public time-and-cost ledger without conflating cost estimands."""

    stages: list[dict[str, Any]] = []
    for source in sources:
        cumulative = _number(source.get("cumulative_conservative_exposure_usd"))
        if cumulative is None:
            continue
        prior = _number(source.get("prior_conservative_exposure_usd"))
        stage = {
            "source_kind": str(source.get("source_kind") or "unknown"),
            "source_id": str(source.get("source_id") or "unknown"),
            "started_at": _text(source.get("started_at")),
            "ended_at": _text(source.get("ended_at")),
            "prior_conservative_exposure_usd": prior,
            "cumulative_conservative_exposure_usd": cumulative,
            "incremental_conservative_exposure_usd": (
                cumulative - prior if prior is not None else None
            ),
            "cost_cap_usd": _number(source.get("cost_cap_usd")),
            "cost_basis": str(source.get("cost_basis") or "source_terminal_summary"),
            "summary_schema_version": source.get("summary_schema_version"),
            "summary_sha256": source.get("summary_sha256"),
            "terminal_status": source.get("terminal_status"),
            "http_402_latched": source.get("http_402_latched") is True,
        }
        if source.get("reconciliation_schema_version") is not None:
            stage["reconciliation_schema_version"] = source.get(
                "reconciliation_schema_version"
            )
            stage["reconciliation_sha256"] = source.get("reconciliation_sha256")
        stages.append(stage)
    stages.sort(key=lambda row: (str(row.get("started_at") or ""), row["source_id"]))
    required_source_ids = {
        str(source.get("source_id") or "")
        for source in sources
        if source.get("cost_summary_required") is True
    }
    staged_source_ids = {str(stage.get("source_id") or "") for stage in stages}
    if not required_source_ids.issubset(staged_source_ids):
        raise DirectReportError(
            "cost ledger omits a manifest-declared cost-bearing source"
        )

    for index, stage in enumerate(stages):
        prior = _number(stage.get("prior_conservative_exposure_usd"))
        cumulative = _number(stage.get("cumulative_conservative_exposure_usd"))
        cap = _number(stage.get("cost_cap_usd"))
        if prior is None or cumulative is None or cap is None:
            raise DirectReportError("cost ledger stage lacks prior, cumulative, or cap")
        if prior < 0 or cumulative < prior or cumulative > cap + 1e-9:
            raise DirectReportError("cost ledger stage has invalid exposure bounds")
        if index:
            previous = stages[index - 1]
            previous_cumulative = float(
                previous["cumulative_conservative_exposure_usd"]
            )
            if not math.isclose(prior, previous_cumulative, rel_tol=0, abs_tol=1e-9):
                raise DirectReportError(
                    "cost ledger stages are not exposure-contiguous"
                )
            if not math.isclose(
                cap,
                float(previous["cost_cap_usd"]),
                rel_tol=0,
                abs_tol=1e-9,
            ):
                previous_cap = float(previous["cost_cap_usd"])
                if cap < previous_cap:
                    raise DirectReportError(
                        "cost ledger cap decreased across owner-authorized stages"
                    )
                stage["cost_cap_revision_from_usd"] = previous_cap
                stage["cost_cap_revision_to_usd"] = cap
            if str(stage.get("started_at")) < str(previous.get("ended_at")):
                raise DirectReportError("cost ledger stage windows overlap")

    terminal_stage = stages[-1] if stages else None
    request_costs = [
        _number(row.get("estimated_cost_usd"))
        if row.get("cost_attributed") is True
        else None
        for row in requests
    ]
    present_costs = [value for value in request_costs if value is not None]
    attributed = sum(present_costs)
    endpoint_attributed = sum(
        _number(row.get("estimated_cost_usd")) or 0.0 for row in endpoint_summaries
    )
    endpoint_request_count = sum(
        _integer(row.get("request_count")) or 0 for row in endpoint_summaries
    )
    if endpoint_request_count != len(requests) or not math.isclose(
        attributed, endpoint_attributed, rel_tol=0, abs_tol=1e-9
    ):
        raise DirectReportError(
            "request-attributed costs disagree with endpoint summary accounting"
        )
    return {
        "schema_version": "digitalocean_public_cost_summary_v1",
        "request_attributed_estimated_cost_usd": attributed,
        "cost_attributed_request_count": len(present_costs),
        "cost_unattributed_request_count": len(requests) - len(present_costs),
        "request_cost_attribution_complete": len(present_costs) == len(requests),
        "conservative_campaign_exposure_usd": (
            terminal_stage["cumulative_conservative_exposure_usd"]
            if terminal_stage is not None
            else None
        ),
        "conservative_exposure_source_id": (
            terminal_stage["source_id"] if terminal_stage is not None else None
        ),
        "conservative_exposure_receipt_schema_version": (
            terminal_stage["summary_schema_version"]
            if terminal_stage is not None
            else None
        ),
        "conservative_exposure_receipt_sha256": (
            terminal_stage["summary_sha256"] if terminal_stage is not None else None
        ),
        "cost_cap_usd": (
            terminal_stage["cost_cap_usd"] if terminal_stage is not None else None
        ),
        "initial_carried_conservative_exposure_usd": (
            stages[0]["prior_conservative_exposure_usd"] if stages else None
        ),
        "source_stages": stages,
        "cost_cap_revision_count": sum(
            1 for stage in stages if stage.get("cost_cap_revision_from_usd") is not None
        ),
        "cost_cap_history_usd": list(
            dict.fromkeys(float(stage["cost_cap_usd"]) for stage in stages)
        ),
        "estimand_relationship": "overlapping_non_additive",
        "billing_credit_http_402_latched": any(
            stage.get("http_402_latched") is True for stage in stages
        ),
        "http_402_latched_source_ids": [
            stage["source_id"]
            for stage in stages
            if stage.get("http_402_latched") is True
        ],
        "interpretation": (
            "Request-attributed estimated cost sums normalized request usage. "
            "Conservative campaign exposure is the authoritative cumulative guard "
            "and retains reservations for incomplete, error, and unknown outcomes; "
            "the two values overlap, are not additive, and are not interchangeable."
            " Each stage is checked against the cap in force for that stage; any "
            "owner-authorized cap increase is preserved explicitly in the ledger."
        ),
    }


def _breadth_cost_summary_required(path: Path) -> bool:
    """Return whether a breadth manifest declares a terminal cost-bearing campaign."""

    manifest_path = Path(path) / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DirectReportError(f"{Path(path).name}: invalid manifest.json") from error
    schema = _text(_mapping(manifest).get("schema_version"))
    return schema in {
        "do_direct_capability_manifest_v3",
        "do_direct_context_manifest_v3",
    }


def analyze_and_write(
    *,
    breadth_directories: Sequence[Path],
    aimd_directories: Sequence[Path],
    soak_directories: Sequence[Path] = (),
    completion_directories: Sequence[Path] = (),
    closure_directories: Sequence[Path] = (),
    cost_only_directories: Sequence[Path] = (),
    endpoint_freeze: Path,
    output_directory: Path,
    seed: int = 20260823,
    bootstrap_replicates: int = 2_000,
    publication_mode: str = "draft",
) -> dict[str, Any]:
    if bootstrap_replicates <= 0:
        raise DirectReportError("bootstrap_replicates must be positive")
    if publication_mode not in {"draft", "final"}:
        raise DirectReportError("publication_mode must be 'draft' or 'final'")
    plans: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    epochs: list[dict[str, Any]] = []
    scope_exclusions: list[dict[str, Any]] = []
    soak_sources: list[dict[str, Any]] = []
    soak_summaries: list[dict[str, Any]] = []
    soak_block_summaries: list[dict[str, Any]] = []
    soak_quality_summaries: list[dict[str, Any]] = []
    soak_recovery_summaries: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    def hosted_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Keep current DigitalOcean-hosted endpoints out of historical partner data.

        Strict loaders still validate old Arcee receipts so cumulative exposure
        remains auditable.  The public performance estimands, coverage matrix,
        and figures are then constructed only from the current hosted allowlist.
        """

        output: list[dict[str, Any]] = []
        for row in rows:
            endpoint = _text(row.get("endpoint_id") or row.get("model_id"))
            if endpoint in EXPECTED_ENDPOINT_SET:
                output.append(dict(row))
        return output

    def hosted_soak_source(loaded: Mapping[str, Any]) -> dict[str, Any]:
        filtered = dict(loaded)
        for key in (
            "plan_cells",
            "requests",
            "epochs",
            "phase_summaries",
            "soak_summaries",
            "block_summaries",
            "quality_summaries",
            "recovery_summaries",
            "cell_rows",
        ):
            filtered[key] = hosted_rows(loaded.get(key, ()))
        return filtered

    def append_soak_evidence(
        loaded: Mapping[str, Any],
        *,
        cost_path: Path | None,
        parent_completion_source_id: str | None = None,
    ) -> None:
        loaded = hosted_soak_source(loaded)
        soak_sources.append(dict(loaded))
        requests.extend(loaded["requests"])
        epochs.extend(loaded["epochs"])
        soak_summaries.extend(loaded["soak_summaries"])
        soak_block_summaries.extend(loaded["block_summaries"])
        soak_quality_summaries.extend(loaded["quality_summaries"])
        soak_recovery_summaries.extend(loaded["recovery_summaries"])
        source = {
            "source_kind": "direct_soak",
            "source_id": loaded["source_id"],
            "campaign_id": loaded["campaign_id"],
            "source_manifest_sha256": loaded["source_manifest_sha256"],
            "planned_cells": len(loaded["plan_cells"]),
            "request_rows": len(loaded["requests"]),
            "epoch_rows": len(loaded["epochs"]),
            "phase_rows": len(loaded["phase_summaries"]),
            "analysis_block_rows": len(loaded["block_summaries"]),
            "quality_pair_rows": len(loaded["quality_summaries"]),
            "recovery_rows": len(loaded["recovery_summaries"]),
            "scientifically_complete_cells": sum(
                row.get("scientifically_complete") is True
                for row in loaded["soak_summaries"]
            ),
            "cost_summary_required": cost_path is not None,
        }
        if cost_path is not None:
            source.update(
                _source_cost_ledger_fields(
                    cost_path,
                    expected_source_kind="direct_soak",
                    required=True,
                )
            )
        else:
            source.update(
                {
                    "cost_stage_included": False,
                    "cost_stage_policy": "parent_completion_summary_is_single_stage",
                    "cost_stage_owner_source_id": parent_completion_source_id,
                }
            )
        sources.append(source)

    for path in breadth_directories:
        cost_summary_required = _breadth_cost_summary_required(path)
        loaded_plans, loaded_requests = load_breadth_directory(path)
        loaded_scope_exclusions = load_breadth_scope_exclusions(path)
        plans.extend(loaded_plans)
        requests.extend(loaded_requests)
        scope_exclusions.extend(loaded_scope_exclusions)
        sources.append(
            {
                "source_kind": "direct_breadth",
                "source_id": Path(path).name,
                "planned_cells": len(loaded_plans),
                "request_rows": len(loaded_requests),
                "scope_exclusion_rows": len(loaded_scope_exclusions),
                "cost_summary_required": cost_summary_required,
                **_source_cost_ledger_fields(
                    path,
                    expected_source_kind="direct_breadth",
                    required=cost_summary_required,
                ),
            }
        )
    for path in aimd_directories:
        loaded_requests, loaded_epochs = load_aimd_directory(path)
        requests.extend(loaded_requests)
        epochs.extend(loaded_epochs)
        sources.append(
            {
                "source_kind": "direct_aimd",
                "source_id": Path(path).name,
                "request_rows": len(loaded_requests),
                "epoch_rows": len(loaded_epochs),
                "cost_summary_required": True,
                **_source_cost_ledger_fields(
                    path,
                    expected_source_kind="direct_aimd",
                    required=True,
                    prefer_portable_reconciliation=True,
                ),
            }
        )
    nested_soak_directories: set[Path] = set()
    for path in completion_directories:
        loaded = load_completion_directory(path)
        plans.extend(loaded["plans"])
        requests.extend(loaded["requests"])
        sources.append(
            {
                "source_kind": "direct_completion",
                "source_id": loaded["source_id"],
                "campaign_id": loaded["campaign_id"],
                "source_manifest_sha256": loaded["source_manifest_sha256"],
                "planned_semantic_probes": len(loaded["plans"]),
                "physical_request_rows": len(loaded["requests"]),
                "terminal_probe_outcomes": len(loaded["outcomes"]),
                "nested_soak_waves": len(loaded["nested_soaks"]),
                "request_rows": len(loaded["requests"]),
                "cost_summary_required": True,
                **_source_cost_ledger_fields(
                    path,
                    expected_source_kind="direct_completion",
                    required=True,
                ),
            }
        )
        for nested in loaded["nested_soaks"]:
            append_soak_evidence(
                nested,
                cost_path=None,
                parent_completion_source_id=loaded["source_id"],
            )
        nested_soak_directories.update(loaded["nested_soak_directories"])
    for path in closure_directories:
        loaded = load_matched_closure_directory(path)
        plans.extend(loaded["plans"])
        requests.extend(loaded["requests"])
        sources.append(
            {
                "source_kind": "direct_completion",
                "source_id": loaded["source_id"],
                "campaign_id": loaded["campaign_id"],
                "source_manifest_sha256": loaded["source_manifest_sha256"],
                "planned_semantic_probes": len(loaded["plans"]),
                "physical_request_rows": len(loaded["requests"]),
                "terminal_probe_outcomes": len(loaded["outcomes"]),
                "conclusive_probe_outcomes": _integer(
                    loaded["summary"].get("conclusive_cells")
                ),
                "http_status_counts": dict(
                    _mapping(loaded["summary"].get("http_status_counts"))
                ),
                "finalization_mode": _text(loaded["summary"].get("finalization_mode")),
                "nested_soak_waves": 0,
                "request_rows": len(loaded["requests"]),
                "cost_summary_required": True,
                **_source_cost_ledger_fields(
                    path,
                    expected_source_kind="direct_completion",
                    required=True,
                ),
            }
        )
    for path in soak_directories:
        if Path(path).resolve() in nested_soak_directories:
            continue
        loaded = load_soak_directory(path)
        append_soak_evidence(loaded, cost_path=Path(path))
    evidence_directories = {
        Path(path).resolve()
        for path in (
            *breadth_directories,
            *aimd_directories,
            *soak_directories,
            *completion_directories,
            *closure_directories,
        )
    }
    for path in cost_only_directories:
        resolved = Path(path).resolve()
        if resolved in evidence_directories:
            raise DirectReportError(
                "a cost-only directory cannot also contribute scientific evidence"
            )
        source_id = Path(path).name
        if any(str(source.get("source_id")) == source_id for source in sources):
            raise DirectReportError("duplicate cost-only source_id")
        sources.append(
            {
                "source_kind": "direct_soak",
                "source_id": source_id,
                "request_rows": 0,
                "epoch_rows": 0,
                "scientific_evidence_included": False,
                "cost_summary_required": True,
                "cost_stage_policy": (
                    "cost_receipt_only_scientific_rows_excluded_due_invalid_"
                    "quality_pair_payload_hashes"
                ),
                **_source_cost_ledger_fields(
                    Path(path),
                    expected_source_kind="direct_soak",
                    required=True,
                ),
            }
        )
    partner_requests = [
        row
        for row in requests
        if _text(row.get("endpoint_id") or row.get("model_id"))
        in HISTORICAL_PARTNER_ENDPOINT_IDS
    ]
    quarantined_partner_rows = {
        "plans": sum(
            _text(row.get("endpoint_id") or row.get("model_id"))
            in HISTORICAL_PARTNER_ENDPOINT_IDS
            for row in plans
        ),
        "requests": sum(
            _text(row.get("endpoint_id") or row.get("model_id"))
            in HISTORICAL_PARTNER_ENDPOINT_IDS
            for row in requests
        ),
        "epochs": sum(
            _text(row.get("endpoint_id") or row.get("model_id"))
            in HISTORICAL_PARTNER_ENDPOINT_IDS
            for row in epochs
        ),
        "scope_exclusions": sum(
            _text(row.get("endpoint_id") or row.get("model_id"))
            in HISTORICAL_PARTNER_ENDPOINT_IDS
            for row in scope_exclusions
        ),
    }
    plans = hosted_rows(plans)
    requests = hosted_rows(requests)
    epochs = hosted_rows(epochs)
    scope_exclusions = hosted_rows(scope_exclusions)
    sources.append(
        {
            "source_kind": "scope_quarantine",
            "source_id": "historical-partner-models",
            "policy": "excluded_from_all_current_hosted_endpoint_estimands",
            "endpoint_ids": sorted(HISTORICAL_PARTNER_ENDPOINT_IDS),
            "quarantined_rows": quarantined_partner_rows,
            "request_attributed_estimated_cost_usd": sum(
                _number(row.get("estimated_cost_usd")) or 0.0
                for row in partner_requests
                if row.get("cost_attributed") is True
            ),
            "cost_policy": (
                "historical campaign exposure remains in cumulative stage receipts"
            ),
            "cost_summary_required": False,
            "cost_stage_included": False,
        }
    )
    # Each source/request identity is immutable. Duplicate rows are an input error,
    # not extra statistical evidence.
    request_keys = [(row["source_id"], row["request_id"]) for row in requests]
    if len(request_keys) != len(set(request_keys)):
        raise DirectReportError("duplicate source_id/request_id rows detected")
    epoch_keys = [(row["source_id"], row["epoch_id"]) for row in epochs]
    if len(epoch_keys) != len(set(epoch_keys)):
        raise DirectReportError("duplicate source_id/epoch_id rows detected")
    plan_keys = [
        (row["source_kind"], row["source_id"], row["cell_id"]) for row in plans
    ]
    if len(plan_keys) != len(set(plan_keys)):
        raise DirectReportError(
            "duplicate source_kind/source_id/cell_id plan rows detected"
        )
    scope_exclusion_keys = [
        (
            row["source_kind"],
            row["source_id"],
            row["endpoint_id"],
            row["scope_exclusion_id"],
        )
        for row in scope_exclusions
    ]
    if len(scope_exclusion_keys) != len(set(scope_exclusion_keys)):
        raise DirectReportError("duplicate manifest scope exclusion rows detected")
    soak_summary_keys = [(row["source_id"], row["cell_id"]) for row in soak_summaries]
    if len(soak_summary_keys) != len(set(soak_summary_keys)):
        raise DirectReportError(
            "duplicate direct soak source/cell summary rows detected"
        )
    soak_block_keys = [
        (row["source_id"], row["analysis_block_id"]) for row in soak_block_summaries
    ]
    if len(soak_block_keys) != len(set(soak_block_keys)):
        raise DirectReportError("duplicate direct soak analysis_block_id rows detected")
    soak_quality_keys = [
        (row["source_id"], row["quality_pair_id"]) for row in soak_quality_summaries
    ]
    if len(soak_quality_keys) != len(set(soak_quality_keys)):
        raise DirectReportError("duplicate direct soak quality_pair_id rows detected")
    requests, orphan_requests = reconcile_request_rows(plans, requests, epochs)
    epochs = _epoch_units_from_requests(requests, epochs)
    if any(row.get("schema_version") != NORMALIZED_REQUEST_SCHEMA for row in requests):
        raise DirectReportError("normalized request schema version mismatch")
    if any(row.get("schema_version") != NORMALIZED_EPOCH_SCHEMA for row in epochs):
        raise DirectReportError("normalized epoch schema version mismatch")
    for row in requests:
        if not isinstance(row.get("cost_attributed"), bool) or (
            (row.get("cost_attributed") is True)
            != (row.get("estimated_cost_usd") is not None)
        ):
            raise DirectReportError(
                "normalized request cost attribution flag is inconsistent"
            )
        for key in (
            "input_tokens",
            "output_tokens",
            "request_seconds",
            "ttft_seconds",
            "generation_seconds",
            "output_tokens_per_second",
            "estimated_cost_usd",
        ):
            value = row.get(key)
            parsed = _number(value)
            if value is not None and (parsed is None or parsed < 0):
                raise DirectReportError(f"normalized request {key} has invalid units")
    for row in epochs:
        scheduled = _integer(row.get("scheduled_count"))
        completed = _integer(row.get("completed_count"))
        successes = _integer(row.get("success_count"))
        if scheduled is None or completed is None or successes is None:
            raise DirectReportError("normalized epoch sample counts must be integers")
        if successes > completed or completed > scheduled:
            raise DirectReportError("normalized epoch sample counts are inconsistent")
        offered_rps = _number(row.get("offered_rps"))
        offered_rpm = _number(row.get("offered_rpm"))
        if offered_rps is not None and (
            offered_rpm is None
            or not math.isclose(offered_rpm, offered_rps * 60, rel_tol=0, abs_tol=1e-9)
        ):
            raise DirectReportError("normalized epoch offered RPM/RPS units disagree")
        for key in (
            "elapsed_seconds",
            "offered_window_seconds",
            "offered_rps",
            "offered_rpm",
            "achieved_rpm",
            "effective_input_tpm",
            "effective_output_tpm",
            "estimated_cost_usd",
        ):
            value = row.get(key)
            parsed = _number(value)
            if value is not None and (parsed is None or parsed < 0):
                raise DirectReportError(f"normalized epoch {key} has invalid units")
    inventory = load_endpoint_inventory(endpoint_freeze)
    endpoint_summaries, workload_summaries = _group_rows(
        requests,
        epochs,
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    # Recompute requested replicate count here; helper defaults remain deterministic.
    for row in workload_summaries:
        row["bootstrap_replicates_requested"] = bootstrap_replicates
    for row in endpoint_summaries:
        row["bootstrap_replicates_requested"] = bootstrap_replicates
    cost_summary = _build_cost_summary(sources, requests, endpoint_summaries)
    coverage_ledger, coverage_matrix, coverage_summary = build_coverage(
        plans, requests, epochs, scope_exclusions, soak_sources
    )
    capacity = build_capacity_summary(
        epochs,
        requests=requests,
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    limits = build_observed_limits(requests, inventory)
    metric_audit_rows, metric_audit_summary = build_metric_audit(requests)
    cache_state_summaries = build_cache_state_summaries(
        requests,
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    capability_evidence = build_capability_evidence(requests)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    for generated_receipt in (
        "publication-scan.json",
        "public-safety-scan.json",
        "manifest.json",
    ):
        candidate = output / generated_receipt
        if candidate.is_file() and not candidate.is_symlink():
            candidate.unlink()
    _write_csv(output / "normalized-requests.csv", requests)
    _write_jsonl(output / "normalized-requests.jsonl", requests)
    _write_csv(output / "orphan-requests.csv", orphan_requests)
    _write_jsonl(output / "orphan-requests.jsonl", orphan_requests)
    _write_csv(output / "normalized-epochs.csv", epochs)
    _write_jsonl(output / "normalized-epochs.jsonl", epochs)
    _write_csv(output / "endpoint-inventory.csv", inventory)
    _write_csv(output / "endpoint-summary.csv", endpoint_summaries)
    _write_csv(output / "endpoint-workload-metrics.csv", workload_summaries)
    _write_csv(output / "capacity-summary.csv", capacity)
    _write_csv(output / "soak-cell-summary.csv", soak_summaries)
    _write_csv(output / "soak-block-summary.csv", soak_block_summaries)
    _write_csv(output / "quality-pair-summary.csv", soak_quality_summaries)
    _write_csv(output / "recovery-summary.csv", soak_recovery_summaries)
    _write_csv(output / "coverage-ledger.csv", coverage_ledger)
    _write_jsonl(output / "coverage-ledger.jsonl", coverage_ledger)
    _write_csv(output / "coverage-matrix.csv", coverage_matrix)
    _write_csv(output / "scope-exclusions.csv", scope_exclusions)
    _write_jsonl(output / "scope-exclusions.jsonl", scope_exclusions)
    _write_csv(output / "observed-limits.csv", limits)
    _write_csv(output / "metric-audit.csv", metric_audit_rows)
    _write_csv(output / "cache-state-metrics.csv", cache_state_summaries)
    _write_csv(output / "capability-evidence.csv", capability_evidence)
    charts = _plot_bundle(
        output,
        requests,
        epochs,
        coverage_matrix,
        soak_blocks=soak_block_summaries,
        quality_pairs=soak_quality_summaries,
        capability_evidence=capability_evidence,
        metric_audit_rows=metric_audit_rows,
        metric_audit_summary=metric_audit_summary,
    )
    analysis: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "endpoint_inventory": inventory,
        "data_sources": sources,
        "cost_summary": cost_summary,
        "request_reconciliation": {
            "matched_request_rows": len(requests),
            "orphan_request_rows": len(orphan_requests),
            "all_requests_reconciled": not orphan_requests,
            "matched_policy_counts": dict(
                sorted(
                    Counter(
                        str(row.get("reconciliation_policy") or "unlabeled")
                        for row in requests
                    ).items()
                )
            ),
            "orphan_reason_counts": dict(
                sorted(
                    Counter(
                        str(row["orphan_reason"]) for row in orphan_requests
                    ).items()
                )
            ),
            "scientific_input_policy": (
                "Only request rows matched to an exact breadth plan cell or an explicitly "
                "persisted load epoch enter summaries, limits, coverage, or plots"
            ),
        },
        "coverage_summary": coverage_summary,
        "coverage_matrix": coverage_matrix,
        "scope_exclusions": scope_exclusions,
        "endpoint_summaries": endpoint_summaries,
        "workload_summaries": workload_summaries,
        "capacity_summaries": capacity,
        "soak_summaries": soak_summaries,
        "soak_block_summaries": soak_block_summaries,
        "soak_quality_summaries": soak_quality_summaries,
        "soak_recovery_summaries": soak_recovery_summaries,
        "observed_limits": limits,
        "metric_audit_summary": metric_audit_summary,
        "cache_state_summaries": cache_state_summaries,
        "capability_evidence": capability_evidence,
        "metric_definitions": METRIC_DEFINITIONS,
        "statistical_methodology": {
            "confidence_level": 0.95,
            "serial_sampling_unit": "request_id",
            "load_sampling_unit": "epoch_id",
            "soak_sampling_unit": "analysis_block_id",
            "soak_quality_sampling_unit": "quality_pair_id",
            "soak_recovery_sampling_unit": "phase_id",
            "soak_recovery_within_phase_binomial_sampling_unit": "request_id",
            "soak_block_dependence_note": (
                "Four contiguous 30-second analysis blocks preserve their exact "
                "analysis_block_id identities; Student-t block intervals are exploratory "
                "and do not model serial correlation"
            ),
            "soak_capacity_policy": (
                "Two-minute soak evidence is reported separately and never relabelled "
                "as an AIMD confirmation epoch"
            ),
            "success_interval_serial": "Wilson score interval",
            "continuous_metric_interval_serial": "seeded request percentile bootstrap",
            "load_intervals": "seeded percentile bootstrap after one estimate per epoch",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": seed,
            "output_tokens_are_not_independent_samples": True,
            "p99_minimum_observations": 1_000,
            "timing_metric_policy": (
                "SSE event spans are audit-only. Headline output throughput is aggregate "
                "successful billed completion tokens per complete wall-clock interval; "
                "request curves use the labelled post-TTFT end-to-end proxy."
            ),
            "cache_policy": (
                "TTFT is stratified by explicitly observed cache state. Prefill proxy is "
                "headline-eligible only for explicit cache misses; unknown is not a miss."
            ),
            "multi_choice_policy": (
                "Requests with n>1 retain aggregate cost/goodput but are excluded from "
                "per-sequence latency-throughput curves."
            ),
        },
        "limitations": list(LIMITATIONS),
        "output_files": {
            "tables": [
                "normalized-requests.csv",
                "normalized-requests.jsonl",
                "orphan-requests.csv",
                "orphan-requests.jsonl",
                "normalized-epochs.csv",
                "normalized-epochs.jsonl",
                "endpoint-inventory.csv",
                "endpoint-summary.csv",
                "endpoint-workload-metrics.csv",
                "capacity-summary.csv",
                "soak-cell-summary.csv",
                "soak-block-summary.csv",
                "quality-pair-summary.csv",
                "recovery-summary.csv",
                "coverage-ledger.csv",
                "coverage-ledger.jsonl",
                "coverage-matrix.csv",
                "scope-exclusions.csv",
                "scope-exclusions.jsonl",
                "observed-limits.csv",
                "metric-audit.csv",
                "cache-state-metrics.csv",
                "capability-evidence.csv",
            ],
            "charts": charts,
        },
        "public_bundle_safety": {
            "input_policy": "allowlisted derived fields only",
            "scanner": PUBLIC_SAFETY_SCAN_SCHEMA,
            "scan_receipt": "public-safety-scan.json",
            "scope_note": (
                "secret/path/file-type safety only; not a scientific or publication gate"
            ),
        },
    }
    contract_gate = validate_public_analysis_contract(
        analysis, require_complete=publication_mode == "final"
    )
    analysis["contract_gate"] = contract_gate
    analysis["publication_mode"] = publication_mode
    analysis["publication_status"] = (
        "publication_ready"
        if publication_mode == "final"
        else (
            "draft_complete_coverage"
            if coverage_summary["is_100_percent"]
            else "draft_incomplete_coverage"
        )
    )
    analysis_path = output / "analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    scan = scan_public_bundle_safety(output)
    if not scan["passed"]:
        raise DirectReportError(
            f"public bundle sanitization failed with {scan['finding_count']} finding(s)"
        )
    (output / "public-safety-scan.json").write_text(
        json.dumps(scan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path != output / "manifest.json"
    )
    manifest = {
        "schema_version": "digitalocean_direct_public_bundle_manifest_v1",
        "files": [
            {
                "path": str(path.relative_to(output)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in manifest_files
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return analysis
