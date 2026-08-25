"""Six-hour, idempotent completion campaign for the direct benchmark.

The campaign does two things and nothing more:

* retries only inconclusive context/capability cells (plus explicit realized
  output anchors) with fsync-before-send reservations and bounded backoff;
* re-runs only failed/gated sustained endpoint-shape cells at predeclared
  descending fractions of their receipt-backed AIMD candidate.

It composes the existing direct runners. There is no external orchestrator and
no hidden retry: every possible provider send has a deterministic request ID,
reservation row, terminal row, and metric-audit row.  A reservation without a
terminal row is treated as an unknown send and is never replayed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

import httpx

from do_benchmark.core import (
    BenchmarkTask,
    JsonlJournal,
    MODEL_BY_ID,
    DIGITALOCEAN_HOSTED_MODEL_IDS,
    StreamResult,
    require_digitalocean_hosted_models,
    canonical_json,
    parse_token_usage,
    score_result,
    stable_hash,
    stream_chat_completion,
    utc_now,
)
from do_benchmark.credentials import digitalocean_credentials
from do_benchmark.direct_aimd import (
    BudgetLedger,
    conservative_request_cost,
    sanitized_header_signals,
)
from do_benchmark.direct_capability import (
    CapabilityCell,
    _coverage_conclusive,
    _sanitized_error_evidence,
    build_capability_cells,
)
from do_benchmark.direct_context import (
    DEFAULT_CHARS_PER_TOKEN,
    _probe_from_plan_row,
    build_retrieval_task,
    classify_failure,
)
from do_benchmark.direct_soak import (
    DirectSoakCampaign,
    OutputDirectoryLease,
    SoakConfig,
)
from do_benchmark.timing_audit import audit_row, cache_observation, timing_evidence


RequestExecutor = Callable[[str, BenchmarkTask, int], Awaitable[StreamResult]]
RETRYABLE_HTTP = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
DEFAULT_RATE_LADDER = (0.75, 0.50, 0.25, 0.125)
DEFAULT_OUTPUT_TOKEN_ANCHORS = (256, 1_024, 4_096, 16_384)
SOAK_CENSOR_SCHEMA = "do_direct_completion_soak_censor_v1"
COMPLETION_SOAK_REPAIR_VERSION = "pairfix-v1"


class CompletionPreflightError(RuntimeError):
    """Raised before credentials are loaded when completion cannot run safely."""


@dataclass(frozen=True)
class CompletionConfig:
    output_dir: Path
    soak_dir: Path
    context_dir: Path
    capability_dir: Path
    aimd_dir: Path
    model_ids: tuple[str, ...]
    prior_cost_usd: float
    max_cost_usd: float = 400.0
    launch_stop_cost_usd: float = 385.0
    duration_hours: float = 6.0
    absolute_hard_deadline: datetime | None = None
    send_reserve_minutes: float = 5.0
    request_timeout_seconds: float = 180.0
    max_concurrency: int = 12
    max_attempts: int = 3
    retry_backoff_seconds: float = 2.0
    rate_ladder: tuple[float, ...] = DEFAULT_RATE_LADDER
    output_token_anchors: tuple[int, ...] = DEFAULT_OUTPUT_TOKEN_ANCHORS
    soak_seconds: float = 120.0
    analysis_block_seconds: float = 30.0
    recovery_seconds: float = 30.0
    soak_concurrency_ceiling: int = 128
    seed: int = 20260824
    aimd_reconciliation_path: Path | None = None
    prior_lineage_root: Path | None = None
    v3_checkpoint_dir: Path | None = None
    accept_conditional_prior_exposure_basis: bool = False

    def validate(self) -> None:
        if not self.model_ids or len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("model_ids must be a non-empty unique sequence")
        unknown = sorted(set(self.model_ids) - MODEL_BY_ID.keys())
        if unknown:
            raise ValueError(f"unknown DigitalOcean models: {', '.join(unknown)}")
        require_digitalocean_hosted_models(self.model_ids)
        for path in (
            self.soak_dir,
            self.context_dir,
            self.capability_dir,
            self.aimd_dir,
        ):
            if not path.is_dir():
                raise ValueError(f"missing source artifact directory: {path}")
        if not 0 <= self.prior_cost_usd <= self.max_cost_usd:
            raise ValueError("invalid cumulative cost envelope")
        if self.max_cost_usd > 400.0 + 1e-12:
            raise ValueError(
                "completion campaign may not exceed the authorized $400 cap"
            )
        if not self.prior_cost_usd <= self.launch_stop_cost_usd <= self.max_cost_usd:
            raise ValueError("launch stop must be between prior exposure and hard cap")
        if self.max_cost_usd - self.launch_stop_cost_usd < 15.0 - 1e-12:
            raise ValueError("completion campaign must preserve a $15 drain reserve")
        if not 0 < self.duration_hours <= 6.0:
            raise ValueError("duration_hours must be in (0,6]")
        if (
            self.absolute_hard_deadline is not None
            and self.absolute_hard_deadline.tzinfo is None
        ):
            raise ValueError("absolute_hard_deadline must include a UTC offset")
        if not 0 < self.send_reserve_minutes < self.duration_hours * 60:
            raise ValueError(
                "send reserve must be positive and shorter than the campaign"
            )
        if self.request_timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        if self.max_concurrency < 1 or self.soak_concurrency_ceiling < 1:
            raise ValueError("concurrency must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry backoff cannot be negative")
        if not self.rate_ladder or any(
            not math.isfinite(item) or not 0 < item < 1 for item in self.rate_ladder
        ):
            raise ValueError("rate_ladder values must be finite and in (0,1)")
        if tuple(sorted(self.rate_ladder, reverse=True)) != self.rate_ladder:
            raise ValueError("rate_ladder must be strictly descending")
        if len(set(self.rate_ladder)) != len(self.rate_ladder):
            raise ValueError("rate_ladder must not contain duplicates")
        if not self.output_token_anchors or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 1
            for item in self.output_token_anchors
        ):
            raise ValueError("output token anchors must be positive integers")
        if not math.isclose(
            self.soak_seconds,
            self.analysis_block_seconds * 4,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("soak_seconds must equal four analysis blocks")


@dataclass(frozen=True)
class CompletionProbe:
    semantic_id: str
    lane: str
    model_id: str
    source_request_id: str | None
    source_probe_id: str
    task: BenchmarkTask
    max_output_tokens: int
    group_id: str | None = None
    group_size: int = 1
    estimated_tokens: int = 1

    def plan_row(self, max_attempts: int) -> dict[str, Any]:
        payload = {
            "messages": self.task.messages,
            "tools": self.task.tools,
            "tool_choice": self.task.tool_choice,
            "response_format": self.task.response_format,
            "parameters": self.task.parameters,
            "max_tokens": self.max_output_tokens,
        }
        return {
            "semantic_id": self.semantic_id,
            "lane": self.lane,
            "model_id": self.model_id,
            "source_request_id": self.source_request_id,
            "source_probe_id": self.source_probe_id,
            "task_id": self.task.task_id,
            "task_family": self.task.family,
            "requested_max_output_tokens": self.max_output_tokens,
            "request_payload_sha256": hashlib.sha256(
                canonical_json(payload).encode("utf-8")
            ).hexdigest(),
            "group_id": self.group_id,
            "group_size": self.group_size,
            "estimated_tokens": self.estimated_tokens,
            "attempt_request_ids": [
                attempt_request_id(self.semantic_id, index)
                for index in range(max_attempts)
            ],
        }


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CompletionPreflightError(
            f"invalid JSON artifact {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CompletionPreflightError(f"JSON artifact is not an object: {path}")
    return value


def _strict_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise CompletionPreflightError(f"missing journal: {path}")
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise CompletionPreflightError(
                    f"torn journal {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict) or not isinstance(value.get(key), str):
                raise CompletionPreflightError(
                    f"journal row lacks {key}: {path}:{line_number}"
                )
            identity = str(value[key])
            if identity in rows:
                raise CompletionPreflightError(
                    f"duplicate {key} {identity!r} in {path}"
                )
            rows[identity] = value
    return rows


def _read_optional_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return _strict_jsonl(path, key) if path.is_file() else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reconcile_invalidated_soak_exposure(
    directory: Path,
    *,
    expected_selected_cells: Sequence[str],
    expected_attempt_label: str,
    expected_prior_cost_usd: float,
) -> float:
    """Conservatively settle an unreferenced child invalidated by pairfix-v1."""

    plan = _strict_json(directory / "plan.json")
    manifest = _strict_json(directory / "manifest.json")
    plan_sha256 = plan.get("plan_sha256")
    unhashed_plan = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        plan.get("schema_version") != "do_direct_soak_plan_v1"
        or manifest.get("schema_version") != "do_direct_soak_campaign_v1"
        or not isinstance(plan_sha256, str)
        or plan_sha256
        != hashlib.sha256(canonical_json(unhashed_plan).encode("utf-8")).hexdigest()
        or manifest.get("plan_sha256") != plan_sha256
        or plan.get("completion_attempt_label") != expected_attempt_label
        or sorted(plan.get("selected_cells") or ()) != sorted(expected_selected_cells)
        or not math.isclose(
            float(plan.get("prior_cost_usd") or 0.0),
            expected_prior_cost_usd,
            rel_tol=0,
            abs_tol=1e-9,
        )
    ):
        raise CompletionPreflightError(
            "invalidated completion soak identity does not reconcile"
        )
    campaign_id = manifest.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise CompletionPreflightError("invalidated completion soak lacks campaign ID")
    reservations = _strict_jsonl(directory / "reservations.jsonl", "request_id")
    requests = _strict_jsonl(directory / "requests.jsonl", "request_id")
    if not set(requests).issubset(reservations):
        raise CompletionPreflightError(
            "invalidated completion soak has an unreserved terminal request"
        )
    settled = 0.0
    for request_id, row in requests.items():
        reservation = reservations[request_id]
        if (
            row.get("schema_version") != "do_direct_soak_request_v1"
            or reservation.get("schema_version") != "do_direct_reservation_v1"
            or row.get("campaign_id") != campaign_id
            or reservation.get("campaign_id") != campaign_id
            or row.get("plan_sha256") != plan_sha256
        ):
            raise CompletionPreflightError(
                "invalidated completion soak request identity does not reconcile"
            )
        accounted = row.get("accounted_cost_usd")
        if (
            not isinstance(accounted, (int, float))
            or not math.isfinite(accounted)
            or accounted < 0
        ):
            raise CompletionPreflightError(
                "invalidated completion soak terminal cost is invalid"
            )
        settled += float(accounted)
    orphan_reserved = 0.0
    for request_id, row in reservations.items():
        if request_id in requests:
            continue
        if (
            row.get("schema_version") != "do_direct_reservation_v1"
            or row.get("campaign_id") != campaign_id
        ):
            raise CompletionPreflightError(
                "invalidated completion soak orphan identity does not reconcile"
            )
        reserved = row.get("reserved_cost_usd")
        if (
            not isinstance(reserved, (int, float))
            or not math.isfinite(reserved)
            or reserved < 0
        ):
            raise CompletionPreflightError(
                "invalidated completion soak orphan cost is invalid"
            )
        orphan_reserved += float(reserved)
    exposure = expected_prior_cost_usd + settled + orphan_reserved
    if exposure > float(plan.get("max_cost_usd") or 0.0) + 1e-9:
        raise CompletionPreflightError(
            "invalidated completion soak exposure exceeds its frozen cap"
        )
    return exposure


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def attempt_request_id(semantic_id: str, attempt_index: int) -> str:
    return stable_hash(
        {"semantic_id": semantic_id, "attempt_index": attempt_index},
        prefix="do-completion-request-",
    )


def _controlled_output_task(model_id: str, max_tokens: int, seed: int) -> BenchmarkTask:
    # Leave headroom for tokenizer/model variance while still requiring a long
    # realized output.  Requested limit acceptance and realized completion
    # usage are reported separately.
    target_words = max(32, math.floor(max_tokens * 0.70))
    marker = stable_hash(
        {"model_id": model_id, "max_tokens": max_tokens, "seed": seed},
        prefix="FIN-",
    )
    return BenchmarkTask(
        task_id=f"completion-output-{model_id}-{max_tokens}",
        family="completion_realized_output_envelope",
        context_bucket="short",
        output_bucket=str(max_tokens),
        messages=[
            {
                "role": "user",
                "content": (
                    f"UNCACHED-{marker} Write exactly {target_words} space-separated words. Every word "
                    "except the final word must be `azure`. The final word must be "
                    f"`{marker}`. Do not add punctuation or other text."
                ),
            }
        ],
        expected={"kind": "controlled_words", "count": target_words, "marker": marker},
        metadata={
            "planned_input_tokens": 80,
            "planned_output_words": target_words,
            "requested_output_limit": max_tokens,
            "cache_intent": "deliberately_uncached_early_nonce",
        },
    )


def _cache_probe_tasks(model_id: str, seed: int) -> tuple[BenchmarkTask, ...]:
    """Return one miss-intended request and one identical-prefix warm/hit pair."""

    shared_prefix = " ".join(f"cacheword{index % 97}" for index in range(2_048))
    miss_nonce = stable_hash(
        {"model_id": model_id, "seed": seed, "lane": "cache-miss"},
        prefix="UNCACHED-",
    )
    miss_prompt = (
        f"{miss_nonce} {shared_prefix}\nReturn exactly the word MISS followed by 7."
    )
    shared_prompt = (
        f"CACHEABLE-PREFIX {shared_prefix}\nReturn exactly the word HIT followed by 7."
    )
    rows: list[BenchmarkTask] = []
    for label, prompt, intent in (
        ("uncached", miss_prompt, "deliberately_uncached_early_nonce"),
        ("warmup", shared_prompt, "cache_warmup_identical_prefix"),
        ("repeat", shared_prompt, "cache_repeat_identical_prefix"),
    ):
        rows.append(
            BenchmarkTask(
                task_id=f"completion-cache-{model_id}-{label}",
                family="completion_cache_observation",
                context_bucket="cache-2k-words",
                output_bucket="short",
                messages=[{"role": "user", "content": prompt}],
                expected={"kind": "contains_all", "required": ["7"]},
                metadata={
                    "planned_input_tokens": 3_000,
                    "cache_intent": intent,
                    "cache_sequence_index": {"uncached": 0, "warmup": 1, "repeat": 2}[
                        label
                    ],
                },
            )
        )
    return tuple(rows)


def _semantic_id(
    *,
    lane: str,
    model_id: str,
    source_request_id: str | None,
    probe_id: str,
    payload_sha256: str,
) -> str:
    return stable_hash(
        {
            "lane": lane,
            "model_id": model_id,
            "source_request_id": source_request_id,
            "probe_id": probe_id,
            "payload_sha256": payload_sha256,
        },
        prefix="do-completion-probe-",
    )


def _payload_sha(task: BenchmarkTask, max_tokens: int) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "messages": task.messages,
                "tools": task.tools,
                "tool_choice": task.tool_choice,
                "response_format": task.response_format,
                "parameters": task.parameters,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
    ).hexdigest()


def build_completion_probes(config: CompletionConfig) -> list[CompletionProbe]:
    """Reconstruct only inconclusive source cells plus explicit output anchors."""

    probes: list[CompletionProbe] = []

    capability_manifest = _strict_json(config.capability_dir / "manifest.json")
    capability_rows = _strict_jsonl(
        config.capability_dir / "records.jsonl", "request_id"
    )
    capability_cells = build_capability_cells(
        config.model_ids, int(capability_manifest.get("seed") or 20260823)
    )
    cells_by_id = {cell.request_id: cell for cell in capability_cells}

    def source_capability_is_conclusive(
        row: Mapping[str, Any], cell: CapabilityCell
    ) -> bool:
        if row.get("coverage_conclusive") is True:
            return True
        deliberately_malformed = (
            cell.bindings.get("malformed") is True
            or cell.bindings.get("malformed_case") is not None
        )
        status = row.get("http_status")
        # A stable 4xx is the expected, conclusive result for a deliberately
        # malformed negative-validation probe.  It does not establish that the
        # positive capability is unsupported.
        return deliberately_malformed and status in {400, 415, 422}

    unresolved_ids = {
        request_id
        for request_id, row in capability_rows.items()
        if request_id in cells_by_id
        and not source_capability_is_conclusive(row, cells_by_id[request_id])
        and cells_by_id[request_id].provider_send_expected
    }
    # A concurrency interaction is one experimental unit.  If one member was
    # inconclusive, re-run every member together with fresh request IDs.
    unresolved_groups = {
        str(cells_by_id[item].bindings.get("concurrency_group"))
        for item in unresolved_ids
        if cells_by_id[item].bindings.get("concurrency_group")
    }
    for cell in capability_cells:
        group = cell.bindings.get("concurrency_group")
        if (
            cell.request_id not in unresolved_ids
            and str(group) not in unresolved_groups
        ):
            continue
        payload_sha = _payload_sha(cell.task, cell.max_output_tokens)
        semantic = _semantic_id(
            lane="capability_retry",
            model_id=cell.model_id,
            source_request_id=cell.request_id,
            probe_id=cell.probe_id,
            payload_sha256=payload_sha,
        )
        probes.append(
            CompletionProbe(
                semantic_id=semantic,
                lane="capability_retry",
                model_id=cell.model_id,
                source_request_id=cell.request_id,
                source_probe_id=cell.probe_id,
                task=cell.task,
                max_output_tokens=cell.max_output_tokens,
                group_id=str(group) if group else None,
                group_size=int(cell.bindings.get("target_concurrency") or 1),
                estimated_tokens=int(
                    cell.task.metadata.get("planned_input_tokens") or 512
                )
                + cell.max_output_tokens,
            )
        )

    context_manifest = _strict_json(config.context_dir / "manifest.json")
    context_summary = _strict_json(config.context_dir / "summary.json")
    context_rows = _strict_jsonl(config.context_dir / "requests.jsonl", "request_id")
    context_plan = _strict_jsonl(config.context_dir / "plan.jsonl", "request_id")
    chars_by_model = {
        model_id: float(
            (context_summary.get("models") or {})
            .get(model_id, {})
            .get("calibrated_chars_per_token")
            or DEFAULT_CHARS_PER_TOKEN
        )
        for model_id in config.model_ids
    }
    max_payload_bytes = int(
        context_manifest.get("max_payload_bytes") or 8 * 1024 * 1024
    )
    for source_request_id, row in context_rows.items():
        if row.get("coverage_conclusive") is True:
            continue
        plan_row = context_plan.get(source_request_id)
        if plan_row is None or plan_row.get("model_id") not in config.model_ids:
            continue
        source_probe = _probe_from_plan_row(plan_row)
        task, _ = build_retrieval_task(
            source_probe,
            chars_per_token=chars_by_model[source_probe.model_id],
            max_payload_bytes=max_payload_bytes,
        )
        payload_sha = _payload_sha(task, source_probe.requested_max_output_tokens)
        probes.append(
            CompletionProbe(
                semantic_id=_semantic_id(
                    lane="context_retry",
                    model_id=source_probe.model_id,
                    source_request_id=source_request_id,
                    probe_id=source_probe.probe_id,
                    payload_sha256=payload_sha,
                ),
                lane="context_retry",
                model_id=source_probe.model_id,
                source_request_id=source_request_id,
                source_probe_id=source_probe.probe_id,
                task=task,
                max_output_tokens=source_probe.requested_max_output_tokens,
                estimated_tokens=(
                    source_probe.estimated_target_prompt_tokens
                    + source_probe.requested_max_output_tokens
                ),
            )
        )

    for model_id in config.model_ids:
        for max_tokens in config.output_token_anchors:
            task = _controlled_output_task(model_id, max_tokens, config.seed)
            payload_sha = _payload_sha(task, max_tokens)
            probes.append(
                CompletionProbe(
                    semantic_id=_semantic_id(
                        lane="realized_output",
                        model_id=model_id,
                        source_request_id=None,
                        probe_id=task.task_id,
                        payload_sha256=payload_sha,
                    ),
                    lane="realized_output",
                    model_id=model_id,
                    source_request_id=None,
                    source_probe_id=task.task_id,
                    task=task,
                    max_output_tokens=max_tokens,
                    estimated_tokens=80 + max_tokens,
                )
            )

        for cache_task in _cache_probe_tasks(model_id, config.seed):
            max_tokens = 32
            payload_sha = _payload_sha(cache_task, max_tokens)
            label = str(cache_task.metadata["cache_intent"])
            probes.append(
                CompletionProbe(
                    semantic_id=_semantic_id(
                        lane="cache_observation",
                        model_id=model_id,
                        source_request_id=None,
                        probe_id=cache_task.task_id,
                        payload_sha256=payload_sha,
                    ),
                    lane="cache_observation",
                    model_id=model_id,
                    source_request_id=None,
                    source_probe_id=label,
                    task=cache_task,
                    max_output_tokens=max_tokens,
                    estimated_tokens=3_032,
                )
            )

    # The randomized order is frozen in the plan.  Concurrency groups remain
    # grouped by the execution-unit builder below.
    random.Random(config.seed).shuffle(probes)
    return probes


def unresolved_soak_cells(soak_summary: Mapping[str, Any]) -> tuple[str, ...]:
    unresolved: list[str] = []
    for row in soak_summary.get("cells") or []:
        if not isinstance(row, Mapping):
            continue
        passed = (
            row.get("scientifically_complete") is True
            and row.get("two_minute_observed_acceptance_pass") is True
            and row.get("post_soak_recovery_predeclared_pass") is True
        )
        if not passed:
            unresolved.append(f"{row.get('model_id')}:{row.get('shape')}")
    return tuple(sorted(set(unresolved)))


def _retryable(row: Mapping[str, Any]) -> bool:
    status = row.get("http_status")
    if isinstance(status, int) and status in RETRYABLE_HTTP:
        return True
    return str(row.get("error_type") or "") in {
        "TimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "ConnectError",
        "ReadError",
    }


def _retry_after_seconds(error: BaseException) -> float | None:
    raw = getattr(error, "retry_after", None)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


class DirectCompletionCampaign:
    def __init__(self, config: CompletionConfig) -> None:
        config.validate()
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lease_path = self.output_dir / ".execution.lock"
        self.probes = build_completion_probes(config)
        self.source_soak_summary = _strict_json(config.soak_dir / "summary.json")
        self.initial_unresolved_soak_cells = unresolved_soak_cells(
            self.source_soak_summary
        )
        source_exposures = {
            "soak": float(
                self.source_soak_summary.get("conservative_exposure_usd") or 0
            ),
            "context": float(
                _strict_json(config.context_dir / "summary.json").get(
                    "conservative_exposure_usd"
                )
                or 0
            ),
            "capability": float(
                _strict_json(config.capability_dir / "summary.json").get(
                    "conservative_exposure_usd"
                )
                or 0
            ),
        }
        if config.prior_cost_usd + 1e-12 < max(source_exposures.values()):
            raise CompletionPreflightError(
                "--prior-cost-usd is below an authoritative source campaign exposure"
            )
        self.source_contract = {
            "source_exposures_usd": source_exposures,
            "soak_summary_sha256": _sha256_file(config.soak_dir / "summary.json"),
            "context_summary_sha256": _sha256_file(config.context_dir / "summary.json"),
            "context_plan_sha256": _sha256_file(config.context_dir / "plan.jsonl"),
            "capability_summary_sha256": _sha256_file(
                config.capability_dir / "summary.json"
            ),
            "capability_plan_sha256": _sha256_file(
                config.capability_dir / "plan.jsonl"
            ),
        }
        plan_rows = [probe.plan_row(config.max_attempts) for probe in self.probes]
        self.plan_identity = {
            "schema_version": "do_direct_completion_plan_v1",
            "source_contract": self.source_contract,
            "models": list(config.model_ids),
            "seed": config.seed,
            "prior_cost_usd": config.prior_cost_usd,
            "max_cost_usd": config.max_cost_usd,
            "launch_stop_cost_usd": config.launch_stop_cost_usd,
            "drain_reserve_usd": config.max_cost_usd - config.launch_stop_cost_usd,
            "duration_hours": config.duration_hours,
            "absolute_hard_deadline": (
                config.absolute_hard_deadline.astimezone(timezone.utc).isoformat()
                if config.absolute_hard_deadline is not None
                else None
            ),
            "send_reserve_minutes": config.send_reserve_minutes,
            "max_attempts": config.max_attempts,
            "retry_backoff_seconds": config.retry_backoff_seconds,
            "rate_ladder": list(config.rate_ladder),
            "output_token_anchors": list(config.output_token_anchors),
            "initial_unresolved_soak_cells": list(self.initial_unresolved_soak_cells),
            "probes": plan_rows,
        }
        self.plan_sha256 = hashlib.sha256(
            canonical_json(self.plan_identity).encode("utf-8")
        ).hexdigest()
        self.campaign_id = f"do-completion-{self.plan_sha256[:20]}"
        self.requests_path = self.output_dir / "requests.jsonl"
        self.reservations_path = self.output_dir / "reservations.jsonl"
        self.outlier_path = self.output_dir / "outlier-audit.jsonl"
        self.outcomes_path = self.output_dir / "probe-outcomes.jsonl"
        self.waves_path = self.output_dir / "soak-waves.jsonl"
        self.soak_censors_path = self.output_dir / "soak-censors.jsonl"
        self.execution_window_path = self.output_dir / "execution-window.json"
        with OutputDirectoryLease(self.lease_path):
            self._reload()
            self._write_or_validate_plan()

    def _reload(self) -> None:
        self.requests = _read_optional_jsonl(self.requests_path, "request_id")
        self.outliers = _read_optional_jsonl(self.outlier_path, "request_id")
        self.outcomes = _read_optional_jsonl(self.outcomes_path, "semantic_id")
        self.waves = _read_optional_jsonl(self.waves_path, "wave_id")
        self.soak_censors = _read_optional_jsonl(self.soak_censors_path, "censor_id")
        self.reservations = _read_optional_jsonl(self.reservations_path, "request_id")
        self.requests_journal = JsonlJournal(self.requests_path)
        self.outlier_journal = JsonlJournal(self.outlier_path)
        self.outcomes_journal = JsonlJournal(self.outcomes_path)
        self.waves_journal = JsonlJournal(self.waves_path)
        self.soak_censors_journal = JsonlJournal(self.soak_censors_path)
        self.budget = BudgetLedger(
            path=self.reservations_path,
            max_cost_usd=self.config.launch_stop_cost_usd,
            prior_cost_usd=self.config.prior_cost_usd,
            terminal_rows=self.requests,
        )

    def _write_or_validate_plan(self) -> None:
        plan_path = self.output_dir / "plan.json"
        manifest_path = self.output_dir / "manifest.json"
        if plan_path.exists() or manifest_path.exists():
            if not plan_path.is_file() or not manifest_path.is_file():
                raise CompletionPreflightError(
                    "incomplete completion plan/manifest pair"
                )
            if canonical_json(_strict_json(plan_path)) != canonical_json(
                self.plan_identity
            ):
                raise CompletionPreflightError("completion plan changed on resume")
            manifest = _strict_json(manifest_path)
            if (
                manifest.get("campaign_id") != self.campaign_id
                or manifest.get("plan_sha256") != self.plan_sha256
            ):
                raise CompletionPreflightError("completion manifest changed on resume")
            return
        plan_path.write_text(
            json.dumps(self.plan_identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        largest_reservation = max(
            (
                conservative_request_cost(
                    MODEL_BY_ID[probe.model_id], probe.task, probe.max_output_tokens
                )[0]
                for probe in self.probes
            ),
            default=0.0,
        )
        manifest = {
            "schema_version": "do_direct_completion_manifest_v1",
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "created_at": utc_now(),
            "planned_semantic_probes": len(self.probes),
            "planned_attempt_slots": len(self.probes) * self.config.max_attempts,
            "planned_descending_soak_cells": len(self.initial_unresolved_soak_cells),
            "largest_single_probe_reservation_usd": largest_reservation,
            "launch_gate_passes": (
                self.config.prior_cost_usd + largest_reservation
                <= self.config.launch_stop_cost_usd + 1e-12
            ),
            "hard_cap_usd": self.config.max_cost_usd,
            "launch_stop_cost_usd": self.config.launch_stop_cost_usd,
            "drain_reserve_usd": self.config.max_cost_usd
            - self.config.launch_stop_cost_usd,
            "deadline_contract": (
                "One process-local six-hour hard deadline; provider sends stop before "
                "the final drain reserve. Every stage reuses that same absolute cutoff."
            ),
            "no_replay_contract": (
                "Every provider send is preceded by an fsync-backed deterministic "
                "reservation. A reservation without a terminal row is unknown and is "
                "never replayed. Retry attempts have distinct predeclared IDs."
            ),
            "metric_contract": (
                "Raw usage and monotonic request/event timing are request-addressable. "
                "SSE event span is never called decode time; invalid rates are null, and "
                "valid extremes are retained and flagged in outlier-audit.jsonl."
            ),
            "sanitization": (
                "No credentials, prompts, outputs, bodies, reasoning, or raw headers are "
                "persisted; only hashes, numeric measurements, scores, and allowlisted "
                "quota signals are retained."
            ),
            "quality_attribution_contract": (
                "This paid campaign sends DigitalOcean traffic only. Deterministic task "
                "scores measure observed behavior, but no DigitalOcean-specific quality "
                "causation or provider comparison is claimed without a separately funded, "
                "matched external control."
            ),
        }
        if not manifest["launch_gate_passes"]:
            raise CompletionPreflightError(
                "largest completion request cannot fit under the cumulative cap"
            )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    async def _append_request(self, row: dict[str, Any]) -> None:
        request_id = str(row["request_id"])
        if request_id in self.requests:
            return
        await self.requests_journal.append(row)
        self.requests[request_id] = row
        await self.budget.settle(request_id, row)
        projected = audit_row(row)
        await self.outlier_journal.append(projected)
        self.outliers[request_id] = projected

    async def _reconcile_audit(self) -> None:
        for request_id, row in self.requests.items():
            if request_id in self.outliers:
                continue
            projected = audit_row(row)
            await self.outlier_journal.append(projected)
            self.outliers[request_id] = projected

    async def _append_outcome(self, row: dict[str, Any]) -> None:
        semantic_id = str(row["semantic_id"])
        if semantic_id in self.outcomes:
            return
        await self.outcomes_journal.append(row)
        self.outcomes[semantic_id] = row

    async def _record_soak_censors(
        self,
        campaign: DirectSoakCampaign,
        *,
        wave_index: int,
        multiplier: float,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Persist the evidence eligibility partition before a re-soak send.

        A failed all-cell soak preflight is not permission to discard the
        receipt-backed candidates that *are* runnable.  Conversely, a cell
        without an admissible AIMD candidate must never inherit a fallback
        rate.  This method records that distinction durably before the
        eligible child campaign is constructed.
        """

        eligible: list[str] = []
        censored: list[str] = []
        for cell in campaign.cell_plans:
            selector = f"{cell.model_id}:{cell.shape}"
            if cell.status == "ready":
                eligible.append(selector)
                continue
            censored.append(selector)
            censor_id = stable_hash(
                {
                    "campaign_id": self.campaign_id,
                    "endpoint_shape": selector,
                    "first_blocked_wave_index": wave_index,
                    "candidate_rate_multiplier": multiplier,
                    "blocked_status": cell.status,
                    "blocked_reason": cell.blocked_reason,
                },
                prefix="do-completion-soak-censor-",
            )
            if censor_id in self.soak_censors:
                continue
            row = {
                "schema_version": SOAK_CENSOR_SCHEMA,
                "campaign_id": self.campaign_id,
                "censor_id": censor_id,
                "endpoint_shape": selector,
                "model_id": cell.model_id,
                "shape": cell.shape,
                "wave_index": wave_index,
                "candidate_rate_multiplier": multiplier,
                "status": "censored_ineligible_aimd_prerequisite",
                "blocked_status": cell.status,
                "blocked_reason": cell.blocked_reason,
                "candidate_rate_rps": cell.candidate_rate_rps,
                "claim_boundary": (
                    "No descending re-soak was launched for this cell because the "
                    "source AIMD evidence did not satisfy the frozen candidate or "
                    "quality-pair scheduling prerequisite. The cell remains unresolved."
                ),
                "recorded_at": utc_now(),
            }
            await self.soak_censors_journal.append(row)
            self.soak_censors[censor_id] = row
        return tuple(sorted(eligible)), tuple(sorted(censored))

    @staticmethod
    def _child_plan_prior_cost(output_dir: Path, current_exposure: float) -> float:
        """Reuse the exact frozen prior on a child-campaign resume.

        Reconstructing a settled exposure can differ from its originally
        serialized IEEE-754 value by a few ulps.  That is economically
        immaterial but it changes the child plan hash.  An existing child plan
        therefore supplies the exact serialized prior after a tight numerical
        reconciliation; new child plans continue to use the live ledger.
        """

        plan_path = output_dir / "plan.json"
        if not plan_path.exists():
            return current_exposure
        plan = _strict_json(plan_path)
        stored = plan.get("prior_cost_usd")
        if isinstance(stored, bool) or not isinstance(stored, (int, float)):
            raise CompletionPreflightError(
                "existing child soak plan has an invalid prior-cost value"
            )
        stored_value = float(stored)
        if not math.isfinite(stored_value) or not math.isclose(
            stored_value,
            current_exposure,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise CompletionPreflightError(
                "existing child soak prior cost does not reconcile with the live ledger"
            )
        return stored_value

    def _deadline_reached(self, cutoff: datetime) -> bool:
        return datetime.now(timezone.utc) >= cutoff

    def _execution_window(self) -> tuple[datetime, datetime, datetime]:
        if self.execution_window_path.is_file():
            row = _strict_json(self.execution_window_path)
            if row.get("campaign_id") != self.campaign_id:
                raise CompletionPreflightError(
                    "execution window belongs to another campaign"
                )
            try:
                started = datetime.fromisoformat(str(row["started_at"]))
                cutoff = datetime.fromisoformat(str(row["send_cutoff"]))
                hard_deadline = datetime.fromisoformat(str(row["hard_deadline"]))
            except (KeyError, ValueError) as error:
                raise CompletionPreflightError(
                    "invalid durable execution window"
                ) from error
            if any(item.tzinfo is None for item in (started, cutoff, hard_deadline)):
                raise CompletionPreflightError(
                    "execution window timestamps need UTC offsets"
                )
            return started, cutoff, hard_deadline
        started = datetime.now(timezone.utc)
        relative_deadline = started + timedelta(hours=self.config.duration_hours)
        hard_deadline = min(
            relative_deadline,
            (
                self.config.absolute_hard_deadline.astimezone(timezone.utc)
                if self.config.absolute_hard_deadline is not None
                else relative_deadline
            ),
        )
        cutoff = hard_deadline - timedelta(minutes=self.config.send_reserve_minutes)
        _atomic_json_write(
            self.execution_window_path,
            {
                "schema_version": "do_direct_completion_execution_window_v1",
                "campaign_id": self.campaign_id,
                "started_at": started.isoformat(),
                "send_cutoff": cutoff.isoformat(),
                "hard_deadline": hard_deadline.isoformat(),
                "resume_contract": (
                    "All restarts reuse this original window; a process restart never "
                    "extends the authorized six-hour campaign."
                ),
            },
        )
        return started, cutoff, hard_deadline

    def _result_row(
        self,
        *,
        probe: CompletionProbe,
        request_id: str,
        attempt_index: int,
        result: StreamResult,
        reserved_cost: float,
        reserved_prompt_tokens: int,
        started_at: str,
        ended_at: str,
        started_ns: int,
        ended_ns: int,
    ) -> dict[str, Any]:
        usage = parse_token_usage(result.usage)
        prompt_complete = usage.get("prompt_tokens", 0) > 0
        output_complete = usage.get("completion_tokens", 0) > 0
        complete = prompt_complete and output_complete
        spec = MODEL_BY_ID[probe.model_id]
        actual_cost = (
            (
                usage.get("prompt_tokens", 0) * spec.input_usd_per_million
                + usage.get("completion_tokens", 0) * spec.output_usd_per_million
            )
            / 1_000_000
            if complete
            else None
        )
        quality = score_result(probe.task, result)
        timing = timing_evidence(
            result,
            monotonic_started_ns=started_ns,
            monotonic_ended_ns=ended_ns,
            intended_cache_state=str(
                probe.task.metadata.get("cache_intent") or "unknown"
            ),
            sequence_count=int(probe.task.parameters.get("n") or 1),
            streaming=probe.task.parameters.get("stream") is not False,
        )
        if probe.lane == "context_retry":
            coverage_conclusive = prompt_complete
        elif probe.lane == "realized_output":
            coverage_conclusive = output_complete
        else:
            coverage_conclusive = True
        response_sha = hashlib.sha256(
            canonical_json(
                {
                    "text": result.text,
                    "reasoning": result.reasoning_text,
                    "tool_calls": result.tool_calls,
                }
            ).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": "do_direct_completion_request_v1",
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "request_id": request_id,
            "semantic_id": probe.semantic_id,
            "attempt_index": attempt_index,
            "provider": "digitalocean-serverless-inference",
            "model_id": probe.model_id,
            "shape": probe.lane,
            "phase": "completion_probe",
            "source_request_id": probe.source_request_id,
            "source_probe_id": probe.source_probe_id,
            "provider_send_attempted": True,
            "started_at": started_at,
            "ended_at": ended_at,
            "status": "success",
            "http_status": result.status_code,
            "coverage_conclusive": coverage_conclusive,
            "scientific_success": complete,
            "functional_valid": float(quality.get("quality_score") or 0) >= 0.999999,
            "quality_score": float(quality.get("quality_score") or 0),
            "score_kind": str(quality.get("score_kind") or "unknown"),
            "finish_reason": result.finish_reason,
            "usage": usage,
            "prompt_usage_complete": prompt_complete,
            "completion_usage_complete": output_complete,
            "usage_complete_for_settlement": complete,
            "cache_observation": cache_observation(
                result.usage,
                intended_state=str(
                    probe.task.metadata.get("cache_intent") or "unknown"
                ),
            ),
            "timing": timing,
            "stream": {
                "event_count": result.event_count,
                "first_event_kind": result.first_event_kind,
            },
            "header_signals": sanitized_header_signals(result.response_headers),
            "response_sha256": response_sha,
            "response_text_bytes": len(result.text.encode("utf-8")),
            "reasoning_text_bytes": len(result.reasoning_text.encode("utf-8")),
            "tool_call_count": len(result.tool_calls),
            "requested_max_output_tokens": probe.max_output_tokens,
            "realized_output_fraction_of_requested": (
                usage["completion_tokens"] / probe.max_output_tokens
                if output_complete and probe.max_output_tokens > 0
                else None
            ),
            "worst_case_reserved_cost_usd": reserved_cost,
            "reserved_prompt_tokens": reserved_prompt_tokens,
            "estimated_cost_usd": actual_cost,
            "accounted_cost_usd": actual_cost
            if actual_cost is not None
            else reserved_cost,
            "retryable": False,
        }

    def _failure_result_row(
        self,
        *,
        probe: CompletionProbe,
        request_id: str,
        attempt_index: int,
        error: BaseException,
        reserved_cost: float,
        reserved_prompt_tokens: int,
        started_at: str,
        ended_at: str,
        started_ns: int,
        ended_ns: int,
    ) -> dict[str, Any]:
        http_status = getattr(error, "status_code", None)
        if probe.lane == "context_retry":
            classification, conclusive, reason_category, reason_sha = classify_failure(
                error
            )
        else:
            classification, reason_category, reason_sha = _sanitized_error_evidence(
                error
            )
            conclusive = _coverage_conclusive(classification)
        row = {
            "schema_version": "do_direct_completion_request_v1",
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "request_id": request_id,
            "semantic_id": probe.semantic_id,
            "attempt_index": attempt_index,
            "provider": "digitalocean-serverless-inference",
            "model_id": probe.model_id,
            "shape": probe.lane,
            "phase": "completion_probe",
            "source_request_id": probe.source_request_id,
            "source_probe_id": probe.source_probe_id,
            "provider_send_attempted": True,
            "started_at": started_at,
            "ended_at": ended_at,
            "status": "error",
            "coverage_classification": classification,
            "coverage_conclusive": conclusive,
            "http_status": http_status if isinstance(http_status, int) else None,
            "error_type": type(error).__name__,
            "retry_after_seconds": _retry_after_seconds(error),
            "provider_reason_category": reason_category,
            "provider_reason_sha256": reason_sha,
            "usage": {},
            "timing": {
                "request_seconds": (ended_ns - started_ns) / 1_000_000_000,
                "ttft_seconds": None,
                "monotonic_timestamps_ns": {
                    "request_started_ns": started_ns,
                    "request_ended_ns": ended_ns,
                },
                "metric_audit_classification": "not_applicable_failure",
                "timing_invalidity_reasons": [],
                "usage_invalidity_reasons": [
                    "missing_or_nonpositive_prompt_tokens",
                    "missing_or_nonpositive_completion_tokens",
                ],
                "sse_chunk_span_invalidity_reasons": ["no_successful_stream"],
                "extreme_metric_triggers": [],
            },
            "quality_score": 0.0,
            "score_kind": str(probe.task.expected.get("kind") or "unknown"),
            "requested_max_output_tokens": probe.max_output_tokens,
            "worst_case_reserved_cost_usd": reserved_cost,
            "reserved_prompt_tokens": reserved_prompt_tokens,
            "estimated_cost_usd": None,
            "accounted_cost_usd": reserved_cost,
        }
        row["retryable"] = _retryable(row)
        return row

    async def _run_attempt(
        self,
        executor: RequestExecutor,
        probe: CompletionProbe,
        attempt_index: int,
        *,
        semaphore: asyncio.Semaphore,
        model_lock: asyncio.Lock | None,
        cutoff: datetime,
    ) -> dict[str, Any]:
        request_id = attempt_request_id(probe.semantic_id, attempt_index)
        existing = self.requests.get(request_id)
        if existing is not None:
            return existing
        if request_id in self.budget.reservations:
            return {
                "request_id": request_id,
                "semantic_id": probe.semantic_id,
                "status": "unknown_prior_reservation",
                "provider_send_attempted": True,
                "retryable": False,
            }
        if self._deadline_reached(cutoff):
            return {
                "request_id": request_id,
                "semantic_id": probe.semantic_id,
                "status": "skipped_deadline",
                "provider_send_attempted": False,
                "retryable": False,
            }
        lock = model_lock or asyncio.Lock()
        async with semaphore, lock:
            if self._deadline_reached(cutoff):
                return {
                    "request_id": request_id,
                    "semantic_id": probe.semantic_id,
                    "status": "skipped_deadline",
                    "provider_send_attempted": False,
                    "retryable": False,
                }
            reserved_cost, reserved_tokens = conservative_request_cost(
                MODEL_BY_ID[probe.model_id], probe.task, probe.max_output_tokens
            )
            reserved = await self.budget.reserve(
                campaign_id=self.campaign_id,
                request_id=request_id,
                epoch_id=probe.semantic_id,
                model_id=probe.model_id,
                shape=probe.lane,
                reserved_cost_usd=reserved_cost,
                reserved_prompt_tokens=reserved_tokens,
                max_output_tokens=probe.max_output_tokens,
            )
            if not reserved:
                return {
                    "request_id": request_id,
                    "semantic_id": probe.semantic_id,
                    "status": "skipped_budget_cap",
                    "provider_send_attempted": False,
                    "retryable": False,
                }
            started_at = utc_now()
            started_ns = time.perf_counter_ns()
            try:
                remaining = max(
                    0.0, (cutoff - datetime.now(timezone.utc)).total_seconds()
                )
                if remaining <= 0:
                    raise asyncio.TimeoutError("completion send cutoff reached")
                result = await asyncio.wait_for(
                    executor(probe.model_id, probe.task, probe.max_output_tokens),
                    timeout=min(self.config.request_timeout_seconds, remaining),
                )
                ended_ns = time.perf_counter_ns()
                row = self._result_row(
                    probe=probe,
                    request_id=request_id,
                    attempt_index=attempt_index,
                    result=result,
                    reserved_cost=reserved_cost,
                    reserved_prompt_tokens=reserved_tokens,
                    started_at=started_at,
                    ended_at=utc_now(),
                    started_ns=started_ns,
                    ended_ns=ended_ns,
                )
            except asyncio.CancelledError as error:
                ended_ns = time.perf_counter_ns()
                row = self._failure_result_row(
                    probe=probe,
                    request_id=request_id,
                    attempt_index=attempt_index,
                    error=error,
                    reserved_cost=reserved_cost,
                    reserved_prompt_tokens=reserved_tokens,
                    started_at=started_at,
                    ended_at=utc_now(),
                    started_ns=started_ns,
                    ended_ns=ended_ns,
                )
                await self._append_request(row)
                raise
            except Exception as error:
                ended_ns = time.perf_counter_ns()
                row = self._failure_result_row(
                    probe=probe,
                    request_id=request_id,
                    attempt_index=attempt_index,
                    error=error,
                    reserved_cost=reserved_cost,
                    reserved_prompt_tokens=reserved_tokens,
                    started_at=started_at,
                    ended_at=utc_now(),
                    started_ns=started_ns,
                    ended_ns=ended_ns,
                )
            await self._append_request(row)
            return row

    async def _run_probe_unit(
        self,
        executor: RequestExecutor,
        unit: Sequence[CompletionProbe],
        *,
        semaphore: asyncio.Semaphore,
        model_locks: Mapping[str, asyncio.Lock],
        cutoff: datetime,
    ) -> None:
        if all(probe.semantic_id in self.outcomes for probe in unit):
            return
        last_rows: list[dict[str, Any]] = []
        for attempt_index in range(self.config.max_attempts):
            rows = await asyncio.gather(
                *(
                    self._run_attempt(
                        executor,
                        probe,
                        attempt_index,
                        semaphore=semaphore,
                        model_lock=(
                            None if len(unit) > 1 else model_locks[probe.model_id]
                        ),
                        cutoff=cutoff,
                    )
                    for probe in unit
                )
            )
            last_rows = list(rows)
            if not any(_retryable(row) for row in rows):
                break
            if attempt_index + 1 >= self.config.max_attempts:
                break
            delay = self.config.retry_backoff_seconds * (2**attempt_index)
            retry_after = max(
                (float(row.get("retry_after_seconds") or 0) for row in rows),
                default=0.0,
            )
            delay = max(delay, retry_after)
            if datetime.now(timezone.utc) + timedelta(seconds=delay) >= cutoff:
                break
            await asyncio.sleep(delay)
        by_semantic = {row.get("semantic_id"): row for row in last_rows}
        for probe in unit:
            row = by_semantic.get(probe.semantic_id) or {}
            await self._append_outcome(
                {
                    "schema_version": "do_direct_completion_probe_outcome_v1",
                    "campaign_id": self.campaign_id,
                    "semantic_id": probe.semantic_id,
                    "lane": probe.lane,
                    "model_id": probe.model_id,
                    "source_request_id": probe.source_request_id,
                    "source_probe_id": probe.source_probe_id,
                    "status": row.get("status", "incomplete"),
                    "coverage_conclusive": row.get("coverage_conclusive"),
                    "functional_valid": row.get("functional_valid"),
                    "final_request_id": row.get("request_id"),
                    "completed_at": utc_now(),
                }
            )

    def _probe_units(self) -> list[list[CompletionProbe]]:
        groups: dict[str, list[CompletionProbe]] = {}
        singles: list[list[CompletionProbe]] = []
        for probe in self.probes:
            if probe.lane == "cache_observation":
                continue
            if probe.group_id:
                groups.setdefault(probe.group_id, []).append(probe)
            else:
                singles.append([probe])
        units = singles + list(groups.values())
        random.Random(self.config.seed).shuffle(units)
        return units

    async def _run_generic_probes(
        self, executor: RequestExecutor, *, cutoff: datetime
    ) -> None:
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        model_locks = {model_id: asyncio.Lock() for model_id in self.config.model_ids}
        await asyncio.gather(
            *(
                self._run_probe_unit(
                    executor,
                    unit,
                    semaphore=semaphore,
                    model_locks=model_locks,
                    cutoff=cutoff,
                )
                for unit in self._probe_units()
            )
        )
        # Cache observation is deliberately ordered per model: a nonce-busted
        # miss-intended request, then two byte-identical prefix requests.  The
        # model chains may run concurrently, but order within a chain is fixed.
        cache_by_model: dict[str, list[CompletionProbe]] = {}
        for probe in self.probes:
            if probe.lane == "cache_observation":
                cache_by_model.setdefault(probe.model_id, []).append(probe)

        async def run_cache_chain(model_id: str, chain: list[CompletionProbe]) -> None:
            chain.sort(key=lambda item: int(item.task.metadata["cache_sequence_index"]))
            for probe in chain:
                if self._deadline_reached(cutoff):
                    break
                await self._run_probe_unit(
                    executor,
                    [probe],
                    semaphore=semaphore,
                    model_locks=model_locks,
                    cutoff=cutoff,
                )

        await asyncio.gather(
            *(
                run_cache_chain(model_id, chain)
                for model_id, chain in cache_by_model.items()
            )
        )

    @staticmethod
    def _wave_passed(row: Mapping[str, Any]) -> bool:
        return (
            row.get("scientifically_complete") is True
            and row.get("two_minute_observed_acceptance_pass") is True
            and row.get("post_soak_recovery_predeclared_pass") is True
        )

    async def _run_soak_waves(
        self, executor: RequestExecutor, *, cutoff: datetime, hard_deadline: datetime
    ) -> float:
        unresolved = set(self.initial_unresolved_soak_cells)
        current_exposure = self.budget.exposure_usd
        for wave_index, multiplier in enumerate(self.config.rate_ladder):
            if not unresolved or self._deadline_reached(cutoff):
                break
            wave_id = stable_hash(
                {
                    "campaign_id": self.campaign_id,
                    "wave_index": wave_index,
                    "multiplier": multiplier,
                    "cells": sorted(unresolved),
                },
                prefix="do-completion-soak-wave-",
            )
            existing = self.waves.get(wave_id)
            if existing is not None:
                unresolved = set(existing.get("unresolved_after") or [])
                current_exposure = float(
                    existing.get("conservative_exposure_usd") or current_exposure
                )
                continue
            # The broad child is an offline evidence audit.  It may contain a
            # mixture of ready and blocked cells, so it is never executed.
            # Keeping it at the historical path also makes the recovery
            # compatible with a campaign that already stopped at this exact
            # preflight (and therefore already wrote this plan).
            wave_dir = (
                self.output_dir / "soak-waves" / f"wave-{wave_index}-{multiplier:g}"
            )
            audit_prior_cost = self._child_plan_prior_cost(wave_dir, current_exposure)
            audit_config = SoakConfig(
                aimd_dir=self.config.aimd_dir,
                output_dir=wave_dir,
                model_ids=self.config.model_ids,
                aimd_reconciliation_path=self.config.aimd_reconciliation_path,
                prior_lineage_root=self.config.prior_lineage_root,
                v3_checkpoint_dir=self.config.v3_checkpoint_dir,
                seed=self.config.seed + wave_index,
                soak_seconds=self.config.soak_seconds,
                analysis_block_seconds=self.config.analysis_block_seconds,
                analysis_block_count=4,
                concurrency_ceiling=self.config.soak_concurrency_ceiling,
                quality_pairs_per_cell=4,
                recovery_seconds=self.config.recovery_seconds,
                request_timeout_seconds=self.config.request_timeout_seconds,
                max_cost_usd=self.config.launch_stop_cost_usd,
                prior_cost_usd=audit_prior_cost,
                accept_conditional_prior_exposure_basis=(
                    self.config.accept_conditional_prior_exposure_basis
                ),
                stop_launch_at=cutoff,
                hard_campaign_deadline=hard_deadline,
                selected_cells=tuple(sorted(unresolved)),
                candidate_rate_multiplier=multiplier,
                completion_attempt_label=f"wave-{wave_index}",
            )
            audit_campaign = DirectSoakCampaign(audit_config)
            eligible, censored = await self._record_soak_censors(
                audit_campaign,
                wave_index=wave_index,
                multiplier=multiplier,
            )
            if not eligible:
                # Every remaining cell is explicitly censored in the durable
                # journal.  No fallback rate exists, so later (smaller)
                # multipliers cannot make the frozen evidence more eligible.
                break

            legacy_eligible_digest = hashlib.sha256(
                canonical_json(list(eligible)).encode("utf-8")
            ).hexdigest()[:12]
            invalidated_directory = (
                self.output_dir
                / "soak-waves"
                / (
                    f"wave-{wave_index}-{multiplier:g}-eligible-{legacy_eligible_digest}"
                )
            )
            if invalidated_directory.is_dir():
                current_exposure = _reconcile_invalidated_soak_exposure(
                    invalidated_directory,
                    expected_selected_cells=eligible,
                    expected_attempt_label=f"wave-{wave_index}",
                    expected_prior_cost_usd=current_exposure,
                )

            eligible_digest = hashlib.sha256(
                canonical_json(
                    {
                        "cells": list(eligible),
                        "repair_version": COMPLETION_SOAK_REPAIR_VERSION,
                    }
                ).encode("utf-8")
            ).hexdigest()[:12]
            execution_name = (
                f"wave-{wave_index}-{multiplier:g}-eligible-{eligible_digest}"
            )
            execution_dir = self.output_dir / "soak-waves" / execution_name
            execution_prior_cost = self._child_plan_prior_cost(
                execution_dir, current_exposure
            )
            execution_config = SoakConfig(
                aimd_dir=self.config.aimd_dir,
                output_dir=execution_dir,
                model_ids=self.config.model_ids,
                aimd_reconciliation_path=self.config.aimd_reconciliation_path,
                prior_lineage_root=self.config.prior_lineage_root,
                v3_checkpoint_dir=self.config.v3_checkpoint_dir,
                seed=self.config.seed + wave_index,
                soak_seconds=self.config.soak_seconds,
                analysis_block_seconds=self.config.analysis_block_seconds,
                analysis_block_count=4,
                concurrency_ceiling=self.config.soak_concurrency_ceiling,
                quality_pairs_per_cell=4,
                recovery_seconds=self.config.recovery_seconds,
                request_timeout_seconds=self.config.request_timeout_seconds,
                max_cost_usd=self.config.launch_stop_cost_usd,
                prior_cost_usd=execution_prior_cost,
                accept_conditional_prior_exposure_basis=(
                    self.config.accept_conditional_prior_exposure_basis
                ),
                stop_launch_at=cutoff,
                hard_campaign_deadline=hard_deadline,
                selected_cells=eligible,
                candidate_rate_multiplier=multiplier,
                completion_attempt_label=(
                    f"wave-{wave_index}-{COMPLETION_SOAK_REPAIR_VERSION}"
                ),
            )
            campaign = DirectSoakCampaign(execution_config)
            summary = await campaign.run(executor)
            current_exposure = float(summary["conservative_exposure_usd"])
            passed = {
                f"{row.get('model_id')}:{row.get('shape')}"
                for row in summary.get("cells") or []
                if self._wave_passed(row)
            }
            unresolved -= passed
            wave_row = {
                "schema_version": "do_direct_completion_soak_wave_v1",
                "campaign_id": self.campaign_id,
                "wave_id": wave_id,
                "wave_index": wave_index,
                "candidate_rate_multiplier": multiplier,
                "soak_campaign_id": summary.get("campaign_id"),
                "soak_plan_sha256": summary.get("plan_sha256"),
                "soak_artifact_relative_path": execution_name,
                "attempted_cells": sorted(
                    f"{row.get('model_id')}:{row.get('shape')}"
                    for row in summary.get("cells") or []
                ),
                "censored_cells": list(censored),
                "censor_reasons_by_cell": {
                    f"{cell.model_id}:{cell.shape}": cell.blocked_reason
                    for cell in audit_campaign.cell_plans
                    if cell.status != "ready"
                },
                "passed_cells": sorted(passed),
                "unresolved_after": sorted(unresolved),
                "conservative_exposure_usd": current_exposure,
                "ended_at": utc_now(),
            }
            await self.waves_journal.append(wave_row)
            self.waves[wave_id] = wave_row
            if unresolved and wave_index + 1 < len(self.config.rate_ladder):
                delay = self.config.retry_backoff_seconds * (2**wave_index)
                if datetime.now(timezone.utc) + timedelta(seconds=delay) < cutoff:
                    await asyncio.sleep(delay)
        return current_exposure

    async def _run_locked(self, executor: RequestExecutor) -> dict[str, Any]:
        self._reload()
        await self._reconcile_audit()
        started, cutoff, hard_deadline = self._execution_window()
        await self._run_generic_probes(executor, cutoff=cutoff)
        final_exposure = await self._run_soak_waves(
            executor, cutoff=cutoff, hard_deadline=hard_deadline
        )
        latest_unresolved = set(self.initial_unresolved_soak_cells)
        for row in sorted(
            self.waves.values(), key=lambda item: int(item["wave_index"])
        ):
            latest_unresolved = set(row.get("unresolved_after") or [])
        conclusive_probes = sum(
            row.get("coverage_conclusive") is True for row in self.outcomes.values()
        )
        censored_by_cell: dict[str, dict[str, Any]] = {}
        for row in sorted(
            self.soak_censors.values(), key=lambda item: int(item["wave_index"])
        ):
            censored_by_cell.setdefault(str(row["endpoint_shape"]), row)
        summary = {
            "schema_version": "do_direct_completion_summary_v1",
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "status": (
                "complete"
                if len(self.outcomes) == len(self.probes) and not latest_unresolved
                else "incomplete_or_censored"
            ),
            "started_at": started.isoformat(),
            "ended_at": utc_now(),
            "send_cutoff": cutoff.isoformat(),
            "hard_deadline": hard_deadline.isoformat(),
            "planned_semantic_probes": len(self.probes),
            "terminal_probe_outcomes": len(self.outcomes),
            "conclusive_probe_outcomes": conclusive_probes,
            "request_rows": len(self.requests),
            "outlier_audit_rows": len(self.outliers),
            "soak_waves": len(self.waves),
            "initial_unresolved_soak_cells": list(self.initial_unresolved_soak_cells),
            "remaining_unresolved_soak_cells": sorted(latest_unresolved),
            "censored_soak_cells": sorted(censored_by_cell),
            "censored_soak_cell_details": [
                {
                    "endpoint_shape": selector,
                    "status": row["status"],
                    "blocked_status": row["blocked_status"],
                    "blocked_reason": row["blocked_reason"],
                    "first_blocked_wave_index": row["wave_index"],
                    "candidate_rate_multiplier": row["candidate_rate_multiplier"],
                }
                for selector, row in sorted(censored_by_cell.items())
            ],
            "prior_cost_usd": self.config.prior_cost_usd,
            "conservative_exposure_usd": max(final_exposure, self.budget.exposure_usd),
            "max_cost_usd": self.config.max_cost_usd,
            "launch_stop_cost_usd": self.config.launch_stop_cost_usd,
            "drain_reserve_usd": self.config.max_cost_usd
            - self.config.launch_stop_cost_usd,
            "metric_claim_gate": (
                "Only rows with complete positive usage and internally consistent, "
                "measurable intervals support per-request rate metrics. Aggregate TPM "
                "must use successful tokens over a predeclared wall-clock block."
            ),
            "quality_claim_gate": (
                "No provider-attributed quality conclusion without a matched external "
                "control; this DigitalOcean-only campaign reports task outcomes only."
            ),
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary

    async def run(self, executor: RequestExecutor | None = None) -> dict[str, Any]:
        if executor is not None:
            with OutputDirectoryLease(self.lease_path):
                return await self._run_locked(executor)
        credentials = digitalocean_credentials()
        limits = httpx.Limits(
            max_connections=max(
                self.config.max_concurrency, self.config.soak_concurrency_ceiling
            ),
            max_keepalive_connections=max(
                self.config.max_concurrency, self.config.soak_concurrency_ceiling
            ),
        )
        timeout = httpx.Timeout(
            self.config.request_timeout_seconds,
            connect=min(30.0, self.config.request_timeout_seconds),
            read=self.config.request_timeout_seconds,
        )
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

            async def live_executor(
                model_id: str, task: BenchmarkTask, max_output_tokens: int
            ) -> StreamResult:
                return await stream_chat_completion(
                    client,
                    api_base=credentials["api_base"],
                    api_key=credentials["api_key"],
                    model_id=model_id,
                    task=task,
                    safety_max_output_tokens=max_output_tokens,
                )

            with OutputDirectoryLease(self.lease_path):
                return await self._run_locked(live_executor)


def default_model_ids() -> tuple[str, ...]:
    return DIGITALOCEAN_HOSTED_MODEL_IDS
