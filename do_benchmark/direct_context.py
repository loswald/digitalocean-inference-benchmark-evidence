"""Direct, secret-free DigitalOcean context-window boundary probes.

The campaign runs one sequential adaptive
chain per model and may overlap those chains only behind one shared account
quota governor and one global concurrency ceiling.  The provider-reported
prompt-token count is the measurement x-axis, and only hashes plus numeric or
status metadata are persisted.  Every provider send is preceded by an
fsync-backed worst-case cost reservation, so a restart cannot replay a request
whose outcome is unknown.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

import httpx

from do_benchmark.core import (
    API_DOC_GENERATED_DATE,
    MODEL_BY_ID,
    MODEL_DOC_VERIFIED_DATE,
    MODEL_SPECS,
    DIGITALOCEAN_HOSTED_MODEL_IDS,
    PRICING_DOC_DATE,
    BenchmarkTask,
    JsonlJournal,
    StreamResult,
    require_digitalocean_hosted_models,
    canonical_json,
    score_result,
    stream_chat_completion,
    utc_now,
)
from do_benchmark.credentials import digitalocean_credentials


REQUEST_SCHEMA = "do_direct_context_request_v3"
RESERVATION_SCHEMA = "do_direct_context_reservation_v3"
PLAN_SCHEMA = "do_direct_context_plan_v3"
MANIFEST_SCHEMA = "do_direct_context_manifest_v3"
SUMMARY_SCHEMA = "do_direct_context_summary_v3"
PAYLOAD_BUILDER_CONTRACT_VERSION = "direct_context_payload_builder_v2"
SCORER_CONTRACT_VERSION = "direct_context_exact_retrieval_v2"

# Latest official quota documentation checked when this scheduler contract was
# written. DigitalOcean documents inference quotas as per-account, not
# per-endpoint. Reset values are Unix-epoch refill projections for the size of
# the request just evaluated or rejected, not fixed-window boundaries.
INFERENCE_LIMITS_DOC_URL = (
    "https://docs.digitalocean.com/products/inference/details/limits/"
)
QUOTA_HEADERS_DOC_URL = (
    "https://docs.digitalocean.com/products/inference/reference/"
    "quota-specific-response-headers/"
)
INFERENCE_LIMITS_DOC_VERIFIED_DATE = "2026-08-20"
QUOTA_HEADERS_DOC_VERIFIED_DATE = "2026-07-13"
DEFAULT_FALLBACK_ACCOUNT_RPM = 120.0
DEFAULT_FALLBACK_ACCOUNT_TPM = 500_000.0

CONTEXT_PERCENTAGES: tuple[float, ...] = (
    0.01,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
)
DEFAULT_CHARS_PER_TOKEN = 4.0
MIN_CHARS_PER_TOKEN = 1.0
MAX_CHARS_PER_TOKEN = 8.0
DEFAULT_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
CONTEXT_WINDOW_ANCHOR_SOURCES: dict[str, str] = {
    spec.model_id: (
        "undocumented_probe_anchor"
        if spec.model_id == "kimi-k3"
        else "advertised_official_documentation"
    )
    for spec in MODEL_SPECS
}
ENDPOINT_FREEZE_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "endpoint-freeze.json"
)
_ENDPOINT_FREEZE_BYTES = ENDPOINT_FREEZE_PATH.read_bytes()
ENDPOINT_FREEZE_ARTIFACT_SHA256 = hashlib.sha256(_ENDPOINT_FREEZE_BYTES).hexdigest()
_ENDPOINT_FREEZE = json.loads(_ENDPOINT_FREEZE_BYTES)
FROZEN_ENDPOINT_BY_ID: dict[str, Mapping[str, Any]] = {
    str(row["model_id"]): row for row in _ENDPOINT_FREEZE["endpoints"]
}


def _validate_model_contract_against_freeze(model_id: str) -> None:
    spec = MODEL_BY_ID[model_id]
    frozen = FROZEN_ENDPOINT_BY_ID.get(model_id)
    if frozen is None:
        raise ValueError(f"model absent from frozen endpoint inventory: {model_id}")
    mismatches: list[str] = []
    frozen_context = frozen.get("context_window")
    if model_id == "kimi-k3":
        if frozen_context is not None or spec.context_window != 65_536:
            mismatches.append("Kimi undocumented probe anchor")
    elif frozen_context is None or spec.context_window != int(frozen_context):
        mismatches.append("context_window")
    for field in ("input_usd_per_million", "output_usd_per_million"):
        frozen_value = frozen.get(field)
        spec_value = float(getattr(spec, field))
        if frozen_value is None or not math.isclose(
            spec_value, float(frozen_value), rel_tol=0.0, abs_tol=1e-12
        ):
            mismatches.append(field)
    if mismatches:
        raise ValueError(
            f"frozen endpoint contract mismatch for {model_id}: {', '.join(mismatches)}"
        )


def _model_contract(model_id: str) -> dict[str, Any]:
    spec = MODEL_BY_ID[model_id]
    return {
        "model_id": model_id,
        "context_window_anchor_tokens": spec.context_window,
        "context_window_anchor_source": CONTEXT_WINDOW_ANCHOR_SOURCES[model_id],
        "input_usd_per_million": spec.input_usd_per_million,
        "output_usd_per_million": spec.output_usd_per_million,
    }


def _model_contract_sha256(model_id: str) -> str:
    return hashlib.sha256(
        canonical_json(_model_contract(model_id)).encode("utf-8")
    ).hexdigest()


def _documentation_contract() -> dict[str, Any]:
    return {
        "api_reference_generated": API_DOC_GENERATED_DATE,
        "model_page_verified": MODEL_DOC_VERIFIED_DATE,
        "pricing_page_date": PRICING_DOC_DATE,
        "endpoint_freeze_artifact_sha256": ENDPOINT_FREEZE_ARTIFACT_SHA256,
    }


DOCUMENTATION_CONTRACT_SHA256 = hashlib.sha256(
    canonical_json(_documentation_contract()).encode("utf-8")
).hexdigest()
SCORER_CONTRACT_SHA256 = hashlib.sha256(
    canonical_json(
        {
            "version": SCORER_CONTRACT_VERSION,
            "expected_kind": "exact_text",
            "expected_value_derivation": (
                "uppercase first 24 hex chars of sha256('needle:' + request_id), "
                "prefixed NEEDLE-"
            ),
            "pass_threshold": 0.999999,
        }
    ).encode("utf-8")
).hexdigest()
PAYLOAD_BUILDER_CONTRACT_SHA256 = hashlib.sha256(
    canonical_json(
        {
            "version": PAYLOAD_BUILDER_CONTRACT_VERSION,
            "stream": True,
            "stream_include_usage": True,
            "temperature": 0,
            "filler": "leading-space cobalt repetitions",
            "needle_position_fractions": [0.10, 0.50, 0.90],
            "chars_per_token_bounds": [MIN_CHARS_PER_TOKEN, MAX_CHARS_PER_TOKEN],
            "serialized_payload_byte_cap": True,
        }
    ).encode("utf-8")
).hexdigest()


class ContextPreflightError(RuntimeError):
    """Raised before a provider call when persisted context state is unsafe."""


class OutputDirectoryLease:
    """Non-blocking process lease for one context campaign output directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> "OutputDirectoryLease":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"\0")
            self.handle.flush()
            os.fsync(self.handle.fileno())
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            self.handle.close()
            self.handle = None
            raise ContextPreflightError(
                "another process holds the context output-directory execution lease"
            ) from error
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


RequestExecutor = Callable[[str, BenchmarkTask, int], Awaitable[StreamResult]]


@dataclass(frozen=True)
class ContextConfig:
    output_dir: Path
    model_ids: tuple[str, ...]
    seed: int = 20260823
    # A model lane is intentionally sequential. Cross-model overlap is
    # controlled separately so boundary refinement never forks within a lane.
    per_model_concurrency: int = 1
    model_parallelism: int = 12
    global_concurrency: int = 12
    fallback_account_rpm: float = DEFAULT_FALLBACK_ACCOUNT_RPM
    fallback_account_tpm: float = DEFAULT_FALLBACK_ACCOUNT_TPM
    quota_utilization_fraction: float = 0.80
    governor_multiplicative_decrease: float = 0.50
    governor_additive_increase_fraction: float = 0.05
    governor_minimum_congestion_factor: float = 0.05
    governor_successes_per_increase: int = 20
    request_timeout_seconds: float = 180.0
    max_cost_usd: float = 200.0
    prior_cost_usd: float = 0.0
    stop_launch_at: datetime | None = None
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
    combined_output_tokens: int = 4_096
    short_output_tokens: int = 32
    max_bisection_rounds: int = 8
    percentages: tuple[float, ...] = CONTEXT_PERCENTAGES
    include_prompt_boundary_triplet: bool = True
    include_combined_boundary_triplet: bool = True
    planning_tolerance_fraction: float = 0.02
    planning_tolerance_tokens: int = 256

    def validate(self) -> None:
        if not self.model_ids:
            raise ValueError("at least one model is required")
        unknown = sorted(set(self.model_ids) - MODEL_BY_ID.keys())
        if unknown:
            raise ValueError(f"unknown DigitalOcean models: {', '.join(unknown)}")
        require_digitalocean_hosted_models(self.model_ids)
        for model_id in self.model_ids:
            _validate_model_contract_against_freeze(model_id)
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("model IDs must be unique")
        if self.per_model_concurrency != 1:
            raise ValueError(
                "per-model concurrency must be exactly one; use model_parallelism "
                "for cross-endpoint overlap"
            )
        if self.model_parallelism < 1:
            raise ValueError("model_parallelism must be positive")
        if self.global_concurrency < 1:
            raise ValueError("global_concurrency must be positive")
        if not all(
            math.isfinite(value)
            for value in (
                self.fallback_account_rpm,
                self.fallback_account_tpm,
                self.quota_utilization_fraction,
                self.governor_multiplicative_decrease,
                self.governor_additive_increase_fraction,
                self.governor_minimum_congestion_factor,
            )
        ):
            raise ValueError("quota governor numeric values must be finite")
        if self.fallback_account_rpm <= 0 or self.fallback_account_tpm <= 0:
            raise ValueError("fallback account RPM and TPM must be positive")
        if not 0 < self.quota_utilization_fraction <= 1:
            raise ValueError("quota_utilization_fraction must be in (0, 1]")
        if not 0 < self.governor_multiplicative_decrease < 1:
            raise ValueError("governor multiplicative decrease must be in (0, 1)")
        if not 0 < self.governor_additive_increase_fraction <= 1:
            raise ValueError("governor additive increase must be in (0, 1]")
        if not 0 < self.governor_minimum_congestion_factor <= 1:
            raise ValueError("minimum congestion factor must be in (0, 1]")
        if self.governor_successes_per_increase < 1:
            raise ValueError("governor successes per increase must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        if self.max_cost_usd <= 0 or self.prior_cost_usd < 0:
            raise ValueError("invalid cost envelope")
        if self.prior_cost_usd > self.max_cost_usd:
            raise ValueError("prior cost already exceeds the cumulative cap")
        if self.max_payload_bytes < 16_384:
            raise ValueError("max payload bytes must be at least 16 KiB")
        if self.combined_output_tokens < 1 or self.short_output_tokens < 1:
            raise ValueError("requested output limits must be positive")
        if self.max_bisection_rounds < 0:
            raise ValueError("max bisection rounds cannot be negative")
        if not self.percentages:
            raise ValueError("at least one percentage anchor is required")
        if any(not 0 < value < 1 for value in self.percentages):
            raise ValueError("percentage anchors must be strictly between zero and one")
        if len(set(self.percentages)) != len(self.percentages):
            raise ValueError("percentage anchors must be unique")
        if self.planning_tolerance_fraction < 0 or self.planning_tolerance_tokens < 0:
            raise ValueError("planning tolerances cannot be negative")
        if self.stop_launch_at is not None and self.stop_launch_at.tzinfo is None:
            raise ValueError("stop_launch_at must be timezone-aware")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA,
            "models": list(self.model_ids),
            "seed": self.seed,
            "per_model_concurrency": self.per_model_concurrency,
            "model_parallelism": self.model_parallelism,
            "global_concurrency": self.global_concurrency,
            "account_quota_governor": {
                "quota_scope": "per_account",
                "fallback_account_rpm": self.fallback_account_rpm,
                "fallback_account_tpm": self.fallback_account_tpm,
                "quota_utilization_fraction": self.quota_utilization_fraction,
                "multiplicative_decrease": (self.governor_multiplicative_decrease),
                "additive_increase_fraction": (
                    self.governor_additive_increase_fraction
                ),
                "minimum_congestion_factor": (self.governor_minimum_congestion_factor),
                "successes_per_increase": self.governor_successes_per_increase,
                "inference_limits_doc_url": INFERENCE_LIMITS_DOC_URL,
                "inference_limits_doc_verified_date": (
                    INFERENCE_LIMITS_DOC_VERIFIED_DATE
                ),
                "quota_headers_doc_url": QUOTA_HEADERS_DOC_URL,
                "quota_headers_doc_verified_date": (QUOTA_HEADERS_DOC_VERIFIED_DATE),
            },
            "request_timeout_seconds": self.request_timeout_seconds,
            "stop_launch_at": (
                self.stop_launch_at.astimezone(timezone.utc).isoformat()
                if self.stop_launch_at is not None
                else None
            ),
            "max_cost_usd": self.max_cost_usd,
            "prior_cost_usd": self.prior_cost_usd,
            "max_payload_bytes": self.max_payload_bytes,
            "combined_output_tokens": self.combined_output_tokens,
            "short_output_tokens": self.short_output_tokens,
            "max_bisection_rounds": self.max_bisection_rounds,
            "percentages": list(self.percentages),
            "include_prompt_boundary_triplet": self.include_prompt_boundary_triplet,
            "include_combined_boundary_triplet": self.include_combined_boundary_triplet,
            "planning_tolerance_fraction": self.planning_tolerance_fraction,
            "planning_tolerance_tokens": self.planning_tolerance_tokens,
            "context_window_anchor_sources": {
                model_id: CONTEXT_WINDOW_ANCHOR_SOURCES[model_id]
                for model_id in self.model_ids
            },
            "documentation_freeze": {
                "api_reference_generated": API_DOC_GENERATED_DATE,
                "model_page_verified": MODEL_DOC_VERIFIED_DATE,
                "pricing_page_date": PRICING_DOC_DATE,
                "endpoint_freeze_artifact_sha256": ENDPOINT_FREEZE_ARTIFACT_SHA256,
            },
            "model_contracts": [
                _model_contract(model_id) for model_id in self.model_ids
            ],
        }


@dataclass(frozen=True)
class ContextProbe:
    request_id: str
    model_id: str
    probe_id: str
    estimated_target_prompt_tokens: int
    requested_max_output_tokens: int
    coverage_tags: tuple[str, ...]
    seed: int = 20260823
    anchor_percentages: tuple[float, ...] = ()
    adaptive_round: int | None = None
    scorer_contract_sha256: str = SCORER_CONTRACT_SHA256
    model_contract_sha256: str = ""
    documentation_contract_sha256: str = DOCUMENTATION_CONTRACT_SHA256
    request_identity_sha256: str = ""

    def sanitized_plan_row(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA,
            "request_id": self.request_id,
            "model_id": self.model_id,
            "probe_id": self.probe_id,
            "estimated_target_prompt_tokens": self.estimated_target_prompt_tokens,
            "requested_max_output_tokens": self.requested_max_output_tokens,
            "coverage_tags": list(self.coverage_tags),
            "seed": self.seed,
            "anchor_percentages": list(self.anchor_percentages),
            "adaptive_round": self.adaptive_round,
            "context_window_anchor_source": CONTEXT_WINDOW_ANCHOR_SOURCES[
                self.model_id
            ],
            "scorer_contract_sha256": self.scorer_contract_sha256,
            "model_contract_sha256": self.model_contract_sha256,
            "documentation_contract_sha256": self.documentation_contract_sha256,
            "request_identity_sha256": self.request_identity_sha256,
        }


def _new_probe(
    *,
    model_id: str,
    estimated_target_prompt_tokens: int,
    requested_max_output_tokens: int,
    coverage_tags: Sequence[str],
    seed: int,
    anchor_percentages: Sequence[float] = (),
    adaptive_round: int | None = None,
) -> ContextProbe:
    tags = tuple(sorted(set(coverage_tags)))
    percentages = tuple(sorted(set(float(item) for item in anchor_percentages)))
    model_contract_sha256 = _model_contract_sha256(model_id)
    identity = {
        "schema": REQUEST_SCHEMA,
        "seed": seed,
        "model_id": model_id,
        "estimated_target_prompt_tokens": int(estimated_target_prompt_tokens),
        "requested_max_output_tokens": int(requested_max_output_tokens),
        "coverage_tags": tags,
        "anchor_percentages": percentages,
        "adaptive_round": adaptive_round,
        "model_contract": _model_contract(model_id),
        "model_contract_sha256": model_contract_sha256,
        "documentation_contract_sha256": DOCUMENTATION_CONTRACT_SHA256,
        "payload_builder_contract_sha256": PAYLOAD_BUILDER_CONTRACT_SHA256,
        "scorer_contract_sha256": SCORER_CONTRACT_SHA256,
    }
    request_identity_sha256 = hashlib.sha256(
        canonical_json(identity).encode("utf-8")
    ).hexdigest()
    request_id = f"do-context-request-{request_identity_sha256[:20]}"
    return ContextProbe(
        request_id=request_id,
        model_id=model_id,
        probe_id=(
            f"context-est-{estimated_target_prompt_tokens}-out-{requested_max_output_tokens}-"
            f"{request_id.rsplit('-', 1)[-1][:8]}"
        ),
        estimated_target_prompt_tokens=max(1, int(estimated_target_prompt_tokens)),
        requested_max_output_tokens=max(1, int(requested_max_output_tokens)),
        coverage_tags=tags,
        seed=seed,
        anchor_percentages=percentages,
        adaptive_round=adaptive_round,
        model_contract_sha256=model_contract_sha256,
        request_identity_sha256=request_identity_sha256,
    )


def _probe_from_plan_row(row: Mapping[str, Any]) -> ContextProbe:
    expected = _new_probe(
        model_id=str(row["model_id"]),
        estimated_target_prompt_tokens=int(row["estimated_target_prompt_tokens"]),
        requested_max_output_tokens=int(row["requested_max_output_tokens"]),
        coverage_tags=tuple(str(item) for item in row.get("coverage_tags", ())),
        anchor_percentages=tuple(
            float(item) for item in row.get("anchor_percentages", ())
        ),
        adaptive_round=(
            int(row["adaptive_round"])
            if row.get("adaptive_round") is not None
            else None
        ),
        seed=int(row["seed"]),
    )
    expected_row = expected.sanitized_plan_row()
    for field in (
        "request_id",
        "probe_id",
        "scorer_contract_sha256",
        "model_contract_sha256",
        "documentation_contract_sha256",
        "request_identity_sha256",
    ):
        if row.get(field) != expected_row[field]:
            raise ContextPreflightError(
                f"context plan row {field} does not match its frozen request contract"
            )
    return expected


def build_context_probes(
    model_ids: Sequence[str],
    *,
    seed: int = 20260823,
    combined_output_tokens: int = 4_096,
    short_output_tokens: int = 32,
    percentages: Sequence[float] = CONTEXT_PERCENTAGES,
    include_prompt_boundary_triplet: bool = True,
    include_combined_boundary_triplet: bool = True,
) -> list[ContextProbe]:
    """Build percentage and anchor-relative probes, deduplicating collisions.

    Most anchors come from the frozen official documentation. Kimi's value is
    kept explicitly separate as an undocumented probe anchor.
    """

    unknown = sorted(set(model_ids) - MODEL_BY_ID.keys())
    if unknown:
        raise ValueError(f"unknown DigitalOcean models: {', '.join(unknown)}")
    probes: list[ContextProbe] = []
    for model_id in model_ids:
        window = MODEL_BY_ID[model_id].context_window
        if not window:
            continue
        merged: dict[tuple[int, int], dict[str, Any]] = {}

        def add(
            target: int,
            output_tokens: int,
            tag: str,
            percentage: float | None = None,
        ) -> None:
            key = (max(1, int(target)), int(output_tokens))
            state = merged.setdefault(key, {"tags": set(), "percentages": set()})
            state["tags"].add(tag)
            if percentage is not None:
                state["percentages"].add(float(percentage))

        anchor_source = CONTEXT_WINDOW_ANCHOR_SOURCES[model_id]
        source_prefix = (
            "undocumented_probe_anchor"
            if anchor_source == "undocumented_probe_anchor"
            else "advertised_context"
        )
        for percentage in percentages:
            add(
                round(window * percentage),
                short_output_tokens,
                f"{source_prefix}_percentage_{percentage:.2f}",
                percentage,
            )
        if include_combined_boundary_triplet:
            add(
                round(window * 0.50),
                combined_output_tokens,
                f"{source_prefix}_combined_bracket_low_anchor",
            )
        boundary_half_width = max(256, round(window * 0.002))
        for delta, label in (
            (-boundary_half_width, "lower"),
            (0, "center"),
            (boundary_half_width, "upper"),
        ):
            # These are uncertainty-aware *planner estimates*, never claims of
            # exact server-token positions. Rejected calls have no server usage.
            if include_prompt_boundary_triplet:
                add(
                    window + delta,
                    short_output_tokens,
                    f"{source_prefix}_prompt_estimate_{label}",
                )
            if include_combined_boundary_triplet:
                add(
                    window - combined_output_tokens + delta,
                    combined_output_tokens,
                    f"{source_prefix}_combined_estimate_{label}",
                )
        for (target, output_tokens), state in sorted(merged.items()):
            probes.append(
                _new_probe(
                    model_id=model_id,
                    estimated_target_prompt_tokens=target,
                    requested_max_output_tokens=output_tokens,
                    coverage_tags=state["tags"],
                    anchor_percentages=state["percentages"],
                    seed=seed,
                )
            )
    return probes


def _marker_for(request_id: str) -> str:
    digest = hashlib.sha256(("needle:" + request_id).encode("utf-8")).hexdigest()
    return "NEEDLE-" + digest[:24].upper()


def _payload(
    model_id: str, task: BenchmarkTask, max_output_tokens: int
) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": task.messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_output_tokens,
        "temperature": 0,
    }


def _payload_bytes(model_id: str, task: BenchmarkTask, max_output_tokens: int) -> int:
    return len(
        canonical_json(_payload(model_id, task, max_output_tokens)).encode("utf-8")
    )


def build_retrieval_task(
    probe: ContextProbe,
    *,
    chars_per_token: float,
    max_payload_bytes: int,
) -> tuple[BenchmarkTask, dict[str, Any]]:
    """Build a deterministic retrieval prompt without retaining its contents.

    The planner targets a byte budget calibrated from an earlier server usage
    count.  Exact provider tokenization is deliberately not asserted; accepted
    calls use ``usage.prompt_tokens`` as their actual coordinate.
    """

    chars_per_token = min(
        MAX_CHARS_PER_TOKEN, max(MIN_CHARS_PER_TOKEN, chars_per_token)
    )
    marker = _marker_for(probe.request_id)
    prefix = (
        "This is a synthetic retrieval test. Read the complete record and return "
        "only the value after UNIQUE NEEDLE. Do not explain. RECORD START "
    )
    marker_text = f" UNIQUE NEEDLE {marker} "
    suffix = " RECORD END"
    empty_task = BenchmarkTask(
        task_id=probe.probe_id,
        family="direct_context_retrieval",
        context_bucket=str(probe.estimated_target_prompt_tokens),
        output_bucket=str(probe.requested_max_output_tokens),
        messages=[{"role": "user", "content": prefix + marker_text + suffix}],
        expected={"kind": "exact_text", "value": marker},
    )
    empty_payload_bytes = _payload_bytes(
        probe.model_id, empty_task, probe.requested_max_output_tokens
    )
    desired_payload_bytes = min(
        max_payload_bytes,
        max(
            empty_payload_bytes,
            round(probe.estimated_target_prompt_tokens * chars_per_token),
        ),
    )
    filler_bytes = max(0, desired_payload_bytes - empty_payload_bytes)
    # Repeating a common leading-space token keeps the bytes/token relation
    # close to linear across modern BPE tokenizers while remaining synthetic.
    chunk = " cobalt"
    filler = (chunk * math.ceil(filler_bytes / len(chunk)))[:filler_bytes]
    marker_fraction = (0.10, 0.50, 0.90)[
        int(hashlib.sha256(probe.request_id.encode()).hexdigest()[:2], 16) % 3
    ]
    split = int(len(filler) * marker_fraction)
    content = prefix + filler[:split] + marker_text + filler[split:] + suffix
    task = BenchmarkTask(
        task_id=probe.probe_id,
        family="direct_context_retrieval",
        context_bucket=str(probe.estimated_target_prompt_tokens),
        output_bucket=str(probe.requested_max_output_tokens),
        messages=[{"role": "user", "content": content}],
        expected={"kind": "exact_text", "value": marker},
        metadata={
            "estimated_target_prompt_tokens": probe.estimated_target_prompt_tokens,
            "requested_max_output_tokens": probe.requested_max_output_tokens,
        },
    )
    exact_payload_bytes = _payload_bytes(
        probe.model_id, task, probe.requested_max_output_tokens
    )
    if (
        exact_payload_bytes > max_payload_bytes
    ):  # pragma: no cover - defensive precision
        overflow = exact_payload_bytes - max_payload_bytes
        filler = filler[:-overflow] if overflow < len(filler) else ""
        split = int(len(filler) * marker_fraction)
        task.messages[0]["content"] = (
            prefix + filler[:split] + marker_text + filler[split:] + suffix
        )
        exact_payload_bytes = _payload_bytes(
            probe.model_id, task, probe.requested_max_output_tokens
        )
    planning = {
        "estimated_target_prompt_tokens": probe.estimated_target_prompt_tokens,
        "planner_chars_per_token": chars_per_token,
        "request_payload_bytes": exact_payload_bytes,
        "max_payload_bytes": max_payload_bytes,
        "payload_byte_cap_applied": desired_payload_bytes >= max_payload_bytes,
        "needle_position_fraction": marker_fraction,
    }
    return task, planning


def _worst_case_prompt_tokens(probe: ContextProbe, config: ContextConfig) -> int:
    # ASCII byte fallback gives the strict client-side ceiling of one token per
    # payload byte; the extra allowance covers transport framing/tokenizer
    # wrappers that are not part of the serialized request body.
    planned_bytes = min(
        config.max_payload_bytes,
        math.ceil(probe.estimated_target_prompt_tokens * MAX_CHARS_PER_TOKEN) + 4_096,
    )
    return planned_bytes + 4_096


def conservative_reservation(
    probe: ContextProbe, config: ContextConfig
) -> tuple[float, int]:
    prompt_tokens = _worst_case_prompt_tokens(probe, config)
    spec = MODEL_BY_ID[probe.model_id]
    cost = (
        prompt_tokens * spec.input_usd_per_million
        + probe.requested_max_output_tokens * spec.output_usd_per_million
    ) / 1_000_000
    return cost, prompt_tokens


def _adaptive_output_limits(config: ContextConfig) -> tuple[tuple[str, int], ...]:
    values: list[tuple[str, int]] = []
    if config.include_prompt_boundary_triplet:
        values.append(("prompt", config.short_output_tokens))
    if config.include_combined_boundary_triplet:
        values.append(("combined", config.combined_output_tokens))
    # Preserve the more specific first label if a caller chooses equal limits.
    deduped: dict[int, str] = {}
    for label, limit in values:
        deduped.setdefault(limit, label)
    return tuple((label, limit) for limit, label in deduped.items())


def _adaptive_request_worst_case_cost(
    model_id: str, requested_output_tokens: int, config: ContextConfig
) -> float:
    spec = MODEL_BY_ID[model_id]
    window = int(spec.context_window or config.max_payload_bytes)
    maximum_bracket_target = window + max(256, round(window * 0.002))
    prompt_tokens = (
        min(
            config.max_payload_bytes,
            math.ceil(maximum_bracket_target * MAX_CHARS_PER_TOKEN) + 4_096,
        )
        + 4_096
    )
    per_request = (
        prompt_tokens * spec.input_usd_per_million
        + requested_output_tokens * spec.output_usd_per_million
    ) / 1_000_000
    return per_request


def _read_rows(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"torn JSONL journal at line {line_number}: {exc}"
                ) from exc
            identity = row.get(key)
            if identity:
                identity_text = str(identity)
                if identity_text in rows:
                    raise ContextPreflightError(
                        f"duplicate {key} in {path.name} at line {line_number}"
                    )
                rows[identity_text] = row
    return rows


class ContextBudgetLedger:
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
        self.reservations = _read_rows(path, "request_id")
        self.terminal_rows = terminal_rows
        self._lock = asyncio.Lock()

    def request_exposure(self, request_id: str) -> float:
        terminal = self.terminal_rows.get(request_id)
        if terminal is not None:
            accounted = float(terminal.get("accounted_cost_usd") or 0.0)
            usage = terminal.get("usage")
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
            if terminal.get("provider_send_attempted") is True and (
                prompt_tokens <= 0 or completion_tokens <= 0
            ):
                reservation = self.reservations.get(request_id)
                reserved = (
                    float(reservation.get("reserved_cost_usd") or 0.0)
                    if reservation is not None
                    else float(terminal.get("worst_case_reserved_cost_usd") or 0.0)
                )
                return max(accounted, reserved)
            return accounted
        reservation = self.reservations.get(request_id)
        return (
            float(reservation.get("reserved_cost_usd") or 0.0) if reservation else 0.0
        )

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
        probe: ContextProbe,
        reserved_cost_usd: float,
        reserved_prompt_tokens: int,
    ) -> bool:
        async with self._lock:
            if (
                probe.request_id in self.reservations
                or probe.request_id in self.terminal_rows
            ):
                return False
            if self.exposure_usd + reserved_cost_usd > self.max_cost_usd + 1e-12:
                return False
            row = {
                "schema_version": RESERVATION_SCHEMA,
                "campaign_id": campaign_id,
                "request_id": probe.request_id,
                "model_id": probe.model_id,
                "reserved_at": utc_now(),
                "reserved_cost_usd": reserved_cost_usd,
                "reserved_prompt_tokens": reserved_prompt_tokens,
                "requested_max_output_tokens": probe.requested_max_output_tokens,
            }
            await self.journal.append(row)
            self.reservations[probe.request_id] = row
            return True

    async def settle(self, request_id: str, row: Mapping[str, Any]) -> None:
        async with self._lock:
            self.terminal_rows[request_id] = dict(row)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_nonnegative_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _sanitized_headers(headers: Mapping[str, Any]) -> dict[str, Any]:
    request_id = headers.get("x-request-id")
    edge_id = headers.get("cf-ray")
    return {
        "request_id_sha256": _sha256_text(str(request_id)) if request_id else None,
        "edge_id_sha256": _sha256_text(str(edge_id)) if edge_id else None,
        "rate_limit_limit_requests": _parse_nonnegative_float(
            headers.get("x-ratelimit-limit-requests")
        ),
        "rate_limit_remaining_requests": _parse_nonnegative_float(
            headers.get("x-ratelimit-remaining-requests")
        ),
        "rate_limit_reset_requests_epoch_seconds": _parse_nonnegative_float(
            headers.get("x-ratelimit-reset-requests")
        ),
        "rate_limit_limit_tokens_per_minute": _parse_nonnegative_float(
            headers.get("x-ratelimit-limit-tokens-per-minute")
        ),
        "rate_limit_remaining_tokens_per_minute": _parse_nonnegative_float(
            headers.get("x-ratelimit-remaining-tokens-per-minute")
        ),
        "rate_limit_reset_tokens_per_minute_epoch_seconds": (
            _parse_nonnegative_float(headers.get("x-ratelimit-reset-tokens-per-minute"))
        ),
        "rate_limit_limit_tokens_per_day": _parse_nonnegative_float(
            headers.get("x-ratelimit-limit-tokens-per-day")
        ),
        "rate_limit_remaining_tokens_per_day": _parse_nonnegative_float(
            headers.get("x-ratelimit-remaining-tokens-per-day")
        ),
        "rate_limit_reset_tokens_per_day_epoch_seconds": _parse_nonnegative_float(
            headers.get("x-ratelimit-reset-tokens-per-day")
        ),
        "retry_after_seconds": _parse_nonnegative_float(headers.get("retry-after")),
    }


class AccountQuotaGovernor:
    """Shared open-loop admission clock for DigitalOcean's account quotas.

    The client schedules request starts from configured/observed RPM and TPM;
    this is intentionally separate from the global in-flight semaphore. A
    request larger than one minute of fallback tokens is not rejected: it is
    admitted once, then advances the token virtual clock by its proportional
    refill time. That preserves very-long-context coverage without pretending
    the fallback rate is a per-request size limit.

    Quota reset headers are used only as request-size-specific refill
    projections. They are never interpreted as fixed-window boundaries.
    Governor state is deliberately not durable; a restart begins from the
    conservative fallbacks while request IDs/reservations remain durable.
    """

    def __init__(
        self,
        config: ContextConfig,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        epoch_time: Callable[[], float] = time.time,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.config = config
        self._monotonic = monotonic
        self._epoch_time = epoch_time
        self._sleeper = sleeper
        self._lock = asyncio.Lock()
        self._request_limit_per_minute = float(config.fallback_account_rpm)
        self._token_limit_per_minute = float(config.fallback_account_tpm)
        self._request_limit_observed = False
        self._token_minute_limit_observed = False
        self._daily_token_limit: float | None = None
        self._latest_signals: dict[str, Any] = {}
        self._next_request_at = 0.0
        self._next_token_at = 0.0
        self._blocked_until_epoch = 0.0
        self._congestion_factor = 1.0
        self._healthy_since_increase = 0
        self._schedule_generation = 0
        self._admissions = 0
        self._observations = 0
        self._http_429_observations = 0
        self._endpoint_pressure_429_observations = 0
        self._limits_rehydrated_from_journal = False

    def restore_observed_limits(self, signals: Mapping[str, Any]) -> bool:
        """Restore only provider-declared account ceilings on an idempotent resume.

        Virtual admission clocks, remaining balances, resets, cooldowns, and
        congestion state are deliberately not restored. Those are transient.
        Reusing the last positive RPM/TPM ceilings prevents a resume from
        scheduling every pending long-context request against stale fallback
        rates before the first new response arrives.
        """

        request_limit = _parse_nonnegative_float(
            signals.get("rate_limit_limit_requests")
        )
        token_limit = _parse_nonnegative_float(
            signals.get("rate_limit_limit_tokens_per_minute")
        )
        if (
            request_limit is None
            or request_limit <= 0
            or token_limit is None
            or token_limit <= 0
        ):
            return False
        self._request_limit_per_minute = request_limit
        self._token_limit_per_minute = token_limit
        self._request_limit_observed = True
        self._token_minute_limit_observed = True
        daily_limit = _parse_nonnegative_float(
            signals.get("rate_limit_limit_tokens_per_day")
        )
        if daily_limit is not None and daily_limit > 0:
            self._daily_token_limit = daily_limit
        self._limits_rehydrated_from_journal = True
        return True

    @property
    def bootstrap_ready(self) -> bool:
        return self._request_limit_observed and self._token_minute_limit_observed

    def _rates(self) -> tuple[float, float]:
        scale = self.config.quota_utilization_fraction * self._congestion_factor
        request_rate = max(1e-9, self._request_limit_per_minute * scale / 60.0)
        token_rate = max(1e-9, self._token_limit_per_minute * scale / 60.0)
        return request_rate, token_rate

    async def acquire(
        self,
        *,
        estimated_tokens: int,
        stop_launch_at: datetime | None,
    ) -> dict[str, Any] | None:
        estimated_tokens = max(1, int(estimated_tokens))
        async with self._lock:
            now_mono = self._monotonic()
            now_epoch = self._epoch_time()
            request_rate, token_rate = self._rates()
            cooldown_delay = max(0.0, self._blocked_until_epoch - now_epoch)
            scheduled = max(
                now_mono,
                self._next_request_at,
                self._next_token_at,
                now_mono + cooldown_delay,
            )
            delay = max(0.0, scheduled - now_mono)
            if (
                stop_launch_at is not None
                and now_epoch + delay >= stop_launch_at.timestamp()
            ):
                return None
            self._next_request_at = scheduled + (1.0 / request_rate)
            self._next_token_at = scheduled + (estimated_tokens / token_rate)
            self._admissions += 1
            admission = {
                "estimated_account_quota_tokens": estimated_tokens,
                "open_loop_wait_seconds": delay,
                "effective_request_rate_per_minute": request_rate * 60.0,
                "effective_token_rate_per_minute": token_rate * 60.0,
                "congestion_factor": self._congestion_factor,
                "governor_schedule_generation": self._schedule_generation,
                "request_limit_source": (
                    "observed_header"
                    if self._request_limit_observed
                    else "configured_fallback"
                ),
                "token_limit_source": (
                    "observed_header"
                    if self._token_minute_limit_observed
                    else "configured_fallback"
                ),
            }
        if delay > 0:
            await self._sleeper(delay)

        # A 429 from another in-flight lane may invalidate admissions that were
        # scheduled at the old rate. Such waiters re-register exactly once in
        # the new generation, which preserves their token cost and staggers
        # them after cooldown instead of releasing a synchronized burst.
        while True:
            async with self._lock:
                now_mono = self._monotonic()
                now_epoch = self._epoch_time()
                if (
                    admission["governor_schedule_generation"]
                    != self._schedule_generation
                ):
                    request_rate, token_rate = self._rates()
                    cooldown_delay = max(0.0, self._blocked_until_epoch - now_epoch)
                    scheduled = max(
                        now_mono,
                        self._next_request_at,
                        self._next_token_at,
                        now_mono + cooldown_delay,
                    )
                    extra_delay = max(0.0, scheduled - now_mono)
                    self._next_request_at = scheduled + (1.0 / request_rate)
                    self._next_token_at = scheduled + (estimated_tokens / token_rate)
                    admission.update(
                        {
                            "effective_request_rate_per_minute": (request_rate * 60.0),
                            "effective_token_rate_per_minute": token_rate * 60.0,
                            "congestion_factor": self._congestion_factor,
                            "governor_schedule_generation": (self._schedule_generation),
                        }
                    )
                else:
                    extra_delay = max(0.0, self._blocked_until_epoch - now_epoch)
                if (
                    stop_launch_at is not None
                    and now_epoch + extra_delay >= stop_launch_at.timestamp()
                ):
                    return None
            if extra_delay <= 0:
                return admission
            admission["open_loop_wait_seconds"] = (
                float(admission["open_loop_wait_seconds"]) + extra_delay
            )
            await self._sleeper(extra_delay)

    async def observe(
        self,
        *,
        headers: Mapping[str, Any],
        http_status: int | None,
        retry_after: Any = None,
    ) -> dict[str, Any]:
        signals = _sanitized_headers(headers)
        if signals["retry_after_seconds"] is None:
            signals["retry_after_seconds"] = _parse_nonnegative_float(retry_after)
        async with self._lock:
            self._observations += 1
            request_limit = signals["rate_limit_limit_requests"]
            token_minute_limit = signals["rate_limit_limit_tokens_per_minute"]
            daily_limit = signals["rate_limit_limit_tokens_per_day"]
            if request_limit is not None and request_limit > 0:
                self._request_limit_per_minute = request_limit
                self._request_limit_observed = True
            if token_minute_limit is not None and token_minute_limit > 0:
                self._token_limit_per_minute = token_minute_limit
                self._token_minute_limit_observed = True
            if daily_limit is not None and daily_limit > 0:
                self._daily_token_limit = daily_limit
            now_epoch = self._epoch_time()
            reset_values = [
                signals["rate_limit_reset_requests_epoch_seconds"],
                signals["rate_limit_reset_tokens_per_minute_epoch_seconds"],
                signals["rate_limit_reset_tokens_per_day_epoch_seconds"],
            ]
            if http_status == 429:
                self._http_429_observations += 1
                request_remaining = signals["rate_limit_remaining_requests"]
                minute_tokens_remaining = signals[
                    "rate_limit_remaining_tokens_per_minute"
                ]
                daily_tokens_remaining = signals["rate_limit_remaining_tokens_per_day"]
                retry_seconds = signals["retry_after_seconds"]
                future_reset_reported = any(
                    value is not None and value > now_epoch for value in reset_values
                )
                explicit_endpoint_pressure = bool(
                    request_remaining is not None
                    and request_remaining > 0
                    and minute_tokens_remaining is not None
                    and minute_tokens_remaining > 0
                    and (
                        daily_tokens_remaining is None or daily_tokens_remaining > 0
                    )
                    and not future_reset_reported
                    and not (retry_seconds is not None and retry_seconds > 0)
                )
                signals["account_quota_congestion_evidence"] = (
                    not explicit_endpoint_pressure
                )
                signals["http_429_scope_classification"] = (
                    "endpoint_pressure_with_account_quota_remaining"
                    if explicit_endpoint_pressure
                    else "account_quota_or_ambiguous_congestion"
                )
                if explicit_endpoint_pressure:
                    # Model- or endpoint-local pressure must remain visible as
                    # a failed probe, but it cannot justify throttling unrelated
                    # endpoints when the provider simultaneously reports full
                    # account RPM and TPM availability.
                    self._endpoint_pressure_429_observations += 1
                    self._latest_signals = dict(signals)
                    return dict(signals)

                self._healthy_since_increase = 0
                self._congestion_factor = max(
                    self.config.governor_minimum_congestion_factor,
                    self._congestion_factor
                    * self.config.governor_multiplicative_decrease,
                )
                reduced_request_rate, _ = self._rates()
                # A 429 without usable Retry-After/reset metadata must still
                # impose a global cooldown. One request interval at the newly
                # reduced account rate is the smallest deterministic pause
                # that prevents an immediate re-send from another model lane.
                headerless_cooldown_epoch = now_epoch + (1.0 / reduced_request_rate)
                cooldown_candidates = [
                    value
                    for value in reset_values
                    if value is not None and value > now_epoch
                ]
                cooldown_candidates.append(headerless_cooldown_epoch)
                if retry_seconds is not None and retry_seconds > 0:
                    cooldown_candidates.append(now_epoch + retry_seconds)
                if cooldown_candidates:
                    self._blocked_until_epoch = max(
                        self._blocked_until_epoch, max(cooldown_candidates)
                    )
                # Discard not-yet-sent virtual admissions made at the old
                # rate. Waiting callers re-register in this new generation.
                self._schedule_generation += 1
                cooldown_mono = self._monotonic() + max(
                    0.0, self._blocked_until_epoch - now_epoch
                )
                self._next_request_at = cooldown_mono
                self._next_token_at = cooldown_mono
            elif http_status is not None and 200 <= http_status < 300:
                self._healthy_since_increase += 1
                if (
                    self._healthy_since_increase
                    >= self.config.governor_successes_per_increase
                ):
                    self._congestion_factor = min(
                        1.0,
                        self._congestion_factor
                        + self.config.governor_additive_increase_fraction,
                    )
                    self._healthy_since_increase = 0

                # A non-zero reset paired with an exhausted bucket is a refill
                # projection for the evaluated request. Zero explicitly means
                # no action is required.
                remaining_reset_pairs = (
                    (
                        signals["rate_limit_remaining_requests"],
                        signals["rate_limit_reset_requests_epoch_seconds"],
                    ),
                    (
                        signals["rate_limit_remaining_tokens_per_minute"],
                        signals["rate_limit_reset_tokens_per_minute_epoch_seconds"],
                    ),
                    (
                        signals["rate_limit_remaining_tokens_per_day"],
                        signals["rate_limit_reset_tokens_per_day_epoch_seconds"],
                    ),
                )
                exhausted_resets: list[float] = []
                for remaining, reset in remaining_reset_pairs:
                    if (
                        remaining is not None
                        and remaining <= 0
                        and reset is not None
                        and reset > now_epoch
                    ):
                        exhausted_resets.append(reset)
                if exhausted_resets:
                    self._blocked_until_epoch = max(
                        self._blocked_until_epoch, max(exhausted_resets)
                    )
                    self._schedule_generation += 1
                    cooldown_mono = self._monotonic() + max(
                        0.0, self._blocked_until_epoch - now_epoch
                    )
                    self._next_request_at = cooldown_mono
                    self._next_token_at = cooldown_mono
            self._latest_signals = dict(signals)
            return dict(signals)

    def snapshot(self) -> dict[str, Any]:
        return {
            "quota_scope": "per_account",
            "bootstrap_ready": self.bootstrap_ready,
            "request_limit_per_minute": self._request_limit_per_minute,
            "token_limit_per_minute": self._token_limit_per_minute,
            "daily_token_limit": self._daily_token_limit,
            "request_limit_observed": self._request_limit_observed,
            "token_minute_limit_observed": self._token_minute_limit_observed,
            "congestion_factor": self._congestion_factor,
            "schedule_generation": self._schedule_generation,
            "admissions": self._admissions,
            "observations": self._observations,
            "http_429_observations": self._http_429_observations,
            "endpoint_pressure_429_observations": (
                self._endpoint_pressure_429_observations
            ),
            "limits_rehydrated_from_journal": self._limits_rehydrated_from_journal,
            "latest_numeric_signals": dict(self._latest_signals),
            "restart_policy": "cold_conservative_governor_no_request_replay",
        }


CONTEXT_LIMIT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"maximum\s+context\s+length",
        r"context\s+(?:length|window)\s+(?:is\s+)?(?:exceeded|exceeds|limit)",
        r"too\s+many\s+(?:input\s+)?tokens",
        r"(?:prompt|input).{0,80}exceed.{0,80}(?:token|context|window)",
        r"(?:prompt|input).{0,80}(?:token|context|window).{0,80}(?:limit|maximum)",
        r"(?:prompt|input).{0,80}(?:less|fewer|maximum).{0,80}tokens",
        r"token.{0,80}exceed.{0,80}context",
    )
)


def classify_failure(error: BaseException) -> tuple[str, bool, str, str]:
    raw_reason = str(error)
    reason_fingerprint = _sha256_text(raw_reason)
    status = getattr(error, "status_code", None)
    if status == 402:
        return "account_blocked_402", False, "account_billing_block", reason_fingerprint
    if status == 429:
        return "rate_limited", False, "rate_limit", reason_fingerprint
    if status == 413:
        if any(pattern.search(raw_reason) for pattern in CONTEXT_LIMIT_PATTERNS):
            return (
                "explicit_context_limit_rejection",
                True,
                "http_413_allowlisted_context_or_token_limit_reason",
                reason_fingerprint,
            )
        return (
            "http_413_payload_size_inconclusive",
            False,
            "http_413_without_context_limit_reason",
            reason_fingerprint,
        )
    if isinstance(status, int) and 400 <= status < 500:
        if any(pattern.search(raw_reason) for pattern in CONTEXT_LIMIT_PATTERNS):
            return (
                "explicit_context_limit_rejection",
                True,
                "allowlisted_context_or_token_limit_reason",
                reason_fingerprint,
            )
        return (
            "other_4xx_inconclusive",
            False,
            "generic_client_rejection",
            reason_fingerprint,
        )
    if isinstance(status, int) and status >= 500:
        return "provider_error", False, "provider_5xx", reason_fingerprint
    if isinstance(error, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        return "timed_out_inconclusive", False, "wall_clock_timeout", reason_fingerprint
    return (
        "transport_error_inconclusive",
        False,
        "transport_or_local_error",
        reason_fingerprint,
    )


def _is_accepted(row: Mapping[str, Any]) -> bool:
    # A transport-level success without server-reported prompt usage has no
    # measured token coordinate.  It is useful operational evidence, but it
    # cannot anchor or refine a context-window boundary.
    return (
        row.get("coverage_classification") == "accepted"
        and row.get("coverage_conclusive") is True
        and row.get("actual_prompt_tokens_x_axis") is not None
    )


def _is_explicit_rejection(row: Mapping[str, Any]) -> bool:
    return row.get("coverage_classification") == "explicit_context_limit_rejection"


class DirectContextCampaign:
    def __init__(self, config: ContextConfig) -> None:
        config.validate()
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_probes = build_context_probes(
            config.model_ids,
            seed=config.seed,
            combined_output_tokens=config.combined_output_tokens,
            short_output_tokens=config.short_output_tokens,
            percentages=config.percentages,
            include_prompt_boundary_triplet=config.include_prompt_boundary_triplet,
            include_combined_boundary_triplet=config.include_combined_boundary_triplet,
        )
        self.adaptive_output_limits = _adaptive_output_limits(config)
        self.maximum_adaptive_requests = (
            len(config.model_ids)
            * len(self.adaptive_output_limits)
            * config.max_bisection_rounds
        )
        self.fixed_settlement_ceiling_usd = sum(
            conservative_reservation(probe, config)[0] for probe in self.fixed_probes
        )
        self.adaptive_settlement_ceiling_usd = sum(
            _adaptive_request_worst_case_cost(model_id, output_tokens, config)
            * config.max_bisection_rounds
            for model_id in config.model_ids
            for _, output_tokens in self.adaptive_output_limits
        )
        self.all_requests_settlement_ceiling_usd = (
            self.fixed_settlement_ceiling_usd + self.adaptive_settlement_ceiling_usd
        )
        self.full_plan_guaranteed_to_fit_budget = (
            config.prior_cost_usd + self.all_requests_settlement_ceiling_usd
            <= config.max_cost_usd + 1e-12
        )
        self.effective_model_parallelism = min(
            len(config.model_ids),
            config.model_parallelism,
            config.global_concurrency,
        )
        self.latency_measurement_scope = (
            "isolated_single_inflight"
            if config.model_parallelism == 1 and config.global_concurrency == 1
            else "concurrent_context_probe"
        )
        per_model_maximum_requests = {
            model_id: (
                sum(probe.model_id == model_id for probe in self.fixed_probes)
                + len(self.adaptive_output_limits) * config.max_bisection_rounds
            )
            for model_id in config.model_ids
        }
        maximum_chain_requests = max(per_model_maximum_requests.values(), default=0)
        maximum_remainder_requests = max(
            (value - 1 for value in per_model_maximum_requests.values()),
            default=0,
        )
        # Worst timeout-only design projection assumes none of the serialized
        # calibration calls exposes the complete request+TPM header contract,
        # then batches the sequential model remainders across the effective
        # endpoint slots. Governor waits are explicitly excluded and can only
        # be bounded by stop_launch_at.
        bootstrap_waves_without_headers = len(config.model_ids)
        remainder_waves = (
            math.ceil(len(config.model_ids) / self.effective_model_parallelism)
            * maximum_remainder_requests
        )
        self.parallel_timeout_only_projection_seconds = (
            bootstrap_waves_without_headers + remainder_waves
        ) * config.request_timeout_seconds
        # If the first calibration exposes both limits, remaining calibration
        # calls become the first step in their concurrent model chains.
        self.first_calibration_header_projection_seconds = (
            1
            + math.ceil(len(config.model_ids) / self.effective_model_parallelism)
            * maximum_chain_requests
        ) * config.request_timeout_seconds
        self.serialized_configured_timeout_sum_seconds = (
            len(self.fixed_probes) + self.maximum_adaptive_requests
        ) * config.request_timeout_seconds
        per_model_peak: list[float] = []
        for model_id in config.model_ids:
            costs = sorted(
                (
                    conservative_reservation(probe, config)[0]
                    for probe in self.fixed_probes
                    if probe.model_id == model_id
                ),
                reverse=True,
            )
            fixed_batch = sum(costs[: config.per_model_concurrency])
            adaptive_single = max(
                (
                    _adaptive_request_worst_case_cost(model_id, output_tokens, config)
                    for _, output_tokens in self.adaptive_output_limits
                ),
                default=0.0,
            )
            per_model_peak.append(max(fixed_batch, adaptive_single))
        self.max_inflight_reservation_usd = sum(
            sorted(per_model_peak, reverse=True)[: self.effective_model_parallelism]
        )
        if (
            config.prior_cost_usd + self.max_inflight_reservation_usd
            > config.max_cost_usd + 1e-12
        ):
            raise ValueError(
                "the worst globally concurrent in-flight reservation set cannot fit "
                "under the "
                "cumulative cap: "
                f"prior=${config.prior_cost_usd:.6f}, "
                f"max_inflight=${self.max_inflight_reservation_usd:.6f}, "
                f"cap=${config.max_cost_usd:.6f}"
            )
        campaign_plan_identity = {
            "config": config.identity_payload(),
            "fixed_probes": [probe.sanitized_plan_row() for probe in self.fixed_probes],
            "adaptive_design": {
                "algorithm": "mixed-coordinate bounded binary refinement",
                "maximum_rounds_per_model_and_output_lane": config.max_bisection_rounds,
                "output_lanes": list(self.adaptive_output_limits),
                "payload_builder_contract_sha256": PAYLOAD_BUILDER_CONTRACT_SHA256,
                "scorer_contract_sha256": SCORER_CONTRACT_SHA256,
            },
        }
        self.campaign_plan_sha256 = hashlib.sha256(
            canonical_json(campaign_plan_identity).encode("utf-8")
        ).hexdigest()
        self.campaign_id = f"do-context-{self.campaign_plan_sha256[:20]}"
        self.requests_path = self.output_dir / "requests.jsonl"
        self.plan_path = self.output_dir / "plan.jsonl"
        self.reservations_path = self.output_dir / "reservations.jsonl"
        self.execution_lease_path = self.output_dir / ".execution.lock"
        self.requests_journal = JsonlJournal(self.requests_path)
        self.plan_journal = JsonlJournal(self.plan_path)
        self.fixed_plan_text = "".join(
            json.dumps(self._plan_row(probe), sort_keys=True) + "\n"
            for probe in self.fixed_probes
        )
        self.fixed_plan_sha256 = hashlib.sha256(
            self.fixed_plan_text.encode("utf-8")
        ).hexdigest()
        self.request_rows: dict[str, dict[str, Any]] = {}
        self.plan_rows: dict[str, dict[str, Any]] = {}
        self.account_blocked_402 = False
        self._append_lock = asyncio.Lock()
        self._global_semaphore: asyncio.Semaphore | None = None
        self.quota_governor: AccountQuotaGovernor | None = None
        with OutputDirectoryLease(self.execution_lease_path):
            self._reload_runtime_state()
            self._write_fixed_plan()
            self._write_or_validate_manifest()

    def _plan_row(self, probe: ContextProbe) -> dict[str, Any]:
        return {
            **probe.sanitized_plan_row(),
            "campaign_plan_sha256": self.campaign_plan_sha256,
        }

    def _reload_runtime_state(self) -> None:
        self.request_rows = _read_rows(self.requests_path, "request_id")
        self.plan_rows = _read_rows(self.plan_path, "request_id")
        self.budget = ContextBudgetLedger(
            path=self.reservations_path,
            max_cost_usd=self.config.max_cost_usd,
            prior_cost_usd=self.config.prior_cost_usd,
            terminal_rows=self.request_rows,
        )
        self.account_blocked_402 = any(
            row.get("coverage_classification") == "account_blocked_402"
            for row in self.request_rows.values()
        )

    def _write_or_validate_manifest(self) -> None:
        path = self.output_dir / "manifest.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("campaign_id") != self.campaign_id:
                raise RuntimeError(
                    "output directory belongs to another context campaign"
                )
            if existing.get("campaign_plan_sha256") != self.campaign_plan_sha256:
                raise ContextPreflightError("context campaign plan contract changed")
            if existing.get("fixed_plan_sha256") != self.fixed_plan_sha256:
                raise ContextPreflightError("context fixed plan hash changed")
            self._validate_plan_prefix_and_rows()
            return
        manifest = {
            **self.config.identity_payload(),
            "campaign_id": self.campaign_id,
            "campaign_plan_sha256": self.campaign_plan_sha256,
            "fixed_plan_sha256": self.fixed_plan_sha256,
            "payload_builder_contract_sha256": PAYLOAD_BUILDER_CONTRACT_SHA256,
            "scorer_contract_sha256": SCORER_CONTRACT_SHA256,
            "documentation_contract": _documentation_contract(),
            "documentation_contract_sha256": DOCUMENTATION_CONTRACT_SHA256,
            "created_at": utc_now(),
            "fixed_planned_requests": len(self.fixed_probes),
            "maximum_adaptive_requests": self.maximum_adaptive_requests,
            "maximum_total_requests": (
                len(self.fixed_probes) + self.maximum_adaptive_requests
            ),
            "parallel_timeout_only_projection_seconds": (
                self.parallel_timeout_only_projection_seconds
            ),
            "first_calibration_header_timeout_projection_seconds": (
                self.first_calibration_header_projection_seconds
            ),
            "serialized_configured_timeout_sum_seconds": (
                self.serialized_configured_timeout_sum_seconds
            ),
            "quota_governor_wait_upper_bound_seconds": None,
            "fixed_all_fail_settlement_ceiling_usd": self.fixed_settlement_ceiling_usd,
            "adaptive_all_fail_settlement_ceiling_usd": self.adaptive_settlement_ceiling_usd,
            "all_requests_all_fail_settlement_ceiling_usd": (
                self.all_requests_settlement_ceiling_usd
            ),
            "max_simultaneous_inflight_reservation_usd": (
                self.max_inflight_reservation_usd
            ),
            "full_plan_guaranteed_to_fit_budget": (
                self.full_plan_guaranteed_to_fit_budget
            ),
            "budget_preflight_contract": (
                "The preflight maximum is the sum of the largest per-model reservation "
                "that can be simultaneously runnable under the effective model/global "
                "concurrency ceiling. A quota and global permit is acquired before each "
                "fsync-backed reservation. Successful calls settle to usage-priced cost; "
                "failed or unknown calls retain their conservative reservation and can "
                "cause later cells to be skipped_budget_cap rather than exceed the cap"
            ),
            "models": [
                {
                    **asdict(MODEL_BY_ID[item]),
                    "context_window_anchor_source": CONTEXT_WINDOW_ANCHOR_SOURCES[item],
                }
                for item in self.config.model_ids
            ],
            "execution_order": (
                "one sequential chain per model; calibration anchor first; calibration "
                "bootstraps serialize until request and TPM limit headers are observed; "
                "then model chains overlap behind shared per-account quota and global "
                "concurrency controls"
            ),
            "effective_model_parallelism": self.effective_model_parallelism,
            "quota_governor_contract": (
                "DigitalOcean documents serverless inference RPM/TPM quotas per account. "
                "One shared open-loop request/token virtual clock governs all model lanes. "
                "Observed numeric quota headers replace conservative fallback limits. Any "
                "HTTP 429 is inconclusive context evidence, triggers account-wide "
                "multiplicative decrease, and honors Retry-After plus non-zero Unix-epoch "
                "refill projections. Reset values are never treated as fixed windows. "
                "Healthy observations additively recover the congestion factor"
            ),
            "quota_governor_restart_contract": (
                "Governor pacing state restarts cold from conservative hash-bound fallback "
                "limits; durable request/reservation journals still forbid replay"
            ),
            "latency_claim_contract": (
                "Request timing recorded during cross-endpoint execution is labelled "
                "concurrent_context_probe and cannot support isolated latency or TTFT "
                "comparisons. Only a run configured with model_parallelism=1 and "
                "global_concurrency=1 is isolated_single_inflight"
            ),
            "timeout_bound_contract": (
                "parallel_timeout_only_projection_seconds is a parallel critical-path "
                "projection with serial no-header bootstrap; it excludes quota pacing, "
                "cooldown, local overhead, and cancellation-completion delay. "
                "serialized_configured_timeout_sum_seconds is only the sum of configured "
                "per-request timeout values, not a wall-clock hard bound: asyncio task "
                "cancellation may finish after the configured timeout. Governor waits "
                "have no finite intrinsic upper bound and stop at stop_launch_at"
            ),
            "deadline_contract": (
                "stop_launch_at is a provider-send cutoff. Every in-flight request has the "
                "independent request_timeout_seconds bound and may settle after the send cutoff"
            ),
            "actual_x_axis": "server-reported usage.prompt_tokens on accepted requests",
            "boundary_design": (
                "Configured model-relative percentage anchors plus uncertainty-aware "
                "lower/center/upper client-planned estimates around the advertised window "
                "(or Kimi's explicitly undocumented "
                "probe anchor), for both prompt-only and prompt-plus-requested-output. These "
                "are never exact server-token claims. A 50% combined-limit low anchor enables "
                "a separate prompt-plus-output bracket. Bounded binary refinement narrows a "
                "mixed-coordinate accepted/rejected interval when a transition is bracketed"
            ),
            "retrieval_design": (
                "Deterministic synthetic filler with a hashed needle at 10%, 50%, or 90%; "
                "acceptance and exact retrieval correctness are reported separately"
            ),
            "payload_cap_contract": (
                "Serialized request payloads never exceed max_payload_bytes. A capped probe "
                "remains useful as a byte-limit observation but cannot establish a higher "
                "token boundary than the server-reported usage count"
            ),
            "combined_limit_contract": (
                "Combined-boundary cells request a large max_tokens value but expect a short "
                "retrieval answer; they test request acceptance, not realized output length"
            ),
            "coverage_contract": (
                "HTTP 2xx acceptance with server prompt usage is conclusive. Only an "
                "allowlisted, safely parsed provider reason explicitly identifying a "
                "context/token limit is a conclusive boundary rejection; status 413 alone "
                "may only identify an intermediary byte limit. Generic 4xx, timeout, 429, "
                "5xx, transport "
                "failure, interruption, deadline, and budget skips are inconclusive"
            ),
            "sanitization": (
                "Credentials, prompts, outputs, response bodies, and raw headers are never "
                "persisted; request/response/provider IDs are SHA-256 fingerprints only"
            ),
            "documentation_freeze": {
                "api_reference_generated": API_DOC_GENERATED_DATE,
                "model_page_verified": MODEL_DOC_VERIFIED_DATE,
                "pricing_page_date": PRICING_DOC_DATE,
                "endpoint_freeze_artifact_sha256": ENDPOINT_FREEZE_ARTIFACT_SHA256,
            },
            "source_role": "benchmark-authored synthetic context retrieval probes",
            "rights_posture": "repository-authored redistributable test definitions",
        }
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _validate_plan_prefix_and_rows(self) -> None:
        if not self.plan_path.is_file():
            raise ContextPreflightError(
                "context manifest exists but plan.jsonl is missing"
            )
        plan_bytes = self.plan_path.read_bytes()
        if not plan_bytes.startswith(self.fixed_plan_text.encode("utf-8")):
            raise ContextPreflightError(
                "context plan fixed prefix does not match its hash"
            )
        for row in self.plan_rows.values():
            if row.get("campaign_plan_sha256") != self.campaign_plan_sha256:
                raise ContextPreflightError(
                    "context plan row campaign contract does not match the manifest"
                )
            _probe_from_plan_row(row)

    def _write_fixed_plan(self) -> None:
        if self.plan_rows:
            expected = {probe.request_id for probe in self.fixed_probes}
            if not expected.issubset(self.plan_rows):
                raise RuntimeError(
                    "existing context plan is incomplete or belongs to older code"
                )
            self._validate_plan_prefix_and_rows()
            return
        self.plan_path.write_text(self.fixed_plan_text, encoding="utf-8", newline="\n")
        self.plan_rows = _read_rows(self.plan_path, "request_id")

    async def _append_plan_probe(self, probe: ContextProbe) -> None:
        async with self._append_lock:
            if probe.request_id in self.plan_rows:
                return
            row = self._plan_row(probe)
            await self.plan_journal.append(row)
            self.plan_rows[probe.request_id] = row

    async def _append_request(self, row: dict[str, Any]) -> None:
        async with self._append_lock:
            request_id = str(row["request_id"])
            if request_id in self.request_rows:
                return
            await self.requests_journal.append(row)
            self.request_rows[request_id] = row
            await self.budget.settle(request_id, row)

    def _deadline_reached(self) -> bool:
        cutoff = self.config.stop_launch_at
        return cutoff is not None and datetime.now(timezone.utc) >= cutoff.astimezone(
            timezone.utc
        )

    def _base_row(
        self,
        probe: ContextProbe,
        task: BenchmarkTask,
        planning: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = _payload(probe.model_id, task, probe.requested_max_output_tokens)
        return {
            "schema_version": REQUEST_SCHEMA,
            "campaign_id": self.campaign_id,
            "campaign_plan_sha256": self.campaign_plan_sha256,
            "request_id": probe.request_id,
            "cell_id": probe.request_id,
            "provider": "digitalocean-serverless-inference",
            "model_id": probe.model_id,
            "probe_id": probe.probe_id,
            "estimated_target_prompt_tokens": probe.estimated_target_prompt_tokens,
            "requested_max_output_tokens": probe.requested_max_output_tokens,
            "coverage_tags": list(probe.coverage_tags),
            "anchor_percentages": list(probe.anchor_percentages),
            "adaptive_round": probe.adaptive_round,
            "context_window_anchor_source": CONTEXT_WINDOW_ANCHOR_SOURCES[
                probe.model_id
            ],
            "request_payload_sha256": _sha256_json(payload),
            "rendered_payload_sha256": _sha256_json(payload),
            "scorer_contract_sha256": probe.scorer_contract_sha256,
            "model_contract_sha256": probe.model_contract_sha256,
            "documentation_contract_sha256": (probe.documentation_contract_sha256),
            "request_identity_sha256": probe.request_identity_sha256,
            "request_payload_bytes": int(planning["request_payload_bytes"]),
            "planning": dict(planning),
            "latency_measurement_scope": self.latency_measurement_scope,
            "latency_comparison_eligible": (
                self.latency_measurement_scope == "isolated_single_inflight"
            ),
        }

    def _timing_evidence(self, values: Mapping[str, Any]) -> dict[str, Any]:
        if self.latency_measurement_scope == "isolated_single_inflight":
            return {
                "timing": dict(values),
                "concurrent_timing_diagnostic": None,
            }
        # The public report normalizer reads only ``timing``. Keep concurrent
        # diagnostics available for audit without allowing them to enter
        # isolated latency/TTFT comparison tables accidentally.
        return {
            "timing": {},
            "concurrent_timing_diagnostic": dict(values),
        }

    async def _append_unlaunched(
        self,
        probe: ContextProbe,
        *,
        task: BenchmarkTask,
        planning: Mapping[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        reservation = self.budget.reservations.get(probe.request_id)
        reserved_cost = (
            float(reservation.get("reserved_cost_usd") or 0.0) if reservation else 0.0
        )
        reserved_tokens = (
            int(reservation.get("reserved_prompt_tokens") or 0) if reservation else 0
        )
        now = utc_now()
        row = {
            **self._base_row(probe, task, planning),
            "provider_send_attempted": reason == "unknown_prior_reservation",
            "started_at": now,
            "ended_at": now,
            "status": reason,
            "coverage_classification": reason,
            "coverage_conclusive": False,
            "http_status": None,
            "error_type": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "actual_prompt_tokens_x_axis": None,
            "planning_error_tokens": None,
            "planning_absolute_error_tokens": None,
            "planning_within_tolerance": None,
            "timing": {"request_seconds": 0.0, "ttft_seconds": None},
            "retrieval_correct": False,
            "worst_case_reserved_cost_usd": reserved_cost,
            "reserved_prompt_tokens": reserved_tokens,
            "estimated_cost_usd": None,
            "accounted_cost_usd": reserved_cost,
        }
        await self._append_request(row)
        return row

    async def _run_probe(
        self,
        executor: RequestExecutor,
        probe: ContextProbe,
        *,
        chars_per_token: float,
    ) -> dict[str, Any]:
        existing = self.request_rows.get(probe.request_id)
        if existing is not None:
            return existing
        task, planning = build_retrieval_task(
            probe,
            chars_per_token=chars_per_token,
            max_payload_bytes=self.config.max_payload_bytes,
        )
        if probe.request_id in self.budget.reservations:
            return await self._append_unlaunched(
                probe, task=task, planning=planning, reason="unknown_prior_reservation"
            )
        if self._deadline_reached():
            return await self._append_unlaunched(
                probe, task=task, planning=planning, reason="skipped_deadline"
            )
        if self.account_blocked_402:
            return await self._append_unlaunched(
                probe, task=task, planning=planning, reason="skipped_http_402_latch"
            )
        if self._global_semaphore is None or self.quota_governor is None:
            raise RuntimeError("context runtime scheduler is not initialized")

        # The global concurrency ceiling is independent of the open-loop
        # account quota clock. Taking the slot first prevents quota admissions
        # from accumulating behind the semaphore and bursting later.
        async with self._global_semaphore:
            admission = await self.quota_governor.acquire(
                estimated_tokens=(
                    probe.estimated_target_prompt_tokens
                    + probe.requested_max_output_tokens
                ),
                stop_launch_at=self.config.stop_launch_at,
            )
            if admission is None or self._deadline_reached():
                return await self._append_unlaunched(
                    probe, task=task, planning=planning, reason="skipped_deadline"
                )
            if self.account_blocked_402:
                return await self._append_unlaunched(
                    probe,
                    task=task,
                    planning=planning,
                    reason="skipped_http_402_latch",
                )

            # The durable reservation is intentionally as late as possible:
            # quota/global permit first, then fsync reservation, then one final
            # deadline/402 check immediately before the provider send.
            reserved_cost, reserved_tokens = conservative_reservation(
                probe, self.config
            )
            reserved = await self.budget.reserve(
                campaign_id=self.campaign_id,
                probe=probe,
                reserved_cost_usd=reserved_cost,
                reserved_prompt_tokens=reserved_tokens,
            )
            if not reserved:
                reason = (
                    "unknown_prior_reservation"
                    if probe.request_id in self.budget.reservations
                    else "skipped_budget_cap"
                )
                return await self._append_unlaunched(
                    probe, task=task, planning=planning, reason=reason
                )
            if self._deadline_reached():
                return await self._append_unlaunched(
                    probe, task=task, planning=planning, reason="skipped_deadline"
                )
            if self.account_blocked_402:
                return await self._append_unlaunched(
                    probe,
                    task=task,
                    planning=planning,
                    reason="skipped_http_402_latch",
                )

            started_at = utc_now()
            started = time.perf_counter()
            base = {
                **self._base_row(probe, task, planning),
                "account_quota_admission": admission,
            }
            try:
                result = await asyncio.wait_for(
                    executor(
                        probe.model_id,
                        task,
                        probe.requested_max_output_tokens,
                    ),
                    timeout=self.config.request_timeout_seconds,
                )
                server_signals = await self.quota_governor.observe(
                    headers=result.response_headers,
                    http_status=result.status_code,
                )
                usage = {
                    "prompt_tokens": int(result.usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(
                        result.usage.get("completion_tokens") or 0
                    ),
                    "total_tokens": int(result.usage.get("total_tokens") or 0),
                }
                prompt_usage_present = (
                    "prompt_tokens" in result.usage
                    and result.usage.get("prompt_tokens") is not None
                )
                completion_usage_present = (
                    "completion_tokens" in result.usage
                    and result.usage.get("completion_tokens") is not None
                )
                actual_prompt_coordinate_present = (
                    prompt_usage_present and usage["prompt_tokens"] > 0
                )
                usage_complete_for_settlement = (
                    actual_prompt_coordinate_present
                    and completion_usage_present
                    and usage["completion_tokens"] > 0
                )
                spec = MODEL_BY_ID[probe.model_id]
                actual_cost = (
                    usage["prompt_tokens"] * spec.input_usd_per_million
                    + usage["completion_tokens"] * spec.output_usd_per_million
                ) / 1_000_000
                quality = score_result(task, result)
                planning_tolerance = max(
                    self.config.planning_tolerance_tokens,
                    math.ceil(
                        probe.estimated_target_prompt_tokens
                        * self.config.planning_tolerance_fraction
                    ),
                )
                planning_error = (
                    usage["prompt_tokens"] - probe.estimated_target_prompt_tokens
                    if actual_prompt_coordinate_present
                    else None
                )
                row = {
                    **base,
                    "provider_send_attempted": True,
                    "started_at": started_at,
                    "ended_at": utc_now(),
                    "status": "success",
                    "coverage_classification": "accepted",
                    "coverage_conclusive": actual_prompt_coordinate_present,
                    "acceptance_conclusive": True,
                    "http_status": result.status_code,
                    "error_type": None,
                    "usage": usage,
                    "actual_prompt_tokens_x_axis": (
                        usage["prompt_tokens"]
                        if actual_prompt_coordinate_present
                        else None
                    ),
                    "prompt_usage_present": prompt_usage_present,
                    "completion_usage_present": completion_usage_present,
                    "usage_complete_for_settlement": usage_complete_for_settlement,
                    "planning_error_tokens": planning_error,
                    "planning_absolute_error_tokens": (
                        abs(planning_error) if planning_error is not None else None
                    ),
                    "planning_tolerance_tokens": planning_tolerance,
                    "planning_within_tolerance": (
                        abs(planning_error) <= planning_tolerance
                        if planning_error is not None
                        else None
                    ),
                    **self._timing_evidence(
                        {
                            "request_seconds": result.request_seconds,
                            "headers_seconds": result.headers_seconds,
                            "ttft_seconds": result.ttft_seconds,
                            "generation_seconds": result.generation_seconds,
                            "stream_seconds": result.stream_seconds,
                        }
                    ),
                    "finish_reason": result.finish_reason,
                    "retrieval_correct": (
                        float(quality.get("quality_score") or 0.0) >= 0.999999
                    ),
                    "response_sha256": _sha256_json(
                        {
                            "text": result.text,
                            "reasoning": result.reasoning_text,
                            "tool_calls": result.tool_calls,
                        }
                    ),
                    "server_signals": server_signals,
                    "worst_case_reserved_cost_usd": reserved_cost,
                    "reserved_prompt_tokens": reserved_tokens,
                    "estimated_cost_usd": (
                        actual_cost if usage_complete_for_settlement else None
                    ),
                    "accounted_cost_usd": (
                        actual_cost if usage_complete_for_settlement else reserved_cost
                    ),
                }
            except asyncio.CancelledError as error:
                row = {
                    **base,
                    "provider_send_attempted": True,
                    "started_at": started_at,
                    "ended_at": utc_now(),
                    "status": "unknown_cancelled",
                    "coverage_classification": "unknown_cancelled",
                    "coverage_conclusive": False,
                    "http_status": None,
                    "error_type": type(error).__name__,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "actual_prompt_tokens_x_axis": None,
                    "planning_error_tokens": None,
                    "planning_absolute_error_tokens": None,
                    "planning_within_tolerance": None,
                    **self._timing_evidence(
                        {
                            "request_seconds": time.perf_counter() - started,
                            "ttft_seconds": None,
                        }
                    ),
                    "retrieval_correct": False,
                    "worst_case_reserved_cost_usd": reserved_cost,
                    "reserved_prompt_tokens": reserved_tokens,
                    "estimated_cost_usd": None,
                    "accounted_cost_usd": reserved_cost,
                }
                await self._append_request(row)
                raise
            except Exception as error:
                classification, conclusive, reason_category, reason_hash = (
                    classify_failure(error)
                )
                http_status = getattr(error, "status_code", None)
                server_signals = await self.quota_governor.observe(
                    headers=getattr(error, "response_headers", {}) or {},
                    http_status=http_status,
                    retry_after=getattr(error, "retry_after", None),
                )
                if classification == "account_blocked_402":
                    self.account_blocked_402 = True
                row = {
                    **base,
                    "provider_send_attempted": True,
                    "started_at": started_at,
                    "ended_at": utc_now(),
                    "status": "error",
                    "coverage_classification": classification,
                    "coverage_conclusive": conclusive,
                    "http_status": http_status,
                    "error_type": type(error).__name__,
                    "sanitized_reason_category": reason_category,
                    "error_fingerprint_sha256": reason_hash,
                    "server_signals": server_signals,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "actual_prompt_tokens_x_axis": None,
                    "planning_error_tokens": None,
                    "planning_absolute_error_tokens": None,
                    "planning_within_tolerance": None,
                    **self._timing_evidence(
                        {
                            "request_seconds": time.perf_counter() - started,
                            "ttft_seconds": None,
                        }
                    ),
                    "retrieval_correct": False,
                    "worst_case_reserved_cost_usd": reserved_cost,
                    "reserved_prompt_tokens": reserved_tokens,
                    "estimated_cost_usd": None,
                    "accounted_cost_usd": reserved_cost,
                }
            await self._append_request(row)
            return row

    def _calibrated_chars_per_token(self, model_id: str) -> float:
        candidates = [
            row
            for row in self.request_rows.values()
            if row.get("model_id") == model_id
            and _is_accepted(row)
            and int(row.get("usage", {}).get("prompt_tokens") or 0) > 0
        ]
        if not candidates:
            return DEFAULT_CHARS_PER_TOKEN
        candidates.sort(
            key=lambda row: int(row.get("estimated_target_prompt_tokens") or 0)
        )
        row = candidates[0]
        actual = int(row["usage"]["prompt_tokens"])
        payload_bytes = int(row.get("request_payload_bytes") or 0)
        if actual <= 0 or payload_bytes <= 0:
            return DEFAULT_CHARS_PER_TOKEN
        return min(
            MAX_CHARS_PER_TOKEN, max(MIN_CHARS_PER_TOKEN, payload_bytes / actual)
        )

    def _transition_bracket(
        self, model_id: str, requested_output_tokens: int
    ) -> tuple[int, int] | None:
        candidates = [
            row
            for row in self.request_rows.values()
            if row.get("model_id") == model_id
            and int(row.get("requested_max_output_tokens") or 0)
            == requested_output_tokens
        ]
        accepted = [
            int(row["estimated_target_prompt_tokens"])
            for row in candidates
            if _is_accepted(row)
        ]
        rejected = [
            int(row["estimated_target_prompt_tokens"])
            for row in candidates
            if _is_explicit_rejection(row)
        ]
        if not accepted or not rejected:
            return None
        lower = max(accepted)
        above = [target for target in rejected if target > lower]
        if not above:
            return None
        return lower, min(above)

    async def _adaptive_refinement(
        self,
        executor: RequestExecutor,
        *,
        model_id: str,
        chars_per_token: float,
        boundary_kind: str,
        requested_output_tokens: int,
    ) -> None:
        specific_tag = f"observed_{boundary_kind}_transition_bisection"
        existing = sorted(
            (
                _probe_from_plan_row(row)
                for row in self.plan_rows.values()
                if row.get("model_id") == model_id
                and specific_tag in row.get("coverage_tags", [])
                and int(row.get("requested_max_output_tokens") or 0)
                == requested_output_tokens
            ),
            key=lambda probe: int(probe.adaptive_round or 0),
        )
        for probe in existing:
            if probe.request_id not in self.request_rows:
                row = await self._run_probe(
                    executor, probe, chars_per_token=chars_per_token
                )
                if not (_is_accepted(row) or _is_explicit_rejection(row)):
                    return
        completed_rounds = sum(
            row.get("model_id") == model_id
            and specific_tag in row.get("coverage_tags", [])
            and int(row.get("requested_max_output_tokens") or 0)
            == requested_output_tokens
            for row in self.plan_rows.values()
        )
        for round_index in range(completed_rounds, self.config.max_bisection_rounds):
            bracket = self._transition_bracket(model_id, requested_output_tokens)
            if bracket is None:
                return
            lower, upper = bracket
            if upper - lower <= 1:
                return
            target = (lower + upper) // 2
            probe = _new_probe(
                model_id=model_id,
                estimated_target_prompt_tokens=target,
                requested_max_output_tokens=requested_output_tokens,
                coverage_tags=("observed_transition_bisection", specific_tag),
                adaptive_round=round_index,
                seed=self.config.seed,
            )
            await self._append_plan_probe(probe)
            row = await self._run_probe(
                executor, probe, chars_per_token=chars_per_token
            )
            if not (_is_accepted(row) or _is_explicit_rejection(row)):
                return

    def _calibration_probe(self, model_id: str) -> ContextProbe:
        probes = [probe for probe in self.fixed_probes if probe.model_id == model_id]
        return min(
            probes,
            key=lambda probe: (
                0 if 0.01 in probe.anchor_percentages else 1,
                probe.estimated_target_prompt_tokens,
            ),
        )

    async def _run_calibration(self, executor: RequestExecutor, model_id: str) -> None:
        calibration = self._calibration_probe(model_id)
        await self._run_probe(
            executor,
            calibration,
            chars_per_token=self._calibrated_chars_per_token(model_id),
        )

    async def _run_model_remainder(
        self, executor: RequestExecutor, model_id: str
    ) -> None:
        probes = [probe for probe in self.fixed_probes if probe.model_id == model_id]
        calibration = self._calibration_probe(model_id)
        chars_per_token = self._calibrated_chars_per_token(model_id)
        pending = [
            probe for probe in probes if probe.request_id != calibration.request_id
        ]
        # This loop is deliberately sequential. A single model/output boundary
        # chain never has more than one provider request in flight.
        for probe in pending:
            await self._run_probe(executor, probe, chars_per_token=chars_per_token)
        for boundary_kind, output_tokens in self.adaptive_output_limits:
            await self._adaptive_refinement(
                executor,
                model_id=model_id,
                chars_per_token=chars_per_token,
                boundary_kind=boundary_kind,
                requested_output_tokens=output_tokens,
            )

    def _boundary_observation(
        self,
        *,
        model_id: str,
        requested_output_tokens: int,
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        relevant = [
            row
            for row in rows
            if int(row.get("requested_max_output_tokens") or 0)
            == requested_output_tokens
        ]
        accepted = [row for row in relevant if _is_accepted(row)]
        rejected = [row for row in relevant if _is_explicit_rejection(row)]
        actual_tokens = [
            int(row["actual_prompt_tokens_x_axis"])
            for row in accepted
            if row.get("actual_prompt_tokens_x_axis") is not None
        ]
        bracket = self._transition_bracket(model_id, requested_output_tokens)
        if bracket is not None:
            lower, upper = bracket
            lower_actuals = [
                int(row["actual_prompt_tokens_x_axis"])
                for row in accepted
                if int(row["estimated_target_prompt_tokens"]) == lower
                and row.get("actual_prompt_tokens_x_axis") is not None
            ]
            return {
                "classification": "interval_censored_mixed_coordinates",
                "requested_max_output_tokens": requested_output_tokens,
                "accepted_lower_estimated_target_tokens": lower,
                "accepted_lower_actual_prompt_tokens": (
                    max(lower_actuals) if lower_actuals else None
                ),
                "rejected_upper_estimated_target_tokens": upper,
                "rejected_upper_actual_prompt_tokens": None,
                "estimated_target_interval_width_tokens": upper - lower,
                "exact_boundary_identified": False,
            }
        if accepted and not rejected:
            return {
                "classification": "right_censored_no_context_rejection",
                "requested_max_output_tokens": requested_output_tokens,
                "accepted_lower_actual_prompt_tokens": (
                    max(actual_tokens) if actual_tokens else None
                ),
                "rejected_upper_actual_prompt_tokens": None,
                "exact_boundary_identified": False,
            }
        if rejected and not accepted:
            return {
                "classification": "left_censored_no_accepted_coordinate",
                "requested_max_output_tokens": requested_output_tokens,
                "accepted_lower_actual_prompt_tokens": None,
                "rejected_upper_estimated_target_tokens": min(
                    int(row["estimated_target_prompt_tokens"]) for row in rejected
                ),
                "rejected_upper_actual_prompt_tokens": None,
                "exact_boundary_identified": False,
            }
        return {
            "classification": "unobserved_or_inconclusive",
            "requested_max_output_tokens": requested_output_tokens,
            "accepted_lower_actual_prompt_tokens": None,
            "rejected_upper_actual_prompt_tokens": None,
            "exact_boundary_identified": False,
        }

    def _summarize(self, started_at: str) -> dict[str, Any]:
        by_model: dict[str, Any] = {}
        for model_id in self.config.model_ids:
            rows = [
                row
                for row in self.request_rows.values()
                if row.get("model_id") == model_id
            ]
            outcomes: dict[str, int] = {}
            for row in rows:
                key = str(row.get("coverage_classification") or "unknown")
                outcomes[key] = outcomes.get(key, 0) + 1
            accepted = [row for row in rows if _is_accepted(row)]
            rejected = [row for row in rows if _is_explicit_rejection(row)]
            prompt_rejected = [
                row
                for row in rejected
                if int(row.get("requested_max_output_tokens") or 0)
                == self.config.short_output_tokens
            ]
            actual_tokens = [
                int(row["actual_prompt_tokens_x_axis"])
                for row in accepted
                if row.get("actual_prompt_tokens_x_axis") is not None
            ]
            combined_rows = [
                row
                for row in rows
                if any(
                    "_combined_estimate_" in str(tag)
                    or str(tag) == "observed_combined_transition_bisection"
                    for tag in row.get("coverage_tags", [])
                )
            ]
            planned_rows = sum(
                row.get("model_id") == model_id for row in self.plan_rows.values()
            )
            execution_complete = len(rows) == planned_rows
            conclusive_rows = sum(bool(row.get("coverage_conclusive")) for row in rows)
            scientifically_complete = (
                execution_complete and conclusive_rows == planned_rows
            )
            within_tolerance = [
                bool(row.get("planning_within_tolerance"))
                for row in accepted
                if row.get("planning_within_tolerance") is not None
            ]
            planning_errors = [
                int(row["planning_absolute_error_tokens"])
                for row in accepted
                if row.get("planning_absolute_error_tokens") is not None
            ]
            boundary_observation = self._boundary_observation(
                model_id=model_id,
                requested_output_tokens=self.config.short_output_tokens,
                rows=rows,
            )
            combined_boundary_observation = self._boundary_observation(
                model_id=model_id,
                requested_output_tokens=self.config.combined_output_tokens,
                rows=rows,
            )
            by_model[model_id] = {
                "planned_rows": planned_rows,
                "terminal_rows": len(rows),
                "execution_complete": execution_complete,
                "scientifically_complete": scientifically_complete,
                "conclusive_rows": conclusive_rows,
                "outcomes": dict(sorted(outcomes.items())),
                "highest_accepted_actual_prompt_tokens": (
                    max(actual_tokens) if actual_tokens else None
                ),
                "lowest_prompt_context_rejection_estimated_target_tokens": min(
                    (
                        int(row["estimated_target_prompt_tokens"])
                        for row in prompt_rejected
                    ),
                    default=None,
                ),
                "boundary_observation": boundary_observation,
                "combined_boundary_observation": combined_boundary_observation,
                "accepted_retrieval_pass_rate": (
                    sum(bool(row.get("retrieval_correct")) for row in accepted)
                    / len(accepted)
                    if accepted
                    else None
                ),
                "planning_within_tolerance_rate": (
                    sum(within_tolerance) / len(within_tolerance)
                    if within_tolerance
                    else None
                ),
                "maximum_planning_absolute_error_tokens": (
                    max(planning_errors) if planning_errors else None
                ),
                "combined_boundary_outcomes": {
                    str(tag): row.get("coverage_classification")
                    for row in combined_rows
                    for tag in row.get("coverage_tags", [])
                    if "_combined_estimate_" in str(tag)
                    or str(tag) == "observed_combined_transition_bisection"
                },
                "context_window_anchor_source": CONTEXT_WINDOW_ANCHOR_SOURCES[model_id],
                "calibrated_chars_per_token": self._calibrated_chars_per_token(
                    model_id
                ),
                "estimated_success_cost_usd": sum(
                    float(row.get("estimated_cost_usd") or 0.0) for row in accepted
                ),
                "accounted_cost_usd": sum(
                    float(row.get("accounted_cost_usd") or 0.0) for row in rows
                ),
            }
        execution_complete = all(
            model["execution_complete"] for model in by_model.values()
        )
        scientifically_complete = all(
            model["scientifically_complete"] for model in by_model.values()
        )
        if scientifically_complete:
            status = "scientifically_complete"
        elif execution_complete:
            status = "execution_complete_scientifically_incomplete"
        else:
            status = "execution_incomplete"
        return {
            "schema_version": SUMMARY_SCHEMA,
            "campaign_id": self.campaign_id,
            "campaign_plan_sha256": self.campaign_plan_sha256,
            "started_at": started_at,
            "ended_at": utc_now(),
            "status": status,
            "execution_complete": execution_complete,
            "scientifically_complete": scientifically_complete,
            "model_count": len(self.config.model_ids),
            "fixed_planned_requests": len(self.fixed_probes),
            "total_plan_rows": len(self.plan_rows),
            "terminal_rows": len(self.request_rows),
            "conservative_exposure_usd": self.budget.exposure_usd,
            "max_cost_usd": self.config.max_cost_usd,
            "prior_cost_usd": self.config.prior_cost_usd,
            "all_requests_all_fail_settlement_ceiling_usd": (
                self.all_requests_settlement_ceiling_usd
            ),
            "full_plan_guaranteed_to_fit_budget": (
                self.full_plan_guaranteed_to_fit_budget
            ),
            "parallel_timeout_only_projection_seconds": (
                self.parallel_timeout_only_projection_seconds
            ),
            "first_calibration_header_timeout_projection_seconds": (
                self.first_calibration_header_projection_seconds
            ),
            "serialized_configured_timeout_sum_seconds": (
                self.serialized_configured_timeout_sum_seconds
            ),
            "quota_governor_wait_upper_bound_seconds": None,
            "effective_model_parallelism": self.effective_model_parallelism,
            "latency_measurement_scope": self.latency_measurement_scope,
            "quota_governor": (
                self.quota_governor.snapshot()
                if self.quota_governor is not None
                else None
            ),
            "http_402_latched": self.account_blocked_402,
            "models": by_model,
        }

    async def _run_with_executor(self, executor: RequestExecutor) -> dict[str, Any]:
        started_at = utc_now()
        self._global_semaphore = asyncio.Semaphore(self.config.global_concurrency)
        self.quota_governor = AccountQuotaGovernor(self.config)

        # Request IDs and reservations are already durable. Rehydrate only the
        # last provider-declared account ceilings so unsent work resumes at the
        # observed quota instead of registering hours of fallback-rate waits.
        # No historical remaining balance, cooldown, or request is replayed.
        for row in sorted(
            self.request_rows.values(),
            key=lambda item: str(item.get("ended_at") or ""),
            reverse=True,
        ):
            signals = row.get("server_signals")
            if isinstance(signals, Mapping) and self.quota_governor.restore_observed_limits(
                signals
            ):
                break

        # Bootstrap conservatively: keep calibration requests serialized until
        # both the account request limit and token-per-minute limit have been
        # observed. If DigitalOcean omits either header, all calibrations remain
        # serial and subsequent lanes use the frozen fallback rate.
        remaining_models = list(self.config.model_ids)
        calibrated_models: set[str] = set()
        while remaining_models and not self.quota_governor.bootstrap_ready:
            model_id = remaining_models.pop(0)
            await self._run_calibration(executor, model_id)
            calibrated_models.add(model_id)

        model_lane_semaphore = asyncio.Semaphore(self.config.model_parallelism)

        async def run_lane(model_id: str) -> None:
            async with model_lane_semaphore:
                if model_id not in calibrated_models:
                    await self._run_calibration(executor, model_id)
                await self._run_model_remainder(executor, model_id)

        await asyncio.gather(
            *(run_lane(model_id) for model_id in self.config.model_ids)
        )
        summary = self._summarize(started_at)
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary

    async def run(self, executor: RequestExecutor | None = None) -> dict[str, Any]:
        # Reload every journal only after taking the cross-process lease. A
        # second process therefore cannot use constructor-time stale state to
        # replay a reservation or request completed by the first process.
        with OutputDirectoryLease(self.execution_lease_path):
            self._reload_runtime_state()
            self._write_fixed_plan()
            self._write_or_validate_manifest()
            if executor is not None:
                return await self._run_with_executor(executor)
            credentials = digitalocean_credentials()
            timeout = httpx.Timeout(
                self.config.request_timeout_seconds,
                connect=min(30.0, self.config.request_timeout_seconds),
                read=self.config.request_timeout_seconds,
                write=min(120.0, self.config.request_timeout_seconds),
                pool=self.config.request_timeout_seconds,
            )
            limits = httpx.Limits(
                max_connections=self.config.global_concurrency,
                max_keepalive_connections=self.config.global_concurrency,
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
