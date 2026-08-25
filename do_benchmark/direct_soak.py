"""Direct two-minute endpoint soak confirmation built from completed AIMD evidence.

This standalone module consumes the
append-only artifacts emitted by :mod:`do_benchmark.direct_aimd`, prefers rates
supported by three separated valid/healthy confirmation epochs, and
executes one endpoint-shape cell at a time.  Every soak is one continuous
120-second open-loop arrival schedule, analysed as four predeclared 30-second
arrival cohorts.  A separate semaphore limits concurrency, so provider latency
cannot silently reduce offered load.

The evidence contract is deliberately narrow: a passing cell supports only the
observed two-minute run for that exact endpoint, payload recipe, rate, and time.
It is not evidence about longer durations, other times of day, or untested
loads.  Prompts, outputs, response bodies, credentials, and raw headers are
never persisted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

import httpx

from do_benchmark.core import (
    API_DOC_GENERATED_DATE,
    MODEL_DOC_VERIFIED_DATE,
    PRICING_DOC_DATE,
    BenchmarkTask,
    JsonlJournal,
    MODEL_BY_ID,
    DIGITALOCEAN_HOSTED_MODEL_IDS,
    ModelSpec,
    StreamResult,
    require_digitalocean_hosted_models,
    canonical_json,
    parse_token_usage,
    percentile,
    score_result,
    stable_hash,
    stream_chat_completion,
    utc_now,
)
from do_benchmark.credentials import digitalocean_credentials
from do_benchmark.direct_aimd import (
    BudgetLedger,
    _task_payload,
    conservative_request_cost,
    make_task,
    sanitized_header_signals,
    wilson_interval,
)
from do_benchmark.direct_aimd_reconcile import (
    AIMDReconciliationError,
    verify_reconciliation_receipt,
)
from do_benchmark.timing_audit import audit_row, timing_evidence


REQUEST_SCHEMA = "do_direct_soak_request_v1"
PHASE_SCHEMA = "do_direct_soak_phase_v1"
BLOCK_SCHEMA = "do_direct_soak_analysis_block_v1"
PAIR_SCHEMA = "do_direct_soak_quality_pair_v1"
CELL_SCHEMA = "do_direct_soak_cell_v1"
SUMMARY_SCHEMA = "do_direct_soak_summary_v1"
MANIFEST_SCHEMA = "do_direct_soak_campaign_v1"
PLAN_SCHEMA = "do_direct_soak_plan_v1"
WINDOW_SCHEMA = "do_direct_soak_execution_window_v1"
SOURCE_MANIFEST_SCHEMA = "do_direct_campaign_v1"
SOURCE_SUMMARY_SCHEMA = "do_direct_summary_v1"
SOURCE_EPOCH_SCHEMA = "do_direct_epoch_v1"
SOURCE_REQUEST_SCHEMA = "do_direct_request_v1"
SOURCE_RESERVATION_SCHEMA = "do_direct_reservation_v1"
TASK_RECIPE_VERSION = "direct-aimd-make-task-v1"
SCORER_CONTRACT_VERSION = "core-score-result-v1"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ENDPOINT_FREEZE_PATH = REPOSITORY_ROOT / "config" / "endpoint-freeze.json"
DIRECT_AIMD_SOURCE_PATH = Path(__file__).resolve().with_name("direct_aimd.py")
CORE_SOURCE_PATH = Path(__file__).resolve().with_name("core.py")

SHAPES = ("short_short", "input32k_short", "short_long", "mixed")
TERMINAL_AIMD_SHAPE_STATUSES = frozenset({"complete", "complete_right_censored"})
TIMEOUT_ERROR_TYPES = frozenset(
    {"ReadTimeout", "ConnectTimeout", "PoolTimeout", "TimeoutException", "TimeoutError"}
)

SoakExecutor = Callable[[str, BenchmarkTask, int], Awaitable[StreamResult]]


class SoakPreflightError(RuntimeError):
    """Raised before credentials are loaded when the complete plan is unsafe."""


class OutputDirectoryLease:
    """Non-blocking process lease released automatically when the process exits."""

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
            raise SoakPreflightError(
                "another process holds the soak output-directory execution lease"
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


@dataclass(frozen=True)
class AIMDCandidate:
    model_id: str
    shape: str
    rate_rps: float
    source_target_rate_rps: float
    source_confirmation_realized_rps: tuple[float, ...]
    source_campaign_id: str
    baseline_epoch_id: str | None
    confirmation_epoch_ids: tuple[str, ...]
    source_shape_status: str
    source_evidence_level: str = "three_separated_healthy_confirmations"
    selection_rule: str = (
        "minimum of the controller target and realized open-loop schedule rates "
        "across three valid, healthy confirmations separated by serial sentinels"
    )


@dataclass(frozen=True)
class CandidateDecision:
    model_id: str
    shape: str
    status: str
    reason: str | None
    candidate: AIMDCandidate | None


@dataclass(frozen=True)
class SoakConfig:
    aimd_dir: Path
    output_dir: Path
    model_ids: tuple[str, ...]
    aimd_reconciliation_path: Path | None = None
    prior_lineage_root: Path | None = None
    v3_checkpoint_dir: Path | None = None
    seed: int = 20260823
    soak_seconds: float = 120.0
    analysis_block_seconds: float = 30.0
    analysis_block_count: int = 4
    concurrency_ceiling: int = 128
    quality_pairs_per_cell: int = 4
    recovery_seconds: float = 30.0
    recovery_rate_fraction: float = 0.5
    request_timeout_seconds: float = 180.0
    max_cost_usd: float = 200.0
    prior_cost_usd: float = 0.0
    accept_conditional_prior_exposure_basis: bool = False
    stop_launch_at: datetime | None = None
    hard_campaign_deadline: datetime | None = None
    selected_cells: tuple[str, ...] = ()
    candidate_rate_multiplier: float = 1.0
    completion_attempt_label: str | None = None

    def validate(self) -> None:
        unknown = sorted(set(self.model_ids) - MODEL_BY_ID.keys())
        if unknown:
            raise ValueError(f"unknown DigitalOcean models: {', '.join(unknown)}")
        require_digitalocean_hosted_models(self.model_ids)
        if not self.model_ids:
            raise ValueError("at least one model is required")
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("model IDs must be unique")
        if self.soak_seconds <= 0 or self.analysis_block_seconds <= 0:
            raise ValueError("soak and analysis-block durations must be positive")
        if self.analysis_block_count != 4:
            raise ValueError(
                "the preregistered design requires exactly four analysis blocks"
            )
        if not math.isclose(
            self.soak_seconds,
            self.analysis_block_seconds * self.analysis_block_count,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("soak duration must equal four complete analysis blocks")
        if self.concurrency_ceiling < 1:
            raise ValueError("concurrency_ceiling must be positive")
        if self.quality_pairs_per_cell != self.analysis_block_count:
            raise ValueError(
                "one preregistered quality pair is required per analysis block"
            )
        if self.recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be positive")
        if not 0 < self.recovery_rate_fraction < 1:
            raise ValueError(
                "recovery_rate_fraction must be strictly between zero and one"
            )
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if not 0 < self.candidate_rate_multiplier <= 1:
            raise ValueError("candidate_rate_multiplier must be in (0, 1]")
        selectors = set(self.selected_cells)
        if len(selectors) != len(self.selected_cells):
            raise ValueError("selected_cells must be unique")
        valid_selectors = {
            f"{model_id}:{shape}" for model_id in self.model_ids for shape in SHAPES
        }
        unknown_selectors = sorted(selectors - valid_selectors)
        if unknown_selectors:
            raise ValueError(
                "unknown selected endpoint-shape cells: " + ", ".join(unknown_selectors)
            )
        if self.max_cost_usd <= 0 or self.prior_cost_usd < 0:
            raise ValueError("invalid cost envelope")
        if self.prior_cost_usd > self.max_cost_usd:
            raise ValueError("prior cost already exceeds the campaign cap")
        for label, value in (
            ("stop_launch_at", self.stop_launch_at),
            ("hard_campaign_deadline", self.hard_campaign_deadline),
        ):
            if value is not None and value.tzinfo is None:
                raise ValueError(f"{label} must be timezone-aware")
        if (self.stop_launch_at is None) != (self.hard_campaign_deadline is None):
            raise ValueError(
                "stop_launch_at and hard_campaign_deadline must be supplied together"
            )
        if (
            self.stop_launch_at is not None
            and self.hard_campaign_deadline is not None
            and self.hard_campaign_deadline <= self.stop_launch_at
        ):
            raise ValueError("hard_campaign_deadline must be after stop_launch_at")


@dataclass(frozen=True)
class SoakCellPlan:
    cell_id: str
    model_id: str
    shape: str
    status: str
    blocked_reason: str | None
    candidate_rate_rps: float | None
    low_load_requests: int
    soak_block_request_counts: tuple[int, int, int, int]
    soak_requests: int
    soak_realized_schedule_rps: float | None
    recovery_rate_rps: float | None
    recovery_requests: int
    recovery_realized_schedule_rps: float | None
    total_requests: int
    max_output_tokens: int | None
    worst_case_reserved_cost_usd: float
    candidate_evidence: dict[str, Any] | None
    workload_contract: dict[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_bindings(model_ids: Sequence[str]) -> dict[str, Any]:
    if not ENDPOINT_FREEZE_PATH.is_file():
        raise SoakPreflightError("missing frozen endpoint/documentation contract")
    freeze = _strict_json(ENDPOINT_FREEZE_PATH)
    endpoints = freeze.get("endpoints")
    if not isinstance(endpoints, list):
        raise SoakPreflightError("frozen endpoint contract has no endpoint list")
    frozen_by_id = {
        str(row.get("model_id")): row
        for row in endpoints
        if isinstance(row, Mapping) and isinstance(row.get("model_id"), str)
    }
    selected: list[dict[str, Any]] = []
    for model_id in model_ids:
        spec = MODEL_BY_ID[model_id]
        frozen = frozen_by_id.get(model_id)
        if frozen is None:
            raise SoakPreflightError(
                f"model absent from frozen endpoint inventory: {model_id}"
            )
        mismatches: list[str] = []
        for field in ("input_usd_per_million", "output_usd_per_million"):
            value = frozen.get(field)
            if value is None or not math.isclose(
                float(value), float(getattr(spec, field)), rel_tol=0, abs_tol=1e-12
            ):
                mismatches.append(field)
        frozen_context = frozen.get("context_window")
        if model_id == "kimi-k3":
            if frozen_context is not None or spec.context_window != 65_536:
                mismatches.append("kimi_undocumented_context_probe_anchor")
        elif frozen_context is None or int(frozen_context) != spec.context_window:
            mismatches.append("context_window")
        if mismatches:
            raise SoakPreflightError(
                f"model/freeze contract mismatch for {model_id}: {', '.join(mismatches)}"
            )
        selected.append(
            {
                "model_spec": asdict(spec),
                "frozen_endpoint_row": frozen,
            }
        )
    return {
        "task_recipe_version": TASK_RECIPE_VERSION,
        "task_recipe_source_sha256": _sha256_file(DIRECT_AIMD_SOURCE_PATH),
        "scorer_contract_version": SCORER_CONTRACT_VERSION,
        "scorer_source_sha256": _sha256_file(CORE_SOURCE_PATH),
        "endpoint_freeze_artifact_sha256": _sha256_file(ENDPOINT_FREEZE_PATH),
        "endpoint_freeze_schema_version": freeze.get("schema_version"),
        "documentation_sources": freeze.get("documentation_sources"),
        "documentation_dates": {
            "api_generated": API_DOC_GENERATED_DATE,
            "models_verified": MODEL_DOC_VERIFIED_DATE,
            "pricing_verified": PRICING_DOC_DATE,
        },
        "selected_model_contracts": selected,
    }


def _strict_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SoakPreflightError(f"missing AIMD artifact: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SoakPreflightError(
            f"invalid AIMD artifact {path.name}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise SoakPreflightError(
            f"AIMD artifact {path.name} must contain a JSON object"
        )
    return value


def _strict_jsonl(path: Path, identity_key: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise SoakPreflightError(f"missing AIMD artifact: {path.name}")
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise SoakPreflightError(
                    f"torn AIMD journal {path.name}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise SoakPreflightError(
                    f"AIMD journal {path.name}:{line_number} is not an object"
                )
            identity = value.get(identity_key)
            if not isinstance(identity, str) or not identity:
                raise SoakPreflightError(
                    f"AIMD journal {path.name}:{line_number} lacks {identity_key}"
                )
            if identity in rows:
                raise SoakPreflightError(
                    f"duplicate {identity_key} {identity!r} in AIMD journal {path.name}"
                )
            rows[identity] = value
    return rows


def _positive_finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _nonnegative_finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _complete_usage(row: Mapping[str, Any]) -> bool:
    usage = parse_token_usage(row.get("usage"))
    return (
        "prompt_tokens" in usage
        and "completion_tokens" in usage
        and usage["prompt_tokens"] > 0
        and usage["completion_tokens"] > 0
    )


def _candidate_decision(
    *,
    model_id: str,
    shape: str,
    source_campaign_id: str,
    shape_summary: Mapping[str, Any] | None,
    epochs: Mapping[str, Mapping[str, Any]],
    requests_by_epoch: Mapping[str, Sequence[Mapping[str, Any]]],
) -> CandidateDecision:
    def blocked(reason: str) -> CandidateDecision:
        return CandidateDecision(model_id, shape, "blocked", reason, None)

    if shape_summary is None:
        return blocked("missing_shape_summary")
    if shape_summary.get("status") not in TERMINAL_AIMD_SHAPE_STATUSES:
        return blocked("aimd_shape_not_complete")
    if shape_summary.get("candidate_confirmed_three_separated_epochs") is not True:
        return blocked("three_separated_confirmations_not_attested")
    # Historical source summaries used a misleading ``candidate_sustainable_rps``
    # key for three short AIMD confirmation epochs. Read it only as a backwards-
    # compatibility alias; new summaries emit the accurately scoped name.
    rate = _positive_finite(shape_summary.get("candidate_confirmed_healthy_rps"))
    if rate is None:
        rate = _positive_finite(shape_summary.get("candidate_sustainable_rps"))
    target = _positive_finite(shape_summary.get("confirmation_target_rps"))
    if rate is None or target is None or not math.isclose(rate, target, rel_tol=1e-9):
        return blocked("missing_or_inconsistent_confirmed_candidate_rate")
    epoch_ids_value = shape_summary.get("epoch_ids")
    if not isinstance(epoch_ids_value, list) or not all(
        isinstance(item, str) for item in epoch_ids_value
    ):
        return blocked("missing_ordered_epoch_lineage")
    epoch_ids = list(epoch_ids_value)
    if len(set(epoch_ids)) != len(epoch_ids):
        return blocked("duplicate_epoch_in_shape_lineage")
    lineage: list[Mapping[str, Any]] = []
    for epoch_id in epoch_ids:
        epoch = epochs.get(epoch_id)
        if epoch is None:
            return blocked("shape_lineage_references_missing_epoch")
        if (
            epoch.get("campaign_id") != source_campaign_id
            or epoch.get("model_id") != model_id
            or epoch.get("shape") != shape
        ):
            return blocked("epoch_identity_mismatch")
        lineage.append(epoch)
    baselines = [row for row in lineage if row.get("phase") == "serial_baseline"]
    if len(baselines) != 1 or not all(
        row.get("valid_for_capacity") is True
        and row.get("healthy") is True
        and row.get("arrival_mode") == "serial"
        for row in baselines
    ):
        return blocked("missing_valid_healthy_serial_baseline")

    def receipts_are_complete(epoch: Mapping[str, Any]) -> bool:
        epoch_id = str(epoch.get("epoch_id") or "")
        request_rows = list(requests_by_epoch.get(epoch_id, ()))
        scheduled = epoch.get("scheduled_requests")
        return (
            isinstance(scheduled, int)
            and scheduled >= 1
            and len(request_rows) == scheduled
            and all(
                request.get("campaign_id") == source_campaign_id
                and request.get("model_id") == model_id
                and request.get("shape") == shape
                and request.get("provider_send_attempted") is True
                and request.get("status") == "success"
                and _complete_usage(request)
                for request in request_rows
            )
        )

    if not receipts_are_complete(baselines[0]):
        return blocked("serial_baseline_request_receipts_invalid")
    confirmations = [
        (index, row)
        for index, row in enumerate(lineage)
        if row.get("phase") == "confirmation"
    ]
    if len(confirmations) != 3:
        return blocked("confirmation_epoch_count_not_three")
    confirmation_ids: list[str] = []
    realized_rates: list[float] = []
    for index, row in confirmations:
        offered = _positive_finite(row.get("offered_rps_target"))
        realized = _positive_finite(row.get("offered_rps_realized_schedule"))
        epoch_seconds = _positive_finite(row.get("epoch_seconds"))
        scheduled_requests = row.get("scheduled_requests")
        if (
            offered is None
            or not math.isclose(offered, rate, rel_tol=1e-9)
            or realized is None
            or epoch_seconds is None
            or not isinstance(scheduled_requests, int)
            or scheduled_requests < 1
            or not math.isclose(
                realized,
                scheduled_requests / epoch_seconds,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            or row.get("arrival_mode") != "open_loop"
            or row.get("valid_for_capacity") is not True
            or row.get("healthy") is not True
        ):
            return blocked("confirmation_epoch_not_valid_healthy_at_candidate")
        epoch_id = str(row["epoch_id"])
        confirmation_ids.append(epoch_id)
        realized_rates.append(realized)
        if not receipts_are_complete(row):
            return blocked("confirmation_request_receipts_invalid")
        # The first and second confirmations must each be followed, before the
        # next confirmation, by a valid serial separator.
        if index != confirmations[-1][0]:
            next_index = next(item[0] for item in confirmations if item[0] > index)
            separators = [
                item
                for item in lineage[index + 1 : next_index]
                if item.get("phase") == "confirmation_separator_serial"
            ]
            if len(separators) != 1 or not all(
                item.get("valid_for_capacity") is True
                and item.get("healthy") is True
                and item.get("arrival_mode") == "serial"
                for item in separators
            ):
                return blocked(
                    "confirmation_epochs_not_separated_by_valid_serial_sentinel"
                )
            if not receipts_are_complete(separators[0]):
                return blocked("confirmation_separator_request_receipts_invalid")
    candidate = AIMDCandidate(
        model_id=model_id,
        shape=shape,
        # Never turn finite-window rounding (for example one request in a very
        # short epoch) into a load above the controller's tested target.
        rate_rps=min(target, *realized_rates),
        source_target_rate_rps=rate,
        source_confirmation_realized_rps=(
            realized_rates[0],
            realized_rates[1],
            realized_rates[2],
        ),
        source_campaign_id=source_campaign_id,
        baseline_epoch_id=str(baselines[0]["epoch_id"]),
        confirmation_epoch_ids=(
            confirmation_ids[0],
            confirmation_ids[1],
            confirmation_ids[2],
        ),
        source_shape_status=str(shape_summary["status"]),
    )
    return CandidateDecision(model_id, shape, "ready", None, candidate)


def _reconciled_exploratory_candidate(
    *,
    model_id: str,
    shape: str,
    source_campaign_id: str,
    shape_summary: Mapping[str, Any] | None,
    epochs: Mapping[str, Mapping[str, Any]],
    requests_by_epoch: Mapping[str, Sequence[Mapping[str, Any]]],
) -> CandidateDecision:
    """Select a conservative soak input when AIMD did not confirm a candidate.

    This is enabled only for the exact hash-bound legacy reconciliation.  It
    makes no sustainable-capacity claim: it chooses the lowest fully receipted,
    valid, healthy observed rate and delegates confirmation to the 120-second
    soak itself.
    """

    def blocked(reason: str) -> CandidateDecision:
        return CandidateDecision(model_id, shape, "blocked", reason, None)

    if (
        shape_summary is None
        or shape_summary.get("status") not in TERMINAL_AIMD_SHAPE_STATUSES
    ):
        return blocked("aimd_shape_not_complete")
    epoch_ids = shape_summary.get("epoch_ids")
    if not isinstance(epoch_ids, list) or not all(
        isinstance(value, str) for value in epoch_ids
    ):
        return blocked("missing_ordered_epoch_lineage")
    candidates: list[tuple[float, float, str, str]] = []
    baseline_id: str | None = None
    for epoch_id in epoch_ids:
        epoch = epochs.get(epoch_id)
        if epoch is None or (
            epoch.get("campaign_id") != source_campaign_id
            or epoch.get("model_id") != model_id
            or epoch.get("shape") != shape
        ):
            return blocked("epoch_identity_mismatch")
        request_rows = list(requests_by_epoch.get(epoch_id, ()))
        scheduled = epoch.get("scheduled_requests")
        fully_receipted = (
            isinstance(scheduled, int)
            and scheduled >= 1
            and len(request_rows) == scheduled
            and all(
                request.get("campaign_id") == source_campaign_id
                and request.get("model_id") == model_id
                and request.get("shape") == shape
                and request.get("provider_send_attempted") is True
                and request.get("status") == "success"
                for request in request_rows
            )
        )
        if not fully_receipted:
            continue
        if epoch.get("phase") == "serial_baseline":
            baseline_id = epoch_id
        target = _positive_finite(epoch.get("offered_rps_target"))
        realized = _positive_finite(epoch.get("offered_rps_realized_schedule"))
        if (
            epoch.get("valid_for_capacity") is True
            and epoch.get("healthy") is True
            and target is not None
            and realized is not None
        ):
            observed_rate = (
                min(target, realized)
                if epoch.get("arrival_mode") == "open_loop"
                else target
            )
            candidates.append(
                (observed_rate, target, epoch_id, str(epoch.get("phase")))
            )
    if not candidates:
        return blocked("no_valid_healthy_fully_receipted_epoch_for_exploratory_soak")
    rate, target, epoch_id, phase = min(
        candidates, key=lambda value: (value[0], value[2])
    )
    return CandidateDecision(
        model_id,
        shape,
        "ready",
        None,
        AIMDCandidate(
            model_id=model_id,
            shape=shape,
            rate_rps=rate,
            source_target_rate_rps=target,
            source_confirmation_realized_rps=(rate,),
            source_campaign_id=source_campaign_id,
            baseline_epoch_id=baseline_id,
            confirmation_epoch_ids=(epoch_id,),
            source_shape_status=str(shape_summary["status"]),
            source_evidence_level="single_valid_healthy_epoch_exploratory",
            selection_rule=(
                "exact-reconciled exploratory soak input: lowest positive rate "
                "from any valid, healthy, fully receipted AIMD epoch "
                f"(source phase {phase}); only the two-minute soak may confirm it"
            ),
        ),
    )


def load_aimd_candidates(
    aimd_dir: Path,
    model_ids: Sequence[str],
    *,
    reconciliation_path: Path | None = None,
    prior_lineage_root: Path | None = None,
    v3_checkpoint_dir: Path | None = None,
) -> tuple[list[CandidateDecision], dict[str, Any]]:
    """Load and reconcile completed AIMD artifacts without provider access."""

    paths = {
        "manifest": aimd_dir / "manifest.json",
        "summary": aimd_dir / "summary.json",
        "epochs": aimd_dir / "epochs.jsonl",
        "requests": aimd_dir / "requests.jsonl",
        "reservations": aimd_dir / "reservations.jsonl",
    }
    manifest = _strict_json(paths["manifest"])
    summary = _strict_json(paths["summary"])
    epochs = _strict_jsonl(paths["epochs"], "epoch_id")
    requests = _strict_jsonl(paths["requests"], "request_id")
    reservations = _strict_jsonl(paths["reservations"], "request_id")
    reconciliation: dict[str, Any] | None = None
    if reconciliation_path is not None:
        if prior_lineage_root is None or v3_checkpoint_dir is None:
            raise SoakPreflightError(
                "prior-lineage and v3 checkpoint roots are required with an AIMD "
                "reconciliation receipt"
            )
        try:
            reconciliation = verify_reconciliation_receipt(
                reconciliation_path,
                aimd_dir,
                endpoint_freeze_path=ENDPOINT_FREEZE_PATH,
                prior_lineage_root=prior_lineage_root,
                v3_checkpoint_dir=v3_checkpoint_dir,
            )
        except AIMDReconciliationError as error:
            raise SoakPreflightError(f"AIMD reconciliation failed: {error}") from error
    if manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise SoakPreflightError("unsupported AIMD manifest schema")
    if summary.get("schema_version") != SOURCE_SUMMARY_SCHEMA:
        raise SoakPreflightError("unsupported AIMD summary schema")
    if any(row.get("schema_version") != SOURCE_EPOCH_SCHEMA for row in epochs.values()):
        raise SoakPreflightError("unsupported AIMD epoch schema")
    if any(
        row.get("schema_version") != SOURCE_REQUEST_SCHEMA for row in requests.values()
    ):
        raise SoakPreflightError("unsupported AIMD request schema")
    if any(
        row.get("schema_version") != SOURCE_RESERVATION_SCHEMA
        for row in reservations.values()
    ):
        raise SoakPreflightError("unsupported AIMD reservation schema")
    source_campaign_id = manifest.get("campaign_id")
    if not isinstance(source_campaign_id, str) or not source_campaign_id:
        raise SoakPreflightError("AIMD manifest lacks campaign_id")
    if summary.get("campaign_id") != source_campaign_id:
        raise SoakPreflightError("AIMD summary/manifest campaign IDs do not match")
    if (
        summary.get("status") not in TERMINAL_AIMD_SHAPE_STATUSES
        or summary.get("all_models_complete") is not True
    ):
        raise SoakPreflightError("source AIMD campaign is not scientifically complete")
    for row in epochs.values():
        if row.get("campaign_id") != source_campaign_id:
            raise SoakPreflightError("AIMD epoch from another campaign is present")
    requests_by_epoch: dict[str, list[Mapping[str, Any]]] = {}
    for row in requests.values():
        if row.get("campaign_id") != source_campaign_id:
            raise SoakPreflightError("AIMD request from another campaign is present")
        epoch_id = row.get("epoch_id")
        if isinstance(epoch_id, str):
            requests_by_epoch.setdefault(epoch_id, []).append(row)
    for request_id, row in reservations.items():
        if row.get("campaign_id") != source_campaign_id:
            raise SoakPreflightError(
                "AIMD reservation from another campaign is present"
            )
        if row.get("request_id") != request_id:
            raise SoakPreflightError("AIMD reservation identity mismatch")

    model_specs = manifest.get("model_specs")
    if not isinstance(model_specs, list):
        raise SoakPreflightError("AIMD manifest lacks model_specs")
    source_specs: dict[str, Mapping[str, Any]] = {}
    for row in model_specs:
        if not isinstance(row, Mapping) or not isinstance(row.get("model_id"), str):
            raise SoakPreflightError("invalid AIMD manifest model_specs row")
        source_model_id = str(row["model_id"])
        if source_model_id in source_specs:
            raise SoakPreflightError("duplicate AIMD manifest model_specs row")
        source_specs[source_model_id] = row
    source_contract_drift = False
    for model_id in model_ids:
        expected = asdict(MODEL_BY_ID[model_id])
        observed = source_specs.get(model_id)
        if observed is None:
            raise SoakPreflightError(f"AIMD manifest lacks model contract: {model_id}")
        if canonical_json(dict(observed)) != canonical_json(expected):
            source_contract_drift = True
        if source_contract_drift and reconciliation is None:
            raise SoakPreflightError(
                f"AIMD model contract does not match current frozen model spec: {model_id}"
            )
    if manifest.get("shapes") != list(SHAPES):
        raise SoakPreflightError(
            "AIMD manifest shape contract is not the expected four shapes"
        )

    validated_terminal_costs: dict[str, float] = {}
    if reconciliation is not None:
        settlement = reconciliation.get("settlement")
        if not isinstance(settlement, Mapping):
            raise SoakPreflightError("AIMD reconciliation lacks settlement evidence")
        prior_lineage = reconciliation.get("prior_lineage")
        if not isinstance(prior_lineage, Mapping):
            raise SoakPreflightError("AIMD reconciliation lacks prior-lineage evidence")
        reservation_bound_status = prior_lineage.get("reservation_bound_status")
        if (
            not isinstance(reservation_bound_status, str)
            or not reservation_bound_status
        ):
            raise SoakPreflightError(
                "AIMD reconciliation lacks the prior reservation-bound status"
            )
        summary_exposure = _nonnegative_finite(
            settlement.get("reconciled_cumulative_exposure_usd")
        )
        if summary_exposure is None:
            raise SoakPreflightError("AIMD reconciliation lacks cumulative exposure")
        reconstructed_exposure = summary_exposure
    for request_id, request in requests.items():
        if reconciliation is not None:
            continue
        provider_attempted = request.get("provider_send_attempted") is True
        reservation = reservations.get(request_id)
        if provider_attempted and reservation is None:
            raise SoakPreflightError(
                "AIMD provider-attempted request lacks its pre-send reservation"
            )
        if reservation is not None:
            reserved_cost = _nonnegative_finite(reservation.get("reserved_cost_usd"))
            reserved_tokens = reservation.get("reserved_prompt_tokens")
            if (
                reserved_cost is None
                or not isinstance(reserved_tokens, int)
                or reserved_tokens < 0
            ):
                raise SoakPreflightError(
                    "AIMD reservation has invalid conservative bounds"
                )
            for field, expected in (
                ("epoch_id", request.get("epoch_id")),
                ("model_id", request.get("model_id")),
                ("shape", request.get("shape")),
                ("max_output_tokens", request.get("requested_max_output_tokens")),
            ):
                if canonical_json(reservation.get(field)) != canonical_json(expected):
                    raise SoakPreflightError(
                        f"AIMD request/reservation identity mismatch for {field}"
                    )
            if (
                _nonnegative_finite(request.get("worst_case_reserved_cost_usd"))
                != reserved_cost
                or request.get("reserved_prompt_tokens") != reserved_tokens
            ):
                raise SoakPreflightError(
                    "AIMD request does not retain its exact reservation bounds"
                )
        else:
            reserved_cost = 0.0
            if (
                _nonnegative_finite(request.get("worst_case_reserved_cost_usd")) != 0.0
                or request.get("reserved_prompt_tokens") != 0
            ):
                raise SoakPreflightError("unreserved AIMD request has nonzero bounds")

        usage = parse_token_usage(request.get("usage"))
        usage_reported = bool(
            int(usage.get("prompt_tokens") or 0) > 0
            or int(usage.get("completion_tokens") or 0) > 0
        )
        usage_complete_for_settlement = bool(
            int(usage.get("prompt_tokens") or 0) > 0
            and int(usage.get("completion_tokens") or 0) > 0
        )
        if request.get("status") == "success" and usage_complete_for_settlement:
            model_id = request.get("model_id")
            if not isinstance(model_id, str) or model_id not in MODEL_BY_ID:
                raise SoakPreflightError(
                    "AIMD request references an unknown model contract"
                )
            spec = MODEL_BY_ID[model_id]
            expected_accounted = (
                int(usage.get("prompt_tokens") or 0) * spec.input_usd_per_million
                + int(usage.get("completion_tokens") or 0) * spec.output_usd_per_million
            ) / 1_000_000
            expected_estimated: float | None = expected_accounted
        elif provider_attempted:
            expected_accounted = reserved_cost
            expected_estimated = None
        else:
            expected_accounted = 0.0
            expected_estimated = None
        if (
            request.get("status") == "success"
            and request.get("usage_reported") is not usage_reported
        ):
            raise SoakPreflightError(
                "AIMD usage-reporting settlement flag is inconsistent"
            )
        observed_settlement_flag = request.get("usage_complete_for_settlement")
        if (
            observed_settlement_flag is not None
            and observed_settlement_flag is not usage_complete_for_settlement
        ):
            raise SoakPreflightError(
                "AIMD usage-completeness settlement flag is inconsistent"
            )
        observed_accounted = _nonnegative_finite(request.get("accounted_cost_usd"))
        observed_estimated = request.get("estimated_cost_usd")
        if (
            observed_accounted is None
            or not math.isclose(
                observed_accounted, expected_accounted, rel_tol=1e-12, abs_tol=1e-15
            )
            or (expected_estimated is None and observed_estimated is not None)
            or (
                expected_estimated is not None
                and (
                    _nonnegative_finite(observed_estimated) is None
                    or not math.isclose(
                        float(observed_estimated),
                        expected_estimated,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                )
            )
        ):
            raise SoakPreflightError("AIMD terminal cost accounting is inconsistent")
        validated_terminal_costs[request_id] = expected_accounted

    if reconciliation is None:
        source_prior = _nonnegative_finite(manifest.get("prior_cost_usd"))
        summary_prior = _nonnegative_finite(summary.get("prior_cost_usd"))
        summary_exposure = _nonnegative_finite(summary.get("conservative_exposure_usd"))
        if source_prior is None or summary_prior is None or summary_exposure is None:
            raise SoakPreflightError(
                "AIMD manifest/summary lacks a valid cumulative exposure contract"
            )
        if not math.isclose(source_prior, summary_prior, rel_tol=0, abs_tol=1e-12):
            raise SoakPreflightError(
                "AIMD manifest/summary prior exposure does not match"
            )
        exposure_ids = set(reservations) | set(requests)
        reconstructed_exposure = source_prior
        for request_id in exposure_ids:
            request = requests.get(request_id)
            if request is not None:
                reconstructed_exposure += validated_terminal_costs[request_id]
            else:
                reserved = _nonnegative_finite(
                    reservations[request_id].get("reserved_cost_usd")
                )
                if reserved is None:
                    raise SoakPreflightError(
                        "AIMD reservation lacks valid reserved_cost_usd"
                    )
                reconstructed_exposure += reserved
        if not math.isclose(
            reconstructed_exposure,
            summary_exposure,
            rel_tol=1e-9,
            abs_tol=1e-8,
        ):
            raise SoakPreflightError(
                "AIMD summary exposure does not reconcile with request/reservation journals"
            )

    summary_models: dict[str, Mapping[str, Any]] = {}
    model_rows = summary.get("models")
    if not isinstance(model_rows, list):
        raise SoakPreflightError("AIMD summary models must be a list")
    for row in model_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("model_id"), str):
            raise SoakPreflightError("invalid model row in AIMD summary")
        model_id = str(row["model_id"])
        if model_id in summary_models:
            raise SoakPreflightError(f"duplicate AIMD summary model: {model_id}")
        summary_models[model_id] = row
    for model_id in model_ids:
        model = summary_models.get(model_id)
        if model is None or model.get("status") not in TERMINAL_AIMD_SHAPE_STATUSES:
            raise SoakPreflightError(f"source AIMD model is not complete: {model_id}")

    decisions: list[CandidateDecision] = []
    for model_id in model_ids:
        model = summary_models.get(model_id)
        shapes: dict[str, Mapping[str, Any]] = {}
        if model is not None and isinstance(model.get("shapes"), list):
            for shape_row in model["shapes"]:
                if isinstance(shape_row, Mapping) and isinstance(
                    shape_row.get("shape"), str
                ):
                    shape_name = str(shape_row["shape"])
                    if shape_name in shapes:
                        raise SoakPreflightError(
                            f"duplicate AIMD shape {model_id}/{shape_name}"
                        )
                    shapes[shape_name] = shape_row
        for shape in SHAPES:
            decision = _candidate_decision(
                model_id=model_id,
                shape=shape,
                source_campaign_id=source_campaign_id,
                shape_summary=shapes.get(shape),
                epochs=epochs,
                requests_by_epoch=requests_by_epoch,
            )
            if decision.status != "ready" and reconciliation is not None:
                decision = _reconciled_exploratory_candidate(
                    model_id=model_id,
                    shape=shape,
                    source_campaign_id=source_campaign_id,
                    shape_summary=shapes.get(shape),
                    epochs=epochs,
                    requests_by_epoch=requests_by_epoch,
                )
            decisions.append(decision)
    source = {
        "source_campaign_id": source_campaign_id,
        "artifact_sha256": {name: _sha256_file(path) for name, path in paths.items()},
        "manifest_schema_version": manifest.get("schema_version"),
        "summary_schema_version": summary.get("schema_version"),
        "source_cumulative_exposure_usd": summary_exposure,
        "source_exposure_reconstructed_usd": reconstructed_exposure,
        "reconciliation": (
            None
            if reconciliation is None
            else {
                "schema_version": reconciliation.get("schema_version"),
                "policy_version": reconciliation.get("policy_version"),
                "receipt_sha256": reconciliation.get("receipt_sha256"),
                "performance_evidence_preserved": reconciliation.get(
                    "performance_evidence", {}
                ).get("preserved"),
                "prior_exposure_basis_status": reservation_bound_status,
                "prior_exposure_basis_is_conditional": reservation_bound_status.startswith(
                    "conditional on "
                ),
            }
        ),
        "source_identity": {
            "input_tokens": manifest.get("input_tokens"),
            "long_output_words": manifest.get("long_output_words"),
            "short_max_output_tokens": manifest.get("short_max_output_tokens"),
            "long_max_output_tokens": manifest.get("long_max_output_tokens"),
            "mixed_max_output_tokens": manifest.get("mixed_max_output_tokens"),
        },
    }
    for key, value in source["source_identity"].items():
        if not isinstance(value, int) or value < 1:
            raise SoakPreflightError(f"AIMD manifest lacks valid {key}")
    return decisions, source


def _workload_contract(
    shape: str,
    source_identity: Mapping[str, int],
) -> dict[str, Any]:
    maximums = {
        "short_short": source_identity["short_max_output_tokens"],
        "input32k_short": source_identity["short_max_output_tokens"],
        "short_long": source_identity["long_max_output_tokens"],
        "mixed": source_identity["mixed_max_output_tokens"],
    }
    if shape == "short_short":
        return {
            "workload_shape": shape,
            "input_class": "short",
            "output_class": "short",
            "task_mix": ["exact_text"],
            "streaming": True,
            "requested_max_output_tokens": maximums[shape],
        }
    if shape == "input32k_short":
        return {
            "workload_shape": shape,
            "input_class": "fixed_long_context",
            "planned_input_tokens": source_identity["input_tokens"],
            "output_class": "short",
            "task_mix": ["long_context_retrieval"],
            "streaming": True,
            "requested_max_output_tokens": maximums[shape],
        }
    if shape == "short_long":
        return {
            "workload_shape": shape,
            "input_class": "short",
            "output_class": "controlled_long",
            "target_output_words": source_identity["long_output_words"],
            "task_mix": ["controlled_generation"],
            "streaming": True,
            "requested_max_output_tokens": maximums[shape],
        }
    return {
        "workload_shape": shape,
        "input_class": "heterogeneous",
        "output_class": "heterogeneous",
        "task_mix": [
            "exact_text",
            "context_retrieval_4k",
            "controlled_generation_512_words",
            "structured_json",
            "tool_call",
        ],
        "task_selection": "deterministic_round_robin",
        "streaming": True,
        "requested_max_output_tokens": maximums[shape],
    }


def _scheduled_count(rate_rps: float, seconds: float) -> int:
    return max(1, math.floor(rate_rps * seconds))


def _representative_cost(
    spec: ModelSpec,
    *,
    shape: str,
    source_identity: Mapping[str, int],
    max_output_tokens: int,
) -> tuple[float, int]:
    # Fourteen decimal digits exceed every 11-hex-digit generated task ordinal.
    # For mixed, take the maximum over all five payload families.
    costs: list[tuple[float, int]] = []
    selectors = range(5) if shape == "mixed" else range(1)
    for selector in selectors:
        ordinal = 99_999_999_999_990 + selector
        task = make_task(
            shape=shape,
            ordinal=ordinal,
            input_tokens=source_identity["input_tokens"],
            long_output_words=source_identity["long_output_words"],
        )
        costs.append(conservative_request_cost(spec, task, max_output_tokens))
    return max(costs, key=lambda item: item[0])


def _t_mean_ci95(values: Sequence[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    critical = {2: 12.706, 3: 4.303, 4: 3.182}.get(len(values), 1.96)
    mean = statistics.fmean(values)
    radius = critical * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - radius, mean + radius]


def _dkw_quantile_ci95(values: Sequence[float], quantile: float) -> list[float] | None:
    if not values:
        return None
    epsilon = math.sqrt(math.log(2.0 / 0.05) / (2.0 * len(values)))
    lower_q = max(0.0, quantile - epsilon)
    upper_q = min(1.0, quantile + epsilon)
    lower = percentile(values, lower_q)
    upper = percentile(values, upper_q)
    if lower is None or upper is None:
        return None
    return [lower, upper]


class DirectSoakCampaign:
    """Crash-safe, endpoint-isolated two-minute soak campaign."""

    def __init__(self, config: SoakConfig) -> None:
        config.validate()
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.decisions, self.source = load_aimd_candidates(
            config.aimd_dir,
            config.model_ids,
            reconciliation_path=config.aimd_reconciliation_path,
            prior_lineage_root=config.prior_lineage_root,
            v3_checkpoint_dir=config.v3_checkpoint_dir,
        )
        if config.selected_cells:
            selected = set(config.selected_cells)
            self.decisions = [
                decision
                for decision in self.decisions
                if f"{decision.model_id}:{decision.shape}" in selected
            ]
        if config.candidate_rate_multiplier != 1.0:
            adjusted: list[CandidateDecision] = []
            for decision in self.decisions:
                candidate = decision.candidate
                if candidate is None:
                    adjusted.append(decision)
                    continue
                adjusted_candidate = replace(
                    candidate,
                    rate_rps=(candidate.rate_rps * config.candidate_rate_multiplier),
                    source_evidence_level=(
                        "descending_rate_reconfirmation_from_prior_aimd_candidate"
                    ),
                    selection_rule=(
                        "predeclared descending retry multiplier "
                        f"{config.candidate_rate_multiplier:g} applied to the "
                        "receipt-backed source candidate after a higher-rate soak did "
                        "not pass its complete acceptance/recovery gate"
                    ),
                )
                adjusted.append(replace(decision, candidate=adjusted_candidate))
            self.decisions = adjusted
        source_exposure = float(self.source["source_cumulative_exposure_usd"])
        if config.prior_cost_usd + 1e-12 < source_exposure:
            raise SoakPreflightError(
                "--prior-cost-usd is below the reconciled cumulative exposure in the "
                "source AIMD artifacts"
            )
        self.source_identity: dict[str, int] = dict(self.source["source_identity"])
        self.contract_bindings = _contract_bindings(config.model_ids)
        self.cell_plans = self._build_cell_plans()
        plan_identity = {
            "schema_version": PLAN_SCHEMA,
            "provider_adapter": "digitalocean-openai-compatible-streaming",
            "source": self.source,
            "seed": config.seed,
            "models": list(config.model_ids),
            "soak_seconds": config.soak_seconds,
            "analysis_block_seconds": config.analysis_block_seconds,
            "analysis_block_count": config.analysis_block_count,
            "quality_pairs_per_cell": config.quality_pairs_per_cell,
            "concurrency_ceiling": config.concurrency_ceiling,
            "recovery_seconds": config.recovery_seconds,
            "recovery_rate_fraction": config.recovery_rate_fraction,
            "request_timeout_seconds": config.request_timeout_seconds,
            "max_cost_usd": config.max_cost_usd,
            "prior_cost_usd": config.prior_cost_usd,
            "accept_conditional_prior_exposure_basis": (
                config.accept_conditional_prior_exposure_basis
            ),
            "selected_cells": list(config.selected_cells),
            "candidate_rate_multiplier": config.candidate_rate_multiplier,
            "completion_attempt_label": config.completion_attempt_label,
            "task_recipe_version": TASK_RECIPE_VERSION,
            "contract_bindings": self.contract_bindings,
            "cells": [asdict(cell) for cell in self.cell_plans],
        }
        self.plan_sha256 = _sha256_bytes(canonical_json(plan_identity).encode("utf-8"))
        self.campaign_id = stable_hash(plan_identity, prefix="do-soak-")
        self.plan_identity = {**plan_identity, "plan_sha256": self.plan_sha256}
        self.requests_path = self.output_dir / "requests.jsonl"
        self.phases_path = self.output_dir / "phases.jsonl"
        self.blocks_path = self.output_dir / "analysis-blocks.jsonl"
        self.pairs_path = self.output_dir / "quality-pairs.jsonl"
        self.cells_path = self.output_dir / "cells.jsonl"
        self.reservations_path = self.output_dir / "reservations.jsonl"
        self.execution_windows_path = self.output_dir / "execution-windows.jsonl"
        self.outlier_audit_path = self.output_dir / "outlier-audit.jsonl"
        self.execution_lease_path = self.output_dir / ".execution.lock"
        # Even the plan-only snapshot takes the same short process lease. Live
        # execution reacquires it and reloads all journals immediately before
        # any send, so no process can act from stale spend or completion state.
        with OutputDirectoryLease(self.execution_lease_path):
            self._reload_runtime_state()
            self._write_or_validate_plan()

    def _reload_runtime_state(self) -> None:
        """Reload and reconcile every runtime journal under the process lease."""

        self.requests = self._read_local(self.requests_path, "request_id")
        self.phases = self._read_local(self.phases_path, "phase_id")
        self.blocks = self._read_local(self.blocks_path, "analysis_block_id")
        self.pairs = self._read_local(self.pairs_path, "quality_pair_id")
        self.cells = self._read_local(self.cells_path, "cell_id")
        self.execution_windows = self._read_local(
            self.execution_windows_path, "execution_window_id"
        )
        self.outlier_audit = self._read_local(self.outlier_audit_path, "request_id")
        self.reservations = self._read_local(self.reservations_path, "request_id")
        self._validate_runtime_state()
        self.requests_journal = JsonlJournal(self.requests_path)
        self.phases_journal = JsonlJournal(self.phases_path)
        self.blocks_journal = JsonlJournal(self.blocks_path)
        self.pairs_journal = JsonlJournal(self.pairs_path)
        self.cells_journal = JsonlJournal(self.cells_path)
        self.execution_windows_journal = JsonlJournal(self.execution_windows_path)
        self.outlier_audit_journal = JsonlJournal(self.outlier_audit_path)
        self.budget = BudgetLedger(
            path=self.reservations_path,
            max_cost_usd=self.config.max_cost_usd,
            prior_cost_usd=self.config.prior_cost_usd,
            terminal_rows=self.requests,
        )
        self.preflight = self._preflight(current_exposure_usd=self.budget.exposure_usd)
        self.account_blocked_402 = any(
            row.get("http_status") == 402 for row in self.requests.values()
        )

    @staticmethod
    def _require_fields(
        row: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
    ) -> None:
        for key, value in expected.items():
            observed = row.get(key)
            if canonical_json(observed) != canonical_json(value):
                raise SoakPreflightError(
                    f"resume {label} identity mismatch for {key}: "
                    f"expected {value!r}, observed {observed!r}"
                )

    def _expected_request_index(
        self,
    ) -> dict[str, tuple[SoakCellPlan, str, dict[str, Any]]]:
        expected: dict[str, tuple[SoakCellPlan, str, dict[str, Any]]] = {}
        for cell in self.cell_plans:
            if cell.status != "ready":
                continue
            for phase in (
                "paired_low_load",
                "two_minute_soak",
                "post_soak_recovery",
            ):
                phase_id = self._phase_id(cell, phase)
                for item in self._schedule(cell, phase):
                    request_id = self._request_id(phase_id, int(item["index"]))
                    if request_id in expected:
                        raise SoakPreflightError(
                            "deterministic soak request ID collision"
                        )
                    expected[request_id] = (cell, phase, item)
        return expected

    def _validate_runtime_state(self) -> None:
        """Fail closed before trusting a terminal row, reservation, or summary."""

        summary_path = self.output_dir / "summary.json"
        if (
            not any(
                (
                    self.requests,
                    self.reservations,
                    self.phases,
                    self.blocks,
                    self.pairs,
                    self.cells,
                    self.execution_windows,
                )
            )
            and not summary_path.exists()
        ):
            return
        expected_cells = {
            cell.cell_id: cell for cell in self.cell_plans if cell.status == "ready"
        }
        expected_requests = self._expected_request_index()
        for request_id, row in self.requests.items():
            expected = expected_requests.get(request_id)
            if expected is None:
                raise SoakPreflightError(
                    "resume request is not part of this science plan"
                )
            cell, phase, item = expected
            phase_id = self._phase_id(cell, phase)
            task = self._task_for_schedule(cell, phase, item)
            max_output_tokens = int(cell.max_output_tokens)
            payload = canonical_json(_task_payload(task, max_output_tokens))
            self._require_fields(
                row,
                {
                    "schema_version": REQUEST_SCHEMA,
                    "campaign_id": self.campaign_id,
                    "plan_sha256": self.plan_sha256,
                    "request_id": request_id,
                    "phase_id": phase_id,
                    "cell_id": cell.cell_id,
                    "provider": "digitalocean-serverless-inference",
                    "endpoint_id": cell.model_id,
                    "model_id": cell.model_id,
                    "shape": cell.shape,
                    "phase": phase,
                    "task_id": task.task_id,
                    "task_family": task.family,
                    "request_payload_sha256": _sha256_bytes(payload.encode("utf-8")),
                    "request_payload_bytes": len(payload.encode("utf-8")),
                    "requested_max_output_tokens": max_output_tokens,
                    "quality_pair_id": item.get("quality_pair_id"),
                    "quality_pair_index": item.get("quality_pair_index"),
                    "quality_pair_role": item.get("quality_pair_role"),
                },
                label="request",
            )
            load = row.get("load")
            if not isinstance(load, Mapping):
                raise SoakPreflightError("resume request lacks load identity")
            self._require_fields(
                load,
                {
                    "arrival_mode": "serial" if item.get("serial") else "open_loop",
                    "scheduled_offset_seconds": item["scheduled_offset_seconds"],
                    "concurrency_ceiling": (
                        1 if item.get("serial") else self.config.concurrency_ceiling
                    ),
                },
                label="request load",
            )
            tags = row.get("workload_tags")
            if not isinstance(tags, Mapping):
                raise SoakPreflightError("resume request lacks workload identity")
            self._require_fields(
                tags,
                {
                    "benchmark_lane": "direct_two_minute_soak",
                    "workload_shape": cell.shape,
                    "load_phase": phase,
                    "analysis_block_index": item.get("analysis_block_index"),
                    "paired_analysis_block_index": item.get(
                        "paired_analysis_block_index"
                    ),
                    "candidate_rate_rps": cell.candidate_rate_rps,
                    "soak_realized_schedule_rps": cell.soak_realized_schedule_rps,
                    "recovery_rate_fraction": self.config.recovery_rate_fraction,
                    "streaming": True,
                    "task_recipe_version": TASK_RECIPE_VERSION,
                },
                label="request workload",
            )

        for request_id, row in self.reservations.items():
            expected = expected_requests.get(request_id)
            if expected is None:
                raise SoakPreflightError(
                    "resume reservation is not part of this science plan"
                )
            cell, phase, item = expected
            phase_id = self._phase_id(cell, phase)
            task = self._task_for_schedule(cell, phase, item)
            cost, prompt_tokens = conservative_request_cost(
                MODEL_BY_ID[cell.model_id], task, int(cell.max_output_tokens)
            )
            self._require_fields(
                row,
                {
                    "schema_version": SOURCE_RESERVATION_SCHEMA,
                    "campaign_id": self.campaign_id,
                    "request_id": request_id,
                    "epoch_id": phase_id,
                    "model_id": cell.model_id,
                    "shape": cell.shape,
                    "reserved_cost_usd": cost,
                    "reserved_prompt_tokens": prompt_tokens,
                    "max_output_tokens": int(cell.max_output_tokens),
                },
                label="reservation",
            )
            request = self.requests.get(request_id)
            if request is not None:
                self._require_fields(
                    request,
                    {
                        "worst_case_reserved_cost_usd": cost,
                        "reserved_prompt_tokens": prompt_tokens,
                    },
                    label="reserved request",
                )

        for request_id, row in self.requests.items():
            if request_id not in self.reservations:
                if row.get("provider_send_attempted") is not False:
                    raise SoakPreflightError(
                        "resume provider attempt has no durable pre-send reservation"
                    )
                self._require_fields(
                    row,
                    {
                        "worst_case_reserved_cost_usd": 0.0,
                        "reserved_prompt_tokens": 0,
                    },
                    label="unreserved unsent request",
                )

            reservation = self.reservations.get(request_id)
            reserved_cost = (
                float(reservation["reserved_cost_usd"])
                if reservation is not None
                else 0.0
            )
            status = row.get("status")
            provider_attempted = row.get("provider_send_attempted") is True
            parsed_usage = parse_token_usage(row.get("usage"))
            input_usage_complete = bool(
                status == "success" and int(parsed_usage.get("prompt_tokens") or 0) > 0
            )
            output_usage_complete = bool(
                status == "success"
                and int(parsed_usage.get("completion_tokens") or 0) > 0
            )
            usage_complete = input_usage_complete and output_usage_complete
            if usage_complete:
                spec = MODEL_BY_ID[str(row["model_id"])]
                expected_accounted = (
                    int(parsed_usage["prompt_tokens"]) * spec.input_usd_per_million
                    + int(parsed_usage["completion_tokens"])
                    * spec.output_usd_per_million
                ) / 1_000_000
                expected_estimated: float | None = expected_accounted
            elif provider_attempted:
                if reservation is None:
                    raise SoakPreflightError(
                        "resume attempted request has no conservative reservation"
                    )
                expected_accounted = reserved_cost
                expected_estimated = None
            else:
                expected_accounted = 0.0
                expected_estimated = None
            self._require_fields(
                row,
                {
                    "input_usage_complete": input_usage_complete,
                    "output_usage_complete": output_usage_complete,
                    "usage_complete_for_settlement": usage_complete,
                    "estimated_cost_usd": expected_estimated,
                    "accounted_cost_usd": expected_accounted,
                },
                label="terminal cost accounting",
            )

        for phase_id, row in self.phases.items():
            matches = [
                (cell, phase)
                for cell in expected_cells.values()
                for phase in (
                    "paired_low_load",
                    "two_minute_soak",
                    "post_soak_recovery",
                )
                if self._phase_id(cell, phase) == phase_id
            ]
            if len(matches) != 1:
                raise SoakPreflightError(
                    "resume phase is not part of this science plan"
                )
            cell, phase = matches[0]
            self._require_fields(
                row,
                {
                    "schema_version": PHASE_SCHEMA,
                    "campaign_id": self.campaign_id,
                    "plan_sha256": self.plan_sha256,
                    "phase_id": phase_id,
                    "cell_id": cell.cell_id,
                    "model_id": cell.model_id,
                    "shape": cell.shape,
                    "phase": phase,
                },
                label="phase",
            )

        for block_id, row in self.blocks.items():
            cell = expected_cells.get(str(row.get("cell_id")))
            block_index = row.get("analysis_block_index")
            if (
                cell is None
                or not isinstance(block_index, int)
                or not 0 <= block_index < 4
            ):
                raise SoakPreflightError("resume analysis block is outside this plan")
            expected_id = stable_hash(
                {
                    "campaign_id": self.campaign_id,
                    "cell_id": cell.cell_id,
                    "block_index": block_index,
                },
                prefix="do-soak-block-",
            )
            self._require_fields(
                row,
                {
                    "schema_version": BLOCK_SCHEMA,
                    "campaign_id": self.campaign_id,
                    "plan_sha256": self.plan_sha256,
                    "analysis_block_id": expected_id,
                    "cell_id": cell.cell_id,
                    "model_id": cell.model_id,
                    "shape": cell.shape,
                    "analysis_block_index": block_index,
                },
                label="analysis block",
            )
            if block_id != expected_id:
                raise SoakPreflightError("resume analysis block key mismatch")

        for pair_id, row in self.pairs.items():
            cell = expected_cells.get(str(row.get("cell_id")))
            pair_index = row.get("quality_pair_index")
            if (
                cell is None
                or not isinstance(pair_index, int)
                or not 0 <= pair_index < cell.low_load_requests
            ):
                raise SoakPreflightError("resume quality pair is outside this plan")
            expected_id = self._quality_pair_id(cell, pair_index)
            self._require_fields(
                row,
                {
                    "schema_version": PAIR_SCHEMA,
                    "campaign_id": self.campaign_id,
                    "plan_sha256": self.plan_sha256,
                    "quality_pair_id": expected_id,
                    "cell_id": cell.cell_id,
                    "model_id": cell.model_id,
                    "shape": cell.shape,
                    "analysis_block_index": self._quality_pair_block_index(
                        cell, pair_index
                    ),
                    "quality_pair_index": pair_index,
                },
                label="quality pair",
            )
            if pair_id != expected_id:
                raise SoakPreflightError("resume quality pair key mismatch")

        for cell_id, row in self.cells.items():
            cell = expected_cells.get(cell_id)
            if cell is None:
                raise SoakPreflightError("resume cell is outside this plan")
            self._require_fields(
                row,
                {
                    "schema_version": CELL_SCHEMA,
                    "campaign_id": self.campaign_id,
                    "plan_sha256": self.plan_sha256,
                    "cell_id": cell.cell_id,
                    "model_id": cell.model_id,
                    "shape": cell.shape,
                },
                label="cell",
            )

        for window_id, row in self.execution_windows.items():
            payload = {
                "campaign_id": row.get("campaign_id"),
                "plan_sha256": row.get("plan_sha256"),
                "stop_launch_at": row.get("stop_launch_at"),
                "hard_campaign_deadline": row.get("hard_campaign_deadline"),
                "request_timeout_seconds": row.get("request_timeout_seconds"),
            }
            expected_id = stable_hash(payload, prefix="do-soak-window-")
            self._require_fields(
                row,
                {
                    "schema_version": WINDOW_SCHEMA,
                    "execution_window_id": expected_id,
                    "campaign_id": self.campaign_id,
                    "plan_sha256": self.plan_sha256,
                },
                label="execution window",
            )
            if window_id != expected_id:
                raise SoakPreflightError("resume execution-window key mismatch")

        if summary_path.exists():
            summary = _strict_json(summary_path)
            self._require_fields(
                summary,
                {
                    "schema_version": SUMMARY_SCHEMA,
                    "campaign_id": self.campaign_id,
                    "plan_sha256": self.plan_sha256,
                },
                label="summary",
            )

    async def _record_execution_window(self) -> dict[str, Any]:
        payload = {
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "stop_launch_at": (
                self.config.stop_launch_at.isoformat()
                if self.config.stop_launch_at is not None
                else None
            ),
            "hard_campaign_deadline": (
                self.config.hard_campaign_deadline.isoformat()
                if self.config.hard_campaign_deadline is not None
                else None
            ),
            "request_timeout_seconds": self.config.request_timeout_seconds,
        }
        window_id = stable_hash(payload, prefix="do-soak-window-")
        existing = self.execution_windows.get(window_id)
        if existing is not None:
            return existing
        row = {
            "schema_version": WINDOW_SCHEMA,
            "execution_window_id": window_id,
            **payload,
            "recorded_at": utc_now(),
        }
        await self.execution_windows_journal.append(row)
        self.execution_windows[window_id] = row
        return row

    @staticmethod
    def _read_local(path: Path, identity_key: str) -> dict[str, dict[str, Any]]:
        if not path.exists():
            return {}
        return _strict_jsonl(path, identity_key)

    def _max_output_tokens(self, shape: str) -> int:
        if shape in {"short_short", "input32k_short"}:
            return self.source_identity["short_max_output_tokens"]
        if shape == "short_long":
            return self.source_identity["long_max_output_tokens"]
        return self.source_identity["mixed_max_output_tokens"]

    def _analysis_block_for_offset(self, offset_seconds: float) -> int:
        # The tiny tolerance assigns mathematically exact boundaries (for
        # example 0.06 / 0.02) to the later block despite binary float noise.
        block = math.floor(offset_seconds / self.config.analysis_block_seconds + 1e-12)
        return min(self.config.analysis_block_count - 1, max(0, block))

    def _build_cell_plans(self) -> list[SoakCellPlan]:
        plans: list[SoakCellPlan] = []
        for decision in self.decisions:
            contract = _workload_contract(decision.shape, self.source_identity)
            candidate = decision.candidate
            identity = {
                "source_campaign_id": self.source["source_campaign_id"],
                "model_id": decision.model_id,
                "shape": decision.shape,
                "candidate": asdict(candidate) if candidate else None,
                "task_recipe_version": TASK_RECIPE_VERSION,
            }
            cell_id = stable_hash(identity, prefix="do-soak-cell-")
            if candidate is None:
                plans.append(
                    SoakCellPlan(
                        cell_id=cell_id,
                        model_id=decision.model_id,
                        shape=decision.shape,
                        status="blocked_no_valid_aimd_candidate",
                        blocked_reason=decision.reason,
                        candidate_rate_rps=None,
                        low_load_requests=0,
                        soak_block_request_counts=(0, 0, 0, 0),
                        soak_requests=0,
                        soak_realized_schedule_rps=None,
                        recovery_rate_rps=None,
                        recovery_requests=0,
                        recovery_realized_schedule_rps=None,
                        total_requests=0,
                        max_output_tokens=None,
                        worst_case_reserved_cost_usd=0.0,
                        candidate_evidence=None,
                        workload_contract=contract,
                    )
                )
                continue
            soak_requests = _scheduled_count(
                candidate.rate_rps, self.config.soak_seconds
            )
            block_counts = [0] * self.config.analysis_block_count
            for index in range(soak_requests):
                offset = index / candidate.rate_rps
                if offset >= self.config.soak_seconds:
                    break
                block = self._analysis_block_for_offset(offset)
                block_counts[block] += 1
            required_per_block = [1, 1, 1, 2 if decision.shape == "mixed" else 1]
            if any(
                observed < required
                for observed, required in zip(block_counts, required_per_block)
            ):
                plans.append(
                    SoakCellPlan(
                        cell_id=cell_id,
                        model_id=decision.model_id,
                        shape=decision.shape,
                        status="blocked_candidate_rate_cannot_populate_quality_pairs",
                        blocked_reason=(
                            "global candidate-rate schedule does not place the required "
                            "quality-pair arrivals in every analysis block"
                        ),
                        candidate_rate_rps=candidate.rate_rps,
                        low_load_requests=0,
                        soak_block_request_counts=(
                            block_counts[0],
                            block_counts[1],
                            block_counts[2],
                            block_counts[3],
                        ),
                        soak_requests=soak_requests,
                        soak_realized_schedule_rps=soak_requests
                        / self.config.soak_seconds,
                        recovery_rate_rps=None,
                        recovery_requests=0,
                        recovery_realized_schedule_rps=None,
                        total_requests=0,
                        max_output_tokens=None,
                        worst_case_reserved_cost_usd=0.0,
                        candidate_evidence=asdict(candidate),
                        workload_contract=contract,
                    )
                )
                continue
            recovery_rate = candidate.rate_rps * self.config.recovery_rate_fraction
            recovery_requests = _scheduled_count(
                recovery_rate, self.config.recovery_seconds
            )
            low_load_requests = 5 if decision.shape == "mixed" else 4
            total_requests = low_load_requests + soak_requests + recovery_requests
            max_output_tokens = self._max_output_tokens(decision.shape)
            per_request_cost, _ = _representative_cost(
                MODEL_BY_ID[decision.model_id],
                shape=decision.shape,
                source_identity=self.source_identity,
                max_output_tokens=max_output_tokens,
            )
            plans.append(
                SoakCellPlan(
                    cell_id=cell_id,
                    model_id=decision.model_id,
                    shape=decision.shape,
                    status="ready",
                    blocked_reason=None,
                    candidate_rate_rps=candidate.rate_rps,
                    low_load_requests=low_load_requests,
                    soak_block_request_counts=(
                        block_counts[0],
                        block_counts[1],
                        block_counts[2],
                        block_counts[3],
                    ),
                    soak_requests=soak_requests,
                    soak_realized_schedule_rps=soak_requests / self.config.soak_seconds,
                    recovery_rate_rps=recovery_rate,
                    recovery_requests=recovery_requests,
                    recovery_realized_schedule_rps=recovery_requests
                    / self.config.recovery_seconds,
                    total_requests=total_requests,
                    max_output_tokens=max_output_tokens,
                    worst_case_reserved_cost_usd=total_requests * per_request_cost,
                    candidate_evidence=asdict(candidate),
                    workload_contract=contract,
                )
            )
        return plans

    def _preflight(self, *, current_exposure_usd: float) -> dict[str, Any]:
        ready = [cell for cell in self.cell_plans if cell.status == "ready"]
        blocked = [cell for cell in self.cell_plans if cell.status != "ready"]
        all_failure_new_ceiling = sum(
            cell.worst_case_reserved_cost_usd for cell in ready
        )
        all_failure_total_ceiling = self.config.prior_cost_usd + all_failure_new_ceiling
        maximum_inflight = 0.0
        for cell in ready:
            if cell.total_requests <= 0:
                continue
            per_request = cell.worst_case_reserved_cost_usd / cell.total_requests
            maximum_phase_arrivals = max(
                cell.low_load_requests,
                cell.soak_requests,
                cell.recovery_requests,
            )
            maximum_inflight = max(
                maximum_inflight,
                per_request
                * min(self.config.concurrency_ceiling, maximum_phase_arrivals),
            )
        launch_gate_exposure = current_exposure_usd + maximum_inflight
        reconciled_source = self.source.get("reconciliation") is not None
        reconciliation = self.source.get("reconciliation")
        conditional_prior_basis = bool(
            isinstance(reconciliation, Mapping)
            and reconciliation.get("prior_exposure_basis_is_conditional") is True
        )
        conditional_prior_basis_accepted = bool(
            not conditional_prior_basis
            or self.config.accept_conditional_prior_exposure_basis
        )
        numeric_budget_passes = launch_gate_exposure <= self.config.max_cost_usd + 1e-12
        return {
            "candidate_rule": (
                (
                    "prefer the minimum of the target and three realized schedule "
                    "rates from separated healthy confirmations; for the exact "
                    "hash-bound legacy reconciliation only, otherwise use the lowest "
                    "valid/healthy fully receipted observed rate as an explicitly "
                    "exploratory soak input; only this two-minute soak may confirm it"
                )
                if reconciled_source
                else (
                    "minimum of the AIMD target and three realized schedule rates, "
                    "with three valid/healthy confirmations and two valid serial "
                    "separators; no fallback rate is permitted"
                )
            ),
            "target_cell_count": len(self.decisions),
            "ready_cell_count": len(ready),
            "blocked_cell_count": len(blocked),
            "blocked_cells": [
                {
                    "model_id": cell.model_id,
                    "shape": cell.shape,
                    "reason": cell.blocked_reason,
                }
                for cell in blocked
            ],
            "planned_request_count": sum(cell.total_requests for cell in ready),
            "planned_low_load_requests": sum(cell.low_load_requests for cell in ready),
            "planned_soak_requests": sum(cell.soak_requests for cell in ready),
            "planned_recovery_requests": sum(cell.recovery_requests for cell in ready),
            "new_campaign_all_failure_reservation_ceiling_usd": all_failure_new_ceiling,
            "total_all_failure_reservation_ceiling_usd": all_failure_total_ceiling,
            "all_failure_ceiling_may_exceed_cap": (
                all_failure_total_ceiling > self.config.max_cost_usd + 1e-12
            ),
            "prior_cost_usd": self.config.prior_cost_usd,
            "prior_exposure_basis_status": (
                reconciliation.get("prior_exposure_basis_status")
                if isinstance(reconciliation, Mapping)
                else "unconditional_current-contract_journal_reconstruction"
            ),
            "prior_exposure_basis_is_conditional": conditional_prior_basis,
            "conditional_prior_exposure_basis_explicitly_accepted": (
                self.config.accept_conditional_prior_exposure_basis
                if conditional_prior_basis
                else None
            ),
            "prior_exposure_basis_gate_passes": conditional_prior_basis_accepted,
            "current_settled_plus_reserved_exposure_usd": current_exposure_usd,
            "maximum_possible_inflight_batch_reservation_usd": maximum_inflight,
            "launch_gate_exposure_usd": launch_gate_exposure,
            "max_cost_usd": self.config.max_cost_usd,
            "launch_gate_remaining_margin_usd": self.config.max_cost_usd
            - launch_gate_exposure,
            "complete_coverage_ready": len(blocked) == 0,
            "numeric_budget_passes": numeric_budget_passes,
            "budget_passes": numeric_budget_passes and conditional_prior_basis_accepted,
            "passes": len(blocked) == 0
            and numeric_budget_passes
            and conditional_prior_basis_accepted,
            "cost_method": (
                "disclose the full all-failure ceiling; gate launch on current exposure "
                "plus the largest possible in-flight batch; reserve each request before "
                "send using a padded tokenizer-independent byte bound; retain full "
                "reservations for failed/unknown calls and censor later requests at cap"
            ),
        }

    def _write_or_validate_plan(self) -> None:
        manifest_path = self.output_dir / "manifest.json"
        plan_path = self.output_dir / "plan.json"
        if manifest_path.exists() or plan_path.exists():
            if not manifest_path.is_file() or not plan_path.is_file():
                raise SoakPreflightError(
                    "resume directory has an incomplete plan/manifest pair"
                )
            existing_manifest = _strict_json(manifest_path)
            existing_plan = _strict_json(plan_path)
            if (
                existing_manifest.get("campaign_id") != self.campaign_id
                or existing_manifest.get("plan_sha256") != self.plan_sha256
                or canonical_json(existing_plan) != canonical_json(self.plan_identity)
            ):
                raise SoakPreflightError(
                    "output directory belongs to a different soak science plan"
                )
            return
        plan_path.write_text(
            json.dumps(self.plan_identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "created_at": utc_now(),
            "provider": "digitalocean-serverless-inference",
            "provider_neutral_execution_contract": (
                "the campaign accepts an injected async executor; this CLI adapter uses "
                "DigitalOcean's OpenAI-compatible streaming endpoint"
            ),
            "source_aimd": self.source,
            "preflight": self.preflight,
            "execution_window_contract": {
                "journal": "execution-windows.jsonl",
                "send_cutoff_and_hard_deadline_required_for_execution": True,
                "execution_windows_are_not_science_plan_identity": True,
            },
            "arrival_contract": {
                "soak": "one continuous open-loop schedule",
                "analysis_blocks": 4,
                "analysis_block_seconds": self.config.analysis_block_seconds,
                "concurrency_ceiling": self.config.concurrency_ceiling,
                "coordinated_omission_avoided": True,
            },
            "quality_contract": (
                "one deterministic exact-payload pair per block, plus a fifth mixed/tool "
                "pair in block four: serial low-load first, then identical tagged arrivals"
            ),
            "recovery_contract": {
                "duration_seconds": self.config.recovery_seconds,
                "rate_fraction_of_soak": self.config.recovery_rate_fraction,
            },
            "sanitization": (
                "journals contain hashes, counts, scores, and timing only; no credentials, "
                "prompts, outputs, error bodies, or raw headers"
            ),
            "claim_scope": (
                "Only the observed two-minute run for the exact endpoint, workload recipe, "
                "rate, and execution time. No extrapolation to longer durations, other "
                "times of day, or other offered loads."
            ),
            "capacity_generalization": "none",
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _deadline_reached(self) -> bool:
        cutoff = self.config.stop_launch_at
        return cutoff is not None and datetime.now(timezone.utc) >= cutoff

    def _hard_deadline_reached(self) -> bool:
        cutoff = self.config.hard_campaign_deadline
        return cutoff is not None and datetime.now(timezone.utc) >= cutoff

    def _remaining_request_timeout(self) -> float:
        timeout = self.config.request_timeout_seconds
        deadline = self.config.hard_campaign_deadline
        if deadline is not None:
            timeout = min(
                timeout,
                max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds()),
            )
        return timeout

    def _remaining_hard_deadline_seconds(self) -> float | None:
        deadline = self.config.hard_campaign_deadline
        if deadline is None:
            return None
        return max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())

    def _phase_id(self, cell: SoakCellPlan, phase: str) -> str:
        return stable_hash(
            {
                "campaign_id": self.campaign_id,
                "cell_id": cell.cell_id,
                "phase": phase,
            },
            prefix="do-soak-phase-",
        )

    @staticmethod
    def _request_id(phase_id: str, index: int) -> str:
        return stable_hash(
            {"phase_id": phase_id, "index": index}, prefix="do-soak-request-"
        )

    @staticmethod
    def _quality_pair_block_index(cell: SoakCellPlan, pair_index: int) -> int:
        if cell.shape == "mixed" and pair_index == 4:
            return 3
        return pair_index

    def _quality_pair_id(self, cell: SoakCellPlan, pair_index: int) -> str:
        return stable_hash(
            {
                "campaign_id": self.campaign_id,
                "cell_id": cell.cell_id,
                "quality_pair_index": pair_index,
                "analysis_block_index": self._quality_pair_block_index(
                    cell, pair_index
                ),
            },
            prefix="do-soak-pair-",
        )

    def _ordinal(self, cell: SoakCellPlan, phase: str, index: int) -> int:
        digest = hashlib.sha256(
            f"{self.config.seed}:{cell.cell_id}:{phase}:{index}".encode("utf-8")
        ).hexdigest()
        return int(digest[:11], 16)

    def _pair_ordinal(self, cell: SoakCellPlan, pair_index: int) -> int:
        digest = hashlib.sha256(
            f"{self.config.seed}:{cell.cell_id}:quality-pair:{pair_index}".encode(
                "utf-8"
            )
        ).hexdigest()
        ordinal = int(digest[:11], 16)
        if cell.shape == "mixed":
            ordinal = ordinal - ordinal % 5 + pair_index % 5
        return ordinal

    def _task(
        self,
        cell: SoakCellPlan,
        phase: str,
        index: int,
        pair_index: int | None,
        paired: bool,
    ) -> BenchmarkTask:
        ordinal = (
            self._pair_ordinal(cell, int(pair_index))
            if paired and pair_index is not None
            else self._ordinal(cell, phase, index)
        )
        if cell.shape == "mixed" and not paired:
            ordinal = ordinal - ordinal % 5 + index % 5
        task = make_task(
            shape=cell.shape,
            ordinal=ordinal,
            input_tokens=self.source_identity["input_tokens"],
            long_output_words=self.source_identity["long_output_words"],
        )
        if self.config.completion_attempt_label and not paired:
            # DigitalOcean automatically caches common prefixes for open
            # models. Completion re-soaks therefore vary ordinary traffic at
            # the first prompt token. Exact low-load/near-load quality pairs
            # are deliberately exempt: changing the nonce by phase made the
            # supposedly identical pair payloads unequal and invalidated the
            # causal quality comparison by construction.
            nonce = stable_hash(
                {
                    "campaign_id": self.campaign_id,
                    "cell_id": cell.cell_id,
                    "phase": phase,
                    "index": index,
                    "pair_index": pair_index,
                },
                prefix="UNCACHED-",
            )
            first = task.messages[0]
            content = first.get("content")
            if isinstance(content, str):
                first["content"] = f"Ignore cache marker {nonce}. {content}"
            task.metadata = {
                **task.metadata,
                "cache_intent": "deliberately_uncached_early_nonce",
                "early_nonce_sha256": hashlib.sha256(nonce.encode("utf-8")).hexdigest(),
            }
        elif self.config.completion_attempt_label and paired:
            task.metadata = {
                **task.metadata,
                "cache_intent": "exact_quality_pair_repeat_cache_possible",
            }
        return task

    def _schedule(self, cell: SoakCellPlan, phase: str) -> list[dict[str, Any]]:
        if phase == "paired_low_load":
            return [
                {
                    "index": pair_index,
                    "scheduled_offset_seconds": 0.0,
                    "analysis_block_index": None,
                    "paired_analysis_block_index": self._quality_pair_block_index(
                        cell, pair_index
                    ),
                    "quality_pair_index": pair_index,
                    "quality_pair_id": self._quality_pair_id(cell, pair_index),
                    "quality_pair_role": "low_load",
                    "serial": True,
                }
                for pair_index in range(cell.low_load_requests)
            ]
        if phase == "two_minute_soak":
            schedule: list[dict[str, Any]] = []
            rate = float(cell.candidate_rate_rps)
            pairs_by_block: dict[int, list[int]] = {
                block: [] for block in range(self.config.analysis_block_count)
            }
            for pair_index in range(cell.low_load_requests):
                pairs_by_block[self._quality_pair_block_index(cell, pair_index)].append(
                    pair_index
                )
            used_in_block = [0] * self.config.analysis_block_count
            for index in range(cell.soak_requests):
                offset = index / rate
                if offset >= self.config.soak_seconds:
                    break
                block = self._analysis_block_for_offset(offset)
                position = used_in_block[block]
                used_in_block[block] += 1
                pair_index = (
                    pairs_by_block[block][position]
                    if position < len(pairs_by_block[block])
                    else None
                )
                schedule.append(
                    {
                        "index": index,
                        "scheduled_offset_seconds": offset,
                        "analysis_block_index": block,
                        "paired_analysis_block_index": block
                        if pair_index is not None
                        else None,
                        "quality_pair_index": pair_index,
                        "quality_pair_id": (
                            self._quality_pair_id(cell, pair_index)
                            if pair_index is not None
                            else None
                        ),
                        "quality_pair_role": (
                            "near_load" if pair_index is not None else None
                        ),
                        "serial": False,
                    }
                )
            return schedule
        if phase != "post_soak_recovery":
            raise ValueError(f"unknown soak phase: {phase}")
        rate = float(cell.recovery_rate_rps)
        return [
            {
                "index": index,
                "scheduled_offset_seconds": index / rate,
                "analysis_block_index": None,
                "paired_analysis_block_index": None,
                "quality_pair_index": None,
                "quality_pair_id": None,
                "quality_pair_role": None,
                "serial": False,
            }
            for index in range(cell.recovery_requests)
        ]

    def _task_for_schedule(
        self, cell: SoakCellPlan, phase: str, item: Mapping[str, Any]
    ) -> BenchmarkTask:
        pair_index = item.get("quality_pair_index")
        return self._task(
            cell,
            phase,
            int(item["index"]),
            int(pair_index) if pair_index is not None else None,
            item.get("quality_pair_id") is not None,
        )

    async def _append_request(self, row: dict[str, Any]) -> None:
        request_id = str(row["request_id"])
        if request_id in self.requests:
            return
        await self.requests_journal.append(row)
        self.requests[request_id] = row
        await self.budget.settle(request_id, row)
        if request_id not in self.outlier_audit:
            projected = audit_row(row)
            await self.outlier_audit_journal.append(projected)
            self.outlier_audit[request_id] = projected

    async def _reconcile_outlier_audit(self) -> None:
        """Backfill the derived audit after a crash between the two journals."""

        for request_id, row in self.requests.items():
            if request_id in self.outlier_audit:
                continue
            projected = audit_row(row)
            await self.outlier_audit_journal.append(projected)
            self.outlier_audit[request_id] = projected

    def _base_request_row(
        self,
        *,
        cell: SoakCellPlan,
        phase: str,
        phase_id: str,
        request_id: str,
        task: BenchmarkTask,
        item: Mapping[str, Any],
        started_at: str,
        ended_at: str,
        reserved_cost_usd: float,
        reserved_prompt_tokens: int,
        schedule_lag_seconds: float,
        observed_concurrency: int,
    ) -> dict[str, Any]:
        max_output_tokens = int(cell.max_output_tokens)
        payload = canonical_json(_task_payload(task, max_output_tokens))
        return {
            "schema_version": REQUEST_SCHEMA,
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "request_id": request_id,
            "phase_id": phase_id,
            "cell_id": cell.cell_id,
            "provider": "digitalocean-serverless-inference",
            "endpoint_id": cell.model_id,
            "model_id": cell.model_id,
            "shape": cell.shape,
            "phase": phase,
            "task_id": task.task_id,
            "task_family": task.family,
            "request_payload_sha256": _sha256_bytes(payload.encode("utf-8")),
            "request_payload_bytes": len(payload.encode("utf-8")),
            "requested_max_output_tokens": max_output_tokens,
            "started_at": started_at,
            "ended_at": ended_at,
            "worst_case_reserved_cost_usd": reserved_cost_usd,
            "reserved_prompt_tokens": reserved_prompt_tokens,
            "quality_pair_id": item.get("quality_pair_id"),
            "quality_pair_index": item.get("quality_pair_index"),
            "quality_pair_role": item.get("quality_pair_role"),
            "workload_tags": {
                "benchmark_lane": "direct_two_minute_soak",
                "workload_shape": cell.shape,
                "load_phase": phase,
                "analysis_block_index": item.get("analysis_block_index"),
                "paired_analysis_block_index": item.get("paired_analysis_block_index"),
                "candidate_rate_rps": cell.candidate_rate_rps,
                "soak_realized_schedule_rps": cell.soak_realized_schedule_rps,
                "recovery_rate_fraction": self.config.recovery_rate_fraction,
                "streaming": True,
                "task_recipe_version": TASK_RECIPE_VERSION,
            },
            "load": {
                "arrival_mode": "serial" if item.get("serial") else "open_loop",
                "scheduled_offset_seconds": item["scheduled_offset_seconds"],
                "schedule_lag_seconds": schedule_lag_seconds,
                "concurrency_ceiling": 1
                if item.get("serial")
                else self.config.concurrency_ceiling,
                "observed_concurrency": observed_concurrency,
            },
        }

    def _success_row(
        self,
        *,
        base: dict[str, Any],
        task: BenchmarkTask,
        result: StreamResult,
        spec: ModelSpec,
    ) -> dict[str, Any]:
        usage = parse_token_usage(result.usage)
        complete_for_settlement = (
            "prompt_tokens" in usage
            and "completion_tokens" in usage
            and usage["prompt_tokens"] > 0
            and usage["completion_tokens"] > 0
        )
        actual_cost = None
        if complete_for_settlement:
            actual_cost = (
                usage["prompt_tokens"] * spec.input_usd_per_million
                + usage["completion_tokens"] * spec.output_usd_per_million
            ) / 1_000_000
        accounted = (
            actual_cost
            if actual_cost is not None
            else float(base["worst_case_reserved_cost_usd"])
        )
        quality = score_result(task, result)
        response_fingerprint = {
            "text": result.text,
            "reasoning": result.reasoning_text,
            "tool_calls": result.tool_calls,
        }
        completion_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        input_usage_complete = prompt_tokens is not None and prompt_tokens > 0
        output_usage_complete = completion_tokens is not None and completion_tokens > 0
        monotonic = base.get("monotonic_timestamps_ns") or {}
        measured_timing = timing_evidence(
            result,
            monotonic_started_ns=monotonic.get("request_started_ns"),
            monotonic_ended_ns=monotonic.get("request_ended_ns"),
            intended_cache_state=str(task.metadata.get("cache_intent") or "unknown"),
            sequence_count=int(task.parameters.get("n") or 1),
            streaming=task.parameters.get("stream") is not False,
        )
        return {
            **base,
            "provider_send_attempted": True,
            "status": "success",
            "http_status": result.status_code,
            "finish_reason": result.finish_reason,
            "usage": usage,
            "input_usage_complete": input_usage_complete,
            "output_usage_complete": output_usage_complete,
            "usage_complete_for_settlement": complete_for_settlement,
            "response_sha256": _sha256_bytes(
                canonical_json(response_fingerprint).encode("utf-8")
            ),
            "response_text_bytes": len(result.text.encode("utf-8")),
            "reasoning_text_bytes": len(result.reasoning_text.encode("utf-8")),
            "tool_call_count": len(result.tool_calls),
            "timing": measured_timing,
            "stream": {
                "event_count": result.event_count,
                "first_event_kind": result.first_event_kind,
            },
            "header_signals": sanitized_header_signals(result.response_headers),
            "quality_score": float(quality["quality_score"]),
            "score_kind": str(quality["score_kind"]),
            "estimated_cost_usd": actual_cost,
            "accounted_cost_usd": accounted,
        }

    @staticmethod
    def _failure_row(
        *,
        base: dict[str, Any],
        task: BenchmarkTask,
        error: BaseException | None,
        elapsed_seconds: float,
        status: str,
        provider_send_attempted: bool,
    ) -> dict[str, Any]:
        status_code = getattr(error, "status_code", None) if error is not None else None
        return {
            **base,
            "provider_send_attempted": provider_send_attempted,
            "status": status,
            "http_status": status_code if isinstance(status_code, int) else None,
            "error_type": type(error).__name__ if error is not None else None,
            "usage": {},
            "input_usage_complete": False,
            "output_usage_complete": False,
            "usage_complete_for_settlement": False,
            "timing": {"request_seconds": elapsed_seconds, "ttft_seconds": None},
            "quality_score": 0.0,
            "score_kind": str(task.expected.get("kind") or "unknown"),
            "estimated_cost_usd": None,
            # A failed/unknown call may be partially billed.  Unsent rows have
            # no reservation and therefore zero exposure.
            "accounted_cost_usd": float(base["worst_case_reserved_cost_usd"]),
        }

    async def _append_unsent(
        self,
        *,
        cell: SoakCellPlan,
        phase: str,
        phase_id: str,
        request_id: str,
        task: BenchmarkTask,
        item: Mapping[str, Any],
        status: str,
        provider_send_attempted: bool = False,
    ) -> dict[str, Any]:
        reservation = self.budget.reservations.get(request_id)
        reserved_cost = (
            float(reservation.get("reserved_cost_usd") or 0.0) if reservation else 0.0
        )
        reserved_tokens = (
            int(reservation.get("reserved_prompt_tokens") or 0) if reservation else 0
        )
        now = utc_now()
        base = self._base_request_row(
            cell=cell,
            phase=phase,
            phase_id=phase_id,
            request_id=request_id,
            task=task,
            item=item,
            started_at=now,
            ended_at=now,
            reserved_cost_usd=reserved_cost,
            reserved_prompt_tokens=reserved_tokens,
            schedule_lag_seconds=0.0,
            observed_concurrency=0,
        )
        row = self._failure_row(
            base=base,
            task=task,
            error=None,
            elapsed_seconds=0.0,
            status=status,
            provider_send_attempted=provider_send_attempted,
        )
        await self._append_request(row)
        return row

    def _phase_summary(
        self,
        *,
        cell: SoakCellPlan,
        phase: str,
        phase_id: str,
        rows: Sequence[Mapping[str, Any]],
        schedule: Sequence[Mapping[str, Any]],
        elapsed_seconds: float,
        max_observed_concurrency: int,
    ) -> dict[str, Any]:
        successes = [row for row in rows if row.get("status") == "success"]
        provider_attempts = [row for row in rows if row.get("provider_send_attempted")]
        input_usage_complete_count = sum(
            row.get("input_usage_complete") is True for row in successes
        )
        output_usage_complete_count = sum(
            row.get("output_usage_complete") is True for row in successes
        )
        input_usage_complete = input_usage_complete_count == len(successes)
        output_usage_complete = output_usage_complete_count == len(successes)
        prompt_tokens = sum(
            int(parse_token_usage(row.get("usage")).get("prompt_tokens") or 0)
            for row in successes
            if row.get("input_usage_complete") is True
        )
        completion_tokens = sum(
            int(parse_token_usage(row.get("usage")).get("completion_tokens") or 0)
            for row in successes
            if row.get("output_usage_complete") is True
        )
        latencies = [
            float(row["timing"]["request_seconds"])
            for row in successes
            if row.get("timing", {}).get("request_seconds") is not None
        ]
        ttfts = [
            float(row["timing"]["ttft_seconds"])
            for row in successes
            if row.get("timing", {}).get("ttft_seconds") is not None
        ]
        quality_passes = sum(
            float(row.get("quality_score") or 0.0) >= 0.999999 for row in successes
        )
        total = len(rows)
        elapsed_minutes = elapsed_seconds / 60.0 if elapsed_seconds > 0 else 0.0
        status = "complete" if total == len(schedule) else "incomplete"
        if any(not row.get("provider_send_attempted") for row in rows):
            status = "incomplete"
        target_rate = (
            cell.candidate_rate_rps
            if phase == "two_minute_soak"
            else (cell.recovery_rate_rps if phase == "post_soak_recovery" else None)
        )
        nominal_window = (
            self.config.soak_seconds
            if phase == "two_minute_soak"
            else (
                self.config.recovery_seconds if phase == "post_soak_recovery" else None
            )
        )
        return {
            "schema_version": PHASE_SCHEMA,
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "phase_id": phase_id,
            "cell_id": cell.cell_id,
            "provider": "digitalocean-serverless-inference",
            "model_id": cell.model_id,
            "shape": cell.shape,
            "phase": phase,
            "status": status,
            "candidate_rate_rps": cell.candidate_rate_rps,
            "offered_rps_target": target_rate,
            "offered_rps_realized_schedule": (
                len(schedule) / nominal_window
                if nominal_window is not None and nominal_window > 0
                else None
            ),
            "scheduled_requests": len(schedule),
            "completed_request_rows": total,
            "provider_send_attempts": len(provider_attempts),
            "successes": len(successes),
            "success_rate": len(successes) / total if total else 0.0,
            "success_rate_ci95_wilson": wilson_interval(len(successes), total),
            "quality_passes": quality_passes,
            "quality_pass_rate": quality_passes / len(successes) if successes else 0.0,
            "quality_pass_rate_ci95_wilson": wilson_interval(
                quality_passes, len(successes)
            ),
            "elapsed_seconds_including_drain": elapsed_seconds,
            "successful_rpm": len(successes) / elapsed_minutes
            if elapsed_minutes
            else 0.0,
            "successful_rows_with_complete_input_usage": input_usage_complete_count,
            "successful_rows_with_complete_output_usage": output_usage_complete_count,
            "input_usage_complete_for_all_successes": input_usage_complete,
            "output_usage_complete_for_all_successes": output_usage_complete,
            "effective_input_tpm": (
                prompt_tokens / elapsed_minutes
                if input_usage_complete and elapsed_minutes
                else (0.0 if input_usage_complete else None)
            ),
            "effective_output_tpm": (
                completion_tokens / elapsed_minutes
                if output_usage_complete and elapsed_minutes
                else (0.0 if output_usage_complete else None)
            ),
            "ttft_p50_seconds": percentile(ttfts, 0.50),
            "ttft_p95_seconds": percentile(ttfts, 0.95),
            "ttft_p95_ci95_dkw_seconds": _dkw_quantile_ci95(ttfts, 0.95),
            "latency_p50_seconds": percentile(latencies, 0.50),
            "latency_p95_seconds": percentile(latencies, 0.95),
            "latency_p95_ci95_dkw_seconds": _dkw_quantile_ci95(latencies, 0.95),
            "max_observed_concurrency": max_observed_concurrency,
            "http_429": sum(row.get("http_status") == 429 for row in rows),
            "http_5xx": sum(
                isinstance(row.get("http_status"), int) and row["http_status"] >= 500
                for row in rows
            ),
            "timeouts": sum(
                str(row.get("error_type")) in TIMEOUT_ERROR_TYPES for row in rows
            ),
            "accounted_cost_usd": sum(
                float(row.get("accounted_cost_usd") or 0.0) for row in rows
            ),
            "claim_scope": "observed phase only",
        }

    def _analysis_block_summary(
        self,
        *,
        cell: SoakCellPlan,
        block_index: int,
        rows: Sequence[Mapping[str, Any]],
        baseline: Mapping[str, Any],
    ) -> dict[str, Any]:
        block_id = stable_hash(
            {
                "campaign_id": self.campaign_id,
                "cell_id": cell.cell_id,
                "block_index": block_index,
            },
            prefix="do-soak-block-",
        )
        successes = [row for row in rows if row.get("status") == "success"]
        input_usage_complete_count = sum(
            row.get("input_usage_complete") is True for row in successes
        )
        output_usage_complete_count = sum(
            row.get("output_usage_complete") is True for row in successes
        )
        input_usage_complete = input_usage_complete_count == len(successes)
        output_usage_complete = output_usage_complete_count == len(successes)
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
            float(row.get("load", {}).get("schedule_lag_seconds") or 0.0)
            for row in rows
        ]
        prompt_tokens = sum(
            int(parse_token_usage(row.get("usage")).get("prompt_tokens") or 0)
            for row in successes
            if row.get("input_usage_complete") is True
        )
        output_tokens = sum(
            int(parse_token_usage(row.get("usage")).get("completion_tokens") or 0)
            for row in successes
            if row.get("output_usage_complete") is True
        )
        total = len(rows)
        quality_passes = sum(
            float(row.get("quality_score") or 0.0) >= 0.999999 for row in successes
        )
        http_429 = sum(row.get("http_status") == 429 for row in rows)
        http_5xx = sum(
            isinstance(row.get("http_status"), int) and row["http_status"] >= 500
            for row in rows
        )
        timeouts = sum(
            str(row.get("error_type")) in TIMEOUT_ERROR_TYPES for row in rows
        )
        scheduled = cell.soak_block_request_counts[block_index]
        ttft_p95 = percentile(ttfts, 0.95)
        latency_p95 = percentile(latencies, 0.95)
        baseline_ttft = baseline.get("ttft_p95_seconds")
        baseline_latency = baseline.get("latency_p95_seconds")
        midpoint = max(1, len(lags) // 2)
        queue_growth = (percentile(lags[midpoint:], 0.5) or 0.0) - (
            percentile(lags[:midpoint], 0.5) or 0.0
        )
        reasons: list[str] = []
        if total != scheduled or any(
            not row.get("provider_send_attempted") for row in rows
        ):
            reasons.append("arrival_cohort_incomplete_or_contains_local_nonsend")
        success_rate = len(successes) / total if total else 0.0
        if success_rate < 0.99:
            reasons.append("success_rate_below_0.99")
        if total and (http_5xx + timeouts) / total > 0.01:
            reasons.append("combined_timeout_5xx_rate_above_0.01")
        if total and http_429 / total > 0.01:
            reasons.append("rate_limit_rate_above_0.01")
        if (
            baseline_ttft is not None
            and ttft_p95 is not None
            and ttft_p95 > 2 * baseline_ttft
        ):
            reasons.append("ttft_p95_above_2x_paired_low_load_phase")
        if (
            baseline_latency is not None
            and latency_p95 is not None
            and latency_p95 > 2 * baseline_latency
        ):
            reasons.append("latency_p95_above_2x_paired_low_load_phase")
        queue_limit = max(0.25, 1.0 / float(cell.candidate_rate_rps))
        if queue_growth > queue_limit:
            reasons.append("arrival_queue_growth")
        pair_indices = [
            pair_index
            for pair_index in range(cell.low_load_requests)
            if self._quality_pair_block_index(cell, pair_index) == block_index
        ]
        pair_details: list[dict[str, Any]] = []
        for pair_index in pair_indices:
            pair_id = self._quality_pair_id(cell, pair_index)
            pair_rows = [
                row
                for row in self.requests.values()
                if row.get("quality_pair_id") == pair_id
            ]
            low_rows = [
                row for row in pair_rows if row.get("quality_pair_role") == "low_load"
            ]
            near_rows = [
                row for row in pair_rows if row.get("quality_pair_role") == "near_load"
            ]
            complete = len(low_rows) == 1 and len(near_rows) == 1
            low = low_rows[0] if len(low_rows) == 1 else None
            near = near_rows[0] if len(near_rows) == 1 else None
            payload_match = bool(
                complete
                and low.get("request_payload_sha256")
                == near.get("request_payload_sha256")
            )
            low_score = float(low.get("quality_score") or 0.0) if low else None
            near_score = float(near.get("quality_score") or 0.0) if near else None
            delta = (
                near_score - low_score
                if low_score is not None and near_score is not None
                else None
            )
            low_pass = bool(
                low is not None
                and low.get("status") == "success"
                and low_score is not None
                and low_score >= 0.999999
            )
            near_pass = bool(
                near is not None
                and near.get("status") == "success"
                and near_score is not None
                and near_score >= 0.999999
            )
            if not complete:
                reasons.append("paired_quality_receipts_incomplete")
            elif not payload_match:
                reasons.append("paired_quality_payload_hash_mismatch")
            if complete and not low_pass:
                reasons.append("paired_low_load_quality_failure")
            if complete and not near_pass:
                reasons.append("paired_near_load_quality_failure")
            if delta is not None and delta < -1e-12:
                reasons.append("paired_quality_regression_near_vs_low")
            pair_details.append(
                {
                    "quality_pair_id": pair_id,
                    "quality_pair_index": pair_index,
                    "task_family": low.get("task_family") if low is not None else None,
                    "complete": complete,
                    "exact_payload_hash_match": payload_match,
                    "low_load_quality_score": low_score,
                    "near_load_quality_score": near_score,
                    "low_load_quality_pass": low_pass,
                    "near_load_quality_pass": near_pass,
                    "quality_delta_near_minus_low": delta,
                }
            )
        reasons = list(dict.fromkeys(reasons))
        interval_minutes = self.config.analysis_block_seconds / 60.0
        cohort_drain_seconds = max(
            self.config.analysis_block_seconds,
            max(
                (
                    float(row.get("load", {}).get("scheduled_offset_seconds") or 0.0)
                    - block_index * self.config.analysis_block_seconds
                    + float(row.get("load", {}).get("schedule_lag_seconds") or 0.0)
                    + float(row.get("timing", {}).get("request_seconds") or 0.0)
                    for row in rows
                ),
                default=0.0,
            ),
        )
        drain_minutes = cohort_drain_seconds / 60.0
        return {
            "schema_version": BLOCK_SCHEMA,
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "analysis_block_id": block_id,
            "cell_id": cell.cell_id,
            "provider": "digitalocean-serverless-inference",
            "model_id": cell.model_id,
            "shape": cell.shape,
            "phase": "two_minute_soak",
            "analysis_block_index": block_index,
            "analysis_block_seconds": self.config.analysis_block_seconds,
            "candidate_rate_rps": cell.candidate_rate_rps,
            "scheduled_requests": scheduled,
            "completed_request_rows": total,
            "successes": len(successes),
            "success_rate": success_rate,
            "success_rate_ci95_wilson": wilson_interval(len(successes), total),
            "quality_passes": quality_passes,
            "quality_pass_rate": quality_passes / len(successes) if successes else 0.0,
            "quality_pass_rate_ci95_wilson": wilson_interval(
                quality_passes, len(successes)
            ),
            "offered_rps_realized_schedule": scheduled
            / self.config.analysis_block_seconds,
            "successful_rpm_per_predeclared_window": (
                len(successes) / interval_minutes if interval_minutes else 0.0
            ),
            "successful_rows_with_complete_input_usage": input_usage_complete_count,
            "successful_rows_with_complete_output_usage": output_usage_complete_count,
            "input_usage_complete_for_all_successes": input_usage_complete,
            "output_usage_complete_for_all_successes": output_usage_complete,
            "effective_input_tpm_per_predeclared_window": (
                prompt_tokens / interval_minutes
                if input_usage_complete and interval_minutes
                else (0.0 if input_usage_complete else None)
            ),
            "effective_output_tpm_per_predeclared_window": (
                output_tokens / interval_minutes
                if output_usage_complete and interval_minutes
                else (0.0 if output_usage_complete else None)
            ),
            "arrival_cohort_elapsed_seconds_including_drain": cohort_drain_seconds,
            "arrival_cohort_successful_rpm_including_drain": (
                len(successes) / drain_minutes if drain_minutes else 0.0
            ),
            "arrival_cohort_effective_input_tpm_including_drain": (
                prompt_tokens / drain_minutes
                if input_usage_complete and drain_minutes
                else (0.0 if input_usage_complete else None)
            ),
            "arrival_cohort_effective_output_tpm_including_drain": (
                output_tokens / drain_minutes
                if output_usage_complete and drain_minutes
                else (0.0 if output_usage_complete else None)
            ),
            "ttft_p50_seconds": percentile(ttfts, 0.50),
            "ttft_p95_seconds": ttft_p95,
            "ttft_p95_ci95_dkw_seconds": _dkw_quantile_ci95(ttfts, 0.95),
            "latency_p50_seconds": percentile(latencies, 0.50),
            "latency_p95_seconds": latency_p95,
            "latency_p95_ci95_dkw_seconds": _dkw_quantile_ci95(latencies, 0.95),
            "schedule_lag_p95_seconds": percentile(lags, 0.95),
            "queue_growth_late_minus_early_median_seconds": queue_growth,
            "http_429": http_429,
            "http_5xx": http_5xx,
            "timeouts": timeouts,
            "quality_pair_count": len(pair_details),
            "quality_pairs": pair_details,
            "predeclared_acceptance_pass": not reasons,
            "acceptance_reasons": reasons,
            "claim_scope": "this predeclared 30-second arrival cohort only",
        }

    async def _ensure_analysis_blocks(
        self,
        *,
        cell: SoakCellPlan,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        baseline = self.phases.get(self._phase_id(cell, "paired_low_load"))
        if baseline is None:
            raise SoakPreflightError(
                "cannot summarize soak blocks without the paired low-load phase"
            )
        for block_index in range(self.config.analysis_block_count):
            block_rows = [
                row
                for row in rows
                if row.get("workload_tags", {}).get("analysis_block_index")
                == block_index
            ]
            block = self._analysis_block_summary(
                cell=cell,
                block_index=block_index,
                rows=block_rows,
                baseline=baseline,
            )
            block_id = str(block["analysis_block_id"])
            if block_id not in self.blocks:
                await self.blocks_journal.append(block)
                self.blocks[block_id] = block

    async def _run_phase(
        self,
        executor: SoakExecutor,
        *,
        cell: SoakCellPlan,
        phase: str,
    ) -> dict[str, Any]:
        phase_id = self._phase_id(cell, phase)
        existing_phase = self.phases.get(phase_id)
        if existing_phase is not None:
            return existing_phase
        schedule = self._schedule(cell, phase)
        request_ids = [
            self._request_id(phase_id, int(item["index"])) for item in schedule
        ]
        tasks = [self._task_for_schedule(cell, phase, item) for item in schedule]
        terminal_count = sum(request_id in self.requests for request_id in request_ids)
        reserved_without_terminal = sum(
            request_id in self.budget.reservations and request_id not in self.requests
            for request_id in request_ids
        )
        serial = phase == "paired_low_load"
        nominal_seconds = (
            0.0
            if serial
            else (
                self.config.soak_seconds
                if phase == "two_minute_soak"
                else self.config.recovery_seconds
            )
        )
        if terminal_count == len(schedule) and reserved_without_terminal == 0:
            rows = [self.requests[request_id] for request_id in request_ids]
            if serial:
                elapsed = sum(
                    float(row.get("load", {}).get("schedule_lag_seconds") or 0.0)
                    + float(row.get("timing", {}).get("request_seconds") or 0.0)
                    for row in rows
                )
            else:
                elapsed = max(
                    nominal_seconds,
                    max(
                        (
                            float(
                                row.get("load", {}).get("scheduled_offset_seconds")
                                or 0.0
                            )
                            + float(
                                row.get("load", {}).get("schedule_lag_seconds") or 0.0
                            )
                            + float(row.get("timing", {}).get("request_seconds") or 0.0)
                            for row in rows
                        ),
                        default=0.0,
                    ),
                )
            summary = self._phase_summary(
                cell=cell,
                phase=phase,
                phase_id=phase_id,
                rows=rows,
                schedule=schedule,
                elapsed_seconds=max(elapsed, 1e-9),
                max_observed_concurrency=max(
                    (
                        int(row.get("load", {}).get("observed_concurrency") or 0)
                        for row in rows
                    ),
                    default=0,
                ),
            )
            summary["reconstructed_from_complete_terminal_request_rows"] = True
            await self.phases_journal.append(summary)
            self.phases[phase_id] = summary
            if phase == "two_minute_soak":
                await self._ensure_analysis_blocks(cell=cell, rows=rows)
            return summary

        if terminal_count or reserved_without_terminal:
            rows: list[dict[str, Any]] = []
            for item, request_id, task in zip(schedule, request_ids, tasks):
                if request_id in self.requests:
                    rows.append(self.requests[request_id])
                    continue
                status = (
                    "unknown_prior_reservation"
                    if request_id in self.budget.reservations
                    else "not_launched_after_interrupted_phase"
                )
                rows.append(
                    await self._append_unsent(
                        cell=cell,
                        phase=phase,
                        phase_id=phase_id,
                        request_id=request_id,
                        task=task,
                        item=item,
                        status=status,
                        provider_send_attempted=request_id in self.budget.reservations,
                    )
                )
            elapsed = max(
                nominal_seconds,
                max(
                    (
                        float(
                            row.get("load", {}).get("scheduled_offset_seconds") or 0.0
                        )
                        + float(row.get("load", {}).get("schedule_lag_seconds") or 0.0)
                        + float(row.get("timing", {}).get("request_seconds") or 0.0)
                        for row in rows
                    ),
                    default=0.0,
                ),
            )
            summary = self._phase_summary(
                cell=cell,
                phase=phase,
                phase_id=phase_id,
                rows=rows,
                schedule=schedule,
                elapsed_seconds=max(elapsed, 1e-9),
                max_observed_concurrency=max(
                    (
                        int(row.get("load", {}).get("observed_concurrency") or 0)
                        for row in rows
                    ),
                    default=0,
                ),
            )
            summary["status"] = "incomplete_interrupted_no_replay"
            await self.phases_journal.append(summary)
            self.phases[phase_id] = summary
            return summary

        # Do not begin a long open-loop phase unless its final scheduled arrival
        # fits inside the provider-send window.  This avoids planned partial soaks.
        if not serial and self.config.stop_launch_at is not None:
            last_offset = max(
                (float(item["scheduled_offset_seconds"]) for item in schedule),
                default=0.0,
            )
            remaining = (
                self.config.stop_launch_at - datetime.now(timezone.utc)
            ).total_seconds()
            if remaining <= last_offset:
                return {
                    "schema_version": PHASE_SCHEMA,
                    "campaign_id": self.campaign_id,
                    "phase_id": phase_id,
                    "cell_id": cell.cell_id,
                    "model_id": cell.model_id,
                    "shape": cell.shape,
                    "phase": phase,
                    "status": "not_started_send_window_insufficient",
                    "scheduled_requests": len(schedule),
                }
        if not serial and self.config.hard_campaign_deadline is not None:
            last_offset = max(
                (float(item["scheduled_offset_seconds"]) for item in schedule),
                default=0.0,
            )
            hard_remaining = self._remaining_hard_deadline_seconds() or 0.0
            if hard_remaining <= last_offset:
                return {
                    "schema_version": PHASE_SCHEMA,
                    "campaign_id": self.campaign_id,
                    "phase_id": phase_id,
                    "cell_id": cell.cell_id,
                    "model_id": cell.model_id,
                    "shape": cell.shape,
                    "phase": phase,
                    "status": "not_started_hard_deadline_insufficient",
                    "scheduled_requests": len(schedule),
                }

        semaphore = asyncio.Semaphore(1 if serial else self.config.concurrency_ceiling)
        rows: list[dict[str, Any]] = []
        rows_lock = asyncio.Lock()
        active = 0
        max_active = 0
        active_lock = asyncio.Lock()
        started_perf = time.perf_counter()

        async def one(
            item: Mapping[str, Any], request_id: str, task: BenchmarkTask
        ) -> None:
            nonlocal active, max_active
            scheduled_offset = float(item["scheduled_offset_seconds"])
            hard_deadline_expired_during_arrival_wait = False
            if not serial:
                delay = started_perf + scheduled_offset - time.perf_counter()
                if delay > 0:
                    remaining = self._remaining_hard_deadline_seconds()
                    if remaining is not None and remaining <= delay:
                        if remaining > 0:
                            await asyncio.sleep(remaining)
                        hard_deadline_expired_during_arrival_wait = True
                    else:
                        await asyncio.sleep(delay)
            scheduled_at = (
                time.perf_counter() if serial else started_perf + scheduled_offset
            )
            async with semaphore:
                if hard_deadline_expired_during_arrival_wait:
                    row = await self._append_unsent(
                        cell=cell,
                        phase=phase,
                        phase_id=phase_id,
                        request_id=request_id,
                        task=task,
                        item=item,
                        status="skipped_hard_campaign_deadline",
                    )
                    async with rows_lock:
                        rows.append(row)
                    return
                if self._deadline_reached():
                    row = await self._append_unsent(
                        cell=cell,
                        phase=phase,
                        phase_id=phase_id,
                        request_id=request_id,
                        task=task,
                        item=item,
                        status="skipped_send_deadline",
                    )
                    async with rows_lock:
                        rows.append(row)
                    return
                if self._hard_deadline_reached():
                    row = await self._append_unsent(
                        cell=cell,
                        phase=phase,
                        phase_id=phase_id,
                        request_id=request_id,
                        task=task,
                        item=item,
                        status="skipped_hard_campaign_deadline",
                    )
                    async with rows_lock:
                        rows.append(row)
                    return
                if self.account_blocked_402:
                    row = await self._append_unsent(
                        cell=cell,
                        phase=phase,
                        phase_id=phase_id,
                        request_id=request_id,
                        task=task,
                        item=item,
                        status="skipped_http_402_latch",
                    )
                    async with rows_lock:
                        rows.append(row)
                    return
                reserved_cost, reserved_prompt_tokens = conservative_request_cost(
                    MODEL_BY_ID[cell.model_id], task, int(cell.max_output_tokens)
                )
                reserved = await self.budget.reserve(
                    campaign_id=self.campaign_id,
                    request_id=request_id,
                    epoch_id=phase_id,
                    model_id=cell.model_id,
                    shape=cell.shape,
                    reserved_cost_usd=reserved_cost,
                    reserved_prompt_tokens=reserved_prompt_tokens,
                    max_output_tokens=int(cell.max_output_tokens),
                )
                if not reserved:
                    status = (
                        "unknown_prior_reservation"
                        if request_id in self.budget.reservations
                        else "skipped_cost_cap"
                    )
                    row = await self._append_unsent(
                        cell=cell,
                        phase=phase,
                        phase_id=phase_id,
                        request_id=request_id,
                        task=task,
                        item=item,
                        status=status,
                        provider_send_attempted=request_id in self.budget.reservations,
                    )
                    async with rows_lock:
                        rows.append(row)
                    return
                provider_started = time.perf_counter()
                provider_started_ns = time.perf_counter_ns()
                schedule_lag = max(0.0, provider_started - scheduled_at)
                started_at = utc_now()
                async with active_lock:
                    active += 1
                    max_active = max(max_active, active)
                    observed = active
                base = self._base_request_row(
                    cell=cell,
                    phase=phase,
                    phase_id=phase_id,
                    request_id=request_id,
                    task=task,
                    item=item,
                    started_at=started_at,
                    ended_at=started_at,
                    reserved_cost_usd=reserved_cost,
                    reserved_prompt_tokens=reserved_prompt_tokens,
                    schedule_lag_seconds=schedule_lag,
                    observed_concurrency=observed,
                )
                base["monotonic_timestamps_ns"] = {
                    "clock": "time.perf_counter_ns",
                    "request_started_ns": provider_started_ns,
                    "request_ended_ns": None,
                }
                try:
                    timeout = self._remaining_request_timeout()
                    if timeout <= 0:
                        raise asyncio.TimeoutError("hard campaign deadline reached")
                    result = await asyncio.wait_for(
                        executor(cell.model_id, task, int(cell.max_output_tokens)),
                        timeout=timeout,
                    )
                    base["monotonic_timestamps_ns"]["request_ended_ns"] = (
                        time.perf_counter_ns()
                    )
                    base["ended_at"] = utc_now()
                    row = self._success_row(
                        base=base,
                        task=task,
                        result=result,
                        spec=MODEL_BY_ID[cell.model_id],
                    )
                except asyncio.CancelledError as error:
                    base["monotonic_timestamps_ns"]["request_ended_ns"] = (
                        time.perf_counter_ns()
                    )
                    base["ended_at"] = utc_now()
                    row = self._failure_row(
                        base=base,
                        task=task,
                        error=error,
                        elapsed_seconds=time.perf_counter() - provider_started,
                        status="unknown_cancelled",
                        provider_send_attempted=True,
                    )
                    await self._append_request(row)
                    raise
                except Exception as error:
                    base["monotonic_timestamps_ns"]["request_ended_ns"] = (
                        time.perf_counter_ns()
                    )
                    if getattr(error, "status_code", None) == 402:
                        self.account_blocked_402 = True
                    base["ended_at"] = utc_now()
                    row = self._failure_row(
                        base=base,
                        task=task,
                        error=error,
                        elapsed_seconds=time.perf_counter() - provider_started,
                        status="error",
                        provider_send_attempted=True,
                    )
                finally:
                    async with active_lock:
                        active -= 1
                await self._append_request(row)
                async with rows_lock:
                    rows.append(row)

        if serial:
            for item, request_id, task in zip(schedule, request_ids, tasks):
                await one(item, request_id, task)
        else:
            await asyncio.gather(
                *(
                    one(item, request_id, task)
                    for item, request_id, task in zip(schedule, request_ids, tasks)
                )
            )
        elapsed = time.perf_counter() - started_perf
        rows.sort(
            key=lambda row: float(
                row.get("load", {}).get("scheduled_offset_seconds") or 0.0
            )
        )
        summary = self._phase_summary(
            cell=cell,
            phase=phase,
            phase_id=phase_id,
            rows=rows,
            schedule=schedule,
            elapsed_seconds=max(elapsed, nominal_seconds, 1e-9),
            max_observed_concurrency=max_active,
        )
        await self.phases_journal.append(summary)
        self.phases[phase_id] = summary
        if phase == "two_minute_soak":
            await self._ensure_analysis_blocks(cell=cell, rows=rows)
        return summary

    async def _write_pairs(self, cell: SoakCellPlan) -> list[dict[str, Any]]:
        cell_rows = [
            row for row in self.requests.values() if row.get("cell_id") == cell.cell_id
        ]
        output: list[dict[str, Any]] = []
        for pair_index in range(cell.low_load_requests):
            block_index = self._quality_pair_block_index(cell, pair_index)
            pair_id = self._quality_pair_id(cell, pair_index)
            existing = self.pairs.get(pair_id)
            if existing is not None:
                output.append(existing)
                continue
            rows = [row for row in cell_rows if row.get("quality_pair_id") == pair_id]
            low = [row for row in rows if row.get("quality_pair_role") == "low_load"]
            near = [row for row in rows if row.get("quality_pair_role") == "near_load"]
            complete = len(low) == 1 and len(near) == 1
            low_row = low[0] if len(low) == 1 else None
            near_row = near[0] if len(near) == 1 else None
            payload_match = bool(
                complete
                and low_row.get("request_payload_sha256")
                == near_row.get("request_payload_sha256")
            )
            low_pass = bool(
                low_row is not None
                and low_row.get("status") == "success"
                and float(low_row.get("quality_score") or 0.0) >= 0.999999
            )
            near_pass = bool(
                near_row is not None
                and near_row.get("status") == "success"
                and float(near_row.get("quality_score") or 0.0) >= 0.999999
            )
            acceptance_reasons: list[str] = []
            if not complete:
                acceptance_reasons.append("paired_quality_receipts_incomplete")
            if complete and not payload_match:
                acceptance_reasons.append("paired_quality_payload_hash_mismatch")
            if complete and not low_pass:
                acceptance_reasons.append("paired_low_load_quality_failure")
            if complete and not near_pass:
                acceptance_reasons.append("paired_near_load_quality_failure")
            row = {
                "schema_version": PAIR_SCHEMA,
                "campaign_id": self.campaign_id,
                "plan_sha256": self.plan_sha256,
                "quality_pair_id": pair_id,
                "cell_id": cell.cell_id,
                "provider": "digitalocean-serverless-inference",
                "model_id": cell.model_id,
                "shape": cell.shape,
                "analysis_block_index": block_index,
                "quality_pair_index": pair_index,
                "status": "complete" if complete and payload_match else "incomplete",
                "exact_request_payload_hash_match": payload_match,
                "low_load_request_id": low_row.get("request_id") if low_row else None,
                "near_load_request_id": near_row.get("request_id")
                if near_row
                else None,
                "low_load_success": low_row.get("status") == "success"
                if low_row
                else None,
                "near_load_success": near_row.get("status") == "success"
                if near_row
                else None,
                "low_load_quality_score": low_row.get("quality_score")
                if low_row
                else None,
                "near_load_quality_score": near_row.get("quality_score")
                if near_row
                else None,
                "predeclared_quality_acceptance_pass": not acceptance_reasons,
                "quality_acceptance_reasons": acceptance_reasons,
                "paired_quality_delta_near_minus_low": (
                    float(near_row.get("quality_score") or 0.0)
                    - float(low_row.get("quality_score") or 0.0)
                    if complete
                    else None
                ),
                "paired_latency_ratio_near_over_low": (
                    float(near_row["timing"]["request_seconds"])
                    / float(low_row["timing"]["request_seconds"])
                    if complete
                    and float(low_row.get("timing", {}).get("request_seconds") or 0.0)
                    > 0
                    else None
                ),
                "claim_scope": "one deterministic low-load/near-load request pair",
            }
            await self.pairs_journal.append(row)
            self.pairs[pair_id] = row
            output.append(row)
        return output

    async def _run_cell(
        self, executor: SoakExecutor, cell: SoakCellPlan
    ) -> dict[str, Any]:
        existing = self.cells.get(cell.cell_id)
        if existing is not None:
            return existing
        baseline = await self._run_phase(executor, cell=cell, phase="paired_low_load")
        if baseline.get("status") != "complete":
            return {
                "schema_version": CELL_SCHEMA,
                "campaign_id": self.campaign_id,
                "cell_id": cell.cell_id,
                "model_id": cell.model_id,
                "shape": cell.shape,
                "status": "incomplete_low_load_phase",
                "execution_complete": False,
                "scientifically_complete": False,
            }
        baseline_error_rate = 1.0 - float(baseline.get("success_rate") or 0.0)
        if baseline_error_rate > 0:
            row = {
                "schema_version": CELL_SCHEMA,
                "campaign_id": self.campaign_id,
                "plan_sha256": self.plan_sha256,
                "cell_id": cell.cell_id,
                "model_id": cell.model_id,
                "shape": cell.shape,
                "status": "baseline_transport_gate_failed",
                "execution_complete": True,
                "scientifically_complete": False,
                "baseline_success_rate": baseline.get("success_rate"),
                "provider_send_attempted": True,
                "claim_scope": "no soak launched after low-load transport failure",
            }
            await self.cells_journal.append(row)
            self.cells[cell.cell_id] = row
            return row
        soak = await self._run_phase(executor, cell=cell, phase="two_minute_soak")
        if soak.get("status") != "complete":
            return {
                "schema_version": CELL_SCHEMA,
                "campaign_id": self.campaign_id,
                "cell_id": cell.cell_id,
                "model_id": cell.model_id,
                "shape": cell.shape,
                "status": "incomplete_two_minute_soak",
                "execution_complete": False,
                "scientifically_complete": False,
            }
        recovery = await self._run_phase(
            executor, cell=cell, phase="post_soak_recovery"
        )
        if recovery.get("status") != "complete":
            return {
                "schema_version": CELL_SCHEMA,
                "campaign_id": self.campaign_id,
                "cell_id": cell.cell_id,
                "model_id": cell.model_id,
                "shape": cell.shape,
                "status": "incomplete_post_soak_recovery",
                "execution_complete": False,
                "scientifically_complete": False,
            }
        pairs = await self._write_pairs(cell)
        block_rows = sorted(
            [row for row in self.blocks.values() if row.get("cell_id") == cell.cell_id],
            key=lambda row: int(row["analysis_block_index"]),
        )
        complete_blocks = len(block_rows) == self.config.analysis_block_count
        pair_complete = len(pairs) == cell.low_load_requests and all(
            pair.get("status") == "complete" for pair in pairs
        )
        acceptance = complete_blocks and all(
            block.get("predeclared_acceptance_pass") is True for block in block_rows
        )
        quality_deltas = [
            float(pair["paired_quality_delta_near_minus_low"])
            for pair in pairs
            if pair.get("paired_quality_delta_near_minus_low") is not None
        ]
        successful_rpms = [
            float(block["successful_rpm_per_predeclared_window"])
            for block in block_rows
        ]
        input_usage_complete = complete_blocks and all(
            block.get("input_usage_complete_for_all_successes") is True
            for block in block_rows
        )
        output_usage_complete = complete_blocks and all(
            block.get("output_usage_complete_for_all_successes") is True
            for block in block_rows
        )
        input_tpms = [
            float(block["effective_input_tpm_per_predeclared_window"])
            for block in block_rows
            if block.get("effective_input_tpm_per_predeclared_window") is not None
        ]
        output_tpms = [
            float(block["effective_output_tpm_per_predeclared_window"])
            for block in block_rows
            if block.get("effective_output_tpm_per_predeclared_window") is not None
        ]
        recovery_reasons: list[str] = []
        if recovery.get("status") != "complete":
            recovery_reasons.append("recovery_phase_incomplete")
        if float(recovery.get("success_rate") or 0.0) < 0.99:
            recovery_reasons.append("recovery_success_rate_below_0.99")
        recovery_quality = float(recovery.get("quality_pass_rate") or 0.0)
        baseline_quality = float(baseline.get("quality_pass_rate") or 0.0)
        if recovery_quality < 0.999999:
            recovery_reasons.append(
                "recovery_deterministic_quality_pass_rate_below_1.0"
            )
        if baseline_quality - recovery_quality > 0.05 + 1e-12:
            recovery_reasons.append("recovery_quality_drop_from_low_load_above_0.05")
        recovery_total = int(recovery.get("completed_request_rows") or 0)
        if (
            recovery_total
            and (
                int(recovery.get("http_5xx") or 0) + int(recovery.get("timeouts") or 0)
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
        recovery_ttft = recovery.get("ttft_p95_seconds")
        baseline_ttft = baseline.get("ttft_p95_seconds")
        if (
            recovery_ttft is not None
            and baseline_ttft is not None
            and float(recovery_ttft) > 2 * float(baseline_ttft)
        ):
            recovery_reasons.append("recovery_ttft_p95_above_2x_low_load")
        recovery_latency = recovery.get("latency_p95_seconds")
        baseline_latency = baseline.get("latency_p95_seconds")
        if (
            recovery_latency is not None
            and baseline_latency is not None
            and float(recovery_latency) > 2 * float(baseline_latency)
        ):
            recovery_reasons.append("recovery_latency_p95_above_2x_low_load")
        row = {
            "schema_version": CELL_SCHEMA,
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "cell_id": cell.cell_id,
            "provider": "digitalocean-serverless-inference",
            "model_id": cell.model_id,
            "shape": cell.shape,
            "status": "complete" if complete_blocks and pair_complete else "incomplete",
            "execution_complete": True,
            "scientifically_complete": complete_blocks and pair_complete,
            "candidate_rate_rps": cell.candidate_rate_rps,
            "source_aimd_evidence": cell.candidate_evidence,
            "two_minute_observed_acceptance_pass": acceptance,
            "analysis_block_count": len(block_rows),
            "quality_pair_count": len(pairs),
            "paired_quality_delta_mean": (
                statistics.fmean(quality_deltas) if quality_deltas else None
            ),
            "paired_quality_delta_mean_ci95_student_t": _t_mean_ci95(quality_deltas),
            "successful_rpm_block_mean": (
                statistics.fmean(successful_rpms) if successful_rpms else None
            ),
            "successful_rpm_block_mean_ci95_student_t": _t_mean_ci95(successful_rpms),
            "input_usage_complete_for_all_blocks": input_usage_complete,
            "output_usage_complete_for_all_blocks": output_usage_complete,
            "successful_rows_with_complete_input_usage": sum(
                int(block.get("successful_rows_with_complete_input_usage") or 0)
                for block in block_rows
            ),
            "successful_rows_with_complete_output_usage": sum(
                int(block.get("successful_rows_with_complete_output_usage") or 0)
                for block in block_rows
            ),
            "effective_input_tpm_block_mean": (
                statistics.fmean(input_tpms)
                if input_usage_complete and input_tpms
                else (0.0 if input_usage_complete else None)
            ),
            "effective_input_tpm_block_mean_ci95_student_t": (
                _t_mean_ci95(input_tpms) if input_usage_complete else None
            ),
            "effective_output_tpm_block_mean": (
                statistics.fmean(output_tpms)
                if output_usage_complete and output_tpms
                else (0.0 if output_usage_complete else None)
            ),
            "effective_output_tpm_block_mean_ci95_student_t": (
                _t_mean_ci95(output_tpms) if output_usage_complete else None
            ),
            "block_ci_note": (
                "exploratory Student-t interval over four contiguous predeclared blocks; "
                "serial correlation is not modeled"
            ),
            "post_soak_recovery_success_rate": recovery.get("success_rate"),
            "post_soak_recovery_quality_pass_rate": recovery.get("quality_pass_rate"),
            "post_soak_recovery_quality_delta_from_low_load": (
                recovery_quality - baseline_quality
            ),
            "post_soak_recovery_ttft_p95_seconds": recovery.get("ttft_p95_seconds"),
            "post_soak_recovery_target_rps": cell.recovery_rate_rps,
            "post_soak_recovery_realized_schedule_rps": recovery.get(
                "offered_rps_realized_schedule"
            ),
            "post_soak_recovery_predeclared_pass": not recovery_reasons,
            "post_soak_recovery_acceptance_reasons": recovery_reasons,
            "workload_contract": cell.workload_contract,
            "claim_scope": (
                "this exact endpoint/workload/rate and observed two-minute interval only; "
                "no duration, diurnal, or rate extrapolation"
            ),
            "capacity_generalization": "none",
        }
        await self.cells_journal.append(row)
        self.cells[cell.cell_id] = row
        return row

    async def _run_locked(self, executor: SoakExecutor) -> dict[str, Any]:
        if (
            self.config.stop_launch_at is None
            or self.config.hard_campaign_deadline is None
        ):
            raise SoakPreflightError(
                "live execution requires both a provider-send cutoff and hard campaign deadline"
            )
        if not self.preflight["passes"]:
            raise SoakPreflightError(
                "soak preflight failed: all 12x4 cells need valid AIMD candidates and "
                "current exposure plus the largest in-flight reservation batch must fit "
                "the cumulative cap"
            )
        await self._reconcile_outlier_audit()
        started_at = utc_now()
        order = list(self.cell_plans)
        random.Random(self.config.seed).shuffle(order)
        results: list[dict[str, Any]] = []
        for cell in order:
            if (
                self._deadline_reached()
                or self._hard_deadline_reached()
                or self.account_blocked_402
            ):
                results.append(
                    {
                        "cell_id": cell.cell_id,
                        "model_id": cell.model_id,
                        "shape": cell.shape,
                        "status": (
                            "skipped_http_402_latch"
                            if self.account_blocked_402
                            else "skipped_campaign_deadline"
                        ),
                        "execution_complete": False,
                        "scientifically_complete": False,
                    }
                )
                continue
            results.append(await self._run_cell(executor, cell))
        execution_complete = len(results) == len(self.cell_plans) and all(
            row.get("execution_complete") is True for row in results
        )
        scientifically_complete = len(results) == len(self.cell_plans) and all(
            row.get("scientifically_complete") is True for row in results
        )
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "status": (
                "complete"
                if scientifically_complete
                else (
                    "execution_complete_science_incomplete"
                    if execution_complete
                    else "incomplete"
                )
            ),
            "execution_complete": execution_complete,
            "scientifically_complete": scientifically_complete,
            "started_at": started_at,
            "ended_at": utc_now(),
            "provider": "digitalocean-serverless-inference",
            "target_cells": len(self.cell_plans),
            "terminal_cells": len(self.cells),
            "request_rows": len(self.requests),
            "analysis_block_rows": len(self.blocks),
            "quality_pair_rows": len(self.pairs),
            "outlier_audit_rows": len(self.outlier_audit),
            "conservative_exposure_usd": self.budget.exposure_usd,
            "max_cost_usd": self.config.max_cost_usd,
            "prior_cost_usd": self.config.prior_cost_usd,
            "source_aimd_cumulative_exposure_usd": self.source[
                "source_cumulative_exposure_usd"
            ],
            "execution_window": self.current_execution_window,
            "http_402_latched": self.account_blocked_402,
            "preflight": self.preflight,
            "cells": results,
            "claim_scope": (
                "individual observed two-minute cells only; no general capacity claim"
            ),
            "capacity_generalization": "none",
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary

    async def _run_with_executor(self, executor: SoakExecutor) -> dict[str, Any]:
        # The lease is acquired before reloading any spend-bearing state and is
        # held through the final summary write. A second process fails rather
        # than waiting with a stale in-memory ledger.
        with OutputDirectoryLease(self.execution_lease_path):
            self._reload_runtime_state()
            self.current_execution_window = await self._record_execution_window()
            return await self._run_locked(executor)

    async def run(self, executor: SoakExecutor | None = None) -> dict[str, Any]:
        if executor is not None:
            return await self._run_with_executor(executor)
        # Credentials are deliberately loaded only after the complete offline
        # candidate and budget preflight passes.
        if not self.preflight["passes"]:
            raise SoakPreflightError(
                "offline soak preflight did not pass; no credentials were loaded"
            )
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
