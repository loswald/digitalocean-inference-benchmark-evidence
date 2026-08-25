"""Matched-control closure campaign for previously inconclusive DO probes.

This runner is intentionally small.  Every prior inconclusive capability probe is
bracketed by two known-good controls on the same endpoint.  A repeated provider
failure is a conclusive benchmark observation, but never becomes a capability-
support claim.  Prompts, outputs, bodies, raw headers, and credentials are never
persisted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

import httpx

from do_benchmark.core import (
    MODEL_BY_ID,
    DIGITALOCEAN_HOSTED_MODEL_IDS,
    BenchmarkTask,
    JsonlJournal,
    StreamResult,
    require_digitalocean_hosted_models,
    canonical_json,
    parse_token_usage,
    score_result,
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
    OutputDirectoryLease,
    build_capability_cells,
    workload_for_cell,
)


TARGET_SCHEMA = "do_closure_targets_v1"
PLAN_SCHEMA = "do_matched_closure_plan_v1"
REQUEST_SCHEMA = "do_matched_closure_request_v1"
OUTCOME_SCHEMA = "do_matched_closure_outcome_v1"
SUMMARY_SCHEMA = "do_matched_closure_summary_v1"
RequestExecutor = Callable[[str, BenchmarkTask, int], Awaitable[StreamResult]]


@dataclass(frozen=True)
class ClosureConfig:
    output_dir: Path
    targets_path: Path
    model_ids: tuple[str, ...]
    prior_cost_usd: float
    max_cost_usd: float = 400.0
    max_model_parallelism: int = 4
    request_timeout_seconds: float = 180.0
    max_attempts: int = 2
    retry_backoff_seconds: float = 2.0
    stop_launch_at: datetime | None = None
    seed: int = 20260824

    def validate(self) -> None:
        if not self.model_ids or len(self.model_ids) != len(set(self.model_ids)):
            raise ValueError("model_ids must be non-empty and unique")
        unknown = sorted(set(self.model_ids) - MODEL_BY_ID.keys())
        if unknown:
            raise ValueError(f"unknown model IDs: {unknown}")
        require_digitalocean_hosted_models(self.model_ids)
        if not 0 <= self.prior_cost_usd <= self.max_cost_usd:
            raise ValueError("invalid cumulative cost envelope")
        if self.max_model_parallelism < 1 or self.max_attempts < 1:
            raise ValueError("parallelism and attempts must be positive")
        if self.request_timeout_seconds <= 0 or self.retry_backoff_seconds < 0:
            raise ValueError("invalid timeout or retry backoff")
        if self.stop_launch_at is not None and self.stop_launch_at.tzinfo is None:
            raise ValueError("stop_launch_at must be timezone-aware")


@dataclass(frozen=True)
class ClosureCell:
    cell_id: str
    endpoint_id: str
    probe_id: str
    workload: str
    task: BenchmarkTask
    max_output_tokens: int
    rendered_payload_sha256: str
    prior_classification: str
    prior_http_status: str
    kind: str = "matched_capability"

    def plan_row(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA,
            "cell_id": self.cell_id,
            "request_id": self.cell_id,
            "model_id": self.endpoint_id,
            "endpoint_id": self.endpoint_id,
            "probe_id": self.probe_id,
            "workload_id": self.workload,
            "shape": "matched_control_closure",
            "phase": self.kind,
            "requested_max_output_tokens": self.max_output_tokens,
            "rendered_payload_sha256": self.rendered_payload_sha256,
            "request_identity_sha256": self.cell_id.removeprefix("do-close-cell-"),
            "coverage_tags": [self.workload],
            "prior_classification": self.prior_classification,
            "prior_http_status": self.prior_http_status,
            "task": {
                "task_id": self.task.task_id,
                "family": self.workload,
                "context_bucket": self.task.context_bucket,
                "output_bucket": self.task.output_bucket,
                "requires_vision": self.task.requires_vision,
            },
        }


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _read_jsonl(path: Path, identity_key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"torn JSONL {path}:{line_number}") from error
            key = row.get(identity_key)
            if not isinstance(key, str) or not key or key in rows:
                raise RuntimeError(f"invalid or duplicate {identity_key} in {path}")
            rows[key] = row
    return rows


def _control_task(cell: ClosureCell) -> BenchmarkTask:
    marker = f"CONTROL-{cell.cell_id[-8:].upper()}"
    return BenchmarkTask(
        task_id=f"control-{cell.cell_id[-12:]}",
        family="matched_control",
        context_bucket="short",
        output_bucket="short",
        messages=[{"role": "user", "content": f"Return only {marker}"}],
        expected={"kind": "exact_text", "value": marker},
        metadata={"planned_input_tokens": 16},
    )


def _output_task(endpoint_id: str, max_tokens: int) -> BenchmarkTask:
    # The word count is deliberately below the token ceiling. The experiment
    # distinguishes transport acceptance, realized length, EOS, and truncation.
    words = {256: 160, 1024: 700, 4096: 3000}[max_tokens]
    return BenchmarkTask(
        task_id=f"realized-output-{max_tokens}",
        family="output_length",
        context_bucket="short",
        output_bucket=f"requested_{max_tokens}",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Write exactly {words} whitespace-separated words. Use azure "
                    f"for the first {words - 1} words and COBALT as the final word."
                ),
            }
        ],
        expected={"kind": "controlled_words", "count": words, "marker": "COBALT"},
        metadata={
            "planned_input_tokens": 42,
            "realized_output_anchor": max_tokens,
            "endpoint_id": endpoint_id,
        },
    )


def build_cells(targets_path: Path, model_ids: Sequence[str]) -> list[ClosureCell]:
    payload = json.loads(targets_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != TARGET_SCHEMA:
        raise ValueError("closure target contract is invalid")
    target_rows = payload.get("capability_targets")
    if not isinstance(target_rows, list):
        raise ValueError("closure targets are missing capability_targets")
    all_capability = {
        (cell.model_id, cell.probe_id): cell
        for cell in build_capability_cells(tuple(model_ids), seed=20260823)
    }
    cells: list[ClosureCell] = []
    for target in target_rows:
        endpoint_id = str(target["endpoint_id"])
        if endpoint_id not in model_ids:
            continue
        probe_id = str(target["probe_id"])
        source: CapabilityCell | None = all_capability.get((endpoint_id, probe_id))
        if source is None:
            raise ValueError(f"closure target not in frozen capability plan: {target}")
        identity = {
            "schema": PLAN_SCHEMA,
            "endpoint_id": endpoint_id,
            "probe_id": probe_id,
            "payload": source.rendered_payload_sha256,
            "design": (
                "transport_usage_control_before_probe_control_after_"
                "max_two_attempts_v2"
            ),
        }
        cells.append(
            ClosureCell(
                cell_id=f"do-close-cell-{_sha256(identity)}",
                endpoint_id=endpoint_id,
                probe_id=probe_id,
                workload=workload_for_cell(source),
                task=source.task,
                max_output_tokens=source.max_output_tokens,
                rendered_payload_sha256=source.rendered_payload_sha256,
                prior_classification=str(target.get("prior_classification") or ""),
                prior_http_status=str(target.get("prior_http_status") or ""),
            )
        )
    for endpoint_id in model_ids:
        for max_tokens in (256, 1024, 4096):
            task = _output_task(endpoint_id, max_tokens)
            identity = {
                "schema": PLAN_SCHEMA,
                "endpoint_id": endpoint_id,
                "probe_id": task.task_id,
                "max_tokens": max_tokens,
                "task": task.expected,
                "design": "realized_output_anchor_v2",
            }
            cells.append(
                ClosureCell(
                    cell_id=f"do-close-cell-{_sha256(identity)}",
                    endpoint_id=endpoint_id,
                    probe_id=task.task_id,
                    workload="output_length",
                    task=task,
                    max_output_tokens=max_tokens,
                    rendered_payload_sha256=_sha256(identity),
                    prior_classification="new_realized_output_anchor",
                    prior_http_status="",
                    kind="realized_output",
                )
            )
    return sorted(cells, key=lambda item: (item.endpoint_id, item.kind, item.probe_id))


def _physical_id(cell: ClosureCell, attempt: int, role: str) -> str:
    return f"do-close-request-{_sha256([cell.cell_id, attempt, role])[:28]}"


def _response_metrics(result: StreamResult, task: BenchmarkTask) -> dict[str, Any]:
    usage = parse_token_usage(result.usage)
    score = score_result(task, result)
    prompt_complete = usage.get("prompt_tokens", 0) > 0
    output_complete = usage.get("completion_tokens", 0) > 0
    return {
        "status": "success",
        "http_status": result.status_code,
        "transport_success": True,
        "scientific_success": prompt_complete and output_complete,
        "functional_valid": float(score["quality_score"]) >= 0.999999,
        "quality_score": float(score["quality_score"]),
        "score_kind": str(score["score_kind"]),
        "finish_reason": result.finish_reason,
        "usage": usage,
        "usage_complete_for_settlement": prompt_complete and output_complete,
        "timing": {
            "request_seconds": result.request_seconds,
            "headers_seconds": result.headers_seconds,
            "ttft_seconds": result.ttft_seconds,
            "generation_seconds": result.generation_seconds,
            "stream_seconds": result.stream_seconds,
        },
        "stream_observation": {
            "event_count": result.event_count,
            "first_event_kind": result.first_event_kind,
        },
        "header_signals": sanitized_header_signals(result.response_headers),
        "response_sha256": _sha256(
            [result.text, result.reasoning_text, result.tool_calls]
        ),
        "response_text_bytes": len(result.text.encode("utf-8")),
        "reasoning_text_bytes": len(result.reasoning_text.encode("utf-8")),
        "tool_call_count": len(result.tool_calls),
    }


def _failure_metrics(error: BaseException, elapsed: float) -> dict[str, Any]:
    status = getattr(error, "status_code", None)
    if isinstance(error, asyncio.TimeoutError):
        classification = "timeout"
    elif status == 429:
        classification = "rate_limited"
    elif status == 402:
        classification = "account_blocked_402"
    elif status in {401, 403}:
        classification = "access_denied"
    elif isinstance(status, int) and 400 <= status < 500:
        classification = "client_rejection"
    elif isinstance(status, int) and status >= 500:
        classification = "provider_error"
    else:
        classification = "transport_error"
    body = getattr(error, "body", None)
    return {
        "status": classification,
        "http_status": status if isinstance(status, int) else None,
        "transport_success": False,
        "scientific_success": False,
        "functional_valid": False,
        "quality_score": 0.0,
        "score_kind": "transport",
        "error_type": type(error).__name__,
        "provider_reason_sha256": (
            hashlib.sha256(str(body).encode()).hexdigest() if body else None
        ),
        "usage": {},
        "usage_complete_for_settlement": False,
        "timing": {"request_seconds": elapsed, "ttft_seconds": None},
        "retry_after_seconds": getattr(error, "retry_after", None),
    }


class MatchedClosureCampaign:
    def __init__(self, config: ClosureConfig) -> None:
        config.validate()
        self.config = config
        self.cells = build_cells(config.targets_path, config.model_ids)
        self.planned_worst_case_reservation_usd = sum(
            self._cell_worst_case_reservation(cell) for cell in self.cells
        )
        if (
            config.prior_cost_usd + self.planned_worst_case_reservation_usd
            > config.max_cost_usd + 1e-12
        ):
            raise ValueError(
                "full matched-closure plan cannot fit under the cumulative cap: "
                f"prior=${config.prior_cost_usd:.6f}, "
                f"plan=${self.planned_worst_case_reservation_usd:.6f}, "
                f"cap=${config.max_cost_usd:.6f}"
            )
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plan_text = "".join(
            json.dumps(cell.plan_row(), sort_keys=True) + "\n" for cell in self.cells
        )
        self.plan_sha256 = hashlib.sha256(self.plan_text.encode()).hexdigest()
        self.campaign_id = f"do-matched-closure-{self.plan_sha256[:20]}"
        self.attempts_path = self.output_dir / "attempts.jsonl"
        self.records_path = self.output_dir / "records.jsonl"
        self.reservations_path = self.output_dir / "reservations.jsonl"
        self.lease_path = self.output_dir / ".execution.lock"
        self._write_or_validate_plan()
        self.attempts = _read_jsonl(self.attempts_path, "request_id")
        self.outcomes = _read_jsonl(self.records_path, "cell_id")
        self.attempt_journal = JsonlJournal(self.attempts_path)
        self.outcome_journal = JsonlJournal(self.records_path)
        self.budget = BudgetLedger(
            path=self.reservations_path,
            max_cost_usd=config.max_cost_usd,
            prior_cost_usd=config.prior_cost_usd,
            terminal_rows=self.attempts,
        )
        self.account_blocked_402 = any(
            row.get("http_status") == 402 for row in self.attempts.values()
        )
        self.global_slots = asyncio.Semaphore(config.max_model_parallelism)

    def _cell_worst_case_reservation(self, cell: ClosureCell) -> float:
        spec = MODEL_BY_ID[cell.endpoint_id]
        cost = conservative_request_cost(spec, cell.task, cell.max_output_tokens)[0]
        if cell.kind == "matched_capability":
            control_cost = conservative_request_cost(spec, _control_task(cell), 32)[0]
            cost += 2 * control_cost
        return cost * self.config.max_attempts

    def _write_or_validate_plan(self) -> None:
        manifest_path = self.output_dir / "manifest.json"
        plan_path = self.output_dir / "plan.jsonl"
        target_sha = hashlib.sha256(self.config.targets_path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": "do_matched_closure_manifest_v1",
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "targets_sha256": target_sha,
            "models": list(self.config.model_ids),
            "planned_cells": len(self.cells),
            "matched_capability_cells": sum(
                cell.kind == "matched_capability" for cell in self.cells
            ),
            "realized_output_cells": sum(
                cell.kind == "realized_output" for cell in self.cells
            ),
            "prior_cost_usd": self.config.prior_cost_usd,
            "max_cost_usd": self.config.max_cost_usd,
            "max_attempts": self.config.max_attempts,
            "runner_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "planned_worst_case_reservation_usd": (
                self.planned_worst_case_reservation_usd
            ),
            "request_timeout_seconds": self.config.request_timeout_seconds,
            "design": (
                "endpoint-sequential matched control-before/probe/control-after; "
                "endpoint chains parallel; repeated provider failures are benchmark-"
                "conclusive but capability-inconclusive"
            ),
        }
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key in ("campaign_id", "plan_sha256", "targets_sha256"):
                if existing.get(key) != manifest[key]:
                    raise RuntimeError(f"closure resume identity mismatch: {key}")
            if hashlib.sha256(plan_path.read_bytes()).hexdigest() != self.plan_sha256:
                raise RuntimeError("closure plan hash mismatch")
            return
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        plan_path.write_text(self.plan_text, encoding="utf-8", newline="\n")

    def _deadline_reached(self) -> bool:
        return bool(
            self.config.stop_launch_at
            and datetime.now(timezone.utc) >= self.config.stop_launch_at
        )

    async def _physical_request(
        self,
        executor: RequestExecutor,
        cell: ClosureCell,
        attempt_index: int,
        role: str,
        task: BenchmarkTask,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        request_id = _physical_id(cell, attempt_index, role)
        if request_id in self.attempts:
            return self.attempts[request_id]
        if request_id in self.budget.reservations:
            return {
                "request_id": request_id,
                "status": "unknown_prior_reservation",
                "provider_send_attempted": True,
                "http_status": None,
            }
        if self._deadline_reached() or self.account_blocked_402:
            return {
                "request_id": request_id,
                "status": (
                    "skipped_http_402_latch"
                    if self.account_blocked_402
                    else "skipped_deadline"
                ),
                "provider_send_attempted": False,
                "http_status": None,
            }
        spec = MODEL_BY_ID[cell.endpoint_id]
        reserved_cost, reserved_tokens = conservative_request_cost(
            spec, task, max_output_tokens
        )
        if not await self.budget.reserve(
            campaign_id=self.campaign_id,
            request_id=request_id,
            epoch_id=cell.cell_id,
            model_id=cell.endpoint_id,
            shape=cell.workload,
            reserved_cost_usd=reserved_cost,
            reserved_prompt_tokens=reserved_tokens,
            max_output_tokens=max_output_tokens,
        ):
            return {
                "request_id": request_id,
                "status": "skipped_budget_cap",
                "provider_send_attempted": False,
                "http_status": None,
            }
        started_at = utc_now()
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                executor(cell.endpoint_id, task, max_output_tokens),
                timeout=self.config.request_timeout_seconds,
            )
            metrics = _response_metrics(result, task)
            usage = metrics["usage"]
            complete_usage = metrics["usage_complete_for_settlement"]
            actual_cost = (
                usage.get("prompt_tokens", 0) * spec.input_usd_per_million
                + usage.get("completion_tokens", 0) * spec.output_usd_per_million
            ) / 1_000_000
            accounted = actual_cost if complete_usage else reserved_cost
        except BaseException as error:
            metrics = _failure_metrics(error, time.perf_counter() - started)
            accounted = reserved_cost
            if metrics.get("http_status") == 402:
                self.account_blocked_402 = True
        row = {
            "schema_version": REQUEST_SCHEMA,
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "request_id": request_id,
            "cell_id": cell.cell_id,
            "model_id": cell.endpoint_id,
            "endpoint_id": cell.endpoint_id,
            "probe_id": cell.probe_id,
            "workload_id": cell.workload,
            "shape": "matched_control_closure",
            "phase": role,
            "attempt_index": attempt_index,
            "provider_send_attempted": True,
            "started_at": started_at,
            "ended_at": utc_now(),
            "requested_max_output_tokens": max_output_tokens,
            "request_payload_sha256": _sha256(
                [cell.endpoint_id, task.task_id, task.expected, task.parameters, role]
            ),
            "worst_case_reserved_cost_usd": reserved_cost,
            "reserved_prompt_tokens": reserved_tokens,
            "estimated_cost_usd": (
                accounted if metrics.get("usage_complete_for_settlement") else None
            ),
            "accounted_cost_usd": accounted,
            **metrics,
        }
        await self.attempt_journal.append(row)
        self.attempts[request_id] = row
        await self.budget.settle(request_id, row)
        return row

    @staticmethod
    def _control_pass(row: Mapping[str, Any]) -> bool:
        # A matched control establishes that the route was healthy around the
        # probe. It must not test whether the model followed an unrelated exact-
        # echo instruction. The latter confounds route health with model quality
        # and previously discarded healthy HTTP-200 controls. Positive, complete
        # usage proves that the request reached and generated on the route.
        return bool(
            row.get("status") == "success"
            and row.get("transport_success") is True
            and row.get("usage_complete_for_settlement") is True
        )

    @staticmethod
    def _retryable(rows: Sequence[Mapping[str, Any]]) -> bool:
        return any(
            row.get("status")
            in {"rate_limited", "provider_error", "timeout", "transport_error"}
            for row in rows
        )

    def _outcome_row(
        self,
        cell: ClosureCell,
        attempt_index: int,
        before: Mapping[str, Any] | None,
        probe: Mapping[str, Any],
        after: Mapping[str, Any] | None,
        *,
        exhausted: bool,
    ) -> dict[str, Any]:
        if cell.kind == "realized_output":
            control_ok = True
        else:
            control_ok = bool(
                before
                and after
                and self._control_pass(before)
                and self._control_pass(after)
            )
        probe_status = str(probe.get("status") or "unknown")
        if control_ok and probe_status == "success":
            classification = "accepted"
            conclusive = True
            capability_status = "accepted"
        elif control_ok and probe_status == "client_rejection":
            classification = "matched_control_rejection"
            conclusive = True
            capability_status = "rejected_exact_tested_state"
        elif (
            control_ok
            and exhausted
            and probe_status
            in {
                "provider_error",
                "timeout",
                "transport_error",
            }
        ):
            classification = "matched_control_repeated_provider_failure"
            conclusive = True
            capability_status = "inconclusive_provider_failure"
        elif cell.kind == "realized_output" and exhausted:
            classification = f"realized_output_{probe_status}"
            conclusive = probe_status not in {
                "access_denied",
                "account_blocked_402",
                "provider_error",
                "rate_limited",
                "timeout",
                "transport_error",
                "unknown",
            }
            capability_status = classification
        else:
            classification = "matched_control_inconclusive"
            conclusive = False
            capability_status = "inconclusive"
        return {
            **cell.plan_row(),
            "schema_version": OUTCOME_SCHEMA,
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "semantic_id": cell.cell_id,
            "semantic_coverage_attempt": True,
            "semantic_final_request_id": probe.get("request_id"),
            "attempt_index": attempt_index,
            "provider_send_attempted": probe.get("provider_send_attempted") is True,
            "started_at": probe.get("started_at") or utc_now(),
            "ended_at": probe.get("ended_at") or utc_now(),
            "status": str(probe.get("status") or "unknown"),
            "coverage_classification": classification,
            "coverage_conclusive": conclusive,
            "capability_status": capability_status,
            "control_before_pass": self._control_pass(before or {}),
            "control_after_pass": self._control_pass(after or {}),
            "control_bracket_complete": control_ok,
            "http_status": probe.get("http_status"),
            "transport_success": probe.get("transport_success") is True,
            "scientific_success": probe.get("scientific_success") is True,
            "functional_valid": probe.get("functional_valid"),
            "quality_score": probe.get("quality_score"),
            "score_kind": probe.get("score_kind"),
            "finish_reason": probe.get("finish_reason"),
            "usage": probe.get("usage") or {},
            "usage_complete_for_settlement": probe.get("usage_complete_for_settlement"),
            "timing": probe.get("timing") or {},
            "stream_observation": probe.get("stream_observation") or {},
            "header_signals": probe.get("header_signals") or {},
            "response_sha256": probe.get("response_sha256"),
            "response_text_bytes": probe.get("response_text_bytes"),
            "reasoning_text_bytes": probe.get("reasoning_text_bytes"),
            "tool_call_count": probe.get("tool_call_count"),
            "requested_max_output_tokens": cell.max_output_tokens,
            "request_payload_sha256": probe.get("request_payload_sha256"),
            "error_type": probe.get("error_type"),
            "provider_reason_sha256": probe.get("provider_reason_sha256"),
            "estimated_cost_usd": probe.get("estimated_cost_usd"),
            "accounted_cost_usd": probe.get("accounted_cost_usd"),
        }

    async def _run_cell(self, executor: RequestExecutor, cell: ClosureCell) -> None:
        if cell.cell_id in self.outcomes:
            return
        control = _control_task(cell)
        final: dict[str, Any] | None = None
        for attempt_index in range(self.config.max_attempts):
            if cell.kind == "matched_capability":
                before = await self._physical_request(
                    executor, cell, attempt_index, "control_before", control, 32
                )
            else:
                before = None
            probe = await self._physical_request(
                executor,
                cell,
                attempt_index,
                "probe",
                cell.task,
                cell.max_output_tokens,
            )
            if cell.kind == "matched_capability":
                after = await self._physical_request(
                    executor, cell, attempt_index, "control_after", control, 32
                )
            else:
                after = None
            rows = [row for row in (before, probe, after) if row is not None]
            exhausted = attempt_index + 1 >= self.config.max_attempts
            final = self._outcome_row(
                cell, attempt_index, before, probe, after, exhausted=exhausted
            )
            if final["coverage_conclusive"] or not self._retryable(rows):
                break
            await asyncio.sleep(self.config.retry_backoff_seconds * (2**attempt_index))
        assert final is not None
        await self.outcome_journal.append(final)
        self.outcomes[cell.cell_id] = final

    async def _run_endpoint(self, executor: RequestExecutor, endpoint_id: str) -> None:
        async with self.global_slots:
            for cell in [
                item for item in self.cells if item.endpoint_id == endpoint_id
            ]:
                await self._run_cell(executor, cell)

    def _summary(self, started_at: str) -> dict[str, Any]:
        rows = [
            self.outcomes[cell.cell_id]
            for cell in self.cells
            if cell.cell_id in self.outcomes
        ]
        return {
            "schema_version": SUMMARY_SCHEMA,
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "status": (
                "complete" if len(rows) == len(self.cells) else "incomplete_or_censored"
            ),
            "started_at": started_at,
            "ended_at": utc_now(),
            "planned_cells": len(self.cells),
            "terminal_cells": len(rows),
            "conclusive_cells": sum(
                row.get("coverage_conclusive") is True for row in rows
            ),
            "matched_capability_cells": sum(
                cell.kind == "matched_capability" for cell in self.cells
            ),
            "realized_output_cells": sum(
                cell.kind == "realized_output" for cell in self.cells
            ),
            "provider_attempts": sum(
                row.get("provider_send_attempted") is True
                for row in self.attempts.values()
            ),
            "http_status_counts": {
                str(status): sum(
                    row.get("http_status") == status for row in self.attempts.values()
                )
                for status in sorted(
                    {
                        row.get("http_status")
                        for row in self.attempts.values()
                        if isinstance(row.get("http_status"), int)
                    }
                )
            },
            "classification_counts": {
                value: sum(row.get("coverage_classification") == value for row in rows)
                for value in sorted(
                    {str(row.get("coverage_classification")) for row in rows}
                )
            },
            "prior_cost_usd": self.config.prior_cost_usd,
            "max_cost_usd": self.config.max_cost_usd,
            "planned_worst_case_reservation_usd": (
                self.planned_worst_case_reservation_usd
            ),
            "conservative_exposure_usd": self.budget.exposure_usd,
            "http_402_latched": self.account_blocked_402,
        }

    async def _run_locked(self, executor: RequestExecutor) -> dict[str, Any]:
        started_at = utc_now()
        order = list(self.config.model_ids)
        random.Random(self.config.seed).shuffle(order)
        await asyncio.gather(
            *(self._run_endpoint(executor, endpoint) for endpoint in order)
        )
        summary = self._summary(started_at)
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary

    async def run(self, executor: RequestExecutor | None = None) -> dict[str, Any]:
        with OutputDirectoryLease(self.lease_path):
            if executor is not None:
                return await self._run_locked(executor)
            credentials = digitalocean_credentials()
            timeout = httpx.Timeout(
                self.config.request_timeout_seconds,
                connect=30.0,
                read=self.config.request_timeout_seconds,
                write=30.0,
                pool=self.config.request_timeout_seconds,
            )
            limits = httpx.Limits(
                max_connections=self.config.max_model_parallelism,
                max_keepalive_connections=self.config.max_model_parallelism,
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

                return await self._run_locked(live_executor)

    def finalize_without_sends(self) -> dict[str, Any]:
        """Seal the durable rows currently present without issuing a request.

        This is for interrupted campaigns whose spend must remain in the
        cumulative ledger even when access failure prevents safe resumption.
        It deliberately leaves missing cells censored and never fabricates an
        outcome from a physical attempt.
        """

        with OutputDirectoryLease(self.lease_path):
            started_values = [
                str(row.get("started_at"))
                for row in self.attempts.values()
                if row.get("started_at")
            ]
            started_at = min(started_values) if started_values else utc_now()
            summary = self._summary(started_at)
            summary["finalization_mode"] = "offline_no_provider_sends"
            (self.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return summary


def default_model_ids() -> tuple[str, ...]:
    return DIGITALOCEAN_HOSTED_MODEL_IDS
