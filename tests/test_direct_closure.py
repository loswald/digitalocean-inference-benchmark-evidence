from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from do_benchmark.core import StreamResult
from do_benchmark.direct_closure import (
    ClosureConfig,
    MatchedClosureCampaign,
    _failure_metrics,
    build_cells,
)
from do_benchmark.direct_report import load_matched_closure_directory


MODEL = "arcee-trinity-large-thinking"


def _targets(path: Path) -> Path:
    payload: dict[str, object] = {
        "schema_version": "do_closure_targets_v1",
        "source_requests_sha256": "1" * 64,
        "source_coverage_sha256": "2" * 64,
        "capability_targets": [
            {
                "endpoint_id": MODEL,
                "probe_id": "temperature--0.01",
                "prior_classification": "client_error_inconclusive",
                "prior_http_status": "400",
            }
        ],
        "unresolved_endpoint_dimensions": [],
    }
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _campaign(tmp_path: Path) -> MatchedClosureCampaign:
    return MatchedClosureCampaign(
        ClosureConfig(
            output_dir=tmp_path / "run",
            targets_path=_targets(tmp_path / "targets.json"),
            model_ids=(MODEL,),
            prior_cost_usd=0,
            max_cost_usd=10,
        )
    )


def test_plan_contains_target_and_realized_output_anchors(tmp_path: Path) -> None:
    cells = build_cells(_targets(tmp_path / "targets.json"), (MODEL,))
    assert len(cells) == 4
    assert {cell.kind for cell in cells} == {"matched_capability", "realized_output"}
    assert {
        cell.max_output_tokens for cell in cells if cell.kind == "realized_output"
    } == {
        256,
        1024,
        4096,
    }


def test_matched_400_is_conclusive_exact_state_rejection(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    cell = next(item for item in campaign.cells if item.kind == "matched_capability")
    control = {"status": "success", "functional_valid": True}
    probe = {
        "request_id": "probe",
        "status": "client_rejection",
        "http_status": 400,
        "provider_send_attempted": True,
        "transport_success": False,
    }
    row = campaign._outcome_row(cell, 0, control, probe, control, exhausted=False)
    assert row["coverage_conclusive"] is True
    assert row["coverage_classification"] == "matched_control_rejection"
    assert row["capability_status"] == "rejected_exact_tested_state"


def test_repeated_500_is_benchmark_conclusive_not_support_claim(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    cell = next(item for item in campaign.cells if item.kind == "matched_capability")
    control = {"status": "success", "functional_valid": True}
    probe = {
        "request_id": "probe",
        "status": "provider_error",
        "http_status": 500,
        "provider_send_attempted": True,
        "transport_success": False,
    }
    row = campaign._outcome_row(cell, 1, control, probe, control, exhausted=True)
    assert row["coverage_conclusive"] is True
    assert row["coverage_classification"] == (
        "matched_control_repeated_provider_failure"
    )
    assert row["capability_status"] == "inconclusive_provider_failure"


def test_failed_control_keeps_probe_inconclusive(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    cell = next(item for item in campaign.cells if item.kind == "matched_capability")
    bad_control = {"status": "timeout", "functional_valid": False}
    probe = {
        "request_id": "probe",
        "status": "client_rejection",
        "http_status": 400,
        "provider_send_attempted": True,
        "transport_success": False,
    }
    row = campaign._outcome_row(
        cell, 1, bad_control, probe, bad_control, exhausted=True
    )
    assert row["coverage_conclusive"] is False
    assert row["coverage_classification"] == "matched_control_inconclusive"


class _AccessDenied(RuntimeError):
    status_code = 403
    body = "not persisted"


def test_access_denial_is_not_parameter_rejection() -> None:
    row = _failure_metrics(_AccessDenied(), 0.1)
    assert row["status"] == "access_denied"


def _successful_result(task) -> StreamResult:
    expected = task.expected
    if expected["kind"] == "controlled_words":
        count = int(expected["count"])
        text = " ".join(["azure"] * (count - 1) + [str(expected["marker"])])
    else:
        text = str(expected["value"])
    return StreamResult(
        status_code=200,
        response_headers={},
        text=text,
        reasoning_text="",
        tool_calls=[],
        usage={"prompt_tokens": 64, "completion_tokens": 32, "total_tokens": 96},
        finish_reason="stop",
        request_seconds=0.2,
        headers_seconds=0.02,
        ttft_seconds=0.04,
        generation_seconds=0.16,
        stream_seconds=0.18,
        event_count=4,
        first_event_kind="content",
    )


def test_terminal_campaign_loads_with_controls_separate_from_probe_evidence(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)

    async def executor(_model_id, task, _max_output_tokens):
        return _successful_result(task)

    summary = asyncio.run(campaign.run(executor))
    assert summary["status"] == "complete"
    loaded = load_matched_closure_directory(tmp_path / "run")
    assert loaded["summary"]["status"] == "complete"
    assert len(loaded["plans"]) == 4
    assert len(loaded["outcomes"]) == 4
    semantic = [
        row for row in loaded["requests"] if row["semantic_coverage_attempt"] is True
    ]
    controls = [
        row for row in loaded["requests"] if row["semantic_coverage_attempt"] is False
    ]
    assert len(semantic) == 4
    assert len(controls) == 2
    assert all(row["coverage_conclusive"] is True for row in semantic)
    assert all(row["coverage_conclusive"] is None for row in controls)


def test_finalize_without_sends_preserves_incomplete_rows(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    cell = next(item for item in campaign.cells if item.kind == "realized_output")
    outcome = {
        "schema_version": "do_matched_closure_outcome_v1",
        "campaign_id": campaign.campaign_id,
        "plan_sha256": campaign.plan_sha256,
        "cell_id": cell.cell_id,
        "request_id": cell.cell_id,
        "model_id": cell.endpoint_id,
        "endpoint_id": cell.endpoint_id,
        "probe_id": cell.probe_id,
        "workload_id": cell.workload,
        "shape": "matched_control_closure",
        "phase": cell.kind,
        "provider_send_attempted": True,
        "started_at": "2026-08-24T00:00:00+00:00",
        "ended_at": "2026-08-24T00:00:01+00:00",
        "status": "access_denied",
        "coverage_classification": "matched_control_inconclusive",
        "coverage_conclusive": False,
        "capability_status": "inconclusive_access_denied",
        "http_status": 403,
        "usage": {},
        "timing": {"request_seconds": 1.0},
        "estimated_cost_usd": 0.0,
        "accounted_cost_usd": 0.0,
    }
    asyncio.run(campaign.outcome_journal.append(outcome))
    campaign.outcomes[cell.cell_id] = outcome

    summary = campaign.finalize_without_sends()

    assert summary["status"] == "incomplete_or_censored"
    assert summary["terminal_cells"] == 1
    assert summary["provider_attempts"] == 0
    assert summary["finalization_mode"] == "offline_no_provider_sends"
