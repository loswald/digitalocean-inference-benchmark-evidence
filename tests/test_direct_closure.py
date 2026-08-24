from __future__ import annotations

import hashlib
import json
from pathlib import Path

from do_benchmark.direct_closure import (
    ClosureConfig,
    MatchedClosureCampaign,
    _failure_metrics,
    build_cells,
)


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
