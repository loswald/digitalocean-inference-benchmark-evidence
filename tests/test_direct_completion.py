from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from do_benchmark.core import JsonlJournal
from do_benchmark.direct_completion import (
    CompletionConfig,
    CompletionPreflightError,
    DirectCompletionCampaign,
    attempt_request_id,
    unresolved_soak_cells,
)


def _config(tmp_path: Path, **overrides: object) -> CompletionConfig:
    sources = {}
    for name in ("soak", "context", "capability", "aimd"):
        path = tmp_path / name
        path.mkdir(exist_ok=True)
        sources[f"{name}_dir"] = path
    values = {
        "output_dir": tmp_path / "out",
        **sources,
        "model_ids": ("deepseek-v4-flash-0731",),
        "prior_cost_usd": 200.0,
    }
    values.update(overrides)
    return CompletionConfig(**values)  # type: ignore[arg-type]


def test_cost_guard_requires_385_launch_stop_and_15_dollar_reserve(
    tmp_path: Path,
) -> None:
    _config(tmp_path).validate()
    with pytest.raises(ValueError, match=r"\$15 drain reserve"):
        _config(tmp_path, launch_stop_cost_usd=390.0).validate()
    with pytest.raises(ValueError, match=r"authorized \$400 cap"):
        _config(tmp_path, max_cost_usd=401.0, launch_stop_cost_usd=385.0).validate()


def test_attempt_ids_are_deterministic_distinct_no_replay_slots() -> None:
    assert attempt_request_id("semantic-a", 0) == attempt_request_id("semantic-a", 0)
    assert attempt_request_id("semantic-a", 0) != attempt_request_id("semantic-a", 1)


def test_soak_cell_is_closed_only_after_science_acceptance_and_recovery() -> None:
    summary = {
        "cells": [
            {
                "model_id": "deepseek-v4-flash-0731",
                "shape": "short_short",
                "scientifically_complete": True,
                "two_minute_observed_acceptance_pass": True,
                "post_soak_recovery_predeclared_pass": True,
            },
            {
                "model_id": "deepseek-v4-flash-0731",
                "shape": "input32k_short",
                "scientifically_complete": True,
                "two_minute_observed_acceptance_pass": False,
                "post_soak_recovery_predeclared_pass": True,
            },
        ]
    }
    assert unresolved_soak_cells(summary) == ("deepseek-v4-flash-0731:input32k_short",)


def test_mixed_soak_preflight_persists_blocked_and_returns_ready_only(
    tmp_path: Path,
) -> None:
    completion = object.__new__(DirectCompletionCampaign)
    completion.campaign_id = "do-completion-test"
    completion.soak_censors = {}
    completion.soak_censors_journal = JsonlJournal(tmp_path / "soak-censors.jsonl")
    audit = SimpleNamespace(
        cell_plans=[
            SimpleNamespace(
                model_id="deepseek-v4-flash-0731",
                shape="short_short",
                status="ready",
                blocked_reason=None,
                candidate_rate_rps=24.0,
            ),
            SimpleNamespace(
                model_id="deepseek-v4-flash-0731",
                shape="mixed",
                status="blocked_candidate_rate_cannot_populate_quality_pairs",
                blocked_reason=(
                    "global candidate-rate schedule does not place the required "
                    "quality-pair arrivals in every analysis block"
                ),
                candidate_rate_rps=0.05,
            ),
        ]
    )

    eligible, censored = asyncio.run(
        completion._record_soak_censors(audit, wave_index=0, multiplier=0.75)
    )
    assert eligible == ("deepseek-v4-flash-0731:short_short",)
    assert censored == ("deepseek-v4-flash-0731:mixed",)
    assert len(completion.soak_censors) == 1

    # Re-entering the same failed preflight is idempotent: it does not create
    # a second censor receipt and cannot mutate the eligible partition.
    eligible_again, censored_again = asyncio.run(
        completion._record_soak_censors(audit, wave_index=0, multiplier=0.75)
    )
    assert (eligible_again, censored_again) == (eligible, censored)
    assert len((tmp_path / "soak-censors.jsonl").read_text().splitlines()) == 1


def test_child_plan_resume_reuses_exact_reconciled_prior(tmp_path: Path) -> None:
    exact_prior = 223.56171980099998
    (tmp_path / "plan.json").write_text(
        '{"prior_cost_usd":223.56171980099998}\n', encoding="utf-8"
    )

    observed = DirectCompletionCampaign._child_plan_prior_cost(tmp_path, 223.561719801)

    assert observed == exact_prior
    with pytest.raises(CompletionPreflightError, match="does not reconcile"):
        DirectCompletionCampaign._child_plan_prior_cost(tmp_path, 224.0)
