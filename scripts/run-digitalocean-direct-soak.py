#!/usr/bin/env python3
"""Plan or run direct two-minute DigitalOcean soak confirmations."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from do_benchmark.direct_soak import (  # noqa: E402
    DirectSoakCampaign,
    SoakConfig,
    default_model_ids,
)


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset or Z")
    return parsed.astimezone(timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read completed direct AIMD artifacts and confirm each exact candidate "
            "rate with a 120-second open-loop soak, four 30-second analysis blocks, "
            "paired low/near-load quality, and explicit post-soak recovery."
        )
    )
    parser.add_argument("--aimd-dir", type=Path, required=True)
    parser.add_argument(
        "--aimd-reconciliation",
        type=Path,
        help="exact verified legacy AIMD reconciliation receipt",
    )
    parser.add_argument(
        "--prior-lineage-root",
        type=Path,
        help="directory containing the three hash-bound direct breadth runs",
    )
    parser.add_argument(
        "--v3-checkpoint-dir",
        type=Path,
        help="exact hash-bound v3 checkpoint used by the reconciliation receipt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="exact hosted model ID; repeat; default is the 11-model DO-hosted allowlist",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--cell",
        action="append",
        dest="cells",
        help=(
            "endpoint:shape selector; repeat to run only named cells; intended for "
            "deterministic descending-rate completion waves"
        ),
    )
    parser.add_argument(
        "--candidate-rate-multiplier",
        type=float,
        default=1.0,
        help="multiply each receipt-backed AIMD candidate by this value in (0,1]",
    )
    parser.add_argument("--completion-attempt-label")
    parser.add_argument("--concurrency-ceiling", type=int, default=128)
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-cost-usd", type=float, default=200.0)
    parser.add_argument(
        "--prior-cost-usd",
        type=float,
        required=True,
        help=(
            "authoritative total cumulative exposure before this soak; must be at "
            "least the exposure reconciled from the source AIMD artifacts"
        ),
    )
    parser.add_argument(
        "--accept-conditional-prior-exposure-basis",
        action="store_true",
        help=(
            "explicitly accept the reconciled breadth prior's hash-bound but "
            "conditional 4,096-output-token reservation basis; this choice is "
            "recorded in the immutable soak plan and is required for live execution"
        ),
    )
    send_window = parser.add_mutually_exclusive_group(required=True)
    send_window.add_argument("--stop-launch-at", type=_utc_datetime)
    send_window.add_argument(
        "--duration-minutes",
        type=float,
        help="relative provider-send window beginning when the command starts",
    )
    hard_window = parser.add_mutually_exclusive_group(required=True)
    hard_window.add_argument("--hard-campaign-deadline", type=_utc_datetime)
    hard_window.add_argument(
        "--drain-minutes",
        type=float,
        help="hard final deadline this many minutes after the provider-send cutoff",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and write the secret-free plan without loading credentials",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    if args.duration_minutes is not None and args.duration_minutes <= 0:
        parser.error("--duration-minutes must be positive")
    if args.drain_minutes is not None and args.drain_minutes <= 0:
        parser.error("--drain-minutes must be positive")
    stop_launch_at = args.stop_launch_at
    if stop_launch_at is None:
        stop_launch_at = now + timedelta(minutes=args.duration_minutes)
    hard_deadline = args.hard_campaign_deadline
    if hard_deadline is None:
        hard_deadline = stop_launch_at + timedelta(minutes=args.drain_minutes)
    config = SoakConfig(
        aimd_dir=args.aimd_dir,
        output_dir=args.output_dir,
        model_ids=tuple(args.models or default_model_ids()),
        aimd_reconciliation_path=args.aimd_reconciliation,
        prior_lineage_root=args.prior_lineage_root,
        v3_checkpoint_dir=args.v3_checkpoint_dir,
        seed=args.seed,
        concurrency_ceiling=args.concurrency_ceiling,
        request_timeout_seconds=args.request_timeout_seconds,
        max_cost_usd=args.max_cost_usd,
        prior_cost_usd=args.prior_cost_usd,
        accept_conditional_prior_exposure_basis=(
            args.accept_conditional_prior_exposure_basis
        ),
        stop_launch_at=stop_launch_at,
        hard_campaign_deadline=hard_deadline,
        selected_cells=tuple(args.cells or ()),
        candidate_rate_multiplier=args.candidate_rate_multiplier,
        completion_attempt_label=args.completion_attempt_label,
    )
    campaign = DirectSoakCampaign(config)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "campaign_id": campaign.campaign_id,
                    "plan_sha256": campaign.plan_sha256,
                    "manifest": str(args.output_dir / "manifest.json"),
                    "plan": str(args.output_dir / "plan.json"),
                    "billable_requests_sent": 0,
                    "credentials_loaded": False,
                    "preflight": campaign.preflight,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if campaign.preflight["passes"] else 2
    summary = asyncio.run(campaign.run())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
