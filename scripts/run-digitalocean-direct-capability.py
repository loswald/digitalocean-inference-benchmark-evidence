#!/usr/bin/env python3
"""Run the bounded direct DigitalOcean capability-envelope benchmark."""

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

from do_benchmark.direct_capability import (  # noqa: E402
    CapabilityConfig,
    DirectCapabilityCampaign,
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
            "Run a bounded direct capability/parameter envelope for all DigitalOcean "
            "endpoints, including a 17-factor strength-two interaction array, "
            "with a standalone direct runner."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="exact model ID; repeat; default is all frozen DigitalOcean endpoints",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--max-workers", type=int, default=48)
    parser.add_argument("--per-model-concurrency", type=int, default=4)
    parser.add_argument("--request-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-cost-usd", type=float, default=200.0)
    parser.add_argument(
        "--prior-cost-usd",
        type=float,
        required=True,
        help=(
            "authoritative total cumulative exposure before this capability run; "
            "required even when zero"
        ),
    )
    deadline = parser.add_mutually_exclusive_group(required=True)
    deadline.add_argument("--stop-launch-at", type=_utc_datetime)
    deadline.add_argument(
        "--duration-minutes",
        type=float,
        help="relative provider-send window beginning when the command starts",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write the sanitized plan and manifest without loading credentials",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.duration_minutes is not None and args.duration_minutes <= 0:
        parser.error("--duration-minutes must be positive")
    stop_launch_at = args.stop_launch_at
    if stop_launch_at is None:
        stop_launch_at = datetime.now(timezone.utc) + timedelta(
            minutes=args.duration_minutes
        )
    config = CapabilityConfig(
        output_dir=args.output_dir,
        model_ids=tuple(args.models or default_model_ids()),
        seed=args.seed,
        max_workers=args.max_workers,
        per_model_concurrency=args.per_model_concurrency,
        request_timeout_seconds=args.request_timeout_seconds,
        max_cost_usd=args.max_cost_usd,
        prior_cost_usd=args.prior_cost_usd,
        stop_launch_at=stop_launch_at,
    )
    campaign = DirectCapabilityCampaign(config)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "campaign_id": campaign.campaign_id,
                    "campaign_identity_sha256": campaign.campaign_identity_sha256,
                    "plan_sha256": campaign.plan_sha256,
                    "planned_requests": len(campaign.cells),
                    "planned_provider_calls": sum(
                        cell.provider_send_expected for cell in campaign.cells
                    ),
                    "planned_worst_case_reservation_usd": (
                        campaign.planned_reservation_usd
                    ),
                    "models": len(config.model_ids),
                    "billable_requests_sent": 0,
                    "manifest": str(args.output_dir / "manifest.json"),
                    "plan": str(args.output_dir / "plan.jsonl"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    summary = asyncio.run(campaign.run())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
