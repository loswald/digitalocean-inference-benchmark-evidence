#!/usr/bin/env python3
"""Plan or run the matched-control DigitalOcean closure campaign."""

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

from do_benchmark.direct_closure import (  # noqa: E402
    ClosureConfig,
    MatchedClosureCampaign,
    default_model_ids,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bracket prior inconclusive capability probes with endpoint-local "
            "controls and add realized-output anchors."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--prior-cost-usd", type=float, required=True)
    parser.add_argument("--max-cost-usd", type=float, default=400.0)
    parser.add_argument("--max-model-parallelism", type=int, default=4)
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--duration-minutes", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.duration_minutes <= 0:
        raise SystemExit("--duration-minutes must be positive")
    config = ClosureConfig(
        output_dir=args.output_dir,
        targets_path=args.targets,
        model_ids=tuple(args.models or default_model_ids()),
        prior_cost_usd=args.prior_cost_usd,
        max_cost_usd=args.max_cost_usd,
        max_model_parallelism=args.max_model_parallelism,
        request_timeout_seconds=args.request_timeout_seconds,
        max_attempts=args.max_attempts,
        retry_backoff_seconds=args.retry_backoff_seconds,
        stop_launch_at=datetime.now(timezone.utc)
        + timedelta(minutes=args.duration_minutes),
        seed=args.seed,
    )
    campaign = MatchedClosureCampaign(config)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "campaign_id": campaign.campaign_id,
                    "planned_cells": len(campaign.cells),
                    "matched_capability_cells": sum(
                        cell.kind == "matched_capability" for cell in campaign.cells
                    ),
                    "realized_output_cells": sum(
                        cell.kind == "realized_output" for cell in campaign.cells
                    ),
                    "plan_sha256": campaign.plan_sha256,
                    "planned_worst_case_reservation_usd": (
                        campaign.planned_worst_case_reservation_usd
                    ),
                    "prior_cost_usd": config.prior_cost_usd,
                    "max_cost_usd": config.max_cost_usd,
                    "billable_requests_sent": 0,
                    "credentials_loaded": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(json.dumps(asyncio.run(campaign.run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
