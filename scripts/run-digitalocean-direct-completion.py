#!/usr/bin/env python3
"""Plan or run the bounded six-hour DigitalOcean completion campaign."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from do_benchmark.direct_completion import (  # noqa: E402
    CompletionConfig,
    DirectCompletionCampaign,
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
            "Close only unresolved DigitalOcean capability/context/output cells and "
            "failed sustained endpoint-shape cells with bounded retries, descending "
            "rate re-soaks, durable no-replay journals, and a cumulative cost guard."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--soak-dir", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--capability-dir", type=Path, required=True)
    parser.add_argument("--aimd-dir", type=Path, required=True)
    parser.add_argument("--aimd-reconciliation", type=Path)
    parser.add_argument("--prior-lineage-root", type=Path)
    parser.add_argument("--v3-checkpoint-dir", type=Path)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--prior-cost-usd", type=float, required=True)
    parser.add_argument("--max-cost-usd", type=float, default=400.0)
    parser.add_argument("--launch-stop-cost-usd", type=float, default=385.0)
    parser.add_argument("--duration-hours", type=float, default=6.0)
    parser.add_argument(
        "--absolute-hard-deadline",
        type=_utc_datetime,
        help="optional shared campaign deadline; never extends the six-hour relative limit",
    )
    parser.add_argument("--send-reserve-minutes", type=float, default=5.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-concurrency", type=int, default=12)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument(
        "--rate-multiplier",
        action="append",
        dest="rate_ladder",
        type=float,
        help="descending re-soak multiplier; repeat; default 0.75,0.5,0.25,0.125",
    )
    parser.add_argument(
        "--output-token-anchor",
        action="append",
        dest="output_anchors",
        type=int,
        help="realized-output request limit; repeat; default 256,1024,4096,16384",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--soak-concurrency-ceiling", type=int, default=128)
    parser.add_argument(
        "--accept-conditional-prior-exposure-basis", action="store_true"
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write and validate a credential-free immutable plan; send zero requests",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = CompletionConfig(
        output_dir=args.output_dir,
        soak_dir=args.soak_dir,
        context_dir=args.context_dir,
        capability_dir=args.capability_dir,
        aimd_dir=args.aimd_dir,
        model_ids=tuple(args.models or default_model_ids()),
        prior_cost_usd=args.prior_cost_usd,
        max_cost_usd=args.max_cost_usd,
        launch_stop_cost_usd=args.launch_stop_cost_usd,
        duration_hours=args.duration_hours,
        absolute_hard_deadline=args.absolute_hard_deadline,
        send_reserve_minutes=args.send_reserve_minutes,
        request_timeout_seconds=args.request_timeout_seconds,
        max_concurrency=args.max_concurrency,
        max_attempts=args.max_attempts,
        retry_backoff_seconds=args.retry_backoff_seconds,
        rate_ladder=tuple(args.rate_ladder or (0.75, 0.5, 0.25, 0.125)),
        output_token_anchors=tuple(args.output_anchors or (256, 1_024, 4_096, 16_384)),
        soak_concurrency_ceiling=args.soak_concurrency_ceiling,
        seed=args.seed,
        aimd_reconciliation_path=args.aimd_reconciliation,
        prior_lineage_root=args.prior_lineage_root,
        v3_checkpoint_dir=args.v3_checkpoint_dir,
        accept_conditional_prior_exposure_basis=(
            args.accept_conditional_prior_exposure_basis
        ),
    )
    campaign = DirectCompletionCampaign(config)
    if args.plan_only:
        manifest = json.loads(
            (args.output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        print(
            json.dumps(
                {
                    "campaign_id": campaign.campaign_id,
                    "plan_sha256": campaign.plan_sha256,
                    "planned_semantic_probes": len(campaign.probes),
                    "unresolved_soak_cells": list(
                        campaign.initial_unresolved_soak_cells
                    ),
                    "launch_gate_passes": manifest["launch_gate_passes"],
                    "billable_requests_sent": 0,
                    "credentials_loaded": False,
                    "manifest": str(args.output_dir / "manifest.json"),
                    "plan": str(args.output_dir / "plan.json"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    summary = asyncio.run(campaign.run())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
