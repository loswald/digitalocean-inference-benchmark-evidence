#!/usr/bin/env python3
"""Run the simple, endpoint-isolated direct DigitalOcean benchmark."""

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

from do_benchmark.direct_aimd import (  # noqa: E402
    DirectAIMDCampaign,
    DirectConfig,
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
            "Direct endpoint-isolated DigitalOcean benchmark: serial baselines, "
            "compact open-loop AIMD for short, 32K-input, output-long, and mixed "
            "workloads, each with three separated confirmations and overload/recovery."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="exact hosted model ID; repeat; default is the 11-model DO-hosted allowlist",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--epoch-seconds", type=float, default=5.0)
    parser.add_argument("--concurrency-ceiling", type=int, default=128)
    parser.add_argument("--initial-rps", type=float, default=2.0)
    parser.add_argument("--additive-step-rps", type=float, default=1.0)
    parser.add_argument("--maximum-rps", type=float, default=32.0)
    parser.add_argument("--input-initial-rps", type=float, default=0.4)
    parser.add_argument("--input-additive-step-rps", type=float, default=0.4)
    parser.add_argument("--input-maximum-rps", type=float, default=2.4)
    parser.add_argument("--rapid-bracket-epochs", type=int, default=5)
    parser.add_argument("--heavy-rapid-bracket-epochs", type=int, default=3)
    parser.add_argument("--additive-aimd-epochs", type=int, default=1)
    parser.add_argument("--baseline-samples", type=int, default=1)
    parser.add_argument("--output-initial-rps", type=float, default=0.4)
    parser.add_argument("--output-additive-step-rps", type=float, default=0.2)
    parser.add_argument("--output-maximum-rps", type=float, default=1.6)
    parser.add_argument("--mixed-initial-rps", type=float, default=0.4)
    parser.add_argument("--mixed-additive-step-rps", type=float, default=0.4)
    parser.add_argument("--mixed-maximum-rps", type=float, default=3.2)
    parser.add_argument("--input-tokens", type=int, default=32_000)
    parser.add_argument("--long-output-words", type=int, default=1_024)
    parser.add_argument("--short-max-output-tokens", type=int, default=64)
    parser.add_argument("--long-max-output-tokens", type=int, default=2_048)
    parser.add_argument("--mixed-max-output-tokens", type=int, default=1_024)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-cost-usd", type=float, default=200.0)
    parser.add_argument(
        "--prior-cost-usd",
        type=float,
        default=0.0,
        help="already incurred exposure counted inside the same cumulative cap",
    )
    deadline = parser.add_mutually_exclusive_group(required=True)
    deadline.add_argument("--stop-launch-at", type=_utc_datetime)
    deadline.add_argument(
        "--duration-minutes",
        type=float,
        help="relative provider-send window beginning when this command starts",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write/validate the secret-free manifest and exit without loading credentials",
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
    config = DirectConfig(
        output_dir=args.output_dir,
        model_ids=tuple(args.models or default_model_ids()),
        seed=args.seed,
        epoch_seconds=args.epoch_seconds,
        concurrency_ceiling=args.concurrency_ceiling,
        initial_rps=args.initial_rps,
        additive_step_rps=args.additive_step_rps,
        maximum_rps=args.maximum_rps,
        input_initial_rps=args.input_initial_rps,
        input_additive_step_rps=args.input_additive_step_rps,
        input_maximum_rps=args.input_maximum_rps,
        rapid_bracket_epochs=args.rapid_bracket_epochs,
        heavy_rapid_bracket_epochs=args.heavy_rapid_bracket_epochs,
        additive_aimd_epochs=args.additive_aimd_epochs,
        baseline_samples=args.baseline_samples,
        output_initial_rps=args.output_initial_rps,
        output_additive_step_rps=args.output_additive_step_rps,
        output_maximum_rps=args.output_maximum_rps,
        mixed_initial_rps=args.mixed_initial_rps,
        mixed_additive_step_rps=args.mixed_additive_step_rps,
        mixed_maximum_rps=args.mixed_maximum_rps,
        input_tokens=args.input_tokens,
        long_output_words=args.long_output_words,
        short_max_output_tokens=args.short_max_output_tokens,
        long_max_output_tokens=args.long_max_output_tokens,
        mixed_max_output_tokens=args.mixed_max_output_tokens,
        request_timeout_seconds=args.request_timeout_seconds,
        max_cost_usd=args.max_cost_usd,
        prior_cost_usd=args.prior_cost_usd,
        stop_launch_at=stop_launch_at,
    )
    campaign = DirectAIMDCampaign(config)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "campaign_id": campaign.campaign_id,
                    "manifest": str(args.output_dir / "manifest.json"),
                    "model_count": len(config.model_ids),
                    "billable_requests_sent": 0,
                    "preflight_worst_case_cost": campaign.preflight,
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
