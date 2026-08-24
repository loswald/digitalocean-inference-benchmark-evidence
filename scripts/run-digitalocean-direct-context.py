#!/usr/bin/env python3
"""Run direct model-relative DigitalOcean context-boundary probes."""

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

from do_benchmark.direct_context import (  # noqa: E402
    CONTEXT_PERCENTAGES,
    DEFAULT_FALLBACK_ACCOUNT_RPM,
    DEFAULT_FALLBACK_ACCOUNT_TPM,
    DEFAULT_MAX_PAYLOAD_BYTES,
    ContextConfig,
    DirectContextCampaign,
    default_model_ids,
)


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset or Z")
    return parsed.astimezone(timezone.utc)


def _existing_campaign_deadline(output_dir: Path) -> datetime | None:
    """Reuse the hash-bound cutoff when plan-only already created a campaign.

    This keeps a credential-free ``--plan-only`` followed by the live command
    resumable. The original wall-clock cutoff remains authoritative; planning
    never silently grants a fresh execution window.
    """

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_deadline = manifest.get("stop_launch_at")
        if not isinstance(raw_deadline, str) or not raw_deadline:
            raise ValueError("missing stop_launch_at")
        return _utc_datetime(raw_deadline)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise argparse.ArgumentTypeError(
            f"cannot reuse existing campaign deadline: {error}"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe all 12 DigitalOcean context envelopes with a direct runner. "
            "Each model has one sequential adaptive chain; model chains overlap "
            "behind a shared per-account RPM/TPM governor and global ceiling. The "
            "fixed design uses 1/10/25/50/75/90/95/99% "
            "anchors plus uncertainty-aware lower/center/upper prompt-only and "
            "prompt-plus-output boundary estimates."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="exact DigitalOcean model ID; repeat; default is all 12 frozen endpoints",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--per-model-concurrency",
        type=int,
        default=1,
        help="must remain 1; boundary chains are sequential",
    )
    parser.add_argument(
        "--model-parallelism",
        type=int,
        default=12,
        help="maximum concurrently active endpoint chains (default 12)",
    )
    parser.add_argument(
        "--global-concurrency",
        type=int,
        default=12,
        help="account-wide in-flight provider request ceiling (default 12)",
    )
    parser.add_argument(
        "--fallback-account-rpm",
        type=float,
        default=DEFAULT_FALLBACK_ACCOUNT_RPM,
        help="cold-start RPM until DigitalOcean quota headers are observed",
    )
    parser.add_argument(
        "--fallback-account-tpm",
        type=float,
        default=DEFAULT_FALLBACK_ACCOUNT_TPM,
        help="cold-start TPM until DigitalOcean quota headers are observed",
    )
    parser.add_argument("--quota-utilization-fraction", type=float, default=0.80)
    parser.add_argument("--governor-multiplicative-decrease", type=float, default=0.50)
    parser.add_argument(
        "--governor-additive-increase-fraction", type=float, default=0.05
    )
    parser.add_argument(
        "--governor-minimum-congestion-factor", type=float, default=0.05
    )
    parser.add_argument("--governor-successes-per-increase", type=int, default=20)
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-cost-usd", type=float, default=200.0)
    parser.add_argument(
        "--prior-cost-usd",
        type=float,
        default=0.0,
        help="already incurred exposure inside the same cumulative budget cap",
    )
    parser.add_argument(
        "--max-payload-bytes",
        type=int,
        default=DEFAULT_MAX_PAYLOAD_BYTES,
        help="hard cap on each exact serialized JSON request (default 8 MiB)",
    )
    parser.add_argument("--combined-output-tokens", type=int, default=4_096)
    parser.add_argument("--short-output-tokens", type=int, default=32)
    parser.add_argument(
        "--max-bisection-rounds",
        type=int,
        default=8,
        help="adaptive refinements when accepted/rejected anchors bracket a transition",
    )
    parser.add_argument(
        "--percentage",
        action="append",
        type=float,
        dest="percentages",
        help=(
            "model-relative anchor as a fraction strictly between 0 and 1; repeat; "
            "default is 0.01,0.10,0.25,0.50,0.75,0.90,0.95,0.99"
        ),
    )
    parser.add_argument(
        "--skip-prompt-boundary-triplet",
        action="store_true",
        help=(
            "omit the uncertainty-aware lower/center/upper estimates around the "
            "context-window anchor"
        ),
    )
    parser.add_argument(
        "--skip-combined-boundary-triplet",
        action="store_true",
        help="omit the three prompt-plus-requested-output boundary estimates",
    )
    parser.add_argument("--planning-tolerance-fraction", type=float, default=0.02)
    parser.add_argument("--planning-tolerance-tokens", type=int, default=256)
    deadline = parser.add_mutually_exclusive_group()
    deadline.add_argument("--stop-launch-at", type=_utc_datetime)
    deadline.add_argument(
        "--duration-minutes",
        type=float,
        default=120.0,
        help="relative provider-send window; default 120 minutes",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write sanitized plan/cost preflight without loading credentials",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.duration_minutes is not None and args.duration_minutes <= 0:
        parser.error("--duration-minutes must be positive")
    stop_launch_at = args.stop_launch_at
    if stop_launch_at is None:
        try:
            stop_launch_at = _existing_campaign_deadline(args.output_dir)
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
        if stop_launch_at is None:
            stop_launch_at = datetime.now(timezone.utc) + timedelta(
                minutes=args.duration_minutes
            )
    config = ContextConfig(
        output_dir=args.output_dir,
        model_ids=tuple(args.models or default_model_ids()),
        seed=args.seed,
        per_model_concurrency=args.per_model_concurrency,
        model_parallelism=args.model_parallelism,
        global_concurrency=args.global_concurrency,
        fallback_account_rpm=args.fallback_account_rpm,
        fallback_account_tpm=args.fallback_account_tpm,
        quota_utilization_fraction=args.quota_utilization_fraction,
        governor_multiplicative_decrease=(args.governor_multiplicative_decrease),
        governor_additive_increase_fraction=(args.governor_additive_increase_fraction),
        governor_minimum_congestion_factor=(args.governor_minimum_congestion_factor),
        governor_successes_per_increase=args.governor_successes_per_increase,
        request_timeout_seconds=args.request_timeout_seconds,
        max_cost_usd=args.max_cost_usd,
        prior_cost_usd=args.prior_cost_usd,
        stop_launch_at=stop_launch_at,
        max_payload_bytes=args.max_payload_bytes,
        combined_output_tokens=args.combined_output_tokens,
        short_output_tokens=args.short_output_tokens,
        max_bisection_rounds=args.max_bisection_rounds,
        percentages=tuple(args.percentages or CONTEXT_PERCENTAGES),
        include_prompt_boundary_triplet=not args.skip_prompt_boundary_triplet,
        include_combined_boundary_triplet=not args.skip_combined_boundary_triplet,
        planning_tolerance_fraction=args.planning_tolerance_fraction,
        planning_tolerance_tokens=args.planning_tolerance_tokens,
    )
    campaign = DirectContextCampaign(config)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "campaign_id": campaign.campaign_id,
                    "models": len(config.model_ids),
                    "fixed_planned_requests": len(campaign.fixed_probes),
                    "maximum_adaptive_requests": (campaign.maximum_adaptive_requests),
                    "all_requests_all_fail_settlement_ceiling_usd": (
                        campaign.all_requests_settlement_ceiling_usd
                    ),
                    "full_plan_guaranteed_to_fit_budget": (
                        campaign.full_plan_guaranteed_to_fit_budget
                    ),
                    "max_simultaneous_inflight_reservation_usd": (
                        campaign.max_inflight_reservation_usd
                    ),
                    "parallel_timeout_only_projection_seconds": (
                        campaign.parallel_timeout_only_projection_seconds
                    ),
                    "first_calibration_header_timeout_projection_seconds": (
                        campaign.first_calibration_header_projection_seconds
                    ),
                    "serialized_configured_timeout_sum_seconds": (
                        campaign.serialized_configured_timeout_sum_seconds
                    ),
                    "quota_governor_wait_upper_bound_seconds": None,
                    "effective_model_parallelism": (
                        campaign.effective_model_parallelism
                    ),
                    "latency_measurement_scope": (campaign.latency_measurement_scope),
                    "stop_launch_at": stop_launch_at.isoformat(),
                    "prior_cost_usd": config.prior_cost_usd,
                    "max_cost_usd": config.max_cost_usd,
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
