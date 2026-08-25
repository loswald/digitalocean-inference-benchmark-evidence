#!/usr/bin/env python3
"""Build the secret-free data tables and charts for the direct DO report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from do_benchmark.direct_report import analyze_and_write  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize direct breadth/AIMD/two-minute-soak/completion journals "
            "into public statistical tables and charts."
        )
    )
    parser.add_argument("--breadth-dir", type=Path, action="append", default=[])
    parser.add_argument("--aimd-dir", type=Path, action="append", default=[])
    parser.add_argument("--soak-dir", type=Path, action="append", default=[])
    parser.add_argument("--completion-dir", type=Path, action="append", default=[])
    parser.add_argument("--closure-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--cost-only-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "terminal campaign directory retained only in the cumulative cost "
            "chain; its scientific rows are deliberately excluded"
        ),
    )
    parser.add_argument(
        "--endpoint-freeze",
        type=Path,
        default=REPO_ROOT / "config" / "endpoint-freeze.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument(
        "--publication-mode",
        choices=("draft", "final"),
        default="draft",
        help=(
            "draft permits incomplete coverage and labels it; final fails unless "
            "the frozen hosted endpoint-by-dimension matrix is complete and all "
            "evidence reconciles"
        ),
    )
    args = parser.parse_args()
    if not (
        args.breadth_dir
        or args.aimd_dir
        or args.soak_dir
        or args.completion_dir
        or args.closure_dir
    ):
        parser.error("at least one evidence source is required")
    analysis = analyze_and_write(
        breadth_directories=args.breadth_dir,
        aimd_directories=args.aimd_dir,
        soak_directories=args.soak_dir,
        completion_directories=args.completion_dir,
        closure_directories=args.closure_dir,
        cost_only_directories=args.cost_only_dir,
        endpoint_freeze=args.endpoint_freeze,
        output_directory=args.output_dir,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        publication_mode=args.publication_mode,
    )
    print(
        json.dumps(
            {
                "status": "public_analysis_ready",
                "endpoint_count": len(analysis["endpoint_inventory"]),
                "coverage_fraction": analysis["coverage_summary"]["coverage_fraction"],
                "is_100_percent_coverage": analysis["coverage_summary"][
                    "is_100_percent"
                ],
                "request_rows": sum(
                    int(row.get("request_rows") or 0)
                    for row in analysis["data_sources"]
                ),
                "epoch_rows": sum(
                    int(row.get("epoch_rows") or 0) for row in analysis["data_sources"]
                ),
                "pdf_created": False,
                "publication_mode": analysis["publication_mode"],
                "publication_status": analysis["publication_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
