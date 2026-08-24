#!/usr/bin/env python3
"""Render the direct public DigitalOcean benchmark PDF from derived artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from do_benchmark.direct_report_pdf import build_pdf  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the clean public PDF from analysis.json or analysis.json.gz, "
            "named derived CSVs, and publication-safe charts. No network access "
            "is used."
        )
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help=(
            "Directory containing analysis.json or analysis.json.gz, derived "
            "CSVs, and charts/."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination PDF path.",
    )
    parser.add_argument(
        "--title",
        default="DigitalOcean Inference Endpoint Technical Benchmark",
        help="Public report title.",
    )
    parser.add_argument(
        "--subtitle",
        default=(
            "Public engineering report - heterogeneous workloads, operating "
            "envelopes, and uncertainty"
        ),
        help="Public report subtitle.",
    )
    parser.add_argument(
        "--mode",
        choices=("draft", "final"),
        default="draft",
        help=(
            "draft visibly watermarks every draft; final fails closed "
            "unless coverage, reconciliation, schema, sample, unit, CI, and safety gates pass"
        ),
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    output = build_pdf(
        arguments.artifacts,
        arguments.output,
        title=arguments.title,
        subtitle=arguments.subtitle,
        mode=arguments.mode,
    )
    print(
        json.dumps(
            {
                "network_access": False,
                "output_pdf": output.name,
                "public_inputs_only": True,
                "mode": arguments.mode,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
