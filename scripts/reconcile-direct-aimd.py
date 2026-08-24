#!/usr/bin/env python3
"""Mint the exact, secret-free 2026-08-23 direct AIMD reconciliation receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from do_benchmark.direct_aimd_reconcile import (  # noqa: E402
    write_reconciliation_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile the exact completed direct AIMD run to current prices."
    )
    parser.add_argument("--aimd-dir", type=Path, required=True)
    parser.add_argument("--prior-lineage-root", type=Path, required=True)
    parser.add_argument("--v3-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument(
        "--endpoint-freeze",
        type=Path,
        default=REPO_ROOT / "config" / "endpoint-freeze.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = write_reconciliation_receipt(
        args.aimd_dir,
        args.output,
        endpoint_freeze_path=args.endpoint_freeze,
        prior_lineage_root=args.prior_lineage_root,
        v3_checkpoint_dir=args.v3_checkpoint_dir,
        source_archive_path=args.source_archive,
    )
    print(
        json.dumps(
            {
                "receipt": str(args.output),
                "receipt_sha256": receipt["receipt_sha256"],
                "reconciled_cumulative_exposure_usd": receipt["settlement"][
                    "reconciled_cumulative_exposure_usd"
                ],
                "provider_requests_sent": 0,
                "credentials_loaded": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
