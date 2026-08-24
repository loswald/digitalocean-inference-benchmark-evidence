#!/usr/bin/env python3
"""Build a small, reviewable closure queue from the public evidence tables.

The output contains identities and classifications only. It never copies prompts,
responses, headers, or credentials. A target is selected only when the latest
capability campaign marked the exact probe inconclusive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_targets(requests_path: Path, coverage_path: Path) -> dict[str, object]:
    targets: list[dict[str, str]] = []
    with requests_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_id") != "do-direct-capability-20260823":
                continue
            if row.get("coverage_conclusive", "").casefold() == "true":
                continue
            endpoint_id = row.get("endpoint_id", "").strip()
            probe_id = row.get("capability_probe_id", "").strip()
            if not endpoint_id or not probe_id:
                continue
            targets.append(
                {
                    "endpoint_id": endpoint_id,
                    "probe_id": probe_id,
                    "prior_classification": row.get(
                        "coverage_classification", "unknown"
                    ),
                    "prior_http_status": row.get("http_status", ""),
                }
            )

    unresolved: list[dict[str, str]] = []
    with coverage_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "inconclusive":
                continue
            unresolved.append(
                {
                    "endpoint_id": row["endpoint_id"],
                    "coverage_dimension": row["coverage_dimension"],
                }
            )

    targets.sort(key=lambda row: (row["endpoint_id"], row["probe_id"]))
    unresolved.sort(key=lambda row: (row["coverage_dimension"], row["endpoint_id"]))
    identity = {
        "schema_version": "do_closure_targets_v1",
        "source_requests_sha256": _sha256(requests_path),
        "source_coverage_sha256": _sha256(coverage_path),
        "capability_targets": targets,
        "unresolved_endpoint_dimensions": unresolved,
    }
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_targets(args.requests, args.coverage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "capability_targets": len(payload["capability_targets"]),
                "unresolved_endpoint_dimensions": len(
                    payload["unresolved_endpoint_dimensions"]
                ),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
