#!/usr/bin/env python3
"""Build a reproducibility manifest for DigitalOcean benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".jsonl",
    ".csv",
    ".toml",
    ".yml",
    ".cff",
    ".txt",
}
TEXT_NAMES = {".gitignore", ".gitattributes", "LICENSE"}


def canonical_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES:
        return data.replace(b"\r\n", b"\n")
    return data


def sha256(path: Path) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def repository_files(root: Path) -> list[Path]:
    """Return version-controlled files when run from a Git checkout.

    A public artifact manifest must not absorb local render directories, downloaded
    checkpoints, or other untracked QA material that happens to sit beside the
    checkout.  The filesystem fallback keeps source archives reproducible after
    Git metadata has been removed.
    """
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if tracked.returncode == 0:
        return [
            root / Path(item.decode("utf-8"))
            for item in tracked.stdout.split(b"\0")
            if item
        ]
    return sorted(root.rglob("*"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    root = args.artifact_dir.resolve()
    files = []
    for path in sorted(repository_files(root)):
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or path.name == "MANIFEST.json"
            or any(
                part
                in {
                    ".git",
                    ".venv",
                    ".pytest_cache",
                    ".ruff_cache",
                    "__pycache__",
                    "evidence",
                    "raw",
                    "runtime",
                    "tmp",
                }
                or part.endswith(".egg-info")
                for part in relative.parts
            )
        ):
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": len(canonical_bytes(path)),
                "sha256": sha256(path),
                "media_type": (
                    "application/jsonl"
                    if path.suffix == ".jsonl"
                    else "application/json"
                    if path.suffix == ".json"
                    else "text/csv"
                    if path.suffix == ".csv"
                    else "text/markdown"
                    if path.suffix == ".md"
                    else "application/pdf"
                    if path.suffix == ".pdf"
                    else "image/png"
                    if path.suffix == ".png"
                    else "application/gzip"
                    if path.suffix == ".gz"
                    else "application/octet-stream"
                ),
            }
        )
    manifest = {
        "schema_version": "digitalocean_inference_benchmark_artifact_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_role": "public inference performance and quality evidence",
        "rights_posture": "programmatic synthetic prompts and provider-generated responses; see DATA-NOTICE.md",
        "provider": "DigitalOcean Serverless Inference",
        "campaign_date": "2026-08-23 through 2026-08-24",
        "files": files,
        "recipe": {
            "runners": [
                "scripts/run-digitalocean-direct-aimd.py",
                "scripts/run-digitalocean-direct-soak.py",
                "scripts/run-digitalocean-direct-capability.py",
                "scripts/run-digitalocean-direct-context.py",
                "scripts/run-digitalocean-direct-completion.py",
            ],
            "analyzer": "scripts/analyze-direct-public-report.py",
            "pdf_builder": "scripts/build-direct-public-report-pdf.py",
            "seeds": [20260823, 20260824],
            "credential_contract": "process environment only; never serialized",
            "documentation_freeze": {
                "verified_at": "2026-08-24",
                "models": "https://docs.digitalocean.com/products/inference/details/models/",
                "pricing": "https://docs.digitalocean.com/products/inference/details/pricing/",
                "limits": "https://docs.digitalocean.com/products/inference/details/limits/",
                "multimodal": "https://docs.digitalocean.com/products/inference/how-to/use-multimodal-inference/",
                "quota_headers": "https://docs.digitalocean.com/products/inference/reference/quota-specific-response-headers/",
            },
        },
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"files": len(files), "bytes": sum(item["bytes"] for item in files)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
