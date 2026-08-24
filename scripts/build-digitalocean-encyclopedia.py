#!/usr/bin/env python3
"""Build the concise public engineering encyclopedia from derived evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from do_benchmark.encyclopedia import build_pdf  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = build_pdf(args.artifacts, args.output)
    print(json.dumps({"status": "encyclopedia_built", "output": output.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
