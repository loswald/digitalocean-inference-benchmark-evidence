"""Environment-only credentials for DigitalOcean Serverless Inference."""

from __future__ import annotations

import os


def digitalocean_credentials() -> dict[str, str]:
    """Return the API key and endpoint without logging or persisting the key."""

    api_key = os.environ.get("DIGITALOCEAN_INFERENCE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Set DIGITALOCEAN_INFERENCE_API_KEY in the process environment."
        )
    api_base = os.environ.get(
        "DIGITALOCEAN_INFERENCE_API_BASE", "https://inference.do-ai.run/v1"
    ).strip()
    return {"api_key": api_key, "api_base": api_base.rstrip("/")}
