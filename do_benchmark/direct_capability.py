"""Compact direct DigitalOcean capability and parameter-envelope benchmark.

This lane is intentionally bounded and standalone. Numeric
boundaries are isolated, all 17 advertised/exploratory interaction factors use
a verified strength-two covering array, and both documented multimodal routes
receive a compact image envelope.

Only sanitized measurements are persisted.  Prompts, model output, provider
response bodies, raw headers, and credentials never enter an artifact.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

import httpx
from PIL import Image

from do_benchmark.core import (
    API_DOC_GENERATED_DATE,
    MODEL_BY_ID,
    MODEL_DOC_VERIFIED_DATE,
    MODEL_SPECS,
    PRICING_DOC_DATE,
    BenchmarkTask,
    JsonlJournal,
    ModelSpec,
    ProviderHTTPError,
    StreamResult,
    canonical_json,
    parse_token_usage,
    quadrant_png_data_uri,
    score_result,
    stream_chat_completion,
    utc_now,
)
from do_benchmark.credentials import digitalocean_credentials
from do_benchmark.direct_aimd import BudgetLedger, sanitized_header_signals


REQUEST_SCHEMA = "do_direct_capability_request_v3"
RESERVATION_SCHEMA = "do_direct_reservation_v1"
MANIFEST_SCHEMA = "do_direct_capability_manifest_v3"
SUMMARY_SCHEMA = "do_direct_capability_summary_v3"
SCORER_CONTRACT_VERSION = "direct_capability_scorer_v3"
RESERVATION_CONTRACT_VERSION = "utf8_bytes_plus_512_or_1p5_planned_v1"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DIRECT_CAPABILITY_SOURCE_PATH = Path(__file__).resolve()
CORE_SOURCE_PATH = DIRECT_CAPABILITY_SOURCE_PATH.with_name("core.py")
DIRECT_AIMD_SOURCE_PATH = DIRECT_CAPABILITY_SOURCE_PATH.with_name("direct_aimd.py")
CAPABILITY_CLI_SOURCE_PATH = (
    REPOSITORY_ROOT / "scripts" / "run-digitalocean-direct-capability.py"
)
DOCUMENTATION_ARTIFACTS = ("config/endpoint-freeze.json",)

# Frozen from the live DigitalOcean /v1/models catalog retained in the checked-in
# endpoint freeze. Kimi K3 did not publish a limit;
# 65,536 is therefore an explicit probe anchor, not a documented-limit claim.
MAX_OUTPUT_TOKEN_ANCHORS: dict[str, tuple[int, str]] = {
    "arcee-trinity-large-thinking": (32_000, "live_catalog"),
    "deepseek-v4-flash-0731": (1_048_576, "live_catalog"),
    "gemma-4-31B-it": (8_192, "live_catalog"),
    "glm-5.2": (262_144, "live_catalog"),
    "kimi-k3": (65_536, "undocumented_probe_anchor"),
    "minimax-m2.5": (65_536, "live_catalog"),
    "mimo-v2.5-pro": (262_144, "live_catalog"),
    "nemotron-3-ultra-550b": (131_072, "live_catalog"),
    "nvidia-nemotron-3-super-120b": (32_768, "live_catalog"),
    "openai-gpt-oss-120b": (4_096, "live_catalog"),
    "qwen3.5-397b-a17b": (131_072, "live_catalog"),
    "qwen3.8-max": (262_144, "live_catalog"),
}

DOCUMENTED_PROMPT_CACHING: dict[str, bool | None] = {
    spec.model_id: (None if spec.model_id == "nemotron-3-ultra-550b" else True)
    for spec in MODEL_SPECS
}
DOCUMENTED_VISION_MODELS = frozenset({"kimi-k3", "qwen3.8-max"})
DOCUMENTED_TOOL_MODELS = frozenset({"glm-5.2", "mimo-v2.5-pro", "qwen3.8-max"})
DOCUMENTED_STRUCTURED_OUTPUT_MODELS = DOCUMENTED_TOOL_MODELS
TOOL_COUNT_ANCHORS = (1, 8, 32, 64)
TOOL_SCHEMA_BYTE_ANCHORS = (256, 1_024, 8_192, 32_768, 65_536)
TOOL_NESTING_DEPTH_ANCHORS = (1, 4, 8, 16)
TOOL_ARGUMENT_BYTE_ANCHORS = (64, 1_024, 8_192, 32_768)
TOOL_REQUIRED_OPTIONAL_MODES = ("required_only", "mixed_required_optional")
TOOL_MALFORMED_CASES = (
    "missing_function_name",
    "invalid_json_schema_type",
    "name_length_65",
)
DEFAULT_QUADRANT_ORDER = ("red", "green", "blue", "yellow")
QUADRANT_ORDERS: tuple[tuple[str, str, str, str], ...] = (
    DEFAULT_QUADRANT_ORDER,
    ("green", "red", "yellow", "blue"),
    ("blue", "yellow", "red", "green"),
    ("yellow", "blue", "green", "red"),
    ("red", "blue", "yellow", "green"),
    ("green", "yellow", "blue", "red"),
    ("blue", "red", "green", "yellow"),
    ("yellow", "green", "red", "blue"),
)

RequestExecutor = Callable[[str, BenchmarkTask, int], Awaitable[StreamResult]]


class CapabilityPreflightError(RuntimeError):
    """Raised before a capability request when persisted state is unsafe."""


class OutputDirectoryLease:
    """Non-blocking process lease for one capability output directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> "OutputDirectoryLease":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"\0")
            self.handle.flush()
            os.fsync(self.handle.fileno())
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            self.handle.close()
            self.handle = None
            raise CapabilityPreflightError(
                "another process holds the capability output-directory execution lease"
            ) from error
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


@dataclass(frozen=True)
class CapabilityConfig:
    output_dir: Path
    model_ids: tuple[str, ...]
    seed: int = 20260823
    max_workers: int = 48
    per_model_concurrency: int = 4
    request_timeout_seconds: float = 90.0
    max_cost_usd: float = 200.0
    prior_cost_usd: float = 0.0
    stop_launch_at: datetime | None = None

    def validate(self) -> None:
        if not self.model_ids:
            raise ValueError("at least one model is required")
        unknown = sorted(set(self.model_ids) - MODEL_BY_ID.keys())
        if unknown:
            raise ValueError(f"unknown DigitalOcean models: {', '.join(unknown)}")
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("model IDs must be unique")
        if self.max_workers < 1 or self.per_model_concurrency < 1:
            raise ValueError("concurrency values must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        if self.max_cost_usd <= 0 or self.prior_cost_usd < 0:
            raise ValueError("invalid cost envelope")
        if self.prior_cost_usd > self.max_cost_usd:
            raise ValueError("prior cost already exceeds the cumulative cap")
        if self.stop_launch_at is not None and self.stop_launch_at.tzinfo is None:
            raise ValueError("stop_launch_at must be timezone-aware")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA,
            "models": list(self.model_ids),
            "seed": self.seed,
            "max_workers": self.max_workers,
            "per_model_concurrency": self.per_model_concurrency,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_cost_usd": self.max_cost_usd,
            "prior_cost_usd": self.prior_cost_usd,
        }


@dataclass(frozen=True)
class CapabilityCell:
    request_id: str
    model_id: str
    probe_id: str
    dimension: str
    state: str
    design_role: str
    coverage_tags: tuple[str, ...]
    bindings: dict[str, Any]
    task: BenchmarkTask
    max_output_tokens: int
    rendered_payload_sha256: str
    scorer_contract_sha256: str
    model_contract_sha256: str
    documentation_contract_sha256: str
    request_identity_sha256: str
    provider_send_expected: bool = True
    local_terminal_status: str | None = None

    def sanitized_plan_row(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_id": self.model_id,
            "probe_id": self.probe_id,
            "dimension": self.dimension,
            "state": self.state,
            "design_role": self.design_role,
            "coverage_tags": list(self.coverage_tags),
            "bindings": self.bindings,
            "requested_max_output_tokens": self.max_output_tokens,
            "rendered_payload_sha256": self.rendered_payload_sha256,
            "scorer_contract_sha256": self.scorer_contract_sha256,
            "model_contract_sha256": self.model_contract_sha256,
            "documentation_contract_sha256": self.documentation_contract_sha256,
            "request_identity_sha256": self.request_identity_sha256,
            "provider_send_expected": self.provider_send_expected,
            "local_terminal_status": self.local_terminal_status,
            "workload_id": workload_for_cell(self),
            "shape": "capability_envelope",
            "phase": self.design_role,
            "cell_id": self.request_id,
            "task": {
                "task_id": self.task.task_id,
                "family": workload_for_cell(self),
                "context_bucket": self.task.context_bucket,
                "output_bucket": self.task.output_bucket,
                "requires_vision": self.task.requires_vision,
            },
        }


def workload_for_cell(cell: CapabilityCell) -> str:
    if "capability_smoke" in cell.coverage_tags:
        return "capability_smoke"
    if (
        cell.dimension == "pairwise_core"
        and cell.bindings.get("response_format") == "json_schema"
    ):
        return "structured_output"
    if cell.dimension == "pairwise_core" or "interaction" in cell.dimension:
        return "parameter_interactions"
    if cell.dimension == "vision":
        return "vision"
    if cell.dimension in {"tools", "parallel_tool_calls"} or cell.dimension.startswith(
        "tool_"
    ):
        return "tool_calling"
    if cell.dimension == "response_format":
        return "structured_output"
    if cell.dimension in {"max_tokens", "max_completion_tokens"}:
        return "output_length"
    return "parameter_validation"


def _tool_definition(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Look up one synthetic station reading.",
            "parameters": {
                "type": "object",
                "properties": {
                    "station": {"type": "string", "enum": ["station-7"]},
                },
                "required": ["station"],
                "additionalProperties": False,
            },
        },
    }


@lru_cache(maxsize=16)
def _synthetic_image_data_uri(
    width: int,
    height: int,
    format_name: str,
    noisy: bool = False,
    quadrant_order: tuple[str, str, str, str] = DEFAULT_QUADRANT_ORDER,
) -> str:
    """Return a deterministic valid synthetic image with no embedded answer text."""

    colour_values = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
    }
    if set(quadrant_order) != set(DEFAULT_QUADRANT_ORDER):
        raise ValueError(
            "quadrant_order must be a permutation of the four signal colours"
        )
    image = Image.new("RGB", (width, height))
    half_width, half_height = width // 2, height // 2
    boxes = (
        (0, 0, half_width, half_height),
        (half_width, 0, width, half_height),
        (0, half_height, half_width, height),
        (half_width, half_height, width, height),
    )
    for colour, box in zip(quadrant_order, boxes, strict=True):
        image.paste(colour_values[colour], box)
    if noisy:
        # Byte-envelope images must still carry the same deterministic visual
        # signal as the small baseline. Low-amplitude deterministic noise keeps
        # PNG payloads large without turning the quality target into fiction.
        raw = random.Random(20260823 + width * 17 + height).randbytes(
            width * height * 3
        )
        noise = Image.frombytes("RGB", (width, height), raw)
        image = Image.blend(image, noise, 0.22)
    normalized = format_name.upper()
    mime = "jpeg" if normalized == "JPEG" else normalized.casefold()
    buffer = io.BytesIO()
    save_kwargs: dict[str, Any] = {}
    if normalized == "JPEG":
        save_kwargs = {"quality": 90, "optimize": False, "progressive": False}
    elif normalized == "WEBP":
        save_kwargs = {"quality": 90, "lossless": False, "method": 4}
    image.save(buffer, format=normalized, **save_kwargs)
    return f"data:image/{mime};base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def _encoded_image_bytes(data_uri: str) -> int:
    encoded = data_uri.split(",", 1)[1]
    return len(base64.b64decode(encoded, validate=True))


def _plain_task(
    task_id: str, *, parameters: Mapping[str, Any] | None = None
) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        family="direct_capability_parameter",
        context_bucket="short",
        output_bucket="short",
        messages=[{"role": "user", "content": "Return only CAP-OK"}],
        expected={"kind": "exact_text", "value": "CAP-OK"},
        parameters=dict(parameters or {}),
        metadata={"planned_input_tokens": 24},
    )


def _structured_task(
    task_id: str,
    response_mode: str,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> BenchmarkTask:
    response_format: dict[str, Any]
    if response_mode == "text":
        return _plain_task(task_id, parameters=parameters)
    if response_mode == "json_object":
        response_format = {"type": "json_object"}
    elif response_mode == "json_schema":
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "capability_probe",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string", "const": "CAP-OK"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            },
        }
    else:  # pragma: no cover - guarded by the design builder
        raise ValueError(f"unknown response mode: {response_mode}")
    return BenchmarkTask(
        task_id=task_id,
        family="direct_capability_structured_output",
        context_bucket="short",
        output_bucket="short",
        messages=[{"role": "user", "content": 'Return only {"status":"CAP-OK"}'}],
        expected={"kind": "json_exact", "value": {"status": "CAP-OK"}},
        response_format=response_format,
        parameters=dict(parameters or {}),
        metadata={"planned_input_tokens": 32},
    )


def _tool_task(
    task_id: str,
    tool_mode: str,
    *,
    response_mode: str = "text",
    parameters: Mapping[str, Any] | None = None,
) -> BenchmarkTask:
    if tool_mode == "none":
        return _structured_task(task_id, response_mode, parameters=parameters)
    target = "lookup_station"
    tool_choice: str | dict[str, Any]
    if tool_mode in {"auto", "required"}:
        tool_choice = tool_mode
    elif tool_mode == "named":
        tool_choice = {"type": "function", "function": {"name": target}}
    else:  # pragma: no cover - guarded by the design builder
        raise ValueError(f"unknown tool mode: {tool_mode}")
    response_format = None
    if response_mode == "json_object":
        response_format = {"type": "json_object"}
    elif response_mode == "json_schema":
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "capability_probe",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string", "const": "CAP-OK"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            },
        }
    elif response_mode != "text":  # pragma: no cover - design builder guards it
        raise ValueError(f"unknown response mode: {response_mode}")
    return BenchmarkTask(
        task_id=task_id,
        family="direct_capability_tools",
        context_bucket="short_tool",
        output_bucket="tool",
        messages=[
            {
                "role": "user",
                "content": "Call lookup_station exactly once with station set to station-7.",
            }
        ],
        expected={
            "kind": "tool_exact",
            "value": {"name": target, "arguments": {"station": "station-7"}},
        },
        tools=[_tool_definition(target)],
        tool_choice=tool_choice,
        response_format=response_format,
        parameters=dict(parameters or {}),
        metadata={"planned_input_tokens": 128},
    )


def _parallel_tool_task(task_id: str, enabled: bool) -> BenchmarkTask:
    first = _tool_definition("lookup_station")
    second = _tool_definition("lookup_backup")
    station_call = {"name": "lookup_station", "arguments": {"station": "station-7"}}
    backup_call = {"name": "lookup_backup", "arguments": {"station": "station-7"}}
    expected_calls = [station_call, backup_call] if enabled else [station_call]
    return BenchmarkTask(
        task_id=task_id,
        family="direct_capability_parallel_tools",
        context_bucket="short_tool",
        output_bucket="tool",
        messages=[
            {
                "role": "user",
                "content": (
                    (
                        "Call lookup_station and lookup_backup, each once, with station "
                        "set to station-7."
                    )
                    if enabled
                    else (
                        "Call lookup_station exactly once with station set to station-7. "
                        "Do not call lookup_backup."
                    )
                ),
            }
        ],
        expected={
            "kind": "tool_exact",
            "value": {
                "name": "lookup_station",
                "arguments": {"station": "station-7"},
            },
        },
        tools=[first, second],
        tool_choice="required",
        parameters={"parallel_tool_calls": enabled},
        metadata={
            "planned_input_tokens": 220,
            "expected_tool_call_count": len(expected_calls),
            "expected_tool_calls": expected_calls,
            "parallel_tool_calls_enabled": enabled,
        },
    )


def _exact_tool_schema_bytes(target_bytes: int) -> list[dict[str, Any]]:
    tool = _tool_definition("lookup_station")
    tool["function"]["description"] = ""
    base_bytes = len(canonical_json([tool]).encode("utf-8"))
    if target_bytes < base_bytes:
        raise ValueError("tool schema byte target is smaller than the minimal schema")
    tool["function"]["description"] = "x" * (target_bytes - base_bytes)
    tools = [tool]
    if len(canonical_json(tools).encode("utf-8")) != target_bytes:
        raise AssertionError("exact tool schema byte construction failed")
    return tools


def _nested_tool(depth: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if depth < 1:
        raise ValueError("tool nesting depth must be positive")
    schema: dict[str, Any] = {"type": "string"}
    arguments: Any = "ok"
    for level in range(depth, 0, -1):
        key = f"level_{level}"
        schema = {
            "type": "object",
            "properties": {key: schema},
            "required": [key],
            "additionalProperties": False,
        }
        arguments = {key: arguments}
    tool = {
        "type": "function",
        "function": {
            "name": "nested_probe",
            "description": "Return a synthetic nested payload.",
            "parameters": schema,
        },
    }
    return tool, arguments


def _tool_envelope_task(
    task_id: str,
    *,
    tools: Sequence[Mapping[str, Any]],
    target_name: str,
    expected_arguments: Mapping[str, Any],
    prompt: str,
    max_output_tokens: int = 256,
) -> tuple[BenchmarkTask, int]:
    return (
        BenchmarkTask(
            task_id=task_id,
            family="direct_capability_tool_envelope",
            context_bucket="tool_envelope",
            output_bucket="tool",
            messages=[{"role": "user", "content": prompt}],
            expected={
                "kind": "tool_exact",
                "value": {"name": target_name, "arguments": dict(expected_arguments)},
            },
            tools=[dict(tool) for tool in tools],
            tool_choice={"type": "function", "function": {"name": target_name}},
            metadata={"planned_input_tokens": 256},
        ),
        max_output_tokens,
    )


def _malformed_tool_task(task_id: str, case: str) -> BenchmarkTask:
    function: dict[str, Any] = {
        "name": "malformed_probe",
        "description": "Deliberately malformed capability probe.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    }
    if case == "missing_function_name":
        function.pop("name")
    elif case == "invalid_json_schema_type":
        function["parameters"]["type"] = "not_a_json_schema_type"
    elif case == "name_length_65":
        function["name"] = "n" * 65
    else:  # pragma: no cover - guarded by the frozen anchor list
        raise ValueError(f"unknown malformed tool case: {case}")
    return BenchmarkTask(
        task_id=task_id,
        family="direct_capability_tool_malformed",
        context_bucket="tool_envelope",
        output_bucket="short",
        messages=[{"role": "user", "content": "Return only CAP-OK"}],
        expected={"kind": "exact_text", "value": "CAP-OK"},
        tools=[{"type": "function", "function": function}],
        tool_choice="required",
        metadata={"planned_input_tokens": 256},
    )


def _tool_envelope_rows(
    model_id: str,
) -> list[tuple[str, str, str, dict[str, Any], BenchmarkTask, int]]:
    if model_id not in DOCUMENTED_TOOL_MODELS:
        return []
    rows: list[tuple[str, str, str, dict[str, Any], BenchmarkTask, int]] = []

    for count in TOOL_COUNT_ANCHORS:
        tools = [_tool_definition(f"tool_{index:03d}") for index in range(count)]
        target = "tool_000"
        task, limit = _tool_envelope_task(
            f"cap-tool-count-{count}",
            tools=tools,
            target_name=target,
            expected_arguments={"station": "station-7"},
            prompt=(
                f"Call {target} exactly once with station set to station-7. "
                "Do not call another tool."
            ),
        )
        rows.append(
            (
                f"tool-count-{count}",
                "tool_count",
                str(count),
                {
                    "tool_count": count,
                    "tool_schema_bytes": len(canonical_json(tools).encode("utf-8")),
                },
                task,
                limit,
            )
        )

    for target_bytes in TOOL_SCHEMA_BYTE_ANCHORS:
        tools = _exact_tool_schema_bytes(target_bytes)
        task, limit = _tool_envelope_task(
            f"cap-tool-schema-bytes-{target_bytes}",
            tools=tools,
            target_name="lookup_station",
            expected_arguments={"station": "station-7"},
            prompt="Call lookup_station exactly once with station set to station-7.",
        )
        rows.append(
            (
                f"tool-schema-bytes-{target_bytes}",
                "tool_schema_bytes",
                str(target_bytes),
                {"tool_schema_bytes": target_bytes, "exact_rendered_bytes": True},
                task,
                limit,
            )
        )

    for depth in TOOL_NESTING_DEPTH_ANCHORS:
        tool, arguments = _nested_tool(depth)
        task, limit = _tool_envelope_task(
            f"cap-tool-nesting-depth-{depth}",
            tools=[tool],
            target_name="nested_probe",
            expected_arguments=arguments,
            prompt=(
                "Call nested_probe exactly once with the exact nested argument "
                f"{canonical_json(arguments)}"
            ),
            max_output_tokens=512,
        )
        rows.append(
            (
                f"tool-nesting-depth-{depth}",
                "tool_nesting_depth",
                str(depth),
                {
                    "nesting_depth": depth,
                    "tool_schema_bytes": len(canonical_json([tool]).encode("utf-8")),
                },
                task,
                limit,
            )
        )

    for target_bytes in TOOL_ARGUMENT_BYTE_ANCHORS:
        prefix_bytes = len(canonical_json({"payload": ""}).encode("utf-8"))
        arguments = {"payload": "x" * (target_bytes - prefix_bytes)}
        actual_bytes = len(canonical_json(arguments).encode("utf-8"))
        if actual_bytes != target_bytes:
            raise AssertionError("exact tool argument byte construction failed")
        tool = {
            "type": "function",
            "function": {
                "name": "argument_probe",
                "description": "Return the requested synthetic argument payload.",
                "parameters": {
                    "type": "object",
                    "properties": {"payload": {"type": "string"}},
                    "required": ["payload"],
                    "additionalProperties": False,
                },
            },
        }
        task, limit = _tool_envelope_task(
            f"cap-tool-argument-bytes-{target_bytes}",
            tools=[tool],
            target_name="argument_probe",
            expected_arguments=arguments,
            prompt=(
                "Call argument_probe exactly once. Set payload to exactly the "
                f"following {target_bytes - prefix_bytes} characters: "
                + str(arguments["payload"])
            ),
            max_output_tokens=target_bytes + 128,
        )
        rows.append(
            (
                f"tool-argument-bytes-{target_bytes}",
                "tool_argument_bytes",
                str(target_bytes),
                {
                    "argument_payload_bytes": target_bytes,
                    "output_headroom_tokens": 128,
                },
                task,
                limit,
            )
        )

    for mode in TOOL_REQUIRED_OPTIONAL_MODES:
        properties: dict[str, Any] = {"station": {"type": "string"}}
        if mode == "mixed_required_optional":
            properties["note"] = {"type": "string"}
        tool = {
            "type": "function",
            "function": {
                "name": "required_optional_probe",
                "description": "Probe required and optional fields.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": ["station"],
                    "additionalProperties": False,
                },
            },
        }
        task, limit = _tool_envelope_task(
            f"cap-tool-fields-{mode}",
            tools=[tool],
            target_name="required_optional_probe",
            expected_arguments={"station": "station-7"},
            prompt=(
                "Call required_optional_probe exactly once with station set to "
                "station-7. Omit every optional field."
            ),
        )
        rows.append(
            (
                f"tool-fields-{mode}",
                "tool_required_optional",
                mode,
                {"required_optional_mode": mode},
                task,
                limit,
            )
        )

    for case in TOOL_MALFORMED_CASES:
        rows.append(
            (
                f"tool-malformed-{case}",
                "tool_malformed_schema",
                case,
                {"malformed_case": case},
                _malformed_tool_task(f"cap-tool-malformed-{case}", case),
                64,
            )
        )
    return rows


def _vision_task(
    task_id: str,
    *,
    image_uris: Sequence[str],
    state: str,
    format_name: str,
    dimensions: str,
    malformed: bool = False,
    mixed_text_tokens: int = 0,
    quadrant_orders: Sequence[Sequence[str]] | None = None,
) -> BenchmarkTask:
    filler = ""
    if mixed_text_tokens:
        filler = (" synthetic telemetry" * mixed_text_tokens) + "\n"
    semantic_orders = tuple(
        tuple(str(colour) for colour in order)
        for order in (quadrant_orders or [DEFAULT_QUADRANT_ORDER] * len(image_uris))
    )
    if len(semantic_orders) != len(image_uris):
        raise ValueError("every image needs one semantic quadrant-order target")
    expected_groups = [", ".join(order) for order in semantic_orders]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                filler
                + "For every image in the order supplied, name the dominant colour "
                "in each quadrant in reading order (top-left, top-right, "
                "bottom-left, bottom-right). Within each image group, return four "
                "lowercase colour names separated by comma-space. Separate image "
                "groups with space-pipe-space and return no other text."
            ),
        }
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_uri}}
        for image_uri in image_uris
    )
    return BenchmarkTask(
        task_id=task_id,
        family="direct_capability_vision",
        context_bucket="image",
        output_bucket="short",
        messages=[{"role": "user", "content": content}],
        expected={"kind": "exact_text", "value": " | ".join(expected_groups)},
        requires_vision=True,
        metadata={
            "planned_input_tokens": 256 + mixed_text_tokens,
            "vision_state": state,
            "image_count": len(image_uris),
            "format": format_name,
            "dimensions": dimensions,
            "malformed": malformed,
            "mixed_text_tokens": mixed_text_tokens,
            "semantic_image_group_count": len(semantic_orders),
        },
    )


def greedy_pairwise(factors: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Return a deterministic IPO-style strength-two covering array.

    Unlike a Cartesian-candidate greedy search, this stays small when the
    advertised factor surface grows. Completeness is independently asserted by
    tests over every cross-factor state pair.
    """

    names = list(factors)
    if len(names) < 2 or any(not factors[name] for name in names):
        raise ValueError("a covering array needs at least two non-empty factors")
    levels = {name: tuple(factors[name]) for name in names}
    rows: list[dict[str, Any]] = [
        {names[0]: first, names[1]: second}
        for first in levels[names[0]]
        for second in levels[names[1]]
    ]
    for position in range(2, len(names)):
        new_name = names[position]
        previous = names[:position]
        uncovered = {
            (old_name, canonical_json(old_value), canonical_json(new_value))
            for old_name in previous
            for old_value in levels[old_name]
            for new_value in levels[new_name]
        }
        for row in rows:
            best = max(
                levels[new_name],
                key=lambda candidate: (
                    sum(
                        (
                            old_name,
                            canonical_json(row[old_name]),
                            canonical_json(candidate),
                        )
                        in uncovered
                        for old_name in previous
                    ),
                    -levels[new_name].index(candidate),
                ),
            )
            row[new_name] = best
            for old_name in previous:
                uncovered.discard(
                    (old_name, canonical_json(row[old_name]), canonical_json(best))
                )
        while uncovered:
            old_name, old_value_key, new_value_key = sorted(uncovered)[0]
            old_value = next(
                value
                for value in levels[old_name]
                if canonical_json(value) == old_value_key
            )
            new_value = next(
                value
                for value in levels[new_name]
                if canonical_json(value) == new_value_key
            )
            row: dict[str, Any] = {}
            for prior_name in previous:
                if prior_name == old_name:
                    row[prior_name] = old_value
                else:
                    candidates = levels[prior_name]
                    row[prior_name] = max(
                        candidates,
                        key=lambda candidate: (
                            (
                                prior_name,
                                canonical_json(candidate),
                                new_value_key,
                            )
                            in uncovered,
                            -candidates.index(candidate),
                        ),
                    )
            row[new_name] = new_value
            rows.append(row)
            for prior_name in previous:
                uncovered.discard(
                    (prior_name, canonical_json(row[prior_name]), new_value_key)
                )
    unique = {canonical_json(row): row for row in rows}
    return [unique[key] for key in sorted(unique)]


PAIRWISE_FACTORS: dict[str, tuple[Any, ...]] = {
    "temperature": (0, 1, 2),
    "top_p": (0, 0.5, 1),
    "presence_penalty": (-2, 0, 2),
    "frequency_penalty": (-2, 0, 2),
    "stream": (True, False),
    "seed_mode": ("omitted", "fixed"),
    "stop_count": (0, 1, 4),
    "logprob_mode": ("off", "top_0", "top_10", "top_20"),
    "response_format": ("text", "json_object", "json_schema"),
    "tool_mode": ("none", "auto", "required", "named"),
    "parallel_tool_calls": (False, True),
    "logit_bias_value": (-100, 0, 100),
    "n": (1, 8, 16),
    "max_completion_tokens": ("omitted", 64, 1_024, 4_096),
    "reasoning_effort": ("omitted", "none", "low", "medium", "high"),
    "user": ("omitted", "present"),
    "output_tokens": (64, 512, 4_096),
}
PAIRWISE_ROWS: tuple[dict[str, Any], ...] = tuple(greedy_pairwise(PAIRWISE_FACTORS))

THREE_WAY_ROWS: tuple[dict[str, Any], ...] = (
    {"temperature": 0, "top_p": 0, "output_tokens": 64},
    {"temperature": 0, "top_p": 1, "output_tokens": 4_096},
    {"temperature": 2, "top_p": 0, "output_tokens": 4_096},
    {"temperature": 2, "top_p": 1, "output_tokens": 64},
    {"temperature": 1, "top_p": 0.5, "output_tokens": 512},
    {"temperature": 0.5, "top_p": 0.9, "output_tokens": 2_048},
)


def _interaction_task(
    task_id: str, binding: Mapping[str, Any]
) -> tuple[BenchmarkTask, int]:
    parameters: dict[str, Any] = {
        "temperature": binding["temperature"],
        "top_p": binding["top_p"],
        "presence_penalty": binding.get("presence_penalty", 0),
        "frequency_penalty": binding.get("frequency_penalty", 0),
        "stream": bool(binding.get("stream", True)),
        "parallel_tool_calls": bool(binding.get("parallel_tool_calls", False)),
        "logit_bias": {"0": int(binding.get("logit_bias_value", 0))},
        "n": int(binding.get("n", 1)),
        # Explicit here so the covering array genuinely exercises the
        # max_tokens × max_completion_tokens pair when both are selected.
        "max_tokens": int(binding.get("output_tokens", 64)),
    }
    max_completion = binding.get("max_completion_tokens", "omitted")
    if max_completion != "omitted":
        parameters["max_completion_tokens"] = int(max_completion)
    reasoning_effort = binding.get("reasoning_effort", "omitted")
    if reasoning_effort != "omitted":
        parameters["reasoning_effort"] = reasoning_effort
    if binding.get("user") == "present":
        parameters["user"] = "capability-envelope"
    if binding.get("seed_mode") == "fixed":
        parameters["seed"] = 42
    stop_count = int(binding.get("stop_count") or 0)
    if stop_count:
        parameters["stop"] = [f"<CAP_STOP_{index}>" for index in range(stop_count)]
    logprob_mode = str(binding.get("logprob_mode") or "off")
    if logprob_mode == "off":
        parameters["logprobs"] = False
    else:
        parameters["logprobs"] = True
        parameters["top_logprobs"] = int(logprob_mode.rsplit("_", 1)[1])
    response_mode = str(binding.get("response_format") or "text")
    tool_mode = str(binding.get("tool_mode") or "none")
    task = _tool_task(
        task_id,
        tool_mode,
        response_mode=response_mode,
        parameters=parameters,
    )
    return task, int(binding.get("output_tokens") or 64)


def _pairwise_tasks() -> list[tuple[str, str, dict[str, Any], BenchmarkTask, int]]:
    rows: list[tuple[str, str, dict[str, Any], BenchmarkTask, int]] = []
    for index, binding in enumerate(PAIRWISE_ROWS):
        task_id = f"cap-pairwise-{index:02d}"
        task, limit = _interaction_task(task_id, binding)
        rows.append((task_id, f"row-{index:02d}", dict(binding), task, limit))
    return rows


def _model_probe_tasks(
    model_id: str,
) -> list[tuple[str, str, str, dict[str, Any], BenchmarkTask, int]]:
    rows: list[tuple[str, str, str, dict[str, Any], BenchmarkTask, int]] = []

    rows.append(
        (
            "capability-smoke",
            "capability_smoke",
            "basic_chat",
            {"stream": True, "temperature": 0},
            _plain_task("capability-smoke", parameters={"stream": True}),
            64,
        )
    )

    for value in (-0.01, 0.0, 0.5, 1.0, 1.5, 2.0, 2.01):
        label = str(value)
        rows.append(
            (
                f"temperature-{label}",
                "temperature",
                label,
                {"temperature": value},
                _plain_task(
                    f"cap-temperature-{label}", parameters={"temperature": value}
                ),
                64,
            )
        )
    for value in (-0.01, 0.0, 0.25, 0.5, 0.75, 1.0, 1.01):
        label = str(value)
        rows.append(
            (
                f"top-p-{label}",
                "top_p",
                label,
                {"top_p": value},
                _plain_task(f"cap-top-p-{label}", parameters={"top_p": value}),
                64,
            )
        )

    penalty_values = (-2.01, -2.0, -1.0, 0.0, 1.0, 2.0, 2.01)
    presence_values = penalty_values[3:] + penalty_values[:3]
    for index, (frequency, presence) in enumerate(
        zip(penalty_values, presence_values, strict=True)
    ):
        parameters = {
            "frequency_penalty": frequency,
            "presence_penalty": presence,
        }
        rows.append(
            (
                f"penalty-boundaries-{index}",
                "parameter_interaction_penalties",
                f"row-{index}",
                parameters,
                _plain_task(f"cap-penalties-{index}", parameters=parameters),
                64,
            )
        )

    for value in (-1, 0, 5, 10, 15, 20, 21):
        parameters = {"logprobs": True, "top_logprobs": value}
        rows.append(
            (
                f"top-logprobs-{value}",
                "top_logprobs",
                str(value),
                parameters,
                _plain_task(f"cap-top-logprobs-{value}", parameters=parameters),
                64,
            )
        )

    extended_values = (
        # logit bias, n, max_completion_tokens, reasoning effort, user
        (-101, 1, 1, None, None),
        (-100, 4, 64, "none", "capability-envelope"),
        (-50, 8, 256, "low", None),
        (0, 12, 1_024, "medium", "capability-envelope"),
        (50, 16, 4_096, "high", None),
        (100, 0, 4_097, None, "capability-envelope"),
        (101, 17, 64, "low", None),
    )
    for index, (bias, n_value, max_completion, reasoning, user_value) in enumerate(
        extended_values
    ):
        parameters = {
            "logit_bias": {"0": bias},
            "n": n_value,
            "max_completion_tokens": max_completion,
            "reasoning_effort": reasoning,
            "user": user_value,
        }
        bindings = {
            "logit_bias_token_id": "0",
            "logit_bias_value": bias,
            "n": n_value,
            "max_completion_tokens": max_completion,
            "reasoning_effort": reasoning,
            "user": user_value,
        }
        rows.append(
            (
                f"extended-parameters-{index}",
                "parameter_interaction_extended",
                f"row-{index}",
                bindings,
                _plain_task(f"cap-extended-{index}", parameters=parameters),
                64,
            )
        )

    # One clean transport probe keeps max_completion_tokens support separable
    # from the compact multi-parameter interaction rows above.
    rows.append(
        (
            "max-completion-tokens-isolated",
            "max_completion_tokens",
            "medium",
            {"max_completion_tokens": 64, "stream": False},
            _plain_task(
                "cap-max-completion-tokens-isolated",
                parameters={"max_completion_tokens": 64, "stream": False},
            ),
            64,
        )
    )

    maximum, source = MAX_OUTPUT_TOKEN_ANCHORS[model_id]
    output_states = (
        ("small", 1),
        ("medium", min(256, maximum)),
        ("high", maximum),
        ("just_over", maximum + 1),
    )
    for label, value in output_states:
        bindings = {
            "max_tokens": value,
            "maximum_anchor": maximum,
            "maximum_anchor_source": source,
        }
        rows.append(
            (
                f"max-tokens-{label}",
                "max_tokens",
                label,
                bindings,
                _plain_task(f"cap-max-tokens-{label}"),
                value,
            )
        )

    for probe_id, state, binding, task, limit in _pairwise_tasks():
        rows.append(
            (
                probe_id,
                "pairwise_core",
                state,
                binding,
                task,
                limit,
            )
        )

    for index, binding in enumerate(THREE_WAY_ROWS):
        task_id = f"cap-temperature-top-p-output-{index:02d}"
        task = _plain_task(
            task_id,
            parameters={
                "temperature": binding["temperature"],
                "top_p": binding["top_p"],
            },
        )
        rows.append(
            (
                task_id,
                "parameter_interaction_temperature_top_p_output",
                f"row-{index:02d}",
                dict(binding),
                task,
                int(binding["output_tokens"]),
            )
        )

    for mode in ("required", "named"):
        rows.append(
            (
                f"tools-{mode}",
                "tools",
                mode,
                {"tool_mode": mode},
                _tool_task(f"cap-tools-{mode}", mode),
                64,
            )
        )
    rows.extend(_tool_envelope_rows(model_id))
    if model_id in DOCUMENTED_STRUCTURED_OUTPUT_MODELS:
        for response_mode in ("json_object", "json_schema"):
            rows.append(
                (
                    f"response-format-isolated-{response_mode}",
                    "response_format",
                    response_mode,
                    {
                        "response_format": response_mode,
                        "documentation_status": "documented_supported",
                    },
                    _structured_task(
                        f"cap-response-format-isolated-{response_mode}",
                        response_mode,
                    ),
                    128,
                )
            )
    for enabled in (False, True):
        rows.append(
            (
                f"parallel-tool-calls-{str(enabled).lower()}",
                "parallel_tool_calls",
                str(enabled).lower(),
                {"parallel_tool_calls": enabled},
                _parallel_tool_task(f"cap-parallel-{str(enabled).lower()}", enabled),
                96,
            )
        )
    rows.append(
        (
            "stop-sequence",
            "stop",
            "present",
            {"stop": [" END"]},
            BenchmarkTask(
                task_id="cap-stop",
                family="direct_capability_stop",
                context_bucket="short",
                output_bucket="short",
                messages=[
                    {
                        "role": "user",
                        "content": "Return exactly CAP-OK END and then the word NEVER.",
                    }
                ],
                expected={"kind": "exact_text", "value": "CAP-OK"},
                parameters={"stop": [" END"]},
                metadata={"planned_input_tokens": 28},
            ),
            64,
        )
    )
    rows.append(
        (
            "seed-explicit",
            "seed",
            "42",
            {"seed": 42},
            _plain_task("cap-seed", parameters={"seed": 42}),
            64,
        )
    )
    rows.append(
        (
            "seed-explicit-replicate-2",
            "seed",
            "42-replicate-2",
            {"seed": 42, "replicate_index": 2},
            # Same payload as seed-explicit; only the durable request identity differs.
            _plain_task("cap-seed", parameters={"seed": 42}),
            64,
        )
    )
    cache_documented = DOCUMENTED_PROMPT_CACHING[model_id]
    rows.append(
        (
            "caching-explicit-option-documentation",
            "caching_option",
            "documented_unavailable",
            {
                "automatic_cache_documented": cache_documented,
                "explicit_request_option": None,
                "documentation_status": (
                    "feature_documented_but_no_explicit_request_option"
                    if cache_documented is True
                    else "documentation_unavailable"
                ),
            },
            _plain_task("cap-caching-option-documentation"),
            64,
        )
    )
    small_png = quadrant_png_data_uri(64)
    rows.append(
        (
            "vision-small-png-one",
            "vision",
            "one_small_png",
            {
                "image_count": 1,
                "format": "png",
                "dimensions": "64x64",
                "encoded_bytes": _encoded_image_bytes(small_png),
                "malformed": False,
            },
            _vision_task(
                "cap-vision-small-png-one",
                image_uris=[small_png],
                state="one_small_png",
                format_name="png",
                dimensions="64x64",
            ),
            64,
        )
    )
    if model_id in DOCUMENTED_VISION_MODELS:
        small_variant_pngs = tuple(
            _synthetic_image_data_uri(
                64,
                64,
                "PNG",
                quadrant_order=order,
            )
            for order in QUADRANT_ORDERS
        )
        large_png = _synthetic_image_data_uri(512, 512, "PNG")
        dimension_png = _synthetic_image_data_uri(2048, 2048, "PNG")
        aspect_png = _synthetic_image_data_uri(4096, 512, "PNG")
        jpeg = _synthetic_image_data_uri(96, 96, "JPEG")
        webp = _synthetic_image_data_uri(96, 96, "WEBP")
        byte_anchors = (
            ("16kb", _synthetic_image_data_uri(74, 74, "PNG", True), 16_384),
            ("256kb", _synthetic_image_data_uri(295, 295, "PNG", True), 262_144),
            ("1mb", _synthetic_image_data_uri(591, 591, "PNG", True), 1_048_576),
            ("4mb", _synthetic_image_data_uri(1182, 1182, "PNG", True), 4_194_304),
        )
        vision_cases = (
            (
                "vision-two-small-png",
                "two_small_png",
                list(small_variant_pngs[:2]),
                "png",
                "64x64",
                False,
                0,
                None,
            ),
            (
                "vision-four-small-png",
                "four_small_png",
                list(small_variant_pngs[:4]),
                "png",
                "64x64",
                False,
                0,
                None,
            ),
            (
                "vision-eight-small-png",
                "eight_small_png",
                list(small_variant_pngs),
                "png",
                "64x64",
                False,
                0,
                None,
            ),
            (
                "vision-large-png-one",
                "one_large_png",
                [large_png],
                "png",
                "512x512",
                False,
                0,
                None,
            ),
            (
                "vision-2048-square-png",
                "one_2048_square_png",
                [dimension_png],
                "png",
                "2048x2048",
                False,
                0,
                None,
            ),
            (
                "vision-small-jpeg-one",
                "one_small_jpeg",
                [jpeg],
                "jpeg",
                "96x96",
                False,
                0,
                None,
            ),
            (
                "vision-small-webp-one",
                "one_small_webp",
                [webp],
                "webp",
                "96x96",
                False,
                0,
                None,
            ),
            (
                "vision-wide-aspect-png",
                "one_wide_aspect_png",
                [aspect_png],
                "png",
                "4096x512",
                False,
                0,
                None,
            ),
            (
                "vision-mixed-context-8k",
                "one_png_mixed_text_8k",
                [small_png],
                "png",
                "64x64",
                False,
                8_192,
                None,
            ),
            (
                "vision-concurrency-a",
                "concurrency_4_a",
                [small_png],
                "png",
                "64x64",
                False,
                0,
                f"{model_id}-vision-concurrency-4",
            ),
            (
                "vision-concurrency-b",
                "concurrency_4_b",
                [small_png],
                "png",
                "64x64",
                False,
                0,
                f"{model_id}-vision-concurrency-4",
            ),
            (
                "vision-concurrency-c",
                "concurrency_4_c",
                [small_png],
                "png",
                "64x64",
                False,
                0,
                f"{model_id}-vision-concurrency-4",
            ),
            (
                "vision-concurrency-d",
                "concurrency_4_d",
                [small_png],
                "png",
                "64x64",
                False,
                0,
                f"{model_id}-vision-concurrency-4",
            ),
            (
                "vision-malformed-data-uri",
                "malformed_data_uri",
                ["data:image/png;base64,not-valid-base64"],
                "png",
                "malformed",
                True,
                0,
                None,
            ),
        ) + tuple(
            (
                f"vision-byte-anchor-{label}",
                f"byte_anchor_{label}",
                [image_uri],
                "png",
                "synthetic_noise_square",
                False,
                0,
                None,
            )
            for label, image_uri, _target_bytes in byte_anchors
        )
        for (
            probe_id,
            state,
            image_uris,
            format_name,
            dimensions,
            malformed,
            mixed_text_tokens,
            concurrency_group,
        ) in vision_cases:
            quadrant_orders = (
                QUADRANT_ORDERS[: len(image_uris)]
                if state in {"two_small_png", "four_small_png", "eight_small_png"}
                else (DEFAULT_QUADRANT_ORDER,) * len(image_uris)
            )
            encoded_bytes = (
                None
                if malformed
                else sum(_encoded_image_bytes(image_uri) for image_uri in image_uris)
            )
            rows.append(
                (
                    probe_id,
                    "vision",
                    state,
                    {
                        "image_count": len(image_uris),
                        "format": format_name,
                        "dimensions": dimensions,
                        "encoded_bytes": encoded_bytes,
                        "malformed": malformed,
                        "mixed_text_tokens": mixed_text_tokens,
                        "concurrency_group": concurrency_group,
                        "target_concurrency": 4 if concurrency_group else 1,
                        "encoded_byte_target": next(
                            (
                                target_bytes
                                for label, _image_uri, target_bytes in byte_anchors
                                if state == f"byte_anchor_{label}"
                            ),
                            None,
                        ),
                    },
                    _vision_task(
                        f"cap-{probe_id}",
                        image_uris=image_uris,
                        state=state,
                        format_name=format_name,
                        dimensions=dimensions,
                        malformed=malformed,
                        mixed_text_tokens=mixed_text_tokens,
                        quadrant_orders=quadrant_orders,
                    ),
                    64,
                )
            )
    return rows


@lru_cache(maxsize=1)
def _documentation_contract() -> dict[str, Any]:
    artifacts: dict[str, str] = {}
    for relative in DOCUMENTATION_ARTIFACTS:
        path = REPOSITORY_ROOT / relative
        artifacts[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "api_reference_generated": API_DOC_GENERATED_DATE,
        "model_page_verified": MODEL_DOC_VERIFIED_DATE,
        "pricing_page_date": PRICING_DOC_DATE,
        "artifact_sha256": artifacts,
    }


@lru_cache(maxsize=1)
def _runner_source_contract() -> dict[str, str]:
    return {
        "do_benchmark/direct_capability.py": hashlib.sha256(
            DIRECT_CAPABILITY_SOURCE_PATH.read_bytes()
        ).hexdigest(),
        "do_benchmark/core.py": hashlib.sha256(
            CORE_SOURCE_PATH.read_bytes()
        ).hexdigest(),
        "do_benchmark/direct_aimd.py": hashlib.sha256(
            DIRECT_AIMD_SOURCE_PATH.read_bytes()
        ).hexdigest(),
        "scripts/run-digitalocean-direct-capability.py": hashlib.sha256(
            CAPABILITY_CLI_SOURCE_PATH.read_bytes()
        ).hexdigest(),
    }


def _scorer_contract(task: BenchmarkTask) -> dict[str, Any]:
    return {
        "version": SCORER_CONTRACT_VERSION,
        "expected": task.expected,
        "task_family": task.family,
        "requires_vision": task.requires_vision,
        "expected_tool_calls": task.metadata.get("expected_tool_calls"),
        "expected_tool_call_count": task.metadata.get("expected_tool_call_count"),
        "parallel_tool_calls_enabled": task.metadata.get("parallel_tool_calls_enabled"),
    }


def build_capability_cells(
    model_ids: Sequence[str], seed: int = 20260823
) -> list[CapabilityCell]:
    unknown = sorted(set(model_ids) - MODEL_BY_ID.keys())
    if unknown:
        raise ValueError(f"unknown DigitalOcean models: {', '.join(unknown)}")
    cells: list[CapabilityCell] = []
    for model_id in model_ids:
        for (
            probe_id,
            dimension,
            state,
            bindings,
            task,
            max_tokens,
        ) in _model_probe_tasks(model_id):
            payload = _render_payload(model_id, task, max_tokens)
            rendered_payload_sha256 = hashlib.sha256(
                canonical_json(payload).encode("utf-8")
            ).hexdigest()
            scorer_contract = _scorer_contract(task)
            scorer_contract_sha256 = hashlib.sha256(
                canonical_json(scorer_contract).encode("utf-8")
            ).hexdigest()
            model_contract = asdict(MODEL_BY_ID[model_id])
            model_contract_sha256 = hashlib.sha256(
                canonical_json(model_contract).encode("utf-8")
            ).hexdigest()
            documentation_contract = _documentation_contract()
            documentation_contract_sha256 = hashlib.sha256(
                canonical_json(documentation_contract).encode("utf-8")
            ).hexdigest()
            request_identity = {
                "schema": REQUEST_SCHEMA,
                "seed": seed,
                "model_id": model_id,
                "probe_id": probe_id,
                "dimension": dimension,
                "state": state,
                "bindings": bindings,
                "rendered_payload_sha256": rendered_payload_sha256,
                "scorer_contract": scorer_contract,
                "model_contract": model_contract,
                "documentation_contract": documentation_contract,
                "provider_send_expected": (
                    probe_id != "caching-explicit-option-documentation"
                ),
            }
            request_identity_sha256 = hashlib.sha256(
                canonical_json(request_identity).encode("utf-8")
            ).hexdigest()
            request_id = f"do-cap-request-{request_identity_sha256[:20]}"
            cells.append(
                CapabilityCell(
                    request_id=request_id,
                    model_id=model_id,
                    probe_id=probe_id,
                    dimension=dimension,
                    state=state,
                    design_role=(
                        "strength_two_covering_array"
                        if dimension == "pairwise_core"
                        else (
                            "explicit_three_way_interaction"
                            if dimension
                            == "parameter_interaction_temperature_top_p_output"
                            else "single_state_or_boundary"
                        )
                    ),
                    coverage_tags=(
                        ("capability_smoke",)
                        if probe_id == "capability-smoke"
                        else (dimension,)
                    ),
                    bindings=bindings,
                    task=task,
                    max_output_tokens=max_tokens,
                    rendered_payload_sha256=rendered_payload_sha256,
                    scorer_contract_sha256=scorer_contract_sha256,
                    model_contract_sha256=model_contract_sha256,
                    documentation_contract_sha256=documentation_contract_sha256,
                    request_identity_sha256=request_identity_sha256,
                    provider_send_expected=(
                        probe_id != "caching-explicit-option-documentation"
                    ),
                    local_terminal_status=(
                        "documented_unavailable"
                        if probe_id == "caching-explicit-option-documentation"
                        else None
                    ),
                )
            )
    random.Random(seed).shuffle(cells)
    return cells


def _render_payload(
    model_id: str, task: BenchmarkTask, max_tokens: int
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": task.messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
    }
    if "max_completion_tokens" not in task.parameters:
        payload["max_tokens"] = max_tokens
    if task.tools:
        payload["tools"] = task.tools
    if task.tool_choice is not None:
        payload["tool_choice"] = task.tool_choice
    if task.response_format is not None:
        payload["response_format"] = task.response_format
    payload.update(task.parameters)
    if payload.get("stream") is False:
        payload.pop("stream_options", None)
    return payload


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


async def _buffered_chat_completion(
    client: httpx.AsyncClient,
    *,
    api_base: str,
    api_key: str,
    model_id: str,
    task: BenchmarkTask,
    max_output_tokens: int,
) -> StreamResult:
    """Issue a clean non-streaming request without ``stream_options``."""

    payload = _render_payload(model_id, task, max_output_tokens)
    payload["stream"] = False
    payload.pop("stream_options", None)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    started = time.perf_counter()
    async with client.stream(
        "POST",
        f"{api_base.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
    ) as response:
        headers_at = time.perf_counter()
        selected_headers = {
            name: response.headers.get(name)
            for name in (
                "x-ratelimit-limit-requests",
                "x-ratelimit-remaining-requests",
                "x-ratelimit-reset-requests",
                "retry-after",
                "x-request-id",
                "cf-ray",
            )
        }
        body = await response.aread()
        ended = time.perf_counter()
        if response.status_code >= 400:
            raise ProviderHTTPError(
                response.status_code,
                body.decode("utf-8", errors="replace"),
                response.headers.get("retry-after"),
            )
    decoded = json.loads(body)
    raw_usage = decoded.get("usage") if isinstance(decoded, Mapping) else None
    usage = parse_token_usage(raw_usage)
    choices = decoded.get("choices") or []
    first_choice = choices[0] if choices else {}
    message = first_choice.get("message") or {}
    text = message.get("content") if isinstance(message.get("content"), str) else ""
    reasoning_value = message.get("reasoning_content") or message.get("reasoning")
    reasoning = reasoning_value if isinstance(reasoning_value, str) else ""
    tool_calls: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw_arguments = function.get("arguments") or ""
        try:
            arguments = (
                json.loads(raw_arguments)
                if isinstance(raw_arguments, str)
                else raw_arguments
            )
        except json.JSONDecodeError:
            arguments = {"_unparsed_sha256": _sha256(str(raw_arguments))}
        tool_calls.append(
            {"name": str(function.get("name") or ""), "arguments": arguments}
        )
    has_content = bool(text or reasoning or tool_calls)
    return StreamResult(
        status_code=response.status_code,
        response_headers=selected_headers,
        text=text,
        reasoning_text=reasoning,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=(
            str(first_choice.get("finish_reason"))
            if first_choice.get("finish_reason") is not None
            else None
        ),
        request_seconds=ended - started,
        headers_seconds=headers_at - started,
        # A buffered API does not expose token-level TTFT. This timestamp is
        # intentionally the full buffered response arrival proxy.
        ttft_seconds=(ended - started) if has_content else None,
        generation_seconds=0.0 if has_content else None,
        stream_seconds=ended - headers_at,
        event_count=1 if has_content else 0,
        first_event_kind="buffered_response" if has_content else None,
    )


def _conservative_cost(spec: ModelSpec, cell: CapabilityCell) -> tuple[float, int]:
    payload_bytes = len(
        canonical_json(
            _render_payload(cell.model_id, cell.task, cell.max_output_tokens)
        ).encode("utf-8")
    )
    planned = int(cell.task.metadata.get("planned_input_tokens") or 0)
    # One token per serialized UTF-8 byte is a tokenizer-independent upper
    # bound for these ASCII-authored payloads.  The extra 512 tokens cover chat
    # framing that the provider may add outside the serialized request body.
    # Do not use an average chars-per-token heuristic for a spend reservation.
    prompt_tokens = max(payload_bytes + 512, math.ceil(planned * 1.5))
    n_value = cell.task.parameters.get("n", 1)
    multiplicity = (
        int(n_value)
        if isinstance(n_value, int) and not isinstance(n_value, bool) and n_value > 0
        else 1
    )
    completion_ceiling = cell.max_output_tokens
    max_completion = cell.task.parameters.get("max_completion_tokens")
    if isinstance(max_completion, int) and not isinstance(max_completion, bool):
        completion_ceiling = max(completion_ceiling, max(0, max_completion))
    cost = (
        prompt_tokens * spec.input_usd_per_million
        + completion_ceiling * multiplicity * spec.output_usd_per_million
    ) / 1_000_000
    return cost, prompt_tokens


_PARAMETER_OR_CAPABILITY = re.compile(
    r"\b(?:temperature|top[_ -]?p|presence[_ -]?penalty|frequency[_ -]?penalty|"
    r"stream(?:ing)?|seed|stop(?: sequence)?|logprobs?|top[_ -]?logprobs|"
    r"response[_ -]?format|json[_ -]?schema|tools?|tool[_ -]?choice|"
    r"parallel[_ -]?tool[_ -]?calls|max[_ -]?(?:completion[_ -]?)?tokens|"
    r"logit[_ -]?bias|reasoning[_ -]?effort|user|image(?:[_ -]?url)?|vision|"
    r"base64|media[_ -]?type|content[_ -]?type)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_WORDING = re.compile(
    r"\b(?:unsupported|not supported|does not support|isn't supported|"
    r"unrecognized|unknown parameter|unknown field|not implemented)\b",
    re.IGNORECASE,
)
_VALIDATION_WORDING = re.compile(
    r"\b(?:invalid (?:value|parameter|field)|must be|should be|"
    r"expected (?:one of|a|an)|out of range|outside (?:the )?range|"
    r"less than|greater than|at most|at least|too many|too few|malformed)\b",
    re.IGNORECASE,
)


def _patterns_are_local(
    clause: str, left: re.Pattern[str], right: re.Pattern[str]
) -> bool:
    left_matches = tuple(left.finditer(clause))
    right_matches = tuple(right.finditer(clause))
    return any(
        min(abs(a.start() - b.end()), abs(b.start() - a.end())) <= 160
        for a in left_matches
        for b in right_matches
    )


def _provider_reason_clauses(body: str) -> tuple[str, ...]:
    """Extract only allowlisted error-reason fields for in-memory classification."""

    bounded = body[:65_536]
    try:
        decoded = json.loads(bounded)
    except (TypeError, json.JSONDecodeError):
        clause = " ".join(bounded.split())[:8_192]
        return (clause,) if clause else ()
    clauses: list[str] = []

    def append_reason(value: Any, *, location: Any = None) -> None:
        if not isinstance(value, str) or len(clauses) >= 64:
            return
        reason = " ".join(value.split())[:1_024]
        if not reason:
            return
        if isinstance(location, list):
            safe_location = ".".join(
                str(item)[:128]
                for item in location
                if isinstance(item, (str, int)) and not isinstance(item, bool)
            )
            if safe_location:
                reason = f"{safe_location}: {reason}"
        clauses.append(reason)

    if isinstance(decoded, Mapping):
        append_reason(decoded.get("message"))
        detail = decoded.get("detail")
        append_reason(detail)
        if isinstance(detail, list):
            for item in detail[:64]:
                if isinstance(item, Mapping):
                    append_reason(
                        item.get("message") or item.get("msg") or item.get("detail"),
                        location=item.get("loc"),
                    )
        error_value = decoded.get("error")
        append_reason(error_value)
        if isinstance(error_value, Mapping):
            append_reason(error_value.get("message"))
            append_reason(error_value.get("detail"))
        errors_value = decoded.get("errors")
        if isinstance(errors_value, list):
            for item in errors_value[:64]:
                if isinstance(item, Mapping):
                    append_reason(
                        item.get("message") or item.get("msg") or item.get("detail"),
                        location=item.get("loc"),
                    )
    return tuple(clauses)


def _sanitized_error_evidence(
    error: BaseException,
) -> tuple[str, str | None, str | None]:
    """Classify an error and return only an allowlisted category plus body hash."""

    status = getattr(error, "status_code", None)
    raw_body = getattr(error, "body", None)
    body = raw_body if isinstance(raw_body, str) else None
    reason_hash = (
        hashlib.sha256(body.encode("utf-8", errors="replace")).hexdigest()
        if body is not None
        else None
    )
    reason_category: str | None = None
    explicit_rejection = False
    if isinstance(status, int) and status in {400, 404, 405, 415, 422} and body:
        clauses = _provider_reason_clauses(body)
        unsupported_clause = next(
            (
                clause
                for clause in clauses
                if _patterns_are_local(
                    clause, _PARAMETER_OR_CAPABILITY, _UNSUPPORTED_WORDING
                )
            ),
            None,
        )
        validation_clause = next(
            (
                clause
                for clause in clauses
                if _patterns_are_local(
                    clause, _PARAMETER_OR_CAPABILITY, _VALIDATION_WORDING
                )
            ),
            None,
        )
        if unsupported_clause is not None:
            explicit_rejection = True
            reason_category = "explicit_unsupported_parameter_or_capability"
        elif validation_clause is not None:
            explicit_rejection = True
            reason_category = "explicit_parameter_or_payload_validation"
        else:
            reason_category = "unclassified_client_error_body"

    if status == 402:
        classification = "account_blocked_402"
    elif status == 429:
        classification = "rate_limited"
    elif explicit_rejection:
        classification = "rejected_or_unsupported"
    elif isinstance(status, int) and 400 <= status < 500:
        classification = "client_error_inconclusive"
    elif isinstance(status, int) and status >= 500:
        classification = "provider_error"
    elif isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        classification = "timed_out"
    else:
        classification = "transport_error"
    return classification, reason_category, reason_hash


def _classification(error: BaseException) -> str:
    return _sanitized_error_evidence(error)[0]


def _coverage_conclusive(classification: str) -> bool:
    return classification in {
        "accepted",
        "rejected_or_unsupported",
        "documented_unavailable",
    }


def _score_capability_result(
    task: BenchmarkTask, result: StreamResult
) -> dict[str, Any]:
    expected_calls = task.metadata.get("expected_tool_calls")
    if not isinstance(expected_calls, list):
        return score_result(task, result)

    def normalized(call: Mapping[str, Any]) -> tuple[str, str]:
        return (
            str(call.get("name") or ""),
            canonical_json(call.get("arguments") or {}),
        )

    expected = sorted(normalized(call) for call in expected_calls)
    observed = sorted(normalized(call) for call in result.tool_calls)
    passed = observed == expected
    parallel_enabled = bool(task.metadata.get("parallel_tool_calls_enabled"))
    return {
        "quality_score": float(passed),
        "score_kind": (
            "parallel_tools_exact_all"
            if parallel_enabled
            else "parallel_disabled_single_call_exact"
        ),
        "score_detail": {
            "expected_call_count": len(expected),
            "observed_call_count": len(observed),
            "all_names_and_arguments_match": passed,
            "parallel_tool_calls_enabled": parallel_enabled,
        },
    }


class DirectCapabilityCampaign:
    def __init__(self, config: CapabilityConfig) -> None:
        config.validate()
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cells = build_capability_cells(config.model_ids, config.seed)
        self.plan_text = "".join(
            json.dumps(cell.sanitized_plan_row(), sort_keys=True) + "\n"
            for cell in self.cells
        )
        self.plan_sha256 = hashlib.sha256(self.plan_text.encode("utf-8")).hexdigest()
        self.documentation_contract = _documentation_contract()
        self.campaign_identity = {
            "config": config.identity_payload(),
            "plan_sha256": self.plan_sha256,
            "documentation_contract": self.documentation_contract,
            "scorer_contract_version": SCORER_CONTRACT_VERSION,
            "reservation_contract_version": RESERVATION_CONTRACT_VERSION,
            "runner_source_sha256": _runner_source_contract(),
            "model_contracts": [asdict(MODEL_BY_ID[item]) for item in config.model_ids],
        }
        self.campaign_identity_sha256 = hashlib.sha256(
            canonical_json(self.campaign_identity).encode("utf-8")
        ).hexdigest()
        required_group_size = max(
            (
                int(cell.bindings.get("target_concurrency") or 1)
                for cell in self.cells
                if cell.bindings.get("concurrency_group")
            ),
            default=1,
        )
        if min(config.max_workers, config.per_model_concurrency) < required_group_size:
            raise ValueError(
                "configured concurrency is below the largest planned interaction: "
                f"need at least {required_group_size} global and per-model slots"
            )
        self.planned_reservation_usd = sum(
            _conservative_cost(MODEL_BY_ID[cell.model_id], cell)[0]
            for cell in self.cells
            if cell.provider_send_expected
        )
        if (
            config.prior_cost_usd + self.planned_reservation_usd
            > config.max_cost_usd + 1e-12
        ):
            raise ValueError(
                "full capability plan cannot fit under the cumulative cost cap: "
                f"prior=${config.prior_cost_usd:.6f}, "
                f"plan_reservation=${self.planned_reservation_usd:.6f}, "
                f"cap=${config.max_cost_usd:.6f}"
            )
        self.campaign_id = f"do-capability-{self.campaign_identity_sha256[:20]}"
        # ``direct_report.load_breadth_directory`` consumes plan.jsonl plus
        # records.jsonl, so the capability lane writes that contract directly.
        self.requests_path = self.output_dir / "records.jsonl"
        self.reservations_path = self.output_dir / "reservations.jsonl"
        self.execution_lease_path = self.output_dir / ".execution.lock"
        self.global_slots = asyncio.Semaphore(config.max_workers)
        self.model_slots = {
            model_id: asyncio.Semaphore(config.per_model_concurrency)
            for model_id in config.model_ids
        }
        concurrency_groups = {
            str(cell.bindings["concurrency_group"])
            for cell in self.cells
            if cell.bindings.get("concurrency_group")
        }
        self.concurrency_events = {
            group: asyncio.Event() for group in concurrency_groups
        }
        self.concurrency_waiters = {group: 0 for group in concurrency_groups}
        self.concurrency_group_sizes = {
            group: max(
                int(cell.bindings.get("target_concurrency") or 1)
                for cell in self.cells
                if cell.bindings.get("concurrency_group") == group
            )
            for group in concurrency_groups
        }
        self._concurrency_gate_lock = asyncio.Lock()
        self._active_send_lock = asyncio.Lock()
        self._active_sends_by_model = {model_id: 0 for model_id in config.model_ids}
        # Plan construction takes the same short process lease used for live
        # execution.  Live execution reacquires it and reloads every journal
        # immediately before credentials can reach a provider call.
        with OutputDirectoryLease(self.execution_lease_path):
            self._write_or_validate_artifacts()
            self._reload_runtime_state()

    @staticmethod
    def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        if not path.exists():
            return rows
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"torn request journal at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise CapabilityPreflightError(
                        f"capability journal {path.name}:{line_number} is not an object"
                    )
                request_id = row.get("request_id")
                if not isinstance(request_id, str) or not request_id:
                    raise CapabilityPreflightError(
                        f"capability journal {path.name}:{line_number} lacks request_id"
                    )
                if request_id in rows:
                    raise CapabilityPreflightError(
                        f"duplicate request_id {request_id!r} in {path.name}"
                    )
                rows[request_id] = row
        return rows

    @staticmethod
    def _require_fields(
        row: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
    ) -> None:
        for key, value in expected.items():
            observed = row.get(key)
            if canonical_json(observed) != canonical_json(value):
                raise CapabilityPreflightError(
                    f"resume {label} identity mismatch for {key}: "
                    f"expected {value!r}, observed {observed!r}"
                )

    @staticmethod
    def _require_cost(
        observed: Any, expected: float, *, label: str, allow_higher: bool = False
    ) -> None:
        try:
            value = float(observed)
        except (TypeError, ValueError) as error:
            raise CapabilityPreflightError(f"resume {label} is not numeric") from error
        if not math.isfinite(value) or value < 0:
            raise CapabilityPreflightError(
                f"resume {label} is not finite and nonnegative"
            )
        if allow_higher:
            valid = value + 1e-12 >= expected
        else:
            valid = math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)
        if not valid:
            raise CapabilityPreflightError(
                f"resume {label} mismatch: expected "
                f"{'at least ' if allow_higher else ''}{expected!r}, observed {value!r}"
            )

    def _validate_runtime_state(
        self,
        request_rows: Mapping[str, Mapping[str, Any]],
        reservations: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Fail closed before trusting a terminal row or spend reservation."""

        cell_by_id = {cell.request_id: cell for cell in self.cells}
        for request_id, reservation in reservations.items():
            cell = cell_by_id.get(request_id)
            if cell is None:
                raise CapabilityPreflightError(
                    "resume reservation is not part of this capability plan"
                )
            if not cell.provider_send_expected:
                raise CapabilityPreflightError(
                    "resume reservation targets a documented no-send capability cell"
                )
            reserved_cost, reserved_tokens = _conservative_cost(
                MODEL_BY_ID[cell.model_id], cell
            )
            self._require_fields(
                reservation,
                {
                    "schema_version": RESERVATION_SCHEMA,
                    "campaign_id": self.campaign_id,
                    "request_id": request_id,
                    "epoch_id": "capability-envelope",
                    "model_id": cell.model_id,
                    "shape": cell.dimension,
                    "reserved_prompt_tokens": reserved_tokens,
                    "max_output_tokens": cell.max_output_tokens,
                },
                label="reservation",
            )
            self._require_cost(
                reservation.get("reserved_cost_usd"),
                reserved_cost,
                label="reservation cost",
            )

        for request_id, row in request_rows.items():
            cell = cell_by_id.get(request_id)
            if cell is None:
                raise CapabilityPreflightError(
                    "resume request is not part of this capability plan"
                )
            self._require_fields(
                row,
                self._base_row(cell),
                label="request",
            )
            attempted = row.get("provider_send_attempted") is True
            reservation = reservations.get(request_id)
            if attempted and reservation is None:
                raise CapabilityPreflightError(
                    "provider-attempted resume request lacks its durable reservation"
                )
            reserved_cost, reserved_tokens = _conservative_cost(
                MODEL_BY_ID[cell.model_id], cell
            )
            if reservation is not None:
                self._require_fields(
                    row,
                    {
                        "worst_case_reserved_cost_usd": reserved_cost,
                        "reserved_prompt_tokens": reserved_tokens,
                    },
                    label="reserved request",
                )
            usage = parse_token_usage(row.get("usage"))
            complete_usage = (
                usage.get("prompt_tokens", 0) > 0
                and usage.get("completion_tokens", 0) > 0
            )
            if attempted and complete_usage:
                actual_cost = (
                    usage["prompt_tokens"]
                    * MODEL_BY_ID[cell.model_id].input_usd_per_million
                    + usage["completion_tokens"]
                    * MODEL_BY_ID[cell.model_id].output_usd_per_million
                ) / 1_000_000
                self._require_cost(
                    row.get("accounted_cost_usd"),
                    actual_cost,
                    label="complete-usage request cost",
                    allow_higher=True,
                )
            elif attempted:
                self._require_cost(
                    row.get("accounted_cost_usd"),
                    reserved_cost,
                    label="incomplete-usage request cost",
                    allow_higher=True,
                )

    def _reload_runtime_state(self) -> None:
        """Reload and reconcile journals while holding the process lease."""

        request_rows = self._read_rows(self.requests_path)
        reservations = self._read_rows(self.reservations_path)
        self._validate_runtime_state(request_rows, reservations)
        self.requests_journal = JsonlJournal(self.requests_path)
        self.request_rows = request_rows
        self.budget = BudgetLedger(
            path=self.reservations_path,
            max_cost_usd=self.config.max_cost_usd,
            prior_cost_usd=self.config.prior_cost_usd,
            terminal_rows=self.request_rows,
        )
        if set(self.budget.reservations) != set(reservations):
            raise CapabilityPreflightError(
                "reservation journal changed during capability state reload"
            )
        if self.budget.exposure_usd > self.config.max_cost_usd + 1e-12:
            raise CapabilityPreflightError(
                "persisted capability exposure exceeds the cumulative cost cap"
            )
        self.account_blocked_402 = any(
            row.get("http_status") == 402
            or row.get("status") == "account_blocked_402"
            or row.get("coverage_classification") == "account_blocked_402"
            for row in self.request_rows.values()
        )

    def _write_or_validate_artifacts(self) -> None:
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("campaign_id") != self.campaign_id:
                raise RuntimeError(
                    "output directory belongs to another capability campaign"
                )
            if (
                existing.get("campaign_identity_sha256")
                != self.campaign_identity_sha256
            ):
                raise RuntimeError("capability campaign identity contract changed")
            if existing.get("plan_sha256") != self.plan_sha256:
                raise RuntimeError("capability manifest plan hash changed")
            if existing.get("documentation_contract") != self.documentation_contract:
                raise RuntimeError("capability documentation contract changed")
            if (
                existing.get("reservation_contract_version")
                != RESERVATION_CONTRACT_VERSION
            ):
                raise RuntimeError("capability reservation contract changed")
            if existing.get("runner_source_sha256") != _runner_source_contract():
                raise RuntimeError("capability runner source contract changed")
            self._require_cost(
                existing.get("planned_worst_case_reservation_usd"),
                self.planned_reservation_usd,
                label="manifest plan reservation",
            )
            plan_path = self.output_dir / "plan.jsonl"
            if not plan_path.is_file():
                raise RuntimeError(
                    "capability manifest exists but plan.jsonl is missing"
                )
            if hashlib.sha256(plan_path.read_bytes()).hexdigest() != self.plan_sha256:
                raise RuntimeError(
                    "capability plan.jsonl does not match its exact hash"
                )
            planned_rows = sum(
                bool(line.strip())
                for line in plan_path.read_text(encoding="utf-8").splitlines()
            )
            if planned_rows != len(self.cells):
                raise RuntimeError(
                    "existing capability plan does not match the current campaign design"
                )
            return
        pairwise_rows = list(PAIRWISE_ROWS)
        model_order = list(self.config.model_ids)
        random.Random(self.config.seed).shuffle(model_order)
        manifest = {
            **self.config.identity_payload(),
            "campaign_id": self.campaign_id,
            "campaign_identity_sha256": self.campaign_identity_sha256,
            "plan_sha256": self.plan_sha256,
            "documentation_contract": self.documentation_contract,
            "scorer_contract_version": SCORER_CONTRACT_VERSION,
            "reservation_contract_version": RESERVATION_CONTRACT_VERSION,
            "runner_source_sha256": _runner_source_contract(),
            "created_at": utc_now(),
            "planned_requests": len(self.cells),
            "planned_provider_calls": sum(
                cell.provider_send_expected for cell in self.cells
            ),
            "planned_worst_case_reservation_usd": self.planned_reservation_usd,
            "planned_cells_per_model": {
                model_id: sum(cell.model_id == model_id for cell in self.cells)
                for model_id in model_order
            },
            "model_order": model_order,
            "model_specs": [asdict(MODEL_BY_ID[item]) for item in model_order],
            "numeric_anchors": {
                "temperature": [-0.01, 0.0, 0.5, 1.0, 1.5, 2.0, 2.01],
                "top_p": [-0.01, 0.0, 0.25, 0.5, 0.75, 1.0, 1.01],
                "frequency_penalty": [-2.01, -2.0, -1.0, 0.0, 1.0, 2.0, 2.01],
                "presence_penalty": [-2.01, -2.0, -1.0, 0.0, 1.0, 2.0, 2.01],
                "top_logprobs": [-1, 0, 5, 10, 15, 20, 21],
                "logit_bias": [-101, -100, -50, 0, 50, 100, 101],
                "n": [0, 1, 4, 8, 12, 16, 17],
                "max_completion_tokens": [1, 64, 256, 1_024, 4_096, 4_097],
            },
            "max_output_token_anchors": {
                model_id: {"value": value, "source": source}
                for model_id, (value, source) in MAX_OUTPUT_TOKEN_ANCHORS.items()
                if model_id in self.config.model_ids
            },
            "pairwise_factors": {
                key: list(value) for key, value in PAIRWISE_FACTORS.items()
            },
            "pairwise_rows": pairwise_rows,
            "pairwise_strength": 2,
            "pairwise_factor_count": len(PAIRWISE_FACTORS),
            "pairwise_completeness": (
                "Every state pair across all listed factors is present in at least one "
                "rendered request; completeness is asserted independently in tests."
            ),
            "explicit_temperature_top_p_output_rows": list(THREE_WAY_ROWS),
            "three_way_scope": (
                "Six predeclared corner/interior temperature × top_p × max_tokens rows; "
                "this is an explicit interaction screen, not a strength-three full factorial."
            ),
            "seed_determinism_repeats": 2,
            "automatic_cache_option_contract": (
                "The frozen documentation describes automatic prompt caching but no "
                "explicit chat-completions request option. Each endpoint therefore gets "
                "an explicit local documented_unavailable terminal cell, with no send."
            ),
            "vision_contract": {
                "documented_models": sorted(DOCUMENTED_VISION_MODELS),
                "image_counts": [1, 2, 4, 8],
                "formats": ["png", "jpeg", "webp"],
                "dimensions": ["64x64", "512x512", "2048x2048", "4096x512"],
                "encoded_byte_targets": [16_384, 262_144, 1_048_576, 4_194_304],
                "mixed_text_tokens": [0, 8_192],
                "concurrency": [1, 4],
            },
            "tool_envelope_contract": {
                "documented_models": sorted(DOCUMENTED_TOOL_MODELS),
                "tool_counts": list(TOOL_COUNT_ANCHORS),
                "schema_bytes": list(TOOL_SCHEMA_BYTE_ANCHORS),
                "nesting_depths": list(TOOL_NESTING_DEPTH_ANCHORS),
                "argument_payload_bytes": list(TOOL_ARGUMENT_BYTE_ANCHORS),
                "required_optional_modes": list(TOOL_REQUIRED_OPTIONAL_MODES),
                "malformed_cases": list(TOOL_MALFORMED_CASES),
                "schema_byte_scope": (
                    "UTF-8 bytes of canonical minified JSON for the complete tools array"
                ),
            },
            "structured_output_contract": {
                "documented_models": sorted(DOCUMENTED_STRUCTURED_OUTPUT_MODELS),
                "isolated_modes": ["json_object", "json_schema"],
            },
            "scope_exclusions": {
                "adaptive_tool_over_limit_followups": (
                    "not part of this fixed plan; report as untested rather than inferred"
                ),
                "conditional_retry_backoff_followups": (
                    "a separate recovery experiment, not part of this no-retry "
                    "capability measurement lane"
                ),
            },
            "coverage_contract": (
                "Every planned cell receives a terminal row. HTTP acceptance and only "
                "allowlisted explicit provider parameter/capability validation reasons are "
                "conclusive; generic 4xx, timeouts, 429, 5xx, budget/deadline skips, and "
                "interrupted reservations are inconclusive."
            ),
            "output_limit_contract": (
                "These are request-acceptance probes with short expected responses, not "
                "claims that the model realized the requested maximum output length."
            ),
            "buffered_timing_note": (
                "For stream=false the first-content timestamp is the complete buffered "
                "response arrival, so it is not token-level TTFT."
            ),
            "documentation_freeze": self.documentation_contract,
            "sanitization": (
                "No credentials, prompts, outputs, response bodies, or raw headers are "
                "persisted. Content and provider request IDs are SHA-256 fingerprints only."
            ),
            "source_role": "benchmark-authored synthetic parameter probes",
            "rights_posture": "repository-authored redistributable test definitions",
        }
        (self.output_dir / "plan.jsonl").write_text(
            self.plan_text, encoding="utf-8", newline="\n"
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _deadline_reached(self) -> bool:
        cutoff = self.config.stop_launch_at
        return cutoff is not None and datetime.now(timezone.utc) >= cutoff.astimezone(
            timezone.utc
        )

    async def _append(self, row: dict[str, Any]) -> None:
        request_id = str(row["request_id"])
        if request_id in self.request_rows:
            return
        await self.requests_journal.append(row)
        self.request_rows[request_id] = row
        await self.budget.settle(request_id, row)

    async def _wait_for_concurrency_group(self, group: str) -> None:
        async with self._concurrency_gate_lock:
            self.concurrency_waiters[group] += 1
            if self.concurrency_waiters[group] >= self.concurrency_group_sizes[group]:
                self.concurrency_events[group].set()
        await asyncio.wait_for(
            self.concurrency_events[group].wait(),
            timeout=min(10.0, self.config.request_timeout_seconds),
        )

    def _base_row(self, cell: CapabilityCell) -> dict[str, Any]:
        payload = _render_payload(cell.model_id, cell.task, cell.max_output_tokens)
        workload = workload_for_cell(cell)
        return {
            "schema_version": REQUEST_SCHEMA,
            "campaign_id": self.campaign_id,
            "campaign_identity_sha256": self.campaign_identity_sha256,
            "campaign_plan_sha256": self.plan_sha256,
            "request_id": cell.request_id,
            "cell_id": cell.request_id,
            "provider": "digitalocean-serverless-inference",
            "model_id": cell.model_id,
            "probe_id": cell.probe_id,
            "task_id": cell.task.task_id,
            "task_family": workload,
            "workload_id": workload,
            "shape": "capability_envelope",
            "phase": cell.design_role,
            "dimension": cell.dimension,
            "state": cell.state,
            "design_role": cell.design_role,
            "coverage_tags": list(cell.coverage_tags),
            "bindings": cell.bindings,
            "requested_max_output_tokens": cell.max_output_tokens,
            "request_payload_sha256": _sha256(payload),
            "rendered_payload_sha256": cell.rendered_payload_sha256,
            "scorer_contract_sha256": cell.scorer_contract_sha256,
            "model_contract_sha256": cell.model_contract_sha256,
            "documentation_contract_sha256": cell.documentation_contract_sha256,
            "request_identity_sha256": cell.request_identity_sha256,
            "request_payload_bytes": len(canonical_json(payload).encode("utf-8")),
        }

    async def _append_local_terminal(self, cell: CapabilityCell) -> None:
        now = utc_now()
        classification = cell.local_terminal_status or "documented_unavailable"
        row = {
            **self._base_row(cell),
            "provider_send_attempted": False,
            "started_at": now,
            "ended_at": now,
            "status": classification,
            "coverage_classification": classification,
            "coverage_conclusive": True,
            "documentation_status": cell.bindings.get("documentation_status"),
            "http_status": None,
            "error_type": None,
            "transport_success": False,
            "scientific_success": False,
            "functional_valid": None,
            "usage": {},
            "prompt_usage_present": False,
            "completion_usage_present": False,
            "usage_complete_for_settlement": False,
            "timing": {"request_seconds": 0.0, "ttft_seconds": None},
            "quality_score": None,
            "score_kind": "documentation_state",
            "worst_case_reserved_cost_usd": 0.0,
            "reserved_prompt_tokens": 0,
            "estimated_cost_usd": 0.0,
            "accounted_cost_usd": 0.0,
        }
        await self._append(row)

    async def _append_unlaunched(self, cell: CapabilityCell, reason: str) -> None:
        reservation = self.budget.reservations.get(cell.request_id)
        reserved_cost = (
            float(reservation.get("reserved_cost_usd") or 0.0) if reservation else 0.0
        )
        reserved_tokens = (
            int(reservation.get("reserved_prompt_tokens") or 0) if reservation else 0
        )
        now = utc_now()
        possibly_sent = reason == "unknown_prior_reservation"
        row = {
            **self._base_row(cell),
            "provider_send_attempted": possibly_sent,
            "started_at": now,
            "ended_at": now,
            "status": reason,
            "coverage_classification": reason,
            "coverage_conclusive": False,
            "http_status": None,
            "error_type": None,
            "transport_success": False,
            "scientific_success": False,
            "functional_valid": None,
            "usage": {},
            "prompt_usage_present": False,
            "completion_usage_present": False,
            "usage_complete_for_settlement": False,
            "timing": {"request_seconds": 0.0, "ttft_seconds": None},
            "quality_score": 0.0,
            "score_kind": str(cell.task.expected.get("kind") or "unknown"),
            "worst_case_reserved_cost_usd": reserved_cost,
            "reserved_prompt_tokens": reserved_tokens,
            "estimated_cost_usd": None,
            "accounted_cost_usd": reserved_cost if possibly_sent else 0.0,
        }
        await self._append(row)

    def _execution_cells(self) -> list[CapabilityCell]:
        """Place each small concurrency interaction before the randomized tail."""

        grouped: dict[str, list[CapabilityCell]] = {}
        regular: list[CapabilityCell] = []
        for cell in self.cells:
            group = cell.bindings.get("concurrency_group")
            if group:
                grouped.setdefault(str(group), []).append(cell)
            else:
                regular.append(cell)
        return [
            cell
            for group in sorted(grouped)
            for cell in sorted(grouped[group], key=lambda item: item.request_id)
        ] + regular

    async def _run_cell(self, executor: RequestExecutor, cell: CapabilityCell) -> None:
        if cell.request_id in self.request_rows:
            return
        if not cell.provider_send_expected:
            await self._append_local_terminal(cell)
            return
        if cell.request_id in self.budget.reservations:
            await self._append_unlaunched(cell, "unknown_prior_reservation")
            return
        if self._deadline_reached():
            await self._append_unlaunched(cell, "skipped_deadline")
            return
        if self.account_blocked_402:
            await self._append_unlaunched(cell, "skipped_http_402_latch")
            return

        async with self.global_slots, self.model_slots[cell.model_id]:
            if self._deadline_reached():
                await self._append_unlaunched(cell, "skipped_deadline")
                return
            if self.account_blocked_402:
                await self._append_unlaunched(cell, "skipped_http_402_latch")
                return
            reserved_cost, reserved_tokens = _conservative_cost(
                MODEL_BY_ID[cell.model_id], cell
            )
            reserved = await self.budget.reserve(
                campaign_id=self.campaign_id,
                request_id=cell.request_id,
                epoch_id="capability-envelope",
                model_id=cell.model_id,
                shape=cell.dimension,
                reserved_cost_usd=reserved_cost,
                reserved_prompt_tokens=reserved_tokens,
                max_output_tokens=cell.max_output_tokens,
            )
            if not reserved:
                reason = (
                    "unknown_prior_reservation"
                    if cell.request_id in self.budget.reservations
                    else "skipped_budget_cap"
                )
                await self._append_unlaunched(cell, reason)
                return

            concurrency_group = cell.bindings.get("concurrency_group")
            if concurrency_group:
                try:
                    await self._wait_for_concurrency_group(str(concurrency_group))
                except asyncio.TimeoutError:
                    await self._append_unlaunched(
                        cell, "concurrency_barrier_inconclusive"
                    )
                    return

            started_at = utc_now()
            started = time.perf_counter()
            async with self._active_send_lock:
                self._active_sends_by_model[cell.model_id] += 1
                observed_concurrency = self._active_sends_by_model[cell.model_id]
            try:
                result = await asyncio.wait_for(
                    executor(cell.model_id, cell.task, cell.max_output_tokens),
                    timeout=self.config.request_timeout_seconds,
                )
                ended_at = utc_now()
                quality = _score_capability_result(cell.task, result)
                usage = parse_token_usage(result.usage)
                prompt_usage_present = "prompt_tokens" in usage
                completion_usage_present = "completion_tokens" in usage
                usage_complete_for_settlement = (
                    prompt_usage_present
                    and completion_usage_present
                    and usage["prompt_tokens"] > 0
                    and usage["completion_tokens"] > 0
                )
                actual_cost = (
                    usage.get("prompt_tokens", 0)
                    * MODEL_BY_ID[cell.model_id].input_usd_per_million
                    + usage.get("completion_tokens", 0)
                    * MODEL_BY_ID[cell.model_id].output_usd_per_million
                ) / 1_000_000
                response_fingerprint = {
                    "text": result.text,
                    "reasoning": result.reasoning_text,
                    "tool_calls": result.tool_calls,
                }
                row = {
                    **self._base_row(cell),
                    "provider_send_attempted": True,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "status": "accepted",
                    "coverage_classification": "accepted",
                    "coverage_conclusive": True,
                    "http_status": result.status_code,
                    "transport_success": True,
                    "scientific_success": usage_complete_for_settlement,
                    "functional_valid": float(quality["quality_score"]) >= 0.999999,
                    "finish_reason": result.finish_reason,
                    "usage": usage,
                    "usage_reported": bool(usage),
                    "prompt_usage_present": prompt_usage_present,
                    "completion_usage_present": completion_usage_present,
                    "usage_complete_for_settlement": usage_complete_for_settlement,
                    "timing": {
                        "request_seconds": result.request_seconds,
                        "headers_seconds": result.headers_seconds,
                        "ttft_seconds": result.ttft_seconds,
                        "generation_seconds": result.generation_seconds,
                        "stream_seconds": result.stream_seconds,
                    },
                    "stream_observation": {
                        "event_count": result.event_count,
                        "first_event_kind": result.first_event_kind,
                    },
                    "header_signals": sanitized_header_signals(result.response_headers),
                    "response_sha256": _sha256(response_fingerprint),
                    "response_text_bytes": len(result.text.encode("utf-8")),
                    "reasoning_bytes": len(result.reasoning_text.encode("utf-8")),
                    "tool_call_count": len(result.tool_calls),
                    "observed_model_concurrency_at_send": observed_concurrency,
                    "quality_score": float(quality["quality_score"]),
                    "score_kind": str(quality["score_kind"]),
                    "worst_case_reserved_cost_usd": reserved_cost,
                    "reserved_prompt_tokens": reserved_tokens,
                    "estimated_cost_usd": (
                        actual_cost if usage_complete_for_settlement else None
                    ),
                    "accounted_cost_usd": (
                        actual_cost if usage_complete_for_settlement else reserved_cost
                    ),
                }
            except BaseException as error:
                classification, reason_category, reason_sha256 = (
                    _sanitized_error_evidence(error)
                )
                if classification == "account_blocked_402":
                    self.account_blocked_402 = True
                status_code = getattr(error, "status_code", None)
                retry_after = getattr(error, "retry_after", None)
                try:
                    retry_after_seconds = (
                        float(retry_after) if retry_after is not None else None
                    )
                except (TypeError, ValueError):
                    retry_after_seconds = None
                row = {
                    **self._base_row(cell),
                    "provider_send_attempted": True,
                    "started_at": started_at,
                    "ended_at": utc_now(),
                    "status": classification,
                    "coverage_classification": classification,
                    "coverage_conclusive": _coverage_conclusive(classification),
                    "http_status": status_code
                    if isinstance(status_code, int)
                    else None,
                    "error_type": type(error).__name__,
                    "provider_reason_category": reason_category,
                    "provider_reason_sha256": reason_sha256,
                    "transport_success": False,
                    "scientific_success": False,
                    "functional_valid": False,
                    "retry_after_seconds": retry_after_seconds,
                    "usage": {},
                    "prompt_usage_present": False,
                    "completion_usage_present": False,
                    "usage_complete_for_settlement": False,
                    "timing": {
                        "request_seconds": time.perf_counter() - started,
                        "ttft_seconds": None,
                    },
                    "quality_score": 0.0,
                    "score_kind": str(cell.task.expected.get("kind") or "unknown"),
                    "worst_case_reserved_cost_usd": reserved_cost,
                    "reserved_prompt_tokens": reserved_tokens,
                    "estimated_cost_usd": None,
                    # A failed or interrupted call may be partially billable.
                    "accounted_cost_usd": reserved_cost,
                    "observed_model_concurrency_at_send": observed_concurrency,
                }
            finally:
                async with self._active_send_lock:
                    self._active_sends_by_model[cell.model_id] -= 1
            await self._append(row)

    def _summary(
        self, started_at: str, runner_errors: Sequence[BaseException]
    ) -> dict[str, Any]:
        models: dict[str, Any] = {}
        cell_by_id = {cell.request_id: cell for cell in self.cells}
        for model_id in self.config.model_ids:
            planned = [cell for cell in self.cells if cell.model_id == model_id]
            rows = [
                row
                for request_id, row in self.request_rows.items()
                if request_id in cell_by_id
                and cell_by_id[request_id].model_id == model_id
            ]
            outcome_counts: dict[str, int] = {}
            dimension_counts: dict[str, dict[str, int]] = {}
            for row in rows:
                outcome = str(
                    row.get("coverage_classification") or row.get("status") or "unknown"
                )
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
                dimension = str(row.get("dimension") or "unknown")
                bucket = dimension_counts.setdefault(
                    dimension, {"planned": 0, "terminal": 0, "conclusive": 0}
                )
                bucket["terminal"] += 1
                bucket["conclusive"] += int(bool(row.get("coverage_conclusive")))
            for cell in planned:
                bucket = dimension_counts.setdefault(
                    cell.dimension, {"planned": 0, "terminal": 0, "conclusive": 0}
                )
                bucket["planned"] += 1
            models[model_id] = {
                "planned_cells": len(planned),
                "terminal_rows": len(rows),
                "provider_attempts": sum(
                    bool(row.get("provider_send_attempted")) for row in rows
                ),
                "conclusive_cells": sum(
                    bool(row.get("coverage_conclusive")) for row in rows
                ),
                "terminal_coverage_complete": len(rows) == len(planned),
                "conclusive_coverage_complete": (
                    len(rows) == len(planned)
                    and all(bool(row.get("coverage_conclusive")) for row in rows)
                ),
                "outcomes": dict(sorted(outcome_counts.items())),
                "dimensions": dict(sorted(dimension_counts.items())),
            }
        return {
            "schema_version": SUMMARY_SCHEMA,
            "campaign_id": self.campaign_id,
            "started_at": started_at,
            "ended_at": utc_now(),
            "planned_requests": len(self.cells),
            "terminal_rows": len(set(self.request_rows).intersection(cell_by_id)),
            "provider_attempts": sum(
                bool(row.get("provider_send_attempted"))
                for request_id, row in self.request_rows.items()
                if request_id in cell_by_id
            ),
            "terminal_coverage_complete": all(
                cell.request_id in self.request_rows for cell in self.cells
            ),
            "conclusive_cells": sum(
                bool(row.get("coverage_conclusive"))
                for request_id, row in self.request_rows.items()
                if request_id in cell_by_id
            ),
            "max_cost_usd": self.config.max_cost_usd,
            "prior_cost_usd": self.config.prior_cost_usd,
            "conservative_exposure_usd": self.budget.exposure_usd,
            "http_402_latched": self.account_blocked_402,
            "internal_runner_error_types": sorted(
                {type(error).__name__ for error in runner_errors}
            ),
            "models": models,
        }

    async def _run_locked(self, executor: RequestExecutor) -> dict[str, Any]:
        started_at = utc_now()
        outcomes = await asyncio.gather(
            *(self._run_cell(executor, cell) for cell in self._execution_cells()),
            return_exceptions=True,
        )
        runner_errors = [item for item in outcomes if isinstance(item, BaseException)]
        summary = self._summary(started_at, runner_errors)
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary

    async def _run_with_executor(self, executor: RequestExecutor) -> dict[str, Any]:
        # Reload spend and terminal state only after taking the cross-process
        # lease, then hold it through the last summary write.  A stale campaign
        # object can therefore never replay work completed by another process.
        with OutputDirectoryLease(self.execution_lease_path):
            self._reload_runtime_state()
            self.concurrency_events = {
                group: asyncio.Event() for group in self.concurrency_group_sizes
            }
            self.concurrency_waiters = {
                group: 0 for group in self.concurrency_group_sizes
            }
            return await self._run_locked(executor)

    async def run(self, executor: RequestExecutor | None = None) -> dict[str, Any]:
        if executor is not None:
            return await self._run_with_executor(executor)

        # Credentials are loaded only after construction completed the entire
        # offline plan and cumulative-budget preflight.
        credentials = digitalocean_credentials()
        limits = httpx.Limits(
            max_connections=self.config.max_workers,
            max_keepalive_connections=self.config.max_workers,
        )
        timeout = httpx.Timeout(
            self.config.request_timeout_seconds,
            connect=min(30.0, self.config.request_timeout_seconds),
            read=self.config.request_timeout_seconds,
            write=min(30.0, self.config.request_timeout_seconds),
            pool=self.config.request_timeout_seconds,
        )
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

            async def live_executor(
                model_id: str, task: BenchmarkTask, max_output_tokens: int
            ) -> StreamResult:
                if task.parameters.get("stream") is False:
                    return await _buffered_chat_completion(
                        client,
                        api_base=credentials["api_base"],
                        api_key=credentials["api_key"],
                        model_id=model_id,
                        task=task,
                        max_output_tokens=max_output_tokens,
                    )
                return await stream_chat_completion(
                    client,
                    api_base=credentials["api_base"],
                    api_key=credentials["api_key"],
                    model_id=model_id,
                    task=task,
                    safety_max_output_tokens=max_output_tokens,
                )

            return await self._run_with_executor(live_executor)


def default_model_ids() -> tuple[str, ...]:
    return tuple(spec.model_id for spec in MODEL_SPECS)
