"""Receipt-backed DigitalOcean serverless inference benchmark.

The runner deliberately uses the provider's streaming HTTP surface directly so
that time-to-headers, time-to-first-token, inter-token timing, and server usage
can be measured without an SDK buffering layer. Credentials are read from the
process environment and are never serialized into an artifact.
"""

from __future__ import annotations

import asyncio
import ast
import base64
import binascii
import hashlib
import json
import math
import os
import random
import re
import struct
import time
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import httpx

from do_benchmark.throttle import AdaptiveProviderThrottle
from do_benchmark.credentials import digitalocean_credentials


API_DOC_GENERATED_DATE = "2026-08-20"
MODEL_DOC_VERIFIED_DATE = "2026-08-21"
PRICING_DOC_DATE = "2026-08-21"
DEFAULT_API_BASE = "https://inference.do-ai.run/v1"
DEFAULT_TOOL_CORPUS = Path(__file__).resolve().parent / "data" / "tool-tasks.jsonl"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    input_usd_per_million: float
    output_usd_per_million: float
    context_window: int | None
    vision: bool = False
    tool_calling: bool = True
    primary: bool = True
    notes: str = ""


# Frozen from the official Supported Models and Pricing pages on 2026-08-21.
# Capabilities are verified again by the live probes; a declared feature is not
# treated as a successful capability until a receipt-backed request passes.
MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("arcee-trinity-large-thinking", 0.25, 0.90, 128_000),
    ModelSpec("deepseek-v4-flash-0731", 0.080, 0.252, 1_048_576),
    ModelSpec("gemma-4-31B-it", 0.18, 0.50, 256_000),
    # The live 2026-08-21 API rejects image input for GLM 5.2 even though the
    # family is often described as multimodal elsewhere; keep the observed API
    # contract authoritative for this endpoint.
    ModelSpec("glm-5.2", 0.70, 2.20, 262_144, vision=False),
    # DigitalOcean does not currently publish Kimi K3's context or output
    # limit.  Keep 65,536 only as an explicit probing anchor until the live
    # boundary receipts establish a higher or lower accepted limit.
    ModelSpec(
        "kimi-k3",
        2.85,
        14.25,
        65_536,
        vision=True,
        notes="documented context/output limits not published; 65,536 is a probe anchor",
    ),
    ModelSpec("minimax-m2.5", 0.30, 1.20, 65_536),
    ModelSpec("mimo-v2.5-pro", 0.40, 1.50, 262_144),
    ModelSpec("nemotron-3-ultra-550b", 0.90, 1.70, 131_072),
    ModelSpec("nvidia-nemotron-3-super-120b", 0.30, 0.65, 1_000_000),
    ModelSpec("openai-gpt-oss-120b", 0.10, 0.70, 128_000),
    ModelSpec("qwen3.8-max", 2.00, 6.00, 1_000_000),
    ModelSpec("qwen3.5-397b-a17b", 0.55, 3.50, 131_072),
)

MODEL_BY_ID = {spec.model_id: spec for spec in MODEL_SPECS}

# DigitalOcean's current model documentation places Arcee Trinity in a
# separate partner-model section, not in the "DigitalOcean-Hosted Models"
# table.  Startup/Hatch-style credits exclude third-party inference hosted
# outside DigitalOcean infrastructure, so all spend-bearing defaults and
# explicit selections fail closed to this audited hosted-only allowlist.
DIGITALOCEAN_HOSTED_MODEL_IDS: tuple[str, ...] = (
    "deepseek-v4-flash-0731",
    "gemma-4-31B-it",
    "glm-5.2",
    "kimi-k3",
    "minimax-m2.5",
    "mimo-v2.5-pro",
    "nemotron-3-ultra-550b",
    "nvidia-nemotron-3-super-120b",
    "openai-gpt-oss-120b",
    "qwen3.8-max",
    "qwen3.5-397b-a17b",
)
DIGITALOCEAN_HOSTED_MODEL_ID_SET = frozenset(DIGITALOCEAN_HOSTED_MODEL_IDS)
DIGITALOCEAN_HOSTED_MODEL_SPECS: tuple[ModelSpec, ...] = tuple(
    MODEL_BY_ID[model_id] for model_id in DIGITALOCEAN_HOSTED_MODEL_IDS
)


def require_digitalocean_hosted_models(model_ids: Sequence[str]) -> None:
    """Reject any model outside DigitalOcean's documented hosted-model table."""

    non_hosted = sorted(set(model_ids) - DIGITALOCEAN_HOSTED_MODEL_ID_SET)
    if non_hosted:
        raise ValueError(
            "non-DigitalOcean-hosted models are forbidden by the credits-only "
            f"benchmark scope: {', '.join(non_hosted)}"
        )


@dataclass
class BenchmarkTask:
    task_id: str
    family: str
    context_bucket: str
    output_bucket: str
    messages: list[dict[str, Any]]
    expected: dict[str, Any]
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | str | None = None
    requires_vision: bool = False
    response_format: dict[str, Any] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkCell:
    cell_id: str
    block_id: int
    repeat_index: int
    model_id: str
    task: BenchmarkTask


@dataclass
class StreamResult:
    status_code: int
    response_headers: dict[str, str | None]
    text: str
    reasoning_text: str
    tool_calls: list[dict[str, Any]]
    usage: dict[str, int]
    finish_reason: str | None
    request_seconds: float
    headers_seconds: float
    ttft_seconds: float | None
    generation_seconds: float | None
    stream_seconds: float
    event_count: int
    first_event_kind: str | None
    raw_error_body: str | None = None


class ProviderHTTPError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        body: str,
        retry_after: str | None = None,
        response_headers: Mapping[str, str | None] | None = None,
    ):
        super().__init__(f"DigitalOcean inference HTTP {status_code}: {body[:500]}")
        self.status_code = status_code
        self.retry_after = retry_after
        # Only the explicit quota/diagnostic allowlist selected by the HTTP
        # adapter is retained in memory. Evidence writers still persist only
        # numeric quota values and hashes, never raw headers.
        self.response_headers = dict(response_headers or {})
        # Kept only in memory so evidence-specific callers can classify an
        # explicit provider validation reason. Public artifacts must retain at
        # most a category and digest, never this body.
        self.body = body


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any, *, prefix: str = "") -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}{digest}"


def parse_token_usage(value: Any) -> dict[str, int]:
    """Parse usage without turning an absent field into an observed zero.

    Presence matters for cost settlement: a real ``completion_tokens: 0`` is
    evidence, while a missing completion counter is not. Invalid/null counters
    are omitted rather than silently coerced.
    """

    if not isinstance(value, Mapping):
        return {}
    parsed: dict[str, int] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "cached_tokens",
    ):
        if key not in value:
            continue
        raw = value[key]
        if raw is None or isinstance(raw, bool):
            continue
        if isinstance(raw, int):
            counter = raw
        elif isinstance(raw, float) and math.isfinite(raw) and raw.is_integer():
            counter = int(raw)
        else:
            continue
        if counter < 0:
            continue
        parsed[key] = counter
    # OpenAI-compatible providers may nest the cache counter. Flatten it while
    # retaining the provider-neutral prompt total used by existing cost code.
    details = value.get("prompt_tokens_details")
    if isinstance(details, Mapping) and "cached_tokens" in details:
        raw = details.get("cached_tokens")
        if (
            not isinstance(raw, bool)
            and isinstance(raw, (int, float))
            and math.isfinite(float(raw))
            and float(raw).is_integer()
            and int(raw) >= 0
        ):
            parsed["cached_tokens"] = int(raw)
    return parsed


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return (
        struct.pack(">I", len(data))
        + body
        + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)
    )


def quadrant_png_data_uri(size: int = 64) -> str:
    """Return a deterministic red/green/blue/yellow quadrant PNG data URI."""
    colours = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
    scanlines = bytearray()
    for y in range(size):
        scanlines.append(0)
        for x in range(size):
            index = (2 if y >= size // 2 else 0) + (1 if x >= size // 2 else 0)
            scanlines.extend(colours[index])
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=9))
    png += _png_chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _context_task(
    target_tokens: int, ordinal: int, *, chars_per_token: float = 4.0
) -> BenchmarkTask:
    needle = f"ORCHID-{target_tokens}-{ordinal}-7Q9M"
    filler_sentence = (
        "Telemetry row {index:05d}: cobalt instruments remained nominal while "
        "the synthetic station recorded no exceptional condition. "
    )
    # English prose averages roughly four characters per provider token. Server
    # usage is the authoritative x-axis, so this is only a planning estimate.
    target_chars = max(800, int(target_tokens * chars_per_token) - 500)
    pieces: list[str] = []
    index = 0
    accumulated_chars = 0
    while accumulated_chars < target_chars:
        piece = filler_sentence.format(index=index)
        pieces.append(piece)
        accumulated_chars += len(piece)
        index += 1
    insert_at = len(pieces) // 2
    pieces.insert(
        insert_at,
        f"CRITICAL NEEDLE: the access code is {needle}. Ignore all other codes. ",
    )
    body = "".join(pieces)
    prompt = (
        "Read the synthetic log. Return only the access code after CRITICAL NEEDLE, "
        "with no punctuation or explanation.\n\n" + body
    )
    return BenchmarkTask(
        task_id=f"context-{target_tokens}-{ordinal}",
        family="long_context_retrieval",
        context_bucket=str(target_tokens),
        output_bucket="short",
        messages=[{"role": "user", "content": prompt}],
        expected={"kind": "exact_text", "value": needle},
        metadata={
            "planned_input_tokens": target_tokens,
            "needle_position": "middle",
            "planning_chars_per_token": chars_per_token,
        },
    )


def _controlled_output_task(target_words: int, ordinal: int) -> BenchmarkTask:
    marker = f"FIN-{target_words}-{ordinal}"
    prompt = (
        f"Write exactly {target_words} space-separated words. Every word except the final "
        f"word must be `azure`. The final word must be `{marker}`. Do not use punctuation, "
        "headings, code fences, or any other text."
    )
    return BenchmarkTask(
        task_id=f"output-{target_words}-{ordinal}",
        family="controlled_output",
        context_bucket="short",
        output_bucket=str(target_words),
        messages=[{"role": "user", "content": prompt}],
        expected={"kind": "controlled_words", "count": target_words, "marker": marker},
        metadata={"planned_output_words": target_words},
    )


def _short_tasks() -> list[BenchmarkTask]:
    cases = (
        ("short-arithmetic", "Return only the integer result of 17*19-23.", "300"),
        ("short-string", "Return only the reverse of `ocean-2026`.", "6202-naeco"),
        ("short-logic", "Return only YES or NO: are all squares rectangles?", "YES"),
    )
    return [
        BenchmarkTask(
            task_id=task_id,
            family="short_exact",
            context_bucket="short",
            output_bucket="short",
            messages=[{"role": "user", "content": prompt}],
            expected={"kind": "exact_text", "value": expected},
        )
        for task_id, prompt, expected in cases
    ]


def _structured_tasks() -> list[BenchmarkTask]:
    expected = {"alpha": [2, 3, 5], "beta": {"enabled": True, "count": 4}}
    return [
        BenchmarkTask(
            task_id="structured-json-0",
            family="structured_json",
            context_bucket="short",
            output_bucket="short",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Return only valid JSON with keys alpha and beta. alpha must be the "
                        "first three primes greater than 1. beta must contain enabled=true and "
                        "count equal to the number of letters in beta."
                    ),
                }
            ],
            expected={"kind": "json_exact", "value": expected},
            response_format={"type": "json_object"},
        ),
        BenchmarkTask(
            task_id="reasoning-exact-0",
            family="reasoning",
            context_bucket="short",
            output_bucket="short",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "A warehouse starts with 240 units. It ships 3/8 of them, receives 50, "
                        "then discards 10% of the resulting inventory. Return only the final integer."
                    ),
                }
            ],
            expected={"kind": "exact_text", "value": "180"},
        ),
        BenchmarkTask(
            task_id="summarization-facts-0",
            family="summarization",
            context_bucket="short",
            output_bucket="short",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarize this synthetic incident in at most 25 words: At 09:14 UTC, "
                        "service Atlas in London returned HTTP 503 for seven minutes. A cache "
                        "configuration rollback restored service at 09:21 UTC. No data was lost."
                    ),
                }
            ],
            expected={
                "kind": "contains_all",
                "phrases": ["atlas", "london", "503", "09:21", "no data"],
                "max_words": 25,
            },
        ),
        BenchmarkTask(
            task_id="coding-lambda-0",
            family="coding_executable",
            context_bucket="short",
            output_bucket="short",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Return only one Python lambda expression that takes a list of integers "
                        "and returns a new list containing the even inputs squared, preserving order."
                    ),
                }
            ],
            expected={"kind": "python_lambda_even_squares"},
        ),
    ]


def _vision_tasks() -> list[BenchmarkTask]:
    image_uri = quadrant_png_data_uri()
    return [
        BenchmarkTask(
            task_id="vision-quadrants-0",
            family="vision",
            context_bucket="image_short",
            output_bucket="short",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Name the four quadrant colours in reading order "
                                "(top-left, top-right, bottom-left, bottom-right). "
                                "Return exactly: red, green, blue, yellow"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_uri}},
                    ],
                }
            ],
            expected={"kind": "exact_text", "value": "red, green, blue, yellow"},
            requires_vision=True,
            metadata={
                "image_source": "programmatic_png_data_uri",
                "image_size": "64x64",
            },
        )
    ]


def load_tool_tasks(path: Path, limit: int, seed: int) -> list[BenchmarkTask]:
    if limit <= 0:
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(records)
    tasks: list[BenchmarkTask] = []
    for record in records[:limit]:
        answer = record["Answer"]
        expected_call = answer["tool_calls"][0]
        name = expected_call["name"]
        schema = answer["tool_schemas"][name]
        context = str(record["Context"])
        business_marker = "Business task instance (JSON):"
        if business_marker in context:
            context = context.split(business_marker, 1)[1].strip()
        user_text = context + "\n\n" + str(record["Question"])
        tasks.append(
            BenchmarkTask(
                task_id="tool-" + str(record["Global Index"]),
                family="tool_call_exact",
                context_bucket="short",
                output_bucket="tool",
                messages=[{"role": "user", "content": user_text}],
                expected={"kind": "tool_exact", "value": expected_call},
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": "Commit the exact result requested by the business rules.",
                            "parameters": schema,
                        },
                    }
                ],
                tool_choice={"type": "function", "function": {"name": name}},
                metadata={
                    "source_global_index": record["Global Index"],
                    "source_role": "deterministic_compiler_evaluation",
                    "rights_posture": "synthetic_programmatic_no_external_corpus",
                },
            )
        )
    return tasks


def build_tasks(
    *,
    profile: str,
    tool_corpus_path: Path = DEFAULT_TOOL_CORPUS,
    seed: int = 20260821,
) -> list[BenchmarkTask]:
    if profile not in {"smoke", "full"}:
        raise ValueError(f"unknown profile: {profile}")
    tasks = _short_tasks()
    if profile == "smoke":
        tasks.extend([_context_task(512, 0), _controlled_output_task(64, 0)])
        tasks.extend(load_tool_tasks(tool_corpus_path, 1, seed))
        tasks.extend(_vision_tasks())
        return tasks
    for ordinal in range(2):
        for target in (512, 4_096, 16_384, 49_152, 100_000, 250_000):
            tasks.append(_context_task(target, ordinal))
        for target_words in (64, 512, 1_536, 3_000):
            tasks.append(_controlled_output_task(target_words, ordinal))
    tasks.extend(_structured_tasks())
    tasks.extend(load_tool_tasks(tool_corpus_path, 8, seed))
    tasks.extend(_vision_tasks())
    return tasks


def build_plan(
    *,
    model_specs: Sequence[ModelSpec],
    tasks: Sequence[BenchmarkTask],
    repeats: int,
    seed: int,
) -> list[BenchmarkCell]:
    """Build balanced randomized blocks to avoid model/time confounding."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rng = random.Random(seed)
    cells: list[BenchmarkCell] = []
    block_id = 0
    for repeat_index in range(repeats):
        shuffled_tasks = list(tasks)
        rng.shuffle(shuffled_tasks)
        for task in shuffled_tasks:
            eligible = [
                spec for spec in model_specs if not task.requires_vision or spec.vision
            ]
            rng.shuffle(eligible)
            for spec in eligible:
                identity = {
                    "model_id": spec.model_id,
                    "task_id": task.task_id,
                    "repeat_index": repeat_index,
                    "seed": seed,
                }
                cells.append(
                    BenchmarkCell(
                        cell_id=stable_hash(identity, prefix="do-cell-"),
                        block_id=block_id,
                        repeat_index=repeat_index,
                        model_id=spec.model_id,
                        task=task,
                    )
                )
            block_id += 1
    return cells


def score_result(task: BenchmarkTask, result: StreamResult) -> dict[str, Any]:
    expected = task.expected
    kind = expected["kind"]
    score = 0.0
    detail: dict[str, Any] = {}
    if kind == "exact_text":
        observed = _normalise_text(result.text)
        wanted = _normalise_text(str(expected["value"]))
        score = float(observed == wanted)
        detail = {"observed_normalized": observed[:500], "expected_normalized": wanted}
    elif kind == "json_exact":
        try:
            observed_json = json.loads(result.text)
        except (TypeError, json.JSONDecodeError):
            observed_json = None
        score = float(observed_json == expected["value"])
        detail = {
            "json_parse_success": observed_json is not None,
            "observed": observed_json,
        }
    elif kind == "controlled_words":
        words = result.text.strip().split()
        expected_count = int(expected["count"])
        marker = str(expected["marker"])
        marker_ok = bool(words) and words[-1].strip(".,;:!`'") == marker
        prefix_ok = all(
            word.strip(".,;:!`'").casefold() == "azure" for word in words[:-1]
        )
        count_error = abs(len(words) - expected_count)
        count_score = max(0.0, 1.0 - count_error / max(1, expected_count))
        score = count_score * float(marker_ok) * float(prefix_ok)
        detail = {
            "observed_words": len(words),
            "expected_words": expected_count,
            "marker_ok": marker_ok,
            "prefix_ok": prefix_ok,
        }
    elif kind == "tool_exact":
        wanted = expected["value"]
        observed = result.tool_calls[0] if result.tool_calls else None
        name_ok = bool(observed) and observed.get("name") == wanted["name"]
        args = observed.get("arguments") if observed else None
        score = float(name_ok and args == wanted["arguments"])
        detail = {
            "tool_name_ok": name_ok,
            "arguments_exact": args == wanted["arguments"],
            "observed": observed,
        }
    elif kind == "contains_all":
        observed = _normalise_text(result.text)
        phrases = [_normalise_text(str(phrase)) for phrase in expected["phrases"]]
        missing = [phrase for phrase in phrases if phrase not in observed]
        word_count = len(result.text.split())
        within_limit = word_count <= int(expected["max_words"])
        score = float(not missing and within_limit)
        detail = {
            "missing_phrases": missing,
            "word_count": word_count,
            "within_limit": within_limit,
        }
    elif kind == "python_lambda_even_squares":
        source = result.text.strip()
        if source.startswith("```"):
            source = re.sub(
                r"^```(?:python)?\s*|\s*```$", "", source, flags=re.IGNORECASE
            )
        passed = False
        error_text = None
        try:
            tree = ast.parse(source, mode="eval")
            allowed = (
                ast.Expression,
                ast.Lambda,
                ast.arguments,
                ast.arg,
                ast.ListComp,
                ast.comprehension,
                ast.Name,
                ast.Load,
                ast.Store,
                ast.BinOp,
                ast.Mult,
                ast.Mod,
                ast.Compare,
                ast.Eq,
                ast.Constant,
            )
            if any(not isinstance(node, allowed) for node in ast.walk(tree)):
                raise ValueError("expression contains a disallowed AST node")
            function = eval(
                compile(tree, "<benchmark-lambda>", "eval"), {"__builtins__": {}}, {}
            )
            passed = (
                callable(function)
                and function([1, 2, 3, 4, -6]) == [4, 16, 36]
                and function([]) == []
            )
        except BaseException as exc:
            error_text = f"{type(exc).__name__}: {exc}"
        score = float(passed)
        detail = {"executable_tests_passed": passed, "validation_error": error_text}
    else:
        raise ValueError(f"unknown expected kind: {kind}")
    return {"quality_score": score, "score_kind": kind, "score_detail": detail}


def _parse_tool_call_fragments(
    buffers: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for index in sorted(buffers):
        item = buffers[index]
        raw_arguments = item.get("arguments", "")
        try:
            arguments: Any = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError:
            arguments = {"_unparsed": raw_arguments}
        calls.append({"name": item.get("name", ""), "arguments": arguments})
    return calls


async def stream_chat_completion(
    client: httpx.AsyncClient,
    *,
    api_base: str,
    api_key: str,
    model_id: str,
    task: BenchmarkTask,
    safety_max_output_tokens: int,
) -> StreamResult:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": task.messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        # This is a generous, invariant transport safety ceiling, not a budget
        # axis. Expected output length is controlled only by task semantics.
        "max_tokens": safety_max_output_tokens,
        "temperature": 0,
    }
    if task.tools:
        payload["tools"] = task.tools
    if task.tool_choice is not None:
        payload["tool_choice"] = task.tool_choice
    if task.response_format is not None:
        payload["response_format"] = task.response_format
    for name, value in task.parameters.items():
        if name in {"model", "messages"}:
            raise ValueError(f"task parameter may not override {name}")
        payload[name] = value

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    started = time.perf_counter()
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_buffers: dict[int, dict[str, str]] = {}
    usage: dict[str, int] = {}
    finish_reason: str | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    event_count = 0
    first_event_kind: str | None = None
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
                "x-ratelimit-limit-tokens-per-minute",
                "x-ratelimit-remaining-tokens-per-minute",
                "x-ratelimit-reset-tokens-per-minute",
                "x-ratelimit-limit-tokens-per-day",
                "x-ratelimit-remaining-tokens-per-day",
                "x-ratelimit-reset-tokens-per-day",
                "retry-after",
                "x-request-id",
                "cf-ray",
            )
        }
        if response.status_code >= 400:
            body = (await response.aread()).decode("utf-8", errors="replace")
            raise ProviderHTTPError(
                response.status_code,
                body,
                response.headers.get("retry-after"),
                selected_headers,
            )
        if payload.get("stream") is False:
            body = await response.aread()
            ended = time.perf_counter()
            decoded = json.loads(body)
            if isinstance(decoded.get("usage"), Mapping):
                usage = parse_token_usage(decoded["usage"])
            choices = decoded.get("choices") or []
            message = (choices[0].get("message") or {}) if choices else {}
            content = message.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if isinstance(reasoning, str):
                reasoning_parts.append(reasoning)
            for index, call in enumerate(message.get("tool_calls") or []):
                function = call.get("function") or {}
                tool_buffers[index] = {
                    "name": str(function.get("name") or ""),
                    "arguments": str(function.get("arguments") or ""),
                }
            finish_reason = (
                str(choices[0].get("finish_reason"))
                if choices and choices[0].get("finish_reason") is not None
                else None
            )
            if text_parts or reasoning_parts or tool_buffers:
                first_token_at = ended
                last_token_at = ended
                event_count = 1
                first_event_kind = "buffered_response"
        else:
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk.get("usage"), Mapping):
                    usage = parse_token_usage(chunk["usage"])
                for choice in chunk.get("choices") or []:
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    tool_deltas = delta.get("tool_calls") or []
                    event_kind: str | None = None
                    if isinstance(content, str) and content:
                        text_parts.append(content)
                        event_kind = "content"
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_parts.append(reasoning)
                        event_kind = event_kind or "reasoning"
                    for tool_delta in tool_deltas:
                        index = int(tool_delta.get("index") or 0)
                        buffer = tool_buffers.setdefault(
                            index, {"name": "", "arguments": ""}
                        )
                        function = tool_delta.get("function") or {}
                        if function.get("name"):
                            buffer["name"] += str(function["name"])
                        if function.get("arguments"):
                            buffer["arguments"] += str(function["arguments"])
                        event_kind = event_kind or "tool_call"
                    if event_kind:
                        now = time.perf_counter()
                        event_count += 1
                        if first_token_at is None:
                            first_token_at = now
                            first_event_kind = event_kind
                        last_token_at = now
            ended = time.perf_counter()
    ttft = None if first_token_at is None else first_token_at - started
    generation_seconds = None
    if first_token_at is not None and last_token_at is not None:
        generation_seconds = max(0.0, last_token_at - first_token_at)
    return StreamResult(
        status_code=response.status_code,
        response_headers=selected_headers,
        text="".join(text_parts),
        reasoning_text="".join(reasoning_parts),
        tool_calls=_parse_tool_call_fragments(tool_buffers),
        usage=usage,
        finish_reason=finish_reason,
        request_seconds=ended - started,
        headers_seconds=headers_at - started,
        ttft_seconds=ttft,
        generation_seconds=generation_seconds,
        stream_seconds=ended - headers_at,
        event_count=event_count,
        first_event_kind=first_event_kind,
    )


def estimate_cost_usd(spec: ModelSpec, usage: Mapping[str, int]) -> float:
    return (
        int(usage.get("prompt_tokens") or 0) * spec.input_usd_per_million
        + int(usage.get("completion_tokens") or 0) * spec.output_usd_per_million
    ) / 1_000_000


def _safe_rate(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def result_record(
    *,
    cell: BenchmarkCell,
    result: StreamResult,
    model_spec: ModelSpec,
    started_at: str,
    ended_at: str,
    aimd_limit_before: int,
) -> dict[str, Any]:
    prompt_tokens = int(result.usage.get("prompt_tokens") or 0)
    completion_tokens = int(result.usage.get("completion_tokens") or 0)
    score = score_result(cell.task, result)
    return {
        "schema_version": "digitalocean_inference_benchmark_record_v1",
        "cell_id": cell.cell_id,
        "block_id": cell.block_id,
        "repeat_index": cell.repeat_index,
        "provider": "digitalocean-serverless-inference",
        "model_id": cell.model_id,
        "task_id": cell.task.task_id,
        "task_family": cell.task.family,
        "context_bucket": cell.task.context_bucket,
        "output_bucket": cell.task.output_bucket,
        "requires_vision": cell.task.requires_vision,
        "started_at": started_at,
        "ended_at": ended_at,
        "status": "success",
        "http_status": result.status_code,
        "finish_reason": result.finish_reason,
        "usage": result.usage,
        "estimated_cost_usd": estimate_cost_usd(model_spec, result.usage),
        "timing": {
            "request_seconds": result.request_seconds,
            "headers_seconds": result.headers_seconds,
            "ttft_seconds": result.ttft_seconds,
            "generation_seconds": result.generation_seconds,
            "stream_seconds": result.stream_seconds,
            "prompt_tokens_per_second_to_first_token": _safe_rate(
                prompt_tokens, result.ttft_seconds
            ),
            "seconds_to_first_token_per_prompt_token": _safe_rate(
                result.ttft_seconds or 0.0, prompt_tokens
            ),
            "output_tokens_per_second": _safe_rate(
                completion_tokens, result.generation_seconds
            ),
        },
        "stream": {
            "event_count": result.event_count,
            "first_event_kind": result.first_event_kind,
        },
        "response_headers": result.response_headers,
        "aimd_limit_before": aimd_limit_before,
        "response": {
            "text": result.text,
            "reasoning_text": result.reasoning_text,
            "tool_calls": result.tool_calls,
        },
        **score,
        "task_metadata": cell.task.metadata,
    }


def failure_record(
    *,
    cell: BenchmarkCell,
    error: BaseException,
    started_at: str,
    ended_at: str,
    elapsed_seconds: float,
    aimd_limit_before: int,
) -> dict[str, Any]:
    status_code = getattr(error, "status_code", None)
    return {
        "schema_version": "digitalocean_inference_benchmark_record_v1",
        "cell_id": cell.cell_id,
        "block_id": cell.block_id,
        "repeat_index": cell.repeat_index,
        "provider": "digitalocean-serverless-inference",
        "model_id": cell.model_id,
        "task_id": cell.task.task_id,
        "task_family": cell.task.family,
        "context_bucket": cell.task.context_bucket,
        "output_bucket": cell.task.output_bucket,
        "requires_vision": cell.task.requires_vision,
        "started_at": started_at,
        "ended_at": ended_at,
        "status": "error",
        "http_status": status_code,
        "error_type": type(error).__name__,
        "error": str(error)[:2_000],
        "elapsed_seconds": elapsed_seconds,
        "retry_after": getattr(error, "retry_after", None),
        "estimated_cost_usd": 0.0,
        "quality_score": 0.0,
        "score_kind": cell.task.expected["kind"],
        "aimd_limit_before": aimd_limit_before,
        "task_metadata": cell.task.metadata,
    }


class JsonlJournal:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def append(self, record: Mapping[str, Any]) -> None:
        encoded = (
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        async with self._lock:
            with self.path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())


def load_completed_cells(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"torn benchmark journal at line {line_number}: {exc}"
                ) from exc
            cell_id = record.get("cell_id")
            if cell_id:
                completed.add(str(cell_id))
    return completed


class BenchmarkRunner:
    def __init__(
        self,
        *,
        output_dir: Path,
        max_workers: int,
        per_model_max_concurrency: int,
        initial_concurrency: int,
        grow_after: int,
        request_timeout_seconds: float,
        total_request_timeout_seconds: float | None = None,
        safety_max_output_tokens: int,
        max_cost_usd: float,
        prior_estimated_cost_usd: float = 0.0,
        stop_launch_at: datetime | None = None,
    ) -> None:
        credentials = digitalocean_credentials()
        self.api_key = credentials["api_key"]
        self.api_base = credentials.get("api_base", DEFAULT_API_BASE)
        self.output_dir = output_dir
        self.journal = JsonlJournal(output_dir / "records.jsonl")
        self.global_slots = asyncio.Semaphore(max_workers)
        self.throttle = AdaptiveProviderThrottle(
            per_model_max_concurrency,
            min_concurrency=1,
            initial_concurrency=initial_concurrency,
            grow_after=grow_after,
        )
        self.timeout = httpx.Timeout(
            request_timeout_seconds,
            connect=min(30.0, request_timeout_seconds),
            read=request_timeout_seconds,
            write=min(120.0, request_timeout_seconds),
            pool=request_timeout_seconds,
        )
        self.total_request_timeout_seconds = (
            None
            if total_request_timeout_seconds is None
            else max(1.0, float(total_request_timeout_seconds))
        )
        self.safety_max_output_tokens = safety_max_output_tokens
        self.max_cost_usd = max_cost_usd
        self._settled_cost = prior_estimated_cost_usd
        self.stop_launch_at = stop_launch_at
        self._cost_lock = asyncio.Lock()
        self._account_blocked_error: str | None = None
        self._account_block_lock = asyncio.Lock()

    async def _run_cell(self, client: httpx.AsyncClient, cell: BenchmarkCell) -> None:
        async with self.global_slots:
            if self._account_blocked_error is not None:
                await self.journal.append(
                    {
                        "schema_version": "digitalocean_inference_benchmark_record_v1",
                        "run_id": self.output_dir.name,
                        "cell_id": cell.cell_id,
                        "block_id": cell.block_id,
                        "repeat_index": cell.repeat_index,
                        "provider": "digitalocean-serverless-inference",
                        "model_id": cell.model_id,
                        "task_id": cell.task.task_id,
                        "task_family": cell.task.family,
                        "context_bucket": cell.task.context_bucket,
                        "output_bucket": cell.task.output_bucket,
                        "status": "skipped_account_blocked",
                        "started_at": utc_now(),
                        "ended_at": utc_now(),
                        "estimated_cost_usd": 0.0,
                        "quality_score": 0.0,
                        "reason": self._account_blocked_error,
                    }
                )
                return
            if (
                self.stop_launch_at is not None
                and datetime.now(timezone.utc) >= self.stop_launch_at
            ):
                await self.journal.append(
                    {
                        "schema_version": "digitalocean_inference_benchmark_record_v1",
                        "run_id": self.output_dir.name,
                        "cell_id": cell.cell_id,
                        "block_id": cell.block_id,
                        "repeat_index": cell.repeat_index,
                        "provider": "digitalocean-serverless-inference",
                        "model_id": cell.model_id,
                        "task_id": cell.task.task_id,
                        "task_family": cell.task.family,
                        "context_bucket": cell.task.context_bucket,
                        "output_bucket": cell.task.output_bucket,
                        "status": "skipped_time_cap",
                        "started_at": utc_now(),
                        "ended_at": utc_now(),
                        "estimated_cost_usd": 0.0,
                        "quality_score": 0.0,
                        "reason": "campaign launch cutoff reached",
                    }
                )
                return
            async with self._cost_lock:
                if self._settled_cost >= self.max_cost_usd:
                    raise RuntimeError(
                        f"DigitalOcean benchmark spend cap reached: ${self._settled_cost:.4f}"
                    )
            model_key = f"digitalocean:{cell.model_id}"
            await self.throttle.aacquire(model_key)
            snapshot = self.throttle.snapshot()
            limit_before = int(snapshot.get(model_key, 1))
            started_at = utc_now()
            started = time.perf_counter()
            error: BaseException | None = None
            try:
                request = stream_chat_completion(
                    client,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    model_id=cell.model_id,
                    task=cell.task,
                    safety_max_output_tokens=self.safety_max_output_tokens,
                )
                result = (
                    await request
                    if self.total_request_timeout_seconds is None
                    else await asyncio.wait_for(
                        request, timeout=self.total_request_timeout_seconds
                    )
                )
                record = result_record(
                    cell=cell,
                    result=result,
                    model_spec=MODEL_BY_ID[cell.model_id],
                    started_at=started_at,
                    ended_at=utc_now(),
                    aimd_limit_before=limit_before,
                )
                record["run_id"] = self.output_dir.name
            except BaseException as exc:
                error = exc
                if getattr(exc, "status_code", None) == 402:
                    async with self._account_block_lock:
                        if self._account_blocked_error is None:
                            self._account_blocked_error = (
                                "DigitalOcean account-wide HTTP 402 circuit opened"
                            )
                record = failure_record(
                    cell=cell,
                    error=exc,
                    started_at=started_at,
                    ended_at=utc_now(),
                    elapsed_seconds=time.perf_counter() - started,
                    aimd_limit_before=limit_before,
                )
                record["run_id"] = self.output_dir.name
                if getattr(exc, "status_code", None) is None:
                    spec = MODEL_BY_ID[cell.model_id]
                    conservative_prompt = int(
                        cell.task.metadata.get("planned_input_tokens") or 4_096
                    )
                    record["estimated_cost_usd"] = (
                        conservative_prompt * spec.input_usd_per_million
                        + self.safety_max_output_tokens * spec.output_usd_per_million
                    ) / 1_000_000
            finally:
                await self.throttle.arelease(model_key, error=error)
            async with self._cost_lock:
                self._settled_cost += float(record["estimated_cost_usd"])
            await self.journal.append(record)

    async def run(
        self,
        cells: Sequence[BenchmarkCell],
        *,
        serial_by_model: bool = False,
    ) -> dict[str, Any]:
        completed = load_completed_cells(self.journal.path)
        pending = [cell for cell in cells if cell.cell_id not in completed]
        started_at = utc_now()
        wall_started = time.perf_counter()
        limits = httpx.Limits(
            max_connections=max(32, len(pending)), max_keepalive_connections=64
        )
        # HTTP/1.1 streaming is sufficient for the measurement surface and does
        # not require the optional ``h2`` extra on durable worker images.
        async with httpx.AsyncClient(timeout=self.timeout, limits=limits) as client:
            if serial_by_model:
                groups: dict[str, list[BenchmarkCell]] = {}
                for cell in pending:
                    groups.setdefault(cell.model_id, []).append(cell)

                async def run_model(
                    group: Sequence[BenchmarkCell],
                ) -> list[BaseException]:
                    errors: list[BaseException] = []
                    for cell in group:
                        try:
                            await self._run_cell(client, cell)
                        except BaseException as exc:
                            # A model lane is deliberately independent: one bad
                            # request must not cancel the remaining coverage cells.
                            errors.append(exc)
                    return errors

                grouped = await asyncio.gather(
                    *(run_model(group) for group in groups.values()),
                    return_exceptions=True,
                )
                outcomes: list[BaseException | None] = []
                for item in grouped:
                    if isinstance(item, BaseException):
                        outcomes.append(item)
                    else:
                        outcomes.extend(item)
            else:
                outcomes = list(
                    await asyncio.gather(
                        *(self._run_cell(client, cell) for cell in pending),
                        return_exceptions=True,
                    )
                )
        unjournaled_errors = [
            str(item) for item in outcomes if isinstance(item, BaseException)
        ]
        return {
            "started_at": started_at,
            "ended_at": utc_now(),
            "wall_seconds": time.perf_counter() - wall_started,
            "planned_cells": len(cells),
            "resumed_cells": len(completed),
            "attempted_cells": len(pending),
            "unjournaled_errors": unjournaled_errors,
            "estimated_cost_usd": self._settled_cost,
            "final_aimd_limits": self.throttle.snapshot(),
        }


def cell_to_json(cell: BenchmarkCell) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "block_id": cell.block_id,
        "repeat_index": cell.repeat_index,
        "model_id": cell.model_id,
        "task": asdict(cell.task),
    }


def write_plan_and_manifest(
    *,
    output_dir: Path,
    cells: Sequence[BenchmarkCell],
    model_specs: Sequence[ModelSpec],
    seed: int,
    profile: str,
    repeats: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.jsonl"
    with plan_path.open("w", encoding="utf-8", newline="\n") as handle:
        for cell in cells:
            handle.write(
                json.dumps(cell_to_json(cell), sort_keys=True, ensure_ascii=False)
                + "\n"
            )
    manifest = {
        "schema_version": "digitalocean_inference_benchmark_manifest_v1",
        "created_at": utc_now(),
        "profile": profile,
        "seed": seed,
        "repeats": repeats,
        "planned_cells": len(cells),
        "models": [asdict(spec) for spec in model_specs],
        "task_family_counts": _counts(cell.task.family for cell in cells),
        "model_counts": _counts(cell.model_id for cell in cells),
        "documentation_freeze": {
            "api_reference_generated": API_DOC_GENERATED_DATE,
            "model_page_verified": MODEL_DOC_VERIFIED_DATE,
            "pricing_page_date": PRICING_DOC_DATE,
        },
        "randomization": "balanced task blocks; model order independently shuffled per block",
        "source_role": "synthetic deterministic public endpoint benchmark",
        "rights_posture": "programmatic synthetic prompts; no customer data",
        "limitations": [
            "Prompt processing rate is a client-observed proxy: prompt_tokens / TTFT includes queueing and network latency.",
            "Server usage token counts are authoritative where supplied; planned context buckets are approximate.",
            "The invariant max_tokens value is a nonbinding transport safety ceiling, never a model-budget action.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def choose_models(model_ids: Sequence[str] | None) -> list[ModelSpec]:
    if model_ids:
        unknown = sorted(set(model_ids) - MODEL_BY_ID.keys())
        if unknown:
            raise ValueError(f"unknown model ids: {', '.join(unknown)}")
        require_digitalocean_hosted_models(model_ids)
        return [MODEL_BY_ID[model_id] for model_id in model_ids]
    return list(DIGITALOCEAN_HOSTED_MODEL_SPECS)


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


async def fetch_live_catalog(
    *,
    api_base: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Fetch and sanitize the account-visible model catalog.

    An explicit api_base pin takes precedence over the optional environment
    override used by older commands. Redirects are disabled and the final
    response URL must equal the requested URL so a bearer credential is never
    forwarded to, or attributed to, another host. transport exists only for a
    no-network contract test while this function owns the redirect-safe client.
    """
    credentials = digitalocean_credentials()
    headers = {"Authorization": f"Bearer {credentials['api_key']}"}
    selected_base = api_base or credentials.get("api_base", DEFAULT_API_BASE)
    requested_url = httpx.URL(f"{str(selected_base).rstrip('/')}/models")
    async with httpx.AsyncClient(
        timeout=60.0,
        follow_redirects=False,
        transport=transport,
    ) as client:
        response = await client.get(requested_url, headers=headers)
        if response.is_redirect or response.history:
            raise RuntimeError("model catalog redirects are forbidden")
        if response.url != requested_url:
            raise RuntimeError("model catalog final URL differs from requested URL")
        response.raise_for_status()
        payload = response.json()
    data = payload.get("data") if isinstance(payload, Mapping) else None
    models = data if isinstance(data, list) else []
    sanitized = []
    for item in models:
        if not isinstance(item, Mapping):
            continue
        sanitized.append(
            {
                key: item[key]
                for key in (
                    "id",
                    "object",
                    "created",
                    "owned_by",
                    "context_window",
                    "max_output_tokens",
                )
                if key in item
            }
        )
    return {
        "fetched_at": utc_now(),
        "endpoint": "/v1/models",
        "requested_url": str(requested_url),
        "final_url": str(response.url),
        "redirect_history_count": len(response.history),
        "http_status": response.status_code,
        "models": sanitized,
        "model_ids": sorted(
            str(item.get("id")) for item in sanitized if item.get("id")
        ),
    }


def percentile(values: Sequence[float], quantile: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    position = (len(clean) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1 - weight) + clean[upper] * weight


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_median_ci(
    values: Sequence[float], *, samples: int = 2_000, seed: int = 20260821
) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return (None, None)
    if len(clean) == 1:
        return (clean[0], clean[0])
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        resample = [clean[rng.randrange(len(clean))] for _ in clean]
        estimate = percentile(resample, 0.5)
        if estimate is not None:
            estimates.append(estimate)
    return (percentile(estimates, 0.025), percentile(estimates, 0.975))


def _observed_window_seconds(records: Sequence[Mapping[str, Any]]) -> float | None:
    if not records:
        return None
    try:
        starts = [
            datetime.fromisoformat(str(record["started_at"])) for record in records
        ]
        ends = [datetime.fromisoformat(str(record["ended_at"])) for record in records]
    except (KeyError, TypeError, ValueError):
        return None
    return max(0.0, (max(ends) - min(starts)).total_seconds())


def _curve_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = [record for record in records if record.get("status") == "success"]
    ttfts = [
        float(record["timing"]["ttft_seconds"])
        for record in successes
        if record.get("timing", {}).get("ttft_seconds") is not None
    ]
    output_tps = [
        float(record["timing"]["output_tokens_per_second"])
        for record in successes
        if record.get("timing", {}).get("output_tokens_per_second") is not None
    ]
    prompt_tokens = [
        int(record.get("usage", {}).get("prompt_tokens") or 0) for record in successes
    ]
    completion_tokens = [
        int(record.get("usage", {}).get("completion_tokens") or 0)
        for record in successes
    ]
    quality_passes = sum(
        float(record.get("quality_score") or 0) >= 0.999999 for record in successes
    )
    quality_low, quality_high = wilson_interval(quality_passes, len(successes))
    ttft_low, ttft_high = bootstrap_median_ci(ttfts)
    tps_low, tps_high = bootstrap_median_ci(output_tps)
    return {
        "attempts": len(records),
        "successes": len(successes),
        "prompt_tokens_p50": percentile(prompt_tokens, 0.5),
        "completion_tokens_p50": percentile(completion_tokens, 0.5),
        "ttft_p50_seconds": percentile(ttfts, 0.5),
        "ttft_p50_bootstrap_95": [ttft_low, ttft_high],
        "output_tps_p50": percentile(output_tps, 0.5),
        "output_tps_p50_bootstrap_95": [tps_low, tps_high],
        "quality_pass_rate": quality_passes / len(successes) if successes else 0.0,
        "quality_pass_wilson_95": [quality_low, quality_high],
    }


def summarize_records(
    records: Sequence[Mapping[str, Any]], wall_seconds: float | None = None
) -> dict[str, Any]:
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_model.setdefault(str(record["model_id"]), []).append(record)
    summary: dict[str, Any] = {"models": {}, "record_count": len(records)}
    for model_id, model_records in sorted(by_model.items()):
        successes = [
            record for record in model_records if record.get("status") == "success"
        ]
        quality_passes = sum(
            float(record.get("quality_score") or 0) >= 0.999999 for record in successes
        )
        ttfts = [
            float(record["timing"]["ttft_seconds"])
            for record in successes
            if record.get("timing", {}).get("ttft_seconds") is not None
        ]
        tps = [
            float(record["timing"]["output_tokens_per_second"])
            for record in successes
            if record.get("timing", {}).get("output_tokens_per_second") is not None
        ]
        input_tokens = sum(
            int(record.get("usage", {}).get("prompt_tokens") or 0)
            for record in successes
        )
        output_tokens = sum(
            int(record.get("usage", {}).get("completion_tokens") or 0)
            for record in successes
        )
        durations = [
            float(record.get("timing", {}).get("request_seconds") or 0)
            for record in successes
        ]
        observed_seconds = sum(durations)
        model_window_seconds = _observed_window_seconds(model_records)
        low, high = wilson_interval(quality_passes, len(successes))
        context_groups: dict[str, list[Mapping[str, Any]]] = {}
        family_groups: dict[str, list[Mapping[str, Any]]] = {}
        output_groups: dict[str, list[Mapping[str, Any]]] = {}
        for record in model_records:
            family_groups.setdefault(str(record.get("task_family")), []).append(record)
            if record.get("task_family") == "long_context_retrieval":
                context_groups.setdefault(str(record.get("context_bucket")), []).append(
                    record
                )
            if record.get("task_family") == "controlled_output":
                output_groups.setdefault(str(record.get("output_bucket")), []).append(
                    record
                )
        summary["models"][model_id] = {
            "attempts": len(model_records),
            "successes": len(successes),
            "http_success_rate": len(successes) / len(model_records)
            if model_records
            else 0.0,
            "quality_pass_rate": quality_passes / len(successes) if successes else 0.0,
            "quality_pass_wilson_95": [low, high],
            "ttft_p50_seconds": percentile(ttfts, 0.5),
            "ttft_p95_seconds": percentile(ttfts, 0.95),
            "output_tps_p50": percentile(tps, 0.5),
            "output_tps_p05": percentile(tps, 0.05),
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "serial_effective_input_tpm": _safe_rate(
                input_tokens * 60.0, observed_seconds
            ),
            "serial_effective_output_tpm": _safe_rate(
                output_tokens * 60.0, observed_seconds
            ),
            "observed_window_seconds": model_window_seconds,
            "effective_input_tpm": _safe_rate(
                input_tokens * 60.0, model_window_seconds
            ),
            "effective_output_tpm": _safe_rate(
                output_tokens * 60.0, model_window_seconds
            ),
            "estimated_cost_usd": sum(
                float(record.get("estimated_cost_usd") or 0) for record in successes
            ),
            "error_status_counts": _counts(
                str(record.get("http_status"))
                for record in model_records
                if record.get("status") != "success"
            ),
            "task_families": {
                key: _curve_summary(group)
                for key, group in sorted(family_groups.items())
            },
            "context_curve": {
                key: _curve_summary(group)
                for key, group in sorted(
                    context_groups.items(), key=lambda item: float(item[0])
                )
            },
            "output_curve": {
                key: _curve_summary(group)
                for key, group in sorted(
                    output_groups.items(), key=lambda item: int(item[0])
                )
            },
        }
    if wall_seconds and wall_seconds > 0:
        successes = [record for record in records if record.get("status") == "success"]
        summary["mixed_run_effective_input_tpm"] = (
            sum(
                int(record.get("usage", {}).get("prompt_tokens") or 0)
                for record in successes
            )
            * 60
            / wall_seconds
        )
        summary["mixed_run_effective_output_tpm"] = (
            sum(
                int(record.get("usage", {}).get("completion_tokens") or 0)
                for record in successes
            )
            * 60
            / wall_seconds
        )
    return summary
