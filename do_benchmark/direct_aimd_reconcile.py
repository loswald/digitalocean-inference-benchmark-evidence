"""Fail-closed reconciliation for the completed 2026-08-23 direct AIMD run.

The run is scientifically usable, but its embedded model contracts predate a
later correction to four price rows and one (expanded) context-window row.
This module does not rewrite any measurement.  It verifies the exact immutable
source artifacts, validates their original accounting under their original
contracts, and emits a deterministic receipt that reprices every terminal
request under the current frozen contracts.  Incomplete/error outcomes retain
their full, newly repriced pre-send reservation.

The policy is intentionally run-specific.  It cannot reconcile another run or
another kind of model-contract drift.
"""

from __future__ import annotations

import hashlib
import gzip
import json
import math
import os
import sqlite3
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping

from do_benchmark.core import MODEL_BY_ID, canonical_json, parse_token_usage


RECONCILIATION_SCHEMA = "do_direct_aimd_reconciliation_v1"
RECONCILIATION_POLICY = "do-direct-aimd-20260823-88bd85a-contract-reprice-v1"
SOURCE_CAMPAIGN_ID = "do-direct-5b8072bf1ef24627b7e7"
SOURCE_GIT_COMMIT = "88bd85a"
SOURCE_ARCHIVE_SHA256 = (
    "cf1657ac5382eb169df2b6801114994889e7a441f8bd2d71f979432863ed1f64"
)
SOURCE_ARTIFACT_SHA256 = {
    "manifest": "16a1912186ab5447e1c293e5376acae7f71a1c7b4c6f74ee4472f3953196a982",
    "summary": "e9e253cd6900ea4d3ebb365181ff93f4f7858c7e6bea701979a6d75746ad79e9",
    "epochs": "b9d6b750baec64af0bebe2b4d82f526931c9b84481c0519a2b0210c45a0a226c",
    "requests": "43eb0a863256dc4c6d4d6f9d6cb4d28b870379fdfeb9a842eb2419aa7e2da1b9",
    "reservations": "c1a8c0ece6c80b6a5b4531523e4f83baf6bcedacc82e9dc99f73a303d73c6772",
}
SOURCE_COUNTS = {"epochs": 554, "requests": 6253, "reservations": 6253}
SOURCE_IDENTITY = {
    "input_tokens": 32_000,
    "long_output_words": 1_024,
    "short_max_output_tokens": 64,
    "long_max_output_tokens": 2_048,
    "mixed_max_output_tokens": 1_024,
    "shapes": ["short_short", "input32k_short", "short_long", "mixed"],
}

OPAQUE_PREFIX_OLD_USD = 7.277863
V3_CHECKPOINT_NAME = "pre-runtime-read-lock-fix-20260823T130122047276Z"
V3_MANIFEST_SHA256 = "0096e341a65dcff8bcf927b66a341dce39e0fe690b63196be866db71e1434a59"
V3_LEDGER_GZIP_SHA256 = (
    "cc64fe29ef10790b6ea1c912e12d186c1601008282f7b3c9ab69e02111c51cbe"
)
V3_LEDGER_DECODED_SHA256 = (
    "959402c89132b3abae995a8ff8a5949dadbf73d003f7df8f30997361c00c90e2"
)
V3_RECONCILED_ROWS_SHA256 = (
    "82b57a97e762a45848a678d584481d2355c5aac150e5a3bb48882eef176352e2"
)
V3_LOCALLY_RECOMPUTED_ROWS_SHA256 = (
    "69b4c2ce98f2f76aa938437e9bc0648359e2f85d73bc16760f341e3ad6b849e0"
)
V3_ATTESTATION_SHA256 = (
    "f6d9d736e4df2d40440745a0fe060a0a82fa56ad7de76cc4da511d0566d5e758"
)
V3_RECONCILED_UPPER_USD = 13.981972928609271
V3_USAGE_COST_USD = 6.036790270
V3_MAX_RESERVE_COST_USD = 6.866863950
V3_LEGACY_TOPUP_REPRICED_USD = 1.0783187086092715
PRIOR_LINEAGE = (
    {
        "name": "do-direct-coverage-20260823",
        "planned": 1263,
        "durable": 1102,
        "old_cumulative_exposure_usd": 21.438454073,
        "artifact_sha256": {
            "live-catalog.json": "fdd983b3826428e54e99d021ae9d21de93842e26ffcdbc5aa6ec8d99c08f45c6",
            "manifest.json": "5d90a1d3b3c3837b80d8c05461b9cbafa6621797a489fd5073d1bf9b61a61baa",
            "plan.jsonl": "e5cd0554f83b8cf2ed9cac36317b0d9deaf11fa3b73bc9841a9621d1bd883cf2",
            "receipt.json": "752765855c516978a6b63622e5de70dab79f85d733574033d4c70137ee5afcf0",
            "records.jsonl": "4f8a96f658bc436bf94a79a42d6ba290982d29dee674bc80e13a93d346fecd03",
            "records.partial.jsonl": "4f8a96f658bc436bf94a79a42d6ba290982d29dee674bc80e13a93d346fecd03",
            "run-summary.json": "fbdad404f69e24c51b7d4a64796f9d77166f70f3f2bf2d91f5df0e24ed3aa40d",
            "summary.json": "3c747ff4c79af337c7736fda056100e32ab5cd3cac04c3f3f1a517fafb3c2e25",
        },
    },
    {
        "name": "do-direct-coverage-completion-20260823",
        "planned": 105,
        "durable": 19,
        "old_cumulative_exposure_usd": 21.479657462,
        "artifact_sha256": {
            "manifest.json": "a2f7cf9f6be8e3250bd7c959e83d0fbf0cccbebc5209b8af8fd50c75f5712c7b",
            "plan.jsonl": "dca77323a1f0bdaf7b0129c728f95aa8dc90b94a83fbe997da03d1e8a12aa64d",
            "receipt.json": "9873cdd889b16af6d212c1c64e9430db7829bbd9f79155987a31fcae1661b2e5",
            "records.jsonl": "0f370eff390926e52f74bc986356e4dd8c58783f58b82dee228ec2ded7e4fd7e",
            "run-summary.json": "1f8b0751eda7bf6e7a257a54cf71c9167195dd65178e1bfe7ca2656886f95d5d",
            "summary.json": "d224c867fbaaa965e105006e9fb0a14df1d817f21b2999534ce1884ac243b787",
        },
    },
    {
        "name": "do-direct-coverage-serial-20260823",
        "planned": 105,
        "durable": 105,
        "old_cumulative_exposure_usd": 21.993123942,
        "artifact_sha256": {
            "manifest.json": "ff27a5ae8dee7e6d738645c9dc683ec330c25a6c403e2489ad41fc1ebf1950d1",
            "plan.jsonl": "4c69e2637cc80c275e40f344a64c8033d1cb42ae299016806368111bf9a65a71",
            "receipt.json": "323c809eb1396098e25831c0cb23f40c1e1fe6adbcc198a7396323a0cb09a050",
            "records.jsonl": "c68cdb51ee0d693ccace45f8e63b5b5c9b98b6b401631475e752df1d432f70b0",
            "run-summary.json": "65377322eaab465e3cf6a560171caadd090fcf7c331d02c26d82b7f095644151",
            "summary.json": "c6ffc0230c3796331604da5cfdeea5763cae681d47873156bb47c2f5e59b25df",
        },
    },
)
PRIOR_SOURCE_ARCHIVES = {
    "do-direct-coverage-20260823": (
        "do-direct-src-5ecc7e3.tar.gz",
        "4829f85bdd78f6de813542ae34684d879cc6e40a4c1a170e1419140b39cc021f",
    ),
    "do-direct-coverage-completion-20260823": (
        "do-direct-src-dbe9ff3.tar.gz",
        "58f39950d1fff0654b2948b9168f4ceaf5527f91380cf534defb0f4a6e6be49b",
    ),
    "do-direct-coverage-serial-20260823": (
        "do-direct-src-0f08999.tar.gz",
        "14a226062358407b56b6e1937bc6fc59eb50d5497c23ba5af54243b86ccd843b",
    ),
}

# Every permitted difference is listed with its exact before and after value.
# Any additional, missing, or changed difference fails closed.
ALLOWED_CONTRACT_TRANSFORMATIONS: dict[str, dict[str, tuple[Any, Any]]] = {
    "glm-5.2": {
        "input_usd_per_million": (0.63, 0.70),
        "output_usd_per_million": (1.98, 2.20),
    },
    "qwen3.5-397b-a17b": {
        "input_usd_per_million": (0.302, 0.55),
        "output_usd_per_million": (1.925, 3.50),
    },
    "minimax-m2.5": {
        "input_usd_per_million": (0.225, 0.30),
        "output_usd_per_million": (0.90, 1.20),
    },
    "nvidia-nemotron-3-super-120b": {
        "input_usd_per_million": (0.165, 0.30),
        "output_usd_per_million": (0.358, 0.65),
    },
    "qwen3.8-max": {"context_window": (262_144, 1_000_000)},
}

ARTIFACT_FILES = {
    "manifest": "manifest.json",
    "summary": "summary.json",
    "epochs": "epochs.jsonl",
    "requests": "requests.jsonl",
    "reservations": "reservations.jsonl",
}


class AIMDReconciliationError(RuntimeError):
    """Raised when the exact legacy evidence cannot be reconciled safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AIMDReconciliationError(
            f"invalid reconciliation input {path.name}"
        ) from error
    if not isinstance(value, dict):
        raise AIMDReconciliationError(f"{path.name} must contain an object")
    return value


def _read_jsonl(path: Path, identity_key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AIMDReconciliationError(f"cannot read {path.name}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise AIMDReconciliationError(
                f"torn reconciliation input {path.name}:{line_number}"
            ) from error
        if not isinstance(row, dict):
            raise AIMDReconciliationError(f"invalid row in {path.name}:{line_number}")
        identity = row.get(identity_key)
        if not isinstance(identity, str) or not identity or identity in rows:
            raise AIMDReconciliationError(
                f"invalid or duplicate {identity_key} in {path.name}:{line_number}"
            )
        rows[identity] = row
    return rows


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AIMDReconciliationError(f"invalid {label}") from error
    if not math.isfinite(result) or result < 0:
        raise AIMDReconciliationError(f"invalid {label}")
    return result


def _cost(
    contract: Mapping[str, Any], prompt_tokens: int, completion_tokens: int
) -> float:
    return (
        prompt_tokens * float(contract["input_usd_per_million"])
        + completion_tokens * float(contract["output_usd_per_million"])
    ) / 1_000_000


def _require_no_under_settlement(old_cost: float, new_cost: float) -> None:
    if new_cost + 1e-15 < old_cost:
        raise AIMDReconciliationError("reconciliation would under-settle evidence")


def _complete_usage(row: Mapping[str, Any]) -> tuple[int, int] | None:
    raw = row.get("usage")
    if not isinstance(raw, Mapping):
        return None
    usage = parse_token_usage(raw)
    if "prompt_tokens" not in usage or "completion_tokens" not in usage:
        return None
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    # Prompt-only counters do not prove full output settlement: several Arcee
    # rows terminated at length after hidden reasoning while reporting zero
    # completion tokens.  Both sides must therefore be strictly positive.
    if prompt_tokens <= 0 or completion_tokens <= 0:
        return None
    return prompt_tokens, completion_tokens


def _historical_usage(row: Mapping[str, Any]) -> tuple[int, int] | None:
    """Reproduce the original runner's settlement rule for ledger validation."""

    raw = row.get("usage")
    if not isinstance(raw, Mapping):
        return None
    usage = parse_token_usage(raw)
    if "prompt_tokens" not in usage or "completion_tokens" not in usage:
        return None
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    if prompt_tokens <= 0 or completion_tokens < 0:
        return None
    return prompt_tokens, completion_tokens


def _model_contracts(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows = manifest.get("model_specs")
    if not isinstance(rows, list):
        raise AIMDReconciliationError("source manifest lacks model_specs")
    old: dict[str, dict[str, Any]] = {}
    for value in rows:
        if not isinstance(value, Mapping) or not isinstance(value.get("model_id"), str):
            raise AIMDReconciliationError("invalid source model contract")
        row = dict(value)
        model_id = str(row["model_id"])
        if model_id in old:
            raise AIMDReconciliationError("duplicate source model contract")
        old[model_id] = row
    current_ids = set(MODEL_BY_ID)
    if set(old) != current_ids:
        raise AIMDReconciliationError("source/current model universes differ")
    new = {model_id: asdict(MODEL_BY_ID[model_id]) for model_id in sorted(old)}
    observed_transformations: list[dict[str, Any]] = []
    for model_id in sorted(old):
        old_row = old[model_id]
        new_row = new[model_id]
        if set(old_row) != set(new_row):
            raise AIMDReconciliationError(f"model contract fields drifted: {model_id}")
        observed: dict[str, tuple[Any, Any]] = {}
        for field in sorted(old_row):
            if canonical_json(old_row[field]) != canonical_json(new_row[field]):
                observed[field] = (old_row[field], new_row[field])
                observed_transformations.append(
                    {
                        "model_id": model_id,
                        "field": field,
                        "old": old_row[field],
                        "new": new_row[field],
                        "kind": (
                            "conservative_reprice"
                            if field.endswith("usd_per_million")
                            else "context_expansion_compatible_with_32k_payload"
                        ),
                    }
                )
        allowed = ALLOWED_CONTRACT_TRANSFORMATIONS.get(model_id, {})
        if canonical_json(observed) != canonical_json(allowed):
            raise AIMDReconciliationError(
                f"unapproved source/current model-contract drift: {model_id}"
            )
    if set(ALLOWED_CONTRACT_TRANSFORMATIONS) - set(old):
        raise AIMDReconciliationError("reconciliation policy references absent models")
    qwen_old = int(old["qwen3.8-max"]["context_window"])
    qwen_new = int(new["qwen3.8-max"]["context_window"])
    if not (SOURCE_IDENTITY["input_tokens"] <= qwen_old <= qwen_new):
        raise AIMDReconciliationError("Qwen context drift is not payload-compatible")
    return old, new, observed_transformations


def _coverage_request_reserve(
    plan_row: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[float, int]:
    task = plan_row.get("task")
    model_id = plan_row.get("model_id")
    if not isinstance(task, Mapping) or not isinstance(model_id, str):
        raise AIMDReconciliationError("invalid prior-lineage plan row")
    messages = task.get("messages")
    parameters = task.get("parameters")
    if not isinstance(messages, list) or not isinstance(parameters, Mapping):
        raise AIMDReconciliationError("prior-lineage task cannot be reconstructed")
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 4_096,
        "temperature": 0,
    }
    if task.get("tools"):
        payload["tools"] = task["tools"]
    if task.get("tool_choice") is not None:
        payload["tool_choice"] = task["tool_choice"]
    if task.get("response_format") is not None:
        payload["response_format"] = task["response_format"]
    for name, value in parameters.items():
        if name in {"model", "messages"}:
            raise AIMDReconciliationError("prior task overrides a protected field")
        payload[str(name)] = value
    metadata = task.get("metadata")
    planned_input = 0
    if (
        isinstance(metadata, Mapping)
        and metadata.get("planned_input_tokens") is not None
    ):
        try:
            planned_input = int(metadata["planned_input_tokens"])
        except (TypeError, ValueError) as error:
            raise AIMDReconciliationError("invalid planned_input_tokens") from error
        if planned_input < 0:
            raise AIMDReconciliationError("invalid planned_input_tokens")
    serialized_bytes = len(canonical_json(payload).encode("utf-8"))
    prompt_bound = max(serialized_bytes + 512, math.ceil(planned_input * 1.5))
    return _cost(contract, prompt_bound, 4_096), prompt_bound


@contextmanager
def _open_hash_verified_sqlite_copy(
    source_sqlite: bytes,
) -> Iterator[sqlite3.Connection]:
    """Open a private rollback-header copy without requiring ``deserialize``.

    The source bytes have already been hash-verified by the caller.  Only the
    private temporary copy receives the two WAL-header byte changes needed to
    read the crash-consistent main database without a companion WAL file.
    """

    runtime_sqlite = bytearray(source_sqlite)
    if len(runtime_sqlite) < 100:
        raise AIMDReconciliationError("v3 decoded SQLite is truncated")
    runtime_sqlite[18] = 1
    runtime_sqlite[19] = 1
    descriptor, raw_path = tempfile.mkstemp(
        prefix="do-v3-reconcile-", suffix=".sqlite3"
    )
    temporary_path = Path(raw_path)
    connection: sqlite3.Connection | None = None
    try:
        try:
            os.chmod(temporary_path, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(runtime_sqlite)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        uri = temporary_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        yield connection
    finally:
        try:
            if connection is not None:
                connection.close()
        finally:
            temporary_path.unlink(missing_ok=True)


def _verify_v3_reconciliation_attestation(v3_checkpoint_dir: Path) -> dict[str, Any]:
    manifest_path = v3_checkpoint_dir / "MANIFEST.json"
    ledger_path = v3_checkpoint_dir / "ledger.sqlite3.gz"
    if not manifest_path.is_file() or _sha256_file(manifest_path) != V3_MANIFEST_SHA256:
        raise AIMDReconciliationError("exact v3 checkpoint manifest hash mismatch")
    if not ledger_path.is_file() or _sha256_file(ledger_path) != V3_LEDGER_GZIP_SHA256:
        raise AIMDReconciliationError("exact v3 checkpoint ledger hash mismatch")
    try:
        source_sqlite = gzip.decompress(ledger_path.read_bytes())
    except (OSError, EOFError) as error:
        raise AIMDReconciliationError(
            "v3 checkpoint ledger cannot be decompressed"
        ) from error
    if hashlib.sha256(source_sqlite).hexdigest() != V3_LEDGER_DECODED_SHA256:
        raise AIMDReconciliationError("v3 decoded SQLite hash mismatch")
    manifest = _read_json(manifest_path)
    snapshot = manifest.get("snapshot")
    sqlite_attestation = manifest.get("sqlite")
    if not isinstance(snapshot, Mapping) or not isinstance(sqlite_attestation, Mapping):
        raise AIMDReconciliationError("v3 checkpoint lacks integrity attestations")
    if (
        manifest.get("checkpoint_id") != V3_CHECKPOINT_NAME
        or int(snapshot.get("attempt_count", -1)) != 985
        or snapshot.get("attempt_state_counts")
        != {"completed": 870, "failed": 72, "unknown": 43}
        or sqlite_attestation.get("integrity_check") != "ok"
        or int(sqlite_attestation.get("foreign_key_check_rows", -1)) != 0
        or sqlite_attestation.get("uncompressed_sha256") != V3_LEDGER_DECODED_SHA256
    ):
        raise AIMDReconciliationError("v3 checkpoint attestation mismatch")
    # The checkpoint was captured in WAL mode.  Change bytes 18/19 only in a
    # private temporary copy; the hash-verified source bytes remain untouched.
    with _open_hash_verified_sqlite_copy(source_sqlite) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise AIMDReconciliationError("v3 temporary-copy integrity_check failed")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise AIMDReconciliationError("v3 temporary-copy foreign_key_check failed")
        rows: list[dict[str, Any]] = []
        state_counts: dict[str, int] = {}
        usage_costs: list[float] = []
        reserve_costs: list[float] = []
        query = (
            "SELECT a.attempt_id,a.state,c.endpoint_id,a.metrics_json,c.payload_json "
            "FROM attempts AS a JOIN cells AS c "
            "ON a.run_id=c.run_id AND a.cell_id=c.cell_id ORDER BY a.attempt_id"
        )
        for (
            attempt_id,
            state,
            endpoint_id,
            metrics_json,
            payload_json,
        ) in connection.execute(query):
            if endpoint_id not in MODEL_BY_ID:
                raise AIMDReconciliationError(
                    "v3 attempt references an unknown endpoint"
                )
            state_counts[str(state)] = state_counts.get(str(state), 0) + 1
            try:
                metrics = json.loads(metrics_json or "{}")
                payload = json.loads(payload_json)
            except json.JSONDecodeError as error:
                raise AIMDReconciliationError("v3 attempt JSON is invalid") from error
            usage = metrics.get("usage") if isinstance(metrics, Mapping) else None
            reservation = (
                metrics.get("reservation") if isinstance(metrics, Mapping) else None
            )
            usage = usage if isinstance(usage, Mapping) else {}
            reservation = reservation if isinstance(reservation, Mapping) else {}
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            if state == "completed" and input_tokens > 0 and output_tokens > 0:
                basis = "server_usage"
            else:
                basis = "full_reservation"
                input_tokens = int(
                    reservation.get("selected_maximum_input_tokens")
                    or payload.get("reserved_input_tokens")
                    or 0
                )
                output_tokens = int(
                    reservation.get("selected_maximum_output_tokens")
                    or payload.get("requested_output_tokens")
                    or 0
                )
            if input_tokens <= 0 or output_tokens <= 0:
                raise AIMDReconciliationError(
                    "v3 attempt lacks conservative token bounds"
                )
            spec = MODEL_BY_ID[str(endpoint_id)]
            cost = (
                input_tokens * spec.input_usd_per_million
                + output_tokens * spec.output_usd_per_million
            ) / 1_000_000
            (usage_costs if basis == "server_usage" else reserve_costs).append(cost)
            rows.append(
                {
                    "attempt_id": attempt_id,
                    "endpoint_id": endpoint_id,
                    "state": state,
                    "basis": basis,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_usd_per_million": spec.input_usd_per_million,
                    "output_usd_per_million": spec.output_usd_per_million,
                    "repriced_cost_usd": cost,
                }
            )
    usage_cost = math.fsum(usage_costs)
    reserve_cost = math.fsum(reserve_costs)
    local_rows_hash = _sha256_json(rows)
    if (
        len(rows) != 985
        or state_counts != {"completed": 870, "failed": 72, "unknown": 43}
        or len(usage_costs) != 815
        or len(reserve_costs) != 170
        or not math.isclose(usage_cost, V3_USAGE_COST_USD, rel_tol=0, abs_tol=1e-12)
        or not math.isclose(
            reserve_cost, V3_MAX_RESERVE_COST_USD, rel_tol=0, abs_tol=1e-12
        )
        or local_rows_hash != V3_LOCALLY_RECOMPUTED_ROWS_SHA256
    ):
        raise AIMDReconciliationError("v3 row-level reconciliation drifted")
    components = usage_cost + reserve_cost + V3_LEGACY_TOPUP_REPRICED_USD
    if not math.isclose(components, V3_RECONCILED_UPPER_USD, rel_tol=0, abs_tol=1e-12):
        raise AIMDReconciliationError("v3 reconciliation component sum drifted")
    return {
        "checkpoint_id": V3_CHECKPOINT_NAME,
        "manifest_sha256": V3_MANIFEST_SHA256,
        "ledger_gzip_sha256": V3_LEDGER_GZIP_SHA256,
        "ledger_decoded_sha256": V3_LEDGER_DECODED_SHA256,
        "sqlite_integrity_check": "ok",
        "foreign_key_check_rows": 0,
        "attempt_count": 985,
        "attempt_state_counts": {"completed": 870, "failed": 72, "unknown": 43},
        "complete_usage_rows": 815,
        "max_reservation_rows": 170,
        "complete_usage_current_price_cost_usd": V3_USAGE_COST_USD,
        "failed_or_unknown_current_price_max_reservation_usd": V3_MAX_RESERVE_COST_USD,
        "legacy_unknown_topup_current_price_upper_usd": V3_LEGACY_TOPUP_REPRICED_USD,
        "reconciled_upper_usd": V3_RECONCILED_UPPER_USD,
        "canonical_reconciled_rows_sha256": V3_RECONCILED_ROWS_SHA256,
        "locally_recomputed_rows_sha256": local_rows_hash,
        "independent_compact_attestation_sha256": V3_ATTESTATION_SHA256,
        "derivation": (
            "completed attempts with strictly positive prompt and completion usage "
            "use current token prices; "
            "failed/unknown attempts use selected maximum input/output reservations; "
            "legacy unknown top-up is inflated by the maximum exact price-drift ratio"
        ),
        "source_bytes_mutated": False,
    }


def _reconcile_prior_lineage(
    prior_lineage_root: Path,
    *,
    v3_checkpoint_dir: Path,
    old_specs: Mapping[str, Mapping[str, Any]],
    new_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    v3 = _verify_v3_reconciliation_attestation(v3_checkpoint_dir)
    source_archives_dir = v3_checkpoint_dir.parent / "source-archives"

    runs: list[dict[str, Any]] = []
    durable_current_total = 0.0
    unknown_current_total = 0.0
    durable_complete_usage = 0
    durable_reservation_retained = 0
    missing_reservation_retained = 0
    previous_old_cumulative = OPAQUE_PREFIX_OLD_USD
    all_plan_ids: set[str] = set()
    all_record_ids: set[str] = set()
    all_unknown_ids: set[str] = set()
    for contract in PRIOR_LINEAGE:
        run_dir = prior_lineage_root / str(contract["name"])
        archive_name, archive_hash = PRIOR_SOURCE_ARCHIVES[str(contract["name"])]
        archive_path = source_archives_dir / archive_name
        if not archive_path.is_file() or _sha256_file(archive_path) != archive_hash:
            raise AIMDReconciliationError(
                "exact prior runner source archive hash mismatch"
            )
        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                script_member = archive.extractfile(
                    "scripts/run-digitalocean-inference-benchmark.py"
                )
                core_member = archive.extractfile("do_benchmark/core.py")
                if script_member is None or core_member is None:
                    raise AIMDReconciliationError(
                        "prior source archive lacks runner code"
                    )
                script_source = script_member.read().decode("utf-8")
                core_source = core_member.read().decode("utf-8")
        except (tarfile.TarError, UnicodeDecodeError, OSError) as error:
            raise AIMDReconciliationError(
                "prior runner source archive is invalid"
            ) from error
        if (
            'parser.add_argument("--safety-max-output-tokens", type=int, default=4096)'
            not in script_source
            or '"max_tokens": safety_max_output_tokens' not in core_source
        ):
            raise AIMDReconciliationError(
                "prior runner 4,096 output ceiling contract drifted"
            )
        hashes = contract["artifact_sha256"]
        for filename, expected_hash in hashes.items():
            path = run_dir / filename
            if not path.is_file() or _sha256_file(path) != expected_hash:
                raise AIMDReconciliationError(
                    f"exact prior-lineage artifact hash mismatch: {contract['name']}/{filename}"
                )
        manifest = _read_json(run_dir / "manifest.json")
        run_summary = _read_json(run_dir / "run-summary.json")
        plans = _read_jsonl(run_dir / "plan.jsonl", "cell_id")
        records = _read_jsonl(run_dir / "records.jsonl", "cell_id")
        if len(plans) != int(contract["planned"]) or len(records) != int(
            contract["durable"]
        ):
            raise AIMDReconciliationError(
                "prior-lineage planned/durable count mismatch"
            )
        unjournaled = run_summary.get("unjournaled_errors")
        if (
            int(run_summary.get("attempted_cells", -1)) != len(plans)
            or not isinstance(unjournaled, list)
            or len(unjournaled) != len(plans) - len(records)
        ):
            raise AIMDReconciliationError(
                "prior attempted/unjournaled counts do not reconcile"
            )
        if not set(records) <= set(plans):
            raise AIMDReconciliationError(
                "prior-lineage record is absent from its plan"
            )
        if all_plan_ids & set(plans) or all_record_ids & set(records):
            raise AIMDReconciliationError("prior-lineage cell IDs overlap across runs")
        all_plan_ids.update(plans)
        all_record_ids.update(records)
        missing = set(plans) - set(records)
        all_unknown_ids.update(missing)
        if len(missing) != int(contract["planned"]) - int(contract["durable"]):
            raise AIMDReconciliationError("prior-lineage unknown count mismatch")
        model_rows = manifest.get("models")
        if not isinstance(model_rows, list):
            raise AIMDReconciliationError("prior manifest lacks model contracts")
        for row in model_rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("model_id"), str):
                raise AIMDReconciliationError("invalid prior model contract")
            model_id = str(row["model_id"])
            if model_id not in old_specs or canonical_json(dict(row)) != canonical_json(
                old_specs[model_id]
            ):
                raise AIMDReconciliationError("prior/source model contracts differ")
        old_record_sum = 0.0
        run_durable_current = 0.0
        run_unknown_current = 0.0
        for cell_id, plan in plans.items():
            model_id = plan.get("model_id")
            if not isinstance(model_id, str) or model_id not in new_specs:
                raise AIMDReconciliationError("prior plan references an unknown model")
            reserve, _prompt_bound = _coverage_request_reserve(
                plan, new_specs[model_id]
            )
            record = records.get(cell_id)
            if record is None:
                run_unknown_current += reserve
                missing_reservation_retained += 1
                continue
            if record.get("model_id") != model_id:
                raise AIMDReconciliationError("prior plan/record model mismatch")
            old_record_sum += _finite_nonnegative(
                record.get("estimated_cost_usd"), "prior record cost"
            )
            usage = _complete_usage(record)
            if record.get("status") == "success" and usage is not None:
                current = _cost(new_specs[model_id], *usage)
                durable_complete_usage += 1
            else:
                current = reserve
                durable_reservation_retained += 1
            run_durable_current += current
        expected_old_cumulative = previous_old_cumulative + old_record_sum
        observed_old_cumulative = _finite_nonnegative(
            run_summary.get("estimated_cost_usd"), "prior cumulative cost"
        )
        if not math.isclose(
            expected_old_cumulative,
            observed_old_cumulative,
            rel_tol=1e-10,
            abs_tol=1e-9,
        ) or not math.isclose(
            observed_old_cumulative,
            float(contract["old_cumulative_exposure_usd"]),
            rel_tol=1e-10,
            abs_tol=1e-9,
        ):
            raise AIMDReconciliationError("prior cumulative chain does not reconcile")
        previous_old_cumulative = observed_old_cumulative
        durable_current_total += run_durable_current
        unknown_current_total += run_unknown_current
        runs.append(
            {
                "name": contract["name"],
                "artifact_sha256": dict(hashes),
                "runner_source_archive": {
                    "filename": archive_name,
                    "sha256": archive_hash,
                    "default_transport_output_ceiling_tokens": 4_096,
                    "transport_binding": (
                        "hash-bound source default; historical receipt names this runner "
                        "but did not serialize the numeric CLI value"
                    ),
                },
                "planned_count": len(plans),
                "durable_count": len(records),
                "unknown_plan_minus_record_count": len(missing),
                "plan_cell_ids_sha256": _sha256_json(sorted(plans)),
                "record_cell_ids_sha256": _sha256_json(sorted(records)),
                "unknown_cell_ids_sha256": _sha256_json(sorted(missing)),
                "old_record_cost_usd": old_record_sum,
                "old_cumulative_exposure_usd": observed_old_cumulative,
                "repriced_durable_cost_usd": run_durable_current,
                "repriced_unknown_full_reservation_usd": run_unknown_current,
            }
        )
    if len(all_unknown_ids) != 247:
        raise AIMDReconciliationError(
            "prior lineage does not bind exactly 247 unknown IDs"
        )
    reconciled = V3_RECONCILED_UPPER_USD + durable_current_total + unknown_current_total
    return {
        "v3_reconciliation": v3,
        "runs": runs,
        "all_plan_cell_ids_sha256": _sha256_json(sorted(all_plan_ids)),
        "all_record_cell_ids_sha256": _sha256_json(sorted(all_record_ids)),
        "all_unknown_cell_ids_sha256": _sha256_json(sorted(all_unknown_ids)),
        "all_plan_count": len(all_plan_ids),
        "all_durable_count": len(all_record_ids),
        "all_unknown_count": len(all_unknown_ids),
        "complete_successes_settled_from_usage": durable_complete_usage,
        "durable_incomplete_or_error_full_reservations": durable_reservation_retained,
        "unknown_full_reservations": missing_reservation_retained,
        "repriced_durable_cost_usd": durable_current_total,
        "repriced_unknown_full_reservation_usd": unknown_current_total,
        "reconciled_prior_exposure_usd": reconciled,
        "unknown_ids_must_never_be_replayed": True,
        "reservation_rule": (
            "max(UTF-8 canonical serialized request bytes + 512, "
            "ceil(planned_input_tokens * 1.5)) prompt tokens plus 4,096 output tokens"
        ),
        "reservation_bound_status": (
            "conditional on the hash-bound runner's 4,096-token source default; "
            "the historical invocation receipt omitted the numeric CLI value"
        ),
    }


def build_reconciliation_receipt(
    aimd_dir: Path,
    *,
    endpoint_freeze_path: Path,
    prior_lineage_root: Path,
    v3_checkpoint_dir: Path,
    source_archive_path: Path | None = None,
    require_source_archive: bool = False,
) -> dict[str, Any]:
    """Recompute the deterministic receipt for the exact completed AIMD run."""

    paths = {name: aimd_dir / filename for name, filename in ARTIFACT_FILES.items()}
    for name, path in paths.items():
        if not path.is_file() or _sha256_file(path) != SOURCE_ARTIFACT_SHA256[name]:
            raise AIMDReconciliationError(
                f"exact source artifact hash mismatch: {name}"
            )
    if require_source_archive and source_archive_path is None:
        raise AIMDReconciliationError("source archive is required to mint a receipt")
    if source_archive_path is not None:
        if not source_archive_path.is_file():
            raise AIMDReconciliationError("source archive is missing")
        if _sha256_file(source_archive_path) != SOURCE_ARCHIVE_SHA256:
            raise AIMDReconciliationError("source archive hash mismatch")
    if not endpoint_freeze_path.is_file():
        raise AIMDReconciliationError("current endpoint freeze is missing")

    manifest = _read_json(paths["manifest"])
    summary = _read_json(paths["summary"])
    epochs = _read_jsonl(paths["epochs"], "epoch_id")
    requests = _read_jsonl(paths["requests"], "request_id")
    reservations = _read_jsonl(paths["reservations"], "request_id")
    if (
        manifest.get("campaign_id") != SOURCE_CAMPAIGN_ID
        or summary.get("campaign_id") != SOURCE_CAMPAIGN_ID
    ):
        raise AIMDReconciliationError("source campaign identity mismatch")
    if summary.get("all_models_complete") is not True or summary.get("status") not in {
        "complete",
        "complete_right_censored",
    }:
        raise AIMDReconciliationError("source campaign is not scientifically complete")
    for label, rows in (
        ("epochs", epochs),
        ("requests", requests),
        ("reservations", reservations),
    ):
        if len(rows) != SOURCE_COUNTS[label]:
            raise AIMDReconciliationError(f"source {label} count mismatch")
        if any(row.get("campaign_id") != SOURCE_CAMPAIGN_ID for row in rows.values()):
            raise AIMDReconciliationError(f"foreign campaign row in {label}")
    if set(requests) != set(reservations):
        raise AIMDReconciliationError("final request/reservation ID sets differ")
    if any(row.get("epoch_id") not in epochs for row in requests.values()):
        raise AIMDReconciliationError("request references an absent epoch")
    if int(summary.get("epoch_rows", -1)) != len(epochs) or int(
        summary.get("request_rows", -1)
    ) != len(requests):
        raise AIMDReconciliationError("summary row counts do not bind final journals")
    observed_identity = {
        key: manifest.get(key)
        for key in (
            "input_tokens",
            "long_output_words",
            "short_max_output_tokens",
            "long_max_output_tokens",
            "mixed_max_output_tokens",
            "shapes",
        )
    }
    if canonical_json(observed_identity) != canonical_json(SOURCE_IDENTITY):
        raise AIMDReconciliationError("source workload/payload identity drifted")

    old_specs, new_specs, transformations = _model_contracts(manifest)
    prior_lineage = _reconcile_prior_lineage(
        prior_lineage_root,
        v3_checkpoint_dir=v3_checkpoint_dir,
        old_specs=old_specs,
        new_specs=new_specs,
    )
    source_prior = _finite_nonnegative(manifest.get("prior_cost_usd"), "source prior")
    if not math.isclose(
        source_prior,
        _finite_nonnegative(summary.get("prior_cost_usd"), "summary prior"),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise AIMDReconciliationError("source prior exposure mismatch")
    if not math.isclose(
        source_prior,
        float(PRIOR_LINEAGE[-1]["old_cumulative_exposure_usd"]),
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise AIMDReconciliationError(
            "AIMD prior does not continue the bound prior lineage"
        )

    original_terminal = 0.0
    repriced_terminal = 0.0
    actual_usage_count = 0
    retained_reservation_count = 0
    for request_id, request in requests.items():
        reservation = reservations[request_id]
        for field, request_field in (
            ("request_id", "request_id"),
            ("epoch_id", "epoch_id"),
            ("model_id", "model_id"),
            ("shape", "shape"),
            ("max_output_tokens", "requested_max_output_tokens"),
        ):
            if canonical_json(reservation.get(field)) != canonical_json(
                request.get(request_field)
            ):
                raise AIMDReconciliationError(
                    f"request/reservation identity mismatch for {field}"
                )
        if request.get("provider_send_attempted") is not True:
            raise AIMDReconciliationError("final request was not provider-attempted")
        model_id = request.get("model_id")
        if not isinstance(model_id, str) or model_id not in old_specs:
            raise AIMDReconciliationError("request references an unknown model")
        prompt_reserve = reservation.get("reserved_prompt_tokens")
        output_reserve = reservation.get("max_output_tokens")
        if (
            not isinstance(prompt_reserve, int)
            or prompt_reserve < 0
            or not isinstance(output_reserve, int)
            or output_reserve < 0
        ):
            raise AIMDReconciliationError("invalid reservation token bounds")
        old_reserved = _cost(old_specs[model_id], prompt_reserve, output_reserve)
        new_reserved = _cost(new_specs[model_id], prompt_reserve, output_reserve)
        if not math.isclose(
            _finite_nonnegative(reservation.get("reserved_cost_usd"), "reserved cost"),
            old_reserved,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise AIMDReconciliationError("source reservation is under/over-accounted")
        if (
            not math.isclose(
                _finite_nonnegative(
                    request.get("worst_case_reserved_cost_usd"), "request reservation"
                ),
                old_reserved,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or request.get("reserved_prompt_tokens") != prompt_reserve
        ):
            raise AIMDReconciliationError(
                "request does not retain its source reservation"
            )
        historical_usage = _historical_usage(request)
        strict_usage = _complete_usage(request)
        if request.get("status") == "success" and historical_usage is not None:
            old_accounted = _cost(old_specs[model_id], *historical_usage)
        else:
            old_accounted = old_reserved
        if request.get("status") == "success" and strict_usage is not None:
            new_accounted = _cost(new_specs[model_id], *strict_usage)
            actual_usage_count += 1
        else:
            new_accounted = new_reserved
            retained_reservation_count += 1
        observed_accounted = _finite_nonnegative(
            request.get("accounted_cost_usd"), "terminal accounted cost"
        )
        if not math.isclose(
            observed_accounted, old_accounted, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise AIMDReconciliationError("source terminal settlement is inconsistent")
        _require_no_under_settlement(old_accounted, new_accounted)
        original_terminal += old_accounted
        repriced_terminal += new_accounted

    original_exposure = source_prior + original_terminal
    summary_exposure = _finite_nonnegative(
        summary.get("conservative_exposure_usd"), "source summary exposure"
    )
    if not math.isclose(
        original_exposure, summary_exposure, rel_tol=1e-9, abs_tol=1e-8
    ):
        raise AIMDReconciliationError("source cumulative exposure does not reconcile")
    reconciled_prior = float(prior_lineage["reconciled_prior_exposure_usd"])
    reconciled_exposure = reconciled_prior + repriced_terminal
    if reconciled_exposure + 1e-12 < summary_exposure:
        raise AIMDReconciliationError("reconciled exposure would under-settle source")

    endpoint_freeze = _read_json(endpoint_freeze_path)
    frozen_rows = {
        str(row.get("model_id")): row
        for row in endpoint_freeze.get("endpoints", [])
        if isinstance(row, Mapping) and isinstance(row.get("model_id"), str)
    }
    for model_id, contract in new_specs.items():
        frozen = frozen_rows.get(model_id)
        if frozen is None:
            raise AIMDReconciliationError(
                f"model absent from endpoint freeze: {model_id}"
            )
        for field in ("input_usd_per_million", "output_usd_per_million"):
            if not math.isclose(
                float(frozen[field]), float(contract[field]), rel_tol=0, abs_tol=1e-12
            ):
                raise AIMDReconciliationError(
                    f"current model/freeze price mismatch: {model_id}/{field}"
                )
        frozen_context = frozen.get("context_window")
        if model_id == "kimi-k3":
            if frozen_context is not None or int(contract["context_window"]) != 65_536:
                raise AIMDReconciliationError("Kimi probe-anchor contract drifted")
        elif frozen_context is None or int(frozen_context) != int(
            contract["context_window"]
        ):
            raise AIMDReconciliationError(
                f"current model/freeze context mismatch: {model_id}"
            )

    body: dict[str, Any] = {
        "schema_version": RECONCILIATION_SCHEMA,
        "policy_version": RECONCILIATION_POLICY,
        "transformation_scope": "cost settlement only; performance rows are immutable",
        "source": {
            "campaign_id": SOURCE_CAMPAIGN_ID,
            "git_commit": SOURCE_GIT_COMMIT,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "artifact_sha256": dict(SOURCE_ARTIFACT_SHA256),
            "counts": dict(SOURCE_COUNTS),
            "request_ids_sha256": _sha256_json(sorted(requests)),
            "reservation_ids_sha256": _sha256_json(sorted(reservations)),
            "epoch_ids_sha256": _sha256_json(sorted(epochs)),
            "workload_identity": dict(SOURCE_IDENTITY),
            "model_contracts": [old_specs[key] for key in sorted(old_specs)],
        },
        "target": {
            "endpoint_freeze_sha256": _sha256_file(endpoint_freeze_path),
            "endpoint_freeze_schema_version": endpoint_freeze.get("schema_version"),
            "model_contracts": [new_specs[key] for key in sorted(new_specs)],
        },
        "transformations": transformations,
        "prior_lineage": prior_lineage,
        "performance_evidence": {
            "preserved": True,
            "reason": (
                "payload/workload identities are hash-bound; the only non-price drift "
                "is a context expansion and every fixed long prompt is 32,000 tokens"
            ),
        },
        "settlement": {
            "source_prior_exposure_usd": source_prior,
            "reconciled_prior_exposure_usd": reconciled_prior,
            "source_terminal_cost_usd": original_terminal,
            "source_cumulative_exposure_usd": original_exposure,
            "repriced_terminal_cost_usd": repriced_terminal,
            "reconciled_cumulative_exposure_usd": reconciled_exposure,
            "complete_successes_settled_from_usage": actual_usage_count,
            "incomplete_or_error_outcomes_retaining_full_reservation": (
                retained_reservation_count
            ),
            "orphan_reservations_retaining_full_reservation": 0,
            "settlement_rule": (
                "success with complete prompt+completion usage uses current frozen "
                "token prices; every incomplete/error outcome retains the full "
                "reservation repriced at current frozen prices"
            ),
            "under_settlement_allowed": False,
        },
        "sanitization": {
            "contains_credentials": False,
            "contains_prompts_or_outputs": False,
            "contains_raw_headers_or_response_bodies": False,
            "identifiers_bound_by_count_and_sha256_only": True,
        },
    }
    return {**body, "receipt_sha256": _sha256_json(body)}


def verify_reconciliation_receipt(
    receipt_path: Path,
    aimd_dir: Path,
    *,
    endpoint_freeze_path: Path,
    prior_lineage_root: Path,
    v3_checkpoint_dir: Path,
) -> dict[str, Any]:
    """Verify a receipt by recomputing it from immutable source evidence."""

    observed = _read_json(receipt_path)
    expected = build_reconciliation_receipt(
        aimd_dir,
        endpoint_freeze_path=endpoint_freeze_path,
        prior_lineage_root=prior_lineage_root,
        v3_checkpoint_dir=v3_checkpoint_dir,
        source_archive_path=None,
        require_source_archive=False,
    )
    if canonical_json(observed) != canonical_json(expected):
        raise AIMDReconciliationError("reconciliation receipt is stale or tampered")
    return expected


def write_reconciliation_receipt(
    aimd_dir: Path,
    output_path: Path,
    *,
    endpoint_freeze_path: Path,
    prior_lineage_root: Path,
    v3_checkpoint_dir: Path,
    source_archive_path: Path,
) -> dict[str, Any]:
    """Mint the receipt only after independently verifying the source archive."""

    receipt = build_reconciliation_receipt(
        aimd_dir,
        endpoint_freeze_path=endpoint_freeze_path,
        prior_lineage_root=prior_lineage_root,
        v3_checkpoint_dir=v3_checkpoint_dir,
        source_archive_path=source_archive_path,
        require_source_archive=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
