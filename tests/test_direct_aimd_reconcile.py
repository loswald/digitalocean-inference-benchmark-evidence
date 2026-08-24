from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pytest

import do_benchmark.direct_aimd_reconcile as reconcile
from do_benchmark.core import MODEL_BY_ID


def _legacy_manifest() -> dict:
    rows = {model_id: asdict(spec) for model_id, spec in MODEL_BY_ID.items()}
    for model_id, fields in reconcile.ALLOWED_CONTRACT_TRANSFORMATIONS.items():
        for field, (old, _new) in fields.items():
            rows[model_id][field] = old
    return {"model_specs": list(rows.values())}


def test_exact_contract_drift_allowlist_and_context_expansion() -> None:
    old, new, transformations = reconcile._model_contracts(_legacy_manifest())
    assert len(old) == len(new) == 12
    assert len(transformations) == 9
    qwen = [row for row in transformations if row["model_id"] == "qwen3.8-max"]
    assert qwen == [
        {
            "model_id": "qwen3.8-max",
            "field": "context_window",
            "old": 262_144,
            "new": 1_000_000,
            "kind": "context_expansion_compatible_with_32k_payload",
        }
    ]


@pytest.mark.parametrize(
    ("model_id", "field", "value"),
    [
        ("deepseek-v4-flash-0731", "input_usd_per_million", 999.0),
        ("qwen3.8-max", "context_window", 16_000),
    ],
)
def test_arbitrary_price_or_context_drift_fails_closed(
    model_id: str, field: str, value: object
) -> None:
    manifest = _legacy_manifest()
    row = next(item for item in manifest["model_specs"] if item["model_id"] == model_id)
    row[field] = value
    with pytest.raises(
        reconcile.AIMDReconciliationError, match="unapproved|payload-compatible"
    ):
        reconcile._model_contracts(manifest)


def test_prompt_only_usage_retains_full_reservation() -> None:
    prompt_only = {
        "usage": {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100}
    }
    assert reconcile._historical_usage(prompt_only) == (100, 0)
    assert reconcile._complete_usage(prompt_only) is None
    assert reconcile._complete_usage(
        {"usage": {"prompt_tokens": 100, "completion_tokens": 1}}
    ) == (100, 1)


def test_under_settlement_is_rejected() -> None:
    reconcile._require_no_under_settlement(1.0, 1.0)
    reconcile._require_no_under_settlement(1.0, 1.1)
    with pytest.raises(reconcile.AIMDReconciliationError, match="under-settle"):
        reconcile._require_no_under_settlement(1.0, 0.999)


def test_strict_prior_reservation_binds_payload_and_4096_output() -> None:
    plan = {
        "model_id": "deepseek-v4-flash-0731",
        "task": {
            "messages": [{"role": "user", "content": "small"}],
            "parameters": {},
            "tools": [],
            "tool_choice": None,
            "response_format": None,
            "metadata": {"planned_input_tokens": 100_000},
        },
    }
    spec = asdict(MODEL_BY_ID["deepseek-v4-flash-0731"])
    cost, prompt_bound = reconcile._coverage_request_reserve(plan, spec)
    assert prompt_bound == 150_000
    assert cost == pytest.approx((150_000 * 0.08 + 4_096 * 0.252) / 1_000_000)


def test_sqlite_attestation_copy_does_not_require_deserialize(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "source.sqlite3"
    source_connection = sqlite3.connect(source_path)
    source_connection.execute("CREATE TABLE evidence(value INTEGER NOT NULL)")
    source_connection.execute("INSERT INTO evidence VALUES (7)")
    source_connection.commit()
    source_connection.close()
    source_bytes = source_path.read_bytes()
    source_sha256 = reconcile.hashlib.sha256(source_bytes).hexdigest()
    temporary_path = tmp_path / "private-runtime-copy.sqlite3"
    real_connect = sqlite3.connect
    observed_connects: list[tuple[str, bool]] = []

    class ConnectionWithoutDeserialize:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def execute(self, *args, **kwargs):
            return self._connection.execute(*args, **kwargs)

        def close(self) -> None:
            self._connection.close()

    def mkstemp(*, prefix: str, suffix: str) -> tuple[int, str]:
        descriptor = os.open(
            temporary_path,
            os.O_CREAT | os.O_EXCL | os.O_RDWR,
            0o600,
        )
        return descriptor, str(temporary_path)

    def connect(database, *args, **kwargs):
        observed_connects.append((str(database), bool(kwargs.get("uri"))))
        return ConnectionWithoutDeserialize(real_connect(database, *args, **kwargs))

    monkeypatch.setattr(reconcile.tempfile, "mkstemp", mkstemp)
    monkeypatch.setattr(reconcile.sqlite3, "connect", connect)
    with reconcile._open_hash_verified_sqlite_copy(source_bytes) as connection:
        assert connection.execute("SELECT value FROM evidence").fetchone() == (7,)
        assert not hasattr(connection, "deserialize")
    assert reconcile.hashlib.sha256(source_bytes).hexdigest() == source_sha256
    assert not temporary_path.exists()
    assert len(observed_connects) == 1
    assert observed_connects[0][0] != ":memory:"
    assert observed_connects[0][0].endswith("?mode=ro")
    assert observed_connects[0][1] is True


def test_sqlite_attestation_copy_is_deleted_when_close_raises(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "source.sqlite3"
    source_connection = sqlite3.connect(source_path)
    source_connection.execute("CREATE TABLE evidence(value INTEGER NOT NULL)")
    source_connection.commit()
    source_connection.close()
    temporary_path = tmp_path / "private-runtime-copy.sqlite3"
    real_connect = sqlite3.connect

    class CloseRaisesConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def execute(self, *args, **kwargs):
            return self._connection.execute(*args, **kwargs)

        def close(self) -> None:
            self._connection.close()
            raise RuntimeError("synthetic close failure")

    def mkstemp(*, prefix: str, suffix: str) -> tuple[int, str]:
        descriptor = os.open(
            temporary_path,
            os.O_CREAT | os.O_EXCL | os.O_RDWR,
            0o600,
        )
        return descriptor, str(temporary_path)

    def connect(database, *args, **kwargs):
        return CloseRaisesConnection(real_connect(database, *args, **kwargs))

    monkeypatch.setattr(reconcile.tempfile, "mkstemp", mkstemp)
    monkeypatch.setattr(reconcile.sqlite3, "connect", connect)
    with pytest.raises(RuntimeError, match="synthetic close failure"):
        with reconcile._open_hash_verified_sqlite_copy(
            source_path.read_bytes()
        ) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert not temporary_path.exists()


def test_receipt_tampering_is_rejected(tmp_path: Path, monkeypatch) -> None:
    body = {"schema_version": reconcile.RECONCILIATION_SCHEMA, "value": 1}
    expected = {**body, "receipt_sha256": reconcile._sha256_json(body)}
    path = tmp_path / "reconciliation.json"
    path.write_text(json.dumps({**expected, "value": 2}), encoding="utf-8")
    monkeypatch.setattr(
        reconcile,
        "build_reconciliation_receipt",
        lambda *args, **kwargs: expected,
    )
    with pytest.raises(reconcile.AIMDReconciliationError, match="stale or tampered"):
        reconcile.verify_reconciliation_receipt(
            path,
            tmp_path,
            endpoint_freeze_path=tmp_path / "freeze.json",
            prior_lineage_root=tmp_path,
            v3_checkpoint_dir=tmp_path,
        )


def test_exact_counts_hashes_and_no_replay_contract_are_frozen() -> None:
    assert reconcile.SOURCE_COUNTS == {
        "epochs": 554,
        "requests": 6_253,
        "reservations": 6_253,
    }
    assert all(len(value) == 64 for value in reconcile.SOURCE_ARTIFACT_SHA256.values())
    assert (
        sum(
            int(run["planned"]) - int(run["durable"]) for run in reconcile.PRIOR_LINEAGE
        )
        == 247
    )
    assert math.isclose(
        reconcile.V3_USAGE_COST_USD
        + reconcile.V3_MAX_RESERVE_COST_USD
        + reconcile.V3_LEGACY_TOPUP_REPRICED_USD,
        reconcile.V3_RECONCILED_UPPER_USD,
        rel_tol=0,
        abs_tol=1e-12,
    )
