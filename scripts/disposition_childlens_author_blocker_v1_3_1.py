#!/usr/bin/env python3
"""Invalidate the attempted v1.3 author pass without exposing restricted state.

This zero-argument operation discovers the already-authorized quarantine through
the v1.3 controller, verifies that predictions were never revealed, preserves
all partial rows, and makes those rows permanently ineligible for scientific
comparison.  Only a fixed aggregate receipt is exported to the repository.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import childlens_author_audit_v1_3 as workflow  # noqa: E402


VERSION = "childlens-author-attempt-disposition-v1.3.1"
OUTPUT = REPO_ROOT / "output/childlens_feasibility_v1_3_1/author_attempt_invalidation_receipt.json"
PRIVATE_RECEIPT = "author_attempt_disposition_v1_3_1.json"
STATUS = "INVALID_FOR_SCIENTIFIC_COMPARISON"
REASONS = ["LANGUAGE_NOT_QUALIFIED", "UI_WINDOW_SCOPE_AMBIGUOUS"]

INVALIDATION_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS v1_3_1_block_item_labels_insert
    BEFORE INSERT ON item_labels BEGIN SELECT RAISE(ABORT,'E_ATTEMPT_INVALIDATED'); END;
CREATE TRIGGER IF NOT EXISTS v1_3_1_block_item_labels_update
    BEFORE UPDATE ON item_labels BEGIN SELECT RAISE(ABORT,'E_ATTEMPT_INVALIDATED'); END;
CREATE TRIGGER IF NOT EXISTS v1_3_1_block_item_labels_delete
    BEFORE DELETE ON item_labels BEGIN SELECT RAISE(ABORT,'E_ATTEMPT_INVALIDATED'); END;
CREATE TRIGGER IF NOT EXISTS v1_3_1_block_utterances_insert
    BEFORE INSERT ON author_utterances BEGIN SELECT RAISE(ABORT,'E_ATTEMPT_INVALIDATED'); END;
CREATE TRIGGER IF NOT EXISTS v1_3_1_block_utterances_update
    BEFORE UPDATE ON author_utterances BEGIN SELECT RAISE(ABORT,'E_ATTEMPT_INVALIDATED'); END;
CREATE TRIGGER IF NOT EXISTS v1_3_1_block_utterances_delete
    BEFORE DELETE ON author_utterances BEGIN SELECT RAISE(ABORT,'E_ATTEMPT_INVALIDATED'); END;
CREATE TRIGGER IF NOT EXISTS v1_3_1_block_mentions_insert
    BEFORE INSERT ON author_mentions BEGIN SELECT RAISE(ABORT,'E_ATTEMPT_INVALIDATED'); END;
CREATE TRIGGER IF NOT EXISTS v1_3_1_block_mentions_update
    BEFORE UPDATE ON author_mentions BEGIN SELECT RAISE(ABORT,'E_ATTEMPT_INVALIDATED'); END;
CREATE TRIGGER IF NOT EXISTS v1_3_1_block_mentions_delete
    BEFORE DELETE ON author_mentions BEGIN SELECT RAISE(ABORT,'E_ATTEMPT_INVALIDATED'); END;
CREATE TRIGGER IF NOT EXISTS v1_3_1_block_workflow_meta_update
    BEFORE UPDATE ON workflow_meta BEGIN SELECT RAISE(ABORT,'E_ATTEMPT_INVALIDATED'); END;
"""


class DispositionError(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if mode == 0o600 else 0o755)
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}")
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        os.chmod(path, mode)
    except Exception:
        try:
            pending.unlink()
        except OSError:
            pass
        raise


def _public_receipt(*, partial_present: bool, disposition_hmac: str) -> dict[str, object]:
    return {
        "schema_version": VERSION,
        "status": STATUS,
        "reason_codes": REASONS,
        "partial_records_preserved_in_quarantine": partial_present,
        "partial_record_counts_exported": False,
        "predictions_remained_blinded": True,
        "prediction_join_enabled": False,
        "author_pass_globally_locked": False,
        "model_human_comparison_executed": False,
        "partial_labels_allowed_to_contribute_to_any_gate": False,
        "translation_as_gold_allowed": False,
        "model_model_agreement_allowed_as_human_validation": False,
        "restricted_payload_exported": False,
        "disposition_hmac_sha256": disposition_hmac,
    }


def execute() -> dict[str, object]:
    root = workflow.discover_runtime_root()
    root, database, secret_path = workflow._workflow_paths(root)
    secret = workflow._load_or_create_secret(secret_path)
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        meta = connection.execute(
            "SELECT workflow_state,author_pass_locked_at_utc,author_record_hmac,"
            "prediction_binding_digest,prediction_store_relpath,prediction_store_sha256 "
            "FROM workflow_meta WHERE singleton=1"
        ).fetchone()
        if meta is None:
            raise DispositionError("E_NOT_INITIALIZED")
        if (
            meta["workflow_state"] != "AUTHOR_BLIND_OPEN"
            or meta["author_pass_locked_at_utc"] is not None
            or meta["author_record_hmac"] is not None
            or any(meta[key] is not None for key in (
                "prediction_binding_digest", "prediction_store_relpath", "prediction_store_sha256"
            ))
        ):
            raise DispositionError("E_BLINDING_PRECONDITION")
        prior_comparison = REPO_ROOT / "output/childlens_feasibility_v1_3/model_human_comparison_receipt.json"
        if prior_comparison.exists():
            raise DispositionError("E_COMPARISON_RECEIPT_EXISTS")
        partial_present = any(
            connection.execute(f"SELECT EXISTS(SELECT 1 FROM {table} LIMIT 1)").fetchone()[0] == 1
            for table in ("item_labels", "author_utterances", "author_mentions")
        )
        private = {
            "schema_version": VERSION,
            "status": STATUS,
            "reason_codes": REASONS,
            "recorded_at_utc": _utc_now(),
            "partial_records_preserved": partial_present,
            "gate_eligible": False,
            "predictions_revealed": False,
            "author_pass_locked": False,
            "comparison_executed": False,
        }
        disposition_hmac = hmac.new(secret, _canonical(private), hashlib.sha256).hexdigest()
        private["record_hmac_sha256"] = disposition_hmac
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS v1_3_1_attempt_disposition (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                schema_version TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status='INVALID_FOR_SCIENTIFIC_COMPARISON'),
                reason_language_not_qualified INTEGER NOT NULL CHECK(reason_language_not_qualified=1),
                reason_ui_window_scope_ambiguous INTEGER NOT NULL CHECK(reason_ui_window_scope_ambiguous=1),
                partial_records_preserved INTEGER NOT NULL CHECK(partial_records_preserved IN (0,1)),
                gate_eligible INTEGER NOT NULL CHECK(gate_eligible=0),
                predictions_revealed INTEGER NOT NULL CHECK(predictions_revealed=0),
                author_pass_locked INTEGER NOT NULL CHECK(author_pass_locked=0),
                comparison_executed INTEGER NOT NULL CHECK(comparison_executed=0),
                recorded_at_utc TEXT NOT NULL,
                record_hmac_sha256 TEXT NOT NULL
            );
            """
        )
        connection.executescript(INVALIDATION_TRIGGER_SQL)
        existing = connection.execute("SELECT record_hmac_sha256 FROM v1_3_1_attempt_disposition WHERE singleton=1").fetchone()
        if existing is None:
            connection.execute(
                "INSERT INTO v1_3_1_attempt_disposition VALUES(1,?,?,1,1,?,0,0,0,0,?,?)",
                (VERSION, STATUS, int(partial_present), private["recorded_at_utc"], disposition_hmac),
            )
            connection.execute(
                "INSERT INTO audit_log(created_at_utc,event) VALUES(?,?)",
                (private["recorded_at_utc"], "V1_3_1_ATTEMPT_INVALIDATED_LANGUAGE_AND_WINDOW_SCOPE"),
            )
        elif existing[0] != disposition_hmac:
            # Idempotency uses the already sealed private record, not a new timestamp.
            sealed_path = root / workflow.WORKFLOW_DIR / PRIVATE_RECEIPT
            sealed = json.loads(sealed_path.read_text(encoding="utf-8")) if sealed_path.is_file() else None
            if not isinstance(sealed, dict) or sealed.get("record_hmac_sha256") != existing[0]:
                raise DispositionError("E_DISPOSITION_CONFLICT")
            disposition_hmac = existing[0]
        connection.commit()
    private_path = root / workflow.WORKFLOW_DIR / PRIVATE_RECEIPT
    if not private_path.exists():
        _atomic(private_path, _canonical(private) + b"\n", 0o600)
    receipt = _public_receipt(partial_present=partial_present, disposition_hmac=disposition_hmac)
    _atomic(OUTPUT, json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n", 0o644)
    return receipt


if __name__ == "__main__":
    try:
        print(json.dumps(execute(), sort_keys=True))
    except (DispositionError, workflow.WorkflowError) as exc:
        print(json.dumps({"status": "BLOCKED", "code": getattr(exc, "code", str(exc))}), file=sys.stderr)
        raise SystemExit(2)
