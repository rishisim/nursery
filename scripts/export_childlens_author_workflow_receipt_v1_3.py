#!/usr/bin/env python3
"""Export the repository-safe aggregate v1.3 author-workflow receipt.

This zero-argument exporter discovers the already initialized owner-private
workflow, reads only aggregate progress/state through the controller, and
writes the fixed public receipt.  It never receives or emits restricted paths,
rows, text, media, identifiers, timings, predictions, or credentials.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import childlens_author_audit_v1_3 as workflow


OUTPUT_PATH = (
    workflow.REPO_ROOT
    / "output/childlens_feasibility_v1_3/author_workflow_receipt.json"
)

REQUIRED_TRUE = (
    "workflow_ready",
    "loopback_only",
    "owner_private",
    "autosave",
    "autosave_to_quarantine_only",
    "resumable",
    "predictions_hidden_before_author_lock",
    "author_record_lock_immutable",
    "comparison_requires_lock",
    "selected_independent_of_model_outputs",
    "sample_independent_of_model_outputs",
    "uncertain_unusable_routes",
    "uncertain_route_available",
    "unusable_route_available",
    "qualification_instruction_present",
)
REQUIRED_FALSE = (
    "external_hosting",
    "network_exposure",
    "model_predictions_revealed_before_lock",
    "human_evidence_fabricated",
    "inter_human_reliability_available",
    "unaudited_pseudo_label_primary_evaluation_truth",
)


def _validate_public_shape(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != workflow.RECEIPT_VERSION
        or value.get("status") != "READY"
        or value.get("route") != workflow.ROUTE
        or value.get("audit_item_count") != workflow.FROZEN_ITEM_COUNT
        or value.get("audit_speech_seconds")
        != workflow.FROZEN_TOTAL_DURATION_MS // 1000
        or value.get("audit_speech_minutes")
        != workflow.FROZEN_TOTAL_DURATION_MS / 60_000
        or any(value.get(key) is not True for key in REQUIRED_TRUE)
        or any(value.get(key) is not False for key in REQUIRED_FALSE)
    ):
        raise workflow.WorkflowError("E_PUBLIC_RECEIPT_SHAPE")
    estimate = value.get("estimated_author_minutes")
    if not isinstance(estimate, int) or isinstance(estimate, bool) or not 1 <= estimate <= 480:
        raise workflow.WorkflowError("E_PUBLIC_RECEIPT_SHAPE")
    if value.get("primary_evaluation_truth") != "SIMULATOR_ORACLE_ONLY":
        raise workflow.WorkflowError("E_PUBLIC_RECEIPT_SHAPE")


def _write_public(value: Mapping[str, Any]) -> None:
    output = OUTPUT_PATH
    output.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    encoded = workflow._canonical(value) + b"\n"
    temporary = output.parent / f".author-workflow-{secrets.token_hex(8)}.tmp"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        os.chmod(output, 0o644)
    except OSError:
        raise workflow.WorkflowError("E_PUBLIC_RECEIPT_WRITE")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def export_receipt() -> Mapping[str, Any]:
    root = workflow.discover_runtime_root()
    receipt = workflow.aggregate_receipt(root)
    _validate_public_shape(receipt)
    _write_public(receipt)
    return {
        "status": "AUTHOR_WORKFLOW_RECEIPT_EXPORTED",
        "workflow_status": receipt["status"],
        "audit_item_count": receipt["audit_item_count"],
        "audit_speech_seconds": receipt["audit_speech_seconds"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    supplied = list(sys.argv[1:] if argv is None else argv)
    if supplied:
        print("E_ARGUMENTS", file=sys.stderr)
        return 2
    try:
        result = export_receipt()
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except workflow.WorkflowError as exc:
        print(exc.code, file=sys.stderr)
        return 2
    except Exception:
        print("E_PUBLIC_RECEIPT_INTERNAL", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
