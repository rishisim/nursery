#!/usr/bin/env python3
"""Create a blinded, restricted human-validation packet skeleton."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


VERSION = "childlens-human-validation-packet-v1.1.0"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def prepare(plan_path: Path, output_path: Path) -> dict[str, object]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    parent = output_path.parent.resolve(strict=True)
    if parent == REPO_ROOT or REPO_ROOT in parent.parents:
        raise ValueError("E_OUTPUT_INSIDE_REPOSITORY")
    items = [
        {
            "blinded_item_key": row["media_key"],
            "acquisition_status": "PENDING_EXACT_REMOTE_BYTES",
            "language_judgment": None,
            "audio_integrity": None,
            "utterance_timing_text_role_review": None,
            "referential_status_review": None,
            "adjudication_status": "NOT_STARTED",
        }
        for row in sorted(plan["selected"], key=lambda value: value["selection_hash"])
    ]
    packet = {
        "schema_version": VERSION,
        "packet_status": "SKELETON_AWAITING_MEDIA_AND_HUMANS",
        "pilot_selection_sha256": plan["pilot_selection_sha256"],
        "selected_count": len(items),
        "human_judgment_required": True,
        "codex_may_act_as_human": False,
        "timing_and_speaker_items_double_coded": True,
        "referential_items_minimum_double_code_fraction": 0.2,
        "learner_training_permitted": False,
        "items": items,
    }
    encoded = _canonical(packet) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return {
        "status": "ok",
        "selected_count": len(items),
        "packet_sha256": hashlib.sha256(_canonical(packet)).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        receipt = prepare(Path(args.plan), Path(args.output))
    except Exception as exc:
        code = str(exc) if str(exc).startswith("E_") else "E_INTERNAL"
        print(json.dumps({"status": "error", "error_code": code}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
