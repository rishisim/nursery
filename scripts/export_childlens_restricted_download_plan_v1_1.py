#!/usr/bin/env python3
"""Export the frozen selected-object plan inside restricted quarantine only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


VERSION = "childlens-restricted-download-plan-v1.1.0"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def export_plan(manifest_path: Path, output_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_resolved_parent = output_path.parent.resolve(strict=True)
    if output_resolved_parent == REPO_ROOT or REPO_ROOT in output_resolved_parent.parents:
        raise ValueError("E_OUTPUT_INSIDE_REPOSITORY")
    objects = {
        row["object_key"]: row
        for row in manifest["records"]["objects"]
        if row["top_level_class"] == "VIDEO"
    }
    selected: list[dict[str, object]] = []
    for row in manifest["pilot_selection"]:
        obj = objects.get(row["object_key"])
        if obj is None:
            raise ValueError("E_SELECTED_OBJECT_LINK")
        locator = obj["source_locator"]
        if not isinstance(locator, str) or not locator.startswith("/ChildLens/videos/"):
            raise ValueError("E_VIDEO_LOCATOR")
        selected.append({
            "selection_rank": row["selection_rank"],
            "media_key": row["media_key"],
            "object_key": row["object_key"],
            "selection_hash": row["selection_hash"],
            "source_locator": locator,
            "expected_size_bytes": None,
            "local_sha256": None,
        })
    payload = {
        "schema_version": VERSION,
        "canonical_restricted_manifest_sha256": manifest["canonical_restricted_manifest_sha256"],
        "pilot_selection_sha256": manifest["pilot_selection_sha256"],
        "video_size_status": "EXACT_BYTES_UNRESOLVED",
        "selected": selected,
    }
    encoded = _canonical(payload) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=output_resolved_parent)
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
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    return {"status": "ok", "selected_count": len(selected), "download_plan_sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        receipt = export_plan(Path(args.manifest), Path(args.output))
    except Exception as exc:
        code = str(exc) if str(exc).startswith("E_") else "E_INTERNAL"
        print(json.dumps({"status": "error", "error_code": code}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
