#!/usr/bin/env python3
"""Validate and hash both video-only v3 passes before transcript release."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.coarse_action_v3 import (  # noqa: E402
    PROTOCOL_ID,
    load_json,
    sha256_file,
    validate_stage1_rows,
    verify_protocol_freeze,
)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v3 stage-1 seal: {path}")
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("output/aea_coarse_action_v3")
    parser.add_argument("--protocol", type=Path, default=root / "preregistered_protocol.json")
    parser.add_argument("--freeze-receipt", type=Path, default=root / "protocol_freeze_receipt.json")
    parser.add_argument("--preregistration", type=Path, default=Path("docs/aea_coarse_action_v3_preregistration.md"))
    parser.add_argument("--codebook", type=Path, default=root / "annotation_codebook.md")
    parser.add_argument("--manifest", type=Path, default=root / "fixed_dense_manifest.json")
    parser.add_argument("--out", type=Path, default=root)
    args = parser.parse_args()
    protocol, freeze = verify_protocol_freeze(
        args.protocol, args.freeze_receipt, args.preregistration, args.codebook
    )
    manifest = load_json(args.manifest)
    expected_ids = {str(row["blind_id"]) for row in manifest["rows"]}
    passes = []
    for pass_name in ("pass_a", "pass_b"):
        artifact = args.out / f"model_assisted_{pass_name}_stage1.json"
        packet = args.out / f"annotation_packet_{pass_name}_stage1_visual.json"
        value = load_json(artifact)
        packet_value = load_json(packet)
        checks = {
            "schema": value.get("schema_version")
            == "aea-coarse-action-model-pass-stage1-v3",
            "protocol": value.get("protocol_id") == PROTOCOL_ID,
            "pass": value.get("pass") == pass_name,
            "stage": value.get("stage")
            == "stage_1_sealed_video_only_visible_action",
            "source_packet_hash": value.get("source_packet_sha256")
            == sha256_file(packet),
            "packet_order_preserved": [row.get("blind_id") for row in value.get("rows", [])]
            == [row.get("blind_id") for row in packet_value.get("rows", [])],
            "completed_video_only": value.get(
                "completed_without_transcript_or_anchored_verb"
            )
            is True,
            "other_pass_not_read": value.get("other_pass_read") is False,
            "prior_completed_labels_not_read": value.get(
                "v1_or_v2_completed_labels_or_rationales_read"
            )
            is False,
            "reserve_not_opened": value.get("reserve_media_opened") is False,
            "raw_vrs_audio_imu_not_opened": value.get(
                "raw_vrs_audio_or_imu_opened"
            )
            is False,
        }
        if not all(checks.values()):
            raise ValueError(f"{pass_name} stage-1 metadata failed: {checks}")
        rows = validate_stage1_rows(value["rows"], expected_ids, protocol)
        passes.append({
            "pass": pass_name,
            "artifact": str(artifact),
            "artifact_sha256": sha256_file(artifact),
            "source_packet": str(packet),
            "source_packet_sha256": sha256_file(packet),
            "rows": len(rows),
            "unique_blind_ids": len({row["blind_id"] for row in rows}),
            "checks": checks,
        })
    stage2_outputs_absent = all(
        not (args.out / f"model_assisted_{pass_name}_stage2.json").exists()
        for pass_name in ("pass_a", "pass_b")
    )
    if not stage2_outputs_absent:
        raise ValueError("a stage-2 output existed before both stage-1 passes were sealed")
    receipt = {
        "schema_version": "aea-coarse-action-stage1-seal-receipt-v3",
        "protocol_id": PROTOCOL_ID,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_freeze_checks": freeze,
        "both_passes_complete_before_transcript_release": True,
        "stage2_outputs_absent_at_seal": True,
        "passes_compared_or_adjudicated_before_seal": False,
        "passes": passes,
    }
    _write_json_new(args.out / "stage1_seal_receipt.json", receipt)
    print(json.dumps({
        "both_passes_complete_before_transcript_release": True,
        "passes": [
            {"pass": item["pass"], "rows": item["rows"], "sha256": item["artifact_sha256"]}
            for item in passes
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
