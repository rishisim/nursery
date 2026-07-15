#!/usr/bin/env python3
"""Reblind the fixed v2 dense evidence for the frozen AEA v3 interface.

This stage reads metadata and checks file existence only. It performs no RGB,
raw VRS, audio, or IMU query and never reads completed v1/v2 annotations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.coarse_action_v3 import (  # noqa: E402
    PROTOCOL_ID,
    canonical_digest,
    packet_order,
    reblind_fixed_rows,
    sha256_file,
    validate_fixed_dense_source,
    verify_protocol_freeze,
)


def _write_json_new(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen v3 artifact: {path}")
    path.write_text(json.dumps(value, indent=2) + "\n")


def _visual_packet_rows(
    ordered: Sequence[Mapping[str, Any]], repository_root: Path
) -> list[dict[str, Any]]:
    rows = []
    for rank, row in enumerate(ordered, start=1):
        contact = (repository_root / str(row["contact_sheet"])).resolve()
        details = [
            (repository_root / str(path)).resolve() for path in row["detail_sheets"]
        ]
        frames = [
            (repository_root / str(path)).resolve() for path in row["dense_frame_paths"]
        ]
        rows.append({
            "packet_rank": rank,
            "blind_id": str(row["blind_id"]),
            "frame_count": 31,
            "anchor_guidance": "Temporal midpoint is between frames 15 and 16; no transcript or anchored verb is available in this sealed stage.",
            "contact_sheet": str(contact),
            "detail_sheet_00_15": str(details[0]),
            "detail_sheet_16_30": str(details[1]),
            "frame_directory": str(frames[0].parent),
            "annotation": {
                "blind_id": str(row["blind_id"]),
                "observable_action": None,
                "visible_confidence": None,
                "evidence_frame_start": None,
                "evidence_frame_end": None,
                "visible_rationale": None,
            },
        })
    return rows


def _language_packet_rows(
    ordered: Sequence[Mapping[str, Any]], repository_root: Path
) -> list[dict[str, Any]]:
    rows = []
    for rank, row in enumerate(ordered, start=1):
        details = [
            (repository_root / str(path)).resolve() for path in row["detail_sheets"]
        ]
        rows.append({
            "packet_rank": rank,
            "blind_id": str(row["blind_id"]),
            "anchored_asr_verb": str(row["anchored_asr_verb"]),
            "transcript": str(row["transcript"]),
            "anchor_guidance": "Centered ASR anchor is between frames 15 and 16; stage-1 visible labels are sealed and immutable.",
            "contact_sheet": str(
                (repository_root / str(row["contact_sheet"])).resolve()
            ),
            "detail_sheet_00_15": str(details[0]),
            "detail_sheet_16_30": str(details[1]),
            "annotation": {
                "blind_id": str(row["blind_id"]),
                "asr_referent": None,
                "temporal_relation": None,
                "language_confidence": None,
                "language_rationale": None,
            },
        })
    return rows


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = Path.cwd().resolve()
    protocol, freeze_checks = verify_protocol_freeze(
        args.protocol, args.freeze_receipt, args.preregistration, args.codebook
    )
    source, source_checks = validate_fixed_dense_source(
        protocol,
        args.v2_manifest,
        args.v2_access_receipt,
        repository_root,
    )
    rows = reblind_fixed_rows(source["rows"], protocol)
    sample_digest = canonical_digest([
        {
            "blind_id": row["blind_id"],
            "example_id": row["example_id"],
            "event_group": row["event_group"],
        }
        for row in rows
    ])
    manifest = {
        "schema_version": "aea-coarse-action-fixed-dense-manifest-v3",
        "protocol_id": PROTOCOL_ID,
        "sample_size": len(rows),
        "source_v2_sample_digest": source["sample_digest"],
        "v3_sample_digest": sample_digest,
        "membership_rule": "exact_same_72_v2_dense_rows_no_add_remove_select_or_substitute",
        "evidence_rule": "reference_exact_existing_31_frame_v2_evidence_no_rematerialization",
        "development_event_groups": len({row["event_group"] for row in rows}),
        "reserve_groups_present": [],
        "rows": rows,
    }
    _write_json_new(args.out / "fixed_dense_manifest.json", manifest)

    role = protocol["annotation_interface"]["pass_role"]
    packet_hashes: dict[str, str] = {}
    for pass_name in ("pass_a", "pass_b"):
        ordered = packet_order(
            rows, str(protocol["sample"]["pass_order_salts"][pass_name])
        )
        common = {
            "protocol_id": PROTOCOL_ID,
            "pass": pass_name,
            "role": role,
            "codebook": str(args.codebook.resolve()),
            "context_isolation_required": True,
            "other_pass_access_forbidden": True,
            "v1_v2_completed_label_or_rationale_access_forbidden": True,
        }
        visual_path = args.out / f"annotation_packet_{pass_name}_stage1_visual.json"
        visual = {
            "schema_version": "aea-coarse-action-blinded-visual-packet-v3",
            **common,
            "stage": "stage_1_sealed_video_only_visible_action",
            "transcript_or_anchored_verb_present": False,
            "must_complete_all_72_before_stage_2": True,
            "rows": _visual_packet_rows(ordered, repository_root),
        }
        _write_json_new(visual_path, visual)
        language_path = args.out / f"annotation_packet_{pass_name}_stage2_language.json"
        language = {
            "schema_version": "aea-coarse-action-blinded-language-packet-v3",
            **common,
            "stage": "stage_2_transcript_referent_and_temporal_relation",
            "requires_sealed_stage_1": True,
            "stage_1_fields_mutable": False,
            "rows": _language_packet_rows(ordered, repository_root),
        }
        _write_json_new(language_path, language)
        packet_hashes[visual_path.name] = sha256_file(visual_path)
        packet_hashes[language_path.name] = sha256_file(language_path)

    access = {
        "schema_version": "aea-coarse-action-reserve-access-receipt-v3",
        "protocol_id": PROTOCOL_ID,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_v2_dense_manifest_sha256": sha256_file(args.v2_manifest),
        "source_v2_access_receipt_sha256": sha256_file(args.v2_access_receipt),
        "fixed_dense_manifest_sha256": sha256_file(
            args.out / "fixed_dense_manifest.json"
        ),
        "source_checks": source_checks,
        "freeze_checks": freeze_checks,
        "v3_existing_dense_frames_referenced": 72 * 31,
        "v3_dense_images_opened_by_preparation": 0,
        "v3_additional_development_rgb_queries": 0,
        "v3_raw_vrs_files_opened": 0,
        "v3_development_imu_arrays_opened": 0,
        "reserve_event_groups_opened": [],
        "reserve_rgb_files_opened": 0,
        "reserve_imu_arrays_opened": 0,
        "audio_opened": False,
        "signed_manifest_loaded": False,
        "signed_urls_loaded_printed_copied_or_used": False,
        "v1_or_v2_completed_annotation_files_opened": 0,
        "packet_sha256": packet_hashes,
    }
    _write_json_new(args.out / "reserve_access_receipt.json", access)
    return access


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("output/aea_coarse_action_v3")
    parser.add_argument("--protocol", type=Path, default=root / "preregistered_protocol.json")
    parser.add_argument("--freeze-receipt", type=Path, default=root / "protocol_freeze_receipt.json")
    parser.add_argument("--preregistration", type=Path, default=Path("docs/aea_coarse_action_v3_preregistration.md"))
    parser.add_argument("--codebook", type=Path, default=root / "annotation_codebook.md")
    parser.add_argument("--v2-manifest", type=Path, default=Path("output/aea_visible_action_v2/dense_clip_manifest.json"))
    parser.add_argument("--v2-access-receipt", type=Path, default=Path("output/aea_visible_action_v2/reserve_access_receipt.json"))
    parser.add_argument("--out", type=Path, default=root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare(args)
    print(json.dumps({
        "protocol_id": result["protocol_id"],
        "existing_dense_frames_referenced": result["v3_existing_dense_frames_referenced"],
        "additional_rgb_queries": result["v3_additional_development_rgb_queries"],
        "reserve_rgb_files_opened": result["reserve_rgb_files_opened"],
        "reserve_imu_arrays_opened": result["reserve_imu_arrays_opened"],
    }, indent=2))


if __name__ == "__main__":
    main()
