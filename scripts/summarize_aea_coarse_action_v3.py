#!/usr/bin/env python3
"""Seal, validate, and compare the two frozen AEA coarse-action v3 passes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.coarse_action_v3 import (  # noqa: E402
    PROTOCOL_ID,
    construct_constrained_folds,
    load_json,
    merge_annotation_stages,
    sha256_file,
    summarize_annotations,
    validate_stage1_rows,
    validate_stage2_rows,
    verify_protocol_freeze,
)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v3 artifact: {path}")
    path.write_text(json.dumps(value, indent=2) + "\n")


def _rate(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.1f}%"


def _validate_pass(
    pass_name: str,
    stage1_path: Path,
    stage2_path: Path,
    visual_packet_path: Path,
    language_packet_path: Path,
    expected_ids: set[str],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    stage1 = load_json(stage1_path)
    stage2 = load_json(stage2_path)
    visual_packet = load_json(visual_packet_path)
    language_packet = load_json(language_packet_path)
    role = protocol["annotation_interface"]["pass_role"]
    stage1_hash = sha256_file(stage1_path)
    checks = {
        "stage1_schema": stage1.get("schema_version")
        == "aea-coarse-action-model-pass-stage1-v3",
        "stage2_schema": stage2.get("schema_version")
        == "aea-coarse-action-model-pass-stage2-v3",
        "stage1_protocol": stage1.get("protocol_id") == PROTOCOL_ID,
        "stage2_protocol": stage2.get("protocol_id") == PROTOCOL_ID,
        "stage1_pass": stage1.get("pass") == pass_name,
        "stage2_pass": stage2.get("pass") == pass_name,
        "stage1_role": stage1.get("role") == role,
        "stage2_role": stage2.get("role") == role,
        "stage1_packet_hash_field": stage1.get("source_packet_sha256")
        == sha256_file(visual_packet_path),
        "stage2_packet_hash_field": stage2.get("source_packet_sha256")
        == sha256_file(language_packet_path),
        "stage1_packet_order_preserved": [
            row.get("blind_id") for row in stage1.get("rows", [])
        ]
        == [row.get("blind_id") for row in visual_packet.get("rows", [])],
        "stage2_packet_order_preserved": [
            row.get("blind_id") for row in stage2.get("rows", [])
        ]
        == [row.get("blind_id") for row in language_packet.get("rows", [])],
        "stage2_seals_exact_stage1": stage2.get("sealed_stage_1_sha256")
        == stage1_hash,
        "stage1_video_only_attestation": stage1.get(
            "completed_without_transcript_or_anchored_verb"
        )
        is True,
        "stage1_other_pass_not_read": stage1.get("other_pass_read") is False,
        "stage2_other_pass_not_read": stage2.get("other_pass_read") is False,
        "stage1_prior_labels_not_read": stage1.get(
            "v1_or_v2_completed_labels_or_rationales_read"
        )
        is False,
        "stage2_prior_labels_not_read": stage2.get(
            "v1_or_v2_completed_labels_or_rationales_read"
        )
        is False,
        "stage2_visible_fields_unmodified": stage2.get("stage_1_fields_modified")
        is False,
        "zero_reserve_media": stage1.get("reserve_media_opened") is False
        and stage2.get("reserve_media_opened") is False,
        "zero_raw_vrs_audio_imu": stage1.get("raw_vrs_audio_or_imu_opened")
        is False
        and stage2.get("raw_vrs_audio_or_imu_opened") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"{pass_name} isolation/metadata validation failed: {checks}")
    visual_rows = validate_stage1_rows(stage1["rows"], expected_ids, protocol)
    language_rows = validate_stage2_rows(stage2["rows"], expected_ids, protocol)
    combined = merge_annotation_stages(visual_rows, language_rows)
    raw_pass = {
        "schema_version": "aea-coarse-action-model-assisted-pass-v3",
        "protocol_id": PROTOCOL_ID,
        "pass": pass_name,
        "role": role,
        "stage_order": [
            "stage_1_sealed_video_only_visible_action",
            "stage_2_transcript_referent_and_temporal_relation",
        ],
        "stage_1_sha256": stage1_hash,
        "stage_2_sha256": sha256_file(stage2_path),
        "no_adjudication": True,
        "rows": combined,
    }
    isolation = {
        "pass": pass_name,
        "checks": checks,
        "stage_1_sha256": stage1_hash,
        "stage_2_sha256": sha256_file(stage2_path),
        "visual_packet_sha256": sha256_file(visual_packet_path),
        "language_packet_sha256": sha256_file(language_packet_path),
        "rows": len(combined),
    }
    return raw_pass, isolation, combined


def _write_agreement_markdown(
    path: Path, summary: Mapping[str, Any], split: Mapping[str, Any]
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v3 report: {path}")
    action = summary["agreement"]["observable_action"]
    referent = summary["agreement"]["asr_referent"]
    temporal = summary["agreement"]["temporal_relation"]
    support_lines = [
        f"| `{label}` | {values['windows']} | {values['event_groups']} | {values['locations']} |"
        for label, values in summary["support"].items()
    ] or ["| _none_ | 0 | 0 | 0 |"]
    lines = [
        "# AEA coarse-action v3 model-assisted agreement",
        "",
        "These are MODEL-ASSISTED DEVELOPMENT LABELS, not human annotations or human inter-rater reliability.",
        "",
        f"Coarse annotation/support gate: **{'PASS' if summary['coarse_annotation_gate_passed'] else 'FAIL'}**.",
        "",
        "| Field | Exact agreement | Cohen's kappa | Confidence-weighted exact |",
        "|---|---:|---:|---:|",
        f"| Coarse visible action | {_rate(action['exact_agreement'])} | {action['cohens_kappa_unweighted'] if action['cohens_kappa_unweighted'] is not None else 'NA'} | {_rate(action['confidence_weighted_exact_agreement'])} |",
        f"| ASR referent | {_rate(referent['exact_agreement'])} | {referent['cohens_kappa_unweighted'] if referent['cohens_kappa_unweighted'] is not None else 'NA'} | {_rate(referent['confidence_weighted_exact_agreement'])} |",
        f"| Temporal relation | {_rate(temporal['exact_agreement'])} | {temporal['cohens_kappa_unweighted'] if temporal['cohens_kappa_unweighted'] is not None else 'NA'} | {_rate(temporal['confidence_weighted_exact_agreement'])} |",
        "",
        f"Modeled consensus: {summary['rates']['modeled_consensus_yield']['successes']}/72 ({_rate(summary['rates']['modeled_consensus_yield']['rate'])}).",
        "",
        "| Consensus class | Windows | Event groups | Locations |",
        "|---|---:|---:|---:|",
        *support_lines,
        "",
        f"Split/donor status: **{split['status']}**; gate passed: **{split.get('split_gate_passed', False)}**.",
        "",
        "No row was adjudicated, relabeled, or selected after comparison.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_language_markdown(path: Path, result: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v3 report: {path}")
    rates = result["rates"]
    agreement = result["referent_agreement"]
    lines = [
        "# AEA coarse-action v3 language-alignment diagnostic",
        "",
        f"Language-alignment gate: **{'PASS' if result['language_alignment_gate_passed'] else 'FAIL'}**.",
        f"Conclusion: **{result['language_conclusion']}**.",
        "",
        f"Consensus aligned anchors: {rates['language_aligned_consensus']['successes']}/72 ({_rate(rates['language_aligned_consensus']['rate'])}); frozen minimum 18/72.",
        f"Pass A wearer-action rate: {rates['wearer_action_pass_a']['successes']}/72 ({_rate(rates['wearer_action_pass_a']['rate'])}).",
        f"Pass B wearer-action rate: {rates['wearer_action_pass_b']['successes']}/72 ({_rate(rates['wearer_action_pass_b']['rate'])}).",
        f"Simplified referent agreement: {_rate(agreement['exact_agreement'])}; kappa {agreement['cohens_kappa_unweighted'] if agreement['cohens_kappa_unweighted'] is not None else 'NA'}.",
        "",
        "The denominator is all 72 rows. Sensor results cannot rescue a failed language gate.",
    ]
    path.write_text("\n".join(lines) + "\n")


def summarize(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol, freeze = verify_protocol_freeze(
        args.protocol, args.freeze_receipt, args.preregistration, args.codebook
    )
    manifest = load_json(args.manifest)
    access = load_json(args.access_receipt)
    if any((
        manifest.get("protocol_id") != PROTOCOL_ID,
        manifest.get("sample_size") != 72,
        manifest.get("reserve_groups_present") != [],
        access.get("reserve_rgb_files_opened") != 0,
        access.get("reserve_imu_arrays_opened") != 0,
        access.get("signed_urls_loaded_printed_copied_or_used") is not False,
    )):
        raise ValueError("v3 fixed manifest/reserve receipt validation failed")
    expected_ids = {str(row["blind_id"]) for row in manifest["rows"]}
    raw_passes = {}
    isolations = []
    combined = {}
    for pass_name in ("pass_a", "pass_b"):
        raw_pass, isolation, rows = _validate_pass(
            pass_name,
            getattr(args, f"{pass_name}_stage1"),
            getattr(args, f"{pass_name}_stage2"),
            getattr(args, f"{pass_name}_visual_packet"),
            getattr(args, f"{pass_name}_language_packet"),
            expected_ids,
            protocol,
        )
        path = args.out / f"model_assisted_{pass_name}.json"
        _write_json_new(path, raw_pass)
        raw_passes[pass_name] = path
        isolations.append(isolation)
        combined[pass_name] = rows
    summary = summarize_annotations(
        combined["pass_a"], combined["pass_b"], manifest["rows"], protocol
    )
    summary["input_integrity"] = {
        "protocol_freeze": freeze,
        "fixed_manifest_sha256": sha256_file(args.manifest),
        "reserve_access_receipt_sha256": sha256_file(args.access_receipt),
        "pass_a_sha256": sha256_file(raw_passes["pass_a"]),
        "pass_b_sha256": sha256_file(raw_passes["pass_b"]),
        "pass_isolation_checks": [item["checks"] for item in isolations],
        "reserve_rgb_files_opened": 0,
        "reserve_imu_arrays_opened": 0,
        "adjudications": 0,
    }
    if summary["coarse_annotation_gate_passed"]:
        split = construct_constrained_folds(
            summary["consensus_rows"], manifest["rows"], protocol
        )
    else:
        split = {
            "schema_version": "aea-coarse-action-split-donor-v3",
            "protocol_id": PROTOCOL_ID,
            "status": "not_run_coarse_annotation_gate_failed",
            "coarse_annotation_gate_passed": False,
            "solver_runs": 0,
            "relaxation_or_retry_performed": False,
            "split_gate_passed": False,
        }
    language = {
        "schema_version": "aea-coarse-action-language-alignment-v3",
        "protocol_id": PROTOCOL_ID,
        "scientific_role": protocol["scientific_role"],
        "n": 72,
        "definition": protocol["language_alignment_gate"]["aligned_consensus_definition"],
        "rates": {
            key: summary["rates"][key]
            for key in (
                "wearer_action_pass_a",
                "wearer_action_pass_b",
                "language_aligned_consensus",
            )
        },
        "referent_agreement": summary["agreement"]["asr_referent"],
        "temporal_agreement": summary["agreement"]["temporal_relation"],
        "gate_checks": summary["language_gate_checks"],
        "language_alignment_gate_passed": summary[
            "language_alignment_gate_passed"
        ],
        "language_conclusion": summary["language_conclusion"],
        "aligned_rows": summary["language_aligned_rows"],
        "sensor_outcomes_can_override": False,
        "acquisition_for_language_if_failed": False,
    }
    _write_json_new(args.out / "annotation_isolation_receipt.json", {
        "schema_version": "aea-coarse-action-annotation-isolation-v3",
        "protocol_id": PROTOCOL_ID,
        "passes": isolations,
        "passes_compared_only_after_both_stages_sealed": True,
        "adjudications": 0,
        "v1_or_v2_completed_labels_or_rationales_read": False,
    })
    _write_json_new(args.out / "agreement_report.json", summary)
    _write_json_new(args.out / "consensus_labels.json", {
        "schema_version": "aea-coarse-action-consensus-labels-v3",
        "protocol_id": PROTOCOL_ID,
        "role": protocol["annotation_interface"]["pass_role"],
        "modeling_consensus_rule": "exact modeled coarse action and medium-or-high visible confidence in both passes",
        "rows": summary["consensus_rows"],
        "support": summary["support"],
    })
    _write_json_new(args.out / "language_alignment_results.json", language)
    _write_json_new(args.out / "split_donor_feasibility.json", split)
    _write_agreement_markdown(args.out / "agreement_report.md", summary, split)
    _write_language_markdown(args.out / "language_alignment_report.md", language)
    return summary, split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("output/aea_coarse_action_v3")
    parser.add_argument("--protocol", type=Path, default=root / "preregistered_protocol.json")
    parser.add_argument("--freeze-receipt", type=Path, default=root / "protocol_freeze_receipt.json")
    parser.add_argument("--preregistration", type=Path, default=Path("docs/aea_coarse_action_v3_preregistration.md"))
    parser.add_argument("--codebook", type=Path, default=root / "annotation_codebook.md")
    parser.add_argument("--manifest", type=Path, default=root / "fixed_dense_manifest.json")
    parser.add_argument("--access-receipt", type=Path, default=root / "reserve_access_receipt.json")
    for pass_name in ("pass_a", "pass_b"):
        parser.add_argument(f"--{pass_name.replace('_', '-')}-stage1", dest=f"{pass_name}_stage1", type=Path, default=root / f"model_assisted_{pass_name}_stage1.json")
        parser.add_argument(f"--{pass_name.replace('_', '-')}-stage2", dest=f"{pass_name}_stage2", type=Path, default=root / f"model_assisted_{pass_name}_stage2.json")
        parser.add_argument(f"--{pass_name.replace('_', '-')}-visual-packet", dest=f"{pass_name}_visual_packet", type=Path, default=root / f"annotation_packet_{pass_name}_stage1_visual.json")
        parser.add_argument(f"--{pass_name.replace('_', '-')}-language-packet", dest=f"{pass_name}_language_packet", type=Path, default=root / f"annotation_packet_{pass_name}_stage2_language.json")
    parser.add_argument("--out", type=Path, default=root)
    return parser.parse_args()


def main() -> None:
    summary, split = summarize(parse_args())
    print(json.dumps({
        "coarse_annotation_gate_passed": summary["coarse_annotation_gate_passed"],
        "language_alignment_gate_passed": summary["language_alignment_gate_passed"],
        "split_status": split["status"],
        "split_gate_passed": split.get("split_gate_passed", False),
    }, indent=2))


if __name__ == "__main__":
    main()
