#!/usr/bin/env python3
"""Validate two blinded model-assisted passes and apply frozen v2 gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.visible_action_v2 import (  # noqa: E402
    apply_protocol_amendment,
    construct_constrained_folds,
    load_json,
    sha256_file,
    summarize_annotations,
)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v2 artifact: {path}")
    path.write_text(json.dumps(value, indent=2) + "\n")


def _rate(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.1f}%"


def _write_markdown(path: Path, summary: Mapping[str, Any], split: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v2 artifact: {path}")
    action = summary["agreement"]["observable_action"]
    agency = summary["agreement"]["agency"]
    temporal = summary["agreement"]["temporal_alignment"]
    referent = summary["agreement"]["asr_refers_to_visible_wearer_action"]
    support_lines = [
        f"| `{label}` | {values['windows']} | {values['event_groups']} | {values['locations']} |"
        for label, values in summary["support"].items()
    ] or ["| _none_ | 0 | 0 | 0 |"]
    lines = [
        "# AEA visible-action v2 model-assisted agreement",
        "",
        "These are MODEL-ASSISTED DEVELOPMENT LABELS, not human annotations and not human inter-rater reliability.",
        "",
        f"Annotation gate: **{'PASS' if summary['annotation_gate_passed'] else 'FAIL'}**. "
        f"Severe failure: **{summary['severe_annotation_failure']}**.",
        "",
        "| Field | Exact agreement | Cohen's kappa | Confidence-weighted exact |",
        "|---|---:|---:|---:|",
        f"| Observable action | {_rate(action['exact_agreement'])} | {action['cohens_kappa_unweighted'] if action['cohens_kappa_unweighted'] is not None else 'NA'} | {_rate(action['confidence_weighted_exact_agreement'])} |",
        f"| Agency | {_rate(agency['exact_agreement'])} | {agency['cohens_kappa_unweighted'] if agency['cohens_kappa_unweighted'] is not None else 'NA'} | {_rate(agency['confidence_weighted_exact_agreement'])} |",
        f"| Temporal alignment | {_rate(temporal['exact_agreement'])} | {temporal['cohens_kappa_unweighted'] if temporal['cohens_kappa_unweighted'] is not None else 'NA'} | {_rate(temporal['confidence_weighted_exact_agreement'])} |",
        f"| ASR refers to wearer action | {_rate(referent['exact_agreement'])} | {referent['cohens_kappa_unweighted'] if referent['cohens_kappa_unweighted'] is not None else 'NA'} | {_rate(referent['confidence_weighted_exact_agreement'])} |",
        "",
        f"Modeled-action consensus: {summary['rates']['modeled_consensus_yield']['successes']}/72 "
        f"({_rate(summary['rates']['modeled_consensus_yield']['rate'])}).",
        "",
        "| Consensus action | Windows | Event groups | Locations |",
        "|---|---:|---:|---:|",
        *support_lines,
        "",
        f"Split/donor status: **{split['status']}**; gate passed: **{split['split_gate_passed']}**.",
        "",
        "No row was adjudicated, relabeled, merged, or selected after outcomes.",
    ]
    path.write_text("\n".join(lines) + "\n")


def summarize(
    protocol_path: Path,
    amendment_path: Path,
    manifest_path: Path,
    access_receipt_path: Path,
    pass_a_path: Path,
    pass_b_path: Path,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = load_json(protocol_path)
    amendment = load_json(amendment_path)
    if amendment.get("parent_protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("protocol amendment parent hash mismatch")
    protocol = apply_protocol_amendment(base, amendment)
    manifest = load_json(manifest_path)
    receipt = load_json(access_receipt_path)
    if any((
        receipt.get("reserve_rgb_files_opened") != 0,
        receipt.get("reserve_imu_arrays_opened") != 0,
        receipt.get("signed_urls_loaded_printed_or_copied") is not False,
        manifest.get("reserve_groups_present") != [],
    )):
        raise ValueError("dense packet reserve/signed-URL receipt failed")
    pass_a = load_json(pass_a_path)
    pass_b = load_json(pass_b_path)
    expected_role = protocol["annotation_schema"]["pass_role"]
    metadata_checks = {
        "pass_a_protocol": pass_a.get("protocol_id") == protocol["protocol_id"],
        "pass_b_protocol": pass_b.get("protocol_id") == protocol["protocol_id"],
        "pass_a_name": pass_a.get("pass") == "pass_a",
        "pass_b_name": pass_b.get("pass") == "pass_b",
        "pass_a_role": pass_a.get("role") == expected_role,
        "pass_b_role": pass_b.get("role") == expected_role,
    }
    if not all(metadata_checks.values()):
        raise ValueError(f"annotation pass metadata invalid: {metadata_checks}")
    summary = summarize_annotations(
        pass_a["rows"], pass_b["rows"], manifest["rows"], protocol
    )
    summary["input_integrity"] = {
        "metadata_checks": metadata_checks,
        "dense_manifest_sha256": sha256_file(manifest_path),
        "reserve_access_receipt_sha256": sha256_file(access_receipt_path),
        "pass_a_sha256": sha256_file(pass_a_path),
        "pass_b_sha256": sha256_file(pass_b_path),
        "reserve_rgb_files_opened": 0,
        "reserve_imu_arrays_opened": 0,
    }
    if summary["annotation_gate_passed"]:
        split = construct_constrained_folds(
            summary["consensus_rows"],
            manifest["rows"],
            sorted(summary["retained_support"]),
            protocol,
        )
    else:
        split = {
            "schema_version": "aea-visible-action-split-donor-v2",
            "protocol_id": protocol["protocol_id"],
            "status": "not_run_annotation_gate_failed",
            "annotation_gate_passed": False,
            "severe_annotation_failure": summary["severe_annotation_failure"],
            "solver_runs": 0,
            "relaxation_or_retry_performed": False,
            "split_gate_passed": False,
        }
    output.mkdir(parents=True, exist_ok=True)
    _write_json_new(output / "agreement_report.json", summary)
    _write_json_new(output / "consensus_labels.json", {
        "schema_version": "aea-visible-action-consensus-labels-v2",
        "protocol_id": protocol["protocol_id"],
        "role": expected_role,
        "modeling_consensus_rule": "exact modeled action plus medium-or-high confidence in both passes",
        "rows": summary["consensus_rows"],
        "retained_support": summary["retained_support"],
    })
    _write_json_new(output / "split_donor_feasibility.json", split)
    _write_markdown(output / "agreement_report.md", summary, split)
    return summary, split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("output/aea_visible_action_v2")
    parser.add_argument("--protocol", type=Path, default=root / "preregistered_protocol.json")
    parser.add_argument("--amendment", type=Path, default=root / "preregistered_protocol_amendment_1.json")
    parser.add_argument("--manifest", type=Path, default=root / "dense_clip_manifest.json")
    parser.add_argument("--access-receipt", type=Path, default=root / "reserve_access_receipt.json")
    parser.add_argument("--pass-a", type=Path, default=root / "model_assisted_pass_a.json")
    parser.add_argument("--pass-b", type=Path, default=root / "model_assisted_pass_b.json")
    parser.add_argument("--out", type=Path, default=root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, split = summarize(
        args.protocol, args.amendment, args.manifest, args.access_receipt,
        args.pass_a, args.pass_b, args.out,
    )
    print(json.dumps({
        "annotation_gate_passed": summary["annotation_gate_passed"],
        "severe_annotation_failure": summary["severe_annotation_failure"],
        "split_status": split["status"],
        "split_gate_passed": split["split_gate_passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
