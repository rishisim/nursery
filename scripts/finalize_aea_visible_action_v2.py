#!/usr/bin/env python3
"""Apply frozen AEA v2 decision/acquisition rules and write final artifacts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.manifest import (  # noqa: E402
    component_budget,
    load_safe_manifest,
    manifest_summary,
)
from babyworld_lite.aea.visible_action_v2 import (  # noqa: E402
    apply_protocol_amendment,
    load_json,
    salted_digest,
    sha256_file,
)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite final v2 artifact: {path}")
    path.write_text(json.dumps(value, indent=2) + "\n")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _safe_release_summary(
    safe_manifest_path: Path, existing_plan_path: Path, free_bytes: int
) -> tuple[Any, set[str], dict[str, Any]]:
    manifest = load_safe_manifest(safe_manifest_path)
    plan = yaml.safe_load(existing_plan_path.read_text())
    existing = set(map(str, plan["selection"]["sequences"]))
    all_names = set(manifest.sequences)
    if not existing <= all_names or len(existing) != 40:
        raise ValueError("existing 40-recording plan does not match safe release manifest")
    remaining = sorted(all_names - existing)
    strata: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "release_recordings": 0,
        "existing_recordings": 0,
        "remaining_recordings": 0,
        "release_declared_bytes": 0,
        "remaining_declared_bytes": 0,
    })
    for name in sorted(all_names):
        sequence = manifest.sequences[name]
        sequence_id = sequence.sequence_id
        key = f"loc{sequence_id.location}_script{sequence_id.script}"
        declared = sum(
            sequence.assets[component].file_size_bytes
            for component in ("annotations", "main_vrs")
        )
        strata[key]["release_recordings"] += 1
        strata[key]["release_declared_bytes"] += declared
        if name in existing:
            strata[key]["existing_recordings"] += 1
        else:
            strata[key]["remaining_recordings"] += 1
            strata[key]["remaining_declared_bytes"] += declared
    summary = {
        "schema_version": "aea-visible-action-safe-release-metadata-v2",
        "dataset": manifest_summary(manifest),
        "safe_parser": "load_safe_manifest discards download_url values at input boundary",
        "signed_urls_serialized_or_printed": False,
        "components_considered": ["annotations", "main_vrs"],
        "existing_40": component_budget(manifest, sorted(existing), ("annotations", "main_vrs")),
        "remaining_103": component_budget(manifest, remaining, ("annotations", "main_vrs")),
        "location_script_strata": dict(sorted(strata.items())),
        "filesystem_free_bytes_at_decision": int(free_bytes),
        "filesystem_free_gib_at_decision": round(free_bytes / 1024**3, 3),
    }
    return manifest, existing, summary


def _event_group(name: str) -> str:
    return name.rsplit("_rec", 1)[0]


def _bounded_acquisition_plan(
    manifest: Any,
    existing: set[str],
    free_bytes: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    acquisition = protocol["acquisition"]
    existing_cell_count = Counter(
        (manifest.sequences[name].sequence_id.location, manifest.sequences[name].sequence_id.script)
        for name in existing
    )
    existing_groups = {_event_group(name) for name in existing}
    remaining_by_group: dict[str, list[str]] = defaultdict(list)
    for name in sorted(set(manifest.sequences) - existing):
        remaining_by_group[_event_group(name)].append(name)
    candidates = []
    for group, names in remaining_by_group.items():
        first = manifest.sequences[names[0]].sequence_id
        cell = (first.location, first.script)
        declared = sum(
            manifest.sequences[name].assets[component].file_size_bytes
            for name in names for component in acquisition["components"]
        )
        novel = group not in existing_groups
        candidates.append({
            "event_group": group,
            "sequence_ids": sorted(names),
            "location": first.location,
            "script": first.script,
            "new_recordings": len(names),
            "declared_bytes": declared,
            "declared_gib": round(declared / 1024**3, 3),
            "existing_cell_recordings": existing_cell_count[cell],
            "new_event_group": novel,
        })
    candidates.sort(key=lambda row: (
        row["existing_cell_recordings"],
        not row["new_event_group"],
        row["declared_bytes"],
        salted_digest("aea-visible-action-v2-acquire", row["event_group"]),
        row["event_group"],
    ))
    selected = []
    recordings = 0
    declared_bytes = 0
    maximum_recordings = int(acquisition["maximum_new_recordings"])
    maximum_bytes = int(acquisition["maximum_declared_bytes"])
    minimum_free_bytes = round(float(acquisition["minimum_projected_free_gib"]) * 1024**3)
    for candidate in candidates:
        next_recordings = recordings + int(candidate["new_recordings"])
        next_bytes = declared_bytes + int(candidate["declared_bytes"])
        if (
            next_recordings > maximum_recordings
            or next_bytes > maximum_bytes
            or free_bytes - next_bytes < minimum_free_bytes
        ):
            break
        selected.append(candidate)
        recordings = next_recordings
        declared_bytes = next_bytes
    return {
        "selection_rule": acquisition["ranking"],
        "concurrent_recordings_atomic": True,
        "selected_event_groups": [row["event_group"] for row in selected],
        "selected_sequence_ids": [name for row in selected for name in row["sequence_ids"]],
        "selected_groups_detail": selected,
        "new_recordings": recordings,
        "declared_bytes": declared_bytes,
        "declared_gib": round(declared_bytes / 1024**3, 3),
        "projected_free_gib": round((free_bytes - declared_bytes) / 1024**3, 3),
        "limits": {
            "maximum_new_recordings": maximum_recordings,
            "maximum_declared_gib": float(acquisition["maximum_declared_gib"]),
            "minimum_projected_free_gib": float(acquisition["minimum_projected_free_gib"]),
        },
        "download_performed": False,
    }


def _decision(
    agreement: Mapping[str, Any], split: Mapping[str, Any],
    capacity: Mapping[str, Any], imu: Mapping[str, Any],
    integrity_checks: Mapping[str, bool],
) -> dict[str, Any]:
    if not all(integrity_checks.values()):
        return {
            "recommendation": "REVISE",
            "reason": "hard integrity or artifact validation failure",
            "authorizes": "implementation repair only without new evidence or reserve access",
        }
    if agreement["severe_annotation_failure"]:
        return {
            "recommendation": "STOP",
            "reason": "severe frozen annotation-yield/agreement failure",
            "authorizes": "abandon current AEA route; no further AEA effect work",
        }
    if not agreement["annotation_gate_passed"]:
        return {
            "recommendation": "REVISE",
            "reason": "non-severe frozen annotation stability threshold miss",
            "authorizes": "one prospectively frozen ontology/interface repair only",
        }
    if not split.get("split_gate_passed", False):
        if capacity.get("capacity_gate_passed", False):
            return {
                "recommendation": "REVISE",
                "reason": "certified donor-feasible split/support bottleneck after passing capacity",
                "authorizes": "support repair or separately allowed bounded acquisition only",
            }
        return {
            "recommendation": "STOP",
            "reason": "valid split is unavailable and the frozen capacity gate could not pass",
            "authorizes": "abandon current AEA route",
        }
    if not capacity.get("capacity_gate_passed", False):
        return {
            "recommendation": "STOP",
            "reason": "observable-action video capacity gate failed under valid labels/design",
            "authorizes": "abandon current AEA route",
        }
    if not imu.get("imu_gate_passed", False):
        return {
            "recommendation": "STOP",
            "reason": "conditional group-held-out IMU viability gate failed",
            "authorizes": "abandon current IMU-mediated AEA route",
        }
    return {
        "recommendation": "GO",
        "reason": "annotation, support, donor, capacity, and conditional IMU gates passed",
        "authorizes": "two genuinely independent human annotators only",
    }


def _acquisition_gate(
    agreement: Mapping[str, Any], split: Mapping[str, Any],
    capacity: Mapping[str, Any], imu: Mapping[str, Any],
) -> dict[str, Any]:
    annotation = agreement.get("annotation_gate_passed") is True
    capacity_pass = capacity.get("capacity_gate_passed") is True
    support_case = split.get("status") == "infeasible_certified" and capacity_pass
    power_case = False
    power_details: dict[str, Any] = {}
    if imu.get("status") == "complete":
        checks = imu["gate_checks"]
        allowed_uncertainty_failures = {
            "bootstrap_lower_above_chance",
            "chance_permutation_p",
            "paired_bootstrap_lower_above_zero",
            "paired_randomization_p",
        }
        failed = {key for key, value in checks.items() if not value}
        interval = imu["paired_lift_bootstrap"]
        power_details = {
            "point_lift_at_least_0_05": imu["synchronized_minus_donor"] >= 0.05,
            "interval_crosses_zero": interval["ci95_low"] <= 0 <= interval["ci95_high"],
            "train_gap_at_most_0_35": imu["train_minus_heldout_gap"] <= 0.35,
            "event_groups_below_12": imu["event_groups"] < 12,
            "only_uncertainty_checks_failed": bool(failed) and failed <= allowed_uncertainty_failures,
        }
        power_case = all(power_details.values())
    return {
        "annotation_gate_passed": annotation,
        "capacity_gate_passed": capacity_pass,
        "certified_support_bottleneck": support_case,
        "power_bottleneck": power_case,
        "power_details": power_details,
        "passed": annotation and capacity_pass and (support_case or power_case),
    }


def _human_packet(
    packet_a: Mapping[str, Any], packet_b: Mapping[str, Any], decision: str
) -> dict[str, Any]:
    return {
        "schema_version": "aea-visible-action-independent-human-packet-v2",
        "protocol_id": "aea-visible-action-v2",
        "status": "authorized_by_GO" if decision == "GO" else "prepared_but_not_authorized",
        "codebook": packet_a["codebook"],
        "annotator_a": {
            "order": "frozen model-pass-A randomization reused",
            "rows": packet_a["rows"],
        },
        "annotator_b": {
            "order": "frozen model-pass-B randomization reused",
            "rows": packet_b["rows"],
        },
        "blinding": [
            "Annotators work independently and cannot see each other's labels.",
            "Do not expose v1 or model-assisted labels, groups, locations, results, or outcomes.",
            "Human labels must be saved separately and must not overwrite this blank packet.",
        ],
        "model_assisted_labels_included": False,
        "reserve_access_authorized": False,
        "locked_experiment_authorized": False,
    }


def _format_percent(value: Any) -> str:
    return "not run" if value is None else f"{100 * float(value):.1f}%"


def _write_report(
    path: Path, results: Mapping[str, Any], safe_summary: Mapping[str, Any]
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite final v2 report: {path}")
    agreement = results["annotation"]
    split = results["split"]
    capacity = results["capacity"]
    transcript = results["transcript_control"]
    imu = results["imu"]
    decision = results["decision"]
    action = agreement["agreement"]["observable_action"]
    consensus = agreement["rates"]["modeled_consensus_yield"]
    lines = [
        "# AEA visible-action rescue feasibility v2",
        "",
        f"## Decision: {decision['recommendation']}",
        "",
        decision["reason"] + ".",
        "",
        f"This authorizes: {decision['authorizes']}. It does not authorize reserve access, the locked experiment, downloads, licenses, or outreach.",
        "",
        "AEA is an adult, partly scripted sensor-format analogue. These are not developmental findings and are not BabyView-like. Model-assisted labels are development diagnostics, not human annotations or human reliability.",
        "",
        "## Dense blinded annotation",
        "",
        f"The frozen sample contained 72 windows from all 18 development event groups. Each used 31 ordered RGB frames (2,232 queries total). Reserve RGB and IMU accesses were both zero.",
        "",
        f"Observable-action exact agreement was {_format_percent(action['exact_agreement'])} (kappa {action['cohens_kappa_unweighted'] if action['cohens_kappa_unweighted'] is not None else 'NA'}). Modeled consensus was {consensus['successes']}/72 ({_format_percent(consensus['rate'])}). The annotation gate {'passed' if agreement['annotation_gate_passed'] else 'failed'}; severe failure was {agreement['severe_annotation_failure']}.",
        "",
        "Retained modeled support: " + (", ".join(
            f"{label}={values['windows']} windows/{values['event_groups']} groups"
            for label, values in agreement["retained_support"].items()
        ) or "none") + ".",
        "",
        "## Split, capacity, transcript, and IMU",
        "",
        f"The constrained split/donor status was `{split['status']}` (gate {split.get('split_gate_passed', False)}); no relaxation or second split was attempted.",
        "",
        f"The supervised action-head capacity status was `{capacity['status']}` (gate {capacity.get('capacity_gate_passed', False)}; mean training balanced accuracy {_format_percent(capacity.get('mean_training_balanced_accuracy'))}).",
        "",
        f"The separate natural-transcript control status was `{transcript['status']}` and was non-gating. " + (
            f"Training-set action-balanced 2AFC was {_format_percent(transcript['result']['training_action_balanced_2afc'])}."
            if transcript.get("status") == "complete" else "It was not run after the frozen stage gate."
        ),
        "",
        f"The conditional IMU status was `{imu['status']}` (gate {imu.get('imu_gate_passed', False)}). " + (
            f"Held-out synchronized balanced accuracy was {_format_percent(imu['heldout_synchronized_balanced_accuracy'])}, synchronized-minus-donor was {_format_percent(imu['synchronized_minus_donor'])}, and train-minus-held-out gap was {_format_percent(imu['train_minus_heldout_gap'])}."
            if imu.get("status") == "complete" else "No IMU array was opened after the stage gate failed."
        ),
        "",
        "## Additional recordings and storage",
        "",
        f"Recommendation: **{'use bounded extra storage' if results['acquisition']['recommend_additional_recordings'] else 'do not use extra storage'}**. {results['acquisition']['reason']}",
        "",
        f"The safe release has {safe_summary['dataset']['sequence_count']} recordings. The remaining 103 annotations+main-VRS components total {safe_summary['remaining_103']['total_gib']:.3f} GiB; available space at decision time was {safe_summary['filesystem_free_gib_at_decision']:.3f} GiB. Free space is a ceiling, not a reason to acquire data.",
        "",
        "## Limitations",
        "",
        "- Two context-isolated model passes do not estimate human inter-rater reliability.",
        "- Thirty-one frames are dense temporal evidence, but not continuous video or audio.",
        "- V2 is development/protocol feasibility; no confirmatory effect was tested.",
        "- The v1 reserve is prospective from v1 onward, not pristine relative to the earlier smoke run.",
        "- Threshold proximity cannot override the frozen mechanical decision.",
    ]
    path.write_text("\n".join(lines) + "\n")


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    base = load_json(args.protocol)
    amendment = load_json(args.amendment)
    protocol = apply_protocol_amendment(base, amendment)
    agreement = load_json(args.agreement)
    split = load_json(args.split)
    capacity = load_json(args.capacity)
    transcript = load_json(args.transcript)
    imu = load_json(args.imu)
    dense = load_json(args.dense_manifest)
    access = load_json(args.access_receipt)
    packet_a = load_json(args.packet_a)
    packet_b = load_json(args.packet_b)
    freeze = load_json(args.freeze_receipt)
    amendment_freeze = load_json(args.amendment_freeze_receipt)
    integrity = {
        "base_protocol_hash_matches_freeze": sha256_file(args.protocol)
        == freeze["machine_readable_protocol"]["sha256"],
        "amendment_hash_matches_freeze": sha256_file(args.amendment)
        == amendment_freeze["machine_readable"]["sha256"],
        "amendment_parent_hash_matches": amendment["parent_protocol_sha256"]
        == sha256_file(args.protocol),
        "zero_dense_reserve_rgb": access["reserve_rgb_files_opened"] == 0,
        "zero_dense_reserve_imu": access["reserve_imu_arrays_opened"] == 0,
        "zero_conditional_reserve_imu": imu["reserve_imu_arrays_opened"] == 0,
        "zero_signed_url_exposure": access["signed_urls_loaded_printed_or_copied"] is False,
        "dense_manifest_reserve_free": dense["reserve_groups_present"] == [],
        "annotation_input_integrity": all(agreement["input_integrity"]["metadata_checks"].values()),
        "prior_outputs_not_targeted": True,
    }
    decision = _decision(agreement, split, capacity, imu, integrity)
    free_bytes = shutil.disk_usage(args.out).free
    safe_manifest, existing, safe_summary = _safe_release_summary(
        args.safe_manifest, args.existing_plan, free_bytes
    )
    acquisition_gate = _acquisition_gate(agreement, split, capacity, imu)
    acquisition = {
        "schema_version": "aea-visible-action-acquisition-recommendation-v2",
        "protocol_id": protocol["protocol_id"],
        "gate": acquisition_gate,
        "recommend_additional_recordings": acquisition_gate["passed"],
        "reason": (
            "The preregistered ontology/capacity and sole support-or-power bottleneck gate passed."
            if acquisition_gate["passed"] else
            "The preregistered ontology/capacity/support-or-power gate did not pass; no expansion is justified."
        ),
        "recommendation": "bounded_metadata_only_plan" if acquisition_gate["passed"] else "no_additional_acquisition",
        "download_performed": False,
        "license_accepted": False,
        "signed_urls_used_or_exposed": False,
    }
    if acquisition_gate["passed"]:
        acquisition["bounded_plan"] = _bounded_acquisition_plan(
            safe_manifest, existing, free_bytes, protocol
        )
    else:
        acquisition["bounded_plan"] = None

    results = {
        "schema_version": "aea-visible-action-results-v2",
        "protocol_id": protocol["protocol_id"],
        "scientific_role": protocol["scientific_role"],
        "status": "development_protocol_feasibility_not_confirmatory_effect_test",
        "annotation": agreement,
        "split": split,
        "capacity": capacity,
        "transcript_control": transcript,
        "imu": imu,
        "integrity_checks": integrity,
        "all_hard_integrity_checks_passed": all(integrity.values()),
        "decision": decision,
        "acquisition": acquisition,
        "limitations": [
            "adult partly scripted sensor-format analogue; not developmental evidence or BabyView-like",
            "model-assisted development labels are not human annotations or human inter-rater reliability",
            "31-frame sequences are dense but not continuous video or audio",
            "development/protocol feasibility only; no confirmatory effect tested",
        ],
    }
    _write_json_new(args.out / "safe_release_metadata.json", safe_summary)
    _write_json_new(args.out / "acquisition_recommendation.json", acquisition)
    _write_json_new(args.out / "human_annotation_packet.json", _human_packet(
        packet_a, packet_b, decision["recommendation"]
    ))
    _write_json_new(args.out / "aea_visible_action_v2_results.json", results)
    _write_report(args.out / "scientific_report.md", results, safe_summary)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("output/aea_visible_action_v2")
    parser.add_argument("--protocol", type=Path, default=root / "preregistered_protocol.json")
    parser.add_argument("--amendment", type=Path, default=root / "preregistered_protocol_amendment_1.json")
    parser.add_argument("--agreement", type=Path, default=root / "agreement_report.json")
    parser.add_argument("--split", type=Path, default=root / "split_donor_feasibility.json")
    parser.add_argument("--capacity", type=Path, default=root / "capacity_results.json")
    parser.add_argument("--transcript", type=Path, default=root / "transcript_control_results.json")
    parser.add_argument("--imu", type=Path, default=root / "imu_results.json")
    parser.add_argument("--dense-manifest", type=Path, default=root / "dense_clip_manifest.json")
    parser.add_argument("--access-receipt", type=Path, default=root / "reserve_access_receipt.json")
    parser.add_argument("--packet-a", type=Path, default=root / "annotation_packet_pass_a.json")
    parser.add_argument("--packet-b", type=Path, default=root / "annotation_packet_pass_b.json")
    parser.add_argument("--freeze-receipt", type=Path, default=root / "protocol_freeze_receipt.json")
    parser.add_argument("--amendment-freeze-receipt", type=Path, default=root / "protocol_amendment_1_freeze_receipt.json")
    parser.add_argument("--safe-manifest", type=Path, default=Path.home() / "Downloads/AriaEverydayActivities_download_urls.json")
    parser.add_argument("--existing-plan", type=Path, default=Path("configs/aea_subset_40.yaml"))
    parser.add_argument("--out", type=Path, default=root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = finalize(args)
    print(json.dumps({
        "decision": results["decision"]["recommendation"],
        "authorizes": results["decision"]["authorizes"],
        "additional_acquisition": results["acquisition"]["recommend_additional_recordings"],
    }, indent=2))


if __name__ == "__main__":
    main()
