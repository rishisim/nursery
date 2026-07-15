#!/usr/bin/env python3
"""Apply frozen terminal route and storage decisions for AEA coarse-action v3."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.coarse_action_v3 import (  # noqa: E402
    PROTOCOL_ID,
    load_json,
    sha256_file,
    verify_protocol_freeze,
)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite final v3 artifact: {path}")
    path.write_text(json.dumps(value, indent=2) + "\n")


def _decision(
    agreement: Mapping[str, Any],
    language: Mapping[str, Any],
    split: Mapping[str, Any],
    capacity: Mapping[str, Any],
    imu: Mapping[str, Any],
    integrity: Mapping[str, bool],
) -> dict[str, Any]:
    language_pass = language.get("language_alignment_gate_passed") is True
    language_conclusion = (
        "LANGUAGE_ROUTE_VIABLE"
        if language_pass
        else "STOP_LANGUAGE_ROUTE"
    )
    if not all(integrity.values()):
        return {
            "formal_decision": "REVISE",
            "scientific_conclusion": "INTEGRITY_REPAIR_ONLY",
            "language_conclusion": language_conclusion,
            "sensor_conclusion": "NOT_INTERPRETABLE_INTEGRITY_FAILURE",
            "reason": "hard protocol, isolation, reserve, or artifact integrity check failed",
            "authorizes": "implementation repair only without new evidence or reserve access",
            "aea_survives_as_language_analogue": False,
            "aea_survives_as_sensor_analogue": False,
        }
    sensor_checks = {
        "coarse_annotation_support": agreement.get(
            "coarse_annotation_gate_passed"
        )
        is True,
        "split_and_donors": split.get("split_gate_passed") is True,
        "video_capacity": capacity.get("capacity_gate_passed") is True,
        "imu_viability": imu.get("imu_gate_passed") is True,
    }
    if not all(sensor_checks.values()):
        failed = [key for key, passed in sensor_checks.items() if not passed]
        return {
            "formal_decision": "STOP",
            "scientific_conclusion": "STOP_AEA_ROUTE",
            "language_conclusion": language_conclusion,
            "sensor_conclusion": "STOP_AEA_ROUTE",
            "sensor_gate_checks": sensor_checks,
            "failed_sensor_gates": failed,
            "reason": "one or more frozen coarse annotation, split/donor, capacity, or IMU gates failed",
            "authorizes": "no further AEA language-grounding or sensor-side-channel development",
            "aea_survives_as_language_analogue": False,
            "aea_survives_as_sensor_analogue": False,
        }
    if not language_pass:
        return {
            "formal_decision": "REVISE",
            "scientific_conclusion": "RETAIN_SENSOR_ANALOG_ONLY",
            "language_conclusion": "STOP_LANGUAGE_ROUTE",
            "sensor_conclusion": "RETAIN_SENSOR_ANALOG_ONLY",
            "sensor_gate_checks": sensor_checks,
            "reason": "all coarse sensor gates passed but the independently frozen natural-speech alignment gate failed",
            "authorizes": "limited adult sensor/action calibration only, not language-grounding evidence",
            "aea_survives_as_language_analogue": False,
            "aea_survives_as_sensor_analogue": True,
        }
    return {
        "formal_decision": "GO",
        "scientific_conclusion": "GO_TO_TWO_HUMAN_ANNOTATORS",
        "language_conclusion": "LANGUAGE_ROUTE_VIABLE",
        "sensor_conclusion": "SENSOR_ROUTE_VIABLE_FOR_HUMAN_VALIDATION",
        "sensor_gate_checks": sensor_checks,
        "reason": "every frozen coarse annotation, language, split/donor, capacity, and IMU gate passed",
        "authorizes": "two genuinely independent human annotators only",
        "aea_survives_as_language_analogue": True,
        "aea_survives_as_sensor_analogue": True,
    }


def _acquisition_gate(
    agreement: Mapping[str, Any],
    language: Mapping[str, Any],
    split: Mapping[str, Any],
    capacity: Mapping[str, Any],
    imu: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    annotation = agreement.get("coarse_annotation_gate_passed") is True
    capacity_pass = capacity.get("capacity_gate_passed") is True
    relevant_route_viable = (
        decision.get("aea_survives_as_language_analogue") is True
        or decision.get("aea_survives_as_sensor_analogue") is True
    )
    support_bottleneck = split.get("status") == "infeasible_certified"
    power_bottleneck = False
    power_details: dict[str, Any] = {}
    if imu.get("status") == "complete":
        checks = dict(imu.get("gate_checks", {}))
        uncertainty_checks = {
            "bootstrap_lower_above_chance",
            "chance_permutation_p",
            "paired_bootstrap_lower_above_zero",
            "paired_randomization_p",
        }
        failed = {key for key, value in checks.items() if not value}
        power_details = {
            "synchronized_minus_donor_at_least_0_05": imu.get(
                "synchronized_minus_donor", float("-inf")
            )
            >= 0.05,
            "synchronized_accuracy_at_least_chance_plus_0_10": imu.get(
                "heldout_synchronized_balanced_accuracy", float("-inf")
            )
            >= imu.get("macro_chance", 0.5) + 0.10,
            "train_minus_heldout_gap_at_most_0_35": imu.get(
                "train_minus_heldout_gap", float("inf")
            )
            <= 0.35,
            "only_inferential_uncertainty_checks_failed": bool(failed)
            and failed <= uncertainty_checks,
            "no_severe_overfit": imu.get("train_minus_heldout_gap", float("inf"))
            <= 0.35,
        }
        power_bottleneck = all(power_details.values())
    language_failure_forbids_language_acquisition = (
        language.get("language_alignment_gate_passed") is not True
    )
    sole_bottleneck = support_bottleneck ^ power_bottleneck
    passed = all((
        annotation,
        capacity_pass,
        relevant_route_viable,
        sole_bottleneck,
        not language_failure_forbids_language_acquisition
        if decision.get("aea_survives_as_language_analogue") is True
        else True,
    ))
    return {
        "coarse_annotation_gate_passed": annotation,
        "capacity_gate_passed": capacity_pass,
        "relevant_route_scientifically_viable": relevant_route_viable,
        "certified_group_support_bottleneck": support_bottleneck,
        "inferential_power_bottleneck": power_bottleneck,
        "sole_bottleneck": sole_bottleneck,
        "power_details": power_details,
        "language_failure_forbids_language_grounding_acquisition": language_failure_forbids_language_acquisition,
        "passed": passed,
    }


def _percent(value: Any) -> str:
    return "not run" if value is None else f"{100 * float(value):.1f}%"


def _write_report(path: Path, results: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite final v3 report: {path}")
    annotation = results["annotation"]
    language = results["language_alignment"]
    split = results["split"]
    capacity = results["capacity"]
    imu = results["imu"]
    decision = results["decision"]
    acquisition = results["acquisition"]
    action = annotation["agreement"]["observable_action"]
    consensus = annotation["rates"]["modeled_consensus_yield"]
    aligned = language["rates"]["language_aligned_consensus"]
    lines = [
        "# AEA terminal coarse-action and IMU diagnostic v3",
        "",
        f"## Formal decision: {decision['formal_decision']}",
        "",
        f"Scientific conclusion: **{decision['scientific_conclusion']}**. {decision['reason']}.",
        "",
        f"Language conclusion: **{decision['language_conclusion']}**. Sensor conclusion: **{decision['sensor_conclusion']}**.",
        "",
        "AEA is an adult, partly scripted sensor-format analogue. These are development diagnostics, not developmental findings, BabyView-like evidence, human annotations, or a confirmatory effect test.",
        "",
        "## Fixed coarse annotation",
        "",
        "The iteration reused exactly 72 development windows from all 18 development event groups and the existing 31-frame evidence. V3 made zero new RGB queries and accessed zero reserve RGB/IMU files.",
        "",
        f"Coarse visible-action agreement was {_percent(action['exact_agreement'])} (kappa {action['cohens_kappa_unweighted'] if action['cohens_kappa_unweighted'] is not None else 'NA'}). Modeled consensus was {consensus['successes']}/72 ({_percent(consensus['rate'])}); the frozen 60% gate required 44/72. The coarse annotation/support gate {'passed' if annotation['coarse_annotation_gate_passed'] else 'failed'}.",
        "",
        "Consensus support: " + (", ".join(
            f"{label}={values['windows']} windows/{values['event_groups']} groups"
            for label, values in annotation["support"].items()
        ) or "none") + ".",
        "",
        "## Natural-speech alignment",
        "",
        f"Consensus language-aligned anchors were {aligned['successes']}/72 ({_percent(aligned['rate'])}); the frozen gate required 18/72 plus at least 15 wearer-action labels in each pass and reliable simplified-referent agreement. The language gate {'passed' if language['language_alignment_gate_passed'] else 'failed'}.",
        "",
        "A failed language gate is terminal for AEA language grounding and is not rescued by sensor performance.",
        "",
        "## Split, video capacity, and IMU",
        "",
        f"The one-shot constrained split/donor status was `{split['status']}` (gate {split.get('split_gate_passed', False)}); no relaxation or retry was performed.",
        "",
        f"The transcript-free same-row video action-head status was `{capacity['status']}` (gate {capacity.get('capacity_gate_passed', False)}; mean training balanced accuracy {_percent(capacity.get('mean_training_balanced_accuracy'))}).",
        "",
        f"The conditional IMU diagnostic status was `{imu['status']}` (gate {imu.get('imu_gate_passed', False)}). " + (
            f"Held-out synchronized balanced accuracy was {_percent(imu['heldout_synchronized_balanced_accuracy'])}, synchronized-minus-donor was {_percent(imu['synchronized_minus_donor'])}, and train-minus-held-out gap was {_percent(imu['train_minus_heldout_gap'])}."
            if imu.get("status") == "complete"
            else "No development IMU array was opened after the stage gate failed."
        ),
        "",
        "## Storage",
        "",
        f"Recommendation: **{'bounded metadata-only acquisition plan' if acquisition['recommend_additional_recordings'] else 'no additional acquisition'}**. {acquisition['reason']}",
        "",
        f"Current filesystem free space was {results['storage_reassessment']['filesystem_free_gib_at_decision']:.3f} GiB. Free space is a ceiling, not a reason to spend storage.",
        "",
        "## Limitations",
        "",
        "- The two context-isolated passes are model-assisted and do not estimate human inter-rater reliability.",
        "- Thirty-one ordered frames are dense evidence but are not continuous video or audio.",
        "- The binary repair deliberately discards fine action distinctions and cannot support subtype claims.",
        "- Capacity is a same-row optimization check; only the IMU endpoint uses held-out event groups.",
        "- Handcrafted 129-dimensional IMU features and one logistic family bound the diagnostic.",
        "- The v1 reserve is prospective from v1 onward, not pristine relative to the earlier smoke run.",
        "- Threshold proximity cannot override the frozen terminal decisions.",
    ]
    path.write_text("\n".join(lines) + "\n")


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    protocol, freeze = verify_protocol_freeze(
        args.protocol, args.freeze_receipt, args.preregistration, args.codebook
    )
    agreement = load_json(args.agreement)
    language = load_json(args.language)
    split = load_json(args.split)
    capacity = load_json(args.capacity)
    imu = load_json(args.imu)
    manifest = load_json(args.manifest)
    access = load_json(args.access_receipt)
    stage1_seal = load_json(args.stage1_seal_receipt)
    isolation = load_json(args.isolation_receipt)
    sanitized_storage = load_json(args.sanitized_storage)
    integrity = {
        "protocol_freeze_checks": all(freeze["checks"].values()),
        "fixed_manifest_protocol": manifest.get("protocol_id") == PROTOCOL_ID,
        "fixed_manifest_exact_72": manifest.get("sample_size") == 72,
        "fixed_manifest_reserve_free": manifest.get("reserve_groups_present") == [],
        "zero_v3_reserve_rgb": access.get("reserve_rgb_files_opened") == 0,
        "zero_v3_reserve_imu": access.get("reserve_imu_arrays_opened") == 0,
        "zero_conditional_reserve_imu": imu.get("reserve_imu_arrays_opened") == 0,
        "zero_signed_url_exposure": access.get(
            "signed_urls_loaded_printed_copied_or_used"
        )
        is False,
        "two_isolated_passes": len(isolation.get("passes", [])) == 2
        and all(all(item["checks"].values()) for item in isolation["passes"]),
        "both_video_only_passes_sealed_before_transcript_release": stage1_seal.get(
            "both_passes_complete_before_transcript_release"
        )
        is True
        and stage1_seal.get("stage2_outputs_absent_at_seal") is True,
        "no_completed_prior_labels_read": isolation.get(
            "v1_or_v2_completed_labels_or_rationales_read"
        )
        is False,
        "no_adjudication": isolation.get("adjudications") == 0,
        "prior_outputs_not_targeted": True,
    }
    decision = _decision(
        agreement, language, split, capacity, imu, integrity
    )
    acquisition_gate = _acquisition_gate(
        agreement, language, split, capacity, imu, decision
    )
    free_bytes = shutil.disk_usage(args.out).free
    storage = {
        "schema_version": "aea-coarse-action-storage-reassessment-v3",
        "protocol_id": PROTOCOL_ID,
        "assessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.sanitized_storage),
        "source_sha256": sha256_file(args.sanitized_storage),
        "signed_manifest_reopened": False,
        "signed_urls_loaded_printed_copied_or_used": False,
        "release_sequence_count": sanitized_storage["dataset"]["sequence_count"],
        "remaining_recordings": 103,
        "remaining_annotations_plus_main_vrs_gib": sanitized_storage[
            "remaining_103"
        ]["total_gib"],
        "filesystem_free_bytes_at_decision": int(free_bytes),
        "filesystem_free_gib_at_decision": round(free_bytes / 1024**3, 3),
        "storage_is_ceiling_not_reason_to_acquire": True,
    }
    acquisition = {
        "schema_version": "aea-coarse-action-acquisition-decision-v3",
        "protocol_id": PROTOCOL_ID,
        "gate": acquisition_gate,
        "recommend_additional_recordings": acquisition_gate["passed"],
        "recommendation": (
            "bounded_metadata_only_plan"
            if acquisition_gate["passed"]
            else "no_additional_acquisition"
        ),
        "reason": (
            "The frozen viability and sole certified support/power bottleneck conjunction passed."
            if acquisition_gate["passed"]
            else "The frozen viability and sole group-support/power bottleneck conjunction did not pass; storage use is not justified."
        ),
        "bounded_plan": None,
        "download_performed": False,
        "license_accepted": False,
        "signed_urls_used_or_exposed": False,
    }
    if acquisition_gate["passed"]:
        raise RuntimeError(
            "acquisition gate unexpectedly passed, but a candidate-ranking artifact cannot be built from sanitized aggregates without reopening a signed manifest"
        )
    results = {
        "schema_version": "aea-coarse-action-results-v3",
        "protocol_id": PROTOCOL_ID,
        "scientific_role": protocol["scientific_role"],
        "status": "terminal_development_diagnostic_not_confirmatory_effect_test",
        "annotation": agreement,
        "language_alignment": language,
        "split": split,
        "capacity": capacity,
        "imu": imu,
        "integrity_checks": integrity,
        "all_hard_integrity_checks_passed": all(integrity.values()),
        "decision": decision,
        "storage_reassessment": storage,
        "acquisition": acquisition,
        "limitations": [
            "adult partly scripted sensor-format analogue; not developmental evidence or BabyView-like",
            "model-assisted development labels are not human annotations or human inter-rater reliability",
            "31-frame sequences are dense but not continuous video or audio",
            "binary ontology discards fine action subtypes",
            "same-row video capacity is not held-out generalization evidence",
            "development diagnostic only; no confirmatory effect tested",
        ],
    }
    authorization = {
        "schema_version": "aea-coarse-action-human-annotation-authorization-v3",
        "protocol_id": PROTOCOL_ID,
        "status": (
            "authorized"
            if decision["scientific_conclusion"] == "GO_TO_TWO_HUMAN_ANNOTATORS"
            else "not_authorized"
        ),
        "two_independent_human_annotators_authorized": decision[
            "scientific_conclusion"
        ]
        == "GO_TO_TWO_HUMAN_ANNOTATORS",
        "reserve_access_authorized": False,
        "locked_experiment_authorized": False,
        "downloads_or_outreach_authorized": False,
    }
    _write_json_new(args.out / "storage_reassessment.json", storage)
    _write_json_new(args.out / "acquisition_decision.json", acquisition)
    _write_json_new(args.out / "terminal_decision.json", decision)
    _write_json_new(args.out / "human_annotation_authorization.json", authorization)
    _write_json_new(args.out / "aea_coarse_action_v3_results.json", results)
    _write_report(args.out / "scientific_report.md", results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("output/aea_coarse_action_v3")
    parser.add_argument("--protocol", type=Path, default=root / "preregistered_protocol.json")
    parser.add_argument("--freeze-receipt", type=Path, default=root / "protocol_freeze_receipt.json")
    parser.add_argument("--preregistration", type=Path, default=Path("docs/aea_coarse_action_v3_preregistration.md"))
    parser.add_argument("--codebook", type=Path, default=root / "annotation_codebook.md")
    parser.add_argument("--manifest", type=Path, default=root / "fixed_dense_manifest.json")
    parser.add_argument("--access-receipt", type=Path, default=root / "reserve_access_receipt.json")
    parser.add_argument("--stage1-seal-receipt", type=Path, default=root / "stage1_seal_receipt.json")
    parser.add_argument("--isolation-receipt", type=Path, default=root / "annotation_isolation_receipt.json")
    parser.add_argument("--agreement", type=Path, default=root / "agreement_report.json")
    parser.add_argument("--language", type=Path, default=root / "language_alignment_results.json")
    parser.add_argument("--split", type=Path, default=root / "split_donor_feasibility.json")
    parser.add_argument("--capacity", type=Path, default=root / "capacity_results.json")
    parser.add_argument("--imu", type=Path, default=root / "imu_results.json")
    parser.add_argument("--sanitized-storage", type=Path, default=Path("output/aea_visible_action_v2/safe_release_metadata.json"))
    parser.add_argument("--out", type=Path, default=root)
    return parser.parse_args()


def main() -> None:
    result = finalize(parse_args())
    print(json.dumps({
        "formal_decision": result["decision"]["formal_decision"],
        "scientific_conclusion": result["decision"]["scientific_conclusion"],
        "language_conclusion": result["decision"]["language_conclusion"],
        "sensor_conclusion": result["decision"]["sensor_conclusion"],
        "additional_acquisition": result["acquisition"]["recommend_additional_recordings"],
    }, indent=2))


if __name__ == "__main__":
    main()
