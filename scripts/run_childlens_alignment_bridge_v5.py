#!/usr/bin/env python3
"""Generate and validate the fail-closed ChildLens v5 Stage 0 record."""

from __future__ import annotations

from collections.abc import Sequence
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/childlens_alignment_bridge_v5.json"
PACKAGE = ROOT / "babyworld_lite/childlens_alignment_bridge_v5/preflight.py"
PROTOCOL = ROOT / "docs/childlens_alignment_bridge_v5_protocol.md"
PUBLIC_ROOT = ROOT / "output/childlens_alignment_bridge_v5"
FREEZE_RECEIPT = PUBLIC_ROOT / "stage0_freeze_and_stop_receipt.json"
POSITIVE_RECEIPT = PUBLIC_ROOT / "synthetic_positive_control.json"
DECISION_REPORT = PUBLIC_ROOT / "development_decision.json"
DECISION_MARKDOWN = PUBLIC_ROOT / "development_decision.md"
VALIDATION_RECEIPT = PUBLIC_ROOT / "validation_receipt.json"

PRIOR_SCOPES = {
    "v1": [
        "configs/childlens_alignment_bridge_v1.json",
        "babyworld_lite/childlens_alignment_bridge_v1",
        "output/childlens_alignment_bridge_v1",
        "scripts/run_childlens_alignment_bridge_preflight_v1.py",
    ],
    "v2": [
        "configs/childlens_alignment_bridge_remediation_v2.json",
        "babyworld_lite/childlens_alignment_bridge_v2",
        "output/childlens_alignment_bridge_remediation_v2",
        "scripts/run_childlens_alignment_bridge_remediation_v2.py",
    ],
    "v3": [
        "configs/childlens_alignment_bridge_expansion_v3.json",
        "babyworld_lite/childlens_alignment_bridge_v3",
        "output/childlens_alignment_bridge_expansion_v3",
        "scripts/run_childlens_alignment_bridge_expansion_v3.py",
        "scripts/acquire_childlens_alignment_bridge_expansion_v3.py",
        "scripts/measure_childlens_alignment_bridge_expansion_v3.py",
    ],
    "v4": [
        "configs/childlens_alignment_bridge_v4.json",
        "babyworld_lite/childlens_alignment_bridge_v4",
        "output/childlens_alignment_bridge_v4",
        "scripts/run_childlens_alignment_bridge_v4.py",
    ],
}
EXPECTED_PRIOR_TREES = {
    "v1": "4f668c666636e47e72a3c2162cb30590f89077f0e4e6ef8e69b149f4966df679",
    "v2": "30df10121591465cd4c357ead6b4aab8c2dff30b2a292106faa277661c621e45",
    "v3": "29718e4741bf31f27e5001e6a6398423745b1d5b8688032b3e9140a3f5ef90fa",
    "v4": "d9b62ecd5c747a22779ea0624bca7c5eca707da86c1727d6abce812ecbc673d2",
}
EXPECTED_DEVELOPMENT_MANIFEST_HASHES = {
    "immutable_original_eight": (
        "9cb87b853eb43636d6baf09c281725eb98255b6cf73439025b47d627e84da5a8"
    ),
    "immutable_v3_expansion_ten": (
        "028efa424ed4dc2fe511a0b723b38f4feccba7f4de7db055934218dff1fe705d"
    ),
}

sys.path.insert(0, str(ROOT))
from babyworld_lite.childlens_alignment_bridge_v5.preflight import (  # noqa: E402
    BridgeV5Error,
    canonical_bytes,
    digest,
    public_guard,
    run_embedding_positive_control,
    terminal_decision,
    trainable_parameter_count,
)


def _sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def _tree_digest(entries: Sequence[str]) -> str:
    rows: list[dict[str, str]] = []
    for entry in entries:
        path = ROOT / entry
        files = (
            [path]
            if path.is_file()
            else sorted(
                value
                for value in path.rglob("*")
                if value.is_file()
                and "__pycache__" not in value.parts
                and value.suffix != ".pyc"
            )
        )
        if not files:
            raise BridgeV5Error("E_PRIOR_TREE")
        for value in files:
            rows.append(
                {
                    "path": value.relative_to(ROOT).as_posix(),
                    "sha256": _sha256_file(value),
                }
            )
    return digest(rows)


def _validate_prior_trees() -> dict[str, str]:
    observed = {
        version: _tree_digest(entries) for version, entries in PRIOR_SCOPES.items()
    }
    if observed != EXPECTED_PRIOR_TREES:
        raise BridgeV5Error("E_PRIOR_IMMUTABILITY")
    return observed


def _config() -> dict[str, Any]:
    value = _read_json(CONFIG)
    if (
        value.get("schema_version")
        != "childlens-calibration-recovery-scale-lag-v5.0.0"
        or value.get("scope", {}).get("development_participants") != 18
        or value.get("scope", {}).get(
            "locked_rows_may_be_loaded_inspected_scored_summarized_or_decoded"
        )
        is not False
        or value.get("learner", {}).get("route_count") != 1
        or value.get("learner", {}).get("alternative_search_after_childlens_outcomes")
        is not False
        or value.get("instruments", {})
        .get("audio", {})
        .get("tokenizer_decoder_ctc_asr_translation_or_language_id_loaded")
        is not False
        or value.get("terminal_states")
        != [
            "PASS_DETECTABLE_STRUCTURE",
            "PASS_PRECISE_WEAK_OR_FLAT",
            "NO_GO_UNINFORMATIVE",
        ]
    ):
        raise BridgeV5Error("E_PROTOCOL")
    expected_parameters = trainable_parameter_count(1024, 384, 256, 128)
    if value["learner"]["trainable_parameter_count"] != expected_parameters:
        raise BridgeV5Error("E_PARAMETER_COUNT")
    return value


def freeze_and_stop() -> dict[str, Any]:
    config = _config()
    prior = _validate_prior_trees()
    positive = run_embedding_positive_control(
        seed=config["positive_control"]["seed"],
        participants=config["positive_control"]["participants"],
        windows_per_participant=config["positive_control"]["windows_per_participant"],
        confidence=config["inference"]["confidence"],
    )
    _write(POSITIVE_RECEIPT, positive)
    gates = {
        "governance": False,
        "support": False,
        "positive_control": bool(positive["pass"]),
        "preprocessing": False,
        "shortcut_interpretable": False,
        "precision": False,
        "heterogeneity": False,
        "detectable_structure": False,
        "precise_weak_or_flat": False,
    }
    decision = terminal_decision(gates)
    freeze = {
        "schema_version": "childlens-calibration-recovery-stage0-v5.0.0",
        "status": "TERMINAL_FAIL_CLOSED_BEFORE_DATA_SELECTION_FREEZE",
        "protocol_created_before_new_media_decoding_or_childlens_outcomes": True,
        "protocol_sha256": _sha256_file(CONFIG),
        "protocol_document_sha256": _sha256_file(PROTOCOL),
        "package_sha256": _sha256_file(PACKAGE),
        "runner_sha256": _sha256_file(Path(__file__)),
        "prior_v1_v2_v3_v4_immutable": True,
        "prior_tree_sha256": prior,
        "development_only_manifest_bindings_from_immutable_v4_record": (
            EXPECTED_DEVELOPMENT_MANIFEST_HASHES
        ),
        "public_model_bindings": {
            modality: {
                "repository": binding["repository"],
                "revision": binding["revision"],
                "weights_sha256": binding["weights_sha256"],
                "config_sha256": binding["config_sha256"],
                "preprocessor_config_sha256": binding[
                    "preprocessor_config_sha256"
                ],
            }
            for modality, binding in config["instruments"].items()
            if modality in {"audio", "vision"}
        },
        "public_model_files_downloaded_for_v5": False,
        "local_model_file_hash_verification_required_before_any_future_execution": True,
        "data_selection_frozen": False,
        "source_object_bindings_or_exact_intervals_exported": False,
        "stage0_governance_gate_pass": False,
        "zero_locked_access_certifiable": False,
        "mixed_scope_legacy_manifests_parsed_during_inventory": 2,
        "locked_row_values_identifiers_intervals_media_or_outcomes_exported": False,
        "new_media_decoded": False,
        "new_media_acquired": False,
        "childlens_embeddings_computed": False,
        "childlens_training_or_scoring_run": False,
        "locked_media_decoded_or_scored": False,
        "external_volume_aea_or_babyview_used": False,
        "temporary_credentials_accessed_or_created": False,
        "transient_full_source_objects_created": 0,
        "simulator_or_side_cue_training_run": False,
        "positive_control_pass": bool(positive["pass"]),
        "development_decision": decision,
        "stop_reason": "ZERO_LOCKED_ROW_LOADING_CANNOT_BE_CERTIFIED_AFTER_MIXED_SCOPE_MANIFEST_PARSE",
    }
    public_guard(freeze)
    _write(FREEZE_RECEIPT, freeze)
    report = {
        "schema_version": "childlens-calibration-recovery-decision-v5.0.0",
        "decision": decision,
        "status": "TERMINAL_DEVELOPMENT_STOP_AT_STAGE0",
        "gates": gates,
        "childlens_audiovisual_estimates_exist": False,
        "multi_hour_calibration_summary_exists": False,
        "interpretation": "The run is uninformative because governance failed before development selection, acquisition, or audiovisual estimation; it is not evidence that ChildLens lacks alignment.",
        "passing_result_would_eventually_permit": "After separate locked confirmation and separate simulator authorization, a distribution-matched simulation could compare otherwise identical learning with and without synchronized training-only physical side cues.",
        "passing_result_would_not_establish": [
            "infant calibration",
            "naturalistic German lexical grounding",
            "causal grounding in ChildLens",
            "real-world physical side-cue lift",
            "authorization to generate a simulator or train side-cue conditions"
        ],
    }
    public_guard(report)
    _write(DECISION_REPORT, report)
    DECISION_MARKDOWN.write_text(
        "\n".join(
            [
                "# ChildLens calibration recovery v5 — development decision",
                "",
                "**Decision: `NO_GO_UNINFORMATIVE`**",
                "",
                "V5 stopped at Stage 0. The inventory process parsed legacy "
                "mixed-scope manifests, so zero locked-row loading cannot be "
                "certified under the prospective governance rule. No identifiers, "
                "intervals, media, or outcomes were exported, but this is still a "
                "fail-closed governance failure.",
                "",
                "No new ChildLens media was acquired or decoded. No ChildLens "
                "embedding, learner training, scoring, lag curve, calibration "
                "summary, locked evaluation, simulator generation, or side-cue "
                "condition ran. The synthetic sensitivity check passed but is not "
                "empirical evidence.",
                "",
                "This decision does not imply that ChildLens is intrinsically "
                "uninformative. A clean rerun would require a separately generated "
                "outer-scope receipt and input containing exactly the immutable 18 "
                "development participants and zero locked rows before the v5 "
                "process starts.",
                "",
                "A future passing development result could only recommend separate "
                "locked confirmation. After that confirmation and separate "
                "authorization, Michael Frank’s bridge would support a "
                "distribution-matched simulation comparing identical learning "
                "with and without synchronized training-only physical side cues. "
                "It would still not establish infant calibration, naturalistic "
                "German lexical grounding, causal ChildLens grounding, or "
                "real-world side-cue lift.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return freeze


def validate(*, write: bool = True) -> dict[str, Any]:
    _config()
    prior = _validate_prior_trees()
    for path in (FREEZE_RECEIPT, POSITIVE_RECEIPT, DECISION_REPORT):
        public_guard(_read_json(path))
    freeze = _read_json(FREEZE_RECEIPT)
    decision = _read_json(DECISION_REPORT)
    positive = _read_json(POSITIVE_RECEIPT)
    if (
        freeze.get("protocol_sha256") != _sha256_file(CONFIG)
        or freeze.get("protocol_document_sha256") != _sha256_file(PROTOCOL)
        or freeze.get("package_sha256") != _sha256_file(PACKAGE)
        or freeze.get("runner_sha256") != _sha256_file(Path(__file__))
        or freeze.get("prior_tree_sha256") != prior
        or freeze.get("public_model_bindings")
        != {
            modality: {
                "repository": binding["repository"],
                "revision": binding["revision"],
                "weights_sha256": binding["weights_sha256"],
                "config_sha256": binding["config_sha256"],
                "preprocessor_config_sha256": binding[
                    "preprocessor_config_sha256"
                ],
            }
            for modality, binding in _config()["instruments"].items()
            if modality in {"audio", "vision"}
        }
        or freeze.get("zero_locked_access_certifiable") is not False
        or freeze.get("new_media_decoded") is not False
        or freeze.get("childlens_training_or_scoring_run") is not False
        or decision.get("decision") != "NO_GO_UNINFORMATIVE"
        or positive.get("pass") is not True
    ):
        raise BridgeV5Error("E_VALIDATION_BINDING")
    receipt = {
        "schema_version": "childlens-calibration-recovery-validation-v5.0.0",
        "status": "PASS_FOR_TERMINAL_STAGE0_RECORD",
        "protocol_sha256": _sha256_file(CONFIG),
        "protocol_document_sha256": _sha256_file(PROTOCOL),
        "package_sha256": _sha256_file(PACKAGE),
        "runner_sha256": _sha256_file(Path(__file__)),
        "prior_v1_v2_v3_v4_immutable": True,
        "prior_tree_sha256": prior,
        "public_privacy_guard_pass": True,
        "public_model_revision_and_upstream_hash_bindings_verified": True,
        "local_model_file_hash_verification_deferred_due_stage0_stop": True,
        "minimum_public_cell_participants": 5,
        "complementary_suppression_required": True,
        "restricted_identifiers_intervals_media_embeddings_scores_or_weights_exported": False,
        "zero_locked_access_certifiable": False,
        "new_media_acquired_or_decoded": False,
        "childlens_outcome_analysis_run": False,
        "external_volume_aea_or_babyview_used": False,
        "selective_transfer_cleanup_required": False,
        "temporary_credentials_created_or_accessed": False,
        "synthetic_positive_control_pass": True,
        "development_decision": "NO_GO_UNINFORMATIVE",
    }
    public_guard(receipt)
    if write:
        _write(VALIDATION_RECEIPT, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("freeze-and-stop", "validate"),
        help="Generate the terminal Stage 0 record or validate it without private access.",
    )
    args = parser.parse_args()
    result = freeze_and_stop() if args.command == "freeze-and-stop" else validate()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
