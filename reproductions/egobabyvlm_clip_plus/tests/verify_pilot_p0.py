#!/usr/bin/env python3
"""Verify the frozen, privacy-safe Pilot P0 contract without dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "spec.json"
CONFIG_PATH = ROOT / "configs" / "pilot.json"
DECISION_PATH = ROOT / "pilots" / "juno_sample" / "decision_record.json"
EXPECTED_COMMIT = "224621caf0628270b6115845ac75a65b984234a3"
EXPECTED_CLASSIFICATION = [
    "engineering_only",
    "non_comparable",
    "not_a_reproduction_result",
]
EXPECTED_SEED_TEXT = "egobabyvlm_clip_plus:juno_sample:P0:engineering_seed"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))

    require(spec["phase"]["number"] == 1, "scientific phase number changed")
    require(
        spec["phase"]["status"] == "blocked_by_exact_usable_subset_evidence",
        "scientific blocker was changed or overclaimed",
    )
    require(spec["scope"]["model_row"] == "BabyView CLIP+", "scientific target changed")
    require(spec["scope"]["benchmark"] == "Machine-DevBench", "benchmark changed")
    require(spec["scope"]["tasks"] == ["lexical", "grammatical"], "task scope changed")
    require(spec["scope"]["exact_replay_status"] == "not_claimed", "exact replay overclaimed")
    unresolved = spec["unresolved_reproduction_variables"]
    require(any(item["id"] == "U018" for item in unresolved), "exact-subset blocker disappeared")
    require(all(item["status"] == "unresolved" for item in unresolved), "scientific variable resolved")

    require(config["pilot_id"] == decision["pilot_id"] == "juno_sample", "pilot IDs disagree")
    require(config["engineering_run_id"] == decision["engineering_run_id"] == "p0-4cc3af23",
            "opaque engineering run IDs disagree")
    require(decision["status"] in {"p0_complete", "p1_complete", "p2_complete", "p3_complete", "p4_incomplete", "p4_complete"}, "P0 status is not preserved")
    require(config["pilot_stage"] in {"P0", "P1", "P2", "P3", "P4"}, "unexpected pilot stage")
    require(decision["p0_completion"]["stage"] == "P0", "P0 completion stage disagrees")
    require(decision["p0_completion"]["next_stage_started"] is False, "later pilot stage started")
    require(config["classification"] == decision["classification"] == EXPECTED_CLASSIFICATION,
            "classification labels disagree")
    require(config["scientific_status_effect"] == decision["scientific_status_effect"] == "none",
            "pilot changes scientific status")
    require((DECISION_PATH.parent / decision["canonical_config"]).resolve() == CONFIG_PATH,
            "decision record does not select the canonical config")

    fixed = decision["fixed_decisions"]
    inputs = config["inputs"]
    require(inputs["count"] == fixed["input_count"] == 2, "pilot must have exactly two inputs")
    require(len(inputs["execution_references"]) == 2, "pilot must have exactly two input slots")
    require(len({item["slot"] for item in inputs["execution_references"]}) == 2, "input slots duplicate")
    require(all(item["addressing"] == "opaque_governed_ledger_key_required_at_execution"
                for item in inputs["execution_references"]), "input is not ledger-key addressed")
    require(inputs["additional_downloads"] is fixed["additional_video_downloads_authorized"] is False,
            "additional downloads were authorized")

    seed = int.from_bytes(hashlib.sha256(EXPECTED_SEED_TEXT.encode()).digest()[:4], "big") % (2**31)
    require(seed == config["engineering_seed"]["value"] == fixed["engineering_seed"],
            "engineering seed is not deterministic or records disagree")
    require(config["engineering_seed"]["scientific_seed"] is False, "pilot seed labeled scientific")
    require(fixed["engineering_seed_is_scientific"] is False, "decision labels pilot seed scientific")
    require(fixed["training_seed_count"] == 1, "pilot must use one seed")

    require(config["tokenizer"]["type"] == fixed["tokenizer"]["type"] == "WordPiece",
            "tokenizer types disagree")
    require(config["tokenizer"]["vocabulary_size"] ==
            fixed["tokenizer"]["target_vocabulary_size"] == 2048, "vocabulary sizes disagree")
    require(config["model"]["vision"] == fixed["vision_architecture"] == "DINOv2_ViT-B/14",
            "vision architectures disagree")
    require(config["model"]["text"] == fixed["text_architecture"] == "BERT-base_12x768x12",
            "text architectures disagree")
    require(config["model"]["learned_component_initialization"] ==
            "random_for_every_learned_pilot_component", "pilot learned components are not all random")
    require(fixed["initialization"] == "random_for_all_pilot_learned_components",
            "decision initialization changed")
    require(config["preprocessing"] == fixed["preprocessing"], "preprocessing decisions disagree")
    require(config["split"]["preferred"] ==
            "recording_level_when_both_records_yield_usable_pairs", "recording split changed")
    require(config["split"]["fallback"] ==
            "explicitly_leaky_engineering_only_when_only_one_record_yields_usable_pairs",
            "engineering-only fallback changed")
    require(fixed["split"] ==
            "different_recordings_when_both_are_usable_otherwise_explicitly_leaky_engineering_only_split",
            "decision split changed")

    steps = config["clip_plus_cycle"]["steps"]
    require(steps == {"contrastive": 100, "mlm": 20, "dino_ibot": 10}, "wrong P0 step cycle")
    require(config["clip_plus_cycle"]["total_steps"] == sum(steps.values()) == 130,
            "cycle total is wrong")
    require(fixed["clip_plus_cycle"] == {"contrastive_steps": 100, "mlm_steps": 20, "dino_steps": 10},
            "decision step cycle disagrees")
    require(config["evaluation"]["pilot_model_evaluations_after_calibration"] == 1,
            "pilot checkpoint evaluation count changed")
    require(config["evaluation"]["calibration"] == "isolated_external_evaluator_calibration_only",
            "evaluator calibration is not isolated")
    require(fixed["evaluator_calibration"] ==
            "isolated_external_evaluator_calibration_only_then_one_frozen_pilot_checkpoint_evaluation",
            "evaluation decisions disagree")
    require(config["source"]["commit"] == spec["official_code"]["commit"] == EXPECTED_COMMIT,
            "upstream source pin disagrees")
    require(config["source"]["pilot_schedule_is_scientific_default"] is False,
            "pilot schedule misrepresented as scientific default")
    inspected = config["source"]["inspected_configuration"]
    require(len(inspected) == 6, "pinned upstream inspection record is incomplete")
    require(all(len(item["sha256"]) == 64 for item in inspected), "upstream config digest is malformed")

    storage = config["storage"]
    require(storage["resolver"] == "storage.py", "storage resolver changed")
    require(storage["durable"]["root_command"] == ["python3", "storage.py", "root", "durable"],
            "durable root bypasses storage.py")
    require(storage["scratch"]["root_command"] == ["python3", "storage.py", "root", "scratch"],
            "scratch root bypasses storage.py")
    require(storage["local_or_worktree_fallback"] is False, "local storage fallback enabled")
    require(config["full_corpus_discrepancy"] == "unresolved_and_out_of_scope",
            "full-corpus discrepancy overclaimed")
    require(config["gpu_memory_policy"]["later_stage_continuation_after_failure"] is False,
            "GPU failure does not stop later stages")
    require(fixed["gpu_memory_policy"] ==
            "use_the_smallest_implementation_valid_microbatch_without_changing_architecture_then_stop_and_review_if_it_does_not_fit",
            "GPU memory decisions disagree")

    retained = config["expected_retained_p0_record"]
    require((ROOT / retained["location"]).resolve() == DECISION_PATH, "retained P0 record moved")
    require("data_identifiers" in retained["forbidden_content"], "privacy exclusion missing")
    digest = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    print(f"Pilot P0 verification passed (config sha256={digest})")


if __name__ == "__main__":
    main()
