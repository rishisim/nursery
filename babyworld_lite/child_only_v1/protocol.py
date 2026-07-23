"""Frozen construction contracts for causal arms and evaluation endpoints."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from .policy import CONSTRUCTION_PROFILE, PolicyViolation, canonical_digest


CONDITION_ORDER = (
    "strong_alignment_ceiling",
    "weak_vl_baseline",
    "weak_synchronized_side",
    "weak_episode_shuffled_side",
    "weak_time_shifted_side",
)
WEAK_CONDITIONS = CONDITION_ORDER[1:]


def load_and_validate_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    expected = {
        "protocol_version",
        "contract_version",
        "status",
        "scientific_outcome_authorized",
        "primary_question",
        "learner",
        "side_bundle",
        "conditions",
        "weak_arm_equality_invariants",
        "individual_cue_ablation_branch",
        "adaptation_policy",
    }
    if set(protocol) != expected:
        raise PolicyViolation("causal protocol has unknown or missing fields")
    if protocol["protocol_version"] != "child-only-causal-protocol-v1.0.0":
        raise PolicyViolation("wrong causal protocol version")
    if protocol["status"] != "CONSTRUCTION_ONLY_NOT_FROZEN_FOR_OUTCOME" or protocol["scientific_outcome_authorized"] is not False:
        raise PolicyViolation("construction protocol cannot authorize an outcome run")
    if protocol["learner"]["architecture"] != "TEMPORAL_CLIP_PLUS_SCRATCH":
        raise PolicyViolation("wrong primary learner")
    conditions = protocol["conditions"]
    if list(conditions) != list(CONDITION_ORDER):
        raise PolicyViolation("conditions must be complete and canonically ordered")
    if conditions["strong_alignment_ceiling"]["language_alignment"] != "STRONG":
        raise PolicyViolation("strong ceiling must use strong language-event alignment")
    for name in WEAK_CONDITIONS:
        if conditions[name]["language_alignment"] != "WEAK":
            raise PolicyViolation("all weak causal arms must share weak RGB/text experience")
        if conditions[name]["evaluation_inputs"] != ["VISION", "TEXT"]:
            raise PolicyViolation("every primary evaluation must withhold side channels")
    return protocol


def validate_matched_condition_bundle(bundle: Mapping[str, Any]) -> None:
    """Fail closed on incomplete, unmatched, or malformed coupled arm bundles."""

    expected = {
        "bundle_version",
        "profile_label",
        "bundle_id",
        "corpus_instance_id",
        "corpus_seed",
        "model_seed",
        "weak_common",
        "episode_inventory",
        "base_side_sequence_digest_by_episode",
        "base_side_sample_multiset_digest_by_episode",
        "arms",
    }
    if set(bundle) != expected:
        raise PolicyViolation("condition bundle has unknown or missing fields")
    if bundle["bundle_version"] != "child-only-matched-bundle-v1":
        raise PolicyViolation("wrong matched-bundle version")
    if bundle["profile_label"] != CONSTRUCTION_PROFILE:
        raise PolicyViolation("construction bundle must retain the fixture label")
    arms = bundle["arms"]
    if list(arms) != list(CONDITION_ORDER):
        raise PolicyViolation("a bundle must contain all five coupled conditions in canonical order")

    common_expected = {
        "rgb_text_experience_digest",
        "architecture_digest",
        "initialization_receipt_id",
        "tokenizer_artifact_id",
        "model_config_digest",
        "optimizer_digest",
        "example_order_digest",
        "update_count",
        "batch_shape_digest",
        "stopping_rule_digest",
        "compute_budget_digest",
        "side_architecture_present",
    }
    common = bundle["weak_common"]
    if set(common) != common_expected or common["side_architecture_present"] is not True:
        raise PolicyViolation("weak arms must share one complete architecture and training contract")
    if not isinstance(common["update_count"], int) or common["update_count"] <= 0:
        raise PolicyViolation("weak-arm update count must be a positive frozen integer")

    inventory = bundle["episode_inventory"]
    base_digests = bundle["base_side_sequence_digest_by_episode"]
    base_multisets = bundle["base_side_sample_multiset_digest_by_episode"]
    if not isinstance(inventory, Mapping) or len(inventory) < 2:
        raise PolicyViolation("matched controls need at least two episodes")
    ids = set(inventory)
    if set(base_digests) != ids or set(base_multisets) != ids:
        raise PolicyViolation("base side manifests must exactly cover the episode inventory")
    for metadata in inventory.values():
        if set(metadata) != {"split", "side_shape", "side_validity_digest"}:
            raise PolicyViolation("episode control metadata is incomplete")

    arm_expected = {
        "condition",
        "language_alignment",
        "side_transform",
        "source_episode_by_episode",
        "side_sequence_digest_by_episode",
        "side_sample_multiset_digest_by_episode",
        "time_shift_samples_by_episode",
        "final_status",
    }
    for name, arm in arms.items():
        if set(arm) != arm_expected or arm["condition"] != name or arm["final_status"] != "CONSTRUCTION_VALIDATED":
            raise PolicyViolation("condition record is incomplete or nonfinal")
        if set(arm["source_episode_by_episode"]) != ids:
            raise PolicyViolation("condition source map must exactly cover all episodes")
        if set(arm["side_sequence_digest_by_episode"]) != ids or set(arm["side_sample_multiset_digest_by_episode"]) != ids:
            raise PolicyViolation("condition side manifests must exactly cover all episodes")
        if set(arm["time_shift_samples_by_episode"]) != ids:
            raise PolicyViolation("condition time-shift map must exactly cover all episodes")

    strong = arms["strong_alignment_ceiling"]
    if strong["language_alignment"] != "STRONG" or strong["side_transform"] != "SYNCHRONIZED":
        raise PolicyViolation("strong alignment ceiling is malformed")
    for name in WEAK_CONDITIONS:
        if arms[name]["language_alignment"] != "WEAK":
            raise PolicyViolation("weak condition has different language experience")

    sync = arms["weak_synchronized_side"]
    for episode_id in ids:
        if sync["source_episode_by_episode"][episode_id] != episode_id:
            raise PolicyViolation("synchronized control must use the same episode")
        if sync["side_sequence_digest_by_episode"][episode_id] != base_digests[episode_id]:
            raise PolicyViolation("synchronized side digest differs from base episode")
        if sync["side_sample_multiset_digest_by_episode"][episode_id] != base_multisets[episode_id]:
            raise PolicyViolation("synchronized side sample inventory differs from base episode")
        if sync["time_shift_samples_by_episode"][episode_id] != 0:
            raise PolicyViolation("synchronized side must not be shifted")

    baseline = arms["weak_vl_baseline"]
    if baseline["side_transform"] != "SHAPE_MATCHED_NULL":
        raise PolicyViolation("weak VL baseline must use a shape-matched null side stream")
    if any(source is not None for source in baseline["source_episode_by_episode"].values()):
        raise PolicyViolation("weak VL baseline cannot have a side donor")

    shuffled = arms["weak_episode_shuffled_side"]
    if shuffled["side_transform"] != "WHOLE_BUNDLE_EPISODE_SHUFFLE":
        raise PolicyViolation("episode-shuffled arm transform is wrong")
    donors = shuffled["source_episode_by_episode"]
    for target, donor in donors.items():
        if donor not in ids or donor == target:
            raise PolicyViolation("shuffled side mapping must be a no-self derangement")
        if inventory[target]["split"] != inventory[donor]["split"]:
            raise PolicyViolation("shuffled donors must stay in split")
        if inventory[target]["side_shape"] != inventory[donor]["side_shape"]:
            raise PolicyViolation("shuffled donors must preserve side tensor shape")
        if inventory[target]["side_validity_digest"] != inventory[donor]["side_validity_digest"]:
            raise PolicyViolation("shuffled donors must preserve missingness/validity pattern")
        if shuffled["side_sequence_digest_by_episode"][target] != base_digests[donor]:
            raise PolicyViolation("shuffled sequence digest does not match its declared donor")
        if shuffled["side_sample_multiset_digest_by_episode"][target] != base_multisets[donor]:
            raise PolicyViolation("shuffled side sample multiset does not match its donor")
    if set(donors.values()) != ids:
        raise PolicyViolation("shuffled donors must be a whole-episode permutation")

    shifted = arms["weak_time_shifted_side"]
    if shifted["side_transform"] != "WITHIN_EPISODE_CIRCULAR_NONZERO_SHIFT":
        raise PolicyViolation("time-shifted arm transform is wrong")
    for episode_id in ids:
        if shifted["source_episode_by_episode"][episode_id] != episode_id:
            raise PolicyViolation("time-shifted side must remain within episode")
        shift = shifted["time_shift_samples_by_episode"][episode_id]
        sequence_length = inventory[episode_id]["side_shape"][0]
        if not isinstance(shift, int) or shift == 0 or shift % sequence_length == 0:
            raise PolicyViolation("time shift must be nonzero modulo the episode sequence length")
        if shifted["side_sample_multiset_digest_by_episode"][episode_id] != base_multisets[episode_id]:
            raise PolicyViolation("time shift must preserve the exact side sample multiset")


def condition_bundle_id(protocol_digest: str, corpus_instance_id: str, corpus_seed: int, model_seed: int) -> str:
    return canonical_digest(
        {
            "protocol_digest": protocol_digest,
            "corpus_instance_id": corpus_instance_id,
            "corpus_seed": corpus_seed,
            "model_seed": model_seed,
        }
    )


def load_and_validate_evaluation_spec(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    expected_tasks = {
        "held_out_noun_object_grounding",
        "held_out_verb_action_grounding",
        "novel_compositions",
        "cross_world_render_seed_generalization",
    }
    if set(spec["tasks"]) != expected_tasks:
        raise PolicyViolation("evaluation spec lacks a required corpus-grounded task")
    if spec["model_call_inputs"] != ["VISION", "TEXT"]:
        raise PolicyViolation("primary evaluation spec must withhold all side and metadata inputs")
    if spec["machine_devbench"]["role"] != "SECONDARY_COVERAGE_GATED":
        raise PolicyViolation("Machine-DevBench must remain coverage-gated and secondary")
    if spec["outcome_execution_authorized"] is not False:
        raise PolicyViolation("construction evaluation spec cannot authorize outcome execution")
    return spec
