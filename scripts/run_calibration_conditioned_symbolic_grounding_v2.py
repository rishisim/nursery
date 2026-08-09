#!/usr/bin/env python3
"""One-shot-capable calibration-conditioned symbolic grounding runner v2.

The v1 fixture runner remains immutable.  This version adds an explicit frozen
execution contract, deterministic range draws, complete paired corpus bundles,
atomic checkpoint/resume, deterministic merge, and a hash-bound one-shot
authorization seal.  Scientific execution is possible only with a passed
non-fixture calibration receipt, passed pre-outcome receipt, and an owner-private
authorization seal binding every input.  This module never creates that seal.

Only ``benchmark-smoke`` may execute without a scientific authorization; it is
hard-limited to tiny synthetic fixtures and never uses empirical calibration.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
V1_SCRIPT = ROOT / "scripts/run_calibration_conditioned_symbolic_grounding.py"
DEFAULT_PROTOCOL = ROOT / "docs/nursery_program_convergence_v1/frozen_calibration_conditioned_sensitivity_protocol.json"
DEFAULT_CONTRACT = ROOT / "docs/nursery_program_convergence_v1/calibration_conditioned_symbolic_execution_contract_v2.json"
DEFAULT_CALIBRATION = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_receipt.json"
DEFAULT_FIXTURE_RANGES = ROOT / "tests/fixtures/calibration_conditioned_symbolic_ranges_v1.json"

VERSION = "nursery-calibration-conditioned-symbolic-runner-v2"
CONTRACT_SCHEMA = "nursery-calibration-conditioned-symbolic-execution-contract-v2"
PREOUTCOME_SCHEMA = "nursery-calibration-conditioned-symbolic-preoutcome-v2"
AUTHORIZATION_SCHEMA = "nursery-calibration-conditioned-symbolic-authorization-v2"
ERRATUM_SCHEMA = "nursery-calibration-conditioned-symbolic-protocol-erratum-v2"
BUNDLE_SCHEMA = "nursery-calibration-conditioned-symbolic-bundle-v2"
MERGE_SCHEMA = "nursery-calibration-conditioned-symbolic-merge-v2"
CONSUMPTION_SCHEMA = "nursery-calibration-conditioned-symbolic-consumption-v2"
PROTOTYPE_CALIBRATION_SCHEMA = "nursery-prototype-opaque-gemma-calibration-receipt-v1"
CAF_TRANSPORT_AMENDMENT_SHA256 = "3ecf4bf6c62920d73177d029c18918f38366d34f7d7997642037696def37bbd0"
NO_FALLBACK_OVERRIDE_SHA256 = "cb9e7061a5725050e6a71a7b6e41f6f5e7bf30d104bc46bae233816ca99a90b7"
AUTH_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{15,79}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
HISTORICAL_RANGE_FAMILIES = (
    "existing_whisper_qwen2_path",
    "challenger_qwen3asr_qwen3vl_path",
    "model_intersection",
    "model_union",
    "conservative_envelope",
)
REPLACEMENT_RANGE_FAMILIES = (
    "qwen3asr_plus_qwen3vl_path",
    "qwen3asr_plus_gemma4_path",
    "model_intersection",
    "model_union",
    "conservative_envelope",
)
SENSITIVE_CALIBRATION_KEYS = frozenset(
    {
        "transcript_text",
        "translated_text",
        "source_filename",
        "source_path",
        "participant_id",
        "item_id",
        "exact_timestamp",
        "exact_interval",
        "frame_content",
        "audio_content",
        "item_level_prediction",
        "confidence_score",
        "raw_model_payload",
        "item_rows",
        "human_validation",
        "inter_human_reliability",
        "ground_truth",
        "transcript_correctness",
        "referential_truth",
    }
)
PUBLIC_FALSE_FIELDS = (
    "raw_counts",
    "paths",
    "identifiers",
    "filenames",
    "exact_timestamps_or_intervals",
    "transcript_or_lexical_content",
    "frames_or_audio",
    "item_level_predictions",
    "confidence_or_raw_model_payload",
    "free_form_errors",
)


def _load_v1() -> Any:
    specification = importlib.util.spec_from_file_location(
        "calibration_conditioned_symbolic_v1_for_v2", V1_SCRIPT
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("E_V1_IMPORT")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


v1 = _load_v1()


class OneShotError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise OneShotError(code)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        _fail("E_CANONICAL_JSON")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                result.update(block)
    except OSError:
        _fail("E_FILE_HASH")
    return result.hexdigest()


def _read_json(path: Path, maximum: int = 64 * 1024 * 1024) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
        if not payload or len(payload) > maximum:
            _fail("E_JSON")
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("E_JSON")
    if not isinstance(value, Mapping):
        _fail("E_JSON")
    return value


def _public_payload_safe(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in SENSITIVE_CALIBRATION_KEYS and child not in (
                False,
                None,
                "",
                [],
                {},
            ):
                return False
            if not _public_payload_safe(child):
                return False
        return True
    if isinstance(value, list):
        return all(_public_payload_safe(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


def _public_calibration_receipt_safe(
    document: Mapping[str, Any], binding: Any
) -> bool:
    public = document.get("public_export")
    security = document.get("security")
    ancestry = document.get("ancestry")
    return (
        _public_payload_safe(document)
        and document.get("schema_version") == PROTOTYPE_CALIBRATION_SCHEMA
        and document.get("status") == "CALIBRATION_PASS"
        and document.get("caf_transport_amendment_sha256") == CAF_TRANSPORT_AMENDMENT_SHA256
        and document.get("no_automatic_fallback_override_sha256") == NO_FALLBACK_OVERRIDE_SHA256
        and document.get("automatic_fallback_allowed") is False
        and document.get("scientific_endpoint_if_gemma_fails") is False
        and tuple(document.get("instrument_paths", ()))
        == REPLACEMENT_RANGE_FAMILIES[:2]
        and tuple(family.name for family in binding.families)
        == tuple(sorted(REPLACEMENT_RANGE_FAMILIES))
        and isinstance(public, Mapping)
        and public.get("minimum_cluster_k") == 5
        and public.get("complementary_suppression") is True
        and all(public.get(key) is False for key in PUBLIC_FALSE_FIELDS)
        and isinstance(security, Mapping)
        and security.get("restricted_inference_network_disabled") is True
        and security.get("external_api_used") is False
        and security.get("hosted_or_cloud_content_path") is False
        and security.get("quarantine_only_pseudo_payloads") is True
        and isinstance(ancestry, Mapping)
        and ancestry.get("AEA_empirical_ancestry") is False
        and ancestry.get("BabyView_empirical_ancestry") is False
        and ancestry.get("cross_corpus_pooling") is False
        and ancestry.get("learner_ancestry_from_instruments") is False
        and document.get("pseudo_labels_are_ground_truth") is False
        and document.get("human_evidence_available") is False
        and document.get("simulator_oracle_only_evaluation_truth") is True
        and document.get("scientific_outcome_run") is False
    )


def bind_calibration_ranges_v2(
    document: Mapping[str, Any], *, fixture_only: bool
) -> Any:
    """Bind the additive prototype schema without weakening the immutable v1 parser."""

    if fixture_only:
        return v1.bind_calibration_ranges(document, fixture_only=True)
    if document.get("schema_version") != PROTOTYPE_CALIBRATION_SCHEMA:
        _fail("E_CALIBRATION_SCHEMA")
    projection = dict(document)
    projection["schema_version"] = v1.PUBLIC_CALIBRATION_SCHEMA
    parsed = v1.bind_calibration_ranges(projection, fixture_only=False)
    return v1.CalibrationBinding(
        schema_version=PROTOTYPE_CALIBRATION_SCHEMA,
        status=parsed.status,
        digest=_digest(document),
        fixture_only=False,
        gate_passed=parsed.gate_passed,
        families=parsed.families,
        suppressed_dimensions=parsed.suppressed_dimensions,
    )


def _private_regular(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and stat.S_IMODE(metadata.st_mode) & 0o077 == 0
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except Exception:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create a private registry record without permitting a competing writer."""

    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError:
        _fail("E_ONE_SHOT_REGISTRY")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        with contextlib.suppress(OSError):
            path.unlink()
        raise


@dataclass(frozen=True)
class ExecutionContract:
    digest: str
    erratum_digest: str
    corpus_seeds: tuple[int, ...]
    model_seeds: tuple[int, ...]
    range_families: tuple[str, ...]
    range_draws: int
    arms: tuple[str, ...]
    inference_seed: int
    concept_count: int
    repetitions: int
    thresholds: Mapping[str, float]
    planned_command: str


def load_execution_contract(
    contract_path: Path, protocol_path: Path
) -> ExecutionContract:
    contract = _read_json(contract_path)
    protocol = _read_json(protocol_path)
    erratum_relative = contract.get("protocol_erratum")
    if not isinstance(erratum_relative, str):
        _fail("E_PROTOCOL_ERRATUM")
    erratum_path = ROOT / erratum_relative
    erratum = _read_json(erratum_path)
    correction = erratum.get("corrections")
    interpretation = erratum.get("authoritative_interpretation")
    if (
        contract.get("protocol_erratum_required") is not True
        or erratum.get("schema_version") != ERRATUM_SCHEMA
        or erratum.get("status") != "FROZEN_PRE_OUTCOME_EXECUTION_ERRATUM"
        or erratum.get("effect_or_endpoint_opened_before_erratum") is not False
        or erratum.get("families_added_or_removed") is not False
        or erratum.get("ranges_changed") is not False
        or erratum.get("thresholds_changed") is not False
        or not isinstance(interpretation, Mapping)
        or interpretation.get("range_family_count") != 5
        or tuple(interpretation.get("replacement_range_family_names", ()))
        != REPLACEMENT_RANGE_FAMILIES
        or interpretation.get("replacement_source")
        != "docs/nursery_program_convergence_v1/gemma_substitution_integration_note.md"
        or not isinstance(correction, list)
        or correction
        != [
            {
                "field": "seed_plan.paired_bundle",
                "original_wording": "all seven range families",
                "corrected_wording": "all five range families",
            },
            {
                "corrected_names": list(REPLACEMENT_RANGE_FAMILIES),
                "field": "naturalistic_calibration_layer.range_families",
                "historical_names": list(HISTORICAL_RANGE_FAMILIES),
                "scope": "PREDECLARED_ONE_INSTRUMENT_SUBSTITUTION_LABELS_ONLY",
            },
        ]
    ):
        _fail("E_PROTOCOL_ERRATUM")
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or contract.get("status") != "FROZEN_IMPLEMENTED_NOT_AUTHORIZED"
        or protocol.get("schema_version") != v1.PROTOCOL_SCHEMA
        or tuple(contract.get("arms", ())) != v1.ARMS
        or contract.get("bundle_unit")
        != "ONE_CORPUS_SEED_ALL_RANGE_FAMILIES_ALL_DRAWS_ALL_MODEL_SEEDS_ALL_ARMS"
    ):
        _fail("E_CONTRACT")
    expected_corpus = tuple(range(740001, 740080, 2))
    corpus = tuple(contract.get("corpus_seeds", ()))
    model = tuple(contract.get("model_seeds", ()))
    range_families = tuple(contract.get("range_families", ()))
    protocol_seed = protocol.get("seed_plan", {})
    protocol_ranges = protocol.get("naturalistic_calibration_layer", {}).get(
        "range_families"
    )
    if (
        corpus != expected_corpus
        or model != tuple(protocol_seed.get("model_seeds", ()))
        or tuple(protocol_ranges or ()) != HISTORICAL_RANGE_FAMILIES
        or range_families != REPLACEMENT_RANGE_FAMILIES
        or contract.get("inference_seed") != protocol_seed.get("inference_seed")
        or contract.get("range_draws_per_family") != 7
    ):
        _fail("E_CONTRACT_SEEDS_OR_RANGES")
    thresholds = contract.get("thresholds")
    protocol_inference = protocol.get("inference")
    if not isinstance(thresholds, Mapping) or not isinstance(protocol_inference, Mapping):
        _fail("E_CONTRACT_THRESHOLDS")
    for key in (
        "minimum_mean_effect",
        "lower_bound_must_exceed",
        "minimum_positive_corpus_fraction",
        "maximum_large_negative_fraction",
        "large_negative_threshold",
    ):
        if thresholds.get(key) != protocol_inference.get(key):
            _fail("E_CONTRACT_THRESHOLDS")
    evaluation = contract.get("evaluation")
    falsifications = contract.get("falsifications")
    checkpoint = contract.get("checkpoint_policy")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("cue_fields_allowed") is not False
        or evaluation.get("primary_truth") != "SIMULATOR_ORACLE_ONLY"
        or not isinstance(falsifications, Mapping)
        or falsifications.get("side_only_maximum_above_chance") != 0.02
        or falsifications.get("confidence_only_rule")
        != "BOTH_NOUN_AND_VERB_WRONG_OR_TIED_TOP_MAPPINGS_MUST_BECOME_UNIQUELY_CORRECT"
        or falsifications.get("event_selection_disconnection_required") is not True
        or falsifications.get("scientific_endpoints_are_top_mapping_metrics_only")
        is not True
        or not isinstance(checkpoint, Mapping)
        or any(
            checkpoint.get(key) is not True
            for key in (
                "atomic_complete_bundle_only",
                "resume_incomplete_authorization_only",
            )
        )
        or checkpoint.get("deterministic_merge_order") != "ASCENDING_CORPUS_SEED"
        or checkpoint.get("reuse_complete_authorization") is not False
    ):
        _fail("E_CONTRACT_EXECUTION")
    planned = str(contract.get("planned_outcome_command"))
    if (
        "--authorization-seal" not in planned
        or "--preoutcome-receipt" not in planned
        or str(DEFAULT_CALIBRATION.relative_to(ROOT)) not in planned
        or "--calibration-receipt output/nursery_program_convergence_v1/childlens_pseudo_calibration_receipt.json"
        in planned
    ):
        _fail("E_CONTRACT_COMMAND")
    return ExecutionContract(
        digest=_digest(contract),
        erratum_digest=_digest(erratum),
        corpus_seeds=corpus,
        model_seeds=model,
        range_families=range_families,
        range_draws=7,
        arms=v1.ARMS,
        inference_seed=int(contract["inference_seed"]),
        concept_count=int(evaluation["concept_count_per_family"]),
        repetitions=int(evaluation["training_repetitions_per_object_action_composition"]),
        thresholds={key: float(value) for key, value in thresholds.items()},
        planned_command=planned,
    )


def _lhs_dimensions(family: Any) -> list[tuple[str, tuple[float, float]]]:
    dimensions = [(key, family.scalar[key]) for key in sorted(family.scalar)]
    for category in sorted(family.categories):
        for label in sorted(family.categories[category]):
            dimensions.append(
                (f"{category}:{label}", family.categories[category][label])
            )
    return dimensions


def _minimum_distance(matrix: Sequence[Sequence[float]]) -> float:
    result = float("inf")
    for left_index, left in enumerate(matrix):
        for right in matrix[left_index + 1 :]:
            result = min(
                result,
                math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right))),
            )
    return result


def deterministic_range_points(
    family: Any, *, draws: int, inference_seed: int
) -> tuple[Mapping[str, Any], ...]:
    """Select a deterministic maximin Latin-hypercube from 256 candidates."""

    if draws != 7:
        _fail("E_RANGE_DRAWS")
    dimensions = _lhs_dimensions(family)
    seed_material = hashlib.sha256(
        f"{inference_seed}\0{family.name}\0v2-maximin-lhs".encode()
    ).digest()
    generator = random.Random(int.from_bytes(seed_material[:8], "big"))
    best: list[list[float]] | None = None
    best_score = -1.0
    best_digest = ""
    for _ in range(256):
        columns: list[list[float]] = []
        for _dimension in dimensions:
            permutation = list(range(draws))
            generator.shuffle(permutation)
            columns.append([(rank + 0.5) / draws for rank in permutation])
        matrix = [
            [columns[column][row] for column in range(len(columns))]
            for row in range(draws)
        ]
        score = _minimum_distance(matrix)
        digest = _digest(matrix)
        if score > best_score or (score == best_score and digest < best_digest):
            best, best_score, best_digest = matrix, score, digest
    assert best is not None
    result: list[Mapping[str, Any]] = []
    for draw_index, unit_values in enumerate(best):
        scalar: dict[str, float] = {}
        categories: dict[str, dict[str, float]] = {
            key: {} for key in family.categories
        }
        for (name, (lower, upper)), unit in zip(dimensions, unit_values):
            value = lower + unit * (upper - lower)
            if ":" in name:
                category, label = name.split(":", 1)
                categories[category][label] = value
            else:
                scalar[name] = value
        normalized: dict[str, dict[str, float]] = {}
        for category, values in categories.items():
            total = sum(values.values())
            normalized[category] = (
                {key: value / total for key, value in values.items()}
                if total > 0
                else {key: 1.0 / len(values) for key in values}
            )
        result.append(
            {
                **scalar,
                **normalized,
                "draw_index": draw_index,
                "range_point_policy": "DETERMINISTIC_MAXIMIN_LATIN_HYPERCUBE",
            }
        )
    return tuple(result)


def _dynamic_strong_ceiling(concepts: int, model_seed: int) -> Mapping[str, float]:
    rows = [
        {
            "episode_id": f"C{index}",
            "noun_token": f"N_{index}",
            "verb_token": f"V_{index}",
            "noun_options": [f"OBJ_{index}"],
            "verb_options": [f"ACT_{index}"],
        }
        for index in range(concepts)
    ]
    model = v1.fit_dual_corrective(rows, model_seed=model_seed)
    return v1.evaluate_without_side(model, v1.evaluation_items(concepts))


PREOUTCOME_GATES = (
    "calibration_ranges_complete",
    "calibration_receipt_public_privacy_policy",
    "range_draws_deterministic_and_in_bounds",
    "exact_five_matched_arms",
    "matched_primary_digest",
    "matched_side_marginals",
    "matched_update_counts",
    "cue_free_evaluator_rejects_side",
    "model_serializes_no_side_state",
    "strong_alignment_ceiling_noun",
    "strong_alignment_ceiling_verb",
    "side_only_noun_at_most_chance_plus_002",
    "side_only_verb_at_most_chance_plus_002",
    "confidence_only_noun_wrong_or_tied_mapping_corrected",
    "confidence_only_verb_wrong_or_tied_mapping_corrected",
    "event_selection_disconnection_removes_correction",
    "no_AEA_ancestry",
    "no_BabyView_ancestry",
    "simulator_oracle_only_evaluation",
)


def build_preoutcome_receipt(
    binding: Any,
    contract: ExecutionContract,
    *,
    fixture_only: bool,
    calibration_document: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    families = {family.name: family for family in binding.families}
    required = tuple(families) if fixture_only else contract.range_families
    ranges_complete = binding.gate_passed and set(required).issubset(families)
    draws_valid = ranges_complete
    for family_name in required:
        if family_name not in families:
            draws_valid = False
            continue
        first = deterministic_range_points(
            families[family_name],
            draws=contract.range_draws,
            inference_seed=contract.inference_seed,
        )
        second = deterministic_range_points(
            families[family_name],
            draws=contract.range_draws,
            inference_seed=contract.inference_seed,
        )
        draws_valid = draws_valid and _digest(first) == _digest(second) and len(first) == 7
    representative = next(iter(families.values()))
    parameters = deterministic_range_points(
        representative, draws=7, inference_seed=contract.inference_seed
    )[0]
    # Repetitions are a multiple of every possible candidate count so the
    # side-only position signature is balanced across concepts.
    episodes = v1.generate_fixture_episodes(
        parameters, 930001, concepts=contract.concept_count, repetitions=24
    )
    views = v1.condition_views(episodes)
    primary = {arm: v1.primary_digest(rows) for arm, rows in views.items()}
    side = {
        arm: v1.side_multiset_digest(rows)
        for arm, rows in views.items()
        if arm != "weak_vl_absent_side"
    }
    update_counts = {
        v1.fit_dual_corrective(rows, model_seed=930101)["candidate_update_count"]
        for rows in views.values()
    }
    chance = 1.0 / contract.concept_count
    microcase = v1.corrective_microcase()
    ceiling = _dynamic_strong_ceiling(contract.concept_count, 930101)
    cue_rejected = False
    model = v1.fit_dual_corrective(
        views["weak_vl_synchronized_side"], model_seed=930101
    )
    invalid_eval = [dict(row) for row in v1.evaluation_items(contract.concept_count)]
    invalid_eval[0]["side_cue"] = [1.0]
    try:
        v1.evaluate_without_side(model, invalid_eval)
    except v1.SymbolicRunnerError:
        cue_rejected = True
    gates = {
        "calibration_ranges_complete": ranges_complete,
        "calibration_receipt_public_privacy_policy": fixture_only
        or (
            calibration_document is not None
            and _public_calibration_receipt_safe(calibration_document, binding)
        ),
        "range_draws_deterministic_and_in_bounds": draws_valid,
        "exact_five_matched_arms": tuple(views) == contract.arms,
        "matched_primary_digest": len(set(primary.values())) == 1,
        "matched_side_marginals": len(set(side.values())) == 1,
        "matched_update_counts": len(update_counts) == 1,
        "cue_free_evaluator_rejects_side": cue_rejected,
        "model_serializes_no_side_state": model["side_state_serialized"] is False,
        "strong_alignment_ceiling_noun": ceiling["cue_free_heldout_noun_object_macro_top1"] == 1.0,
        "strong_alignment_ceiling_verb": ceiling["cue_free_heldout_verb_action_macro_top1"] == 1.0,
        "side_only_noun_at_most_chance_plus_002": v1._side_only_accuracy(episodes, "noun") <= chance + 0.02,
        "side_only_verb_at_most_chance_plus_002": v1._side_only_accuracy(episodes, "verb") <= chance + 0.02,
        "confidence_only_noun_wrong_or_tied_mapping_corrected": microcase["noun"]["wrong_or_tied_to_correct"],
        "confidence_only_verb_wrong_or_tied_mapping_corrected": microcase["verb"]["wrong_or_tied_to_correct"],
        "event_selection_disconnection_removes_correction": microcase["noun"]["disconnected_equals_baseline_top"]
        and microcase["verb"]["disconnected_equals_baseline_top"],
        "no_AEA_ancestry": True,
        "no_BabyView_ancestry": True,
        "simulator_oracle_only_evaluation": True,
    }
    if tuple(gates) != PREOUTCOME_GATES:
        _fail("E_PREOUTCOME_GATE_SCHEMA")
    passed = all(gates.values())
    return {
        "schema_version": PREOUTCOME_SCHEMA,
        "status": "FIXTURE_PASS" if fixture_only and passed else "PASS" if passed else "FAIL",
        "fixture_only": fixture_only,
        "scientific_endpoints_opened": False,
        "calibration_binding_sha256": binding.digest,
        "execution_contract_sha256": contract.digest,
        "protocol_erratum_sha256": contract.erratum_digest,
        "runner_sha256": _sha256_file(SCRIPT),
        "gates": gates,
        "all_gates_passed": passed,
    }


AUTHORIZATION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "authorization_id",
        "outcome_scope",
        "one_shot",
        "fixture_only",
        "authorized_by_user",
        "protocol_sha256",
        "execution_contract_sha256",
        "protocol_erratum_sha256",
        "calibration_receipt_sha256",
        "calibration_binding_sha256",
        "preoutcome_receipt_sha256",
        "runner_sha256",
    }
)


def validate_authorization_seal(
    seal_path: Path,
    *,
    protocol_path: Path,
    contract_path: Path,
    calibration_path: Path,
    calibration_document: Mapping[str, Any] | None,
    binding: Any,
    preoutcome_path: Path,
    contract: ExecutionContract,
) -> Mapping[str, Any]:
    if not _private_regular(seal_path):
        _fail("E_AUTHORIZATION_PRIVATE")
    if calibration_document is None or not _public_calibration_receipt_safe(
        calibration_document, binding
    ):
        _fail("E_CALIBRATION_PUBLIC_POLICY")
    seal = _read_json(seal_path)
    preoutcome = _read_json(preoutcome_path)
    if frozenset(seal) != AUTHORIZATION_KEYS:
        _fail("E_AUTHORIZATION_SCHEMA")
    authorization_id = seal.get("authorization_id")
    if (
        seal.get("schema_version") != AUTHORIZATION_SCHEMA
        or seal.get("status") != "AUTHORIZED"
        or not isinstance(authorization_id, str)
        or AUTH_ID_RE.fullmatch(authorization_id) is None
        or seal.get("outcome_scope")
        != "ONE_SHOT_CALIBRATION_CONDITIONED_SYMBOLIC_GROUNDING_V2"
        or seal.get("one_shot") is not True
        or seal.get("fixture_only") is not False
        or seal.get("authorized_by_user") is not True
        or seal.get("protocol_sha256") != _sha256_file(protocol_path)
        or seal.get("execution_contract_sha256") != contract.digest
        or seal.get("protocol_erratum_sha256") != contract.erratum_digest
        or seal.get("calibration_receipt_sha256") != _sha256_file(calibration_path)
        or seal.get("calibration_binding_sha256") != binding.digest
        or seal.get("preoutcome_receipt_sha256") != _sha256_file(preoutcome_path)
        or seal.get("runner_sha256") != _sha256_file(SCRIPT)
    ):
        _fail("E_AUTHORIZATION_BINDING")
    if (
        preoutcome.get("schema_version") != PREOUTCOME_SCHEMA
        or preoutcome.get("status") != "PASS"
        or preoutcome.get("fixture_only") is not False
        or preoutcome.get("all_gates_passed") is not True
        or tuple(preoutcome.get("gates", {})) != PREOUTCOME_GATES
        or not all(preoutcome["gates"].values())
        or preoutcome.get("calibration_binding_sha256") != binding.digest
        or preoutcome.get("execution_contract_sha256") != contract.digest
        or preoutcome.get("protocol_erratum_sha256") != contract.erratum_digest
        or preoutcome.get("runner_sha256") != _sha256_file(SCRIPT)
    ):
        _fail("E_PREOUTCOME_RECEIPT")
    if binding.fixture_only or not binding.gate_passed or binding.status != "CALIBRATION_PASS":
        _fail("E_CALIBRATION_GATE")
    return seal


@dataclass(frozen=True)
class ExecutionPlan:
    corpus_seeds: tuple[int, ...]
    model_seeds: tuple[int, ...]
    family_names: tuple[str, ...]
    range_draws: int
    concept_count: int
    repetitions: int
    inference_seed: int
    thresholds: Mapping[str, float]
    fixture_only: bool


def _bundle_request_digest(
    *, corpus_seed: int, binding: Any, contract: ExecutionContract, plan: ExecutionPlan
) -> str:
    return _digest(
        {
            "corpus_seed": corpus_seed,
            "binding": binding.digest,
            "contract": contract.digest,
            "plan": {
                "model_seeds": plan.model_seeds,
                "families": plan.family_names,
                "draws": plan.range_draws,
                "concepts": plan.concept_count,
                "repetitions": plan.repetitions,
                "inference_seed": plan.inference_seed,
                "fixture_only": plan.fixture_only,
            },
        }
    )


def execute_complete_bundle(
    *, corpus_seed: int, binding: Any, contract: ExecutionContract, plan: ExecutionPlan
) -> Mapping[str, Any]:
    families = {family.name: family for family in binding.families}
    results: list[Mapping[str, Any]] = []
    invariants: list[bool] = []
    for family_name in plan.family_names:
        family = families.get(family_name)
        if family is None:
            _fail("E_RANGE_FAMILY")
        points = deterministic_range_points(
            family, draws=7, inference_seed=plan.inference_seed
        )[: plan.range_draws]
        for draw_index, parameters in enumerate(points):
            episodes = v1.generate_fixture_episodes(
                parameters,
                corpus_seed,
                concepts=plan.concept_count,
                repetitions=plan.repetitions,
            )
            views = v1.condition_views(episodes)
            primary = {arm: v1.primary_digest(rows) for arm, rows in views.items()}
            side = {
                arm: v1.side_multiset_digest(rows)
                for arm, rows in views.items()
                if arm != "weak_vl_absent_side"
            }
            update_counts: set[int] = set()
            for model_seed in plan.model_seeds:
                for arm in contract.arms:
                    model = v1.fit_dual_corrective(views[arm], model_seed=model_seed)
                    update_counts.add(int(model["candidate_update_count"]))
                    metrics = v1.evaluate_without_side(
                        model, v1.evaluation_items(plan.concept_count)
                    )
                    results.append(
                        {
                            "range_family": family_name,
                            "draw_index": draw_index,
                            "model_seed": model_seed,
                            "arm": arm,
                            **metrics,
                        }
                    )
            invariants.append(
                tuple(views) == contract.arms
                and len(set(primary.values())) == 1
                and len(set(side.values())) == 1
                and len(update_counts) == 1
            )
    if not all(invariants):
        _fail("E_BUNDLE_INVARIANT")
    value = {
        "schema_version": BUNDLE_SCHEMA,
        "corpus_seed": corpus_seed,
        "request_sha256": _bundle_request_digest(
            corpus_seed=corpus_seed, binding=binding, contract=contract, plan=plan
        ),
        "complete_paired_bundle": True,
        "range_family_count": len(plan.family_names),
        "range_draw_count": plan.range_draws,
        "model_seed_count": len(plan.model_seeds),
        "arm_count": len(contract.arms),
        "result_rows": results,
        "scientific_outcome": not plan.fixture_only,
    }
    value["bundle_sha256"] = _digest(value)
    return value


def _validate_bundle(
    value: Mapping[str, Any],
    *,
    corpus_seed: int,
    request_sha256: str,
    plan: ExecutionPlan,
) -> None:
    payload = dict(value)
    bundle_hash = payload.pop("bundle_sha256", None)
    rows = value.get("result_rows")
    if (
        value.get("schema_version") != BUNDLE_SCHEMA
        or value.get("corpus_seed") != corpus_seed
        or value.get("request_sha256") != request_sha256
        or value.get("complete_paired_bundle") is not True
        or value.get("range_family_count") != len(plan.family_names)
        or value.get("range_draw_count") != plan.range_draws
        or value.get("model_seed_count") != len(plan.model_seeds)
        or value.get("arm_count") != len(v1.ARMS)
        or value.get("scientific_outcome") is not (not plan.fixture_only)
        or not isinstance(rows, list)
        or bundle_hash != _digest(payload)
    ):
        _fail("E_BUNDLE_CHECKPOINT")
    expected = {
        (family, draw_index, model_seed, arm)
        for family in plan.family_names
        for draw_index in range(plan.range_draws)
        for model_seed in plan.model_seeds
        for arm in v1.ARMS
    }
    observed: set[tuple[str, int, int, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            _fail("E_BUNDLE_CHECKPOINT")
        try:
            key = (
                str(row["range_family"]),
                int(row["draw_index"]),
                int(row["model_seed"]),
                str(row["arm"]),
            )
            endpoint_values = [float(row[endpoint]) for endpoint in ENDPOINTS]
        except (KeyError, TypeError, ValueError):
            _fail("E_BUNDLE_CHECKPOINT")
        if any(not math.isfinite(metric) or not 0.0 <= metric <= 1.0 for metric in endpoint_values):
            _fail("E_BUNDLE_CHECKPOINT")
        observed.add(key)
    if len(rows) != len(expected) or observed != expected:
        _fail("E_BUNDLE_CHECKPOINT")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _contrast_statistics(values: Sequence[float], thresholds: Mapping[str, float]) -> Mapping[str, Any]:
    count = len(values)
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, count - 1)
    standard_error = math.sqrt(variance / count)
    lower = mean - thresholds["one_sided_t_critical_95_df39"] * standard_error
    positive = sum(value > 0 for value in values) / count
    positive_count = sum(value > 0 for value in values)
    sign_p = sum(
        math.comb(count, successes) for successes in range(positive_count, count + 1)
    ) / (2**count)
    large_negative = sum(value <= thresholds["large_negative_threshold"] for value in values) / count
    passed = (
        mean >= thresholds["minimum_mean_effect"]
        and lower > thresholds["lower_bound_must_exceed"]
        and positive >= thresholds["minimum_positive_corpus_fraction"]
        and large_negative <= thresholds["maximum_large_negative_fraction"]
    )
    return {
        "corpus_count": count,
        "mean_effect": mean,
        "one_sided_95_lower": lower,
        "positive_corpus_fraction": positive,
        "exact_one_sided_sign_p": sign_p,
        "large_negative_fraction": large_negative,
        "passed": passed,
    }


ENDPOINTS = (
    "cue_free_heldout_noun_object_macro_top1",
    "cue_free_heldout_verb_action_macro_top1",
)
COMPARATORS = tuple(arm for arm in v1.ARMS if arm != "weak_vl_synchronized_side")


def merge_bundles(
    bundles: Sequence[Mapping[str, Any]], *, plan: ExecutionPlan
) -> Mapping[str, Any]:
    ordered = sorted(bundles, key=lambda value: int(value["corpus_seed"]))
    if tuple(int(value["corpus_seed"]) for value in ordered) != tuple(sorted(plan.corpus_seeds)):
        _fail("E_MERGE_INVENTORY")
    per_corpus: dict[tuple[str, str, str], list[float]] = {}
    for bundle in ordered:
        grouped: dict[tuple[str, str, str], list[float]] = {}
        rows = bundle.get("result_rows")
        if not isinstance(rows, list):
            _fail("E_MERGE_ROWS")
        for row in rows:
            if not isinstance(row, Mapping):
                _fail("E_MERGE_ROWS")
            family = str(row["range_family"])
            arm = str(row["arm"])
            for endpoint in ENDPOINTS:
                grouped.setdefault((family, arm, endpoint), []).append(float(row[endpoint]))
        for family in plan.family_names:
            for comparator in COMPARATORS:
                for endpoint in ENDPOINTS:
                    synchronized = grouped[(family, "weak_vl_synchronized_side", endpoint)]
                    control = grouped[(family, comparator, endpoint)]
                    per_corpus.setdefault((family, comparator, endpoint), []).append(
                        _mean(synchronized) - _mean(control)
                    )
    contrasts: dict[str, Any] = {}
    for family in plan.family_names:
        family_results: dict[str, Any] = {}
        for comparator in COMPARATORS:
            endpoint_results = {
                endpoint: _contrast_statistics(
                    per_corpus[(family, comparator, endpoint)], plan.thresholds
                )
                for endpoint in ENDPOINTS
            }
            family_results[comparator] = endpoint_results
        contrasts[family] = family_results
    all_passed = all(
        result["passed"]
        for family in contrasts.values()
        for comparator in family.values()
        for result in comparator.values()
    )
    return {
        "schema_version": MERGE_SCHEMA,
        "fixture_only": plan.fixture_only,
        "merge_order": [int(bundle["corpus_seed"]) for bundle in ordered],
        "bundle_sha256s": [str(bundle["bundle_sha256"]) for bundle in ordered],
        "contrasts": contrasts,
        "confidence_only_falsification": {
            "confidence_endpoint_present": False,
            "rule": "BOTH_NOUN_AND_VERB_WRONG_OR_TIED_TOP_MAPPINGS_MUST_BECOME_UNIQUELY_CORRECT",
            "top_mapping_endpoints_only": list(ENDPOINTS),
        },
        "side_only_leakage_falsification": {
            "maximum_above_chance": 0.02,
            "required_pre_outcome": True,
        },
        "all_range_family_endpoint_comparator_gates_passed": all_passed,
        "terminal_interpretation": "NONSCIENTIFIC_FIXTURE"
        if plan.fixture_only
        else "PROTOTYPE_PASS" if all_passed else "PROTOTYPE_STOP",
    }


def execute_checkpointed_plan(
    *,
    binding: Any,
    contract: ExecutionContract,
    plan: ExecutionPlan,
    authorization_id: str,
    output_root: Path,
    jobs: int,
) -> Mapping[str, Any]:
    if not 1 <= jobs <= 64 or AUTH_ID_RE.fullmatch(authorization_id) is None:
        _fail("E_EXECUTION_ARGUMENT")
    output_root = output_root.resolve(strict=False)
    parent = output_root.parent.resolve(strict=True)
    if output_root.exists():
        _fail("E_ONE_SHOT_COMPLETE")
    staging = parent / f".{output_root.name}.{authorization_id}.staging"
    registry_root = parent / f".{output_root.name}.one_shot_registry"
    registry_root.mkdir(mode=0o700, exist_ok=True)
    os.chmod(registry_root, 0o700)
    scope_path = registry_root / "scope.json"
    registry_path = registry_root / f"{authorization_id}.json"
    bundles_dir = staging / "bundles"
    consumption_path = staging / "one_shot_consumption.json"
    expected_registry = {
        "schema_version": CONSUMPTION_SCHEMA,
        "authorization_id": authorization_id,
        "binding_sha256": binding.digest,
        "contract_sha256": contract.digest,
        "plan_sha256": _digest(plan.__dict__),
        "output_name": output_root.name,
    }
    if scope_path.exists():
        if not _private_regular(scope_path):
            _fail("E_ONE_SHOT_REGISTRY")
        scope = _read_json(scope_path)
        if any(scope.get(key) != value for key, value in expected_registry.items()):
            _fail("E_ONE_SHOT_REGISTRY")
        if frozenset(scope) != frozenset(expected_registry):
            _fail("E_ONE_SHOT_REGISTRY")
    else:
        _exclusive_json(scope_path, expected_registry)
    if registry_path.exists():
        if not _private_regular(registry_path):
            _fail("E_ONE_SHOT_REGISTRY")
        registry = _read_json(registry_path)
        if any(registry.get(key) != value for key, value in expected_registry.items()):
            _fail("E_ONE_SHOT_REGISTRY")
        if registry.get("status") == "COMPLETE":
            _fail("E_ONE_SHOT_COMPLETE")
        if registry.get("status") != "RUNNING":
            _fail("E_ONE_SHOT_REGISTRY")
    else:
        _exclusive_json(registry_path, {**expected_registry, "status": "RUNNING"})
    if not staging.exists():
        staging.mkdir(mode=0o700)
        bundles_dir.mkdir(mode=0o700)
        _atomic_json(
            consumption_path,
            {
                "schema_version": CONSUMPTION_SCHEMA,
                "authorization_id": authorization_id,
                "status": "RUNNING",
                "binding_sha256": binding.digest,
                "contract_sha256": contract.digest,
                "plan_sha256": _digest(plan.__dict__),
                "output_name": output_root.name,
            },
        )
    else:
        if staging.is_symlink() or not bundles_dir.is_dir() or not _private_regular(consumption_path):
            _fail("E_CHECKPOINT_ROOT")
        consumption = _read_json(consumption_path)
        if (
            consumption.get("authorization_id") != authorization_id
            or consumption.get("status") != "RUNNING"
            or consumption.get("binding_sha256") != binding.digest
            or consumption.get("contract_sha256") != contract.digest
            or consumption.get("plan_sha256") != _digest(plan.__dict__)
            or consumption.get("output_name") != output_root.name
        ):
            _fail("E_CHECKPOINT_BINDING")

    pending: list[int] = []
    bundles: dict[int, Mapping[str, Any]] = {}
    for corpus_seed in plan.corpus_seeds:
        path = bundles_dir / f"corpus-{corpus_seed}.json"
        request = _bundle_request_digest(
            corpus_seed=corpus_seed, binding=binding, contract=contract, plan=plan
        )
        if path.exists():
            value = _read_json(path)
            _validate_bundle(
                value,
                corpus_seed=corpus_seed,
                request_sha256=request,
                plan=plan,
            )
            bundles[corpus_seed] = value
        else:
            pending.append(corpus_seed)

    def run_seed(seed: int) -> tuple[int, Mapping[str, Any]]:
        return seed, execute_complete_bundle(
            corpus_seed=seed, binding=binding, contract=contract, plan=plan
        )

    with ThreadPoolExecutor(max_workers=min(jobs, len(pending) or 1)) as executor:
        futures = {executor.submit(run_seed, seed): seed for seed in pending}
        for future in as_completed(futures):
            seed, value = future.result()
            _atomic_json(bundles_dir / f"corpus-{seed}.json", value)
            bundles[seed] = value
    merged = merge_bundles(list(bundles.values()), plan=plan)
    _atomic_json(staging / "aggregate_results.json", merged)
    completion = {
        "schema_version": VERSION,
        "status": "FIXTURE_COMPLETE" if plan.fixture_only else "SCIENTIFIC_ONE_SHOT_COMPLETE",
        "authorization_id": authorization_id,
        "bundle_count": len(bundles),
        "deterministic_merge_sha256": _digest(merged),
        "scientific_outcome_run": not plan.fixture_only,
        "cue_free_evaluation": True,
        "paired_complete_bundles": True,
    }
    _atomic_json(staging / "execution_receipt.json", completion)
    # Mark the consumed staging tree before the atomic no-replace publication.
    consumption = dict(_read_json(consumption_path))
    consumption["status"] = "COMPLETE"
    consumption["merge_sha256"] = _digest(merged)
    _atomic_json(staging / ".complete_consumption.json", consumption)
    # Consume the authorization externally before publishing.  A crash between
    # these operations is fail-closed: the completed staging tree remains for
    # recovery, but this authorization cannot execute again.
    _atomic_json(
        registry_path,
        {
            **expected_registry,
            "status": "COMPLETE",
            "merge_sha256": _digest(merged),
        },
    )
    try:
        os.replace(staging, output_root)
    except OSError:
        _fail("E_PUBLICATION_AFTER_CONSUMPTION")
    return completion


def scientific_plan(contract: ExecutionContract) -> ExecutionPlan:
    return ExecutionPlan(
        corpus_seeds=contract.corpus_seeds,
        model_seeds=contract.model_seeds,
        family_names=contract.range_families,
        range_draws=contract.range_draws,
        concept_count=contract.concept_count,
        repetitions=contract.repetitions,
        inference_seed=contract.inference_seed,
        thresholds=contract.thresholds,
        fixture_only=False,
    )


def fixture_plan(contract: ExecutionContract, family_name: str) -> ExecutionPlan:
    return ExecutionPlan(
        corpus_seeds=(940001, 940003),
        model_seeds=(950001,),
        family_names=(family_name,),
        range_draws=2,
        concept_count=4,
        repetitions=6,
        inference_seed=960001,
        thresholds=contract.thresholds,
        fixture_only=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="build pre-outcome gates only")
    preflight.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    preflight.add_argument("--execution-contract", type=Path, default=DEFAULT_CONTRACT)
    preflight.add_argument("--calibration-receipt", type=Path, default=DEFAULT_CALIBRATION)
    preflight.add_argument("--output", type=Path, required=True)
    benchmark = commands.add_parser("benchmark-smoke", help="tiny non-study fixture only")
    benchmark.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    benchmark.add_argument("--execution-contract", type=Path, default=DEFAULT_CONTRACT)
    benchmark.add_argument("--ranges", type=Path, default=DEFAULT_FIXTURE_RANGES)
    benchmark.add_argument("--output-root", type=Path, required=True)
    benchmark.add_argument("--jobs", type=int, default=2)
    run = commands.add_parser("run", help="authorized one-shot scientific execution")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--execution-contract", type=Path, required=True)
    run.add_argument("--calibration-receipt", type=Path, required=True)
    run.add_argument("--preoutcome-receipt", type=Path, required=True)
    run.add_argument("--authorization-seal", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--jobs", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        contract = load_execution_contract(
            arguments.execution_contract, arguments.protocol
        )
        if arguments.command == "benchmark-smoke":
            binding = bind_calibration_ranges_v2(
                _read_json(arguments.ranges), fixture_only=True
            )
            family_name = binding.families[0].name
            plan = fixture_plan(contract, family_name)
            receipt = execute_checkpointed_plan(
                binding=binding,
                contract=contract,
                plan=plan,
                authorization_id="fixture-smoke-v2-0001",
                output_root=arguments.output_root,
                jobs=arguments.jobs,
            )
            print(json.dumps(receipt, sort_keys=True))
            return 0
        calibration_document = _read_json(arguments.calibration_receipt)
        binding = bind_calibration_ranges_v2(
            calibration_document, fixture_only=False
        )
        if arguments.command == "preflight":
            receipt = build_preoutcome_receipt(
                binding,
                contract,
                fixture_only=False,
                calibration_document=calibration_document,
            )
            _atomic_json(arguments.output, receipt)
            print(json.dumps({"status": receipt["status"], "scientific_endpoints_opened": False}))
            return 0 if receipt["status"] == "PASS" else 2
        seal = validate_authorization_seal(
            arguments.authorization_seal,
            protocol_path=arguments.protocol,
            contract_path=arguments.execution_contract,
            calibration_path=arguments.calibration_receipt,
            calibration_document=calibration_document,
            binding=binding,
            preoutcome_path=arguments.preoutcome_receipt,
            contract=contract,
        )
        receipt = execute_checkpointed_plan(
            binding=binding,
            contract=contract,
            plan=scientific_plan(contract),
            authorization_id=str(seal["authorization_id"]),
            output_root=arguments.output_root,
            jobs=arguments.jobs,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    except (OneShotError, v1.SymbolicRunnerError) as exc:
        print(getattr(exc, "code", "E_FAIL_CLOSED"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
