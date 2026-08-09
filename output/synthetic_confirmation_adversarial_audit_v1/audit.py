#!/usr/bin/env python3
"""Independent, read-only audit of the closed Nursery V3 development result.

This program never imports or calls project runners, generators, learners,
detectors, firewalls, or adjudicators. It reads already-persisted evidence and
the prior execution task's retained recomputation tree, writes only inside this
new audit directory, and verifies every pre-existing working-tree file against
the baseline captured before this audit began.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DEV = ROOT / "output/synthetic_development_v3_one_shot"
PKG = ROOT / "output/synthetic_development_launch_package_v3"
RECOMPUTE = Path("/tmp/nursery-development-v3-audit.sIEJgF/recompute")
BASELINE_SHA = Path("/private/tmp/nursery_confirmation_baseline_20260716.sha256")
BASELINE_STAT = Path("/private/tmp/nursery_confirmation_baseline_20260716.stat")
BASELINE_LINKS = Path("/private/tmp/nursery_confirmation_baseline_20260716.symlinks")

EXPECTED_HASHES = {
    "ONE_SHOT_EXECUTION.json": "2e043cc042ddf2727f9ba10cbe7cdac46a413839c6b9ec5f64c338753954c9ee",
    "cohort_summary.json": "0217203e707a761cf488a85a11c52c1711309cc3c310d9fb2fc7a43ccf561727",
    "recomputable/development_inference.json": "e4dfc945ff2bb3777a3f0354d9b2fd6ede18a272480b5fdd932d0e610506dd6d",
    "identifier_reference_audit.json": "c75778a012448984c88160ee07f6a8a7c2ee0716d76e2b8f075490e061653edd",
    "persisted_inputs/input_manifest.json": "d8e31df113f47c10397abe76a58eaebd79b9100f13defb92b9e8aa1581e64453",
}
EXPECTED_INPUT_DIGEST = "31b2600e1a6685b21a857f1054b926d483e72c847decb6ba54a8edcd738e2a68"
EXPECTED_CLOSED_INVENTORY = "8b97ec7c5f4667b1fd0236bd556d92ff6c0f4ea807bdfabd5d9b44d974d5150f"
EXPECTED_RECOMPUTABLE = {
    "compute_operation_log.json",
    "control_results.json",
    "core_gates.json",
    "development_controls.json",
    "development_inference.json",
    "factor_audit.json",
    "factor_results.json",
    "leakage_audit.json",
    "learner_numeric_audit.json",
    "model_averages.json",
    "model_results.json",
    "null_capacity_audit.json",
    "order_invariance_audit.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_bytes(canonical_bytes(value))
    else:
        path.write_text(str(value))


def verify_manifest(root: Path, manifest_path: Path, *, exclude_manifest: bool) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    rows = manifest["files"]
    problems: list[dict[str, Any]] = []
    paths = [str(row["path"]) for row in rows]
    for row in rows:
        relative = str(row["path"])
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            problems.append({"path": relative, "problem": "path_escape"})
            continue
        if not path.is_file():
            problems.append({"path": relative, "problem": "missing"})
            continue
        observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if observed != {"bytes": int(row["bytes"]), "sha256": str(row["sha256"])}:
            problems.append({"path": relative, "problem": "bytes_or_sha256", "observed": observed})
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and (not exclude_manifest or path.resolve() != manifest_path.resolve())
    }
    declared = set(paths)
    return {
        "status": "PASS" if not problems and actual == declared and len(paths) == len(set(paths))
        and int(manifest["file_count"]) == len(rows) and canonical_digest(rows) == manifest["digest"] else "FAIL",
        "declared_count": len(rows),
        "unique_path_count": len(set(paths)),
        "actual_count": len(actual),
        "file_count_field": int(manifest["file_count"]),
        "manifest_digest": str(manifest["digest"]),
        "manifest_digest_recomputed": canonical_digest(rows),
        "missing": sorted(declared - actual),
        "unexpected": sorted(actual - declared),
        "problems": problems,
    }


def development_integrity() -> dict[str, Any]:
    hashes = {}
    for relative, expected in EXPECTED_HASHES.items():
        observed = sha256_file(DEV / relative)
        hashes[relative] = {"expected": expected, "observed": observed, "match": observed == expected}

    inputs = DEV / "persisted_inputs"
    input_manifest = verify_manifest(inputs, inputs / "input_manifest.json", exclude_manifest=True)
    input_manifest["expected_digest"] = EXPECTED_INPUT_DIGEST
    input_manifest["expected_digest_match"] = input_manifest["manifest_digest"] == EXPECTED_INPUT_DIGEST

    package_manifest = verify_manifest(
        PKG, PKG / "complete_file_manifest.json", exclude_manifest=True
    )
    snapshot_manifest = verify_manifest(
        PKG / "frozen_source_snapshot",
        PKG / "frozen_source_snapshot/snapshot_manifest.json",
        exclude_manifest=True,
    )

    recompute_rows = []
    for name in sorted(EXPECTED_RECOMPUTABLE):
        original = DEV / "recomputable" / name
        staged = RECOMPUTE / name
        original_hash = sha256_file(original)
        staged_hash = sha256_file(staged)
        recompute_rows.append(
            {
                "path": name,
                "original_bytes": original.stat().st_size,
                "staged_bytes": staged.stat().st_size,
                "original_sha256": original_hash,
                "staged_sha256": staged_hash,
                "byte_identical": original_hash == staged_hash and original.stat().st_size == staged.stat().st_size,
            }
        )

    inventory = hashlib.sha256()
    inventory_files = []
    total_bytes = 0
    for path in sorted((path for path in DEV.rglob("*") if path.is_file()), key=lambda value: value.relative_to(DEV).as_posix()):
        relative = path.relative_to(DEV).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        inventory.update(f"{relative}\0{size}\0{digest}\n".encode())
        total_bytes += size
        inventory_files.append(relative)
    inventory_digest = inventory.hexdigest()

    markers = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.glob("output/**/ONE_SHOT_EXECUTION.json")
    )
    one_shot = load_json(DEV / "ONE_SHOT_EXECUTION.json")
    cohort = load_json(DEV / "cohort_summary.json")
    closed_gates = cohort["gates"]
    core = load_json(DEV / "recomputable/core_gates.json")
    return {
        "status": "PASS" if all(row["match"] for row in hashes.values())
        and input_manifest["status"] == "PASS" and input_manifest["expected_digest_match"]
        and package_manifest["status"] == "PASS" and snapshot_manifest["status"] == "PASS"
        and len(recompute_rows) == 13 and all(row["byte_identical"] for row in recompute_rows)
        and inventory_digest == EXPECTED_CLOSED_INVENTORY else "FAIL",
        "key_hashes": hashes,
        "persisted_input_manifest": input_manifest,
        "launch_package_manifest": package_manifest,
        "source_snapshot_manifest": snapshot_manifest,
        "persisted_recomputation": {
            "status": "PASS" if all(row["byte_identical"] for row in recompute_rows) else "FAIL",
            "source": str(RECOMPUTE),
            "files_compared": len(recompute_rows),
            "rows": recompute_rows,
            "execution_transcript": {
                "task_id": "019f6d42-ac65-7c31-9fd8-82e06fd7b548",
                "rollout_path": "/Users/rishisim/.codex/sessions/2026/07/16/rollout-2026-07-16T16-28-33-019f6d42-ac65-7c31-9fd8-82e06fd7b548.jsonl",
                "recorded_run_development_invocations": 1,
                "run_development_exit_code": 0,
                "recorded_recompute_invocations": 1,
                "recompute_exit_code": 0,
            },
        },
        "closed_output_inventory": {
            "file_count": len(inventory_files),
            "total_bytes": total_bytes,
            "digest_scheme": "sorted path\\0bytes\\0sha256\\n",
            "expected_digest": EXPECTED_CLOSED_INVENTORY,
            "observed_digest": inventory_digest,
            "match": inventory_digest == EXPECTED_CLOSED_INVENTORY,
        },
        "one_shot_markers_in_output": markers,
        "marker": one_shot,
        "cohort_terminal": {
            "status": cohort["status"],
            "development_decision": cohort["development_decision"],
            "development_outcomes": cohort["development_outcomes"],
            "confirmation_outcomes": cohort["confirmation_outcomes"],
            "confirmation_authorized": cohort["confirmation_authorized"],
        },
        "closed_gates": {
            "passed": sum(value is True for value in closed_gates.values()),
            "total": len(closed_gates),
            "failed": sorted(key for key, value in closed_gates.items() if value is not True),
        },
        "recomputable_gates": {
            "passed": sum(value is True for value in core["gates"].values()),
            "total": len(core["gates"]),
            "failed": sorted(key for key, value in core["gates"].items() if value is not True),
        },
        "guard_defects": [
            "Authorization validates a command string stored in the payload but does not bind actual parsed CLI paths.",
            "The caller may select any absent output root, so the same authorization is replayable.",
            "The frozen package outcome registry remains at its prelaunch zero state and is not consumed.",
            "The frozen manifest verifier checks listed rows but not the exact file set, file_count, duplicates, or path confinement.",
            "Current bytes were independently closed by stricter checks above; these are mechanism defects, not evidence that a replay occurred.",
        ],
    }


def correlation(left: Iterable[float], right: Iterable[float]) -> float:
    return float(np.corrcoef(np.asarray(list(left), dtype=float), np.asarray(list(right), dtype=float))[0, 1])


def statistical_and_scientific_audit() -> dict[str, Any]:
    averaged = load_json(DEV / "recomputable/model_averages.json")
    results = load_json(DEV / "recomputable/model_results.json")
    inference = load_json(DEV / "recomputable/development_inference.json")
    config = load_json(DEV / "persisted_inputs/config.json")
    seeds = list(map(int, config["resolved_registries"]["development"]["corpus"]))
    model_seeds = list(map(int, config["resolved_registries"]["development"]["model"]))

    index = {
        (int(row["corpus_seed"]), str(row["condition"]), str(row["kind"]), str(row["presence"])): row["metrics"]
        for row in averaged["rows"]
    }
    definitions = {
        "synchronized_minus_absent_channel": ("present", "absent_channel", 0.005, "co_primary"),
        "synchronized_minus_randomized_shuffle": ("present", "randomized_shuffle", 0.005, "co_primary"),
        "synchronized_minus_absent_channel_null": ("null", "absent_channel", -0.02, "null_noninferiority"),
        "synchronized_minus_randomized_shuffle_null": ("null", "randomized_shuffle", -0.02, "null_noninferiority"),
    }
    contrast_rows = {}
    vectors: dict[str, list[float]] = {}
    for name, (presence, comparator, threshold, section) in definitions.items():
        values = [
            float(index[(seed, "synchronized", "action", presence)]["mean_correct_probability"])
            - float(index[(seed, comparator, "action", presence)]["mean_correct_probability"])
            for seed in seeds
        ]
        vectors[name] = values
        seed_bytes = hashlib.sha256(f"development-v3|281201|{name}".encode()).digest()[:8]
        rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
        samples = rng.integers(0, len(values), size=(20000, len(values)))
        means = np.asarray(values, dtype=float)[samples].mean(axis=1)
        lower = float(np.quantile(means, 0.05, method="linear"))
        saved = inference[section][name]
        shifted = [value - threshold for value in values]
        positive = sum(value > 0 for value in shifted)
        sign_p = sum(math.comb(len(shifted), count) for count in range(positive, len(shifted) + 1)) / (2 ** len(shifted))
        contrast_rows[name] = {
            "values": values,
            "value_count": len(values),
            "distinct_value_count": len(set(values)),
            "mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)),
            "minimum": min(values),
            "maximum": max(values),
            "positive_relative_to_threshold": positive,
            "threshold": threshold,
            "sign_test_p": float(sign_p),
            "bootstrap_lcb_recomputed": lower,
            "bootstrap_lcb_saved": float(saved["lower_confidence_bound"]),
            "mean_saved": float(saved["mean_difference"]),
            "values_digest_recomputed": canonical_digest(values),
            "values_digest_saved": str(saved["values_digest"]),
            "exact_match": lower == float(saved["lower_confidence_bound"])
            and float(np.mean(values)) == float(saved["mean_difference"])
            and canonical_digest(values) == saved["values_digest"],
        }

    present_absent = vectors["synchronized_minus_absent_channel"]
    null_absent = vectors["synchronized_minus_absent_channel_null"]
    present_random = vectors["synchronized_minus_randomized_shuffle"]
    null_random = vectors["synchronized_minus_randomized_shuffle_null"]

    all_action_cells = []
    focal_action_cells = []
    probability_groups: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    model_groups: defaultdict[tuple[int, str, str], list[float]] = defaultdict(list)
    digest_groups: defaultdict[tuple[int, str], set[str]] = defaultdict(set)
    for row in results:
        corpus_seed = int(row["corpus_seed"])
        condition = str(row["condition"])
        digest_groups[(corpus_seed, condition)].add(str(row["model_digest"]))
        for presence in ("present", "null"):
            metrics = row["metrics"]["by_kind_presence"]["action"][presence]
            cell = {
                "fractional_accuracy": float(metrics["fractional_accuracy"]),
                "tie_frequency": float(metrics["tie_frequency"]),
            }
            all_action_cells.append(cell)
            if condition in {"synchronized", "absent_channel", "randomized_shuffle"}:
                focal_action_cells.append(cell)
            probability_groups[(condition, presence)].append(float(metrics["mean_correct_probability"]))
            model_groups[(corpus_seed, condition, presence)].append(float(metrics["mean_correct_probability"]))
    model_sds = [statistics.stdev(values) for values in model_groups.values()]

    manifest = load_json(DEV / "persisted_inputs/input_manifest.json")
    suffix_digests: defaultdict[str, list[str]] = defaultdict(list)
    for row in manifest["files"]:
        path = str(row["path"])
        if path.startswith("corpus_"):
            suffix = path.split("/", 1)[1]
            suffix_digests[suffix].append(str(row["sha256"]))
    distinct_streams = {
        suffix: {"corpora": len(digests), "unique_sha256": len(set(digests))}
        for suffix, digests in sorted(suffix_digests.items())
    }

    learner = config["learner"]
    frozen_source_checks = {}
    for relative in (
        "babyworld_lite/sensor_alignment_v8/learner.py",
        "babyworld_lite/sensor_alignment_v8/benchmark.py",
        "babyworld_lite/development_launch_v3/statistics.py",
        "babyworld_lite/development_launch_v3/study.py",
    ):
        live = ROOT / relative
        frozen = PKG / "frozen_source_snapshot" / relative
        frozen_source_checks[relative] = {
            "live_sha256": sha256_file(live),
            "frozen_sha256": sha256_file(frozen),
            "byte_identical": sha256_file(live) == sha256_file(frozen),
        }

    means = {
        f"{condition}|{presence}": float(np.mean(values))
        for (condition, presence), values in sorted(probability_groups.items())
    }
    all_accuracy_one = all(cell["fractional_accuracy"] == 1.0 for cell in all_action_cells)
    all_ties_zero = all(cell["tie_frequency"] == 0.0 for cell in all_action_cells)
    code_path_evidence = [
        {"path": "output/synthetic_development_launch_package_v3/frozen_config_snapshot.yaml", "lines": "103-126", "fact": "joint update and null calibration weights are 0.0; sharpening weight is 0.50"},
        {"path": "output/synthetic_development_launch_package_v3/frozen_source_snapshot/babyworld_lite/sensor_alignment_v8/learner.py", "lines": "249-327", "fact": "detector contribution requires agreement with the learner's current maximum"},
        {"path": "output/synthetic_development_launch_package_v3/frozen_source_snapshot/babyworld_lite/sensor_alignment_v8/learner.py", "lines": "385-393", "fact": "the only detector-selected semantic update is multiplied by the frozen zero joint-update weight"},
        {"path": "output/synthetic_development_launch_package_v3/frozen_source_snapshot/babyworld_lite/sensor_alignment_v8/learner.py", "lines": "450-460", "fact": "the operative path power-sharpens an already inferred distribution and preserves component ordering"},
        {"path": "output/synthetic_development_launch_package_v3/frozen_source_snapshot/babyworld_lite/sensor_alignment_v8/learner.py", "lines": "461-487", "fact": "null mass is derived from semantic top/runner-up margins; declared presence_smoothing does not enter the update"},
        {"path": "output/synthetic_development_launch_package_v3/frozen_source_snapshot/babyworld_lite/sensor_alignment_v8/benchmark.py", "lines": "575-615", "fact": "action factor schedule and candidate lattice are corpus-seed invariant"},
        {"path": "output/synthetic_development_launch_package_v3/frozen_source_snapshot/babyworld_lite/development_launch_v3/statistics.py", "lines": "209-250", "fact": "the co-primary endpoint is mean correct probability, not top-1 accuracy"},
        {"path": "docs/synthetic_identifiability_qualification_v7_repair_ledger.json", "lines": "70-99", "fact": "fixture cycles selected the corroboration/sharpening rule after no-effect and harmful variants"},
    ]
    conclusion = {
        "terminal": "CONFIRMATION_STOP",
        "reason_code": "ALREADY_CORRECT_MAPPING_CONFIDENCE_SHARPENING_IS_NOT_A_CONFIRMATORY_SEMANTIC_GROUNDING_TEST",
        "reason": (
            "The frozen sensor path cannot supply or correct an action meaning: disagreement is ignored, "
            "the detector-selected semantic update has weight zero, and the operative power transform only "
            "sharpens the learner's existing ordering. Every persisted action-present and action-null cell "
            "already has perfect fractional top accuracy in every condition. The declared co-primary gains "
            "therefore measure detector-agreement-conditioned confidence sharpening of already-correct "
            "mappings. The nearly identical present/null effects are a structural consequence of that shared "
            "distribution, not independent grounding and presence evidence. Repeating the same seed-invariant "
            "geometry on 289xxx would confirm this constructed calibration mechanism, not the frozen claim "
            "of sensor-assisted lexical/action grounding. Renaming the claim or changing the learner/design "
            "would broaden or alter the frozen science and is prohibited."
        ),
    }
    return {
        "status": "PASS" if all(row["exact_match"] for row in contrast_rows.values()) else "FAIL",
        "frozen_inference_reconstruction": contrast_rows,
        "present_null_dependence": {
            "sync_minus_absent_correlation": correlation(present_absent, null_absent),
            "sync_minus_randomized_correlation": correlation(present_random, null_random),
            "sync_minus_absent_present_minus_null_mean": float(np.mean(np.asarray(present_absent) - np.asarray(null_absent))),
            "sync_minus_randomized_present_minus_null_mean": float(np.mean(np.asarray(present_random) - np.asarray(null_random))),
            "interpretation": "shared alignment-conditioned confidence sharpening; not independent present and null evidence",
        },
        "action_ceiling": {
            "unaveraged_action_presence_cells_all_conditions": len(all_action_cells),
            "unaveraged_focal_action_presence_cells": len(focal_action_cells),
            "all_fractional_accuracy_exactly_one": all_accuracy_one,
            "all_tie_frequency_zero": all_ties_zero,
            "averaged_action_presence_cells": sum(1 for row in averaged["rows"] if row["kind"] == "action"),
            "averaged_all_fractional_accuracy_exactly_one": all(
                float(row["metrics"]["fractional_accuracy"]) == 1.0
                for row in averaged["rows"] if row["kind"] == "action"
            ),
            "mean_correct_probability_by_condition_presence": means,
        },
        "model_replicates": {
            "model_seeds": model_seeds,
            "corpus_condition_groups": len(digest_groups),
            "all_groups_have_three_distinct_serialized_digests": all(len(values) == 3 for values in digest_groups.values()),
            "serialized_digest_includes_model_seed_metadata": True,
            "mean_within_corpus_condition_presence_sample_sd": float(np.mean(model_sds)),
            "minimum_sd": min(model_sds),
            "maximum_sd": max(model_sds),
            "permitted_claim": "numerical convergence for these three tiny deterministic hash jitters",
            "forbidden_claim": "model-initialization-population, architecture, or hyperparameter robustness",
        },
        "corpus_units": {
            "seeds": seeds,
            "seed_count": len(seeds),
            "unique_seed_count": len(set(seeds)),
            "stream_sha256_uniqueness": distinct_streams,
            "all_visible_episode_files_unique": distinct_streams["visible_episodes.jsonl"]["unique_sha256"] == 40,
            "all_oracle_episode_files_unique": distinct_streams["oracle_episodes.jsonl"]["unique_sha256"] == 40,
            "all_evaluation_prompt_files_unique": distinct_streams["evaluation_prompts.jsonl"]["unique_sha256"] == 40,
            "all_lexicon_files_unique": distinct_streams["lexicon_oracle.json"]["unique_sha256"] == 40,
            "structural_heterogeneity": "fixed, exactly balanced action geometry; stochastic raw noise/token/donor realizations only",
            "pseudoreplication": False,
            "ecological_heterogeneity_supported": False,
        },
        "frozen_weights": {
            "sensor_weight": float(learner["sensor_weight"]),
            "sensor_null_calibration_weight": float(learner["sensor_null_calibration_weight"]),
            "sensor_joint_update_weight": float(learner["sensor_joint_update_weight"]),
            "sensor_semantic_sharpening_weight": float(learner["sensor_semantic_sharpening_weight"]),
            "initialization_jitter": float(learner["initialization_jitter"]),
        },
        "frozen_source_matches_live": frozen_source_checks,
        "code_path_evidence": code_path_evidence,
        "direct_leakage_audit": {
            "oracle_or_keys_consumed_by_primary_fit": False,
            "side_modalities_present_at_language_evaluation": False,
            "condition_state_reuse_found": False,
            "order_effect_found": False,
            "post_development_adaptation_found": False,
            "predevelopment_fixture_mechanism_selection_found": True,
            "note": "The STOP is measurement/construct invalidity, not direct answer-key leakage.",
        },
        "terminal_decision": conclusion,
    }


def registry_audit() -> dict[str, Any]:
    seed_log = load_json(DEV / "seed_operation_log.json")
    operations = seed_log["operations"]
    compute = load_json(DEV / "recomputable/compute_operation_log.json")
    operation_counts = Counter(str(row["operation"]) for row in operations)
    references: defaultdict[str, set[int]] = defaultdict(set)
    for row in operations:
        for reference in row["references"]:
            references[str(reference["role"])].add(int(reference["value"]))

    parsed_logs = []
    confirmation_references = []
    candidates = sorted(
        set(ROOT.glob("output/**/*operation_log*.json"))
        | set(ROOT.glob("output/**/seed_operation_log.json"))
    )
    for path in candidates:
        value = load_json(path)
        rows = value.get("operations", []) if isinstance(value, dict) else value
        if not isinstance(rows, list):
            continue
        parsed_logs.append(path.relative_to(ROOT).as_posix())
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            for reference in row.get("references", []):
                if not isinstance(reference, dict):
                    continue
                raw_number = reference.get("value", reference.get("seed"))
                if raw_number is None:
                    continue
                number = int(raw_number)
                if 289000 <= number < 290000:
                    confirmation_references.append(
                        {"path": path.relative_to(ROOT).as_posix(), "index": index, "reference": reference}
                    )

    detector_runtime = load_json(DEV / "detector_runtime_audit.json")
    detector_capacity = load_json(DEV / "detector_capacity_audit.json")
    top_confirmation_dirs = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "output").iterdir()
        if path.is_dir() and "confirmation" in path.name.lower()
        and path.resolve() != OUT.resolve()
    )
    return {
        "status": "PASS" if len(operations) == 8688 and int(compute["operation_count"]) == 8128
        and not confirmation_references and not top_confirmation_dirs else "FAIL",
        "development_operation_count": len(operations),
        "compute_operation_count": int(compute["operation_count"]),
        "development_operation_types": dict(sorted(operation_counts.items())),
        "all_development_operations_allowed": all(row["status"] == "ALLOW" and row["purpose"] == "development" for row in operations),
        "references_by_role": {key: sorted(values) for key, values in sorted(references.items())},
        "operation_logs_parsed": len(parsed_logs),
        "operation_log_paths": parsed_logs,
        "289xxx_operation_references": confirmation_references,
        "confirmation_output_directories_before_this_audit": top_confirmation_dirs,
        "confirmation_outcome_count": int(load_json(DEV / "ONE_SHOT_EXECUTION.json")["confirmation_outcome_count"]),
        "confirmation_authorized": bool(load_json(DEV / "ONE_SHOT_EXECUTION.json")["confirmation_authorized"]),
        "detector_runtime": {
            "status": detector_runtime["status"],
            "inference_calls": int(detector_runtime["inference_calls"]),
            "minimum_required_calls": int(detector_runtime["minimum_required_calls"]),
            "hash_only_placeholder": bool(detector_runtime["hash_only_placeholder"]),
            "perturbation_output_changed": bool(detector_runtime["perturbation"]["output_changed"]),
        },
        "detector_capacity": {
            "status": detector_capacity["status"],
            "overall_fractional_top_event_or_null_accuracy": float(detector_capacity["overall_fractional_top_event_or_null_accuracy"]),
            "overall_minimum": float(detector_capacity["overall_minimum"]),
            "overall_maximum": float(detector_capacity["overall_maximum"]),
        },
    }


def parse_baseline() -> tuple[list[tuple[str, str]], dict[str, tuple[int, int]]]:
    hashes = []
    for line in BASELINE_SHA.read_text().splitlines():
        digest = line[:64]
        relative = line[66:]
        hashes.append((relative, digest))
    stats = {}
    for line in BASELINE_STAT.read_text().splitlines():
        size_text, mode_text, relative = line.split(" ", 2)
        stats[relative] = (int(size_text), int(mode_text, 8))
    return hashes, stats


def preservation_check() -> dict[str, Any]:
    copied = OUT / "preservation"
    copied.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(BASELINE_SHA, copied / "baseline.sha256")
    shutil.copyfile(BASELINE_STAT, copied / "baseline.stat")
    shutil.copyfile(BASELINE_LINKS, copied / "baseline.symlinks")

    hashes, stats = parse_baseline()
    changed = []
    missing = []
    after_lines = []
    for relative, expected_digest in hashes:
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        observed_digest = sha256_file(path)
        expected_size, expected_mode = stats[relative]
        observed_size = path.stat().st_size
        observed_mode = path.stat().st_mode & 0o7777
        after_lines.append(f"{observed_digest}  {relative}")
        if observed_digest != expected_digest or observed_size != expected_size or observed_mode != expected_mode:
            changed.append(
                {
                    "path": relative,
                    "expected_sha256": expected_digest,
                    "observed_sha256": observed_digest,
                    "expected_bytes": expected_size,
                    "observed_bytes": observed_size,
                    "expected_mode": expected_mode,
                    "observed_mode": observed_mode,
                }
            )
    links_before = BASELINE_LINKS.read_text().splitlines()
    links_after = []
    link_changes = []
    for line in links_before:
        relative, target = line.split("\t", 1)
        path = ROOT / relative
        observed = path.readlink().as_posix() if path.is_symlink() else None
        links_after.append(f"{relative}\t{observed}")
        if observed != target:
            link_changes.append({"path": relative, "expected": target, "observed": observed})

    write_new(copied / "after.sha256", "\n".join(after_lines) + "\n")
    write_new(copied / "after.symlinks", "\n".join(links_after) + "\n")
    proof = {
        "status": "PASS" if not changed and not missing and not link_changes and len(after_lines) == len(hashes) else "FAIL",
        "baseline_file_count": len(hashes),
        "verified_file_count": len(after_lines),
        "baseline_symlink_count": len(links_before),
        "verified_symlink_count": len(links_after),
        "baseline_sha256_manifest_sha256": sha256_file(BASELINE_SHA),
        "baseline_stat_manifest_sha256": sha256_file(BASELINE_STAT),
        "baseline_symlink_manifest_sha256": sha256_file(BASELINE_LINKS),
        "missing_paths": missing,
        "changed_paths": changed,
        "changed_symlinks": link_changes,
        "all_preexisting_bytes_unchanged": not changed and not missing and len(after_lines) == len(hashes),
        "new_audit_files_are_not_part_of_the_preedit_baseline": True,
    }
    write_new(OUT / "preservation_proof.json", proof)
    return proof


def package_manifest() -> dict[str, Any]:
    rows = []
    for path in sorted((path for path in OUT.rglob("*") if path.is_file() and path.name != "complete_file_manifest.json"), key=lambda value: value.relative_to(OUT).as_posix()):
        rows.append(
            {
                "path": path.relative_to(OUT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "protocol_id": "synthetic-confirmation-adversarial-audit-v1",
        "terminal": "CONFIRMATION_STOP",
        "self_excluded_by_definition": True,
        "file_count": len(rows),
        "files": rows,
        "digest": canonical_digest(rows),
    }
    write_new(OUT / "complete_file_manifest.json", manifest)
    return manifest


def main() -> int:
    for required in (DEV, PKG, RECOMPUTE, BASELINE_SHA, BASELINE_STAT, BASELINE_LINKS):
        if not required.exists():
            raise FileNotFoundError(required)
    generated = [
        OUT / "development_integrity_audit.json",
        OUT / "scientific_diagnostics.json",
        OUT / "registry_audit.json",
        OUT / "CONFIRMATION_STOP.md",
        OUT / "preservation_proof.json",
        OUT / "complete_file_manifest.json",
    ]
    if any(path.exists() for path in generated):
        raise FileExistsError("audit is append-only and has already been generated")

    integrity = development_integrity()
    science = statistical_and_scientific_audit()
    registry = registry_audit()
    write_new(OUT / "development_integrity_audit.json", integrity)
    write_new(OUT / "scientific_diagnostics.json", science)
    write_new(OUT / "registry_audit.json", registry)

    terminal = science["terminal_decision"]
    summary = f"""# CONFIRMATION_STOP

The untouched 289xxx confirmation must not run.

## Decisive reason

{terminal['reason']}

## Integrity disposition

- Closed development bytes: {integrity['status']}
- Persisted-input exact manifest: {integrity['persisted_input_manifest']['status']} ({integrity['persisted_input_manifest']['declared_count']} files)
- Independent recomputation identity: {integrity['persisted_recomputation']['status']} ({integrity['persisted_recomputation']['files_compared']}/13)
- Frozen inference reconstruction: {science['status']}
- Registry/untouched reserve: {registry['status']}
- Confirmation outcome count: {registry['confirmation_outcome_count']}

## Construct audit

- All {science['action_ceiling']['unaveraged_action_presence_cells_all_conditions']} unaveraged action present/null cells across all seven conditions have fractional accuracy exactly 1.0 and no ties.
- Present/null contrast correlations are {science['present_null_dependence']['sync_minus_absent_correlation']:.12f} and {science['present_null_dependence']['sync_minus_randomized_correlation']:.12f}.
- Mean within-corpus model-replicate SD is {science['model_replicates']['mean_within_corpus_condition_presence_sample_sd']:.12g}; it supports convergence for three frozen jitters, not model-population robustness.
- The 40 corpus inputs are byte-distinct stochastic realizations, but the action factor schedule and candidate lattice are deliberately seed-invariant.
- No direct key/oracle leakage into the primary fit was found. The failure is construct validity: corroborative sharpening of an already-correct mapping is not evidence that sensors supplied or corrected lexical/action grounding.

No confirmation package, authorization, or launch command was created. Renaming the result as confidence calibration or redesigning the learner would alter the frozen claim/design and is outside this confirmation.
"""
    write_new(OUT / "CONFIRMATION_STOP.md", summary)

    preservation = preservation_check()
    if integrity["status"] != "PASS" or science["status"] != "PASS" or registry["status"] != "PASS" or preservation["status"] != "PASS":
        raise RuntimeError("audit evidence or preservation verification failed")
    manifest = package_manifest()
    print(json.dumps({
        "terminal": "CONFIRMATION_STOP",
        "integrity": integrity["status"],
        "science_reconstruction": science["status"],
        "registry": registry["status"],
        "preservation": preservation["status"],
        "package_files": manifest["file_count"],
        "package_digest": manifest["digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
