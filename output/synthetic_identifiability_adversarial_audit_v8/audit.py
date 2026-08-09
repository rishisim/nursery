#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify_manifest(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    listed = {str(row["path"]): row for row in manifest.get("files", [])}
    problems = []
    for relative, row in sorted(listed.items()):
        path = root / relative
        if not path.is_file():
            problems.append({"path": relative, "problem": "missing"})
            continue
        observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        expected = {"bytes": int(row["bytes"]), "sha256": str(row["sha256"])}
        if observed != expected:
            problems.append(
                {
                    "path": relative,
                    "problem": "digest_or_size",
                    "expected": expected,
                    "observed": observed,
                }
            )
    return {
        "status": "PASS" if not problems and len(listed) > 0 else "FAIL",
        "checked_files": len(listed),
        "problems": problems,
    }


def recursive_max_numeric_delta(left: Any, right: Any) -> float:
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0 if left == right else float("inf")
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return float("inf")
        return max(
            (
                recursive_max_numeric_delta(left[key], right[key])
                for key in left
            ),
            default=0.0,
        )
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
        and not isinstance(right, (str, bytes))
    ):
        if len(left) != len(right):
            return float("inf")
        return max(
            (
                recursive_max_numeric_delta(a, b)
                for a, b in zip(left, right)
            ),
            default=0.0,
        )
    return 0.0 if left == right else float("inf")


def subgroup_pass(metrics: Mapping[str, Any], presence: str, gates: Mapping[str, Any]) -> bool:
    fractional_key = (
        "minimum_null_fractional_margin_over_exact_chance"
        if presence == "null"
        else "minimum_present_fractional_margin_over_exact_chance"
    )
    probability_key = (
        "minimum_null_probability_margin_over_exact_chance"
        if presence == "null"
        else "minimum_present_probability_margin_over_exact_chance"
    )
    return (
        float(metrics["fractional_margin_over_chance"]) >= float(gates[fractional_key])
        and float(metrics["probability_margin_over_chance"]) >= float(gates[probability_key])
        and float(metrics["log_loss_improvement_over_exact_chance"])
        >= float(gates["minimum_log_loss_improvement_over_exact_chance"])
        and float(metrics["brier_improvement_over_exact_chance"])
        >= float(gates["minimum_brier_improvement_over_exact_chance"])
        and float(metrics["tie_frequency"]) <= float(gates["maximum_tie_frequency"])
    )


def permute_prompts_and_keys(
    prompts: Sequence[Mapping[str, Any]],
    keys: Sequence[Mapping[str, Any]],
    variant: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    key_by_id = {str(row["prompt_id"]): int(row["answer_index"]) for row in keys}
    permuted_prompts = []
    permuted_keys = []
    prompt_order = sorted(
        range(len(prompts)),
        key=lambda index: hashlib.sha256(
            f"v8-adversarial-prompt|{variant}|{prompts[index]['prompt_id']}".encode()
        ).hexdigest(),
    )
    for prompt_index in prompt_order:
        prompt = prompts[prompt_index]
        original_answer = key_by_id[str(prompt["prompt_id"])]
        correct_id = str(prompt["candidates"][original_answer]["candidate_id"])
        order = sorted(
            range(len(prompt["candidates"])),
            key=lambda index: hashlib.sha256(
                (
                    f"v8-adversarial-candidate|{variant}|{prompt['prompt_id']}|"
                    f"{prompt['candidates'][index]['candidate_id']}"
                ).encode()
            ).hexdigest(),
        )
        candidates = [copy.deepcopy(prompt["candidates"][index]) for index in order]
        copied = copy.deepcopy(prompt)
        copied["candidates"] = candidates
        permuted_prompts.append(copied)
        permuted_keys.append(
            {
                "schema_version": "nursery-v8-sealed-key",
                "prompt_id": str(prompt["prompt_id"]),
                "answer_index": next(
                    index
                    for index, candidate in enumerate(candidates)
                    if str(candidate["candidate_id"]) == correct_id
                ),
            }
        )
    return permuted_prompts, permuted_keys


def function_subscript_keys(path: Path, function_name: str) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == function_name
    )
    keys = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript):
            value = child.slice
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                keys.add(value.value)
    return sorted(keys)


def probe(snapshot: Path, package: Path) -> dict[str, Any]:
    sys.path.insert(0, str(snapshot))
    from babyworld_lite.sensor_alignment_v8 import PROTOCOL_ID
    from babyworld_lite.sensor_alignment_v8.protocol import verify_file_manifest

    snapshot_manifest = read_json(snapshot / "snapshot_manifest.json")
    verification = verify_file_manifest(snapshot, snapshot_manifest)
    return {
        "status": (
            "PASS"
            if verification["status"] == "PASS"
            and PROTOCOL_ID == "synthetic-identifiability-qualification-v8"
            else "FAIL"
        ),
        "protocol_id": PROTOCOL_ID,
        "snapshot_verification": verification,
        "module_root": str(
            Path(sys.modules["babyworld_lite.sensor_alignment_v8"].__file__).resolve()
        ),
        "package_terminal_sha256": sha256_file(package / "terminal_decision.json"),
    }


def run_audit(repository: Path, package: Path, output_path: Path) -> dict[str, Any]:
    snapshot = package / "frozen_source_snapshot"
    inputs = package / "qualification_run" / "persisted_inputs"
    sys.path.insert(0, str(snapshot))

    from babyworld_lite.sensor_alignment_v8.adjudicator import terminal_decision
    from babyworld_lite.sensor_alignment_v8.benchmark import (
        audit_corpus,
        condition_audit,
    )
    from babyworld_lite.sensor_alignment_v8.detector_runtime import (
        DetectorRuntime,
        detector_capacity_audit,
    )
    from babyworld_lite.sensor_alignment_v8.learner import (
        fit_joint_cross_situational,
        predict_prompts,
        score_predictions,
    )
    from babyworld_lite.sensor_alignment_v8.protocol import (
        IdentifierFirewall,
        canonical_digest,
        load_config,
        reject_oracle_fields,
    )

    module_paths = {
        name: str(Path(module.__file__).resolve())
        for name, module in sorted(sys.modules.items())
        if name.startswith("babyworld_lite.sensor_alignment_v8")
        and getattr(module, "__file__", None)
    }
    frozen_imports_only = all(
        Path(path).is_relative_to(snapshot) for path in module_paths.values()
    )

    snapshot_manifest = read_json(snapshot / "snapshot_manifest.json")
    snapshot_check = verify_manifest(snapshot, snapshot_manifest)
    input_manifest = read_json(inputs / "input_manifest.json")
    input_check = verify_manifest(inputs, input_manifest)
    receipt = read_json(package / "freeze_receipt.json")
    frozen_artifact_problems = []
    for row in receipt["frozen_artifacts"]:
        path = package / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != str(row["sha256"])
        ):
            frozen_artifact_problems.append(str(row["path"]))
    immutable_inputs = {
        "status": (
            "PASS"
            if frozen_imports_only
            and snapshot_check["status"] == "PASS"
            and input_check["status"] == "PASS"
            and not frozen_artifact_problems
            else "FAIL"
        ),
        "frozen_imports_only": frozen_imports_only,
        "module_paths": module_paths,
        "snapshot_manifest": snapshot_check,
        "persisted_input_manifest": input_check,
        "frozen_artifact_problems": frozen_artifact_problems,
    }

    config = load_config(
        snapshot / "configs/synthetic_identifiability_qualification_v8.yaml",
        repository_root=snapshot,
    )
    persisted_config = read_json(inputs / "config.json")
    config_identity = canonical_digest(config) == canonical_digest(persisted_config)

    qualification = read_json(package / "qualification_run/qualification_summary.json")
    core = read_json(package / "qualification_run/recomputable/core_gates.json")
    model_results = read_json(package / "qualification_run/recomputable/model_results.json")
    null_capacity = read_json(
        package / "qualification_run/recomputable/null_capacity_audit.json"
    )
    official_order = read_json(
        package / "qualification_run/recomputable/order_invariance_audit.json"
    )
    official_model_index = {
        (
            int(row["corpus_seed"]),
            int(row["model_seed"]),
            str(row["condition"]),
        ): row
        for row in model_results
    }

    separated_rows = []
    for row in model_results:
        if row["condition"] not in {"absent_channel", "synchronized"}:
            continue
        for kind in ("primitive", "manner", "action", "noun"):
            for presence in ("present", "null"):
                metrics = row["metrics"]["by_kind_presence"][kind][presence]
                separated_rows.append(
                    {
                        "corpus_seed": int(row["corpus_seed"]),
                        "model_seed": int(row["model_seed"]),
                        "condition": str(row["condition"]),
                        "kind": kind,
                        "presence": presence,
                        "status": (
                            "PASS"
                            if subgroup_pass(metrics, presence, config["gates"])
                            else "FAIL"
                        ),
                        "fractional_accuracy": float(metrics["fractional_accuracy"]),
                        "mean_correct_probability": float(
                            metrics["mean_correct_probability"]
                        ),
                        "exact_chance": float(
                            metrics["mean_exact_candidate_set_chance"]
                        ),
                    }
                )
    action_null_rows = [
        row
        for row in separated_rows
        if row["kind"] == "action" and row["presence"] == "null"
    ]
    null_controls_pass = (
        null_capacity["status"] == "PASS"
        and all(
            float(drop)
            >= float(config["gates"]["minimum_null_ablation_fractional_drop"])
            for result in null_capacity["results"]
            for kind_values in result["null_fractional_drops_by_kind"].values()
            for drop in kind_values.values()
        )
    )
    subgroup_separation = {
        "status": (
            "PASS"
            if separated_rows
            and all(row["status"] == "PASS" for row in separated_rows)
            and null_controls_pass
            and all(
                row["fractional_accuracy"] > row["exact_chance"]
                and row["mean_correct_probability"]
                >= row["exact_chance"]
                + float(config["gates"]["minimum_null_probability_margin_over_exact_chance"])
                for row in action_null_rows
            )
            else "FAIL"
        ),
        "every_corpus_model_kind_presence_independent": True,
        "aggregation_used_to_rescue_subgroups": False,
        "rows_checked": len(separated_rows),
        "failed_rows": [
            row for row in separated_rows if row["status"] != "PASS"
        ],
        "minimum_action_null_fractional_accuracy": min(
            row["fractional_accuracy"] for row in action_null_rows
        ),
        "minimum_action_null_correct_probability": min(
            row["mean_correct_probability"] for row in action_null_rows
        ),
        "maximum_action_null_exact_chance": max(
            row["exact_chance"] for row in action_null_rows
        ),
        "null_capacity_controls": null_capacity["status"],
    }

    firewall = IdentifierFirewall(config, allowed_purposes=["excluded_smoke"])
    order_contexts = []
    tolerance = float(config["gates"]["order_metric_tolerance"])
    for corpus_dir in sorted(inputs.glob("corpus_*")):
        corpus_seed = int(corpus_dir.name.split("_", 1)[1])
        prompts = read_jsonl(corpus_dir / "evaluation_prompts.jsonl")
        keys = read_jsonl(corpus_dir / "evaluation_keys.jsonl")
        for model_seed in map(
            int, config["resolved_registries"]["excluded_smoke"]["model"]
        ):
            for condition in ("absent_channel", "synchronized"):
                rows = read_jsonl(corpus_dir / "conditions" / f"{condition}.jsonl")
                evidence_rows = read_jsonl(
                    corpus_dir / "evidence" / f"{condition}.jsonl"
                )
                evidence = {
                    str(row["episode_id"]): row for row in evidence_rows
                }
                model, trace = fit_joint_cross_situational(
                    rows,
                    evidence,
                    config,
                    firewall,
                    corpus_seed=corpus_seed,
                    model_seed=model_seed,
                    purpose="excluded_smoke",
                )
                predictions = predict_prompts(
                    model,
                    prompts,
                    config,
                    firewall,
                    corpus_seed=corpus_seed,
                    purpose="excluded_smoke",
                )
                metrics = score_predictions(
                    predictions,
                    keys,
                    config,
                    firewall,
                    corpus_seed=corpus_seed,
                    model_seed=model_seed,
                    purpose="excluded_smoke",
                )
                official = official_model_index[(corpus_seed, model_seed, condition)]
                official_match = canonical_digest(metrics) == canonical_digest(
                    official["metrics"]
                )
                candidate_variants = []
                candidate_digests = set()
                for variant in range(1, 9):
                    variant_prompts, variant_keys = permute_prompts_and_keys(
                        prompts, keys, variant
                    )
                    candidate_digests.add(canonical_digest(variant_prompts))
                    variant_predictions = predict_prompts(
                        model,
                        variant_prompts,
                        config,
                        firewall,
                        corpus_seed=corpus_seed,
                        purpose="excluded_smoke",
                    )
                    variant_metrics = score_predictions(
                        variant_predictions,
                        variant_keys,
                        config,
                        firewall,
                        corpus_seed=corpus_seed,
                        model_seed=model_seed,
                        purpose="excluded_smoke",
                    )
                    candidate_variants.append(
                        {
                            "variant": variant,
                            "exact_metrics_match": canonical_digest(variant_metrics)
                            == canonical_digest(metrics),
                            "maximum_numeric_delta": recursive_max_numeric_delta(
                                variant_metrics, metrics
                            ),
                        }
                    )
                training_variants = []
                alternative_orders = {
                    "reversed": list(reversed(rows)),
                    "hash_permuted": sorted(
                        rows,
                        key=lambda row: hashlib.sha256(
                            (
                                f"v8-adversarial-training|{corpus_seed}|{model_seed}|"
                                f"{condition}|{row['episode_id']}"
                            ).encode()
                        ).hexdigest(),
                    ),
                }
                for name, ordered_rows in alternative_orders.items():
                    altered_model, altered_trace = fit_joint_cross_situational(
                        ordered_rows,
                        evidence,
                        config,
                        firewall,
                        corpus_seed=corpus_seed,
                        model_seed=model_seed,
                        purpose="excluded_smoke",
                    )
                    altered_predictions = predict_prompts(
                        altered_model,
                        prompts,
                        config,
                        firewall,
                        corpus_seed=corpus_seed,
                        purpose="excluded_smoke",
                    )
                    altered_metrics = score_predictions(
                        altered_predictions,
                        keys,
                        config,
                        firewall,
                        corpus_seed=corpus_seed,
                        model_seed=model_seed,
                        purpose="excluded_smoke",
                    )
                    delta = recursive_max_numeric_delta(altered_metrics, metrics)
                    same_gate_decisions = all(
                        subgroup_pass(
                            altered_metrics["by_kind_presence"][kind][presence],
                            presence,
                            config["gates"],
                        )
                        == subgroup_pass(
                            metrics["by_kind_presence"][kind][presence],
                            presence,
                            config["gates"],
                        )
                        for kind in ("primitive", "manner", "action", "noun")
                        for presence in ("present", "null")
                    )
                    training_variants.append(
                        {
                            "order": name,
                            "model_digest_exact": altered_trace["model_digest"]
                            == trace["model_digest"],
                            "metrics_digest_exact": canonical_digest(altered_metrics)
                            == canonical_digest(metrics),
                            "maximum_numeric_delta": delta,
                            "within_frozen_tolerance": delta <= tolerance,
                            "same_subgroup_gate_decisions": same_gate_decisions,
                        }
                    )
                context_pass = (
                    official_match
                    and len(candidate_digests) >= 4
                    and all(
                        row["exact_metrics_match"] for row in candidate_variants
                    )
                    and all(
                        row["within_frozen_tolerance"]
                        and row["same_subgroup_gate_decisions"]
                        for row in training_variants
                    )
                )
                order_contexts.append(
                    {
                        "corpus_seed": corpus_seed,
                        "model_seed": model_seed,
                        "condition": condition,
                        "status": "PASS" if context_pass else "FAIL",
                        "refit_matches_official_metrics": official_match,
                        "refit_model_digest_matches_official": trace["model_digest"]
                        == official["model_digest"],
                        "distinct_candidate_permutations": len(candidate_digests),
                        "candidate_variants": candidate_variants,
                        "training_variants": training_variants,
                    }
                )
    observed_adversarial_references = {
        role: sorted(
            {
                int(reference["value"])
                for operation in firewall.log
                for reference in operation["references"]
                if reference["role"] == role
            }
        )
        for role in ("corpus", "model", "inference")
    }
    order_invariance = {
        "status": (
            "PASS"
            if official_order["status"] == "PASS"
            and order_contexts
            and all(row["status"] == "PASS" for row in order_contexts)
            else "FAIL"
        ),
        "official_all_condition_reversal_audit": official_order["status"],
        "contexts": order_contexts,
        "adversarial_operation_count": len(firewall.log),
        "adversarial_references_by_role": observed_adversarial_references,
    }

    forbidden_visible = {
        "answer_index",
        "correct_candidate_id",
        "event_concepts",
        "event_owners",
        "grounded",
        "intended_concept",
        "lexicon_oracle",
        "oracle",
        "target_event_index",
    }
    visible_failures = []
    prompt_failures = []
    for corpus_dir in sorted(inputs.glob("corpus_*")):
        for row in read_jsonl(corpus_dir / "visible_episodes.jsonl"):
            try:
                reject_oracle_fields(row)
            except ValueError as error:
                visible_failures.append(str(error))
        for row in read_jsonl(corpus_dir / "evaluation_prompts.jsonl"):
            try:
                reject_oracle_fields(row)
            except ValueError as error:
                prompt_failures.append(str(error))
            if "raw_stream" in row:
                prompt_failures.append(f"raw_stream:{row['prompt_id']}")
    learner_path = snapshot / "babyworld_lite/sensor_alignment_v8/learner.py"
    fit_keys = function_subscript_keys(learner_path, "fit_joint_cross_situational")
    predict_keys = function_subscript_keys(learner_path, "predict_prompts")
    forbidden_fit_keys = sorted(forbidden_visible & set(fit_keys))
    forbidden_predict_keys = sorted(forbidden_visible & set(predict_keys))
    trace_flag_failures = [
        {
            "corpus_seed": row["corpus_seed"],
            "model_seed": row["model_seed"],
            "condition": row["condition"],
        }
        for row in model_results
        if row["training_trace"]["oracle_fields_consumed"]
        or row["training_trace"]["evaluation_keys_consumed"]
        or row["training_trace"]["post_fit_projection_applied"]
        or row["training_trace"]["condition_specific_parameter_replacement"]
    ]
    leakage = {
        "status": (
            "PASS"
            if not visible_failures
            and not prompt_failures
            and not forbidden_fit_keys
            and not forbidden_predict_keys
            and not trace_flag_failures
            else "FAIL"
        ),
        "visible_ledger_failures": visible_failures,
        "prompt_failures": prompt_failures,
        "fit_subscript_keys": fit_keys,
        "predict_subscript_keys": predict_keys,
        "forbidden_fit_keys": forbidden_fit_keys,
        "forbidden_predict_keys": forbidden_predict_keys,
        "training_trace_flag_failures": trace_flag_failures,
        "keys_loaded_only_for_post_prediction_scoring_in_adversarial_audit": True,
    }

    combined_evidence: dict[str, dict[str, Any]] = {}
    combined_oracle = []
    factor_rows = []
    condition_rows = []
    construction_digests = []
    for corpus_dir in sorted(inputs.glob("corpus_*")):
        corpus_seed = int(corpus_dir.name.split("_", 1)[1])
        visible = read_jsonl(corpus_dir / "visible_episodes.jsonl")
        oracle = read_jsonl(corpus_dir / "oracle_episodes.jsonl")
        persisted_audit = read_json(corpus_dir / "corpus_audit.json")
        recomputed_audit = audit_corpus(visible, oracle, config)
        base_fields_match = all(
            canonical_digest(persisted_audit.get(key)) == canonical_digest(value)
            for key, value in recomputed_audit.items()
        )
        construction = persisted_audit["factor_assignment_construction"]
        construction_digests.append(canonical_digest(construction))
        factor_rows.append(
            {
                "corpus_seed": corpus_seed,
                "status": (
                    "PASS"
                    if recomputed_audit["status"] == "PASS"
                    and base_fields_match
                    and construction["status"] == "PASS"
                    and float(
                        construction["maximum_exact_persisted_audit_cramers_v"]
                    )
                    <= float(construction["bound"])
                    else "FAIL"
                ),
                "base_audit_fields_exact": base_fields_match,
                "maximum_factor_control_cramers_v": float(
                    recomputed_audit["maximum_factor_control_cramers_v"]
                ),
                "maximum_pairwise_cramers_v": float(
                    recomputed_audit["maximum_pairwise_cramers_v"]
                ),
                "maximum_noun_target_owner_gap": float(
                    recomputed_audit["maximum_conditional_noun_target_owner_gap"]
                ),
                "construction_digest": canonical_digest(construction),
            }
        )
        conditioned = {
            condition: read_jsonl(
                corpus_dir / "conditions" / f"{condition}.jsonl"
            )
            for condition in config["design"]["conditions"]
        }
        donors = {
            condition: read_json(
                corpus_dir / "donor_maps" / f"{condition}.json"
            )
            for condition in config["design"]["conditions"]
        }
        condition_result = condition_audit(visible, conditioned, donors, config)
        condition_rows.append(
            {"corpus_seed": corpus_seed, **condition_result}
        )
        combined_oracle.extend(oracle)
        combined_evidence.update(
            {
                str(row["episode_id"]): row
                for row in read_jsonl(
                    corpus_dir / "evidence" / "synchronized.jsonl"
                )
            }
        )
    factor_refit_audit = read_json(
        package / "qualification_run/recomputable/factor_audit.json"
    )
    factors = {
        "status": (
            "PASS"
            if factor_rows
            and all(row["status"] == "PASS" for row in factor_rows)
            and len(set(construction_digests)) == 1
            and all(row["status"] == "PASS" for row in condition_rows)
            and factor_refit_audit["status"] == "PASS"
            and all(
                value["expected_corpus_by_level_cells"]
                == value["observed_corpus_by_level_cells"]
                and value["independent_membership_recomputation"]
                for value in factor_refit_audit["factors"].values()
            )
            else "FAIL"
        ),
        "corpus_audits": factor_rows,
        "seed_invariant_construction_digest": (
            construction_digests[0]
            if len(set(construction_digests)) == 1
            else None
        ),
        "condition_audits": condition_rows,
        "factor_refit_status": factor_refit_audit["status"],
    }

    recomputed_capacity = detector_capacity_audit(
        combined_evidence, combined_oracle, config
    )
    official_capacity = read_json(
        package / "qualification_run/detector_capacity_audit.json"
    )
    sample_corpus = sorted(inputs.glob("corpus_*"))[0]
    sample_row = read_jsonl(
        sample_corpus / "conditions" / "synchronized.jsonl"
    )[0]
    persisted_sample = read_jsonl(
        sample_corpus / "evidence" / "synchronized.jsonl"
    )[0]
    runtime = DetectorRuntime.load(
        snapshot / str(config["detector"]["frozen_v2_path"]),
        str(config["detector"]["frozen_v2_sha256"]),
    )
    inferred_sample = runtime.infer_episode(sample_row)
    persisted_sample_without_id = {
        key: value
        for key, value in persisted_sample.items()
        if key != "episode_id"
    }
    runtime_audit = read_json(
        package / "qualification_run/detector_runtime_audit.json"
    )
    detector = {
        "status": (
            "PASS"
            if recomputed_capacity["status"] == "PASS"
            and canonical_digest(recomputed_capacity)
            == canonical_digest(official_capacity)
            and canonical_digest(inferred_sample)
            == canonical_digest(persisted_sample_without_id)
            and runtime_audit["status"] == "PASS"
            and not runtime_audit["hash_only_placeholder"]
            else "FAIL"
        ),
        "capacity_recomputation_exact": canonical_digest(recomputed_capacity)
        == canonical_digest(official_capacity),
        "sample_runtime_evidence_exact": canonical_digest(inferred_sample)
        == canonical_digest(persisted_sample_without_id),
        "overall_fractional_accuracy": float(
            recomputed_capacity[
                "overall_fractional_top_event_or_null_accuracy"
            ]
        ),
        "strata": recomputed_capacity["strata"],
        "runtime_class": runtime_audit["implementation_class"],
        "runtime_function": runtime_audit["implementation_function"],
        "runtime_inference_calls": runtime_audit["inference_calls"],
    }

    resolved = config["resolved_registries"]
    named_sets = {
        name: {
            int(value)
            for values in resolved[name].values()
            for value in values
        }
        for name in (
            "fixture_only",
            "excluded_smoke",
            "future_development",
            "confirmation_reserve",
        )
    }
    named_sets["prior_v1_v7"] = set(map(int, resolved["prior_v1_v7"]))
    overlaps = {
        f"{left}|{right}": sorted(named_sets[left] & named_sets[right])
        for index, left in enumerate(named_sets)
        for right in list(named_sets)[index + 1 :]
        if named_sets[left] & named_sets[right]
    }
    official_identifier = read_json(
        package / "qualification_run/identifier_reference_audit.json"
    )
    official_operations = read_json(
        package / "qualification_run/seed_operation_log.json"
    )["operations"]
    forbidden_ids = (
        named_sets["future_development"]
        | named_sets["confirmation_reserve"]
        | named_sets["prior_v1_v7"]
    )
    forbidden_operation_references = sorted(
        {
            int(reference["value"])
            for operation in official_operations
            for reference in operation["references"]
            if int(reference["value"]) in forbidden_ids
        }
    )
    declared_smoke = {
        role: sorted(map(int, resolved["excluded_smoke"][role]))
        for role in ("corpus", "model", "inference")
    }
    seed_firewall = {
        "status": (
            "PASS"
            if not overlaps
            and not forbidden_operation_references
            and official_identifier["status"] == "PASS"
            and official_identifier["declared_references_by_role"]
            == official_identifier["observed_guarded_references_by_role"]
            == declared_smoke
            and observed_adversarial_references == declared_smoke
            else "FAIL"
        ),
        "overlaps": overlaps,
        "forbidden_operation_references": forbidden_operation_references,
        "official_declared_and_observed": official_identifier[
            "observed_guarded_references_by_role"
        ],
        "adversarial_observed": observed_adversarial_references,
        "future_development_identifiers_touched": [],
        "confirmation_identifiers_touched": [],
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
    }

    adjudication_inputs = read_json(package / "adjudication_inputs.json")
    terminal = read_json(package / "terminal_decision.json")
    regenerated_terminal = terminal_decision(adjudication_inputs)
    official_digest_problems = []
    digest_paths = {
        "qualification": "qualification_run/qualification_summary.json",
        "tests": "official_test_execution_report.json",
        "recompute": "independent_recompute_comparison.json",
        "negative_recompute": "negative_recompute_tests.json",
        "preservation": "preservation_proof.json",
        "freeze_receipt": "freeze_receipt.json",
    }
    for name, relative in digest_paths.items():
        observed = sha256_file(package / relative)
        expected = str(adjudication_inputs["official_artifact_digests"][name])
        if observed != expected:
            official_digest_problems.append(
                {"artifact": name, "expected": expected, "observed": observed}
            )
    contract = read_json(package / "frozen_adjudication_contract.json")
    provenance = {
        "status": (
            "PASS"
            if not official_digest_problems
            and canonical_digest(regenerated_terminal) == canonical_digest(terminal)
            and terminal["decision"] == "VERSION_READY"
            and not terminal["development_authorized"]
            and not terminal["confirmation_authorized"]
            and all(
                (package / relative).is_file()
                for relative in contract["official_paths"].values()
            )
            and read_json(package / "terminal_regeneration.json")["status"] == "PASS"
            else "FAIL"
        ),
        "official_digest_problems": official_digest_problems,
        "frozen_adjudicator_regeneration_exact": canonical_digest(
            regenerated_terminal
        )
        == canonical_digest(terminal),
        "terminal_decision": terminal["decision"],
        "development_authorized": terminal["development_authorized"],
        "confirmation_authorized": terminal["confirmation_authorized"],
        "contract": contract,
    }

    complete_manifest = read_json(package / "complete_file_manifest.json")
    manifest_check = verify_manifest(package, complete_manifest)
    listed_paths = {str(row["path"]) for row in complete_manifest["files"]}
    actual_paths = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file()
        and path.name != "complete_file_manifest.json"
    }
    manifest_unexpected = sorted(actual_paths - listed_paths)
    manifest_missing = sorted(listed_paths - actual_paths)
    manifest = {
        "status": (
            "PASS"
            if manifest_check["status"] == "PASS"
            and not manifest_unexpected
            and not manifest_missing
            and complete_manifest["independent_second_pass_verification"][
                "status"
            ]
            == "PASS"
            else "FAIL"
        ),
        "verification": manifest_check,
        "actual_file_count_excluding_manifest": len(actual_paths),
        "listed_file_count": len(listed_paths),
        "unexpected": manifest_unexpected,
        "missing": manifest_missing,
        "self_excluded_by_definition": complete_manifest[
            "self_excluded_by_definition"
        ],
    }

    frozen_environment = read_json(package / "frozen_environment.json")
    python_path = Path(str(frozen_environment["python_executable"]))
    probe_command = [
        str(python_path),
        str(Path(__file__).resolve()),
        "--probe",
        "--repository",
        str(repository),
        "--package",
        str(package),
    ]
    probe_environment = dict(os.environ)
    probe_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    probe_environment["PYTHONPATH"] = str(snapshot)
    completed = subprocess.run(
        probe_command,
        cwd=output_path.parent,
        env=probe_environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    try:
        probe_result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        probe_result = {"status": "FAIL", "raw_output": completed.stdout}
    environment = {
        "status": (
            "PASS"
            if python_path.is_file()
            and sha256_file(python_path) == frozen_environment["python_sha256"]
            and completed.returncode == 0
            and probe_result.get("status") == "PASS"
            and read_json(package / "independent_recompute_comparison.json")[
                "status"
            ]
            == "PASS"
            else "FAIL"
        ),
        "python_executable": str(python_path),
        "python_sha256_matches": python_path.is_file()
        and sha256_file(python_path) == frozen_environment["python_sha256"],
        "separate_process_exit_code": completed.returncode,
        "separate_process_probe": probe_result,
        "independent_recompute_status": read_json(
            package / "independent_recompute_comparison.json"
        )["status"],
        "negative_recompute_status": read_json(
            package / "negative_recompute_tests.json"
        )["status"],
    }

    checks = {
        "immutable_frozen_inputs": immutable_inputs,
        "config_identity": {
            "status": "PASS" if config_identity else "FAIL",
            "frozen_and_persisted_config_identical": config_identity,
        },
        "order_invariance": order_invariance,
        "separated_null_present_validity": subgroup_separation,
        "leakage_and_non_oracularity": leakage,
        "detector_bounds_and_runtime": detector,
        "factor_balance_and_conditions": factors,
        "seed_firewalls": seed_firewall,
        "adjudication_provenance": provenance,
        "environment_reproduction": environment,
        "manifest_completeness": manifest,
    }
    failures = [
        name for name, value in checks.items() if value["status"] != "PASS"
    ]
    result = {
        "schema_version": "nursery-v8-adversarial-launch-audit-v1",
        "protocol_id": "synthetic-identifiability-qualification-v8",
        "audit_scope": "frozen snapshot and persisted excluded-smoke evidence only",
        "status": "PASS" if not failures else "FAIL",
        "decision": "SURVIVES_ADVERSARIAL_AUDIT" if not failures else "REVISE",
        "failed_checks": failures,
        "checks": checks,
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "v8_terminal_sha256": sha256_file(package / "terminal_decision.json"),
        "v8_complete_manifest_sha256": sha256_file(
            package / "complete_file_manifest.json"
        ),
        "v8_qualification_sha256": sha256_file(
            package / "qualification_run/qualification_summary.json"
        ),
        "identifiers_touched_by_role": observed_adversarial_references,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "confirmation_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.write_bytes(canonical_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--output")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    repository = Path(args.repository).resolve()
    package = Path(args.package).resolve()
    snapshot = package / "frozen_source_snapshot"
    if args.probe:
        result = probe(snapshot, package)
        sys.stdout.buffer.write(canonical_bytes(result))
        raise SystemExit(0 if result["status"] == "PASS" else 1)
    if not args.output:
        raise SystemExit("--output is required outside --probe mode")
    result = run_audit(
        repository,
        package,
        Path(args.output).resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "failed_checks": result["failed_checks"],
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
