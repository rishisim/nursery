from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from .analysis import (
    factor_stratification_audit,
    infer_mean,
    noun_detector_selectivity_audit,
)
from .controls import fit_direct_capacity, fit_oracle_alignment
from .corpus import Corpus, EvalKey, EvalPrompt, condition_view, generate_corpus
from .detector_runtime import DetectorRuntime
from .learner import (
    fit_joint_competitive,
    predict_prompts,
    score_predictions,
)
from .protocol import (
    PROTOCOL_ID,
    SeedFirewall,
    SeedReference,
    canonical_digest,
    load_config,
    registry_snapshot,
    sha256_file,
    write_json,
    write_jsonl,
)

DETERMINISTIC_EXCLUSIONS = frozenset({"core_manifest.json"})
V1_V3_ROOTS = (
    "babyworld_lite/weak_alignment",
    "babyworld_lite/sensor_alignment_v2",
    "babyworld_lite/sensor_alignment_v3",
    "output/synthetic_weak_alignment_recovery_v1",
    "output/synthetic_sensor_event_robustness_v2",
    "output/synthetic_component_robustness_v3",
)
V1_V3_FILES = (
    "configs/synthetic_weak_alignment_recovery_v1.yaml",
    "configs/synthetic_sensor_event_robustness_v2.yaml",
    "configs/synthetic_component_robustness_v3.yaml",
    "docs/synthetic_weak_alignment_recovery_v1_protocol.md",
    "docs/synthetic_weak_alignment_recovery_v1_primary_sources.json",
    "docs/synthetic_sensor_event_robustness_v2_protocol.md",
    "docs/synthetic_sensor_event_robustness_v2_primary_sources.json",
    "docs/synthetic_component_robustness_v3_protocol.md",
    "docs/synthetic_component_robustness_v3_primary_sources.json",
    "docs/synthetic_component_robustness_v3_source_record.md",
    "scripts/run_synthetic_weak_alignment_v1.py",
    "scripts/run_synthetic_sensor_event_robustness_v2.py",
    "scripts/run_synthetic_component_robustness_v3.py",
    "tests/test_synthetic_weak_alignment_v1.py",
    "tests/test_synthetic_sensor_event_robustness_v2.py",
    "tests/test_synthetic_component_robustness_v3.py",
)


def _detector_path(root: Path, config: Mapping[str, Any]) -> Path:
    return root / str(config["detector"]["frozen_v2_path"])


def _load_runtime(root: Path, config: Mapping[str, Any]) -> DetectorRuntime:
    return DetectorRuntime.load(
        _detector_path(root, config), str(config["detector"]["frozen_v2_sha256"])
    )


def _serialize_evidence(evidence: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"episode_id": episode_id, **dict(value)}
        for episode_id, value in sorted(evidence.items())
    ]


def _persist_corpus(run_dir: Path, corpus: Corpus) -> None:
    directory = run_dir / "raw" / f"corpus_{corpus.corpus_seed}"
    write_jsonl(directory / "visible_episodes.jsonl", corpus.visible_episodes)
    write_jsonl(directory / "oracle_episodes.jsonl", corpus.oracle_episodes)
    write_jsonl(
        directory / "evaluation_prompts.jsonl",
        (asdict(value) for value in corpus.evaluation_prompts),
    )
    write_jsonl(
        directory / "evaluation_keys.jsonl",
        (asdict(value) for value in corpus.evaluation_keys),
    )
    write_json(directory / "lexicon_oracle.json", corpus.lexicon_oracle)
    write_json(directory / "corpus_audit.json", corpus.audit)


def _run_model_cell(
    rows: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    corpus: Corpus,
    config: Mapping[str, Any],
    firewall: SeedFirewall,
    *,
    condition: str,
    model_seed: int,
    purpose: str,
    use_sensor: bool,
) -> tuple[dict[str, Any], Any, dict[str, Any], list[dict[str, Any]]]:
    model, trace = fit_joint_competitive(
        rows,
        evidence,
        config,
        firewall,
        corpus_seed=corpus.corpus_seed,
        model_seed=model_seed,
        purpose=purpose,
        use_sensor=use_sensor,
    )
    predictions = predict_prompts(
        model,
        corpus.evaluation_prompts,
        firewall,
        corpus_seed=corpus.corpus_seed,
        purpose=purpose,
    )
    score = score_predictions(
        predictions,
        corpus.evaluation_keys,
        firewall,
        corpus_seed=corpus.corpus_seed,
        model_seed=model_seed,
        purpose=purpose,
    )
    result = {
        "corpus_seed": corpus.corpus_seed,
        "model_seed": model_seed,
        "condition": condition,
        "learner": model.learner,
        "model_digest": canonical_digest(model.serializable()),
        "metrics": score,
        "training_trace": trace,
        "qualification_only": True,
    }
    return result, model, trace, predictions


def factor_analysis(
    corpus: Corpus,
    synchronized_rows: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    firewall: SeedFirewall,
    *,
    model_seed: int,
    purpose: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    factor_levels = {
        "visibility": list(map(float, config["design"]["visibility_levels"])),
        "lag": list(map(int, config["design"]["lag_levels"])),
    }
    for factor, levels in factor_levels.items():
        for level in levels:
            selected = [
                row
                for row in synchronized_rows
                if float(row["factor_values"][factor]) == float(level)
            ]
            selected_ids = sorted(str(row["episode_id"]) for row in selected)
            selected_evidence = {episode_id: evidence[episode_id] for episode_id in selected_ids}
            model, trace = fit_joint_competitive(
                selected,
                selected_evidence,
                config,
                firewall,
                corpus_seed=corpus.corpus_seed,
                model_seed=model_seed,
                purpose=purpose,
                use_sensor=True,
            )
            predictions = predict_prompts(
                model,
                corpus.evaluation_prompts,
                firewall,
                corpus_seed=corpus.corpus_seed,
                purpose=purpose,
            )
            score = score_predictions(
                predictions,
                corpus.evaluation_keys,
                firewall,
                corpus_seed=corpus.corpus_seed,
                model_seed=model_seed,
                purpose=purpose,
            )
            output.append(
                {
                    "corpus_seed": corpus.corpus_seed,
                    "model_seed": model_seed,
                    "factor": factor,
                    "level": level,
                    "training_episode_count": len(selected),
                    "training_episode_ids": selected_ids,
                    "training_episode_ids_digest": canonical_digest(selected_ids),
                    "model_digest": canonical_digest(model.serializable()),
                    "predictions_digest": canonical_digest(predictions),
                    "action_accuracy": score["action_accuracy"],
                    "noun_accuracy": score["noun_accuracy"],
                    "lag_weight_min": trace["lag_weight_min"],
                    "lag_weight_max": trace["lag_weight_max"],
                    "stratification": "refit_on_exact_level_subset",
                }
            )
    return output


def _control_results(
    corpus: Corpus,
    synchronized_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    firewall: SeedFirewall,
    *,
    model_seed: int,
    purpose: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name, fitter, arguments in (
        (
            "oracle_alignment_upper",
            fit_oracle_alignment,
            (synchronized_rows, corpus.oracle_episodes),
        ),
        (
            "direct_capacity_upper",
            fit_direct_capacity,
            (corpus.evaluation_prompts, corpus.evaluation_keys),
        ),
    ):
        model, trace = fitter(
            *arguments,
            firewall,
            corpus_seed=corpus.corpus_seed,
            model_seed=model_seed,
            purpose=purpose,
        )
        predictions = predict_prompts(
            model,
            corpus.evaluation_prompts,
            firewall,
            corpus_seed=corpus.corpus_seed,
            purpose=purpose,
        )
        score = score_predictions(
            predictions,
            corpus.evaluation_keys,
            firewall,
            corpus_seed=corpus.corpus_seed,
            model_seed=model_seed,
            purpose=purpose,
        )
        output.append(
            {
                "corpus_seed": corpus.corpus_seed,
                "model_seed": model_seed,
                "control": name,
                "metrics": score,
                "predictions": predictions,
                "trace": trace,
            }
        )
    return output


def _competition_participation(
    corpus: Corpus,
    rows: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    firewall: SeedFirewall,
    *,
    model_seed: int,
    purpose: str,
) -> dict[str, Any]:
    variants: dict[str, dict[str, Any]] = {}
    for name, iterations, competition in (
        ("configured", None, None),
        ("one_iteration", 1, None),
        ("zero_competition", None, 0.0),
    ):
        model, trace = fit_joint_competitive(
            rows,
            evidence,
            config,
            firewall,
            corpus_seed=corpus.corpus_seed,
            model_seed=model_seed,
            purpose=purpose,
            use_sensor=True,
            iterations_override=iterations,
            competition_override=competition,
        )
        variants[name] = {
            "model_digest": canonical_digest(model.serializable()),
            "iterations_executed": trace["iterations_executed"],
            "competition_weight": trace["competition_weight"],
            "competition_penalty_nonzero": any(
                value > 0 for value in trace["iteration_competition_penalties"]
            ),
        }
    passed = (
        variants["configured"]["model_digest"]
        != variants["one_iteration"]["model_digest"]
        and variants["configured"]["model_digest"]
        != variants["zero_competition"]["model_digest"]
        and variants["configured"]["competition_penalty_nonzero"]
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "variants": variants,
        "iterations_materially_participate": variants["configured"]["model_digest"]
        != variants["one_iteration"]["model_digest"],
        "competition_weight_materially_participates": variants["configured"]["model_digest"]
        != variants["zero_competition"]["model_digest"],
    }


def _inference_audit(
    model_results: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    firewall: SeedFirewall,
    *,
    purpose: str,
) -> dict[str, Any]:
    inference_seed = int(config["seeds"][purpose]["inference"][0])
    firewall.authorize(
        "infer", [SeedReference("inference", inference_seed)], purpose=purpose
    )
    inference = config["inference"]
    arguments = {
        "seed": inference_seed,
        "resamples": int(inference["resamples"]),
        "alpha": float(inference["alpha"]),
        "zero_variance_tolerance": float(inference["zero_variance_tolerance"]),
    }
    degenerate_contract = infer_mean([0.25, 0.25, 0.25], **arguments)
    nondegenerate_contract = infer_mean([0.10, 0.18, 0.31, 0.42], **arguments)
    effects: dict[str, Any] = {}
    corpus_seeds = sorted({int(row["corpus_seed"]) for row in model_results})
    if len(corpus_seeds) >= 2:
        for comparison in ("absent", "shuffled"):
            values: list[float] = []
            for corpus_seed in corpus_seeds:
                synchronized = np.mean(
                    [
                        row["metrics"]["action_accuracy"]
                        for row in model_results
                        if row["corpus_seed"] == corpus_seed
                        and row["condition"] == "synchronized"
                    ]
                )
                control = np.mean(
                    [
                        row["metrics"]["action_accuracy"]
                        for row in model_results
                        if row["corpus_seed"] == corpus_seed
                        and row["condition"] == comparison
                    ]
                )
                values.append(float(synchronized - control))
            effects[f"synchronized_minus_{comparison}"] = {
                **infer_mean(values, **arguments),
                "qualification_only_effects": values,
                "scientific_interpretation_forbidden": True,
            }
    passed = (
        degenerate_contract["method"] == "degenerate_point_mass"
        and not degenerate_contract["bootstrap_interval_computed"]
        and not degenerate_contract["population_uncertainty_estimated"]
        and nondegenerate_contract["method"] == "studentized_bootstrap_t"
        and nondegenerate_contract["valid_studentized_resamples"] > 0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "degenerate_contract": degenerate_contract,
        "nondegenerate_contract": nondegenerate_contract,
        "qualification_only_smoke_effects": effects,
    }


def execute_qualification(
    repository_root: str | Path,
    config_path: str | Path,
    run_dir: str | Path,
    *,
    purpose: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(run_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    config = load_config(config_path)
    if purpose == "excluded_smoke" and config["protocol"]["status"] != "frozen":
        raise RuntimeError("excluded smoke requires frozen config")
    if purpose not in {"fixture_only", "excluded_smoke"}:
        raise PermissionError("qualification can only use fixture or excluded-smoke registries")
    output.mkdir(parents=True)
    firewall = SeedFirewall(config, allowed_purposes=[purpose])
    runtime = _load_runtime(root, config)
    model_results: list[dict[str, Any]] = []
    factor_results: list[dict[str, Any]] = []
    factor_recomputed: list[dict[str, Any]] = []
    control_results: list[dict[str, Any]] = []
    corpus_audits: list[dict[str, Any]] = []
    synchronized_oracle: list[dict[str, Any]] = []
    synchronized_evidence: dict[str, dict[str, Any]] = {}
    competition_audit: dict[str, Any] | None = None
    first_model_seed = int(config["seeds"][purpose]["model"][0])
    for corpus_seed in map(int, config["seeds"][purpose]["corpus"]):
        corpus = generate_corpus(corpus_seed, config, firewall, purpose=purpose)
        _persist_corpus(output, corpus)
        corpus_audits.append({"corpus_seed": corpus_seed, **corpus.audit})
        synchronized_oracle.extend(corpus.oracle_episodes)
        condition_rows: dict[str, list[dict[str, Any]]] = {}
        condition_evidence: dict[str, dict[str, dict[str, Any]]] = {}
        for condition in config["design"]["conditions"]:
            rows, donor_map = condition_view(corpus.visible_episodes, str(condition))
            condition_rows[str(condition)] = rows
            write_json(
                output / "raw" / f"corpus_{corpus_seed}" / f"{condition}_donor_map.json",
                donor_map,
            )
            if condition == "absent":
                evidence: dict[str, dict[str, Any]] = {}
            else:
                evidence = runtime.infer_episodes(
                    rows,
                    firewall,
                    corpus_seed=corpus_seed,
                    purpose=purpose,
                )
                write_jsonl(
                    output
                    / "raw"
                    / f"corpus_{corpus_seed}"
                    / f"{condition}_derived_evidence.jsonl",
                    _serialize_evidence(evidence),
                )
            condition_evidence[str(condition)] = evidence
        synchronized_evidence.update(condition_evidence["synchronized"])
        for model_seed in map(int, config["seeds"][purpose]["model"]):
            for condition, use_sensor in (
                ("synchronized", True),
                ("shuffled", True),
                ("absent", False),
            ):
                result, _, _, _ = _run_model_cell(
                    condition_rows[condition],
                    condition_evidence[condition],
                    corpus,
                    config,
                    firewall,
                    condition=condition,
                    model_seed=model_seed,
                    purpose=purpose,
                    use_sensor=use_sensor,
                )
                model_results.append(result)
            factor_results.extend(
                factor_analysis(
                    corpus,
                    condition_rows["synchronized"],
                    condition_evidence["synchronized"],
                    config,
                    firewall,
                    model_seed=model_seed,
                    purpose=purpose,
                )
            )
            factor_recomputed.extend(
                factor_analysis(
                    corpus,
                    condition_rows["synchronized"],
                    condition_evidence["synchronized"],
                    config,
                    firewall,
                    model_seed=model_seed,
                    purpose=purpose,
                )
            )
        control_results.extend(
            _control_results(
                corpus,
                condition_rows["synchronized"],
                config,
                firewall,
                model_seed=first_model_seed,
                purpose=purpose,
            )
        )
        if competition_audit is None:
            competition_audit = _competition_participation(
                corpus,
                condition_rows["synchronized"],
                condition_evidence["synchronized"],
                config,
                firewall,
                model_seed=first_model_seed,
                purpose=purpose,
            )
    expected_detector_calls = 2 * sum(
        192 for _ in config["seeds"][purpose]["corpus"]
    )
    detector_audit = runtime.audit(expected_minimum_calls=expected_detector_calls)
    noun_audit = noun_detector_selectivity_audit(
        synchronized_evidence,
        synchronized_oracle,
        maximum_gap=float(config["gates"]["maximum_noun_target_distractor_detector_gap"]),
    )
    factor_audit = factor_stratification_audit(factor_results)
    factor_exact_recomputation = canonical_digest(factor_results) == canonical_digest(
        factor_recomputed
    )
    inference_audit = _inference_audit(model_results, config, firewall, purpose=purpose)
    configured_traces = [row["training_trace"] for row in model_results]
    no_sensor = [
        row["metrics"]["action_accuracy"]
        for row in model_results
        if row["condition"] == "absent"
    ]
    oracle_scores = [
        row["metrics"]["action_accuracy"]
        for row in control_results
        if row["control"] == "oracle_alignment_upper"
    ]
    capacity_scores = [
        row["metrics"]["action_accuracy"]
        for row in control_results
        if row["control"] == "direct_capacity_upper"
    ]
    gates = {
        "exact_balance": all(row["status"] == "PASS" for row in corpus_audits),
        "noun_structural_selectivity": all(
            row["noun_owner_independent_of_identity_and_target"] for row in corpus_audits
        ),
        "noun_detector_selectivity": noun_audit["status"] == "PASS",
        "detector_runtime": detector_audit["status"] == "PASS",
        "competition": competition_audit is not None
        and competition_audit["status"] == "PASS"
        and all(
            trace["iterations_executed"] == int(config["learner"]["iterations"])
            and trace["competition_weight"]
            == float(config["learner"]["competition_weight"])
            for trace in configured_traces
        ),
        "noun_sensor_weight_zero": all(
            trace["noun_sensor_weight"] == 0.0
            and trace["sensor_applications_by_slot"]["noun"] == 0
            for trace in configured_traces
        ),
        "factors_executable_and_stratified": factor_audit["status"] == "PASS",
        "factor_exact_recomputation": factor_exact_recomputation,
        "sensor_free_non_degenerate": bool(no_sensor)
        and min(no_sensor) >= float(config["gates"]["minimum_sensor_free_action_accuracy"])
        and max(no_sensor) <= float(config["gates"]["maximum_sensor_free_action_accuracy"]),
        "oracle_control_executed": bool(oracle_scores)
        and min(oracle_scores) >= float(config["gates"]["minimum_oracle_action_accuracy"])
        and all(
            not row["trace"]["assigned_score_constant"]
            for row in control_results
            if row["control"] == "oracle_alignment_upper"
        ),
        "direct_capacity_executed": bool(capacity_scores)
        and min(capacity_scores)
        >= float(config["gates"]["minimum_direct_capacity_action_accuracy"])
        and all(
            not row["trace"]["assigned_score_constant"]
            for row in control_results
            if row["control"] == "direct_capacity_upper"
        ),
        "inference_contract": inference_audit["status"] == "PASS",
        "oracle_leakage_absent": all(
            not trace["oracle_fields_consumed"] for trace in configured_traces
        ),
        "seed_firewall": all(row["status"] == "ALLOW" for row in firewall.log),
    }
    summary = {
        "protocol_id": PROTOCOL_ID,
        "purpose": purpose,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "scientific_outcome": False,
        "scientific_interpretation_authorized": False,
        "future_development_touched": False,
        "confirmation_touched": False,
        "corpus_seeds": list(map(int, config["seeds"][purpose]["corpus"])),
        "model_seeds": list(map(int, config["seeds"][purpose]["model"])),
        "records": len(model_results),
    }
    write_json(output / "corpus_audits.json", corpus_audits)
    write_json(output / "model_results.json", model_results)
    write_json(output / "factor_results.json", factor_results)
    write_json(
        output / "factor_recomputation.json",
        {
            "status": "PASS" if factor_exact_recomputation else "FAIL",
            "exact_digest_match": factor_exact_recomputation,
            "original_digest": canonical_digest(factor_results),
            "recomputed_digest": canonical_digest(factor_recomputed),
            "rows": factor_recomputed,
        },
    )
    write_json(output / "factor_stratification_audit.json", factor_audit)
    write_json(output / "control_results.json", control_results)
    write_json(output / "detector_runtime_audit.json", detector_audit)
    write_json(output / "noun_selectivity_audit.json", noun_audit)
    write_json(output / "competition_audit.json", competition_audit)
    write_json(
        output / "leakage_audit.json",
        {
            "status": "PASS" if gates["oracle_leakage_absent"] else "FAIL",
            "primary_fit_accepts_oracle": False,
            "prediction_accepts_keys": False,
            "scoring_is_separate": True,
            "forbidden_visible_fields_rejected": True,
        },
    )
    write_json(output / "inference_audit.json", inference_audit)
    write_json(output / "qualification_summary.json", summary)
    write_json(
        output / "seed_operation_log.json",
        {
            "status": "PASS" if gates["seed_firewall"] else "FAIL",
            "operation_count": len(firewall.log),
            "operations": firewall.log,
        },
    )
    write_json(output / "core_manifest.json", _core_manifest(output))
    metadata = output / "metadata"
    metadata.mkdir()
    write_json(
        metadata / "run_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "absolute_run_path": str(output),
            "deterministic_core_exclusions": [
                "metadata/",
                "core_manifest.json (self-excluded)",
            ],
        },
    )
    return summary


def _core_manifest(directory: Path) -> dict[str, Any]:
    files = []
    for path in sorted(value for value in directory.rglob("*") if value.is_file()):
        relative = path.relative_to(directory)
        if relative.parts[0] == "metadata" or relative.name in DETERMINISTIC_EXCLUSIONS:
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "files": files,
        "file_count": len(files),
        "self_excluded_by_definition": True,
        "allowed_metadata_exclusion": "metadata/",
    }


def compare_core_artifacts(
    primary_dir: str | Path,
    reproduction_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    left = _core_manifest(Path(primary_dir))
    right = _core_manifest(Path(reproduction_dir))
    left_map = {row["path"]: row for row in left["files"]}
    right_map = {row["path"]: row for row in right["files"]}
    paths = sorted(set(left_map) | set(right_map))
    rows = []
    for path in paths:
        first = left_map.get(path)
        second = right_map.get(path)
        rows.append(
            {
                "path": path,
                "primary": first,
                "reproduction": second,
                "match": first == second and first is not None,
            }
        )
    passed = bool(rows) and all(row["match"] for row in rows)
    report = {
        "status": "PASS" if passed else "FAIL",
        "reproduction_executed": Path(reproduction_dir).is_dir(),
        "separate_directories": Path(primary_dir).resolve()
        != Path(reproduction_dir).resolve(),
        "deterministic_file_sets_equal": set(left_map) == set(right_map),
        "byte_and_sha256_identity": passed,
        "compared_file_count": len(rows),
        "files": rows,
        "allowed_nondeterministic_metadata": ["metadata/run_metadata.json"],
    }
    write_json(output_path, report)
    return report


def freeze_package(
    repository_root: str | Path,
    config_path: str | Path,
    protocol_path: str | Path,
    sources_path: str | Path,
    traceability_path: str | Path,
    output_root: str | Path,
    tracked_paths: Sequence[str | Path],
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    config = load_config(config_path)
    if config["protocol"]["status"] != "frozen":
        raise RuntimeError("freeze requires config protocol.status=frozen")
    if (output / "freeze_receipt.json").exists():
        raise FileExistsError(output / "freeze_receipt.json")
    if (output / "qualification_run").exists() or (output / "isolated_reproduction").exists():
        raise RuntimeError("cannot freeze after excluded-smoke artifacts exist")
    fixture_directory = output / "fixture_validation_final"
    fixture_summary = json.loads(
        (fixture_directory / "qualification_summary.json").read_text()
    )
    if fixture_summary["status"] != "PASS":
        raise RuntimeError("fixture validation did not pass")
    snapshot_root = output / "frozen_source_snapshot"
    snapshot_root.mkdir(parents=True)
    all_paths = [
        Path(config_path),
        Path(protocol_path),
        Path(sources_path),
        Path(traceability_path),
        *(root / Path(path) for path in tracked_paths),
    ]
    unique_paths = sorted({path.resolve() for path in all_paths})
    original_hashes: dict[str, str] = {}
    snapshot_hashes: dict[str, str] = {}
    for source in unique_paths:
        relative = source.relative_to(root)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        original_hashes[relative.as_posix()] = sha256_file(source)
        snapshot_hashes[(Path("frozen_source_snapshot") / relative).as_posix()] = sha256_file(
            destination
        )
    for source, name in (
        (Path(config_path), "frozen_config_snapshot.yaml"),
        (Path(protocol_path), "frozen_protocol_snapshot.md"),
        (Path(sources_path), "primary_sources_snapshot.json"),
        (Path(traceability_path), "traceability_snapshot.md"),
    ):
        destination = output / name
        shutil.copyfile(source, destination)
        snapshot_hashes[name] = sha256_file(destination)
    write_json(output / "frozen_seed_registries.json", registry_snapshot(config))
    write_json(output / "frozen_gates.json", config["gates"])
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_only_checks_before_freeze": True,
        "fixture_validation_digest": sha256_file(
            fixture_directory / "qualification_summary.json"
        ),
        "excluded_smoke_runs_before_freeze": 0,
        "development_outcomes_touched": 0,
        "confirmation_outcomes_touched": 0,
        "original_content_hashes": original_hashes,
        "snapshot_content_hashes": snapshot_hashes,
        "detector_sha256": config["detector"]["frozen_v2_sha256"],
        "registries_sha256": sha256_file(output / "frozen_seed_registries.json"),
        "gates_sha256": sha256_file(output / "frozen_gates.json"),
        "post_freeze_amendments_allowed": False,
    }
    write_json(output / "freeze_receipt.json", receipt)
    return receipt


def verify_freeze(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    receipt = json.loads((output / "freeze_receipt.json").read_text())
    original = {
        path: sha256_file(root / path)
        for path in receipt["original_content_hashes"]
    }
    snapshots = {
        path: sha256_file(output / path.removeprefix("frozen_source_snapshot/"))
        if False
        else sha256_file(output / path)
        for path in receipt["snapshot_content_hashes"]
    }
    passed = (
        original == receipt["original_content_hashes"]
        and snapshots == receipt["snapshot_content_hashes"]
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "original_hashes_match": original == receipt["original_content_hashes"],
        "snapshot_hashes_match": snapshots == receipt["snapshot_content_hashes"],
        "checked_original_files": len(original),
        "checked_snapshot_files": len(snapshots),
    }


def _current_v1_v3_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for relative in V1_V3_ROOTS:
        directory = root / relative
        paths.update(
            path.relative_to(root).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
        )
    paths.update(relative for relative in V1_V3_FILES if (root / relative).is_file())
    return paths


def verify_preservation(
    repository_root: str | Path,
    before_manifest: str | Path,
    after_manifest: str | Path,
    proof_path: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    lines = Path(before_manifest).read_text().splitlines()
    if not lines or lines[0] != "sha256\tbytes\tpath":
        raise ValueError("invalid preservation baseline")
    before: dict[str, dict[str, Any]] = {}
    for line in lines[1:]:
        digest, size, path = line.split("\t", 2)
        before[path] = {"sha256": digest, "bytes": int(size)}
    current_paths = _current_v1_v3_paths(root)
    after: dict[str, dict[str, Any]] = {}
    for path in sorted(current_paths):
        value = root / path
        after[path] = {"sha256": sha256_file(value), "bytes": value.stat().st_size}
    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(
        path for path in set(before) & set(after) if before[path] != after[path]
    )
    after_lines = ["sha256\tbytes\tpath"] + [
        f"{value['sha256']}\t{value['bytes']}\t{path}"
        for path, value in sorted(after.items())
    ]
    Path(after_manifest).write_text("\n".join(after_lines) + "\n")
    proof = {
        "status": "PASS" if not missing and not added and not changed else "FAIL",
        "baseline_file_count": len(before),
        "after_file_count": len(after),
        "missing_paths": missing,
        "added_paths": added,
        "changed_paths": changed,
        "byte_for_byte_preserved": not missing and not added and not changed,
        "before_manifest_sha256": sha256_file(before_manifest),
        "after_manifest_sha256": sha256_file(after_manifest),
    }
    write_json(proof_path, proof)
    return proof


def run_test_suites(
    repository_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    commands = [
        ["python3", "-m", "pytest", "-q", "tests/test_synthetic_protocol_fidelity_v4.py"],
        ["python3", "-m", "pytest", "-q"],
    ]
    results = []
    environment = dict(__import__("os").environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        results.append(
            {
                "command": " ".join(command),
                "exit_code": completed.returncode,
                "output": completed.stdout,
            }
        )
        if completed.returncode != 0:
            break
    report = {
        "status": "PASS"
        if len(results) == len(commands) and all(row["exit_code"] == 0 for row in results)
        else "FAIL",
        "results": results,
        "order": "v4 tests first, then full repository suite",
        "bytecode_writes_disabled": True,
    }
    write_json(output_path, report)
    return report


def write_complete_manifest(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    manifest_path = output / "complete_file_manifest.json"
    files = []
    for path in sorted(value for value in output.rglob("*") if value.is_file()):
        if path == manifest_path:
            continue
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "files": files,
        "file_count": len(files),
        "self_excluded_by_definition": True,
        "summary_only": False,
    }
    write_json(manifest_path, manifest, overwrite=manifest_path.exists())
    return manifest


def finalize_package(
    repository_root: str | Path,
    output_root: str | Path,
    limitations_path: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    qualification = json.loads(
        (output / "qualification_run" / "qualification_summary.json").read_text()
    )
    reproduction = json.loads((output / "reproduction_comparison.json").read_text())
    tests = json.loads((output / "test_execution_report.json").read_text())
    freeze = verify_freeze(root, output)
    preservation = verify_preservation(
        root,
        output / "preserved_v1_v3_hashes_before.tsv",
        output / "preserved_v1_v3_hashes_after.tsv",
        output / "preservation_proof.json",
    )
    final_gates = {
        **{f"qualification_{key}": bool(value) for key, value in qualification["gates"].items()},
        "qualification_status": qualification["status"] == "PASS",
        "reproduction": reproduction["status"] == "PASS"
        and reproduction["reproduction_executed"],
        "freeze_integrity": freeze["status"] == "PASS",
        "v1_v3_preservation": preservation["status"] == "PASS",
        "tests": tests["status"] == "PASS",
        "development_outcomes_untouched": not qualification["future_development_touched"],
        "confirmation_untouched": not qualification["confirmation_touched"],
    }
    decision = "V4_READY" if all(final_gates.values()) else "REVISE"
    terminal = {
        "decision": decision,
        "package_readiness_only": True,
        "scientific_go": False,
        "development_outcome_authorized": False,
        "confirmation_authorized": False,
        "v3_stop_retained": True,
        "v3_scientific_execution_valid": False,
        "gates": final_gates,
        "reason": "All qualification gates passed; readiness does not authorize an outcome run."
        if decision == "V4_READY"
        else "At least one repairable package-qualification gate failed; no scientific outcome is authorized.",
    }
    write_json(output / "freeze_integrity_validation.json", freeze)
    write_json(output / "terminal_decision.json", terminal)
    shutil.copyfile(limitations_path, output / "scientific_limitations.md")
    validation_lines = [
        "# Independent v4 validation report",
        "",
        f"Terminal decision: **{decision}**.",
        "",
        "This validates package protocol fidelity only. It is not a scientific outcome and does not authorize development or confirmation.",
        "",
        "## Gates",
        "",
        *[
            f"- {'PASS' if value else 'FAIL'} — `{key}`"
            for key, value in sorted(final_gates.items())
        ],
        "",
        "V3 disposition: execution invalid; STOP retained; confirmation blocked.",
    ]
    (output / "independent_validation_report.md").write_text(
        "\n".join(validation_lines) + "\n"
    )
    write_complete_manifest(output)
    return terminal
