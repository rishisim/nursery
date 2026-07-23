from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .analysis import (
    leakage_audit,
    learnability_audit,
    pairing_fairness_audit,
    reserve_guard_audit,
    stochastic_replicate_audit,
    summarize_results,
    terminal_decision,
)
from .detector import (
    SensorEventDetector,
    evaluate_detector,
    evidence_for_episodes,
    fit_detector,
)
from .learners import (
    evaluate_final,
    evaluation_firewall_contract,
    fit_learner,
    make_evaluation_policy,
)
from .protocol import (
    PROTOCOL_ID,
    canonical_digest,
    create_freeze_receipt,
    guard_records,
    load_json,
    load_jsonl,
    load_protocol_config,
    make_confirmation_manifest,
    sha256_file,
    verify_confirmation_manifest,
    verify_freeze_receipt,
    write_json,
    write_jsonl,
)
from .synthetic import (
    SyntheticCorpus,
    condition_view,
    generate_calibration_data,
    generate_corpus,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text(path: Path, value: str, *, refuse_overwrite: bool = True) -> None:
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"refusing to overwrite preserved artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _factor_equal(value: Any, level: Any) -> bool:
    try:
        return float(value) == float(level)
    except (TypeError, ValueError):
        return str(value) == str(level)


def freeze_study(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    protocol_path: str | Path,
    sources_path: str | Path,
    output_dir: str | Path,
    tracked_paths: Sequence[str | Path],
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_dir).resolve()
    config = load_protocol_config(config_path)
    outcome_paths = (
        output / "raw",
        output / "detector",
        output / "audits",
        output / "aggregate_results.json",
        output / "factor_sensitivity.json",
        output / "terminal_decision.json",
        output / "scientific_report.md",
        output / "artifact_manifest.json",
    )
    if any(path.exists() for path in outcome_paths):
        raise RuntimeError("cannot freeze after a v2 outcome-producing artifact exists")
    output.mkdir(parents=True, exist_ok=True)
    confirmation_path = output / "confirmation_reserve_manifest.json"
    write_json(
        confirmation_path,
        make_confirmation_manifest(config),
        refuse_overwrite=True,
    )
    snapshots = (
        (Path(config_path), output / "frozen_config_snapshot.yaml"),
        (Path(protocol_path), output / "frozen_protocol_snapshot.md"),
        (Path(sources_path), output / "primary_sources_snapshot.json"),
    )
    for source, destination in snapshots:
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite frozen snapshot: {destination}")
        shutil.copyfile(source, destination)
    receipt_path = output / "freeze_receipt.json"
    receipt = create_freeze_receipt(
        repository_root=root,
        config_path=config_path,
        protocol_path=protocol_path,
        sources_path=sources_path,
        confirmation_manifest_path=confirmation_path,
        tracked_paths=[
            *tracked_paths,
            *(destination for _, destination in snapshots),
        ],
        output_path=receipt_path,
        created_at_utc=_utc_now(),
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "freeze_receipt": str(receipt_path),
        "confirmation_manifest": str(confirmation_path),
        "tracked_files": len(receipt["content_hashes"]),
        "outcome_producing_runs_before_freeze": 0,
        "fixture_only_smoke_checks_before_freeze": True,
    }


def _persist_calibration(calibration: Any, root: Path) -> None:
    split = str(calibration.split)
    write_jsonl(
        root / f"{split}_visible.jsonl",
        calibration.visible_records,
        refuse_overwrite=True,
    )
    write_jsonl(
        root / f"{split}_oracle.jsonl",
        calibration.oracle_records,
        refuse_overwrite=True,
    )
    write_json(
        root / f"{split}_provenance.json",
        calibration.provenance,
        refuse_overwrite=True,
    )


def _persist_corpus(corpus: SyntheticCorpus, root: Path) -> None:
    corpus_dir = root / f"corpus_{corpus.corpus_seed}"
    write_jsonl(
        corpus_dir / "visible_episodes.jsonl",
        corpus.visible_episodes,
        refuse_overwrite=True,
    )
    write_jsonl(
        corpus_dir / "oracle_episodes.jsonl",
        corpus.oracle_episodes,
        refuse_overwrite=True,
    )
    write_jsonl(
        corpus_dir / "evaluation_items.jsonl",
        (asdict(item) for item in corpus.evaluation_items),
        refuse_overwrite=True,
    )
    write_jsonl(
        corpus_dir / "evaluation_oracle.jsonl",
        corpus.evaluation_oracle,
        refuse_overwrite=True,
    )
    write_json(
        corpus_dir / "lexicon_oracle.json",
        corpus.lexicon_oracle,
        refuse_overwrite=True,
    )
    write_json(
        corpus_dir / "generation_audit.json",
        corpus.audits,
        refuse_overwrite=True,
    )
    write_json(
        corpus_dir / "shuffled_donor_map.json",
        corpus.donor_map,
        refuse_overwrite=True,
    )


def _serializable_evidence(
    evidence: Mapping[str, Mapping[str, Any]], *, corpus_seed: int, condition: str
) -> Iterable[dict[str, Any]]:
    for episode_id, value in sorted(evidence.items()):
        yield {
            "schema_version": "synthetic-sensor-derived-event-evidence-v2",
            "corpus_seed": int(corpus_seed),
            "condition": condition,
            "episode_id": episode_id,
            "event_logits": list(value["event_logits"]),
            "null_logit": float(value["null_logit"]),
            "owner_probabilities": list(value["owner_probabilities"]),
            "quality": float(value["quality"]),
            "availability": float(value["availability"]),
            "top_event_index": value["top_event_index"],
        }


def _condition_payloads(
    corpus: SyntheticCorpus,
    detector: SensorEventDetector,
    config: Mapping[str, Any],
    evidence_path: Path,
) -> tuple[
    dict[str, tuple[dict[str, Any], ...]],
    dict[str, dict[str, dict[str, Any]]],
]:
    views: dict[str, tuple[dict[str, Any], ...]] = {}
    evidence: dict[str, dict[str, dict[str, Any]]] = {}
    serializable: list[dict[str, Any]] = []
    for condition in config["conditions"]["names"]:
        name = str(condition)
        rows = condition_view(corpus, name, config)
        derived = evidence_for_episodes(detector, rows)
        views[name] = rows
        evidence[name] = derived
        serializable.extend(
            _serializable_evidence(
                derived,
                corpus_seed=corpus.corpus_seed,
                condition=name,
            )
        )
    write_jsonl(evidence_path, serializable, refuse_overwrite=True)
    return views, evidence


def _main_runs_for_corpus(
    corpus: SyntheticCorpus,
    views: Mapping[str, Sequence[Mapping[str, Any]]],
    evidence: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    policy = make_evaluation_policy(config)
    for model_seed in map(int, config["seeds"]["development"]["model"]):
        for condition in map(str, config["conditions"]["names"]):
            episodes = views[condition]
            derived = evidence[condition]
            for learner in map(str, config["learners"]["names"]):
                model, trace = fit_learner(
                    episodes=episodes,
                    learner=learner,
                    condition=condition,
                    corpus_seed=corpus.corpus_seed,
                    model_seed=model_seed,
                    config=config,
                    derived_evidence=derived,
                    oracle_rows=(
                        corpus.oracle_episodes
                        if learner
                        in {"oracle_event_alignment_upper", "v1_pointer_style_upper"}
                        else None
                    ),
                )
                metrics = evaluate_final(
                    model,
                    corpus.evaluation_items,
                    corpus_seed=corpus.corpus_seed,
                    policy=policy,
                )
                records.append(
                    {
                        "schema_version": "synthetic-sensor-event-development-run-v2",
                        "protocol_id": PROTOCOL_ID,
                        "analysis_kind": "main",
                        "corpus_seed": corpus.corpus_seed,
                        "model_seed": model_seed,
                        "condition": condition,
                        "learner": learner,
                        "paired_design_key": (
                            f"corpus={corpus.corpus_seed}|model={model_seed}|learner={learner}"
                        ),
                        "metrics": metrics,
                        "training_trace": trace,
                        "model_digest": canonical_digest(model.serializable()),
                    }
                )
    return records


def _sensitivity_runs_for_corpus(
    corpus: SyntheticCorpus,
    views: Mapping[str, Sequence[Mapping[str, Any]]],
    evidence: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    learner = str(config["sensitivity"]["learner"])
    policy = make_evaluation_policy(config)
    for factor, levels in config["factors"].items():
        for level in levels:
            indices = [
                index
                for index, row in enumerate(corpus.oracle_episodes)
                if _factor_equal(row["factors"][factor], level)
            ]
            if not indices:
                raise RuntimeError(f"empty frozen factor stratum: {factor}={level}")
            for model_seed in map(int, config["seeds"]["development"]["model"]):
                for condition in map(str, config["sensitivity"]["conditions"]):
                    episodes = tuple(views[condition][index] for index in indices)
                    episode_ids = {str(row["episode_id"]) for row in episodes}
                    derived = {
                        episode_id: value
                        for episode_id, value in evidence[condition].items()
                        if episode_id in episode_ids
                    }
                    model, trace = fit_learner(
                        episodes=episodes,
                        learner=learner,
                        condition=condition,
                        corpus_seed=corpus.corpus_seed,
                        model_seed=model_seed,
                        config=config,
                        derived_evidence=derived,
                        sensitivity=True,
                    )
                    metrics = evaluate_final(
                        model,
                        corpus.evaluation_items,
                        corpus_seed=corpus.corpus_seed,
                        policy=policy,
                    )
                    records.append(
                        {
                            "schema_version": "synthetic-sensor-event-sensitivity-run-v2",
                            "protocol_id": PROTOCOL_ID,
                            "analysis_kind": "sensitivity",
                            "corpus_seed": corpus.corpus_seed,
                            "model_seed": model_seed,
                            "condition": condition,
                            "learner": learner,
                            "factor": factor,
                            "level": level,
                            "n_training_episodes": len(episodes),
                            "metrics": metrics,
                            "training_trace": trace,
                            "model_digest": canonical_digest(model.serializable()),
                        }
                    )
    return records


def _format_pp(value: float) -> str:
    return f"{100.0 * float(value):+.2f} pp"


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _report_markdown(
    *,
    aggregate: Mapping[str, Any],
    factor: Mapping[str, Any],
    audits: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any],
    detector_validation: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> str:
    recommendation = decision["recommendation_for_later_v2_confirmation"]
    means = aggregate["primary_condition_means"]
    co_primary = aggregate["co_primary_estimands"]
    endpoint = aggregate["primary_learner_endpoint_condition_means"]
    lines = [
        "# Synthetic sensor-to-event robustness study v2",
        "",
        f"## Terminal recommendation: {recommendation}",
        "",
        (
            "This frozen development-only experiment tests whether synchronized raw synthetic "
            "six-axis IMU, proprioceptive state, and contact streams add training value for weak "
            "action-language grounding. The learner never receives an oracle target-event pointer, "
            "and final evaluation structurally accepts only learned lexical prototypes plus cue-free "
            "action/object observations."
        ),
        "",
        f"Recommendation rationale: {decision['reason']}.",
        "",
        "## Co-primary causal findings",
        "",
        "| Contrast (combined learner, held-out composition endpoint) | Corpus-level lift | 95% t CI | One-sided t p | Positive corpora | Exact sign p | Sign-flip p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for control in ("absent", "shuffled"):
        row = co_primary[control]
        lines.append(
            f"| synchronized − {control} | {_format_pp(row['point_estimate'])} | "
            f"[{_format_pp(row['ci95_low'])}, {_format_pp(row['ci95_high'])}] | "
            f"{row['one_sided_t_pvalue']:.6g} | {row['positive_corpus_count']}/"
            f"{row['n_independent_corpus_seeds']} | "
            f"{row['exact_sign_test_one_sided_pvalue']:.6g} | "
            f"{row['exhaustive_sign_flip_one_sided_pvalue']:.6g} |"
        )
    lines.extend(
        [
            "",
            "Both contrasts are co-primary under a predeclared intersection-union rule: both must "
            "pass their full effect, interval, one-sided test, and sign gates. Model seeds are "
            "stochastic algorithmic replicates averaged inside each of 20 independent corpus seeds.",
            "",
            "## Condition and time-shift characterization",
            "",
            "| Raw-sensor training condition | Primary accuracy | Difference from absence |",
            "| --- | ---: | ---: |",
        ]
    )
    for condition in map(str, config["conditions"]["names"]):
        difference = means[condition] - means["absent"]
        lines.append(
            f"| {condition} | {_format_percent(means[condition])} | {_format_pp(difference)} |"
        )
    lines.extend(
        [
            "",
            "Signed offsets were analyzed separately, rather than collapsed into a single "
            "adversarial condition. Shuffled and uninformative channels are safety controls; the "
            "frozen reliability mechanism may set token-level trust to zero and additionally applies "
            "training-time cue dropout.",
            "",
            "## Detector calibration and boundary capacity",
            "",
            "The fixed detector was trained only on independent generic calibration episodes with "
            "wearer-activity and boundary supervision. It had no lexical, referent, groundedness, "
            "or randomized-mapping targets and was reused unchanged in every sensor arm.",
            "",
            "| Held-out generic calibration measure | Result | Gate |",
            "| --- | ---: | ---: |",
            f"| Informative activity precision | {_format_percent(detector_validation['informative_timepoint']['precision'])} | ≥ {_format_percent(config['gates']['detector']['minimum_informative_timepoint_precision'])} |",
            f"| Informative activity recall | {_format_percent(detector_validation['informative_timepoint']['recall'])} | ≥ {_format_percent(config['gates']['detector']['minimum_informative_timepoint_recall'])} |",
            f"| Informative boundary F1 | {_format_percent(detector_validation['informative_boundary']['f1'])} | ≥ {_format_percent(config['gates']['detector']['minimum_informative_boundary_f1'])} |",
            f"| Informative candidate-owner AUC | {detector_validation['informative_candidate_owner_auc']:.3f} | ≥ {config['gates']['detector']['minimum_informative_candidate_owner_auc']:.2f} |",
            f"| Zero-information candidate-owner AUC | {detector_validation['zero_information_candidate_owner_auc']:.3f} | within ±{config['gates']['detector']['maximum_zero_information_auc_distance_from_chance']:.2f} of .50 |",
            "",
            "## Learners and transfer endpoints",
            "",
            "| Learner | Synchronized | Shuffled | Absent |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for learner in map(str, config["learners"]["names"]):
        row = aggregate["learner_condition_means"][learner]
        lines.append(
            f"| {learner} | {_format_percent(row['synchronized'])} | "
            f"{_format_percent(row['shuffled'])} | {_format_percent(row['absent'])} |"
        )
    lines.extend(
        [
            "",
            "`exact_window_symbolic` is the existing nearest-window assumption. The two upper "
            "controls use oracle event alignment and the v1 pointer-style score only to establish "
            "learnability; neither is the primary learner.",
            "",
            "| Cue-free synchronized endpoint (primary learner) | Accuracy |",
            "| --- | ---: |",
        ]
    )
    endpoint_labels = (
        ("seen_lexical_component_accuracy_on_independent_tokens", "Seen lexical components, independent tokens"),
        ("new_action_instance_accuracy", "New action instances"),
        ("seen_combination_action_6way_accuracy", "Seen action combinations"),
        ("heldout_object_action_composition_action_6way_macro_accuracy", "Held-out object-action compositions"),
        ("structured_heldout_concept_action_6way_accuracy", "Structured primitive × manner held-out concept"),
        ("matched_noun_6way_macro_accuracy", "Matched noun/non-action task"),
        ("zero_exposure_word_6way_accuracy", "Truly zero-exposure word panel"),
    )
    for key, label in endpoint_labels:
        lines.append(f"| {label} | {_format_percent(endpoint[key]['synchronized'])} |")
    info = factor["factors"]["sensor_informativeness"]
    lines.extend(
        [
            "",
            "The structured concept panel withholds one primitive × manner combination while "
            "exposing both components elsewhere. It does not claim recovery of an arbitrary, "
            "completely unexposed word meaning; that stronger leakage control is the zero-exposure "
            "panel and remains a chance benchmark.",
            "",
            "## Informativeness and matched selectivity controls",
            "",
            "| Sensor informativeness | Sync − absent | Sync − shuffled |",
            "| ---: | ---: | ---: |",
        ]
    )
    for level, values in sorted(info.items(), key=lambda item: float(item[0])):
        lines.append(
            f"| {level} | {_format_pp(values['synchronized_minus_absent']['point_estimate'])} | "
            f"{_format_pp(values['synchronized_minus_shuffled']['point_estimate'])} |"
        )
    noun = aggregate["matched_noun_contrasts"]
    lines.extend(
        [
            "",
            f"Matched noun lift was {_format_pp(noun['absent']['point_estimate'])} versus absence "
            f"and {_format_pp(noun['shuffled']['point_estimate'])} versus shuffled. Nouns use the "
            "same ambiguous bags, latent selection, cross-occurrence updates, reliability fusion, "
            "and final evaluator; only causal ownership is made irrelevant.",
            "",
            "## Audits and validation",
            "",
        ]
    )
    for name, artifact in sorted(audits.items()):
        lines.append(f"- `{name}`: **{'PASS' if artifact.get('valid') else 'FAIL'}**")
    lines.extend(
        [
            "",
            "An independent validation pass recomputed record counts, per-condition means, and both "
            "co-primary corpus effects directly from raw run records. Exact tables are used instead "
            "of a chart because the inferential unit, gates, and audit values are more inspectable "
            "in tabular form.",
            "",
            "## Inference and estimand definitions",
            "",
            "The independent unit is the generated corpus seed. Each corpus effect first averages "
            "the two stochastic model seeds and then subtracts matched conditions. Primary uncertainty "
            "uses Student t inference over 20 corpus effects (19 degrees of freedom), with exact sign "
            "and exhaustive sign-flip p-values as corroboration. A generic percentile bootstrap is "
            "not used. The synchronized-minus-absence contrast establishes positive added value; "
            "synchronized-minus-shuffled establishes synchronization specificity.",
            "",
            "Factor-stratified refits are secondary and descriptive; no additional discovery claim "
            "or multiplicity-adjusted factor inference is made.",
            "",
            "## Primary sources",
            "",
        ]
    )
    for source in sources:
        lines.append(
            f"- [{source['title']}]({source['url']}) — {source['publication']}. "
            f"{source['justification']}"
        )
    lines.extend(
        [
            "",
            "## Limitations and scope",
            "",
            "- This is a synthetic feature-level study, not BabyView/ChildLens evidence, infant evidence, raw-pixel/audio learning, or ecological validation.",
            "- Visual action/object observations are generated categorical feature vectors; direct capacity controls establish their learnability but do not validate a perceptual front end.",
            "- Detector supervision comes from a separate synthetic generic calibration distribution; transfer to real sensors remains untested.",
            "- Development evidence may motivate only a later separately authorized v2 confirmation. It does not authorize or reveal any v1 or v2 confirmation outcome.",
            "- AEA remains a terminal STOP; Machine-DevBench remains secondary at 13/1,414 coverage; BabyView and ChildLens access remain pending.",
            "",
            f"Recommendation: **{recommendation}**. Confirmation remains unauthorized.",
        ]
    )
    return "\n".join(lines) + "\n"


def _independent_validation(
    *,
    main_records: Sequence[Mapping[str, Any]],
    sensitivity_records: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    audits: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    primary = str(config["learners"]["primary"])
    metric = str(config["evaluation"]["primary_metric"])
    corpora = list(map(int, config["seeds"]["development"]["corpus"]))
    models = list(map(int, config["seeds"]["development"]["model"]))
    conditions = list(map(str, config["conditions"]["names"]))
    learners = list(map(str, config["learners"]["names"]))
    expected_main = len(corpora) * len(models) * len(conditions) * len(learners)
    expected_sensitivity = (
        sum(len(levels) for levels in config["factors"].values())
        * len(corpora)
        * len(models)
        * len(config["sensitivity"]["conditions"])
    )
    lookup: dict[tuple[int, int, str], float] = {}
    condition_values: dict[str, list[float]] = {condition: [] for condition in conditions}
    for row in main_records:
        if row["learner"] != primary:
            continue
        key = (int(row["corpus_seed"]), int(row["model_seed"]), str(row["condition"]))
        value = float(row["metrics"][metric])
        lookup[key] = value
        condition_values[str(row["condition"])].append(value)
    recomputed_means = {
        condition: float(np.mean(values))
        for condition, values in condition_values.items()
    }
    recomputed_effects: dict[str, dict[str, Any]] = {}
    for control in ("absent", "shuffled"):
        effects = []
        for corpus in corpora:
            sync = np.mean([lookup[(corpus, model, "synchronized")] for model in models])
            comparison = np.mean([lookup[(corpus, model, control)] for model in models])
            effects.append(float(sync - comparison))
        recomputed_effects[control] = {
            "mean": float(np.mean(effects)),
            "positive_corpora": int(np.sum(np.asarray(effects) > 0)),
            "unit_effects": effects,
        }
    checks = {
        "main_record_count": len(main_records) == expected_main,
        "sensitivity_record_count": len(sensitivity_records) == expected_sensitivity,
        "condition_means_match": all(
            abs(recomputed_means[name] - aggregate["primary_condition_means"][name])
            < 1e-12
            for name in conditions
        ),
        "co_primary_means_match": all(
            abs(
                recomputed_effects[name]["mean"]
                - aggregate["co_primary_estimands"][name]["point_estimate"]
            )
            < 1e-12
            for name in ("absent", "shuffled")
        ),
        "co_primary_sign_counts_match": all(
            recomputed_effects[name]["positive_corpora"]
            == aggregate["co_primary_estimands"][name]["positive_corpus_count"]
            for name in ("absent", "shuffled")
        ),
        "all_saved_audits_valid": all(bool(value.get("valid")) for value in audits.values()),
        "corpus_is_inferential_unit": aggregate["independent_unit"] == "20 corpus seeds",
        "primary_bootstrap_not_used": aggregate["primary_bootstrap_used"] is False,
    }
    return {
        "schema_version": "synthetic-sensor-event-independent-validation-v2",
        "protocol_id": PROTOCOL_ID,
        "valid": all(checks.values()),
        "checks": checks,
        "expected_and_observed_counts": {
            "main_expected": expected_main,
            "main_observed": len(main_records),
            "sensitivity_expected": expected_sensitivity,
            "sensitivity_observed": len(sensitivity_records),
        },
        "recomputed_primary_condition_means": recomputed_means,
        "recomputed_corpus_effects": recomputed_effects,
        "validation_scope": (
            "Independent arithmetic recomputation from raw run records plus saved-audit review; "
            "not an independent generator implementation."
        ),
    }


def _validation_markdown(validation: Mapping[str, Any]) -> str:
    lines = [
        "# Independent validation report — synthetic sensor study v2",
        "",
        f"Overall status: **{'PASS' if validation['valid'] else 'FAIL'}**",
        "",
        "This pass independently recomputed record counts, primary condition means, corpus-level "
        "paired effects, and positive-corpus counts from the raw run ledger.",
        "",
        "| Check | Status |",
        "| --- | --- |",
    ]
    for name, passed in validation["checks"].items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(["", "Scope: " + validation["validation_scope"], ""])
    return "\n".join(lines)


def _deterministic_paths(output: Path) -> list[Path]:
    paths = [
        output / "aggregate_results.json",
        output / "factor_sensitivity.json",
        output / "terminal_decision.json",
        output / "scientific_report.md",
        output / "validation_report.json",
        output / "validation_report.md",
    ]
    for directory in (output / "raw", output / "detector", output / "audits"):
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    return sorted(set(paths), key=lambda path: str(path.relative_to(output)))


def _artifact_manifest(output: Path) -> dict[str, Any]:
    files = {
        str(path.relative_to(output)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in _deterministic_paths(output)
    }
    return {
        "schema_version": "synthetic-sensor-event-artifact-manifest-v2",
        "protocol_id": PROTOCOL_ID,
        "deterministic_file_count": len(files),
        "files": files,
    }


def run_development_study(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    freeze_receipt_path: str | Path,
    confirmation_manifest_path: str | Path,
    sources_path: str | Path,
    output_dir: str | Path,
    command: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_dir).resolve()
    config = load_protocol_config(config_path)
    freeze = verify_freeze_receipt(
        repository_root=root,
        receipt_path=freeze_receipt_path,
    )
    confirmation = load_json(confirmation_manifest_path)
    verify_confirmation_manifest(confirmation, config)
    if any(
        path.exists()
        for path in (
            output / "raw",
            output / "detector",
            output / "audits",
            output / "aggregate_results.json",
            output / "terminal_decision.json",
        )
    ):
        raise FileExistsError("refusing to overwrite an existing v2 development run")
    start = time.monotonic()
    write_json(
        output / "development_run_manifest.json",
        {
            "schema_version": "synthetic-sensor-event-development-manifest-v2",
            "protocol_id": PROTOCOL_ID,
            "started_at_utc": _utc_now(),
            "command": command,
            "freeze_verification": freeze,
            "development_corpus_seeds": config["seeds"]["development"]["corpus"],
            "development_model_seeds": config["seeds"]["development"]["model"],
            "generic_calibration_train_seeds": config["seeds"]["generic_calibration"]["train"],
            "generic_calibration_validation_seeds": config["seeds"]["generic_calibration"]["validation"],
            "confirmation_outcomes_accessed": 0,
            "protocol_amendments": [],
            "invalidated_pre_freeze_outcomes": [],
        },
        refuse_overwrite=True,
    )

    calibration_train = generate_calibration_data(config, split="train")
    calibration_validation = generate_calibration_data(config, split="validation")
    _persist_calibration(calibration_train, output / "raw" / "calibration")
    _persist_calibration(calibration_validation, output / "raw" / "calibration")
    detector, detector_trace = fit_detector(calibration_train, config)
    detector_validation = evaluate_detector(detector, calibration_validation, config)
    write_json(
        output / "detector" / "frozen_detector.json",
        detector.serializable(),
        refuse_overwrite=True,
    )
    write_json(
        output / "detector" / "fit_trace.json",
        detector_trace,
        refuse_overwrite=True,
    )
    write_json(
        output / "detector" / "heldout_validation.json",
        detector_validation,
        refuse_overwrite=True,
    )
    print("[v2] frozen detector fitted and validated", file=sys.stderr, flush=True)

    main_records: list[dict[str, Any]] = []
    sensitivity_records: list[dict[str, Any]] = []
    generation_audits: list[Mapping[str, Any]] = []
    lexicons: list[Mapping[str, Any]] = []
    evaluation_items: list[Sequence[Any]] = []
    evaluation_oracles: list[Sequence[Mapping[str, Any]]] = []
    corpus_seeds = list(map(int, config["seeds"]["development"]["corpus"]))
    for corpus_index, seed in enumerate(corpus_seeds, start=1):
        corpus = generate_corpus(config, seed)
        _persist_corpus(corpus, output / "raw" / "corpora")
        views, evidence = _condition_payloads(
            corpus,
            detector,
            config,
            output / "raw" / "derived_evidence" / f"corpus_{seed}.jsonl",
        )
        main_records.extend(_main_runs_for_corpus(corpus, views, evidence, config))
        sensitivity_records.extend(
            _sensitivity_runs_for_corpus(corpus, views, evidence, config)
        )
        generation_audits.append(corpus.audits)
        lexicons.append(corpus.lexicon_oracle)
        evaluation_items.append(corpus.evaluation_items)
        evaluation_oracles.append(corpus.evaluation_oracle)
        print(
            f"[v2] completed corpus {corpus_index}/{len(corpus_seeds)} seed={seed}",
            file=sys.stderr,
            flush=True,
        )
    write_jsonl(
        output / "raw" / "development_runs.jsonl",
        main_records,
        refuse_overwrite=True,
    )
    write_jsonl(
        output / "raw" / "sensitivity_runs.jsonl",
        sensitivity_records,
        refuse_overwrite=True,
    )
    guard_records(main_records, config, operation="read")
    guard_records(sensitivity_records, config, operation="read")

    audits: dict[str, dict[str, Any]] = {
        "protocol_freeze": {"valid": bool(freeze["valid"]), **freeze},
        "detector_validation": dict(detector_validation),
        "calibration_boundary": {
            "schema_version": "synthetic-sensor-event-calibration-boundary-audit-v2",
            "valid": bool(calibration_train.provenance["valid"])
            and bool(calibration_validation.provenance["valid"])
            and detector_trace["lexical_supervision_used"] is False
            and detector_trace["referent_supervision_used"] is False
            and detector_trace["randomized_mapping_used"] is False,
            "train_provenance": calibration_train.provenance,
            "validation_provenance": calibration_validation.provenance,
            "detector_fit_trace": detector_trace,
        },
        "pairing_fairness": pairing_fairness_audit(
            main_records,
            generation_audits,
            config,
        ),
        "stochastic_model_replicates": stochastic_replicate_audit(
            main_records,
            config,
        ),
        "leakage": leakage_audit(
            main_records,
            lexicons,
            generation_audits,
            [calibration_train.provenance, calibration_validation.provenance],
            config,
        ),
        "learnability": learnability_audit(
            main_records,
            evaluation_items,
            evaluation_oracles,
            config,
        ),
        "evaluation_firewall": evaluation_firewall_contract(),
        "confirmation_reserve_guard": reserve_guard_audit(config, confirmation),
    }
    for name, value in audits.items():
        write_json(
            output / "audits" / f"{name}.json",
            value,
            refuse_overwrite=True,
        )
    aggregate, factor = summarize_results(main_records, sensitivity_records, config)
    aggregate["development_run_counts"] = {
        "main": len(main_records),
        "sensitivity": len(sensitivity_records),
        "corpora": len(config["seeds"]["development"]["corpus"]),
        "model_seeds_per_corpus": len(config["seeds"]["development"]["model"]),
    }
    write_json(output / "aggregate_results.json", aggregate, refuse_overwrite=True)
    write_json(output / "factor_sensitivity.json", factor, refuse_overwrite=True)
    decision = terminal_decision(aggregate, factor, audits, config)
    write_json(output / "terminal_decision.json", decision, refuse_overwrite=True)
    source_payload = load_json(sources_path)
    report = _report_markdown(
        aggregate=aggregate,
        factor=factor,
        audits=audits,
        decision=decision,
        detector_validation=detector_validation,
        sources=list(source_payload["sources"]),
        config=config,
    )
    _write_text(output / "scientific_report.md", report)
    validation = _independent_validation(
        main_records=main_records,
        sensitivity_records=sensitivity_records,
        aggregate=aggregate,
        audits=audits,
        config=config,
    )
    write_json(output / "validation_report.json", validation, refuse_overwrite=True)
    _write_text(output / "validation_report.md", _validation_markdown(validation))
    manifest = _artifact_manifest(output)
    write_json(output / "artifact_manifest.json", manifest, refuse_overwrite=True)
    elapsed = time.monotonic() - start
    reproducibility = {
        "schema_version": "synthetic-sensor-event-reproducibility-v2",
        "protocol_id": PROTOCOL_ID,
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": elapsed,
        "command": command,
        "python_executable": sys.executable,
        "freeze_receipt_sha256": sha256_file(freeze_receipt_path),
        "confirmation_manifest_sha256": sha256_file(confirmation_manifest_path),
        "artifact_manifest_sha256": sha256_file(output / "artifact_manifest.json"),
        "deterministic_file_count": manifest["deterministic_file_count"],
        "raw_main_runs": len(main_records),
        "raw_sensitivity_runs": len(sensitivity_records),
        "confirmation_outcomes_generated_trained_evaluated_read_or_summarized": 0,
        "protocol_amendments": [],
        "reproduction_commands": [
            ".venv/bin/python scripts/run_synthetic_sensor_event_robustness_v2.py verify",
            (
                ".venv/bin/python scripts/run_synthetic_sensor_event_robustness_v2.py "
                "run-development --out /tmp/synthetic_sensor_event_robustness_v2_reproduction"
            ),
            (
                ".venv/bin/python scripts/run_synthetic_sensor_event_robustness_v2.py compare "
                "--left output/synthetic_sensor_event_robustness_v2 "
                "--right /tmp/synthetic_sensor_event_robustness_v2_reproduction"
            ),
            ".venv/bin/python -m pytest tests/test_synthetic_sensor_event_robustness_v2.py -q",
            ".venv/bin/python -m pytest -q",
        ],
    }
    write_json(output / "reproducibility.json", reproducibility, refuse_overwrite=True)
    print(
        f"[v2] study complete: {decision['recommendation_for_later_v2_confirmation']}",
        file=sys.stderr,
        flush=True,
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "recommendation": decision["recommendation_for_later_v2_confirmation"],
        "co_primary_effects": aggregate["co_primary_estimands"],
        "output_dir": str(output),
        "elapsed_seconds": elapsed,
        "all_audits_valid": all(value.get("valid") for value in audits.values()),
        "independent_validation_valid": validation["valid"],
        "confirmation_outcomes_accessed": 0,
    }


def _load_audits(output: Path) -> dict[str, dict[str, Any]]:
    return {
        path.stem: load_json(path)
        for path in sorted((output / "audits").glob("*.json"))
    }


def verify_study(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    freeze_receipt_path: str | Path,
    confirmation_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_dir).resolve()
    config = load_protocol_config(config_path)
    freeze = verify_freeze_receipt(
        repository_root=root,
        receipt_path=freeze_receipt_path,
    )
    confirmation = load_json(confirmation_manifest_path)
    confirmation_checks = verify_confirmation_manifest(confirmation, config)
    main = load_jsonl(output / "raw" / "development_runs.jsonl")
    sensitivity = load_jsonl(output / "raw" / "sensitivity_runs.jsonl")
    guard_records(main, config, operation="read")
    guard_records(sensitivity, config, operation="read")
    artifact_manifest = load_json(output / "artifact_manifest.json")
    file_checks = {
        relative: (output / relative).is_file()
        and sha256_file(output / relative) == metadata["sha256"]
        and (output / relative).stat().st_size == int(metadata["bytes"])
        for relative, metadata in artifact_manifest["files"].items()
    }
    audits = _load_audits(output)
    saved_aggregate = load_json(output / "aggregate_results.json")
    saved_factor = load_json(output / "factor_sensitivity.json")
    recomputed_aggregate, recomputed_factor = summarize_results(main, sensitivity, config)
    recomputed_aggregate["development_run_counts"] = saved_aggregate[
        "development_run_counts"
    ]
    saved_decision = load_json(output / "terminal_decision.json")
    recomputed_decision = terminal_decision(
        recomputed_aggregate,
        recomputed_factor,
        audits,
        config,
    )
    reproducibility = load_json(output / "reproducibility.json")
    validation = load_json(output / "validation_report.json")
    expected_main = (
        len(config["seeds"]["development"]["corpus"])
        * len(config["seeds"]["development"]["model"])
        * len(config["conditions"]["names"])
        * len(config["learners"]["names"])
    )
    expected_sensitivity = (
        sum(len(levels) for levels in config["factors"].values())
        * len(config["seeds"]["development"]["corpus"])
        * len(config["seeds"]["development"]["model"])
        * len(config["sensitivity"]["conditions"])
    )
    checks = {
        "freeze": bool(freeze["valid"]),
        "confirmation_manifest": all(confirmation_checks.values()),
        "artifact_manifest_schema": artifact_manifest.get("protocol_id") == PROTOCOL_ID,
        "deterministic_artifact_hashes": all(file_checks.values()),
        "all_audits_valid": all(bool(value.get("valid")) for value in audits.values()),
        "raw_main_run_count": len(main) == expected_main,
        "raw_sensitivity_run_count": len(sensitivity) == expected_sensitivity,
        "aggregate_recomputes_exactly": canonical_digest(saved_aggregate)
        == canonical_digest(recomputed_aggregate),
        "factor_recomputes_exactly": canonical_digest(saved_factor)
        == canonical_digest(recomputed_factor),
        "decision_recomputes_exactly": canonical_digest(saved_decision)
        == canonical_digest(recomputed_decision),
        "independent_validation": bool(validation.get("valid")),
        "artifact_manifest_self_hash": sha256_file(output / "artifact_manifest.json")
        == reproducibility["artifact_manifest_sha256"],
        "no_confirmation_outcomes": reproducibility[
            "confirmation_outcomes_generated_trained_evaluated_read_or_summarized"
        ]
        == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v2 study integrity verification failed: {checks}")
    return {
        "valid": True,
        "checks": checks,
        "artifact_file_checks": file_checks,
        "deterministic_file_count": len(file_checks),
        "recommendation": saved_decision["recommendation_for_later_v2_confirmation"],
    }


def compare_reproductions(
    *, left_dir: str | Path, right_dir: str | Path
) -> dict[str, Any]:
    left = Path(left_dir).resolve()
    right = Path(right_dir).resolve()
    left_manifest = load_json(left / "artifact_manifest.json")
    right_manifest = load_json(right / "artifact_manifest.json")
    left_files = left_manifest["files"]
    right_files = right_manifest["files"]
    inventory_equal = set(left_files) == set(right_files)
    comparisons = {
        relative: bool(
            relative in right_files
            and left_files[relative]["sha256"] == right_files[relative]["sha256"]
            and int(left_files[relative]["bytes"])
            == int(right_files[relative]["bytes"])
        )
        for relative in sorted(left_files)
    }
    valid = inventory_equal and all(comparisons.values())
    if not valid:
        failures = [name for name, passed in comparisons.items() if not passed]
        raise RuntimeError(
            f"v2 deterministic reproduction comparison failed; inventory_equal={inventory_equal}, "
            f"failures={failures[:20]}"
        )
    return {
        "schema_version": "synthetic-sensor-event-reproduction-comparison-v2",
        "valid": True,
        "left": str(left),
        "right": str(right),
        "inventory_equal": True,
        "byte_identical_deterministic_files": len(comparisons),
        "comparisons": comparisons,
    }
