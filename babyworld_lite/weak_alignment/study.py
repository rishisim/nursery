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

from babyworld_lite.weak_alignment.analysis import (
    condition_means,
    leakage_shortcut_checks,
    learnability_checks,
    manipulation_checks,
    pairing_fairness_checks,
    reserve_guard_audit,
    summarize_results,
    terminal_decision,
)
from babyworld_lite.weak_alignment.learners import (
    LEARNERS,
    evaluate_without_auxiliary_modality,
    fit_learner,
    lexical_mapping_accuracy,
    modality_withholding_contract,
)
from babyworld_lite.weak_alignment.protocol import (
    PROTOCOL_ID,
    create_freeze_receipt,
    guard_records_for_read_or_summary,
    load_json,
    load_protocol_config,
    make_confirmation_manifest,
    sha256_file,
    verify_confirmation_manifest,
    verify_freeze_receipt,
    write_json,
)
from babyworld_lite.weak_alignment.synthetic import (
    ACTIONS,
    SIDE_CONDITIONS,
    SyntheticCorpus,
    generate_corpus,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text(path: Path, value: str, *, refuse_overwrite: bool = True) -> None:
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"refusing to overwrite preserved artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _write_jsonl(
    path: Path, values: Iterable[Mapping[str, Any]], *, refuse_overwrite: bool = True
) -> None:
    _write_text(
        path,
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        refuse_overwrite=refuse_overwrite,
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
    out = Path(output_dir).resolve()
    config = load_protocol_config(config_path)
    forbidden_outcomes = (
        out / "raw",
        out / "aggregate_results.json",
        out / "scientific_report.md",
        out / "terminal_decision.json",
    )
    if any(path.exists() for path in forbidden_outcomes):
        raise RuntimeError("cannot freeze after outcome-producing artifacts exist")
    out.mkdir(parents=True, exist_ok=True)
    confirmation_path = out / "confirmation_reserve_manifest.json"
    write_json(
        confirmation_path, make_confirmation_manifest(config), refuse_overwrite=True
    )
    snapshot_config = out / "frozen_config_snapshot.yaml"
    snapshot_protocol = out / "frozen_protocol_snapshot.md"
    snapshot_sources = out / "primary_sources_snapshot.json"
    for source, destination in (
        (Path(config_path), snapshot_config),
        (Path(protocol_path), snapshot_protocol),
        (Path(sources_path), snapshot_sources),
    ):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite frozen snapshot: {destination}")
        shutil.copyfile(source, destination)
    receipt_path = out / "freeze_receipt.json"
    receipt = create_freeze_receipt(
        repository_root=root,
        config_path=config_path,
        protocol_path=protocol_path,
        sources_path=sources_path,
        confirmation_manifest_path=confirmation_path,
        tracked_paths=[*tracked_paths, snapshot_config, snapshot_protocol, snapshot_sources],
        output_path=receipt_path,
        created_at_utc=_utc_now(),
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "freeze_receipt": str(receipt_path),
        "confirmation_manifest": str(confirmation_path),
        "tracked_files": len(receipt["content_hashes"]),
        "outcome_producing_runs_before_freeze": 0,
    }


def _persist_corpus(corpus: SyntheticCorpus, root: Path) -> None:
    corpus_dir = root / f"corpus_{corpus.corpus_seed}"
    _write_jsonl(corpus_dir / "visible_episodes.jsonl", corpus.visible_episodes)
    _write_jsonl(corpus_dir / "oracle_episodes.jsonl", corpus.oracle_episodes)
    _write_jsonl(
        corpus_dir / "synchronized_side.jsonl",
        (
            {"episode_id": episode_id, "scores": list(scores)}
            for episode_id, scores in sorted(corpus.synchronized_side.items())
        ),
    )
    _write_jsonl(
        corpus_dir / "evaluation_items.jsonl",
        (asdict(item) for item in corpus.evaluation_items),
    )
    _write_jsonl(corpus_dir / "evaluation_oracle.jsonl", corpus.evaluation_oracle)
    write_json(corpus_dir / "lexicon_oracle.json", corpus.lexicon_oracle, refuse_overwrite=True)
    write_json(corpus_dir / "generation_audit.json", corpus.audits, refuse_overwrite=True)
    write_json(corpus_dir / "shuffled_donor_map.json", corpus.donor_map, refuse_overwrite=True)


def _main_runs(
    corpora: Sequence[SyntheticCorpus], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    model_seeds = list(map(int, config["seeds"]["development"]["model"]))
    for corpus in corpora:
        primary_words = [
            word
            for panel_index in corpus.lexicon_oracle["primary_scoring_panels"]
            for word in corpus.lexicon_oracle["action_panels"][panel_index]
        ]
        for model_seed in model_seeds:
            for condition in config["conditions"]["training_side_modality"]:
                episodes = corpus.condition_views[str(condition)]
                for learner in config["learners"]["names"]:
                    model, trace = fit_learner(
                        episodes=episodes,
                        learner=str(learner),
                        corpus_seed=corpus.corpus_seed,
                        model_seed=model_seed,
                        config=config,
                        oracle_rows=(
                            corpus.oracle_episodes if learner == "oracle_alignment" else None
                        ),
                    )
                    metrics = evaluate_without_auxiliary_modality(
                        model,
                        corpus.evaluation_items,
                        corpus_seed=corpus.corpus_seed,
                        config=config,
                    )
                    metrics["primary_heldout_lexical_mapping_accuracy"] = lexical_mapping_accuracy(
                        model,
                        corpus.lexicon_oracle["action_word_to_action"],
                        eligible_words=primary_words,
                    )
                    records.append({
                        "schema_version": "synthetic-weak-alignment-run-v1",
                        "protocol_id": PROTOCOL_ID,
                        "analysis_kind": "main",
                        "corpus_seed": corpus.corpus_seed,
                        "model_seed": model_seed,
                        "condition": str(condition),
                        "learner": str(learner),
                        "paired_design_key": f"corpus={corpus.corpus_seed}|model={model_seed}|learner={learner}",
                        "metrics": metrics,
                        "training_trace": trace,
                        "model_digest": trace["serialized_model_digest"],
                    })
    return records


def _factor_equal(value: Any, level: Any) -> bool:
    try:
        return float(value) == float(level)
    except (TypeError, ValueError):
        return str(value) == str(level)


def _sensitivity_runs(
    corpora: Sequence[SyntheticCorpus], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    learner = str(config["evaluation"]["primary_learner"])
    conditions = list(config["sensitivity_analysis"]["conditions"])
    model_seeds = list(map(int, config["seeds"]["development"]["model"]))
    for corpus in corpora:
        for factor, levels in config["factors"].items():
            for level in levels:
                indices = [
                    index for index, oracle in enumerate(corpus.oracle_episodes)
                    if _factor_equal(oracle["factors"][factor], level)
                ]
                if not indices:
                    raise RuntimeError(f"empty frozen sensitivity stratum: {factor}={level}")
                eligible_words = sorted({
                    str(corpus.visible_episodes[index]["utterance"]["action_word"])
                    for index in indices
                    if int(str(corpus.visible_episodes[index]["utterance"]["action_word"]).split("_")[1]) in {0, 1}
                })
                for model_seed in model_seeds:
                    for condition in conditions:
                        episodes = [corpus.condition_views[str(condition)][index] for index in indices]
                        model, trace = fit_learner(
                            episodes=episodes,
                            learner=learner,
                            corpus_seed=corpus.corpus_seed,
                            model_seed=model_seed,
                            config=config,
                        )
                        metric = lexical_mapping_accuracy(
                            model,
                            corpus.lexicon_oracle["action_word_to_action"],
                            eligible_words=eligible_words,
                        )
                        if not np.isfinite(metric):
                            raise RuntimeError(f"non-finite sensitivity metric: {factor}={level}")
                        records.append({
                            "schema_version": "synthetic-weak-alignment-sensitivity-run-v1",
                            "protocol_id": PROTOCOL_ID,
                            "analysis_kind": "sensitivity",
                            "corpus_seed": corpus.corpus_seed,
                            "model_seed": model_seed,
                            "condition": str(condition),
                            "learner": learner,
                            "stratum_factor": factor,
                            "stratum_level": level,
                            "n_training_episodes": len(episodes),
                            "n_supported_primary_words": len(eligible_words),
                            "metrics": {
                                "supported_primary_lexical_mapping_accuracy": metric,
                            },
                            "training_trace": trace,
                        })
    return records


def _format_pp(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def _report_markdown(
    *,
    aggregate: Mapping[str, Any],
    audits: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    main_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> str:
    primary = aggregate["primary_estimand"]
    means = aggregate["primary_condition_means"]
    recommendation = decision["recommendation_for_later_frozen_synthetic_confirmation"]
    learner_rows = aggregate["learner_condition_means"]
    lines = [
        "# Synthetic weak-alignment recovery study v1",
        "",
        f"## Terminal recommendation: {recommendation}",
        "",
        (
            "This development-only symbolic study asks whether lexical/action meanings can be "
            "recovered from temporally extended, weakly aligned bags and transferred to new visible "
            "action instances in held-out object-action compositions. It does **not** test complete "
            "language acquisition or timestamp agreement."
        ),
        "",
        "## Primary finding",
        "",
        (
            f"For the frozen combined latent-MIL + cross-occurrence learner, synchronized minus "
            f"group-safe shuffled side information was {_format_pp(primary['point_estimate'])} "
            f"(paired hierarchical 95% CI {_format_pp(primary['ci95_low'])} to "
            f"{_format_pp(primary['ci95_high'])}; {primary['positive_pair_count']}/"
            f"{primary['n_paired_units']} corpus/model pairs positive)."
        ),
        "",
        "The primary condition means were:",
        "",
        "| Training-time side condition | Held-out action 6-way accuracy |",
        "| --- | ---: |",
    ]
    for condition in config["conditions"]["training_side_modality"]:
        lines.append(f"| {condition} | {100 * means[condition]:.2f}% |")
    lines.extend([
        "",
        "Side information was structurally absent from the final evaluator and serialized lexical "
        "model. The final test used only learned lexical prototypes and new action/scene observations.",
        "",
        "## Learner comparison",
        "",
        "| Learner | Synchronized | Shuffled | Absent |",
        "| --- | ---: | ---: | ---: |",
    ])
    for learner in config["learners"]["names"]:
        row = learner_rows[learner]
        lines.append(
            f"| {learner} | {100 * row['synchronized']:.2f}% | "
            f"{100 * row['shuffled']:.2f}% | {100 * row['absent']:.2f}% |"
        )
    lines.extend([
        "",
        "`exact_window` selects the event nearest the utterance and has no null option. "
        "`latent_mil_single_occurrence` isolates latent selection without repetition; "
        "`cross_situational_uniform` aggregates repetitions without latent selection; "
        "`latent_mil_cross_no_null` removes only the null state; and `oracle_alignment` is the "
        "strong-alignment positive control.",
        "",
        "## Manipulations, validity, and controls",
        "",
    ])
    for name, artifact in audits.items():
        lines.append(f"- {name}: **{'PASS' if artifact.get('valid') else 'FAIL'}**")
    manipulation = audits["manipulation_checks"]["results"]["synchronized"]
    info = manipulation["configured_informative"]
    zero = manipulation["configured_uninformative"]
    learnability = audits["learnability_controls"]["results"]
    leakage = audits["leakage_shortcut_checks"]
    lines.extend([
        "",
        (
            f"When configured informative, synchronized cues identified the true event or null "
            f"{100 * info['referent_or_null_top1_accuracy']:.1f}% of the time versus a "
            f"{100 * info['mean_chance']:.1f}% chance reference. At configured informativeness 0, "
            f"the corresponding rate was {100 * zero['referent_or_null_top1_accuracy']:.1f}% "
            f"versus {100 * zero['mean_chance']:.1f}% chance."
        ),
        "",
        (
            f"Oracle alignment reached {100 * learnability['oracle_alignment_primary_accuracy']:.1f}% "
            f"on the primary endpoint. The zero-exposure lexical-type control was "
            f"{100 * leakage['zero_exposure_action_accuracy']:.1f}% (6-way chance "
            f"{100 * leakage['chance']:.1f}%)."
        ),
        "",
        "## Factor sensitivity",
        "",
        "Each secondary analysis refit the frozen primary learner within one level of one manipulated "
        "factor and retained corpus/model pairing. These are descriptive; no multiplicity-adjusted "
        "claims are made.",
        "",
    ])
    for factor, levels in aggregate["factor_sensitivity"].items():
        descriptions = ", ".join(
            f"{level}: {_format_pp(result['point_estimate'])}"
            for level, result in levels.items()
        )
        lines.append(f"- {factor} — synchronized minus shuffled lexical mapping: {descriptions}")
    lines.extend([
        "",
        "## Estimands and uncertainty",
        "",
        "The sole primary estimand is held-out-composition action 6-way macro accuracy for the "
        "combined learner under synchronized training minus group-safe shuffled training. The "
        "interval resamples four corpus seeds and then three model seeds within each sampled corpus; "
        "episodes and windows are never treated as independent inferential units. All other condition, "
        "component, noun-control, and factor-stratified contrasts are secondary/descriptive.",
        "",
        "## Decision rule outcome",
        "",
    ])
    for name, passed in decision["go_checks"].items():
        lines.append(f"- GO gate `{name}`: {'pass' if passed else 'fail'}")
    lines.extend([
        "",
        f"Recommendation: **{recommendation}** — {decision['reason']}. This artifact does not "
        "authorize the reserved confirmation seeds; they remain inaccessible without a separate, "
        "future explicit authorization.",
        "",
        "## Current primary sources (accessed 2026-07-15)",
        "",
    ])
    for source in sources:
        lines.append(
            f"- [{source['title']}]({source['url']}) — {source['publication']}. "
            f"Justification: {source['justification']}"
        )
    lines.extend([
        "",
        "## Relationship to prior repository evidence",
        "",
        "The original exact-window pilot remains a negative infrastructure result (synchronized "
        "minus shuffled -20.37 percentage points, 95% CI [-33.80, -1.39]). AEA v3 remains a terminal "
        "STOP: only 7/72 audited windows had consensus natural-speech/action alignment, so this study "
        "does not reopen AEA language, sensor, or acquisition work. AEA is an adult, partly scripted "
        "sensor-format analogue, not developmental or BabyView-like evidence. Machine-DevBench remains "
        "secondary and was not rerun because exact target-word coverage is only 13/1,414.",
        "",
        "Two stale-document contradictions are preserved rather than silently repaired: "
        "`docs/frank_preaccess_experimental_plan.md` says Machine-DevBench was not reproduced and "
        "mentions 24 tests, whereas newer artifacts document completed reproduction/integration and "
        "the parent audit found 69 passing tests; README AEA acquisition instructions are superseded "
        "by the v3 terminal STOP artifact.",
        "",
        "## Limitations",
        "",
        "- This is a controlled symbolic feature study, not raw-pixel or raw-audio learning and not infant evidence.",
        "- Canonical perceptual action dimensions are assumed learnable; the study isolates lexical assignment and transfer.",
        "- Development seeds are used for method diagnosis only. No confirmation outcome was generated, read, evaluated, or summarized.",
        "- The fixed fractional design balances the complete non-repetition factor grid within both repetition levels, but only four corpus seeds support uncertainty estimation.",
        "- A positive result would establish only synthetic lexical/action grounding under this generator and learner family.",
    ])
    return "\n".join(lines) + "\n"


def _deterministic_hashes(output: Path) -> dict[str, str]:
    candidates = [
        output / "raw" / "development_runs.jsonl",
        output / "raw" / "sensitivity_runs.jsonl",
        output / "aggregate_results.json",
        output / "scientific_report.md",
        output / "terminal_decision.json",
        *sorted((output / "audits").glob("*.json")),
    ]
    return {
        str(path.relative_to(output)): sha256_file(path)
        for path in candidates if path.is_file()
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
    freeze = verify_freeze_receipt(repository_root=root, receipt_path=freeze_receipt_path)
    confirmation = load_json(confirmation_manifest_path)
    verify_confirmation_manifest(confirmation, config)
    if (output / "raw").exists() or (output / "aggregate_results.json").exists():
        raise FileExistsError("refusing to overwrite an existing development study run")
    start = time.monotonic()
    write_json(
        output / "development_run_manifest.json",
        {
            "schema_version": "synthetic-weak-alignment-development-run-manifest-v1",
            "protocol_id": PROTOCOL_ID,
            "started_at_utc": _utc_now(),
            "command": command,
            "freeze_verification": freeze,
            "development_corpus_seeds": config["seeds"]["development"]["corpus"],
            "development_model_seeds": config["seeds"]["development"]["model"],
            "confirmation_outcomes_accessed": 0,
            "protocol_amendments": [],
            "invalidated_earlier_outcomes": [],
        },
        refuse_overwrite=True,
    )
    corpora = [
        generate_corpus(config, int(seed))
        for seed in config["seeds"]["development"]["corpus"]
    ]
    for corpus in corpora:
        _persist_corpus(corpus, output / "raw" / "corpora")
    main_records = _main_runs(corpora, config)
    sensitivity_records = _sensitivity_runs(corpora, config)
    _write_jsonl(output / "raw" / "development_runs.jsonl", main_records)
    _write_jsonl(output / "raw" / "sensitivity_runs.jsonl", sensitivity_records)
    guard_records_for_read_or_summary(main_records, config, operation="read")
    guard_records_for_read_or_summary(sensitivity_records, config, operation="read")

    audits: dict[str, dict[str, Any]] = {
        "protocol_freeze": {"valid": bool(freeze["valid"]), **freeze},
        "manipulation_checks": manipulation_checks(corpora, config),
        "pairing_fairness_checks": pairing_fairness_checks(main_records, corpora, config),
        "leakage_shortcut_checks": leakage_shortcut_checks(main_records, corpora, config),
        "learnability_controls": learnability_checks(main_records, corpora, config),
        "modality_withholding_audit": modality_withholding_contract(),
        "confirmation_reserve_guard": reserve_guard_audit(config, confirmation),
    }
    for name, value in audits.items():
        write_json(output / "audits" / f"{name}.json", value, refuse_overwrite=True)

    aggregate = summarize_results(main_records, sensitivity_records, config)
    primary_metric = str(config["evaluation"]["primary_metric"])
    aggregate["learner_condition_means"] = {
        learner: condition_means(
            main_records, config, learner=learner, metric=primary_metric
        )
        for learner in config["learners"]["names"]
    }
    aggregate["development_run_counts"] = {
        "main": len(main_records),
        "sensitivity": len(sensitivity_records),
        "corpora": len(corpora),
        "model_seeds_per_corpus": len(config["seeds"]["development"]["model"]),
    }
    write_json(output / "aggregate_results.json", aggregate, refuse_overwrite=True)
    decision = terminal_decision(aggregate, audits, config)
    write_json(output / "terminal_decision.json", decision, refuse_overwrite=True)
    source_payload = load_json(sources_path)
    sources = list(source_payload["sources"])
    report = _report_markdown(
        aggregate=aggregate,
        audits=audits,
        decision=decision,
        sources=sources,
        main_records=main_records,
        config=config,
    )
    _write_text(output / "scientific_report.md", report)
    elapsed = time.monotonic() - start
    deterministic = _deterministic_hashes(output)
    provenance = {
        "schema_version": "synthetic-weak-alignment-reproducibility-v1",
        "protocol_id": PROTOCOL_ID,
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": elapsed,
        "command": command,
        "python_executable": sys.executable,
        "freeze_receipt_sha256": sha256_file(freeze_receipt_path),
        "confirmation_manifest_sha256": sha256_file(confirmation_manifest_path),
        "deterministic_artifact_hashes": deterministic,
        "raw_main_runs": len(main_records),
        "raw_sensitivity_runs": len(sensitivity_records),
        "confirmation_outcomes_generated_trained_evaluated_read_or_summarized": 0,
        "protocol_amendments": [],
        "reproduction_commands": [
            ".venv/bin/python scripts/run_synthetic_weak_alignment_v1.py verify",
            (
                ".venv/bin/python scripts/run_synthetic_weak_alignment_v1.py run-development "
                "--out /tmp/synthetic_weak_alignment_recovery_v1_reproduction"
            ),
            ".venv/bin/python -m pytest tests/test_synthetic_weak_alignment_v1.py -q",
            ".venv/bin/python -m pytest -q",
        ],
    }
    write_json(output / "reproducibility.json", provenance, refuse_overwrite=True)
    return {
        "protocol_id": PROTOCOL_ID,
        "recommendation": decision["recommendation_for_later_frozen_synthetic_confirmation"],
        "primary_effect": aggregate["primary_estimand"],
        "output_dir": str(output),
        "elapsed_seconds": elapsed,
        "all_audits_valid": all(value.get("valid") for value in audits.values()),
        "confirmation_outcomes_accessed": 0,
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
    freeze = verify_freeze_receipt(repository_root=root, receipt_path=freeze_receipt_path)
    confirmation = load_json(confirmation_manifest_path)
    manifest_checks = verify_confirmation_manifest(confirmation, config)
    main = _load_jsonl(output / "raw" / "development_runs.jsonl")
    sensitivity = _load_jsonl(output / "raw" / "sensitivity_runs.jsonl")
    guard_records_for_read_or_summary(main, config, operation="read")
    guard_records_for_read_or_summary(sensitivity, config, operation="read")
    provenance = load_json(output / "reproducibility.json")
    hash_checks = {
        relative: (output / relative).is_file()
        and sha256_file(output / relative) == expected
        for relative, expected in provenance["deterministic_artifact_hashes"].items()
    }
    audit_values = [load_json(path) for path in sorted((output / "audits").glob("*.json"))]
    checks = {
        "freeze": bool(freeze["valid"]),
        "confirmation_manifest": all(manifest_checks.values()),
        "artifact_hashes": all(hash_checks.values()),
        "all_audits_valid": all(value.get("valid") for value in audit_values),
        "main_run_count": len(main) == int(provenance["raw_main_runs"]),
        "sensitivity_run_count": len(sensitivity) == int(provenance["raw_sensitivity_runs"]),
        "no_confirmation_outcomes": provenance[
            "confirmation_outcomes_generated_trained_evaluated_read_or_summarized"
        ] == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"study integrity verification failed: {checks}")
    return {
        "valid": True,
        "checks": checks,
        "artifact_hash_checks": hash_checks,
        "recommendation": load_json(output / "terminal_decision.json")[
            "recommendation_for_later_frozen_synthetic_confirmation"
        ],
    }
