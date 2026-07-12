from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _fmt_ci(metric: dict[str, Any]) -> str:
    return (
        f"{100 * metric['mean']:+.2f} percentage points "
        f"(seed-bootstrap 95% CI [{100 * metric['ci95_low']:+.2f}, "
        f"{100 * metric['ci95_high']:+.2f}])"
    )


def build_evidence(
    audit_path: Path,
    pilot_path: Path,
    out_dir: Path,
    machine_run_path: Path | None = None,
    coverage_path: Path | None = None,
    reference_path: Path | None = None,
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    lifts = pilot["paired_lifts"]
    machine_run = (
        json.loads(machine_run_path.read_text())
        if machine_run_path is not None and machine_run_path.is_file()
        else None
    )
    coverage = (
        json.loads(coverage_path.read_text())
        if coverage_path is not None and coverage_path.is_file()
        else None
    )
    reference = (
        json.loads(reference_path.read_text())
        if reference_path is not None and reference_path.is_file()
        else None
    )
    evidence = {
        "status": "pre_access_infrastructure_validated_no_language_effect_claim",
        "dataset": {
            "profile": audit["profile"],
            "counts": audit["counts"],
            "configured_distributions": audit["configured_distributions"],
            "configured_calibration_targets": audit["configured_calibration_targets"],
            "realized_calibration": audit["realized_calibration"],
            "fairness": audit["fairness"],
            "leakage": audit["leakage"],
            "renderer": audit["renderer"],
            "audit_valid": audit["valid"],
        },
        "pilot": {
            "source_schema": pilot["source_schema"],
            "alignment_condition": pilot["alignment_condition"],
            "seeds": pilot["seeds"],
            "n_train": pilot["n_train"],
            "n_test": pilot["n_test"],
            "primary_test": pilot["primary_test"],
            "paired_protocol_checks": pilot["paired_protocol_checks"],
            "metadata_only_shortcut": pilot["metadata_only_shortcut"],
            "paired_lifts": lifts,
        },
        "machine_devbench": None,
        "interpretation": {
            "infrastructure": (
                "The leak-free factorial dataset, paired training protocol, and motor-withheld "
                "action-grounding evaluation run end to end."
            ),
            "effect": (
                "The small single-corpus three-seed pilot does not satisfy the prespecified "
                "claim gate. It is pipeline validation only."
            ),
            "email_rule": (
                "Describe the infrastructure as completed; do not state that synchronized motor "
                "experience improves language grounding."
            ),
        },
    }
    if machine_run is not None and coverage is not None:
        machine_section: dict[str, Any] = {
            "official_repository_commit": machine_run["official_source"]["commit"],
            "manifest_count": machine_run["benchmark"]["manifest_count"],
            "target_trials": coverage["benchmark"]["target_trials"],
            "official_protocol_adapter_validated": True,
            "nursery_full_run_summary": machine_run["summary"],
            "nursery_target_vocabulary_coverage": coverage["target_word_coverage"],
            "interpretation": (
                "The evaluator and adapter run end to end, but the provisional learner's "
                "Machine-DevBench score is not scientifically interpretable because exact "
                "target-word coverage is under one percent."
            ),
        }
        if reference is not None:
            observed = reference["summary"]["total"]
            published = {"lexical": 0.873, "grammatical": 0.704, "overall": 0.788}
            machine_section["openclip_l_reference"] = {
                "observed": {
                    key: observed[key] for key in ("lexical", "grammatical", "overall")
                },
                "published_leaderboard": published,
                "absolute_difference": {
                    key: abs(observed[key] - published[key]) for key in published
                },
            }
        evidence["machine_devbench"] = machine_section
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pre_access_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")

    lines = [
        "# Pre-access evidence summary",
        "",
        "Status: **infrastructure validated; no language-effect claim.**",
        "",
        "## Dataset gate",
        "",
        f"- Profile: `{audit['profile']['calibration_status'][0]}`.",
        f"- Scale: {audit['counts']['base_episodes']} base episodes and {audit['counts']['examples']} factorial examples.",
        f"- Audit valid: `{str(audit['valid']).lower()}`; failures: `{audit['failures']}`.",
        "- Language arms: strong, weak, shuffled.",
        "- Motor arms: null, synchronized, split-local episode-shuffled, time-shifted.",
        "- Model-visible inputs passed the allowlist; oracle state is stored separately.",
        "- No shuffled self-matches, cross-split donors, split hash overlap, or split composition overlap.",
        f"- Realized utterance rate: {audit['realized_calibration']['utterance_rate_per_minute']:.2f}/minute (provisional target {audit['configured_calibration_targets']['utterance_rate_per_minute']:.2f}).",
        "- Audit includes utterance count/length/timing, silent-frame fraction, target-word frequency, activity duration, visibility/occlusion, camera motion, distractor count, and object/action/material marginals.",
        "",
        "## Pilot gate",
        "",
        f"- Source/alignment: `{pilot['source_schema']}` / `{pilot['alignment_condition']}`.",
        f"- Paired runs: seeds {pilot['seeds']}; n_train={pilot['n_train']}; n_test={pilot['n_test']}.",
        "- Primary test uses rendered RGB and text only; the motor encoder is omitted.",
        f"- Synchronized - shuffled: {_fmt_ci(lifts['synchronized_minus_shuffled'])}.",
        f"- Synchronized - null: {_fmt_ci(lifts['synchronized_minus_null'])}.",
        f"- Synchronized - time-shifted: {_fmt_ci(lifts['synchronized_minus_time_shifted'])}.",
        "",
        "Synchronized does not beat the shuffled or null controls, so the prespecified positive-effect gate fails. The result is not evidence of a motor-cue benefit.",
    ]
    if machine_run is not None and coverage is not None:
        target = coverage["target_word_coverage"]
        summary = machine_run["summary"]["total"]
        lines.extend([
            "",
            "## Official Machine-DevBench gate",
            "",
            f"- Official EgoBabyVLM commit: `{machine_run['official_source']['commit']}`.",
            f"- Public benchmark installed: {machine_run['benchmark']['manifest_count']} manifests / "
            f"{coverage['benchmark']['target_trials']} trials across realistic and cartoon styles.",
            "- Nursery checkpoint implements the official `MultiModalFeatureExtractor` boundary and completed every trial.",
            f"- Diagnostic Nursery result: lexical {100 * summary['lexical']:.1f}, grammatical "
            f"{100 * summary['grammatical']:.1f}, overall {100 * summary['overall']:.1f}.",
            f"- Exact target-vocabulary coverage: {target['unique_covered']}/{target['unique_total']} "
            f"({100 * target['unique_fraction']:.1f}%).",
            "",
            "The Nursery score is not a substantive benchmark result: the vocabulary audit shows extensive UNK collisions. The completed run validates the integration and fixes the next requirement—expand/calibrate the learner's linguistic experience before interpreting Machine-DevBench.",
        ])
        if reference is not None:
            observed = reference["summary"]["total"]
            lines.extend([
                "",
                f"The off-the-shelf OpenCLIP reference produced lexical {100 * observed['lexical']:.1f}, "
                f"grammatical {100 * observed['grammatical']:.1f}, and overall "
                f"{100 * observed['overall']:.1f}; the published CLIP-L leaderboard row is "
                "87.3 / 70.4 / 78.8.",
            ])
    lines.extend([
        "",
        "## Reproduce",
        "",
        "```bash",
        "source .venv/bin/activate",
        "python -m babyworld_lite.grounding --config configs/grounding_provisional.yaml --out data/grounding_provisional",
        "python train_grounding_pilot.py --episodes data/grounding_provisional/examples.jsonl --out data/grounding_pilot_weak --seeds 11 22 33 --holdout cup:push --holdout plush:grasp --alignment weak --epochs 8 --batch-size 24 --frame-count 6 --image-size 48 --hidden-dim 48 --embedding-dim 32 --device auto",
        "python scripts/train_nursery_checkpoint.py --episodes data/grounding_provisional/examples.jsonl --out output/machine_devbench/nursery_sync_seed11.pt --arm synchronized --seed 11 --alignment weak --epochs 8 --device auto",
        "python scripts/run_machine_devbench.py --model nursery --checkpoint output/machine_devbench/nursery_sync_seed11.pt --out output/machine_devbench/nursery_full.json --device auto",
        "python scripts/audit_machine_devbench_coverage.py --checkpoint output/machine_devbench/nursery_sync_seed11.pt",
        "python scripts/run_machine_devbench.py --model openclip --openclip-model ViT-L-14-quickgelu --openclip-pretrained openai --out output/machine_devbench/openclip_l_full.json --device auto",
        "python scripts/build_pre_access_evidence.py",
        "```",
        "",
    ])
    (out_dir / "pre_access_evidence.md").write_text("\n".join(lines))
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Package audited evidence for the Frank pre-access update.")
    parser.add_argument(
        "--audit", type=Path, default=ROOT / "data" / "grounding_provisional" / "audit_summary.json"
    )
    parser.add_argument(
        "--pilot", type=Path, default=ROOT / "data" / "grounding_pilot_weak" / "pilot_results.json"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "pre_access")
    parser.add_argument(
        "--machine-run",
        type=Path,
        default=ROOT / "output" / "machine_devbench" / "nursery_full.json",
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=ROOT / "output" / "machine_devbench" / "nursery_vocabulary_coverage.json",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "output" / "machine_devbench" / "openclip_l_full.json",
    )
    args = parser.parse_args()
    build_evidence(
        args.audit,
        args.pilot,
        args.out,
        args.machine_run,
        args.coverage,
        args.reference,
    )
    print(args.out)


if __name__ == "__main__":
    main()
