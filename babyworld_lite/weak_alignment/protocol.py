from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


PROTOCOL_ID = "synthetic-weak-alignment-recovery-v1"
SCHEMA_VERSION = "synthetic-weak-alignment-protocol-v1"
AUTHORIZATION_SCHEMA = "synthetic-confirmation-authorization-v1"
RESERVED_OPERATIONS = frozenset({"generate", "train", "evaluate", "read", "summarize"})


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value: Any, *, refuse_overwrite: bool = False) -> None:
    output = Path(path)
    if refuse_overwrite and output.exists():
        raise FileExistsError(f"refusing to overwrite preserved artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _as_number_list(value: Any, name: str, cast: type) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = [cast(item) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def validate_protocol_config(config: Mapping[str, Any]) -> None:
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a mapping")
    if protocol.get("id") != PROTOCOL_ID:
        raise ValueError(f"protocol.id must be {PROTOCOL_ID!r}")
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"protocol.schema_version must be {SCHEMA_VERSION!r}")
    if protocol.get("status") != "frozen":
        raise ValueError("protocol.status must be 'frozen'")
    if protocol.get("study_phase") != "development_only":
        raise ValueError("this implementation only permits study_phase=development_only")

    seeds = config.get("seeds")
    if not isinstance(seeds, Mapping):
        raise ValueError("seeds must be a mapping")
    dev = seeds.get("development")
    reserve = seeds.get("confirmation_reserve")
    if not isinstance(dev, Mapping) or not isinstance(reserve, Mapping):
        raise ValueError("development and confirmation_reserve seed mappings are required")
    dev_corpus = _as_number_list(dev.get("corpus"), "development corpus seeds", int)
    dev_model = _as_number_list(dev.get("model"), "development model seeds", int)
    reserve_corpus = _as_number_list(
        reserve.get("corpus"), "confirmation corpus seeds", int
    )
    reserve_model = _as_number_list(
        reserve.get("model"), "confirmation model seeds", int
    )
    if set(dev_corpus) & set(reserve_corpus):
        raise ValueError("development and confirmation corpus seeds overlap")
    if set(dev_model) & set(reserve_model):
        raise ValueError("development and confirmation model seeds overlap")

    factors = config.get("factors")
    if not isinstance(factors, Mapping):
        raise ValueError("factors must be a mapping")
    expected = {
        "speech_action_lag",
        "grounded_utterance_rate",
        "candidate_event_count",
        "action_visibility_rate",
        "word_occurrence_count",
        "side_informativeness",
    }
    if set(factors) != expected:
        raise ValueError(f"factors must contain exactly {sorted(expected)}")
    lags = _as_number_list(factors["speech_action_lag"], "speech_action_lag", float)
    grounded = _as_number_list(
        factors["grounded_utterance_rate"], "grounded_utterance_rate", float
    )
    candidates = _as_number_list(
        factors["candidate_event_count"], "candidate_event_count", int
    )
    visibility = _as_number_list(
        factors["action_visibility_rate"], "action_visibility_rate", float
    )
    repetitions = _as_number_list(
        factors["word_occurrence_count"], "word_occurrence_count", int
    )
    information = _as_number_list(
        factors["side_informativeness"], "side_informativeness", float
    )
    if len(lags) < 3 or not any(value < 0 for value in lags) or not any(value > 0 for value in lags):
        raise ValueError("speech_action_lag must contain lead, aligned, and lag values")
    if any(not 0 < value < 1 for value in grounded + visibility):
        raise ValueError("grounded and visibility rates must lie strictly between zero and one")
    if any(value < 2 for value in candidates):
        raise ValueError("candidate_event_count values must be at least two")
    if len(repetitions) != 2 or any(value < 2 for value in repetitions):
        raise ValueError("word_occurrence_count must contain two values of at least two")
    if 0.0 not in information or any(not 0 <= value <= 1 for value in information):
        raise ValueError("side_informativeness must include 0.0 and remain in [0, 1]")

    design = config.get("design")
    learners = config.get("learners")
    conditions = config.get("conditions")
    evaluation = config.get("evaluation")
    inference = config.get("inference")
    decision = config.get("decision_rule")
    for name, value in (
        ("design", design),
        ("learners", learners),
        ("conditions", conditions),
        ("evaluation", evaluation),
        ("inference", inference),
        ("decision_rule", decision),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")
    if int(design.get("action_count", 0)) != 6 or int(design.get("object_count", 0)) != 6:
        raise ValueError("v1 requires exactly six action and six object concepts")
    if int(design.get("lexical_panels", 0)) != 3:
        raise ValueError("v1 requires three balanced lexical panels")
    if int(design.get("lexemes_per_panel", 0)) != 6:
        raise ValueError("each lexical panel must contain six lexemes")
    if set(conditions.get("training_side_modality", [])) != {
        "synchronized",
        "shuffled",
        "time_shifted",
        "absent",
        "uninformative",
    }:
        raise ValueError("the five required training side-modality conditions are mandatory")
    required_learners = {
        "exact_window",
        "latent_mil_single_occurrence",
        "cross_situational_uniform",
        "latent_mil_cross_occurrence",
        "latent_mil_cross_no_null",
        "oracle_alignment",
    }
    if set(learners.get("names", [])) != required_learners:
        raise ValueError("learner inventory does not match the frozen v1 comparison")
    if evaluation.get("primary_learner") != "latent_mil_cross_occurrence":
        raise ValueError("v1 primary learner must be latent_mil_cross_occurrence")
    if evaluation.get("primary_metric") != "heldout_composition_action_6way_macro_accuracy":
        raise ValueError("unexpected v1 primary metric")
    if int(inference.get("bootstrap_samples", 0)) < 1000:
        raise ValueError("bootstrap_samples must be at least 1000")
    if decision.get("primary_contrast") != ["synchronized", "shuffled"]:
        raise ValueError("primary contrast must be synchronized minus shuffled")


def load_protocol_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ValueError("protocol config must be a YAML mapping")
    config = dict(loaded)
    validate_protocol_config(config)
    return config


def reserve_seed_sets(config: Mapping[str, Any]) -> tuple[set[int], set[int]]:
    reserve = config["seeds"]["confirmation_reserve"]
    return set(map(int, reserve["corpus"])), set(map(int, reserve["model"]))


def _authorization_valid(
    authorization_path: str | Path | None,
    *,
    manifest_path: str | Path | None,
    freeze_receipt_path: str | Path | None,
) -> bool:
    if authorization_path is None:
        return False
    path = Path(authorization_path)
    if not path.is_file() or manifest_path is None or freeze_receipt_path is None:
        return False
    value = load_json(path)
    return bool(
        isinstance(value, Mapping)
        and value.get("schema_version") == AUTHORIZATION_SCHEMA
        and value.get("protocol_id") == PROTOCOL_ID
        and value.get("explicit_future_confirmation_authorization") is True
        and value.get("authorized_by_user") is True
        and value.get("confirmation_manifest_sha256") == sha256_file(manifest_path)
        and value.get("freeze_receipt_sha256") == sha256_file(freeze_receipt_path)
    )


def guard_seed_operation(
    config: Mapping[str, Any],
    *,
    operation: str,
    corpus_seed: int | None = None,
    model_seed: int | None = None,
    authorization_path: str | Path | None = None,
    confirmation_manifest_path: str | Path | None = None,
    freeze_receipt_path: str | Path | None = None,
) -> None:
    if operation not in RESERVED_OPERATIONS:
        raise ValueError(f"unknown guarded operation: {operation}")
    reserve_corpus, reserve_model = reserve_seed_sets(config)
    touches_reserve = (
        corpus_seed is not None and int(corpus_seed) in reserve_corpus
    ) or (model_seed is not None and int(model_seed) in reserve_model)
    if touches_reserve and not _authorization_valid(
        authorization_path,
        manifest_path=confirmation_manifest_path,
        freeze_receipt_path=freeze_receipt_path,
    ):
        raise PermissionError(
            f"confirmation reserve guard blocked {operation}: no valid future explicit authorization"
        )


def guard_records_for_read_or_summary(
    records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    operation: str,
) -> None:
    if operation not in {"read", "summarize"}:
        raise ValueError("record guard operation must be read or summarize")
    for record in records:
        corpus_seed = record.get("corpus_seed")
        model_seed = record.get("model_seed")
        guard_seed_operation(
            config,
            operation=operation,
            corpus_seed=None if corpus_seed is None else int(corpus_seed),
            model_seed=None if model_seed is None else int(model_seed),
        )


def make_confirmation_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    reserve = config["seeds"]["confirmation_reserve"]
    return {
        "schema_version": "synthetic-confirmation-reserve-manifest-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "reserved_untouched_no_outcomes",
        "corpus_seeds": list(map(int, reserve["corpus"])),
        "model_seeds": list(map(int, reserve["model"])),
        "outcome_fields": [],
        "authorization_present": False,
        "authorized_operations": [],
        "note": (
            "Identifiers are preregistered only. This manifest contains no generated corpus, "
            "model, evaluation, or summary outcome and grants no confirmation authorization."
        ),
    }


def verify_confirmation_manifest(
    manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, bool]:
    expected = make_confirmation_manifest(config)
    checks = {
        "protocol_id": manifest.get("protocol_id") == PROTOCOL_ID,
        "status": manifest.get("status") == "reserved_untouched_no_outcomes",
        "corpus_seeds": manifest.get("corpus_seeds") == expected["corpus_seeds"],
        "model_seeds": manifest.get("model_seeds") == expected["model_seeds"],
        "no_outcome_fields": manifest.get("outcome_fields") == [],
        "no_authorization": manifest.get("authorization_present") is False
        and manifest.get("authorized_operations") == [],
    }
    if not all(checks.values()):
        raise ValueError(f"invalid confirmation reserve manifest: {checks}")
    return checks


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def create_freeze_receipt(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    protocol_path: str | Path,
    sources_path: str | Path,
    confirmation_manifest_path: str | Path,
    tracked_paths: Sequence[str | Path],
    output_path: str | Path,
    created_at_utc: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = load_protocol_config(config_path)
    manifest = load_json(confirmation_manifest_path)
    verify_confirmation_manifest(manifest, config)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite freeze receipt: {output}")
    paths = [Path(config_path), Path(protocol_path), Path(sources_path), Path(confirmation_manifest_path)]
    paths.extend(Path(item) for item in tracked_paths)
    resolved = [path if path.is_absolute() else root / path for path in paths]
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze missing inputs: {missing}")
    hashes = {
        str(path.resolve().relative_to(root)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in resolved
    }
    status = _git(root, "status", "--porcelain=v1")
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    receipt = {
        "schema_version": "synthetic-weak-alignment-freeze-receipt-v1",
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": created_at_utc,
        "phase_at_freeze": "pre_outcome_development_protocol_freeze",
        "content_hashes": hashes,
        "confirmation_manifest_sha256": sha256_file(confirmation_manifest_path),
        "code_provenance": {
            "git_head": _git(root, "rev-parse", "HEAD"),
            "git_branch": _git(root, "branch", "--show-current"),
            "git_status_porcelain_at_freeze": status.splitlines() if status else [],
            "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        },
        "runtime_provenance": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pid": os.getpid(),
        },
        "outcome_producing_runs_before_freeze": 0,
        "accidental_pre_freeze_outcome_incidents": [],
    }
    write_json(output, receipt, refuse_overwrite=True)
    return receipt


def verify_freeze_receipt(
    *, repository_root: str | Path, receipt_path: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    receipt = load_json(receipt_path)
    if receipt.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("freeze receipt protocol ID mismatch")
    checks: dict[str, bool] = {}
    for relative, metadata in receipt.get("content_hashes", {}).items():
        path = root / relative
        checks[relative] = path.is_file() and sha256_file(path) == metadata.get("sha256")
    checks["all_frozen_files_present"] = bool(receipt.get("content_hashes"))
    if not all(checks.values()):
        failures = sorted(key for key, passed in checks.items() if not passed)
        raise ValueError(f"protocol freeze verification failed: {failures}")
    return {"valid": True, "checks": checks, "receipt_sha256": sha256_file(receipt_path)}
