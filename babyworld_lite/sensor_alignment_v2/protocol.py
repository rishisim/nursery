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


PROTOCOL_ID = "synthetic-sensor-event-robustness-v2"
SCHEMA_VERSION = "synthetic-sensor-event-protocol-v2"
RESERVED_OPERATIONS = frozenset(
    {"generate", "calibrate", "train", "evaluate", "read", "summarize"}
)


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


def write_jsonl(
    path: str | Path,
    values: Iterable[Mapping[str, Any]],
    *,
    refuse_overwrite: bool = False,
) -> None:
    output = Path(path)
    if refuse_overwrite and output.exists():
        raise FileExistsError(f"refusing to overwrite preserved artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in values))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def _number_list(value: Any, name: str, cast: type) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = [cast(item) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def validate_protocol_config(
    config: Mapping[str, Any], *, require_development_size: bool = True
) -> None:
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a mapping")
    if protocol.get("id") != PROTOCOL_ID:
        raise ValueError(f"protocol.id must be {PROTOCOL_ID!r}")
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"protocol.schema_version must be {SCHEMA_VERSION!r}")
    if protocol.get("status") != "frozen":
        raise ValueError("protocol.status must be frozen")
    if protocol.get("study_phase") != "development_only":
        raise ValueError("v2 permits development_only runs")
    if protocol.get("confirmation_authorized") is not False:
        raise ValueError("v2 confirmation must remain unauthorized")

    seeds = config.get("seeds")
    if not isinstance(seeds, Mapping):
        raise ValueError("seeds must be a mapping")
    development = seeds.get("development")
    calibration = seeds.get("generic_calibration")
    reserve = seeds.get("confirmation_reserve")
    fixture = seeds.get("fixture_only")
    forbidden = seeds.get("v1_forbidden")
    if not all(isinstance(value, Mapping) for value in (
        development, calibration, reserve, fixture, forbidden
    )):
        raise ValueError("all seed registries are required")
    dev_corpus = _number_list(development.get("corpus"), "development corpus", int)
    dev_model = _number_list(development.get("model"), "development model", int)
    cal_train = _number_list(calibration.get("train"), "calibration train", int)
    cal_validation = _number_list(
        calibration.get("validation"), "calibration validation", int
    )
    reserve_corpus = _number_list(reserve.get("corpus"), "reserve corpus", int)
    reserve_model = _number_list(reserve.get("model"), "reserve model", int)
    reserve_calibration = _number_list(
        reserve.get("calibration"), "reserve calibration", int
    )
    if require_development_size and len(dev_corpus) < 20:
        raise ValueError("at least 20 independent development corpus seeds are required")
    if require_development_size and len(dev_model) < 2:
        raise ValueError("at least two algorithmic model replicates are required")
    all_active = set(dev_corpus + dev_model + cal_train + cal_validation)
    all_reserve = set(reserve_corpus + reserve_model + reserve_calibration)
    all_fixture = {int(value) for value in fixture.values()}
    all_v1 = set(map(int, forbidden.get("corpus", []))) | set(
        map(int, forbidden.get("model", []))
    )
    if len(all_active) != len(dev_corpus + dev_model + cal_train + cal_validation):
        raise ValueError("active seed namespaces must be numerically disjoint")
    if require_development_size:
        if all_active & all_reserve or all_active & all_fixture or all_active & all_v1:
            raise ValueError("active v2 seeds overlap a reserve, fixture, or v1 seed")
        if all_reserve & all_fixture or all_reserve & all_v1:
            raise ValueError("v2 reserve overlaps fixture or v1 seed registry")
    else:
        if all_active & all_reserve:
            raise ValueError("fixture active and fixture reserve seeds must be disjoint")
        if not (all_active | all_reserve) <= all_fixture:
            raise ValueError("fixture protocol may use only explicitly registered fixture seeds")
        if (all_active | all_reserve) & all_v1:
            raise ValueError("fixture protocol overlaps v1 seeds")
    if all_fixture & all_v1:
        raise ValueError("v2 fixture seeds overlap v1 seeds")

    required_factors = {
        "speech_action_lag_samples",
        "grounded_utterance_rate",
        "candidate_event_count",
        "action_visibility_rate",
        "lexical_repetition_count",
        "sensor_informativeness",
        "sensor_snr",
        "sensor_dropout_rate",
        "false_positive_sensor_rate",
        "wearer_event_prevalence",
    }
    factors = config.get("factors")
    if not isinstance(factors, Mapping) or set(factors) != required_factors:
        raise ValueError(f"factors must contain exactly {sorted(required_factors)}")
    for name, levels in factors.items():
        _number_list(levels, name, float)
    lags = list(map(int, factors["speech_action_lag_samples"]))
    if not any(value < 0 for value in lags) or 0 not in lags or not any(
        value > 0 for value in lags
    ):
        raise ValueError("lag levels must include lead, aligned, and lag")
    if 0.0 not in list(map(float, factors["sensor_informativeness"])):
        raise ValueError("sensor_informativeness must include zero")
    if any(int(value) < 2 for value in factors["candidate_event_count"]):
        raise ValueError("candidate event counts must be at least two")

    conditions = config.get("conditions")
    expected_conditions = {
        "synchronized",
        "shuffled",
        "shifted_m16",
        "shifted_m8",
        "shifted_p8",
        "shifted_p16",
        "absent",
        "uninformative",
    }
    if not isinstance(conditions, Mapping) or set(conditions.get("names", [])) != expected_conditions:
        raise ValueError("the eight frozen sensor conditions are mandatory")
    offsets = conditions.get("time_shift_offsets")
    if not isinstance(offsets, Mapping) or set(offsets) != expected_conditions - {
        "synchronized", "shuffled", "absent", "uninformative"
    }:
        raise ValueError("all four signed time-shift offsets are required")
    if sorted(map(int, offsets.values())) != [-16, -8, 8, 16]:
        raise ValueError("unexpected time-shift offsets")

    learners = config.get("learners")
    required_learners = {
        "exact_window_symbolic",
        "sensor_latent_single_occurrence",
        "cross_occurrence_no_sensor",
        "sensor_latent_cross_occurrence",
        "sensor_latent_cross_no_null",
        "structural_absence",
        "oracle_event_alignment_upper",
        "v1_pointer_style_upper",
    }
    if not isinstance(learners, Mapping) or set(learners.get("names", [])) != required_learners:
        raise ValueError("learner inventory does not match the frozen v2 comparison")
    if learners.get("primary") != "sensor_latent_cross_occurrence":
        raise ValueError("unexpected primary learner")

    evaluation = config.get("evaluation")
    inference = config.get("inference")
    gates = config.get("gates")
    if not all(isinstance(value, Mapping) for value in (evaluation, inference, gates)):
        raise ValueError("evaluation, inference, and gates mappings are required")
    if evaluation.get("primary_metric") != (
        "heldout_object_action_composition_action_6way_macro_accuracy"
    ):
        raise ValueError("unexpected primary endpoint")
    if inference.get("independent_unit") != "corpus_seed":
        raise ValueError("corpus seed must be the independent unit")
    if inference.get("multiplicity") != (
        "intersection_union_both_one_sided_tests_must_pass_no_alpha_splitting"
    ):
        raise ValueError("v2 requires the frozen intersection-union rule")
    if inference.get("bootstrap_primary_inference") != "forbidden":
        raise ValueError("generic bootstrap primary inference is forbidden")


def load_protocol_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("protocol config must be a YAML mapping")
    config = dict(value)
    validate_protocol_config(config)
    return config


def reserve_seed_set(config: Mapping[str, Any]) -> set[int]:
    reserve = config["seeds"]["confirmation_reserve"]
    return {
        *map(int, reserve["corpus"]),
        *map(int, reserve["model"]),
        *map(int, reserve["calibration"]),
    }


def guard_seed_operation(
    config: Mapping[str, Any], *, operation: str, seeds: Sequence[int | None]
) -> None:
    if operation not in RESERVED_OPERATIONS:
        raise ValueError(f"unknown guarded operation: {operation}")
    reserved = reserve_seed_set(config)
    touched = sorted({int(seed) for seed in seeds if seed is not None} & reserved)
    if touched:
        raise PermissionError(
            f"v2 confirmation reserve guard blocked {operation} for {touched}; "
            "this development task has no confirmation authorization path"
        )


def guard_records(
    records: Iterable[Mapping[str, Any]], config: Mapping[str, Any], *, operation: str
) -> None:
    if operation not in {"read", "summarize"}:
        raise ValueError("record guards only accept read or summarize")
    fields = ("corpus_seed", "model_seed", "calibration_seed")
    for record in records:
        guard_seed_operation(
            config,
            operation=operation,
            seeds=[record.get(field) for field in fields],
        )


def make_confirmation_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    reserve = config["seeds"]["confirmation_reserve"]
    return {
        "schema_version": "synthetic-sensor-event-confirmation-reserve-v2",
        "protocol_id": PROTOCOL_ID,
        "status": "reserved_untouched_identifiers_only",
        "corpus_seeds": list(map(int, reserve["corpus"])),
        "model_seeds": list(map(int, reserve["model"])),
        "calibration_seeds": list(map(int, reserve["calibration"])),
        "outcome_fields": [],
        "authorization_present": False,
        "authorized_operations": [],
        "note": (
            "Identifiers only: no reserve corpus, raw stream, detector, model, evaluation, "
            "or summary has been generated or read, and no confirmation is authorized."
        ),
    }


def verify_confirmation_manifest(
    manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, bool]:
    expected = make_confirmation_manifest(config)
    checks = {
        "protocol_id": manifest.get("protocol_id") == PROTOCOL_ID,
        "status": manifest.get("status") == "reserved_untouched_identifiers_only",
        "corpus_seeds": manifest.get("corpus_seeds") == expected["corpus_seeds"],
        "model_seeds": manifest.get("model_seeds") == expected["model_seeds"],
        "calibration_seeds": manifest.get("calibration_seeds")
        == expected["calibration_seeds"],
        "no_outcomes": manifest.get("outcome_fields") == [],
        "no_authorization": manifest.get("authorization_present") is False
        and manifest.get("authorized_operations") == [],
    }
    if not all(checks.values()):
        raise ValueError(f"invalid v2 confirmation manifest: {checks}")
    return checks


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


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
    paths = [
        Path(config_path),
        Path(protocol_path),
        Path(sources_path),
        Path(confirmation_manifest_path),
        *(Path(item) for item in tracked_paths),
    ]
    resolved = [path if path.is_absolute() else root / path for path in paths]
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze missing inputs: {missing}")
    content_hashes = {
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
        "schema_version": "synthetic-sensor-event-freeze-receipt-v2",
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": created_at_utc,
        "phase_at_freeze": "pre_outcome_development_protocol_freeze",
        "content_hashes": content_hashes,
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
        "fixture_only_smoke_checks_before_freeze": True,
        "accidental_pre_freeze_outcome_incidents": [],
        "protocol_amendments": [],
    }
    write_json(output, receipt, refuse_overwrite=True)
    return receipt


def verify_freeze_receipt(
    *, repository_root: str | Path, receipt_path: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    receipt = load_json(receipt_path)
    if receipt.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("v2 freeze receipt protocol ID mismatch")
    checks: dict[str, bool] = {}
    for relative, metadata in receipt.get("content_hashes", {}).items():
        path = root / relative
        checks[relative] = path.is_file() and sha256_file(path) == metadata.get("sha256")
    checks["all_frozen_files_present"] = bool(receipt.get("content_hashes"))
    checks["zero_pre_freeze_outcomes"] = (
        receipt.get("outcome_producing_runs_before_freeze") == 0
    )
    if not all(checks.values()):
        failures = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"v2 protocol freeze verification failed: {failures}")
    return {
        "valid": True,
        "checks": checks,
        "receipt_sha256": sha256_file(receipt_path),
    }
