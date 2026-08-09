#!/usr/bin/env python3
"""Package validated v2 aggregate outcome files for the frozen report shell.

This bridge reads only repository-public aggregate artifacts plus a caller-
supplied SHA-256 of the owner-private authorization seal.  It never reads the
seal, bundle checkpoints, restricted data, per-seed rows, or free-form text.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_calibration_conditioned_symbolic_grounding_v2.py"
GENERATOR_PATH = ROOT / "scripts/generate_nursery_causal_central_figure_v1.py"
OUTPUT_ROOT = (ROOT / "output").resolve()
VERSION = "nursery-causal-outcome-reporting-packager-v1"

PROTOCOL_FILE_SHA256 = "c6b7afab6c8914a968b0f06706a222d2771863c2a3fd2017184c17d431bfa7d6"
CONTRACT_FILE_SHA256 = "db760f4135f81f7f78184a80f8abae16578918ff10b44c952f83a76cf5daa810"
CONTRACT_CANONICAL_SHA256 = "cd4ae4af338ae0414cbaa8327790637a6e55ae4a10a340f1554dd24cb0273426"
ERRATUM_CANONICAL_SHA256 = "679ba6956f36e42adcb30f415c0832380390d9201c87cb738a704fd6b007f4ab"
RUNNER_FILE_SHA256 = "8518daec3f3f6490d9eb8bca985523220c1091cb7ae7836fe85a5e728a7caab9"
OVERRIDE_SHA256 = "cb9e7061a5725050e6a71a7b6e41f6f5e7bf30d104bc46bae233816ca99a90b7"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
AUTH_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{15,79}$")


def load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError("E_MODULE_LOAD")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


runner = load_module("nursery_reporting_runner_v2", RUNNER_PATH)
figure = load_module("nursery_reporting_figure_v1", GENERATOR_PATH)

PREOUTCOME_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "fixture_only",
        "scientific_endpoints_opened",
        "calibration_binding_sha256",
        "execution_contract_sha256",
        "protocol_erratum_sha256",
        "runner_sha256",
        "gates",
        "all_gates_passed",
    }
)
AGGREGATE_KEYS = frozenset(
    {
        "schema_version",
        "fixture_only",
        "merge_order",
        "bundle_sha256s",
        "contrasts",
        "confidence_only_falsification",
        "side_only_leakage_falsification",
        "all_range_family_endpoint_comparator_gates_passed",
        "terminal_interpretation",
    }
)
RUNNER_CELL_KEYS = frozenset(
    {
        "corpus_count",
        "mean_effect",
        "one_sided_95_lower",
        "positive_corpus_fraction",
        "exact_one_sided_sign_p",
        "large_negative_fraction",
        "passed",
    }
)
EXECUTION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "authorization_id",
        "bundle_count",
        "deterministic_merge_sha256",
        "scientific_outcome_run",
        "cue_free_evaluation",
        "paired_complete_bundles",
    }
)


class PackagingError(RuntimeError):
    pass


def fail(code: str) -> None:
    raise PackagingError(code)


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        fail("E_CANONICAL_JSON")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                result.update(block)
    except OSError:
        fail("E_INPUT_READ")
    return result.hexdigest()


def read_json(path: Path, *, maximum: int = 64 * 1024 * 1024) -> Mapping[str, Any]:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > maximum
        ):
            fail("E_INPUT_FILE")
        value = json.loads(path.read_bytes())
    except PackagingError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        fail("E_INPUT_READ")
    if not isinstance(value, Mapping):
        fail("E_INPUT_SCHEMA")
    return value


def exact_keys(value: Any, expected: frozenset[str] | set[str], code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(str(key) for key in value) != frozenset(expected):
        fail(code)
    return value


def safe_repo_input(path: Path) -> None:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(OUTPUT_ROOT)
    except (OSError, ValueError):
        fail("E_INPUT_LOCATION")
    forbidden = {"quarantine", "restricted", ".external", "aea", "babyview"}
    if any(part.casefold() in forbidden for part in resolved.parts):
        fail("E_INPUT_LOCATION")


def safe_repo_output(path: Path) -> None:
    try:
        resolved_parent = path.parent.resolve(strict=True)
        resolved_parent.relative_to(OUTPUT_ROOT)
    except (OSError, ValueError):
        fail("E_OUTPUT_LOCATION")
    forbidden = {"quarantine", "restricted", ".external", "aea", "babyview"}
    if any(part.casefold() in forbidden for part in resolved_parent.parts):
        fail("E_OUTPUT_LOCATION")


def validate_calibration(document: Mapping[str, Any]) -> Any:
    try:
        binding = runner.bind_calibration_ranges_v2(document, fixture_only=False)
    except Exception:
        fail("E_CALIBRATION_SCHEMA")
    if (
        not runner._public_calibration_receipt_safe(document, binding)
        or document.get("no_automatic_fallback_override_sha256") != OVERRIDE_SHA256
        or document.get("automatic_fallback_allowed") is not False
        or document.get("scientific_endpoint_if_gemma_fails") is not False
        or tuple(document.get("instrument_paths", ())) != runner.REPLACEMENT_RANGE_FAMILIES[:2]
        or binding.gate_passed is not True
        or binding.status != "CALIBRATION_PASS"
    ):
        fail("E_CALIBRATION_GATE")
    return binding


def validate_preoutcome(document: Mapping[str, Any], calibration_binding_sha256: str) -> None:
    exact_keys(document, PREOUTCOME_KEYS, "E_PREOUTCOME_SCHEMA")
    gates = exact_keys(document.get("gates"), set(runner.PREOUTCOME_GATES), "E_PREOUTCOME_GATES")
    if (
        document.get("schema_version") != runner.PREOUTCOME_SCHEMA
        or document.get("status") != "PASS"
        or document.get("fixture_only") is not False
        or document.get("scientific_endpoints_opened") is not False
        or document.get("calibration_binding_sha256") != calibration_binding_sha256
        or document.get("execution_contract_sha256") != CONTRACT_CANONICAL_SHA256
        or document.get("protocol_erratum_sha256") != ERRATUM_CANONICAL_SHA256
        or document.get("runner_sha256") != RUNNER_FILE_SHA256
        or document.get("all_gates_passed") is not True
        or any(gates[key] is not True for key in runner.PREOUTCOME_GATES)
    ):
        fail("E_PREOUTCOME_GATE")


def transform_aggregate(document: Mapping[str, Any], preoutcome: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
    exact_keys(document, AGGREGATE_KEYS, "E_AGGREGATE_SCHEMA")
    merge_order = document.get("merge_order")
    bundle_hashes = document.get("bundle_sha256s")
    if (
        document.get("schema_version") != runner.MERGE_SCHEMA
        or document.get("fixture_only") is not False
        or merge_order != list(range(740001, 740080, 2))
        or not isinstance(bundle_hashes, list)
        or len(bundle_hashes) != 40
        or any(not isinstance(value, str) or HEX64.fullmatch(value) is None for value in bundle_hashes)
        or len(set(bundle_hashes)) != 40
    ):
        fail("E_AGGREGATE_INVENTORY")
    confidence = exact_keys(
        document.get("confidence_only_falsification"),
        {"confidence_endpoint_present", "rule", "top_mapping_endpoints_only"},
        "E_AGGREGATE_FALSIFICATION",
    )
    side = exact_keys(
        document.get("side_only_leakage_falsification"),
        {"maximum_above_chance", "required_pre_outcome"},
        "E_AGGREGATE_FALSIFICATION",
    )
    if (
        confidence.get("confidence_endpoint_present") is not False
        or confidence.get("rule")
        != "BOTH_NOUN_AND_VERB_WRONG_OR_TIED_TOP_MAPPINGS_MUST_BECOME_UNIQUELY_CORRECT"
        or tuple(confidence.get("top_mapping_endpoints_only", ())) != figure.ENDPOINTS
        or side.get("maximum_above_chance") != 0.02
        or side.get("required_pre_outcome") is not True
    ):
        fail("E_AGGREGATE_FALSIFICATION")

    contrasts = exact_keys(document.get("contrasts"), set(figure.FAMILIES), "E_AGGREGATE_FAMILY")
    projected: dict[str, Any] = {}
    all_passed = True
    for family in figure.FAMILIES:
        controls = exact_keys(contrasts[family], set(figure.COMPARATORS), "E_AGGREGATE_COMPARATOR")
        projected[family] = {}
        for comparator in figure.COMPARATORS:
            endpoints = exact_keys(controls[comparator], set(figure.ENDPOINTS), "E_AGGREGATE_ENDPOINT")
            projected[family][comparator] = {}
            for endpoint in figure.ENDPOINTS:
                cell = exact_keys(endpoints[endpoint], RUNNER_CELL_KEYS, "E_AGGREGATE_CELL")
                if cell.get("corpus_count") != 40 or isinstance(cell.get("corpus_count"), bool):
                    fail("E_AGGREGATE_CORPUS_COUNT")
                converted = {"synthetic_corpus_count": 40}
                for key in RUNNER_CELL_KEYS - {"corpus_count"}:
                    converted[key] = cell[key]
                projected[family][comparator][endpoint] = converted
                all_passed = all_passed and cell.get("passed") is True
    if (
        document.get("all_range_family_endpoint_comparator_gates_passed") is not all_passed
        or document.get("terminal_interpretation")
        != ("PROTOTYPE_PASS" if all_passed else "PROTOTYPE_STOP")
    ):
        fail("E_AGGREGATE_DECISION")

    gates = preoutcome["gates"]
    falsifications = {
        "confidence_only_rule_passed": gates["confidence_only_noun_wrong_or_tied_mapping_corrected"] is True
        and gates["confidence_only_verb_wrong_or_tied_mapping_corrected"] is True,
        "side_only_leakage_passed": gates["side_only_noun_at_most_chance_plus_002"] is True
        and gates["side_only_verb_at_most_chance_plus_002"] is True,
        "event_selection_disconnection_passed": gates["event_selection_disconnection_removes_correction"] is True,
    }
    if not all(falsifications.values()):
        fail("E_AGGREGATE_FALSIFICATION")
    return {"contrasts": projected, "falsifications": falsifications}, all_passed


def validate_execution(document: Mapping[str, Any], aggregate: Mapping[str, Any]) -> None:
    exact_keys(document, EXECUTION_KEYS, "E_EXECUTION_SCHEMA")
    authorization_id = document.get("authorization_id")
    if (
        document.get("schema_version") != runner.VERSION
        or document.get("status") != "SCIENTIFIC_ONE_SHOT_COMPLETE"
        or not isinstance(authorization_id, str)
        or AUTH_ID.fullmatch(authorization_id) is None
        or document.get("bundle_count") != 40
        or document.get("deterministic_merge_sha256") != digest(aggregate)
        or document.get("scientific_outcome_run") is not True
        or document.get("cue_free_evaluation") is not True
        or document.get("paired_complete_bundles") is not True
    ):
        fail("E_EXECUTION_GATE")


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
    except OSError:
        fail("E_OUTPUT_EXISTS_OR_UNSAFE")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        fail("E_OUTPUT_WRITE")


def package(
    *,
    calibration_path: Path,
    preoutcome_path: Path,
    authorization_seal_sha256: str,
    aggregate_path: Path,
    execution_path: Path,
    output_path: Path,
    enforce_repo_paths: bool = True,
) -> Mapping[str, Any]:
    inputs = (calibration_path, preoutcome_path, aggregate_path, execution_path)
    if enforce_repo_paths:
        for path in inputs:
            safe_repo_input(path)
        safe_repo_output(output_path)
    if not isinstance(authorization_seal_sha256, str) or HEX64.fullmatch(authorization_seal_sha256) is None:
        fail("E_AUTHORIZATION_DIGEST")
    if output_path.exists() or not output_path.parent.is_dir():
        fail("E_OUTPUT_EXISTS_OR_UNSAFE")

    calibration = read_json(calibration_path)
    preoutcome = read_json(preoutcome_path)
    aggregate = read_json(aggregate_path)
    execution = read_json(execution_path)
    binding = validate_calibration(calibration)
    validate_preoutcome(preoutcome, binding.digest)
    transformed, _ = transform_aggregate(aggregate, preoutcome)
    validate_execution(execution, aggregate)

    projection = {
        "schema_version": figure.INPUT_SCHEMA,
        "status": "SCIENTIFIC_AGGREGATE_VALIDATED",
        "bindings": {
            "no_automatic_fallback_override_sha256": OVERRIDE_SHA256,
            "protocol_sha256": PROTOCOL_FILE_SHA256,
            "execution_contract_sha256": CONTRACT_FILE_SHA256,
            "public_calibration_receipt_sha256": sha256_file(calibration_path),
            "preoutcome_receipt_sha256": sha256_file(preoutcome_path),
            "one_shot_authorization_sha256": authorization_seal_sha256,
            "aggregate_results_sha256": sha256_file(aggregate_path),
            "execution_receipt_sha256": sha256_file(execution_path),
        },
        "validity": {
            "gemma_path_passed": True,
            "automatic_fallback_used": False,
            "scientific_outcome_run": True,
            "fixture_only": False,
            "preoutcome_all_passed": True,
            "one_shot_complete": True,
            "complete_paired_bundles": True,
            "cue_free_evaluation": True,
            "simulator_oracle_only": True,
            "privacy_passed": True,
            "ancestry_passed": True,
            "restricted_payload_absent": True,
            "item_level_rows_absent": True,
        },
        "thresholds": dict(figure.THRESHOLDS),
        "contrasts": transformed["contrasts"],
        "falsifications": transformed["falsifications"],
    }
    figure.validate_projection(projection)
    exclusive_json(output_path, projection)
    return projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-receipt", required=True, type=Path)
    parser.add_argument("--preoutcome-receipt", required=True, type=Path)
    parser.add_argument("--authorization-seal-sha256", required=True)
    parser.add_argument("--aggregate-results", required=True, type=Path)
    parser.add_argument("--execution-receipt", required=True, type=Path)
    parser.add_argument("--output-projection", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        value = package(
            calibration_path=args.calibration_receipt,
            preoutcome_path=args.preoutcome_receipt,
            authorization_seal_sha256=args.authorization_seal_sha256,
            aggregate_path=args.aggregate_results,
            execution_path=args.execution_receipt,
            output_path=args.output_projection,
        )
        decision, passing, _ = figure.validate_projection(value)
    except PackagingError as error:
        print(str(error), file=sys.stderr)
        return 2
    except figure.ReportingError:
        print("E_REPORTING_PROJECTION", file=sys.stderr)
        return 2
    print(json.dumps({"status": "PROJECTION_WRITTEN", "decision": decision, "passing_cells": passing}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
