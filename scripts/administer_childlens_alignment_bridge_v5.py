#!/usr/bin/env python3
"""Create an owner-private, exact-18 ChildLens v5 scientific input.

This is an administrative metadata-only process.  It is intentionally
separate from the scientific v5 runner and is the only v5 code allowed to open
the legacy mixed-scope membership/release manifests.  It never opens media,
annotation contents, embeddings, model results, or learned weights.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/childlens_alignment_bridge_v5.json"
INCIDENT = ROOT / "output/childlens_alignment_bridge_v5/stage0_freeze_and_stop_receipt.json"
PUBLIC_RECEIPT = (
    ROOT
    / "output/childlens_alignment_bridge_v5/"
    "administrative_correction_receipt.json"
)
PRIVATE_RELATIVE = Path(
    "provisional_calibration_v1/childlens_alignment_bridge_v5/administrative"
)
PRIVATE_INPUT = PRIVATE_RELATIVE / "development_only_scientific_input.json"
PRIVATE_ATTESTATION = PRIVATE_RELATIVE / "development_only_attestation.json"

SOURCE_RELATIVE = {
    "v4_exact_18": Path(
        "provisional_calibration_v1/childlens_alignment_bridge_v4/"
        "restricted_split_window_manifest.json"
    ),
    "original_8": Path(
        "provisional_calibration_v1/childlens_alignment_bridge_remediation_v2/"
        "restricted_development_manifest.json"
    ),
    "expansion_10": Path(
        "provisional_calibration_v1/childlens_alignment_bridge_expansion_v3/"
        "restricted_measurement_manifest.json"
    ),
    "expansion_10_plan": Path(
        "provisional_calibration_v1/childlens_alignment_bridge_expansion_v3/"
        "restricted_expansion_plan.json"
    ),
    "v1_mixed_split": Path(
        "provisional_calibration_v1/childlens_alignment_bridge_v1/"
        "restricted_split_manifest.json"
    ),
    "canonical_release": Path("preselection_manifest.json"),
}
EXPECTED_SOURCE_HASHES = {
    "v4_exact_18": "0196ccca546fb881aae8194e996ee902ac2e2c39c62721f66579a59f1eeb17b8",
    "original_8": "9cb87b853eb43636d6baf09c281725eb98255b6cf73439025b47d627e84da5a8",
    "expansion_10": "028efa424ed4dc2fe511a0b723b38f4feccba7f4de7db055934218dff1fe705d",
    "expansion_10_plan": "796eccc748cd61590bd0c9d4499e92e81277327f11e6ad4a19ce106afa4b4cb6",
    "v1_mixed_split": "926391535972830289315b95e6b4e4893a3e82aeac03b371ab3212ca324b906a",
    "canonical_release": "a603239d2c96946662390c4dc45c543f1652055e6edd21e156a6f351f78db22a",
}
FROZEN_CONFIG_SHA256 = (
    "b048ca4f4950eaf37d8e751a88ee358d5eabeb83b5187302502d9e08d62b130d"
)
INCIDENT_SHA256 = (
    "10d46eda4dc08e5cb98b0e9752fe9e4d94cac431549c6eb2bf6505b1a3cbc71d"
)
CLEAN_RUN_ID = "childlens-v5-clean-20260723-a"


class AdministrationError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise AdministrationError(code)


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
        _fail("E_CANONICAL")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(block)
    except OSError:
        _fail("E_FILE")
    return value.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        if not path.is_file() or path.is_symlink():
            _fail("E_FILE")
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("E_FILE")


def _private_directory(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) & 0o077 == 0
    )


def _private_file(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) & 0o077 == 0
    )


def _discover_runtime() -> Path:
    candidates: list[Path] = []
    for hidden in ROOT.parent.iterdir():
        if (
            hidden.name.startswith(".")
            and "childlens" in hidden.name.casefold()
            and _private_directory(hidden)
        ):
            candidates.extend(
                path.parent
                for path in hidden.rglob("restricted_manifest/preselection_manifest.json")
                if _private_file(path)
            )
    unique = sorted({path.resolve() for path in candidates})
    if len(unique) != 1:
        _fail("E_RUNTIME_DISCOVERY")
    runtime = unique[0]
    if not _private_file(runtime / ".metadata_never_index"):
        _fail("E_RUNTIME_CONTROL")
    return runtime


def _write_atomic(path: Path, value: Any, *, private: bool) -> None:
    payload = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    if private:
        os.chmod(path.parent, 0o700)
    pending = path.parent / f".pending-{secrets.token_hex(12)}"
    try:
        descriptor = os.open(
            pending,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600 if private else 0o644,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        os.chmod(path, 0o600 if private else 0o644)
    except OSError:
        _fail("E_WRITE")
    finally:
        with contextlib.suppress(FileNotFoundError):
            pending.unlink()


def _validated_sources(runtime: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, relative in SOURCE_RELATIVE.items():
        path = runtime / relative
        if not _private_file(path) or _sha256_file(path) != EXPECTED_SOURCE_HASHES[name]:
            _fail("E_SOURCE_BINDING")
        result[name] = _read_json(path)
    if (
        _sha256_file(CONFIG) != FROZEN_CONFIG_SHA256
        or _sha256_file(INCIDENT) != INCIDENT_SHA256
    ):
        _fail("E_HISTORY_BINDING")
    return result


def create() -> Mapping[str, Any]:
    runtime = _discover_runtime()
    sources = _validated_sources(runtime)
    v4 = sources["v4_exact_18"]
    original = sources["original_8"]
    expansion = sources["expansion_10"]
    expansion_plan = sources["expansion_10_plan"]
    mixed_split = sources["v1_mixed_split"]
    release = sources["canonical_release"]
    if (
        v4.get("development_participants") != 18
        or v4.get("locked_rows_loaded_scored_summarized_or_inspected") != 0
        or not isinstance(v4.get("items"), list)
        or len(v4["items"]) != 18
        or original.get("development_count") != 8
        or expansion.get("development_count") != 10
        or original.get("locked_rows_copied") != 0
        or expansion.get("locked_rows_copied_or_evaluated") != 0
    ):
        _fail("E_DEVELOPMENT_BASE")
    plan_by_participant = {
        row.get("participant_key"): row
        for row in expansion_plan.get("items", [])
        if isinstance(row, Mapping)
    }
    if len(plan_by_participant) != 10:
        _fail("E_DEVELOPMENT_BASE")
    expansion_rows: list[dict[str, Any]] = []
    for row in expansion["items"]:
        plan = plan_by_participant.get(row.get("participant_key"))
        if not isinstance(plan, Mapping):
            _fail("E_DEVELOPMENT_BASE")
        expansion_rows.append({**dict(row), **dict(plan)})
    source_rows = [
        *((row, "original_eight") for row in original["items"]),
        *((row, "v3_expansion_ten") for row in expansion_rows),
    ]
    if any(not isinstance(row, Mapping) for row, _ in source_rows):
        _fail("E_DEVELOPMENT_ROWS")
    participants = [str(row.get("participant_key", "")) for row, _ in source_rows]
    anchor_media_keys = [str(row.get("media_key", "")) for row, _ in source_rows]
    v4_participants = {
        str(row.get("participant_key", ""))
        for row in v4["items"]
        if isinstance(row, Mapping)
    }
    if (
        len(participants) != 18
        or len(set(participants)) != 18
        or not all(participants)
        or len(set(anchor_media_keys)) != 18
        or not all(anchor_media_keys)
        or set(participants) != v4_participants
    ):
        _fail("E_ALLOWLIST")
    locked = {
        str(row.get("participant_key", ""))
        for row in mixed_split.get("items", [])
        if isinstance(row, Mapping) and row.get("split") == "locked"
    }
    if len(locked) != 22 or "" in locked or set(participants) & locked:
        _fail("E_LOCKED_OVERLAP")

    records = release.get("records")
    if not isinstance(records, Mapping):
        _fail("E_RELEASE")
    for name in ("objects", "groupings", "linkages", "selection_metadata"):
        if not isinstance(records.get(name), list):
            _fail("E_RELEASE")
    cohort_by_participant = {
        str(row["participant_key"]): cohort for row, cohort in source_rows
    }
    groupings = {
        str(row.get("media_key")): row
        for row in records["groupings"]
        if isinstance(row, Mapping)
        and row.get("participant_key") in cohort_by_participant
    }
    allowed_media = set(groupings)
    metadata = {
        str(row.get("media_key")): row
        for row in records["selection_metadata"]
        if isinstance(row, Mapping) and row.get("media_key") in allowed_media
    }
    linkages: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records["linkages"]:
        if isinstance(row, Mapping) and row.get("linked_media_key") in allowed_media:
            linkages[str(row["linked_media_key"])].append(row)
    selected_object_keys = {
        str(row.get("object_key"))
        for row in metadata.values()
    } | {
        str(row.get("object_key"))
        for rows in linkages.values()
        for row in rows
    }
    objects = {
        str(row.get("object_key")): row
        for row in records["objects"]
        if isinstance(row, Mapping) and row.get("object_key") in selected_object_keys
    }
    if (
        len(groupings) < 18
        or set(metadata) != allowed_media
        or any(not linkages[key] for key in allowed_media)
        or set(selected_object_keys) != set(objects)
    ):
        _fail("E_FILTER_SUPPORT")

    anchor_by_media = {str(row["media_key"]): row for row, _ in source_rows}
    items: list[dict[str, Any]] = []
    for media_key in sorted(allowed_media):
        grouping = groupings[media_key]
        selection = metadata[media_key]
        participant = str(grouping.get("participant_key", ""))
        cohort = cohort_by_participant.get(participant)
        raw = anchor_by_media.get(media_key, {})
        media_object = objects.get(str(selection.get("object_key")))
        if (
            cohort is None
            or not isinstance(media_object, Mapping)
            or media_object.get("top_level_class") != "VIDEO"
            or media_object.get("available_for_selective_copy") is not True
            or not isinstance(media_object.get("source_locator"), str)
            or type(media_object.get("size_bytes")) is not int
            or type(selection.get("duration_milliseconds")) is not int
        ):
            _fail("E_SOURCE_ROW")
        annotations: list[dict[str, Any]] = []
        for linkage in sorted(
            linkages[media_key], key=lambda row: str(row.get("object_key"))
        ):
            obj = objects.get(str(linkage.get("object_key")))
            if (
                not isinstance(obj, Mapping)
                or obj.get("top_level_class") != "ANNOTATION"
                or not isinstance(obj.get("source_locator"), str)
                or not isinstance(obj.get("local_sha256"), str)
            ):
                _fail("E_ANNOTATION_ROW")
            annotations.append(
                {
                    "object_key": obj["object_key"],
                    "source_locator": obj["source_locator"],
                    "local_sha256": obj["local_sha256"],
                    "representation_kind": linkage.get("representation_kind"),
                }
            )
        items.append(
            {
                "participant_key": participant,
                "cohort": cohort,
                "media_key": media_key,
                "source_object_key": media_object["object_key"],
                "source_locator": media_object["source_locator"],
                "source_size_bytes": media_object["size_bytes"],
                "source_duration_milliseconds": selection["duration_milliseconds"],
                "expected_media_sha256": raw.get("expected_media_sha256"),
                "released_activity_label": (
                    raw.get("activity_label")
                    if raw.get("activity_label") is not None
                    else selection.get("coarse_activity_label")
                ),
                "released_location_label": (
                    raw.get("location_label")
                    if raw.get("location_label") is not None
                    else grouping.get("location_label")
                ),
                "released_speech_support_bin": selection.get("speech_presence_bin"),
                "annotation_bindings": annotations,
            }
        )
    items.sort(key=lambda row: str(row["participant_key"]))
    allowlist_sha256 = _digest(sorted(participants))
    locked_set_sha256 = _digest(sorted(locked))
    zero_overlap_sha256 = _digest(
        {
            "development_allowlist_sha256": allowlist_sha256,
            "locked_set_sha256": locked_set_sha256,
            "intersection": [],
        }
    )
    private_input = {
        "schema_version": "childlens-v5-development-only-scientific-input-v1.0.0",
        "clean_run_id": CLEAN_RUN_ID,
        "frozen_v5_config_sha256": FROZEN_CONFIG_SHA256,
        "incident_receipt_sha256": INCIDENT_SHA256,
        "administrative_only": True,
        "development_participant_count": 18,
        "locked_participant_count": 0,
        "source_recording_count": len(items),
        "cohort_counts": {
            "original_eight": 8,
            "v3_expansion_ten": 10,
        },
        "development_allowlist_sha256": allowlist_sha256,
        "locked_set_sha256": locked_set_sha256,
        "zero_overlap_sha256": zero_overlap_sha256,
        "items": items,
    }
    private_path = runtime / PRIVATE_INPUT
    _write_atomic(private_path, private_input, private=True)
    input_sha256 = _sha256_file(private_path)
    code_sha256 = _sha256_file(Path(__file__))
    attestation = {
        "schema_version": "childlens-v5-development-only-attestation-v1.0.0",
        "status": "PASS",
        "clean_run_id": CLEAN_RUN_ID,
        "frozen_v5_config_sha256": FROZEN_CONFIG_SHA256,
        "incident_receipt_sha256": INCIDENT_SHA256,
        "filter_code_sha256": code_sha256,
        "source_hashes": EXPECTED_SOURCE_HASHES,
        "development_allowlist_sha256": allowlist_sha256,
        "locked_set_sha256": locked_set_sha256,
        "zero_overlap_sha256": zero_overlap_sha256,
        "development_participant_count": 18,
        "locked_participant_count": 0,
        "source_recording_count": len(items),
        "cohort_counts": {
            "original_eight": 8,
            "v3_expansion_ten": 10,
        },
        "scientific_input_sha256": input_sha256,
        "private_input_mode": "0600",
        "locked_media_annotations_outcomes_or_model_results_opened": False,
        "legacy_mixed_scope_inputs_available_to_scientific_process": False,
    }
    attestation_path = runtime / PRIVATE_ATTESTATION
    _write_atomic(attestation_path, attestation, private=True)
    if not _private_file(private_path) or not _private_file(attestation_path):
        _fail("E_PRIVATE_MODE")
    public = {
        "schema_version": "childlens-v5-administrative-correction-v1.0.0",
        "status": "PASS",
        "clean_run_id": CLEAN_RUN_ID,
        "original_incident_commit": "020fd29f5919b5fa49d5d47787ba0ade41a01dc5",
        "incident_receipt_sha256": INCIDENT_SHA256,
        "frozen_v5_config_sha256": FROZEN_CONFIG_SHA256,
        "scientific_method_changed": False,
        "filter_code_sha256": code_sha256,
        "source_hashes": EXPECTED_SOURCE_HASHES,
        "development_allowlist_sha256": allowlist_sha256,
        "locked_set_sha256": locked_set_sha256,
        "zero_overlap_sha256": zero_overlap_sha256,
        "scientific_input_sha256": input_sha256,
        "attestation_sha256": _sha256_file(attestation_path),
        "development_participant_count": 18,
        "locked_participant_count": 0,
        "source_recording_count": len(items),
        "cohort_counts": {
            "original_eight": 8,
            "v3_expansion_ten": 10,
        },
        "private_files_owner_only_mode_0600": True,
        "administrative_membership_and_source_metadata_only": True,
        "locked_media_annotations_outcomes_or_model_results_opened": False,
        "legacy_mixed_scope_inputs_available_to_scientific_process": False,
        "restricted_values_exported": False,
    }
    _write_atomic(PUBLIC_RECEIPT, public, private=False)
    return public


def validate() -> Mapping[str, Any]:
    runtime = _discover_runtime()
    sources = _validated_sources(runtime)
    del sources
    private_path = runtime / PRIVATE_INPUT
    attestation_path = runtime / PRIVATE_ATTESTATION
    public = _read_json(PUBLIC_RECEIPT)
    attestation = _read_json(attestation_path)
    scientific_input = _read_json(private_path)
    if (
        not _private_file(private_path)
        or not _private_file(attestation_path)
        or public.get("status") != "PASS"
        or public.get("filter_code_sha256") != _sha256_file(Path(__file__))
        or public.get("scientific_input_sha256") != _sha256_file(private_path)
        or public.get("attestation_sha256") != _sha256_file(attestation_path)
        or attestation.get("scientific_input_sha256") != _sha256_file(private_path)
        or scientific_input.get("development_participant_count") != 18
        or scientific_input.get("locked_participant_count") != 0
        or len(scientific_input.get("items", [])) < 18
        or len(
            {
                row.get("participant_key")
                for row in scientific_input.get("items", [])
                if isinstance(row, Mapping)
            }
        )
        != 18
    ):
        _fail("E_VALIDATION")
    return {
        "status": "PASS",
        "clean_run_id": CLEAN_RUN_ID,
        "development_participant_count": 18,
        "locked_participant_count": 0,
        "source_recording_count": len(scientific_input["items"]),
        "scientific_input_sha256": _sha256_file(private_path),
        "attestation_sha256": _sha256_file(attestation_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("create", "validate"))
    args = parser.parse_args()
    try:
        result = create() if args.command == "create" else validate()
    except AdministrationError as exc:
        print(json.dumps({"status": "ERROR", "error_code": exc.code}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
