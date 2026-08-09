from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_childlens_post_acquisition_v1_2.py"
SPEC = importlib.util.spec_from_file_location("run_childlens_post_acquisition_v1_2", SCRIPT)
assert SPEC and SPEC.loader
post = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = post
SPEC.loader.exec_module(post)

VALIDATOR_SCRIPT = ROOT / "scripts" / "validate_childlens_feasibility_v1_2.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_childlens_feasibility_v1_2_post_contract", VALIDATOR_SCRIPT
)
assert VALIDATOR_SPEC and VALIDATOR_SPEC.loader
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
sys.modules[VALIDATOR_SPEC.name] = validator
VALIDATOR_SPEC.loader.exec_module(validator)


def _private_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    path.write_bytes(payload)
    os.chmod(path, 0o600)


def _config() -> dict:
    bounds = []
    base = post.FROZEN_CONSERVATIVE_ADMISSION_BYTES // 15
    remainder = post.FROZEN_CONSERVATIVE_ADMISSION_BYTES - base * 15
    for index in range(15):
        bounds.append(
            {
                "object_key": hashlib.sha256(f"object-{index}".encode()).hexdigest(),
                "rounded_display_size_bytes": 1,
                "rounding_quantum_bytes": 1,
                "transfer_overhead_bytes": 0,
                "upper_bound_bytes": base + (1 if index < remainder else 0),
            }
        )
    return {
        "immutable_commit_id": None,
        "quarantine_attestations": {
            "owner_only_access_verified": True,
            "outside_git_repository_verified": True,
            "spotlight_excluded_verified": True,
            "backup_excluded_or_encrypted_local_only_verified": True,
            "signed_agreement_controls_verified": True,
            "retention_deadline": post.RETENTION_DEADLINE,
        },
        "admission": {
            "predeclared_nonraw_reserve_bytes": post.NONRAW_RESERVE_BYTES,
            "conservative_bounds": bounds,
        },
    }


def _all_pass_audit(selection: str = "a" * 64) -> dict:
    metric = {
        "status": "ALL_PASS",
        "pass_count": 15,
        "fail_count": 0,
        "cell_suppressed": False,
    }
    return {
        "schema_version": "childlens-local-media-audit-v1.2.1",
        "status": "ALL_STRUCTURAL_CHECKS_PASS",
        "pilot_selection_sha256": selection,
        "container_audio_decode_metrics": {
            name: dict(metric)
            for name in (
                "restricted_manifest_file_integrity_match",
                "probe_success",
                "video_stream_present",
                "audio_stream_present",
                "duration_consistency_evidenced",
                "full_decode_success",
                "corruption_absence_evidenced",
                "speech_presence_window_expectation_satisfied",
            )
        },
    }


def test_sealed_bundle_does_not_revalidate_controller_normalized_config(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "sealed-receipt.json"
    _private_file(receipt_path, b"{}\n")
    normalized_plan = {"selected": [object()] * 15}
    normalized_config = {
        "admission": {
            "conservative_bounds": [
                {"object_key": "a" * 64, "upper_bound_bytes": 1}
            ]
        }
    }
    sealed = {"status": "COMPLETE", "items": [{} for _ in range(15)]}
    bundle = SimpleNamespace(
        plan=normalized_plan,
        config=normalized_config,
        restricted_receipt_path=receipt_path,
    )

    class Launcher:
        @staticmethod
        def discover_bundle(**_kwargs):
            return bundle

    class Transfer:
        @staticmethod
        def _read_json(path, code):
            assert path == receipt_path
            assert code == "E_RECEIPT_READ"
            return sealed

        @staticmethod
        def _validate_receipt(value, plan, config):
            assert value is sealed
            assert plan is normalized_plan
            assert config is normalized_config
            return sealed

        @staticmethod
        def validate_config(*_args, **_kwargs):
            raise AssertionError("normalized config must not be validated twice")

    loaded_bundle, plan, config, receipt = post._load_sealed_bundle(
        search_root=tmp_path,
        repository_root=tmp_path / "repo",
        transfer=Transfer,
        launcher=Launcher,
    )
    assert loaded_bundle is bundle
    assert plan is normalized_plan
    assert receipt is sealed
    assert config["admission"] is normalized_config["admission"]
    assert config["_native_receipt_path"] == str(receipt_path)


def test_extracts_only_explicit_speech_like_timed_bouts_and_coalesces() -> None:
    document = {
        "annotations": [
            {"eventId": "other person talking", "startTime": 1.0, "endTime": 2.0},
            {"eventId": "overheard speech", "start": 1.5, "duration": 2.0},
            {"eventId": "playing", "startTime": 4.0, "endTime": 7.0},
            {"eventId": "child vocalizing", "time": 8.0, "duration": 1.0},
        ]
    }
    windows, invalid = post.extract_official_speech_windows(document)
    assert windows == [(1.0, 3.5), (8.0, 9.0)]
    assert invalid == 0


def test_no_speech_rows_is_an_explicit_empty_window_set() -> None:
    windows, invalid = post.extract_official_speech_windows(
        {"annotations": [{"eventId": "playing", "startTime": 0, "endTime": 4}]}
    )
    assert windows == []
    assert invalid == 0


def test_malformed_speech_timing_is_counted_not_promoted_to_an_utterance() -> None:
    windows, invalid = post.extract_official_speech_windows(
        {"annotations": [{"eventId": "speech", "startTime": 2.0}]}
    )
    assert windows == []
    assert invalid == 1


def test_canonical_document_discovery_is_unique_and_digest_bound(tmp_path: Path) -> None:
    root = tmp_path / "restricted"
    root.mkdir(mode=0o700)
    value = {"schema_version": "synthetic", "items": [1, 2, 3]}
    path = root / "opaque.json"
    _private_file(path, post._canonical(value) + b"\n")
    found, parsed = post._find_canonical_document(root, post._digest(value))
    assert found == path
    assert parsed == value
    duplicate = root / "second.json"
    _private_file(duplicate, post._canonical(value) + b"\n")
    with pytest.raises(post.OrchestratorError, match="E_FROZEN_DOCUMENT_AMBIGUOUS"):
        post._find_canonical_document(root, post._digest(value))


class _FakeWorkflow:
    VERSION = "childlens-human-validation-workflow-v1.2.0"

    def __init__(self, keys: set[str], frozen: dict, native: dict, selection: str):
        self.keys = keys
        self.frozen = frozen
        self.native = native
        self.selection = selection

    def _validate_frozen_skeleton(self, _path: Path, _root: Path):
        return self.keys, self.selection

    def _validate_restricted_input(self, _path: Path, _root: Path, _keys: set[str]):
        return self.frozen

    def _validate_native_receipt(self, _path: Path, _root: Path, _selection: str):
        return self.native


def _restricted_fixture(tmp_path: Path):
    root = tmp_path / "restricted"
    root.mkdir(mode=0o700)
    os.chmod(root, 0o700)
    selection = "a" * 64
    keys: set[str] = set()
    frozen: dict[str, dict] = {}
    native: dict[str, dict] = {}
    media_rows = []
    objects = []
    annotations = []
    for index in range(15):
        media_key = f"opaqueitemkey{index:04d}"
        object_key = hashlib.sha256(f"source-object-{index}".encode()).hexdigest()
        annotation_object = hashlib.sha256(f"annotation-object-{index}".encode()).hexdigest()
        media_payload = f"synthetic-media-{index}".encode()
        media_sha = hashlib.sha256(media_payload).hexdigest()
        media_relpath = f"raw_v1_2/{media_sha}.bin"
        _private_file(root / media_relpath, media_payload)
        annotation_document = {
            "annotations": [
                {
                    "eventId": "other person talking",
                    "startTime": float(index),
                    "endTime": float(index + 180),
                }
            ]
        }
        annotation_path = root / "annotations" / f"opaque-{index}.json"
        annotation_bytes = post._canonical(annotation_document) + b"\n"
        _private_file(annotation_path, annotation_bytes)
        annotation_sha = hashlib.sha256(annotation_bytes).hexdigest()
        linkage = hashlib.sha256(
            media_key.encode() + b"\0" + annotation_sha.encode()
        ).hexdigest()
        keys.add(media_key)
        frozen[media_key] = {
            "source_object_key": object_key,
            "duration_ms": 300_000,
            "stratum_key": hashlib.sha256(f"stratum-{index}".encode()).hexdigest(),
            "annotation_linkage_sha256": linkage,
        }
        native[object_key] = {
            "media_sha256": media_sha,
            "media_relpath": media_relpath,
            "transferred_bytes": len(media_payload),
        }
        media_rows.append(
            {
                "media_key": media_key,
                "object_key": object_key,
                "speech_presence_bin": "PRESENT",
            }
        )
        objects.append(
            {
                "object_key": annotation_object,
                "source_locator": str(annotation_path),
                "top_level_class": "ANNOTATION",
                "local_sha256": annotation_sha,
            }
        )
        annotations.append({"linked_media_key": media_key, "object_key": annotation_object})
    skeleton_path = root / "skeleton.json"
    input_path = root / "input.json"
    receipt_path = root / "receipt.json"
    for path in (skeleton_path, input_path, receipt_path):
        _private_file(path, b"{}\n")
    restricted = {"media": media_rows, "objects": objects, "annotations": annotations}
    workflow = _FakeWorkflow(keys, frozen, native, selection)
    config = _config()
    config["_native_receipt_path"] = str(receipt_path)
    receipt = {
        "status": "COMPLETE",
        "canonical_restricted_manifest_sha256": "b" * 64,
        "restricted_receipt_sha256": "c" * 64,
        "download_plan_sha256": "d" * 64,
        "immutable_commit_id": None,
    }
    prepared = post.prepare_restricted_artifacts(
        root=root,
        plan={},
        config=config,
        receipt=receipt,
        skeleton_path=skeleton_path,
        skeleton={},
        restricted_input_path=input_path,
        restricted_input=restricted,
        workflow=workflow,
    )
    return root, config, prepared


def test_restricted_manifests_bind_all_15_rows_without_utterance_inference(tmp_path: Path) -> None:
    _root, _config_value, prepared = _restricted_fixture(tmp_path)
    measurement = json.loads(prepared.measurement_path.read_text())
    packet = json.loads(prepared.packet_path.read_text())
    assert len(measurement["items"]) == len(packet["items"]) == 15
    assert all(row["speech_presence_expectation"] == "PRESENT" for row in measurement["items"])
    assert all(len(row["speech_windows"]) == 1 for row in measurement["items"])
    assert all(set(row) == {
        "internal_key", "display_key", "media_relpath", "stratum_key", "duration_ms",
        "batch_number", "source_object_key", "media_sha256", "annotation_linkage_sha256"
    } for row in packet["items"])
    assert stat.S_IMODE(prepared.measurement_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(prepared.packet_path.stat().st_mode) == 0o600
    assert prepared.media_verified_count == 15
    assert prepared.annotation_verified_count == 15
    assert prepared.speech_window_count == 15


def test_annotation_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    root, config, prepared = _restricted_fixture(tmp_path)
    # The successful preparation proves the baseline; direct verification must
    # reject a changed digest without exposing the locator.
    annotation = next((root / "annotations").glob("*.json"))
    with pytest.raises(post.OrchestratorError, match="E_ANNOTATION_INTEGRITY"):
        post._verify_file_inside(root, str(annotation), "f" * 64)
    assert config and prepared


def test_annotation_parent_symlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "restricted"
    root.mkdir(mode=0o700)
    actual = root / "actual"
    actual.mkdir(mode=0o700)
    payload = post._canonical({"annotations": []}) + b"\n"
    target = actual / "opaque.json"
    _private_file(target, payload)
    (root / "alias").symlink_to(actual, target_is_directory=True)
    linked = root / "alias" / "opaque.json"
    with pytest.raises(post.OrchestratorError, match="E_ANNOTATION_PATH_BOUNDARY"):
        post._verify_file_inside(root, str(linked), hashlib.sha256(payload).hexdigest())


def test_macos_var_alias_does_not_create_false_quarantine_escape(tmp_path: Path) -> None:
    root = tmp_path / "restricted"
    root.mkdir(mode=0o700)
    payload = post._canonical({"annotations": []}) + b"\n"
    target = root / "opaque.json"
    _private_file(target, payload)
    resolved = str(target.resolve())
    if not resolved.startswith("/private/var/"):
        pytest.skip("macOS /var alias not present")
    aliased = Path(resolved.replace("/private/var/", "/var/", 1))
    _path, document = post._verify_file_inside(
        root.resolve(), str(aliased), hashlib.sha256(payload).hexdigest()
    )
    assert document == {"annotations": []}


def test_public_receipts_match_hardened_runtime_contract(tmp_path: Path) -> None:
    _root, config, prepared = _restricted_fixture(tmp_path)
    acquisition, diagnostics, workflow = post.build_public_receipts(
        prepared=prepared,
        config=config,
        audit_receipt=_all_pass_audit(prepared.pilot_selection_sha256),
        workflow_readiness={"runtime_handoff_ready": True},
        namespace_bytes=9 * 1024**3,
        free_bytes=110 * 1024**3,
    )
    assert acquisition["acquisition_complete"] is True
    assert acquisition["pre_transfer_upper_bound_bytes"] == post.FROZEN_CONSERVATIVE_ADMISSION_BYTES
    assert acquisition["live_to_public_snapshot_byte_equivalence"] == "NOT_PROVEN"
    assert diagnostics["structural_metrics"]["annotation_linkage_verified"]["status"] == "ALL_PASS"
    assert diagnostics["official_speech_window_union_minutes"] == 45.0
    assert workflow["candidate_unit_type"] == "OFFICIAL_SPEECH_WINDOW_NOT_UTTERANCE"
    assert workflow["populated_candidate_utterance_count"] == 0
    assert workflow["referential_assignment_eligible_count"] == 0
    assert workflow["ready_for_authorized_human"] is True
    assert workflow["estimated_total_human_minutes"] == sum(
        workflow[key]
        for key in (
            "estimated_first_authorized_human_minutes",
            "estimated_second_independent_human_minutes",
            "estimated_adjudication_minutes",
        )
    )


def test_no_window_packet_is_honestly_not_human_ready(tmp_path: Path) -> None:
    _root, config, prepared = _restricted_fixture(tmp_path)
    prepared = post.RestrictedPreparation(
        **{
            **prepared.__dict__,
            "speech_window_source_count": 0,
            "no_speech_window_source_count": 15,
            "speech_window_count": 0,
            "unioned_speech_seconds": 0.0,
        }
    )
    _acquisition, diagnostics, workflow = post.build_public_receipts(
        prepared=prepared,
        config=config,
        audit_receipt=_all_pass_audit(prepared.pilot_selection_sha256),
        workflow_readiness={"runtime_handoff_ready": True},
        namespace_bytes=9 * 1024**3,
        free_bytes=110 * 1024**3,
    )
    assert diagnostics["official_speech_window_union_minutes"] == 0
    assert workflow["populated_candidate_speech_window_count"] == 0
    assert workflow["ready_for_authorized_human"] is False


def test_emitted_receipts_pass_hardened_runtime_contract(tmp_path: Path) -> None:
    _root, config, prepared = _restricted_fixture(tmp_path)
    prepared = post.RestrictedPreparation(
        **{
            **prepared.__dict__,
            "pilot_selection_sha256": validator.FROZEN_SELECTION_DIGEST,
        }
    )
    acquisition, diagnostics, workflow = post.build_public_receipts(
        prepared=prepared,
        config=config,
        audit_receipt=_all_pass_audit(prepared.pilot_selection_sha256),
        workflow_readiness={"runtime_handoff_ready": True},
        namespace_bytes=9 * 1024**3,
        free_bytes=110 * 1024**3,
    )
    issues = []
    dummy = tmp_path / "aggregate.json"
    validator._check_acquisition(acquisition, dummy, tmp_path, issues)
    validator._check_diagnostics(diagnostics, acquisition, dummy, tmp_path, issues)
    validator._check_workflow(workflow, acquisition, diagnostics, dummy, tmp_path, issues)
    assert issues == []


def test_small_window_count_is_suppressed_in_public_workflow(tmp_path: Path) -> None:
    _root, config, prepared = _restricted_fixture(tmp_path)
    prepared = post.RestrictedPreparation(
        **{
            **prepared.__dict__,
            "speech_window_source_count": 3,
            "no_speech_window_source_count": 12,
            "speech_window_count": 3,
        }
    )
    _acquisition, _diagnostics, workflow = post.build_public_receipts(
        prepared=prepared,
        config=config,
        audit_receipt=_all_pass_audit(prepared.pilot_selection_sha256),
        workflow_readiness={"runtime_handoff_ready": True},
        namespace_bytes=9 * 1024**3,
        free_bytes=110 * 1024**3,
    )
    assert workflow["populated_candidate_speech_window_count"] is None
    assert workflow["populated_candidate_speech_window_count_suppressed"] is True


def test_public_guard_rejects_paths_email_and_media_names() -> None:
    for value in (
        {"note": "/Users/example/restricted"},
        {"note": "person@example.test"},
        {"note": "source-video.mp4"},
    ):
        with pytest.raises(post.OrchestratorError, match="E_PUBLIC_PRIVACY"):
            post._public_guard(value)


def test_atomic_public_batch_writes_only_named_aggregate_documents(tmp_path: Path) -> None:
    output = tmp_path / "output"
    post._atomic_public_batch(
        output,
        {
            "acquisition_receipt.json": {"status": "ok"},
            "automated_diagnostics_receipt.json": {"status": "ok"},
            "human_validation_workflow_receipt.json": {"status": "ok"},
        },
    )
    assert sorted(path.name for path in output.iterdir()) == [
        "acquisition_receipt.json",
        "automated_diagnostics_receipt.json",
        "human_validation_workflow_receipt.json",
    ]


def test_main_rejects_arguments_without_running(monkeypatch, capsys) -> None:
    monkeypatch.setattr(post, "execute", lambda: pytest.fail("execute called"))
    assert post.main(["unexpected"]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "error_code": "E_ARGUMENTS",
        "status": "error",
    }


def test_main_success_stdout_is_fixed(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        post,
        "execute",
        lambda: {"status": "ok", "state": "POST_ACQUISITION_RECEIPTS_READY"},
    )
    assert post.main([]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "state": "POST_ACQUISITION_RECEIPTS_READY",
        "status": "ok",
    }


def test_quarantine_control_sentinel_may_live_on_private_ancestor(
    tmp_path: Path,
) -> None:
    control_root = tmp_path / ".childlens-control"
    data_root = control_root / "bundle"
    data_root.mkdir(mode=0o700, parents=True)
    os.chmod(control_root, 0o700)
    os.chmod(data_root, 0o700)
    _private_file(control_root / ".metadata_never_index", b"")
    _private_file(data_root / "opaque.bin", b"synthetic")
    namespace, free = post._recheck_quarantine(data_root, _config())
    assert namespace >= len(b"synthetic")
    assert free >= post.FREE_SPACE_FLOOR_BYTES


def test_synthetic_end_to_end_execute_initializes_workflow_and_emits_only_aggregates(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / ".childlens-synthetic" / "bundle"
    root.mkdir(mode=0o700, parents=True)
    os.chmod(root.parent, 0o700)
    os.chmod(root, 0o700)
    _private_file(root.parent / ".metadata_never_index", b"")
    selection = "a" * 64
    skeleton = {
        "schema_version": "childlens-human-validation-packet-v1.1.0",
        "packet_status": "SKELETON_AWAITING_MEDIA_AND_HUMANS",
        "pilot_selection_sha256": selection,
        "selected_count": 15,
        "human_judgment_required": True,
        "codex_may_act_as_human": False,
        "timing_and_speaker_items_double_coded": True,
        "referential_items_minimum_double_code_fraction": 0.2,
        "learner_training_permitted": False,
        "items": [],
    }
    media_rows = []
    object_rows = []
    annotation_rows = []
    native_items = []
    for index in range(15):
        media_key = f"opaqueitemkey{index:04d}"
        object_key = hashlib.sha256(f"e2e-source-{index}".encode()).hexdigest()
        annotation_key = hashlib.sha256(f"e2e-annotation-{index}".encode()).hexdigest()
        skeleton["items"].append(
            {
                "blinded_item_key": media_key,
                "acquisition_status": "PENDING_EXACT_REMOTE_BYTES",
                "language_judgment": None,
                "audio_integrity": None,
                "utterance_timing_text_role_review": None,
                "referential_status_review": None,
                "adjudication_status": "NOT_STARTED",
            }
        )
        media_payload = f"synthetic-e2e-media-{index}".encode()
        media_sha = hashlib.sha256(media_payload).hexdigest()
        media_relpath = f"raw_v1_2/{media_sha}.bin"
        _private_file(root / media_relpath, media_payload)
        annotation_document = {
            "annotations": [
                {
                    "eventId": "other person talking",
                    "startTime": 0.0,
                    "endTime": 180.0,
                }
            ]
        }
        annotation_path = root / "annotations" / f"annotation-{index}.json"
        annotation_payload = post._canonical(annotation_document) + b"\n"
        _private_file(annotation_path, annotation_payload)
        annotation_sha = hashlib.sha256(annotation_payload).hexdigest()
        object_rows.extend(
            [
                {"object_key": object_key, "local_sha256": None},
                {
                    "object_key": annotation_key,
                    "source_locator": str(annotation_path),
                    "top_level_class": "ANNOTATION",
                    "local_sha256": annotation_sha,
                },
            ]
        )
        annotation_rows.append({"linked_media_key": media_key, "object_key": annotation_key})
        media_rows.append(
            {
                "media_key": media_key,
                "object_key": object_key,
                "duration_milliseconds": 300_000,
                "coarse_activity_label": f"ACTIVITY_{index % 3}",
                "speech_presence_bin": "PRESENT",
                "location_label": f"LOCATION_{index % 2}",
            }
        )
        native_items.append(
            {
                "selection_rank": index + 1,
                "object_key": object_key,
                "status": "COMPLETE",
                "local_sha256": media_sha,
                "stored_relative_path": media_relpath,
                "transferred_bytes": len(media_payload),
            }
        )
    restricted_input = {
        "schema_version": "synthetic",
        "objects": object_rows,
        "annotations": annotation_rows,
        "media": media_rows,
    }
    skeleton_path = root.parent / "frozen-skeleton.json"
    input_path = root.parent / "frozen-input.json"
    _private_file(skeleton_path, post._canonical(skeleton) + b"\n")
    _private_file(input_path, post._canonical(restricted_input) + b"\n")
    native_receipt = {
        "schema_version": "childlens-native-transfer-restricted-receipt-v1.2.0",
        "status": "COMPLETE",
        "pilot_selection_sha256": selection,
        "canonical_restricted_manifest_sha256": "b" * 64,
        "download_plan_sha256": "d" * 64,
        "immutable_commit_id": None,
        "items": native_items,
        "restricted_receipt_sha256": None,
    }
    native_receipt["restricted_receipt_sha256"] = hashlib.sha256(
        post._canonical(native_receipt)
    ).hexdigest()
    native_receipt_path = root / "native-transfer-receipt.json"
    _private_file(native_receipt_path, post._canonical(native_receipt) + b"\n")
    config = _config()
    config["_native_receipt_path"] = str(native_receipt_path)
    bundle = SimpleNamespace(quarantine_root=root)

    def loader(**_kwargs):
        return bundle, {"pilot_selection_sha256": selection}, config, native_receipt

    _transfer, _launcher, auditor, workflow = post._modules()
    monkeypatch.setattr(
        workflow,
        "FROZEN_V1_1_HUMAN_PACKET_SHA256",
        post._digest(skeleton),
    )
    monkeypatch.setattr(
        workflow,
        "FROZEN_V1_1_RESTRICTED_INPUT_SHA256",
        post._digest(restricted_input),
    )
    monkeypatch.setattr(
        auditor,
        "audit",
        lambda *_args, **_kwargs: _all_pass_audit(selection),
    )
    fake_tool = tmp_path / "fake-local-tool"
    _private_file(fake_tool, b"synthetic")
    os.chmod(fake_tool, 0o700)
    output = tmp_path / "repo-output"
    result = post.execute(
        search_root=tmp_path,
        repository_root=tmp_path / "repo",
        output_root=output,
        frozen_skeleton_sha256=post._digest(skeleton),
        frozen_restricted_input_sha256=post._digest(restricted_input),
        bundle_loader=loader,
        ffprobe=fake_tool,
        ffmpeg=fake_tool,
    )
    assert result["state"] == "POST_ACQUISITION_RECEIPTS_READY"
    assert sorted(path.name for path in output.iterdir()) == [
        "acquisition_receipt.json",
        "automated_diagnostics_receipt.json",
        "human_validation_workflow_receipt.json",
    ]
    workflow_receipt = json.loads((output / "human_validation_workflow_receipt.json").read_text())
    assert workflow_receipt["ready_for_authorized_human"] is True
    assert workflow_receipt["populated_candidate_utterance_count"] == 0
    database = root / workflow.WORKFLOW_DIR / workflow.DATABASE_FILE
    assert database.is_file()
    assert stat.S_IMODE(database.stat().st_mode) == 0o600

    resumed = post.execute(
        search_root=tmp_path,
        repository_root=tmp_path / "repo",
        output_root=output,
        frozen_skeleton_sha256=post._digest(skeleton),
        frozen_restricted_input_sha256=post._digest(restricted_input),
        bundle_loader=loader,
        ffprobe=fake_tool,
        ffmpeg=fake_tool,
    )
    assert resumed["state"] == "POST_ACQUISITION_RECEIPTS_READY"
