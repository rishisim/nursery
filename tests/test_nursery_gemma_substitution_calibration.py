from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import wave

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


worker = _load("test_gemma_worker", "scripts/nursery_gemma4_referential_worker.py")
canary = _load("test_gemma_canary", "scripts/run_gemma4_prototype_common_schema_canary.py")
coordinator = _load("test_gemma_coordinator", "scripts/nursery_gemma_substitution_calibration.py")


def _items(*, with_media_fd: bool = False) -> list[dict]:
    result = []
    for index in range(15):
        count = 10 if index < 2 else 9
        row = {
            "opaque_key": f"{index:064x}",
            "intervals": [(0.0, 60.0)],
            "candidate_windows": [(float(window), float(window + 1)) for window in range(count)],
            "baseline_candidates": [],
            "baseline_segments": [],
            "baseline_language": "de",
        }
        if with_media_fd:
            row["media_fd"] = 10 + index
        result.append(row)
    assert sum(len(row["candidate_windows"]) for row in result) == 137
    return result


def _common(index: int, start: float, end: float) -> dict:
    return {
        "window_index": index,
        "window_start_seconds": start,
        "window_end_seconds": end,
        "candidate_count_bin": "one",
        "visibility_bin": "clear",
        "referential_status": "visible_candidate",
        "lexical_support": "noun_object",
        "lag_event_unit_bin": "overlap",
        "schema_valid": True,
    }


def _qwen_documents(items: list[dict]) -> tuple[dict, dict]:
    asr = {
        "schema_version": coordinator.QWEN3_ASR_SCHEMA,
        "pseudo_labels_are_ground_truth": False,
        "network_disabled_during_inference": True,
        "items": [{"opaque_key": row["opaque_key"]} for row in items],
    }
    vlm = {
        "schema_version": coordinator.QWEN3_VLM_SCHEMA,
        "pseudo_labels_are_ground_truth": False,
        "network_disabled_during_inference": True,
        "items": [
            {
                "opaque_key": row["opaque_key"],
                "candidates": [
                    _common(index, float(start), float(end))
                    for index, (start, end) in enumerate(row["candidate_windows"])
                ],
            }
            for row in items
        ],
    }
    return asr, vlm


def _private_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.chmod(path, 0o600)


def test_prompt_and_exact_schema_match_frozen_contract() -> None:
    assert worker.prompt_digest() == canary.PROMPT_SHA256
    assert worker.exact_schema_digest() == canary.EXACT_OUTPUT_SCHEMA_SHA256
    messages = worker._prompt_messages()
    assert [row["role"] for row in messages] == ["system", "user"]
    assert [row["type"] for row in messages[1]["content"]] == [
        "audio", "image", "image", "image", "image", "image", "text"
    ]


def test_rendered_prompt_fails_closed_on_content_reordering() -> None:
    class Processor:
        def __init__(self, valid: bool):
            self.valid = valid

        def apply_chat_template(self, messages, **_):
            if self.valid:
                return worker.SYSTEM_PROMPT + "<|audio|>" + "<|image|>" * 5 + worker.USER_PROMPT
            return worker.SYSTEM_PROMPT + "<|image|>" * 5 + worker.USER_PROMPT + "<|audio|>"

    assert "<|audio|>" in worker._render_frozen_prompt(Processor(True))
    with pytest.raises(RuntimeError, match="E_CONTENT_ORDER"):
        worker._render_frozen_prompt(Processor(False))


def test_exact_parser_rejects_prose_duplicates_extras_and_two_objects() -> None:
    valid = json.dumps({field: next(iter(values)) for field, values in worker.ENUMS.items()}, separators=(",", ":"))
    assert set(worker._parse_exact(valid)) == set(worker.ALLOWED)
    invalid = [
        "prose " + valid,
        valid + valid,
        valid[:-1] + ',"extra":"x"}',
        valid[:-1] + ',"candidate_count_bin":"one"}',
    ]
    for payload in invalid:
        with pytest.raises(RuntimeError, match="E_SCHEMA"):
            worker._parse_exact(payload)
    fenced = "```json\n" + valid + "\n```"
    with pytest.raises(RuntimeError, match="E_SCHEMA"):
        worker._parse_exact(fenced, "PRIMARY")
    assert worker._parse_exact(fenced, "ONE_OUTER_FENCE")


def test_job_is_exact_15_900_137_and_contains_no_transcript() -> None:
    items = _items(with_media_fd=True)
    job = coordinator._make_job(items, 0, "PRIMARY")
    clean, resume, parser = worker._validate_job(job)
    assert len(clean) == 15
    assert sum(len(row["candidate_windows"]) for row in clean) == 137
    assert resume == 0 and parser == "PRIMARY"
    encoded = json.dumps(job).casefold()
    assert "transcript" not in encoded
    assert "media_relpath" not in encoded
    job["candidate_window_count"] = 136
    with pytest.raises(RuntimeError, match="E_JOB"):
        worker._validate_job(job)


def test_exact_audio_slice_uses_half_open_16khz_sample_indices(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 16000)
    worker._slice_exact_audio(source, 0.1, 0.2, target)
    with wave.open(str(target), "rb") as stream:
        assert stream.getframerate() == 16000
        assert stream.getnframes() == 1600


def test_checkpoint_resume_accepts_only_canonical_prefix(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(mode=0o700)
    expected = coordinator._expected_windows(_items())
    raw = "{}"
    row = {
        "schema_version": worker.CHECKPOINT_SCHEMA,
        **coordinator._checkpoint_expected_row(0, expected[0]),
        "candidate_count_bin": "one",
        "visibility_bin": "clear",
        "referential_status": "visible_candidate",
        "lexical_support": "noun_object",
        "lag_event_unit_bin": "overlap",
        "schema_valid": True,
        "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "raw_response": raw,
        "pseudo_labels_are_ground_truth": False,
        "human_validation": False,
    }
    directory_fd = os.open(checkpoint, os.O_RDONLY)
    try:
        worker._write_checkpoint(directory_fd, 0, row)
    finally:
        os.close(directory_fd)
    assert len(coordinator._load_checkpoint_prefix(checkpoint, expected)) == 1
    (checkpoint / "000000.json").rename(checkpoint / "000001.json")
    with pytest.raises(coordinator.GemmaSubstitutionError, match="E_CHECKPOINT_PREFIX"):
        coordinator._load_checkpoint_prefix(checkpoint, expected)


def test_qwen3_loader_is_private_read_only_and_never_regenerates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items = _items()
    asr, vlm = _qwen_documents(items)
    namespace = tmp_path / "provisional_calibration_v1"
    asr_path = namespace / "qwen3_asr_restricted.json"
    vlm_path = namespace / "qwen3_vl_restricted.json"
    _private_json(asr_path, asr)
    _private_json(vlm_path, vlm)
    sample_digest = "a" * 64
    binding = {
        "schema_version": "nursery-childlens-pseudo-calibration-binding-v1",
        "sample_digest": sample_digest,
        "protocol_sha256": coordinator.EXPECTED_PROTOCOL_SHA256,
        "worker_sha256": coordinator.EXPECTED_QWEN3_WORKER_SHA256,
        "asr_revision": coordinator.EXPECTED_QWEN3_ASR_REVISION,
        "aligner_revision": coordinator.EXPECTED_QWEN3_ALIGNER_REVISION,
        "vlm_revision": coordinator.EXPECTED_QWEN3_VLM_REVISION,
        "asr_output_sha256": coordinator._sha256_file(asr_path),
        "vlm_output_sha256": coordinator._sha256_file(vlm_path),
        "network_disabled": True,
        "pseudo_labels_are_ground_truth": False,
    }
    _private_json(namespace / "execution_binding.json", binding)
    monkeypatch.setattr(
        coordinator.base,
        "_challenger_outputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Qwen3 rerun forbidden")),
    )
    loaded_asr, loaded_vlm, hashes = coordinator._load_qwen3_outputs_read_only(
        tmp_path, items, sample_digest
    )
    assert loaded_asr["schema_version"] == coordinator.QWEN3_ASR_SCHEMA
    assert loaded_vlm["schema_version"] == coordinator.QWEN3_VLM_SCHEMA
    assert set(hashes) == {"binding_sha256", "asr_output_sha256", "vlm_output_sha256"}
    assert stat_mode(asr_path) == 0o600 and stat_mode(vlm_path) == 0o600


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_runtime_erratum_is_bound_before_public_canary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        canary,
        "_tree_manifest",
        lambda _path: (canary.ARTIFACT_MANIFEST_SHA256, 5_179_241_512, 10),
    )
    seal = canary._public_preflight("PRIMARY")
    assert seal["canary_contract_sha256"] == canary.CANARY_CONTRACT_SHA256
    assert seal["runtime_pin_erratum_sha256"] == canary.RUNTIME_ERRATUM_SHA256


def _public_canary_and_runtime() -> tuple[dict, dict, bytes, bytes]:
    hex64 = "a" * 64
    runtime = {
        "schema_version": "nursery-gemma4-e4b-prototype-runtime-lock-v1",
        "runtime_pin_erratum_path": "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_runtime_pin_erratum_v1_1.json",
        "runtime_pin_erratum_sha256": canary.RUNTIME_ERRATUM_SHA256,
        "runtime_versions": dict(canary.EXPECTED_RUNTIME),
        "distributions": [{"name": "fixture", "version": "1", "record_sha256": hex64}],
        "interpreter_executable_sha256": hex64,
        "mlx_vlm_installed_tree_manifest_sha256": hex64,
        "ffmpeg_executable_sha256": hex64,
        "worker_adapter_sha256": hex64,
        "network_denial_profile_sha256": hex64,
        "say_executable_sha256": hex64,
        "os_build_and_voice_inventory_sha256": hex64,
    }
    runtime_bytes = canary._json_document(runtime)
    receipt = {
        "schema_version": "nursery-gemma4-e4b-prototype-common-schema-public-canary-receipt-v1",
        "status": "PUBLIC_COMMON_SCHEMA_CANARY_PASS",
        "amendment_sha256": canary.OPAQUE_AMENDMENT_SHA256,
        "canary_contract_sha256": canary.CANARY_CONTRACT_SHA256,
        "runtime_pin_erratum_path": "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_runtime_pin_erratum_v1_1.json",
        "runtime_pin_erratum_sha256": canary.RUNTIME_ERRATUM_SHA256,
        "instrument_id": canary.validator.OPAQUE_INSTRUMENT_ID,
        "conversion_revision": canary.CONVERSION_REVISION,
        "artifact_manifest_sha256": canary.ARTIFACT_MANIFEST_SHA256,
        "restricted_common_schema_sha256": canary.BASELINE_COMMON_SCHEMA_SHA256,
        "exact_output_schema_sha256": canary.EXACT_OUTPUT_SCHEMA_SHA256,
        "tts_executable_sha256": hex64,
        "os_build_and_voice_inventory_sha256": hex64,
        "fixture_manifest_sha256": hex64,
        "self_generated_public_fixture_only": True,
        "joint_audio_plus_five_frames": True,
        "five_frames_consumed_in_order": True,
        "audio_consumed": True,
        "exact_schema_valid": True,
        "all_parser_negative_fixtures_pass": True,
        "network_denial_sentinel_passed": True,
        "no_external_request": True,
        "corrections_used": 0,
        "correction_scope": "NONE",
        "semantic_tuning_performed": False,
        "semantic_quality_gate_used": False,
        "restricted_payload_accessed": False,
        "download_performed": False,
        "clickthrough_accepted": False,
        "hosted_or_cloud_inference_used": False,
        "restricted_inference_authorized_by_canary": False,
        "runtime_lock_file_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
    }
    return receipt, runtime, canary._json_document(receipt), runtime_bytes


def test_public_activation_builder_validates_effective_schema_without_restricted_access() -> None:
    receipt, runtime, receipt_bytes, runtime_bytes = _public_canary_and_runtime()
    activation = canary._build_activation_receipt(
        receipt,
        runtime,
        receipt_bytes,
        runtime_bytes,
        {
            "launch_rehash_passed": True,
            "launch_disk_floor_passed": True,
            "launch_memory_pressure_passed": True,
        },
    )
    activation_schema = canary._read_json(canary.ACTIVATION_SCHEMA)
    erratum = canary._read_json(canary.RUNTIME_ERRATUM)
    assert not canary.validator.validate_activation(
        activation,
        activation_schema,
        canary.ACTIVATION_SCHEMA_SHA256,
        erratum,
        canary.RUNTIME_ERRATUM_SHA256,
        hashlib.sha256(receipt_bytes).hexdigest(),
    )
    assert activation["schema_version"] == canary.validator.EFFECTIVE_ACTIVATION_SCHEMA
    assert activation["immutable_bindings"]["runtime_pin_erratum"]["sha256"] == canary.RUNTIME_ERRATUM_SHA256
    assert activation["qwen3_read_only_binding"]["restricted_digest_exported"] is False
    assert activation["privacy_and_scientific_attestations"]["restricted_content_accessed_to_issue_receipt"] is False


def test_activation_builder_fails_closed_and_activation_write_is_once_only(tmp_path: Path) -> None:
    receipt, runtime, receipt_bytes, runtime_bytes = _public_canary_and_runtime()
    receipt["status"] = "PUBLIC_COMMON_SCHEMA_CANARY_REVISE"
    with pytest.raises(canary.CanaryError, match="E_ACTIVATION_PREREQUISITE"):
        canary._build_activation_receipt(
            receipt,
            runtime,
            receipt_bytes,
            runtime_bytes,
            {
                "launch_rehash_passed": True,
                "launch_disk_floor_passed": True,
                "launch_memory_pressure_passed": True,
            },
        )
    target = tmp_path / "activation.json"
    canary._atomic_once(target, b"first")
    with pytest.raises(FileExistsError):
        canary._atomic_once(target, b"second")
    assert target.read_bytes() == b"first"


def test_activation_failure_occurs_before_quarantine_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coordinator, "ACTIVATION", tmp_path / "missing-activation.json")
    monkeypatch.setattr(
        coordinator.base,
        "_restricted_inputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("quarantine opened")),
    )
    with pytest.raises(coordinator.GemmaSubstitutionError, match="E_ACTIVATION_MISSING"):
        coordinator._public_gemma_seal()


def _public_receipt(status: str, activation_sha: str) -> dict:
    passed = status == "CALIBRATION_PASS"
    return {
        "schema_version": coordinator.PUBLIC_SCHEMA,
        "status": status,
        "decision": "CALIBRATION_PASS_INTERNAL_PROTOTYPE" if passed else "CALIBRATION_REVISE_INTERNAL_PROTOTYPE",
        "activation_receipt_sha256": activation_sha,
        "no_automatic_fallback_override_sha256": coordinator.replacement_validator.NO_FALLBACK_OVERRIDE_SHA256,
        "referential_instrument_ids": [coordinator.QWEN3_INSTRUMENT, coordinator.GEMMA_INSTRUMENT],
        "qwen3_reused_read_only": True,
        "primary_item_count": 15,
        "primary_total_seconds": 900,
        "candidate_window_count": 137,
        "sample_reselected": False,
        "all_frozen_calibration_gates_passed": passed,
        "aggregate_only": True,
        "simulator_oracle_only_evaluation_truth": True,
        "pseudo_labels_are_ground_truth": False,
        "human_validation_claimed": False,
        "scientific_outcome_run": False,
        "scientific_endpoint_opened": False,
        "semantic_tuning_performed": False,
        "gate_failures": [] if passed else ["SCHEMA_VALIDITY"],
        "public_export": coordinator._public_export_policy(),
    }


@pytest.mark.parametrize("status", ["CALIBRATION_PASS", "CALIBRATION_REVISE"])
def test_public_status_mapping_and_validator(status: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    activation = tmp_path / "activation.json"
    activation.write_text("{}", encoding="utf-8")
    activation_sha = hashlib.sha256(activation.read_bytes()).hexdigest()
    monkeypatch.setattr(coordinator, "ACTIVATION", activation)
    coordinator._validate_public_receipt(_public_receipt(status, activation_sha))


def test_public_sources_have_no_cloud_path_or_public_restricted_digest() -> None:
    sources = "\n".join(
        (ROOT / path).read_text(encoding="utf-8").casefold()
        for path in (
            "scripts/nursery_gemma4_referential_worker.py",
            "scripts/run_gemma4_prototype_common_schema_canary.py",
            "scripts/nursery_gemma_substitution_calibration.py",
        )
    )
    for token in ("api.openai", "anthropic", "requests.post", "httpx.post"):
        assert token not in sources
    assert '"restricted_digest_exported") is not false' in sources
    assert '"qwen3_private_digest_exported": false' in sources
    assert '"raw_model_output_exported": false' in sources


def test_historical_code_and_receipts_remain_byte_immutable() -> None:
    expected = {
        "scripts/nursery_pseudo_calibration.py": coordinator.EXPECTED_BASE_SCRIPT_SHA256,
        "scripts/nursery_local_challenger_worker.py": coordinator.EXPECTED_QWEN3_WORKER_SHA256,
        "docs/nursery_program_convergence_v1/frozen_childlens_pseudo_calibration_protocol.json": coordinator.EXPECTED_PROTOCOL_SHA256,
        "output/nursery_program_convergence_v1/instrument_activation_receipt.json": coordinator.EXPECTED_HISTORICAL_ACTIVATION_SHA256,
        "output/nursery_program_convergence_v1/childlens_pseudo_calibration_receipt.json": coordinator.EXPECTED_HISTORICAL_RECEIPT_SHA256,
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
