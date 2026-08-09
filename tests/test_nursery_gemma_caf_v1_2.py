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


caf = _load("test_gemma_caf_runner_v1_2", "scripts/run_gemma4_prototype_common_schema_canary_v1_2.py")
coordinator = _load("test_gemma_caf_coordinator_v1_2", "scripts/nursery_gemma_substitution_calibration_v1_2.py")


def _fake_local_tools(command: list[str], timeout: int = 120) -> None:
    del timeout
    if command[0] == str(caf.legacy.SAY):
        target = Path(command[command.index("-o") + 1])
        assert target.suffix == ".caf"
        target.write_bytes(b"public synthetic caf")
        return
    assert command[0] == str(caf.legacy.FFMPEG)
    source = Path(command[command.index("-i") + 1])
    target = Path(command[-1])
    assert source.suffix == ".caf" and source.is_file() and target.suffix == ".wav"
    with wave.open(str(target), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 160000)


def _runtime_lock() -> dict:
    hex64 = "a" * 64
    return {
        "schema_version": caf.RUNTIME_SCHEMA,
        "runtime_pin_erratum_path": "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_runtime_pin_erratum_v1_1.json",
        "runtime_pin_erratum_sha256": caf.legacy.RUNTIME_ERRATUM_SHA256,
        "runtime_versions": dict(caf.legacy.EXPECTED_RUNTIME),
        "distributions": [{"name": "fixture", "version": "1", "record_sha256": hex64}],
        "interpreter_executable_sha256": hex64,
        "mlx_vlm_installed_tree_manifest_sha256": hex64,
        "ffmpeg_executable_sha256": hex64,
        "worker_adapter_sha256": hex64,
        "network_denial_profile_sha256": hex64,
        "say_executable_sha256": hex64,
        "os_build_and_voice_inventory_sha256": hex64,
        "caf_transport_amendment_path": str(caf.CAF_AMENDMENT.relative_to(ROOT)),
        "caf_transport_amendment_sha256": caf.CAF_AMENDMENT_SHA256,
        "no_automatic_fallback_override_sha256": caf.NO_FALLBACK_OVERRIDE_SHA256,
        "legacy_runner_sha256": caf.LEGACY_RUNNER_SHA256,
        "runner_sha256": hex64,
        "intermediate_container": "CAF",
        "final_container": "WAV",
    }


def test_caf_amendment_override_and_prior_attempt_are_exact() -> None:
    amendment, override = caf._validate_caf_amendment()
    assert amendment["exact_transport_delta"]["semantic_fields_changed"] == 0
    assert override["failure_behavior"]["restricted_calibration_fallback_allowed"] is False
    assert caf._sha256_file(caf.LEGACY_RUNNER) == caf.LEGACY_RUNNER_SHA256
    assert caf._sha256_file(caf.legacy.FAILURE) == caf.LEGACY_FAILURE_SHA256


def test_caf_fixture_preserves_frames_and_validates_final_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(caf.legacy, "_quiet", _fake_local_tools)
    audio, frames, manifest = caf._create_fixture(tmp_path)
    assert audio.is_file() and not (tmp_path / "german.caf").exists()
    assert manifest["intermediate_container"] == "CAF"
    assert manifest["audio_contract"] == {
        "container": "WAV", "channels": 1, "sample_width_bytes": 2,
        "sample_rate_hz": 16000, "sample_count": 160000,
    }
    assert manifest["frame_sha256s_in_order"] == caf.LEGACY_FRAME_SHA256S
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in frames] == caf.LEGACY_FRAME_SHA256S


def test_caf_synthetic_receipt_builds_valid_additive_activation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seal = {"schema_version": "synthetic", "parser_mode": "PRIMARY"}
    monkeypatch.setattr(caf, "_public_preflight", lambda _mode: seal)
    monkeypatch.setattr(caf.legacy, "_network_denied", lambda: True)
    monkeypatch.setattr(caf.legacy, "_runtime_lock", _runtime_lock)
    monkeypatch.setattr(caf.legacy, "_quiet", _fake_local_tools)
    monkeypatch.setattr(caf.legacy.worker, "_load_instrument", lambda _path: (object(), object(), object(), object()))
    monkeypatch.setattr(caf.legacy.worker, "_infer_one", lambda *_args: ({}, True, "synthetic raw output"))
    monkeypatch.setattr(caf.legacy, "_parser_negative_fixtures", lambda _mode: True)
    result = caf.sandboxed_execute(seal)
    receipt = dict(result["receipt"])
    runtime = result["runtime_lock"]
    runtime_bytes = caf._json_document(runtime)
    receipt["runtime_lock_file_sha256"] = hashlib.sha256(runtime_bytes).hexdigest()
    canary_bytes = caf._json_document(receipt)
    activation = caf._build_activation_receipt(
        receipt,
        runtime,
        canary_bytes,
        runtime_bytes,
        {
            "launch_rehash_passed": True,
            "launch_disk_floor_passed": True,
            "launch_memory_pressure_passed": True,
        },
    )
    caf._validate_caf_canary(receipt)
    caf._validate_caf_activation(activation, hashlib.sha256(canary_bytes).hexdigest())
    assert activation["schema_version"] == caf.ACTIVATION_SCHEMA
    assert activation["caf_transport_binding"]["intermediate_container"] == "CAF"
    assert activation["no_automatic_fallback_override"]["automatic_fallback_allowed"] is False


def test_coordinator_is_additive_and_fails_before_restricted_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coordinator, "ACTIVATION", tmp_path / "missing-activation.json")
    monkeypatch.setattr(coordinator, "PUBLIC_CANARY", tmp_path / "missing-canary.json")
    monkeypatch.setattr(
        coordinator.legacy.base,
        "_restricted_inputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("restricted discovery")),
    )
    with pytest.raises(coordinator.GemmaSubstitutionError, match="E_CAF_ACTIVATION_MISSING"):
        coordinator._public_gemma_seal()
    assert coordinator.LEGACY_COORDINATOR_SHA256 == "4fdacb4a0bf4e03627c4c036ec65107bb249bf0a4b35564f3cf7a5a58dbeeeed"


def test_coordinator_accepts_synthetic_shaped_caf_success_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    activation = tmp_path / "activation.json"
    activation.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(coordinator.legacy, "ACTIVATION", activation)
    receipt = {
        "schema_version": coordinator.legacy.PUBLIC_SCHEMA,
        "status": "CALIBRATION_PASS",
        "decision": "CALIBRATION_PASS_INTERNAL_PROTOTYPE",
        "activation_receipt_sha256": hashlib.sha256(activation.read_bytes()).hexdigest(),
        "no_automatic_fallback_override_sha256": caf.NO_FALLBACK_OVERRIDE_SHA256,
        "caf_transport_amendment_sha256": caf.CAF_AMENDMENT_SHA256,
        "referential_instrument_ids": [coordinator.legacy.QWEN3_INSTRUMENT, coordinator.legacy.GEMMA_INSTRUMENT],
        "qwen3_reused_read_only": True,
        "primary_item_count": 15,
        "primary_total_seconds": 900,
        "candidate_window_count": 137,
        "sample_reselected": False,
        "all_frozen_calibration_gates_passed": True,
        "aggregate_only": True,
        "simulator_oracle_only_evaluation_truth": True,
        "pseudo_labels_are_ground_truth": False,
        "human_validation_claimed": False,
        "scientific_outcome_run": False,
        "scientific_endpoint_opened": False,
        "semantic_tuning_performed": False,
        "automatic_fallback_allowed": False,
        "scientific_endpoint_if_gemma_fails": False,
        "public_export": {
            "minimum_cluster_k": 5,
            "complementary_suppression": True,
            "rate_rounding_step": 0.1,
            "lag_rounding_step_event_units": 0.5,
            "raw_counts": False,
            "item_level_results": False,
            "transcript_or_lexical_content": False,
            "frames_or_audio": False,
            "identifiers_or_paths": False,
            "exact_timestamps": False,
            "confidence_or_raw_payload": False,
        },
    }
    coordinator._validate_public_receipt(receipt)


def test_new_sources_bind_no_fallback_and_never_name_cloud_content_paths() -> None:
    sources = "\n".join(
        (ROOT / path).read_text(encoding="utf-8").casefold()
        for path in (
            "scripts/run_gemma4_prototype_common_schema_canary_v1_2.py",
            "scripts/nursery_gemma_substitution_calibration_v1_2.py",
        )
    )
    assert caf.NO_FALLBACK_OVERRIDE_SHA256 in sources
    for token in ("api.openai", "anthropic", "requests.post", "httpx.post"):
        assert token not in sources
    assert "automatic_fallback_allowed" in sources
    assert str(coordinator.PUBLIC_RECEIPT.relative_to(ROOT)) == "output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_receipt.json"
