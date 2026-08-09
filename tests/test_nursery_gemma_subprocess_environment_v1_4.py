from __future__ import annotations

import hashlib
import importlib.util
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


runner = _load(
    "test_gemma_subprocess_environment_runner_v1_4",
    "scripts/run_gemma4_prototype_common_schema_canary_v1_4.py",
)
coordinator = _load(
    "test_gemma_subprocess_environment_coordinator_v1_4",
    "scripts/nursery_gemma_substitution_calibration_v1_4.py",
)


def _fake_tools(command: list[str], timeout: int = 120) -> None:
    del timeout
    if command[0] == str(runner.v13.v12.legacy.SAY):
        Path(command[command.index("-o") + 1]).write_bytes(b"synthetic caf")
        return
    target = Path(command[-1])
    with wave.open(str(target), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 160000)


def _runtime() -> dict:
    return {
        "schema_version": "legacy",
        "runtime_versions": dict(runner.v13.v12.legacy.EXPECTED_RUNTIME),
        "interpreter_executable_sha256": "a" * 64,
        "mlx_vlm_installed_tree_manifest_sha256": "a" * 64,
        "ffmpeg_executable_sha256": "a" * 64,
        "worker_adapter_sha256": "a" * 64,
        "network_denial_profile_sha256": "a" * 64,
        "say_executable_sha256": "a" * 64,
        "os_build_and_voice_inventory_sha256": "a" * 64,
    }


def test_amendment_binds_exact_v13_failure_runner_worker_and_environment() -> None:
    amendment = runner._validate_environment_amendment()
    contract = amendment["exact_subprocess_environment_delta"]["scrubber_mandatory_environment_contract"]
    assert runner._sha256_file(runner.V13_RUNNER) == runner.V13_RUNNER_SHA256
    assert runner._sha256_file(runner.WORKER) == runner.WORKER_SHA256
    assert runner._sha256_file(runner.V13_FAILURE) == runner.V13_FAILURE_SHA256
    assert contract["exact_returned_mapping"] == runner.EXPECTED_ENVIRONMENT
    assert runner._sha256_bytes(runner._canonical(runner.EXPECTED_ENVIRONMENT)) == runner.ENVIRONMENT_CONTRACT_SHA256


def test_public_launch_uses_zero_argument_exact_unmodified_scrubbed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Backend:
        def command(self, argv):
            return argv

    class Lock:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class StopBeforeLaunch(RuntimeError):
        pass

    calls = {"scrubber": 0, "popen": 0}

    def scrubber():
        calls["scrubber"] += 1
        return dict(runner.EXPECTED_ENVIRONMENT)

    def popen(*_args, **kwargs):
        calls["popen"] += 1
        assert kwargs["env"] == runner.EXPECTED_ENVIRONMENT
        assert kwargs["env"] is not runner.EXPECTED_ENVIRONMENT
        raise StopBeforeLaunch

    monkeypatch.setattr(runner, "_public_preflight", lambda _mode: {"seal": "synthetic"})
    monkeypatch.setattr(runner.v13.v12.legacy.firewall.NetworkIsolationBackend, "detect", lambda: Backend())
    monkeypatch.setattr(runner.v13.v12.legacy.firewall, "verify_network_isolation", lambda _backend: None)
    monkeypatch.setattr(runner.v13.v12.legacy, "_launch_resource_recheck", lambda: {})
    monkeypatch.setattr(runner.v13.v12.legacy, "_exclusive_public_mps_lock", Lock)
    monkeypatch.setattr(runner.v13.v12.legacy.firewall, "scrubbed_subprocess_environment", scrubber)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    with pytest.raises(StopBeforeLaunch):
        runner.public_execute("PRIMARY")
    assert calls == {"scrubber": 1, "popen": 1}


def test_environment_assertion_fails_before_popen(monkeypatch: pytest.MonkeyPatch) -> None:
    class Backend:
        def command(self, argv):
            return argv

    class Lock:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    bad = dict(runner.EXPECTED_ENVIRONMENT)
    bad["DO_NOT_TRACK"] = "0"
    monkeypatch.setattr(runner, "_public_preflight", lambda _mode: {"seal": "synthetic"})
    monkeypatch.setattr(runner.v13.v12.legacy.firewall.NetworkIsolationBackend, "detect", lambda: Backend())
    monkeypatch.setattr(runner.v13.v12.legacy.firewall, "verify_network_isolation", lambda _backend: None)
    monkeypatch.setattr(runner.v13.v12.legacy, "_launch_resource_recheck", lambda: {})
    monkeypatch.setattr(runner.v13.v12.legacy, "_exclusive_public_mps_lock", Lock)
    monkeypatch.setattr(runner.v13.v12.legacy.firewall, "scrubbed_subprocess_environment", lambda: bad)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Popen reached")),
    )
    with pytest.raises(runner.EnvironmentCanaryError, match="E_SUBPROCESS_ENV"):
        runner.public_execute("PRIMARY")


def test_synthetic_environment_canary_and_activation_validate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seal = {"schema_version": "synthetic", "parser_mode": "PRIMARY"}
    monkeypatch.setattr(runner, "_public_preflight", lambda _mode: seal)
    monkeypatch.setattr(runner.v13.v12.legacy, "_network_denied", lambda: True)
    monkeypatch.setattr(runner.v13.v12.legacy, "_runtime_lock", _runtime)
    monkeypatch.setattr(runner.v13.v12.legacy, "_quiet", _fake_tools)
    monkeypatch.setattr(runner.worker, "_load_instrument", lambda _path: (object(), object(), object(), object()))
    monkeypatch.setattr(runner.worker, "_infer_one", lambda *_args: ({}, True, "synthetic raw output"))
    monkeypatch.setattr(runner.v13.v12.legacy, "_parser_negative_fixtures", lambda _mode: True)
    result = runner.sandboxed_execute(seal)
    receipt = dict(result["receipt"])
    runtime = result["runtime_lock"]
    runtime_bytes = runner._json_document(runtime)
    receipt["runtime_lock_file_sha256"] = hashlib.sha256(runtime_bytes).hexdigest()
    canary_bytes = runner._json_document(receipt)
    activation = runner._build_activation_receipt(
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
    runner._validate_environment_canary(receipt)
    runner._validate_environment_activation(activation, hashlib.sha256(canary_bytes).hexdigest())
    binding = activation["subprocess_environment_correction_binding"]
    assert binding["scrubber_positional_argument_count"] == 0
    assert binding["scrubber_extra_mapping"] is None
    assert binding["parser_correction_spent"] is False


def test_coordinator_fails_before_restricted_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(coordinator, "ACTIVATION", tmp_path / "missing-activation.json")
    monkeypatch.setattr(coordinator, "PUBLIC_CANARY", tmp_path / "missing-canary.json")
    monkeypatch.setattr(
        coordinator.legacy.base,
        "_restricted_inputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("restricted discovery")),
    )
    with pytest.raises(coordinator.GemmaSubstitutionError, match="E_ENVIRONMENT_ACTIVATION_MISSING"):
        coordinator._public_gemma_seal()


def test_coordinator_accepts_synthetic_shaped_environment_success_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    activation = tmp_path / "activation.json"
    activation.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(coordinator.legacy, "ACTIVATION", activation)
    receipt = {
        "schema_version": coordinator.legacy.PUBLIC_SCHEMA,
        "status": "CALIBRATION_PASS",
        "decision": "CALIBRATION_PASS_INTERNAL_PROTOTYPE",
        "activation_receipt_sha256": hashlib.sha256(activation.read_bytes()).hexdigest(),
        "no_automatic_fallback_override_sha256": runner.v13.v12.NO_FALLBACK_OVERRIDE_SHA256,
        "caf_transport_amendment_sha256": runner.v13.v12.CAF_AMENDMENT_SHA256,
        "template_placeholder_order_amendment_sha256": runner.v13.TEMPLATE_AMENDMENT_SHA256,
        "template_order_correction_digest": runner.worker.template_order_correction_digest(),
        "subprocess_environment_correction_amendment_sha256": runner.ENVIRONMENT_AMENDMENT_SHA256,
        "subprocess_environment_contract_sha256": runner.ENVIRONMENT_CONTRACT_SHA256,
        "referential_instrument_ids": [
            coordinator.legacy.QWEN3_INSTRUMENT,
            coordinator.legacy.GEMMA_INSTRUMENT,
        ],
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


def test_coordinator_preserves_original_planned_success_path() -> None:
    assert str(coordinator.PUBLIC_RECEIPT.relative_to(ROOT)) == (
        "output/nursery_program_convergence_v1/"
        "childlens_pseudo_calibration_gemma_substitution_receipt.json"
    )
