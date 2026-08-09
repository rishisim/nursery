from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import wave

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/nursery_gemma4_referential_worker_v1_3.py"
SPEC = importlib.util.spec_from_file_location("test_gemma_template_worker_v1_3", PATH)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)

RUNNER_PATH = ROOT / "scripts/run_gemma4_prototype_common_schema_canary_v1_3.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("test_gemma_template_runner_v1_3", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)

COORDINATOR_PATH = ROOT / "scripts/nursery_gemma_substitution_calibration_v1_3.py"
COORDINATOR_SPEC = importlib.util.spec_from_file_location(
    "test_gemma_template_coordinator_v1_3", COORDINATOR_PATH
)
assert COORDINATOR_SPEC is not None and COORDINATOR_SPEC.loader is not None
coordinator = importlib.util.module_from_spec(COORDINATOR_SPEC)
sys.modules[COORDINATOR_SPEC.name] = coordinator
COORDINATOR_SPEC.loader.exec_module(coordinator)


class Processor:
    def __init__(self, rendered: str):
        self.rendered = rendered
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return self.rendered


def _observed() -> str:
    return "<s>" + worker.SYSTEM_PROMPT + worker.IMAGE_TOKEN * 5 + worker.USER_PROMPT + worker.AUDIO_TOKEN + "<end>"


def test_relocates_only_audio_token_and_preserves_declarative_messages() -> None:
    processor = Processor(_observed())
    corrected = worker._render_frozen_prompt(processor)
    assert corrected == "<s>" + worker.SYSTEM_PROMPT + worker.AUDIO_TOKEN + worker.IMAGE_TOKEN * 5 + worker.USER_PROMPT + "<end>"
    assert corrected.replace(worker.AUDIO_TOKEN, "") == _observed().replace(worker.AUDIO_TOKEN, "")
    assert [row["type"] for row in processor.messages[1]["content"]] == [
        "audio", "image", "image", "image", "image", "image", "text"
    ]


def test_requires_exact_observed_precondition_and_does_not_fallback() -> None:
    already_correct = worker.SYSTEM_PROMPT + worker.AUDIO_TOKEN + worker.IMAGE_TOKEN * 5 + worker.USER_PROMPT
    with pytest.raises(RuntimeError, match="E_TEMPLATE_ORDER_PRECONDITION"):
        worker._render_frozen_prompt(Processor(already_correct))
    noncontiguous = worker.SYSTEM_PROMPT + (worker.IMAGE_TOKEN + " ") * 5 + worker.USER_PROMPT + worker.AUDIO_TOKEN
    with pytest.raises(RuntimeError, match="E_TEMPLATE_ORDER_PRECONDITION"):
        worker._render_frozen_prompt(Processor(noncontiguous))


def test_prompt_schema_and_parser_are_byte_identical_to_legacy() -> None:
    assert worker.prompt_digest() == worker.legacy.prompt_digest()
    assert worker.exact_schema_digest() == worker.legacy.exact_schema_digest()
    assert worker.prompt_and_schema_digest() == worker.legacy.prompt_and_schema_digest()
    assert worker.ALLOWED == worker.legacy.ALLOWED
    assert worker._parse_exact is worker.legacy._parse_exact
    assert len(worker.template_order_correction_digest()) == 64


def _fake_tools(command: list[str], timeout: int = 120) -> None:
    del timeout
    if command[0] == str(runner.v12.legacy.SAY):
        Path(command[command.index("-o") + 1]).write_bytes(b"synthetic caf")
        return
    target = Path(command[-1])
    with wave.open(str(target), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 160000)


def _runtime() -> dict:
    value = {
        "schema_version": "legacy",
        "runtime_versions": dict(runner.v12.legacy.EXPECTED_RUNTIME),
        "interpreter_executable_sha256": "a" * 64,
        "mlx_vlm_installed_tree_manifest_sha256": "a" * 64,
        "ffmpeg_executable_sha256": "a" * 64,
        "worker_adapter_sha256": "a" * 64,
        "network_denial_profile_sha256": "a" * 64,
        "say_executable_sha256": "a" * 64,
        "os_build_and_voice_inventory_sha256": "a" * 64,
    }
    return value


def test_template_amendment_binds_exact_v12_failure_and_runner() -> None:
    amendment = runner._validate_template_amendment()
    assert amendment["exact_post_render_delta"]["relocated_substring_utf8"] == "<|audio|>"
    assert len("<|audio|>".encode("utf-8")) == 9
    assert runner._sha256_file(runner.V12_RUNNER) == runner.V12_RUNNER_SHA256
    assert runner._sha256_file(runner.V12_FAILURE) == runner.V12_FAILURE_SHA256


def test_synthetic_template_canary_and_activation_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seal = {"schema_version": "synthetic", "parser_mode": "PRIMARY"}
    monkeypatch.setattr(runner, "_public_preflight", lambda _mode: seal)
    monkeypatch.setattr(runner.v12.legacy, "_network_denied", lambda: True)
    monkeypatch.setattr(runner.v12.legacy, "_runtime_lock", _runtime)
    monkeypatch.setattr(runner.v12.legacy, "_quiet", _fake_tools)
    monkeypatch.setattr(runner.worker, "_load_instrument", lambda _path: (object(), object(), object(), object()))
    monkeypatch.setattr(runner.worker, "_infer_one", lambda *_args: ({}, True, "synthetic raw output"))
    monkeypatch.setattr(runner.v12.legacy, "_parser_negative_fixtures", lambda _mode: True)
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
    runner._validate_template_canary(receipt)
    runner._validate_template_activation(activation, hashlib.sha256(canary_bytes).hexdigest())
    assert activation["template_placeholder_order_binding"]["relocated_substring_bytes"] == 9
    assert activation["template_placeholder_order_binding"]["parser_correction_spent"] is False


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
    with pytest.raises(coordinator.GemmaSubstitutionError, match="E_TEMPLATE_ACTIVATION_MISSING"):
        coordinator._public_gemma_seal()


def test_coordinator_accepts_synthetic_shaped_template_success_receipt(
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
        "no_automatic_fallback_override_sha256": runner.v12.NO_FALLBACK_OVERRIDE_SHA256,
        "caf_transport_amendment_sha256": runner.v12.CAF_AMENDMENT_SHA256,
        "template_placeholder_order_amendment_sha256": runner.TEMPLATE_AMENDMENT_SHA256,
        "template_order_correction_digest": worker.template_order_correction_digest(),
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


def test_coordinator_preserves_planned_calibration_success_path() -> None:
    assert str(coordinator.PUBLIC_RECEIPT.relative_to(ROOT)) == (
        "output/nursery_program_convergence_v1/"
        "childlens_pseudo_calibration_gemma_substitution_receipt.json"
    )
