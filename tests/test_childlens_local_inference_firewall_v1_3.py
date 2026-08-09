from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "childlens_local_inference_firewall_v1_3.py"
SPEC = importlib.util.spec_from_file_location("childlens_local_inference_firewall_v1_3", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
import sys

sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

SYNTH_PATH = ROOT / "scripts" / "synthesize_childlens_terminal_v1_3.py"
SYNTH_SPEC = importlib.util.spec_from_file_location("synthesize_childlens_terminal_v1_3_for_firewall", SYNTH_PATH)
assert SYNTH_SPEC is not None and SYNTH_SPEC.loader is not None
SYNTH = importlib.util.module_from_spec(SYNTH_SPEC)
sys.modules[SYNTH_SPEC.name] = SYNTH
SYNTH_SPEC.loader.exec_module(SYNTH)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _private_file(path: Path, data: bytes, mode: int = 0o600) -> Path:
    path.write_bytes(data)
    path.chmod(mode)
    return path


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    quarantine = tmp_path / "private"
    quarantine.mkdir(mode=0o700)
    (quarantine / ".metadata_never_index").write_bytes(b"")
    (quarantine / ".metadata_never_index").chmod(0o600)
    return repository.resolve(), quarantine.resolve()


def _receipt(
    executable: Path,
    model: Path,
    *,
    profile: str = "silero_vad",
    instrument_key: str = "silero-vad-fixed",
) -> dict[str, object]:
    profile_row = MODULE.FIXED_ADAPTER_PROFILES[profile]
    version = str(profile_row.get("required_model_revision", "6.2.0"))
    return {
        "schema_version": MODULE.INSTRUMENT_RECEIPT_SCHEMA,
        "instrument_key": instrument_key,
        "adapter_profile": profile,
        "task": profile_row["task"],
        "resource_class": profile_row["resource_class"],
        "software": {
            "name": "offline-adapter",
            "version": "1.0.0",
            "code_sha256": "1" * 64,
            "executable_sha256": _sha(executable),
            "adapter_contract_sha256": MODULE.ADAPTER_CONTRACT_SHA256,
            "license_identifier": "MIT",
            "license_evidence_sha256": "3" * 64,
        },
        "model": {
            "name": str(profile_row["model_family"]),
            "version": version,
            "artifact_set_sha256": _sha(model),
            "license_identifier": "MIT",
            "license_evidence_sha256": "4" * 64,
            "authoritative_source_verified": True,
        },
        "acquisition": {
            "completed_before_restricted_processing": True,
            "hashes_verified": True,
            "license_review_status": "PASS",
            "clickthrough_accepted_by_automation": False,
            "telemetry_disabled": True,
        },
        "boundary": {
            "offline_only": True,
            "external_api_allowed": False,
            "external_upload_allowed": False,
            "network_isolation_required": True,
            "restricted_outputs_quarantine_only": True,
            "learner_weight_ancestry_allowed": False,
            "learner_tokenizer_or_vocabulary_ancestry_allowed": False,
            "learner_embedding_or_feature_ancestry_allowed": False,
            "learner_score_or_confidence_ancestry_allowed": False,
            "primary_evaluation_truth_allowed": False,
            "simulator_oracle_replacement_allowed": False,
        },
    }


def _complete_task_counts() -> dict[str, dict[str, int]]:
    return {
        task: {"PENDING": 0, "RUNNING": 0, "COMPLETE": 15, "FAILED": 0}
        for task in MODULE.TASKS
    }


class _IdentityBackend:
    name = "MACOS_SANDBOX_DENY_NETWORK"

    @staticmethod
    def command(argv: object, **_: object) -> list[str]:
        return list(argv)  # type: ignore[arg-type]


class _SyntheticBackend:
    name = "MACOS_SANDBOX_DENY_NETWORK"

    @staticmethod
    def command(argv: object, **_: object) -> list[str]:
        return ["synthetic-isolator", *list(argv)]  # type: ignore[arg-type]


class _SyntheticProcessRunner:
    def __init__(self) -> None:
        self.sentinel_calls = 0
        self.adapter_calls = 0
        self.environments: list[dict[str, str]] = []

    def __call__(self, command: list[str], **kwargs: object) -> SimpleNamespace:
        self.environments.append(dict(kwargs["env"]))  # type: ignore[arg-type]
        if "-c" in command:
            self.sentinel_calls += 1
            return SimpleNamespace(returncode=0)
        self.adapter_calls += 1
        descriptors = kwargs["pass_fds"]
        output_fd = descriptors[-1]  # type: ignore[index]
        os.write(output_fd, b'{"synthetic_only":true}')
        return SimpleNamespace(returncode=0)


def test_fixed_adapter_profiles_cover_frozen_local_stack() -> None:
    assert set(MODULE.FIXED_ADAPTER_PROFILES) == {
        "silero_vad",
        "whisper_cpp_large_v3_turbo",
        "conservative_role_aid",
        "qwen2_vl_2b_instruct",
    }
    qwen = MODULE.FIXED_ADAPTER_PROFILES["qwen2_vl_2b_instruct"]
    assert qwen["resource_class"] == "MPS_HEAVY"
    assert qwen["required_model_revision"] == "895c3a49bc3fa70a340399125c650a463535e71c"
    role = MODULE.FIXED_ADAPTER_PROFILES["conservative_role_aid"]
    assert role["required_fallback"] == "UNCERTAIN"
    assert "UNCERTAIN" in role["allowed_labels"]
    assert role["required_model_revision"] == "childlens-role-rules-v1.3.0"
    assert "ECAPA" not in role["model_family"]
    assert MODULE.FIXED_ADAPTER_PROFILES["whisper_cpp_large_v3_turbo"]["resource_class"] == "MPS_HEAVY"


def test_instrument_receipt_is_strict_and_seals_pre_download_phase(tmp_path: Path) -> None:
    _, quarantine = _roots(tmp_path)
    executable = _private_file(quarantine / "adapter", b"adapter", 0o700)
    model = _private_file(quarantine / "model.bin", b"model")
    receipt = _receipt(executable, model)
    digest = MODULE.validate_instrument_receipt(receipt)
    assert len(digest) == 64
    seal = MODULE.seal_instruments([receipt])
    assert seal["phase"] == "RESTRICTED_EXECUTION_NETWORK_DISABLED"
    assert seal["all_downloads_completed_before_restricted_processing"] is True
    MODULE.validate_instrument_seal(seal, [receipt])
    bad = dict(receipt)
    bad["free_form_note"] = "not allowlisted"
    with pytest.raises(MODULE.FirewallError, match="E_INSTRUMENT_RECEIPT_SCHEMA"):
        MODULE.validate_instrument_receipt(bad)
    wrong_contract = json.loads(json.dumps(receipt))
    wrong_contract["software"]["adapter_contract_sha256"] = "0" * 64
    with pytest.raises(MODULE.FirewallError, match="E_ADAPTER_CONTRACT_DIGEST"):
        MODULE.validate_instrument_receipt(wrong_contract)


def test_receipt_rejects_boundary_weakening_and_unfrozen_qwen_revision(tmp_path: Path) -> None:
    _, quarantine = _roots(tmp_path)
    executable = _private_file(quarantine / "adapter", b"adapter", 0o700)
    model = _private_file(quarantine / "model.bin", b"model")
    receipt = _receipt(executable, model)
    receipt["boundary"]["external_api_allowed"] = True  # type: ignore[index]
    with pytest.raises(MODULE.FirewallError, match="E_INSTRUMENT_BOUNDARY"):
        MODULE.validate_instrument_receipt(receipt)
    qwen = _receipt(
        executable,
        model,
        profile="qwen2_vl_2b_instruct",
        instrument_key="qwen2-vl-fixed",
    )
    qwen["model"]["version"] = "moving-main"  # type: ignore[index]
    with pytest.raises(MODULE.FirewallError, match="E_MODEL_REVISION"):
        MODULE.validate_instrument_receipt(qwen)


def test_adapter_flags_are_exact_ordered_and_nonduplicated(tmp_path: Path) -> None:
    _, quarantine = _roots(tmp_path)
    executable = _private_file(quarantine / "adapter", b"adapter", 0o700)
    model = _private_file(quarantine / "model.bin", b"model")
    receipt = _receipt(executable, model)
    invocation = MODULE.AdapterInvocation(
        instrument_key="silero-vad-fixed",
        adapter_profile="silero_vad",
        executable=executable,
        model_artifact=model,
        task="VAD_SEGMENTATION",
        resource_class="CPU",
        extra_flags=("--offline", "--offline"),
    )
    with pytest.raises(MODULE.FirewallError, match="E_ADAPTER_FLAGS"):
        MODULE.validate_adapter_invocation(invocation, receipt, quarantine)


def test_subprocess_environment_is_allowlisted_and_forces_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://synthetic.invalid")
    monkeypatch.setenv("KEEPER_REPO_TOKEN", "synthetic-secret")
    environment = MODULE.scrubbed_subprocess_environment({"OMP_NUM_THREADS": "2"})
    assert "OPENAI_API_KEY" not in environment
    assert "HTTPS_PROXY" not in environment
    assert "KEEPER_REPO_TOKEN" not in environment
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["NO_PROXY"] == "*"
    assert "HOME" not in environment
    with pytest.raises(MODULE.FirewallError, match="E_SUBPROCESS_ENV"):
        MODULE.scrubbed_subprocess_environment({"UNAPPROVED": "value"})


@pytest.mark.parametrize("workers", [2, 3, 4])
def test_resource_budget_accepts_only_measured_cpu_worker_range(workers: int) -> None:
    MODULE.ResourceBudget(workers, 1).validate()


@pytest.mark.parametrize("workers", [0, 1, 5, True])
def test_resource_budget_rejects_unsafe_parallelism(workers: object) -> None:
    with pytest.raises(MODULE.FirewallError, match="E_CPU_WORKER_LIMIT"):
        MODULE.ResourceBudget(workers, 1).validate()  # type: ignore[arg-type]
    with pytest.raises(MODULE.FirewallError, match="E_MPS_PROCESS_LIMIT"):
        MODULE.ResourceBudget(2, 2).validate()


def test_active_network_sentinel_fails_when_socket_operations_are_available() -> None:
    with pytest.raises(MODULE.FirewallError, match="E_NETWORK_SENTINEL"):
        MODULE.verify_network_isolation(
            _IdentityBackend(), timeout_seconds=10, process_runner=subprocess.run
        )


def test_active_network_sentinel_accepts_only_zero_from_isolated_child() -> None:
    calls: list[dict[str, object]] = []

    def isolated(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append({"command": command, **kwargs})
        return SimpleNamespace(returncode=0)

    MODULE.verify_network_isolation(_SyntheticBackend(), process_runner=isolated)
    assert calls
    assert calls[0]["stdout"] is subprocess.DEVNULL
    assert calls[0]["stderr"] is subprocess.DEVNULL
    assert calls[0]["close_fds"] is True


def test_network_backend_rejects_forged_prefix() -> None:
    forged = MODULE.NetworkIsolationBackend(
        "MACOS_SANDBOX_DENY_NETWORK", ("/usr/bin/env",)
    )
    with pytest.raises(MODULE.FirewallError, match="E_NETWORK_ISOLATION_BACKEND"):
        forged.command(("python3", "-c", "pass"))


def test_quarantine_validation_is_explicit_private_and_outside_repository(tmp_path: Path) -> None:
    repository, quarantine = _roots(tmp_path)
    assert MODULE.validate_quarantine_root(quarantine, repository) == quarantine
    with pytest.raises(MODULE.FirewallError, match="E_QUARANTINE_IN_REPOSITORY"):
        MODULE.validate_quarantine_root(repository, repository)
    quarantine.chmod(0o755)
    with pytest.raises(MODULE.FirewallError, match="E_QUARANTINE_NOT_PRIVATE"):
        MODULE.validate_quarantine_root(quarantine, repository)


def test_checkpoint_rejects_duplicate_media_and_resumes_running_work(tmp_path: Path) -> None:
    _, quarantine = _roots(tmp_path)
    media = _private_file(quarantine / "opaque.bin", b"synthetic media")
    item = MODULE.WorkItem(_sha(media), media)
    instrument_digest = "a" * 64
    checkpoint_path = quarantine / "state" / "checkpoint.sqlite3"
    with MODULE.CheckpointStore(checkpoint_path, quarantine) as store:
        with pytest.raises(MODULE.FirewallError, match="E_DUPLICATE_MEDIA"):
            store.register(
                [item, item],
                instrument_sha256=instrument_digest,
                task="VAD_SEGMENTATION",
            )
        store.register(
            [item], instrument_sha256=instrument_digest, task="VAD_SEGMENTATION"
        )
        claimed = store.claim_next(instrument_digest, "VAD_SEGMENTATION")
        assert claimed is not None
        assert store.resume_after_interruption() == 1
        assert store.aggregate_counts()["PENDING"] == 1


def test_synthetic_fd_adapter_runs_offline_checkpoints_and_exports_aggregates_only(
    tmp_path: Path,
) -> None:
    repository, quarantine = _roots(tmp_path)
    executable = _private_file(quarantine / "adapter", b"synthetic adapter", 0o700)
    model = _private_file(quarantine / "model.bin", b"synthetic model")
    media = _private_file(quarantine / "opaque.bin", b"synthetic media")
    receipt = _receipt(executable, model)
    seal = MODULE.seal_instruments([receipt])
    invocation = MODULE.AdapterInvocation(
        instrument_key="silero-vad-fixed",
        adapter_profile="silero_vad",
        executable=executable,
        model_artifact=model,
        task="VAD_SEGMENTATION",
        resource_class="CPU",
        extra_flags=("--offline",),
    )
    runner_process = _SyntheticProcessRunner()
    with MODULE.CheckpointStore(
        quarantine / "state" / "checkpoint.sqlite3", quarantine
    ) as store:
        instrument_digest = MODULE.validate_instrument_receipt(receipt)
        store.register(
            [MODULE.WorkItem(_sha(media), media)],
            instrument_sha256=instrument_digest,
            task="VAD_SEGMENTATION",
        )
        runner = MODULE.RestrictedInferenceRunner(
            quarantine_root=quarantine,
            repository_root=repository,
            checkpoint=store,
            seal=seal,
            receipts=[receipt],
            backend=_SyntheticBackend(),
            resource_budget=MODULE.ResourceBudget(2, 1),
            process_runner=runner_process,
        )
        aggregate = runner.run(invocation)
    assert runner_process.sentinel_calls == 1
    assert runner_process.adapter_calls == 1
    assert aggregate["status"] == "INCOMPLETE"
    assert aggregate["schema_version"] == "childlens-v1.3-pseudo-annotation-receipt-v1"
    assert aggregate["candidate_window_count"] == 912
    assert aggregate["candidate_speech_minutes"] == 135.25
    assert aggregate["progress"]["work_units_complete"] == 1
    assert aggregate["network_disabled_during_restricted_inference"] is True
    assert aggregate["inference_subprocess_network_blocked"] is True
    assert aggregate["no_hosted_or_cloud_content_path"] is True
    assert aggregate["quarantine_only"] is True
    assert aggregate["restricted_data_egress"] is False
    assert aggregate["instrument_features_entered_learner"] is False
    assert aggregate["unaudited_pseudo_labels_primary_evaluation_truth"] is False
    assert aggregate["learner_training_executed"] is False
    serialized = json.dumps(aggregate, sort_keys=True)
    assert str(quarantine) not in serialized
    assert "synthetic media" not in serialized
    assert "synthetic_only" not in serialized
    assert all("OPENAI_API_KEY" not in env for env in runner_process.environments)
    outputs = list((quarantine / "pseudo_annotations_v1_3").rglob("*.json"))
    assert len(outputs) == 1
    assert stat.S_IMODE(outputs[0].stat().st_mode) == 0o600


def test_real_host_isolation_executes_synthetic_fd_adapter_without_network(
    tmp_path: Path,
) -> None:
    try:
        backend = MODULE.NetworkIsolationBackend.detect()
    except MODULE.FirewallError:
        pytest.skip("no supported OS isolation backend")
    repository, quarantine = _roots(tmp_path)
    executable = _private_file(
        quarantine / "adapter",
        b"#!/bin/sh\n"
        b"out=''\n"
        b"while [ \"$#\" -gt 0 ]; do\n"
        b"  if [ \"$1\" = '--output-fd' ]; then shift; out=\"$1\"; fi\n"
        b"  shift\n"
        b"done\n"
        b"[ -n \"$out\" ] || exit 12\n"
        b"eval \"printf '%s' '{\\\"synthetic_only\\\":true}' >&$out\"\n",
        0o700,
    )
    model = _private_file(quarantine / "model.bin", b"synthetic model")
    media = _private_file(quarantine / "opaque.bin", b"synthetic media")
    receipt = _receipt(executable, model)
    invocation = MODULE.AdapterInvocation(
        instrument_key="silero-vad-fixed",
        adapter_profile="silero_vad",
        executable=executable,
        model_artifact=model,
        task="VAD_SEGMENTATION",
        resource_class="CPU",
        extra_flags=("--offline",),
    )
    with MODULE.CheckpointStore(
        quarantine / "state" / "checkpoint.sqlite3", quarantine
    ) as store:
        digest = MODULE.validate_instrument_receipt(receipt)
        store.register(
            [MODULE.WorkItem(_sha(media), media)],
            instrument_sha256=digest,
            task="VAD_SEGMENTATION",
        )
        runner = MODULE.RestrictedInferenceRunner(
            quarantine_root=quarantine,
            repository_root=repository,
            checkpoint=store,
            seal=MODULE.seal_instruments([receipt]),
            receipts=[receipt],
            backend=backend,
            resource_budget=MODULE.ResourceBudget(2, 1),
        )
        aggregate = runner.run(invocation)
    assert aggregate["status"] == "INCOMPLETE"


def test_real_host_isolation_denies_adapter_write_outside_quarantine(tmp_path: Path) -> None:
    try:
        backend = MODULE.NetworkIsolationBackend.detect()
    except MODULE.FirewallError:
        pytest.skip("no supported OS isolation backend")
    repository, quarantine = _roots(tmp_path)
    outside = tmp_path / "outside.txt"
    script = (
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--output-fd' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        f"printf 'forbidden' > '{outside}'\n"
        "eval \"printf '%s' '{\\\"synthetic_only\\\":true}' >&$out\"\n"
    ).encode()
    executable = _private_file(quarantine / "adapter", script, 0o700)
    model = _private_file(quarantine / "model.bin", b"synthetic model")
    media = _private_file(quarantine / "opaque.bin", b"synthetic media")
    receipt = _receipt(executable, model)
    invocation = MODULE.AdapterInvocation(
        instrument_key="silero-vad-fixed",
        adapter_profile="silero_vad",
        executable=executable,
        model_artifact=model,
        task="VAD_SEGMENTATION",
        resource_class="CPU",
        extra_flags=("--offline",),
    )
    with MODULE.CheckpointStore(
        quarantine / "state" / "checkpoint.sqlite3", quarantine
    ) as store:
        digest = MODULE.validate_instrument_receipt(receipt)
        store.register(
            [MODULE.WorkItem(_sha(media), media)],
            instrument_sha256=digest,
            task="VAD_SEGMENTATION",
        )
        MODULE.RestrictedInferenceRunner(
            quarantine_root=quarantine,
            repository_root=repository,
            checkpoint=store,
            seal=MODULE.seal_instruments([receipt]),
            receipts=[receipt],
            backend=backend,
            resource_budget=MODULE.ResourceBudget(2, 1),
        ).run(invocation)
    assert not outside.exists()


def test_cpu_runner_uses_bounded_parallel_workers(tmp_path: Path) -> None:
    repository, quarantine = _roots(tmp_path)
    executable = _private_file(quarantine / "adapter", b"synthetic adapter", 0o700)
    model = _private_file(quarantine / "model.bin", b"synthetic model")
    receipt = _receipt(executable, model)
    invocation = MODULE.AdapterInvocation(
        instrument_key="silero-vad-fixed",
        adapter_profile="silero_vad",
        executable=executable,
        model_artifact=model,
        task="VAD_SEGMENTATION",
        resource_class="CPU",
        extra_flags=("--offline",),
    )
    lock = threading.Lock()
    active = 0
    maximum = 0

    def bounded_runner(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal active, maximum
        if "-c" in command:
            return SimpleNamespace(returncode=0)
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.02)
            os.write(kwargs["pass_fds"][-1], b'{"synthetic_only":true}')  # type: ignore[index]
        finally:
            with lock:
                active -= 1
        return SimpleNamespace(returncode=0)

    media_items = []
    for index in range(4):
        media = _private_file(quarantine / f"opaque-{index}.bin", f"synthetic-{index}".encode())
        media_items.append(MODULE.WorkItem(_sha(media), media))
    with MODULE.CheckpointStore(
        quarantine / "state" / "checkpoint.sqlite3", quarantine
    ) as store:
        digest = MODULE.validate_instrument_receipt(receipt)
        store.register(
            media_items,
            instrument_sha256=digest,
            task="VAD_SEGMENTATION",
        )
        runner = MODULE.RestrictedInferenceRunner(
            quarantine_root=quarantine,
            repository_root=repository,
            checkpoint=store,
            seal=MODULE.seal_instruments([receipt]),
            receipts=[receipt],
            backend=_SyntheticBackend(),
            resource_budget=MODULE.ResourceBudget(3, 1),
            process_runner=bounded_runner,
        )
        result = runner.run(invocation)
    assert result["progress"]["work_units_complete"] == 4
    assert result["status"] == "INCOMPLETE"
    assert 2 <= maximum <= 3


def test_mps_lock_fails_closed_on_second_heavy_process(tmp_path: Path) -> None:
    _, quarantine = _roots(tmp_path)
    lock_path = quarantine / ".locks" / "mps.lock"
    with MODULE._exclusive_mps(lock_path, quarantine):
        with pytest.raises(MODULE.FirewallError, match="E_MPS_PROCESS_LIMIT"):
            with MODULE._exclusive_mps(lock_path, quarantine):
                raise AssertionError("unreachable")


def test_aggregate_receipt_has_exact_fail_closed_security_and_ancestry_fields() -> None:
    receipt = MODULE.build_aggregate_receipt(
        {"PENDING": 0, "RUNNING": 0, "COMPLETE": 60, "FAILED": 0},
        task_counts=_complete_task_counts(),
        adapter_profiles=tuple(MODULE.FIXED_ADAPTER_PROFILES),
        isolation_backend="MACOS_SANDBOX_DENY_NETWORK",
        cpu_workers=4,
    )
    expected_true = {
        "local_offline_only",
        "network_disabled_during_restricted_inference",
        "inference_subprocess_network_blocked",
        "no_hosted_or_cloud_content_path",
        "quarantine_only",
        "fixed_versioned_instruments",
        "all_instrument_licenses_audited",
        "model_hashes_verified",
        "model_downloads_completed_before_restricted_processing",
        "checkpoint_resume_enabled",
        "pseudo_labels_marked_not_ground_truth",
        "author_audit_only_human_labeled_childlens_evidence",
        "simulator_oracle_labels_primary_evaluation_truth",
        "aggregate_safe_receipt_only",
    }
    expected_false = {
        "external_api_used",
        "external_upload",
        "telemetry_enabled",
        "restricted_data_egress",
        "instrument_embeddings_entered_learner",
        "instrument_features_entered_learner",
        "instrument_weights_entered_learner",
        "instrument_tokenizers_entered_learner",
        "instrument_vocabularies_entered_learner",
        "instrument_scores_entered_learner",
        "unaudited_pseudo_labels_primary_evaluation_truth",
        "learner_training_executed",
        "corpus_tokenizer_trained",
        "causal_arm_executed",
        "scientific_acquisition_outcome_executed",
    }
    assert all(receipt[key] is True for key in expected_true)
    assert all(receipt[key] is False for key in expected_false)
    assert receipt["resource_policy"]["cpu_worker_range_enforced"] == [2, 4]
    assert receipt["resource_policy"]["mps_heavy_process_limit"] == 1
    assert SYNTH._pseudo_valid(receipt)


def test_aggregate_builder_rejects_unallowlisted_state_or_backend() -> None:
    with pytest.raises(MODULE.FirewallError, match="E_AGGREGATE_SCHEMA"):
        MODULE.build_aggregate_receipt(
            {"PENDING": 0, "RUNNING": 0, "COMPLETE": 1, "FAILED": 0, "TEXT": 1},
            task_counts=_complete_task_counts(),
            adapter_profiles=tuple(MODULE.FIXED_ADAPTER_PROFILES),
            isolation_backend="MACOS_SANDBOX_DENY_NETWORK",
            cpu_workers=2,
        )
    with pytest.raises(MODULE.FirewallError, match="E_AGGREGATE_SCHEMA"):
        MODULE.build_aggregate_receipt(
            {"PENDING": 0, "RUNNING": 0, "COMPLETE": 60, "FAILED": 0},
            task_counts=_complete_task_counts(),
            adapter_profiles=tuple(MODULE.FIXED_ADAPTER_PROFILES),
            isolation_backend="UNISOLATED",
            cpu_workers=2,
        )


def test_cli_is_non_operational_and_content_free(capsys: pytest.CaptureFixture[str]) -> None:
    assert MODULE.main([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "LIBRARY_READY_NO_CONTENT_ACCESSED"
    assert output["operational_cli_enabled"] is False
    assert MODULE.main(["unexpected"]) == 2
    failure = json.loads(capsys.readouterr().out)
    assert failure["error_code"] == "E_ARGUMENTS"
