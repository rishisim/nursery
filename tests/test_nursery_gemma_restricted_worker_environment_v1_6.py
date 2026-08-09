from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/nursery_gemma_substitution_calibration_v1_6.py"
SPEC = importlib.util.spec_from_file_location("test_gemma_worker_environment_v1_6", PATH)
assert SPEC is not None and SPEC.loader is not None
coordinator = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = coordinator; SPEC.loader.exec_module(coordinator)


def test_amendment_erratum_and_v15_public_receipts_are_exact() -> None:
    amendment, erratum = coordinator._validate_worker_environment_amendment()
    assert amendment["status"] == "FROZEN_PRE_RESTRICTED_RERUN_WORKER_ENVIRONMENT_ONLY"
    assert erratum["status"] == "FROZEN_PRE_RESTRICTED_RERUN_SUCCESS_PATH_ONLY"
    assert coordinator._sha256_file(coordinator.PUBLIC_CANARY) == coordinator.PUBLIC_CANARY_SHA256
    assert coordinator._sha256_file(coordinator.RUNTIME_LOCK) == coordinator.RUNTIME_LOCK_SHA256
    assert coordinator._sha256_file(coordinator.ACTIVATION) == coordinator.ACTIVATION_SHA256
    assert coordinator._sha256_file(coordinator.V15_FAILURE) == coordinator.V15_FAILURE_SHA256


def test_worker_environment_passes_only_two_extras_and_adds_only_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = tmp_path / "work"; work.mkdir(mode=0o700); os.chmod(tmp_path, 0o700); os.chmod(work, 0o700)
    mandatory = dict(coordinator.runner.v14.EXPECTED_ENVIRONMENT)
    observed = []

    def scrubber(extra):
        observed.append(dict(extra)); return {**mandatory, **extra}

    monkeypatch.setattr(coordinator.firewall, "scrubbed_subprocess_environment", scrubber)
    environment = coordinator._worker_environment(work)
    assert observed == [{"OMP_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2"}]
    assert len(environment) == 17 and environment["TMPDIR"] == str(work)
    assert set(environment) == set(mandatory) | set(coordinator.THREAD_EXTRAS) | {"TMPDIR"}
    assert all(environment[key] == "1" for key in coordinator.OFFLINE_FLAGS)


def test_worker_environment_fails_closed_before_return_on_mandatory_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = tmp_path / "work"; work.mkdir(mode=0o700); os.chmod(tmp_path, 0o700); os.chmod(work, 0o700)
    bad = dict(coordinator.runner.v14.EXPECTED_ENVIRONMENT); bad["DO_NOT_TRACK"] = "0"
    monkeypatch.setattr(coordinator.firewall, "scrubbed_subprocess_environment", lambda extra: {**bad, **extra})
    with pytest.raises(coordinator.GemmaSubstitutionError, match="E_SUBPROCESS_ENV"):
        coordinator._worker_environment(work)


def test_original_success_path_is_exclusive_owner_only_and_failure_is_namespaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt_path = tmp_path / "success.json"
    monkeypatch.setattr(coordinator, "PUBLIC_RECEIPT", receipt_path)
    monkeypatch.setattr(coordinator, "_public_gemma_seal", lambda: {"seal": True})
    monkeypatch.setattr(coordinator, "_validate_public_receipt", lambda receipt: None)
    monkeypatch.setattr(coordinator.firewall.NetworkIsolationBackend, "detect", lambda: type("B", (), {"command": lambda self, argv: argv})())
    monkeypatch.setattr(coordinator.firewall, "verify_network_isolation", lambda _backend: None)
    monkeypatch.setattr(coordinator.firewall, "scrubbed_subprocess_environment", lambda: {})

    class Process:
        def __init__(self, _argv, **kwargs):
            result_fd = kwargs["pass_fds"][1]
            os.write(result_fd, json.dumps({"status": "CALIBRATION_PASS"}).encode("utf-8"))
        def wait(self, timeout):
            assert timeout == 8 * 60 * 60; return 0

    monkeypatch.setattr(coordinator.subprocess, "Popen", Process)
    coordinator.public_execute()
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert coordinator.PUBLIC_FAILURE.parent.name == "gemma4_restricted_worker_environment_v1_6"
    with pytest.raises(FileExistsError): coordinator.public_execute()


def test_config_preserves_checkpoint_resume_no_fallback_and_no_canary_rerun() -> None:
    coordinator._configure()
    assert coordinator.legacy._worker_environment is coordinator._worker_environment
    assert coordinator.PUBLIC_RECEIPT == ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_receipt.json"
    source = PATH.read_text(encoding="utf-8")
    assert "checkpoint_resume_preserved" in source
    assert '"automatic_fallback_allowed": False' in source
    assert "runner.public_execute" not in source
