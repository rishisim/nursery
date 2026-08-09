from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / "scripts" / "childlens_human_validation_v1_2.py"
LAUNCHER_PATH = REPO_ROOT / "scripts" / "launch_childlens_human_validation_v1_2.py"

WORKFLOW_SPEC = importlib.util.spec_from_file_location(
    "childlens_human_validation_v1_2", WORKFLOW_PATH
)
assert WORKFLOW_SPEC and WORKFLOW_SPEC.loader
workflow = importlib.util.module_from_spec(WORKFLOW_SPEC)
sys.modules[WORKFLOW_SPEC.name] = workflow
WORKFLOW_SPEC.loader.exec_module(workflow)

LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "launch_childlens_human_validation_v1_2", LAUNCHER_PATH
)
assert LAUNCHER_SPEC and LAUNCHER_SPEC.loader
launcher = importlib.util.module_from_spec(LAUNCHER_SPEC)
LAUNCHER_SPEC.loader.exec_module(launcher)


def _private_file(path: Path, payload: str = "synthetic") -> None:
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o600)


def _initialized_runtime(search: Path, repository: Path, hidden_name: str) -> Path:
    hidden = search / hidden_name
    hidden.mkdir(mode=0o700)
    root = hidden / "feasibility"
    root.mkdir(mode=0o700)
    workflow_dir = root / workflow.WORKFLOW_DIR
    workflow_dir.mkdir(mode=0o700)
    _private_file(root / workflow.INDEX_SENTINEL)
    policy = {
        "schema_version": workflow.VERSION,
        "retention_deadline": workflow.RETENTION_DEADLINE,
        "owner_only_verified": True,
        "git_exclusion_verified": True,
        "indexing_exclusion_verified": True,
        "backup_exclusion_verified_or_encrypted_local_only": True,
        "local_only": True,
    }
    _private_file(root / workflow.POLICY_FILE, json.dumps(policy))
    for name in (
        workflow.DATABASE_FILE,
        workflow.SECRET_FILE,
        workflow.CODER_ASSIGNMENTS_FILE,
    ):
        _private_file(workflow_dir / name)
    assert repository.parent == search
    return root


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    search = tmp_path / "research"
    search.mkdir(mode=0o700, parents=True)
    repository = search / "nursery"
    repository.mkdir(mode=0o700)
    return search, repository


def _error(function, *args, **kwargs) -> str:
    with pytest.raises(workflow.WorkflowError) as caught:
        function(*args, **kwargs)
    return caught.value.code


def test_no_argument_discovery_authorizes_one_initialized_private_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    search, repository = _workspace(tmp_path)
    root = _initialized_runtime(search, repository, ".childlens-synthetic")
    monkeypatch.delenv(workflow.ROOT_ENV, raising=False)
    monkeypatch.setattr(workflow, "_DISCOVERED_RUNTIME_ROOT", None)

    discovered = workflow.discover_runtime_root(
        search_root=search, repository_root=repository
    )

    assert discovered == root.resolve()
    assert workflow.validate_quarantine_root(discovered) == root.resolve()
    assert workflow.ROOT_ENV not in os.environ


def test_discovery_fails_closed_for_ambiguous_or_nonprivate_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    search, repository = _workspace(tmp_path)
    _initialized_runtime(search, repository, ".childlens-one")
    _initialized_runtime(search, repository, ".childlens-two")
    monkeypatch.setattr(workflow, "_DISCOVERED_RUNTIME_ROOT", None)
    assert (
        _error(
            workflow.discover_runtime_root,
            search_root=search,
            repository_root=repository,
        )
        == "E_RUNTIME_ROOT_AMBIGUOUS"
    )
    assert workflow._DISCOVERED_RUNTIME_ROOT is None

    second_search, second_repository = _workspace(tmp_path / "second")
    nonprivate = _initialized_runtime(
        second_search, second_repository, ".childlens-nonprivate"
    )
    nonprivate.chmod(0o750)
    assert (
        _error(
            workflow.discover_runtime_root,
            search_root=second_search,
            repository_root=second_repository,
        )
        == "E_RUNTIME_ROOT_NOT_FOUND"
    )


def test_runtime_environment_and_browser_argv_never_contain_root(
    tmp_path: Path,
) -> None:
    search, repository = _workspace(tmp_path)
    root = _initialized_runtime(search, repository, ".childlens-synthetic")
    environment = launcher.restricted_runtime_environment(
        root,
        "synthetic-nonce",
        source={
            "PATH": "/usr/bin:/bin",
            workflow.ROOT_ENV: str(root),
            "SYNTHETIC_ALIAS": f"prefix:{root}:suffix",
        },
    )
    assert workflow.ROOT_ENV not in environment
    assert "SYNTHETIC_ALIAS" not in environment
    assert all(str(root) not in value for value in environment.values())

    command = launcher.confined_browser_command(
        root, Path("/usr/bin/true"), "http://127.0.0.1:8501/?launch_token=synthetic"
    )
    assert all(str(root) not in argument for argument in command)
    cwd = launcher.browser_working_directory(root)
    stores = {
        argument.split("=", 1)[1]
        for argument in command
        if argument.startswith(
            ("--user-data-dir=", "--disk-cache-dir=", "--download-default-directory=")
        )
    }
    assert stores == {"browser_profile", "browser_cache", "browser_downloads"}
    for relative in stores:
        resolved = (cwd / relative).resolve(strict=True)
        assert workflow._is_relative_to(resolved, root.resolve())
        assert stat.S_IMODE(resolved.stat().st_mode) == 0o700


def test_launcher_rejects_every_argument_without_echo(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def must_not_discover():
        nonlocal called
        called = True

    monkeypatch.setattr(workflow, "discover_runtime_root", must_not_discover)
    sentinel = "restricted-root-sentinel"
    assert launcher.main([sentinel]) == 2
    captured = capsys.readouterr()
    assert not called
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert captured.err.strip() == "E_ARGUMENTS"


def test_main_passes_root_only_as_browser_cwd_not_argv_or_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    search, repository = _workspace(tmp_path)
    root = _initialized_runtime(search, repository, ".childlens-synthetic")
    monkeypatch.setattr(workflow, "discover_runtime_root", lambda: root)
    monkeypatch.setattr(launcher, "find_browser", lambda: Path("/usr/bin/true"))
    monkeypatch.setattr(launcher, "wait_for_local_server", lambda _server: None)
    monkeypatch.setenv(workflow.ROOT_ENV, str(root))
    calls: list[tuple[list[str], dict]] = []

    class FakeProcess:
        def __init__(self, returncode: int | None = None):
            self.returncode = returncode

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

    def fake_popen(command, **kwargs):
        calls.append((list(command), kwargs))
        return FakeProcess(None)

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    assert launcher.main([]) == 0
    assert len(calls) == 2
    for command, kwargs in calls:
        assert all(str(root) not in argument for argument in command)
        assert workflow.ROOT_ENV not in kwargs["env"]
        assert all(str(root) not in value for value in kwargs["env"].values())
    assert calls[0][1]["cwd"] == workflow.REPO_ROOT
    assert calls[1][1]["cwd"] == root / workflow.WORKFLOW_DIR


def test_runtime_process_failure_emits_only_fixed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    search, repository = _workspace(tmp_path)
    root = _initialized_runtime(search, repository, ".childlens-synthetic")
    monkeypatch.setattr(workflow, "discover_runtime_root", lambda: root)
    monkeypatch.setattr(launcher, "find_browser", lambda: Path("/usr/bin/true"))

    def fail_without_disclosure(*_args, **_kwargs):
        raise OSError(str(root))

    monkeypatch.setattr(launcher.subprocess, "Popen", fail_without_disclosure)
    assert launcher.main([]) == 2
    captured = capsys.readouterr()
    assert str(root) not in captured.out
    assert str(root) not in captured.err
    assert captured.err.strip() == "E_LOCAL_RUNTIME"


def test_app_discovers_runtime_instead_of_reading_root_environment() -> None:
    source = (REPO_ROOT / "scripts/childlens_human_validation_app_v1_2.py").read_text(
        encoding="utf-8"
    )
    assert "discover_runtime_root" in source
    assert "os.environ.get(workflow.ROOT_ENV" not in source
