from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = REPO_ROOT / "scripts" / "childlens_native_transfer_v1_2.py"
LAUNCHER_PATH = REPO_ROOT / "scripts" / "launch_childlens_native_transfer_v1_2.py"

CONTROLLER_SPEC = importlib.util.spec_from_file_location(
    "childlens_native_transfer_v1_2", CONTROLLER_PATH
)
assert CONTROLLER_SPEC and CONTROLLER_SPEC.loader
controller = importlib.util.module_from_spec(CONTROLLER_SPEC)
sys.modules[CONTROLLER_SPEC.name] = controller
CONTROLLER_SPEC.loader.exec_module(controller)

LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "launch_childlens_native_transfer_v1_2", LAUNCHER_PATH
)
assert LAUNCHER_SPEC and LAUNCHER_SPEC.loader
launcher = importlib.util.module_from_spec(LAUNCHER_SPEC)
sys.modules[LAUNCHER_SPEC.name] = launcher
LAUNCHER_SPEC.loader.exec_module(launcher)


def _key(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _plan() -> dict:
    return {
        "schema_version": controller.PLAN_SCHEMA,
        "canonical_restricted_manifest_sha256": _key("manifest"),
        "pilot_selection_sha256": _key("selection"),
        "video_size_status": "EXACT_BYTES_UNRESOLVED",
        "selected": [
            {
                "selection_rank": index + 1,
                "media_key": _key(f"media-{index}"),
                "object_key": _key(f"object-{index}"),
                "selection_hash": _key(f"selection-{index}"),
                "source_locator": f"/ChildLens/videos/synthetic-{index:02d}.bin",
                "expected_size_bytes": None,
                "local_sha256": None,
            }
            for index in range(15)
        ],
    }


def _config(plan: dict) -> dict:
    return {
        "schema_version": controller.CONFIG_SCHEMA,
        "api_base_url": "https://keeper.mpdl.mpg.de",
        "repository_id": "11111111-2222-3333-4444-555555555555",
        "immutable_commit_id": None,
        "allowed_download_origins": ["https://keeper.mpdl.mpg.de"],
        "authentication": {
            "environment_variable": launcher.TOKEN_ENVIRONMENT_VARIABLE,
            "scheme": "Bearer",
            "send_authorization_to_download": True,
        },
        "request_timeout_seconds": 120,
        "metadata_strategy": "FILE_DETAIL",
        "expected_download_plan_sha256": controller._digest(plan),
        "quarantine_attestations": {
            "owner_only_access_verified": True,
            "outside_git_repository_verified": True,
            "spotlight_excluded_verified": True,
            "backup_excluded_or_encrypted_local_only_verified": True,
            "retention_deadline": "2027-07-31",
            "signed_agreement_controls_verified": True,
        },
        "admission": {
            "mode": "CONSERVATIVE_ROUNDED",
            "amendment_frozen_before_media_open": True,
            "predeclared_nonraw_reserve_bytes": 0,
            "conservative_bounds": [
                {
                    "object_key": row["object_key"],
                    "rounded_display_size_bytes": 997_000_000,
                    "rounding_quantum_bytes": 1_000_000,
                    "transfer_overhead_bytes": 12_416,
                }
                for row in plan["selected"]
            ],
        },
    }


def _private_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def _workspace(tmp_path: Path, *, hidden_name: str = ".childlens-synthetic"):
    search = tmp_path / "research"
    search.mkdir(mode=0o700)
    repository = search / "nursery"
    repository.mkdir(mode=0o700)
    hidden = search / hidden_name
    hidden.mkdir(mode=0o700)
    bundle_dir = hidden / "bundle"
    bundle_dir.mkdir(mode=0o700)
    plan = _plan()
    config = _config(plan)
    plan_path = bundle_dir / "restricted_download_plan_v1_1.json"
    config_path = bundle_dir / "native_transfer_config_v1_2.json"
    _private_json(plan_path, plan)
    _private_json(config_path, config)
    return search, repository, hidden, bundle_dir, plan, config


def _error(callable_, *args, **kwargs) -> str:
    with pytest.raises(launcher.LaunchError) as caught:
        callable_(*args, **kwargs)
    return caught.value.code


class FakeKeychain:
    def __init__(self, token: str = "synthetic-token-value-123456"):
        self.token = token
        self.read_count = 0
        self.delete_count = 0

    def read(self) -> str:
        self.read_count += 1
        return self.token

    def delete(self) -> None:
        self.delete_count += 1


def test_discovers_exactly_one_digest_matched_private_bundle(tmp_path: Path) -> None:
    search, repository, hidden, bundle_dir, plan, _ = _workspace(tmp_path)
    bundle = launcher.discover_bundle(search_root=search, repository_root=repository)
    assert bundle.quarantine_root == bundle_dir
    assert bundle.plan_path == bundle_dir / "restricted_download_plan_v1_1.json"
    assert bundle.config_path == bundle_dir / "native_transfer_config_v1_2.json"
    assert bundle.plan["canonical_restricted_manifest_sha256"] == plan[
        "canonical_restricted_manifest_sha256"
    ]
    assert bundle.aggregate_output_path == (
        repository / "output/childlens_feasibility_v1_2/native_transfer_aggregate.json"
    )


def test_exact_coordinator_shape_config_json_uses_deep_common_private_root(
    tmp_path: Path,
) -> None:
    search = tmp_path / "research"
    search.mkdir(mode=0o700)
    repository = search / "nursery"
    repository.mkdir(mode=0o700)
    hidden = search / ".childlens-synthetic"
    hidden.mkdir(mode=0o700)
    feasibility_root = hidden / "feasibility_v1_1"
    feasibility_root.mkdir(mode=0o700)
    plans = feasibility_root / "restricted_manifest"
    plans.mkdir(mode=0o700)
    transfer_dir = feasibility_root / "transfer_v1_2"
    transfer_dir.mkdir(mode=0o700)
    plan = _plan()
    _private_json(plans / "restricted_download_plan_v1_1.json", plan)
    _private_json(transfer_dir / "config.json", _config(plan))

    bundle = launcher.discover_bundle(search_root=search, repository_root=repository)
    assert bundle.config_path == transfer_dir / "config.json"
    assert bundle.quarantine_root == feasibility_root
    assert bundle.restricted_receipt_path == (
        transfer_dir / "native_transfer_receipt_v1_2.json"
    )


def test_duplicate_matching_bundles_fail_ambiguous(tmp_path: Path) -> None:
    search, repository, _, _, _, _ = _workspace(tmp_path)
    second = search / ".childlens-second"
    second.mkdir(mode=0o700)
    nested = second / "bundle"
    nested.mkdir(mode=0o700)
    plan = _plan()
    _private_json(nested / "restricted_download_plan.json", plan)
    _private_json(nested / "native_transfer_config.json", _config(plan))
    assert (
        _error(launcher.discover_bundle, search_root=search, repository_root=repository)
        == "E_BUNDLE_AMBIGUOUS"
    )


def test_nonmatching_or_nonbearer_config_fails_closed(tmp_path: Path) -> None:
    search, repository, _, bundle_dir, _, config = _workspace(tmp_path)
    config["authentication"]["scheme"] = "Token"
    _private_json(bundle_dir / "native_transfer_config_v1_2.json", config)
    assert (
        _error(launcher.discover_bundle, search_root=search, repository_root=repository)
        == "E_BUNDLE_INVALID"
    )


def test_execute_holds_token_only_in_environment_and_deletes_after_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    search, repository, _, _, _, _ = _workspace(tmp_path)
    keychain = FakeKeychain()
    monkeypatch.delenv(launcher.TOKEN_ENVIRONMENT_VARIABLE, raising=False)
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs["phase"])
        assert os.environ[launcher.TOKEN_ENVIRONMENT_VARIABLE] == keychain.token
        if kwargs["phase"] == "prepare":
            return {"status": "PREPARED"}
        return {
            "status": "COMPLETE",
            "transfer": {"complete": True, "completed_count": 15},
        }

    result = launcher.execute(
        search_root=search,
        repository_root=repository,
        keychain=keychain,
        controller_run=fake_run,
    )
    assert result == {"status": "COMPLETE", "launcher_version": launcher.VERSION}
    assert calls == ["prepare", "acquire"]
    assert keychain.read_count == 1
    assert keychain.delete_count == 1
    assert launcher.TOKEN_ENVIRONMENT_VARIABLE not in os.environ


def test_failed_controller_retains_keychain_item_and_restores_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    search, repository, _, _, _, _ = _workspace(tmp_path)
    keychain = FakeKeychain()
    monkeypatch.setenv(launcher.TOKEN_ENVIRONMENT_VARIABLE, "preexisting-owner-secret")

    def fail_run(**kwargs):
        del kwargs
        raise controller.TransferError("E_SYNTHETIC")

    assert (
        _error(
            launcher.execute,
            search_root=search,
            repository_root=repository,
            keychain=keychain,
            controller_run=fail_run,
        )
        == "E_CONTROLLER"
    )
    assert keychain.delete_count == 0
    assert os.environ[launcher.TOKEN_ENVIRONMENT_VARIABLE] == "preexisting-owner-secret"


def test_wrong_complete_state_never_deletes_keychain_item(tmp_path: Path) -> None:
    search, repository, _, _, _, _ = _workspace(tmp_path)
    keychain = FakeKeychain()

    def incomplete(**kwargs):
        if kwargs["phase"] == "prepare":
            return {"status": "PREPARED"}
        return {
            "status": "IN_PROGRESS",
            "transfer": {"complete": False, "completed_count": None},
        }

    assert (
        _error(
            launcher.execute,
            search_root=search,
            repository_root=repository,
            keychain=keychain,
            controller_run=incomplete,
        )
        == "E_ACQUIRE_STATE"
    )
    assert keychain.delete_count == 0


def test_valid_existing_receipt_resumes_without_overwriting_prepare(tmp_path: Path) -> None:
    search, repository, _, _, _, _ = _workspace(tmp_path)
    bundle = launcher.discover_bundle(search_root=search, repository_root=repository)
    receipt = controller._new_receipt(bundle.plan, bundle.config, None)
    controller._atomic_json(bundle.restricted_receipt_path, receipt, 0o600)
    keychain = FakeKeychain()
    calls = []

    def acquire_only(**kwargs):
        calls.append(kwargs["phase"])
        return {
            "status": "COMPLETE",
            "transfer": {"complete": True, "completed_count": 15},
        }

    result = launcher.execute(
        search_root=search,
        repository_root=repository,
        keychain=keychain,
        controller_run=acquire_only,
    )
    assert result["status"] == "COMPLETE"
    assert calls == ["acquire"]
    assert keychain.delete_count == 1


def test_invalid_existing_receipt_fails_before_keychain_read(tmp_path: Path) -> None:
    search, repository, _, _, _, _ = _workspace(tmp_path)
    bundle = launcher.discover_bundle(search_root=search, repository_root=repository)
    bundle.restricted_receipt_path.write_text("{}", encoding="utf-8")
    bundle.restricted_receipt_path.chmod(0o600)
    keychain = FakeKeychain()
    assert (
        _error(
            launcher.execute,
            search_root=search,
            repository_root=repository,
            keychain=keychain,
        )
        == "E_EXISTING_RECEIPT"
    )
    assert keychain.read_count == 0
    assert keychain.delete_count == 0


def test_macos_keychain_commands_use_fixed_service_account_and_capture_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = tmp_path / "security"
    tool.write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(launcher, "KEYCHAIN_TOOL", tool)
    calls = []

    def fake_subprocess(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "find-generic-password":
            return subprocess.CompletedProcess(command, 0, b"synthetic-keychain-token-123\n", b"")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(launcher.subprocess, "run", fake_subprocess)
    keychain = launcher.MacOSKeychain()
    assert keychain.read() == "synthetic-keychain-token-123"
    keychain.delete()
    assert calls[0][0] == [
        str(tool),
        "find-generic-password",
        "-s",
        launcher.KEYCHAIN_SERVICE,
        "-a",
        launcher.KEYCHAIN_ACCOUNT,
        "-w",
    ]
    assert calls[0][1]["stdout"] == subprocess.PIPE
    assert calls[0][1]["stderr"] == subprocess.DEVNULL
    assert calls[1][0] == [
        str(tool),
        "delete-generic-password",
        "-s",
        launcher.KEYCHAIN_SERVICE,
        "-a",
        launcher.KEYCHAIN_ACCOUNT,
    ]


def test_keychain_delete_failure_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = tmp_path / "security"
    tool.write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(launcher, "KEYCHAIN_TOOL", tool)

    def fake_subprocess(command, **kwargs):
        del kwargs
        if command[1] == "find-generic-password":
            return subprocess.CompletedProcess(command, 0, b"synthetic-keychain-token-123\n", b"")
        return subprocess.CompletedProcess(command, 44, b"", b"")

    monkeypatch.setattr(launcher.subprocess, "run", fake_subprocess)
    keychain = launcher.MacOSKeychain()
    assert keychain.read() == "synthetic-keychain-token-123"
    assert _error(keychain.delete) == "E_KEYCHAIN_DELETE"


def test_launcher_rejects_arguments_without_echo(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def must_not_execute():
        nonlocal called
        called = True

    monkeypatch.setattr(launcher, "execute", must_not_execute)
    sentinel = "restricted-path-or-token-sentinel"
    assert launcher.main([sentinel]) == 2
    captured = capsys.readouterr()
    assert called is False
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert json.loads(captured.out) == {"status": "error", "error_code": "E_ARGUMENTS"}
