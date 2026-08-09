from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / "scripts/childlens_author_audit_v1_3.py"
LAUNCHER_PATH = REPO_ROOT / "scripts/launch_childlens_author_audit_v1_3.py"
V12_PATH = REPO_ROOT / "scripts/childlens_human_validation_v1_2.py"
INITIALIZER_PATH = REPO_ROOT / "scripts/initialize_childlens_author_audit_v1_3.py"
EXPORTER_PATH = REPO_ROOT / "scripts/export_childlens_author_workflow_receipt_v1_3.py"

WORKFLOW_SPEC = importlib.util.spec_from_file_location("childlens_author_audit_v1_3", WORKFLOW_PATH)
assert WORKFLOW_SPEC and WORKFLOW_SPEC.loader
workflow = importlib.util.module_from_spec(WORKFLOW_SPEC)
sys.modules[WORKFLOW_SPEC.name] = workflow
WORKFLOW_SPEC.loader.exec_module(workflow)

V12_SPEC = importlib.util.spec_from_file_location("childlens_human_validation_v1_2", V12_PATH)
assert V12_SPEC and V12_SPEC.loader
v12 = importlib.util.module_from_spec(V12_SPEC)
sys.modules[V12_SPEC.name] = v12
V12_SPEC.loader.exec_module(v12)

LAUNCHER_SPEC = importlib.util.spec_from_file_location("launch_childlens_author_audit_v1_3", LAUNCHER_PATH)
assert LAUNCHER_SPEC and LAUNCHER_SPEC.loader
launcher = importlib.util.module_from_spec(LAUNCHER_SPEC)
LAUNCHER_SPEC.loader.exec_module(launcher)

INITIALIZER_SPEC = importlib.util.spec_from_file_location(
    "initialize_childlens_author_audit_v1_3", INITIALIZER_PATH
)
assert INITIALIZER_SPEC and INITIALIZER_SPEC.loader
initializer = importlib.util.module_from_spec(INITIALIZER_SPEC)
INITIALIZER_SPEC.loader.exec_module(initializer)

EXPORTER_SPEC = importlib.util.spec_from_file_location(
    "export_childlens_author_workflow_receipt_v1_3", EXPORTER_PATH
)
assert EXPORTER_SPEC and EXPORTER_SPEC.loader
exporter = importlib.util.module_from_spec(EXPORTER_SPEC)
EXPORTER_SPEC.loader.exec_module(exporter)


def private_file(path: Path, payload: str = "synthetic") -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o600)


def initialized_runtime(search: Path, repository: Path, hidden_name: str) -> Path:
    hidden = search / hidden_name
    hidden.mkdir(mode=0o700)
    root = hidden / "feasibility"
    root.mkdir(mode=0o700)
    workflow_dir = root / workflow.WORKFLOW_DIR
    workflow_dir.mkdir(mode=0o700)
    private_file(root / workflow.INDEX_SENTINEL)
    policy = {
        "schema_version": workflow.VERSION,
        "retention_deadline": workflow.RETENTION_DEADLINE,
        "owner_only_verified": True,
        "git_exclusion_verified": True,
        "indexing_exclusion_verified": True,
        "backup_exclusion_verified_or_encrypted_local_only": True,
        "local_only": True,
        "signed_agreement_controls_inherited": True,
    }
    private_file(root / workflow.POLICY_FILE, json.dumps(policy))
    for name in (
        workflow.DATABASE_FILE,
        workflow.SECRET_FILE,
        workflow.AUTHOR_ASSIGNMENT_FILE,
        workflow.SAMPLE_SNAPSHOT_FILE,
    ):
        private_file(workflow_dir / name)
    assert repository.parent == search
    return root


def workspace(tmp_path: Path) -> tuple[Path, Path]:
    search = tmp_path / "research"
    search.mkdir(mode=0o700, parents=True)
    repository = search / "nursery"
    repository.mkdir(mode=0o700)
    return search, repository


def error(function, *args, **kwargs) -> str:
    with pytest.raises(workflow.WorkflowError) as caught:
        function(*args, **kwargs)
    return caught.value.code


def test_discovery_authorizes_exactly_one_private_initialized_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    search, repository = workspace(tmp_path)
    root = initialized_runtime(search, repository, ".childlens-synthetic")
    monkeypatch.delenv(workflow.ROOT_ENV, raising=False)
    monkeypatch.setattr(workflow, "_DISCOVERED_RUNTIME_ROOT", None)
    discovered = workflow.discover_runtime_root(search_root=search, repository_root=repository)
    assert discovered == root.resolve()
    assert workflow.validate_quarantine_root(discovered) == root.resolve()


def test_discovery_fails_closed_for_ambiguous_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    search, repository = workspace(tmp_path)
    initialized_runtime(search, repository, ".childlens-one")
    initialized_runtime(search, repository, ".childlens-two")
    monkeypatch.setattr(workflow, "_DISCOVERED_RUNTIME_ROOT", None)
    assert error(workflow.discover_runtime_root, search_root=search, repository_root=repository) == "E_RUNTIME_ROOT_AMBIGUOUS"
    assert workflow._DISCOVERED_RUNTIME_ROOT is None


def test_browser_and_environment_are_loopback_only_and_quarantine_confined(tmp_path: Path) -> None:
    search, repository = workspace(tmp_path)
    root = initialized_runtime(search, repository, ".childlens-synthetic")
    environment = launcher.restricted_runtime_environment(
        root,
        "synthetic-nonce",
        source={
            "PATH": "/usr/bin:/bin",
            workflow.ROOT_ENV: str(root),
            "HTTPS_PROXY": "https://proxy.invalid",
            "ROOT_ALIAS": f"x:{root}:y",
        },
    )
    assert workflow.ROOT_ENV not in environment
    assert "HTTPS_PROXY" not in environment
    assert "ROOT_ALIAS" not in environment
    assert environment["NO_PROXY"] == "127.0.0.1,localhost"
    assert all(str(root) not in value for value in environment.values())
    command = launcher.confined_browser_command(
        root, Path("/usr/bin/true"), "http://127.0.0.1:8502/?launch_token=synthetic"
    )
    assert "--disable-background-networking" in command
    assert "--host-resolver-rules=MAP * 0.0.0.0, EXCLUDE 127.0.0.1" in command
    assert all(str(root) not in argument for argument in command)
    cwd = launcher.browser_working_directory(root)
    for relative in ("browser_profile", "browser_cache", "browser_downloads"):
        resolved = (cwd / relative).resolve(strict=True)
        assert workflow._is_relative_to(resolved, root.resolve())
        assert stat.S_IMODE(resolved.stat().st_mode) == 0o700
    streamlit = launcher.streamlit_command()
    assert "--server.address=127.0.0.1" in streamlit
    assert "--browser.gatherUsageStats=false" in streamlit


def test_launcher_rejects_arguments_without_echo(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def must_not_discover():
        nonlocal called
        called = True

    monkeypatch.setattr(workflow, "discover_runtime_root", must_not_discover)
    sentinel = "restricted-sentinel"
    assert launcher.main([sentinel]) == 2
    captured = capsys.readouterr()
    assert not called
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert captured.err.strip() == "E_ARGUMENTS"


def test_main_never_places_root_in_argv_or_child_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    search, repository = workspace(tmp_path)
    root = initialized_runtime(search, repository, ".childlens-synthetic")
    monkeypatch.setattr(workflow, "discover_runtime_root", lambda: root)
    monkeypatch.setattr(launcher, "find_browser", lambda: Path("/usr/bin/true"))
    monkeypatch.setattr(launcher, "wait_for_local_server", lambda _server: None)
    calls: list[tuple[list[str], dict]] = []

    class FakeProcess:
        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            self.returncode = 0
            return 0

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

    def fake_popen(command, **kwargs):
        calls.append((list(command), kwargs))
        return FakeProcess()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    assert launcher.main([]) == 0
    assert len(calls) == 2
    for command, kwargs in calls:
        assert all(str(root) not in argument for argument in command)
        assert workflow.ROOT_ENV not in kwargs["env"]
        assert all(str(root) not in value for value in kwargs["env"].values())
    assert calls[0][1]["cwd"] == workflow.REPO_ROOT
    assert calls[1][1]["cwd"] == root / workflow.WORKFLOW_DIR


def test_sources_contain_no_hosted_content_route() -> None:
    sources = "\n".join(
        (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in (
            "scripts/childlens_author_audit_v1_3.py",
            "scripts/childlens_author_audit_app_v1_3.py",
            "scripts/launch_childlens_author_audit_v1_3.py",
        )
    )
    urls = [token for token in sources.split() if token.startswith(("http://", "https://"))]
    assert all("127.0.0.1" in url for url in urls)
    assert "requests" not in sources
    assert "openai" not in sources.lower()
    assert "anthropic" not in sources.lower()
    assert "telemetry=true" not in sources.lower()


def test_initializer_locates_exactly_one_private_sample_by_canonical_digest(tmp_path: Path) -> None:
    root = tmp_path / "restricted"
    root.mkdir(mode=0o700)
    value = {"schema_version": workflow.PACKET_VERSION, "synthetic": True}
    candidate = root / "sealed/sample.json"
    private_file(candidate, json.dumps(value))
    digest = workflow._sha256(workflow._canonical(value))
    assert initializer.locate_primary_sample(root, digest) == candidate.resolve()
    duplicate = root / "sealed/copy.json"
    private_file(duplicate, json.dumps(value))
    assert error(initializer.locate_primary_sample, root, digest) == "E_SAMPLE_DISCOVERY_AMBIGUOUS"


def test_zero_argument_initializer_inherits_v12_controls_without_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "restricted"
    root.mkdir(mode=0o700)
    private_file(root / workflow.INDEX_SENTINEL)
    source_policy = {
        "schema_version": v12.VERSION,
        "retention_deadline": v12.RETENTION_DEADLINE,
        "owner_only_verified": True,
        "git_exclusion_verified": True,
        "indexing_exclusion_verified": True,
        "backup_exclusion_verified_or_encrypted_local_only": True,
        "local_only": True,
    }
    private_file(root / v12.POLICY_FILE, json.dumps(source_policy))
    sample_value = {"schema_version": workflow.PACKET_VERSION, "synthetic": True}
    sample = root / "sealed/primary.json"
    private_file(sample, json.dumps(sample_value))
    digest = workflow._sha256(workflow._canonical(sample_value))

    def no_v13_runtime():
        raise workflow.WorkflowError("E_RUNTIME_ROOT_NOT_FOUND")

    monkeypatch.setattr(workflow, "discover_runtime_root", no_v13_runtime)
    monkeypatch.setattr(v12, "discover_runtime_root", lambda: root)
    monkeypatch.setattr(v12, "_validate_root_controls", lambda value: value)
    monkeypatch.setattr(initializer, "_load_public_sample_digest", lambda: digest)
    calls: list[tuple[Path, Path]] = []

    def fake_initialize(runtime_root, packet):
        calls.append((Path(runtime_root), Path(packet)))
        return {"status": "initialized", "audit_item_count": 15}

    monkeypatch.setattr(workflow, "initialize", fake_initialize)
    result = initializer.initialize_existing_quarantine()
    assert result == {
        "status": "AUTHOR_AUDIT_INITIALIZED",
        "route": "AUTHOR_AUDIT_A",
        "audit_item_count": 15,
        "audit_speech_minutes": 15.0,
        "predictions_mounted": False,
    }
    assert calls == [(root, sample.resolve())]
    inherited = json.loads((root / workflow.POLICY_FILE).read_text())
    assert inherited["signed_agreement_controls_inherited"] is True
    assert inherited["prediction_store_mounted_before_author_lock"] is False
    assert not (root / workflow.WORKFLOW_DIR / workflow.PREDICTION_BINDING_SNAPSHOT_FILE).exists()


def test_initializer_rejects_arguments_without_disclosing_them(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def must_not_initialize():
        nonlocal called
        called = True

    monkeypatch.setattr(initializer, "initialize_existing_quarantine", must_not_initialize)
    sentinel = "restricted-path-sentinel"
    assert initializer.main([sentinel]) == 2
    captured = capsys.readouterr()
    assert not called
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert captured.err.strip() == "E_ARGUMENTS"


def test_zero_argument_exporter_writes_only_validated_aggregate_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synthetic_root = tmp_path / "opaque-runtime"
    receipt = {
        "schema_version": workflow.RECEIPT_VERSION,
        "status": "READY",
        "workflow_ready": True,
        "audit_item_count": 15,
        "audit_speech_minutes": 15.0,
        "audit_speech_seconds": 900,
        "route": "AUTHOR_AUDIT_A",
        "loopback_only": True,
        "owner_private": True,
        "autosave": True,
        "autosave_to_quarantine_only": True,
        "resumable": True,
        "predictions_hidden": True,
        "predictions_hidden_before_author_lock": True,
        "author_record_lock_immutable": True,
        "comparison_requires_lock": True,
        "selected_independent_of_model_outputs": True,
        "sample_independent_of_model_outputs": True,
        "uncertain_unusable_routes": True,
        "uncertain_route_available": True,
        "unusable_route_available": True,
        "qualification_instruction_present": True,
        "external_hosting": False,
        "network_exposure": False,
        "model_predictions_revealed_before_lock": False,
        "human_evidence_fabricated": False,
        "estimated_author_minutes": 75,
        "human_audit_complete": False,
        "prediction_join_enabled": False,
        "inter_human_reliability_available": False,
        "primary_evaluation_truth": "SIMULATOR_ORACLE_ONLY",
        "unaudited_pseudo_label_permitted_uses": [
            "AGGREGATE_CALIBRATION",
            "CANDIDATE_GENERATION",
        ],
        "unaudited_pseudo_label_primary_evaluation_truth": False,
    }
    output = tmp_path / "public/author_workflow_receipt.json"
    monkeypatch.setattr(workflow, "discover_runtime_root", lambda: synthetic_root)
    monkeypatch.setattr(workflow, "aggregate_receipt", lambda root: receipt)
    monkeypatch.setattr(exporter, "OUTPUT_PATH", output)
    result = exporter.export_receipt()
    assert result["status"] == "AUTHOR_WORKFLOW_RECEIPT_EXPORTED"
    assert json.loads(output.read_text()) == receipt
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
    rendered = output.read_text()
    assert str(synthetic_root) not in rendered
    assert "transcript" not in rendered


def test_exporter_rejects_arguments_without_writing(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def must_not_export():
        nonlocal called
        called = True

    monkeypatch.setattr(exporter, "export_receipt", must_not_export)
    sentinel = "restricted-sentinel"
    assert exporter.main([sentinel]) == 2
    captured = capsys.readouterr()
    assert not called
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert captured.err.strip() == "E_ARGUMENTS"
