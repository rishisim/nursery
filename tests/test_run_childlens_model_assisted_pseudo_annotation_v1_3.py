from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_childlens_model_assisted_pseudo_annotation_v1_3.py"
ADAPTER_PATH = ROOT / "scripts/childlens_local_pseudo_adapter_v1_3.py"
SMOKE_PATH = ROOT / "scripts/smoke_childlens_local_pseudo_adapters_v1_3.py"
SPEC = importlib.util.spec_from_file_location("childlens_v13_pseudo_runner_test", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    raw = root / "raw"
    raw.mkdir(mode=0o700)
    rows: list[dict[str, object]] = []
    per_window = RUNNER.EXPECTED_SPEECH_MINUTES * 60.0 / RUNNER.EXPECTED_WINDOWS
    for index in range(RUNNER.EXPECTED_ITEMS):
        payload = f"synthetic-media-{index}".encode()
        digest = hashlib.sha256(payload).hexdigest()
        media = raw / f"{digest}.bin"
        media.write_bytes(payload)
        media.chmod(0o600)
        count = 61 if index < 12 else 60
        cursor = 0.0
        windows = []
        for _ in range(count):
            windows.append({"start_seconds": cursor, "end_seconds": cursor + per_window})
            cursor += per_window
        rows.append(
            {
                "expected_media_sha256": digest,
                "media_relpath": f"raw/{digest}.bin",
                "speech_windows": windows,
                "reference_duration_seconds": cursor + 1.0,
            }
        )
    return root, {
        "schema_version": RUNNER.MEASUREMENT_SCHEMA,
        "pilot_selection_sha256": RUNNER.FROZEN_SELECTION_DIGEST,
        "items": rows,
    }


def _complete_counts() -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    task = {
        name: {"PENDING": 0, "RUNNING": 0, "COMPLETE": 15, "FAILED": 0}
        for name in RUNNER.firewall.TASKS
    }
    return {"PENDING": 0, "RUNNING": 0, "COMPLETE": 60, "FAILED": 0}, task


def _with_storage(receipt: dict[str, object]) -> dict[str, object]:
    receipt["storage_policy"] = {
        "namespace_cap_gib": 73,
        "free_space_floor_gib": 50,
        "preflight_projection_enforced": True,
        "during_process_monitoring_enforced": True,
        "post_item_cleanup_recheck_enforced": True,
    }
    return receipt


def test_manifest_accepts_exact_frozen_aggregate_only(tmp_path: Path) -> None:
    root, document = _manifest(tmp_path)
    items = RUNNER.validate_measurement_manifest(document, root)
    assert len(items) == 15
    assert sum(len(item.windows) for item in items) == 912


def test_manifest_rejects_model_dependent_selection_field(tmp_path: Path) -> None:
    root, document = _manifest(tmp_path)
    document["items"][0]["model_score"] = 0.99  # type: ignore[index]
    with pytest.raises(RUNNER.RunnerError, match="E_MODEL_DEPENDENT_SELECTION"):
        RUNNER.validate_measurement_manifest(document, root)


def test_manifest_rejects_duplicate_media_even_when_rows_differ(tmp_path: Path) -> None:
    root, document = _manifest(tmp_path)
    first = document["items"][0]  # type: ignore[index]
    document["items"][1]["expected_media_sha256"] = first["expected_media_sha256"]  # type: ignore[index]
    with pytest.raises(RUNNER.RunnerError, match="E_FROZEN_MANIFEST"):
        RUNNER.validate_measurement_manifest(document, root)


def test_aggregate_receipt_has_no_item_cells_or_content() -> None:
    counts, task_counts = _complete_counts()
    receipt = _with_storage(
        dict(
            RUNNER.firewall.build_aggregate_receipt(
                counts,
                task_counts=task_counts,
                adapter_profiles=tuple(RUNNER.firewall.FIXED_ADAPTER_PROFILES),
                isolation_backend="MACOS_SANDBOX_DENY_NETWORK",
                cpu_workers=2,
            )
        )
    )
    validated = RUNNER._validate_public_receipt(receipt)
    assert validated["status"] == "COMPLETE"
    assert validated["progress"]["item_or_task_cells_exported"] is False
    encoded = json.dumps(validated)
    assert "/Users/" not in encoded
    assert ".mp4" not in encoded


def test_public_receipt_rejects_restricted_path_or_media_name() -> None:
    counts, task_counts = _complete_counts()
    receipt = _with_storage(
        dict(
            RUNNER.firewall.build_aggregate_receipt(
                counts,
                task_counts=task_counts,
                adapter_profiles=tuple(RUNNER.firewall.FIXED_ADAPTER_PROFILES),
                isolation_backend="MACOS_SANDBOX_DENY_NETWORK",
                cpu_workers=2,
            )
        )
    )
    receipt["unexpected"] = "/Users/synthetic/restricted.mp4"
    with pytest.raises(RUNNER.RunnerError, match="E_PUBLIC_RECEIPT_PRIVACY"):
        RUNNER._validate_public_receipt(receipt)


def test_adapter_is_direct_onnx_and_keeps_prediction_independent_frames() -> None:
    source = ADAPTER_PATH.read_text(encoding="utf-8")
    assert "import onnxruntime as ort" in source
    assert "from silero_vad" not in source
    assert "FRAME_OFFSETS_SECONDS = (-5.0, -2.5, 0.0, 2.5, 5.0)" in source
    assert "FIXED_MAX_FRAME_EDGE_PIXELS = 448" in source
    assert "Image.Resampling.LANCZOS" in source
    assert '"confidence_adaptive_resampling": False' in source
    assert '"pseudo_labels_are_ground_truth": False' in source
    assert '"primary_evaluation_truth_allowed": False' in source


def test_role_aid_is_explicit_zero_decision_coverage_fallback() -> None:
    source = ADAPTER_PATH.read_text(encoding="utf-8")
    assert '"role": "UNCERTAIN"' in source
    assert "CONSERVATIVE_NO_SEMANTIC_SPEAKER_ANCHOR" in source
    assert "NON_CHILD" in source and "OVERLAP" in source


def test_zero_argument_public_entry_reexecutes_restricted_phase_under_os_sandbox() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "NetworkIsolationBackend.detect()" in source
    assert "backend.command(" in source
    assert "verify_network_isolation" in source
    assert "stdout=subprocess.DEVNULL" in source
    assert "stderr=subprocess.DEVNULL" in source
    assert "--restricted-phase" in source
    assert "--seal-fd" in source and "--result-fd" in source


def test_storage_guard_enforces_namespace_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(RUNNER, "_namespace_bytes", lambda _root: RUNNER.NAMESPACE_CAP_BYTES + 1)
    monkeypatch.setattr(
        RUNNER.shutil,
        "disk_usage",
        lambda _root: SimpleNamespace(free=RUNNER.FREE_SPACE_FLOOR_BYTES + 10),
    )
    with pytest.raises(RUNNER.RunnerError, match="E_NAMESPACE_CAP"):
        RUNNER.StorageGuard(tmp_path).assert_limits()


def test_storage_guard_enforces_projected_free_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(RUNNER, "_namespace_bytes", lambda _root: 1)
    monkeypatch.setattr(
        RUNNER.shutil,
        "disk_usage",
        lambda _root: SimpleNamespace(free=RUNNER.FREE_SPACE_FLOOR_BYTES + 100),
    )
    with pytest.raises(RUNNER.RunnerError, match="E_FREE_SPACE_FLOOR"):
        RUNNER.StorageGuard(tmp_path).assert_limits(projected_growth=101)


def test_fixed_diagnostic_becomes_strict_aggregate_incomplete() -> None:
    diagnostic = RUNNER._diagnostic_envelope(RUNNER.RunnerError("E_RUNTIME_ROOT"))
    receipt = RUNNER._incomplete_receipt_from_diagnostic(
        diagnostic, isolation_backend="MACOS_SANDBOX_DENY_NETWORK"
    )
    validated = RUNNER._validate_public_receipt(receipt)
    assert validated["status"] == "INCOMPLETE"
    assert validated["diagnostic"] == {
        "code": "E_RUNTIME_ROOT",
        "stage": "RUNTIME_DISCOVERY",
        "progress_basis": "CONSERVATIVE_ZERO_COMPLETE_DIAGNOSTIC_FALLBACK",
        "restricted_payload_exported": False,
        "paths_or_item_rows_exported": False,
    }
    assert validated["progress"]["work_units_complete"] == 0
    assert validated["progress"]["work_units_pending"] == 60


def test_unknown_exception_is_redacted_to_allowlisted_fail_closed() -> None:
    diagnostic = RUNNER._diagnostic_envelope(RuntimeError("/Users/private/file.mp4"))
    assert diagnostic["code"] == "E_FAIL_CLOSED"
    assert diagnostic["stage"] == "FAIL_CLOSED"
    assert "/Users" not in json.dumps(diagnostic)


def test_result_fd_regular_success_envelope_roundtrips_over_real_pipe() -> None:
    read_fd, write_fd = os.pipe()
    try:
        value = {"schema_version": "synthetic-success-v1", "status": "COMPLETE"}
        RUNNER._write_fd_json(write_fd, value)
        os.close(write_fd)
        write_fd = -1
        payload = os.read(read_fd, RUNNER.MAX_RESULT_BYTES)
        assert json.loads(payload) == value
        assert payload == RUNNER._canonical(value)
    finally:
        for descriptor in (read_fd, write_fd):
            if descriptor >= 0:
                os.close(descriptor)


def test_restricted_main_returns_diagnostic_on_result_fd(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seal_read, seal_write = os.pipe()
    result_read, result_write = os.pipe()
    try:
        os.write(seal_write, b"{}")
        os.close(seal_write)
        seal_write = -1
        monkeypatch.setattr(
            RUNNER,
            "restricted_execute",
            lambda _envelope: (_ for _ in ()).throw(RUNNER.RunnerError("E_FROZEN_MANIFEST")),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(RUNNER_PATH),
                "--restricted-phase",
                "--seal-fd",
                str(seal_read),
                "--result-fd",
                str(result_write),
            ],
        )
        assert RUNNER.main() == 1
        os.close(result_write)
        result_write = -1
        payload = os.read(result_read, RUNNER.MAX_RESULT_BYTES)
        diagnostic = RUNNER._validate_diagnostic(json.loads(payload))
        assert diagnostic["code"] == "E_FROZEN_MANIFEST"
        assert diagnostic["stage"] == "FROZEN_MANIFEST"
        assert capsys.readouterr().out == ""
    finally:
        for descriptor in (seal_read, seal_write, result_read, result_write):
            if descriptor >= 0:
                os.close(descriptor)


@pytest.mark.skipif(
    os.environ.get("CHILDLENS_RUN_V13_PUBLIC_INSTRUMENT_SMOKE") != "1",
    reason="opt-in public model smoke; no ChildLens input",
)
def test_actual_audio_and_vlm_adapters_under_network_sandbox() -> None:
    completed = subprocess.run(
        [sys.executable, str(SMOKE_PATH)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        close_fds=True,
        timeout=600,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "CHILDLENS_V13_PUBLIC_SYNTHETIC_ADAPTER_SMOKE_PASS"
