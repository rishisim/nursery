from __future__ import annotations

import importlib.util
import json
import os
import plistlib
import sys
from collections import namedtuple
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "preflight_childlens_quarantine_v1_2.py"
SPEC = importlib.util.spec_from_file_location("preflight_childlens_quarantine_v1_2", SCRIPT)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preflight
SPEC.loader.exec_module(preflight)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    research = tmp_path / "research"
    repository = research / "repo"
    quarantine = research / ".restricted" / "quarantine"
    repository.mkdir(parents=True)
    quarantine.mkdir(parents=True)
    os.chmod(quarantine.parent, 0o700)
    os.chmod(quarantine, 0o700)
    sentinel = quarantine / ".metadata_never_index"
    sentinel.write_bytes(b"")
    os.chmod(sentinel, 0o600)
    payload = quarantine / "opaque"
    payload.write_bytes(b"must-not-be-read")
    os.chmod(payload, 0o600)
    return research, repository, quarantine


def _mock_host_pass(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "_filevault_on", lambda: True)
    monkeypatch.setattr(preflight, "_time_machine_excluded", lambda _root: True)
    monkeypatch.setattr(preflight, "_has_acl", lambda _path: (False, False))
    disk_usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda _root: disk_usage(200 * preflight.GIB, 50 * preflight.GIB, 150 * preflight.GIB),
    )


def test_pass_receipt_is_path_and_payload_free(tmp_path: Path, monkeypatch) -> None:
    research, repository, quarantine = _fixture(tmp_path)
    _mock_host_pass(monkeypatch)
    receipt = preflight.build_receipt(repository, research, date(2026, 7, 21))
    rendered = str(receipt)

    assert receipt["status"] == "PASS_QUARANTINE_AND_RESOURCE_SNAPSHOT"
    assert receipt["quarantine_controls_pass"] is True
    assert receipt["resource_snapshot_pass"] is True
    assert str(quarantine) not in rendered
    assert "opaque" not in rendered
    assert "must-not-be-read" not in rendered
    assert receipt["privacy"]["restricted_file_content_opened"] is False


def test_symlink_fails_closed(tmp_path: Path, monkeypatch) -> None:
    research, repository, quarantine = _fixture(tmp_path)
    _mock_host_pass(monkeypatch)
    (quarantine / "link").symlink_to(quarantine / "opaque")
    receipt = preflight.build_receipt(repository, research, date(2026, 7, 21))

    assert receipt["status"] == "FAIL_CLOSED_QUARANTINE_OR_RESOURCE_CONTROL"
    assert receipt["quarantine"]["symlink_count"] == 1
    assert receipt["quarantine_controls_pass"] is False


def test_nonunique_candidate_fails_without_paths(tmp_path: Path, monkeypatch) -> None:
    research, repository, _ = _fixture(tmp_path)
    second = research / ".another" / "quarantine"
    second.mkdir(parents=True)
    (second / ".metadata_never_index").write_bytes(b"")
    receipt = preflight.build_receipt(repository, research, date(2026, 7, 21))

    assert receipt["status"] == "FAIL_CLOSED_QUARANTINE_CANDIDATE_NOT_UNIQUE"
    assert receipt["candidate_count"] == 2
    assert "quarantine" not in {str(value) for value in receipt.values()}


def test_time_machine_xml_receipt_is_parsed(monkeypatch, tmp_path: Path) -> None:
    payload = plistlib.dumps([{"Path": str(tmp_path), "IsExcluded": 1}]).decode("utf-8")
    monkeypatch.setattr(preflight, "_captured", lambda _argv: (0, payload))
    assert preflight._time_machine_excluded(tmp_path) is True


def test_owner_only_mode_repair_uses_metadata_only(tmp_path: Path, monkeypatch) -> None:
    _, _, quarantine = _fixture(tmp_path)
    _mock_host_pass(monkeypatch)
    payload = quarantine / "opaque"
    os.chmod(payload, 0o644)
    repaired_directories, repaired_files = preflight.repair_owner_only_modes(quarantine)
    assert repaired_directories == 0
    assert repaired_files == 1
    assert payload.read_bytes() == b"must-not-be-read"
    assert (payload.stat().st_mode & 0o777) == 0o600


def test_current_time_scoped_admission_receipt_passes() -> None:
    path = REPO_ROOT / "output/childlens_feasibility_v1_2/quarantine_resource_readmission_receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    quarantine = receipt["quarantine"]
    resource = receipt["resource_snapshot"]
    controls = receipt["capacity_controls"]

    assert receipt["status"] == "PASS_QUARANTINE_AND_RESOURCE_SNAPSHOT"
    assert receipt["quarantine_controls_pass"] is True
    assert receipt["resource_snapshot_pass"] is True
    assert quarantine["root_mode"] == "0700"
    assert quarantine["file_mode_violation_count"] == 0
    assert quarantine["directory_mode_violation_count"] == 0
    assert quarantine["symlink_count"] == 0
    assert quarantine["time_machine_excluded"] is True
    assert quarantine["spotlight_no_index_sentinel_present"] is True
    assert resource["current_namespace_allocated_bytes"] <= controls["namespace_peak_cap_bytes"]
    assert resource["shared_volume_free_bytes"] >= controls["minimum_free_after_peak_bytes"]
    assert resource["maximum_additional_bytes_now"] >= controls["raw_cap_bytes"]
    assert receipt["admission_scope"]["media_transfer_authorized_by_this_receipt"] is False
    assert all(value is False for value in receipt["privacy"].values())
