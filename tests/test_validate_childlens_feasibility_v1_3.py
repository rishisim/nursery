from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/validate_childlens_feasibility_v1_3.py"
SPEC = importlib.util.spec_from_file_location("validate_childlens_feasibility_v1_3", SCRIPT)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def test_all_three_historical_namespaces_match_frozen_baselines() -> None:
    status = validator.historical_status(REPO_ROOT)
    assert set(status) == {"v1", "v1_1", "v1_2"}
    assert all(entry["preserved"] is True for entry in status.values())


def test_v1_2_baseline_excludes_v1_3_files() -> None:
    relative = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in validator.historical_files(REPO_ROOT, "v1_2")
    ]
    assert relative
    assert all("v1_3" not in path and "v1.3" not in path for path in relative)
    assert "output/childlens_feasibility_v1_2/decision_record.json" in relative


def test_any_v1_2_file_mutation_breaks_the_frozen_set_digest(tmp_path: Path) -> None:
    copied: list[Path] = []
    for source in validator.historical_files(REPO_ROOT, "v1_2"):
        relative = source.relative_to(REPO_ROOT)
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
    before = validator.artifact_set_digest(tmp_path, validator.historical_files(tmp_path, "v1_2"))
    assert before == validator.EXPECTED_HISTORY["v1_2"]
    copied[0].write_bytes(copied[0].read_bytes() + b"\n")
    after = validator.artifact_set_digest(tmp_path, validator.historical_files(tmp_path, "v1_2"))
    assert after[0] == before[0]
    assert after[1] != before[1]


def test_privacy_scanner_reports_key_without_echoing_payload(tmp_path: Path) -> None:
    path = tmp_path / "output/childlens_feasibility_v1_3/bad.json"
    path.parent.mkdir(parents=True)
    secret = "NEVER_ECHO_THIS_RESTRICTED_TEXT"
    path.write_text(json.dumps({"transcript_text": secret}), encoding="utf-8")
    issues: list[validator.Issue] = []
    validator._scan_file(path, tmp_path, issues)
    serialized = json.dumps([issue.__dict__ for issue in issues])
    assert any(issue.code == "RESTRICTED_MACHINE_KEY" for issue in issues)
    assert secret not in serialized


def test_hosted_model_import_in_v1_3_script_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "scripts/childlens_bad_v1_3.py"
    path.parent.mkdir(parents=True)
    path.write_text("from openai import OpenAI\n", encoding="utf-8")
    issues: list[validator.Issue] = []
    validator._scan_file(path, tmp_path, issues)
    assert any(issue.code == "HOSTED_CONTENT_CODE_PATH" for issue in issues)


def test_payload_extension_in_public_namespace_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "docs/childlens_feasibility_v1_3/example.wav"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not real media")
    issues: list[validator.Issue] = []
    validator._scan_file(path, tmp_path, issues)
    assert any(issue.code == "RESTRICTED_PAYLOAD_EXTENSION" for issue in issues)


def test_static_scanner_has_no_quarantine_discovery_primitive() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "glob(\"**/*\")" not in source
    assert "rglob(\"*\")" in source  # only explicit public/historical namespaces
    assert "find /" not in source
