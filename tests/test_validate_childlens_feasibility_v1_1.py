from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_childlens_feasibility_v1_1.py"
SPEC = importlib.util.spec_from_file_location("validate_childlens_feasibility_v1_1", SCRIPT)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def _copy_tree(tmp_path: Path) -> Path:
    for relative in (
        "docs/childlens_feasibility_v1",
        "output/childlens_feasibility_v1",
        "docs/childlens_feasibility_v1_1",
        "output/childlens_feasibility_v1_1",
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
    for relative in (
        "scripts/validate_childlens_feasibility_v1.py",
        "tests/test_childlens_feasibility_v1.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, target)
    return tmp_path


def _codes(root: Path) -> set[str]:
    return {issue.code for issue in validator.validate(root)}


def test_current_artifact_set_passes() -> None:
    assert validator.validate(REPO_ROOT) == []


def test_preselection_cannot_be_upgraded_to_acquisition(tmp_path: Path) -> None:
    root = _copy_tree(tmp_path)
    path = root / "output/childlens_feasibility_v1_1/pilot_preselection_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["video_acquisition_started"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    assert "PILOT_STATE_MISMATCH" in _codes(root)


def test_restricted_machine_field_is_rejected(tmp_path: Path) -> None:
    root = _copy_tree(tmp_path)
    path = root / "output/childlens_feasibility_v1_1/extra.json"
    path.write_text(json.dumps({"participant_id": "synthetic-secret"}), encoding="utf-8")
    assert "RESTRICTED_MACHINE_FIELD" in _codes(root)


def test_quarantine_media_count_is_rejected(tmp_path: Path) -> None:
    root = _copy_tree(tmp_path)
    path = root / "output/childlens_feasibility_v1_1/quarantine_admission_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["video_file_count"] = 1
    path.write_text(json.dumps(value), encoding="utf-8")
    assert "QUARANTINE_SCOPE_MISMATCH" in _codes(root)


def test_historical_v1_mutation_is_rejected(tmp_path: Path) -> None:
    root = _copy_tree(tmp_path)
    path = root / "docs/childlens_feasibility_v1/executive_decision_report.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nsynthetic mutation\n", encoding="utf-8")
    assert "V1_IMMUTABILITY_MISMATCH" in _codes(root)
