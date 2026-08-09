from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_childlens_feasibility_v1.py"
SPEC = importlib.util.spec_from_file_location("validate_childlens_feasibility_v1", SCRIPT)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def _rubric() -> dict:
    return {
        "rubric_id": "childlens-feasibility-rubric-test",
        "statuses": ["PASS", "CONDITIONAL", "FAIL", "UNRESOLVED", "NOT_APPLICABLE"],
        "gates": [{"gate_id": f"G{index}", "essential": True} for index in range(1, 11)],
    }


def _decision(state: str, statuses: list[str] | None = None) -> dict:
    statuses = statuses or ["PASS"] * 10
    rows = []
    for index, status in enumerate(statuses, start=1):
        row = {"gate_id": f"G{index}", "status": status}
        if status != "PASS":
            row["bounded_correction"] = "Complete the named release-specific validation before protocol freeze."
        rows.append(row)
    return {
        "record_id": "safe-test-record",
        "rubric_id": "childlens-feasibility-rubric-test",
        "terminal_decision": state,
        "scientific_outcome_authorized": False,
        "empirical_corpus": "CURRENTLY_ACCESSIBLE_CHILDLENS_KEEPER_RELEASE_ONLY",
        "gate_assessments": rows,
    }


def _complete_tree(tmp_path: Path, state: str = "CHILDLENS_FEASIBILITY_GO", statuses: list[str] | None = None) -> Path:
    docs = tmp_path / "docs" / "childlens_feasibility_v1"
    output = tmp_path / "output" / "childlens_feasibility_v1"
    docs.mkdir(parents=True)
    output.mkdir(parents=True)
    for filename in validator.DOCS_REQUIRED:
        path = docs / filename
        if path.suffix == ".json":
            path.write_text(json.dumps({"artifact": "safe-test"}), encoding="utf-8")
        else:
            path.write_text("# Safe aggregate-only test artifact\n", encoding="utf-8")
    for filename in validator.OUTPUT_REQUIRED:
        (output / filename).write_text(json.dumps({"artifact": "safe-test"}), encoding="utf-8")
    (docs / "frozen_feasibility_rubric_v1.json").write_text(json.dumps(_rubric()), encoding="utf-8")
    (docs / "decision_record_v1.json").write_text(json.dumps(_decision(state, statuses)), encoding="utf-8")
    (docs / "executive_decision_report.md").write_text(f"# Decision\n\nTerminal decision: `{state}`\n", encoding="utf-8")
    return tmp_path


def _codes(root: Path) -> set[str]:
    return {issue.code for issue in validator.validate(root)}


def test_complete_go_inventory_passes(tmp_path: Path) -> None:
    root = _complete_tree(tmp_path)
    assert validator.validate(root) == []


def test_missing_required_artifact_fails(tmp_path: Path) -> None:
    root = _complete_tree(tmp_path)
    (root / "docs" / "childlens_feasibility_v1" / "commands_tests_and_limitations.md").unlink()
    assert "MISSING_REQUIRED_FILE" in _codes(root)


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        ({"participant_id": "private-value"}, "RESTRICTED_MACHINE_FIELD"),
        ({"utterance_start": 1.25}, "EXACT_MEDIA_TIME_FIELD"),
        ({"tokenizer_source": "BabyView"}, "FORBIDDEN_EMPIRICAL_ANCESTRY"),
        ({"checkpoint_ancestry": "AEA"}, "FORBIDDEN_EMPIRICAL_ANCESTRY"),
    ],
)
def test_machine_payload_and_forbidden_ancestry_are_flagged(
    tmp_path: Path, payload: dict, expected_code: str
) -> None:
    root = _complete_tree(tmp_path)
    bad = root / "output" / "childlens_feasibility_v1" / "extra_evidence.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    assert expected_code in _codes(root)


def test_explicit_false_ancestry_receipt_is_allowed(tmp_path: Path) -> None:
    root = _complete_tree(tmp_path)
    receipt = root / "output" / "childlens_feasibility_v1" / "boundary_receipt.json"
    receipt.write_text(
        json.dumps({"forbidden_ancestry": {"AEA": False, "BabyView": False}}),
        encoding="utf-8",
    )
    assert validator.validate(root) == []


def test_public_audit_datetime_is_allowed(tmp_path: Path) -> None:
    root = _complete_tree(tmp_path)
    receipt = root / "output" / "childlens_feasibility_v1" / "dated_receipt.json"
    receipt.write_text(json.dumps({"observed_at": "2026-07-21T12:34:56-07:00"}), encoding="utf-8")
    assert validator.validate(root) == []


@pytest.mark.parametrize(
    ("text", "expected_code"),
    [
        ("participant_id: private-value\n", "DIRECT_IDENTIFIER_ASSIGNMENT"),
        ("example_clip.mp4\n", "MEDIA_FILENAME_PATTERN"),
        ("00:01:02.250 --> 00:01:04.000\n", "SUBTITLE_CUE_PATTERN"),
        ("CHILD: example dialogue\n", "DIALOGUE_PAYLOAD_PATTERN"),
    ],
)
def test_text_payload_patterns_are_flagged(tmp_path: Path, text: str, expected_code: str) -> None:
    root = _complete_tree(tmp_path)
    bad = root / "docs" / "childlens_feasibility_v1" / "extra.md"
    bad.write_text(text, encoding="utf-8")
    assert expected_code in _codes(root)


def test_revise_requires_nonpass_gate_and_bounded_correction(tmp_path: Path) -> None:
    statuses = ["PASS"] * 10
    statuses[2] = "UNRESOLVED"
    root = _complete_tree(tmp_path, "CHILDLENS_FEASIBILITY_REVISE", statuses)
    assert validator.validate(root) == []
    record_path = root / "docs" / "childlens_feasibility_v1" / "decision_record_v1.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["gate_assessments"][2].pop("bounded_correction")
    record_path.write_text(json.dumps(record), encoding="utf-8")
    assert "REVISE_CORRECTION_MISSING" in _codes(root)


def test_decision_record_and_executive_must_agree(tmp_path: Path) -> None:
    root = _complete_tree(tmp_path)
    executive = root / "docs" / "childlens_feasibility_v1" / "executive_decision_report.md"
    executive.write_text("Terminal decision: `CHILDLENS_FEASIBILITY_STOP`\n", encoding="utf-8")
    assert "EXECUTIVE_DECISION_MISMATCH" in _codes(root)


def test_decision_record_must_contain_exactly_one_terminal_literal(tmp_path: Path) -> None:
    root = _complete_tree(tmp_path)
    record_path = root / "docs" / "childlens_feasibility_v1" / "decision_record_v1.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["contradictory_state"] = "CHILDLENS_FEASIBILITY_STOP"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    assert "TERMINAL_STATE_CARDINALITY" in _codes(root)


def test_stop_requires_an_essential_failure(tmp_path: Path) -> None:
    root = _complete_tree(tmp_path, "CHILDLENS_FEASIBILITY_STOP")
    assert "STOP_LOGIC_VIOLATION" in _codes(root)


def test_absent_decision_record_fails_terminal_validation(tmp_path: Path) -> None:
    root = _complete_tree(tmp_path)
    (root / "docs" / "childlens_feasibility_v1" / "decision_record_v1.json").unlink()
    codes = _codes(root)
    assert "MISSING_REQUIRED_FILE" in codes
    assert "DECISION_RECORD_ABSENT" in codes
