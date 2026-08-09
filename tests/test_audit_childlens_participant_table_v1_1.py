from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_childlens_participant_table_v1_1.py"
SPEC = importlib.util.spec_from_file_location("participant_audit_v1_1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
audit = MODULE.audit


def test_aggregate_receipt_does_not_return_values(tmp_path: Path) -> None:
    path = tmp_path / "participants.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["anonymized_child", "recording_date", "language", "age_years"])
        for index in range(6):
            writer.writerow([f"SECRET_CHILD_{index}", f"2099-01-{index + 1:02d}", "SECRET_LANG", "4"])
    receipt = audit(path)
    rendered = str(receipt)
    assert receipt["row_count"] == 6
    assert receipt["unique_participant_count"] == 6
    assert receipt["participant_session_complete_row_count"] == 6
    assert receipt["language_distinct_value_count"] == 1
    assert "SECRET" not in rendered
    assert receipt["raw_values_returned"] is False


def test_rejects_malformed_delimited_table(tmp_path: Path) -> None:
    path = tmp_path / "participants.csv"
    path.write_text("child,date\none,2099-01-01\ntwo\n", encoding="utf-8")
    try:
        audit(path)
    except ValueError as exc:
        assert str(exc) in {"E_DELIMITER", "E_ROW_WIDTH"}
    else:
        raise AssertionError("expected failure")
