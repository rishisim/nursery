from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_childlens_immutability_v1_2.py"
SPEC = importlib.util.spec_from_file_location("validate_childlens_immutability_v1_2", SCRIPT)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def test_v1_baseline_matches_prior_receipt() -> None:
    receipt = validator.build_receipt(REPO_ROOT, date(2026, 7, 21))
    assert receipt["v1"]["artifact_count"] == 23
    assert receipt["v1"]["artifact_set_digest"] == validator.EXPECTED_V1_DIGEST
    assert receipt["v1"]["matches_prior_baseline"] is True


def test_v1_1_set_is_nonempty_and_excludes_v1_2() -> None:
    files = validator.v1_1_files(REPO_ROOT)
    relative = [path.relative_to(REPO_ROOT).as_posix() for path in files]
    assert files
    assert all("v1_2" not in path for path in relative)
    assert "output/childlens_feasibility_v1_1/decision_record.json" in relative
    assert "scripts/validate_childlens_feasibility_v1_1.py" in relative
    receipt = validator.build_receipt(REPO_ROOT, date(2026, 7, 21))
    assert receipt["v1_1"]["artifact_count"] == validator.EXPECTED_V1_1_COUNT
    assert receipt["v1_1"]["artifact_set_digest"] == validator.EXPECTED_V1_1_DIGEST
    assert receipt["v1_1"]["matches_v1_2_baseline"] is True


def test_canonical_serialization_is_stable_under_input_order(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    serialized_a, digest_a = validator.canonical_artifact_set(tmp_path, [first, second])
    serialized_b, digest_b = validator.canonical_artifact_set(tmp_path, [second, first])
    assert serialized_a == serialized_b
    assert digest_a == digest_b
    assert all("  " in line for line in serialized_a.splitlines())
