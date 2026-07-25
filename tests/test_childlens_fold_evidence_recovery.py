import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_recovery_is_exact_and_cue_lift_is_sealed() -> None:
    receipt = json.loads(
        (ROOT / "output/childlens_fold_evidence_recovery/recovery_receipt.json")
        .read_text(encoding="utf-8")
    )
    assert receipt["recovery"]["all_bound_hashes_match"] is True
    assert receipt["recovery"]["fold_counts"] == [6, 6, 6]
    assert receipt["recovery"]["locked_participant_count"] == 0
    assert receipt["reproduction"]["byte_identical"] is True
    assert receipt["cue_lift_arms_run"] is False


def test_terminal_is_crossfit_no_go() -> None:
    receipt = json.loads(
        (ROOT / "output/childlens_fold_evidence_recovery/recovery_receipt.json")
        .read_text(encoding="utf-8")
    )
    assert receipt["decision"] == "NO_GO_CROSSFIT_MEDIA_CALIBRATION"
    assert receipt["crossfit"]["generator_configurations_scored"] == 0
    assert receipt["readiness_controls_run"] is False
