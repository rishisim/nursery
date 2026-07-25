import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_amendment_is_prospective_and_seals_cue_lift():
    config = json.loads(
        (ROOT / "configs/childlens_asset_rich_grounding_amendment.json").read_text()
    )
    assert config["scope"]["sole_empirical_child_source"] == "ChildLens"
    assert config["scope"]["locked_access_permitted"] is False
    assert config["scope"]["cue_lift_experiment_authorized"] is False
    assert config["historical_records"]["historical_no_go_files_immutable"] is True


def test_alignment_unit_error_is_explicitly_corrected():
    config = json.loads(
        (ROOT / "configs/childlens_asset_rich_grounding_amendment.json").read_text()
    )
    alignment = config["alignment_amendment"]
    assert alignment["historical_values_preserved"] == [0.0, 0.00177, 0.0062]
    assert alignment["historical_values_are_seconds"] is False
    assert alignment["childlens_referential_lag_identifiable"] is False
    assert alignment["treatment_construction_status"] == "sealed"


def test_pilot_gate_precedes_distribution_gate():
    config = json.loads(
        (ROOT / "configs/childlens_asset_rich_grounding_amendment.json").read_text()
    )
    assert config["pilot_gate"]["minimum_complete_episodes"] >= 3
    assert config["pilot_gate"]["visible_nontelekinetic_effector"] is True
    assert config["distribution_gate_if_pilot_passes"]["assessment_adaptation_forbidden"] is True


def test_terminal_no_go_stops_before_distribution_and_cue_lift():
    receipt = json.loads(
        (ROOT / "output/childlens_asset_rich_grounding/engineering_receipt.json").read_text()
    )
    assert receipt["decision"] == "NO_GO_ASSET_RICH_FIDELITY"
    assert receipt["frozen_pilot_gate"]["semantic_appearance_fidelity"] == "FAIL"
    assert receipt["frozen_pilot_gate"]["distribution_generation_permitted"] is False
    assert receipt["downstream"]["crossfit_scored"] is False
    assert receipt["downstream"]["learner_outcomes_run"] is False
    assert receipt["downstream"]["cue_lift_arms_instantiated"] is False
