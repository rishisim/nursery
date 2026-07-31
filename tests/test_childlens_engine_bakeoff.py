import json
from pathlib import Path


def test_engine_bakeoff_protocol_is_frozen_and_childlens_only():
    protocol = json.loads(
        Path("configs/childlens_mimo_molmospaces_engine_bakeoff.json").read_text()
    )
    assert protocol["frozen_before_outcome_rendering"] is True
    assert protocol["empirical_source"] == "ChildLens only"
    assert protocol["appearance_gates"]["minimum_target_projected_area_fraction"] == 0.015
    assert protocol["developmental_scope"]["mimo_age_months"] == 24
    assert len(protocol["terminal_ladder"]) == 6
    assert protocol["camera_uncertainty_set"]["canonical"] == "equisolid_140_diagonal"
    assert protocol["camera_uncertainty_set"]["learner_outcome_tuning_forbidden"] is True


def test_historical_protocol_does_not_authorize_private_media_or_causal_claims():
    protocol = json.loads(
        Path("configs/childlens_mimo_molmospaces_spec_kernel.json").read_text()
    )
    boundaries = protocol["claim_boundaries"]
    assert boundaries["learner_or_cue_lift_run"] is False
    assert boundaries["private_childlens_material_permitted"] is False
    assert boundaries["developmental_calibration"] == "provisional_not_age_matched"
