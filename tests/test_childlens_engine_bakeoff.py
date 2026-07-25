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


def test_terminal_decision_is_route_by_route_and_privacy_safe():
    evidence = json.loads(
        Path(
            "output/childlens_mimo_molmospaces_engine_bakeoff/aggregate_evidence.json"
        ).read_text()
    )
    matrix = json.loads(
        Path(
            "output/childlens_mimo_molmospaces_engine_bakeoff/engine_decision_matrix.json"
        ).read_text()
    )
    assert evidence["decision"] == "NO_GO_ENGINE_LADDER_EXHAUSTED"
    assert evidence["private_childlens_material_in_evidence"] is False
    assert matrix["terminal_decision"] == evidence["decision"]
    routes = {row["route"] for row in matrix["routes"]}
    assert {
        "stock_MIMo_v2",
        "stock_MolmoSpaces_classic",
        "stock_MolmoSpaces_Filament",
        "hybrid_A_custom_robot_API",
        "hybrid_B_direct_MJCF",
        "hybrid_C_mesh_import",
        "Blender_exact_replay",
    } <= routes
