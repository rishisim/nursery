from pathlib import Path

import numpy as np

from babyworld_lite.childlens_engine_bakeoff.unity_native_gate.anatomical_rig import (
    compare_central_difference,
    load_manifest,
)


def test_manifest_is_single_authority_and_exact_clock():
    manifest = load_manifest()
    assert manifest["authority"]["physics"].endswith("ArticulationBody/PhysX")
    assert manifest["authority"]["assistance_ledger_expected_entries"] == 0
    assert manifest["clock"] == {"physics_hz": 240, "render_hz": 30, "steps_per_frame": 8}
    source = Path("babyworld_lite/childlens_engine_bakeoff/unity_native_gate/AnatomicalPhysicsRigBuilder.cs").read_text()
    assert 'collision_check_available = false' in source
    assert 'collision_status = "NOT_MEASURED"' in source
    assert "UnintendedSelfContact" not in source
    assert "DlsStep" not in source


def test_measured_anatomical_lengths_are_child_sized_and_nonzero():
    landmarks = load_manifest()["landmarks_m"]
    distance = lambda a, b: np.linalg.norm(np.asarray(landmarks[a]) - np.asarray(landmarks[b]))
    assert 0.14 < distance("shoulder", "elbow") < 0.22
    assert 0.14 < distance("elbow", "wrist") < 0.24
    for digit in ("thumb", "index", "middle", "ring", "little"):
        assert 0.01 < distance(f"{digit}_1", f"{digit}_2") < 0.04
        assert 0.01 < distance(f"{digit}_2", f"{digit}_3") < 0.04


def test_analytic_jacobian_matches_central_difference_at_multiple_poses():
    manifest = load_manifest()
    site = np.asarray(manifest["landmarks_m"]["middle_3"])
    poses = (
        np.radians([5, 12, -8, 55, 15, -10, 8]),
        np.radians([-20, 35, 18, 80, -25, 20, -12]),
        np.radians([30, -20, 25, 40, 30, -25, 15]),
    )
    tolerance = manifest["frozen_tolerances"]
    for pose in poses:
        rows = compare_central_difference(manifest, pose, site)
        assert max(row.direction_error_deg for row in rows) <= tolerance["jacobian_column_direction_max_deg"]
        assert max(row.relative_magnitude_error for row in rows) <= tolerance["jacobian_column_relative_magnitude_max"]


def test_frozen_gate_tolerances_match_contract():
    tolerance = load_manifest()["frozen_tolerances"]
    assert tolerance["collider_skin_max_m"] <= 0.0075
    assert tolerance["palm_position_max_m"] <= 0.010
    assert tolerance["palm_orientation_max_deg"] <= 7
    assert tolerance["finger_object_penetration_max_m"] <= 0.003
    assert tolerance["support_penetration_max_m"] <= 0.002
    assert tolerance["minimum_lift_m"] >= 0.08
    assert tolerance["minimum_rotation_deg"] >= 20
