from pathlib import Path


SOURCE = Path(
    "babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate/PhysicsTruthRecorder.cs"
).read_text()


def test_configurable_joint_recorder_reports_the_configured_active_drive():
    assert "bool isSlerp = joint.rotationDriveMode == RotationDriveMode.Slerp;" in SOURCE
    assert "JointDrive activeDrive = isSlerp ? joint.slerpDrive : joint.angularXDrive;" in SOURCE
    assert "active_drive_mode = activeDriveMode" in SOURCE
    assert "active_drive_spring_n_m_rad = activeDrive.positionSpring" in SOURCE
    assert "active_drive_damper_n_m_s_rad = activeDrive.positionDamper" in SOURCE
    assert "active_drive_max_force_n_m = activeDrive.maximumForce" in SOURCE
    assert "inactive fields" in SOURCE
