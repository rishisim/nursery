import mujoco

from babyworld_lite.childlens_engine_bakeoff.controllers import (
    ContactTriggeredGrasp,
    GraspState,
    minimum_hand_target_distance,
)


def _distance_model(target_x):
    xml = f"""
    <mujoco>
      <worldbody>
        <geom name="hand_a" type="sphere" pos="0 0 0" size="0.05"/>
        <geom name="hand_b" type="sphere" pos="0.2 0 0" size="0.05"/>
        <geom name="target" type="sphere" pos="{target_x} 0 0" size="0.05"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_distance_certificate_selects_closest_of_all_hand_geoms():
    model, data = _distance_model(0.32)
    sample = minimum_hand_target_distance(model, data, [0, 1], 2)
    assert abs(sample.signed_distance_m - 0.02) < 1e-9
    assert sample.hand_geom_id == 1


def test_distance_certificate_reports_overlap():
    model, data = _distance_model(0.24)
    sample = minimum_hand_target_distance(model, data, [0, 1], 2)
    assert sample.signed_distance_m < 0


def test_grasp_state_machine_forbids_precontact_engagement_and_records_clock():
    grasp = ContactTriggeredGrasp()
    try:
        grasp.engage(time_s=0.1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("precontact engagement was accepted")
    assert grasp.observe_contact(time_s=0.2, contact_count=1)
    grasp.engage(time_s=0.2)
    grasp.release(time_s=1.5)
    assert grasp.state is GraspState.RELEASED
    assert grasp.receipt()["contact_precedes_or_equals_engagement"] is True
