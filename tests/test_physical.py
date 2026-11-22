import numpy as np

from nursery.simulator.physical import (contact_area, contact_target, draw_properties, load_materials, load_stats,
                                        object_acceleration, skin_dynamics, window_force)

TABLE = load_materials()


def test_static_lifted_object_weighs_mass_times_g_downward():
    transl = np.tile([0.0, 0.0, 1.0], (60, 1))
    force = window_force(object_acceleration(transl, 120.0), transl[:, 2], 0.8, 0.5)
    assert np.allclose(force, [0, 0, -0.5 * 9.81], atol=1e-6)
    stats = load_stats(force, np.eye(3))
    assert np.isclose(stats[3], 0.5 * 9.81) and np.isclose(stats[4], 0) and np.isclose(stats[2], -0.5 * 9.81)


def test_load_is_zero_unless_lifted_and_follows_acceleration():
    t = np.arange(120) / 120.0
    z = 1.0 + 0.5 * 2.0 * t ** 2  # constant 2 m/s^2 upward acceleration
    transl = np.stack([np.zeros_like(t), np.zeros_like(t), z], 1)
    force = window_force(object_acceleration(transl, 120.0), z, 1.0, 1.0)
    assert np.allclose(force[30:90, 2], -(9.81 + 2.0), atol=1e-3)
    assert not force[z <= 1.01].any()


def test_camera_frame_rotation():
    rotation = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # camera-to-world, world z = camera -y
    stats = load_stats(np.tile([0.0, 0.0, -1.0], (5, 1)), rotation)
    assert np.allclose(stats[:3], rotation.T @ [0, 0, -1.0])


def test_metal_cooler_than_wood_and_target_ignores_area():
    assert contact_target(TABLE, "metal") < contact_target(TABLE, "wood") < 0
    assert contact_area(50) == 0.5 and contact_area(400) == 1.0 and contact_area(0) == 0.0


def test_skin_dynamics_rise_fast_recover_slow_and_reset():
    target = np.full(10, -4.0)
    touching = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 1], bool)
    new_clip = np.zeros(10, bool)
    new_clip[7] = True
    dt = skin_dynamics(target, touching, new_clip)
    assert dt[0] < -2.0 and dt[3] < dt[0]
    assert dt[4] > dt[3] and abs(dt[4]) > abs(dt[3]) * 0.8  # slow recovery
    assert dt[7] > -2.6 and np.isclose(dt[7], -4.0 * (1 - np.exp(-1)))  # reset, then rise from 0


def test_property_draws_are_deterministic_and_vary_within_a_word():
    assert draw_properties(TABLE, "spoon", "inst_a") == draw_properties(TABLE, "spoon", "inst_a")
    draws = [draw_properties(TABLE, "spoon", f"inst_{i}") for i in range(60)]
    assert len({m for m, _ in draws}) > 1 and len({round(g, 6) for _, g in draws}) > 30
    assert all(0.015 <= g <= 0.080 for _, g in draws)


def test_acceleration_recovers_constant_and_matches_at_edges():
    t = np.arange(200) / 120.0
    transl = np.stack([0.3 * t, np.zeros_like(t), 0.5 * 3.0 * t ** 2], 1)
    acc = object_acceleration(transl, 120.0)
    assert np.allclose(acc[:, 2], 3.0, atol=1e-6) and np.allclose(acc[:, :2], 0, atol=1e-6)


def test_key_by_clip_changes_key_and_is_deterministic(monkeypatch):
    import nursery.simulator.grab as grab
    from nursery.simulator.physical import _Clips

    fake = {"object": {"params": {"transl": np.zeros((40, 3))}}, "framerate": 120.0, "obj_name": "mug"}
    monkeypatch.setattr(grab, "load_clip", lambda path: fake)
    vocabulary = {"extras": {"mug": "mug"}}
    by_instance = _Clips("/x", {"sources": {}}, TABLE, vocabulary)
    by_clip = _Clips("/x", {"sources": {}}, TABLE, vocabulary, key_by="clip")
    again = _Clips("/x", {"sources": {}}, TABLE, vocabulary, key_by="clip")
    assert by_instance.get("s1/mug_a_1.npz")["key"] == "grab:mug"
    one, two = by_clip.get("s1/mug_a_1.npz"), by_clip.get("s1/mug_a_2.npz")
    assert one["key"] == "clip:s1/mug_a_1.npz" and one["key"] != two["key"]
    assert (one["material"], one["mass"]) == (again.get("s1/mug_a_1.npz")["material"], again.get("s1/mug_a_1.npz")["mass"])
    assert one["mass"] != two["mass"]
