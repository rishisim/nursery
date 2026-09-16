"""Execute a converted interaction with the general InterMimic controller.

No reference replay: frame zero initializes the actors, then only controller
actions move the human and PhysX moves the selected object. Room meshes remain
static. The same path is used for one or many environments; the canonical
configuration starts with one environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

from ..embodiedgen_hoidini.prepare_scene import Pose, _load_obj, _combine_meshes
from ..embodiedgen_hoidini.run_motion import _output_directory
from .schema import FIELDS, WIDTH, validate_reference


DEFAULT_CONFIG = Path(__file__).with_name("execution.json")
FAILURE_CATEGORIES = {
    "invalid_reference", "retargeting", "controller_mismatch", "asset_mismatch",
    "simulation_failure", "interaction_failure",
}


class ExecutionError(ValueError):
    """An execution failure with one of the protocol's general diagnoses."""

    def __init__(self, category, message):
        if category not in FAILURE_CATEGORIES:
            raise ValueError(f"Unknown execution failure category: {category}")
        super().__init__(message)
        self.category = category


def load_execution_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported execution configuration schema")
    physics_hz = config["rates_hz"]["physics"]
    control_hz = config["rates_hz"]["control"]
    if (not isinstance(config["num_environments"], int) or
            config["num_environments"] < 1):
        raise ValueError("num_environments must be a positive integer")
    if physics_hz <= 0 or control_hz <= 0 or physics_hz % control_hz:
        raise ValueError("physics_hz must be a positive integer multiple of control_hz")
    if config["controller"]["role"] != "general_student":
        raise ValueError("Only a metadata-declared general student controller is supported")
    if config["termination"]["contact_miss_streak_steps"] != 11:
        raise ValueError("InterMimic contact termination is fixed at 11 missed steps")
    if config["recording"] != {
            "format": "npz_compressed", "states": True, "contacts": True,
            "actions": True, "video": False}:
        raise ValueError("The canonical synchronized state recording cannot be disabled")
    return config


def checkpoint_signature(checkpoint):
    """Extract the network contract without using filenames or activity names."""
    try:
        model = checkpoint["model"]
        running = checkpoint["running_mean_std"]
        actor_input = model["a2c_network.actor_mlp.0.weight"]
        action_output = model["a2c_network.mu.weight"]
        mean = running["running_mean"]
    except (KeyError, TypeError) as exc:
        raise ExecutionError("controller_mismatch", "Checkpoint lacks the InterMimic policy contract") from exc
    return {
        "observation_dimension": int(actor_input.shape[1]),
        "action_dimension": int(action_output.shape[0]),
        "normalization_dimension": int(mean.shape[0]),
    }


def validate_controller(checkpoint_path, checkpoint, manifest, config):
    """Validate artifact and network metadata, independent of the activity."""
    expected = config["controller"]
    digest = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    if digest != expected["checkpoint_sha256"]:
        raise ExecutionError(
            "controller_mismatch",
            f"Controller SHA-256 {digest} does not match the declared general student artifact",
        )
    signature = checkpoint_signature(checkpoint)
    required = {
        "observation_dimension": expected["observation_dimension"],
        "action_dimension": expected["action_dimension"],
        "normalization_dimension": expected["observation_dimension"],
    }
    if signature != required:
        raise ExecutionError(
            "controller_mismatch", f"Controller network contract {signature} != {required}")
    artifact = {
        "units": manifest.get("units"), "up_axis": manifest.get("up_axis"),
        "body_count": len(manifest.get("body_names", [])),
        "reference_width": WIDTH,
    }
    declared = {key: expected[key] for key in artifact}
    if artifact != declared:
        raise ExecutionError(
            "controller_mismatch", f"Converted artifact contract {artifact} != {declared}")
    return {"sha256": digest, **signature, "role": expected["role"]}


def room_collision_mesh(receipt):
    """Reuse recorded mesh transforms; omit the separately simulated target."""
    meshes, sources = [], []
    target = receipt["active_interaction"]["target_name"]
    for name, node in receipt["nodes"].items():
        if name == target:
            continue
        geometries = node["collision_geometry"]
        if not geometries:
            if node["visual_geometry"]:
                raise ExecutionError("asset_mismatch", f"Scene node has no collision geometry: {name}")
            continue
        pose = Pose(tuple(node["pose"][:3]), tuple(node["pose"][3:])).matrix
        for geometry in geometries:
            mesh = _load_obj(Path(geometry["mesh_path"]))
            meshes.append(mesh.transformed(pose @ np.asarray(geometry["mesh_to_asset"])))
            sources.append({"node": name, "mesh": geometry["mesh_path"]})
    if not meshes:
        raise ExecutionError("asset_mismatch", "No room collision geometry")
    return _combine_meshes(meshes), sources


def _write_decision(output, category, outcome, evidence):
    decision = {
        "outcome": outcome,
        "failure_category": category,
        "execution_logic_activity_independent": True,
        "evidence": evidence,
    }
    (Path(output) / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def prepare_execution(converted, output, config):
    converted = Path(converted).resolve()
    try:
        manifest = json.loads((converted / "manifest.json").read_text())
        if manifest.get("units") != "metres" or manifest.get("up_axis") != "+Z":
            raise ExecutionError("invalid_reference", "Expected metre, +Z converted motion")
        for key in ("motion_file", "humanoid_file", "object_mesh", "object_points"):
            if not (converted / manifest[key]).is_file():
                raise ExecutionError("invalid_reference", f"Missing converted artifact: {manifest[key]}")
        receipt = json.loads((Path(manifest["source_run"]) / "input.json").read_text())
        mesh = _load_obj(converted / manifest["object_mesh"])
        original = receipt["active_interaction"]["canonical_target_mesh"]
        if not np.array_equal(mesh.faces, original["faces"]) or not np.allclose(
                mesh.vertices, original["vertices"], atol=1e-8, rtol=0):
            raise ExecutionError(
                "asset_mismatch", "Converted object differs from the motion-generation object")
        room, sources = room_collision_mesh(receipt)
    except ExecutionError:
        raise
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise ExecutionError("invalid_reference", str(exc)) from exc

    output = _output_directory(output)
    simulation = output / "simulation"
    simulation.mkdir()
    motions = simulation / "motion"
    motions.mkdir()
    reference = motions / f"sub0_{manifest['object_name']}_000.pt"
    shutil.copyfile(converted / manifest["motion_file"], reference)
    for source, destination in [(manifest["humanoid_file"], "humanoid.xml"),
                                (manifest["object_mesh"], "object.obj"),
                                (manifest["object_points"], "sample_points.npy")]:
        shutil.copyfile(converted / source, simulation / destination)
    robot = ET.Element("robot", name="interaction_object")
    link = ET.SubElement(robot, "link", name="object")
    for kind in ("visual", "collision"):
        geometry = ET.SubElement(ET.SubElement(link, kind), "geometry")
        ET.SubElement(geometry, "mesh", filename="object.obj")
    ET.ElementTree(robot).write(simulation / "object.urdf", encoding="unicode")
    record = {
        "prompt": receipt["active_interaction"]["prompt"], "seed": config["seed"],
        "converted_run": str(converted), "source_scene": manifest["source_scene"],
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "source_motion_validation": manifest["source_motion_validation"],
        "room_collision_sources": sources,
        "execution_config": config,
        "stages": {"prepared": True, "controller_validated": False, "physics_executed": False},
        "video": None,
    }
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return output, simulation, room, record, manifest, reference


def _quaternion_angle(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.abs(np.sum(a * b, axis=-1)).clip(0, 1)
    return 2 * np.arccos(dot)


def humanoid_body_states(raw_states, num_environments, body_count):
    """Select humanoid bodies from Isaac Gym's all-actor body tensor."""
    shaped = np.asarray(raw_states).reshape(num_environments, -1, 13)
    if shaped.shape[1] < body_count:
        raise ExecutionError(
            "asset_mismatch",
            f"Simulator exposes {shaped.shape[1]} bodies; expected at least {body_count}",
        )
    return shaped[:, :body_count]


def measure_execution(reference, initial_actors, states, bodies, contacts,
                      object_contacts, actions, config, terminated,
                      tracking_termination, contact_miss_streak):
    """Compute general, activity-independent measurements from saved state."""
    control_hz = config["rates_hz"]["control"]
    expected_steps = len(reference) - 1
    steps = len(states)
    finite = all(np.isfinite(value).all() for value in (
        initial_actors, states, bodies, contacts, object_contacts, actions))
    reference_steps = reference[1:steps + 1]
    reference_bodies = reference_steps[:, FIELDS["body_pos"]].reshape(steps, 52, 3)
    reference_bodies = reference_bodies[:, None]
    human_error = np.linalg.norm(bodies[..., :3] - reference_bodies, axis=-1)
    reference_object_position = reference_steps[:, FIELDS["obj_pos"]][:, None]
    object_position_error = np.linalg.norm(
        states[:, :, 1, :3] - reference_object_position, axis=-1)
    reference_object_rotation = reference_steps[:, FIELDS["obj_rot"]][:, None]
    object_rotation_error = _quaternion_angle(
        states[:, :, 1, 3:7], reference_object_rotation)

    initial_root_position_error = np.linalg.norm(
        initial_actors[:, 0, :3] - reference[0, FIELDS["root_pos"]], axis=-1)
    initial_root_rotation_error = _quaternion_angle(
        initial_actors[:, 0, 3:7], reference[0, FIELDS["root_rot"]])
    initial_object_position_error = np.linalg.norm(
        initial_actors[:, 1, :3] - reference[0, FIELDS["obj_pos"]], axis=-1)
    initial_object_rotation_error = _quaternion_angle(
        initial_actors[:, 1, 3:7], reference[0, FIELDS["obj_rot"]])
    position_tolerance = config["measurement"]["initialization_tolerance_m"]
    rotation_tolerance = config["measurement"]["initialization_tolerance_rad"]
    initialized = bool(
        max(initial_root_position_error.max(), initial_object_position_error.max()) <= position_tolerance
        and max(initial_root_rotation_error.max(), initial_object_rotation_error.max()) <= rotation_tolerance)

    threshold = config["measurement"]["contact_force_threshold_n"]
    actual_human_contact = np.linalg.norm(contacts, axis=-1) > threshold
    expected_human_contact = (
        reference_steps[:, FIELDS["contact_human"]] == 1)[:, None]
    expected_human_entries = int(expected_human_contact.sum() * states.shape[1])
    matched_human_entries = int((actual_human_contact & expected_human_contact).sum())
    actual_object_contact = (np.linalg.norm(object_contacts, axis=-1) > threshold).any(axis=-1)
    expected_object_contact = (
        reference_steps[:, FIELDS["contact_obj"]].reshape(steps, 1) == 1)
    expected_object_entries = int(expected_object_contact.sum() * states.shape[1])
    matched_object_entries = int((actual_object_contact & expected_object_contact).sum())
    expected_contact_frames = np.any(expected_human_contact, axis=-1)
    matched_contact_frames = np.any(actual_human_contact & expected_human_contact, axis=-1)

    displacement = states[:, :, 1, :3] - initial_actors[None, :, 1, :3]
    root_height = bodies[:, :, 0, 2]
    root_height_threshold = config["termination"]["root_height_m"]
    timestamps = (np.arange(steps, dtype=np.float64) + 1) / control_hz
    synchronization = {
        "state_samples": steps, "action_samples": len(actions),
        "timestamps": len(timestamps),
        "first_timestamp_s": float(timestamps[0]) if steps else None,
        "last_timestamp_s": float(timestamps[-1]) if steps else None,
        "sample_period_s": 1 / control_hz,
        "aligned": bool(steps == len(actions) == len(timestamps)),
    }
    return {
        "initialization": {
            "within_tolerance": initialized,
            "maximum_root_position_error_m": float(initial_root_position_error.max()),
            "maximum_root_rotation_error_rad": float(initial_root_rotation_error.max()),
            "maximum_object_position_error_m": float(initial_object_position_error.max()),
            "maximum_object_rotation_error_rad": float(initial_object_rotation_error.max()),
        },
        "finite": bool(finite),
        "completion_fraction": float(steps / expected_steps),
        "human_tracking": {
            "mean_body_position_error_m": float(human_error.mean()),
            "maximum_body_position_error_m": float(human_error.max()),
        },
        "object_tracking": {
            "mean_position_error_m": float(object_position_error.mean()),
            "maximum_position_error_m": float(object_position_error.max()),
            "mean_rotation_error_rad": float(object_rotation_error.mean()),
            "maximum_rotation_error_rad": float(object_rotation_error.max()),
        },
        "expected_contact": {
            "human_expected_entries": expected_human_entries,
            "human_matched_entries": matched_human_entries,
            "human_agreement": (float(matched_human_entries / expected_human_entries)
                                if expected_human_entries else None),
            "object_expected_frames": expected_object_entries,
            "object_matched_frames": matched_object_entries,
            "object_agreement": (float(matched_object_entries / expected_object_entries)
                                 if expected_object_entries else None),
            "expected_duration_s": float(expected_contact_frames.sum() / control_hz),
            "matched_duration_s": float(matched_contact_frames.sum() / control_hz),
        },
        "termination": {
            "terminated": bool(terminated),
            "tracking_termination": bool(tracking_termination),
            "fall": bool((root_height[2:] < root_height_threshold).any()),
            "minimum_root_height_m": float(root_height.min()),
            "hand_contact_miss_streak": list(contact_miss_streak),
        },
        "object_movement": {
            "maximum_displacement_m": float(np.linalg.norm(displacement, axis=-1).max()),
            "net_displacement_m": float(np.linalg.norm(displacement[-1], axis=-1).max()),
            "maximum_vertical_change_m": float(displacement[..., 2].max()),
            "minimum_vertical_change_m": float(displacement[..., 2].min()),
        },
        "synchronization": synchronization,
    }


def diagnose_result(measurements, completed_reference):
    if not measurements["finite"] or not measurements["synchronization"]["aligned"]:
        return "simulation_failure", ["non-finite or unsynchronized simulated state"]
    if not measurements["initialization"]["within_tolerance"]:
        return "retargeting", ["frame-zero simulated actors do not match the converted reference"]
    if not completed_reference:
        termination = measurements["termination"]
        return "interaction_failure", [
            f"completion_fraction={measurements['completion_fraction']:.6f}",
            f"terminated={termination['terminated']}",
            f"tracking_termination={termination['tracking_termination']}",
            f"fall={termination['fall']}",
        ]
    return None, ["reference duration completed with finite synchronized state"]


def execute(converted, output, upstream, checkpoint, config_path=DEFAULT_CONFIG):
    # Isaac Gym must be imported before Torch; use its isolated Python 3.8 env.
    from isaacgym import gymapi, gymutil
    import torch
    import yaml

    config = load_execution_config(config_path)
    upstream, checkpoint = Path(upstream).resolve(), Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(upstream / "isaacgym/src"))
    from intermimic.env.tasks.intermimic import InterMimic

    try:
        output, simulation, room, record, manifest, reference_path = prepare_execution(
            converted, output, config)
    except ExecutionError as exc:
        failed_output = _output_directory(output)
        decision = _write_decision(failed_output, exc.category, "failed", [str(exc)])
        (failed_output / "run.json").write_text(json.dumps({
            "converted_run": str(Path(converted).resolve()),
            "execution_config": config,
            "stages": {"prepared": False, "controller_validated": False,
                       "physics_executed": False},
            "error": str(exc), "decision": decision,
        }, indent=2) + "\n")
        raise
    environment = None
    category = "simulation_failure"
    try:
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        controller = validate_controller(checkpoint, checkpoint_data, manifest, config)
        record["controller"] = controller
        record["checkpoint"] = str(checkpoint)
        record["stages"]["controller_validated"] = True
        category = "asset_mismatch"

        cfg_path = upstream / "isaacgym/src/intermimic/data/cfg/omomo_all.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        rates = config["rates_hz"]
        cfg["env"].update(
            numEnvs=config["num_environments"], dataSub=["sub0"],
            motion_file=str(simulation / "motion"), robotType="humanoid.xml",
            stateInit="Start", physicalBufferSize=1, enableEvaluation=False,
            playdataset=False, saveImages=False,
            objectDensity=config["object"]["density_kg_m3"],
            controlFrequencyInv=rates["physics"] // rates["control"],
            enableEarlyTermination=config["termination"]["early_termination"],
        )
        cfg["env"]["asset"]["assetRoot"] = str(simulation)
        cfg["env"]["plane"].update(
            staticFriction=config["plane"]["static_friction"],
            dynamicFriction=config["plane"]["dynamic_friction"],
            restitution=config["plane"]["restitution"],
        )
        physics = config["physics"]
        cfg["sim"].update(substeps=physics["substeps"])
        cfg["sim"]["physx"].update(
            num_threads=physics["num_threads"], solver_type=physics["solver_type"],
            num_position_iterations=physics["num_position_iterations"],
            num_velocity_iterations=physics["num_velocity_iterations"],
            contact_offset=physics["contact_offset_m"], rest_offset=physics["rest_offset_m"],
            bounce_threshold_velocity=physics["bounce_threshold_velocity_m_s"],
            max_depenetration_velocity=physics["maximum_depenetration_velocity_m_s"],
        )
        cfg["device_type"] = config["device"]["type"]
        cfg["device_id"] = config["device"]["id"]
        cfg["headless"] = True
        params = gymapi.SimParams()
        params.dt = 1 / rates["physics"]
        gymutil.parse_sim_config(cfg["sim"], params)
        params.use_gpu_pipeline = config["device"]["gpu_pipeline"]
        params.physx.use_gpu = config["device"]["type"] == "cuda"
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])

        class SceneInteraction(InterMimic):
            def _build_termination_heights(self):
                self._termination_heights = torch.tensor(
                    config["termination"]["root_height_m"], device=self.device)

            def _create_ground_plane(self):
                super()._create_ground_plane()
                geometry = gymapi.TriangleMeshParams()
                geometry.nb_vertices = len(room.vertices)
                geometry.nb_triangles = len(room.faces)
                geometry.static_friction = self.plane_static_friction
                geometry.dynamic_friction = self.plane_dynamic_friction
                geometry.restitution = self.plane_restitution
                self.gym.add_triangle_mesh(
                    self.sim, room.vertices.astype(np.float32).ravel(),
                    room.faces.astype(np.uint32).ravel(), geometry)

            def _load_target_asset(self):
                options = gymapi.AssetOptions()
                options.density = self.object_density
                options.override_com = True
                options.override_inertia = True
                options.angular_damping = options.linear_damping = 0.01
                options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                options.vhacd_enabled = True
                options.vhacd_params.max_convex_hulls = 64
                options.vhacd_params.max_num_vertices_per_ch = 64
                options.vhacd_params.resolution = 300000
                asset = self.gym.load_asset(self.sim, str(simulation), "object.urdf", options)
                if asset is None:
                    raise ExecutionError("asset_mismatch", "PhysX could not load the canonical object")
                self._target_asset = [asset]
                self.object_points = torch.tensor(
                    np.load(simulation / "sample_points.npy"),
                    dtype=torch.float32, device=self._init_device)[None]

        environment = SceneInteraction(
            cfg, params, gymapi.SIM_PHYSX, config["device"]["type"],
            config["device"]["id"], True)
        if environment.num_bodies != config["controller"]["body_count"] or environment.num_dof != 153:
            raise ExecutionError("asset_mismatch", "Loaded humanoid does not match converted SMPL-X skeleton")
        category = "controller_mismatch"
        module_path = upstream / "isaaclab/src/intermimic_lab/policy_loader.py"
        spec = importlib.util.spec_from_file_location("upstream_policy_loader", module_path)
        policy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy_module)
        portable_checkpoint = simulation / "checkpoint.pt"
        torch.save(checkpoint_data, portable_checkpoint)
        try:
            policy = policy_module.load_policy(
                str(portable_checkpoint), environment.num_obs, environment.num_actions,
                environment.device, str(upstream / config["controller"]["policy_config"]))
        finally:
            portable_checkpoint.unlink()
        if policy.running_mean_std is None:
            raise ExecutionError("controller_mismatch", "Controller checkpoint lacks observation normalization")
        if environment.num_obs != controller["observation_dimension"] or environment.num_actions != controller["action_dimension"]:
            raise ExecutionError("controller_mismatch", "Environment and controller dimensions disagree")

        category = "simulation_failure"
        environment.reset()
        num_envs = config["num_environments"]
        initial_actors = environment._root_states.detach().cpu().numpy().copy().reshape(num_envs, -1, 13)
        states, bodies, contacts, object_contacts, actions = [], [], [], [], []

        def capture():
            states.append(environment._root_states.detach().cpu().numpy().copy().reshape(num_envs, -1, 13))
            bodies.append(humanoid_body_states(
                environment._rigid_body_state.detach().cpu().numpy().copy(),
                num_envs, environment.num_bodies))
            contacts.append(environment._contact_forces.detach().cpu().numpy().copy().reshape(num_envs, -1, 3))
            object_contacts.append(environment._tar_contact_forces.detach().cpu().numpy().copy().reshape(num_envs, -1, 3))

        frame_count = int(environment.max_episode_length.min().item())
        with torch.no_grad():
            for _ in range(frame_count - 1):
                action = policy.get_action(environment.obs_buf.clamp(-5, 5)).clamp(-1, 1)
                if not torch.isfinite(action).all():
                    raise ExecutionError("simulation_failure", "Non-finite controller action")
                environment.step(action)
                environment.gym.fetch_results(environment.sim, True)
                actions.append(action.cpu().numpy().copy())
                capture()
                if (config["termination"]["stop_on_any_environment"] and
                        environment.reset_buf.any()):
                    break
                if environment.reset_buf.all():
                    break
        states, bodies, contacts, object_contacts, actions = map(
            np.asarray, (states, bodies, contacts, object_contacts, actions))
        reference = torch.load(reference_path, map_location="cpu").numpy()
        validate_reference(reference)
        timestamps = (np.arange(len(states), dtype=np.float64) + 1) / rates["control"]
        np.savez_compressed(
            simulation / "executed.npz", time=timestamps,
            step_index=np.arange(1, len(states) + 1, dtype=np.int64),
            initial_actor_states=initial_actors, actor_states=states,
            body_states=bodies, contact_forces=contacts,
            object_contact_forces=object_contacts, actions=actions)
        terminated = bool(environment._terminate_buf.any())
        tracking_termination = bool(environment.kinematic_reset.any())
        contact_miss_streak = environment.contact_reset.max(dim=0).values.cpu().tolist()
        measurements = measure_execution(
            reference, initial_actors, states, bodies, contacts, object_contacts,
            actions, config, terminated, tracking_termination, contact_miss_streak)
        props = environment.gym.get_actor_rigid_body_properties(
            environment.envs[0], environment._target_handles[0])
        completed = bool(
            measurements["finite"] and measurements["initialization"]["within_tolerance"]
            and measurements["synchronization"]["aligned"]
            and len(states) == frame_count - 1 and not terminated)
        failure_category, evidence = diagnose_result(measurements, completed)
        decision = _write_decision(
            output, failure_category, "completed" if completed else "failed", evidence)
        record.update({
            "intermimic_commit": subprocess.check_output(
                ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
            "steps": len(states), "expected_steps": frame_count - 1,
            "measurements": measurements,
            "object_density_kg_m3": config["object"]["density_kg_m3"],
            "object_mass_kg": float(sum(prop.mass for prop in props)),
            "effective_environment_config": cfg,
            "physics_dt_s": params.dt, "substeps": params.substeps,
            "decision": decision,
            "limitations": [
                "Source floor failure remains unchanged",
                "Object density is an InterMimic student-policy assumption, not a measured object mass",
                "Room collision meshes are static; only the selected object is dynamic",
                "General execution measurements do not define activity-specific success",
            ],
            "completed_reference": completed,
        })
        record["stages"]["physics_executed"] = bool(len(actions) and measurements["finite"])
        (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({
            "steps": record["steps"], "expected_steps": record["expected_steps"],
            "finite": measurements["finite"], "completion_fraction": measurements["completion_fraction"],
            "terminated": terminated, "failure_category": failure_category,
        }, indent=2))
        return record
    except Exception as exc:
        diagnosed = exc.category if isinstance(exc, ExecutionError) else category
        record["error"] = str(exc)
        record["decision"] = _write_decision(output, diagnosed, "failed", [str(exc)])
        (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        raise
    finally:
        if environment is not None:
            environment.gym.destroy_sim(environment.sim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--intermimic", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    result = execute(args.converted, args.output, args.intermimic, args.checkpoint, args.config)
    sys.exit(0 if result["completed_reference"] else 1)
