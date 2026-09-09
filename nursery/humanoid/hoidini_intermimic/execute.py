"""Execute one converted interaction with the upstream InterMimic teacher.

No reference replay: after initialization only controller actions move the
human, and the object moves through PhysX. Room meshes remain static.
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
                raise ValueError(f"Scene node has no collision geometry: {name}")
            continue  # Empty structural URDF link.
        pose = Pose(tuple(node["pose"][:3]), tuple(node["pose"][3:])).matrix
        for geometry in geometries:
            mesh = _load_obj(Path(geometry["mesh_path"]))
            meshes.append(mesh.transformed(pose @ np.asarray(geometry["mesh_to_asset"])))
            sources.append({"node": name, "mesh": geometry["mesh_path"]})
    if not meshes:
        raise ValueError("No room collision geometry")
    return _combine_meshes(meshes), sources


def prepare_execution(converted, output):
    converted = Path(converted).resolve()
    manifest = json.loads((converted / "manifest.json").read_text())
    receipt = json.loads((Path(manifest["source_run"]) / "input.json").read_text())
    if manifest["units"] != "metres" or manifest["up_axis"] != "+Z":
        raise ValueError("Expected metre, +Z converted motion")
    mesh = _load_obj(converted / manifest["object_mesh"])
    original = receipt["active_interaction"]["canonical_target_mesh"]
    if not np.array_equal(mesh.faces, original["faces"]) or not np.allclose(
            mesh.vertices, original["vertices"], atol=1e-8, rtol=0):
        raise ValueError("Converted object differs from the motion-generation object")
    room, sources = room_collision_mesh(receipt)
    output = _output_directory(output)
    simulation = output / "simulation"
    simulation.mkdir()
    motions = simulation / "motion"
    motions.mkdir()
    # Accept existing converter runs with the old nursery_ prefix, too.
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
    # No fabricated measured mass: PhysX computes inertia from the configured
    # upstream density, and the resulting mass is recorded after asset loading.
    ET.ElementTree(robot).write(simulation / "object.urdf", encoding="unicode")
    record = {
        "prompt": receipt["active_interaction"]["prompt"], "seed": 0,
        "converted_run": str(converted), "source_scene": manifest["source_scene"],
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "source_motion_validation": manifest["source_motion_validation"],
        "room_collision_sources": sources,
        "stages": {"prepared": True, "physics_executed": False},
        "video": None,
    }
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return output, simulation, room, record


def execute(converted, output, upstream, checkpoint):
    # Isaac Gym must be imported before Torch; use its isolated Python 3.8 env.
    from isaacgym import gymapi, gymutil
    import torch
    import yaml

    upstream, checkpoint = Path(upstream).resolve(), Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(upstream / "isaacgym/src"))
    from intermimic.env.tasks.intermimic import InterMimic

    output, simulation, room, record = prepare_execution(converted, output)
    config_path = upstream / "isaacgym/src/intermimic/data/cfg/omomo_train_new.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    cfg["env"].update(numEnvs=1, dataSub=["sub0"], motion_file=str(simulation / "motion"),
                      robotType="humanoid.xml", stateInit="Start", physicalBufferSize=1,
                      enableEvaluation=False, playdataset=False, saveImages=False)
    cfg["env"]["asset"]["assetRoot"] = str(simulation)
    params = gymapi.SimParams()
    params.dt = 1 / 60
    gymutil.parse_sim_config(cfg["sim"], params)
    params.use_gpu_pipeline = True
    params.physx.use_gpu = True
    np.random.seed(0)
    torch.manual_seed(0)

    class SceneInteraction(InterMimic):
        # These two native extension points keep asset registration out of the
        # controller, reference loader, reward, and physics stepping code.
        def _create_ground_plane(self):
            super()._create_ground_plane()
            geometry = gymapi.TriangleMeshParams()
            geometry.nb_vertices = len(room.vertices)
            geometry.nb_triangles = len(room.faces)
            geometry.static_friction = self.plane_static_friction
            geometry.dynamic_friction = self.plane_dynamic_friction
            geometry.restitution = self.plane_restitution
            self.gym.add_triangle_mesh(self.sim, room.vertices.astype(np.float32).ravel(),
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
                raise RuntimeError("PhysX could not load the canonical object")
            self._target_asset = [asset]
            # Upstream normally subtracts the mesh mean. Our reference is
            # already in its canonical object frame: preserve it exactly.
            self.object_points = torch.tensor(np.load(simulation / "sample_points.npy"),
                                               dtype=torch.float32, device=self._init_device)[None]

    environment = None
    try:
        environment = SceneInteraction(cfg, params, gymapi.SIM_PHYSX, "cuda", 0, True)
        if environment.num_bodies != 52 or environment.num_dof != 153:
            raise ValueError("Loaded humanoid does not match converted SMPL-X skeleton")
        module_path = upstream / "isaaclab/src/intermimic_lab/policy_loader.py"
        spec = importlib.util.spec_from_file_location("upstream_policy_loader", module_path)
        policy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy_module)
        # The upstream wrapper loads serialized CUDA device IDs directly.
        # Stage the same weights on CPU for portability to Slurm/MIG devices;
        # the native wrapper still builds, normalizes, and runs the policy.
        portable_checkpoint = simulation / "checkpoint.pt"
        torch.save(torch.load(checkpoint, map_location="cpu"), portable_checkpoint)
        try:
            policy = policy_module.load_policy(str(portable_checkpoint), environment.num_obs,
                                                environment.num_actions, environment.device,
                                                str(upstream / "isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml"))
        finally:
            portable_checkpoint.unlink()
        if policy.running_mean_std is None:
            raise ValueError("Controller checkpoint lacks observation normalization")
        environment.reset()
        initial_actors = environment._root_states.detach().cpu().numpy().copy()
        states, bodies, contacts, object_contacts, actions = [], [], [], [], []

        def capture():
            states.append(environment._root_states.detach().cpu().numpy().copy())
            bodies.append(environment._rigid_body_state.detach().cpu().numpy().copy())
            contacts.append(environment._contact_forces.detach().cpu().numpy().copy())
            object_contacts.append(environment._tar_contact_forces.detach().cpu().numpy().copy())

        # Rigid-body tensors immediately after reset may still describe the
        # pre-reset scene. Record only states returned after an actual step.
        frame_count = int(environment.max_episode_length[0])
        with torch.no_grad():
            for _ in range(frame_count - 1):
                # Match the native VecTaskPythonWrapper's pre-normalization clip.
                action = policy.get_action(environment.obs_buf.clamp(-5, 5)).clamp(-1, 1)
                if not torch.isfinite(action).all():
                    raise ValueError("Non-finite controller action")
                environment.step(action)
                environment.gym.fetch_results(environment.sim, True)
                actions.append(action.cpu().numpy().copy())
                capture()
                if environment.reset_buf.any():
                    break  # Record the first attempt; never reset and stitch.
        states, bodies, contacts = map(np.asarray, (states, bodies, contacts))
        np.savez_compressed(simulation / "executed.npz", time=(np.arange(len(states)) + 1) / 30,
                            initial_actor_states=initial_actors,
                            actor_states=states, body_states=bodies, contact_forces=contacts,
                            object_contact_forces=np.asarray(object_contacts), actions=np.asarray(actions))
        finite = all(np.isfinite(x).all() for x in (states, bodies, contacts, object_contacts))
        props = environment.gym.get_actor_rigid_body_properties(
            environment.envs[0], environment._target_handles[0])
        record.update({
            "intermimic_commit": subprocess.check_output(
                ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "steps": len(states), "expected_steps": frame_count - 1, "fps": 30,
            "finite": bool(finite), "terminated": bool(environment._terminate_buf.any()),
            "tracking_termination": bool(environment.kinematic_reset.any()),
            "hand_contact_miss_streak": environment.contact_reset[0].cpu().tolist(),
            "maximum_object_lift_m": max(0.0, float((states[:, 1, 2] - initial_actors[1, 2]).max())),
            "object_density_kg_m3": cfg["env"]["objectDensity"],
            "object_mass_kg": float(sum(p.mass for p in props)),
            "settings": cfg, "physics_dt_s": params.dt, "substeps": params.substeps,
            "limitations": ["Source floor failure remains unchanged",
                            "Object density is an upstream default, not a measured mug mass",
                            "Room collision meshes are static; only the selected mug is dynamic",
                            "Controller execution alone does not establish grasp success; no video exported"],
        })
        record["stages"]["physics_executed"] = bool(len(actions) and finite)
        record["completed_reference"] = bool(finite and len(states) == frame_count - 1 and not environment._terminate_buf.any())
        (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: record[k] for k in ("steps", "expected_steps", "finite",
                                               "terminated", "maximum_object_lift_m")}, indent=2))
        return record
    except Exception as exc:
        record["error"] = str(exc)
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
    args = parser.parse_args()
    result = execute(args.converted, args.output, args.intermimic, args.checkpoint)
    sys.exit(0 if result["completed_reference"] else 1)
