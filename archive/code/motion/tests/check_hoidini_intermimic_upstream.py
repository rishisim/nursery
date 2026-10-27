"""Isolated upstream loader/FK check; no simulator or policy is instantiated.

Run in the existing HOIDiNi Torch environment, supplying external upstream
checkouts and the ignored converted output directory. AST extraction executes
the actual loader body while avoiding imports that start/require Isaac Gym.
Two SDK tensor primitives (tensor creation and angle wrapping) are bound below;
quaternion math, SDF, loading, slicing, velocities, and packing use upstream code.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F


def extract_functions(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names), f"Upstream functions changed: {path}"
    for node in selected:
        node.decorator_list = []
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def check(output, intermimic, interact):
    output, intermimic, interact = map(lambda p: Path(p).resolve(), (output, intermimic, interact))
    manifest = json.loads((output / "manifest.json").read_text())
    data = torch.load(output / manifest["motion_file"], map_location="cpu", weights_only=True)
    xml = output / "humanoid.xml"

    # Use the same PoseLib implementation that the upstream conversion invokes.
    sys.path.insert(0, str(interact / "simulation"))
    from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
    tree = SkeletonTree.from_mjcf(str(xml))
    assert tree.node_names == manifest["body_names"]
    local = np.concatenate([
        data[:, 3:7].numpy()[:, None],
        Rotation.from_rotvec(data[:, 9:162].numpy().reshape(-1, 3)).as_quat().reshape(len(data), 51, 4),
    ], axis=1)
    state = SkeletonState.from_rotation_and_root_translation(
        tree, torch.tensor(local, dtype=torch.float32), data[:, :3], is_local=True)
    position_error = float((state.global_translation - data[:, 162:318].reshape(-1, 52, 3)).abs().max())
    orientation_error = float((1 - (state.global_rotation * data[:, 383:591].reshape(-1, 52, 4)).sum(-1).abs()).abs().max())
    assert position_error < 2e-5 and orientation_error < 2e-6

    # Actual upstream pure-tensor quaternion implementation, in xyzw order.
    math_path = intermimic / "isaaclab/src/intermimic_lab/torch_utils_gym.py"
    spec = importlib.util.spec_from_file_location("upstream_quaternion_math", math_path)
    math_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math_module)
    math_namespace = dict(vars(math_module))
    math_namespace["normalize_angle"] = lambda angle: torch.atan2(torch.sin(angle), torch.cos(angle))
    native_utils = intermimic / "isaacgym/src/intermimic/utils/torch_utils.py"
    extract_functions(native_utils, ["quat_to_angle_axis", "angle_axis_to_exp_map", "quat_to_exp_map"], math_namespace)
    native_path = intermimic / "isaacgym/src/intermimic/env/tasks/intermimic.py"
    namespace = {
        "torch": torch, "F": F, "np": np,
        "torch_utils": SimpleNamespace(**math_namespace),
        "quat_rotate": math_module.quat_rotate,
        "to_torch": lambda values, dtype=torch.float32, device="cpu": torch.tensor(values, dtype=dtype, device=device),
    }
    extract_functions(native_path, ["compute_sdf"], namespace)
    extract_functions(native_path, ["_load_motion", "create_component_stat"], namespace, "InterMimic")
    holder = SimpleNamespace(
        object_points=torch.tensor(np.load(output / "sample_points.npy"))[None],
        object_id=torch.tensor([0]), init_vel=False, ref_hoi_obs_size=1211,
        device="cpu", num_envs=1, enable_evaluation=False,
    )
    holder.create_component_stat = lambda loaded: namespace["create_component_stat"](holder, loaded)
    loaded = namespace["_load_motion"](holder, [str(output / manifest["motion_file"])])
    assert loaded.shape == (1, len(data), 1211) and torch.isfinite(loaded).all()
    assert holder.hoi_refs.shape == (1, 1, len(data), 332)
    assert holder.fps_data == manifest["conversion"]["fps"] == 30
    for name, source_slice in [("root_pos", slice(0, 3)), ("root_rot", slice(3, 7)),
                               ("dof_pos", slice(9, 162)), ("obj_pos", slice(318, 321)),
                               ("obj_rot", slice(321, 325)), ("contact_human", slice(331, 383)),
                               ("contact_obj", slice(330, 331))]:
        index = holder.data_component_order.index(name)
        actual = loaded[0, :, holder.data_component_index[index]:holder.data_component_index[index + 1]]
        assert torch.equal(actual, data[:, source_slice]), name

    def commit(root):
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()

    report = {
        "passed": True, "intermimic_commit": commit(intermimic), "interact_commit": commit(interact),
        "motion_sha256": hashlib.sha256((output / manifest["motion_file"]).read_bytes()).hexdigest(),
        "raw_shape": list(data.shape), "upstream_loaded_shape": list(loaded.shape),
        "upstream_reference_shape": list(holder.hoi_refs.shape),
        "upstream_fk_max_position_error_m": position_error,
        "upstream_fk_max_quaternion_dot_error": orientation_error,
        "source_floor_check_passed": manifest["source_motion_validation"]["checks"]["body_not_below_floor"],
        "scope": "Actual upstream _load_motion and PoseLib FK, isolated from simulator construction",
        "limitations": ["No Isaac Gym/Isaac Lab asset load, physics step, or controller execution",
                        "SDK tensor creation and angle wrapping use equivalent Torch primitives"],
    }
    subprocess.run(["git", "-C", str(output), "check-ignore", "--quiet", str(output)], check=True)
    (output / "upstream_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--intermimic", required=True)
    parser.add_argument("--interact", required=True)
    args = parser.parse_args()
    check(args.output, args.intermimic, args.interact)
