"""Convert one saved scene-conditioned HOIDiNi run to InterMimic SMPL-X."""

import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

from ..embodiedgen_hoidini.prepare_scene import _load_obj, sample_point_cloud
from ..embodiedgen_hoidini.run_motion import _output_directory
from .contacts import map_contacts
from .coordinates import continuous_quaternions, resample_motion
from .schema import FIELDS, FPS, WIDTH, validate_reference
from .skeleton import map_skeleton, read_skeleton


def convert_arrays(arrays, source_fps, skeleton):
    sampled, times = resample_motion(arrays, source_fps)
    dofs, positions, rotations = map_skeleton(sampled["poses"], sampled["trans"], skeleton)
    human_contact, object_contact = map_contacts(sampled["contact"], skeleton[0])
    fields = {
        "root_pos": sampled["trans"], "root_rot": rotations[:, 0],
        "dof_pos": dofs, "body_pos": positions.reshape(len(times), -1),
        "body_rot": rotations.reshape(len(times), -1),
        "obj_pos": sampled["trans_obj"],
        "obj_rot": continuous_quaternions(sampled["poses_obj"][:, 0]),
        "contact_human": human_contact, "contact_obj": object_contact[:, None],
    }
    data = np.zeros((len(times), WIDTH), dtype=np.float32)
    for name, value in fields.items():
        data[:, FIELDS[name]] = value
    validate_reference(data)
    # This includes both skeleton proportions and HOIDiNi's separate predicted
    # joint-position channels. It is not a claim of contact-preserving retargeting.
    displacement = np.linalg.norm(positions - sampled["joints"][:, skeleton[3]], axis=-1)
    stats = {
        "source_frames": len(arrays["poses"]), "source_fps": source_fps,
        "frames": len(data), "fps": FPS,
        "source_duration_s": len(arrays["poses"]) / source_fps,
        "duration_s": len(data) / FPS,
        "last_sample_time_s": float(times[-1]),
        "maximum_body_position_change_m": float(displacement.max()),
        "mean_body_position_change_m": float(displacement.mean()),
        "minimum_target_joint_height_m": float(positions[..., 2].min()),
        "left_hand_contact_frames": int(human_contact[:, skeleton[0].index("L_Wrist")].sum()),
        "right_hand_contact_frames": int(human_contact[:, skeleton[0].index("R_Wrist")].sum()),
    }
    return data, stats


def convert_run(run, humanoid_xml, output, object_name):
    import torch

    run, humanoid_xml = Path(run).resolve(), Path(humanoid_xml).resolve()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", object_name):
        raise ValueError("Object name must be one alphanumeric token for InterMimic's filename parser")
    receipt = json.loads((run / "input.json").read_text())
    source_validation = json.loads((run / "validation.json").read_text())
    if receipt.get("units") != "metres" or receipt.get("up_axis") != "+Z":
        raise ValueError("Only the existing metre, +Z scene-conditioned run is supported")
    if receipt.get("human_prefix", {}).get("body_model") != "SMPL-X":
        raise ValueError("Expected the existing SMPL-X HOIDiNi run")
    with np.load(run / "motion.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if source_validation.get("frames") != len(arrays["poses"]):
        raise ValueError("Source validation does not match the saved motion length")
    if source_validation.get("fps") != receipt["human_prefix"].get("fps"):
        raise ValueError("Source validation fps disagrees with the inference input")
    skeleton = read_skeleton(humanoid_xml)
    data, stats = convert_arrays(arrays, source_validation["fps"], skeleton)
    mesh_data = receipt["active_interaction"]["canonical_target_mesh"]
    mesh = _load_obj(run / "object.obj")
    recorded_vertices = np.asarray(mesh_data["vertices"])
    if (mesh.vertices.shape != recorded_vertices.shape or
            not np.allclose(mesh.vertices, recorded_vertices, atol=1e-8, rtol=0) or
            not np.array_equal(mesh.faces, mesh_data["faces"])):
        raise ValueError("Object mesh disagrees with the scene-conditioned inference input")
    points = sample_point_cloud(mesh, point_count=1024, seed=0).points.astype(np.float32)

    output = _output_directory(output)
    # The full environment parses int(subject[3:]); 0 denotes this synthetic
    # subject, not a GRAB/OMOMO training subject or a controller selection.
    motion_path = output / f"sub0_{object_name}_000.pt"
    torch.save(torch.from_numpy(data), motion_path)
    loaded = torch.load(motion_path, map_location="cpu", weights_only=True).numpy()
    validate_reference(loaded)
    if not np.array_equal(data, loaded):
        raise ValueError("Saved InterMimic tensor did not round-trip exactly")
    shutil.copyfile(humanoid_xml, output / "humanoid.xml")
    # The scene's canonical object frame stays unchanged, including its origin.
    shutil.copyfile(run / "object.obj", output / "object.obj")
    np.save(output / "sample_points.npy", points, allow_pickle=False)
    manifest = {
        "motion_file": motion_path.name, "object_name": object_name,
        "humanoid_file": "humanoid.xml", "object_mesh": "object.obj",
        "object_points": "sample_points.npy", "units": "metres", "up_axis": "+Z",
        "quaternion_order": "xyzw", "dof_representation": "local exponential map, radians",
        "body_names": skeleton[0], "source_joint_indices": skeleton[3].tolist(),
        "body_local_basis_quaternion_xyzw": [0.5, 0.5, 0.5, 0.5],
        "world_transform": np.eye(4).tolist(),
        "source_run": str(run), "source_scene": receipt["layout_path"],
        "source_motion_sha256": hashlib.sha256((run / "motion.npz").read_bytes()).hexdigest(),
        "humanoid_xml_sha256": hashlib.sha256(humanoid_xml.read_bytes()).hexdigest(),
        "source_motion_validation": source_validation,
        "conversion": stats,
        "conversion_checks": {"finite_valid_tensor": True, "save_load_exact": True},
        "contact_mapping": {
            "threshold": 0.4, "aggregation": "any of 30 anchors per hand, represented at the wrist",
            "labels": "1 desired hand contact; 0 unconstrained; no inferred -1 labels",
            "object_contact": "any predicted hand contact, not object-support contact",
        },
        "limitations": [
            "Source floor failure is retained and accepted for conversion; no collision correction is performed",
            "Joint-angle transfer to the supplied humanoid proportions does not preserve hand-object distances",
            "Finger rotations are retained; contact intent is hand-level, not per-finger or full-body supervision",
            "Reference loading does not demonstrate physics execution; object physics assets and scene registration remain subsequent work",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"motion": str(motion_path), **stats}, indent=2))
    return manifest
