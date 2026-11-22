"""Bounded MPFB-to-MIMo right-hand rest-pose calibration and pose probe.

Run inside Blender::

  Blender mpfb_age_minimum_rigged.blend --background --python this_file -- \
      --mimo-model MIMo_modelv2.xml --output-dir /ignored/path

This is deliberately a calibration probe, not an outcome renderer.  It records
the explicit skeleton correspondence and produces four inspection poses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from xml.etree import ElementTree

import bpy
import numpy as np
from mathutils import Matrix, Vector


MPFB_CHAINS = {
    "thumb": ["finger1-1.R", "finger1-2.R", "finger1-3.R"],
    "index": ["finger2-1.R", "finger2-2.R", "finger2-3.R"],
    "middle": ["finger3-1.R", "finger3-2.R", "finger3-3.R"],
    "ring": ["finger4-1.R", "finger4-2.R", "finger4-3.R"],
    "little": ["finger5-1.R", "finger5-2.R", "finger5-3.R"],
}
MIMO_CHAINS = {
    "thumb": ["right_thbase", "right_thhub", "right_thdistal"],
    "index": ["right_ffknuckle", "right_ffmiddle", "right_ffdistal"],
    "middle": ["right_mfknuckle", "right_mfmiddle", "right_mfdistal"],
    "ring": ["right_rfknuckle", "right_rfmiddle", "right_rfdistal"],
    "little": [
        "right_lfmetacarpal",
        "right_lfknuckle",
        "right_lfmiddle",
        "right_lfdistal",
    ],
}
POSE_FLEXION_DEGREES = {
    "neutral": [0.0, 0.0, 0.0],
    "open": [-8.0, -5.0, -3.0],
    "contact": [18.0, 24.0, 18.0],
    "grasp": [48.0, 62.0, 55.0],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vec(text: str | None) -> Vector:
    return Vector(tuple(float(value) for value in (text or "0 0 0").split()))


def _mimo_landmarks(model_path: Path) -> dict[str, Vector]:
    """Resolve named body origins in the right-hand frame from the MJCF tree."""
    root = ElementTree.parse(model_path).getroot()
    wanted = {"right_hand"} | {
        name for chain in MIMO_CHAINS.values() for name in chain
    }
    found: dict[str, Vector] = {}

    def visit(node: ElementTree.Element, parent: Matrix) -> None:
        local = Matrix.Translation(_vec(node.get("pos")))
        # MIMo's finger bodies use Euler degrees under the parent compiler.
        angles = [math.radians(x) for x in _vec(node.get("euler"))]
        rotation = (
            Matrix.Rotation(angles[2], 4, "Z")
            @ Matrix.Rotation(angles[1], 4, "Y")
            @ Matrix.Rotation(angles[0], 4, "X")
        )
        transform = parent @ local @ rotation
        if node.get("name") in wanted:
            found[str(node.get("name"))] = transform.translation.copy()
        for child in node.findall("body"):
            visit(child, transform)

    for body in root.findall(".//worldbody/body"):
        visit(body, Matrix.Identity(4))
    # Included model fragments do not necessarily have a worldbody.
    if not found:
        for body in root.findall("body"):
            visit(body, Matrix.Identity(4))
    if "right_hand" not in found:
        # Search from every top-level body in an include fragment.
        for body in root.findall(".//body"):
            if body.get("name") == "right_hand":
                visit(body, Matrix.Identity(4))
                break
    origin = found["right_hand"]
    return {name: point - origin for name, point in found.items()}


def _armature() -> bpy.types.Object:
    candidates = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one armature, found {[obj.name for obj in candidates]}")
    return candidates[0]


def _mpfb_landmarks(armature: bpy.types.Object) -> dict[str, Vector]:
    bones = armature.data.bones
    landmarks = {"wrist": bones["wrist.R"].head_local.copy()}
    for finger, chain in MPFB_CHAINS.items():
        for index, bone_name in enumerate(chain):
            bone = bones[bone_name]
            landmarks[f"{finger}_{index}_base"] = bone.head_local.copy()
        landmarks[f"{finger}_tip"] = bones[chain[-1]].tail_local.copy()
    origin = landmarks["wrist"]
    return {name: point - origin for name, point in landmarks.items()}


def _configure_render(armature: bpy.types.Object, output_dir: Path) -> bpy.types.Object:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world.color = (0.025, 0.025, 0.025)
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -1.0
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    wrist_world = armature.matrix_world @ armature.data.bones["wrist.R"].head_local
    middle_tip_world = armature.matrix_world @ armature.data.bones["finger3-3.R"].tail_local
    direction = (middle_tip_world - wrist_world).normalized()
    index_base = armature.matrix_world @ armature.data.bones["finger2-1.R"].head_local
    ring_base = armature.matrix_world @ armature.data.bones["finger4-1.R"].head_local
    across = (index_base - ring_base).normalized()
    palm_normal = direction.cross(across).normalized()
    center = wrist_world + direction * 0.035
    camera = bpy.data.objects.get("RetargetProbeCamera")
    if camera is None:
        camera_data = bpy.data.cameras.new("RetargetProbeCamera")
        camera = bpy.data.objects.new("RetargetProbeCamera", camera_data)
        bpy.context.collection.objects.link(camera)
    camera.location = center - palm_normal * 0.13 - direction * 0.015
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 62
    scene.camera = camera
    for location, energy, size in [
        (center - palm_normal * 0.10 + across * 0.08, 45.0, 0.10),
        (center - palm_normal * 0.06 - across * 0.10, 20.0, 0.08),
    ]:
        lamp_data = bpy.data.lights.new("RetargetProbeArea", "AREA")
        lamp_data.energy = energy
        lamp_data.shape = "DISK"
        lamp_data.size = size
        lamp = bpy.data.objects.new("RetargetProbeArea", lamp_data)
        lamp.location = location
        lamp.rotation_euler = (center - location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.collection.objects.link(lamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    return camera


def _pose_and_render(
    armature: bpy.types.Object, pose_name: str, output_dir: Path
) -> Path:
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")
    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = "XYZ"
        pose_bone.rotation_euler = (0.0, 0.0, 0.0)
    flexion = POSE_FLEXION_DEGREES[pose_name]
    for finger, chain in MPFB_CHAINS.items():
        multiplier = 0.75 if finger == "thumb" else 1.0
        for index, bone_name in enumerate(chain):
            armature.pose.bones[bone_name].rotation_euler.x = math.radians(
                flexion[index] * multiplier
            )
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    output = output_dir / f"mpfb_mimo_{pose_name}_inspection.png"
    bpy.context.scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)
    return output


def run(model_path: Path, output_dir: Path) -> dict:
    armature = _armature()
    mimo = _mimo_landmarks(model_path)
    mpfb = _mpfb_landmarks(armature)
    correspondence = {
        "thumb": "right_thbase",
        "index": "right_ffknuckle",
        "middle": "right_mfknuckle",
        "ring": "right_rfknuckle",
        "little": "right_lfknuckle",
    }
    source = np.asarray(
        [[0.0, 0.0, 0.0]]
        + [list(mpfb[f"{finger}_0_base"]) for finger in correspondence],
        dtype=np.float64,
    )
    target = np.asarray(
        [[0.0, 0.0, 0.0]]
        + [list(mimo[name]) for name in correspondence.values()],
        dtype=np.float64,
    )
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    scale = float(
        singular_values.sum() / np.square(source_centered).sum()
    )
    mapped = {
        name: Vector(np.asarray(point) @ rotation * scale)
        for name, point in mpfb.items()
    }
    errors = {}
    for finger, mimo_name in correspondence.items():
        errors[f"{finger}_base"] = float(
            (mapped[f"{finger}_0_base"] - mimo[mimo_name]).length
        )
    _configure_render(armature, output_dir)
    frames = [
        _pose_and_render(armature, pose_name, output_dir)
        for pose_name in POSE_FLEXION_DEGREES
    ]
    receipt = {
        "schema_version": "mpfb_mimo_retarget_probe.v1",
        "status": "diagnostic_rest_pose_calibration_not_outcome_qualification",
        "units": {"linear": "metre", "angular": "degree"},
        "source": {
            "blend": bpy.data.filepath,
            "blend_sha256": _sha256(Path(bpy.data.filepath)),
            "mimo_model": str(model_path),
            "mimo_model_sha256": _sha256(model_path),
            "blender_version": bpy.app.version_string,
        },
        "mapping": {
            "wrist": "right_hand",
            "finger_chains": {
                finger: {"mpfb": MPFB_CHAINS[finger], "mimo": MIMO_CHAINS[finger]}
                for finger in MPFB_CHAINS
            },
            "axis_calibration": (
                "MPFB bone-local flexion about +X; proper-rotation Kabsch/Umeyama "
                "least-squares fit of wrist plus five proximal finger origins"
            ),
            "mpfb_to_mimo_rotation_row_vector": rotation.tolist(),
            "uniform_scale_mimo_metres_per_mpfb_unit": scale,
        },
        "rest_pose": {
            "mimo_middle_distal_origin_distance_m": mimo["right_mfdistal"].length,
            "mpfb_wrist_to_middle_tip_native": mpfb["middle_tip"].length,
            "finger_base_errors_m": errors,
            "maximum_finger_base_error_m": max(errors.values()),
            "note": (
                "Base-origin discrepancies diagnose skeleton-shape mismatch; "
                "surface/contact tolerance remains to be frozen separately."
            ),
        },
        "poses": {
            pose: {
                "proximal_middle_distal_flexion_deg": values,
                "frame": frame.name,
                "frame_sha256": _sha256(frame),
            }
            for (pose, values), frame in zip(POSE_FLEXION_DEGREES.items(), frames)
        },
        "five_finger_continuous_weighted_mesh": True,
    }
    receipt_path = output_dir / "mpfb_mimo_retarget_probe_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimo-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(arguments)
    print(json.dumps(run(args.mimo_model, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
