#!/usr/bin/env python3
"""Generate a mesh-backed EmbodiedGen room and compose preserved SAM3D assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import types
from typing import Any


EMBODIEDGEN_COMMIT = "9b333554254af196bace88c1a171a3bf047fa09c"
PANO_MODEL_REVISION = "d20770f5ec0000a2aaa579a1209d11dc92ddb08a"
BIG_LAMA_REVISION = "05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9"
SD2_INPAINT_REVISION = "5f74973cbb64c8568780732c17f43eb269d63a0d"
REALESRGAN_REVISION = "a64fcdebeea17287d830736cd0853df1093b97ab"
OMNIDATA_COMMIT = "152cf1465313b68cdf5d47bb09568a9357c57086"
OMNIDATA_DEPTH_REVISION = "3df69f18233d1ffd6161117ee816977d54630d70"
OMNIDATA_NORMAL_REVISION = "387d2be61a3f475c0a5ac969768a9c6d0e75dee1"
TINY_CUDA_NN_COMMIT = "749dd70c5afc5a9dadb85e5652ed65d55e0ba187"
NATIVE_PROFILE = "embodiedgen-v2.0.1-sapien-rh-zup-m-xyzw"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run EmbodiedGen panorama-to-mesh room generation and compose it "
            "with an existing accepted SAM3D object scene."
        )
    )
    parser.add_argument("--candidate-scene", type=Path, required=True)
    parser.add_argument("--generation-work-dir", type=Path, required=True)
    parser.add_argument("--embodiedgen-source", type=Path, required=True)
    parser.add_argument("--pano-model", type=Path, required=True)
    parser.add_argument("--big-lama-zip", type=Path, required=True)
    parser.add_argument("--sd2-inpaint-model", type=Path, required=True)
    parser.add_argument("--realesrgan-model", type=Path, required=True)
    parser.add_argument("--omnidata-source", type=Path, required=True)
    parser.add_argument("--omnidata-depth-model", type=Path, required=True)
    parser.add_argument("--omnidata-normal-model", type=Path, required=True)
    parser.add_argument("--reuse-pano", type=Path)
    parser.add_argument("--reuse-pano-sha256")
    parser.add_argument("--reuse-pano-source-job-id")
    parser.add_argument("--room-prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--job-id", required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise RuntimeError(f"inventory refuses symlink file: {path}")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _patch_pinned_model_loaders(args: argparse.Namespace) -> None:
    import torch
    import timm

    original_timm_create_model = timm.create_model

    def task_checkpoint_backbone(model_name, *positional, **kwargs):
        if model_name in {
            "vit_base_resnet50_384",
            "vit_base_r50_s16_384.orig_in21k_ft_in1k",
        }:
            kwargs["pretrained"] = False
        return original_timm_create_model(model_name, *positional, **kwargs)

    timm.create_model = task_checkpoint_backbone

    task_weights = {
        "https://huggingface.co/sashasax/omnidata_depth_dpt_hybrid_384/resolve/main/omnidata_depth_dpt_hybrid.pth": args.omnidata_depth_model,
        "https://huggingface.co/sashasax/omnidata_normal_dpt_hybrid_384/resolve/main/omnidata_normal_dpt_hybrid.pth": args.omnidata_normal_model,
    }
    original_state_dict_from_url = torch.hub.load_state_dict_from_url

    def pinned_state_dict_from_url(url, *positional, **kwargs):
        local_path = task_weights.get(url)
        if local_path is None:
            return original_state_dict_from_url(url, *positional, **kwargs)
        map_location = kwargs.get("map_location", "cpu")
        return torch.load(local_path, map_location=map_location, weights_only=True)

    torch.hub.load_state_dict_from_url = pinned_state_dict_from_url

    original_hub_load = torch.hub.load
    omnidata_source = str(args.omnidata_source)

    def pinned_hub_load(repo_or_dir, model, *positional, **kwargs):
        if repo_or_dir == "alexsax/omnidata_models":
            kwargs.pop("source", None)
            return original_hub_load(
                omnidata_source, model, *positional, source="local", **kwargs
            )
        return original_hub_load(repo_or_dir, model, *positional, **kwargs)

    torch.hub.load = pinned_hub_load

    from diffusers import StableDiffusionInpaintPipeline

    original_from_pretrained = StableDiffusionInpaintPipeline.from_pretrained
    sd2_path = str(args.sd2_inpaint_model)

    def pinned_from_pretrained(model_name, *positional, **kwargs):
        if model_name == "sd2-community/stable-diffusion-2-inpainting":
            model_name = sd2_path
        return original_from_pretrained(model_name, *positional, **kwargs)

    StableDiffusionInpaintPipeline.from_pretrained = staticmethod(
        pinned_from_pretrained
    )

    import embodied_gen.utils.monkey_patch.pano2room as pano_patch

    original_hf_download = pano_patch.hf_hub_download
    big_lama_zip = str(args.big_lama_zip)

    def pinned_hf_download(*, repo_id, filename, **kwargs):
        if repo_id == "smartywu/big-lama" and filename == "big-lama.zip":
            return big_lama_zip
        return original_hf_download(repo_id=repo_id, filename=filename, **kwargs)

    pano_patch.hf_hub_download = pinned_hf_download


def _install_torchvision_functional_tensor_compat() -> None:
    """Provide BasicSR's removed TorchVision import without patching site-packages."""
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    from torchvision.transforms import functional

    module = types.ModuleType("torchvision.transforms.functional_tensor")
    module.rgb_to_grayscale = functional.rgb_to_grayscale
    sys.modules[module.__name__] = module


def _install_equilib_argument_order_patch(trainer: Any) -> None:
    """Align EmbodiedGen's height/width call with pyequilib 0.3.0."""
    original_cube2equi = trainer.cube2equi

    def corrected_cube2equi(cubemap, cube_format, pano_h, pano_w, *positional, **kwargs):
        return original_cube2equi(
            cubemap, cube_format, pano_w, pano_h, *positional, **kwargs
        )

    trainer.cube2equi = corrected_cube2equi


def _generate_room(
    args: argparse.Namespace,
) -> tuple[Path, Path, dict[str, Any]]:
    _install_torchvision_functional_tensor_compat()
    _patch_pinned_model_loaders(args)
    import torch
    from embodied_gen.models.sr_model import ImageRealESRGAN
    from embodied_gen.trainer import pono2mesh_trainer as trainer
    from embodied_gen.utils.config import Pano2MeshSRConfig

    real_esrgan_path = str(args.realesrgan_model)

    def pinned_realesrgan(outscale: int):
        return ImageRealESRGAN(outscale=outscale, model_path=real_esrgan_path)

    trainer.ImageRealESRGAN = pinned_realesrgan
    _install_equilib_argument_order_patch(trainer)
    work = args.generation_work_dir
    if work.exists() and any(work.iterdir()):
        raise RuntimeError("generation work directory must be new or empty")
    work.mkdir(parents=True, exist_ok=True)
    pano_path = work / "pano_image.png"
    (work / "prompt.txt").write_text(args.room_prompt + "\n", encoding="utf-8")

    expanded_prompt = (
        f"{args.room_prompt}, spacious, empty, wide open, open floor, minimal furniture"
    )
    if args.reuse_pano is not None:
        actual_sha256 = _sha256(args.reuse_pano)
        if actual_sha256 != args.reuse_pano_sha256:
            raise RuntimeError("reused panorama SHA-256 mismatch")
        shutil.copy2(args.reuse_pano, pano_path)
        panorama_stage = {
            "status": "reused",
            "source_job_id": args.reuse_pano_source_job_id,
            "sha256": actual_sha256,
            "expanded_prompt": expanded_prompt,
            "seed": args.seed,
            "inference_steps": 40,
        }
    else:
        from txt2panoimg import Text2360PanoramaImagePipeline

        panorama = Text2360PanoramaImagePipeline(
            str(args.pano_model), torch_dtype=torch.float16, device="cuda"
        )
        image = panorama(
            {
                "prompt": expanded_prompt,
                "num_inference_steps": 40,
                "upscale": False,
                "seed": args.seed,
            }
        )
        image.save(pano_path)
        panorama_stage = {
            "status": "generated",
            "source_job_id": args.job_id,
            "sha256": _sha256(pano_path),
            "expanded_prompt": expanded_prompt,
            "seed": args.seed,
            "inference_steps": 40,
        }

    config = Pano2MeshSRConfig()
    config.trajectory_dir = str(
        args.embodiedgen_source / "apps" / "assets" / "example_scene" / "camera_trajectory"
    )
    pipeline = trainer.Pano2MeshSRPipeline(config)
    pipeline(str(pano_path), str(work))
    raw_mesh = work / config.mesh_file
    if not raw_mesh.is_file() or raw_mesh.stat().st_size == 0:
        raise RuntimeError("EmbodiedGen Pano2Mesh did not produce mesh_model.ply")
    torch.set_default_device("cpu")
    return pano_path, raw_mesh, panorama_stage


def _canonicalize_room_mesh(raw_mesh: Path, output_mesh: Path) -> dict[str, Any]:
    import numpy as np
    import trimesh

    loaded = trimesh.load(raw_mesh, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    mesh = loaded.copy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise RuntimeError("generated room mesh has invalid vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 2:
        raise RuntimeError("generated room mesh has invalid triangular faces")
    if not np.isfinite(vertices).all():
        raise RuntimeError("generated room mesh contains non-finite vertices")

    # EmbodiedGen Pano2Mesh is Y-up. Bake a right-handed Y-up -> Z-up transform:
    # world_x=source_x, world_y=-source_z, world_z=source_y.
    world = np.column_stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1]))
    minimum = world.min(axis=0)
    maximum = world.max(axis=0)
    source_height = maximum[2] - minimum[2]
    if not math.isfinite(source_height) or source_height <= 0.0:
        raise RuntimeError("generated room mesh has zero height")
    scale = 3.0 / source_height
    world *= scale
    minimum = world.min(axis=0)
    maximum = world.max(axis=0)
    world[:, 0] -= (minimum[0] + maximum[0]) / 2.0
    world[:, 1] -= (minimum[1] + maximum[1]) / 2.0
    world[:, 2] -= minimum[2]
    mesh.vertices = world
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    if min(extents[:2]) < 2.5 or max(extents[:2]) > 20.0:
        raise RuntimeError(
            f"generated room horizontal scale is implausible after metric alignment: {extents.tolist()}"
        )
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_mesh, file_type="ply")
    return {
        "source_coordinate_system": "right_handed_Y_up",
        "target_coordinate_system": "right_handed_Z_up",
        "baked_axis_map": ["world_x=source_x", "world_y=-source_z", "world_z=source_y"],
        "uniform_scale": float(scale),
        "centering": "horizontal_AABB_center_to_world_origin",
        "floor_alignment": "AABB_min_z_to_world_z_0",
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "aabb_min_m": [float(value) for value in bounds[0]],
        "aabb_max_m": [float(value) for value in bounds[1]],
        "extents_m": [float(value) for value in extents],
    }


def _box_vertices_faces(
    minimum: tuple[float, float, float], maximum: tuple[float, float, float], offset: int
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    vertices = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    local_faces = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ]
    return vertices, [tuple(index + offset for index in face) for face in local_faces]


def _write_wall_collision(path: Path, bounds: dict[str, Any]) -> None:
    minimum = bounds["aabb_min_m"]
    maximum = bounds["aabb_max_m"]
    thickness = 0.05
    z0, z1 = 0.0, maximum[2]
    boxes = [
        ((minimum[0] - thickness, minimum[1] - thickness, z0), (minimum[0], maximum[1] + thickness, z1)),
        ((maximum[0], minimum[1] - thickness, z0), (maximum[0] + thickness, maximum[1] + thickness, z1)),
        ((minimum[0], minimum[1] - thickness, z0), (maximum[0], minimum[1], z1)),
        ((minimum[0], maximum[1], z0), (maximum[0], maximum[1] + thickness, z1)),
    ]
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for minimum_box, maximum_box in boxes:
        box_vertices, box_faces = _box_vertices_faces(
            minimum_box, maximum_box, len(vertices) + 1
        )
        vertices.extend(box_vertices)
        faces.extend(box_faces)
    lines = ["# Explicit Nursery room wall collision; world-frame meters."]
    lines.extend("v " + " ".join(f"{value:.9g}" for value in vertex) for vertex in vertices)
    lines.extend("f " + " ".join(str(index) for index in face) for face in faces)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_preserved_activity_binding(
    activity: dict[str, Any], original_receipt: dict[str, Any]
) -> None:
    receipt_activity = original_receipt.get("activity")
    if not isinstance(receipt_activity, dict):
        raise RuntimeError("accepted generation receipt lacks its activity binding")
    if receipt_activity.get("activity_id") != activity.get("activity_id"):
        raise RuntimeError("accepted generation receipt activity ID changed")
    if receipt_activity.get("prompt") != activity.get("prompt"):
        raise RuntimeError("accepted generation receipt prompt changed")
    seeds = activity.get("seeds")
    if not isinstance(seeds, dict):
        raise RuntimeError("ActivitySpec lacks its frozen seeds")
    expected_embodiedgen_seeds = {
        "image": seeds.get("embodiedgen_image"),
        "asset": seeds.get("embodiedgen_asset"),
        "layout": seeds.get("embodiedgen_layout"),
    }
    if original_receipt.get("seeds") != expected_embodiedgen_seeds:
        raise RuntimeError("accepted generation receipt EmbodiedGen seeds changed")


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate = args.candidate_scene.resolve(strict=True)
    source_root = args.embodiedgen_source.resolve(strict=True)
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if not args.room_prompt.strip():
        raise ValueError("--room-prompt must be non-empty")
    if not (candidate / "layout.json").is_file():
        raise ValueError("candidate scene lacks layout.json")
    for path in (
        args.pano_model,
        args.sd2_inpaint_model,
        args.omnidata_source,
    ):
        if not path.resolve(strict=True).is_dir():
            raise ValueError(f"required pinned model/source directory is missing: {path}")
    for path in (args.big_lama_zip, args.realesrgan_model):
        if not path.resolve(strict=True).is_file():
            raise ValueError(f"required pinned model file is missing: {path}")
    for path in (args.omnidata_depth_model, args.omnidata_normal_model):
        if not path.resolve(strict=True).is_file():
            raise ValueError(f"required pinned Omnidata task model is missing: {path}")
    reuse_values = (
        args.reuse_pano,
        args.reuse_pano_sha256,
        args.reuse_pano_source_job_id,
    )
    if any(value is not None for value in reuse_values) and not all(
        value is not None for value in reuse_values
    ):
        raise ValueError("panorama reuse requires path, SHA-256, and source job ID")
    if args.reuse_pano is not None and not args.reuse_pano.resolve(strict=True).is_file():
        raise ValueError("reused panorama is missing")

    activity_path = candidate / "metadata" / "activity.json"
    request_path = candidate / "metadata" / "embodiedgen_request.json"
    protocol_path = candidate / "metadata" / "protocol.json"
    original_receipt_path = candidate / "generation_receipt.json"
    activity = json.loads(activity_path.read_text(encoding="utf-8"))
    original_receipt = json.loads(original_receipt_path.read_text(encoding="utf-8"))
    activity_prompt = activity["prompt"]
    seeds = activity["seeds"]
    _validate_preserved_activity_binding(activity, original_receipt)
    if args.seed != seeds["embodiedgen_image"]:
        raise RuntimeError("room generation must reuse the frozen EmbodiedGen image seed")

    asset_root = candidate / "asset3d"
    preserved_inventory = _inventory(asset_root)
    preserved_inventory_sha = _json_sha256(preserved_inventory)
    target_node = "red mug"
    target_instance_id = "egv2_" + hashlib.sha256(target_node.encode()).hexdigest()[:24]

    pano_path, raw_mesh_path, panorama_stage = _generate_room(args)
    background = candidate / "background"
    if background.exists():
        shutil.rmtree(background)
    provenance_root = background / "provenance"
    provenance_root.mkdir(parents=True)
    retained_pano = provenance_root / "pano_image.png"
    retained_raw_mesh = provenance_root / "raw_mesh_model.ply"
    shutil.copy2(pano_path, retained_pano)
    shutil.copy2(raw_mesh_path, retained_raw_mesh)
    (provenance_root / "prompt.txt").write_text(args.room_prompt + "\n", encoding="utf-8")
    final_mesh = background / "mesh_model.ply"
    transform_receipt = _canonicalize_room_mesh(retained_raw_mesh, final_mesh)
    walls_path = background / "collision" / "walls.obj"
    _write_wall_collision(walls_path, transform_receipt)

    source_inventory = {
        "schema": "InteractMoveRoomSourceInventory",
        "schema_version": 1,
        "job_id": args.job_id,
        "prompt": args.room_prompt,
        "activity_prompt": activity_prompt,
        "seed": args.seed,
        "pins": {
            "embodiedgen_commit": EMBODIEDGEN_COMMIT,
            "pano_model_revision": PANO_MODEL_REVISION,
            "big_lama_revision": BIG_LAMA_REVISION,
            "sd2_inpaint_revision": SD2_INPAINT_REVISION,
            "realesrgan_revision": REALESRGAN_REVISION,
            "omnidata_commit": OMNIDATA_COMMIT,
            "omnidata_depth_revision": OMNIDATA_DEPTH_REVISION,
            "omnidata_normal_revision": OMNIDATA_NORMAL_REVISION,
            "tiny_cuda_nn_commit": TINY_CUDA_NN_COMMIT,
        },
        "execution_profile": {
            "pipeline": "Pano2MeshSRPipeline",
            "pano_geo_refiner_executed": False,
            "tinycudann_installed": True,
            "tinycudann_required_by_pano_joint_predictor": True,
            "omnidata_timm_imagenet_bootstrap": False,
            "omnidata_task_checkpoints_supply_full_model": True,
            "equilib_cube2equi_argument_order_patch": {
                "upstream_call_order": "height_width",
                "equilib_0_3_0_signature": "width_height",
                "effect": "swap_arguments_only_no_resampling_change",
            },
        },
        "panorama_stage": panorama_stage,
        "retained_source_files": _inventory(provenance_root),
        "canonical_transform": transform_receipt,
    }
    source_inventory_path = background / "source_inventory.json"
    _write_json(source_inventory_path, source_inventory)
    source_inventory_sha = _sha256(source_inventory_path)
    manifest = {
        "schema": "InteractMoveEnvironmentGeometry",
        "schema_version": 1,
        "native_profile": NATIVE_PROFILE,
        "reference_mesh": {
            "path": "mesh_model.ply",
            "frame": "world",
            "units": "m",
            "scale_xyz": [1.0, 1.0, 1.0],
            "purposes": ["interactmove_pointcloud", "raster_render"],
        },
        "collision_geometry": [
            {
                "collision_id": "floor",
                "role": "floor",
                "geometry_type": "plane",
                "frame": "world",
                "z_m": 0.0,
                "normal": [0.0, 0.0, 1.0],
            },
            {
                "collision_id": "walls",
                "role": "walls",
                "geometry_type": "mesh",
                "frame": "world",
                "path": "collision/walls.obj",
                "units": "m",
                "scale_xyz": [1.0, 1.0, 1.0],
            },
        ],
        "render_mode": "raster_reference_mesh",
        "provenance": {
            "source": "EmbodiedGenV2 scene3d panorama-to-mesh workflow",
            "embodiedgen_commit": EMBODIEDGEN_COMMIT,
            "source_job_id": args.job_id,
            "source_artifact_path": "source_inventory.json",
            "source_artifact_sha256": source_inventory_sha,
            "creation_method": (
                "released panorama and Pano2Mesh stages; baked Y-up-to-Z-up metric "
                "transform; separate authored floor/wall collision"
            ),
        },
    }
    _write_json(background / "environment_geometry.json", manifest)

    layout_path = candidate / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    background_node = layout["relation"]["background"]
    manipulated_nodes = layout["relation"]["manipulated_objs"]
    distractor_nodes = layout["relation"]["distractor_objs"]
    if manipulated_nodes != [target_node]:
        raise RuntimeError(
            "room repair refuses a changed or ambiguous accepted target identity"
        )
    if layout["relation"]["context"] != "table":
        raise RuntimeError("room repair refuses a changed accepted support identity")
    non_background_asset_count = 1 + len(manipulated_nodes) + len(distractor_nodes)
    layout["position"][background_node] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    _write_json(layout_path, layout)

    if _json_sha256(_inventory(asset_root)) != preserved_inventory_sha:
        raise RuntimeError("accepted SAM3D asset bytes changed during room repair")
    original_receipt_sha = _sha256(original_receipt_path)
    preserved_receipt_path = candidate / "metadata" / "pre_room_repair_generation_receipt.json"
    shutil.move(original_receipt_path, preserved_receipt_path)
    receipt = {
        "schema": "InteractMoveInterMimicFreshGenerationReceipt",
        "schema_version": 1,
        "status": "passed",
        "activity_id": activity["activity_id"],
        "prompt": activity_prompt,
        "seeds": {
            "image": seeds["embodiedgen_image"],
            "asset": seeds["embodiedgen_asset"],
            "layout": seeds["embodiedgen_layout"],
        },
        "layout": {
            "path": "layout.json",
            "sha256": _sha256(layout_path),
            "native_robot_metadata_declared": True,
            "native_robot_source_key": layout["relation"].get("robot"),
            "native_robot_loaded": False,
            "non_background_asset_count": non_background_asset_count,
        },
        "robot_policy": {
            "render_insert_robot": False,
            "native_layout_robot_metadata_may_be_present": True,
            "upstream_sim_cli_invoked": False,
            "robot_actor_loaded": False,
        },
        "models": {
            **original_receipt["models"],
            "room_scene3d": source_inventory["pins"],
        },
        "room_completion": {
            "job_id": args.job_id,
            "room_prompt": args.room_prompt,
            "seed": args.seed,
            "source_inventory": {
                "path": "background/source_inventory.json",
                "sha256": source_inventory_sha,
            },
            "environment_manifest": {
                "path": "background/environment_geometry.json",
                "sha256": _sha256(background / "environment_geometry.json"),
            },
            "reference_mesh": {
                "path": "background/mesh_model.ply",
                "sha256": _sha256(final_mesh),
                **transform_receipt,
            },
            "wall_collision": {
                "path": "background/collision/walls.obj",
                "sha256": _sha256(walls_path),
            },
        },
        "preservation": {
            "pre_repair_generation_receipt_sha256": original_receipt_sha,
            "accepted_asset_inventory_sha256": preserved_inventory_sha,
            "target_source_node_key": target_node,
            "target_instance_id": target_instance_id,
            "accepted_assets_regenerated": False,
        },
        "metadata_hashes": {
            "activity.json": _sha256(activity_path),
            "embodiedgen_request.json": _sha256(request_path),
            "protocol.json": _sha256(protocol_path),
        },
    }
    _write_json(candidate / "generation_receipt.json", receipt)
    return receipt


def main() -> int:
    args = _parser().parse_args()
    receipt = run(args)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
