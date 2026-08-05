from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .contracts import (
    CONFIG_PATH,
    OUTPUT_ROOT,
    REPOSITORY_ROOT,
    assert_output_root_ignored,
    compile_contract_matrix,
    load_frozen_config,
)


EDITOR_SOURCE_FILES = (
    "GateContracts.cs",
    "ProceduralSceneGateBuilder.cs",
    "EmbodimentGarments.cs",
    "FullBodyBimanualMotion.cs",
    "ProceduralSceneCompiler.cs",
    "PhysicsTruthRecorder.cs",
    "RegisteredCapture.cs",
)
SHADER_SOURCE_FILES = ("MetricDepth.shader", "SemanticInstance.shader")
FURNITURE_MEMBERS = (
    "bookcaseOpen.obj",
    "books.obj",
    "chairCushion.obj",
    "lampSquareFloor.obj",
    "loungeSofaLong.obj",
    "pottedPlant.obj",
    "rugRectangle.obj",
    "tableCoffee.obj",
)
AVATAR_SHA256 = "b766981d9d3504cea220c0d72ad8aa56cbd80453e910fc76dc8c8814fbd980de"
UNITY_VERSION = "6000.0.80f1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_unity_editor() -> Path:
    configured = os.environ.get("PROCEDURAL_GATE_UNITY")
    candidates = [Path(configured)] if configured else []
    candidates.append(Path("/Applications/Unity/Hub/Editor") / UNITY_VERSION / "Unity.app/Contents/MacOS/Unity")
    candidates.extend(
        Path.home().glob(
            f".codex/worktrees/*/nursery/.external/unity-editors/{UNITY_VERSION}/Unity.app/Contents/MacOS/Unity"
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            version = subprocess.run(
                [str(candidate), "-version"], capture_output=True, text=True, check=False
            )
            if version.returncode == 0 and version.stdout.strip() == UNITY_VERSION:
                return candidate.resolve()
    raise FileNotFoundError(f"Unity {UNITY_VERSION} editor was not found")


def _valid_asset_project(candidate: Path) -> bool:
    avatar = candidate / "Assets" / "Avatar" / "child.fbx"
    furniture = candidate / "Assets" / "Furniture"
    return (
        avatar.is_file()
        and _sha256(avatar) == AVATAR_SHA256
        and all((furniture / name).is_file() for name in FURNITURE_MEMBERS)
    )


def discover_public_asset_project() -> Path:
    configured = os.environ.get("PROCEDURAL_GATE_ASSET_PROJECT")
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        Path.home().glob(
            ".codex/worktrees/*/nursery/runs/embodied_simulation/*/project"
        )
    )
    for candidate in candidates:
        if candidate and _valid_asset_project(candidate):
            return candidate.resolve()
    raise FileNotFoundError("no verified public CC0 avatar/furniture Unity asset project was found")


def _copy_with_meta(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_meta = source.with_name(source.name + ".meta")
    if source_meta.is_file():
        shutil.copy2(source_meta, destination.with_name(destination.name + ".meta"))


def prepare_project(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    assert_output_root_ignored(output_root)
    unity = discover_unity_editor()
    source_project = discover_public_asset_project()
    project = output_root / "project"
    project.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_project / "ProjectSettings", project / "ProjectSettings", dirs_exist_ok=True)
    shutil.copytree(source_project / "Packages", project / "Packages", dirs_exist_ok=True)
    avatar_source = source_project / "Assets" / "Avatar" / "child.fbx"
    _copy_with_meta(avatar_source, project / "Assets" / "Avatar" / "child.fbx")
    furniture_hashes: dict[str, str] = {}
    for name in FURNITURE_MEMBERS:
        source = source_project / "Assets" / "Furniture" / name
        _copy_with_meta(source, project / "Assets" / "Furniture" / name)
        furniture_hashes[name] = _sha256(source)
    module_root = Path(__file__).resolve().parent
    for name in EDITOR_SOURCE_FILES:
        source = module_root / name
        if not source.is_file():
            raise FileNotFoundError(f"canonical Unity module is missing: {source}")
        _copy_with_meta(source, project / "Assets" / "Editor" / name)
    for name in SHADER_SOURCE_FILES:
        source = module_root / name
        if not source.is_file():
            raise FileNotFoundError(f"canonical capture shader is missing: {source}")
        _copy_with_meta(source, project / "Assets" / "Shaders" / name)
    receipt = {
        "schema": "embodied.unity_project_receipt.v1",
        "unity_editor": str(unity),
        "unity_version": UNITY_VERSION,
        "source_project": str(source_project),
        "project": str(project),
        "output_root_ignored": True,
        "avatar": {
            "path": "Assets/Avatar/child.fbx",
            "sha256": _sha256(avatar_source),
            "license": "CC0",
        },
        "furniture": {
            "root": "Assets/Furniture",
            "member_sha256": furniture_hashes,
            "license": "CC0",
        },
        "config_path": str(CONFIG_PATH),
        "config_sha256": _sha256(CONFIG_PATH),
        "canonical_editor_sources": list(EDITOR_SOURCE_FILES),
        "canonical_shaders": list(SHADER_SOURCE_FILES),
    }
    (output_root / "unity_project_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def contract_by_episode_id(episode_id: str) -> dict[str, Any]:
    for contract in compile_contract_matrix(load_frozen_config()):
        if contract["episode_id"] == episode_id:
            return contract
    raise ValueError(f"unknown frozen episode id: {episode_id}")


def run_unity_episode(
    episode_id: str,
    *,
    output_root: Path = OUTPUT_ROOT,
    capture_mode: str = "all",
    stage: str = "integrated",
) -> dict[str, Any]:
    if capture_mode not in {"all", "qualification", "none"}:
        raise ValueError("capture_mode must be all, qualification, or none")
    if stage not in {"garment_sweep", "motion_camera", "bimanual_cell", "integrated"}:
        raise ValueError("stage must be garment_sweep, motion_camera, bimanual_cell, or integrated")
    receipt = prepare_project(output_root)
    contract = contract_by_episode_id(episode_id)
    episode_root = output_root.resolve() / "episodes" / episode_id
    run_root = episode_root / stage
    run_root.mkdir(parents=True, exist_ok=True)
    contract_path = episode_root / "compiled_contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log_path = run_root / "unity.log"
    environment = os.environ.copy()
    environment.update(
        {
            "PROCEDURAL_GATE_OUTPUT": str(run_root),
            "PROCEDURAL_GATE_EPISODE_ID": episode_id,
            "PROCEDURAL_GATE_SEED": str(contract["scene_spec"]["seed"]),
            "PROCEDURAL_GATE_ROOM_FAMILY": contract["scene_spec"]["room_family"],
            "PROCEDURAL_GATE_GARMENT_CONFIG": contract["avatar_spec"]["garment_configuration_id"],
            "PROCEDURAL_GATE_CAPTURE_MODE": capture_mode,
            "PROCEDURAL_GATE_STAGE": stage,
            "PROCEDURAL_GATE_CONTRACT": str(contract_path),
        }
    )
    command = [
        receipt["unity_editor"],
        "-batchmode",
        "-force-metal",
        "-projectPath",
        receipt["project"],
        "-executeMethod",
        "ProceduralSceneGate.ProceduralSceneGateBuilder.Run",
        "-logFile",
        str(log_path),
    ]
    result = subprocess.run(command, env=environment, check=False)
    run_receipt = {
        "schema": "embodied.unity_episode_execution.v1",
        "episode_id": episode_id,
        "capture_mode": capture_mode,
        "stage": stage,
        "returncode": result.returncode,
        "command": command,
        "contract_sha256": contract["contract_sha256"],
        "run_root": str(run_root),
        "unity_log": str(log_path),
    }
    (run_root / "execution_receipt.json").write_text(
        json.dumps(run_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(f"Unity episode failed with exit code {result.returncode}; see {log_path}")
    return run_receipt


def encode_videos(
    episode_id: str,
    output_root: Path = OUTPUT_ROOT,
    *,
    stage: str = "integrated",
) -> dict[str, Any]:
    if stage not in {"garment_sweep", "motion_camera", "bimanual_cell", "integrated"}:
        raise ValueError("unknown ordered gate stage")
    run_root = output_root.resolve() / "episodes" / episode_id / stage
    products: dict[str, dict[str, Any]] = {}
    streams = {
        "head_rgb": run_root / "head" / "rgb",
        "head_depth": run_root / "head" / "depth",
        "head_semantic": run_root / "head" / "semantic",
        "head_instance": run_root / "head" / "instance",
        "external_clean": run_root / "external" / "clean",
        "external_overlay": run_root / "external" / "overlay",
    }
    for name, frames in streams.items():
        first = frames / "frame_0000.png"
        if not first.is_file():
            continue
        output = run_root / f"{name}.mp4"
        command = [
            "ffmpeg",
            "-y",
            "-framerate",
            "30",
            "-i",
            str(frames / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "18" if name in {"head_rgb", "external_clean", "external_overlay"} else "0",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {name}: {result.stderr[-2000:]}")
        products[name] = {"path": str(output), "sha256": _sha256(output), "frames_root": str(frames)}
    receipt = {
        "schema": "embodied.encoded_video_manifest.v1",
        "episode_id": episode_id,
        "stage": stage,
        "products": products,
    }
    (run_root / "encoded_video_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
