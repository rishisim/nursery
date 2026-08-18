from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any, Mapping

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
PYTHON_SOURCE_FILES = (
    "__init__.py", "__main__.py", "contracts.py", "runner.py", "cli.py", "independent_qa.py",
)
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
SOURCE_AUDIT_SCHEMA = "embodied.procedural_gate_source_audit.v1"
STAGE_ORDER = {
    "microcell": "A",
    "clipping": "B",
    "motion_camera": "C",
    "polished_cell": "D",
    "integrated": "E",
    "robustness": "F",
    "replay": "F",
    "rerender": "F",
    "qa": "F",
}
UNITY_EXECUTION_STAGES = tuple(stage for stage in STAGE_ORDER if stage != "qa")
ROBUSTNESS_VARIANTS = ("nominal", "lateral_target_shift", "mass_friction_change")
_STAGE_PREREQUISITE = {
    "clipping": ("microcell", "A"),
    "motion_camera": ("clipping", "B"),
    "polished_cell": ("motion_camera", "C"),
    "integrated": ("polished_cell", "D"),
    "robustness": ("integrated", "E"),
    "replay": ("integrated", "E"),
    "rerender": ("integrated", "E"),
}

# This is intentionally an explicit, narrow source-level allowlist.  Target
# rigidbody mutations are checked before these rules, so a broad non-target
# category can never excuse an operation on the manipulated object.
SOURCE_AUTHORITY_ALLOWLIST = (
    {
        "files": ("ProceduralSceneGateBuilder.cs",),
        "operations": ("pose_write", "transform_pose_call"),
        "scope": "rerender_only",
        "reason": "explicit Physics-disabled source-trace render playback excluded from manipulation evidence",
        "methods": ("ApplyReplayPose", "ApplyReplayFingerBody", "ApplyReplayTargetState"),
        "statements": (
            "target.SetPositionAndRotation(pose.position_world_m, pose.rotation_world_xyzw);",
            "binding.body.position = binding.bone.TransformPoint(binding.positionInBone);",
            "binding.body.rotation = binding.bone.rotation * binding.rotationInBone;",
            "binding.body.position = state.pose.position_world_m;",
            "binding.body.rotation = state.pose.rotation_world_xyzw;",
            "binding.body.linearVelocity = state.linear_velocity_world_m_s;",
            "binding.body.angularVelocity = state.angular_velocity_world_rad_s;",
            "body.position = state.pose.position_world_m;",
            "body.rotation = state.pose.rotation_world_xyzw;",
            "body.linearVelocity = state.linear_velocity_world_m_s;",
            "body.angularVelocity = state.angular_velocity_world_rad_s;",
        ),
    },
    {
        "files": ("ProceduralSceneCompiler.cs",),
        "operations": ("parenting", "pose_write", "kinematic_change", "transform_pose_call", "gravity_change"),
        "scope": "initialization",
        "reason": "deterministic scene compilation before the first physics step",
    },
    {
        "files": ("EmbodimentGarments.cs",),
        "operations": ("parenting", "pose_write", "kinematic_change", "move_pose", "transform_pose_call", "gravity_change", "compliant_finger_joint"),
        "scope": "non_target",
        "reason": "avatar, garment, and registered anatomical-collider state only",
    },
    {
        "files": ("FullBodyBimanualMotion.cs",),
        "operations": ("pose_write", "transform_pose_call", "compliant_finger_joint"),
        "scope": "non_target",
        "reason": "bounded body targets and dynamic articulated finger drives only",
    },
    {
        "files": ("RegisteredCapture.cs",),
        "operations": ("parenting", "pose_write", "transform_pose_call"),
        "scope": "non_target",
        "reason": "one-time camera mount and fixed external diagnostic camera configuration",
    },
)

_AUTHORITY_PATTERNS = (
    ("force_or_torque", re.compile(r"\.(?:AddForce|AddRelativeForce|AddTorque|AddRelativeTorque|AddExplosionForce)\s*\(")),
    ("forbidden_joint", re.compile(r"AddComponent\s*(?:<\s*(?:FixedJoint|SpringJoint|HingeJoint)\s*>|\(\s*typeof\s*\(\s*(?:FixedJoint|SpringJoint|HingeJoint)\s*\)\s*\))")),
    ("compliant_finger_joint", re.compile(r"AddComponent\s*<\s*ConfigurableJoint\s*>")),
    ("parenting", re.compile(r"\.SetParent\s*\(|\.parent\s*=(?!=)")),
    ("kinematic_change", re.compile(r"\.isKinematic\s*=")),
    ("move_pose", re.compile(r"\.(?:MovePosition|MoveRotation)\s*\(")),
    ("transform_pose_call", re.compile(r"\.(?:SetPositionAndRotation|Translate|Rotate)\s*\(")),
    ("gravity_change", re.compile(r"\.useGravity\s*=")),
    (
        "pose_write",
        re.compile(
            r"\.(?:position|rotation|localPosition|localRotation|velocity|linearVelocity|angularVelocity)\s*(?:\+|-)?="
        ),
    ),
)
_TARGET_TOKENS = re.compile(
    r"\b(?:TargetBody|targetBody|target_object|interactive_target|manipulated_object)\b",
    re.IGNORECASE,
)
_TARGET_INITIALIZATION_LINES = {
    "body.isKinematic = false;",
    "body.useGravity = true;",
    "target.transform.SetPositionAndRotation(",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _allowed_source_authority_operation(
    file_name: str, operation: str, statement: str, method_name: str | None
) -> Mapping[str, Any] | None:
    for rule in SOURCE_AUTHORITY_ALLOWLIST:
        if file_name in rule["files"] and operation in rule["operations"]:
            allowed_statements = rule.get("statements")
            if allowed_statements is not None and statement not in allowed_statements:
                continue
            allowed_methods = rule.get("methods")
            if allowed_methods is not None and method_name not in allowed_methods:
                continue
            return rule
    return None


def _verified_render_only_target_replay(lines: list[str]) -> bool:
    text = "\n".join(lines)
    try:
        replay_body = text[text.index("private static void RunRenderOnlyReplay"):text.index("private static void ApplyRenderOnlyReplayState")]
    except ValueError:
        return False
    required = (
        'if (stage == "rerender")',
        "Physics.simulationMode = SimulationMode.Script;",
        "ApplyRenderOnlyReplayState(context, row, true);",
        'if (!renderOnly) throw new InvalidOperationException("trace-driven pose writes are forbidden in physics execution")',
        "ApplyReplayTargetState(context.TargetBody, row.objects[0], renderOnly);",
        'if (!renderOnly) throw new InvalidOperationException("target replay writes are forbidden during physics execution")',
    )
    return all(marker in text for marker in required) and "Physics.Simulate" not in replay_body


def _enclosing_csharp_method(lines: list[str], line_index: int) -> str | None:
    declaration = re.compile(
        r"\b(?:public|private|internal|protected)\s+(?:static\s+)?(?:[\w<>\[\],]+\s+)+(\w+)\s*\("
    )
    for index in range(line_index, -1, -1):
        match = declaration.search(lines[index])
        if match:
            return match.group(1)
    return None


def audit_canonical_sources(
    module_root: Path | None = None, *, raise_on_forbidden: bool = True
) -> dict[str, Any]:
    """Fail-closed audit of APIs able to assist the manipulated target.

    This audit is mandatory pre-execution evidence, not a replacement for the
    runtime ledgers.  It records every recognized authority mutation, permits
    only the frozen initialization/non-target categories above, and rejects any
    force/torque, attachment, or post-initialization target authority write.
    """
    root = (module_root or Path(__file__).resolve().parent).resolve()
    source_names = EDITOR_SOURCE_FILES + SHADER_SOURCE_FILES + PYTHON_SOURCE_FILES
    missing = [name for name in source_names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"canonical gate sources missing from source audit: {missing}")

    findings: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for name in source_names:
        source = root / name
        source_hashes[name] = _sha256(source)
        lines = source.read_text(encoding="utf-8").splitlines()
        if name in SHADER_SOURCE_FILES + PYTHON_SOURCE_FILES:
            continue
        target_aliases = {
            match.group(1)
            for line in lines
            for match in [re.search(
                r"(?:Rigidbody|Transform|var)\s+(\w+)\s*=\s*[^;]*\bTargetBody\b",
                line,
                re.IGNORECASE,
            )]
            if match is not None
        }
        verified_render_only_target_replay = (
            name == "ProceduralSceneGateBuilder.cs" and _verified_render_only_target_replay(lines)
        )
        for line_number, line in enumerate(lines, 1):
            operations = [operation for operation, pattern in _AUTHORITY_PATTERNS if pattern.search(line)]
            for operation in operations:
                stripped = line.strip()
                method_name = _enclosing_csharp_method(lines, line_number - 1)
                target_related = method_name == "ApplyReplayTargetState" or bool(_TARGET_TOKENS.search(line)) or any(
                    re.search(rf"\b{re.escape(alias)}\s*\.", line) for alias in target_aliases
                )
                rule = _allowed_source_authority_operation(name, operation, stripped, method_name)
                initialization_exception = (
                    name == "ProceduralSceneCompiler.cs"
                    and stripped in _TARGET_INITIALIZATION_LINES
                    and operation in {"kinematic_change", "gravity_change", "transform_pose_call"}
                )
                render_only_target_exception = (
                    target_related
                    and verified_render_only_target_replay
                    and method_name == "ApplyReplayTargetState"
                    and isinstance(rule, Mapping)
                    and rule.get("scope") == "rerender_only"
                )
                forbidden_reason = None
                if operation in {"force_or_torque", "forbidden_joint"}:
                    forbidden_reason = "object assistance/attachment API is never allowed"
                elif target_related and not initialization_exception and not render_only_target_exception:
                    forbidden_reason = "target authority mutation is not an allowlisted initialization write"
                elif rule is None and not initialization_exception and not render_only_target_exception:
                    forbidden_reason = "authority mutation is outside the explicit initialization/non-target allowlist"
                finding = {
                    "file": name,
                    "line": line_number,
                    "operation": operation,
                    "statement_sha256": hashlib.sha256(stripped.encode("utf-8")).hexdigest(),
                    "target_related": target_related,
                    "method": method_name,
                    "status": "FORBIDDEN" if forbidden_reason else "ALLOWLISTED",
                    "scope": "initialization" if initialization_exception else (rule or {}).get("scope"),
                    "reason": forbidden_reason or (rule or {}).get("reason", "frozen target initialization"),
                }
                findings.append(finding)

    forbidden = [finding for finding in findings if finding["status"] == "FORBIDDEN"]
    body = {
        "schema": SOURCE_AUDIT_SCHEMA,
        "audit_policy": "fail_closed_explicit_initialization_and_non_target_allowlist",
        "source_root": str(root),
        "source_sha256": source_hashes,
        "source_set_sha256": _canonical_json_sha256(source_hashes),
        "allowlist": [dict(
            rule,
            files=list(rule["files"]),
            operations=list(rule["operations"]),
            **({"methods": list(rule["methods"])} if "methods" in rule else {}),
            **({"statements": list(rule["statements"])} if "statements" in rule else {}),
        )
                      for rule in SOURCE_AUTHORITY_ALLOWLIST],
        "findings": findings,
        "forbidden_findings": forbidden,
        "passed": not forbidden,
    }
    body["receipt_sha256"] = _canonical_json_sha256(body)
    if forbidden and raise_on_forbidden:
        locations = ", ".join(
            f"{row['file']}:{row['line']}:{row['operation']}" for row in forbidden[:12]
        )
        raise RuntimeError(f"canonical source audit rejected forbidden target assistance: {locations}")
    return body


def _require_backend_environment_contract(name: str, module_root: Path | None = None) -> None:
    root = (module_root or Path(__file__).resolve().parent).resolve()
    builder = (root / "ProceduralSceneGateBuilder.cs").read_text(encoding="utf-8")
    code = re.sub(r"/\*.*?\*/", "", builder, flags=re.DOTALL)
    code = "\n".join(line.split("//", 1)[0] for line in code.splitlines())
    semantic_markers = {
        "PROCEDURAL_GATE_CONTRACT": (
            "LoadAndVerifyCompiledContract()",
            "BuildAuthoritativeContext(contract, output, stage)",
            "PROCEDURAL_GATE_CONTRACT_FILE_SHA256",
            "RequireExactEnvironment(\"PROCEDURAL_GATE_EPISODE_ID\"",
        ),
        "PROCEDURAL_GATE_VARIANT": (
            "ApplyFrozenRobustnessVariant(context, contract.robustness_variants)",
            "context.TargetLateralBiasM += variant.target_lateral_shift_m",
            "context.TargetMassKg *= variant.mass_scale",
        ),
        "PROCEDURAL_GATE_REPLAY_TRACE": (
            "LoadReplayTraceIfRequested(context)",
            "ValidateReplayRow(context, rows[step]",
            "RunRenderOnlyReplay(context, replayRows",
            "CompareReplayObjectState(context, replayRows[step])",
        ),
    }
    markers = semantic_markers.get(name)
    if markers is None or name not in code or not all(marker in code for marker in markers):
        raise RuntimeError(f"Unity backend does not semantically consume required frozen environment field: {name}")


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
    output_root.mkdir(parents=True, exist_ok=True)
    source_audit = audit_canonical_sources(raise_on_forbidden=False)
    source_audit_path = output_root / "source_audit_receipt.json"
    source_audit_path.write_text(
        json.dumps(source_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not source_audit["passed"]:
        raise RuntimeError(f"canonical source audit failed; see {source_audit_path}")
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
        "canonical_python_sources": list(PYTHON_SOURCE_FILES),
    }
    receipt["source_audit_receipt"] = str(source_audit_path)
    receipt["source_audit_sha256"] = source_audit["receipt_sha256"]
    (output_root / "unity_project_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def contract_by_episode_id(episode_id: str) -> dict[str, Any]:
    for contract in compile_contract_matrix(load_frozen_config()):
        if contract["episode_id"] == episode_id:
            return contract
    raise ValueError(f"unknown frozen episode id: {episode_id}")


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _last_trace_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    last = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    if not last:
        return {}
    value = json.loads(last)
    return value if isinstance(value, dict) else {}


def _bool_from(primary: Mapping[str, Any], fallback: Mapping[str, Any], *names: str) -> bool:
    for source in (primary, fallback):
        for name in names:
            if name in source:
                return source.get(name) is True
    return False


def _actual_run_summary(run_receipt: Mapping[str, Any]) -> dict[str, Any]:
    run_root = Path(str(run_receipt["run_root"]))
    contract = contract_by_episode_id(str(run_receipt["episode_id"]))
    authority = _read_json_if_present(run_root / "authority_receipt.json")
    trace = _read_json_if_present(run_root / "episode_trace_manifest.json")
    validation = _read_json_if_present(run_root / "scene_compiler_validation.json")
    scene = _read_json_if_present(run_root / "scene_spec.json")
    interaction = _read_json_if_present(run_root / "interaction_summary.json")
    final_row = _last_trace_row(run_root / "episode_trace.jsonl")
    qualification = final_row.get("force_bearing_qualification", {})
    if not isinstance(qualification, Mapping):
        qualification = {}
    registration = _read_json_if_present(run_root / "registration_report.json")
    capture = _read_json_if_present(run_root / "registered_capture_manifest.json")
    projection = _read_json_if_present(run_root / "contact_projection.json")
    independent_qa = _read_json_if_present(run_root / "independent_qa_report.json")
    right_force = _bool_from(
        interaction, {}, "right_measured_impulse_qualified", "right_force_opposition_qualified"
    ) and _bool_from(qualification, {}, "right_measured_impulse_qualified", "right_force_opposition_qualified")
    left_force = _bool_from(
        interaction, {}, "left_measured_impulse_qualified", "meaningful_left_force_support_qualified"
    ) and _bool_from(qualification, {}, "left_measured_impulse_qualified", "meaningful_left_force_support_qualified")
    lift = _bool_from(interaction, {}, "lift_over_0_10_m", "lift_threshold_passed") \
        and _bool_from(qualification, {}, "lift_over_0_10_m", "lift_threshold_passed")
    turn = _bool_from(interaction, {}, "turn_over_30_deg", "turn_threshold_passed") \
        and _bool_from(qualification, {}, "turn_over_30_deg", "turn_threshold_passed")
    destination_release = interaction.get("release_at_required_destination") is True
    interaction_passed = (
        interaction.get("schema") == "embodied.physics_truth.interaction_summary.v1"
        and right_force and left_force and lift and turn and destination_release
        and interaction.get("object_unsupported_during_qualified_manipulation") is True
        and interaction.get("finger_object_penetration_passed") is True
        and interaction.get("target_support_penetration_passed") is True
    )
    registration_passed = (
        registration.get("schema") == "embodied.embodiment_registration.v2"
        and registration.get("passed") is True
        and registration.get("self_clearance_sampled_every_physics_step") is True
        and registration.get("non_adjacent_anatomy_clearance_passed") is True
    )
    capture_passed = (
        capture.get("schema") == "embodied.registered_capture_manifest.v1"
        and isinstance(capture.get("frames_captured"), int) and capture["frames_captured"] > 0
        and capture.get("contiguous_frames") is True
        and capture.get("exact_integer_clock") is True
        and capture.get("all_modalities_state_invariant") is True
        and capture.get("hero_contains_proxy_pixels") is False
    )
    projection_rows = projection.get("records")
    projection_passed = (
        projection.get("schema") == "embodied.registered_contact_projection.v1"
        and isinstance(projection_rows, list) and bool(projection_rows)
        and all(
            isinstance(row, Mapping)
            and row.get("contact_projects_to_expected_visible_surface") is True
            and row.get("contact_visible_in_registered_frame") is True
            for row in projection_rows
        )
    )
    independent_gate_summary = independent_qa.get("gate_summary")
    independent_qa_passed = (
        independent_qa.get("schema") == "embodied.independent_qa_report.v1"
        and isinstance(independent_gate_summary, Mapping)
        and all(independent_gate_summary.get(gate) == "PASS" for gate in "ABCD")
    )
    completed = (
        run_receipt.get("returncode") == 0
        and authority.get("runtime_accounting_passed") is True
        and trace.get("trace_complete") is True
        and validation.get("target_unparented") is True
        and validation.get("target_non_kinematic") is True
        and validation.get("all_visible_context_physics_backed") is True
        and interaction_passed
        and registration_passed
        and capture_passed
        and projection_passed
        and independent_qa_passed
    )
    return {
        "episode_id": contract["episode_id"],
        "room_family": contract["scene_spec"]["room_family"],
        "room_seed": contract["scene_spec"]["seed"],
        "garment_configuration_id": contract["avatar_spec"]["garment_configuration_id"],
        "destination_id": contract["episode_spec"]["destination_id"],
        "destination_relation": contract["episode_spec"]["destination_id"],
        "contact_strategy": contract["episode_spec"]["contact_strategy"],
        "compiler_id": "ProceduralSceneCompiler.canonical",
        "catalog_id": scene.get("catalog_id", "missing_runtime_catalog_receipt"),
        "controller_profile_id": contract["activity_plan"]["controller_profile_id"],
        "seed_specific_retuning": False,
        "assistance_entries": authority.get("assistance_ledger_entries"),
        "contextual_object_count": validation.get("contextual_visible_object_count", 0),
        "closed_finished_room": bool(validation.get("all_visible_context_physics_backed"))
        and bool(validation.get("camera_aware_target_and_destination_sightlines")),
        "no_visible_primitive_furniture": scene.get("no_visible_primitive_furniture") is True,
        "all_reachable_objects_physics_backed": validation.get("reachable_elements_have_physx_colliders") is True,
        "compiler_generated": validation.get("deterministic_prospective_rejection_only") is True,
        "interaction_passed": interaction_passed,
        "right_force_opposition_qualified": right_force,
        "meaningful_left_force_support_qualified": left_force,
        "lift_threshold_passed": lift,
        "turn_threshold_passed": turn,
        "release_at_required_destination": destination_release,
        "registration_passed": registration_passed,
        "anatomical_self_clearance_passed": registration_passed,
        "registered_capture_passed": capture_passed,
        "contact_projection_passed": projection_passed,
        "independent_qa_passed": independent_qa_passed,
        "independent_qa": independent_qa,
        "automated_visual_review_claimed": False,
        "completed": completed,
        "stage": run_receipt.get("stage"),
        "run_root": str(run_root),
    }


def _existing_execution_receipts(output_root: Path) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    episodes_root = output_root.resolve() / "episodes"
    if not episodes_root.is_dir():
        return receipts
    for path in sorted(episodes_root.glob("*/**/execution_receipt.json")):
        try:
            receipt = _read_json_if_present(path)
        except (OSError, json.JSONDecodeError):
            continue
        if receipt:
            receipts.append(receipt)
    return receipts


def _require_stage_prerequisites(episode_id: str, output_root: Path, stage: str) -> None:
    chain: list[tuple[str, str]] = []
    cursor = stage
    while cursor in _STAGE_PREREQUISITE:
        prior_stage, gate = _STAGE_PREREQUISITE[cursor]
        chain.append((prior_stage, gate))
        cursor = prior_stage
    if not chain:
        return
    chain.reverse()
    expected_contract_sha = contract_by_episode_id(episode_id)["contract_sha256"]
    expected_source_audit_sha = audit_canonical_sources()["receipt_sha256"]
    failures: list[str] = []
    for prior_stage, required_gate in chain:
        prior_root = (
            output_root.resolve() / "episodes" / episode_id
            / f"{STAGE_ORDER[prior_stage]}_{prior_stage}"
        )
        report = _read_json_if_present(prior_root / "independent_qa_report.json")
        execution = _read_json_if_present(prior_root / "execution_receipt.json")
        gate_summary = report.get("gate_summary")
        provenance = report.get("contract_provenance")
        passed = (
            report.get("schema") == "embodied.independent_qa_report.v1"
            and report.get("run_root") == str(prior_root)
            and isinstance(gate_summary, Mapping)
            and gate_summary.get(required_gate) == "PASS"
            and execution.get("schema") == "embodied.unity_episode_execution.v1"
            and execution.get("episode_id") == episode_id
            and execution.get("stage") == prior_stage
            and execution.get("returncode") == 0
            and execution.get("run_root") == str(prior_root)
            and execution.get("contract_sha256") == expected_contract_sha
            and execution.get("source_audit_sha256") == expected_source_audit_sha
            and isinstance(provenance, Mapping)
            and provenance.get("executed_episode_contract_sha256") == execution.get("contract_sha256")
        )
        if not passed:
            failures.append(f"gate {required_gate} in {prior_root / 'independent_qa_report.json'}")
    if failures:
        raise RuntimeError(
            f"ordered gate prerequisite not satisfied: {stage} requires stage-specific actual PASS receipts for "
            + ", ".join(failures)
        )


def _write_episode_matrix(
    output_root: Path,
    *,
    current_receipt: Mapping[str, Any] | None = None,
    robustness_runs: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    receipts = _existing_execution_receipts(output_root)
    if current_receipt is not None:
        current_root = str(current_receipt["run_root"])
        receipts = [row for row in receipts if str(row.get("run_root")) != current_root]
        receipts.append(dict(current_receipt))
    integrated = {
        str(row.get("episode_id")): _actual_run_summary(row)
        for row in receipts
        if row.get("stage") == "integrated" and row.get("returncode") == 0
    }
    robustness_source = robustness_runs or [
        row for row in receipts if row.get("stage") == "robustness" and row.get("returncode") == 0
    ]
    robustness = []
    for row in robustness_source:
        summary = _actual_run_summary(row)
        robustness.append(
            {
                "variant": row.get("variant"),
                "episode_id": row.get("episode_id"),
                "completed": summary["completed"],
                "controller_profile_id": summary["controller_profile_id"],
                "retuned": False,
                "assistance_entries": summary["assistance_entries"],
                "independent_qa": summary["independent_qa"],
                "run_root": summary["run_root"],
            }
        )
    report = {
        "schema": "embodied.episode_matrix.actual_results.v1",
        "episodes": [integrated[key] for key in sorted(integrated)],
        "primary_robustness": robustness,
    }
    destinations = {Path(str(row["run_root"])) for row in receipts if row.get("returncode") == 0}
    if current_receipt is not None:
        destinations.add(Path(str(current_receipt["run_root"])))
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "episode_matrix.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def refresh_episode_matrix(run_root: Path) -> dict[str, Any]:
    resolved = run_root.resolve()
    episodes_root = next(
        (candidate for candidate in (resolved, *resolved.parents) if candidate.name == "episodes"),
        None,
    )
    if episodes_root is None:
        raise ValueError(f"run root is not under a canonical episodes directory: {resolved}")
    return _write_episode_matrix(episodes_root.parent)


def run_unity_episode(
    episode_id: str,
    *,
    output_root: Path = OUTPUT_ROOT,
    capture_mode: str = "all",
    stage: str = "integrated",
    variant: str | None = None,
    replay_trace: Path | None = None,
) -> dict[str, Any]:
    if capture_mode not in {"all", "qualification", "none"}:
        raise ValueError("capture_mode must be all, qualification, or none")
    if stage not in UNITY_EXECUTION_STAGES:
        raise ValueError(f"Unity stage must be one of: {', '.join(UNITY_EXECUTION_STAGES)}; run QA independently")
    if variant is not None and (not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", variant) or "/" in variant):
        raise ValueError("variant must be a simple lowercase run label")
    _require_backend_environment_contract("PROCEDURAL_GATE_CONTRACT")
    if stage == "robustness":
        _require_backend_environment_contract("PROCEDURAL_GATE_VARIANT")
    if replay_trace is not None:
        _require_backend_environment_contract("PROCEDURAL_GATE_REPLAY_TRACE")
    _require_stage_prerequisites(episode_id, output_root, stage)
    receipt = prepare_project(output_root)
    contract = contract_by_episode_id(episode_id)
    episode_root = output_root.resolve() / "episodes" / episode_id
    run_root = episode_root / f"{STAGE_ORDER[stage]}_{stage}"
    if variant:
        run_root /= variant
    run_root.mkdir(parents=True, exist_ok=True)
    contract_path = episode_root / "compiled_contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_audit_path = run_root / "source_audit_receipt.json"
    shutil.copy2(receipt["source_audit_receipt"], source_audit_path)
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
            "PROCEDURAL_GATE_CONTRACT_SHA256": contract["contract_sha256"],
            "PROCEDURAL_GATE_CONTRACT_FILE_SHA256": _sha256(contract_path),
            "PROCEDURAL_GATE_CELL_ID": contract["episode_spec"]["cell_id"],
            "PROCEDURAL_GATE_TARGET_ID": contract["episode_spec"]["target_id"],
            "PROCEDURAL_GATE_DESTINATION_ID": contract["episode_spec"]["destination_id"],
            "PROCEDURAL_GATE_CONTACT_STRATEGY": contract["episode_spec"]["contact_strategy"],
            "PROCEDURAL_GATE_FINAL_GAZE_ZONE": contract["episode_spec"]["final_gaze_zone"],
            "PROCEDURAL_GATE_SOURCE_AUDIT_SHA256": receipt["source_audit_sha256"],
        }
    )
    if variant and stage == "robustness":
        environment["PROCEDURAL_GATE_VARIANT"] = variant
    if replay_trace is not None:
        trace = replay_trace.resolve()
        if not trace.is_file():
            raise FileNotFoundError(f"replay trace is missing: {trace}")
        environment["PROCEDURAL_GATE_REPLAY_TRACE"] = str(trace)
        environment["PROCEDURAL_GATE_REPLAY_TRACE_SHA256"] = _sha256(trace)
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
        "ordered_gate": STAGE_ORDER[stage],
        "variant": variant,
        "returncode": result.returncode,
        "command": command,
        "contract_sha256": contract["contract_sha256"],
        "run_root": str(run_root),
        "unity_log": str(log_path),
        "source_audit_receipt": str(source_audit_path),
        "source_audit_sha256": receipt["source_audit_sha256"],
        "machine_id": socket.gethostname(),
        "replay_trace": str(replay_trace.resolve()) if replay_trace is not None else None,
    }
    (run_root / "execution_receipt.json").write_text(
        json.dumps(run_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(f"Unity episode failed with exit code {result.returncode}; see {log_path}")
    _write_episode_matrix(output_root, current_receipt=run_receipt)
    return run_receipt


def run_robustness_variants(
    episode_id: str,
    *,
    output_root: Path = OUTPUT_ROOT,
    capture_mode: str = "qualification",
) -> dict[str, Any]:
    """Run the frozen nominal/shift/property robustness cells in fresh processes."""
    runs = [
        run_unity_episode(
            episode_id,
            output_root=output_root,
            capture_mode=capture_mode,
            stage="robustness",
            variant=variant,
        )
        for variant in ROBUSTNESS_VARIANTS
    ]
    matrix = _write_episode_matrix(output_root, robustness_runs=runs)
    return {
        "schema": "embodied.robustness_execution_set.v1",
        "episode_id": episode_id,
        "variants": list(ROBUSTNESS_VARIANTS),
        "runs": runs,
        "episode_matrix": matrix,
    }


def run_fresh_process_replay(
    episode_id: str,
    trace: Path,
    *,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    source_trace = trace.resolve()
    run = run_unity_episode(
        episode_id,
        output_root=output_root,
        capture_mode="qualification",
        stage="replay",
        variant="fresh_process",
        replay_trace=source_trace,
    )
    replay_trace = Path(run["run_root"]) / "episode_trace.jsonl"
    metrics = _compare_object_traces(source_trace, replay_trace)
    consumption = _read_json_if_present(Path(run["run_root"]) / "replay_consumption_receipt.json")
    source_sha = _sha256(source_trace)
    report = {
        "schema": "embodied.replay_report.actual_results.v1",
        "episode_id": episode_id,
        "fresh_process": consumption.get("physics_simulated") is True
        and consumption.get("source_trace_sha256") == source_sha,
        "same_trace_rerender": False,
        "source_trace_sha256": source_sha,
        "replay_trace_sha256": _sha256(replay_trace),
        "rerender_trace_sha256": None,
        "source_render_sha256": _render_set_sha256(source_trace.parent),
        "rerender_render_sha256": None,
        "source_machine_id": _source_machine_id(source_trace.parent),
        "replay_machine_id": socket.gethostname(),
        "translation_max_m": metrics["translation_max_m"],
        "rotation_max_deg": metrics["rotation_max_deg"],
        "object_velocity_max_m_s": metrics["object_velocity_max_m_s"],
        "rerender_min_psnr_db": 0.0,
        "comparison_rows": metrics["rows"],
    }
    _write_replay_report(report, output_root, episode_id, source_trace, Path(run["run_root"]))
    return {**run, "replay_report": report}


def run_same_trace_rerender(
    episode_id: str,
    trace: Path,
    *,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    source_trace = trace.resolve()
    run = run_unity_episode(
        episode_id,
        output_root=output_root,
        capture_mode="all",
        stage="rerender",
        variant="same_trace",
        replay_trace=source_trace,
    )
    report_path = output_root.resolve() / "episodes" / episode_id / "replay_report.json"
    report = _read_json_if_present(report_path)
    source_sha = _sha256(source_trace)
    if report.get("source_trace_sha256") != source_sha:
        report = {
            "schema": "embodied.replay_report.actual_results.v1",
            "episode_id": episode_id,
            "fresh_process": False,
            "source_trace_sha256": source_sha,
            "translation_max_m": 1.0e9,
            "rotation_max_deg": 1.0e9,
            "object_velocity_max_m_s": 1.0e9,
        }
    consumption = _read_json_if_present(Path(run["run_root"]) / "replay_consumption_receipt.json")
    source_render_sha = _render_set_sha256(source_trace.parent)
    rerender_render_sha = _render_set_sha256(Path(run["run_root"]))
    report.update(
        {
            "same_trace_rerender": consumption.get("render_only") is True
            and consumption.get("eligible_manipulation_evidence") is False
            and consumption.get("source_trace_sha256") == source_sha,
            "rerender_trace_sha256": source_sha,
            "source_render_sha256": source_render_sha,
            "rerender_render_sha256": rerender_render_sha,
            "source_machine_id": report.get("source_machine_id") or _source_machine_id(source_trace.parent),
            "replay_machine_id": socket.gethostname(),
            "rerender_min_psnr_db": 100.0
            if source_render_sha is not None and source_render_sha == rerender_render_sha
            else 0.0,
            "rerender_comparison": "exact_registered_png_set_sha256",
        }
    )
    _write_replay_report(report, output_root, episode_id, source_trace, Path(run["run_root"]))
    return {**run, "replay_report": report}


def _trace_object(row: Mapping[str, Any], *, expected_step: int) -> Mapping[str, Any]:
    clock = row.get("clock")
    objects = row.get("objects")
    if not isinstance(clock, Mapping) or clock.get("physics_step") != expected_step:
        raise ValueError(f"trace clock is malformed at row {expected_step}")
    if not isinstance(objects, list) or len(objects) != 1 or not isinstance(objects[0], Mapping):
        raise ValueError(f"trace object authority is malformed at row {expected_step}")
    return objects[0]


def _xyz(value: Any, *, label: str) -> tuple[float, float, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a vector")
    try:
        return float(value["x"]), float(value["y"]), float(value["z"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} is malformed") from error


def _xyzw(value: Any, *, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a quaternion")
    try:
        return float(value["x"]), float(value["y"]), float(value["z"]), float(value["w"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} is malformed") from error


def _distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _quaternion_distance_deg(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0.0 or second_norm == 0.0:
        raise ValueError("trace contains a zero quaternion")
    dot = abs(sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def _compare_object_traces(source: Path, replay: Path) -> dict[str, Any]:
    translation_max = rotation_max = velocity_max = 0.0
    rows = 0
    with source.open("r", encoding="utf-8") as left, replay.open("r", encoding="utf-8") as right:
        for step, pair in enumerate(itertools.zip_longest(left, right)):
            source_line, replay_line = pair
            if source_line is None or replay_line is None:
                raise ValueError("source and replay traces have different row counts")
            source_object = _trace_object(json.loads(source_line), expected_step=step)
            replay_object = _trace_object(json.loads(replay_line), expected_step=step)
            if source_object.get("persistent_id") != replay_object.get("persistent_id"):
                raise ValueError(f"source and replay target identities differ at row {step}")
            source_pose = source_object.get("pose")
            replay_pose = replay_object.get("pose")
            if not isinstance(source_pose, Mapping) or not isinstance(replay_pose, Mapping):
                raise ValueError(f"source/replay object pose is malformed at row {step}")
            translation_max = max(
                translation_max,
                _distance(
                    _xyz(source_pose.get("position_world_m"), label="source position"),
                    _xyz(replay_pose.get("position_world_m"), label="replay position"),
                ),
            )
            rotation_max = max(
                rotation_max,
                _quaternion_distance_deg(
                    _xyzw(source_pose.get("rotation_world_xyzw"), label="source rotation"),
                    _xyzw(replay_pose.get("rotation_world_xyzw"), label="replay rotation"),
                ),
            )
            velocity_max = max(
                velocity_max,
                _distance(
                    _xyz(source_object.get("linear_velocity_world_m_s"), label="source velocity"),
                    _xyz(replay_object.get("linear_velocity_world_m_s"), label="replay velocity"),
                ),
            )
            rows += 1
    if rows == 0:
        raise ValueError("source/replay trace is empty")
    return {
        "rows": rows,
        "translation_max_m": translation_max,
        "rotation_max_deg": rotation_max,
        "object_velocity_max_m_s": velocity_max,
    }


def _render_set_sha256(run_root: Path) -> str | None:
    rows = []
    for root_name in ("head", "external"):
        for path in sorted((run_root / root_name).glob("**/*.png")):
            rows.append({"path": str(path.relative_to(run_root)), "sha256": _sha256(path)})
    return _canonical_json_sha256(rows) if rows else None


def _source_machine_id(source_root: Path) -> str:
    receipt = _read_json_if_present(source_root / "execution_receipt.json")
    return str(receipt.get("machine_id") or "UNKNOWN_SOURCE_MACHINE")


def _write_replay_report(
    report: Mapping[str, Any],
    output_root: Path,
    episode_id: str,
    source_trace: Path,
    run_root: Path,
) -> None:
    episode_root = output_root.resolve() / "episodes" / episode_id
    destinations = {episode_root, run_root}
    try:
        source_trace.parent.relative_to(output_root.resolve())
    except ValueError:
        pass
    else:
        destinations.add(source_trace.parent)
    fresh_root = episode_root / f"{STAGE_ORDER['replay']}_replay" / "fresh_process"
    rerender_root = episode_root / f"{STAGE_ORDER['rerender']}_rerender" / "same_trace"
    for candidate in (fresh_root, rerender_root):
        if candidate.is_dir():
            destinations.add(candidate)
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "replay_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )


def encode_videos(
    episode_id: str,
    output_root: Path = OUTPUT_ROOT,
    *,
    stage: str = "integrated",
) -> dict[str, Any]:
    if stage not in UNITY_EXECUTION_STAGES:
        raise ValueError("unknown ordered gate stage")
    run_root = output_root.resolve() / "episodes" / episode_id / f"{STAGE_ORDER[stage]}_{stage}"
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
