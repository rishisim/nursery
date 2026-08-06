"""Stage and run the ignored Unity project from tracked reference-hand sources."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "runs/embodied_simulation/reference_hand_microcell"
PROJECT_ROOT = RUN_ROOT / "project"
OUTPUT_ROOT = RUN_ROOT / "qualification"
PACKAGE_ROOT = ROOT / ".external/ultraleap-unityplugin-7.3.0-artifact/Packages/Tracking"
EDITOR_DEFAULT = Path(
    "/Users/rishisim/.codex/worktrees/9122/nursery/.external/unity-editors/"
    "6000.0.80f1/Unity.app/Contents/MacOS/Unity"
)


def _assert_ignored(path: Path) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path.relative_to(ROOT))],
        cwd=ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"generated path is not ignored: {path}")


def stage() -> None:
    _assert_ignored(RUN_ROOT)
    (PROJECT_ROOT / "Assets/ReferenceHand").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "Assets/Editor").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "Packages").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "ProjectSettings").mkdir(parents=True, exist_ok=True)
    for obsolete in (
        PROJECT_ROOT / "Assets/ReferenceHand/ReferenceHandMicrocell.cs",
        PROJECT_ROOT / "Assets/Editor/ReferenceHandMicrocellBuilder.cs",
    ):
        if obsolete.exists():
            obsolete.unlink()
    for name in ("SyntheticLeapProvider.cs", "ReferenceHandQualification.cs"):
        shutil.copy2(SOURCE_ROOT / name, PROJECT_ROOT / "Assets/ReferenceHand" / name)
    shutil.copy2(
        SOURCE_ROOT / "ReferenceHandMicrocellBuilder.cs",
        PROJECT_ROOT / "Assets/Editor/ReferenceHandMicrocellBuilder.cs",
    )
    manifest = {
        "dependencies": {
            "com.ultraleap.tracking": "file:../../../../../.external/ultraleap-unityplugin-7.3.0-artifact/Packages/Tracking",
            "com.unity.collab-proxy": "2.4.3",
            "com.unity.ide.rider": "3.0.31",
            "com.unity.ide.vscode": "1.2.5",
            "com.unity.test-framework": "1.6.0",
            "com.unity.textmeshpro": "3.0.6",
            "com.unity.timeline": "1.7.6",
            "com.unity.ugui": "2.0.0",
            "com.unity.modules.physics": "1.0.0",
            "com.unity.modules.physics2d": "1.0.0",
            "com.unity.modules.imageconversion": "1.0.0",
            "com.unity.modules.jsonserialize": "1.0.0",
            "com.unity.modules.video": "1.0.0",
            "com.unity.modules.xr": "1.0.0",
            "com.unity.modules.vr": "1.0.0",
        }
    }
    (PROJECT_ROOT / "Packages/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (PROJECT_ROOT / "ProjectSettings/ProjectVersion.txt").write_text(
        "m_EditorVersion: 6000.0.80f1\nm_EditorVersionWithRevision: 6000.0.80f1 (2dfd32957da2)\n"
    )


def unity_path(value: str | None) -> Path:
    candidate = Path(value or os.environ.get("UNITY_EDITOR", EDITOR_DEFAULT))
    if not candidate.is_file():
        raise FileNotFoundError(f"Unity editor not found: {candidate}")
    return candidate


def build(editor: Path) -> None:
    stage()
    log = RUN_ROOT / "qualification_build.log"
    command = [
        str(editor),
        "-batchmode",
        "-nographics",
        "-quit",
        "-projectPath",
        str(PROJECT_ROOT),
        "-executeMethod",
        "EmbodiedReferenceHand.ReferenceHandMicrocellBuilder.BuildPlayer",
    ]
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run_player(visual: bool = False) -> None:
    _assert_ignored(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    player = RUN_ROOT / "ReferenceHandQualification.app/Contents/MacOS/Reference Hand Microcell"
    if not player.is_file():
        raise FileNotFoundError(f"player not found: {player}; run build first")
    log = RUN_ROOT / "qualification_player.log"
    env = dict(os.environ)
    env["REFERENCE_HAND_OUTPUT"] = str(OUTPUT_ROOT)
    if visual:
        env["REFERENCE_HAND_VISUAL"] = "1"
    command = [str(player), "-batchmode"]
    if not visual:
        command.append("-nographics")
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode not in (0, 3):
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("stage", "build", "run", "build-and-run"))
    parser.add_argument("--unity")
    parser.add_argument("--visual", action="store_true")
    args = parser.parse_args()
    if args.command == "stage":
        stage()
    elif args.command == "build":
        build(unity_path(args.unity))
    elif args.command == "run":
        run_player(args.visual)
    else:
        build(unity_path(args.unity))
        run_player(args.visual)


if __name__ == "__main__":
    main()
