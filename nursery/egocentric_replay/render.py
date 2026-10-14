"""Launch Blender with the existing EmbodiedGen URDF transform helpers."""

import json
from pathlib import Path
import tempfile
import subprocess

from ..embodiedgen_scene import _BLENDER_URDF_PREVIEW_SCRIPT, _run_native


REPO = Path(__file__).resolve().parents[2]


def render_job(job, config, output):
    output = Path(output).absolute()
    parent = output
    while not parent.exists():
        parent = parent.parent
    if subprocess.run(["git", "-C", str(parent), "check-ignore", "--quiet", str(output)],
                      check=False).returncode:
        raise ValueError("Render output must be inside an ignored repository directory")
    # Reuse the URDF loader's actual helper definitions, including joint traversal.
    prelude = _BLENDER_URDF_PREVIEW_SCRIPT.split('scene = bpy.context.scene')[0]
    with tempfile.TemporaryDirectory(prefix="nursery-replay-") as temporary:
        temporary = Path(temporary)
        script = temporary / "render.py"
        script.write_text(prelude + '\n' + Path(__file__).with_name("blender.py").read_text())
        payload = temporary / "job.json"
        payload.write_text(json.dumps({**job, "config": config, "output": str(output)}))
        return _run_native([config["blender"], "-b", "--factory-startup", "-t", "1", "--python-exit-code", "1", "-P", str(script),
                            "--", job["room_urdf"], str(output), str(payload)],
                           cwd=REPO, operation="Egocentric replay rendering")
