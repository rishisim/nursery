"""Training-only embodied cues for the pinned vlm-headcam learner."""

import importlib
import json
from pathlib import Path
import subprocess
import sys


DEFAULT_CONFIG = Path(__file__).with_name("config.json")
UPSTREAM = Path(__file__).resolve().parents[2] / ".external" / "vlm-headcam"


def load_config():
    config = json.loads(DEFAULT_CONFIG.read_text())
    if config["schema_version"] != 1:
        raise ValueError("Unsupported cue-learning configuration")
    return config


def _output_directory(path):
    path = Path(path).expanduser().resolve()
    parent = path
    while not parent.exists():
        parent = parent.parent
    ignored = subprocess.run(
        ["git", "-C", str(parent), "check-ignore", "--quiet", str(path)], check=False,
    )
    if ignored.returncode != 0:
        raise ValueError("Cue outputs must be inside an ignored repository directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _upstream_module(name):
    if not (UPSTREAM / "src" / "train_frame_mil.py").is_file():
        raise FileNotFoundError(f"Pinned vlm-headcam checkout absent: {UPSTREAM}")
    pin = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if pin != load_config()["upstream"]["commit"]:
        raise ValueError(f"vlm-headcam checkout is not at the configured pin: {pin}")
    src = str(UPSTREAM / "src")
    # Upstream uses sibling absolute imports; refuse unrelated modules with those names.
    for sibling in ("common", "train", "train_boot", "train_region_mil", "train_frame_mil", "embed_regions"):
        module = sys.modules.get(sibling)
        if module is not None and Path(module.__file__).resolve().parent != Path(src):
            raise ImportError(f"Module name conflicts with upstream: {sibling}")
    sys.path.insert(0, src)
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(src)
