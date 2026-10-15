"""Training-only embodied cues for the pinned vlm-headcam learner."""

import importlib
from pathlib import Path
import subprocess
import sys

from nursery.common import REPO, load_config


UPSTREAM = REPO / ".external" / "vlm-headcam"


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
