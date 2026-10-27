"""Shared repository paths, configuration and output-directory guard."""

import json
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]


def load_config():
    """Speech (generator, cues, vocabulary) plus learner (upstream, training, konkle) settings in one dict."""
    speech = json.loads((REPO / "nursery" / "speech" / "config.json").read_text())
    learner = json.loads((REPO / "nursery" / "learner" / "config.json").read_text())
    if speech["schema_version"] != 1 or learner["schema_version"] != 1:
        raise ValueError("Unsupported cue-learning configuration")
    return {"schema_version": 1, "upstream": learner["upstream"], "generator": speech["generator"],
            "cues": speech["cues"], "training": learner["training"], "vocabulary": speech["vocabulary"],
            "konkle": learner["konkle"]}


def _output_directory(path, *, exist_ok=True):
    path = Path(path).expanduser().resolve()
    parent = path
    while not parent.exists():
        parent = parent.parent
    ignored = subprocess.run(
        ["git", "-C", str(parent), "check-ignore", "--quiet", str(path)], check=False,
    )
    if ignored.returncode != 0:
        raise ValueError("Cue outputs must be inside an ignored repository directory")
    path.mkdir(parents=True, exist_ok=exist_ok)
    return path
