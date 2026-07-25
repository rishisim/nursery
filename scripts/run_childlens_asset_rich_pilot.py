#!/usr/bin/env python3
"""Render the prospective pilot and mux event-derived German audio."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLENDER = Path("/Users/rishisim/blender/blender-4.2.1-macos-arm64/Blender.app/Contents/MacOS/Blender")
ASSETS = ROOT / "tmp/childlens_asset_rich/assets"
OUTPUT = ROOT / "tmp/childlens_asset_rich/pilots"
SCENE = ROOT / "babyworld_lite/childlens_asset_rich/pilot_scene.py"
EPISODES = {
    "playroom_duck_push": "Schau, das ist Dax. Dax ist die Ente. Jetzt berührt die Hand Dax. Jetzt schiebt die Hand Dax. Schau, Dax ist dort.",
    "tabletop_apple_lift": "Schau, das ist Koba. Koba ist der Apfel. Jetzt berührt die Hand Koba. Jetzt hebt die Hand Koba. Schau, Koba ist oben.",
    "livingroom_ball_roll": "Schau, das ist Mipa. Mipa ist der Ball. Jetzt berührt die Hand Mipa. Jetzt rollt die Hand Mipa. Schau, Mipa ist dort.",
}


def run(command):
    subprocess.run([str(item) for item in command], check=True)


def main() -> None:
    for episode, transcript in EPISODES.items():
        destination = OUTPUT / episode
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "transcript.txt").write_text(transcript + "\n")
        run([BLENDER, "--background", "--python", SCENE, "--", episode, ASSETS, OUTPUT])
        aiff = destination / "speech.aiff"
        wav = destination / "speech.wav"
        run(["say", "-v", "Anna", "-r", "175", "-o", aiff, transcript])
        run(["ffmpeg", "-y", "-i", aiff, "-af", "apad=pad_dur=20", "-t", "20",
             "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", wav])
        run(["ffmpeg", "-y", "-i", destination / "rgb_no_audio.mp4", "-i", wav,
             "-c:v", "copy", "-c:a", "aac", "-shortest", destination / "pilot.mp4"])
        aiff.unlink()
    print(json.dumps({"episodes": sorted(EPISODES), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
