#!/usr/bin/env python3
"""Public synthetic, network-denied smoke for both v1.3 adapter modes."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_childlens_model_assisted_pseudo_annotation_v1_3.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("childlens_v13_runner_for_smoke", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("E_SMOKE_IMPORT")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    runner = _load()
    directory = Path(tempfile.mkdtemp(prefix="childlens-v13-public-synthetic-"))
    os.chmod(directory, 0o700)
    try:
        runner.prepare_instrument_seal()
        backend = runner.firewall.NetworkIsolationBackend.detect()
        runner.firewall.verify_network_isolation(backend)
        media = directory / "synthetic.mp4"
        completed = subprocess.run(
            [
                str(runner.FFMPEG),
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=green:s=224x224:r=2:d=3",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=16000:duration=3",
                "-shortest",
                "-c:v",
                "h264",
                "-c:a",
                "aac",
                "-y",
                str(media),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            close_fds=True,
            timeout=60,
        )
        if completed.returncode != 0:
            raise RuntimeError("E_SMOKE_MEDIA")
        os.chmod(media, 0o600)
        item = runner.PilotItem(
            hashlib.sha256(media.read_bytes()).hexdigest(), media, ((0.0, 3.0),), 3.0
        )
        windows = [{"start_seconds": 0.0, "end_seconds": 3.0}]
        audio = directory / "audio.json"
        runner._invoke_adapter(
            backend=backend,
            mode="audio",
            item=item,
            job={
                "schema_version": "childlens-restricted-audio-job-v1.3.0",
                "windows": windows,
            },
            output=audio,
            scratch_root=directory,
        )
        runner._validate_audio_output(audio, 1)
        vlm = directory / "vlm.json"
        runner._invoke_adapter(
            backend=backend,
            mode="vlm",
            item=item,
            job={
                "schema_version": "childlens-restricted-vlm-job-v1.3.0",
                "windows": windows,
            },
            output=vlm,
            scratch_root=directory,
            audio_output=audio,
        )
        runner._validate_vlm_output(vlm, 1)
        print("CHILDLENS_V13_PUBLIC_SYNTHETIC_ADAPTER_SMOKE_PASS")
        return 0
    except Exception:
        print("E_PUBLIC_SYNTHETIC_ADAPTER_SMOKE")
        return 1
    finally:
        shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
