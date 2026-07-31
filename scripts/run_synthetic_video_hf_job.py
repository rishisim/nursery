#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface-hub==1.26.0",
#   "piper-tts==1.6.0",
# ]
# ///
"""Run one public-only family on Hugging Face Jobs and persist every attempt."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

OFFLINE_SAFE_ENV = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "WANDB_DISABLED": "true",
    "WANDB_MODE": "disabled",
    "COMET_DISABLE_AUTO_LOGGING": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def run(
    argv: list[str] | tuple[str, ...],
    *,
    cwd: Path | None = None,
    capture: bool = False,
    output_log: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **OFFLINE_SAFE_ENV}
    if output_log:
        output_log.parent.mkdir(parents=True, exist_ok=True)
        with output_log.open("w") as handle:
            return subprocess.run(
                list(argv),
                cwd=cwd,
                env=env,
                check=True,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def clone_at(url: str, revision: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "init", str(destination)])
    run(["git", "-C", str(destination), "remote", "add", "origin", url])
    run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", revision])
    run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"])


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    if not shutil.which("apt-get"):
        raise RuntimeError("ffmpeg and ffprobe are missing and apt-get is unavailable")
    run(["apt-get", "update"])
    run(["apt-get", "install", "-y", "--no-install-recommends", "ffmpeg"])


def nvidia_query(fields: str) -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


class PeakGpuMonitor:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_memory_mib: int | None = None
        self.peak_utilization_percent: int | None = None

    def start(self) -> None:
        if not shutil.which("nvidia-smi"):
            return
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(1.0):
            raw = nvidia_query("memory.used,utilization.gpu")
            if not raw:
                continue
            for line in raw.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 2:
                    continue
                try:
                    memory, utilization = (int(float(field)) for field in fields)
                except ValueError:
                    continue
                self.peak_memory_mib = max(self.peak_memory_mib or 0, memory)
                self.peak_utilization_percent = max(self.peak_utilization_percent or 0, utilization)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)


def write_json(path: Path, value: Any, canonical_json_bytes: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def setup_wan(
    config: dict[str, Any],
    external_root: Path,
    patch_wan_sdpa_fallback: Any,
) -> tuple[Path, Path, str, list[str], list[dict[str, str]]]:
    model = config["models"]["wan"]
    source = external_root / "Wan2.2"
    clone_at(model["source_repository"], model["source_commit"], source)
    runtime_source_adaptations = [patch_wan_sdpa_fallback(source)]
    weights = external_root / "weights" / "wan"
    snapshot_download(
        repo_id=model["weights_repository"],
        revision=model["weights_revision"],
        local_dir=weights,
        token=os.environ["HF_TOKEN"],
    )
    environment = external_root / "envs" / "wan"
    run(["uv", "venv", "--python", "3.12", str(environment)])
    requirements = []
    for line in (source / "requirements.txt").read_text().splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or clean.lower().startswith("flash_attn"):
            continue
        requirements.append(clean)
    run(["uv", "pip", "install", "--python", str(environment / "bin" / "python"), *requirements])
    freeze = run(
        ["uv", "pip", "freeze", "--python", str(environment / "bin" / "python")],
        capture=True,
    ).stdout.splitlines()
    return (
        source,
        weights,
        str(environment / "bin" / "python"),
        freeze,
        runtime_source_adaptations,
    )


def setup_ltx(
    config: dict[str, Any], external_root: Path
) -> tuple[Path, Path, str, list[str], list[dict[str, str]]]:
    model = config["models"]["ltx"]
    source = external_root / "LTX-2"
    clone_at(model["source_repository"], model["source_commit"], source)
    run(["uv", "sync", "--frozen"], cwd=source, output_log=external_root / "ltx_uv_sync.log")
    weights = external_root / "weights" / "ltx"
    snapshot_download(
        repo_id=model["weights_repository"],
        revision=model["weights_revision"],
        local_dir=weights,
        allow_patterns=[model["checkpoint_filename"], model["spatial_upsampler_filename"]],
        token=os.environ["HF_TOKEN"],
    )
    snapshot_download(
        repo_id=model["text_encoder_repository"],
        revision=model["text_encoder_revision"],
        local_dir=weights / "gemma",
        token=os.environ["HF_TOKEN"],
    )
    freeze = run(
        ["uv", "pip", "freeze", "--python", str(source / ".venv" / "bin" / "python")],
        capture=True,
    ).stdout.splitlines()
    return source, weights, sys.executable, freeze, []


def setup_tts(config: dict[str, Any], external_root: Path) -> Path:
    tts = config["tts"]
    root = external_root / "weights" / "piper"
    snapshot_download(
        repo_id=tts["voice_repository"],
        revision=tts["voice_revision"],
        local_dir=root,
        allow_patterns=[tts["model_path"], tts["config_path"], tts["model_card_path"]],
        token=os.environ["HF_TOKEN"],
    )
    return root / tts["model_path"]


def upload_file(api: HfApi, repo_id: str, local: Path, remote: str) -> None:
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=remote,
        repo_id=repo_id,
        repo_type="dataset",
    )


def persist_control_files(api: HfApi, repo_id: str, run_id: str, family: str, family_root: Path) -> None:
    prefix = f"{run_id}/families/{family}"
    for filename in ("work_order.json", "environment.json", "run_status.json"):
        path = family_root / filename
        if path.exists():
            upload_file(api, repo_id, path, f"{prefix}/{filename}")
    gallery = family_root / "gallery" / "index.html"
    if gallery.exists():
        upload_file(api, repo_id, gallery, f"{prefix}/gallery/index.html")


def persist_attempt(
    api: HfApi,
    repo_id: str,
    run_id: str,
    family: str,
    attempt_root: Path,
) -> None:
    api.upload_folder(
        folder_path=str(attempt_root),
        path_in_repo=f"{run_id}/families/{family}/attempts/{attempt_root.name}",
        repo_id=repo_id,
        repo_type="dataset",
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--nursery-repository", default="https://github.com/rishisim/nursery.git")
    result.add_argument("--nursery-revision", required=True)
    result.add_argument("--expected-protocol-sha256", required=True)
    result.add_argument("--family", choices=("wan", "ltx"), required=True)
    result.add_argument("--profile", choices=("cloud_preflight", "preview"), required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--output-repository", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN is required as an encrypted job secret")
    job_monotonic_started = time.monotonic()
    started_at = now()
    task_root = Path("/tmp") / f"nursery-{args.run_id}-{args.family}"
    source_root = task_root / "nursery"
    family_root = task_root / "output" / args.run_id / "families" / args.family
    external_root = task_root / "external"
    clone_at(args.nursery_repository, args.nursery_revision, source_root)
    sys.path.insert(0, str(source_root / "src"))

    from nursery_egobaby_preflight.contract import (
        canonical_json_bytes,
        canonical_json_sha256,
    )
    from nursery_egobaby_preflight.synthetic_video_pilot import (
        build_model_command,
        compile_work_order,
        complete_attempt_record,
        load_config,
        media_summary,
        mux_modular_audio,
        patch_wan_sdpa_fallback,
        render_gallery,
        synthesize_tts,
        utc_now,
        write_work_order,
    )

    config_path = source_root / "configs" / "synthetic_video_public_pilot.json"
    config = load_config(config_path)
    protocol_sha256 = canonical_json_sha256(config)
    if protocol_sha256 != args.expected_protocol_sha256:
        raise RuntimeError("frozen protocol hash mismatch")

    work_order = compile_work_order(config, args.profile, args.run_id, family=args.family)
    write_work_order(family_root, config, work_order)
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(
        repo_id=args.output_repository,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    environment_record = {
        "schema_version": 1,
        "started_at": started_at,
        "provider": "hugging_face_jobs",
        "job_id": os.environ.get("HF_JOB_ID"),
        "family": args.family,
        "profile": args.profile,
        "nursery_revision": args.nursery_revision,
        "protocol_sha256": protocol_sha256,
        "gpu": nvidia_query("name,memory.total,driver_version"),
        "resolved_python_packages": [],
    }
    write_json(family_root / "environment.json", environment_record, canonical_json_bytes)
    status = {
        "schema_version": 1,
        "run_id": args.run_id,
        "family": args.family,
        "profile": args.profile,
        "status": "compiled",
        "completed_attempt_ids": [],
        "failed_attempt_ids": [],
        "started_at": started_at,
        "finished_at": None,
    }
    write_json(family_root / "run_status.json", status, canonical_json_bytes)
    persist_control_files(api, args.output_repository, args.run_id, args.family, family_root)

    if not config["profiles"][args.profile]["execute_generation"]:
        status["status"] = "preflight_pass"
        status["finished_at"] = now()
        write_json(family_root / "run_status.json", status, canonical_json_bytes)
        persist_control_files(api, args.output_repository, args.run_id, args.family, family_root)
        print(json.dumps({"status": "PREFLIGHT_PASS", "run_id": args.run_id, "family": args.family}))
        return

    ensure_ffmpeg()
    if args.family == "wan":
        model_source, weights_root, model_python, freeze, runtime_source_adaptations = setup_wan(
            config,
            external_root,
            patch_wan_sdpa_fallback,
        )
    else:
        model_source, weights_root, model_python, freeze, runtime_source_adaptations = setup_ltx(
            config,
            external_root,
        )
    voice_model = setup_tts(config, external_root)
    environment_record["resolved_python_packages"] = freeze
    environment_record["runtime_source_adaptations"] = runtime_source_adaptations
    environment_record["model_source_commit"] = config["models"][args.family]["source_commit"]
    environment_record["model_weights_revision"] = config["models"][args.family]["weights_revision"]
    environment_record["tts_voice_revision"] = config["tts"]["voice_revision"]
    write_json(family_root / "environment.json", environment_record, canonical_json_bytes)
    persist_control_files(api, args.output_repository, args.run_id, args.family, family_root)

    status["status"] = "running"
    write_json(family_root / "run_status.json", status, canonical_json_bytes)
    persist_control_files(api, args.output_repository, args.run_id, args.family, family_root)
    active_attempt_id: str | None = None
    active_generator_log: Path | None = None
    try:
        for attempt in work_order["attempts"]:
            active_attempt_id = attempt["attempt_id"]
            attempt_root = family_root / "attempts" / attempt["attempt_id"]
            raw_video = attempt_root / "raw.mp4"
            video_only = attempt_root / "video_only.mp4"
            tts_audio = attempt_root / "utterance.wav"
            final_video = attempt_root / "final.mp4"
            attempt_started = utc_now()
            monotonic_started = time.monotonic()
            monitor = PeakGpuMonitor()
            monitor.start()
            try:
                active_generator_log = attempt_root / "generator.log"
                command = build_model_command(
                    config,
                    attempt,
                    source_root=model_source,
                    weights_root=weights_root,
                    raw_output=raw_video,
                    python_executable=model_python,
                    uv_executable="uv",
                )
                run(command.argv, cwd=Path(command.cwd), output_log=active_generator_log)
            except Exception:
                if active_attempt_id not in status["failed_attempt_ids"]:
                    status["failed_attempt_ids"].append(active_attempt_id)
                if active_generator_log.exists():
                    log_tail = active_generator_log.read_text(errors="replace")[-16000:]
                    print("Generator log tail follows:", file=sys.stderr)
                    print(log_tail, file=sys.stderr)
                    try:
                        upload_file(
                            api,
                            args.output_repository,
                            active_generator_log,
                            (
                                f"{args.run_id}/families/{args.family}/attempts/"
                                f"{active_attempt_id}/generator.log"
                            ),
                        )
                    except Exception as upload_error:
                        print(
                            f"Could not persist generator log: {upload_error}",
                            file=sys.stderr,
                        )
                raise
            finally:
                monitor.stop()
            generation_seconds = time.monotonic() - monotonic_started
            synthesize_tts(
                config,
                attempt["scene_id"],
                voice_model=voice_model,
                output_wav=tts_audio,
                python_executable=sys.executable,
            )
            mux_modular_audio(
                config,
                attempt["scene_id"],
                raw_video=raw_video,
                tts_wav=tts_audio,
                video_only=video_only,
                final_video=final_video,
            )
            raw = media_summary(raw_video)
            final = media_summary(final_video)
            record = complete_attempt_record(
                config,
                attempt,
                started_at=attempt_started,
                finished_at=utc_now(),
                raw_summary=raw,
                final_summary=final,
                runtime={
                    "generation_wall_seconds": generation_seconds,
                    "peak_gpu_memory_mib": monitor.peak_memory_mib,
                    "peak_gpu_utilization_percent": monitor.peak_utilization_percent,
                    "gpu": environment_record["gpu"],
                    "provider": "hugging_face_jobs",
                    "hardware_flavor": config["cloud"]["hardware_flavor"],
                    "hourly_price_usd_at_freeze": config["cloud"]["hourly_price_usd_at_freeze"],
                },
            )
            write_json(attempt_root / "attempt.json", record, canonical_json_bytes)
            if record["status"] != "media_valid":
                status["failed_attempt_ids"].append(attempt["attempt_id"])
                raise RuntimeError(f"media conformance failed for {attempt['attempt_id']}")
            status["completed_attempt_ids"].append(attempt["attempt_id"])
            write_json(family_root / "run_status.json", status, canonical_json_bytes)
            render_gallery(config, work_order, family_root)
            persist_attempt(api, args.output_repository, args.run_id, args.family, attempt_root)
            persist_control_files(api, args.output_repository, args.run_id, args.family, family_root)
    except Exception as error:
        status["status"] = "failed"
        status["finished_at"] = now()
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "active_attempt_id": active_attempt_id,
            "generator_log_tail": (
                active_generator_log.read_text(errors="replace")[-16000:]
                if active_generator_log is not None and active_generator_log.exists()
                else None
            ),
        }
        write_json(family_root / "failure.json", failure, canonical_json_bytes)
        write_json(family_root / "run_status.json", status, canonical_json_bytes)
        upload_file(
            api,
            args.output_repository,
            family_root / "failure.json",
            f"{args.run_id}/families/{args.family}/failure.json",
        )
        persist_control_files(api, args.output_repository, args.run_id, args.family, family_root)
        raise

    status["status"] = "complete"
    status["finished_at"] = now()
    status["job_wall_seconds"] = time.monotonic() - job_monotonic_started
    write_json(family_root / "run_status.json", status, canonical_json_bytes)
    render_gallery(config, work_order, family_root)
    persist_control_files(api, args.output_repository, args.run_id, args.family, family_root)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "run_id": args.run_id,
                "family": args.family,
                "completed_attempts": len(status["completed_attempt_ids"]),
                "output_repository": args.output_repository,
            }
        )
    )


if __name__ == "__main__":
    main()
