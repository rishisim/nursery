#!/usr/bin/env python3
"""Generate one LTX-2.3 comparison from the frozen public episode prompt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scripts.run_synthetic_video_prototype import PrototypeError, canonical, digest, private_write, probe, remux_tts


MODEL = "fal-ai/ltx-2.3/text-to-video"
QUEUE_ROOT = "https://queue.fal.run"
PROMPT_COMMITMENT = "63e20f20c29cf6d88116172ec245a10b788753581fc4643e773104f7eb6a71cb"


def fal_key() -> str:
    key = os.environ.get("FAL_KEY", "").strip()
    if not key:
        raise PrototypeError("E_FAL_CREDENTIAL_UNAVAILABLE")
    return key


def verify_plan(plan: dict) -> None:
    declared = plan.get("commitment_sha256")
    unsigned = {key: value for key, value in plan.items() if key != "commitment_sha256"}
    if declared != PROMPT_COMMITMENT or digest(unsigned) != PROMPT_COMMITMENT:
        raise PrototypeError("E_FROZEN_PROMPT_COMMITMENT")
    if plan.get("duration") != 10 or plan.get("aspect_ratio") != "16:9" or plan.get("generate_audio") is not False:
        raise PrototypeError("E_FROZEN_PLAN_SHAPE")
    prompt = plan.get("prompt")
    if not isinstance(prompt, str) or not 1 <= len(prompt) <= 5000:
        raise PrototypeError("E_FROZEN_PROMPT_LENGTH")


def ltx_payload(plan: dict) -> dict:
    verify_plan(plan)
    return {
        "prompt": plan["prompt"],
        "duration": 10,
        "resolution": "1080p",
        "aspect_ratio": "16:9",
        "fps": 24,
        "generate_audio": False,
    }


def request_json(url: str, key: str, *, payload: dict | None = None, timeout: int = 180) -> dict:
    headers = {"Authorization": f"Key {key}", "User-Agent": "nursery-public-ltx-comparison/1"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = canonical(payload)
    try:
        with urlopen(Request(url, data=data, headers=headers), timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        raise PrototypeError(f"E_FAL_HTTP_{exc.code}") from None


def download(url: str, target: Path) -> None:
    try:
        with urlopen(Request(url, headers={"User-Agent": "nursery-public-ltx-comparison/1"}), timeout=300) as response:
            target.write_bytes(response.read())
    except (HTTPError, URLError):
        raise PrototypeError("E_FAL_VIDEO_DOWNLOAD") from None
    os.chmod(target, 0o600)


def generate(plan: dict, root: Path, timeout_seconds: int) -> tuple[Path, bool]:
    raw = root / "synthetic_raw.mp4"
    if raw.is_file():
        return raw, False

    key = fal_key()
    job_path = root / "video_job.json"
    if job_path.is_file():
        job = json.loads(job_path.read_text())
        if job.get("model") != MODEL or job.get("prompt_commitment_sha256") != PROMPT_COMMITMENT:
            raise PrototypeError("E_FAL_JOB_PROVENANCE")
        submitted_now = False
    else:
        job = request_json(f"{QUEUE_ROOT}/{MODEL}", key, payload=ltx_payload(plan))
        request_id = job.get("request_id")
        if not request_id:
            raise PrototypeError("E_FAL_SUBMISSION")
        job = {
            "model": MODEL,
            "prompt_commitment_sha256": PROMPT_COMMITMENT,
            "request_id": request_id,
            "status_url": job.get("status_url") or f"{QUEUE_ROOT}/{MODEL}/requests/{request_id}/status",
            "response_url": job.get("response_url") or f"{QUEUE_ROOT}/{MODEL}/requests/{request_id}",
            "submission_count": 1,
        }
        private_write(job_path, job)
        submitted_now = True

    if job.get("submission_count") != 1:
        raise PrototypeError("E_FAL_SUBMISSION_COUNT")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            status = request_json(job["status_url"], key, timeout=60)
        except URLError:
            time.sleep(10)
            continue
        state = status.get("status")
        if state == "COMPLETED":
            result = request_json(job["response_url"], key, timeout=180)
            video_url = result.get("video", {}).get("url")
            if not video_url:
                raise PrototypeError("E_FAL_RESULT")
            download(video_url, raw)
            return raw, submitted_now
        if state not in {"IN_QUEUE", "IN_PROGRESS"}:
            raise PrototypeError("E_FAL_GENERATION_FAILED")
        time.sleep(10)
    raise PrototypeError("E_FAL_GENERATION_TIMEOUT")


def assert_ignored(path: Path) -> None:
    result = subprocess.run(["git", "check-ignore", "-q", "--", str(path)], check=False)
    if result.returncode != 0:
        raise PrototypeError("E_OUTPUT_ROOT_NOT_IGNORED")


def run(plan_path: Path, output_root: Path, timeout_seconds: int) -> dict:
    assert_ignored(output_root)
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    plan = json.loads(plan_path.read_text())
    verify_plan(plan)
    raw, submitted_now = generate(plan, output_root, timeout_seconds)
    final = output_root / "synthetic_10s.mp4"
    if not final.is_file():
        final = remux_tts(raw, plan, output_root)
    media = probe(final)
    checks = {
        "valid_10_second_decode": abs(media["duration"] - 10) <= 0.08,
        "1920x1080_video": media["width"] == 1920 and media["height"] == 1080,
        "synchronized_audio_track": media["audio"],
        "exact_frozen_prompt_commitment": True,
        "one_submission_only": json.loads((output_root / "video_job.json").read_text())["submission_count"] == 1,
    }
    if not all(checks.values()):
        raise PrototypeError("E_LTX_MEDIA_VALIDATION")
    result = {
        "schema_version": 1,
        "status": "EXPLORATORY_LTX_CLIP_COMPLETE",
        "public_only": True,
        "provider": "fal.ai",
        "generator_model": MODEL,
        "accepted_synthetic_clip_count": 1,
        "accepted_seconds": 10,
        "submission_count": 1,
        "submitted_in_this_invocation": submitted_now,
        "prompt_commitment_sha256": PROMPT_COMMITMENT,
        "manual_prompt_edit": False,
        "native_generated_audio": False,
        "digital_silence": media["digital_silence"],
        "checks": checks,
        "request_cost_usd": 0.8,
        "claim_limits": ["one_public_clip_only", "exploratory_generator_comparison", "no_equivalence", "no_scale_up_authorization"],
    }
    private_write(output_root / "compact_result.json", result)
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("outputs/public_single_clip_api/episode_plan.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/public_single_clip_ltx_api"))
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    run(args.plan, args.output_root, args.timeout_seconds)


if __name__ == "__main__":
    main()
