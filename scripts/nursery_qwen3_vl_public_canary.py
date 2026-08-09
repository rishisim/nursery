#!/usr/bin/env python3
"""Public synthetic, network-denied Qwen3-VL five-frame schema canary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import time

import mlx.core as mx
from mlx_vlm import apply_chat_template, generate, load
from PIL import Image, ImageDraw


MODEL = Path("/Users/rishisim/Library/Application Support/ChildLens Instruments/provisional-calibration-v1/models/Qwen3-VL-8B-Instruct-4bit")
REVISION = "defcdea7cc7a4b0858fea563cbbce171d328e457"
OUTPUT = Path("/Users/rishisim/Documents/research/nursery/output/nursery_program_convergence_v1/qwen3_vl_public_canary.json")
ALLOWED = {
    "candidate_count_bin": {"zero", "one", "two", "three_or_more", "abstain"},
    "visibility_bin": {"none", "partial", "clear", "abstain"},
    "referential_status": {"visible_candidate", "null_or_irrelevant", "ambiguous", "undecidable"},
    "lexical_support": {"noun_object", "verb_action", "both", "neither", "undecidable"},
    "lag_event_unit_bin": {"lead_two_plus", "lead_one", "overlap", "lag_one", "lag_two_plus", "undecidable"},
}


def network_is_denied() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=0.5):
            return False
    except OSError:
        return True


def tree_manifest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total = 0
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or ".cache" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        member = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                member.update(block)
        size = path.stat().st_size
        digest.update(f"{relative}\0{size}\0{member.hexdigest()}\n".encode())
        total += size
        count += 1
    if not count:
        raise RuntimeError("E_EMPTY_MODEL")
    return digest.hexdigest(), total, count


def fixtures(directory: Path) -> list[str]:
    paths: list[str] = []
    for index in range(5):
        image = Image.new("RGB", (320, 240), (235, 235, 235))
        draw = ImageDraw.Draw(image)
        x = 36 + index * 40
        draw.rectangle((x, 80, x + 64, 144), fill=(220, 30, 30))
        draw.ellipse((225, 82, 285, 142), fill=(35, 75, 200))
        path = directory / f"frame-{index}.png"
        image.save(path)
        paths.append(str(path))
    return paths


def parse_exact(text: str) -> dict[str, str]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines)
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or set(parsed) != set(ALLOWED):
        raise RuntimeError("E_SCHEMA_KEYS")
    for key, allowed in ALLOWED.items():
        if parsed[key] not in allowed:
            raise RuntimeError("E_SCHEMA_VALUE")
    return parsed


def execute() -> dict[str, object]:
    if not network_is_denied():
        raise RuntimeError("E_NETWORK_NOT_DENIED")
    artifact_digest, artifact_bytes, artifact_files = tree_manifest(MODEL)
    mx.reset_peak_memory()
    started = time.perf_counter()
    model, processor = load(str(MODEL), lazy=False, strict=True)
    with tempfile.TemporaryDirectory(prefix="nursery-qwen3-vl-public-") as temporary:
        images = fixtures(Path(temporary))
        instruction = (
            "Five synthetic frames show a red square moving while one blue circle remains visible. "
            "The German noun phrase is 'das rote Quadrat'. Return only one JSON object with exactly "
            "these keys and allowed values: candidate_count_bin [zero,one,two,three_or_more,abstain]; "
            "visibility_bin [none,partial,clear,abstain]; referential_status "
            "[visible_candidate,null_or_irrelevant,ambiguous,undecidable]; lexical_support "
            "[noun_object,verb_action,both,neither,undecidable]; lag_event_unit_bin "
            "[lead_two_plus,lead_one,overlap,lag_one,lag_two_plus,undecidable]. "
            "candidate_count_bin counts de-duplicated noun/verb referential candidates for the phrase, "
            "not every visible shape. No prose or markdown."
        )
        prompt = apply_chat_template(
            processor,
            model.config,
            instruction,
            num_images=len(images),
            enable_thinking=False,
        )
        response = generate(
            model,
            processor,
            prompt,
            image=images,
            max_tokens=192,
            temperature=0.0,
            verbose=False,
            enable_thinking=False,
        )
        parsed = parse_exact(response.text)
    elapsed = time.perf_counter() - started
    semantic = (
        # The frozen bridge counts de-duplicated lexical/referential candidates,
        # not all visible objects.  The phrase names only the red square.
        parsed["candidate_count_bin"] == "one"
        and parsed["visibility_bin"] == "clear"
        and parsed["referential_status"] == "visible_candidate"
        and parsed["lexical_support"] in {"noun_object", "both"}
    )
    return {
        "schema_version": "nursery-qwen3-vl-public-canary-v1",
        "status": "PASS" if semantic else "FAIL_SEMANTIC_CANARY",
        "scope": "SELF_GENERATED_FIVE_FRAME_FIXTURE_ONLY",
        "childlens_or_quarantine_accessed": False,
        "hosted_or_cloud_inference_used": False,
        "network_denial_sentinel_passed": True,
        "revision": REVISION,
        "artifact_manifest_sha256": artifact_digest,
        "artifact_bytes": artifact_bytes,
        "artifact_file_count": artifact_files,
        "exact_schema_valid": True,
        "synthetic_semantic_fixture_passed": semantic,
        "five_frames_consumed": True,
        "wall_time_seconds": round(elapsed, 3),
        "peak_mlx_memory_gib": round(mx.get_peak_memory() / 1024**3, 3),
        "pseudo_output_is_human_evidence": False,
        "restricted_inference_authorized_by_canary": False,
    }


def main() -> int:
    try:
        result = execute()
    except Exception as exc:
        result = {
            "schema_version": "nursery-qwen3-vl-public-canary-v1",
            "status": "FAIL",
            "failure_class": type(exc).__name__,
            "childlens_or_quarantine_accessed": False,
            "hosted_or_cloud_inference_used": False,
            "pseudo_output_is_human_evidence": False,
            "restricted_inference_authorized_by_canary": False,
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pending = OUTPUT.with_suffix(".pending")
    pending.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(pending, 0o600)
    os.replace(pending, OUTPUT)
    print(result["status"])
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
