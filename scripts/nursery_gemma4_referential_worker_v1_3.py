#!/usr/bin/env python3
"""Additive rendered-template-order correction for the frozen Gemma worker.

The declarative messages remain audio, five images, then exact user text. The
pinned processor renders the sole audio placeholder after the user text; this
adapter relocates only that token before the five contiguous image tokens and
then revalidates the entire ordering/text/count contract.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
LEGACY_WORKER = ROOT / "scripts/nursery_gemma4_referential_worker.py"
TEMPLATE_AMENDMENT = ROOT / "docs/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3/frozen_template_placeholder_order_amendment_v1_3.json"
LEGACY_WORKER_SHA256 = "6f6ad2f2605c94aec8f92cd927248367eef74e16222768cced3a4cc25c050a2f"
TEMPLATE_AMENDMENT_SHA256 = "2d117cf0f1619d43d6e71a90aa853241f0a5a7c331e84fc04af8231c5d6081fb"
AUDIO_TOKEN = "<|audio|>"
IMAGE_TOKEN = "<|image|>"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("E_MODULE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if _sha256_file(LEGACY_WORKER) != LEGACY_WORKER_SHA256 or _sha256_file(TEMPLATE_AMENDMENT) != TEMPLATE_AMENDMENT_SHA256:
    raise RuntimeError("E_LEGACY_WORKER_IMMUTABILITY")

_amendment = json.loads(TEMPLATE_AMENDMENT.read_bytes())
if (
    not isinstance(_amendment, dict)
    or _amendment.get("schema_version") != "nursery-gemma4-public-canary-rendered-template-placeholder-order-amendment-v1.3"
    or _amendment.get("status") != "FROZEN_PRE_CANARY_TEMPLATE_PLACEHOLDER_ORDER_ONLY"
    or _amendment.get("exact_post_render_delta", {}).get("relocated_substring_utf8") != AUDIO_TOKEN
    or len(AUDIO_TOKEN.encode("utf-8")) != 9
):
    raise RuntimeError("E_TEMPLATE_AMENDMENT")

legacy = _load_module(LEGACY_WORKER, "nursery_gemma_template_order_legacy_worker")
_legacy_prompt_messages = legacy._prompt_messages


SYSTEM_PROMPT = legacy.SYSTEM_PROMPT
USER_PROMPT = legacy.USER_PROMPT
FRAME_OFFSETS = legacy.FRAME_OFFSETS
ENUMS = legacy.ENUMS
ALLOWED = legacy.ALLOWED
EXACT_SCHEMA = legacy.EXACT_SCHEMA
INVALID = legacy.INVALID
WINDOW_COUNT = legacy.WINDOW_COUNT
JOB_SCHEMA = legacy.JOB_SCHEMA
CHECKPOINT_SCHEMA = legacy.CHECKPOINT_SCHEMA


def _prompt_messages() -> list[dict[str, Any]]:
    messages = _legacy_prompt_messages()
    content = messages[1]["content"]
    if [row["type"] for row in content] != ["audio", "image", "image", "image", "image", "image", "text"]:
        raise RuntimeError("E_DECLARATIVE_CONTENT_ORDER")
    if content[-1].get("text") != USER_PROMPT or messages[0]["content"][0].get("text") != SYSTEM_PROMPT:
        raise RuntimeError("E_PROMPT_TEXT")
    return messages


def _placeholder_positions(value: str, token: str) -> list[int]:
    result: list[int] = []
    cursor = 0
    while True:
        position = value.find(token, cursor)
        if position < 0:
            return result
        result.append(position)
        cursor = position + len(token)


def _render_frozen_prompt(processor: Any) -> str:
    rendered = processor.apply_chat_template(
        _prompt_messages(),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise RuntimeError("E_TEMPLATE_ORDER_PRECONDITION")
    if (
        rendered.count(SYSTEM_PROMPT) != 1
        or rendered.count(USER_PROMPT) != 1
        or rendered.count(AUDIO_TOKEN) != 1
        or rendered.count(IMAGE_TOKEN) != 5
        or (IMAGE_TOKEN * 5) not in rendered
    ):
        raise RuntimeError("E_TEMPLATE_ORDER_PRECONDITION")
    system_position = rendered.find(SYSTEM_PROMPT)
    user_position = rendered.find(USER_PROMPT)
    audio_position = rendered.find(AUDIO_TOKEN)
    image_block_position = rendered.find(IMAGE_TOKEN * 5)
    if not (system_position < image_block_position < user_position < audio_position):
        raise RuntimeError("E_TEMPLATE_ORDER_PRECONDITION")
    without_audio = rendered[:audio_position] + rendered[audio_position + len(AUDIO_TOKEN) :]
    insert_at = without_audio.find(IMAGE_TOKEN * 5)
    if insert_at < 0:
        raise RuntimeError("E_TEMPLATE_ORDER_CORRECTION")
    corrected = without_audio[:insert_at] + AUDIO_TOKEN + without_audio[insert_at:]
    images = _placeholder_positions(corrected, IMAGE_TOKEN)
    if (
        corrected.count(SYSTEM_PROMPT) != 1
        or corrected.count(USER_PROMPT) != 1
        or corrected.count(AUDIO_TOKEN) != 1
        or len(images) != 5
        or corrected.find(IMAGE_TOKEN * 5) != images[0]
        or corrected.replace(AUDIO_TOKEN, "") != rendered.replace(AUDIO_TOKEN, "")
        or not (
            corrected.find(SYSTEM_PROMPT)
            < corrected.find(AUDIO_TOKEN)
            < images[0]
            < images[-1]
            < corrected.find(USER_PROMPT)
        )
    ):
        raise RuntimeError("E_TEMPLATE_ORDER_CORRECTION")
    return corrected


def template_order_correction_digest() -> str:
    contract = {
        "legacy_worker_sha256": LEGACY_WORKER_SHA256,
        "declarative_order": ["audio", "image", "image", "image", "image", "image", "text"],
        "rendered_precondition": "SYSTEM_FIVE_CONTIGUOUS_IMAGES_USER_AUDIO",
        "rendered_correction": "RELOCATE_EXACTLY_ONE_AUDIO_TOKEN_BEFORE_IMAGE_BLOCK",
        "rendered_postcondition": "SYSTEM_AUDIO_FIVE_CONTIGUOUS_IMAGES_USER",
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(USER_PROMPT.encode("utf-8")).hexdigest(),
        "prompt_digest": legacy.prompt_digest(),
        "exact_schema_digest": legacy.exact_schema_digest(),
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


legacy._prompt_messages = _prompt_messages
legacy._render_frozen_prompt = _render_frozen_prompt

prompt_and_schema_digest = legacy.prompt_and_schema_digest
prompt_digest = legacy.prompt_digest
exact_schema_digest = legacy.exact_schema_digest
_validate_job = legacy._validate_job
_parse_exact = legacy._parse_exact
_slice_exact_audio = legacy._slice_exact_audio
_load_instrument = legacy._load_instrument
_infer_one = legacy._infer_one
_write_checkpoint = legacy._write_checkpoint
run = legacy.run


def main(argv: Sequence[str] | None = None) -> int:
    return legacy.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
