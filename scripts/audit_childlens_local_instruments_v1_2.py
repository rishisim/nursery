#!/usr/bin/env python3
"""Local-only package/executable inventory for ChildLens v1.2 instruments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


VERSION = "childlens-local-instrument-inventory-v1.2.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_OUTPUT_BYTES = 12288
PACKAGE_SPECS = (
    ("faster-whisper", "1.2.1", "MIT", "ASR_PROPOSAL"),
    ("whisperx", "3.8.5", "BSD-2-Clause", "ALIGNMENT_PROPOSAL"),
    ("openai-whisper", None, "MIT", "ASR_REFERENCE_IMPLEMENTATION"),
    ("ctranslate2", None, "MIT", "ASR_RUNTIME"),
    ("torch", None, "BSD-3-Clause", "MODEL_RUNTIME"),
    ("torchaudio", None, "BSD-2-Clause", "AUDIO_RUNTIME"),
    ("pyannote.audio", None, "MIT_CODE_ONLY_MODEL_SEPARATE", "DIARIZATION_RUNTIME"),
    ("speechbrain", None, "Apache-2.0", "SPEECH_RUNTIME"),
    ("silero-vad", None, "MODEL_LICENSE_REQUIRES_SEPARATE_REVIEW", "VAD_RUNTIME"),
    ("streamlit", "1.56.0", "Apache-2.0", "LOCAL_ANNOTATION_UI"),
    ("jiwer", None, "Apache-2.0", "TRANSCRIPT_ERROR_METRIC"),
    ("krippendorff", None, "MIT", "RELIABILITY_METRIC"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _public_file_digest(relative: str) -> str:
    path = REPO_ROOT / relative
    return _sha256_file(path) if path.is_file() else "ABSENT"


def _executable(name: str) -> tuple[Path | None, str | None, str | None]:
    located = shutil.which(name)
    if located is None:
        return None, None, None
    path = Path(located).resolve()
    try:
        completed = subprocess.run(
            [str(path), "-version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
            env={"PATH": os.environ.get("PATH", "")},
        )
    except (OSError, subprocess.TimeoutExpired):
        return path, None, _sha256_file(path)
    first = completed.stdout.decode("utf-8", errors="replace").splitlines()
    version = None
    if first:
        match = re.search(r"version\s+([^\s]+)", first[0])
        if match:
            version = match.group(1)
    return path, version, _sha256_file(path)


def audit() -> dict[str, Any]:
    ffmpeg_path, ffmpeg_version, ffmpeg_sha = _executable("ffmpeg")
    ffprobe_path, ffprobe_version, ffprobe_sha = _executable("ffprobe")
    ffplay_path, ffplay_version, ffplay_sha = _executable("ffplay")
    package_inventory = []
    for name, frozen_version, license_state, role in PACKAGE_SPECS:
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        package_inventory.append(
            {
                "package": name,
                "role": role,
                "installed": installed is not None,
                "installed_version": installed,
                "frozen_candidate_version": frozen_version,
                "license_state_from_v1_1_review": license_state,
                "frozen_version_match": (
                    installed == frozen_version if installed is not None and frozen_version else None
                ),
            }
        )
    ffmpeg_license_mode = "NOT_AVAILABLE"
    if ffmpeg_path is not None:
        try:
            completed = subprocess.run(
                [str(ffmpeg_path), "-version"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=15,
                env={"PATH": os.environ.get("PATH", "")},
            )
            text = completed.stdout.decode("utf-8", errors="replace")
            if "--enable-gpl" in text and "--enable-version3" in text:
                ffmpeg_license_mode = "GPL_V3_OR_LATER_LOCAL_EXECUTION_ONLY_NO_REDISTRIBUTION"
            elif "--enable-gpl" in text:
                ffmpeg_license_mode = "GPL_BUILD_LOCAL_EXECUTION_ONLY_NO_REDISTRIBUTION"
            else:
                ffmpeg_license_mode = "LICENSE_MODE_REQUIRES_BUILD_REVIEW"
        except (OSError, subprocess.TimeoutExpired):
            ffmpeg_license_mode = "VERSION_QUERY_FAILED"
    package_by_name = {row["package"]: row for row in package_inventory}
    asr_ready = all(
        package_by_name[name]["installed"] for name in ("faster-whisper", "ctranslate2")
    )
    alignment_ready = bool(package_by_name["whisperx"]["installed"])
    receipt: dict[str, Any] = {
        "schema_version": VERSION,
        "status": "STRUCTURAL_DECODE_READY_HUMAN_FIRST_AUTOMATED_SPEECH_NOT_READY",
        "scope": "LOCAL_HOST_INVENTORY_NO_RESTRICTED_INPUT",
        "inherited_public_review": {
            "v1_1_instrument_preflight_sha256": _public_file_digest(
                "output/childlens_feasibility_v1_1/instrument_preflight.json"
            ),
            "v1_1_instrument_contract_sha256": _public_file_digest(
                "docs/childlens_feasibility_v1_1/annotation_instrument_contract_amendment_v1_1.md"
            ),
        },
        "executables": {
            "ffmpeg": {
                "available": ffmpeg_path is not None,
                "version": ffmpeg_version,
                "sha256": ffmpeg_sha,
                "license_state": ffmpeg_license_mode,
            },
            "ffprobe": {
                "available": ffprobe_path is not None,
                "version": ffprobe_version,
                "sha256": ffprobe_sha,
            },
            "ffplay": {
                "available": ffplay_path is not None,
                "version": ffplay_version,
                "sha256": ffplay_sha,
                "use": "LOCAL_PLAYBACK_FALLBACK_ONLY",
            },
        },
        "packages": package_inventory,
        "route_disposition": {
            "container_audio_decode": "READY" if ffmpeg_path and ffprobe_path else "BLOCKED",
            "automated_language_identification": "NOT_APPROVED_HUMAN_ESTABLISHES_LANGUAGE",
            "offline_asr_proposals": "READY" if asr_ready else "NOT_LOCAL_DO_NOT_RUN",
            "offline_alignment_proposals": "READY" if alignment_ready else "NOT_LOCAL_DO_NOT_RUN",
            "automated_speaker_role": "EXCLUDED_HUMAN_ONLY",
            "referential_annotation": "HUMAN_ONLY",
            "reliability_scoring": (
                "PACKAGE_READY"
                if package_by_name["jiwer"]["installed"]
                and package_by_name["krippendorff"]["installed"]
                else "IMPLEMENTATION_NOT_FROZEN"
            ),
        },
        "license_and_access_controls": {
            "model_or_weight_license_inferred_from_code_license": False,
            "gated_conditions_accepted": False,
            "model_or_weight_download_performed": False,
            "pyannote_model_use": "PROHIBITED_PENDING_USER_ACCEPTANCE_AND_SEPARATE_REVIEW",
            "vtc_use": "PROHIBITED_LICENSE_UNRESOLVED",
            "new_external_instrument_use": "FAIL_CLOSED_UNLESS_SEPARATELY_PINNED_LICENSED_AND_APPROVED",
        },
        "instrument_to_learner_firewall": {
            "instrument_weights_or_checkpoints_to_learner": False,
            "instrument_tokenizers_or_vocabularies_to_learner": False,
            "instrument_features_or_embeddings_to_learner": False,
            "instrument_scores_or_confidences_to_learner": False,
            "raw_or_unreviewed_proposals_to_learner": False,
            "human_disposition_required_for_any_future_scientific_text": True,
            "learner_or_tokenizer_training_authorized_now": False,
        },
        "boundary_receipt": {
            "restricted_data_accessed": False,
            "external_api_or_upload_used": False,
            "model_or_weight_acquired": False,
            "license_or_terms_accepted": False,
            "learner_or_tokenizer_trained": False,
            "causal_outcome_run": False,
        },
        "privacy_export": {
            "paths_exported": False,
            "identifiers_exported": False,
            "timestamps_exported": False,
            "transcript_or_lexical_content_exported": False,
        },
    }
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError("E_OUTPUT_SIZE")
    return receipt


def main() -> int:
    try:
        receipt = audit()
    except Exception as exc:
        code = str(exc) if str(exc).startswith("E_") else "E_INTERNAL"
        print(json.dumps({"schema_version": VERSION, "status": "error", "error_code": code}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
