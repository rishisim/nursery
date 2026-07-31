"""Public-only four-family synthetic-video qualitative bakeoff."""

from __future__ import annotations

import base64
import html
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .contract import canonical_json_bytes, canonical_json_sha256, file_sha256
from .synthetic_video_pilot import (
    compile_prompt,
    load_config as load_public_pilot_config,
    media_summary,
    scene_by_id,
    validate_final_media,
)
from .synthetic_video_quality_ceiling import (
    _audio_payload_sha256,
    load_quality_config,
    normalize_candidate_with_ltx_audio,
    validate_quality_config,
    verify_ltx_baseline,
)

COMPARISON_ID = "synthetic-video-four-family-quality-ceiling"
SCENE_IDS = ("pick_up", "transfer", "occlusion_persistence", "action_transition")
REFERENCE_FAMILIES = ("ltx", "seedance")
NEW_FAMILIES = ("gemini_omni_flash", "minimax_h3")
ALL_FAMILIES = REFERENCE_FAMILIES + NEW_FAMILIES
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(value))


def _write_bytes_atomic(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    try:
        with partial.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def load_bakeoff_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def planned_cost_usd(config: Mapping[str, Any]) -> Decimal:
    return Decimal(str(config["provider_cost"]["maximum_expected_new_charge_usd"]))


def validate_bakeoff_config(
    config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    completed_quality: Mapping[str, Any],
    repository_root: str | Path,
) -> None:
    """Reject changes that weaken the frozen comparison or its paid-call gate."""

    repository = Path(repository_root).resolve()
    _require(config.get("schema_version") == 1, "schema_version must be 1")
    _require(config.get("comparison_id") == COMPARISON_ID, "unexpected comparison_id")
    status = config.get("status")
    _require(
        status
        in {
            "FROZEN_AWAITING_CREDENTIALS_AND_SPEND_APPROVAL",
            "FROZEN_EXECUTION_AUTHORIZED",
        },
        "unexpected bakeoff protocol status",
    )

    authorization = config.get("authorization", {})
    _require(
        authorization.get("public_only_qualitative_comparison_authorized") is True,
        "public-only comparison is not authorized",
    )
    _require(
        authorization.get("scientific_training_use_authorized") is False,
        "training use must remain unauthorized",
    )
    _require(
        authorization.get("governed_or_child_derived_generator_work_authorized")
        is False,
        "governed generator work must remain unauthorized",
    )
    paid_authorized = authorization.get("paid_execution_authorized")
    authorized_spend = config.get("provider_cost", {}).get(
        "user_authorized_new_spend_usd"
    )
    if status == "FROZEN_AWAITING_CREDENTIALS_AND_SPEND_APPROVAL":
        _require(paid_authorized is False, "pre-approval protocol cannot authorize spend")
        _require(authorized_spend is None, "pre-approval spend must be null")
    else:
        _require(paid_authorized is True, "execution status requires paid authorization")
        _require(
            Decimal(str(authorized_spend)) == planned_cost_usd(config),
            "authorized spend must match the frozen ceiling",
        )

    distinction = config.get("scientific_distinction", {})
    _require(
        distinction.get("completed_public_pilot_sha256")
        == canonical_json_sha256(public_pilot),
        "completed public-pilot protocol hash mismatch",
    )
    _require(
        distinction.get("completed_two_family_protocol_sha256")
        == canonical_json_sha256(completed_quality),
        "completed two-family protocol hash mismatch",
    )
    result_path = repository / distinction["completed_two_family_result"]
    _require(result_path.is_file(), "completed two-family result is missing")
    _require(
        distinction.get("completed_two_family_result_file_sha256")
        == file_sha256(str(result_path)),
        "completed two-family result hash mismatch",
    )
    _require(
        distinction.get("prior_result_is_not_mutated") is True,
        "the completed comparison must remain immutable",
    )

    prohibited = " ".join(config.get("privacy_boundary", {}).get("prohibited", []))
    _require(
        "ChildLens" in prohibited and "BabyView" in prohibited,
        "restricted input sources must remain prohibited",
    )
    privacy = config["privacy_boundary"]
    _require(
        privacy.get("provider_request_payload_storage") is False,
        "provider request storage must remain disabled",
    )
    _require(
        privacy.get("provider_native_audio_retained_in_comparison_final") is False,
        "provider native audio must not enter comparison finals",
    )

    references = config.get("reference_families", {})
    _require(set(references) == set(REFERENCE_FAMILIES), "reference families changed")
    _require(
        references["ltx"].get("local_run_root")
        == "runs/synthetic_video_public_pilot/public-preview-20260730/families/ltx",
        "LTX reference root changed",
    )
    _require(references["ltx"].get("seed") == 314159, "LTX seed changed")
    _require(
        references["seedance"].get("local_run_root")
        == "runs/synthetic_video_quality_ceiling/quality-ceiling-20260731",
        "Seedance reference root changed",
    )
    _require(
        all(
            references[family].get("expected_media_valid_count") == 4
            for family in REFERENCE_FAMILIES
        ),
        "reference-family media count changed",
    )

    families = config.get("new_families", {})
    _require(set(families) == set(NEW_FAMILIES), "new family set changed")
    gemini = families["gemini_omni_flash"]
    _require(
        gemini.get("model") == "gemini-omni-flash-preview",
        "Gemini model changed",
    )
    _require(
        gemini.get("endpoint_url")
        == "https://generativelanguage.googleapis.com/v1beta/interactions",
        "Gemini endpoint changed",
    )
    _require(
        gemini.get("credential_environment_variable") == "GEMINI_API_KEY",
        "Gemini credential variable changed",
    )
    _require(
        gemini.get("request")
        == {
            "model": "gemini-omni-flash-preview",
            "response_format": {
                "type": "video",
                "aspect_ratio": "16:9",
                "duration": "5",
                "delivery": "uri",
            },
            "generation_config": {"video_config": {"task": "text_to_video"}},
            "store": False,
            "background": False,
            "stream": False,
        },
        "Gemini request settings changed",
    )
    minimax = families["minimax_h3"]
    _require(minimax.get("model") == "MiniMax-H3", "MiniMax model changed")
    _require(
        minimax.get("endpoint_url")
        == "https://api.minimax.io/v2/video_generation",
        "MiniMax endpoint changed",
    )
    _require(
        minimax.get("query_url_template")
        == "https://api.minimax.io/v2/query/video_generation/{task_id}",
        "MiniMax query endpoint changed",
    )
    _require(
        minimax.get("credential_environment_variable") == "MINIMAX_API_KEY",
        "MiniMax credential variable changed",
    )
    _require(
        minimax.get("request")
        == {
            "model": "MiniMax-H3",
            "resolution": "2K",
            "duration": 5,
            "ratio": "16:9",
        },
        "MiniMax request settings changed",
    )
    for family in NEW_FAMILIES:
        candidate = families[family]
        _require(
            candidate.get("provider_retries") == 0
            and candidate.get("protocol_retries") == 0
            and candidate.get("attempts_per_scene") == 1,
            f"{family} must retain one attempt and no retries",
        )

    comparison = config.get("comparison", {})
    _require(
        tuple(comparison.get("scene_ids", ())) == SCENE_IDS,
        "scene order or identity changed",
    )
    _require(
        comparison.get("provider_specific_prompt_changes") == "none",
        "provider-specific prompt changes are prohibited",
    )
    _require(
        comparison.get("new_attempt_count") == 8
        and comparison.get("new_attempts_per_family") == 4,
        "attempt count changed",
    )
    _require(
        comparison.get("generation_seconds_per_attempt") == 5,
        "attempt duration changed",
    )
    _require(
        comparison.get("reference_media_policy")
        == (
            "reuse and hash-verify the completed LTX and Seedance finals; "
            "do not regenerate either reference family"
        ),
        "reference-media policy changed",
    )
    _require(
        comparison.get("no_polished_subset") is True,
        "the full four-scene result must be retained",
    )

    review = config.get("review", {})
    _require(
        tuple(review.get("primary_visual_item_ids", ()))
        == (
            "continuous_egocentric_shot",
            "anatomy",
            "contact_action",
            "identity",
            "transition_order",
            "referent_timing",
            "safety",
        ),
        "primary review items changed",
    )
    _require(
        tuple(review.get("audio_control_item_ids", ())) == ("speech",),
        "speech must remain the sole audio-control item",
    )
    _require(
        [item.get("id") for item in review.get("secondary_presentation_items", [])]
        == ["presentation_text_artifacts", "presentation_frame_composition"],
        "secondary presentation items changed",
    )
    expected_public_items = [
        item["id"] for item in public_pilot["human_qa"]["items"]
    ]
    _require(
        set(expected_public_items)
        == set(
            review["primary_visual_item_ids"]
            + review["audio_control_item_ids"]
        ),
        "primary and audio items must preserve the completed QA instrument",
    )
    _require(
        review.get("clear_task_advantage_over_ltx_rule")
        == {
            "media_valid_count_required": 4,
            "minimum_total_primary_visual_pass_advantage": 4,
            "minimum_scene_wins": 2,
            "maximum_scene_losses": 0,
            "equal_judgeable_primary_items_required_per_scene": True,
            "critical_failures_must_not_exceed_ltx": True,
            "maximum_safety_failures": 0,
        },
        "clear-advantage rule changed",
    )
    _require(
        review.get("competitive_with_seedance_rule")
        == {
            "media_valid_count_required": 4,
            "minimum_total_primary_visual_pass_difference": -2,
            "minimum_scene_wins_or_ties": 3,
            "maximum_additional_critical_failures": 1,
            "equal_judgeable_primary_items_required_per_scene": True,
            "maximum_safety_failures": 0,
        },
        "Seedance-competitiveness rule changed",
    )

    cost = config.get("provider_cost", {})
    gemini_cost = cost.get("gemini", {})
    output_cost = (
        Decimal(str(gemini_cost.get("video_output_tokens_per_second_720p")))
        * Decimal(str(gemini_cost.get("planned_generated_seconds")))
        * Decimal(str(gemini_cost.get("video_output_usd_per_million_tokens")))
        / Decimal("1000000")
    )
    input_cost = (
        Decimal(str(gemini_cost.get("conservative_maximum_input_tokens")))
        * Decimal(str(gemini_cost.get("text_input_usd_per_million_tokens")))
        / Decimal("1000000")
    )
    gemini_ceiling = Decimal(str(gemini_cost.get("maximum_expected_charge_usd")))
    _require(
        output_cost + input_cost <= gemini_ceiling == Decimal("2.06"),
        "Gemini expected-charge ceiling changed or is insufficient",
    )
    minimax_cost = cost.get("minimax", {})
    computed_minimax = (
        Decimal(str(minimax_cost.get("two_k_usd_per_second")))
        * Decimal(str(minimax_cost.get("planned_generated_seconds")))
    )
    minimax_ceiling = Decimal(str(minimax_cost.get("maximum_expected_charge_usd")))
    _require(
        computed_minimax == minimax_ceiling == Decimal("2.6"),
        "MiniMax expected-charge ceiling changed",
    )
    _require(
        planned_cost_usd(config) == gemini_ceiling + minimax_ceiling == Decimal("4.66"),
        "combined expected-charge ceiling changed",
    )
    _require(
        cost.get("paid_launch_requires_explicit_new_spend_confirmation") is True,
        "paid launch must require explicit new confirmation",
    )


def _attempt_id(family: str, scene_id: str) -> str:
    return f"{family}__{scene_id}__sprovider__a1__modular"


def _ltx_attempt_id(scene_id: str) -> str:
    return f"ltx__{scene_id}__s314159__a1__modular"


def _seedance_attempt_id(scene_id: str) -> str:
    return f"seedance__{scene_id}__sprovider__a1__modular"


def _blind_id(comparison_id: str, run_id: str, attempt_id: str) -> str:
    return canonical_json_sha256(
        {"comparison_id": comparison_id, "run_id": run_id, "attempt_id": attempt_id}
    )[:12]


def _new_attempt_paths(attempt_id: str) -> dict[str, str]:
    root = f"attempts/{attempt_id}"
    return {
        "request": f"{root}/request.json",
        "planned_attempt": f"{root}/planned_attempt.json",
        "submission": f"{root}/submission.json",
        "provider_status": f"{root}/provider_status.json",
        "provider_response": f"{root}/provider_response.json",
        "raw_video": f"{root}/raw.mp4",
        "video_only": f"{root}/video_only.mp4",
        "final_video": f"{root}/final.mp4",
        "record": f"{root}/attempt.json",
        "failure": f"{root}/failure.json",
    }


def _candidate_request(
    config: Mapping[str, Any],
    family: str,
    prompt: str,
) -> dict[str, Any]:
    settings = config["new_families"][family]["request"]
    if family == "gemini_omni_flash":
        return {**settings, "input": prompt}
    if family == "minimax_h3":
        return {
            **settings,
            "content": [{"type": "text", "text": prompt}],
        }
    raise ValueError(f"unsupported candidate family {family!r}")


def compile_bakeoff_work_order(
    config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    completed_quality: Mapping[str, Any],
    repository_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Compile eight immutable API calls and 16 blinded comparison cards."""

    validate_bakeoff_config(
        config,
        public_pilot,
        completed_quality,
        repository_root,
    )
    _require(
        bool(SAFE_ID.fullmatch(run_id)),
        "run_id must contain only lowercase letters, digits, underscores, and hyphens",
    )
    references = config["reference_families"]
    attempts: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    blinding: dict[str, dict[str, str]] = {}

    for scene_id in SCENE_IDS:
        prompt = compile_prompt(public_pilot, "ltx", scene_id)
        for family in NEW_FAMILIES:
            attempt_id = _attempt_id(family, scene_id)
            request = _candidate_request(config, family, prompt)
            family_cost = Decimal(
                str(
                    config["provider_cost"][
                        "gemini" if family == "gemini_omni_flash" else "minimax"
                    ]["maximum_expected_charge_usd"]
                )
            )
            attempts.append(
                {
                    "attempt_id": attempt_id,
                    "family": family,
                    "scene_id": scene_id,
                    "provider_seed_requested": None,
                    "provider_seed_policy": config["new_families"][family][
                        "input_seed_control"
                    ],
                    "attempt_number": 1,
                    "mode": "modular",
                    "prompt": prompt,
                    "prompt_sha256": canonical_json_sha256(prompt),
                    "request": request,
                    "request_sha256": canonical_json_sha256(request),
                    "planned_charge_usd": float(family_cost / 4),
                    "status": "planned",
                    "paths": _new_attempt_paths(attempt_id),
                }
            )

        ltx_id = _ltx_attempt_id(scene_id)
        seedance_id = _seedance_attempt_id(scene_id)
        cards: list[dict[str, Any]] = [
            {
                "family": "ltx",
                "attempt_id": ltx_id,
                "paths": {
                    "final_video": str(
                        Path(references["ltx"]["local_run_root"])
                        / "attempts"
                        / ltx_id
                        / "final.mp4"
                    ),
                    "record": str(
                        Path(references["ltx"]["local_run_root"])
                        / "attempts"
                        / ltx_id
                        / "attempt.json"
                    ),
                },
            },
            {
                "family": "seedance",
                "attempt_id": seedance_id,
                "paths": {
                    "final_video": str(
                        Path(references["seedance"]["local_run_root"])
                        / "attempts"
                        / seedance_id
                        / "final.mp4"
                    ),
                    "record": str(
                        Path(references["seedance"]["local_run_root"])
                        / "attempts"
                        / seedance_id
                        / "attempt.json"
                    ),
                },
            },
        ]
        for family in NEW_FAMILIES:
            attempt_id = _attempt_id(family, scene_id)
            cards.append(
                {
                    "family": family,
                    "attempt_id": attempt_id,
                    "paths": _new_attempt_paths(attempt_id),
                }
            )
        for card in cards:
            blind_id = _blind_id(COMPARISON_ID, run_id, card["attempt_id"])
            card["blinded_display_id"] = blind_id
            blinding[blind_id] = {
                "family": card["family"],
                "attempt_id": card["attempt_id"],
                "scene_id": scene_id,
            }
        cards.sort(
            key=lambda card: canonical_json_sha256(
                {
                    "comparison_id": COMPARISON_ID,
                    "run_id": run_id,
                    "scene_id": scene_id,
                    "attempt_id": card["attempt_id"],
                }
            )
        )
        for index, card in enumerate(cards):
            card["slot"] = chr(ord("A") + index)
        scenes.append({"scene_id": scene_id, "presentation": cards})

    work_order = {
        "schema_version": 1,
        "comparison_id": COMPARISON_ID,
        "run_id": run_id,
        "compiled_at": utc_now(),
        "bakeoff_protocol_sha256": canonical_json_sha256(config),
        "public_pilot_protocol_sha256": canonical_json_sha256(public_pilot),
        "completed_quality_protocol_sha256": canonical_json_sha256(
            completed_quality
        ),
        "completed_quality_result_file_sha256": file_sha256(
            str(
                Path(repository_root).resolve()
                / config["scientific_distinction"]["completed_two_family_result"]
            )
        ),
        "purpose": config["scientific_distinction"]["new_question"],
        "scientific_training_use_authorized": False,
        "privacy_boundary": config["privacy_boundary"],
        "planned_cost_usd": float(planned_cost_usd(config)),
        "attempts": attempts,
        "scenes": scenes,
        "blinding_key_sha256": canonical_json_sha256(blinding),
    }
    work_order["work_order_sha256"] = canonical_json_sha256(work_order)
    work_order["blinding_key"] = blinding
    return work_order


def validate_work_order(
    config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    completed_quality: Mapping[str, Any],
    repository_root: str | Path,
    work_order: Mapping[str, Any],
) -> None:
    validate_bakeoff_config(
        config,
        public_pilot,
        completed_quality,
        repository_root,
    )
    _require(
        work_order.get("comparison_id") == COMPARISON_ID,
        "work-order comparison ID changed",
    )
    run_id = str(work_order.get("run_id", ""))
    _require(bool(SAFE_ID.fullmatch(run_id)), "work-order run ID is invalid")
    _require(
        work_order.get("bakeoff_protocol_sha256") == canonical_json_sha256(config),
        "work-order bakeoff protocol hash changed",
    )
    _require(
        work_order.get("public_pilot_protocol_sha256")
        == canonical_json_sha256(public_pilot),
        "work-order public-pilot protocol hash changed",
    )
    _require(
        work_order.get("completed_quality_protocol_sha256")
        == canonical_json_sha256(completed_quality),
        "work-order completed-quality protocol hash changed",
    )
    hashed_order = {
        key: value
        for key, value in work_order.items()
        if key not in {"work_order_sha256", "blinding_key"}
    }
    _require(
        canonical_json_sha256(hashed_order) == work_order.get("work_order_sha256"),
        "work-order content hash mismatch",
    )
    expected = compile_bakeoff_work_order(
        config,
        public_pilot,
        completed_quality,
        repository_root,
        run_id,
    )
    for field in (
        "schema_version",
        "comparison_id",
        "bakeoff_protocol_sha256",
        "public_pilot_protocol_sha256",
        "completed_quality_protocol_sha256",
        "completed_quality_result_file_sha256",
        "purpose",
        "scientific_training_use_authorized",
        "privacy_boundary",
        "planned_cost_usd",
        "attempts",
        "scenes",
        "blinding_key_sha256",
    ):
        _require(
            work_order.get(field) == expected[field],
            f"work-order {field} changed",
        )
    if "blinding_key" in work_order:
        _require(
            work_order["blinding_key"] == expected["blinding_key"],
            "work-order blinding key changed",
        )


def _qa_template(
    config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    run_id: str,
    scene_id: str,
    blind_id: str,
) -> dict[str, Any]:
    items = [
        {
            "id": item["id"],
            "scope": (
                "audio_control"
                if item["id"] in config["review"]["audio_control_item_ids"]
                else "primary_visual"
            ),
            "question": item["question"],
            "response": None,
            "confidence": None,
            "note": None,
        }
        for item in public_pilot["human_qa"]["items"]
    ]
    items.extend(
        {
            "id": item["id"],
            "scope": "secondary_presentation",
            "question": item["question"],
            "response": None,
            "confidence": None,
            "note": None,
        }
        for item in config["review"]["secondary_presentation_items"]
    )
    return {
        "schema_version": 1,
        "comparison_id": COMPARISON_ID,
        "run_id": run_id,
        "scene_id": scene_id,
        "blinded_display_id": blind_id,
        "rater_id": None,
        "rated_at": None,
        "items": items,
        "overall_note": None,
    }


def write_bakeoff_work_order(
    run_root: str | Path,
    work_order: Mapping[str, Any],
    config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
) -> Path:
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    public_order = {
        key: value for key, value in work_order.items() if key != "blinding_key"
    }
    _write_json(root / "work_order.json", public_order)
    key_path = root / "blinding_key.json"
    _write_json(key_path, work_order["blinding_key"])
    key_path.chmod(0o600)
    for attempt in work_order["attempts"]:
        _write_json(root / attempt["paths"]["request"], attempt["request"])
        _write_json(
            root / attempt["paths"]["planned_attempt"],
            attempt,
        )
    for scene in work_order["scenes"]:
        for card in scene["presentation"]:
            _write_json(
                root / "qa" / f"{card['blinded_display_id']}.json",
                _qa_template(
                    config,
                    public_pilot,
                    str(work_order["run_id"]),
                    str(scene["scene_id"]),
                    str(card["blinded_display_id"]),
                ),
            )
    _write_json(
        root / "run_status.json",
        {
            "schema_version": 1,
            "comparison_id": COMPARISON_ID,
            "run_id": work_order["run_id"],
            "status": "planned",
            "completed_attempt_ids": [],
            "failed_attempt_ids": [],
        },
    )
    return root / "work_order.json"


def verify_reference_media(
    config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    completed_quality: Mapping[str, Any],
    repository_root: str | Path,
    *,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
) -> dict[str, dict[str, dict[str, Any]]]:
    """Hash-verify the eight completed LTX and Seedance reference finals."""

    repository = Path(repository_root).resolve()
    ltx = verify_ltx_baseline(completed_quality, public_pilot, repository)
    result_path = (
        repository / config["scientific_distinction"]["completed_two_family_result"]
    )
    _require(
        file_sha256(str(result_path))
        == config["scientific_distinction"][
            "completed_two_family_result_file_sha256"
        ],
        "completed Seedance result hash changed",
    )
    result = json.loads(result_path.read_text())
    _require(
        result.get("status") == "COMPLETE_QUALITATIVE_SCREEN",
        "completed Seedance comparison is not complete",
    )
    result_records = {
        record["scene_id"]: record for record in result.get("seedance_attempts", [])
    }
    _require(set(result_records) == set(SCENE_IDS), "Seedance result scene set changed")
    seedance_root = repository / config["reference_families"]["seedance"][
        "local_run_root"
    ]
    verified_seedance: dict[str, dict[str, Any]] = {}
    for scene_id in SCENE_IDS:
        attempt_id = _seedance_attempt_id(scene_id)
        attempt_root = seedance_root / "attempts" / attempt_id
        record_path = attempt_root / "attempt.json"
        final_video = attempt_root / "final.mp4"
        _require(record_path.is_file(), f"{attempt_id} record is missing")
        _require(final_video.is_file(), f"{attempt_id} final is missing")
        record = json.loads(record_path.read_text())
        result_record = result_records[scene_id]
        expected_hash = result_record["final_sha256"]
        _require(record.get("status") == "media_valid", f"{attempt_id} is invalid")
        _require(
            record.get("final_media", {}).get("sha256") == expected_hash,
            f"{attempt_id} record hash differs from curated result",
        )
        _require(
            file_sha256(str(final_video)) == expected_hash,
            f"{attempt_id} final hash mismatch",
        )
        final_summary = media_summary(final_video, ffprobe_executable)
        _require(
            not validate_final_media(public_pilot, final_summary),
            f"{attempt_id} no longer conforms to delivery settings",
        )
        ltx_audio = _audio_payload_sha256(
            ltx[scene_id]["final_video"],
            ffmpeg_executable,
        )
        seedance_audio = _audio_payload_sha256(final_video, ffmpeg_executable)
        _require(
            ltx_audio == seedance_audio,
            f"{attempt_id} paired audio identity changed",
        )
        verified_seedance[scene_id] = {
            "attempt_id": attempt_id,
            "record": record,
            "final_video": final_video,
            "final_sha256": expected_hash,
        }
    return {"ltx": ltx, "seedance": verified_seedance}


class _JSONAPIClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        if not api_key:
            raise ValueError("API key is required")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._opener = opener

    def _request(
        self,
        method: str,
        url: str,
        *,
        body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> bytes:
        request_headers = {"Accept": "application/json"}
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(
            url,
            data=canonical_json_bytes(body) if body is not None else None,
            headers=request_headers,
            method=method,
        )
        try:
            with self._opener(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:2000]
            raise RuntimeError(
                f"provider API returned HTTP {error.code}: {detail}"
            ) from None
        except urllib.error.URLError as error:
            raise RuntimeError(f"provider API request failed: {error.reason}") from None

    def _json_request(
        self,
        method: str,
        url: str,
        *,
        body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        payload = self._request(method, url, body=body, headers=headers)
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise RuntimeError("provider API returned non-object JSON")
        return decoded


class GeminiInteractionsClient(_JSONAPIClient):
    """Secret-safe minimal client for Gemini Omni Flash URI delivery."""

    HOST = "generativelanguage.googleapis.com"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 1800.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        super().__init__(
            api_key,
            timeout_seconds=timeout_seconds,
            opener=opener,
        )

    @classmethod
    def _require_url(cls, url: str, *, file_path: bool = False) -> None:
        parsed = urllib.parse.urlparse(url)
        if (
            parsed.scheme != "https"
            or (parsed.hostname or "").lower() != cls.HOST
            or parsed.username is not None
        ):
            raise ValueError("refusing Gemini request to an untrusted URL")
        if file_path and not parsed.path.startswith("/v1beta/files/"):
            raise ValueError("unexpected Gemini file URL")

    def create(
        self,
        endpoint_url: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._require_url(endpoint_url)
        return self._json_request(
            "POST",
            endpoint_url,
            body=payload,
            headers={"x-goog-api-key": self._api_key},
        )

    @staticmethod
    def video_output(response: Mapping[str, Any]) -> dict[str, Any]:
        direct = response.get("output_video")
        if isinstance(direct, Mapping):
            return dict(direct)
        for step in reversed(response.get("steps", [])):
            if not isinstance(step, Mapping):
                continue
            for content in reversed(step.get("content", [])):
                if isinstance(content, Mapping) and content.get("type") == "video":
                    return dict(content)
        raise RuntimeError("Gemini response is missing video output")

    @classmethod
    def file_name_from_uri(cls, uri: str) -> str:
        cls._require_url(uri, file_path=True)
        match = re.search(r"/v1beta/files/([^/:?]+)", urllib.parse.urlparse(uri).path)
        if not match:
            raise RuntimeError("Gemini video URI does not contain a file name")
        return match.group(1)

    def file_status(self, file_name: str) -> dict[str, Any]:
        _require(
            bool(re.fullmatch(r"[A-Za-z0-9_-]+", file_name)),
            "invalid Gemini file name",
        )
        url = f"https://{self.HOST}/v1beta/files/{file_name}"
        return self._json_request(
            "GET",
            url,
            headers={"x-goog-api-key": self._api_key},
        )

    def download_uri(self, uri: str, destination: str | Path) -> None:
        self._require_url(uri, file_path=True)
        payload = self._request(
            "GET",
            uri,
            headers={"x-goog-api-key": self._api_key},
        )
        _write_bytes_atomic(destination, payload)


class MiniMaxVideoClient(_JSONAPIClient):
    """Minimal client for the official asynchronous MiniMax H3 V2 API."""

    API_HOST = "api.minimax.io"
    DOWNLOAD_SUFFIXES = ("minimax.io", "minimax.chat", "hailuoai.com")

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 120.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        super().__init__(
            api_key,
            timeout_seconds=timeout_seconds,
            opener=opener,
        )

    @classmethod
    def _require_api_url(cls, url: str) -> None:
        parsed = urllib.parse.urlparse(url)
        if (
            parsed.scheme != "https"
            or (parsed.hostname or "").lower() != cls.API_HOST
            or parsed.username is not None
        ):
            raise ValueError("refusing MiniMax credentials to an untrusted URL")

    @classmethod
    def _require_download_url(cls, url: str) -> None:
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").lower()
        allowed = any(
            host == suffix or host.endswith(f".{suffix}")
            for suffix in cls.DOWNLOAD_SUFFIXES
        )
        if parsed.scheme != "https" or not allowed or parsed.username is not None:
            raise ValueError("refusing MiniMax download from an untrusted URL")

    def submit(
        self,
        endpoint_url: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._require_api_url(endpoint_url)
        return self._json_request(
            "POST",
            endpoint_url,
            body=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

    def status(self, query_url: str) -> dict[str, Any]:
        self._require_api_url(query_url)
        return self._json_request(
            "GET",
            query_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

    def download_public_file(self, url: str, destination: str | Path) -> None:
        self._require_download_url(url)
        payload = self._request("GET", url, headers={"Accept": "video/mp4"})
        _write_bytes_atomic(destination, payload)


def _gemini_submission(
    response: Mapping[str, Any],
    output: Mapping[str, Any],
) -> dict[str, Any]:
    uri = output.get("uri")
    data = output.get("data")
    return {
        "interaction_id": response.get("id"),
        "status": response.get("status"),
        "model": response.get("model"),
        "video_uri": uri if isinstance(uri, str) else None,
        "inline_video_present": isinstance(data, str) and bool(data),
        "mime_type": output.get("mime_type"),
        "usage": response.get("usage"),
    }


def _sanitized_gemini_response(
    response: Mapping[str, Any],
    output: Mapping[str, Any],
) -> dict[str, Any]:
    uri = str(output.get("uri", ""))
    return {
        "interaction_id_utf8_sha256": canonical_json_sha256(
            str(response.get("id", ""))
        ),
        "status": response.get("status"),
        "model": response.get("model"),
        "usage": response.get("usage"),
        "video": {
            "uri_utf8_sha256": canonical_json_sha256(uri) if uri else None,
            "host": urllib.parse.urlparse(uri).hostname if uri else None,
            "mime_type": output.get("mime_type"),
            "inline_video_present": bool(output.get("data")),
        },
    }


def _sanitized_minimax_status(response: Mapping[str, Any]) -> dict[str, Any]:
    task = response.get("task", {})
    content = task.get("content", {}) if isinstance(task, Mapping) else {}
    url = str(content.get("url", ""))
    return {
        "task": {
            "id_utf8_sha256": canonical_json_sha256(str(task.get("id", ""))),
            "model": task.get("model"),
            "status": task.get("status"),
            "error": task.get("error"),
            "created_at": task.get("created_at"),
            "updated_at": task.get("updated_at"),
            "resolution": task.get("resolution"),
            "duration": task.get("duration"),
            "usage": task.get("usage"),
            "ratio": task.get("ratio"),
            "task_type": task.get("task_type"),
            "content": {
                "url_utf8_sha256": canonical_json_sha256(url) if url else None,
                "host": urllib.parse.urlparse(url).hostname if url else None,
            },
        }
    }


def _validate_existing_record(
    public_pilot: Mapping[str, Any],
    record_path: Path,
    final_path: Path,
    ffprobe_executable: str,
) -> dict[str, Any] | None:
    if not record_path.exists():
        return None
    record = json.loads(record_path.read_text())
    if record.get("status") != "media_valid":
        raise RuntimeError("existing candidate record is not media-valid")
    if not final_path.is_file():
        raise RuntimeError("existing candidate record is missing final media")
    _require(
        record.get("final_media", {}).get("sha256")
        == file_sha256(str(final_path)),
        "existing candidate final hash mismatch",
    )
    summary = media_summary(final_path, ffprobe_executable)
    _require(
        not validate_final_media(public_pilot, summary),
        "existing candidate final no longer conforms",
    )
    return record


def _run_gemini_attempt(
    config: Mapping[str, Any],
    attempt: Mapping[str, Any],
    paths: Mapping[str, Path],
    client: GeminiInteractionsClient,
    *,
    poll_interval_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if paths["submission"].exists():
        submission = json.loads(paths["submission"].read_text())
        if paths["raw_video"].is_file():
            response_record = (
                json.loads(paths["provider_response"].read_text())
                if paths["provider_response"].exists()
                else {
                    "status": submission.get("status"),
                    "model": submission.get("model"),
                }
            )
            return submission, response_record
        uri = submission.get("video_uri")
        if not isinstance(uri, str) or not uri:
            raise RuntimeError("saved Gemini submission has no resumable video URI")
    else:
        response = client.create(
            config["new_families"]["gemini_omni_flash"]["endpoint_url"],
            attempt["request"],
        )
        output = client.video_output(response)
        submission = _gemini_submission(response, output)
        _write_json(paths["submission"], submission)
        sanitized = _sanitized_gemini_response(response, output)
        _write_json(paths["provider_response"], sanitized)
        inline = output.get("data")
        if isinstance(inline, str) and inline:
            try:
                _write_bytes_atomic(paths["raw_video"], base64.b64decode(inline))
            except ValueError as error:
                raise RuntimeError("Gemini returned invalid base64 video") from error
            return submission, sanitized
        uri = output.get("uri")
        if not isinstance(uri, str) or not uri:
            raise RuntimeError("Gemini response has neither video URI nor inline data")

    file_name = client.file_name_from_uri(uri)
    while True:
        status = client.file_status(file_name)
        _write_json(paths["provider_status"], status)
        state = status.get("state")
        if isinstance(state, Mapping):
            state = state.get("name")
        if state == "ACTIVE":
            break
        if state == "FAILED":
            raise RuntimeError("Gemini output file processing failed")
        if state not in {"PROCESSING", "STATE_UNSPECIFIED", None}:
            raise RuntimeError(f"unexpected Gemini file state {state!r}")
        time.sleep(poll_interval_seconds)
    client.download_uri(uri, paths["raw_video"])
    response_record = (
        json.loads(paths["provider_response"].read_text())
        if paths["provider_response"].exists()
        else {"status": submission.get("status"), "model": submission.get("model")}
    )
    return submission, response_record


def _run_minimax_attempt(
    config: Mapping[str, Any],
    attempt: Mapping[str, Any],
    paths: Mapping[str, Path],
    client: MiniMaxVideoClient,
    *,
    poll_interval_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    family = config["new_families"]["minimax_h3"]
    if paths["submission"].exists():
        submission = json.loads(paths["submission"].read_text())
        if paths["raw_video"].is_file():
            response_record = (
                json.loads(paths["provider_response"].read_text())
                if paths["provider_response"].exists()
                else {"task": {"status": "succeeded"}}
            )
            return submission, response_record
    else:
        submission = client.submit(family["endpoint_url"], attempt["request"])
        _write_json(paths["submission"], submission)
    task_id = submission.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise RuntimeError("MiniMax submission is missing task_id")
    _require(
        bool(re.fullmatch(r"[A-Za-z0-9_-]+", task_id)),
        "MiniMax task_id is invalid",
    )
    query_url = family["query_url_template"].format(task_id=task_id)
    while True:
        status_response = client.status(query_url)
        sanitized = _sanitized_minimax_status(status_response)
        _write_json(paths["provider_status"], sanitized)
        task = status_response.get("task", {})
        state = task.get("status")
        if state == "succeeded":
            break
        if state in {"failed", "cancelled", "expired"}:
            raise RuntimeError(
                f"MiniMax generation ended as {state}: {task.get('error')}"
            )
        if state not in {"queued", "running"}:
            raise RuntimeError(f"unexpected MiniMax task state {state!r}")
        time.sleep(poll_interval_seconds)
    video_url = task.get("content", {}).get("url")
    if not isinstance(video_url, str) or not video_url:
        raise RuntimeError("MiniMax success response is missing content.url")
    _write_json(paths["provider_response"], sanitized)
    client.download_public_file(video_url, paths["raw_video"])
    return submission, sanitized


def _run_one_attempt(
    config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    attempt: Mapping[str, Any],
    ltx_reference: Mapping[str, Any],
    run_root: Path,
    gemini_client: GeminiInteractionsClient,
    minimax_client: MiniMaxVideoClient,
    *,
    implementation_commit: str,
    poll_interval_seconds: float,
    ffmpeg_executable: str,
    ffprobe_executable: str,
) -> dict[str, Any]:
    paths = {key: run_root / value for key, value in attempt["paths"].items()}
    existing = _validate_existing_record(
        public_pilot,
        paths["record"],
        paths["final_video"],
        ffprobe_executable,
    )
    if existing is not None:
        return existing
    if paths["failure"].exists():
        if not paths["submission"].exists() and not paths["raw_video"].exists():
            raise RuntimeError(
                f"{attempt['attempt_id']} has a non-resumable failed call; "
                "a new provider submission requires new authorization"
            )

    started_at = utc_now()
    try:
        if attempt["family"] == "gemini_omni_flash":
            submission, provider_response = _run_gemini_attempt(
                config,
                attempt,
                paths,
                gemini_client,
                poll_interval_seconds=poll_interval_seconds,
            )
            provider_name = "Gemini Developer API"
            provider_identifier = submission.get("interaction_id")
        elif attempt["family"] == "minimax_h3":
            submission, provider_response = _run_minimax_attempt(
                config,
                attempt,
                paths,
                minimax_client,
                poll_interval_seconds=poll_interval_seconds,
            )
            provider_name = "MiniMax API"
            provider_identifier = submission.get("task_id")
        else:
            raise RuntimeError(f"unsupported family {attempt['family']!r}")

        audio_hashes = normalize_candidate_with_ltx_audio(
            public_pilot,
            raw_video=paths["raw_video"],
            ltx_final_video=ltx_reference["final_video"],
            video_only=paths["video_only"],
            final_video=paths["final_video"],
            ffmpeg_executable=ffmpeg_executable,
        )
        raw_summary = media_summary(paths["raw_video"], ffprobe_executable)
        final_summary = media_summary(paths["final_video"], ffprobe_executable)
        errors = validate_final_media(public_pilot, final_summary)
        record = {
            "schema_version": 1,
            "attempt": dict(attempt),
            "status": "media_valid" if not errors else "invalid_media",
            "started_at": started_at,
            "finished_at": utc_now(),
            "provider": {
                "name": provider_name,
                "model": config["new_families"][attempt["family"]]["model"],
                "nursery_adapter_commit": implementation_commit,
                "provider_identifier_utf8_sha256": canonical_json_sha256(
                    str(provider_identifier or "")
                ),
                "request_sha256": attempt["request_sha256"],
                "automatic_retries_disabled": True,
                "protocol_retries": 0,
                "request_storage_disabled": config["privacy_boundary"][
                    "provider_request_payload_storage"
                ],
                "terminal_response": provider_response,
            },
            "runtime": {
                "planned_charge_usd": attempt["planned_charge_usd"],
                "actual_provider_invoice_usd": None,
            },
            "raw_media": raw_summary,
            "final_media": final_summary,
            "paired_ltx_attempt_id": ltx_reference["attempt_id"],
            "audio_identity": audio_hashes,
            "conformance_errors": errors,
            "human_qa_status": "pending",
        }
        _write_json(paths["record"], record)
        return record
    except Exception as error:
        _write_json(
            paths["failure"],
            {
                "schema_version": 1,
                "attempt_id": attempt["attempt_id"],
                "status": "failed",
                "started_at": started_at,
                "failed_at": utc_now(),
                "nursery_adapter_commit": implementation_commit,
                "provider_submission_created": paths["submission"].exists(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def execute_bakeoff(
    config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    completed_quality: Mapping[str, Any],
    work_order: Mapping[str, Any],
    *,
    repository_root: str | Path,
    run_root: str | Path,
    approved_spend_usd: Decimal,
    gemini_api_key: str,
    minimax_api_key: str,
    implementation_commit: str,
    poll_interval_seconds: float = 10.0,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
) -> dict[str, Any]:
    """Run exactly eight paid calls sequentially, with no generated retries."""

    validate_work_order(
        config,
        public_pilot,
        completed_quality,
        repository_root,
        work_order,
    )
    _require(
        config.get("status") == "FROZEN_EXECUTION_AUTHORIZED"
        and config.get("authorization", {}).get("paid_execution_authorized") is True,
        "paid execution is not authorized in the frozen protocol",
    )
    ceiling = planned_cost_usd(config)
    _require(
        approved_spend_usd == ceiling,
        f"approved spend must exactly match the frozen ceiling {ceiling}",
    )
    _require(
        Decimal(str(config["provider_cost"]["user_authorized_new_spend_usd"]))
        == ceiling,
        "protocol authorization does not match the frozen ceiling",
    )
    _require(
        bool(COMMIT_SHA.fullmatch(implementation_commit)),
        "a clean 40-character Git execution commit is required",
    )
    _require(
        Decimal(str(work_order["planned_cost_usd"])) == ceiling,
        "work-order cost differs from the frozen ceiling",
    )
    references = verify_reference_media(
        config,
        public_pilot,
        completed_quality,
        repository_root,
        ffmpeg_executable=ffmpeg_executable,
        ffprobe_executable=ffprobe_executable,
    )
    root = Path(run_root).resolve()
    _write_json(
        root / "approval.json",
        {
            "schema_version": 1,
            "approved_spend_usd": float(approved_spend_usd),
            "frozen_maximum_expected_new_charge_usd": float(ceiling),
            "authorized_provider_submission_count": 8,
            "authorized_retries": 0,
            "nursery_adapter_commit": implementation_commit,
            "recorded_at": utc_now(),
        },
    )
    gemini_client = GeminiInteractionsClient(gemini_api_key)
    minimax_client = MiniMaxVideoClient(minimax_api_key)
    completed: list[str] = []
    failed: list[str] = []
    started_at = utc_now()
    for attempt in work_order["attempts"]:
        try:
            record = _run_one_attempt(
                config,
                public_pilot,
                attempt,
                references["ltx"][attempt["scene_id"]],
                root,
                gemini_client,
                minimax_client,
                implementation_commit=implementation_commit,
                poll_interval_seconds=poll_interval_seconds,
                ffmpeg_executable=ffmpeg_executable,
                ffprobe_executable=ffprobe_executable,
            )
            if record["status"] != "media_valid":
                raise RuntimeError(
                    f"{attempt['attempt_id']} produced invalid final media"
                )
            completed.append(attempt["attempt_id"])
        except Exception:
            failed.append(attempt["attempt_id"])
            status = {
                "schema_version": 1,
                "comparison_id": COMPARISON_ID,
                "run_id": work_order["run_id"],
                "status": "failed",
                "nursery_adapter_commit": implementation_commit,
                "started_at": started_at,
                "finished_at": utc_now(),
                "completed_attempt_ids": completed,
                "failed_attempt_ids": failed,
            }
            _write_json(root / "run_status.json", status)
            raise
        _write_json(
            root / "run_status.json",
            {
                "schema_version": 1,
                "comparison_id": COMPARISON_ID,
                "run_id": work_order["run_id"],
                "status": "running",
                "nursery_adapter_commit": implementation_commit,
                "started_at": started_at,
                "completed_attempt_ids": completed,
                "failed_attempt_ids": failed,
            },
        )
    status = {
        "schema_version": 1,
        "comparison_id": COMPARISON_ID,
        "run_id": work_order["run_id"],
        "status": "complete",
        "nursery_adapter_commit": implementation_commit,
        "started_at": started_at,
        "finished_at": utc_now(),
        "completed_attempt_ids": completed,
        "failed_attempt_ids": failed,
        "maximum_expected_new_charge_usd": float(ceiling),
        "actual_provider_invoice_usd": None,
    }
    _write_json(root / "run_status.json", status)
    return status


def _card_video_path(
    card: Mapping[str, Any],
    repository_root: Path,
    run_root: Path,
) -> Path:
    if card["family"] in REFERENCE_FAMILIES:
        return repository_root / card["paths"]["final_video"]
    return run_root / card["paths"]["final_video"]


def render_blinded_gallery(
    config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    completed_quality: Mapping[str, Any],
    work_order: Mapping[str, Any],
    *,
    repository_root: str | Path,
    run_root: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Render all 16 clips without family-revealing filenames or labels."""

    validate_work_order(
        config,
        public_pilot,
        completed_quality,
        repository_root,
        work_order,
    )
    repository = Path(repository_root).resolve()
    root = Path(run_root).resolve()
    output = (
        Path(output_path).resolve()
        if output_path
        else root / "gallery" / "index.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    alias_root = output.parent / "media"
    alias_root.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    for scene_group in work_order["scenes"]:
        scene = scene_by_id(public_pilot, scene_group["scene_id"])
        cards: list[str] = []
        for card in scene_group["presentation"]:
            video = _card_video_path(card, repository, root)
            alias = alias_root / f"{card['blinded_display_id']}.mp4"
            if alias.exists() or alias.is_symlink():
                alias.unlink()
            if video.exists():
                try:
                    os.link(video, alias)
                except OSError:
                    shutil.copy2(video, alias)
            video_rel = os.path.relpath(alias, output.parent)
            qa = root / "qa" / f"{card['blinded_display_id']}.json"
            qa_rel = os.path.relpath(qa, output.parent)
            missing = (
                ""
                if video.exists()
                else '<p class="missing">Media not present yet.</p>'
            )
            cards.append(
                f"""
                <article class="attempt">
                  <h3>Clip {html.escape(str(card['slot']))}</h3>
                  <video controls preload="metadata" src="{html.escape(video_rel)}"></video>
                  {missing}
                  <p>Blinded ID: <code>{html.escape(str(card['blinded_display_id']))}</code></p>
                  <p><a href="{html.escape(qa_rel)}">QA record</a></p>
                </article>
                """
            )
        sections.append(
            f"""
            <section>
              <h2>{html.escape(scene['label'])}</h2>
              <p class="utterance" lang="de">{html.escape(scene['utterance_de'])}</p>
              <div class="attempts">{''.join(cards)}</div>
            </section>
            """
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blinded four-family synthetic-video bakeoff</title>
  <style>
    :root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin: 0 auto; max-width: 1500px; padding: 2rem; background: #111827; color: #f3f4f6; }}
    header, section {{ background: #1f2937; border: 1px solid #374151; border-radius: 14px; margin: 1rem 0; padding: 1.25rem; }}
    h1, h2, h3 {{ margin-top: 0; }}
    .warning {{ color: #fbbf24; }}
    .attempts {{ display: grid; grid-template-columns: repeat(4, minmax(250px, 1fr)); gap: 1rem; }}
    .attempt {{ background: #111827; border-radius: 10px; padding: 1rem; }}
    video {{ width: 100%; aspect-ratio: 1280 / 704; background: #000; border-radius: 8px; }}
    .utterance {{ font-size: 1.1rem; }}
    .missing {{ color: #fca5a5; }}
    a {{ color: #93c5fd; }}
    code {{ color: #d1fae5; }}
    @media (max-width: 1100px) {{ .attempts {{ grid-template-columns: repeat(2, minmax(250px, 1fr)); }} }}
    @media (max-width: 620px) {{ .attempts {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Blinded public-only four-family bakeoff</h1>
    <p>Blinded run <code>{html.escape(str(work_order['work_order_sha256'])[:12])}</code></p>
    <p class="warning">Do not open the separate blinding key until all 16 QA records are frozen. Qualitative comparison only; no training use is authorized.</p>
  </header>
  {''.join(sections)}
</body>
</html>
"""
    output.write_text(page)
    return output


def _expected_qa_items(
    config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
) -> list[dict[str, str]]:
    items = [
        {
            "id": item["id"],
            "scope": (
                "audio_control"
                if item["id"] in config["review"]["audio_control_item_ids"]
                else "primary_visual"
            ),
            "question": item["question"],
        }
        for item in public_pilot["human_qa"]["items"]
    ]
    items.extend(
        {
            "id": item["id"],
            "scope": "secondary_presentation",
            "question": item["question"],
        }
        for item in config["review"]["secondary_presentation_items"]
    )
    return items


def _validate_completed_qa(
    config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    work_order: Mapping[str, Any],
    qa: Mapping[str, Any],
    *,
    expected_scene_id: str,
    expected_blind_id: str,
) -> None:
    _require("family" not in qa and "attempt_id" not in qa, "QA record is unblinded")
    _require(
        qa.get("comparison_id") == COMPARISON_ID
        and qa.get("run_id") == work_order["run_id"],
        "QA comparison identity changed",
    )
    _require(qa.get("scene_id") == expected_scene_id, "QA scene ID changed")
    _require(
        qa.get("blinded_display_id") == expected_blind_id,
        "QA blinded ID changed",
    )
    rater = qa.get("rater_id")
    _require(isinstance(rater, str) and bool(rater.strip()), "QA rater ID is missing")
    rated_at = qa.get("rated_at")
    _require(isinstance(rated_at, str), "QA rated_at is missing")
    try:
        parsed_at = datetime.fromisoformat(rated_at.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("QA rated_at is not ISO-8601") from None
    _require(parsed_at.tzinfo is not None, "QA rated_at must include a timezone")
    expected_items = _expected_qa_items(config, public_pilot)
    items = qa.get("items")
    _require(isinstance(items, list), "QA items must be a list")
    _require(
        [item.get("id") for item in items]
        == [item["id"] for item in expected_items],
        "QA item order or identity changed",
    )
    allowed_responses = set(config["review"]["response_values"])
    allowed_confidence = set(config["review"]["confidence_values"])
    for item, expected in zip(items, expected_items, strict=True):
        _require(item.get("scope") == expected["scope"], "QA item scope changed")
        _require(item.get("question") == expected["question"], "QA question changed")
        _require(
            item.get("response") in allowed_responses,
            f"QA response is invalid for {item.get('id')}",
        )
        _require(
            item.get("confidence") in allowed_confidence,
            f"QA confidence is invalid for {item.get('id')}",
        )


def inspect_blinded_review_status(
    config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    completed_quality: Mapping[str, Any],
    work_order: Mapping[str, Any],
    *,
    repository_root: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    validate_work_order(
        config,
        public_pilot,
        completed_quality,
        repository_root,
        work_order,
    )
    root = Path(run_root).resolve()
    counts = {"complete": 0, "pending": 0, "missing": 0}
    problems: list[dict[str, str]] = []
    for scene in work_order["scenes"]:
        for card in scene["presentation"]:
            blind_id = str(card["blinded_display_id"])
            path = root / "qa" / f"{blind_id}.json"
            if not path.exists():
                counts["missing"] += 1
                problems.append({"blinded_display_id": blind_id, "problem": "missing"})
                continue
            qa = json.loads(path.read_text())
            try:
                _validate_completed_qa(
                    config,
                    public_pilot,
                    work_order,
                    qa,
                    expected_scene_id=str(scene["scene_id"]),
                    expected_blind_id=blind_id,
                )
            except ValueError as error:
                counts["pending"] += 1
                problems.append(
                    {"blinded_display_id": blind_id, "problem": str(error)}
                )
            else:
                counts["complete"] += 1
    return {
        "status": (
            "READY_TO_UNBLIND"
            if counts == {"complete": 16, "pending": 0, "missing": 0}
            else "BLINDED_QA_INCOMPLETE"
        ),
        "record_counts": counts,
        "problems": problems,
        "family_key_opened": False,
    }


def _verify_new_execution(
    config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    work_order: Mapping[str, Any],
    run_root: Path,
    *,
    ffmpeg_executable: str,
    ffprobe_executable: str,
) -> dict[str, Any]:
    status = json.loads((run_root / "run_status.json").read_text())
    expected_ids = {attempt["attempt_id"] for attempt in work_order["attempts"]}
    _require(status.get("status") == "complete", "new-family run is not complete")
    _require(
        set(status.get("completed_attempt_ids", [])) == expected_ids,
        "completed new-family attempt set changed",
    )
    _require(status.get("failed_attempt_ids") == [], "new-family run has failures")
    records: dict[str, dict[str, Any]] = {}
    for attempt in work_order["attempts"]:
        paths = {key: run_root / value for key, value in attempt["paths"].items()}
        record = _validate_existing_record(
            public_pilot,
            paths["record"],
            paths["final_video"],
            ffprobe_executable,
        )
        _require(record is not None, f"{attempt['attempt_id']} record is missing")
        _require(
            record["attempt"]["request_sha256"] == attempt["request_sha256"],
            f"{attempt['attempt_id']} request hash changed",
        )
        ltx_audio = record.get("audio_identity", {}).get(
            "baseline_audio_payload_sha256"
        )
        final_audio = _audio_payload_sha256(
            paths["final_video"],
            ffmpeg_executable,
        )
        _require(
            ltx_audio == final_audio,
            f"{attempt['attempt_id']} paired audio identity changed",
        )
        records[attempt["attempt_id"]] = record
    return {
        "planned_attempt_count": 8,
        "media_valid_count": len(records),
        "generated_attempt_failure_count": 0,
        "records": records,
    }


def _family_metrics(
    clips: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    primary_ids = set(config["review"]["primary_visual_item_ids"])
    secondary_ids = {
        item["id"] for item in config["review"]["secondary_presentation_items"]
    }
    critical_ids = set(config["review"]["critical_visual_item_ids"])
    counts_by_scope = {
        "primary_visual": {"pass": 0, "fail": 0, "cannot_judge": 0},
        "secondary_presentation": {"pass": 0, "fail": 0, "cannot_judge": 0},
        "audio_control": {"pass": 0, "fail": 0, "cannot_judge": 0},
    }
    item_counts: dict[str, dict[str, int]] = {}
    for clip in clips:
        for item_id, response in clip["responses"].items():
            item_counts.setdefault(
                item_id,
                {"pass": 0, "fail": 0, "cannot_judge": 0},
            )[response] += 1
            if item_id in primary_ids:
                scope = "primary_visual"
            elif item_id in secondary_ids:
                scope = "secondary_presentation"
            else:
                scope = "audio_control"
            counts_by_scope[scope][response] += 1
    primary = counts_by_scope["primary_visual"]
    judgeable = primary["pass"] + primary["fail"]
    critical_failures = sum(
        clip["responses"][item_id] == "fail"
        for clip in clips
        for item_id in critical_ids
    )
    return {
        "clip_count": len(clips),
        "counts_by_scope": counts_by_scope,
        "primary_visual_judgeable_count": judgeable,
        "primary_visual_pass_rate": (
            primary["pass"] / judgeable if judgeable else None
        ),
        "critical_visual_failure_count": critical_failures,
        "safety_failure_count": item_counts["safety"]["fail"],
        "item_counts": item_counts,
    }


def _pairwise_comparison(
    candidate_family: str,
    reference_family: str,
    clips_by_family: Mapping[str, list[dict[str, Any]]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    primary_ids = set(config["review"]["primary_visual_item_ids"])
    wins = losses = ties = incomparable = 0
    scenes: list[dict[str, Any]] = []
    for scene_id in SCENE_IDS:
        candidate = next(
            clip
            for clip in clips_by_family[candidate_family]
            if clip["scene_id"] == scene_id
        )
        reference = next(
            clip
            for clip in clips_by_family[reference_family]
            if clip["scene_id"] == scene_id
        )
        counts: dict[str, dict[str, int]] = {}
        for family, clip in (
            (candidate_family, candidate),
            (reference_family, reference),
        ):
            responses = [
                response
                for item_id, response in clip["responses"].items()
                if item_id in primary_ids
            ]
            counts[family] = {
                "pass": sum(response == "pass" for response in responses),
                "judgeable": sum(
                    response != "cannot_judge" for response in responses
                ),
            }
        comparable = (
            counts[candidate_family]["judgeable"]
            == counts[reference_family]["judgeable"]
        )
        if not comparable:
            outcome = "inconclusive"
            incomparable += 1
        elif counts[candidate_family]["pass"] > counts[reference_family]["pass"]:
            outcome = candidate_family
            wins += 1
        elif counts[candidate_family]["pass"] < counts[reference_family]["pass"]:
            outcome = reference_family
            losses += 1
        else:
            outcome = "tie"
            ties += 1
        scenes.append(
            {
                "scene_id": scene_id,
                "outcome": outcome,
                "judgeable_counts_equal": comparable,
                "candidate_primary_visual_pass_count": counts[candidate_family][
                    "pass"
                ],
                "reference_primary_visual_pass_count": counts[reference_family][
                    "pass"
                ],
            }
        )
    candidate_total = sum(
        clip["responses"][item_id] == "pass"
        for clip in clips_by_family[candidate_family]
        for item_id in primary_ids
    )
    reference_total = sum(
        clip["responses"][item_id] == "pass"
        for clip in clips_by_family[reference_family]
        for item_id in primary_ids
    )
    return {
        "candidate_family": candidate_family,
        "reference_family": reference_family,
        "primary_visual_pass_difference": candidate_total - reference_total,
        "scene_wins": wins,
        "scene_losses": losses,
        "scene_ties": ties,
        "incomparable_scenes": incomparable,
        "scenes": scenes,
    }


def _candidate_decision(
    family: str,
    metrics: Mapping[str, Mapping[str, Any]],
    versus_ltx: Mapping[str, Any],
    versus_seedance: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    labels = config["review"]["family_labels"]
    clear_rule = config["review"]["clear_task_advantage_over_ltx_rule"]
    competitive_rule = config["review"]["competitive_with_seedance_rule"]
    candidate_metrics = metrics[family]
    ltx_metrics = metrics["ltx"]
    seedance_metrics = metrics["seedance"]

    clear_criteria = {
        "media_valid_count": (
            candidate_metrics["clip_count"]
            == clear_rule["media_valid_count_required"]
        ),
        "minimum_total_primary_visual_pass_advantage": (
            versus_ltx["primary_visual_pass_difference"]
            >= clear_rule["minimum_total_primary_visual_pass_advantage"]
        ),
        "minimum_scene_wins": (
            versus_ltx["scene_wins"] >= clear_rule["minimum_scene_wins"]
        ),
        "maximum_scene_losses": (
            versus_ltx["scene_losses"] <= clear_rule["maximum_scene_losses"]
        ),
        "equal_judgeable_primary_items_per_scene": (
            versus_ltx["incomparable_scenes"] == 0
        ),
        "critical_failures_not_increased": (
            candidate_metrics["critical_visual_failure_count"]
            <= ltx_metrics["critical_visual_failure_count"]
        ),
        "maximum_safety_failures": (
            candidate_metrics["safety_failure_count"]
            <= clear_rule["maximum_safety_failures"]
        ),
    }
    competitive_criteria = {
        "media_valid_count": (
            candidate_metrics["clip_count"]
            == competitive_rule["media_valid_count_required"]
        ),
        "minimum_total_primary_visual_pass_difference": (
            versus_seedance["primary_visual_pass_difference"]
            >= competitive_rule["minimum_total_primary_visual_pass_difference"]
        ),
        "minimum_scene_wins_or_ties": (
            versus_seedance["scene_wins"] + versus_seedance["scene_ties"]
            >= competitive_rule["minimum_scene_wins_or_ties"]
        ),
        "maximum_additional_critical_failures": (
            candidate_metrics["critical_visual_failure_count"]
            - seedance_metrics["critical_visual_failure_count"]
            <= competitive_rule["maximum_additional_critical_failures"]
        ),
        "equal_judgeable_primary_items_per_scene": (
            versus_seedance["incomparable_scenes"] == 0
        ),
        "maximum_safety_failures": (
            candidate_metrics["safety_failure_count"]
            <= competitive_rule["maximum_safety_failures"]
        ),
    }
    if (
        versus_ltx["incomparable_scenes"]
        or versus_seedance["incomparable_scenes"]
        or candidate_metrics["clip_count"] != 4
    ):
        label = labels["inconclusive"]
    elif all(clear_criteria.values()):
        label = labels["clear_task_advantage"]
    elif all(competitive_criteria.values()):
        label = labels["competitive"]
    else:
        label = labels["below_screen"]
    return {
        "label": label,
        "clear_task_advantage_over_ltx": all(clear_criteria.values()),
        "competitive_with_seedance": all(competitive_criteria.values()),
        "clear_task_advantage_criteria": clear_criteria,
        "competitive_with_seedance_criteria": competitive_criteria,
    }


def _recommendation_markdown(summary: Mapping[str, Any]) -> str:
    labels = {
        "ltx": "LTX-2.3",
        "seedance": "Seedance 2.0",
        "gemini_omni_flash": "Gemini Omni Flash",
        "minimax_h3": "MiniMax H3",
    }
    lines = [
        "# Four-family public synthetic-video quality screen",
        "",
        "This is a single-rater, four-scene exploratory screen using only frozen "
        "public prompts. It is not a formal model ranking and does not authorize "
        "any hosted output as learner training data.",
        "",
        "## Family-level result",
        "",
        "| Family | Primary pass | Primary fail | Presentation pass | Critical failures |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in ALL_FAMILIES:
        metric = summary["qualitative"]["families"][family]
        primary = metric["counts_by_scope"]["primary_visual"]
        presentation = metric["counts_by_scope"]["secondary_presentation"]
        lines.append(
            f"| {labels[family]} | {primary['pass']} | {primary['fail']} | "
            f"{presentation['pass']} | {metric['critical_visual_failure_count']} |"
        )
    lines.extend(["", "## New-family decisions", ""])
    for family in NEW_FAMILIES:
        decision = summary["decisions"][family]
        lines.extend(
            [
                f"### {labels[family]}",
                "",
                f"**Screen label:** `{decision['label']}`",
                "",
                (
                    f"Primary-pass difference versus LTX: "
                    f"{summary['qualitative']['pairwise'][family]['versus_ltx']['primary_visual_pass_difference']:+d}. "
                    f"Primary-pass difference versus Seedance: "
                    f"{summary['qualitative']['pairwise'][family]['versus_seedance']['primary_visual_pass_difference']:+d}."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Cost and boundary",
            "",
            (
                "Maximum expected new charge: "
                f"${summary['cost']['maximum_expected_new_charge_usd']:.2f}. "
                "Actual provider invoices are separate billing sources and were "
                "not available to the runner."
            ),
            "",
            "All new native audio was discarded and replaced with the exact paired "
            "LTX AAC stream. ChildLens/BabyView-derived inputs remain prohibited. "
            "Hosted outputs remain blocked from learner training pending written "
            "provider and institutional clearance.",
            "",
            "## Provenance",
            "",
            f"- Bakeoff protocol SHA-256: `{summary['provenance']['bakeoff_protocol_sha256']}`",
            f"- Work-order SHA-256: `{summary['provenance']['work_order_sha256']}`",
            f"- Blinded QA bundle SHA-256: `{summary['provenance']['qa_bundle_sha256']}`",
            f"- Adapter commit: `{summary['provenance']['nursery_adapter_commit']}`",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_blinded_review(
    config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    completed_quality: Mapping[str, Any],
    work_order: Mapping[str, Any],
    *,
    repository_root: str | Path,
    run_root: str | Path,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
) -> dict[str, Any]:
    """Freeze all 16 QA records before opening the family mapping and scoring."""

    validate_work_order(
        config,
        public_pilot,
        completed_quality,
        repository_root,
        work_order,
    )
    root = Path(run_root).resolve()
    review_status = inspect_blinded_review_status(
        config,
        public_pilot,
        completed_quality,
        work_order,
        repository_root=repository_root,
        run_root=root,
    )
    _require(
        review_status["status"] == "READY_TO_UNBLIND",
        "all 16 blinded QA records must be complete before unblinding",
    )

    qa_records: dict[str, dict[str, Any]] = {}
    qa_hashes: dict[str, str] = {}
    rater_ids: set[str] = set()
    for scene in work_order["scenes"]:
        for card in scene["presentation"]:
            blind_id = str(card["blinded_display_id"])
            qa_path = root / "qa" / f"{blind_id}.json"
            qa = json.loads(qa_path.read_text())
            qa_records[blind_id] = qa
            qa_hashes[blind_id] = file_sha256(str(qa_path))
            rater_ids.add(str(qa["rater_id"]).strip())
    qa_bundle_sha256 = canonical_json_sha256(qa_hashes)
    freeze_path = root / "review" / "qa_freeze.json"
    freeze = {
        "schema_version": 1,
        "comparison_id": COMPARISON_ID,
        "run_id": work_order["run_id"],
        "frozen_at": utc_now(),
        "work_order_sha256": work_order["work_order_sha256"],
        "blinding_key_sha256": work_order["blinding_key_sha256"],
        "qa_file_sha256": qa_hashes,
        "qa_bundle_sha256": qa_bundle_sha256,
        "rater_id_set_sha256": canonical_json_sha256(sorted(rater_ids)),
    }
    if freeze_path.exists():
        existing = json.loads(freeze_path.read_text())
        _require(
            existing.get("qa_bundle_sha256") == qa_bundle_sha256,
            "QA files changed after the existing freeze",
        )
        freeze = existing
    else:
        _write_json(freeze_path, freeze)

    blinding_key = json.loads((root / "blinding_key.json").read_text())
    _require(
        canonical_json_sha256(blinding_key) == work_order["blinding_key_sha256"],
        "blinding key hash mismatch",
    )
    _require(set(blinding_key) == set(qa_records), "blinding and QA sets differ")
    references = verify_reference_media(
        config,
        public_pilot,
        completed_quality,
        repository_root,
        ffmpeg_executable=ffmpeg_executable,
        ffprobe_executable=ffprobe_executable,
    )
    execution = _verify_new_execution(
        config,
        public_pilot,
        work_order,
        root,
        ffmpeg_executable=ffmpeg_executable,
        ffprobe_executable=ffprobe_executable,
    )

    clips_by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in ALL_FAMILIES
    }
    for blind_id, qa in qa_records.items():
        mapping = blinding_key[blind_id]
        clips_by_family[mapping["family"]].append(
            {
                "scene_id": mapping["scene_id"],
                "blinded_display_id": blind_id,
                "responses": {
                    item["id"]: item["response"] for item in qa["items"]
                },
                "overall_note": qa.get("overall_note"),
            }
        )
    family_metrics = {
        family: _family_metrics(clips, config)
        for family, clips in clips_by_family.items()
    }
    pairwise: dict[str, dict[str, Any]] = {}
    decisions: dict[str, dict[str, Any]] = {}
    for family in NEW_FAMILIES:
        versus_ltx = _pairwise_comparison(
            family,
            "ltx",
            clips_by_family,
            config,
        )
        versus_seedance = _pairwise_comparison(
            family,
            "seedance",
            clips_by_family,
            config,
        )
        pairwise[family] = {
            "versus_ltx": versus_ltx,
            "versus_seedance": versus_seedance,
        }
        decisions[family] = _candidate_decision(
            family,
            family_metrics,
            versus_ltx,
            versus_seedance,
            config,
        )

    commits = {
        record["provider"]["nursery_adapter_commit"]
        for record in execution["records"].values()
    }
    _require(len(commits) == 1, "new-family adapter commits differ")
    reference_records = {
        family: [
            {
                "attempt_id": references[family][scene_id]["attempt_id"],
                "scene_id": scene_id,
                "final_sha256": (
                    references[family][scene_id]["record"]["final_media"]["sha256"]
                ),
            }
            for scene_id in SCENE_IDS
        ]
        for family in REFERENCE_FAMILIES
    }
    new_records = {
        family: [
            {
                "attempt_id": attempt_id,
                "scene_id": record["attempt"]["scene_id"],
                "request_sha256": record["attempt"]["request_sha256"],
                "final_sha256": record["final_media"]["sha256"],
                "status": record["status"],
            }
            for attempt_id, record in execution["records"].items()
            if record["attempt"]["family"] == family
        ]
        for family in NEW_FAMILIES
    }
    summary = {
        "schema_version": 1,
        "comparison_id": COMPARISON_ID,
        "run_id": work_order["run_id"],
        "status": "COMPLETE_QUALITATIVE_SCREEN",
        "finalized_at": utc_now(),
        "technical": {
            "reference_families": reference_records,
            "new_families": new_records,
            "new_attempt_count": execution["planned_attempt_count"],
            "new_media_valid_count": execution["media_valid_count"],
            "new_generated_attempt_failure_count": execution[
                "generated_attempt_failure_count"
            ],
            "canonical_delivery": {
                "width": 1280,
                "height": 704,
                "fps": 24,
                "frame_count": 121,
                "paired_ltx_audio_identity_required": True,
            },
        },
        "qualitative": {
            "profile": config["review"]["profile"],
            "families": family_metrics,
            "pairwise": pairwise,
            "clip_notes": clips_by_family,
        },
        "decisions": decisions,
        "cost": {
            "currency": "USD",
            "maximum_expected_gemini_charge_usd": config["provider_cost"]["gemini"][
                "maximum_expected_charge_usd"
            ],
            "maximum_expected_minimax_charge_usd": config["provider_cost"][
                "minimax"
            ]["maximum_expected_charge_usd"],
            "maximum_expected_new_charge_usd": float(planned_cost_usd(config)),
            "actual_provider_invoice_usd": None,
        },
        "authorization_and_boundary": {
            "public_only_comparison_authorized": True,
            "restricted_or_child_derived_input_used": False,
            "childlens_or_babyview_tuning_performed": False,
            "scientific_training_use_authorized": False,
            "gemini_output_as_training_data": config["governance"][
                "gemini_output_as_training_data"
            ],
            "minimax_output_as_training_data": config["governance"][
                "minimax_output_as_training_data"
            ],
        },
        "provenance": {
            "bakeoff_protocol_sha256": work_order["bakeoff_protocol_sha256"],
            "public_pilot_protocol_sha256": work_order[
                "public_pilot_protocol_sha256"
            ],
            "completed_quality_protocol_sha256": work_order[
                "completed_quality_protocol_sha256"
            ],
            "completed_quality_result_file_sha256": work_order[
                "completed_quality_result_file_sha256"
            ],
            "work_order_sha256": work_order["work_order_sha256"],
            "blinding_key_sha256": work_order["blinding_key_sha256"],
            "qa_bundle_sha256": freeze["qa_bundle_sha256"],
            "nursery_adapter_commit": next(iter(commits)),
        },
    }
    summary_path = root / "review" / "review_summary.json"
    recommendation_path = root / "review" / "recommendation.md"
    _write_json(summary_path, summary)
    recommendation_path.parent.mkdir(parents=True, exist_ok=True)
    recommendation_path.write_text(_recommendation_markdown(summary))
    return {
        "status": summary["status"],
        "decisions": decisions,
        "summary": str(summary_path),
        "recommendation": str(recommendation_path),
        "qa_freeze": str(freeze_path),
    }


def load_default_configs(
    bakeoff_config_path: str | Path,
    public_pilot_path: str | Path,
    completed_quality_path: str | Path,
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bakeoff = load_bakeoff_config(bakeoff_config_path)
    public = load_public_pilot_config(public_pilot_path)
    completed_quality = load_quality_config(completed_quality_path)
    validate_quality_config(completed_quality, public)
    validate_bakeoff_config(
        bakeoff,
        public,
        completed_quality,
        repository_root,
    )
    return bakeoff, public, completed_quality
