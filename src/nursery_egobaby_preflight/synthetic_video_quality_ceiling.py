"""Public-only LTX-2.3 versus Seedance 2.0 quality-ceiling workflow."""

from __future__ import annotations

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

COMPARISON_ID = "synthetic-video-ltx-seedance-quality-ceiling"
SCENE_IDS = ("pick_up", "transfer", "occlusion_persistence", "action_transition")
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
FAL_CDN_TOKEN_URL = (
    "https://rest.fal.ai/storage/auth/token?storage_type=fal-cdn-v3"
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(value))


def load_quality_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    return config


def validate_quality_config(
    config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
) -> None:
    """Reject changes that weaken comparability, privacy, cost, or governance."""

    _require(config.get("schema_version") == 1, "schema_version must be 1")
    _require(config.get("comparison_id") == COMPARISON_ID, "unexpected comparison_id")
    authorization = config.get("authorization", {})
    _require(
        authorization.get("public_only_quality_ceiling_authorized") is True,
        "public-only quality ceiling is not authorized",
    )
    _require(
        authorization.get("scientific_training_use_authorized") is False,
        "Seedance training use must remain unauthorized",
    )
    _require(
        authorization.get("governed_or_child_derived_generator_work_authorized")
        is False,
        "governed generator work must remain unauthorized",
    )

    distinction = config.get("scientific_distinction", {})
    _require(
        distinction.get("completed_open_model_protocol_sha256")
        == canonical_json_sha256(public_pilot),
        "completed public-pilot protocol hash mismatch",
    )
    prohibited = " ".join(config.get("privacy_boundary", {}).get("prohibited", []))
    _require(
        "ChildLens" in prohibited and "BabyView" in prohibited,
        "restricted input sources must be prohibited",
    )
    privacy = config["privacy_boundary"]
    _require(
        privacy.get("provider_request_payload_storage") is False,
        "provider request payload storage must be disabled",
    )
    _require(
        privacy.get("provider_output_acl") == "private_to_calling_fal_account",
        "provider outputs must use the private ACL",
    )
    _require(
        privacy.get("provider_output_expiration_seconds") == 86400,
        "provider output expiration must remain one day",
    )

    baseline = config.get("baseline", {})
    _require(baseline.get("family") == "ltx", "baseline must be LTX")
    _require(
        baseline.get("run_id") == "public-preview-20260730",
        "unexpected LTX baseline run",
    )
    _require(
        baseline.get("local_run_root")
        == "runs/synthetic_video_public_pilot/public-preview-20260730/families/ltx",
        "unexpected LTX baseline path",
    )
    _require(baseline.get("seed") == 314159, "unexpected LTX baseline seed")
    _require(
        baseline.get("expected_attempt_count") == len(SCENE_IDS),
        "baseline must contain four attempts",
    )

    candidate = config.get("candidate", {})
    _require(candidate.get("family") == "seedance", "candidate must be Seedance")
    _require(candidate.get("provider") == "fal", "provider must be fal")
    _require(
        candidate.get("endpoint") == "bytedance/seedance-2.0/text-to-video",
        "comparison must use the standard Seedance 2.0 text-to-video endpoint",
    )
    _require(
        candidate.get("endpoint_url")
        == "https://queue.fal.run/bytedance/seedance-2.0/text-to-video",
        "unexpected Seedance endpoint URL",
    )
    _require(
        candidate.get("credential_environment_variable") == "FAL_KEY",
        "unexpected credential variable",
    )
    request = candidate.get("request", {})
    expected_request = {
        "resolution": "720p",
        "duration": "5",
        "aspect_ratio": "16:9",
        "generate_audio": False,
        "bitrate_mode": "high",
        "end_user_id": "nursery-public-quality-ceiling",
    }
    _require(request == expected_request, "Seedance request settings changed")
    _require(
        candidate.get("seed_control") == "UNSUPPORTED_BY_CURRENT_LIVE_INPUT_SCHEMA",
        "Seedance seed-control limitation changed",
    )
    _require(
        candidate.get("returned_seed_policy")
        == "record the provider-returned generation seed in the attempt manifest",
        "Seedance returned-seed policy changed",
    )
    expected_lifecycle = canonical_json_bytes(
        {
            "expiration_duration_seconds": 86400,
            "initial_acl": {"default": "forbid", "rules": []},
        }
    ).decode()
    expected_headers = {
        "X-Fal-No-Retry": "1",
        "X-Fal-Store-IO": "0",
        "X-Fal-Object-Lifecycle-Preference": expected_lifecycle,
    }
    _require(
        candidate.get("platform_headers") == expected_headers,
        "fal privacy or no-retry headers changed",
    )
    _require(
        candidate.get("provider_retries") == 0
        and candidate.get("protocol_retries") == 0
        and candidate.get("attempts_per_scene") == 1,
        "comparison must retain one request per scene and no retries",
    )

    comparison = config.get("comparison", {})
    _require(
        tuple(comparison.get("scene_ids", ())) == SCENE_IDS,
        "scene order or identity changed",
    )
    _require(
        comparison.get("baseline_seed") == baseline.get("seed") == 314159,
        "baseline seed must remain 314159",
    )
    _require(
        comparison.get("candidate_seed_comparability")
        == (
            "The live Seedance endpoint does not expose an input seed, so its "
            "provider-selected seed cannot be matched to the LTX seed; this "
            "limitation must be reported."
        ),
        "candidate seed-comparability limitation changed",
    )
    _require(
        comparison.get("prompt_changes_for_seedance") == "none",
        "provider-specific prompt edits are prohibited",
    )
    _require(
        comparison.get("no_polished_subset") is True,
        "the full four-scene result must be retained",
    )

    review = config.get("review", {})
    expected_item_ids = tuple(
        item["id"] for item in public_pilot["human_qa"]["items"]
    )
    visual_item_ids = tuple(review.get("visual_item_ids", ()))
    audio_control_item_ids = tuple(review.get("audio_control_item_ids", ()))
    _require(
        review.get("profile")
        == "single-rater blinded qualitative screening; not a formal model-selection result",
        "unexpected review profile",
    )
    _require(
        review.get("required_unique_raters") == 1,
        "quality-ceiling review must use one blinded rater",
    )
    _require(
        tuple(review.get("response_values", ()))
        == tuple(public_pilot["human_qa"]["response_values"]),
        "review response values changed",
    )
    _require(
        tuple(review.get("confidence_values", ()))
        == tuple(public_pilot["human_qa"]["confidence_values"]),
        "review confidence values changed",
    )
    _require(
        visual_item_ids
        == (
            "continuous_egocentric_shot",
            "anatomy",
            "contact_action",
            "identity",
            "transition_order",
            "referent_timing",
            "safety",
        ),
        "visual review items changed",
    )
    _require(
        audio_control_item_ids == ("speech",),
        "speech must remain the sole audio-control item",
    )
    _require(
        set(visual_item_ids + audio_control_item_ids) == set(expected_item_ids),
        "review items do not cover the frozen QA instrument exactly",
    )
    _require(
        tuple(review.get("critical_visual_item_ids", ()))
        == ("anatomy", "contact_action", "identity", "transition_order", "safety"),
        "critical visual review items changed",
    )
    _require(
        review.get("material_visual_win_rule")
        == {
            "candidate_media_valid_count_required": 4,
            "minimum_total_visual_pass_advantage": 4,
            "minimum_candidate_scene_wins": 2,
            "maximum_candidate_scene_losses": 0,
            "equal_judgeable_visual_items_required_per_scene": True,
            "candidate_critical_failures_must_not_exceed_baseline": True,
            "maximum_candidate_safety_failures": 0,
        },
        "material visual-win rule changed",
    )
    _require(
        review.get("decision_labels")
        == {
            "material_win": "PURSUE_WRITTEN_PROVIDER_AND_INSTITUTIONAL_CLEARANCE",
            "no_material_win": "NO_CLEAR_ADVANTAGE_KEEP_LTX_REPRODUCIBLE_BASELINE",
            "inconclusive": "INCONCLUSIVE_REQUIRES_NEW_AUTHORIZATION_TO_REPEAT",
        },
        "review decision labels changed",
    )

    provider_cost = config.get("provider_cost", {})
    rate = Decimal(str(provider_cost.get("standard_720p_text_to_video_usd_per_second")))
    count = int(provider_cost.get("planned_scene_count", 0))
    seconds = int(provider_cost.get("planned_seconds_per_scene", 0))
    computed = rate * count * seconds
    _require(count == 4 and seconds == 5, "planned output must be four five-second clips")
    _require(
        computed == Decimal(str(provider_cost.get("maximum_generation_charge_usd"))),
        "frozen cost ceiling does not match rate, scenes, and duration",
    )
    _require(
        provider_cost.get("paid_launch_requires_explicit_spend_confirmation") is True,
        "paid launch must require explicit confirmation",
    )
    _require(
        config.get("governance", {}).get("seedance_output_as_training_data")
        == "BLOCKED_PENDING_WRITTEN_PROVIDER_AND_INSTITUTIONAL_CLEARANCE",
        "Seedance output training gate changed",
    )


def planned_cost_usd(config: Mapping[str, Any]) -> Decimal:
    cost = config["provider_cost"]
    return (
        Decimal(str(cost["standard_720p_text_to_video_usd_per_second"]))
        * int(cost["planned_scene_count"])
        * int(cost["planned_seconds_per_scene"])
    )


def _attempt_id(family: str, scene_id: str, seed: int) -> str:
    return f"{family}__{scene_id}__s{seed}__a1__modular"


def _candidate_attempt_id(scene_id: str) -> str:
    return f"seedance__{scene_id}__sprovider__a1__modular"


def _blind_id(comparison_id: str, run_id: str, attempt_id: str) -> str:
    return canonical_json_sha256(
        {"comparison_id": comparison_id, "run_id": run_id, "attempt_id": attempt_id}
    )[:12]


def compile_comparison_work_order(
    quality_config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Compile four immutable API requests and their paired LTX references."""

    validate_quality_config(quality_config, public_pilot)
    _require(
        bool(SAFE_ID.fullmatch(run_id)),
        "run_id must contain only lowercase letters, digits, underscores, and hyphens",
    )
    baseline_seed = int(quality_config["baseline"]["seed"])
    baseline_root = Path(quality_config["baseline"]["local_run_root"])
    candidate_settings = quality_config["candidate"]["request"]
    attempts: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    blinding: dict[str, dict[str, str]] = {}

    for scene_id in SCENE_IDS:
        prompt = compile_prompt(public_pilot, "ltx", scene_id)
        ltx_id = _attempt_id("ltx", scene_id, baseline_seed)
        seedance_id = _candidate_attempt_id(scene_id)
        request = {"prompt": prompt, **candidate_settings}
        seedance_paths = {
            "request": f"attempts/{seedance_id}/request.json",
            "submission": f"attempts/{seedance_id}/submission.json",
            "status": f"attempts/{seedance_id}/provider_status.json",
            "provider_response": f"attempts/{seedance_id}/provider_response.json",
            "raw_video": f"attempts/{seedance_id}/raw.mp4",
            "video_only": f"attempts/{seedance_id}/video_only.mp4",
            "final_video": f"attempts/{seedance_id}/final.mp4",
            "record": f"attempts/{seedance_id}/attempt.json",
            "failure": f"attempts/{seedance_id}/failure.json",
        }
        baseline_paths = {
            "final_video": str(
                baseline_root / "attempts" / ltx_id / "final.mp4"
            ),
            "record": str(
                baseline_root / "attempts" / ltx_id / "attempt.json"
            ),
        }
        attempts.append(
            {
                "attempt_id": seedance_id,
                "family": "seedance",
                "scene_id": scene_id,
                "provider_seed_requested": None,
                "provider_seed_policy": quality_config["candidate"][
                    "returned_seed_policy"
                ],
                "attempt_number": 1,
                "mode": "modular",
                "prompt": prompt,
                "prompt_sha256": canonical_json_sha256(prompt),
                "request": request,
                "request_sha256": canonical_json_sha256(request),
                "planned_charge_usd": float(
                    Decimal(
                        str(
                            quality_config["provider_cost"][
                                "standard_720p_text_to_video_usd_per_second"
                            ]
                        )
                    )
                    * int(candidate_settings["duration"])
                ),
                "status": "planned",
                "paths": seedance_paths,
            }
        )
        cards = []
        for family, attempt_id, paths in (
            ("ltx", ltx_id, baseline_paths),
            ("seedance", seedance_id, seedance_paths),
        ):
            blind_id = _blind_id(COMPARISON_ID, run_id, attempt_id)
            cards.append(
                {
                    "family": family,
                    "attempt_id": attempt_id,
                    "blinded_display_id": blind_id,
                    "paths": paths,
                }
            )
            blinding[blind_id] = {
                "family": family,
                "attempt_id": attempt_id,
                "scene_id": scene_id,
            }
        cards.sort(
            key=lambda item: canonical_json_sha256(
                {
                    "comparison_id": COMPARISON_ID,
                    "run_id": run_id,
                    "scene_id": scene_id,
                    "attempt_id": item["attempt_id"],
                }
            )
        )
        for index, card in enumerate(cards):
            card["slot"] = chr(ord("A") + index)
        pairs.append({"scene_id": scene_id, "presentation": cards})

    work_order = {
        "schema_version": 1,
        "comparison_id": COMPARISON_ID,
        "run_id": run_id,
        "compiled_at": utc_now(),
        "quality_protocol_sha256": canonical_json_sha256(quality_config),
        "public_pilot_protocol_sha256": canonical_json_sha256(public_pilot),
        "purpose": quality_config["scientific_distinction"]["new_question"],
        "scientific_training_use_authorized": False,
        "privacy_boundary": quality_config["privacy_boundary"],
        "candidate_endpoint": quality_config["candidate"]["endpoint"],
        "planned_cost_usd": float(planned_cost_usd(quality_config)),
        "attempts": attempts,
        "pairs": pairs,
        "blinding_key_sha256": canonical_json_sha256(blinding),
    }
    work_order["work_order_sha256"] = canonical_json_sha256(work_order)
    work_order["blinding_key"] = blinding
    return work_order


def validate_work_order(
    quality_config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    work_order: Mapping[str, Any],
) -> None:
    """Verify a persisted work order before it can drive API calls or review."""

    validate_quality_config(quality_config, public_pilot)
    _require(
        work_order.get("comparison_id") == COMPARISON_ID,
        "work-order comparison ID changed",
    )
    _require(
        bool(SAFE_ID.fullmatch(str(work_order.get("run_id", "")))),
        "work-order run ID is invalid",
    )
    _require(
        work_order.get("quality_protocol_sha256")
        == canonical_json_sha256(quality_config),
        "work-order quality protocol hash changed",
    )
    _require(
        work_order.get("public_pilot_protocol_sha256")
        == canonical_json_sha256(public_pilot),
        "work-order public-pilot protocol hash changed",
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
    expected = compile_comparison_work_order(
        quality_config,
        public_pilot,
        str(work_order["run_id"]),
    )
    for field in (
        "schema_version",
        "comparison_id",
        "quality_protocol_sha256",
        "public_pilot_protocol_sha256",
        "purpose",
        "scientific_training_use_authorized",
        "privacy_boundary",
        "candidate_endpoint",
        "planned_cost_usd",
        "attempts",
        "pairs",
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
    comparison_id: str,
    run_id: str,
    scene_id: str,
    blind_id: str,
    public_pilot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "comparison_id": comparison_id,
        "run_id": run_id,
        "scene_id": scene_id,
        "blinded_display_id": blind_id,
        "rater_id": None,
        "rated_at": None,
        "items": [
            {
                "id": item["id"],
                "question": item["question"],
                "response": None,
                "confidence": None,
                "note": None,
            }
            for item in public_pilot["human_qa"]["items"]
        ],
        "overall_note": None,
    }


def write_comparison_work_order(
    run_root: str | Path,
    work_order: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
) -> Path:
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    public_order = {
        key: value for key, value in work_order.items() if key != "blinding_key"
    }
    _write_json(root / "work_order.json", public_order)
    blinding_key_path = root / "blinding_key.json"
    _write_json(blinding_key_path, work_order["blinding_key"])
    blinding_key_path.chmod(0o600)
    for attempt in work_order["attempts"]:
        _write_json(root / attempt["paths"]["request"], attempt["request"])
        _write_json(
            root / "attempts" / attempt["attempt_id"] / "planned_attempt.json",
            attempt,
        )
    for pair in work_order["pairs"]:
        for card in pair["presentation"]:
            _write_json(
                root / "qa" / f"{card['blinded_display_id']}.json",
                _qa_template(
                    work_order["comparison_id"],
                    work_order["run_id"],
                    pair["scene_id"],
                    card["blinded_display_id"],
                    public_pilot,
                ),
            )
    _write_json(
        root / "run_status.json",
        {
            "schema_version": 1,
            "comparison_id": work_order["comparison_id"],
            "run_id": work_order["run_id"],
            "status": "planned",
            "completed_attempt_ids": [],
            "failed_attempt_ids": [],
        },
    )
    return root / "work_order.json"


def verify_ltx_baseline(
    quality_config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    repository_root: str | Path,
) -> dict[str, dict[str, Any]]:
    """Verify the exact local LTX baseline before any paid request."""

    root = Path(repository_root).resolve()
    baseline_root = root / quality_config["baseline"]["local_run_root"]
    status = json.loads((baseline_root / "run_status.json").read_text())
    _require(status.get("status") == "complete", "LTX baseline run is not complete")
    expected_ids = {
        _attempt_id("ltx", scene_id, quality_config["baseline"]["seed"])
        for scene_id in SCENE_IDS
    }
    _require(
        set(status.get("completed_attempt_ids", [])) == expected_ids,
        "LTX baseline completed-attempt set changed",
    )
    _require(
        status.get("failed_attempt_ids") == [],
        "LTX baseline contains failed attempts",
    )

    verified: dict[str, dict[str, Any]] = {}
    for scene_id in SCENE_IDS:
        attempt_id = _attempt_id("ltx", scene_id, quality_config["baseline"]["seed"])
        attempt_root = baseline_root / "attempts" / attempt_id
        record = json.loads((attempt_root / "attempt.json").read_text())
        final_video = attempt_root / "final.mp4"
        expected_prompt = compile_prompt(public_pilot, "ltx", scene_id)
        _require(record.get("status") == "media_valid", f"{attempt_id} is not media-valid")
        _require(
            record["attempt"]["prompt_sha256"] == canonical_json_sha256(expected_prompt),
            f"{attempt_id} prompt hash changed",
        )
        _require(
            record["final_media"]["sha256"] == file_sha256(str(final_video)),
            f"{attempt_id} final hash mismatch",
        )
        verified[scene_id] = {
            "attempt_id": attempt_id,
            "record": record,
            "final_video": final_video,
        }
    return verified


class FalQueueClient:
    """Minimal secret-safe client for fal's persistent queue and private CDN."""

    def __init__(
        self,
        api_key: str,
        platform_headers: Mapping[str, str],
        *,
        timeout_seconds: float = 60.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        if not api_key:
            raise ValueError("FAL_KEY is required")
        self._api_key = api_key
        self._platform_headers = dict(platform_headers)
        self._timeout_seconds = timeout_seconds
        self._opener = opener
        self._cdn_token: str | None = None

    @staticmethod
    def _require_https_host(
        url: str,
        *,
        exact_host: str | None = None,
        host_suffix: str | None = None,
    ) -> None:
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").lower()
        valid_host = (
            host == exact_host
            if exact_host is not None
            else host == host_suffix or host.endswith(f".{host_suffix}")
        )
        if parsed.scheme != "https" or not valid_host or parsed.username is not None:
            raise ValueError("refusing to send credentials to an untrusted URL")

    def _json_request(
        self,
        method: str,
        url: str,
        *,
        body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        data = canonical_json_bytes(body) if body is not None else None
        headers = {
            "Authorization": f"Key {self._api_key}",
            "Accept": "application/json",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        if extra_headers:
            headers.update(extra_headers)
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:2000]
            raise RuntimeError(
                f"fal API returned HTTP {error.code}: {detail}"
            ) from None
        except urllib.error.URLError as error:
            raise RuntimeError(f"fal API request failed: {error.reason}") from None
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise RuntimeError("fal API returned a non-object JSON response")
        return decoded

    def submit(self, endpoint_url: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._require_https_host(endpoint_url, exact_host="queue.fal.run")
        return self._json_request(
            "POST",
            endpoint_url,
            body=payload,
            extra_headers=self._platform_headers,
        )

    def status(self, status_url: str) -> dict[str, Any]:
        self._require_https_host(status_url, exact_host="queue.fal.run")
        separator = "&" if urllib.parse.urlparse(status_url).query else "?"
        return self._json_request("GET", f"{status_url}{separator}logs=1")

    def result(self, response_url: str) -> dict[str, Any]:
        self._require_https_host(response_url, exact_host="queue.fal.run")
        return self._json_request("GET", response_url)

    def _get_cdn_token(self) -> str:
        if self._cdn_token is None:
            response = self._json_request("POST", FAL_CDN_TOKEN_URL, body={})
            token = response.get("token")
            if not isinstance(token, str) or not token:
                raise RuntimeError("fal CDN token response is missing token")
            self._cdn_token = token
        return self._cdn_token

    def download_private_file(self, url: str, output_path: str | Path) -> None:
        self._require_https_host(url, host_suffix="fal.media")
        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        request = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self._get_cdn_token()}"},
            method="GET",
        )
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                with partial.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            os.replace(partial, destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise


def _audio_payload_sha256(
    video: str | Path,
    ffmpeg_executable: str = "ffmpeg",
) -> str:
    result = subprocess.run(
        [
            ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(Path(video).resolve()),
            "-map",
            "0:a:0",
            "-c",
            "copy",
            "-f",
            "hash",
            "-hash",
            "sha256",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().split("=", 1)[1]


def normalize_seedance_with_ltx_audio(
    public_pilot: Mapping[str, Any],
    *,
    raw_video: str | Path,
    ltx_final_video: str | Path,
    video_only: str | Path,
    final_video: str | Path,
    ffmpeg_executable: str = "ffmpeg",
) -> dict[str, str]:
    """Normalize Seedance video and stream-copy the exact paired LTX audio."""

    delivery = public_pilot["delivery"]
    raw = str(Path(raw_video).resolve())
    baseline = str(Path(ltx_final_video).resolve())
    video_path = str(Path(video_only).resolve())
    final = str(Path(final_video).resolve())
    common = [ffmpeg_executable, "-hide_banner", "-loglevel", "error", "-y"]
    filter_graph = (
        f"fps={delivery['fps']},"
        "scale=1280:720:force_original_aspect_ratio=increase:flags=lanczos,"
        "crop=1280:704,"
        "tpad=stop_mode=clone:stop_duration=1"
    )
    subprocess.run(
        [
            *common,
            "-i",
            raw,
            "-an",
            "-vf",
            filter_graph,
            "-frames:v",
            str(delivery["num_frames"]),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            video_path,
        ],
        check=True,
    )
    subprocess.run(
        [
            *common,
            "-i",
            video_path,
            "-i",
            baseline,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c",
            "copy",
            "-t",
            f"{delivery['nominal_duration_s']:.9f}",
            "-movflags",
            "+faststart",
            final,
        ],
        check=True,
    )
    baseline_hash = _audio_payload_sha256(baseline, ffmpeg_executable)
    final_hash = _audio_payload_sha256(final, ffmpeg_executable)
    if baseline_hash != final_hash:
        raise RuntimeError("candidate final does not contain the exact paired LTX audio")
    return {
        "baseline_audio_payload_sha256": baseline_hash,
        "candidate_audio_payload_sha256": final_hash,
    }


def _sanitized_provider_output(response: Mapping[str, Any]) -> dict[str, Any]:
    video = response.get("video", {})
    url = str(video.get("url", ""))
    return {
        "returned_seed": response.get("seed"),
        "video": {
            "url_sha256": canonical_json_sha256(url),
            "host": urllib.parse.urlparse(url).hostname,
            "content_type": video.get("content_type"),
            "file_name": video.get("file_name"),
            "file_size": video.get("file_size"),
        },
    }


def _run_one_attempt(
    quality_config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    attempt: Mapping[str, Any],
    baseline: Mapping[str, Any],
    run_root: Path,
    client: FalQueueClient,
    *,
    implementation_commit: str,
    poll_interval_seconds: float,
    ffmpeg_executable: str,
    ffprobe_executable: str,
) -> dict[str, Any]:
    paths = {key: run_root / value for key, value in attempt["paths"].items()}
    if paths["record"].exists():
        existing = json.loads(paths["record"].read_text())
        if existing.get("status") == "media_valid":
            if not paths["final_video"].is_file():
                raise RuntimeError("completed candidate record is missing final media")
            if (
                existing.get("final_media", {}).get("sha256")
                != file_sha256(str(paths["final_video"]))
            ):
                raise RuntimeError("completed candidate final hash mismatch")
            final_summary = media_summary(paths["final_video"], ffprobe_executable)
            if validate_final_media(public_pilot, final_summary):
                raise RuntimeError("completed candidate final no longer conforms")
            return existing

    started_at = utc_now()
    try:
        if paths["submission"].exists():
            submission = json.loads(paths["submission"].read_text())
        else:
            submission = client.submit(
                quality_config["candidate"]["endpoint_url"],
                attempt["request"],
            )
            _write_json(paths["submission"], submission)
        for required in ("request_id", "response_url", "status_url"):
            if not submission.get(required):
                raise RuntimeError(f"fal submission is missing {required}")

        while True:
            status = client.status(str(submission["status_url"]))
            _write_json(paths["status"], status)
            state = status.get("status")
            if state == "COMPLETED":
                if status.get("error"):
                    raise RuntimeError(
                        f"fal request completed with {status.get('error_type')}: "
                        f"{status['error']}"
                    )
                break
            if state not in {"IN_QUEUE", "IN_PROGRESS"}:
                raise RuntimeError(f"unexpected fal request state {state!r}")
            time.sleep(poll_interval_seconds)

        response = client.result(str(submission["response_url"]))
        _write_json(paths["provider_response"], response)
        video_url = response.get("video", {}).get("url")
        if not isinstance(video_url, str) or not video_url:
            raise RuntimeError("fal result is missing video.url")
        client.download_private_file(video_url, paths["raw_video"])
        audio_hashes = normalize_seedance_with_ltx_audio(
            public_pilot,
            raw_video=paths["raw_video"],
            ltx_final_video=baseline["final_video"],
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
                "name": "fal",
                "endpoint": quality_config["candidate"]["endpoint"],
                "nursery_adapter_commit": implementation_commit,
                "request_id": submission["request_id"],
                "request_sha256": attempt["request_sha256"],
                "automatic_retries_disabled": True,
                "payload_storage_disabled": True,
                "output_acl": quality_config["privacy_boundary"][
                    "provider_output_acl"
                ],
                "output_expiration_seconds": quality_config["privacy_boundary"][
                    "provider_output_expiration_seconds"
                ],
                "terminal_status": status,
                "output": _sanitized_provider_output(response),
            },
            "runtime": {
                "planned_charge_usd": attempt["planned_charge_usd"],
                "provider_inference_seconds": (status.get("metrics") or {}).get(
                    "inference_time"
                ),
            },
            "raw_media": raw_summary,
            "final_media": final_summary,
            "paired_ltx_attempt_id": baseline["attempt_id"],
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


def execute_comparison(
    quality_config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    work_order: Mapping[str, Any],
    *,
    repository_root: str | Path,
    run_root: str | Path,
    approved_spend_usd: Decimal,
    api_key: str,
    implementation_commit: str = "",
    poll_interval_seconds: float = 10.0,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
) -> dict[str, Any]:
    """Run the four paid requests sequentially, persisting each before the next."""

    validate_work_order(quality_config, public_pilot, work_order)
    ceiling = planned_cost_usd(quality_config)
    _require(
        approved_spend_usd == ceiling,
        f"approved spend must exactly match the frozen ceiling {ceiling}",
    )
    _require(
        bool(re.fullmatch(r"[0-9a-f]{40}", implementation_commit)),
        "a clean 40-character Git execution commit is required",
    )
    _require(
        Decimal(str(work_order["planned_cost_usd"])) == ceiling,
        "work-order cost differs from frozen ceiling",
    )
    root = Path(run_root).resolve()
    baseline_by_scene = verify_ltx_baseline(
        quality_config,
        public_pilot,
        repository_root,
    )
    _write_json(
        root / "approval.json",
        {
            "schema_version": 1,
            "approved_spend_usd": float(approved_spend_usd),
            "frozen_maximum_generation_charge_usd": float(ceiling),
            "nursery_adapter_commit": implementation_commit,
            "recorded_at": utc_now(),
        },
    )
    client = FalQueueClient(
        api_key,
        quality_config["candidate"]["platform_headers"],
    )
    completed: list[str] = []
    failed: list[str] = []
    started_at = utc_now()
    for attempt in work_order["attempts"]:
        try:
            record = _run_one_attempt(
                quality_config,
                public_pilot,
                attempt,
                baseline_by_scene[attempt["scene_id"]],
                root,
                client,
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
                "comparison_id": work_order["comparison_id"],
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
                "comparison_id": work_order["comparison_id"],
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
        "comparison_id": work_order["comparison_id"],
        "run_id": work_order["run_id"],
        "status": "complete",
        "nursery_adapter_commit": implementation_commit,
        "started_at": started_at,
        "finished_at": utc_now(),
        "completed_attempt_ids": completed,
        "failed_attempt_ids": failed,
        "planned_generation_charge_usd": float(ceiling),
    }
    _write_json(root / "run_status.json", status)
    return status


def render_blinded_gallery(
    quality_config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    work_order: Mapping[str, Any],
    *,
    repository_root: str | Path,
    run_root: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Render a family-blinded gallery; the key remains a separate local record."""

    validate_work_order(quality_config, public_pilot, work_order)
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
    sections = []
    for pair in work_order["pairs"]:
        scene = scene_by_id(public_pilot, pair["scene_id"])
        cards = []
        for card in pair["presentation"]:
            if card["family"] == "ltx":
                video = repository / card["paths"]["final_video"]
            else:
                video = root / card["paths"]["final_video"]
            qa = root / "qa" / f"{card['blinded_display_id']}.json"
            alias = alias_root / f"{card['blinded_display_id']}.mp4"
            if alias.exists() or alias.is_symlink():
                alias.unlink()
            if video.exists():
                try:
                    os.link(video, alias)
                except OSError:
                    shutil.copy2(video, alias)
            video_rel = os.path.relpath(alias, output.parent)
            qa_rel = os.path.relpath(qa, output.parent)
            missing = (
                ""
                if video.exists()
                else '<p class="missing">Media not present yet.</p>'
            )
            cards.append(
                f"""
                <article class="attempt">
                  <h3>Clip {html.escape(card['slot'])}</h3>
                  <video controls preload="metadata" src="{html.escape(video_rel)}"></video>
                  {missing}
                  <p>Blinded ID: <code>{html.escape(card['blinded_display_id'])}</code></p>
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
  <title>Blinded public quality-ceiling comparison</title>
  <style>
    :root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin: 0 auto; max-width: 1180px; padding: 2rem; background: #111827; color: #f3f4f6; }}
    header, section {{ background: #1f2937; border: 1px solid #374151; border-radius: 14px; margin: 1rem 0; padding: 1.25rem; }}
    h1, h2, h3 {{ margin-top: 0; }}
    .warning {{ color: #fbbf24; }}
    .attempts {{ display: grid; grid-template-columns: repeat(2, minmax(300px, 1fr)); gap: 1rem; }}
    .attempt {{ background: #111827; border-radius: 10px; padding: 1rem; }}
    video {{ width: 100%; aspect-ratio: 1280 / 704; background: #000; border-radius: 8px; }}
    .utterance {{ font-size: 1.1rem; }}
    .missing {{ color: #fca5a5; }}
    a {{ color: #93c5fd; }}
    code {{ color: #d1fae5; }}
  </style>
</head>
<body>
  <header>
    <h1>Blinded public-only quality-ceiling comparison</h1>
    <p>Blinded run <code>{html.escape(str(work_order['work_order_sha256'])[:12])}</code></p>
    <p class="warning">Do not open the separate blinding key until every QA record is frozen. Qualitative comparison only; no training use is authorized.</p>
  </header>
  {''.join(sections)}
</body>
</html>
"""
    output.write_text(page)
    return output


def _validate_completed_qa(
    public_pilot: Mapping[str, Any],
    work_order: Mapping[str, Any],
    qa: Mapping[str, Any],
    *,
    expected_scene_id: str,
    expected_blind_id: str,
) -> None:
    _require("family" not in qa and "attempt_id" not in qa, "QA record is unblinded")
    _require(
        qa.get("comparison_id") == work_order["comparison_id"]
        and qa.get("run_id") == work_order["run_id"],
        "QA record comparison identity changed",
    )
    _require(qa.get("scene_id") == expected_scene_id, "QA scene ID changed")
    _require(
        qa.get("blinded_display_id") == expected_blind_id,
        "QA blinded display ID changed",
    )
    rater_id = qa.get("rater_id")
    _require(isinstance(rater_id, str) and bool(rater_id.strip()), "QA rater ID is missing")
    rated_at = qa.get("rated_at")
    _require(isinstance(rated_at, str), "QA rated_at is missing")
    try:
        parsed_at = datetime.fromisoformat(rated_at.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("QA rated_at is not an ISO-8601 timestamp") from None
    _require(parsed_at.tzinfo is not None, "QA rated_at must include a timezone")

    expected_items = public_pilot["human_qa"]["items"]
    items = qa.get("items")
    _require(isinstance(items, list), "QA items must be a list")
    _require(
        [item.get("id") for item in items]
        == [item["id"] for item in expected_items],
        "QA item order or identity changed",
    )
    allowed_responses = set(public_pilot["human_qa"]["response_values"])
    allowed_confidence = set(public_pilot["human_qa"]["confidence_values"])
    for item, expected in zip(items, expected_items, strict=True):
        _require(item.get("question") == expected["question"], "QA question changed")
        _require(
            item.get("response") in allowed_responses,
            f"QA response for {item['id']} is incomplete or invalid",
        )
        _require(
            item.get("confidence") in allowed_confidence,
            f"QA confidence for {item['id']} is incomplete or invalid",
        )
        _require(
            item.get("note") is None or isinstance(item.get("note"), str),
            f"QA note for {item['id']} must be text or null",
        )


def inspect_blinded_review_status(
    quality_config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    work_order: Mapping[str, Any],
    *,
    run_root: str | Path,
) -> dict[str, Any]:
    """Report QA completeness without loading or exposing the blinding key."""

    validate_work_order(quality_config, public_pilot, work_order)
    root = Path(run_root).resolve()
    records: list[dict[str, Any]] = []
    rater_ids: set[str] = set()
    for pair in work_order["pairs"]:
        for card in pair["presentation"]:
            blind_id = card["blinded_display_id"]
            qa_path = root / "qa" / f"{blind_id}.json"
            state = "missing"
            issue = "QA record is missing"
            if qa_path.is_file():
                qa = json.loads(qa_path.read_text())
                try:
                    _validate_completed_qa(
                        public_pilot,
                        work_order,
                        qa,
                        expected_scene_id=pair["scene_id"],
                        expected_blind_id=blind_id,
                    )
                except ValueError as error:
                    state = "pending"
                    issue = str(error)
                else:
                    state = "complete"
                    issue = None
                    rater_ids.add(str(qa["rater_id"]).strip())
            records.append(
                {
                    "blinded_display_id": blind_id,
                    "scene_id": pair["scene_id"],
                    "state": state,
                    "issue": issue,
                }
            )
    counts = {
        state: sum(record["state"] == state for record in records)
        for state in ("complete", "pending", "missing")
    }
    expected_raters = int(quality_config["review"]["required_unique_raters"])
    ready = (
        counts["complete"] == len(records)
        and len(rater_ids) == expected_raters
    )
    return {
        "schema_version": 1,
        "comparison_id": work_order["comparison_id"],
        "run_id": work_order["run_id"],
        "status": "READY_TO_UNBLIND" if ready else "BLINDED_QA_INCOMPLETE",
        "record_counts": counts,
        "unique_completed_rater_count": len(rater_ids),
        "required_unique_rater_count": expected_raters,
        "records": records,
    }


def _verify_candidate_execution(
    quality_config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    work_order: Mapping[str, Any],
    run_root: Path,
    *,
    ffprobe_executable: str,
) -> dict[str, Any]:
    run_status = json.loads((run_root / "run_status.json").read_text())
    expected_ids = [attempt["attempt_id"] for attempt in work_order["attempts"]]
    _require(run_status.get("status") == "complete", "candidate run is not complete")
    _require(
        run_status.get("completed_attempt_ids") == expected_ids,
        "candidate completed-attempt order changed",
    )
    _require(
        run_status.get("failed_attempt_ids") == [],
        "candidate run contains generated-attempt failures",
    )
    records = []
    for attempt in work_order["attempts"]:
        paths = {key: run_root / value for key, value in attempt["paths"].items()}
        _require(paths["submission"].is_file(), "candidate submission record is missing")
        _require(paths["record"].is_file(), "candidate attempt record is missing")
        _require(paths["final_video"].is_file(), "candidate final media is missing")
        record = json.loads(paths["record"].read_text())
        _require(record.get("status") == "media_valid", "candidate media is invalid")
        _require(
            record["attempt"]["request_sha256"] == attempt["request_sha256"],
            "candidate request hash changed",
        )
        _require(
            record["final_media"]["sha256"] == file_sha256(str(paths["final_video"])),
            "candidate final hash changed",
        )
        summary = media_summary(paths["final_video"], ffprobe_executable)
        _require(
            validate_final_media(public_pilot, summary) == [],
            "candidate final media no longer conforms",
        )
        records.append(
            {
                "attempt_id": attempt["attempt_id"],
                "scene_id": attempt["scene_id"],
                "status": record["status"],
                "request_id": record["provider"]["request_id"],
                "request_sha256": attempt["request_sha256"],
                "returned_seed": record["provider"]["output"]["returned_seed"],
                "started_at": record["started_at"],
                "finished_at": record["finished_at"],
                "final_sha256": record["final_media"]["sha256"],
                "nursery_adapter_commit": record["provider"][
                    "nursery_adapter_commit"
                ],
            }
        )
    submissions = len(list(run_root.glob("attempts/*/submission.json")))
    return {
        "planned_attempt_count": len(expected_ids),
        "provider_submission_count": submissions,
        "media_valid_count": len(records),
        "generated_attempt_failure_count": submissions - len(records),
        "generated_attempt_failure_rate": (
            (submissions - len(records)) / submissions if submissions else None
        ),
        "estimated_successful_output_charge_usd": float(
            sum(
                Decimal(str(attempt["planned_charge_usd"]))
                for attempt in work_order["attempts"]
            )
        ),
        "actual_provider_invoice_usd": None,
        "records": records,
    }


def _family_review_metrics(
    clips: list[dict[str, Any]],
    quality_config: Mapping[str, Any],
) -> dict[str, Any]:
    visual_ids = set(quality_config["review"]["visual_item_ids"])
    critical_ids = set(quality_config["review"]["critical_visual_item_ids"])
    item_counts = {
        item_id: {"pass": 0, "fail": 0, "cannot_judge": 0}
        for item_id in (
            quality_config["review"]["visual_item_ids"]
            + quality_config["review"]["audio_control_item_ids"]
        )
    }
    for clip in clips:
        for item_id, response in clip["responses"].items():
            item_counts[item_id][response] += 1
    visual = [
        response
        for clip in clips
        for item_id, response in clip["responses"].items()
        if item_id in visual_ids
    ]
    critical_failures = sum(
        response == "fail"
        for clip in clips
        for item_id, response in clip["responses"].items()
        if item_id in critical_ids
    )
    counts = {
        response: sum(value == response for value in visual)
        for response in ("pass", "fail", "cannot_judge")
    }
    judgeable = counts["pass"] + counts["fail"]
    return {
        "clip_count": len(clips),
        "visual_counts": counts,
        "visual_judgeable_count": judgeable,
        "visual_pass_rate": counts["pass"] / judgeable if judgeable else None,
        "critical_visual_failure_count": critical_failures,
        "safety_failure_count": item_counts["safety"]["fail"],
        "item_counts": item_counts,
    }


def _recommendation_markdown(summary: Mapping[str, Any]) -> str:
    family = summary["qualitative"]["families"]
    scenes = summary["qualitative"]["scene_comparisons"]
    lines = [
        "# LTX-2.3 versus Seedance 2.0 public quality-ceiling result",
        "",
        f"**Decision:** `{summary['decision']['label']}`",
        "",
        "This is a single-rater, four-scene qualitative screen using only the "
        "frozen public prompts. It is not a formal model-selection result and "
        "does not authorize Seedance output as learner training data.",
        "",
        "## Technical and cost result",
        "",
        "| Family | Planned | Media valid | Generated failures |",
        "|---|---:|---:|---:|",
        "| LTX-2.3 | 4 | 4 | 0 |",
        (
            "| Seedance 2.0 | 4 | "
            f"{summary['technical']['seedance']['media_valid_count']} | "
            f"{summary['technical']['seedance']['generated_attempt_failure_count']} |"
        ),
        "",
        (
            "Estimated Seedance successful-output charge: "
            f"${summary['cost']['estimated_successful_output_charge_usd']:.3f}. "
            "The actual provider invoice was not available to the runner."
        ),
        "",
        "## Blinded qualitative result",
        "",
        "| Family | Visual pass | Visual fail | Cannot judge | Pass rate | Critical failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family_id, label in (("ltx", "LTX-2.3"), ("seedance", "Seedance 2.0")):
        metrics = family[family_id]
        rate = (
            f"{metrics['visual_pass_rate']:.1%}"
            if metrics["visual_pass_rate"] is not None
            else "n/a"
        )
        lines.append(
            f"| {label} | {metrics['visual_counts']['pass']} | "
            f"{metrics['visual_counts']['fail']} | "
            f"{metrics['visual_counts']['cannot_judge']} | {rate} | "
            f"{metrics['critical_visual_failure_count']} |"
        )
    lines.extend(
        [
            "",
            "| Scene | Result | LTX passes | Seedance passes |",
            "|---|---|---:|---:|",
        ]
    )
    for scene in scenes:
        lines.append(
            f"| {scene['scene_id']} | {scene['winner']} | "
            f"{scene['ltx_visual_pass_count']} | "
            f"{scene['seedance_visual_pass_count']} |"
        )
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            summary["decision"]["explanation"],
            "",
            "Even a material Seedance win means only: pursue written provider "
            "and institutional clearance. ChildLens/BabyView-derived prompts "
            "or inputs remain prohibited, and no learner may consume Seedance "
            "outputs until the separate training-use gate is cleared.",
            "",
            "## Provenance",
            "",
            f"- Quality protocol SHA-256: `{summary['provenance']['quality_protocol_sha256']}`",
            f"- Work-order SHA-256: `{summary['provenance']['work_order_sha256']}`",
            f"- Blinded QA bundle SHA-256: `{summary['provenance']['qa_bundle_sha256']}`",
            f"- Adapter commit: `{summary['provenance']['nursery_adapter_commit']}`",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_blinded_review(
    quality_config: Mapping[str, Any],
    public_pilot: Mapping[str, Any],
    work_order: Mapping[str, Any],
    *,
    repository_root: str | Path,
    run_root: str | Path,
    ffprobe_executable: str = "ffprobe",
) -> dict[str, Any]:
    """Freeze complete blinded QA, then unblind and compute the frozen screen."""

    validate_work_order(quality_config, public_pilot, work_order)
    root = Path(run_root).resolve()
    review_status = inspect_blinded_review_status(
        quality_config,
        public_pilot,
        work_order,
        run_root=root,
    )
    _require(
        review_status["status"] == "READY_TO_UNBLIND",
        "all eight blinded QA records must be complete before unblinding",
    )

    qa_records: dict[str, dict[str, Any]] = {}
    qa_hashes: dict[str, str] = {}
    rater_ids: set[str] = set()
    for pair in work_order["pairs"]:
        for card in pair["presentation"]:
            blind_id = card["blinded_display_id"]
            qa_path = root / "qa" / f"{blind_id}.json"
            qa = json.loads(qa_path.read_text())
            qa_records[blind_id] = qa
            qa_hashes[blind_id] = file_sha256(str(qa_path))
            rater_ids.add(str(qa["rater_id"]).strip())
    qa_bundle_sha256 = canonical_json_sha256(qa_hashes)
    freeze_path = root / "review" / "qa_freeze.json"
    freeze = {
        "schema_version": 1,
        "comparison_id": work_order["comparison_id"],
        "run_id": work_order["run_id"],
        "frozen_at": utc_now(),
        "work_order_sha256": work_order["work_order_sha256"],
        "blinding_key_sha256": work_order["blinding_key_sha256"],
        "qa_file_sha256": qa_hashes,
        "qa_bundle_sha256": qa_bundle_sha256,
        "rater_id_set_sha256": canonical_json_sha256(sorted(rater_ids)),
    }
    if freeze_path.exists():
        existing_freeze = json.loads(freeze_path.read_text())
        _require(
            existing_freeze["qa_bundle_sha256"] == qa_bundle_sha256,
            "QA files changed after the existing freeze",
        )
        freeze = existing_freeze
    else:
        _write_json(freeze_path, freeze)

    blinding_key = json.loads((root / "blinding_key.json").read_text())
    _require(
        canonical_json_sha256(blinding_key) == work_order["blinding_key_sha256"],
        "blinding key hash mismatch",
    )
    _require(
        set(blinding_key) == set(qa_records),
        "blinding key and QA record sets differ",
    )

    baseline = verify_ltx_baseline(quality_config, public_pilot, repository_root)
    candidate = _verify_candidate_execution(
        quality_config,
        public_pilot,
        work_order,
        root,
        ffprobe_executable=ffprobe_executable,
    )
    clips_by_family: dict[str, list[dict[str, Any]]] = {"ltx": [], "seedance": []}
    for blind_id, qa in qa_records.items():
        mapping = blinding_key[blind_id]
        responses = {item["id"]: item["response"] for item in qa["items"]}
        clips_by_family[mapping["family"]].append(
            {
                "scene_id": mapping["scene_id"],
                "blinded_display_id": blind_id,
                "responses": responses,
            }
        )
    family_metrics = {
        family: _family_review_metrics(clips, quality_config)
        for family, clips in clips_by_family.items()
    }
    scene_comparisons = []
    candidate_scene_wins = 0
    candidate_scene_losses = 0
    incomparable_scenes = 0
    visual_ids = set(quality_config["review"]["visual_item_ids"])
    for scene_id in SCENE_IDS:
        by_family = {
            family: next(clip for clip in clips if clip["scene_id"] == scene_id)
            for family, clips in clips_by_family.items()
        }
        scene_counts = {}
        for family, clip in by_family.items():
            visual = [
                response
                for item_id, response in clip["responses"].items()
                if item_id in visual_ids
            ]
            scene_counts[family] = {
                "pass": sum(response == "pass" for response in visual),
                "judgeable": sum(response != "cannot_judge" for response in visual),
            }
        comparable = (
            scene_counts["ltx"]["judgeable"]
            == scene_counts["seedance"]["judgeable"]
        )
        if not comparable:
            winner = "inconclusive"
            incomparable_scenes += 1
        elif scene_counts["seedance"]["pass"] > scene_counts["ltx"]["pass"]:
            winner = "seedance"
            candidate_scene_wins += 1
        elif scene_counts["seedance"]["pass"] < scene_counts["ltx"]["pass"]:
            winner = "ltx"
            candidate_scene_losses += 1
        else:
            winner = "tie"
        scene_comparisons.append(
            {
                "scene_id": scene_id,
                "winner": winner,
                "judgeable_counts_equal": comparable,
                "ltx_visual_pass_count": scene_counts["ltx"]["pass"],
                "seedance_visual_pass_count": scene_counts["seedance"]["pass"],
            }
        )

    rule = quality_config["review"]["material_visual_win_rule"]
    visual_pass_advantage = (
        family_metrics["seedance"]["visual_counts"]["pass"]
        - family_metrics["ltx"]["visual_counts"]["pass"]
    )
    criteria = {
        "candidate_media_valid_count": (
            candidate["media_valid_count"]
            == rule["candidate_media_valid_count_required"]
        ),
        "minimum_total_visual_pass_advantage": (
            visual_pass_advantage >= rule["minimum_total_visual_pass_advantage"]
        ),
        "minimum_candidate_scene_wins": (
            candidate_scene_wins >= rule["minimum_candidate_scene_wins"]
        ),
        "maximum_candidate_scene_losses": (
            candidate_scene_losses <= rule["maximum_candidate_scene_losses"]
        ),
        "equal_judgeable_visual_items_per_scene": incomparable_scenes == 0,
        "critical_failures_not_increased": (
            family_metrics["seedance"]["critical_visual_failure_count"]
            <= family_metrics["ltx"]["critical_visual_failure_count"]
        ),
        "maximum_candidate_safety_failures": (
            family_metrics["seedance"]["safety_failure_count"]
            <= rule["maximum_candidate_safety_failures"]
        ),
    }
    labels = quality_config["review"]["decision_labels"]
    if incomparable_scenes:
        decision_label = labels["inconclusive"]
        explanation = (
            "At least one scene had unequal judgeable-item denominators, so the "
            "prospective visual-win rule cannot be applied."
        )
    elif all(criteria.values()):
        decision_label = labels["material_win"]
        explanation = (
            "Seedance met every prospectively frozen material-visual-win "
            "criterion. This supports pursuing written terms and governance "
            "clearance only; it does not authorize training use."
        )
    else:
        decision_label = labels["no_material_win"]
        explanation = (
            "Seedance did not meet every prospectively frozen material-visual-win "
            "criterion, so LTX remains the reproducible quality baseline."
        )

    baseline_records = [
        {
            "attempt_id": baseline[scene_id]["attempt_id"],
            "scene_id": scene_id,
            "final_sha256": baseline[scene_id]["record"]["final_media"]["sha256"],
        }
        for scene_id in SCENE_IDS
    ]
    adapter_commits = {
        record["nursery_adapter_commit"] for record in candidate["records"]
    }
    _require(len(adapter_commits) == 1, "candidate adapter commits differ")
    summary = {
        "schema_version": 1,
        "comparison_id": work_order["comparison_id"],
        "run_id": work_order["run_id"],
        "status": "COMPLETE_QUALITATIVE_SCREEN",
        "finalized_at": utc_now(),
        "technical": {
            "ltx": {
                "planned_attempt_count": 4,
                "media_valid_count": 4,
                "generated_attempt_failure_count": 0,
                "records": baseline_records,
            },
            "seedance": candidate,
        },
        "qualitative": {
            "profile": quality_config["review"]["profile"],
            "families": family_metrics,
            "scene_comparisons": scene_comparisons,
            "candidate_visual_pass_advantage": visual_pass_advantage,
            "candidate_scene_wins": candidate_scene_wins,
            "candidate_scene_losses": candidate_scene_losses,
            "incomparable_scene_count": incomparable_scenes,
        },
        "cost": {
            "currency": "USD",
            "estimated_successful_output_charge_usd": candidate[
                "estimated_successful_output_charge_usd"
            ],
            "actual_provider_invoice_usd": None,
            "billing_note": (
                "Estimate uses the frozen per-second rate; actual provider "
                "invoice data was not available to the runner."
            ),
        },
        "decision": {
            "label": decision_label,
            "criteria": criteria,
            "explanation": explanation,
            "scientific_training_use_authorized": False,
        },
        "provenance": {
            "quality_protocol_sha256": work_order["quality_protocol_sha256"],
            "public_pilot_protocol_sha256": work_order[
                "public_pilot_protocol_sha256"
            ],
            "work_order_sha256": work_order["work_order_sha256"],
            "blinding_key_sha256": work_order["blinding_key_sha256"],
            "qa_bundle_sha256": freeze["qa_bundle_sha256"],
            "nursery_adapter_commit": next(iter(adapter_commits)),
        },
    }
    summary_path = root / "review" / "review_summary.json"
    recommendation_path = root / "review" / "recommendation.md"
    _write_json(summary_path, summary)
    recommendation_path.write_text(_recommendation_markdown(summary))
    return {
        "status": summary["status"],
        "decision": decision_label,
        "summary": str(summary_path),
        "recommendation": str(recommendation_path),
        "qa_freeze": str(freeze_path),
    }


def load_default_configs(
    quality_config_path: str | Path,
    public_pilot_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    quality = load_quality_config(quality_config_path)
    public = load_public_pilot_config(public_pilot_path)
    validate_quality_config(quality, public)
    return quality, public
