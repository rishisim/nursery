#!/usr/bin/env python3
"""Export an aggregate-only language diagnostic from existing local hypotheses."""

from __future__ import annotations

import json
import os
import stat
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import childlens_author_audit_v1_3 as workflow  # noqa: E402


VERSION = "childlens-language-diagnostic-v1.3.1"
OUTPUT = REPO_ROOT / "output/childlens_feasibility_v1_3_1/language_diagnostic_receipt.json"
EXPECTED_ITEMS = 15
EXPECTED_WINDOWS = 912
EXPECTED_SECONDS = 135.25 * 60
DOMINANCE_THRESHOLD = 0.80
CELL_SUPPRESSION_K = 5
LANGUAGE_NAMES = {"de": "German", "de-de": "German", "german": "German"}


class DiagnosticError(RuntimeError):
    pass


def _private_file(path: Path) -> bool:
    metadata = path.lstat()
    return stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode) and metadata.st_uid == os.getuid() and stat.S_IMODE(metadata.st_mode) & 0o077 == 0


def _atomic_public(value: object) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pending = OUTPUT.with_suffix(".json.pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(pending, OUTPUT)


def execute() -> dict[str, object]:
    root = workflow.discover_runtime_root()
    audio_dir = root / "model_assisted_v1_3/pseudo_annotations/audio"
    if not audio_dir.is_dir() or audio_dir.is_symlink():
        raise DiagnosticError("E_AUDIO_STORE")
    files = sorted(audio_dir.glob("*.json"))
    if len(files) != EXPECTED_ITEMS or any(not _private_file(path) for path in files):
        raise DiagnosticError("E_ITEM_COUNT")
    aggregates: dict[str, dict[str, float]] = defaultdict(lambda: {"items": 0.0, "windows": 0.0, "seconds": 0.0})
    total_windows = 0
    total_seconds = 0.0
    for path in files:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise DiagnosticError("E_AUDIO_RECORD") from exc
        if value.get("schema_version") != "childlens-restricted-audio-pseudo-labels-v1.3.0":
            raise DiagnosticError("E_AUDIO_SCHEMA")
        language = str(value.get("language_hypothesis", "")).strip().casefold()
        windows = value.get("official_window_count")
        seconds = value.get("official_window_seconds")
        if not language or not isinstance(windows, int) or isinstance(windows, bool) or windows < 0 or not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds < 0:
            raise DiagnosticError("E_AUDIO_FIELDS")
        aggregates[language]["items"] += 1
        aggregates[language]["windows"] += windows
        aggregates[language]["seconds"] += float(seconds)
        total_windows += windows
        total_seconds += float(seconds)
    # Per-item public receipts are millisecond-derived floating-point sums; a
    # sub-100 ms aggregate discrepancy is numerical, not a sampling change.
    if total_windows != EXPECTED_WINDOWS or abs(total_seconds - EXPECTED_SECONDS) > 0.10:
        raise DiagnosticError("E_FROZEN_TOTALS")
    dominant_code, dominant = max(aggregates.items(), key=lambda pair: (pair[1]["items"], pair[1]["windows"], pair[0]))
    item_share = dominant["items"] / EXPECTED_ITEMS
    window_share = dominant["windows"] / total_windows
    seconds_share = dominant["seconds"] / total_seconds
    dominance_pass = min(item_share, window_share, seconds_share) >= DOMINANCE_THRESHOLD and dominant["items"] >= CELL_SUPPRESSION_K
    label = LANGUAGE_NAMES.get(dominant_code, "OTHER_OR_UNRESOLVED") if dominance_pass else "NO_DOMINANT_LANGUAGE"
    receipt: dict[str, object] = {
        "schema_version": VERSION,
        "status": "MODEL_DIAGNOSTIC_NOT_HUMAN_VALIDATED",
        "instrument": "whisper.cpp-large-v3-turbo-fixed-v1.3",
        "frozen_selected_item_count": EXPECTED_ITEMS,
        "frozen_candidate_window_count": EXPECTED_WINDOWS,
        "dominance_rule_frozen_before_read": True,
        "dominance_threshold": DOMINANCE_THRESHOLD,
        "dominance_requires_item_window_and_duration_shares": True,
        "dominance_threshold_passed": dominance_pass,
        "aggregate_dominant_language": label,
        "dominant_item_share_band": "GE_80_PERCENT" if item_share >= 0.80 else "LT_80_PERCENT",
        "dominant_window_share_band": "GE_80_PERCENT" if window_share >= 0.80 else "LT_80_PERCENT",
        "dominant_duration_share_band": "GE_80_PERCENT" if seconds_share >= 0.80 else "LT_80_PERCENT",
        "alternative_languages_suppressed": True,
        "cell_suppression_k": CELL_SUPPRESSION_K,
        "per_item_languages_exported": False,
        "transcript_or_lexical_content_exported": False,
        "human_validation_claimed": False,
        "qualified_human_ground_truth_replaced": False,
    }
    _atomic_public(receipt)
    return receipt


if __name__ == "__main__":
    try:
        print(json.dumps(execute(), sort_keys=True))
    except (DiagnosticError, workflow.WorkflowError) as exc:
        print(json.dumps({"status": "BLOCKED", "code": getattr(exc, "code", str(exc))}), file=sys.stderr)
        raise SystemExit(2)
