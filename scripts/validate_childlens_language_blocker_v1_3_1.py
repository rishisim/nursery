#!/usr/bin/env python3
"""Fail-closed repository validator for the ChildLens v1.3.1 disposition."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HISTORY = {
    "v1": (23, "35ba9acbba0fc11fc3419c88ea57d0721e08486c6d37595812ce6c77ce994abf"),
    "v1_1": (40, "3acd969804d71979ba8071e2446e8ee1ceb3194fc90dcfda802a99e296b7bf48"),
    "v1_2": (53, "f3bbfd9051e33fc33e44b9cab3c7546b77b1acba4904cc73fb116a3cbe7da01f"),
    "v1_3": (43, "af293859abe4fbd835fcbfc1281ec9514e92f41cf24ad571d6592e6b58783292"),
}
REQUIRED = (
    "docs/childlens_feasibility_v1_3_1/author_language_blocker_disposition.md",
    "docs/childlens_feasibility_v1_3_1/pivot_decision_memo.md",
    "docs/childlens_feasibility_v1_3_1/model_upgrade_appendix_v1_3_1.md",
    "output/childlens_feasibility_v1_3_1/author_attempt_invalidation_receipt.json",
    "output/childlens_feasibility_v1_3_1/language_diagnostic_receipt.json",
    "output/childlens_feasibility_v1_3_1/decision_record.json",
    "output/childlens_feasibility_v1_3_1/immutability_receipt.json",
    "output/childlens_feasibility_v1_3_1/public_model_shortlist_primary_source_evidence.json",
    "output/childlens_feasibility_v1_3_1/gemma4_public_synthetic_runtime_bakeoff.json",
    "output/childlens_feasibility_v1_3_1/qwen3_asr_public_synthetic_runtime_receipt.json",
)
PROHIBITED_SUFFIXES = {".mp4", ".mov", ".mkv", ".wav", ".aiff", ".mp3", ".jpg", ".jpeg", ".png", ".db", ".sqlite", ".csv", ".tsv"}
ABSOLUTE_PATH = re.compile(r"(?:/" + r"Users/|file" + r"://)")
HOSTED_IMPORT = re.compile(r"(?m)^\s*(?:from|import)\s+(?:openai|anthropic|google\.generativeai)\b")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_digest(paths: list[Path]) -> tuple[int, str]:
    lines = sorted(f"{_sha(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in paths)
    return len(paths), hashlib.sha256("".join(lines).encode()).hexdigest()


def _history(version: str) -> list[Path]:
    files: set[Path] = set()
    for directory in (f"docs/childlens_feasibility_{version}", f"output/childlens_feasibility_{version}"):
        path = ROOT / directory
        if path.is_dir():
            files.update(child for child in path.rglob("*") if child.is_file() and "__pycache__" not in child.parts)
    for directory in ("scripts", "tests"):
        for path in (ROOT / directory).glob(f"*childlens*{version}*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if version == "v1" and any(token in path.name for token in ("v1_1", "v1_2", "v1_3")):
                continue
            if version == "v1_1" and any(token in path.name for token in ("v1_2", "v1_3")):
                continue
            if version == "v1_2" and "v1_3" in path.name:
                continue
            if version == "v1_3" and "v1_3_1" in path.name:
                continue
            files.add(path)
    if version == "v1":
        files.add(ROOT / "scripts/validate_childlens_feasibility_v1.py")
        files.add(ROOT / "tests/test_childlens_feasibility_v1.py")
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def validate() -> dict[str, object]:
    issues: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            issues.append(f"MISSING:{relative}")
    history: dict[str, object] = {}
    for version, expected in EXPECTED_HISTORY.items():
        observed = _set_digest(_history(version))
        preserved = observed == expected
        history[version] = {"artifact_count": observed[0], "artifact_set_sha256": observed[1], "preserved": preserved}
        if not preserved:
            issues.append(f"HISTORY_MUTATED:{version}")
    disposition_root = ROOT / "docs/childlens_feasibility_v1_3_1"
    output_root = ROOT / "output/childlens_feasibility_v1_3_1"
    files = [path for base in (disposition_root, output_root) for path in base.rglob("*") if path.is_file()]
    files += [path for base in (ROOT / "scripts", ROOT / "tests") for path in base.glob("*v1_3_1*") if path.is_file()]
    for path in files:
        if path.suffix.lower() in PROHIBITED_SUFFIXES:
            issues.append(f"RESTRICTED_PAYLOAD_TYPE:{path.relative_to(ROOT)}")
            continue
        if path.suffix.lower() not in {".py", ".json", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if ABSOLUTE_PATH.search(text):
            issues.append(f"ABSOLUTE_PATH:{path.relative_to(ROOT)}")
        if path.suffix == ".py" and HOSTED_IMPORT.search(text):
            issues.append(f"HOSTED_CONTENT_CODE_PATH:{path.relative_to(ROOT)}")
        if path.suffix == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError:
                issues.append(f"INVALID_JSON:{path.relative_to(ROOT)}")
    if all((ROOT / relative).is_file() for relative in REQUIRED):
        decision = json.loads((output_root / "decision_record.json").read_text())
        invalidation = json.loads((output_root / "author_attempt_invalidation_receipt.json").read_text())
        language = json.loads((output_root / "language_diagnostic_receipt.json").read_text())
        if decision.get("terminal_state") != "CHILDLENS_LEXICAL_FEASIBILITY_STOP_CURRENT_RESOURCES":
            issues.append("WRONG_TERMINAL_STATE")
        required_false = (
            "learner_or_causal_outcome_run", "aea_empirical_ancestry", "babyview_empirical_ancestry",
            "babyview_or_childlens_pooling", "pseudo_labels_treated_as_human_ground_truth",
            "hosted_model_received_restricted_content", "restricted_childlens_upgrade_rerun",
        )
        if any(decision.get(key) is not False for key in required_false):
            issues.append("SCIENTIFIC_BOUNDARY")
        if invalidation.get("predictions_remained_blinded") is not True or invalidation.get("partial_labels_allowed_to_contribute_to_any_gate") is not False:
            issues.append("INVALIDATION_BOUNDARY")
        if language.get("status") != "MODEL_DIAGNOSTIC_NOT_HUMAN_VALIDATED" or language.get("per_item_languages_exported") is not False:
            issues.append("LANGUAGE_DIAGNOSTIC_BOUNDARY")
    return {
        "schema_version": "childlens-language-blocker-validator-v1.3.1",
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": sorted(issues),
        "history": history,
    }


if __name__ == "__main__":
    result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
