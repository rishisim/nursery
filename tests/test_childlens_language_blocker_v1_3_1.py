from __future__ import annotations

import ast
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


policy = load("childlens_language_blocker_policy_test", "scripts/childlens_language_blocker_policy_v1_3_1.py")
clips = load("childlens_bounded_media_test", "scripts/childlens_bounded_audit_media_v1_3_1.py")
app = load("childlens_corrected_app_test", "scripts/childlens_author_audit_app_v1_3_1.py")
disposition_module = load("childlens_disposition_test", "scripts/disposition_childlens_author_blocker_v1_3_1.py")


def test_legacy_app_served_full_source_with_seek_hints():
    source = (ROOT / "scripts/childlens_author_audit_app_v1_3.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "video"]
    assert len(calls) == 1
    assert {kw.arg for kw in calls[0].keywords} == {"start_time", "end_time"}
    assert "media_path" in ast.unparse(calls[0].args[0])


def test_physically_bounded_media_and_player_api(tmp_path: Path):
    source = tmp_path / "synthetic-source.mp4"
    os.chmod(tmp_path, 0o700)
    subprocess.run(
        [
            "/opt/homebrew/bin/ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=8",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
            "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
        ],
        check=True,
    )
    os.chmod(source, 0o600)
    bounded = clips.materialize_bounded_clips(source, [(2_000, 4_000)], tmp_path / "bounded")
    assert len(bounded) == 1
    assert 1.8 <= bounded[0]["duration_seconds"] <= 2.2
    assert clips.probe_duration_seconds(source) >= 7.8
    assert Path(bounded[0]["path"]) != source

    class FakeUI:
        def __init__(self):
            self.captions = []
            self.videos = []

        def caption(self, value):
            self.captions.append(value)

        def video(self, value):
            self.videos.append(value)

    ui = FakeUI()
    app.render_bounded_windows(ui, bounded, item_number=1, item_count=15, completed_seconds=0)
    assert ui.videos == [str(bounded[0]["path"])]
    assert "physically limited" in " ".join(ui.captions)
    assert "15.0 audit minutes" in " ".join(ui.captions)


def test_language_qualification_gate_and_current_resource_terminal():
    assert not policy.author_is_language_qualified("German", ["English"], "FLUENT")
    assert policy.author_is_language_qualified("German", ["German"], "PROFICIENT")
    assert policy.disposition(german_dominance_diagnostic=True, qualified_author_available=False) == policy.TERMINAL_STATE


@pytest.mark.parametrize(
    "translation_as_gold,model_model_as_human,code",
    [
        (True, False, "E_TRANSLATION_AS_GOLD_PROHIBITED"),
        (False, True, "E_MODEL_MODEL_AS_HUMAN_PROHIBITED"),
    ],
)
def test_circular_substitutes_rejected(translation_as_gold, model_model_as_human, code):
    with pytest.raises(ValueError, match=code):
        policy.validate_evidence_uses(
            translation_as_gold=translation_as_gold,
            model_model_as_human=model_model_as_human,
        )


def test_invalidated_partial_rows_cannot_change_or_be_locked(tmp_path: Path):
    database = tmp_path / "synthetic.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE item_labels(value TEXT);
        CREATE TABLE author_utterances(value TEXT);
        CREATE TABLE author_mentions(value TEXT);
        CREATE TABLE workflow_meta(workflow_state TEXT);
        INSERT INTO workflow_meta VALUES('AUTHOR_BLIND_OPEN');
        """
    )
    connection.executescript(disposition_module.INVALIDATION_TRIGGER_SQL)
    for statement in (
        "INSERT INTO item_labels VALUES('x')",
        "INSERT INTO author_utterances VALUES('x')",
        "INSERT INTO author_mentions VALUES('x')",
        "UPDATE workflow_meta SET workflow_state='AUTHOR_LOCKED'",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="E_ATTEMPT_INVALIDATED"):
            connection.execute(statement)
    assert connection.execute("SELECT workflow_state FROM workflow_meta").fetchone()[0] == "AUTHOR_BLIND_OPEN"
    connection.close()


def test_real_public_receipts_preserve_blinding_and_exclude_partial_gate_use():
    invalidation = json.loads((ROOT / "output/childlens_feasibility_v1_3_1/author_attempt_invalidation_receipt.json").read_text())
    language = json.loads((ROOT / "output/childlens_feasibility_v1_3_1/language_diagnostic_receipt.json").read_text())
    assert invalidation["status"] == "INVALID_FOR_SCIENTIFIC_COMPARISON"
    assert invalidation["predictions_remained_blinded"] is True
    assert invalidation["model_human_comparison_executed"] is False
    assert invalidation["partial_labels_allowed_to_contribute_to_any_gate"] is False
    assert invalidation["translation_as_gold_allowed"] is False
    assert invalidation["model_model_agreement_allowed_as_human_validation"] is False
    assert language["status"] == "MODEL_DIAGNOSTIC_NOT_HUMAN_VALIDATED"
    assert language["aggregate_dominant_language"] == "German"
    assert language["dominance_threshold_passed"] is True
    assert language["per_item_languages_exported"] is False
