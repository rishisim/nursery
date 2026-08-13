#!/usr/bin/env python3
"""Dependency-free permanent verification for Pilot P4."""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "pilot_p4.json"
AGGREGATE = ROOT / "pilots" / "juno_sample" / "p4_aggregate.json"
CLIP_L_CONFIG = ROOT / "configs" / "pilot_p4_clip_l_diagnostic.json"
CLIP_L_AGGREGATE = ROOT / "pilots" / "juno_sample" / "p4_clip_l_diagnostic_aggregate.json"


def require(value, message):
    if not value:
        raise AssertionError(message)


def load_p4():
    spec = importlib.util.spec_from_file_location("pilot_p4", ROOT / "scripts" / "pilot_p4.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def expect_failure(call, message):
    try:
        call()
    except Exception:
        return
    raise AssertionError(message)


def archive_fixture(path, member_name="MachineDevBench/Lexical/a.txt", kind="file"):
    with tarfile.open(path, "w") as bundle:
        info = tarfile.TarInfo(member_name)
        if kind == "symlink":
            info.type = tarfile.SYMTYPE; info.linkname = "target"; bundle.addfile(info)
        else:
            payload = b"x"; info.size = len(payload); bundle.addfile(info, io.BytesIO(payload))


def synthetic_tests(config):
    p4 = load_p4()
    require(p4.prediction_lexical(1.0, 0.0) == 0, "lexical positive comparison failed")
    require(p4.prediction_lexical(1.0, 1.0) == 1, "lexical tie must be incorrect")
    require(p4.prediction_grammatical([[2, 0], [0, 2]]) == 0, "grammar matched sum failed")
    require(p4.prediction_grammatical([[1, 1], [1, 1]]) == 1, "grammar tie must be incorrect")
    require(p4.merge_bin("[1,2)") == p4.merge_bin("[2,4)") == "[1,4)", "bin merge failed")
    require(p4.merge_bin("[4,8)") == "[4,8)", "unrelated bin changed")
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        good = root / "good.tar"; archive_fixture(good)
        fixture = json.loads(json.dumps(config)); fixture["archive"]["size_bytes"] = good.stat().st_size
        fixture["archive"]["sha256"] = p4.sha256(good); fixture["archive"]["expected_roots"] = ["Lexical"]
        inventory = p4.safe_extract(good, root / "out", fixture)
        require(len(inventory) == 1 and len(inventory[0]["sha256"]) == 64, "checksum inventory failed")
        traversal = root / "traversal.tar"; archive_fixture(traversal, "../escape")
        bad = json.loads(json.dumps(fixture)); bad["archive"]["size_bytes"] = traversal.stat().st_size
        bad["archive"]["sha256"] = p4.sha256(traversal)
        expect_failure(lambda: p4.safe_extract(traversal, root / "bad1", bad), "traversal accepted")
        link = root / "link.tar"; archive_fixture(link, kind="symlink")
        bad["archive"]["size_bytes"] = link.stat().st_size; bad["archive"]["sha256"] = p4.sha256(link)
        expect_failure(lambda: p4.safe_extract(link, root / "bad2", bad), "symlink accepted")
        wrong = json.loads(json.dumps(fixture)); wrong["archive"]["sha256"] = "0" * 64
        expect_failure(lambda: p4.safe_extract(good, root / "bad3", wrong), "checksum mismatch accepted")


def main():
    config_text = CONFIG.read_text(); config = json.loads(config_text)
    synthetic_tests(config)
    inventory = config["inventory"]
    require(inventory["styles"] == ["realistic", "cartoon"], "style inventory changed")
    require(len(inventory["lexical_tasks"]) == 2 and len(inventory["grammatical_tasks"]) == 8,
            "fixed ten-task inventory changed")
    require(config["execution"]["authorized_completed_score_runs"] == 1 and
            config["execution"]["retry_before_score_only"], "single execution policy changed")
    require(config["execution"]["offline_after_assets_pinned"], "offline policy changed")
    require(config["separation"]["cannot_initialize_training"] and
            config["separation"]["export_to_learned_initialization_prohibited"], "weight boundary changed")
    forbidden = [config["storage"]["scratch_namespace"], config["model"]["model_name"],
                 config["model"]["artifact_sha256"], "ViT-B-16.pt"]
    for name in ("pilot.json", "pilot_p2.json", "pilot_p3.json"):
        text = (ROOT / "configs" / name).read_text()
        require(not any(token in text for token in forbidden), f"P4 reference leaked into {name}")
    if not AGGREGATE.exists():
        print("Pilot P4 pre-execution verification passed")
        return
    text = AGGREGATE.read_text(); aggregate = json.loads(text)
    require(aggregate["status"] in {"p4_complete", "p4_incomplete"}, "invalid P4 status")
    require(all(aggregate[key] for key in ("p0_status_preserved", "p1_status_preserved",
            "p2_status_preserved", "p3_status_preserved")), "prior status changed")
    require(aggregate["p5_started"] is False and aggregate["scientific_status_effect"] == "none",
            "scientific/later status changed")
    require(aggregate["verification"]["single_completed_execution"], "single score marker absent")
    require(not any(token in text.lower() for token in ("/work/", "/scratch/", "image_", "caption_",
            "raw_predictions", "participant", "session")), "aggregate is not privacy-safe")
    diagnostic_config = json.loads(CLIP_L_CONFIG.read_text())
    diagnostic_text = CLIP_L_AGGREGATE.read_text(); diagnostic = json.loads(diagnostic_text)
    require(diagnostic_config["model"]["model_name"] == "ViT-L-14" and
            diagnostic_config["model"]["pretrained"] == "openai", "CLIP-L identity changed")
    require(diagnostic_config["execution"]["authorized_completed_score_runs"] == 1,
            "CLIP-L diagnostic execution count changed")
    require(len(diagnostic_config["published_targets_percent"]) == 13,
            "published per-task/subgroup target vector incomplete")
    require(diagnostic["status"] == "diagnostic_complete" and
            diagnostic["verification"]["single_completed_execution"], "CLIP-L diagnostic incomplete")
    require(diagnostic["comparison"]["closer_than_recorded_clip_b"] and
            diagnostic["comparison"]["within_original_p4_overall_window"], "CLIP-L comparison changed")
    require(diagnostic["verification"]["original_clip_b_record_untouched"], "CLIP-B record overwritten")
    require(diagnostic["cannot_initialize_training"] and diagnostic["p5_started"] is False,
            "diagnostic crossed training boundary")
    require(not any(token in diagnostic_text.lower() for token in ("/work/", "/scratch/", "image_",
            "caption_", "raw_predictions", "participant", "session")), "diagnostic aggregate is not privacy-safe")
    print("Pilot P4 verification passed")


if __name__ == "__main__":
    main()
