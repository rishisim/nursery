#!/usr/bin/env python3
"""Dependency-free permanent verification for Pilot P4."""

from __future__ import annotations

import importlib.util
import hashlib
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
SELECTION = ROOT / "pilots" / "juno_sample" / "p4_selection.json"
PILOT_CONFIG = ROOT / "configs" / "pilot.json"
DECISION = ROOT / "pilots" / "juno_sample" / "decision_record.json"
HISTORICAL_HASHES = {
    CONFIG: "c168ea207a7bb822c0a490514e2dee841f59d1c7a6cba15a1031e2bf9d03d2fb",
    AGGREGATE: "56b1d99303cedf1088486071c6ae73002bd3b1d10845dc1a67c81ce492e19d25",
    CLIP_L_CONFIG: "06d8aefca22f25b0dbccf6c91c94f84fbf3fff9a2cd3f99403ac81923d5f3339",
    CLIP_L_AGGREGATE: "90065d5791734230c88308ad2e70a984ba026bd6639d08680c61fc113379fc49",
}


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
                 config["model"]["artifact_sha256"], "ViT-B-16.pt", "ViT-L-14.pt",
                 "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"]
    pilot_without_p4 = json.loads(PILOT_CONFIG.read_text())
    del pilot_without_p4["p4"]
    training_texts = {"pilot.json outside p4 metadata": json.dumps(pilot_without_p4)}
    training_texts.update({name: (ROOT / "configs" / name).read_text()
                           for name in ("pilot_p2.json", "pilot_p3.json")})
    for name, text in training_texts.items():
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
    for path, expected in HISTORICAL_HASHES.items():
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f"immutable executed P4 record changed: {path.name}")
    selection_text = SELECTION.read_text(); selection = json.loads(selection_text)
    selected = selection["selected_calibration"]
    historical = selection["historical_initial_calibration"]
    require(selection["status"] == "p4_complete", "canonical P4 selection is not complete")
    require(selected["model_name"] == "ViT-L-14" and selected["pretrained"] == "openai",
            "selected evaluator is not ViT-L-14/openai")
    require(selected["observed_overall_percent"] == 78.972910 and
            selected["acceptance_minimum_percent"] <= selected["observed_overall_percent"] <=
            selected["acceptance_maximum_percent"] and selected["acceptance_passed"],
            "selected result does not pass original acceptance window")
    require(selected["single_completed_execution"], "selected CLIP-L execution is not singular/complete")
    require(historical["model_name"] == "ViT-B-16-quickgelu" and
            historical["observed_overall_percent"] == 80.081716 and
            historical["acceptance_passed"] is False and historical["immutable_historical_provenance"],
            "failed historical CLIP-B record changed")
    require(selection["cannot_initialize_training"] and
            selection["full_reproduction_recalibration_required"], "selection crossed calibration boundary")
    require(selection["p0_p1_p2_p3_status_preserved"] and selection["next_stage"] == "P5" and
            selection["p5_started"] is False, "phase transition state changed")
    require(not any(token in selection_text.lower() for token in ("/work/", "/scratch/", "image_",
            "caption_", "raw_predictions", "participant", "session")), "selection is not privacy-safe")
    pilot = json.loads(PILOT_CONFIG.read_text()); decision = json.loads(DECISION.read_text())
    require(pilot["pilot_stage"] in {"P4", "P5", "P6", "P7", "P8", "P9"} and pilot["status"] in {"p4_complete", "p5_complete", "p6_complete", "p7_complete", "p8_complete", "p9_complete"} and
            pilot["next_stage"] in {"P5", "P6", "P7", "P8", "P9", "mandatory_full_reproduction_reset_and_readiness_gates"} and
            pilot["p4"]["status"] == "p4_complete" and
            pilot["p4"]["canonical_selection_record"].endswith("p4_selection.json") and
            pilot["p4"]["selected_model"] == "ViT-L-14/openai" and
            pilot["p4"]["next_stage"] == "P5" and isinstance(pilot["p4"]["next_stage_started"], bool),
            "canonical pilot P4/P5 state is inconsistent")
    require(decision["status"] in {"p4_complete", "p5_complete", "p6_complete", "p7_complete", "p8_complete", "p9_complete"} and
            decision["p4_completion"]["canonical_selection_record"] == "p4_selection.json" and
            decision["p4_completion"]["acceptance_passed"] and
            decision["full_reproduction_recalibration_required"], "decision record is inconsistent")
    print("Pilot P4 verification passed")


if __name__ == "__main__":
    main()
