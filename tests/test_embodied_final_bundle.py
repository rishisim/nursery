import copy
import json
from pathlib import Path

import h5py
import numpy as np

from babyworld_lite.childlens_engine_bakeoff.bundle import sha256_file
from babyworld_lite.childlens_engine_bakeoff.final_bundle import (
    compose_preview,
    describe_truth_bundle,
    gate_from_checks,
    validate_manifest,
)


CONFIG_PATH = Path("configs/embodied_simulation_final_bundle.json")
CONFIG = json.loads(CONFIG_PATH.read_text())


def test_final_contract_freezes_selected_phase5_candidate_and_boundaries():
    selection = CONFIG["source_selection"]
    assert CONFIG["frozen_before_phase_6_bundle_outcomes_at_utc"]
    assert selection["cell_id"] == "sparse__yellow_cup_authored__20260731"
    assert selection["target_persistent_id"] == "yellow_cup_authored"
    assert selection["expected_artifacts"]["episode_trace.npz"] == (
        selection["expected_artifacts"]["episode_trace_replay.npz"]
    )
    assert CONFIG["authoritative_appearance"]["neural_appearance_selected"] is False
    assert CONFIG["authoritative_appearance"]["mpfb_role"] == "diagnostic_only"
    assert CONFIG["acceptance"]["frozen_thresholds_changed"] is False
    assert (
        CONFIG["privacy_and_governance"][
            "restricted_childlens_access_permitted"
        ]
        is False
    )
    for binding_name in (
        "vertical_slice_config",
        "continuous_episode_config",
        "appearance_config",
        "generalization_config",
        "asset_registry",
    ):
        binding = selection["bindings"][binding_name]
        assert sha256_file(Path(binding["path"])) == binding["sha256"]


def test_final_contract_contains_every_required_deliverable():
    copied = set(CONFIG["bundle"]["copy_selected_files"])
    generated = set(CONFIG["bundle"]["generated_files"])
    assert {
        "accepted_episode.mp4",
        "baseline_rgb.mp4",
        "episode_trace.npz",
        "episode_trace_replay.npz",
        "render_streams.h5",
        "speech.wav",
        "speech_alignment.json",
        "transcript.txt",
        "physics_qa.json",
        "render_qa.json",
        "cross_modal_qa.json",
    } <= copied
    assert {
        "truth_schema.json",
        "qa_preview.png",
        "asset_dependency_provenance.json",
        "replay_instructions.md",
        "final_acceptance_qa.json",
        "final_decision_report.json",
        "final_bundle_manifest.json",
    } <= generated
    samples = CONFIG["preview"]["samples"]
    assert len(samples) == 16
    assert [row["time_s"] for row in samples] == sorted(
        row["time_s"] for row in samples
    )


def test_truth_schema_describes_synchronized_npz_and_hdf5(tmp_path):
    config = copy.deepcopy(CONFIG)
    config["truth_contract"]["required_trace_streams"] = [
        "time_s",
        "phase",
        "assist_active",
        "head_pose",
        "camera_pose",
    ]
    config["truth_contract"]["required_render_streams"] = [
        "time_s",
        "truth_index",
        "depth_m",
        "segmentation",
        "target_area_fraction",
        "clutter_area_fraction",
    ]
    trace_path = tmp_path / "episode_trace.npz"
    np.savez(
        trace_path,
        time_s=np.asarray([0.0, 1.0]),
        phase=np.asarray(["look", "touch"]),
        assist_active=np.asarray([False, True]),
        head_pose=np.zeros((2, 7)),
        camera_pose=np.zeros((2, 12)),
    )
    render_path = tmp_path / "render_streams.h5"
    with h5py.File(render_path, "w") as h5:
        h5.attrs["clock"] = "episode_seconds"
        h5.create_dataset("time_s", data=np.asarray([0.0, 1.0]))
        h5.create_dataset("truth_index", data=np.asarray([0, 1]))
        h5.create_dataset("depth_m", data=np.zeros((2, 2, 3)))
        h5.create_dataset("segmentation", data=np.zeros((2, 2, 3, 2)))
        h5.create_dataset("target_area_fraction", data=np.zeros(2))
        h5.create_dataset("clutter_area_fraction", data=np.zeros(2))
    body_names_path = tmp_path / "body_names.json"
    body_names_path.write_text('["head"]\n')

    schema = describe_truth_bundle(
        trace_path, render_path, body_names_path, config
    )
    assert schema["trace_streams"]["camera_pose"]["shape"] == [2, 12]
    assert schema["render_streams"]["segmentation"]["shape"] == [2, 2, 3, 2]
    assert schema["render_attributes"]["clock"] == "episode_seconds"
    assert schema["rgb"]["neural_appearance"] is None


def test_preview_composition_and_gate_are_strict():
    frames = [
        np.full((8, 12, 3), fill_value, dtype=np.uint8)
        for fill_value in (0, 127, 255)
    ]
    samples = [
        {"time_s": float(index), "label": f"sample-{index}"}
        for index in range(3)
    ]
    sheet = compose_preview(
        frames, samples, columns=2, thumbnail_px=(12, 8)
    )
    assert sheet.size == (24, 76)
    assert gate_from_checks({"one": True, "two": True})["gate_decision"] == (
        "PASS"
    )
    failed = gate_from_checks({"one": True, "two": False})
    assert failed["gate_decision"] == "STOP"
    assert failed["failed_checks"] == ["two"]


def test_manifest_validation_detects_hash_changes(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("authoritative\n")
    manifest = {
        "files": [
            {
                "path": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assert validate_manifest(tmp_path, manifest_path.name)["passed"]
    artifact.write_text("changed\n")
    receipt = validate_manifest(tmp_path, manifest_path.name)
    assert receipt["passed"] is False
    assert receipt["failures"] == ["bytes:artifact.txt", "sha256:artifact.txt"]
