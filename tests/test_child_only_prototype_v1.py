from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from babyworld_lite.child_only_v1.adapters import bind_selected_corpus_adapter
from babyworld_lite.child_only_v1.calibration import (
    SelectedCorpusCalibrationAdapter,
    load_measurement_spec,
)
from babyworld_lite.child_only_v1.fixtures import build_construction_fixture
from babyworld_lite.child_only_v1.model import (
    FreshCorpusTokenizer,
    TemporalCLIPConfig,
    TemporalCLIPPlusTrainingModel,
    VisionTextEvalBatch,
)
from babyworld_lite.child_only_v1.parallel import (
    build_parallel_plan,
    canonical_merge,
    validate_plan,
)
from babyworld_lite.child_only_v1.policy import (
    CONSTRUCTION_PROFILE,
    PolicyViolation,
    canonical_digest,
    load_policy,
    validate_fresh_corpus_initializations,
    validate_provenance,
)
from babyworld_lite.child_only_v1.protocol import (
    CONDITION_ORDER,
    load_and_validate_evaluation_spec,
    load_and_validate_protocol,
    validate_matched_condition_bundle,
)
from babyworld_lite.child_only_v1.schema import (
    SchemaViolation,
    validate_bundle,
    validate_visible_episode,
)


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "child_only_prototype_v1"


def _construction_provenance() -> dict:
    return json.loads((DOCS / "construction_provenance_v1.json").read_text(encoding="utf-8"))


def _scientific_provenance(selected: str, sources: list[str]) -> dict:
    document = _construction_provenance()
    document.update(
        {
            "study_instance_id": f"{selected.lower()}-instance-alpha",
            "selected_corpus": selected,
            "restricted_access_available": True,
            "claim_tier": "SCIENTIFIC_CHILD_ONLY",
            "profile_label": None,
        }
    )
    document["artifact_nodes"] = [
        {
            "artifact_id": f"artifact-{index}",
            "digest": hashlib.sha256(f"artifact-{index}".encode()).hexdigest(),
            "role": "CORPUS_MEASUREMENT",
            "source_family": source,
            "study_instance_id": document["study_instance_id"],
            "corpus_instance_id": f"{source.lower()}-release-alpha",
            "parents": [],
            "claim_tier": "SCIENTIFIC_CHILD_ONLY",
        }
        for index, source in enumerate(sources)
    ]
    return document


def _small_model() -> TemporalCLIPPlusTrainingModel:
    config = TemporalCLIPConfig(
        image_size=32,
        patch_size=8,
        max_frames=4,
        max_tokens=12,
        vocabulary_size=32,
        text_width=32,
        text_layers=1,
        text_heads=4,
        vision_width=32,
        spatial_layers=1,
        temporal_layers=1,
        vision_heads=4,
        side_input_dim=6,
        side_width=24,
        embedding_dim=16,
        dropout=0.0,
    )
    return TemporalCLIPPlusTrainingModel(
        config,
        initialization_receipt={
            "model_initialization": "SCRATCH",
            "parent_checkpoint": None,
            "model_artifact_id": "fixture-model",
            "corpus_instance_id": "fixture-instance",
            "construction_receipt_id": "fixture-receipt",
        },
    )


def _matched_bundle() -> dict:
    ids = ["ep0", "ep1", "ep2", "ep3"]
    inventory = {
        episode_id: {
            "split": "train" if index < 2 else "test",
            "side_shape": [4, 6],
            "side_validity_digest": "valid-all",
        }
        for index, episode_id in enumerate(ids)
    }
    base = {episode_id: f"sequence-{episode_id}" for episode_id in ids}
    multisets = {episode_id: f"multiset-{episode_id}" for episode_id in ids}
    identity = {episode_id: episode_id for episode_id in ids}
    zero = {episode_id: 0 for episode_id in ids}
    donors = {"ep0": "ep1", "ep1": "ep0", "ep2": "ep3", "ep3": "ep2"}
    arm_base = {
        "source_episode_by_episode": identity,
        "side_sequence_digest_by_episode": base,
        "side_sample_multiset_digest_by_episode": multisets,
        "time_shift_samples_by_episode": zero,
        "final_status": "CONSTRUCTION_VALIDATED",
    }
    arms = {
        "strong_alignment_ceiling": {
            **copy.deepcopy(arm_base),
            "condition": "strong_alignment_ceiling",
            "language_alignment": "STRONG",
            "side_transform": "SYNCHRONIZED",
        },
        "weak_vl_baseline": {
            **copy.deepcopy(arm_base),
            "condition": "weak_vl_baseline",
            "language_alignment": "WEAK",
            "side_transform": "SHAPE_MATCHED_NULL",
            "source_episode_by_episode": {episode_id: None for episode_id in ids},
            "side_sequence_digest_by_episode": {episode_id: "null-sequence" for episode_id in ids},
            "side_sample_multiset_digest_by_episode": {episode_id: "null-multiset" for episode_id in ids},
        },
        "weak_synchronized_side": {
            **copy.deepcopy(arm_base),
            "condition": "weak_synchronized_side",
            "language_alignment": "WEAK",
            "side_transform": "SYNCHRONIZED",
        },
        "weak_episode_shuffled_side": {
            **copy.deepcopy(arm_base),
            "condition": "weak_episode_shuffled_side",
            "language_alignment": "WEAK",
            "side_transform": "WHOLE_BUNDLE_EPISODE_SHUFFLE",
            "source_episode_by_episode": donors,
            "side_sequence_digest_by_episode": {target: base[donor] for target, donor in donors.items()},
            "side_sample_multiset_digest_by_episode": {target: multisets[donor] for target, donor in donors.items()},
        },
        "weak_time_shifted_side": {
            **copy.deepcopy(arm_base),
            "condition": "weak_time_shifted_side",
            "language_alignment": "WEAK",
            "side_transform": "WITHIN_EPISODE_CIRCULAR_NONZERO_SHIFT",
            "side_sequence_digest_by_episode": {episode_id: f"shifted-{episode_id}" for episode_id in ids},
            "time_shift_samples_by_episode": {episode_id: 1 for episode_id in ids},
        },
    }
    return {
        "bundle_version": "child-only-matched-bundle-v1",
        "profile_label": CONSTRUCTION_PROFILE,
        "bundle_id": "fixture-bundle",
        "corpus_instance_id": "fixture-instance",
        "corpus_seed": 101,
        "model_seed": 11,
        "weak_common": {
            "rgb_text_experience_digest": "rgb-text",
            "architecture_digest": "architecture",
            "initialization_receipt_id": "init",
            "tokenizer_artifact_id": "tokenizer",
            "model_config_digest": "model",
            "optimizer_digest": "optimizer",
            "example_order_digest": "order",
            "update_count": 10,
            "batch_shape_digest": "batch-shape",
            "stopping_rule_digest": "stopping",
            "compute_budget_digest": "compute",
            "side_architecture_present": True,
        },
        "episode_inventory": inventory,
        "base_side_sequence_digest_by_episode": base,
        "base_side_sample_multiset_digest_by_episode": multisets,
        "arms": arms,
    }


def test_construction_provenance_passes_and_adult_marker_fails_closed() -> None:
    validate_provenance(_construction_provenance())
    machine_policy = load_policy(DOCS / "provenance_policy_v1.json")
    assert machine_policy["one_corpus_at_a_time"]["exactly_one_for_scientific_instance"] is True
    contaminated = _construction_provenance()
    contaminated["artifact_nodes"][0]["role"] = "adult_sensor_analogue_prior"
    with pytest.raises(PolicyViolation, match="forbidden adult-data"):
        validate_provenance(contaminated)


def test_one_child_corpus_at_a_time_and_no_pooling() -> None:
    validate_provenance(_scientific_provenance("BABYVIEW", ["BABYVIEW"]))
    validate_provenance(_scientific_provenance("CHILDLENS", ["CHILDLENS"]))
    with pytest.raises(PolicyViolation, match="exactly the selected"):
        validate_provenance(_scientific_provenance("BABYVIEW", ["BABYVIEW", "CHILDLENS"]))
    with pytest.raises(PolicyViolation, match="exactly the selected"):
        validate_provenance(_scientific_provenance("CHILDLENS", ["BABYVIEW"]))


def test_provenance_cycle_and_foreign_study_instance_fail_closed() -> None:
    document = _construction_provenance()
    second = copy.deepcopy(document["artifact_nodes"][0])
    second["artifact_id"] = "construction-fixture-child-v1"
    second["digest"] = "1" * 64
    document["artifact_nodes"][0]["parents"] = [second["artifact_id"]]
    second["parents"] = [document["artifact_nodes"][0]["artifact_id"]]
    document["artifact_nodes"].append(second)
    with pytest.raises(PolicyViolation, match="cycle"):
        validate_provenance(document)

    foreign = _construction_provenance()
    foreign["artifact_nodes"][0]["study_instance_id"] = "foreign-instance"
    with pytest.raises(PolicyViolation, match="different study"):
        validate_provenance(foreign)


def test_fresh_initialization_receipts_reject_shared_lineage() -> None:
    first = {
        "selected_corpus": "BABYVIEW",
        "corpus_instance_id": "babyview-alpha",
        "tokenizer_artifact_id": "tok-a",
        "tokenizer_training_corpus_instance_id": "babyview-alpha",
        "model_artifact_id": "model-a",
        "model_initialization": "SCRATCH",
        "parent_checkpoint": None,
        "construction_receipt_id": "receipt-a",
    }
    second = {
        "selected_corpus": "CHILDLENS",
        "corpus_instance_id": "childlens-alpha",
        "tokenizer_artifact_id": "tok-b",
        "tokenizer_training_corpus_instance_id": "childlens-alpha",
        "model_artifact_id": "model-b",
        "model_initialization": "SCRATCH",
        "parent_checkpoint": None,
        "construction_receipt_id": "receipt-b",
    }
    validate_fresh_corpus_initializations([first, second])
    second["tokenizer_artifact_id"] = "tok-a"
    with pytest.raises(PolicyViolation, match="cannot be shared"):
        validate_fresh_corpus_initializations([first, second])


def test_fixture_schema_oracle_separation_and_determinism(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_construction_fixture(first)
    second_manifest = build_construction_fixture(second)
    first_result = validate_bundle(first)
    second_result = validate_bundle(second)
    assert first_result["valid"] and second_result["valid"]
    assert first_manifest["canonical_digest"] == second_manifest["canonical_digest"]
    visible_bytes = (first / "model_visible" / "episodes.jsonl").read_bytes()
    oracle_bytes = (first / "hidden_oracle" / "events.jsonl").read_bytes()
    assert visible_bytes != oracle_bytes
    assert b"event_type" not in visible_bytes
    assert b"counterfactual_truth" not in visible_bytes
    assert (first / "model_visible").resolve() != (first / "hidden_oracle").resolve()


def test_visible_schema_rejects_unknown_oracle_key_without_opening_oracle(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    build_construction_fixture(root)
    visible_file = root / "model_visible" / "episodes.jsonl"
    record = json.loads(visible_file.read_text(encoding="utf-8").splitlines()[0])
    record["oracle"] = {"sentinel": True}
    with pytest.raises(SchemaViolation, match="keys must be exactly"):
        validate_visible_episode(record, root / "model_visible")


def test_bundle_rejects_mutation_and_unexpected_file(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    build_construction_fixture(root)
    frame = next((root / "model_visible" / "frames").rglob("*.png"))
    frame.write_bytes(frame.read_bytes() + b"mutation")
    with pytest.raises(SchemaViolation, match="digest mismatch"):
        validate_bundle(root)

    other = tmp_path / "extra"
    build_construction_fixture(other)
    (other / "model_visible" / "unexpected.bin").write_bytes(b"x")
    with pytest.raises(SchemaViolation, match="missing or unexpected"):
        validate_bundle(other)


def test_separate_calibration_specs_have_no_values_and_adapter_is_access_gated() -> None:
    babyview = load_measurement_spec(DOCS / "calibration" / "babyview_measurement_spec_v1.json", "BABYVIEW")
    childlens = load_measurement_spec(DOCS / "calibration" / "childlens_measurement_spec_v1.json", "CHILDLENS")
    assert babyview["corpus"] != childlens["corpus"]
    assert babyview["measurement_spec_version"] != childlens["measurement_spec_version"]
    with pytest.raises(PolicyViolation, match="exactly one selected"):
        SelectedCorpusCalibrationAdapter(
            selected_corpus=None,
            corpus_instance_id=None,
            restricted_access_available=False,
            measurement_spec=babyview,
        )


def test_selected_corpus_adapter_cannot_bind_before_access_or_cross_corpus() -> None:
    class FakeAdapter:
        selected_corpus = "BABYVIEW"
        corpus_instance_id = "babyview-alpha"
        adapter_version = "fake-interface-v1"

        def iter_timed_utterance_text(self):
            raise AssertionError("binding must not iterate restricted data")

        def iter_video_episode_descriptors(self):
            raise AssertionError("binding must not iterate restricted data")

    adapter = FakeAdapter()
    with pytest.raises(PolicyViolation, match="exactly one child corpus"):
        bind_selected_corpus_adapter(
            adapter,
            selected_corpus=None,
            corpus_instance_id=None,
            restricted_access_available=False,
        )
    with pytest.raises(PolicyViolation, match="other child corpus"):
        bind_selected_corpus_adapter(
            adapter,
            selected_corpus="CHILDLENS",
            corpus_instance_id="childlens-alpha",
            restricted_access_available=True,
        )
    bound = bind_selected_corpus_adapter(
        adapter,
        selected_corpus="BABYVIEW",
        corpus_instance_id="babyview-alpha",
        restricted_access_available=True,
    )
    assert bound.selected_corpus == "BABYVIEW"


def test_scratch_temporal_clip_plus_training_shapes() -> None:
    model = _small_model().eval()
    outputs = model.training_forward(
        video=torch.zeros((2, 4, 3, 32, 32), dtype=torch.uint8),
        tokens=torch.tensor([[2, 5, 3], [2, 6, 3]], dtype=torch.long),
        attention_mask=torch.ones((2, 3), dtype=torch.bool),
        raw_side=torch.zeros((2, 4, 6)),
        candidate_event_mask=torch.tensor(
            [[[1, 1, 0, 0], [0, 0, 1, 1]], [[1, 0, 1, 0], [0, 1, 0, 1]]],
            dtype=torch.float32,
        ),
        candidate_valid=torch.ones((2, 2), dtype=torch.bool),
    )
    assert outputs["video_embedding"].shape == (2, 16)
    assert outputs["text_embedding"].shape == (2, 16)
    assert outputs["contrastive_logits"].shape == (2, 2)
    assert outputs["training_alignment_logits"].shape == (2, 3)  # two candidates plus null


@pytest.mark.parametrize("forbidden_key", ["raw_side", "detector_outputs", "oracle", "corpus_metadata"])
def test_primary_evaluation_rejects_every_extra_input(forbidden_key: str) -> None:
    inputs = {
        "video": torch.zeros((1, 4, 3, 32, 32), dtype=torch.uint8),
        "tokens": torch.tensor([[2, 5, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.bool),
        forbidden_key: torch.zeros(1),
    }
    with pytest.raises(PolicyViolation, match="accepts only vision/text"):
        VisionTextEvalBatch.from_mapping(inputs)


def test_exported_primary_evaluator_has_no_side_state_and_never_calls_side(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _small_model().eval()
    evaluator = model.export_primary_evaluation_model().eval()

    def fail(*args, **kwargs):
        raise AssertionError("training-only side path was called")

    monkeypatch.setattr(model.side_encoder, "forward", fail)
    monkeypatch.setattr(model.training_event_null_aligner, "forward", fail)
    batch = VisionTextEvalBatch.from_mapping(
        {
            "video": torch.zeros((1, 4, 3, 32, 32), dtype=torch.uint8),
            "tokens": torch.tensor([[2, 5, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.bool),
        }
    )
    result = evaluator(batch)
    assert result["similarity"].shape == (1, 1)
    forbidden = ("side", "aligner", "event", "oracle", "detector", "corpus")
    assert not any(any(token in key.lower() for token in forbidden) for key in evaluator.state_dict())
    assert not hasattr(evaluator, "side_encoder")


def test_tokenizer_is_fixture_labeled_and_corpus_bound() -> None:
    tokenizer = FreshCorpusTokenizer.fit(
        ["see the object", "the object moves"],
        source_family="SYNTHETIC_FIXTURE",
        corpus_instance_id="fixture-instance",
        tokenizer_artifact_id="fixture-tokenizer",
        max_vocabulary_size=32,
        profile_label=CONSTRUCTION_PROFILE,
    )
    tokens, mask = tokenizer.encode(["object moves"], max_tokens=8)
    assert tokens.shape == mask.shape
    assert tokenizer.receipt()["profile_label"] == CONSTRUCTION_PROFILE
    with pytest.raises(PolicyViolation, match="construction label"):
        FreshCorpusTokenizer.fit(
            ["fixture text"],
            source_family="SYNTHETIC_FIXTURE",
            corpus_instance_id="fixture-instance",
            tokenizer_artifact_id="unsafe",
            max_vocabulary_size=32,
        )


def test_causal_and_evaluation_specs_are_complete() -> None:
    protocol = load_and_validate_protocol(DOCS / "causal_protocol_v1.json")
    assert list(protocol["conditions"]) == list(CONDITION_ORDER)
    spec = load_and_validate_evaluation_spec(DOCS / "evaluation_spec_v1.json")
    assert spec["model_call_inputs"] == ["VISION", "TEXT"]


def test_matched_control_bundle_passes_and_controls_fail_closed() -> None:
    bundle = _matched_bundle()
    validate_matched_condition_bundle(bundle)

    self_shuffle = copy.deepcopy(bundle)
    self_shuffle["arms"]["weak_episode_shuffled_side"]["source_episode_by_episode"]["ep0"] = "ep0"
    with pytest.raises(PolicyViolation, match="no-self derangement"):
        validate_matched_condition_bundle(self_shuffle)

    cross_split = copy.deepcopy(bundle)
    cross_split["arms"]["weak_episode_shuffled_side"]["source_episode_by_episode"] = {
        "ep0": "ep2", "ep1": "ep3", "ep2": "ep0", "ep3": "ep1"
    }
    cross_split["arms"]["weak_episode_shuffled_side"]["side_sequence_digest_by_episode"] = {
        "ep0": "sequence-ep2", "ep1": "sequence-ep3", "ep2": "sequence-ep0", "ep3": "sequence-ep1"
    }
    cross_split["arms"]["weak_episode_shuffled_side"]["side_sample_multiset_digest_by_episode"] = {
        "ep0": "multiset-ep2", "ep1": "multiset-ep3", "ep2": "multiset-ep0", "ep3": "multiset-ep1"
    }
    with pytest.raises(PolicyViolation, match="stay in split"):
        validate_matched_condition_bundle(cross_split)

    zero_shift = copy.deepcopy(bundle)
    zero_shift["arms"]["weak_time_shifted_side"]["time_shift_samples_by_episode"]["ep0"] = 0
    with pytest.raises(PolicyViolation, match="nonzero"):
        validate_matched_condition_bundle(zero_shift)

    full_rotation = copy.deepcopy(bundle)
    full_rotation["arms"]["weak_time_shifted_side"]["time_shift_samples_by_episode"]["ep0"] = 4
    with pytest.raises(PolicyViolation, match="modulo"):
        validate_matched_condition_bundle(full_rotation)


def test_parallel_plan_keeps_conditions_coupled_and_mps_serial() -> None:
    plan = build_parallel_plan(
        protocol_digest="p" * 64,
        corpus_instance_id="fixture-instance",
        corpus_seeds=[2, 1],
        model_seeds=[4, 3],
        backend="mps",
        max_trainers=1,
        cpu_workers=2,
        profile_label=CONSTRUCTION_PROFILE,
        outcome_task_authorized=False,
    )
    validate_plan(plan)
    assert all(bundle["conditions_serial_on_one_device"] == list(CONDITION_ORDER) for bundle in plan["bundles"])
    with pytest.raises(PolicyViolation, match="cap concurrent training"):
        build_parallel_plan(
            protocol_digest="p" * 64,
            corpus_instance_id="fixture-instance",
            corpus_seeds=[1],
            model_seeds=[1],
            backend="mps",
            max_trainers=2,
            cpu_workers=2,
            profile_label=CONSTRUCTION_PROFILE,
            outcome_task_authorized=False,
        )


def test_parallel_merge_is_canonical_and_fails_on_incomplete_inventory(tmp_path: Path) -> None:
    plan = build_parallel_plan(
        protocol_digest="q" * 64,
        corpus_instance_id="fixture-instance",
        corpus_seeds=[1],
        model_seeds=[7, 8],
        backend="slurm_cuda",
        max_trainers=2,
        cpu_workers=2,
        profile_label=CONSTRUCTION_PROFILE,
        outcome_task_authorized=False,
    )
    for bundle in plan["bundles"]:
        directory = tmp_path / bundle["bundle_id"]
        directory.mkdir()
        payload = b'{"fixture":"no-outcome"}\n'
        (directory / "result_payload.json").write_bytes(payload)
        manifest = {
            "result_manifest_version": "child-only-result-manifest-v1",
            "bundle_id": bundle["bundle_id"],
            "plan_digest": plan["plan_digest"],
            "protocol_digest": plan["protocol_digest"],
            "corpus_instance_id": bundle["corpus_instance_id"],
            "corpus_seed": bundle["corpus_seed"],
            "model_seed": bundle["model_seed"],
            "condition_order": list(CONDITION_ORDER),
            "condition_status": {condition: "COMPLETE" for condition in CONDITION_ORDER},
            "shared_digest_receipts": {
                "data": "d", "tokenizer": "t", "architecture": "a", "initialization": "i",
                "optimizer": "o", "example_order": "e", "updates": "u", "compute": "c",
            },
            "payload_path": "result_payload.json",
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "final_status": "COMPLETE",
        }
        (directory / "result_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    merged = canonical_merge(plan, tmp_path)
    assert merged["complete"] is True and merged["bundle_count"] == 2
    missing = tmp_path / plan["bundles"][0]["bundle_id"]
    for child in missing.iterdir():
        child.unlink()
    missing.rmdir()
    with pytest.raises(PolicyViolation, match="inventory incomplete"):
        canonical_merge(plan, tmp_path)


def test_new_namespace_has_no_legacy_study_imports() -> None:
    forbidden_fragments = (
        "babyworld_lite.aea",
        "sensor_alignment",
        "corrective_alignment",
        "development_launch",
        "weak_alignment",
        "babyworld_lite.grounding",
    )
    for path in sorted((ROOT / "babyworld_lite" / "child_only_v1").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(fragment in imported for fragment in forbidden_fragments for imported in imports), path


def test_canonical_digest_is_order_stable_and_mutation_sensitive() -> None:
    first = {"b": [2, 1], "a": {"x": True}}
    second = {"a": {"x": True}, "b": [2, 1]}
    assert canonical_digest(first) == canonical_digest(second)
    second["b"][0] = 3
    assert canonical_digest(first) != canonical_digest(second)
