from __future__ import annotations

import pytest
import torch

from babyworld_lite.grounding.pilot_data import (
    ArmDataset,
    BabyWorldEpisodeAdapter,
    GroundingCorpus,
    MOTOR_INPUT_KEYS,
    WordTokenizer,
    deranged_index_map,
    held_out_composition_indices,
)
from babyworld_lite.grounding.pilot_experiment import PilotConfig, evaluate_without_motor
from babyworld_lite.grounding.pilot_model import GroundingModel
from babyworld_lite.sim import simulate_episode


def _records(n: int = 256) -> list[dict]:
    return [simulate_episode(i, 1000 + i * 1009).to_jsonable() for i in range(n)]


def test_motor_derangement_has_no_self_matches() -> None:
    mapping = deranged_index_map(list(range(31)), seed=9)
    assert set(mapping) == set(range(31))
    assert set(mapping.values()) == set(range(31))
    assert all(target != donor for target, donor in mapping.items())


def test_adapter_uses_rgb_and_only_low_level_motor() -> None:
    record = _records(1)[0]
    adapter = BabyWorldEpisodeAdapter()
    video = adapter.video(record, frame_count=3, image_size=32)
    motor = adapter.motor(record, frame_count=3)
    assert video.shape == (3, 3, 32, 32)
    assert video.dtype == torch.uint8
    assert motor.shape == (3, len(MOTOR_INPUT_KEYS))
    assert motor.dtype == torch.float32
    # The raw categorical action is not numerically encoded into the motor tensor.
    assert MOTOR_INPUT_KEYS == ("x", "y", "vx", "vy")


def test_composition_split_and_arm_manipulations() -> None:
    records = _records()
    adapter = BabyWorldEpisodeAdapter()
    train, test = held_out_composition_indices(
        records, adapter, (("cup", "push"),)
    )
    assert train and test
    corpus = GroundingCorpus(records, adapter, frame_count=4, image_size=24)
    tokenizer = WordTokenizer.fit([corpus.text(index) for index in train])
    sync = ArmDataset(corpus, train, tokenizer, "synchronized", 16, 44, time_shift=1)
    shuffled = ArmDataset(corpus, train, tokenizer, "shuffled", 16, 44, time_shift=1)
    null = ArmDataset(corpus, train, tokenizer, "null", 16, 44, time_shift=1)
    shifted = ArmDataset(corpus, train, tokenizer, "time_shifted", 16, 44, time_shift=1)
    index = train[0]
    assert torch.equal(sync._motor(index), corpus.motor(index))
    assert torch.count_nonzero(null._motor(index)) == 0
    assert shuffled.donors[index] != index
    assert torch.equal(shuffled._motor(index), corpus.motor(shuffled.donors[index]))
    assert torch.count_nonzero(shifted._motor(index)[0]) == 0


def test_primary_evaluator_never_calls_motor(monkeypatch: pytest.MonkeyPatch) -> None:
    records = _records()
    adapter = BabyWorldEpisodeAdapter()
    train, test = held_out_composition_indices(records, adapter, (("cup", "push"),))
    corpus = GroundingCorpus(records, adapter, frame_count=2, image_size=20)
    tokenizer = WordTokenizer.fit([corpus.text(index) for index in train])
    model = GroundingModel(len(tokenizer), hidden_dim=8, embedding_dim=6)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("motor encoder was called during the primary test")

    monkeypatch.setattr(model, "encode_motor", forbidden)
    monkeypatch.setattr(corpus, "motor", forbidden)
    result = evaluate_without_motor(
        model,
        corpus,
        test[:8],
        tokenizer,
        ("tap", "push", "grasp", "poke"),
        PilotConfig(frame_count=2, image_size=20, hidden_dim=8, embedding_dim=6, batch_size=4),
        torch.device("cpu"),
    )
    assert result["motor_hard_masked"] is True
    assert result["primary_test_modalities"] == ["rendered_rgb_frames", "utterance_text"]
