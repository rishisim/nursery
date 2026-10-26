"""CPU-only synthetic validation of caregiver transcripts and cue weights."""

import json
from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_frames

from nursery.common import _output_directory, load_config
from nursery.speech.transcripts import clip_bounds, cue_timelines, generate_transcripts, session_scene, write_transcripts

pytestmark = pytest.mark.usefixtures("offline")


def test_transcript_statistics_and_determinism():
    config = load_config()
    frames = synthetic_frames(n=15000, videos=4)
    frames["held_category"] = "mug"
    manifest, _ = generate_transcripts(frames, config, 1701)
    again, _ = generate_transcripts(frames, config, 1701)
    pd.testing.assert_frame_equal(manifest, again)
    different, _ = generate_transcripts(frames, config, 1702)
    assert not manifest.equals(different)
    refs = manifest[manifest.referential]
    measured = {"utterances": len(manifest), "rate_hz": len(manifest) / len(frames),
                "p_ref": float(manifest.referential.mean()), "p_held": float((refs.channel == "held").mean()),
                "p_name": float(refs.named.mean()), "jitter_mean": float(manifest.jitter_draw_frames.mean()),
                "jitter_std": float(manifest.jitter_draw_frames.std())}
    print("statistics:", json.dumps(measured, sort_keys=True))
    assert measured["rate_hz"] == pytest.approx(1 / 3, abs=0.01)
    # Always-held scene: no unseen channel, so held share is p_held after redraws.
    for key in ("p_ref", "p_held", "p_name"):
        assert measured[key] == pytest.approx(config["generator"][key], abs=0.035)
    assert abs(measured["jitter_mean"]) < 0.04
    assert measured["jitter_std"] == pytest.approx(np.sqrt(1 + 1 / 12), abs=0.04)
    assert ((manifest.frame_idx >= 0) & (manifest.frame_idx < 15000)).all()
    altered = load_config()
    altered["cues"]["noise_eps"] = [0, 1]
    changed_cues, _ = generate_transcripts(frames, altered, 1701)
    speech_columns = [c for c in manifest if not c.startswith("weight_")]
    pd.testing.assert_frame_equal(manifest[speech_columns], changed_cues[speech_columns])


def test_mean_weights_and_oracle():
    frames = synthetic_frames(1000, 4)
    # Visibility changes so the oracle must use the shifted frame, not the source.
    frames.loc[frames.frame_idx % 2 == 0, "objects"] = '[]'
    config = load_config()
    config["generator"].update(p_ref=1, p_name=0.5)
    frames["held_category"] = "mug"
    manifest, _ = generate_transcripts(frames, config, 34)
    means = manifest.filter(like="weight_").mean()
    np.testing.assert_allclose(means, 1, atol=1e-12)
    expected_visible = (manifest.frame_idx % 2 != 0).to_numpy()
    np.testing.assert_array_equal(manifest.referent_visible, expected_visible)
    high = manifest.referential & manifest.named & manifest.referent_visible
    oracle = np.where(high, 1.0, 0.2)
    np.testing.assert_allclose(manifest.weight_oracle, oracle / oracle.mean())
    cue_seed = np.random.SeedSequence(34).spawn(2)[1]
    cues, _ = cue_timelines(frames, config["cues"], np.random.default_rng(cue_seed))
    for name in cues["v0"]:
        values = [cues[r.video_id][name][r.frame_idx] for r in manifest.itertuples()]
        weights = np.where(values, 1.0, 0.2)
        np.testing.assert_allclose(manifest[f"weight_{name}"], weights / weights.mean())
    print("weight means:", means.to_dict(), "oracle positives:", int(high.sum()))


def test_shuffled_marginals_shift_and_noise():
    frames = synthetic_frames(1000, 4)
    config = load_config()["cues"]
    config["noise_eps"] = [0, 0.25, 1]
    cues, donors = cue_timelines(frames, config, np.random.default_rng(3))
    contact = np.concatenate([c["contact"] for c in cues.values()])
    shuffled = np.concatenate([c["shuffled"] for c in cues.values()])
    assert contact.sum() == shuffled.sum()
    for video, timeline in cues.items():
        assert donors[video] != video
        np.testing.assert_array_equal(timeline["shuffled"], cues[donors[video]]["contact"])
        np.testing.assert_array_equal(timeline["shifted"], np.roll(timeline["contact"], -10))
        np.testing.assert_array_equal(timeline["noisy_0"], timeline["contact"])
        np.testing.assert_array_equal(timeline["noisy_1"], ~timeline["contact"])
    flip_rate = np.concatenate([c["noisy_0.25"] ^ c["contact"] for c in cues.values()]).mean()
    assert flip_rate == pytest.approx(0.25, abs=0.03)
    disagreement = float((contact != shuffled).mean())
    assert disagreement > 0.3
    print(f"cue marginals: {contact.sum()}/{len(contact)} = {shuffled.sum()}/{len(shuffled)}; "
          f"shuffled disagreement={disagreement:.4f}; noisy_0.25 flip rate={flip_rate:.4f}")


def test_no_held_referents_and_category_independent_frames():
    frames = synthetic_frames(1000, 2)
    frames["held_category"] = pd.Series(pd.NA, index=frames.index, dtype="string")
    config = load_config()
    config["generator"].update(p_ref=1, p_name=1)
    manifest, _ = generate_transcripts(frames, config, 10)
    frequencies = manifest.referent.value_counts(normalize=True)
    assert set(frequencies.index) == {"mug", "bowl", "apple", "banana"}
    assert not manifest.held_referent.any() and not manifest.held_referent.isna().any()
    np.testing.assert_allclose(frequencies, 0.25, atol=0.06)
    swapped = frames.copy()
    swapped["objects"] = swapped.objects.str.replace('"mug"', '"plane"').str.replace('"apple"', '"glass"')
    changed_config = load_config()
    changed_config["generator"] = config["generator"]
    changed_config["vocabulary"].update(plane="plane", glass="glass")
    # Template draws cannot depend on object category; compare a one-object scene.
    for table, category in ((frames, "mug"), (swapped, "plane")):
        table["held_category"] = category
        table["objects"] = json.dumps([{"category": category, "visible_px": 5, "cells": [0]}])
    original, _ = generate_transcripts(frames, config, 10)
    changed, _ = generate_transcripts(swapped, changed_config, 10)
    assert original.text.str.replace("mug", "{noun}").tolist() == changed.text.str.replace("plane", "{noun}").tolist()
    pool = json.loads((Path(__file__).parents[1] / "nursery/speech/frames.json").read_text())
    assert sum(map(len, pool.values())) == 150
    nouns = set(config["vocabulary"].values())
    for kind, templates in pool.items():
        assert len(set(templates)) == len(templates)
        for template in templates:
            assert ("{noun}" in template) == (kind == "named")
            assert not nouns.intersection(template.replace("{noun}", "").strip(".!?").split())


def test_channels_follow_window_and_visibility_and_calibrate():
    cats = ["mug", "bowl", "apple", "banana", "cup", "pan"]
    frames = synthetic_frames(n=6000, videos=2)
    rng = np.random.default_rng(0)
    for video in ("v0", "v1"):
        mask = frames.video_id == video
        shown = [sorted(rng.choice(cats, 3, replace=False)) for _ in range(6000)]
        frames.loc[mask, "objects"] = [json.dumps([{"category": c, "visible_px": 9, "cells": [0]} for c in cs])
                                       for cs in shown]
        held = np.full(6000, None, dtype=object)
        for start in rng.choice(6000, 300, replace=False):
            held[start:start + 3] = cats[int(rng.integers(len(cats)))]
        frames.loc[mask, "held_category"] = held
    config = load_config()
    config["vocabulary"].update(cup="cup", pan="pan")
    manifest, _ = generate_transcripts(frames, config, 5)
    window = config["generator"]["hold_window_s"]
    refs = manifest[manifest.referential]
    assert set(refs.channel) == {"held", "seen", "unseen"} and manifest.channel[~manifest.referential].isna().all()
    for r in refs.itertuples():
        group = frames[frames.video_id == r.video_id].reset_index(drop=True)
        near = set(group.held_category.iloc[max(0, r.source_frame_idx - window):r.source_frame_idx + window + 1].dropna())
        visible = {o["category"] for o in json.loads(group.objects[r.source_frame_idx])}
        assert (r.referent in near) == (r.channel == "held")
        assert (r.referent in visible - near) == (r.channel == "seen")
        if r.channel == "unseen":
            assert r.referent not in visible | near
    shares = refs.channel.value_counts(normalize=True)
    assert shares["held"] == pytest.approx(0.35, abs=0.02)
    assert manifest.attrs["p_held_internal"] > 0.35


def test_referential_draw_without_candidate_becomes_nonreferential():
    frames = synthetic_frames(1000, 2)
    frames["objects"] = '[]'
    frames["held_category"] = None
    config = load_config()
    config["generator"].update(p_ref=1, p_name=1)
    manifest, _ = generate_transcripts(frames, config, 10)
    pool = json.loads((Path(__file__).parents[1] / "nursery/speech/frames.json").read_text())
    assert len(manifest) > 0
    assert not manifest.referential.any() and manifest.referent.isna().all()
    assert not manifest.named.any() and not manifest.held_referent.any()
    assert set(manifest.text) <= set(pool["nonreferential"])


def test_frames_have_no_frequent_pool_exclusive_words():
    pool = json.loads((Path(__file__).parents[1] / "nursery/speech/frames.json").read_text())
    counts = [Counter(re.findall(r"[a-z]+(?:'[a-z]+)?", " ".join(templates).lower().replace("{noun}", "")))
              for templates in pool.values()]
    total = sum(counts, Counter())
    exclusive = {word for word, count in total.items() if count >= 3 and sum(word in c for c in counts) == 1}
    assert not exclusive
    for word in ("the", "where's", "what's", "it", "that", "this", "your", "yours"):
        rates = [c[word] for c in counts]
        assert min(rates) > 0 and max(rates) <= 3 * min(rates)


def test_reject_nonignored_outputs_and_unmatched_sessions(tmp_path):
    with pytest.raises(ValueError, match="ignored"):
        _output_directory(tmp_path / "unignored")
    with pytest.raises(ValueError, match="two train sessions"):
        cue_timelines(synthetic_frames(5, 1), load_config()["cues"], np.random.default_rng(0))


def test_transcript_writer_filters_train(episodes):
    root, run = episodes
    manifest = write_transcripts(root, run, 4)
    assert set(manifest.video_id) == {"v0", "v1"}
    saved = pd.read_parquet(run / "manifest.parquet")
    pd.testing.assert_frame_equal(manifest, saved)
    record = json.loads((run / "generator.json").read_text())
    assert record["n_frames"] == 240
    with pytest.raises(FileExistsError):
        write_transcripts(root, run, 4)


def test_clip_local_windows_and_jitter_preserve_weak_speech():
    frames = synthetic_frames(30, 2)
    # Banana/juice cut, then the same ID again: it is a separate segment.
    clips = ["banana"] * 9 + ["juice"] * 10 + ["banana"] * 10 + ["last"]
    held = ["banana"] * 9 + ["juice"] * 10 + [None] * 11
    for video in ("v0", "v1"):
        mask = frames.video_id == video
        frames.loc[mask, "clip"] = clips
        frames.loc[mask, "held_category"] = held
    config = load_config()
    config["vocabulary"]["juice"] = "juice"
    config["generator"].update(rate_hz=100, sigma=8, p_ref=1, p_name=1)
    group = frames[frames.video_id == "v0"].reset_index(drop=True)
    starts, ends = clip_bounds(group)
    np.testing.assert_array_equal(starts, [0] * 9 + [9] * 10 + [19] * 10 + [29])
    np.testing.assert_array_equal(ends, [8] * 9 + [18] * 10 + [28] * 10 + [29])
    _, _, windows, _ = session_scene(group, config["vocabulary"], 5)
    assert windows[8] == ["banana"] and windows[9] == ["juice"]
    assert windows[19] == [] and windows[29] == []
    # Within-clip delayed naming remains a valid held channel candidate.
    delayed = group.copy()
    delayed["held_category"] = None
    delayed.loc[3, "held_category"] = "banana"
    assert session_scene(delayed, config["vocabulary"], 5)[2][8] == ["banana"]
    manifest, _ = generate_transcripts(frames, config, 7)
    source = manifest.source_frame_idx.to_numpy()
    aligned = manifest.frame_idx.to_numpy()
    assert ((aligned >= starts[source]) & (aligned <= ends[source])).all()
    np.testing.assert_array_equal(manifest.jitter_frames, aligned - source)
    np.testing.assert_array_equal(aligned, np.clip(source + manifest.jitter_draw_frames, starts[source], ends[source]))
    assert (manifest.jitter_frames > 0).any() and (manifest.jitter_frames < 0).any()
    assert (manifest.jitter_frames != manifest.jitter_draw_frames).any()
    assert set(manifest.channel) == {"held", "seen", "unseen"}
    assert manifest.named.all() and (~manifest.held_referent).any()
    for row in manifest.itertuples():
        assert row.held_referent == (row.referent == held[row.frame_idx])
        assert row.source_held_referent == (row.referent == held[row.source_frame_idx])
        if row.channel == "held":
            assert row.referent in windows[row.source_frame_idx]
    assert (manifest.source_frame_idx == 29).any()
    assert (manifest.loc[manifest.source_frame_idx == 29, "jitter_frames"] == 0).all()


def test_delayed_names_and_aligned_holding_provenance():
    frames = synthetic_frames(30, 2)
    frames["held_category"] = None
    frames.loc[frames.frame_idx == 10, "held_category"] = "banana"
    config = load_config()
    config["generator"].update(rate_hz=100, sigma=2, p_ref=1, p_name=1)
    manifest, _ = generate_transcripts(frames, config, 12)
    delayed = manifest[(manifest.channel == "held") & (manifest.source_frame_idx != 10)]
    assert len(delayed) > 0 and (delayed.referent == "banana").all()
    assert not delayed.source_held_referent.any()
    assert (manifest.held_referent != manifest.source_held_referent).any()
    np.testing.assert_array_equal(manifest.held_referent,
                                  (manifest.referent == "banana") & (manifest.frame_idx == 10))
    np.testing.assert_array_equal(manifest.source_held_referent,
                                  (manifest.referent == "banana") & (manifest.source_frame_idx == 10))


@pytest.mark.parametrize("invalid", [None, "", " ", 12])
def test_reject_missing_or_invalid_clip_identity(invalid):
    frames = synthetic_frames(30, 2)
    frames["clip"] = frames["clip"].astype(object)
    frames.loc[0, "clip"] = invalid
    with pytest.raises(ValueError, match="clip identity"):
        generate_transcripts(frames, load_config(), 4)
    with pytest.raises(ValueError, match="clip identity"):
        generate_transcripts(frames.drop(columns="clip"), load_config(), 4)
