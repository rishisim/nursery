"""Freeze U2 caregiver speech and source-clip compositional trials (CPU only)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nursery.common import REPO, _output_directory, load_config
from nursery.learner import _upstream_module
from nursery.learner.acquisition_data import _sha
from nursery.learner.body import clip_index
from nursery.learner.gaze_data import TemporalVideoData, partition_window_keys
from nursery.speech.transcripts import session_scene

STEP9 = REPO / "outputs/runs/step9_acquisition_curves/frozen"
RENDER = REPO / "outputs/renders/clean_856"
MASS = REPO / "outputs/step7_scouting/instance_draw_mass.csv"
FROZEN = REPO / "outputs/runs/unified_u2_data/frozen"

KEY = ["video_id", "frame_idx"]
HELD_OUT = ("test", "s8")
STRADDLING = {"bell", "bottle", "doll", "rock"}
MASS_NOUNS = {
    "heavy": "bank binoculars cake clock controller glass hammer headphones mug pan teapot umbrella".split(),
    "light": "balloon bird bulb cookie glasses key knife knob pen pipe scissors spoon".split(),
}

# Every choice in a trial uses the same template for its predicate. Evaluation
# carriers are reserved in their entirety, including the VP-swap carrier.
TEST_ACTION = {
    "pass": "pass the {noun} over here",
    "lift": "lift the {noun} up high",
    "inspect": "take a look at the {noun}",
    "offhand": "switch hands with the {noun}",
    "drink": "have a drink from the {noun}",
}
TEST_PROPERTY = "point to the {mass} {noun}"
TEST_NEGATION = "holding the {noun}, not the {other}"
TEST_VP = "the {noun} is {mass}"

TRAIN_NAMED = (
    "Can you see the {noun}?",
    "Look, there's the {noun}.",
    "Where did the {noun} go?",
    "Do you see the {noun}?",
    "I can see the {noun}.",
    "What about the {noun}?",
    "Let's find the {noun}.",
    "Oh, there's the {noun}!",
    "You found the {noun}!",
    "Shall we get the {noun}?",
    "Look at the {noun} there.",
    "Have you got the {noun}?",
)
TRAIN_PRONOUN = (
    "Can you see it now?", "You've got it right there.", "Here, let me see it.",
    "Want to play with that one?", "Where did it go now?", "Oh, look at that one!",
    "Are you holding it now?", "Can I have it back?", "Is it over there?",
    "See if you can find it.", "That one is over here.",
    "Can you point to it?", "Can I take a look?",
    "Let's switch hands now.", "Can you reach up high?",
)
TRAIN_CHAT = (
    "Oh wow, there we go!", "Okay, now it's your turn.", "What should we do now?",
    "Did you see that too?", "Here, let me help you.", "All right, here we go.",
    "You can do it yourself.", "That's it, there you go.", "Did you hear that?",
    "Shall we do that again?", "Thanks for helping me!", "Oh, I see it now.",
)
TRAIN_ACTION = {
    "pass": ("Can you pass {obj}?", "Pass me {obj}, please.", "Let's pass {obj} back."),
    "lift": ("Can you lift {obj}?", "Let's lift {obj} together.", "Lift {obj} for me."),
    "inspect": ("Let's inspect {obj} together.", "We can inspect {obj} now.", "Let's look at {obj} closely."),
    "offhand": ("Try {obj} in your other hand.", "Can your other hand hold {obj}?"),
    "drink": ("Let's drink from {obj}.", "Can you drink from {obj}?", "Want to drink from {obj}?"),
}
TRAIN_PROPERTY = (
    "Look at this {mass} {noun}.", "What a {mass} thing, that {noun}!",
    "Oh, the {mass} {noun}!", "See that {mass} {noun} there?",
)
TRAIN_PROPERTY_CHAT = (
    "That feels {mass}. See the {noun}?", "Oh, {mass}! Where's the {noun}?",
    "Feels {mass} to me. The {noun}?",
)
TRAIN_NEGATION = (
    "I've got the {noun}, not the {other}.",
    "It's the {noun} in my hand, not the {other}.",
    "Look, the {noun}, not the {other}.",
)


def vp_swap(rng):
    splits = {k: {} for k in ("seen", "held_out")}
    for mass, nouns in MASS_NOUNS.items():
        order = rng.permutation(nouns).tolist()
        splits["seen"][mass], splits["held_out"][mass] = sorted(order[:8]), sorted(order[8:])
    rows = []
    for split, nouns in splits.items():
        for heavy in nouns["heavy"]:
            for light in nouns["light"]:
                rows.append(dict(split=split, heavy_noun=heavy, light_noun=light,
                                 true_heavy=TEST_VP.format(noun=heavy, mass="heavy"),
                                 true_light=TEST_VP.format(noun=light, mass="light"),
                                 swap_heavy=TEST_VP.format(noun=heavy, mass="light"),
                                 swap_light=TEST_VP.format(noun=light, mass="heavy")))
    return pd.DataFrame(rows), splits


def _speech_alignment(candidates, rng):
    # Step 9 estimates available channels before sampling. Fit the channel
    # probabilities again for U2's aligned-frame targets and eligible nouns.
    target = np.array([.16, .19, .65])
    available = np.array([[bool(pool) for pool in row] for row in candidates])
    weights = target.copy()
    for _ in range(100):
        prob = available * weights
        prob /= prob.sum(1, keepdims=True)
        achieved = prob.mean(0)
        if np.max(abs(achieved - target)) < 1e-8:
            break
        weights *= target / achieved
    return [rng.choice(3, p=p) for p in prob]


def generate_speech(schedule, frames, noun_splits, rng):
    """Reuse Step 9 times, jitter, cue weights and source-event annotations."""
    upstream = _upstream_module("train")
    result = schedule.copy()
    vocabulary = load_config()["vocabulary"]
    forbidden = set(sum(noun_splits["held_out"].values(), []))
    matching_mass = {noun: mass for mass, nouns in noun_splits["seen"].items() for noun in nouns}

    timeline = frames.set_index(KEY)
    scenes = frames.groupby("video_id", sort=False).visible.agg(lambda v: set().union(*v))
    named = result[result.named]
    pools = []
    for row in named.itertuples():
        moment = timeline.loc[row.video_id, row.frame_idx]
        visible = moment.visible
        held = {moment.held_category} & visible
        other = visible - held
        absent = scenes[row.video_id] - visible - {moment.held_category}
        eligible = set(vocabulary)
        if pd.notna(row.property):
            eligible -= forbidden
        if row.action == "drink":
            eligible.discard("bell")
        pools.append(tuple(sorted(p & eligible) for p in (held, other, absent or set(vocabulary) - visible)))

    assignments = {}
    for row, pool, bucket in zip(named.itertuples(), pools, _speech_alignment(pools, rng)):
        choices = pool[bucket]
        # Retain the original noun when it belongs to the selected channel.
        # Prefer plausible mass wording if a seen VP noun is available.
        plausible = [noun for noun in choices if matching_mass.get(noun) == row.property]
        if plausible and bucket != 1:
            choices = plausible
        noun = row.referent if row.referent in choices else str(rng.choice(choices))
        assignments[row.Index] = noun, ("matching", "other_visible", "absent")[bucket]

    result["noun_alignment"] = None
    result["negated_noun"] = None
    for row in result.itertuples():
        i = row.Index
        if not row.named:
            result.at[i, "text"] = rng.choice(TRAIN_PRONOUN if row.referential else TRAIN_CHAT)
            continue
        noun, alignment = assignments[i]
        moment = timeline.loc[row.video_id, row.frame_idx]
        source = timeline.loc[row.video_id, row.source_frame_idx]
        result.at[i, "referent"] = noun
        result.at[i, "noun_alignment"] = alignment
        result.at[i, "channel"] = dict(matching="held", other_visible="seen", absent="unseen")[alignment]
        result.at[i, "referent_visible"] = noun in moment.visible
        result.at[i, "held_referent"] = noun == moment.held_category
        result.at[i, "source_held_referent"] = noun == source.held_category
        result.at[i, "gazed_referent"] = noun == moment.gaze_category
        result.at[i, "source_gazed_referent"] = noun == source.gaze_category

        has_action, has_mass = pd.notna(row.action), pd.notna(row.property)
        distractors = sorted(moment.visible - {noun})
        names_mass = has_mass and noun == moment.rendered_noun
        if has_action:
            obj = f"the {row.property} {noun}" if names_mass else f"the {noun}"
            text = rng.choice(TRAIN_ACTION[row.action]).format(obj=obj)
            if has_mass and not names_mass:
                text += f" It feels {row.property}."
        elif has_mass:
            frames_for_mass = TRAIN_PROPERTY if names_mass else TRAIN_PROPERTY_CHAT
            text = rng.choice(frames_for_mass).format(mass=row.property, noun=noun)
        elif alignment == "matching" and distractors and rng.random() < .5:
            other = rng.choice(distractors)
            result.at[i, "negated_noun"] = other
            text = rng.choice(TRAIN_NEGATION).format(noun=noun, other=other)
        else:
            text = rng.choice(TRAIN_NAMED).format(noun=noun)
        result.at[i, "text"] = text

    config = load_config()["cues"]
    oracle = np.where(result.named & result.referent_visible, config["w_hi"], config["w_lo"])
    result["weight_oracle"] = oracle / oracle.mean()
    tokens = result.text.map(upstream.tokenize)
    if any(forbidden.intersection(words) and {"heavy", "light"}.intersection(words) for words in tokens):
        raise ValueError("Held-out noun appears in mass speech")
    return result


def action_trials(events, timeline, rng):
    events = events[events.referent.isin(TEST_ACTION)].merge(timeline[[*KEY, "visible"]], on=KEY)
    events = events[~(events.referent.eq("drink") & events.rendered_noun.eq("bell"))]
    events = events[[noun in visible for noun, visible in zip(events.rendered_noun, events.visible)]]
    clips = events.drop_duplicates("clip")
    by_label = clips.groupby(["referent", "rendered_noun"])["clip"].agg(list).to_dict()
    rows = []
    for clip, moments in events.groupby("clip", sort=True):
        for row in moments.itertuples():
            action, noun = row.referent, row.rendered_noun
            choices = [(a, n) for a, n in sorted(by_label)
                       if a != action and n != noun and n not in row.visible
                       and (a, noun) in by_label and (action, n) in by_label]
            if choices:
                other_action, other_noun = choices[rng.integers(len(choices))]
                labels = [(action, noun), (other_action, noun), (action, other_noun), (other_action, other_noun)]
                rows.append(dict(video_id=row.video_id, frame_idx=row.frame_idx, clip=clip, split=row.split,
                                 subject=row.subject,
                                 action=action, noun=noun, action_foil=other_action, noun_foil=other_noun,
                                 choices=[TEST_ACTION[a].format(noun=n) for a, n in labels],
                                 foil_clips=[str(rng.choice(by_label[label])) for label in labels[1:]]))
                break
    return pd.DataFrame(rows)


def property_trials(events, timeline, rng):
    mass_frames = timeline[timeline.split.isin(HELD_OUT) & timeline.mass.isin(MASS_NOUNS)]
    groups = mass_frames.groupby("clip")
    lifted = events.set_index("clip")
    catalogue = mass_frames.drop_duplicates("clip")
    rows = []
    for clip, moments in groups:
        if clip in lifted.index:
            frame = lifted.loc[clip]
        else:
            held = moments[moments.held_category.eq(moments.rendered_noun)]
            visible = held[[noun in vis for noun, vis in zip(held.rendered_noun, held.visible)]]
            options = next(group for group in (visible, held, moments) if len(group))
            frame = options.iloc[len(options) // 2]
        noun, mass = frame.rendered_noun, moments.mass.iloc[0]
        opposite = "heavy" if mass == "light" else "light"
        matched = catalogue[catalogue.rendered_noun.eq(noun) & catalogue.mass.eq(opposite)]
        others = catalogue[catalogue.rendered_noun.ne(noun)]
        window = mass_frames.loc[(mass_frames.video_id == frame.video_id) &
                                 (mass_frames.frame_idx == frame.frame_idx), "visible"].iloc[0]
        absent = others[~others.rendered_noun.isin(window)]
        if len(absent):
            others = absent
        # Prefer a noun attested at both masses, so both noun foils are plausible.
        both = set(others[others.mass.eq(mass)].rendered_noun) & set(others[others.mass.eq(opposite)].rendered_noun)
        other = str(rng.choice(sorted(both or set(others[others.mass.eq(mass)].rendered_noun))))
        labels = [(mass, noun), (opposite, noun), (mass, other), (opposite, other)]
        rows.append(dict(video_id=frame.video_id, frame_idx=int(frame.frame_idx), clip=clip, split=frame.split,
                         subject=frame.subject,
                         mass=mass, noun=noun, noun_foil=other, instance_key=moments.instance_key.iloc[0],
                         straddling=noun in STRADDLING, lifted_event=clip in lifted.index,
                         same_noun_opposite_clip=str(rng.choice(matched["clip"])) if len(matched) else None,
                         choices=[TEST_PROPERTY.format(mass=m, noun=n) for m, n in labels]))
    return pd.DataFrame(rows)


def negation_trials(timeline, rng):
    rows = []
    for clip, moments in timeline[timeline.split.isin(HELD_OUT)].groupby("clip"):
        eligible = [(r, sorted(r.visible - {r.held_category})) for r in moments.itertuples()
                    if r.held_category in r.visible and r.visible - {r.held_category}]
        if eligible:
            row, distractors = eligible[len(eligible) // 2]
            noun, other = row.held_category, str(rng.choice(distractors))
            rows.append(dict(video_id=row.video_id, frame_idx=row.frame_idx, clip=clip, split=row.split,
                             subject=row.subject,
                             noun=noun, noun_foil=other,
                             choices=[TEST_NEGATION.format(noun=noun, other=other),
                                      TEST_NEGATION.format(noun=other, other=noun)]))
    return pd.DataFrame(rows)


def _read_events(kind):
    return pd.concat([pd.read_parquet(STEP9 / "panels" / f"{kind}_{split}" / "manifest.parquet")
                      .assign(split=split) for split in HELD_OUT], ignore_index=True)


def cached_replays(frames, protocol, gaze):
    """Pick replay moments with complete existing static and temporal inputs."""
    visual_index = Path(protocol["inputs"]) / "train_cache/index.parquet"
    closeup_index = Path(protocol["gaze"]["closeup_cache"]) / "index.parquet"
    partitions_path = Path(protocol["gaze"]["partitions"])

    def keys(rows):
        return (rows.video_id + "|" + rows.frame_idx.astype(str)).tolist()

    lookup = {key: (0, i) for i, key in enumerate(keys(pd.read_parquet(visual_index)))}
    closeups = set(keys(pd.read_parquet(closeup_index)))
    needed = set(keys(gaze[gaze.gaze_x.notna()]))
    supported = (lookup.keys() & set(keys(gaze))) - (needed - closeups)
    partitions = json.loads(partitions_path.read_text())
    videos = sorted(frames.video_id.unique())
    timeline = (dict(zip(videos, range(len(videos)))), clip_index(frames, videos, frames.frame_idx.max() + 1)[0])
    rows = frames[frames.split.isin(HELD_OUT)]
    static = partition_window_keys(rows, lookup, partitions, 2)
    temporal = TemporalVideoData(rows, (), lookup, timeline).frame_keys
    keep = [bool(a) and supported.issuperset(a) and supported.issuperset(k for k in b if k is not None)
            for a, b in zip(static, temporal)]
    return rows[keep], [visual_index, closeup_index, partitions_path]


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def training_rows(frozen, subset):
    clips = set(json.loads((subset / "stats.json").read_text())["selected_clips"])
    manifest = pd.read_parquet(frozen / "manifest.parquet")
    rows = manifest[manifest["clip"].isin(clips)].reset_index(drop=True)
    if set(rows["clip"]) != clips:
        raise ValueError("U2 speech does not cover the selected Step 9 clips")
    return rows


def freeze(out, seed):
    split_path, manifest_path = STEP9 / "splits.json", STEP9 / "subsets/1.0/manifest.parquet"
    protocol_path = STEP9 / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    gaze_path = Path(protocol["gaze"]["points"])
    frame_path = RENDER / "frames.parquet"
    splits = json.loads(split_path.read_text())["source_keys"]
    split_by_clip = {clip: split for split, clips in splits.items() for clip in clips}
    gaze = pd.read_parquet(gaze_path, columns=[*KEY, "gaze_category", "gaze_x"])
    frames = pd.read_parquet(frame_path).merge(gaze[[*KEY, "gaze_category"]],
                                             on=KEY, how="left", validate="one_to_one")
    if set(split_by_clip) != set(frames["clip"]):
        raise ValueError("Frame source keys differ from the frozen split")
    frames["split"] = frames["clip"].map(split_by_clip)
    vocabulary = load_config()["vocabulary"]
    scenes, instance_keys, physical_paths = {}, {}, []
    for video, group in frames.groupby("video_id", sort=True):
        group = group.sort_values("frame_idx")
        scenes.update(zip(group.index, map(set, session_scene(group, vocabulary, 0)[0])))
        physical_paths.append(RENDER / "physical_instance" / f"{video}.npz")
        with np.load(physical_paths[-1]) as data:
            np.testing.assert_array_equal(data["frame_idx"], group.frame_idx)
            for clip, key in set(zip(group["clip"], data["instance_key"])):
                if instance_keys.setdefault(clip, key) != key:
                    raise ValueError(f"Mass instance changes across replays of {clip}")
    frames["visible"] = pd.Series(scenes)
    frames["instance_key"] = frames["clip"].map(instance_keys)
    masses = pd.read_csv(MASS).set_index("instance_key")
    frames["mass"] = frames.instance_key.map(masses.tertile)
    frames["rendered_noun"] = frames.instance_key.map(masses.rendered_noun)
    if frames.rendered_noun.isna().any():
        raise ValueError("A rendered instance is missing from the mass draw")

    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    quads, noun_splits = vp_swap(streams[0])
    speech = generate_speech(pd.read_parquet(manifest_path), frames, noun_splits, streams[1])
    covered, cache_indices = cached_replays(frames, protocol, gaze)
    property_events = _read_events("property").merge(covered[KEY], on=KEY).drop_duplicates("clip")
    panels = {"action_object": action_trials(_read_events("action"), covered, streams[2]),
              "property_object": property_trials(property_events, covered, streams[3]),
              "negation": negation_trials(covered, streams[4])}

    upstream = _upstream_module("train")
    test_text = {c for panel in panels.values() for choices in panel.choices for c in choices}
    test_text.update(quads[["true_heavy", "true_light", "swap_heavy", "swap_light"]].to_numpy().ravel())
    train_sequences = {tuple(upstream.tokenize(s)) for s in speech.text}
    test_sequences = {tuple(upstream.tokenize(s)) for s in test_text}
    if train_sequences & test_sequences:
        raise ValueError("Train and test surface forms overlap")
    unknown = set().union(*test_sequences) - set().union(*train_sequences)
    if unknown:
        raise ValueError(f"Full training speech lacks panel vocabulary: {sorted(unknown)}")
    counts = {}
    for name, panel in panels.items():
        if panel["clip"].duplicated().any():
            raise ValueError(f"{name}: expected one trial per clip")
        counts[name] = {split: dict(clips=len(rows), trials=len(rows))
                        for split, rows in panel.groupby("split")}
        counts[name]["pooled"] = dict(clips=len(panel), trials=len(panel))
    properties = panels["property_object"]
    counts["property_object"].update(mass=properties.mass.value_counts().to_dict(),
                                     straddling=dict(clips=int(properties.straddling.sum()),
                                                     trials=int(properties.straddling.sum())),
                                     outside_lifted_panel=int((~properties.lifted_event).sum()))
    counts["vp_swap"] = {split: dict(quads=len(part), pairs=2 * len(part))
                         for split, part in quads.groupby("split")}
    tokens = speech.text.map(upstream.tokenize)
    stats = dict(seed=seed, split_clips={k: len(v) for k, v in splits.items()},
                 clips=speech["clip"].nunique(), utterances=len(speech),
                 utterances_per_clip=len(speech) / speech["clip"].nunique(),
                 tokens_mean=float(tokens.map(len).mean()),
                 jitter_sigma_s=load_config()["generator"]["sigma"],
                 jitter_rounded_sd_s=float(speech.jitter_draw_frames.std(ddof=0)),
                 jitter_effective_sd_s=float(speech.jitter_frames.std(ddof=0)),
                 noun_alignment=speech.noun_alignment.value_counts(normalize=True).to_dict(),
                 action_alignment=speech.action_alignment.value_counts(normalize=True).to_dict(),
                 property_alignment=speech.property_alignment.value_counts(normalize=True).to_dict(),
                 negation_utterances=int(speech.negated_noun.notna().sum()), vp_nouns=noun_splits)
    stats["vp_training_cooccurrences"] = {
        noun: {mass: sum(noun in words and mass in words for words in tokens) for mass in MASS_NOUNS}
        for group in noun_splits.values() for nouns in group.values() for noun in nouns
    }
    if any(stats["vp_training_cooccurrences"][noun][mass] == 0
           for mass, nouns in noun_splits["seen"].items() for noun in nouns):
        raise ValueError("A seen VP noun has no correct mass co-occurrence in training")
    inputs = [frame_path, MASS, split_path, manifest_path, gaze_path, protocol_path, *cache_indices, *physical_paths]
    inputs += [STEP9 / "panels" / f"{kind}_{split}" / "manifest.parquet"
               for kind in ("action", "property") for split in HELD_OUT]
    input_hashes = {str(path.relative_to(REPO)): _sha(path) for path in sorted(inputs)}

    out = _output_directory(out, exist_ok=False)
    for name, data in {"manifest": speech, **panels, "vp_swap": quads}.items():
        data.to_parquet(out / f"{name}.parquet", index=False)
    _write_json(out / "panel_counts.json", counts)
    _write_json(out / "speech_stats.json", stats)
    _write_json(out / "hashes.json", dict(inputs=input_hashes,
                                         outputs={p.name: _sha(p) for p in sorted(out.iterdir())}))
    print(json.dumps(dict(speech=stats, panels=counts), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=FROZEN)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    freeze(args.out_dir, args.seed)


if __name__ == "__main__":
    main()
