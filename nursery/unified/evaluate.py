"""Evaluate frozen U1 and U2 panels with per-word Soft-MIL."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nursery.common import REPO
from nursery.learner import _upstream_module
from nursery.learner.acquisition import _gaze_data
from nursery.learner.acquisition_data import _panel_donors
from nursery.learner.acquisition_evaluate import StudyEvaluation, _metrics
from nursery.learner.body import clip_index, load_store, shuffle_store
from nursery.learner.gaze_data import (
    TemporalVideoData,
    collate_frame_indices,
    partition_window_keys,
)
from nursery.learner.train import _pair_meta
from nursery.learner.windows import BodyData

from .language import CausalLanguage
from .model import UnifiedEmbodiedLearner
from .u2_data import FROZEN as U2_FROZEN, training_rows


STUDY = REPO / "outputs/runs/step9_acquisition_curves/frozen"
ROUTING_COLUMNS = ["p_vis", "p_temp", "p_body"]
TEXT_ONLY = "Text-Only LM"
U2_PANELS = ("action_object", "property_object", "negation")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def step9_subset(protocol, n_clips):
    for fraction in protocol["fractions"]:
        subset = STUDY / "subsets" / str(float(fraction))
        if read_json(subset / "stats.json")["selected_clip_count"] == n_clips:
            return subset
    raise ValueError(f"No frozen Step 9 subset with {n_clips} clips")


def sentence_inputs(sentences, vocab, device, *, tokenize=None):
    tokenize = tokenize or _upstream_module("train").tokenize
    tokenized = [tokenize(sentence) for sentence in sentences]
    words = torch.zeros((len(tokenized), max(map(len, tokenized))), dtype=torch.long, device=device)
    counts = torch.tensor(list(map(len, tokenized)), device=device)
    known = []
    for index, tokens in enumerate(tokenized):
        known.append(bool(tokens) and all(token in vocab for token in tokens))
        words[index, :len(tokens)] = torch.tensor([vocab.get(token, 0) for token in tokens], device=device)
    return words, counts, np.array(known)


def choice_credits(scores, known):
    """Column zero is true. Split exact ties uniformly across top choices."""
    top = scores == scores.max(axis=1, keepdims=True)
    return np.where(known, top[:, 0] / top.sum(axis=1), 1 / scores.shape[1])


def score_sentence_choices(model, sensory, words, counts, n_choices):
    ordered = words.sort(dim=1, descending=True).values if model.bow_only else words
    vectors = model.language.encode_tokens(ordered, counts, bow_only=model.bow_only)
    matrix = model.score(sensory, vectors)
    clips = torch.arange(len(matrix), device=words.device)[:, None]
    slots = torch.arange(len(words), device=words.device).reshape(-1, n_choices)
    return matrix[clips, slots], vectors


@torch.no_grad()
def evaluate_vp_swap(language, vocab, path, device):
    """Score within-noun preferences and their summed margin (chance 25% and 50%)."""
    language.eval()
    rows = pd.read_parquet(path / "vp_swap.parquet")
    columns = ["true_heavy", "true_light", "swap_heavy", "swap_light"]
    queries = sorted(set(rows[columns].to_numpy().ravel()))
    words, counts, known = sentence_inputs(queries, vocab, device)
    scores = language.sentence_logprob(words, counts).cpu().numpy()
    lookup = {s: (float(p), bool(k)) for s, p, k in zip(queries, scores, known)}
    results = {}
    for split, group in rows.groupby("split"):
        trials = []
        for row in group.itertuples():
            scores, known = (np.array(a).reshape(2, 2) for a in zip(*(lookup[getattr(row, c)] for c in columns)))
            quad_known = bool(known.all())
            differences = scores[0] - scores[1]  # One true-minus-swapped score per noun.
            swap_margin = float(differences.sum()) if quad_known else None
            quad_credit = float((differences > 0).all()) if quad_known else .25
            swap_margin_credit = float(swap_margin > 0) if quad_known else .5
            trials.append(dict(heavy_noun=row.heavy_noun, light_noun=row.light_noun,
                               scores=[float(p) if k else None for p, k in zip(scores.flat, known.flat)],
                               quad_known=quad_known, quad_credit=quad_credit,
                               swap_margin=swap_margin, swap_margin_credit=swap_margin_credit))
        results[split] = dict(quads=len(trials),
                              quad_accuracy=100 * float(np.mean([t["quad_credit"] for t in trials])),
                              swap_margin_accuracy=100 * float(np.mean([t["swap_margin_credit"] for t in trials])),
                              quad_coverage=float(np.mean([t["quad_known"] for t in trials])),
                              trials=trials)
    return results


class UnifiedEvaluation(StudyEvaluation):
    """Reuse Step 9 sensory inputs and the historical U1 query path."""

    def __init__(self, n_clips, condition, device, *, shuffle_body=False, u2_path=None):
        if shuffle_body and condition != "V+G+B":
            raise ValueError("Shuffled body requires V+G+B")
        protocol = read_json(STUDY / "protocol.json")
        subset = step9_subset(protocol, n_clips)

        inputs, episodes = Path(protocol["inputs"]), Path(protocol["episodes"])
        caches, lut = _upstream_module("train_frame_mil").load_multi([inputs / "train_cache"])
        if caches[0].shape[1:] != (257, 768):
            raise ValueError("U1 requires the cached DINOv2-Base 16x16 patch grid and CLS")

        with np.load(subset / "fit_mask.npz", allow_pickle=False) as fit:
            videos, include = fit["video_ids"].tolist(), fit["include"]
        frames = pd.read_parquet(episodes / "frames.parquet",
                                 columns=["video_id", "frame_idx", "clip", "held_category"])
        self.source_clips = frames.set_index(["video_id", "frame_idx"])["clip"]
        store = None
        if condition == "V+G+B":
            store = load_store(episodes, frames, videos, set(), steps=include.shape[1],
                               exclude=~include, physical=protocol["physical"],
                               representation=protocol["arms"]["video_gaze_body_current"]["body_representation"])
        self.u2_rows = training_rows(u2_path, subset) if u2_path is not None else None
        vocab = (read_json(subset / "vocab.json") if self.u2_rows is None else
                 _upstream_module("train").build_vocab(self.u2_rows.text, min_freq=1))
        super().__init__(STUDY, inputs, caches, lut, vocab, store, device)
        self.word_types = {rung: self.word_types[rung] for rung in ("noun", "action")}
        self.condition, self.subset = condition, subset
        self.body_mode = "shuffled" if shuffle_body else "real"
        self.include = include
        self.shuffle_info = None
        self.allowed_keys = {f"{video}|{frame}" for i, video in enumerate(videos)
                             for frame in np.flatnonzero(include[i])}
        clip = store["clip"] if store is not None else clip_index(frames, videos, include.shape[1])[0]
        self.timeline = {video: i for i, video in enumerate(videos)}, clip

    def gaze_data(self, rows, **kwargs):
        return _gaze_data(self.protocol, rows, self.lut, "video_gaze", **kwargs)

    def temporal_data(self, rows, *, allowed_keys=None):
        adapter = TemporalVideoData(rows, self.caches, self.lut, self.timeline,
                                    allowed_keys=allowed_keys)
        if self.condition != "V":
            adapter.gaze = self.gaze_data(rows, frame_keys=adapter.frame_keys,
                                          window=5, allowed_keys=allowed_keys)
        return adapter

    @property
    def body_dims(self):
        if self.store is not None:
            return {name: values.shape[-1] for name, values in self.store["groups"].items()}
        return None

    def training_data(self, *, seed=0):
        rows = self.u2_rows if self.u2_rows is not None else pd.read_parquet(self.subset / "manifest.parquet")
        window = self.protocol["window"]
        ds = self.upstream.FramePairs(self.caches, self.lut, rows, self.vocab, window, False,
                                      max_len=32 if self.u2_rows is not None else 16)
        if len(ds) < 2 or ds.n_noframe or ds.n_novocab:
            raise ValueError("Frozen subset lost utterances or cached frames")
        keys = partition_window_keys(rows, self.lut, self.partitions, window)
        keys = [[key for key in here if key in self.allowed_keys] for here in keys]
        if any(not here for here in keys):
            raise ValueError("Training utterance has no in-subset visual frames")
        ds.locs = [[self.lut[key] for key in here] for here in keys]
        gaze = None
        if self.condition != "V":
            gaze = self.gaze_data(rows, frame_keys=keys, allowed_keys=self.allowed_keys)
        temporal = self.temporal_data(rows, allowed_keys=self.allowed_keys)
        return ds, gaze, temporal, self._training_body(rows, window, seed)

    def _training_body(self, rows, window, seed):
        store = self.store
        if store is None:
            return None
        if self.body_mode == "shuffled":
            # Limit both donors and recipients to the frozen training seconds.
            training_store = dict(store, clip=np.where(self.include, store["clip"], -1),
                                  present=store["present"] & self.include)
            store, self.shuffle_info = shuffle_store(training_store, seed)
            if not self.shuffle_info["moved"]:
                raise ValueError("No body sequences could be shuffled in the frozen training subset")
        ps, pt, win = _pair_meta(rows, store, self.lut, window)
        return BodyData(store, ps, pt, win, None, None, None, None, self.device,
                        half=self.protocol["body_window"], dropout=self.protocol["body_dropout"])

    def sensory_batch(self, model, ds, ids, video, gaze, temporal, body,
                      *, gaze_rng=None, body_rng=None, body_ids=None):
        device = self.device
        video = video.to(device=device, dtype=torch.float32).reshape(len(ids), ds.Fmax, 257, 768)
        lengths = torch.tensor([len(ds.locs[int(i)]) for i in ids], device=device)
        valid = torch.arange(ds.Fmax, device=device) < lengths[:, None]
        gaze_ctx = None
        if gaze is not None:
            dropout = self.protocol["gaze"]["dropout"] if gaze_rng is not None else 0.0
            gaze_ctx = gaze.batch(ids, gaze_rng, device=device, dropout=dropout)
        temporal_ctx = temporal.batch(model, ids, device=device,
                                      gaze_mode="real" if gaze_ctx is not None else "removed",
                                      keep=gaze_ctx["keep"] if gaze_ctx is not None else None)
        body_ctx = None
        if body is not None:
            body_ctx = body.batch(ids if body_ids is None else body_ids, body_rng,
                                  training=body_rng is not None)
        return video, valid, temporal_ctx, gaze_ctx, body_ctx

    @torch.no_grad()
    def evaluate(self, model, panel, body_mode="real", *, gaze_mode=None, body_weights=None,
                 routing=False):
        # The inherited dev/test aggregators call this hook on each frozen panel.
        if body_weights is not None or body_mode not in ("real", "shuffled", "none") or gaze_mode not in (None, "removed"):
            raise ValueError("Unsupported U1 evaluation mode")
        model.eval()
        rows, trials, ds, body, _ = self._panel(panel)
        if body_mode == "shuffled" and body is None:
            raise ValueError("Shuffled evaluation requires body histories")
        donors = np.asarray(trials["donor_rows"]) if body_mode == "shuffled" else None
        if gaze_mode != "removed" and self.condition != "V" and panel not in self.gaze_panels:
            self.gaze_panels[panel] = self.gaze_data(rows, frame_keys=self.panel_frame_keys[panel])
        gaze = self.gaze_panels.get(panel) if gaze_mode != "removed" else None
        if panel not in self.temporal_panels:
            self.temporal_panels[panel] = self.temporal_data(rows)

        queries = rows.iloc[trials["trial_rows"]].referent.tolist()
        word_index, text, _ = self._query_vectors(model, queries)
        if routing:
            row_ids = np.unique(trials["trial_rows"])
            row_ids = row_ids[rows.iloc[row_ids].referent.isin(word_index).to_numpy()]
            targets = rows.iloc[row_ids]
            query_ids = targets.referent.map(word_index).to_numpy(dtype=int)
            probabilities = np.empty((len(targets), 3), dtype=np.float32)
        else:
            row_ids = np.arange(len(rows)) if text is not None else np.array([], dtype=int)
            matrix = np.empty((len(rows), len(word_index)), dtype=np.float32)

        for start in range(0, len(row_ids), 64):
            ids = row_ids[start:start + 64]
            _, video, _, _ = collate_frame_indices(ds, ids)
            args = self.sensory_batch(model, ds, ids, video, gaze, self.temporal_panels[panel],
                                      body if body_mode != "none" else None,
                                      body_ids=donors[ids] if donors is not None else None)
            sensory = model.encode_sensory(*args)
            if routing:
                batch = slice(start, start + len(ids))
                posterior = model.posterior_routing(sensory, text).cpu().numpy()
                probabilities[batch] = posterior[np.arange(len(ids)), query_ids[batch]]
            else:
                matrix[ids] = model.score(sensory, text).cpu().numpy()
        if routing:
            matched = pd.DataFrame(probabilities, columns=ROUTING_COLUMNS)
            matched["word"] = targets.referent.to_numpy()
            keys = pd.MultiIndex.from_frame(targets[["video_id", "frame_idx"]])
            matched["clip"] = self.source_clips.loc[keys].to_numpy()
            return matched

        candidates = np.asarray(trials["candidate_rows"], dtype=int)
        scores = np.zeros(candidates.shape, dtype=np.float32)
        if text is not None:
            for i, word in enumerate(queries):
                if word in word_index:
                    scores[i] = matrix[candidates[i], word_index[word]]
        result = _metrics(scores, [q in word_index for q in queries], candidates,
                          trials["correct_slots"], queries)
        result["body_mode"] = body_mode if body is not None else "none"
        if donors is not None:
            result["donor_rows"] = donors.tolist()
        return result

    def development(self, model):
        return {rung: self.evaluate(model, spec["dev"], body_mode=self.body_mode)
                for rung, spec in self.word_types.items()}

    @torch.no_grad()
    def compositional(self, model, path):
        model.eval()
        results, routing = {}, []
        tokenize = _upstream_module("train").tokenize
        for panel in U2_PANELS:
            rows = pd.read_parquet(path / f"{panel}.parquet")
            images = rows.assign(text=next(w for w in self.vocab if self.upstream.encode(w, self.vocab)))
            ds = self.upstream.FramePairs(self.caches, self.lut, images, self.vocab, 2, False)
            keys = partition_window_keys(rows, self.lut, self.partitions, 2)
            if len(ds) != len(rows) or any(not k for k in keys):
                raise ValueError(f"Missing visual input for U2 panel {panel}")
            ds.locs = [[self.lut[k] for k in here] for here in keys]
            gaze = self.gaze_data(rows, frame_keys=keys) if self.condition != "V" else None
            temporal = self.temporal_data(rows)
            body, donors = None, None
            if self.store is not None:
                ps, pt, win = _pair_meta(rows, self.store, self.lut, 2)
                body = BodyData(self.store, ps, pt, win, None, None, None, None,
                                self.device, 5, 0., 0.)
                if self.body_mode == "shuffled":
                    donors = np.asarray(_panel_donors(rows.subject.tolist(), rows["clip"].tolist()))

            scores, known = [], []
            n_choices = len(rows.choices.iloc[0])
            for start in range(0, len(rows), 8):
                ids = np.arange(start, min(start + 8, len(rows)))
                batch = rows.iloc[ids]
                _, video, _, _ = collate_frame_indices(ds, ids)
                sensory = model.encode_sensory(*self.sensory_batch(
                    model, ds, ids, video, gaze, temporal, body,
                    body_ids=donors[ids] if donors is not None else None))
                sentences = batch.choices.tolist()
                words, counts, covered = sentence_inputs([s for trial in sentences for s in trial],
                                                         self.vocab, self.device, tokenize=tokenize)
                batch_scores, vectors = score_sentence_choices(model, sensory, words, counts, n_choices)
                scores.extend(batch_scores.cpu().numpy())
                known.extend(covered.reshape(-1, n_choices).all(axis=1))

                if model.bow_only:
                    vectors = model.language.encode_tokens(words, counts, bow_only=True)
                vectors, content = (v[::n_choices] for v in vectors)
                trial_ids, word_ids = content.nonzero(as_tuple=True)
                posterior = model.posterior_routing(sensory, vectors[trial_ids, word_ids])
                probabilities = posterior[trial_ids, torch.arange(len(trial_ids), device=self.device)].cpu().numpy()
                target_words = [tokenize(trial[0]) for trial in sentences]
                for i, position, p in zip(trial_ids.tolist(), word_ids.tolist(), probabilities):
                    row = batch.iloc[i]
                    routing.append(dict(panel=panel, split=row.split, clip=row["clip"],
                                        word=target_words[i][position - 1], position=position - 1,
                                        **dict(zip(ROUTING_COLUMNS, map(float, p)))))
            scores, known = np.array(scores), np.array(known)
            credits = choice_credits(scores, known)
            subsets = {split: rows.split.eq(split) for split in ("test", "s8")}
            subsets["pooled"] = np.ones(len(rows), dtype=bool)
            if panel == "property_object":
                subsets["straddling"] = rows.straddling
            results[panel] = {
                name: dict(accuracy=float(100 * credits[keep].mean()), n_trials=int(keep.sum()),
                           coverage=float(known[keep].mean()))
                for name, keep in subsets.items() if keep.any()
            }
            results[panel]["trials"] = [
                dict(clip=clip, scores=s.tolist() if k else None, credit=float(c))
                for clip, s, k, c in zip(rows["clip"], scores, known, credits)
            ]
        return results, routing

    def held_out(self, model):
        result = {"as_trained": self.word_type_test(model, self.body_mode)}
        if self.condition == "V+G+B":
            result["all_extras_removed"] = self.word_type_test(model, "none", gaze_mode="removed")
        return result


def log_routing(model, study, out_dir):
    rows = []
    for kind, spec in study.word_types.items():
        matched = pd.concat([
            study.evaluate(model, panel, body_mode=study.body_mode, routing=True)
            for panel in dict.fromkeys(spec["held_out"].values())
        ])
        # Each source clip contributes once per word, even with multiple target frames.
        clips = matched.groupby(["word", "clip"])[ROUTING_COLUMNS].mean()
        for word, values in clips.groupby(level="word"):
            rows.append(dict(word=word, kind=kind, n_clips=len(values), **values.mean().to_dict()))
    write_json(out_dir / "routing.json", rows)
    with (out_dir / "routing.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["word", "kind", "n_clips", *ROUTING_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model_args = dict(checkpoint["model_args"])
    if checkpoint.get("run", {}).get("condition") == TEXT_ONLY:
        model = CausalLanguage(**model_args).to(device)
    else:
        # U1 checkpoints predating U2 have 16-word position tables.
        model_args.setdefault("max_words", checkpoint["state_dict"]["language.position.weight"].shape[0] - 2)
        model = UnifiedEmbodiedLearner(**model_args).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def accuracy_summary(results):
    return {mode: {"noun": panels["noun"]["test"]["accuracy"],
                   "action": panels["action"]["pooled"]["accuracy"]}
            for mode, panels in results.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--shuffle-body", action=argparse.BooleanOptionalAction, default=None,
                        help="Override the checkpoint's body mode for evaluation")
    parser.add_argument("--u2", nargs="?", const=U2_FROZEN, type=Path)
    args = parser.parse_args()
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    run = checkpoint.get("run", {})
    training_u2 = Path(run["u2_path"]) if run.get("u2_path") else None
    u2 = args.u2 or training_u2
    if run.get("condition") == TEXT_ONLY:
        vocab = checkpoint["vocab"]
        print(json.dumps(evaluate_vp_swap(model, vocab, u2, args.device), indent=2))
        return
    shuffle_body = args.shuffle_body
    if shuffle_body is None:
        shuffle_body = checkpoint.get("run", {}).get("shuffle_body", False)
    study = UnifiedEvaluation(checkpoint["n_clips"], model.condition, args.device,
                              shuffle_body=shuffle_body, u2_path=training_u2)
    results = study.held_out(model)
    summary = accuracy_summary(results)
    if shuffle_body:
        summary["body_mode"] = "shuffled"
    if u2 is not None:
        panels, routing = study.compositional(model, u2)
        summary.update(compositional=panels, sentence_routing=routing,
                       vp_swap=evaluate_vp_swap(model.language, study.vocab, u2, args.device))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
