"""Evaluate the frozen Step 9 noun and action panels with one learned gate."""

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
from nursery.learner.acquisition_evaluate import StudyEvaluation, _metrics
from nursery.learner.body import clip_index, load_store, shuffle_store
from nursery.learner.gaze_data import (
    TemporalVideoData,
    collate_frame_indices,
    partition_window_keys,
)
from nursery.learner.train import _pair_meta
from nursery.learner.windows import BodyData

from .model import UnifiedEmbodiedLearner


STUDY = REPO / "outputs/runs/step9_acquisition_curves/frozen"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


class UnifiedEvaluation(StudyEvaluation):
    """Reuse panel construction, query encoding and fixed trial aggregation."""

    def __init__(self, n_clips, condition, device, *, shuffle_body=False):
        if shuffle_body and condition != "V+G+B":
            raise ValueError("Shuffled body requires V+G+B")
        protocol = read_json(STUDY / "protocol.json")
        for fraction in protocol["fractions"]:
            subset = STUDY / "subsets" / str(float(fraction))
            if read_json(subset / "stats.json")["selected_clip_count"] == n_clips:
                break
        else:
            raise ValueError(f"No frozen Step 9 subset with {n_clips} clips")

        inputs, episodes = Path(protocol["inputs"]), Path(protocol["episodes"])
        caches, lut = _upstream_module("train_frame_mil").load_multi([inputs / "train_cache"])
        if caches[0].shape[1:] != (257, 768):
            raise ValueError("U1 requires the cached DINOv2-Base 16x16 patch grid and CLS")

        with np.load(subset / "fit_mask.npz", allow_pickle=False) as fit:
            videos, include = fit["video_ids"].tolist(), fit["include"]
        frames = pd.read_parquet(episodes / "frames.parquet",
                                 columns=["video_id", "frame_idx", "clip", "held_category"])
        store = None
        if condition == "V+G+B":
            store = load_store(episodes, frames, videos, set(), steps=include.shape[1],
                               exclude=~include, physical=protocol["physical"],
                               representation=protocol["arms"]["video_gaze_body_current"]["body_representation"])
        super().__init__(STUDY, inputs, caches, lut, read_json(subset / "vocab.json"), store, device)
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
        rows = pd.read_parquet(self.subset / "manifest.parquet")
        window = self.protocol["window"]
        ds = self.upstream.FramePairs(self.caches, self.lut, rows, self.vocab, window, False)
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
    def evaluate(self, model, panel, body_mode="real", *, gaze_mode=None, body_weights=None):
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
        candidates = np.asarray(trials["candidate_rows"], dtype=int)
        scores = np.zeros(candidates.shape, dtype=np.float32)
        if text is not None:
            matrix = np.empty((len(rows), len(word_index)), dtype=np.float32)
            for start in range(0, len(rows), 64):
                ids = np.arange(start, min(start + 64, len(rows)))
                _, video, _, _ = collate_frame_indices(ds, ids)
                args = self.sensory_batch(model, ds, ids, video, gaze, self.temporal_panels[panel],
                                          body if body_mode != "none" else None,
                                          body_ids=donors[ids] if donors is not None else None)
                matrix[ids] = model.score(model.encode_sensory(*args), text).cpu().numpy()
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

    def held_out(self, model):
        result = {"as_trained": self.word_type_test(model, self.body_mode)}
        if self.condition == "V+G+B":
            result["all_extras_removed"] = self.word_type_test(model, "none", gaze_mode="removed")
        return result


@torch.no_grad()
def log_gate(model, vocab, device, out_dir):
    model.eval()
    entries = [(word, idx) for word, idx in sorted(vocab.items()) if idx != 0]
    tokens = torch.tensor([[idx] for _, idx in entries], device=device)
    text = model.enc_text(tokens, torch.ones_like(tokens[:, 0]))
    active = torch.tensor([True, True, model.condition == "V+G+B"], device=device)
    values = model.gate(text, active).cpu().tolist()
    rows = [dict(word=word, g_vis=g[0], g_temp=g[1], g_body=g[2])
            for (word, _), g in zip(entries, values)]
    write_json(out_dir / "gate.json", rows)
    with (out_dir / "gate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = UnifiedEmbodiedLearner(**checkpoint["model_args"]).to(device)
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
    args = parser.parse_args()
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    shuffle_body = args.shuffle_body
    if shuffle_body is None:
        shuffle_body = checkpoint.get("run", {}).get("shuffle_body", False)
    study = UnifiedEvaluation(checkpoint["n_clips"], model.condition, args.device,
                              shuffle_body=shuffle_body)
    results = study.held_out(model)
    summary = accuracy_summary(results)
    if shuffle_body:
        summary["body_mode"] = "shuffled"
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
