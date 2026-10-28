"""Fixed noun-query panels for acquisition curves, including unseen words."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import _upstream_module
from .interaction import centre_clip_valid
from .train import _pair_meta, _window_layout
from .windows import BodyData


def _metrics(scores, known, candidates, correct, queries):
    """Keep every frozen trial; absent vocabulary receives expected chance credit."""
    known = np.asarray(known, dtype=bool)
    correct = np.asarray(correct, dtype=int)
    predictions = np.asarray(scores).argmax(1)
    credits = np.where(known, predictions == correct, .25).astype(float)
    return {"accuracy": float(100 * credits.mean()),
            "known_accuracy": float(100 * credits[known].mean()) if known.any() else None,
            "coverage": float(known.mean()), "n_trials": len(credits),
            "n_known": int(known.sum()), "credits": credits.tolist(),
            "queries": list(queries), "candidate_rows": np.asarray(candidates).tolist(),
            "correct_slots": correct.tolist(),
            "predicted_slots": [int(p) if k else None for p, k in zip(predictions, known)],
            "scores": [s.tolist() if k else None for s, k in zip(np.asarray(scores), known)],
            "unknown_rule": "25% expected accuracy; retained in fixed denominator"}


class StudyEvaluation:
    """Scores the saved panels without filtering candidates by training vocabulary."""

    def __init__(self, root, inputs, caches, lut, vocab, store, device):
        self.root, self.inputs = Path(root), Path(inputs)
        self.caches, self.lut, self.vocab, self.store, self.device = caches, lut, vocab, store, device
        self.upstream = _upstream_module("train_frame_mil")
        self.protocol = json.loads((self.root / "protocol.json").read_text())
        self.panels, self.pictures = {}, {}

    def _query_vectors(self, model, queries):
        words = sorted(set(queries))
        tokens = {word: self.upstream.encode(word, self.vocab) for word in words}
        known_words = [word for word in words if tokens[word]]
        if not known_words:
            return {}, None
        counts = torch.tensor([len(tokens[w]) for w in known_words], device=self.device)
        text = torch.zeros(len(known_words), int(counts.max()), dtype=torch.long, device=self.device)
        for i, word in enumerate(known_words):
            text[i, :len(tokens[word])] = torch.tensor(tokens[word], device=self.device)
        return {w: i for i, w in enumerate(known_words)}, model.enc_text(text, counts)

    def _panel(self, name):
        if name not in self.panels:
            path = self.root / "panels" / name
            rows = pd.read_parquet(path / "manifest.parquet")
            trials = json.loads((path / "trials.json").read_text())
            # FramePairs needs a known token to retain every candidate image. The
            # placeholder is never used as the evaluation query or a model input.
            images = rows.copy()
            images["text"] = next(w for w in self.vocab if self.upstream.encode(w, self.vocab))
            ds = self.upstream.FramePairs(self.caches, self.lut, images, self.vocab, 2, False)
            if len(ds) != len(rows) or ds.n_noframe:
                raise ValueError("Fixed candidate panel lost rows")
            _, valid = _window_layout(rows, self.lut, 2)
            data = None
            if self.store is not None:
                ps, pt, win = _pair_meta(rows, self.store, self.lut, 2)
                data = BodyData(self.store, ps, pt, win, None, None, None, None,
                                self.device, 5, 0., 0.)
            self.panels[name] = (rows, trials, ds, data, valid)
        return self.panels[name]

    @torch.no_grad()
    def evaluate(self, model, panel, body_mode="real"):
        if body_mode not in ("real", "shuffled", "none"):
            raise ValueError("Unknown body evaluation mode")
        model.eval()
        rows, trials, ds, data, valid = self._panel(panel)
        queries = rows.iloc[trials["trial_rows"]].referent.tolist()
        word_index, T = self._query_vectors(model, queries)
        candidates = np.asarray(trials["candidate_rows"], dtype=int)
        scores = np.zeros((len(queries), 4), dtype=np.float32)
        if T is not None:
            matrix = np.empty((len(rows), len(word_index)), dtype=np.float32)
            for start in range(0, len(rows), 128):
                ids = np.arange(start, min(start + 128, len(rows)))
                _, v, _, _ = self.upstream.collate([ds[int(i)] for i in ids])
                R = model.enc_regions(v.to(self.device))
                ctx = None
                if model.encoder is not None and body_mode != "none":
                    if data is None:
                        raise ValueError("Body evaluation needs sensory histories")
                    donors = np.asarray(trials["donor_rows"])[ids] if body_mode == "shuffled" else ids
                    ctx = data.batch(donors, None, training=False)
                    ctx["keep"] *= centre_clip_valid(ctx).any(1)
                    ctx["visual_valid"] = torch.as_tensor(valid[ids], device=self.device).repeat_interleave(17, dim=1)
                matrix[ids] = model.score_encoded(R, T, ctx).float().cpu().numpy()
            for i, word in enumerate(queries):
                if word in word_index:
                    scores[i] = matrix[candidates[i], word_index[word]]
        result = _metrics(scores, [q in word_index for q in queries], candidates, trials["correct_slots"], queries)
        result["body_mode"] = body_mode if model.encoder is not None else "none"
        if body_mode == "shuffled":
            result["donor_rows"] = trials["donor_rows"]
        return result

    @torch.no_grad()
    def evaluate_picture(self, model, panel):
        model.eval()
        if panel not in self.pictures:
            spec = self.protocol["picture_panels"][panel]
            # Panels are resolved relative to the experiment so the root can move.
            trials = json.loads((self.root / "panels" / panel / "trials.json").read_text())["trials"]
            cache = np.load(Path(spec["cache"]) / "emb.f16.npy", mmap_mode="r")
            self.pictures[panel] = (trials, cache)
        trials, cache = self.pictures[panel]
        queries = [t["query"] for t in trials]
        candidates = np.asarray([t["candidate_rows"] for t in trials], dtype=int)
        word_index, T = self._query_vectors(model, queries)
        scores = np.zeros((len(trials), 4), dtype=np.float32)
        if T is not None:
            images = np.unique(candidates)
            matrix = np.empty((len(images), len(word_index)), dtype=np.float32)
            for start in range(0, len(images), 128):
                ids = images[start:start + 128]
                v = torch.as_tensor(np.array(cache[ids], dtype=np.float32), device=self.device)
                matrix[start:start + len(ids)] = model.score_encoded(model.enc_regions(v), T, None).float().cpu().numpy()
            for i, word in enumerate(queries):
                if word in word_index:
                    scores[i] = matrix[np.searchsorted(images, candidates[i]), word_index[word]]
        return _metrics(scores, [q in word_index for q in queries], candidates,
                        [t["correct_slot"] for t in trials], queries)
