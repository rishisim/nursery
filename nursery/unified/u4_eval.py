"""Evaluate U4 grammar, grounded generation and sensory ablations on frozen U2."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
import torch

from nursery.common import REPO, _output_directory
from nursery.learner import _upstream_module
from nursery.learner.gaze_data import collate_frame_indices

from .baselines import BASELINE_CONDITIONS, BabyLLaVANursery, CLIPGlobal
from .evaluate import (
    TEXT_ONLY, U2_FROZEN, U2_PANELS, UnifiedEvaluation, choice_credits,
    load_checkpoint, sentence_inputs, write_json,
)
from .model import CONDITIONS
from .u2_data import HELD_OUT, MASS_NOUNS, PLURAL_MASS_NOUNS


FAMILIAR = "familiar_frame_novel_pair"
UNSEEN = "unseen_frame"
CATEGORIES = ("agreement", "word_order", "negation_syntax", "verb_argument")
SHUFFLED = "Shuffled body"
ARMS = (*CONDITIONS, SHUFFLED, *BASELINE_CONDITIONS, TEXT_ONLY)
EMBEDDING_ARMS = ("V+G+B", "V", CLIPGlobal.condition)
ACTIONS = {
    "drink": (("drink",),), "lift": (("lift",),), "pass": (("pass",),),
    "inspect": (("inspect",), ("look",)), "offhand": (("switch", "hands"),),
}


def build_blimp(path=U2_FROZEN):
    """Generate covered pairs and classify the grammatical frame by corpus attestation."""
    upstream = _upstream_module("train")
    rows = pd.read_parquet(path / "manifest.parquet")
    assert rows["clip"].nunique() == 630
    vocab = upstream.build_vocab(rows.text, min_freq=1)
    # Exclude exact token sequences even when they occur inside longer utterances.
    utterances = {tuple(upstream.tokenize(text)) for text in rows.text}
    fragments = {tokens[i:j] for tokens in utterances
                 for i in range(len(tokens)) for j in range(i + 1, len(tokens) + 1)}
    nouns = set(rows.loc[rows.named, "referent"])
    masses = set(MASS_NOUNS)
    mass_nouns = sorted(nouns.intersection(set().union(*MASS_NOUNS.values())))

    def frame(tokens):
        return tuple("{noun}" if word in nouns else "{mass}" if word in masses else word
                     for word in tokens)

    frames = set(map(frame, fragments))
    pairs = []
    seen = set()

    def add(category, good, bad, *, allow_unseen=False):
        sequences = [tuple(upstream.tokenize(text)) for text in (good, bad)]
        if any(seq in fragments or not set(seq) <= vocab.keys() for seq in sequences):
            return
        familiar = frame(sequences[0]) in frames
        if not familiar and not allow_unseen:
            return
        key = tuple(sequences)
        if key not in seen:
            seen.add(key)
            pairs.append(dict(category=category, tier=FAMILIAR if familiar else UNSEEN,
                              good=good, bad=bad))

    for noun in mass_nouns:
        plural = noun in PLURAL_MASS_NOUNS
        verb, mismatch = ("are", "is") if plural else ("is", "are")
        determiners = ("these", "those") if plural else ("this", "that")
        for mass in sorted(masses):
            for det in (*determiners, "the"):
                add("agreement", f"{det} {noun} {verb} {mass}",
                    f"{det} {noun} {mismatch} {mass}", allow_unseen=det == "the")
            # "Light the bulb" and "the bulb light" can both be grammatical.
            if mass == "heavy":
                add("word_order", f"the {mass} {noun}", f"{mass} the {noun}")
                add("word_order", f"the {mass} {noun}", f"the {noun} {mass}")

    # Use contrastive noun combinations that are actually named together in speech.
    negations = rows.loc[rows.negated_noun.notna(), ["referent", "negated_noun"]]
    for noun, other in sorted(set(negations.itertuples(index=False, name=None))):
        if noun not in mass_nouns or other not in mass_nouns or noun == other:
            continue
        for start in ("", "holding "):
            add("negation_syntax", f"{start}the {noun}, not the {other}",
                f"{start}the {noun}, the not {other}", allow_unseen=bool(start))

    for noun in sorted(nouns):
        # Restrict drinking to containers so the foil tests syntax rather than affordance.
        if noun in {"bottle", "bowl", "bucket", "glass", "mug", "teapot"}:
            add("verb_argument", f"drink from the {noun}", f"drink at the {noun}")
        add("verb_argument", f"look at the {noun}", f"look the {noun} at")
        for verb in ("lift", "pass"):
            add("verb_argument", f"{verb} the {noun}", f"{verb} {noun} the")

    test = [tuple(upstream.tokenize(pair[side])) for pair in pairs for side in ("good", "bad")]
    assert {p["category"] for p in pairs} == set(CATEGORIES)
    assert {word for seq in test for word in seq} <= vocab.keys()
    assert fragments.isdisjoint(test)
    return pairs, vocab


@torch.no_grad()
def score_blimp(language, pairs, vocab, device):
    language.eval()
    scores = []
    for start in range(0, len(pairs), 128):
        batch = pairs[start:start + 128]
        words, counts, known = sentence_inputs(
            [pair[side] for pair in batch for side in ("good", "bad")], vocab, device)
        assert known.all()
        scores.extend(language.sentence_logprob(words, counts).reshape(-1, 2).cpu().tolist())
    credits = choice_credits(scores, [True] * len(pairs))
    trials = [dict(pair, scores=score, credit=float(credit))
              for pair, score, credit in zip(pairs, scores, credits)]
    return dict(overall=100 * mean(credits),
                by_category={c: 100 * mean(t["credit"] for t in trials if t["category"] == c)
                             for c in CATEGORIES},
                by_tier={tier: 100 * mean(t["credit"] for t in trials if t["tier"] == tier)
                         for tier in (FAMILIAR, UNSEEN)},
                trials=trials)


def sensory_prefix(model, sensory):
    """Extract the prefix and mask supplied to the LM during training."""
    if isinstance(model, BabyLLaVANursery):
        return sensory
    if isinstance(model, CLIPGlobal):
        active = torch.ones_like(sensory[:, :1], dtype=torch.bool)
        return sensory[:, None], active
    return sensory[:2]


def lm_logits(language, inputs, prefix=None, *, embedded=False):
    if prefix is None:
        hidden = language._hidden(inputs)
    elif embedded:
        hidden = language.hidden_with_prefix(inputs, *prefix)
    else:
        hidden = language._hidden(inputs, *prefix)
    return language.lm_head(hidden)


@torch.no_grad()
def greedy_decode(model, sensory, *, max_len=24):
    """Decode up to max_len words, stopping each sample at its first EOS."""
    model.eval()
    language = model.language
    prefix = sensory_prefix(model, sensory)
    tokens = torch.full((len(prefix[0]), 1), language.bos,
                        dtype=torch.long, device=prefix[0].device)
    done = torch.zeros(len(tokens), dtype=torch.bool, device=tokens.device)
    for _ in range(min(max_len, language.position.num_embeddings)):
        logits = lm_logits(language, tokens, prefix, embedded=isinstance(model, BabyLLaVANursery))
        next_logits = logits[:, -1]
        next_logits[:, [0, language.bos]] = -torch.inf
        next_word = next_logits.argmax(-1).masked_fill(done, language.eos)
        tokens = torch.cat((tokens, next_word[:, None]), dim=1)
        done |= next_word.eq(language.eos)
        if done.all():
            break
    captions = []
    for words in tokens[:, 1:].tolist():
        stop = words.index(language.eos) if language.eos in words else len(words)
        captions.append(words[:stop])
    return captions


def perplexity_batch(language, sentences, vocab, device, prefix=None, *, embedded=False):
    words, counts, known = sentence_inputs(sentences, vocab, device)
    if not known.all():
        raise ValueError("A U2 perplexity reference has out-of-vocabulary tokens")
    tokens = language._tokens(words, counts)
    logits = lm_logits(language, tokens[:, :-1], prefix, embedded=embedded)
    targets = tokens[:, 1:]
    nll = -logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    return nll.masked_fill(targets.eq(0), 0).sum().item(), int((counts + 1).sum())


def lexical_predictions(words, nouns):
    actions = set()
    for label, forms in ACTIONS.items():
        if any(words[i:i + len(form)] == list(form) for form in forms for i in range(len(words))):
            actions.add(label)
    return dict(object=set(words) & nouns, action=actions, property=set(words) & {"heavy", "light"})


@torch.no_grad()
def evaluate_generation(model, vocab, u2_path, device, nouns, study=None):
    """Use the grammatical, scene-matched U2 choice as the held-out reference."""
    model.eval()
    language = model if study is None else model.language
    inverse_vocab = {idx: word for word, idx in vocab.items()}
    outcomes = defaultdict(Counter)
    panels = {}
    batch_size = 8 if study is not None else 128
    for panel in U2_PANELS:
        rows = pd.read_parquet(u2_path / f"{panel}.parquet")
        rows = rows.loc[rows.split.isin(HELD_OUT)].reset_index(drop=True)
        panel_nll, panel_tokens = 0., 0
        samples = []
        sampled = Counter()
        if study is not None:
            ds, gaze, temporal, body, donors = study._compositional_inputs(
                rows, panel, study.body_mode, None)
        for start in range(0, len(rows), batch_size):
            ids = np.arange(start, min(start + batch_size, len(rows)))
            batch = rows.iloc[ids]
            references = [choices[0] for choices in batch.choices]
            prefix = None
            if study is not None:
                _, video, _, _ = collate_frame_indices(ds, ids)
                sensory = model.encode_sensory(*study.sensory_batch(
                    model, ds, ids, video, gaze, temporal, body,
                    body_ids=donors[ids] if donors is not None else None))
                prefix = sensory_prefix(model, sensory)
                for row, caption, reference in zip(batch.itertuples(), greedy_decode(model, sensory),
                                                   references):
                    words = [inverse_vocab[token] for token in caption]
                    predicted = lexical_predictions(words, nouns)
                    gold = {"object": row.noun}
                    if panel == "action_object":
                        gold["action"] = row.action
                    if panel == "property_object":
                        gold["property"] = row.mass
                    for kind, target in gold.items():
                        guesses = predicted[kind]
                        hit = int(target in guesses)
                        outcomes[kind].update(tp=hit, fp=len(guesses) - hit, fn=1 - hit)
                    if sampled[row.split] < 2:
                        samples.append(dict(clip=row.clip, split=row.split, caption=" ".join(words),
                                            reference=reference, gold=gold))
                        sampled[row.split] += 1
            nll, tokens = perplexity_batch(language, references, vocab, device, prefix,
                                           embedded=isinstance(model, BabyLLaVANursery))
            panel_nll += nll
            panel_tokens += tokens
        panels[panel] = dict(nll=panel_nll, tokens=panel_tokens,
                             perplexity=math.exp(panel_nll / panel_tokens))
        if study is not None:
            panels[panel]["captions"] = samples

    result = dict(reference="Frozen U2 scene-matched choices[0]", panels=panels,
                  perplexity=math.exp(sum(p["nll"] for p in panels.values()) /
                                      sum(p["tokens"] for p in panels.values())))
    if study is not None:
        result["lexical"] = {}
        for kind, counts in outcomes.items():
            tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
            result["lexical"][kind] = dict(tp=tp, fp=fp, fn=fn,
                                           precision=tp / max(tp + fp, 1),
                                           recall=tp / (tp + fn),
                                           f1=2 * tp / (2 * tp + fp + fn))
    return result


@torch.no_grad()
def action_embeddings(model, study):
    """Export mean embeddings over unique U1 held-out action target moments per clip."""
    clips = {}
    for split, panel in study.word_types["action"]["held_out"].items():
        rows, trials, ds, body, _ = study._panel(panel)
        if panel not in study.temporal_panels and not isinstance(model, CLIPGlobal):
            study.temporal_panels[panel] = study.temporal_data(rows)
        gaze = None
        if study.condition == "V+G+B":
            if panel not in study.gaze_panels:
                study.gaze_panels[panel] = study.gaze_data(rows, frame_keys=study.panel_frame_keys[panel])
            gaze = study.gaze_panels[panel]
        ids = np.unique(trials["trial_rows"])
        ids = ids[~rows.iloc[ids].duplicated(["clip", "video_id", "frame_idx"]).to_numpy()]
        for start in range(0, len(ids), 8):
            batch = ids[start:start + 8]
            _, video, _, _ = collate_frame_indices(ds, batch)
            sensory = model.encode_sensory(*study.sensory_batch(
                model, ds, batch, video, gaze, study.temporal_panels.get(panel), body))
            tokens, active = sensory_prefix(model, sensory)
            pooled = tokens.masked_fill(~active[..., None], 0).sum(1) / active.sum(1, keepdim=True)
            for row, z, summary in zip(rows.iloc[batch].itertuples(), tokens.cpu().numpy(),
                                       pooled.cpu().numpy()):
                key = (split, row.clip)
                if key not in clips:
                    clips[key] = (row.referent, [], [], [])
                action, visual, motion, summaries = clips[key]
                if action != row.referent:
                    raise ValueError(f"Conflicting U1 action labels for {row.clip}")
                visual.append(z[0])
                if not isinstance(model, CLIPGlobal):
                    motion.append(z[1])
                summaries.append(summary)
    return [dict(split=split, clip=clip, action=action, moments=len(visual),
                 z_vis=np.mean(visual, axis=0).tolist(),
                 z_temp=np.mean(motion, axis=0).tolist() if motion else None,
                 pooled_sensory=np.mean(summaries, axis=0).tolist())
            for (split, clip), (action, visual, motion, summaries) in sorted(clips.items())]


def evaluate_checkpoint(path, requested_arm, u2_path, pairs, vocab, nouns, device, *,
                        embeddings_only=False):
    if path.is_dir():
        path = path / ("last.pt" if requested_arm == TEXT_ONLY or
                       not (path / "best.pt").exists() else "best.pt")
    model, checkpoint = load_checkpoint(path, device)
    run = checkpoint["run"]
    arm = SHUFFLED if run.get("shuffle_body") else run["condition"]
    if requested_arm is not None and arm != requested_arm:
        raise ValueError(f"Requested {requested_arm}; {path} contains {arm}")
    n_clips = checkpoint["n_clips"]
    if not run.get("u2_path") or (not embeddings_only and n_clips != 630):
        raise ValueError("U4 scores require a 630-clip U2-trained checkpoint")
    if embeddings_only and arm not in EMBEDDING_ARMS:
        raise ValueError(f"Action embeddings are unavailable for {arm}")
    study = None
    if arm == TEXT_ONLY:
        if checkpoint["vocab"] != vocab:
            raise ValueError("Text-only checkpoint vocabulary differs from frozen U2")
        language = model
    else:
        study = UnifiedEvaluation(n_clips, model.condition, device, shuffle_body=arm == SHUFFLED,
                                  u2_path=u2_path)
        if model.language.bos != len(study.vocab) or (not embeddings_only and study.vocab != vocab):
            raise ValueError("Sensory checkpoint vocabulary differs from frozen U2")
        language = model.language

    result = dict(arm=arm, seed=run["seed"], epoch=checkpoint["epoch"], checkpoint=str(path),
                  n_clips=n_clips)
    if arm in EMBEDDING_ARMS:
        result["action_embeddings"] = action_embeddings(model, study)
    if embeddings_only:
        return result
    result.update(blimp=score_blimp(language, pairs, vocab, device),
                  generation=evaluate_generation(model, vocab, u2_path, device, nouns, study))
    if study is not None:
        result["ablations"] = {}
        modes = [("as_trained", study.body_mode, None)]
        if arm == "V+G+B":
            modes += [("body_removed", "none", None), ("all_extras_removed", "none", "removed")]
        for mode, body_mode, gaze_mode in modes:
            panels = study.word_type_test(model, body_mode, gaze_mode=gaze_mode)
            compositional, _ = study.compositional(model, u2_path, body_mode=body_mode, gaze_mode=gaze_mode)
            result["ablations"][mode] = {**panels, **compositional}
    return result


def summarize(results):
    def stats(values):
        return dict(mean=mean(values), sd=stdev(values) if len(values) > 1 else None)

    def transfer_stats(runs, mode, panel, split):
        return {metric: stats([r["ablations"][mode][panel][split][metric] for r in runs])
                for metric in ("accuracy", "coverage")}

    summary = {}
    for arm in sorted({r["arm"] for r in results}):
        runs = sorted((r for r in results if r["arm"] == arm), key=lambda r: r["seed"])
        grammar = [r["blimp"] for r in runs]
        summary[arm] = dict(
            seeds=[r["seed"] for r in runs],
            blimp=dict(overall=stats([g["overall"] for g in grammar]),
                       by_category={name: stats([g["by_category"][name] for g in grammar])
                                    for name in CATEGORIES},
                       by_tier={name: stats([g["by_tier"][name] for g in grammar])
                                for name in (FAMILIAR, UNSEEN)}),
            held_out_perplexity=stats([r["generation"]["perplexity"] for r in runs]),
        )
        if "lexical" in runs[0]["generation"]:
            summary[arm]["lexical"] = {
                kind: {metric: stats([r["generation"]["lexical"][kind][metric] for r in runs])
                       for metric in ("precision", "recall", "f1")}
                for kind in runs[0]["generation"]["lexical"]
            }
        if "ablations" in runs[0]:
            summary[arm]["ablations"] = {
                mode: {panel: {split: transfer_stats(runs, mode, panel, split)
                               for split in splits if split != "trials"}
                       for panel, splits in panels.items()}
                for mode, panels in runs[0]["ablations"].items()
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path, help="Checkpoint files or run directories")
    parser.add_argument("--arm", choices=ARMS, help="Require this arm for every input checkpoint")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--u2", type=Path, default=U2_FROZEN)
    parser.add_argument("--embedding-checkpoint", type=Path, action="append", default=[],
                        help="Additional U2 checkpoint to export action embeddings only (repeatable)")
    parser.add_argument("--text-only-summary", type=Path, help="Optional U4 text-only reference summary")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.out_dir.resolve().is_relative_to(REPO / "outputs"):
        parser.error("--out-dir must be under outputs/")
    pairs, vocab = build_blimp(args.u2)
    manifest = pd.read_parquet(args.u2 / "manifest.parquet", columns=["named", "referent"])
    nouns = set(manifest.loc[manifest.named, "referent"])
    for panel in U2_PANELS:
        rows = pd.read_parquet(args.u2 / f"{panel}.parquet", columns=["noun", "noun_foil"])
        nouns.update(rows.to_numpy().ravel())
    out = args.out_dir.resolve()
    if (REPO / ".git").exists():
        out = _output_directory(out)
    else:
        # Juno runs from a synced tree under the same outputs/ restriction.
        out.mkdir(parents=True, exist_ok=True)
    totals = []
    seen = set()
    for paths, embeddings_only in ((args.checkpoints, False), (args.embedding_checkpoint, True)):
        for path in paths:
            print(f"Evaluating {path}", flush=True)
            result = evaluate_checkpoint(path, args.arm, args.u2, pairs, vocab, nouns, args.device,
                                         embeddings_only=embeddings_only)
            arm, n_clips, seed = (result[k] for k in ("arm", "n_clips", "seed"))
            key = (arm, n_clips, seed)
            if key in seen:
                raise ValueError(f"Duplicate arm, clip count and seed for {path}")
            seen.add(key)
            stem = f"{arm.replace(' ', '_')}_seed_{seed}"
            if "action_embeddings" in result:
                embeddings = f"{stem}_n{n_clips}_action_embeddings.json"
                metadata = {k: result[k] for k in ("arm", "n_clips", "seed", "epoch", "checkpoint")}
                write_json(out / embeddings, dict(metadata, action_embeddings=result.pop("action_embeddings")))
                result["action_embeddings_file"] = embeddings
            if embeddings_only:
                stem += f"_n{n_clips}"
            write_json(out / f"{stem}.json", result)
            if not embeddings_only:
                result["blimp"].pop("trials")
                for panels in result.get("ablations", {}).values():
                    for splits in panels.values():
                        splits.pop("trials", None)
                        for split, metrics in splits.items():
                            splits[split] = {k: metrics[k] for k in ("accuracy", "coverage")}
                totals.append(result)
    counts = Counter((p["category"], p["tier"]) for p in pairs)
    pair_counts = {c: {tier: counts[c, tier] for tier in (FAMILIAR, UNSEEN)} for c in CATEGORIES}
    summary = dict(pair_counts=pair_counts, arms=summarize(totals))
    if args.text_only_summary is not None and args.text_only_summary.is_file():
        reference = json.loads(args.text_only_summary.read_text())
        if reference["pair_counts"] != pair_counts:
            raise ValueError("Text-only reference pair counts differ from frozen U2")
        summary["arms"][TEXT_ONLY] = reference["arms"][TEXT_ONLY]
    write_json(out / "summary.json", summary)


if __name__ == "__main__":
    main()
