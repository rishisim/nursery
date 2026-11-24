"""Train embodied or text-only models on frozen Nursery speech."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

from nursery.common import _output_directory
from nursery.learner import _upstream_module
from nursery.learner.gaze_data import collate_frame_indices

from .evaluate import (
    STUDY, TEXT_ONLY, U2_FROZEN, UnifiedEvaluation, accuracy_summary, evaluate_vp_swap,
    load_checkpoint, log_routing, read_json, sentence_inputs, step9_subset, write_json,
)
from .language import CausalLanguage
from .model import CONDITIONS, UnifiedEmbodiedLearner
from .u2_data import training_rows


def train_text_only(args, out_dir, u2_path):
    protocol = read_json(STUDY / "protocol.json")
    rows = training_rows(u2_path, step9_subset(protocol, args.n_clips))
    vocab = _upstream_module("train").build_vocab(rows.text, min_freq=1)
    words, counts, _ = sentence_inputs(rows.text, vocab, "cpu")
    model_args = dict(vocab_size=len(vocab))
    torch.manual_seed(args.seed)
    model = CausalLanguage(**model_args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["lr"],
                                  weight_decay=protocol["weight_decay"])
    batch_size = getattr(args, "batch_size", None) or protocol["batch"]
    order = torch.Generator().manual_seed(args.seed)
    batches = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(range(len(rows)), generator=order),
        batch_size=batch_size, drop_last=False,
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        token_loss, tokens = 0.0, 0
        for indices in batches:
            x, n = words[indices].to(args.device), counts[indices].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.lm_loss(x, n)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite text-only training loss")
            loss.backward()
            optimizer.step()
            n_tokens = int(n.sum()) + len(indices)
            token_loss += float(loss.detach()) * n_tokens
            tokens += n_tokens
        history.append(dict(epoch=epoch, loss=token_loss / tokens))
        elapsed = time.perf_counter() - epoch_started
        print(f"Epoch {epoch}/{args.epochs} {elapsed:.2f}s loss={history[-1]['loss']:.4f}",
              file=sys.stderr, flush=True)
    write_json(out_dir / "history.json", history)
    run = dict(condition=TEXT_ONLY, seed=args.seed, n_clips=args.n_clips, epochs=args.epochs,
               batch_size=batch_size, utterances=len(rows), u2_path=str(u2_path))
    checkpoint_path = out_dir / "last.pt"
    # The LM baseline uses the configured epoch budget; neither VP split selects it.
    torch.save(dict(state_dict=model.state_dict(), model_args=model_args, vocab=vocab,
                    n_clips=args.n_clips, epoch=args.epochs, run=run), checkpoint_path)
    reloaded, _ = load_checkpoint(checkpoint_path, args.device)
    if any(not torch.equal(value, reloaded.state_dict()[key]) for key, value in model.state_dict().items()):
        raise RuntimeError("Text-only checkpoint reload changed a parameter")
    scores = evaluate_vp_swap(reloaded, vocab, u2_path, args.device)
    summary = dict(epoch=args.epochs, loss=history[-1]["loss"], reload_identical=True,
                   vp_swap={split: {k: v for k, v in metrics.items() if k != "trials"}
                            for split, metrics in scores.items()})
    write_json(out_dir / "metrics.json", dict(run=run, summary=summary, vp_swap=scores))
    return dict(summary, run=run)


def train(args):
    started = time.perf_counter()
    if args.epochs < 1:
        raise ValueError("Expected at least one epoch")
    u2_path = getattr(args, "u2", None)
    bow_only = getattr(args, "bow_only", False)
    if args.condition == TEXT_ONLY and (u2_path is None or bow_only):
        raise ValueError("Text-Only LM requires U2 and has no retrieval ablation")
    if args.shuffle_body and args.condition != "V+G+B":
        raise ValueError("Shuffled body requires V+G+B")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("The requested MPS device is unavailable")
    out_dir = _output_directory(args.out_dir, exist_ok=False)
    if args.condition == TEXT_ONLY:
        return train_text_only(args, out_dir, u2_path)
    study_options = dict(shuffle_body=args.shuffle_body)
    if u2_path is not None:
        study_options["u2_path"] = u2_path
    study = UnifiedEvaluation(args.n_clips, args.condition, args.device, **study_options)
    ds, gaze, temporal, body = study.training_data(seed=args.seed)
    run = dict(name="V+G+B_shuffled_body" if args.shuffle_body else args.condition,
               condition=args.condition, shuffle_body=args.shuffle_body,
               training_body_shuffle=study.shuffle_info, seed=args.seed, n_clips=args.n_clips,
               epochs=args.epochs, lambda_gen=args.lambda_gen, aux_weight=args.aux_weight,
               tau_m=args.tau_m)
    batch_size = getattr(args, "batch_size", None) or study.protocol["batch"]
    if u2_path is not None:
        run.update(u2_path=str(u2_path), batch_size=batch_size, utterances=len(ds))

    torch.manual_seed(args.seed)
    order_rng = torch.Generator().manual_seed(args.seed)
    gaze_rng = torch.Generator().manual_seed(args.seed + 1000)
    body_rng = torch.Generator().manual_seed(args.seed + 2000)
    model_args = dict(vocab_size=len(study.vocab), body_dims=study.body_dims,
                      condition=args.condition, lambda_gen=args.lambda_gen,
                      aux_weight=args.aux_weight, tau_m=args.tau_m, bow_only=bow_only)
    model = UnifiedEmbodiedLearner(**model_args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=study.protocol["lr"],
                                  weight_decay=study.protocol["weight_decay"])
    batches = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(range(len(ds)), generator=order_rng),
        batch_size=batch_size, drop_last=False,
    )

    history = []
    best_score, best_dev = -float("inf"), None
    checkpoint_path = out_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total = 0.0
        for indices in batches:
            ids, video, words, counts = collate_frame_indices(ds, indices)
            sensory = study.sensory_batch(model, ds, ids.numpy(), video, gaze, temporal, body,
                                          gaze_rng=gaze_rng, body_rng=body_rng)
            optimizer.zero_grad(set_to_none=True)
            loss = model(words.to(args.device), counts.to(args.device), *sensory)["loss"]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            del sensory, loss

        dev = study.development(model)
        by_rung = {rung: result["accuracy"] for rung, result in dev.items()}
        score = sum(by_rung.values()) / len(by_rung)
        history.append(dict(epoch=epoch, loss=total / len(ds), dev=by_rung, dev_mean=score))
        write_json(out_dir / "history.json", history)
        if score > best_score:
            best_score, best_dev = score, dev
            torch.save(dict(state_dict=model.state_dict(), model_args=model_args,
                            n_clips=args.n_clips, seed=args.seed, epoch=epoch, run=run), checkpoint_path)
        elapsed = time.perf_counter() - epoch_started
        print(f"Epoch {epoch}/{args.epochs} {elapsed:.2f}s loss={history[-1]['loss']:.4f} "
              f"dev={by_rung}", file=sys.stderr, flush=True)

    model, checkpoint = load_checkpoint(checkpoint_path, args.device)
    if study.development(model) != best_dev:
        raise RuntimeError("Checkpoint reload changed development trial scores")
    results = study.held_out(model)
    log_routing(model, study, out_dir)
    summary = dict(best_epoch=checkpoint["epoch"], dev=history[checkpoint["epoch"] - 1]["dev"],
                   reload_identical=True, wall_seconds=time.perf_counter() - started,
                   accuracy=accuracy_summary(results))
    if u2_path is not None:
        results["compositional"], routing = study.compositional(model, u2_path)
        results["vp_swap"] = evaluate_vp_swap(model.language, study.vocab, u2_path, args.device)
        write_json(out_dir / "sentence_routing.json", routing)
        summary["compositional"] = {
            panel: {k: v for k, v in metrics.items() if k != "trials"}
            for panel, metrics in results["compositional"].items()
        }
        summary["vp_swap"] = {
            split: {k: v for k, v in metrics.items() if k != "trials"}
            for split, metrics in results["vp_swap"].items()
        }
        summary["loss"] = history[-1]["loss"]
        summary["wall_seconds"] = time.perf_counter() - started
    metrics = out_dir / "metrics.json"
    pending = metrics.with_suffix(".tmp")
    write_json(pending, dict(run=run, summary=summary, held_out=results))
    pending.replace(metrics)
    return dict(summary, run=run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=(*CONDITIONS, TEXT_ONLY), required=True)
    parser.add_argument("--n-clips", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--aux-weight", type=float, default=0.25)
    parser.add_argument("--lambda-gen", type=float, default=0.2)
    parser.add_argument("--tau-m", type=float, default=0.1)
    parser.add_argument("--u2", nargs="?", const=U2_FROZEN, type=Path)
    parser.add_argument("--bow-only", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--shuffle-body", action="store_true", help="Train and evaluate with shuffled body")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    print(json.dumps(train(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
