"""Train a unified learner on a frozen Step 9 subset."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

from nursery.common import _output_directory
from nursery.learner.gaze_data import collate_frame_indices

from .evaluate import UnifiedEvaluation, accuracy_summary, load_checkpoint, log_routing, write_json
from .model import CONDITIONS, UnifiedEmbodiedLearner


def train(args):
    started = time.perf_counter()
    if args.epochs < 1:
        raise ValueError("Expected at least one epoch")
    if args.shuffle_body and args.condition != "V+G+B":
        raise ValueError("Shuffled body requires V+G+B")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("The requested MPS device is unavailable")
    out_dir = _output_directory(args.out_dir, exist_ok=False)
    study = UnifiedEvaluation(args.n_clips, args.condition, args.device, shuffle_body=args.shuffle_body)
    ds, gaze, temporal, body = study.training_data(seed=args.seed)
    run = dict(name="V+G+B_shuffled_body" if args.shuffle_body else args.condition,
               condition=args.condition, shuffle_body=args.shuffle_body,
               training_body_shuffle=study.shuffle_info, seed=args.seed, n_clips=args.n_clips,
               epochs=args.epochs, lambda_gen=args.lambda_gen, aux_weight=args.aux_weight,
               tau_m=args.tau_m)

    torch.manual_seed(args.seed)
    order_rng = torch.Generator().manual_seed(args.seed)
    gaze_rng = torch.Generator().manual_seed(args.seed + 1000)
    body_rng = torch.Generator().manual_seed(args.seed + 2000)
    model_args = dict(vocab_size=len(study.vocab), body_dims=study.body_dims,
                      condition=args.condition, lambda_gen=args.lambda_gen,
                      aux_weight=args.aux_weight, tau_m=args.tau_m)
    model = UnifiedEmbodiedLearner(**model_args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=study.protocol["lr"],
                                  weight_decay=study.protocol["weight_decay"])
    batches = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(range(len(ds)), generator=order_rng),
        batch_size=study.protocol["batch"], drop_last=False,
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
    metrics = out_dir / "metrics.json"
    pending = metrics.with_suffix(".tmp")
    write_json(pending, dict(run=run, summary=summary, held_out=results))
    pending.replace(metrics)
    return dict(summary, run=run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--n-clips", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--aux-weight", type=float, default=0.25)
    parser.add_argument("--lambda-gen", type=float, default=0.2)
    parser.add_argument("--tau-m", type=float, default=0.1)
    parser.add_argument("--shuffle-body", action="store_true", help="Train and evaluate with shuffled body")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    print(json.dumps(train(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
