from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from babyworld_lite.grounding.pilot_checkpoint import save_grounding_checkpoint
from babyworld_lite.grounding.pilot_data import (
    GroundingCorpus,
    WordTokenizer,
    held_out_composition_indices,
    load_pilot_source,
    preassigned_split_indices,
)
from babyworld_lite.grounding.pilot_experiment import (
    PilotConfig,
    resolve_device,
    train_one_arm,
)


def _composition(value: str) -> tuple[str, str]:
    try:
        shape, action = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("composition must be SHAPE:ACTION") from exc
    return shape, action


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one Nursery learner and save an evaluation checkpoint."
    )
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--arm", default="synchronized", choices=("synchronized", "shuffled", "null", "time_shifted"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--alignment", choices=("strong", "weak", "shuffled"), default="weak")
    parser.add_argument("--holdout", action="append", type=_composition, dest="holdouts")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--frame-count", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--max-text-length", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--motor-weight", type=float, default=0.25)
    parser.add_argument("--time-shift", type=int, default=2)
    args = parser.parse_args()

    config = PilotConfig(
        frame_count=args.frame_count,
        image_size=args.image_size,
        max_text_length=args.max_text_length,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        motor_weight=args.motor_weight,
        time_shift=args.time_shift,
    )
    records, adapter, source_schema = load_pilot_source(
        args.episodes, alignment_condition=args.alignment
    )
    if source_schema == "grounding-v0" and args.holdouts is None:
        train_indices, test_indices = preassigned_split_indices(records)
    else:
        train_indices, test_indices = held_out_composition_indices(
            records, adapter, args.holdouts or (("cup", "push"), ("plush", "grasp"))
        )
    corpus = GroundingCorpus(records, adapter, config.frame_count, config.image_size)
    tokenizer = WordTokenizer.fit([corpus.text(index) for index in train_indices])
    device = resolve_device(args.device)
    model, training_protocol, _ = train_one_arm(
        corpus, train_indices, tokenizer, args.arm, args.seed, config, device
    )
    checkpoint = save_grounding_checkpoint(
        args.out,
        model,
        tokenizer,
        asdict(config),
        {
            "arm": args.arm,
            "seed": args.seed,
            "alignment_condition": args.alignment,
            "source_schema": source_schema,
            "episodes_path": str(args.episodes),
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "training_protocol": training_protocol,
        },
    )
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "device": str(device),
        "vocabulary_size": len(tokenizer),
        "n_train": len(train_indices),
        "n_test": len(test_indices),
    }, indent=2))


if __name__ == "__main__":
    main()
