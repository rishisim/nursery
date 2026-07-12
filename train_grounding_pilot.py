from __future__ import annotations

import argparse
import json

from babyworld_lite.grounding.pilot_data import ARM_NAMES
from babyworld_lite.grounding.pilot_experiment import (
    PilotConfig,
    run_pilot,
)


def _composition(value: str) -> tuple[str, str]:
    try:
        shape, action = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("composition must be SHAPE:ACTION") from exc
    return shape, action


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the controlled pixel/text/low-level-motor grounding pilot."
    )
    parser.add_argument("--episodes", required=True, help="Legacy episodes.jsonl or grounding-v0 examples.jsonl")
    parser.add_argument("--out", default="data/grounding_pilot")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--arms", nargs="+", choices=ARM_NAMES, default=list(ARM_NAMES))
    parser.add_argument("--holdout", action="append", type=_composition, dest="holdouts")
    parser.add_argument("--alignment", choices=("strong", "weak", "shuffled"), default="weak")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frame-count", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--motor-weight", type=float, default=0.25)
    parser.add_argument("--time-shift", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    config = PilotConfig(
        frame_count=args.frame_count,
        image_size=args.image_size,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        motor_weight=args.motor_weight,
        time_shift=args.time_shift,
        bootstrap_samples=args.bootstrap_samples,
    )
    result = run_pilot(
        episodes_path=args.episodes,
        out_dir=args.out,
        seeds=args.seeds,
        arms=args.arms,
        heldouts=args.holdouts,
        max_episodes=args.max_episodes,
        alignment_condition=args.alignment,
        device_name=args.device,
        config=config,
    )
    print(json.dumps({
        "results": f"{args.out}/pilot_results.json",
        "device": result["device"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "aggregate": result["aggregate"],
        "paired_lifts": result["paired_lifts"],
        "claim_gate": result["claim_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
