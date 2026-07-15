from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.config import load_aea_config
from babyworld_lite.aea.experiment import run_aea_experiment
from babyworld_lite.grounding.pilot_experiment import PilotConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired four-arm AEA grounding evaluations with IMU withheld at test."
    )
    parser.add_argument("--examples", default="data/aea_processed/examples.jsonl")
    parser.add_argument("--config", default="configs/aea_real.yaml")
    parser.add_argument("--out", default="output/aea_real")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--families", nargs="+",
        choices=("held_out_location", "held_out_wearer_session", "held_out_composition"),
        default=["held_out_location", "held_out_wearer_session", "held_out_composition"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--motor-sample-count", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--motor-weight", type=float, default=0.25)
    parser.add_argument("--time-shift", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--infrastructure-smoke", action="store_true")
    parser.add_argument("--maximum-splits", type=int, default=0)
    args = parser.parse_args()
    result = run_aea_experiment(
        args.examples,
        load_aea_config(args.config),
        args.out,
        PilotConfig(
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
            motor_sample_count=args.motor_sample_count,
        ),
        seeds=args.seeds,
        families=args.families,
        device_name=args.device,
        infrastructure_smoke=args.infrastructure_smoke,
        maximum_splits=args.maximum_splits,
    )
    print(json.dumps({
        "results": str(Path(args.out) / "aea_results.json"),
        "report": str(Path(args.out) / "aea_report.md"),
        "scientific_status": result["scientific_status"],
        "primary_estimand": result["primary_estimand"],
    }, indent=2))


if __name__ == "__main__":
    main()
