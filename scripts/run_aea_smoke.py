from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.config import load_aea_config
from babyworld_lite.aea.experiment import run_aea_experiment
from babyworld_lite.aea.smoke import build_aea_smoke_fixture
from babyworld_lite.grounding.pilot_experiment import PilotConfig


def main() -> None:
    fixture = build_aea_smoke_fixture("data/aea_smoke_fixture")
    config = load_aea_config("configs/aea_real.yaml")
    config["experiment"]["minimum_test_examples_per_action"] = 2
    config["experiment"]["bootstrap_samples"] = 200
    result = run_aea_experiment(
        fixture,
        config,
        "output/aea_smoke",
        PilotConfig(
            frame_count=3,
            image_size=24,
            max_text_length=10,
            hidden_dim=12,
            embedding_dim=8,
            batch_size=24,
            epochs=1,
            learning_rate=1e-3,
            motor_weight=0.25,
            time_shift=4,
            bootstrap_samples=200,
            motor_sample_count=15,
        ),
        seeds=(11, 22),
        families=("held_out_location",),
        device_name="cpu",
        infrastructure_smoke=True,
        maximum_splits=2,
    )
    print(json.dumps({
        "scientific_status": result["scientific_status"],
        "results": "output/aea_smoke/aea_results.json",
        "report": "output/aea_smoke/aea_report.md",
        "primary_estimand": result["primary_estimand"],
    }, indent=2))


if __name__ == "__main__":
    main()
