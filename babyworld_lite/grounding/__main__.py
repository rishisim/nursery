from __future__ import annotations

import argparse
import json

from babyworld_lite.grounding.pipeline import generate_grounding_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the calibration-ready grounding corpus")
    parser.add_argument("--config", default="configs/grounding_provisional.yaml")
    parser.add_argument("--out", default="data/grounding_provisional")
    args = parser.parse_args()
    print(json.dumps(generate_grounding_dataset(args.config, args.out), indent=2))


if __name__ == "__main__":
    main()
