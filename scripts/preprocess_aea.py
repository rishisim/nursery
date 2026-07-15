from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.config import load_aea_config
from babyworld_lite.aea.preprocess import preprocess_aea


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build leakage-audited video/ASR/six-axis-IMU AEA grounding windows."
    )
    parser.add_argument("--config", default="configs/aea_real.yaml")
    parser.add_argument("--out", default="data/aea_processed")
    parser.add_argument(
        "--allow-missing-vrs", action="store_true",
        help="Useful during staged acquisition; skipped sequences are reported and are not findings.",
    )
    args = parser.parse_args()
    result = preprocess_aea(
        load_aea_config(args.config), args.out, allow_missing_vrs=args.allow_missing_vrs
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
