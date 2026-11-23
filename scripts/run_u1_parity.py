"""Run the six U1 parity and ablation arms on the frozen 630-clip subset."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
ROOT = Path("outputs/runs/unified_u1_parity")
ARMS = (
    ("V", "V", ()),
    ("V+G", "V+G", ()),
    ("V+G+B", "V+G+B", ()),
    ("V+G+B_shuffled_body", "V+G+B", ("--shuffle-body",)),
    ("V+G+B_lambda_gen_0", "V+G+B", ("--lambda-gen", "0.0")),
    ("V+G+B_no_mhsa", "V+G+B", ("--no-mhsa",)),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for arm, condition, flags in ARMS:
        for seed in range(3):
            out_dir = ROOT / arm / f"seed{seed}"
            # Training publishes final metrics only after checkpoint verification and evaluation.
            if (REPO / out_dir / "metrics.json").is_file():
                print(f"Skipping completed run {out_dir}", flush=True)
                continue
            command = [sys.executable, "-B", "-m", "nursery.unified.train", "--condition", condition,
                       "--n-clips", "630", "--seed", str(seed), "--epochs", "60",
                       "--device", args.device, "--out-dir", str(out_dir), *flags]
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                if (REPO / out_dir).exists():
                    parser.exit(1, f"Incomplete run at {out_dir}. Move it aside before retrying.\n")
                subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
