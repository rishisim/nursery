from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from babyworld_lite.sim import simulate_episode, render_gif, flatten_for_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BabyWorld-Lite synthetic episodes.")
    parser.add_argument("--n", type=int, default=500, help="Number of episodes to generate")
    parser.add_argument("--out", type=str, default="data/run", help="Output directory")
    parser.add_argument("--seed", type=int, default=123, help="Base RNG seed")
    parser.add_argument("--render-first", type=int, default=12, help="Render GIFs for first K episodes")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "episodes.jsonl"
    manifest_rows = []
    t0 = time.time()
    with jsonl_path.open("w") as f:
        for i in range(args.n):
            ep = simulate_episode(i, args.seed + i * 1009)
            record = ep.to_jsonable()
            if i < args.render_first:
                gif_dir = out / "gifs"
                gif_path = gif_dir / f"episode_{i:05d}.gif"
                render_gif(ep, gif_path)
                record["gif_path"] = str(gif_path.relative_to(out))
            f.write(json.dumps(record) + "\n")
            manifest_rows.append(flatten_for_manifest(ep))
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    elapsed = time.time() - t0
    summary = {
        "episodes": args.n,
        "rendered_gifs": min(args.n, args.render_first),
        "elapsed_seconds": elapsed,
        "episodes_per_second": args.n / max(elapsed, 1e-9),
        "jsonl": str(jsonl_path),
        "manifest": str(out / "manifest.csv"),
    }
    (out / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
