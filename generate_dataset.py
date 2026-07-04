from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from babyworld_lite.sim import simulate_episode, render_gif, flatten_for_manifest


def _rounded_dict(values: dict[str, Any], digits: int = 3) -> dict[str, Any]:
    return {
        key: round(value, digits) if isinstance(value, (float, int)) else value
        for key, value in values.items()
    }


def _demo_episode(record: dict[str, Any]) -> dict[str, Any]:
    obj = record["object"]
    frames = record["frames"]
    first = frames[0]["object"]
    last = frames[-1]["object"]
    demo = {
        "episode_id": record["episode_id"],
        "language_before": record["utterance_pre"],
        "language_after": record["utterance_post"],
        "action": record["action"],
        "event_label": record["event_label"],
        "hidden_object_state": {
            "shape": obj["shape"],
            "color": obj["color_name"],
            "material": obj["material"],
            "mass": round(obj["mass"], 3),
            "friction": round(obj["friction"], 3),
            "bounciness": round(obj["bounciness"], 3),
            "hardness": round(obj["hardness"], 3),
        },
        "trajectory_summary": {
            "frames": len(frames),
            "object_start": _rounded_dict(first),
            "object_end": _rounded_dict(last),
        },
        "tactile_summary": _rounded_dict(record["tactile_summary"]),
        "counterfactuals": record["counterfactuals"],
    }
    if "gif_path" in record:
        demo["rendered_gif"] = record["gif_path"]
    return demo


def _print_count_table(title: str, values: list[str]) -> None:
    print(f"\n{title}")
    for label, count in sorted(Counter(values).items(), key=lambda item: (-item[1], item[0])):
        print(f"  {label:<18} {count:>6}")


def _print_demo_log(
    summary: dict[str, Any],
    manifest: pd.DataFrame,
    preview_records: list[dict[str, Any]],
    out: Path,
) -> None:
    print("\nBabyWorld-Lite generation demo")
    print("=" * 34)
    print(
        f"Generated {summary['episodes']:,} fully instrumented episodes "
        f"in {summary['elapsed_seconds']:.2f}s "
        f"({summary['episodes_per_second']:.0f} episodes/sec)."
    )
    print("\nArtifacts written")
    print(f"  episodes_jsonl   {summary['jsonl']}")
    print(f"  manifest_csv     {summary['manifest']}")
    print(f"  summary_json     {out / 'generation_summary.json'}")
    if summary["rendered_gifs"]:
        print(f"  rendered_gifs    {out / 'gifs'} (first {summary['rendered_gifs']} episodes)")

    _print_count_table("Coverage by object", manifest["shape"].astype(str).tolist())
    _print_count_table("Coverage by action", manifest["action"].astype(str).tolist())
    _print_count_table("Coverage by material", manifest["material"].astype(str).tolist())
    _print_count_table("Coverage by event label", manifest["event_label"].astype(str).tolist())

    print("\nGenerated episode previews")
    print("Each preview is a compact view of a full JSONL record with frame traces, hidden state, touch, causal labels, and counterfactuals.")
    for record in preview_records:
        print(f"\n--- episode {record['episode_id']:05d} ---")
        print(json.dumps(_demo_episode(record), indent=2))
    print("\nGeneration complete. Stop here for the scale/instrumentation demo.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BabyWorld-Lite synthetic episodes.")
    parser.add_argument("--n", type=int, default=500, help="Number of episodes to generate")
    parser.add_argument("--out", type=str, default="data/run", help="Output directory")
    parser.add_argument("--seed", type=int, default=123, help="Base RNG seed")
    parser.add_argument("--render-first", type=int, default=12, help="Render GIFs for first K episodes")
    parser.add_argument("--demo-log", action="store_true", help="Print a presentation-friendly generation log with coverage tables and episode previews")
    parser.add_argument("--preview-count", type=int, default=12, help="Number of compact episode previews to print with --demo-log")
    parser.add_argument("--progress-every", type=int, default=1000, help="Print generation progress every N episodes with --demo-log; use 0 to disable")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "episodes.jsonl"
    manifest_rows = []
    preview_records = []
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
            if args.demo_log and len(preview_records) < args.preview_count:
                preview_records.append(record)
            if args.demo_log and args.progress_every > 0 and (i + 1) % args.progress_every == 0:
                elapsed = time.time() - t0
                eps = (i + 1) / max(elapsed, 1e-9)
                print(f"generated {i + 1:>6}/{args.n:<6} episodes  ({eps:.0f} episodes/sec)")
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
    if args.demo_log:
        _print_demo_log(summary, manifest, preview_records, out)
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
