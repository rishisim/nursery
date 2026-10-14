"""Generate transcripts, embed episode images, and train a cue learner."""

import argparse

from .embed import embed_episodes
from .train import train_run
from .transcripts import write_transcripts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    transcripts = commands.add_parser("transcripts")
    transcripts.add_argument("--episodes", required=True)
    transcripts.add_argument("--output", required=True)
    transcripts.add_argument("--seed", type=int, required=True)
    for flag in ("p-ref", "q-held", "p-name", "sigma", "rate-hz"):
        transcripts.add_argument(f"--{flag}", type=float)
    embed = commands.add_parser("embed")
    embed.add_argument("--episodes", required=True)
    embed.add_argument("--output", required=True)
    embed.add_argument("--device", help="Torch device (default: cuda > mps > cpu)")
    embed.add_argument("--batch", type=int, default=64)
    train = commands.add_parser("train")
    train.add_argument("--run", required=True)
    train.add_argument("--condition", required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--epochs", type=int)
    args = parser.parse_args()
    if args.command == "transcripts":
        write_transcripts(args.episodes, args.output, args.seed,
                          **{k: getattr(args, k) for k in ("p_ref", "q_held", "p_name", "sigma", "rate_hz")})
    elif args.command == "embed":
        embed_episodes(args.episodes, args.output, args.device, args.batch)
    else:
        train_run(args.run, args.condition, args.seed, args.epochs)


if __name__ == "__main__":
    main()
