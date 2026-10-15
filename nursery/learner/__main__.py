"""Embed episode images and train the cue learner on a speech run."""

import argparse

from .embed import embed_episodes
from .train import train_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
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
    train.add_argument("--body", help="Body signals for training-only region guidance: all, geometry, or a comma list of contact,hand,joints")
    train.add_argument("--body-control", choices=("none", "shuffle", "oracle"), default="none")
    train.add_argument("--body-drop", type=float, default=0.5, help="Per-pair probability of dropping the whole body")
    train.add_argument("--pool", choices=("max", "soft"), default="max", help="Region pooling in the training loss; --body requires soft")
    train.add_argument("--tau-p", type=float, default=0.1, help="Soft-pooling temperature")
    train.add_argument("--name", help="Output directory prefix (default: the condition)")
    args = parser.parse_args()
    if args.command == "embed":
        embed_episodes(args.episodes, args.output, args.device, args.batch)
    else:
        train_run(args.run, args.condition, args.seed, args.epochs, args.body, args.body_control, args.body_drop, args.name, args.pool, args.tau_p)


if __name__ == "__main__":
    main()
