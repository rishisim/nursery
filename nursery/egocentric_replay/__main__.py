"""Build or preview egocentric sessions from recorded GRAB motion."""

import argparse

from .build import DEFAULT_CONFIG, build
from .preview import preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--config", default=str(DEFAULT_CONFIG))
    builder.add_argument("--rooms", required=True)
    builder.add_argument("--output", required=True)
    builder.add_argument("--sessions-per-subject", type=int, default=1)
    builder.add_argument("--seed", type=int, default=7)
    builder.add_argument("--session-index", type=int, help="One deterministic session per worker; probes excluded")
    viewer = commands.add_parser("preview")
    viewer.add_argument("--episodes", required=True)
    viewer.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.config, args.rooms, args.output, args.sessions_per_subject, args.seed,
              args.session_index)
    else:
        preview(args.episodes, args.output)


if __name__ == "__main__":
    main()
