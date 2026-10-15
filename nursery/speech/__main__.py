"""Generate caregiver transcripts and cue weights for rendered episodes."""

import argparse

from .transcripts import write_transcripts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    transcripts = commands.add_parser("transcripts")
    transcripts.add_argument("--episodes", required=True)
    transcripts.add_argument("--output", required=True)
    transcripts.add_argument("--seed", type=int, required=True)
    transcripts.add_argument("--hold-window-s", type=int)
    for flag in ("p-ref", "p-name", "p-held", "p-unseen", "sigma", "rate-hz"):
        transcripts.add_argument(f"--{flag}", type=float)
    args = parser.parse_args()
    write_transcripts(args.episodes, args.output, args.seed,
                      **{k: getattr(args, k) for k in ("p_ref", "p_name", "p_held", "p_unseen", "hold_window_s", "sigma", "rate_hz")})


if __name__ == "__main__":
    main()
