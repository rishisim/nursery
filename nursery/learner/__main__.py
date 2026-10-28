"""Embed images, train shared learners, and run the frozen acquisition comparison."""

import argparse

from .embed import embed_episodes
from .train import BODY_MODES, EPISODE_MODES, train_run
from .losses import LOSSES


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
    train.add_argument("--body", choices=BODY_MODES, default="none", help="Body input: none, real, shuffled (whole sequences swapped across clips) or oracle (true held-object identity)")
    train.add_argument("--episodes", choices=EPISODE_MODES, default="none", help="Bags for the optional episode losses: contact episodes, temporal control blocks, or none (default)")
    train.add_argument("--body-dropout", type=float, help="Probability of scoring a pair without its body (default in config.json)")
    for loss in LOSSES:
        train.add_argument(f"--w-{loss}", type=float, help=f"Weight of the {loss} loss (0 switches it off; default in config.json)")
    train.add_argument("--heldout-subject", help="Subject excluded from training (default in config.json; 'none' keeps everyone)")
    train.add_argument("--pool", choices=("max", "soft"), default="max", help="Region pooling in the training loss; --body requires soft")
    train.add_argument("--tau-p", type=float, default=0.1, help="Soft-pooling temperature")
    train.add_argument("--distill-from", help="ORACLE DIAGNOSTIC: finished oracle-body run directory used as a frozen teacher (picture-only student)")
    train.add_argument("--distill-weight", type=float, default=1.0, help="Weight of KL(teacher || student) (with --distill-from; 0 = plain run)")
    train.add_argument("--distill-temp", type=float, default=2.0, help="Distillation temperature")
    train.add_argument("--inspect-epoch", type=int, help="Also record probe scores, word-group accuracy and teacher agreement after this many epochs")
    train.add_argument("--labels", action="store_true", help="LABEL DIAGNOSTIC: train on the word of an object visible in the frame instead of the speech (picture-only)")
    train.add_argument("--split", choices=("subject", "clips"), default="subject", help="subject: s8 held out (default). clips: also hold out dev/test source clips of s1-s7 and select epochs on body-present dev (needs --pool soft)")
    train.add_argument("--arch", choices=("additive", "joint"), default="additive", help="additive: picture score + body-word cosine (default). joint: HISTORICAL width-128 pilot from archive/code, not the proposed joint learner; one transformer over visual and body tokens (needs --pool soft, --body none|real, --episodes none, no aux losses)")
    train.add_argument("--physical", help="Directory of per-clip physical signals (load, skin temperature, contact area); adds two body-encoder groups (needs --body real)")
    train.add_argument("--physical-shuffle", action="store_true", help="CONTROL: load/skin taken from another source clip of the same subject and partition (needs --physical)")
    train.add_argument("--name", help="Output directory prefix (default: the condition)")
    acquisition = commands.add_parser("acquisition", help="Frozen three-learner acquisition curves")
    acquisition.add_argument("action", choices=("prepare", "timing", "run", "report"))
    acquisition.add_argument("--root", default="outputs/runs/acquisition_curves")
    acquisition.add_argument("--inputs", help="Existing corrected speech and feature-cache run (prepare only)")
    acquisition.add_argument("--episodes", help="Existing episodes with physical_by_clip (prepare only)")
    acquisition.add_argument("--plot-python", help="Python with matplotlib for PNG/PDF output (report only)")
    args = parser.parse_args()
    if args.command == "embed":
        embed_episodes(args.episodes, args.output, args.device, args.batch)
    elif args.command == "acquisition":
        if args.action == "prepare":
            if not args.inputs or not args.episodes:
                parser.error("acquisition prepare requires --inputs and --episodes")
            from .acquisition_data import prepare, audit
            prepare(args.root, args.inputs, args.episodes)
            audit(args.root)
        elif args.action == "report":
            from .acquisition_report import report
            report(args.root, plot_python=args.plot_python)
        else:
            from .acquisition import run, timing
            (timing if args.action == "timing" else run)(args.root)
    else:
        weights = {loss: getattr(args, f"w_{loss}") for loss in LOSSES}
        train_run(args.run, args.condition, args.seed, args.epochs, args.name, args.pool, args.tau_p, args.body,
                  args.episodes, weights, args.heldout_subject, args.body_dropout, args.distill_from, args.distill_weight,
                  args.distill_temp, args.inspect_epoch, args.labels, args.split, args.arch, args.physical, args.physical_shuffle)


if __name__ == "__main__":
    main()
