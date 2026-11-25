"""Aggregate saved U3 acquisition runs and render a five-panel PNG/PDF figure."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RUN_ROOT = OUTPUTS / "runs/unified_u3_curves"
PROPERTY_PANEL = OUTPUTS / "runs/unified_u2_data/frozen/property_object.parquet"
CLIPS = (63, 158, 315, 473, 630)
SEEDS = tuple(range(5))
ARMS = ("V+G+B", "V+G", "V", "CLIP-Global", "BabyLLaVA-Nursery")
BABY = ARMS[-1]
SHUFFLED = "V+G+B_shuffled_body"
PANELS = {
    "noun": ("U1 Nouns", 4),
    "action": ("U1 Actions", 4),
    "action_object": ("Action + Object", 4),
    "property_object": ("Property + Object", 4),
    "negation": ("Negation", 2),
}
LEXICAL_SPLITS = {"noun": "test", "action": "pooled"}
COMPOSITIONAL = tuple(PANELS)[2:]
STRADDLING = "property_object_straddling"
N_BOOT = 10_000
SLATE = "#1a2332"


def load_straddling_clips(path):
    """Read subset membership from the frozen panel, since metrics omit the flag."""
    if path.is_file():
        try:
            from pyarrow.parquet import read_table
        except ModuleNotFoundError as exc:
            if exc.name != "pyarrow":
                raise
        else:
            rows = read_table(path, columns=["clip", "straddling"]).to_pydict()
            return {clip for clip, keep in zip(rows["clip"], rows["straddling"]) if keep}
    print(f"Straddling subset membership unavailable ({path} or PyArrow missing). "
          "Its intervals will use seeds only.", file=sys.stderr)
    return None


def trial_values(credits, recorded, choices, candidates, clusters=None):
    values = np.asarray(credits, dtype=float) * 100
    if (values.size == 0 or values.size != recorded["n_trials"]
            or not np.all((0 <= values) & (values <= 100))
            or not np.isclose(values.mean(), recorded["accuracy"])):
        raise ValueError("Trial credits disagree with the recorded count or accuracy")
    if any(row is not None and len(row) != choices for row in candidates):
        raise ValueError(f"Expected {choices} choices per trial")
    return values, clusters


def run_values(data, arm, straddling_clips):
    """Yield (scoring, panel) and a seed's percentage credits or saved mean."""
    held = data["held_out"]
    summary = data["summary"]
    for panel, split in LEXICAL_SPLITS.items():
        trials = held["as_trained"][panel][split]
        yield ("as_trained", panel), trial_values(
            trials["credits"], trials, 4, trials["candidate_rows"], trials.get("clusters")
        )

    for panel in COMPOSITIONAL:
        result = held["compositional"][panel]
        trials = result["trials"]
        if sum(result[s]["n_trials"] for s in ("test", "s8") if s in result) != len(trials):
            raise ValueError(f"{panel} trials do not match pooled test + S8 counts")
        scoring = "pmi" if arm == BABY else "as_trained"
        yield (scoring, panel), trial_values(
            [row["credit"] for row in trials], result["pooled"], PANELS[panel][1],
            [row["scores"] for row in trials], [row["clip"] for row in trials],
        )
        if panel == "property_object" and "straddling" in result:
            subset = result["straddling"]
            if straddling_clips is None:
                yield (scoring, STRADDLING), (subset["accuracy"], None)
            else:
                selected = [row for row in trials if row["clip"] in straddling_clips]
                yield (scoring, STRADDLING), trial_values(
                    [row["credit"] for row in selected], subset, 4,
                    [row["scores"] for row in selected], [row["clip"] for row in selected],
                )

    if arm == BABY:
        for panel, result in summary.get("compositional_raw_ll", {}).items():
            yield ("raw_ll", panel), (result["pooled"]["accuracy"], None)
            if panel == "property_object" and "straddling" in result:
                yield ("raw_ll", STRADDLING), (result["straddling"]["accuracy"], None)
        for panel, accuracy in summary.get("u1_bare_diagnostic", {}).items():
            yield ("u1_bare_diagnostic", panel), (accuracy, None)


def bootstrap(observations, rng):
    """Resample seed occurrences, with an independent inner draw for each occurrence."""
    means = np.array([np.mean(values) for values, _ in observations])
    if not np.all((0 <= means) & (means <= 100)):
        raise ValueError("Expected finite percentage accuracies")
    seed_only = np.isscalar(observations[0][0])
    if seed_only and len(means) == 1:
        return float(means[0]), None, None, "seed_only"
    method = "seed_only"
    if not seed_only:
        method = "seed_and_clip" if observations[0][1] is not None else "seed_and_trial"

    draws = rng.integers(len(means), size=(N_BOOT, len(means)))
    sampled = np.empty(draws.shape)
    for seed, (values, clips) in enumerate(observations):
        mask = draws == seed
        count = int(mask.sum())
        if seed_only:
            sampled[mask] = values
        elif clips is None or len(set(clips)) == len(values):
            # Multinomial counts give the same distribution as drawing trial indices.
            credits, counts = np.unique(values, return_counts=True)
            sampled[mask] = rng.multinomial(len(values), counts / len(values), size=count) @ (
                credits / len(values)
            )
        else:
            # Resample whole source clips and retain trial weighting within the seed.
            unique, inverse = np.unique(clips, return_inverse=True)
            totals = np.bincount(inverse, weights=values)
            counts = np.bincount(inverse)
            indices = rng.integers(len(unique), size=(count, len(unique)))
            sampled[mask] = totals[indices].sum(axis=1) / counts[indices].sum(axis=1)
    low, high = np.quantile(sampled.mean(axis=1), [0.025, 0.975])
    return float(means.mean()), float(low), float(high), method


def aggregate(root, property_panel):
    rows, missing = [], []
    straddling_clips = load_straddling_clips(property_panel)
    rng = np.random.default_rng(0)
    for arm in (*ARMS, SHUFFLED):
        for n in ((158, 630) if arm == SHUFFLED else CLIPS):
            keys = [
                ("pmi" if arm == BABY and panel not in LEXICAL_SPLITS else "as_trained", panel)
                for panel in (*PANELS, STRADDLING)
            ]
            if arm == BABY:
                keys.extend(("raw_ll", p) for p in (*COMPOSITIONAL, STRADDLING))
                keys.extend(("u1_bare_diagnostic", p) for p in LEXICAL_SPLITS)
            observations = {key: {} for key in keys}
            found = []
            for seed in SEEDS:
                path = root / arm / f"n{n}" / f"seed{seed}" / "metrics.json"
                if not path.is_file():
                    missing.append(str(path.relative_to(root)))
                    continue
                try:
                    data = json.loads(path.read_text())
                    run = data["run"]
                    if (run["name"], run["n_clips"], run["seed"]) != (arm, n, seed):
                        raise ValueError("Run name, clip count or seed differs from the path")
                    for key, values in run_values(data, arm, straddling_clips):
                        observations[key][seed] = values
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"{path}: {exc}") from exc
                found.append(seed)

            for (scoring, panel), by_seed in observations.items():
                mean, low, high, method = (
                    bootstrap(list(by_seed.values()), rng) if by_seed
                    else (None, None, None, None)
                )
                unavailable = [seed for seed in found if seed not in by_seed]
                if unavailable:
                    print(f"{arm}/n{n} {panel} ({scoring}) unavailable in seeds {unavailable}",
                          file=sys.stderr)
                rows.append(dict(
                    arm=arm, n_clips=n, panel=panel, scoring=scoring, n_seeds=len(by_seed),
                    n_runs_found=len(found), seeds_used=" ".join(map(str, by_seed)),
                    mean_accuracy_pct=mean, ci95_low_pct=low, ci95_high_pct=high,
                    ci_method=method,
                ))
    if missing:
        print(f"Missing {len(missing)} run(s) under {root}:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
    return rows, missing


def first_crossing(target, means):
    """Find the first upward crossing on adjacent observed log-budget points."""
    if target is None:
        return None, "missing_target"
    previous = None
    for n, mean in zip(CLIPS, means):
        if mean is None:
            return None, "incomplete_ladder"
        if mean >= target:
            if previous is None:
                return n, "at_or_below_minimum"
            low_n, low_mean = previous
            fraction = (target - low_mean) / (mean - low_mean)
            return float(low_n * (n / low_n) ** fraction), "interpolated"
        previous = n, mean
    return ">630", "not_reached"


def sample_efficiency(rows):
    means = {(r["arm"], r["n_clips"], r["panel"], r["scoring"]): r["mean_accuracy_pct"]
             for r in rows}
    results = []
    for panel in PANELS:
        for n in (158, 315):
            target = means["V+G+B", n, panel, "as_trained"]
            for arm in ("V", "CLIP-Global", BABY):
                scoring = "pmi" if arm == BABY and panel in COMPOSITIONAL else "as_trained"
                ladder = [means[arm, clips, panel, scoring] for clips in CLIPS]
                match, status = first_crossing(target, ladder)
                multiplier = match / n if isinstance(match, (int, float)) else None
                if status == "not_reached":
                    multiplier = f">{CLIPS[-1] / n:.3f}"
                results.append(dict(
                    panel=panel, reference_n_clips=n, target_accuracy_pct=target,
                    arm=arm, scoring=scoring, clips_to_match=match,
                    multiplier=multiplier, status=status,
                ))
    return results


def write_table(out, stem, rows, **metadata):
    with (out / f"{stem}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (out / f"{stem}.json").write_text(json.dumps(dict(rows=rows, **metadata), indent=2) + "\n")


def plot(rows, missing, out):
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans"], "font.serif": ["Georgia", "DejaVu Serif"],
        "text.color": SLATE, "axes.labelcolor": SLATE, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False, "xtick.color": SLATE,
        "ytick.color": SLATE, "legend.frameon": False, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 5, figsize=(16.2, 4.6), sharey=True)
    fig.subplots_adjust(left=0.059, right=0.983, top=0.68, bottom=0.22, wspace=0.20)
    fig.text(0.058, 0.91, "Language acquisition by training experience",
             family="serif", fontsize=24)
    fig.text(0.058, 0.835, "U3  ·  Held-out word and sentence accuracy",
             color="#536673", fontsize=11)
    styles = [
        (ARMS[0], "as_trained", ARMS[0], "#bd663f", "s", "-"),
        (ARMS[1], "as_trained", ARMS[1], "#208575", "o", "-"),
        (ARMS[2], "as_trained", ARMS[2], "#475b88", "^", "-"),
        (ARMS[3], "as_trained", ARMS[3], "#795483", "P", "-"),
        (BABY, "pmi", f"{BABY} (PMI for sentences)", "#987319", "D", "-"),
        (BABY, "raw_ll", f"{BABY} (Raw-LL)", "#987319", "v", "--"),
        (SHUFFLED, "as_trained", "Shuffled body", "#bd663f", "X", "none"),
    ]
    lookup = {(r["arm"], r["n_clips"], r["panel"], r["scoring"]): r for r in rows}
    handles = {}
    for ax, (panel, (label, choices)) in zip(axes, PANELS.items()):
        chance = 100 / choices
        ax.set_title(label, loc="left", family="serif", fontsize=15, pad=9)
        ax.set(xscale="log", xlim=(55, 715), ylim=(0, 101), yticks=range(0, 101, 25))
        ax.set_xticks(CLIPS, labels=CLIPS)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.tick_params(length=0, pad=5, labelsize=9)
        ax.spines["bottom"].set_color("#b9c3ca")
        ax.grid(axis="y", color="#e7ebed", lw=0.65)
        ax.axhline(chance, color="#788893", lw=0.9, ls=":", zorder=2)
        for arm, scoring, name, color, marker, line in styles:
            if arm == BABY and scoring == "pmi" and panel in LEXICAL_SPLITS:
                scoring = "as_trained"
            points = [lookup.get((arm, n, panel, scoring), {}) for n in CLIPS]
            mean, low, high = (np.array([p.get(k) for p in points], dtype=float) for k in
                               ("mean_accuracy_pct", "ci95_low_pct", "ci95_high_pct"))
            if np.isnan(mean).all():
                continue
            ax.fill_between(CLIPS, low, high, color=color, alpha=0.11, linewidth=0,
                            where=np.isfinite(low) & (arm != SHUFFLED))
            available = np.isfinite(low)
            adjacent = np.r_[False, available[:-1]] | np.r_[available[1:], False]
            isolated = available & (~adjacent | (arm == SHUFFLED))
            ax.vlines(np.asarray(CLIPS)[isolated], low[isolated], high[isolated],
                      color=color, lw=1.25, alpha=0.7, zorder=3)
            handle, = ax.plot(CLIPS, mean, color=color, marker=marker, ls=line,
                              ms=5.5, mec="white", mew=0.6, lw=1.75, zorder=4)
            handles[name] = handle
    axes[0].set_ylabel("Accuracy")
    fig.supxlabel("Training clips (log scale)", y=0.125, fontsize=10)
    if handles:
        names = [style[2] for style in styles if style[2] in handles]
        fig.legend([handles[name] for name in names], names, loc="center left",
                   bbox_to_anchor=(0.052, 0.779), ncol=len(names), fontsize=9)
    fig.text(
        0.058, 0.087,
        "95% intervals resample seeds and held-out trials or source clips. Raw-LL uses seed means only and covers sentence panels.",
        color="#536673", fontsize=9,
    )
    missing_note = f"  ·  {len(missing)} of 135 runs missing (see aggregation)." if missing else ""
    fig.text(
        0.058, 0.045,
        "Dotted lines mark chance. Sentence panels pool test + S8. U1 Actions uses the pooled held-out split."
        + missing_note, color="#536673", fontsize=9,
    )
    try:
        for ext in ("png", "pdf"):
            fig.savefig(out / f"u3_acquisition_curves.{ext}", dpi=300, facecolor="white")
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, help="Directory under the repository's outputs/")
    parser.add_argument("--property-panel", type=Path, default=PROPERTY_PANEL,
                        help="Frozen property_object.parquet with straddling clip membership")
    args = parser.parse_args()
    out = (args.out_dir or RUN_ROOT / "figures").resolve()
    if not out.is_relative_to(OUTPUTS.resolve()) or out == OUTPUTS.resolve():
        parser.error("--out-dir must be under the repository's outputs/")
    rows, missing = aggregate(args.run_root.resolve(), args.property_panel)
    out.mkdir(parents=True, exist_ok=True)
    write_table(out, "u3_aggregation", rows, missing_runs=missing, bootstrap_draws=N_BOOT,
                run_root=str(args.run_root.resolve()))
    write_table(out, "u3_sample_efficiency", sample_efficiency(rows))
    plot(rows, missing, out)
    print(f"U3 tables and figures written to {out}")


if __name__ == "__main__":
    main()
