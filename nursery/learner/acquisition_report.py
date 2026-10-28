"""Aggregate and plot the complete frozen acquisition-curve matrix."""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from nursery.common import _output_directory

CONDITIONS = ("video_only", "additive", "joint_interaction")
CONDITION_LABELS = {
    "video_only": "Video only",
    "additive": "Video + body (additive)",
    "joint_interaction": "Video + body (joint)",
}
PAIRS = (("additive", "video_only"), ("joint_interaction", "video_only"),
         ("joint_interaction", "additive"))


def _read(path):
    return json.loads(Path(path).read_text())


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite numeric {label}, got {value!r}")
    return float(value)


def _accuracy(metrics, section, panel, label):
    try:
        result = metrics[section][panel]
        value = result["accuracy"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing {label} accuracy in {section}.{panel}") from exc
    if value is None:
        raise ValueError(f"Missing {label} accuracy in {section}.{panel}")
    value = _number(value, f"{label} accuracy")
    if not 0 <= value <= 100:
        raise ValueError(f"Out-of-range {label} accuracy: {value}")
    if result.get("unknown_rule") != "25% expected accuracy; retained in fixed denominator":
        raise ValueError(f"Unexpected or missing unknown-query scoring rule for {label}")
    coverage = _number(result.get("coverage"), f"{label} coverage fraction")
    if not 0 <= coverage <= 1:
        raise ValueError(f"Out-of-range {label} vocabulary coverage: {coverage}")
    for field in ("n_trials", "n_known"):
        count = result.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Invalid {label} {field}: {count!r}")
    if result["n_trials"] == 0 or result["n_known"] > result["n_trials"]:
        raise ValueError(f"Invalid {label} coverage counts")
    if not math.isclose(coverage, result["n_known"] / result["n_trials"], abs_tol=1e-12):
        raise ValueError(f"{label} coverage disagrees with its known/total counts")
    return value


def _check_coverage(metrics, section, panel, label, expected):
    result = metrics[section][panel]
    if result["n_trials"] != int(expected["total"]) or result["n_known"] != int(expected["known"]):
        raise ValueError(f"{label} run coverage counts disagree with frozen subset metadata")


def _known_accuracy(metrics, section, panel, label):
    result = metrics[section][panel]
    value = result.get("known_accuracy")
    if result["n_known"] == 0:
        if value is not None:
            raise ValueError(f"{label} has known-word accuracy with zero known queries")
        return None
    value = _number(value, f"{label} known-word accuracy")
    if not 0 <= value <= 100:
        raise ValueError(f"Out-of-range {label} known-word accuracy: {value}")
    return value


def aggregate(root):
    """Validate the full 45-run matrix and return paired summaries.

    Every run must be a final, non-smoke result for the exact protocol hash.
    No data or plotting libraries are required for aggregation.
    """
    root = Path(root).resolve()
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing frozen protocol: {protocol_path}")
    protocol = _read(protocol_path)
    fractions, seeds = protocol.get("fractions"), protocol.get("seeds")
    if not isinstance(fractions, list) or len(fractions) != 5 or not isinstance(seeds, list) or len(seeds) != 3:
        raise ValueError("Protocol must define exactly five fractions and three seeds")
    fractions = [float(x) for x in fractions]
    seeds = [int(x) for x in seeds]
    if protocol.get("unknown_query_score") != 0.25:
        raise ValueError("Frozen protocol must score unknown queries at the 25% 4AFC chance rate")
    protocol_hash = __import__("hashlib").sha256(protocol_path.read_bytes()).hexdigest()

    runs, missing = {}, []
    for fraction in fractions:
        fraction_key = str(float(fraction))
        stats_path = root / "subsets" / fraction_key / "stats.json"
        if not stats_path.is_file():
            raise ValueError(f"Missing frozen subset metadata: {stats_path}")
        stats = _read(stats_path)
        if float(stats.get("fraction", fraction)) != fraction:
            raise ValueError(f"Subset fraction mismatch in {stats_path}")
        for seed in seeds:
            for condition in CONDITIONS:
                path = root / "models" / fraction_key / f"seed_{seed}" / condition / "metrics.json"
                if not path.is_file():
                    missing.append(str(path))
                    continue
                metrics = _read(path)
                expected = {"status": "complete", "smoke": False, "fraction": fraction,
                            "seed": seed, "condition": condition, "protocol_sha256": protocol_hash}
                bad = [key for key, value in expected.items() if metrics.get(key) != value]
                if bad:
                    raise ValueError(f"Run identity/status mismatch ({', '.join(bad)}): {path}")
                expected_epochs = protocol.get("epochs")
                if metrics.get("epochs") != expected_epochs:
                    raise ValueError(f"Run epoch count mismatch: {path}")
                if not isinstance(metrics.get("selection"), dict) or not {"body_present", "picture"} <= metrics["selection"].keys():
                    raise ValueError(f"Incomplete development checkpoint selection record: {path}")
                data = metrics.get("data", {})
                coverage = data.get("vocabulary_coverage", {})
                if coverage != stats.get("vocabulary_coverage"):
                    raise ValueError(f"Run vocabulary coverage differs from frozen subset metadata: {path}")
                panel_sources = {"clips_test": ("primary", "clips_test"),
                                 "s8": ("primary", "s8"),
                                 "picture_test": ("picture", "picture_test"),
                                 "konkle_test": ("picture", "konkle_test")}
                for coverage_name, (section, panel) in panel_sources.items():
                    if coverage_name not in coverage:
                        raise ValueError(f"Missing {coverage_name} frozen vocabulary coverage: {path}")
                    _check_coverage(metrics, section, panel, coverage_name, coverage[coverage_name])
                row = {
                    "fraction": fraction,
                    "seed": seed,
                    "condition": condition,
                    "clips_test": _accuracy(metrics, "primary", "clips_test", "new-source-clip test"),
                    "s8": _accuracy(metrics, "primary", "s8", "held-out-person test"),
                    "picture_test": _accuracy(metrics, "picture", "picture_test", "picture-only transfer"),
                    "konkle_test": _accuracy(metrics, "picture", "konkle_test", "Konkle transfer"),
                    "known_accuracy": {
                        "clips_test": _known_accuracy(metrics, "primary", "clips_test", "new-source-clip test"),
                        "s8": _known_accuracy(metrics, "primary", "s8", "held-out-person test"),
                        "picture_test": _known_accuracy(metrics, "picture", "picture_test", "picture-only transfer"),
                        "konkle_test": _known_accuracy(metrics, "picture", "konkle_test", "Konkle transfer"),
                    },
                    "shuffled_clips_test": None,
                    "shuffled_s8": None,
                    "metrics": metrics,
                }
                if condition != "video_only":
                    row["shuffled_clips_test"] = _accuracy(metrics, "shuffled", "clips_test", "shuffled-body test")
                    row["shuffled_s8"] = _accuracy(metrics, "shuffled", "s8", "shuffled-body s8 test")
                    _check_coverage(metrics, "shuffled", "clips_test", "shuffled-body test", coverage["clips_test"])
                    _check_coverage(metrics, "shuffled", "s8", "shuffled-body s8 test", coverage["s8"])
                # Match all protocol-frozen exposure fields against the subset record.
                for field in ("selected_clip_count", "exposed_clip_count", "utterances", "replay_exposure_seconds"):
                    if field not in stats or field not in data or data[field] != stats[field]:
                        raise ValueError(f"Run/subset {field} mismatch: {path}")
                row["data"] = stats
                runs[(fraction, seed, condition)] = row
    if missing:
        raise ValueError(f"Refusing a partial acquisition report: {len(missing)} of 45 runs are missing; first missing: {missing[0]}")
    if len(runs) != 45:
        raise ValueError(f"Expected 45 complete runs, found {len(runs)}")

    panels = ("clips_test", "s8", "picture_test", "konkle_test", "shuffled_clips_test", "shuffled_s8")
    summary = {
        "status": "complete",
        "protocol_sha256": protocol_hash,
        "unknown_query_score": protocol["unknown_query_score"],
        "fractions": fractions,
        "seeds": seeds,
        "conditions": list(CONDITIONS),
        "n_runs": len(runs),
        "interpretation": "These estimates describe the measured acquisition range; they do not establish a higher asymptotic ceiling.",
        "points": [],
    }
    for fraction in fractions:
        point = {"fraction": fraction, "data": runs[(fraction, seeds[0], CONDITIONS[0])]["data"],
                 "conditions": {}, "paired_differences": {}}
        for condition in CONDITIONS:
            point["conditions"][condition] = {}
            for panel in panels:
                values = [runs[(fraction, seed, condition)][panel] for seed in seeds]
                if all(v is None for v in values):
                    continue
                if any(v is None for v in values):
                    raise ValueError(f"Incomplete {panel} values at fraction {fraction}, {condition}")
                point["conditions"][condition][panel] = {
                    "mean": statistics.mean(values),
                    "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "seed_values": dict(zip(map(str, seeds), values)),
                    "n_seeds": len(values),
                }
            point["conditions"][condition]["known_accuracy"] = {}
            for panel in ("clips_test", "s8", "picture_test", "konkle_test"):
                values = [runs[(fraction, seed, condition)]["known_accuracy"][panel] for seed in seeds]
                if all(value is None for value in values):
                    point["conditions"][condition]["known_accuracy"][panel] = None
                elif any(value is None for value in values):
                    raise ValueError(f"Incomplete known-word {panel} values at fraction {fraction}, {condition}")
                else:
                    point["conditions"][condition]["known_accuracy"][panel] = {
                        "mean": statistics.mean(values),
                        "sample_std": statistics.stdev(values),
                        "seed_values": dict(zip(map(str, seeds), values)),
                        "n_seeds": len(values),
                    }
        for left, right in PAIRS:
            key = f"{left}_minus_{right}"
            point["paired_differences"][key] = {}
            for panel in ("clips_test", "s8", "picture_test", "konkle_test"):
                deltas = [runs[(fraction, seed, left)][panel] - runs[(fraction, seed, right)][panel]
                          for seed in seeds]
                point["paired_differences"][key][panel] = {
                    "mean": statistics.mean(deltas),
                    "sample_std": statistics.stdev(deltas),
                    "seed_values": dict(zip(map(str, seeds), deltas)),
                    "n_seeds": len(deltas),
                }
        summary["points"].append(point)
    return summary


def _summary_rows(summary):
    rows = []
    for point in summary["points"]:
        base = {"fraction": point["fraction"],
                "selected_clips": point["data"]["selected_clip_count"],
                "exposed_clips": point["data"]["exposed_clip_count"],
                "utterances": point["data"]["utterances"],
                "replay_exposure_seconds": point["data"]["replay_exposure_seconds"]}
        for panel, coverage in point["data"].get("vocabulary_coverage", {}).items():
            total = int(coverage["total"])
            known = int(coverage["known"])
            rows.append({**base, "kind": "vocabulary_coverage", "name": "known_queries", "panel": panel,
                         "mean_percent": 100 * known / total if total else None,
                         "sample_std_percent": None, "n_seeds": 0,
                         "known_queries": known, "total_queries": total})
        for condition, panels in point["conditions"].items():
            for panel, result in panels.items():
                if panel == "known_accuracy":
                    for known_panel, known_result in result.items():
                        if known_result is None:
                            continue
                        rows.append({**base, "kind": "known_word_accuracy", "name": condition,
                                     "panel": known_panel, "mean_percent": known_result["mean"],
                                     "sample_std_percent": known_result["sample_std"],
                                     "n_seeds": known_result["n_seeds"]})
                    continue
                rows.append({**base, "kind": "condition", "name": condition, "panel": panel,
                             "mean_percent": result["mean"], "sample_std_percent": result["sample_std"],
                             "n_seeds": result["n_seeds"]})
        for pair, panels in point["paired_differences"].items():
            for panel, result in panels.items():
                rows.append({**base, "kind": "paired_difference", "name": pair, "panel": panel,
                             "mean_percent": result["mean"], "sample_std_percent": result["sample_std"],
                             "n_seeds": result["n_seeds"]})
    return rows


def _write_tables(root, summary):
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    rows = _summary_rows(summary)
    fields = ["fraction", "selected_clips", "exposed_clips", "utterances", "replay_exposure_seconds",
              "kind", "name", "panel", "mean_percent", "sample_std_percent", "n_seeds",
              "known_queries", "total_queries"]
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def cell(result):
        return f'{result["mean"]:.1f} ± {result["sample_std"]:.1f}'

    lines = ["# Acquisition curves report", "",
             f"Complete matrix: {summary['n_runs']} runs, {len(summary['fractions'])} experience amounts, "
             f"{len(summary['seeds'])} paired seeds. Values are mean ± sample standard deviation across seeds; "
             "paired differences use within-seed differences.", "",
             "The x-axis uses exposed independent source clips. Replay exposure is reported separately and does not "
             "count as new independent experience. Accuracy is the fixed four-choice situation-matching task "
             "(25% chance). Queries absent from a subset vocabulary receive 25% expected credit and remain in the fixed denominator. "
             "Coverage is checked against the frozen panel counts. No seed-pooled trial estimate is used.", "",
             "## New-source-clip test", "",
             "| Fraction | Exposed clips | Utterances | Video only | Additive | Joint |",
             "|---:|---:|---:|---:|---:|---:|"]
    for p in summary["points"]:
        lines.append(f"| {p['fraction']:.0%} | {p['data']['exposed_clip_count']} | {p['data']['utterances']} | "
                     f"{cell(p['conditions']['video_only']['clips_test'])} | "
                     f"{cell(p['conditions']['additive']['clips_test'])} | "
                     f"{cell(p['conditions']['joint_interaction']['clips_test'])} |")
    lines += ["", "## Paired accuracy differences", "",
              "Mean ± sample standard deviation of paired percentage-point differences; positive values favor the first condition.", "",
              "| Fraction | Additive − video | Joint − video | Joint − additive |",
              "|---:|---:|---:|---:|"]
    for p in summary["points"]:
        d = p["paired_differences"]
        lines.append(f"| {p['fraction']:.0%} | {cell(d['additive_minus_video_only']['clips_test'])} | "
                     f"{cell(d['joint_interaction_minus_video_only']['clips_test'])} | "
                     f"{cell(d['joint_interaction_minus_additive']['clips_test'])} |")
    lines += ["", "## Supporting panels", "",
              "Shuffled-body values are evaluation controls on the same additive or joint learners, not separately trained conditions.", "",
              "| Fraction | Additive shuffled | Joint shuffled |", "|---:|---:|---:|"]
    for p in summary["points"]:
        lines.append(f"| {p['fraction']:.0%} | {cell(p['conditions']['additive']['shuffled_clips_test'])} | "
                     f"{cell(p['conditions']['joint_interaction']['shuffled_clips_test'])} |")
    for title, panel, keys in (("Held-out person (s8)", "s8", CONDITIONS),
                               ("Picture-only transfer", "picture_test", CONDITIONS),
                               ("Konkle transfer", "konkle_test", CONDITIONS)):
        lines += ["", f"### {title}", "", "| Fraction | " + " | ".join(CONDITION_LABELS[k] for k in keys) + " |",
                  "|---:|" + "---:|" * len(keys)]
        for p in summary["points"]:
            vals = [cell(p["conditions"][k][panel]) for k in keys]
            lines.append(f"| {p['fraction']:.0%} | " + " | ".join(vals) + " |")
    lines += ["", "## Known-word accuracy (secondary)", "",
              "These scores use only queries present in the subset vocabulary; OOV queries are excluded here and remain chance-scored in the primary results.", ""]
    for title, panel in (("New-source-clip test", "clips_test"), ("Held-out person (s8)", "s8"),
                         ("Picture-only transfer", "picture_test"), ("Konkle transfer", "konkle_test")):
        lines += [f"### {title}", "", "| Fraction | " + " | ".join(CONDITION_LABELS[k] for k in CONDITIONS) + " |",
                  "|---:|" + "---:|" * len(CONDITIONS)]
        for p in summary["points"]:
            vals = []
            for condition in CONDITIONS:
                result = p["conditions"][condition]["known_accuracy"][panel]
                vals.append("not available" if result is None else cell(result))
            lines.append(f"| {p['fraction']:.0%} | " + " | ".join(vals) + " |")
    lines += ["", "## Exposure and vocabulary coverage", "",
              "Selected and exposed clip counts are shown separately because visual windows can expose neighboring source clips. "
              "Replay exposure is training-time exposure duration, not independent experience. Vocabulary coverage is known queries / total fixed-panel queries.", "",
              "| Fraction | Selected clips | Exposed clips | Utterances | Unique source duration (s) | Replay exposure (s) | New-clip coverage | s8 coverage | Picture coverage | Konkle coverage |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in summary["points"]:
        data = p["data"]
        coverage = data.get("vocabulary_coverage", {})
        def coverage_cell(name):
            value = coverage.get(name)
            return f"{value['known']}/{value['total']}" if value and value.get("total") else "—"
        duration = data.get("unique_source_duration_seconds")
        duration = "not recorded" if duration is None else str(duration)
        lines.append(f"| {p['fraction']:.0%} | {data['selected_clip_count']} | {data['exposed_clip_count']} | "
                     f"{data['utterances']} | {duration} | {data['replay_exposure_seconds']} | "
                     f"{coverage_cell('clips_test')} | {coverage_cell('s8')} | "
                     f"{coverage_cell('picture_test')} | {coverage_cell('konkle_test')} |")
    lines += ["", "This report describes the observed learning curves over the tested data range. It does not claim a higher asymptotic ceiling.", ""]
    lines.insert(-1, "Results on test panels consulted during earlier work are exploratory.")
    (root / "report.md").write_text("\n".join(lines))


_PLOT_CODE = r'''import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1])
s = json.loads((root / "summary.json").read_text())
conds = s["conditions"]
labels = {"video_only":"Video only", "additive":"Video + body (additive)", "joint_interaction":"Video + body (joint)"}
colors = {"video_only":"#3b6fb6", "additive":"#e28e2c", "joint_interaction":"#39845a"}
points = s["points"]
x = np.array([p["data"]["exposed_clip_count"] for p in points], dtype=float)
fig, ax = plt.subplots(figsize=(8.2, 5.1), constrained_layout=True)
for condition in conds:
    y = np.array([p["conditions"][condition]["clips_test"]["mean"] for p in points])
    sd = np.array([p["conditions"][condition]["clips_test"]["sample_std"] for p in points])
    seed_values = np.array([[p["conditions"][condition]["clips_test"]["seed_values"][str(seed)] for seed in s["seeds"]] for p in points], dtype=float)
    ax.errorbar(x, y, yerr=sd, color=colors[condition], marker="o", linewidth=2, capsize=3, label=labels[condition])
    for j in range(seed_values.shape[1]):
        ax.scatter(x, seed_values[:, j], color=colors[condition], s=15, alpha=.42, zorder=3)
ax.axhline(25, color="#555555", linestyle="--", linewidth=1.2, label="4AFC chance (25%)")
ax.set(xlabel="Exposed independent source clips", ylabel="Held-out situation-matching accuracy (%)", ylim=(0, 100))
ax.set_title("Language acquisition across training experience")
ax.grid(axis="y", alpha=.22)
ax.legend(frameon=False, loc="best")
fig.savefig(root / "acquisition_curves.png", dpi=180)
fig.savefig(root / "acquisition_curves.pdf")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharey=True, constrained_layout=True)
for ax, panel, title in ((axes[0], "picture_test", "Picture-only transfer"), (axes[1], "konkle_test", "Konkle transfer")):
    for condition in conds:
        y = np.array([p["conditions"][condition][panel]["mean"] for p in points])
        sd = np.array([p["conditions"][condition][panel]["sample_std"] for p in points])
        seed_values = np.array([[p["conditions"][condition][panel]["seed_values"][str(seed)] for seed in s["seeds"]] for p in points], dtype=float)
        ax.errorbar(x, y, yerr=sd, color=colors[condition], marker="o", linewidth=1.8, capsize=3, label=labels[condition])
        for j in range(seed_values.shape[1]):
            ax.scatter(x, seed_values[:, j], color=colors[condition], s=13, alpha=.4, zorder=3)
    ax.axhline(25, color="#555555", linestyle="--", linewidth=1)
    ax.set(title=title, xlabel="Exposed independent source clips", ylim=(0, 100))
    ax.grid(axis="y", alpha=.22)
axes[0].set_ylabel("Held-out accuracy (%)")
axes[1].legend(frameon=False, loc="best")
fig.savefig(root / "picture_transfer.png", dpi=180)
fig.savefig(root / "picture_transfer.pdf")
plt.close(fig)
'''


def _plot(root, plot_python=None):
    # Preserve venv symlinks: resolving their interpreter link bypasses the venv site-packages.
    python = os.path.abspath(os.path.expanduser(str(plot_python))) if plot_python else sys.executable
    with tempfile.TemporaryDirectory(prefix=".mpl-", dir=root) as mpl_config:
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = mpl_config
        result = subprocess.run([python, "-c", _PLOT_CODE, str(root)], capture_output=True, text=True, env=env)
        if result.returncode:
            detail = result.stderr.strip().splitlines()[-1:] or ["plotting process failed"]
            raise RuntimeError(f"Could not render acquisition plots with {python}: {detail[0]}")


def report(root, plot_python=None):
    """Write aggregate tables, prose and plots after validating all 45 runs."""
    root = Path(root).expanduser().resolve()
    summary = aggregate(root)
    root = _output_directory(root)
    _write_tables(root, summary)
    _plot(root, plot_python)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Ignored acquisition run root")
    parser.add_argument("--plot-python", type=Path,
                        help="Python interpreter with matplotlib installed; defaults to this interpreter")
    args = parser.parse_args(argv)
    result = report(args.root, args.plot_python)
    print(f"Wrote acquisition report for {result['n_runs']} complete runs under {args.root}")


if __name__ == "__main__":
    main()
