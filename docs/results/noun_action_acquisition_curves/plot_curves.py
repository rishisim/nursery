"""Plot Step 9 noun and action acquisition curves from the archived results.json.

Usage: python plot_curves.py   (needs matplotlib; writes acquisition_curves.png next to this file)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = json.loads((HERE / "results.json").read_text())["by_clip_count"]
CLIPS = sorted(int(c) for c in RESULTS)
ARMS = [("V", "Video (V)", "#7f7f7f", "o"),
        ("V+G", "Video + gaze (V+G)", "#2a9d5c", "s"),
        ("V+G+B", "Video + gaze + body (V+G+B)", "#2f6fd6", "D")]
PANELS = [("noun", "Nouns (held-out test)"), ("action", "Actions (test + s8 pooled)")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

for ax, (rung, title) in zip(axes, PANELS):
    for arm, label, color, marker in ARMS:
        acc = [RESULTS[str(c)][rung]["accuracy"][arm] for c in CLIPS]
        mean = [a["mean"] for a in acc]
        sd = [a["sd"] for a in acc]
        ax.fill_between(CLIPS, [m - s for m, s in zip(mean, sd)], [m + s for m, s in zip(mean, sd)],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(CLIPS, mean, color=color, marker=marker, markersize=6, linewidth=2.2, label=label)
    shuffled = [(c, RESULTS[str(c)][rung]["accuracy"]["V+G+B shuffled"]) for c in CLIPS
                if "V+G+B shuffled" in RESULTS[str(c)][rung]["accuracy"]]
    ax.errorbar([c for c, _ in shuffled], [a["mean"] for _, a in shuffled], yerr=[a["sd"] for _, a in shuffled],
                fmt="X", color="#2f6fd6", mfc="white", mew=1.6, markersize=9, capsize=3, linestyle="none",
                label="V+G+B, shuffled body (control)", zorder=5)
    ax.axhline(25, color="#999999", linestyle="--", linewidth=1)
    ax.text(CLIPS[-1], 26, "chance (25%)", ha="right", va="bottom", fontsize=9, color="#777777")
    ax.set_title(title)
    ax.set_xlabel("Training clips")
    ax.set_xticks(CLIPS)
    ax.set_ylim(20, 85)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Four-choice accuracy (%)")

# Callout: V+G+B at 315 clips roughly matches V at 630 clips on actions.
act = axes[1]
vgb315 = RESULTS["315"]["action"]["accuracy"]["V+G+B"]["mean"]
v630 = RESULTS["630"]["action"]["accuracy"]["V"]["mean"]
act.plot([315, 630], [vgb315, v630], color="#555555", linestyle=":", linewidth=1.3)
act.annotate(f"V+G+B at 315 clips ({vgb315:.1f}%)\n≈ V at 630 clips ({v630:.1f}%)\n~2× data efficiency",
             xy=(315, vgb315), xytext=(330, 37), fontsize=9.5, color="#333333",
             arrowprops=dict(arrowstyle="->", color="#555555", linewidth=1),
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.text(0.5, -0.07, "Lines: mean over 5 seeds. Bands and error bars: ±1 SD across seeds.",
         ha="center", fontsize=9, color="#666666")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(HERE / "acquisition_curves.png", dpi=200, facecolor="white", bbox_inches="tight")
