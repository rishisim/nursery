#!/usr/bin/env python3
"""Render the frozen public prototype performance and cost projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"Real": "#444444", "Hailuo": "#D97706", "LTX": "#2563EB"}
HATCHES = {"Real": "", "Hailuo": "//", "LTX": ".."}
ARMS = (("Real", "real"), ("Hailuo", "hailuo"), ("LTX", "ltx"))
PROBES = (("Noun", "noun"), ("Adjective", "adjective"), ("Action", "action_guardrail"), ("Viewpoint*", "viewpoint_guardrail"))


def assert_ignored(path: Path) -> None:
    if subprocess.run(["git", "check-ignore", "-q", "--", str(path)], check=False).returncode != 0:
        raise RuntimeError("E_OUTPUT_ROOT_NOT_IGNORED")


def style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#777777")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)


def grouped_bars(axis, values: dict, ylabel: str, ylim: tuple[float, float]) -> None:
    x = np.arange(len(PROBES))
    width = 0.24
    for offset, (label, key) in zip((-width, 0, width), ARMS):
        heights = [values[probe][key] for _, probe in PROBES]
        bars = axis.bar(x + offset, heights, width, label=label, color=COLORS[label], edgecolor="#222222", linewidth=0.6, hatch=HATCHES[label])
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in heights], padding=2, fontsize=8)
    axis.set_xticks(x, [label for label, _ in PROBES])
    axis.set_ylabel(ylabel)
    axis.set_ylim(*ylim)
    axis.axvspan(2.5, 3.5, color="#F3F4F6", zorder=-2)
    axis.text(3, ylim[0] + (ylim[1] - ylim[0]) * 0.04, "guardrail failed", ha="center", va="bottom", fontsize=8, color="#6B7280")
    style_axis(axis)


def render_performance(metrics: dict, output: Path) -> None:
    top1 = {probe: {arm: metrics[probe][arm]["top1_accuracy"] for _, arm in ARMS} for _, probe in PROBES}
    margin = {probe: {arm: metrics[probe][arm]["mean_target_minus_best_distractor_margin"] for _, arm in ARMS} for _, probe in PROBES}
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    grouped_bars(axes[0], top1, "Top-1 accuracy over 10 frames", (0, 1.18))
    axes[0].set_title("Matched grounded-lexical top-1 accuracy", loc="left", fontsize=13, weight="bold")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    grouped_bars(axes[1], margin, "Mean target − best distractor", (-0.52, 0.55))
    axes[1].axhline(0, color="#111827", linewidth=1.0)
    axes[1].set_title("Matched target-versus-distractor margin", loc="left", fontsize=13, weight="bold")
    figure.suptitle("Real, Hailuo, and LTX performance — one public 10-second source clip", x=0.02, ha="left", fontsize=16, weight="bold")
    figure.text(0.02, 0.955, "Same 10 timestamps and frozen CLIP evaluator; *viewpoint has negative margin in every arm and cannot support a ranking.", ha="left", va="top", fontsize=10, color="#4B5563")
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def render_cost(costs: dict, projection: dict, output: Path) -> dict:
    per_hour = {name: value * projection["clips_per_generated_hour"] for name, value in costs.items()}
    yields = projection["yield_scenarios"]
    real_costs = projection["real_fully_loaded_cost_sensitivity_USD_per_usable_hour"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)

    x = np.arange(len(yields))
    width = 0.34
    for offset, label in ((-width / 2, "Hailuo"), (width / 2, "LTX")):
        values = [per_hour[label] / value for value in yields]
        bars = axes[0].bar(x + offset, values, width, label=label, color=COLORS[label], edgecolor="#222222", linewidth=0.6, hatch=HATCHES[label])
        axes[0].bar_label(bars, labels=[f"${value:,.0f}" for value in values], padding=3, fontsize=8)
    axes[0].set_xticks(x, [f"{int(value * 100)}%" for value in yields])
    axes[0].set_xlabel("Accepted output yield")
    axes[0].set_ylabel("Direct generation API cost per usable hour")
    axes[0].set_title("Generation cost by acceptance yield", loc="left", fontsize=13, weight="bold")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    for label in ("Hailuo", "LTX"):
        savings = [value - per_hour[label] for value in real_costs]
        axes[1].plot(real_costs, savings, label=label, color=COLORS[label], linewidth=2.4, marker="o", markersize=4, linestyle="--" if label == "Hailuo" else "-")
        axes[1].axvline(per_hour[label], color=COLORS[label], linewidth=1.0, linestyle=":" if label == "LTX" else "--", alpha=0.8)
        axes[1].text(per_hour[label] + 8, 18, f"{label} break-even\n${per_hour[label]:,.0f}/h", color=COLORS[label], fontsize=8, va="bottom")
    axes[1].axhline(0, color="#111827", linewidth=1.0)
    axes[1].set_xlabel("Assumed prospective real-data cost per usable hour")
    axes[1].set_ylabel("Real cost − synthetic generation cost")
    axes[1].set_title("Generation-only break-even sensitivity", loc="left", fontsize=13, weight="bold")
    axes[1].legend(frameon=False)
    style_axis(axes[1])

    figure.suptitle("Direct video-generation API cost projections", x=0.02, ha="left", fontsize=16, weight="bold")
    figure.text(0.02, 0.935, "Observed accepted-request charges projected linearly from 10 seconds to one usable hour; descriptor, local compute, storage, QA, labor, and paid failures are excluded.", ha="left", va="top", fontsize=9.5, color="#4B5563")
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return per_hour


def run(config_path: Path, metrics_path: Path, output_root: Path) -> None:
    assert_ignored(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())["public_single_clip_cost_projection"]
    if config["status"] != "FROZEN_BEFORE_COST_PROJECTION_RENDER":
        raise RuntimeError("E_COST_PROTOCOL_NOT_FROZEN")
    result = json.loads(metrics_path.read_text())
    costs = {
        "Hailuo": float(config["observed_accepted_request_costs_USD"]["hailuo_10_seconds_2K"]),
        "LTX": float(config["observed_accepted_request_costs_USD"]["ltx_2_3_10_seconds_1080p"]),
    }
    render_performance(result["grounded_lexical_metrics"], output_root / "performance_comparison.png")
    per_hour = render_cost(costs, config["projection"], output_root / "cost_comparison.png")
    compact = {
        "status": "EXPLORATORY_COST_PROJECTION_COMPLETE",
        "accepted_request_cost_USD_per_10_seconds": costs,
        "generation_only_cost_USD_per_hour_at_100_percent_yield": per_hour,
        "generation_only_break_even_real_cost_USD_per_usable_hour": per_hour,
        "unpriced_synthetic_categories": config["unpriced_synthetic_categories"],
        "real_fully_loaded_cost_observed": False,
        "lower_cost_claim_supported": False,
    }
    (output_root / "compact_projection.json").write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(compact, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic_video_preregistration.json"))
    parser.add_argument("--metrics", type=Path, default=Path("results/synthetic_video_public_three_arm_grounding.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/public_cost_performance"))
    args = parser.parse_args()
    run(args.config, args.metrics, args.output_root)


if __name__ == "__main__":
    main()
