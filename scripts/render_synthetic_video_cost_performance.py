#!/usr/bin/env python3
"""Render the frozen public prototype performance and cost projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"Real": "#444444", "Hailuo": "#D97706", "LTX API": "#2563EB", "LTX Juno": "#7C3AED"}
HATCHES = {"Real": "", "Hailuo": "//", "LTX API": "..", "LTX Juno": "xx"}
ARMS = (("Real", "real"), ("Hailuo", "hailuo"), ("LTX API", "ltx_api"), ("LTX Juno", "ltx_juno"))
PROBES = (("Noun", "noun"), ("Adjective", "adjective"), ("Action", "action_guardrail"))


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
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(ARMS))
    for offset, (label, key) in zip(offsets, ARMS):
        heights = [values[probe][key] for _, probe in PROBES]
        bars = axis.bar(x + offset, heights, width, label=label, color=COLORS[label], edgecolor="#222222", linewidth=0.6, hatch=HATCHES[label])
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in heights], padding=2, fontsize=8)
    axis.set_xticks(x, [label for label, _ in PROBES])
    axis.set_ylabel(ylabel)
    axis.set_ylim(*ylim)
    style_axis(axis)


def render_performance(metrics: dict, output: Path) -> None:
    top1 = {probe: {arm: metrics[probe][arm]["top1_accuracy"] for _, arm in ARMS} for _, probe in PROBES}
    margin = {probe: {arm: metrics[probe][arm]["mean_target_minus_best_distractor_margin"] for _, arm in ARMS} for _, probe in PROBES}
    figure, axes = plt.subplots(2, 1, figsize=(11, 8))
    grouped_bars(axes[0], top1, "Top-1 accuracy over 10 frames", (0, 1.18))
    axes[0].set_title("Matched grounded-lexical top-1 accuracy", loc="left", fontsize=13, weight="bold")
    axes[0].legend(frameon=False, ncol=4, loc="upper left")
    grouped_bars(axes[1], margin, "Mean target - best distractor", (-0.05, 0.55))
    axes[1].axhline(0, color="#111827", linewidth=1.0)
    axes[1].set_title("Matched target-versus-distractor margin", loc="left", fontsize=13, weight="bold")
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.82, hspace=0.34)
    figure.suptitle(
        "Route 2 (T2V): Evaluation Metrics across Real and Synthetic Sample 10 Sec Clip",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    figure.text(0.02, 0.925, "Same 10 timestamps and frozen CLIP evaluator.", ha="left", va="top", fontsize=10, color="#4B5563")
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def render_cost(cost_result: dict, projection: dict, output: Path) -> dict:
    yields = projection["yield_scenarios"]
    real_costs = projection["real_fully_loaded_cost_sensitivity_USD_per_usable_hour"]
    rows = cost_result["generation_only_projection_USD_per_usable_hour"]
    row_keys = {
        1.0: "100_percent_accepted_output_yield",
        0.8: "80_percent_accepted_output_yield",
        0.5: "50_percent_accepted_output_yield",
    }
    series = (
        ("Hailuo", "hailuo"),
        ("LTX API", "ltx_api"),
        ("LTX Juno", "ltx_juno_shadow_at_4_USD_per_GPU_hour"),
    )
    per_hour = {label: rows[row_keys[1.0]][key] for label, key in series}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    x = np.arange(len(yields))
    width = 0.24
    for offset, (label, key) in zip((-width, 0, width), series):
        values = [rows[row_keys[value]][key] for value in yields]
        bars = axes[0].bar(x + offset, values, width, label=label, color=COLORS[label], edgecolor="#222222", linewidth=0.6, hatch=HATCHES[label])
        axes[0].bar_label(bars, labels=[f"${value:,.0f}" for value in values], padding=3, fontsize=8)
    axes[0].set_xticks(x, [f"{int(value * 100)}%" for value in yields])
    axes[0].set_xlabel("Accepted output yield")
    axes[0].set_ylabel("Generation cost per usable hour")
    axes[0].set_title("API charges vs Juno shadow cost", loc="left", fontsize=13, weight="bold")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    for label, _ in series:
        savings = [value - per_hour[label] for value in real_costs]
        linestyle = {"Hailuo": "--", "LTX API": "-", "LTX Juno": ":"}[label]
        axes[1].plot(real_costs, savings, label=label, color=COLORS[label], linewidth=2.4, marker="o", markersize=4, linestyle=linestyle)
        axes[1].axvline(per_hour[label], color=COLORS[label], linewidth=1.0, linestyle=linestyle, alpha=0.8)
        y = {"Hailuo": -90, "LTX API": 80, "LTX Juno": 170}[label]
        axes[1].text(per_hour[label] + 8, y, f"{label}\n${per_hour[label]:,.0f}/h", color=COLORS[label], fontsize=8, va="bottom" if y > 0 else "top")
    axes[1].axhline(0, color="#111827", linewidth=1.0)
    axes[1].set_xlabel("Assumed prospective real-data cost per usable hour")
    axes[1].set_ylabel("Real cost − synthetic generation cost")
    axes[1].set_title("Generation-only break-even sensitivity", loc="left", fontsize=13, weight="bold")
    axes[1].legend(frameon=False)
    style_axis(axes[1])

    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.14, top=0.78, wspace=0.24)
    figure.suptitle("Generation-only cost projection from one accepted 10-second clip", x=0.02, y=0.98, ha="left", fontsize=16, weight="bold")
    figure.text(0.02, 0.91, "Hailuo/LTX API use observed charges; Juno uses 14.2 allocated H100-hours per generated hour at the predeclared \\$4/GPU-hour shadow rate. Juno direct charge was \\$0.", ha="left", va="top", fontsize=9.3, color="#4B5563")
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return per_hour


def run(config_path: Path, metrics_path: Path, costs_path: Path, output_root: Path) -> None:
    assert_ignored(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())["public_single_clip_cost_projection"]
    if config["status"] != "FROZEN_BEFORE_COST_PROJECTION_RENDER":
        raise RuntimeError("E_COST_PROTOCOL_NOT_FROZEN")
    result = json.loads(metrics_path.read_text())
    costs = json.loads(costs_path.read_text())
    if costs["status"] != "EXPLORATORY_COST_PROJECTION_COMPLETE":
        raise RuntimeError("E_COST_RESULT_INCOMPLETE")
    render_performance(result["grounded_lexical_metrics"], output_root / "performance_comparison.png")
    per_hour = render_cost(costs, config["projection"], output_root / "cost_comparison.png")
    compact = {
        "status": "EXPLORATORY_COST_PROJECTION_COMPLETE",
        "accepted_request_cost_USD_per_10_seconds": {
            "Hailuo": costs["observed_per_10_seconds"]["hailuo_2K_direct_API_USD"],
            "LTX API": costs["observed_per_10_seconds"]["ltx_2_3_1080p_direct_API_USD"],
            "LTX Juno direct": costs["observed_per_10_seconds"]["ltx_2_3_juno_direct_incremental_USD"],
        },
        "generation_only_cost_USD_per_hour_at_100_percent_yield": per_hour,
        "generation_only_break_even_real_cost_USD_per_usable_hour": per_hour,
        "juno_allocated_GPU_hours_per_generated_hour": costs["juno_compute_projection"]["allocated_GPU_hours_per_generated_hour_at_100_percent_yield"],
        "juno_shadow_rate_USD_per_GPU_hour": costs["juno_shadow_cost_sensitivity"]["base_visualization_rate_USD_per_GPU_hour"],
        "unpriced_synthetic_categories": costs["unpriced_synthetic_categories"],
        "real_fully_loaded_cost_observed": False,
        "lower_cost_claim_supported": False,
    }
    (output_root / "compact_projection.json").write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(compact, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic_video_preregistration.json"))
    parser.add_argument("--metrics", type=Path, default=Path("results/synthetic_video_public_four_arm_grounding.json"))
    parser.add_argument("--costs", type=Path, default=Path("results/synthetic_video_public_cost_projection.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/public_cost_performance"))
    args = parser.parse_args()
    run(args.config, args.metrics, args.costs, args.output_root)


if __name__ == "__main__":
    main()
