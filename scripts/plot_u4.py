"""Render the locked U4 results and measure separation in saved sensory embeddings."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

try:
    from sklearn.metrics import silhouette_score
except ModuleNotFoundError as exc:
    if exc.name != "sklearn":
        raise
    silhouette_score = None


RUN = Path(__file__).resolve().parents[1] / "outputs/runs/unified_u4_eval"
OUT = RUN / "figures"
ARMS = ("V+G+B", "Shuffled body", "V+G", "V", "CLIP-Global", "BabyLLaVA-Nursery", "Text-Only LM")
COLORS = dict(zip(ARMS, ("#bd663f", "#bf907d", "#208575", "#475b88", "#795483",
                        "#987319", "#687986")))
TRANSFER = (
    ("V+G+B", "as_trained", "V+G+B as trained", COLORS["V+G+B"], ""),
    ("V+G+B", "body_removed", "V+G+B body removed", "#d59472", "//"),
    ("V+G+B", "all_extras_removed", "V+G+B video only", "#e9bda7", "xx"),
    *((arm, "as_trained", arm, COLORS[arm], "") for arm in ARMS[2:4] + ARMS[1:2] + ARMS[4:6]),
)
PANELS = (
    ("noun", "test", "U1 Nouns"),
    ("action", "pooled", "U1 Actions"),
    ("action_object", "pooled", "Action + Object"),
    ("property_object", "pooled", "Property + Object"),
    ("negation", "pooled", "Negation"),
)
GENERATION = (
    ("object", "open_vocab_accuracy", "Object\nOpen-vocab top 1"),
    ("action", "in_category_accuracy", "Action\nIn-category top 1"),
    ("property", "open_vocab_accuracy", "Property\nOpen-vocab top 1"),
    ("property_straddling", "open_vocab_accuracy", "Straddling property\nOpen-vocab top 1"),
)
CLIPS = (63, 158, 315, 630)
SEEDS = tuple(range(5))
TRAJECTORY_ARMS = ("V+G+B", "V", "CLIP-Global")
ACTIONS = ("lift", "pass", "offhand", "drink", "inspect")
ACTION_COLORS = ("#bd663f", "#208575", "#475b88", "#795483", "#987319")
MARKERS = ("o", "s", "^", "D")
EMBEDDING = "pooled_sensory"
SLATE = "#1a2332"
MUTED = "#536673"
ERROR = dict(ecolor=SLATE, lw=0.85, capsize=2.5, capthick=0.85)


def read(path):
    return json.loads(path.read_text())


def save(fig, stem):
    try:
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{stem}.{ext}", dpi=300, facecolor="white")
    finally:
        plt.close(fig)


def heading(fig, title, subtitle):
    fig.text(0.07, 0.97, title, family="serif", fontsize=24, va="top")
    fig.text(0.07, 0.87, subtitle, color=MUTED, fontsize=11)


def style_axis(ax, title, *, horizontal=False):
    ax.set_title(title, loc="left", family="serif", fontsize=15, pad=10)
    ax.tick_params(length=0, pad=5, labelsize=9)
    ax.spines["bottom"].set_color("#b9c3ca")
    ax.set_axisbelow(True)
    ax.grid(axis="x" if horizontal else "y", color="#e7ebed", lw=0.65)


def percent_axis(ax, *, horizontal=False):
    axis = ax.xaxis if horizontal else ax.yaxis
    axis.set_major_formatter(PercentFormatter(100))
    axis.set_ticks(range(0, 101, 25))
    (ax.set_xlim if horizontal else ax.set_ylim)(0, 105)


def mean_sd(rows):
    return [[row[key] for row in rows] for key in ("mean", "sd")]


def scorecard(results, prompted, word_order_pairs):
    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, 2, left=0.07, right=0.97, top=0.77, bottom=0.12,
                           wspace=0.52, hspace=0.68)
    generation = fig.add_subplot(grid[0, :])
    style_axis(generation, "(a) Prompted next-token accuracy")
    percent_axis(generation)
    x = np.arange(len(GENERATION))
    for i, arm in enumerate(ARMS):
        means, errors = mean_sd([prompted[arm]["prompted"][task][metric]
                                 for task, metric, _ in GENERATION])
        generation.bar(x + (i - 3) * 0.1, means, width=0.09, color=COLORS[arm],
                       yerr=errors, error_kw=ERROR)
    generation.set_xticks(x, [label for _, _, label in GENERATION])
    fig.legend(handles=[Patch(color=COLORS[arm], label=arm) for arm in ARMS], ncol=7,
               loc="center left", bbox_to_anchor=(0.064, 0.842), fontsize=9)

    for i, title in enumerate(("(b) Held-out perplexity", "(c) Nursery-BLiMP overall")):
        ax = fig.add_subplot(grid[1, i])
        style_axis(ax, title, horizontal=True)
        rows = [results[arm]["held_out_perplexity"] if i == 0
                else results[arm]["blimp"]["overall"] for arm in ARMS]
        means, errors = mean_sd(rows)
        bars = ax.barh(ARMS, means, xerr=errors, color=[COLORS[arm] for arm in ARMS],
                       error_kw=ERROR, height=0.63)
        ax.invert_yaxis()
        if i == 0:
            bars[-1].set_hatch("//")
            bars[-1].set_edgecolor(SLATE)
            ax.set_xlabel("Perplexity (lower is better)", fontsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(6))
        else:
            percent_axis(ax, horizontal=True)
    # Reserve the left margin for the full arm labels on the lower panels.
    for ax in fig.axes[1:]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + 0.085, pos.y0, pos.width - 0.085, pos.height])

    heading(fig, "Grounded completions and grammar", "U4  ·  630 training clips  ·  Five-seed mean ± SD")
    fig.text(0.07, 0.045,
             f"Perplexity is conditioned on the scene except for Text-Only LM (hatched). The word-order bucket has "
             f"only {word_order_pairs} pairs and varies across seeds.", color=MUTED, fontsize=9)
    save(fig, "u4_generation_grammar_scorecard")


def transfer_plot(results):
    fig, axes = plt.subplots(1, len(PANELS), sharey=True, figsize=(15, 4.8))
    fig.subplots_adjust(left=0.07, right=0.977, top=0.63, bottom=0.16, wspace=0.20)
    heading(fig, "What transfers when body cues are removed?",
            "U4  ·  630 training clips  ·  Held-out accuracy, five-seed mean ± SD")
    handles = [Patch(facecolor=color, edgecolor=SLATE if hatch else color, hatch=hatch,
                     label=label) for _, _, label, color, hatch in TRANSFER]
    fig.legend(handles=handles, ncol=4, loc="center left",
               bbox_to_anchor=(0.065, 0.778), fontsize=9)
    for ax, (panel, split, label) in zip(axes, PANELS):
        style_axis(ax, label)
        percent_axis(ax)
        for x, (arm, mode, _, color, hatch) in enumerate(TRANSFER):
            row = results[arm]["ablations"][mode][panel][split]["accuracy"]
            ax.bar(x, row["mean"], yerr=row["sd"], width=0.8, color=color, hatch=hatch,
                   edgecolor=SLATE if hatch else color, linewidth=0.5, error_kw=ERROR)
        ax.set_xticks([])
    fig.text(0.07, 0.063, "Body removed keeps gaze. Video only removes gaze and body. "
             "U1 Nouns uses test. All other panels pool test + S8.", color=MUTED, fontsize=9)
    save(fig, "u4_body_removed_transfer")


def load_embeddings(arm, clips, seed):
    path = RUN / f"{arm}_seed_{seed}_n{clips}_action_embeddings.json"
    data = read(path)
    if (data["arm"], data["n_clips"], data["seed"]) != (arm, clips, seed):
        raise ValueError(f"Checkpoint metadata disagrees with filename: {path}")
    rows = data["action_embeddings"]
    labels = np.array([row["action"] for row in rows])
    vectors = np.array([row[EMBEDDING] for row in rows])
    if set(labels) != set(ACTIONS) or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid action labels or embeddings in {path}")
    keys = [(row["split"], row["clip"], row["action"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate action clips in {path}")
    return labels, vectors, set(keys)


def separation(labels, vectors):
    """Use cosine distances in the original space, excluding self-pairs."""
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("Cosine distance is undefined for zero-norm embeddings")
    unit = vectors / norms[:, None]
    distances = np.clip(1 - unit @ unit.T, 0, 2)
    np.fill_diagonal(distances, 0)
    same = labels[:, None] == labels
    upper = np.triu_indices(len(labels), k=1)
    same_pair = same[upper]
    pair_distances = distances[upper]
    ratio = pair_distances[~same_pair].mean() / pair_distances[same_pair].mean()
    if silhouette_score is not None:
        score = silhouette_score(distances, labels, metric="precomputed")
    else:
        members = labels[:, None] == np.array(ACTIONS)
        counts = members.sum(axis=0)
        sums = distances @ members.astype(float)
        size = same.sum(axis=1)
        within = np.divide(sums[members], size - 1, out=np.zeros(len(labels)), where=size > 1)
        nearest = np.where(members, np.inf, sums / counts).min(axis=1)
        scale = np.maximum(within, nearest)
        scores = np.divide(nearest - within, scale, out=np.zeros(len(labels)),
                           where=(size > 1) & (scale > 0))
        score = scores.mean()
    return {"cosine_silhouette": float(score), "between_within_cosine_ratio": float(ratio)}


def action_trajectory():
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), subplot_kw={"projection": "3d"})
    fig.subplots_adjust(left=0.045, right=0.96, bottom=0.15, top=0.78, wspace=0.06)
    metrics = ("cosine_silhouette", "between_within_cosine_ratio")
    rows, summaries = [], []
    expected_keys = None
    for ax, arm in zip(axes, TRAJECTORY_ARMS):
        runs = {}
        for clips in CLIPS:
            for seed in SEEDS if clips == CLIPS[-1] else SEEDS[:1]:
                labels, vectors, keys = load_embeddings(arm, clips, seed)
                if expected_keys is None:
                    expected_keys = keys
                if keys != expected_keys:
                    raise ValueError(f"Held-out action clips differ for {arm}, {clips}, seed {seed}")
                if seed == 0:
                    runs[clips] = labels, vectors
                rows.append(dict(arm=arm, n_clips=clips, seed=seed, n_eval_clips=len(labels),
                                 **separation(labels, vectors)))
        final = [row for row in rows if row["arm"] == arm and row["n_clips"] == CLIPS[-1]]
        aggregate = dict(arm=arm, n_clips=CLIPS[-1], seeds=[row["seed"] for row in final])
        for metric in metrics:
            values = [row[metric] for row in final]
            aggregate[metric] = dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)))
        summaries.append(aggregate)

        pooled = np.concatenate([vectors for _, vectors in runs.values()])
        center = pooled.mean(axis=0)
        _, _, basis = np.linalg.svd(pooled - center, full_matrices=False)
        projected = [(labels, (vectors - center) @ basis[:3].T)
                     for labels, vectors in runs.values()]
        for action, color in zip(ACTIONS, ACTION_COLORS):
            groups = [points[labels == action] for labels, points in projected]
            centroids = np.array([group.mean(axis=0) for group in groups])
            ax.plot(*centroids.T, color=color, lw=2.0, zorder=3)
            for marker, group, centroid in zip(MARKERS, groups, centroids):
                ax.scatter(*group.T, marker=marker, color=color, s=8, alpha=0.13,
                           linewidths=0, depthshade=False, zorder=1)
                ax.scatter(*centroid, marker=marker, color=color, s=46, edgecolor="white",
                           linewidth=0.65, depthshade=False, zorder=4)
        ax.computed_zorder = False
        ax.set_title(arm, family="serif", fontsize=16, pad=3)
        ax.view_init(elev=23, azim=-61)
        for dimension, axis in enumerate((ax.xaxis, ax.yaxis, ax.zaxis), 1):
            axis.set_label_text(f"PC{dimension}", fontsize=9)
            axis.set_major_locator(MaxNLocator(4))
            axis.set_pane_color((0.965, 0.975, 0.98, 1))
        ax.tick_params(labelsize=7, pad=0)

    heading(fig, "Action embeddings across training sizes",
            "U4  ·  Pooled sensory embeddings, seed 0  ·  One PCA basis per arm")
    handles = [Line2D([], [], color=color, lw=2, label=action)
               for action, color in zip(ACTIONS, ACTION_COLORS)]
    handles += [Line2D([], [], color=MUTED, marker=marker, ls="", label=str(clips))
                for marker, clips in zip(MARKERS, CLIPS)]
    fig.legend(handles=handles, fontsize=9, ncol=len(handles), loc="center left",
               bbox_to_anchor=(0.064, 0.12), columnspacing=1.35)
    fig.text(0.07, 0.042, "Faint points are held-out clips. Lines join class centroids in training-size "
             "order. PCA axes are local to each arm.", color=MUTED, fontsize=9)
    save(fig, "u4_action_trajectory_3d")

    (OUT / "u4_action_separation.json").write_text(json.dumps(dict(
        embedding=EMBEDDING, actions=ACTIONS,
        distance="cosine distance in original embedding space",
        ratio="mean between-class pair distance / mean within-class pair distance (distinct clips)",
        sd="sample standard deviation across seeds",
        per_seed=rows, summary_630=summaries,
    ), indent=2, allow_nan=False) + "\n")
    print("| Arm | Training clips | Seed(s) | Cosine silhouette | Between / within |")
    print("|---|---:|---|---:|---:|")
    for arm, summary in zip(TRAJECTORY_ARMS, summaries):
        for row in rows:
            if row["arm"] == arm and row["seed"] == 0:
                values = [f"{row[metric]:.3f}" for metric in metrics]
                print(f"| {arm} | {row['n_clips']} | 0 | " + " | ".join(values) + " |")
        values = [f"{summary[m]['mean']:.3f} ± {summary[m]['sd']:.3f}" for m in metrics]
        print(f"| {arm} | {summary['n_clips']} | 0–4 | " + " | ".join(values) + " |")


def main():
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans"], "font.serif": ["Georgia", "DejaVu Serif"],
        "text.color": SLATE, "axes.labelcolor": SLATE, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False, "xtick.color": SLATE,
        "ytick.color": SLATE, "legend.frameon": False, "pdf.fonttype": 42,
    })
    summary = read(RUN / "summary.json")
    text_only = read(RUN.parent / "unified_u4_text_only/u4_eval/summary.json")
    if summary["pair_counts"] != text_only["pair_counts"]:
        raise ValueError("Text-only and multimodal Nursery-BLiMP pair counts differ")
    results = summary["arms"] | text_only["arms"]
    prompted = read(RUN / "summary_prompted.json")["arms"]
    if any(tuple(source[arm]["seeds"]) != SEEDS for source in (results, prompted) for arm in ARMS):
        raise ValueError("The locked U4 scorecard requires seeds 0–4 for every arm")
    OUT.mkdir(parents=True, exist_ok=True)
    scorecard(results, prompted, sum(summary["pair_counts"]["word_order"].values()))
    transfer_plot(results)
    action_trajectory()


if __name__ == "__main__":
    main()
