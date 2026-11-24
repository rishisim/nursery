"""Render the U2 routing and dissociation figures and the Step 9 acquisition comparison.

Run with Python, NumPy, Matplotlib and SciPy. Install Plotly to also export the
two interactive figures as standalone HTML files. All inputs are saved results.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.graph_objects as go
except ModuleNotFoundError as exc:
    if exc.name != "plotly":
        raise
    go = None


SLATE = "#1a2332"
SENSES = ("#397AA8", "#C78C32", "#D36D61")


def save(fig, out, stem):
    """Export a matching print-ready pair and release the Matplotlib canvas."""
    try:
        for extension in ("png", "pdf"):
            path = out / f"{stem}.{extension}"
            fig.savefig(path, dpi=300, facecolor="white")
            print(path)
    finally:
        plt.close(fig)


def save_html(fig, out, stem):
    """Embed the Plotly runtime so the local HTML can also be opened offline."""
    path = out / f"{stem}.html"
    fig.write_html(path, include_plotlyjs=True, config={"displaylogo": False})
    print(path)


F1_WORDS = ("light", "heavy", "lift", "pass", "drink", "not", "holding")
F1_PANELS = ("action_object", "property_object", "negation")
F1_LABELS = ("Action + object", "Property + object", "Negation")
F1_INKS = ("#b3524e", "#8f485e", "#326879", "#433b79", "#458370", "#866534", "#566779")


def f1_read(root, arm):
    """Pool contextual occurrences within seed, then weight the three seeds equally."""
    from collections import defaultdict

    seeds, examples = [], []
    for seed in range(3):
        path = root / "outputs/runs/unified_u2_v1" / arm / f"seed{seed}"
        word_rows = json.loads((path / "routing.json").read_text())
        token_rows = json.loads((path / "sentence_routing.json").read_text())
        groups, sentences = defaultdict(list), defaultdict(list)
        for row in word_rows:
            if row["kind"] == "noun":
                groups[row["word"]].append(
                    [row[k] for k in ("p_vis", "p_temp", "p_body")]
                )
        for row in token_rows:
            shares = [row[k] for k in ("p_vis", "p_temp", "p_body")]
            if row["word"] in F1_WORDS:
                groups[row["word"]].append(shares)
            if row["split"] == "test":
                sentences[row["panel"], row["clip"]].append(
                    (row["position"], row["word"], shares)
                )
        seeds.append({w: np.mean(v, axis=0) for w, v in groups.items()})
        examples.append(sentences)
    keys = seeds[0].keys()
    if any(s.keys() != keys for s in seeds) or len(set(keys) - set(F1_WORDS)) != 76:
        raise ValueError(f"Inconsistent U2 routing vocabulary in {arm}")
    per_seed = {w: np.stack([s[w] for s in seeds]) for w in keys}
    return per_seed, examples


def f1_cards(examples):
    """Use the first shared test clip for a matched three-panel illustration."""
    by_panel = [{clip for p, clip in examples[0] if p == panel} for panel in F1_PANELS]
    clip = min(set.intersection(*by_panel))
    cards = []
    for panel in F1_PANELS:
        rows = [sorted(seed[panel, clip]) for seed in examples]
        tokens = [(pos, w) for pos, w, _ in rows[0]]
        if any([(pos, w) for pos, w, _ in r] != tokens for r in rows[1:]):
            raise ValueError(f"Token mismatch in {panel}, {clip}")
        cards.append(
            [
                (w, np.mean([r[i][2] for r in rows], axis=0))
                for i, (_, w) in enumerate(tokens)
            ]
        )
    return clip, cards


def f1_xyz(probabilities):
    """Equilateral barycentric projection with analytic normalized entropy height."""
    from scipy.special import xlogy

    p = np.asarray(probabilities)
    return (
        p[..., 1] + 0.5 * p[..., 2],
        np.sqrt(0.75) * p[..., 2],
        1 + xlogy(p, p).sum(axis=-1) / np.log(3),
    )


def fig1(root, out):
    from matplotlib import colors
    from scipy.spatial import Delaunay

    aligned_seeds, examples = f1_read(root, "V+G+B")
    shuffled_seeds, shuffled_examples = f1_read(root, "V+G+B_shuffled_body")
    if aligned_seeds.keys() != shuffled_seeds.keys():
        raise ValueError("Aligned and shuffled arms have different vocabularies")
    aligned = {word: seeds.mean(axis=0) for word, seeds in aligned_seeds.items()}
    shuffled = {word: seeds.mean(axis=0) for word, seeds in shuffled_seeds.items()}
    shift = {w: (shuffled_seeds[w] - aligned_seeds[w]) * 100 for w in aligned}
    clip, cards = f1_cards(examples)
    other_clip, other_cards = f1_cards(shuffled_examples)
    if clip != other_clip or [w for w, _ in cards[1]] != [w for w, _ in other_cards[1]]:
        raise ValueError("Property example differs across training conditions")
    stem = "fig1_routing_landscape_3d"
    nouns = sorted(set(aligned) - set(F1_WORDS))
    words = sorted(F1_WORDS, key=lambda word: shift[word][:, 2].mean())
    noun_shift = np.array([shift[w][:, 2].mean() for w in nouns])
    figure = plt.figure(figsize=(12.7, 7.8), facecolor="white")
    figure.text(
        0.06,
        0.935,
        "Shuffling body cues sends weight words toward static vision",
        family="serif",
        fontsize=22,
    )
    figure.text(
        0.06,
        0.885,
        "“Lift” shifts toward temporal vision. Each row compares the same word across three matched training seeds.",
        fontsize=11,
    )
    body = figure.add_axes([0.15, 0.41, 0.355, 0.34])
    visual = figure.add_axes([0.62, 0.41, 0.32, 0.34], sharey=body)
    figure.text(0.06, 0.805, "Change in body share", family="serif", fontsize=16)
    figure.text(0.585, 0.805, "Where routing shifts", family="serif", fontsize=16)
    figure.text(0.06, 0.766, "Open circles = seeds     ◆ = mean", fontsize=9.2)
    figure.text(
        0.585, 0.766, "●  Static vision", color=SENSES[0], weight="bold", fontsize=9.5
    )
    figure.text(
        0.72, 0.766, "■  Temporal vision", color="#9a5e06", weight="bold", fontsize=9.5
    )
    y = np.arange(len(words))[::-1]
    for ax in (body, visual):
        ax.set_ylim(-0.65, len(words) - 0.5)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0, labelsize=9.2, pad=4)
        ax.axvline(0, color="#8291a0", linewidth=1)
        for yy in y:
            ax.axhline(yy, color="#e6eaee", linewidth=0.7, zorder=0)
    body.set_xlim(-80, 44)
    body.set_xticks([-80, -60, -40, -20, 0, 20, 40])
    body.set_yticks(y, words, fontfamily="monospace", fontsize=11)
    body.set_xlabel("Shuffled − aligned (percentage points)", fontsize=9.5)
    visual.set_xlim(-12, 75)
    visual.set_xticks([0, 20, 40, 60])
    visual.tick_params(labelleft=False)
    visual.set_xlabel("Shuffled − aligned (percentage points)", fontsize=9.5)
    for yy, word in zip(y, words):
        delta = shift[word]
        scores = delta[:, 2]
        body.plot([scores.min(), scores.max()], [yy, yy], color=SENSES[2], lw=2)
        body.scatter(
            scores,
            yy + np.array([-0.095, 0, 0.095]),
            s=26,
            color="white",
            edgecolors=SENSES[2],
            zorder=3,
        )
        body.scatter(
            scores.mean(),
            yy,
            s=32,
            marker="D",
            color="#993d38",
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )
        body.text(
            scores.max() + 4,
            yy,
            f"{scores.mean():+.1f}",
            va="center",
            fontsize=8.4,
            color="#873e3a",
        )
        for channel, offset, marker, ink in [
            (0, 0.145, "o", SENSES[0]),
            (1, -0.145, "s", "#ad711b"),
        ]:
            value = delta[:, channel].mean()
            visual.plot([0, value], [yy + offset] * 2, color=ink, lw=2)
            visual.scatter(value, yy + offset, s=29, marker=marker, color=ink, zorder=3)
            visual.text(
                value + 2,
                yy + offset,
                f"{value:+.1f}",
                va="center",
                fontsize=8.1,
                color=ink,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25},
            )
    reference = figure.add_axes([0.15, 0.221, 0.355, 0.064], sharex=body)
    reference.set_ylim(-0.5, 0.5)
    reference.eventplot(
        noun_shift, lineoffsets=0, linelengths=0.7, linewidths=0.85, colors="#697c90"
    )
    reference.axvline(0, color="#84919c", lw=0.65, ls="--")
    reference.set_axis_off()
    figure.text(
        0.06, 0.303, "Body shifts across 76 noun probes", family="serif", fontsize=14
    )
    figure.text(
        0.06,
        0.177,
        f"One tick per noun on the body-change scale above. Median {np.median(noun_shift):+.1f} pp.",
        fontsize=9.5,
    )
    figure.text(
        0.585,
        0.303,
        "Inside one held-out property sentence",
        family="serif",
        fontsize=14,
    )
    example = cards[1]
    figure.text(
        0.585,
        0.252,
        "“" + " ".join(w for w, _ in example) + "”",
        family="monospace",
        fontsize=12.5,
    )
    tokens = [
        (w, real, control) for (w, real), (_, control) in zip(example, other_cards[1])
    ]
    prop = next((w, a, b) for w, a, b in tokens if w in ("heavy", "light"))
    noun = tokens[-1]
    figure.text(
        0.585,
        0.204,
        f"Body share for ‘{prop[0]}’   {prop[1][2]:.1%} → {prop[2][2]:.1%}",
        fontsize=10.2,
    )
    figure.text(
        0.585,
        0.164,
        f"Vision share for ‘{noun[0]}’   {noun[1][0]:.1%} → {noun[2][0]:.1%}",
        fontsize=10.2,
    )
    figure.text(
        0.06,
        0.071,
        "Word rows pool contextual tokens from test + S8 within each seed. Nouns use single-word probes. The example uses three-seed means.",
        color="#526174",
        fontsize=9.2,
    )
    save(figure, out, stem)
    if go is None:
        return

    interactive = go.Figure()
    n = 35
    mesh = (
        np.array([(n - i - j, i, j) for i in range(n + 1) for j in range(n + 1 - i)])
        / n
    )
    x, y, z = f1_xyz(mesh)
    topology = Delaunay(np.column_stack((x, y))).simplices
    corners = np.eye(3)
    interactive.add_mesh3d(
        x=x,
        y=y,
        z=z,
        i=topology[:, 0],
        j=topology[:, 1],
        k=topology[:, 2],
        intensity=z,
        colorscale="Viridis",
        cmin=0,
        cmax=1,
        opacity=0.50,
        colorbar={
            "x": 0,
            "y": 0.68,
            "len": 0.37,
            "thickness": 10,
            "title": "Sharpness",
        },
        hoverinfo="skip",
        name="Normalized specialization",
    )
    for label, p in zip(("Vision", "Temporal", "Body"), corners):
        cx, cy, cz = f1_xyz(p)
        interactive.add_scatter3d(
            x=[cx],
            y=[cy],
            z=[cz + 0.06],
            mode="text",
            text=[label],
            showlegend=False,
            hoverinfo="skip",
        )
    for data, name, filled in (
        (aligned, "Aligned", True),
        (shuffled, "Shuffled", False),
    ):
        points = np.array([data[w] for w in nouns])
        xx, yy, zz = f1_xyz(points)
        interactive.add_scatter3d(
            x=xx,
            y=yy,
            z=zz,
            mode="markers",
            text=nouns,
            customdata=points,
            marker={
                "size": 3,
                "color": SENSES[0],
                "symbol": "circle" if filled else "circle-open",
            },
            name=f"Nouns · {name}",
            showlegend=False,
            hovertemplate=name
            + " · %{text}<br>Vision %{customdata[0]:.2%}<br>Temporal %{customdata[1]:.2%}<br>Body %{customdata[2]:.2%}<extra></extra>",
        )

    for w, ink in zip(F1_WORDS, F1_INKS):
        a, b = aligned[w], shuffled[w]
        xx, yy, zz = f1_xyz(np.linspace(a, b, 60))
        interactive.add_scatter3d(
            x=xx,
            y=yy,
            z=zz,
            mode="lines",
            line={"color": ink, "width": 5},
            name=w,
            showlegend=False,
            hoverinfo="skip",
        )
        for j, point, symbol, arm in (
            (0, a, "circle", "Aligned"),
            (-1, b, "circle-open", "Shuffled"),
        ):
            interactive.add_scatter3d(
                x=[xx[j]],
                y=[yy[j]],
                z=[zz[j]],
                mode="markers",
                marker={
                    "color": ink,
                    "symbol": symbol,
                    "size": 6,
                    "line": {"color": "white", "width": 1},
                },
                name=f"{w} · {arm}",
                showlegend=False,
                hovertemplate=(
                    f"{w} · {arm}<br>Vision {point[0]:.3%}<br>Temporal {point[1]:.3%}"
                    f"<br>Body {point[2]:.3%}<extra></extra>"
                ),
            )
    layout_axis = {"visible": False}
    interactive.update_layout(
        template="plotly_white",
        height=860,
        title="How words route across three senses",
        margin={"l": 35, "r": 45, "t": 55, "b": 25},
        scene={
            "domain": {"x": [0.06, 0.78], "y": [0.32, 1]},
            "xaxis": layout_axis,
            "yaxis": layout_axis,
            "zaxis": layout_axis,
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 0.87, "z": 0.50},
            "camera": {"eye": {"x": 0.35, "y": -1.6, "z": 0.98}},
        },
    )

    def note(xpos, ypos, label, **kwargs):
        interactive.add_annotation(
            x=xpos,
            y=ypos,
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            **kwargs,
        )

    note(
        0.79,
        0.89,
        "<b>Body share</b><br>Aligned → shuffled",
        xanchor="left",
        align="left",
    )
    for i, (w, col) in enumerate(zip(F1_WORDS, F1_INKS)):
        note(
            0.79,
            0.81 - 0.045 * i,
            f"{w}　{aligned[w][2]:.1%} → {shuffled[w][2]:.1%}",
            xanchor="left",
            font={"color": col, "family": "monospace"},
        )
    note(
        0.02,
        0.335,
        "76 nouns (single-word probes) · Contextual word means (test + S8) · Filled = aligned, open = shuffled",
        xanchor="left",
    )
    note(
        0.02,
        0.267,
        f"<b>Test clip　{clip}</b>　·　Aligned token shares in % (vision / temporal / body)",
        xanchor="left",
    )
    for j, (label, tokens) in enumerate(zip(F1_LABELS, cards)):
        ypos = 0.187 - j * 0.079
        note(0.02, ypos, label, xanchor="left", yanchor="middle")
        for i, (w, s) in enumerate(tokens):
            hue = colors.to_rgb(SENSES[np.argmax(s)])
            tint = colors.to_hex(0.16 * np.array(hue) + 0.84)
            note(
                0.215 + 0.13 * i,
                ypos,
                f"<b>{w}</b><br>" + "/".join(f"{v * 100:.2f}" for v in s),
                xanchor="center",
                yanchor="middle",
                bgcolor=tint,
                borderpad=4,
                font={"size": 11, "family": "monospace"},
            )
    save_html(interactive, out, stem)


def fig2(root, out):
    """Compare test lexical accuracy and pooled held-out composition by matched seed."""
    conditions = ("V", "V+G", "V+G+B", "V+G+B_shuffled_body", "V+G+B_bow_only")
    panels = ("action_object", "property_object", "negation")
    labels = ("Nouns", "Actions", "Action + Object", "Property + Object", "Negation")
    run_dir = root / "outputs/runs/unified_u2_v1"
    scores = np.empty((len(conditions), 3, len(labels)))
    mixed_mass = {}
    subset_sizes = set()
    for i, condition in enumerate(conditions):
        for seed in range(3):
            path = run_dir / condition / f"seed{seed}" / "metrics.json"
            data = json.loads(path.read_text())
            if data["run"]["seed"] != seed:
                raise ValueError(f"Seed metadata does not match directory in {path}")
            summary = data["summary"]
            if condition in ("V+G+B", "V+G+B_shuffled_body"):
                subset = summary["compositional"]["property_object"]["straddling"]
                subset_sizes.add(subset["n_trials"])
                mixed_mass.setdefault(condition, []).append(subset["accuracy"])
            lexical = summary["accuracy"]["as_trained"]
            scores[i, seed] = [lexical["noun"], lexical["action"]] + [
                summary["compositional"][panel]["pooled"]["accuracy"]
                for panel in panels
            ]
    if not (np.isfinite(scores) & (scores >= 0) & (scores <= 100)).all():
        raise ValueError("Expected finite percentage accuracies for all five U2 arms")

    if len(subset_sizes) != 1:
        raise ValueError("Mixed-mass trial counts differ across paired runs")

    comparisons = (
        ("Gaze", "Video + gaze minus video", scores[1] - scores[0], "#267770"),
        (
            "Body alignment",
            "Aligned minus shuffled body",
            scores[2] - scores[3],
            "#B15336",
        ),
        (
            "Word order",
            "Ordered text minus bag-of-words",
            scores[2] - scores[4],
            "#725486",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 7.65))
    fig.subplots_adjust(left=0.16, right=0.955, top=0.69, bottom=0.29, wspace=0.31)
    fig.text(
        0.045,
        0.936,
        "Gaze helps most on nouns, body alignment on mass, and word order on negation",
        family="serif",
        fontsize=21,
        color=SLATE,
    )
    fig.text(
        0.046,
        0.874,
        "Accuracy gain in percentage points across three matched training seeds at 630 clips.",
        color="#52616E",
        fontsize=12,
    )
    rows = np.arange(len(labels))[::-1]
    ticks = [-10, 0, 10, 20, 30]
    for col, (ax, (title, subtitle, deltas, color)) in enumerate(
        zip(axes, comparisons)
    ):
        means = deltas.mean(axis=0)
        std = deltas.std(axis=0, ddof=1)
        peak = means.argmax()
        bounds = ax.get_position()
        fig.text(bounds.x0, 0.788, title, fontsize=17, color=color, fontfamily="serif")
        fig.text(bounds.x0, 0.746, subtitle, fontsize=10, color="#52616E")
        ax.set_xlim(-10.5, 36)
        ax.set_ylim(-0.58, 4.58)
        ax.set_xticks(ticks)
        ax.set_yticks(rows, labels if col == 0 else [""] * len(rows))
        ax.tick_params(length=0, labelsize=10, pad=7)
        ax.spines[:].set_visible(False)
        ax.axvspan(-10.5, 0, color="#F2F4F5", zorder=0)
        ax.axvline(0, color="#8A9298", lw=0.9, zorder=1)
        for row_idx, y in enumerate(rows):
            value = means[row_idx]
            selected = row_idx == peak
            ink = color if selected else "#586878"
            # Slight vertical offsets keep seeds with identical effects visible.
            ax.hlines(y, 0, value, color=ink, alpha=0.24, lw=3, zorder=2)
            ax.scatter(
                deltas[:, row_idx],
                y + np.array([-0.15, 0, 0.15]),
                s=20,
                facecolors="white",
                edgecolors=ink,
                lw=0.9,
                zorder=3,
            )
            ax.scatter(
                value,
                y,
                s=63 if selected else 38,
                marker="D",
                color=ink,
                edgecolors="white",
                lw=0.6,
                zorder=4,
            )
            ax.text(
                1.02,
                y,
                f"{value:+.1f}",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=11,
                fontfamily="monospace",
                fontweight="bold" if selected else "normal",
                color=ink,
            )
        ax.text(
            1.02,
            4.58,
            "Mean",
            transform=ax.get_yaxis_transform(),
            color="#657280",
            ha="left",
            va="bottom",
            fontsize=9,
        )
        ax.set_xlabel("Gain (pp)", fontsize=10, color="#52616E", labelpad=8)
        fig.text(
            bounds.x0,
            0.191,
            f"{labels[peak]}   {means[peak]:+.1f} ± {std[peak]:.1f} pp",
            color=color,
            fontweight="bold",
            fontsize=11.7,
        )
    fig.text(
        0.046,
        0.123,
        f"Among nouns with both heavy and light examples ({next(iter(subset_sizes))} clips), "
        f"property accuracy was {np.mean(mixed_mass['V+G+B']):.1f}% with aligned body "
        f"and {np.mean(mixed_mass['V+G+B_shuffled_body']):.1f}% with shuffled body.",
        fontsize=12,
        color=SLATE,
    )
    fig.text(
        0.046,
        0.065,
        "Open circles show individual seeds. Diamonds show means. Highlighted gains give mean ± sample SD.",
        fontsize=10,
        color="#52616E",
    )
    fig.text(
        0.046,
        0.032,
        "Chance accuracy is 25% on the first four tasks and 50% on negation. Sentence scores pool test and S8.",
        fontsize=10,
        color="#52616E",
    )
    save(fig, out, "fig2_dissociation_fingerprint")


def f3_tokens(path):
    """Return the measured six-position Panel C posteriors, ordered by clip."""
    clips = {}
    fixed = {0: "holding", 1: "the", 3: "not", 4: "the"}
    for row in json.loads(path.read_text()):
        if row["panel"] != "negation" or row["split"] not in ("test", "s8"):
            continue
        key = (row["split"], row["clip"])
        pos = row["position"]
        slots = clips.setdefault(key, {})
        if (
            pos not in range(6)
            or pos in slots
            or fixed.get(pos, row["word"]) != row["word"]
        ):
            raise ValueError(
                f"Unexpected negation token at {key}, position {pos} in {path}"
            )
        slots[pos] = (row["word"], [row[s] for s in ("p_vis", "p_temp", "p_body")])
    if not clips or any(len(slots) != 6 for slots in clips.values()):
        raise ValueError(f"Incomplete negation sentences in {path}")
    items = [(key, [clips[key][i] for i in range(6)]) for key in sorted(clips)]
    keys = [(key, tuple(word for word, _ in slots)) for key, slots in items]
    values = np.array([[p for _, p in slots] for _, slots in items])
    if (values < 0).any() or not np.allclose(values.sum(axis=2), 1, atol=1e-5):
        raise ValueError(f"Invalid routing probabilities in {path}")
    return keys, values.mean(axis=0)


def fig3(root, out):
    """Compare scored noun swaps within the same seed and clip."""
    base = root / "outputs/runs/unified_u2_v1"
    arms = ("V+G+B", "V+G+B_shuffled_body", "V+G+B_bow_only")
    routing, outcomes = [], []
    clips = None
    for arm in arms:
        posteriors, scores = [], []
        for seed in range(3):
            seed_dir = base / arm / f"seed{seed}"
            if arm != arms[2]:
                keys, mean = f3_tokens(seed_dir / "sentence_routing.json")
                if clips is None:
                    clips = keys
                if keys != clips:
                    raise ValueError(f"Negation examples differ in {seed_dir}")
                posteriors.append(mean)
            panel = json.loads((seed_dir / "metrics.json").read_text())["held_out"][
                "compositional"
            ]["negation"]
            trials = panel["trials"]
            by_clip = {row["clip"]: row for row in trials}
            if len(by_clip) != len(trials) or set(by_clip) != {
                key[1] for key, _ in clips
            }:
                raise ValueError(f"Negation trial IDs differ in {seed_dir}")
            if any(row["scores"] is None for row in trials):
                raise ValueError(f"Incomplete vocabulary coverage in {seed_dir}")
            credits = np.array([by_clip[key[1]]["credit"] for key, _ in clips])
            if not np.isin(credits, (0, 0.5, 1)).all() or not np.isclose(
                100 * credits.mean(), panel["pooled"]["accuracy"]
            ):
                raise ValueError(
                    f"Negation trial credits disagree with summary in {seed_dir}"
                )
            scores.append(credits)
        if posteriors:
            routing.append(np.mean(posteriors, axis=0) * 100)
        outcomes.append(np.array(scores))

    aligned, shuffled, bow = outcomes
    total = aligned.size
    clean = (aligned != 0.5) & (shuffled != 0.5)
    paired_counts = [
        np.sum(clean & (aligned == 1) & (shuffled == 0)),
        np.sum(clean & (shuffled == 1) & (aligned == 0)),
        np.sum(clean & (aligned == 1) & (shuffled == 1)),
        np.sum(clean & (aligned == 0) & (shuffled == 0)),
        np.sum(~clean),
    ]
    ties = int(np.sum(bow == 0.5))
    muted, teal, gray = "#526174", "#1c756f", "#798998"
    fig = plt.figure(figsize=(13, 7.6), facecolor="white")
    fig.text(
        0.055,
        0.925,
        "Can the model tell which object is held?",
        family="serif",
        size=24,
        color=SLATE,
    )
    fig.text(
        0.055,
        0.872,
        f"Role reversal in {len(clips)} held-out clips  ·  Test + S8  ·  Three matched seeds",
        size=11,
        color=muted,
    )

    # The frozen foil swaps the two nouns, preserving every token and its count.
    _, words = clips[0]
    target = " ".join(words[:3]) + ", " + " ".join(words[3:])
    foil = f"holding the {words[5]}, not the {words[2]}"
    fig.text(
        0.055,
        0.783,
        "What changes between the choices?",
        family="serif",
        size=17,
        color=SLATE,
    )
    for y, label, sentence in ((0.722, "Target", target), (0.666, "Foil", foil)):
        fig.text(0.055, y, label, color=muted, size=10)
        fig.text(0.125, y, sentence, color=SLATE, family="monospace", size=11)
    fig.text(
        0.055, 0.567, f"{ties:,} / {total:,} ties", family="serif", size=25, color=teal
    )
    fig.text(
        0.055,
        0.517,
        f"The evaluator scores every bag-of-words comparison as a tie.\n"
        f"Every word was in vocabulary. A tie earns half credit ({100 * bow.mean():.1f}% accuracy).",
        color=muted,
        size=10,
        va="top",
        linespacing=1.55,
    )

    fig.text(
        0.526,
        0.783,
        "Who gets the same trial right?",
        family="serif",
        size=17,
        color=SLATE,
    )
    fig.text(
        0.526,
        0.738,
        f"Aligned body  {100 * aligned.mean():.1f}%     Shuffled body  {100 * shuffled.mean():.1f}%",
        size=10.5,
        color=muted,
    )
    ax = fig.add_axes([0.538, 0.458, 0.389, 0.242])
    labels = (
        "Only aligned",
        "Only shuffled",
        "Both correct",
        "Both wrong",
        "Either tied",
    )
    for i, (label, count, color) in enumerate(
        zip(labels, paired_counts, (teal, "#b26449", gray, gray, gray))
    ):
        # Counts and bars share the full paired trial population across all five rows.
        ax.barh(i, count, height=0.48, color=color)
        ax.text(
            -0.02,
            i,
            label,
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="right",
            size=9.5,
        )
        ax.text(
            count + total * 0.011,
            i,
            f"{count:,}",
            va="center",
            color=SLATE,
            size=10,
            family="monospace",
        )
    ax.set_xlim(0, max(paired_counts) * 1.16)
    ax.set_ylim(4.6, -0.6)
    ax.axis("off")

    fig.text(
        0.055,
        0.363,
        "Where each position draws sensory evidence",
        family="serif",
        size=16,
        color=SLATE,
    )
    for i, (s, c) in enumerate(zip(("Vision", "Temporal", "Body"), SENSES)):
        fig.text(0.665 + i * 0.096, 0.364, "■", color=c, size=10)
        fig.text(0.681 + i * 0.096, 0.364, s, color=muted, size=9.5)
    ax = fig.add_axes([0.171, 0.138, 0.775, 0.163])
    ax.set(xlim=(-0.5, 5.5), ylim=(-0.45, 1.55))
    ax.axis("off")
    for pos, word in enumerate(
        ("holding", "the", "held noun", "not", "the", "unheld noun")
    ):
        ax.text(
            pos, 1.48, word, va="center", ha="center", family="monospace", size=10.5
        )
    for y, values, label in zip(
        (0.84, 0.09), routing, ("Aligned body", "Shuffled body")
    ):
        ax.text(-0.51, y, label, ha="right", va="center", color=SLATE, size=10)
        for pos, shares in enumerate(values):
            left = pos - 0.44
            for share, color in zip(shares, SENSES):
                width = 0.88 * share / 100
                ax.barh(
                    y,
                    width,
                    height=0.36,
                    left=left,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                )
                left += width
    aligned_vision = routing[0][:, 0]
    fig.text(
        0.055,
        0.073,
        f"Each strip sums to 100%. With aligned body, vision receives {aligned_vision[2]:.0f}% of routing at the held noun "
        f"and {aligned_vision[5]:.0f}% at the unheld noun.",
        color=muted,
        size=10,
    )
    save(fig, out, "fig3_negation_anatomy")


def fig4(root, out):
    """Contrast paired gains at each observed budget; expose raw curves in the HTML."""
    from matplotlib.lines import Line2D
    from scipy.interpolate import PchipInterpolator

    source = root / "outputs/runs/step9_acquisition_curves/analysis/results.json"
    results = json.loads(source.read_text())["by_clip_count"]
    clips = np.array([63, 158, 315, 473, 630])
    arms = ("V", "V+G", "V+G+B")
    colors = ("#475b88", "#208575", "#bd663f")
    tasks = ("Nouns", "Actions", "Konkle real photos")
    muted = "#536673"
    fig = plt.figure(figsize=(12.8, 8.7), facecolor="white")
    fig.text(
        0.064,
        0.941,
        "Gaze lifts nouns and body lifts actions at every measured budget",
        family="serif",
        fontsize=23,
        color=SLATE,
    )
    fig.text(
        0.065,
        0.895,
        "Step 9   ·   Change in four-choice accuracy when one source of information is added",
        fontsize=11,
        color=muted,
    )
    legend = [
        Line2D([], [], color=c, marker=m, label=label, lw=1.8)
        for c, m, label in zip(
            colors[1:], ("o", "s"), ("Add gaze  (V+G − V)", "Add body  (V+G+B − V+G)")
        )
    ]
    fig.legend(
        handles=legend,
        frameon=False,
        ncol=2,
        bbox_to_anchor=(0.95, 0.859),
        loc="center right",
    )

    for index, (task, label) in enumerate(zip(("noun", "action"), tasks)):
        ax = fig.add_axes([0.075 + index * 0.478, 0.526, 0.401, 0.269])
        ax.set_title(label, loc="left", family="serif", fontsize=17, pad=11)
        ax.axhline(0, color="#788893", linewidth=1, zorder=1)
        for name, color, marker in zip(("Gaze", "Body"), colors[1:], ("o", "s")):
            gains = [results[str(n)][task]["gains"][name] for n in clips]
            values = np.array([g["pp"] for g in gains])
            lows, highs = np.array([g["ci95"] for g in gains]).T
            ax.errorbar(
                clips,
                values,
                yerr=[values - lows, highs - values],
                fmt=marker + "-",
                color=color,
                ms=6,
                mfc=color,
                mec="white",
                mew=0.9,
                lw=1.8,
                capsize=3.5,
                elinewidth=1.2,
                zorder=3,
            )
        ax.set(
            xlim=(32, 660),
            ylim=(-8, 27),
            xticks=clips,
            yticks=(-5, 0, 5, 10, 15, 20, 25),
            xlabel="Training clips",
        )
        if index == 0:
            ax.set_ylabel("Gain in accuracy (pp)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#b9c3ca")
        ax.grid(axis="y", color="#e7ebed", lw=0.65)
        ax.tick_params(length=0, pad=5)
    fig.text(
        0.065,
        0.454,
        "Whiskers show paired 95% bootstrap intervals over seeds and held-out source clips. Lines join the five measured budgets.",
        fontsize=9,
        color=muted,
    )
    fig.add_artist(
        Line2D(
            [0.065, 0.957],
            [0.426, 0.426],
            transform=fig.transFigure,
            color="#d4dcdf",
            lw=0.9,
        )
    )

    at_half = results["315"]["action"]["accuracy"]["V+G+B"]
    at_full = results["630"]["action"]["accuracy"]["V"]
    transfer = results["630"]["picture_transfer"]
    fig.text(
        0.065, 0.372, "How close does half the data get?", family="serif", fontsize=16
    )
    fig.text(
        0.065,
        0.333,
        "Action accuracy, mean ± SD across five seeds",
        color=muted,
        fontsize=10,
    )
    for i, (name, count, summary, color) in enumerate(
        [
            ("Vision + gaze + body", 315, at_half, colors[2]),
            ("Vision only", 630, at_full, colors[0]),
        ]
    ):
        y = 0.258 - 0.073 * i
        fig.text(0.065, y, name, fontsize=11, color=color)
        fig.text(0.237, y, f"{count} clips", fontsize=10, color=muted)
        fig.text(
            0.335,
            y,
            f"{summary['mean']:.1f} ± {summary['sd']:.1f}%",
            family="monospace",
            fontsize=11,
        )
    gap = at_full["mean"] - at_half["mean"]
    fig.text(
        0.065,
        0.099,
        f"The half-budget model scores {gap:.1f} pp lower.",
        fontsize=11,
        color=SLATE,
    )

    fig.text(
        0.542,
        0.372,
        "What carries over to pictures alone?",
        family="serif",
        fontsize=16,
    )
    fig.text(
        0.542,
        0.333,
        "Accuracy after 630 training clips, mean ± SD across five seeds",
        color=muted,
        fontsize=10,
    )
    fig.text(0.704, 0.282, "Rendered objects", color=muted, fontsize=10)
    fig.text(0.84, 0.282, "Real photos", color=muted, fontsize=10)
    for i, (arm, label) in enumerate(
        zip((*arms, "V+G+B shuffled"), (*arms, "Shuffled body"))
    ):
        y = 0.236 - i * 0.042
        fig.text(0.542, y, label, fontsize=10.5, color=SLATE)
        for x, domain in ((0.704, "Rendered"), (0.84, "Konkle")):
            summary = transfer[domain]["accuracy"][arm]
            fig.text(
                x,
                y,
                f"{summary['mean']:.1f} ± {summary['sd']:.1f}%",
                family="monospace",
                fontsize=9.5,
            )
    photo = transfer["Konkle"]["accuracy"]
    gaze_gain = photo["V+G"]["mean"] - photo["V"]["mean"]
    body_gain = photo["V+G+B"]["mean"] - photo["V+G"]["mean"]
    fig.text(
        0.542,
        0.046,
        f"On real photos, gaze adds {gaze_gain:.1f} pp and body adds {body_gain:.1f} pp.",
        fontsize=11,
    )
    stem = "fig4_acquisition_surface_3d"
    save(fig, out, stem)
    if go is None:
        return
    from plotly.subplots import make_subplots

    html = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}] * 3],
        subplot_titles=tasks,
        horizontal_spacing=0.035,
    )
    smooth_x = np.linspace(clips[0], clips[-1], 90)
    iso = 69.0
    for col, task in enumerate(tasks):
        panel = [results[str(n)] for n in clips]
        if col == 2:
            panel = [v["picture_transfer"]["Konkle"] for v in panel]
        else:
            panel = [v[("noun", "action")[col]] for v in panel]
        crossings = []
        for lane, (arm, color) in enumerate(zip(arms, colors)):
            stats = [p["accuracy"][arm] for p in panel]
            means = np.array([s["mean"] for s in stats])
            sds = np.array([s["sd"] for s in stats])
            curve = PchipInterpolator(clips, means)
            smooth_z = curve(smooth_x)
            # Separate curved ribbons keep the three categorical arms discrete.
            x, y = np.meshgrid(smooth_x, [lane - 0.13, lane + 0.13])
            z = np.broadcast_to(smooth_z, x.shape)
            if col == 1:
                (crossing,) = curve.solve(iso, extrapolate=False)
                crossings.append(crossing)
            html.add_trace(
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    colorscale=[[0, color], [1, color]],
                    opacity=0.38,
                    showscale=False,
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=col + 1,
            )
            html.add_trace(
                go.Scatter3d(
                    x=smooth_x,
                    y=np.full_like(smooth_x, lane),
                    z=smooth_z,
                    mode="lines",
                    line={"color": color, "width": 5},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=col + 1,
            )
            html.add_trace(
                go.Scatter3d(
                    x=clips,
                    y=[lane] * len(clips),
                    z=means,
                    customdata=sds,
                    mode="markers",
                    name=arm,
                    legendgroup=arm,
                    showlegend=col == 0,
                    marker={
                        "size": 5,
                        "color": color,
                        "line": {"color": "white", "width": 1},
                    },
                    hovertemplate=task
                    + " · "
                    + arm
                    + "<br>%{x} clips<br>%{z:.2f}% ± %{customdata:.2f} SD<extra></extra>",
                ),
                row=1,
                col=col + 1,
            )
        if col == 1:
            html.add_trace(
                go.Scatter3d(
                    x=crossings,
                    y=list(range(3)),
                    z=[iso] * 3,
                    mode="lines+markers",
                    name="69% interpolated",
                    line={"color": SLATE, "dash": "dash", "width": 3},
                    marker={"size": 3, "color": SLATE},
                    hovertemplate="69% interpolated · %{x:.0f} clips<extra></extra>",
                ),
                row=1,
                col=2,
            )
        html.update_scenes(
            xaxis={"title": "Training clips", "tickvals": clips.tolist()},
            yaxis={"title": "", "tickvals": list(range(3)), "ticktext": list(arms)},
            zaxis={"title": "Accuracy (%)", "range": [20, 88]},
            aspectmode="manual",
            aspectratio={"x": 1.6, "y": 0.8, "z": 1.1},
            camera={"eye": {"x": 1.1, "y": -2.1, "z": 1.0}},
            row=1,
            col=col + 1,
        )

    rendered = transfer["Rendered"]["accuracy"]["V+G+B"]["mean"]
    real = photo["V+G+B"]["mean"]
    for x, caption in [
        (
            0.04,
            f"<b>Action accuracy at half the budget</b><br>V+G+B at 315 clips: {at_half['mean']:.1f}% · V at 630: {at_full['mean']:.1f}%",
        ),
        (
            0.55,
            f"<b>Picture-only transfer after 630 clips</b><br>V+G+B: {rendered:.1f}% rendered · {real:.1f}% real photos<br>On real photos, adding body to V+G changes accuracy by {body_gain:+.1f} pp.",
        ),
    ]:
        html.add_annotation(
            x=x,
            y=-0.10,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            text=caption,
        )
    html.update_layout(
        title="Acquisition by training budget · Step 9, five-seed means",
        font={"color": SLATE},
        template="plotly_white",
        height=670,
        legend={"orientation": "h", "y": 1.16},
        margin={"l": 12, "r": 12, "b": 134, "t": 112},
    )
    save_html(html, out, stem)


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs/runs/unified_u2_v1/figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.serif": ["Georgia", "DejaVu Serif"],
            "font.monospace": ["DejaVu Sans Mono"],
            "font.size": 10,
            "text.color": SLATE,
            "axes.labelcolor": SLATE,
            "axes.edgecolor": "#A4AFBB",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )
    if go is None:
        print("Plotly is unavailable. Exporting the static PNG and PDF figures.")
    for render in (fig1, fig2, fig3, fig4):
        render(root, out)


if __name__ == "__main__":
    main()
