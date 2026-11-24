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
                groups[row["word"]].append([row[k] for k in ("p_vis", "p_temp", "p_body")])
        for row in token_rows:
            shares = [row[k] for k in ("p_vis", "p_temp", "p_body")]
            if row["word"] in F1_WORDS:
                groups[row["word"]].append(shares)
            if row["split"] == "test":
                sentences[row["panel"], row["clip"]].append((row["position"], row["word"], shares))
        seeds.append({w: np.mean(v, axis=0) for w, v in groups.items()})
        examples.append(sentences)
    keys = seeds[0].keys()
    if any(s.keys() != keys for s in seeds) or len(set(keys) - set(F1_WORDS)) != 76:
        raise ValueError(f"Inconsistent U2 routing vocabulary in {arm}")
    means = {w: np.mean([s[w] for s in seeds], axis=0) for w in keys}
    return means, examples


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
        cards.append([(w, np.mean([r[i][2] for r in rows], axis=0)) for i, (_, w) in enumerate(tokens)])
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
    from matplotlib import cm, colors, patches, tri
    from scipy.spatial import Delaunay

    aligned, examples = f1_read(root, "V+G+B")
    shuffled, _ = f1_read(root, "V+G+B_shuffled_body")
    clip, cards = f1_cards(examples)
    stem = "fig1_routing_landscape_3d"
    nouns = sorted(set(aligned) - set(F1_WORDS))
    n = 35
    mesh = np.array([(n - i - j, i, j) for i in range(n + 1) for j in range(n + 1 - i)]) / n
    x, y, z = f1_xyz(mesh)
    topology = Delaunay(np.column_stack((x, y))).simplices
    triangulation = tri.Triangulation(x, y, triangles=topology)
    fig = plt.figure(figsize=(13, 9))
    fig.text(0.055, 0.942, "How words route across three senses", family="serif", size=26, color=SLATE)
    fig.text(
        0.055,
        0.905,
        "Three-seed mean routing  ·  Height measures specialization from equal shares to a single sense.",
        size=11,
        color=SLATE,
    )
    ax = fig.add_axes([0.075, 0.375, 0.58, 0.50], projection="3d", computed_zorder=False)
    ax.plot_trisurf(
        triangulation,
        z,
        cmap="viridis",
        vmin=0,
        vmax=1,
        alpha=0.50,
        linewidth=0.36,
        edgecolor=(1, 1, 1, 0.25),
        antialiased=True,
        zorder=0,
    )
    ax.set_axis_off()
    ax.view_init(elev=31, azim=-78)
    ax.set_box_aspect((1, 0.87, 0.48), zoom=1.23)
    ax.set(xlim=(-0.04, 1.04), ylim=(-0.05, 0.93), zlim=(0, 1.13))
    corners = np.eye(3)
    for label, color, p in zip(("Vision", "Temporal", "Body"), SENSES, corners):
        cx, cy, cz = f1_xyz(p)
        ax.text(cx, cy, cz + 0.06, label, color=color, ha="center", size=12, weight="bold")
    cbax = fig.add_axes([0.068, 0.51, 0.009, 0.20])
    cb = fig.colorbar(
        cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap="viridis"), cax=cbax, ticks=[0, 0.5, 1]
    )
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9)
    cb.ax.set_ylabel("Specialization", rotation=90, labelpad=9)
    cb.ax.yaxis.set_label_position("left")

    interactive = go.Figure() if go is not None else None
    if interactive is not None:
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
            colorbar={"x": 0, "y": 0.68, "len": 0.37, "thickness": 10, "title": "Sharpness"},
            hoverinfo="skip",
            name="Normalized specialization",
        )
        for label, p in zip(("Vision", "Temporal", "Body"), corners):
            cx, cy, cz = f1_xyz(p)
            interactive.add_scatter3d(
                x=[cx], y=[cy], z=[cz + 0.06], mode="text", text=[label], showlegend=False, hoverinfo="skip"
            )
    for data, name, filled in ((aligned, "Aligned", True), (shuffled, "Shuffled", False)):
        points = np.array([data[w] for w in nouns])
        xx, yy, zz = f1_xyz(points)
        ax.scatter(
            xx,
            yy,
            zz,
            s=9 if filled else 8,
            c=SENSES[0] if filled else "white",
            edgecolors=SENSES[0],
            linewidths=0.45,
            alpha=0.75,
            depthshade=False,
            zorder=2,
        )
        if interactive is not None:
            interactive.add_scatter3d(
                x=xx,
                y=yy,
                z=zz,
                mode="markers",
                text=nouns,
                customdata=points,
                marker={"size": 3, "color": SENSES[0], "symbol": "circle" if filled else "circle-open"},
                name=f"Nouns · {name}",
                showlegend=False,
                hovertemplate=name
                + " · %{text}<br>Vision %{customdata[0]:.2%}<br>Temporal %{customdata[1]:.2%}<br>Body %{customdata[2]:.2%}<extra></extra>",
            )

    fig.text(0.663, 0.815, "Body share by word", family="serif", size=17, color=SLATE)
    fig.text(0.663, 0.779, "Filled  aligned     ○  shuffled", size=10, color=SLATE)
    for i, (w, ink) in enumerate(zip(F1_WORDS, F1_INKS)):
        a, b = aligned[w], shuffled[w]
        # Only endpoints are observed. Straight interpolation of probabilities curves on the entropy surface.
        xx, yy, zz = f1_xyz(np.linspace(a, b, 60))
        ax.plot(xx, yy, zz, color=ink, lw=1.7, alpha=0.85, zorder=3)
        for j, face in ((0, ink), (-1, "white")):
            ax.scatter(xx[j], yy[j], zz[j], s=72, color=ink, alpha=0.16, edgecolors="none", zorder=4)
            ax.scatter(
                xx[j],
                yy[j],
                zz[j],
                s=32,
                color=face,
                edgecolors="white" if j == 0 else ink,
                linewidths=1.05,
                depthshade=False,
                zorder=5,
            )
        row_y = 0.736 - i * 0.039
        fig.text(0.665, row_y, "●", color=ink, size=11)
        fig.text(0.689, row_y, w, family="monospace", color=SLATE, size=11)
        fig.text(0.814, row_y, f"{a[2]:.1%}  →  {b[2]:.1%}", family="monospace", color=SLATE, size=10)
        if interactive is not None:
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
            for j, point, symbol, arm in ((0, a, "circle", "Aligned"), (-1, b, "circle-open", "Shuffled")):
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
    fig.text(
        0.665,
        0.419,
        f"Blue dots and rings show {len(nouns)} nouns.\nCurves connect observed word means\nacross the two training conditions.",
        fontsize=9.6,
        color="#586270",
        linespacing=1.4,
    )
    fig.text(0.055, 0.341, "One test clip, three utterances", family="serif", size=18, color=SLATE)
    fig.text(
        0.055,
        0.312,
        f"{clip}   ·   Aligned model   ·   Token shares in % (vision / temporal / body)",
        size=10,
        color=SLATE,
    )
    for row_i, (label, tokens) in enumerate(zip(F1_LABELS, cards)):
        base = 0.233 - row_i * 0.077
        fig.text(0.056, base + 0.011, label, weight="bold", color=SLATE, size=10)
        for col, (word, shares) in enumerate(tokens):
            left = 0.23 + 0.12 * col
            hue = SENSES[np.argmax(shares)]
            fig.add_artist(
                patches.FancyBboxPatch(
                    (left, base - 0.011),
                    0.111,
                    0.053,
                    transform=fig.transFigure,
                    boxstyle="round,pad=.003,rounding_size=.005",
                    facecolor=colors.to_rgba(hue, 0.13),
                    edgecolor="none",
                )
            )
            fig.text(
                left + 0.0555, base + 0.020, word, ha="center", family="monospace", size=11, color=SLATE
            )
            values = "/".join(f"{p * 100:.2f}" for p in shares)
            fig.text(
                left + 0.0555, base - 0.001, values, ha="center", size=8.5, family="monospace", color=SLATE
            )
    fig.text(
        0.055,
        0.024,
        "Nouns use single-word probes. Compositional words pool contextual occurrences across test and S8 within each seed.",
        size=9.4,
        color="#586270",
    )
    save(fig, out, stem)
    if interactive is None:
        return
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
            x=xpos, y=ypos, xref="paper", yref="paper", text=label, showarrow=False, **kwargs
        )

    note(0.79, 0.89, "<b>Body share</b><br>Aligned → shuffled", xanchor="left", align="left")
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
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    conditions = ("V", "V+G", "V+G+B", "V+G+B_shuffled_body", "V+G+B_bow_only")
    panels = ("action_object", "property_object", "negation")
    labels = ("Nouns", "Actions", "Action + Object", "Property + Object", "Negation")
    run_dir = root / "outputs/runs/unified_u2_v1"
    scores = np.empty((len(conditions), 3, len(labels)))
    for i, condition in enumerate(conditions):
        for seed in range(3):
            path = run_dir / condition / f"seed{seed}" / "metrics.json"
            data = json.loads(path.read_text())
            if data["run"]["seed"] != seed:
                raise ValueError(f"Seed metadata does not match directory in {path}")
            summary = data["summary"]
            lexical = summary["accuracy"]["as_trained"]
            scores[i, seed] = [lexical["noun"], lexical["action"]] + [
                summary["compositional"][panel]["pooled"]["accuracy"] for panel in panels
            ]
    if not (np.isfinite(scores) & (scores >= 0) & (scores <= 100)).all():
        raise ValueError("Expected finite percentage accuracies for all five U2 arms")

    means = scores.mean(axis=1)
    sd = scores.std(axis=1, ddof=1)
    differences = np.stack([scores[1] - scores[0], scores[2] - scores[3], scores[2] - scores[4]])
    effects = differences.mean(axis=1)
    effect_sd = differences.std(axis=1, ddof=1)
    names = ("V", "V+G", "V+G+B", "Shuffled B", "Bag-of-Words")
    ink = "#34475B"
    muted = "#546777"
    grid = "#E6EAEF"
    colors = (ink, "#23817D", "#BE5946", "#80659B", "#536780")
    markers = ("o", "o", "o", "s", "D")
    effect_colors = (colors[1], colors[2], "#74588B")
    fig, (progression, forest) = plt.subplots(
        1, 2, figsize=(13.4, 7.5), gridspec_kw={"width_ratios": [1.13, 1]}
    )
    fig.subplots_adjust(left=0.142, right=0.97, top=0.741, bottom=0.225, wspace=0.115)
    fig.text(
        0.04,
        0.947,
        "The contributions of gaze, body, and word order",
        family="serif",
        fontsize=24,
        color=SLATE,
        va="top",
    )
    fig.text(
        0.041,
        0.881,
        "Five language tasks  ·  630 training clips  ·  Three matched training seeds",
        color=muted,
        fontsize=11,
    )
    fig.text(0.142, 0.808, "A    Accuracy by condition", family="serif", fontsize=15, color=SLATE)
    fig.text(
        forest.get_position().x0,
        0.808,
        "B    Paired accuracy differences",
        family="serif",
        fontsize=15,
        color=SLATE,
    )
    rows = np.arange(len(labels))[::-1]
    for ax in (progression, forest):
        ax.set_ylim(-0.50, 4.5)
        for y in rows[:-1] - 0.5:
            ax.axhline(y, color=grid, lw=0.65, zorder=0)
        ax.grid(axis="x", color=grid, lw=0.65)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", which="both", length=0, colors=muted, labelsize=10)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.set_yticks(rows)
    progression.set_yticklabels(labels, color=ink)
    progression.tick_params(axis="y", pad=12)
    progression.set_xlim(np.floor((means - sd).min() / 5) * 5 - 1, np.ceil((means + sd).max() / 5) * 5 + 1)
    progression.set_xlabel("Accuracy (%)", labelpad=10, color=ink)
    offsets = (0.25, 0.125, 0, -0.145, -0.29)
    for r, row in enumerate(rows):
        for i in (1, 2):
            progression.annotate(
                "",
                xy=(means[i, r], row + offsets[i]),
                xytext=(means[i - 1, r], row + offsets[i - 1]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": 1.5,
                    "color": colors[i],
                    "mutation_scale": 10,
                    "shrinkA": 3.2,
                    "shrinkB": 4,
                },
            )
        for i, offset in enumerate(offsets):
            progression.errorbar(
                means[i, r],
                row + offset,
                xerr=sd[i, r],
                marker=markers[i],
                markersize=5.4,
                ls="none",
                mec=colors[i],
                mfc=colors[i] if i < 3 else "white",
                ecolor=colors[i],
                elinewidth=1,
                capsize=2.2,
                zorder=3,
            )

    forest.set_yticklabels([])
    forest.axvline(0, color="#92A0AC", lw=1.1, zorder=1)
    forest.set_xlim(
        np.floor((effects - effect_sd).min() / 5) * 5 - 1, np.ceil((effects + effect_sd).max() / 5) * 5 + 1
    )
    forest.set_xlabel("Difference in accuracy (percentage points)", labelpad=10, color=ink)
    for i, (offset, color) in enumerate(zip((0.23, 0, -0.23), effect_colors)):
        for r, row in enumerate(rows):
            y = row + offset
            forest.hlines(y, 0, effects[i, r], color=color, alpha=0.23, lw=3)
            forest.errorbar(
                effects[i, r],
                y,
                xerr=effect_sd[i, r],
                fmt="o",
                ms=5,
                color=color,
                capsize=2.5,
                elinewidth=1.3,
                zorder=3,
            )
    accuracy_handles = [
        Line2D([], [], ls="none", marker=m, markersize=5, mec=c, mfc=c if i < 3 else "white")
        for i, (m, c) in enumerate(zip(markers, colors))
    ]
    progression.legend(
        accuracy_handles,
        names,
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.012),
        frameon=False,
        ncol=5,
        fontsize=8.4,
        handletextpad=0.3,
        columnspacing=0.85,
        borderaxespad=0,
    )
    effect_handles = [Line2D([], [], color=c, marker="o", markersize=4) for c in effect_colors]
    forest.legend(
        effect_handles,
        ["Gaze lift", "Real body lift", "Word-order lift"],
        loc="lower left",
        bbox_to_anchor=(0, 1.012),
        frameon=False,
        ncol=3,
        fontsize=8.5,
        handlelength=1.4,
        handletextpad=0.45,
        borderaxespad=0,
    )

    fig.text(
        0.041,
        0.104,
        f"Property + Object    {effects[1, 3]:+.1f} pp with aligned vs. shuffled body",
        fontsize=11.2,
        color=colors[2],
        fontweight="bold",
    )
    fig.text(
        0.554,
        0.104,
        f"Negation    {means[2, 4]:.1f}% → {means[4, 4]:.1f}% with Bag-of-Words",
        fontsize=11.2,
        color=effect_colors[2],
        fontweight="bold",
    )
    fig.text(
        0.041,
        0.056,
        "Intervals show ±1 sample SD across seeds. Sentence scores pool test and S8. "
        "Gaze = V+G − V. Real body = V+G+B − Shuffled B. Word order = V+G+B − Bag-of-Words.",
        color=muted,
        fontsize=8.8,
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
        if pos not in range(6) or pos in slots or fixed.get(pos, row["word"]) != row["word"]:
            raise ValueError(f"Unexpected negation token at {key}, position {pos} in {path}")
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
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Rectangle

    base = root / "outputs/runs/unified_u2_v1"
    arms = ("V+G+B", "V+G+B_shuffled_body", "V+G+B_bow_only")
    routing, accuracy = [], []
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
            summary = json.loads((seed_dir / "metrics.json").read_text())["summary"]
            result = summary["compositional"]["negation"]["pooled"]
            if result["n_trials"] != len(clips):
                raise ValueError(f"Negation accuracy and routing sample counts differ in {seed_dir}")
            scores.append(result["accuracy"])
        if posteriors:
            routing.append(np.mean(posteriors, axis=0) * 100)
        accuracy.append(np.mean(scores))

    fig = plt.figure(figsize=(13.6, 7.0), facecolor="white")
    muted = "#526174"
    fig.text(0.055, 0.916, "How the model reads negation", fontsize=26, color=SLATE, fontfamily="serif")
    fig.text(
        0.056,
        0.862,
        f"Panel C  ·  {len(clips)} held-out clips (test + S8)  ·  3 seeds",
        color=muted,
        fontsize=12,
    )
    ax = fig.add_axes([0.055, 0.16, 0.625, 0.60])
    ax.set_xlim(-1.15, 6)
    ax.set_ylim(-0.08, 8.36)
    ax.axis("off")
    positions = ("holding", "the", "{held_X}", "not", "the", "{unheld_Y}")
    for i, label in enumerate(positions):
        ax.text(
            i + 0.5,
            8.12,
            label,
            ha="center",
            va="center",
            color=SLATE,
            fontfamily="monospace",
            fontsize=10.7,
        )
    for arm_label, values, top in zip(("V+G+B", "Shuffled B"), routing, (6.9, 2.93)):
        ax.text(-1.14, top + 0.23, arm_label, color=SLATE, fontweight="bold", fontsize=12)
        for j, sense in enumerate(("Vision", "Temporal", "Body")):
            bottom = top - (j + 1) * 0.92
            color = np.array(to_rgb(SENSES[j]))
            ax.text(-0.13, bottom + 0.43, sense, ha="right", va="center", fontsize=10.5, color=muted)
            for i, value in enumerate(values[:, j]):
                alpha = 0.06 + 0.72 * value / 100
                ax.add_patch(
                    Rectangle(
                        (i + 0.025, bottom + 0.03),
                        0.95,
                        0.80,
                        facecolor=(1 - alpha) + color * alpha,
                        edgecolor="none",
                    )
                )
                ax.text(
                    i + 0.5,
                    bottom + 0.43,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color=SLATE,
                    fontfamily="monospace",
                    fontsize=10.8,
                )
    fig.text(
        0.055,
        0.094,
        "Mean routing share (%) by token position. Each column sums to 100% before rounding.",
        fontsize=10.3,
        color=muted,
    )

    ax = fig.add_axes([0.726, 0.27, 0.224, 0.42])
    ax.set_title("Negation accuracy", loc="left", fontsize=16, fontfamily="serif", color=SLATE, pad=17)
    ax.set_xlim(0, 102)
    ax.set_ylim(-0.47, 2.65)
    ax.axvline(50, color="#9aa6ad", ls=(0, (2, 3)), lw=1, zorder=0)
    for y, label, value, color in zip(
        (2, 1, 0), ("V+G+B", "Shuffled B", "Bag-of-words"), accuracy, ("#2a626a", "#75919b", "#a8b3bb")
    ):
        ax.barh(y, value, height=0.14, color=color)
        ax.text(0, y + 0.18, label, color=SLATE, fontsize=11)
        ax.text(
            value + 2, y, f"{value:.1f}%", va="center", fontfamily="monospace", fontsize=10.3, color=SLATE
        )
    ax.set_xticks((0, 50, 100), ("0", "50", "100%"))
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=muted, labelsize=9, length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#d9dfe3")
    fig.text(0.726, 0.168, "Dashed line marks two-choice chance.", color=muted, fontsize=10)
    save(fig, out, "fig3_negation_anatomy")


def fig4(root, out):
    """Step 9 five-seed acquisition means; interpolate only along clip count."""
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    from scipy.interpolate import PchipInterpolator

    source = root / "outputs/runs/step9_acquisition_curves/analysis/results.json"
    results = json.loads(source.read_text())["by_clip_count"]
    clips = np.array([63, 158, 315, 473, 630])
    smooth_x = np.linspace(clips[0], clips[-1], 90)
    arms = ("V", "V+G", "V+G+B")
    colors = ("#475b88", "#208575", "#ca7950")
    tasks = ("Nouns", "Actions", "Konkle real photos")
    fig = plt.figure(figsize=(14, 6.9), facecolor="white")
    fig.text(
        0.045,
        0.936,
        "Language acquisition across training budgets",
        family="serif",
        fontsize=25,
        color=SLATE,
    )
    fig.text(
        0.045,
        0.875,
        "Step 9  ·  Four-choice accuracy across five seeds. Dots mark measured budgets.",
        fontsize=11,
        color="#4e6069",
    )
    handles = [Line2D([], [], color=c, marker="o", lw=2.5, label=a) for c, a in zip(colors, arms)]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.965, 0.923),
        frameon=False,
        ncol=3,
        handlelength=1.35,
        columnspacing=1.3,
    )
    if go is not None:
        from plotly.subplots import make_subplots

        html = make_subplots(
            rows=1, cols=3, specs=[[{"type": "scene"}] * 3], subplot_titles=tasks, horizontal_spacing=0.035
        )

    iso = 69.0
    for col, task in enumerate(tasks):
        panel = [results[str(n)] for n in clips]
        if col == 2:
            panel = [v["picture_transfer"]["Konkle"] for v in panel]
        else:
            panel = [v[("noun", "action")[col]] for v in panel]
        ax = fig.add_axes(
            [0.022 + 0.325 * col, 0.285, 0.315, 0.525], projection="3d", computed_zorder=False
        )
        ax.set_title(task, fontsize=15, family="serif", color=SLATE, pad=1)
        ax.view_init(elev=19, azim=-63)
        ax.set_box_aspect((1.65, 0.78, 1.04))
        ax.set(
            xlim=(55, 655),
            ylim=(-0.35, 2.35),
            zlim=(20, 88),
            xticks=clips,
            yticks=[],
            zticks=[25, 45, 65, 85],
        )
        ax.set_xlabel("Training clips", fontsize=9, labelpad=2)
        if col == 0:
            ax.set_zlabel("Accuracy (%)", fontsize=9, labelpad=0)
        ax.tick_params(labelsize=8, pad=0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((0.96, 0.97, 0.97, 0.45))
            axis.pane.set_edgecolor("#edf0f1")
            axis.line.set_color("#b5c1c4")
            axis._axinfo["grid"].update(color=(0.7, 0.75, 0.76, 0.35), linewidth=0.5)
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
            ax.plot_surface(
                x,
                y,
                z,
                color=to_rgba(color, 0.35),
                shade=False,
                edgecolor=(1, 1, 1, 0.25),
                linewidth=0.35,
                ccount=20,
                zorder=3 - lane * 0.2,
            )
            ax.plot(smooth_x, np.full_like(smooth_x, lane), smooth_z, color=color, lw=2.3, zorder=4)
            ax.scatter(
                clips,
                np.full(len(clips), lane),
                means,
                color=color,
                s=28,
                edgecolor="white",
                linewidth=1,
                depthshade=False,
                zorder=5,
            )
            ax.scatter(
                clips[-1], lane, means[-1], s=90, facecolor="none", edgecolor=color, linewidth=0.9, zorder=5
            )
            if col == 1:
                (crossing,) = curve.solve(iso, extrapolate=False)
                crossings.append(crossing)
                if arm in ("V", "V+G+B"):
                    idx = -1 if arm == "V" else 2
                    ax.scatter(
                        clips[idx],
                        lane,
                        means[idx],
                        s=140,
                        facecolor="none",
                        edgecolor=SLATE,
                        lw=1.2,
                        zorder=6,
                    )
            if go is not None:
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
                        marker={"size": 5, "color": color, "line": {"color": "white", "width": 1}},
                        hovertemplate=task
                        + " · "
                        + arm
                        + "<br>%{x} clips<br>%{z:.2f}% ± %{customdata:.2f} SD<extra></extra>",
                    ),
                    row=1,
                    col=col + 1,
                )
        if col == 1:
            ax.plot(crossings, range(3), [iso] * 3, ls="--", lw=1.2, color=SLATE, zorder=4.5)
            if go is not None:
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
        if go is not None:
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

    at_half = results["315"]["action"]["accuracy"]["V+G+B"]["mean"]
    at_full = results["630"]["action"]["accuracy"]["V"]["mean"]
    transfer = results["630"]["picture_transfer"]
    rendered, real = [transfer[k]["accuracy"]["V+G+B"]["mean"] for k in ("Rendered", "Konkle")]
    cards = [
        (
            0.04,
            0.49,
            "Action accuracy at half the training budget",
            f"{at_half:.1f}% at 315 clips    /    {at_full:.1f}% at 630 clips",
            "V+G+B versus V. The dashed guide connects interpolated 69% crossings.",
        ),
        (
            0.55,
            0.41,
            "Picture-only transfer after 630 training clips",
            f"{rendered:.1f}% rendered    /    {real:.1f}% real photos",
            "V+G+B training. Both test panels use pictures alone.",
        ),
    ]
    for left, width, title, values, note in cards:
        card = FancyBboxPatch(
            (left, 0.042),
            width,
            0.188,
            boxstyle="round,pad=.004,rounding_size=.012",
            transform=fig.transFigure,
            facecolor="#f1f5f4",
            edgecolor="none",
        )
        fig.add_artist(card)
        fig.text(left + 0.015, 0.177, title, family="serif", fontsize=13, color=SLATE)
        fig.text(left + 0.015, 0.120, values, family="monospace", fontsize=13, color=SLATE)
        fig.text(left + 0.015, 0.071, note, fontsize=8.4, color="#475c61")
    stem = "fig4_acquisition_surface_3d"
    save(fig, out, stem)
    if go is not None:
        for left, _, title, values, note in cards:
            html.add_annotation(
                x=left,
                y=-0.10,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                align="left",
                showarrow=False,
                text=f"<b>{title}</b><br>{values}<br>{note}",
            )
        html.update_layout(
            title="Language acquisition across training budgets · Step 9, five-seed means",
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
