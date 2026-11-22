"""Verify and analyze Step 9 saved predictions on CPU."""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ANALYSIS = Path(__file__).resolve().parent
ROOT = ANALYSIS.parent
FROZEN = ROOT / "frozen"
REPO = ROOT.parents[2]
STEP7 = ROOT.parent / "step7_word_types" / "analysis" / "analyze.py"
LADDER = {0.1: 63, 0.25: 158, 0.5: 315, 0.75: 473, 1.0: 630}
PICTURES = {"picture_test": "Rendered", "konkle_test": "Konkle"}
RUNGS = ("noun", "action")
SEEDS = list(range(5))

# Load Step 7's frozen panel and scoring routines without creating a cache there.
spec = importlib.util.spec_from_file_location("step7_analysis", STEP7)
step7 = importlib.util.module_from_spec(spec)
previous = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    spec.loader.exec_module(step7)
finally:
    sys.dont_write_bytecode = previous
step7.FROZEN = FROZEN
ARMS, LABELS, MODES = step7.ARMS, step7.LABELS, step7.MODES


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite number: {value}")
    return number


def read_json(path):
    return json.loads(path.read_bytes(), parse_float=finite_float, parse_constant=step7.reject_nonfinite)


def verify_selection(metrics, protocol, uses_body):
    cap = protocol["epochs"]
    panel = protocol["selection"]["panel"]
    epochs = list(range(1, cap + 1))
    history = metrics["history"]
    assert [row["epoch"] for row in history] == epochs
    assert len(metrics["epoch_durations_seconds"]) == cap
    for row in history:
        scores = row["dev_by_rung"]
        assert row["word_type_dev_accuracy"] == (scores["noun"] + scores["action"]) / 2

    sweep = metrics["dev_body_weight_table"]
    if uses_body:
        grid = protocol["selection"]["body_weight_grid"]
        assert [(row["epoch"], row["body_weight"]) for row in sweep] == [
            (epoch, weight) for epoch in epochs for weight in grid
        ]
        for row in sweep:
            scores = row["accuracy_by_rung"]
            assert row["accuracy"] == (scores["noun"] + scores["action"]) / 2
        for i, row in enumerate(history):
            epoch_sweep = sweep[i * len(grid):(i + 1) * len(grid)]
            best = min(epoch_sweep, key=lambda r: (-r["accuracy"], r["body_weight"]))
            assert (row["word_type_dev_accuracy"], row["dev_by_rung"]) == (
                best["accuracy"], best["accuracy_by_rung"]
            )
        best = min(sweep, key=lambda r: (-r["accuracy"], r["body_weight"], r["epoch"]))
        expected = dict(epoch=best["epoch"], weight=best["body_weight"],
                        dev_score=best["accuracy"], panel=panel)
    else:
        assert not sweep
        best = max(history, key=lambda r: r["word_type_dev_accuracy"])
        expected = dict(epoch=best["epoch"], dev_score=best["word_type_dev_accuracy"], panel=panel)
    assert metrics["selection"] == {"body_present": expected}
    return expected


def picture_panels(protocol):
    panels = {}
    assert set(protocol["picture_panels"]) == set(PICTURES)
    for name in PICTURES:
        path = FROZEN / "panels" / name / "trials.json"
        assert step7.sha256(path) == protocol["panel_hashes"][str(path.relative_to(FROZEN))]
        trials = read_json(path)["trials"]
        panels[name] = dict(count=len(trials), queries=[row["query"] for row in trials],
                            correct_slots=[row["correct_slot"] for row in trials],
                            candidate_rows=[row["candidate_rows"] for row in trials])
    return panels


def verify_runs(protocol, panels, pictures):
    assert protocol["epochs"] == 60 and protocol["temporal_video"] is True
    assert protocol["temporal_weight_init"] == "per_word"
    assert protocol["selection"]["word_type_rungs"] == list(RUNGS)
    assert protocol["selection"]["panel"] == "equal mean of noun and action dev accuracies"
    assert protocol["selection"]["test_panels_select"] is False
    assert protocol["fractions"] == list(LADDER) and protocol["seeds"] == SEEDS
    assert protocol["conditions"] == list(ARMS)
    assert protocol["unknown_query_score"] == 0.25
    subsets = [protocol["subsets"][str(f)]["selected_clips"] for f in LADDER]
    for (fraction, count), clips in zip(LADDER.items(), subsets, strict=True):
        subset = protocol["subsets"][str(fraction)]
        assert subset["selected_clip_count"] == len(set(clips)) == count
    assert all(set(a) <= set(b) for a, b in zip(subsets, subsets[1:]))

    n_arms = {f: 4 if amount in (158, 630) else 3 for f, amount in LADDER.items()}
    queue = read_json(FROZEN / "queue.json")
    assert queue == [[f, seed, arm] for f in LADDER for seed in SEEDS for arm in ARMS[:n_arms[f]]]
    run_root = FROZEN / "models"
    expected = {run_root / str(f) / f"seed_{seed}" / arm / "metrics.json" for f, seed, arm in queue}
    assert len(expected) == 85
    actual = set(run_root.rglob("metrics.json"))
    assert expected == actual, (sorted(expected - actual), sorted(actual - expected))

    frozen_hashes = read_json(FROZEN / "frozen_inputs.json")["prepared_hashes"]
    for name in ("protocol.json", "queue.json"):
        path = FROZEN / name
        assert step7.sha256(path) == frozen_hashes[str(path)]
    protocol_hash = step7.sha256(FROZEN / "protocol.json")
    scores = {f: {name: np.empty((len(MODES), len(SEEDS), n_arms[f], panel["count"]))
                  for name, panel in panels.items()} for f in LADDER}
    transfer = {f: {name: np.empty((len(SEEDS), n_arms[f])) for name in pictures} for f in LADDER}
    selections = {f: {arm: [] for arm in ARMS[:n_arms[f]]} for f in LADDER}
    vocabs = {f: read_json(FROZEN / "subsets" / str(f) / "vocab.json") for f in LADDER}
    action_names = list(protocol["word_type_panels"]["action"]["held_out"].values())
    action_mask = np.concatenate([panels[name]["subset"] for name in action_names])
    action_words = np.concatenate([panels[name]["queries"] for name in action_names])
    unknown_actions = {
        f: ~np.isin(action_words, list(vocab)) for f, vocab in vocabs.items()
    }

    for fraction, seed, arm in queue:
        run = run_root / str(fraction) / f"seed_{seed}" / arm
        try:
            metrics = read_json(run / "metrics.json")
            status = read_json(run / "status.json")
            for record in (metrics, status):
                assert (record["fraction"], record["seed"], record["condition"]) == (fraction, seed, arm)
                assert record["protocol_sha256"] == protocol_hash
                assert record["status"] == "complete" and record["epochs"] == protocol["epochs"]
            assert not metrics["smoke"] and not metrics["development_only"]
            assert metrics["data"] == protocol["subsets"][str(fraction)]
            temporal = read_json(run / "temporal_weights.json")
            assert temporal.keys() == vocabs[fraction].keys()
            assert all(isinstance(value, (int, float)) for value in temporal.values())
            assert all(math.isfinite(metrics[f"mean_{rung}_temporal_weight"]) for rung in RUNGS)

            i = ARMS.index(arm)
            selection = verify_selection(metrics, protocol, uses_body=i >= 2)
            selections[fraction][arm].append({"seed": seed, **selection})
            weight = selection.get("weight", 0.0)
            for mode_i, mode in enumerate(MODES):
                removed = mode_i == 1
                for rung, rung_spec in protocol["word_type_panels"].items():
                    for split, name in rung_spec["held_out"].items():
                        saved = metrics[mode][rung][split]
                        body_mode = "none" if removed or i < 2 else protocol["arms"][arm]["body_mode"]
                        gaze_mode = None if i == 0 else ("removed" if removed else "real")
                        assert saved["body_mode"] == body_mode and saved.get("gaze_mode") == gaze_mode
                        assert saved.get("body_weight", 0.0) == (0.0 if removed else weight)
                        credits = step7.score_panel(saved, panels[name], name, protocol["unknown_query_score"])
                        assert math.isclose(saved["accuracy"], 100 * credits.mean(), abs_tol=1e-12)
                        scores[fraction][name][mode_i, seed, i] = credits
                    if rung == "action":
                        joined = np.concatenate(
                            [scores[fraction][name][mode_i, seed, i] for name in action_names]
                        )
                        for name, credits in (("pooled", joined), ("object_matched", joined[action_mask])):
                            saved = metrics[mode][rung][name]
                            np.testing.assert_array_equal(credits, saved["credits"])
                            assert math.isclose(saved["accuracy"], 100 * credits.mean(), abs_tol=1e-12)
                        np.testing.assert_array_equal(joined[unknown_actions[fraction]], 0.25)
            if i == 0:
                for data in scores[fraction].values():
                    np.testing.assert_array_equal(data[0, seed, i], data[1, seed, i])

            assert set(metrics["picture"]) == set(pictures)
            for name, panel in pictures.items():
                # Picture metrics use frozen row order and omit the clip-panel identifiers.
                saved = dict(metrics["picture"][name], trial_panels=[name] * panel["count"],
                             trial_indices=list(range(panel["count"])))
                credits = step7.score_panel(saved, panel, name, protocol["unknown_query_score"])
                assert math.isclose(saved["accuracy"], 100 * credits.mean(), abs_tol=1e-12)
                transfer[fraction][name][seed, i] = 100 * credits.mean()
        except (AssertionError, KeyError, ValueError) as error:
            raise ValueError(f"Verification failed in {run.relative_to(FROZEN)}: {error}") from error

    verification = dict(
        metrics_count=len(expected), protocol_sha256=protocol_hash,
        queue_sha256=step7.sha256(FROZEN / "queue.json"), epochs=protocol["epochs"],
        temporal_weight_init=protocol["temporal_weight_init"], selection=protocol["selection"]["panel"],
        metrics_finite=True, late_epoch_threshold=protocol["epochs"] - 4,
    )
    for fraction, amount in LADDER.items():
        present = selections[fraction]
        late = [(LABELS[ARMS.index(arm)], row["seed"], row["epoch"])
                for arm, rows in present.items() for row in rows if row["epoch"] >= 56]
        print(f"{amount} clips: {sum(map(len, present.values()))} verified, late selections {late}", flush=True)
    return verification, scores, transfer, selections


def arm_stats(per_seed, labels):
    return {
        label: {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                "per_seed": values.tolist()}
        for label, values in zip(labels, per_seed.T, strict=True)
    }


def summarize(scores, clusters, rng):
    """Step 7 joint bootstrap with the available columns at this clip count."""
    primary, removed = scores
    n_seeds, n_arms, n_trials = primary.shape
    cluster_ids, index, sizes = np.unique(clusters, return_inverse=True, return_counts=True)
    n_clusters = len(cluster_ids)
    sums = np.zeros((n_clusters, n_seeds * n_arms))
    np.add.at(sums, index, primary.reshape(-1, n_trials).T)

    seed_indices = rng.integers(n_seeds, size=(step7.DRAWS, n_seeds))
    cluster_counts = rng.multinomial(
        n_clusters, np.full(n_clusters, 1 / n_clusters), size=step7.DRAWS
    )
    by_cluster = (cluster_counts.astype(float) @ sums) / (cluster_counts @ sizes)[:, None]
    by_cluster = by_cluster.reshape(step7.DRAWS, n_seeds, n_arms)
    joint = by_cluster[np.arange(step7.DRAWS)[:, None], seed_indices].mean(axis=1)

    labels = LABELS[:n_arms]
    accuracy = arm_stats(primary.mean(axis=-1) * 100, labels)
    pairs = [(1, 0), (2, 1), (2, 0), (3, 1)][:n_arms]
    gains = {}
    for label, (left, right) in zip(step7.GAINS, pairs):
        gains[label] = {
            "pp": accuracy[labels[left]]["mean"] - accuracy[labels[right]]["mean"],
            "ci95": np.quantile((joint[:, left] - joint[:, right]) * 100, [.025, .975]).tolist(),
        }
    return {
        "trials": n_trials, "clusters": n_clusters, "accuracy": accuracy, "gains": gains,
        "all_extra_inputs_removed": arm_stats(removed.mean(axis=-1) * 100, labels),
    }


def fraction_results(protocol, panels, scores):
    # Restart at each amount and preserve the Step 7 ordering through the action subsets.
    rng = np.random.default_rng(step7.BOOTSTRAP_SEED)
    half_names = protocol["word_type_panels"]["action"]["held_out"]
    noun_name = protocol["word_type_panels"]["noun"]["held_out"]["test"]
    noun = summarize(scores[noun_name], panels[noun_name]["clusters"], rng)
    names = list(half_names.values())
    full = np.concatenate([scores[name] for name in names], axis=-1)
    clusters = np.concatenate([panels[name]["clusters"] for name in names])
    flags = np.concatenate([panels[name]["subset"] for name in names])
    words = np.concatenate([panels[name]["queries"] for name in names])
    action = summarize(full, clusters, rng)
    action["object_matched"] = summarize(full[..., flags], clusters[flags], rng)
    action["halves"] = {}
    for split, name in half_names.items():
        part_clusters = np.array(panels[name]["clusters"])
        mask = np.array(panels[name]["subset"])
        half = summarize(scores[name], part_clusters, rng)
        half["object_matched"] = summarize(scores[name][..., mask], part_clusters[mask], rng)
        action["halves"][split] = half
    action["by_word"] = {}
    for word in sorted(set(words)):
        mask = words == word
        action["by_word"][word] = summarize(full[..., mask], clusters[mask], rng)
    return {"noun": noun, "action": action}


def historical_comparisons(protocol, rows):
    step7_doc = "docs/step_7_word_types.md"
    step7_frozen = STEP7.parents[1] / "frozen"
    prior_protocol = read_json(step7_frozen / "protocol.json")
    for rung in RUNGS:
        for name in protocol["word_type_panels"][rung]["held_out"].values():
            for filename in ("manifest.parquet", "trials.json"):
                key = f"panels/{name}/{filename}"
                assert step7.sha256(step7_frozen / key) == protocol["panel_hashes"][key]
    assert protocol["subsets"]["1.0"] == prior_protocol["subsets"]["1.0"]
    published = (REPO / step7_doc).read_text().split("## Completed results", 1)[1]
    step7_rows = {}
    for rung, heading in (("noun", "Nouns (test)"), ("action", "Actions (test + s8)")):
        line, = (line for line in published.splitlines() if line.startswith(f"| {heading} |"))
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        old_accuracy = dict(zip(LABELS, map(float, cells[1:5]), strict=True))
        old_gains = {}
        for label, cell in zip(step7.GAINS[:3], cells[5:], strict=True):
            estimate, bounds = cell.replace("−", "-").split("[")
            old_gains[label] = dict(pp=float(estimate), ci95=list(map(float, bounds[:-1].split(","))))
        row = rows["630"][rung]
        step7_rows[rung] = {
            "published_accuracy": old_accuracy, "published_gains": old_gains,
            "accuracy_delta_pp": {label: row["accuracy"][label]["mean"] - old_accuracy[label]
                                  for label in LABELS},
            "gain_delta_pp": {label: row["gains"][label]["pp"] - old_gains[label]["pp"]
                              for label in old_gains},
        }

    step6_source = "outputs/runs/step6_acquisition_curves/analysis/summary.json"
    step6 = read_json(REPO / step6_source)
    for filename in ("manifest.parquet", "trials.json"):
        key = f"panels/clips_test/{filename}"
        assert step6["frozen_identity_sha256"][key] == protocol["panel_hashes"][key]
    step6_rows = {}
    for fraction, amount in LADDER.items():
        old = step6["fractions"][str(fraction)]
        assert old["source_clips"] == amount
        panel = old["panels"]["clips_test"]
        accuracy = {label: {"mean": panel[arm]["primary"]["mean"], "sd": panel[arm]["primary"]["seed_sd"]}
                    for arm, label in zip(ARMS[:3], LABELS[:3], strict=True)}
        new = rows[str(amount)]["noun"]["accuracy"]
        paired = old["paired_differences"]["clips_test"]
        old_gains = {
            label: {"pp": paired[key]["mean"], "ci95": paired[key]["clip_bootstrap"]["percentile_95_ci"]}
            for label, key in zip(step7.GAINS[:3], ("gaze−video", "body−gaze", "total"), strict=True)
        }
        step6_rows[str(amount)] = {
            "accuracy": accuracy, "gains": old_gains,
            "accuracy_delta_pp": {label: new[label]["mean"] - v["mean"] for label, v in accuracy.items()},
        }
    return {"step7": {"source": step7_doc, "at_630": step7_rows},
            "step6": {"source": step6_source, "noun_curve": step6_rows}}


def accuracy_cells(stats, labels=LABELS[:3]):
    return [f"{stats[label]['mean']:.2f} ± {stats[label]['sd']:.2f}" if label in stats else "—"
            for label in labels]


def gain_cells(row, labels=step7.GAINS[:3]):
    return step7.intervals([row["gains"][label]["pp"] for label in labels],
                           [row["gains"][label]["ci95"] for label in labels], signed=True)


def report(results):
    rows = results["by_clip_count"]
    verified = results["verification"]
    table = step7.table
    final_increase = {
        label: rows["630"]["action"]["accuracy"][label]["mean"]
        - rows["473"]["action"]["accuracy"][label]["mean"]
        for label in ("V", "V+G+B")
    }
    lines = [
        "# Step 9 acquisition curves", "",
        f"Verified all {verified['metrics_count']} metrics and completion records against the frozen queue "
        "and protocol hash. Each run completed 60 epochs with per-word temporal weights and "
        "noun+action dev selection. Recomputing selection from dev histories and body-weight sweeps "
        "reproduced every selected epoch and weight. All numeric values in the metrics are finite. "
        "Saved predictions, trial credits and aggregate accuracies agree with the frozen panels.", "",
        "Accuracy is mean ± sample SD (%) over seeds 0–4. Gains are percentage points with paired "
        f"joint percentile 95% intervals ({step7.DRAWS:,} draws, RNG seed {step7.BOOTSTRAP_SEED} "
        "restarted at each clip count). As in Step 7, each draw resamples five seed IDs and the "
        "observed number of held-out source clips with replacement, sharing both draws across arms. "
        "All trials in each sampled clip are kept. Pooled action panels weight trials equally. "
        "Gaze = V+G − V, body = V+G+B − V+G, total = V+G+B − V.", "",
        "## Reading the curves", "",
        "Aligned body improves pooled action accuracy at every clip count, including 63 clips. "
        "The gain also holds for object-matched actions at every amount. Gaze improves noun "
        "accuracy throughout the ladder, and every noun body-gain interval includes zero.", "",
        f"From 473 to 630 clips, action accuracy rises by {final_increase['V']:.2f} pp for V "
        f"and {final_increase['V+G+B']:.2f} pp for V+G+B. Video-only action accuracy is still rising at the largest budget, "
        "so the observed modality gain leaves the ceiling question open. The next step toward "
        "testing a ceiling is measuring these arms beyond 630 clips. The low-data action results "
        "leave Step 8's wordless-pretraining trigger unmet.", "",
    ]
    for rung, title in (("noun", "Nouns, clips_test"), ("action", "Actions, test + s8 pooled")):
        panel = rows["630"][rung]
        lines += [f"## {title}", "", f"{panel['trials']:,} trials across {panel['clusters']} source clips.", ""]
        lines += table(["Clips", *LABELS[:3], *step7.GAINS[:3]],
                       [[amount, *accuracy_cells(row[rung]["accuracy"]), *gain_cells(row[rung])]
                        for amount, row in rows.items()])

    lines += ["## Shuffled body", "", "Gain is shuffled V+G+B minus V+G.", ""]
    lines += table(["Clips", "Rung", LABELS[3], "Shuffled body gain"],
                   [[amount, rung.title(), *accuracy_cells(row[rung]["accuracy"], LABELS[3:]),
                     *gain_cells(row[rung], ("Shuffled body",))]
                    for amount, row in rows.items() if LABELS[3] in row["noun"]["accuracy"]
                    for rung in RUNGS])

    lines += ["## Dev-selected body weights", ""]
    lines += table(["Clips", "Arm", *[f"Seed {s}" for s in SEEDS]],
                   [[amount, label, *[f"{r['weight']:.2f}" for r in row["selection"][label]]]
                    for amount, row in rows.items() for label in LABELS[2:] if label in row["selection"]])

    lines += ["## Selected epochs", "",
              "Epochs 56–60 flag possible under-training. Counts include the shuffled arm where queued.", ""]
    lines += table(["Clips", "Models", "Selected epoch range", "Late selections"],
                   [[amount, sum(map(len, row["selection"].values())),
                     f"{min(epochs)}–{max(epochs)}",
                     ", ".join(f"{r['arm']} seed {r['seed']} (epoch {r['epoch']})"
                               for r in row["late_selections"]) or "0"]
                    for amount, row in rows.items()
                    for epochs in [[r["epoch"] for values in row["selection"].values() for r in values]]])

    lines += ["## Action halves and object-matched actions", ""]
    subset_rows = []
    for amount, row in rows.items():
        action = row["action"]
        subsets = [("Test", action["halves"]["test"]), ("s8", action["halves"]["s8"]),
                   ("Object-matched pooled", action["object_matched"]),
                   ("Object-matched test", action["halves"]["test"]["object_matched"]),
                   ("Object-matched s8", action["halves"]["s8"]["object_matched"])]
        for name, subset in subsets:
            subset_rows.append([amount, name, subset["trials"], subset["clusters"],
                                *accuracy_cells(subset["accuracy"], LABELS),
                                *gain_cells(subset, ("Body",))])
    lines += table(["Clips", "Panel", "Trials", "Source clips", *LABELS, "Body gain"], subset_rows)

    lines += ["## Body gain by action word", "",
              "At 63 clips, drink has three training utterances and is out of vocabulary. Its frozen "
              "trials receive 25% expected credit in every arm.", ""]
    words = list(rows["630"]["action"]["by_word"])
    lines += table(["Clips", *words],
                   [[amount, *[gain_cells(row["action"]["by_word"][word], ("Body",))[0] for word in words]]
                    for amount, row in rows.items()])
    lines += ["## All extra inputs removed at evaluation", ""]
    lines += table(["Clips", "Rung", *LABELS],
                   [[amount, rung.title(), *accuracy_cells(row[rung]["all_extra_inputs_removed"], LABELS)]
                    for amount, row in rows.items() for rung in RUNGS])

    lines += ["## Picture transfer", "",
              "Rendered and Konkle panels use the selected checkpoint with picture inputs only.", ""]
    lines += table(["Clips", "Panel", "Trials", *LABELS],
                   [[amount, panel, p["trials"], *accuracy_cells(p["accuracy"], LABELS)]
                    for amount, row in rows.items() for panel, p in row["picture_transfer"].items()])

    comparisons = results["comparisons"]
    lines += [
        "## 630 clips versus published Step 7", "",
        "The held-out panels, five seeds and 630 training clips match Step 7. Step 9 changes temporal "
        "initialization and dev selection, which limits attribution to either change alone. "
        "[Step 7 published table](../../../../docs/step_7_word_types.md#completed-results--19-november-2026)", "",
    ]
    accuracy_rows, gain_rows = [], []
    for rung, old in comparisons["step7"]["at_630"].items():
        current = rows["630"][rung]
        accuracy_rows.append([rung.title(), "Step 7", *step7.percent(old["published_accuracy"].values())])
        accuracy_rows.append([rung.title(), "Step 9", *accuracy_cells(current["accuracy"], LABELS)])
        accuracy_rows.append([rung.title(), "Δ pp", *[f"{v:+.2f}" for v in old["accuracy_delta_pp"].values()]])
        gain_rows.append([rung.title(), "Step 7", *gain_cells({"gains": old["published_gains"]})])
        gain_rows.append([rung.title(), "Step 9", *gain_cells(current)])
        gain_rows.append([rung.title(), "Δ pp", *[f"{v:+.2f}" for v in old["gain_delta_pp"].values()]])
    lines += table(["Rung", "Study", *LABELS], accuracy_rows)
    lines += table(["Rung", "Study", *step7.GAINS[:3]], gain_rows)

    lines += [
        "## Noun curve versus Step 6", "",
        "Both studies use the same noun test panel. Step 6 used static video, different speech and "
        "dev selection, and three seeds. Differences are Step 9 minus Step 6 and are descriptive. "
        "[Step 6 published curve](../../../../docs/step_6_language_acquisition_curves.md#completed-results--17-november-2026)", "",
    ]
    lines += table(["Clips", *[f"Step 6 {label}" for label in LABELS[:3]],
                    *[f"Δ {label} (pp)" for label in LABELS[:3]]],
                   [[amount, *accuracy_cells(row["accuracy"]),
                     *[f"{v:+.2f}" for v in row["accuracy_delta_pp"].values()]]
                    for amount, row in comparisons["step6"]["noun_curve"].items()])
    lines += ["Published Step 6 gain intervals resampled source clips with seeds held fixed.", ""]
    lines += table(["Clips", *[f"Step 6 {label}" for label in step7.GAINS[:3]]],
                   [[amount, *gain_cells(row)] for amount, row in comparisons["step6"]["noun_curve"].items()])
    return lines


def analyze(protocol, panels, pictures, verification, scores, transfer, selections):
    rows = {}
    for fraction, amount in LADDER.items():
        row = fraction_results(protocol, panels, scores[fraction])
        selected = {LABELS[ARMS.index(arm)]: values for arm, values in selections[fraction].items()}
        row.update(
            fraction=fraction,
            selection=selected,
            late_selections=[{"arm": label, **r} for label, values in selected.items() for r in values
                             if r["epoch"] >= verification["late_epoch_threshold"]],
            picture_transfer={
                PICTURES[name]: {"trials": picture["count"],
                                 "accuracy": arm_stats(transfer[fraction][name], tuple(selected))}
                for name, picture in pictures.items()
            },
        )
        rows[str(amount)] = row
    results = {
        "verification": verification,
        "method": {"bootstrap_draws": step7.DRAWS, "bootstrap_seed": step7.BOOTSTRAP_SEED,
                   "interval": "joint percentile 95% over paired seed IDs and source clips",
                   "sd_ddof": 1, "pooled_panel_weighting": "by trial", "unknown_query_credit": 0.25},
        "by_clip_count": rows,
        "comparisons": historical_comparisons(protocol, rows),
    }
    markdown = "\n".join(report(results))
    serialized = json.dumps(results, indent=2, allow_nan=False)
    (ANALYSIS / "results.md").write_text(markdown + "\n")
    (ANALYSIS / "results.json").write_text(serialized + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    protocol = read_json(FROZEN / "protocol.json")
    panels = step7.read_panels(protocol)
    pictures = picture_panels(protocol)
    verified = verify_runs(protocol, panels, pictures)
    if not args.verify_only:
        analyze(protocol, panels, pictures, *verified)


if __name__ == "__main__":
    main()
