from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(categorical: List[str], numeric: List[str]) -> Pipeline:
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric),
    ])
    clf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42, min_samples_leaf=3)
    return Pipeline([("prep", preprocess), ("clf", clf)])


def evaluate(df: pd.DataFrame, feature_sets: Dict[str, Tuple[List[str], List[str]]]) -> Dict[str, Dict[str, float]]:
    y = df["event_label"]
    # stratify when possible
    stratify = y if y.value_counts().min() >= 2 else None
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=stratify)
    results: Dict[str, Dict[str, float]] = {}
    reports: Dict[str, str] = {}
    for name, (cat, num) in feature_sets.items():
        pipe = build_pipeline(cat, num)
        pipe.fit(train_df[cat + num], train_df["event_label"])
        preds = pipe.predict(test_df[cat + num])
        results[name] = {
            "accuracy": float(accuracy_score(test_df["event_label"], preds)),
            "macro_f1": float(f1_score(test_df["event_label"], preds, average="macro")),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        }
        reports[name] = classification_report(test_df["event_label"], preds, zero_division=0)
    return results, reports


def plot_results(results: Dict[str, Dict[str, float]], out_path: Path) -> None:
    names = list(results.keys())
    acc = [results[n]["accuracy"] for n in names]
    f1 = [results[n]["macro_f1"] for n in names]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - width/2, acc, width, label="accuracy")
    ax.bar(x + width/2, f1, width, label="macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Does richer synthetic modality improve causal event prediction?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BabyWorld-Lite data-usability benchmark.")
    parser.add_argument("--data", type=str, default="data/run/manifest.csv")
    parser.add_argument("--out", type=str, default="data/run")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)

    # Same model class and training budget; only the input modalities change.
    feature_sets = {
        "vision+language proxy": (
            ["shape", "color"],
            ["initial_obj_x", "initial_obj_y"],
        ),
        "plus action": (
            ["shape", "color", "action"],
            ["initial_obj_x", "initial_obj_y"],
        ),
        "plus sensorimotor": (
            ["shape", "color", "action"],
            ["initial_obj_x", "initial_obj_y", "first_contact_force", "max_force", "contact_count", "slip_estimate", "vibration_energy", "hardness_proxy"],
        ),
        "oracle physical state": (
            ["shape", "color", "action", "material"],
            ["initial_obj_x", "initial_obj_y", "mass", "friction", "bounciness", "hardness", "rollability", "graspability",
             "first_contact_force", "max_force", "contact_count", "slip_estimate", "vibration_energy", "hardness_proxy"],
        ),
    }
    results, reports = evaluate(df, feature_sets)
    (out / "benchmark_results.json").write_text(json.dumps(results, indent=2))
    (out / "classification_reports.txt").write_text("\n\n".join([f"=== {k} ===\n{v}" for k, v in reports.items()]))
    plot_results(results, out / "benchmark_plot.png")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
