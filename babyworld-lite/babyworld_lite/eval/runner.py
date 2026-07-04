from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from babyworld_lite.eval.dataset import (
    categorical_numeric_columns,
    feature_columns_for_arm,
    load_episodes,
    touch_feature_columns,
    build_eval_frame,
)
from babyworld_lite.eval.coverage import build_coverage_report, coverage_report_text
from babyworld_lite.eval.features import ARM_MODALITIES
from babyworld_lite.eval.metrics import (
    bootstrap_metric,
    mae,
    paired_mae_improvement,
    regression_report_text,
    rmse,
    r2_score,
)
from babyworld_lite.eval.splits import EvalSplit, make_split


REGRESSION_TARGETS = (
    "target_delta_x",
    "target_delta_y",
    "target_displacement",
    "target_final_x",
    "target_final_y",
    "target_topple_angle",
)
PRIMARY_BASE_ARM = "vision_proprio"
PRIMARY_TOUCH_ARM = "vision_proprio_touch"


class Progress:
    def __init__(self, total_units: int) -> None:
        self.total_units = max(1, int(total_units))
        self.done_units = 0
        self.started_at = time.perf_counter()
        self.current_label: Optional[str] = None
        self.current_started_at = self.started_at

    def _format_seconds(self, seconds: float) -> str:
        seconds = max(0.0, seconds)
        minutes, secs = divmod(int(round(seconds)), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    def _eta(self) -> str:
        if self.done_units == 0:
            return "estimating"
        elapsed = time.perf_counter() - self.started_at
        seconds_per_unit = elapsed / self.done_units
        remaining = (self.total_units - self.done_units) * seconds_per_unit
        return self._format_seconds(remaining)

    def note(self, message: str) -> None:
        elapsed = self._format_seconds(time.perf_counter() - self.started_at)
        print(f"[eval] {message} | elapsed {elapsed}", flush=True)

    def start(self, label: str) -> None:
        self.current_label = label
        self.current_started_at = time.perf_counter()
        print(
            f"[eval] start {self.done_units + 1}/{self.total_units}: {label} "
            f"| eta {self._eta()}",
            flush=True,
        )

    def done(self) -> None:
        self.done_units += 1
        label = self.current_label or "unit"
        duration = time.perf_counter() - self.current_started_at
        elapsed = time.perf_counter() - self.started_at
        print(
            f"[eval] done  {self.done_units}/{self.total_units}: {label} "
            f"| step {self._format_seconds(duration)} | elapsed {self._format_seconds(elapsed)} "
            f"| eta {self._eta()}",
            flush=True,
        )
        self.current_label = None


def _missing_sklearn(exc: ModuleNotFoundError) -> SystemExit:
    return SystemExit(
        "The honest eval runner requires scikit-learn. Install requirements first: "
        "python3 -m pip install -r requirements.txt"
    )


def _one_hot_encoder() -> Any:
    try:
        from sklearn.preprocessing import OneHotEncoder
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in missing-env runs
        raise _missing_sklearn(exc) from exc

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _preprocess(categorical: Sequence[str], numeric: Sequence[str]) -> Any:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in missing-env runs
        raise _missing_sklearn(exc) from exc

    transformers = []
    if categorical:
        transformers.append(("cat", _one_hot_encoder(), list(categorical)))
    if numeric:
        transformers.append(("num", StandardScaler(), list(numeric)))
    if not transformers:
        raise ValueError("no feature columns were provided")
    return ColumnTransformer(transformers)


def _regressor(categorical: Sequence[str], numeric: Sequence[str], seed: int) -> Any:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in missing-env runs
        raise _missing_sklearn(exc) from exc

    return Pipeline(
        [
            ("prep", _preprocess(categorical, numeric)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=160,
                    max_depth=12,
                    min_samples_leaf=4,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _metric_bundle_regression(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    return {
        "mae": bootstrap_metric(y_true, y_pred, mae, n_bootstrap=n_bootstrap, seed=seed),
        "rmse": bootstrap_metric(y_true, y_pred, rmse, n_bootstrap=n_bootstrap, seed=seed + 1),
        "r2": bootstrap_metric(y_true, y_pred, r2_score, n_bootstrap=n_bootstrap, seed=seed + 2),
    }


def _fit_predict_arm(
    df: pd.DataFrame,
    split: EvalSplit,
    columns: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    categorical, numeric = categorical_numeric_columns(df, columns)
    train_df = df.loc[split.train_index]
    test_df = df.loc[split.test_index]
    x_train = train_df[list(columns)]
    x_test = test_df[list(columns)]

    regression_predictions: Dict[str, np.ndarray] = {}
    for offset, target in enumerate(REGRESSION_TARGETS):
        reg = _regressor(categorical, numeric, seed + 101 + offset)
        reg.fit(x_train, train_df[target])
        regression_predictions[target] = reg.predict(x_test)

    return regression_predictions


def _evaluate_predictions(
    df: pd.DataFrame,
    split: EvalSplit,
    predictions: Mapping[str, Any],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    test_df = df.loc[split.test_index]
    regression: Dict[str, Any] = {}
    for offset, target in enumerate(REGRESSION_TARGETS):
        regression[target] = _metric_bundle_regression(
            test_df[target].to_numpy(dtype=float),
            predictions[target],
            n_bootstrap=n_bootstrap,
            seed=seed + 10 + offset,
        )
    return {
        "regression": regression,
    }


def _priors_only_predictions(df: pd.DataFrame, split: EvalSplit) -> Dict[str, Any]:
    train_df = df.loc[split.train_index]
    test_df = df.loc[split.test_index]
    return {
        target: np.full(len(test_df), float(train_df[target].mean()), dtype=float)
        for target in REGRESSION_TARGETS
    }


def _shuffle_columns(
    df: pd.DataFrame,
    split: EvalSplit,
    columns: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    shuffled = df.copy()
    rng = np.random.default_rng(seed)
    for column in columns:
        train_values = shuffled.loc[split.train_index, column].to_numpy(copy=True)
        test_values = shuffled.loc[split.test_index, column].to_numpy(copy=True)
        shuffled.loc[split.train_index, column] = rng.permutation(train_values)
        shuffled.loc[split.test_index, column] = rng.permutation(test_values)
    return shuffled


def _arm_columns(df: pd.DataFrame, arm: str) -> List[str]:
    return feature_columns_for_arm(df, arm)


def _evaluate_named_arm(
    df: pd.DataFrame,
    split: EvalSplit,
    name: str,
    columns: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    predictions = _fit_predict_arm(df, split, columns, seed=seed)
    metrics = _evaluate_predictions(df, split, predictions, n_bootstrap=n_bootstrap, seed=seed)
    metrics["n_train"] = int(len(split.train_index))
    metrics["n_test"] = int(len(split.test_index))
    metrics["n_features"] = int(len(columns))
    metrics["feature_groups"] = name
    report = regression_report_text(metrics)
    return metrics, predictions, report


def evaluate_split(
    df: pd.DataFrame,
    split: EvalSplit,
    n_bootstrap: int = 1000,
    seed: int = 42,
    progress: Optional[Progress] = None,
) -> tuple[Dict[str, Any], List[str]]:
    split_result: Dict[str, Any] = {
        "support": split.support,
        "n_train": int(len(split.train_index)),
        "n_test": int(len(split.test_index)),
        "arms": {},
        "controls": {},
        "paired_bootstrap": {"regression": {}},
    }
    reports: list[str] = []
    prediction_store: Dict[str, Dict[str, Any]] = {}

    priors_predictions = _priors_only_predictions(df, split)
    split_result["arms"]["priors_only"] = _evaluate_predictions(
        df,
        split,
        priors_predictions,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    split_result["arms"]["priors_only"]["n_train"] = int(len(split.train_index))
    split_result["arms"]["priors_only"]["n_test"] = int(len(split.test_index))
    prediction_store["priors_only"] = priors_predictions
    reports.append(
        f"=== {split.name} :: priors_only ===\n"
        + regression_report_text(split_result["arms"]["priors_only"])
    )

    for arm_index, arm in enumerate(ARM_MODALITIES.keys()):
        if progress:
            progress.start(f"{split.name} arm {arm}")
        columns = _arm_columns(df, arm)
        metrics, predictions, report = _evaluate_named_arm(
            df,
            split,
            arm,
            columns,
            n_bootstrap=n_bootstrap,
            seed=seed + arm_index * 17,
        )
        split_result["arms"][arm] = metrics
        prediction_store[arm] = predictions
        reports.append(f"=== {split.name} :: {arm} ===\n{report}")
        if progress:
            progress.done()

    if PRIMARY_BASE_ARM in prediction_store and PRIMARY_TOUCH_ARM in prediction_store:
        for offset, target in enumerate(REGRESSION_TARGETS):
            y_true = df.loc[split.test_index, target].to_numpy(dtype=float)
            split_result["paired_bootstrap"]["regression"][
                f"{target}:{PRIMARY_BASE_ARM}_to_{PRIMARY_TOUCH_ARM}:mae_improvement"
            ] = paired_mae_improvement(
                y_true,
                prediction_store[PRIMARY_BASE_ARM][target],
                prediction_store[PRIMARY_TOUCH_ARM][target],
                n_bootstrap=n_bootstrap,
                seed=seed + 311 + offset,
            )

    impulse_columns = feature_columns_for_arm(df, PRIMARY_BASE_ARM, extra_columns=["hidden_impulse"])
    if progress:
        progress.start(f"{split.name} control {PRIMARY_BASE_ARM}_hidden_impulse")
    metrics, predictions, report = _evaluate_named_arm(
        df,
        split,
        f"{PRIMARY_BASE_ARM}_hidden_impulse",
        impulse_columns,
        n_bootstrap=n_bootstrap,
        seed=seed + 401,
    )
    touch_metrics = split_result["arms"].get(PRIMARY_TOUCH_ARM, {})
    metrics["comparison_to_touch_arm"] = {
        "displacement_mae_delta": (
            metrics["regression"]["target_displacement"]["mae"]["point"]
            - touch_metrics.get("regression", {})
            .get("target_displacement", {})
            .get("mae", {})
            .get("point", 0.0)
        ),
        "meaning": "near zero means hidden impulse in the pre-touch baseline reached the touch arm",
    }
    split_result["controls"][f"{PRIMARY_BASE_ARM}_hidden_impulse"] = metrics
    reports.append(f"=== {split.name} :: control {PRIMARY_BASE_ARM}_hidden_impulse ===\n{report}")
    if progress:
        progress.done()

    split_result["controls"]["all_features_shuffled"] = {}
    for arm_index, arm in enumerate(ARM_MODALITIES.keys()):
        if progress:
            progress.start(f"{split.name} control {arm}_all_features_shuffled")
        columns = _arm_columns(df, arm)
        shuffled_df = _shuffle_columns(df, split, columns, seed=seed + 501 + arm_index)
        metrics, _, report = _evaluate_named_arm(
            shuffled_df,
            split,
            f"{arm}_all_features_shuffled",
            columns,
            n_bootstrap=n_bootstrap,
            seed=seed + 601 + arm_index,
        )
        split_result["controls"]["all_features_shuffled"][arm] = metrics
        reports.append(f"=== {split.name} :: control {arm}_all_features_shuffled ===\n{report}")
        if progress:
            progress.done()

    split_result["controls"]["touch_features_shuffled"] = {}
    touch_columns = touch_feature_columns(df)
    for arm_index, arm in enumerate(("vision_proprio_touch", "oracle_full_state")):
        if progress:
            progress.start(f"{split.name} control {arm}_touch_features_shuffled")
        columns = _arm_columns(df, arm)
        shuffled_df = _shuffle_columns(df, split, touch_columns, seed=seed + 701 + arm_index)
        metrics, _, report = _evaluate_named_arm(
            shuffled_df,
            split,
            f"{arm}_touch_features_shuffled",
            columns,
            n_bootstrap=n_bootstrap,
            seed=seed + 801 + arm_index,
        )
        split_result["controls"]["touch_features_shuffled"][arm] = metrics
        reports.append(f"=== {split.name} :: control {arm}_touch_features_shuffled ===\n{report}")
        if progress:
            progress.done()

    return split_result, reports


def load_decision_rules(path: Path) -> Mapping[str, Any]:
    with path.open() as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded


def evaluate_decision_rules(results: Mapping[str, Any], rules: Mapping[str, Any]) -> Dict[str, Any]:
    evaluated: Dict[str, Any] = {}
    for name, rule in (rules.get("rules") or {}).items():
        split_name = rule.get("split", "held_out_material")
        split_result = results.get("splits", {}).get(split_name)
        if split_result is None:
            evaluated[name] = {"status": "fail", "reason": f"split {split_name} was not evaluated"}
            continue

        rule_type = rule.get("type")
        if rule_type == "touch_helps":
            target = rule.get("target", "target_displacement")
            key = f"{target}:{rule.get('base_arm', PRIMARY_BASE_ARM)}_to_{rule.get('touch_arm', PRIMARY_TOUCH_ARM)}:mae_improvement"
            delta = split_result["paired_bootstrap"]["regression"].get(key)
            if delta is None:
                evaluated[name] = {"status": "fail", "reason": f"missing paired bootstrap {key}"}
                continue
            lower, upper = delta["ci95"]
            passed = lower > 0.0
            evaluated[name] = {
                "status": "pass" if passed else "fail",
                "point": delta["point"],
                "ci95": delta["ci95"],
                "reason": (
                    "touch lowered held-out MAE with CI excluding 0"
                    if passed
                    else "touch MAE improvement CI did not exclude 0"
                ),
                "ci_upper": upper,
            }
        elif rule_type == "honest_negative_escalation":
            target = rule.get("target", "target_displacement")
            base_arm = rule.get("base_arm", PRIMARY_BASE_ARM)
            touch_arm = rule.get("touch_arm", PRIMARY_TOUCH_ARM)
            threshold = float(rule.get("r2_threshold", 0.9))
            key = f"{target}:{base_arm}_to_{touch_arm}:mae_improvement"
            delta = split_result["paired_bootstrap"]["regression"].get(key)
            base_r2 = (
                split_result["arms"]
                .get(base_arm, {})
                .get("regression", {})
                .get(target, {})
                .get("r2", {})
                .get("point")
            )
            if delta is None or base_r2 is None:
                evaluated[name] = {"status": "fail", "reason": "missing R2 or paired MAE delta"}
                continue
            lower, upper = delta["ci95"]
            triggered = base_r2 > threshold and lower <= 0.0 <= upper
            evaluated[name] = {
                "status": "pass",
                "triggered": bool(triggered),
                "base_r2": float(base_r2),
                "mae_improvement_ci95": delta["ci95"],
                "reason": (
                    "honest negative triggered; report redundancy and make the task harder"
                    if triggered
                    else "honest negative condition not triggered"
                ),
            }
        else:
            evaluated[name] = {"status": "fail", "reason": f"unknown rule type: {rule_type}"}
    return evaluated


def _plot_heldout_bars(results: Mapping[str, Any], split_name: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    split_result = results["splits"].get(split_name)
    if not split_result:
        return
    arms = [arm for arm in ARM_MODALITIES.keys() if arm in split_result["arms"]]
    values = [
        split_result["arms"][arm]["regression"]["target_displacement"]["mae"]["point"]
        for arm in arms
    ]
    lows = [
        split_result["arms"][arm]["regression"]["target_displacement"]["mae"]["ci95"][0]
        for arm in arms
    ]
    highs = [
        split_result["arms"][arm]["regression"]["target_displacement"]["mae"]["ci95"][1]
        for arm in arms
    ]
    yerr = [
        [max(0.0, value - low) for value, low in zip(values, lows)],
        [max(0.0, high - value) for value, high in zip(values, highs)],
    ]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, values, yerr=yerr, capsize=5, color=["#4c78a8", "#59a14f", "#f28e2b", "#b07aa1"])
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=18, ha="right")
    ax.set_ylabel("MAE: post-contact displacement")
    ax.set_title(f"{split_name}: held-out displacement error with 95% CI")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_k_sweep(k_sweep: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arm in ARM_MODALITIES.keys():
        xs = [row["k_offset"] for row in k_sweep]
        ys = [row["mae_by_arm"][arm] for row in k_sweep]
        ax.plot(xs, ys, marker="o", label=arm)
    ax.set_xlabel("Frames after contact prediction time")
    ax.set_ylabel("MAE: post-contact displacement")
    ax.set_title("Error vs prediction frame")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_k_sweep(
    episodes: Sequence[Mapping[str, Any]],
    split_name: str,
    max_offset: int,
    min_test_cell_size: int,
    seed: int,
    random_test_size: float,
    holdout_material: str,
    heldout_compositions: Sequence[tuple[str, str]],
    impulse_quantile: float,
    mass_quantile: float,
    progress: Optional[Progress] = None,
) -> List[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for offset in range(max_offset + 1):
        df = build_eval_frame(episodes, k_offset=offset)
        split = make_split(
            df,
            split_name,
            seed=seed,
            min_test_cell_size=min_test_cell_size,
            random_test_size=random_test_size,
            holdout_material=holdout_material,
            heldout_compositions=heldout_compositions,
            impulse_quantile=impulse_quantile,
            mass_quantile=mass_quantile,
        )
        mae_by_arm: Dict[str, float] = {}
        for arm_index, arm in enumerate(ARM_MODALITIES.keys()):
            if progress:
                progress.start(f"k-sweep {split_name} offset={offset} arm {arm}")
            columns = _arm_columns(df, arm)
            predictions = _fit_predict_arm(df, split, columns, seed=seed + offset * 31 + arm_index)
            y_true = df.loc[split.test_index, "target_displacement"].to_numpy(dtype=float)
            mae_by_arm[arm] = mae(y_true, predictions["target_displacement"])
            if progress:
                progress.done()
        rows.append({"k_offset": offset, "mae_by_arm": mae_by_arm})
    return rows


def parse_compositions(values: Optional[Sequence[str]]) -> tuple[tuple[str, str], ...]:
    if not values:
        return (("cup", "push"),)
    parsed = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"composition must be formatted shape:action, got {value!r}")
        shape, action = value.split(":", 1)
        parsed.append((shape, action))
    return tuple(parsed)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run leakage-aware BabyWorld-Lite evals.")
    parser.add_argument("--episodes", default="data/sample/episodes.jsonl", help="Input episodes.jsonl path")
    parser.add_argument("--out", default="data/eval", help="Output directory")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["held_out_material", "held_out_impulse_mass", "held_out_composition"],
        choices=["random", "held_out_material", "held_out_impulse_mass", "held_out_composition"],
    )
    parser.add_argument("--rules", default="decision_rules.yaml", help="Pre-registered decision rules YAML")
    parser.add_argument("--min-test-cell-size", type=int, default=200)
    parser.add_argument("--random-test-size", type=float, default=0.25)
    parser.add_argument("--holdout-material", default="metal")
    parser.add_argument("--heldout-composition", action="append", help="Held-out shape:action cell; repeatable")
    parser.add_argument("--impulse-quantile", type=float, default=0.75)
    parser.add_argument("--mass-quantile", type=float, default=0.50)
    parser.add_argument("--coverage-sparse-threshold", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-sweep-max-offset", type=int, default=6)
    parser.add_argument("--k-sweep-split", default="held_out_material")
    parser.add_argument("--skip-k-sweep", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(args.episodes)
    df = build_eval_frame(episodes)
    heldout_compositions = parse_compositions(args.heldout_composition)
    split_model_units = len(ARM_MODALITIES) + 1 + len(ARM_MODALITIES) + 2
    k_sweep_units = 0 if args.skip_k_sweep else (args.k_sweep_max_offset + 1) * len(ARM_MODALITIES)
    progress = Progress(total_units=len(args.splits) * split_model_units + k_sweep_units)
    progress.note(
        f"loaded {len(df)} episodes; progress unit = one model arm "
        "(all forward-prediction regressors)"
    )
    coverage = build_coverage_report(df, sparse_threshold=args.coverage_sparse_threshold)

    results: Dict[str, Any] = {
        "config": {
            "episodes": str(args.episodes),
            "n_episodes": int(len(df)),
            "splits": args.splits,
            "min_test_cell_size": args.min_test_cell_size,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "heldout_material": args.holdout_material,
            "heldout_compositions": [list(pair) for pair in heldout_compositions],
            "impulse_quantile": args.impulse_quantile,
            "mass_quantile": args.mass_quantile,
            "targets": list(REGRESSION_TARGETS),
        },
        "coverage_summary": {
            "event_distribution": coverage["event_distribution"],
            "grid": coverage["grid"],
            "degenerate_cells": {
                key: len(value) for key, value in coverage["degenerate_cells"].items()
            },
        },
        "splits": {},
    }
    all_reports: list[str] = []
    for split_index, split_name in enumerate(args.splits):
        progress.note(f"preparing split {split_name}")
        split = make_split(
            df,
            split_name,
            seed=args.seed,
            min_test_cell_size=args.min_test_cell_size,
            random_test_size=args.random_test_size,
            holdout_material=args.holdout_material,
            heldout_compositions=heldout_compositions,
            impulse_quantile=args.impulse_quantile,
            mass_quantile=args.mass_quantile,
        )
        split_result, reports = evaluate_split(
            df,
            split,
            n_bootstrap=args.bootstrap,
            seed=args.seed + split_index * 1000,
            progress=progress,
        )
        results["splits"][split_name] = split_result
        all_reports.extend(reports)
        progress.note(f"finished split {split_name}")

    rules_path = Path(args.rules)
    if rules_path.exists():
        rules = load_decision_rules(rules_path)
        results["decision_rules"] = evaluate_decision_rules(results, rules)
    else:
        results["decision_rules"] = {"_rules_file": {"status": "fail", "reason": f"missing {rules_path}"}}

    if not args.skip_k_sweep:
        k_sweep = run_k_sweep(
            episodes,
            split_name=args.k_sweep_split,
            max_offset=args.k_sweep_max_offset,
            min_test_cell_size=args.min_test_cell_size,
            seed=args.seed,
            random_test_size=args.random_test_size,
            holdout_material=args.holdout_material,
            heldout_compositions=heldout_compositions,
            impulse_quantile=args.impulse_quantile,
            mass_quantile=args.mass_quantile,
            progress=progress,
        )
        results["k_sweep"] = k_sweep
        _plot_k_sweep(k_sweep, out_dir / "k_sweep_error_vs_k.png")

    for split_name in results["splits"]:
        _plot_heldout_bars(results, split_name, out_dir / f"{split_name}_displacement_mae.png")

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "forward_prediction_report.txt").write_text("\n\n".join(all_reports))
    (out_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2))
    (out_dir / "coverage_report.txt").write_text(coverage_report_text(coverage))

    print(
        json.dumps(
            {
                "results": str(out_dir / "results.json"),
                "report": str(out_dir / "forward_prediction_report.txt"),
                "coverage": str(out_dir / "coverage_report.txt"),
            },
            indent=2,
        )
    )
    if results.get("decision_rules"):
        print("Decision rules:")
        for name, rule_result in results["decision_rules"].items():
            print(f"- {name}: {rule_result.get('status', 'unknown').upper()} - {rule_result.get('reason', '')}")


if __name__ == "__main__":
    main()
