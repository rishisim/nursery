from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Sequence

import numpy as np


def _as_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(list(values))


def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    truth = _as_array(y_true)
    pred = _as_array(y_pred)
    return float(np.mean(truth == pred)) if len(truth) else 0.0


def macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    truth = _as_array(y_true)
    pred = _as_array(y_pred)
    labels = sorted(set(truth.tolist()) | set(pred.tolist()))
    if not labels:
        return 0.0
    scores: list[float] = []
    for label in labels:
        tp = float(np.sum((truth == label) & (pred == label)))
        fp = float(np.sum((truth != label) & (pred == label)))
        fn = float(np.sum((truth == label) & (pred != label)))
        precision = tp / (tp + fp) if tp + fp > 0.0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0.0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
        scores.append(f1)
    return float(np.mean(scores))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(truth - pred))) if len(truth) else 0.0


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((truth - pred) ** 2))) if len(truth) else 0.0


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if len(truth) == 0:
        return 0.0
    denom = float(np.sum((truth - np.mean(truth)) ** 2))
    if denom <= 1e-12:
        return 0.0
    numer = float(np.sum((truth - pred) ** 2))
    return float(1.0 - numer / denom)


def bootstrap_metric(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    metric_fn: Callable[[Sequence[Any], Sequence[Any]], float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    truth = _as_array(y_true)
    pred = _as_array(y_pred)
    point = float(metric_fn(truth, pred))
    if len(truth) == 0 or n_bootstrap <= 0:
        return {"point": point, "ci95": [point, point]}
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_bootstrap):
        sample = rng.integers(0, len(truth), len(truth))
        scores.append(float(metric_fn(truth[sample], pred[sample])))
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return {"point": point, "ci95": [float(lower), float(upper)]}


def paired_mae_improvement(
    y_true: Sequence[float],
    base_pred: Sequence[float],
    new_pred: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    truth = np.asarray(y_true, dtype=float)
    base = np.asarray(base_pred, dtype=float)
    new = np.asarray(new_pred, dtype=float)
    deltas = np.abs(truth - base) - np.abs(truth - new)
    point = float(np.mean(deltas)) if len(deltas) else 0.0
    if len(deltas) == 0 or n_bootstrap <= 0:
        return {"point": point, "ci95": [point, point], "meaning": "positive means the new arm lowers MAE"}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        sample = rng.integers(0, len(deltas), len(deltas))
        samples.append(float(np.mean(deltas[sample])))
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return {
        "point": point,
        "ci95": [float(lower), float(upper)],
        "meaning": "positive means the new arm lowers MAE",
    }


def paired_macro_f1_delta(
    y_true: Sequence[Any],
    base_pred: Sequence[Any],
    new_pred: Sequence[Any],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    truth = _as_array(y_true)
    base = _as_array(base_pred)
    new = _as_array(new_pred)
    point = float(macro_f1(truth, new) - macro_f1(truth, base))
    if len(truth) == 0 or n_bootstrap <= 0:
        return {"point": point, "ci95": [point, point], "meaning": "positive means the new arm improves macro-F1"}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        sample = rng.integers(0, len(truth), len(truth))
        samples.append(float(macro_f1(truth[sample], new[sample]) - macro_f1(truth[sample], base[sample])))
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return {
        "point": point,
        "ci95": [float(lower), float(upper)],
        "meaning": "positive means the new arm improves macro-F1",
    }


def classification_report_text(y_true: Sequence[Any], y_pred: Sequence[Any]) -> str:
    truth = _as_array(y_true)
    pred = _as_array(y_pred)
    labels = sorted(set(truth.tolist()) | set(pred.tolist()))
    lines = [f"{'label':<18} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>9}"]
    for label in labels:
        tp = float(np.sum((truth == label) & (pred == label)))
        fp = float(np.sum((truth != label) & (pred == label)))
        fn = float(np.sum((truth == label) & (pred != label)))
        support = int(np.sum(truth == label))
        precision = tp / (tp + fp) if tp + fp > 0.0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0.0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
        lines.append(f"{str(label):<18} {precision:>9.3f} {recall:>9.3f} {f1:>9.3f} {support:>9d}")
    lines.append("")
    lines.append(f"accuracy={accuracy(truth, pred):.3f} macro_f1={macro_f1(truth, pred):.3f} n={len(truth)}")
    return "\n".join(lines)


def regression_report_text(results: Dict[str, Any]) -> str:
    lines = [f"{'target':<24} {'MAE':>10} {'MAE CI95':>27} {'RMSE':>10} {'R2':>10}"]
    for target, metrics in results["regression"].items():
        mae_metric = metrics["mae"]
        rmse_metric = metrics["rmse"]
        r2_metric = metrics["r2"]
        ci = mae_metric["ci95"]
        lines.append(
            f"{target:<24} {mae_metric['point']:>10.3f} "
            f"[{ci[0]:>8.3f}, {ci[1]:>8.3f}] "
            f"{rmse_metric['point']:>10.3f} {r2_metric['point']:>10.3f}"
        )
    lines.append("")
    lines.append(f"n_train={results.get('n_train')} n_test={results.get('n_test')} n_features={results.get('n_features')}")
    return "\n".join(lines)
