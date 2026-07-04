from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


Composition = Tuple[str, str]


@dataclass(frozen=True)
class EvalSplit:
    name: str
    train_index: np.ndarray
    test_index: np.ndarray
    support: Dict[str, Any]


def _check_support(name: str, support: Dict[str, int], min_test_cell_size: int) -> None:
    too_small = {cell: count for cell, count in support.items() if count < min_test_cell_size}
    if too_small:
        details = ", ".join(f"{cell}={count}" for cell, count in sorted(too_small.items()))
        raise ValueError(
            f"{name} split has test cells below min_test_cell_size={min_test_cell_size}: {details}"
        )


def _check_nonempty_train_test(name: str, train_index: np.ndarray, test_index: np.ndarray) -> None:
    if len(train_index) == 0:
        raise ValueError(f"{name} split produced an empty train set")
    if len(test_index) == 0:
        raise ValueError(f"{name} split produced an empty test set")


def random_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    seed: int = 42,
    min_test_cell_size: int = 200,
) -> EvalSplit:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = df.index.to_numpy()
    shuffled = rng.permutation(indices)
    test_count = max(1, int(round(len(shuffled) * test_size)))
    test_index = np.sort(shuffled[:test_count])
    train_index = np.sort(shuffled[test_count:])
    _check_nonempty_train_test("random", train_index, test_index)
    support = {"random_test": int(len(test_index))}
    _check_support("random", support, min_test_cell_size)
    return EvalSplit("random", train_index, test_index, support)


def held_out_material_split(
    df: pd.DataFrame,
    material: str = "metal",
    min_test_cell_size: int = 200,
) -> EvalSplit:
    test_mask = df["material"] == material
    test_index = df.index[test_mask].to_numpy()
    train_index = df.index[~test_mask].to_numpy()
    _check_nonempty_train_test("held_out_material", train_index, test_index)
    support = {f"material={material}": int(test_mask.sum())}
    _check_support("held_out_material", support, min_test_cell_size)
    return EvalSplit("held_out_material", train_index, test_index, support)


def held_out_composition_split(
    df: pd.DataFrame,
    compositions: Sequence[Composition] = (("cup", "push"),),
    min_test_cell_size: int = 200,
) -> EvalSplit:
    if not compositions:
        raise ValueError("at least one held-out composition is required")

    test_mask = pd.Series(False, index=df.index)
    support: Dict[str, int] = {}
    for shape, action in compositions:
        cell_mask = (df["shape"] == shape) & (df["action"] == action)
        support[f"shape={shape}|action={action}"] = int(cell_mask.sum())
        test_mask |= cell_mask
        has_shape_without_action = bool(((df["shape"] == shape) & (df["action"] != action)).any())
        has_action_without_shape = bool(((df["shape"] != shape) & (df["action"] == action)).any())
        if not has_shape_without_action or not has_action_without_shape:
            raise ValueError(
                "held_out_composition requires train support for each held-out "
                f"shape and action separately; missing support for {shape} x {action}"
            )

    test_index = df.index[test_mask].to_numpy()
    train_index = df.index[~test_mask].to_numpy()
    _check_nonempty_train_test("held_out_composition", train_index, test_index)
    _check_support("held_out_composition", support, min_test_cell_size)
    return EvalSplit("held_out_composition", train_index, test_index, support)


def held_out_impulse_mass_split(
    df: pd.DataFrame,
    impulse_quantile: float = 0.75,
    mass_quantile: float = 0.50,
    min_test_cell_size: int = 200,
) -> EvalSplit:
    if "hidden_impulse" not in df or "mass" not in df:
        raise ValueError("held_out_impulse_mass requires hidden_impulse and mass columns")
    if not 0.0 < impulse_quantile < 1.0:
        raise ValueError("impulse_quantile must be between 0 and 1")
    if not 0.0 < mass_quantile < 1.0:
        raise ValueError("mass_quantile must be between 0 and 1")

    impulse_threshold = float(df["hidden_impulse"].quantile(impulse_quantile))
    mass_threshold = float(df["mass"].quantile(mass_quantile))
    test_mask = (df["hidden_impulse"] >= impulse_threshold) & (df["mass"] >= mass_threshold)
    test_index = df.index[test_mask].to_numpy()
    train_index = df.index[~test_mask].to_numpy()
    _check_nonempty_train_test("held_out_impulse_mass", train_index, test_index)
    support: Dict[str, Any] = {
        f"impulse>=q{impulse_quantile:.2f}|mass>=q{mass_quantile:.2f}": int(test_mask.sum()),
        "impulse_threshold": impulse_threshold,
        "mass_threshold": mass_threshold,
    }
    _check_support(
        "held_out_impulse_mass",
        {key: value for key, value in support.items() if "|" in key},
        min_test_cell_size,
    )
    return EvalSplit("held_out_impulse_mass", train_index, test_index, support)


def make_split(
    df: pd.DataFrame,
    name: str,
    seed: int = 42,
    min_test_cell_size: int = 200,
    random_test_size: float = 0.25,
    holdout_material: str = "metal",
    heldout_compositions: Iterable[Composition] = (("cup", "push"),),
    impulse_quantile: float = 0.75,
    mass_quantile: float = 0.50,
) -> EvalSplit:
    if name == "random":
        return random_split(
            df,
            test_size=random_test_size,
            seed=seed,
            min_test_cell_size=min_test_cell_size,
        )
    if name == "held_out_material":
        return held_out_material_split(
            df,
            material=holdout_material,
            min_test_cell_size=min_test_cell_size,
        )
    if name == "held_out_composition":
        return held_out_composition_split(
            df,
            compositions=tuple(heldout_compositions),
            min_test_cell_size=min_test_cell_size,
        )
    if name == "held_out_impulse_mass":
        return held_out_impulse_mass_split(
            df,
            impulse_quantile=impulse_quantile,
            mass_quantile=mass_quantile,
            min_test_cell_size=min_test_cell_size,
        )
    raise KeyError(f"unknown split: {name}")
