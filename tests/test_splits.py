from __future__ import annotations

import pandas as pd
import pytest

from babyworld_lite.eval.splits import (
    held_out_composition_split,
    held_out_impulse_mass_split,
    held_out_material_split,
)


def _rows(count: int, material: str, shape: str, action: str) -> list[dict[str, str]]:
    return [{"material": material, "shape": shape, "action": action} for _ in range(count)]


def test_held_out_material_support_check_passes_and_fails() -> None:
    df = pd.DataFrame(
        _rows(205, "metal", "cup", "push")
        + _rows(250, "foam", "cup", "tap")
        + _rows(250, "wood", "block", "push")
    )

    split = held_out_material_split(df, material="metal", min_test_cell_size=200)
    assert split.support == {"material=metal": 205}
    assert len(split.test_index) == 205

    with pytest.raises(ValueError, match="below min_test_cell_size"):
        held_out_material_split(df, material="metal", min_test_cell_size=300)


def test_held_out_composition_keeps_shape_and_action_support_in_train() -> None:
    df = pd.DataFrame(
        _rows(210, "metal", "cup", "push")
        + _rows(220, "foam", "cup", "tap")
        + _rows(230, "wood", "block", "push")
    )

    split = held_out_composition_split(df, compositions=(("cup", "push"),), min_test_cell_size=200)
    train = df.loc[split.train_index]
    test = df.loc[split.test_index]

    assert len(test) == 210
    assert not ((train["shape"] == "cup") & (train["action"] == "push")).any()
    assert (train["shape"] == "cup").any()
    assert (train["action"] == "push").any()

    with pytest.raises(ValueError, match="below min_test_cell_size"):
        held_out_composition_split(df, compositions=(("cup", "push"),), min_test_cell_size=300)


def test_held_out_impulse_mass_split_holds_out_high_high_region() -> None:
    df = pd.DataFrame(
        [
            {
                "material": "foam",
                "shape": "cup",
                "action": "push",
                "hidden_impulse": impulse,
                "mass": mass,
            }
            for impulse, mass in [
                (0.1, 0.5),
                (0.2, 1.0),
                (0.3, 1.5),
                (0.4, 2.0),
                (0.5, 2.5),
                (0.6, 3.0),
                (0.7, 3.5),
                (0.8, 4.0),
            ]
        ]
    )

    split = held_out_impulse_mass_split(df, impulse_quantile=0.5, mass_quantile=0.5, min_test_cell_size=1)
    test = df.loc[split.test_index]
    train = df.loc[split.train_index]

    assert len(test) > 0
    assert (test["hidden_impulse"] >= split.support["impulse_threshold"]).all()
    assert (test["mass"] >= split.support["mass_threshold"]).all()
    assert len(train) + len(test) == len(df)
