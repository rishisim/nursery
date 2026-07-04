from __future__ import annotations

import pandas as pd

from babyworld_lite.eval.coverage import build_coverage_report, coverage_report_text


def test_coverage_report_counts_grid_and_degenerate_cells() -> None:
    df = pd.DataFrame(
        [
            {"shape": "cup", "action": "push", "material": "metal", "event_label": "topples"},
            {"shape": "cup", "action": "push", "material": "metal", "event_label": "topples"},
            {"shape": "ball", "action": "tap", "material": "foam", "event_label": "rolls_far"},
        ]
    )

    report = build_coverage_report(df, sparse_threshold=2)
    text = coverage_report_text(report)

    assert report["n_episodes"] == 3
    assert report["event_distribution"] == {"rolls_far": 1, "topples": 2}
    assert report["grid"]["expected_cells"] == 8
    assert report["grid"]["observed_cells"] == 2
    assert len(report["degenerate_cells"]["empty"]) == 6
    assert "Coverage / Diversity Report" in text
