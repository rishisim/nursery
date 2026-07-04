from __future__ import annotations

from itertools import product
from typing import Any, Dict

import pandas as pd


def build_coverage_report(df: pd.DataFrame, sparse_threshold: int = 20, dominant_threshold: float = 0.95) -> Dict[str, Any]:
    shapes = sorted(df["shape"].dropna().unique().tolist())
    actions = sorted(df["action"].dropna().unique().tolist())
    materials = sorted(df["material"].dropna().unique().tolist())
    grid = list(product(shapes, actions, materials))

    grouped = df.groupby(["shape", "action", "material"], dropna=False)
    cell_counts = grouped.size().to_dict()
    event_distribution = df["event_label"].value_counts().sort_index().to_dict()
    material_distribution = df["material"].value_counts().sort_index().to_dict()

    cells = []
    empty_cells = []
    sparse_cells = []
    single_outcome_cells = []
    dominant_outcome_cells = []
    for shape, action, material in grid:
        count = int(cell_counts.get((shape, action, material), 0))
        cell_df = df[(df["shape"] == shape) & (df["action"] == action) & (df["material"] == material)]
        event_counts = cell_df["event_label"].value_counts().sort_index().to_dict()
        dominant_fraction = 0.0
        if count:
            dominant_fraction = max(event_counts.values()) / count
        cell = {
            "shape": shape,
            "action": action,
            "material": material,
            "count": count,
            "event_counts": {str(key): int(value) for key, value in event_counts.items()},
            "dominant_event_fraction": float(dominant_fraction),
        }
        cells.append(cell)
        label = f"{shape} x {action} x {material}"
        if count == 0:
            empty_cells.append(label)
        elif count < sparse_threshold:
            sparse_cells.append({"cell": label, "count": count})
        if count > 0 and len(event_counts) == 1:
            single_outcome_cells.append(label)
        elif count > 0 and dominant_fraction >= dominant_threshold:
            dominant_outcome_cells.append({"cell": label, "dominant_fraction": float(dominant_fraction)})

    return {
        "n_episodes": int(len(df)),
        "event_distribution": {str(key): int(value) for key, value in event_distribution.items()},
        "material_distribution": {str(key): int(value) for key, value in material_distribution.items()},
        "grid": {
            "shape_count": len(shapes),
            "action_count": len(actions),
            "material_count": len(materials),
            "expected_cells": len(grid),
            "observed_cells": sum(1 for cell in cells if cell["count"] > 0),
            "coverage_fraction": float(sum(1 for cell in cells if cell["count"] > 0) / max(1, len(grid))),
            "sparse_threshold": int(sparse_threshold),
            "dominant_threshold": float(dominant_threshold),
        },
        "degenerate_cells": {
            "empty": empty_cells,
            "sparse": sparse_cells,
            "single_outcome": single_outcome_cells,
            "dominant_outcome": dominant_outcome_cells,
        },
        "cells": cells,
    }


def coverage_report_text(report: Dict[str, Any]) -> str:
    grid = report["grid"]
    degenerate = report["degenerate_cells"]
    lines = [
        "# Coverage / Diversity Report",
        "",
        f"Episodes: {report['n_episodes']}",
        f"Grid coverage: {grid['observed_cells']}/{grid['expected_cells']} cells ({grid['coverage_fraction']:.1%})",
        f"Sparse cells (< {grid['sparse_threshold']} episodes): {len(degenerate['sparse'])}",
        f"Empty cells: {len(degenerate['empty'])}",
        f"Single-outcome cells: {len(degenerate['single_outcome'])}",
        f"Dominant-outcome cells (>= {grid['dominant_threshold']:.0%} one label): {len(degenerate['dominant_outcome'])}",
        "",
        "Event distribution:",
    ]
    for event, count in report["event_distribution"].items():
        lines.append(f"- {event}: {count}")
    lines.extend(["", "Material distribution:"])
    for material, count in report["material_distribution"].items():
        lines.append(f"- {material}: {count}")
    if degenerate["empty"]:
        lines.extend(["", "Empty cells:"])
        lines.extend(f"- {cell}" for cell in degenerate["empty"])
    if degenerate["sparse"]:
        lines.extend(["", "Sparse cells:"])
        lines.extend(f"- {cell['cell']}: {cell['count']}" for cell in degenerate["sparse"])
    if degenerate["single_outcome"]:
        lines.extend(["", "Single-outcome cells:"])
        lines.extend(f"- {cell}" for cell in degenerate["single_outcome"][:50])
        if len(degenerate["single_outcome"]) > 50:
            lines.append(f"- ... {len(degenerate['single_outcome']) - 50} more")
    return "\n".join(lines)
