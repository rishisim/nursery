"""Episode tables and visibility computed from rendered instance IDs."""

import json

import numpy as np


FIELDS = {
    "frames": [("video_id", "string"), ("frame_idx", "int64"), ("image", "string"),
               ("subject", "string"), ("room", "string"), ("clip", "string"),
               ("contact_left", "bool"), ("contact_right", "bool"),
               ("held_category", "string"), ("objects", "string")],
    "sessions": [("video_id", "string"), ("subject", "string"), ("room", "string"),
                 ("split", "string"), ("clips", "string"), ("distractors", "string"),
                 ("duration_s", "int64")],
    "probes": [("probe_id", "string"), ("image", "string"), ("category", "string"),
               ("room", "string"), ("split", "string")],
}


def table(rows, name):
    import pyarrow as pa

    schema = pa.schema([pa.field(key, getattr(pa, "bool_" if kind == "bool" else kind)(), nullable=key == "held_category")
                        for key, kind in FIELDS[name]])
    for row in rows:
        if set(row) != set(schema.names):
            raise ValueError(f"Invalid {name} columns")
        if any(row[key] is None for key in schema.names if key != "held_category"):
            raise ValueError("Unexpected null")
    return pa.Table.from_pylist(rows, schema=schema)


def visible_objects(mask, categories):
    if mask.ndim != 2 or min(mask.shape) < 4:
        raise ValueError("Expected an instance-ID image at least 4 x 4")
    height, width = mask.shape
    result = []
    for instance, category in categories.items():
        y, x = np.nonzero(mask == instance)
        cells = np.unique(np.minimum(y * 4 // height, 3) * 4 +
                              np.minimum(x * 4 // width, 3)).tolist()
        result.append({"category": category, "visible_px": len(x), "cells": cells})
    return json.dumps(result)
