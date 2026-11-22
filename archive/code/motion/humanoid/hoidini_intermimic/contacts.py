"""Preserve hand-contact intent; do not invent whole-body contact labels."""

import numpy as np


def map_contacts(scores, body_names, threshold=0.4):
    if scores.ndim != 2 or scores.shape[1] != 60 or not np.isfinite(scores).all():
        raise ValueError("Expected 60 finite HOIDiNi hand-anchor scores")
    # Native scores are unconstrained predictions, not calibrated probabilities.
    human = np.zeros((len(scores), 52), dtype=np.float32)
    for hand, span in (("L", slice(0, 30)), ("R", slice(30, 60))):
        human[:, body_names.index(f"{hand}_Wrist")] = np.any(scores[:, span] > threshold, axis=1)
    # InterMimic uses 0 for unconstrained, +1 for desired contact, -1 for
    # forbidden contact. A missing anchor prediction cannot prove -1.
    obj = np.any(human > 0, axis=1).astype(np.float32)
    return human, obj
