"""Aria Everyday Activities real-data integration.

AEA is an adult, partly scripted sensor-format analogue.  Nothing in this
package treats it as developmental evidence.
"""

from babyworld_lite.aea.manifest import (
    AEASequenceId,
    build_balanced_subset_plan,
    load_safe_manifest,
)

__all__ = ["AEASequenceId", "build_balanced_subset_plan", "load_safe_manifest"]
