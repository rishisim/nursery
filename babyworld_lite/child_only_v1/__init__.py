"""Child-only, corpus-neutral construction scaffold.

This namespace is deliberately independent of every historical adult-data and
feature-level study line.  It contains construction validators and interfaces,
not a scientific outcome runner.
"""

from .policy import (
    CONSTRUCTION_PROFILE,
    CONTRACT_VERSION,
    PolicyViolation,
    validate_provenance,
)

__all__ = [
    "CONSTRUCTION_PROFILE",
    "CONTRACT_VERSION",
    "PolicyViolation",
    "validate_provenance",
]
