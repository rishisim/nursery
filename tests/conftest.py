from __future__ import annotations

from pathlib import Path

import pytest


RETIRED_QUALIFICATION_MODULES = {
    "test_synthetic_identifiability_qualification_v6.py": (
        "V6 is preserved as INVALIDATED_PRE_FREEZE; its retired identifiers "
        "must not be executed or repaired in place."
    )
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = Path(str(item.path))
        reason = RETIRED_QUALIFICATION_MODULES.get(path.name)
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))
