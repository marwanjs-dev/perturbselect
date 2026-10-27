"""Keep local-data checks discoverable in VS Code but opt-in at execution."""
import os

import pytest


def pytest_collection_modifyitems(items):
    if os.environ.get("PERTURBSELECT_RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="Set PERTURBSELECT_RUN_INTEGRATION=1 in VS Code's test environment to opt in.")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
