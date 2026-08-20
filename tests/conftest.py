"""Collection policy for the split current/legacy test suites.

The project tests are intentionally kept as unittest modules.  This hook adds
pytest markers without changing their test implementation or import behavior.
"""

from pathlib import Path


CURRENT_AGENTFLOW_TESTS = frozenset(
    {
        "test_agent_behavioral_loop.py",
        "test_agentflow_v1.py",
        "test_agentflow_wsi_runtime.py",
        "test_reviewer_contract_schemas.py",
        "test_mucosa_extractor.py",
        "test_architecture_experiment.py",
        "test_architecture_models.py",
        "test_11_class_workflow.py",
        "test_dual_branch_dysplasia.py",
    }
)


def pytest_collection_modifyitems(config, items):
    agentflow = config.getoption("-m", default="")
    del agentflow  # marker selection is handled by pytest after this hook.
    import pytest

    for item in items:
        path = getattr(item, "path", None)
        if path is None:
            path = getattr(item, "fspath", "")
        filename = Path(str(path)).name
        marker = "agentflow" if filename in CURRENT_AGENTFLOW_TESTS else "legacy"
        item.add_marker(getattr(pytest.mark, marker))
