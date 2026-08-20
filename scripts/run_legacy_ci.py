#!/usr/bin/env python3
"""Run legacy tests with an explicit, reviewable failure baseline."""

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PATH = ROOT / "tests" / "legacy_expected_failures.txt"


class _FailureCollector:
    def __init__(self):
        self.failed = set()

    def pytest_runtest_logreport(self, report):
        if report.when in ("setup", "call") and report.failed:
            self.failed.add(report.nodeid)


def _expected():
    return {
        line.strip()
        for line in EXPECTED_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def main():
    try:
        import pytest
    except Exception as exc:
        raise SystemExit("Legacy CI requires pytest: {0}".format(exc))

    collector = _FailureCollector()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    old_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = env["PYTHONPATH"]
    try:
        # Legacy runs remain compatible with older local pytest installations;
        # the marker declarations are still enforced by the current CI job.
        exit_code = pytest.main(["-q", "-m", "legacy"], plugins=[collector])
    finally:
        if old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pythonpath

    expected = _expected()
    unexpected = sorted(collector.failed - expected)
    retired = sorted(expected - collector.failed)
    if unexpected or retired:
        if unexpected:
            print("Unexpected legacy failures:")
            print("\n".join("- " + item for item in unexpected))
        if retired:
            print("Retired legacy failures still listed in baseline:")
            print("\n".join("- " + item for item in retired))
        return 1
    if exit_code not in (0, 1):
        return exit_code
    print("Legacy CI PASS: only the reviewed expected-failure baseline is present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
