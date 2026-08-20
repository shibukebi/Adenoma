#!/usr/bin/env python3
"""Run and strictly verify the current AgentFlow CI baseline.

This command is intentionally independent of the legacy suite.  It fails on
any failure, error, or skip so missing test dependencies cannot be mistaken
for a green baseline.
"""

import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT_TESTS = [
    "tests/test_agent_behavioral_loop.py",
    "tests/test_agentflow_v1.py",
    "tests/test_agentflow_wsi_runtime.py",
    "tests/test_reviewer_contract_schemas.py",
    "tests/test_mucosa_extractor.py",
    "tests/test_architecture_experiment.py",
    "tests/test_architecture_models.py",
    "tests/test_11_class_workflow.py",
    "tests/test_dual_branch_dysplasia.py",
]


def _preflight():
    missing = []
    for module in ("pytest", "jsonschema", "torch"):
        try:
            __import__(module)
        except Exception as exc:  # pragma: no cover - environment-specific
            missing.append("{0}: {1}".format(module, exc))
    if missing:
        raise SystemExit("AgentFlow CI dependency preflight failed:\n- " + "\n- ".join(missing))
    try:
        __import__("adenoma_agent.architecture_models")
    except Exception as exc:  # pragma: no cover - environment-specific
        raise SystemExit("Architecture model preflight failed: {0}".format(exc))


def _summary(xml_path):
    root = ET.parse(str(xml_path)).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for key in totals:
            totals[key] += int(suite.attrib.get(key, 0) or 0)
    return totals


def _run(name, expected, args):
    with tempfile.NamedTemporaryFile(prefix="agentflow-", suffix=".xml", delete=False) as handle:
        xml_path = Path(handle.name)
    try:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--strict-markers",
            "--junitxml",
            str(xml_path),
        ] + list(args)
        print("\n[AgentFlow CI] {0}: {1}".format(name, " ".join(command)))
        completed = subprocess.run(command, cwd=str(ROOT), env=env)
        summary = _summary(xml_path)
        print("[AgentFlow CI] {0}: {1}".format(name, summary))
        if completed.returncode != 0 or summary != {
            "tests": expected,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
        }:
            raise SystemExit("AgentFlow CI baseline mismatch in {0}".format(name))
    finally:
        xml_path.unlink(missing_ok=True)


def main():
    _preflight()
    _run("reviewer-contract", 11, ["-m", "agentflow", "tests/test_reviewer_contract_schemas.py"])
    _run("behavioral-control-plane", 17, ["-m", "agentflow", "tests/test_agent_behavioral_loop.py"])
    _run("focused-regression", 81, ["-m", "agentflow"] + CURRENT_TESTS)
    print("\nAgentFlow CI PASS: Reviewer 11/11, Behavioral 17/17, Focused 81/81")


if __name__ == "__main__":
    main()
