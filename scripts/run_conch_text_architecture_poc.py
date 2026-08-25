#!/usr/bin/env python3
"""Entry point for Expert-Confirmed 5x Architecture Benchmark v1."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from adenoma_agent.conch_text_architecture_poc.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
