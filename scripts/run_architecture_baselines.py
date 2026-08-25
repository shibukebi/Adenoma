#!/usr/bin/env python3
"""Entry point for Frozen-CONCH 5x architecture baseline commands."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from adenoma_agent.architecture_baselines.conch import ensure_cuda_linker_environment  # noqa: E402


ensure_cuda_linker_environment()

from adenoma_agent.architecture_baselines.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
