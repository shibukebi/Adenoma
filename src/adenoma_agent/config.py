from pathlib import Path

import yaml

from adenoma_agent.utils import ensure_dir, getenv_or_default, read_json, write_json


DEFAULT_RUNTIME_PATH = Path("/data15/data15_5/yuexin2/adenoma_agent/configs/runtime.yaml")
DEFAULT_BUDGET_PATH = Path("/data15/data15_5/yuexin2/adenoma_agent/configs/budget.yaml")


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_bundle(runtime_path=None, budget_path=None):
    runtime_path = Path(runtime_path or getenv_or_default("ADENOMA_AGENT_RUNTIME", DEFAULT_RUNTIME_PATH))
    budget_path = Path(budget_path or getenv_or_default("ADENOMA_AGENT_BUDGET", DEFAULT_BUDGET_PATH))
    runtime = _load_yaml(runtime_path)
    budget = _load_yaml(budget_path)
    bundle = {
        "runtime_path": str(runtime_path),
        "budget_path": str(budget_path),
        "runtime": runtime,
        "budget": budget.get("budget", {}),
    }
    ensure_runtime_dirs(bundle)
    return bundle


def ensure_runtime_dirs(bundle):
    runtime = bundle["runtime"]
    project_root = Path(runtime["project"]["project_root"])
    artifacts_root = Path(runtime["project"]["artifacts_root"])
    ensure_dir(project_root)
    ensure_dir(artifacts_root)
    ensure_dir(Path(runtime["data"]["pilot_root"]))
    ensure_dir(Path(runtime["cache"]["thumbnail_cache_root"]))
    ensure_dir(Path(runtime["cache"]["trace_cache_root"]))
    ensure_dir(Path(runtime["cache"]["description_cache_root"]))


def budget_value(bundle, key, default=None):
    return bundle.get("budget", {}).get(key, default)


def runtime_value(bundle, section, key, default=None):
    return bundle.get("runtime", {}).get(section, {}).get(key, default)


def save_run_metadata(path, payload):
    return write_json(path, payload)


def load_run_metadata(path):
    return read_json(path)
