#!/usr/bin/env python3
"""Aggregate per-fold benchmark metrics into publication tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "CLAM-SB": ["2p5x", "5x", "10x", "20x"],
    "TransMIL": ["2p5x", "5x", "10x", "20x"],
    "DSMIL": ["2p5x", "5x", "10x", "20x"],
    "MIST": ["2p5x_5x", "5x_10x"],
}
LEGACY_NAMES = {"CLAM-SB": "CLAM-SB", "TransMIL": "transmil", "DSMIL": "dsmil", "MIST": "MIST"}
METRICS = ["accuracy", "macro_f1", "weighted_f1", "auc_ovr_macro", "auc_ovr_weighted"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=BENCHMARK_ROOT / "results")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def resolve_metrics(
    results_root: Path,
    legacy_root: Path | None,
    model: str,
    feature: str,
    fold: int,
) -> tuple[Path | None, str | None, str | None]:
    primary_candidates = [
        results_root / model / feature / f"fold-{fold}" / "metrics.json",
        results_root / model / "11class" / feature / f"fold-{fold}" / "metrics.json",
    ]
    for path in primary_candidates:
        if path.exists():
            return path, "primary", str(path.relative_to(results_root))
    if fold != 4 or legacy_root is None:
        return None, None, None
    legacy_model = LEGACY_NAMES[model]
    if model == "MIST":
        base = legacy_root / "MIST" if feature == "2p5x_5x" else legacy_root / "MIST/5x_10x"
    else:
        base = legacy_root / legacy_model / "11class" / feature
    path = base / "metrics.json"
    if path.exists():
        return path, "legacy_fold5", str(path.relative_to(legacy_root))
    return None, None, None


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    legacy_root = args.legacy_root.resolve() if args.legacy_root else None
    rows = []
    missing = []
    for model, features in EXPERIMENTS.items():
        for feature in features:
            for fold in range(5):
                path, source_root, relative_path = resolve_metrics(results_root, legacy_root, model, feature, fold)
                if path is None:
                    missing.append(f"{model}/{feature}/fold-{fold}/metrics.json")
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                row = {
                    "model": model,
                    "feature": feature,
                    "fold": fold,
                    "source_root": source_root,
                    "metrics_path": relative_path,
                }
                row.update({metric: payload.get(metric) for metric in METRICS})
                rows.append(row)
    if args.strict and missing:
        raise SystemExit("Missing metric files:\n" + "\n".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_fold_path = args.output_dir / "per_fold_metrics.csv"
    with per_fold_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "feature", "fold", *METRICS, "source_root", "metrics_path"])
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for model, features in EXPERIMENTS.items():
        for feature in features:
            selected = [row for row in rows if row["model"] == model and row["feature"] == feature]
            summary = {"model": model, "feature": feature, "folds_found": len(selected)}
            for metric in METRICS:
                values = [float(row[metric]) for row in selected if row.get(metric) is not None]
                summary[f"{metric}_mean"] = mean(values) if values else None
                summary[f"{metric}_sd"] = stdev(values) if len(values) > 1 else (0.0 if values else None)
            summary_rows.append(summary)
    summary_path = args.output_dir / "benchmark_summary.csv"
    fields = ["model", "feature", "folds_found"] + [item for metric in METRICS for item in (f"{metric}_mean", f"{metric}_sd")]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    status = {"per_fold": str(per_fold_path), "summary": str(summary_path), "runs_found": len(rows), "missing": len(missing)}
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
