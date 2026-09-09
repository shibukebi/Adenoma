#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Route C CLAM SSL/others dataset, ready manifest, and cleaned splits.")
    parser.add_argument("--label-csv", required=True)
    parser.add_argument("--route-manifest-csv", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--raw-fold-dir", required=True)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--ready-csv", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--min-positive-per-split", type=int, default=5)
    return parser.parse_args()


def read_dict_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_id_list(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row[0].strip() for row in csv.reader(handle) if row and row[0].strip()]


def write_dict_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_id_csv(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for value in values:
            writer.writerow([value])


def main() -> None:
    args = parse_args()
    labels = read_dict_csv(Path(args.label_csv))
    route_manifest = read_dict_csv(Path(args.route_manifest_csv))
    feature_dir = Path(args.feature_dir)
    feat_h5_ids = {path.stem for path in (feature_dir / "h5_files").glob("*.h5")}
    feat_pt_ids = {path.stem for path in (feature_dir / "pt_files").glob("*.pt")}

    route_map = {row["slide_id"]: row for row in route_manifest}
    dataset_rows = []
    ready_rows = []

    for row in labels:
        slide_id = row["slide_id"].strip()
        route_row = route_map.get(slide_id)
        selection_mode = route_row["mode_used"] if route_row else ""
        route_status = route_row["status"] if route_row else ""
        record = {
            "case_id": slide_id,
            "slide_id": slide_id,
            "label": row["label_name"].strip(),
            "label_name": row["label_name"].strip(),
            "type": row["type"].strip(),
            "grade": row["grade"].strip().lower(),
            "selection_mode": selection_mode,
            "route_c_status": route_status,
        }
        dataset_rows.append(record)
        if route_row and route_status in {"success_patho_r1", "success_fallback"} and slide_id in feat_h5_ids and slide_id in feat_pt_ids:
            ready_rows.append(record)

    fieldnames = ["case_id", "slide_id", "label", "label_name", "type", "grade", "selection_mode", "route_c_status"]
    dataset_rows.sort(key=lambda row: row["slide_id"])
    ready_rows.sort(key=lambda row: row["slide_id"])
    write_dict_csv(Path(args.dataset_csv), dataset_rows, fieldnames)
    write_dict_csv(Path(args.ready_csv), ready_rows, fieldnames)

    ready_ids = {row["slide_id"] for row in ready_rows}
    label_name_map = {row["slide_id"]: row["label_name"] for row in dataset_rows}
    mode_map = {row["slide_id"]: row["selection_mode"] for row in dataset_rows}
    split_dir = Path(args.split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "dataset_rows": len(dataset_rows),
        "ready_rows": len(ready_rows),
        "missing_feature_h5": len([row for row in dataset_rows if row["slide_id"] not in feat_h5_ids]),
        "missing_feature_pt": len([row for row in dataset_rows if row["slide_id"] not in feat_pt_ids]),
        "selection_mode_counts_ready": dict(Counter(row["selection_mode"] for row in ready_rows)),
        "splits": {},
        "formal_ready": True,
        "warnings": [],
    }

    for split_name in ["train", "val", "test"]:
        raw_ids = read_id_list(Path(args.raw_fold_dir) / f"flod-0-{split_name}.csv")
        filtered_ids = sorted([slide_id for slide_id in raw_ids if slide_id in ready_ids])
        write_id_csv(split_dir / f"flod-0-{split_name}.csv", filtered_ids)
        label_counts = dict(Counter(label_name_map[slide_id] for slide_id in filtered_ids))
        mode_counts = dict(Counter(mode_map[slide_id] for slide_id in filtered_ids))
        stats["splits"][split_name] = {
            "raw_count": len(raw_ids),
            "filtered_count": len(filtered_ids),
            "dropped_count": len(raw_ids) - len(filtered_ids),
            "label_counts": label_counts,
            "selection_mode_counts": mode_counts,
        }
        if label_counts.get("SSL", 0) < args.min_positive_per_split:
            stats["formal_ready"] = False
            stats["warnings"].append(f"{split_name} split has only {label_counts.get('SSL', 0)} SSL samples (< {args.min_positive_per_split})")
        if label_counts.get("others", 0) == 0 or label_counts.get("SSL", 0) == 0:
            stats["formal_ready"] = False
            stats["warnings"].append(f"{split_name} split is missing at least one class")

    stats_path = split_dir / "flod-0_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dataset_csv={args.dataset_csv}")
    print(f"ready_csv={args.ready_csv}")
    print(f"stats_json={stats_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
