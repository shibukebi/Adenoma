#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_FORMAT_COLUMNS = (
    "task_mode",
    "stream_a_name",
    "stream_b_name",
    "selection_mode",
    "route_c_status",
)

DEFAULT_DATA_DIR = Path("/data15/data15_5/yuexin2/adenoma/data")
DEFAULT_CANONICAL_READY_CSV = DEFAULT_DATA_DIR / "clam_ssl_others_uni_ready.csv"
DEFAULT_CANONICAL_OUTPUT_DIR = DEFAULT_DATA_DIR / "canonical_uni_splits_8_2"
DEFAULT_CANONICAL_FEATURE_DIRS = (
    "/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_uni",
    "/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_10x_uni",
    "/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_5x_uni",
    "/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_2p5x_uni",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one deterministic canonical stratified 8:2 train/test split."
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        default=[],
        help=(
            "Input ready csv. Can be passed more than once and will be merged by slide_id in "
            "canonical mode. Defaults to the canonical UNI ready csv."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory searched when --input-csv is not provided.",
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_CANONICAL_READY_CSV.name,
        help="Input glob under --data-dir when --input-csv is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_CANONICAL_OUTPUT_DIR),
        help=(
            "Directory for the single canonical split. Pass an empty string to restore the "
            "legacy behavior that writes one split directory per input csv."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default="_splits_8_2",
        help="Legacy-mode suffix appended after removing _ready/_dataset from the input stem.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index in output file names.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Held-out test ratio.")
    parser.add_argument(
        "--reference-csv",
        default="",
        help=(
            "Optional canonical ready csv used to create the split. "
            "Each input csv then reuses the reference split and filters unavailable slide_ids."
        ),
    )
    parser.add_argument(
        "--label-column",
        default="label_name",
        help="Column used as the label. Falls back to label when missing.",
    )
    parser.add_argument(
        "--format-column",
        action="append",
        default=[],
        help="Optional format column for additional stratification. Can be passed more than once.",
    )
    parser.add_argument(
        "--feature-dir",
        action="append",
        default=None,
        help=(
            "UNI feature directory required for canonical eligibility. Can be passed more than "
            "once. The directory may be the feature root or the pt_files directory. Defaults to "
            "the current labeled yx UNI feature directories."
        ),
    )
    parser.add_argument(
        "--feature-subdir",
        default="pt_files",
        help="Subdirectory containing feature files when --feature-dir points to a feature root.",
    )
    parser.add_argument(
        "--feature-extension",
        default=".pt",
        help="Feature file extension used to collect available slide_ids.",
    )
    parser.add_argument(
        "--feature-match-mode",
        choices=("all", "any"),
        default="all",
        help="Require slide_ids to be present in all feature dirs or any feature dir.",
    )
    parser.add_argument(
        "--no-feature-filter",
        action="store_true",
        help="Disable the default canonical UNI feature existence filter.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_id_csv(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for value in values:
            writer.writerow([value])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def output_dir_for(input_csv: Path, suffix: str) -> Path:
    stem = input_csv.stem
    for ending in ("_ready", "_dataset"):
        if stem.endswith(ending):
            stem = stem[: -len(ending)]
            break
    return input_csv.with_name(f"{stem}{suffix}")


def resolve_split_columns(
    fieldnames: list[str],
    *,
    label_column: str,
    source: Path,
) -> tuple[str, str]:
    slide_column = "slide_id" if "slide_id" in fieldnames else "case_id"
    resolved_label_column = label_column if label_column in fieldnames else "label"
    if slide_column not in fieldnames:
        raise KeyError(f"{source} is missing slide_id/case_id")
    if resolved_label_column not in fieldnames:
        raise KeyError(f"{source} is missing {label_column}/label")
    return slide_column, resolved_label_column


def feature_listing_dir(feature_dir: Path, feature_subdir: str) -> Path:
    nested = feature_dir / feature_subdir
    if nested.is_dir():
        return nested
    return feature_dir


def collect_feature_id_sets(
    feature_dirs: list[str],
    *,
    feature_subdir: str,
    feature_extension: str,
) -> tuple[dict[str, set[str]], dict]:
    feature_id_sets: dict[str, set[str]] = {}
    feature_dir_stats = {}
    for feature_dir_raw in feature_dirs:
        feature_root = Path(feature_dir_raw)
        listing_dir = feature_listing_dir(feature_root, feature_subdir)
        if not listing_dir.is_dir():
            raise FileNotFoundError(f"Feature directory does not exist: {listing_dir}")
        feature_ids = {path.stem for path in listing_dir.glob(f"*{feature_extension}") if path.is_file()}
        feature_id_sets[str(feature_root)] = feature_ids
        feature_dir_stats[str(feature_root)] = {
            "listing_dir": str(listing_dir),
            "feature_extension": feature_extension,
            "feature_count": len(feature_ids),
        }
    return feature_id_sets, feature_dir_stats


def filter_rows_by_feature_dirs(
    rows: list[dict[str, str]],
    *,
    slide_column: str,
    feature_dirs: list[str],
    feature_subdir: str,
    feature_extension: str,
    feature_match_mode: str,
) -> tuple[list[dict[str, str]], dict]:
    if not feature_dirs:
        return rows, {"feature_filter_enabled": False}

    feature_id_sets, feature_dir_stats = collect_feature_id_sets(
        feature_dirs,
        feature_subdir=feature_subdir,
        feature_extension=feature_extension,
    )
    ordered_sets = list(feature_id_sets.values())
    if feature_match_mode == "all":
        eligible_feature_ids = set.intersection(*ordered_sets) if ordered_sets else set()
    else:
        eligible_feature_ids = set.union(*ordered_sets) if ordered_sets else set()

    row_ids = {str(row[slide_column]) for row in rows}
    filtered_rows = [row for row in rows if str(row[slide_column]) in eligible_feature_ids]
    missing_by_feature_dir = {
        feature_dir: len(row_ids - feature_ids) for feature_dir, feature_ids in feature_id_sets.items()
    }

    return filtered_rows, {
        "feature_filter_enabled": True,
        "feature_match_mode": feature_match_mode,
        "feature_dirs": feature_dir_stats,
        "eligible_feature_ids": len(eligible_feature_ids),
        "rows_before_feature_filter": len(rows),
        "rows_after_feature_filter": len(filtered_rows),
        "dropped_missing_features": len(rows) - len(filtered_rows),
        "missing_ready_ids_by_feature_dir": missing_by_feature_dir,
    }


def load_canonical_rows(
    input_csvs: list[Path],
    *,
    label_column: str,
) -> tuple[list[dict[str, str]], list[str], dict]:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    seen_by_slide_id: dict[str, dict[str, str]] = {}
    duplicate_slide_ids = 0
    input_stats = []

    for input_csv in input_csvs:
        source_rows, source_fieldnames = read_rows(input_csv)
        slide_column, source_label_column = resolve_split_columns(
            source_fieldnames,
            label_column=label_column,
            source=input_csv,
        )
        for fieldname in source_fieldnames:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
        if "slide_id" not in fieldnames:
            fieldnames.insert(0, "slide_id")
        if label_column not in fieldnames:
            fieldnames.append(label_column)

        for row in source_rows:
            slide_id = str(row[slide_column]).strip()
            label = str(row[source_label_column]).strip()
            if not slide_id:
                raise ValueError(f"{input_csv} contains a row with an empty {slide_column}")
            if not label:
                raise ValueError(f"{input_csv} contains {slide_id} with an empty {source_label_column}")

            normalized = dict(row)
            normalized["slide_id"] = slide_id
            normalized[label_column] = label

            previous = seen_by_slide_id.get(slide_id)
            if previous is not None:
                duplicate_slide_ids += 1
                if previous[label_column] != label:
                    raise ValueError(
                        f"Conflicting labels for {slide_id}: "
                        f"{previous[label_column]} vs {label}"
                    )
                continue

            seen_by_slide_id[slide_id] = normalized
            rows.append(normalized)

        input_stats.append(
            {
                "input_csv": str(input_csv),
                "rows": len(source_rows),
                "slide_column": slide_column,
                "label_column": source_label_column,
            }
        )

    return rows, fieldnames, {
        "input_csvs": input_stats,
        "duplicate_slide_ids_dropped": duplicate_slide_ids,
        "unique_slide_ids_before_feature_filter": len(rows),
    }


def label_counts(rows: list[dict[str, str]], label_column: str) -> dict[str, int]:
    return dict(Counter(str(row[label_column]) for row in rows))


def split_stratum(
    rows: list[dict[str, str]],
    *,
    slide_column: str,
    test_ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str | None]:
    shuffled = sorted(rows, key=lambda row: str(row[slide_column]))
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        return shuffled, [], "stratum has one sample; kept in train because an 8:2 split is impossible"

    n_test = round(len(shuffled) * test_ratio)
    n_test = max(1, min(len(shuffled) - 1, n_test))
    n_train = len(shuffled) - n_test
    return shuffled[:n_train], shuffled[n_train:], None


def build_split(
    rows: list[dict[str, str]],
    *,
    slide_column: str,
    label_column: str,
    format_columns: list[str],
    seed: int,
    test_ratio: float,
) -> tuple[dict[str, list[str]], dict]:
    rng = random.Random(seed)
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(column, "")) for column in [label_column, *format_columns])
        grouped[key].append(row)

    train_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    warnings: list[str] = []
    strata = {}

    for key in sorted(grouped):
        stratum_train, stratum_test, warning = split_stratum(
            grouped[key],
            slide_column=slide_column,
            test_ratio=test_ratio,
            rng=rng,
        )
        train_rows.extend(stratum_train)
        test_rows.extend(stratum_test)
        stratum_name = " | ".join(
            f"{column}={value}" for column, value in zip([label_column, *format_columns], key)
        )
        strata[stratum_name] = {
            "total": len(grouped[key]),
            "train": len(stratum_train),
            "test": len(stratum_test),
        }
        if warning:
            warnings.append(f"{stratum_name}: {warning}")

    train_ids = sorted(str(row[slide_column]) for row in train_rows)
    test_ids = sorted(str(row[slide_column]) for row in test_rows)
    train_id_set = set(train_ids)
    test_id_set = set(test_ids)

    stats = {
        "seed": seed,
        "test_ratio": test_ratio,
        "total_rows": len(rows),
        "format_columns": format_columns,
        "split_sizes": {"train": len(train_ids), "test": len(test_ids)},
        "label_counts": {
            "all": label_counts(rows, label_column),
            "train": label_counts(train_rows, label_column),
            "test": label_counts(test_rows, label_column),
        },
        "strata": strata,
        "overlap_count": len(train_id_set & test_id_set),
        "covered_unique_ids": len(train_id_set | test_id_set),
        "warnings": warnings,
        "formal_ready": bool(rows) and not warnings and not (train_id_set & test_id_set),
    }
    if not rows:
        stats["warnings"].append("input csv has no data rows")
    return {"train": train_ids, "test": test_ids}, stats


def build_filtered_stats(
    rows: list[dict[str, str]],
    *,
    slide_column: str,
    label_column: str,
    format_columns: list[str],
    reference_split_map: dict[str, list[str]],
    reference_stats: dict,
    reference_csv: Path,
) -> tuple[dict[str, list[str]], dict]:
    row_by_id = {str(row[slide_column]): row for row in rows}
    ready_ids = set(row_by_id)
    split_map = {
        split_name: sorted([slide_id for slide_id in ids if slide_id in ready_ids])
        for split_name, ids in reference_split_map.items()
    }
    train_ids = split_map["train"]
    test_ids = split_map["test"]
    train_rows = [row_by_id[slide_id] for slide_id in train_ids]
    test_rows = [row_by_id[slide_id] for slide_id in test_ids]
    train_id_set = set(train_ids)
    test_id_set = set(test_ids)
    warnings = []
    if not rows:
        warnings.append("input csv has no data rows")
    dropped_by_split = {
        split_name: len(reference_split_map[split_name]) - len(split_map[split_name])
        for split_name in reference_split_map
    }

    stats = {
        "reference_csv": str(reference_csv),
        "reference_total_rows": reference_stats["total_rows"],
        "reference_split_sizes": reference_stats["split_sizes"],
        "seed": reference_stats["seed"],
        "test_ratio": reference_stats["test_ratio"],
        "total_rows": len(rows),
        "format_columns": format_columns,
        "split_sizes": {"train": len(train_ids), "test": len(test_ids)},
        "label_counts": {
            "all": label_counts(rows, label_column),
            "train": label_counts(train_rows, label_column),
            "test": label_counts(test_rows, label_column),
        },
        "dropped_from_reference_by_split": dropped_by_split,
        "overlap_count": len(train_id_set & test_id_set),
        "covered_unique_ids": len(train_id_set | test_id_set),
        "warnings": warnings,
        "formal_ready": bool(rows) and not warnings and not (train_id_set & test_id_set),
    }
    return split_map, stats


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input_csv:
        return sorted(Path(path) for path in args.input_csv)
    return sorted(Path(args.data_dir).glob(args.glob))


def main() -> None:
    args = parse_args()
    if not 0 < args.test_ratio < 1:
        raise ValueError("--test-ratio must be between 0 and 1")
    if args.no_feature_filter:
        feature_dirs = []
    elif args.feature_dir is None:
        feature_dirs = list(DEFAULT_CANONICAL_FEATURE_DIRS)
    else:
        feature_dirs = args.feature_dir

    input_csvs = collect_inputs(args)
    if not input_csvs:
        raise FileNotFoundError("No input csv files found.")

    if args.output_dir:
        if args.reference_csv:
            raise ValueError("--reference-csv is only supported in legacy per-input output mode")

        rows, fieldnames, canonical_stats = load_canonical_rows(
            input_csvs,
            label_column=args.label_column,
        )
        rows, feature_stats = filter_rows_by_feature_dirs(
            rows,
            slide_column="slide_id",
            feature_dirs=feature_dirs,
            feature_subdir=args.feature_subdir,
            feature_extension=args.feature_extension,
            feature_match_mode=args.feature_match_mode,
        )
        requested_format_columns = args.format_column or list(DEFAULT_FORMAT_COLUMNS)
        format_columns = [column for column in requested_format_columns if column in fieldnames]
        split_map, split_stats = build_split(
            rows,
            slide_column="slide_id",
            label_column=args.label_column,
            format_columns=format_columns,
            seed=args.seed,
            test_ratio=args.test_ratio,
        )

        out_dir = Path(args.output_dir)
        write_id_csv(out_dir / f"flod-{args.fold}-train.csv", split_map["train"])
        write_id_csv(out_dir / f"flod-{args.fold}-test.csv", split_map["test"])
        stats = {
            "canonical_mode": True,
            "output_dir": str(out_dir),
            **canonical_stats,
            **feature_stats,
            **split_stats,
        }
        write_json(out_dir / f"flod-{args.fold}_stats.json", stats)
        summary = [
            {
                "output_dir": str(out_dir),
                "split_sizes": split_stats["split_sizes"],
                "label_counts": split_stats["label_counts"],
                "feature_filter_enabled": feature_stats["feature_filter_enabled"],
                "formal_ready": split_stats["formal_ready"],
                "warnings": split_stats["warnings"],
            }
        ]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    reference_split_map = None
    reference_stats = None
    reference_csv = Path(args.reference_csv) if args.reference_csv else None
    if reference_csv is not None:
        reference_rows, reference_fieldnames = read_rows(reference_csv)
        reference_slide_column, reference_label_column = resolve_split_columns(
            reference_fieldnames,
            label_column=args.label_column,
            source=reference_csv,
        )
        requested_format_columns = args.format_column or list(DEFAULT_FORMAT_COLUMNS)
        reference_format_columns = [
            column for column in requested_format_columns if column in reference_fieldnames
        ]
        reference_split_map, reference_stats = build_split(
            reference_rows,
            slide_column=reference_slide_column,
            label_column=reference_label_column,
            format_columns=reference_format_columns,
            seed=args.seed,
            test_ratio=args.test_ratio,
        )

    summary = []
    for input_csv in input_csvs:
        rows, fieldnames = read_rows(input_csv)
        slide_column, label_column = resolve_split_columns(
            fieldnames,
            label_column=args.label_column,
            source=input_csv,
        )

        requested_format_columns = args.format_column or list(DEFAULT_FORMAT_COLUMNS)
        format_columns = [column for column in requested_format_columns if column in fieldnames]
        if reference_split_map is None or reference_stats is None or reference_csv is None:
            split_map, stats = build_split(
                rows,
                slide_column=slide_column,
                label_column=label_column,
                format_columns=format_columns,
                seed=args.seed,
                test_ratio=args.test_ratio,
            )
        else:
            split_map, stats = build_filtered_stats(
                rows,
                slide_column=slide_column,
                label_column=label_column,
                format_columns=format_columns,
                reference_split_map=reference_split_map,
                reference_stats=reference_stats,
                reference_csv=reference_csv,
            )

        out_dir = output_dir_for(input_csv, args.output_suffix)
        write_id_csv(out_dir / f"flod-{args.fold}-train.csv", split_map["train"])
        write_id_csv(out_dir / f"flod-{args.fold}-test.csv", split_map["test"])
        write_json(out_dir / f"flod-{args.fold}_stats.json", {"input_csv": str(input_csv), **stats})
        summary.append(
            {
                "input_csv": str(input_csv),
                "output_dir": str(out_dir),
                "split_sizes": stats["split_sizes"],
                "label_counts": stats["label_counts"],
                "formal_ready": stats["formal_ready"],
                "warnings": stats["warnings"],
            }
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
