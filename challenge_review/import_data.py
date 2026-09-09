#!/usr/bin/env python3
from collections import Counter
import argparse
import csv
import os
from pathlib import Path

from .config import (
    CHALLENGE_CSV,
    CLASS_BY_ID,
    HP_WSI_ROOTS,
    METADATA_CSV,
    OLD_RESULT_ROOT,
    RESULT_ROOT,
    YX_WSI_ROOTS,
)
from .db import initialize_database, transaction, utc_now


EXPERIMENTS = []
for model, old_model in [("CLAM-SB", "CLAM-SB"), ("TransMIL", "transmil"), ("DSMIL", "dsmil")]:
    for feature in ["2p5x", "5x", "10x", "20x"]:
        EXPERIMENTS.append((model, old_model, feature))
for feature in ["2p5x_5x", "5x_10x"]:
    EXPERIMENTS.append(("MIST", "MIST", feature))

PROBABILITY_COLUMNS = {
    "prob_ssl": "SSL",
    "prob_hp": "HP",
    "prob_tsa": "TSA",
    "prob_usa": "USA",
    "prob_ta": "TA",
    "prob_tva": "TVA",
    "prob_ip": "IP",
    "prob_ssl_with_highgrade_dysplasia": "SSLD",
    "prob_tsa_with_highgrade_dysplasia": "TSAD",
    "prob_ta_with_highgrade_dysplasia": "TAD",
    "prob_tva_with_highgrade_dysplasia": "TVAD",
}


def prediction_paths(model, old_model, feature):
    if model == "MIST":
        return [
            RESULT_ROOT / model / "11class" / feature / f"fold-{fold}" / "predictions.csv"
            for fold in range(5)
        ]
    paths = [
        RESULT_ROOT / model / "11class" / feature / f"fold-{fold}" / "predictions.csv"
        for fold in range(4)
    ]
    paths.append(OLD_RESULT_ROOT / old_model / "11class" / feature / "predictions.csv")
    return paths


def read_csv_by_id(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["slide_id"]: row for row in csv.DictReader(handle)}


def scan_wsi_roots(challenge_rows, metadata):
    wanted = {"hp": set(), "yx": set()}
    for row in challenge_rows:
        slide_id = row["slide_id"]
        source = metadata.get(slide_id, {}).get("scanner", "")
        extension = ".isyntax" if source == "hp" else ".svs"
        wanted.setdefault(source, set()).add(f"{slide_id}{extension}")

    resolved = {}
    for source, roots in [("hp", HP_WSI_ROOTS), ("yx", YX_WSI_ROOTS)]:
        remaining = set(wanted[source])
        for root in roots:
            if not remaining or not root.exists():
                continue
            print(f"Scanning WSI root: {root}", flush=True)
            try:
                with os.scandir(root) as entries:
                    for entry in entries:
                        if entry.name not in remaining:
                            continue
                        try:
                            if entry.stat().st_size <= 0:
                                continue
                        except OSError:
                            continue
                        resolved[entry.name.rsplit(".", 1)[0]] = (
                            entry.path,
                            Path(entry.name).suffix.lstrip("."),
                            1,
                        )
                        remaining.remove(entry.name)
            except OSError as exc:
                print(f"Unable to scan {root}: {exc}", flush=True)
        for filename in remaining:
            slide_id = filename.rsplit(".", 1)[0]
            resolved[slide_id] = (None, Path(filename).suffix.lstrip("."), 0)
    return resolved


def load_challenge_rows():
    with CHALLENGE_CSV.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 3153:
        raise ValueError(f"Expected 3153 challenge slides, found {len(rows)}")
    return rows


def load_predictions(challenge_ids):
    records = []
    counts = Counter()
    for model, old_model, feature in EXPERIMENTS:
        configuration = f"{model} {feature}"
        seen = set()
        for path in prediction_paths(model, old_model, feature):
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    slide_id = row["slide_id"]
                    if row.get("split", "").lower() != "test" or slide_id not in challenge_ids:
                        continue
                    if slide_id in seen:
                        raise ValueError(f"Duplicate prediction for {configuration}: {slide_id}")
                    seen.add(slide_id)
                    predicted_id = int(float(row["pred"]))
                    label_id = int(float(row["label"]))
                    predicted_label = CLASS_BY_ID[predicted_id]
                    probabilities = {
                        label: float(row[column])
                        for column, label in PROBABILITY_COLUMNS.items()
                    }
                    second_label, second_confidence = max(
                        (
                            (label, confidence)
                            for label, confidence in probabilities.items()
                            if label != predicted_label
                        ),
                        key=lambda item: item[1],
                    )
                    records.append(
                        (
                            slide_id,
                            model,
                            feature,
                            configuration,
                            predicted_label,
                            float(row["confidence"]),
                            second_label,
                            second_confidence,
                            int(predicted_id == label_id),
                        )
                    )
                    counts[slide_id] += 1
        if seen != challenge_ids:
            missing = sorted(challenge_ids - seen)[:10]
            raise ValueError(f"{configuration} is missing {len(challenge_ids - seen)} slides: {missing}")

    invalid = {slide_id: count for slide_id, count in counts.items() if count != 14}
    if invalid:
        raise ValueError(f"Slides without 14 predictions: {list(invalid.items())[:10]}")
    return records


def import_all():
    initialize_database()
    challenge_rows = load_challenge_rows()
    challenge_ids = {row["slide_id"] for row in challenge_rows}
    metadata = read_csv_by_id(METADATA_CSV)
    wsi_index = scan_wsi_roots(challenge_rows, metadata)
    now = utc_now()
    slide_records = []
    missing_wsi = []
    for row in challenge_rows:
        slide_id = row["slide_id"]
        meta = metadata.get(slide_id, {})
        source = meta.get("scanner", "")
        wsi_path, wsi_format, available = wsi_index[slide_id]
        if not available:
            missing_wsi.append(slide_id)
        slide_records.append(
            (
                slide_id,
                row["true_class"],
                row.get("type") or meta.get("type"),
                row.get("grade") or meta.get("grade"),
                source,
                int(row["cv_fold"]),
                wsi_path,
                wsi_format,
                available,
                int(row["hardness_rank"]),
                int(row["wrong_configurations"]),
                int(row["total_configurations"]),
                float(row["wrong_pct"]),
                row["consensus_wrong_class"],
                int(row["consensus_wrong_count"]),
                float(row["mean_wrong_confidence"]),
                float(row["max_wrong_confidence"]),
                row["all_prediction_summary"],
                now,
            )
        )

    print("Loading 14 model/magnification predictions...", flush=True)
    predictions = load_predictions(challenge_ids)
    with transaction() as connection:
        connection.executemany(
            """
            INSERT INTO slides(
                slide_id, original_label, pathology_type, grade, source, cv_fold,
                wsi_path, wsi_format, wsi_available, hardness_rank,
                wrong_configurations, total_configurations, wrong_pct,
                consensus_wrong_class, consensus_wrong_count, mean_wrong_confidence,
                max_wrong_confidence, all_prediction_summary, imported_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slide_id) DO UPDATE SET
                original_label=excluded.original_label,
                pathology_type=excluded.pathology_type,
                grade=excluded.grade,
                source=excluded.source,
                cv_fold=excluded.cv_fold,
                wsi_path=excluded.wsi_path,
                wsi_format=excluded.wsi_format,
                wsi_available=excluded.wsi_available,
                hardness_rank=excluded.hardness_rank,
                wrong_configurations=excluded.wrong_configurations,
                total_configurations=excluded.total_configurations,
                wrong_pct=excluded.wrong_pct,
                consensus_wrong_class=excluded.consensus_wrong_class,
                consensus_wrong_count=excluded.consensus_wrong_count,
                mean_wrong_confidence=excluded.mean_wrong_confidence,
                max_wrong_confidence=excluded.max_wrong_confidence,
                all_prediction_summary=excluded.all_prediction_summary,
                imported_at=excluded.imported_at
            """,
            slide_records,
        )
        connection.execute("DELETE FROM predictions")
        connection.executemany(
            """
            INSERT INTO predictions(
                slide_id, model, feature, configuration, predicted_label, confidence,
                second_predicted_label, second_confidence, is_correct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            predictions,
        )

    core_count = sum(int(row["wrong_configurations"]) == 14 for row in challenge_rows)
    if core_count != 193:
        raise ValueError(f"Expected 193 unanimous errors, found {core_count}")
    print(
        f"Imported {len(slide_records)} slides, {len(predictions)} predictions, "
        f"{core_count} core slides; missing WSI: {len(missing_wsi)}",
        flush=True,
    )
    if missing_wsi:
        print("Missing WSI examples:", ", ".join(missing_wsi[:10]), flush=True)


def update_predictions():
    initialize_database()
    challenge_ids = {row["slide_id"] for row in load_challenge_rows()}
    predictions = load_predictions(challenge_ids)
    with transaction() as connection:
        connection.execute("DELETE FROM predictions")
        connection.executemany(
            """
            INSERT INTO predictions(
                slide_id, model, feature, configuration, predicted_label, confidence,
                second_predicted_label, second_confidence, is_correct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            predictions,
        )
    print(f"Updated {len(predictions)} predictions with second candidates", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-all", action="store_true", help="Import challenge data")
    parser.add_argument(
        "--update-predictions",
        action="store_true",
        help="Refresh predictions and second candidates without scanning WSI files",
    )
    args = parser.parse_args()
    if args.import_all:
        import_all()
    elif args.update_predictions:
        update_predictions()
    else:
        parser.error("Use --import-all or --update-predictions")


if __name__ == "__main__":
    main()
