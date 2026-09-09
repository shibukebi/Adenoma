#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

if not hasattr(np, "Inf"):
    np.Inf = np.inf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAM_ROOT = PROJECT_ROOT.parent / "CLAM"
sys.path.insert(0, str(CLAM_ROOT))

from dataset_modules.dataset_generic import Generic_MIL_Dataset, Generic_Split  # noqa: E402
from utils.core_utils import train  # noqa: E402
from utils.file_utils import save_pkl  # noqa: E402


LABEL_DICT = {"others": 0, "SSL": 1}
INV_LABEL_DICT = {value: key for key, value in LABEL_DICT.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Route C CLAM MIL 20X SSL vs others baseline.")
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--ready-csv", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--reg", type=float, default=1e-5)
    parser.add_argument("--drop-out", type=float, default=0.25)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--weighted-sample", action="store_true", default=False)
    parser.add_argument("--early-stopping", action="store_true", default=False)
    parser.add_argument("--allow-small-splits", action="store_true", default=False)
    parser.add_argument("--smoke", action="store_true", default=False)
    return parser.parse_args()


def read_id_list(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row[0].strip() for row in csv.reader(handle) if row and row[0].strip()]


def build_split(df: pd.DataFrame, ids: list[str], data_dir: str) -> Generic_Split:
    subset = df[df["slide_id"].isin(ids)].reset_index(drop=True)
    return Generic_Split(subset, data_dir=data_dir, num_classes=2)


def compute_metrics(predictions_df: pd.DataFrame, split_sizes: dict[str, int], label_counts: dict[str, dict[str, int]], mode_counts: dict[str, dict[str, int]]) -> tuple[dict, pd.DataFrame]:
    y_true = predictions_df["label"].to_numpy(dtype=int)
    y_pred = predictions_df["pred"].to_numpy(dtype=int)
    y_score = predictions_df["prob_ssl"].to_numpy(dtype=float)
    auc = roc_auc_score(y_true, y_score)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    metrics = {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "recall": float(recall),
        "specificity": float(specificity),
        "n_train": split_sizes["train"],
        "n_val": split_sizes["val"],
        "n_test": split_sizes["test"],
        "ssl_train": label_counts["train"].get("SSL", 0),
        "ssl_val": label_counts["val"].get("SSL", 0),
        "ssl_test": label_counts["test"].get("SSL", 0),
        "others_train": label_counts["train"].get("others", 0),
        "others_val": label_counts["val"].get("others", 0),
        "others_test": label_counts["test"].get("others", 0),
        "n_success_patho_r1": sum(mode_counts[split_name].get("patho-r1", 0) for split_name in mode_counts),
        "n_success_fallback": sum(mode_counts[split_name].get("heuristic", 0) for split_name in mode_counts),
    }
    cm_df = pd.DataFrame(cm, index=["true_others", "true_ssl"], columns=["pred_others", "pred_ssl"])
    return metrics, cm_df


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.split_dir) / f"flod-{args.fold}_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if not stats.get("formal_ready", False) and not args.allow_small_splits:
            raise RuntimeError(
                f"Split is not marked formal_ready in {stats_path}. "
                "Pass --allow-small-splits for smoke/debug runs."
            )

    ready_df = pd.read_csv(args.ready_csv, dtype={"slide_id": str, "case_id": str, "label": str})
    dataset = Generic_MIL_Dataset(
        csv_path=args.ready_csv,
        data_dir=args.feature_dir,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict=LABEL_DICT,
        ignore=[],
        patient_strat=False,
        label_col="label",
    )

    train_ids = read_id_list(Path(args.split_dir) / f"flod-{args.fold}-train.csv")
    val_ids = read_id_list(Path(args.split_dir) / f"flod-{args.fold}-val.csv")
    test_ids = read_id_list(Path(args.split_dir) / f"flod-{args.fold}-test.csv")

    split_sizes = {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}
    label_counts = {}
    mode_counts = {}
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        subset = ready_df[ready_df["slide_id"].isin(ids)]
        label_counts[split_name] = subset["label_name"].value_counts().to_dict()
        mode_counts[split_name] = subset["selection_mode"].value_counts().to_dict()

    train_split = build_split(dataset.slide_data, train_ids, args.feature_dir)
    val_split = build_split(dataset.slide_data, val_ids, args.feature_dir)
    test_split = build_split(dataset.slide_data, test_ids, args.feature_dir)

    if min(split_sizes.values()) == 0:
        raise RuntimeError(
            "Route C cleaned splits contain an empty split. "
            f"split_sizes={split_sizes}. "
            "Run Route C on more slides before attempting training."
        )

    train_args = SimpleNamespace(
        results_dir=str(results_dir),
        log_data=False,
        testing=False,
        early_stopping=args.early_stopping,
        bag_loss="ce",
        n_classes=2,
        drop_out=args.drop_out,
        model_size=None,
        model_type="mil",
        subtyping=False,
        B=8,
        inst_loss=None,
        no_inst_cluster=True,
        bag_weight=0.7,
        max_epochs=2 if args.smoke else args.max_epochs,
        lr=args.lr,
        reg=args.reg,
        weighted_sample=args.weighted_sample,
        opt="adam",
        seed=args.seed,
        embed_dim=args.embed_dim,
        exp_code=f"clam_ssl_vs_others_route_c_fold{args.fold}",
        label_frac=1.0,
    )

    config_snapshot = {
        "dataset_csv": args.dataset_csv,
        "ready_csv": args.ready_csv,
        "split_dir": args.split_dir,
        "feature_dir": args.feature_dir,
        "results_dir": str(results_dir),
        "fold": args.fold,
        "max_epochs": train_args.max_epochs,
        "seed": args.seed,
        "lr": args.lr,
        "reg": args.reg,
        "drop_out": args.drop_out,
        "embed_dim": args.embed_dim,
        "weighted_sample": args.weighted_sample,
        "early_stopping": args.early_stopping,
        "allow_small_splits": args.allow_small_splits,
        "smoke": args.smoke,
        "label_dict": LABEL_DICT,
        "split_sizes": split_sizes,
        "label_counts": label_counts,
        "selection_mode_counts": mode_counts,
    }
    (results_dir / "experiment_config.json").write_text(json.dumps(config_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    results_dict, test_auc, val_auc, test_acc, val_acc = train((train_split, val_split, test_split), args.fold, train_args)
    save_pkl(results_dir / f"split_{args.fold}_results.pkl", results_dict)
    pd.DataFrame([{"fold": args.fold, "test_auc": test_auc, "val_auc": val_auc, "test_acc": test_acc, "val_acc": val_acc}]).to_csv(results_dir / "summary.csv", index=False)

    mode_map = ready_df.set_index("slide_id")["selection_mode"].to_dict()
    predictions = []
    for slide_id, payload in sorted(results_dict.items()):
        probs = np.asarray(payload["prob"]).reshape(-1)
        label = int(payload["label"])
        pred = int(np.argmax(probs))
        predictions.append({
            "slide_id": str(slide_id),
            "label": label,
            "label_name": INV_LABEL_DICT[label],
            "pred": pred,
            "pred_name": INV_LABEL_DICT[pred],
            "prob_others": float(probs[0]),
            "prob_ssl": float(probs[1]),
            "selection_mode": mode_map.get(str(slide_id), ""),
        })
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(results_dir / "predictions.csv", index=False)

    metrics, cm_df = compute_metrics(predictions_df, split_sizes, label_counts, mode_counts)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    cm_df.to_csv(results_dir / "confusion_matrix.csv")

    print(json.dumps({"results_dir": str(results_dir), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
