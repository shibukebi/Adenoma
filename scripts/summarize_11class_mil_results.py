#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CLASS_NAMES = [
    "ssl",
    "hp",
    "TSA",
    "USA",
    "TA",
    "TVA",
    "IP",
    "ssl with highgrade dysplasia",
    "TSA with highgrade dysplasia",
    "TA with highgrade dysplasia",
    "TVA with highgrade dysplasia",
]
LABEL_IDS = list(range(len(CLASS_NAMES)))
CM_LABEL_IDS = [6, 1, 0, 7, 2, 8, 3, 4, 9, 5, 10]
CM_DISPLAY_NAMES = ["IP", "HP", "SSL", "SSLD", "TSA", "TSAD", "USA", "TA", "TAD", "TVA", "TVAD"]
SAFE_NAMES = [
    "ssl",
    "hp",
    "tsa",
    "usa",
    "ta",
    "tva",
    "ip",
    "ssl_hgd",
    "tsa_hgd",
    "ta_hgd",
    "tva_hgd",
]
PROB_ALIASES = [
    ["prob_ssl"],
    ["prob_hp"],
    ["prob_tsa", "prob_TSA"],
    ["prob_usa", "prob_USA"],
    ["prob_ta", "prob_TA"],
    ["prob_tva", "prob_TVA"],
    ["prob_ip", "prob_IP"],
    ["prob_ssl_hgd", "prob_ssl_with_highgrade_dysplasia"],
    ["prob_tsa_hgd", "prob_TSA_with_highgrade_dysplasia", "prob_tsa_with_highgrade_dysplasia"],
    ["prob_ta_hgd", "prob_TA_with_highgrade_dysplasia", "prob_ta_with_highgrade_dysplasia"],
    ["prob_tva_hgd", "prob_TVA_with_highgrade_dysplasia", "prob_tva_with_highgrade_dysplasia"],
]


def safe_float(value):
    if value is None:
        return None
    value = float(value)
    if np.isnan(value):
        return None
    return value


def compute_metrics(pred_df: pd.DataFrame) -> dict[str, object]:
    y_true = pred_df["label"].to_numpy(dtype=int)
    y_pred = pred_df["pred"].to_numpy(dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_IDS, zero_division=0
    )
    metrics = {
        "n": int(len(pred_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    prob_cols = []
    for aliases in PROB_ALIASES:
        prob_cols.append(next((col for col in aliases if col in pred_df.columns), None))
    if all(col is not None for col in prob_cols):
        class_aucs = []
        for class_idx, col in enumerate(prob_cols):
            binary_true = (y_true == class_idx).astype(int)
            if len(np.unique(binary_true)) < 2:
                class_aucs.append(np.nan)
            else:
                class_aucs.append(float(roc_auc_score(binary_true, pred_df[col].to_numpy(dtype=float))))
        metrics["auc_ovr_macro"] = safe_float(np.nanmean(class_aucs))
        weights = support.astype(float)
        valid = ~np.isnan(np.asarray(class_aucs, dtype=float))
        metrics["auc_ovr_weighted"] = (
            float(np.average(np.asarray(class_aucs, dtype=float)[valid], weights=weights[valid])) if valid.any() else None
        )

    for i, safe in enumerate(SAFE_NAMES):
        metrics[f"{safe}_precision"] = float(precision[i])
        metrics[f"{safe}_recall"] = float(recall[i])
        metrics[f"{safe}_f1"] = float(f1[i])
        metrics[f"{safe}_support"] = int(support[i])
        if all(col is not None for col in prob_cols):
            metrics[f"{safe}_auc"] = safe_float(class_aucs[i])
    return metrics


def plot_confusion(pred_df: pd.DataFrame, out_path: Path, title: str, normalize: bool = False) -> pd.DataFrame:
    y_true = pred_df["label"].to_numpy(dtype=int)
    y_pred = pred_df["pred"].to_numpy(dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=CM_LABEL_IDS)
    display_cm = cm.astype(float)
    fmt = "d"
    if normalize:
        denom = display_cm.sum(axis=1, keepdims=True)
        display_cm = np.divide(display_cm, denom, out=np.zeros_like(display_cm), where=denom != 0)
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=(13.5, 11), dpi=180)
    im = ax.imshow(display_cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(CM_DISPLAY_NAMES)),
        yticks=np.arange(len(CM_DISPLAY_NAMES)),
        xticklabels=CM_DISPLAY_NAMES,
        yticklabels=CM_DISPLAY_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = display_cm.max() / 2 if display_cm.size else 0
    for i in range(display_cm.shape[0]):
        for j in range(display_cm.shape[1]):
            value = display_cm[i, j]
            text = format(int(value), fmt) if not normalize else format(value, fmt)
            ax.text(j, i, text, ha="center", va="center", color="white" if value > threshold else "black", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in CM_DISPLAY_NAMES],
        columns=[f"pred_{name}" for name in CM_DISPLAY_NAMES],
    )
    return cm_df


def normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "prediction" in out.columns and "pred" not in out.columns:
        out = out.rename(columns={"prediction": "pred"})
    if "slide_name" in out.columns and "slide_id" not in out.columns:
        out["slide_id"] = out["slide_name"].astype(str).map(lambda value: Path(value).stem)
    out["label"] = out["label"].astype(int)
    out["pred"] = out["pred"].astype(int)
    return out


def run_mist_inference(
    checkpoint: Path,
    manifest: Path,
    output_csv: Path,
    gpu: int,
    feats_size: int = 5120,
    batch_name: str = "",
) -> pd.DataFrame:
    if output_csv.exists():
        return normalize_prediction_columns(pd.read_csv(output_csv))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    mist_root = Path("/data15/data15_5/yuexin2/MIST")
    sys.path.insert(0, str(mist_root))
    import dsmil as mil  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i_classifier = mil.FCLayer(in_size=feats_size, out_size=len(CLASS_NAMES)).to(device)
    b_classifier = mil.BClassifier(input_size=feats_size, output_class=len(CLASS_NAMES), dropout_v=0).to(device)
    milnet = mil.MILNet(i_classifier, b_classifier).to(device)
    milnet.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    milnet.eval()

    df = pd.read_csv(manifest)
    rows = []
    with torch.inference_mode():
        for idx, row in df.iterrows():
            feat_path = str(row["path"])
            label = int(row["label"])
            feats = pd.read_csv(feat_path).to_numpy(dtype=np.float32)
            bag_feats = torch.from_numpy(feats).to(device).view(-1, feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)
            score = (0.5 * torch.sigmoid(max_prediction) + 0.5 * torch.sigmoid(bag_prediction)).squeeze()
            prob = score.detach().cpu().numpy().astype(float)
            pred = int(np.argmax(prob))
            record = {
                "slide_id": Path(feat_path).stem,
                "path": feat_path,
                "label": label,
                "pred": pred,
                "label_name": CLASS_NAMES[label],
                "pred_name": CLASS_NAMES[pred],
                "split": batch_name,
            }
            for class_idx, safe in enumerate(SAFE_NAMES):
                record[f"prob_{safe}"] = float(prob[class_idx])
            rows.append(record)
            if (idx + 1) % 100 == 0:
                print(f"MIST {batch_name}: {idx + 1}/{len(df)}", flush=True)
    pred_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_csv, index=False)
    return pred_df


def collect_run(model: str, magnification: str, root: Path, out_dir: Path) -> list[dict[str, object]]:
    rows = []
    split_candidates = [
        ("val", ["val_predictions.csv"]),
        ("test", ["predictions.csv", "test_predictions.csv"]),
    ]
    for split_name, pred_names in split_candidates:
        pred_path = next((root / pred_name for pred_name in pred_names if (root / pred_name).exists()), None)
        if pred_path is None:
            continue
        pred_df = normalize_prediction_columns(pd.read_csv(pred_path))
        metrics = compute_metrics(pred_df)
        cm_dir = out_dir / "confusion_matrices"
        stem = f"{model}_{magnification}_{split_name}".replace("/", "_").replace("+", "plus")
        cm_csv = cm_dir / f"{stem}_cm.csv"
        cm_png = cm_dir / f"{stem}_cm.png"
        cm_norm_png = cm_dir / f"{stem}_cm_normalized.png"
        cm_df = plot_confusion(pred_df, cm_png, f"{model} {magnification} {split_name} confusion matrix")
        plot_confusion(pred_df, cm_norm_png, f"{model} {magnification} {split_name} normalized confusion matrix", normalize=True)
        cm_df.to_csv(cm_csv)
        row = {
            "model": model,
            "magnification": magnification,
            "split": split_name,
            "result_dir": str(root),
            "predictions_csv": str(pred_path),
            "confusion_matrix_csv": str(cm_csv),
            "confusion_matrix_png": str(cm_png),
            "confusion_matrix_normalized_png": str(cm_norm_png),
            **metrics,
        }
        rows.append(row)
    return rows


def write_markdown(summary_df: pd.DataFrame, out_path: Path) -> None:
    metric_cols = [
        "model",
        "magnification",
        "split",
        "n",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "auc_ovr_macro",
        "auc_ovr_weighted",
    ]
    lines = ["# 11-class MIL Baseline Summary", ""]
    display = summary_df[metric_cols].copy()
    for col in ["accuracy", "macro_f1", "weighted_f1", "auc_ovr_macro", "auc_ovr_weighted"]:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    lines.append("| " + " | ".join(display.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(display.columns)) + " |")
    for record in display.astype(str).to_dict(orient="records"):
        lines.append("| " + " | ".join(record[col] for col in display.columns) + " |")
    lines.append("")
    lines.append("All listed runs use the same fold5 split (flod-4) and the same 11-class label order.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data15/zhengke_usb2/yuexin_data/result/fold5_hp+yx/summary_11class")
    parser.add_argument("--mist-gpu", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    result_root = Path("/data15/zhengke_usb2/yuexin_data/result/fold5_hp+yx")

    # MIST completed run: 2.5x+5x low4cat.
    mist_root = result_root / "MIST"
    mist_eval = mist_root / "eval_11class"
    mist_manifests = Path("/data15/zhengke_usb2/yuexin_data/adenoma_feature/MIST/manifests_11class_fold5")
    mist_paths = {
        "val": mist_manifests / "mist_hp_yx_11class_fold5_val/mist_hp_yx_11class_fold5_val.csv",
        "test": mist_manifests / "mist_hp_yx_11class_fold5_test/mist_hp_yx_11class_fold5_test.csv",
    }
    if (mist_root / "1.pth").exists():
        for split_name, manifest in mist_paths.items():
            pred_csv = mist_eval / f"{split_name}_predictions.csv"
            run_mist_inference(
                checkpoint=mist_root / "1.pth",
                manifest=manifest,
                output_csv=pred_csv,
                gpu=args.mist_gpu,
                batch_name=split_name,
            )
        rows.extend(collect_run("MIST", "2.5x+5x", mist_eval, out_dir))

    # CLAM-SB and TransMIL completed 11-class runs.
    for mag in ["2p5x", "5x", "10x", "20x"]:
        rows.extend(collect_run("CLAM-SB", mag, result_root / "CLAM-SB/11class" / mag, out_dir))
        rows.extend(collect_run("TransMIL", mag, result_root / "TransMIL/11class" / mag, out_dir))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "summary_metrics_val_test.csv", index=False)
    write_markdown(summary_df, out_dir / "summary_metrics_val_test.md")
    with (out_dir / "summary_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "output_dir": str(out_dir),
                "class_names": CLASS_NAMES,
                "summary_csv": str(out_dir / "summary_metrics_val_test.csv"),
                "summary_md": str(out_dir / "summary_metrics_val_test.md"),
                "confusion_matrix_dir": str(out_dir / "confusion_matrices"),
                "note": "MIST includes completed 2.5x+5x run; MIST 5x+10x was still training when this summary was generated.",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(summary_df[["model", "magnification", "split", "n", "accuracy", "macro_f1", "auc_ovr_macro"]].to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
