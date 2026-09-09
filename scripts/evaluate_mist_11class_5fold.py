#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


MIST_ROOT = Path("/data15/data15_5/yuexin2/MIST")
sys.path.insert(0, str(MIST_ROOT))
import dsmil as mil  # noqa: E402


RESULT_ROOT = Path("/data15/zhengke_usb2/yuexin_data/result/5fold_11class")
OLD_ROOT = Path("/data15/zhengke_usb2/yuexin_data/result/fold5_hp+yx")
MANIFEST_ROOT = Path("/data15/zhengke_usb2/yuexin_data/adenoma_feature/MIST/manifests_11class_5fold")

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
METRIC_PREFIX = [
    "ssl",
    "hp",
    "tsa",
    "usa",
    "ta",
    "tva",
    "ip",
    "ssl_with_highgrade_dysplasia",
    "tsa_with_highgrade_dysplasia",
    "ta_with_highgrade_dysplasia",
    "tva_with_highgrade_dysplasia",
]
PROB_COLS = [
    "prob_ssl",
    "prob_hp",
    "prob_tsa",
    "prob_usa",
    "prob_ta",
    "prob_tva",
    "prob_ip",
    "prob_ssl_with_highgrade_dysplasia",
    "prob_tsa_with_highgrade_dysplasia",
    "prob_ta_with_highgrade_dysplasia",
    "prob_tva_with_highgrade_dysplasia",
]


def one_hot(label: int, n_classes: int = 11):
    arr = np.zeros(n_classes, dtype=np.float32)
    arr[int(label)] = 1.0
    return arr


def build_model(feats_size: int, num_classes: int, checkpoint: Path, device: torch.device):
    i_classifier = mil.FCLayer(in_size=feats_size, out_size=num_classes).to(device)
    b_classifier = mil.BClassifier(input_size=feats_size, output_class=num_classes, dropout_v=0).to(device)
    model = mil.MILNet(i_classifier, b_classifier).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def infer_manifest(manifest: Path, checkpoint: Path, out_dir: Path, fold: int, combo: str, gpu_index: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")

    model = build_model(5120, 11, checkpoint, device)
    criterion = nn.BCEWithLogitsLoss()
    df = pd.read_csv(manifest)

    labels = []
    preds = []
    scores = []
    losses = []
    rows = []
    start = time.perf_counter()
    with torch.no_grad():
        for idx, row in df.iterrows():
            feat_path = Path(row["path"])
            label = int(row["label"])
            feats = pd.read_csv(feat_path).to_numpy(dtype=np.float32)
            bag_feats = torch.from_numpy(feats).to(device).view(-1, 5120)
            target = torch.from_numpy(one_hot(label)).to(device).view(1, -1)
            ins_prediction, bag_prediction, _, _ = model(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)
            bag_loss = criterion(bag_prediction.view(1, -1), target)
            max_loss = criterion(max_prediction.view(1, -1), target)
            loss = 0.5 * bag_loss + 0.5 * max_loss
            score = (0.5 * torch.sigmoid(max_prediction) + 0.5 * torch.sigmoid(bag_prediction).squeeze(0)).detach().cpu().numpy()
            pred = int(np.argmax(score))

            labels.append(label)
            preds.append(pred)
            scores.append(score)
            losses.append(float(loss.item()))
            slide_id = feat_path.stem
            record = {
                "slide_id": slide_id,
                "fold": fold,
                "label": label,
                "label_name": CLASS_NAMES[label],
                "pred": pred,
                "pred_name": CLASS_NAMES[pred],
                "is_correct": label == pred,
                "confidence": float(score[pred]),
            }
            record.update({col: float(score[i]) for i, col in enumerate(PROB_COLS)})
            record["split"] = "test"
            rows.append(record)
            if (idx + 1) % 100 == 0:
                print(f"{combo} fold-{fold}: evaluated {idx + 1}/{len(df)}", flush=True)

    elapsed = time.perf_counter() - start
    y_true = np.asarray(labels, dtype=int)
    y_pred = np.asarray(preds, dtype=int)
    y_score = np.asarray(scores, dtype=float)
    y_bin = np.eye(11, dtype=int)[y_true]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_samples": int(len(y_true)),
        "n_test": int(len(y_true)),
        "auc_ovr_macro": float(roc_auc_score(y_bin, y_score, average="macro")),
        "auc_ovr_weighted": float(roc_auc_score(y_bin, y_score, average="weighted")),
        "model_type": "MIST",
        "task_name": f"adenoma_11class_{combo}_mist_fold{fold}",
        "task_mode": "adenoma_11class",
        "fold": int(fold),
        "test_total_inference_time_sec": float(elapsed),
        "test_num_cases": int(len(y_true)),
        "test_mean_inference_time_sec_per_case": float(elapsed / max(len(y_true), 1)),
    }
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(11)), zero_division=0
    )
    for i, key in enumerate(METRIC_PREFIX):
        metrics[f"{key}_precision"] = float(precisions[i])
        metrics[f"{key}_recall"] = float(recalls[i])
        metrics[f"{key}_f1"] = float(f1s[i])
        metrics[f"{key}_support"] = int(supports[i])
        metrics[f"{key}_auc"] = float(roc_auc_score(y_bin[:, i], y_score[:, i]))

    train_history = out_dir / "training_history.csv"
    if train_history.exists():
        hist = pd.read_csv(train_history)
        metrics["epochs_completed"] = int(hist["epoch"].max()) if "epoch" in hist else int(len(hist))

    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=list(range(11))), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "confusion_matrix.csv"
    )
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"{combo} fold-{fold}: acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} auc={metrics['auc_ovr_macro']:.4f}", flush=True)


def infer_old_epoch_count(src_log: Path, dst_dir: Path):
    if not src_log.exists() or (dst_dir / "training_history.csv").exists():
        return
    data = src_log.read_bytes()[-5_000_000:].decode("utf-8", errors="ignore")
    matches = re.findall(r"Epoch \[(\d+)/(\d+)\]", data)
    if matches:
        epoch = int(matches[-1][0])
        pd.DataFrame([{"epoch": epoch}]).to_csv(dst_dir / "training_history.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--combo", choices=["2p5x_5x", "5x_10x"], default=None)
    parser.add_argument("--fold", type=int, choices=[0, 1, 2, 3, 4], default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    jobs = []
    for combo in ["2p5x_5x", "5x_10x"]:
        for fold in range(4):
            jobs.append((
                combo,
                fold,
                MANIFEST_ROOT / combo / f"fold-{fold}" / "test" / f"mist_11class_{combo}_fold{fold}_test.csv",
                RESULT_ROOT / "MIST" / "11class" / combo / f"fold-{fold}" / "1.pth",
                RESULT_ROOT / "MIST" / "11class" / combo / f"fold-{fold}",
            ))

    old_jobs = [
        (
            "2p5x_5x",
            4,
            MANIFEST_ROOT / "2p5x_5x" / "fold-4" / "test" / "mist_11class_2p5x_5x_fold4_test.csv",
            OLD_ROOT / "MIST" / "1.pth",
            RESULT_ROOT / "MIST" / "11class" / "2p5x_5x" / "fold-4",
            OLD_ROOT / "MIST" / "train.log",
        ),
        (
            "5x_10x",
            4,
            MANIFEST_ROOT / "5x_10x" / "fold-4" / "test" / "mist_11class_5x_10x_fold4_test.csv",
            OLD_ROOT / "MIST" / "5x_10x" / "1.pth",
            RESULT_ROOT / "MIST" / "11class" / "5x_10x" / "fold-4",
            OLD_ROOT / "MIST" / "5x_10x" / "train.log",
        ),
    ]
    for combo, fold, manifest, checkpoint, out_dir, src_log in old_jobs:
        out_dir.mkdir(parents=True, exist_ok=True)
        infer_old_epoch_count(src_log, out_dir)
        jobs.append((combo, fold, manifest, checkpoint, out_dir))

    for combo, fold, manifest, checkpoint, out_dir in jobs:
        if args.combo is not None and combo != args.combo:
            continue
        if args.fold is not None and fold != args.fold:
            continue
        if args.skip_existing and (out_dir / "metrics.json").exists() and (out_dir / "predictions.csv").exists():
            print(f"{combo} fold-{fold}: skip existing", flush=True)
            continue
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        infer_manifest(manifest, checkpoint, out_dir, fold, combo, args.gpu_index)


if __name__ == "__main__":
    main()
