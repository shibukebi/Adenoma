#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


NEW_ROOT = Path("/data15/zhengke_usb2/yuexin_data/result/5fold_11class")
OLD_ROOT = Path("/data15/zhengke_usb2/yuexin_data/result/fold5_hp+yx")
OUT_ROOT = NEW_ROOT / "summary_11class_5fold"
FIG_ROOT = OUT_ROOT / "figures"

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
CLASS_SHORT = ["SSL", "HP", "TSA", "USA", "TA", "TVA", "IP", "SSLD", "TSAD", "TAD", "TVAD"]
PLOT_ORDER = [6, 1, 0, 7, 2, 8, 3, 4, 9, 5, 10]
PLOT_SHORT = [CLASS_SHORT[i] for i in PLOT_ORDER]
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

EXPERIMENTS = []
for model, old_model in [
    ("CLAM-SB", "CLAM-SB"),
    ("TransMIL", "transmil"),
    ("DSMIL", "dsmil"),
]:
    for mag in ["2p5x", "5x", "10x", "20x"]:
        EXPERIMENTS.append((model, old_model, mag))
for mag in ["2p5x_5x", "5x_10x"]:
    EXPERIMENTS.append(("MIST", "MIST", mag))

METRICS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("weighted_f1", "Weighted-F1"),
    ("auc_ovr_macro", "Macro AUC"),
    ("auc_ovr_weighted", "Weighted AUC"),
    ("test_mean_inference_time_sec_per_case", "Inference s/case"),
    ("epochs_completed", "Epochs"),
]


def fold_paths(model: str, old_model: str, mag: str):
    if model == "MIST":
        return [
            (
                fold,
                NEW_ROOT / "MIST" / "11class" / mag / f"fold-{fold}" / "metrics.json",
                NEW_ROOT / "MIST" / "11class" / mag / f"fold-{fold}" / "predictions.csv",
            )
            for fold in range(5)
        ]

    paths = []
    for fold in range(4):
        d = NEW_ROOT / model / "11class" / mag / f"fold-{fold}"
        paths.append((fold, d / "metrics.json", d / "predictions.csv"))
    old = OLD_ROOT / old_model / "11class" / mag
    paths.append((4, old / "metrics.json", old / "predictions.csv"))
    return paths


def read_metric(path: Path):
    with path.open() as f:
        return json.load(f)


def fmt_mean_sd(values, digits=4):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return "NA"
    mean = float(np.nanmean(arr))
    sd = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def nanmean_or_nan(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr))


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                vals.append("NA")
            else:
                vals.append(str(val).replace("\n", " "))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def safe_stem(model: str, mag: str):
    return f"{model}_{mag}".replace("-", "").replace("+", "_").replace("/", "_")


def load_predictions(spec):
    frames = []
    for fold, _, pred_path in spec:
        df = pd.read_csv(pred_path)
        df = df[df["split"].astype(str).str.lower().eq("test")].copy()
        df["cv_fold"] = fold
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_confusion(df: pd.DataFrame, title: str, out_path: Path):
    y_true = df["label"].astype(int).to_numpy()
    y_pred = df["pred"].astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=PLOT_ORDER)

    fig, ax = plt.subplots(figsize=(10.5, 8.6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(PLOT_SHORT)))
    ax.set_yticks(np.arange(len(PLOT_SHORT)))
    ax.set_xticklabels(PLOT_SHORT, rotation=45, ha="right")
    ax.set_yticklabels(PLOT_SHORT)
    threshold = cm.max() * 0.55 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    pd.DataFrame(cm, index=PLOT_SHORT, columns=PLOT_SHORT).to_csv(out_path.with_suffix(".csv"))


def plot_multifold_roc(df: pd.DataFrame, title: str, out_path: Path):
    mean_fpr = np.linspace(0, 1, 201)
    tprs = []
    aucs = []
    fig, ax = plt.subplots(figsize=(8.2, 7.2))

    for fold in sorted(df["cv_fold"].unique()):
        fold_df = df[df["cv_fold"] == fold]
        y_true = fold_df["label"].astype(int).to_numpy()
        y_score = fold_df[PROB_COLS].to_numpy(dtype=float)
        y_bin = label_binarize(y_true, classes=list(range(11)))
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())
        fold_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(fold_auc)
        ax.plot(fpr, tpr, color="#7aa6c2", lw=1.0, alpha=0.38, label=f"Fold {fold + 1} AUC={fold_auc:.3f}")

    tprs = np.asarray(tprs)
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    sd_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
    sd_tpr = tprs.std(axis=0, ddof=1) if len(tprs) > 1 else np.zeros_like(mean_tpr)
    lower = np.maximum(mean_tpr - sd_tpr, 0)
    upper = np.minimum(mean_tpr + sd_tpr, 1)

    ax.plot(mean_fpr, mean_tpr, color="#0b4f71", lw=2.8, label=f"Mean ROC AUC={mean_auc:.3f} ± {sd_auc:.3f}")
    ax.fill_between(mean_fpr, lower, upper, color="#0b4f71", alpha=0.18, label="±1 SD")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    complete = []
    incomplete = []
    fold_rows = []
    summary_rows = []

    for model, old_model, mag in EXPERIMENTS:
        spec = fold_paths(model, old_model, mag)
        missing = [str(p) for _, m, p in spec if not (m.exists() and p.exists())]
        if missing:
            incomplete.append({"model": model, "feature": mag, "reason": "missing metrics/predictions", "missing": "; ".join(missing)})
            continue

        metrics_by_fold = [(fold, read_metric(m)) for fold, m, _ in spec]
        row = {"Model": model, "Feature": mag, "Folds": 5}
        raw_means = {}
        for key, label in METRICS:
            values = [met.get(key, np.nan) for _, met in metrics_by_fold]
            row[label] = fmt_mean_sd(values, digits=4 if key != "epochs_completed" else 1)
            raw_means[key] = nanmean_or_nan(values)
        summary_rows.append({**row, **{f"mean_{k}": v for k, v in raw_means.items()}})

        for fold, met in metrics_by_fold:
            fold_rows.append({
                "Model": model,
                "Feature": mag,
                "Fold": fold + 1,
                **{label: met.get(key, np.nan) for key, label in METRICS},
                "n_test": met.get("n_test", met.get("n_samples", np.nan)),
            })

        df = load_predictions(spec)
        n_pooled = int(len(df))
        unique_slides = int(df["slide_id"].nunique()) if "slide_id" in df.columns else n_pooled
        duplicate_slides = n_pooled - unique_slides
        stem = safe_stem(model, mag)
        cm_path = FIG_ROOT / f"{stem}_pooled_confusion_matrix.png"
        roc_path = FIG_ROOT / f"{stem}_multifold_roc.png"
        plot_confusion(df, f"{model} {mag} pooled test confusion matrix", cm_path)
        plot_multifold_roc(df, f"{model} {mag} five-fold micro-average ROC", roc_path)
        complete.append({
            "Model": model,
            "Feature": mag,
            "stem": stem,
            "cm": cm_path,
            "roc": roc_path,
            "mean_macro_f1": raw_means["macro_f1"],
            "mean_accuracy": raw_means["accuracy"],
            "mean_auc": raw_means["auc_ovr_macro"],
            "n_pooled": n_pooled,
            "unique_slides": unique_slides,
            "duplicate_slides": duplicate_slides,
        })

    summary_df = pd.DataFrame(summary_rows)
    sort_cols = ["mean_macro_f1", "mean_accuracy", "mean_auc_ovr_macro"]
    summary_df = summary_df.sort_values(sort_cols, ascending=False)
    display_cols = ["Model", "Feature", "Folds"] + [label for _, label in METRICS]
    summary_df[display_cols].to_csv(OUT_ROOT / "fivefold_mean_sd_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT_ROOT / "per_fold_metrics.csv", index=False)

    best = max(complete, key=lambda x: (x["mean_macro_f1"], x["mean_accuracy"], x["mean_auc"])) if complete else None

    lines = []
    lines.append("# 11-Class Five-Fold Experiment Summary\n")
    lines.append(f"Generated from `{NEW_ROOT}` and fold-5 results in `{OLD_ROOT}`.\n")
    lines.append("\n")
    lines.append("## Scope\n")
    lines.append("- Included experiments: complete five-fold results with both `metrics.json` and `predictions.csv`.\n")
    lines.append("- Five folds are `fold-0..fold-3` from the补跑目录 plus the previous fold5 result, treated as Fold 5.\n")
    lines.append("- MIST folds are read from the normalized five-fold evaluation outputs under `MIST/11class/<feature>/fold-0..fold-4`.\n")
    lines.append("- Confusion matrices use class order: IP, HP, SSL, SSLD, TSA, TSAD, USA, TA, TAD, TVA, TVAD.\n")
    lines.append("- ROC plots use micro-average one-vs-rest ROC per fold, with mean ROC and ±1 SD band.\n\n")

    lines.append("## Core Table: Five-Fold Mean ± SD\n\n")
    lines.append(markdown_table(summary_df[display_cols]))
    lines.append("\n\n")

    if best:
        rel_cm = best["cm"].relative_to(OUT_ROOT)
        rel_roc = best["roc"].relative_to(OUT_ROOT)
        lines.append("## Core Figures\n\n")
        lines.append(f"Best completed configuration by mean Macro-F1: **{best['Model']} {best['Feature']}**.\n\n")
        lines.append("### 1. Pooled Confusion Matrix\n\n")
        lines.append(f"![Pooled confusion matrix]({rel_cm})\n\n")
        lines.append("### 2. Multi-Fold ROC Curve\n\n")
        lines.append(f"![Multi-fold ROC]({rel_roc})\n\n")

    integrity = pd.DataFrame([
        {
            "Model": item["Model"],
            "Feature": item["Feature"],
            "Pooled test rows": item["n_pooled"],
            "Unique slide IDs": item["unique_slides"],
            "Duplicate slide IDs": item["duplicate_slides"],
        }
        for item in complete
    ]).sort_values(["Model", "Feature"])
    lines.append("## Pooled Test Set Integrity Check\n\n")
    lines.append(markdown_table(integrity))
    lines.append("\n\n")

    lines.append("## Figure Index\n\n")
    fig_index = pd.DataFrame([
        {
            "Model": item["Model"],
            "Feature": item["Feature"],
            "Pooled Confusion Matrix": str(item["cm"].relative_to(OUT_ROOT)),
            "Multi-Fold ROC": str(item["roc"].relative_to(OUT_ROOT)),
        }
        for item in complete
    ]).sort_values(["Model", "Feature"])
    lines.append(markdown_table(fig_index))
    lines.append("\n\n")

    if incomplete:
        lines.append("## Not Yet Included\n\n")
        lines.append(markdown_table(pd.DataFrame(incomplete)))
        lines.append("\n\n")

    (OUT_ROOT / "RESULTS_11CLASS_5FOLD.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUT_ROOT / 'RESULTS_11CLASS_5FOLD.md'}")
    print(f"Wrote {OUT_ROOT / 'fivefold_mean_sd_summary.csv'}")
    print(f"Wrote {OUT_ROOT / 'per_fold_metrics.csv'}")
    print(f"Wrote figures under {FIG_ROOT}")
    if best:
        print(f"Best by macro-F1: {best['Model']} {best['Feature']}")


if __name__ == "__main__":
    main()
