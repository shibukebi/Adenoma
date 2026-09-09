#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


TRAIN_RE = re.compile(r"Epoch:?\s+(\d+),\s+train_loss:?\s+([0-9.eE+-]+).*train_error:?\s+([0-9.eE+-]+)")
VAL_RE = re.compile(r"Val Set,\s+val_loss:\s+([0-9.eE+-]+),\s+val_error:\s+([0-9.eE+-]+)")
MIST_RE = re.compile(
    r"Epoch\s+\[(\d+)/(\d+)\]\s+train loss:\s+([0-9.eE+-]+)\s+test loss:\s+([0-9.eE+-]+),\s+"
    r"accuracy:\s+([0-9.eE+-]+)"
)


def split_log_text(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines()


def parse_epoch_style_log(log_path: Path, model: str, mag: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current_index: int | None = None
    if not log_path.exists():
        return pd.DataFrame()

    for line in split_log_text(log_path):
        train_match = TRAIN_RE.search(line)
        if train_match:
            rows.append(
                {
                    "model": model,
                    "magnification": mag,
                    "epoch": int(train_match.group(1)),
                    "train_loss": float(train_match.group(2)),
                    "train_error": float(train_match.group(3)),
                    "val_loss": math.nan,
                    "val_error": math.nan,
                    "source_log": str(log_path),
                }
            )
            current_index = len(rows) - 1
            continue

        val_match = VAL_RE.search(line)
        if val_match and current_index is not None and pd.isna(rows[current_index]["val_loss"]):
            rows[current_index]["val_loss"] = float(val_match.group(1))
            rows[current_index]["val_error"] = float(val_match.group(2))

    return pd.DataFrame(rows)


def parse_mist_log(log_path: Path, mag: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not log_path.exists():
        return pd.DataFrame()

    for line in split_log_text(log_path):
        match = MIST_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "model": "MIST",
                "magnification": mag,
                "epoch": int(match.group(1)),
                "train_loss": float(match.group(3)),
                "train_error": math.nan,
                "val_loss": float(match.group(4)),
                "val_error": 1.0 - float(match.group(5)),
                "source_log": str(log_path),
            }
        )
    return pd.DataFrame(rows)


def best_epoch_log(result_dir: Path, model: str, mag: str) -> pd.DataFrame:
    best = pd.DataFrame()
    for log_path in sorted(result_dir.glob("train*.log")):
        parsed = parse_epoch_style_log(log_path, model, mag)
        if len(parsed) > len(best):
            best = parsed
    return best


def collect_loss_rows(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    mist_specs = [
        ("2.5x+5x", root / "MIST" / "train.log"),
        ("5x+10x", root / "MIST" / "5x_10x" / "train.log"),
    ]
    for mag, log_path in mist_specs:
        parsed = parse_mist_log(log_path, mag)
        if not parsed.empty:
            frames.append(parsed)

    for model, rel in [("CLAM-SB", "CLAM-SB/11class"), ("TransMIL", "TransMIL/11class")]:
        for mag in ["2p5x", "5x", "10x", "20x"]:
            parsed = best_epoch_log(root / rel / mag, model, mag)
            if not parsed.empty:
                frames.append(parsed)

    dsmil_root = root / "DSMIL" / "11class"
    if dsmil_root.exists():
        for mag in ["2p5x", "5x", "10x", "20x"]:
            parsed = parse_epoch_style_log(dsmil_root / mag / "train.log", "DSMIL", mag)
            if not parsed.empty:
                frames.append(parsed)

    if not frames:
        raise RuntimeError(f"No loss records found under {root}")

    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["model", "magnification", "epoch"]).reset_index(drop=True)


def plot_one(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=180)
    ax.plot(df["epoch"], df["train_loss"], marker="o", markersize=3, linewidth=1.6, label="train loss")
    val_df = df.dropna(subset=["val_loss"])
    if not val_df.empty:
        ax.plot(val_df["epoch"], val_df["val_loss"], marker="s", markersize=3, linewidth=1.6, label="val loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_model_grid(all_df: pd.DataFrame, model: str, out_path: Path) -> None:
    model_df = all_df[all_df["model"] == model]
    if model_df.empty:
        return
    groups = list(model_df.groupby("magnification", sort=False))
    ncols = 2
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4, 4 * nrows)), dpi=180)
    axes_list = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, (mag, df) in zip(axes_list, groups):
        df = df.sort_values("epoch")
        ax.plot(df["epoch"], df["train_loss"], marker="o", markersize=2.5, linewidth=1.4, label="train")
        val_df = df.dropna(subset=["val_loss"])
        if not val_df.empty:
            ax.plot(val_df["epoch"], val_df["val_loss"], marker="s", markersize=2.5, linewidth=1.4, label="val")
        ax.set_title(f"{model} {mag}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes_list[len(groups) :]:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss curves for MIST, CLAM-SB, TransMIL and DSMIL runs.")
    parser.add_argument("--result-root", default="/data15/zhengke_usb2/yuexin_data/result/fold5_hp+yx")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir) if args.output_dir else result_root / "summary_11class" / "loss_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_df = collect_loss_rows(result_root)
    all_df.to_csv(output_dir / "all_baseline_loss_curves.csv", index=False)

    for (model, mag), df in all_df.groupby(["model", "magnification"], sort=False):
        safe_model = model.replace("-", "_").lower()
        safe_mag = mag.replace("+", "plus").replace(".", "p")
        plot_one(
            df.sort_values("epoch"),
            f"{model} {mag} loss curve",
            output_dir / f"{safe_model}_{safe_mag}_loss_curve.png",
        )

    for model in all_df["model"].drop_duplicates().tolist():
        safe_model = model.replace("-", "_").lower()
        plot_model_grid(all_df, model, output_dir / f"{safe_model}_loss_curves_combined.png")

    summary = all_df.groupby(["model", "magnification"])["epoch"].agg(["min", "max", "count"]).reset_index()
    summary.to_csv(output_dir / "all_baseline_loss_curve_summary.csv", index=False)
    print(f"Wrote {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
