#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


TRAIN_RE = re.compile(r"Epoch\s+(\d+),\s+train_loss\s+([0-9.eE+-]+),\s+train_error\s+([0-9.eE+-]+)")
VAL_RE = re.compile(r"val loss\s+([0-9.eE+-]+),\s+val error\s+([0-9.eE+-]+)")


def parse_log(log_path: Path, mag: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current_index: int | None = None
    if not log_path.exists():
        return pd.DataFrame()

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        train_match = TRAIN_RE.search(line)
        if train_match:
            rows.append(
                {
                    "magnification": mag,
                    "epoch": int(train_match.group(1)),
                    "train_loss": float(train_match.group(2)),
                    "train_error": float(train_match.group(3)),
                    "val_loss": None,
                    "val_error": None,
                }
            )
            current_index = len(rows) - 1
            continue

        val_match = VAL_RE.search(line)
        if val_match and current_index is not None and rows[current_index]["val_loss"] is None:
            rows[current_index]["val_loss"] = float(val_match.group(1))
            rows[current_index]["val_error"] = float(val_match.group(2))

    return pd.DataFrame(rows)


def plot_single(df: pd.DataFrame, mag: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=180)
    ax.plot(df["epoch"], df["train_loss"], marker="o", linewidth=1.8, label="train loss")
    val_df = df.dropna(subset=["val_loss"])
    if not val_df.empty:
        ax.plot(val_df["epoch"], val_df["val_loss"], marker="s", linewidth=1.8, label="val loss")
    ax.set_title(f"DSMIL {mag} loss curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_combined(all_df: pd.DataFrame, mags: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180, sharex=False, sharey=False)
    for ax, mag in zip(axes.ravel(), mags):
        df = all_df[all_df["magnification"] == mag].sort_values("epoch")
        if df.empty:
            ax.set_title(f"DSMIL {mag} loss curve (no data)")
            ax.axis("off")
            continue
        ax.plot(df["epoch"], df["train_loss"], marker="o", linewidth=1.5, label="train")
        val_df = df.dropna(subset=["val_loss"])
        if not val_df.empty:
            ax.plot(val_df["epoch"], val_df["val_loss"], marker="s", linewidth=1.5, label="val")
        ax.set_title(f"DSMIL {mag}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DSMIL loss curves from train.log files.")
    parser.add_argument(
        "--result-root",
        default="/data15/zhengke_usb2/yuexin_data/result/fold5_hp+yx/DSMIL/11class",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir) if args.output_dir else result_root / "loss_curves"
    mags = ["2p5x", "5x", "10x", "20x"]

    frames = []
    for mag in mags:
        df = parse_log(result_root / mag / "train.log", mag)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No DSMIL loss records found under {result_root}")

    all_df = pd.concat(frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(output_dir / "dsmil_loss_curves.csv", index=False)

    plot_combined(all_df, mags, output_dir / "dsmil_loss_curves_combined.png")
    for mag in mags:
        df = all_df[all_df["magnification"] == mag].sort_values("epoch")
        if not df.empty:
            plot_single(df, mag, output_dir / f"dsmil_{mag}_loss_curve.png")

    print(f"Wrote {output_dir}")
    print(all_df.groupby("magnification")["epoch"].max().to_string())


if __name__ == "__main__":
    main()
