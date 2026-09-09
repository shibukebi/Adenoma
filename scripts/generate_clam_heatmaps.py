#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from clam_experiment_utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAM_ROOT = PROJECT_ROOT.parent / "CLAM"
DEFAULT_CLAM_PYTHON = PROJECT_ROOT.parent / "anaconda3" / "envs" / "clam_latest" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CLAM heatmaps for selected slides.")
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--ready-csv", required=True)
    parser.add_argument("--slide-dir", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-template", default=str(PROJECT_ROOT / "config" / "clam_ssl_2p5x_heatmap_template.yaml"))
    parser.add_argument("--clam-python", default=str(DEFAULT_CLAM_PYTHON))
    parser.add_argument("--encoder-weights-path", default="")
    parser.add_argument("--exp-code", default="clam_ssl_2p5x_clam_sb_heatmaps")
    parser.add_argument("--model-type", default="clam_sb")
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--drop-out", type=float, default=0.25)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patch-level", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--encoder-model-name", default="uni_v1")
    parser.add_argument("--target-img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--vis-level", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--slide-ext", default=".svs")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_template(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_config(args: argparse.Namespace, process_list_path: Path, output_dir: Path) -> dict:
    template = load_template(Path(args.config_template))
    config = {}

    for section_name, section_payload in template.items():
        if isinstance(section_payload, dict):
            config.setdefault(section_name, {})
            config[section_name].update(section_payload)
        else:
            config[section_name] = section_payload

    # Command-line arguments should win over the template so that per-magnification
    # wrappers can safely override fields like patch_size / overlap / patch_level.
    config.setdefault("exp_arguments", {})
    config["exp_arguments"].update(
        {
            "n_classes": 2,
            "save_exp_code": args.exp_code,
            "raw_save_dir": str(output_dir / "raw"),
            "production_save_dir": str(output_dir / "production"),
            "batch_size": args.batch_size,
        }
    )

    config.setdefault("data_arguments", {})
    config["data_arguments"].update(
        {
            "data_dir": str(args.slide_dir),
            "data_dir_key": "source",
            "process_list": str(process_list_path),
            "preset": str(CLAM_ROOT / "presets" / "bwh_biopsy.csv"),
            "slide_ext": args.slide_ext,
            "label_dict": {"others": 0, "SSL": 1},
        }
    )

    config.setdefault("patching_arguments", {})
    config["patching_arguments"].update(
        {
            "patch_size": args.patch_size,
            "overlap": args.overlap,
            "patch_level": args.patch_level,
            "custom_downsample": 1,
        }
    )

    config.setdefault("encoder_arguments", {})
    config["encoder_arguments"].update(
        {
            "model_name": args.encoder_model_name,
            "target_img_size": args.target_img_size,
        }
    )

    config.setdefault("model_arguments", {})
    config["model_arguments"].update(
        {
            "ckpt_path": str(args.checkpoint_path),
            "model_type": args.model_type,
            "initiate_fn": "initiate_model",
            "model_size": args.model_size,
            "drop_out": args.drop_out,
            "embed_dim": args.embed_dim,
        }
    )

    config.setdefault("heatmap_arguments", {})
    config["heatmap_arguments"].update(
        {
            "vis_level": args.vis_level,
            "alpha": 0.4,
            "blank_canvas": False,
            "save_orig": True,
            "save_ext": "jpg",
            "use_ref_scores": True,
            "blur": False,
            "use_center_shift": True,
            "use_roi": False,
            "calc_heatmap": True,
            "binarize": False,
            "binary_thresh": -1,
            "custom_downsample": 1,
            "cmap": "jet",
        }
    )

    config["sample_arguments"] = {
        "samples": [
            {
                "name": "topk_high_attention",
                "sample": True,
                "seed": 1,
                "k": args.top_k,
                "mode": "topk",
            }
        ]
    }
    return config


def write_slide_summary(row: pd.Series, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"slide_id: {row.slide_id}",
            f"selection_category: {row.selection_category}",
            f"label: {row.label_name}",
            f"prediction: {row.pred_name}",
            f"prob_ssl: {row.prob_ssl:.6f}",
            f"prob_others: {row.prob_others:.6f}",
            f"confidence: {row.confidence:.6f}",
            f"type: {row.get('type', '')}",
            f"grade: {row.get('grade', '')}",
        ]
    )
    summary_path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.slide_dir = Path(args.slide_dir)
    args.checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_df = pd.read_csv(args.selection_csv)
    ready_df = pd.read_csv(args.ready_csv, dtype={"slide_id": str})
    merged = selection_df.merge(
        ready_df[["slide_id", "type", "grade"]],
        on="slide_id",
        how="left",
    )

    if args.limit is not None:
        merged = merged.head(args.limit).copy()

    deduped = (
        merged.sort_values(["selection_category", "rank_within_category", "slide_id"])
        .groupby("slide_id", as_index=False)
        .agg(
            {
                "selection_category": lambda values: "|".join(dict.fromkeys(values)),
                "rank_within_category": "min",
                "fold": "first",
                "split": "first",
                "label": "first",
                "label_name": "first",
                "pred": "first",
                "pred_name": "first",
                "prob_others": "first",
                "prob_ssl": "first",
                "confidence": "first",
                "is_correct": "first",
                "outcome_group": "first",
                "type": "first",
                "grade": "first",
            }
        )
    )

    configs_dir = output_dir / "configs"
    process_list_dir = output_dir / "process_lists"
    logs_dir = output_dir / "logs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    process_list_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    timing_rows = []
    for row in deduped.itertuples(index=False):
        slide_name = f"{row.slide_id}{args.slide_ext}"
        slide_path = args.slide_dir / slide_name
        if not slide_path.exists():
            timing_rows.append(
                {
                    "slide_id": row.slide_id,
                    "selection_category": row.selection_category,
                    "duration_sec": float("nan"),
                    "exit_code": 1,
                    "status": "missing_slide",
                    "config_path": "",
                    "process_list_path": "",
                }
            )
            continue

        process_list_path = process_list_dir / f"{row.slide_id}.csv"
        process_df = pd.DataFrame(
            [
                {
                    "slide_id": slide_name,
                    "label": row.label_name,
                    "process": 1,
                }
            ]
        )
        process_df.to_csv(process_list_path, index=False)

        config = build_config(args, process_list_path, output_dir)
        config_path = configs_dir / f"{row.slide_id}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        log_path = logs_dir / f"{row.slide_id}.log"
        cmd = [
            str(args.clam_python),
            str(CLAM_ROOT / "create_heatmaps.py"),
            "--config_file",
            str(config_path),
        ]

        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=CLAM_ROOT,
            env={
                **dict(os.environ),
                **(
                    {"TIMM_RESNET50_WEIGHTS_PATH": str(args.encoder_weights_path)}
                    if args.encoder_weights_path and args.encoder_model_name == "resnet50_trunc"
                    else {}
                ),
                **(
                    {"UNI_CKPT_PATH": str(args.encoder_weights_path)}
                    if args.encoder_weights_path and args.encoder_model_name == "uni_v1"
                    else {}
                ),
                **(
                    {"CONCH_CKPT_PATH": str(args.encoder_weights_path)}
                    if args.encoder_weights_path and args.encoder_model_name in {"conch_v1", "conch_v1_5"}
                    else {}
                ),
            },
            input="Y\n",
            text=True,
            capture_output=True,
            check=False,
        )
        duration = time.perf_counter() - start
        log_path.write_text(proc.stdout + "\n\n[stderr]\n" + proc.stderr, encoding="utf-8")

        timing_rows.append(
            {
                "slide_id": row.slide_id,
                "selection_category": row.selection_category,
                "duration_sec": duration,
                "exit_code": proc.returncode,
                "status": "ok" if proc.returncode == 0 else "failed",
                "config_path": str(config_path),
                "process_list_path": str(process_list_path),
                "log_path": str(log_path),
            }
        )

        if proc.returncode == 0:
            raw_slide_dir = output_dir / "raw" / args.exp_code / str(row.label_name) / row.slide_id
            write_slide_summary(pd.Series(row._asdict()), raw_slide_dir / "slide_prediction_summary.txt")

    timing_df = pd.DataFrame(timing_rows)
    timing_csv = output_dir / "heatmap_timing.csv"
    timing_df.to_csv(timing_csv, index=False)

    summary = {
        "selection_csv": str(args.selection_csv),
        "ready_csv": str(args.ready_csv),
        "checkpoint_path": str(args.checkpoint_path),
        "output_dir": str(output_dir),
        "exp_code": args.exp_code,
        "slides_requested": int(len(deduped)),
        "slides_succeeded": int((timing_df["exit_code"] == 0).sum()) if not timing_df.empty else 0,
        "slides_failed": int((timing_df["exit_code"] != 0).sum()) if not timing_df.empty else 0,
        "mean_duration_sec": float(timing_df["duration_sec"].dropna().mean()) if not timing_df.empty else float("nan"),
        "timing_csv": str(timing_csv),
    }
    write_json(output_dir / "heatmap_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
