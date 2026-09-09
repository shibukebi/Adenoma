#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAM_ROOT = PROJECT_ROOT.parent / "CLAM"
sys.path.insert(0, str(CLAM_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from transmil.model import TransMIL  # noqa: E402
from vis_utils.heatmap_utils import drawHeatmap, initialize_wsi  # noqa: E402
from utils.file_utils import save_hdf5  # noqa: E402
from clam_experiment_utils import write_json  # noqa: E402


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Gradient×Input TransMIL heatmaps.")
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--ready-csv", required=True)
    parser.add_argument("--slide-dir", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--feature-h5-dir", required=True)
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--drop-out", type=float, default=0.25)
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--patch-level", type=int, required=True)
    parser.add_argument("--slide-ext", default=".svs")
    parser.add_argument("--vis-level", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--exp-code", default="transmil_gradient_heatmaps")
    return parser.parse_args()


def load_segmentation_defaults() -> tuple[dict, dict]:
    preset_df = pd.read_csv(CLAM_ROOT / "presets" / "bwh_biopsy.csv")
    seg_params = {
        "seg_level": int(preset_df.loc[0, "seg_level"]),
        "sthresh": int(preset_df.loc[0, "sthresh"]),
        "mthresh": int(preset_df.loc[0, "mthresh"]),
        "close": int(preset_df.loc[0, "close"]),
        "use_otsu": bool(preset_df.loc[0, "use_otsu"]),
        "keep_ids": "none",
        "exclude_ids": "none",
    }
    filter_params = {
        "a_t": float(preset_df.loc[0, "a_t"]),
        "a_h": float(preset_df.loc[0, "a_h"]),
        "max_n_holes": int(preset_df.loc[0, "max_n_holes"]),
    }
    return seg_params, filter_params


def load_model(args: argparse.Namespace) -> TransMIL:
    model = TransMIL(embed_dim=args.embed_dim, model_dim=args.model_dim, n_classes=2, dropout=args.drop_out)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device).eval()
    return model


def compute_patch_scores(model: TransMIL, features: np.ndarray) -> tuple[np.ndarray, dict]:
    input_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits, y_prob, y_hat, _, _ = model(input_tensor)
    ssl_score = logits[:, 1].sum()
    grads = torch.autograd.grad(ssl_score, input_tensor)[0]
    raw_scores = torch.sum(torch.abs(grads * input_tensor), dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    percentile_scores = (pd.Series(raw_scores).rank(method="average", pct=True).to_numpy(dtype=np.float32) * 100.0)
    payload = {
        "prob_ssl": float(y_prob[0, 1].item()),
        "prob_others": float(y_prob[0, 0].item()),
        "pred": int(y_hat[0].item()),
        "ssl_logit": float(logits[0, 1].item()),
    }
    return raw_scores, percentile_scores, payload


def save_slide_outputs(
    slide_id: str,
    coords: np.ndarray,
    raw_scores: np.ndarray,
    percentile_scores: np.ndarray,
    args: argparse.Namespace,
    output_root: Path,
    row: pd.Series,
) -> dict:
    raw_dir = output_root / "raw" / args.exp_code / row.label_name / slide_id
    production_dir = output_root / "production" / args.exp_code / row.label_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    production_dir.mkdir(parents=True, exist_ok=True)

    score_csv = raw_dir / f"{slide_id}_patch_scores.csv"
    score_h5 = raw_dir / f"{slide_id}_patch_scores.h5"
    png_path = production_dir / f"{slide_id}_gradientxinput_heatmap.png"
    pdf_path = production_dir / f"{slide_id}_gradientxinput_heatmap.pdf"
    summary_path = raw_dir / "slide_prediction_summary.txt"

    pd.DataFrame(
        {
            "slide_id": slide_id,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "raw_score": raw_scores,
            "percentile_score": percentile_scores,
        }
    ).to_csv(score_csv, index=False)

    save_hdf5(
        str(score_h5),
        {"attention_scores": percentile_scores.reshape(-1, 1).astype(np.float32), "coords": coords.astype(np.int32)},
        attr_dict=None,
        mode="w",
    )

    summary_lines = [
        f"slide_id: {slide_id}",
        f"selection_category: {row.selection_category}",
        f"label: {row.label_name}",
        f"prediction: {row.pred_name}",
        f"prob_ssl: {row.prob_ssl:.6f}",
        f"prob_others: {row.prob_others:.6f}",
        f"confidence: {row.confidence:.6f}",
        f"outcome_group: {row.outcome_group}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "raw_dir": raw_dir,
        "production_dir": production_dir,
        "score_csv": score_csv,
        "score_h5": score_h5,
        "png_path": png_path,
        "pdf_path": pdf_path,
    }


def render_heatmap(
    slide_path: Path,
    coords: np.ndarray,
    percentile_scores: np.ndarray,
    patch_size: int,
    patch_level: int,
    vis_level: int,
    segmentation_pkl_path: Path,
    png_path: Path,
    pdf_path: Path,
) -> None:
    seg_params, filter_params = load_segmentation_defaults()
    wsi_object = initialize_wsi(
        str(slide_path),
        seg_mask_path=str(segmentation_pkl_path),
        seg_params=seg_params,
        filter_params=filter_params,
    )
    ref_downsample = wsi_object.level_downsamples[patch_level]
    vis_patch_size = tuple((np.array([patch_size, patch_size]) * np.array(ref_downsample)).astype(int))
    use_segmentation = bool(getattr(wsi_object, "contours_tissue", None))
    heatmap = drawHeatmap(
        percentile_scores,
        coords,
        slide_path=str(slide_path),
        wsi_object=wsi_object,
        cmap="jet",
        alpha=0.4,
        use_holes=use_segmentation,
        binarize=False,
        vis_level=vis_level,
        blank_canvas=False,
        thresh=-1,
        patch_size=vis_patch_size,
        convert_to_percentiles=False,
        segment=use_segmentation,
    )
    heatmap.save(png_path)
    heatmap.save(pdf_path, "PDF")


def main() -> None:
    args = parse_args()
    args.slide_dir = Path(args.slide_dir)
    args.checkpoint_path = Path(args.checkpoint_path)
    args.feature_h5_dir = Path(args.feature_h5_dir)
    args.patch_dir = Path(args.patch_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_df = pd.read_csv(args.selection_csv)
    ready_df = pd.read_csv(args.ready_csv, dtype={"slide_id": str})
    selected = selection_df.merge(ready_df[["slide_id", "type", "grade"]], on="slide_id", how="left")
    model = load_model(args)

    timing_rows = []
    for row in selected.itertuples(index=False):
        slide_id = str(row.slide_id)
        slide_path = args.slide_dir / f"{slide_id}{args.slide_ext}"
        feature_h5_path = args.feature_h5_dir / f"{slide_id}.h5"
        patch_h5_path = args.patch_dir / f"{slide_id}.h5"

        start = time.perf_counter()
        status = "ok"
        try:
            with h5py.File(feature_h5_path, "r") as feature_file:
                features = feature_file["features"][:].astype(np.float32)
                coords = feature_file["coords"][:].astype(np.int32)
            if patch_h5_path.exists():
                with h5py.File(patch_h5_path, "r") as patch_file:
                    patch_dset = patch_file["coords"]
                    patch_size = int(patch_dset.attrs.get("patch_size", args.patch_size))
                    patch_level = int(patch_dset.attrs.get("patch_level", args.patch_level))
            else:
                patch_size = args.patch_size
                patch_level = args.patch_level

            raw_scores, percentile_scores, payload = compute_patch_scores(model, features)
            saved = save_slide_outputs(
                slide_id,
                coords,
                raw_scores,
                percentile_scores,
                args,
                output_dir,
                pd.Series(row._asdict()),
            )
            render_heatmap(
                slide_path=slide_path,
                coords=coords,
                percentile_scores=percentile_scores,
                patch_size=patch_size,
                patch_level=patch_level,
                vis_level=args.vis_level,
                segmentation_pkl_path=saved["raw_dir"] / f"{slide_id}_segmentation.pkl",
                png_path=saved["png_path"],
                pdf_path=saved["pdf_path"],
            )
            result_meta = {
                "slide_id": slide_id,
                "selection_category": row.selection_category,
                "status": status,
                "patch_count": int(len(coords)),
                "prob_ssl_forward": payload["prob_ssl"],
                "prob_others_forward": payload["prob_others"],
                "ssl_logit": payload["ssl_logit"],
                "score_csv": str(saved["score_csv"]),
                "score_h5": str(saved["score_h5"]),
                "png_path": str(saved["png_path"]),
                "pdf_path": str(saved["pdf_path"]),
            }
            write_json(saved["raw_dir"] / "metadata.json", result_meta)
        except Exception as exc:  # noqa: BLE001
            status = f"failed: {exc}"
        timing_rows.append(
            {
                "slide_id": slide_id,
                "selection_category": row.selection_category,
                "status": status,
                "duration_sec": time.perf_counter() - start,
            }
        )

    timing_df = pd.DataFrame(timing_rows)
    timing_csv = output_dir / "heatmap_timing.csv"
    timing_df.to_csv(timing_csv, index=False)
    summary = {
        "selection_csv": str(args.selection_csv),
        "checkpoint_path": str(args.checkpoint_path),
        "output_dir": str(output_dir),
        "slides_requested": int(len(selected)),
        "slides_succeeded": int((timing_df["status"] == "ok").sum()) if not timing_df.empty else 0,
        "slides_failed": int((timing_df["status"] != "ok").sum()) if not timing_df.empty else 0,
        "timing_csv": str(timing_csv),
        "attribution_method": "gradient_x_input_abs_sum",
    }
    write_json(output_dir / "heatmap_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
