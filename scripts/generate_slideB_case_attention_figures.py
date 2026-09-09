#!/usr/bin/env python3
from collections import Counter
import argparse
from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAM_ROOT = PROJECT_ROOT / "CLAM"
sys.path.insert(0, str(CLAM_ROOT))

from models.model_clam import CLAM_SB  # noqa: E402
from wsi_core.WholeSlideImage import WholeSlideImage  # noqa: E402


RESULT_ROOT = Path("/data15/zhengke_usb2/yuexin_data/result/5fold_11class")
FEATURE_ROOT = Path(
    "/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/joint_hp_yx/features/10x"
)
GROUP_ROOT = RESULT_ROOT / "summary_11class_5fold" / "group_meeting_error_slides"
OUT_ROOT = PROJECT_ROOT / "outputs" / "slideB_case_figures"
COMPONENT_ROOT = OUT_ROOT / "components"
PREDICTIONS_PATH = GROUP_ROOT / "selected_case_predictions_14config.csv"

CLASS_SHORT = {
    0: "SSL",
    1: "HP",
    2: "TSA",
    3: "USA",
    4: "TA",
    5: "TVA",
    6: "IP",
    7: "SSLD",
    8: "TSAD",
    9: "TAD",
    10: "TVAD",
}
CLASS_ORDER = ["IP", "HP", "SSL", "SSLD", "TSA", "TSAD", "USA", "TA", "TAD", "TVA", "TVAD"]

CASES = [
    {
        "index": 1,
        "slide_id": "13b5fac2-a2f9-47ba-9c91-f52b6a9aa2f8",
        "pattern": "HP -> IP",
        "true_class": "HP",
        "fold": 3,
        "checkpoint": RESULT_ROOT / "CLAM-SB/11class/10x/fold-3/s_3_checkpoint.pt",
        "wsi_path": Path(
            "/data15/zhengke_usb2/yuexin_data/Adenoma_hp/13b5fac2-a2f9-47ba-9c91-f52b6a9aa2f8.isyntax"
        ),
        "review_focus": "Reactive/inflammatory morphology may dominate the slide-level signal.",
    },
    {
        "index": 2,
        "slide_id": "649957 1",
        "pattern": "IP -> HP",
        "true_class": "IP",
        "fold": 2,
        "checkpoint": RESULT_ROOT / "CLAM-SB/11class/10x/fold-2/s_2_checkpoint.pt",
        "wsi_path": Path("/data15/zhengke_usb/Adenoma_yx/649957 1.svs"),
        "review_focus": "Serrated/reactive glands may resemble a hyperplastic polyp.",
    },
    {
        "index": 3,
        "slide_id": "2691bae9-1a36-463d-8107-217ffe9005c9",
        "pattern": "TAD -> TA",
        "true_class": "TAD",
        "fold": 3,
        "checkpoint": RESULT_ROOT / "CLAM-SB/11class/10x/fold-3/s_3_checkpoint.pt",
        "wsi_path": Path(
            "/data15/zhengke_usb2/yuexin_data/Adenoma_hp/2691bae9-1a36-463d-8107-217ffe9005c9.isyntax"
        ),
        "review_focus": "A focal high-grade component may be diluted by abundant low-grade patches.",
    },
    {
        "index": 4,
        "slide_id": "80bb9bd9-6f7b-45ac-9c6b-90221808a146",
        "pattern": "TVA -> TA",
        "true_class": "TVA",
        "fold": 3,
        "checkpoint": RESULT_ROOT / "CLAM-SB/11class/10x/fold-3/s_3_checkpoint.pt",
        "wsi_path": Path(
            "/data15/zhengke_usb2/yuexin_data/Adenoma_hp/80bb9bd9-6f7b-45ac-9c6b-90221808a146.isyntax"
        ),
        "review_focus": "The villous component may be under-sampled or receive low attention.",
    },
    {
        "index": 5,
        "slide_id": "650816 4",
        "pattern": "SSL -> USA",
        "true_class": "SSL",
        "fold": 4,
        "checkpoint": Path(
            "/data15/zhengke_usb2/yuexin_data/result/fold5_hp+yx/CLAM-SB/11class/10x/s_5_checkpoint.pt"
        ),
        "wsi_path": Path("/data15/zhengke_usb/Adenoma_yx/650816 4.svs"),
        "review_focus": "Overlapping serrated crypt architecture may favor the adjacent subtype.",
    },
]

ZOOM_CASE_INDICES = {1, 3, 4}


def safe_stem(case):
    pattern = case["pattern"].replace(" -> ", "_to_")
    slide = case["slide_id"].replace(" ", "_")
    prefix = f"{case['index']:02d}_" if case.get("show_case_number", True) else ""
    return f"{prefix}{pattern}_{slide}"


def percentile_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    if len(values) > 1:
        ranks /= len(values) - 1
    return ranks


def select_upper_left_tissue_component(coords, step_size):
    grid = coords // step_size
    cell_to_indices = {}
    for index, cell in enumerate(grid):
        cell_to_indices.setdefault(tuple(map(int, cell)), []).append(index)

    remaining = set(cell_to_indices)
    components = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        cells = []
        while stack:
            cell = stack.pop()
            cells.append(cell)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor = (cell[0] + dx, cell[1] + dy)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        indices = [index for cell in cells for index in cell_to_indices[cell]]
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        components.append(
            {
                "indices": np.asarray(indices, dtype=int),
                "size": len(indices),
                "min_x": min(xs),
                "min_y": min(ys),
                "max_x": max(xs),
                "max_y": max(ys),
            }
        )

    largest = max(component["size"] for component in components)
    candidates = [
        component
        for component in components
        if component["size"] >= 0.5 * largest
        and not (component["min_x"] <= 2 and component["min_y"] <= 2)
    ]
    top_y = min(component["min_y"] for component in candidates)
    top_row = [component for component in candidates if component["min_y"] <= top_y + 15]
    selected = min(top_row, key=lambda component: (component["min_x"], component["min_y"]))
    return selected


def load_model(checkpoint):
    model = CLAM_SB(
        dropout=0.25,
        n_classes=11,
        embed_dim=1024,
        size_arg="small",
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    clean = {
        key.replace(".module", ""): value
        for key, value in state.items()
        if "instance_loss_fn" not in key
    }
    model.load_state_dict(clean, strict=True)
    model.eval()
    return model


def choose_thumbnail_level(wsi, max_source_size=3200):
    levels = list(enumerate(wsi.level_dimensions))
    eligible = [(level, dims) for level, dims in levels if max(dims) <= max_source_size]
    return eligible[0][0] if eligible else levels[-1][0]


def read_case_assets(case, coords, patch_level, patch_size, extent_level0, top_indices):
    wsi_object = WholeSlideImage(str(case["wsi_path"]))
    wsi = wsi_object.wsi
    level = choose_thumbnail_level(wsi)
    thumbnail = wsi.read_region((0, 0), level, wsi.level_dimensions[level]).convert("RGB")
    thumbnail.thumbnail((2400, 1600), Image.Resampling.LANCZOS)

    level0_width, level0_height = map(float, wsi.level_dimensions[0])
    scale_x = thumbnail.width / level0_width
    scale_y = thumbnail.height / level0_height

    x_min = max(0.0, float(coords[:, 0].min()) - extent_level0)
    y_min = max(0.0, float(coords[:, 1].min()) - extent_level0)
    x_max = min(level0_width, float(coords[:, 0].max()) + 2 * extent_level0)
    y_max = min(level0_height, float(coords[:, 1].max()) + 2 * extent_level0)
    crop_box = (
        max(0, int(x_min * scale_x)),
        max(0, int(y_min * scale_y)),
        min(thumbnail.width, int(np.ceil(x_max * scale_x))),
        min(thumbnail.height, int(np.ceil(y_max * scale_y))),
    )
    cropped = thumbnail.crop(crop_box)

    cropped_coords = coords.astype(float).copy()
    cropped_coords[:, 0] = cropped_coords[:, 0] * scale_x - crop_box[0]
    cropped_coords[:, 1] = cropped_coords[:, 1] * scale_y - crop_box[1]
    scaled_extent = (extent_level0 * scale_x, extent_level0 * scale_y)

    top_patches = []
    for index in top_indices:
        location = tuple(map(int, coords[index]))
        patch = wsi.read_region(location, patch_level, (patch_size, patch_size)).convert("RGB")
        top_patches.append(patch)

    if hasattr(wsi, "close"):
        wsi.close()
    return cropped, cropped_coords, scaled_extent, top_patches


def build_attention_overlay(wsi_image, coords, scaled_extent, attention, top_indices):
    base = np.asarray(wsi_image, dtype=np.float32) / 255.0
    height, width = base.shape[:2]
    heat = np.zeros((height, width), dtype=np.float32)
    covered = np.zeros((height, width), dtype=bool)
    extent_x = max(2, int(round(scaled_extent[0])))
    extent_y = max(2, int(round(scaled_extent[1])))

    for (x, y), score in zip(coords, attention):
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(width, x0 + extent_x)
        y1 = min(height, y0 + extent_y)
        if x1 <= x0 or y1 <= y0:
            continue
        region = heat[y0:y1, x0:x1]
        np.maximum(region, score, out=region)
        covered[y0:y1, x0:x1] = True

    colors = plt.get_cmap("turbo")(heat)[..., :3]
    alpha = np.zeros((height, width), dtype=np.float32)
    alpha[covered] = 0.18 + 0.62 * heat[covered]
    blended = base * (1.0 - alpha[..., None]) + colors * alpha[..., None]
    image = Image.fromarray(np.clip(blended * 255, 0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(image)
    line_width = max(2, int(round(min(width, height) / 300)))
    for rank, index in enumerate(top_indices, start=1):
        x, y = coords[index]
        x0 = int(round(x))
        y0 = int(round(y))
        x1 = min(width - 1, x0 + extent_x)
        y1 = min(height - 1, y0 + extent_y)
        draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 0), width=line_width)
        draw.text((x0 + 3, y0 + 3), str(rank), fill=(0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 0))
    return image


def plot_prediction_distribution(ax, case_predictions):
    counts = Counter(case_predictions["pred_class"])
    values = [counts.get(name, 0) for name in CLASS_ORDER]
    colors = [plt.get_cmap("tab20")(index / len(CLASS_ORDER)) for index in range(len(CLASS_ORDER))]
    bars = ax.bar(CLASS_ORDER, values, color=colors, edgecolor="white")
    ax.set_ylim(0, 14.8)
    ax.set_ylabel("Votes (14 configurations)")
    ax.set_title("Prediction distribution across models and magnifications", fontsize=12)
    ax.tick_params(axis="x", rotation=40)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        if value:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, str(value), ha="center", fontsize=9)


def save_case_figure(case, wsi_image, attention_image, top_patches, attention, top_indices, prediction, confidence, case_predictions):
    stem = safe_stem(case)
    fig = plt.figure(figsize=(17, 9.5), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.05, 1.05, 0.70],
        height_ratios=[0.76, 0.24],
        hspace=0.22,
        wspace=0.12,
    )

    ax_wsi = fig.add_subplot(grid[0, 0])
    ax_wsi.imshow(wsi_image)
    ax_wsi.set_title("Whole-slide image", fontsize=15)
    ax_wsi.axis("off")

    ax_attention = fig.add_subplot(grid[0, 1])
    ax_attention.imshow(attention_image)
    ax_attention.set_title(
        f"CLAM-SB 10x attention | Pred: {prediction} ({confidence:.1%})",
        fontsize=14,
    )
    ax_attention.axis("off")
    scalar = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="turbo")
    colorbar = fig.colorbar(scalar, ax=ax_attention, fraction=0.035, pad=0.02)
    colorbar.set_label("Attention percentile", fontsize=9)

    patch_grid = grid[0, 2].subgridspec(3, 1, hspace=0.28)
    for rank, (patch, index) in enumerate(zip(top_patches, top_indices), start=1):
        ax_patch = fig.add_subplot(patch_grid[rank - 1, 0])
        ax_patch.imshow(patch)
        ax_patch.set_title(f"Top {rank} patch | {attention[index]:.1%} percentile", fontsize=10)
        ax_patch.axis("off")

    ax_votes = fig.add_subplot(grid[1, :2])
    plot_prediction_distribution(ax_votes, case_predictions)

    ax_meta = fig.add_subplot(grid[1, 2])
    ax_meta.axis("off")
    vote_summary = ", ".join(
        f"{name}:{count}" for name, count in Counter(case_predictions["pred_class"]).most_common()
    )
    ax_meta.text(0, 0.95, f"True label: {case['true_class']}", fontsize=12, weight="bold", va="top")
    ax_meta.text(0, 0.72, f"14-config votes: {vote_summary}", fontsize=10, va="top", wrap=True)
    ax_meta.text(0, 0.43, "Review focus", fontsize=11, weight="bold", color="#a13f37", va="top")
    ax_meta.text(0, 0.27, case["review_focus"], fontsize=9.5, va="top", wrap=True)

    title = f"{case['pattern']} | {case['slide_id']}"
    if case.get("show_case_number", True):
        title = f"Case {case['index']} | {title}"
    fig.suptitle(title, fontsize=20, weight="bold", y=0.985)
    out_path = OUT_ROOT / f"{stem}.jpg"
    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
        format="jpg",
        pil_kwargs={"quality": 93, "optimize": True},
    )
    plt.close(fig)
    return out_path


def process_case(case, predictions):
    feature_path = FEATURE_ROOT / "h5_files" / f"{case['slide_id']}.h5"
    with h5py.File(feature_path, "r") as handle:
        features = handle["features"][:]
        coords = handle["coords"][:].astype(np.int64)
        patch_level = int(handle.attrs["legacy_patch_level"])
        patch_size = int(handle.attrs["legacy_patch_size"])
        extent_level0 = int(handle.attrs["physical_level_0_extent"])
        step_size = int(handle.attrs.get("physical_level0_step_size", extent_level0))

    model = load_model(case["checkpoint"])
    with torch.no_grad():
        logits, probabilities, predicted, raw_attention, _ = model(torch.from_numpy(features).float())
    predicted_id = int(predicted.item())
    predicted_class = CLASS_SHORT[predicted_id]
    confidence = float(probabilities[0, predicted_id].item())
    attention = percentile_ranks(raw_attention.squeeze(0).cpu().numpy())

    selected_mask = np.ones(len(coords), dtype=bool)
    selected_component = None
    if case.get("zoom_upper_left", case["index"] in ZOOM_CASE_INDICES):
        selected_component = select_upper_left_tissue_component(coords, step_size)
        selected_mask[:] = False
        selected_mask[selected_component["indices"]] = True
    selected_coords = coords[selected_mask]
    selected_attention = attention[selected_mask]
    top_indices = np.argsort(selected_attention)[-3:][::-1]

    case_predictions = predictions[predictions["slide_id"].eq(case["slide_id"])].copy()
    clam_row = case_predictions[
        case_predictions["model"].eq("CLAM-SB") & case_predictions["feature"].eq("10x")
    ]
    if len(clam_row) != 1:
        raise ValueError(f"Expected one CLAM-SB 10x prediction for {case['slide_id']}")
    expected = clam_row["pred_class"].iat[0]
    if predicted_class != expected:
        raise ValueError(
            f"Checkpoint prediction mismatch for {case['slide_id']}: {predicted_class} != {expected}"
        )

    wsi_image, cropped_coords, scaled_extent, top_patches = read_case_assets(
        case,
        selected_coords,
        patch_level,
        patch_size,
        extent_level0,
        top_indices,
    )
    attention_image = build_attention_overlay(
        wsi_image,
        cropped_coords,
        scaled_extent,
        selected_attention,
        top_indices,
    )

    stem = safe_stem(case)
    wsi_path = COMPONENT_ROOT / f"{stem}_wsi.jpg"
    attention_path = COMPONENT_ROOT / f"{stem}_attention.jpg"
    wsi_image.save(wsi_path, quality=93, optimize=True)
    attention_image.save(attention_path, quality=93, optimize=True)
    for rank, patch in enumerate(top_patches, start=1):
        patch.save(COMPONENT_ROOT / f"{stem}_top{rank}.jpg", quality=95, optimize=True)

    score_frame = pd.DataFrame(
        {
            "patch_index": np.arange(len(coords)),
            "x_level0": coords[:, 0],
            "y_level0": coords[:, 1],
            "attention_raw": raw_attention.squeeze(0).cpu().numpy(),
            "attention_percentile": attention,
            "selected_upper_left_component": selected_mask,
        }
    ).sort_values("attention_percentile", ascending=False)
    score_frame.to_csv(COMPONENT_ROOT / f"{stem}_attention_scores.csv", index=False)

    figure_path = save_case_figure(
        case,
        wsi_image,
        attention_image,
        top_patches,
        selected_attention,
        top_indices,
        predicted_class,
        confidence,
        case_predictions,
    )
    return {
        "case_index": case["index"],
        "slide_id": case["slide_id"],
        "pattern": case["pattern"],
        "true_class": case["true_class"],
        "clam_10x_prediction": predicted_class,
        "clam_10x_confidence": confidence,
        "n_patches": len(coords),
        "displayed_patches": len(selected_coords),
        "selected_component_bbox_grid": (
            "all"
            if selected_component is None
            else f"{selected_component['min_x']},{selected_component['min_y']},"
            f"{selected_component['max_x']},{selected_component['max_y']}"
        ),
        "figure": str(figure_path),
        "wsi_component": str(wsi_path),
        "attention_component": str(attention_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", type=int, default=None)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    COMPONENT_ROOT.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTIONS_PATH, dtype={"slide_id": str})
    rows = []
    cases = CASES if not args.only else [case for case in CASES if case["index"] in set(args.only)]
    for case in cases:
        print(f"Processing {case['index']}/5: {case['slide_id']}", flush=True)
        rows.append(process_case(case, predictions))
    index = pd.DataFrame(rows)
    index.to_csv(OUT_ROOT / "case_figure_index.csv", index=False)
    print(index.to_string(index=False))


if __name__ == "__main__":
    main()
