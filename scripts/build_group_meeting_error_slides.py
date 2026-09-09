#!/usr/bin/env python3
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from analyze_11class_confusions import EXPERIMENTS, CLASS_SHORT, load_experiment


SUMMARY_ROOT = Path(
    "/data15/zhengke_usb2/yuexin_data/result/5fold_11class/summary_11class_5fold"
)
ANALYSIS_ROOT = SUMMARY_ROOT / "confusion_error_analysis"
OUT_ROOT = SUMMARY_ROOT / "group_meeting_error_slides"
META_PATH = Path(
    "/data15/zhengke_usb2/yuexin_data/splits/adenoma_uni_hp_yx_ssl_5fold/"
    "joint_hp_yx_ssl_ready.csv"
)

PLOT_ORDER = [6, 1, 0, 7, 2, 8, 3, 4, 9, 5, 10]
PLOT_LABELS = [CLASS_SHORT[index] for index in PLOT_ORDER]

PATHWAY_CLASSES = {
    "Benign": [1, 6],
    "Serrated": [0, 7, 2, 8],
    "Conventional": [4, 9, 5, 10],
    "Mixed / USA": [3],
}
PATHWAY_COLORS = {
    "Benign": "#4c78a8",
    "Serrated": "#59a14f",
    "Conventional": "#e15759",
    "Mixed / USA": "#b07aa1",
}
PATHWAY_ORDER = [index for indices in PATHWAY_CLASSES.values() for index in indices]
PATHWAY_LABELS = [CLASS_SHORT[index] for index in PATHWAY_ORDER]

SELECTED_CASES = [
    {
        "slide_id": "13b5fac2-a2f9-47ba-9c91-f52b6a9aa2f8",
        "pattern": "HP -> IP",
        "wsi_path": "/data15/zhengke_usb2/yuexin_data/Adenoma_hp/13b5fac2-a2f9-47ba-9c91-f52b6a9aa2f8.isyntax",
        "wsi_status": "available iSyntax; dedicated decoder required",
    },
    {
        "slide_id": "649957 1",
        "pattern": "IP -> HP",
        "wsi_path": "/data15/zhengke_usb/Adenoma_yx/649957 1.svs",
        "wsi_status": "available SVS; source mount currently slow",
    },
    {
        "slide_id": "2691bae9-1a36-463d-8107-217ffe9005c9",
        "pattern": "TAD -> TA",
        "wsi_path": "/data15/zhengke_usb2/yuexin_data/Adenoma_hp/2691bae9-1a36-463d-8107-217ffe9005c9.isyntax",
        "wsi_status": "available iSyntax; dedicated decoder required",
    },
    {
        "slide_id": "80bb9bd9-6f7b-45ac-9c6b-90221808a146",
        "pattern": "TVA -> TA",
        "wsi_path": "/data15/zhengke_usb2/yuexin_data/Adenoma_hp/80bb9bd9-6f7b-45ac-9c6b-90221808a146.isyntax",
        "wsi_status": "available iSyntax; dedicated decoder required",
    },
    {
        "slide_id": "650816 4",
        "pattern": "SSL -> USA",
        "wsi_path": "/data15/zhengke_usb/Adenoma_yx/650816 4.svs",
        "wsi_status": "available SVS; source mount currently slow",
    },
]


def load_all_predictions():
    frames = []
    for model, old_model, feature in EXPERIMENTS:
        frames.append(load_experiment(model, old_model, feature))
    return pd.concat(frames, ignore_index=True)


def save_slide_a(all_predictions):
    cm = confusion_matrix(
        all_predictions["label"].astype(int),
        all_predictions["pred"].astype(int),
        labels=PLOT_ORDER,
    ).astype(float)
    cm /= np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.23, 0.77], hspace=0.23, wspace=0.18)
    ax_cm = fig.add_subplot(grid[:, 0])
    ax_pairs = fig.add_subplot(grid[0, 1])
    ax_high = fig.add_subplot(grid[1, 1])

    image = ax_cm.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_title("Mean error landscape across 14 configurations", fontsize=17, pad=14)
    ax_cm.set_xlabel("Predicted class", fontsize=12)
    ax_cm.set_ylabel("True class", fontsize=12)
    ax_cm.set_xticks(range(11), PLOT_LABELS, rotation=45, ha="right")
    ax_cm.set_yticks(range(11), PLOT_LABELS)
    for row in range(11):
        for col in range(11):
            value = cm[row, col]
            ax_cm.text(
                col,
                row,
                f"{100 * value:.1f}%",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if value > 0.52 else "black",
            )
    colorbar = fig.colorbar(image, ax=ax_cm, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized proportion", fontsize=10)

    ax_pairs.set_title("Major bidirectional confusions", fontsize=16, pad=10)
    ax_pairs.axis("off")
    pairs = [
        ("HP", "IP", 12.82, 33.92),
        ("TA", "HP", 6.92, 7.77),
        ("TA", "TVA", 5.18, 20.58),
        ("SSL", "HP", 14.20, 6.70),
        ("SSL", "USA", 11.26, 17.98),
    ]
    y_positions = np.linspace(0.84, 0.12, len(pairs))
    for y, (left, right, left_to_right, right_to_left) in zip(y_positions, pairs):
        ax_pairs.text(
            0.12,
            y,
            left,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#e8f1f7", ec="#447a9c"),
        )
        ax_pairs.text(
            0.88,
            y,
            right,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f7ece8", ec="#a46046"),
        )
        ax_pairs.add_patch(
            FancyArrowPatch(
                (0.22, y + 0.025),
                (0.78, y + 0.025),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.4 + left_to_right / 13,
                color="#b64b3c",
            )
        )
        ax_pairs.add_patch(
            FancyArrowPatch(
                (0.78, y - 0.025),
                (0.22, y - 0.025),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.4 + right_to_left / 13,
                color="#386f91",
            )
        )
        ax_pairs.text(0.5, y + 0.052, f"{left_to_right:.2f}%", ha="center", fontsize=8.5)
        ax_pairs.text(0.5, y - 0.083, f"{right_to_left:.2f}%", ha="center", fontsize=8.5)

    ax_high.set_title("High-grade regression to adjacent classes", fontsize=16, pad=10)
    ax_high.axis("off")
    arrows = [
        ("TAD", "TA", 19.98),
        ("TVAD", "TVA", 22.57),
        ("TSAD", "TVA", 23.67),
        ("TSAD", "TVAD", 18.04),
    ]
    y_positions = [0.80, 0.59, 0.35, 0.14]
    for y, (source, target, rate) in zip(y_positions, arrows):
        ax_high.text(
            0.12,
            y,
            source,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f7e7e7", ec="#a44b4b"),
        )
        ax_high.text(
            0.88,
            y,
            target,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#e8f1f7", ec="#447a9c"),
        )
        ax_high.add_patch(
            FancyArrowPatch(
                (0.22, y),
                (0.78, y),
                arrowstyle="simple",
                mutation_scale=10 + rate / 2,
                color="#c55245",
                alpha=0.72,
            )
        )
        ax_high.text(0.5, y + 0.065, f"{rate:.2f}%", ha="center", fontsize=9, weight="bold")

    fig.suptitle("Slide A | Overall 11-class error atlas", fontsize=22, weight="bold", y=0.985)
    fig.savefig(OUT_ROOT / "slide_A_overall_error_atlas.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _directional_confusions(all_predictions):
    counts = (
        all_predictions.groupby(["label", "pred"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = all_predictions.groupby("label").size()
    counts["rate"] = counts["count"] / counts["label"].map(totals)
    counts["true_class"] = counts["label"].map(CLASS_SHORT)
    counts["pred_class"] = counts["pred"].map(CLASS_SHORT)
    counts["confusion"] = counts["true_class"] + " -> " + counts["pred_class"]
    class_pathway = {
        index: pathway
        for pathway, indices in PATHWAY_CLASSES.items()
        for index in indices
    }
    counts["true_pathway"] = counts["label"].map(class_pathway)
    counts["pred_pathway"] = counts["pred"].map(class_pathway)
    return counts[counts["label"].ne(counts["pred"])].copy(), class_pathway


def _plot_ranked_confusions(ax, rows, title, color, limit):
    rows = rows.sort_values(["rate", "count"], ascending=False).head(limit).copy()
    ax.set_title(title, fontsize=12.5, weight="bold", pad=7, loc="left")
    if rows.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No within-pathway pair", ha="center", va="center")
        return

    y = np.arange(len(rows))
    values = 100 * rows["rate"].to_numpy()
    ax.barh(y, values, color=color, alpha=0.88, height=0.62)
    ax.set_yticks(y, rows["confusion"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.28)
    ax.set_xlabel("Within-true-class error rate (%)", fontsize=8.5)
    ax.tick_params(axis="both", labelsize=8.2)
    ax.grid(axis="x", color="#d9dde2", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for position, value in enumerate(values):
        ax.text(value + max(values) * 0.025, position, f"{value:.2f}%", va="center", fontsize=8.1)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#aeb5bd")


def save_slide_a_pathway(all_predictions, output_path=None):
    output_path = output_path or OUT_ROOT / "slide_A_pathway_confusion_atlas.png"
    directional, class_pathway = _directional_confusions(all_predictions)

    class_cm = confusion_matrix(
        all_predictions["label"].astype(int),
        all_predictions["pred"].astype(int),
        labels=PATHWAY_ORDER,
    ).astype(float)
    class_cm /= np.maximum(class_cm.sum(axis=1, keepdims=True), 1)

    pathway_names = list(PATHWAY_CLASSES)
    true_pathway = all_predictions["label"].map(class_pathway)
    pred_pathway = all_predictions["pred"].map(class_pathway)
    pathway_cm = pd.crosstab(true_pathway, pred_pathway).reindex(
        index=pathway_names, columns=pathway_names, fill_value=0
    ).to_numpy(dtype=float)
    pathway_cm /= np.maximum(pathway_cm.sum(axis=1, keepdims=True), 1)

    fig = plt.figure(figsize=(19.2, 10.8), facecolor="white")
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.36, 0.82, 0.82],
        height_ratios=[0.30, 0.40, 0.30],
        left=0.055,
        right=0.985,
        bottom=0.085,
        top=0.91,
        hspace=0.43,
        wspace=0.36,
    )
    ax_cm = fig.add_subplot(grid[:, 0])
    ax_benign = fig.add_subplot(grid[0, 1])
    ax_serrated = fig.add_subplot(grid[0, 2])
    ax_conventional = fig.add_subplot(grid[1, 1])
    ax_cross = fig.add_subplot(grid[1, 2])
    ax_pathway = fig.add_subplot(grid[2, 1:])

    image = ax_cm.imshow(class_cm, cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_title(
        "11-class confusion matrix, ordered by pathway",
        fontsize=16,
        weight="bold",
        pad=34,
    )
    ax_cm.set_xlabel("Predicted class", fontsize=11)
    ax_cm.set_ylabel("True class", fontsize=11)
    ax_cm.set_xticks(range(11), PATHWAY_LABELS, rotation=45, ha="right")
    ax_cm.set_yticks(range(11), PATHWAY_LABELS)
    for row in range(11):
        for col in range(11):
            value = class_cm[row, col]
            ax_cm.text(
                col,
                row,
                f"{100 * value:.1f}%",
                ha="center",
                va="center",
                fontsize=7.1,
                color="white" if value > 0.52 else "#17202a",
            )

    starts = np.cumsum([0] + [len(indices) for indices in PATHWAY_CLASSES.values()])
    for boundary in starts[1:-1]:
        ax_cm.axhline(boundary - 0.5, color="white", linewidth=3)
        ax_cm.axvline(boundary - 0.5, color="white", linewidth=3)
    for pathway, start, end in zip(pathway_names, starts[:-1], starts[1:]):
        color = PATHWAY_COLORS[pathway]
        ax_cm.add_patch(
            Rectangle(
                (start - 0.47, start - 0.47),
                end - start - 0.06,
                end - start - 0.06,
                fill=False,
                ec=color,
                lw=2.4,
            )
        )
        center = (start + end - 1) / 2
        ax_cm.text(
            center,
            -1.22,
            pathway,
            ha="center",
            va="center",
            fontsize=9.5,
            weight="bold",
            color=color,
            clip_on=False,
        )
        for tick in range(start, end):
            ax_cm.get_xticklabels()[tick].set_color(color)
            ax_cm.get_yticklabels()[tick].set_color(color)
            ax_cm.get_xticklabels()[tick].set_weight("bold")
            ax_cm.get_yticklabels()[tick].set_weight("bold")
    colorbar = fig.colorbar(image, ax=ax_cm, fraction=0.042, pad=0.035)
    colorbar.set_label("Row-normalized proportion", fontsize=9)

    within = directional[directional["true_pathway"].eq(directional["pred_pathway"])]
    _plot_ranked_confusions(
        ax_benign,
        within[within["true_pathway"].eq("Benign")],
        "Benign: ranked within-pathway errors",
        PATHWAY_COLORS["Benign"],
        4,
    )
    _plot_ranked_confusions(
        ax_serrated,
        within[within["true_pathway"].eq("Serrated")],
        "Serrated: ranked within-pathway errors",
        PATHWAY_COLORS["Serrated"],
        5,
    )
    _plot_ranked_confusions(
        ax_conventional,
        within[within["true_pathway"].eq("Conventional")],
        "Conventional: ranked within-pathway errors",
        PATHWAY_COLORS["Conventional"],
        6,
    )
    cross = directional[directional["true_pathway"].ne(directional["pred_pathway"])]
    _plot_ranked_confusions(
        ax_cross,
        cross,
        "Mixed / cross-pathway errors (ranked)",
        "#b07aa1",
        8,
    )

    pathway_image = ax_pathway.imshow(pathway_cm, cmap="YlOrRd", vmin=0, vmax=1)
    ax_pathway.set_aspect("auto")
    ax_pathway.set_title(
        "Pathway-level confusion matrix",
        fontsize=12.5,
        weight="bold",
        pad=7,
        loc="left",
    )
    ax_pathway.set_xlabel("Predicted pathway", fontsize=9)
    ax_pathway.set_ylabel("True pathway", fontsize=9)
    ax_pathway.set_xticks(range(4), pathway_names, rotation=20, ha="right")
    ax_pathway.set_yticks(range(4), pathway_names)
    ax_pathway.tick_params(axis="both", labelsize=8.2)
    for row in range(4):
        for col in range(4):
            value = pathway_cm[row, col]
            ax_pathway.text(
                col,
                row,
                f"{100 * value:.1f}%",
                ha="center",
                va="center",
                fontsize=8.3,
                color="white" if value > 0.48 else "#241a12",
                weight="bold" if row == col else "normal",
            )
    pathway_colorbar = fig.colorbar(pathway_image, ax=ax_pathway, fraction=0.035, pad=0.025)
    pathway_colorbar.set_label("Row-normalized proportion", fontsize=8)

    fig.suptitle(
        "Slide A | Pathway-oriented major confusion atlas",
        fontsize=22,
        weight="bold",
        y=0.975,
    )
    fig.text(
        0.055,
        0.025,
        "Pooled test predictions from 14 model/magnification configurations. "
        "Bars are sorted by directional within-true-class error rate; USA is shown as the mixed/ambiguous category.",
        fontsize=9.2,
        color="#4d5966",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def selected_case_tables(all_predictions):
    meta = pd.read_csv(META_PATH, dtype={"slide_id": str})
    meta = meta[["slide_id", "scanner", "type", "grade", "cv_test_fold"]].drop_duplicates("slide_id")
    selected_ids = [case["slide_id"] for case in SELECTED_CASES]
    selected_predictions = all_predictions[all_predictions["slide_id"].isin(selected_ids)].copy()
    selected_predictions[
        [
            "slide_id",
            "model",
            "feature",
            "configuration",
            "true_class",
            "pred_class",
            "confidence",
            "is_correct",
        ]
    ].sort_values(["slide_id", "model", "feature"]).to_csv(
        OUT_ROOT / "selected_case_predictions_14config.csv", index=False
    )

    rows = []
    for case in SELECTED_CASES:
        slide_id = case["slide_id"]
        group = selected_predictions[selected_predictions["slide_id"].eq(slide_id)]
        metadata = meta[meta["slide_id"].eq(slide_id)].iloc[0]
        counts = Counter(group["pred_class"])
        rows.append(
            {
                **case,
                "true_class": group["true_class"].iat[0],
                "scanner_source": metadata["scanner"],
                "type": metadata["type"],
                "grade": metadata["grade"],
                "cv_test_fold": int(metadata["cv_test_fold"]),
                "prediction_distribution": "; ".join(
                    f"{name}:{count}" for name, count in counts.most_common()
                ),
                "majority_wrong_class": counts.most_common(1)[0][0],
                "majority_votes": counts.most_common(1)[0][1],
                "attention_status": "not generated; use CLAM-SB 10x checkpoint",
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_ROOT / "selected_case_summary.csv", index=False)
    return summary


def save_slide_b(case_summary):
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    grid = fig.add_gridspec(2, 5, height_ratios=[0.28, 0.72], hspace=0.18, wspace=0.10)

    kpis = [
        ("77.78%", "3,153 / 4,054\nwrong in >=1 configuration"),
        ("4.76%", "193 / 4,054\nwrong in all 14 configurations"),
        ("23.32%", "45 / 193\nsame wrong class in all 14"),
    ]
    kpi_ax = fig.add_subplot(grid[0, :])
    kpi_ax.axis("off")
    for x, (number, label) in zip([0.18, 0.50, 0.82], kpis):
        kpi_ax.text(x, 0.64, number, ha="center", va="center", fontsize=29, weight="bold", color="#173f5f")
        kpi_ax.text(x, 0.22, label, ha="center", va="center", fontsize=11, color="#3c4858")

    class_colors = {
        "HP -> IP": "#d95f59",
        "IP -> HP": "#4c78a8",
        "TAD -> TA": "#b05a9d",
        "TVA -> TA": "#59a14f",
        "SSL -> USA": "#f28e2b",
    }
    for index, case in case_summary.iterrows():
        ax = fig.add_subplot(grid[1, index])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        color = class_colors[case["pattern"]]
        ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, fc="white", ec="#c6ccd2", lw=1.2))
        ax.text(0.5, 0.935, case["pattern"], ha="center", va="center", fontsize=13, weight="bold", color=color)
        short_id = case["slide_id"] if len(case["slide_id"]) <= 20 else f"{case['slide_id'][:8]}...{case['slide_id'][-6:]}"
        ax.text(0.5, 0.885, short_id, ha="center", va="center", fontsize=7.2, color="#4d5966")

        ax.add_patch(Rectangle((0.07, 0.57), 0.40, 0.25, fc="#f3f5f7", ec="#9ba5ae", lw=1))
        ax.add_patch(Rectangle((0.53, 0.57), 0.40, 0.25, fc="#f3f5f7", ec="#9ba5ae", lw=1))
        ax.text(0.27, 0.695, "WSI\nthumbnail", ha="center", va="center", fontsize=8.5, color="#65717d")
        ax.text(0.73, 0.695, "10x attention\n+ top patches", ha="center", va="center", fontsize=8.2, color="#65717d")

        ax.text(0.07, 0.50, f"True: {case['true_class']}", fontsize=9.5, weight="bold")
        ax.text(0.07, 0.45, f"Votes: {case['prediction_distribution']}", fontsize=8.3)
        ax.text(0.07, 0.40, f"Source: {case['scanner_source']} | Fold {case['cv_test_fold'] + 1}", fontsize=8.3)

        reason = {
            "HP -> IP": "Reactive / inflammatory\nmorphology may dominate.",
            "IP -> HP": "Serrated / reactive glands\nmay resemble HP.",
            "TAD -> TA": "Focal high-grade region\nmay be diluted.",
            "TVA -> TA": "Villous component may be\nunder-sampled.",
            "SSL -> USA": "Overlapping serrated\ncrypt morphology.",
        }[case["pattern"]]
        ax.text(0.07, 0.30, "Review focus", fontsize=9, weight="bold", color=color)
        ax.text(0.07, 0.245, reason, fontsize=8.2, linespacing=1.25)
        ax.text(0.07, 0.105, "Review: label / mixed lesion /\nROI coverage", fontsize=7.4, color="#5f6b76", linespacing=1.2)

    fig.suptitle("Slide B | Cross-model consistent hard cases", fontsize=22, weight="bold", y=0.985)
    fig.savefig(OUT_ROOT / "slide_B_hard_case_layout.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_backup_slide():
    hard = pd.read_csv(ANALYSIS_ROOT / "hard_slides_cross_experiment.csv", dtype={"slide_id": str})
    unanimous = hard[hard["wrong_configurations"].eq(14)].copy()
    meta = pd.read_csv(META_PATH, dtype={"slide_id": str})[["slide_id", "scanner"]].drop_duplicates("slide_id")
    unanimous = unanimous.merge(meta, on="slide_id", how="left")

    class_counts = unanimous["true_class"].value_counts().reindex(PLOT_LABELS, fill_value=0)
    agreement_counts = unanimous["consensus_wrong_count"].value_counts().sort_index()
    source_total = meta["scanner"].value_counts()
    source_wrong = unanimous["scanner"].value_counts()
    source_rate = 100 * source_wrong / source_total

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.6), facecolor="white")
    axes[0].bar(class_counts.index, class_counts.values, color="#4c78a8")
    axes[0].set_title("True-class distribution of 193 slides")
    axes[0].set_ylabel("Slides")
    axes[0].tick_params(axis="x", rotation=45)
    for i, value in enumerate(class_counts.values):
        axes[0].text(i, value + 0.7, str(value), ha="center", fontsize=8)

    axes[1].bar(agreement_counts.index.astype(str), agreement_counts.values, color="#f28e2b")
    axes[1].set_title("Agreement on the wrong class")
    axes[1].set_xlabel("Votes for majority wrong class (out of 14)")
    axes[1].set_ylabel("Slides")
    for i, value in enumerate(agreement_counts.values):
        axes[1].text(i, value + 0.7, str(value), ha="center", fontsize=8)

    axes[2].bar(source_rate.index, source_rate.values, color=["#59a14f", "#e15759"])
    axes[2].set_title("All-14-wrong rate by source domain")
    axes[2].set_ylabel("Rate within source (%)")
    axes[2].set_ylim(0, max(source_rate.values) * 1.25)
    for i, source in enumerate(source_rate.index):
        axes[2].text(
            i,
            source_rate[source] + 0.12,
            f"{source_rate[source]:.2f}%\n({source_wrong[source]}/{source_total[source]})",
            ha="center",
            fontsize=10,
        )

    fig.suptitle("Backup | What characterizes the 193 universally difficult slides?", fontsize=20, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_ROOT / "backup_193_slide_audit.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_design_document(case_summary):
    case_lines = []
    for _, case in case_summary.iterrows():
        case_lines.append(
            f"| {case['pattern']} | `{case['slide_id']}` | {case['prediction_distribution']} | "
            f"{case['scanner_source']} | {case['wsi_status']} |"
        )

    text = f"""# 组会错误分析幻灯片设计

## Slide A：总体错误图谱

标题建议：`Where do the 11-class models fail?`

版式：左侧约60%放14个配置合并后的11类行归一化 confusion matrix；右上放5组主要双向混淆；右下放高级别类别向相邻类别退化的箭头图。

正文只保留这些信息：

- HP -> IP 12.82%，IP -> HP 33.92%。
- TA -> HP 6.92%，HP -> TA 7.77%。
- TA -> TVA 5.18%，TVA -> TA 20.58%。
- SSL -> HP 14.20%，HP -> SSL 6.70%。
- SSL -> USA 11.26%，USA -> SSL 17.98%。
- TAD -> TA 19.98%，TVAD -> TVA 22.57%，TSAD -> TVA 23.67%，TSAD -> TVAD 18.04%。

讲述重点：错误主要沿着病理连续谱发生，而不是均匀散布在11个类别之间；低级别形态、锯齿状谱系和局灶高级别区域是主要难点。

对应图片：`slide_A_overall_error_atlas.png`

## Slide B：跨模型一致困难样本

标题建议：`Hard cases persist across architectures and magnifications`

顶部只放三个大数字：

- 3,153 / 4,054 = 77.78%：至少被一个配置误判。
- 193 / 4,054 = 4.76%：被全部14个配置误判。
- 45 / 193 = 23.32%：14个配置还一致预测成同一个错误类别。

底部放5个病例卡片，每个卡片包含WSI thumbnail、CLAM-SB 10x attention、2–3个最高attention patch、真实标签、14配置预测票数和一句病理复核假设。不要在正文列完整CSV路径。

病例候选：

| Pattern | Slide ID | 14-config predictions | Source | WSI status |
| --- | --- | --- | --- | --- |
{chr(10).join(case_lines)}

对应版式图：`slide_B_hard_case_layout.png`

## Backup：193张困难切片审计

- 类别分布：HP 38、TA 33、TVA 24、TVAD 23、IP 16、TAD 13、SSL 12、SSLD 10、TSAD 9、TSA 9、USA 6。
- 45/193张的14个配置预测为同一个错误类别；其余148张虽然全部预测错误，但错误类别存在分歧。
- hp来源108/2446，全部配置均错率4.42%；yx来源85/1608，全部配置均错率5.29%。目前没有显示出压倒性的单一来源集中。
- 当前元数据只有hp/yx来源域，没有具体扫描仪型号和染色批次，不能回答硬件或染色批次富集问题。
- 标签可信度尚未经过本轮人工复核。建议由两名病理医师盲法复核，记录：标签确认、诊断争议、混合病变、诊断区域不足，并计算一致性。

对应图片：`backup_193_slide_audit.png`

## Attention制作建议

- 正文统一使用CLAM-SB 10x attention，因为10x在CLAM-SB中错误率最低，且attention含义一致。
- 每例先展示整张WSI低倍图，再叠加attention heatmap，并单独放大最高attention的3个patch。
- 对高级别病例额外检查高等级区域是否存在但未被高attention覆盖；对TA/TVA检查绒毛成分是否位于低attention区域。
- 当前五折结果未保存attention/heatmap，需要从对应fold checkpoint重新推理生成。
"""
    (OUT_ROOT / "GROUP_MEETING_SLIDE_DESIGN.md").write_text(text, encoding="utf-8")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_predictions = load_all_predictions()
    case_summary = selected_case_tables(all_predictions)
    save_slide_a(all_predictions)
    save_slide_a_pathway(all_predictions)
    save_slide_b(case_summary)
    save_backup_slide()
    write_design_document(case_summary)
    for path in sorted(OUT_ROOT.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
