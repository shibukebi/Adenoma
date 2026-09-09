#!/usr/bin/env python3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


NEW_ROOT = Path("/data15/zhengke_usb2/yuexin_data/result/5fold_11class")
OLD_ROOT = Path("/data15/zhengke_usb2/yuexin_data/result/fold5_hp+yx")
OUT_ROOT = NEW_ROOT / "summary_11class_5fold" / "confusion_error_analysis"

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

EXPERIMENTS = []
for model, old_model in [
    ("CLAM-SB", "CLAM-SB"),
    ("TransMIL", "transmil"),
    ("DSMIL", "dsmil"),
]:
    for feature in ["2p5x", "5x", "10x", "20x"]:
        EXPERIMENTS.append((model, old_model, feature))
for feature in ["2p5x_5x", "5x_10x"]:
    EXPERIMENTS.append(("MIST", "MIST", feature))


def prediction_paths(model, old_model, feature):
    if model == "MIST":
        return [
            (fold + 1, NEW_ROOT / model / "11class" / feature / f"fold-{fold}" / "predictions.csv")
            for fold in range(5)
        ]

    paths = [
        (fold + 1, NEW_ROOT / model / "11class" / feature / f"fold-{fold}" / "predictions.csv")
        for fold in range(4)
    ]
    paths.append((5, OLD_ROOT / old_model / "11class" / feature / "predictions.csv"))
    return paths


def load_experiment(model, old_model, feature):
    frames = []
    for cv_fold, path in prediction_paths(model, old_model, feature):
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame = frame[frame["split"].astype(str).str.lower().eq("test")].copy()
        frame["cv_fold"] = cv_fold
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    if df["slide_id"].duplicated().any():
        duplicated = df.loc[df["slide_id"].duplicated(), "slide_id"].head().tolist()
        raise ValueError(f"Duplicate slide IDs in {model} {feature}: {duplicated}")

    df["label"] = df["label"].astype(int)
    df["pred"] = df["pred"].astype(int)
    df["true_class"] = df["label"].map(CLASS_SHORT)
    df["pred_class"] = df["pred"].map(CLASS_SHORT)
    df["is_correct"] = df["label"].eq(df["pred"])
    df["model"] = model
    df["feature"] = feature
    df["configuration"] = model + " " + feature
    return df


def ranked_pairs(df):
    errors = df[~df["is_correct"]].copy()
    rows = []
    for (label, pred), group in errors.groupby(["label", "pred"], sort=False):
        true_total = int(df["label"].eq(label).sum())
        rows.append(
            {
                "model": df["model"].iat[0],
                "feature": df["feature"].iat[0],
                "configuration": df["configuration"].iat[0],
                "true_class": CLASS_SHORT[label],
                "predicted_class": CLASS_SHORT[pred],
                "confusion": f"{CLASS_SHORT[label]} -> {CLASS_SHORT[pred]}",
                "count": int(len(group)),
                "true_class_total": true_total,
                "within_true_class_pct": 100.0 * len(group) / true_total,
                "share_of_all_errors_pct": 100.0 * len(group) / len(errors),
                "mean_confidence": float(group["confidence"].mean()),
                "slide_ids": "|".join(sorted(group["slide_id"].astype(str))),
            }
        )
    pairs = pd.DataFrame(rows)
    pairs["rank_by_count"] = pairs["count"].rank(method="first", ascending=False).astype(int)
    pairs["rank_by_class_pct"] = pairs["within_true_class_pct"].rank(
        method="first", ascending=False
    ).astype(int)
    return pairs.sort_values(["rank_by_count", "rank_by_class_pct"])


def class_error_rows(df):
    rows = []
    for label, group in df.groupby("label"):
        errors = group[~group["is_correct"]]
        if errors.empty:
            common_pred = "NA"
            common_count = 0
        else:
            common_pred_id, common_count = Counter(errors["pred"].astype(int)).most_common(1)[0]
            common_pred = CLASS_SHORT[common_pred_id]
        rows.append(
            {
                "model": df["model"].iat[0],
                "feature": df["feature"].iat[0],
                "configuration": df["configuration"].iat[0],
                "true_class": CLASS_SHORT[int(label)],
                "n_test": int(len(group)),
                "n_correct": int(group["is_correct"].sum()),
                "n_error": int((~group["is_correct"]).sum()),
                "recall_pct": 100.0 * group["is_correct"].mean(),
                "error_pct": 100.0 * (~group["is_correct"]).mean(),
                "most_common_wrong_class": common_pred,
                "most_common_wrong_count": int(common_count),
            }
        )
    return rows


def format_predictions(counter):
    return "; ".join(f"{name}:{count}" for name, count in counter.most_common())


def markdown_table(df):
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.select_dtypes(include=["float"]).columns:
        display[col] = display[col].map(lambda value: f"{value:.2f}")
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [header, divider]
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(rows)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    experiment_frames = []
    pair_frames = []
    class_rows = []
    for model, old_model, feature in EXPERIMENTS:
        df = load_experiment(model, old_model, feature)
        experiment_frames.append(df)
        pair_frames.append(ranked_pairs(df))
        class_rows.extend(class_error_rows(df))

    all_predictions = pd.concat(experiment_frames, ignore_index=True)
    all_pairs = pd.concat(pair_frames, ignore_index=True)
    class_errors = pd.DataFrame(class_rows)
    errors = all_predictions[~all_predictions["is_correct"]].copy()

    error_columns = [
        "slide_id",
        "cv_fold",
        "model",
        "feature",
        "configuration",
        "label",
        "true_class",
        "pred",
        "pred_class",
        "confidence",
        "type",
        "grade",
    ]
    errors[error_columns].sort_values(
        ["model", "feature", "true_class", "pred_class", "confidence"],
        ascending=[True, True, True, True, False],
    ).to_csv(OUT_ROOT / "misclassified_slides_by_experiment.csv", index=False)
    all_pairs.to_csv(OUT_ROOT / "confusion_pairs_by_experiment.csv", index=False)
    class_errors.to_csv(OUT_ROOT / "class_error_summary_by_experiment.csv", index=False)

    overall_rows = []
    for (label, pred), group in errors.groupby(["label", "pred"]):
        true_instances = int(all_predictions["label"].eq(label).sum())
        overall_rows.append(
            {
                "true_class": CLASS_SHORT[int(label)],
                "predicted_class": CLASS_SHORT[int(pred)],
                "confusion": f"{CLASS_SHORT[int(label)]} -> {CLASS_SHORT[int(pred)]}",
                "error_instances": int(len(group)),
                "within_true_class_pct": 100.0 * len(group) / true_instances,
                "configurations_affected": int(group["configuration"].nunique()),
                "unique_slides": int(group["slide_id"].nunique()),
                "slide_ids": "|".join(sorted(group["slide_id"].astype(str).unique())),
            }
        )
    overall_pairs = pd.DataFrame(overall_rows).sort_values(
        ["error_instances", "within_true_class_pct"], ascending=False
    )
    overall_pairs.to_csv(OUT_ROOT / "overall_confusion_pairs.csv", index=False)

    hard_rows = []
    n_configurations = len(EXPERIMENTS)
    for slide_id, group in all_predictions.groupby("slide_id", sort=False):
        wrong = group[~group["is_correct"]]
        if wrong.empty:
            continue
        true_classes = group["true_class"].unique()
        if len(true_classes) != 1:
            raise ValueError(f"Inconsistent labels for slide {slide_id}: {true_classes}")
        wrong_counter = Counter(wrong["pred_class"])
        all_counter = Counter(group["pred_class"])
        first = group.iloc[0]
        model_counts = wrong.groupby("model").size().to_dict()
        hard_rows.append(
            {
                "slide_id": slide_id,
                "true_class": true_classes[0],
                "type": first.get("type", ""),
                "grade": first.get("grade", ""),
                "cv_fold": int(first["cv_fold"]),
                "wrong_configurations": int(len(wrong)),
                "total_configurations": n_configurations,
                "wrong_pct": 100.0 * len(wrong) / n_configurations,
                "models_wrong": int(wrong["model"].nunique()),
                "CLAM-SB_wrong": int(model_counts.get("CLAM-SB", 0)),
                "TransMIL_wrong": int(model_counts.get("TransMIL", 0)),
                "DSMIL_wrong": int(model_counts.get("DSMIL", 0)),
                "MIST_wrong": int(model_counts.get("MIST", 0)),
                "consensus_wrong_class": wrong_counter.most_common(1)[0][0],
                "consensus_wrong_count": int(wrong_counter.most_common(1)[0][1]),
                "wrong_prediction_summary": format_predictions(wrong_counter),
                "all_prediction_summary": format_predictions(all_counter),
                "mean_wrong_confidence": float(wrong["confidence"].mean()),
                "max_wrong_confidence": float(wrong["confidence"].max()),
                "wrong_configuration_names": "|".join(wrong["configuration"].tolist()),
            }
        )
    hard_slides = pd.DataFrame(hard_rows).sort_values(
        ["wrong_configurations", "models_wrong", "consensus_wrong_count", "mean_wrong_confidence"],
        ascending=False,
    )
    hard_slides.insert(0, "hardness_rank", np.arange(1, len(hard_slides) + 1))
    hard_slides.to_csv(OUT_ROOT / "hard_slides_cross_experiment.csv", index=False)
    unanimous_wrong = hard_slides[hard_slides["wrong_configurations"].eq(n_configurations)].copy()
    unanimous_wrong.to_csv(OUT_ROOT / "unanimously_wrong_slides.csv", index=False)

    config_summary = []
    for df in experiment_frames:
        n_error = int((~df["is_correct"]).sum())
        config_summary.append(
            {
                "Model": df["model"].iat[0],
                "Feature": df["feature"].iat[0],
                "Test slides": int(len(df)),
                "Errors": n_error,
                "Error rate (%)": 100.0 * n_error / len(df),
            }
        )
    config_summary = pd.DataFrame(config_summary).sort_values("Error rate (%)")

    lines = [
        "# 11分类不同模型与倍率混淆问题及困难切片分析\n",
        "## 分析范围\n",
        f"- 共纳入14个模型/倍率配置，以及{all_predictions['slide_id'].nunique()}张互不重复的五折 pooled-test 切片。",
        "- 混淆方向采用 `真实类别 -> 预测类别` 表示。",
        "- 类内混淆比例以该真实类别的测试切片数作为分母。",
        "- 完整 slide ID 位于本报告同目录的 CSV 文件中。\n",
        "## 核心结论\n",
        "- CLAM-SB和DSMIL均在10x达到最低错误率，分别为30.69%和30.54%；单倍率实验中10x整体最稳定。",
        "- TransMIL四个倍率的错误率均高于CLAM-SB和DSMIL，其中2.5x最高，为39.07%。",
        "- 最突出的双向混淆是HP与IP：累计HP -> IP为2051次，IP -> HP为1491次；IP被误判为HP的类内比例达到33.92%。",
        "- 腺瘤谱系中TA与HP、TA与TVA混淆明显；锯齿状谱系中SSL、USA和HP之间存在持续混淆。",
        "- 高级别异型增生相关类别常被预测为对应低级别类别或相邻高级别类别，例如TVAD -> TVA、TAD -> TA、TSAD -> TVA/TVAD。",
        "- MIST两组融合特征均表现出明显的IP -> HP偏置，类内误分比例为52.87%和51.59%；同时TVA -> TA达到35.80%和41.12%。",
        f"- 共{len(hard_slides)}张切片至少被一个配置误判，其中{len(unanimous_wrong)}张在全部14个配置中均被误判，应优先复核标签、混合成分和诊断区域覆盖情况。\n",
        "## 可能的病理与建模原因\n",
        "- HP与IP可能共享炎症、修复和腺体结构信号，模型容易把局部反应性形态当作类别主特征。",
        "- TA与TVA的区分依赖绒毛成分比例；当诊断性区域较小或patch采样覆盖不足时，容易发生双向误判。",
        "- SSL、USA与HP均属于锯齿状形态相关病变，有限视野下基底扩张、隐窝形态和成熟模式可能未被完整捕获。",
        "- 高级别异型增生常呈局灶分布，MIL聚合可能被大量低级别patch稀释；该解释属于基于结果模式的推断，建议结合困难切片复核。\n",
        "## 各模型与倍率总体错误率\n",
        markdown_table(config_summary),
        "\n## 跨全部配置最常见的混淆方向\n",
        markdown_table(
            overall_pairs.head(15)[
                [
                    "confusion",
                    "error_instances",
                    "within_true_class_pct",
                    "configurations_affected",
                    "unique_slides",
                ]
            ].rename(
                columns={
                    "confusion": "Confusion",
                    "error_instances": "Error instances",
                    "within_true_class_pct": "Within true class (%)",
                    "configurations_affected": "Configurations affected",
                    "unique_slides": "Unique slides",
                }
            )
        ),
        "\n## 每个模型与倍率最常见的混淆方向\n",
    ]

    for model, _, feature in EXPERIMENTS:
        config = f"{model} {feature}"
        config_pairs = all_pairs[all_pairs["configuration"].eq(config)]
        top_count = config_pairs.nsmallest(3, "rank_by_count").copy()
        top_count["Slide IDs (first 10)"] = top_count["slide_ids"].map(
            lambda value: ", ".join(value.split("|")[:10])
        )
        table = top_count[
            ["confusion", "count", "within_true_class_pct", "share_of_all_errors_pct", "Slide IDs (first 10)"]
        ].rename(
            columns={
                "confusion": "Confusion",
                "count": "Count",
                "within_true_class_pct": "Within true class (%)",
                "share_of_all_errors_pct": "Share of errors (%)",
            }
        )
        lines.extend([f"\n### {config}\n", markdown_table(table)])

    hard_display = hard_slides.head(30)[
        [
            "hardness_rank",
            "slide_id",
            "true_class",
            "wrong_configurations",
            "models_wrong",
            "consensus_wrong_class",
            "consensus_wrong_count",
            "wrong_prediction_summary",
        ]
    ].rename(
        columns={
            "hardness_rank": "Rank",
            "slide_id": "Slide ID",
            "true_class": "True class",
            "wrong_configurations": "Wrong / 14",
            "models_wrong": "Models wrong / 4",
            "consensus_wrong_class": "Main wrong class",
            "consensus_wrong_count": "Main wrong count",
            "wrong_prediction_summary": "Wrong predictions",
        }
    )
    lines.extend(
        [
            "\n## 跨模型与倍率最困难的切片\n",
            "下表列出被最多配置误判的前30张切片；完整列表和全部14个配置均错的切片见CSV。\n",
            markdown_table(hard_display),
            "\n## 输出文件\n",
            "- `confusion_pairs_by_experiment.csv`: every directional confusion pair with all corresponding slide IDs.",
            "- `class_error_summary_by_experiment.csv`: recall/error rate and main wrong class for every true class.",
            "- `misclassified_slides_by_experiment.csv`: one row per wrong prediction, including slide ID and confidence.",
            "- `overall_confusion_pairs.csv`: confusion pairs aggregated across all 14 configurations.",
            "- `hard_slides_cross_experiment.csv`: cross-configuration difficult-slide ranking with all slide IDs.",
            "- `unanimously_wrong_slides.csv`: slides misclassified by all 14 configurations.",
        ]
    )

    report_path = OUT_ROOT / "CONFUSION_ERROR_ANALYSIS_11CLASS.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")
    for path in sorted(OUT_ROOT.glob("*.csv")):
        print(f"Wrote {path}")
    print(f"Hard slides: {len(hard_slides)}")
    print(f"Unanimously wrong: {len(unanimous_wrong)}")


if __name__ == "__main__":
    main()
