# 09. 可解释性、误差分析与报告模块

## 模块目标

将 slide-level 预测回溯到 patch / WSI 区域，并生成 ROC、PR、混淆矩阵、训练耗时和 markdown 报告。

## CLAM heatmap

| 脚本 | 作用 |
| --- | --- |
| `scripts/select_clam_heatmap_samples.py` | 按分数/排序选择代表性样本 |
| `scripts/generate_clam_heatmaps.py` | 根据 CLAM checkpoint 和 heatmap template 生成热图 |
| `scripts/run_clam_sb_error_heatmaps.sh` | 对错误样本生成 CLAM-SB 热图 |

## TransMIL gradient heatmap

| 脚本 | 作用 |
| --- | --- |
| `scripts/select_error_heatmap_samples.py` | 选择误分类样本 |
| `scripts/generate_transmil_gradient_heatmaps.py` | 计算 patch gradient score 并渲染热图 |
| `scripts/run_transmil_error_heatmaps.sh` | TransMIL 错误样本热图入口 |

## 指标图表与报告

| 脚本 | 作用 |
| --- | --- |
| `scripts/build_combined_roc_comparison.py` | 构建组合 ROC 对比 |
| `scripts/build_clamsb_multiscale_roc_comparison.py` | 构建 CLAM-SB 多倍率 ROC |
| `scripts/build_clamsb_auc_bar_chart.py` | 构建 AUC 柱状图 |
| `scripts/build_fold5_pr_comparisons.py` | 构建 fold-5 PR 曲线对比 |
| `scripts/build_fold5_confusion_matrices.py` | 构建 fold-5 混淆矩阵 |
| `scripts/build_clam_roc_time_size_package.py` | 汇总 ROC、耗时和文件大小 |
| `scripts/build_clam_ssl_report.py` | 生成 markdown 实验报告 |

## 结果目录

| 分析类型 | 结果目录 / 文件 |
| --- | --- |
| `CLAM-SB 1X heatmaps` | `/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_1x_uni_clam_sb/fold-5/heatmaps` |
| `CLAM-SB 1X report` | `/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_1x_uni_clam_sb/fold-5/analysis/clam_ssl_report.md` |
| `CLAM-SB 2.5X heatmaps` | `/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_2p5x_uni_clam_sb/fold-0/heatmaps`；正式 `fold-5` 分析可放在同根目录的 `fold-5/heatmaps` |
| `CLAM-SB 5X heatmaps` | `/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_5x_uni_clam_sb/fold-0/heatmaps`；正式 `fold-5` 分析可放在同根目录的 `fold-5/heatmaps` |
| `CLAM-SB 20X heatmaps` | `/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_20x_uni_clam_sb/fold-0/heatmaps`；正式 `fold-5` 分析可放在同根目录的 `fold-5/heatmaps` |
| ROC / PR / AUC 对比图 | `/data15/data15_5/yuexin2/adenoma/outputs/roc_comparisons/` |
| TransMIL error heatmaps | 通常写入对应 `transmil_ssl_vs_others_*_uni/fold-<fold>/heatmaps` 或脚本参数指定目录 |

## 方法原理

WSI 分类模型需要可解释性来判断模型是否关注合理病理区域。CLAM heatmap 使用 attention 或模型分数投影 patch；TransMIL gradient heatmap 用 gradient score 估计 patch 对预测的影响。

误差分析通常按 TP/FP/FN/TN、confidence 和 probability 组织样本，辅助判断 false positive / false negative 是否来自组织质量、取样不足、形态相似或模型偏差。
