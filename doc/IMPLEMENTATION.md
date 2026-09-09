# adenoma 项目实现总览

本文档是实现文档入口。详细实现已经按项目模块拆到 `doc/modules/` 下，每个模块文档都包含：

- 模块目标
- 入口脚本
- 关键函数 / 类
- 输入输出和结果目录
- 与方法原理的对应关系

## 1. 总数据流

```text
WSI
  -> CLAM tissue segmentation / coords h5
  -> UNI features h5 / pt
  -> label / split / ready csv
  -> CLAM / TransMIL / PatchGCN / DSMIL / hierarchical training
  -> metrics / predictions / heatmaps / reports
```

## 2. 模块文档入口

| 模块 | 文档 | 主要内容 |
| --- | --- | --- |
| 项目结构与数据流 | `modules/00_project_overview.md` | 项目目录、外部依赖、正式结果口径 |
| WSI 预处理 | `modules/01_preprocessing.md` | CLAM 分割、patch 坐标、预处理结果目录 |
| UNI 特征提取 | `modules/02_uni_feature_extraction.md` | 物理视野、UNI encoder、h5/pt 特征目录 |
| 标签、split 与指标 | `modules/03_labels_splits_metrics.md` | SSL/dysplasia 标签、fold split、metrics |
| CLAM / ABMIL | `modules/04_clam_abmil.md` | CLAM-SB 训练、checkpoint、predictions、heatmap 入口 |
| TransMIL | `modules/05_transmil.md` | Transformer MIL、自实现和 official-style 路线 |
| PatchGCN | `modules/06_patch_gcn.md` | KNN graph cache、图卷积、PatchGCN 训练 |
| DSMIL | `modules/07_dsmil.md` | 双倍率双流 DSMIL、结果目录 |
| 层级分类 | `modules/08_hierarchical_classification.md` | Stage 1/Stage 2 和三分类合并 |
| 可解释性与报告 | `modules/09_interpretability_reports.md` | CLAM heatmap、TransMIL gradient heatmap、ROC/PR/report |
| Route C / Patho-R1 | `modules/10_route_c_patho_r1.md` | VLM 选区、box 转 h5、保留路线 |

## 3. 当前正式实现口径

当前正式汇报优先使用：

- 数据：`Adenoma_yx`
- 特征：`UNI`
- 划分：`fold-5`
- 主任务：`SSL vs others`
- 模型：`CLAM-SB`、`TransMIL`、`PatchGCN`、`DSMIL`
- 扩展：`dysplasia vs no_dysplasia` 二阶段任务作为探索性结果

旧 `fold-0` 结果和图表仍可作为历史探索参考，但不作为当前正式主口径。

## 4. 历史来源文档

原始说明和阶段性汇总保留在：

- `doc/1/PREPROCESSING.md`
- `doc/1/UNI_FEATURE_EXTRACTION_SPEC.md`
- `doc/1/CLAM_MULTISCALE_METHOD_SUMMARY.md`
- `doc/1/BASELINE_TRAINING_SUMMARY_20260512.md`
- `doc/1/MAG_WORKFLOW_PLAN.md`
- `doc/1/codex.md`
