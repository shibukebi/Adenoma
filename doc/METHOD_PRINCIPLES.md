# adenoma 项目原理总览

本文档是方法原理入口。每个项目模块的具体原理已经和对应实现合并到 `doc/modules/` 下，便于在同一处查看“为什么这样做”和“代码在哪里”。

## 1. 方法主线

当前项目采用计算病理中常见的 slide-level MIL 流水线：

```text
WSI
  -> tissue segmentation
  -> patch coordinate extraction
  -> pathology foundation encoder features
  -> slide-level aggregation / multi-scale modeling
  -> evaluation and interpretation
```

核心思想是：WSI 太大，不能直接整体输入模型；项目先抽取组织 patch，再用 UNI 编码 patch，最后用 MIL、Transformer、Graph 或双流模型聚合为 slide-level 预测。

## 2. 模块原理入口

| 原理主题 | 模块文档 | 要点 |
| --- | --- | --- |
| WSI 与 patch | `modules/01_preprocessing.md` | level 0 坐标、组织分割、低倍/高倍物理视野 |
| UNI 特征 | `modules/02_uni_feature_extraction.md` | `physical_level_0_extent`、`target_patch_size`、不依赖 legacy `patch_level` |
| 标签和评估 | `modules/03_labels_splits_metrics.md` | `SSL vs others`、`dysplasia vs no_dysplasia`、AUC/F1/混淆矩阵 |
| MIL / CLAM | `modules/04_clam_abmil.md` | bag/instance、attention pooling、CLAM-SB instance clustering |
| TransMIL | `modules/05_transmil.md` | patch token、Transformer 全局交互、`PPEG` 位置编码 |
| PatchGCN | `modules/06_patch_gcn.md` | patch 坐标 KNN 图、邻域消息传递、空间结构建模 |
| DSMIL | `modules/07_dsmil.md` | 双倍率双流融合、多尺度互补 |
| 层级分类 | `modules/08_hierarchical_classification.md` | 先识别 SSL，再在 SSL 子集识别 dysplasia |
| 可解释性 | `modules/09_interpretability_reports.md` | attention heatmap、gradient heatmap、误差分析 |
| Route C | `modules/10_route_c_patho_r1.md` | VLM ROI selection、thumbnail box 到 level 0 coords |

## 3. 当前方法口径

最稳妥的汇报方式是：

- 先说明 `SSL vs others` 是 slide-level MIL 主任务。
- 再说明 WSI 被分解为 patch，patch 经 UNI 编码为 feature。
- 然后比较 CLAM-SB、TransMIL、PatchGCN、DSMIL 在不同倍率下的表现。
- 最后把 dysplasia 二阶段作为探索性扩展，并明确阳性样本少带来的不确定性。

旧 `fold-0` 图表可作为历史探索结果；当前正式主口径应统一到 `fold-5`、`UNI` 特征和已完成的多模型 baseline。

## 4. 倍率与 h5 生成口径

已确认 `Adenoma_hp` 和 `Adenoma_yx` 两组 WSI 的扫描基准倍率均为 40x。因此项目中所有 1x、2.5x、5x、20x 的 patch h5 必须严格按目标物理倍率生成，而不是按 WSI pyramid 的 level index 或临时 downsample 近似采样。

统一换算公式：

```text
scale_to_level0 = base_magnification / target_magnification
physical_level_0_extent = target_patch_size * scale_to_level0
```

其中 `base_magnification = 40x`。如果目标输入 patch 为 `256 x 256`，对应关系为：

| 目标倍率 | level 0 缩放倍数 | level 0 物理范围 |
| --- | ---: | ---: |
| 20x | 2 | 512 x 512 |
| 5x | 8 | 2048 x 2048 |
| 2.5x | 16 | 4096 x 4096 |
| 1x | 40 | 10240 x 10240 |

如果某一路线使用不同的 `target_patch_size`，例如 `320 x 320`，则必须用同一公式重新计算并写入 h5 元数据。h5 中至少应保留 `base_magnification`、`target_magnification`、`target_patch_size` 和 `physical_level_0_extent`，用于后续 UNI 特征、MIL 图构建和可解释性热图复现。
