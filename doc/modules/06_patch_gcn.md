# 06. PatchGCN 模块

## 模块目标

在 UNI patch features 之外显式使用 patch 坐标构图，让模型利用空间邻接关系。

## 实现位置

图构建：

```text
patch_gcn/graph_utils.py
```

| 位置 | 职责 |
| --- | --- |
| `graph_utils.py::build_knn_graph` | 根据 patch 坐标构建 KNN 空间邻接图 |
| `graph_utils.py::ensure_graph_cache` | 检查或生成 slide 级 graph cache |

数据集：

```text
patch_gcn/dataset.py
```

| 位置 | 职责 |
| --- | --- |
| `dataset.py::PatchGCNDataset` | 读取 slide 特征、标签和 graph cache |
| `dataset.py::build_patch_gcn_split` | 构建 train/val/test split dataset |

模型：

```text
patch_gcn/model.py
```

| 位置 | 职责 |
| --- | --- |
| `model.py::SpatialGraphConv` | 基于邻接关系聚合邻居 patch 表征 |
| `model.py::PatchGCN` | 图卷积后接 attention pooling，输出 slide-level logits |

训练：

```text
patch_gcn/training.py
```

| 位置 | 职责 |
| --- | --- |
| `training.py::run_epoch` | 单 epoch 训练 |
| `training.py::summarize_split` | split 评估 |
| `training.py::run_patch_gcn_training` | 完整 PatchGCN 训练入口 |

CLI：

| 脚本 | 说明 |
| --- | --- |
| `scripts/prepare_patch_gcn_ssl_data.py` | 准备数据与 graph cache |
| `scripts/train_patch_gcn_ssl.py` | PatchGCN 训练 CLI |
| `scripts/run_patch_gcn_ssl_*.sh` | 各倍率启动脚本 |

## 结果目录

| 路线 | graph cache 目录 | 训练结果根目录 |
| --- | --- | --- |
| `PatchGCN 1X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_patch_gcn_graphs_1x_uni_k8` | `/data15/data15_5/yuexin2/adenoma/outputs/patch_gcn_ssl_vs_others_1x_uni` |
| `PatchGCN 2.5X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_patch_gcn_graphs_2p5x_uni_k8` | `/data15/data15_5/yuexin2/adenoma/outputs/patch_gcn_ssl_vs_others_2p5x_uni` |
| `PatchGCN 5X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_patch_gcn_graphs_5x_uni_k8` | `/data15/data15_5/yuexin2/adenoma/outputs/patch_gcn_ssl_vs_others_5x_uni` |
| `PatchGCN 20X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_patch_gcn_graphs_uni_k8` | `/data15/data15_5/yuexin2/adenoma/outputs/patch_gcn_ssl_vs_others_20x_uni` |
| `Hierarchical PatchGCN 20X UNI` | 复用 20X graph cache | `/data15/data15_5/yuexin2/adenoma/outputs/hierarchical_patch_gcn_20x_uni` |

## 方法原理

PatchGCN 将 patch 作为图节点，用坐标构建 KNN 边：

```text
patch coords -> KNN graph -> graph convolution -> attention pooling -> slide logits
```

相邻 patch 往往来自连续组织区域。图卷积让 patch 表征融合邻域信息，有助于建模腺体排列、隐窝形态和病灶边界等空间结构。
