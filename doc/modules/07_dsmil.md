# 07. DSMIL 模块

## 模块目标

用双倍率双流模型融合不同尺度的 UNI features，验证多尺度互补是否优于单一倍率。

## 实现位置

数据：

```text
dsmil/dataset.py
```

| 位置 | 职责 |
| --- | --- |
| `dataset.py::load_feature_tensor` | 读取单个 slide 的 `.pt` 特征 |
| `dataset.py::DualStreamBagDataset` | 同时读取两个倍率的 bag 特征和标签 |
| `dataset.py::build_dual_stream_split` | 构建双流 train/val/test split |

模型：

```text
dsmil/model.py
```

| 位置 | 职责 |
| --- | --- |
| `model.py::BagClassifier` | 单流 attention bag classifier |
| `model.py::SingleStreamDSMIL` | 单倍率 DSMIL 分支 |
| `model.py::DualStreamDSMIL` | 双倍率分支融合并输出 slide-level logits |

训练：

```text
dsmil/training.py
```

| 位置 | 职责 |
| --- | --- |
| `training.py::compute_dsmil_loss` | 组合 bag logits 和 stream logits 的训练损失 |
| `training.py::run_epoch` | 单 epoch 训练 |
| `training.py::summarize_split` | split 评估 |
| `training.py::run_dsmil_training` | 完整 DSMIL 训练入口 |

CLI：

| 脚本 | 说明 |
| --- | --- |
| `scripts/prepare_dsmil_ssl_data.py` | 准备双流 ready csv 与 split |
| `scripts/train_dsmil_ssl.py` | DSMIL 训练 CLI |
| `scripts/run_dsmil_ssl_2p5x_5x.sh` | 2.5X + 5X |
| `scripts/run_dsmil_ssl_5x_20x.sh` | 5X + 20X |
| `scripts/run_dsmil_ssl_2p5x_20x.sh` | 2.5X + 20X |

## 结果目录

| 路线 | feature stream A | feature stream B | 训练结果根目录 |
| --- | --- | --- | --- |
| `DSMIL 2.5X + 5X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_2p5x_uni` | `/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_5x_uni` | `/data15/data15_5/yuexin2/adenoma/outputs/dsmil_ssl_vs_others_2p5x_5x_uni` |
| `DSMIL 5X + 20X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_5x_uni` | `/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_uni` | `/data15/data15_5/yuexin2/adenoma/outputs/dsmil_ssl_vs_others_5x_20x_uni` |
| `DSMIL 2.5X + 20X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_2p5x_uni` | `/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_uni` | `/data15/data15_5/yuexin2/adenoma/outputs/dsmil_ssl_vs_others_2p5x_20x_uni` |
| `DSMIL fold-5 外置汇报结果` | `/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/yx/...` | `/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/yx/...` | `/data15/zhengke_usb2/yuexin_data/training/fold_5/DS-MIL/` |
| `Hierarchical DSMIL 5X + 20X UNI` | 5X stream | 20X stream | `/data15/data15_5/yuexin2/adenoma/outputs/hierarchical_dsmil_5x_20x_uni` |

## 方法原理

双流设计的直觉：

- 低倍 stream 提供大范围结构上下文。
- 高倍 stream 提供局部细节。
- 融合后模型可以同时利用结构和细节信息。

当前配置包括 `2.5x + 5x`、`5x + 20x`、`2.5x + 20x`，用于验证多倍率互补价值。
