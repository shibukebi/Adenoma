# 05. TransMIL 模块

## 模块目标

使用 Transformer 聚合 slide 内的 patch tokens，建模 patch 之间的全局关系。

## 实现位置

模型：

```text
transmil/model.py
```

| 位置 | 职责 |
| --- | --- |
| `transmil/model.py::TransLayer` | Transformer attention block |
| `transmil/model.py::PPEG` | 位置编码生成模块 |
| `transmil/model.py::TransMIL` | 聚合 patch token 并输出 slide-level logits |

训练：

```text
transmil/training.py
```

| 位置 | 职责 |
| --- | --- |
| `transmil/training.py::build_loader` | 构建训练/评估 DataLoader |
| `transmil/training.py::run_epoch` | 单 epoch 训练 |
| `transmil/training.py::summarize_split` | 验证/测试 split 推理与指标汇总 |
| `transmil/training.py::run_transmil_training` | 完整训练入口 |

CLI：

| 脚本 | 说明 |
| --- | --- |
| `scripts/train_transmil_ssl.py` | TransMIL 训练 CLI |
| `scripts/run_transmil_ssl_1x.sh` | 1X-style 路线 |
| `scripts/run_transmil_ssl_2p5x.sh` | 2.5X 路线 |
| `scripts/run_transmil_ssl_5x.sh` | 5X 路线 |
| `scripts/run_transmil_ssl_20x.sh` | 20X 路线 |

`transmil_official/` 下保留 official-style TransMIL 复现路线，入口为 `transmil_official/train.py` 和 `scripts/run_transmil_official_*.sh`。

## 结果目录

| 路线 | 结果根目录 | fold 结果目录 |
| --- | --- | --- |
| `TransMIL 1X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_1x_uni` | `fold-<fold>/` |
| `TransMIL 2.5X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_2p5x_uni` | `fold-<fold>/` |
| `TransMIL 5X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_5x_uni` | `fold-<fold>/` |
| `TransMIL 20X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/transmil_ssl_vs_others_20x_uni` | `fold-<fold>/` |
| `Hierarchical TransMIL 20X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/hierarchical_transmil_20x_uni` | `fold-<fold>/stage1_*`、`fold-<fold>/stage2_*`、合并预测与三分类指标 |
| `Official-style TransMIL` | `transmil_official/configs/adenoma_ssl_*_fold5.yaml` 中定义 | 由对应 yaml 和 `scripts/run_transmil_official_*.sh` 决定 |

## 方法原理

TransMIL 将 patch features 视作 token 序列，用 Transformer attention 建模 token 间的全局依赖。普通 Transformer 对二维空间不敏感，因此项目中的 `PPEG` 用卷积式位置编码向 token 注入空间位置信息。

与 CLAM 相比，CLAM 更强调 attention pooling 和 instance selection；TransMIL 更强调 patch token 间的全局交互。
