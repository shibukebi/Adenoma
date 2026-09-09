# 04. CLAM / ABMIL 模块

## 模块目标

使用 UNI patch features 做 slide-level MIL 分类，当前正式 CLAM 主线是 `UNI + CLAM-SB + fold-5`。

## 实现入口

主训练脚本：

```text
scripts/train_clam_ssl_20x.py
```

包装脚本：

| 脚本 | 说明 |
| --- | --- |
| `scripts/train_clam_ssl_1x.py` | 1X-style 包装入口 |
| `scripts/train_clam_ssl_2p5x.py` | 2.5X 包装入口 |
| `scripts/train_clam_ssl_5x.py` | 5X 包装入口 |
| `scripts/run_clam_ssl_*_clam_sb.sh` | 各倍率正式 CLAM-SB 启动脚本 |
| `scripts/run_hierarchical_clam_20x.sh` | 层级 CLAM 路线入口 |

## 关键函数

| 位置 | 职责 |
| --- | --- |
| `train_clam_ssl_20x.py::build_split` | 根据 split id 从 CLAM `slide_data` 构造 `Generic_Split` |
| `train_clam_ssl_20x.py::evaluate_checkpoint` | 加载 checkpoint，对 val/test 重新推理并构造 predictions csv |
| `train_clam_ssl_20x.py::main` | 读取参数、检查 split、创建 dataset、调用 CLAM `train`、保存结果 |

外部 CLAM 依赖：

| 外部位置 | 用途 |
| --- | --- |
| `dataset_modules.dataset_generic::Generic_MIL_Dataset` | 读取 slide-level bag 数据 |
| `dataset_modules.dataset_generic::Generic_Split` | 训练/验证/测试 split 封装 |
| `utils.core_utils::train` | CLAM/MIL/ABMIL 训练主循环 |
| `utils.core_utils::summary` | split 级推理汇总 |
| `utils.eval_utils::initiate_model` | checkpoint 加载和模型初始化 |

## 输出文件

| 输出 | 说明 |
| --- | --- |
| `experiment_config.json` | 实验配置快照 |
| `s_<fold>_checkpoint.pt` | CLAM checkpoint |
| `summary.csv` | CLAM 原始训练摘要 |
| `predictions.csv` | test predictions |
| `val_predictions.csv` | val predictions |
| `metrics.json` | 标准化指标 |
| `confusion_matrix.csv` | 混淆矩阵 |

## 结果目录

| 路线 | 配置中的结果根目录 | fold 结果目录 |
| --- | --- | --- |
| `CLAM-SB 1X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_1x_uni_clam_sb` | `fold-5/` |
| `CLAM-SB 2.5X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_2p5x_uni_clam_sb` | `fold-5/` 为当前正式结果；部分分析配置仍保留 `fold-0/` |
| `CLAM-SB 5X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_5x_uni_clam_sb` | `fold-5/` 为当前正式结果；部分分析配置仍保留 `fold-0/` |
| `CLAM-SB 20X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_20x_uni_clam_sb` | `fold-5/` 或 `fold-5_rerun_gpu0_20260424/` 为当前正式结果；部分分析配置仍保留 `fold-0/` |
| `Hierarchical CLAM 20X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/hierarchical_clam_20x_uni` | `fold-<fold>/stage1_*`、`fold-<fold>/stage2_*`、合并预测与三分类指标 |

## 方法原理

MIL 将每张 slide 视为 bag，每个 patch feature 是 instance。模型只有 slide-level 标签，需要学习哪些 patch 对 slide 分类更有贡献。

CLAM-SB 在 bag-level 分类之外加入 instance-level clustering：

- bag loss 负责 slide-level 分类。
- instance clustering 让高 attention 和低 attention patch 更有判别性。
- `K_SAMPLE` 控制 top/bottom attention instances 数量。
- `BAG_WEIGHT` 控制 bag loss 和 instance loss 的相对权重。
