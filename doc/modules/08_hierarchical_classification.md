# 08. 层级分类模块

## 模块目标

采用两阶段层级分类，把 `SSL vs others` 主任务和 `dysplasia vs no_dysplasia` 细粒度任务连接起来。

```text
Stage 1: all slides -> SSL vs others
Stage 2: true SSL subset -> dysplasia vs no_dysplasia
Merge: others / SSL-no-dysplasia / SSL-with-dysplasia
```

## 实现入口

| 路线 | 入口 |
| --- | --- |
| `Hierarchical CLAM` | `scripts/run_hierarchical_clam_20x.sh`、`scripts/run_hierarchical_clam_like_20x.sh` |
| `Hierarchical TransMIL` | `scripts/run_hierarchical_transmil_20x.sh` |
| `Hierarchical PatchGCN` | `scripts/run_hierarchical_patch_gcn_20x.sh` |
| `Hierarchical DSMIL` | `scripts/run_hierarchical_dsmil_pair.sh`、`scripts/run_hierarchical_dsmil_5x_20x.sh` |
| 合并预测 | `scripts/merge_hierarchical_predictions.py` |

## 关键函数

| 位置 | 职责 |
| --- | --- |
| `clam_experiment_utils.py::build_hierarchical_label_row` | 构建 SSL、dysplasia 和最终三分类标签 |
| `clam_experiment_utils.py::prepare_task_row` | Stage 2 自动过滤非 SSL 样本 |
| `clam_experiment_utils.py::select_routing_threshold` | 根据 Stage 1 验证集选择 routing threshold |
| `clam_experiment_utils.py::merge_hierarchical_predictions` | 合并 Stage 1 / Stage 2，生成最终三分类预测 |
| `clam_experiment_utils.py::compute_multiclass_metrics` | 计算最终三分类指标 |

## 结果目录

| 路线 | 结果根目录 |
| --- | --- |
| `Hierarchical CLAM 20X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/hierarchical_clam_20x_uni` |
| `Hierarchical TransMIL 20X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/hierarchical_transmil_20x_uni` |
| `Hierarchical PatchGCN 20X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/hierarchical_patch_gcn_20x_uni` |
| `Hierarchical DSMIL 5X + 20X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/hierarchical_dsmil_5x_20x_uni` |

每个 fold 下通常包含 Stage 1 结果、Stage 2 结果、merged predictions、multiclass metrics 和 confusion matrix。

## 方法原理

`dysplasia` 只在 SSL 语义下有意义，所以项目先识别 SSL，再在 SSL 子集上识别 dysplasia。这样可以避免直接三分类时少数类被多数类淹没。

当前 `SSL + dysplasia` 样本很少，fold-5 test 中 dysplasia 阳性也很少，因此二阶段结果适合作为可行性验证，不适合做过强结论。
