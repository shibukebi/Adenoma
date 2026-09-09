# 03. 标签、split 与指标模块

## 模块目标

统一构建 `SSL vs others`、`dysplasia vs no_dysplasia`、最终三分类标签和 MIST hp+yx 多分类标签，生成 ready csv / split，并输出标准评估指标。

## 任务定义

Stage 1 主任务：

| 类别 | 定义 | 标签 |
| --- | --- | --- |
| `SSL` | `type == Sessile serrated adenoma` | positive |
| `others` | 其余全部类型 | negative |

Stage 2 二阶段任务：

| 类别 | 定义 | 标签 |
| --- | --- | --- |
| `dysplasia` | `grade == high` | positive |
| `no_dysplasia` | `grade == low` | negative |

最终三分类：

```text
others
SSL-no-dysplasia
SSL-with-dysplasia
```

MIST hp+yx 最新十分类口径：

| 新标签 | 定义 | 由当前 11 类 manifest 合并方式 |
| --- | --- | --- |
| `ssl` | SSL，不再单独拆出 SSL high-grade dysplasia | `ssl` + `ssl with highgrade dysplasia` |
| `hp` | Hyperplastic polyps | `hp` |
| `TSA` | Traditional serrated adenoma, low grade | `TSA` |
| `USA` | Unclassified serrated adenoma | `USA` |
| `TA` | Tubular adenoma, low grade | `TA` |
| `TVA` | Tubulovillous adenoma, low grade | `TVA` |
| `IP` | Inflammatory polyp | `IP` |
| `TSA with highgrade dysplasia` | TSA high grade | 原 11 类 label 8 |
| `TA with highgrade dysplasia` | TA high grade | 原 11 类 label 9 |
| `TVA with highgrade dysplasia` | TVA high grade | 原 11 类 label 10 |

当前已生成并正在跑的 MIST manifest 仍是 `11class`，即额外保留 `ssl with highgrade dysplasia` 作为独立 label 7，训练脚本使用 `num_classes=11`。若正式切换到十分类，需要同步重建 manifest，并把 MIST 训练/测试命令改为 `num_classes=10`。

## 实现位置

通用工具：

```text
scripts/clam_experiment_utils.py
```

关键函数：

| 位置 | 职责 |
| --- | --- |
| `build_hierarchical_label_row` | 从 `type` 和 `grade` 构建 SSL 标签、dysplasia 标签与最终三分类标签 |
| `prepare_task_row` | 按 `task_mode` 生成训练 ready row；二阶段任务过滤非 SSL 样本 |
| `load_split_ids` | 读取 `flod-*-train/val/test.csv`，并兼容 fold 索引 |
| `compute_binary_metrics` | 计算 AUC、accuracy、macro-F1、precision、recall、specificity、confusion matrix |
| `compute_multiclass_metrics` | 计算层级合并后三分类指标 |
| `merge_hierarchical_predictions` | 合并 Stage 1 和 Stage 2 预测 |
| `build_prediction_frame_from_results` | 将 CLAM evaluation results 转为标准 predictions dataframe |

数据准备脚本：

| 脚本 | 作用 |
| --- | --- |
| `scripts/build_ssl_binary_labels.py` | 构建 SSL/others 标签 |
| `scripts/build_ssl_fold.py` | 构建 fold split |
| `scripts/build_train_test_split.py` | 按标签与数据格式分层构建 8:2 train/test split；支持以 20x ready csv 作为 reference split |
| `scripts/build_train_inner_folds.py` | 将 canonical 训练集进一步分成分层 inner folds |
| `scripts/prepare_clam_ssl_*.py` | 准备 CLAM ready csv 和 split 文件 |
| `scripts/prepare_patch_gcn_ssl_data.py` | 准备 PatchGCN 数据并可构建 graph cache |
| `scripts/prepare_dsmil_ssl_data.py` | 准备 DSMIL 双流数据 |
| `scripts/merge_hierarchical_predictions.py` | 层级预测合并 CLI |
| `scripts/build_mist_11class_manifests.py` | 从 hp+yx ready csv 和 fold split 生成当前 MIST 11 类 manifest |
| `scripts/build_mist_uni_low4cat_features.py` | 构建 MIST low4cat UNI 拼接特征 |

## 结果文件目录

| 类型 | 路径 |
| --- | --- |
| 原始/主标签 | `/data15/data15_5/yuexin2/adenoma/data/adenoma_yx_labels*.csv`、`/data15/data15_5/yuexin2/adenoma/data/adenoma_yx_ssl_others_labels.csv` |
| CLAM ready csv | `/data15/data15_5/yuexin2/adenoma/data/clam_ssl_others*_uni_ready.csv` |
| CLAM fold split | `/data15/data15_5/yuexin2/adenoma/data/clam_ssl_splits*_uni/` |
| TransMIL ready / split | 复用 CLAM ready csv 和 CLAM split |
| PatchGCN ready csv | `/data15/data15_5/yuexin2/adenoma/data/patch_gcn_ssl_others*_uni_ready.csv` |
| PatchGCN split | `/data15/data15_5/yuexin2/adenoma/data/patch_gcn_splits*_uni/` |
| DSMIL ready csv | `/data15/data15_5/yuexin2/adenoma/data/dsmil_ssl_others_*_uni_ready.csv` |
| DSMIL split | `/data15/data15_5/yuexin2/adenoma/data/dsmil_ssl_splits_*_uni/` |
| 层级 Stage 1 ready / split | `/data15/data15_5/yuexin2/adenoma/data/hier*_ssl_binary*_ready.csv`、`/data15/data15_5/yuexin2/adenoma/data/hier*_ssl_binary*_splits*/` |
| 层级 Stage 2 ready / split | `/data15/data15_5/yuexin2/adenoma/data/hier*_dysplasia*_ready.csv`、`/data15/data15_5/yuexin2/adenoma/data/hier*_dysplasia*_splits*/` |
| 8:2 train/test split | `/data15/data15_5/yuexin2/adenoma/data/*_splits_8_2/` |
| canonical train inner folds | `/data15/data15_5/yuexin2/adenoma/data/*_splits_8_2/train_5folds/` |
| MIST hp+yx 11 类 manifest | `/data15/data15_5/yuexin2/MIST/datasets/mist_hp_yx_11class_fold5_*`、`/data15/zhengke_usb2/yuexin_data/adenoma_feature/MIST/manifests_11class_fold5*` |
| MIST hp+yx fold split | `/data15/zhengke_usb2/yuexin_data/splits/adenoma_uni_hp_yx_ssl_5fold/splits/` |

## MIST hp+yx 当前数据划分

MIST hp+yx 使用外置统一 5-fold split，构建脚本为 `/data15/zhengke_usb2/yuexin_data/scripts/build_joint_hp_yx_ssl_5fold.py`。该 split 按 `scanner` 和 `SSL vs others` 分层；当前 MIST 正式运行使用 `flod-4`，文档中按汇报习惯称为 `fold5`。`val` 是下一个 held-out fold，用于兼容训练流程。

当前实际运行的 11 类 manifest：

| split | 总数 | ssl | hp | TSA | USA | TA | TVA | IP | ssl high | TSA high | TA high | TVA high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2433 | 184 | 691 | 28 | 98 | 799 | 210 | 191 | 22 | 57 | 53 | 100 |
| val | 811 | 64 | 242 | 16 | 29 | 267 | 62 | 62 | 5 | 19 | 13 | 32 |
| test | 810 | 68 | 210 | 11 | 22 | 300 | 66 | 61 | 1 | 23 | 18 | 30 |

按最新十分类口径把 `ssl high` 合并回 `ssl` 后，对应分布为：

| split | 总数 | ssl | hp | TSA | USA | TA | TVA | IP | TSA high | TA high | TVA high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2433 | 206 | 691 | 28 | 98 | 799 | 210 | 191 | 57 | 53 | 100 |
| val | 811 | 69 | 242 | 16 | 29 | 267 | 62 | 62 | 19 | 13 | 32 |
| test | 810 | 69 | 210 | 11 | 22 | 300 | 66 | 61 | 23 | 18 | 30 |

当前 MIST manifest 路径：

| 特征组合 | manifest 根目录 | 备注 |
| --- | --- | --- |
| `2.5x + 5x low4cat` | `/data15/data15_5/yuexin2/MIST/datasets/mist_hp_yx_11class_fold5_*`；外置备份 `/data15/zhengke_usb2/yuexin_data/adenoma_feature/MIST/manifests_11class_fold5` | `run_mist_fold5_11class_full.sh` 使用这组训练 |
| `5x + 10x low4cat` | `/data15/zhengke_usb2/yuexin_data/adenoma_feature/MIST/manifests_11class_fold5_5x_10x` | 与 `2.5x + 5x` 共用同一份 `flod-4` split 和同一套 label |

本地二分类 MIST 特征集 `adenoma_uni_2p5x_5x_low4cat` / `adenoma_uni_2p5x_5x_cat` 仍保留，用于 `SSL vs others` 自检；其 fold5 split 为 train 1125 (`others=939`, `SSL=186`)、val 241 (`others=201`, `SSL=40`)、test 241 (`others=201`, `SSL=40`)，不作为 hp+yx 多分类训练 manifest。

## 8:2 train/test split

后续补充数据后的正式口径：以最完整的 20x ready csv 作为 canonical split source，只在 20x 上随机生成一次 train/test；其它所有倍率、模型格式和双流组合都复用这套 slide_id split，再按各自 ready slide 做过滤。这样同一任务、同一 fold 在不同倍率之间不会因为重新随机切分而产生标签分布差异。

### 20x reference split 逻辑

原则：

| 规则 | 说明 |
| --- | --- |
| canonical source | 每个任务只选择一个最完整的 20x ready csv 作为 split 源 |
| split unit | 始终按 slide_id 划分，避免 patch 泄漏 |
| random split | 只在 canonical 20x ready csv 上执行一次 |
| reuse split | 1x、2.5x、5x、PatchGCN、DSMIL、TransMIL 等全部复用 canonical split 的 slide_id |
| unavailable slide | 某倍率暂时缺失的 slide 只从该倍率 split 中过滤，不触发重新随机划分 |
| supplement update | 后续补齐低倍率后，重新用同一个 20x canonical source 生成 split，低倍率应自动恢复到与 20x 相同的标签分布 |
| task isolation | `ssl_binary` 和 `dysplasia_binary` 分别使用各自的 20x canonical ready csv，不能混用 |

当前推荐 canonical 文件：

| 任务 | 20x canonical ready csv | 复用对象 |
| --- | --- | --- |
| Stage 1 `ssl_binary` | `/data15/data15_5/yuexin2/adenoma/data/clam_ssl_others_uni_ready.csv` | CLAM/TransMIL/PatchGCN/DSMIL 的 SSL vs others split |
| Stage 2 `dysplasia_binary` | `/data15/data15_5/yuexin2/adenoma/data/hier_dysplasia_20x_uni_ready.csv` | 层级二阶段 dysplasia vs no_dysplasia split |

命令模板：

```bash
/data15/data15_5/yuexin2/anaconda3/envs/patho-r1/bin/python \
  /data15/data15_5/yuexin2/adenoma/scripts/build_train_test_split.py \
  --reference-csv /data15/data15_5/yuexin2/adenoma/data/clam_ssl_others_uni_ready.csv \
  --input-csv /data15/data15_5/yuexin2/adenoma/data/clam_ssl_others_1x_uni_ready.csv \
  --input-csv /data15/data15_5/yuexin2/adenoma/data/clam_ssl_others_2p5x_uni_ready.csv \
  --input-csv /data15/data15_5/yuexin2/adenoma/data/clam_ssl_others_5x_uni_ready.csv \
  --input-csv /data15/data15_5/yuexin2/adenoma/data/clam_ssl_others_uni_ready.csv
```

二阶段任务使用独立的 20x reference：

```bash
/data15/data15_5/yuexin2/anaconda3/envs/patho-r1/bin/python \
  /data15/data15_5/yuexin2/adenoma/scripts/build_train_test_split.py \
  --reference-csv /data15/data15_5/yuexin2/adenoma/data/hier_dysplasia_20x_uni_ready.csv \
  --input-csv /data15/data15_5/yuexin2/adenoma/data/hier_dysplasia_1x_uni_ready.csv \
  --input-csv /data15/data15_5/yuexin2/adenoma/data/hier_dysplasia_2p5x_uni_ready.csv \
  --input-csv /data15/data15_5/yuexin2/adenoma/data/hier_dysplasia_5x_uni_ready.csv \
  --input-csv /data15/data15_5/yuexin2/adenoma/data/hier_dysplasia_20x_uni_ready.csv
```

`--reference-csv` 模式会把 `reference_csv`、`reference_split_sizes` 和每个输入数据相对 reference 被过滤掉的 slide 数写入 `flod-0_stats.json`，用于检查某倍率是否仍有缺失。

当前已经为 `adenoma/data/*ready.csv` 生成过一版 8:2 训练集/测试集划分：

```bash
/data15/data15_5/yuexin2/anaconda3/envs/patho-r1/bin/python \
  /data15/data15_5/yuexin2/adenoma/scripts/build_train_test_split.py
```

输出目录命名为 `{ready_csv_stem_without_ready}_splits_8_2/`，每个目录包含：

```text
flod-0-train.csv
flod-0-test.csv
flod-0_stats.json
```

无 `--reference-csv` 时，脚本会对每个 ready csv 自己分层划分；该模式仅适合单个数据集自检，不作为后续跨倍率正式比较口径。

### 跨倍率一致性复查

同一 fold 的跨倍率比较必须限定在同一 ready 样本池内。`*_uni_*` 与非 `uni` 文件不是同一批 ready slide，不能直接要求标签分布相同。

| 可比样本池 | 对应 split | slide 数 | train 是否一致 | test 是否一致 | 标签分布 |
| --- | --- | --- | --- | --- | --- |
| SSL binary UNI | `clam_ssl_others_{1x,2p5x,5x,uni}_splits_8_2/flod-0` | 1608 | yes | yes | train `SSL=213, others=1074`; test `SSL=53, others=268` |
| SSL binary non-UNI | `clam_ssl_others_{2p5x,5x}_splits_8_2/flod-0` | 1535 | yes | yes | train `SSL=200, others=1028`; test `SSL=50, others=257` |
| Dysplasia binary UNI | `hier_dysplasia_{1x,2p5x,5x,20x,uni}_splits_8_2/flod-0` | 266 | yes | yes | train `dysplasia=13, no_dysplasia=200`; test `dysplasia=3, no_dysplasia=50` |
| 既有 SSL fold UNI | `clam_ssl_splits_{1x,2p5x,5x,uni}/flod-5` | 1608 | yes | yes | train `SSL=186, others=940`; test `SSL=40, others=201` |
| 既有 SSL fold non-UNI | `clam_ssl_splits_{2p5x,5x}/flod-0` | 1535 | yes | yes | train `SSL=175, others=893`; test `SSL=37, others=195` |
| 既有 Dysplasia fold UNI | `hier_dysplasia_splits_{1x,2p5x,5x,20x,uni}/flod-5` | 266 | yes | yes | train `dysplasia=12, no_dysplasia=174`; test `dysplasia=1, no_dysplasia=39` |

非 `uni` 的 `clam_ssl_others_2p5x_ready.csv` / `clam_ssl_others_5x_ready.csv` 相比 `uni` 样本池少 73 个 slide，其中 `others=57`、`SSL=16`，因此它们的 8:2 和 fold 分布与 `uni` 样本池不同是预期结果。



### Canonical 训练集+测试集 label 分布

在 canonical `8:2` split 的训练集内部继续构建 5 个分层 fold；测试集保持独立，不参与 inner fold。每个 `flod-*` 文件表示该 inner fold 的 held-out 子集，同时输出 `flod-*-train.csv` 和 `flod-*-val.csv` 便于后续做训练集内部交叉验证。下表同时补充固定 canonical test set 的标签分布。

Stage 1 命令：

```bash
/data15/data15_5/yuexin2/anaconda3/envs/patho-r1/bin/python \
  /data15/data15_5/yuexin2/adenoma/scripts/build_train_inner_folds.py \
  --ready-csv /data15/data15_5/yuexin2/adenoma/data/clam_ssl_others_uni_ready.csv \
  --train-ids-csv /data15/data15_5/yuexin2/adenoma/data/clam_ssl_others_uni_splits_8_2/flod-0-train.csv \
  --output-dir /data15/data15_5/yuexin2/adenoma/data/clam_ssl_others_uni_splits_8_2/train_5folds \
  --stratify-column type \
  --folds 5 \
  --seed 2026
```

Stage 1 输出目录：`/data15/data15_5/yuexin2/adenoma/data/clam_ssl_others_uni_splits_8_2/train_5folds/`。下表汇总每个 inner fold held-out 子集的息肉类型分布，并在最后补充固定 test set 分布。

| split | 总数 | inner train 总数 | HP | Inflammatory | SSL | TSA | TA | TVA | USA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flod-0` | 260 | 1027 | 89 | 14 | 43 | 14 | 62 | 19 | 19 |
| `flod-1` | 259 | 1028 | 88 | 14 | 43 | 14 | 62 | 19 | 19 |
| `flod-2` | 257 | 1030 | 88 | 14 | 43 | 13 | 62 | 18 | 19 |
| `flod-3` | 256 | 1031 | 88 | 14 | 42 | 13 | 62 | 18 | 19 |
| `flod-4` | 255 | 1032 | 88 | 14 | 42 | 13 | 61 | 18 | 19 |
| **train total** | **1287** | - | **441** | **70** | **213** | **67** | **309** | **92** | **95** |
| **fixed test** | **321** | - | **119** | **18** | **53** | **17** | **79** | **18** | **17** |

Stage 2 命令：

```bash
/data15/data15_5/yuexin2/anaconda3/envs/patho-r1/bin/python \
  /data15/data15_5/yuexin2/adenoma/scripts/build_train_inner_folds.py \
  --ready-csv /data15/data15_5/yuexin2/adenoma/data/hier_dysplasia_20x_uni_ready.csv \
  --train-ids-csv /data15/data15_5/yuexin2/adenoma/data/hier_dysplasia_20x_uni_splits_8_2/flod-0-train.csv \
  --output-dir /data15/data15_5/yuexin2/adenoma/data/hier_dysplasia_20x_uni_splits_8_2/train_5folds \
  --stratify-column label_name \
  --folds 5 \
  --seed 2026
```

Stage 2 输出目录：`/data15/data15_5/yuexin2/adenoma/data/hier_dysplasia_20x_uni_splits_8_2/train_5folds/`。二阶段 inner fold 只包含 SSL 样本，下表按 dysplasia 标签汇总，并在最后补充固定 test set 分布。

| split | 总数 | inner train 总数 | dysplasia/high | no_dysplasia/low |
| --- | --- | --- | --- | --- |
| `flod-0` | 43 | 170 | 3 | 40 |
| `flod-1` | 43 | 170 | 3 | 40 |
| `flod-2` | 43 | 170 | 3 | 40 |
| `flod-3` | 42 | 171 | 2 | 40 |
| `flod-4` | 42 | 171 | 2 | 40 |
| **train total** | **213** | - | **13** | **200** |
| **fixed test** | **53** | - | **3** | **50** |

### 既有 fold split 标签分布

下表汇总当前已有 fold split 的训练集和测试集标签分布；验证集统计见对应 `flod-*_stats.json`。

| 数据集 | fold | 训练集标签分布 | 测试集标签分布 | 状态 |
| --- | --- | --- | --- | --- |
| `clam_ssl_splits` | `flod-0` | 1126 (`SSL=186`, `others=940`) | 241 (`SSL=40`, `others=201`) | ready |
| `clam_ssl_splits_1x_uni` | `flod-5` | 1126 (`SSL=186`, `others=940`) | 241 (`SSL=40`, `others=201`) | ready |
| `clam_ssl_splits_2p5x` | `flod-0` | 1068 (`SSL=175`, `others=893`) | 232 (`SSL=37`, `others=195`) | ready |
| `clam_ssl_splits_2p5x_uni` | `flod-5` | 1126 (`SSL=186`, `others=940`) | 241 (`SSL=40`, `others=201`) | ready |
| `clam_ssl_splits_5x` | `flod-0` | 1068 (`SSL=175`, `others=893`) | 232 (`SSL=37`, `others=195`) | ready |
| `clam_ssl_splits_5x_uni` | `flod-5` | 1126 (`SSL=186`, `others=940`) | 241 (`SSL=40`, `others=201`) | ready |
| `clam_ssl_splits_uni` | `flod-5` | 1126 (`SSL=186`, `others=940`) | 241 (`SSL=40`, `others=201`) | ready |
| `dsmil_ssl_splits_2p5x_20x_uni` | `flod-5` | 1126 (`SSL=186`, `others=940`) | 241 (`SSL=40`, `others=201`) | ready |
| `dsmil_ssl_splits_2p5x_5x_uni` | `flod-5` | 1125 (`SSL=186`, `others=939`) | 241 (`SSL=40`, `others=201`) | ready |
| `dsmil_ssl_splits_5x_20x_uni` | `flod-5` | 1125 (`SSL=186`, `others=939`) | 241 (`SSL=40`, `others=201`) | ready |
| `hier_dsmil_dysplasia_5x_20x_splits` | `flod-5` | 186 (`dysplasia=12`, `no_dysplasia=174`) | 40 (`dysplasia=1`, `no_dysplasia=39`) | ready |
| `hier_dsmil_ssl_binary_5x_20x_splits` | `flod-5` | 1126 (`SSL=186`, `others=940`) | 241 (`SSL=40`, `others=201`) | ready |
| `hier_dysplasia_splits_1x_uni` | `flod-5` | 186 (`dysplasia=12`, `no_dysplasia=174`) | 40 (`dysplasia=1`, `no_dysplasia=39`) | ready |
| `hier_dysplasia_splits_20x` | `flod-5` | 0 (-) | 0 (-) | not_ready |
| `hier_dysplasia_splits_20x_uni` | `flod-5` | 186 (`dysplasia=12`, `no_dysplasia=174`) | 40 (`dysplasia=1`, `no_dysplasia=39`) | ready |
| `hier_dysplasia_splits_2p5x_uni` | `flod-5` | 186 (`dysplasia=12`, `no_dysplasia=174`) | 40 (`dysplasia=1`, `no_dysplasia=39`) | ready |
| `hier_dysplasia_splits_5x_uni` | `flod-5` | 186 (`dysplasia=12`, `no_dysplasia=174`) | 40 (`dysplasia=1`, `no_dysplasia=39`) | ready |
| `hier_dysplasia_splits_uni` | `flod-5` | 186 (`dysplasia=12`, `no_dysplasia=174`) | 40 (`dysplasia=1`, `no_dysplasia=39`) | ready |
| `hier_patch_gcn_dysplasia_splits_uni` | `flod-5` | 186 (`dysplasia=12`, `no_dysplasia=174`) | 40 (`dysplasia=1`, `no_dysplasia=39`) | ready |
| `hier_patch_gcn_ssl_binary_splits_uni` | `flod-5` | 1126 (`SSL=186`, `others=940`) | 241 (`SSL=40`, `others=201`) | not_ready |
| `hier_ssl_binary_splits_uni` | `flod-5` | 1126 (`SSL=186`, `others=940`) | 241 (`SSL=40`, `others=201`) | ready |
| `route_c_clam_splits` | `flod-0` | 2 (`SSL=1`, `others=1`) | 0 (-) | not_ready |

状态沿用对应 `flod-*_stats.json` 的 `formal_ready`。其中 `hier_patch_gcn_ssl_binary_splits_uni` 的 `not_ready` 来自 graph cache 构建错误，不是 train/test 标签缺失；`hier_dysplasia_splits_20x` 和 `route_c_clam_splits` 是样本不足或类别缺失。

## 方法原理

slide-level 任务必须保证 split 单位清晰，避免 patch 泄漏。当前正式口径使用 `fold-5`。二阶段 dysplasia 任务只在真实 SSL 子集上训练和评估，因为 dysplasia 语义只在 SSL 内部成立。

二阶段阳性样本少，因此结果更适合作为可行性验证，不宜做过强结论。
