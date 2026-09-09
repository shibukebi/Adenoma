# 11. MIST 模块

## 模块目标

复现并扩展 MIST 多倍率 MIL 路线，当前重点是 hp+yx 联合数据上的多分类实验，以及 `2.5x + 5x` / `5x + 10x` UNI low4cat 特征。

## 当前分类口径

最新十分类汇报口径为：

| id | 类别 |
| ---: | --- |
| 0 | `ssl` |
| 1 | `hp` |
| 2 | `TSA` |
| 3 | `USA` |
| 4 | `TA` |
| 5 | `TVA` |
| 6 | `IP` |
| 7 | `TSA with highgrade dysplasia` |
| 8 | `TA with highgrade dysplasia` |
| 9 | `TVA with highgrade dysplasia` |

当前已经生成并正在跑的 MIST manifest 仍是 `11class`：它把 `ssl with highgrade dysplasia` 作为独立 class 7，后续 `TSA/TA/TVA high` 分别为 8/9/10，训练脚本使用 `num_classes=11`。按十分类口径汇报时，需要把 `ssl with highgrade dysplasia` 合并回 `ssl`；如果重新训练十分类模型，需要重建 manifest 并把训练/测试命令改成 `num_classes=10`。

## 数据与特征

| 项 | 路径 |
| --- | --- |
| MIST 代码 | `/data15/data15_5/yuexin2/MIST` |
| hp+yx ready csv | `/data15/zhengke_usb2/yuexin_data/splits/adenoma_uni_hp_yx_ssl_5fold/joint_hp_yx_ssl_ready.csv` |
| hp+yx split | `/data15/zhengke_usb2/yuexin_data/splits/adenoma_uni_hp_yx_ssl_5fold/splits` |
| `2.5x + 5x` low4cat 特征 | `/data15/zhengke_usb2/yuexin_data/adenoma_feature/MIST/adenoma_uni_hp_yx_2p5x_5x_low4cat` |
| `5x + 10x` low4cat 特征 | `/data15/zhengke_usb2/yuexin_data/adenoma_feature/MIST/adenoma_uni_hp_yx_5x_10x_low4cat` |
| 本地 11 类 manifest | `/data15/data15_5/yuexin2/MIST/datasets/mist_hp_yx_11class_fold5_*` |
| 外置 11 类 manifest | `/data15/zhengke_usb2/yuexin_data/adenoma_feature/MIST/manifests_11class_fold5`、`/data15/zhengke_usb2/yuexin_data/adenoma_feature/MIST/manifests_11class_fold5_5x_10x` |
| 训练结果 | `/data15/zhengke_usb2/yuexin_data/result/fold5_hp+yx/MIST` |

low4cat 每个低倍 patch 生成一行实例特征：

```text
[low_mag_UNI, high_mag_TL, high_mag_TR, high_mag_BL, high_mag_BR]
```

单个 UNI 向量为 1024 维，拼接后 `feats_size=5120`。缺失高倍象限用 0 向量补齐，多候选取离象限中心最近的 patch。

## 当前 MIST Split

当前 MIST 正式运行使用 `flod-4`，文档按汇报习惯称为 `fold5`。`val` 使用下一个 held-out fold，不与 test 重叠。

| split | 总数 | ssl | hp | TSA | USA | TA | TVA | IP | ssl high | TSA high | TA high | TVA high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2433 | 184 | 691 | 28 | 98 | 799 | 210 | 191 | 22 | 57 | 53 | 100 |
| val | 811 | 64 | 242 | 16 | 29 | 267 | 62 | 62 | 5 | 19 | 13 | 32 |
| test | 810 | 68 | 210 | 11 | 22 | 300 | 66 | 61 | 1 | 23 | 18 | 30 |

十分类合并后：

| split | 总数 | ssl | hp | TSA | USA | TA | TVA | IP | TSA high | TA high | TVA high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2433 | 206 | 691 | 28 | 98 | 799 | 210 | 191 | 57 | 53 | 100 |
| val | 811 | 69 | 242 | 16 | 29 | 267 | 62 | 62 | 19 | 13 | 32 |
| test | 810 | 69 | 210 | 11 | 22 | 300 | 66 | 61 | 23 | 18 | 30 |

## 运行入口

| 脚本 | 作用 |
| --- | --- |
| `adenoma/scripts/build_mist_11class_manifests.py` | 从 ready csv、特征根目录和 split 生成 MIST manifest |
| `adenoma/scripts/build_mist_uni_low4cat_features.py` | 生成 low4cat 拼接特征 |
| `/data15/data15_5/yuexin2/MIST/run_mist_fold5_11class_full.sh` | 本地 `2.5x + 5x` 11 类 200 epoch 训练入口 |

当前本地训练命令核心参数：

```bash
python train_ade.py \
  --dataset_train mist_hp_yx_11class_fold5_train \
  --dataset_val mist_hp_yx_11class_fold5_val \
  --num_classes 11 \
  --feats_size 5120 \
  --num_epochs 200
```

## 注意事项

- MIST hp+yx split 文件名是 `flod-4-*`，但汇报命名为 `fold5`。
- hp+yx split 构建时按 `scanner + SSL/others` 分层；MIST 的 10/11 类标签是在 manifest 阶段由 `type` 和 `grade` 映射出来的。
- `adenoma_uni_2p5x_5x_low4cat_fold5_train/val/test` 这类自动生成 manifest 仍是本地二分类 `SSL vs others` 标签，不等同于 hp+yx 多分类 manifest。
