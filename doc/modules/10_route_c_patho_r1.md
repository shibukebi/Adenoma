# 10. Route C / Patho-R1 模块

## 模块目标

Route C 当前保留但不是正式主线。目标是用 VLM/Patho-R1 在缩略图上选择 ROI，再转换为 CLAM 兼容 coords h5。

```text
WSI thumbnail
  -> Patho-R1 / heuristic boxes
  -> boxes mapped to level 0 coords
  -> CLAM-compatible coords h5
  -> feature extraction / training
```

## 实现入口

| 脚本 | 作用 |
| --- | --- |
| `scripts/patho_r1_route_c_select.py` | 生成 thumbnail，运行 Patho-R1 或 heuristic，输出 boxes json 和可视化 |
| `scripts/route_c_boxes_to_h5.py` | 将 thumbnail boxes 映射回 level 0 坐标，并生成 CLAM 兼容 `coords` h5 |
| `scripts/run_route_c_batch.py` | 批量处理 Route C manifest |
| `scripts/build_route_c_feature_input.py` | 构建 Route C 特征提取输入 |
| `scripts/prepare_route_c_clam_ssl_20x_data.py` | 准备 Route C CLAM 训练数据 |
| `scripts/train_clam_ssl_20x_route_c.py` | Route C 上的 CLAM 训练入口 |

## 关键函数

`patho_r1_route_c_select.py`：

| 位置 | 职责 |
| --- | --- |
| `generate_thumbnail` | 从 WSI 生成缩略图 |
| `heuristic_boxes` | 无模型或 smoke 场景下生成启发式 boxes |
| `load_patho_r1_stack` | 加载 Patho-R1 processor/model |
| `run_patho_r1_generation` | 调用模型生成选区文本 |
| `parse_boxes_from_text` | 从模型输出解析 boxes |
| `process_slide` | 单张切片 Route C 处理主流程 |

`route_c_boxes_to_h5.py`：

| 位置 | 职责 |
| --- | --- |
| `boxes_to_coords` | 将 box 按 patch size / step size 展开成 level 0 patch 坐标 |
| `write_coords_h5` | 写出 CLAM 兼容 h5 |
| `convert_boxes_json_to_h5` | 单个 boxes json 到 h5 的转换主流程 |

## 结果目录

| 步骤 | 结果目录 / 文件 |
| --- | --- |
| Patho-R1 / heuristic 选区 | `/data15/data15_5/yuexin2/adenoma/outputs/route_c_patho_r1/<slide_id>/` |
| 选区 manifest | `/data15/data15_5/yuexin2/adenoma/outputs/route_c_patho_r1/manifest_route_c.csv` |
| Route C coords root | `/data15/data15_5/yuexin2/adenoma/outputs/route_c_coords` |
| Route C CLAM-compatible patch h5 | `/data15/data15_5/yuexin2/adenoma/outputs/route_c_coords/patches/<slide_id>.h5` |
| Route C patch export | `/data15/data15_5/yuexin2/adenoma/outputs/route_c_patch_export/<slide_id>/` |
| Route C feature input csv | `/data15/data15_5/yuexin2/adenoma/data/route_c_feature_input.csv` |
| Route C UNI features | `/data15/data15_5/yuexin2/adenoma/outputs/route_c_20x_features` |
| Route C ready csv / split | `/data15/data15_5/yuexin2/adenoma/data/route_c_ssl_others_ready.csv`、`/data15/data15_5/yuexin2/adenoma/data/route_c_clam_splits/` |
| Route C CLAM results | `/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_20x_route_c/fold-0` |

## 方法原理

Route C 尝试让 VLM 参与 ROI selection，潜在优点是选区更语义化，并可能绕开纯阈值分割的局限。

当前暂停原因：

- CPU 上真实 `Patho-R1-3B` 推理过慢。
- VLM 输出 boxes 的稳定性需要额外验证。
- 当前 CLAM 预处理 + UNI 特征 + 多模型 baseline 已形成可复现闭环。

恢复条件是换到可用 GPU 机器，或替换为更轻量、更稳定的 VLM/ROI selection 部署。
