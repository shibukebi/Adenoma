# 02. UNI 特征提取模块

## 模块目标

将 CLAM 生成的 patch 坐标转换为 UNI patch features，输出 h5 和 pt 两种格式，供 CLAM、TransMIL、PatchGCN、DSMIL 使用。

## 实现入口

核心脚本：

```text
scripts/extract_features_route_a.py
```

包装脚本：

| 入口 | 作用 |
| --- | --- |
| `scripts/run_extract_features.sh` | 默认 20X 特征提取 |
| `scripts/run_extract_features_1x.sh` | 1X-style UNI 特征 |
| `scripts/run_extract_features_2p5x.sh` | 2.5X UNI 特征 |
| `scripts/run_extract_features_5x.sh` | 5X UNI 特征 |
| `scripts/run_extract_features_10x.sh` | 10X UNI 特征；如脚本尚未落地，应按本节 10X 约定新增 |
| `scripts/run_route_c_extract_features.sh` | Route C 坐标上的特征提取 |

## 关键函数 / 类

| 位置 | 职责 |
| --- | --- |
| `extract_features_route_a.py::read_patch_metadata` | 读取 h5 中 `patch_size`、`patch_level`、`downsample`，计算 legacy 物理视野 |
| `extract_features_route_a.py::PhysicalExtentWholeSlideBag` | 按 level 0 坐标和固定 `physical_level0_extent` 从 WSI 读取图像块 |
| `extract_features_route_a.py::configure_encoder_env` | 根据 `model_name` 和 `weights_path` 设置 UNI/CONCH/ResNet 权重环境变量 |
| `extract_features_route_a.py::compute_w_loader` | 批量编码 patch，将 `features` 和 `coords` 写入 h5 |
| `extract_features_route_a.py::main` | 串联 manifest、WSI 打开、encoder 加载、特征输出、`.pt` 保存 |

## 输出格式

| 输出 | 说明 |
| --- | --- |
| `outputs/<feature_run>/h5_files/<slide_id>.h5` | 包含 `features` 与 `coords` |
| `outputs/<feature_run>/pt_files/<slide_id>.pt` | 训练常用特征张量 |

h5 attrs 会记录 `extraction_mode`、`physical_level_0_extent`、`target_patch_size`、`model_name` 等 provenance。

## 结果目录

| 路线 | 特征目录 | h5 子目录 | pt 子目录 |
| --- | --- | --- | --- |
| `Adenoma_yx 20X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_uni` | `h5_files/` | `pt_files/` |
| `Adenoma_yx 1X-style UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_1x_uni` | `h5_files/` | `pt_files/` |
| `Adenoma_yx 2.5X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_2p5x_uni` | `h5_files/` | `pt_files/` |
| `Adenoma_yx 5X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_5x_uni` | `h5_files/` | `pt_files/` |
| `Adenoma_yx 10X UNI` | `/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_10x_uni` | `h5_files/` | `pt_files/` |
| `Adenoma_hp 1X UNI` | `/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/hp/1x` | `h5_files/` | `pt_files/` |
| `Adenoma_hp 2.5X UNI` | `/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/hp/2p5x` | `h5_files/` | `pt_files/` |
| `Adenoma_hp 5X UNI` | `/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/hp/5x` | `h5_files/` | `pt_files/` |
| `Adenoma_hp 10X UNI` | `/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/hp/10x` | `h5_files/` | `pt_files/` |
| `Adenoma_hp 20X UNI` | `/data15/zhengke_usb2/yuexin_data/adenoma_feature/UNI/hp/adenoma_hp_features_uni_20x` | `h5_files/` | `pt_files/` |

## 方法原理

核心区分：

| 概念 | 含义 |
| --- | --- |
| `physical_level_0_extent` | patch 在 WSI level 0 上覆盖的实际边长 |
| `target_patch_size` | 送入 UNI 前 resize 后的图像尺寸 |

例如 `physical_level_0_extent = 4096` 且 `target_patch_size = 224`，表示从 WSI level 0 读取 `4096 x 4096` 的物理区域，再 resize 成 `224 x 224` 输入 UNI。

正式 UNI 路线不依赖 legacy `patch_level`，因为不同 WSI 的同一层级索引可能对应不同 downsample。项目使用 `--expected-physical-level0-extent` 显式统一物理视野。

## h5 倍率一致性要求

由于 `Adenoma_hp` 和 `Adenoma_yx` 的 WSI 扫描基准倍率均为 40x，UNI 输入所依赖的 h5 必须严格表达目标倍率对应的 level 0 物理视野。生成或读取 h5 时，不允许把 `patch_level`、`source_level_downsample` 或某个 pyramid downsample 当作最终倍率定义。

倍率换算只使用：

```text
base_magnification = 40
scale_to_level0 = base_magnification / target_magnification
physical_level_0_extent = physical_extent_patch_size * scale_to_level0
```

当 `physical_extent_patch_size = 256` 时：

| UNI / h5 倍率 | `physical_level_0_extent` |
| --- | ---: |
| 20x | 512 |
| 10x | 1024 |
| 5x | 2048 |
| 2.5x | 4096 |
| 1x | 10240 |

UNI 特征提取的标准流程是：用 h5 的 level 0 `coords` 定位左上角，从 WSI level 0 坐标系读取 `physical_level_0_extent x physical_level_0_extent` 的物理区域，再 resize 到 UNI 所需输入大小。这样 hp/yx 即使 WSI pyramid 层级组织不同，最终 h5 和 UNI feature 仍严格对应同一目标倍率。
