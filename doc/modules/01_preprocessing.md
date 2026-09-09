# 01. WSI 预处理模块

## 模块目标

从 WSI 中完成组织区域分割并生成 patch 坐标，供后续 UNI 特征提取使用。

输入：

- `Adenoma_yx`：`/data15/zhengke_usb/Adenoma_yx/*.svs`
- `Adenoma_hp`：`/data15/zhengke_usb/Adenoma_hp/*.isyntax`

## 实现入口

| 入口 | 作用 |
| --- | --- |
| `scripts/run_preprocess.sh` | 默认 20X / `Adenoma_yx` 预处理 |
| `scripts/run_preprocess_1x.sh` | 1X-style 低倍路线 |
| `scripts/run_preprocess_2p5x.sh` | 2.5X 低倍路线 |
| `scripts/run_preprocess_5x.sh` | 5X 低倍路线 |
| `scripts/run_preprocess_10x.sh` | 10X 路线；如脚本尚未落地，应按本节 10X 约定新增 |
| `scripts/start_preprocess_background.sh` | 后台启动全量预处理 |
| `scripts/build_slide_manifest.sh` | 构建 slide manifest |
| `scripts/make_smoke_input.sh` | 构建 smoke test 输入 |
| `scripts/check_preprocess_status.sh` | 查看预处理状态 |

外部核心：

```text
/data15/data15_5/yuexin2/adenoma/CLAM/create_patches_fp.py
```

## 实现细节

项目脚本读取 `.env` 配置并组织 CLAM 命令；CLAM 负责 tissue segmentation、patch coordinate extraction、mask visualization、`process_list` 状态记录和 `auto_skip` 断点续跑。

常见配置：

| 配置 | 作用 |
| --- | --- |
| `config/adenoma_yx_preprocess.env` | 20X `.svs` 默认预处理 |
| `config/route_lowmag_1x.env` | 1X-style 低倍路线 |
| `config/route_lowmag_2p5x.env` | 2.5X 路线 |
| `config/route_lowmag_5x.env` | 5X 路线 |
| `config/route_lowmag_10x.env` | 10X yx 路线；如配置尚未落地，应按 `TARGET_MAGNIFICATION=10` 新增 |
| `config/adenoma_hp_isyntax_preprocess*.env` | `.isyntax` 数据预处理 |

典型输出：

| 输出 | 说明 |
| --- | --- |
| `runs/<run_name>/patches/<slide_id>.h5` | CLAM 坐标文件，核心 dataset 是 `coords` |
| `runs/<run_name>/masks/<slide_id>.jpg` | 组织分割 / Tissue Mask 可视化 |
| `runs/<run_name>/process_list*.csv` | 每张切片处理状态、参数与断点续跑依据 |


## CLAM Tissue Mask 与组织切割

当前标准流程直接使用 CLAM 组织分割结果进行组织切割和 patch 坐标生成，不再执行额外的污渍去除或 stain refinement 后处理。CLAM 输出的 `segmentations/<slide_id>.pkl` 是结构化组织来源，包含 level 0 坐标系下的 `tissue` 和 `holes` contours；后续 patch extraction、thumbnail tissue crop 和 TraceAgent grid 均应优先使用这份 CLAM segmentation。

当前推荐流程：

```text
WSI
  -> CLAM tissue segmentation
  -> segmentations/<slide_id>.pkl
  -> masks/<slide_id>.jpg for QC
  -> patch coordinate extraction
  -> patches/<slide_id>.h5
```

关键约定：

- 不再从 CLAM mask 中扣除 HSV / LAB 阈值识别出的暗污渍区域。
- 不再用连通域、形态学膨胀或细长区域规则修改 tissue mask。
- 不再把 stain-filtered mask 作为默认 patch extraction 输入。
- `runs/<run_name>/masks/<slide_id>.jpg` 仅用于质控和可视化，不作为结构化坐标来源。
- `runs/<run_name>/segmentations/<slide_id>.pkl` 是当前推荐的组织边界来源。

当前已接入和 10X 约定的 hp / yx 图像预处理入口：

| 入口 / 配置 | 当前接入方式 |
| --- | --- |
| `scripts/run_preprocess.sh` + `config/adenoma_hp_isyntax_preprocess_20x.env` | 20X hp 预处理直接使用 CLAM segmentation 生成 patch h5 |
| `scripts/run_preprocess.sh` + `config/adenoma_hp_isyntax_preprocess.env` | 默认 hp 配置直接使用 CLAM segmentation |
| `scripts/run_preprocess.sh` + `config/adenoma_yx_preprocess.env` | 20X yx 预处理直接使用 CLAM segmentation |
| `scripts/run_preprocess.sh` + `config/route_lowmag_1x.env` | 1X yx 按 1X 物理倍率生成 coords；如需复用 segmentation，应复用 CLAM segmentation pkl |
| `scripts/run_preprocess.sh` + `config/route_lowmag_2p5x.env` | 2.5X yx 按 2.5X 物理倍率生成 coords；如需复用 segmentation，应复用 CLAM segmentation pkl |
| `scripts/run_preprocess.sh` + `config/route_lowmag_5x.env` | 5X yx 按 5X 物理倍率生成 coords；如需复用 segmentation，应复用 CLAM segmentation pkl |
| `scripts/run_preprocess.sh` + `config/route_lowmag_10x.env` | 10X yx 按 10X 物理倍率生成 coords；如需复用 segmentation，应复用 CLAM segmentation pkl |
| `scripts/run_preprocess.sh` + `config/adenoma_hp_isyntax_preprocess_1x.env` | 1X hp 按低倍物理视野重新生成 coords；如需复用 segmentation，应复用 CLAM segmentation pkl |
| `scripts/run_preprocess.sh` + `config/adenoma_hp_isyntax_preprocess_2p5x.env` | 2.5X hp 按低倍物理视野重新生成 coords；如需复用 segmentation，应复用 CLAM segmentation pkl |
| `scripts/run_preprocess.sh` + `config/adenoma_hp_isyntax_preprocess_5x.env` | 5X hp 按 5X 物理倍率生成 coords；如需复用 segmentation，应复用 CLAM segmentation pkl |
| `scripts/run_preprocess.sh` + `config/adenoma_hp_isyntax_preprocess_10x.env` | 10X hp 按 10X 物理倍率生成 coords；如需复用 segmentation，应复用 CLAM segmentation pkl |

hp / yx 相关配置项：

| 配置项 | 当前约定 |
| --- | --- |
| `SEGMENTATION_MASK_DIR` | 可选：指定 CLAM-compatible segmentation pkl 目录以复用组织轮廓 |
| `STAIN_REFINE_SEGMENTATION` | 历史兼容配置；当前推荐保持关闭，不作为默认流程 |
| `STAIN_REFINED_OUTPUT_DIR` | 历史 stain refinement 输出目录；当前标准流程不依赖 |
| `STAIN_REFINE_TARGET_DOWNSAMPLE` | 历史 stain refinement 参数；当前标准流程不依赖 |
| `STAIN_REFINE_COLOR_SPACE` | 历史 stain refinement 参数；当前标准流程不依赖 |
| `STAIN_REFINE_THRESHOLD` | 历史 stain refinement 参数；当前标准流程不依赖 |
| `STAIN_REFINE_EXTRA_ARGS` | 历史 stain refinement 参数；当前标准流程不依赖 |

当前统一约定：`Adenoma_hp` 和 `Adenoma_yx` 的 patch 坐标应来自 CLAM segmentation 定义的组织区域，而不是 stain-refined segmentation。1X、2.5X、5X、10X 低倍率输出仍然按 `TARGET_MAGNIFICATION` 和 `PHYSICAL_EXTENT_PATCH_SIZE` 重新计算 level 0 物理视野与 patch coords，因此不是把 20X 图像缩放到低倍率。

如果某次实验确实需要历史 stain refinement，应显式在实验记录中标注，并把产物目录与当前标准 CLAM 产物区分开。默认文档、TraceAgent thumbnail grid 和下游 ROI 选择均以 CLAM segmentation 为准。

实现时建议保留以下产物，便于质控和回溯：

| 产物 | 建议路径 | 说明 |
| --- | --- | --- |
| 组织 mask 可视化 | `runs/<run_name>/masks/<slide_id>.jpg` | CLAM segmentation 可视化，仅用于质控 |
| CLAM segmentation pkl | `runs/<run_name>/segmentations/<slide_id>.pkl` | 保存 `tissue` / `holes` contours，作为结构化组织来源 |
| patch h5 | `runs/<run_name>/patches/<slide_id>.h5` | 保存 level 0 patch 左上角坐标 |

注意：最终进入 patch 提取逻辑的应是 CLAM segmentation pkl 中的结构化 tissue / holes contours，而不是仅用于展示的彩色 mask 图。

## 结果目录

| 路线 | patch h5 目录 | mask / process_list 所在目录 |
| --- | --- | --- |
| `Adenoma_yx 20X` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess/patches` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess` |
| `Adenoma_yx 1X-style` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess_1x/patches` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess_1x` |
| `Adenoma_yx 2.5X` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess_2p5x/patches` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess_2p5x` |
| `Adenoma_yx 5X` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess_5x/patches` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess_5x` |
| `Adenoma_yx 10X` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess_10x/patches` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess_10x` |
| `Adenoma_hp 1X` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_isyntax_preprocess_1x/patches` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_isyntax_preprocess_1x` |
| `Adenoma_hp 2.5X` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_isyntax_preprocess_2p5x/patches` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_isyntax_preprocess_2p5x` |
| `Adenoma_hp 5X` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_isyntax_preprocess_5x/patches` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_isyntax_preprocess_5x` |
| `Adenoma_hp 10X` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_isyntax_preprocess_10x/patches` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_isyntax_preprocess_10x` |
| `Adenoma_hp 20X` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_isyntax_preprocess_20x/patches` | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_isyntax_preprocess_20x` |

历史实验产物：hp 10 张 stain-filtered Tissue Mask 结果。以下目录只用于追溯旧实验，不作为当前标准预处理输入：

| 产物 | 路径 |
| --- | --- |
| 二值 mask | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_tissue_masks_stain_filtered/masks/*_tissue_mask.png` |
| debug 三联图 | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_tissue_masks_stain_filtered/debug/*_stain_filter_debug.png` |
| CLAM-compatible segmentation pkl | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_tissue_masks_stain_filtered/segmentations/*.pkl` |
| 处理 summary | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_tissue_masks_stain_filtered/tissue_mask_summary.csv` |

历史实验产物：基于已有 CLAM segmentation pkl 重新做 stain refinement 的 hp 10 张结果。以下目录只用于追溯旧实验，不作为当前标准预处理输入：

| 产物 | 路径 |
| --- | --- |
| 输入 10 张 slide manifest | `/data15/data15_5/yuexin2/adenoma/data/adenoma_hp_tissue_mask_10_slides.csv` |
| CLAM + stain refined 二值 mask | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_clam_stain_refined_10/masks/*_tissue_mask_refined.png` |
| CLAM + stain refinement debug 五联图 | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_clam_stain_refined_10/debug/*_clam_stain_refine_debug.png` |
| refined CLAM-compatible segmentation pkl | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_clam_stain_refined_10/segmentations/*.pkl` |
| refinement summary | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_clam_stain_refined_10/refined_tissue_mask_summary.csv` |
| refined CLAM patch h5 | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_isyntax_preprocess_20x_stain_refined_10/patches/*.h5` |
| refined CLAM mask 可视化 | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_isyntax_preprocess_20x_stain_refined_10/masks/*.jpg` |
| refined CLAM process list | `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_isyntax_preprocess_20x_stain_refined_10/process_list_autogen_adenoma_hp_clam_stain_refined_10_process_list.csv` |

## 方法原理

WSI 图像过大，不能直接整体输入模型。预处理先在低分辨率层级使用 CLAM 做组织分割，识别组织轮廓并过滤无效白背景区域，再在 CLAM 组织轮廓内生成 patch 坐标。

当前标准不额外执行污渍去除。组织边界以 CLAM segmentation pkl 中的 `tissue` 和 `holes` contours 为准；HSV / LAB 阈值、形态学滤波和 stain refinement 只属于历史实验流程，默认不参与 patch 坐标生成。

项目约定 `coords` 使用 WSI level 0 坐标系：

```text
coord = [level0_x, level0_y]
```

level 0 坐标能稳定连接 WSI、patch、特征和热图，是后续不同倍率路线可比较的基础。

## 倍率标准化规则

`Adenoma_hp` 和 `Adenoma_yx` 的 WSI 扫描基准倍率均按 40x 处理。预处理生成 1x、2.5x、5x、10x、20x patch h5 时，必须严格对应目标物理倍率，不能用 WSI pyramid 的 level index 或临时 downsample 作为倍率定义。

统一规则：

```text
base_magnification = 40
scale_to_level0 = base_magnification / target_magnification
physical_level_0_extent = physical_extent_patch_size * scale_to_level0
```

以 `physical_extent_patch_size = 256` 为例：

| h5 目标倍率 | `scale_to_level0` | 每个 patch 对应 level 0 物理范围 |
| --- | ---: | ---: |
| 20x | 2 | 512 x 512 |
| 10x | 4 | 1024 x 1024 |
| 5x | 8 | 2048 x 2048 |
| 2.5x | 16 | 4096 x 4096 |
| 1x | 40 | 10240 x 10240 |

hp 和 yx 的处理都遵循这套换算逻辑：目标倍率不是直接由 WSI pyramid 的原生 `downsample` 层级命名，而是先按 40x 扫描基准换算成 level 0 物理视野，再在组织轮廓内生成 level 0 坐标。换句话说，低倍率 patch h5 不是把 20x patch 缩放得到，也不是简单复用某个 WSI 金字塔层；其倍率语义由 `physical_level_0_extent` 决定。

当前 hp 本地 patch h5 attrs 已记录该规则：

```text
magnification_rule = physical_level0_extent = physical_extent_patch_size * base_magnification / target_magnification
base_magnification = 40
physical_extent_patch_size = 256
```

对应目标倍率：

| 数据 | 目标倍率 | `target_magnification` | `physical_level_0_extent` | `scale_to_level0` |
| --- | ---: | ---: | ---: | ---: |
| hp / yx | 1x | 1 | 10240 | 40 |
| hp / yx | 2.5x | 2.5 | 4096 | 16 |
| hp / yx | 5x | 5 | 2048 | 8 |
| hp / yx | 10x | 10 | 1024 | 4 |
| hp / yx | 20x | 20 | 512 | 2 |

实现时仍可能选择一个 WSI pyramid level 作为读取来源或坐标生成辅助层。以 hp 样本 `0001b5bb-4953-4618-bb32-3019cd09f464` 的本地 patch h5 为例：

| 目标倍率 | `patch_level` | WSI 原生/估计 downsample | level 0 物理视野 |
| --- | ---: | ---: | ---: |
| 1x | 5 | 约 32 | 10240 |
| 2.5x | 4 | 约 16 | 4096 |
| 5x | 3 | 约 8 | 2048 |
| 10x | 2 | 4 | 1024 |
| 20x | 1 | 2 | 512 |

因此 hp 更准确的描述是：使用 WSI 金字塔层作为读取来源或 provenance，但倍率定义通过 40x 基准换算得到。yx 同样按该公式定义目标倍率；即使某些 `.svs` 不存在原生 `downsample=2` 或其他目标层，也不能把最接近的金字塔层直接当作最终倍率，应以 level 0 物理视野和后续 resize 作为统一语义。

实现约束：

- `coords` 始终保存 level 0 左上角坐标。
- h5 attrs 必须记录 `base_magnification`、`target_magnification`、`physical_extent_patch_size`、`physical_level_0_extent`。
- `patch_level`、`level_downsample`、`source_level_downsample` 只能作为历史兼容或读取诊断信息，不能决定目标倍率。
- 如果底层读取接口必须选择某个 pyramid level，应先按目标倍率确定 level 0 物理范围，再选择最合适的读取层并 resize；最终 h5 的语义仍以 `physical_level_0_extent` 为准。
