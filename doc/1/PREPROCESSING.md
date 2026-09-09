# adenoma 预处理流程

这份文档描述的是当前 `adenoma` 项目里已经落地的 CLAM 预处理方案。

目前支持两类输入：

- `Adenoma_yx`：`/data15/zhengke_usb/Adenoma_yx/*.svs`
- `Adenoma_hp`：`/data15/zhengke_usb/Adenoma_hp/*.isyntax`

其中：

- `.svs` 默认使用 `clam_latest` 环境
- `.isyntax` 需要使用安装了 `pyisyntax` 的 `mag-gltrans` 环境
- `.isyntax` 读取已经接入 `CLAM/wsi_core/WholeSlideImage.py` 的兼容层，预处理阶段可以直接跑，不需要先转成 `.svs`

## 目标

- 输入数据：`/data15/zhengke_usb/Adenoma_yx/*.svs`
- 预处理工具：`CLAM/create_patches_fp.py`
- 当前处理内容：
  - 组织区域分割
  - patch 坐标提取
  - 生成掩膜图
- 当前不做的内容：
  - `stitch` 拼图可视化，默认关闭
  - 特征提取和训练，属于下一阶段

说明：

- `Adenoma_yx` 仍然是默认示例数据集
- `Adenoma_hp` 需要通过单独的配置文件切换

## 当前目录结构

项目根目录：`/data15/data15_5/yuexin2/adenoma`

关键文件和目录：

- 配置文件：
  - `.../config/adenoma_yx_preprocess.env`
  - `.../config/adenoma_hp_isyntax_preprocess.env`
- 说明文档：`/data15/data15_5/yuexin2/adenoma/PREPROCESSING.md`
- 清单生成脚本：`/data15/data15_5/yuexin2/adenoma/scripts/build_slide_manifest.sh`
- 烟雾测试输入脚本：`/data15/data15_5/yuexin2/adenoma/scripts/make_smoke_input.sh`
- 预处理启动脚本：`/data15/data15_5/yuexin2/adenoma/scripts/run_preprocess.sh`
- 状态查看脚本：`/data15/data15_5/yuexin2/adenoma/scripts/check_preprocess_status.sh`
- 数据清单目录：`/data15/data15_5/yuexin2/adenoma/data`
- 预处理输出目录：`/data15/data15_5/yuexin2/adenoma/runs`
- 日志目录：`/data15/data15_5/yuexin2/adenoma/logs`

外部依赖路径：

- CLAM 仓库：`/data15/data15_5/yuexin2/CLAM`
- `.svs` 环境：`/data15/data15_5/yuexin2/anaconda3/envs/clam_latest/bin/python`
- `.isyntax` 环境：`/data15/data15_5/yuexin2/anaconda3/envs/mag-gltrans/bin/python`

## 配置切换

默认配置还是 `Adenoma_yx (.svs)`：

```bash
export CONFIG_PATH=/data15/data15_5/yuexin2/adenoma/config/adenoma_yx_preprocess.env
```

如果要处理 `Adenoma_hp (.isyntax)`，先切到：

```bash
export CONFIG_PATH=/data15/data15_5/yuexin2/adenoma/config/adenoma_hp_isyntax_preprocess.env
```

下面第 1-4 步里的命令都可以在设置好 `CONFIG_PATH` 后直接复用。

## 当前参数

当前默认参数来自
`/data15/data15_5/yuexin2/adenoma/config/adenoma_yx_preprocess.env`：

- `PATCH_SIZE=256`
- `STEP_SIZE=256`
- `PATCH_LEVEL=0`
- `ENABLE_SEG=1`
- `ENABLE_PATCH=1`
- `ENABLE_STITCH=0`

对应到 CLAM 命令大致等价于：

```bash
/data15/data15_5/yuexin2/anaconda3/envs/clam_latest/bin/python \
  /data15/data15_5/yuexin2/CLAM/create_patches_fp.py \
  --source /data15/zhengke_usb/Adenoma_yx \
  --save_dir /data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess \
  --patch_size 256 \
  --step_size 256 \
  --patch_level 0 \
  --seg \
  --patch
```

当前这套流程没有传 `--preset`，所以使用的是 `CLAM/create_patches_fp.py`
内置默认参数。

## 参数明细

下面把当前实际生效的参数拆成 3 层说明：

- `adenoma` 项目层参数
- `create_patches_fp.py` 命令行参数
- `CLAM` 脚本内部默认参数

### 1. adenoma 项目层参数

这些参数定义在
`/data15/data15_5/yuexin2/adenoma/config/adenoma_yx_preprocess.env`：

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `SOURCE_DIR` | `/data15/zhengke_usb/Adenoma_yx` | `.svs` 原始切片目录 |
| `SMOKE_SLIDE` | `138189_751666001.svs` | 烟雾测试使用的单张切片 |
| `RUN_ROOT` | `/data15/data15_5/yuexin2/adenoma/runs` | 预处理输出根目录 |
| `DATA_DIR` | `/data15/data15_5/yuexin2/adenoma/data` | 清单文件目录 |
| `LOG_DIR` | `/data15/data15_5/yuexin2/adenoma/logs` | 日志目录 |
| `WORK_DIR` | `/data15/data15_5/yuexin2/adenoma/workdirs` | 烟雾测试等临时目录 |
| `PATCH_SIZE` | `256` | patch 边长，单位像素 |
| `STEP_SIZE` | `256` | patch 步长，等于 `PATCH_SIZE` 时表示无重叠滑窗 |
| `PATCH_LEVEL` | `0` | 在金字塔第 0 层切 patch，也就是最高分辨率层 |
| `ENABLE_SEG` | `1` | 开启组织分割 |
| `ENABLE_PATCH` | `1` | 开启 patch 坐标提取 |
| `ENABLE_STITCH` | `0` | 关闭 stitched 预览图生成 |

### 2. create_patches_fp.py 命令行参数

当前 `adenoma/scripts/run_preprocess.sh` 传给 CLAM 的主要参数如下：

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `--source` | `SOURCE_DIR` | 输入切片目录 |
| `--save_dir` | `runs/adenoma_yx_preprocess` | 输出目录 |
| `--patch_size` | `256` | patch 尺寸 |
| `--step_size` | `256` | 滑窗步长 |
| `--patch_level` | `0` | patch 提取层级 |
| `--seg` | 开启 | 执行分割 |
| `--patch` | 开启 | 执行 patch 提取 |
| `--stitch` | 关闭 | 不生成 stitched 预览图 |
| `--preset` | 未设置 | 不使用预设 csv 参数模板 |
| `--process_list` | 默认未设置 | 全量跑时不指定，局部重跑时才用 |

### 3. CLAM 内部默认参数

因为当前没有传 `--preset`，所以 `create_patches_fp.py` 使用下面这些内部默认值。

#### 3.1 seg_params

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `seg_level` | `-1` | 自动选择分割层级；CLAM 会选一个较低分辨率层来做组织分割 |
| `sthresh` | `8` | 分割阈值相关参数 |
| `mthresh` | `7` | 中值滤波核大小相关参数 |
| `close` | `4` | 形态学闭运算强度，用于连通组织区域 |
| `use_otsu` | `False` | 不使用 Otsu 自动阈值 |
| `keep_ids` | `none` | 不强制保留指定轮廓 |
| `exclude_ids` | `none` | 不强制排除指定轮廓 |

说明：

- `seg_level=-1` 不是“禁用分割”，而是“让 CLAM 自动选层”
- 自动选层逻辑会尝试找接近 `64x downsample` 的层级

#### 3.2 filter_params

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `a_t` | `100` | 最小组织区域面积阈值 |
| `a_h` | `16` | 最小空洞面积阈值 |
| `max_n_holes` | `8` | 每个组织区域最多保留的空洞数 |

这些参数决定哪些区域会被当作有效组织、哪些小孔洞会被忽略。

#### 3.3 vis_params

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `vis_level` | `-1` | 自动选择可视化层级 |
| `line_thickness` | `250` | 轮廓可视化线宽 |

说明：

- `masks/*.jpg` 就是基于这组参数输出的可视化结果
- `vis_level=-1` 同样表示自动选层，不是关闭可视化

#### 3.4 patch_params

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `use_padding` | `True` | 边缘区域允许 padding，减少漏掉边界 patch |
| `contour_fn` | `four_pt` | 使用 CLAM 的 `four_pt` 轮廓判定方式 |

#### 3.5 运行行为参数

除了上面这些显式参数，当前流程还有几个默认运行行为：

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `save_mask` | `True` | 总是保存分割掩膜图 |
| `use_default_params` | `False` | 优先按 dataframe/当前参数字典组织参数，而不是走简化默认分支 |
| `auto_skip` | `True` | 如果某张切片的 `patches/<slide>.h5` 已经存在，就自动跳过 |
| `process_list` | `None` | 默认处理整个目录中的 `.svs` |

这也是为什么当前流程支持断点续跑：已经产出 `.h5` 的切片，后续再次运行时通常会被跳过。

## 参数怎么改

最常改的是外层这几个参数：

- 想改 patch 大小：修改 `PATCH_SIZE`
- 想改 patch 重叠：把 `STEP_SIZE` 调小到小于 `PATCH_SIZE`
- 想打开 stitched 预览：把 `ENABLE_STITCH=1`
- 想临时开 stitched：运行时加 `--stitch`
- 想只跑少量切片：配合 `--process-list`

几个常见例子：

### 例 1：改成 50% overlap

把配置文件里的：

```text
PATCH_SIZE=256
STEP_SIZE=128
```

这表示 patch 还是 `256x256`，但每次滑动 `128` 像素，会产生 50% 重叠。

### 例 2：打开 stitched 预览图

改配置：

```text
ENABLE_STITCH=1
```

或者临时运行：

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_preprocess.sh --stitch
```

这样 `runs/.../stitches` 目录里就会生成拼接预览图，但会增加运行时间和存储开销。

### 例 3：只处理一个子集

准备一个筛选过的 `process_list` 后运行：

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_preprocess.sh \
  --process-list process_list_subset.csv
```

更稳的做法是把这个 `csv` 放到对应的 `save_dir` 目录下，再交给 CLAM 读取。

## 第 1 步：生成切片清单

先把 `.svs` 文件整理成后续可复用的清单。

运行：

```bash
CONFIG_PATH=${CONFIG_PATH:-/data15/data15_5/yuexin2/adenoma/config/adenoma_yx_preprocess.env} \
/data15/data15_5/yuexin2/adenoma/scripts/build_slide_manifest.sh
```

会生成两份文件：

- `data/<manifest_prefix>_manifest.csv`
- `data/<manifest_prefix>_feature_input.csv`

它们的用途分别是：

- `adenoma_yx_manifest.csv`
  - 记录 `slide_id`、原始文件名、完整路径
  - 适合人工核对和后续自定义数据整理
- `adenoma_yx_feature_input.csv`
  - 只有一列 `slide_id`
  - 后续 `extract_features_fp.py` 时可以直接复用

## 第 2 步：先做一张切片的烟雾测试

先跑单张切片，确认 OpenSlide、CLAM 和当前参数能正常工作。

1. 准备单张测试输入：

```bash
CONFIG_PATH=${CONFIG_PATH:-/data15/data15_5/yuexin2/adenoma/config/adenoma_yx_preprocess.env} \
/data15/data15_5/yuexin2/adenoma/scripts/make_smoke_input.sh
```

默认会把下面这张切片链接到测试目录：

```text
/data15/zhengke_usb/Adenoma_yx/138189_751666001.svs
```

如果当前 `CONFIG_PATH` 指向 `adenoma_hp_isyntax_preprocess.env`，则会改为：

```text
/data15/zhengke_usb/Adenoma_hp/033319a0-e49d-45c9-927f-e485c83e3463.isyntax
```

2. 运行烟雾测试：

```bash
CONFIG_PATH=${CONFIG_PATH:-/data15/data15_5/yuexin2/adenoma/config/adenoma_yx_preprocess.env} \
/data15/data15_5/yuexin2/adenoma/scripts/run_preprocess.sh --smoke
```

烟雾测试输出在：

会输出到配置文件中对应的 smoke 目录，例如：

- `adenoma_yx`：`/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_smoke`
- `adenoma_hp`：`/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_isyntax_smoke`

成功时会看到：

- `masks/<slide>.jpg`
- `patches/<slide>.h5`
- `process_list_autogen.csv`

这表示：

- 组织区域分割成功
- patch 坐标已经写进 `.h5`
- 后续可以继续做特征提取

## 第 3 步：启动全量预处理

确认烟雾测试没问题之后，启动全量 `.svs` 预处理：

```bash
CONFIG_PATH=${CONFIG_PATH:-/data15/data15_5/yuexin2/adenoma/config/adenoma_yx_preprocess.env} \
/data15/data15_5/yuexin2/adenoma/scripts/run_preprocess.sh
```

默认输入目录和输出目录由当前 `CONFIG_PATH` 决定，例如：

- `adenoma_yx`：
  - source: `/data15/zhengke_usb/Adenoma_yx`
  - save: `/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess`
- `adenoma_hp`：
  - source: `/data15/zhengke_usb/Adenoma_hp`
  - save: `/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_isyntax_preprocess`

如果你想改输出目录，可以这样运行：

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_preprocess.sh \
  --save-dir /你的输出目录
```

## 第 4 步：查看运行进度

如果你把命令直接跑在自己的终端里，可以用：

```bash
tail -f /data15/data15_5/yuexin2/adenoma/logs/adenoma_yx_preprocess_live.log
```

当前常用的状态查看脚本：

```bash
CONFIG_PATH=${CONFIG_PATH:-/data15/data15_5/yuexin2/adenoma/config/adenoma_yx_preprocess.env} \
/data15/data15_5/yuexin2/adenoma/scripts/check_preprocess_status.sh
```

日志里每张切片通常会输出：

- 正在处理的文件名
- 轮廓数量
- bounding box

## 第 5 步：8 分片并行运行

如果想把全量数据切成 8 份并行跑，现在可以直接用：

```bash
CONFIG_PATH=/data15/data15_5/yuexin2/adenoma/config/adenoma_hp_isyntax_preprocess.env \
bash /data15/data15_5/yuexin2/adenoma/scripts/start_preprocess_shards.sh
```

这条脚本会自动完成：

- 重新生成 manifest
- 按文件大小把切片均衡分成 8 份
- 生成 `data/adenoma_hp_preprocess_shards/process_list_shard_0.csv` 到 `process_list_shard_7.csv`
- 启动对应 shard 的预处理任务

分片摘要保存在：

```text
/data15/data15_5/yuexin2/adenoma/data/adenoma_hp_preprocess_shards/shard_summary.json
```

每个 shard 的独立状态文件保存在共享输出目录下：

```text
/data15/data15_5/yuexin2/adenoma/runs/adenoma_hp_isyntax_preprocess/process_list_autogen_process_list_shard_<N>.csv
```

例如：

- `.../process_list_autogen_process_list_shard_0.csv`
- `.../process_list_autogen_process_list_shard_7.csv`

这些文件比总日志更适合统计每个 shard 已经完成多少张切片。

## 第 6 步：Adenoma_hp 的 UNI 特征提取

对于 `Adenoma_hp (.isyntax)`，当前推荐使用：

- 配置文件：`/data15/data15_5/yuexin2/adenoma/config/adenoma_hp_uni_20x.env`
- Python 环境：`clam_latest`
- `pyisyntax` 路径：通过 `PYISYNTAX_SITE_PACKAGES` 指向 `mag-gltrans` 环境中的 site-packages

待提特征切片会从已经存在的 `patches/*.h5` 自动生成：

```text
/data15/data15_5/yuexin2/adenoma/data/adenoma_hp_feature_input_pending.csv
```

四卡分片启动脚本：

```bash
GPU_IDS="0 5 6 7" \
CONFIG_PATH=/data15/data15_5/yuexin2/adenoma/config/adenoma_hp_uni_20x.env \
bash /data15/data15_5/yuexin2/adenoma/scripts/start_feature_shards.sh
```

会生成：

- 分片 csv：`data/adenoma_hp_feature_shards/feature_shard_0.csv` 到 `feature_shard_3.csv`
- 分片摘要：`data/adenoma_hp_feature_shards/feature_shard_summary.json`
- 输出目录：`/data15/data15_5/yuexin2/adenoma/outputs/adenoma_hp_features_uni`

特征输出结构：

- `pt_files/<slide_id>.pt`
- `h5_files/<slide_id>.h5`

查看分片状态可以用：

```bash
CONFIG_PATH=/data15/data15_5/yuexin2/adenoma/config/adenoma_hp_uni_20x.env \
bash /data15/data15_5/yuexin2/adenoma/scripts/check_feature_shards.sh
```
- 提取到的 patch 数量
- segmentation 用时
- patching 用时

## 第 5 步：理解输出结果

全量预处理的输出目录结构如下：

```text
runs/adenoma_yx_preprocess/
├── masks/
├── patches/
├── stitches/
└── process_list_autogen.csv
```

每一部分的含义：

- `masks/`
  - 每张切片的组织分割可视化图
- `patches/`
  - 每张切片一个 `.h5`
  - 当前存的是 patch 坐标，不是已经裁好的图片
- `stitches/`
  - 当前流程默认不开 `--stitch`
  - 所以这个目录通常为空
- `process_list_autogen.csv`
  - 记录每张切片的处理状态和参数
  - 可用于排查失败样本或局部重跑

## 第 6 步：断点续跑与局部重跑

CLAM 这条脚本默认会跳过已经生成过 `patches/<slide>.h5` 的切片。

这意味着：

- 如果任务中断，直接重新执行 `run_preprocess.sh` 即可
- 已经成功处理过的切片通常会被自动跳过
- 失败或未完成的切片会继续处理

如果你想只处理一个子集，可以基于 `process_list_autogen.csv` 做筛选，然后运行：

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_preprocess.sh \
  --process-list 你的csv文件名
```

注意：

- 这里传给 `--process-list` 的是 CLAM 使用的 csv 文件名
- CLAM 会把这个文件解释为相对于 `save_dir` 的文件
- 因此更稳的做法是先把要用的 csv 放到对应的 `save_dir` 目录下

## 第 7 步：预处理结束后做什么

预处理完成后，下一阶段通常是特征提取：

```bash
/data15/data15_5/yuexin2/anaconda3/envs/clam_latest/bin/python \
  /data15/data15_5/yuexin2/CLAM/extract_features_fp.py \
  --data_h5_dir /data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess \
  --data_slide_dir /data15/zhengke_usb/Adenoma_yx \
  --csv_path /data15/data15_5/yuexin2/adenoma/data/adenoma_yx_feature_input.csv \
  --feat_dir /data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features \
  --batch_size 256 \
  --slide_ext .svs
```

特征提取完成后会生成：

- `h5_files/*.h5`
- `pt_files/*.pt`

这些 `.pt` 文件才是后续 CLAM 训练最直接使用的输入。

## 注意事项

- `Adenoma_yx` 一共有很多张 `.svs`，总数据量很大，预处理会持续很久
- 当前流程不开 `--stitch`，这是为了节省时间和存储
- 当前 `CLAM` 环境已经可用，但 GPU 驱动状态不正常，所以现阶段更适合按 CPU 流程理解和推进
- 如果后续要加速特征提取或训练，建议先单独处理 GPU 驱动和 CUDA 兼容性
