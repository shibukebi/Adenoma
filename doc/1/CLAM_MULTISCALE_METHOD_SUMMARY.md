# CLAM 多倍率方法汇总（当前汇报口径）

## 1. 当前推荐汇报口径

- 当前建议对师姐汇报的主口径是：
  - `UNI 特征`
  - `CLAM-SB`
  - `SSL vs others`
  - `slide-level fold-5`
- 数据来源：`/data15/zhengke_usb/Adenoma_yx`
- 标签文件：`/data15/data15_5/yuexin2/adenoma/data/adenoma_yx_ssl_others_labels.csv`
- 训练环境：`/data15/data15_5/yuexin2/anaconda3/envs/clam_latest`

## 2. 任务定义

- 任务：`SSL vs others` 二分类
- 正类定义：`SSL = Sessile serrated adenoma`
- 负类定义：`others = 其余全部类型`
- 当前正式模型主线：
  - `20X UNI + CLAM-SB`
  - `2.5X UNI + CLAM-SB`
  - `5X UNI + CLAM-SB`
- 当前对比分析重点：
  - 不同倍率下同一 `CLAM-SB` 框架的效果差异
  - 低倍视野对结构性病理特征的保留作用

## 3. 预处理方法

- 预处理主程序：`CLAM/create_patches_fp.py`
- 作用：
  - 在低分辨率层级完成组织区域分割
  - 基于组织轮廓生成 patch 坐标
  - 输出 `mask jpg + coords h5`
- 通用设置：
  - `patch_size = 256`
  - `step_size = 256`
  - `ENABLE_SEG = 1`
  - `ENABLE_PATCH = 1`
  - `ENABLE_STITCH = 0`
- 20X 路线：
  - 配置：`/data15/data15_5/yuexin2/adenoma/config/adenoma_yx_preprocess.env`
  - `patch_level = 0`
  - 输出目录：`/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess`
- 2.5X 路线：
  - 配置：`/data15/data15_5/yuexin2/adenoma/config/route_lowmag_2p5x.env`
  - `patch_level = 2`
  - 输出目录：`/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess_2p5x`
- 5X 路线：
  - 配置：`/data15/data15_5/yuexin2/adenoma/config/route_lowmag_5x.env`
  - `patch_level = 2`
  - 输出目录：`/data15/data15_5/yuexin2/adenoma/runs/adenoma_yx_preprocess_5x`

## 4. 特征提取方法

- 特征提取脚本：`/data15/data15_5/yuexin2/adenoma/scripts/extract_features_route_a.py`
- 当前主汇报编码器：`UNI`
- 通用编码器设置：
  - `FEATURE_MODEL_NAME = uni_v1`
  - `FEATURE_INIT_MODE = pretrained`
  - 权重：`/data15/data15_5/yuexin2/adenoma/models/UNI/pytorch_model.bin`
  - `UNI_HF_REPO = MahmoodLab/UNI`
  - `FEATURE_BATCH_SIZE = 64`
  - `FEATURE_TARGET_PATCH_SIZE = 224`
- 输出格式：
  - `h5_files/*.h5`
  - `pt_files/*.pt`
- 20X 特征目录：`/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_uni`
- 2.5X 特征目录：`/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_2p5x_uni`
- 5X 特征目录：`/data15/data15_5/yuexin2/adenoma/outputs/adenoma_yx_features_5x_uni`

## 5. CLAM-SB 训练参数

- 训练主程序：`/data15/data15_5/yuexin2/adenoma/scripts/train_clam_ssl_20x.py`
- `2.5X / 5X` 训练入口是对同一主程序的包装：
  - `train_clam_ssl_2p5x.py`
  - `train_clam_ssl_5x.py`
- 通用 CLAM-SB 设置：
  - `MODEL_TYPE = clam_sb`
  - `BAG_LOSS = ce`
  - `MODEL_SIZE = small`
  - `EMBED_DIM = 1024`
  - `DROPOUT = 0.25`
  - `LR = 1e-4`
  - `REG = 1e-5`
  - `MAX_EPOCHS = 100`
  - `SEED = 2023`
  - `WEIGHTED_SAMPLE = 1`
  - `EARLY_STOPPING = 1`
  - `K_SAMPLE = 8`
  - `INST_LOSS = ce`
  - `BAG_WEIGHT = 0.7`
  - `NO_INST_CLUSTER = 0`
- 当前正式配置文件：
  - 20X：`/data15/data15_5/yuexin2/adenoma/config/clam_ssl_20x_clam_sb.env`
  - 2.5X：`/data15/data15_5/yuexin2/adenoma/config/clam_ssl_2p5x_clam_sb.env`
  - 5X：`/data15/data15_5/yuexin2/adenoma/config/clam_ssl_5x_clam_sb.env`

## 6. 当前数据划分与样本量（fold-5）

- 20X `fold-5`：
  - `train = 1126`
  - `val = 241`
  - `test = 241`
  - `ssl_train = 186`
  - `ssl_val = 40`
  - `ssl_test = 40`
- 2.5X `fold-5`：
  - `train = 1126`
  - `val = 241`
  - `test = 241`
  - `ssl_train = 186`
  - `ssl_val = 40`
  - `ssl_test = 40`
- 5X `fold-5`：
  - `train = 1126`
  - `val = 241`
  - `test = 241`
  - `ssl_train = 186`
  - `ssl_val = 40`
  - `ssl_test = 40`

## 7. 当前结果概览（UNI + CLAM-SB + fold-5）

- 20X UNI CLAM-SB：
  - 结果目录：`/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_20x_uni_clam_sb/fold-5_rerun_gpu0_20260424`
  - `AUC = 0.9289`
  - `Accuracy = 0.8797`
  - `SSL precision = 0.6410`
  - `SSL recall = 0.6250`
  - `SSL F1 = 0.6329`
  - `total_training_time_sec = 1672.38`
- 2.5X UNI CLAM-SB：
  - 结果目录：`/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_2p5x_uni_clam_sb/fold-5`
  - `AUC = 0.9506`
  - `Accuracy = 0.9129`
  - `SSL precision = 0.7317`
  - `SSL recall = 0.7500`
  - `SSL F1 = 0.7407`
  - `total_training_time_sec = 473.59`
- 5X UNI CLAM-SB：
  - 结果目录：`/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_5x_uni_clam_sb/fold-5`
  - `AUC = 0.9443`
  - `Accuracy = 0.8880`
  - `SSL precision = 0.6226`
  - `SSL recall = 0.8250`
  - `SSL F1 = 0.7097`
  - `total_training_time_sec = 483.82`

## 8. 当前结论

- 在当前 `UNI + CLAM-SB + fold-5` 设置下，低倍路线整体优于高倍路线：
  - `2.5X` 的综合表现最佳
  - `5X` 次之
  - `20X` 相对较弱
- 一个合理解释是：
  - 该任务更依赖腺体 / 隐窝等架构性特征
  - 低倍 patch 提供了更大的形态学上下文
  - 在当前不显式建模 patch 空间关系的 MIL / CLAM 框架下，低倍更容易保留对 `SSL` 有用的结构信息

## 9. 历史结果说明

- 当前仓库里还保留了一套较早的 `fold-0` 结果与图表，用于早期探索：
  - 包括 `resnet50_trunc` 特征和部分旧版非 UNI 输出
- 这些历史结果可以作为探索性参考，但**不建议作为当前正式汇报主结果口径**
- 当前正式主口径建议统一使用：
  - `UNI 特征`
  - `CLAM-SB`
  - `fold-5`

## 10. 当前可直接展示的图表

- 当前 `fold-5` 结果文件：
  - 20X：`/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_20x_uni_clam_sb/fold-5_rerun_gpu0_20260424/metrics.json`
  - 2.5X：`/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_2p5x_uni_clam_sb/fold-5/metrics.json`
  - 5X：`/data15/data15_5/yuexin2/adenoma/outputs/clam_ssl_vs_others_5x_uni_clam_sb/fold-5/metrics.json`
- 当前仓库中已经现成可展示的图主要还是早期 `fold-0` 图：
  - `MIL vs CLAM-SB` ROC 对比：
    - `/data15/data15_5/yuexin2/adenoma/outputs/roc_comparisons/fold0_mil_vs_clamsb_2p5x_20x/roc_comparison_mil_vs_clamsb_2p5x_20x_fold0.pdf`
  - `CLAM-SB 2.5X / 5X / 20X` ROC 对比：
    - `/data15/data15_5/yuexin2/adenoma/outputs/roc_comparisons/fold0_clamsb_2p5x_5x_20x/roc_comparison_clamsb_2p5x_5x_20x_fold0.pdf`
  - `CLAM-SB 2.5X / 5X / 20X` AUC 柱状图：
    - `/data15/data15_5/yuexin2/adenoma/outputs/roc_comparisons/bar_clamsb_2p5x_5x_20x/auc_bar_clamsb_2p5x_5x_20x_fold0.pdf`
