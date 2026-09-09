# Codex Project Memory

本文件用于让 Codex 在本项目中长期记住课题背景、目录约定、默认工作偏好和传统计算病理的基本要求。

## 1. 项目目标

- 这是一个围绕腺瘤相关病理图像的计算病理课题项目。
- 主要工作包括：
  - WSI 预处理
  - 组织分割与 patch 提取
  - patch / slide 特征提取
  - MIL / Transformer / 多倍率建模
  - 结果评估、可解释性分析和实验记录
- 当前项目重点是先把 `Adenoma_yx` 的 `.svs` 数据链路跑通，再逐步接入更复杂模型。

## 2. 默认交流与协作规则

- 默认使用中文回答，必要时可保留英文模型名、命令和技术术语。
- 优先直接落地到代码、脚本、配置和文档，不只停留在口头建议。
- 在做较大改动前，先检查本地现有结构和脚本，尽量复用已有工作。
- 尽量把项目自定义逻辑写在 `adenoma/` 下，不优先修改上游仓库。
- 若必须改上游仓库，优先做最小修改，并在文档中记录原因。
- 每次新增重要流程、脚本、实验路线时，更新相应 markdown 文档。

## 3. 当前项目根目录与关键路径

- 项目根目录：`/data15/data15_5/yuexin2/adenoma`
- CLAM 仓库：`/data15/data15_5/yuexin2/CLAM`
- Patho-R1 仓库：`/data15/data15_5/yuexin2/Patho-R1`
- MAG-GLTrans 仓库：`/data15/data15_5/yuexin2/MAG-GLTrans`

数据路径：

- `.svs` 数据：`/data15/zhengke_usb/Adenoma_yx`
- `.isyntax` 数据：`/data15/zhengke_usb/Adenoma_hp`

环境路径：

- CLAM 环境：`/data15/data15_5/yuexin2/anaconda3/envs/clam_latest`
- Patho-R1 环境：`/data15/data15_5/yuexin2/anaconda3/envs/patho-r1`
- MAG-GLTrans 环境：`/data15/data15_5/yuexin2/anaconda3/envs/mag-gltrans`

## 4. 当前模型定位

### CLAM

- 用于主前处理链路：
  - tissue segmentation
  - patch coordinate extraction
  - 20X 特征提取
- 当前是项目最核心的基础设施。
- 当前首个正式训练闭环固定为：
  - `CLAM MIL @20X`
  - `SSL vs Others`
  - `fold-0`

### Patho-R1 / Patho-CLIP

- 用于辅助预处理和 patch 级分析：
  - patch 语义解释
  - patch 质控
  - 零样本筛选
- 不替代 CLAM 的 patching 主流程。
- 当前 Route C 暂停推进：
  - 保留脚本与产物
  - 原因是 CPU 上真实 `Patho-R1-3B` 推理过慢
  - 后续换到 GPU 机器再恢复

### MAG-GLTrans

- 用于后半段建模：
  - GLTrans 基线
  - 多倍率建模
  - MAG alignment
- 当前推荐优先走 Route A：
  - 先做 `20X` 基线
  - 再扩展到 `10X / 5X / 10X-A / 5X-A`

## 5. 传统计算病理的基本要求

### 5.1 数据与任务定义

- 在开始训练前，必须明确：
  - 任务类型：二分类、多分类、分层、预后、检索或质控
  - 样本单位：patient-level、slide-level 或 patch-level
  - 标签来源：病理报告、人工复核、数据库字段或研究定义
- 所有标签字段都应明确记录，不允许使用含糊类别名。
- 若任务最终是 slide / patient 级，patch 只作为中间表示，不能把 patch 当成独立样本泄漏到不同数据集。

### 5.2 数据划分原则

- 默认按 `patient-level` 划分训练、验证、测试集。
- 同一患者的多张切片不能分到不同 split。
- 在写 fold 或 split 文件时，必须明确：
  - 划分单位
  - 随机种子
  - 各类样本分布
- 若暂时只能用 slide-level 划分，必须在文档中显式注明风险。

### 5.3 WSI 预处理原则

- 预处理必须保留 patch 坐标与原始 WSI 的映射关系。
- 默认保留：
  - 原始 slide id
  - patch 坐标
  - patch level
  - patch size
  - patch step size
- 对于异常切片，要记录原因，例如：
  - 无有效组织
  - 打开失败
  - 层级异常
  - 坐标为空
- 需要尽量保留 `process_list` 或等价日志，方便断点续跑和回溯。

### 5.4 病理质量控制

- 传统计算病理中，必须关注以下常见问题：
  - 大面积背景
  - 模糊 / 失焦
  - tissue fold
  - pen mark
  - stain artifact
  - mucus / blood / debris 主导区域
  - 几乎无组织 patch
- 在项目中，优先把这些问题做成：
  - 显式过滤规则
  - patch 级质控脚本
  - 或模型辅助筛选流程

### 5.5 特征提取要求

- 每次特征提取都要明确记录：
  - encoder 名称
  - 权重来源
  - patch 输入尺寸
  - 实际 target size
  - 特征维度
  - 归一化方式
- 不允许只保留 `.pt` 或 `.npy` 而不记录对应的 encoder 信息。
- 若使用随机初始化只为验证流程，必须明确标记为 smoke / debug，不可当作正式实验结果。

### 5.6 训练与评估要求

- 默认至少报告：
  - AUC
  - Accuracy
  - F1
  - Sensitivity / Recall
  - Specificity
- 多分类任务至少报告：
  - macro-F1
  - confusion matrix
- 任何正式结果都应注明：
  - 数据划分方式
  - 样本量
  - 使用的倍率
  - encoder / backbone
  - 随机种子
- 如果结果只来自 smoke test 或 debug run，必须明确写出来。

### 5.7 可解释性与误差分析

- 传统计算病理项目中，尽量保留以下分析入口：
  - heatmap
  - top attention patches
  - 误分类样本列表
  - patch 级质控结果
- 对失败样本，优先做：
  - 类别混淆分析
  - 组织质量分析
  - 染色 / 扫描质量分析

### 5.8 可复现性

- 每条完整实验链路都应尽量固定：
  - 配置文件
  - 命令
  - 输出路径
  - 日志路径
- 默认不要把实验参数只藏在命令历史里。
- 重要实验尽量落成：
  - `.env`
  - `.yaml`
  - `.md`
  - 可复用脚本

## 6. 当前项目优先级

当前优先级从高到低：

1. 跑通 `Adenoma_yx` 的 CLAM 主前处理
2. 完成 20X 特征提取
3. 打通 `CLAM .pt/.h5 -> MAG-GLTrans .npy`
4. 生成训练可用的 ready manifest 与 cleaned split
5. 先跑 `CLAM MIL @20X` 的 `fold-0` 基线
6. 再跑 `CLAM MIL @2.5X` 的 low-mag 路线
7. 最后再恢复 Patho-R1 与扩展到 MAG alignment、多倍率和更复杂实验

## 7. 当前默认研究路线

### Route A

- 先跑通 `20X` 基线
- 当前首个任务固定为：`type = SSL / others` 二分类
- 默认正类定义：`SSL = Sessile serrated adenoma`
- 暂不依赖 MAG 对齐权重
- 先验证数据链路、标签、split、训练配置是否正确

### Route B

- 当前低倍路线固定为：
  - `2.5X low-mag`
  - 用作当前项目里的 `5X` 代理路线

### Route C

- Patho-R1 脚本与结果保留
- 当前不纳入正式主线
- 换到 GPU 机器后再恢复

## 8. 文件组织偏好

- `config/`：项目配置
- `scripts/`：项目脚本
- `data/`：清单、标签、split、fold
- `runs/`：预处理输出
- `outputs/`：特征、模型输入、中间产物
- `logs/`：日志
- `models/`：本地权重或模型缓存路径
- `workdirs/`：临时工作目录

## 9. 对 Codex 的具体要求

- 默认优先：
  - 检查现有脚本
  - 复用已有数据结构
  - 增量扩展
- 在病理课题里，不要把“流程 smoke 跑通”和“正式实验结论”混为一谈。
- 如果缺少关键外部资产，必须明确指出，例如：
  - 预训练权重
  - 标签 csv
  - fold 文件
  - Hugging Face 模型
- 如果当前机器不适合 GPU 训练，要明确说明，不要假装能高效训练。
- 每次推进一个新阶段时，尽量留下：
  - 脚本
  - 配置
  - markdown 说明

## 10. 当前已存在的重要文档

- `README.md`
- `PREPROCESSING.md`
- `MAG_WORKFLOW_PLAN.md`

如果后续路线、路径、模型角色发生变化，应同步更新这些文档和本文件。
