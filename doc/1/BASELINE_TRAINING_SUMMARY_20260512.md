# Baseline Training Summary

截至 `2026-05-12`，当前项目已经完成了 `SSL vs others` 主任务的多模型 baseline 训练，并新增了 `dysplasia` 的二阶段逻辑。下面汇总训练数据规模、任务定义、已测试模型和当前效果，便于汇报。

## 1. 数据规模与标签分布

### 1.1 原始总数据

- 总样本数：`1608`
- `SSL` 总数：`266`
- `others` 总数：`1342`

### 1.2 Stage 1: `SSL vs others`

当前正式汇报口径以 `fold-5` 为主，`1x / 2.5x / 5x / 20x` 的 split 统计一致：

| Split | 总数 | SSL | others |
|---|---:|---:|---:|
| Train | 1126 | 186 | 940 |
| Val | 241 | 40 | 201 |
| Test | 241 | 40 | 201 |

说明：
- 数据来自 `adenoma/data/clam_ssl_splits_*_uni/flod-5_stats.json`
- `1x / 2.5x / 5x / 20x` 的 `SSL vs others` 数据规模一致

### 1.3 Stage 2: `dysplasia vs no_dysplasia`

二阶段只在真实 `SSL` 子集上训练和评估：

| Split | 总数 | dysplasia | no_dysplasia |
|---|---:|---:|---:|
| Train | 186 | 12 | 174 |
| Val | 40 | 3 | 37 |
| Test | 40 | 1 | 39 |

原始标签主表中的总量：
- `SSL + dysplasia`：`16`
- `SSL + no_dysplasia`：`250`

说明：
- `dysplasia` 对应 `high grade`
- `no_dysplasia` 对应 `low grade`

## 2. 新增的 dysplasia 逻辑

当前实现采用两阶段层级分类，不做联合多头训练：

1. Stage 1 在全体样本上做 `SSL vs others`
2. Stage 2 仅在真实 `SSL` 子集上做 `dysplasia vs no_dysplasia`

标签定义固定为：
- `type == Sessile serrated adenoma` 记为 `SSL`
- `grade == high` 记为 `dysplasia`
- `grade == low` 记为 `no_dysplasia`

最终三分类语义为：
- `others`
- `SSL-no-dysplasia`
- `SSL-with-dysplasia`

当前状态：
- `1x / 2.5x / 5x` 的 `CLAM-SB dysplasia` 已完成
- `20x CLAM-SB dysplasia` 已建好清单并启动过训练，但截至 `2026-05-12` 结果目录里还没有最终 `metrics.json`

## 3. 已测试模型

### 3.1 SSL 主任务（已有正式结果）

- `CLAM-SB`
  - `1x`
  - `2.5x`
  - `5x`
  - `20x`
- `TransMIL`
  - `1x`
  - `2.5x`
  - `5x`
  - `20x`
- `PatchGCN`
  - `1x`
  - `2.5x`
  - `5x`
  - `20x`
- `DS-MIL`
  - `2.5x + 5x`
  - `5x + 20x`
  - `2.5x + 20x`
- `MIST`
  - `2.5x + 5x UNI low4cat`
  - 当前新增 `hp+yx fold5` 的 11 分类实验

### 3.2 已接入但当前不作为正式结果主表

- `ABMIL`
  - 已完成代码接入与 smoke 级验证
- `hierarchical ABMIL / CLAM / TransMIL / PatchGCN / DS-MIL`
  - 已完成分层代码接入
  - 当前正式汇报建议仍以已跑完的主任务和已完成的 `CLAM-SB dysplasia` 为主

## 4. SSL 主任务效果汇总

下面重点关注 `SSL` 的 `accuracy / recall / precision / AUC`。

### 4.1 CLAM-SB

| 倍率 | Accuracy | SSL Recall | SSL Precision | SSL F1 | AUC |
|---|---:|---:|---:|---:|---:|
| 1x | 0.8797 | 0.6250 | 0.6410 | 0.6329 | 0.9187 |
| 2.5x | 0.9129 | 0.7500 | 0.7317 | 0.7407 | 0.9506 |
| 5x | 0.8880 | 0.8250 | 0.6226 | 0.7097 | 0.9443 |
| 20x | 0.8797 | 0.6250 | 0.6410 | 0.6329 | 0.9289 |

观察：
- `2.5x` 的整体效果最均衡，`accuracy` 和 `precision` 都较高
- `5x` 的 `SSL recall` 最高，为 `0.825`
- `20x` 目前并未优于 `2.5x / 5x`

### 4.2 TransMIL

| 倍率 | Accuracy | SSL Recall | SSL Precision | SSL F1 | AUC |
|---|---:|---:|---:|---:|---:|
| 1x | 0.8548 | 0.5000 | 0.5714 | 0.5333 | 0.8965 |
| 2.5x | 0.8672 | 0.7000 | 0.5833 | 0.6364 | 0.9108 |
| 5x | 0.8423 | 0.7000 | 0.5185 | 0.5957 | 0.8857 |
| 20x | 0.8755 | 0.9250 | 0.5781 | 0.7115 | 0.9368 |

观察：
- `20x` 的 `SSL recall` 最高，为 `0.925`
- 但 `20x` 的 `precision` 一般，说明假阳性更多

### 4.3 PatchGCN

| 倍率 | Accuracy | SSL Recall | SSL Precision | SSL F1 | AUC |
|---|---:|---:|---:|---:|---:|
| 1x | 0.8133 | 0.8000 | 0.4638 | 0.5872 | 0.8900 |
| 2.5x | 0.8548 | 0.8500 | 0.5397 | 0.6602 | 0.9450 |
| 5x | 0.9087 | 0.8250 | 0.6875 | 0.7500 | 0.9725 |
| 20x | 0.8672 | 0.8500 | 0.5667 | 0.6800 | 0.9409 |

观察：
- `5x PatchGCN` 是当前单尺度 baseline 里最强的一组之一
- `AUC=0.9725`，`SSL recall=0.825`，`SSL precision=0.6875`

### 4.4 DS-MIL

| 组合倍率 | Accuracy | SSL Recall | SSL Precision | SSL F1 | AUC |
|---|---:|---:|---:|---:|---:|
| 2.5x + 5x | 0.9212 | 0.7750 | 0.7561 | 0.7654 | 0.9549 |
| 5x + 20x | 0.9212 | 0.7750 | 0.7561 | 0.7654 | 0.9552 |
| 2.5x + 20x | 0.8880 | 0.7750 | 0.6327 | 0.6966 | 0.9422 |

观察：
- `DS-MIL` 是当前 `SSL` 主任务效果最好的 baseline
- 最优组合是 `5x + 20x`，`AUC=0.9552`
- `2.5x + 5x` 与 `5x + 20x` 基本持平

### 4.5 MIST 11 分类 hp+yx fold5

当前新增 `MIST` 多分类实验，用于从二分类 `SSL vs others` 扩展到 subtype + high-grade dysplasia 的 11 分类。当前执行两个多尺度组合：

- `2.5x + 5x`
- `5x + 10x`

`10x + 20x` 暂不纳入本轮实验。

类别顺序固定为：

| id | 类别 |
|---:|---|
| 0 | `ssl` |
| 1 | `hp` |
| 2 | `TSA` |
| 3 | `USA` |
| 4 | `TA` |
| 5 | `TVA` |
| 6 | `IP` |
| 7 | `ssl with highgrade dysplasia` |
| 8 | `TSA with highgrade dysplasia` |
| 9 | `TA with highgrade dysplasia` |
| 10 | `TVA with highgrade dysplasia` |

标签映射逻辑：

- `type == Sessile serrated adenoma` 且 `grade == low` -> `ssl`
- `type == Hyperplastic polyps` -> `hp`
- `type == Traditional serrated adenoma` 且 `grade == low` -> `TSA`
- `type == Unclassified serrated adenoma` -> `USA`
- `type == Tubular adenoma` 且 `grade == low` -> `TA`
- `type == Tubulovillous adenoma` 且 `grade == low` -> `TVA`
- `type == Inflammatory polyp` -> `IP`
- `grade == high` 的 `SSL / TSA / TA / TVA` 分别进入对应 `with highgrade dysplasia` 类

特征构造采用 MIST 风格的低倍 patch 中心拼接，所有组合都遵循同一个 `low4cat` 规则：

- 每个低倍 patch 形成一行实例特征
- `2.5x + 5x`: `[2.5x_UNI, 5x_TL, 5x_TR, 5x_BL, 5x_BR]`
- `5x + 10x`: `[5x_UNI, 10x_TL, 10x_TR, 10x_BL, 10x_BR]`
- 单个 UNI 特征维度为 `1024`，拼接后 `feats_size = 5120`
- 四个高倍子 patch 按低倍 patch 内四象限匹配；缺失象限用 0 向量补齐，多候选取最靠近象限中心者

本次实验使用外置统一 hp+yx 5-fold 划分：

| Split | 总数 | 说明 |
|---|---:|---|
| Train | 2433 | `/data15/zhengke_usb2/yuexin_data/splits/adenoma_uni_hp_yx_ssl_5fold/splits/flod-4-train.csv` |
| Val | 811 | `/data15/zhengke_usb2/yuexin_data/splits/adenoma_uni_hp_yx_ssl_5fold/splits/flod-4-val.csv` |
| Test | 810 | `/data15/zhengke_usb2/yuexin_data/splits/adenoma_uni_hp_yx_ssl_5fold/splits/flod-4-test.csv` |

特征与结果路径：

- MIST 特征：
  - `/data15/zhengke_usb2/yuexin_data/adenoma_feature/MIST/adenoma_uni_hp_yx_2p5x_5x_low4cat`
  - `/data15/zhengke_usb2/yuexin_data/adenoma_feature/MIST/adenoma_uni_hp_yx_5x_10x_low4cat`
- MIST 训练 manifest：
  - `2.5x + 5x`: `/data15/zhengke_usb2/yuexin_data/adenoma_feature/MIST/manifests_11class_fold5`
  - `5x + 10x`: `/data15/zhengke_usb2/yuexin_data/adenoma_feature/MIST/manifests_11class_fold5_5x_10x`
- Manifest 一致性：
  - 两个组合复用同一份 `flod-4` train/val/test split
  - 两个组合的 train/val/test manifest 中 slide 顺序和 11 分类 label 完全一致
  - 特征生成脚本自动产生的 `*_fold4_train/val/test.csv` 已被同 split、同顺序、同 label 的 11 分类 manifest 覆盖，避免后续误用二分类标签
- MIST 结果目录：
  - `2.5x + 5x`: `/data15/zhengke_usb2/yuexin_data/result/fold5_hp+yx/MIST`
  - `5x + 10x`: `/data15/zhengke_usb2/yuexin_data/result/fold5_hp+yx/MIST/5x_10x`

当前状态：

- `2.5x + 5x` 已启动 200 epoch 正式训练，使用 GPU 1，`num_classes=11`
- `5x + 10x` 特征已完成拼接并通过 manifest 一致性校验，正式训练使用 GPU 2

## 5. 当前可汇报的关键结论

### 5.1 SSL 主任务

- 当前已完成的 baseline 包括 `CLAM-SB / TransMIL / PatchGCN / DS-MIL`
- 从 `SSL recall` 看：
  - `TransMIL 20x` 最高，达到 `0.925`
  - `PatchGCN 2.5x / 20x` 和 `CLAM-SB 5x` 也较高，在 `0.825~0.85`
- 从整体平衡性看：
  - `DS-MIL 5x+20x` 和 `DS-MIL 2.5x+5x` 最好
  - 其 `accuracy=0.9212`，`SSL precision=0.7561`，`SSL recall=0.7750`
- 从单尺度结果看：
  - `PatchGCN 5x` 是当前单尺度中最强的一组
  - `CLAM-SB 2.5x` 也表现稳定

### 5.2 Dysplasia 二阶段任务

- 新增逻辑已经打通，定义明确：
  - `high grade -> dysplasia`
  - `low grade -> no_dysplasia`
- `fold-5` 的二阶段测试集非常小：
  - `dysplasia = 1`
  - `no_dysplasia = 39`
- 当前 `CLAM-SB` 已完成 `1x / 2.5x / 5x` 的二阶段训练
- `20x` 二阶段训练已启动过，但截至当前目录里还没有正式 `metrics.json`

### 5.3 二阶段样本基数提醒

- 原始 `SSL + dysplasia` 总共只有 `16` 例
- `fold-5` 测试集中真实 `dysplasia` 只有 `1` 例
- 因此二阶段结果目前更适合作为“可行性验证”，不适合做过强结论

## 6. 推荐汇报口径

建议汇报时把任务拆成两部分：

1. `SSL 主任务`
   - 强调已完成多模型、多倍率基线
   - 重点指标看 `SSL recall / precision / AUC`
   - 推荐突出：
     - `DS-MIL 5x+20x`
     - `PatchGCN 5x`
     - `CLAM-SB 2.5x`
     - `TransMIL 20x`

2. `Dysplasia 二阶段任务`
   - 强调逻辑已接入并完成 `CLAM-SB` 的多倍率初步训练
   - 同时说明当前阳性样本数偏少，尤其 `fold-5 test` 只有 `1` 个 `dysplasia`
   - 现阶段建议作为探索性结果汇报

## 7. 结果路径

### SSL 主任务结果

- `fold-5` 结果根目录：
  - `/data15/zhengke_usb2/yuexin_data/training/fold_5`

### Dysplasia 二阶段结果

- `fold-5` 结果根目录：
  - `/data15/zhengke_usb2/yuexin_data/training/dysplasia_fold_5`

---

如果后续需要，我建议再补两份表：

1. 一份只保留每个模型的“最佳倍率/最佳组合”
2. 一份专门列出 `dysplasia` 的测试样本 `slide_id` 和预测结果，方便病理复核

## 8. 四个 baseline 的算法侧重点与结果解释

下面这部分分成两层：

- **文献支持的算法差异**：来自模型原始论文/官方仓库
- **结合本项目结果的解释**：这是基于当前 `fold-5` 结果做的推断，不是论文原文结论

### 8.1 CLAM-SB：更强调“少量关键 patch”的可解释聚合

算法侧重点：

- `CLAM` 本质上是 attention-based MIL 的增强版
- 它不仅做 bag-level attention pooling，还会对高 attention 的代表性 patch 做 instance-level clustering constraint
- 这类设计更适合“病变区域相对稀疏、但局部形态很关键”的任务

文献支撑：

- CLAM 原文明确强调：它通过 attention 找到高诊断价值区域，并通过 representative regions 的 instance-level clustering 来 refine feature space
  来源：Lu et al., *Data-efficient and weakly supervised computational pathology on whole-slide images*, Nature Biomedical Engineering, 2021
  链接：https://www.nature.com/articles/s41551-020-00682-w

对你当前结果的解释：

- `CLAM-SB` 在 `2.5x` 上最均衡，在 `5x` 上 `SSL recall` 最高，说明它比较吃“中低倍的结构性线索”
- `SSL` 的判别更像是 **crypt architecture / serrated pattern / gland arrangement** 的问题，而不只是细胞级纹理问题
- 所以 `20x` 没有明显优于 `2.5x / 5x`，从算法上很合理：高倍局部纹理更细，但全局结构上下文更弱；而 CLAM 的优势恰恰是抓“少量关键区域”，如果这些区域缺少足够大尺度结构背景，收益会下降

### 8.2 TransMIL：更强调“patch 之间的全局相关性”

算法侧重点：

- `TransMIL` 的核心不是只看单个 patch，而是显式建模 patch 之间的 correlation
- 论文专门指出传统 MIL 常假设实例独立同分布，而 TransMIL 试图打破这个假设
- 它同时利用 morphological information 和 spatial information

文献支撑：

- 原文摘要明确写到：现有 MIL 往往基于 i.i.d. 假设，忽略实例相关性；TransMIL 通过 correlated MIL 和 Transformer 探索形态与空间信息
  来源：Shao et al., *TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification*, NeurIPS 2021
  链接：https://proceedings.neurips.cc/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html

对你当前结果的解释：

- `TransMIL 20x` 的 `SSL recall` 最高，达到 `0.925`
- 这说明它在高倍下很擅长把“分散但相关的阳性线索”串起来，因此不容易漏掉 `SSL`
- 但它的 `precision` 没有同步升高，说明全局相关建模在高倍下也更容易把一些“长得像 SSL 的局部结构”一起放大，导致假阳性增多
- 换句话说，**TransMIL 更像“高敏感度模型”**：擅长少漏检，但代价是更容易过报

### 8.3 PatchGCN：更强调“空间邻接关系”和局部组织拓扑

算法侧重点：

- `PatchGCN` 把 WSI 看成图：patch 是节点，邻近 patch 之间连边
- 官方仓库总结得很直接：它不是简单池化实例，而是通过 message passing 学 context-aware embeddings
- 原始 Patch-GCN 论文场景是 survival prediction，不是你这里的 `SSL vs others` 分类；你当前是在借用它的 **图结构 inductive bias**

文献支撑：

- 官方仓库对方法的总结：将 patch feature 作为节点，并基于坐标构图，用 GCN message passing 学习 context-aware embeddings
  来源：Mahmood Lab 官方仓库 `Patch-GCN`
  链接：https://github.com/mahmoodlab/Patch-GCN
- 仓库中对应论文信息：Chen et al., *Whole Slide Images are 2D Point Clouds: Context-Aware Survival Prediction using Patch-based Graph Convolutional Networks*, MICCAI 2021
  DOI：https://doi.org/10.1007/978-3-030-87237-3_33

对你当前结果的解释：

- `PatchGCN 5x` 是当前最强的单尺度 baseline 之一，`AUC=0.9725`
- 这说明在你的任务里，**patch 与 patch 之间的空间关系** 很重要，而 `5x` 正好给了图模型一个比较合适的粒度
- 如果倍率太低，图上的节点会过粗，很多关键局部结构被平均掉
- 如果倍率太高，图会更碎，局部纹理噪声和边连接的复杂性会上升
- 因此 `5x` 成为图模型的“甜点位”，这和 `SSL` 本身依赖腺体/隐窝空间结构的临床直觉也是一致的

### 8.4 DS-MIL：更强调“关键实例 + 关键实例周围关系 + 多尺度互补”

算法侧重点：

- `DSMIL` 的核心是 dual-stream MIL aggregator
- 论文中一个关键点是：它不只看 max-score instance，而是建模“其他实例与最高分实例的关系”
- 原文还强调了 pyramidal fusion / multiscale fusion

文献支撑：

- 原文摘要写得很明确：
  1. 提出 dual-stream architecture with trainable distance measurement
  2. 使用 self-supervised contrastive learning 改善 MIL 表征
  3. 采用 pyramidal fusion mechanism for multiscale WSI features
  来源：Li et al., *Dual-Stream Multiple Instance Learning Network for Whole Slide Image Classification With Self-Supervised Contrastive Learning*, CVPR 2021
  链接：https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.html

对你当前结果的解释：

- `DS-MIL` 是当前 `SSL` 主任务效果最好的 baseline
- 它优于单尺度模型，很符合论文设计目标：**先抓关键实例，再看与关键实例相关的实例，同时融合多尺度信息**
- 对 `SSL` 来说，这很重要，因为：
  - 低倍/中倍更适合看整体 serrated architecture
  - 高倍更适合看局部细节
  - 单一尺度通常只覆盖其中一部分
- `2.5x+5x` 和 `5x+20x` 基本持平，说明只要是“一个更偏结构的尺度 + 一个更偏细节的尺度”，都会带来明显增益
- `2.5x+20x` 比另外两组稍差，我更倾向于把它解释为：**2.5x 和 20x 的尺度跨度太大，而 5x 在这里像一个更自然的桥接尺度**
- 这点是基于你当前结果的推断，不是论文原文直接结论

## 9. 从当前结果可以提炼出的算法层面结论

### 9.1 你的 `SSL` 任务更像“结构驱动”而不是“纯细节驱动”

证据：

- `CLAM-SB` 最稳的是 `2.5x / 5x`
- `PatchGCN` 最强单尺度在 `5x`
- `DS-MIL` 的最佳组合也都包含 `5x`

解释：

- 如果任务主要由细胞级局部纹理驱动，`20x` 应该更稳定地拉开差距
- 但你的结果里 `20x` 并没有在所有模型上都占优
- 这更像是在说明：`SSL` 的核心诊断信号里，**中尺度的腺体结构和空间排布** 占比更高

### 9.2 不同模型的 precision-recall 取向很不一样

- `TransMIL 20x`：高 recall，适合“宁可多报、尽量少漏”
- `DS-MIL`：precision / recall 更均衡，适合主汇报模型
- `PatchGCN 5x`：单尺度最强，说明图结构对这个任务非常有帮助
- `CLAM-SB 2.5x`：作为经典 MIL baseline，很稳，也比较好解释

### 9.3 二阶段 dysplasia 结果目前不能过度解读

虽然 `1x / 2.5x / 5x` 的 `CLAM-SB dysplasia` 都已经跑完，但从算法上目前只能给出很保守的解释：

- `fold-5 test` 只有 `1` 个真实 `dysplasia`
- 因此当前二阶段结果更多是在说明“流程可跑、模型能学到东西”
- 还不足以说明“某个倍率或某个算法对 dysplasia 明显更优”

## 10. 建议你汇报时怎么讲

如果是口头汇报，我建议把“算法差异”讲成下面这四句话：

1. `CLAM-SB` 更像经典、可解释的 attention MIL，重点找少量关键病灶 patch。
2. `TransMIL` 更像全局关系建模器，能把远距离相关区域串起来，因此 recall 往往更高。
3. `PatchGCN` 更强调空间拓扑和局部组织结构，适合抓腺体/隐窝结构性模式。
4. `DS-MIL` 同时抓关键实例关系和多尺度互补，所以在当前任务上整体最平衡、最好。

如果再加一句总括，可以说：

> 当前结果提示，`SSL` 更依赖中尺度结构信息；`5x` 是最稳定的关键尺度，而多尺度融合（尤其包含 `5x`）最有优势。

## 11. 参考文献

1. Lu MY, Williamson DFK, Chen TY, et al. *Data-efficient and weakly supervised computational pathology on whole-slide images*. Nature Biomedical Engineering, 2021.
   https://www.nature.com/articles/s41551-020-00682-w

2. Shao Z, Bian H, Chen Y, et al. *TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification*. NeurIPS, 2021.
   https://proceedings.neurips.cc/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html

3. Li B, Li Y, Eliceiri KW. *Dual-Stream Multiple Instance Learning Network for Whole Slide Image Classification With Self-Supervised Contrastive Learning*. CVPR, 2021.
   https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.html

4. Mahmood Lab. *Patch-GCN: Context-Aware Survival Prediction using Patch-based Graph Convolutional Networks*. Official repository.
   https://github.com/mahmoodlab/Patch-GCN
