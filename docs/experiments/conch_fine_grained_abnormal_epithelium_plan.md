# CONCH 细粒度异常上皮识别实验计划

## 1. 研究背景与目标

当前系统主要将 CONCH 作为低层级组织证据和 lesion prior 使用，但尚未验证其是否能够识别“已经异常、但尚未达到明确肿瘤上皮”的细粒度结直肠黏膜形态。本计划建立一个专家标注的 ROI/patch benchmark，并进一步在 WSI 上验证 CONCH lesion prior 对病变覆盖率和下游任务的实际价值。

核心研究问题如下：

1. CONCH 能否区分 `normal_epi`、`abnormal_epi_non_tumor` 和 `tumor_epi`？
2. CONCH 能否在 `abnormal_epi_non_tumor` 内进一步识别：
   - `serrated_non_TSA_like`
   - `TSA_like`
   - `tubular_like`
   - `villous_like`
3. 如果 zero-shot 性能有限，CONCH embedding 是否仍包含可由轻量 linear probe 提取的细粒度形态信息？
4. CONCH-derived lesion prior 能否在固定 patch budget 下提高 WSI 病变覆盖率和下游分类性能？

本实验首先评价 CONCH 是否具有独立于明显 `tumor_epi` 的早期异常识别能力，避免将“能识别癌或高级别异型”误解释为“能识别非肿瘤性异常上皮”。

## 2. 核心假设

### 2.1 主要假设

- H1：CONCH 的 abnormal score 在 `abnormal_epi_non_tumor` 中高于 `normal_epi`。
- H2：这种差异并非仅由高级别异型或 `tumor_epi` 驱动。
- H3：CONCH 对不同异常形态具有不同但可测量的细粒度分辨能力。
- H4：即使 zero-shot 分类阈值不够稳定，CONCH embedding 仍可能支持有效的线性分类器。

### 2.2 系统级假设

- H5：CONCH lesion prior 能够在相同 patch 数量下提高专家标注病变区域的覆盖率。
- H6：更好的 lesion-aware sampling 能够改善或保持下游分类性能，同时减少需要处理的 patch 数量。

## 3. 层级标签体系

标签必须描述 ROI 内实际可见的局部形态，不能直接将 WSI 诊断机械地赋给其中所有 patch。

### 3.1 标签层级

```text
normal_epi

abnormal_epi_non_tumor
├── serrated pathway
│   ├── serrated_non_TSA_like
│   └── TSA_like
└── conventional pathway
    ├── tubular_like
    └── villous_like

tumor_epi
non_epithelial
uncertain
```

每个 ROI 至少记录以下独立字段：

| 字段 | 推荐取值 | 用途 |
| --- | --- | --- |
| `epithelium_state` | `normal_epi`, `abnormal_epi_non_tumor`, `tumor_epi`, `non_epithelial`, `uncertain` | 核心上皮状态 |
| `pathway` | `normal`, `serrated`, `conventional`, `tumor`, `not_applicable` | 通路层级分类 |
| `morphology` | `normal`, `serrated_non_TSA_like`, `TSA_like`, `tubular_like`, `villous_like`, `mixed`, `uncertain` | 细粒度形态 |
| `dysplasia_grade` | `none`, `low`, `high`, `indeterminate` | 控制异型程度混杂 |
| `purity` | 0–1 连续值及分档 | 描述目标形态占比 |
| `challenge_flags` | 多值字段 | 标记炎症、再生、切向切片、混合形态等困难因素 |

### 3.2 叶级形态定义

- `serrated_non_TSA_like`：HP/SSL-like 锯齿状形态，包括表面锯齿、隐窝扩张、基底结构异常或水平生长等；不包含典型 TSA-like 区域。
- `TSA_like`：具有 TSA 样绒毛状或丝状结构、裂隙样锯齿、嗜酸性细胞质、细长核或异位隐窝形成等证据的区域。
- `tubular_like`：以管状、拥挤腺体为主的 conventional adenoma-like 局部形态。
- `villous_like`：存在明确绒毛或 tubulovillous 成分的局部形态。该标签不直接等同于整张 WSI 的 TVA 或 villous adenoma 诊断。
- `mixed`：同一 ROI 中存在两种或以上主要形态，且无法通过重新圈定 ROI 得到高纯度区域。

### 3.3 `tumor_epi` 边界

在正式标注前必须由病理专家冻结书面标准，至少明确：

- 高级别异型增生是否统一归入 `tumor_epi`，或作为独立 challenge group；
- 黏膜内癌和浸润癌的处理方式；
- 同一 ROI 中低级别与高级别成分并存时的标注规则；
- 明显坏死、促纤维间质反应和浸润结构是否作为辅助标志记录。

主分析应保证 `abnormal_epi_non_tumor` 不包含符合冻结标准的 `tumor_epi`。

## 4. 专家 ROI Benchmark

### 4.1 标注单位

采用“专家圈定 ROI + 同中心多尺度 patch”作为主要验证单位。WSI 弱标签不能替代 ROI 级真值。

每个 ROI 建议记录：

```text
patient_id
case_id
slide_id
roi_id
x
y
mpp
patch_size
epithelium_state
pathway
morphology
dysplasia_grade
purity
annotator
annotation_confidence
challenge_flags
split
```

### 4.2 主 benchmark 纳入标准

- ROI 包含足够的可评价结直肠上皮。
- 目标上皮占 ROI 上皮面积至少 70%。
- 目标主形态占可评价目标上皮至少 70%。
- 图像质量足以判断结构或细胞学特征。
- ROI 与其局部标签由病理专家确认。

### 4.3 排除标准

- 严重失焦、折叠、压碎、电灼或染色异常。
- 目标上皮过少。
- 无法区分的复杂混合形态。
- 标签仅来自 WSI 诊断、但 ROI 内无对应形态证据。
- 无法可靠判定 normal、abnormal non-tumor 或 tumor 边界。

### 4.4 Challenge set

以下区域不进入初始高纯度主分析，单独组成困难测试集：

- lesion boundary 或 normal-to-lesion transition；
- mixed morphology；
- 炎症、糜烂和再生性改变；
- 切向切片或腺体挤压；
- lesion 邻近正常黏膜；
- 黏液丰富或上皮比例较低的区域；
- 不同等级 dysplasia 混合区域。

### 4.5 数据切分

- 所有切分必须在 patient level 完成。
- 同一患者、同一病灶及空间相邻 ROI 不得跨 split。
- 建议设置 `prompt development`、`linear-probe train/validation` 和 `final test` 三组互斥患者。
- prompt 在 development set 冻结后，不得根据 final test 结果继续挑选或修改。

## 5. 实验任务

### Experiment 1：Normal、Abnormal 与 Tumor 对照

任务：

1. `normal_epi vs abnormal_epi_non_tumor`
2. `normal_epi vs tumor_epi`
3. `abnormal_epi_non_tumor vs tumor_epi`
4. `normal_epi vs abnormal_epi_non_tumor vs tumor_epi`

核心分析是比较三组 score distribution。若模型表现为：

```text
score(normal) ≈ score(abnormal_non_tumor) << score(tumor)
```

则只能说明 CONCH 能够识别明显肿瘤形态，不能证明其具有 early lesion prior 能力。

### Experiment 2：层级 Zero-shot 分类

依次评估：

1. `normal vs abnormal`
2. `serrated pathway vs conventional pathway`
3. `serrated_non_TSA_like vs TSA_like`
4. `tubular_like vs villous_like`
5. `normal / serrated_non_TSA_like / TSA_like / tubular_like / villous_like` 五分类

同时报告以下关键 pairwise task：

- serrated_non_TSA-like vs normal；
- tubular-like vs normal；
- villous-like vs tubular-like；
- TSA-like vs serrated_non-TSA-like；
- TSA-like vs villous-like。

层级任务是主要分析，五分类作为综合结果。这样可以区分模型失败发生在 lesion detection、pathway discrimination 还是叶级 morphology discrimination。

### Experiment 3：Prompt Ensemble

每个类别建立并版本化三组 prompt：

1. 简单诊断名称；
2. architecture-aware 结构描述；
3. diagnostic-aware 且显式声明 non-invasive/non-tumor 的描述。

示例：

```text
normal:
- normal non-neoplastic colorectal mucosa
- colorectal mucosa with preserved regularly spaced crypt architecture

serrated_non_TSA_like:
- non-invasive serrated colorectal mucosal lesion
- colorectal mucosa with serrated and abnormally shaped crypt architecture

TSA_like:
- traditional serrated adenoma-like colorectal mucosa
- colorectal mucosa with villiform serration and ectopic crypt formation

tubular_like:
- non-invasive tubular colorectal adenoma morphology
- colorectal mucosa with crowded dysplastic tubular glands

villous_like:
- non-invasive colorectal adenomatous mucosa with villous architecture
- colorectal mucosa with elongated villous or tubulovillous projections

tumor:
- invasive colorectal tumor epithelium
- colorectal epithelium with invasive malignant glandular morphology
```

每次推理保存：

- 每条 prompt 的原始 cosine similarity；
- prompt ensemble score；
- 相对 normal prompt 的 similarity margin；
- 相对 tumor prompt 的 similarity margin；
- prompt set 与文本编码器版本。

不得只报告测试集上表现最好的单条 prompt。主结果应使用预先冻结的 prompt ensemble。

### Experiment 4：CONCH Embedding Linear Probe

比较 CONCH 与 UNI embedding，在完全相同的 ROI、尺度和 patient split 上训练：

- logistic regression，作为主要分类器；
- L2 regularization；
- class-balanced weight；
- 仅在 validation set 选择超参数；
- linear SVM 或小型 MLP 仅作为辅助分析。

任务包括：

- normal vs abnormal non-tumor；
- normal vs abnormal non-tumor vs tumor；
- serrated vs conventional pathway；
- 五类细粒度 morphology。

增加 label-efficiency curve，每类使用 5、10、25、50、100 和全部训练 ROI，使用多个随机种子重复采样。

### Experiment 5：尺度消融

从同一 ROI 中心生成不同真实视野：

- 5×：完整隐窝、锯齿和绒毛结构；
- 10×：主要腺体 architecture；
- 20×：细胞学、成熟和 dysplasia。

比较：

- 单尺度 zero-shot；
- 多尺度 score averaging；
- 多尺度 embedding concatenation；
- architecture-aware 与 cytology-aware prompt 的尺度交互。

同中心 patch 必须保持物理坐标一致，并记录目标 MPP、实际读取 level 和 resize 参数。

### Experiment 6：WSI Lesion-prior 与 Patch Efficiency

ROI benchmark 完成并冻结方法后，在 WSI 上生成连续 heatmap。比较：

1. tissue 内随机采样；
2. UNI tissue/epithelium mask；
3. CONCH coarse lesion score；
4. CONCH fine-grained zero-shot score；
5. CONCH fine-grained linear-probe score；
6. 专家 lesion mask 上限。

所有方法必须使用相同 patch budget，并分别评价所有 abnormal lesions 以及四种细粒度 morphology 的覆盖率。

下游任务可包括：

- HP/SSL；
- serrated/conventional；
- TSA-like detection；
- TA/TVA 或 tubular/villous architecture；
- 最终 WSI diagnosis。

CONCH 不进入正式 Mucosa Extractor 主路径。只有 benchmark 证明细粒度 score 稳定有效后，才单独规划 AgentFlow routing-signal 集成实验。

## 6. 评价指标与统计方案

### 6.1 ROI级指标

- AUROC；
- AUPRC；
- macro F1；
- balanced accuracy；
- sensitivity at 90% specificity；
- specificity at 90% sensitivity；
- confusion matrix；
- 每类及每患者 score distribution。

普通 accuracy 仅作为辅助指标，不作为类别不平衡任务的主要判断依据。

### 6.2 统计方案

- 置信区间使用 patient-level bootstrap 计算 95% CI。
- 同一患者的全部 ROI 必须作为一个 bootstrap unit。
- 模型或 prompt 间的差异使用配对 patient-level bootstrap。
- 多次随机采样的 linear probe 报告均值、标准差和 bootstrap CI。
- 同时报告总体结果和各 morphology、dysplasia grade、尺度、purity 分层结果。

### 6.3 WSI级指标

- lesion patch recall@K；
- lesion area coverage@K；
- 每张 WSI 至少命中一个目标 ROI 的 recall@K；
- 达到固定 lesion recall 所需 patch 数量；
- coverage-budget curve；
- patch reduction ratio；
- 推理时间；
- 下游 case-level AUROC、AUPRC 和 macro F1。

## 7. 数据到位前的工程准备

### 7.1 Manifest Schema

建立版本化 ROI manifest，至少包含：

```csv
patient_id,case_id,slide_id,roi_id,x,y,mpp,patch_size,
epithelium_state,pathway,morphology,dysplasia_grade,purity,
annotator,annotation_confidence,challenge_flags,split
```

实现 schema validator，检查缺失字段、非法标签、重复 ROI、患者泄漏和不一致层级组合。

### 7.2 Prompt Registry

使用 YAML 或 JSON 保存：

- class ID；
- prompt ID；
- prompt group；
- prompt text；
- positive/negative 配对；
- ensemble method；
- registry version。

### 7.3 Embedding Cache

每个 embedding 必须关联：

- patch/ROI ID；
- encoder name 和模型版本；
- preprocessing version；
- scale/MPP；
- patch size；
- embedding 文件或索引。

CONCH 与 UNI 使用独立 cache，避免不同预处理结果相互覆盖。

### 7.4 Evaluator

提前实现可从统一 `metadata + predictions` 输入计算以下结果的 evaluator：

- 二分类和多分类指标；
- patient-level bootstrap；
- pairwise comparison；
- confusion matrix；
- score distribution；
- calibration；
- recall@K；
- coverage-budget curve。

### 7.5 Mock-data Smoke Test

在全量 WSI 到位前，使用 mock manifest 和少量现有 patch 验证：

- 多尺度坐标读取；
- RGB、MPP 和 resize 参数；
- CONCH/UNI preprocessing；
- prompt 与类别索引映射；
- embedding cache 可重复性；
- patient-level split 无泄漏；
- evaluator 对人工构造预测的计算正确性；
- 五分类 zero-shot 和 linear-probe 端到端流程可运行。

## 8. 结果解释规则

| 实验结果 | 解释与后续策略 |
| --- | --- |
| zero-shot 和 linear probe 均表现良好 | CONCH 可直接或轻量适配后作为细粒度 semantic prior |
| zero-shot 较差，但 linear probe 良好 | CONCH embedding 含有形态信息，采用小型 lesion classifier |
| normal vs tumor 良好，但 normal vs abnormal non-tumor 较差 | CONCH 主要识别明显肿瘤/高级别异型，不支持 early lesion prior 结论 |
| tubular/TSA-like 较好，但 SSL-like 较差 | 模型可能偏向细胞异型或显著结构，缺乏 subtle crypt architecture 能力 |
| ROI分类一般，但 WSI recall@K 提升 | score 具有排序价值，可用于采样 prior，但不适合作为硬分类器 |
| zero-shot、probe 和 WSI sampling 均无收益 | 当前 CONCH、输入尺度或任务定义不适合作为该 lesion detector |

## 9. 风险与控制

- **标签轴混淆**：TSA 属于 serrated pathway，而 villous 是局部结构；通过 pathway 与 morphology 两层标签解决。
- **WSI标签污染**：只使用专家 ROI 作为 patch-level 真值。
- **tumor混杂**：单独保留 tumor 对照，并按 dysplasia grade 分层。
- **病例泄漏**：强制 patient-level split 和 bootstrap。
- **尺度不足**：使用同中心 5×、10×、20× 多尺度实验。
- **prompt overfitting**：development set 冻结 prompt，final test 只运行一次主分析。
- **mixed morphology**：从主互斥分类排除，进入 challenge 或多标签分析。
- **类别数量不足**：先完成二分类和层级任务，再解释五分类结果。

## 10. 成功标准

本项目不以单一 accuracy 阈值定义成功。支持 CONCH 具有细粒度 abnormal epithelium 能力，需要同时满足：

1. 在独立患者测试集上，`normal_epi vs abnormal_epi_non_tumor` 的表现显著高于随机，并具有稳定的 patient-level CI；
2. 该能力不能仅由 `tumor_epi` 或高级别 dysplasia 驱动；
3. 至少部分叶级形态在层级或 pairwise task 中具有可重复的区分信号；
4. 结果在冻结的 prompt ensemble、合理尺度和患者分层下保持稳定；
5. zero-shot 或 linear-probe score 在固定 WSI patch budget 下提高目标病变 recall/coverage，或在固定 recall 下减少 patch 数量；
6. 如果用于 agent，必须在下游性能不下降的前提下带来覆盖率、效率或诊断性能上的明确收益。

## 11. 当前冻结决策

- serrated 叶级类别表示 HP/SSL-like；TSA-like 单列，但二者在 pathway 层均属于 serrated。
- villous-like 只表示局部形态，不直接等同于 WSI 级 TVA 诊断。
- 主实验只使用高纯度、互斥 ROI；mixed morphology 进入次要分析。
- ROI benchmark 与 WSI 弱标签实验分阶段执行。
- 正式 Mucosa Extractor 仍以 UNI + PathPrism 为事实来源，待 benchmark 得出证据后再规划独立 CONCH 集成。
