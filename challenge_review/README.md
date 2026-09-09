# Challenge Set WSI Review

局域网病理切片审核平台，覆盖 3,153 张 challenge slides，并支持筛选 193 张 14/14 配置均误判病例。

## 初始化

```bash
cd /data15/data15_5/yuexin2/adenoma
challenge_review/.venv/bin/python -m challenge_review.import_data --import-all
challenge_review/.venv/bin/python -m challenge_review.manage_users create admin --role admin
```

预测文件更新后，可仅刷新第一、第二候选及置信度，不重新扫描 WSI：

```bash
challenge_review/.venv/bin/python -m challenge_review.import_data --update-predictions
```

右侧模型预测表同时展示每个配置的第一候选和第二候选类别及置信度。

## 局部高倍镜

WSI 工具栏中的显微镜按钮可开启 320 x 240 局部高倍镜，默认使用 20x。开启后可在 10x、20x、40x 间切换；高于切片扫描倍率的选项会自动禁用，按 `Esc` 可关闭。

高倍窗口跟随鼠标并复用当前切片的 Deep Zoom 金字塔，直接读取对应位置的高分辨率瓦片，不对主视图截图做插值放大。主视图仍可正常拖动、滚轮缩放和双击操作。

## Challenge ROI 审核

关键证据 ROI 任务仅对 193 张 14/14 配置均误判病例开放。右侧 `Case Review` 保存最终病变类型、HGD、标签处理、主要鉴别问题、困难因素和专家信心；WSI 工具栏中的矩形按钮可创建最多 3 个 level-0 坐标 ROI。

`Challenge disposition` 用于区分真正的模型困难病例与疑似原标签错误：

- `Retain as true challenge`：人工结论支持原标签，继续保留为模型共同失败病例；
- `Possible label error - adjudication`：人工结论支持模型共识，进入第二位审核者仲裁；
- `Recommend exclusion - label error`：审核者明确建议作为标签问题排除。

系统不会因单人意见直接删除病例。只有至少两位已提交审核者给出相同的 11 类修订标签，且该标签与模型共识一致、与原标签不同，聚合状态才变为 `Eligible for exclusion`。原始 193 张候选集始终保留，筛选和 `challenge_resolutions.csv` 另行提供审核后的确认集、待仲裁集及可排除集。

### 专家勾选与填写说明

以下字段是专家审核时需要选择或填写的完整字典。建议先查看 WSI 全图，再使用局部高倍镜确认细胞学和腺体结构，最后填写病例级判断和 ROI 证据。

#### 1. Final lesion diagnosis：最终病变类型

选择人工审核后最支持的病变类型。该字段描述病变本身，不包含 HGD；HGD 需要在下一项单独选择。

| 选项 | 具体含义 |
| --- | --- |
| `HP` | Hyperplastic polyp，增生性息肉。整体形态支持增生性息肉，未见足以诊断 SSL 的基底部结构异常。 |
| `SSL` | Sessile serrated lesion，SSL。支持基底隐窝扩张、横向生长、靴形或 L 形隐窝等锯齿状病变特征。 |
| `TSA` | Traditional serrated adenoma，传统锯齿状腺瘤。重点考虑异位隐窝形成、嗜酸性胞质、铅笔状细胞核和明显锯齿结构。 |
| `TA` | Tubular adenoma，管状腺瘤。以管状腺体和常规腺瘤性异型增生为主，绒毛成分不占主要地位。 |
| `TVA` | Tubulovillous adenoma，管状绒毛状腺瘤。同时具有明显管状和绒毛状结构，不能仅凭局部小区域决定。 |
| `IP` | Inflammatory polyp，炎性息肉或炎性相关病变。炎症、糜烂、修复性改变更能解释当前形态。 |
| `USA` | Unclassified serrated adenoma，未分类锯齿状腺瘤。锯齿状和腺瘤性特征存在，但不足以可靠归入 HP、SSL、TSA 或其他明确类别。 |
| `Other` | 当前受控分类之外的其他病变。必须在 `Difficulty note` 中说明具体考虑的诊断。 |
| `Ambiguous` | 现有 WSI 或取材无法作出可靠最终分类，或多种诊断证据冲突。选择此项时不要强行指定标签错误排除。 |

#### 2. HGD status：高级别异型增生状态

| 选项 | 具体含义 |
| --- | --- |
| `Absent` | 在可评估区域内未发现明确高级别异型增生。可以存在低级别腺瘤性异型增生。 |
| `Present` | 至少一个可定位区域具有明确 HGD 证据，例如复杂腺体结构、筛状结构、明显细胞学异型、极性丧失或腔内坏死。 |
| `Not assessable` | 组织量、方向、碎片化、折叠或图像质量使 HGD 无法可靠评估。 |
| `Uncertain/conflicting` | 不同区域或不同形态证据相互冲突，存在 HGD 可能但不足以明确判断。 |

对于 11 分类标签，`Present` 会将 SSL、TSA、TA、TVA 分别对应为 SSLD、TSAD、TAD、TVAD；`Absent` 则对应无 HGD 后缀的标签。选择 `Not assessable` 或 `Uncertain/conflicting` 时，系统不会把该记录自动视为一个确定的 11 类标签修订。

#### 3. Label action：原始标签如何处理

该字段只描述专家对原始标签的处理方式，不等同于最终是否保留在 Challenge Set。

| 选项 | 什么时候选择 |
| --- | --- |
| `Confirm original label` | 专家认为原始标签仍然正确。即使模型预测错误，也应保留原标签。 |
| `Correct original label` | 专家认为原始标签错误，并在 `Final lesion diagnosis` 与 `HGD status` 中给出修订结果。 |
| `Remains ambiguous` | 专家不能可靠确认原标签，也不能提出足够有把握的替代标签。应补充原因，必要时交由仲裁。 |

#### 4. Challenge disposition：病例最终处置建议

这是区分“真正模型困难病例”和“原标签可能错误”的核心字段。单个专家的选择不会删除病例。

| 选项 | 具体含义和使用条件 |
| --- | --- |
| `Retain as true challenge` | 专家认为原始标签可信，或认为模型共识并不能推翻原始标签。病例继续作为模型共同失败的真实 challenge。 |
| `Possible label error - adjudication` | 专家认为人工最终标签与模型共识一致、且不同于原始标签，提示原始标签可能错误，但需要第二位专家独立确认。 |
| `Recommend exclusion - label error` | 专家明确认为这是标签错误而非模型困难，建议从“最终确认的 Challenge Set”中排除。仍需至少另一位已提交专家给出相同修订标签后，系统才会标记为 `Eligible for exclusion`。 |

界面会根据最终诊断、原标签和模型共识自动提示：

- 人工标签与原标签一致：通常应选择 `Retain as true challenge`；
- 人工标签与模型共识一致、但与原标签不同：可以选择标签问题相关选项；
- 人工标签既不同于原标签，也不同于模型共识：通常选择 `Possible label error - adjudication` 或保留为存疑，并在备注中解释；
- 无法映射为确定的 11 类标签：不要选择 `Recommend exclusion - label error`。

后端还会强制校验：标签问题处置必须使用 `Correct original label`，最终标签必须能映射为 11 类，并且必须与模型共识一致、与原始标签不同。

#### 5. Primary challenge：主要鉴别困难

提交审核时必须选择一个最主要的困难来源。它不是“模型预测结果”，而是专家认为最能解释该病例为什么容易误判的病理学问题。

| 选项 | 具体含义 |
| --- | --- |
| `HP vs SSL` | 增生性息肉与 SSL 的鉴别，重点观察基底隐窝扩张、横向生长、靴形/L 形隐窝和表面锯齿。 |
| `SSL vs TSA` | SSL 与 TSA 的鉴别，重点观察异位隐窝、嗜酸性胞质、铅笔状细胞核和锯齿结构。 |
| `Serrated vs conventional adenoma` | 锯齿状病变与常规腺瘤之间的鉴别，重点判断锯齿结构和常规腺瘤性异型增生谁占主导。 |
| `TA vs TVA` | 管状腺瘤与管状绒毛状腺瘤的鉴别，需要结合多个区域判断管状、绒毛状和混合结构比例。 |
| `Adenoma vs inflammatory` | 腺瘤性异型增生与炎症、糜烂或修复性异型的鉴别。 |
| `Focal HGD` | 主要困难是 HGD 只存在于局灶区域，低倍全图可能被漏掉。 |
| `Borderline HGD` | 细胞学或结构异常接近 HGD 阈值，存在诊断边界或专家意见差异。 |
| `Insufficient / fragmented specimen` | 碎片化、取材少、方向差或组织缺失导致关键结构无法判断。 |
| `Other differential` | 不属于上述主要鉴别。必须填写 `Other differential` 的补充说明。 |

#### 6. Difficulty modifiers：困难因素多选

这些选项可以多选。只勾选确实影响诊断的因素，不要把所有观察到的技术问题全部勾上。

| 选项 | 具体含义 |
| --- | --- |
| `Focal evidence` | 决定性证据只出现在很小、很局灶的区域，容易被全图或低倍观察遗漏。 |
| `Subtle morphology` | 形态差异非常细微，需要经验或高倍才能识别。 |
| `Heterogeneous lesion` | 同一切片不同区域形态不一致，局部观察可能不能代表整体。 |
| `Poor orientation` | 隐窝、腺体或表面方向不理想，导致结构关系难以判断。 |
| `Limited sampling` | 取材范围有限，缺少足够组织来评估完整结构。 |
| `Requires high magnification` | 需要 20×/40× 等高倍才能确认细胞学或微小结构。 |
| `Requires multiple regions` | 单一区域不足以作出判断，需要比较多个区域。 |
| `Technical artifact` | 折叠、压挤、染色不均、刀痕、气泡或其他技术伪影影响诊断。 |
| `Competing / misleading morphology` | 存在容易误导的形态，例如修复性异型模拟腺瘤，或锯齿与常规腺瘤特征同时存在。 |

#### 7. Expert confidence：专家信心评分

这是专家对本次最终诊断整体可靠性的评分，不是模型置信度，也不是 ROI 证据强度。

| 分数 | 具体含义 |
| --- | --- |
| `1` | 极低信心。组织或图像严重受限，多个诊断均有可能。 |
| `2` | 低信心。有倾向性判断，但关键证据不足或存在明显冲突。 |
| `3` | 中等信心。主要诊断较合理，但仍有一定鉴别或取材限制。 |
| `4` | 高信心。大部分关键证据一致，仅存在轻微不确定性。 |
| `5` | 极高信心。形态证据充分、相互一致，几乎没有实质性鉴别疑问。 |

#### 8. No localizable key evidence：没有可定位的关键证据

该复选框表示专家认为没有一个局部矩形区域可以代表决定性证据。它不是“我没有时间画 ROI”。

勾选后必须选择原因，并且不能同时保留 ROI：

| 原因 | 具体含义 |
| --- | --- |
| `Global architecture required` | 诊断依赖整张切片的总体结构、分布或多个区域关系，单个局部区域不能代表结论。 |
| `Insufficient tissue` | 组织太少，关键区域不存在或无法定位。 |
| `Relevant evidence absent` | 当前切片中没有可见的决定性形态证据。 |
| `Genuine diagnostic ambiguity` | 即使完整浏览，证据仍然相互矛盾或处于真正诊断边界。 |
| `Technical limitation` | 图像损坏、染色、折叠、扫描或其他技术问题阻止可靠定位。 |
| `Other` | 不属于上述情况。必须填写具体原因。 |

如果没有勾选 `No localizable key evidence`，仍然允许 0 个 ROI 提交；这表示专家认为没有必要标出局部证据，但不主张“没有任何可定位证据”。

#### 9. ROI 基本规则

每位专家每张病例最多绘制 3 个矩形 ROI，也允许不画 ROI。ROI 的目标是标记解决主要鉴别诊断最重要的局部区域，不要求覆盖全部病变。

- 使用工具栏矩形按钮，在 WSI 上拖动创建 ROI；
- ROI 坐标以 WSI level-0 原始像素保存，不是屏幕截图坐标；
- 可以选择、移动、缩放、删除和调整 ROI 顺序；
- 每个 ROI 提交前必须填写证据类型、诊断作用、鉴别方向和证据强度；
- 如果选择 `Other evidence`，必须填写自由文本说明；
- ROI 备注用于补充该区域为什么重要，不替代病例级 `Difficulty note`。

#### 10. ROI Diagnostic role：证据的诊断作用

| 选项 | 具体含义 |
| --- | --- |
| `Discriminative` | 该 ROI 能帮助区分两个或多个候选诊断，是最重要的鉴别证据。 |
| `Confirmatory` | 该 ROI 支持已经形成的诊断，但本身未必能排除所有替代诊断。 |
| `Contradictory` | 该 ROI 与当前最终诊断不一致，提示标签、取材或诊断仍需重新讨论。 |
| `Mimic / confounder` | 该 ROI 显示容易模拟其他病变的形态，是模型或专家被误导的原因。 |
| `HGD-defining` | 该 ROI 包含支持或排除 HGD 的关键结构或细胞学证据。 |
| `Assessability / quality` | 该 ROI 主要用于记录组织质量、方向、碎片化或技术限制。 |

#### 11. ROI Evidence strength：证据强度

| 选项 | 具体含义 |
| --- | --- |
| `Weak` | 仅有轻微倾向性，不能单独支持诊断。 |
| `Moderate` | 有一定诊断价值，但需要结合其他区域或全图结构。 |
| `Strong` | 对当前鉴别有明显支持作用，是较可靠的局部证据。 |
| `Decisive` | 几乎决定当前鉴别结论，其他专家通常也应能在该区域看到关键特征。 |

#### 12. ROI Differential direction：鉴别方向

该字段由 `Primary challenge` 动态决定，表示 ROI 支持哪一侧诊断，或是否与最终结论冲突。

对于双类别挑战：

| 方向 | 含义 |
| --- | --- |
| `Supports A / Supports B` | 该区域更支持挑战标题中的 A 或 B。比如 HP vs SSL 中的 `Supports SSL` 表示该 ROI 更支持 SSL。 |
| `Supports both / non-discriminative` | 两侧都能解释该形态，不能有效鉴别。 |
| `Conflicts with final diagnosis` | 该区域与专家最终诊断不一致，应在备注中解释。 |
| `Uncertain` | 方向性不足，不能可靠支持任何一侧。 |

对于 HGD 挑战：

- `Definite HGD evidence`：存在明确 HGD 证据；
- `Suspicious but insufficient`：可疑但未达到明确 HGD 标准；
- `No definite HGD identified`：观察区域未见明确 HGD；
- `Not assessable`：该区域无法可靠评估。

对于其他挑战：

- `Supports final diagnosis`：支持专家最终诊断；
- `Supports alternative diagnosis`：支持替代诊断；
- `Non-discriminative`：对鉴别没有明显帮助；
- `Conflicts with final diagnosis`：与最终诊断冲突；
- `Uncertain`：方向不确定。

#### 13. ROI Evidence type：形态证据多选

证据类型由 `Primary challenge` 自动提供推荐项，同时保留通用证据和 `Other`。可以多选，但每个选项都应能在当前 ROI 中直接观察到。

HP vs SSL：

- `Basal crypt dilation`：隐窝基底部扩张；
- `Horizontal crypt growth`：隐窝沿黏膜肌板方向横向生长；
- `Boot / L-shaped crypt`：靴形或 L 形隐窝；
- `Asymmetric crypt proliferation`：隐窝增生或扩张分布不对称；
- `Surface serration`：表面或隐窝腔内锯齿状结构。

SSL vs TSA：

- `Ectopic crypt formation`：异位隐窝形成；
- `Eosinophilic cytoplasm`：细胞胞质明显嗜酸性；
- `Pencillate nuclei`：铅笔状、细长的细胞核；
- `Slit-like serration`：裂隙样锯齿；
- `Surface serration`：表面锯齿结构。

Serrated vs conventional adenoma：

- `Serrated architecture`：整体或局部锯齿状腺体结构；
- `Conventional dysplasia`：常规腺瘤性异型增生；
- `Crypt elongation`：隐窝延长；
- `Eosinophilic cytoplasm`：嗜酸性胞质；
- `Villous architecture`：绒毛状结构。

TA vs TVA：

- `Tubular architecture`：管状腺体结构；
- `Villous architecture`：绒毛状结构；
- `Tubulovillous architecture`：管状与绒毛状结构混合存在。

Adenoma vs inflammatory：

- `Adenomatous dysplasia`：腺瘤性异型增生；
- `Regenerative atypia`：炎症或损伤后的修复性异型；
- `Lamina propria inflammation`：固有层明显炎症；
- `Erosion / ulceration`：糜烂或溃疡。

Focal HGD / Borderline HGD：

- `Complex glandular architecture`：复杂、拥挤或分支的腺体结构；
- `Cribriforming`：筛状结构；
- `Marked cytologic atypia`：明显细胞学异型；
- `Loss of polarity`：细胞极性丧失；
- `Luminal necrosis`：腺腔内坏死。

Insufficient / fragmented specimen：

- `Fragmentation`：组织碎片化；
- `Poor orientation`：组织方向不佳；
- `Limited tissue`：组织量有限；
- `Cautery artifact`：电灼或烧灼伪影。

所有挑战都可以使用通用证据：

- `Surface serration`：表面锯齿；
- `Crypt architecture`：隐窝总体结构；
- `Cytologic atypia`：细胞学异型；
- `Inflammation`：炎症；
- `Tissue quality limitation`：组织或图像质量限制；
- `Other`：不在列表中的证据，必须填写具体文字。

#### 14. 兼容旧版审核表单

页面下方的“重新审核”表单用于保留原有 3,153 张病例审核流程。它与上面的 `Challenge Case Review` 相互独立：正式的 193 张关键病例分析应优先填写 `Challenge Case Review`；旧表单只用于兼容历史审核、快速记录或非核心病例浏览。

旧表单选项解释如下：

| 字段/选项 | 具体含义 |
| --- | --- |
| `修订标签` | 对原始 11 分类标签的直接修订。应选择当前最支持的标签；如果诊断仍有争议，结合“存疑”状态和备注说明。 |
| `已完成` | 本次旧版审核已经形成可以保存的判断。 |
| `存疑` | 证据不足、存在标签争议、需要进一步专家讨论或不能可靠确定标签。 |
| `低` | 旧版审核对修订标签的信心较低。 |
| `中` | 有一定支持证据，但仍存在明显不确定性。 |
| `高` | 形态证据较充分，修订标签较可靠。 |
| `图像质量差` | 扫描、清晰度、压缩、失焦或其他图像问题影响判断。 |
| `组织折叠` | 组织折叠、重叠或压挤造成结构失真。 |
| `染色异常` | 染色过深、过浅、不均或颜色失真影响细胞学、黏液或结构判断。 |
| `组织不完整` | 关键组织缺失、取材范围不足或病变被截断。 |
| `标签争议` | 专家认为原始标签本身存在疑问，或不同诊断意见之间存在明确冲突。 |
| `备注` | 记录支持修订标签的主要原因、疑难点、质量问题或后续建议。 |

旧表单的“存疑”或“标签争议”不会自动改变 Challenge resolution，也不会自动把病例从 193 张候选集排除。需要排除原标签错误时，必须在上方的新 `Challenge Case Review` 中填写 `Final lesion diagnosis`、`HGD status`、`Label action` 和 `Challenge disposition`，并遵循双专家一致规则。

每个 ROI 可记录形态证据、诊断作用、鉴别方向、证据强度和备注。草稿在修改后 800 ms 自动保存，正式提交后仍可修订；管理员锁定后该审核记录只读。其他审核者的具体结论和 ROI 坐标在本人提交前隐藏，提交后仅显示聚合一致性。

导出入口提供：

- `challenge_case_reviews.csv`：每位审核者每张病例一行；
- `challenge_rois.csv`：每个 ROI 一行；
- `challenge_rois.geojson`：level-0 像素坐标矩形及诊断、MPP、物理尺寸和 evidence 属性。
- `challenge_resolutions.csv`：193 张候选病例的聚合处置状态与建议修订标签。

启动时会保留原 `reviews` 和 `review_history` 表。旧审核自动映射为新系统草稿，不覆盖原始 manifest；正式迁移前备份保存在 `data/challenge_review_pre_roi_*.sqlite3`。

## 启动

```bash
challenge_review/run_server.sh
```

默认监听 `0.0.0.0:8765`。可通过 `CHALLENGE_REVIEW_HOST`、`CHALLENGE_REVIEW_PORT`、`CHALLENGE_REVIEW_DB`、`CHALLENGE_REVIEW_SESSION_HOURS`、`CHALLENGE_REVIEW_COOKIE_SECURE` 和 `CHALLENGE_REVIEW_WSI_CACHE_SIZE` 调整运行配置。

## HP iSyntax 缓存

HP 切片使用本地持久 Deep Zoom 缓存。默认目录为 `cache/wsi`，达到 300 GB 后按 LRU 清理至 270 GB。可通过以下环境变量调整：

- `CHALLENGE_REVIEW_WSI_DISK_CACHE`
- `CHALLENGE_REVIEW_WSI_CACHE_MAX_GB`
- `CHALLENGE_REVIEW_WSI_CACHE_TRIM_GB`
- `CHALLENGE_REVIEW_WSI_OVERVIEW_MAX_DIMENSION`
- `CHALLENGE_REVIEW_ISYNTAX_TIMEOUT`

将 108 张核心均错病例和 `hardness_rank <= 200` 病例合并后的 114 张 HP 切片加入可恢复预热队列：

```bash
cd /data15/data15_5/yuexin2/adenoma
challenge_review/.venv/bin/python -m challenge_review.prewarm --wait
```

预热由 Web 服务内的单一低优先级工作队列执行，审核者的交互瓦片请求优先。管理员可通过 `/api/cache/status` 查看缓存空间、队列和失败病例。

## 数据安全

- 原始 manifest 和模型结果只读。
- 审核记录保存在 `data/challenge_review.sqlite3`。
- 每次审核修改都会写入 `review_history`。
- Challenge Case/ROI 修改会另行写入不可变的 `challenge_review_history`。
- 对外网络部署前必须配置 HTTPS，并设置 `CHALLENGE_REVIEW_COOKIE_SECURE=1`。
