# Adenoma AgentFlow Workflow

> 本文件是当前 AgentFlow 的唯一上位流程规范。正式 Architecture 输入固定为
> `five_x_patch_manifest.jsonl`，canonical primary scale 固定为 `5x`。
> `src/adenoma_agent/agentflow/` 已实现 contract-first、non-clinical 控制面；
> 尚未完成训练或医学验证的模型必须显式报 unavailable，不能由 heuristic 或
> synthetic checkpoint 静默替代。

Module 1: Multi-scale Evidence Package Construction
模块目标
Evidence Package Construction 负责将原始 WSI 转换为结构化、可追踪、
可持续更新的病理证据。
Evidence Package 包含：
- Tissue Context Evidence
- Architecture Evidence
- Spatial Evidence
- Reviewer Evidence
其中前三类证据由初始分析流程生成，Reviewer Evidence 在后续诊断循环中动态补充。

---
## 1.1 Mucosa Extractor

### 1.1.1 模块职责

`Mucosa Extractor` 是 Evidence Package Construction 的第一个可执行模块，负责把原始 WSI 转换为可追踪的 `Tissue Context Evidence`，同时生成下游 5x 分析所需的空间入口。

它产生两个正式输出方向：

1. `High-Recall Mucosa Search Mask`：用于生成与 WSI level-0 坐标系对齐的 5x patch manifest；
2. `Tissue Context Evidence`：保存原始 tissue probabilities、任务语义映射、空间分布和不确定性。

本模块不负责：

- 生成 HP、SSL、TSA、adenoma 或 inflammatory polyp 等诊断假设；
- 判断 dysplasia 是否存在；
- 选择 Reviewer 或决定最终 ROI review 顺序；
- 提取 Architecture Evidence。

因此，`abnormal_epithelial_candidate` 只能理解为“异常上皮候选信号”，不能理解为 dysplasia-positive evidence。

### 1.1.2 v1 数据流

```text
Raw WSI
  ↓
Read WSI Metadata
  - level dimensions
  - objective magnification / MPP
  - level-0 coordinate system
  ↓
Tissue-aware, level-0-aligned 20x Tile Sampling
  ↓
UNI Encoder + PathPrism CRC100K Classifier
  ↓
Nine-class Raw Tissue Probabilities
  ↓
Seven-channel Task Context Mapping
  ├── Tissue Context Evidence
  └── Mucosa Score = P(NORM) + P(TUM)
          ↓
      High-Recall Mucosa Search Mask
          ↓
      Coordinate-aligned 5x Patch Manifest
```

当前 v1 只使用 `UNI + PathPrism CRC100K` 作为事实来源。CONCH 不参与 probability fusion、mask rescue 或 v1 context score 计算。

CONCH 可以在后续作为独立实验验证：

- CRC100K 与本任务之间是否存在稳定的 domain mismatch；
- 是否能为不确定区域提供额外的 reviewer-routing signal；
- 是否能提高漏诊保护而不显著扩大 mask。

在完成独立验证和 calibration 之前，不将 CONCH 写入正式 Mucosa Extractor 主路径。

### 1.1.3 Tissue Context Mapping

PathPrism 输出九类 CRC100K 概率：

```text
DEB, ADI, BACK, MUS, NORM, LYM, MUC, STR, TUM
```

它们被映射为七个任务语义通道：

```yaml
background_or_artifact: [DEB, ADI, BACK]
smooth_muscle_or_deep_tissue: [MUS]
normal_epi_context: [NORM]
inflammatory_context: [LYM]
mucus_rich_context: [MUC]
stromal_context: [STR]
abnormal_epithelial_candidate: [TUM]
```

九类原始概率属于不可替代的 source evidence；七通道 context 是可版本化重算的 derived evidence。每条 tile evidence 均保留 level-0 bbox、grid index、tissue coverage、uncertainty 和最终 mask inclusion 状态。

### 1.1.4 High-Recall Mask Policy

v1 固定使用：

```text
mucosa_score = P(normal_epi_context) + P(abnormal_epithelial_candidate)
candidate = mucosa_score >= mucosa_threshold
final_mask = anatomical_postprocess(candidate)
```

当前默认 `mucosa_threshold = 0.30`。`LYM`、`MUC`、`STR` 和 `MUS` 作为 soft context evidence 保存，但不单独激活 mask。

该阈值是当前实验的 operational default，不代表已完成医学性能校准。mask 的目标是为下游提供高召回搜索范围，而不是输出组织学诊断。

### 1.1.5 Compact Output Contract

```text
mucosa_extractor/
├── manifest.json
├── tile_index.jsonl
├── five_x_patch_manifest.jsonl
├── errors.jsonl                 # optional，仅发生错误时生成
└── slides/<slide_id>/
    ├── maps.npz
    └── qc_panel.png
```

文件职责：

- `manifest.json`：模型、配置、channel order、坐标参数、运行统计和每张 WSI 的索引；
- `tile_index.jsonl`：合并后的 tile-level source/derived evidence 表；
- `maps.npz`：9-channel raw maps、7-channel context maps、uncertainty、mucosa score、final mask 和 valid mask；
- `five_x_patch_manifest.jsonl`：Mucosa Extractor 与 Mucosal Architecture 模块之间的正式接口；
- `qc_panel.png`：六面板可视化，仅用于快速 QC；
- `errors.jsonl`：结构化失败记录，正常运行时不创建。

详细字段、运行方式和数组定义见 [Mucosa Extractor v1](model/mucosa_extractor_v1.md)。

### 1.1.6 当前实现状态

- compact v1 pipeline 已实现；
- 已完成 10 张 40x WSI 的端到端运行；
- 运行结果：3,390 个有效 20x tiles、304 个 coordinate-aligned 5x patches、0 errors；
- 当前结果用于检查 pipeline contract、坐标一致性和 QC，不用于声明模型敏感度或特异度。

本次运行记录见 [10-sample Mucosa Extractor manifest](../artifacts/mucosa_extractor_10sample_20260721_104930/mucosa_extractor/manifest.json)。


---
2.3 Mucosal Architecture Classifier
功能
Mucosal Architecture Classifier 读取 `five_x_patch_manifest.jsonl`，按照其中的
level-0 bbox 从原始 WSI 提取 5x patches，并进行多标签形态分类。
主要输出：
- Serrated Score
- Tubular Score
- Villous Score
- Patch Embedding
- Prediction Uncertainty
这些结果共同组成 Architecture Evidence。

---
2.4 Spatial Evidence Evaluator
功能
Spatial Evidence Evaluator 不再单独判断 patch 类型，而是分析不同 architecture
pattern 在整张切片中的分布方式。
主要分析：
- architecture ratio；
- spatial cluster；
- co-localization；
- spatial mixing；
- heterogeneity；
- uncertainty region；
- ROI candidate。
输出写入 Spatial Evidence。

---
Module 2: Hypothesis-driven Planning Agent
3.1 模块目标
Planning Agent 根据当前 Evidence Package 和 Structured Knowledge Base，
维护多个竞争性假设，并主动寻找能够区分这些假设的补充证据。
Planning Agent 不负责输出最终诊断，而负责回答：
当前最需要检查什么证据？

本模块已经在独立 AgentFlow v1 控制面中实现：

```text
Evidence Snapshot -> Planner -> one Reviewer action -> append Ledger -> resnapshot
```

实现入口为 `scripts/run_agentflow_v1.py`，核心模块位于
`src/adenoma_agent/agentflow/`。该实现验证数据合同、确定性空间分析、单步规划、
Reviewer 调用边界、追加式 Ledger 和 Chief 投影；它仍是 non-clinical 控制面，
不代表 Architecture、dysplasia hotspot 或 Reviewer 模型已经获得医学有效性。

Planning Agent 应实现为无副作用的决策内核。它只读取一个不可变的
Evidence Ledger Snapshot，并输出单步 `PlanDecision`。它不直接裁图、
不调用 Reviewer、不修改 Evidence Ledger，也不输出最终诊断。

实际 Reviewer 调用、失败重试、成本记录和 Evidence 写回由外部
Orchestrator 执行。标准运行边界如下：

```text
Evidence Ledger Snapshot
        ↓
Evidence Reducer
        ↓
Hypothesis Engine
        ↓
Discriminative Question Generator
        ↓
ROI + Reviewer Action Binder
        ↓
Action Scorer + Stop Policy
        ↓
PlanDecision
        ↓
Orchestrator invokes Reviewer
        ↓
Append Reviewer Evidence
        └────────── loop ──────────┘
```

Evidence Ledger 是 Evidence Package 内部的追加式证据账本。每次写入新证据后
形成新的 Snapshot，下一轮 Planning 必须基于新的 `snapshot_id` 重新执行，
以保证规划过程可回放、可审计且不会读取到中途变化的状态。

---
3.2 Hypothesis Representation and Generation

Hypothesis 在数据 contract 中保留三个核心字段：

```json
{
  "hypothesis_id": "H_SSL",
  "pathway": "serrated",
  "subtype": "ssl",
  "dysplasia_state": "unassessed",
  "dysplasia_applicability": "class_defining"
}
```

其中：

- `pathway`：病变所属诊断通路；
- `subtype`：当前候选形态亚型；
- `dysplasia_state`：当前 high-grade/definite dysplasia 证据状态；
- `dysplasia_applicability`：dysplasia 是否参与当前 11 类标签的后缀映射。

虽然 contract 保留 `pathway / subtype / dysplasia_state` 三个字段，推理过程
不采用 `pathway -> subtype -> dysplasia` 的硬性三级串行剪枝，而采用
“形态学两级层次 + 独立 dysplasia 轴”的 `2+1` 结构：

```text
pathway -> subtype ---------+
                            +-> final label projection
dysplasia_state ------------+
```

`pathway -> subtype` 构成 morphology hypothesis；`dysplasia_state` 由
branch-agnostic DysplasiaReviewer 的证据独立更新，再结合 subtype、空间共定位
和 reviewer provenance 投影到最终标签。这样可以避免在 pathway 尚未解决时
过早阻断 dysplasia review，也可以保留 serrated、conventional 和 inflammatory
之间的竞争关系。

v1 不使用 LLM 自由生成 hypothesis，而是从 Structured Knowledge Base 固定加载
七个合法 morphology hypothesis template：

```text
serrated / ssl
serrated / hp
serrated / tsa
serrated / unclassified_serrated_adenoma
conventional_adenoma / tubular_adenoma
conventional_adenoma / tubulovillous_adenoma
inflammatory / inflammatory_reactive
```

所有合法 hypothesis 可以在初始轮次同时存在，由 Evidence 排序和降权；
不应因为一次低置信度 pathway 判断而永久删除竞争候选。

`dysplasia_state` 使用以下枚举：

```text
unassessed
not_evaluable
not_supported
supported
conflicting
```

语义如下：

- `unassessed`：尚未完成充分检查；
- `not_evaluable`：已经检查，但 ROI、倍率或图像质量不足；
- `not_supported`：在充分可评估的证据中未发现 high-grade/definite dysplasia；
- `supported`：存在 high-grade/definite dysplasia 支持证据；
- `conflicting`：不同 ROI、Reviewer 或证据来源之间存在未解决冲突。

`not_supported` 不等于整张切片绝对不存在 dysplasia；它只表示当前充分采样证据
尚不支持 dysplasia-positive mapping。

最终 11 类由 morphology hypothesis 和 dysplasia state 投影：

| Morphology hypothesis | Dysplasia mapping | Final label |
|---|---|---|
| `ssl` | `not_supported` | `SSL` |
| `ssl` | `supported` | `SSLD` |
| `hp` | not class-defining | `HP` |
| `tsa` | `not_supported` | `TSA` |
| `tsa` | `supported` | `TSAD` |
| `unclassified_serrated_adenoma` | not class-defining | `Unclassified serrated adenoma` |
| `tubular_adenoma` | `not_supported` | `Tubular adenoma` |
| `tubular_adenoma` | `supported` | `TAD` |
| `tubulovillous_adenoma` | `not_supported` | `Tubulovillous adenoma` |
| `tubulovillous_adenoma` | `supported` | `TVAD` |
| `inflammatory_reactive` | not class-defining | `Inflammatory` |

该表只定义 candidate final-label mapping，不代表 Planning Agent 已经作出最终诊断。
当 `dysplasia_state` 为 `unassessed`、`not_evaluable` 或 `conflicting` 时，不能仅因
尚无 positive evidence 就自动投影为 non-D 类；这些状态必须进入 unresolved evidence
或 uncertain diagnosis 路径，由 Chief Agent 最终综合。

对于 `HP`、`Unclassified serrated adenoma` 和 `Inflammatory`，dysplasia 不参与
当前 final-label 后缀映射，但 dysplasia evidence 仍必须保留。若出现强 dysplasia
证据，应建立 Conflict Object 并重新检查 subtype，而不是直接忽略该证据。

---
3.3 Evidence Ledger Snapshot and Evidence Reducer

Planning Agent 每轮只读取一个不可变 Snapshot。Snapshot 至少包含：

```json
{
  "snapshot_id": "ledger_v17",
  "evidence_ids": ["E001", "E002", "E003"],
  "created_after_action_id": "planner_action_006"
}
```

Evidence Reducer 将 tile、patch、cluster 和 reviewer-level evidence 归一化为
feature-level Evidence View。Reducer 负责：

- 按 feature、ROI、scale、source 和 spatial cluster 聚合证据；
- 保留 `present / absent / uncertain / not_evaluable` 的原始语义；
- 计算 evidence strength、quality、spatial coverage 和 source agreement；
- 标记冲突和高度相关的重复证据；
- 保留所有 evidence id 和 provenance 引用。

Reducer 不负责 hypothesis ranking，也不能把 `missing` 或 `not_evaluable`
转换为 contradictory evidence。只有 quality adequate 的 ROI 中得到的 `absent`
才能作为潜在负证据。

---
3.4 Evidence Analysis and Hypothesis Ranking

对于每个 morphology hypothesis，Hypothesis Engine 从 Hypothesis Template 获取：

- `supporting_evidence`；
- `contradictory_evidence`；
- `required_evidence`；
- `discriminative_features`；
- `reviewer_triggers`；
- `dysplasia_applicability`；
- `final_label_mapping`。

然后与当前 Evidence View 对比。Hypothesis 排序不只保留一个不透明的
`working_score`，而应区分 plausibility、evidence coverage 和 conflict：

```json
{
  "hypothesis": {
    "hypothesis_id": "H_SSL",
    "pathway": "serrated",
    "subtype": "ssl",
    "dysplasia_state": "unassessed",
    "dysplasia_applicability": "class_defining"
  },
  "observed_supporting_evidence": [],
  "observed_conflict_evidence": [],
  "missing_required_evidence": [],
  "plausibility_score": 0.72,
  "required_evidence_coverage": 0.50,
  "conflict_score": 0.18,
  "decision_ready": false
}
```

`plausibility_score` 是工作排序分数，不是校准后的临床诊断概率。
`missing_required_evidence` 用于生成下一步问题和限制 decision readiness，
但 missing 本身不应直接作为 contradictory evidence。

---
3.5 Discriminative Question Generation and Action Binding

Reviewer Planning 分为两个步骤。

第一步，Discriminative Question Generator 比较当前 top hypotheses，优先生成
能够区分竞争假设的问题，而不是简单枚举每个 hypothesis 的全部 missing evidence：

```json
{
  "question_id": "Q008",
  "target_feature": "serration_to_crypt_base",
  "question": "Does serration extend to the crypt base?",
  "discriminates": ["ssl", "hp"],
  "expected_effect": {
    "present": "supports_ssl",
    "absent_in_adequate_roi": "supports_hp"
  }
}
```

第二步，ROI + Reviewer Action Binder 根据 Knowledge Base 中的
feature-reviewer-scale compatibility，将问题绑定到 ROI Manager 提供的候选池：

```json
{
  "action_id": "planner_action_007",
  "question_id": "Q008",
  "reviewer": "SerratedArchitectureReviewer",
  "roi_id": "ROI_C03",
  "scale": 10.0,
  "target_features": [
    "serration_to_crypt_base",
    "basal_crypt_dilation"
  ],
  "goal": "Distinguish SSL-like from HP-like architecture"
}
```

Binder 只能选择已有 ROI candidate，不直接发明像素坐标、不裁图、也不调用
Reviewer。候选 action 必须满足 reviewer、target feature、scale 和 ROI semantics
之间的兼容约束。

### 3.5.1 Reviewer Task System

Reviewer 是由 Planner 选择、由 Orchestrator 调用的 ROI 级证据提取工具。六个逻辑
Reviewer 可以共享同一个 VLM backend，但必须使用相互独立、可版本化的
`reviewer / task_profile / prompt_version`：

| Reviewer | Task profiles | Primary scale | Evidence boundary |
|---|---|---|---|
| `QualityMucosaReviewer` | `overview_evaluability`, `crypt_evaluability`, `cytology_evaluability` | `2.5x / 5x / 10x / 20x` | 判断 ROI 与目标 feature 是否可评估，不判断病变类型 |
| `SerratedArchitectureReviewer` | `serrated_overview`, `ssl_hp_discrimination` | `2.5x / 5x`, `5x / 10x` | 提取 serration、crypt-base 与 basal dilation 等 SSL/HP 证据 |
| `TSAReviewer` | `tsa_overview`, `tsa_architecture`, `tsa_signature_cytology` | `2.5x / 5x`, `5x / 10x`, `10x / 20x` | 提取 ectopic crypt、slit-like serration、eosinophilia 与 pencillate nuclei；不得判断 HGD |
| `ConventionalArchitectureReviewer` | `conventional_overview`, `tubular_villous_resolution`, `villous_extent_estimation` | `2.5x / 5x`, `5x / 10x` | 提取 tubular/villous、component extent、mixing 与 gland complexity |
| `DysplasiaReviewer` | `high_grade_dysplasia_assessment` | `20x` primary，可附 `10x` parent | branch-agnostic 地提取 high-grade/definite dysplasia，不输出 D-suffix 类别 |
| `InflammatoryReactiveReviewer` | `inflammatory_overview`, `reactive_mimic_resolution` | `2.5x / 5x`, `5x / 10x` | 提取炎症、损伤、reactive mimic 证据与 ROI-local 反证 |

`Unclassified serrated adenoma` 仍由 Chief 综合 serrated、TSA、conventional 和
Spatial Evidence 后完成最终投影，不新增专门 Reviewer。

倍率语义固定为：

- `2.5x`：正式的 ROI overview evidence；不能替代 Spatial Evidence 的全切片
  ratio、cluster 或 heterogeneity；
- `5x`：整体 architecture、component extent 和 mixing；
- `10x`：crypt-level architecture；
- `20x`：cytology 与 high-grade/definite dysplasia。

`TSAReviewer.tsa_signature_cytology` 与
`DysplasiaReviewer.high_grade_dysplasia_assessment` 是两个独立任务。前者观察
eosinophilic cytoplasm、pencillate nuclei 等 TSA signature，不能把这些特征直接
解释为 high-grade/definite dysplasia；后者不能反向输出 TSA subtype 或 `TSAD`。

### 3.5.2 Reviewer Visibility Boundary

Reviewer 模型可见输入只包含中性的 `task_profile`、完整 `target_features`、primary
ROI、允许的 context views，以及图像 hash、坐标、倍率等技术元数据。以下信息不得
进入模型可见 payload：

- hypothesis ranking 或 top-hypothesis score；
- Planner 的 `expected_effect`、预期答案或倾向性诊断提示；
- final 11-class label mapping。

冲突复核可以额外提供 feature-level disagreement，但不得提供哪个诊断“应该获胜”。
每个 target feature 必须且只能返回一次；额外发现只能通过 Registry allowlist
约束的 incidental finding 返回。

正式 contract 与机器可读定义见：

- [Reviewer 设计](model/architecture_evidence_reviewer_design.md)
- [Reviewer Contract](model/contracts/reviewer.md)
- [Reviewer Registry](model/contracts/schemas/reviewer_registry_v1.json)
- [Reviewer Registry Schema](model/contracts/schemas/reviewer_registry_v1.schema.json)
- [Reviewer Common Schema](model/contracts/schemas/reviewer_common_v1.schema.json)
- [Reviewer Task Request Schema](model/contracts/schemas/reviewer_task_request_v1.schema.json)
- [Reviewer Observation Schema](model/contracts/schemas/reviewer_observation_v1.schema.json)
- [Reviewer Ledger Record Schema](model/contracts/schemas/reviewer_ledger_record_v1.schema.json)
- [AgentFlow 实现与模型缺口](model/agentflow_architecture_and_model_gaps.md)

### 3.5.3 Legacy `*_hits` Import Mapping

旧 Trace/Navigate/Observe artifact 若需要离线迁移到当前 Reviewer contract，可按以下
task profile 解释其证据来源。该映射不是当前 AgentFlow 的运行路径：

| Legacy artifact field | Current reviewer task profile |
|---|---|
| `serrated_hits` | `SerratedArchitectureReviewer.serrated_overview` |
| `ssl_hits`, `hp_hits`, `abnormal_crypt_hits` | `SerratedArchitectureReviewer.ssl_hp_discrimination` |
| `tsa_hits` | `TSAReviewer.tsa_architecture` |
| `tsa_cytological_atypia_hits` | `TSAReviewer.tsa_signature_cytology` |
| `conventional_hits`, `conventional_architecture_checklist` | `ConventionalArchitectureReviewer.conventional_overview`, `tubular_villous_resolution` 或 `villous_extent_estimation`，按目标 feature 选择 |
| `serrated_dysplasia_hits`, `conventional_dysplasia_hits`, `dysplasia_hits` | `DysplasiaReviewer.high_grade_dysplasia_assessment` |
| `inflammatory_hits` | `InflammatoryReactiveReviewer.inflammatory_overview` 或 `reactive_mimic_resolution` |
| 当前 observation evaluability/limitations | 对应尺度的 `QualityMucosaReviewer.*_evaluability` |

该表仅是历史 artifact 的兼容说明，不是把旧字段直接重命名为新记录的 runtime adapter。
旧 `supporting / opposing / uncertain` 必须结合 ROI quality 和 provenance 转换，不能
机械等同于新 contract 的 `present / absent / uncertain / not_evaluable`。

---
3.6 Action Scoring and Stop Policy

所有评分项先归一化到 `[0, 1]`。候选 Reviewer action 的 v1 评分形式为：

```text
Reviewer Action Score
= wd * Discriminative Value
+ ws * ROI Suitability
+ wc * Spatial Coverage
- wk * Normalized Crop and Invocation Cost
- wr * Evidence Redundancy
```

其中：

- `Discriminative Value`：该 feature 区分当前 top hypotheses 的能力；
- `ROI Suitability`：ROI quality、feature visibility、scale fit 和 candidate signal；
- `Spatial Coverage`：是否补充新的 cluster、component 或未覆盖区域；
- `Normalized Cost`：裁图、模型调用和预计运行成本；
- `Evidence Redundancy`：ROI overlap、重复 feature、相同 cluster 和相关 source。

Action Scorer 回答“如果继续，哪个 action 最值得执行”；Stop Policy 单独回答
“是否还应该继续”。Stop Policy 至少支持：

```text
diagnostic_ready
no_useful_action
budget_exhausted
non_diagnostic_or_quality_limited
```

只有 `diagnostic_ready` 表示 required evidence coverage、top-hypothesis margin、
dysplasia assessment 和 conflict state 已达到预设标准。其余停止原因必须保留
unresolved questions，不能因为循环停止而自动提高诊断置信度。

---
3.7 PlanDecision, Reviewer Invocation and Evidence Update

Planning Agent 每轮输出一个单步 `PlanDecision`：

```json
{
  "plan_id": "PLAN_007",
  "snapshot_id": "ledger_v17",
  "decision": "invoke_reviewer",
  "ranked_hypotheses": [],
  "selected_question": {
    "question_id": "Q008",
    "target_feature": "serration_to_crypt_base"
  },
  "selected_action": {
    "action_id": "planner_action_007",
    "reviewer": "SerratedArchitectureReviewer",
    "task_profile": "ssl_hp_discrimination",
    "roi_id": "ROI_C03",
    "scale": 10.0
  },
  "stop": false,
  "stop_reason": null
}
```

v1 默认每轮只选择一个 Reviewer action，使新证据能够在下一次调用前改变后续
规划。未来只有在多个 action 相互独立且不存在顺序依赖时，才考虑批量并行调用。

Orchestrator 接收 `PlanDecision` 后负责裁图、调用、retry、schema validation、
成本记录和错误处理。Reviewer 只返回结构化形态学观察，不直接返回最终诊断：

```json
{
  "feature_id": "basal_crypt_dilation",
  "status": "present",
  "status_confidence": 0.84,
  "feature_evaluability": "adequate",
  "scope": "roi_local",
  "evidence_text": "The crypt base is visibly dilated in the supplied ROI.",
  "limitations": []
}
```

以上为单条 finding 的简化展示；正式传输使用
`ReviewerTaskRequestV1 -> ReviewerObservationV1 -> ReviewerLedgerRecordV1`。
ROI、倍率、Reviewer 和 action 引用分别位于 Observation 顶层与 Ledger provenance，
不在单条 finding 中重复。
Orchestrator 必须从 `PlanDecision` 构造去除 hypothesis ranking、expected answer 和
final-label mapping 的模型可见请求，再补齐 provenance 后追加写入 Ledger。

Reviewer Evidence 以 append-only 方式写入 Evidence Ledger，不覆盖既有记录。
写入后产生新的 Snapshot，并重新执行完整 Planning loop。Reviewer 调用失败、
ROI 不可评估或 schema 校验失败也应形成结构化 invocation/limitation record，
避免系统无限重复同一 action。

---
Module 3: Chief Agent
4.1 模块目标
Chief Agent 对完整诊断轨迹进行最终综合，包括：
- Evidence Package；
- ranked hypotheses；
- reviewer findings；
- unresolved conflicts；
- Structured Knowledge Base。

---
4.2 Final Diagnostic Decision
Chief Agent 判断：
- 当前证据是否足以支持最终 diagnosis；
- 是否仍存在 major conflict；
- 是否仍需补充 discriminative evidence；如需补充，则交回 Planning Agent 规划
  具体 Reviewer action；
- 是否只能给出 uncertain diagnosis。

---
4.3 Guideline Retrieval
在诊断确定后，根据最终诊断从知识库检索：
- treatment recommendations；
- resection requirements；
- surveillance recommendations；
- pathology reporting considerations。
该部分应采用检索式生成，而不是由 LLM 自由生成。

---
4.4 Structured Output

```json
{
  "status": "final",
  "final_label": "SSLD",
  "final_diagnosis": "Sessile serrated lesion with dysplasia",
  "diagnostic_confidence": 0.86,
  "morphology_hypothesis_id": "H_SSL",
  "dysplasia_state": "supported",
  "supporting_evidence_ids": ["EV_ARCH_003", "EV_DYS_012"],
  "conflicting_evidence_ids": [],
  "unresolved_questions": [],
  "conflicts": [],
  "management_recommendation": {
    "status": "retrieved",
    "source": "versioned local guideline entry",
    "source_version": "...",
    "recommendation": "..."
  },
  "knowledge_version": "kb_v1",
  "non_clinical": true
}
```

当证据未达到 `diagnostic_ready`、dysplasia 尚未解决或存在 active major conflict 时，
`status` 必须为 `uncertain`，`final_label/final_diagnosis` 必须为 `null`，并保留
`unresolved_questions` 与 `conflicts`。所有 evidence ID 必须能够回溯到 Ledger。

## 建议统一的中英文规则

| 内容类型 | 推荐语言 |
|---|---|
| Module 名称 | 英文 |
| Agent 名称 | 英文 |
| Reviewer 名称 | 英文 |
| JSON 字段 | 英文 |
| 代码和公式 | 英文 |
| 章节解释 | 中文 |
| 设计动机 | 中文 |
| 实现细节 | 中文 |
| 病理术语 | 首次中英对照，随后英文 |
| 流程图节点 | 英文为主 |

例如首次写：

> 基底部腺体扩张（`basal crypt dilation`）是 SSL-like hypothesis 的重要支持证据。

后续直接使用：

> `basal crypt dilation` 尚未完成 review。
