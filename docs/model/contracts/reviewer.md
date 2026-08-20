# Reviewer Contract

## 目标

本 contract 定义当前 AgentFlow v1 中 Reviewer 的可见输入、结构化输出、调度边界和失败记录。

Reviewer 是由 Planner 选择、由 Orchestrator 调用的 ROI 级证据提取工具。它只回答当前任务要求的可见形态学问题，不负责：

- hypothesis generation 或 ranking；
- 选择、生成或裁切 ROI；
- 决定下一步 action；
- 写入或覆盖 Evidence Ledger；
- 解决病例级冲突；
- 输出最终 11-class diagnosis。

当前 contract-first Reviewer 调用链已经接入 `src/adenoma_agent/agentflow/`。
视觉 backend 未配置或调用失败时必须写入 `invocation_failure`；不得用 synthetic 输出
冒充医学有效结果。

## Reviewer 集合

v1 使用六个逻辑 Reviewer contract：

- `QualityMucosaReviewer`
- `SerratedArchitectureReviewer`
- `TSAReviewer`
- `ConventionalArchitectureReviewer`
- `DysplasiaReviewer`
- `InflammatoryReactiveReviewer`

逻辑 Reviewer 不等于独立模型服务。六个 contract 可以共享同一个 VLM backend，但必须分别记录 `reviewer`、`task_profile`、prompt version 和 model version。

具体 task profile、feature allowlist 和倍率兼容关系以机器可读的
[`reviewer_registry_v1.json`](schemas/reviewer_registry_v1.json) 为准。

## 倍率语义

Reviewer contract 固定支持以下 primary scale：

- `2.5x`：正式的 ROI overview evidence，用于局部总体分布、代表性和低倍形态；不能输出 whole-slide ratio、cluster 或 heterogeneity 结论。
- `5x`：整体 architecture、component extent 和 mixing。
- `10x`：crypt-level architecture 和局部结构细化。
- `20x`：细胞学细节和 high-grade/definite dysplasia。

每条 finding 必须声明 `scope`：

- `roi_overview`：只概括当前 overview ROI；
- `roi_local`：只描述当前局部 ROI。

Reviewer 不得把任一 scope 的 finding 外推为整张 WSI 的绝对存在或不存在。

## 可见上下文

`ReviewerTaskRequestV1` 明确区分 orchestration metadata 与模型可见的 `model_input`。

模型可以看到：

- 中性的 `task_profile`；
- `target_features`；
- primary ROI；
- registry 允许的 overview、parent 或 peer context views；
- 图像倍率、坐标、尺寸、MPP 和必要的技术元数据。

模型不能看到：

- top hypothesis ranking；
- candidate final labels；
- expected answer 或 expected effect；
- 哪一种 finding 会提高哪一个 hypothesis；
- guideline recommendation；
- Planner 的诊断倾向。

正常调用不需要额外的 mode 字段。冲突复核通过可选的
`model_input.feature_disagreements` 表达，此时只允许提供 feature-level
disagreement 类型、已观察 status/evaluability 和相关 observation id；仍不得提供
倾向性诊断结论。

`branch_context`、hypothesis references 和 action score 可以由 Orchestrator 保留在 ledger provenance 中，但不得进入模型可见 payload。

## 输入 Contract

正式输入接口为 `ReviewerTaskRequestV1`，其 schema 位于：

- [`reviewer_task_request_v1.schema.json`](schemas/reviewer_task_request_v1.schema.json)

每个 request 至少包含：

- `schema_version`
- `request_id`
- `action_id`
- `plan_id`
- `question_id`
- `snapshot_id`
- `reviewer`
- `model_input`

`model_input` 严格限定为 `task_profile`、`target_features`、`primary_roi`、
`context_views` 和可选的 `feature_disagreements`。Registry version 由 Orchestrator
在 Ledger provenance 中固化，不暴露给模型。

Orchestrator 在调用前必须验证：

1. Reviewer 和 task profile 存在于 registry；
2. primary scale 与 task profile 兼容；
3. 所有 target feature 位于该 profile 的 allowlist；
4. context view role 和 scale 被 registry 允许；
5. `model_input` 不包含 hypothesis、final label 或 expected effect；
6. image hash、level-0 bbox、ROI id 和倍率 provenance 完整。

## 输出 Contract

Reviewer 的结构化观察接口为 `ReviewerObservationV1`：

- [`reviewer_observation_v1.schema.json`](schemas/reviewer_observation_v1.schema.json)

所有 Reviewer 使用相同顶层结构：

- `schema_version`
- `observation_id`
- `request_id`
- `reviewer`
- `task_profile`
- `primary_roi_id`
- `primary_magnification`
- `target_features`
- `quality`
- `findings`
- `incidental_findings`
- `overall_evidence_strength`
- `limitations`
- `does_not_decide_final_diagnosis`

`does_not_decide_final_diagnosis` 必须恒为 `true`。

### Finding 语义

每个 requested feature 必须恰好返回一条 finding。Finding 固定包含：

- `feature_id`
- `status`
- `status_confidence`
- `feature_evaluability`
- `scope`
- `evidence_text`
- `limitations`

`status` 只能是：

- `present`
- `absent`
- `uncertain`
- `not_evaluable`

`feature_evaluability` 只能是：

- `adequate`
- `limited`
- `not_evaluable`

约束如下：

- `absent` 只允许与 `feature_evaluability = "adequate"` 组合；
- `not_evaluable` 必须与 `feature_evaluability = "not_evaluable"` 组合；
- `status_confidence` 表示对当前 status 的置信度，因此明确判断“不可评估”时可以很高；
- `uncertain` 不是 negative evidence；
- `not_evaluable` 只是 limitation；
- cross-domain negative finding 必须由 registry 明确允许，并始终保持 ROI-local 语义。

`incidental_findings` 只能包含当前 task profile registry 中声明的 incidental allowlist，不能作为绕过 target feature contract 的自由输出通道。

### Quality block

所有 Reviewer 都必须返回 quality block，即使没有单独调用 `QualityMucosaReviewer`。Quality block 至少记录：

- `overall_evaluability`
- `status_confidence`
- `adequate_for_requested_features`
- `limitations`

`QualityMucosaReviewer` 仅在以下情况单独调用：

- upstream quality 为 uncertain；
- specialist 返回 `limited` 或 `not_evaluable`；
- 一个关键 `absent` 将改变 hypothesis ranking 或 stop decision；
- Reviewer disagreement 可能由质量或倍率不匹配导致。

Quality evidence 不覆盖已有 finding。若 Quality Reviewer 与 specialist 对同一 ROI/scale 的可评估性结论不一致，系统应建立 `quality_disagreement` Conflict Object；冲突解决前，相关 `absent` 不得进入 negative evidence。

## TSA 与 Dysplasia 边界

`TSAReviewer` 可以评估 TSA 固有形态，包括：

- ectopic crypt formation；
- slit-like serration；
- villiform/filiform serrated architecture；
- eosinophilic cytoplasm；
- pencillate nuclei。

这些 finding 不能自动解释为 high-grade dysplasia。

`DysplasiaReviewer` 保持 branch-agnostic，只评估 high-grade/definite dysplasia 证据。它不得输出 `SSLD`、`TSAD`、`TAD` 或 `TVAD`，也不得因当前 branch 未确定而拒绝评估。

当任一 D-capable morphology hypothesis 仍有竞争力，或存在 dysplasia-risk hotspot 时，`diagnostic_ready` 前必须完成充分的 `DysplasiaReviewer` 检查。若 HP、unclassified serrated 或 inflammatory candidate 中出现强 dysplasia evidence，应创建 morphology-dysplasia conflict 并重新检查 subtype，而不是忽略该证据。

## Ledger 与调用失败

Orchestrator 使用 `ReviewerLedgerRecordV1` 将 reviewer 结果追加到 Evidence Ledger：

- [`reviewer_ledger_record_v1.schema.json`](schemas/reviewer_ledger_record_v1.schema.json)

Ledger record 使用两种 `record_type`：

- `evidence`：包含 schema-valid `ReviewerObservationV1` 和完整 provenance；
- `invocation_failure`：记录 transport、timeout、model 或 schema failure，不伪造 morphology finding。

`not_evaluable` 是合法 observation，不属于 invocation failure。

默认 failure policy：

- transport、timeout 或 model failure：同一 request 最多重试一次；
- schema invalid：允许一次受约束 repair，repair 只能整理原始明确输出，不得补充新 finding；
- repair 后仍无效：追加 `invocation_failure`；
- ROI 不可评估：Planner 必须改选 ROI、倍率或 context，不在同一 ROI 上机械重试。

## 冲突处理

v1 不定义 `ConflictReviewer`。

Planner 应通过以下方式补充独立证据：

- 选择新的 ROI；
- 改变 primary scale 或 context view；
- 使用允许的独立信息源或 backend；
- 调用负责相关 feature 的专科 Reviewer。

Chief 负责判断冲突是否解决。默认 `max_conflict_resolution_rounds = 2`。同一 backend 对近似 crop 的重复输出不得通过简单多数投票当作独立证据。

## Legacy Runtime Import Mapping

历史 Trace/Navigate/Observe artifact 中的 `*_hits` 可在显式离线迁移时映射为：

| 当前字段 | Reviewer / task profile |
|---|---|
| `serrated_hits` | `SerratedArchitectureReviewer / serrated_overview` |
| `ssl_hits`, `hp_hits`, `abnormal_crypt_hits` | `SerratedArchitectureReviewer / ssl_hp_discrimination` |
| `tsa_hits` | `TSAReviewer / tsa_architecture` |
| `tsa_cytological_atypia_hits` | `TSAReviewer / tsa_signature_cytology` |
| `conventional_hits`, `conventional_architecture_checklist` | `ConventionalArchitectureReviewer` profiles |
| `dysplasia_hits`, `serrated_dysplasia_hits`, `conventional_dysplasia_hits` | `DysplasiaReviewer / high_grade_dysplasia_assessment` |
| `inflammatory_hits` | `InflammatoryReactiveReviewer` profiles |

该表只定义历史 artifact 的迁移语义，不属于当前 AgentFlow wire contract。

## 失败条件

以下任一情况都应判定 contract violation 或 invocation failure：

- Reviewer/task profile/scale 不在 registry 中；
- target feature 不属于 profile allowlist；
- requested feature 漏回或重复返回；
- 输出 registry 未允许的 incidental finding；
- `status_confidence` 超出 `[0, 1]`；
- `absent` 来自非 adequate feature；
- 模型输出最终诊断、D-suffix label 或病例级 hypothesis ranking；
- `TSAReviewer` 输出 high-grade dysplasia 结论；
- `DysplasiaReviewer` 输出 branch-specific final label；
- 缺失 ROI、image hash、坐标、倍率、model/prompt version 或 action provenance；
- Reviewer 自行请求新 ROI、调用其他工具或给出下一步调度命令。
