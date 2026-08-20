# Architecture Evidence Reviewer Design

## 1. Status and Purpose

本文定义当前 AgentFlow v1 的 Reviewer 层及其公开 contract。Reviewer 是由 Planner 调度、面向单个 ROI 的结构化证据提取工具；它不是诊断 Agent。

当前项目状态必须明确区分工程接入与医学有效性：

- `src/adenoma_agent/agentflow/` 已接入 Planner、Orchestrator、Evidence Ledger、Reviewer Registry 和 Chief 的 contract-first 控制流。
- 当前正式 wire contract 由本文、`contracts/reviewer.md`、Registry 和 JSON Schemas 共同定义。
- Architecture、dysplasia hotspot 与 Reviewer 视觉模型仍需训练或任务级验证；控制面可运行不等于模型具有临床性能。
- 六个 Reviewer 是六个逻辑 contract，可以共享同一个 VLM backend；每次调用仍须记录独立的 reviewer、task profile、prompt version、model version 和 schema version。

Reviewer 的硬边界：

- 只观察 Orchestrator 提供的 primary ROI 及允许的 context views。
- 只回答请求中列出的中性 `target_features`，以及 registry allowlist 允许的 incidental findings。
- 不读取 hypothesis ranking、expected answer、final-label mapping 或 Chief 的倾向性结论。
- 不选择 ROI、不生成坐标、不安排下一步、不解决 case-level conflict。
- 不输出最终 11 类诊断，也不把 ROI-local evidence 外推为全切片结论。

Planner 负责选择 Reviewer、task profile 和已有 ROI candidate；Orchestrator 负责裁图、调用、校验、retry 和 provenance；Chief 负责跨 ROI、跨来源和空间证据的最终综合。

## 2. Magnification Semantics

Reviewer Registry 固定使用以下四档倍率语义：

| Scale | Reviewer evidence semantics |
|---|---|
| `2.5x` | 正式的 `roi_overview` evidence，用于大视野形态、component 分布和可评估性；不能替代 Spatial Evidence 的全切片 ratio、cluster 或 heterogeneity 统计。 |
| `5x` | 整体 architecture、component extent、mixing 以及 cluster-level context。 |
| `10x` | crypt-level architecture、crypt base 和局部结构复核。 |
| `20x` | 细胞学细节以及 high-grade/definite dysplasia 评估。 |

倍率是 feature/profile compatibility 的一部分，不只是图像元数据：

- `2.5x` 只允许 overview/evaluability profiles，不得用于 crypt-base、signature cytology 或 dysplasia 结论。
- `5x` 和 `10x` 可按 profile 作为 primary ROI；同一次请求可附 registry 允许的 parent/context view。
- `20x` 的 `DysplasiaReviewer` primary ROI 可附一个同区域的 `10x` parent context，但 10x context 不能代替 20x cytology assessment。
- `20x` 的 `TSAReviewer` 只提取 TSA signature cytology，不拥有 high-grade dysplasia feature。

## 3. Reviewer Registry

`ReviewerRegistryV1` 是 Reviewer、task profile、feature allowlist、倍率和 context-view 兼容性的唯一机器可读来源。下面的表是 registry 的规范性设计摘要；实现不得仅凭 reviewer 名称推断可见 feature。

| Reviewer | Task profile | Allowed primary scales | Main target features |
|---|---|---|---|
| `QualityMucosaReviewer` | `overview_evaluability` | `2.5x`, `5x` | overview tissue、mucosa、epithelium 和 artifact 可评估性 |
|  | `crypt_evaluability` | `10x` | crypt、crypt base、truncation 和 architecture visibility；可附 `5x` parent context |
|  | `cytology_evaluability` | `20x` | epithelial/cytology detail 是否足以评估；可附 `10x` parent context |
| `SerratedArchitectureReviewer` | `serrated_overview` | `2.5x`, `5x` | serration 分布和 overview morphology |
|  | `ssl_hp_discrimination` | `5x`, `10x` | crypt-base serration、basal dilation、horizontal/boot-shaped crypt 等 SSL/HP 鉴别证据 |
| `TSAReviewer` | `tsa_overview` | `2.5x`, `5x` | villiform/filiform serrated overview 和 TSA-like 分布 |
|  | `tsa_architecture` | `5x`, `10x` | ectopic crypt、slit-like serration 和 TSA architecture |
|  | `tsa_signature_cytology` | `10x`, `20x` | cytoplasmic eosinophilia、pencillate nuclei；不评估 HGD |
| `ConventionalArchitectureReviewer` | `conventional_overview` | `2.5x`, `5x` | conventional/tubular/villous overview 和 mixing |
|  | `tubular_villous_resolution` | `5x`, `10x` | tubular versus villous structure、gland complexity |
|  | `villous_extent_estimation` | `5x`, `10x` | ROI 内 villous component extent category |
| `DysplasiaReviewer` | `high_grade_dysplasia_assessment` | `20x` | branch-agnostic high-grade/definite dysplasia morphology；可附 `10x` parent context |
| `InflammatoryReactiveReviewer` | `inflammatory_overview` | `2.5x`, `5x` | inflammation、erosion/ulceration、granulation 和 stromal/lymphoid dominance |
|  | `reactive_mimic_resolution` | `5x`, `10x` | reactive/regenerative change、injury distortion 和 ROI-local counter-evidence |

### 3.1 `QualityMucosaReviewer`

职责是判断 ROI 和请求 feature 是否可评估，不判断病变类型。v1 `feature_allowlist` 为：

- `reviewable_mucosa`
- `epithelium_present`
- `crypts_visible`
- `crypt_base_visible`
- `cytology_resolvable`
- `artifact_limited`
- `necrosis_limited`
- `fold_or_blur_limited`
- `truncation_limited`
- `background_or_stroma_dominant`

它不得输出 serrated、TSA、conventional、inflammatory 或 dysplasia diagnosis。Quality evidence 只新增记录，不覆盖 specialist 已产生的 evidence。

### 3.2 `SerratedArchitectureReviewer`

职责是提取 SSL/HP 鉴别所需的 serrated architecture。v1 `feature_allowlist` 为：

- `serration_present`
- `serration_to_crypt_base`
- `basal_crypt_dilation`
- `horizontal_or_boot_shaped_crypt`
- `crypt_branching`
- `abnormal_maturation`
- `surface_limited_serration`
- `straight_crypt_bases`

它不得输出 `SSL`、`SSLD`、`HP`、`TSA`、`TSAD` 或 `Unclassified serrated adenoma`。TSA-specific architecture 和 cytology 由独立 `TSAReviewer` 拥有，避免同一 feature 被两个 specialist 作为主要结论重复提取。

### 3.3 `TSAReviewer`

职责是提取 traditional serrated adenoma (TSA) 的 architecture 和 signature cytology。v1 `feature_allowlist` 为：

- `ectopic_crypt_formation`
- `slit_like_serration`
- `villiform_or_filiform_serrated_architecture`
- `cytoplasmic_eosinophilia`
- `pencillate_nuclei`

它不得输出 `TSA`、`TSAD` 或其他最终标签，也不得输出 `high_grade_focus`、`high_grade_or_definite_dysplasia` 等 HGD 结论。Dysplasia hotspot 和后续 `DysplasiaReviewer` 调度由独立 routing signal 与 Planner 处理，不能借 TSA finding 越权完成。

### 3.4 `ConventionalArchitectureReviewer`

职责是提取 conventional adenoma architecture，区分 tubular、villous 和 mixed component，并在 ROI 内估计 villous extent。v1 `feature_allowlist` 为：

- `tubular_architecture`
- `villous_component_present`
- `villous_component_extent_estimate`
- `high_villous_component`
- `tubular_villous_mixing`
- `crowded_adenomatous_glands`
- `adenomatous_gland_complexity`
- `conventional_adenoma_like_epithelium`

`villous_component_extent_estimate` 使用结构化 category：`less_than_25_percent`、`25_to_75_percent`、`greater_than_75_percent` 或 `indeterminate`；该估计的 scope 仍是 `roi_local` 或 `roi_overview`，不能直接成为全切片比例。Reviewer 不得输出 `Tubular adenoma`、`TAD`、`Tubulovillous adenoma` 或 `TVAD`。

### 3.5 `DysplasiaReviewer`

职责是 branch-agnostic 地提取 high-grade/definite dysplasia 证据。v1 `feature_allowlist` 为：

- `nuclear_stratification`
- `hyperchromasia`
- `loss_of_polarity`
- `marked_cytologic_atypia`
- `mitotic_activity`
- `cribriform_or_complex_crowding`
- `high_grade_focus`
- `necrosis_or_dirty_necrosis`
- `surface_maturation_loss`
- `high_grade_or_definite_dysplasia`

本工作流中所有 `D` suffix 都表示 high-grade 或 definite dysplasia。普通 low-grade adenomatous dysplasia 不得单独触发 `SSLD`、`TSAD`、`TAD` 或 `TVAD`。

`DysplasiaReviewer` 的模型可见请求中不得出现 `branch_context`、serrated/conventional ranking 或目标 D-suffix。Chief 在 Ledger 中结合 architecture、空间共定位和 routing provenance，将独立 dysplasia evidence 投影到最终分支。

### 3.6 `InflammatoryReactiveReviewer`

职责是提取 inflammatory/reactive mimic 的正向证据和 ROI-local 反证。v1 `feature_allowlist` 为：

- `erosion`
- `ulceration`
- `granulation_tissue`
- `mixed_inflammation`
- `lymphoid_or_stromal_dominance`
- `reactive_regenerative_change`
- `architectural_distortion_from_injury`
- `adenomatous_architecture_absent`
- `serrated_architecture_absent`
- `dysplasia_not_evaluable_due_to_reactive_change`

它不得输出最终 `Inflammatory` diagnosis。任何 `*_absent` 只能在对应 feature evaluability 为 `adequate` 时返回，并必须标记 ROI scope。

### 3.7 No Dedicated Unclassified Reviewer

不新增 `UnclassifiedSerratedAdenomaReviewer`。`Unclassified serrated adenoma` 由 Chief 综合 serrated、TSA、conventional、dysplasia 和 Spatial Evidence 后投影。Reviewer 不应把“未满足 SSL/TSA 条件”直接等同于该最终类别。

## 4. Public Contracts

机器可读 contract 位于 `docs/model/contracts/schemas/`，使用 JSON Schema Draft 2020-12、显式 version 和严格未知字段拒绝策略：

- `reviewer_common_v1.schema.json`
- `reviewer_registry_v1.schema.json`
- `reviewer_task_request_v1.schema.json`
- `reviewer_observation_v1.schema.json`
- `reviewer_ledger_record_v1.schema.json`
- concrete registry `reviewer_registry_v1.json`

四个公开接口如下。

### 4.1 `ReviewerRegistryV1`

Registry 为每个逻辑 Reviewer 声明：

- reviewer id 和 version；
- reviewer 级 `feature_allowlist`；
- task profile 的 `task_profile`、`allowed_primary_magnifications`、`allowed_context_views` 和 `target_feature_allowlist`；
- context view 的 `view_role`、`allowed_magnifications` 和 `max_count`；
- `prohibited_outputs`；
- `prompt_version` 和 `observation_schema_version`。

Planner/Action Binder 必须使用 Registry 校验 reviewer/profile/feature/scale 兼容性，不能依赖 prompt 中的自然语言约定。

### 4.2 `ReviewerTaskRequestV1`

Request envelope 的 required 顶层引用为 `request_id`、`action_id`、`plan_id`、`question_id`、`snapshot_id` 和 `reviewer`，并包含严格收敛的 `model_input`。`model_input` 包含：

- task profile；
- 完整且去重的 `target_features`；
- primary ROI；
- 可选 context views；
- ROI/image id、image reference/hash、level-0 bbox、倍率、pixel dimensions 及可选 MPP 等技术元数据；
- 冲突复核时可选的 feature-level disagreement。

Orchestrator 必须从完整 envelope 构造中性的 model-visible payload。模型只看到：

- `task_profile`；
- `target_features`；
- ROI image/context images；
- 图像 hash、坐标、倍率和必要的技术元数据；
- 如为 conflict re-review，只看到相关 feature 的状态/可评估性 disagreement。

以下信息不得进入 model-visible payload：hypothesis ranking、expected effect/expected answer、最终类别映射、Chief diagnosis、branch preference 或 action score。

### 4.3 `ReviewerObservationV1`

所有 Reviewer 返回统一结构：

- request/reviewer/profile 引用；
- 统一 `quality` block；
- `findings`，请求中的每个 target feature 必须且只能出现一次；
- task-profile `incidental_feature_allowlist` 限制的 `incidental_findings`（该 allowlist 本身必须是 Reviewer `feature_allowlist` 的子集），且不得与请求的 target features 重叠；
- overall evidence strength，以 `level = none/weak/moderate/strong` 和 `[0, 1]` `score` 表示；
- limitations；
- `does_not_decide_final_diagnosis: true`。

每个 finding 至少包含：

- `feature_id`；
- `status`；
- `status_confidence`；
- `feature_evaluability`；
- `scope`；
- `evidence_text` 和 `limitations`；
- profile 允许时的可选结构化 quantitation，例如 villous extent category。

`confidence` 不再作为 finding 字段；`status_confidence` 明确表示模型对所选 status 的置信程度。

### 4.4 `ReviewerLedgerRecordV1`

Reviewer 本身不负责生成完整 provenance。Orchestrator 在 schema validation 后补齐 Ledger record，并以 append-only 方式写入：

- `evidence`：合法 Reviewer observation 及其完整 provenance；
- `invocation_failure`：transport、timeout、model 或 schema failure 及已执行 retry/repair。

Ledger provenance 固定保留 case、request/action/snapshot、planner action、reviewer、ROI、image hash、level-0 bbox、magnification、`model_id/model_version`、`prompt_id/prompt_version`、registry version、source models 和 recorded time；Ledger record 本身保留 schema version。新记录不覆盖旧 evidence，Quality re-review 或 conflict re-review 也必须保留各自独立 evidence id。

## 5. Finding Semantics

每个 finding 的 `status` 只能是：

- `present`：目标形态在该 ROI 中可见。
- `absent`：该 feature 在该 ROI 中可充分评估，并且未见目标形态。
- `uncertain`：可能存在目标形态，但证据弱、歧义大或局部质量不足以确定。
- `not_evaluable`：缺少组织、目标结构、合适倍率/视野或受 artifact/truncation 影响，无法评估该 feature。

必要约束：

- `absent` 仅在该 finding 的 `feature_evaluability = adequate` 时合法；limited ROI 中未看到目标不得转换为反证。
- `not_evaluable` 可以具有很高的 `status_confidence`，因为模型可以高度确定“无法评估”；不得用 `0.0` 隐式代表 not evaluable。
- `uncertain` 与 `not_evaluable` 不等同于 `absent`，不能进入 opposing evidence。
- 所有证据必须标记 `scope = roi_local` 或 `scope = roi_overview`。
- `roi_overview` 只说明提供的 overview ROI，不代表 whole-slide absence、ratio、cluster distribution 或 heterogeneity。
- 同一请求的 target feature 不得遗漏或重复。额外观察只能进入 `incidental_findings`，必须属于该 task profile 的 `incidental_feature_allowlist`，并且不能重复任一 requested target feature。

统一 quality block 固定包含 `overall_evaluability`、`status_confidence`、`adequate_for_requested_features` 和 `limitations`；per-feature `feature_evaluability` 描述具体 feature 能否评价。总体 quality adequate 不能自动使每个 feature adequate，反之单个 feature not evaluable 也不必否定整张 ROI。

## 6. Scheduling and Quality Policy

Planner 默认选择性调用 Reviewer，不在每个 ROI 上运行全部 Reviewer。

典型 routing：

- `2.5x`：overview/evaluability，定位大视野形态或 component；不作为高倍结论。
- `5x`：serrated/TSA/conventional/inflammatory overview、mixing 和 extent。
- `10x`：SSL/HP crypt-base、TSA architecture、tubular-villous resolution 和 reactive mimic。
- `20x`：TSA signature cytology 或 high-grade/definite dysplasia；两类任务由不同 Reviewer/profile 执行。

所有 specialist 都必须返回 quality block，但 `QualityMucosaReviewer` 是条件调用，而不是每个 ROI 的强制前置步骤。仅在以下情况额外调用：

- ROI 或目标 feature 的质量预估不确定；
- 某个关键 negative evidence 将依赖 `absent`；
- 不同证据源存在 quality-related conflict；
- specialist 返回总体 `limited`、`not_evaluable`，或关键 feature 不可评估。

Quality evidence 不覆盖已有 specialist evidence。若 Quality Reviewer 与 specialist 对可评估性有实质分歧，Orchestrator/Planner 建立 `quality_disagreement`：

- 保留双方 evidence id；
- 在分歧解决前，不得把受影响的 `absent` 当作反证；
- 优先换更合适倍率、ROI 或独立信息源复核，而不是在同一 crop 上多数投票。

### 6.1 Dysplasia readiness gate

当任一 D-capable hypothesis 仍有竞争力，或 candidate pool 中存在 dysplasia/atypia hotspot 时，在 `diagnostic_ready` 前必须完成充分的 `DysplasiaReviewer` 检查。不能因为 architecture branch 尚未完全确定而延迟 branch-agnostic dysplasia sampling，也不能把 TSA signature cytology 当作 HGD review。

## 7. Conflict, Retry, and Failure Policy

不新增 `ConflictReviewer`。Reviewer evidence 产生或澄清冲突，Planner 和 Chief 管理 Conflict Object 并决定是否解决。

冲突复核规则：

- 只向模型暴露 feature-level disagreement，不暴露哪一个 hypothesis 应获支持。
- 复核应更换 ROI、倍率或独立信息源；同 backend 对近似 crop 的重复调用不能按简单多数投票处理。
- 不同 ROI 的相反结果可能表示真实 spatial heterogeneity，不能自动视为模型冲突。
- 默认 `max_conflict_resolution_rounds = 2`；达到预算后保留 unresolved conflict，不强行提升诊断置信度。

失败处理规则：

- transport/model failure 默认重试一次，并把最终 outcome 写为 evidence 或 `invocation_failure`。
- schema invalid 允许一次受约束 repair；repair 仍失败则写 `invocation_failure`。
- 合法的 `not_evaluable` 是观察结果，不是调用失败；不得在同一 ROI 上机械重试。
- 对 not-evaluable 结果仍有高价值问题时，Planner 应选择新 ROI、合适倍率或新的信息源。

Reviewer 调用停止不等于诊断就绪。`no_useful_action`、`budget_exhausted` 或 `non_diagnostic_or_quality_limited` 必须保留 unresolved questions；只有满足 evidence coverage、dysplasia gate、空间覆盖和 major-conflict 条件才可使用 `diagnostic_ready`。

## 8. Legacy `*_hits` Import Mapping

历史 Trace/Navigate/Observe artifact 可由离线兼容适配层按 feature ownership 转换为 Reviewer task profiles：

| Current field | Next-generation profile mapping |
|---|---|
| `serrated_hits` | `serrated_overview` 或 `ssl_hp_discrimination`；其中 TSA-specific feature 转到对应 `tsa_*` profile。 |
| `abnormal_crypt_hits` | 按具体 feature 映射到 `ssl_hp_discrimination`、`tsa_architecture` 或 `tubular_villous_resolution`，不能按字段名整体假定 branch。 |
| `conventional_hits` | `conventional_overview`、`tubular_villous_resolution` 或 `villous_extent_estimation`。 |
| `serrated_dysplasia_hits` | `high_grade_dysplasia_assessment`；原 serrated 信息只通过 action/snapshot routing provenance 供 Chief 追溯，不进入 Reviewer 的 model-visible payload。 |
| `conventional_dysplasia_hits` | `high_grade_dysplasia_assessment`；原 conventional 信息只通过 action/snapshot routing provenance 供 Chief 追溯。 |
| `dysplasia_hits` | `high_grade_dysplasia_assessment`。 |
| quality/limitation text in findings | 按目标层级映射到 `overview_evaluability`、`crypt_evaluability` 或 `cytology_evaluability`。 |
| inflammatory/reactive text in level findings | 映射到 `inflammatory_overview` 或 `reactive_mimic_resolution`；当前没有专门 `inflammatory_hits` 时不得伪造旧字段。 |

状态转换规则：

- quality-adequate 的 `present` 可成为 supporting evidence；
- quality-adequate 的 `absent` 可成为 ROI-scoped opposing evidence；
- `uncertain` 保持 uncertain evidence；
- `not_evaluable` 只进入 limitation/evaluability，不得成为 opposing evidence。

该映射只用于历史 replay 或显式迁移工具，不属于当前 AgentFlow 主路径，也不允许将旧 `*_hits` 的 branch 标签泄露给 Reviewer 模型。

## 9. Chief Integration Boundaries

Chief 接收完整 Ledger 和空间证据后：

- 将 SSL architecture 与 high-grade/definite dysplasia evidence 投影为 `SSLD`；
- 将 TSA architecture 与独立 dysplasia evidence 投影为 `TSAD`；
- 将 tubular 或 tubulovillous architecture 与独立 dysplasia evidence 投影为 `TAD` 或 `TVAD`；
- 在 serrated、TSA、conventional 和 Spatial Evidence 均不足以归入既定 subtype 时，才考虑 `Unclassified serrated adenoma`；
- 对 quality disagreement、spatial heterogeneity 和未解决冲突保持显式记录。

这些是 Chief 的综合职责，不应回填到 Reviewer prompt 或 Reviewer observation。

## 10. Contract Validation Requirements

本设计的 fixture/schema validation 至少覆盖：

- 六个 Reviewer 各一组合法 request/observation；
- 成功、`not_evaluable` 和 `invocation_failure` Ledger records；
- 非法 reviewer/profile/scale、缺少 provenance、target feature 漏回或重复、越界 `status_confidence`、非法 status 和未知字段；
- final diagnosis 越界、`TSAReviewer` 输出 HGD、`DysplasiaReviewer` 输出 D-suffix；
- `2.5x` 只能用于 overview/evaluability profile；
- TSA 20x cytology 与 HGD feature ownership 隔离；
- 关键 `absent` 的 adequate-quality gate；
- hypothesis、branch preference 和 expected answer 不进入 model-visible payload。

这些 synthetic fixtures 只验证 contract 和 orchestration behavior，不证明 Reviewer 的医学性能。

## 11. Consistency Invariants

后续实现和文档更新必须保持：

- Reviewer set 固定包含本文六个逻辑 Reviewer，包括独立 `TSAReviewer`。
- 倍率语义固定为 `2.5x / 5x / 10x / 20x`。
- `D` 固定表示 high-grade/definite dysplasia，而不是普通 low-grade adenomatous dysplasia。
- Reviewer 不输出最终 11 类 diagnosis，不选择 ROI，不做 hypothesis ranking。
- Finding 固定使用 `present / absent / uncertain / not_evaluable` 和 `status_confidence`。
- 每条 evidence 都有 ROI scope；单个 ROI 不能证明 whole-slide absence。
- Quality 和 conflict re-review 采用 append-only evidence，不覆盖旧记录。
- 控制面实现状态与 Reviewer 医学性能必须分开表述；不得把 contract 已接入写成模型已临床验证。
