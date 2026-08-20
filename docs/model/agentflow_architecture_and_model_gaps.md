# AgentFlow 架构与模型缺口交付说明

## 1. 文档目的与基准决策

本文说明依据 [`docs/Agent_workflow.md`](../Agent_workflow.md) 搭建的当前
adenoma AgentFlow v1 控制面、现有仓库中可以直接复用的能力，以及仍需训练、验证或补齐的模型与数据。

本轮实现采用以下不可变基准：

1. `five_x_patch_manifest.jsonl` 是 Mucosal Architecture Classifier 的正式输入接口；
2. Architecture Classifier 的 canonical primary scale 是 `5x`；
3. `10x` 和 `20x` 不替代该输入，它们由 ROI Manager 在后续 Reviewer 阶段按任务需要生成；
4. AgentFlow v1 使用独立 package/config/CLI；历史 Trace/Navigate/Observe 代码不再定义当前文档合同；
5. 没有临床有效权重时，只允许运行显式标记的 scripted/stub 模式，不允许静默使用 synthetic checkpoint 或 heuristic 冒充正式模型；
6. 当前实现属于 contract-first、non-clinical 控制面，不代表已经获得医学性能。

`Agent_workflow.md` 中的直接依据包括：

- Mucosa Extractor 输出与空间入口：第 1.1 节；
- `five_x_patch_manifest.jsonl` 正式接口：第 1.1.5 节；
- 5x Architecture Classifier：第 2.3 节；
- Spatial Evidence Evaluator：第 2.4 节；
- 无副作用 Planning Agent 与 append-only Ledger：第 3.1、3.3、3.7 节；
- morphology `2+1` dysplasia 结构：第 3.2 节；
- Chief 与检索式 guideline 输出：第 4.1 至 4.4 节。

## 2. Canonical AgentFlow

```text
Raw WSI
  -> Mucosa Extractor
       -> Tissue Context Evidence
       -> High-Recall Mucosa Search Mask
       -> five_x_patch_manifest.jsonl
  -> 5x Mucosal Architecture Predictor
       -> serrated / tubular / villous scores
       -> quality and context scores
       -> patch embedding reference
       -> prediction uncertainty
  -> Spatial Evidence Evaluator
       -> soft architecture ratios
       -> clusters / co-localization / mixing
       -> heterogeneity / uncertainty regions
  -> ROI Manager
       -> deterministic ROI candidate pool
  -> Evidence Ledger Snapshot
  -> Evidence Reducer
  -> Hypothesis Engine
  -> Discriminative Question Generator
  -> ROI + Reviewer Action Binder
  -> Action Scorer + Stop Policy
  -> PlanDecision
  -> Orchestrator
       -> validates latest snapshot
       -> crops selected ROI
       -> invokes one Reviewer action
       -> retries/repairs according to policy
       -> appends evidence or invocation failure
       -> creates a new Snapshot
       -> repeats Planning loop
  -> Chief Agent + Structured Knowledge Base
       -> final diagnosis or uncertain diagnosis
       -> guideline retrieval after diagnosis
       -> structured result with evidence provenance
```

该流程有四条必须保持的边界：

- Mucosa Extractor 只寻找和描述组织上下文，不生成 HP、SSL、TSA、adenoma 或 dysplasia 假设；
- Planning Agent 只读取不可变 Snapshot 并输出单步 `PlanDecision`，不裁图、不调用 Reviewer、不写 Ledger；
- Reviewer 只返回 ROI-local 或 ROI-overview 的结构化形态学观察，不输出最终 11 类诊断；
- Chief 可以决定证据是否足够，但需要补证时必须交回 Planner，不能自己发明 ROI 或直接调用 Reviewer。

## 3. 已搭骨架的模块边界

新控制面放在 `src/adenoma_agent/agentflow/` 下，并通过依赖注入隔离训练模型与确定性逻辑。
“已搭骨架”表示数据 contract、控制流边界和可替换端口已经定义；不表示相应视觉模型已完成训练。

### 3.1 Contracts

文件：`src/adenoma_agent/agentflow/contracts.py`

负责定义跨模块不可变对象及基本校验，包括：

- `ArchitecturePatchPrediction`
  - 强制 `scale = 5.0`；
  - 保存 slide、patch、level-0 bbox 和 mucosa coverage；
  - 保存 evaluability、serrated/tubular/villous、context 和 uncertainty；
  - 正式模型还必须保存 `embedding_ref` 或等价的可追溯 embedding 引用；
- `SpatialEvidenceSummary`；
- `ROICandidate`；
- `EvidenceRecord` 和 `LedgerSnapshot`；
- `ReducedFeature` 和 `EvidenceView`；
- `HypothesisAssessment`、`DiscriminativeQuestion`、`ReviewerAction`、`PlanDecision`；
- `ReviewerTaskRequest`、`ReviewerFinding`、`ReviewerObservation`；
- `ChiefDecision` 和 `AgentFlowResult`。

Contract 层负责拒绝以下语义错误：

- 非 5x Architecture primary prediction；
- 非法 level-0 bbox；
- 超出 `[0, 1]` 的分数；
- `absent` 与非 adequate evaluability 组合；
- Reviewer 输出最终诊断；
- 非 stop 决策没有 Reviewer action；
- 未知 stop reason。

### 3.2 Evidence Ledger

文件：`src/adenoma_agent/agentflow/ledger.py`

职责是：

- append-only 保存所有 evidence；
- 为每次有效追加生成新的 `snapshot_id`；
- 返回不可变 Snapshot；
- 拒绝 stale Snapshot；
- 同一个 `evidence_id` 的相同内容可幂等重放，不同内容必须报错；
- 支持 JSONL replay；
- 保存 `planner_action_id`、input snapshot 和 source provenance。

Ledger 不负责 evidence ranking、hypothesis ranking 或最终诊断。

### 3.3 Architecture Predictor Port

Architecture Predictor 必须只消费 Mucosa Extractor 生成的 canonical
`five_x_patch_manifest.jsonl`，按照其中的 `level0_bbox` 从原始 WSI 提取 5x 图像。

正式输出至少包括：

```json
{
  "patch_id": "slide_001__5x__r00001_c00002",
  "slide_id": "slide_001",
  "level0_bbox": [2048, 4096, 4096, 6144],
  "scale": 5.0,
  "mucosa_coverage": 0.74,
  "evaluable": 0.94,
  "architecture": {
    "serrated": 0.72,
    "tubular": 0.18,
    "villous": 0.35
  },
  "context": {
    "normal_mucosa_present": 0.05,
    "reactive_inflammatory_present": 0.22,
    "other_pattern_present": 0.03
  },
  "embedding_ref": "embeddings/slide_001.npy#row=42",
  "uncertainty": 0.31,
  "source_model": "mucosal_architecture_classifier_v1"
}
```

骨架允许注入：

- production predictor：加载经过验证的 checkpoint；
- scripted predictor：只用于测试控制流；
- unavailable predictor：生产模式缺少权重时明确抛出 `ModelUnavailableError`。

不允许在 production mode 下从 unavailable predictor 静默降级为 synthetic 或 scripted 输出。

### 3.4 Spatial Evidence Evaluator

文件：`src/adenoma_agent/agentflow/spatial.py`

该模块是确定性程序，不单独判断诊断类别。输入为整张 WSI 的 5x patch soft predictions，输出：

- serrated、tubular、villous、normal、reactive 和 non-evaluable ratio；
- spatial cluster；
- tubular-villous mixing；
- serrated-villous co-localization；
- architecture heterogeneity；
- high-uncertainty region；
- 产生 ROI candidate 所需的 component 和 provenance。

比率必须使用 soft score 和 mucosa coverage 加权，不能只统计 argmax/hard label patch 数量。

### 3.5 ROI Manager

文件：`src/adenoma_agent/agentflow/spatial.py`

ROI Manager 是确定性程序，是 Planner 唯一合法的像素坐标来源。

职责包括：

- 从大 component 或 multi-cluster context 生成 2.5x ROI-overview candidates；
- 从 5x architecture cluster 生成 5x overview candidates；
- 从高分或高不确定区域生成 10x crypt-level candidates；
- 在选定的 10x parent 内生成多个 20x child candidates；
- 对 20x candidates 使用 abnormal epithelial、uncertainty、dysplasia-risk proxy 等信号排序；
- clipping、overlap 去重、field-of-view 和 parent-child containment 校验；
- 记录 source patch、cluster、component、scale 和 level-0 provenance。

Planner 只能从该候选池选择 `roi_id`，不得构造新的 bbox。

### 3.6 Evidence Reducer

文件：`src/adenoma_agent/agentflow/planner.py`

Reducer 将 tile、patch、cluster 和 Reviewer records 聚合为 feature-level `EvidenceView`。

必须保持以下语义：

- `missing`、`uncertain`、`not_evaluable` 不是 contradictory evidence；
- 只有 adequate ROI 上的 `absent` 才能成为 ROI-local negative evidence；
- 近重复 evidence 标记为 redundancy，但不删除原始 provenance；
- 相反 findings 建立 conflict，不使用简单多数投票自动消除；
- 输出 strength、quality、spatial coverage、source agreement、limitations 和 evidence ids。

### 3.7 Planning Agent

文件：`src/adenoma_agent/agentflow/planner.py`

Planning Agent 是纯决策内核，由以下部分组成：

1. 固定加载七个 morphology hypotheses；
2. 计算 plausibility、required evidence coverage 和 conflict；
3. 比较 top hypotheses，生成最有区分力的问题；
4. 将问题绑定到兼容的 Reviewer、task profile、scale 和现有 ROI candidate；
5. 按 discriminative value、ROI suitability、spatial coverage、cost 和 redundancy 排序；
6. 每轮最多输出一个 Reviewer action；
7. 输出 `diagnostic_ready`、`no_useful_action`、`budget_exhausted` 或
   `non_diagnostic_or_quality_limited`。

七个 morphology hypotheses 为：

```text
serrated / ssl
serrated / hp
serrated / tsa
serrated / unclassified_serrated_adenoma
conventional_adenoma / tubular_adenoma
conventional_adenoma / tubulovillous_adenoma
inflammatory / inflammatory_reactive
```

Dysplasia 必须保持独立轴；当 D-capable hypothesis 仍有竞争力时，没有完成充分的
`DysplasiaReviewer` 检查不能进入 `diagnostic_ready`。

### 3.8 Reviewer Runtime

文件：`src/adenoma_agent/agentflow/reviewer.py`

Reviewer Runtime 负责：

- 按 registry 校验 Reviewer、task profile、feature allowlist 和 scale compatibility；
- 构建不泄漏 hypothesis ranking、expected answer 和 final-label mapping 的模型可见请求；
- 调用共享或独立 VLM backend；
- 校验 requested feature 恰好返回一次；
- 执行最多一次 transport/model retry 或一次受约束 schema repair；
- 将合法 observation 或 `invocation_failure` 交给 Orchestrator 追加到 Ledger。

正式运行入口要求安装 `jsonschema>=4.18`，并对 request、observation 和
Reviewer ledger record 执行 Draft 2020-12 校验；scripted/offline 控制流测试在
依赖缺失时仍保留 registry-aware 的语义校验。Schema repair 只能整理原始明确输出，
不得新增、删除或改变 morphology finding。

逻辑 Reviewer 可以共享同一 VLM 权重，但必须分别记录 reviewer id、task profile、prompt version 和 model version。

当前 Registry 按最新版 `Agent_workflow.md` 固定 14 个 task profiles：

- Quality：`overview_evaluability`、`crypt_evaluability`、`cytology_evaluability`；
- Serrated：`serrated_overview`、`ssl_hp_discrimination`；
- TSA：`tsa_overview`、`tsa_architecture`、`tsa_signature_cytology`；
- Conventional：`conventional_overview`、`tubular_villous_resolution`、`villous_extent_estimation`；
- Dysplasia：`high_grade_dysplasia_assessment`；
- Inflammatory：`inflammatory_overview`、`reactive_mimic_resolution`。

其中 `Unclassified serrated adenoma` 由 Chief 综合 Spatial Evidence 与多个 Reviewer
证据后投影，不额外创建专用 Reviewer。

### 3.9 Chief Agent 与 Knowledge Base

文件：

- `src/adenoma_agent/agentflow/chief.py`；
- `src/adenoma_agent/agentflow/knowledge.py`。

Chief 接收：

- ranked hypotheses；
- Architecture/Spatial/Reviewer evidence；
- Evidence Ledger summary；
- active conflicts；
- Structured Knowledge Base。

Chief 负责 morphology 与独立 dysplasia 轴的最终 11 类投影。对于
`unassessed`、`not_evaluable` 和 `conflicting` dysplasia state，禁止因没有 positive finding
而自动投影为 non-D 类。

Guideline retrieval 只能在诊断确定后执行，且 recommendation 必须带：

- source；
- guideline/rule version；
- retrieval key；
- recommendation text。

若本地知识库没有匹配规则，系统应返回 guideline unavailable，而不是由 LLM 自由生成。

当前 `KnowledgeBase` 已保留结构化 `guidelines` mapping、加载接口和
`guideline_for_label()` 查询边界，但默认 guideline entries 仍为空。这只能验收
retrieval contract 和缺失规则时的行为，不能视为已经整理或审核了权威指南库。
真实 treatment、resection、surveillance 和 pathology reporting recommendation
仍需根据权威来源整理、版本化、审阅并补充引用。

### 3.10 Orchestrator

文件：`src/adenoma_agent/agentflow/orchestrator.py`

配置与入口：

- `configs/agentflow/runtime_v1.yaml`；
- `scripts/run_agentflow_v1.py`。

CLI 可从 compact Mucosa output、canonical 5x manifest 或预计算 Architecture
predictions 起步；Mucosa output 模式会同时桥接 tissue-context evidence。脚本化
Architecture/Reviewer 只在显式传入 `--allow-synthetic-stub` 时允许。

Orchestrator 是唯一有副作用的控制模块，负责：

- 初始化 Ledger 并追加 Tissue、Architecture 和 Spatial Evidence；
- 请求最新 Snapshot 并调用 Planner；
- 拒绝基于 stale Snapshot 的 PlanDecision；
- 根据 action 裁图并调用 Reviewer；
- 记录 request、response、retry、repair、latency 和完整 provenance；
- append Reviewer evidence 或 invocation failure；
- 在每次写入后重新执行完整 Planning loop；
- stop 后调用 Chief；
- 保存可 replay 的最终结果。

## 4. 正式产物建议

Mucosa Extractor 已有正式目录 contract：

```text
mucosa_extractor/
├── manifest.json
├── tile_index.jsonl
├── five_x_patch_manifest.jsonl
├── errors.jsonl
└── slides/<slide_id>/
    ├── maps.npz
    └── qc_panel.png
```

新 AgentFlow 建议保持以下运行目录：

```text
agentflow_run/
├── run_manifest.json
├── mucosa_extractor/
├── architecture/
│   ├── manifest.json
│   ├── patch_evidence.jsonl
│   ├── embeddings/
│   └── errors.jsonl
├── spatial/
│   ├── spatial_evidence.json
│   ├── clusters.jsonl
│   └── roi_candidates.jsonl
├── ledger/
│   └── evidence.jsonl
├── planning/
│   └── plan_decisions.jsonl
├── reviewers/
│   ├── requests.jsonl
│   ├── observations.jsonl
│   ├── invocation_records.jsonl
│   └── invocation_audit.jsonl
└── chief/
    ├── final_decision.json
    └── guideline_retrieval.json
```

所有产物必须带 schema/model/config/knowledge version，并允许从 JSONL 与 manifest 重建最终轨迹。

其中 `reviewers/invocation_records.jsonl` 保存正式的
`ReviewerLedgerRecordV1` envelope；`ledger/evidence.jsonl` 保存供 Reducer/Planner
使用的 feature-level normalized records。二者通过 `request_id`、`action_id` 和
`reviewer_ledger_record_id` 互相追溯，不把 formal Reviewer observation 展平后丢失。

## 5. 现有能力复用清单

### 5.1 可直接复用

| 能力 | 现有位置 | 复用方式 |
|---|---|---|
| Mucosa Extractor v1 | `src/adenoma_agent/mucosa_extractor.py` | 直接作为新 flow 首阶段 |
| Mucosa CLI | `scripts/run_mucosa_extractor.py` | raw WSI/artifact 运行入口 |
| UNI + PathPrism 服务 | `scripts/uni_prismnet_roi_server.py` | 生成 CRC100K 九类概率 |
| UNI 权重 | `models/UNI/weights/pytorch_model.bin` | Mucosa 与 Architecture embedding backbone |
| PathPrism 权重 | `models/PathPrism/prismnet_linprobe.pt` | Mucosa v1 classifier |
| bbox/坐标工具 | `src/adenoma_agent/utils.py` | ROI clipping、overlap、level-0 映射 |
| WSI crop 流程 | `src/adenoma_agent/adapters/cropper.py`、crop helper | 改造后服务 ROICandidate |
| JSONL logger/replay 模式 | `logger.py`、`replay.py` | 扩展 AgentFlow 事件和 replay |
| Reviewer contract/registry | `docs/model/contracts/reviewer.md`、`schemas/` | 作为 Reviewer runtime 机器合同 |
| Chief 模型权重 | `models/DeepSeek-R1-Distill-Qwen-14B/32B` | 可作为结构化 Chief backend |
| Image VLM 与 adapter | `models/Qwen2.5-VL-7B-Instruct`、`models/PathReasoner-R1` | 可作为共享 Reviewer backend 起点 |

### 5.2 可复用但必须适配

| 能力 | 已知限制 | 需要的适配 |
|---|---|---|
| `architecture_models.py` | 当前主要返回七维 logits | 增加 5x inference wrapper、embedding reference、uncertainty 和 model card |
| `architecture_experiment.py` | 当前按 10x parent + 20x children 构造实验 ROI | 新建 5x manifest dataset，不直接接 runtime |
| `run_mucosal_architecture_experiment.py` | smoke 使用 synthetic labels | 保留 encoder/train utilities，替换数据入口和临床 checkpoint 选择 |
| 现有 Observe/Chief | 面向 ObservationRecord/global_reviews | 改为 ReviewerObservation、Ledger summary 和 Knowledge Base 输入 |
| 当前 11 类 projection 代码 | 与旧 branch-gated workflow 耦合 | 迁移为 morphology `2+1` dysplasia 规则并覆盖 unresolved states |
| dashboard/replay | 不认识 Ledger、PlanDecision、Conflict | 增加新 flow 视图，不覆盖旧链视图 |

## 6. 模型缺口分级

### 6.1 必须训练：5x Mucosal Architecture Classifier

这是 `Agent_workflow.md` 明确要求、但当前没有医学有效权重的核心模型。

模型至少应输出：

- `p(evaluable)`；
- `p(serrated)`；
- `p(tubular)`；
- `p(villous)`；
- `p(normal_mucosa_present)`；
- `p(reactive_inflammatory_present)`；
- `p(other_pattern_present)`；
- patch embedding 或稳定 embedding reference；
- calibrated prediction uncertainty。

现有 A/B/C/D0/D1/D2 checkpoint 只来自 synthetic smoke annotations。现有 smoke report 已明确：

- 输入图像来自真实 WSI；
- 所有当前 WSI 均为 SSLD；
- 标签是 synthetic；
- 只验证 crop、embedding、forward/backward、checkpoint 和软件 contract；
- 不允许医学性能解释。

因此这些 checkpoint 不能成为 production fallback，也不能用于阈值选择或性能声明。

### 6.2 建议训练：20x Dysplasia/Atypia Hotspot Model

AgentFlow 骨架可以先使用以下 high-recall proxy 生成 20x candidates：

- Mucosa Extractor 的 abnormal epithelial candidate；
- architecture/model uncertainty；
- 图像质量和核密度等确定性 proxy；
- 可选 CONCH/VLM prompt score。

但要降低遗漏 high-grade/definite dysplasia hotspot 的风险，建议单独训练 20x patch-level 模型，输出：

- high-grade/definite dysplasia risk；
- evaluability；
- uncertainty；
- hotspot localization/ranking score。

该模型只负责高召回候选排序，不输出 SSLD、TSAD、TAD 或 TVAD。

### 6.3 建议训练或微调：Reviewer VLM

六个逻辑 Reviewer 不要求六套独立权重，可以共享一个 VLM backend。
现有 Qwen2.5-VL + PathReasoner adapter 可以作为工程起点，但尚未证明满足新的 feature-level contract。

建议在积累结构化 ROI supervision 后进行 task-specific LoRA 或 instruction tuning，重点包括：

- adequate/limited/not-evaluable 判定；
- feature-level `present/absent/uncertain/not_evaluable`；
- SSL/HP crypt-base discrimination；
- TSA signature morphology 与 HGD 的边界；
- tubular/villous extent；
- inflammatory/reactive mimics；
- branch-agnostic high-grade/definite dysplasia。

Reviewer 微调不是软件骨架运行的硬阻塞项，但在进入医学评价前必须完成独立验证。

### 6.4 建议校准但不一定重新训练

- Mucosa `mucosa_threshold = 0.30`：当前是 operational default，需要高召回校准；
- Architecture 每个 sigmoid head 的 operating threshold；
- uncertainty threshold；
- Spatial cluster 和 co-localization 阈值；
- Planner action score 权重与 stop thresholds；
- Chief diagnostic confidence 映射。

### 6.5 无需训练的组件

以下组件应保持确定性或规则驱动：

- Evidence Ledger 和 Snapshot；
- Evidence Reducer；
- Spatial Evidence 基础统计；
- ROI Manager 几何和 overlap 规则；
- 七个固定 hypothesis template；
- Discriminative Question template 与 compatibility binding；
- Action scoring 和 budget/stop policy；
- morphology+dysplasia 的 11 类投影；
- guideline key-based retrieval；
- schema validation、retry、repair、cost 和 replay。

Chief 可以使用 LLM 做结构化综合和文字生成，但最终标签合法性、D suffix 规则、未解决状态和 guideline source 必须由确定性合同约束。

### 6.6 当前已有、无需重新获取的基础权重

- UNI；
- PathPrism CRC100K linear probe；
- CONCH，当前只作为独立验证候选，不进入 Mucosa v1 probability fusion；
- Qwen2.5-VL-7B-Instruct；
- PathReasoner-R1 adapter；
- DeepSeek-R1-Distill-Qwen-14B/32B。

拥有权重不等于模型已通过当前任务验证。

## 7. Architecture 训练数据要求

### 7.1 Canonical 样本单位

每个训练样本必须能回溯到 `five_x_patch_manifest.jsonl` 中的一条记录：

- `slide_id`；
- `patch_id`；
- `level0_bbox`；
- `mucosa_coverage`；
- `source_component_ids`；
- 原始 WSI 与 crop hash；
- extraction/magnification metadata。

不得只按 slide diagnosis 把所有 patch 自动赋成相同 architecture label。

### 7.2 标注合同

建议每个 5x ROI 使用多标签标注：

```json
{
  "quality": {
    "state": "evaluable",
    "limitations": []
  },
  "architecture": {
    "serrated": "present",
    "tubular": "absent",
    "villous": "uncertain"
  },
  "context": {
    "normal_mucosa_present": "absent",
    "reactive_inflammatory_present": "present",
    "other_pattern_present": "absent"
  },
  "annotation_confidence": 0.9,
  "annotator_id": "...",
  "label_source": "pathologist_roi_annotation"
}
```

训练语义：

- quality label 可用时始终计算 quality loss；
- non-evaluable ROI 屏蔽 architecture/context loss；
- `uncertain` 只屏蔽对应 head，不作为 negative；
- architecture/context 使用独立 sigmoid，不使用互斥 softmax；
- 同一 ROI 允许 serrated、tubular、villous 或 context 共存。

### 7.3 数据组成

训练集必须包含：

- SSL、HP、TSA 和其他 serrated morphology；
- tubular 与不同 villous extent 的 conventional adenoma；
- inflammatory/reactive polyp 和 injury mimic；
- normal mucosa；
- mucus/stroma/background-dominant hard negatives；
- fold、blur、crush、cautery、truncation 等 non-evaluable ROI；
- mixed architecture 与 transition region；
- 不同扫描仪、染色批次、机构和组织制备差异。

当前只有 SSLD WSI 的 smoke 集合不能满足上述要求。

### 7.4 划分与防泄漏

- 必须按 patient/slide 划分，不允许同一 WSI patch 跨 train/val/test；
- 同一患者多张切片应在同一 split；
- 若有多中心数据，至少保留一个独立 external test cohort；
- hard-negative mining 只能使用训练集或在冻结模型后进行；
- threshold 和 calibration 只能在 validation set 上确定，test set 不参与调参。

### 7.5 Coverage 建议

作为初始、未校准的 operational policy，可采用：

- inference：`mucosa_coverage >= 0.30`；
- primary training：`mucosa_coverage >= 0.60`；
- 更低 coverage ROI：保留为 quality、hard-negative 或 uncertainty 数据。

这些值来自既有 architecture 设计建议，不是 `Agent_workflow.md` 的医学性能结论，最终必须通过实际 recall/precision 与病理 review 校准。

## 8. Dysplasia Hotspot 与 Reviewer 数据要求

若训练 20x hotspot model，样本至少应包含：

- high-grade/definite dysplasia positive foci；
- ordinary low-grade adenomatous dysplasia；
- TSA signature cytology without HGD；
- reactive/regenerative atypia；
- inflammation、erosion、ulceration、dirty necrosis 等 mimics；
- non-evaluable high-power fields；
- 10x parent context 与 20x child containment provenance。

评价重点应为 region/top-k recall，而不是只看 patch accuracy。

Reviewer supervision 每条记录应包含：

- reviewer/task profile；
- target features；
- primary ROI 和允许的 context views；
- feature status、confidence、evaluability 和 scope；
- limitations；
- pathologist adjudication；
- model/prompt/schema version。

训练 payload 不应包含 top hypothesis、expected answer 或 final label，以避免诊断倾向泄漏。

## 9. 骨架验收门槛

以下验收不需要正式视觉模型权重。

### 9.1 Contract 与坐标

- 非 5x Architecture prediction 被拒绝；
- 每个 Architecture prediction 可反查原 manifest row；
- level-0 bbox 合法并位于 slide 边界内；
- 20x child 完全包含于 10x parent；
- ROI Manager 输出确定性 id 和顺序；
- Planner 不产生候选池外坐标。

### 9.2 Ledger 与 replay

- append 后旧 Snapshot 不变化；
- 每次有效 append 生成新 Snapshot；
- stale `PlanDecision.snapshot_id` 被拒绝；
- duplicate id 的不同内容被拒绝；
- JSONL replay 重建相同 evidence ids、Plan 顺序与结果；
- `not_evaluable` 和 `invocation_failure` 都能被保存且语义不同。

### 9.3 Reducer 与 Planner

- missing/uncertain/not-evaluable 不成为 contradictory evidence；
- 只有 adequate absent 才成为 ROI-local negative；
- 初始同时保留七个 morphology hypotheses；
- 一次低置信度证据不永久删除 hypothesis；
- 每轮至多一个 Reviewer action；
- 固定输入产生确定性 ranking 和 tie-break；
- 未完成必要 dysplasia assessment 时不得 `diagnostic_ready`；
- budget/quality stop 保留 unresolved questions，不提高诊断置信度。

### 9.4 Reviewer

- requested feature 恰好返回一次；
- `absent + limited` 被拒绝；
- `does_not_decide_final_diagnosis` 恒为 true；
- Reviewer 输出 D-suffix label 被拒绝；
- `DysplasiaReviewer` 保持 branch-agnostic；
- `TSAReviewer` 不输出 high-grade dysplasia；
- model-visible payload 不包含 hypothesis ranking、expected effect 或 final-label mapping；
- transport/model retry 和 schema repair 均不超过配置预算。

### 9.5 Chief

- 11 类 morphology+dysplasia mapping 全覆盖；
- ordinary low-grade atypia 不触发 D suffix；
- `unassessed/not_evaluable/conflicting` 不自动变成 non-D；
- HP、Unclassified serrated 或 Inflammatory 遇到强 dysplasia evidence 时创建 conflict；
- supporting evidence ids 全部可在 Ledger 中找到；
- guideline recommendation 带 source/version；
- 没有 guideline 时不自由生成。

### 9.6 无权重 E2E

至少应有一条完全离线 scripted E2E：

```text
saved/synthetic mucosa evidence
  -> scripted 5x architecture predictor
  -> real Spatial Evaluator + ROI Manager
  -> real Ledger/Reducer/Planner
  -> scripted Reviewer backend
  -> real Orchestrator loop
  -> rule-based Chief + local KB fixture
```

该测试必须断言：

- 所有输出显式带 `synthetic_stub=true`、`non_clinical=true`；
- action 使用最新 Snapshot；
- Reviewer evidence append-only；
- stop 后不再调用 Reviewer；
- 网络和生产权重不会被意外加载；
- replay 后结果一致。

## 10. 模型验收门槛

### 10.1 Architecture 模型进入集成测试前

- checkpoint metadata 明确 `synthetic_smoke_only=false`；
- 训练数据来自 pathologist ROI annotation；
- patient/slide split 无泄漏；
- 每个 head 报告 support、prevalence、AUROC、AUPRC 和 confidence interval；
- 报告 quality gate 前后各类 sensitivity/precision；
- 提供 Brier score、ECE 或等价 calibration 指标；
- uncertainty 能富集错误/疑难样本，而不是与 confidence 简单重复；
- embedding 版本、维度和存储格式固定；
- inference manifest 保存 encoder、head、checkpoint hash、input scale/FOV 和 preprocessing；
- 所有阈值在 test 前冻结。

### 10.2 Architecture 模型进入医学评价前

- 由病理负责人预先定义每个 head 的最低 sensitivity/precision 或 recall-at-review-budget 门槛；
- 完成独立机构或扫描仪外部验证；
- 对 serrated/conventional/inflammatory mimics 进行盲法错误分析；
- 评估 Mucosa mask 漏检对 Architecture recall 的级联影响；
- 对 mixed architecture 和低 coverage ROI 单独报告；
- 禁止只使用 patch-level random split 或 slide diagnosis weak label 声明性能。

### 10.3 Dysplasia hotspot 模型

- 以 lesion/region-level sensitivity 和 top-k hotspot recall 为主指标；
- 单独报告 HGD、low-grade、TSA cytology 和 reactive mimic；
- 评估在固定 Reviewer budget 下的漏检率；
- 保留 uncertainty/not-evaluable 通道；
- 不直接输出 D-suffix diagnosis。

### 10.4 Reviewer VLM

- schema-valid response rate；
- requested-feature completeness；
- evaluability agreement；
- present/absent/uncertain/not-evaluable 与病理医师一致性；
- 对 negative evidence 的 false-absence rate；
- TSA cytology/HGD 边界错误；
- dysplasia branch leakage 和 final-label leakage；
- 不同 ROI、scale、scanner 的稳定性。

未达到上述门槛时，可以继续用于数据生产或受监督实验，但不能标记为 clinical/production-ready。

## 11. 已解决的 5x 与 10x 设计冲突

### 11.1 冲突来源

`Agent_workflow.md` 第 2.3 节明确规定：

```text
Mucosal Architecture Classifier
  reads five_x_patch_manifest.jsonl
  extracts 5x patches
  predicts serrated / tubular / villous / embedding / uncertainty
```

已删除的历史 10x Architecture 流程文档以及仍保留的实验代码采用另一套研究设计：

- 以 `1024 x 1024` level-0 bbox 作为 10x parent；
- 使用四个 `512 x 512` 20x children；
- 比较 A/B/C/D0/D1/D2 representation；
- 当前 `architecture_experiment.py` 读取旧 artifact `manifest.jsonl`，不是正式 `five_x_patch_manifest.jsonl`。

两者不能在 runtime 中被视为同一个输入合同。冲突文档已从当前 `docs/` 移除；
实验代码仅保留为研究资产。

### 11.2 本轮解决方案

本轮以用户指定的 `Agent_workflow.md` 为最高优先级：

1. Architecture runtime 只接受 `5x` canonical predictions；
2. `ArchitecturePatchPrediction` 应在 contract 层拒绝非 5x scale；
3. `five_x_patch_manifest.jsonl` 是唯一正式上游接口；
4. 现有 10x/20x A/B/C/D experiment 保留为研究和表征复用，不直接接 production runtime；
5. 10x/20x 由 ROI Manager 在 Architecture/Spatial Evidence 之后生成，服务 crypt-level Reviewer 和 DysplasiaReviewer；
6. 现有 architecture model head、masked loss 和 UNI embedding 工具可以复用，但必须新建 5x dataset/inference wrapper；
7. 模型 card 和 checkpoint metadata 必须明确写出 `primary_scale=5x` 和物理 FOV。

### 11.3 未来若重新引入 10x Architecture Model

未来允许通过独立、显式版本进行实验，例如：

```text
architecture_input_v2_10x
```

但必须满足：

- 新 schema/version；
- 独立模型 card 和 checkpoint；
- 与 5x 模型的 paired held-out comparison；
- 不复用同一字段名掩盖 scale/FOV 差异；
- 不静默替换 v1 canonical input；
- 经评审后才允许更改 AgentFlow runtime 配置。

## 12. 当前不能宣称完成的事项

即使控制流骨架和 scripted E2E 全部通过，仍不能宣称：

- 5x Architecture Classifier 已具备医学性能；
- Mucosa mask 已达到临床高召回；
- Reviewer VLM 能可靠判断 crypt-base morphology 或 high-grade dysplasia；
- Chief confidence 已校准；
- guideline recommendation 已完成临床审核；
- AgentFlow 已可用于临床诊断。

进入上述阶段前，至少还需要：

1. pathologist-labelled 5x Architecture dataset；
2. clinical architecture checkpoint；
3. Mucosa/Architecture/Reviewer operating-point calibration；
4. Structured Knowledge Base 和 guideline corpus 的版本化审核；
5. 独立 WSI cohort 上的端到端错误分析；
6. 人工 review 和 deployment safety gate。

## 13. 总结

当前工程可先完成并验收的是：

- 5x canonical 数据 contract；
- append-only Evidence Ledger 与 Snapshot；
- deterministic Spatial Evidence 和 ROI Manager；
- pure Planning Agent；
- schema-constrained Reviewer orchestration；
- rule-constrained Chief 与 guideline retrieval；
- scripted、可 replay、non-clinical E2E。

当前最明确的必训模型是 **5x Mucosal Architecture Classifier**。20x Dysplasia/Atypia
Hotspot Model 和 Reviewer VLM 微调属于强烈建议项，但不阻塞控制流骨架。Spatial、Ledger、Reducer、Planner、ROI 几何、11 类投影和 guideline retrieval 本身无需训练，应保持可测试、可回放和确定性。
