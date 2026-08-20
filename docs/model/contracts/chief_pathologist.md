# Chief Agent Contract

## 目标

Chief Agent 是当前 AgentFlow 的病例级综合与最终投影层。它读取已经完成 provenance
校验的 Evidence Package，不直接浏览运行目录、不裁图、不选择 ROI，也不调用 Reviewer。
需要补充证据时，控制权回到 Planning Agent。

当前实现位于 `src/adenoma_agent/agentflow/chief.py`，属于 rule-based、
contract-first、non-clinical 控制面。

## 输入边界

Chief 至少读取：

- 最终不可变 `LedgerSnapshot` 与 `snapshot_id`；
- `EvidenceView` / reducer 结果；
- 最终 `PlanDecision`、`stop_reason` 和 unresolved questions；
- 七个 morphology hypothesis 的排序与 coverage/conflict 分数；
- `SpatialEvidenceSummary`；
- active `ConflictObject`；
- versioned Structured Knowledge Base。

Reviewer evidence 可以来自以下六个逻辑 Reviewer：

- `QualityMucosaReviewer`
- `SerratedArchitectureReviewer`
- `TSAReviewer`
- `ConventionalArchitectureReviewer`
- `DysplasiaReviewer`
- `InflammatoryReactiveReviewer`

Reviewer 的 `2.5x / 5x / 10x / 20x` 证据均保留 ROI scope。`20x`
`DysplasiaReviewer` 负责 high-grade/definite dysplasia 评估；
`TSAReviewer` 的 signature cytology 不能替代该评估。

Chief 不得读取 Planner 的预期答案作为证据，也不得把 invocation failure、
`not_evaluable` 或未观察 feature 当作 negative evidence。

## Final 与 Uncertain Gate

只有同时满足以下条件，Chief 才能输出 `status = final`：

1. 最终 `PlanDecision.stop_reason = diagnostic_ready`；
2. 存在可合法投影的最高 morphology hypothesis；
3. required evidence coverage 达到 Planner/KB 门槛；
4. 没有 active major conflict；
5. 对 dysplasia-class-defining subtype，`dysplasia_state` 必须是
   `supported` 或 `not_supported`；
6. 最终标签存在于 versioned Knowledge Base 的合法映射中。

以下任一情况必须输出 `status = uncertain`，不得强制归入 non-D 类：

- `stop_reason` 是 `no_useful_action`、`budget_exhausted` 或
  `non_diagnostic_or_quality_limited`；
- dysplasia 状态是 `unassessed`、`not_evaluable` 或 `conflicting`；
- spatial/reviewer evidence 存在未解决冲突；
- 没有合法 morphology/dysplasia 投影；
- 图像质量或覆盖不足。

## 2+1 投影

Chief 使用“pathway -> subtype + 独立 dysplasia 轴”，不采用 branch-gated
dysplasia review。

| Morphology subtype | Dysplasia state | Final label |
|---|---|---|
| `ssl` | `not_supported` / `supported` | `SSL` / `SSLD` |
| `hp` | not class-defining | `HP` |
| `tsa` | `not_supported` / `supported` | `TSA` / `TSAD` |
| `unclassified_serrated_adenoma` | not class-defining | `Unclassified serrated adenoma` |
| `tubular_adenoma` | `not_supported` / `supported` | `Tubular adenoma` / `TAD` |
| `tubulovillous_adenoma` | `not_supported` / `supported` | `Tubulovillous adenoma` / `TVAD` |
| `inflammatory_reactive` | not class-defining | `Inflammatory` |

对于 `HP`、`Unclassified serrated adenoma` 和 `Inflammatory`，若存在强
high-grade/definite dysplasia evidence，Chief 必须创建 morphology-dysplasia
conflict 并返回 uncertain，而不是丢弃该证据。

## 输出 Contract

`ChiefDecision` 固定包含：

```json
{
  "status": "final | uncertain",
  "final_label": "SSLD | null",
  "final_diagnosis": "... | null",
  "diagnostic_confidence": 0.0,
  "morphology_hypothesis_id": "H_SSL | null",
  "dysplasia_state": "supported",
  "supporting_evidence_ids": [],
  "conflicting_evidence_ids": [],
  "unresolved_questions": [],
  "conflicts": [],
  "management_recommendation": null,
  "knowledge_version": "kb_v1",
  "non_clinical": true
}
```

所有 supporting/conflicting evidence 必须引用 Ledger 中实际存在的 evidence ID。
Chief 不覆盖旧 evidence，不修改 Reviewer observation，也不隐藏 unresolved questions。

## Guideline Retrieval

management recommendation 只能在 final diagnosis 之后，从 versioned local
Knowledge Base 检索。没有匹配条目时返回 `not_configured`；不得自由生成指南、来源或
management 建议。uncertain diagnosis 不返回 recommendation。

## 禁止行为

- 发明 ROI、坐标、倍率或 Reviewer finding；
- 直接调用 `QualityMucosaReviewer`、`SerratedArchitectureReviewer`、
  `TSAReviewer`、`ConventionalArchitectureReviewer`、`DysplasiaReviewer` 或
  `InflammatoryReactiveReviewer`；
- 用单个 ROI 证明 whole-slide absence；
- 把 `not_evaluable`、missing evidence 或 invocation failure 当作 absent；
- 把 TSA intrinsic atypia 直接映射为 `TSAD`；
- 在 active conflict 或 dysplasia 未评估时输出确定性 non-D 诊断；
- 自由生成 guideline citation。
