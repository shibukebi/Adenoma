# Adenoma AgentFlow Vocabulary

本文件只定义当前 AgentFlow 的稳定诊断与证据术语。流程边界以
[`docs/Agent_workflow.md`](../../Agent_workflow.md) 为准。

## Morphology hypotheses

v1 从 Structured Knowledge Base 固定加载七个 morphology hypothesis，不由 LLM
自由生成：

| Pathway | Subtype |
|---|---|
| `serrated` | `ssl` |
| `serrated` | `hp` |
| `serrated` | `tsa` |
| `serrated` | `unclassified_serrated_adenoma` |
| `conventional_adenoma` | `tubular_adenoma` |
| `conventional_adenoma` | `tubulovillous_adenoma` |
| `inflammatory` | `inflammatory_reactive` |

所有合法 hypothesis 在初始轮次并存，通过 Evidence Reducer 和 Hypothesis Engine
排序；低置信度证据不能永久删除竞争 hypothesis。

## Independent dysplasia axis

`dysplasia_state` 固定为：

- `unassessed`
- `not_evaluable`
- `not_supported`
- `supported`
- `conflicting`

`D` 后缀只表示 high-grade/definite dysplasia，不表示普通 low-grade adenomatous
dysplasia。TSA intrinsic cytologic atypia 不能单独触发 `TSAD`。

## Final 11-class labels

| Morphology | Dysplasia mapping | Final label |
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

当 dysplasia 是 `unassessed`、`not_evaluable` 或 `conflicting` 时，不能自动投影为
non-D 类。`Unclassified serrated adenoma` 也不能作为 insufficient evidence 的兜底标签。

## Evidence statuses

Reviewer finding 固定使用：

- `present`
- `absent`
- `uncertain`
- `not_evaluable`

`absent` 必须建立在 adequate-quality ROI 上；`not_evaluable`、missing evidence 和
invocation failure 均不是 negative evidence。每条 finding 都保留 ROI、倍率、模型、
prompt、action 和 snapshot provenance。

## Magnification semantics

- `2.5x`：ROI overview / evaluability；
- `5x`：整体 architecture、component extent 与 mixing；
- `10x`：crypt-level architecture；
- `20x`：cytology 与 high-grade/definite dysplasia review。

Architecture Classifier 的 canonical 输入是独立的 `5x`
`five_x_patch_manifest.jsonl`；Reviewer 的 `10x/20x` 是后续 ROI 派生层，不能替代
正式 Architecture 输入。
