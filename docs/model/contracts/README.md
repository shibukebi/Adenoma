# AgentFlow Contracts

本目录只索引当前 `Evidence -> Planner -> Reviewer -> Ledger -> Chief` 工作流的正式合同。
上位规范以 [`docs/Agent_workflow.md`](../../Agent_workflow.md) 为准。

## Reviewer

- [Reviewer contract](reviewer.md)
- [Reviewer design](../architecture_evidence_reviewer_design.md)
- [Reviewer Registry](schemas/reviewer_registry_v1.json)
- [Reviewer Registry Schema](schemas/reviewer_registry_v1.schema.json)
- [Reviewer Common Schema](schemas/reviewer_common_v1.schema.json)
- [Reviewer Task Request Schema](schemas/reviewer_task_request_v1.schema.json)
- [Reviewer Observation Schema](schemas/reviewer_observation_v1.schema.json)
- [Reviewer Ledger Record Schema](schemas/reviewer_ledger_record_v1.schema.json)
- [Reviewer examples](examples/reviewer/)

Reviewer 的机器合同固定以下边界：

- Planner 只选择已有 ROI candidate、Reviewer 和 task profile；
- Reviewer 只能查看中性任务、primary ROI 和 Registry 允许的 context；
- finding 状态固定为 `present / absent / uncertain / not_evaluable`；
- `absent` 必须有 adequate quality，`not_evaluable` 不能当作 negative evidence；
- 成功观察和最终调用失败都以 append-only 记录写入 Ledger；
- Reviewer 不输出最终 11 类诊断。

## Chief

- [Chief contract](chief_pathologist.md)

Chief 读取最终 Snapshot、PlanDecision、ranked hypotheses、Conflict Objects 和
Structured Knowledge Base。只有 `diagnostic_ready`、关键冲突已解决且 dysplasia
轴可合法投影时，才输出 final；否则输出 structured uncertain decision。

## Runtime implementation

- `src/adenoma_agent/agentflow/contracts.py`
- `src/adenoma_agent/agentflow/ledger.py`
- `src/adenoma_agent/agentflow/planner.py`
- `src/adenoma_agent/agentflow/reviewer.py`
- `src/adenoma_agent/agentflow/orchestrator.py`
- `src/adenoma_agent/agentflow/chief.py`

这些合同已进入 non-clinical 控制面。合同接入不代表 Architecture、dysplasia hotspot、
Reviewer 或 Chief 模型已经完成临床训练与验证；模型状态见
[AgentFlow 架构与模型缺口](../agentflow_architecture_and_model_gaps.md)。
