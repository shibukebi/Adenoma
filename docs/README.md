# Documentation Map

`docs/Agent_workflow.md` 是当前工作流的唯一上位规范。若其他说明与其冲突，以该文件、
机器可读 Registry/Schemas 和 `configs/agentflow/runtime_v1.yaml` 为准。

## Current workflow

- [Canonical AgentFlow](Agent_workflow.md)
- [AgentFlow implementation and model gaps](model/agentflow_architecture_and_model_gaps.md)
- [Mucosa Extractor v1](model/mucosa_extractor_v1.md)
- [AgentFlow vocabulary](model/vocab/adenoma_agent_vocabulary.md)

## Reviewer and Chief contracts

- [Contract index](model/contracts/README.md)
- [Reviewer design](model/architecture_evidence_reviewer_design.md)
- [Reviewer contract](model/contracts/reviewer.md)
- [Chief contract](model/contracts/chief_pathologist.md)
- [Reviewer schemas](model/contracts/schemas/)
- [Reviewer examples](model/contracts/examples/reviewer/)

## Current experiments

- [Frozen CONCH 5x baseline data audit](model/architecture_baseline_data_audit.md)
- [Frozen CONCH 5x baseline report](model/architecture_baseline_report.md)
- [Frozen CONCH 5x baseline runbook](model/architecture_baseline_runbook.md)
- [CONCH text architecture annotation preparation](model/conch_text_architecture_annotation_preparation_report.md)
- [Pathology evidence annotation MVP](../annotation-system/README.md)

These documents describe active experiment state and execution gates. They must not be
read as evidence that GPU execution, expert annotation, or model evaluation is complete.

## Reference and research notes

以下文件是独立实验或外部工具调研，不改变正式运行合同：

- [CONCH fine-grained abnormal epithelium experiment](experiments/conch_fine_grained_abnormal_epithelium_plan.md)
- [11-class external-tool search plan](model/external_tools/11_class_polyp_tool_search_plan.md)
- [External-tool candidate table](model/external_tools/external_tool_candidate_table.md)
- [Current architecture behavioral audit](model/current_architecture_behavioral_audit.md)
- [Agent behavioral loop engineering report](model/agent_behavioral_loop_integration_report.md)

## Legacy boundary

旧 `Trace -> Navigate -> Observe(+Chief)`、Junior、grid harness/dashboard、旧 10x
Architecture 主干及其评分文档已从当前文档树移除。相关兼容代码或历史 artifact 即使仍在
仓库中，也不构成当前 AgentFlow 规范。当前没有需要保留在 `docs/legacy/` 的 Markdown；
若以后恢复历史文档，应明确标记为 legacy/reference，不能与本页列出的 current normative
文档并列作为规范来源。
