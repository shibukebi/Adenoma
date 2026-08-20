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

## Research notes

以下文件是独立实验或外部工具调研，不改变正式运行合同：

- [CONCH fine-grained abnormal epithelium experiment](experiments/conch_fine_grained_abnormal_epithelium_plan.md)
- [11-class external-tool search plan](model/external_tools/11_class_polyp_tool_search_plan.md)
- [External-tool candidate table](model/external_tools/external_tool_candidate_table.md)

旧 `Trace -> Navigate -> Observe(+Chief)`、Junior、grid harness/dashboard、旧 10x
Architecture 主干及其评分文档已从当前文档树移除。相关兼容代码或历史 artifact 即使仍在
仓库中，也不构成当前 AgentFlow 规范。
