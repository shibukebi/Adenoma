from adenoma_agent.schemas import AuditReport, CaseResult
from adenoma_agent.utils import bbox_area


class AuditAgent(object):
    def __init__(self, bundle):
        self.bundle = bundle

    def run(self, case_spec, trace_result, navigation_result, observe_result, segmentation_artifact, timings):
        warnings = []
        errors = []
        slide_w, slide_h = trace_result["payload"]["thumbnail_meta"]["slide_dimensions_level0"]
        overlap_threshold = float(self.bundle["runtime"]["navigate"].get("overlap_threshold", 0.30))

        for cluster in trace_result["clusters"]:
            if cluster.l not in self.bundle["runtime"]["trace"]["labels"]:
                errors.append("Invalid trace label for {0}".format(cluster.cluster_id))
            if not isinstance(cluster.s, int):
                errors.append("Priority s_k must be an integer for {0}".format(cluster.cluster_id))

        review_steps = [step for step in navigation_result["steps"] if step.metadata.get("action") != "stop"]
        for step in review_steps:
            if not (0 <= step.x <= slide_w and 0 <= step.y <= slide_h):
                errors.append("Navigation step out of bounds: {0}".format(step.step_id))
            if not (1.0 <= float(step.m) <= 5.0):
                errors.append("Navigation magnification is invalid: {0}".format(step.step_id))
            if not step.o:
                errors.append("Navigation step missing need_to_see text: {0}".format(step.step_id))

        evidence_chain = [record.to_dict() for record in observe_result["records"]]
        if not evidence_chain:
            errors.append("No observation evidence generated.")
        if not observe_result["report"]:
            errors.append("No pathological report generated.")
        if not observe_result["report_checklist"]:
            errors.append("SSA checklist is missing from the final report.")
        if segmentation_artifact.get("coords_returncode") != 0:
            errors.append("Failed to generate CLAM-compatible coords.h5.")

        supporting_count = 0
        checklist = observe_result["report_checklist"]
        for value in checklist.values():
            if value.get("status") == "supporting":
                supporting_count += 1
        if supporting_count == 0:
            warnings.append("No checklist criterion reached supporting status.")
        if case_spec.binary_target is None:
            warnings.append("Binary target is missing for proxy evaluation.")

        status = "ok"
        if errors:
            status = "fail"
        elif warnings:
            status = "warn"

        final_prediction = observe_result["final_binary_prediction"]
        is_correct = None
        if case_spec.binary_target is not None and "positive" in final_prediction:
            is_correct = int(bool(final_prediction["positive"])) == int(case_spec.binary_target)

        audit_report = AuditReport(
            status=status,
            warnings=warnings,
            errors=errors,
            metrics={
                "trace_cluster_count": len(trace_result["clusters"]),
                "trajectory_length": len(review_steps),
                "observation_count": len(observe_result["records"]),
                "report_checklist_completeness": len(
                    [value for value in checklist.values() if value.get("status") != "not_assessed"]
                )
                / float(max(1, len(checklist))),
                "proxy_correct": is_correct,
                "navigate_overlap_threshold": overlap_threshold,
            },
        )
        return CaseResult(
            case_id=case_spec.case_id,
            binary_target=case_spec.binary_target,
            final_binary_prediction=final_prediction,
            report_checklist=checklist,
            pathological_report=observe_result["report"],
            segmentation_artifact=segmentation_artifact,
            trace_clusters=[cluster.to_dict() for cluster in trace_result["clusters"]],
            trajectory=[step.to_dict() for step in navigation_result["steps"]],
            evidence_chain=evidence_chain,
            cost={
                "estimated_case_cost_units": len(trace_result["clusters"]) + len(observe_result["records"]),
                "trace_attempts": len(trace_result["selection"].get("attempts", [])),
                "observation_steps": len(observe_result["records"]),
            },
            timing=timings,
            audit=audit_report.to_dict(),
            label=case_spec.label,
            status=status,
            metadata={
                "question": case_spec.question,
                "reasoning_state": observe_result["reasoning_state"].to_dict(),
                "selector_mode": trace_result["payload"].get("mode"),
                "binary_positive_label": self.bundle["runtime"]["data"]["binary_positive_label"],
            },
        )
