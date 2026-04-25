from adenoma_agent.schemas import AuditReport, CaseResult


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
            if not isinstance(cluster.d, bool):
                errors.append("d_k must be boolean for {0}".format(cluster.cluster_id))

        review_steps = [step for step in navigation_result["steps"] if step.metadata.get("action") != "stop"]
        seen_abnormal_crypt = {}
        for step in review_steps:
            if not (0 <= step.x <= slide_w and 0 <= step.y <= slide_h):
                errors.append("Navigation step out of bounds: {0}".format(step.step_id))
            if not (1.0 <= float(step.m) <= 5.0):
                errors.append("Navigation magnification is invalid: {0}".format(step.step_id))
            if not step.o:
                errors.append("Navigation step missing need_to_see text: {0}".format(step.step_id))
            if not step.review_goal:
                errors.append("Navigation step missing review_goal: {0}".format(step.step_id))
            if not step.stage_gate:
                errors.append("Navigation step missing stage_gate: {0}".format(step.step_id))
            cluster_id = step.metadata.get("cluster_id")
            if step.stage_gate == "abnormal_crypt" and cluster_id:
                seen_abnormal_crypt[cluster_id] = True
            if step.stage_gate == "dysplasia" and cluster_id and not seen_abnormal_crypt.get(cluster_id):
                errors.append(
                    "Dysplasia review occurred before abnormal crypt review for {0}".format(step.step_id)
                )

        evidence_chain = [record.to_dict() for record in observe_result["records"]]
        if not evidence_chain:
            errors.append("No observation evidence generated.")
        if not observe_result["integrated_report"]:
            errors.append("No integrated layered report generated.")
        if not observe_result["hierarchical_prediction"]:
            errors.append("Hierarchical prediction is missing.")
        if not observe_result["serrated_checklist"]:
            errors.append("Serrated lesion checklist is missing from the final report.")
        if not observe_result["abnormal_crypt_checklist"]:
            errors.append("Abnormal crypt checklist is missing from the final report.")
        if not observe_result["dysplasia_checklist"]:
            errors.append("Dysplasia checklist is missing from the final report.")
        if segmentation_artifact.get("coords_returncode") != 0:
            errors.append("Failed to generate CLAM-compatible coords.h5.")

        serrated_checklist = observe_result["serrated_checklist"]
        abnormal_crypt_checklist = observe_result["abnormal_crypt_checklist"]
        dysplasia_checklist = observe_result["dysplasia_checklist"]
        serrated_support = len([value for value in serrated_checklist.values() if value.get("status") == "supporting"])
        abnormal_crypt_support = len(
            [value for value in abnormal_crypt_checklist.values() if value.get("status") == "supporting"]
        )
        dysplasia_support = len([value for value in dysplasia_checklist.values() if value.get("status") == "supporting"])
        if serrated_support == 0:
            warnings.append("No serrated lesion checklist criterion reached supporting status.")

        status = "ok"
        if errors:
            status = "fail"
        elif warnings:
            status = "warn"

        hierarchy = observe_result["hierarchical_prediction"]
        serrated_prediction = hierarchy.get("serrated_lesion_assessment", {})
        abnormal_crypt_prediction = hierarchy.get("abnormal_crypt_assessment", {})
        dysplasia_prediction = hierarchy.get("dysplasia_assessment", {})
        serrated_correct = None
        abnormal_crypt_correct = None
        dysplasia_correct = None
        if case_spec.serrated_target is not None and "positive" in serrated_prediction:
            serrated_correct = int(bool(serrated_prediction["positive"])) == int(case_spec.serrated_target)
        if case_spec.abnormal_crypt_target is not None and "positive" in abnormal_crypt_prediction:
            abnormal_crypt_correct = int(bool(abnormal_crypt_prediction["positive"])) == int(
                case_spec.abnormal_crypt_target
            )
        if case_spec.dysplasia_proxy_target is not None and "positive" in dysplasia_prediction:
            dysplasia_correct = int(bool(dysplasia_prediction["positive"])) == int(case_spec.dysplasia_proxy_target)

        audit_report = AuditReport(
            status=status,
            warnings=warnings,
            errors=errors,
            metrics={
                "trace_cluster_count": len(trace_result["clusters"]),
                "trajectory_length": len(review_steps),
                "observation_count": len(observe_result["records"]),
                "serrated_checklist_completeness": len(
                    [value for value in serrated_checklist.values() if value.get("status") != "not_assessed"]
                )
                / float(max(1, len(serrated_checklist))),
                "abnormal_crypt_checklist_completeness": len(
                    [value for value in abnormal_crypt_checklist.values() if value.get("status") != "not_assessed"]
                )
                / float(max(1, len(abnormal_crypt_checklist))),
                "dysplasia_checklist_completeness": len(
                    [value for value in dysplasia_checklist.values() if value.get("status") != "not_assessed"]
                )
                / float(max(1, len(dysplasia_checklist))),
                "serrated_proxy_correct": serrated_correct,
                "abnormal_crypt_proxy_correct": abnormal_crypt_correct,
                "dysplasia_proxy_correct": dysplasia_correct,
                "navigate_overlap_threshold": overlap_threshold,
                "abnormal_crypt_support_count": abnormal_crypt_support,
                "dysplasia_support_count": dysplasia_support,
            },
        )
        return CaseResult(
            case_id=case_spec.case_id,
            serrated_target=case_spec.serrated_target,
            abnormal_crypt_target=case_spec.abnormal_crypt_target,
            dysplasia_proxy_target=case_spec.dysplasia_proxy_target,
            hierarchical_prediction=hierarchy,
            serrated_checklist=serrated_checklist,
            abnormal_crypt_checklist=abnormal_crypt_checklist,
            dysplasia_checklist=dysplasia_checklist,
            integrated_report=observe_result["integrated_report"],
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
                "serrated_labels": self.bundle["runtime"]["data"]["serrated_labels"],
            },
        )
