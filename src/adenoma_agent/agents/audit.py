from adenoma_agent.schemas import AuditReport, CaseResult


class AuditAgent(object):
    def _checklist_entries(self, payload):
        if isinstance(payload, dict):
            return list(payload.values())
        if isinstance(payload, list):
            return list(payload)
        return []

    def _checklist_completeness(self, payload):
        entries = self._checklist_entries(payload)
        if not entries:
            return 0.0
        if isinstance(payload, dict):
            assessed = [value for value in entries if isinstance(value, dict) and value.get("status") != "not_assessed"]
            return float(len(assessed)) / float(max(1, len(entries)))
        # list-based PathReasoner outputs are already selected evidence lists
        return 1.0 if entries else 0.0

    def _supporting_count(self, payload):
        entries = self._checklist_entries(payload)
        if not entries:
            return 0
        if isinstance(payload, dict):
            return len([value for value in entries if isinstance(value, dict) and value.get("status") == "supporting"])
        return len(entries)

    def _normalize_integrated_report(self, payload):
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            summary = str(payload.get("summary", "")).strip()
            recommendations = str(payload.get("recommendations", "")).strip()
            parts = [item for item in (summary, recommendations) if item]
            return "\n".join(parts)
        return str(payload or "")

    def _hierarchy_branch(self, hierarchy, key):
        payload = hierarchy.get(key, {})
        return payload if isinstance(payload, dict) else {}

    def __init__(self, bundle):
        self.bundle = bundle

    def run(self, case_spec, trace_result, navigation_result, observe_result, segmentation_artifact, timings):
        warnings = []
        errors = []
        input_mode = segmentation_artifact.get("input_mode", case_spec.input_mode)
        grid_first_bypassed_coords_export = bool(segmentation_artifact.get("grid_first_bypassed_coords_export", False))
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
        seen_dysplasia_gate = {}
        for step in review_steps:
            if not (0 <= step.x <= slide_w and 0 <= step.y <= slide_h):
                errors.append("Navigation step out of bounds: {0}".format(step.step_id))
            if float(step.m) not in (2.5, 5.0, 10.0):
                errors.append("Navigation magnification is invalid: {0}".format(step.step_id))
            if not step.need_to_see:
                errors.append("Navigation step missing need_to_see text: {0}".format(step.step_id))
            if not step.review_goal:
                errors.append("Navigation step missing review_goal: {0}".format(step.step_id))
            if not step.stage_gate:
                errors.append("Navigation step missing stage_gate: {0}".format(step.step_id))
            cluster_id = step.metadata.get("cluster_id")
            if step.stage_gate == "abnormal_crypt" and cluster_id:
                seen_dysplasia_gate[("serrated", cluster_id)] = True
            if step.stage_gate == "conventional_adenoma" and cluster_id:
                seen_dysplasia_gate[("conventional", cluster_id)] = True
            if step.stage_gate == "dysplasia" and cluster_id:
                branch = step.metadata.get("workflow_branch")
                gate_source = step.metadata.get("gate_source")
                if gate_source == "abnormal_crypt":
                    branch = "serrated"
                elif gate_source == "conventional_adenoma":
                    branch = "conventional"
                if branch not in ("serrated", "conventional", "conventional_adenoma"):
                    errors.append("Dysplasia review missing branch source for {0}".format(step.step_id))
                elif not seen_dysplasia_gate.get((branch, cluster_id)):
                    errors.append(
                        "Dysplasia review occurred before its branch gate for {0}".format(step.step_id)
                    )

        for cluster in trace_result["clusters"]:
            branch = cluster.metadata.get("workflow_branch")
            label = str(cluster.l or "").strip()
            conventional_labels = {"conventional", "conventional_adenoma_like", "tubular_adenoma_like", "tubulovillous_adenoma_like"}
            serrated_labels = {"serrated", "ssl_like_mucosa", "ssl_suspicious_mucosa", "ssl_high_priority_mucosa", "hp_like_mucosa", "tsa_like_mucosa", "unclassified_serrated_like_mucosa"}
            if label in conventional_labels and branch not in ("conventional", "conventional_adenoma"):
                errors.append(
                    "Conventional adenoma trace cluster missing conventional workflow branch for {0}".format(cluster.cluster_id)
                )
            if label in serrated_labels and branch != "serrated":
                errors.append("Serrated trace cluster missing serrated workflow branch for {0}".format(cluster.cluster_id))

        evidence_chain = [record.to_dict() for record in observe_result["records"]]
        if not evidence_chain:
            errors.append("No observation evidence generated.")
        for review in observe_result.get("global_reviews", []):
            if str(review.metadata.get("review_source", "")).strip() != "chief_model":
                errors.append("Observation global review must come from the real Chief model.")
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
        if input_mode == "grid_thumbnail" and grid_first_bypassed_coords_export:
            warnings.append("coords.h5 and patch export were skipped for grid-first execution.")
        elif segmentation_artifact.get("coords_returncode") != 0:
            errors.append("Failed to generate CLAM-compatible coords.h5.")

        serrated_checklist = observe_result["serrated_checklist"]
        abnormal_crypt_checklist = observe_result["abnormal_crypt_checklist"]
        dysplasia_checklist = observe_result["dysplasia_checklist"]
        conventional_adenoma_checklist = observe_result.get("conventional_adenoma_checklist", {})
        serrated_dysplasia_checklist = observe_result.get("serrated_dysplasia_checklist", {})
        conventional_dysplasia_checklist = observe_result.get("conventional_dysplasia_checklist", {})
        ssl_checklist = observe_result.get("ssl_checklist", {})
        hp_checklist = observe_result.get("hp_checklist", {})
        tsa_checklist = observe_result.get("tsa_checklist", {})
        tsa_cytological_atypia_checklist = observe_result.get("tsa_cytological_atypia_checklist", {})
        conventional_architecture_checklist = observe_result.get("conventional_architecture_checklist", conventional_adenoma_checklist)
        inflammatory_checklist = observe_result.get("inflammatory_checklist", {})
        serrated_support = self._supporting_count(serrated_checklist)
        abnormal_crypt_support = self._supporting_count(abnormal_crypt_checklist)
        dysplasia_support = self._supporting_count(dysplasia_checklist)
        conventional_support = self._supporting_count(conventional_adenoma_checklist)
        if serrated_support == 0 and conventional_support == 0:
            warnings.append("No serrated or conventional adenoma checklist criterion reached supporting status.")

        status = "ok"
        if errors:
            status = "fail"
        elif warnings:
            status = "warn"

        hierarchy = observe_result["hierarchical_prediction"]
        final_case_assessment = self._hierarchy_branch(hierarchy, "final_case_assessment")
        serrated_prediction = self._hierarchy_branch(hierarchy, "serrated_lesion_assessment")
        abnormal_crypt_prediction = self._hierarchy_branch(hierarchy, "abnormal_crypt_assessment")
        dysplasia_prediction = self._hierarchy_branch(hierarchy, "dysplasia_assessment")
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
                "serrated_checklist_completeness": self._checklist_completeness(serrated_checklist),
                "abnormal_crypt_checklist_completeness": self._checklist_completeness(abnormal_crypt_checklist),
                "dysplasia_checklist_completeness": self._checklist_completeness(dysplasia_checklist),
                "conventional_adenoma_checklist_completeness": self._checklist_completeness(conventional_adenoma_checklist),
                "serrated_dysplasia_checklist_completeness": self._checklist_completeness(serrated_dysplasia_checklist),
                "conventional_dysplasia_checklist_completeness": self._checklist_completeness(conventional_dysplasia_checklist),
                "ssl_checklist_completeness": self._checklist_completeness(ssl_checklist),
                "hp_checklist_completeness": self._checklist_completeness(hp_checklist),
                "tsa_checklist_completeness": self._checklist_completeness(tsa_checklist),
                "tsa_cytological_atypia_checklist_completeness": self._checklist_completeness(tsa_cytological_atypia_checklist),
                "conventional_architecture_checklist_completeness": self._checklist_completeness(conventional_architecture_checklist),
                "inflammatory_checklist_completeness": self._checklist_completeness(inflammatory_checklist),
                "serrated_proxy_correct": serrated_correct,
                "abnormal_crypt_proxy_correct": abnormal_crypt_correct,
                "dysplasia_proxy_correct": dysplasia_correct,
                "navigate_overlap_threshold": overlap_threshold,
                "abnormal_crypt_support_count": abnormal_crypt_support,
                "dysplasia_support_count": dysplasia_support,
                "conventional_adenoma_support_count": conventional_support,
                "input_mode": input_mode,
                "grid_first_bypassed_coords_export": grid_first_bypassed_coords_export,
                "classification_status": hierarchy.get("classification_status", final_case_assessment.get("classification_status")),
                "final_11_class": hierarchy.get("final_11_class", final_case_assessment.get("label")),
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
            integrated_report=self._normalize_integrated_report(observe_result["integrated_report"]),
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
            conventional_adenoma_checklist=conventional_adenoma_checklist,
            serrated_dysplasia_checklist=serrated_dysplasia_checklist,
            conventional_dysplasia_checklist=conventional_dysplasia_checklist,
            ssl_checklist=ssl_checklist,
            hp_checklist=hp_checklist,
            tsa_checklist=tsa_checklist,
            tsa_cytological_atypia_checklist=tsa_cytological_atypia_checklist,
            conventional_architecture_checklist=conventional_architecture_checklist,
            inflammatory_checklist=inflammatory_checklist,
            final_case_assessment=final_case_assessment,
            label=case_spec.label,
            status=status,
            metadata={
                "question": case_spec.question,
                "reasoning_state": observe_result["reasoning_state"].to_dict(),
                "selector_mode": trace_result["payload"].get("mode"),
                "serrated_labels": self.bundle["runtime"]["data"]["serrated_labels"],
            },
        )
