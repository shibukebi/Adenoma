import json
import tempfile
from pathlib import Path

from adenoma_agent.utils import env_with_cuda_visible_devices, read_json, run_command, write_json


class BackendUnavailableError(RuntimeError):
    pass


class BackendExecutionError(RuntimeError):
    pass


class MultimodalStageBackend(object):
    name = "base"
    supported_stages = ()

    def invoke(self, request, bundle):
        raise NotImplementedError


class ExternalCommandStageBackend(MultimodalStageBackend):
    name = "external_command"
    supported_stages = ("trace", "navigate", "observe_step", "observe_report")

    def invoke(self, request, bundle):
        config = bundle["runtime"].get("backends", {}).get("external_command", {})
        if not config.get("enabled") or not config.get("command_prefix"):
            raise BackendUnavailableError("external_command backend is disabled")
        with tempfile.TemporaryDirectory() as tmpdir:
            request_path = Path(tmpdir) / "request.json"
            response_path = Path(tmpdir) / "response.json"
            write_json(request_path, request)
            command = list(config["command_prefix"]) + [
                "--stage",
                request["stage"],
                "--request-json",
                str(request_path),
                "--response-json",
                str(response_path),
            ]
            result = run_command(command, timeout=300)
            if result["returncode"] != 0:
                raise BackendExecutionError(result["stderr"] or result["stdout"])
            if not response_path.exists():
                raise BackendExecutionError("No response_json produced by external command backend")
            return {
                "backend": self.name,
                "output": read_json(response_path),
                "raw_text": result["stdout"],
                "latency_ms": result["latency_ms"],
            }


class LocalPathoR1StageBackend(MultimodalStageBackend):
    name = "local_patho_r1"
    supported_stages = ("trace", "observe_step")

    def invoke(self, request, bundle):
        config = bundle["runtime"].get("backends", {}).get("local_patho_r1", {})
        if not config.get("enabled"):
            raise BackendUnavailableError("local_patho_r1 backend is disabled")
        image_path = request["images"][0]
        if request["stage"] == "trace":
            question = _build_trace_patho_r1_prompt(request, bundle)
        elif request["stage"] == "observe_step":
            question = request["prompt"]["question"]
        else:
            raise BackendUnavailableError("local_patho_r1 only supports trace and observe_step")
        command = [
            bundle["runtime"]["paths"]["patho_r1_python"],
            bundle["runtime"]["paths"]["patho_r1_patch_qa_script"],
            "--image",
            image_path,
            "--question",
            question,
            "--model-id",
            bundle["runtime"]["models"]["patho_r1_model_id"],
            "--cache-dir",
            bundle["runtime"]["models"]["patho_r1_cache_dir"],
            "--max-new-tokens",
            str(config.get("max_new_tokens", 256)),
        ]
        patho_r1_env = env_with_cuda_visible_devices(
            bundle["runtime"].get("execution", {}).get("patho_r1_cuda_visible_devices")
        )
        result = run_command(command, timeout=300, env_overrides=patho_r1_env)
        if result["returncode"] != 0:
            raise BackendExecutionError(result["stderr"] or result["stdout"])
        text = (result["stdout"] or "").strip()
        if not text:
            raise BackendExecutionError("Empty Patho-R1 response")
        if request["stage"] == "trace":
            output = _build_trace_output_from_text(text, request, bundle)
        else:
            output = _build_text_driven_output(text, request, bundle)
        return {
            "backend": self.name,
            "output": output,
            "raw_text": text,
            "latency_ms": result["latency_ms"],
        }


class HeuristicStageBackend(MultimodalStageBackend):
    name = "heuristic"
    supported_stages = ("trace", "navigate", "observe_step", "observe_report")

    def invoke(self, request, bundle):
        stage = request["stage"]
        if stage == "trace":
            output = self._trace_output(request, bundle)
        elif stage == "navigate":
            output = self._navigate_output(request, bundle)
        elif stage == "observe_step":
            output = self._observe_step_output(request, bundle)
        elif stage == "observe_report":
            output = self._observe_report_output(request, bundle)
        else:
            raise BackendUnavailableError("Unsupported heuristic stage: {0}".format(stage))
        return {"backend": self.name, "output": output, "raw_text": "", "latency_ms": 0}

    def _trace_output(self, request, bundle):
        serrated_labels = bundle["runtime"]["trace"]["labels"]
        serrated_criteria = list(bundle["runtime"]["observe"].get("serrated_criteria", []))
        abnormal_crypt_criteria = list(bundle["runtime"]["observe"].get("abnormal_crypt_criteria", []))
        clusters = []
        for proposal in request["metadata"]["proposals"]:
            tissue_fraction = float(proposal["metadata"].get("tissue_fraction", 0.0))
            pale_fraction = float(proposal["metadata"].get("pale_fraction", 0.0))
            artifact_fraction = float(proposal["metadata"].get("artifact_fraction", 0.0))
            route_c_overlap = float(proposal["metadata"].get("route_c_hint_overlap", 0.0))
            area_fraction = float(proposal["metadata"].get("area_fraction", 0.0))
            if tissue_fraction < 0.08:
                label = "background"
                priority = 0
                need_crypt_review = False
                crypt_disorder_risk = 0
                review_stage = "mucosa_screening"
                reasons = ["low tissue fraction on overview screening"]
            elif artifact_fraction > 0.28:
                label = "artifact"
                priority = 0
                need_crypt_review = False
                crypt_disorder_risk = 0
                review_stage = "mucosa_screening"
                reasons = ["high artifact-like color fraction"]
            else:
                serrated_score = 1
                reasons = ["mucosal tissue retained after overview filtering"]
                if pale_fraction > 0.10:
                    serrated_score += 1
                    reasons.append("surface pallor / mucus-rich pattern supports a serrated lesion impression")
                if route_c_overlap > 0.05:
                    serrated_score += 1
                    reasons.append("region overlaps a route-C low-resolution hint")
                if 0.02 <= area_fraction <= 0.35:
                    serrated_score += 1
                    reasons.append("region size is suitable for structured lesion review")
                label = "serrated_suspicious_mucosa" if serrated_score >= 2 else "non_serrated_mucosa"
                priority = min(5, max(1, int(serrated_score)))
                crypt_disorder_risk = min(5, serrated_score + (1 if pale_fraction > 0.18 else 0))
                need_crypt_review = label == "serrated_suspicious_mucosa" and crypt_disorder_risk >= 3
                review_stage = "serrated_screening"
                if need_crypt_review:
                    reasons.append("cluster should enter abnormal crypt review")

            if label not in serrated_labels:
                raise BackendExecutionError("Heuristic trace produced unsupported label: {0}".format(label))
            clusters.append(
                {
                    "cluster_id": proposal["cluster_id"],
                    "l": label,
                    "s": priority,
                    "d": need_crypt_review,
                    "review_stage": review_stage,
                    "crypt_disorder_risk": crypt_disorder_risk,
                    "dysplasia_review_needed": False,
                    "desc": "; ".join(reasons),
                    "evidence": reasons,
                    "metadata": {
                        **proposal["metadata"],
                        "mucosa_retained": label not in ("background", "artifact"),
                        "serrated_criteria_focus": serrated_criteria,
                        "abnormal_crypt_criteria_focus": abnormal_crypt_criteria,
                    },
                }
            )
        return {"clusters": clusters}

    def _navigate_output(self, request, bundle):
        slide_dims = request["metadata"]["slide_dimensions_level0"]
        overlap_threshold = float(bundle["runtime"]["navigate"].get("overlap_threshold", 0.30))
        mag_to_region = bundle["budget"].get("magnification_to_region_size", {})
        clusters = sorted(
            request["metadata"]["clusters"],
            key=lambda item: (int(item["s"]), item["cluster_id"]),
            reverse=True,
        )

        steps = []
        prior_windows = []
        step_index = 0
        max_steps = int(bundle["budget"].get("max_navigation_steps", 8))
        for cluster in clusters:
            if int(cluster["s"]) <= 0 or cluster["l"] in ("background", "artifact"):
                continue
            if step_index >= max_steps:
                break
            bbox = cluster["cluster_bbox_level0"]
            center_x = int(round((bbox["x1"] + bbox["x2"]) / 2.0))
            center_y = int(round((bbox["y1"] + bbox["y2"]) / 2.0))
            step_specs = [
                (
                    1.0,
                    "serrated_lesion_assessment",
                    "mucosa_or_serrated",
                    "Confirm that this cluster contains reviewable mucosa and determine whether it belongs to the serrated lesion pathway.",
                )
            ]
            if cluster["d"]:
                step_specs.append(
                    (
                        2.5,
                        "abnormal_crypt_assessment",
                        "abnormal_crypt",
                        "Search for abnormal crypt architecture, including basal dilatation, branching, horizontal growth, and serration extending toward the crypt base.",
                    )
                )

            for magnification, review_goal, stage_gate, need_to_see in step_specs:
                if step_index >= max_steps:
                    break
                region_size = int(mag_to_region.get(str(float(magnification)), 256))
                from adenoma_agent.utils import bbox_overlap_ratio, clamp_center_point, normalized_point

                step_bbox = {
                    "x1": center_x - region_size // 2,
                    "y1": center_y - region_size // 2,
                    "x2": center_x + region_size // 2,
                    "y2": center_y + region_size // 2,
                }
                should_skip = False
                for prior_item in prior_windows:
                    if abs(float(prior_item["m"]) - float(magnification)) > 1e-6:
                        continue
                    if bbox_overlap_ratio(step_bbox, prior_item["bbox"]) > overlap_threshold:
                        should_skip = True
                        break
                if should_skip:
                    continue
                fixed_x, fixed_y = clamp_center_point(center_x, center_y, region_size, slide_dims)
                step_bbox = {
                    "x1": fixed_x - region_size // 2,
                    "y1": fixed_y - region_size // 2,
                    "x2": fixed_x + region_size // 2,
                    "y2": fixed_y + region_size // 2,
                }
                prior_windows.append({"bbox": step_bbox, "m": float(magnification)})
                steps.append(
                    {
                        "step_id": "step_{0:02d}".format(step_index),
                        "x": fixed_x,
                        "y": fixed_y,
                        "m": float(magnification),
                        "o": need_to_see,
                        "review_goal": review_goal,
                        "stage_gate": stage_gate,
                        "metadata": {
                            "cluster_id": cluster["cluster_id"],
                            "cluster_label": cluster["l"],
                            "cluster_priority": cluster["s"],
                            "region_size_level0": region_size,
                            "normalized_center": normalized_point(fixed_x, fixed_y, slide_dims),
                            "action": "inspect",
                        },
                    }
                )
                step_index += 1

        if not steps:
            steps.append(
                {
                    "step_id": "step_00",
                    "x": 0,
                    "y": 0,
                    "m": 1.0,
                    "o": "Stop navigation because no reviewable lesion cluster was retained.",
                    "review_goal": "integrated_impression",
                    "stage_gate": "end",
                    "metadata": {"action": "stop", "region_size_level0": 256},
                }
            )
        else:
            last = steps[-1]
            steps.append(
                {
                    "step_id": "step_{0:02d}".format(len(steps)),
                    "x": last["x"],
                    "y": last["y"],
                    "m": 1.0,
                    "o": "Stop navigation and consolidate the mucosa, serrated, abnormal crypt, and dysplasia evidence gathered so far.",
                    "review_goal": "integrated_impression",
                    "stage_gate": "end",
                    "metadata": {"action": "stop", "region_size_level0": 256},
                }
            )
        return {"steps": steps}

    def _observe_step_output(self, request, bundle):
        stats = request["metadata"]["image_stats"]
        step = request["metadata"]["step"]
        cluster = request["metadata"].get("cluster", {})
        serrated_criteria = list(bundle["runtime"]["observe"].get("serrated_criteria", []))
        abnormal_crypt_criteria = list(bundle["runtime"]["observe"].get("abnormal_crypt_criteria", []))
        dysplasia_criteria = list(bundle["runtime"]["observe"].get("dysplasia_criteria", []))
        background_fraction = float(stats.get("background_fraction", 0.0))
        pale_fraction = float(stats.get("pale_fraction", 0.0))
        tissue_fraction = float(stats.get("tissue_fraction", 0.0))
        cluster_priority = int(cluster.get("s", 0))
        crypt_disorder_risk = int(cluster.get("crypt_disorder_risk", cluster_priority))
        review_goal = step.get("review_goal")

        serrated_hits = {criterion: "not_assessed" for criterion in serrated_criteria}
        abnormal_crypt_hits = {criterion: "not_assessed" for criterion in abnormal_crypt_criteria}
        dysplasia_hits = {criterion: "not_assessed" for criterion in dysplasia_criteria}

        if review_goal == "serrated_lesion_assessment":
            if cluster.get("l") == "serrated_suspicious_mucosa":
                serrated_hits["serrated_lesion_context"] = "supporting"
                serrated_hits["serrated_surface_pattern"] = "supporting" if pale_fraction > 0.10 else "uncertain"
                serrated_hits["mucus_rich_surface"] = "supporting" if pale_fraction > 0.18 else "uncertain"
            elif cluster.get("l") == "non_serrated_mucosa":
                serrated_hits["serrated_lesion_context"] = "opposing"
                serrated_hits["serrated_surface_pattern"] = "opposing" if pale_fraction < 0.08 else "uncertain"
                serrated_hits["mucus_rich_surface"] = "opposing" if pale_fraction < 0.08 else "uncertain"
        elif review_goal == "abnormal_crypt_assessment":
            if pale_fraction > 0.18 and cluster_priority >= 4:
                abnormal_crypt_hits["serration_to_base"] = "supporting"
                abnormal_crypt_hits["mucus_cap"] = "supporting"
                abnormal_crypt_hits["abnormal_maturation"] = "supporting" if pale_fraction > 0.18 else "uncertain"
            else:
                abnormal_crypt_hits["serration_to_base"] = "uncertain"
                abnormal_crypt_hits["mucus_cap"] = "uncertain"
                abnormal_crypt_hits["abnormal_maturation"] = "uncertain"
            if crypt_disorder_risk >= 5 and tissue_fraction > 0.60 and pale_fraction > 0.15:
                abnormal_crypt_hits["basal_dilatation"] = "supporting"
                abnormal_crypt_hits["crypt_branching"] = "supporting"
                abnormal_crypt_hits["horizontal_growth"] = "supporting"
                abnormal_crypt_hits["boot_l_t_shaped_crypt"] = "supporting"
            elif crypt_disorder_risk >= 3:
                abnormal_crypt_hits["basal_dilatation"] = "uncertain"
                abnormal_crypt_hits["crypt_branching"] = "uncertain"
                abnormal_crypt_hits["horizontal_growth"] = "uncertain"
                abnormal_crypt_hits["boot_l_t_shaped_crypt"] = "uncertain"
        elif review_goal == "dysplasia_assessment":
            if cluster_priority >= 5 and tissue_fraction > 0.70 and pale_fraction < 0.12:
                dysplasia_hits["nuclear_enlargement_stratification"] = "supporting"
                dysplasia_hits["hyperchromasia"] = "supporting"
                dysplasia_hits["architectural_crowding"] = "uncertain"
                dysplasia_hits["mitotic_activity_atypia"] = "uncertain"
            elif tissue_fraction > 0.40:
                dysplasia_hits["nuclear_enlargement_stratification"] = "uncertain"
                dysplasia_hits["hyperchromasia"] = "uncertain"
                dysplasia_hits["architectural_crowding"] = "uncertain"
                dysplasia_hits["mitotic_activity_atypia"] = "uncertain"

        level_1_findings = _supporting_findings_from_hits(serrated_hits)
        level_2_findings = _supporting_findings_from_hits(abnormal_crypt_hits)
        level_3_findings = _supporting_findings_from_hits(dysplasia_hits)

        if background_fraction > 0.7:
            observation = "The crop is background-heavy and provides limited diagnostic tissue."
        elif review_goal == "serrated_lesion_assessment":
            observation = "Low magnification preserves the overall mucosal context for serrated pathway screening."
        elif review_goal == "abnormal_crypt_assessment":
            observation = "Intermediate magnification targets crypt architecture and abnormal serration distribution."
        else:
            observation = "High magnification focuses on cytologic atypia and dysplasia after abnormal crypt support."

        if review_goal == "serrated_lesion_assessment":
            stage_decision = "supports_serrated_lesion" if level_1_findings else "leans_non_serrated_or_indeterminate"
            reasoning = "This view decides whether retained mucosa belongs to the serrated pathway before abnormal crypt review."
            next_step = (
                "Proceed to abnormal crypt review." if cluster.get("d") else "Consolidate as a non-serrated or low-priority serrated mucosal region."
            )
        elif review_goal == "abnormal_crypt_assessment":
            stage_decision = (
                "supports_abnormal_crypt"
                if level_2_findings
                else "serrated_but_no_support_for_abnormal_crypt"
            )
            reasoning = "This view evaluates whether the crypt pattern supports abnormal crypt architecture within the serrated pathway."
            next_step = (
                "Proceed to gated dysplasia review."
                if stage_decision == "supports_abnormal_crypt"
                else "Do not enter dysplasia because abnormal crypt support is not established."
            )
        else:
            stage_decision = "dysplasia_supported" if level_3_findings else "dysplasia_not_supported_or_indeterminate"
            reasoning = "This view evaluates dysplasia or atypia after abnormal crypt support has been established."
            next_step = "Integrate mucosa, serrated, abnormal crypt, and dysplasia evidence into the final layered report."

        support_count = len(level_1_findings) + len(level_2_findings) + len(level_3_findings)
        confidence = min(
            0.95,
            max(
                0.05,
                0.20 + 0.20 * tissue_fraction + 0.10 * pale_fraction + 0.08 * support_count + 0.04 * cluster_priority,
            ),
        )
        return {
            "observation": observation,
            "reasoning": reasoning,
            "next_step": next_step,
            "level_1_findings": level_1_findings,
            "level_2_findings": level_2_findings,
            "level_3_findings": level_3_findings,
            "stage_decision": stage_decision,
            "serrated_hits": serrated_hits,
            "abnormal_crypt_hits": abnormal_crypt_hits,
            "dysplasia_hits": dysplasia_hits,
            "confidence": round(confidence, 4),
        }

    def _observe_report_output(self, request, bundle):
        serrated_criteria = list(bundle["runtime"]["observe"].get("serrated_criteria", []))
        abnormal_crypt_criteria = list(bundle["runtime"]["observe"].get("abnormal_crypt_criteria", []))
        dysplasia_criteria = list(bundle["runtime"]["observe"].get("dysplasia_criteria", []))
        records = request["metadata"]["records"]
        trace_clusters = request["metadata"]["trace_clusters"]

        serrated_checklist = _aggregate_hits(records, "serrated_hits", serrated_criteria)
        abnormal_crypt_checklist = _aggregate_hits(records, "abnormal_crypt_hits", abnormal_crypt_criteria)
        dysplasia_checklist = _aggregate_hits(records, "dysplasia_hits", dysplasia_criteria)

        serrated_assessment = _serrated_assessment(trace_clusters, serrated_checklist)
        abnormal_crypt_assessment = _abnormal_crypt_assessment(serrated_assessment, abnormal_crypt_checklist)
        dysplasia_assessment = _dysplasia_assessment(abnormal_crypt_assessment, dysplasia_checklist)
        integrated_impression = _integrated_impression(
            serrated_assessment,
            abnormal_crypt_assessment,
            dysplasia_assessment,
        )

        lines = []
        lines.append("Integrated Pathological Report")
        lines.append("Task: mucosa -> serrated lesion -> abnormal crypt -> dysplasia")
        lines.append("")
        lines.append("Serrated lesion assessment:")
        lines.append("- Impression: {0}".format(serrated_assessment["label"]))
        lines.append("- Supporting findings: {0}".format(_render_supporting_lines(serrated_checklist)))
        lines.append("")
        lines.append("Abnormal crypt assessment:")
        lines.append("- Impression: {0}".format(abnormal_crypt_assessment["label"]))
        lines.append("- Supporting findings: {0}".format(_render_supporting_lines(abnormal_crypt_checklist)))
        lines.append("")
        lines.append("Dysplasia assessment:")
        lines.append("- Impression: {0}".format(dysplasia_assessment["label"]))
        lines.append("- Supporting findings: {0}".format(_render_supporting_lines(dysplasia_checklist)))
        lines.append("")
        lines.append("Integrated impression:")
        lines.append("- {0}".format(integrated_impression))
        return {
            "hierarchical_prediction": {
                "serrated_lesion_assessment": serrated_assessment,
                "abnormal_crypt_assessment": abnormal_crypt_assessment,
                "dysplasia_assessment": dysplasia_assessment,
                "integrated_impression": integrated_impression,
            },
            "serrated_checklist": serrated_checklist,
            "abnormal_crypt_checklist": abnormal_crypt_checklist,
            "dysplasia_checklist": dysplasia_checklist,
            "integrated_report": "\n".join(lines),
        }


class StageBackendChain(object):
    def __init__(self, bundle):
        self.bundle = bundle
        self.backends = {
            "external_command": ExternalCommandStageBackend(),
            "local_patho_r1": LocalPathoR1StageBackend(),
            "heuristic": HeuristicStageBackend(),
        }

    def invoke(self, stage, chain_names, request):
        attempts = []
        request = {**request, "stage": stage}
        for backend_name in chain_names:
            backend = self.backends[backend_name]
            try:
                response = backend.invoke(request, self.bundle)
                response["attempts"] = attempts + [
                    {"backend": backend_name, "status": "ok", "latency_ms": response.get("latency_ms", 0)}
                ]
                return response
            except BackendUnavailableError as exc:
                attempts.append({"backend": backend_name, "status": "unavailable", "error": str(exc)})
            except Exception as exc:
                attempts.append({"backend": backend_name, "status": "error", "error": str(exc)})
        raise BackendExecutionError("All backends failed for stage {0}: {1}".format(stage, attempts))


def _blank_hits(criteria):
    return {criterion: "not_assessed" for criterion in criteria}


def _build_trace_patho_r1_prompt(request, bundle):
    header = request["prompt"]["question"]
    proposals = request["metadata"].get("proposals", [])
    lines = [header, "", "Candidate proposals in thumbnail pixel space:"]
    for proposal in proposals:
        bbox = proposal["cluster_bbox_thumb"]
        meta = proposal.get("metadata", {})
        lines.append(
            "- {cluster_id}: bbox=({x1},{y1},{x2},{y2}), tissue_fraction={tissue:.4f}, pale_fraction={pale:.4f}, artifact_fraction={artifact:.4f}, area_fraction={area:.4f}".format(
                cluster_id=proposal["cluster_id"],
                x1=bbox["x1"],
                y1=bbox["y1"],
                x2=bbox["x2"],
                y2=bbox["y2"],
                tissue=float(meta.get("tissue_fraction", 0.0)),
                pale=float(meta.get("pale_fraction", 0.0)),
                artifact=float(meta.get("artifact_fraction", 0.0)),
                area=float(meta.get("area_fraction", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "Return JSON only in the form:",
            '{',
            '  "clusters": [',
            '    {',
            '      "cluster_id": "cluster_00",',
            '      "l": "serrated_suspicious_mucosa",',
            '      "s": 4,',
            '      "d": true,',
            '      "review_stage": "serrated_screening",',
            '      "desc": "short reason",',
            '      "evidence": ["reason 1", "reason 2"]',
            '    }',
            '  ]',
            '}',
            "",
            'Allowed labels for "l": serrated_suspicious_mucosa, non_serrated_mucosa, background, artifact.',
            'Only use cluster_id values from the provided candidate proposals.',
        ]
    )
    return "\n".join(lines)


def _extract_first_json_object(text):
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _build_trace_output_from_text(text, request, bundle):
    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        json_blob = _extract_first_json_object(text)
        if json_blob:
            parsed = json.loads(json_blob)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("clusters"), list):
        raise BackendExecutionError("Patho-R1 trace response did not contain valid JSON clusters")

    proposals = request["metadata"].get("proposals", [])
    proposal_lookup = {proposal["cluster_id"]: proposal for proposal in proposals}
    output_clusters = []
    for item in parsed.get("clusters", []):
        cluster_id = item.get("cluster_id")
        if cluster_id not in proposal_lookup:
            continue
        label = item.get("l", "non_serrated_mucosa")
        if label not in bundle["runtime"]["trace"]["labels"]:
            label = "non_serrated_mucosa"
        evidence = item.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        output_clusters.append(
            {
                "cluster_id": cluster_id,
                "l": label,
                "s": int(item.get("s", 1)),
                "d": bool(item.get("d", False)),
                "review_stage": item.get("review_stage", "serrated_screening"),
                "crypt_disorder_risk": int(item.get("crypt_disorder_risk", item.get("s", 1))),
                "dysplasia_review_needed": False,
                "desc": item.get("desc", "Patho-R1 trace selection."),
                "evidence": [str(value) for value in evidence],
                "metadata": {"source": "patho_r1_trace"},
            }
        )
    if not output_clusters:
        raise BackendExecutionError("Patho-R1 trace response did not select any valid proposal cluster_id")
    return {"clusters": output_clusters}


def _build_text_driven_output(text, request, bundle):
    review_goal = request["metadata"]["step"].get("review_goal")
    serrated_criteria = list(bundle["runtime"]["observe"].get("serrated_criteria", []))
    abnormal_crypt_criteria = list(bundle["runtime"]["observe"].get("abnormal_crypt_criteria", []))
    dysplasia_criteria = list(bundle["runtime"]["observe"].get("dysplasia_criteria", []))
    lower_text = text.lower()
    serrated_hits = _blank_hits(serrated_criteria)
    abnormal_crypt_hits = _blank_hits(abnormal_crypt_criteria)
    dysplasia_hits = _blank_hits(dysplasia_criteria)

    keyword_map = {
        "serrated_surface_pattern": ["serrated", "serration"],
        "mucus_rich_surface": ["mucus", "mucin"],
        "serrated_lesion_context": ["serrated lesion", "serrated polyp", "mucosal lesion"],
        "basal_dilatation": ["dilat", "dilated", "dilation"],
        "crypt_branching": ["branch", "branched"],
        "horizontal_growth": ["horizontal"],
        "boot_l_t_shaped_crypt": ["boot", "l-shaped", "t-shaped"],
        "serration_to_base": ["serration", "to the base"],
        "mucus_cap": ["mucus cap", "mucus", "mucin"],
        "abnormal_maturation": ["maturation", "abnormal maturation", "dysmaturation"],
        "nuclear_enlargement_stratification": ["nuclear enlargement", "stratification"],
        "hyperchromasia": ["hyperchrom", "hyperchromasia"],
        "mitotic_activity_atypia": ["mitotic", "atypia", "atypical"],
        "architectural_crowding": ["crowding"],
    }
    if review_goal == "serrated_lesion_assessment":
        for criterion in serrated_hits:
            serrated_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
    elif review_goal == "abnormal_crypt_assessment":
        for criterion in abnormal_crypt_hits:
            abnormal_crypt_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )
    else:
        for criterion in dysplasia_hits:
            dysplasia_hits[criterion] = (
                "supporting" if any(word in lower_text for word in keyword_map.get(criterion, [])) else "uncertain"
            )

    if review_goal == "serrated_lesion_assessment":
        stage_decision = "supports_serrated_lesion"
        next_step = "Proceed to abnormal crypt review if the lesion remains within the serrated pathway."
    elif review_goal == "abnormal_crypt_assessment":
        stage_decision = "supports_abnormal_crypt"
        next_step = "Proceed to dysplasia review only if abnormal crypt support is convincing."
    else:
        stage_decision = "dysplasia_supported"
        next_step = "Integrate the layered impression and finalize the report."

    return {
        "observation": text.splitlines()[-1][:320],
        "reasoning": "Local Patho-R1 textual evidence was used to summarize the requested diagnostic layer.",
        "next_step": next_step,
        "level_1_findings": _supporting_findings_from_hits(serrated_hits),
        "level_2_findings": _supporting_findings_from_hits(abnormal_crypt_hits),
        "level_3_findings": _supporting_findings_from_hits(dysplasia_hits),
        "stage_decision": stage_decision,
        "serrated_hits": serrated_hits,
        "abnormal_crypt_hits": abnormal_crypt_hits,
        "dysplasia_hits": dysplasia_hits,
        "confidence": 0.62,
    }


def _supporting_findings_from_hits(hits):
    return [criterion for criterion, status in hits.items() if status == "supporting"]


def _aggregate_hits(records, hits_key, criteria):
    checklist = {criterion: {"status": "not_assessed", "evidence_steps": []} for criterion in criteria}
    for record in records:
        hits = record.get("metadata", {}).get(hits_key, {})
        for criterion, status in hits.items():
            if criterion not in checklist or status == "not_assessed":
                continue
            current = checklist[criterion]["status"]
            if status == "supporting":
                checklist[criterion]["status"] = "supporting"
            elif status == "opposing" and current == "not_assessed":
                checklist[criterion]["status"] = "opposing"
            elif status == "uncertain" and current == "not_assessed":
                checklist[criterion]["status"] = "uncertain"
            elif current != "supporting":
                checklist[criterion]["status"] = status
            checklist[criterion]["evidence_steps"].append(record["step_id"])
    return checklist


def _serrated_assessment(trace_clusters, checklist):
    support_count = len([value for value in checklist.values() if value["status"] == "supporting"])
    oppose_count = len([value for value in checklist.values() if value["status"] == "opposing"])
    trace_support = any(cluster.get("l") == "serrated_suspicious_mucosa" for cluster in trace_clusters)
    positive = trace_support or support_count >= 2
    score = min(0.95, max(0.05, 0.20 + 0.16 * support_count + 0.10 * int(trace_support) - 0.08 * oppose_count))
    return {
        "label": "serrated_lesion" if positive else "non_serrated_lesion",
        "positive": positive,
        "score": round(score, 4),
    }


def _abnormal_crypt_assessment(serrated_assessment, checklist):
    support_count = len([value for value in checklist.values() if value["status"] == "supporting"])
    uncertain_count = len([value for value in checklist.values() if value["status"] == "uncertain"])
    structural_support = len(
        [
            key
            for key, value in checklist.items()
            if key in ("basal_dilatation", "crypt_branching", "horizontal_growth", "boot_l_t_shaped_crypt")
            and value["status"] == "supporting"
        ]
    )
    positive = serrated_assessment["positive"] and support_count >= 3 and structural_support >= 1
    if not serrated_assessment["positive"]:
        label = "not_applicable_non_serrated"
    elif positive:
        label = "abnormal_crypt_supported"
    elif support_count == 0 and uncertain_count == 0:
        label = "serrated_but_no_abnormal_crypt"
    elif support_count == 0 and uncertain_count > 0:
        label = "indeterminate_abnormal_crypt"
    else:
        label = "serrated_but_no_abnormal_crypt"
    score = min(0.95, max(0.05, 0.18 + 0.10 * support_count + 0.08 * structural_support))
    return {
        "label": label,
        "positive": positive,
        "score": round(score, 4),
    }


def _dysplasia_assessment(abnormal_crypt_assessment, checklist):
    if not abnormal_crypt_assessment["positive"]:
        return {
            "label": "not_entered_due_to_crypt_gate",
            "positive": False,
            "score": 0.0,
        }
    support_count = len([value for value in checklist.values() if value["status"] == "supporting"])
    assessed_count = len([value for value in checklist.values() if value["status"] != "not_assessed"])
    positive = support_count >= 2
    if positive:
        label = "dysplasia_supported"
    elif assessed_count == 0:
        label = "dysplasia_indeterminate"
    else:
        label = "dysplasia_not_supported"
    score = min(0.95, max(0.05, 0.15 + 0.12 * support_count))
    return {
        "label": label,
        "positive": positive,
        "score": round(score, 4),
    }


def _integrated_impression(serrated_assessment, abnormal_crypt_assessment, dysplasia_assessment):
    if not serrated_assessment["positive"]:
        return "Overall impression favors a non-serrated lesion."
    if abnormal_crypt_assessment["positive"] and dysplasia_assessment["positive"]:
        return "Overall impression favors a serrated lesion with abnormal crypt architecture and dysplasia / atypia."
    if abnormal_crypt_assessment["positive"] and not dysplasia_assessment["positive"]:
        return "Overall impression favors a serrated lesion with abnormal crypt architecture without convincing dysplasia."
    if abnormal_crypt_assessment["label"] == "serrated_but_no_abnormal_crypt":
        return "Overall impression favors a serrated lesion without convincing abnormal crypt architecture."
    return "Overall impression remains within the serrated pathway but abnormal crypt architecture is indeterminate."


def _render_supporting_lines(checklist):
    findings = []
    for criterion, payload in checklist.items():
        if payload["status"] == "supporting":
            findings.append("{0} ({1})".format(criterion, ", ".join(payload["evidence_steps"])))
    return "; ".join(findings) if findings else "No decisive supporting item recorded."
