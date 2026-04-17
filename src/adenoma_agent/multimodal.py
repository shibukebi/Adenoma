import tempfile
from pathlib import Path

from adenoma_agent.utils import read_json, run_command, write_json


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
    supported_stages = ("observe_step",)

    def invoke(self, request, bundle):
        config = bundle["runtime"].get("backends", {}).get("local_patho_r1", {})
        if not config.get("enabled"):
            raise BackendUnavailableError("local_patho_r1 backend is disabled")
        if request["stage"] != "observe_step":
            raise BackendUnavailableError("local_patho_r1 only supports observe_step")
        image_path = request["images"][0]
        question = request["prompt"]["question"]
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
        result = run_command(command, timeout=300)
        if result["returncode"] != 0:
            raise BackendExecutionError(result["stderr"] or result["stdout"])
        text = (result["stdout"] or "").strip()
        if not text:
            raise BackendExecutionError("Empty Patho-R1 response")
        lower_text = text.lower()
        support_words = {
            "basal_dilatation": ["dilat", "dilated", "dilation"],
            "crypt_branching": ["branch", "branched"],
            "horizontal_growth": ["horizontal", "l-shaped", "t-shaped", "boot"],
            "boot_l_t_shaped_crypt": ["boot", "l-shaped", "t-shaped"],
            "serration_to_base": ["serration", "serrated", "to the base"],
            "mucus_cap": ["mucus", "mucin"],
            "abnormal_maturation": ["maturation", "dysmaturation", "abnormal maturation"],
        }
        criteria_hits = {}
        for key, words in support_words.items():
            criteria_hits[key] = "supporting" if any(word in lower_text for word in words) else "uncertain"
        output = {
            "observation": text.splitlines()[-1][:320],
            "reasoning": "Local Patho-R1 textual evidence was used to summarize the targeted crop.",
            "next_step": "Integrate this crop with adjacent views and targeted SSA checklist review.",
            "criteria_hits": criteria_hits,
            "confidence": 0.62,
            "raw_text": text,
        }
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
        clusters = []
        criteria = list(bundle["runtime"]["observe"].get("criteria", []))
        for proposal in request["metadata"]["proposals"]:
            tissue_fraction = float(proposal["metadata"].get("tissue_fraction", 0.0))
            pale_fraction = float(proposal["metadata"].get("pale_fraction", 0.0))
            artifact_fraction = float(proposal["metadata"].get("artifact_fraction", 0.0))
            route_c_overlap = float(proposal["metadata"].get("route_c_hint_overlap", 0.0))
            area_fraction = float(proposal["metadata"].get("area_fraction", 0.0))

            if tissue_fraction < 0.08:
                label = "background"
                priority = 0
                need_high_mag = False
                reasons = ["low tissue fraction on overview screening"]
            elif artifact_fraction > 0.28:
                label = "artifact"
                priority = 0
                need_high_mag = False
                reasons = ["high artifact-like color fraction"]
            else:
                risk_score = 1
                reasons = ["mucosal tissue retained after overview filtering"]
                if pale_fraction > 0.18:
                    risk_score += 1
                    reasons.append("surface pallor / mucus-like appearance warrants SSA review")
                if route_c_overlap > 0.05:
                    risk_score += 1
                    reasons.append("overlaps a route-C screening hint")
                if 0.02 <= area_fraction <= 0.35:
                    risk_score += 1
                    reasons.append("region size is suitable for focused crypt review")
                label = "ssa_suspicious_mucosa" if risk_score >= 3 else "non_ssa_mucosa"
                priority = min(5, max(1, int(risk_score)))
                need_high_mag = priority >= 3 or pale_fraction > 0.12
                if need_high_mag:
                    reasons.append("high magnification is required to inspect crypt architecture")

            clusters.append(
                {
                    "cluster_id": proposal["cluster_id"],
                    "l": label,
                    "s": priority,
                    "d": need_high_mag,
                    "desc": "; ".join(reasons),
                    "evidence": reasons,
                    "metadata": {
                        **proposal["metadata"],
                        "criteria_focus": criteria,
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
            schedule = [1.0]
            if cluster["d"]:
                schedule.extend([2.5, 5.0])
            elif int(cluster["s"]) >= 2:
                schedule.append(2.5)

            for magnification in schedule:
                if step_index >= max_steps:
                    break
                region_size = int(mag_to_region.get(str(float(magnification)), 256))
                if float(magnification) == 1.0:
                    need_to_see = (
                        "Survey the mucosal region and look for serrated surface contour, mucus cap, "
                        "and crypt-rich hotspots that deserve closer review."
                    )
                elif float(magnification) < 5.0:
                    need_to_see = (
                        "Check serration extension, crypt distribution, and suspicious basal contour before "
                        "committing to high-magnification crypt review."
                    )
                else:
                    need_to_see = (
                        "Inspect crypt base for basal dilatation, branching, horizontal growth, and "
                        "boot/L/T-shaped crypts."
                    )
                step_bbox = {
                    "x1": center_x - region_size // 2,
                    "y1": center_y - region_size // 2,
                    "x2": center_x + region_size // 2,
                    "y2": center_y + region_size // 2,
                }
                should_skip = False
                for prior_item in prior_windows:
                    from adenoma_agent.utils import bbox_overlap_ratio, clamp_center_point, normalized_point

                    if abs(float(prior_item["m"]) - float(magnification)) > 1e-6:
                        continue
                    if bbox_overlap_ratio(step_bbox, prior_item["bbox"]) > overlap_threshold:
                        should_skip = True
                        break
                if should_skip:
                    continue
                from adenoma_agent.utils import clamp_center_point, normalized_point

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
                        "metadata": {
                            "cluster_id": cluster["cluster_id"],
                            "cluster_label": cluster["l"],
                            "cluster_priority": cluster["s"],
                            "region_size_level0": region_size,
                            "normalized_center": normalized_point(fixed_x, fixed_y, slide_dims),
                            "criteria_focus": cluster["metadata"].get("criteria_focus", []),
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
                    "o": "Stop navigation because no reviewable mucosal cluster was retained.",
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
                    "o": "Stop navigation and consolidate the collected multi-scale evidence.",
                    "metadata": {"action": "stop", "region_size_level0": 256},
                }
            )
        return {"steps": steps}

    def _observe_step_output(self, request, bundle):
        stats = request["metadata"]["image_stats"]
        step = request["metadata"]["step"]
        cluster = request["metadata"].get("cluster", {})
        criteria = bundle["runtime"]["observe"].get("criteria", [])
        background_fraction = float(stats.get("background_fraction", 0.0))
        pale_fraction = float(stats.get("pale_fraction", 0.0))
        tissue_fraction = float(stats.get("tissue_fraction", 0.0))
        cluster_priority = int(cluster.get("s", 0))

        criteria_hits = {criterion: "not_assessed" for criterion in criteria}
        if float(step["m"]) >= 2.5:
            for criterion in ("serration_to_base", "mucus_cap", "abnormal_maturation"):
                if criterion in criteria_hits:
                    criteria_hits[criterion] = "supporting" if pale_fraction > 0.18 or cluster_priority >= 3 else "uncertain"
        if float(step["m"]) >= 5.0:
            for criterion in (
                "basal_dilatation",
                "crypt_branching",
                "horizontal_growth",
                "boot_l_t_shaped_crypt",
            ):
                if criterion in criteria_hits:
                    if cluster.get("l") == "ssa_suspicious_mucosa" and tissue_fraction > 0.45:
                        criteria_hits[criterion] = "supporting" if cluster_priority >= 4 else "uncertain"
                    else:
                        criteria_hits[criterion] = "uncertain"

        if background_fraction > 0.7:
            observation = "The crop is background-heavy and provides limited direct crypt detail."
        elif float(step["m"]) >= 5.0:
            observation = "High-magnification review targets crypt architecture in a tissue-rich mucosal crop."
        elif float(step["m"]) >= 2.5:
            observation = "Intermediate magnification preserves mucosal detail for surface serration and mucus-cap review."
        else:
            observation = "Low magnification captures the overall mucosal layout and regional context."

        supporting_count = len([value for value in criteria_hits.values() if value == "supporting"])
        if supporting_count >= 2:
            reasoning = "This view adds moderate support to SSA-oriented review because multiple checklist criteria are favored."
        elif supporting_count == 1:
            reasoning = "This view adds weak SSA-oriented support but still needs corroborating high-magnification evidence."
        else:
            reasoning = "This view remains indeterminate and should be interpreted with the full multi-scale trajectory."

        if float(step["m"]) < 5.0 and cluster.get("d"):
            next_step = "Move to a higher magnification and inspect the crypt base directly."
        else:
            next_step = "Integrate this evidence with the remaining trajectory and finalize the checklist."

        confidence = min(
            0.95,
            max(0.05, 0.25 + 0.20 * tissue_fraction + 0.15 * pale_fraction + 0.10 * supporting_count + 0.05 * cluster_priority),
        )
        return {
            "observation": observation,
            "reasoning": reasoning,
            "next_step": next_step,
            "criteria_hits": criteria_hits,
            "confidence": round(confidence, 4),
        }

    def _observe_report_output(self, request, bundle):
        criteria = bundle["runtime"]["observe"].get("criteria", [])
        records = request["metadata"]["records"]
        checklist = {criterion: {"status": "not_assessed", "evidence_steps": []} for criterion in criteria}
        for record in records:
            for criterion, status in record["criteria_hits"].items():
                if status == "not_assessed":
                    continue
                current = checklist[criterion]["status"]
                if status == "supporting" or current == "not_assessed":
                    checklist[criterion]["status"] = status
                elif current != "supporting":
                    checklist[criterion]["status"] = status
                checklist[criterion]["evidence_steps"].append(record["step_id"])

        supporting_findings = []
        opposing_findings = []
        uncertain_findings = []
        supporting_count = 0
        structural_support = 0
        for criterion, payload in checklist.items():
            if payload["status"] == "supporting":
                supporting_count += 1
                if criterion in (
                    "basal_dilatation",
                    "crypt_branching",
                    "horizontal_growth",
                    "boot_l_t_shaped_crypt",
                ):
                    structural_support += 1
                supporting_findings.append("{0}: supported by {1}".format(criterion, ", ".join(payload["evidence_steps"])))
            elif payload["status"] == "opposing":
                opposing_findings.append("{0}: opposed by {1}".format(criterion, ", ".join(payload["evidence_steps"])))
            elif payload["status"] == "uncertain":
                uncertain_findings.append("{0}: uncertain after {1}".format(criterion, ", ".join(payload["evidence_steps"])))

        positive = supporting_count >= 2 and structural_support >= 1
        score = min(0.95, max(0.05, 0.18 + 0.10 * supporting_count + 0.08 * structural_support))
        final_prediction = {
            "label": "SSA" if positive else "others",
            "positive": positive,
            "score": round(score, 4),
        }
        lines = []
        lines.append("Pathological Report")
        lines.append("Task: SSA vs others")
        lines.append("")
        lines.append("Supporting findings:")
        for item in supporting_findings or ["None confirmed at a decisive level."]:
            lines.append("- " + item)
        lines.append("")
        lines.append("Opposing findings:")
        for item in opposing_findings or ["No strong opposing finding was documented."]:
            lines.append("- " + item)
        lines.append("")
        lines.append("Uncertain findings:")
        for item in uncertain_findings or ["No uncertain checklist item was recorded."]:
            lines.append("- " + item)
        lines.append("")
        lines.append(
            "Final SSA vs others judgement: {0} (score={1:.4f})".format(
                final_prediction["label"],
                final_prediction["score"],
            )
        )
        return {
            "report_checklist": checklist,
            "pathological_report": "\n".join(lines),
            "final_binary_prediction": final_prediction,
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
