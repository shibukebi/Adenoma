from pathlib import Path

import numpy as np
from PIL import Image

from adenoma_agent.cache import JsonCache
from adenoma_agent.schemas import NavigationStep, ObservationRecord, ReasoningState
from adenoma_agent.utils import write_json


class ObserveReasonAgent(object):
    def __init__(self, bundle, cropper_adapter, backend_chain):
        self.bundle = bundle
        self.cropper_adapter = cropper_adapter
        self.backend_chain = backend_chain
        self.description_cache = JsonCache(bundle["runtime"]["cache"]["description_cache_root"])

    def run(self, case_spec, trace_result, navigation_result, case_dir, logger):
        observe_dir = Path(case_dir) / "observe"
        planned_review_steps = [step for step in navigation_result["steps"] if step.metadata.get("action") != "stop"]
        stop_steps = [step for step in navigation_result["steps"] if step.metadata.get("action") == "stop"]
        cluster_lookup = {cluster.cluster_id: cluster for cluster in trace_result["clusters"]}

        records = []
        support_summaries = []
        conflicts = []
        trajectory_steps = []
        crop_manifests = []
        crop_results = []
        pending_steps = list(planned_review_steps)
        dysplasia_added = set()

        step_index = 0
        while step_index < len(pending_steps):
            step = pending_steps[step_index]
            step_index += 1
            export_result = self.cropper_adapter.export_crops(case_spec, [step], observe_dir / step.step_id)
            crop_manifests.append(export_result["manifest"])
            crop_results.append(export_result["result"])
            crops = export_result["manifest"].get("crops", [])
            if not crops:
                continue
            crop = crops[0]
            cluster = cluster_lookup.get(step.metadata.get("cluster_id"))
            image_stats = self._image_stats(crop["image_path"])
            cache_payload = {
                "case_id": case_spec.case_id,
                "agent_version": "hierarchical_v3_abnormal_crypt_gate",
                "step_id": step.step_id,
                "m": step.m,
                "cluster_id": step.metadata.get("cluster_id"),
                "image_stats": image_stats,
                "backend_chain": self.bundle["runtime"]["observe"]["backend_chain"],
            }
            cached = self.description_cache.load("observe_step", cache_payload, "record.json")
            if cached is not None and "level_1_findings" not in cached:
                cached = None
            if cached is None:
                backend_response = self.backend_chain.invoke(
                    "observe_step",
                    self.bundle["runtime"]["observe"]["backend_chain"],
                    {
                        "images": [crop["image_path"]],
                        "prompt": {
                            "question": self.bundle["runtime"]["observe"]["patho_r1_question"],
                            "task": "mucosa_serrated_abnormal_crypt_dysplasia_observe_step",
                        },
                        "metadata": {
                            "case_id": case_spec.case_id,
                            "step": step.to_dict(),
                            "cluster": cluster.to_dict() if cluster else {},
                            "image_stats": image_stats,
                        },
                    },
                )
                step_output = backend_response["output"]
                payload = {
                    "backend": backend_response["backend"],
                    "backend_attempts": backend_response["attempts"],
                    **step_output,
                }
                self.description_cache.save("observe_step", cache_payload, "record.json", payload)
            else:
                payload = cached
            record = ObservationRecord(
                step_id=step.step_id,
                crop_path=crop["image_path"],
                observation=payload["observation"],
                reasoning=payload["reasoning"],
                next_step=payload["next_step"],
                level_1_findings=payload["level_1_findings"],
                level_2_findings=payload["level_2_findings"],
                level_3_findings=payload["level_3_findings"],
                stage_decision=payload["stage_decision"],
                confidence=float(payload["confidence"]),
                metadata={
                    "backend": payload.get("backend"),
                    "background_fraction": image_stats["background_fraction"],
                    "tissue_fraction": image_stats["tissue_fraction"],
                    "pale_fraction": image_stats["pale_fraction"],
                    "cluster_id": step.metadata.get("cluster_id"),
                    "magnification": step.m,
                    "need_to_see": step.o,
                    "review_goal": step.review_goal,
                    "stage_gate": step.stage_gate,
                    "serrated_hits": payload.get("serrated_hits", {}),
                    "abnormal_crypt_hits": payload.get("abnormal_crypt_hits", {}),
                    "dysplasia_hits": payload.get("dysplasia_hits", {}),
                },
            )
            records.append(record)
            trajectory_steps.append(step)
            support_summaries.append(record.reasoning)
            if image_stats["background_fraction"] > 0.7:
                conflicts.append("Background-heavy crop at {0}".format(step.step_id))
            logger.log(
                state="OBSERVE",
                agent="ObserveReasonAgent",
                input_ref=crop["image_path"],
                output_ref=record.step_id,
                payload=record.to_dict(),
            )
            if (
                step.review_goal == "abnormal_crypt_assessment"
                and record.stage_decision == "supports_abnormal_crypt"
                and step.metadata.get("cluster_id") not in dysplasia_added
            ):
                pending_steps.insert(
                    step_index,
                    self._build_dysplasia_step(step, cluster, len(trajectory_steps) + len(stop_steps)),
                )
                dysplasia_added.add(step.metadata.get("cluster_id"))

        trajectory_steps.extend(stop_steps)
        combined_manifest = {"crops": []}
        for manifest in crop_manifests:
            combined_manifest["crops"].extend(manifest.get("crops", []))
        combined_manifest_json = write_json(observe_dir / "crop_manifest.json", combined_manifest)
        combined_result = {
            "returncode": 0 if all(result.get("returncode", 1) == 0 for result in crop_results) else 1,
            "stdout": "\n".join(result.get("stdout", "") for result in crop_results if result.get("stdout")),
            "stderr": "\n".join(result.get("stderr", "") for result in crop_results if result.get("stderr")),
            "latency_ms": sum(int(result.get("latency_ms", 0)) for result in crop_results),
        }

        report_response = self.backend_chain.invoke(
            "observe_report",
            self.bundle["runtime"]["observe"]["backend_chain"],
            {
                "images": [],
                "prompt": {
                    "question": "Synthesize a four-stage pathology report with serrated lesion, abnormal crypt, and gated dysplasia sections.",
                    "task": "mucosa_serrated_abnormal_crypt_dysplasia_report",
                },
                "metadata": {
                    "case_id": case_spec.case_id,
                    "records": [record.to_dict() for record in records],
                    "trace_clusters": [cluster.to_dict() for cluster in trace_result["clusters"]],
                },
            },
        )
        reasoning_state = ReasoningState(
            hypotheses=["mucosa_serrated_abnormal_crypt_dysplasia_report_ready"],
            supporting_evidence=support_summaries[:8],
            conflicts=conflicts,
            stop_reason="trajectory_completed",
            metadata={"observation_count": len(records), "trajectory_length": len(trajectory_steps)},
        )
        observation_json = write_json(
            observe_dir / "observation_records.json",
            {"observations": [record.to_dict() for record in records]},
        )
        report_json = write_json(
            observe_dir / "pathological_report.json",
            {
                "hierarchical_prediction": report_response["output"]["hierarchical_prediction"],
                "serrated_checklist": report_response["output"]["serrated_checklist"],
                "abnormal_crypt_checklist": report_response["output"]["abnormal_crypt_checklist"],
                "dysplasia_checklist": report_response["output"]["dysplasia_checklist"],
                "integrated_report": report_response["output"]["integrated_report"],
                "backend_attempts": report_response["attempts"],
            },
        )
        reasoning_json = write_json(observe_dir / "reasoning_state.json", reasoning_state.to_dict())
        return {
            "records": records,
            "reasoning_state": reasoning_state,
            "observation_json": observation_json,
            "report_json": report_json,
            "reasoning_json": reasoning_json,
            "crop_manifest_json": combined_manifest_json,
            "crop_result": combined_result,
            "trajectory_steps": trajectory_steps,
            "hierarchical_prediction": report_response["output"]["hierarchical_prediction"],
            "serrated_checklist": report_response["output"]["serrated_checklist"],
            "abnormal_crypt_checklist": report_response["output"]["abnormal_crypt_checklist"],
            "dysplasia_checklist": report_response["output"]["dysplasia_checklist"],
            "integrated_report": report_response["output"]["integrated_report"],
            "report_backend_attempts": report_response["attempts"],
        }

    def _build_dysplasia_step(self, parent_step, cluster, step_index):
        cluster_id = parent_step.metadata.get("cluster_id")
        region_size = int(self.bundle["budget"].get("magnification_to_region_size", {}).get("5.0", 256))
        return NavigationStep(
            step_id="step_dyn_{0:02d}_{1}".format(int(step_index), cluster_id or "cluster"),
            x=parent_step.x,
            y=parent_step.y,
            m=5.0,
            o="Perform gated dysplasia review only after abnormal crypt support has been established.",
            review_goal="dysplasia_assessment",
            stage_gate="dysplasia",
            metadata={
                **parent_step.metadata,
                "cluster_id": cluster_id,
                "cluster_label": cluster.l if cluster else parent_step.metadata.get("cluster_label"),
                "cluster_priority": cluster.s if cluster else parent_step.metadata.get("cluster_priority"),
                "region_size_level0": region_size,
                "action": "inspect",
                "generated_by": "ObserveReasonAgent",
                "gate_source_step": parent_step.step_id,
            },
        )

    def _image_stats(self, image_path):
        image = Image.open(image_path).convert("RGB")
        arr = np.array(image, dtype=np.float32)
        mean_rgb = arr.mean(axis=2)
        sat = arr.max(axis=2) - arr.min(axis=2)
        tissue_mask = (mean_rgb < 235.0) & (sat > 8.0)
        pale_mask = (mean_rgb > 165.0) & (mean_rgb < 235.0) & (sat < 28.0) & tissue_mask
        background_fraction = float((mean_rgb > 235.0).mean())
        tissue_fraction = float(tissue_mask.mean())
        pale_fraction = float(pale_mask.mean())
        return {
            "background_fraction": round(background_fraction, 4),
            "tissue_fraction": round(tissue_fraction, 4),
            "pale_fraction": round(pale_fraction, 4),
        }
