from pathlib import Path

import numpy as np
from PIL import Image

from adenoma_agent.cache import JsonCache
from adenoma_agent.schemas import ObservationRecord, ReasoningState
from adenoma_agent.utils import write_json


class ObserveReasonAgent(object):
    def __init__(self, bundle, cropper_adapter, backend_chain):
        self.bundle = bundle
        self.cropper_adapter = cropper_adapter
        self.backend_chain = backend_chain
        self.description_cache = JsonCache(bundle["runtime"]["cache"]["description_cache_root"])

    def run(self, case_spec, trace_result, navigation_result, case_dir, logger):
        observe_dir = Path(case_dir) / "observe"
        review_steps = [step for step in navigation_result["steps"] if step.metadata.get("action") != "stop"]
        export_result = self.cropper_adapter.export_crops(case_spec, review_steps, observe_dir)

        step_lookup = {step.step_id: step for step in review_steps}
        cluster_lookup = {cluster.cluster_id: cluster for cluster in trace_result["clusters"]}

        records = []
        support_summaries = []
        conflicts = []
        for crop in export_result["manifest"].get("crops", []):
            step = step_lookup[crop["step_id"]]
            cluster = cluster_lookup.get(step.metadata.get("cluster_id"))
            image_stats = self._image_stats(crop["image_path"])
            cache_payload = {
                "case_id": case_spec.case_id,
                "step_id": step.step_id,
                "m": step.m,
                "cluster_id": step.metadata.get("cluster_id"),
                "image_stats": image_stats,
                "backend_chain": self.bundle["runtime"]["observe"]["backend_chain"],
            }
            cached = self.description_cache.load("observe_step", cache_payload, "record.json")
            if cached is None:
                backend_response = self.backend_chain.invoke(
                    "observe_step",
                    self.bundle["runtime"]["observe"]["backend_chain"],
                    {
                        "images": [crop["image_path"]],
                        "prompt": {
                            "question": self.bundle["runtime"]["observe"]["patho_r1_question"],
                            "task": "ssa_observe_reason_step",
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
                criteria_hits=payload["criteria_hits"],
                confidence=float(payload["confidence"]),
                metadata={
                    "backend": payload.get("backend"),
                    "background_fraction": image_stats["background_fraction"],
                    "tissue_fraction": image_stats["tissue_fraction"],
                    "pale_fraction": image_stats["pale_fraction"],
                    "cluster_id": step.metadata.get("cluster_id"),
                    "magnification": step.m,
                    "need_to_see": step.o,
                },
            )
            records.append(record)
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

        report_response = self.backend_chain.invoke(
            "observe_report",
            self.bundle["runtime"]["observe"]["backend_chain"],
            {
                "images": [],
                "prompt": {
                    "question": "Synthesize the SSA checklist and produce a final pathological report.",
                    "task": "ssa_observe_reason_report",
                },
                "metadata": {
                    "case_id": case_spec.case_id,
                    "records": [record.to_dict() for record in records],
                    "trace_clusters": [cluster.to_dict() for cluster in trace_result["clusters"]],
                },
            },
        )
        reasoning_state = ReasoningState(
            hypotheses=["ssa_vs_others_pathological_report_ready"],
            supporting_evidence=support_summaries[:8],
            conflicts=conflicts,
            stop_reason="trajectory_completed",
            metadata={"observation_count": len(records)},
        )
        observation_json = write_json(
            observe_dir / "observation_records.json",
            {"observations": [record.to_dict() for record in records]},
        )
        report_json = write_json(
            observe_dir / "pathological_report.json",
            {
                "pathological_report": report_response["output"]["pathological_report"],
                "report_checklist": report_response["output"]["report_checklist"],
                "final_binary_prediction": report_response["output"]["final_binary_prediction"],
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
            "crop_manifest_json": export_result["manifest_json"],
            "crop_result": export_result["result"],
            "report": report_response["output"]["pathological_report"],
            "report_checklist": report_response["output"]["report_checklist"],
            "final_binary_prediction": report_response["output"]["final_binary_prediction"],
            "report_backend_attempts": report_response["attempts"],
        }

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
