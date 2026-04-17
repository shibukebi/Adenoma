import time
from pathlib import Path

from adenoma_agent.agents.audit import AuditAgent
from adenoma_agent.agents.navigate import NavigateAgent
from adenoma_agent.agents.observe_reason import ObserveReasonAgent
from adenoma_agent.agents.trace import TraceAgent
from adenoma_agent.adapters.clam_coords import ClamCoordsAdapter
from adenoma_agent.adapters.cropper import NavigationCropperAdapter
from adenoma_agent.adapters.patch_export import PatchExportAdapter
from adenoma_agent.adapters.route_c import RouteCSelectorAdapter
from adenoma_agent.logger import JsonlLogger
from adenoma_agent.multimodal import StageBackendChain
from adenoma_agent.schemas import InterventionEvent
from adenoma_agent.utils import ensure_dir, write_json


class AdenomaAgentOrchestrator(object):
    def __init__(self, bundle):
        self.bundle = bundle
        selector = RouteCSelectorAdapter(bundle)
        cropper = NavigationCropperAdapter(bundle)
        backend_chain = StageBackendChain(bundle)
        self.trace_agent = TraceAgent(bundle, selector, backend_chain)
        self.navigate_agent = NavigateAgent(bundle, backend_chain)
        self.observe_agent = ObserveReasonAgent(bundle, cropper, backend_chain)
        self.audit_agent = AuditAgent(bundle)
        self.coords_adapter = ClamCoordsAdapter(bundle)
        self.patch_export_adapter = PatchExportAdapter(bundle)

    def run_case(self, case_spec, output_root, interventions=None, trace_mode=None):
        output_root = ensure_dir(output_root)
        case_dir = ensure_dir(Path(output_root) / case_spec.case_id)
        logger = JsonlLogger(case_spec.case_id, case_dir / "events.jsonl")
        interventions = interventions or InterventionEvent()
        write_json(
            case_dir / "run_metadata.json",
            {
                "case": case_spec.to_dict(),
                "interventions": interventions.to_dict(),
                "runtime_path": self.bundle.get("runtime_path"),
                "budget_path": self.bundle.get("budget_path"),
                "trace_mode": trace_mode,
            },
        )
        logger.log(
            state="START",
            agent="Orchestrator",
            input_ref=case_spec.slide_path,
            output_ref=str(case_dir),
            payload={"interventions": interventions.to_dict()},
        )

        started = time.time()
        timings = {}
        trace_result = self._run_stage(
            "TRACE",
            logger,
            lambda: self.trace_agent.run(case_spec, case_dir, logger, interventions, preferred_mode=trace_mode),
            timings,
        )

        segmentation_dir = ensure_dir(case_dir / "segmentation")
        coords_h5 = segmentation_dir / "{0}.h5".format(case_spec.case_id)
        coords_result = self.coords_adapter.boxes_to_h5(
            trace_result["selection"]["paths"]["boxes_json"],
            coords_h5,
        )
        logger.log(
            state="TRACE_TO_SEGMENT",
            agent="ClamCoordsAdapter",
            input_ref=str(trace_result["selection"]["paths"]["boxes_json"]),
            output_ref=str(coords_h5),
            latency_ms=coords_result["latency_ms"],
            status="ok" if coords_result["returncode"] == 0 else "error",
            payload={"stdout": coords_result["stdout"], "stderr": coords_result["stderr"]},
        )

        patch_export = self.patch_export_adapter.export_samples(
            case_spec.slide_path,
            coords_h5,
            segmentation_dir / "patch_samples",
            max_patches=16,
        )
        logger.log(
            state="PATCH_EXPORT",
            agent="PatchExportAdapter",
            input_ref=str(coords_h5),
            output_ref=str(patch_export["manifest"]),
            latency_ms=patch_export["result"]["latency_ms"],
            status="ok" if patch_export["result"]["returncode"] == 0 else "error",
            payload={"stdout": patch_export["result"]["stdout"], "stderr": patch_export["result"]["stderr"]},
        )

        navigation_result = self._run_stage(
            "NAVIGATE",
            logger,
            lambda: self.navigate_agent.run(case_spec, trace_result, case_dir, logger, interventions),
            timings,
        )
        observe_result = self._run_stage(
            "OBSERVE",
            logger,
            lambda: self.observe_agent.run(case_spec, trace_result, navigation_result, case_dir, logger),
            timings,
        )

        total_runtime_ms = int(round((time.time() - started) * 1000.0))
        timings["total_runtime_ms"] = total_runtime_ms
        segmentation_artifact = {
            "thumbnail_path": str(trace_result["selection"]["paths"]["thumbnail"]),
            "boxes_json": str(trace_result["selection"]["paths"]["boxes_json"]),
            "boxes_visualization": str(trace_result["selection"]["paths"]["visualization"]),
            "coords_h5": str(coords_h5),
            "patch_manifest": str(patch_export["manifest"]),
            "trace_clusters_json": str(trace_result["trace_clusters_json"]),
            "navigation_json": str(navigation_result["navigation_json"]),
            "report_json": str(observe_result["report_json"]),
            "coords_returncode": coords_result["returncode"],
            "patch_export_returncode": patch_export["result"]["returncode"],
        }
        case_result = self.audit_agent.run(
            case_spec,
            trace_result,
            navigation_result,
            observe_result,
            segmentation_artifact,
            timings,
        )
        case_result_path = write_json(case_dir / "case_result.json", case_result.to_dict())
        logger.log(
            state="END",
            agent="AuditAgent",
            input_ref=str(case_dir),
            output_ref=str(case_result_path),
            latency_ms=0,
            status=case_result.status,
            payload=case_result.audit,
        )
        return {
            "case_dir": case_dir,
            "case_result": case_result,
            "case_result_path": case_result_path,
        }

    def _run_stage(self, stage_name, logger, func, timings):
        retries = int(self.bundle["budget"].get("max_retries_per_stage", 1))
        stage_started = time.time()
        last_error = None
        for attempt in range(retries + 1):
            try:
                result = func()
                timings[stage_name.lower() + "_runtime_ms"] = int(round((time.time() - stage_started) * 1000.0))
                return result
            except Exception as exc:
                last_error = exc
                logger.log(
                    state=stage_name,
                    agent="Orchestrator",
                    status="error",
                    payload={"attempt": attempt + 1, "error": str(exc)},
                )
        raise last_error
