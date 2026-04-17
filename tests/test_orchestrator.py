import tempfile
import unittest
from pathlib import Path

from adenoma_agent.schemas import CaseSpec, InterventionEvent
from adenoma_agent.utils import read_jsonl, write_json


class _FakeTraceAgent(object):
    def run(self, case_spec, case_dir, logger, interventions=None, preferred_mode=None):
        trace_dir = Path(case_dir) / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        boxes_json = trace_dir / "{0}_route_c_boxes.json".format(case_spec.case_id)
        payload = {
            "mode": "heuristic",
            "thumbnail_meta": {
                "slide_dimensions_level0": [4000, 2000],
                "thumbnail_size": [1000, 500],
            },
            "boxes": [{"x1": 100, "y1": 50, "x2": 300, "y2": 250, "score": 0.9, "label": "roi"}],
        }
        write_json(boxes_json, payload)
        from adenoma_agent.schemas import TraceCluster

        candidate = TraceCluster(
            cluster_id="cluster_00",
            cluster_bbox_thumb={"x1": 100, "y1": 50, "x2": 300, "y2": 250},
            cluster_bbox_level0={"x1": 400, "y1": 200, "x2": 1200, "y2": 1000},
            regions_thumb=[{"x1": 100, "y1": 50, "x2": 300, "y2": 250}],
            regions_level0=[{"x1": 400, "y1": 200, "x2": 1200, "y2": 1000}],
            l="ssa_suspicious_mucosa",
            s=4,
            d=True,
            desc="roi",
            evidence=["crypt-risk prefilter"],
            metadata={"area_fraction": 0.1},
        )
        logger.log("TRACE", "FakeTraceAgent", output_ref=str(boxes_json), payload={"candidate_count": 1})
        return {
            "selection": {
                "paths": {
                    "thumbnail": trace_dir / "{0}_thumbnail.jpg".format(case_spec.case_id),
                    "boxes_json": boxes_json,
                    "visualization": trace_dir / "{0}_route_c_boxes.png".format(case_spec.case_id),
                },
                "attempts": [],
            },
            "payload": payload,
            "clusters": [candidate],
            "trace_clusters_json": trace_dir / "trace_clusters.json",
            "trace_backend_json": trace_dir / "trace_backend_attempts.json",
            "trace_dir": trace_dir,
        }


class _FakeNavigateAgent(object):
    def run(self, case_spec, trace_result, case_dir, logger, interventions=None):
        from adenoma_agent.schemas import NavigationStep

        steps = [
            NavigationStep(
                step_id="step_00",
                x=800,
                y=600,
                m=1.0,
                o="Survey mucosal architecture.",
                metadata={"region_size_level0": 1024, "cluster_id": "cluster_00", "action": "inspect"},
            ),
            NavigationStep(
                step_id="step_01",
                x=800,
                y=600,
                m=1.0,
                o="Stop navigation.",
                metadata={"region_size_level0": 1024, "action": "stop"},
            ),
        ]
        logger.log("NAVIGATE", "FakeNavigateAgent", output_ref="step_00", payload=steps[0].to_dict())
        return {
            "steps": steps,
            "navigation_json": Path(case_dir) / "navigation" / "navigation_steps.json",
            "backend_attempts": [],
            "backend": "heuristic",
        }


class _FakeObserveAgent(object):
    def run(self, case_spec, trace_result, navigation_result, case_dir, logger):
        from adenoma_agent.schemas import ObservationRecord, ReasoningState

        record = ObservationRecord(
            step_id="step_00",
            crop_path=str(Path(case_dir) / "observe" / "step_00.png"),
            observation="Tissue-rich patch.",
            reasoning="Supports SSA-oriented review.",
            next_step="Consolidate the checklist.",
            criteria_hits={"mucus_cap": "supporting"},
            confidence=0.8,
            metadata={"background_fraction": 0.1},
        )
        reasoning = ReasoningState(
            hypotheses=["ssa_vs_others_pathological_report_ready"],
            supporting_evidence=[record.reasoning],
            conflicts=[],
            stop_reason="trajectory_complete",
            metadata={},
        )
        logger.log("OBSERVE", "FakeObserveAgent", output_ref="step_00", payload=record.to_dict())
        return {
            "records": [record],
            "reasoning_state": reasoning,
            "report": "Pathological Report\n\nFinal SSA vs others judgement: SSA",
            "report_checklist": {"mucus_cap": {"status": "supporting", "evidence_steps": ["step_00"]}},
            "final_binary_prediction": {"label": "SSA", "positive": True, "score": 0.8},
            "report_json": Path(case_dir) / "observe" / "pathological_report.json",
        }


class _FakeAuditAgent(object):
    def run(self, case_spec, trace_result, navigation_result, observe_result, segmentation_artifact, timings):
        from adenoma_agent.schemas import CaseResult

        return CaseResult(
            case_id=case_spec.case_id,
            binary_target=case_spec.binary_target,
            final_binary_prediction=observe_result["final_binary_prediction"],
            report_checklist=observe_result["report_checklist"],
            pathological_report=observe_result["report"],
            segmentation_artifact=segmentation_artifact,
            trace_clusters=[cluster.to_dict() for cluster in trace_result["clusters"]],
            trajectory=[step.to_dict() for step in navigation_result["steps"]],
            evidence_chain=[record.to_dict() for record in observe_result["records"]],
            cost={"estimated_case_cost_units": 2},
            timing=timings,
            audit={
                "status": "ok",
                "warnings": [],
                "errors": [],
                "metrics": {"trajectory_length": 1, "report_checklist_completeness": 1.0},
            },
            label=case_spec.label,
            status="ok",
            metadata={},
        )


class _FakeCoordsAdapter(object):
    def boxes_to_h5(self, boxes_json, output_h5, patch_size=256, step_size=256, patch_level=0):
        Path(output_h5).parent.mkdir(parents=True, exist_ok=True)
        Path(output_h5).write_text("fake-h5", encoding="utf-8")
        return {"returncode": 0, "stdout": "ok", "stderr": "", "latency_ms": 1}


class _FakePatchExportAdapter(object):
    def export_samples(self, slide_path, coords_h5, output_dir, max_patches=16):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = output_dir / "patch_manifest.csv"
        manifest.write_text("slide_id\nfake\n", encoding="utf-8")
        return {"result": {"returncode": 0, "stdout": "ok", "stderr": "", "latency_ms": 1}, "manifest": manifest}


class OrchestratorIntegrationTest(unittest.TestCase):
    def test_orchestrator_writes_case_result_and_log(self):
        from adenoma_agent.orchestrator import AdenomaAgentOrchestrator

        bundle = {
            "runtime": {
                "project": {
                    "project_root": "/tmp/adenoma_agent_test",
                    "artifacts_root": "/tmp/adenoma_agent_test/artifacts",
                },
                "cache": {
                    "thumbnail_cache_root": "/tmp/adenoma_agent_test/cache/thumb",
                    "trace_cache_root": "/tmp/adenoma_agent_test/cache/trace",
                    "description_cache_root": "/tmp/adenoma_agent_test/cache/desc",
                },
                "data": {"binary_positive_label": "Sessile serrated adenoma"},
            },
            "budget": {"max_retries_per_stage": 0},
        }
        orchestrator = AdenomaAgentOrchestrator(bundle)
        orchestrator.trace_agent = _FakeTraceAgent()
        orchestrator.navigate_agent = _FakeNavigateAgent()
        orchestrator.observe_agent = _FakeObserveAgent()
        orchestrator.audit_agent = _FakeAuditAgent()
        orchestrator.coords_adapter = _FakeCoordsAdapter()
        orchestrator.patch_export_adapter = _FakePatchExportAdapter()

        case_spec = CaseSpec(
            case_id="case_001",
            slide_path="/tmp/case_001.svs",
            task_type="ssa_vs_others_huge_region_agent",
            question="Review for SSA vs others.",
            label="Hyperplastic polyps",
            binary_target=0,
            metadata={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = orchestrator.run_case(case_spec, tmpdir, interventions=InterventionEvent())
            case_dir = Path(result["case_dir"])
            self.assertTrue((case_dir / "case_result.json").exists())
            self.assertTrue((case_dir / "events.jsonl").exists())
            rows = read_jsonl(case_dir / "events.jsonl")
            self.assertGreaterEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
