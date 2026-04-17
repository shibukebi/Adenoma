import tempfile
import unittest
from pathlib import Path

from adenoma_agent.eval import evaluate_run
from adenoma_agent.replay import build_replay_report
from adenoma_agent.utils import append_jsonl, write_json


class ReplayEvalTest(unittest.TestCase):
    def test_replay_and_eval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            case_dir = run_dir / "case_001"
            case_dir.mkdir(parents=True, exist_ok=True)
            append_jsonl(
                case_dir / "events.jsonl",
                {
                    "timestamp": "2026-04-15T00:00:00",
                    "case_id": "case_001",
                    "state": "TRACE",
                    "agent": "TraceAgent",
                    "input_ref": "a",
                    "output_ref": "b",
                    "latency_ms": 1,
                    "status": "ok",
                    "payload": {},
                },
            )
            write_json(
                case_dir / "case_result.json",
                {
                    "status": "warn",
                    "binary_target": 1,
                    "final_binary_prediction": {"label": "SSA", "positive": True, "score": 0.8},
                    "report_checklist": {"mucus_cap": {"status": "supporting", "evidence_steps": ["step_00"]}},
                    "trace_clusters": [{"cluster_id": "cluster_00"}],
                    "trajectory": [{"step_id": "step_00"}],
                    "audit": {
                        "warnings": ["w"],
                        "errors": [],
                        "metrics": {"trajectory_length": 3, "report_checklist_completeness": 1.0},
                    },
                    "timing": {"total_runtime_ms": 100},
                    "cost": {"estimated_case_cost_units": 4},
                },
            )

            report = build_replay_report(case_dir)
            self.assertIn("Replay for case_001", report)
            summary = evaluate_run(run_dir)
            self.assertEqual(summary["case_count"], 1)
            self.assertEqual(summary["warn_cases"], 1)
            self.assertEqual(summary["ssa_binary_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
