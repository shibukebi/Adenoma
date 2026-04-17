from pathlib import Path

from adenoma_agent.schemas import NavigationStep
from adenoma_agent.utils import write_json


class NavigateAgent(object):
    def __init__(self, bundle, backend_chain):
        self.bundle = bundle
        self.backend_chain = backend_chain

    def run(self, case_spec, trace_result, case_dir, logger, interventions=None):
        slide_dims = trace_result["payload"]["thumbnail_meta"]["slide_dimensions_level0"]
        backend_response = self.backend_chain.invoke(
            "navigate",
            self.bundle["runtime"]["navigate"]["backend_chain"],
            {
                "images": [str(trace_result["selection"]["paths"]["thumbnail"])],
                "prompt": {
                    "question": case_spec.question,
                    "task": "ssa_navigation_planning",
                },
                "metadata": {
                    "case_id": case_spec.case_id,
                    "slide_dimensions_level0": slide_dims,
                    "clusters": [cluster.to_dict() for cluster in trace_result["clusters"]],
                    "interventions": interventions.to_dict() if interventions else {},
                },
            },
        )
        steps = []
        for payload in backend_response["output"]["steps"]:
            step = NavigationStep(
                step_id=payload["step_id"],
                x=int(payload["x"]),
                y=int(payload["y"]),
                m=float(payload["m"]),
                o=payload["o"],
                metadata=payload.get("metadata", {}),
            )
            steps.append(step)
            logger.log(
                state="NAVIGATE",
                agent="NavigateAgent",
                input_ref=step.metadata.get("cluster_id"),
                output_ref=step.step_id,
                payload=step.to_dict(),
            )
        navigation_json = write_json(
            Path(case_dir) / "navigation" / "navigation_steps.json",
            {"steps": [step.to_dict() for step in steps], "backend_attempts": backend_response["attempts"]},
        )
        return {
            "steps": steps,
            "navigation_json": navigation_json,
            "backend_attempts": backend_response["attempts"],
            "backend": backend_response["backend"],
        }
