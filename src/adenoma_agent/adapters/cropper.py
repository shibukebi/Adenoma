from pathlib import Path

from adenoma_agent.utils import ensure_dir, write_json, run_command, read_json


class NavigationCropperAdapter(object):
    def __init__(self, bundle):
        self.bundle = bundle

    def export_crops(self, case_spec, steps, output_dir):
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        steps_json = output_dir / "navigation_steps_for_crop.json"
        manifest_json = output_dir / "crop_manifest.json"
        payload = {"case_id": case_spec.case_id, "steps": [step.to_dict() for step in steps]}
        write_json(steps_json, payload)
        command = [
            self.bundle["runtime"]["paths"]["clam_python"],
            self.bundle["runtime"]["paths"]["crop_helper_script"],
            "--slide-path",
            case_spec.slide_path,
            "--steps-json",
            str(steps_json),
            "--output-dir",
            str(output_dir),
            "--manifest-json",
            str(manifest_json),
            "--output-size",
            str(self.bundle["budget"].get("default_output_crop_size", 256)),
        ]
        result = run_command(
            command,
            timeout=self.bundle["budget"].get("stage_timeout_seconds", {}).get("observe", 300),
        )
        manifest = read_json(manifest_json) if manifest_json.exists() else {"crops": []}
        return {
            "result": result,
            "manifest": manifest,
            "steps_json": steps_json,
            "manifest_json": manifest_json,
        }
