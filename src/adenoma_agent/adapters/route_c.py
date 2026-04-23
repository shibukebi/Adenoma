import shutil
from pathlib import Path

from adenoma_agent.utils import (
    env_with_cuda_visible_devices,
    ensure_dir,
    read_json,
    run_command,
    sha1_text,
    write_json,
)


class RouteCSelectorAdapter(object):
    def __init__(self, bundle):
        self.bundle = bundle

    def _copy_outputs(self, src_dir, dst_dir):
        ensure_dir(dst_dir)
        for child in Path(src_dir).iterdir():
            if child.is_file():
                shutil.copy2(str(child), str(Path(dst_dir) / child.name))

    def _expected_paths(self, output_dir, slide_id):
        output_dir = Path(output_dir)
        return {
            "thumbnail": output_dir / "{0}_thumbnail.jpg".format(slide_id),
            "raw_response": output_dir / "{0}_route_c_raw_response.txt".format(slide_id),
            "boxes_json": output_dir / "{0}_route_c_boxes.json".format(slide_id),
            "visualization": output_dir / "{0}_route_c_boxes.png".format(slide_id),
        }

    def select(self, case_spec, output_dir, manual_boxes=None, preferred_mode=None):
        runtime = self.bundle["runtime"]
        trace_cfg = runtime["trace"]
        paths = runtime["paths"]
        cache_root = Path(runtime["cache"]["trace_cache_root"])
        ensure_dir(cache_root)
        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        mode = preferred_mode or trace_cfg.get("preferred_mode", "auto")
        prompt_hash = sha1_text("route_c_selector_v1")
        cache_payload = {
            "slide_id": case_spec.case_id,
            "agent_version": "trace_v1",
            "model_id": runtime["models"]["patho_r1_model_id"],
            "thumbnail_size": trace_cfg["thumbnail_max_size"],
            "patch_size": 256,
            "step_size": 256,
            "level": 0,
            "prompt_hash": prompt_hash,
            "mode": mode,
            "manual_boxes": manual_boxes,
        }
        cache_dir = cache_root / sha1_text(str(cache_payload))
        expected_cache = self._expected_paths(cache_dir, case_spec.case_id)
        if expected_cache["boxes_json"].exists():
            self._copy_outputs(cache_dir, output_dir)
            payload = read_json(expected_cache["boxes_json"])
            return {
                "payload": payload,
                "mode": payload.get("mode", mode),
                "cache_hit": True,
                "paths": self._expected_paths(output_dir, case_spec.case_id),
                "attempts": [],
            }

        if manual_boxes:
            modes = ["manual"]
        elif mode == "auto":
            modes = ["patho-r1", "heuristic"]
        else:
            modes = [mode]

        attempts = []
        final_payload = None
        final_mode = None
        patho_r1_env = env_with_cuda_visible_devices(
            runtime.get("execution", {}).get("patho_r1_cuda_visible_devices")
        )
        for current_mode in modes:
            command = [
                paths["patho_r1_python"],
                paths["route_c_select_script"],
                "--slide-path",
                case_spec.slide_path,
                "--output-dir",
                str(output_dir),
                "--mode",
                current_mode,
                "--thumbnail-max-size",
                str(trace_cfg["thumbnail_max_size"]),
            ]
            if current_mode == "patho-r1":
                command.extend(
                    [
                        "--model-id",
                        runtime["models"]["patho_r1_model_id"],
                        "--cache-dir",
                        runtime["models"]["patho_r1_cache_dir"],
                    ]
                )
            if current_mode == "manual" and manual_boxes:
                command.extend(["--manual-boxes", manual_boxes])
            result = run_command(
                command,
                timeout=self.bundle["budget"].get("stage_timeout_seconds", {}).get("trace", 300),
                env_overrides=patho_r1_env if current_mode == "patho-r1" else None,
            )
            attempts.append(result)
            expected = self._expected_paths(output_dir, case_spec.case_id)
            if result["returncode"] == 0 and expected["boxes_json"].exists():
                final_payload = read_json(expected["boxes_json"])
                final_mode = current_mode
                break

        if final_payload is None:
            raise RuntimeError(
                "Route C selection failed for case {0}. Attempts: {1}".format(
                    case_spec.case_id, attempts
                )
            )

        ensure_dir(cache_dir)
        self._copy_outputs(output_dir, cache_dir)
        write_json(cache_dir / "attempts.json", attempts)
        return {
            "payload": final_payload,
            "mode": final_mode,
            "cache_hit": False,
            "paths": self._expected_paths(output_dir, case_spec.case_id),
            "attempts": attempts,
        }
