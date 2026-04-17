from pathlib import Path

from adenoma_agent.utils import ensure_dir, run_command


class ClamCoordsAdapter(object):
    def __init__(self, bundle):
        self.bundle = bundle

    def boxes_to_h5(self, boxes_json, output_h5, patch_size=256, step_size=256, patch_level=0):
        output_h5 = Path(output_h5)
        ensure_dir(output_h5.parent)
        command = [
            self.bundle["runtime"]["paths"]["clam_python"],
            self.bundle["runtime"]["paths"]["route_c_boxes_to_h5_script"],
            "--boxes-json",
            str(boxes_json),
            "--output-h5",
            str(output_h5),
            "--patch-size",
            str(patch_size),
            "--step-size",
            str(step_size),
            "--patch-level",
            str(patch_level),
        ]
        return run_command(command, timeout=300)
