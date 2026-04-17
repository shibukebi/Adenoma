from pathlib import Path

from adenoma_agent.utils import ensure_dir, run_command


class PatchExportAdapter(object):
    def __init__(self, bundle):
        self.bundle = bundle

    def export_samples(self, slide_path, coords_h5, output_dir, max_patches=16):
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        command = [
            self.bundle["runtime"]["paths"]["clam_python"],
            self.bundle["runtime"]["paths"]["export_patch_samples_script"],
            "--slide-path",
            str(slide_path),
            "--coords-h5",
            str(coords_h5),
            "--output-dir",
            str(output_dir),
            "--max-patches",
            str(max_patches),
        ]
        result = run_command(command, timeout=300)
        return {
            "result": result,
            "manifest": output_dir / "patch_manifest.csv",
        }
