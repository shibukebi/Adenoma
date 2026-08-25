import subprocess
import time
from dataclasses import replace
from pathlib import Path

from adenoma_agent.utils import ensure_dir, read_json


class ThumbnailGridPreprocessor(object):
    """Prepare TraceAgent grid-thumbnail inputs when configured artifacts exist."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.config = bundle.get("runtime", {}).get("thumbnail_grid_preprocess", {})

    def prepare(self, case_spec, case_dir):
        started = time.time()
        if case_spec.input_mode == "grid_thumbnail":
            return {
                "case_spec": case_spec,
                "status": "skipped",
                "reason": "already_grid_thumbnail",
                "latency_ms": 0,
            }
        if not bool(self.config.get("enabled", False)):
            return {
                "case_spec": case_spec,
                "status": "skipped",
                "reason": "disabled",
                "latency_ms": 0,
            }

        output_dir = ensure_dir(self._output_dir(case_spec, case_dir))
        existing = self._find_artifacts(case_spec, output_dir)
        if existing is None and bool(self.config.get("generate_if_missing", False)):
            command_result = self._run_generator(case_spec, output_dir)
            existing = self._find_artifacts(case_spec, output_dir)
        else:
            command_result = None

        latency_ms = int(round((time.time() - started) * 1000.0))
        if existing is None:
            return {
                "case_spec": case_spec,
                "status": "skipped",
                "reason": "artifacts_missing",
                "output_dir": str(output_dir),
                "latency_ms": latency_ms,
                "command_result": command_result,
            }

        metadata_path, thumbnail_path, overview_path = existing
        grid_payload = read_json(metadata_path)
        next_metadata = dict(case_spec.metadata)
        next_metadata.update(
            {
                "thumbnail_grid_preprocess": {
                    "status": "ready",
                    "thumbnail_mode": grid_payload.get("thumbnail_mode"),
                    "grid_metadata_path": str(metadata_path),
                    "grid_thumbnail_path": str(thumbnail_path),
                    "overview_thumbnail_path": str(overview_path) if overview_path else None,
                }
            }
        )
        prepared_case = replace(
            case_spec,
            input_mode="grid_thumbnail",
            grid_thumbnail_path=str(thumbnail_path),
            grid_metadata_path=str(metadata_path),
            overview_thumbnail_path=str(overview_path) if overview_path else None,
            metadata=next_metadata,
        )
        return {
            "case_spec": prepared_case,
            "status": "ready",
            "reason": "artifacts_ready",
            "output_dir": str(output_dir),
            "grid_metadata_path": str(metadata_path),
            "grid_thumbnail_path": str(thumbnail_path),
            "overview_thumbnail_path": str(overview_path) if overview_path else None,
            "thumbnail_mode": grid_payload.get("thumbnail_mode"),
            "latency_ms": latency_ms,
            "command_result": command_result,
        }

    def _output_dir(self, case_spec, case_dir):
        configured_root = str(self.config.get("output_root", "")).strip()
        root = Path(configured_root) if configured_root else Path(case_dir) / "preprocess" / "thumbnail_grid"
        if bool(self.config.get("case_subdir", True)):
            return root / case_spec.case_id
        return root

    def _find_artifacts(self, case_spec, output_dir):
        metadata_path = self._configured_metadata_path(case_spec)
        candidates = [metadata_path] if metadata_path else []
        candidates.extend(sorted(Path(output_dir).glob("{0}_*tissuegrid*_ds32_level*_grid.json".format(case_spec.case_id))))
        candidates.extend(sorted(Path(output_dir).glob("{0}_*grid.json".format(case_spec.case_id))))
        for candidate in candidates:
            if candidate is None or not Path(candidate).exists():
                continue
            try:
                payload = read_json(candidate)
            except Exception:
                continue
            if not self._accept_metadata(payload):
                continue
            thumbnail_path = self._thumbnail_path_for_metadata(Path(candidate))
            if thumbnail_path is None or not thumbnail_path.exists():
                continue
            overview_path = self._overview_path_for_thumbnail(thumbnail_path)
            return Path(candidate), thumbnail_path, overview_path
        return None

    def _configured_metadata_path(self, case_spec):
        metadata_root = str(self.config.get("metadata_root", "")).strip()
        if not metadata_root:
            return None
        root = Path(metadata_root)
        direct = root / "{0}_grid.json".format(case_spec.case_id)
        if direct.exists():
            return direct
        matches = sorted(root.glob("{0}_*tissuegrid*_ds32_level*_grid.json".format(case_spec.case_id)))
        return matches[0] if matches else direct

    def _accept_metadata(self, payload):
        accepted_modes = set(self.config.get("accepted_thumbnail_modes") or ["tissue_grid32x_svs", "tissue_grid32x_isyntax"])
        return payload.get("thumbnail_mode") in accepted_modes and isinstance(payload.get("grid_cells", []), list)

    def _thumbnail_path_for_metadata(self, metadata_path):
        configured_thumbnail_root = str(self.config.get("thumbnail_root", "")).strip()
        if configured_thumbnail_root:
            candidate = Path(configured_thumbnail_root) / metadata_path.name.replace("_grid.json", "_grid.jpg")
            if candidate.exists():
                return candidate
        return metadata_path.with_name(metadata_path.name.replace("_grid.json", "_grid.jpg"))

    def _overview_path_for_thumbnail(self, thumbnail_path):
        overview_path = thumbnail_path.with_name(thumbnail_path.name.replace("_grid.jpg", ".jpg"))
        return overview_path if overview_path.exists() else None

    def _run_generator(self, case_spec, output_dir):
        script = Path(str(self.config.get("script_path", "scripts/generate_wsi_thumbnails.py")))
        python = str(self.config.get("python", "python3"))
        slide_path = Path(case_spec.slide_path)
        if not slide_path.exists():
            return {"returncode": None, "stdout": "", "stderr": "slide_path does not exist", "latency_ms": 0}
        input_dir = slide_path.parent
        command = [
            python,
            str(script),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--mode",
            str(self.config.get("mode", "tissue_grid32x_svs")),
            "--grid-cell-size-thumb",
            str(int(self.config.get("grid_cell_size_thumb", 64))),
            "--grid-physical-level0-extent",
            str(int(self.config.get("grid_physical_level0_extent", 2048))),
            "--tissue-coverage-threshold",
            str(float(self.config.get("tissue_coverage_threshold", 0.05))),
        ]
        if bool(self.config.get("recursive", False)):
            command.append("--recursive")
        if bool(self.config.get("overwrite", False)):
            command.append("--overwrite")
        for key, flag in (
            ("segmentations_dir", "--segmentations-dir"),
            ("patch_h5_dir", "--patch-h5-dir"),
            ("mask_jpg_dir", "--mask-jpg-dir"),
        ):
            value = str(self.config.get(key, "")).strip()
            if value:
                command.extend([flag, value])
        if bool(self.config.get("grid_align_to_patch_h5", False)):
            command.append("--grid-align-to-patch-h5")

        started = time.time()
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "latency_ms": int(round((time.time() - started) * 1000.0)),
            "command": command,
        }
