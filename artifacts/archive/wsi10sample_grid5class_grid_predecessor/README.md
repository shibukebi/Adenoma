# WSI 10-sample Grid Five-Class Predecessor Archive

Status: SUPERSEDED
Date: 2026-06-05 (filesystem run provenance)
Purpose: Preserve the provenance of the selected-cell predecessor run without retaining its recomputable crops.
Code commit: `23bafe773ca93a74566347b1befc386b8232841c`
Main script: `scripts/run_wsi_grid_5class_experiment.py`
Config: `run_config.json`
Input cohort/data source: `/data1/yuexin/Adenoma/data/WSI_10sample`

The configuration records `include_all_cells=false`. The result JSONL files are empty and the runner log records an OpenSlide failure on an unreadable WSI. The later all-cells run is the canonical replacement for this predecessor.

Retained files are the run configuration, logs, and empty result-file markers. The source `crops/` tree is intentionally omitted as recomputable intermediate output.
