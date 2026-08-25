# WSI 10-sample Grid Five-Class All-Cells Archive

Status: FROZEN_EVIDENCE
Date: 2026-06-13 (filesystem run provenance)
Purpose: Preserve the semantic provenance of the complete UNI PrismNet, CONCH zero-shot, and DIgePath five-class all-cells grid run while removing recomputable patch crops.
Code commit: `23bafe773ca93a74566347b1befc386b8232841c`
Main script: `scripts/run_wsi_grid_5class_experiment.py`
Config: `run_config.json`
Input cohort/data source: `/data1/yuexin/Adenoma/data/WSI_10sample`

The source run processed 48,111 patches and recorded 144,333 model predictions across six readable WSI files. Four unreadable WSI files are recorded in `skipped_wsi.jsonl` and `RUN_REPORT.md`.

Retained files include the run configuration, output schema, run report, summary, manifests, fused assignments, model predictions, tables, tissue-context and mucosa summaries, architecture smoke metadata, and representative overlay QC. The source `crops/` tree is intentionally omitted from this semantic archive because it is recomputable from the input WSI and manifest.

The original source directory remains as a reduced compatibility shell until downstream consumers are rechecked. See `provenance.json` for byte counts, hashes, and the exact reduction record.
