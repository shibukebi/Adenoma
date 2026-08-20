# Mucosa Extractor v1

Mucosa Extractor converts raw WSI or saved PathPrism tile predictions into a
coordinate-traceable tissue-context evidence package. It does not generate
diagnostic hypotheses and does not run architecture or reviewer models.

## Pipeline

```text
Raw WSI
  -> tissue-aware, level-0-aligned 20x lattice
  -> UNI + PathPrism CRC100K inference
  -> nine raw probability maps
  -> seven task-context maps
  -> high-recall mucosa search mask
  -> level-0-aligned 5x extraction manifest
```

The seven context channels are:

```text
DEB + ADI + BACK -> background_or_artifact
MUS              -> smooth_muscle_or_deep_tissue
NORM             -> normal_epi_context
LYM              -> inflammatory_context
MUC              -> mucus_rich_context
STR              -> stromal_context
TUM              -> abnormal_epithelial_candidate
```

`abnormal_epithelial_candidate` is a reviewer-routing candidate signal. It is
not evidence that dysplasia is present.

The binary mask remains deliberately simple:

```text
mucosa_score = P(normal_epi_context) + P(abnormal_epithelial_candidate)
mask = anatomical_postprocess(mucosa_score >= 0.30)
```

Inflammatory, mucus-rich, stromal, and muscle channels are retained as soft
evidence but do not activate the v1 mask.

## Usage

Raw WSI through the PathPrism HTTP service:

```bash
python3 scripts/run_mucosa_extractor.py \
  --wsi /path/to/slide.svs \
  --output-dir artifacts/example/mucosa_extractor \
  --pathprism-url http://127.0.0.1:8400/predict
```

Resume from an existing artifact containing `manifest.jsonl` and
`predictions.jsonl`:

```bash
python3 scripts/run_mucosa_extractor.py \
  --artifact-dir artifacts/existing_run
```

Use `--base-magnification` when the WSI reader cannot recover objective power.
The extractor never silently assumes a 40x base scan.

## Output Contract

The compact experiment contract is:

```text
mucosa_extractor/
├── manifest.json
├── tile_index.jsonl
├── five_x_patch_manifest.jsonl
├── errors.jsonl                 # only when errors occur
└── slides/<slide_id>/
    ├── maps.npz
    └── qc_panel.png
```

`manifest.json` combines run configuration, model identity, channel order,
coordinate metadata, counts, per-slide statistics, and output paths.

`tile_index.jsonl` is the single tile-level table. Each row includes coordinates,
raw CRC100K probabilities, seven task-context values, hard context, uncertainty,
mucosa score, final mask coverage, and inclusion status.

`maps.npz` contains only reusable spatial arrays:

```text
raw_tissue_probabilities [9, H, W]
task_context_scores      [7, H, W]
uncertainty              [H, W]
mucosa_score             [H, W]
mucosa_candidate_mask    [H, W]
valid_mask               [H, W]
```

`qc_panel.png` combines the WSI thumbnail, hard context, context overlay, mucosa
score, uncertainty, and final mask overlay. The individual PNGs are regenerated
from `maps.npz` when needed rather than stored permanently.

Presence flags use unvalidated operational defaults:

```text
hard_area_fraction >= 0.005 and peak_probability >= 0.50
```

Consumers must retain `presence_calibration_status=unvalidated`; these flags are not a
medical performance claim.

## Complete Output Example

An anonymized, presentation-oriented example is available at
[Mucosa Extractor v1 complete output example](../../artifacts/examples/mucosa_extractor_output_example/README.md).
It includes a real successful case, a synthetic error/empty-mask case, every formal
compact-v1 output type, and rendered views of all NPZ channels. The additional figures
are explanatory assets and are not part of the production output contract.
