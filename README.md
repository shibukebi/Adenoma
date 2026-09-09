# Adenoma

This repository contains the reproducible code and configuration for the
adenoma MIL experiments, including CLAM-SB, TransMIL, DSMIL, Patch-GCN and the
UNI/MIST pipelines. It is prepared for GitHub publication without bundling
WSIs, private feature tensors, review databases or local caches.

The main 11-class comparison is organized as a versioned baseline benchmark
under `baseline/adenoma_11class/`. Start there for the task definition, shared
five-fold protocol, unified training launcher, result aggregation and
checkpoint manifest.

The locally modified MIST implementation used by the experiments is vendored
under `third_party/mist/`; its provenance and excluded generated files are
listed in `third_party/mist/ORIGIN.md`.

Fetch the pinned upstream CLAM and Patch-GCN working copies when needed:

```bash
./scripts/bootstrap_upstream_dependencies.sh
```

The GitHub hand-off procedure is documented in `release/README.md`. The
previous experiment checkpoints are collected only after their result root is
mounted and verified; use `scripts/prepare_github_release.py` to inventory and
copy model weights into the Git LFS-compatible release area.

This folder wraps the local `CLAM` checkout so we can preprocess the
`Adenoma_yx` `.svs` slides without editing the upstream CLAM repository.

Detailed preprocessing guide:

- `doc/IMPLEMENTATION.md`
- `doc/METHOD_PRINCIPLES.md`
- `codex.md`
- `PREPROCESSING.md`
- `MAG_WORKFLOW_PLAN.md`

## Paths

- Project root: `/data15/data15_5/yuexin2/adenoma`
- CLAM repo: `/data15/data15_5/yuexin2/CLAM`
- CLAM env: `/data15/data15_5/yuexin2/anaconda3/envs/clam_latest`
- SVS source: `/data15/zhengke_usb/Adenoma_yx`

## Important folders

- `config/`: project-specific CLAM paths and defaults
- `scripts/`: wrappers for manifest creation and preprocessing
- `data/`: generated slide manifests for later CLAM stages
- `runs/`: CLAM preprocessing outputs
- `logs/`: background job logs and pid files

## Encoder note

Feature extraction is configured to use `UNI` (`MahmoodLab/UNI`) by default.
The model is gated on Hugging Face, so before running extraction/heatmaps you
should either:

- export a valid Hugging Face token, for example `HF_TOKEN=...`, after being granted access to `https://huggingface.co/MahmoodLab/UNI`
- or point `UNI_CKPT_PATH` / `--weights-path` to a local `pytorch_model.bin`

## Common commands

Build the slide manifest:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/build_slide_manifest.sh
```

Run a smoke test on one slide:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/make_smoke_input.sh
/data15/data15_5/yuexin2/adenoma/scripts/run_preprocess.sh --smoke
```

Start the full preprocessing job in the background:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/start_preprocess_background.sh
```

Check status:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/check_preprocess_status.sh
```

Prepare CLAM SSL/others training data and run a smoke training:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_clam_ssl_smoke.sh
```

Run 2.5X low-mag smoke preprocessing:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_preprocess_2p5x_smoke.sh
```

Run 5X preprocessing and feature extraction along the same low-mag path:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_preprocess_5x.sh
/data15/data15_5/yuexin2/adenoma/scripts/run_extract_features_5x.sh
```

Run formal 5X `clam_sb` training:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_clam_ssl_5x_clam_sb.sh
```

Run 5X smoke preprocessing and feature extraction:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_preprocess_5x_smoke.sh
/data15/data15_5/yuexin2/adenoma/scripts/run_extract_features_5x.sh --smoke
```

Run 2.5X low-mag smoke training:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_clam_ssl_2p5x_smoke.sh
```

Run the formal 2.5X `clam_sb` attention training route:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_clam_ssl_2p5x_clam_sb.sh
```

Run the 1X-style low-mag route. Because the slides only expose levels `0/1/2`,
this path approximates `1X` by using larger level-2 patches before resizing for
feature extraction:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_preprocess_1x.sh
/data15/data15_5/yuexin2/adenoma/scripts/run_extract_features_1x.sh
/data15/data15_5/yuexin2/adenoma/scripts/run_clam_ssl_1x_clam_sb.sh
```

TransMIL embedded-route training is also available for all four magnifications:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_transmil_ssl_20x.sh --fold 5
/data15/data15_5/yuexin2/adenoma/scripts/run_transmil_ssl_2p5x.sh --fold 5
/data15/data15_5/yuexin2/adenoma/scripts/run_transmil_ssl_5x.sh --fold 5
/data15/data15_5/yuexin2/adenoma/scripts/run_transmil_ssl_1x.sh --fold 5
```

Patch-GCN-style graph MIL is wired for all four UNI magnifications:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_patch_gcn_ssl_20x.sh --fold 5
/data15/data15_5/yuexin2/adenoma/scripts/run_patch_gcn_ssl_2p5x.sh --fold 5
/data15/data15_5/yuexin2/adenoma/scripts/run_patch_gcn_ssl_5x.sh --fold 5
/data15/data15_5/yuexin2/adenoma/scripts/run_patch_gcn_ssl_1x.sh --fold 5
```

Dual-stream DS-MIL is configured for the three paired-magnification baselines:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_dsmil_ssl_2p5x_5x.sh --fold 5
/data15/data15_5/yuexin2/adenoma/scripts/run_dsmil_ssl_5x_20x.sh --fold 5
/data15/data15_5/yuexin2/adenoma/scripts/run_dsmil_ssl_2p5x_20x.sh --fold 5
```

An official-style TransMIL subtree is vendored under `transmil_official/` with
separate configs, manifests, and wrappers:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_transmil_official_20x.sh
/data15/data15_5/yuexin2/adenoma/scripts/run_transmil_official_2p5x.sh
/data15/data15_5/yuexin2/adenoma/scripts/run_transmil_official_5x.sh
/data15/data15_5/yuexin2/adenoma/scripts/run_transmil_official_1x.sh
```

After training, generate representative heatmaps, efficiency summaries, and the markdown report:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_clam_ssl_2p5x_clam_sb_analysis.sh
```

20X `clam_sb` route is now configured in the same style:

```bash
/data15/data15_5/yuexin2/adenoma/scripts/run_clam_ssl_20x_clam_sb.sh
/data15/data15_5/yuexin2/adenoma/scripts/run_clam_ssl_20x_clam_sb_analysis.sh
```

Route C / Patho-R1 scripts are retained but currently parked until a GPU machine is available.
