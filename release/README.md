# GitHub Release Area

This directory is the controlled hand-off area for files that are safe and
useful to publish with the experiment code.

The long-term baseline benchmark definition now lives in
`baseline/adenoma_11class/`. This `release/` directory remains the repository
publication checklist and generic artifact collection area.

## Contents

- `weights/`: verified model checkpoints only. Large files are configured for
  Git LFS by the repository `.gitattributes` file.
- `experiment_artifacts.json`: machine-generated inventory of checkpoints and
  result files collected from local or mounted experiment result roots.
- `sha256sums.txt`: checksums for copied weights, when weights are available.

For the structured 70-checkpoint benchmark manifest, use
`baseline/adenoma_11class/scripts/build_weights_manifest.py`.

The repository deliberately does not publish WSI files, UNI feature tensors,
WSI tile caches, SQLite review data, or raw private manifests. Those files are
large, data-dependent, or contain local server paths. The inventory records
their expected external location without copying them into GitHub.

## Collecting old experiment weights

The previous USB2 result root must be mounted before collecting weights. From
the repository root, run:

```bash
python scripts/prepare_github_release.py \
  --weights-root <mounted-result-root>/5fold_11class \
  --weights-root <mounted-result-root>/fold5_hp+yx \
  --output release/experiment_artifacts.json \
  --copy-weights
```

The command copies only files whose names identify them as model checkpoints
(`*.pth`, `*.ckpt`, `s_*checkpoint.pt`, `best*.pt`, `model*.pt`) and records a
SHA-256 checksum. Feature files named after slides are not copied.

If a result root is unavailable, the command still writes an inventory with a
clear `missing_root` entry. Do not fabricate or rename a checkpoint to make a
missing experiment appear complete.

The default inventory redacts absolute server paths and is suitable for GitHub
publication. Use `--include-source-paths` only when generating an internal
archive.

## GitHub upload sequence

Install Git LFS before adding `release/weights/`:

```bash
git lfs install
git add .gitattributes .gitignore release/weights release/experiment_artifacts.json
git commit -m "Add verified experiment checkpoints"
git push origin main
```

For a public repository, review the inventory and every result CSV for slide
identifiers, local paths, patient identifiers, and unpublished labels before
publishing. A private repository is the appropriate default for the current
data and checkpoint collection.
