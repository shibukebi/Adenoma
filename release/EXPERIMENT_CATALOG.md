# Adenoma experiment catalog

The canonical machine-readable and runnable definition is now maintained in
`baseline/adenoma_11class/benchmark.yaml`; this file is retained as a concise
publication catalog.

This catalog describes the experiments prepared by the repository. The
original result storage is external to the code repository. A checkpoint is
listed as available only after `prepare_github_release.py` finds it and records
its SHA-256 checksum.

## Core 11-class five-fold experiments

The following experiments use the same 11-class labels and the same five-fold
train/validation/test split family:

| Model | Feature scale | Expected artifact |
| --- | --- | --- |
| CLAM-SB | 2.5x | one checkpoint per fold |
| CLAM-SB | 5x | one checkpoint per fold |
| CLAM-SB | 10x | one checkpoint per fold |
| CLAM-SB | 20x | one checkpoint per fold |
| TransMIL | 2.5x | one checkpoint per fold |
| TransMIL | 5x | one checkpoint per fold |
| TransMIL | 10x | one checkpoint per fold |
| TransMIL | 20x | one checkpoint per fold |
| DSMIL | 2.5x | one checkpoint per fold |
| DSMIL | 5x | one checkpoint per fold |
| DSMIL | 10x | one checkpoint per fold |
| DSMIL | 20x | one checkpoint per fold |
| MIST | 2.5x + 5x | one checkpoint per fold |
| MIST | 5x + 10x | one checkpoint per fold |

The cancelled MIST 10x + 20x experiment is intentionally not included in the
catalog.

## What belongs in GitHub

Include the model implementation, training/evaluation scripts, configuration
templates, split-generation logic, documentation, tests and verified model
checkpoints. Keep WSI files, UNI feature tensors, patch files, tile caches,
SQLite review data and local virtual environments outside the repository.

For a public repository, also review prediction CSVs and figures because slide
identifiers may still be identifiable even when raw WSI files are excluded.

## Reproducibility inputs

The code expects the data and feature roots to be supplied on the target
machine. The repository does not claim that a run is reproducible from code
alone unless the matching data, split files, feature version, environment and
checkpoint checksum are recorded in the release inventory.
