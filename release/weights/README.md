# Verified model checkpoints

This folder is reserved for verified training checkpoints from the adenoma
experiments. It is empty until the original result storage is mounted and the
files are collected with `scripts/prepare_github_release.py`.

Expected checkpoint families include:

- CLAM-SB: `s_*checkpoint.pt` or equivalent fold checkpoint;
- TransMIL: `best*.pt`, `model*.pt` or equivalent fold checkpoint;
- DSMIL: `*.pth` or `*.pt` model checkpoint;
- MIST: fold checkpoint such as `1.pth`.

UNI feature tensors, slide-level `.pt` files, WSI files and cached tiles are
not model checkpoints and must not be placed here.
