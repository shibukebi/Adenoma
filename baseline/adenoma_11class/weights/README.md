# Benchmark checkpoints

Verified checkpoints are arranged as:

```text
weights/<model>/<feature>/fold-<0..4>/checkpoint.pt
```

The standardized filename is independent of the historical training filename.
`weights_manifest.json` preserves the original filename, historical result
root, SHA-256 checksum and optional download URL.

Checkpoint binaries are not present while the USB2 result storage is
unmounted. Use `scripts/build_weights_manifest.py --copy-weights` after the
storage is restored. Install Git LFS before adding copied binaries to Git.
