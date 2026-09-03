# CONCH Text Architecture Benchmark v1 — Evaluation Batch

This is a GitHub-safe result/provenance bundle for the frozen
`conch_text_architecture_benchmark_v1`. It contains pseudonymous ROI IDs,
frozen public manifests, prompt snapshots, raw prediction tables, metrics,
transition analysis, and reports.

The bundle intentionally excludes raw WSI, reviewed PNG files, CONCH/UNI/
PathPrism checkpoints, Mucosa caches, absolute local paths, and restricted
recruitment provenance. Reproduction requires access to those private inputs
and the exact checkpoint SHA recorded in `freeze_manifest.json`.

## Frozen identity

- annotation SHA-256: `30f712a103cf77ac1cc68c25e504039bc8a5d0bb1027511b3d52ba9f0e90d68f`
- primary manifest SHA-256: `e034e3342bd564dd69e75650e11f110942e40e5fab78ee167c1b7d1e61c0bd33`
- split SHA-256: `b2dde478d72bba586819d7086fd1959f27b0a9dc14b8dfffb221d777c8725b77`
- prompt SHA-256: `59dde335493321b6a75f23e2550b47e6241fef40e709813f8a718be903120ef4`

## Evaluation scope

- Primary: 94 ROIs (`serrated=40`, `tubular=40`, `villous=14`)
- Stress: 26 ROIs; three-class stress metrics use the same comparable subset rule as the source run
- Executed: `single_class_name`, `matched_class_name_ensemble`, `morphology_rich`
- Aggregation: per-prompt L2 normalize → same-class arithmetic mean → prototype L2 normalize → cosine argmax
- Label tuning: false
- Reserve: not used
- K=5/K=10/K=20 classifiers: not run

## Headline primary results

| Prompt set | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| single class-name | 0.4681 | 0.4440 | 0.3698 |
| matched class-name ensemble | 0.1489 | 0.3333 | 0.0864 |
| morphology-rich | 0.6064 | 0.4905 | 0.4770 |

Morphology-rich prompting restored tubular recall from `0.0000` to `0.6000`
with 33 predicted tubular ROIs; matched ensemble predicted zero tubular ROIs.

## Provenance

`SHA256SUMS` covers every payload file in this bundle (the self-describing
`batch_manifest.json` is written after the checksum list). `batch_manifest.json` records
the source artifact hashes, model identity, evaluation status, excluded data
classes, and the exact generated file list.

The source implementation remains in:

```text
src/adenoma_agent/conch_text_architecture_poc/
scripts/run_conch_text_architecture_poc.py
```

No model evaluation command should be re-run against this GitHub bundle alone;
it is a shareable result package, not a public image dataset.
