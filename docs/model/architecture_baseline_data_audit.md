# Frozen CONCH 5x Architecture Baseline — Data Audit

Status: implementation baseline; final GPU execution is pending manual
execution in a CUDA-visible environment.

Execution commands and stage gates are fixed in
[`architecture_baseline_runbook.md`](architecture_baseline_runbook.md).

## Scope and read-only boundary

The formal training cohort is `Adenoma_yx` (`.svs`). The
`Adenoma_hp` (`.isyntax`) source is audit-only in this baseline because its
reader and conflicting workbook-key handling require a separate reviewed
decision. Both source roots are read-only. Artifacts must be written below
`artifacts/architecture_baselines/frozen_conch_5x_mil_v1/`.

Ground-truth workbook values are permitted only for cohort eligibility,
split construction, training labels, and post-hoc evaluation. They must not
be copied into the Mucosa manifest, CONCH input, embedding metadata, or model
provenance.

## Known audit results

The current workbook audit found 4,419 data rows and 4,104 unique slide keys.
There are 315 duplicate keys; 50 keys have conflicting `(type, grade)` values.
The YX source currently contains 1,608 matched `.svs` files. Its class counts
are HP 560, tubular adenoma 388, sessile serrated adenoma 266,
unclassified serrated adenoma 112, tubulovillous adenoma 110,
inflammatory polyp 88, and traditional serrated adenoma 84. Grade is
post-hoc only (1,513 low; 95 high). The formal split
builder must use the seven-class `type` vocabulary, exclude conflicting keys,
and keep the conservative specimen-family group entirely within one split.

The HP source contains 2,496 `.isyntax` files. All are present in the
workbook, but 50 slide keys have conflicting diagnosis/grade rows; those keys
are excluded, leaving 2,446 clean-label cases for audit purposes. HP remains
outside the formal training cohort in this experiment.

YX grouping uses the filename's accession-like second portion with its final
three section/scan digits removed when parseable. This produces 1,238
heuristic families. It is a conservative leakage control, not a verified
patient identifier.

The YX source has no accompanying real patch/ROI architecture annotation,
mask, or reviewed region table. The repository's 60-row annotation fixtures
are unlabeled/synthetic smoke fixtures and are not eligible as training
targets.

## Baseline status

### Baseline 1 — Frozen CONCH → Linear/MLP → patch/ROI label

`annotation_blocked`.

The implementation must retain the probe interface and provenance guard, but
must not broadcast a slide label to patches or treat synthetic fixtures as
real annotation. It becomes runnable only after a reviewed patch/ROI
annotation manifest is supplied.

### Baseline 2 — Frozen CONCH → ABMIL/DSMIL → slide label

Eligible for the YX weakly supervised experiment after the following checks
pass in the user's GPU-visible environment:

1. WSI reader and current 5x Mucosa manifest generation;
2. manifest physical provenance and coverage validation;
3. Frozen CONCH embedding cache validation;
4. deterministic stratified-group split validation;
5. no-label-leakage validation.

## Required execution evidence

The final audit artifact must include source-safe slide aliases, file counts,
label conflict handling, group statistics, split hashes, manifest hash,
CONCH checkpoint hash, GPU preflight details, and the exact command used.
Patient identifiers and raw source paths must not appear in the shareable
report.
