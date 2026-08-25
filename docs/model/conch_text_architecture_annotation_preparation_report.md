# Expert 5x Architecture ROI Annotation Preparation Report

This is Stage A only. No benchmark freeze, split generation, low-data subset,
CONCH evaluation, linear probe, text-prior training, or MIL training was run.

## 1. Source audit

- source villous WSI: 110
- label-valid WSI: 110
- WSI file exists: 110
- diagnosis counts: `{'Tubulovillous adenoma': 110}`

## 2. Existing Mucosa coverage

- already complete: 4
- missing before backfill: 106
- invalid WSI: 0
- invalid physical metadata: 0
- valid 5x manifest: 4
- eligible after current gates: 4

## 3. Selective Mucosa backfill

- requested villous WSI: 70
- reused valid full-baseline artifacts: 1
- reused valid POC-backfill artifacts on verification: 69
- newly processed: 69
- failures: 0
- outputs: `artifacts/conch_text_architecture_poc_v1/mucosa_backfill/`

## 4. Final annotation package

- candidate ROIs: 180
- unique WSI: 180
- unique families: 145
- internal recruitment: serrated=60, tubular=60, villous=60

## 5. Independence and blinding

- one default ROI per WSI: yes
- duplicate ROI: 0
- duplicate WSI: 0
- slide diagnosis exposure: 0
- recruitment stratum exposure: 0
- model score exposure: 0

## 6. Image provenance

- PNG count: 180
- PNG SHA completeness: 180/180
- MPP completeness: 180/180
- FOV completeness: 180/180
- Mucosa provenance completeness: 180/180

## 7. Stop gate

`FINAL_GATE=WAITING_FOR_EXPERT_ANNOTATION`  
`MODEL_EVALUATION_STARTED=false`  
`BENCHMARK_FREEZE_COMPLETE=false`  
`PROMPT_REVIEW_REQUIRED_LATER=true`

## 8. Next action

`EXPERT_REVIEW_ANNOTATION_XLSX`
