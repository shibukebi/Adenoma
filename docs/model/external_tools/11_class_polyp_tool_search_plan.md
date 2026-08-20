# 11-Class Polyp/Adenoma External Tool Search Plan

## Purpose

The current AgentFlow should not rely on low-magnification descriptive text as quantitative final evidence. This plan searches for external tools, datasets, models, and diagnostic references that can support or audit the final 11-class colorectal polyp/adenoma label space:

- `SSL`
- `SSLD`
- `HP`
- `TSA`
- `TSAD`
- `Unclassified serrated adenoma`
- `Tubular adenoma`
- `TAD`
- `Tubulovillous adenoma`
- `TVAD`
- `Inflammatory`

The output of the search is not assumed to replace the agent. The goal is to identify reusable external evidence sources that can become baselines, auxiliary classifiers, or structured references for Reviewer/Chief evidence integration and final audit.

## Search Keyword Strategy

Use three layers in each query:

1. Disease/subtype term.
2. AI/tool/data term.
3. Reproducibility term.

Core broad queries:

```text
colorectal polyp histopathology classification deep learning
colorectal adenoma serrated lesion classification AI
colorectal polyp whole slide image classifier adenoma serrated
colon polyp histology classification GitHub
colorectal polyp adenoma classification HuggingFace
```

Serrated lesion queries:

```text
sessile serrated lesion hyperplastic polyp classification deep learning
sessile serrated lesion traditional serrated adenoma histopathology AI
serrated colorectal polyp classification whole slide image
SSL HP TSA classification pathology AI
sessile serrated adenoma polyp deep learning classifier
```

Dysplasia-positive serrated queries:

```text
sessile serrated lesion dysplasia classification histopathology
traditional serrated adenoma dysplasia AI pathology
serrated polyp high-grade dysplasia deep learning
SSLD TSAD pathology classification
```

Conventional adenoma queries:

```text
tubular adenoma tubulovillous adenoma classification deep learning
colorectal adenoma villous component classification AI
colorectal adenoma high-grade dysplasia classification histopathology
tubular adenoma high grade dysplasia WSI classifier
tubulovillous adenoma dysplasia deep learning pathology
```

Inflammatory/reactive queries:

```text
inflammatory polyp colorectal histopathology classification AI
reactive inflammatory colorectal polyp deep learning
colorectal inflammatory polyp adenoma differential diagnosis AI
```

Source suffixes to rotate through:

```text
GitHub
HuggingFace
open source
pretrained model
dataset
benchmark
whole slide image
ROI classifier
patch classifier
foundation model
```

## Candidate Evaluation Matrix

For each candidate paper, tool, dataset, or model, record:

- Name and source URL.
- Year and institution/project.
- Input level: WSI, ROI, patch, image tile, or text/report.
- Output label space.
- Whether code is available.
- Whether model weights are available.
- License or access constraint.
- Evidence level: external validation, test-set size, metrics, and pathologist reference standard.
- 11-class coverage:
  - `direct`: output matches a project class.
  - `partial`: output supports a branch/subtype but not the exact final class.
  - `derived`: output can support the class only with added rules/checklists.
  - `missing`: no useful signal for the class.

Rating:

- `A`: runnable, labels close to the 11-class target, public validation, practical local reproduction path.
- `B`: strong paper/dataset, but incomplete labels or missing weights.
- `C`: useful auxiliary evidence only, such as a general foundation model or tissue classifier.
- `D`: background reference only.

## First-Pass Findings

### Highest-priority candidates: Dartmouth/Hassanpour colorectal polyp WSI classifiers

This line of work is the closest match found in the first pass. It targets WSI-level colorectal polyp classification and includes common serrated and conventional adenoma categories such as hyperplastic polyp, sessile serrated adenoma/lesion, traditional serrated adenoma, tubular adenoma, and tubulovillous/villous adenoma. It is promising for subtype coverage but does not appear to directly solve the project's dysplasia-positive split classes (`SSLD`, `TSAD`, `TAD`, `TVAD`) without an additional dysplasia classifier or checklist.

Initial rating: `B`.

### Dataset candidate: UniToPatho

UniToPatho is useful for patch-level colorectal polyp histology and grading tasks. It may help benchmark adenoma tissue, low-grade/high-grade dysplasia, and classification pipelines, but it does not directly provide the full project 11-class WSI diagnosis taxonomy.

Initial rating: `B/C`.

### General histology/foundation-model candidates

CONCH, UNI, PRISM-style models and CRC100K-style tissue classifiers can help screen tissue context or produce embeddings, but they generally do not directly output the 11 clinical classes. They should be treated as feature extractors or auxiliary evidence, not as final class tools.

Initial rating: `C`.

### Diagnostic references

WHO/CAP/PathologyOutlines-style diagnostic criteria remain important for converting model evidence into structured checklists, especially for:

- SSL vs HP vs TSA.
- Dysplasia in serrated lesions.
- Tubular vs tubulovillous/villous component thresholds.
- Inflammatory/reactive polyp differential diagnosis.

Initial rating: `D` as a tool, but `A` as checklist authority.

## Implementation Plan

1. Build `external_tool_candidate_table.md` with one row per candidate.
2. Build an 11-class coverage heatmap for the candidate rows.
3. Prioritize candidates in this order:
   - WSI-level polyp subtype classifiers.
   - Dysplasia/high-grade dysplasia classifiers.
   - Patch/ROI datasets with reusable labels.
   - Foundation-model embedding baselines.
   - Diagnostic-rule references.
4. For candidates rated `A` or `B`, check whether code/weights are available and whether local inference can be run on current ROI crops or WSI thumbnails.
5. For candidates rated `C`, define them only as auxiliary signals in the current workflow.
6. For classes not covered by any candidate, mark them as requiring the current Reviewer/Chief evidence pathway.

## Initial Sources To Track

- Korbar et al. colorectal polyp WSI classification paper: https://arxiv.org/abs/1703.01550
- Wei et al. multi-institution colorectal polyp WSI classification paper: https://arxiv.org/abs/1909.12959
- UniToPatho dataset/paper and any associated code repositories.
- PathologyOutlines pages for serrated lesions, adenomas, dysplasia, and inflammatory polyps.
- CAP/WHO-derived colorectal polyp diagnostic criteria summaries where accessible.
- GitHub/HuggingFace searches for runnable colorectal polyp classifiers.

## Acceptance Criteria

The first implementation pass is complete when:

- At least 8-12 candidate rows are collected.
- Each row has a source URL and a coverage rating.
- The table clearly identifies which of the 11 final classes remain unsupported by external tools.
- The recommended integration path states whether each candidate should be used as a baseline classifier, auxiliary evidence, checklist authority, or excluded.
