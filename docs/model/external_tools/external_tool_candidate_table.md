# External Candidate Table For 11-Class Polyp/Adenoma Support

Legend:

- `D`: direct support.
- `P`: partial branch/subtype support.
- `R`: rule/checklist-derived support only.
- `M`: missing or not clinically specific enough.

## Candidate Summary

| Candidate | Type | Input | Reproducibility | Coverage rating | Integration recommendation | Source |
| --- | --- | --- | --- | --- | --- | --- |
| Korbar et al. colorectal polyp WSI classifier | Paper/model family | WSI | Paper available; code/weights still need confirmation | B | Highest-priority baseline candidate for 5-way subtype audit; add dysplasia checklist separately | https://arxiv.org/abs/1703.01550 |
| Wei et al. multi-institution colorectal polyp WSI classifier | Paper/model family | WSI | Paper available; code/weights still need confirmation | B | Strong external-validation candidate for 4-way subtype audit; missing TSA and dysplasia splits | https://arxiv.org/abs/1909.12959 |
| Computer-aided diagnosis of colorectal polyps using linked color imaging colonoscopy images | Clinical image AI paper | Endoscopy image | Not histology/WSI; useful only as exclusion/background | D | Exclude from histology baseline | https://pubmed.ncbi.nlm.nih.gov/32994549/ |
| UniToPatho | Dataset/benchmark | Histology patches | Dataset/paper available; code availability to confirm | B/C | Use for patch-level adenoma/dysplasia benchmark, not final WSI 11-class output | https://ieeexplore.ieee.org/document/9434064 |
| CRC100K / NCT-CRC-HE tissue classification family | Dataset/model family | Histology patches | Many public implementations; label space is tissue-type, not polyp diagnosis | C | Auxiliary tissue-context evidence only | https://zenodo.org/records/1214456 |
| CONCH-style pathology vision-language model | Foundation model | Patch/ROI text-image | Public model family; current repo already uses CONCH-style evidence | C | Keep as embedding/zero-shot auxiliary evidence, not final classifier | https://www.nature.com/articles/s41591-024-02856-4 |
| UNI pathology foundation model | Foundation model | Patch/ROI embedding | Public model family; current repo has UNI/PRISMNET server path | C | Use for embedding nearest-neighbor or linear probe only | https://www.nature.com/articles/s41591-024-02857-3 |
| PathologyOutlines serrated lesions criteria | Diagnostic reference | Text/rules | Public reference; not a model | D/A-reference | Convert into SSL/HP/TSA checklist authority | https://www.pathologyoutlines.com/topic/colontumorserrated.html |
| PathologyOutlines conventional adenoma criteria | Diagnostic reference | Text/rules | Public reference; not a model | D/A-reference | Convert into tubular/tubulovillous/dysplasia checklist authority | https://www.pathologyoutlines.com/topic/colontumoradenoma.html |
| PathologyOutlines inflammatory polyp criteria | Diagnostic reference | Text/rules | Public reference; not a model | D/A-reference | Convert into inflammatory/reactive checklist authority | https://www.pathologyoutlines.com/topic/colonnontumorinflammatory.html |
| GitHub/HuggingFace direct 11-class colorectal polyp model search | Model registry search | Varies | First pass found no clear direct 11-class pretrained model | D pending | Continue targeted search, but do not depend on finding one | https://huggingface.co/models |

## 11-Class Coverage Heatmap

| Candidate | SSL | SSLD | HP | TSA | TSAD | Unclassified serrated adenoma | Tubular adenoma | TAD | Tubulovillous adenoma | TVAD | Inflammatory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Korbar et al. WSI classifier | D/P | R | D/P | D/P | R | M | D/P | R | D/P | R | M |
| Wei et al. multi-institution WSI classifier | D/P | R | D/P | M | M | M | D/P | R | D/P | R | M |
| UniToPatho | M/P | R/P | M | M | M | M | P | R/P | M/P | R/P | M |
| CRC100K/NCT tissue classifier | M | M | M | M | M | M | M | M | M | M | M |
| CONCH-style foundation model | P/R | R | P/R | P/R | R | R | P/R | R | P/R | R | P/R |
| UNI-style foundation model | P/R | R | P/R | P/R | R | R | P/R | R | P/R | R | P/R |
| PathologyOutlines serrated criteria | R | R | R | R | R | R | M | M | M | M | M |
| PathologyOutlines adenoma criteria | M | M | M | M | M | M | R | R | R | R | M |
| PathologyOutlines inflammatory criteria | M | M | M | M | M | M | M | M | M | M | R |

## Immediate Gaps

- No first-pass candidate directly covers all 11 classes.
- Dysplasia-positive split classes need a separate high-grade/definite dysplasia evidence source:
  - `SSLD`
  - `TSAD`
  - `TAD`
  - `TVAD`
- `Unclassified serrated adenoma` is unlikely to be directly predicted by public models and should remain checklist/rule-derived.
- `Inflammatory` is better handled by a differential diagnosis checklist than by current polyp subtype classifiers.

## Next Actions

1. Verify whether Korbar/Wei/Dartmouth-Hassanpour code or weights are publicly available.
2. Search specifically for colorectal polyp high-grade dysplasia histology classifiers.
3. Search PubMed and GitHub for `serrated polyp dysplasia`, `traditional serrated adenoma dysplasia`, and `villous component adenoma classifier`.
4. If no runnable direct model is found, define a pragmatic baseline:
   - external WSI subtype classifier where available;
   - foundation-model embedding similarity for ROI support;
   - explicit dysplasia and subtype evidence checks in Reviewer/Chief integration.
