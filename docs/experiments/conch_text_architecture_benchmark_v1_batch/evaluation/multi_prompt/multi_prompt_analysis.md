# Multi-prompt CONCH Architecture Secondary Evaluation

The aggregation rule was fixed before executing either multi-prompt condition and was not selected or tuned using benchmark labels or performance.

`MULTI_PROMPT_AGGREGATION_RULE=per-prompt L2 normalize -> same-class arithmetic mean -> class-prototype L2 normalize -> cosine argmax`  
`MULTI_PROMPT_LABEL_TUNING=false`  
`BENCHMARK_VERSION=conch_text_architecture_benchmark_v1`  
`PROMPT_SHA256=59dde335493321b6a75f23e2550b47e6241fef40e709813f8a718be903120ef4`

## Primary comparison

| Prompt set | Accuracy | Balanced Accuracy | Macro F1 | Serrated F1 | Tubular F1 | Villous F1 |
|---|---:|---:|---:|---:|---:|---:|
| morphology_rich | 0.6064 | 0.4905 | 0.4770 | 0.6400 | 0.6575 | 0.1333 |
| single_class_name | 0.4681 | 0.4440 | 0.3698 | 0.6094 | 0.0000 | 0.5000 |
| matched_class_name_ensemble | 0.1489 | 0.3333 | 0.0864 | 0.0000 | 0.0000 | 0.2593 |

Best executed balanced-accuracy/Macro-F1 row: `morphology_rich` (only among executed conditions).

## Recall and predicted class counts

| Prompt set | Serrated Recall | Tubular Recall | Villous Recall | Pred Serrated N | Pred Tubular N | Pred Villous N |
|---|---:|---:|---:|---:|---:|---:|
| morphology_rich | 0.8000 | 0.6000 | 0.0714 | 60 | 33 | 1 |
| single_class_name | 0.9750 | 0.0000 | 0.3571 | 88 | 0 | 6 |
| matched_class_name_ensemble | 0.0000 | 0.0000 | 1.0000 | 0 | 0 | 94 |

## Prediction transitions from single-name

- `matched_class_name_ensemble__primary`: `{"CORRECT_BECAME_ERROR": 39, "ERRORS_CORRECTED": 9, "NET_CORRECT_GAIN": -30, "SERRATED_TO_TUBULAR": 0, "SERRATED_TO_VILLOUS": 39, "TOTAL_CHANGED_PREDICTIONS": 88, "TUBULAR_RESCUED": 0, "VILLOUS_RESCUED": 9}`
- `matched_class_name_ensemble__stress`: `{"CORRECT_BECAME_ERROR": 2, "ERRORS_CORRECTED": 1, "NET_CORRECT_GAIN": -1, "SERRATED_TO_TUBULAR": 0, "SERRATED_TO_VILLOUS": 2, "TOTAL_CHANGED_PREDICTIONS": 23, "TUBULAR_RESCUED": 0, "VILLOUS_RESCUED": 1}`
- `morphology_rich__primary`: `{"CORRECT_BECAME_ERROR": 11, "ERRORS_CORRECTED": 24, "NET_CORRECT_GAIN": 13, "SERRATED_TO_TUBULAR": 7, "SERRATED_TO_VILLOUS": 0, "TOTAL_CHANGED_PREDICTIONS": 37, "TUBULAR_RESCUED": 24, "VILLOUS_RESCUED": 0}`
- `morphology_rich__stress`: `{"CORRECT_BECAME_ERROR": 1, "ERRORS_CORRECTED": 3, "NET_CORRECT_GAIN": 2, "SERRATED_TO_TUBULAR": 1, "SERRATED_TO_VILLOUS": 0, "TOTAL_CHANGED_PREDICTIONS": 10, "TUBULAR_RESCUED": 3, "VILLOUS_RESCUED": 0}`

## Stress

Stress raw predictions remain separate. Metrics use the same three-class comparable subset rule as single-name; mixed/uncertain rows are not remapped.
- `single_class_name`: raw N=26, comparable N=11, accuracy=0.1818, balanced accuracy=0.3333, Macro-F1=0.1026
- `matched_class_name_ensemble`: raw N=26, comparable N=11, accuracy=0.0909, balanced accuracy=0.3333, Macro-F1=0.0556
- `morphology_rich`: raw N=26, comparable N=11, accuracy=0.3636, balanced accuracy=0.2917, Macro-F1=0.2407

No prompt wording was changed after evaluation began. No linear probe, K=5 classifier, MIL, or text-prior training was run.
