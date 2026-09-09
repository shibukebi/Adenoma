# Adenoma 11-class benchmark results

Status: historical result storage is not mounted on this server.

The final table will report five-fold Mean and sample SD for Accuracy,
Macro-F1, Weighted-F1, Macro AUC and Weighted AUC across all 14 configurations.
It must be regenerated from the original per-fold `metrics.json` files with:

```bash
python baseline/adenoma_11class/scripts/evaluate.py \
  --results-root /path/to/benchmark-runs \
  --legacy-root /path/to/result/fold5_hp+yx \
  --output-dir baseline/adenoma_11class/results \
  --strict
```

Do not enter values manually when the original per-fold files are unavailable.
