# Blended Stage C — Results Consistency Change Log

**Source of truth:** `blended_finalNB.ipynb`

## Final STRIP settings
- `alpha_test=0.1`, matching the Blended training trigger strength.
- `n_samples=500` total: 250 triggered + 250 clean.
- `STRIP_ALPHA=0.5` is STRIP's separate overlay weight; it is not the Blended trigger strength.

## Correct final STRIP results

| Poison rate | Clean entropy μ | Std σ | Threshold | TPR | FPR |
|---|---:|---:|---:|---:|---:|
| 1% | 0.4298 | 0.1467 | 0.0885 | 28.0% | 2.4% |
| 5% | 0.4662 | 0.1568 | 0.1014 | 40.8% | 1.6% |
| 10% | 0.4671 | 0.1562 | 0.1037 | 0.0% | 2.0% |

The old TPR values `79.6% / 92.8% / 68.0%` came from an older/mismatched run and are superseded.

## Required file updates

1. **Final results workbook (`results.xlsx`)**
   - Replace Blended STRIP TPR with `28.0% / 40.8% / 0.0%`.
   - Align μ, σ, threshold, and FPR too if those columns are present.
   - Keep old values only in a clearly labelled history note, not as current results.

2. **`scripts/evaluate_controller.py`**
   - Align the Blended STRIP reference rows with the table above.
   - Check downstream use of the fields before changing controller logic. This is a data correction, not a controller redesign.

3. **`blended_stage_C_status.md`**
   - Replace the longer version with the concise updated status document supplied alongside this log.

## Finding to preserve

At 10% poison rate, the matched-condition Blended model still has 100% ASR, but STRIP TPR is 0%. AC remains strong (PCA-2D silhouette `0.638`, PDR `99.96%`). Report this as a STRIP detection failure at 10%, not as absence of a backdoor.

**No experiment rerun is required** just to update the workbook and documentation.
