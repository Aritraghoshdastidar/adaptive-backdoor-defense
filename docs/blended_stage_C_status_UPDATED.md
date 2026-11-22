# Blended Stage C — Status

**Source of truth:** `blended_finalNB.ipynb`

## 1. Main sweep

| Poison rate | Clean accuracy | ASR @ `alpha_test=0.5` | ASR @ `alpha_test=0.1` |
|---|---:|---:|---:|
| 1% | 94.83% | 100.0% | 99.90% |
| 5% | 94.73% | 100.0% | 100.00% |
| 10% | 94.39% | 100.0% | 100.00% |

`alpha_test=0.1` matches `ALPHA_TRAIN=0.1` and is the matched-condition setting used for the final STRIP comparison.

## 2. Activation Clustering (AC)

| Poison rate | DR method | Silhouette | Suspicious fraction | PDR | Recall |
|---|---|---:|---:|---:|---:|
| 1% | PCA-2D | 0.3778 | 26.69% | 11.16% | 90.00% |
| 1% | ICA-10D | 0.1102 | 9.09% | 100.00% | 100.00% |
| 5% | PCA-2D | 0.5408 | 33.37% | 99.76% | 99.88% |
| 5% | ICA-10D | 0.0983 | 33.33% | 100.00% | 100.00% |
| 10% | PCA-2D | 0.6380 | 50.00% | 99.96% | 99.98% |
| 10% | ICA-10D | 0.1744 | 49.99% | 99.96% | 100.00% |

**Open decision:** The official DR method for Blended is not locked. PCA-2D gives a clearer monotonic silhouette trend; ICA-10D gives stronger PDR at 1%. Avoid silently mixing methods in controller inputs.

## 3. STRIP — final matched-condition results

Settings: `alpha_test=0.1`, `n_samples=500` total (250 triggered + 250 clean); threshold set at FRR=1%.

| Poison rate | Clean entropy μ | Std σ | Threshold | TPR | FPR |
|---|---:|---:|---:|---:|---:|
| 1% | 0.4298 | 0.1467 | 0.0885 | 28.0% | 2.4% |
| 5% | 0.4662 | 0.1568 | 0.1014 | 40.8% | 1.6% |
| 10% | 0.4671 | 0.1562 | 0.1037 | 0.0% | 2.0% |

`alpha_test=0.1` is the Blended trigger strength. `STRIP_ALPHA=0.5` is STRIP's separate overlay weight; do not conflate them.

The old TPR values `79.6% / 92.8% / 68.0%` are superseded. Update any workbook/controller reference tables that still contain them or inconsistent companion statistics.

## 4. Main finding

At 10% poison rate, the Blended model remains a strong backdoor under matched conditions (ASR=100%), but STRIP detects none of the triggered samples (TPR=0%). This is not evidence that the backdoor disappeared:
- Minimum triggered entropy is `0.1392`, above the threshold `0.1037`.
- AC remains strong on the same checkpoint (PCA-2D silhouette `0.638`, PDR `99.96%`).

**Conclusion:** STRIP has a detection failure at 10% poison rate while AC remains effective. The cause is not established; any explanation involving STRIP's blending mechanism remains a hypothesis until tested.

## 5. Additional analyses

### Low-count sweep (`alpha_test=0.1`)

| Poison count | 1 | 2 | 5 | 10 | 15 | 25 | 250 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASR | 0.3% | 0.6% | 0.9% | 1.8% | 5.5% | 49.3% | 100.0% |

STRIP TPR was 0.0% for counts 1–25 and 94.4% at count 250. At very low counts, AC silhouette/PDR should not be treated as a reliable measure of poison-driven clustering.

### Grad-CAM mean attention correlation

| Poison count/rate | 1 | 2 | 5 | 10 | 15 | 25 | 250 | 1% | 5% | 10% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Correlation | 0.878 | 0.911 | 0.873 | 0.825 | 0.850 | 0.710 | 0.728 | 0.803 | 0.725 | 0.609 |

Correlation generally decreases at higher poison levels; the 10% checkpoint has the lowest value.

## 6. Next steps
- Keep this as the concise Stage C record.
- Align the final results workbook and `scripts/evaluate_controller.py` with the STRIP table in Section 3.
- Run the same AC/STRIP evaluation for remaining attack types before full cross-attack calibration.
- Optional: test `STRIP_ALPHA` at 10% poison rate to investigate the possible mechanism. This is not required to report the observed result.
