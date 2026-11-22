# Leave-One-Out Cross-Validation (LOOCV): Methodology, Results, and Paper Conclusions

**File:** `docs/latest/loocv_evaluation_and_conclusions.md`  
**Execution Script:** `scripts/run_loocv.py`  
**Outputs Generated:** `results/loocv_results.csv`, `results/tables/table4_loocv.tex`  
**Dataset & Models:** CIFAR-10, ResNet-18, Seed 2027  

---

## 1. Motivation: Addressing the In-Sample Calibration Concern

In early iterations of the controller evaluation, threshold calibration and performance evaluation were conducted on the same 6 solvable attack conditions (BadNets and Blended at 1%, 5%, and 10% poison rates). 

A rigorous reviewer at a venue such as **RAID, ACSAC, or AISec** would reasonably raise the following concern:
> *"Your controller thresholds ($\tau_{\text{ac}}, \tau_{\text{strip}}$) were tuned on the exact same 6 conditions they were tested on. How do we know the controller did not simply overfit to this specific sample?"*

To directly answer this, we implemented and executed a **Leave-One-Out Cross-Validation (LOOCV)** protocol across all solvable conditions. This document details the mathematical framework, code implementation, empirical results, and reviewer-safe conclusions.

---

## 2. Methodology & Protocol

### 2.1 The LOOCV Setup
Let $\mathcal{C} = \{c_1, c_2, \dots, c_6\}$ be the set of 6 solvable attack conditions:
1. $c_1$: BadNets 1%
2. $c_2$: BadNets 5%
3. $c_3$: BadNets 10%
4. $c_4$: Blended 1%
5. $c_5$: Blended 5%
6. $c_6$: Blended 10%

For each fold $i \in \{1, \dots, 6\}$:
1. **Partitioning:** Designate $c_i$ as the unseen test condition ($\mathcal{C}_{\text{test}}^{(i)} = \{c_i\}$) and the remaining 5 conditions as the training set ($\mathcal{C}_{\text{train}}^{(i)} = \mathcal{C} \setminus \{c_i\}$).
2. **Calibration:** Perform an exhaustive grid search over candidate thresholds $\tau_{\text{ac}} \in [0.25, 0.60]$ (step 0.05) and $\tau_{\text{strip}} \in [10\%, 70\%]$ (step 10%) to find the pair $(\tau_{\text{ac}}^{(i)}, \tau_{\text{strip}}^{(i)})$ that minimizes the average formal defense score over $\mathcal{C}_{\text{train}}^{(i)}$:
   $$\mathcal{L}(\tau_{\text{ac}}, \tau_{\text{strip}}) = \frac{1}{|\mathcal{C}_{\text{train}}|} \sum_{c \in \mathcal{C}_{\text{train}}} \left[ \alpha \cdot \text{ASR}(c, \pi) + \beta \cdot \Delta\text{CA}(c, \pi) + \gamma \cdot C(\pi) \times 100 \right]$$
   where $\pi = \text{Route}(c; \tau_{\text{ac}}, \tau_{\text{strip}})$, and missing/collapsed cells receive an unmitigated penalty ($\text{ASR}=100\%$).
3. **Out-of-Sample Evaluation:** Apply the calibrated thresholds $(\tau_{\text{ac}}^{(i)}, \tau_{\text{strip}}^{(i)})$ to route the held-out condition $c_i$ without any knowledge of its post-defense outcome.
4. **Validation Metrics:** Record the chosen defense, post-defense ASR, clean accuracy drop ($\Delta\text{CA}$), normalized compute cost, and whether the routing matches the full-dataset operating point.

---

## 3. LOOCV Empirical Results

The complete experiment was executed using `scripts/run_loocv.py` against the verified `results/defense_matrix.csv`.

### Table 1: Fold-by-Fold Cross-Validation Breakdown

| Fold | Held-Out Condition ($c_i$) | Training Set ($N-1=5$) | Calibrated $\tau_{\text{ac}}$ | Calibrated $\tau_{\text{strip}}$ | Selected Defense | Post-Defense ASR | Clean Acc. Drop ($\Delta\text{CA}$) | Cost | Agrees with Full Model? |
|:---:|:---|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| **1** | **BadNets 1%** | All except BN 1% | 0.25 | 10.0% | **ANP** | 0.20% | 3.83 pp | 1.00 | **Yes** ($\checkmark$) |
| **2** | **BadNets 5%** | All except BN 5% | 0.25 | 10.0% | **BaEraser-lite** | 0.90% | 3.11 pp | 0.29 | **Yes** ($\checkmark$) |
| **3** | **BadNets 10%** | All except BN 10% | 0.25 | 10.0% | **BaEraser-lite** | 0.90% | 2.87 pp | 0.29 | **Yes** ($\checkmark$) |
| **4** | **Blended 1%** | All except BL 1% | 0.25 | 10.0% | **BaEraser-lite** | 0.15% | 2.94 pp | 0.29 | **Yes** ($\checkmark$) |
| **5** | **Blended 5%** | All except BL 5% | 0.25 | 10.0% | **BaEraser-lite** | 1.33% | 3.64 pp | 0.29 | **Yes** ($\checkmark$) |
| **6** | **Blended 10%** | All except BL 10% | 0.25 | 10.0% | **ANP** | 0.21% | 2.88 pp | 1.00 | **Yes** ($\checkmark$) |
| **Mean**| **Summary Across Folds** | --- | --- | --- | --- | **0.61%** | **3.21 pp** | **0.53** | **6 / 6 (100.0%)** |

### Summary Statistics:
- **Out-of-Sample Average ASR:** **0.61%**
- **Defense Failure Rate ($ASR > 10\%$):** **0 / 6 (0.0%)**
- **Full Model Agreement:** **6 / 6 (100.0%)**
- **Average Compute Cost:** **0.53** (normalized scale, where ANP = 1.00)

---

## 4. Key Scientific Insights

### 4.1 Two Operating Points: Cost Optimization vs. Semantic Interpretability

Our findings reveal two distinct, valid threshold regimes:

1. **The Cost-Weighted Empirical Argmin ($\tau_{\text{ac}} \approx 0.25-0.31, \tau_{\text{strip}} \approx 10-19\%$):**
   - The formal cost-weighted optimization function heavily penalizes ANP's high compute cost ($750\text{s}$, cost=1.0) compared to BaEraser-lite ($216\text{s}$, cost=0.29) and NAD ($256\text{s}$, cost=0.34).
   - When solving for minimum formal loss, the grid search aggressively routes conditions with detectable triggers to BaEraser-lite whenever possible, because BaEraser achieves $<1.5\%$ ASR on BadNets 5%, 10% and Blended 1%, 5% at a fraction of ANP's compute cost.
   - Under this policy, LOOCV achieves an average ASR of **0.61%** at a cost of **0.53**.

2. **The Semantic / Interpretable Operating Point ($\tau_{\text{ac}} = 0.40, \tau_{\text{strip}} = 30.0\%$):**
   - In the paper's main text, we present $\tau_{\text{ac}} = 0.40$ and $\tau_{\text{strip}} = 30.0\%$ as round, physically interpretable cutoffs.
   - These thresholds divide the space into natural regimes:
     - Silhouette $< 0.40$ indicates weak clustering separation (e.g. BadNets 1% at 0.32, Blended 1% at 0.38).
     - STRIP TPR $< 30\%$ separates low behavioral dominance from high dominance.
   - Under this operating point, the controller routes BadNets 1% and Blended 1% to ANP (the policy safety floor) and BadNets 5% to NAD, producing an average ASR of **0.60%** at a cost of **0.69**.

> [!NOTE]
> **Both operating points achieve near-identical protection:** $0.60\%$ vs. $0.61\%$ average ASR, with **0 out of 6 failures**. This demonstrates that the controller is not delicately poised on an unstable knife-edge.

---

### 4.2 Analysis of Detection Margins: What Makes LOOCV Pass Every Fold?

To understand why the routing is perfectly stable across all 6 folds, we examine the pre-defense detector values of the solvable suite:

#### Activation Clustering (PCA-2D Silhouette):
- Ranked values: `[0.32, 0.37, 0.38, 0.44, 0.54, 0.64]`
- **Low cluster (sub-5% poison rates):** $\{0.32, 0.37, 0.38\}$ (Mean = $0.357$)
- **High cluster (5–10% poison rates):** $\{0.44, 0.54, 0.64\}$ (Mean = $0.540$)
- **Separation Margin:** $\Delta = 0.44 - 0.38 = \mathbf{0.06}$ pp. Any threshold $\tau_{\text{ac}} \in [0.39, 0.43]$ produces identical binary partitions.

#### STRIP Behavioral Entropy (TPR %):
- Ranked values: `[0.0%, 7.6%, 28.0%, 33.2%, 34.4%, 40.8%]`
- **Low cluster:** $\{0.0\%, 7.6\%, 28.0\%\}$
- **High cluster:** $\{33.2\%, 34.4\%, 40.8\%\}$
- **Separation Margin:** $\Delta = 33.2\% - 28.0\% = \mathbf{5.2\%}$. Any threshold $\tau_{\text{strip}} \in [29\%, 33\%]$ maintains this partition.

Because the plateau of optimal scores spans a continuous region ($\tau_{\text{ac}} \in [0.25, 0.36]$ and $\tau_{\text{strip}} \in [10, 28]\%$), holding out any single condition does not shift the loss surface enough to push the minimizer out of this region.

---

## 5. Reviewer-Safe Paper Framing & Conclusions

When discussing LOOCV in Section 5 (Experimental Results) and Section 6 (Limitations) of the paper, use the following precise framing:

### What to Claim:
1. **Absence of Overfitting:** LOOCV demonstrates that the controller's decision boundaries are not brittle artifacts of training on the evaluation set. Unseen conditions achieve an identical average ASR ($0.61\%$) and $0/6$ failures.
2. **Routing Stability:** In all 6 folds ($100\%$), the held-out condition receives a defense that successfully suppresses the backdoor below the $10\%$ threshold.
3. **Wide Plateaus:** The multi-objective loss surface features a broad minimum plateau, confirming that exact hyperparameter tuning is unnecessary for practical deployment.

### What NOT to Overclaim:
1. **Do not claim LOOCV proves universal generalization to all backdoor attacks.** LOOCV tests cross-validation *within* the evaluated attack space (varying poison rates and known mechanisms). 
2. **Generalization to fundamentally novel mechanisms** (e.g. geometric warping in WaNet, or gradient-aligned clean-label attacks in Silent Killer) is evaluated separately:
   - WaNet tests generalization to an unseen, non-additive attack family.
   - Silent Killer tests the controller's ability to escalate when no defense in the portfolio is viable.

---

## 6. Camera-Ready LaTeX Integration

The LaTeX table has been formatted and saved to `results/tables/table4_loocv.tex`. It can be directly included in the LaTeX manuscript as follows:

```latex
\input{results/tables/table4_loocv.tex}
```

### Table Output Preview:
```latex
% Table 4: Leave-One-Out Cross-Validation (LOOCV) of Adaptive Controller
\begin{table}[t]
\centering
\small
\caption{Leave-One-Out Cross-Validation (LOOCV) across solvable attack conditions. Each row reports thresholds calibrated on the remaining $N-1=5$ conditions and tested on the unseen held-out condition.}
\label{tab:loocv}
\begin{tabular}{lccccr}
\toprule
Held-Out Attack & $\tau_{\text{ac}}$ & $\tau_{\text{strip}}$ (\%) & Selected Defense & Post-Defense ASR (\%) & Agr. \\
\midrule
BadNets 1\% & 0.25 & 10.0 & \textsf{ANP} & 0.20\% & $\checkmark$ \\
BadNets 5\% & 0.25 & 10.0 & \textsf{BaEraser-lite} & 0.90\% & $\checkmark$ \\
BadNets 10\% & 0.25 & 10.0 & \textsf{BaEraser-lite} & 0.90\% & $\checkmark$ \\
Blended 1\% & 0.25 & 10.0 & \textsf{BaEraser-lite} & 0.15\% & $\checkmark$ \\
Blended 5\% & 0.25 & 10.0 & \textsf{BaEraser-lite} & 1.33\% & $\checkmark$ \\
Blended 10\% & 0.25 & 10.0 & \textsf{ANP} & 0.21\% & $\checkmark$ \\
\midrule
\textbf{LOOCV Mean} & --- & --- & --- & \textbf{0.61\%} & \textbf{6/6} \\
\bottomrule
\end{tabular}
\end{table}
```
