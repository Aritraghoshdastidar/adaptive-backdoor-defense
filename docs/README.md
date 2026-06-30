# Project Context Docs — Index

These documents consolidate everything decided across the three planning rounds (Jan / March / May) in `jhuthpathar_context` into a clean, implementation-ready context set. **The Set-3 (May) numeric experimental results have been deliberately excluded** — those experiments are being redone from scratch. Everything else (design, scope, architecture, controller logic, defense methods, dataset strategy, reproducibility rules, roadmap) is carried forward as the locked plan.

## Reading Order

1. **`00_PROJECT_MASTER.md`** — Start here. What the project is, the core research claim, contributions, what to claim/not claim, locked scope.
2. **`01_SYSTEM_ARCHITECTURE.md`** — Full pipeline diagram, module-by-module spec, recommended repo structure, tech stack.
3. **`02_ATTACKS_AND_DATASETS.md`** — BadNets/Blended/Label-Consistent details, dataset roles (CIFAR-10/STL-10/Custom/GTSRB), what's explicitly out of scope.
4. **`03_CONTROLLER_LOGIC.md`** — The Adaptive Hybrid Controller: state vector, objective, 2×2 decision quadrant matrix, calibration approach, the open "4th regime" question.
5. **`04_DEFENSE_METHODS.md`** — Fine-tuning, Pruning+FT, BAERASER-lite, conditional NAD — implementation notes, ablation matrix, required tables.
6. **`05_DETECTION_AND_VERIFICATION.md`** — Activation Clustering, STRIP, Grad-CAM/TAR — inputs, outputs, what each can and cannot tell you.
7. **`06_EXPERIMENT_PROTOCOL_AND_REPRODUCIBILITY.md`** — Seed/index discipline, shared-script workflow, per-run logging checklist. Critical for team coordination.
8. **`07_POST_DEPLOYMENT_AND_PHASE2.md`** — Continuous monitoring loop, catastrophic-forgetting mitigation, fingerprint caching, productization (CLI/audit reports).
9. **`08_ROADMAP_AND_NEXT_STEPS.md`** — What's locked vs. still open, suggested re-execution order (Stages A–I), team roles, "definition of done" checklist.
10. **`09_REFERENCES_AND_PAPER_FRAMING.md`** — Reviewer-safe sentences, paper section outline, required tables, industry relevance, publication targets, background concepts.

## Repo structure
```text
adaptive-backdoor-defense/
│
├── core/                # shared utilities — model definitions, data loading,
│                         # attack injection, detection logic, evaluation metrics
│
├── docs/                # project planning & architecture docs — scope,
│                         # controller logic, defense methods, experiment
│                         # protocol, roadmap
│
├── notebooks/           # experiment notebooks — attack runs, detection runs,
│                         # cached activations, saved model checkpoints
│
├── results/             # experiment outputs — CSV logs, plots/images,
│                         # summary writeups
│
├── attacks/             # (planned) standalone attack implementations
│                         # (BadNets, Blended, Label-Consistent)
│
├── detection/           # (planned) severity detection — activation
│                         # clustering + STRIP scoring
│
├── controller/          # (planned) adaptive decision logic that picks
│                         # a defense based on severity
│
├── defenses/            # (planned) remediation methods — fine-tuning,
│                         # pruning, unlearning, NAD
│
├── verification/        # (planned) post-defense checks — STRIP + Grad-CAM
│
├── deployment/          # (planned) post-deployment monitoring loop
│
└── scripts/             # (planned) CLI entry points & pipeline runners
```
## Quick Orientation

- **Project:** Adaptive Hybrid Machine Unlearning for Backdoor Defense (PW26_NVP_01, PES University)
- **Core idea:** Detect backdoor severity (Activation Clustering) → decide defense via a calibrated rule-based controller → remediate (fine-tune / prune / BAERASER-lite / NAD) → verify (STRIP + Grad-CAM) → monitor post-deployment.
- **Model:** ResNet-18 only. **Training dataset:** CIFAR-10 only.
- **Attacks:** BadNets, Blended, Label-Consistent.
- **Current state:** Design and architecture are stable and well-documented across three planning rounds; experimental results need to be regenerated cleanly following the reproducibility protocol in doc 06.

## If You're Picking This Up Cold

Read doc 00, then doc 08 (roadmap) to see exactly what to do next, then dive into 01–07 as needed for implementation detail while building.
