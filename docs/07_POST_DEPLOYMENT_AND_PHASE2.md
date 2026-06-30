# Post-Deployment Monitoring & Phase-2 Extensions

---

## Why This Exists

The base pipeline (detect → decide → defend → verify) treats remediation as a **one-time pre-deployment check.** This section turns it into a **continuous defense lifecycle** — sanitize once, then keep monitoring after deployment. This was explicitly pushed by the project guide and is a strong, realistic, publishable addition once Phase-1 is solid.

**Final framing for the whole project:**
> **"Adaptive Hybrid Machine Unlearning Pipeline for Backdoor Defense Across the Model Lifecycle"**

Stages:
```
1. Pre-deployment scanning
2. Adaptive remediation
3. Post-deployment monitoring
4. Continual defensive updates
```

This reframes the project from "an experiment" to "a complete ML security system" — a stronger narrative.

---

## Current Pipeline (Before This Extension)

```
train model → detect backdoor → unlearn/fine-tune → verify with STRIP → deploy model
```

Once deployed → nothing else happens. This extension fixes that.

---

## What to Add: Post-Deployment Feedback Loop

```
Deploy model
     ↓
Users send new images
     ↓
Model predicts
     ↓
System logs: image, prediction, entropy, STRIP result
     ↓
Later the true label arrives (simulated)
     ↓
Add these samples to a trusted dataset
     ↓
Periodically fine-tune / update defenses
```

This is **continual learning for security monitoring** — not full online learning, and the wording matters (see "Important Wording" below).

---

## Architecture: Deployment Monitoring Layer

```
                 Production
                     │
           Incoming images stream
                     │
            Prediction + logging
                     │
          Security monitor (STRIP)
                     │
          suspicious samples buffer
                     │
              periodic retraining
                     │
           adaptive controller runs
```

This creates a **self-healing system**.

---

## Implementation Plan (Simulated, No Real Infra Needed)

### Step 1 — Simulate Production Stream
```python
production_stream/
   clean_images/
   triggered_images/
   real_photos/        # from Custom dataset + STL-10
```
Feed sequentially, not all at once:
```python
stream = torch.utils.data.ConcatDataset([stl_dataset, custom_dataset])
stream_loader = DataLoader(stream, batch_size=1, shuffle=True)

for image in stream_loader:
    prediction = model(image)
    entropy = STRIP(image)
```

### Step 2 — Detect Suspicious Samples
```python
if STRIP_entropy < threshold:
    mark_as("suspected_backdoor")
    store_in("suspicious_buffer/")
```

### Step 3 — Simulate Later Ground-Truth Labels
Classify buffered samples into: `clean_correct`, `clean_misclassified`, `triggered`, `unknown`

### Step 4 — Build Trusted Update Dataset
```python
trusted_update_set = clean_correct + confirmed_trigger_samples
```

### Step 5 — Periodic Model Update
Every N samples (or on a fixed schedule):
```python
fine_tune(model, trusted_update_set)
```
Or re-run the full pipeline: AC detection → controller decision → unlearning → verification.

---

## Critical Risk: Catastrophic Forgetting

Continual updates can cause the model to forget previously-learned classes/knowledge while adapting to new data.

**Mitigation (simple, sufficient for this project's scope):**
- Maintain a fixed `clean_memory_buffer` (~500 CIFAR images)
- Every update trains on `new_samples + memory_buffer`, never on new samples alone

```python
memory_buffer = sample(500, "CIFAR clean images")
# during updates:
train_on(new_samples + memory_buffer)
```

---

## Important Wording (Reviewer-Safe — Use Exactly)

- ❌ Do NOT say "online learning"
- ❌ Do NOT say "adaptive defense" (in the continual-learning context — this term is already used for the controller)
- ✅ Say: **"Periodic safety re-evaluation under incremental data updates."**

This framing avoids overclaiming a fully online/adaptive system while still describing genuinely useful behavior.

---

## Example Experiment Sequence

**Phase 1 — pre-deployment:** Train + sanitize model. Measure `CA`, `ASR`.

**Phase 2 — simulated deployment:** Stream e.g. 1,000 normal images + 100 triggered images. Log detection rate and false alarm rate.

**Phase 3 — adaptive update:** Use detected/flagged samples to fine-tune. Re-measure `CA_after_update`, `ASR_after_update`.

**Example table shape (illustrative structure only — fill with real redone numbers):**

| Stage | Clean Accuracy | Attack Success Rate |
|-------|----------------|---------------------|
| Before deployment | ... | ... |
| After defense | ... | ... |
| After deployment attacks | ... | ... |
| After continual update | ... | ... |

This 4-row lifecycle table is the target final output for this section of the paper. Reviewers respond well to this because it shows system adaptation, not just a static before/after snapshot.

---

## Other Phase-2 Extensions (After Base Pipeline + Post-Deployment Loop Both Work)

### Cache Fingerprinting (Light-Touch — Not Attack Classification)

Cache per-checkpoint diagnostic signatures:

| Cached Signal | From Where |
|---------------|------------|
| AC cluster size | training-time |
| Silhouette score | training-time |
| STRIP entropy stats | inference-time |
| Failure outcome | after remediation |

Then claim:
> "We observe recurring severity signatures across similar poisoning instances."

That's it — no ML model needed for this, no overclaiming. Build a fast cosine-similarity matcher against the fingerprint database so that a new model checkpoint can be quickly compared against previously-seen severity signatures.

### Incremental / Continual Learning (Safe Scope)

Train model in rounds:
```
t0: clean
t1: poisoned batch arrives
t2: remediation
t3: new data arrives again
```
At each round: re-run AC, recompute severity, re-run controller. This demonstrates controller stability under data drift — frame as **"periodic safety re-evaluation,"** not "online learning" or "adaptive defense."

### Productization Touches (Helps, Doesn't Hurt, Publication)

Package the pipeline as a CLI tool: `scan_model.py`

- CLI flags: `--dataset`, `--use-fingerprint`, `--unlearn-method`, `--fingerprint-check`
- JSON audit report output (summarizing severity score, chosen defense, before/after metrics)
- Sample CI/CD YAML hook (simulate integration into a Model Registry pipeline)
- Industry framing: **"Pre-deployment Model Safety Scanner"** — not "Enterprise-ready security product." Keep it modest and credible.

This is recommended as the final Phase-2/Week-6-7 task, after all core experiments are validated.

---

## Sequencing Recommendation

1. Finish and validate the redone Phase-1 pipeline (attacks, AC, controller, defenses, STRIP/Grad-CAM verification) completely first.
2. Add STL-10 cross-dataset evaluation (no retraining).
3. Add Custom dataset OOD evaluation.
4. Build the post-deployment monitoring loop using STL-10 + Custom as the simulated stream.
5. Only then attempt fingerprint caching, continual learning rounds, and CLI productization — these are genuine "if time permits" items, not core to the paper's contribution.
