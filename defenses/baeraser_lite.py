"""
BAERASER-LITE — shared defense core for BadNets / Blended / Silent Killer.

This is the compute-budgeted refactor of baeraser_lite_latest.py.

Design (unchanged from the previous version):
  - Shared BaEraser-lite trigger-recovery + candidate-selection + unlearning core.
  - Attack-specific TriggerOperator controls how a recovered trigger is applied.
  - BadNets: localized 4x4 patch.
  - Blended: full-image key pattern, x'=(1-alpha)x+alpha*k.
  - Silent Killer: full-image additive perturbation, x'=clip(x+delta).

WHAT CHANGED IN THIS VERSION AND WHY:

  1. Recovery is now coarse-to-fine with early exit (see `recover_trigger_pool`).
     Every checkpoint still gets genuine Generator+MINE trigger recovery — we do
     NOT skip recovery for the "hard" attacks — we just stop searching once we
     have enough evidence instead of exhaustively sweeping every epsilon
     regardless of how early candidates are already succeeding or failing.

  2. Recovery budgets (epsilon count, epochs, candidates) are now per-attack,
     declared explicitly in ATTACK_RECOVERY_CONFIG below, the same pattern
     core/detection.py already uses for ATTACK_DR_CONFIG. Don't tune these by
     editing magic numbers inline — update this dict (and document why in
     04_DEFENSE_METHODS.md) so the choice is auditable.

  3. (SUPERSEDED by v3 item 8e: warm-starting/chaining is now OFF by default.)
     Warm-starting: recovery for a given attack's 2nd/3rd poison
     rate/count can initialize the Generator from the previous rate's
     converged weights instead of random init (`warm_start_state`). Same
     trigger family, different checkpoint — this cuts convergence time
     without changing what information the defense has access to.

  4. omega (the dynamic clean-data penalty weighting) is no longer
     recomputed every single epoch by default. `omega_update_freq` controls
     how often it refreshes; the previous behaviour (every epoch) is still
     available via omega_update_freq=1 for anyone who wants to reproduce it
     as an ablation.

  5. oracle_trigger is now explicitly gated as a DIAGNOSTIC-ONLY path, not
     an alternate way to get "BAERASER-lite" numbers. It exists to answer
     "does the unlearning objective work at all if recovery were perfect?" —
     it is NOT a deployable method (no real deployment has the attacker's
     literal trigger file) and must never be reported as a BAERASER-lite
     result in the main defense x attack matrix. Calling it requires
     diagnostic_only=True and results are tagged "mode": "oracle_diagnostic".

  6. BlendedTriggerOperator.alpha is now a REQUIRED constructor argument
     (no more silently-mismatched default of 0.15) — pass the exact
     alpha_train used to generate the Blended checkpoint you're defending.

  7. v2 FIXES (review of the executed Blended notebook):
       a. MODEL SELECTION NO LONGER TOUCHES TEST DATA. v1 chose the best epoch, the
          early-stop epoch and the CA floor from test-set CA and test-set ASR, then
          reported those same numbers (a min over <=15 epochs on the reporting set).
          Pass `select_loader` (a held-out slice of the defense budget; see
          make_select_split). Selection then uses held-out clean CA and a PROXY ASR
          (recovered triggers on held-out non-target images); test loaders are
          monitoring-only. Without select_loader the old leaky path still runs but
          warns and tags the result selection_source="test_monitor (LEAKY)".
       b. "exact ASR" excludes target-class images (v1 counted them as successes).
       c. Candidate acceptance can use a held-out split (`heldout_loader`) instead of
          the images the generator was trained on.
       d. Reported unlearning cost excludes test-set monitoring (v1 included it).
       e. Fine-phase seeds no longer repeat the coarse-phase seeds.
       f. Optional `freeze_bn` (default False): see baeraser_unlearning.

  8. v3 FIXES (review of the v2 files; items marked [protocol] also need notebook changes):
       a. NO ORACLE SELECTION. The checkpoint is chosen only from defender-computable
          signals: held-out clean CA (floor) + proxy ASR on the recovered triggers,
          with ties inside `asr_tie_tol` broken by held-out CA. Optional
          `fresh_select_k>1` re-ranks the top-k epochs by the ASR of FRESHLY recovered
          triggers on that epoch's sanitized weights (harder to game than a proxy
          the model was trained against). Early stopping is OFF by default
          (`asr_stop_threshold=None`): all epochs run, real ASR is logged next to proxy
          ASR, and the real ASR AT THE SELECTED EPOCH is the headline number. The
          epoch with the best real ASR is reported separately as `oracle_*`
          (upper bound, never a result). select_loader is now REQUIRED unless
          allow_leaky_selection=True.
       b. [protocol] Held-out split. The poisoned checkpoints saw all 50k train images,
          so any defense slice drawn from train overlaps by construction.
          `split_test_set` gives a stratified val/report split of the CIFAR TEST set
          (default 2,500 / 7,500): build the defense budget AND select slice from the
          val half; the ASR set and CA monitor must use the report half only.
          Re-run FT / ANP / NAD on the same split.
       c. [protocol] `assert_baseline_ca` checks a loaded checkpoint reproduces its
          recorded CA (normalisation constants are not stored in a checkpoint).
          Run it on the FULL test loader before switching to the report half.
       d. Target-class images are excluded from the recovery hinge and from the
          triggered batches in unlearning (`exclude_target=True`).
          `trigger_loss_mode="true_label"` (minimise CE of triggered->TRUE label) is
          kept as an ablation; `trigger_ascent_cap` optionally bounds the ascent term.
       e. Recovery no longer chains generator/MINE state between epsilons. Every
          epsilon starts from a fresh random init (`chain_epsilons=False`); a
          caller-supplied warm_start_state, if any, is applied identically to every
          epsilon. Init/final weight hashes are logged per candidate and checked
          (`recovery_meta["fresh_init_verified"]`). Warm-starting stays OFF in the
          main matrix. final_*_state are only returned with return_states=True.
       f. Clean-model control + generality checks: run the identical pipeline on a
          CLEAN model, compare with `summarize_result`; `fresh_recovery_audit`
          (fresh triggers on the sanitized model) and `universal_adv_vulnerability`
          (fresh L_inf universal perturbation) show whether the gain is specific to
          the backdoor or generic adversarial robustness.
       g. `baeraser_lite(inplace=False)` copies the victim (the original is kept for
          before/after audits); the returned model has the SELECTED weights loaded
          (v2 returned last-epoch weights; callers had to load best_state_dict).
          Reported cost now includes recovery + selection audits, not only unlearning.
       h. Removed the dead `ma_et` bookkeeping in MINE (it forced a GPU sync per step).
       i. Fine-tune the defender also gets the blend family and alpha: disclose this.
          Use `BlendedTriggerOperator(alpha=...)` with a MISMATCHED alpha as an
          additional robustness test.

IMPORTANT (unchanged):
  - This is the project's BAERASER-lite adaptation, not a claim of exact
    reproduction of the original BAERASER paper implementation. It performs
    genuine generative trigger recovery (Generator + MINE) followed by
    explicit ASR-driven candidate selection and a dynamic-penalty unlearning
    objective — this is heavier than the "skip generative reconstruction"
    surrogate originally sketched in 04_DEFENSE_METHODS.md. If you keep this
    implementation, update that doc's description and its reviewer-safe
    sentence in 09_REFERENCES_AND_PAPER_FRAMING.md to match (see project
    discussion) rather than leaving the "full generative trigger
    reconstruction is left as future work" line in place — it's no longer
    an accurate caveat for this code.
  - The Blended and Silent Killer trigger-recovery configurations are
    empirical extensions of the BadNets notebook implementation and must be
    validated (check acceptance rate per attack before trusting downstream
    unlearning results — see recover_trigger_pool's returned
    `all_candidates` log).
  - Every defense x attack matrix cell must use recovered triggers, not
    oracle_trigger. oracle_trigger results are a separate, clearly labeled
    diagnostic subsection, never substituted into the main matrix or into
    the controller-vs-baselines-vs-oracle table (that table's "Oracle" row
    means best-performing-defense-per-cell, a different oracle entirely —
    do not conflate the two).

Expected input images inside the defense loaders:
  normalized CIFAR-10 tensors [B,3,32,32], using CIFAR_MEAN/CIFAR_STD.
Recovered trigger tensors are stored in RAW [0,1] RGB space.
"""

import copy
import gc
import hashlib
import math
import time
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 2027
TARGET_CLASS = 0

# Shared BaEraser-lite recovery settings (fallback defaults only — real
# per-attack budgets live in ATTACK_RECOVERY_CONFIG below).
BAERASER_G_INPUT_DIM = 64
BAERASER_G_HIDDEN = 2048
BAERASER_TRIGGER_LR = 2e-4
BAERASER_TRIGGER_BETAS = (0.5, 0.999)
BAERASER_ETA = 0.1
BAERASER_RECOVERY_ASR_THRESHOLD = 0.80

# Shared unlearning settings.
BAERASER_ALPHA = 1.0
BAERASER_BETA = 1.0
BAERASER_UNLEARN_LR = 1e-4
BAERASER_UNLEARN_MOMENTUM = 0.9
BAERASER_MAX_UNLEARNING_EPOCHS = 15
BAERASER_ASR_STOP_THRESHOLD = 0.01
CA_DROP_TOLERANCE = 0.05


# ---------------------------------------------------------------------------
# Per-attack recovery budgets (auditable, not magic numbers) --------------
# ---------------------------------------------------------------------------
# epsilons:        explicit coarse-pass epsilon values to try FIRST. If the
#                  coarse pass already yields `min_accepted_to_stop` accepted
#                  candidates, recovery stops there — the remaining
#                  `fine_epsilons` are only used if the coarse pass wasn't
#                  conclusive (see recover_trigger_pool).
# fine_epsilons:   extra epsilon values tried only if the coarse pass fails
#                  to produce enough accepted candidates.
# epochs:          generator/MINE training epochs per epsilon.
# n_candidates:    concrete triggers sampled from the generator per epsilon.
# min_accepted_to_stop: stop searching once this many triggers are accepted
#                  across all epsilons tried so far.
#
# Rationale for the starting values below: BadNets' trigger is small
# (3x4x4=48 values) and empirically recovers fast/cleanly, so it gets the
# smallest budget. Blended and Silent Killer search a full 3x32x32=3072
# value space and are unvalidated in the literature for BAERASER-style
# recovery — they get a larger coarse pass and access to a fine pass, but
# NOT the original 10-epsilon x 10-epoch x 64-candidate budget for every
# checkpoint; that exhaustive budget was compute-prohibitive across the full
# 3-attack x 3-rate matrix. Revisit these numbers after the first pass by
# checking the acceptance-rate logs, and document any change made here in
# 04_DEFENSE_METHODS.md.
ATTACK_RECOVERY_CONFIG = {
    "badnets": dict(
        epsilons=[0.3, 0.6, 0.9],
        fine_epsilons=[0.1, 0.45, 0.75],
        epochs=5,
        n_candidates=24,
        min_accepted_to_stop=2,
    ),
    "blended": dict(
        epsilons=[0.2, 0.5, 0.8],
        fine_epsilons=[0.35, 0.65, 0.95, 0.05],
        epochs=6,
        n_candidates=24,
        min_accepted_to_stop=2,
    ),
    "silent_killer": dict(
        epsilons=[0.2, 0.5, 0.8],
        fine_epsilons=[0.35, 0.65, 0.95, 0.05],
        epochs=6,
        n_candidates=24,
        min_accepted_to_stop=2,
    ),
}


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Attack-specific trigger operators
# ---------------------------------------------------------------------------

class TriggerOperator:
    """Interface used by the shared recovery/evaluation/unlearning code."""

    name = "base"

    def apply(self, images, trigger):
        raise NotImplementedError

    def trigger_shape(self):
        raise NotImplementedError

    def trigger_to_image(self, trigger):
        """Return a displayable [C,H,W] tensor in [0,1] when possible."""
        return trigger.detach().cpu()

    def metadata(self):
        return {"trigger_type": self.name}

    def recovery_key(self):
        """Key into ATTACK_RECOVERY_CONFIG. Override if name != config key."""
        return self.name


class BadNetsTriggerOperator(TriggerOperator):
    """4x4 localized BadNets patch."""

    name = "badnets"

    def __init__(self, size=4, placement="random"):
        self.size = size
        self.placement = placement

    def trigger_shape(self):
        return (3, self.size, self.size)

    def apply(self, images, trigger):
        # images are normalized; trigger is raw [0,1] RGB.
        B, C, H, W = images.shape
        if trigger.dim() == 1:
            trigger = trigger.view(1, *self.trigger_shape())
        elif trigger.dim() == 3:
            trigger = trigger.unsqueeze(0)
        trigger = trigger.to(images.device)
        mean = torch.tensor(CIFAR_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD, device=images.device).view(1, 3, 1, 1)
        trigger_norm = (trigger - mean) / std

        result = images.clone()
        for i in range(B):
            if self.placement == "random":
                y = torch.randint(0, H - self.size + 1, (1,), device=images.device).item()
                x = torch.randint(0, W - self.size + 1, (1,), device=images.device).item()
            else:
                y, x = H - self.size, W - self.size
            result[i, :, y:y+self.size, x:x+self.size] = trigger_norm[
                i if trigger_norm.shape[0] > 1 else 0
            ]
        return result

    def metadata(self):
        return {
            "trigger_type": self.name,
            "trigger_height": self.size,
            "trigger_width": self.size,
            "placement": self.placement,
        }


class BlendedTriggerOperator(TriggerOperator):
    """Full-image Blended trigger: x'=(1-alpha)x+alpha*k.

    `alpha` is REQUIRED (no default) — it must match the exact alpha_train
    used to poison the checkpoint you're defending
    (docs/02_ATTACKS_AND_DATASETS.md fixes alpha=0.1 for Blended; the
    previous version of this file silently defaulted to 0.15, which would
    have quietly mismatched every Blended run — see project discussion).
    """

    name = "blended"

    def __init__(self, alpha, image_shape=(3, 32, 32)):
        self.alpha = alpha
        self.image_shape = image_shape

    def trigger_shape(self):
        return self.image_shape

    def apply(self, images, trigger):
        if trigger.dim() == 3:
            trigger = trigger.unsqueeze(0)
        trigger = trigger.to(images.device)
        mean = torch.tensor(CIFAR_MEAN, device=images.device).view(1, 3, 1, 1)
        std  = torch.tensor(CIFAR_STD,  device=images.device).view(1, 3, 1, 1)
        return (1.0 - self.alpha) * images + self.alpha * ((trigger - mean) / std)

    def metadata(self):
        return {
            "trigger_type": self.name,
            "trigger_shape": list(self.image_shape),
            "blend_alpha": self.alpha,
        }


class SilentKillerTriggerOperator(TriggerOperator):
    """Silent Killer additive full-image trigger: x'=clip(x+delta).

    `trigger` is interpreted as a RAW-space additive perturbation. For
    normalized inputs, the perturbation is converted to normalized units
    using CIFAR std before addition.
    """

    name = "silent_killer"

    def __init__(self, image_shape=(3, 32, 32), clip_raw=True):
        self.image_shape = image_shape
        self.clip_raw = clip_raw

    def trigger_shape(self):
        return self.image_shape

    def apply(self, images, trigger):
        if trigger.dim() == 3:
            trigger = trigger.unsqueeze(0)
        trigger = trigger.to(images.device)

        # Interpret recovered trigger as a RAW-space delta around zero.
        std = torch.tensor(CIFAR_STD, device=images.device).view(1, 3, 1, 1)
        delta_norm = trigger / std
        out = images + delta_norm

        if self.clip_raw:
            mean = torch.tensor(CIFAR_MEAN, device=images.device).view(1, 3, 1, 1)
            lo = (0.0 - mean) / std
            hi = (1.0 - mean) / std
            out = torch.maximum(torch.minimum(out, hi), lo)
        return out

    def metadata(self):
        return {
            "trigger_type": self.name,
            "trigger_shape": list(self.image_shape),
            "clip_raw": self.clip_raw,
        }


# ---------------------------------------------------------------------------
# Generator + MINE
# ---------------------------------------------------------------------------

class BaEraserGenerator(nn.Module):
    """Max-entropy staircase-approximator generator used by the project."""

    def __init__(self, out_size, in_size=BAERASER_G_INPUT_DIM,
                 hidden_size=BAERASER_G_HIDDEN):
        super().__init__()
        self.in_size = in_size
        self.skip_size = in_size // 4
        self.out_size = out_size

        h = self.skip_size
        self.fc1 = nn.Linear(h, hidden_size)
        self.fc2 = nn.Linear(hidden_size + h, hidden_size)
        self.fc3 = nn.Linear(hidden_size + h, hidden_size)
        self.fc4 = nn.Linear(hidden_size + h, out_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

    def forward(self, z):
        h = self.skip_size
        x = F.leaky_relu(self.bn1(self.fc1(z[:, :h])), 0.2)
        x = F.leaky_relu(self.bn2(
            self.fc2(torch.cat([x, z[:, h:2*h]], dim=1))
        ), 0.2)
        x = F.leaky_relu(self.bn3(
            self.fc3(torch.cat([x, z[:, 2*h:3*h]], dim=1))
        ), 0.2)
        return torch.sigmoid(
            self.fc4(torch.cat([x, z[:, 3*h:4*h]], dim=1))
        )

    def gen_noise(self, num, device):
        return torch.rand(num, self.in_size, device=device)


class BaEraserMINE(nn.Module):
    """MINE statistics network for trigger recovery."""

    def __init__(self, y_size, x_size=BAERASER_G_INPUT_DIM,
                 hidden_size=BAERASER_G_HIDDEN):
        super().__init__()
        self.fc1_x = nn.Linear(x_size, hidden_size, bias=False)
        self.fc1_y = nn.Linear(y_size, hidden_size, bias=False)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x, y):
        h = self.fc1_x(x) + self.fc1_y(y) + self.fc1_bias
        h = F.leaky_relu(h, 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        return self.fc3(h)

    def mi(self, x, y, x_prime):
        t_joint = self.forward(x, y).mean()
        t_marginal = self.forward(x_prime, y)
        return t_joint - torch.log(torch.exp(t_marginal).mean() + 1e-8)


# ---------------------------------------------------------------------------
# Defender-side selection helpers (no test data) ---------------------------
# ---------------------------------------------------------------------------

def make_select_split(dataset, n_select=500, batch_size=128, seed=SEED, num_workers=0):
    """Split the defense budget into (train_loader, select_loader).

    Same total budget (e.g. 2,000 + 500 = 2,500); the selection slice is used ONLY
    for choosing epochs / accepting triggers. Use the same split for every defense
    you compare so they get identical information.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    sel_idx, tr_idx = perm[:n_select], perm[n_select:]
    train_loader = DataLoader(Subset(dataset, tr_idx), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers,
                              generator=torch.Generator().manual_seed(seed))
    select_loader = DataLoader(Subset(dataset, sel_idx), batch_size=batch_size, shuffle=False,
                               num_workers=num_workers)
    return train_loader, select_loader


@torch.no_grad()
def _clean_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def _proxy_asr(model, loader, trigger_pool, operator, target_label, device):
    """Worst-case (max over pool) target-hit rate of the RECOVERED triggers on
    held-out NON-target defense images. Defender-computable stand-in for ASR.
    Caveat: the model is trained against these triggers, so this proxy can reach ~0
    even if the real trigger still works; always check real ASR as a diagnostic."""
    model.eval()
    per_trigger = []
    for entry in trigger_pool:
        trig = entry["tensor"].to(device)
        hit = tot = 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            keep = lbls != target_label
            if keep.sum().item() == 0:
                continue
            out = operator.apply(imgs[keep], trig)
            hit += (model(out).argmax(1) == target_label).sum().item()
            tot += int(keep.sum().item())
        per_trigger.append(hit / max(tot, 1))
    return max(per_trigger) if per_trigger else 1.0


def state_hash(model_or_state):
    """Stable fingerprint of weights + buffers (16 hex chars)."""
    sd = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    h = hashlib.sha256()
    for k, v in sorted(sd.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def split_test_set(dataset, n_val=2500, seed=SEED):
    """Stratified, disjoint split of the CIFAR TEST set -> (val_idx, report_idx).

    WHY: the poisoned checkpoints were trained on all 50k training images, so a defense
    budget drawn from train overlaps with what the model memorised. Drawing it from the
    test set's `val_idx` half (and reporting only on `report_idx`) is the cheap fix.
    Build the defense budget AND the select slice from Subset(test, val_idx); build the
    ASR set and the CA monitor from Subset(test, report_idx). Use the same split for
    every defense (FT, ANP, NAD, BaEraser)."""
    labels = getattr(dataset, "targets", None)
    if labels is None:
        labels = [int(dataset[i][1]) for i in range(len(dataset))]
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    val_idx = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        val_idx.extend(idx[: int(round(n_val * len(idx) / len(labels)))].tolist())
    val_idx = sorted(val_idx)
    vs = set(val_idx)
    report_idx = [i for i in range(len(labels)) if i not in vs]
    assert_disjoint(val_idx, report_idx, "val", "report")
    return val_idx, report_idx


def assert_disjoint(idx_a, idx_b, name_a="A", name_b="B"):
    """Raise if two index collections share any element."""
    inter = set(map(int, idx_a)) & set(map(int, idx_b))
    if inter:
        raise AssertionError(f"{name_a} and {name_b} overlap on {len(inter)} indices")


def assert_baseline_ca(model, loader, expected_ca, tol=0.001, device=DEVICE, name="model"):
    """Check a loaded checkpoint reproduces its recorded clean accuracy.

    A checkpoint does not carry its normalisation constants, so this is the real test
    that CIFAR_MEAN/CIFAR_STD match the ones used in training. Run it on the FULL test
    loader (the recorded baselines were measured on 10k images). `expected_ca` in [0,1],
    e.g. 0.9483. Default tolerance 0.1 pp."""
    ca = _clean_acc(model, loader, device)
    if abs(ca - expected_ca) > tol:
        raise AssertionError(
            f"{name}: CA {ca*100:.2f}% != recorded {expected_ca*100:.2f}% "
            f"(tol {tol*100:.2f}pp) -> normalisation / checkpoint mismatch."
        )
    return ca


# ---------------------------------------------------------------------------
# Trigger recovery
# ---------------------------------------------------------------------------

def recover_trigger_candidate(
    victim_model,
    defense_loader,
    target_label,
    epsilon,
    operator,
    device=DEVICE,
    epochs=5,
    lr=BAERASER_TRIGGER_LR,
    betas=BAERASER_TRIGGER_BETAS,
    eta=BAERASER_ETA,
    warm_start_state=None,
    exclude_target=True,
):
    """Recover one candidate generator for the supplied trigger operator.

    `exclude_target` (v3, default True): target-class images are dropped from every
    batch, so the hinge asks "does this trigger CAUSE the target class on images that
    are not already the target class?" (v2 counted target-class images, which satisfy
    the hinge trivially).

    `warm_start_state`: optional (G_state_dict, M_state_dict) pair used as the INITIAL
    weights. Off in the main matrix (see recover_trigger_pool). The initial-weight
    hashes are attached as G._init_hash / M._init_hash so callers can verify that a
    candidate really started fresh.

    NOTE: for Blended and Silent Killer this is an empirical extension of the original
    BadNets-oriented notebook implementation. Full-image recovery has a much larger
    search space (3072 values on CIFAR-10).
    """
    out_size = int(np.prod(operator.trigger_shape()))
    G = BaEraserGenerator(out_size=out_size).to(device)
    M = BaEraserMINE(y_size=out_size).to(device)

    if warm_start_state is not None:
        g_state, m_state = warm_start_state
        try:
            G.load_state_dict(g_state)
            M.load_state_dict(m_state)
        except RuntimeError as e:
            warnings.warn(
                f"warm_start_state incompatible with this operator's trigger "
                f"shape, falling back to random init: {e}"
            )
    G._init_hash = state_hash(G)
    M._init_hash = state_hash(M)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_M = optim.Adam(M.parameters(), lr=lr, betas=betas)

    victim_model.eval()
    old_flags = {id(p): p.requires_grad for p in victim_model.parameters()}
    for p in victim_model.parameters():
        p.requires_grad_(False)

    logs = []

    for epoch in range(1, epochs + 1):
        G.train()
        M.train()
        sums = {"hinge": 0.0, "mi": 0.0, "prob": 0.0}
        n_batches = 0

        for imgs, lbls in defense_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            if exclude_target:
                imgs = imgs[lbls != target_label]
            B = imgs.size(0)
            if B < 2:                      # BatchNorm1d in G needs >1 sample
                continue

            z = G.gen_noise(B, device)
            trig_flat = G(z)
            trig = trig_flat.view(B, *operator.trigger_shape())

            # Small diversity noise, retained from the project implementation.
            trig_noisy = (trig + 0.01 * torch.randn_like(trig)).clamp(0, 1)
            triggered = operator.apply(imgs, trig_noisy)

            probs = F.softmax(victim_model(triggered), dim=1)
            target_prob = probs[:, target_label].mean()

            hinge = F.relu(epsilon - probs[:, target_label]).mean()

            z_prime = G.gen_noise(B, device)
            mi_estimate = M.mi(z, trig_flat, z_prime)
            g_loss = hinge - eta * mi_estimate

            opt_G.zero_grad()
            g_loss.backward(retain_graph=True)
            opt_G.step()

            z2 = G.gen_noise(B, device)
            trig2 = G(z2)
            z2_prime = G.gen_noise(B, device)
            mi_loss = -M.mi(z2, trig2.detach(), z2_prime)

            opt_M.zero_grad()
            mi_loss.backward()
            opt_M.step()

            sums["hinge"] += hinge.item()
            sums["mi"] += mi_estimate.item()
            sums["prob"] += target_prob.item()
            n_batches += 1

        logs.append({
            "epoch": epoch,
            "hinge_loss": sums["hinge"]/max(n_batches,1),
            "mi_estimate": sums["mi"]/max(n_batches,1),
            "target_prob": sums["prob"]/max(n_batches,1),
        })

    for p in victim_model.parameters():
        p.requires_grad_(old_flags[id(p)])

    return G, M, logs


@torch.no_grad()
def evaluate_exact_trigger_asr(
    victim_model,
    trigger_tensor,
    defense_loader,
    target_label,
    operator,
    device=DEVICE,
    n_eval_batches=5,
):
    """Evaluate a concrete trigger under the supplied attack operator."""
    victim_model.eval()
    trigger_tensor = trigger_tensor.to(device)
    total_target = 0
    total = 0

    for batch_idx, (imgs, lbls) in enumerate(defense_loader):
        if batch_idx >= n_eval_batches:
            break
        imgs, lbls = imgs.to(device), lbls.to(device)
        keep = lbls != target_label            # ASR is defined on NON-target images
        if keep.sum().item() == 0:
            continue
        triggered = operator.apply(imgs[keep], trigger_tensor)
        preds = victim_model(triggered).argmax(1)
        total_target += (preds == target_label).sum().item()
        total += int(keep.sum().item())

    return total_target / max(total, 1)


def recover_trigger_pool(
    victim_model,
    defense_loader,
    target_label,
    operator,
    device=DEVICE,
    recovery_asr_threshold=None,
    seed=SEED,
    verbose=True,
    warm_start_state=None,
    config_override=None,
    heldout_loader=None,
    chain_epsilons=False,
    return_states=False,
):
    """Recover and rank exact trigger candidates for one victim model.

    `heldout_loader`: optional held-out defense slice. If given, candidate ASR /
    acceptance is measured there instead of on the images the generator trained on.

    INITIALISATION (v3): every epsilon starts from a FRESH random init. v2 silently
    carried the previous epsilon's G/M weights into the next one even with
    warm_start_state=None, so epsilons were not independent. Now:
      chain_epsilons=False (default): fresh init per epsilon. If `warm_start_state`
          is passed it is applied identically to EVERY epsilon (explicit opt-in;
          keep it off in the main matrix).
      chain_epsilons=True: legacy chained behaviour (ablation only).
    Initial/final weight hashes are logged per candidate; with fresh random inits
    recovery_meta["fresh_init_verified"] is True iff every init hash is distinct and
    differs from every earlier run's final weights.

    COARSE-TO-FINE WITH EARLY EXIT: tries `epsilons` (the coarse pass) first. If
    `min_accepted_to_stop` triggers are accepted, recovery stops there; `fine_epsilons`
    are only consulted if the coarse pass wasn't conclusive.

    Per-attack budgets come from ATTACK_RECOVERY_CONFIG (keyed by
    `operator.recovery_key()`) unless `config_override` is supplied.

    Returns: (trigger_pool, all_candidates, recovery_meta)
    """
    cfg = config_override or ATTACK_RECOVERY_CONFIG.get(operator.recovery_key())
    if cfg is None:
        raise ValueError(
            f"No recovery config for '{operator.recovery_key()}' — add an "
            f"entry to ATTACK_RECOVERY_CONFIG or pass config_override."
        )
    threshold = (
        recovery_asr_threshold
        if recovery_asr_threshold is not None
        else BAERASER_RECOVERY_ASR_THRESHOLD
    )

    trigger_pool = []
    all_candidates = []
    epsilons_tried = []
    t_start = time.time()
    eval_loader = heldout_loader if heldout_loader is not None else defense_loader
    eval_split = "heldout" if heldout_loader is not None else "train"

    if chain_epsilons:
        init_mode = "chained"
    elif warm_start_state is not None:
        init_mode = "warm_fixed"
    else:
        init_mode = "random"
    if verbose and init_mode != "random":
        print(f"  NOTE: recovery init mode = {init_mode} (main matrix should use 'random').")

    chained_ws = warm_start_state if chain_epsilons else None
    fixed_ws = None if chain_epsilons else warm_start_state
    last_g_state, last_m_state = None, None
    init_hashes, final_hashes = [], set()

    def _run_epsilon_batch(epsilon_list, phase):
        nonlocal chained_ws, last_g_state, last_m_state
        for eps_val in epsilon_list:
            set_seed(seed + len(epsilons_tried) + target_label * 100)
            epsilons_tried.append(eps_val)

            if verbose:
                print(f"  [{phase}] recovery epsilon={eps_val:.2f}")

            ws = chained_ws if chain_epsilons else fixed_ws
            G, M, logs = recover_trigger_candidate(
                victim_model, defense_loader, target_label, eps_val,
                operator, device=device,
                epochs=cfg["epochs"],
                warm_start_state=ws,
            )
            init_g, init_m = G._init_hash, M._init_hash
            init_hashes.append((init_g, init_m))
            final_g = state_hash(G)
            final_hashes.add(final_g)

            if chain_epsilons or return_states:
                last_g_state = {k: v.detach().clone() for k, v in G.state_dict().items()}
                last_m_state = {k: v.detach().clone() for k, v in M.state_dict().items()}
                if chain_epsilons:
                    chained_ws = (last_g_state, last_m_state)

            G.eval()
            with torch.no_grad():
                z = G.gen_noise(cfg["n_candidates"], device)
                candidates = G(z)

            best_asr = -1.0
            best_tensor = None
            for c in range(cfg["n_candidates"]):
                candidate = candidates[c].view(*operator.trigger_shape())
                asr = evaluate_exact_trigger_asr(
                    victim_model, candidate, eval_loader,
                    target_label, operator, device=device
                )
                if asr > best_asr:
                    best_asr = asr
                    best_tensor = candidate.detach().cpu()

            accepted = best_asr >= threshold
            info = {
                "target_label": target_label,
                "epsilon": float(eps_val),
                "candidate_asr": float(best_asr),
                "accepted": bool(accepted),
                "phase": phase,
                "eval_split": eval_split,
                "init_mode": init_mode,
                "init_hash_G": init_g,
                "init_hash_M": init_m,
                "final_hash_G": final_g,
                **operator.metadata(),
            }
            all_candidates.append(info)

            if verbose:
                print(
                    f"    exact ASR={best_asr*100:.2f}% "
                    f"[{'ACCEPTED' if accepted else 'rejected'}]"
                )

            if accepted:
                trigger_pool.append({
                    "tensor": best_tensor,
                    "target_label": target_label,
                    "epsilon": float(eps_val),
                    "asr": float(best_asr),
                    "logs": logs,
                })

            del G, M
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if len(trigger_pool) >= cfg["min_accepted_to_stop"]:
                if verbose:
                    print(
                        f"    -> {len(trigger_pool)} accepted triggers, "
                        f"stopping recovery early."
                    )
                return True  # early exit
        return False

    stopped_early = _run_epsilon_batch(cfg["epsilons"], "coarse")
    if not stopped_early and cfg.get("fine_epsilons"):
        if verbose:
            print("  Coarse pass inconclusive — trying fine epsilons.")
        _run_epsilon_batch(cfg["fine_epsilons"], "fine")

    fresh_verified = None
    if init_mode == "random":
        g_inits = [g for g, _ in init_hashes]
        fresh_verified = (len(set(g_inits)) == len(g_inits)
                          and not (set(g_inits) & final_hashes))
        if not fresh_verified:
            warnings.warn("recover_trigger_pool: initial generator weights repeated or equal "
                          "to an earlier run's final weights -> epsilons are NOT independent.")

    recovery_meta = {
        "attack": operator.recovery_key(),
        "epsilons_tried": epsilons_tried,
        "n_epsilons_tried": len(epsilons_tried),
        "n_accepted": len(trigger_pool),
        "wall_clock_seconds": time.time() - t_start,
        "init_mode": init_mode,
        "init_hashes": init_hashes,
        "fresh_init_verified": fresh_verified,
        "eval_split": eval_split,
        # only populated with return_states=True / chain_epsilons=True (warm-start use):
        "final_generator_state": last_g_state,
        "final_mine_state": last_m_state,
    }

    if not trigger_pool and verbose:
        print(
            f"  WARNING: recovery accepted 0 triggers for "
            f"'{operator.recovery_key()}' after trying "
            f"{len(epsilons_tried)} epsilons. Log this as a failure case — "
            f"do not silently retry with a lower threshold without "
            f"documenting the change."
        )

    return trigger_pool, all_candidates, recovery_meta


# ---------------------------------------------------------------------------
# Generality / control checks (defender-side or diagnostic) -----------------
# ---------------------------------------------------------------------------

def fresh_recovery_audit(
    model,
    defense_loader,
    eval_loader,
    target_label,
    operator,
    device=DEVICE,
    seed=SEED + 9000,
    config_override=None,
    threshold=None,
    verbose=False,
):
    """Run a FRESH, reduced-budget trigger recovery against `model` (typically the
    sanitized model) and report how well the freshly recovered triggers work.

    Defender-computable (uses only the defense budget). `eval_loader` should be held
    out from `defense_loader`. Use it (a) as the tie-break inside baeraser_unlearning
    (fresh_select_k>1) and (b) as the headline "did the sanitized model just become
    vulnerable to new triggers?" audit. Compare against the same audit on the
    poisoned model and on a CLEAN model: a clean model that also accepts triggers
    means the recovery finds generic universal perturbations, not the backdoor.

    Returns dict(max_asr, n_accepted, n_tried, per_epsilon, seconds).
    """
    base = ATTACK_RECOVERY_CONFIG.get(operator.recovery_key(), {})
    cfg = config_override or dict(
        epsilons=list(base.get("epsilons", [0.3, 0.6]))[:2],
        fine_epsilons=[],
        epochs=base.get("epochs", 5),
        n_candidates=base.get("n_candidates", 24),
        min_accepted_to_stop=10 ** 9,        # never stop early: try every epsilon
    )
    t0 = time.time()
    pool, cands, meta = recover_trigger_pool(
        model, defense_loader, target_label, operator, device=device,
        recovery_asr_threshold=threshold, seed=seed, verbose=verbose,
        config_override=cfg, heldout_loader=eval_loader,
    )
    return {
        "max_asr": max((c["candidate_asr"] for c in cands), default=0.0),
        "n_accepted": len(pool),
        "n_tried": len(cands),
        "per_epsilon": [(c["epsilon"], c["candidate_asr"]) for c in cands],
        "seconds": time.time() - t0,
    }


def universal_adv_vulnerability(
    model,
    fit_loader,
    eval_loader,
    target_label,
    device=DEVICE,
    eps=8 / 255,
    step=2 / 255,
    epochs=3,
    seed=SEED + 7,
):
    """Targeted L_inf universal perturbation (raw-pixel space, budget `eps`), fitted on
    `fit_loader` non-target images, scored on `eval_loader` non-target images.

    Run on the poisoned model, the sanitized model and a clean model. If sanitizing
    drops this number too, part of the BaEraser gain is generic adversarial
    robustness (adversarial-training-like), not backdoor removal."""
    op = SilentKillerTriggerOperator(clip_raw=True)
    set_seed(seed)
    model.eval()
    flags = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)

    delta = torch.zeros(1, *op.trigger_shape(), device=device)
    for _ in range(epochs):
        for x, y in fit_loader:
            x, y = x.to(device), y.to(device)
            keep = y != target_label
            if keep.sum().item() == 0:
                continue
            x = x[keep]
            d = delta.clone().requires_grad_(True)
            with torch.enable_grad():
                tgt = torch.full((x.size(0),), target_label, dtype=torch.long, device=device)
                loss = F.cross_entropy(model(op.apply(x, d)), tgt)
                (g,) = torch.autograd.grad(loss, d)
            delta = (delta - step * g.sign()).clamp(-eps, eps).detach()

    for p in model.parameters():
        p.requires_grad_(flags[id(p)])

    asr = evaluate_exact_trigger_asr(
        model, delta, eval_loader, target_label, op, device=device, n_eval_batches=10 ** 9
    )
    return {"uap_asr": float(asr), "eps": float(eps), "delta_linf": float(delta.abs().max().item())}


# ---------------------------------------------------------------------------
# Shared unlearning
# ---------------------------------------------------------------------------

def compute_omega(model, clean_loader, criterion, device=DEVICE):
    """Clean-data gradient sensitivity used by the dynamic penalty."""
    was_training = model.training
    model.eval()

    omega = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            omega[name] = torch.zeros_like(param, device=device)

    count = 0
    for imgs, lbls in clean_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        model.zero_grad(set_to_none=True)
        loss = criterion(model(imgs), lbls)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                omega[name] += param.grad.abs()
        count += 1

    for name in omega:
        omega[name] = (omega[name] / max(count, 1)).detach()

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return omega


def make_triggered_batch(
    clean_imgs,
    trigger_pool,
    target_label,
    operator,
    device=DEVICE,
    true_labels=None,
    exclude_target=True,
    label_mode="target",
):
    """Apply one randomly chosen recovered trigger to a batch.

    true_labels + exclude_target (v3): target-class images are dropped first (a
    "triggered" target-class image carries no backdoor information). Returns
    (None, None) if nothing is left.
    label_mode: "target" -> labels are the target class (backdoor unlearning term is
    ascent on CE(triggered, target)); "true" -> the images' true labels (ablation:
    minimise CE(triggered, true label))."""
    if not trigger_pool:
        raise ValueError("Empty trigger pool.")

    if exclude_target and true_labels is not None:
        keep = true_labels != target_label
        clean_imgs = clean_imgs[keep]
        true_labels = true_labels[keep]
    B = clean_imgs.shape[0]
    if B == 0:
        return None, None

    entry = trigger_pool[np.random.randint(0, len(trigger_pool))]
    trigger = entry["tensor"].to(device)

    if trigger.dim() == 3:
        trigger = trigger.unsqueeze(0)
    trigger = trigger.expand(B, -1, -1, -1)

    triggered = operator.apply(clean_imgs, trigger)
    if label_mode == "true":
        if true_labels is None:
            raise ValueError("label_mode='true' needs true_labels")
        labels = true_labels
    else:
        labels = torch.full((B,), target_label, dtype=torch.long, device=device)
    return triggered, labels


def _bucket_key(asr, ca, tol):
    """Sort key: ASR bucket (ties inside `tol` count as equal), then higher CA."""
    b = math.floor(asr / tol) if tol and tol > 0 else asr
    return (b, -(ca if ca is not None else 0.0))


def baeraser_unlearning(
    model,
    defense_loader,
    trigger_pool,
    operator,
    target_label,
    calculate_ca=None,
    ca_loader=None,
    calculate_asr=None,
    asr_loader=None,
    device=DEVICE,
    alpha=BAERASER_ALPHA,
    beta=BAERASER_BETA,
    lr=BAERASER_UNLEARN_LR,
    momentum=BAERASER_UNLEARN_MOMENTUM,
    max_epochs=BAERASER_MAX_UNLEARNING_EPOCHS,
    ca_before=None,
    asr_before=None,
    ca_drop_tolerance=CA_DROP_TOLERANCE,
    asr_stop_threshold=None,
    omega_update_freq=3,
    select_loader=None,
    freeze_bn=False,
    exclude_target=True,
    trigger_loss_mode="target_ascent",
    trigger_ascent_cap=None,
    asr_tie_tol=0.005,
    fresh_select_k=1,
    fresh_audit_kwargs=None,
    allow_leaky_selection=False,
    restore_best=True,
    seed=SEED,
):
    """Shared BaEraser-lite dynamic-penalty unlearning (v3).

    Objective (default, trigger_loss_mode="target_ascent"):
        alpha * (clean_loss - trigger_loss) + beta * penalty,
        penalty = sum_k omega_k * |theta_k - theta_0,k|
    where trigger_loss = CE(model(triggered non-target images), target). This term is
    an UNBOUNDED ascent; gradient clipping is the only brake. Options:
      trigger_ascent_cap=c : ascent stops (zero gradient) once trigger_loss > c.
      trigger_loss_mode="true_label": trigger term becomes + CE(model(triggered),
          TRUE label), i.e. the model is trained to ignore the trigger. Bounded;
          report as an ablation.
      exclude_target=True  : target-class images are removed from the triggered batch.

    CHECKPOINT SELECTION (no oracle, no test data):
      1. Only epochs whose held-out clean CA (select_loader) is >= its pre-unlearning
         value - ca_drop_tolerance are eligible.
      2. Rank by proxy ASR of the recovered triggers on held-out non-target images;
         proxy values within `asr_tie_tol` tie, and ties go to higher held-out CA.
      3. If fresh_select_k > 1, the k best epochs are re-ranked by the ASR of FRESHLY
         recovered triggers on that epoch's weights (fresh_recovery_audit; pass
         `fresh_audit_kwargs` to change its budget). The proxy can be driven to ~0
         without removing the real backdoor; fresh triggers are much harder to game.
      calculate_ca/asr + ca_loader/asr_loader are MONITORING ONLY: logged every epoch
      and reported at the selected epoch (CA_after / ASR_after = the HEADLINE numbers).
      The epoch with the best real ASR is reported separately as oracle_* (an upper
      bound to be labelled as such; never a defense result).
    Early stopping is off unless asr_stop_threshold is given. It then fires on the
    PROXY ASR, which can be gamed; prefer running all epochs.

    select_loader is required unless allow_leaky_selection=True (legacy v1 behaviour:
    select on the monitoring/test loaders; result tagged LEAKY).

    The model is modified in place. With restore_best=True it ends holding the
    SELECTED weights (or the original weights if no epoch was eligible, status
    NO_VALID_CHECKPOINT); v2 left the last-epoch weights in it.

    freeze_bn: keep BatchNorm layers in eval mode while unlearning (ablation; default
    False).
    """
    if not trigger_pool:
        raise ValueError("Cannot unlearn without recovered triggers.")
    if trigger_loss_mode not in ("target_ascent", "true_label"):
        raise ValueError("trigger_loss_mode must be 'target_ascent' or 'true_label'")

    held_out = select_loader is not None
    if not held_out and not allow_leaky_selection:
        raise ValueError(
            "baeraser_unlearning needs select_loader (held-out defense slice). Selecting "
            "on the monitoring/test loaders is leaky; pass allow_leaky_selection=True "
            "only to reproduce v1 numbers."
        )

    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    theta0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    if held_out:
        selection_source = "defense_holdout"
        ca_min = _clean_acc(model, select_loader, device) - ca_drop_tolerance
    else:
        selection_source = "test_monitor (LEAKY)"
        warnings.warn(
            "baeraser_unlearning: allow_leaky_selection=True -> best epoch / early stop / "
            "CA floor are chosen on the monitoring (test) loaders. Reported CA/ASR are "
            "selection-biased."
        )
        ca_min = None if ca_before is None else ca_before - ca_drop_tolerance

    k_keep = max(1, int(fresh_select_k)) if held_out else 1
    logs = []
    cands = []          # eligible epochs kept for the final pick (state on CPU)
    omega = None

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        if omega is None or (epoch - 1) % omega_update_freq == 0:
            omega = compute_omega(model, defense_loader, criterion, device)

        model.train()
        if freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        sums = {"clean": 0.0, "trigger": 0.0, "penalty": 0.0, "total": 0.0}
        n_batches = 0

        for imgs, lbls in defense_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)

            clean_loss = criterion(model(imgs), lbls)

            triggered, trig_labels = make_triggered_batch(
                imgs, trigger_pool, target_label, operator, device,
                true_labels=lbls, exclude_target=exclude_target,
                label_mode="true" if trigger_loss_mode == "true_label" else "target",
            )
            if triggered is None:
                trigger_loss = torch.zeros((), device=device)
                trigger_term = torch.zeros((), device=device)
            else:
                trigger_loss = criterion(model(triggered), trig_labels)
                if trigger_loss_mode == "true_label":
                    trigger_term = trigger_loss
                else:
                    ascent = trigger_loss if trigger_ascent_cap is None else \
                        torch.clamp(trigger_loss, max=trigger_ascent_cap)
                    trigger_term = -ascent

            penalty = torch.zeros((), device=device)
            for name, param in model.named_parameters():
                if name in omega and name in theta0:
                    penalty = penalty + (
                        omega[name] * (param - theta0[name]).abs()
                    ).sum()

            total_loss = alpha * (clean_loss + trigger_term) + beta * penalty

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            sums["clean"] += clean_loss.item()
            sums["trigger"] += trigger_loss.item()
            sums["penalty"] += penalty.item()
            sums["total"] += total_loss.item()
            n_batches += 1
        train_time = time.time() - t0

        # --- defender-side selection signals (counted as defence cost) -------
        t1 = time.time()
        sel_ca = sel_asr = None
        if held_out:
            sel_ca = _clean_acc(model, select_loader, device)
            sel_asr = _proxy_asr(model, select_loader, trigger_pool, operator,
                                 target_label, device)
        sel_time = time.time() - t1

        # --- monitoring on test loaders (NOT counted as cost, NOT used if held_out)
        ca = asr = None
        if calculate_ca is not None and ca_loader is not None:
            ca = calculate_ca(model, ca_loader, device)
        if calculate_asr is not None and asr_loader is not None:
            asr = calculate_asr(model, asr_loader, target_label, device)

        param_dist = sum(
            (p - theta0[n]).norm().item()
            for n, p in model.named_parameters() if n in theta0
        )

        logs.append({
            "epoch": epoch,
            "clean_loss": sums["clean"] / max(n_batches, 1),
            "trigger_loss": sums["trigger"] / max(n_batches, 1),
            "penalty": sums["penalty"] / max(n_batches, 1),
            "total_loss": sums["total"] / max(n_batches, 1),
            "CA": ca,                  # monitoring (test/report)
            "ASR": asr,                # monitoring (test/report)  <- REAL ASR
            "sel_CA": sel_ca,          # selection signal (held-out defense data)
            "sel_proxy_ASR": sel_asr,  # selection signal (recovered-trigger proxy)
            "sel_fresh_ASR": None,     # filled for the finalists if fresh_select_k>1
            "param_dist": param_dist,
            "omega_refreshed": (epoch - 1) % omega_update_freq == 0,
            "time": train_time + sel_time,
        })

        crit_ca, crit_asr = (sel_ca, sel_asr) if held_out else (ca, asr)
        valid_ca = (crit_ca is None) or (ca_min is None) or (crit_ca >= ca_min)
        if valid_ca and crit_asr is not None:
            cands.append({
                "epoch": epoch, "crit_asr": crit_asr, "crit_ca": crit_ca,
                "mon_ca": ca, "mon_asr": asr, "fresh_asr": None,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            })
            cands.sort(key=lambda c: _bucket_key(c["crit_asr"], c["crit_ca"], asr_tie_tol))
            del cands[k_keep:]

        if asr_stop_threshold is not None and crit_asr is not None and valid_ca \
                and crit_asr < asr_stop_threshold:
            break

    # --- final pick: optional fresh-trigger re-rank of the finalists -----------
    audit_sec = 0.0
    chosen = cands[0] if cands else None
    if held_out and len(cands) > 1:
        probe = copy.deepcopy(model)
        ta = time.time()
        for c in cands:
            probe.load_state_dict({k: v.to(device) for k, v in c["state"].items()})
            aud = fresh_recovery_audit(
                probe, defense_loader, select_loader, target_label, operator,
                device=device, seed=seed + 9000 + c["epoch"], **(fresh_audit_kwargs or {}))
            c["fresh_asr"] = aud["max_asr"]
            logs[c["epoch"] - 1]["sel_fresh_ASR"] = aud["max_asr"]
        audit_sec = time.time() - ta
        del probe
        chosen = min(cands, key=lambda c: _bucket_key(c["fresh_asr"], c["crit_ca"], asr_tie_tol))

    # --- oracle row: best REAL ASR among CA-eligible epochs (upper bound only) --
    oracle = None
    for l in logs:
        if l["ASR"] is None or l["CA"] is None:
            continue
        if ca_before is not None and l["CA"] < ca_before - ca_drop_tolerance:
            continue
        if oracle is None or (l["ASR"], -l["CA"]) < (oracle["ASR"], -oracle["CA"]):
            oracle = l

    if restore_best:
        model.load_state_dict(
            {k: v.to(device) for k, v in (chosen["state"] if chosen else init_state).items()})
    model.eval()

    best_state = None if chosen is None else chosen["state"]
    return {
        "model": model,
        "best_state_dict": best_state,
        "logs": logs,
        "best_epoch": 0 if chosen is None else chosen["epoch"],
        # HEADLINE: real CA/ASR at the epoch chosen WITHOUT looking at them
        "CA_after": None if chosen is None else chosen["mon_ca"],
        "ASR_after": None if chosen is None else chosen["mon_asr"],
        # what selection actually saw:
        "sel_CA_after": None if chosen is None else chosen["crit_ca"],
        "sel_ASR_after": None if chosen is None else chosen["crit_asr"],
        "sel_fresh_ASR_after": None if chosen is None else chosen["fresh_asr"],
        "selection_source": selection_source,
        "selection_rule": ("heldout CA floor -> proxy ASR (tie_tol=%g)%s"
                           % (asr_tie_tol, " -> fresh-trigger ASR on top-%d" % k_keep
                              if (held_out and k_keep > 1) else "")
                           if held_out else "LEAKY test-monitor"),
        # ORACLE UPPER BOUND (uses real ASR; not a defense result):
        "oracle_best_epoch": None if oracle is None else oracle["epoch"],
        "oracle_CA": None if oracle is None else oracle["CA"],
        "oracle_ASR": None if oracle is None else oracle["ASR"],
        "CA_before": ca_before,
        "ASR_before": asr_before,
        "status": "SUCCESS" if chosen is not None else "NO_VALID_CHECKPOINT",
        "operator": operator.metadata(),
        "config": {
            "exclude_target": exclude_target, "trigger_loss_mode": trigger_loss_mode,
            "trigger_ascent_cap": trigger_ascent_cap, "freeze_bn": freeze_bn,
            "asr_stop_threshold": asr_stop_threshold, "fresh_select_k": k_keep,
            "asr_tie_tol": asr_tie_tol, "lr": lr, "alpha": alpha, "beta": beta,
            "max_epochs": max_epochs, "ca_drop_tolerance": ca_drop_tolerance,
        },
        "selection_audit_seconds": audit_sec,
        # unlearning + selection signals + fresh audits; EXCLUDES test monitoring and
        # recovery (baeraser_lite() adds recovery into total_cost_seconds):
        "total_compute_seconds": sum(l["time"] for l in logs) + audit_sec,
    }


# ---------------------------------------------------------------------------
# One-call defense API
# ---------------------------------------------------------------------------

@dataclass
class BaEraserResult:
    model: nn.Module
    trigger_pool: list
    recovery_candidates: list
    recovery_meta: Optional[dict]
    unlearning: dict
    mode: str  # "recovered" or "oracle_diagnostic"


def baeraser_lite(
    victim_model,
    defense_loader,
    operator,
    target_label=TARGET_CLASS,
    device=DEVICE,
    oracle_trigger=None,
    diagnostic_only=False,
    recovery_asr_threshold=None,
    warm_start_state=None,
    config_override=None,
    inplace=False,
    chain_epsilons=False,
    return_states=False,
    seed=SEED,
    **unlearning_kwargs,
):
    """Main API for the defense x attack matrix.

    DEPLOYABLE / MAIN-MATRIX PATH (use this for every cell of the defense x attack
    matrix and for anything feeding the controller-vs-baselines-vs-oracle comparison):

        train_loader, select_loader = make_select_split(val_subset)   # val half of test
        res = baeraser_lite(model, train_loader, BlendedTriggerOperator(alpha=ALPHA),
                            select_loader=select_loader,
                            calculate_ca=..., ca_loader=report_testloader,
                            calculate_asr=..., asr_loader=report_asr_loader,
                            ca_before=..., asr_before=...)
        res.unlearning["CA_after"], ["ASR_after"]   # HEADLINE (selected epoch)
        res.unlearning["oracle_ASR"]                # upper bound row, label it as such
        res.unlearning["total_cost_seconds"]        # recovery + unlearning + audits

    Defaults that matter (v3): fresh random init per epsilon (no warm start / chaining),
    no early stopping, target-class images excluded, select_loader REQUIRED,
    victim copied (inplace=False) and the returned model holds the SELECTED weights.

    CLEAN-MODEL CONTROL: call this exactly the same way on a clean checkpoint and
    compare with summarize_result(); see also fresh_recovery_audit and
    universal_adv_vulnerability.

    DIAGNOSTIC-ONLY PATH — oracle_trigger:
    This does NOT produce a BAERASER-lite result. It substitutes the attacker's actual,
    ground-truth trigger for the recovery stage, which no real defense has access to.
    Its only legitimate use is to separate "did recovery fail?" from "did unlearning
    fail?" in a separately labelled ablation — never in the main matrix, and never
    conflated with the best-defense "Oracle" row of the controller comparison.

        baeraser_lite(model, loader, BlendedTriggerOperator(alpha=0.1),
                      oracle_trigger=known_blended_pattern,
                      diagnostic_only=True, select_loader=select_loader)
    """
    if unlearning_kwargs.get("select_loader") is None and \
            not unlearning_kwargs.get("allow_leaky_selection", False):
        raise ValueError(
            "baeraser_lite needs select_loader=... (held-out defense slice, see "
            "make_select_split). Selecting on test data is leaky; pass "
            "allow_leaky_selection=True only to reproduce v1 numbers."
        )

    if not inplace:
        victim_model = copy.deepcopy(victim_model)

    if oracle_trigger is not None:
        if not diagnostic_only:
            raise ValueError(
                "oracle_trigger requires diagnostic_only=True. This path "
                "substitutes the attacker's real trigger and is NOT a "
                "deployable BAERASER-lite result — see the DIAGNOSTIC-ONLY "
                "docstring section above. Set diagnostic_only=True to "
                "acknowledge this is for the recovery-vs-unlearning "
                "ablation only, and label results accordingly."
            )
        print(
            "=" * 70 + "\n"
            "DIAGNOSTIC-ONLY RUN: using oracle_trigger (attacker's real "
            "trigger, ground truth).\n"
            "This is NOT a BAERASER-lite result and must not be reported "
            "in the main\n"
            "defense x attack matrix or the controller-vs-baselines-vs-"
            "oracle table.\n" + "=" * 70
        )
        trigger_pool = [{
            "tensor": oracle_trigger.detach().cpu(),
            "target_label": target_label,
            "epsilon": None,
            "asr": None,
            "oracle": True,
        }]
        recovery_candidates = []
        recovery_meta = None
        mode = "oracle_diagnostic"
    else:
        trigger_pool, recovery_candidates, recovery_meta = recover_trigger_pool(
            victim_model,
            defense_loader,
            target_label,
            operator,
            device=device,
            recovery_asr_threshold=recovery_asr_threshold,
            seed=seed,
            warm_start_state=warm_start_state,
            config_override=config_override,
            heldout_loader=unlearning_kwargs.get("select_loader"),
            chain_epsilons=chain_epsilons,
            return_states=return_states,
        )
        mode = "recovered"

    if not trigger_pool:
        result = {
            "model": victim_model, "status": "FAILED_NO_TRIGGERS", "best_epoch": 0,
            "CA_after": None, "ASR_after": None, "logs": [], "mode": mode,
            "total_compute_seconds": 0.0,
            "total_cost_seconds": (recovery_meta or {}).get("wall_clock_seconds", 0.0),
        }
        return BaEraserResult(model=victim_model, trigger_pool=[],
                              recovery_candidates=recovery_candidates,
                              recovery_meta=recovery_meta, unlearning=result, mode=mode)

    result = baeraser_unlearning(
        victim_model,
        defense_loader,
        trigger_pool,
        operator,
        target_label,
        device=device,
        seed=seed,
        **unlearning_kwargs,
    )
    result["mode"] = mode
    result["total_cost_seconds"] = (
        result["total_compute_seconds"]
        + (recovery_meta["wall_clock_seconds"] if recovery_meta else 0.0)
    )

    return BaEraserResult(
        model=result["model"],
        trigger_pool=trigger_pool,
        recovery_candidates=recovery_candidates,
        recovery_meta=recovery_meta,
        unlearning=result,
        mode=mode,
    )


def summarize_result(res):
    """Flat dict of a BaEraserResult for the main table / clean-model control table.

    Run the identical config on a backdoored model and on a CLEAN model and compare
    n_accepted / max_candidate_asr: if the clean model accepts triggers just as often,
    the recovery is finding generic universal perturbations, not the backdoor (reframe
    the method accordingly). Either way disclose that the defender is given the blend
    family and alpha."""
    u = res.unlearning
    rm = res.recovery_meta or {}
    cands = res.recovery_candidates or []
    return {
        "mode": res.mode,
        "status": u.get("status"),
        "n_accepted": rm.get("n_accepted", 0),
        "n_epsilons_tried": rm.get("n_epsilons_tried", 0),
        "max_candidate_asr": max((c["candidate_asr"] for c in cands), default=None),
        "accepted_eps": [round(t["epsilon"], 3) for t in res.trigger_pool if t.get("epsilon") is not None],
        "fresh_init_verified": rm.get("fresh_init_verified"),
        "selected_epoch": u.get("best_epoch"),
        "CA_before": u.get("CA_before"),
        "ASR_before": u.get("ASR_before"),
        "CA_after_selected": u.get("CA_after"),
        "ASR_after_selected": u.get("ASR_after"),          # headline
        "sel_proxy_ASR": u.get("sel_ASR_after"),
        "sel_fresh_ASR": u.get("sel_fresh_ASR_after"),
        "oracle_epoch": u.get("oracle_best_epoch"),        # upper bound, label as such
        "oracle_CA": u.get("oracle_CA"),
        "oracle_ASR": u.get("oracle_ASR"),
        "selection_rule": u.get("selection_rule"),
        "recovery_s": rm.get("wall_clock_seconds"),
        "unlearning_s": u.get("total_compute_seconds"),
        "total_cost_s": u.get("total_cost_seconds"),
    }


# ---------------------------------------------------------------------------
# Recommended protocol (v3)
# ---------------------------------------------------------------------------
#
#   val_idx, report_idx = split_test_set(testset, n_val=2500)
#   val_set, report_set = Subset(testset, val_idx), Subset(testset, report_idx)
#   train_loader, select_loader = make_select_split(val_set, n_select=500)
#   # CA monitor + ASR set: built from report_set ONLY.
#   assert_baseline_ca(model, FULL_testloader, 0.9483, name="pr01")   # before the split
#
#   res = baeraser_lite(model, train_loader, BlendedTriggerOperator(alpha=0.1),
#                       select_loader=select_loader, fresh_select_k=3,
#                       calculate_ca=calculate_ca, ca_loader=report_loader,
#                       calculate_asr=calculate_asr, asr_loader=report_asr_loader,
#                       ca_before=CA_before, asr_before=ASR_before)
#   row = summarize_result(res)
#
#   # same call on the CLEAN checkpoint -> control row
#   # audits: fresh_recovery_audit(...) and universal_adv_vulnerability(...) on
#   #         poisoned / sanitized / clean models
#   # ablations: trigger_loss_mode="true_label", trigger_ascent_cap=5.0, freeze_bn=True,
#   #            BlendedTriggerOperator(alpha=0.05 / 0.2) (mismatched alpha)