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

  3. Warm-starting: recovery for a given attack's 2nd/3rd poison
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

import gc
import time
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
        self.ma_et = None
        self.ma_rate = 0.001

    def forward(self, x, y):
        h = self.fc1_x(x) + self.fc1_y(y) + self.fc1_bias
        h = F.leaky_relu(h, 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        return self.fc3(h)

    def mi(self, x, y, x_prime):
        t_joint = self.forward(x, y).mean()
        t_marginal = self.forward(x_prime, y)
        exp_t = torch.exp(t_marginal)
        current = exp_t.mean().detach().item()
        if self.ma_et is None:
            self.ma_et = current
        else:
            self.ma_et = (1-self.ma_rate)*self.ma_et + self.ma_rate*current
        return t_joint - torch.log(exp_t.mean() + 1e-8)


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
):
    """Recover one candidate generator for the supplied trigger operator.

    `warm_start_state`: optional (G_state_dict, M_state_dict) pair from a
    previously-converged recovery run on the SAME attack family (e.g. the
    1% checkpoint's generator, reused as the starting point for the 5%
    checkpoint). This only reuses information already available to the
    defense (its own prior recovery runs on this attack family) — it does
    not give the defense any new access to attacker information — and
    typically needs fewer epochs to reconverge on a new checkpoint.

    NOTE: for Blended and Silent Killer this is an empirical extension of
    the original BadNets-oriented notebook implementation. Full-image
    recovery has a much larger search space (3072 values on CIFAR-10).
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

        for imgs, _ in defense_loader:
            imgs = imgs.to(device)
            B = imgs.size(0)

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

    for batch_idx, (imgs, _) in enumerate(defense_loader):
        if batch_idx >= n_eval_batches:
            break
        imgs = imgs.to(device)
        triggered = operator.apply(imgs, trigger_tensor)
        preds = victim_model(triggered).argmax(1)
        total_target += (preds == target_label).sum().item()
        total += imgs.size(0)

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
):
    """Recover and rank exact trigger candidates for one victim model.

    COARSE-TO-FINE WITH EARLY EXIT: tries `epsilons` (the coarse pass)
    first. If `min_accepted_to_stop` triggers are accepted, recovery stops
    there — the `fine_epsilons` list is only consulted if the coarse pass
    wasn't conclusive. This keeps genuine recovery running for every
    checkpoint while not paying for an exhaustive sweep once enough
    evidence exists.

    Per-attack budgets come from ATTACK_RECOVERY_CONFIG (keyed by
    `operator.recovery_key()`) unless `config_override` is supplied.

    Returns: (trigger_pool, all_candidates, recovery_meta)
      recovery_meta includes wall-clock time and which epsilons were
      actually tried, for the compute-cost table.
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

    last_g_state, last_m_state = None, None
    if warm_start_state is not None:
        last_g_state, last_m_state = warm_start_state

    def _run_epsilon_batch(epsilon_list, phase):
        nonlocal last_g_state, last_m_state
        for eps_idx, eps_val in enumerate(epsilon_list):
            set_seed(seed + eps_idx + target_label * 100)
            epsilons_tried.append(eps_val)

            if verbose:
                print(f"  [{phase}] recovery epsilon={eps_val:.2f}")

            ws = (last_g_state, last_m_state) if last_g_state is not None else None
            G, M, logs = recover_trigger_candidate(
                victim_model, defense_loader, target_label, eps_val,
                operator, device=device,
                epochs=cfg["epochs"],
                warm_start_state=ws,
            )

            # Cache this run's converged weights for the next epsilon /
            # next call's warm start (same attack family).
            last_g_state = {k: v.detach().clone() for k, v in G.state_dict().items()}
            last_m_state = {k: v.detach().clone() for k, v in M.state_dict().items()}

            G.eval()
            with torch.no_grad():
                z = G.gen_noise(cfg["n_candidates"], device)
                candidates = G(z)

            best_asr = -1.0
            best_tensor = None
            for c in range(cfg["n_candidates"]):
                candidate = candidates[c].view(*operator.trigger_shape())
                asr = evaluate_exact_trigger_asr(
                    victim_model, candidate, defense_loader,
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

    recovery_meta = {
        "attack": operator.recovery_key(),
        "epsilons_tried": epsilons_tried,
        "n_epsilons_tried": len(epsilons_tried),
        "n_accepted": len(trigger_pool),
        "wall_clock_seconds": time.time() - t_start,
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
):
    if not trigger_pool:
        raise ValueError("Empty trigger pool.")

    B = clean_imgs.shape[0]
    entry = trigger_pool[np.random.randint(0, len(trigger_pool))]
    trigger = entry["tensor"].to(device)

    if trigger.dim() == 3:
        trigger = trigger.unsqueeze(0)
    trigger = trigger.expand(B, -1, -1, -1)

    triggered = operator.apply(clean_imgs, trigger)
    labels = torch.full(
        (B,), target_label, dtype=torch.long, device=device
    )
    return triggered, labels


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
    asr_stop_threshold=BAERASER_ASR_STOP_THRESHOLD,
    omega_update_freq=3,
):
    """Shared BaEraser-lite dynamic-penalty unlearning.

    The attack-specific part is ONLY `operator.apply()`. The objective is:
        alpha * (clean_loss - trigger_loss) + beta * penalty
    where penalty = sum_k omega_k * |theta_k - theta_0,k|.

    `omega_update_freq`: recompute omega every N epochs instead of every
    epoch (previous default was every epoch, i.e. equivalent to
    omega_update_freq=1). omega is a full forward+backward pass over the
    defense set on top of the actual training step, so refreshing it less
    often meaningfully cuts per-epoch cost. Set to 1 to reproduce the
    original every-epoch behaviour as an ablation.
    """
    if not trigger_pool:
        raise ValueError("Cannot unlearn without recovered triggers.")

    theta0 = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
    }

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum
    )

    ca_min = None
    if ca_before is not None:
        ca_min = ca_before - ca_drop_tolerance

    logs = []
    best_state = None
    best_asr = float("inf")
    best_ca = -float("inf")
    best_epoch = 0
    omega = None

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        if omega is None or (epoch - 1) % omega_update_freq == 0:
            omega = compute_omega(model, defense_loader, criterion, device)

        model.train()
        sums = {"clean": 0.0, "trigger": 0.0, "penalty": 0.0, "total": 0.0}
        n_batches = 0

        for imgs, lbls in defense_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)

            clean_loss = criterion(model(imgs), lbls)

            triggered, trig_labels = make_triggered_batch(
                imgs, trigger_pool, target_label, operator, device
            )
            trigger_loss = criterion(model(triggered), trig_labels)

            penalty = torch.zeros((), device=device)
            for name, param in model.named_parameters():
                if name in omega and name in theta0:
                    penalty = penalty + (
                        omega[name] * (param - theta0[name]).abs()
                    ).sum()

            total_loss = alpha * (clean_loss - trigger_loss) + beta * penalty

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            sums["clean"] += clean_loss.item()
            sums["trigger"] += trigger_loss.item()
            sums["penalty"] += penalty.item()
            sums["total"] += total_loss.item()
            n_batches += 1

        ca = None
        asr = None
        if calculate_ca is not None and ca_loader is not None:
            ca = calculate_ca(model, ca_loader, device)
        if calculate_asr is not None and asr_loader is not None:
            asr = calculate_asr(model, asr_loader, target_label, device)

        param_dist = sum(
            (p - theta0[n]).norm().item()
            for n, p in model.named_parameters()
            if n in theta0
        )

        log = {
            "epoch": epoch,
            "clean_loss": sums["clean"]/max(n_batches,1),
            "trigger_loss": sums["trigger"]/max(n_batches,1),
            "penalty": sums["penalty"]/max(n_batches,1),
            "total_loss": sums["total"]/max(n_batches,1),
            "CA": ca,
            "ASR": asr,
            "param_dist": param_dist,
            "omega_refreshed": (epoch - 1) % omega_update_freq == 0,
            "time": time.time() - t0,
        }
        logs.append(log)

        # Valid checkpoint criterion: minimize ASR without exceeding CA floor.
        valid_ca = (ca is None) or (ca_min is None) or (ca >= ca_min)
        if valid_ca and asr is not None:
            if asr < best_asr or (asr == best_asr and (ca or 0) > best_ca):
                best_asr = asr
                best_ca = ca if ca is not None else best_ca
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        if asr is not None and valid_ca and asr < asr_stop_threshold:
            break

    return {
        "model": model,
        "best_state_dict": best_state,
        "logs": logs,
        "best_epoch": best_epoch,
        "CA_after": None if best_ca == -float("inf") else best_ca,
        "ASR_after": None if best_asr == float("inf") else best_asr,
        "CA_before": ca_before,
        "ASR_before": asr_before,
        "status": "SUCCESS" if best_state is not None else "NO_VALID_CHECKPOINT",
        "operator": operator.metadata(),
        "total_compute_seconds": sum(l["time"] for l in logs),
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
    **unlearning_kwargs,
):
    """Main API for the defense x attack matrix.

    DEPLOYABLE / MAIN-MATRIX PATH (use this for every cell of the defense x
    attack matrix and for anything feeding the controller-vs-baselines-vs-
    oracle comparison):

        baeraser_lite(model, clean_loader, BadNetsTriggerOperator())
        baeraser_lite(model, clean_loader,
                      BlendedTriggerOperator(alpha=ALPHA_TRAIN))
        baeraser_lite(model, clean_loader, SilentKillerTriggerOperator())

    To warm-start recovery from a previous poison-rate/count's converged
    generator (same attack family, cheaper reconvergence):

        result_1pct = baeraser_lite(model_1pct, loader, BadNetsTriggerOperator())
        ws = (result_1pct.recovery_meta["final_generator_state"],
              result_1pct.recovery_meta["final_mine_state"])
        result_5pct = baeraser_lite(model_5pct, loader, BadNetsTriggerOperator(),
                                     warm_start_state=ws)

    DIAGNOSTIC-ONLY PATH — oracle_trigger:
    This does NOT produce a BAERASER-lite result. It substitutes the
    attacker's actual, ground-truth trigger for the recovery stage, which no
    real defense (and no real post-deployment monitor) has access to. Its
    only legitimate use is to separate "did recovery fail?" from "did
    unlearning fail?" as a diagnostic ablation, reported in its own labeled
    subsection — never in the main defense x attack matrix, and never
    conflated with the ground-truth-best-defense "Oracle" row used in the
    controller-vs-baselines-vs-oracle comparison (that's a different oracle
    entirely: best defense, not best trigger).

        baeraser_lite(model, clean_loader, BlendedTriggerOperator(alpha=0.1),
                      oracle_trigger=known_blended_pattern,
                      diagnostic_only=True)   # <- required ack, not optional
    """
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
            warm_start_state=warm_start_state,
            config_override=config_override,
        )
        mode = "recovered"

    result = baeraser_unlearning(
        victim_model,
        defense_loader,
        trigger_pool,
        operator,
        target_label,
        device=device,
        **unlearning_kwargs,
    )
    result["mode"] = mode

    return BaEraserResult(
        model=result["model"],
        trigger_pool=trigger_pool,
        recovery_candidates=recovery_candidates,
        recovery_meta=recovery_meta,
        unlearning=result,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Example configuration
# ---------------------------------------------------------------------------
#
# Main matrix (deployable, use for every cell):
#
#   result = baeraser_lite(
#       victim_model,
#       defense_loader,
#       BadNetsTriggerOperator(size=4, placement="random"),
#       target_label=TARGET_CLASS,
#       calculate_ca=calculate_ca,
#       ca_loader=testloader,
#       calculate_asr=calculate_asr,
#       asr_loader=asr_loader,
#       ca_before=CA_before,
#       asr_before=ASR_before,
#   )
#   # result.mode == "recovered"
#   # result.recovery_meta["n_accepted"], ["wall_clock_seconds"] -> cost table
#
# Diagnostic-only, Blended/Silent Killer recovery-vs-unlearning ablation:
#
#   result = baeraser_lite(
#       victim_model, defense_loader,
#       BlendedTriggerOperator(alpha=0.1),
#       oracle_trigger=blended_pattern_seed777,   # loaded from disk
#       diagnostic_only=True,
#   )
#   # result.mode == "oracle_diagnostic" -- report separately, never in the
#   # main matrix.
