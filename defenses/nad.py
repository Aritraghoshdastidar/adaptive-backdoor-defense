"""
defenses/nad.py — Neural Attention Distillation (Li et al., ICLR 2021)

Faithful to the authors' repo (bboylyg/NAD) where it matters for the method, with
the Issue 9 leak fix and the v2 fixes from the executed-notebook review.

THREAT MODEL (Issue 9)
    Defender holds: the poisoned model + the clean defense budget. Nothing else.
        teacher = fine-tune( deepcopy(poisoned_model), defense budget )
        student = poisoned_model  --attention distillation from teacher-->  sanitized
    No clean-baseline checkpoint is ever loaded.

WHAT MATCHES THE ORIGINAL REPO / PAPER
    * attention map  : A = sum_c |F_c|^p (p=2), divided by its spatial L2 norm (+1e-6)
    * AT loss        : F.mse_loss(A_s, A_t)  -> MEAN over batch AND H*W   (at.py)
    * betas          : 500 / 1000 / 1000 for low / mid / high groups       (config.py)
    * total loss     : CE + sum_l beta_l * AT_l, gradients flow through AT
    * optimiser      : SGD m=0.9, wd=1e-4, lr /10 every 2 epochs, 10 epochs
    * augmentation   : random crop(pad 4, BLACK pad) + h-flip + Cutout(9px)
    * teacher        : fine-tuned poisoned model, same recipe, frozen, eval mode
    * one distillation pass only (paper App. G)

DELIBERATE DEVIATIONS (v2)
    * lr default is 0.01, NOT the paper's 0.1.  The paper's 0.1 was tuned for a
      WRN-16-1 at 85% CA. On an already-converged ResNet-18 (94% CA) it wrecked
      the teacher (CA ~36-50%) and the student (CA ~26-34%) in the executed run, and
      the step schedule then froze the model in the broken state. Because the best
      lr is model-dependent, use select_teacher_lr() (defender-side, CA-only) or
      sweep it and REPORT the sweep.
    * Sanity gate: pass val_loader (held-out clean defense images). Teacher and
      student CA are checked against the poisoned model's CA; res["valid_run"] is
      False (and a warning is raised) if either drops more than max_ca_drop.
      A run with valid_run=False must not be reported as a defense result.
    * Layers: ResNet-18 has 4 groups, the paper's WRN-16-1 has 3. Default distills
      layer2/3/4 (500/1000/1000). Pass `betas=` to change.
    * bf16 autocast only on GPUs with native bf16 (compute capability >= 8). On a T4
      torch.cuda.is_bf16_supported() can return True through emulation, which is slow
      and imprecise; v1 used it and the timings were inflated.

v3 FIXES (review of the v2 file)
    * Collapse gate tightened: max_ca_drop default 0.10 -> 0.05 (teacher AND student).
    * Teacher-ASR flag: a NAD teacher that still carries the backdoor (ASR above
      `teacher_asr_flag_threshold`) cannot teach a student to forget it. Pass eval_fn;
      info["teacher_asr_flag"] is set and res["interpretable"] is False. The ASR is a
      DIAGNOSTIC (uses the real trigger via eval_fn) and is never used to select
      anything.
    * Matched fine-tuning baseline: run_matched_ft() uses the SAME recipe as the
      teacher (lr, augmentation, schedule, black-pad fill) for teacher_epochs+nad_epochs
      (equal compute to NAD). The v1 FT baseline (lr 1e-4, no augmentation) was a straw
      man; teacher_ft itself is also a matched, half-budget FT row.
    * Student lr / beta sweep: sweep_nad_student() reports the whole grid and picks the
      strongest distillation (lr * beta_scale) that still passes the held-out CA gate.
      Defender-computable; ASR columns (diag_fn) are diagnostics only.
    * Cost accounting: nad_full_pipeline(select_lr=True, search_sec=...) folds the
      teacher-lr search time into res["total_cost_sec"].
    * res["teacher_final_hash"/"student_final_hash"/"teacher_student_distinct"]: proves the
      student is not the teacher (equal CA on a finite test set can be a coincidence).
    * assert_baseline_ca(): a checkpoint does not store its normalisation constants;
      this checks the loaded model reproduces its recorded CA (run on the FULL test
      loader before switching to the report half).
    * PROTOCOL (needs notebook changes): val_loader/defense_loader must not overlap the
      poisoned model's training set. Use the test-set split (val half for the defense
      budget + val_loader, report half for CA/ASR), identical to BaEraser/FT/ANP.

COST ACCOUNTING: teacher fine-tuning, distillation and any lr search all count as NAD
cost:  total = search_sec + teacher_ft_sec + distill_sec   (pipeline does this when
select_lr=True or search_sec is passed).
"""

import copy
import hashlib
import math
import time
import warnings
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_BETAS = {"layer2": 500.0, "layer3": 1000.0, "layer4": 1000.0}
NORM_MEAN = (0.4914, 0.4822, 0.4465)   # override if your loaders normalise differently
NORM_STD = (0.2470, 0.2435, 0.2616)


# ── utilities ────────────────────────────────────────────────────────────────

def state_hash(model: nn.Module) -> str:
    """Stable fingerprint of weights + buffers."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def _sync(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _native_bf16(device) -> bool:
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(dev)[0] >= 8 and torch.cuda.is_bf16_supported()


def _amp_ctx(device, amp: bool):
    dev = torch.device(device)
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                          enabled=bool(amp) and _native_bf16(device))


@torch.no_grad()
def clean_acc(model: nn.Module, loader, device) -> float:
    """Top-1 accuracy on a clean loader (sets eval mode)."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def assert_baseline_ca(model: nn.Module, loader, expected_ca: float, device,
                       tol: float = 0.001, name: str = "model") -> float:
    """Check a loaded checkpoint reproduces its recorded clean accuracy (in [0,1]).
    Catches normalisation / checkpoint mismatches that a constant check cannot."""
    ca = clean_acc(model.to(device), loader, device)
    if abs(ca - expected_ca) > tol:
        raise AssertionError(f"{name}: CA {ca*100:.2f}% != recorded {expected_ca*100:.2f}% "
                             f"(tol {tol*100:.2f}pp) -> normalisation/checkpoint mismatch")
    return ca


def _black_fill(device, dtype=torch.float32) -> torch.Tensor:
    """Normalised value of a raw-black pixel, shape (1,C,1,1)."""
    m = torch.tensor(NORM_MEAN, device=device, dtype=dtype).view(1, -1, 1, 1)
    s = torch.tensor(NORM_STD, device=device, dtype=dtype).view(1, -1, 1, 1)
    return -m / s


@torch.no_grad()
def gpu_augment(x: torch.Tensor, pad: int = 4, cutout_len: int = 9,
                fill: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Random crop + h-flip + Cutout(1 hole), vectorised.

    `fill` (1,C,1,1) is the value of the crop padding. The original pads raw
    images with black BEFORE normalisation, i.e. -mean/std in normalised space;
    zero-padding normalised tensors (v1) paints mean-gray instead.
    """
    B, C, H, W = x.shape
    dev = x.device
    fill = torch.zeros(1, C, 1, 1, device=dev, dtype=x.dtype) if fill is None else fill.to(dev, x.dtype)
    xp = F.pad(x - fill, (pad, pad, pad, pad)) + fill
    i = torch.randint(0, 2 * pad + 1, (B,), device=dev)
    j = torch.randint(0, 2 * pad + 1, (B,), device=dev)
    rows = i[:, None] + torch.arange(H, device=dev)
    cols = j[:, None] + torch.arange(W, device=dev)
    x = xp[torch.arange(B, device=dev)[:, None, None, None],
           torch.arange(C, device=dev)[None, :, None, None],
           rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(B, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    # Cutout: zeros in normalised space, as in the original (applied after Normalize)
    cy = torch.randint(0, H, (B,), device=dev)
    cx = torch.randint(0, W, (B,), device=dev)
    h2 = cutout_len // 2
    ar_h, ar_w = torch.arange(H, device=dev), torch.arange(W, device=dev)
    ym = (ar_h[None] >= (cy - h2).clamp(0, H)[:, None]) & (ar_h[None] < (cy + h2).clamp(0, H)[:, None])
    xm = (ar_w[None] >= (cx - h2).clamp(0, W)[:, None]) & (ar_w[None] < (cx + h2).clamp(0, W)[:, None])
    hole = ym[:, :, None] & xm[:, None, :]
    return x * (~hole)[:, None].to(x.dtype)


# ── attention transfer (matches bboylyg/NAD at.py) ───────────────────────────

def attention_map(fm: torch.Tensor, p: int = 2, eps: float = 1e-6) -> torch.Tensor:
    """(B,C,H,W) -> (B,1,H,W): sum_c |F|^p, divided by its per-sample spatial L2 norm."""
    am = fm.float().abs().pow(p).sum(dim=1, keepdim=True)
    norm = torch.norm(am, dim=(2, 3), keepdim=True)
    return am / (norm + eps)


def at_loss(fm_s: torch.Tensor, fm_t: torch.Tensor, p: int = 2) -> torch.Tensor:
    """MSE over batch and spatial positions, exactly as in the original repo."""
    return F.mse_loss(attention_map(fm_s, p), attention_map(fm_t, p))


class FeatureExtractor:
    """Forward hooks that capture the output of named residual groups."""

    def __init__(self, model: nn.Module, layer_names):
        mods = dict(model.named_modules())
        self.features, self._hooks = {}, []
        for name in layer_names:
            if name not in mods:
                raise KeyError(f"layer '{name}' not found in model")
            self._hooks.append(mods[name].register_forward_hook(self._make(name)))

    def _make(self, name):
        def fn(_m, _i, out):
            self.features[name] = out
        return fn

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── shared training loop (teacher: CE only; student: CE + beta*AT) ───────────

def _fit(model, loader, device, epochs, lr, momentum, weight_decay, augment, amp,
         teacher=None, betas=None, p=2, verbose=True, tag="FT", lr_step=2, lr_gamma=0.1):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)
    layers = list(betas) if betas else []
    s_ext = FeatureExtractor(model, layers) if teacher is not None else None
    t_ext = FeatureExtractor(teacher, layers) if teacher is not None else None
    fill = _black_fill(device) if augment else None
    hist = {"cls": [], "at": []}
    try:
        model.train()
        for ep in range(epochs):
            cls_sum = at_sum = n = 0.0
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                if augment:
                    x = gpu_augment(x, fill=fill)
                opt.zero_grad(set_to_none=True)
                with _amp_ctx(device, amp):
                    out = model(x)
                    if teacher is not None:
                        with torch.no_grad():
                            teacher(x)
                cls = F.cross_entropy(out.float(), y)
                loss, at_val = cls, torch.zeros((), device=x.device)
                if teacher is not None:
                    at_val = sum(betas[l] * at_loss(s_ext.features[l], t_ext.features[l], p)
                                 for l in layers)
                    loss = cls + at_val
                loss.backward()
                opt.step()
                b = x.size(0)
                cls_sum += cls.item() * b
                at_sum += at_val.detach().item() * b
                n += b
            sched.step()
            if not math.isfinite(cls_sum):
                raise RuntimeError(f"[{tag}] loss became non-finite at epoch {ep+1}; lower lr")
            hist["cls"].append(cls_sum / n)
            hist["at"].append(at_sum / n)
            if verbose:
                print(f"  [{tag}] epoch {ep+1}/{epochs}  ce={hist['cls'][-1]:.4f}"
                      + (f"  beta*at={hist['at'][-1]:.4f}" if teacher is not None else ""))
    finally:
        if s_ext:
            s_ext.remove()
            t_ext.remove()
    model.eval()
    return hist


# ── defender-side teacher-lr selection (CA only, no ASR, no test data) ───────

def select_teacher_lr(poisoned_model, train_loader, val_loader, device,
                      lrs=(0.02, 0.01, 0.005, 0.001), max_ca_drop=0.05, epochs=10,
                      momentum=0.9, weight_decay=1e-4, augment=True, amp=True,
                      diag_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
                      seed: int = 2027):
    """Largest lr whose fine-tuned teacher keeps held-out clean accuracy within
    `max_ca_drop` of the poisoned model's. Uses only defender-available data.
    Falls back to the smallest lr if none qualify (info["fallback"]=True).
    `diag_fn(model)->dict` (e.g. {"ca":..,"asr":..}) is logged per lr for REPORTING
    ONLY; selection never looks at it. The search time is real NAD cost:
    add info["search_sec"] to the total."""
    t0 = time.time()
    base = clean_acc(poisoned_model.to(device), val_loader, device)
    table = []
    for lr in sorted(lrs, reverse=True):
        torch.manual_seed(seed)
        t = copy.deepcopy(poisoned_model).to(device)
        _fit(t, train_loader, device, epochs, lr, momentum, weight_decay, augment, amp,
             verbose=False, tag="lr-search")
        ca = clean_acc(t, val_loader, device)
        row = {"lr": lr, "val_ca": round(ca, 4), "drop": round(base - ca, 4)}
        if diag_fn is not None:
            row.update({f"diag_{k}": v for k, v in diag_fn(t).items()})
        table.append(row)
        del t
    ok = [r for r in table if r["drop"] <= max_ca_drop]
    chosen = max(r["lr"] for r in ok) if ok else min(lrs)
    _sync(device)
    return chosen, {"table": table, "base_val_ca": round(base, 4), "fallback": not ok,
                    "search_sec": round(time.time() - t0, 2)}


# ── Issue 9: leak-free teacher ───────────────────────────────────────────────

def build_nad_teacher(
    poisoned_model: nn.Module,
    defense_loader,
    device: str,
    epochs: int = 10,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    augment: bool = True,
    amp: bool = True,
    eval_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
    clean_baseline_model: Optional[nn.Module] = None,
    val_loader=None,
    max_ca_drop: float = 0.05,
    strict: bool = False,
    verbose: bool = True,
    teacher_asr_flag_threshold: float = 0.5,
) -> Tuple[nn.Module, Dict]:
    """teacher = fine-tune(deepcopy(poisoned_model)) on the clean budget. Poisoned model untouched.

    With val_loader (held-out clean defense images) the teacher's CA is gated against
    the poisoned model's CA; info["teacher_gate_passed"] reports the result
    (strict=True raises instead of warning)."""
    before = state_hash(poisoned_model)
    teacher = copy.deepcopy(poisoned_model).to(device)
    init_hash = state_hash(teacher)
    assert init_hash == before, "deepcopy did not preserve weights"
    if clean_baseline_model is not None and init_hash == state_hash(clean_baseline_model):
        raise RuntimeError("NAD teacher initialised from the clean baseline (leak).")
    base_val = clean_acc(teacher, val_loader, device) if val_loader is not None else None

    _sync(device)
    t0 = time.time()
    _fit(teacher, defense_loader, device, epochs, lr, momentum, weight_decay,
         augment, amp, verbose=verbose, tag="teacher")
    _sync(device)
    ft_sec = time.time() - t0

    teacher.eval()
    for q in teacher.parameters():
        q.requires_grad_(False)

    assert state_hash(poisoned_model) == before, "poisoned model modified while building teacher"
    assert state_hash(teacher) != init_hash, "teacher fine-tuning changed nothing"

    info = {
        "teacher_source": "deepcopy(poisoned_model)",
        "teacher_ft_epochs": epochs,
        "teacher_lr": lr,
        "teacher_ft_sec": round(ft_sec, 2),
        "teacher_init_hash": init_hash,
        "teacher_gate_passed": None,
    }
    if val_loader is not None:
        t_val = clean_acc(teacher, val_loader, device)
        ok = (base_val - t_val) <= max_ca_drop
        info.update(poisoned_val_ca=round(base_val, 4), teacher_val_ca=round(t_val, 4),
                    teacher_gate_passed=bool(ok))
        if not ok:
            msg = (f"NAD teacher collapsed: held-out CA {base_val:.3f} -> {t_val:.3f} "
                   f"(> {max_ca_drop:.2f} drop) at lr={lr}. Lower the lr.")
            if strict:
                raise RuntimeError(msg)
            warnings.warn(msg)
    if eval_fn is not None:
        m = eval_fn(teacher)
        info["teacher_ca"], info["teacher_asr"] = m.get("ca"), m.get("asr")
        # DIAGNOSTIC (real trigger): a teacher that still has the backdoor cannot make
        # the student forget it, so the NAD number is not interpretable as "NAD works".
        flag = info["teacher_asr"] is not None and info["teacher_asr"] > teacher_asr_flag_threshold
        info["teacher_asr_flag"] = bool(flag)
        if flag:
            warnings.warn(f"NAD teacher still backdoored (ASR {info['teacher_asr']:.3f} > "
                          f"{teacher_asr_flag_threshold:.2f}); distillation cannot remove it. "
                          f"Report NAD as 'teacher ineffective', not as a defense result.")
    else:
        info["teacher_asr_flag"] = None
    return teacher, info


# ── NAD distillation ─────────────────────────────────────────────────────────

def run_nad(
    poisoned_model: nn.Module,
    teacher_model: nn.Module,
    defense_loader,
    device: str = "cuda",
    epochs: int = 10,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    betas: Optional[Dict[str, float]] = None,
    power: int = 2,
    augment: bool = True,
    amp: bool = True,
    inplace: bool = False,
    verbose: bool = True,
) -> Tuple[nn.Module, dict]:
    """Distil attention from the (frozen) teacher into the poisoned student."""
    betas = dict(DEFAULT_BETAS if betas is None else betas)
    student = poisoned_model if inplace else copy.deepcopy(poisoned_model)
    student = student.to(device)
    teacher = teacher_model.to(device).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)

    _sync(device)
    t0 = time.time()
    hist = _fit(student, defense_loader, device, epochs, lr, momentum, weight_decay,
                augment, amp, teacher=teacher, betas=betas, p=power,
                verbose=verbose, tag="NAD")
    _sync(device)
    sec = time.time() - t0

    return student, {
        "defense": "NAD", "epochs": epochs, "lr": lr, "betas": betas, "power": power,
        "distill_sec": round(sec, 2), "history": hist,
    }


# ── matched fine-tuning baseline ─────────────────────────────────────────────

def run_matched_ft(poisoned_model: nn.Module, defense_loader, device: str = "cuda",
                   epochs: int = 20, lr: float = 0.01, momentum: float = 0.9,
                   weight_decay: float = 1e-4, augment: bool = True, amp: bool = True,
                   verbose: bool = False) -> Tuple[nn.Module, dict]:
    """Fine-tune a copy of the poisoned model with EXACTLY the NAD recipe (same lr,
    augmentation, black-pad fill, step schedule). Use epochs = teacher_epochs +
    nad_epochs for an equal-compute baseline. Pass the lr chosen for the teacher."""
    m = copy.deepcopy(poisoned_model).to(device)
    _sync(device)
    t0 = time.time()
    _fit(m, defense_loader, device, epochs, lr, momentum, weight_decay, augment, amp,
         verbose=verbose, tag="matched-FT")
    _sync(device)
    return m, {"defense": "FT-matched", "epochs": epochs, "lr": lr,
               "ft_sec": round(time.time() - t0, 2)}


# ── student lr / beta sweep ──────────────────────────────────────────────────

def sweep_nad_student(poisoned_model, teacher_model, defense_loader, val_loader, device,
                      lrs=(0.02, 0.01, 0.005), beta_scales=(0.5, 1.0, 2.0), epochs=10,
                      max_ca_drop=0.05, betas=None,
                      diag_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
                      **run_kwargs):
    """Grid over student lr x beta scale (all betas multiplied by the scale).

    Selection is defender-computable: the STRONGEST distillation (largest lr*beta_scale)
    whose held-out CA drop is <= max_ca_drop. Held-out CA alone cannot see backdoor
    removal, so report the WHOLE grid; `diag_fn` adds diagnostic ASR columns that are
    never used for selection. Returns (chosen_row_or_None, table, info); add
    info["search_sec"] to the NAD cost."""
    base_betas = dict(DEFAULT_BETAS if betas is None else betas)
    t0 = time.time()
    base = clean_acc(poisoned_model.to(device), val_loader, device)
    table = []
    for lr in lrs:
        for bs in beta_scales:
            sb = {k: v * bs for k, v in base_betas.items()}
            student, _ = run_nad(poisoned_model, teacher_model, defense_loader, device=device,
                                 epochs=epochs, lr=lr, betas=sb, verbose=False, **run_kwargs)
            ca = clean_acc(student, val_loader, device)
            row = {"lr": lr, "beta_scale": bs, "val_ca": round(ca, 4), "drop": round(base - ca, 4)}
            if diag_fn is not None:
                row.update({f"diag_{k}": v for k, v in diag_fn(student).items()})
            table.append(row)
            del student
    ok = [r for r in table if r["drop"] <= max_ca_drop]
    chosen = max(ok, key=lambda r: r["lr"] * r["beta_scale"]) if ok else None
    _sync(device)
    return chosen, table, {"base_val_ca": round(base, 4), "search_sec": round(time.time() - t0, 2)}


# ── full pipeline ────────────────────────────────────────────────────────────

def nad_full_pipeline(
    poisoned_model: nn.Module,
    defense_loader,
    device: str = "cuda",
    eval_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
    clean_baseline_model: Optional[nn.Module] = None,
    teacher_epochs: int = 10,
    nad_epochs: int = 10,
    betas: Optional[Dict[str, float]] = None,
    lr: float = 0.01,
    teacher_lr: Optional[float] = None,
    val_loader=None,
    max_ca_drop: float = 0.05,
    augment: bool = True,
    amp: bool = True,
    verbose: bool = True,
    select_lr: bool = False,
    lr_grid=(0.02, 0.01, 0.005, 0.001),
    search_sec: float = 0.0,
    matched_ft: bool = False,
    teacher_asr_flag_threshold: float = 0.5,
    **kwargs,
) -> Tuple[nn.Module, dict]:
    """
    Leak-free NAD: build teacher from the poisoned model, distill into a copy of it.

    eval_fn(model) -> {"ca","asr"}: logs the TEACHER's CA/ASR (diagnostic, not selection).
    val_loader: held-out clean defense images (NOT in defense_loader) enabling the
        collapse gate on teacher AND student. res["valid_run"] is False if either loses
        more than max_ca_drop (default 5pp) held-out CA; such runs must not be reported.
    select_lr=True (needs val_loader): choose the teacher lr with select_teacher_lr over
        lr_grid (CA-only); the search time is added to the cost. The student keeps `lr`
        unless you also sweep it with sweep_nad_student.
    matched_ft=True: also fine-tune a copy with the identical recipe for
        teacher_epochs+nad_epochs (equal compute) -> res["matched_ft"] (+ "matched_ft_model").
    res["interpretable"]: valid_run AND the teacher is not still backdoored
        (needs eval_fn). Report NAD as a defense only when it is True.
    clean_baseline_model: leak guard only, never a teacher.
    res["total_cost_sec"] = search_sec + teacher_ft_sec + distill_sec.
    """
    if select_lr:
        if val_loader is None:
            raise ValueError("select_lr=True needs val_loader")
        teacher_lr, linfo = select_teacher_lr(poisoned_model, defense_loader, val_loader, device,
                                              lrs=lr_grid, max_ca_drop=max_ca_drop,
                                              epochs=teacher_epochs, augment=augment, amp=amp)
        search_sec += linfo["search_sec"]
    else:
        linfo = None
    t_lr = lr if teacher_lr is None else teacher_lr

    teacher, tinfo = build_nad_teacher(
        poisoned_model, defense_loader, device, epochs=teacher_epochs, lr=t_lr,
        augment=augment, amp=amp, eval_fn=eval_fn, clean_baseline_model=clean_baseline_model,
        val_loader=val_loader, max_ca_drop=max_ca_drop, verbose=verbose,
        teacher_asr_flag_threshold=teacher_asr_flag_threshold)
    sanitized, res = run_nad(poisoned_model, teacher, defense_loader, device=device,
                             epochs=nad_epochs, lr=lr, betas=betas, augment=augment,
                             amp=amp, verbose=verbose, **kwargs)
    res.update(tinfo)
    # weight fingerprints: identical CA on a finite test set can be a coincidence; identical hashes cannot
    res["teacher_final_hash"] = state_hash(teacher)
    res["student_final_hash"] = state_hash(sanitized)
    res["teacher_student_distinct"] = res["teacher_final_hash"] != res["student_final_hash"]
    assert res["teacher_student_distinct"], "NAD student is bit-identical to the teacher"
    res["lr_search"] = linfo
    res["search_sec"] = round(search_sec, 2)
    res["total_cost_sec"] = round(search_sec + tinfo["teacher_ft_sec"] + res["distill_sec"], 2)

    res["student_gate_passed"], res["valid_run"] = None, None
    if val_loader is not None:
        s_val = clean_acc(sanitized, val_loader, device)
        s_ok = (tinfo["poisoned_val_ca"] - s_val) <= max_ca_drop
        res.update(student_val_ca=round(s_val, 4), student_gate_passed=bool(s_ok),
                   valid_run=bool(s_ok and tinfo["teacher_gate_passed"]))
        if not s_ok:
            warnings.warn(f"NAD student collapsed: held-out CA {tinfo['poisoned_val_ca']:.3f} "
                          f"-> {s_val:.3f}. Result is INVALID.")
    flag = tinfo.get("teacher_asr_flag")
    res["interpretable"] = (None if (res["valid_run"] is None or flag is None)
                            else bool(res["valid_run"] and not flag))

    if matched_ft:
        mft, minfo = run_matched_ft(poisoned_model, defense_loader, device,
                                    epochs=teacher_epochs + nad_epochs, lr=t_lr,
                                    augment=augment, amp=amp)
        if val_loader is not None:
            minfo["val_ca"] = round(clean_acc(mft, val_loader, device), 4)
        if eval_fn is not None:
            m = eval_fn(mft)
            minfo["ca"], minfo["asr"] = m.get("ca"), m.get("asr")
        res["matched_ft"] = minfo
        res["matched_ft_model"] = mft
    return sanitized, res


# ── self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            mk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
            self.layer1, self.layer2, self.layer3, self.layer4 = mk(3, 4), mk(4, 8), mk(8, 8), mk(8, 8)
            self.fc = nn.Linear(8, 3)

        def forward(self, x):
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            return self.fc(x.mean((2, 3)))

    torch.manual_seed(2027)
    net = Tiny()
    mk_data = lambda n: [(torch.randn(16, 3, 16, 16), torch.randint(0, 3, (16,))) for _ in range(n)]
    data, val = mk_data(4), mk_data(2)
    before = state_hash(net)
    ev = lambda m: {"ca": 0.0, "asr": 0.0}

    out, res = nad_full_pipeline(net, data, "cpu", eval_fn=ev, teacher_epochs=2, nad_epochs=2,
                                 val_loader=val, max_ca_drop=1.0, verbose=False)
    assert state_hash(net) == before, "poisoned model was modified"
    assert state_hash(out) != before, "student did not change"
    assert res["total_cost_sec"] >= res["distill_sec"]
    assert res["valid_run"] is True and res["teacher_gate_passed"] is True

    with warnings.catch_warnings(record=True) as w:        # gate must fire when CA collapses
        warnings.simplefilter("always")
        _, info = build_nad_teacher(net, data, "cpu", epochs=1, val_loader=val,
                                    max_ca_drop=-1.0, verbose=False)   # impossible bound
        assert info["teacher_gate_passed"] is False and any("collapsed" in str(x.message) for x in w)

    lr_pick, linfo = select_teacher_lr(net, data, val, "cpu", lrs=(0.05, 0.01), epochs=1,
                                       max_ca_drop=1.0)
    assert lr_pick == 0.05 and not linfo["fallback"]

    # augmentation: padding must equal raw-black in normalised space
    fill = _black_fill("cpu")
    xb = fill.expand(8, 3, 32, 32).clone()
    assert torch.allclose(gpu_augment(xb, cutout_len=0, fill=fill), xb, atol=1e-5)
    assert gpu_augment(torch.randn(8, 3, 32, 32), fill=fill).shape == (8, 3, 32, 32)
    a = attention_map(torch.randn(2, 8, 4, 4))
    assert torch.allclose(torch.norm(a, dim=(2, 3)), torch.ones(2, 1), atol=1e-4)
    # v3 checks
    assert res["interpretable"] is True and res["teacher_asr_flag"] is False
    assert res["teacher_student_distinct"] is True
    out2, res2 = nad_full_pipeline(net, data, "cpu", eval_fn=lambda m: {"ca": 0.0, "asr": 0.99},
                                   teacher_epochs=1, nad_epochs=1, val_loader=val,
                                   max_ca_drop=1.0, verbose=False, select_lr=True,
                                   lr_grid=(0.05, 0.01), matched_ft=True)
    assert res2["teacher_asr_flag"] is True and res2["interpretable"] is False
    assert res2["search_sec"] > 0 and res2["total_cost_sec"] >= res2["search_sec"]
    assert res2["matched_ft"]["epochs"] == 2 and "matched_ft_model" in res2
    ch, tab, inf = sweep_nad_student(net, build_nad_teacher(net, data, "cpu", epochs=1,
                                     verbose=False)[0], data, val, "cpu", lrs=(0.05, 0.01),
                                     beta_scales=(1.0, 2.0), epochs=1, max_ca_drop=1.0)
    assert len(tab) == 4 and ch["lr"] * ch["beta_scale"] == max(r["lr"] * r["beta_scale"] for r in tab)
    assert abs(assert_baseline_ca(net, val, clean_acc(net, val, "cpu"), "cpu") - clean_acc(net, val, "cpu")) < 1e-9
    print("OK", {k: v for k, v in res.items() if k not in ("history", "matched_ft_model")})