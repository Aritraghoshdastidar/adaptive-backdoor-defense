"""
defenses/nad.py — Neural Attention Distillation (Li et al., ICLR 2021)

Faithful to the authors' repo (bboylyg/NAD) and the paper's Sec. 4.1 recipe,
with the Issue 9 leak fix merged in.

THREAT MODEL (Issue 9)
    Defender holds: the poisoned model + the 2,500-image clean budget. Nothing else.
        teacher = fine-tune( deepcopy(poisoned_model), defense budget )
        student = poisoned_model  --attention distillation from teacher-->  sanitized
    No clean-baseline checkpoint is ever loaded.

WHAT MATCHES THE ORIGINAL REPO
    * attention map  : A = sum_c |F_c|^p (p=2), divided by its L2 norm over the
                       spatial dims (+1e-6), per sample          (at.py)
    * AT loss        : F.mse_loss(A_student, A_teacher) -> MEAN over batch AND
                       H*W positions                            (at.py)
                       (the old nad.py SUMMED over H*W; with the paper's betas that
                        is 16x-256x too strong depending on the layer)
    * betas          : 500 / 1000 / 1000 for low / mid / high residual groups (config.py)
    * total loss     : CE + sum_l beta_l * AT_l, gradients flow through AT (the repo
                       once had a .detach() bug here, since fixed)
    * optimiser      : SGD m=0.9, wd=1e-4, lr=0.1, /10 every 2 epochs, 10 epochs (paper 4.1)
    * augmentation   : random crop(pad 4) + h-flip + Cutout(1 hole, 9px) (paper 4.1)
    * teacher        : fine-tuned poisoned model, same recipe, frozen, eval mode
    * student modified once only, no iterating (paper App. G)

ADAPTATIONS (flagged, all configurable)
    * Repo/paper use WRN-16-1 with 3 residual groups. ResNet-18 has 4; default
      distills layer2/3/4 with 500/1000/1000 (low/mid/high). Pass `betas=` to change.
    * Augmentation is done on the GPU per batch (cheap). If your defense_loader
      already augments, pass augment=False.

COMPUTE-LIGHT CHOICES
    * 10 epochs for teacher and student (old code: 20 student epochs, which the paper
      shows overfits to the teacher)
    * teacher forward under no_grad, frozen, eval
    * bf16 autocast on CUDA when supported (attention maps computed in fp32)
    * student is deep-copied by default so the poisoned model can be reused by other defenses

COST ACCOUNTING: BOTH teacher fine-tuning and distillation count as NAD cost.
    results["total_cost_sec"] = teacher_ft_sec + distill_sec
"""

import copy
import hashlib
import time
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_BETAS = {"layer2": 500.0, "layer3": 1000.0, "layer4": 1000.0}


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


def _amp_ctx(device, amp: bool):
    dev_type = str(device).split(":")[0]
    on = bool(amp) and dev_type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=on)


@torch.no_grad()
def gpu_augment(x: torch.Tensor, pad: int = 4, cutout_len: int = 9) -> torch.Tensor:
    """Random crop (zero pad) + horizontal flip + Cutout(1 hole), vectorised on-device."""
    B, C, H, W = x.shape
    dev = x.device
    # random crop
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (B,), device=dev)
    j = torch.randint(0, 2 * pad + 1, (B,), device=dev)
    rows = i[:, None] + torch.arange(H, device=dev)
    cols = j[:, None] + torch.arange(W, device=dev)
    x = xp[torch.arange(B, device=dev)[:, None, None, None],
           torch.arange(C, device=dev)[None, :, None, None],
           rows[:, None, :, None], cols[:, None, None, :]]
    # horizontal flip
    flip = torch.rand(B, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    # cutout
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
         teacher=None, betas=None, p=2, verbose=True, tag="FT"):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)  # /10 every 2 epochs
    layers = list(betas) if betas else []
    s_ext = FeatureExtractor(model, layers) if teacher is not None else None
    t_ext = FeatureExtractor(teacher, layers) if teacher is not None else None
    hist = {"cls": [], "at": []}
    try:
        model.train()
        for ep in range(epochs):
            cls_sum = at_sum = n = 0.0
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                if augment:
                    x = gpu_augment(x)
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
                at_sum += float(at_val) * b
                n += b
            sched.step()
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


# ── Issue 9: leak-free teacher ───────────────────────────────────────────────

def build_nad_teacher(
    poisoned_model: nn.Module,
    defense_loader,
    device: str,
    epochs: int = 10,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    augment: bool = True,
    amp: bool = True,
    eval_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
    clean_baseline_model: Optional[nn.Module] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict]:
    """teacher = fine-tune(deepcopy(poisoned_model)) on the clean budget. Poisoned model untouched."""
    before = state_hash(poisoned_model)
    teacher = copy.deepcopy(poisoned_model).to(device)
    init_hash = state_hash(teacher)
    assert init_hash == before, "deepcopy did not preserve weights"
    if clean_baseline_model is not None and init_hash == state_hash(clean_baseline_model):
        raise RuntimeError("NAD teacher initialised from the clean baseline (leak).")

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
        "teacher_ft_sec": round(ft_sec, 2),
        "teacher_init_hash": init_hash,
    }
    if eval_fn is not None:
        m = eval_fn(teacher)
        info["teacher_ca"], info["teacher_asr"] = m.get("ca"), m.get("asr")
    return teacher, info


# ── NAD distillation ─────────────────────────────────────────────────────────

def run_nad(
    poisoned_model: nn.Module,
    teacher_model: nn.Module,
    defense_loader,
    device: str = "cuda",
    epochs: int = 10,
    lr: float = 0.1,
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
    **kwargs,
) -> Tuple[nn.Module, dict]:
    """
    Leak-free NAD: build teacher from the poisoned model, distill into a copy of it.

    eval_fn(model) -> {"ca": ..., "asr": ...}; used to log the TEACHER's CA/ASR.
    clean_baseline_model is only used as a leak guard (never as a teacher).
    Returns (sanitized_model, results). results["total_cost_sec"] is the NAD cost.
    """
    teacher, tinfo = build_nad_teacher(
        poisoned_model, defense_loader, device, epochs=teacher_epochs,
        eval_fn=eval_fn, clean_baseline_model=clean_baseline_model,
        augment=kwargs.get("augment", True), amp=kwargs.get("amp", True),
        verbose=kwargs.get("verbose", True))
    sanitized, res = run_nad(poisoned_model, teacher, defense_loader, device=device,
                             epochs=nad_epochs, betas=betas, **kwargs)
    res.update(tinfo)
    res["total_cost_sec"] = round(tinfo["teacher_ft_sec"] + res["distill_sec"], 2)
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
    data = [(torch.randn(16, 3, 16, 16), torch.randint(0, 3, (16,))) for _ in range(4)]
    before = state_hash(net)
    ev = lambda m: {"ca": 0.0, "asr": 0.0}
    out, res = nad_full_pipeline(net, data, "cpu", eval_fn=ev, teacher_epochs=2, nad_epochs=2,
                                 verbose=False)
    assert state_hash(net) == before, "poisoned model was modified"
    assert state_hash(out) != before, "student did not change"
    assert res["total_cost_sec"] >= res["distill_sec"]
    aug = gpu_augment(torch.randn(8, 3, 32, 32))
    assert aug.shape == (8, 3, 32, 32)
    a = attention_map(torch.randn(2, 8, 4, 4))
    assert torch.allclose(torch.norm(a, dim=(2, 3)), torch.ones(2, 1), atol=1e-4)
    print("OK", {k: v for k, v in res.items() if k != "history"})