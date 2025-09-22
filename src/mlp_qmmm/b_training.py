#!/usr/bin/env python3
# b_training.py — QMMM training with correct weighting, masking, and padded-MM sanitization
# - eval uses autograd (for gradient-based heads)
# - AMP defaults to OFF (safer across different GPUs)
# - Progress mirroring to file (tee)
# - Optional periodic TorchScript export
# - Optional threshold-gated weighting (TGW): OFF by default

from __future__ import annotations
import argparse, importlib, json, os, time, math, platform, sys
from fnmatch import fnmatch
from typing import Dict, List, Tuple, Any, Optional

import yaml, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from mlp_qmmm.a_parser import get_adapter

# ---------------- constants & units ----------------
KCALMOL_TO_EV = 0.0433641153087705
EV_TO_KCALMOL = 1.0 / KCALMOL_TO_EV  # ~23.06055

def _unit_multiplier_for_key(key: str, print_units: str) -> float:
    if print_units.lower().startswith("kcal"):
        k = key.lower()
        if (k in ("de", "energy") or k.startswith("e_") or "grad" in k or k.endswith("_esp")):
            return EV_TO_KCALMOL
    return 1.0

# ---------------- [progress] tee helper ----------------
class _ProgressTee:
    """Mirror important prints to a file with line buffering."""
    def __init__(self, path: Optional[str]):
        self.path = (path or "").strip()
        self.fh = None
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self.fh = open(self.path, "a", buffering=1, encoding="utf-8")
    def log(self, *parts: Any) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        if self.fh:
            try:
                self.fh.write(line + "\n")
                self.fh.flush()
            except Exception:
                pass
    def close(self) -> None:
        try:
            if self.fh:
                self.fh.close()
        except Exception:
            pass

# ---------------- utils ----------------
def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    if x.dtype == np.float64:
        t = torch.from_numpy(x.astype(np.float32, copy=False))
    elif x.dtype.kind in ("f",):
        t = torch.from_numpy(x)
    elif x.dtype.kind in ("i","u"):
        t = torch.from_numpy(x.astype(np.int64, copy=False))
    else:
        t = torch.from_numpy(x)
    return t  # keep on CPU; moved to device later

def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    def _move(v):
        return v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    out = {k: _move(v) for k, v in batch.items() if k != "targets"}
    if "targets" in batch and isinstance(batch["targets"], dict):
        out["targets"] = {k: _move(v) for k, v in batch["targets"].items()}
    return out

def build_species_map_and_order(qm_Z_first: np.ndarray, cfg_species: List[int] | None
                                ) -> Tuple[Dict[int,int], List[int]]:
    if cfg_species:
        species_order = list(cfg_species)
    else:
        z = np.asarray(qm_Z_first).reshape(-1)
        species_order = sorted({int(v) for v in z.tolist()})
    z2idx = {z:i for i,z in enumerate(species_order)}
    return z2idx, species_order

def z_to_types(z_vec: np.ndarray, z2idx: Dict[int,int]) -> np.ndarray:
    out = np.empty_like(z_vec, dtype=np.int64)
    for z in np.unique(z_vec):
        zi = int(z)
        if zi not in z2idx:
            raise ValueError(f"Atomic number {zi} not in species map {sorted(z2idx.keys())}")
        out[z_vec == zi] = z2idx[zi]
    return out

def make_loss(kind: str) -> nn.Module:
    k = str(kind).lower()
    if k in ("l2","mse"):
        return nn.MSELoss(reduction="none")
    if k in ("l1","mae"):
        return nn.L1Loss(reduction="none")
    if k in ("huber","smoothl1"):
        return nn.SmoothL1Loss(beta=1.0, reduction="none")
    raise ValueError(f"Unknown loss type: {kind}")

# ------- masking & reductions (frame-equal weighting for mm_*) -------
def masked_sum_and_count(values: torch.Tensor, mask: torch.Tensor | None
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        return values.sum(), torch.tensor(values.numel(), device=values.device, dtype=values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    mask = (mask != 0).to(values.dtype)
    return (values*mask).sum(), mask.sum().clamp_min(1.0)

def masked_mean_per_frame(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if values.dim() == 1:
        return values.mean()
    B = values.shape[0]
    if mask is None:
        return values.reshape(B, -1).mean(dim=1).mean()

    m = mask
    while m.dim() < values.dim():
        m = m.unsqueeze(-1)
    m = (m != 0).to(values.dtype)

    reduce_dims = tuple(range(1, values.dim()))
    sums   = (values * m).sum(dim=reduce_dims)
    counts = m.sum(dim=reduce_dims)

    valid = counts > 0
    if not torch.any(valid):
        return torch.tensor(0.0, device=values.device, dtype=values.dtype)

    per = torch.zeros_like(sums)
    per[valid] = sums[valid] / counts[valid]
    return per[valid].mean()

def masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    s, n = masked_sum_and_count(values, mask)
    return s / n

def masked_rmse_from_residual(residual: torch.Tensor, mask: torch.Tensor | None,
                              *, frame_equal: bool) -> float:
    sq = residual * residual
    if frame_equal:
        m = masked_mean_per_frame(sq, mask)
    else:
        m = masked_mean(sq, mask)
    return float(torch.sqrt(m.detach()).cpu().item())

# ---------- rigid transforms ----------
def _rand_rot3x3(device: torch.device) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
    if torch.det(q) < 0:
        q[:, -1] = -q[:, -1]
    return q

def _center_on_qm(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    c = batch["qm_coords"].mean(dim=1, keepdim=True)
    batch["qm_coords"] = batch["qm_coords"] - c
    if "mm_coords" in batch:
        batch["mm_coords"] = batch["mm_coords"] - c
    return c

def _rotate_batch_in_place(batch: Dict[str, torch.Tensor], R: torch.Tensor) -> None:
    batch["qm_coords"] = batch["qm_coords"] @ R
    if "mm_coords" in batch:
        batch["mm_coords"] = batch["mm_coords"] @ R
    t = batch.get("targets", {})
    for k, v in list(t.items()):
        if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[-1] == 3:
            t[k] = v @ R

def apply_physical_transform(batch: Dict[str, torch.Tensor], *, do_center=True, do_rotate=True) -> None:
    if do_center:
        _center_on_qm(batch)
    if do_rotate:
        R = _rand_rot3x3(batch["qm_coords"].device)
        _rotate_batch_in_place(batch, R)

def deep_clone_cpu_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if k == "targets" and isinstance(v, dict):
            out["targets"] = {tk: tv.clone() if isinstance(tv, torch.Tensor) else tv for tk, tv in v.items()}
        else:
            out[k] = v.clone() if isinstance(v, torch.Tensor) else v
    return out

# --------- dataset ---------
class QMMMDataset(Dataset):
    def __init__(self, frames: List[Dict[str, np.ndarray]], device: torch.device, species_cfg: List[int] | None):
        if not frames:
            raise ValueError("Empty dataset")
        self.device = device
        keys = set().union(*[f.keys() for f in frames]) - {"__file__"}
        arr: Dict[str,np.ndarray] = {}
        for k in sorted(keys):
            vals = [np.asarray(fr[k]) for fr in frames if k in fr]
            vals = [v if v.ndim>0 else v.reshape(1) for v in vals]
            try:
                arr[k] = np.stack(vals, axis=0)
            except Exception as e:
                shapes = [tuple(v.shape) for v in vals]
                raise ValueError(f"Inconsistent shapes for '{k}': {shapes}") from e

        # Back-compat for input charges
        if "mm_Q" not in arr and "mm_charges" in arr:
            arr["mm_Q"] = arr["mm_charges"]

        if "qm_Z" not in arr:
            raise KeyError("Need 'qm_Z' for species mapping")

        z2idx, species_order = build_species_map_and_order(arr["qm_Z"][0], species_cfg)
        atom_types = np.zeros_like(arr["qm_Z"], dtype=np.int64)
        for i in range(arr["qm_Z"].shape[0]):
            atom_types[i] = z_to_types(arr["qm_Z"][i], z2idx)
        arr["atom_types"] = atom_types

        self.arr = arr
        self.n = arr["atom_types"].shape[0]
        self.species_order = species_order
        self.z2idx = z2idx

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Dict[str, Any]:
        dev = self.device
        def get(name: str) -> torch.Tensor:
            return to_tensor(self.arr[name][i], dev)
        sample: Dict[str,Any] = {}
        for k in self.arr.keys():
            t = get(k)
            if t.dim()==1 and ((k.endswith("_coords") or ("grad" in k)) and (t.numel()%3==0)):
                t = t.view(-1, 3)
            sample[k] = t
        targets: Dict[str,torch.Tensor] = {}
        for k,v in sample.items():
            if k in ("qm_coords","atom_types","qm_Z","mm_coords","mm_Q","mm_type"):
                continue
            targets[k] = v
        sample["targets"] = targets
        return sample

def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str,Any] = {}
    common = set(samples[0].keys())
    for s in samples[1:]:
        common &= set(s.keys())
    for k in sorted(common):
        if isinstance(samples[0][k], torch.Tensor):
            out[k] = torch.stack([s[k] for s in samples], dim=0)
        elif isinstance(samples[0][k], dict) and k=="targets":
            all_t = set().union(*[s["targets"].keys() for s in samples])
            targ: Dict[str,torch.Tensor] = {}
            for tk in sorted(all_t):
                if all(tk in s["targets"] for s in samples):
                    targ[tk] = torch.stack([s["targets"][tk] for s in samples], dim=0)
            out["targets"] = targ
    return out

# ---------- adaptive weighting ----------
class UncertaintyMTL(nn.Module):
    """exp(-s)*L + s; s = log(sigma^2) learned per loss key"""
    def __init__(self, keys):
        super().__init__()
        self.logvars = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in keys})
    def weight(self, key: str, mean_loss: torch.Tensor) -> torch.Tensor:
        lv = self.logvars[key]
        return torch.exp(-lv) * mean_loss + lv
    def sigmas(self) -> Dict[str,float]:
        return {k: float(torch.exp(0.5*v).detach().cpu().item()) for k,v in self.logvars.items()}

# ---------- sanitization: remove ANY padded-MM influence ----------
def _broadcast_mm_mask(m01: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    while m01.dim() < like.dim():
        m01 = m01.unsqueeze(-1)
    return m01.to(like.dtype)

def sanitize_padded_mm_inplace(batch: Dict[str, torch.Tensor]) -> None:
    if "mm_type" not in batch:
        return
    m01 = (batch["mm_type"] != 0)  # [B, Nmm] bool
    if "mm_coords" in batch and isinstance(batch["mm_coords"], torch.Tensor):
        batch["mm_coords"] = batch["mm_coords"] * _broadcast_mm_mask(m01, batch["mm_coords"])
    if "mm_Q" in batch and isinstance(batch["mm_Q"], torch.Tensor):
        batch["mm_Q"] = batch["mm_Q"] * _broadcast_mm_mask(m01, batch["mm_Q"])
    for k, v in list(batch.items()):
        if not isinstance(v, torch.Tensor):
            continue
        if k.startswith("mm_") and k not in ("mm_coords", "mm_Q", "mm_type"):
            batch[k] = v * _broadcast_mm_mask(m01, v)

# ---------- small helpers ----------
def _resolve_key(d: Dict[str, torch.Tensor], name: str) -> Optional[str]:
    if name in d:
        return name
    aliases = {
        "energy": ["dE"],
        "dE": ["energy"],
        "qm_grad": ["qm_dgrad", "qm_grad_high", "qm_grad_low"],
        "qm_dgrad": ["qm_grad", "qm_grad_high", "qm_grad_low"],
        "mm_grad": ["mm_dgrad", "mm_grad_high", "mm_grad_low"],
        "mm_dgrad": ["mm_grad", "mm_grad_high", "mm_grad_low"],
        "mm_espgrad_d": ["mm_espgrad", "mm_esp_grad"],
        "mm_espgrad": ["mm_espgrad_d", "mm_esp_grad"],
    }
    for alt in aliases.get(name, []):
        if alt in d:
            return alt
    return None

# ---------- eval (weighted same as training) ----------
def eval_split(
    loader, model, input_keys, criterions, device, lr_now, tag="val", rot_eval=False,
    center_inputs=True, pick_mask_fn=None, print_units="eV", print_keys: Optional[List[str]]=None,
    mm_equal_frame_weight: bool=False,
    *,
    base_weights: Dict[str, float],
    energy_keys: List[str],
    grad_keys: List[str],
    w_ene_base: float,
    w_grad_max: float,
    start_lr: float,
    mtl: Optional[nn.Module] = None,
):
    model.eval()

    def _normalize_shapes(p: torch.Tensor, t: torch.Tensor):
        if p.shape == t.shape:
            return p, t
        if p.dim()==3 and t.dim()==2 and p.shape[-1]==3 and t.shape[-1]==p.shape[-2]*3:
            return p, t.view(t.shape[0], p.shape[-2], 3)
        if t.dim()==3 and p.dim()==2 and t.shape[-1]==3 and p.shape[-1]==t.shape[-2]*3:
            return p.view(p.shape[0], t.shape[-2], 3), t
        if p.dim()==2 and p.shape[-1]==1 and t.dim()==1:
            return p.squeeze(-1), t
        if t.dim()==2 and t.shape[-1]==1 and p.dim()==1:
            return p, t.squeeze(-1)
        return p, t

    def _sched_weight(name: str, lr_now: float) -> float:
        r = float(lr_now / max(start_lr, 1e-12))
        w_grad = w_ene_base + (w_grad_max - w_ene_base) * r
        if (name in energy_keys) or any(fnmatch(name, p) for p in energy_keys):
            return w_ene_base
        if (name in grad_keys)   or any(fnmatch(name, p) for p in grad_keys):
            return w_grad
        return 1.0

    def _one_pass(cpu_batch: Dict[str,Any], *, do_center: bool) -> Tuple[float, Dict[str,float], Dict[str,float]]:
        work = deep_clone_cpu_batch(cpu_batch)
        if do_center:
            apply_physical_transform(work, do_center=True, do_rotate=False)
        batch = move_batch_to_device(work, device)
        sanitize_padded_mm_inplace(batch)

        inputs = [batch[k] for k in input_keys]
        with torch.enable_grad():
            out = model(*inputs)
        pred = out if isinstance(out, dict) else {}

        total = torch.tensor(0.0, device=device)
        raw_means: Dict[str,float] = {}
        rmses: Dict[str,float] = {}

        for name, crit in criterions.items():
            pk = _resolve_key(pred, name)
            tk = _resolve_key(batch["targets"], name)
            if pk is None or tk is None:
                continue
            p, t = _normalize_shapes(pred[pk], batch["targets"][tk])
            raw = crit(p, t)
            mask = pick_mask_fn(name, batch) if pick_mask_fn is not None else None

            if mm_equal_frame_weight and name.startswith("mm_"):
                mean = masked_mean_per_frame(raw, mask)
                rmse = masked_rmse_from_residual(p - t, mask, frame_equal=True)
            else:
                mean = masked_mean(raw, mask)
                rmse = masked_rmse_from_residual(p - t, mask, frame_equal=False)

            bw = base_weights.get(name, 1.0)
            sw = _sched_weight(name, lr_now)
            val_term = (mtl.weight(name, mean) if mtl is not None else mean) * bw * sw

            total = total + val_term
            raw_means[name] = float(mean.detach().cpu().item())
            rmses[name] = rmse

        return float(total.detach().cpu().item()), raw_means, rmses

    steps = 0
    total_loss_sum = 0.0
    raw_acc   = {k: 0.0 for k in criterions.keys()}
    rmse_acc  = {k: 0.0 for k in criterions.keys()}

    for cpu_batch in loader:
        tl, rm, rr = _one_pass(cpu_batch, do_center=center_inputs)
        total_loss_sum += tl; steps += 1
        for k in rm:
            raw_acc[k]  += rm[k]
            rmse_acc[k] += rr[k]

    total_loss_avg = total_loss_sum / max(1, steps)
    per_key_mean = {k: raw_acc[k] / max(1, steps) for k in raw_acc}
    per_key_rmse = {k: rmse_acc[k] / max(1, steps) for k in rmse_acc}

    per_key_mean_rot, per_key_rmse_rot = {}, {}
    if rot_eval:
        raw_acc_r  = {k: 0.0 for k in criterions.keys()}
        rmse_acc_r = {k: 0.0 for k in criterions.keys()}
        steps_r = 0
        for cpu_batch in loader:
            rot = deep_clone_cpu_batch(cpu_batch)
            apply_physical_transform(rot, do_center=True, do_rotate=True)
            tl, rm, rr = _one_pass(rot, do_center=False)
            steps_r += 1
            for k in rm:
                raw_acc_r[k]  += rm[k]
                rmse_acc_r[k] += rr[k]
        per_key_mean_rot = {k: raw_acc_r[k]  / max(1, steps_r) for k in raw_acc_r}
        per_key_rmse_rot = {k: rmse_acc_r[k] / max(1, steps_r) for k in rmse_acc_r}

    return total_loss_avg, per_key_mean, per_key_rmse, per_key_mean_rot, per_key_rmse_rot

# ---------------- training ----------------
def run(cfg: Dict[str, Any]) -> None:
    trcfg = cfg.get("trainer", {})

    # [progress] configure stdout line buffering & optional tee file
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    progress_txt = str(trcfg.get("progress_txt", "") or "")
    if not progress_txt and bool(trcfg.get("save_progress", False)):
        out_guess = str(trcfg.get("out_dir", "")) or "runs"
        progress_txt = os.path.join(out_guess, "progress.txt")
    tee = _ProgressTee(progress_txt if progress_txt else None)

    # threading
    if "torch_num_threads" in trcfg:
        torch.set_num_threads(int(trcfg["torch_num_threads"]))
    if "torch_interop_threads" in trcfg:
        torch.set_num_interop_threads(int(trcfg["torch_interop_threads"]))

    device = torch.device(trcfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(trcfg.get("seed", 42))
    set_seed(seed)

    # output dirs
    save_path = str(trcfg.get("save_path", "runs/ckpt.pt"))
    out_dir = str(trcfg.get("out_dir") or os.path.dirname(save_path) or f"runs/{time.strftime('%Y%m%d-%H%M%S')}")
    plots_dir = os.path.join(out_dir, "plots")
    logs_dir  = os.path.join(out_dir, "logs")
    os.makedirs(out_dir, exist_ok=True); os.makedirs(plots_dir, exist_ok=True); os.makedirs(logs_dir, exist_ok=True)

    # printing options
    print_units  = str(trcfg.get("print_units", "eV"))
    print_detail = str(trcfg.get("print_detail", "short")).lower()
    print_keys: Optional[List[str]] = trcfg.get("print_keys", None)

    # equalize MM frames (each frame equal weight for mm_* losses)
    mm_equal_frame_weight = bool(trcfg.get("mm_equal_frame_weight", True))

    # timing/meta
    wall_start = time.time()
    per_epoch_seconds: List[float] = []
    best_epoch: int | None = None

    # data
    reader = get_adapter(cfg["adapter"])
    frames: List[Dict[str,np.ndarray]] = reader(cfg["input"], **dict(cfg.get("adapter_kwargs", {})))
    if not isinstance(frames, list) or not frames:
        tee.log("Adapter returned no frames")
        raise RuntimeError("Adapter returned no frames")

    dataset = QMMMDataset(frames, device=device, species_cfg=trcfg.get("species"))
    n_total = len(dataset)
    val_frac  = float(trcfg.get("val_fraction", 0.2))
    test_frac = float(trcfg.get("test_fraction", 0.1))
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")
    n_test = max(1, int(round(n_total*test_frac)))
    n_val  = max(1, int(round(n_total*val_frac)))
    n_train = max(1, n_total - n_val - n_test)
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    bs = int(trcfg.get("batch_size", 16))
    nw = int(trcfg.get("num_workers", 0))
    pin = bool(trcfg.get("pin_memory", True))
    pw  = bool(nw > 0) and bool(trcfg.get("persistent_workers", False))
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=pin,
                              persistent_workers=pw, collate_fn=collate_batch)
    val_loader   = DataLoader(val_set,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin,
                              persistent_workers=pw, collate_fn=collate_batch)
    test_loader  = DataLoader(test_set,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin,
                              persistent_workers=pw, collate_fn=collate_batch)

    # model
    mcfg = cfg["model"]
    module = importlib.import_module(mcfg["module"])
    ModelClass = getattr(module, mcfg["class"])
    model: nn.Module = ModelClass(cfg).to(device)
    if torch.cuda.device_count() > 1 and bool(trcfg.get("dataparallel", False)):
        model = torch.nn.DataParallel(model)
    real_model = model.module if isinstance(model, nn.DataParallel) else model

    # param stats
    initial_stats = {
        "parameters_total": sum(p.numel() for p in real_model.parameters()),
        "parameters_trainable": sum(p.numel() for p in real_model.parameters() if p.requires_grad),
        "buffers": sum(b.numel() for b in real_model.buffers()),
    }

    input_keys: List[str] = list(mcfg.get("inputs", []))
    if not input_keys:
        raise ValueError("model.inputs must list batch keys for forward(...)")

    # optim
    ocfg = cfg.get("optim", {})
    lr0 = float(ocfg.get("lr", 5e-4))
    wd  = float(ocfg.get("weight_decay", 0.0))

    # losses & weights
    losses_cfg: Dict[str, Dict[str, Any]] = cfg.get("losses", {})
    criterions: Dict[str, nn.Module] = {name: make_loss(spec.get("type", "mse"))
                                        for name, spec in losses_cfg.items()}
    base_weights: Dict[str, float] = {name: float(spec.get("weight", 1.0))
                                      for name, spec in losses_cfg.items()}

    # optional uncertainty weighting
    use_uncert = bool(cfg.get("weighting", {}).get("uncertainty", True))
    mtl = UncertaintyMTL(keys=list(criterions.keys())).to(device) if use_uncert else None

    # optimizer (include MTL params if used)
    params = list(real_model.parameters()) + (list(mtl.parameters()) if mtl is not None else [])
    optimizer = torch.optim.Adam(params, lr=lr0, weight_decay=wd)
    start_lr = lr0

    # scheduler
    sched_type = str(ocfg.get("schedule", "plateau")).lower()
    scheduler = None
    start_lr = lr0
    exp_min_lr = None
    mix_state = None

    if sched_type == "onecycle":
        max_lr = float(ocfg.get("max_lr", lr0 * 5.0))
        pct_start = float(ocfg.get("pct_start", 0.3))
        div_factor = float(ocfg.get("div_factor", 25.0))
        final_div = float(ocfg.get("final_div_factor", 1e4))
        anneal = str(ocfg.get("anneal", "cos")).lower()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=int(trcfg.get("epochs", 500)),
            steps_per_epoch=max(1, len(train_loader)),
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div,
            anneal_strategy=("cos" if anneal.startswith("cos") else "linear"),
        )
        start_lr = max_lr / div_factor

    elif sched_type == "exponential":
        exp_cfg = ocfg.get("exponential", {})
        gamma = float(exp_cfg.get("factor", 0.99))
        exp_min_lr = float(exp_cfg.get("min_lr", 1e-6))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    elif sched_type == "mixed":
        mx = ocfg.get("mixed", {})
        gamma = float(mx.get("exp_factor", 0.99))
        exp_min_lr = float(mx.get("min_lr", 1e-6))
        mix_patience = int(mx.get("patience", 8))
        mix_threshold = float(mx.get("threshold", 1e-4))
        mix_plat_factor = float(mx.get("plat_factor", 0.5))
        mix_cooldown_default = int(mx.get("cooldown", 0))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        mix_state = {
            "best": float("inf"),
            "bad_epochs": 0,
            "cooldown": 0,
            "patience": mix_patience,
            "threshold": mix_threshold,
            "plat_factor": mix_plat_factor,
            "cooldown_default": mix_cooldown_default,
        }

    else:  # plateau
        plat_cfg = ocfg.get("plateau", {"factor":0.5, "patience":10, "min_lr":1e-6, "threshold":1e-4})
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(plat_cfg.get("factor", 0.5)),
            patience=int(plat_cfg.get("patience", 10)),
            threshold=float(plat_cfg.get("threshold", 1e-4)),
            min_lr=float(plat_cfg.get("min_lr", 1e-6)),
        )

    def _clamp_min_lr(opt: torch.optim.Optimizer, min_lr: float) -> None:
        for g in opt.param_groups:
            if g["lr"] < min_lr:
                g["lr"] = min_lr

    # masks
    masks_cfg: Dict[str, str] = cfg.get("masks", {})
    def pick_mask(name: str, batch: Dict[str,Any]) -> torch.Tensor | None:
        if name in masks_cfg and masks_cfg[name] in batch:
            return batch[masks_cfg[name]]
        for pat, mk in masks_cfg.items():
            if ("*" in pat or "?" in pat or "[" in pat) and fnmatch(name, pat) and mk in batch:
                return batch[mk]
        if name.startswith("mm_") and "mm_type" in batch:
            return (batch["mm_type"] != 0)
        return None

    # LR-scheduled group weights
    wcfg = cfg.get("weight_schedule", {
        "energy_keys": ["dE", "energy"],
        "grad_keys":   ["qm_*", "mm_*"],
        "w_ene": 1.0,
        "w_grad_max": 100.0
    })
    energy_keys = list(wcfg.get("energy_keys", ["dE", "energy"]))
    grad_keys   = list(wcfg.get("grad_keys",   ["qm_*", "mm_*"]))
    w_ene_base  = float(wcfg.get("w_ene", 1.0))
    w_grad_max  = float(wcfg.get("w_grad_max", 100.0))

    # AMP
    amp_mode_cfg = str(trcfg.get("amp", "off")).lower()
    amp_mode = "off"
    amp_dtype = None
    use_amp = False
    if device.type == "cuda":
        if amp_mode_cfg == "bf16" and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            amp_mode, amp_dtype, use_amp = "bf16", torch.bfloat16, True
        elif amp_mode_cfg == "fp16":
            amp_mode, amp_dtype, use_amp = "fp16", torch.float16, True
    else:
        if amp_mode_cfg in ("bf16","fp16"):
            print("[warn] AMP requested on non-CUDA device → disabling AMP.")
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_mode == "fp16"))

    # ckpt & early stop
    ckpt_path = save_path if save_path.endswith(".pt") else os.path.join(out_dir, "best.pt")
    best_val = float("inf")
    epochs = int(trcfg.get("epochs", 500))
    es_pat = int(trcfg.get("early_stop_patience", 20))
    es_min_delta = float(trcfg.get("early_stop_min_delta", 1e-4))
    no_improve = 0

    # [script_every] config
    save_script_every = int(trcfg.get("save_script_every", 0))  # 0 disables

    # ---------- TGW CONFIG (Optional; disabled by default) ----------
    tgw_cfg = cfg.get("weighting", {}).get("threshold_gate", {}) or {}
    tgw_enabled = bool(tgw_cfg.get("enabled", False))
    tgw_metric_kind = str(tgw_cfg.get("metric", "rmse")).lower()   # "rmse" | "mean"
    tgw_units_kind  = str(tgw_cfg.get("units", "eV")).lower()      # "eV" | "kcal"
    tgw_threshold_items = [(pat, float(val)) for pat, val in (tgw_cfg.get("thresholds", {}) or {}).items()]
    if tgw_units_kind.startswith("kcal"):
        tgw_threshold_items = [(pat, val / EV_TO_KCALMOL) for pat, val in tgw_threshold_items]
    tgw_check_every = int(tgw_cfg.get("check_every", 5))
    tgw_patience    = int(tgw_cfg.get("patience", 1))
    tgw_up_factor   = float(tgw_cfg.get("up_factor", 1.30))
    tgw_down_factor = float(tgw_cfg.get("down_factor", 0.85))
    tgw_min_gate    = float(tgw_cfg.get("min_w", 0.10))
    tgw_max_gate    = float(tgw_cfg.get("max_w", 10.0))
    tgw_hysteresis  = float(tgw_cfg.get("hysteresis", 0.95))
    tgw_ema_alpha   = float(tgw_cfg.get("ema", 0.0))
    tgw_use_val     = bool(tgw_cfg.get("use_val", True))
    tgw_renorm_avg1 = bool(tgw_cfg.get("renorm", True))

    # Activation / deactivation controls
    tgw_activate_any = bool(tgw_cfg.get("activate_when_any_met", True))  # start TGW once any key is ≤ thr
    tgw_deactivate_all_met      = bool(tgw_cfg.get("deactivate_when_all_met", True))  # stop TGW when all keys are ≤ thr
    tgw_deactivate_scale        = float(tgw_cfg.get("deactivate_scale", 1.0))         # scale×thr for "all-met" check
    tgw_reset_on_deactivate     = bool(tgw_cfg.get("reset_weights_on_deactivate", True))  # reset gates→1.0 on stop
    tgw_freeze_after_deactivate = bool(tgw_cfg.get("freeze_after_deactivate", True))      # keep TGW off permanently

    tgw_active = False
    tgw_frozen = False  # once frozen, TGW updates won't run again this training

    tgw_gate_weights: Dict[str, float] = {k: 1.0 for k in criterions.keys()}
    tgw_gate_state = {k: {"ema": None, "above": 0, "below": 0} for k in criterions.keys()}

    def _tgw_threshold_for(key_name: str) -> float | None:
        for pat, thr in tgw_threshold_items:
            if key_name == pat or fnmatch(key_name, pat):
                return thr
        return None

    # --- CSV logs (eV units) ---
    csv_path = os.path.join(logs_dir, "train_val_metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            cols = ["epoch", "split", "lr", "total_loss"]
            for k in sorted(criterions.keys()):
                cols += [f"loss_{k}", f"rmse_{k}", f"weight_{k}"]
            if use_uncert:
                for k in sorted(criterions.keys()):
                    cols.append(f"sigma_{k}")
            f.write(",".join(cols) + "\n")

    # helpers
    def normalize_shapes(pred: torch.Tensor, targ: torch.Tensor):
        if pred.shape == targ.shape:
            return pred, targ
        if pred.dim()==3 and targ.dim()==2 and pred.shape[-1]==3 and targ.shape[-1]==pred.shape[-2]*3:
            return pred, targ.view(targ.shape[0], pred.shape[-2], 3)
        if targ.dim()==3 and pred.dim()==2 and targ.shape[-1]==3 and pred.shape[-1]==targ.shape[-2]*3:
            return pred.view(pred.shape[0], targ.shape[-2], 3), targ
        if pred.dim()==2 and pred.shape[-1]==1 and targ.dim()==1:
            return pred.squeeze(-1), targ
        if targ.dim()==2 and targ.shape[-1]==1 and pred.dim()==1:
            return pred, targ.squeeze(-1)
        raise ValueError(f"Shape mismatch: pred {tuple(pred.shape)} vs targ {tuple(targ.shape)}")

    def dynamic_group_weight(name: str, lr_now: float) -> float:
        r = float(lr_now / max(start_lr, 1e-12))
        w_grad = w_ene_base + (w_grad_max - w_ene_base) * r
        if name in energy_keys or any(fnmatch(name, p) for p in energy_keys):
            return w_ene_base
        if name in grad_keys   or any(fnmatch(name, p) for p in grad_keys):
            return w_grad
        return 1.0

    def compute_losses(pred: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                       batch: Dict[str,Any], lr_now: float
                       ) -> Tuple[torch.Tensor, Dict[str,float], Dict[str,float], Dict[str,float]]:
        total = torch.tensor(0.0, device=device)
        raw_means: Dict[str,float] = {}
        rmses: Dict[str,float] = {}
        eff_w: Dict[str,float] = {}

        for name, crit in criterions.items():
            pk = _resolve_key(pred, name)
            tk = _resolve_key(targets, name)
            if pk is None or tk is None:
                continue
            p, t = normalize_shapes(pred[pk], targets[tk])
            raw = crit(p, t)
            mask = pick_mask(name, batch)

            if mm_equal_frame_weight and name.startswith("mm_"):
                mean = masked_mean_per_frame(raw, mask)
                rmse = masked_rmse_from_residual(p - t, mask, frame_equal=True)
            else:
                mean = masked_mean(raw, mask)
                rmse = masked_rmse_from_residual(p - t, mask, frame_equal=False)

            base_w = base_weights.get(name, 1.0)
            sched_w = dynamic_group_weight(name, lr_now)
            # training-only additional multiplicative gate
            gate_w = tgw_gate_weights.get(name, 1.0) if tgw_enabled else 1.0
            val = (mtl.weight(name, mean) if mtl is not None else mean) * base_w * sched_w * gate_w

            total = total + val
            raw_means[name] = float(mean.detach().cpu().item())
            rmses[name] = rmse
            eff_w[name] = base_w * sched_w * gate_w
        return total, raw_means, rmses, eff_w

    # --- save NPZ test pack & metadata ---
    test_idx = np.array(getattr(test_set, "indices", []), dtype=np.int64)
    npz_path = os.path.join(out_dir, "test_data.npz")
    save_pack = {}
    for k, arr in dataset.arr.items():
        try:
            save_pack[k] = arr[test_idx]
        except Exception as e:
            print(f"[warn] could not save key {k}: {type(e).__name__}: {e}")
    np.savez_compressed(npz_path, **save_pack)

    meta_io = {
        "input_keys": input_keys,
        "loss_keys": list(criterions.keys()),
        "masks": masks_cfg,
        "species": trcfg.get("species"),
        "amp": amp_mode,
        "device": str(device),
    }
    with open(os.path.join(out_dir, "model_io.json"), "w") as f:
        json.dump(meta_io, f, indent=2)

    # --- train/val controls ---
    rot_aug = bool(trcfg.get("rot_aug", True))
    center_inputs = bool(trcfg.get("center", True))
    val_rot_eval = bool(trcfg.get("val_rot_eval", False))
    test_rot_eval = bool(trcfg.get("test_rot_eval", False))
    max_norm = float(trcfg.get("grad_clip", 5.0))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # --- train loop ---
    for epoch in range(1, epochs+1):
        epoch_t0 = time.time()

        real_model.train()
        tr_loss_sum, tr_steps = 0.0, 0
        tr_raw_acc: Dict[str,float] = {k:0.0 for k in criterions.keys()}
        tr_rmse_acc: Dict[str,float] = {k:0.0 for k in criterions.keys()}
        tr_seen: Dict[str,int] = {k:0 for k in criterions.keys()}
        clip_hits = 0
        last_total_norm = float("nan")

        for cpu_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            work = deep_clone_cpu_batch(cpu_batch)
            if center_inputs or rot_aug:
                apply_physical_transform(work, do_center=center_inputs, do_rotate=rot_aug)

            batch = move_batch_to_device(work, device)
            sanitize_padded_mm_inplace(batch)

            inputs = [batch[k] for k in input_keys]

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True):
                    out = real_model(*inputs)
                    pred = out if isinstance(out, dict) else {}
                    loss, raw_logs, rmses, _ = compute_losses(pred, batch["targets"], batch, optimizer.param_groups[0]["lr"])
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    params_to_clip = [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad]
                    total_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm)
                    last_total_norm = float(total_norm.detach().cpu().item())
                    if last_total_norm > max_norm:
                        clip_hits += 1
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    params_to_clip = [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad]
                    total_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm)
                    last_total_norm = float(total_norm.detach().cpu().item())
                    if last_total_norm > max_norm:
                        clip_hits += 1
                    optimizer.step()
            else:
                out = real_model(*inputs)
                pred = out if isinstance(out, dict) else {}
                loss, raw_logs, rmses, _ = compute_losses(pred, batch["targets"], batch, optimizer.param_groups[0]["lr"])
                loss.backward()
                params_to_clip = [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad]
                total_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm)
                last_total_norm = float(total_norm.detach().cpu().item())
                if last_total_norm > max_norm:
                    clip_hits += 1
                optimizer.step()

            if sched_type == "onecycle":
                scheduler.step()

            tr_loss_sum += float(loss.detach().cpu().item()); tr_steps += 1
            for k,v in raw_logs.items():
                tr_raw_acc[k] += v; tr_rmse_acc[k] += rmses[k]; tr_seen[k] += 1

        tr_loss = tr_loss_sum / max(1,tr_steps)
        tr_logs_avg = {k: tr_raw_acc[k]/max(1,tr_seen[k]) for k in tr_raw_acc}
        tr_rmse_avg = {k: tr_rmse_acc[k]/max(1,tr_seen[k]) for k in tr_rmse_acc}

        # --- validation ---
        va_loss, va_logs_avg, va_rmse_avg, va_logs_avg_rot, va_rmse_avg_rot = eval_split(
            val_loader, real_model, input_keys, criterions, device,
            lr_now=optimizer.param_groups[0]["lr"], tag="val", rot_eval=val_rot_eval, center_inputs=center_inputs,
            pick_mask_fn=pick_mask, print_units=print_units, print_keys=print_keys,
            mm_equal_frame_weight=mm_equal_frame_weight,
            base_weights=base_weights, energy_keys=energy_keys, grad_keys=grad_keys,
            w_ene_base=w_ene_base, w_grad_max=w_grad_max, start_lr=start_lr, mtl=mtl
        )

        if sched_type != "onecycle":
            if sched_type == "plateau":
                scheduler.step(va_loss)
            elif sched_type == "exponential":
                scheduler.step()
                if exp_min_lr is not None:
                    _clamp_min_lr(optimizer, exp_min_lr)
            elif sched_type == "mixed":
                scheduler.step()
                if exp_min_lr is not None:
                    _clamp_min_lr(optimizer, exp_min_lr)
                ms = mix_state
                improved_enough = (ms["best"] - va_loss) >= ms["threshold"]
                if improved_enough:
                    ms["best"] = va_loss
                    ms["bad_epochs"] = 0
                    if ms["cooldown"] > 0:
                        ms["cooldown"] = 0
                else:
                    if ms["cooldown"] > 0:
                        ms["cooldown"] -= 1
                    else:
                        ms["bad_epochs"] += 1
                        if ms["bad_epochs"] >= ms["patience"]:
                            for g in optimizer.param_groups:
                                g["lr"] = max(exp_min_lr if exp_min_lr is not None else 0.0,
                                              g["lr"] * ms["plat_factor"])
                            tee.log(f"[lr-mix] plateau fallback: lr→{optimizer.param_groups[0]['lr']:.3e} "
                                    f"(factor={ms['plat_factor']}, bad_epochs={ms['bad_epochs']})")
                            ms["bad_epochs"] = 0
                            ms["cooldown"] = ms["cooldown_default"]

        # ---------- TGW UPDATE (periodic, training-only) ----------
        if tgw_enabled and (not tgw_frozen) and (epoch % tgw_check_every == 0):
            metric_map = (va_rmse_avg if tgw_metric_kind == "rmse" else va_logs_avg) if tgw_use_val \
                         else (tr_rmse_avg if tgw_metric_kind == "rmse" else tr_logs_avg)

            # Activation: begin TGW once any watched key is within threshold
            if tgw_activate_any and not tgw_active:
                for key_name in criterions.keys():
                    thr = _tgw_threshold_for(key_name)
                    if thr is None:
                        continue
                    mval = float(metric_map.get(key_name, float('nan')))
                    if not math.isnan(mval) and (mval <= thr):
                        tgw_active = True
                        tee.log(f"[tgw] activation triggered at epoch {epoch} (≥1 key within threshold)")
                        break

            # Auto-deactivate: stop TGW when ALL watched keys are ≤ scale×thr
            if tgw_deactivate_all_met:
                all_watched = []
                all_met = True
                for key_name in criterions.keys():
                    thr = _tgw_threshold_for(key_name)
                    if thr is None:
                        continue
                    mval = float(metric_map.get(key_name, float('nan')))
                    if math.isnan(mval):
                        all_met = False
                        continue
                    if not (mval <= thr * tgw_deactivate_scale):
                        all_met = False
                    all_watched.append(key_name)

                if all_watched and all_met:
                    tgw_active = False
                    if tgw_reset_on_deactivate:
                        for k in tgw_gate_weights:
                            tgw_gate_weights[k] = 1.0
                    msg = "[tgw] all targets ≤ thresholds → TGW deactivated"
                    if tgw_freeze_after_deactivate:
                        tgw_frozen = True
                        msg += " (frozen)"
                    tee.log(msg)

            # Per-key updates only when TGW is active (and not frozen)
            if tgw_active and (not tgw_frozen):
                changes = []
                for key_name in sorted(criterions.keys()):
                    thr = _tgw_threshold_for(key_name)
                    if thr is None:
                        continue
                    mval = float(metric_map.get(key_name, float('nan')))
                    if math.isnan(mval):
                        continue
                    st = tgw_gate_state[key_name]

                    # EMA smoothing (optional)
                    if tgw_ema_alpha > 0.0:
                        st["ema"] = mval if (st["ema"] is None) else (tgw_ema_alpha*mval + (1.0 - tgw_ema_alpha)*st["ema"])
                        m_eff = float(st["ema"])
                    else:
                        m_eff = mval

                    # Patience + hysteresis logic
                    if m_eff > thr:
                        st["above"] += 1; st["below"] = 0
                        if st["above"] >= tgw_patience:
                            neww = min(tgw_max_gate, tgw_gate_weights[key_name] * tgw_up_factor)
                            if neww != tgw_gate_weights[key_name]:
                                tgw_gate_weights[key_name] = neww
                                changes.append((key_name, m_eff, thr, neww, "↑"))
                            st["above"] = 0
                    elif m_eff < (tgw_hysteresis * thr):
                        st["below"] += 1; st["above"] = 0
                        if st["below"] >= tgw_patience:
                            neww = max(tgw_min_gate, tgw_gate_weights[key_name] * tgw_down_factor)
                            if neww != tgw_gate_weights[key_name]:
                                tgw_gate_weights[key_name] = neww
                                changes.append((key_name, m_eff, thr, neww, "↓"))
                            st["below"] = 0
                    else:
                        st["above"] = 0; st["below"] = 0

                # Optional re-normalization so the average gate stays ~1.0
                if tgw_renorm_avg1 and tgw_gate_weights:
                    avgw = sum(tgw_gate_weights.values()) / len(tgw_gate_weights)
                    if avgw > 0:
                        s = 1.0 / avgw
                        for kk in tgw_gate_weights:
                            tgw_gate_weights[kk] *= s

                for kname, m_eff, thr, gw, arrow in changes:
                    tee.log(f"[tgw]{arrow} {kname}: metric={m_eff:.3e} thr={thr:.3e} gate→{gw:.3f}")

        # --- CSV logs ---
        def write_row(split: str, total_loss: float, raw_avg: Dict[str,float], rmse_avg: Dict[str,float]):
            with open(csv_path, "a") as f:
                line = [str(epoch), split,
                        f"{optimizer.param_groups[0]['lr']:.8e}",
                        f"{total_loss:.8e}"]
                for k in sorted(criterions.keys()):
                    r = float(optimizer.param_groups[0]['lr'] / max(start_lr, 1e-12))
                    w_grad = w_ene_base + (w_grad_max - w_ene_base) * r
                    basew = base_weights.get(k,1.0)
                    schedw = (w_ene_base if (k in energy_keys or any(fnmatch(k,p) for p in energy_keys))
                              else (w_grad if (k in grad_keys or any(fnmatch(k,p) for p in grad_keys)) else 1.0))
                    gatew = tgw_gate_weights.get(k, 1.0) if tgw_enabled else 1.0
                    effw = basew * schedw * gatew
                    line.append(f"{raw_avg.get(k, float('nan')):.8e}")
                    line.append(f"{rmse_avg.get(k, float('nan')):.8e}")
                    line.append(f"{effw:.8e}")
                if mtl is not None:
                    sig = mtl.sigmas()
                    for k in sorted(criterions.keys()):
                        line.append(f"{sig.get(k, float('nan')):.8e}")
                f.write(",".join(line) + "\n")

        write_row("train", tr_loss, tr_logs_avg, tr_rmse_avg)
        write_row("val",   va_loss, va_logs_avg, va_rmse_avg)
        if val_rot_eval:
            write_row("val_rot", va_loss, va_logs_avg_rot or va_logs_avg, va_rmse_avg_rot or va_rmse_avg)

        # --- pretty console+file line ---
        def _fmt_key(k: str, rmse_map: Dict[str,float], mean_map: Dict[str,float]) -> str:
            mul = _unit_multiplier_for_key(k, print_units)
            rmse = float(rmse_map.get(k, float('nan')))
            mean = float(mean_map.get(k, float('nan')))
            rmse_u = rmse * mul if not math.isnan(rmse) else float('nan')
            mean_u = mean * mul if not math.isnan(mean) else float('nan')
            unit_tag = "kcal/mol" if print_units.lower().startswith("kcal") else "eV"
            if "grad" in k.lower():
                unit_tag += "(grad)"
            if print_detail == "full":
                return f"{k}:RMSE={rmse_u:.5f} {unit_tag} (mean={mean_u:.3e})"
            else:
                return f"{k}:RMSE={rmse_u:.5f} {unit_tag}"

        shown_keys = sorted(criterions.keys()) if not print_keys else [k for k in sorted(criterions.keys()) if k in print_keys]
        parts = [f"[{epoch:03d}/{epochs}] lr={optimizer.param_groups[0]['lr']:.3e}",
                 f"train={tr_loss:.6f}", f"val={va_loss:.6f}"]
        parts += [_fmt_key(k, va_rmse_avg, va_logs_avg) for k in shown_keys]
        if clip_hits > 0:
            parts.append(f"[grad-clip hits={clip_hits}, last_total_norm={last_total_norm:.2f} > {max_norm}]")
        if mtl is not None and print_detail == "full":
            sig = mtl.sigmas()
            parts += [f"{k}_sigma={sig[k]:.3e}" for k in sorted(sig.keys())]
        tee.log("  ".join(parts))

        # --- early stopping & checkpoint ---
        improved = va_loss + es_min_delta < best_val
        if improved:
            best_val = va_loss
            best_epoch = epoch
            no_improve = 0
            if ckpt_path:
                torch.save({
                    "epoch": epoch,
                    "model_state": real_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "cfg": cfg
                }, ckpt_path)
            tee.log(f"[epoch {epoch}] decision: improved → checkpoint saved")
        else:
            no_improve += 1
            tee.log(f"[epoch {epoch}] decision: no_improve={no_improve}/{es_pat}")

        # [script_every] optional periodic TorchScript export (inference-ready)
        if save_script_every > 0 and (epoch % save_script_every == 0):
            real_model.eval()
            try:
                scripted = torch.jit.script(real_model)
                script_epoch_path = os.path.join(out_dir, f"model_script_ep{epoch:04d}.pt")
                scripted.save(script_epoch_path)
                tee.log(f"[epoch {epoch}] saved TorchScript: {script_epoch_path}")
            except Exception as e:
                tee.log(f"[warn] TorchScript export (epoch {epoch}) failed: {type(e).__name__}: {e}")

        per_epoch_seconds.append(time.time() - epoch_t0)
        if no_improve >= es_pat:
            tee.log(f"[EarlyStop] no improvement in {es_pat} epochs (best={best_val:.6f}). Stopping.")
            break

    # --- reload best and TEST ---
    if os.path.exists(ckpt_path):
        tee.log(f"[test] reloading best checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        real_model.load_state_dict(ckpt["model_state"])

    test_loss, test_metrics, test_rmse, test_rot_metrics, test_rot_rmse = eval_split(
        test_loader, real_model, input_keys, criterions, device,
        lr_now=optimizer.param_groups[0]["lr"], tag="test",
        rot_eval=test_rot_eval, center_inputs=bool(trcfg.get("center", True)),
        pick_mask_fn=pick_mask, print_units=print_units, print_keys=print_keys,
        mm_equal_frame_weight=mm_equal_frame_weight,
        base_weights=base_weights, energy_keys=energy_keys, grad_keys=grad_keys,
        w_ene_base=w_ene_base, w_grad_max=w_grad_max, start_lr=start_lr, mtl=mtl
    )

    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump({"loss": test_loss, "per_key": test_metrics, "per_key_rmse": test_rmse}, f, indent=2)
    if test_rot_eval:
        with open(os.path.join(out_dir, "test_metrics_rot.json"), "w") as f:
            json.dump({"per_key": test_rot_metrics, "per_key_rmse": test_rot_rmse}, f, indent=2)

    # --- save model (state_dict + TorchScript if possible) ---
    torch.save(real_model.state_dict(), os.path.join(out_dir, "model_state.pt"))
    real_model.eval()
    script_path = os.path.join(out_dir, "model_script.pt")
    try:
        scripted = torch.jit.script(real_model)
        scripted.save(script_path)
    except Exception as e:
        print(f"[warn] TorchScript export failed: {type(e).__name__}: {e}")
        script_path = None

    with open(os.path.join(out_dir, "config_used.yml"), "w") as f:
        yaml.safe_dump(cfg, f)

    # ---------- training_summary.json ----------
    wall_seconds = time.time() - wall_start
    peak_mem = None
    if device.type == "cuda":
        try:
            peak_mem = int(torch.cuda.max_memory_allocated(device))
        except Exception:
            peak_mem = None

    def _file_size(path: Optional:str) -> int:
        if not path: return -1
        try:    return os.path.getsize(path)
        except: return -1

    sizes = {
        "model_script.pt": _file_size(script_path),
        "model_state.pt": _file_size(os.path.join(out_dir, "model_state.pt")),
        "best.pt": _file_size(ckpt_path),
        "test_data.npz": _file_size(npz_path),
        "model_io.json": _file_size(os.path.join(out_dir, "model_io.json")),
        "train_val_metrics.csv": _file_size(os.path.join(logs_dir, "train_val_metrics.csv")),
    }

    sched_label = {
        "onecycle": "OneCycleLR",
        "exponential": "ExponentialLR",
        "mixed": "Mixed(Exp+Plateau)",
        "plateau": "ReduceLROnPlateau",
    }.get(sched_type, sched_type)

    summary = {
        "host": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": torch.cuda.device_count(),
            "device": str(device),
        },
        "run": {
            "out_dir": out_dir,
            "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(wall_start)),
            "wall_seconds_total": wall_seconds,
            "epochs_ran": len(per_epoch_seconds),
            "per_epoch_seconds": per_epoch_seconds,
            "batch_size": bs,
            "num_workers": nw,
            "amp_mode": amp_mode,
            "grad_clip": float(trcfg.get("grad_clip", 5.0)),
            "rot_aug": bool(trcfg.get("rot_aug", True)),
            "center_inputs": bool(trcfg.get("center", True)),
            "val_frac": val_frac,
            "test_frac": test_frac,
            "dataset_sizes": {"train": len(train_set), "val": len(val_set), "test": len(test_set), "total": n_total},
            "peak_cuda_bytes": peak_mem,
        },
        "model": {
            "class": f"{mcfg.get('module')}::{mcfg.get('class')}",
            "param_counts": initial_stats,
        },
        "optimizer": {
            "type": "Adam",
            "lr_start": start_lr,
            "weight_decay": wd,
            "scheduler": {"type": sched_label},
        },
        "best": {
            "epoch": best_epoch,
            "val_loss": best_val,
            "test_loss": test_loss,
        },
        "artifacts_bytes": sizes,
    }
    with open(os.path.join(out_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # explicit DataLoader cleanup to avoid hang with persistent_workers
    try:
        for _ldr in (train_loader, val_loader, test_loader):
            it = getattr(_ldr, "_iterator", None)
            if it is not None:
                try:
                    it._shutdown_workers()
                except Exception:
                    pass
    except Exception:
        pass

    # --- final console/file footer ---
    tee.log(f"Done. Best val={best_val:.6f}")
    tee.log(f"Artifacts saved in: {out_dir}")
    tee.log(f" - Logs: {os.path.join(logs_dir,'train_val_metrics.csv')}")
    tee.log(f" - Test data: {npz_path}")
    tee.log(f" - Test metrics: {os.path.join(out_dir,'test_metrics.json')}"
            + (f", rotated: {os.path.join(out_dir,'test_metrics_rot.json')}" if test_rot_eval else ""))
    tee.log(f" - Model: {('saved ' + script_path) if script_path else 'TorchScript export failed; state_dict saved'}")
    tee.log(f" - Training summary: {os.path.join(out_dir,'training_summary.json')}")
    tee.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

if __name__ == "__main__":
    main()
