from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

from mlp_qmmm.a0_structure import Keys, NON_TENSOR_KEYS


KCALMOL_TO_EV = 0.0433641153087705
EV_TO_KCALMOL = 1.0 / KCALMOL_TO_EV

_NON_TENSOR = set(NON_TENSOR_KEYS)
QM_TYPE_KEY = getattr(Keys, "QM_TYPE", "qm_type")


def _unit_multiplier(key: str, print_units: str) -> float:
    if str(print_units).lower().startswith("kcal"):
        k = key.lower()
        if k in ("de", "energy") or k.startswith("e_") or "grad" in k or k.endswith("_esp"):
            return EV_TO_KCALMOL
    return 1.0


class ProgressTee:
    def __init__(self, path: Optional[str]) -> None:
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


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    if x.dtype == np.float64:
        return torch.from_numpy(x.astype(np.float32, copy=False))
    if x.dtype.kind in ("f",):
        return torch.from_numpy(x.astype(np.float32, copy=False))
    if x.dtype.kind in ("i", "u"):
        return torch.from_numpy(x.astype(np.int64, copy=False))
    return torch.from_numpy(x)


def move_batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    def _move(v: Any) -> Any:
        return v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v

    out = {k: _move(v) for k, v in batch.items() if k != "targets"}
    if "targets" in batch and isinstance(batch["targets"], dict):
        out["targets"] = {k: _move(v) for k, v in batch["targets"].items()}
    return out


def make_loss(kind: str) -> nn.Module:
    k = str(kind).lower()
    if k in ("l2", "mse"):
        return nn.MSELoss(reduction="none")
    if k in ("l1", "mae"):
        return nn.L1Loss(reduction="none")
    if k in ("huber", "smoothl1"):
        return nn.SmoothL1Loss(beta=1.0, reduction="none")
    raise ValueError(f"Unknown loss type: {kind}")


# ---------------------------------------------------------------------------
# masking / reductions
# ---------------------------------------------------------------------------

def _broadcast_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    m = mask
    while m.dim() < like.dim():
        m = m.unsqueeze(-1)
    return m.to(like.dtype)


def masked_sum_and_count(
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        return values.sum(), torch.tensor(values.numel(), device=values.device, dtype=values.dtype)
    m = _broadcast_mask((mask != 0), values)
    return (values * m).sum(), m.sum().clamp_min(1.0)


def masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    s, n = masked_sum_and_count(values, mask)
    return s / n


def masked_mean_per_frame(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if values.dim() == 1:
        return values.mean()

    B = values.shape[0]
    if mask is None:
        return values.reshape(B, -1).mean(dim=1).mean()

    m = _broadcast_mask((mask != 0), values)
    reduce_dims = tuple(range(1, values.dim()))
    sums = (values * m).sum(dim=reduce_dims)
    counts = m.sum(dim=reduce_dims)

    valid = counts > 0
    if not torch.any(valid):
        return torch.tensor(0.0, device=values.device, dtype=values.dtype)

    per = torch.zeros_like(sums)
    per[valid] = sums[valid] / counts[valid]
    return per[valid].mean()


def masked_rmse_from_residual(
    residual: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    frame_equal: bool,
) -> float:
    sq = residual * residual
    m = masked_mean_per_frame(sq, mask) if frame_equal else masked_mean(sq, mask)
    return float(torch.sqrt(m.detach()).cpu().item())


# ---------------------------------------------------------------------------
# sanitisation
# ---------------------------------------------------------------------------

def sanitize_padded_mm_inplace(batch: Dict[str, torch.Tensor]) -> None:
    if Keys.MM_TYPE not in batch:
        return

    mm_mask = (batch[Keys.MM_TYPE] != 0)
    if Keys.MM_COORDS in batch and isinstance(batch[Keys.MM_COORDS], torch.Tensor):
        batch[Keys.MM_COORDS] = batch[Keys.MM_COORDS] * _broadcast_mask(mm_mask, batch[Keys.MM_COORDS])
    if Keys.MM_Q in batch and isinstance(batch[Keys.MM_Q], torch.Tensor):
        batch[Keys.MM_Q] = batch[Keys.MM_Q] * _broadcast_mask(mm_mask, batch[Keys.MM_Q])

    for k, v in list(batch.items()):
        if not isinstance(v, torch.Tensor):
            continue
        if k.startswith("mm_") and k not in (Keys.MM_COORDS, Keys.MM_Q, Keys.MM_TYPE):
            batch[k] = v * _broadcast_mask(mm_mask, v)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class QMMMDataset(Dataset):
    """
    1.x dataset with two modes:
    - stacked: closest to old 0.x semantics
    - lazy: avoids eagerly stacking everything
    """
    def __init__(
        self,
        frames: List[Dict[str, np.ndarray]],
        *,
        input_keys: Sequence[str],
        mode: str = "stacked",
    ) -> None:
        if not frames:
            raise ValueError("Empty dataset")

        self.mode = str(mode).lower()
        if self.mode not in {"stacked", "lazy"}:
            raise ValueError("dataset mode must be 'stacked' or 'lazy'")

        self.input_keys = set(input_keys)
        keys = set().union(*[f.keys() for f in frames]) - _NON_TENSOR
        self._available_target_keys = sorted(
            k for k in keys if k not in self.input_keys and all(k in f for f in frames)
        )

        self.n = len(frames)
        self.arr: Dict[str, np.ndarray] = {}
        self.frames: List[Dict[str, np.ndarray]] = []

        if self.mode == "stacked":
            for k in sorted(keys):
                vals = [np.asarray(fr[k]) for fr in frames if k in fr]
                vals = [v if v.ndim > 0 else v.reshape(1) for v in vals]
                try:
                    self.arr[k] = np.stack(vals, axis=0)
                except Exception as e:
                    shapes = [tuple(v.shape) for v in vals]
                    raise ValueError(f"Inconsistent shapes for '{k}': {shapes}") from e
        else:
            self.frames = frames

    def __len__(self) -> int:
        return self.n

    def available_target_keys(self) -> List[str]:
        return list(self._available_target_keys)

    def composition_key(self, i: int) -> Tuple[Any, ...]:
        """
        Used only for variable_qm bucketed batching.
        Prefer atom_types, then qm_Z, then N_QM.
        """
        if self.mode == "stacked":
            arr = self.arr
            if Keys.ATOM_TYPES in arr:
                a = np.asarray(arr[Keys.ATOM_TYPES][i]).reshape(-1)
                if QM_TYPE_KEY in arr:
                    qmask = np.asarray(arr[QM_TYPE_KEY][i]).reshape(-1) > 0.5
                    if qmask.shape[0] == a.shape[0]:
                        return tuple(int(x) for x in a[qmask])
                return tuple(int(x) for x in a)

            if Keys.QM_Z in arr:
                z = np.asarray(arr[Keys.QM_Z][i]).reshape(-1)
                if QM_TYPE_KEY in arr:
                    qmask = np.asarray(arr[QM_TYPE_KEY][i]).reshape(-1) > 0.5
                    if qmask.shape[0] == z.shape[0]:
                        return tuple(int(x) for x in z[qmask])
                return tuple(int(x) for x in z if int(x) != 0)

            if Keys.N_QM in arr:
                return ("N_qm", int(np.asarray(arr[Keys.N_QM][i]).reshape(-1)[0]))
            return ()

        frame = self.frames[i]
        if Keys.ATOM_TYPES in frame:
            a = np.asarray(frame[Keys.ATOM_TYPES]).reshape(-1)
            if QM_TYPE_KEY in frame:
                qmask = np.asarray(frame[QM_TYPE_KEY]).reshape(-1) > 0.5
                if qmask.shape[0] == a.shape[0]:
                    return tuple(int(x) for x in a[qmask])
            return tuple(int(x) for x in a)

        if Keys.QM_Z in frame:
            z = np.asarray(frame[Keys.QM_Z]).reshape(-1)
            if QM_TYPE_KEY in frame:
                qmask = np.asarray(frame[QM_TYPE_KEY]).reshape(-1) > 0.5
                if qmask.shape[0] == z.shape[0]:
                    return tuple(int(x) for x in z[qmask])
            return tuple(int(x) for x in z if int(x) != 0)

        if Keys.N_QM in frame:
            return ("N_qm", int(np.asarray(frame[Keys.N_QM]).reshape(-1)[0]))
        return ()

    def __getitem__(self, i: int) -> Dict[str, Any]:
        sample: Dict[str, Any] = {}

        if self.mode == "stacked":
            for k, arr in self.arr.items():
                t = to_tensor(arr[i])
                if t.dim() == 1 and ((k.endswith("_coords") or ("grad" in k)) and (t.numel() % 3 == 0)):
                    t = t.view(-1, 3)
                sample[k] = t
        else:
            frame = self.frames[i]
            for k, v in frame.items():
                if k in _NON_TENSOR:
                    continue
                t = to_tensor(np.asarray(v))
                if t.dim() == 1 and ((k.endswith("_coords") or ("grad" in k)) and (t.numel() % 3 == 0)):
                    t = t.view(-1, 3)
                sample[k] = t

        targets: Dict[str, torch.Tensor] = {}
        for k, v in sample.items():
            if k in self.input_keys:
                continue
            if isinstance(v, torch.Tensor):
                targets[k] = v
        sample["targets"] = targets
        return sample


def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    common = set(samples[0].keys())
    for s in samples[1:]:
        common &= set(s.keys())

    for k in sorted(common):
        v0 = samples[0][k]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([s[k] for s in samples], dim=0)
        elif k == "targets" and isinstance(v0, dict):
            all_t = set().union(*[s["targets"].keys() for s in samples])
            targ: Dict[str, torch.Tensor] = {}
            for tk in sorted(all_t):
                if all(tk in s["targets"] for s in samples):
                    targ[tk] = torch.stack([s["targets"][tk] for s in samples], dim=0)
            out["targets"] = targ

    return out


class BucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: QMMMDataset,
        local_indices: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

        local_indices = list(local_indices)
        buckets: Dict[Tuple[Any, ...], List[int]] = {}
        for local_pos, global_idx in enumerate(local_indices):
            key = dataset.composition_key(global_idx)
            buckets.setdefault(key, []).append(local_pos)

        self._batches: List[List[int]] = []
        rng = np.random.default_rng(self.seed)
        for _, positions in buckets.items():
            pos = np.array(positions, dtype=np.int64)
            if self.shuffle:
                rng.shuffle(pos)
            for start in range(0, len(pos), self.batch_size):
                chunk = pos[start:start + self.batch_size].tolist()
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                self._batches.append(chunk)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self._epoch)
            order = rng.permutation(len(self._batches)).tolist()
            for i in order:
                yield self._batches[i]
        else:
            yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


def make_standard_loader(
    ds: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_batch,
    )


def make_variable_qm_loader(
    ds: Any,
    dataset: QMMMDataset,
    *,
    local_indices: List[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    sampler = BucketBatchSampler(
        dataset,
        local_indices=local_indices,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        drop_last=False,
    )
    return DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_batch,
    )


# ---------------------------------------------------------------------------
# shapes / masks
# ---------------------------------------------------------------------------

def normalize_shapes(pred: torch.Tensor, targ: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred.shape == targ.shape:
        return pred, targ
    if pred.dim() == 3 and targ.dim() == 2 and pred.shape[-1] == 3 and targ.shape[-1] == pred.shape[-2] * 3:
        return pred, targ.view(targ.shape[0], pred.shape[-2], 3)
    if targ.dim() == 3 and pred.dim() == 2 and targ.shape[-1] == 3 and pred.shape[-1] == targ.shape[-2] * 3:
        return pred.view(pred.shape[0], targ.shape[-2], 3), targ
    if pred.dim() == 2 and pred.shape[-1] == 1 and targ.dim() == 1:
        return pred.squeeze(-1), targ
    if targ.dim() == 2 and targ.shape[-1] == 1 and pred.dim() == 1:
        return pred, targ.squeeze(-1)
    raise ValueError(f"Shape mismatch: pred {tuple(pred.shape)} vs targ {tuple(targ.shape)}")


def build_mask_picker(cfg: Dict[str, Any]) -> Callable:
    masks_cfg: Dict[str, str] = cfg.get("masks", {}) or {}

    def pick_mask(name: str, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        if name in masks_cfg and masks_cfg[name] in batch:
            return batch[masks_cfg[name]]
        for pat, mk in masks_cfg.items():
            if ("*" in pat or "?" in pat or "[" in pat) and fnmatch(name, pat) and mk in batch:
                return batch[mk]
        if name.startswith("mm_") and Keys.MM_TYPE in batch:
            return batch[Keys.MM_TYPE] != 0
        if name.startswith("qm_") and QM_TYPE_KEY in batch:
            return batch[QM_TYPE_KEY] != 0
        return None

    return pick_mask


def _mask_compatible(mask: Optional[torch.Tensor], values: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if values.dim() < 2:
        return None
    if mask.dim() < 2:
        return None
    if mask.shape[0] != values.shape[0]:
        return None
    if mask.shape[1] != values.shape[1]:
        return None
    return mask


# ---------------------------------------------------------------------------
# scheduler / weights
# ---------------------------------------------------------------------------

@dataclass
class SchedulerBundle:
    scheduler: Any
    sched_type: str
    start_lr: float
    exp_min_lr: Optional[float]
    mix_state: Optional[Dict[str, Any]]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    ocfg: Dict[str, Any],
    *,
    epochs: int,
    steps_per_epoch: int,
    start_lr: float,
) -> SchedulerBundle:
    sched_type = str(ocfg.get("schedule", "plateau")).lower()
    exp_min_lr: Optional[float] = None
    mix_state: Optional[Dict[str, Any]] = None
    scheduler: Any = None
    eff_start_lr = start_lr

    if sched_type == "onecycle":
        max_lr = float(ocfg.get("max_lr", start_lr * 5.0))
        pct_start = float(ocfg.get("pct_start", 0.3))
        div_factor = float(ocfg.get("div_factor", 25.0))
        final_div = float(ocfg.get("final_div_factor", 1e4))
        anneal = str(ocfg.get("anneal", "cos")).lower()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=max(1, steps_per_epoch),
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div,
            anneal_strategy=("cos" if anneal.startswith("cos") else "linear"),
        )
        eff_start_lr = max_lr / div_factor

    elif sched_type == "exponential":
        ecfg = ocfg.get("exponential", {}) or {}
        gamma = float(ecfg.get("factor", 0.99))
        exp_min_lr = float(ecfg.get("min_lr", 1e-6))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    elif sched_type == "mixed":
        mcfg = ocfg.get("mixed", {}) or {}
        gamma = float(mcfg.get("exp_factor", 0.99))
        exp_min_lr = float(mcfg.get("min_lr", 1e-6))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        mix_state = {
            "best": float("inf"),
            "bad_epochs": 0,
            "cooldown": 0,
            "patience": int(mcfg.get("patience", 8)),
            "threshold": float(mcfg.get("threshold", 1e-4)),
            "plat_factor": float(mcfg.get("plat_factor", 0.5)),
            "cooldown_default": int(mcfg.get("cooldown", 0)),
        }

    else:
        pcfg = ocfg.get("plateau", {}) or {}
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(pcfg.get("factor", 0.5)),
            patience=int(pcfg.get("patience", 10)),
            threshold=float(pcfg.get("threshold", 1e-4)),
            min_lr=float(pcfg.get("min_lr", 1e-6)),
        )
        sched_type = "plateau"

    return SchedulerBundle(
        scheduler=scheduler,
        sched_type=sched_type,
        start_lr=eff_start_lr,
        exp_min_lr=exp_min_lr,
        mix_state=mix_state,
    )


def _clamp_min_lr(optimizer: torch.optim.Optimizer, min_lr: Optional[float]) -> None:
    if min_lr is None:
        return
    for g in optimizer.param_groups:
        if g["lr"] < min_lr:
            g["lr"] = min_lr


def step_scheduler(
    bundle: SchedulerBundle,
    optimizer: torch.optim.Optimizer,
    *,
    val_unw: float,
    tee: ProgressTee,
) -> None:
    s = bundle
    if s.scheduler is None or s.sched_type == "onecycle":
        return

    if s.sched_type == "plateau":
        s.scheduler.step(val_unw)

    elif s.sched_type == "exponential":
        s.scheduler.step()
        _clamp_min_lr(optimizer, s.exp_min_lr)

    elif s.sched_type == "mixed":
        s.scheduler.step()
        _clamp_min_lr(optimizer, s.exp_min_lr)
        ms = s.mix_state
        improved = (ms["best"] - val_unw) >= ms["threshold"]
        if improved:
            ms["best"] = val_unw
            ms["bad_epochs"] = 0
            ms["cooldown"] = 0
        else:
            if ms["cooldown"] > 0:
                ms["cooldown"] -= 1
            else:
                ms["bad_epochs"] += 1
                if ms["bad_epochs"] >= ms["patience"]:
                    new_lr = max(
                        s.exp_min_lr or 0.0,
                        optimizer.param_groups[0]["lr"] * ms["plat_factor"],
                    )
                    for g in optimizer.param_groups:
                        g["lr"] = new_lr
                    tee.log(
                        f"[lr-mixed] plateau fallback: lr→{new_lr:.3e} (factor={ms['plat_factor']})"
                    )
                    ms["bad_epochs"] = 0
                    ms["cooldown"] = ms["cooldown_default"]


def make_group_weight_fn(
    cfg: Dict[str, Any],
    *,
    start_lr: float,
) -> Callable[[str, float], float]:
    wcfg = cfg.get("weight_schedule", {}) or {}
    energy_keys = list(wcfg.get("energy_keys", ["dE", "energy"]))
    grad_keys = list(wcfg.get("grad_keys", ["qm_*", "mm_*"]))
    w_ene_base = float(wcfg.get("w_ene", 1.0))
    w_grad_max = float(wcfg.get("w_grad_max", 100.0))

    def _weight(name: str, lr_now: float) -> float:
        r = float(lr_now / max(start_lr, 1e-12))
        w_grad = w_ene_base + (w_grad_max - w_ene_base) * r
        if name in energy_keys or any(fnmatch(name, p) for p in energy_keys):
            return w_ene_base
        if name in grad_keys or any(fnmatch(name, p) for p in grad_keys):
            return w_grad
        return 1.0

    return _weight


# ---------------------------------------------------------------------------
# forward / losses / eval
# ---------------------------------------------------------------------------

def forward_batch(
    core: nn.Module,
    batch: Dict[str, Any],
    required_keys: List[str],
) -> Dict[str, torch.Tensor]:
    if hasattr(core, "forward_from_frame"):
        pred = core.forward_from_frame(batch)
    else:
        pred = core(*[batch[k] for k in required_keys])
    if not isinstance(pred, dict):
        raise TypeError(f"Model forward must return Dict[str, Tensor], got {type(pred)}")
    return pred


def compute_losses(
    pred: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    criterions: Dict[str, nn.Module],
    base_weights: Dict[str, float],
    group_weight_fn: Callable[[str, float], float],
    pick_mask: Callable,
    *,
    lr_now: float,
    mm_equal_frame_weight: bool,
    qm_equal_frame_weight: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float], Dict[str, float]]:
    total = torch.tensor(0.0, device=device)
    raw_means: Dict[str, float] = {}
    rmses: Dict[str, float] = {}
    eff_w: Dict[str, float] = {}

    for name, crit in criterions.items():
        if name not in pred or name not in targets:
            continue

        p, t = normalize_shapes(pred[name], targets[name])
        raw = crit(p, t)
        mask = _mask_compatible(pick_mask(name, batch), raw)

        if mm_equal_frame_weight and name.startswith("mm_"):
            mean = masked_mean_per_frame(raw, mask)
            rmse = masked_rmse_from_residual(p - t, mask, frame_equal=True)
        elif qm_equal_frame_weight and name.startswith("qm_"):
            mean = masked_mean_per_frame(raw, mask)
            rmse = masked_rmse_from_residual(p - t, mask, frame_equal=True)
        else:
            mean = masked_mean(raw, mask)
            rmse = masked_rmse_from_residual(p - t, mask, frame_equal=False)

        bw = base_weights.get(name, 1.0)
        sw = group_weight_fn(name, lr_now)
        total = total + mean * bw * sw

        raw_means[name] = float(mean.detach().cpu().item())
        rmses[name] = rmse
        eff_w[name] = bw * sw

    return total, raw_means, rmses, eff_w


def eval_split(
    loader: DataLoader,
    core: nn.Module,
    required_keys: List[str],
    criterions: Dict[str, nn.Module],
    base_weights: Dict[str, float],
    group_weight_fn: Callable[[str, float], float],
    pick_mask: Callable,
    device: torch.device,
    *,
    lr_now: float,
    mm_equal_frame_weight: bool,
    qm_equal_frame_weight: bool,
) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
    core.eval()

    steps = 0
    w_sum = 0.0
    uw_sum = 0.0
    raw_acc = {k: 0.0 for k in criterions.keys()}
    rmse_acc = {k: 0.0 for k in criterions.keys()}

    for cpu_batch in loader:
        batch = move_batch_to_device(cpu_batch, device)
        sanitize_padded_mm_inplace(batch)

        with torch.enable_grad():
            pred = forward_batch(core, batch, required_keys)

        total_w = torch.tensor(0.0, device=device)
        sum_unw = torch.tensor(0.0, device=device)
        cnt_unw = 0

        for name, crit in criterions.items():
            if name not in pred or name not in batch["targets"]:
                continue

            p, t = normalize_shapes(pred[name], batch["targets"][name])
            raw = crit(p, t)
            mask = _mask_compatible(pick_mask(name, batch), raw)

            if mm_equal_frame_weight and name.startswith("mm_"):
                mean = masked_mean_per_frame(raw, mask)
                rmse = masked_rmse_from_residual(p - t, mask, frame_equal=True)
            elif qm_equal_frame_weight and name.startswith("qm_"):
                mean = masked_mean_per_frame(raw, mask)
                rmse = masked_rmse_from_residual(p - t, mask, frame_equal=True)
            else:
                mean = masked_mean(raw, mask)
                rmse = masked_rmse_from_residual(p - t, mask, frame_equal=False)

            bw = base_weights.get(name, 1.0)
            sw = group_weight_fn(name, lr_now)
            total_w = total_w + mean * bw * sw
            sum_unw = sum_unw + mean
            cnt_unw += 1

            raw_acc[name] += float(mean.detach().cpu().item())
            rmse_acc[name] += rmse

        uw_sum += float((sum_unw / max(cnt_unw, 1)).detach().cpu().item())
        w_sum += float(total_w.detach().cpu().item())
        steps += 1

    n = max(1, steps)
    return (
        w_sum / n,
        uw_sum / n,
        {k: raw_acc[k] / n for k in raw_acc},
        {k: rmse_acc[k] / n for k in rmse_acc},
    )


def clip_gradients(
    optimizer: torch.optim.Optimizer,
    max_norm: float,
) -> Tuple[float, bool]:
    params = [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad]
    norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm).detach().cpu().item())
    return norm, norm > max_norm


# ---------------------------------------------------------------------------
# preparation / io
# ---------------------------------------------------------------------------

def prepare_frames_with_model(
    frames: List[Dict[str, np.ndarray]],
    prepare_inputs: Optional[Callable[..., Any]],
    cfg: Dict[str, Any],
    *,
    prep_output_keys: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    if not callable(prepare_inputs):
        return {"frames_prepared": 0, "keys_written": []}

    keys_written = set()
    n_done = 0

    for frame in frames:
        tensor_frame: Dict[str, Any] = {}
        for k, v in frame.items():
            if isinstance(v, np.ndarray):
                tensor_frame[k] = to_tensor(np.asarray(v))
            else:
                tensor_frame[k] = v

        before = set(tensor_frame.keys())
        result = prepare_inputs(tensor_frame, cfg, inplace=True, overwrite=False)
        after = set(result.keys())

        copy_keys = list(prep_output_keys or sorted(after - before))
        for k in copy_keys:
            if k not in result:
                continue
            v = result[k]
            if isinstance(v, torch.Tensor):
                frame[k] = v.detach().cpu().numpy().copy()
            else:
                frame[k] = np.asarray(v).copy()
            keys_written.add(k)
        n_done += 1

    info = {"frames_prepared": n_done, "keys_written": sorted(keys_written)}
    if verbose and info["keys_written"]:
        print(f"[prepare] wrote keys {info['keys_written']} on {n_done} frame(s)")
    return info


def resolve_train_dir(cfg: Dict[str, Any]) -> str:
    trcfg = cfg.get("trainer", {}) or {}
    pcfg = cfg.get("parser", {}) or {}

    train_dir = str(trcfg.get("out_dir", "") or "").strip()
    parse_dir = str(pcfg.get("out_dir", "") or "").strip()

    if train_dir:
        return os.path.abspath(train_dir)
    if parse_dir:
        return os.path.abspath(parse_dir)
    return os.getcwd()


def init_csv(path: str, criterions: Dict[str, nn.Module]) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["epoch", "split", "lr", "total_loss", "total_loss_unw"]
    for k in sorted(criterions.keys()):
        cols += [f"loss_{k}", f"rmse_{k}", f"weight_{k}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")


def write_csv_row(
    path: str,
    epoch: int,
    split: str,
    lr: float,
    total_loss: float,
    total_unw: float,
    raw_means: Dict[str, float],
    rmses: Dict[str, float],
    criterions: Dict[str, nn.Module],
    group_weight_fn: Callable[[str, float], float],
    base_weights: Dict[str, float],
) -> None:
    line = [
        str(epoch),
        split,
        f"{lr:.8e}",
        f"{total_loss:.8e}",
        f"{total_unw:.8e}",
    ]
    for k in sorted(criterions.keys()):
        bw = base_weights.get(k, 1.0)
        sw = group_weight_fn(k, lr)
        line += [
            f"{raw_means.get(k, float('nan')):.8e}",
            f"{rmses.get(k, float('nan')):.8e}",
            f"{bw * sw:.8e}",
        ]
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(line) + "\n")


def save_script_checkpoint(
    core: nn.Module,
    out_dir: str,
    epoch: int,
    tee: ProgressTee,
) -> Optional[str]:
    core.eval()
    path = os.path.join(out_dir, f"model_script_ep{epoch:04d}.pt")
    try:
        torch.jit.script(core).save(path)
        tee.log(f"[epoch {epoch}] TorchScript saved → {path}")
        return path
    except Exception as exc:
        tee.log(f"[warn] TorchScript export (epoch {epoch}) failed: {type(exc).__name__}: {exc}")
        return None


def write_training_summary(
    out_dir: str,
    cfg: Dict[str, Any],
    *,
    device: torch.device,
    amp_mode: str,
    wall_seconds: float,
    wall_start: float,
    per_epoch_seconds: List[float],
    n_train: int,
    n_val: int,
    n_test: int,
    n_total: int,
    batch_size: int,
    num_workers: int,
    val_frac: float,
    test_frac: float,
    initial_stats: Dict[str, int],
    required_keys: List[str],
    output_keys: List[str],
    allowed_loss_keys: List[str],
    criterions: Dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    sched_bundle: SchedulerBundle,
    best_epoch: Optional[int],
    best_val_unw: float,
    test_w: float,
    test_unw: float,
    peak_cuda_bytes: Optional[int],
    tee: ProgressTee,
) -> None:
    mcfg = cfg.get("model", {}) or {}
    wd = float((cfg.get("optim", {}) or {}).get("weight_decay", 0.0))

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
            "batch_size": batch_size,
            "num_workers": num_workers,
            "amp_mode": amp_mode,
            "grad_clip": float((cfg.get("trainer", {}) or {}).get("grad_clip", 5.0)),
            "val_frac": val_frac,
            "test_frac": test_frac,
            "dataset_sizes": {
                "train": n_train,
                "val": n_val,
                "test": n_test,
                "total": n_total,
            },
            "peak_cuda_bytes": peak_cuda_bytes,
        },
        "model": {
            "name": str(mcfg.get("name", "")),
            "max_qm": int((cfg.get("model", {}) or {}).get("max_qm", 0)),
            "param_counts": initial_stats,
            "required_keys": required_keys,
            "output_keys": output_keys,
            "allowed_loss_keys": allowed_loss_keys,
            "loss_keys": sorted(criterions.keys()),
        },
        "optimizer": {
            "type": type(optimizer).__name__,
            "lr_start": sched_bundle.start_lr,
            "weight_decay": wd,
            "scheduler": sched_bundle.sched_type,
        },
        "best": {
            "epoch": best_epoch,
            "val_unweighted": best_val_unw,
            "test_weighted": test_w,
            "test_unweighted": test_unw,
        },
    }

    path = os.path.join(out_dir, "training_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    tee.log(f" - Training summary: {path}")