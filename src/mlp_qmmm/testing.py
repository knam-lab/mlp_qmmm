#!/usr/bin/env python3
# src/mlp_qmmm/testing.py
"""
Standalone model evaluation for mlp_qmmm (v1.x)

Goals
-----
- Work from either:
    1) a global run directory containing training artefacts, or
    2) explicit file paths for summary / model_io / parse_summary / config / model_pt
- Accept external test data and evaluate the trained model on it.
- Reuse the same high-level data flow as training:
      parse_dataset -> optional model-specific prepare_inputs -> batch inference
- Support both:
      * TorchScript model .pt
      * state-dict / checkpoint .pt via config + b_nn_loader.load_model()
- Remain model-agnostic at the evaluation layer.

Typical usage
-------------
1) Everything in one run directory, test data supplied separately:
    python -m mlp_qmmm.testing \
        --run_dir runs/train_xxx \
        --test_data /path/to/test_folder

2) Explicit files:
    python -m mlp_qmmm.testing \
        --summary runs/train_xxx/training_summary.json \
        --model_io runs/train_xxx/model_io.json \
        --config runs/train_xxx/config_used.yml \
        --model_pt runs/train_xxx/model_state.pt \
        --test_data /path/to/test_folder

3) Re-test the saved test_data.npz from the run directory:
    python -m mlp_qmmm.testing \
        --run_dir runs/train_xxx

Outputs
-------
All outputs are written under:
    <out_dir>  (default: <run_dir>/test_results or next to the summary)

Files written
-------------
test_metrics.csv
test_metrics.json
predictions_test.npz
model_stats.json
bridge_meta.txt
resolved_inputs.json
plots_test/
    scatter_<key>.png
    density_<key>.png
    loss_total.png
    loss_<key>.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
from fnmatch import fnmatch
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlp_qmmm.a0_structure import Keys
from mlp_qmmm.a2_parser import parse_dataset
from mlp_qmmm.b_nn_loader import get_model, load_model
from mlp_qmmm.trainer_utils import (
    BucketBatchSampler,
    QMMMDataset,
    collate_batch,
    move_batch_to_device,
    prepare_frames_with_model,
    sanitize_padded_batch,
)

EV_TO_KCALMOL = 23.060547830619
QM_TYPE_KEY = getattr(Keys, "QM_TYPE", "qm_type")


# ---------------------------------------------------------------------------
# Path / metadata resolution
# ---------------------------------------------------------------------------

def _abspath_or_empty(path: str) -> str:
    p = str(path or "").strip()
    return os.path.abspath(p) if p else ""


def _pick_existing(*paths: str) -> str:
    for p in paths:
        q = _abspath_or_empty(p)
        if q and os.path.exists(q):
            return q
    return ""


def _latest_glob(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    return os.path.abspath(hits[-1]) if hits else ""


def _resolve_run_artifacts(args: argparse.Namespace) -> Dict[str, str]:
    run_dir = _abspath_or_empty(args.run_dir)
    summary = _abspath_or_empty(args.summary)
    model_io = _abspath_or_empty(args.model_io)
    parse_summary = _abspath_or_empty(args.parse_summary)
    config = _abspath_or_empty(args.config)
    model_pt = _abspath_or_empty(args.model_pt)
    logs_csv = _abspath_or_empty(args.logs_csv)
    test_data = _abspath_or_empty(args.test_data)

    if run_dir:
        summary = summary or _pick_existing(os.path.join(run_dir, "training_summary.json"))
        model_io = model_io or _pick_existing(os.path.join(run_dir, "model_io.json"))
        parse_summary = parse_summary or _pick_existing(os.path.join(run_dir, "parse_summary.json"))
        config = config or _pick_existing(os.path.join(run_dir, "config_used.yml"))
        logs_csv = logs_csv or _pick_existing(os.path.join(run_dir, "logs", "train_val_metrics.csv"))
        if not model_pt:
            model_pt = _pick_existing(
                os.path.join(run_dir, "model_script.pt"),
                os.path.join(run_dir, "model_state.pt"),
                os.path.join(run_dir, "best.pt"),
                os.path.join(run_dir, "restart.pt"),
            ) or _latest_glob(os.path.join(run_dir, "model_script_ep*.pt"))
        if not test_data:
            test_data = _pick_existing(os.path.join(run_dir, "test_data.npz"))

    base_dir = ""
    for cand in (run_dir, summary, model_io, parse_summary, config, model_pt):
        if cand:
            base_dir = cand if os.path.isdir(cand) else os.path.dirname(cand)
            break
    if not base_dir:
        base_dir = os.getcwd()

    out_dir = _abspath_or_empty(args.out_dir) or os.path.join(base_dir, "test_results")
    os.makedirs(out_dir, exist_ok=True)

    return {
        "run_dir": run_dir,
        "summary": summary,
        "model_io": model_io,
        "parse_summary": parse_summary,
        "config": config,
        "model_pt": model_pt,
        "logs_csv": logs_csv,
        "test_data": test_data,
        "out_dir": out_dir,
        "base_dir": base_dir,
    }


def _read_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_yaml(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_metadata(
    summary: Dict[str, Any],
    model_io: Dict[str, Any],
    parse_summary: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    explicit_adapter: str,
    explicit_keys: Sequence[str],
) -> Dict[str, Any]:
    model_info = summary.get("model", {}) if isinstance(summary.get("model", {}), dict) else {}
    run_info = summary.get("run", {}) if isinstance(summary.get("run", {}), dict) else {}

    model_name = str(
        model_info.get("name")
        or (cfg.get("model", {}) or {}).get("name", "")
    ).strip()

    required_keys = list(
        model_info.get("required_keys")
        or model_io.get("required_keys")
        or model_io.get("input_keys")
        or []
    )
    output_keys = list(
        model_info.get("output_keys")
        or model_io.get("output_keys")
        or []
    )
    allowed_loss_keys = list(
        model_info.get("allowed_loss_keys")
        or model_io.get("allowed_loss_keys")
        or []
    )
    loss_keys = list(
        explicit_keys
        or model_info.get("loss_keys")
        or model_io.get("loss_keys")
        or []
    )

    masks_cfg = dict(model_io.get("masks", {}) or {})
    adapter = str(
        explicit_adapter
        or parse_summary.get("adapter")
        or cfg.get("adapter", "numpy_folder")
    ).strip()
    if not adapter:
        adapter = "numpy_folder"

    model_max_qm = int(
        model_info.get("max_qm")
        or (cfg.get("model", {}) or {}).get("max_qm", 0)
        or 0
    )
    mm_pad_to = int(
        ((parse_summary.get("padding", {}) or {}).get("mm_pad_to", 0))
        or ((cfg.get("adapter_kwargs", {}) or {}).get("max_mm", 0))
        or 0
    )

    return {
        "model_name": model_name,
        "required_keys": required_keys,
        "output_keys": output_keys,
        "allowed_loss_keys": allowed_loss_keys,
        "loss_keys": loss_keys,
        "masks_cfg": masks_cfg,
        "adapter": adapter,
        "model_max_qm": model_max_qm,
        "mm_pad_to": mm_pad_to,
        "run_info": run_info,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _try_torchscript(model_pt: str, device: torch.device) -> Optional[torch.jit.ScriptModule]:
    try:
        m = torch.jit.load(model_pt, map_location=device)
        m.eval()
        return m
    except Exception:
        return None


def _load_eval_model(
    model_pt: str,
    cfg: Dict[str, Any],
    device: torch.device,
    *,
    debug: bool = False,
) -> Tuple[Any, bool]:
    """
    Return (model, is_torchscript).

    Priority:
      1. TorchScript .pt
      2. state-dict / checkpoint via load_model(cfg)
    """
    if not model_pt:
        raise FileNotFoundError("No model .pt path could be resolved.")

    m = _try_torchscript(model_pt, device)
    if m is not None:
        if debug:
            print(f"[test] loaded TorchScript model: {model_pt}")
        return m, True

    if not cfg:
        raise RuntimeError(
            "Model .pt is not TorchScript and no config was available.\n"
            "State-dict / best.pt / restart.pt loading requires config_used.yml "
            "or --config."
        )

    cfg = dict(cfg)
    cfg.setdefault("model", {})
    cfg["model"]["ckpt"] = model_pt
    cfg["model"]["ckpt_strict"] = True

    mio = load_model(cfg, device=device, verbose=debug)
    model = mio.model
    model.eval()
    if debug:
        print(f"[test] loaded state-dict/checkpoint via b_nn_loader: {model_pt}")
    return model, False


# ---------------------------------------------------------------------------
# Parsing + model-specific prep
# ---------------------------------------------------------------------------

def _prepare_test_frames(
    test_data: str,
    *,
    adapter: str,
    cfg: Dict[str, Any],
    summary: Dict[str, Any],
    parse_summary: Dict[str, Any],
    out_dir: str,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    if not test_data:
        raise FileNotFoundError(
            "No test data path was provided and no default test_data.npz was found."
        )

    model_max_qm = int(
        (summary.get("model", {}) or {}).get("max_qm", 0)
        or (cfg.get("model", {}) or {}).get("max_qm", 0)
        or 0
    )
    mm_pad_to = int(
        ((parse_summary.get("padding", {}) or {}).get("mm_pad_to", 0))
        or ((cfg.get("adapter_kwargs", {}) or {}).get("max_mm", 0))
        or 0
    )

    parse_out = os.path.join(out_dir, "parsed_test_input")
    os.makedirs(parse_out, exist_ok=True)

    frames = parse_dataset(
        adapter=adapter,
        path=test_data,
        adapter_kwargs=dict(cfg.get("adapter_kwargs", {}) or {}),
        postprocess_kwargs=dict(cfg.get("postprocess", {}) or {}),
        out_dir=parse_out,
        verbose=bool(debug),
        required_keys_all_frames=[],
        required_keys_warn_only=True,
        qm_pad_to=(model_max_qm if model_max_qm > 0 else None),
        mm_pad_to=(mm_pad_to if mm_pad_to > 0 else None),
    )
    return frames


def _maybe_prepare_with_model(
    frames: List[Dict[str, Any]],
    *,
    cfg: Dict[str, Any],
    model_name: str,
    debug: bool = False,
) -> Dict[str, Any]:
    if not cfg or not model_name:
        return {"frames_prepared": 0, "keys_written": []}
    try:
        ModelCls = get_model(model_name)
    except Exception:
        return {"frames_prepared": 0, "keys_written": []}

    prepare_inputs = getattr(ModelCls, "prepare_inputs", None)
    prep_output_keys = list(getattr(ModelCls, "PREP_OUTPUT_KEYS", ()) or [])
    return prepare_frames_with_model(
        frames,
        prepare_inputs,
        cfg,
        prep_output_keys=prep_output_keys,
        verbose=debug,
    )


# ---------------------------------------------------------------------------
# Dataset / batching
# ---------------------------------------------------------------------------

def _build_loader(
    frames: List[Dict[str, Any]],
    *,
    required_keys: Sequence[str],
    batch_size: int,
    dataset_mode: str,
    variable_qm: bool,
    seed: int,
) -> torch.utils.data.DataLoader:
    dataset = QMMMDataset(frames, input_keys=required_keys, mode=dataset_mode)
    if variable_qm:
        sampler = BucketBatchSampler(
            dataset,
            local_indices=list(range(len(dataset))),
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            drop_last=False,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_batch,
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_batch,
    )


# ---------------------------------------------------------------------------
# Metrics + masks
# ---------------------------------------------------------------------------

def _build_mask_fn(
    masks_cfg: Dict[str, str],
    arrays: Dict[str, np.ndarray],
):
    def pick(name: str) -> Optional[np.ndarray]:
        if name in masks_cfg and masks_cfg[name] in arrays:
            return arrays[masks_cfg[name]]
        for pat, mk in masks_cfg.items():
            if any(c in pat for c in "*?[") and fnmatch(name, pat):
                if mk in arrays:
                    return arrays[mk]
        if name.startswith("qm_"):
            if QM_TYPE_KEY in arrays and arrays[QM_TYPE_KEY].ndim == 2:
                return (arrays[QM_TYPE_KEY] > 0.5).astype(np.bool_)
            if Keys.QM_Z in arrays and arrays[Keys.QM_Z].ndim == 2:
                return (arrays[Keys.QM_Z] != 0).astype(np.bool_)
        if name.startswith("mm_") and Keys.MM_TYPE in arrays and arrays[Keys.MM_TYPE].ndim == 2:
            return (arrays[Keys.MM_TYPE] > 0.5).astype(np.bool_)
        return None
    return pick


def _looks_like_energy(name: str) -> bool:
    n = name.lower()
    return any(t in n for t in ("energy", "e_low", "e_high", "de"))


def _looks_like_grad(name: str) -> bool:
    n = name.lower()
    return "grad" in n or "espgrad" in n or n.endswith("_dgrad") or n.endswith("_grad")


def _looks_like_esp(name: str) -> bool:
    n = name.lower()
    return n.endswith("_esp") or n == "esp" or "espgrad" in n


def _convert_units(name: str, arr: np.ndarray, units: str) -> Tuple[np.ndarray, str]:
    if units.lower() == "ev":
        if _looks_like_energy(name):
            return arr, "eV"
        if _looks_like_grad(name):
            if _looks_like_esp(name):
                return arr, "eV/Å/e"
            return arr, "eV/Å"
        return arr, "(native)"
    f = EV_TO_KCALMOL
    if _looks_like_energy(name):
        return arr * f, "kcal/mol"
    if _looks_like_grad(name):
        if _looks_like_esp(name):
            return arr * f, "kcal/mol/Å/e"
        return arr * f, "kcal/mol/Å"
    return arr, "(native)"


def _normalize_shapes_np(pred: np.ndarray, targ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if pred.shape == targ.shape:
        return pred, targ
    if pred.ndim == 3 and targ.ndim == 2 and pred.shape[-1] == 3 and targ.shape[-1] == pred.shape[-2] * 3:
        return pred, targ.reshape(targ.shape[0], pred.shape[-2], 3)
    if targ.ndim == 3 and pred.ndim == 2 and targ.shape[-1] == 3 and pred.shape[-1] == targ.shape[-2] * 3:
        return pred.reshape(pred.shape[0], targ.shape[-2], 3), targ
    if pred.ndim == 2 and pred.shape[-1] == 1 and targ.ndim == 1 and targ.shape[0] == pred.shape[0]:
        return pred.squeeze(-1), targ
    if targ.ndim == 2 and targ.shape[-1] == 1 and pred.ndim == 1 and pred.shape[0] == targ.shape[0]:
        return pred, targ.squeeze(-1)
    raise ValueError(f"Shape mismatch: pred {pred.shape} vs targ {targ.shape}")


def _flatten_vec(
    pred: np.ndarray,
    true: np.ndarray,
    mask: Optional[np.ndarray],
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if pred.ndim == 3 and pred.shape[-1] == 3:
        m2: Optional[np.ndarray] = None
        if mask is not None:
            m2 = mask.astype(bool)
            if m2.ndim == 3:
                m2 = m2[..., 0]
            elif m2.ndim > 2:
                m2 = m2.any(axis=-1)

        if mode == "magnitude":
            pm = np.linalg.norm(pred.reshape(-1, 3), axis=1)
            tm = np.linalg.norm(true.reshape(-1, 3), axis=1)
            if m2 is not None:
                mf = m2.reshape(-1)
                return pm[mf], tm[mf], None
            return pm, tm, None

        if mode == "component":
            p = pred.reshape(-1)
            t = true.reshape(-1)
            if m2 is not None:
                mf = np.repeat(m2.reshape(-1), 3)
                return p[mf], t[mf], None
            return p, t, None

        raise ValueError(f"Unknown vector_mode '{mode}'")

    p = pred.reshape(-1)
    t = true.reshape(-1)
    m = mask.reshape(-1) if mask is not None else None
    return p, t, m


def _masked_mae_rmse_r2(
    pred: np.ndarray,
    true: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    if mask is not None:
        m = mask.astype(bool)
        pred = pred[m]
        true = true[m]
    if pred.size == 0:
        return float("nan"), float("nan"), float("nan")
    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return mae, rmse, r2


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _annotate(ax: Any, mae: float, rmse: float, r2: float) -> None:
    ax.text(
        0.05, 0.95,
        f"MAE={mae:.3e}\nRMSE={rmse:.3e}\nR²={r2:.4f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
    )


def _save_scatter_and_density(
    key: str,
    true: np.ndarray,
    pred: np.ndarray,
    unit_lbl: str,
    out_dir: str,
    *,
    max_points: int,
    rng: random.Random,
) -> Tuple[str, str]:
    idx: Any
    if len(true) > max_points > 0:
        idx = sorted(rng.sample(range(len(true)), max_points))
    else:
        idx = slice(None)

    tx = true[idx]
    px = pred[idx]
    if tx.size == 0:
        return "", ""

    lim = float(max(1e-8, np.max(np.abs(np.concatenate([tx, px])))))
    diag = [-lim, lim]

    scatter_path = os.path.join(out_dir, f"scatter_{key}.png")
    density_path = os.path.join(out_dir, f"density_{key}.png")

    mae, rmse, r2 = _masked_mae_rmse_r2(px, tx, None)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(tx, px, s=5, alpha=0.35)
    ax.plot(diag, diag, "--", linewidth=1.0)
    ax.set_xlabel(f"True [{unit_lbl}]")
    ax.set_ylabel(f"Pred [{unit_lbl}]")
    ax.set_title(key)
    _annotate(ax, mae, rmse, r2)
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    hb = ax.hexbin(tx, px, gridsize=80, mincnt=1)
    fig.colorbar(hb, ax=ax)
    ax.plot(diag, diag, "--", linewidth=1.0)
    ax.set_xlabel(f"True [{unit_lbl}]")
    ax.set_ylabel(f"Pred [{unit_lbl}]")
    ax.set_title(key)
    _annotate(ax, mae, rmse, r2)
    fig.tight_layout()
    fig.savefig(density_path, dpi=180)
    plt.close(fig)

    return scatter_path, density_path


def _plot_training_curves(logs_csv: str, out_dir: str) -> List[str]:
    if not logs_csv or not os.path.exists(logs_csv):
        return []
    with open(logs_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    saved: List[str] = []
    train_rows = [r for r in rows if str(r.get("split", "")).lower() == "train"]
    val_rows = [r for r in rows if str(r.get("split", "")).lower() == "val"]

    def _series(rs: List[Dict[str, str]], field: str) -> Tuple[List[int], List[float]]:
        xs: List[int] = []
        ys: List[float] = []
        for r in rs:
            try:
                xs.append(int(r["epoch"]))
                ys.append(float(r[field]))
            except Exception:
                continue
        return xs, ys

    # total
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    xt, yt = _series(train_rows, "total_loss")
    xv, yv = _series(val_rows, "total_loss")
    if xt and yt:
        ax.plot(xt, yt, label="train")
    if xv and yv:
        ax.plot(xv, yv, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.set_title("Training / validation total loss")
    if ax.has_data():
        ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "loss_total.png")
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(p)

    # per-key RMSE columns are named "<key>__rmse"
    if rows:
        keys = [k[:-6] for k in rows[0].keys() if k.endswith("__rmse")]
        for key in sorted(keys):
            field = f"{key}__rmse"
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            xt, yt = _series(train_rows, field)
            xv, yv = _series(val_rows, field)
            if xt and yt:
                ax.plot(xt, yt, label="train")
            if xv and yv:
                ax.plot(xv, yv, label="val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("RMSE")
            ax.set_title(key)
            if ax.has_data():
                ax.legend()
            fig.tight_layout()
            p = os.path.join(out_dir, f"loss_{key}.png")
            fig.savefig(p, dpi=180)
            plt.close(fig)
            saved.append(p)

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    ap = argparse.ArgumentParser(description="Standalone mlp_qmmm model evaluation")
    ap.add_argument("--run_dir", default="", help="Directory containing training artefacts")
    ap.add_argument("--summary", default="", help="Path to training_summary.json")
    ap.add_argument("--model_io", default="", help="Path to model_io.json")
    ap.add_argument("--parse_summary", default="", help="Path to parse_summary.json")
    ap.add_argument("--config", default="", help="Path to config_used.yml or training config")
    ap.add_argument("--model_pt", default="", help="Path to model .pt (TorchScript or state-dict/checkpoint)")
    ap.add_argument("--logs_csv", default="", help="Path to logs/train_val_metrics.csv")
    ap.add_argument("--test_data", default="", help="Path to test data (folder, file, glob)")
    ap.add_argument("--adapter", default="", help="Override adapter name")
    ap.add_argument("--out_dir", default="", help="Output directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--units", choices=["ev", "kcal"], default="ev")
    ap.add_argument("--vector_mode", choices=["magnitude", "component"], default="magnitude")
    ap.add_argument("--max_points", type=int, default=200_000)
    ap.add_argument("--dataset_mode", choices=["lazy", "stacked"], default="")
    ap.add_argument("--keys", nargs="*", default=[], help="Override evaluated target keys")
    ap.add_argument("--save_data_dir", default="", help="Optional directory for per-key true/pred CSV")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--variable_qm", action="store_true", help="Force composition-aware batching")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    resolved = _resolve_run_artifacts(args)
    out_dir = resolved["out_dir"]
    plots_dir = os.path.join(out_dir, "plots_test")
    os.makedirs(plots_dir, exist_ok=True)
    if args.save_data_dir:
        os.makedirs(args.save_data_dir, exist_ok=True)

    summary = _read_json(resolved["summary"])
    model_io = _read_json(resolved["model_io"])
    parse_summary = _read_json(resolved["parse_summary"])
    cfg = _read_yaml(resolved["config"])

    meta = _resolve_metadata(
        summary, model_io, parse_summary, cfg,
        explicit_adapter=args.adapter,
        explicit_keys=args.keys,
    )

    model_name = meta["model_name"]
    required_keys = list(meta["required_keys"])
    output_keys = list(meta["output_keys"])
    allowed_loss_keys = list(meta["allowed_loss_keys"])
    loss_keys = list(meta["loss_keys"])
    masks_cfg = dict(meta["masks_cfg"])
    adapter = meta["adapter"]

    if not required_keys:
        raise ValueError(
            "Could not resolve required model input keys from training_summary.json "
            "or model_io.json."
        )
    if not loss_keys:
        raise ValueError(
            "Could not resolve evaluation target keys from --keys, training_summary.json, "
            "or model_io.json."
        )

    if args.debug:
        print(f"[test] resolved files: {json.dumps(resolved, indent=2)}")
        print(f"[test] model_name         : {model_name}")
        print(f"[test] adapter            : {adapter}")
        print(f"[test] required_keys      : {required_keys}")
        print(f"[test] output_keys        : {output_keys}")
        print(f"[test] allowed_loss_keys  : {allowed_loss_keys}")
        print(f"[test] loss_keys          : {loss_keys}")
        print(f"[test] masks              : {masks_cfg}")

    # Step 1 — parse test data with the same high-level pipeline used by training
    frames = _prepare_test_frames(
        resolved["test_data"],
        adapter=adapter,
        cfg=cfg,
        summary=summary,
        parse_summary=parse_summary,
        out_dir=out_dir,
        debug=args.debug,
    )

    # Step 2 — model-specific input preparation (e.g. atom_types)
    prep_info = _maybe_prepare_with_model(
        frames,
        cfg=cfg,
        model_name=model_name,
        debug=args.debug,
    )
    if args.debug:
        print(f"[test] prepare_inputs info: {prep_info}")

    # Step 3 — validate required inputs exist after preparation
    missing = {k: sum(1 for f in frames if k not in f) for k in required_keys}
    missing = {k: c for k, c in missing.items() if c > 0}
    if missing:
        raise KeyError(
            "Required model input keys missing after parsing / preparation:\n" +
            "\n".join(f"  - {k}: missing in {c}/{len(frames)} frames" for k, c in sorted(missing.items()))
        )

    dataset = QMMMDataset(
        frames,
        input_keys=required_keys,
        mode=(args.dataset_mode or str((cfg.get("trainer", {}) or {}).get("dataset_mode", "lazy"))),
    )
    available_targets = set(dataset.available_target_keys())
    eval_keys = [k for k in loss_keys if k in available_targets]
    skipped_targets = [k for k in loss_keys if k not in available_targets]

    if not eval_keys:
        raise ValueError(
            "None of the requested evaluation keys are present in the parsed test dataset.\n"
            f"Requested: {loss_keys}\n"
            f"Available targets: {sorted(available_targets)}"
        )

    variable_qm = bool(args.variable_qm or (cfg.get("trainer", {}) or {}).get("variable_qm", False))
    loader = _build_loader(
        frames,
        required_keys=required_keys,
        batch_size=max(1, int(args.batch_size)),
        dataset_mode=dataset.mode,
        variable_qm=variable_qm,
        seed=args.seed,
    )

    # Step 4 — load model
    device = torch.device(args.device)
    model, is_script = _load_eval_model(
        resolved["model_pt"],
        cfg,
        device,
        debug=args.debug,
    )
    model.eval()

    # Step 5 — inference
    preds_lists: Dict[str, List[torch.Tensor]] = {k: [] for k in eval_keys}
    true_lists: Dict[str, List[torch.Tensor]] = {k: [] for k in eval_keys}
    batch_arrays: Dict[str, List[np.ndarray]] = {}
    first_out_keys: List[str] = []

    with torch.set_grad_enabled(True):
        for cpu_batch in loader:
            batch = move_batch_to_device(cpu_batch, device)
            sanitize_padded_batch(batch)

            inputs = [batch[k] for k in required_keys]
            out = model(*inputs)
            if not isinstance(out, dict):
                raise TypeError(
                    f"Model forward must return Dict[str, Tensor], got {type(out)}."
                )

            if not first_out_keys:
                first_out_keys = sorted(out.keys())

            # stash full input/mask arrays for later mask building
            for k in (required_keys + [QM_TYPE_KEY, Keys.QM_Z, Keys.MM_TYPE]):
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch_arrays.setdefault(k, []).append(batch[k].detach().cpu().numpy())

            targets = batch.get("targets", {})
            for k in eval_keys:
                if k not in out or k not in targets:
                    continue
                preds_lists[k].append(out[k].detach().cpu())
                true_lists[k].append(targets[k].detach().cpu())

    if args.debug:
        print(f"[test] model output keys (first batch): {first_out_keys}")

    preds_np: Dict[str, np.ndarray] = {}
    true_np: Dict[str, np.ndarray] = {}
    for k in eval_keys:
        if preds_lists[k] and true_lists[k]:
            preds_np[k] = torch.cat(preds_lists[k], dim=0).numpy()
            true_np[k] = torch.cat(true_lists[k], dim=0).numpy()

    arrays_for_masks: Dict[str, np.ndarray] = {}
    for k, blocks in batch_arrays.items():
        if blocks:
            arrays_for_masks[k] = np.concatenate(blocks, axis=0)
    pick_mask = _build_mask_fn(masks_cfg, arrays_for_masks)

    # Step 6 — metrics, plots, save data
    rng = random.Random(args.seed)
    metrics_json: Dict[str, Any] = {}
    metrics_rows: List[List[str]] = [[
        "key", "units", "N", "MAE", "RMSE", "R2",
        "true_min", "true_max", "pred_min", "pred_max",
    ]]
    combined_save: Dict[str, np.ndarray] = {}

    for key in eval_keys:
        if key not in preds_np or key not in true_np:
            continue

        pred = preds_np[key]
        true = true_np[key]
        pred, true = _normalize_shapes_np(pred, true)

        pred_u, unit_lbl = _convert_units(key, pred, args.units)
        true_u, _ = _convert_units(key, true, args.units)

        mask = pick_mask(key)
        pflat, tflat, mflat = _flatten_vec(pred_u, true_u, mask, args.vector_mode)
        mae, rmse, r2 = _masked_mae_rmse_r2(pflat, tflat, mflat)

        npts = int(len(pflat) if mflat is None else mflat.astype(bool).sum())
        tmin = float(np.min(tflat)) if tflat.size else float("nan")
        tmax = float(np.max(tflat)) if tflat.size else float("nan")
        pmin = float(np.min(pflat)) if pflat.size else float("nan")
        pmax = float(np.max(pflat)) if pflat.size else float("nan")

        metrics_json[key] = {
            "units": unit_lbl,
            "N": npts,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "true_min": tmin,
            "true_max": tmax,
            "pred_min": pmin,
            "pred_max": pmax,
        }
        metrics_rows.append([
            key, unit_lbl, str(npts),
            f"{mae:.6e}", f"{rmse:.6e}", f"{r2:.6f}",
            f"{tmin:.6e}", f"{tmax:.6e}", f"{pmin:.6e}", f"{pmax:.6e}",
        ])
        print(
            f"[test] {key:24s} MAE={mae:.4e} RMSE={rmse:.4e} "
            f"R²={r2:.4f} [{unit_lbl}] N={npts}"
        )

        if mflat is None:
            combined_save[f"{key}_true"] = tflat
            combined_save[f"{key}_pred"] = pflat
        else:
            m = mflat.astype(bool)
            combined_save[f"{key}_true"] = tflat[m]
            combined_save[f"{key}_pred"] = pflat[m]

        if args.save_data_dir:
            csv_path = os.path.join(args.save_data_dir, f"{key}_true_pred.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["true", "pred"])
                if mflat is None:
                    for a, b in zip(tflat, pflat):
                        w.writerow([f"{a:.10e}", f"{b:.10e}"])
                else:
                    mb = mflat.astype(bool)
                    for i, (a, b) in enumerate(zip(tflat, pflat)):
                        if mb[i]:
                            w.writerow([f"{a:.10e}", f"{b:.10e}"])

        if tflat.size > 0 and pflat.size > 0:
            _save_scatter_and_density(
                key, tflat, pflat, unit_lbl, plots_dir,
                max_points=max(0, int(args.max_points)),
                rng=rng,
            )

    with open(os.path.join(out_dir, "test_metrics.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerows(metrics_rows)

    with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "is_torchscript": bool(is_script),
                "evaluated_keys": eval_keys,
                "skipped_requested_targets": skipped_targets,
                "metrics": metrics_json,
            },
            f,
            indent=2,
        )

    if combined_save:
        np.savez(os.path.join(out_dir, "predictions_test.npz"), **combined_save)

    # Step 7 — training curves if logs are available
    saved_plots = _plot_training_curves(resolved["logs_csv"], plots_dir)

    # Step 8 — model stats and bridge metadata
    model_stats: Dict[str, Any] = {
        "model_pt": resolved["model_pt"],
        "is_torchscript": bool(is_script),
        "file_size_bytes": os.path.getsize(resolved["model_pt"]) if resolved["model_pt"] and os.path.exists(resolved["model_pt"]) else -1,
    }
    try:
        model_stats["parameters_total"] = int(sum(p.numel() for p in model.parameters()))
        model_stats["parameters_trainable"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    except Exception:
        pass
    with open(os.path.join(out_dir, "model_stats.json"), "w", encoding="utf-8") as f:
        json.dump(model_stats, f, indent=2)

    max_qm = 0
    max_mm = 0
    species = set()
    for fr in frames:
        if Keys.QM_COORDS in fr:
            max_qm = max(max_qm, int(np.asarray(fr[Keys.QM_COORDS]).shape[0]))
        if Keys.MM_COORDS in fr:
            max_mm = max(max_mm, int(np.asarray(fr[Keys.MM_COORDS]).shape[0]))
        if Keys.QM_Z in fr:
            z = np.asarray(fr[Keys.QM_Z]).reshape(-1)
            species.update(int(x) for x in z if int(x) > 0)

    with open(os.path.join(out_dir, "bridge_meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"model_name={model_name}\n")
        f.write(f"max_qm={max_qm}\n")
        f.write(f"max_mm={max_mm}\n")
        f.write(f"species_Z={sorted(species)}\n")
        f.write(f"required_keys={required_keys}\n")
        f.write(f"eval_keys={eval_keys}\n")

    with open(os.path.join(out_dir, "resolved_inputs.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "resolved_paths": resolved,
                "adapter": adapter,
                "required_keys": required_keys,
                "output_keys": output_keys,
                "allowed_loss_keys": allowed_loss_keys,
                "loss_keys_requested": loss_keys,
                "loss_keys_evaluated": eval_keys,
                "prep_info": prep_info,
                "dataset_mode": dataset.mode,
                "variable_qm": variable_qm,
                "training_curves_written": saved_plots,
            },
            f,
            indent=2,
        )

    print(f"[test] out_dir   : {out_dir}")
    print(f"[test] model_pt  : {resolved['model_pt']}")
    print(f"[test] test_data : {resolved['test_data']}")
    print(f"[test] metrics   : {os.path.join(out_dir, 'test_metrics.json')}")
    print(f"[test] plots     : {plots_dir}")


if __name__ == "__main__":
    main()
