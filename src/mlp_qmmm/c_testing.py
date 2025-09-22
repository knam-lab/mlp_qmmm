#!/usr/bin/env python3
# c_testing.py  — UTF-8 safe, CSV-safe, headless-friendly
from __future__ import annotations

import argparse, json, os, math, csv, random
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

# headless-safe rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fnmatch import fnmatch


# --------- small io helpers ---------
def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    if x.dtype == np.float64:
        t = torch.from_numpy(x.astype(np.float32, copy=False))
    elif x.dtype.kind in ("f",):
        t = torch.from_numpy(x)
    elif x.dtype.kind in ("i", "u"):
        t = torch.from_numpy(x.astype(np.int64, copy=False))
    else:
        t = torch.from_numpy(x)
    return t.to(device)

def _csv_safe_cell(x: Any) -> str:
    """Best-effort string for CSV that won't raise on non-ASCII / bytes."""
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    try:
        return str(x)
    except Exception:
        return repr(x)


# --------- metrics ---------
def masked_mae_rmse_r2(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> Tuple[float, float, float]:
    if mask is not None:
        m = mask.astype(bool)
        pred = pred[m]
        true = true[m]
    if pred.size == 0:
        return float('nan'), float('nan'), float('nan')
    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    # R^2 (coefficient of determination)
    ss_res = float(np.sum(diff ** 2))
    mean_true = float(np.mean(true)) if true.size else 0.0
    ss_tot = float(np.sum((true - mean_true) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return mae, rmse, r2


# --------- shape normalization mirroring training ---------
def normalize_shapes_np(pred: np.ndarray, targ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    raise ValueError(f"[test] Shape mismatch: pred {pred.shape} vs targ {targ.shape}")


# --------- unit conversion helpers ---------
EV_TO_KCALMOL = 23.060547830619  # 1 eV = 23.06055 kcal/mol

def _looks_like_energy(name: str) -> bool:
    n = name.lower()
    return any(tok in n for tok in ("energy", "e_low", "e_high", "de"))

def _looks_like_grad(name: str) -> bool:
    n = name.lower()
    return ("grad" in n) or ("espgrad" in n) or n.endswith("_dgrad") or n.endswith("_grad")

def _looks_like_esp(name: str) -> bool:
    n = name.lower()
    return (n.endswith("_esp") or "esp" == n or "espgrad" in n)

def convert_units(name: str, arr: np.ndarray, units: str) -> Tuple[np.ndarray, str]:
    """
    Convert eV-based arrays to kcal/mol-based units when requested.
    - Energy-like: eV -> kcal/mol
    - Gradient-like (per distance): eV/Å -> kcal/mol/Å
    - ESP (per charge): eV/Å/e -> kcal/mol/Å/e
    """
    if units.lower() == "ev":
        if _looks_like_energy(name):
            return arr, "eV"
        if _looks_like_grad(name):
            # mm_espgrad_* are also "grad-like" but per charge; label below
            if _looks_like_esp(name):
                return arr, "eV/Å/e"
            return arr, "eV/Å"
        return arr, "(native)"
    factor = EV_TO_KCALMOL
    if _looks_like_energy(name):
        return arr * factor, "kcal/mol"
    if _looks_like_grad(name):
        if _looks_like_esp(name):
            return arr * factor, "kcal/mol/Å/e"
        return arr * factor, "kcal/mol/Å"
    return arr, "(native)"


# --------- vector flattening ---------
def flatten_vec(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None, mode: str
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    mode='magnitude'  : use vector magnitudes (non-negative)
    mode='component'  : flatten all components (signed)
    """
    # Vector fields (B,N,3)
    if pred.ndim == 3 and pred.shape[-1] == 3:
        m = None
        if mask is not None:
            m = mask.astype(bool)
            # If mask is vector-like, reduce it to (B,N)
            if m.ndim == pred.ndim:
                if m.shape[-1] == 3:
                    m = m[..., 0]
                else:
                    m = m.any(axis=-1)
        if mode == "magnitude":
            pm = np.linalg.norm(pred.reshape(-1, 3), axis=1)
            tm = np.linalg.norm(true.reshape(-1, 3), axis=1)
            if m is not None:
                mflat = m.reshape(-1)
                return pm[mflat], tm[mflat], None
            return pm, tm, None
        elif mode == "component":
            p = pred.reshape(-1)
            t = true.reshape(-1)
            if m is not None:
                mflat = np.repeat(m.reshape(-1), 3)
                return p[mflat], t[mflat], None
            return p, t, None
        else:
            raise ValueError(f"Unknown vector_mode '{mode}'")
    # (B,N) or (B,)
    if pred.ndim == 2:
        return pred.reshape(-1), true.reshape(-1), (mask.reshape(-1) if mask is not None else None)
    if pred.ndim == 1:
        return pred, true, mask
    # fallback
    return pred.reshape(-1), true.reshape(-1), (mask.reshape(-1) if mask is not None else None)


def stats_str(x: torch.Tensor, name: str) -> str:
    if x.numel() == 0:
        return f"{name}: EMPTY"
    return (f"{name}: shape={tuple(x.shape)} dtype={x.dtype} "
            f"min={x.min().item():.3e} max={x.max().item():.3e} "
            f"mean={x.mean().item():.3e} std={x.std(unbiased=False).item():.3e} "
            f"nnz={(x != 0).sum().item()} / {x.numel()}")


def annotate_metrics(ax, mae: float, rmse: float, r2: float):
    txt = f"MAE={mae:.3e}\nRMSE={rmse:.3e}\nR²={r2:.4f}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, ha="left", va="top")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=str, help="Directory produced by b_training.py")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_points", type=int, default=200000, help="max points for dense scatter")
    ap.add_argument("--vector_mode", type=str, choices=["magnitude", "component"], default="magnitude",
                    help="How to flatten vector fields for metrics/plots")
    ap.add_argument("--units", type=str, choices=["ev", "kcal"], default="ev",
                    help="Report metrics/plots in eV or kcal/mol (and derived units)")
    ap.add_argument("--save_data_dir", type=str, default="",
                    help="If set, save per-key CSVs of true/pred arrays (after masking & unit conversion)")
    ap.add_argument("--debug", action="store_true", help="print extra checks and first-batch summaries")
    args = ap.parse_args()

    out_dir = args.out_dir
    logs_csv = os.path.join(out_dir, "logs", "train_val_metrics.csv")
    npz_path = os.path.join(out_dir, "test_data.npz")
    io_meta_path = os.path.join(out_dir, "model_io.json")
    script_path = os.path.join(out_dir, "model_script.pt")
    plots_dir = os.path.join(out_dir, "plots_test")
    ensure_dir(plots_dir)
    if args.save_data_dir:
        ensure_dir(args.save_data_dir)

    # ---- load artifacts ----
    test_data = load_npz(npz_path)
    with open(io_meta_path, "r", encoding="utf-8") as f:
        io_meta = json.load(f)
    input_keys: List[str] = io_meta["input_keys"]
    loss_keys: List[str] = io_meta["loss_keys"]
    masks_cfg: Dict[str, str] = io_meta.get("masks", {})
    device = torch.device(args.device)

    # ---- debug dataset summary
    if args.debug:
        print("\n[debug] Keys in test_data.npz:", sorted(test_data.keys()))
        for k in input_keys:
            a = test_data[k]
            print(f"[debug] input '{k}': shape={a.shape} dtype={a.dtype}")
        for k in loss_keys:
            if k in test_data:
                a = test_data[k]
                print(f"[debug] target '{k}': shape={a.shape} dtype={a.dtype}")
        for pat, mk in masks_cfg.items():
            if mk in test_data:
                a = test_data[mk]
                print(f"[debug] mask '{mk}' (for '{pat}'): shape={a.shape} dtype={a.dtype} "
                      f"nnz={(a!=0).sum()} total={a.size}")

    # ---- build input tensors ----
    B = None
    inputs = []
    for k in input_keys:
        arr = test_data[k]
        B = arr.shape[0] if B is None else B
        inputs.append(to_tensor(arr, device))
    if B is None:
        raise RuntimeError("No inputs found in test_data.npz")

    # ---- optional masks to torch (only for debug prints)
    mask_tensors: Dict[str, torch.Tensor] = {}
    for pat, mk in masks_cfg.items():
        if mk in test_data:
            mask_tensors[mk] = to_tensor(test_data[mk], device)

    # ---- targets to torch (not required for inference, but handy for debug)
    targets: Dict[str, torch.Tensor] = {}
    for k in loss_keys:
        if k in test_data:
            targets[k] = to_tensor(test_data[k], device)

    # ---- load TorchScript model ----
    model = torch.jit.load(script_path, map_location=device)
    model.eval()
    if args.debug:
        print("\n[debug] Loaded model:", type(model))
        try:
            code_lines = model.code.splitlines()
            print("[debug] model.code (first 40 lines):")
            print("\n".join(code_lines[:40]))
        except Exception as e:
            print(f"[debug] model.code unavailable ({type(e).__name__}: {e})")

    # ---- inference ----
    preds = {k: [] for k in loss_keys}
    with torch.set_grad_enabled(True):  # TorchScript forward often needs no grad, but safe here
        bs = max(1, int(args.batch_size))
        first_out = None
        for i0 in range(0, B, bs):
            ib = slice(i0, min(B, i0 + bs))
            batch_inputs = [x[ib] for x in inputs]
            out = model(*batch_inputs)
            if first_out is None and isinstance(out, dict):
                first_out = out
            if isinstance(out, dict):
                for k in loss_keys:
                    if k in out and isinstance(out[k], torch.Tensor):
                        preds[k].append(out[k].detach().cpu())
            else:
                print("[warn] Model did not return a dict; skipping batch outputs")

    if args.debug and isinstance(first_out, dict):
        print("\n[debug] First batch outputs:")
        for k, v in first_out.items():
            if isinstance(v, torch.Tensor):
                print(" ", stats_str(v, k))

    preds_np: Dict[str, np.ndarray] = {}
    for k, blocks in preds.items():
        if not blocks:
            continue
        preds_np[k] = torch.cat(blocks, dim=0).cpu().numpy()

    # ========= NEW: derive mm_grad_* from mm_espgrad_* using mm_Q =========
    # We will create additional keys like:
    #   mm_espgrad_d   -> mm_grad_d
    #   mm_espgrad_low -> mm_grad_low
    #   mm_espgrad_high-> mm_grad_high
    derived_pairs: List[Tuple[str, str]] = []  # (src_esp_key, new_grad_key)

    def _esp_to_grad_name(src: str) -> str:
        # robust but simple: replace the first "espgrad" occurrence
        return src.replace("espgrad", "grad", 1)

    if "mm_Q" in test_data:
        Q = test_data["mm_Q"]  # shape (B, Mmax)
        if Q.ndim != 2:
            print("[warn] test_data['mm_Q'] must be 2-D (B,Mmax); skipping mm_grad derivations")
        else:
            for src in sorted(preds_np.keys()):
                # only for MM ESP grads, and only if we have true targets for the same src
                if ("mm_espgrad" in src) and (src in test_data):
                    tgt = _esp_to_grad_name(src)
                    # shapes must align: esp: (B,Mmax,3), Q: (B,Mmax)
                    esp_pred = preds_np[src]
                    esp_true = test_data[src]
                    if esp_pred.ndim == 3 and esp_pred.shape[-1] == 3 \
                       and esp_true.shape == esp_pred.shape and Q.shape[0] == esp_pred.shape[0] \
                       and Q.shape[1] == esp_pred.shape[1]:
                        # broadcast multiply per-atom, per-component
                        q3 = Q[..., None]  # (B,Mmax,1)
                        grad_pred = esp_pred * q3
                        grad_true = esp_true * q3
                        preds_np[tgt] = grad_pred
                        # inject true field for uniform handling downstream
                        test_data[tgt] = grad_true
                        derived_pairs.append((src, tgt))
                    else:
                        print(f"[warn] cannot derive '{tgt}' from '{src}': "
                              f"esp_pred shape {esp_pred.shape}, Q shape {Q.shape}")
    else:
        print("[warn] test_data lacks 'mm_Q'; cannot derive mm_grad_* from mm_espgrad_*")

    # Expand evaluation set to include derived mm_grad_* keys
    eval_keys: List[str] = list(loss_keys)
    for _, newk in derived_pairs:
        if newk not in eval_keys:
            eval_keys.append(newk)

    # ---- mask lookup with sensible fallback for MM fields ----
    # Try io_meta patterns first; if nothing matches and key looks like MM per-atom,
    # fallback to (mm_type > 0) when available.
    def maybe_mask_for(name: str) -> np.ndarray | None:
        # exact mapping
        if name in masks_cfg and masks_cfg[name] in test_data:
            return test_data[masks_cfg[name]]
        # wildcard pattern mapping
        for pat, mk in masks_cfg.items():
            if any(ch in pat for ch in "*?[") and fnmatch(name, pat):
                return test_data.get(mk, None)
        # fallback for MM atom-wise tensors
        if name.startswith("mm_") and "mm_type" in test_data:
            mt = test_data["mm_type"]
            if mt.ndim == 2:  # (B, Mmax)
                return (mt > 0.5).astype(np.bool_)
        return None

    # ---- compute metrics + plots + optional CSV dumps ----
    metrics_rows: List[List[str]] = [["key", "units", "N", "MAE", "RMSE", "R2", "true_min", "true_max", "pred_min", "pred_max"]]
    combined_save: Dict[str, np.ndarray] = {}

    rng = random.Random(1337)

    for k in eval_keys:
        if k not in preds_np or k not in test_data:
            continue
        pred = preds_np[k]
        true = test_data[k]

        # Align shapes as in training
        try:
            pred, true = normalize_shapes_np(pred, true)
        except ValueError as e:
            print(f"[warn] {e}; skipping key '{k}'")
            continue

        # Unit conversion (per-field)
        pred, units_lbl = convert_units(k, pred, args.units)
        true, _ = convert_units(k, true, args.units)

        # Mask selection (with fallback for mm_* -> mm_type)
        mask = maybe_mask_for(k)

        # Flatten per vector_mode
        pflat, tflat, mflat = flatten_vec(pred, true, mask, args.vector_mode)

        # Metrics
        mae, rmse, r2 = masked_mae_rmse_r2(pflat, tflat, mflat)
        npts = int(len(pflat) if mflat is None else (mflat.astype(bool).sum()))
        tmin = float(np.min(tflat)) if tflat.size else float('nan')
        tmax = float(np.max(tflat)) if tflat.size else float('nan')
        pmin = float(np.min(pflat)) if pflat.size else float('nan')
        pmax = float(np.max(pflat)) if pflat.size else float('nan')
        metrics_rows.append([
            _csv_safe_cell(k), _csv_safe_cell(units_lbl), str(npts),
            f"{mae:.6e}", f"{rmse:.6e}", f"{r2:.6f}",
            f"{tmin:.6e}", f"{tmax:.6e}", f"{pmin:.6e}", f"{pmax:.6e}"
        ])

        # Save arrays (optional)
        if args.save_data_dir:
            out_csv = os.path.join(args.save_data_dir, f"{k}_true_pred.csv")
            with open(out_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["true", "pred"])
                if mflat is None:
                    for a, b in zip(tflat, pflat):
                        w.writerow([f"{a:.10e}", f"{b:.10e}"])
                else:
                    m = mflat.astype(bool)
                    for a, b, mm in zip(tflat, pflat, m):
                        if mm:
                            w.writerow([f"{a:.10e}", f"{b:.10e}"])

        # Keep in combined NPZ too (already unit-converted & flattened)
        if mflat is None:
            combined_save[f"{k}_true"] = tflat
            combined_save[f"{k}_pred"] = pflat
        else:
            m = mflat.astype(bool)
            combined_save[f"{k}_true"] = tflat[m]
            combined_save[f"{k}_pred"] = pflat[m]

        # Plots: density + scatter with metrics
        if pflat.size and tflat.size:
            n = min(len(pflat), max(0, int(args.max_points)))
            if n <= 0:
                idx = slice(None)
            elif len(pflat) > n:
                # deterministic-ish sampling without numpy RNG
                idx = sorted(rng.sample(range(len(pflat)), n))
            else:
                idx = slice(None)
            px = pflat[idx]
            tx = tflat[idx]

            if px.size and tx.size:
                lim = float(max(1e-8, np.max(np.abs(np.concatenate([px, tx])))))

                # Density
                plt.figure(figsize=(5.2, 5.2))
                hb = plt.hexbin(tx, px, gridsize=80, bins='log')
                cb = plt.colorbar(hb)
                cb.set_label("log10(count)")
                plt.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1)
                plt.xlabel(f"{k} true [{units_lbl}]")
                plt.ylabel(f"{k} pred [{units_lbl}]")
                plt.title(f"{k} density ({args.vector_mode})")
                ax = plt.gca()
                annotate_metrics(ax, mae, rmse, r2)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{k}_density.png"), dpi=180)
                plt.close()

                # Scatter
                plt.figure(figsize=(5.2, 5.2))
                plt.scatter(tx, px, s=3, alpha=0.35)
                plt.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1)
                plt.xlabel(f"{k} true [{units_lbl}]")
                plt.ylabel(f"{k} pred [{units_lbl}]")
                plt.title(f"{k} scatter ({args.vector_mode})")
                ax = plt.gca()
                annotate_metrics(ax, mae, rmse, r2)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{k}_scatter.png"), dpi=180)
                plt.close()

    # --- Write metrics CSV (UTF-8 via csv.writer) ---
    out_csv_path = os.path.join(out_dir, "test_metrics.csv")
    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for row in metrics_rows:
            safe = []
            for x in row:
                if x is None:
                    safe.append("")
                elif isinstance(x, bytes):
                    safe.append(x.decode("utf-8", "replace"))
                else:
                    try:
                        safe.append(str(x))
                    except Exception:
                        safe.append(repr(x))
            w.writerow(safe)

    # Save combined predictions/targets
    np.savez_compressed(os.path.join(out_dir, "predictions_test.npz"), **combined_save)

    # ---- training curves from CSV (optional) ----
    if os.path.exists(logs_csv):
        try:
            import pandas as pd
            df = pd.read_csv(logs_csv)

            plt.figure(figsize=(6.4, 4.0))
            for split in ("train", "val"):
                sub = df[df["split"] == split]
                if not sub.empty:
                    plt.plot(sub["epoch"], sub["total_loss"], label=split)
            plt.xlabel("epoch"); plt.ylabel("total loss"); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "loss_total.png"), dpi=180); plt.close()

            for col in df.columns:
                if not col.startswith("loss_"):
                    continue
                plt.figure(figsize=(6.4, 4.0))
                for split in ("train", "val"):
                    sub = df[df["split"] == split]
                    if not sub.empty:
                        plt.plot(sub["epoch"], sub[col], label=split)
                plt.xlabel("epoch"); plt.ylabel(col); plt.legend(); plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{col}.png"), dpi=180); plt.close()
        except Exception as e:
            print(f"[warn] failed to render training curves from {logs_csv}: {type(e).__name__}: {e}")

    # ---- model statistics (parameter counts & file sizes) ----
    model_stats: Dict[str, Any] = {}
    try:
        n_params = 0
        n_trainable = 0
        for p in model.parameters():
            c = int(np.prod(p.size()))
            n_params += c
            if p.requires_grad:
                n_trainable += c
        n_buffers = 0
        for b in model.buffers():
            n_buffers += int(np.prod(b.size()))
        model_stats["num_parameters_total"] = n_params
        model_stats["num_parameters_trainable"] = n_trainable
        model_stats["num_buffers"] = n_buffers
    except Exception as e:
        model_stats["param_enum_error"] = f"{type(e).__name__}: {e}"

    # File sizes
    def _file_size(path: str) -> int:
        try:
            return os.path.getsize(path)
        except OSError:
            return -1
    model_stats["files"] = {
        "model_script.pt": _file_size(script_path),
        "model_state.pt": _file_size(os.path.join(out_dir, "model_state.pt")),
        "checkpoint.pt/best.pt": _file_size(os.path.join(out_dir, "best.pt")),
    }
    with open(os.path.join(out_dir, "model_stats.json"), "w", encoding="utf-8") as f:
        json.dump(model_stats, f, indent=2, ensure_ascii=False)

    print(
        "Test evaluation complete.\n"
        f" - Plots: {plots_dir}\n"
        f" - Metrics CSV: {os.path.join(out_dir, 'test_metrics.csv')}\n"
        f" - Predictions NPZ: {os.path.join(out_dir, 'predictions_test.npz')}\n"
        f" - Model stats: {os.path.join(out_dir, 'model_stats.json')}\n"
        + (f" - Saved per-key CSVs: {args.save_data_dir}\n" if args.save_data_dir else "")
    )


if __name__ == "__main__":
    # Force UTF-8 on stdout/stderr in notoriously ASCII-default environments
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
