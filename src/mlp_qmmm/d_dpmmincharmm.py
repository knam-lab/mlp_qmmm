#!/usr/bin/env python3
# TXT-only, single-frame inference for dp_qmmm_implicit
# d_dpmmincharmm.py
from __future__ import annotations

import os, sys, argparse, json
from typing import Dict, Any, Tuple, Optional

# -----------------------
# CLI (set env early)
# -----------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Independent inference driver (TXT-only) for dp_qmmm_implicit"
    )
    ap.add_argument("-pttype", type=str, default="dp_qmmm_implicit",
                    help="Model type tag (reserved for future multi-backend support)")
    ap.add_argument("-pt", type=str, required=True, help="Path to TorchScript .pt file")
    ap.add_argument("-ptinfo", type=str, required=True, help="Path to bridge meta file (json or txt)")
    ap.add_argument("-in", dest="inp", type=str, required=True, help="TXT input file (single frame)")
    ap.add_argument("-out", dest="out", type=str, required=True, help="TXT output file")
    ap.add_argument("--usegpu", type=int, default=None,
                    help="GPU index to use (sets CUDA_VISIBLE_DEVICES). Omit for CPU.")
    ap.add_argument("--usempi", type=int, default=None,
                    help="Number of MPI ranks to use. For TXT single-frame this is a no-op; rank 0 runs, others idle.")
    ap.add_argument("--units", type=str, choices=["ev", "kcal"], default="ev",
                    help="Units for writing results (energy/grad conversions). Default: ev")
    ap.add_argument("--mm_charge_zero_eps", type=float, default=None,
                    help="If set, zero-charge (|q|<=eps) MM sites are masked (mm_type=0). "
                         "Overrides value in bridge meta; default falls back to 0.0.")
    ap.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    return ap.parse_args()

# Parse once before importing torch so we can set CUDA_VISIBLE_DEVICES correctly
args_early = parse_args()
if args_early.usegpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_early.usegpu)
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

# Try MPI if requested (TXT single-frame stays rank-0 only for strict order)
_MPI_AVAILABLE = False
world_size = 1
rank = 0
if args_early.usempi and args_early.usempi > 1:
    try:
        from mpi4py import MPI  # type: ignore
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        rank = comm.Get_rank()
        _MPI_AVAILABLE = True
    except Exception:
        _MPI_AVAILABLE = False
        world_size = 1
        rank = 0
        comm = None
else:
    comm = None

# -----------------------
# Utilities
# -----------------------
EV_TO_KCALMOL = 23.060547830619
SCI_FMT = "{:.16e}"  # scientific notation, consistent everywhere

_QUIET = bool(args_early.quiet)
def log(*msg: Any) -> None:
    if not _QUIET and rank == 0:
        print(*msg, flush=True)

def load_bridge_meta(path: str) -> Dict[str, Any]:
    """
    Supports:
      - JSON: keys: species_z_ordered (list), z_to_type (dict), max_mm, n_qm, n_types, (optional) mm_charge_zero_eps
      - TXT : key value pairs, e.g.:
              n_qm 64
              max_mm 4096
              n_types 3
              species_z_ordered 1 6 8
              z_to_type 1:0 6:1 8:2
              mm_charge_zero_eps 0.0
    """
    if path.lower().endswith(".json"):
        with open(path, "r") as f:
            meta = json.load(f)
        if "species_z_ordered" not in meta and "species" in meta:
            meta["species_z_ordered"] = list(map(int, meta["species"]))
        if "z_to_type" not in meta and "species_z_ordered" in meta:
            meta["z_to_type"] = {int(z): i for i, z in enumerate(meta["species_z_ordered"])}
        return meta

    # TXT fallback
    meta: Dict[str, Any] = {}
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            key, vals = parts[0], parts[1:]
            if key in ("n_qm", "n_types", "max_mm"):
                meta[key] = int(vals[0])
            elif key in ("species_z_ordered", "species"):
                meta["species_z_ordered"] = [int(v) for v in vals]
            elif key == "z_to_type":
                z2t = {}
                for tok in vals:
                    if ":" in tok:
                        z, t = tok.split(":")
                        z2t[int(z)] = int(t)
                if z2t:
                    meta["z_to_type"] = z2t
            elif key == "mm_charge_zero_eps":
                try:
                    meta["mm_charge_zero_eps"] = float(vals[0])
                except Exception:
                    pass
            else:
                meta[key] = " ".join(vals) if len(vals) > 1 else (vals[0] if vals else "")
    if "z_to_type" not in meta and "species_z_ordered" in meta:
        meta["z_to_type"] = {int(z): i for i, z in enumerate(meta["species_z_ordered"])}
    return meta

def parse_txt_single_frame(path: str) -> Dict[str, np.ndarray]:
    """
    Format:
      line 1: N_qm  N_mm
      lines 2..(1+N_qm):  qmx qmy qmz qmcg qm_Z
      lines (2+N_qm)..(1+N_qm+N_mm):  mmx mmy mmz mmcg
    Returns raw (unpadded) arrays and counts; padding done later.
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError(f"Empty TXT input: {path}")
    try:
        first = lines[0].split()
        N_qm = int(first[0]); N_mm = int(first[1])
    except Exception as e:
        raise ValueError(f"First line must be 'N_qm N_mm': got '{lines[0]}'") from e

    if len(lines) != 1 + N_qm + N_mm:
        raise ValueError(f"File length mismatch: expected {1+N_qm+N_mm} lines, got {len(lines)}")

    qm_block = lines[1:1+N_qm]
    mm_block = lines[1+N_qm:1+N_qm+N_mm]

    qm_coords = np.zeros((N_qm, 3), dtype=np.float32)
    qm_Z      = np.zeros((N_qm,),   dtype=np.int64)
    for i, ln in enumerate(qm_block):
        toks = ln.split()
        if len(toks) < 5:
            raise ValueError(f"QM line {i+2} must have 5 columns: qmx qmy qmz qmcg qm_Z")
        qm_coords[i] = [float(toks[0]), float(toks[1]), float(toks[2])]
        qm_Z[i]      = int(float(toks[4]))

    mm_coords = np.zeros((N_mm, 3), dtype=np.float32)
    mm_Q      = np.zeros((N_mm,),   dtype=np.float32)
    for j, ln in enumerate(mm_block):
        toks = ln.split()
        if len(toks) < 4:
            raise ValueError(f"MM line {j+2+N_qm} must have 4 columns: mmx mmy mmz mmcg")
        mm_coords[j] = [float(toks[0]), float(toks[1]), float(toks[2])]
        mm_Q[j]      = float(toks[3])

    return {
        "N_qm": np.array([N_qm], dtype=np.int64),
        "N_mm": np.array([N_mm], dtype=np.int64),
        "qm_coords_raw": qm_coords,
        "qm_Z_raw": qm_Z,
        "mm_coords_raw": mm_coords,
        "mm_Q_raw": mm_Q,
    }

def map_atom_types(qm_Z: np.ndarray, z_to_type: Dict[int,int]) -> np.ndarray:
    at = np.empty_like(qm_Z, dtype=np.int64)
    uniq = np.unique(qm_Z)
    for z in uniq:
        zi = int(z)
        if zi not in z_to_type:
            raise KeyError(f"Atomic number {zi} not found in bridge meta z_to_type map")
        at[qm_Z == zi] = z_to_type[zi]
    return at

def pad_mm(
    mm_coords_raw: np.ndarray,
    mm_Q_raw: np.ndarray,
    max_mm: int,
    mm_charge_zero_eps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pads MM blocks to (max_mm,3)/(max_mm,) and builds mm_type (1=active, 0=pad/dummy).
    Also flips mm_type to 0 for zero/near-zero charge sites: |q|<=mm_charge_zero_eps.
    """
    N_mm = mm_coords_raw.shape[0]
    if N_mm > max_mm:
        raise ValueError(f"N_mm={N_mm} exceeds max_mm={max_mm} from bridge meta")
    mm_coords = np.zeros((max_mm, 3), dtype=np.float32)
    mm_Q      = np.zeros((max_mm,),   dtype=np.float32)
    mm_type   = np.zeros((max_mm,),   dtype=np.float32)
    if N_mm > 0:
        mm_coords[:N_mm] = mm_coords_raw
        mm_Q[:N_mm]      = mm_Q_raw
        # start as real
        mm_type[:N_mm]   = 1.0
        # flip to dummy where |q|<=eps
        if mm_charge_zero_eps >= 0.0:
            zero_mask = np.isfinite(mm_Q[:N_mm]) & (np.abs(mm_Q[:N_mm]) <= mm_charge_zero_eps)
            mm_type[:N_mm][zero_mask] = 0.0
    return mm_coords, mm_Q, mm_type

# -----------------------
# Main
# -----------------------
def main() -> None:
    args = args_early

    if rank == 0:
        log("[info] d_inference (TXT-only) starting")
        log(f"[info] cuda_available={torch.cuda.is_available()} | requested GPU={args.usegpu} | world_size={world_size} rank={rank}")
        if args.usempi and (not _MPI_AVAILABLE or world_size == 1):
            log("[warn] --usempi requested but mpi4py/world size unavailable; proceeding single-process.")
        if args.usempi and world_size > 1:
            log("[note] TXT single-frame mode: only rank 0 performs I/O and inference to preserve strict order. Other ranks idle.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Only rank 0 performs work; others idle for TXT single-frame
    if rank != 0:
        return

    # Bridge meta
    meta = load_bridge_meta(args.ptinfo)
    z_to_type = {int(k): int(v) for k, v in meta.get("z_to_type", {}).items()}
    if not z_to_type and "species_z_ordered" in meta:
        z_to_type = {int(z): i for i, z in enumerate(meta["species_z_ordered"])}
    max_mm = int(meta.get("max_mm", 0))
    if max_mm <= 0:
        raise ValueError("bridge meta must include a positive 'max_mm' for padding")

    # zero-charge masking epsilon: CLI > meta > default 0.0
    if args.mm_charge_zero_eps is not None:
        mm_charge_zero_eps = float(args.mm_charge_zero_eps)
    else:
        mm_charge_zero_eps = float(meta.get("mm_charge_zero_eps", 0.0))

    log(f"[info] species z_to_type: {z_to_type}")
    log(f"[info] max_mm from bridge meta: {max_mm}")
    log(f"[info] mm_charge_zero_eps: {mm_charge_zero_eps:g}")

    # Input load (TXT)
    raw = parse_txt_single_frame(args.inp)
    N_qm = int(raw["N_qm"][0]); N_mm = int(raw["N_mm"][0])

    # Build inputs (batch = 1), float32 for coords/charges, int64 for types
    qm_coords = raw["qm_coords_raw"][None, ...].astype(np.float32, copy=False)  # (1,Nq,3)
    qm_Z = raw["qm_Z_raw"]
    atom_types_np = map_atom_types(qm_Z, z_to_type).reshape(1, -1).astype(np.int64, copy=False)

    mm_coords_pad, mm_Q_pad, mm_type_pad = pad_mm(
        raw["mm_coords_raw"], raw["mm_Q_raw"], max_mm, mm_charge_zero_eps
    )
    # report how many MM sites were masked as dummy
    n_dummy = int((mm_type_pad[:N_mm] <= 0.5).sum()) if N_mm > 0 else 0
    if n_dummy > 0:
        log(f"[info] masked {n_dummy}/{N_mm} MM sites as dummy (|q|<=eps)")

    mm_coords = mm_coords_pad[None, ...].astype(np.float32, copy=False)   # (1,max_mm,3)
    mm_Q      = mm_Q_pad[None, ...].astype(np.float32, copy=False)        # (1,max_mm)
    mm_type   = mm_type_pad[None, ...].astype(np.float32, copy=False)     # (1,max_mm)

    # Load model
    model = torch.jit.load(args.pt, map_location=device)
    model.eval()

    energy_factor = 1.0 if args.units == "ev" else EV_TO_KCALMOL
    grad_factor   = energy_factor  # eV/Å -> kcal/mol/Å

    # Inference
    with torch.inference_mode():
        out = model(
            torch.from_numpy(qm_coords).to(device),
            torch.from_numpy(atom_types_np).to(device),
            torch.from_numpy(mm_coords).to(device),
            torch.from_numpy(mm_Q).to(device),
            torch.from_numpy(mm_type).to(device),
        )
        if not isinstance(out, dict):
            raise RuntimeError("Model did not return a dict; expected dp_qmmm_implicit-style outputs.")

        # Energy
        e = out.get("energy", out.get("dE", None))
        if e is None:
            raise KeyError("Model output missing key 'energy' (or 'dE')")
        energy = float(e.detach().cpu().numpy().reshape(-1)[0]) * energy_factor

        # QM grads
        qg = out.get("qm_grad", out.get("qm_dgrad", None))
        qm_grad: Optional[np.ndarray] = None
        if qg is not None:
            qg_np = qg.detach().cpu().numpy()[0].astype(np.float64, copy=False) * grad_factor  # (Nq,3)
            qm_grad = qg_np

        # MM gradient field:
        # Prefer a direct mm_grad/mm_dgrad if model provides it; otherwise build from ESP-grad * q
        mm_grad: Optional[np.ndarray] = None
        if "mm_grad" in out or "mm_dgrad" in out:
            mg_key = "mm_grad" if "mm_grad" in out else "mm_dgrad"
            mg_np = out[mg_key].detach().cpu().numpy()[0].astype(np.float64, copy=False)  # (max_mm,3) in eV/Å
            # apply mask to ensure dummy/padded sites are zeroed
            mg_np = mg_np * mm_type[0][:, None].astype(np.float64, copy=False)
            mm_grad = (mg_np[:N_mm] * grad_factor)
        else:
            mg_d = out.get("mm_espgrad_d", out.get("mm_esp_grad", None))
            if mg_d is not None:
                esp_np = mg_d.detach().cpu().numpy()[0].astype(np.float64, copy=False)       # (max_mm,3) eV/(Å·e)
                # multiply by charge and mask by mm_type to zero dummy/padded sites
                mm_grad_full = esp_np * mm_Q[0][:, None].astype(np.float64, copy=False)      # eV/Å
                mm_grad_full *= mm_type[0][:, None].astype(np.float64, copy=False)           # respect mask
                mm_grad = (mm_grad_full[:N_mm] * grad_factor)                                 # (N_mm,3)

    # Write TXT output (strict scientific notation)
    with open(args.out, "w", encoding="utf-8") as f:
        # line 1: energy only
        f.write(SCI_FMT.format(energy) + "\n")

        # QM grads (if present)
        if qm_grad is not None:
            for i in range(qm_grad.shape[0]):
                dx, dy, dz = qm_grad[i]
                f.write(f"{SCI_FMT.format(dx)} {SCI_FMT.format(dy)} {SCI_FMT.format(dz)}\n")

        # MM grads (only actual N_mm lines; zeros appear for masked dummy sites)
        if mm_grad is not None:
            for i in range(mm_grad.shape[0]):
                dx, dy, dz = mm_grad[i]
                f.write(f"{SCI_FMT.format(dx)} {SCI_FMT.format(dy)} {SCI_FMT.format(dz)}\n")

    log(f"[info] wrote results to {args.out}")

if __name__ == "__main__":
    main()
