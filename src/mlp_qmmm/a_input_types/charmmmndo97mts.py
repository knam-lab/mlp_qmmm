from __future__ import annotations
from typing import Dict, Iterator, List, Sequence, Tuple, Union, Optional
import os, glob, time, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from mlp_qmmm.a_parser import register_adapter

"""
Adapter for CHARMM MNDO97 MTS logs.

Key points:
- Units converted to eV (KCALMOL_TO_EV).
- Canonical MM charge name is mm_Q (no "mm_charges").
- Pads MM arrays to max_mm and zeros out pads; mm_type = {1 real, 0 pad}.
- Zero-charge MM sites: mm_type is forced to 0 (mask) when |mm_Q| <= mm_charge_zero_eps (default 0.0 for exact zero).
- Exposes gradients for high, low, and delta; also ESP-like per-charge grads = (∂E/∂R)/q.
- Filters outliers (Tukey) on E_high/E_low and optionally on dE, plus optional grad-magnitude filters.
- Computes stats (means/std) on KEPT frames only.
- Writes a SINGLE energy target under key "energy". If using a dE mode, also writes a back-compat alias "dE".

Supported adapter_kwargs:
  max_mm: int = 5000
  workers: int = 1
  mp_chunk: int = 4
  compute_deltas: bool = True

  # Energy target transform:
  #   diff | diff_demean | diff_divmean | diff_zscore
  #   high | high_demean | low | low_demean
  energy_mode: Optional[str] = None
  demean: bool = True

  # Energy outlier filtering (applied BEFORE stats/transform)
  energy_outlier: bool = True
  energy_outlier_fence_k: float = 3.5
  de_outlier: bool = False
  de_outlier_fence_k: float = 3.5

  # Gradient outlier filtering (per-frame magnitude; QM + MM)
  grad_outlier: bool = False
  grad_outlier_fence_k: float = 3.5
  grad_outlier_keys: Sequence[str] = [
      "qm_grad_high", "qm_grad_low", "qm_dgrad",
      "mm_grad_high", "mm_grad_low", "mm_dgrad"
  ]
  grad_outlier_metric: str = "mean_l2"   # mean_l2 | max_l2 | median_l2

  # Mask dummy MM sites:
  mm_charge_zero_eps: float = 0.0  # <= this threshold → mm_type := 0

  # Parse summary sidecar files (written BEFORE yielding to training)
  write_summary: bool = True
  summary_json_path: Optional[str] = None  # default: <common-dir>/parse_summary.json
  summary_txt_path:  Optional[str] = None  # default: <common-dir>/parse_summary.txt

  verbose: bool = False
  progress_every: int = 50

Format notes:
- In '!E:' line, first number is E_high, second is E_low (kcal/mol).
- For gradient blocks, cols 5:8 = HIGH, cols 8:11 = LOW.
"""

KCALMOL_TO_EV = 0.0433641153087705
EV_TO_KCALMOL = 1.0 / KCALMOL_TO_EV
PathLikeOrGlob = Union[str, os.PathLike]
SCI = "{:.6e}"

def _atomic_write(path: str, data: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)

def _expand_paths(path_or_glob: PathLikeOrGlob) -> List[str]:
    p = str(path_or_glob)
    if os.path.isdir(p):
        files = [os.path.join(p, fn) for fn in os.listdir(p)
                 if os.path.isfile(os.path.join(p, fn))]
        files.sort()
        return files
    if os.path.isfile(p):
        return [p]
    files = sorted(glob.glob(p))
    if not files:
        raise FileNotFoundError(f"No input files matched: {path_or_glob}")
    return files

def _parse_numeric_row(line: str) -> np.ndarray:
    return np.asarray([float(t) for t in line.strip().split()], dtype=np.float64)

def _expect_header(lines: List[str], i: int, prefix: str) -> None:
    if i >= len(lines) or not lines[i].startswith(prefix):
        got = lines[i][:80] if i < len(lines) else "<EOF>"
        raise ValueError(f"Expected header '{prefix}', got: {got}")

def _read_tail(lines: List[str], start: int, nrows: int, keep_cols: int) -> np.ndarray:
    out = np.empty((nrows, keep_cols), dtype=np.float64)
    i = start
    for r in range(nrows):
        row = _parse_numeric_row(lines[i])
        if row.size < keep_cols:
            raise ValueError(f"Row too short at line {i}: need ≥{keep_cols}, got {row.size}")
        out[r] = row[-keep_cols:]
        i += 1
    return out

def _read_first_tokens_as_int(lines: List[str], start: int, nrows: int) -> np.ndarray:
    idxs = np.empty((nrows,), dtype=np.int32)
    i = start
    for _ in range(nrows):
        tok0 = lines[i].split()[0]
        idxs[i - start] = int(float(tok0))
        i += 1
    return idxs

def _pad1d(a: np.ndarray, L: int, fill: float=0.0, dtype=np.float32) -> np.ndarray:
    out = np.full((L,), fill, dtype=dtype)
    if a.size:
        out[:a.size] = a.astype(dtype, copy=False)
    return out

def _pad3(a: np.ndarray, L: int, fill: float=0.0, dtype=np.float32) -> np.ndarray:
    out = np.full((L,3), fill, dtype=dtype)
    if a.size:
        out[:a.shape[0], :] = a.astype(dtype, copy=False)
    return out

def _parse_one_frame_abs(
    lines: List[str],
    k: int,
    *,
    max_mm: int,
    mm_charge_zero_eps: float = 0.0,
    esp_eps: float = 1e-12,
) -> Tuple[Dict[str, np.ndarray], int]:
    _expect_header(lines, k, "!E:")
    parts = lines[k].split()
    if len(parts) < 3:
        raise ValueError(f"!E: line must have two numbers after '!E:'; got: {lines[k]}")
    # First token is E_high, second is E_low, both in kcal/mol → convert to eV for storage
    E_high = float(parts[1]) * KCALMOL_TO_EV
    E_low  = float(parts[2]) * KCALMOL_TO_EV
    k += 1

    _expect_header(lines, k, "!QM region:")
    N_qm = int(lines[k].split()[2]); k += 1
    qm_idx = _read_first_tokens_as_int(lines, k, N_qm)
    qm = _read_tail(lines, k, N_qm, 11); k += N_qm
    qm_Z, qm_Q = qm[:,0].astype(np.int32, copy=False), qm[:,1].astype(np.float32, copy=False)
    qm_xyz = qm[:,2:5].astype(np.float32, copy=False)
    # FIRST triplet is HIGH, SECOND is LOW
    qm_gH  = (qm[:,5:8]  * KCALMOL_TO_EV).astype(np.float32, copy=False)
    qm_gL  = (qm[:,8:11] * KCALMOL_TO_EV).astype(np.float32, copy=False)

    _expect_header(lines, k, "!MM region:")
    N_mm_raw = int(lines[k].split()[2]); k += 1
    mm_idx = _read_first_tokens_as_int(lines, k, N_mm_raw)
    mm = _read_tail(lines, k, N_mm_raw, 11); k += N_mm_raw
    mm_type = mm[:,0].astype(np.float32, copy=False)   # 1 real, 0 pad
    mm_Q    = mm[:,1].astype(np.float32, copy=False)
    mm_xyz  = mm[:,2:5].astype(np.float32, copy=False)
    # FIRST triplet is HIGH, SECOND is LOW
    mm_gH   = (mm[:,5:8]  * KCALMOL_TO_EV).astype(np.float32, copy=False)
    mm_gL   = (mm[:,8:11] * KCALMOL_TO_EV).astype(np.float32, copy=False)

    if N_mm_raw > max_mm:
        raise ValueError(f"N_mm={N_mm_raw} exceeds max_mm={max_mm}. Increase adapter_kwargs.max_mm or reduce MM selection.")

    mm_idx_pad  = np.zeros((max_mm,), dtype=np.int32)
    if mm_idx.size:
        mm_idx_pad[:mm_idx.size] = mm_idx
    mm_type_pad = _pad1d(mm_type, max_mm, fill=0.0, dtype=np.float32)
    mm_Q_pad    = _pad1d(mm_Q,    max_mm, fill=0.0, dtype=np.float32)
    mm_xyz_pad  = _pad3(mm_xyz,   max_mm, fill=0.0, dtype=np.float32)
    mm_gH_pad   = _pad3(mm_gH,    max_mm, fill=0.0, dtype=np.float32)
    mm_gL_pad   = _pad3(mm_gL,    max_mm, fill=0.0, dtype=np.float32)

    # Mask zero-charge MM sites (dummy atoms)
    if mm_charge_zero_eps >= 0.0:
        zero_q_mask = np.isfinite(mm_Q_pad) & (np.abs(mm_Q_pad) <= mm_charge_zero_eps)
        if np.any(zero_q_mask):
            mm_type_pad[zero_q_mask] = 0.0
            mm_gH_pad[zero_q_mask, :] = 0.0
            mm_gL_pad[zero_q_mask, :] = 0.0

    # ESP-like per-charge gradient = (∂E/∂R)/q (guard zero/near-zero charges)
    q64 = mm_Q_pad.astype(np.float64, copy=False)
    invq = np.zeros_like(q64)
    mask = np.abs(q64) > esp_eps
    invq[mask] = 1.0 / q64[mask]
    espH = (mm_gH_pad.astype(np.float64) * invq[:, None]).astype(np.float32, copy=False)
    espL = (mm_gL_pad.astype(np.float64) * invq[:, None]).astype(np.float32, copy=False)

    rec: Dict[str, np.ndarray] = {
        "E_low":  np.asarray([E_low],  dtype=np.float32),   # eV
        "E_high": np.asarray([E_high], dtype=np.float32),   # eV
        "N_qm": np.asarray([N_qm], dtype=np.int32),
        "N_mm": np.asarray([N_mm_raw], dtype=np.int32),

        "qm_idx": qm_idx.astype(np.int32, copy=False),
        "qm_Z":   qm_Z,
        "qm_Q":   qm_Q,
        "qm_coords": qm_xyz,
        "qm_grad_low":  qm_gL,   # eV/Å
        "qm_grad_high": qm_gH,   # eV/Å

        "mm_idx":    mm_idx_pad,
        "mm_type":   mm_type_pad,
        "mm_Q":      mm_Q_pad,
        "mm_coords": mm_xyz_pad,
        "mm_grad_low":  mm_gL_pad,   # eV/Å
        "mm_grad_high": mm_gH_pad,   # eV/Å
        "mm_espgrad_low":  espL,     # eV/(Å·e)
        "mm_espgrad_high": espH,     # eV/(Å·e)
    }
    return rec, k

def _parse_file_abs(args: Tuple[str, int, float]) -> Tuple[str, List[Dict[str, np.ndarray]]]:
    fp, max_mm, mm_charge_zero_eps = args
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    frames: List[Dict[str, np.ndarray]] = []
    k = 0
    while k < len(lines):
        if not lines[k].startswith("!E:"):
            k += 1
            continue
        rec, k = _parse_one_frame_abs(lines, k, max_mm=max_mm, mm_charge_zero_eps=mm_charge_zero_eps)
        rec["__file__"] = np.asarray([fp], dtype=object)
        frames.append(rec)
    return fp, frames

def iter_charmmmndo97mts_frames(
    paths: Union[Sequence[PathLikeOrGlob], PathLikeOrGlob],
    *,
    max_mm: int = 5000,
    workers: int = 1,
    mp_chunk: int = 4,
    compute_deltas: bool = True,
    # energy scaling
    demean: bool = True,
    energy_mode: Optional[str] = None,  # None | diff | diff_demean | diff_divmean | diff_zscore | high | high_demean | low | low_demean
    # energy outlier filtering
    energy_outlier: bool = True,
    energy_outlier_fence_k: float = 3.5,
    de_outlier: bool = False,
    de_outlier_fence_k: float = 3.5,
    # Gradient outlier controls (QM + MM)
    grad_outlier: bool = False,
    grad_outlier_fence_k: float = 3.5,
    grad_outlier_keys: Optional[Sequence[str]] = None,
    grad_outlier_metric: str = "mean_l2",  # mean_l2 | max_l2 | median_l2
    # Mask zero-charge MM sites
    mm_charge_zero_eps: float = 0.0,
    # NEW: summary sidecars
    write_summary: bool = True,
    summary_json_path: Optional[str] = None,
    summary_txt_path: Optional[str] = None,
    verbose: bool = False,
    progress_every: int = 50,
) -> Iterator[Dict[str, np.ndarray]]:
    """Parse → optional outlier filters (energy, dE, QM/MM-grad) → compute stats on KEPT → write summary → write targets → yield KEPT only."""
    t0 = time.time()

    # ---------- Build file list ----------
    if isinstance(paths, (str, os.PathLike)):
        files = _expand_paths(paths)
    else:
        files: List[str] = []
        for p in paths:
            files.extend(_expand_paths(p))
    files = sorted(files)
    if verbose:
        print(f"[parse] discovered {len(files)} file(s)")

    # ---------- Phase 1: parse per file ----------
    results: Dict[str, List[Dict[str, np.ndarray]]] = {}
    if workers <= 1:
        if verbose:
            print("[parse] workers=1 → sequential parsing")
        for i, fp in enumerate(files, start=1):
            _, frames = _parse_file_abs((fp, max_mm, mm_charge_zero_eps))
            results[fp] = frames
            if verbose and (i % progress_every == 0 or i == len(files)):
                print(f"[parse] parsed {i}/{len(files)} files | elapsed {time.time()-t0:.1f}s")
    else:
        if verbose:
            print(f"[parse] multiprocessing: workers={workers}, mp_chunk={mp_chunk}")
        for start in range(0, len(files), mp_chunk * workers):
            batch = files[start:start + mp_chunk * workers]
            if verbose:
                print(f"[parse] submit batch: {start+1}..{start+len(batch)} of {len(files)}")
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_parse_file_abs, (fp, max_mm, mm_charge_zero_eps)) for fp in batch]
                for _fut in as_completed(futs):
                    fp, frames = _fut.result()
                    results[fp] = frames
            if verbose:
                done = min(start+len(batch), len(files))
                print(f"[parse] completed {done}/{len(files)} files | elapsed {time.time()-t0:.1f}s")

    # ---------- Phase 1.5: collect in stable order ----------
    all_frames: List[Dict[str, np.ndarray]] = []
    for fp in files:
        all_frames.extend(results.get(fp, []))
    if not all_frames:
        if verbose:
            print("[parse] no frames found; exiting")
        return

    # ---------- Phase 2: outlier filtering (energies and optional gradients) ----------
    E_low_all  = np.concatenate([f["E_low"]  for f in all_frames]).astype(np.float64, copy=False)
    E_high_all = np.concatenate([f["E_high"] for f in all_frames]).astype(np.float64, copy=False)
    dE_all     = (E_high_all - E_low_all)

    def _tukey_keep(values: np.ndarray, k: float) -> np.ndarray:
        q1 = float(np.nanpercentile(values, 25.0))
        q3 = float(np.nanpercentile(values, 75.0))
        iqr = max(q3 - q1, 1e-12)
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        v = values.astype(np.float64, copy=False)
        return (v >= lo) & (v <= hi)

    if energy_outlier or de_outlier:
        keep_lo = _tukey_keep(E_low_all,  energy_outlier_fence_k) if energy_outlier else np.ones_like(E_low_all,  dtype=bool)
        keep_hi = _tukey_keep(E_high_all, energy_outlier_fence_k) if energy_outlier else np.ones_like(E_high_all, dtype=bool)
        keep_de = _tukey_keep(dE_all,     de_outlier_fence_k)     if de_outlier     else np.ones_like(dE_all,     dtype=bool)
        keep = keep_lo & keep_hi & keep_de
    else:
        keep = np.ones(len(all_frames), dtype=bool)
        if verbose:
            print("[parse] energy-outlier filter: disabled")

    dropped_grad_ranges: List[Tuple[str, float, float]] = []
    if grad_outlier:
        keys = list(grad_outlier_keys) if grad_outlier_keys else [
            "qm_grad_high", "qm_grad_low", "qm_dgrad",
            "mm_grad_high", "mm_grad_low", "mm_dgrad"
        ]

        def _reduce_mags(mags: np.ndarray) -> float:
            if mags.size == 0:
                return 0.0
            if grad_outlier_metric == "max_l2":
                return float(np.max(mags))
            elif grad_outlier_metric == "median_l2":
                return float(np.median(mags))
            else:
                return float(np.mean(mags))

        def _frame_score_qm(arr: np.ndarray) -> float:
            mags = np.linalg.norm(arr.astype(np.float64, copy=False), axis=1)
            return _reduce_mags(mags)

        def _frame_score_mm(arr: np.ndarray, mm_type: np.ndarray) -> float:
            mask = (mm_type.astype(np.float64, copy=False) > 0.5)
            if not np.any(mask):
                return 0.0
            mags = np.linalg.norm(arr[mask].astype(np.float64, copy=False), axis=1)
            return _reduce_mags(mags)

        def _collect_scores(key: str) -> np.ndarray:
            scores = np.empty((len(all_frames),), dtype=np.float64)
            if key == "qm_dgrad":
                for i, f in enumerate(all_frames):
                    scores[i] = _frame_score_qm(f["qm_grad_high"] - f["qm_grad_low"])
            elif key in ("qm_grad_high", "qm_grad_low"):
                for i, f in enumerate(all_frames):
                    scores[i] = _frame_score_qm(f[key])
            elif key == "mm_dgrad":
                for i, f in enumerate(all_frames):
                    scores[i] = _frame_score_mm(f["mm_grad_high"] - f["mm_grad_low"], f["mm_type"])
            elif key in ("mm_grad_high", "mm_grad_low"):
                for i, f in enumerate(all_frames):
                    scores[i] = _frame_score_mm(f[key], f["mm_type"])
            else:
                # unrecognized → try to infer
                for i, f in enumerate(all_frames):
                    arr = f[key]
                    if arr.ndim == 2 and arr.shape[1] == 3 and "mm_type" in f and arr.shape[0] == f["mm_type"].shape[0]:
                        scores[i] = _frame_score_mm(arr, f["mm_type"])
                    else:
                        scores[i] = _frame_score_qm(arr)
            return scores

        for kname in keys:
            try:
                sc = _collect_scores(kname)
            except KeyError:
                continue
            keep_k = _tukey_keep(sc, grad_outlier_fence_k)
            dropped = sc[~keep_k]
            lo = float(np.nanmin(dropped)) if dropped.size else 0.0
            hi = float(np.nanmax(dropped)) if dropped.size else 0.0
            dropped_grad_ranges.append((kname, lo, hi))
            keep = keep & keep_k

    kept_frames = [f for f, kf in zip(all_frames, keep) if kf]
    if len(kept_frames) == 0:
        kept_frames = all_frames
        if verbose:
            print("[parse] outlier filter(s) would drop all frames → disabled for this dataset.")
    else:
        if verbose:
            total = len(all_frames); dropped = total - len(kept_frames)
            def _rng(arr):
                return (float(np.nanmin(arr)) if arr.size else 0.0,
                        float(np.nanmax(arr)) if arr.size else 0.0)
            lo_rng = _rng(E_low_all[~_tukey_keep(E_low_all, energy_outlier_fence_k)])  if energy_outlier else (0.0, 0.0)
            hi_rng = _rng(E_high_all[~_tukey_keep(E_high_all, energy_outlier_fence_k)]) if energy_outlier else (0.0, 0.0)
            de_rng = _rng(dE_all[~_tukey_keep(dE_all, de_outlier_fence_k)])             if de_outlier     else (0.0, 0.0)
            line = (f"[parse] outlier filter: kept {len(kept_frames)}/{total} (dropped {dropped}) | "
                    f"E_low dropped [{SCI.format(lo_rng[0])}, {SCI.format(lo_rng[1])}] | "
                    f"E_high dropped [{SCI.format(hi_rng[0])}, {SCI.format(hi_rng[1])}] | "
                    f"dE dropped [{SCI.format(de_rng[0])}, {SCI.format(de_rng[1])}]")
            if dropped_grad_ranges:
                for nm, lo, hi in dropped_grad_ranges:
                    line += f" | {nm} dropped [{SCI.format(lo)}, {SCI.format(hi)}]"
            print(line)

    # ---------- Phase 3: compute stats on KEPT frames ----------
    E_low_kept  = np.concatenate([f["E_low"]  for f in kept_frames]).astype(np.float64, copy=False)   # eV
    E_high_kept = np.concatenate([f["E_high"] for f in kept_frames]).astype(np.float64, copy=False)   # eV
    dE_kept     = (E_high_kept - E_low_kept)  # eV

    E_low_mean_eV   = float(E_low_kept.mean())
    E_high_mean_eV  = float(E_high_kept.mean())
    dE_mean_eV      = float(dE_kept.mean())
    dE_std_eV       = float(dE_kept.std(ddof=0))

    # kcal/mol versions (what you asked for)
    E_low_mean_kcal  = E_low_mean_eV  * EV_TO_KCALMOL
    E_high_mean_kcal = E_high_mean_eV * EV_TO_KCALMOL
    dE_mean_kcal     = dE_mean_eV     * EV_TO_KCALMOL
    dE_std_kcal      = dE_std_eV      * EV_TO_KCALMOL

    eps = 1e-8
    denom_divmean = max(abs(dE_mean_eV), eps)
    denom_zscore  = max(dE_std_eV, eps)

    if energy_mode is None:
        energy_mode = "diff_demean" if demean else "diff"
    valid_modes = {"diff","diff_demean","diff_divmean","diff_zscore","high","high_demean","low","low_demean"}
    if energy_mode not in valid_modes:
        raise ValueError(f"energy_mode must be one of {sorted(valid_modes)}, got {energy_mode}")
    if verbose:
        print(
            f"[parse] energy_mode={energy_mode} | (kept) "
            f"E_high_mean={SCI.format(E_high_mean_eV)} eV, "
            f"E_low_mean={SCI.format(E_low_mean_eV)} eV, "
            f"dE_mean={SCI.format(dE_mean_eV)} eV, dE_std={SCI.format(dE_std_eV)} eV"
        )

    # ---------- NEW: write parse summary (JSON + TXT) BEFORE yielding ----------
    if write_summary:
        # default dir = common parent of input files (fallback CWD)
        try:
            base_dir = os.path.commonpath([os.path.abspath(os.path.dirname(f)) for f in files])
        except ValueError:
            base_dir = os.getcwd()
        json_path = summary_json_path or os.path.join(base_dir, "parse_summary.json")
        txt_path  = summary_txt_path  or os.path.join(base_dir, "parse_summary.txt")

        summary = {
            "adapter": "charmmmndo97mts",
            "timestamp": float(time.time()),
            "files": len(files),
            "counts": {
                "total_frames": int(len(all_frames)),
                "kept_frames": int(len(kept_frames)),
                "dropped_frames": int(len(all_frames) - len(kept_frames)),
            },
            "settings": {
                "energy_mode": energy_mode,
                "energy_outlier": bool(energy_outlier),
                "energy_outlier_fence_k": float(energy_outlier_fence_k),
                "de_outlier": bool(de_outlier),
                "de_outlier_fence_k": float(de_outlier_fence_k),
                "grad_outlier": bool(grad_outlier),
                "grad_outlier_fence_k": float(grad_outlier_fence_k),
                "grad_outlier_keys": list(grad_outlier_keys) if grad_outlier_keys else None,
                "grad_outlier_metric": grad_outlier_metric,
                "mm_charge_zero_eps": float(mm_charge_zero_eps),
            },
            "stats_eV": {
                "E_high_mean": E_high_mean_eV,
                "E_low_mean":  E_low_mean_eV,
                "dE_mean":     dE_mean_eV,
                "dE_std":      dE_std_eV,
            },
            "stats_kcal_per_mol": {
                "E_high_mean": E_high_mean_kcal,
                "E_low_mean":  E_low_mean_kcal,
                "dE_mean":     dE_mean_kcal,
                "dE_std":      dE_std_kcal,
            },
            "notes": "Means/std computed over KEPT frames after outlier/gradient filtering. Energies in frames are stored as eV."
        }
        _atomic_write(json_path, json.dumps(summary, indent=2, sort_keys=True))

        txt_lines = [
            "=== QMMM Parse Summary ===",
            f"adapter: charmmndo97mts",
            f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(summary['timestamp']))}",
            f"files: {len(files)}",
            f"frames: total={len(all_frames)} kept={len(kept_frames)} dropped={len(all_frames)-len(kept_frames)}",
            "",
            f"settings:",
            f"  energy_mode = {energy_mode}",
            f"  energy_outlier = {energy_outlier} (k={energy_outlier_fence_k})",
            f"  de_outlier = {de_outlier} (k={de_outlier_fence_k})",
            f"  grad_outlier = {grad_outlier} (k={grad_outlier_fence_k}, metric={grad_outlier_metric})",
            f"  grad_outlier_keys = {list(grad_outlier_keys) if grad_outlier_keys else 'default'}",
            f"  mm_charge_zero_eps = {mm_charge_zero_eps}",
            "",
            "stats (eV):",
            f"  E_high_mean = {SCI.format(E_high_mean_eV)}",
            f"  E_low_mean  = {SCI.format(E_low_mean_eV)}",
            f"  dE_mean     = {SCI.format(dE_mean_eV)}",
            f"  dE_std      = {SCI.format(dE_std_eV)}",
            "",
            "stats (kcal/mol):",
            f"  E_high_mean = {SCI.format(E_high_mean_kcal)}",
            f"  E_low_mean  = {SCI.format(E_low_mean_kcal)}",
            f"  dE_mean     = {SCI.format(dE_mean_kcal)}",
            f"  dE_std      = {SCI.format(dE_std_kcal)}",
            "",
            "Note: summary written before dataset is yielded to training.",
            ""
        ]
        _atomic_write(txt_path, "\n".join(txt_lines))

        if verbose:
            print(f"[parse] wrote summary JSON → {json_path}")
            print(f"[parse] wrote summary TXT  → {txt_path}")

    # ---------- Phase 4: write targets per kept frame ----------
    is_diff_mode = energy_mode.startswith("diff")
    for f in kept_frames:
        f["E_high_dm"] = f["E_high"].astype(np.float32) - np.float32(E_high_mean_eV)
        f["E_low_dm"]  = f["E_low"].astype(np.float32)  - np.float32(E_low_mean_eV)

        if energy_mode == "diff":
            y = (f["E_high"] - f["E_low"]).astype(np.float32, copy=False)
        elif energy_mode == "diff_demean":
            y = (f["E_high"].astype(np.float32) - np.float32(E_high_mean_eV)) \
              - (f["E_low"].astype(np.float32)  - np.float32(E_low_mean_eV))
        elif energy_mode == "diff_divmean":
            y = ((f["E_high"] - f["E_low"]) / np.float32(denom_divmean)).astype(np.float32, copy=False)
        elif energy_mode == "diff_zscore":
            dE_frame = (f["E_high"] - f["E_low"]).astype(np.float32, copy=False)
            y = ((dE_frame - np.float32(dE_mean_eV)) / np.float32(denom_zscore)).astype(np.float32, copy=False)
        elif energy_mode == "high":
            y = f["E_high"].astype(np.float32, copy=False)
        elif energy_mode == "high_demean":
            y = (f["E_high"].astype(np.float32) - np.float32(E_high_mean_eV)).astype(np.float32, copy=False)
        elif energy_mode == "low":
            y = f["E_low"].astype(np.float32, copy=False)
        elif energy_mode == "low_demean":
            y = (f["E_low"].astype(np.float32) - np.float32(E_low_mean_eV)).astype(np.float32, copy=False)
        else:
            raise AssertionError("unreachable")

        f["energy"] = y
        if is_diff_mode:
            f["dE"] = y

        if compute_deltas:
            f["qm_dgrad"]     = (f["qm_grad_high"]    - f["qm_grad_low"]).astype(np.float32, copy=False)
            f["mm_dgrad"]     = (f["mm_grad_high"]    - f["mm_grad_low"]).astype(np.float32, copy=False)
            f["mm_espgrad_d"] = (f["mm_espgrad_high"] - f["mm_espgrad_low"]).astype(np.float32, copy=False)

    # ---------- Phase 5: yield ONLY KEPT frames ----------
    total = len(kept_frames)
    if verbose:
        print(f"[parse] yielding {total} frame(s)")
    for i, f in enumerate(kept_frames, start=1):
        if verbose and (i % (max(1, total // 10)) == 0 or i == total):
            print(f"[parse] yield progress {i}/{total}")
        yield f

def _adapter_universal(
    path: Union[Sequence[PathLikeOrGlob], PathLikeOrGlob],
    *,
    max_mm: int = 5000,
    workers: int = 1,
    mp_chunk: int = 4,
    compute_deltas: bool = True,
    demean: bool = True,
    energy_mode: Optional[str] = None,
    energy_outlier: bool = True,
    energy_outlier_fence_k: float = 3.5,
    de_outlier: bool = False,
    de_outlier_fence_k: float = 3.5,
    grad_outlier: bool = False,
    grad_outlier_fence_k: float = 3.5,
    grad_outlier_keys: Optional[Sequence[str]] = None,
    grad_outlier_metric: str = "mean_l2",
    mm_charge_zero_eps: float = 0.0,
    # NEW:
    write_summary: bool = True,
    summary_json_path: Optional[str] = None,
    summary_txt_path: Optional[str] = None,
    verbose: bool = False,
    progress_every: int = 50,
    **_legacy_ignored,
) -> List[Dict[str, np.ndarray]]:
    if _legacy_ignored:
        print("[parse] note: ignoring legacy outlier_* kwargs:", ", ".join(sorted(_legacy_ignored.keys())))
    return list(iter_charmmmndo97mts_frames(
        path,
        max_mm=max_mm,
        workers=workers,
        mp_chunk=mp_chunk,
        compute_deltas=compute_deltas,
        demean=demean,
        energy_mode=energy_mode,
        energy_outlier=energy_outlier,
        energy_outlier_fence_k=energy_outlier_fence_k,
        de_outlier=de_outlier,
        de_outlier_fence_k=de_outlier_fence_k,
        grad_outlier=grad_outlier,
        grad_outlier_fence_k=grad_outlier_fence_k,
        grad_outlier_keys=grad_outlier_keys,
        grad_outlier_metric=grad_outlier_metric,
        mm_charge_zero_eps=mm_charge_zero_eps,
        write_summary=write_summary,
        summary_json_path=summary_json_path,
        summary_txt_path=summary_txt_path,
        verbose=verbose,
        progress_every=progress_every,
    ))

register_adapter("charmmmndo97mts", _adapter_universal)
