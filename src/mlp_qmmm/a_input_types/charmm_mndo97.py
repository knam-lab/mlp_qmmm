"""
Parse-only adapter for CHARMM MNDO97 MTS logs (v1.x)

This adapter fills raw canonical frame keys and performs only format-local work:
- reads CHARMM MNDO97 MTS frames
- converts units to canonical output units
- pads QM-backed arrays to max_qm with zeros
- pads MM-backed arrays to max_mm with zeros
- preserves file-read MM type values for real MM rows, but padded MM rows use 0.0
- optionally masks zero-charge MM sites by forcing mm_type := 0.0 and zeroing MM grads
- optionally derives mm_espgrad_high / mm_espgrad_low when the required keys exist

Filled directly by this adapter
------------------------------
Counts / metadata:
  N_qm, N_mm, source_file, frame_idx

QM raw keys:
  qm_idx, qm_Z, qm_Q, qm_type, qm_coords, qm_grad_high, qm_grad_low

MM raw keys:
  mm_idx, mm_type, mm_Q, mm_coords, mm_grad_high, mm_grad_low

Energy keys:
  E_high, E_low

Conditionally derived here:
  mm_espgrad_high, mm_espgrad_low

Not filled here:
  qm_energy, qm_grad, species_order, atom_types,
  dE / demeaned values / delta gradients / postprocess-derived targets

Canonical units on output
-------------------------
  Energies:   eV
  Coordinates: Å
  Gradients:  eV/Å
  ESP-grad:   eV/(Å·e)
  Charges:    e
  Counts/idx: int32
  Types:      float32  (1.0=real for this adapter, 0.0=pad/dummy)

Format notes (CHARMM MNDO97 MTS)
--------------------------------
  '!E:' line:  E_high, E_low in kcal/mol (converted here → eV)
  Atom lines:  last 11 columns used:
               [Z_or_type, Q, x, y, z, gHx, gHy, gHz, gLx, gLy, gLz]
  Gradient columns are kcal/mol/Å (converted here → eV/Å)

Registered adapter names: "charmm_mndo97" (preferred), "charmmmndo97mts" (legacy)
"""

from __future__ import annotations

import glob
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterator, List, Sequence, Tuple, Union

import numpy as np

from mlp_qmmm.a2_parser import register_adapter
from mlp_qmmm.a0_structure import Keys

try:
    from mlp_qmmm.a0_structure import Defaults  # type: ignore
except ImportError:  # pragma: no cover
    class Defaults:
        MAX_QM = 100
        MAX_MM = 5000

try:
    from mlp_qmmm.a0_structure import Units  # type: ignore
except ImportError:  # pragma: no cover
    class Units:
        KCALMOL_TO_EV = 0.0433641153087705
        EV_TO_KCALMOL = 1.0 / KCALMOL_TO_EV

PathLike = Union[str, os.PathLike]
Frame = Dict[str, np.ndarray]

QM_TYPE_KEY = getattr(Keys, "QM_TYPE", "qm_type")
MAX_QM_KEY = getattr(Keys, "MAX_QM", "max_qm")
MAX_MM_KEY = getattr(Keys, "MAX_MM", "max_mm")
MM_ESPGRAD_HIGH_KEY = getattr(Keys, "MM_ESPGRAD_HIGH", "mm_espgrad_high")
MM_ESPGRAD_LOW_KEY = getattr(Keys, "MM_ESPGRAD_LOW", "mm_espgrad_low")


def _expand_paths(path_or_glob: PathLike) -> List[str]:
    p = str(path_or_glob)
    if os.path.isdir(p):
        files = [
            os.path.join(p, fn)
            for fn in os.listdir(p)
            if os.path.isfile(os.path.join(p, fn))
        ]
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
            raise ValueError(
                f"Row too short at line {i}: need ≥{keep_cols}, got {row.size}"
            )
        out[r] = row[-keep_cols:]
        i += 1
    return out


def _read_first_tokens_as_int(lines: List[str], start: int, nrows: int) -> np.ndarray:
    idxs = np.empty((nrows,), dtype=np.int32)
    i = start
    for r in range(nrows):
        idxs[r] = int(float(lines[i].split()[0]))
        i += 1
    return idxs


def _pad1d(a: np.ndarray, L: int, *, fill: float = 0.0, dtype=np.float32) -> np.ndarray:
    out = np.full((L,), fill, dtype=dtype)
    if a.size:
        out[: a.size] = a.astype(dtype, copy=False)
    return out


def _pad1d_int(a: np.ndarray, L: int, *, fill: int = 0, dtype=np.int32) -> np.ndarray:
    out = np.full((L,), fill, dtype=dtype)
    if a.size:
        out[: a.size] = a.astype(dtype, copy=False)
    return out


def _pad3(a: np.ndarray, L: int, *, fill: float = 0.0, dtype=np.float32) -> np.ndarray:
    out = np.full((L, 3), fill, dtype=dtype)
    if a.size:
        out[: a.shape[0], :] = a.astype(dtype, copy=False)
    return out


def _maybe_compute_mm_espgrad(
    rec: Frame,
    *,
    esp_eps: float,
    warn_prefix: str = "[charmm_mndo97]",
) -> None:
    needed = [Keys.MM_Q, Keys.MM_GRAD_HIGH, Keys.MM_GRAD_LOW]
    missing = [k for k in needed if k not in rec]
    if missing:
        warnings.warn(
            f"{warn_prefix} cannot derive mm_espgrad_high/low; missing keys: {', '.join(missing)}",
            stacklevel=2,
        )
        return

    mm_q = rec[Keys.MM_Q]
    mm_gh = rec[Keys.MM_GRAD_HIGH]
    mm_gl = rec[Keys.MM_GRAD_LOW]

    if mm_q.ndim != 1 or mm_gh.ndim != 2 or mm_gl.ndim != 2 or mm_gh.shape[1] != 3 or mm_gl.shape[1] != 3:
        warnings.warn(
            f"{warn_prefix} cannot derive mm_espgrad_high/low; incompatible shapes for mm_Q/mm_grad_*",
            stacklevel=2,
        )
        return
    if mm_gh.shape[0] != mm_q.shape[0] or mm_gl.shape[0] != mm_q.shape[0]:
        warnings.warn(
            f"{warn_prefix} cannot derive mm_espgrad_high/low; length mismatch between mm_Q and mm_grad_*",
            stacklevel=2,
        )
        return

    q64 = mm_q.astype(np.float64, copy=False)
    invq = np.zeros_like(q64)
    good = np.abs(q64) > esp_eps
    invq[good] = 1.0 / q64[good]
    rec[MM_ESPGRAD_HIGH_KEY] = (mm_gh.astype(np.float64, copy=False) * invq[:, None]).astype(np.float32, copy=False)
    rec[MM_ESPGRAD_LOW_KEY] = (mm_gl.astype(np.float64, copy=False) * invq[:, None]).astype(np.float32, copy=False)



def _parse_one_frame(
    lines: List[str],
    k: int,
    *,
    max_qm: int,
    max_mm: int,
    mm_charge_zero_eps: float = 0.0,
    compute_mm_espgrad: bool = True,
    esp_eps: float = 1e-12,
) -> Tuple[Frame, int]:
    _expect_header(lines, k, "!E:")
    parts = lines[k].split()
    if len(parts) < 3:
        raise ValueError(f"!E: line must have two numbers after '!E:'; got: {lines[k]}")
    E_high = float(parts[1]) * Units.KCALMOL_TO_EV
    E_low = float(parts[2]) * Units.KCALMOL_TO_EV
    k += 1

    _expect_header(lines, k, "!QM region:")
    N_qm = int(lines[k].split()[2])
    k += 1
    if N_qm > max_qm:
        raise ValueError(
            f"N_qm={N_qm} exceeds max_qm={max_qm}. Increase adapter_kwargs.max_qm or reduce QM selection."
        )

    qm_idx_raw = _read_first_tokens_as_int(lines, k, N_qm)
    qm = _read_tail(lines, k, N_qm, 11)
    k += N_qm

    qm_Z_raw = qm[:, 0].astype(np.int32, copy=False)
    qm_Q_raw = qm[:, 1].astype(np.float32, copy=False)
    qm_xyz_raw = qm[:, 2:5].astype(np.float32, copy=False)
    qm_gH_raw = (qm[:, 5:8] * Units.KCALMOL_TO_EV).astype(np.float32, copy=False)
    qm_gL_raw = (qm[:, 8:11] * Units.KCALMOL_TO_EV).astype(np.float32, copy=False)
    qm_type_raw = np.ones((N_qm,), dtype=np.float32)

    _expect_header(lines, k, "!MM region:")
    N_mm_raw = int(lines[k].split()[2])
    k += 1
    if N_mm_raw > max_mm:
        raise ValueError(
            f"N_mm={N_mm_raw} exceeds max_mm={max_mm}. Increase adapter_kwargs.max_mm or reduce MM selection."
        )

    mm_idx_raw = _read_first_tokens_as_int(lines, k, N_mm_raw)
    mm = _read_tail(lines, k, N_mm_raw, 11)
    k += N_mm_raw

    mm_type_raw = mm[:, 0].astype(np.float32, copy=False)
    mm_Q_raw = mm[:, 1].astype(np.float32, copy=False)
    mm_xyz_raw = mm[:, 2:5].astype(np.float32, copy=False)
    mm_gH_raw = (mm[:, 5:8] * Units.KCALMOL_TO_EV).astype(np.float32, copy=False)
    mm_gL_raw = (mm[:, 8:11] * Units.KCALMOL_TO_EV).astype(np.float32, copy=False)

    qm_idx = _pad1d_int(qm_idx_raw, max_qm, fill=0, dtype=np.int32)
    qm_Z = _pad1d_int(qm_Z_raw, max_qm, fill=0, dtype=np.int32)
    qm_Q = _pad1d(qm_Q_raw, max_qm, fill=0.0, dtype=np.float32)
    qm_type = _pad1d(qm_type_raw, max_qm, fill=0.0, dtype=np.float32)
    qm_xyz = _pad3(qm_xyz_raw, max_qm, fill=0.0, dtype=np.float32)
    qm_gH = _pad3(qm_gH_raw, max_qm, fill=0.0, dtype=np.float32)
    qm_gL = _pad3(qm_gL_raw, max_qm, fill=0.0, dtype=np.float32)

    mm_idx = _pad1d_int(mm_idx_raw, max_mm, fill=0, dtype=np.int32)
    mm_type = _pad1d(mm_type_raw, max_mm, fill=0.0, dtype=np.float32)
    mm_Q = _pad1d(mm_Q_raw, max_mm, fill=0.0, dtype=np.float32)
    mm_xyz = _pad3(mm_xyz_raw, max_mm, fill=0.0, dtype=np.float32)
    mm_gH = _pad3(mm_gH_raw, max_mm, fill=0.0, dtype=np.float32)
    mm_gL = _pad3(mm_gL_raw, max_mm, fill=0.0, dtype=np.float32)

    if mm_charge_zero_eps >= 0.0:
        mask0 = np.isfinite(mm_Q) & (np.abs(mm_Q) <= mm_charge_zero_eps)
        if np.any(mask0):
            mm_type[mask0] = 0.0
            mm_gH[mask0, :] = 0.0
            mm_gL[mask0, :] = 0.0

    rec: Frame = {
        Keys.N_QM: np.asarray([N_qm], dtype=np.int32),
        Keys.N_MM: np.asarray([N_mm_raw], dtype=np.int32),
        MAX_QM_KEY: np.asarray([max_qm], dtype=np.int32),
        MAX_MM_KEY: np.asarray([max_mm], dtype=np.int32),
        Keys.E_HIGH: np.asarray([E_high], dtype=np.float32),
        Keys.E_LOW: np.asarray([E_low], dtype=np.float32),
        Keys.QM_IDX: qm_idx,
        Keys.QM_Z: qm_Z,
        Keys.QM_Q: qm_Q,
        QM_TYPE_KEY: qm_type,
        Keys.QM_COORDS: qm_xyz,
        Keys.QM_GRAD_HIGH: qm_gH,
        Keys.QM_GRAD_LOW: qm_gL,
        Keys.MM_IDX: mm_idx,
        Keys.MM_TYPE: mm_type,
        Keys.MM_Q: mm_Q,
        Keys.MM_COORDS: mm_xyz,
        Keys.MM_GRAD_HIGH: mm_gH,
        Keys.MM_GRAD_LOW: mm_gL,
    }
    if compute_mm_espgrad:
        _maybe_compute_mm_espgrad(rec, esp_eps=esp_eps)
    return rec, k



def _parse_file(args: Tuple[str, int, int, float, bool, float]) -> Tuple[str, List[Frame]]:
    fp, max_qm, max_mm, mm_charge_zero_eps, compute_mm_espgrad, esp_eps = args
    with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    frames: List[Frame] = []
    k = 0
    frame_idx = 0
    while k < len(lines):
        if not lines[k].startswith("!E:"):
            k += 1
            continue
        rec, k = _parse_one_frame(
            lines,
            k,
            max_qm=max_qm,
            max_mm=max_mm,
            mm_charge_zero_eps=mm_charge_zero_eps,
            compute_mm_espgrad=compute_mm_espgrad,
            esp_eps=esp_eps,
        )
        rec[Keys.SOURCE_FILE] = fp
        rec[Keys.FRAME_IDX] = frame_idx
        frame_idx += 1
        frames.append(rec)
    return fp, frames



def iter_charmm_mndo97_frames(
    paths: Union[Sequence[PathLike], PathLike],
    *,
    max_qm: int = Defaults.MAX_QM,
    max_mm: int = Defaults.MAX_MM,
    workers: int = 1,
    mp_chunk: int = 4,
    mm_charge_zero_eps: float = 0.0,
    compute_mm_espgrad: bool = True,
    esp_eps: float = 1e-12,
    verbose: bool = False,
    progress_every: int = 50,
) -> Iterator[Frame]:
    if isinstance(paths, (str, os.PathLike)):
        files = _expand_paths(paths)
    else:
        files = []
        for p in paths:
            files.extend(_expand_paths(p))
    files = sorted(files)

    if verbose:
        print(f"[charmm_mndo97] discovered {len(files)} file(s)")

    if workers <= 1:
        for i, fp in enumerate(files, start=1):
            _, frames = _parse_file((fp, max_qm, max_mm, mm_charge_zero_eps, compute_mm_espgrad, esp_eps))
            yield from frames
            if verbose and (i % progress_every == 0 or i == len(files)):
                print(f"[charmm_mndo97] parsed {i}/{len(files)} files")
        return

    results: Dict[str, List[Frame]] = {}
    for start in range(0, len(files), mp_chunk * workers):
        batch = files[start : start + mp_chunk * workers]
        if verbose:
            print(f"[charmm_mndo97] submit batch {start + 1}..{start + len(batch)} of {len(files)}")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(_parse_file, (fp, max_qm, max_mm, mm_charge_zero_eps, compute_mm_espgrad, esp_eps))
                for fp in batch
            ]
            for fut in as_completed(futs):
                fp, frames = fut.result()
                results[fp] = frames
        if verbose:
            done = min(start + len(batch), len(files))
            print(f"[charmm_mndo97] completed {done}/{len(files)} files")

    for fp in files:
        yield from results.get(fp, [])



def read_charmm_mndo97(
    path: Union[Sequence[PathLike], PathLike],
    *,
    max_qm: int = Defaults.MAX_QM,
    max_mm: int = Defaults.MAX_MM,
    workers: int = 1,
    mp_chunk: int = 4,
    mm_charge_zero_eps: float = 0.0,
    compute_mm_espgrad: bool = True,
    esp_eps: float = 1e-12,
    verbose: bool = False,
    progress_every: int = 50,
    **_ignored,
) -> List[Frame]:
    if _ignored and verbose:
        print(
            "[charmm_mndo97] ignoring unrecognised kwargs: "
            + ", ".join(sorted(_ignored.keys()))
        )
    return list(
        iter_charmm_mndo97_frames(
            path,
            max_qm=max_qm,
            max_mm=max_mm,
            workers=workers,
            mp_chunk=mp_chunk,
            mm_charge_zero_eps=mm_charge_zero_eps,
            compute_mm_espgrad=compute_mm_espgrad,
            esp_eps=esp_eps,
            verbose=verbose,
            progress_every=progress_every,
        )
    )


register_adapter("charmm_mndo97", read_charmm_mndo97)
register_adapter("charmmmndo97mts", read_charmm_mndo97)
