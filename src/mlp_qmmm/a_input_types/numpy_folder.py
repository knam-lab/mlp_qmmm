"""
NumPy-folder adapter for mlp_qmmm (v1.x)

Philosophy
----------
Parse-only adapter for datasets stored as a folder of NumPy files (.npy / .npz).
This is the canonical round-trip format:

    0) The units are expected in canonical units as per a0_structure — no unit conversion is performed by this adapter. 
       The files must already be in the correct units.
    1) Parse raw data with any adapter (e.g. charmm_mndo97)
       → dump to a folder with dump_frames_to_numpy() in a2_parser
    2) Later: reload with this adapter ("numpy_folder") and get frames back

This adapter can also normalise frame layout to the current fixed-width padding
contract used by raw adapters:
  - QM-backed arrays may be padded to max_qm
  - MM-backed arrays may be padded to max_mm
  - qm_type / mm_type are synthesised when absent (1.0 for real rows, 0.0 pad)
  - mm_espgrad_high / mm_espgrad_low are derived when absent but mm_grad_* and
    mm_Q are available

Rules
-----
- Only keys in KEY_POOL are accepted; unknown keys are ignored with a warning.
- Arrays are cast to canonical dtypes on load (float32 for energies/coords/grads,
  int32 for atomic numbers/indices/counts, int64 for atom_types, float32 for
  qm_type/mm_type).
- Shape of each array is validated against expected ndim and last-dimension rules.
- No unit conversion is performed — files must already be in canonical units.

Registered adapter name: "numpy_folder"
"""

from __future__ import annotations

import glob
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from mlp_qmmm.a2_parser import register_adapter
from mlp_qmmm.a0_structure import Keys, KEY_POOL, NON_TENSOR_KEYS

try:
    from mlp_qmmm.a0_structure import Defaults
    _DEFAULT_MAX_QM = int(getattr(Defaults, "MAX_QM", 100))
    _DEFAULT_MAX_MM = int(getattr(Defaults, "MAX_MM", 5000))
except Exception:
    _DEFAULT_MAX_QM = 100
    _DEFAULT_MAX_MM = 5000

# Optional newer keys: keep adapter robust if structure file lags behind.
_QM_TYPE_KEY = getattr(Keys, "QM_TYPE", "qm_type")
_MAX_QM_KEY = getattr(Keys, "MAX_QM", "max_qm")
_MAX_MM_KEY = getattr(Keys, "MAX_MM", "max_mm")

PathLike = Union[str, os.PathLike]
Frame = Dict[str, Any]

_DTYPE_FLOAT32: frozenset = frozenset({
    Keys.E_HIGH, Keys.E_LOW, Keys.E_HIGH_DM, Keys.E_LOW_DM,
    Keys.DE, Keys.DE_DM,
    Keys.QM_ENERGY, Keys.QM_GRAD,
    Keys.QM_COORDS, Keys.QM_Q,
    Keys.QM_GRAD_HIGH, Keys.QM_GRAD_LOW, Keys.QM_DGRAD,
    Keys.MM_COORDS, Keys.MM_Q, Keys.MM_TYPE,
    Keys.MM_GRAD_HIGH, Keys.MM_GRAD_LOW, Keys.MM_DGRAD,
    Keys.MM_ESP_HIGH, Keys.MM_ESP_LOW, Keys.MM_DESP,
    Keys.MM_ESPGRAD_HIGH, Keys.MM_ESPGRAD_LOW, Keys.MM_DESPGRAD,
    _QM_TYPE_KEY,
})

_DTYPE_INT32: frozenset = frozenset({
    Keys.N_QM, Keys.N_MM,
    Keys.QM_Z, Keys.QM_IDX, Keys.MM_IDX,
    Keys.SPECIES_ORDER,
    _MAX_QM_KEY, _MAX_MM_KEY,
})

_DTYPE_INT64: frozenset = frozenset({Keys.ATOM_TYPES})

_EXPECTED_NDIM: Dict[str, int] = {
    Keys.N_QM: 1,
    Keys.N_MM: 1,
    Keys.E_HIGH: 1,
    Keys.E_LOW: 1,
    Keys.E_HIGH_DM: 1,
    Keys.E_LOW_DM: 1,
    Keys.DE: 1,
    Keys.DE_DM: 1,
    Keys.QM_ENERGY: 1,
    _MAX_QM_KEY: 1,
    _MAX_MM_KEY: 1,
    Keys.QM_Z: 1,
    Keys.QM_Q: 1,
    Keys.QM_IDX: 1,
    _QM_TYPE_KEY: 1,
    Keys.SPECIES_ORDER: 1,
    Keys.ATOM_TYPES: 1,
    Keys.MM_Q: 1,
    Keys.MM_TYPE: 1,
    Keys.MM_IDX: 1,
    Keys.MM_ESP_HIGH: 1,
    Keys.MM_ESP_LOW: 1,
    Keys.MM_DESP: 1,
    Keys.QM_COORDS: 2,
    Keys.QM_GRAD: 2,
    Keys.QM_GRAD_HIGH: 2,
    Keys.QM_GRAD_LOW: 2,
    Keys.QM_DGRAD: 2,
    Keys.MM_COORDS: 2,
    Keys.MM_GRAD_HIGH: 2,
    Keys.MM_GRAD_LOW: 2,
    Keys.MM_DGRAD: 2,
    Keys.MM_ESPGRAD_HIGH: 2,
    Keys.MM_ESPGRAD_LOW: 2,
    Keys.MM_DESPGRAD: 2,
}

_LAST_DIM_3: frozenset = frozenset({
    Keys.QM_COORDS, Keys.QM_GRAD,
    Keys.QM_GRAD_HIGH, Keys.QM_GRAD_LOW, Keys.QM_DGRAD,
    Keys.MM_COORDS,
    Keys.MM_GRAD_HIGH, Keys.MM_GRAD_LOW, Keys.MM_DGRAD,
    Keys.MM_ESPGRAD_HIGH, Keys.MM_ESPGRAD_LOW, Keys.MM_DESPGRAD,
})


def _expand_to_dirs(path: PathLike) -> List[str]:
    p = str(path)
    if os.path.isdir(p):
        return [os.path.abspath(p)]
    hits = sorted(glob.glob(p))
    dirs = [os.path.abspath(h) for h in hits if os.path.isdir(h)]
    if not dirs:
        raise FileNotFoundError(f"[numpy_folder] no directories matched: {path}")
    return dirs


def _stem(filepath: str) -> str:
    base = os.path.basename(filepath)
    for ext in (".npy", ".npz"):
        if base.endswith(ext):
            return base[:-len(ext)]
    return os.path.splitext(base)[0]


def _load_file(filepath: str) -> Dict[str, np.ndarray]:
    if filepath.endswith(".npy"):
        return {_stem(filepath): np.load(filepath, allow_pickle=False)}
    if filepath.endswith(".npz"):
        z = np.load(filepath, allow_pickle=False)
        return {k: z[k] for k in z.files}
    raise ValueError(f"[numpy_folder] unsupported file type: {filepath}")


def _coerce_dtype(key: str, arr: np.ndarray) -> np.ndarray:
    if key in _DTYPE_FLOAT32:
        return arr.astype(np.float32, copy=False)
    if key in _DTYPE_INT32:
        return arr.astype(np.int32, copy=False)
    if key in _DTYPE_INT64:
        return arr.astype(np.int64, copy=False)
    return arr


def _validate_per_frame(key: str, arr: np.ndarray, frame_idx: int) -> None:
    expected_ndim = _EXPECTED_NDIM.get(key)
    if expected_ndim is not None and arr.ndim != expected_ndim:
        raise ValueError(
            f"[numpy_folder] frame {frame_idx}: key '{key}' has ndim={arr.ndim}, "
            f"expected ndim={expected_ndim}."
        )
    if key in _LAST_DIM_3 and (arr.ndim < 2 or arr.shape[-1] != 3):
        raise ValueError(
            f"[numpy_folder] frame {frame_idx}: key '{key}' must have last "
            f"dimension = 3 (got shape {arr.shape})."
        )


def _infer_n_frames(arr: np.ndarray) -> int:
    return int(arr.shape[0]) if arr.ndim >= 1 else 1


def _pad1d_float(a: np.ndarray, L: int) -> np.ndarray:
    out = np.zeros((L,), dtype=np.float32)
    if a.size:
        out[:a.shape[0]] = a.astype(np.float32, copy=False)
    return out


def _pad1d_int32(a: np.ndarray, L: int) -> np.ndarray:
    out = np.zeros((L,), dtype=np.int32)
    if a.size:
        out[:a.shape[0]] = a.astype(np.int32, copy=False)
    return out


def _pad1d_int64(a: np.ndarray, L: int) -> np.ndarray:
    out = np.zeros((L,), dtype=np.int64)
    if a.size:
        out[:a.shape[0]] = a.astype(np.int64, copy=False)
    return out


def _pad2_float3(a: np.ndarray, L: int) -> np.ndarray:
    out = np.zeros((L, 3), dtype=np.float32)
    if a.size:
        out[:a.shape[0], :] = a.astype(np.float32, copy=False)
    return out


def _split_into_frames(
    key_arrays: Dict[str, np.ndarray],
    *,
    n_frames: int,
    require_same_n: bool,
    validate_shapes: bool,
) -> List[Frame]:
    frames: List[Frame] = [{} for _ in range(n_frames)]
    for key, arr in key_arrays.items():
        if arr.ndim == 0:
            for f in frames:
                f[key] = arr
            continue
        actual_n = int(arr.shape[0])
        if require_same_n and actual_n != n_frames:
            raise ValueError(
                f"[numpy_folder] key '{key}' has {actual_n} frames but expected {n_frames}."
            )
        limit = min(actual_n, n_frames)
        for i in range(limit):
            sliced = arr[i]
            if validate_shapes:
                _validate_per_frame(key, sliced, i)
            frames[i][key] = sliced
    return frames


def _derive_mm_espgrad(frame: Frame, *, esp_eps: float, warn_missing: bool) -> None:
    have_q = Keys.MM_Q in frame
    q = frame.get(Keys.MM_Q)
    if q is None or np.asarray(q).ndim != 1:
        if warn_missing:
            warnings.warn(
                "[numpy_folder] cannot derive mm_espgrad_*: missing or invalid mm_Q.",
                RuntimeWarning,
                stacklevel=2,
            )
        return

    qf = np.asarray(q, dtype=np.float32)
    invq = np.zeros_like(qf, dtype=np.float64)
    mask = np.isfinite(qf) & (np.abs(qf.astype(np.float64)) > esp_eps)
    invq[mask] = 1.0 / qf.astype(np.float64)[mask]

    for grad_key, esp_key in (
        (Keys.MM_GRAD_HIGH, Keys.MM_ESPGRAD_HIGH),
        (Keys.MM_GRAD_LOW, Keys.MM_ESPGRAD_LOW),
    ):
        if esp_key in frame:
            continue
        arr = frame.get(grad_key)
        if arr is None:
            if warn_missing:
                warnings.warn(
                    f"[numpy_folder] cannot derive {esp_key}: missing {grad_key}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] != qf.shape[0]:
            if warn_missing:
                warnings.warn(
                    f"[numpy_folder] cannot derive {esp_key}: shape mismatch between "
                    f"{grad_key} {arr.shape} and {Keys.MM_Q} {qf.shape}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue
        frame[esp_key] = (arr.astype(np.float64) * invq[:, None]).astype(np.float32, copy=False)


def _finalise_padding_and_types(
    frame: Frame,
    *,
    max_qm: int,
    max_mm: int,
    compute_mm_espgrad: bool,
    esp_eps: float,
    warn_missing_espgrad: bool,
) -> None:
    n_qm = int(np.asarray(frame.get(Keys.N_QM, [0]), dtype=np.int32).reshape(-1)[0]) if Keys.N_QM in frame else None
    n_mm = int(np.asarray(frame.get(Keys.N_MM, [0]), dtype=np.int32).reshape(-1)[0]) if Keys.N_MM in frame else None

    # QM padding
    if n_qm is not None:
        frame[_MAX_QM_KEY] = np.asarray([max_qm], dtype=np.int32)
        qm_float_1d = [Keys.QM_Q, _QM_TYPE_KEY]
        qm_int_1d = [Keys.QM_IDX, Keys.QM_Z]
        qm_float_2d = [Keys.QM_COORDS, Keys.QM_GRAD, Keys.QM_GRAD_HIGH, Keys.QM_GRAD_LOW, Keys.QM_DGRAD]

        for key in qm_int_1d:
            if key in frame:
                arr = np.asarray(frame[key], dtype=np.int32)
                if arr.ndim != 1:
                    continue
                if arr.shape[0] > max_qm:
                    raise ValueError(f"[numpy_folder] key '{key}' length {arr.shape[0]} exceeds max_qm={max_qm}.")
                frame[key] = _pad1d_int32(arr[:n_qm], max_qm)

        for key in qm_float_1d:
            if key in frame:
                arr = np.asarray(frame[key], dtype=np.float32)
                if arr.ndim != 1:
                    continue
                if arr.shape[0] > max_qm:
                    raise ValueError(f"[numpy_folder] key '{key}' length {arr.shape[0]} exceeds max_qm={max_qm}.")
                frame[key] = _pad1d_float(arr[:n_qm], max_qm)

        for key in qm_float_2d:
            if key in frame:
                arr = np.asarray(frame[key], dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != 3:
                    continue
                if arr.shape[0] > max_qm:
                    raise ValueError(f"[numpy_folder] key '{key}' length {arr.shape[0]} exceeds max_qm={max_qm}.")
                frame[key] = _pad2_float3(arr[:n_qm], max_qm)

        if _QM_TYPE_KEY not in frame:
            qm_type = np.zeros((max_qm,), dtype=np.float32)
            qm_type[:min(n_qm, max_qm)] = 1.0
            frame[_QM_TYPE_KEY] = qm_type

        if Keys.ATOM_TYPES in frame:
            arr = np.asarray(frame[Keys.ATOM_TYPES], dtype=np.int64)
            if arr.ndim == 1:
                if arr.shape[0] > max_qm:
                    raise ValueError(f"[numpy_folder] key '{Keys.ATOM_TYPES}' length {arr.shape[0]} exceeds max_qm={max_qm}.")
                frame[Keys.ATOM_TYPES] = _pad1d_int64(arr[:n_qm], max_qm)

    # MM padding
    if n_mm is not None:
        frame[_MAX_MM_KEY] = np.asarray([max_mm], dtype=np.int32)
        mm_float_1d = [Keys.MM_Q, Keys.MM_TYPE, Keys.MM_ESP_HIGH, Keys.MM_ESP_LOW, Keys.MM_DESP]
        mm_int_1d = [Keys.MM_IDX]
        mm_float_2d = [Keys.MM_COORDS, Keys.MM_GRAD_HIGH, Keys.MM_GRAD_LOW, Keys.MM_DGRAD,
                       Keys.MM_ESPGRAD_HIGH, Keys.MM_ESPGRAD_LOW, Keys.MM_DESPGRAD]

        for key in mm_int_1d:
            if key in frame:
                arr = np.asarray(frame[key], dtype=np.int32)
                if arr.ndim != 1:
                    continue
                if arr.shape[0] > max_mm:
                    raise ValueError(f"[numpy_folder] key '{key}' length {arr.shape[0]} exceeds max_mm={max_mm}.")
                frame[key] = _pad1d_int32(arr[:n_mm], max_mm)

        for key in mm_float_1d:
            if key in frame:
                arr = np.asarray(frame[key], dtype=np.float32)
                if arr.ndim != 1:
                    continue
                if arr.shape[0] > max_mm:
                    raise ValueError(f"[numpy_folder] key '{key}' length {arr.shape[0]} exceeds max_mm={max_mm}.")
                frame[key] = _pad1d_float(arr[:n_mm], max_mm)

        for key in mm_float_2d:
            if key in frame:
                arr = np.asarray(frame[key], dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != 3:
                    continue
                if arr.shape[0] > max_mm:
                    raise ValueError(f"[numpy_folder] key '{key}' length {arr.shape[0]} exceeds max_mm={max_mm}.")
                frame[key] = _pad2_float3(arr[:n_mm], max_mm)

        if Keys.MM_TYPE not in frame:
            mm_type = np.zeros((max_mm,), dtype=np.float32)
            mm_type[:min(n_mm, max_mm)] = 1.0
            frame[Keys.MM_TYPE] = mm_type

        if compute_mm_espgrad:
            _derive_mm_espgrad(frame, esp_eps=esp_eps, warn_missing=warn_missing_espgrad)


def read_numpy_folder(
    inputs: Union[PathLike, Sequence[PathLike]],
    *,
    allowed_keys: Optional[Sequence[str]] = None,
    warn_unknown_keys: bool = True,
    n_frames: Optional[int] = None,
    require_same_n: bool = True,
    coerce_dtypes: bool = True,
    validate_shapes: bool = True,
    prefer_npy_over_npz: bool = True,
    warn_duplicate_keys: bool = True,
    max_qm: int = _DEFAULT_MAX_QM,
    max_mm: int = _DEFAULT_MAX_MM,
    apply_padding: bool = True,
    compute_mm_espgrad: bool = True,
    esp_eps: float = 1.0e-12,
    warn_missing_espgrad: bool = True,
    verbose: bool = False,
) -> List[Frame]:
    pool_keys = set(KEY_POOL) - set(NON_TENSOR_KEYS)
    pool_keys.update({_QM_TYPE_KEY, _MAX_QM_KEY, _MAX_MM_KEY})
    keyset = set(allowed_keys) if allowed_keys is not None else pool_keys

    if isinstance(inputs, (str, os.PathLike)):
        folders = _expand_to_dirs(inputs)
    else:
        folders = []
        for x in inputs:
            folders.extend(_expand_to_dirs(x))

    if not folders:
        raise FileNotFoundError("[numpy_folder] no input directories found.")

    all_frames: List[Frame] = []

    for folder in folders:
        npy_files = sorted(glob.glob(os.path.join(folder, "*.npy")))
        npz_files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        files = npy_files + npz_files
        if not files:
            raise FileNotFoundError(f"[numpy_folder] no .npy or .npz files found in: {folder}")

        key_arrays: Dict[str, np.ndarray] = {}
        key_sources: Dict[str, str] = {}

        def _prefer_new(existing_src: str, new_src: str) -> bool:
            if prefer_npy_over_npz:
                if new_src.endswith(".npy") and not existing_src.endswith(".npy"):
                    return True
                if existing_src.endswith(".npy") and not new_src.endswith(".npy"):
                    return False
            return True

        for fp in files:
            try:
                loaded = _load_file(fp)
            except Exception as exc:
                warnings.warn(f"[numpy_folder] could not load {fp}: {exc}", RuntimeWarning, stacklevel=2)
                continue

            for k, arr in loaded.items():
                if k not in keyset:
                    if warn_unknown_keys:
                        warnings.warn(
                            f"[numpy_folder] ignoring key '{k}' from {os.path.basename(fp)} — not in allowed keys.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    continue
                if k in key_arrays:
                    if warn_duplicate_keys:
                        winner = os.path.basename(fp) if _prefer_new(key_sources[k], fp) else os.path.basename(key_sources[k])
                        warnings.warn(
                            f"[numpy_folder] duplicate key '{k}' in folder '{folder}': "
                            f"existing={os.path.basename(key_sources[k])}, new={os.path.basename(fp)} → keeping '{winner}'.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    if not _prefer_new(key_sources[k], fp):
                        continue
                arr = np.asarray(arr)
                if coerce_dtypes:
                    arr = _coerce_dtype(k, arr)
                key_arrays[k] = arr
                key_sources[k] = fp

        if not key_arrays:
            raise FileNotFoundError(f"[numpy_folder] no recognised keys found in: {folder}")

        counts = {k: _infer_n_frames(v) for k, v in key_arrays.items()}
        if n_frames is not None:
            nf = int(n_frames)
        else:
            unique_counts = set(counts.values())
            if require_same_n and len(unique_counts) > 1:
                detail = ", ".join(f"{k}:{counts[k]}" for k in sorted(counts))
                raise ValueError(
                    f"[numpy_folder] mismatched frame counts in '{folder}': {detail}\n"
                    f"Set require_same_n=False to allow, or pass n_frames= to force a count."
                )
            nf = int(max(unique_counts))

        if nf <= 0:
            raise ValueError(f"[numpy_folder] inferred n_frames={nf} for folder '{folder}'")

        frames = _split_into_frames(
            key_arrays,
            n_frames=nf,
            require_same_n=require_same_n,
            validate_shapes=validate_shapes,
        )

        for i, f in enumerate(frames):
            f.setdefault(Keys.SOURCE_FILE, folder)
            f.setdefault(Keys.FRAME_IDX, i)
            if apply_padding:
                _finalise_padding_and_types(
                    f,
                    max_qm=max_qm,
                    max_mm=max_mm,
                    compute_mm_espgrad=compute_mm_espgrad,
                    esp_eps=esp_eps,
                    warn_missing_espgrad=warn_missing_espgrad,
                )

        if verbose:
            loaded_keys = sorted(key_arrays.keys())
            print(f"[numpy_folder] {folder}: {len(loaded_keys)} key(s), {nf} frame(s)")
            if len(loaded_keys) <= 40:
                print(f"  keys: {loaded_keys}")

        all_frames.extend(frames)

    if verbose:
        print(f"[numpy_folder] total frames: {len(all_frames)}")

    return all_frames


register_adapter("numpy_folder", read_numpy_folder)
