"""
Three-stage post-processing for parsed frames (v1.x)

Purpose
-------
Post-processing enriches and prunes the already-parsed per-frame dictionaries
produced by adapters.  The pipeline is intentionally linear and ordered.

Default behaviour
-----------------
Unless explicitly overridden in YAML / parser kwargs, the full pipeline runs:

  Stage 1 — Tukey filtering
  Stage 2 — Energy demeaning
  Stage 3 — Delta derivation

Stages and enforced order
-------------------------
1) Tukey filtering
   Drop whole frames if any requested filter key is outside the Tukey fence.
   Filtering always acts on the current frame values and is applied before any
   dataset-level mean is computed.

2) Energy demeaning
   Compute E_high_dm and / or E_low_dm from the filtered frame set only.
   Demeaning is allowed only if Tukey filtering is enabled in the pipeline.

3) Delta derivation
   Fill available *_high / *_low derived keys after demeaning has run.
   Delta derivation is allowed only if demeaning is enabled in the pipeline.

Notes
-----
- Missing keys are skipped gracefully; they never crash a stage.
- Whole-frame filtering means that once a frame is marked as an outlier by any
  selected key, that frame is removed from the output list entirely.
- This module uses canonical Keys.* strings from a0_structure only.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from mlp_qmmm.a0_structure import Keys

Frame = Dict[str, np.ndarray]

# Default Tukey key list: run everything unless YAML narrows or disables it.
_DEFAULT_TUKEY_KEYS: Tuple[str, ...] = (
    Keys.E_HIGH,
    Keys.E_LOW,
    Keys.QM_GRAD_HIGH,
    Keys.QM_GRAD_LOW,
    Keys.MM_ESP_HIGH,
    Keys.MM_ESP_LOW,
    Keys.MM_ESPGRAD_HIGH,
    Keys.MM_ESPGRAD_LOW,
)

# Default demean key list: build both demeaned raw energies unless overridden.
_DEFAULT_DEMEAN_KEYS: Tuple[str, ...] = (
    Keys.E_HIGH,
    Keys.E_LOW,
)

# High/low -> delta mappings, in intended derivation order.
_DELTA_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    (Keys.E_HIGH, Keys.E_LOW, Keys.DE),
    (Keys.E_HIGH_DM, Keys.E_LOW_DM, Keys.DE_DM),
    (Keys.QM_GRAD_HIGH, Keys.QM_GRAD_LOW, Keys.QM_DGRAD),
    (Keys.MM_GRAD_HIGH, Keys.MM_GRAD_LOW, Keys.MM_DGRAD),
    (Keys.MM_ESP_HIGH, Keys.MM_ESP_LOW, Keys.MM_DESP),
    (Keys.MM_ESPGRAD_HIGH, Keys.MM_ESPGRAD_LOW, Keys.MM_DESPGRAD),
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_scalar_value(a: np.ndarray) -> Optional[float]:
    arr = np.asarray(a)
    if arr.size == 0:
        return None
    return float(arr.reshape(-1)[0])


def _real_qm_mask(frame: Frame, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=bool)

    if getattr(Keys, "QM_TYPE", "qm_type") in frame:
        qmt = np.asarray(frame[getattr(Keys, "QM_TYPE", "qm_type")])
        if qmt.ndim == 1 and qmt.shape[0] >= n:
            return qmt[:n].astype(np.float64, copy=False) > 0.5

    if Keys.QM_Z in frame:
        qz = np.asarray(frame[Keys.QM_Z])
        if qz.ndim == 1 and qz.shape[0] >= n:
            return qz[:n].astype(np.int64, copy=False) != 0

    if Keys.N_QM in frame:
        nq = int(np.asarray(frame[Keys.N_QM]).reshape(-1)[0])
        out = np.zeros((n,), dtype=bool)
        out[: max(0, min(nq, n))] = True
        return out

    return np.ones((n,), dtype=bool)


def _real_mm_mask(frame: Frame, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=bool)

    if Keys.MM_TYPE in frame:
        mmt = np.asarray(frame[Keys.MM_TYPE])
        if mmt.ndim == 1 and mmt.shape[0] >= n:
            return mmt[:n].astype(np.float64, copy=False) > 0.5

    if Keys.N_MM in frame:
        nm = int(np.asarray(frame[Keys.N_MM]).reshape(-1)[0])
        out = np.zeros((n,), dtype=bool)
        out[: max(0, min(nm, n))] = True
        return out

    return np.ones((n,), dtype=bool)


def _reduce_abs(values: np.ndarray, metric: str) -> float:
    if values.size == 0:
        return 0.0
    av = np.abs(values.astype(np.float64, copy=False))
    metric = (metric or "mean").strip().lower()
    if metric == "max":
        return float(np.max(av))
    if metric == "median":
        return float(np.median(av))
    return float(np.mean(av))


def _reduce_vec(values: np.ndarray, metric: str) -> float:
    if values.size == 0:
        return 0.0
    mags = np.linalg.norm(values.astype(np.float64, copy=False), axis=1)
    metric = (metric or "mean").strip().lower()
    if metric == "max":
        return float(np.max(mags))
    if metric == "median":
        return float(np.median(mags))
    return float(np.mean(mags))


def _frame_score_for_key(frame: Frame, key: str, *, reduce_metric: str = "mean") -> Optional[float]:
    if key not in frame:
        return None

    arr = np.asarray(frame[key])
    if arr.size == 0:
        return None

    # Scalar / singleton arrays like E_high, E_low
    if arr.ndim == 0 or (arr.ndim == 1 and arr.shape[0] == 1):
        return _as_scalar_value(arr)

    # 1-D per-site arrays like mm_esp_high/low
    if arr.ndim == 1:
        if key in (Keys.MM_ESP_HIGH, Keys.MM_ESP_LOW, Keys.MM_DESP):
            mask = _real_mm_mask(frame, arr.shape[0])
            return _reduce_abs(arr[mask], reduce_metric) if np.any(mask) else 0.0
        if key in (Keys.QM_Q, getattr(Keys, "QM_TYPE", "qm_type")):
            mask = _real_qm_mask(frame, arr.shape[0])
            return _reduce_abs(arr[mask], reduce_metric) if np.any(mask) else 0.0
        if key in (Keys.MM_Q, Keys.MM_TYPE):
            mask = _real_mm_mask(frame, arr.shape[0])
            return _reduce_abs(arr[mask], reduce_metric) if np.any(mask) else 0.0
        return _reduce_abs(arr, reduce_metric)

    # 2-D spatial arrays
    if arr.ndim == 2 and arr.shape[1] == 3:
        if key in (
            Keys.QM_GRAD,
            Keys.QM_GRAD_HIGH,
            Keys.QM_GRAD_LOW,
            Keys.QM_DGRAD,
            Keys.QM_COORDS,
        ):
            mask = _real_qm_mask(frame, arr.shape[0])
            return _reduce_vec(arr[mask], reduce_metric) if np.any(mask) else 0.0
        if key in (
            Keys.MM_COORDS,
            Keys.MM_GRAD_HIGH,
            Keys.MM_GRAD_LOW,
            Keys.MM_DGRAD,
            Keys.MM_ESPGRAD_HIGH,
            Keys.MM_ESPGRAD_LOW,
            Keys.MM_DESPGRAD,
        ):
            mask = _real_mm_mask(frame, arr.shape[0])
            return _reduce_vec(arr[mask], reduce_metric) if np.any(mask) else 0.0
        return _reduce_vec(arr, reduce_metric)

    return None


def _tukey_keep(values: np.ndarray, fence_k: float) -> np.ndarray:
    v = values.astype(np.float64, copy=False)
    q1 = float(np.nanpercentile(v, 25.0))
    q3 = float(np.nanpercentile(v, 75.0))
    iqr = max(q3 - q1, 1.0e-12)
    lo = q1 - fence_k * iqr
    hi = q3 + fence_k * iqr
    return (v >= lo) & (v <= hi)


def _dm_key_for(raw_key: str) -> Optional[str]:
    if raw_key == Keys.E_HIGH:
        return Keys.E_HIGH_DM
    if raw_key == Keys.E_LOW:
        return Keys.E_LOW_DM
    return None


def _normalise_key_list(keys: Optional[Sequence[str]], default_keys: Sequence[str]) -> List[str]:
    if keys is None:
        return list(default_keys)
    out: List[str] = []
    seen = set()
    for k in keys:
        if k in seen:
            continue
        out.append(k)
        seen.add(k)
    return out


# ---------------------------------------------------------------------------
# Stage 1 — Tukey filtering
# ---------------------------------------------------------------------------

def filter_tukey(
    frames: List[Frame],
    *,
    keys: Optional[Sequence[str]] = None,
    fence_k: float = 3.5,
    reduce_metric: str = "mean",
    verbose: bool = False,
) -> Tuple[List[Frame], Dict[str, Any]]:
    """
    Remove whole-frame outliers using Tukey fences on the requested keys.

    Each key contributes at most one scalar score per frame:
      - scalar arrays use their scalar value
      - 1-D site arrays use mean/max/median absolute magnitude over real rows
      - 2-D (N, 3) arrays use mean/max/median L2 norm over real rows

    A frame is dropped if ANY requested key marks it as an outlier.
    Missing keys are skipped and reported in the summary notes.
    """
    n_total = len(frames)
    if n_total == 0:
        return frames, {"total": 0, "kept": 0, "dropped": 0, "keys": [], "notes": "no frames"}

    keys_use = _normalise_key_list(keys, _DEFAULT_TUKEY_KEYS)
    keep = np.ones((n_total,), dtype=bool)
    skipped: List[str] = []
    scored: List[str] = []

    for key in keys_use:
        scores = np.full((n_total,), np.nan, dtype=np.float64)
        any_scored = False
        for i, f in enumerate(frames):
            try:
                val = _frame_score_for_key(f, key, reduce_metric=reduce_metric)
            except Exception:
                val = None
            if val is None or not np.isfinite(val):
                continue
            scores[i] = float(val)
            any_scored = True

        if not any_scored:
            skipped.append(key)
            continue

        finite = np.isfinite(scores)
        if not np.any(finite):
            skipped.append(key)
            continue

        keep_key = np.ones((n_total,), dtype=bool)
        keep_key[finite] = _tukey_keep(scores[finite], fence_k)
        keep &= keep_key
        scored.append(key)

    kept = [f for f, ok in zip(frames, keep) if ok]

    notes: List[str] = []
    if skipped:
        notes.append("skipped missing/unscorable keys: " + ", ".join(skipped))

    # Safety guard: never silently return an empty dataset.
    if len(kept) == 0:
        notes.append("all frames would be dropped → Tukey filter disabled for this dataset")
        kept = frames

    info: Dict[str, Any] = {
        "total": n_total,
        "kept": len(kept),
        "dropped": int(n_total - len(kept)),
        "keys_requested": list(keys_use),
        "keys_scored": scored,
        "fence_k": float(fence_k),
        "reduce_metric": reduce_metric,
        "notes": "; ".join(notes) if notes else "",
    }

    if verbose:
        msg = f"[postprocess.filter_tukey] kept {info['kept']}/{info['total']} (dropped {info['dropped']})"
        if info["notes"]:
            msg += f" | {info['notes']}"
        print(msg)

    return kept, info


# ---------------------------------------------------------------------------
# Stage 2 — Energy demeaning
# ---------------------------------------------------------------------------

def energy_demean(
    frames: List[Frame],
    *,
    keys: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compute demeaned energy arrays from the filtered frame set.

    Only Keys.E_HIGH and Keys.E_LOW are supported inputs.  The stage writes
    Keys.E_HIGH_DM / Keys.E_LOW_DM only for the requested raw keys that are
    actually present in the frame list.
    """
    keys_use = _normalise_key_list(keys, _DEFAULT_DEMEAN_KEYS)
    supported = {Keys.E_HIGH, Keys.E_LOW}
    bad = [k for k in keys_use if k not in supported]
    if bad:
        raise ValueError(
            "energy_demean only supports raw energy keys E_high and E_low; "
            f"got unsupported keys: {bad}"
        )

    stats: Dict[str, Any] = {
        "keys_requested": list(keys_use),
        "keys_written": [],
        "notes": "",
    }
    notes: List[str] = []

    for raw_key in keys_use:
        dm_key = _dm_key_for(raw_key)
        if dm_key is None:
            continue

        vals: List[np.ndarray] = []
        for f in frames:
            if raw_key not in f:
                continue
            arr = np.asarray(f[raw_key], dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            vals.append(arr)

        if not vals:
            notes.append(f"missing {raw_key} → skipped")
            continue

        concat = np.concatenate(vals).astype(np.float64, copy=False)
        mu = float(np.mean(concat))
        sd = float(np.std(concat, ddof=0))

        for f in frames:
            if raw_key not in f:
                continue
            arr = np.asarray(f[raw_key], dtype=np.float32)
            f[dm_key] = (arr - np.float32(mu)).astype(np.float32, copy=False)

        stats[f"{raw_key}_mean"] = mu
        stats[f"{raw_key}_std"] = sd
        stats["keys_written"].append(dm_key)

    stats["notes"] = "; ".join(notes) if notes else ""

    if verbose:
        msg = f"[postprocess.energy_demean] wrote {stats['keys_written']}"
        if stats["notes"]:
            msg += f" | {stats['notes']}"
        print(msg)

    return stats


# ---------------------------------------------------------------------------
# Stage 3 — Delta derivation
# ---------------------------------------------------------------------------

def derive_deltas(
    frames: List[Frame],
    *,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Derive delta keys from available *_high / *_low pairs.

    Pairs are attempted in the canonical order:
      E_high/E_low               -> dE
      E_high_dm/E_low_dm         -> dE_dm
      qm_grad_high/qm_grad_low   -> qm_dgrad
      mm_grad_high/mm_grad_low   -> mm_dgrad
      mm_esp_high/mm_esp_low     -> mm_desp
      mm_espgrad_high/low        -> mm_despgrad

    Missing source pairs are skipped gracefully.
    Existing destination keys are overwritten so the stage always reflects the
    current filtered+demeaned frame set.
    """
    written: List[str] = []
    skipped: List[str] = []

    for high_key, low_key, out_key in _DELTA_PAIRS:
        any_written = False
        for f in frames:
            if high_key not in f or low_key not in f:
                continue
            hi = np.asarray(f[high_key], dtype=np.float32)
            lo = np.asarray(f[low_key], dtype=np.float32)
            if hi.shape != lo.shape:
                continue
            f[out_key] = (hi - lo).astype(np.float32, copy=False)
            any_written = True
        if any_written:
            written.append(out_key)
        else:
            skipped.append(out_key)

    info = {
        "written": written,
        "skipped": skipped,
    }

    if verbose:
        msg = f"[postprocess.derive_deltas] wrote {written}"
        if skipped:
            msg += f" | skipped {skipped}"
        print(msg)

    return info


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_postprocess_pipeline(
    frames: List[Frame],
    *,
    # Stage 1 — Tukey filtering
    apply_tukey: bool = True,
    tukey_keys: Optional[Sequence[str]] = None,
    tukey_fence_k: float = 3.5,
    tukey_reduce_metric: str = "mean",
    # Stage 2 — demean
    apply_demean: bool = True,
    demean_keys: Optional[Sequence[str]] = None,
    # Stage 3 — deltas
    derive_delta_keys: bool = True,
    # Misc
    verbose: bool = False,
) -> Tuple[List[Frame], Dict[str, Any]]:
    """
    Run the strict three-stage postprocess pipeline.

    Default policy: do everything unless YAML explicitly overrides it.

    Enforced conditions
    -------------------
    - apply_demean=True requires apply_tukey=True
    - derive_delta_keys=True requires apply_demean=True
    """
    if not frames:
        return frames, {
            "total": 0,
            "kept": 0,
            "dropped": 0,
            "tukey": {},
            "demean": {},
            "deltas": {},
        }

    if apply_demean and not apply_tukey:
        raise ValueError(
            "Postprocess contract violation: demeaning requires Tukey filtering to be enabled first. "
            "Set apply_tukey=True or disable apply_demean explicitly."
        )
    if derive_delta_keys and not apply_demean:
        raise ValueError(
            "Postprocess contract violation: delta derivation requires demeaning to be enabled first. "
            "Set apply_demean=True or disable derive_delta_keys explicitly."
        )

    kept = frames
    tukey_info: Dict[str, Any] = {}
    demean_info: Dict[str, Any] = {}
    delta_info: Dict[str, Any] = {}

    # Stage 1
    if apply_tukey:
        kept, tukey_info = filter_tukey(
            frames,
            keys=tukey_keys,
            fence_k=tukey_fence_k,
            reduce_metric=tukey_reduce_metric,
            verbose=verbose,
        )
    else:
        tukey_info = {
            "total": len(frames),
            "kept": len(frames),
            "dropped": 0,
            "keys_requested": [],
            "keys_scored": [],
            "fence_k": float(tukey_fence_k),
            "reduce_metric": tukey_reduce_metric,
            "notes": "Tukey filtering disabled",
        }

    # Stage 2
    if apply_demean:
        demean_info = energy_demean(
            kept,
            keys=demean_keys,
            verbose=verbose,
        )
    else:
        demean_info = {
            "keys_requested": [],
            "keys_written": [],
            "notes": "demeaning disabled",
        }

    # Stage 3
    if derive_delta_keys:
        delta_info = derive_deltas(kept, verbose=verbose)
    else:
        delta_info = {
            "written": [],
            "skipped": [],
        }

    return kept, {
        "total": len(frames),
        "kept": len(kept),
        "dropped": int(len(frames) - len(kept)),
        "tukey": tukey_info,
        "demean": demean_info,
        "deltas": delta_info,
    }


__all__ = [
    "filter_tukey",
    "energy_demean",
    "derive_deltas",
    "run_postprocess_pipeline",
]
