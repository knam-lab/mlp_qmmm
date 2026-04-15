"""
High-level dataset parser for mlp_qmmm (v1.x)

Responsibilities
----------------
1. Select a registered adapter and parse raw frames from disk.
2. Run the post-processing pipeline to fill derived keys.
3. Warn / report which canonical keys are available after parsing and after
   postprocess, before model construction begins.
4. Optionally normalise final frame padding for consistency.
5. Optionally dump the final frames as NumPy arrays for round-trip loading via
   the numpy_folder adapter.
6. Write a parse summary (JSON + TXT) with adapter, counts, postprocess stats,
   key availability, padding, energy stats, and timing.

Design notes
------------
- This module is adapter-agnostic. Adapters self-register via register_adapter().
- The module does not perform any unit conversion itself; adapters are expected
  to return canonical units as defined in a0_structure.
- Final padding here is a safety / consistency pass. In many cases the adapter
  already emits fixed-width arrays, so this step becomes a no-op.
"""

from __future__ import annotations

import importlib
import json
import os
import time
import warnings
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from mlp_qmmm.a0_structure import Keys, KEYS_POOL, KEY_POOL, NON_TENSOR_KEYS
from mlp_qmmm.a1_postprocess import run_postprocess_pipeline

try:
    from mlp_qmmm.a0_structure import Defaults  # type: ignore
except Exception:  # pragma: no cover
    class Defaults:
        MAX_QM = 100
        MAX_MM = 5000

Frame = Dict[str, Any]
Frames = List[Frame]
_Adapter = Callable[..., Union[Frame, Frames, Iterable[Frame]]]

_ALLOW_PREFIXES = ("feat_", "dbg_", "tmp_", "mode_", "aux_")
QM_TYPE_KEY = getattr(Keys, "QM_TYPE", "qm_type")
MAX_QM_KEY = getattr(Keys, "MAX_QM", "max_qm")
MAX_MM_KEY = getattr(Keys, "MAX_MM", "max_mm")


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

_ADAPTERS: Dict[str, _Adapter] = {}


def register_adapter(name: str, adapter: _Adapter) -> None:
    if not name or not callable(adapter):
        raise ValueError("register_adapter requires a non-empty name and a callable.")
    _ADAPTERS[name] = adapter


def _lazy_import(name: str) -> None:
    tried: List[str] = []

    try:
        importlib.import_module(name)
        return
    except Exception as e:
        tried.append(f"{name} ({type(e).__name__}: {e})")

    if "." not in name:
        qualified = f"mlp_qmmm.a_input_types.{name}"
        try:
            importlib.import_module(qualified)
            return
        except Exception as e:
            tried.append(f"{qualified} ({type(e).__name__}: {e})")

    raise ImportError("Could not import adapter module. Tried:\n  - " + "\n  - ".join(tried))


def get_adapter(name: str) -> _Adapter:
    if name in _ADAPTERS:
        return _ADAPTERS[name]

    try:
        _lazy_import(name)
    except ImportError:
        if "." in name:
            try:
                _lazy_import(name.rsplit(".", 1)[-1])
            except Exception:
                pass

    if name in _ADAPTERS:
        return _ADAPTERS[name]
    if "." in name:
        base = name.rsplit(".", 1)[-1]
        if base in _ADAPTERS:
            return _ADAPTERS[base]

    raise KeyError(
        f"Unknown adapter '{name}'. Known adapters: {sorted(_ADAPTERS.keys())}. "
        "Make sure the adapter module has been imported and called register_adapter()."
    )


def list_adapters() -> List[str]:
    return sorted(_ADAPTERS.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _atomic_write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _coerce_raw_frames(raw: Union[Frame, Frames, Iterable[Frame]]) -> Frames:
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return raw
    return list(raw)


def _collect_key_presence(frames: Frames) -> Dict[str, List[str]]:
    if not frames:
        return {"all": [], "in_all": [], "partial": []}
    all_keys = sorted(set().union(*[f.keys() for f in frames]))
    in_all = sorted(k for k in all_keys if all(k in f for f in frames))
    partial = sorted(k for k in all_keys if k not in in_all)
    return {"all": all_keys, "in_all": in_all, "partial": partial}


def _report_frame_keys(
    frames: Frames,
    *,
    label: str,
    prev_keys: Optional[Sequence[str]] = None,
) -> List[str]:
    info = _collect_key_presence(frames)
    all_keys = info["all"]
    in_all = info["in_all"]
    partial = info["partial"]

    if not frames:
        print(f"[keys:{label}] no frames")
        return []

    new_keys: List[str] = []
    if prev_keys is not None:
        prev = set(prev_keys)
        new_keys = sorted(k for k in all_keys if k not in prev)

    print(f"[keys:{label}] {len(frames)} frame(s) — {len(all_keys)} unique key(s)")
    if new_keys:
        print(f"  added by this stage ({len(new_keys)}): {new_keys}")
    print(f"  present in ALL frames ({len(in_all)}): {in_all}")
    if partial:
        print(f"  present in SOME frames only ({len(partial)}): {partial}")

    unexpected = KEYS_POOL.unexpected_keys({k: None for k in all_keys}, allow_prefixes=_ALLOW_PREFIXES)
    if unexpected:
        print(f"  [WARN] not in KEY_POOL ({len(unexpected)}): {unexpected}")

    return all_keys


def _warn_available_keys(frames: Frames, *, label: str) -> None:
    info = _collect_key_presence(frames)
    if not info["all"]:
        warnings.warn(f"[parse:{label}] no frames / no keys available")
        return
    msg = (
        f"[parse:{label}] available keys before model stage: "
        f"ALL={info['in_all']}"
    )
    if info["partial"]:
        msg += f" | SOME={info['partial']}"
    unexpected = KEYS_POOL.unexpected_keys({k: None for k in info["all"]}, allow_prefixes=_ALLOW_PREFIXES)
    if unexpected:
        msg += f" | not-in-KEY_POOL={unexpected}"
    warnings.warn(msg)


# ---------------------------------------------------------------------------
# Final padding / consistency helpers
# ---------------------------------------------------------------------------


def _pad_1d(a: np.ndarray, L: int, *, fill: Union[int, float], dtype: np.dtype) -> np.ndarray:
    out = np.full((L,), fill, dtype=dtype)
    n = min(int(L), int(a.shape[0]))
    if n > 0:
        out[:n] = a[:n].astype(dtype, copy=False)
    return out


def _pad_2d_last3(a: np.ndarray, L: int, *, fill: float = 0.0, dtype: np.dtype = np.float32) -> np.ndarray:
    out = np.full((L, 3), fill, dtype=dtype)
    n = min(int(L), int(a.shape[0]))
    if n > 0:
        out[:n, :] = a[:n].astype(dtype, copy=False)
    return out


def _scalar_int_from_frame(frame: Frame, key: str) -> Optional[int]:
    if key not in frame:
        return None
    arr = np.asarray(frame[key]).reshape(-1)
    if arr.size == 0:
        return None
    return int(arr[0])


def _infer_existing_pad(frames: Frames, key: str) -> Optional[int]:
    vals = {_scalar_int_from_frame(f, key) for f in frames if _scalar_int_from_frame(f, key) is not None}
    vals.discard(None)
    if len(vals) == 1:
        return int(next(iter(vals)))
    return None


def _count_real_qm(frame: Frame) -> int:
    if QM_TYPE_KEY in frame:
        a = np.asarray(frame[QM_TYPE_KEY]).reshape(-1)
        return int(np.sum(a > 0.5))
    if Keys.QM_Z in frame:
        a = np.asarray(frame[Keys.QM_Z]).reshape(-1)
        return int(np.sum(a != 0))
    n_qm = _scalar_int_from_frame(frame, Keys.N_QM)
    return int(n_qm or 0)


def _count_real_mm(frame: Frame) -> int:
    if Keys.MM_TYPE in frame:
        a = np.asarray(frame[Keys.MM_TYPE]).reshape(-1)
        return int(np.sum(a > 0.5))
    n_mm = _scalar_int_from_frame(frame, Keys.N_MM)
    return int(n_mm or 0)


def _stored_qm_width(frame: Frame) -> int:
    candidates = (
        Keys.QM_Z, QM_TYPE_KEY, Keys.ATOM_TYPES,
        Keys.QM_COORDS, Keys.QM_GRAD, Keys.QM_GRAD_HIGH,
        Keys.QM_GRAD_LOW, Keys.QM_DGRAD,
    )
    w = 0
    for k in candidates:
        if k in frame:
            w = max(w, int(np.asarray(frame[k]).shape[0]))
    return w


def _stored_mm_width(frame: Frame) -> int:
    candidates = (
        Keys.MM_TYPE, Keys.MM_Q, Keys.MM_COORDS,
        Keys.MM_GRAD_HIGH, Keys.MM_GRAD_LOW, Keys.MM_DGRAD,
        Keys.MM_ESP_HIGH, Keys.MM_ESP_LOW, Keys.MM_DESP,
        Keys.MM_ESPGRAD_HIGH, Keys.MM_ESPGRAD_LOW, Keys.MM_DESPGRAD,
    )
    w = 0
    for k in candidates:
        if k in frame:
            w = max(w, int(np.asarray(frame[k]).shape[0]))
    return w


def _pad_final_frames(
    frames: Frames,
    *,
    qm_pad_to: Optional[int],
    mm_pad_to: Optional[int],
) -> Dict[str, Any]:
    if not frames:
        return {
            "qm_pad_to": 0,
            "mm_pad_to": 0,
            "observed_max_qm_real": 0,
            "observed_max_mm_real": 0,
            "observed_max_qm_width": 0,
            "observed_max_mm_width": 0,
            "qm_frames_padded_by_parser": 0,
            "mm_frames_padded_by_parser": 0,
            "qm_type_filled_by_parser": 0,
            "mm_type_filled_by_parser": 0,
        }

    qm_1d_i32 = {Keys.QM_Z, Keys.QM_IDX}
    qm_1d_f32 = {Keys.QM_Q, QM_TYPE_KEY}
    qm_1d_i64 = {Keys.ATOM_TYPES}
    qm_2d_f32 = {Keys.QM_COORDS, Keys.QM_GRAD, Keys.QM_GRAD_HIGH, Keys.QM_GRAD_LOW, Keys.QM_DGRAD}

    mm_1d_i32 = {Keys.MM_IDX}
    mm_1d_f32 = {Keys.MM_TYPE, Keys.MM_Q, Keys.MM_ESP_HIGH, Keys.MM_ESP_LOW, Keys.MM_DESP}
    mm_2d_f32 = {
        Keys.MM_COORDS, Keys.MM_GRAD_HIGH, Keys.MM_GRAD_LOW, Keys.MM_DGRAD,
        Keys.MM_ESPGRAD_HIGH, Keys.MM_ESPGRAD_LOW, Keys.MM_DESPGRAD,
    }

    obs_qm_real = max(_count_real_qm(f) for f in frames)
    obs_mm_real = max(_count_real_mm(f) for f in frames)
    obs_qm_width = max(_stored_qm_width(f) for f in frames)
    obs_mm_width = max(_stored_mm_width(f) for f in frames)

    existing_q = _infer_existing_pad(frames, MAX_QM_KEY)
    existing_m = _infer_existing_pad(frames, MAX_MM_KEY)

    qL = int(qm_pad_to) if qm_pad_to is not None else int(
        existing_q if existing_q is not None else max(obs_qm_width, Defaults.MAX_QM)
    )
    mL = int(mm_pad_to) if mm_pad_to is not None else int(
        existing_m if existing_m is not None else max(obs_mm_width, Defaults.MAX_MM)
    )

    if qL <= 0 or mL <= 0:
        raise ValueError(f"Final pad targets must be > 0, got qm_pad_to={qL}, mm_pad_to={mL}.")
    if qL < obs_qm_width:
        raise ValueError(f"qm_pad_to={qL} is smaller than stored QM width {obs_qm_width}.")
    if mL < obs_mm_width:
        raise ValueError(f"mm_pad_to={mL} is smaller than stored MM width {obs_mm_width}.")

    qpad = 0
    mpad = 0
    qtype_filled = 0
    mtype_filled = 0

    for f in frames:
        n_qm_real = _count_real_qm(f)
        n_mm_real = _count_real_mm(f)

        padded_q = False
        padded_m = False

        if QM_TYPE_KEY not in f:
            arr = np.zeros((qL,), dtype=np.float32)
            arr[:min(max(n_qm_real, 0), qL)] = 1.0
            f[QM_TYPE_KEY] = arr
            padded_q = True
            qtype_filled += 1

        if Keys.MM_TYPE not in f:
            arr = np.zeros((mL,), dtype=np.float32)
            arr[:min(max(n_mm_real, 0), mL)] = 1.0
            f[Keys.MM_TYPE] = arr
            padded_m = True
            mtype_filled += 1

        for k in list(f.keys()):
            a = np.asarray(f[k])
            if a.ndim == 0:
                continue

            if k in qm_1d_i32 and a.shape[0] != qL:
                f[k] = _pad_1d(a, qL, fill=0, dtype=np.int32)
                padded_q = True
            elif k in qm_1d_f32 and a.shape[0] != qL:
                fill = 0.0
                f[k] = _pad_1d(a, qL, fill=fill, dtype=np.float32)
                padded_q = True
            elif k in qm_1d_i64 and a.shape[0] != qL:
                f[k] = _pad_1d(a, qL, fill=0, dtype=np.int64)
                padded_q = True
            elif k in qm_2d_f32 and a.shape[0] != qL:
                f[k] = _pad_2d_last3(a, qL, fill=0.0, dtype=np.float32)
                padded_q = True

            elif k in mm_1d_i32 and a.shape[0] != mL:
                f[k] = _pad_1d(a, mL, fill=0, dtype=np.int32)
                padded_m = True
            elif k in mm_1d_f32 and a.shape[0] != mL:
                f[k] = _pad_1d(a, mL, fill=0.0, dtype=np.float32)
                padded_m = True
            elif k in mm_2d_f32 and a.shape[0] != mL:
                f[k] = _pad_2d_last3(a, mL, fill=0.0, dtype=np.float32)
                padded_m = True

        # Enforce mask consistency with real counts if arrays were shorter and got padded
        if QM_TYPE_KEY in f:
            qt = np.asarray(f[QM_TYPE_KEY], dtype=np.float32).reshape(-1)
            qt[min(n_qm_real, qL):] = 0.0
            f[QM_TYPE_KEY] = qt
        if Keys.MM_TYPE in f:
            mt = np.asarray(f[Keys.MM_TYPE], dtype=np.float32).reshape(-1)
            mt[min(n_mm_real, mL):] = 0.0
            f[Keys.MM_TYPE] = mt

        f[MAX_QM_KEY] = np.asarray([qL], dtype=np.int32)
        f[MAX_MM_KEY] = np.asarray([mL], dtype=np.int32)

        qpad += int(padded_q)
        mpad += int(padded_m)

    return {
        "qm_pad_to": qL,
        "mm_pad_to": mL,
        "observed_max_qm_real": obs_qm_real,
        "observed_max_mm_real": obs_mm_real,
        "observed_max_qm_width": obs_qm_width,
        "observed_max_mm_width": obs_mm_width,
        "qm_frames_padded_by_parser (optional)": qpad,
        "mm_frames_padded_by_parser (optional)": mpad,
        "qm_type_filled_by_parser (optional)": qtype_filled,
        "mm_type_filled_by_parser (optional)": mtype_filled,
    }


def _check_required_keys_all_frames(
    frames: Frames,
    required_keys: Sequence[str],
    *,
    warn_only: bool,
) -> Dict[str, Dict[str, int]]:
    info: Dict[str, Dict[str, int]] = {}
    n = len(frames)
    if n == 0:
        return info

    problems = []
    for k in required_keys:
        present = sum(1 for f in frames if k in f)
        info[str(k)] = {"present": int(present), "total": int(n)}
        if present != n:
            problems.append((k, present, n))

    if problems:
        msg = "Required final-frame keys missing after parsing/postprocess:\n" + "\n".join(
            f"  - {k}: present in {p}/{n} frames" for k, p, n in problems
        )
        if warn_only:
            warnings.warn(msg)
        else:
            raise KeyError(msg)

    return info


# ---------------------------------------------------------------------------
# NumPy dump
# ---------------------------------------------------------------------------


def dump_frames_to_numpy(
    frames: Frames,
    out_dir: str,
    *,
    fmt: str = "npy",
    overwrite: bool = False,
    verbose: bool = False,
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    pool_set = set(KEY_POOL) - set(NON_TENSOR_KEYS)
    candidate_keys = sorted(k for k in pool_set if any(k in f for f in frames))

    all_frame_keys = set().union(*[f.keys() for f in frames]) if frames else set()
    unexpected = KEYS_POOL.unexpected_keys({k: None for k in all_frame_keys}, allow_prefixes=_ALLOW_PREFIXES)
    if unexpected and verbose:
        warnings.warn(
            f"[dump] frame keys not in KEY_POOL and not dumped: {unexpected}"
        )

    dumped: Dict[str, np.ndarray] = {}
    skipped: List[str] = []

    for k in candidate_keys:
        if not all(k in f for f in frames):
            skipped.append(k)
            continue
        try:
            dumped[k] = np.stack([np.asarray(f[k]) for f in frames], axis=0)
        except Exception as exc:
            skipped.append(k)
            if verbose:
                warnings.warn(f"[dump] skip '{k}': could not stack — {exc}")

    if not dumped:
        raise ValueError("No dumpable canonical keys found.")

    if fmt == "npy":
        for k, arr in dumped.items():
            fp = os.path.join(out_dir, f"{k}.npy")
            if os.path.exists(fp) and not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing file: {fp}")
            np.save(fp, arr)
        if verbose:
            print(f"[dump] wrote {len(dumped)} .npy files → {out_dir}")
    elif fmt == "npz":
        fp = os.path.join(out_dir, "data.npz")
        if os.path.exists(fp) and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {fp}")
        np.savez(fp, **dumped)
        if verbose:
            print(f"[dump] wrote data.npz ({len(dumped)} arrays) → {out_dir}")
    else:
        raise ValueError(f"dump fmt must be 'npy' or 'npz', got '{fmt}'")

    manifest = {
        "created_at": time.time(),
        "n_frames": len(frames),
        "format": fmt,
        "keys_dumped": sorted(dumped.keys()),
        "keys_skipped": sorted(skipped),
    }
    _atomic_write(os.path.join(out_dir, "manifest.json"), json.dumps(manifest, indent=2, sort_keys=True))
    return os.path.abspath(out_dir)


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

_SCI = "{:.6e}"


def _collect_energy_stats(frames: Frames) -> Dict[str, Dict[str, float]]:
    energy_keys = (
        Keys.E_HIGH,
        Keys.E_LOW,
        Keys.E_HIGH_DM,
        Keys.E_LOW_DM,
        Keys.DE,
        Keys.DE_DM,
        Keys.QM_ENERGY,
    )
    stats: Dict[str, Dict[str, float]] = {}
    for k in energy_keys:
        vals: List[float] = []
        for f in frames:
            if k in f:
                arr = np.asarray(f[k]).reshape(-1)
                if arr.size > 0:
                    vals.append(float(arr[0]))
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        stats[k] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr, ddof=0)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
        }
    return stats


def write_parse_summary(
    frames_raw: Frames,
    frames_final: Frames,
    *,
    adapter: str,
    path: Any,
    pp_summary: Dict[str, Any],
    elapsed_s: float,
    out_dir: str,
    dump_dir: Optional[str],
    summary_json_path: Optional[str] = None,
    summary_txt_path: Optional[str] = None,
    verbose: bool = False,
    required_key_check: Optional[Dict[str, Dict[str, int]]] = None,
    padding_info: Optional[Dict[str, Any]] = None,
    available_keys: Optional[Dict[str, List[str]]] = None,
) -> Tuple[str, str]:
    json_path = summary_json_path or os.path.join(out_dir, "parse_summary.json")
    txt_path = summary_txt_path or os.path.join(out_dir, "parse_summary.txt")

    energy_stats = _collect_energy_stats(frames_final)
    ts = time.time()

    payload: Dict[str, Any] = {
        "adapter": adapter,
        "path": str(path),
        "timestamp": ts,
        "elapsed_s": elapsed_s,
        "frame_counts": {"raw": len(frames_raw), "final": len(frames_final)},
        "out_dir": out_dir,
        "dump_dir": dump_dir,
        "postprocess": pp_summary,
        "energy_stats": energy_stats,
        "required_key_check": required_key_check or {},
        "padding": padding_info or {},
        "available_keys": available_keys or {},
    }
    _atomic_write(json_path, json.dumps(payload, indent=2, sort_keys=True))

    tuk = pp_summary.get("tukey", {})
    dem = pp_summary.get("demean", {})
    dlt = pp_summary.get("deltas", {})

    lines: List[str] = [
        "=== mlp_qmmm parse_dataset summary ===",
        f"adapter      : {adapter}",
        f"path         : {path}",
        f"timestamp    : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}",
        f"elapsed      : {elapsed_s:.3f} s",
        f"frames       : raw={len(frames_raw)}  final={len(frames_final)}",
        f"out_dir      : {out_dir}",
        f"dump_dir     : {dump_dir or '(not saved)'}",
        "",
        "postprocess:",
    ]

    if tuk:
        lines.append(
            f"  tukey      : kept={tuk.get('kept')}  total={tuk.get('total')}  dropped={tuk.get('dropped')}"
        )
        if tuk.get("keys_scored"):
            lines.append(f"               keys_scored={tuk.get('keys_scored')}")
        if tuk.get("notes"):
            lines.append(f"               {tuk.get('notes')}")
    else:
        lines.append("  tukey      : (not run)")

    if dem:
        lines.append(f"  demean     : keys_written={dem.get('keys_written', [])}")
        for rk in dem.get("keys_requested", []):
            mu = dem.get(f"{rk}_mean")
            sd = dem.get(f"{rk}_std")
            if mu is not None and sd is not None:
                lines.append(f"               {rk} mean={_SCI.format(mu)}  std={_SCI.format(sd)}")
        if dem.get("notes"):
            lines.append(f"               {dem.get('notes')}")
    else:
        lines.append("  demean     : (not run)")

    if dlt:
        lines.append(f"  deltas     : written={dlt.get('written', [])}")
        if dlt.get("skipped"):
            lines.append(f"               skipped={dlt.get('skipped')}")
    else:
        lines.append("  deltas     : (not run)")

    if available_keys:
        lines.append("")
        lines.append("available keys on final frames:")
        lines.append(f"  in ALL frames : {available_keys.get('in_all', [])}")
        if available_keys.get("partial"):
            lines.append(f"  in SOME only  : {available_keys.get('partial', [])}")

    if required_key_check:
        lines.append("")
        lines.append("required key presence on final frames:")
        for k, d in sorted(required_key_check.items()):
            lines.append(f"  {k}: {d.get('present', 0)}/{d.get('total', 0)}")

    if padding_info:
        lines.append("")
        lines.append("padding:")
        for k, v in padding_info.items():
            lines.append(f"  {k}: {v}")

    if energy_stats:
        lines.append("")
        lines.append("energy stats on final frames (mean / std / min / max):")
        for k, d in energy_stats.items():
            lines.append(
                f"  {k:15s}: mean={_SCI.format(d['mean'])}  std={_SCI.format(d['std'])}  "
                f"min={_SCI.format(d['min'])}  max={_SCI.format(d['max'])}"
            )
    lines.append("")

    _atomic_write(txt_path, "\n".join(lines))
    if verbose:
        print(f"[parse] summary JSON → {json_path}")
        print(f"[parse] summary TXT  → {txt_path}")
    return json_path, txt_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_dataset(
    *,
    adapter: str,
    path: Any,
    adapter_kwargs: Optional[Mapping[str, Any]] = None,
    postprocess_kwargs: Optional[Mapping[str, Any]] = None,
    out_dir: Optional[str] = None,
    summary_json: Optional[str] = None,
    summary_txt: Optional[str] = None,
    dump_arrays: bool = False,
    dump_dir: Optional[str] = None,
    dump_fmt: str = "npy",
    dump_overwrite: bool = False,
    required_keys_all_frames: Optional[Sequence[str]] = None,
    required_keys_warn_only: bool = False,
    qm_pad_to: Optional[int] = None,
    mm_pad_to: Optional[int] = None,
    warn_available_keys: bool = True,
    verbose: bool = False,
) -> Frames:
    """
    Parse a dataset end-to-end and return the final frame list.

    The parser is adapter-agnostic: it selects the adapter, obtains the raw
    keys, calls postprocess to fill derived keys, optionally warns about final
    key availability before model construction, and can dump a NumPy round-trip
    representation.
    """
    t0 = time.time()

    adapter_kwargs = dict(adapter_kwargs or {})
    postprocess_kwargs = dict(postprocess_kwargs or {})

    resolved_out_dir = os.path.abspath(out_dir or os.getcwd())
    os.makedirs(resolved_out_dir, exist_ok=True)

    if verbose:
        print(f"[parse] adapter='{adapter}'  path={path}")

    fn = get_adapter(adapter)
    frames_raw = _coerce_raw_frames(fn(path, **adapter_kwargs))
    if not frames_raw:
        raise ValueError(f"Adapter '{adapter}' returned no frames for path: {path}")

    if verbose:
        print(f"[parse] adapter returned {len(frames_raw)} raw frame(s)")

    unexpected = KEYS_POOL.unexpected_keys(frames_raw[0], allow_prefixes=_ALLOW_PREFIXES)
    if unexpected:
        warnings.warn(
            f"[parse] adapter '{adapter}' produced keys not in KEY_POOL: {unexpected}. "
            "They will be carried through but not dumped."
        )

    keys_after_parse: List[str] = []
    if verbose:
        keys_after_parse = _report_frame_keys(frames_raw, label="after parsing")

    postprocess_kwargs = dict(postprocess_kwargs or {})
    pp_verbose = bool(postprocess_kwargs.pop("verbose", verbose))
    frames_final, pp_summary = run_postprocess_pipeline(frames_raw, verbose=pp_verbose, **postprocess_kwargs)

    if verbose:
        print(f"[parse] postprocess: {len(frames_final)}/{len(frames_raw)} frames kept")
        _report_frame_keys(frames_final, label="after postprocess", prev_keys=keys_after_parse)

    required_key_check = _check_required_keys_all_frames(
        frames_final,
        required_keys_all_frames or (),
        warn_only=bool(required_keys_warn_only),
    )
    padding_info = _pad_final_frames(frames_final, qm_pad_to=qm_pad_to, mm_pad_to=mm_pad_to)

    available_final = _collect_key_presence(frames_final)
    if warn_available_keys:
        _warn_available_keys(frames_final, label="final")

    resolved_dump_dir: Optional[str] = None
    if dump_arrays:
        resolved_dump_dir = os.path.abspath(dump_dir or os.path.join(resolved_out_dir, "arrays"))
        dump_frames_to_numpy(
            frames_final,
            resolved_dump_dir,
            fmt=dump_fmt,
            overwrite=dump_overwrite,
            verbose=verbose,
        )

    elapsed = time.time() - t0
    write_parse_summary(
        frames_raw,
        frames_final,
        adapter=adapter,
        path=path,
        pp_summary=pp_summary,
        elapsed_s=elapsed,
        out_dir=resolved_out_dir,
        dump_dir=resolved_dump_dir,
        summary_json_path=summary_json,
        summary_txt_path=summary_txt,
        verbose=verbose,
        required_key_check=required_key_check,
        padding_info=padding_info,
        available_keys=available_final,
    )

    return frames_final


__all__ = [
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "dump_frames_to_numpy",
    "write_parse_summary",
    "parse_dataset",
]
