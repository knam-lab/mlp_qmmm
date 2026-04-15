
"""
Model registry and loader for mlp_qmmm (v1.x)

Responsibilities
----------------
1. Model registry — register_model() / get_model()
2. Model loading — load_model() builds the model, loads checkpoints, wraps
   DataParallel, resolves model-declared IO contracts, and returns a ModelIO bundle.
3. Model summary — writes model_summary.json / model_summary.txt to a user-chosen
   directory or a sensible fallback location.

Design notes
------------
- Prefer v1-style class attributes:
    INPUT_KEYS
    OUTPUT_KEYS
    ALLOWED_LOSS_KEYS
  with backward-compatible fallbacks to legacy names / methods.
- The loader is model-agnostic. Models self-register at import time.
- The loader does not prepare model-specific derived inputs itself, but if a model
  exposes prepare_inputs(), the bound callable is surfaced in ModelIO for the
  training/data-prep stage.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(
    name: str,
    cls: Optional[Type[nn.Module]] = None,
) -> Union[Callable[[Type[nn.Module]], Type[nn.Module]], Type[nn.Module]]:
    """
    Register a model class under a short name.

    Supports both call styles::

        @register_model("dpmm_v1")
        class DeepPotMM(nn.Module): ...

        register_model("dpmm_v1", DeepPotMM)
    """
    name = str(name).strip()
    if not name:
        raise ValueError("register_model requires a non-empty name.")

    def _register(c: Type[nn.Module]) -> Type[nn.Module]:
        if not isinstance(c, type) or not issubclass(c, nn.Module):
            raise TypeError(
                f"register_model: '{name}' must be an nn.Module subclass, got {c!r}."
            )
        _MODEL_REGISTRY[name] = c
        return c

    return _register if cls is None else _register(cls)


def list_models() -> List[str]:
    return sorted(_MODEL_REGISTRY.keys())


def _resolve_import_spec(spec: str) -> Type[nn.Module]:
    """Resolve 'module:Class' or 'module.Class' to an nn.Module subclass."""
    if ":" in spec:
        mod_name, attr = spec.split(":", 1)
    elif "." in spec:
        mod_name, attr = spec.rsplit(".", 1)
    else:
        raise ValueError(
            f"Cannot resolve model spec '{spec}'. "
            f"Use a registered name, 'module:Class', or 'module.Class'."
        )
    mod = import_module(mod_name)
    obj = getattr(mod, attr)
    if not isinstance(obj, type) or not issubclass(obj, nn.Module):
        raise TypeError(f"Resolved '{spec}' but it is not an nn.Module subclass: {obj!r}")
    return obj


def get_model(name: str) -> Type[nn.Module]:
    """
    Return a model class given its registered name or import spec.

    Resolution order
    ----------------
    1. Registry hit
    2. Lazy import of mlp_qmmm.b_nn_types.<name>
    3. Import spec "module:Class" or "module.Class"
    """
    key = str(name).strip()
    if not key:
        raise ValueError("get_model requires a non-empty name.")

    if key in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[key]

    try:
        import_module(f"mlp_qmmm.b_nn_types.{key}")
        if key in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[key]
    except ModuleNotFoundError:
        pass

    cls = _resolve_import_spec(key)
    _MODEL_REGISTRY[key] = cls
    return cls


# ---------------------------------------------------------------------------
# ModelIO bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelIO:
    """
    Bundle returned by load_model().

    Attributes
    ----------
    model :
        The nn.Module ready for use (possibly wrapped in DataParallel).
    core :
        The underlying module (never DataParallel).
    input_keys :
        Frame keys required as direct model inputs.
    output_keys :
        Keys the model can return from forward().
    allowed_loss_keys :
        Keys the trainer may supervise on for this model.
    prepare_inputs :
        Optional bound callable for model-specific one-time input preparation.
    """
    model: nn.Module
    core: nn.Module
    input_keys: List[str]
    output_keys: List[str]
    allowed_loss_keys: List[str]
    prepare_inputs: Optional[Callable[..., Any]] = None

    # Backward-compatible alias
    @property
    def required_keys(self) -> List[str]:
        return self.input_keys


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_checkpoint(
    core: nn.Module,
    ckpt_path: str,
    *,
    strict: bool = True,
    map_location: str = "cpu",
) -> None:
    """
    Load a checkpoint into `core` (the non-DataParallel module).

    Accepts:
      - a raw state_dict
      - a dict with a "model_state" or "state_dict" key
    """
    raw = torch.load(ckpt_path, map_location=map_location)

    if isinstance(raw, dict):
        if "model_state" in raw:
            state_dict = raw["model_state"]
        elif "state_dict" in raw:
            state_dict = raw["state_dict"]
        else:
            state_dict = raw
    else:
        raise TypeError(f"Unsupported checkpoint type at '{ckpt_path}': {type(raw)}")

    try:
        core.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        stripped = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
        core.load_state_dict(stripped, strict=strict)


# ---------------------------------------------------------------------------
# Contract resolution helpers
# ---------------------------------------------------------------------------

def _resolve_input_keys(core: nn.Module, cfg_fallback: List[str]) -> List[str]:
    """
    Resolve the model's required input keys.

    Resolution order
    ----------------
    1. cls.INPUT_KEYS              (v1 style, preferred)
    2. cls.REQUIRED_KEYS           (backward-compatible alias)
    3. core.required_input_keys()  (legacy)
    4. cfg_fallback                (config model.inputs)
    """
    attr = getattr(type(core), "INPUT_KEYS", None)
    if attr is not None:
        return list(attr)

    attr = getattr(type(core), "REQUIRED_KEYS", None)
    if attr is not None:
        return list(attr)

    method = getattr(core, "required_input_keys", None)
    if callable(method):
        keys = method()
        if keys:
            return list(keys)

    if cfg_fallback:
        return list(cfg_fallback)

    raise ValueError(
        "Could not determine model input keys.\n"
        "Define INPUT_KEYS (preferred), REQUIRED_KEYS, implement "
        "required_input_keys(), or set model.inputs in config."
    )


def _resolve_output_keys(core: nn.Module, cfg_fallback: List[str]) -> List[str]:
    """
    Resolve the model's forward output keys.

    Resolution order
    ----------------
    1. cls.OUTPUT_KEYS
    2. core.supported_loss_keys()  (legacy fallback)
    3. cfg_fallback                (config model.output_keys or model.loss_keys)
    """
    attr = getattr(type(core), "OUTPUT_KEYS", None)
    if attr is not None:
        return list(attr)

    method = getattr(core, "supported_loss_keys", None)
    if callable(method):
        keys = method()
        if keys:
            return list(keys)

    if cfg_fallback:
        return list(cfg_fallback)

    raise ValueError(
        "Could not determine model output keys.\n"
        "Define OUTPUT_KEYS (preferred), implement supported_loss_keys(), "
        "or set model.output_keys / model.loss_keys in config."
    )


def _resolve_allowed_loss_keys(
    core: nn.Module,
    output_keys: List[str],
    cfg_fallback: List[str],
) -> List[str]:
    """
    Resolve which loss keys the trainer may supervise on.

    Resolution order
    ----------------
    1. cls.ALLOWED_LOSS_KEYS
    2. cls.LOSS_KEYS              (legacy / shorthand alias)
    3. output_keys
    4. cfg_fallback
    """
    attr = getattr(type(core), "ALLOWED_LOSS_KEYS", None)
    if attr is not None:
        return list(attr)

    attr = getattr(type(core), "LOSS_KEYS", None)
    if attr is not None:
        return list(attr)

    if output_keys:
        return list(output_keys)

    if cfg_fallback:
        return list(cfg_fallback)

    return []


def _resolve_prepare_inputs(core: nn.Module) -> Optional[Callable[..., Any]]:
    """
    Return a bound preparation helper if the model exposes one.

    Preferred name:
      prepare_inputs(frame, cfg, ...)
    """
    method = getattr(core, "prepare_inputs", None)
    return method if callable(method) else None


# ---------------------------------------------------------------------------
# Model summary writer
# ---------------------------------------------------------------------------

def _resolve_summary_dir(cfg: Dict[str, Any]) -> str:
    """
    Resolve directory for model summary files.

    Priority
    --------
    1. cfg["model"]["spec_dir"]
    2. cfg["parser"]["out_dir"]
    3. os.getcwd()
    """
    mcfg = cfg.get("model", {}) or {}
    pcfg = cfg.get("parser", {}) or {}

    spec_dir = str(mcfg.get("spec_dir", "") or "").strip()
    parse_dir = str(pcfg.get("out_dir", "") or "").strip()

    if spec_dir:
        return os.path.abspath(spec_dir)
    if parse_dir:
        return os.path.abspath(parse_dir)
    return os.getcwd()


def _atomic_write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _write_model_summary(
    core: nn.Module,
    *,
    name: str,
    input_keys: List[str],
    output_keys: List[str],
    allowed_loss_keys: List[str],
    device: torch.device,
    ckpt_path: Optional[str],
    ckpt_strict: bool,
    use_dp: bool,
    cfg: Dict[str, Any],
    out_dir: str,
    verbose: bool = False,
) -> Tuple[str, str]:
    """
    Write model_summary.json and model_summary.txt to `out_dir`.
    """
    json_path = os.path.join(out_dir, "model_summary.json")
    txt_path = os.path.join(out_dir, "model_summary.txt")

    n_total = sum(p.numel() for p in core.parameters())
    n_trainable = sum(p.numel() for p in core.parameters() if p.requires_grad)
    n_buffers = sum(b.numel() for b in core.buffers())

    prepare_fn = _resolve_prepare_inputs(core)
    ts = time.time()

    payload: Dict[str, Any] = {
        "timestamp": ts,
        "model": {
            "name": name,
            "class": type(core).__qualname__,
            "module": type(core).__module__,
            "device": str(device),
            "dataparallel": use_dp,
            "checkpoint": ckpt_path,
            "ckpt_strict": ckpt_strict,
            "has_prepare_inputs": bool(prepare_fn),
        },
        "parameters": {
            "total": n_total,
            "trainable": n_trainable,
            "non_trainable": n_total - n_trainable,
            "buffers": n_buffers,
        },
        "input_keys": input_keys,
        "output_keys": output_keys,
        "allowed_loss_keys": allowed_loss_keys,
        "config_model": cfg.get("model", {}),
        "summary_dir": out_dir,
    }
    _atomic_write(json_path, json.dumps(payload, indent=2, sort_keys=True))

    lines: List[str] = [
        "=== mlp_qmmm model_summary ===",
        f"timestamp         : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}",
        f"name              : {name}",
        f"class             : {type(core).__module__}.{type(core).__qualname__}",
        f"device            : {device}",
        f"dataparallel      : {use_dp}",
        f"checkpoint        : {ckpt_path or '(none)'}",
        f"ckpt_strict       : {ckpt_strict}",
        f"has_prepare_inputs: {bool(prepare_fn)}",
        "",
        "parameters:",
        f"  total          : {n_total:,}",
        f"  trainable      : {n_trainable:,}",
        f"  non-trainable  : {n_total - n_trainable:,}",
        f"  buffers        : {n_buffers:,}",
        "",
        f"input_keys ({len(input_keys)}):",
    ]
    for k in input_keys:
        lines.append(f"  {k}")
    lines.append("")
    lines.append(f"output_keys ({len(output_keys)}):")
    for k in output_keys:
        lines.append(f"  {k}")
    lines.append("")
    lines.append(f"allowed_loss_keys ({len(allowed_loss_keys)}):")
    for k in allowed_loss_keys:
        lines.append(f"  {k}")
    lines.append("")
    lines.append(f"summary_dir       : {out_dir}")
    lines.append("")

    _atomic_write(txt_path, "\n".join(lines))

    if verbose:
        print(f"[model] summary JSON → {json_path}")
        print(f"[model] summary TXT  → {txt_path}")

    return json_path, txt_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_model(
    cfg: Dict[str, Any],
    *,
    device: torch.device,
    verbose: bool = False,
) -> ModelIO:
    """
    Load a model end-to-end and return a ModelIO bundle.

    Config keys used
    ----------------
    model.name         : registered name or import spec (required)
    model.ckpt         : checkpoint path (optional)
    model.ckpt_strict  : bool, default True
    model.inputs       : fallback input keys (optional)
    model.output_keys  : fallback output keys (optional)
    model.loss_keys    : fallback allowed loss keys (optional)
    model.spec_dir     : summary directory (optional)
    parser.out_dir     : fallback summary directory (optional)
    trainer.dataparallel : bool, wrap in DataParallel if True and >1 GPU
    """
    mcfg = cfg.get("model", {}) or {}
    trcfg = cfg.get("trainer", {}) or {}

    name = str(mcfg.get("name", "")).strip()
    if not name:
        raise ValueError("Config missing model.name.")

    ModelCls = get_model(name)
    model = ModelCls(cfg).to(device)

    ckpt = mcfg.get("ckpt") or None
    ckpt_strict = bool(mcfg.get("ckpt_strict", True))
    if ckpt:
        _load_checkpoint(model, str(ckpt), strict=ckpt_strict, map_location=str(device))

    requested_dp = bool(trcfg.get("dataparallel", False))
    use_dp = requested_dp and torch.cuda.device_count() > 1
    if requested_dp and not use_dp:
        warnings.warn(
            "[load_model] trainer.dataparallel=True but fewer than 2 CUDA devices are visible; "
            "continuing without DataParallel."
        )
    if use_dp:
        model = nn.DataParallel(model)

    core = model.module if isinstance(model, nn.DataParallel) else model

    cfg_input_keys = list(mcfg.get("inputs", []) or [])
    cfg_output_keys = list(mcfg.get("output_keys", []) or [])
    cfg_loss_keys = list(mcfg.get("loss_keys", []) or [])

    input_keys = _resolve_input_keys(core, cfg_input_keys)
    output_keys = _resolve_output_keys(core, cfg_output_keys or cfg_loss_keys)
    allowed_loss_keys = _resolve_allowed_loss_keys(core, output_keys, cfg_loss_keys)
    prepare_inputs = _resolve_prepare_inputs(core)

    summary_dir = _resolve_summary_dir(cfg)
    _write_model_summary(
        core,
        name=name,
        input_keys=input_keys,
        output_keys=output_keys,
        allowed_loss_keys=allowed_loss_keys,
        device=device,
        ckpt_path=str(ckpt) if ckpt else None,
        ckpt_strict=ckpt_strict,
        use_dp=use_dp,
        cfg=cfg,
        out_dir=summary_dir,
        verbose=verbose,
    )

    return ModelIO(
        model=model,
        core=core,
        input_keys=input_keys,
        output_keys=output_keys,
        allowed_loss_keys=allowed_loss_keys,
        prepare_inputs=prepare_inputs,
    )


__all__ = [
    "register_model",
    "list_models",
    "get_model",
    "load_model",
    "ModelIO",
]
