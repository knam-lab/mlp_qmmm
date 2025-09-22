# a_parser.py
from __future__ import annotations
from typing import Any, Callable, Dict, List, Union
import importlib

# Adapters return dict[str, Any] or list[dict[str, Any]]
_Adapter = Callable[..., Union[Dict[str, Any], List[Dict[str, Any]]]]
_ADAPTERS: Dict[str, _Adapter] = {}


def register_adapter(name: str, adapter: _Adapter) -> None:
    """
    Register an adapter callable under a short, human-readable name.
    Typically called from inside the adapter module.
    """
    if not name or not callable(adapter):
        raise ValueError("register_adapter requires a non-empty name and a callable.")
    _ADAPTERS[name] = adapter


def _lazy_import_by_name(name: str) -> None:
    """
    Lazy-import the adapter module so it can call register_adapter(...).
    Try exact module path first; if name has no dots, also try
    'mlp_qmmm.a_input_types.<name>'.
    """
    tried: List[str] = []
    # 1) exact import attempt
    try:
        importlib.import_module(name)
        return
    except Exception as e:
        tried.append(f"{name} ({type(e).__name__}: {e})")

    # 2) if it's a bare name, try the default namespace
    if "." not in name:
        mod = f"mlp_qmmm.a_input_types.{name}"
        try:
            importlib.import_module(mod)
            return
        except Exception as e:
            tried.append(f"{mod} ({type(e).__name__}: {e})")

    raise ImportError("Could not import adapter module. Tried:\n  - " + "\n  - ".join(tried))


def get_adapter(name: str) -> _Adapter:
    """
    Return a registered adapter callable. If it's not registered yet,
    attempt to import a module that registers it.

    Resolution order:
      1) exact match on 'name'
      2) if 'name' is a module path, try its last component as the registered name
      3) else → raise with helpful hints
    """
    # Fast path: already registered under this key
    if name in _ADAPTERS:
        return _ADAPTERS[name]

    # Try to import something that will register it
    try:
        _lazy_import_by_name(name)
    except ImportError:
        # Before failing, if the user provided a module path, try importing its basename
        if "." in name:
            base = name.rsplit(".", 1)[-1]
            try:
                _lazy_import_by_name(base)
            except Exception:
                # Fall through to error below
                pass
        # Re-check after attempts
        if name in _ADAPTERS:
            return _ADAPTERS[name]
        if "." in name:
            base = name.rsplit(".", 1)[-1]
            if base in _ADAPTERS:
                return _ADAPTERS[base]
        # Nothing registered; re-raise a clearer error
        raise

    # After import, check again for exact name
    if name in _ADAPTERS:
        return _ADAPTERS[name]

    # If user passed a module path, many adapters register a short name.
    if "." in name:
        base = name.rsplit(".", 1)[-1]
        if base in _ADAPTERS:
            return _ADAPTERS[base]

    # No silent fallback to "the only adapter" — that hides config mistakes.
    known = sorted(_ADAPTERS.keys())
    hint = ""
    if "." in name:
        base = name.rsplit(".", 1)[-1]
        if base in known:
            hint = f" (hint: adapter registered as '{base}')"
    raise KeyError(f"Unknown adapter '{name}'. Known adapters: {known}{hint}")
