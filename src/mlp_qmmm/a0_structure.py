# src/mlp_qmmm/a0_structure.py
"""
Canonical frame key pool and shared constants for mlp_qmmm (v1.x)

Purpose
-------
Single shared vocabulary for all adapters, models, and the training loop.
Only canonical naming contracts, unit conventions, shared defaults, shape docs,
and minimal validation/introspection helpers live here. No parsing, batching,
training, or model logic belongs in this module.

Using Keys.* constants (instead of raw strings) turns typos into ImportErrors.

Global unit conventions (ALL adapters MUST convert into these)
--------------------------------------------------------------
Energies            eV          float32     E_high, E_low, dE, qm_energy, ...
Coordinates         Å           float32     qm_coords, mm_coords
Forces/gradients    eV/Å        float32     qm_grad_*, mm_grad_*, *_dgrad
ESP (∂E/∂q)         eV/e        float32     mm_esp_*, mm_desp
ESP-grad (∂E/∂R)/q  eV/(Å·e)    float32     mm_espgrad_*, mm_despgrad
Charges             e           float32     qm_Q, mm_Q
Atomic numbers      —           int32       qm_Z
Atom indices        —           int32       qm_idx, mm_idx  (adapter documents 0- vs 1-based)
Counts              —           int32       N_qm, N_mm, max_qm, max_mm
Type flags          —           float32     qm_type, mm_type  {1.0=real, 0.0=pad/dummy}
Species type idx    —           int64       atom_types  (optional later mapping)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple


class Units:
    """Canonical unit-conversion constants shared across adapters."""

    KCALMOL_TO_EV: float = 0.0433641153087705
    EV_TO_KCALMOL: float = 1.0 / KCALMOL_TO_EV


class Defaults:
    """Shared runtime defaults that adapters may import and override via YAML."""

    MAX_QM: int = 100
    MAX_MM: int = 5000


# ---------------------------------------------------------------------------
# 1.  String constants — import these; never write raw key strings
# ---------------------------------------------------------------------------
#
# HOW TO ADD A NEW KEY
# --------------------
# Step 1 — add a class attribute to Keys (pick a section or add a new one):
#
#     MY_NEW_KEY = "my_new_key"
#
# Step 2 — add the string to KEY_POOL (section 2 below), in logical order:
#
#     Keys.MY_NEW_KEY,
#
# That's it. _KEY_SET, KEYS_POOL.as_set(), and validate_frame() all update
# automatically since they are derived from KEY_POOL at import time.
#
# MODEL / DATA MODES
# ------------------
# This module is intentionally mode-agnostic. Different datasets or models may
# use different subsets of the canonical pool. Typical examples include:
#   - QM-only mode: qm_* keys present, mm_* keys absent
#   - QM/MM energy mode: E_high/E_low/dE present, gradient keys optional
#   - QM/MM force mode: gradient keys present in addition to energies
#   - ESP-aware mode: mm_esp_* and/or mm_espgrad_* present
#   - Demeaned-target mode: *_dm keys present alongside raw targets
#
# The rule is: add a new key here only if it is stable, reusable, and part of a
# cross-module contract. Experimental or model-specific transient keys should
# stay prefixed until they stabilise.
#
# NAMING CONVENTIONS FOR KEYS NOT YET IN THE POOL
# ------------------------------------------------
# Use a recognised prefix so validate_frame(allow_prefixes=(...)) stays clean:
#   "feat_"  — computed features / descriptors
#   "dbg_"   — debug / diagnostic quantities
#   "tmp_"   — temporary intermediates, never saved to disk
#   "mode_"  — mode-specific metadata flags / routing hints
#   "aux_"   — auxiliary targets or side-channel supervision
# Once a prefixed key stabilises, promote it here following the two steps above.


class Keys:
    """
    Namespace of canonical frame key strings.

    Usage::

        from mlp_qmmm.a0_structure import Keys
        frame[Keys.QM_COORDS]  # instead of frame["qm_coords"]
    """

    # --- bookkeeping metadata (not model tensors; strip before collating/batching) ---
    SOURCE_FILE = "source_file"   # str  — replaces dunder __file__
    FRAME_IDX   = "frame_idx"     # int  — replaces dunder __frame__

    # --- counts / pad targets ---
    N_QM   = "N_qm"               # (1,) int32  real QM atom count in this frame
    N_MM   = "N_mm"               # (1,) int32  real MM atom count in this frame
    MAX_QM = "max_qm"             # (1,) int32  configured QM pad width used by adapter / batching
    MAX_MM = "max_mm"             # (1,) int32  configured MM pad width used by adapter / batching

    # --- atom identity ---
    QM_IDX  = "qm_idx"            # (max_qm,) int32    padded index array; 0 beyond N_qm
    QM_Z    = "qm_Z"              # (max_qm,) int32    atomic number; 0 marks padded QM rows
    QM_Q    = "qm_Q"              # (max_qm,) float32  partial charge (e); 0 on padded rows
    QM_TYPE = "qm_type"           # (max_qm,) float32  {1.0=real, 0.0=pad/dummy} for now
    MM_IDX  = "mm_idx"            # (max_mm,) int32    padded index array aligned with MM arrays

    # --- species / type indexing (optional later mapping) ---
    SPECIES_ORDER = "species_order"  # (N_types,) int32  sorted unique Z values used to build atom_types
    ATOM_TYPES    = "atom_types"     # (max_qm,)  int64  downstream type index; pad handling is downstream policy

    # --- coordinates (Å) ---
    QM_COORDS = "qm_coords"       # (max_qm, 3) float32  padded with zeros beyond N_qm
    MM_COORDS = "mm_coords"       # (max_mm, 3) float32

    # --- type flags / pad markers ---
    MM_TYPE = "mm_type"           # (max_mm,) float32  {1.0=real, 0.0=pad/dummy} for now
    MM_Q    = "mm_Q"              # (max_mm,) float32  MM charge (e)

    # --- RAW energies (eV) ---
    E_HIGH = "E_high"             # (1,) float32
    E_LOW  = "E_low"              # (1,) float32

    # --- QM-only energy/gradient (no MM; absent in pure QM/MM datasets) ---
    QM_ENERGY = "qm_energy"       # (1,) float32  (eV)
    QM_GRAD   = "qm_grad"         # (max_qm, 3) float32  (eV/Å) padded with zeros

    # --- QM gradients from QM/MM (eV/Å) ---
    QM_GRAD_HIGH = "qm_grad_high"  # (max_qm, 3) float32
    QM_GRAD_LOW  = "qm_grad_low"   # (max_qm, 3) float32

    # --- MM gradients (eV/Å) ---
    MM_GRAD_HIGH = "mm_grad_high"  # (max_mm, 3) float32
    MM_GRAD_LOW  = "mm_grad_low"   # (max_mm, 3) float32

    # --- MM ESP quantities ---
    MM_ESP_HIGH     = "mm_esp_high"      # (max_mm,)   float32  ∂E/∂q  (eV/e)
    MM_ESP_LOW      = "mm_esp_low"       # (max_mm,)   float32
    MM_ESPGRAD_HIGH = "mm_espgrad_high"  # (max_mm, 3) float32  (1/q)·∂E/∂R  (eV/(Å·e))
    MM_ESPGRAD_LOW  = "mm_espgrad_low"   # (max_mm, 3) float32

    # --- demeaned energies (eV) ---
    E_HIGH_DM = "E_high_dm"       # (1,) float32
    E_LOW_DM  = "E_low_dm"        # (1,) float32

    # --- delta quantities (eV or eV/Å) ---
    DE          = "dE"            # (1,)        float32  E_high − E_low  (always raw, never demeaned)
    DE_DM       = "dE_dm"         # (1,)        float32  demeaned delta
    QM_DGRAD    = "qm_dgrad"      # (max_qm, 3) float32  qm_grad_high − qm_grad_low
    MM_DGRAD    = "mm_dgrad"      # (max_mm, 3) float32
    MM_DESPGRAD = "mm_despgrad"   # (max_mm, 3) float32  (eV/(Å·e))
    MM_DESP     = "mm_desp"       # (max_mm,)   float32  (eV/e)


# ---------------------------------------------------------------------------
# 2.  Full key pool (ordered: bookkeeping → counts → identity → coords → ...)
# ---------------------------------------------------------------------------

KEY_POOL: Tuple[str, ...] = (
    # bookkeeping
    Keys.SOURCE_FILE,
    Keys.FRAME_IDX,

    # counts
    Keys.N_QM,
    Keys.N_MM,
    Keys.MAX_QM,
    Keys.MAX_MM,

    # atom identity / indexing
    Keys.QM_IDX,
    Keys.QM_Z,
    Keys.QM_Q,
    Keys.QM_TYPE,
    Keys.MM_IDX,
    Keys.SPECIES_ORDER,
    Keys.ATOM_TYPES,

    # coordinates
    Keys.QM_COORDS,
    Keys.MM_COORDS,

    # type flags & charges
    Keys.MM_TYPE,
    Keys.MM_Q,

    # raw energies
    Keys.E_HIGH,
    Keys.E_LOW,

    # QM-only mode
    Keys.QM_ENERGY,
    Keys.QM_GRAD,

    # QM/MM gradients
    Keys.QM_GRAD_HIGH,
    Keys.QM_GRAD_LOW,

    # MM gradients
    Keys.MM_GRAD_HIGH,
    Keys.MM_GRAD_LOW,

    # MM ESP
    Keys.MM_ESP_HIGH,
    Keys.MM_ESP_LOW,
    Keys.MM_ESPGRAD_HIGH,
    Keys.MM_ESPGRAD_LOW,

    # demeaned
    Keys.E_HIGH_DM,
    Keys.E_LOW_DM,

    # delta
    Keys.DE,
    Keys.DE_DM,
    Keys.QM_DGRAD,
    Keys.MM_DGRAD,
    Keys.MM_DESPGRAD,
    Keys.MM_DESP,
)

# Fast lookup set (module-level, built once)
_KEY_SET: Set[str] = set(KEY_POOL)

# Canonical shape note for padded atom-backed arrays:
#   N_QM and N_MM store the real QM/MM atom counts for the frame.
#   QM_IDX / QM_Z / QM_Q / QM_TYPE / QM_COORDS / QM_GRAD* / QM_DGRAD and any QM-derived
#   fixed-width targets may be padded to max_qm, with padded rows zero-filled and QM_Z = 0
#   marking padded QM entries.
#   MM_COORDS / MM_Q / MM_TYPE / MM_IDX and MM-derived targets may be padded to max_mm when
#   batching or when an adapter chooses fixed-width storage.
#   QM_TYPE and MM_TYPE currently mark real vs padded entries {1.0=real, 0.0=pad/dummy}.
#   The type keys are kept float32 for consistency with frame storage; later adapters may
#   assign richer positive codes while still reserving 0.0 for pad.
#   MAX_QM / MAX_MM are optional per-frame metadata copies of configured pad widths so
#   downstream stages can validate assumptions without rereading YAML.

# Keys that are metadata fields (for example str/int), not model tensors — strip before
# collating/batching unless a downstream utility explicitly keeps them.
NON_TENSOR_KEYS: Tuple[str, ...] = (
    Keys.SOURCE_FILE,
    Keys.FRAME_IDX,
)


# ---------------------------------------------------------------------------
# 3.  KeyPool helper — validation and introspection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeyPool:
    """
    Wraps KEY_POOL with minimal validation/introspection helpers.

    The module-level instance ``KEYS_POOL`` is the singleton to import when a
    helper object is preferred over using KEY_POOL / _KEY_SET directly.
    """

    keys: Tuple[str, ...] = field(default=KEY_POOL)

    def as_set(self) -> Set[str]:
        return _KEY_SET

    def as_list(self) -> List[str]:
        return list(self.keys)

    def unexpected_keys(
        self,
        frame: Dict[str, object],
        *,
        allow_prefixes: Sequence[str] = (),
    ) -> List[str]:
        """
        Return frame keys not in the pool (useful for adapter debugging).

        Parameters
        ----------
        allow_prefixes:
            Keys starting with these prefixes are silently allowed,
            e.g. ``("feat_", "dbg_")`` for experimental outputs.
        """
        out: List[str] = []
        for k in frame:
            if k in _KEY_SET:
                continue
            if any(k.startswith(p) for p in allow_prefixes):
                continue
            out.append(k)
        return sorted(out)

    def validate_frame(
        self,
        frame: Dict[str, object],
        *,
        required: Optional[Sequence[str]] = None,
        allow_prefixes: Sequence[str] = (),
        strict: bool = False,
    ) -> List[str]:
        """
        Validate a frame dict. Returns a list of warning/error strings.
        """
        issues: List[str] = []

        if required:
            for k in required:
                if k not in frame:
                    issues.append(f"[ERROR] missing required key '{k}'")

        unexpected = self.unexpected_keys(frame, allow_prefixes=allow_prefixes)
        for k in unexpected:
            tag = "[ERROR]" if strict else "[WARN]"
            issues.append(f"{tag} unexpected key '{k}' (not in KEY_POOL)")

        return issues


# ---------------------------------------------------------------------------
# 4.  Module-level singleton
# ---------------------------------------------------------------------------

KEYS_POOL = KeyPool()
