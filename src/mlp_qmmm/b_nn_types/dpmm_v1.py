from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mlp_qmmm.a0_structure import Keys
from mlp_qmmm.b_nn_loader import register_model

QM_TYPE_KEY = getattr(Keys, "QM_TYPE", "qm_type")

PREP_INPUT_KEYS: Tuple[str, ...] = (
    Keys.QM_Z,
    QM_TYPE_KEY,
)
PREP_OUTPUT_KEYS: Tuple[str, ...] = (
    Keys.ATOM_TYPES,
)

INPUT_KEYS: Tuple[str, ...] = (
    Keys.QM_COORDS,
    QM_TYPE_KEY,
    Keys.ATOM_TYPES,
    Keys.MM_COORDS,
    Keys.MM_Q,
    Keys.MM_TYPE,
)

ALLOWED_LOSS_KEYS: Tuple[str, ...] = (
    "energy",
    "dE",
    "dE_dm",
    "qm_grad",
    "qm_grad_high",
    "qm_grad_low",
    "qm_dgrad",
    "mm_grad",
    "mm_grad_high",
    "mm_grad_low",
    "mm_dgrad",
    "mm_esp",
    "mm_esp_high",
    "mm_esp_low",
    "mm_espgrad",
    "mm_espgrad_high",
    "mm_espgrad_low",
    "mm_espgrad_d",
    "mm_despgrad",
    "mm_desp",
)

OUTPUT_KEYS: Tuple[str, ...] = ALLOWED_LOSS_KEYS


# ---------------------------------------------------------------------------
# Preparation helper
# For this v1:
#   - user manually includes 0:0 in z_to_type for padded QM atoms
#   - user manually sets n_types = real_types + 1
#   - real QM atoms must not map to 0
# ---------------------------------------------------------------------------

def _build_type_lookup_from_cfg(cfg: Dict[str, Any]) -> Tuple[Tensor, int, str, int]:
    m = cfg.get("model", {})
    tcfg = m.get("atom_types", {})

    if not isinstance(tcfg, dict):
        raise ValueError("model.atom_types must be a mapping in the YAML config.")

    z_to_type = dict(tcfg.get("z_to_type", {}))
    if not z_to_type:
        raise ValueError(
            "model.atom_types.z_to_type is required to prepare Keys.ATOM_TYPES."
        )

    n_types = int(tcfg.get("n_types", 0))
    if n_types <= 0:
        raise ValueError("model.atom_types.n_types must be > 0.")

    max_Z = int(tcfg.get("max_Z", 118))
    if max_Z <= 0:
        raise ValueError("model.atom_types.max_Z must be > 0.")

    unknown_policy = str(tcfg.get("unknown_policy", "error")).strip().lower()
    if unknown_policy not in {"error", "map_to_zero"}:
        raise ValueError(
            "model.atom_types.unknown_policy must be 'error' or 'map_to_zero'."
        )

    pad_fill_type = int(tcfg.get("pad_fill_type", 0))
    if pad_fill_type < 0 or pad_fill_type >= n_types:
        raise ValueError(
            f"model.atom_types.pad_fill_type={pad_fill_type} must be in [0, {n_types - 1}]."
        )

    lut = torch.full((max_Z + 1,), -1, dtype=torch.long)
    for z_raw, t_raw in z_to_type.items():
        z = int(z_raw)
        t = int(t_raw)
        if z < 0 or z > max_Z:
            raise ValueError(f"z_to_type key Z={z} out of range [0, {max_Z}].")
        if t < 0 or t >= n_types:
            raise ValueError(f"z_to_type[{z}]={t} invalid for n_types={n_types}.")
        lut[z] = t

    return lut, n_types, unknown_policy, pad_fill_type


def build_atom_types_from_qm(
    qm_Z: Tensor,
    qm_type: Tensor,
    *,
    cfg: Dict[str, Any],
) -> Tensor:
    single_frame = False

    if qm_Z.dim() == 1:
        qm_Z = qm_Z.unsqueeze(0)
        single_frame = True
    elif qm_Z.dim() != 2:
        raise ValueError(
            f"Keys.QM_Z must be (max_qm,) or (B, max_qm), got shape {tuple(qm_Z.shape)}."
        )

    if qm_type.dim() == 1:
        qm_type = qm_type.unsqueeze(0)
    elif qm_type.dim() != 2:
        raise ValueError(
            f"Keys.QM_TYPE must be (max_qm,) or (B, max_qm), got shape {tuple(qm_type.shape)}."
        )

    if qm_type.shape != qm_Z.shape:
        raise ValueError(
            f"Keys.QM_TYPE must match Keys.QM_Z shape; got {tuple(qm_type.shape)} vs {tuple(qm_Z.shape)}."
        )

    lut, n_types, unknown_policy, pad_fill_type = _build_type_lookup_from_cfg(cfg)

    qm_Z = qm_Z.to(torch.long)

    if torch.any(qm_Z < 0):
        raise ValueError("Keys.QM_Z contains negative values.")

    max_Z = lut.numel() - 1
    if torch.any(qm_Z > max_Z):
        raise ValueError(
            f"Keys.QM_Z contains Z={int(qm_Z.max())} > model.atom_types.max_Z={max_Z}."
        )

    qm_real = qm_type > 0.5
    atom_types = torch.full_like(qm_Z, fill_value=pad_fill_type, dtype=torch.long)

    if torch.any(qm_real):
        z_real = qm_Z[qm_real]
        mapped = lut[z_real]

        if unknown_policy == "error":
            unknown = mapped < 0
            if torch.any(unknown):
                bad_z = z_real[unknown].unique().tolist()
                raise KeyError(
                    f"Atomic numbers not found in model.atom_types.z_to_type: {bad_z}"
                )
        else:
            mapped = torch.where(mapped < 0, torch.zeros_like(mapped), mapped)

        if torch.any((mapped < 0) | (mapped >= n_types)):
            raise ValueError("Prepared atom_types contains values outside [0, n_types-1].")

        if torch.any(mapped == 0):
            bad_z = z_real[mapped == 0].unique().tolist()
            raise ValueError(
                f"dpmm_v1 reserves atom type 0 for padded QM atoms only. "
                f"Real QM atoms mapped to 0 for Z values: {bad_z}"
            )

        atom_types[qm_real] = mapped

    if single_frame:
        atom_types = atom_types.squeeze(0)

    return atom_types


def prepare_atom_types_(
    frame: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    inplace: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    out = frame if inplace else dict(frame)

    if (not overwrite) and (Keys.ATOM_TYPES in out):
        return out

    missing = [k for k in PREP_INPUT_KEYS if k not in out]
    if missing:
        raise KeyError(
            f"dpmm_v1 atom-type preparation missing required keys: {missing}."
        )

    out[Keys.ATOM_TYPES] = build_atom_types_from_qm(
        out[Keys.QM_Z],
        out[QM_TYPE_KEY],
        cfg=cfg,
    )
    return out


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def _coulomb_factor() -> float:
    return 27.2114 * 0.529177249


def _local_environment_full(coords: Tensor, *, dist_eps: float) -> Tuple[Tensor, Tensor]:
    """
    coords: (B, N, 3)
    returns:
      loc_env_r: (B, N, N)
      loc_env_a: (B, N, N, 3)
    """
    rij = coords[:, :, None] - coords[:, None]         # (B,N,N,3)
    dij = torch.linalg.norm(rij, dim=3)                # (B,N,N)

    dij_inv = 1.0 / dij.clamp_min(dist_eps)
    loc_env_r = dij_inv
    loc_env_a = rij * (dij_inv * dij_inv).unsqueeze(-1)
    return loc_env_r, loc_env_a


def _remove_diag_2d(x: Tensor) -> Tensor:
    """
    x: (B, N, N) -> (B, N, N-1)
    """
    b, n, _ = x.shape
    mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
    return torch.masked_select(x, mask.unsqueeze(0)).view(b, n, n - 1)


def _remove_diag_3d(x: Tensor) -> Tensor:
    """
    x: (B, N, N, C) -> (B, N, N-1, C)
    """
    b, n, _, c = x.shape
    mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
    return torch.masked_select(x, mask.unsqueeze(0).unsqueeze(-1)).view(b, n, n - 1, c)


def _build_qm_pair_mask(qm_mask: Tensor) -> Tensor:
    """
    qm_mask: (B,N) bool
    returns: (B,N,N) bool; valid real-real non-self QM pairs
    """
    b, n = qm_mask.shape
    eye = torch.eye(n, dtype=torch.bool, device=qm_mask.device).unsqueeze(0)
    return qm_mask[:, :, None] & qm_mask[:, None, :] & (~eye)


def _esp_efield_masked(
    atom_coords: Tensor,      # (B,Nq,3)
    charge_coords: Tensor,    # (B,Nm,3)
    charges: Tensor,          # (B,Nm)
    real_mask_bool: Tensor,   # (B,Nm)
    *,
    dist_eps: float,
) -> Tuple[Tensor, Tensor]:
    b, nq, _ = atom_coords.shape
    nm = charge_coords.shape[1]

    rij = atom_coords[:, :, None] - charge_coords[:, None, :]   # (B,Nq,Nm,3)
    dij = torch.linalg.norm(rij, dim=3)                         # (B,Nq,Nm)

    pair_mask = real_mask_bool[:, None, :].expand(b, nq, nm)

    d_safe = dij.clamp_min(dist_eps)
    inv_d_all = 1.0 / d_safe
    inv_d3_all = inv_d_all * inv_d_all * inv_d_all

    zeros = torch.zeros_like(dij)
    inv_d = torch.where(pair_mask, inv_d_all, zeros)
    inv_d3 = torch.where(pair_mask, inv_d3_all, zeros)

    factor = _coulomb_factor()
    esp = torch.sum(charges[:, None, :] * inv_d, dim=-1) * factor
    efield = torch.sum(
        charges[:, None, :, None] * rij * inv_d3[..., None], dim=-2
    ) * factor
    return esp, efield


# ---------------------------------------------------------------------------
# Network blocks
# ---------------------------------------------------------------------------

class Sequential(nn.Sequential):
    def forward(self, input: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x, channels = input
        for m in self:
            x, channels = m((x, channels))
        return x, channels


class Dense(nn.Module):
    """
    Same as dpmm_v0, but supports per-sample channel routing:
      channels: (L,) or (B,L)
    """

    def __init__(
        self,
        num_channels: int,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        activation: bool = False,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.activation = bool(activation)

        self.weight = nn.Parameter(
            torch.empty(self.num_channels, self.out_features, self.in_features)
        )
        self.bias: Optional[Tensor]
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_channels, self.out_features))
        else:
            self.register_parameter("bias", None)

        if residual:
            if self.out_features == self.in_features:
                self._res_mode = 1
            elif self.out_features == 2 * self.in_features:
                self._res_mode = 2
            else:
                raise RuntimeError("Residual shape mismatch")
        else:
            self._res_mode = 0

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for w in self.weight:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        if self.bias is not None:
            for b, w in zip(self.bias, self.weight):
                fan_in = w.shape[-1]
                bound = 1.0 / math.sqrt(max(int(fan_in), 1))
                nn.init.uniform_(b, -bound, bound)

    def forward(self, input: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x, channels = input
        if channels.dtype != torch.long:
            channels = channels.long()

        if channels.dim() == 1:
            w = self.weight[channels]  # (L,Cout,Cin)
            y = torch.bmm(x.transpose(0, 1), w.transpose(1, 2)).transpose(0, 1)
            if self.bias is not None:
                y = y + self.bias[channels]
        elif channels.dim() == 2:
            if channels.shape != x.shape[:2]:
                raise ValueError(
                    f"channels shape {tuple(channels.shape)} incompatible with x shape {tuple(x.shape)}."
                )
            w = self.weight[channels]                         # (B,L,Cout,Cin)
            y = torch.einsum("bloc,bli->blo", w, x)          # (B,L,Cout)
            if self.bias is not None:
                y = y + self.bias[channels]
        else:
            raise ValueError(f"channels ndim must be 1 or 2, got {channels.dim()}.")

        if self.activation:
            y = torch.tanh(y)

        if self._res_mode == 1:
            y = y + x
        elif self._res_mode == 2:
            y = y + torch.cat([x, x], dim=-1)

        return y, channels


# ---------------------------------------------------------------------------
# Descriptors: exact dpmm_v0 contraction pattern on N x (N-1)
# ---------------------------------------------------------------------------

class Feature(nn.Module):
    def __init__(
        self,
        n_types: int = 6,
        neuron: Sequence[int] = (25, 50, 100),
        axis_neuron: int = 4,
        dist_eps: float = 1.0e-12,
    ):
        super().__init__()
        self.n_types = int(n_types)
        self.neuron = list(neuron)
        self.axis_neuron = int(axis_neuron)
        self.dist_eps = float(dist_eps)

        layers = [Dense(self.n_types * self.n_types, 1, self.neuron[0], activation=True)]
        for i in range(len(self.neuron) - 1):
            layers.append(
                Dense(
                    self.n_types * self.n_types,
                    self.neuron[i],
                    self.neuron[i + 1],
                    activation=True,
                    residual=True,
                )
            )
        self.local_embedding = Sequential(*layers)

    def forward(self, coords: Tensor, atom_types: Tensor, qm_mask: Tensor) -> Tensor:
        nb, nc, _ = coords.size()

        pair_mask_full = _build_qm_pair_mask(qm_mask)                  # (B,N,N)
        loc_env_r_full, loc_env_a_full = _local_environment_full(coords, dist_eps=self.dist_eps)

        pair_idx_full = atom_types[:, :, None] * self.n_types + atom_types[:, None, :]  # (B,N,N)

        loc_env_r = _remove_diag_2d(loc_env_r_full)                   # (B,N,N-1)
        loc_env_a = _remove_diag_3d(loc_env_a_full)                   # (B,N,N-1,3)
        pair_idx = _remove_diag_2d(pair_idx_full)                     # (B,N,N-1)
        pair_mask = _remove_diag_2d(pair_mask_full)                   # (B,N,N-1)

        x = loc_env_r.view(nb, -1, 1)
        channels = pair_idx.reshape(nb, -1)
        out, _ = self.local_embedding((x, channels))
        out = out.view(nb, nc, nc - 1, -1)

        # exact v0 logic + mask invalid pairs before contraction
        out = out * pair_mask.unsqueeze(-1).to(out.dtype)
        loc_env_a = loc_env_a * pair_mask.unsqueeze(-1).to(loc_env_a.dtype)

        out = torch.transpose(out, 2, 3) @ (
            loc_env_a @ (torch.transpose(loc_env_a, 2, 3) @ out[..., : self.axis_neuron])
        )
        out = out.view(nb, nc, -1)
        out = out * qm_mask.unsqueeze(-1).to(out.dtype)
        return out

    @property
    def output_length(self) -> int:
        return self.neuron[-1] * self.axis_neuron


class ElectrostaticPotential(nn.Module):
    def __init__(
        self,
        n_types: int = 6,
        neuron: Sequence[int] = (5, 10, 20),
        axis_neuron: int = 4,
        dist_eps: float = 1.0e-12,
    ):
        super().__init__()
        self.n_types = int(n_types)
        self.neuron = list(neuron)
        self.axis_neuron = int(axis_neuron)
        self.dist_eps = float(dist_eps)

        esp_layers = [Dense(self.n_types, 1, self.neuron[0], bias=True, activation=True)]
        for i in range(len(self.neuron) - 1):
            esp_layers.append(
                Dense(
                    self.n_types,
                    self.neuron[i],
                    self.neuron[i + 1],
                    bias=True,
                    activation=True,
                    residual=True,
                )
            )
        self.esp_embedding = Sequential(*esp_layers)

        efield_layers = [Dense(self.n_types * self.n_types, 1, self.neuron[0], bias=True, activation=True)]
        for i in range(len(self.neuron) - 1):
            efield_layers.append(
                Dense(
                    self.n_types * self.n_types,
                    self.neuron[i],
                    self.neuron[i + 1],
                    bias=True,
                    activation=True,
                    residual=True,
                )
            )
        self.efield_embedding = Sequential(*efield_layers)

    def forward(
        self,
        atom_coords: Tensor,
        atom_types: Tensor,
        qm_mask: Tensor,
        charge_coords: Tensor,
        charges: Tensor,
        real_mask_bool: Tensor,
    ) -> Tensor:
        nb, nc, _ = atom_coords.size()

        pair_mask_full = _build_qm_pair_mask(qm_mask)                  # (B,N,N)
        _, loc_env_a_full = _local_environment_full(atom_coords, dist_eps=self.dist_eps)

        esp_scalar, efield = _esp_efield_masked(
            atom_coords, charge_coords, charges, real_mask_bool, dist_eps=self.dist_eps
        )

        esp_scalar = esp_scalar * qm_mask.to(esp_scalar.dtype)
        esp_out, _ = self.esp_embedding((esp_scalar.unsqueeze(-1), atom_types))
        esp_out = esp_out * qm_mask.unsqueeze(-1).to(esp_out.dtype)

        loc_env_a = _remove_diag_3d(loc_env_a_full)                   # (B,N,N-1,3)
        pair_idx = _remove_diag_2d(
            atom_types[:, :, None] * self.n_types + atom_types[:, None, :]
        )                                                             # (B,N,N-1)
        pair_mask = _remove_diag_2d(pair_mask_full)                   # (B,N,N-1)

        loc_env_a = loc_env_a * pair_mask.unsqueeze(-1).to(loc_env_a.dtype)

        proj = torch.bmm(
            loc_env_a.view(-1, nc - 1, 3),
            efield.view(-1, 3, 1),
        ).view(nb, nc, nc - 1)
        proj = proj * pair_mask.to(proj.dtype)

        x = proj.view(nb, -1, 1)
        channels = pair_idx.reshape(nb, -1)

        out, _ = self.efield_embedding((x, channels))
        out = out.view(nb, nc, nc - 1, -1)
        out = out * pair_mask.unsqueeze(-1).to(out.dtype)

        # exact v0 contraction
        out = torch.transpose(out, 2, 3) @ out[..., : self.axis_neuron]
        out = out.view(nb, nc, -1) * 2.0
        out = out * qm_mask.unsqueeze(-1).to(out.dtype)

        return torch.cat((esp_out, out), dim=2)

    @property
    def output_length(self) -> int:
        return self.neuron[-1] * (self.axis_neuron + 1)


class Fitting(nn.Module):
    def __init__(
        self,
        n_types: int = 6,
        in_features: int = 0,
        neuron: Sequence[int] = (240, 240, 240),
    ):
        super().__init__()
        self.n_types = int(n_types)
        self.neuron = list(neuron)

        layers = [Dense(self.n_types, in_features, self.neuron[0], activation=True)]
        for i in range(len(self.neuron) - 1):
            layers.append(
                Dense(
                    self.n_types,
                    self.neuron[i],
                    self.neuron[i + 1],
                    activation=True,
                    residual=True,
                )
            )
        layers.append(Dense(self.n_types, self.neuron[-1], 1))
        self.fitting_net = Sequential(*layers)

    def forward(self, input: Tuple[Tensor, Tensor]) -> Tensor:
        y, _ = self.fitting_net(input)
        return y


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DeepPotMMV1(nn.Module):
    """
    Fast variable-QM model derived from dpmm_v0.

    Assumptions
    -----------
    - User manually includes 0:0 in z_to_type for padded QM atoms.
    - User manually sets n_types = real_types + 1.
    - Real QM atoms must have nonzero atom_types.
    - Different QM orders across frames are allowed.
    - Fixed max_qm batching is used.
    """

    INPUT_KEYS: Tuple[str, ...] = INPUT_KEYS
    PREP_INPUT_KEYS: Tuple[str, ...] = PREP_INPUT_KEYS
    PREP_OUTPUT_KEYS: Tuple[str, ...] = PREP_OUTPUT_KEYS
    REQUIRED_KEYS: Tuple[str, ...] = INPUT_KEYS
    OUTPUT_KEYS: Tuple[str, ...] = OUTPUT_KEYS
    ALLOWED_LOSS_KEYS: Tuple[str, ...] = ALLOWED_LOSS_KEYS

    @classmethod
    def check_required_keys(cls, frame: Dict[str, Tensor]) -> None:
        missing = [k for k in cls.REQUIRED_KEYS if k not in frame]
        if missing:
            raise KeyError(
                f"[DeepPotMMV1] missing required frame keys: {missing}\n"
                f"Available keys: {sorted(frame.keys())}"
            )

    @classmethod
    def prepare_inputs(
        cls,
        frame: Dict[str, Any],
        cfg: Dict[str, Any],
        *,
        inplace: bool = True,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        return prepare_atom_types_(frame, cfg, inplace=inplace, overwrite=overwrite)

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get("model", {})
        dcfg = m.get("descriptor", {})
        ecfg = m.get("esp", {})
        fcfg = m.get("fitting_net", {})
        tcfg = m.get("atom_types", {})

        self.n_types = int(tcfg.get("n_types", dcfg.get("n_types", 6)))
        if self.n_types <= 1:
            raise ValueError(
                "dpmm_v1 expects n_types >= 2 when type 0 is reserved for padded QM atoms."
            )

        self.dist_eps = float(m.get("dist_eps", 1.0e-12))

        self.descriptor = Feature(
            n_types=self.n_types,
            neuron=tuple(dcfg.get("neuron", [25, 50, 100])),
            axis_neuron=int(dcfg.get("axis_neuron", 4)),
            dist_eps=self.dist_eps,
        )
        self.esp = ElectrostaticPotential(
            n_types=self.n_types,
            neuron=tuple(ecfg.get("neuron", [5, 10, 20])),
            axis_neuron=int(ecfg.get("axis_neuron", 4)),
            dist_eps=self.dist_eps,
        )
        self.fitting_net = Fitting(
            n_types=self.n_types,
            in_features=self.descriptor.output_length + self.esp.output_length,
            neuron=tuple(fcfg.get("neuron", [240, 240, 240])),
        )

    def _validate_atom_types(self, atom_types: Tensor, qm_type: Tensor) -> None:
        qm_real = qm_type > 0.5
        if torch.any(atom_types < 0):
            raise ValueError("atom_types contains negative values.")
        if torch.any(atom_types >= self.n_types):
            raise ValueError(
                f"atom_types contains value >= n_types ({self.n_types})."
            )
        if torch.any(atom_types[qm_real] == 0):
            raise ValueError(
                "dpmm_v1 forbids using atom type 0 for any real QM atom. "
                "Type 0 is reserved for padded QM atoms only."
            )

    def forward_from_frame(self, frame: Dict[str, Any]) -> Dict[str, Tensor]:
        self.check_required_keys(frame)
        return self.forward(
            qm_coords=frame[Keys.QM_COORDS],
            qm_type=frame[QM_TYPE_KEY],
            atom_types=frame[Keys.ATOM_TYPES],
            mm_coords=frame[Keys.MM_COORDS],
            mm_Q=frame[Keys.MM_Q],
            mm_type=frame[Keys.MM_TYPE],
        )

    def forward(
        self,
        qm_coords: Tensor,
        qm_type: Tensor,
        atom_types: Tensor,
        mm_coords: Tensor,
        mm_Q: Tensor,
        mm_type: Tensor,
    ) -> Dict[str, Tensor]:
        if atom_types.dtype != torch.long:
            atom_types = atom_types.long()

        real_qm = qm_type > 0.5
        real_mm = mm_type > 0

        self._validate_atom_types(atom_types, qm_type)

        qm_coords = qm_coords.requires_grad_(True)
        mm_coords = mm_coords.requires_grad_(True)
        mm_Q = mm_Q.requires_grad_(True)

        # same MM logic as dpmm_v0
        mm_mask = real_mm.to(mm_Q.dtype)
        q_eff = mm_Q * mm_mask

        desc = self.descriptor(qm_coords, atom_types, real_qm)
        esp = self.esp(qm_coords, atom_types, real_qm, mm_coords, q_eff, real_mm)
        feats = torch.cat((desc, esp), dim=-1)

        atom_E = self.fitting_net((feats, atom_types))
        atom_E = atom_E * real_qm.unsqueeze(-1).to(atom_E.dtype)
        energy = atom_E.sum(dim=1)

        retain = bool(self.training)
        create = bool(self.training)
        g_qm_opt, g_qeff_opt, g_r_opt = torch.autograd.grad(
            outputs=[energy.sum()],
            inputs=[qm_coords, q_eff, mm_coords],
            retain_graph=retain,
            create_graph=create,
            allow_unused=True,
        )

        g_qm = g_qm_opt if g_qm_opt is not None else torch.zeros_like(qm_coords)
        g_qeff = g_qeff_opt if g_qeff_opt is not None else torch.zeros_like(q_eff)
        g_r = g_r_opt if g_r_opt is not None else torch.zeros_like(mm_coords)

        qm_mask3 = real_qm.unsqueeze(-1).to(g_qm.dtype)
        mm_mask3 = mm_mask.unsqueeze(-1)

        g_qm_masked = g_qm * qm_mask3
        g_r_masked = g_r * mm_mask3
        mm_esp = g_qeff * mm_mask

        valid = real_mm & (mm_Q.abs() > 0)
        safe_q = torch.where(valid, mm_Q, torch.ones_like(mm_Q))
        mm_esp_grad = (g_r / safe_q.unsqueeze(-1)) * valid.unsqueeze(-1).to(g_r.dtype)

        return {
            "energy": energy,
            "dE": energy,
            "dE_dm": energy,

            "qm_grad": g_qm_masked,
            "qm_grad_high": g_qm_masked,
            "qm_grad_low": g_qm_masked,
            "qm_dgrad": g_qm_masked,

            "mm_grad": g_r_masked,
            "mm_grad_high": g_r_masked,
            "mm_grad_low": g_r_masked,
            "mm_dgrad": g_r_masked,

            "mm_esp": mm_esp,
            "mm_esp_high": mm_esp,
            "mm_esp_low": mm_esp,

            "mm_espgrad": mm_esp_grad,
            "mm_espgrad_high": mm_esp_grad,
            "mm_espgrad_low": mm_esp_grad,
            "mm_espgrad_d": mm_esp_grad,
            "mm_despgrad": mm_esp_grad,
            "mm_desp": mm_esp,
        }


register_model("dpmm_v1", DeepPotMMV1)