from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mlp_qmmm.a0_structure import Keys
from mlp_qmmm.b_nn_loader import register_model

QM_TYPE_KEY = getattr(Keys, "QM_TYPE", "qm_type")
N_QM_KEY = getattr(Keys, "N_QM", "n_qm")

PREP_INPUT_KEYS: Tuple[str, ...] = (
    Keys.QM_Z,
    QM_TYPE_KEY,
)
PREP_OUTPUT_KEYS: Tuple[str, ...] = (
    Keys.ATOM_TYPES,
)

INPUT_KEYS: Tuple[str, ...] = (
    Keys.QM_COORDS,
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
# ---------------------------------------------------------------------------
def _debug_print_type_mapping_once(
    qm_Z: Tensor,
    qm_type: Tensor,
    atom_types: Tensor,
    *,
    cfg: Dict[str, Any],
    prefix: str = "[dpmm_v0:type_debug]",
) -> None:
    lut, n_types, unknown_policy, pad_fill_type = _build_type_lookup_from_cfg(cfg)

    if qm_Z.dim() == 2:
        qm_Z = qm_Z[0]
    if qm_type.dim() == 2:
        qm_type = qm_type[0]
    if atom_types.dim() == 2:
        atom_types = atom_types[0]

    qm_Z = qm_Z.detach().cpu().to(torch.long)
    qm_type = qm_type.detach().cpu()
    atom_types = atom_types.detach().cpu().to(torch.long)

    real_mask = qm_type > 0.5
    z_real = qm_Z[real_mask]
    t_real = atom_types[real_mask]

    print(prefix, "qm_Z(all)      =", qm_Z.tolist(), flush=True)
    print(prefix, "qm_type(all)   =", qm_type.tolist(), flush=True)
    print(prefix, "atom_types(all)=", atom_types.tolist(), flush=True)

    rows = []
    bad_rows = []
    for z, t in zip(z_real.tolist(), t_real.tolist()):
        expected = int(lut[int(z)].item()) if 0 <= int(z) < lut.numel() else -999
        rows.append((int(z), expected, int(t)))
        if expected != int(t):
            bad_rows.append((int(z), expected, int(t)))

    print(prefix, "real-only (Z, expected_type, got_type) =", rows, flush=True)

    uniq_z = sorted(set(int(z) for z in z_real.tolist()))
    summary = []
    for z in uniq_z:
        expected = int(lut[z].item()) if 0 <= z < lut.numel() else -999
        got = sorted(set(int(t) for zz, t in zip(z_real.tolist(), t_real.tolist()) if int(zz) == z))
        summary.append((z, expected, got))
    print(prefix, "unique summary (Z, expected_type, observed_types) =", summary, flush=True)

    if bad_rows:
        raise ValueError(
            f"{prefix} atom-type mismatch detected for real QM atoms: {bad_rows}"
        )






def _build_type_lookup_from_cfg(cfg: Dict[str, Any]) -> Tuple[Tensor, int, str, int]:
    m = cfg.get("model", {})
    tcfg = m.get("atom_types", {})

    if not isinstance(tcfg, dict):
        raise ValueError("model.atom_types must be a mapping in the YAML config.")

    z_to_type = dict(tcfg.get("z_to_type", {}))
    if not z_to_type:
        raise ValueError("model.atom_types.z_to_type is required.")

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
        if z <= 0 or z > max_Z:
            raise ValueError(f"z_to_type key Z={z} out of range [1, {max_Z}].")
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

        atom_types[qm_real] = mapped

    if single_frame:
        atom_types = atom_types.squeeze(0)

    return atom_types


#def prepare_atom_types_(
#    frame: Dict[str, Any],
#    cfg: Dict[str, Any],
#    *,
#    inplace: bool = True,
#    overwrite: bool = False,
#) -> Dict[str, Any]:
#    out = frame if inplace else dict(frame)
#
#    if (not overwrite) and (Keys.ATOM_TYPES in out):
#        return out
#
#    missing = [k for k in PREP_INPUT_KEYS if k not in out]
#    if missing:
#        raise KeyError(
#            f"dpmm_v0 atom-type preparation missing required keys: {missing}."
#        )
#
#    out[Keys.ATOM_TYPES] = build_atom_types_from_qm(
#        out[Keys.QM_Z],
#        out[QM_TYPE_KEY],
#        cfg=cfg,
#    )
#    return out

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
            f"dpmm_v0 atom-type preparation missing required keys: {missing}."
        )

    out[Keys.ATOM_TYPES] = build_atom_types_from_qm(
        out[Keys.QM_Z],
        out[QM_TYPE_KEY],
        cfg=cfg,
    )

    debug_cfg = ((cfg.get("trainer", {}) or {}).get("debug_type_mapping", False))
    if debug_cfg:
        _debug_print_type_mapping_once(
            out[Keys.QM_Z],
            out[QM_TYPE_KEY],
            out[Keys.ATOM_TYPES],
            cfg=cfg,
        )

    return out

# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def _coulomb_factor() -> float:
    return 27.2114 * 0.529177249


def _local_environment(coords: Tensor, *, dist_eps: float) -> Tuple[Tensor, Tensor]:
    nb, nc, _ = coords.size()
    rij = coords[:, :, None] - coords[:, None]
    dij = torch.linalg.norm(rij, dim=3)

    mask = ~torch.eye(nc, dtype=torch.bool, device=coords.device)
    rij = torch.masked_select(rij, mask.unsqueeze(0).unsqueeze(-1)).view(nb, nc, nc - 1, 3)
    dij = torch.masked_select(dij, mask.unsqueeze(0)).view(nb, nc, nc - 1)

    dij_inv = 1.0 / dij.clamp_min(dist_eps)
    loc_env_r = dij_inv
    loc_env_a = rij * (dij_inv * dij_inv).unsqueeze(-1)
    return loc_env_r, loc_env_a


def _esp_efield_masked(
    atom_coords: Tensor,
    charge_coords: Tensor,
    charges: Tensor,
    real_mask_bool: Tensor,
    *,
    dist_eps: float,
) -> Tuple[Tensor, Tensor]:
    b, nq, _ = atom_coords.shape
    nm = charge_coords.shape[1]

    rij = atom_coords[:, :, None] - charge_coords[:, None]
    dij = torch.linalg.norm(rij, dim=3)

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
# TorchScript-safer network blocks
# ---------------------------------------------------------------------------

class ChannelDense(nn.Module):
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

    def forward(self, x: Tensor, channels: Tensor) -> Tuple[Tensor, Tensor]:
        if channels.dim() != 1:
            raise ValueError(
                f"dpmm_v0 expects shared channel layout across batch; got channels shape {tuple(channels.shape)}."
            )
        if channels.dtype != torch.long:
            channels = channels.long()

        w = self.weight[channels]
        y = torch.bmm(x.transpose(0, 1), w.transpose(1, 2)).transpose(0, 1)

        if self.bias is not None:
            y = y + self.bias[channels]
        if self.activation:
            y = torch.tanh(y)

        if self._res_mode == 1:
            y = y + x
        elif self._res_mode == 2:
            y = y + torch.cat([x, x], dim=-1)

        return y, channels


class ChannelStack(nn.Module):
    def __init__(self, layers: List[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor, channels: Tensor) -> Tuple[Tensor, Tensor]:
        for layer in self.layers:
            x, channels = layer(x, channels)
        return x, channels


# ---------------------------------------------------------------------------
# Descriptors
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

        layers: List[nn.Module] = [
            ChannelDense(self.n_types * self.n_types, 1, self.neuron[0], activation=True)
        ]
        for i in range(len(self.neuron) - 1):
            layers.append(
                ChannelDense(
                    self.n_types * self.n_types,
                    self.neuron[i],
                    self.neuron[i + 1],
                    activation=True,
                    residual=True,
                )
            )
        self.local_embedding = ChannelStack(layers)

    def forward(self, coords: Tensor, atom_types: Tensor) -> Tensor:
        nb, nc, _ = coords.size()
        loc_env_r, loc_env_a = _local_environment(coords, dist_eps=self.dist_eps)

        neighbor_types = atom_types.repeat(nc).view(nc, nc)
        mask = ~torch.eye(nc, dtype=torch.bool, device=coords.device)
        neighbor_types = torch.masked_select(neighbor_types, mask).view(nc, nc - 1)
        pair_idx = (atom_types * self.n_types).unsqueeze(-1) + neighbor_types
        channels = pair_idx.reshape(-1).long()

        x = loc_env_r.view(nb, -1, 1)
        out, _ = self.local_embedding(x, channels)
        out = out.view(nb, nc, nc - 1, -1)

        out = torch.transpose(out, 2, 3) @ (
            loc_env_a @ (torch.transpose(loc_env_a, 2, 3) @ out[..., : self.axis_neuron])
        )
        return out.view(nb, nc, -1)

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

        esp_layers: List[nn.Module] = [
            ChannelDense(self.n_types, 1, self.neuron[0], bias=True, activation=True)
        ]
        for i in range(len(self.neuron) - 1):
            esp_layers.append(
                ChannelDense(
                    self.n_types,
                    self.neuron[i],
                    self.neuron[i + 1],
                    bias=True,
                    activation=True,
                    residual=True,
                )
            )
        self.esp_embedding = ChannelStack(esp_layers)

        efield_layers: List[nn.Module] = [
            ChannelDense(self.n_types * self.n_types, 1, self.neuron[0], bias=True, activation=True)
        ]
        for i in range(len(self.neuron) - 1):
            efield_layers.append(
                ChannelDense(
                    self.n_types * self.n_types,
                    self.neuron[i],
                    self.neuron[i + 1],
                    bias=True,
                    activation=True,
                    residual=True,
                )
            )
        self.efield_embedding = ChannelStack(efield_layers)

    def forward(
        self,
        atom_coords: Tensor,
        atom_types: Tensor,
        charge_coords: Tensor,
        charges: Tensor,
        real_mask_bool: Tensor,
    ) -> Tensor:
        nb, nc, _ = atom_coords.size()
        _, loc_env_a = _local_environment(atom_coords, dist_eps=self.dist_eps)

        esp_scalar, efield = _esp_efield_masked(
            atom_coords, charge_coords, charges, real_mask_bool, dist_eps=self.dist_eps
        )

        esp_out, _ = self.esp_embedding(esp_scalar.unsqueeze(-1), atom_types)

        proj = torch.bmm(
            loc_env_a.view(-1, nc - 1, 3),
            efield.view(-1, 3, 1),
        ).view(nb, nc, nc - 1)

        neighbor_types = atom_types.repeat(nc).view(nc, nc)
        mask = ~torch.eye(nc, dtype=torch.bool, device=atom_coords.device)
        neighbor_types = torch.masked_select(neighbor_types, mask).view(nc, nc - 1)
        pair_idx = (atom_types * self.n_types).unsqueeze(-1) + neighbor_types
        channels = pair_idx.reshape(-1).long()

        x = proj.view(nb, -1, 1)
        out, _ = self.efield_embedding(x, channels)
        out = out.view(nb, nc, nc - 1, -1)
        out = torch.transpose(out, 2, 3) @ out[..., : self.axis_neuron]
        out = out.view(nb, nc, -1) * 2.0

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

        layers: List[nn.Module] = [
            ChannelDense(self.n_types, in_features, self.neuron[0], activation=True)
        ]
        for i in range(len(self.neuron) - 1):
            layers.append(
                ChannelDense(
                    self.n_types,
                    self.neuron[i],
                    self.neuron[i + 1],
                    activation=True,
                    residual=True,
                )
            )
        layers.append(ChannelDense(self.n_types, self.neuron[-1], 1))
        self.fitting_net = ChannelStack(layers)

    def forward(self, feats: Tensor, atom_types: Tensor) -> Tensor:
        y, _ = self.fitting_net(feats, atom_types)
        return y


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DeepPotMMV0(nn.Module):
    """
    Strict 0.x-style model.

    Assumptions
    -----------
    - variable_qm must be false
    - no QM padding is allowed
    - every frame must have the same real QM atom count
    - every frame must have the same QM atom order
    - every batch must have the same atom_types layout across samples
    """

    INPUT_KEYS: Tuple[str, ...] = INPUT_KEYS
    PREP_INPUT_KEYS: Tuple[str, ...] = PREP_INPUT_KEYS
    PREP_OUTPUT_KEYS: Tuple[str, ...] = PREP_OUTPUT_KEYS
    REQUIRED_KEYS: Tuple[str, ...] = INPUT_KEYS
    OUTPUT_KEYS: Tuple[str, ...] = OUTPUT_KEYS
    ALLOWED_LOSS_KEYS: Tuple[str, ...] = ALLOWED_LOSS_KEYS

    SUPPORTS_VARIABLE_QM = False
    SUPPORTS_QM_PADDING = False

    _warned_mixed_qm = False

    @classmethod
    def check_required_keys(cls, frame: Dict[str, Tensor]) -> None:
        missing = [k for k in cls.REQUIRED_KEYS if k not in frame]
        if missing:
            raise KeyError(
                f"[DeepPotMMV0] missing required frame keys: {missing}\n"
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

        trcfg = cfg.get("trainer", {}) or {}
        if bool(trcfg.get("variable_qm", False)):
            raise ValueError(
                "dpmm_v0 requires trainer.variable_qm = false. "
                "This model assumes fixed QM size/order with no variable-QM batching."
            )

        m = cfg.get("model", {})
        dcfg = m.get("descriptor", {})
        ecfg = m.get("esp", {})
        fcfg = m.get("fitting_net", {})
        tcfg = m.get("atom_types", {})

        self.max_qm = int(m.get("max_qm", 0))
        if self.max_qm <= 0:
            raise ValueError("dpmm_v0 requires model.max_qm > 0.")

        self.n_types = int(tcfg.get("n_types", dcfg.get("n_types", 6)))
        if self.n_types <= 0:
            raise ValueError("model.atom_types.n_types must be > 0.")

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

    def _check_frame_constraints(self, frame: Dict[str, Any]) -> None:
        if Keys.QM_COORDS in frame:
            q = frame[Keys.QM_COORDS]
            if isinstance(q, torch.Tensor) and q.dim() >= 2:
                if int(q.shape[-2]) != self.max_qm:
                    raise ValueError(
                        f"dpmm_v0 requires fixed QM length equal to model.max_qm={self.max_qm}, "
                        f"but got qm_coords shape {tuple(q.shape)}."
                    )

        if N_QM_KEY in frame:
            n_qm = frame[N_QM_KEY]
            if isinstance(n_qm, torch.Tensor):
                vals = n_qm.reshape(-1).to(torch.long)
                unique_vals = torch.unique(vals)
                if unique_vals.numel() != 1:
                    raise ValueError(
                        f"dpmm_v0 requires fixed QM count. Found varying {N_QM_KEY} values: "
                        f"{unique_vals.tolist()}"
                    )
                n_val = int(unique_vals[0].item())
                if n_val != self.max_qm:
                    raise ValueError(
                        f"dpmm_v0 requires {N_QM_KEY} == model.max_qm == {self.max_qm}, "
                        f"but got {n_val}."
                    )

        if QM_TYPE_KEY in frame:
            qm_type = frame[QM_TYPE_KEY]
            if isinstance(qm_type, torch.Tensor):
                vals = qm_type.reshape(-1)
                if torch.any(vals <= 0.5):
                    raise ValueError(
                        "dpmm_v0 does not allow padded QM atoms. "
                        f"Found non-real entries in {QM_TYPE_KEY}."
                    )

        if Keys.QM_Z in frame:
            qm_z = frame[Keys.QM_Z]
            if isinstance(qm_z, torch.Tensor):
                vals = qm_z.reshape(-1).to(torch.long)
                if torch.any(vals == 0):
                    raise ValueError(
                        "dpmm_v0 does not allow padded QM atoms. Found zeros in Keys.QM_Z."
                    )

    def _select_atom_types(self, atom_types: Tensor) -> Tensor:
        if atom_types.dim() == 1:
            return atom_types

        ref = atom_types[0]
        if not torch.jit.is_scripting():
            if atom_types.shape[0] > 1:
                same = bool(torch.all(atom_types.eq(ref)).item())
                if (not same) and (not type(self)._warned_mixed_qm):
                    warnings.warn(
                        "dpmm_v0 expects every batch to have the same QM atom count, "
                        "the same QM ordering, and the same atom_types layout across samples. "
                        "This batch appears heterogeneous. Results may be unreliable.",
                        stacklevel=2,
                    )
                    type(self)._warned_mixed_qm = True
        return ref

    def forward_from_frame(self, frame: Dict[str, Any]) -> Dict[str, Tensor]:
        self.check_required_keys(frame)
        self._check_frame_constraints(frame)
        return self.forward(
            qm_coords=frame[Keys.QM_COORDS],
            atom_types=frame[Keys.ATOM_TYPES],
            mm_coords=frame[Keys.MM_COORDS],
            mm_Q=frame[Keys.MM_Q],
            mm_type=frame[Keys.MM_TYPE],
        )

    def forward(
        self,
        qm_coords: Tensor,
        atom_types: Tensor,
        mm_coords: Tensor,
        mm_Q: Tensor,
        mm_type: Tensor,
    ) -> Dict[str, Tensor]:
        if qm_coords.dim() != 3:
            raise ValueError(f"qm_coords must be (B, Nq, 3), got shape {tuple(qm_coords.shape)}.")
        if int(qm_coords.shape[1]) != self.max_qm:
            raise ValueError(
                f"dpmm_v0 requires fixed QM length equal to model.max_qm={self.max_qm}, "
                f"but got qm_coords shape {tuple(qm_coords.shape)}."
            )

        qm_coords = qm_coords.requires_grad_(True)
        mm_coords = mm_coords.requires_grad_(True)
        mm_Q = mm_Q.requires_grad_(True)

        real_mask_bool = (mm_type > 0)
        mm_mask = real_mask_bool.to(mm_Q.dtype)
        q_eff = mm_Q * mm_mask

        at0 = self._select_atom_types(atom_types)

        desc = self.descriptor(qm_coords, at0)
        esp = self.esp(qm_coords, at0, mm_coords, q_eff, real_mask_bool)
        feats = torch.cat((desc, esp), dim=-1)

        atom_E = self.fitting_net(feats, at0)
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

        mm_mask3 = mm_mask.unsqueeze(-1)
        g_r_masked = g_r * mm_mask3

        mm_esp = g_qeff * mm_mask

        valid = real_mask_bool & (mm_Q.abs() > 0)
        safe_q = torch.where(valid, mm_Q, torch.ones_like(mm_Q))
        mm_esp_grad = (g_r / safe_q.unsqueeze(-1)) * valid.unsqueeze(-1).to(g_r.dtype)

        return {
            "energy": energy,
            "dE": energy,
            "dE_dm": energy,

            "qm_grad": g_qm,
            "qm_grad_high": g_qm,
            "qm_grad_low": g_qm,
            "qm_dgrad": g_qm,

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


register_model("dpmm_v0", DeepPotMMV0)