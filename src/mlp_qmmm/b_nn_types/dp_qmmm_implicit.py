# mlp_qmmm/b_nn_types/dp_qmmm_implicit.py
from __future__ import annotations
from typing import Sequence, Tuple, Dict, Optional
import math
import torch
import torch.nn as nn
from torch import Tensor


def _coulomb_factor() -> float:
    # Hartree * Bohr in eV·Å units (same as before)
    return 27.2114 * 0.529177249


class Sequential(nn.Sequential):
    def forward(self, input: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x, channels = input
        for m in self:
            x, channels = m((x, channels))
        return x, channels


class Dense(nn.Module):
    def __init__(
        self,
        num_channels: int,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        activation: bool = False,
        residual: bool = False
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.activation = bool(activation)

        self.weight = nn.Parameter(torch.empty(self.num_channels, self.out_features, self.in_features))
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
        x, channels = input  # x: (B,L,Cin), channels: (L,)
        W = self.weight[channels]  # (L,Cout,Cin)
        y = torch.bmm(x.transpose(0, 1), W.transpose(1, 2)).transpose(0, 1)  # (B,L,Cout)
        if self.bias is not None:
            y = y + self.bias[channels]
        if self.activation:
            y = torch.tanh(y)

        if self._res_mode == 1:
            y = y + x
        elif self._res_mode == 2:
            y = y + torch.cat([x, x], dim=-1)

        return y, channels


# ---------- physics helpers ----------

def _local_environment(coords: Tensor) -> Tuple[Tensor, Tensor]:
    """
    coords: (B, N, 3)
    returns:
      loc_env_r: (B, N, N-1)      -> 1/rij
      loc_env_a: (B, N, N-1, 3)   -> rij / r^2
    """
    nb, nc, _ = coords.size()
    rij = coords[:, :, None] - coords[:, None]        # (B,N,N,3)
    dij = torch.linalg.norm(rij, dim=3)               # (B,N,N)

    mask = ~torch.eye(nc, dtype=torch.bool, device=coords.device)
    rij = torch.masked_select(rij, mask.unsqueeze(0).unsqueeze(-1)).view(nb, nc, nc - 1, 3)
    dij = torch.masked_select(dij, mask.unsqueeze(0)).view(nb, nc, nc - 1)

    dij_inv = 1.0 / dij  # caller ensures coords not collapsed

    loc_env_r = dij_inv
    loc_env_a = rij * (dij_inv * dij_inv).unsqueeze(-1)
    return loc_env_r, loc_env_a


def _esp_efield_masked(
    atom_coords: Tensor,      # (B,Nq,3)
    charge_coords: Tensor,    # (B,Nm,3)
    charges: Tensor,          # (B,Nm)   (pads may be zero)
    real_mask_bool: Tensor    # (B,Nm)   (True=real, False=pad)
) -> Tuple[Tensor, Tensor]:
    B, Nq, _ = atom_coords.shape
    Nm = charge_coords.shape[1]
    rij = atom_coords[:, :, None] - charge_coords[:, None]   # (B,Nq,Nm,3)
    dij = torch.linalg.norm(rij, dim=3)                      # (B,Nq,Nm)

    pair_mask = real_mask_bool[:, None, :].expand(B, Nq, Nm)

    eps = 1e-12
    d_safe = dij.clamp_min(eps)
    inv_d_all = 1.0 / d_safe
    inv_d3_all = inv_d_all * inv_d_all * inv_d_all

    zeros = torch.zeros_like(dij)
    inv_d  = torch.where(pair_mask, inv_d_all, zeros)
    inv_d3 = torch.where(pair_mask, inv_d3_all, zeros)

    factor = _coulomb_factor()
    esp = torch.sum(charges[:, None, :] * inv_d, dim=-1) * factor                       # (B,Nq)
    efield = torch.sum(charges[:, None, :, None] * rij * inv_d3[..., None], dim=-2) * factor  # (B,Nq,3)
    return esp, efield


# ---------- descriptors ----------

class Feature(nn.Module):
    def __init__(self, n_types: int = 6, neuron: Sequence[int] = (25, 50, 100), axis_neuron: int = 4):
        super().__init__()
        self.n_types = int(n_types)
        self.neuron = list(neuron)
        self.axis_neuron = int(axis_neuron)

        layers = [Dense(self.n_types * self.n_types, 1, self.neuron[0], activation=True)]
        for i in range(len(self.neuron) - 1):
            layers.append(Dense(self.n_types * self.n_types, self.neuron[i], self.neuron[i + 1], activation=True, residual=True))
        self.local_embedding = Sequential(*layers)

    def forward(self, coords: Tensor, atom_types: Tensor) -> Tensor:
        nb, nc, _ = coords.size()
        loc_env_r, loc_env_a = _local_environment(coords)

        at0 = atom_types[0] if atom_types.dim() == 2 else atom_types   # (N,)
        neighbor_types = at0.repeat(nc).view(nc, nc)
        mask = ~torch.eye(nc, dtype=torch.bool, device=coords.device)
        neighbor_types = torch.masked_select(neighbor_types, mask).view(nc, nc - 1)
        pair_idx = (at0 * self.n_types).unsqueeze(-1) + neighbor_types
        channels = pair_idx.reshape(-1).long()                         # (L=N*(N-1),)

        x = loc_env_r.view(nb, -1, 1)
        out, _ = self.local_embedding((x, channels))
        out = out.view(nb, nc, nc - 1, -1)

        out = torch.transpose(out, 2, 3) @ (loc_env_a @ (torch.transpose(loc_env_a, 2, 3) @ out[..., :self.axis_neuron]))
        out = out.view(nb, nc, -1)
        return out

    @property
    def output_length(self) -> int:
        return self.neuron[-1] * self.axis_neuron


class ElectrostaticPotential(nn.Module):
    def __init__(self, n_types: int = 6, neuron: Sequence[int] = (5, 10, 20), axis_neuron: int = 4):
        super().__init__()
        self.n_types = int(n_types)
        self.neuron = list(neuron)
        self.axis_neuron = int(axis_neuron)

        esp_layers = [Dense(self.n_types, 1, self.neuron[0], bias=True, activation=True)]
        for i in range(len(self.neuron) - 1):
            esp_layers.append(Dense(self.n_types, self.neuron[i], self.neuron[i + 1], bias=True, activation=True, residual=True))
        self.esp_embedding = Sequential(*esp_layers)

        efield_layers = [Dense(self.n_types * self.n_types, 1, self.neuron[0], bias=True, activation=True)]
        for i in range(len(self.neuron) - 1):
            efield_layers.append(Dense(self.n_types * self.n_types, self.neuron[i], self.neuron[i + 1], bias=True, activation=True, residual=True))
        self.efield_embedding = Sequential(*efield_layers)

    def forward(
        self,
        atom_coords: Tensor,
        atom_types: Tensor,
        charge_coords: Tensor,
        charges: Tensor,
        real_mask_bool: Tensor
    ) -> Tensor:
        nb, nc, _ = atom_coords.size()
        _, loc_env_a = _local_environment(atom_coords)

        esp_scalar, efield = _esp_efield_masked(atom_coords, charge_coords, charges, real_mask_bool)

        at0 = atom_types[0] if atom_types.dim() == 2 else atom_types
        esp_out, _ = self.esp_embedding((esp_scalar.unsqueeze(-1), at0))    # (B,N,F0)

        # project efield onto local axes
        proj = torch.bmm(loc_env_a.view(-1, nc - 1, 3), efield.view(-1, 3, 1)).view(nb, nc, nc - 1)

        neighbor_types = at0.repeat(nc).view(nc, nc)
        mask = ~torch.eye(nc, dtype=torch.bool, device=atom_coords.device)
        neighbor_types = torch.masked_select(neighbor_types, mask).view(nc, nc - 1)
        pair_idx = (at0 * self.n_types).unsqueeze(-1) + neighbor_types
        channels = pair_idx.reshape(-1).long()

        x = proj.view(nb, -1, 1)
        out, _ = self.efield_embedding((x, channels))
        out = out.view(nb, nc, nc - 1, -1)
        out = torch.transpose(out, 2, 3) @ out[..., :self.axis_neuron]
        out = out.view(nb, nc, -1) * 2.0

        return torch.cat((esp_out, out), dim=2)

    @property
    def output_length(self) -> int:
        return self.neuron[-1] * (self.axis_neuron + 1)


class Fitting(nn.Module):
    def __init__(self, n_types: int = 6, in_features: int = 0, neuron: Sequence[int] = (240, 240, 240)):
        super().__init__()
        self.n_types = int(n_types)
        self.neuron = list(neuron)

        layers = [Dense(self.n_types, in_features, self.neuron[0], activation=True)]
        for i in range(len(self.neuron) - 1):
            layers.append(Dense(self.n_types, self.neuron[i], self.neuron[i + 1], activation=True, residual=True))
        layers.append(Dense(self.n_types, self.neuron[-1], 1))
        self.fitting_net = Sequential(*layers)

    def forward(self, input: Tuple[Tensor, Tensor]) -> Tensor:
        y, _ = self.fitting_net(input)  # (B, N_qm, 1)
        return y


class DeepPotMM(nn.Module):
    """
    Inputs (ORDER):
      qm_coords:  (B, N_qm, 3)
      atom_types:(B, N_qm) or (N_qm,)
      mm_coords:  (B, N_mm, 3)
      mm_Q:       (B, N_mm)      〈— canonical charge name
      mm_type:    (B, N_mm)  1 = real, 0 = padded

    Returns (keys chosen to match trainer loss names):
      energy            (B,1)         # same scalar used for any energy target (adapter decides what "energy" is)
      dE                (B,1)         # alias of energy for legacy configs

      qm_grad           (B,N_qm,3)
      qm_grad_high      (B,N_qm,3)    # alias of qm_grad
      qm_grad_low       (B,N_qm,3)    # alias of qm_grad
      qm_dgrad          (B,N_qm,3)    # alias of qm_grad

      mm_grad           (B,N_mm,3)    # ∂E/∂R (masked)
      mm_grad_high      (B,N_mm,3)    # alias of mm_grad
      mm_grad_low       (B,N_mm,3)    # alias of mm_grad
      mm_dgrad          (B,N_mm,3)    # alias of mm_grad

      mm_esp            (B,N_mm)      # ∂E/∂q (masked)
      mm_espgrad_d      (B,N_mm,3)    # (∂E/∂R)/q (masked, safe divide)
      mm_espgrad_high   (B,N_mm,3)    # alias of mm_espgrad_d
      mm_espgrad_low    (B,N_mm,3)    # alias of mm_espgrad_d
    """
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get("model", {})
        dcfg = m.get("descriptor", {})
        ecfg = m.get("esp", {})
        fcfg = m.get("fitting_net", {})

        self.descriptor = Feature(
            n_types=int(dcfg.get("n_types", 6)),
            neuron=tuple(dcfg.get("neuron", [25, 50, 100])),
            axis_neuron=int(dcfg.get("axis_neuron", 4)),
        )
        self.esp = ElectrostaticPotential(
            n_types=int(ecfg.get("n_types", 6)),
            neuron=tuple(ecfg.get("neuron", [5, 10, 20])),
            axis_neuron=int(ecfg.get("axis_neuron", 4)),
        )
        self.fitting_net = Fitting(
            n_types=int(fcfg.get("n_types", 6)),
            in_features=self.descriptor.output_length + self.esp.output_length,
            neuron=tuple(fcfg.get("neuron", [240, 240, 240])),
        )

    def forward(
        self,
        qm_coords: Tensor,
        atom_types: Tensor,
        mm_coords: Tensor,
        mm_Q: Tensor,      # <- canonical name now
        mm_type: Tensor
    ) -> Dict[str, Tensor]:

        # require grads on inputs (for autograd-based heads)
        qm_coords = qm_coords.requires_grad_(True)
        mm_coords = mm_coords.requires_grad_(True)
        mm_Q      = mm_Q.requires_grad_(True)

        # mask pads
        real_mask_bool = (mm_type > 0)
        mm_mask = real_mask_bool.to(mm_Q.dtype)
        q_eff   = mm_Q * mm_mask

        # descriptors -> atomic energies -> total energy
        desc  = self.descriptor(qm_coords, atom_types)
        esp   = self.esp(qm_coords, atom_types, mm_coords, q_eff, real_mask_bool)
        feats = torch.cat((desc, esp), dim=-1)
        at0   = atom_types[0] if atom_types.dim() == 2 else atom_types
        atom_E = self.fitting_net((feats, at0))        # (B,Nq,1)
        energy = atom_E.sum(dim=1)                     # (B,1)

        # gradients of scalar energy wrt inputs
        retain = bool(self.training)
        create = bool(self.training)
        g_qm_opt, g_qeff_opt, g_R_opt = torch.autograd.grad(
            outputs=[energy.sum()],
            inputs=[qm_coords, q_eff, mm_coords],
            retain_graph=retain,
            create_graph=create,
            allow_unused=True,
        )

        # fill Nones
        g_qm   = g_qm_opt   if g_qm_opt   is not None else torch.zeros_like(qm_coords)
        g_qeff = g_qeff_opt if g_qeff_opt is not None else torch.zeros_like(q_eff)
        g_R    = g_R_opt    if g_R_opt    is not None else torch.zeros_like(mm_coords)

        # mask padded MM atoms
        mm_mask3 = mm_mask.unsqueeze(-1)
        g_R_masked = g_R * mm_mask3

        # ∂E/∂q (zero on pads)
        mm_esp = g_qeff * mm_mask

        # (∂E/∂R)/q with safe divide & mask
        valid  = real_mask_bool & (mm_Q.abs() > 0)
        safe_q = torch.where(valid, mm_Q, torch.ones_like(mm_Q))
        mm_esp_grad = (g_R / safe_q.unsqueeze(-1)) * valid.unsqueeze(-1).to(g_R.dtype)

        # emit aliases so the trainer can pick any target name
        return {
            # energy
            "energy": energy,
            "dE": energy,  # alias for legacy "dE" configs

            # QM grads
            "qm_grad": g_qm,
            "qm_grad_high": g_qm,
            "qm_grad_low": g_qm,
            "qm_dgrad": g_qm,

            # MM cartesian forces (∂E/∂R)
            "mm_grad": g_R_masked,
            "mm_grad_high": g_R_masked,
            "mm_grad_low": g_R_masked,
            "mm_dgrad": g_R_masked,

            # MM charge/ESP heads
            "mm_esp": mm_esp,                  # ∂E/∂q
            "mm_espgrad_d": mm_esp_grad,       # (∂E/∂R)/q
            "mm_espgrad_high": mm_esp_grad,    # aliases for convenience
            "mm_espgrad_low": mm_esp_grad,
        }


# NOTE: This model assumes every batch has the same QM composition/order.
# If you want a hard check, replace occurrences of:
#   at0 = atom_types[0] if atom_types.dim() == 2 else atom_types
# with the guard below and call it to enforce homogeneity:
#
# def _assert_batch_atom_types_constant(atom_types: Tensor) -> Tensor:
#     if atom_types.dim() == 1:
#         return atom_types
#     ref = atom_types[0]
#     if not torch.all(atom_types.eq(ref)):
#         raise ValueError("Batch contains mixed atom_types layouts; bucket by composition/order.")
#     return ref
