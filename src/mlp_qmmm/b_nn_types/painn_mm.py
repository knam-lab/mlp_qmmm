# src/mlp_qmmm/b_nn_types/dpmm_equ.py
from __future__ import annotations

from typing import Sequence, Tuple, Dict, Optional, List
import math

import torch
import torch.nn as nn
from torch import Tensor


# ============================================================
# Physics helper
# ============================================================
def _coulomb_factor() -> float:
    # Hartree * Bohr in eV·Å units
    return 27.2114 * 0.529177249


def _esp_efield_masked(
    atom_coords: Tensor,      # (B,Nq,3)
    charge_coords: Tensor,    # (B,Nm,3)
    charges: Tensor,          # (B,Nm) (pads may be zero)
    real_mask_bool: Tensor    # (B,Nm) True real / False pad
) -> Tuple[Tensor, Tensor]:
    """
    Electrostatic potential phi and electric field E on QM atoms from MM charges.

    Returns:
      phi:   (B,Nq)    in eV/e
      Evec:  (B,Nq,3)  in eV/(e·Å)
    """
    B, Nq, _ = atom_coords.shape
    Nm = charge_coords.shape[1]

    rij = atom_coords[:, :, None, :] - charge_coords[:, None, :, :]     # (B,Nq,Nm,3)
    dij = torch.linalg.norm(rij, dim=-1)                                # (B,Nq,Nm)

    pair_mask = real_mask_bool[:, None, :].expand(B, Nq, Nm)

    eps = 1e-12
    d_safe = dij.clamp_min(eps)
    inv_d_all = 1.0 / d_safe
    inv_d3_all = inv_d_all * inv_d_all * inv_d_all

    zeros = torch.zeros_like(dij)
    inv_d  = torch.where(pair_mask, inv_d_all, zeros)
    inv_d3 = torch.where(pair_mask, inv_d3_all, zeros)

    factor = _coulomb_factor()
    phi = torch.sum(charges[:, None, :] * inv_d, dim=-1) * factor
    Evec = torch.sum(charges[:, None, :, None] * rij * inv_d3[..., None], dim=-2) * factor
    return phi, Evec


# ============================================================
# v0.1-style type-conditioned Dense blocks
# ============================================================
class Sequential(nn.Sequential):
    def forward(self, input: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x, channels = input
        for m in self:
            x, channels = m((x, channels))
        return x, channels


class Dense(nn.Module):
    """
    Per-channel (type-conditioned) linear:
      channels: (L,) long, selecting W[channel]
      x: (B,L,Cin) -> y: (B,L,Cout)
    """
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
            y = torch.tanh(y)  # keep v0.1 behavior (TorchScript-stable)

        if self._res_mode == 1:
            y = y + x
        elif self._res_mode == 2:
            y = y + torch.cat([x, x], dim=-1)

        return y, channels


def _assert_batch_atom_types_constant(atom_types: Tensor) -> Tensor:
    """
    v0.1 assumption: batch has same QM composition/order => same atom_types per frame.
    Returns a (N,) vector used as channels.
    """
    if atom_types.dim() == 1:
        return atom_types.long()
    ref = atom_types[0].long()
    if not torch.all(atom_types.long().eq(ref)):
        raise ValueError("Batch contains mixed atom_types layouts; bucket by composition/order.")
    return ref


# ============================================================
# Radial basis + cutoff (TorchScript-stable)
# ============================================================
class CosineCutoff(nn.Module):
    def __init__(self, rc: float):
        super().__init__()
        self.rc = float(rc)

    def forward(self, r: Tensor) -> Tensor:
        x = (r / self.rc).clamp(0.0, 1.0)
        return 0.5 * (torch.cos(math.pi * x) + 1.0) * (r <= self.rc).to(r.dtype)


class GaussianRBF(nn.Module):
    """
    Gaussian radial basis (TorchScript-stable).
    """
    def __init__(self, n_rbf: int, rc: float, *, gamma: Optional[float] = None):
        super().__init__()
        self.n_rbf = int(n_rbf)
        self.rc = float(rc)
        centers = torch.linspace(0.0, self.rc, steps=self.n_rbf, dtype=torch.float32)
        self.register_buffer("centers", centers, persistent=False)
        if gamma is None:
            delta = self.rc / max(self.n_rbf - 1, 1)
            gamma = 1.0 / (delta * delta + 1e-12)
        self.gamma = float(gamma)

    def forward(self, r: Tensor) -> Tensor:
        # Avoid .to(dtype, device) chaining in TorchScript-sensitive ways
        centers = self.centers.to(r.device).type_as(r)  # (n_rbf,)
        diff = r[..., None] - centers
        return torch.exp(-self.gamma * diff * diff)


class BesselRBF(nn.Module):
    """
    Sine/Bessel (j0-like) radial basis (TorchScript-stable).

      phi_n(r) = sqrt(2/rc) * sin(n*pi*r/rc) / r

    r->0 limit:
      phi_n(0) = sqrt(2/rc) * (n*pi/rc)

    Output: (B,N,N,n_rbf)
    """
    def __init__(self, n_rbf: int, rc: float):
        super().__init__()
        self.n_rbf = int(n_rbf)
        self.rc = float(rc)

        n = torch.arange(1, self.n_rbf + 1, dtype=torch.float32)  # (n_rbf,)
        self.register_buffer("n", n, persistent=False)
        self.register_buffer("pi", torch.tensor(math.pi, dtype=torch.float32), persistent=False)
        self.norm = float(math.sqrt(2.0 / self.rc))

    def forward(self, r: Tensor) -> Tensor:
        eps = 1e-12
        r_safe = r.clamp_min(eps)

        n = self.n.to(r.device).type_as(r)    # (n_rbf,)
        pi = self.pi.to(r.device).type_as(r)  # scalar

        x = (r[..., None] * (n * pi) / self.rc)   # (B,N,N,n_rbf)
        out = torch.sin(x) / r_safe[..., None]    # (B,N,N,n_rbf)

        limit = (n * pi / self.rc)  # (n_rbf,)
        out = torch.where((r > eps)[..., None], out, limit[None, None, None, :])

        return out * self.norm


# ============================================================
# Equivariant message passing (PAiNN-ish) layer
#   - scalar: (B,N,Cs)
#   - vector: (B,N,3,Cv)
# ============================================================
class EquivMPNNLayer(nn.Module):
    def __init__(
        self,
        *,
        Cs: int,
        Cv: int,
        Er: int,
        msg_hidden: Sequence[int],
        upd_hidden: Sequence[int],
        act: str = "tanh",
    ):
        super().__init__()
        self.Cs = int(Cs)
        self.Cv = int(Cv)
        self.Er = int(Er)

        inv_dim = 2 * self.Cs + self.Er + 3  # [h_i, h_j, eij, c1,c2,c3]
        self.phi_s = self._mlp(inv_dim, self.Cs, msg_hidden, act=act)
        self.phi_ab = self._mlp(inv_dim, 2 * self.Cv, msg_hidden, act=act)

        upd_in = 2 * self.Cs + 3  # [h_i, M_s, |v_i|, |M_v|, <v_i,M_v>]
        self.U_s = self._mlp(upd_in, self.Cs, upd_hidden, act=act)
        self.gate = self._mlp(upd_in, self.Cv, upd_hidden, act=act)

    @staticmethod
    def _mlp(in_dim: int, out_dim: int, hidden: Sequence[int], *, act: str) -> nn.Module:
        layers: List[nn.Module] = []
        dims = [int(in_dim)] + [int(x) for x in hidden] + [int(out_dim)]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if act == "silu":
                    layers.append(nn.SiLU())
                elif act == "relu":
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    @staticmethod
    def _vec_norm(v: Tensor) -> Tensor:
        mags = (v * v).sum(dim=2).sum(dim=2).clamp_min(1e-12)
        return torch.sqrt(mags).unsqueeze(-1)

    @staticmethod
    def _vec_dot(v: Tensor, w: Tensor) -> Tensor:
        dot = (v * w).sum(dim=2).sum(dim=2)
        return dot.unsqueeze(-1)

    def forward(
        self,
        h_s: Tensor,          # (B,N,Cs)
        h_v: Tensor,          # (B,N,3,Cv)
        rhat: Tensor,         # (B,N,N,3)
        eij: Tensor,          # (B,N,N,Er)
        pair_mask: Tensor,    # (B,N,N) bool
    ) -> Tuple[Tensor, Tensor]:
        B, N, _ = h_s.shape

        hi = h_s[:, :, None, :].expand(B, N, N, self.Cs)
        hj = h_s[:, None, :, :].expand(B, N, N, self.Cs)

        vi = h_v[:, :, None, :, :].expand(B, N, N, 3, self.Cv)
        vj = h_v[:, None, :, :, :].expand(B, N, N, 3, self.Cv)

        c1 = (vi * vj).sum(dim=(3, 4)).unsqueeze(-1)  # <v_i, v_j>
        c2 = (vi * rhat[..., None]).sum(dim=3).sum(dim=3, keepdim=True)  # <v_i, rhat>
        c3 = (vj * rhat[..., None]).sum(dim=3).sum(dim=3, keepdim=True)  # <v_j, rhat>

        inv = torch.cat([hi, hj, eij, c1, c2, c3], dim=-1)

        m_s = self.phi_s(inv) * pair_mask[..., None].to(h_s.dtype)
        M_s = m_s.sum(dim=2)

        ab = self.phi_ab(inv) * pair_mask[..., None].to(h_s.dtype)
        alpha, beta = ab[..., : self.Cv], ab[..., self.Cv :]

        m_v = alpha[..., None, :] * vj + beta[..., None, :] * rhat[..., None]
        M_v = m_v.sum(dim=2)

        nv = self._vec_norm(h_v)
        nMv = self._vec_norm(M_v)
        dot = self._vec_dot(h_v, M_v)

        upd = torch.cat([h_s, M_s, nv, nMv, dot], dim=-1)
        dh = self.U_s(upd)
        g = torch.tanh(self.gate(upd))

        h_s_new = h_s + dh
        h_v_new = h_v + g[:, :, None, :] * M_v
        return h_s_new, h_v_new


# ============================================================
# Descriptor (graph + message passing)
# ============================================================
class EquivDescriptor(nn.Module):
    def __init__(
        self,
        *,
        Cs: int,
        Cv: int,
        n_layers: int,
        rc: float,
        kmax: int,
        n_rbf: int,
        msg_hidden: Sequence[int],
        upd_hidden: Sequence[int],
        act: str,
        radial_basis: str = "gaussian",  # "gaussian" | "bessel"
        rbf_gamma: Optional[float] = None,
    ):
        super().__init__()
        self.Cs = int(Cs)
        self.Cv = int(Cv)
        self.n_layers = int(n_layers)
        self.rc = float(rc)
        self.kmax = int(kmax)
        self.n_rbf = int(n_rbf)

        self.cutoff = CosineCutoff(self.rc)

        rb = str(radial_basis).lower()
        # Only used in __init__; no dynamic switching inside forward (TorchScript-friendly)
        if rb == "bessel":
            self.rbf = BesselRBF(self.n_rbf, self.rc)
        else:
            self.rbf = GaussianRBF(self.n_rbf, self.rc, gamma=rbf_gamma)

        self.layers = nn.ModuleList([
            EquivMPNNLayer(
                Cs=self.Cs, Cv=self.Cv, Er=self.n_rbf,
                msg_hidden=msg_hidden, upd_hidden=upd_hidden, act=act
            )
            for _ in range(self.n_layers)
        ])

    def _build_pairs(self, qm_coords: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, N, _ = qm_coords.shape
        rij = qm_coords[:, :, None, :] - qm_coords[:, None, :, :]  # (B,N,N,3)
        r = torch.linalg.norm(rij, dim=-1)                         # (B,N,N)

        eye = torch.eye(N, device=qm_coords.device, dtype=torch.bool)[None, :, :]
        base = (~eye) & (r <= self.rc)

        if self.kmax and self.kmax > 0:
            inf = torch.full_like(r, float("inf"))
            r_eff = torch.where(base, r, inf)

            k = min(self.kmax, max(N - 1, 1))
            vals, idx = torch.topk(r_eff, k=k, dim=-1, largest=False)
            finite = torch.isfinite(vals)

            pair_mask_f = torch.zeros_like(r, dtype=torch.float32)
            src = finite.to(pair_mask_f.dtype)
            pair_mask_f.scatter_(-1, idx, src)
            pair_mask = (pair_mask_f > 0.5) & base
        else:
            pair_mask = base

        eps = 1e-12
        r_safe = r.clamp_min(eps)
        rhat = rij / r_safe[..., None]
        rhat = torch.where(pair_mask[..., None], rhat, torch.zeros_like(rhat))
        return r, rhat, pair_mask

    def _radial_embed(self, r: Tensor) -> Tensor:
        fc = self.cutoff(r)
        rb = self.rbf(r)
        return rb * fc[..., None]  # (B,N,N,Er)

    def forward(self, qm_coords: Tensor, h_s: Tensor, h_v: Tensor) -> Tuple[Tensor, Tensor]:
        r, rhat, pair_mask = self._build_pairs(qm_coords)
        eij = self._radial_embed(r)

        for layer in self.layers:
            h_s, h_v = layer(h_s, h_v, rhat, eij, pair_mask)

        return h_s, h_v


# ============================================================
# ESP feature injector (TorchScript-safe) + optional "heads" for fitting
# ============================================================
class ESPInjector(nn.Module):
    """
    Always returns 5 tensors (TorchScript-safe; no None):
      h_s:      (B,N,Cs)
      h_v:      (B,N,3,Cv)
      phi_feat: (B,N,Cs_phi)  (zeros if disabled)
      Enorm:    (B,N,1)       (zeros if disabled)
      Evec:     (B,N,3)       (zeros if disabled)
    """
    def __init__(self, *, Cs: int, Cv: int, esp_cfg: dict, act: str):
        super().__init__()
        self.Cs = int(Cs)
        self.Cv = int(Cv)

        self.use_phi = bool(esp_cfg.get("use_phi", True))
        self.use_E = bool(esp_cfg.get("use_E", True))
        self.inject_E_into_v = bool(esp_cfg.get("inject_E_into_v", True))
        self.add_E_norm_to_s = bool(esp_cfg.get("add_E_norm_to_s", True))

        self.Cs_phi = int(esp_cfg.get("Cs_phi", 32))
        phi_hidden = esp_cfg.get("phi_hidden", [64, 64])
        E_gate_hidden = esp_cfg.get("E_gate_hidden", [64, 64])

        self.phi_mlp = self._mlp(1, self.Cs_phi, phi_hidden, act=act)

        extra_s = 0
        extra_s += self.Cs_phi if self.use_phi else 0
        extra_s += 1 if (self.use_E and self.add_E_norm_to_s) else 0
        self.s_proj = nn.Linear(self.Cs + extra_s, self.Cs)

        self.E_gate = self._mlp(self.Cs + 1, self.Cv, E_gate_hidden, act=act)

    @staticmethod
    def _mlp(in_dim: int, out_dim: int, hidden: Sequence[int], *, act: str) -> nn.Module:
        layers: List[nn.Module] = []
        dims = [int(in_dim)] + [int(x) for x in hidden] + [int(out_dim)]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if act == "silu":
                    layers.append(nn.SiLU())
                elif act == "relu":
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(
        self,
        h_s0: Tensor,            # (B,N,Cs)
        h_v0: Tensor,            # (B,N,3,Cv)
        qm_coords: Tensor,       # (B,N,3)
        mm_coords: Tensor,       # (B,M,3)
        q_eff: Tensor,           # (B,M)
        real_mm: Tensor,         # (B,M) bool
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        B = qm_coords.size(0)
        N = qm_coords.size(1)

        phi_feat = torch.zeros((B, N, self.Cs_phi), device=qm_coords.device, dtype=qm_coords.dtype)
        Enorm = torch.zeros((B, N, 1), device=qm_coords.device, dtype=qm_coords.dtype)
        Evec = torch.zeros((B, N, 3), device=qm_coords.device, dtype=qm_coords.dtype)

        if not (self.use_phi or self.use_E):
            return h_s0, h_v0, phi_feat, Enorm, Evec

        phi, Evec_raw = _esp_efield_masked(qm_coords, mm_coords, q_eff, real_mm)

        extras = torch.jit.annotate(List[Tensor], [])

        if self.use_phi:
            phi_feat = self.phi_mlp(phi.unsqueeze(-1))
            extras.append(phi_feat)

        if self.use_E:
            Evec = Evec_raw
            Enorm = torch.linalg.norm(Evec, dim=-1, keepdim=True).clamp_min(1e-12)
            if self.add_E_norm_to_s:
                extras.append(Enorm)

        if len(extras) > 0:
            h_s = self.s_proj(torch.cat([h_s0] + extras, dim=-1))
        else:
            h_s = h_s0

        h_v = h_v0
        if self.use_E and self.inject_E_into_v:
            gate_in = torch.cat([h_s, Enorm], dim=-1)
            gamma = torch.tanh(self.E_gate(gate_in))
            h_v = h_v + gamma[:, :, None, :] * Evec[:, :, :, None]

        return h_s, h_v, phi_feat, Enorm, Evec


# ============================================================
# Fitting (v0.1 Dense style, type-conditioned on atom type)
# ============================================================
class Fitting(nn.Module):
    def __init__(self, n_types: int, in_features: int, neuron: Sequence[int]):
        super().__init__()
        self.n_types = int(n_types)
        self.neuron = [int(x) for x in neuron]

        layers: List[nn.Module] = [Dense(self.n_types, in_features, self.neuron[0], activation=True)]
        for i in range(len(self.neuron) - 1):
            layers.append(Dense(self.n_types, self.neuron[i], self.neuron[i + 1], activation=True, residual=True))
        layers.append(Dense(self.n_types, self.neuron[-1], 1))
        self.net = Sequential(*layers)

    def forward(self, x: Tensor, channels: Tensor) -> Tensor:
        y, _ = self.net((x, channels))
        return y  # (B,N,1)


# ============================================================
# Main model (v0.1 style) + configurable extra fitting features (TorchScript-safe)
# ============================================================
class PAiNNMMEquivariant(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get("model", {}) or {}
        dcfg = m.get("descriptor", {}) or {}
        ecfg = m.get("esp", {}) or {}
        fcfg = m.get("fitting_net", {}) or {}

        self.n_types = int(m.get("n_types", dcfg.get("n_types", fcfg.get("n_types", 6))))

        Cs = int(dcfg.get("Cs", 128))
        Cv = int(dcfg.get("Cv", 32))
        act = str(dcfg.get("act", "tanh")).lower()

        # Type embedding
        self.type_embed = nn.Embedding(self.n_types, Cs)
        nn.init.normal_(self.type_embed.weight, mean=0.0, std=0.1)

        # ESP injector
        self.esp_injector = ESPInjector(Cs=Cs, Cv=Cv, esp_cfg=ecfg, act=act)

        # Optional gaussian gamma coercion (TorchScript-safe / deterministic)
        gamma_val = dcfg.get("rbf_gamma", None)
        rbf_gamma: Optional[float] = float(gamma_val) if gamma_val is not None else None

        self.descriptor = EquivDescriptor(
            Cs=Cs, Cv=Cv,
            n_layers=int(dcfg.get("n_layers", 3)),
            rc=float(dcfg.get("rc", 6.0)),
            kmax=int(dcfg.get("kmax", 0)),
            n_rbf=int(dcfg.get("n_rbf", 32)),
            msg_hidden=tuple(dcfg.get("msg_hidden", [256, 256])),
            upd_hidden=tuple(dcfg.get("upd_hidden", [256, 256])),
            act=act,
            radial_basis=str(dcfg.get("radial_basis", "gaussian")),
            rbf_gamma=rbf_gamma,
        )

        # ----------------------------
        # Fitting feature toggles
        # ----------------------------
        inv_cfg = (fcfg.get("extra_invariants", {}) or {})
        espf_cfg = (fcfg.get("extra_esp_features", {}) or {})

        self.fit_use_v_norm = bool(inv_cfg.get("use_v_norm", True))
        self.fit_use_v_sum_norm = bool(inv_cfg.get("use_v_sum_norm", False))
        self.fit_use_v2 = bool(inv_cfg.get("use_v2", False))

        self.fit_add_phi_feat = bool(espf_cfg.get("add_phi_feat", False))
        self.fit_add_E_norm = bool(espf_cfg.get("add_E_norm", False))
        self.fit_add_vsum_dot_Ehat = bool(espf_cfg.get("add_vsum_dot_Ehat", False))

        fit_in = Cs
        if self.fit_use_v_norm:
            fit_in += Cv
        if self.fit_use_v_sum_norm:
            fit_in += 1
        if self.fit_use_v2:
            fit_in += 1
        if self.fit_add_phi_feat:
            fit_in += int(self.esp_injector.Cs_phi)
        if self.fit_add_E_norm:
            fit_in += 1
        if self.fit_add_vsum_dot_Ehat:
            fit_in += 1

        self.fitting = Fitting(
            n_types=self.n_types,
            in_features=fit_in,
            neuron=tuple(fcfg.get("neuron", [240, 240, 240])),
        )

    def forward(
        self,
        qm_coords: Tensor,
        atom_types: Tensor,
        mm_coords: Tensor,
        mm_Q: Tensor,
        mm_type: Tensor
    ) -> Dict[str, Tensor]:

        qm_coords = qm_coords.requires_grad_(True)
        mm_coords = mm_coords.requires_grad_(True)
        mm_Q = mm_Q.requires_grad_(True)

        atom_types_l = atom_types.long()
        ch = _assert_batch_atom_types_constant(atom_types_l)

        real_mm = (mm_type > 0)
        mm_mask = real_mm.to(mm_Q.dtype)
        q_eff = mm_Q * mm_mask

        h_s0 = self.type_embed(atom_types_l)  # (B,N,Cs)

        B, Nq, _ = qm_coords.shape
        Cv = self.descriptor.Cv
        h_v0 = torch.zeros((B, Nq, 3, Cv), device=qm_coords.device, dtype=qm_coords.dtype)

        # ESP inject + heads
        h_s, h_v, phi_feat, Enorm, Evec = self.esp_injector(h_s0, h_v0, qm_coords, mm_coords, q_eff, real_mm)

        # Descriptor
        h_s, h_v = self.descriptor(qm_coords, h_s, h_v)

        # Fitting feature build (TorchScript-safe)
        parts = torch.jit.annotate(List[Tensor], [h_s])

        v = h_v  # (B,N,3,Cv)

        if self.fit_use_v_norm:
            v_norm = torch.sqrt((v * v).sum(dim=2).clamp_min(1e-12))  # (B,N,Cv)
            parts.append(v_norm)

        if self.fit_use_v_sum_norm:
            v_sum = v.sum(dim=3)  # (B,N,3)
            v_sum_norm = torch.linalg.norm(v_sum, dim=-1, keepdim=True).clamp_min(1e-12)  # (B,N,1)
            parts.append(v_sum_norm)

        if self.fit_use_v2:
            v2 = (v * v).sum(dim=2).sum(dim=2, keepdim=True)  # (B,N,1)
            parts.append(v2)

        if self.fit_add_phi_feat:
            parts.append(phi_feat)

        if self.fit_add_E_norm:
            parts.append(Enorm)

        if self.fit_add_vsum_dot_Ehat:
            eps = 1e-12
            v_sum = v.sum(dim=3)  # (B,N,3)
            Ehat = Evec / (Enorm.clamp_min(eps))  # safe even if Evec=0
            vsum_dot_Ehat = (v_sum * Ehat).sum(dim=-1, keepdim=True)  # (B,N,1)
            parts.append(vsum_dot_Ehat)

        feats = torch.cat(parts, dim=-1)

        atom_E = self.fitting(feats, ch)  # (B,N,1)
        energy = atom_E.sum(dim=1)        # (B,1)

        retain = bool(self.training)
        create = bool(self.training)
        g_qm_opt, g_qeff_opt, g_R_opt = torch.autograd.grad(
            outputs=[energy.sum()],
            inputs=[qm_coords, q_eff, mm_coords],
            retain_graph=retain,
            create_graph=create,
            allow_unused=True,
        )

        g_qm = g_qm_opt if g_qm_opt is not None else torch.zeros_like(qm_coords)
        g_qeff = g_qeff_opt if g_qeff_opt is not None else torch.zeros_like(q_eff)
        g_R = g_R_opt if g_R_opt is not None else torch.zeros_like(mm_coords)

        g_R_masked = g_R * mm_mask.unsqueeze(-1)

        mm_esp = g_qeff * mm_mask

        valid = real_mm & (mm_Q.abs() > 0)
        safe_q = torch.where(valid, mm_Q, torch.ones_like(mm_Q))
        mm_esp_grad = (g_R / safe_q.unsqueeze(-1)) * valid.unsqueeze(-1).to(g_R.dtype)

        return {
            "energy": energy,
            "dE": energy,

            "qm_grad": g_qm,
            "qm_grad_high": g_qm,
            "qm_grad_low": g_qm,
            "qm_dgrad": g_qm,

            "mm_grad": g_R_masked,
            "mm_grad_high": g_R_masked,
            "mm_grad_low": g_R_masked,
            "mm_dgrad": g_R_masked,

            "mm_esp": mm_esp,
            "mm_espgrad_d": mm_esp_grad,
            "mm_espgrad_high": mm_esp_grad,
            "mm_espgrad_low": mm_esp_grad,
        }


#model:
#  descriptor:
#    rc: 6.0
#    n_rbf: 32
#    radial_basis: gaussian   # gaussian | bessel
#    rbf_gamma: null          # only for gaussian (optional)
