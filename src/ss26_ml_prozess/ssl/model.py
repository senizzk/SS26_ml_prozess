"""
LeJEPA — a lightweight Joint-Embedding Predictive Architecture for tabular
time series, inspired by Yann LeCun's JEPA family (I-JEPA / VICReg).

Architecture
------------
1. **Encoder** — MLP applied *per time step*: each reading → embedding.
2. **Pool** — mean over the time axis → single vector per window.
3. **Predictor** — MLP that predicts pooled target embedding from pooled
   context embedding.
4. **Loss** — VICReg-style: invariance (MSE), variance (hinge on std),
   covariance (decorrelate).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── MLP building block ───────────────────────────────────────────────────────

def _mlp(
    dims: tuple[int, ...],
    act: str = "gelu",
    final_act: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_act:
            if act == "gelu":
                layers.append(nn.GELU())
            elif act == "relu":
                layers.append(nn.ReLU(inplace=True))
            else:
                raise ValueError(f"Unknown activation: {act}")
            layers.append(nn.LayerNorm(dims[i + 1], elementwise_affine=False))
    return nn.Sequential(*layers)


# ── Encoder ───────────────────────────────────────────────────────────────────

class TimeStepEncoder(nn.Module):
    """Per-step MLP encoder.

    Input:  (B, T, F)  →  Output: (B, T, D)
    """

    def __init__(self, num_features: int, dims: tuple[int, ...], act: str = "gelu") -> None:
        super().__init__()
        assert dims
        self.net = _mlp((num_features, *dims), act=act)
        self.embed_dim = dims[-1]

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ── Predictor ─────────────────────────────────────────────────────────────────

class Predictor(nn.Module):
    """MLP from pooled context embedding → pooled target embedding."""

    def __init__(self, dims: tuple[int, ...], act: str = "gelu") -> None:
        super().__init__()
        assert len(dims) >= 2
        self.net = _mlp(dims, act=act)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ── VICReg-style loss ─────────────────────────────────────────────────────────

class VICRegLoss(nn.Module):
    """VICReg loss on a batch of embeddings.

    Reference: Bardes et al., ICLR 2022.
    """

    def __init__(
        self,
        inv_weight: float = 1.0,
        var_weight: float = 1.0,
        cov_weight: float = 0.04,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.gamma = gamma

    def forward(self, z_pred: Tensor, z_target: Tensor) -> dict[str, Tensor]:
        B = z_pred.shape[0]

        # Invariance — MSE
        inv_loss = F.mse_loss(z_pred, z_target)

        # Variance — hinge to keep std >= gamma
        std_pred = torch.sqrt(z_pred.var(dim=0, unbiased=False) + 1e-8)
        std_tgt = torch.sqrt(z_target.var(dim=0, unbiased=False) + 1e-8)
        var_loss = (
            F.relu(self.gamma - std_pred).mean()
            + F.relu(self.gamma - std_tgt).mean()
        )

        # Covariance — push off-diagonal toward zero
        def _cov_loss(z: Tensor) -> Tensor:
            zc = z - z.mean(dim=0, keepdim=True)
            cov = (zc.T @ zc) / max(B - 1, 1)  # (D, D)
            off = cov - torch.diag(torch.diag(cov))
            return off.pow(2).sum() / cov.shape[0]

        cov_loss = _cov_loss(z_pred) + _cov_loss(z_target)

        total = (
            self.inv_weight * inv_loss
            + self.var_weight * var_loss
            + self.cov_weight * cov_loss
        )

        return {
            "loss": total,
            "inv": inv_loss.detach(),
            "var": var_loss.detach(),
            "cov": cov_loss.detach(),
        }


# ── LeJEPA model ──────────────────────────────────────────────────────────────

class LeJEPA(nn.Module):
    """Lightweight Joint-Embedding Predictive Architecture."""

    def __init__(
        self,
        num_features: int,
        *,
        encoder_dims: tuple[int, ...] = (64, 128),
        encoder_act: str = "gelu",
        predictor_dims: tuple[int, ...] = (128, 64, 64),
        predictor_act: str = "gelu",
        inv_weight: float = 1.0,
        var_weight: float = 1.0,
        cov_weight: float = 0.04,
        var_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        embed_dim = encoder_dims[-1]
        assert predictor_dims[-1] == embed_dim

        self.encoder = TimeStepEncoder(num_features, encoder_dims, act=encoder_act)
        self.predictor = Predictor(predictor_dims, act=predictor_act)
        self.loss_fn = VICRegLoss(
            inv_weight=inv_weight,
            var_weight=var_weight,
            cov_weight=cov_weight,
            gamma=var_gamma,
        )
        self.embed_dim = embed_dim

    def encode_context(self, ctx: Tensor) -> Tensor:
        h = self.encoder(ctx)
        return h.mean(dim=1)

    def encode_target(self, tgt: Tensor) -> Tensor:
        h = self.encoder(tgt)
        return h.mean(dim=1)

    def forward(self, ctx: Tensor, tgt: Tensor) -> dict[str, Tensor]:
        z_ctx = self.encode_context(ctx)
        z_tgt = self.encode_target(tgt)
        z_pred = self.predictor(z_ctx)
        return self.loss_fn(z_pred, z_tgt)

    @torch.no_grad()
    def predict(self, ctx: Tensor) -> Tensor:
        return self.predictor(self.encode_context(ctx))

    def embed(self, x: Tensor) -> Tensor:
        return self.encoder(x)
