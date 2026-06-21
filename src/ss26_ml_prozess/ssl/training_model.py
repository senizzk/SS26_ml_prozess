"""LeJEPA training model: TCN encoder + VICReg + LeJEPA slicing-loss SSL.

Each training step:
  1. Draw ``(original, views)`` from the dataset.
  2. Encode original → z_orig, encode every view → z_view.
  3. Predict: predictor(z_view) ← asymmetric predictor on view side only.
  4. Compute total loss = VICReg(z_orig_expanded, z_view_pred) + lejepa_stat,
     where lejepa_stat measures distributional distance via sliced
     univariate tests on the stacked embeddings.

The encoder re-uses the ``ResidualBlock`` / TCN stack from ``model.py``
but drops the supervised linear head — the last timestep's hidden state
is projected into the embedding space by a small MLP head.
"""

from __future__ import annotations

import math
from pathlib import Path

import lejepa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, random_split

from ss26_ml_prozess.ssl.config import LeJEPAConfig
from ss26_ml_prozess.ssl.data import LeJEPADataset
from ss26_ml_prozess.ssl.model import ResidualBlock

univariate_test = lejepa.univariate.EppsPulley(n_points=17)
lejepa_loss_fn = lejepa.multivariate.SlicingUnivariateTest(
    univariate_test=univariate_test, num_slices=1024
)
# ================================================================== #
#  Activation helper                                                  #
# ================================================================== #

_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def _make_act(name: str) -> nn.Module:
    return _ACTIVATIONS[name.lower()]()


# ================================================================== #
#  TCN Encoder                                                        #
# ================================================================== #


class TCNEncoder(nn.Module):
    """TCN backbone + projection head → embedding space.

    Input:  ``(B, F, W)``  — batch of time-series windows.
            F = number of features, W = window length.

    Output: ``(B, D)``  — embedding vector where D = ``embedding_dim``.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: tuple[int, ...],
        embedding_dim: int,
        kernel_size: int = 2,
        act: str = "gelu",
    ) -> None:
        super().__init__()
        # --- TCN backbone (causal convolutions) ---
        backbone_layers: list[nn.Module] = []
        in_ch = num_features
        for i, ch in enumerate(hidden_dims):
            dilation = 2**i
            backbone_layers.append(ResidualBlock(in_ch, ch, kernel_size, dilation))
            in_ch = ch
        self.backbone = nn.Sequential(*backbone_layers)
        backbone_out_dim = hidden_dims[-1] if hidden_dims else num_features

        # --- Projection head: backbone_out_dim → embedding_dim ---
        self.projector = nn.Sequential(
            nn.Linear(backbone_out_dim, embedding_dim),
            _make_act(act),
        )

        self._backbone_out_dim = backbone_out_dim
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, F, W) → (B, D)."""
        h = self.backbone(x)  # (B, C_last, Wloss_fn)
        h = h[:, :, -1]  # (B, C_last) — last timestep
        z = self.projector(h)  # (B, D)
        return z


# ================================================================== #
#  Predictor                                                          #
# ================================================================== #


class Predictor(nn.Module):
    """MLP that maps a view embedding towards the original embedding.

    In LeJEPA the predictor is applied asymmetrically — only on the
    view branch — which prevents the trivial solution where both
    branches collapse to a constant.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
        act: str = "gelu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = embedding_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(_make_act(act))
            in_dim = h
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.mlp(z)


# ================================================================== #
#  VICReg loss                                                        #
# ================================================================== #


class VICRegLoss(nn.Module):
    """VICReg-style loss: invariance + variance + covariance.

    Parameters
    ----------
    inv_weight : float
        Weight for the MSE invariance term.
    var_weight : float
        Weight for the variance regularisation.
    cov_weight : float
        Weight for the covariance (decorrelation) penalty.
    var_gamma : float
        Target standard deviation per embedding dimension (default 1.0).
    """

    def __init__(
        self,
        inv_weight: float = 1.0,
        var_weight: float = 1.0,
        cov_weight: float = 0.04,
        var_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_gamma = var_gamma

    def forward(self, z_orig: Tensor, z_view: Tensor) -> dict[str, Tensor]:
        """Compute the three VICReg terms between two sets of embeddings.

        Parameters
        ----------
        z_orig : Tensor, shape ``(B, D)``
        z_view : Tensor, shape ``(B, D)``

        Returns
        -------
        dict with keys ``"loss"``, ``"inv"``, ``"var"``, ``"cov"``, ``"lejepa"``.
        """
        # --- Invariance: MSE between paired embeddings ---
        inv = F.mse_loss(z_orig, z_view)

        # --- Variance: prevent collapse by keeping std above gamma ---
        std_orig = z_orig.std(dim=0)  # (D,)
        std_view = z_view.std(dim=0)  # (D,)
        var = (
            F.relu(self.var_gamma - std_orig).mean()
            + F.relu(self.var_gamma - std_view).mean()
        ) / 2.0

        # --- Covariance: decorrelate off-diagonal entries ---
        z_orig_c = z_orig - z_orig.mean(dim=0)
        z_view_c = z_view - z_view.mean(dim=0)
        B = z_orig.shape[0]
        cov_orig = (z_orig_c.T @ z_orig_c) / (B - 1)  # (D, D)
        cov_view = (z_view_c.T @ z_view_c) / (B - 1)  # (D, D)

        D = z_orig.shape[1]
        off_diag = torch.ones(D, D, device=z_orig.device) - torch.eye(
            D, device=z_orig.device
        )
        cov = (
            ((cov_orig**2) * off_diag).sum() / (D * (D - 1))
            + ((cov_view**2) * off_diag).sum() / (D * (D - 1))
        ) / 2.0

        # --- LeJEPA slicing test: distributional distance of embeddings ---
        stacked = torch.stack([z_orig, z_view], dim=0)  # (2, N, D)
        lejepa_stat = lejepa_loss_fn(stacked)

        loss = (
            self.inv_weight * inv
            + self.var_weight * var
            + self.cov_weight * cov
            + lejepa_stat
        )
        return {"loss": loss, "inv": inv, "var": var, "cov": cov, "lejepa": lejepa_stat}


# ================================================================== #
#  LeJEPA training wrapper                                            #
# ================================================================== #


class LeJEPATrainer:
    """Wraps encoder, predictor, loss, and optimiser for LeJEPA training.

    Typical usage::

        cfg = LeJEPAConfig()
        trainer = LeJEPATrainer(cfg)
        trainer.fit()
    """

    def __init__(self, cfg: LeJEPAConfig) -> None:
        self.cfg = cfg
        self.device = self._resolve_device()

        # Move lejepa loss module to the same device as tensors.
        lejepa_loss_fn.to(self.device)

        # --- Dataset & loaders ---
        full_dataset = LeJEPADataset(
            filepath=cfg.data_path,
            window_size=cfg.window_size,
            num_views=cfg.num_views,
        )

        val_len = int(len(full_dataset) * cfg.val_split)
        train_len = len(full_dataset) - val_len
        self.train_ds, self.val_ds = random_split(full_dataset, [train_len, val_len])

        # Compute per-channel stats on the training subset for normalisation.
        self._set_dataset_stats()

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        # --- Model components ---
        num_features = full_dataset.num_features
        self.encoder = TCNEncoder(
            num_features=num_features,
            hidden_dims=cfg.encoder_dims[:-1],
            embedding_dim=cfg.encoder_dims[-1],
            kernel_size=2,
            act=cfg.encoder_act,
        ).to(self.device)

        self.predictor = Predictor(
            embedding_dim=cfg.encoder_dims[-1],
            hidden_dims=cfg.predictor_dims,
            act=cfg.predictor_act,
        ).to(self.device)

        self.loss_fn = VICRegLoss(
            inv_weight=cfg.inv_weight,
            var_weight=cfg.var_weight,
            cov_weight=cfg.cov_weight,
            var_gamma=cfg.var_gamma,
        )

        # --- Optimiser (predictor + encoder jointly) ---
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # --- Learning-rate warmup + cosine schedule ---
        self.scheduler = self._build_scheduler()

    # ---------------------------------------------------------------- #
    #  Device / stats helpers                                          #
    # ---------------------------------------------------------------- #

    def _resolve_device(self) -> torch.device:
        if self.cfg.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.cfg.device)

    def _set_dataset_stats(self) -> None:
        """Compute and store per-channel mean/std on the training subset."""
        # random_split wraps the original dataset; access it via .dataset.
        base: LeJEPADataset = self.train_ds.dataset  # type: ignore[union-attr]
        all_data = base.data
        indices = self.train_ds.indices  # type: ignore[attr-defined]
        train_data = all_data[indices]
        mean = torch.from_numpy(train_data.mean(axis=0)).float()
        std = torch.from_numpy(train_data.std(axis=0)).float()
        base.set_stats(mean, std)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Linear warmup → cosine decay."""
        warmup = self.cfg.warmup_steps
        total = self.cfg.max_epochs * len(self.train_loader)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ---------------------------------------------------------------- #
    #  Training step                                                   #
    # ---------------------------------------------------------------- #

    def _prepare_batch(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Move a batch to device and reshape for the TCN.

        Dataset returns a tuple ``(original, views)`` where:
            original: (B, W, F)
            views:    (B, K, W, F)  where K = num_views

        Returns:
            orig:  (B, F, W)  — ready for Conv1d
            views: (B*K, F, W)
        """
        orig, views = batch
        orig = orig.to(self.device)  # (B, W, F)
        views = views.to(self.device)  # (B, K, W, F)

        # Conv1d expects (B, channels, seq_len) = (B, F, W)
        orig = orig.permute(0, 2, 1)  # (B, F, W)
        B, K, W, F = views.shape
        views = views.permute(0, 1, 3, 2).reshape(B * K, F, W)
        return orig, views

    def train_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        """Single training step.

        1. Encode original window  → z_orig  (B, D)
        2. Encode each augmented view → z_view (B·K, D)
        3. Predict: z_view_pred = predictor(z_view)  (B·K, D)
        4. Expand z_orig K times and compute VICReg loss
           between z_orig_expanded and z_view_pred.

        Returns dict of scalar loss values (detached, on CPU).
        """
        self.encoder.train()
        self.predictor.train()

        orig, views = self._prepare_batch(batch)
        K = self.cfg.num_views

        # 1. Encode original
        z_orig = self.encoder(orig)  # (B, D)

        # 2. Encode all views
        z_view = self.encoder(views)  # (B*K, D)

        # 3. Predictor on view side only (asymmetric)
        z_view_pred = self.predictor(z_view)  # (B*K, D)

        # 4. Expand original to match views for pairwise loss
        z_orig_expanded = z_orig.repeat(K, 1)  # (B*K, D)

        # 5. VICReg between expanded original and predicted view
        metrics = self.loss_fn(z_orig_expanded, z_view_pred)

        self.optimizer.zero_grad()
        metrics["loss"].backward()
        self.optimizer.step()
        self.scheduler.step()

        return {k: v.item() for k, v in metrics.items()}

    @torch.no_grad()
    def val_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        """Single validation step (no gradients, no parameter update)."""
        self.encoder.eval()
        self.predictor.eval()

        orig, views = self._prepare_batch(batch)
        K = self.cfg.num_views

        z_orig = self.encoder(orig)
        z_view = self.encoder(views)
        z_view_pred = self.predictor(z_view)
        z_orig_expanded = z_orig.repeat(K, 1)

        metrics = self.loss_fn(z_orig_expanded, z_view_pred)
        return {k: v.item() for k, v in metrics.items()}

    # ---------------------------------------------------------------- #
    #  Full training loop                                              #
    # ---------------------------------------------------------------- #

    def fit(self) -> dict[str, list[float]]:
        """Run the full training loop for ``max_epochs``.

        Returns a dict of per-epoch training losses.
        """
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_inv": [],
            "train_var": [],
            "train_cov": [],
            "train_lejepa": [],
        }
        global_step = 0

        for epoch in range(1, self.cfg.max_epochs + 1):
            epoch_loss = 0.0
            epoch_inv = 0.0
            epoch_var = 0.0
            epoch_cov = 0.0
            epoch_lejepa = 0.0
            n_batches = 0

            for batch in self.train_loader:
                metrics = self.train_step(batch)
                epoch_loss += metrics["loss"]
                epoch_inv += metrics["inv"]
                epoch_var += metrics["var"]
                epoch_cov += metrics["cov"]
                epoch_lejepa += metrics["lejepa"]
                n_batches += 1

                if global_step % self.cfg.log_every == 0:
                    print(
                        f"step {global_step:>5d} | "
                        f"loss {metrics['loss']:.4f}  "
                        f"inv {metrics['inv']:.4f}  "
                        f"var {metrics['var']:.4f}  "
                        f"cov {metrics['cov']:.6f}"
                        f"  lejepa {metrics['lejepa']:.4f}"
                    )
                global_step += 1

            def _avg(total: float, n: int = n_batches) -> float:
                return total / max(1, n)

            history["train_loss"].append(_avg(epoch_loss))
            history["train_inv"].append(_avg(epoch_inv))
            history["train_var"].append(_avg(epoch_var))
            history["train_cov"].append(_avg(epoch_cov))
            history["train_lejepa"].append(_avg(epoch_lejepa))

            print(
                f"Epoch {epoch:>3d}/{self.cfg.max_epochs} | "
                f"loss {_avg(epoch_loss):.4f}  "
                f"inv {_avg(epoch_inv):.4f}  "
                f"var {_avg(epoch_var):.4f}  "
                f"cov {_avg(epoch_cov):.6f}"
                f"  lejepa {_avg(epoch_lejepa):.4f}"
            )

            # --- Validation ---
            if len(self.val_loader) > 0:
                val_metrics = self._validate()
                print(
                    f"           val | "
                    f"loss {val_metrics['loss']:.4f}  "
                    f"inv {val_metrics['inv']:.4f}  "
                    f"var {val_metrics['var']:.4f}  "
                    f"cov {val_metrics['cov']:.6f}"
                    f"  lejepa {val_metrics['lejepa']:.4f}"
                )

            # --- Checkpoint ---
            self._save_checkpoint(epoch)

        return history

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Run one full pass over the validation set."""
        self.encoder.eval()
        self.predictor.eval()

        total = {"loss": 0.0, "inv": 0.0, "var": 0.0, "cov": 0.0, "lejepa": 0.0}
        n = 0
        for batch in self.val_loader:
            metrics = self.val_step(batch)
            for k in total:
                total[k] += metrics[k]
            n += 1
        return {k: v / max(1, n) for k, v in total.items()}

    def _save_checkpoint(self, epoch: int) -> None:
        """Persist model + optimiser state to ``checkpoint_dir``."""
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"lejepa_epoch{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "encoder": self.encoder.state_dict(),
                "predictor": self.predictor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            path,
        )
