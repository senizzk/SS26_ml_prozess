"""Forecasting probe: transfer-learning head on top of a frozen LeJEPA encoder.

The pretrained TCN encoder extracts a fixed representation from each input
window.  A lightweight MLP head is then trained to predict future values of
a chosen sensor column — e.g. ``% Silica Concentrate`` — from that frozen
embedding.  Only the head parameters receive gradients; the encoder weights
are locked to their pretrained values.

Usage
-----
    # Train a probe that forecasts % Silica Concentrate:
    python -m ss26_ml_prozess.ssl.forecasting_probe

    # Override settings via env vars:
    TARGET_COLUMN="% Iron Concentrate" FORECAST_HORIZON=12 python -m ss26_ml_prozess.ssl.forecasting_probe

    # Probe a specific checkpoint:
    CHECKPOINT=path/to/lejepa_epoch050.pt python -m ss26_ml_prozess.ssl.forecasting_probe
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, random_split

from ss26_ml_prozess.ssl.data import load_and_parse
from ss26_ml_prozess.ssl.training_model import TCNEncoder

# ================================================================== #
#  Probe config                                                       #
# ================================================================== #


@dataclass
class ProbeConfig:
    """Configuration for the forecasting probe."""

    # --- Data ---
    data_path: str = "data/MiningProcess_Flotation_Plant_Database.csv"
    target_column: str = "% Silica Concentrate"
    context_len: int = 48  # input timesteps (all features)
    forecast_horizon: int = 6  # how many future steps of *target* to predict
    val_split: float = 0.1
    batch_size: int = 64

    # --- Encoder (loaded from checkpoint) ---
    checkpoint: str = "checkpoints/lejepa_epoch050.pt"
    encoder_dims: tuple = (64, 128)  # must match the checkpoint
    kernel_size: int = 2

    # --- Probe head ---
    head_hidden_dims: tuple = (128, 64)
    head_act: str = "gelu"

    # --- Optimiser ---
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 30
    warmup_steps: int = 200

    # --- Logging / saving ---
    log_every: int = 20
    output_dir: str = "probe_checkpoints"

    # --- Device ---
    device: str = "auto"
    seed: int = 42


# ================================================================== #
#  Forecasting dataset                                                #
# ================================================================== #


class ForecastDataset(torch.utils.data.Dataset):
    """Sliding-window dataset returning (context, target) for forecasting.

    Parameters
    ----------
    filepath : str or Path
        Path to the flotation-plant CSV.
    target_column : str
        Column name whose future values we predict.
    context_len : int
        Number of past timesteps used as input (all features).
    forecast_horizon : int
        Number of future timesteps of *target_column* to predict.
    mean, std : Tensor or None
        Per-channel z-score parameters.  Call ``set_stats`` after
        constructing the dataset (or pass them here).
    """

    def __init__(
        self,
        filepath: str | Path,
        target_column: str,
        context_len: int = 48,
        forecast_horizon: int = 6,
        mean: Tensor | None = None,
        std: Tensor | None = None,
    ) -> None:
        super().__init__()
        df = load_and_parse(filepath, mode="train")
        self.feature_names: list[str] = [c for c in df.columns if c != "date"]
        self.num_features = len(self.feature_names)
        self.target_idx = self.feature_names.index(target_column)
        self.context_len = context_len
        self.forecast_horizon = forecast_horizon

        # Total window = context + forecast_horizon
        self.window_size = context_len + forecast_horizon
        self._raw_data = df[self.feature_names].to_numpy(dtype=np.float32)

        # Normalisation
        self._mean = mean
        self._std = std.clamp_min(1e-8) if std is not None else None

    def set_stats(self, mean: Tensor, std: Tensor) -> None:
        self._mean = mean
        self._std = std.clamp_min(1e-8)

    def __len__(self) -> int:
        return max(0, len(self._raw_data) - self.window_size + 1)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        window = self._raw_data[index : index + self.window_size]  # (W, F)
        window = torch.from_numpy(window)

        if self._mean is not None and self._std is not None:
            window = (window - self._mean) / self._std

        ctx = window[: self.context_len]  # (context_len, F)
        # Target: future values of the chosen column only
        tgt = window[self.context_len :, self.target_idx]  # (forecast_horizon,)

        # Permute context to Conv1d format: (F, W) → (F, context_len)
        ctx = ctx.permute(1, 0)  # (F, context_len)

        return ctx, tgt


# ================================================================== #
#  Probe model                                                         #
# ================================================================== #


class ForecastingProbe(nn.Module):
    """Frozen encoder + trainable forecasting head.

    The encoder is loaded from a LeJEPA checkpoint and its parameters are
    frozen (``requires_grad = False``).  A small MLP maps the encoder's
    embedding to ``forecast_horizon`` scalar predictions for the target
    column.
    """

    def __init__(
        self,
        encoder: TCNEncoder,
        forecast_horizon: int,
        head_hidden_dims: tuple[int, ...] = (128, 64),
        head_act: str = "gelu",
    ) -> None:
        super().__init__()
        # Freeze encoder
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        embedding_dim = encoder.embedding_dim

        # Build head
        layers: list[nn.Module] = []
        in_dim = embedding_dim
        _ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU}
        act_fn = _ACTIVATIONS[head_act.lower()]
        for h in head_hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_fn())
            in_dim = h
        layers.append(nn.Linear(in_dim, forecast_horizon))
        self.head = nn.Sequential(*layers)

        self._forecast_horizon = forecast_horizon

    def forward(self, x: Tensor) -> Tensor:
        """(B, F, W) → (B, forecast_horizon)."""
        with torch.no_grad():
            z = self.encoder(x)  # (B, D) — frozen, no grads
        return self.head(z)  # (B, forecast_horizon)

    @property
    def forecast_horizon(self) -> int:
        return self._forecast_horizon


# ================================================================== #
#  Probe trainer                                                       #
# ================================================================== #


class ProbeTrainer:
    """Training loop for the forecasting probe (encoder frozen)."""

    def __init__(self, cfg: ProbeConfig) -> None:
        self.cfg = cfg
        self.device = self._resolve_device()

        # --- Dataset ---
        full_ds = ForecastDataset(
            filepath=cfg.data_path,
            target_column=cfg.target_column,
            context_len=cfg.context_len,
            forecast_horizon=cfg.forecast_horizon,
        )

        val_len = int(len(full_ds) * cfg.val_split)
        train_len = len(full_ds) - val_len
        self.train_ds, self.val_ds = random_split(full_ds, [train_len, val_len])

        # Compute z-score stats on training set only
        self._set_dataset_stats(full_ds)

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
            num_workers=0,
        )

        # --- Build encoder and load checkpoint ---
        encoder = TCNEncoder(
            num_features=full_ds.num_features,
            hidden_dims=cfg.encoder_dims[:-1],
            embedding_dim=cfg.encoder_dims[-1],
            kernel_size=cfg.kernel_size,
        )
        ckpt = torch.load(cfg.checkpoint, map_location=self.device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.to(self.device)
        print(f"Loaded encoder from {cfg.checkpoint}")

        # --- Probe model ---
        self.model = ForecastingProbe(
            encoder=encoder,
            forecast_horizon=cfg.forecast_horizon,
            head_hidden_dims=cfg.head_hidden_dims,
            head_act=cfg.head_act,
        ).to(self.device)

        # --- Optimiser: only head parameters ---
        head_params = [p for p in self.model.head.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            head_params, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = self._build_scheduler()

        # --- Loss ---
        self.loss_fn = nn.MSELoss()

        # --- Bookkeeping ---
        self._target_idx = full_ds.target_idx
        self._feature_names = full_ds.feature_names
        # Store target std for denormalised RMSE reporting
        if full_ds._std is not None:
            self._target_std = full_ds._std[self._target_idx].item()
        else:
            self._target_std = 1.0

    # ---------------------------------------------------------------- #
    #  Helpers                                                          #
    # ---------------------------------------------------------------- #

    def _resolve_device(self) -> torch.device:
        if self.cfg.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.cfg.device)

    def _set_dataset_stats(self, full_ds: ForecastDataset) -> None:
        indices = self.train_ds.indices  # type: ignore[attr-defined]
        all_data = full_ds._raw_data
        train_data = all_data[indices]
        mean = torch.from_numpy(train_data.mean(axis=0)).float()
        std = torch.from_numpy(train_data.std(axis=0)).float()
        full_ds.set_stats(mean, std)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        warmup = self.cfg.warmup_steps
        total = self.cfg.max_epochs * len(self.train_loader)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ---------------------------------------------------------------- #
    #  Training / validation                                            #
    # ---------------------------------------------------------------- #

    def train_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        ctx, tgt = batch
        ctx = ctx.to(self.device)  # (B, F, context_len)
        tgt = tgt.to(self.device)  # (B, forecast_horizon)

        self.model.train()
        # encoder is frozen; only head has grads
        pred = self.model(ctx)  # (B, forecast_horizon)

        loss = self.loss_fn(pred, tgt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return {"loss": loss.item(), "rmse": loss.sqrt().item()}

    @torch.no_grad()
    def val_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        ctx, tgt = batch
        ctx = ctx.to(self.device)
        tgt = tgt.to(self.device)

        self.model.eval()
        pred = self.model(ctx)
        loss = self.loss_fn(pred, tgt)

        # Denormalised RMSE in original units
        rmse_orig = (loss.sqrt() * self._target_std).item()

        return {"loss": loss.item(), "rmse": loss.sqrt().item(), "rmse_orig": rmse_orig}

    # ---------------------------------------------------------------- #
    #  Full loop                                                        #
    # ---------------------------------------------------------------- #

    def fit(self) -> dict[str, list[float]]:
        """Run the probe training loop. Returns per-epoch history."""
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_rmse": [],
            "val_loss": [],
            "val_rmse": [],
            "val_rmse_orig": [],
        }
        global_step = 0

        print(
            f"\nProbe: forecasting '{self.cfg.target_column}' "
            f"({self.cfg.forecast_horizon} steps ahead) "
            f"using {self.cfg.context_len}-step context\n"
        )

        for epoch in range(1, self.cfg.max_epochs + 1):
            epoch_loss = 0.0
            epoch_rmse = 0.0
            n = 0

            for batch in self.train_loader:
                metrics = self.train_step(batch)
                epoch_loss += metrics["loss"]
                epoch_rmse += metrics["rmse"]
                n += 1

                if global_step % self.cfg.log_every == 0:
                    print(
                        f"step {global_step:>5d} | "
                        f"loss {metrics['loss']:.4f}  "
                        f"rmse {metrics['rmse']:.4f}"
                    )
                global_step += 1

            def _avg(total: float, count: int = n) -> float:
                return total / max(1, count)

            avg_loss = _avg(epoch_loss)
            avg_rmse = _avg(epoch_rmse)
            history["train_loss"].append(avg_loss)
            history["train_rmse"].append(avg_rmse)

            # --- Validation ---
            val_metrics = self._validate()
            history["val_loss"].append(val_metrics["loss"])
            history["val_rmse"].append(val_metrics["rmse"])
            history["val_rmse_orig"].append(val_metrics["rmse_orig"])

            print(
                f"Epoch {epoch:>3d}/{self.cfg.max_epochs} | "
                f"train loss {avg_loss:.4f} rmse {avg_rmse:.4f} | "
                f"val loss {val_metrics['loss']:.4f} "
                f"rmse {val_metrics['rmse']:.4f} "
                f"rmse(orig) {val_metrics['rmse_orig']:.4f}"
            )

            # --- Checkpoint ---
            self._save_checkpoint(epoch)

        return history

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        self.model.eval()
        total = {"loss": 0.0, "rmse": 0.0, "rmse_orig": 0.0}
        n = 0
        for batch in self.val_loader:
            metrics = self.val_step(batch)
            for k in total:
                total[k] += metrics[k]
            n += 1
        return {k: v / max(1, n) for k, v in total.items()}

    def _save_checkpoint(self, epoch: int) -> None:
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"probe_epoch{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "head": self.model.head.state_dict(),
                "encoder_state_dict_keys": list(self.model.encoder.state_dict().keys()),
                "config": {
                    "target_column": self.cfg.target_column,
                    "context_len": self.cfg.context_len,
                    "forecast_horizon": self.cfg.forecast_horizon,
                    "encoder_dims": self.cfg.encoder_dims,
                    "head_hidden_dims": self.cfg.head_hidden_dims,
                },
            },
            path,
        )


# ================================================================== #
#  CLI entry point                                                     #
# ================================================================== #


def main() -> None:
    cfg = ProbeConfig(
        data_path=os.environ.get("DATA_PATH", ProbeConfig.data_path),
        checkpoint=os.environ.get("CHECKPOINT", ProbeConfig.checkpoint),
        target_column=os.environ.get("TARGET_COLUMN", ProbeConfig.target_column),
        forecast_horizon=int(
            os.environ.get("FORECAST_HORIZON", ProbeConfig.forecast_horizon)
        ),
        context_len=int(os.environ.get("CONTEXT_LEN", ProbeConfig.context_len)),
        max_epochs=int(os.environ.get("MAX_EPOCHS_PROBE", ProbeConfig.max_epochs)),
        batch_size=int(os.environ.get("BATCH_SIZE", ProbeConfig.batch_size)),
        lr=float(os.environ.get("LR", ProbeConfig.lr)),
        device=os.environ.get("DEVICE", ProbeConfig.device),
    )

    print("=" * 60)
    print("LeJEPA Forecasting Probe (transfer learning)")
    print("=" * 60)
    print(f"  Checkpoint    : {cfg.checkpoint}")
    print(f"  Target column : {cfg.target_column}")
    print(f"  Context len   : {cfg.context_len} timesteps")
    print(f"  Forecast horiz: {cfg.forecast_horizon} timesteps")
    print(f"  Batch size    : {cfg.batch_size}")
    print(f"  Learning rate : {cfg.lr}")
    print(f"  Max epochs    : {cfg.max_epochs}")
    print(f"  Device        : {cfg.device}")
    print(f"  Seed          : {cfg.seed}")
    print("=" * 60)

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    trainer = ProbeTrainer(cfg)
    history = trainer.fit()

    # --- Summary ---
    print("\nProbe training complete!")
    final = {k: v[-1] for k, v in history.items()}
    print(f"  Final val RMSE (normalised): {final['val_rmse']:.4f}")
    print(f"  Final val RMSE (original)   : {final['val_rmse_orig']:.4f}")

    # --- Save history ---
    history_path = Path(cfg.output_dir) / "probe_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved to {history_path}")


if __name__ == "__main__":
    main()
