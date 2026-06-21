"""Apply a trained forecasting probe to the held-out test set.

Loads the encoder from a LeJEPA pretraining checkpoint and the probe head
from a forecasting probe checkpoint, then runs inference on ``mode="test"``
data.  The same z-score normalisation statistics computed on the training
split during probing are reused here (not recomputed on the test data).

Outputs
-------
- Per-timestep RMSE and MAE (in original units) printed to console.
- A CSV file saved to ``output_dir`` with columns:
  ``timestamp, actual, predicted, horizon_step`` — one row per
  (sample, horizon_step) pair — so you can plot actual vs. predicted
  for each forecast horizon step.
- A JSON file with aggregate metrics.

Usage
-----
    # Use the latest probe checkpoint:
    python -m ss26_ml_prozess.ssl.apply_to_test

    # Specify checkpoints and target:
    PRETRAIN_CKPT=checkpoints/lejepa_epoch050.pt \
    PROBE_CKPT=probe_checkpoints/probe_epoch030.pt \
    TARGET_COLUMN="% Silica Concentrate" \
    python -m ss26_ml_prozess.ssl.apply_to_test
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ss26_ml_prozess.ssl.data import load_and_parse
from ss26_ml_prozess.ssl.training_model import TCNEncoder


# ================================================================== #
#  Test dataset (no augmentation, no target — just context windows)    #
# ================================================================== #


class TestDataset(torch.utils.data.Dataset):
    """Sliding-window test dataset that returns (context, target, timestamp).

    Uses ``mode="test"`` in :func:`load_and_parse` so only the held-out
    portion of the data is loaded.

    Parameters
    ----------
    filepath : str or Path
        Path to the flotation-plant CSV.
    target_column : str
        Column name whose future values we evaluate.
    context_len : int
        Number of past timesteps used as input (all features).
    forecast_horizon : int
        Number of future timesteps of *target_column* to predict.
    mean, std : Tensor
        Per-channel z-score parameters **computed on the training set**.
    """

    def __init__(
        self,
        filepath: str | Path,
        target_column: str,
        context_len: int,
        forecast_horizon: int,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        super().__init__()
        df = load_and_parse(filepath, mode="test")
        self.feature_names: list[str] = [c for c in df.columns if c != "date"]
        self.num_features = len(self.feature_names)
        self.target_idx = self.feature_names.index(target_column)
        self.context_len = context_len
        self.forecast_horizon = forecast_horizon
        self.window_size = context_len + forecast_horizon

        # Preserve timestamps for output
        timestamps = df.index if "date" not in df.columns else df["date"]
        self._timestamps = timestamps

        self._raw_data = df[self.feature_names].to_numpy(dtype=np.float32)
        self._mean = mean
        self._std = std.clamp_min(1e-8)

    def __len__(self) -> int:
        return max(0, len(self._raw_data) - self.window_size + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        window = self._raw_data[index : index + self.window_size]  # (W, F)
        window = torch.from_numpy(window)

        window = (window - self._mean) / self._std

        ctx = window[: self.context_len].permute(1, 0)  # (F, context_len)
        tgt = window[self.context_len :, self.target_idx]  # (forecast_horizon,)

        # Timestamp of the first forecast step
        forecast_start_idx = index + self.context_len
        if forecast_start_idx < len(self._timestamps):
            ts = str(self._timestamps[forecast_start_idx])
        else:
            ts = ""

        return ctx, tgt, ts


# ================================================================== #
#  Reconstruct probe model from checkpoints                            #
# ================================================================== #


def load_probe_model(
    pretrain_ckpt: str | Path,
    probe_ckpt: str | Path,
    num_features: int,
    encoder_dims: tuple[int, ...],
    kernel_size: int,
    forecast_horizon: int,
    head_hidden_dims: tuple[int, ...],
    head_act: str,
    device: torch.device,
) -> nn.Module:
    """Reconstruct ForecastingProbe from two checkpoints.

    Encoder weights come from the LeJEPA pretraining checkpoint;
    head weights come from the probe training checkpoint.
    """
    from ss26_ml_prozess.ssl.forecasting_probe import ForecastingProbe

    # Build encoder skeleton and load pretrained weights
    encoder = TCNEncoder(
        num_features=num_features,
        hidden_dims=encoder_dims[:-1],
        embedding_dim=encoder_dims[-1],
        kernel_size=kernel_size,
    )
    pretrain = torch.load(pretrain_ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(pretrain["encoder"])
    encoder.to(device)

    # Assemble full probe
    model = ForecastingProbe(
        encoder=encoder,
        forecast_horizon=forecast_horizon,
        head_hidden_dims=head_hidden_dims,
        head_act=head_act,
    ).to(device)

    # Load trained head
    probe = torch.load(probe_ckpt, map_location=device, weights_only=False)
    model.head.load_state_dict(probe["head"])
    model.eval()

    return model


# ================================================================== #
#  Evaluation                                                         #
# ================================================================== #


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    target_std: float,
    target_mean: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[dict[str, float], list[dict]]:
    """Run inference and collect per-sample predictions.

    Returns
    -------
    metrics : dict
        Aggregate MSE, RMSE, MAE (both normalised and in original units).
    records : list[dict]
        One dict per (sample, horizon_step) with keys
        ``timestamp``, ``actual``, ``predicted``, ``horizon_step``.
    """
    all_preds = []
    all_actuals = []
    records = []

    for ctx, tgt, ts_batch in dataloader:
        # ts_batch is a tuple of strings from __getitem__
        ctx = ctx.to(device)  # (B, F, context_len)
        tgt = tgt.to(device)  # (B, forecast_horizon)

        pred = model(ctx)  # (B, forecast_horizon)

        all_preds.append(pred.cpu())
        all_actuals.append(tgt.cpu())

        # Build per-step records
        batch_size = ctx.shape[0]
        for b in range(batch_size):
            for h in range(pred.shape[1]):
                actual_val = tgt[b, h].item() * target_std + target_mean
                pred_val = pred[b, h].item() * target_std + target_mean
                records.append(
                    {
                        "timestamp": ts_batch[b] if isinstance(ts_batch, (list, tuple)) else ts_batch,
                        "horizon_step": h + 1,
                        "actual": actual_val,
                        "predicted": pred_val,
                        "error": actual_val - pred_val,
                    }
                )

    preds = torch.cat(all_preds, dim=0)  # (N, H)
    actuals = torch.cat(all_actuals, dim=0)  # (N, H)

    mse = torch.mean((preds - actuals) ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(preds - actuals)).item()

    # Per-horizon-step metrics
    mse_per_h = ((preds - actuals) ** 2).mean(dim=0)  # (H,)
    rmse_per_h = mse_per_h.sqrt()

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "rmse_original": rmse * target_std,
        "mae_original": mae * target_std,
        "mse_normalised": mse,
        "rmse_normalised": rmse,
    }

    # Add per-horizon-step RMSE in original units
    for h in range(preds.shape[1]):
        metrics[f"rmse_h{h+1}_original"] = rmse_per_h[h].item() * target_std

    return metrics, records


# ================================================================== #
#  CLI entry point                                                     #
# ================================================================== #


def main() -> None:
    from ss26_ml_prozess.ssl.forecasting_probe import ProbeConfig

    # Use ProbeConfig defaults as the source of truth for model architecture
    defaults = ProbeConfig()

    pretrain_ckpt = os.environ.get("PRETRAIN_CKPT", defaults.checkpoint)
    probe_ckpt = os.environ.get(
        "PROBE_CKPT", str(Path(defaults.output_dir) / "probe_epoch030.pt")
    )
    data_path = os.environ.get("DATA_PATH", defaults.data_path)
    target_column = os.environ.get("TARGET_COLUMN", defaults.target_column)
    context_len = int(os.environ.get("CONTEXT_LEN", defaults.context_len))
    forecast_horizon = int(os.environ.get("FORECAST_HORIZON", defaults.forecast_horizon))
    output_dir = os.environ.get("OUTPUT_DIR", "test_results")

    device_str = os.environ.get("DEVICE", defaults.device)
    device = torch.device("cuda" if device_str == "auto" and torch.cuda.is_available() else (
        "cuda" if device_str == "cuda" else "cpu"
    ) if device_str == "auto" else device_str)

    print("=" * 60)
    print("Apply Forecasting Probe to Test Set")
    print("=" * 60)
    print(f"  Pretrain ckpt    : {pretrain_ckpt}")
    print(f"  Probe ckpt       : {probe_ckpt}")
    print(f"  Data path        : {data_path}")
    print(f"  Target column    : {target_column}")
    print(f"  Context len      : {context_len}")
    print(f"  Forecast horizon  : {forecast_horizon}")
    print(f"  Device           : {device}")
    print("=" * 60)

    # --- Step 1: Compute normalisation stats from TRAINING data ---
    print("\n[1/4] Computing normalisation stats from training data...")
    train_df = load_and_parse(data_path, mode="train")
    feature_names = [c for c in train_df.columns if c != "date"]
    train_data = train_df[feature_names].to_numpy(dtype=np.float32)
    mean = torch.from_numpy(train_data.mean(axis=0)).float()
    std = torch.from_numpy(train_data.std(axis=0)).float()
    target_idx = feature_names.index(target_column)
    target_std = std[target_idx].item()
    target_mean = mean[target_idx].item()
    num_features = len(feature_names)
    print(f"  Training mean/std computed over {len(train_data)} rows, "
          f"{num_features} features")
    print(f"  Target column '{target_column}' std (original units): "
          f"{target_std:.4f}")
    print(f"  Target column mean (original units): "
          f"{target_mean:.4f}")

    # --- Step 2: Load probe model ---
    print("\n[2/4] Loading probe model...")
    model = load_probe_model(
        pretrain_ckpt=pretrain_ckpt,
        probe_ckpt=probe_ckpt,
        num_features=num_features,
        encoder_dims=defaults.encoder_dims,
        kernel_size=defaults.kernel_size,
        forecast_horizon=forecast_horizon,
        head_hidden_dims=defaults.head_hidden_dims,
        head_act=defaults.head_act,
        device=device,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Encoder frozen params: {frozen:,}")
    print(f"  Head trainable params: {trainable:,}")

    # --- Step 3: Build test dataset ---
    print("\n[3/4] Building test dataset (mode='test')...")
    test_ds = TestDataset(
        filepath=data_path,
        target_column=target_column,
        context_len=context_len,
        forecast_horizon=forecast_horizon,
        mean=mean,
        std=std,
    )
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    print(f"  Test windows: {len(test_ds)}")

    # --- Step 4: Evaluate ---
    print("\n[4/4] Running inference on test set...")
    metrics, records = evaluate(model, test_loader, target_std, target_mean, device)

    # --- Print results ---
    print("\n" + "=" * 60)
    print("Test Set Results")
    print("=" * 60)
    print(f"  MSE  (normalised): {metrics['mse_normalised']:.6f}")
    print(f"  RMSE (normalised): {metrics['rmse_normalised']:.6f}")
    print(f"  RMSE (original)  : {metrics['rmse_original']:.4f}")
    print(f"  MAE  (original)  : {metrics['mae_original']:.4f}")
    print()
    print("Per-horizon-step RMSE (original units):")
    for h in range(forecast_horizon):
        key = f"rmse_h{h+1}_original"
        print(f"  h={h+1:2d}: {metrics[key]:.4f}")

    # --- Save results ---
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save detailed predictions CSV
    df_preds = pd.DataFrame(records)
    csv_path = out / "test_predictions.csv"
    df_preds.to_csv(csv_path, index=False)
    print(f"\n  Predictions saved to {csv_path}")

    # Save aggregate metrics
    json_path = out / "test_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()