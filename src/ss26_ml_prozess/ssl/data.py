"""
Data loading for the flotation plant CSV.

The file uses European decimal format: comma as decimal separator.
Timestamps are hourly but sensors sample every ~20 s — we reconstruct
sub-minute timestamps by counting rows within each hourly group.

Windowed dataset yields (context, target) tensor pairs for JEPA-style SSL.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


def load_and_parse(
    filepath: str | Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Parse European-decimal CSV with pandas, add a 20 s counter.

    Returns (dataframe with a ``_timestamp`` column, feature-name list).
    """
    df = pd.read_csv(
        filepath,
        decimal=",",  # European decimal: "55,2" → 55.2
        thousands=None,  # no thousands separator
        parse_dates=["date"],
        dayfirst=False,
    )
    # Drop the original date column; we'll reconstruct precise timestamps.
    feature_names = [c for c in df.columns if c != "date"]

    # ── Reconstruct 20 s timestamps within each hourly group ──────────────
    # Rows with the same date (hour) are in chronological order.
    # Group by date, assign a 0-based counter, multiply by 20 s.
    df["_group_idx"] = df.groupby("date").cumcount()
    df["_timestamp"] = df["date"] + pd.to_timedelta(df["_group_idx"] * 20, unit="s")
    df = df.drop(columns=["date", "_group_idx"])

    return df, feature_names


class FlotationCSVDataset(Dataset[tuple[Tensor, Tensor]]):
    """Sliding-window dataset over the flotation time series.

    Each sample is a window of *window_size* consecutive rows.  The first
    ``context_ratio`` of the window is the **context** sub-sequence, the
    remainder is the **target**.  Both tensors are ``(seq_len, num_features)``.
    """

    def __init__(
        self,
        filepath: str | Path,
        window_size: int = 60,
        context_ratio: float = 0.7,
    ) -> None:
        super().__init__()

        df, feature_names = load_and_parse(filepath)
        self.feature_names = feature_names
        self.num_features = len(feature_names)

        # Numeric matrix, float64 at this point
        raw = torch.from_numpy(df[feature_names].to_numpy(dtype="float64"))

        # Normalisation statistics (may be overridden by set_stats later)
        self.mean = raw.mean(dim=0)
        self.std = raw.std(dim=0).clamp_min(1e-8)

        self.data = ((raw - self.mean) / self.std).to(torch.float32)
        self.window_size = window_size
        self.context_len = int(window_size * context_ratio)

        # Expose the reconstructed timestamps for downstream use
        self.timestamps = df["_timestamp"].to_numpy()

    def __len__(self) -> int:
        return max(0, len(self.data) - self.window_size + 1)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        window = self.data[idx : idx + self.window_size]
        ctx = window[: self.context_len]
        tgt = window[self.context_len :]
        return ctx, tgt

    def set_stats(self, mean: Tensor, std: Tensor) -> None:
        self.mean = mean
        self.std = std.clamp_min(1e-8)


def build_dataloaders(
    cfg,
    *,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, FlotationCSVDataset]:
    """Create train / validation DataLoaders from configuration.

    Returns (train_loader, val_loader, dataset).
    """
    full = FlotationCSVDataset(
        cfg.data_path,
        window_size=cfg.window_size,
        context_ratio=cfg.context_ratio,
    )

    val_size = int(len(full) * cfg.val_split)
    train_size = len(full) - val_size

    train_ds, val_ds = random_split(
        full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Normalisation statistics from training split only
    train_indices = train_ds.indices  # type: ignore[attr-defined]
    train_data = full.data[train_indices]
    train_mean = train_data.mean(dim=0)
    train_std = train_data.std(dim=0).clamp_min(1e-8)
    full.set_stats(train_mean, train_std)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader, full
