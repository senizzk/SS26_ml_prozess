"""
Data loading for the flotation plant CSV.

The file uses European decimal format: comma as decimal separator.
Timestamps are hourly but sensors sample every ~20 s — we reconstruct
sub-minute timestamps by counting rows within each hourly group.

Windowed dataset yields (context, target) tensor pairs for JEPA-style SSL.
"""

from pathlib import Path

import pandas as pd
import torch
from sympy.physics.units import hour
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


def load_and_parse(
    filepath: str | Path,
) -> pd.DataFrame:
    """Parse European-decimal CSV with pandas, add a 20 s counter."""
    df = pd.read_csv(
        filepath,
        decimal=",",  # European decimal: "55,2" → 55.2
        thousands=None,  # no thousands separator
        parse_dates=["date"],
        dayfirst=False,
    )
    hourly_grouped_df = df.groupby("date").mean()

    return hourly_grouped_df


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

        df = load_and_parse(filepath)
        self.feature_names = [column for column in df.columns if column != "date"]
        self.num_features = len(self.feature_names)

        # Numeric matrix, float64 at this point
        self.data = df[self.feature_names].to_numpy()
        self.window_size = window_size
        self.context_len = int(window_size * context_ratio)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.window_size + 1)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        window = self.data[index : index + self.window_size]
        ctx = window[: self.context_len]
        tgt = window[self.context_len :]
        return ctx, tgt

    def set_stats(self, mean: Tensor, std: Tensor) -> None:
        self.mean = mean
        self.std = std.clamp_min(1e-8)
