"""
Data loading for the flotation plant CSV.

The file uses European decimal format: comma as decimal separator.
Timestamps are hourly but sensors sample every ~20 s — we reconstruct
sub-minute timestamps by counting rows within each hourly group.

LeJEPA dataset yields ``(original, views)`` pairs where ``original`` is the
clean window and ``views`` contains *num_views* augmented copies.  Each
augmented view is produced by randomly composing four perturbations:

  1. **Temporal masking**  — zero out random contiguous time blocks
  2. **Feature masking**   — zero out random sensor channels (column drop)
  3. **Gaussian noise**    — additive noise scaled per-channel
  4. **Random scaling**    — per-channel multiplicative jitter

This forces the encoder to learn representations invariant to measurement
noise, sensor drop-out, and calibration drift — all realistic in an
industrial flotation-plant setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

# ------------------------------------------------------------------ #
#  Parsing                                                            #
# ------------------------------------------------------------------ #


def load_and_parse(filepath: str | Path, mode: str) -> pd.DataFrame:
    """Parse European-decimal CSV with pandas, add a 20 s counter."""
    df = pd.read_csv(
        filepath,
        decimal=",",  # European decimal: "55,2" → 55.2
        thousands=None,  # no thousands separator
    )
    match mode:
        case "train":
            split_point = int(len(df) * 0.75)
            out: pd.DataFrame = df.sort_index(ascending=True).iloc[:split_point]
        case "test":
            split_point = int(len(df) * 0.75)
            out = df.sort_index(ascending=True).iloc[split_point:]
    return out


# ------------------------------------------------------------------ #
#  Augmentation config                                                 #
# ------------------------------------------------------------------ #


@dataclass
class AugmentationConfig:
    """Controls intensity and type of stochastic augmentations per view.

    Each field can be turned off by setting it to zero (ratios / std)
    or to ``(1.0, 1.0)`` (scale range).
    """

    # Temporal masking — randomly zero out blocks of consecutive steps.
    temporal_mask_ratio: float = 0.15
    """Fraction of time steps to mask in total (approximate)."""
    temporal_mask_span: int = 3
    """Maximum span (in steps) of one contiguous mask block."""

    # Feature masking — zero entire sensor channels for the whole window.
    feature_mask_ratio: float = 0.10
    """Fraction of feature columns to drop."""

    # Gaussian noise — additive, zero-mean, scaled per channel.
    noise_std: float = 0.05
    """Standard deviation relative to per-channel std."""

    # Random scaling — per-channel multiplicative jitter.
    scale_range: tuple[float, float] = (0.9, 1.1)
    """Uniform range for the per-channel scale factor."""


# ------------------------------------------------------------------ #
#  LeJEPA dataset (multi-view)                                        #
# ------------------------------------------------------------------ #


class LeJEPADataset(Dataset[tuple[Tensor, Tensor]]):
    """Sliding-window dataset with multi-view augmentation for LeJEPA SSL.

    Each sample extracts a window of ``window_size`` consecutive time steps
    and returns:

    - ``"original"`` — the clean, normalised window ``(W, F)``
    - ``"views"``    — ``num_views`` augmented copies ``(K, W, F)``

    Call :meth:`set_stats` once before training to enable z-score
    normalisation (each channel centred by *mean* and scaled by *std*).
    """

    def __init__(
        self,
        filepath: str | Path,
        window_size: int = 60,
        num_views: int = 8,
        aug_config: AugmentationConfig | None = None,
        drop_columns=("NOX", "CO"),
    ) -> None:
        super().__init__()
        df = load_and_parse(filepath, mode="train")
        self.feature_names = [c for c in df.columns if c not in drop_columns]
        self.num_features = len(self.feature_names)
        self.window_size = window_size
        self.num_views = num_views
        self.aug_config = aug_config or AugmentationConfig()

        # Raw numeric matrix (float32)
        self.data = df[self.feature_names].to_numpy(dtype=np.float32)

        # Normalisation stats (set via set_stats)
        self._mean: Tensor | None = None
        self._std: Tensor | None = None

    # ---- Normalisation ------------------------------------------------

    def set_stats(self, mean: Tensor, std: Tensor) -> None:
        """Store per-channel mean and std for z-score normalisation."""
        self._mean = mean
        self._std = std.clamp_min(1e-8)

    def _normalise(self, x: Tensor) -> Tensor:
        if self._mean is not None and self._std is not None:
            return (x - self._mean) / self._std
        return x

    # ---- Dataset protocol ---------------------------------------------

    def __len__(self) -> int:
        return max(0, len(self.data) - self.window_size + 1)

    def __getitem__(self, index: int):
        window = self.data[index : index + self.window_size]  # (W, F)
        original = self._normalise(torch.from_numpy(window.copy()))

        views = torch.stack(
            [
                self._augment(self._normalise(torch.from_numpy(window.copy())))
                for _ in range(self.num_views)
            ]
        )

        return original, views

    # ---- Augmentation pipeline ----------------------------------------

    def _augment(self, x: Tensor) -> Tensor:
        """Compose all active augmentations in a fixed order."""
        if torch.rand(1).item() < 0.7:  # 70% chance to scale
            x = self._random_scale(x)

        if torch.rand(1).item() < 0.7:  # 70% chance to add noise
            x = self._add_noise(x)

        # 2. Destructive masking LAST
        if torch.rand(1).item() < 0.5:  # 50% chance to temporally mask
            x = self._temporal_mask(x)

        if torch.rand(1).item() < 0.5:  # 50% chance to feature mask
            x = self._feature_mask(x)
        return x

    def _temporal_mask(self, x: Tensor) -> Tensor:
        """Zero out random contiguous time blocks.

        Repeatedly places blocks of length ``1..mask_span`` at random
        start positions until approximately ``mask_ratio`` of the
        sequence is covered.
        """
        cfg = self.aug_config
        if cfg.temporal_mask_ratio <= 0:
            return x

        W = x.shape[0]
        num_to_mask = max(1, int(W * cfg.temporal_mask_ratio))
        out = x.clone()
        mask = torch.zeros(W, dtype=torch.bool)

        while mask.sum().item() < num_to_mask:
            start = torch.randint(0, W, (1,)).item()
            span = torch.randint(1, cfg.temporal_mask_span + 1, (1,)).item()
            end = min(start + span, W)
            mask[start:end] = True

        out[mask] = 0.0
        return out

    def _feature_mask(self, x: Tensor) -> Tensor:
        """Zero out entire sensor channels for the whole window."""
        cfg = self.aug_config
        if cfg.feature_mask_ratio <= 0:
            return x

        F = x.shape[1]
        num_to_mask = max(1, int(F * cfg.feature_mask_ratio))
        indices = torch.randperm(F)[:num_to_mask]

        out = x.clone()
        out[:, indices] = 0.0
        return out

    def _add_noise(self, x: Tensor) -> Tensor:
        """Add zero-mean Gaussian noise, scaled by per-channel std."""
        cfg = self.aug_config
        if cfg.noise_std <= 0:
            return x

        channel_std = x.std(dim=0, keepdim=True).clamp_min(1e-8)
        noise = torch.randn_like(x) * cfg.noise_std * channel_std
        return x + noise

    def _random_scale(self, x: Tensor) -> Tensor:
        """Multiply each feature channel by a random uniform factor."""
        cfg = self.aug_config
        lo, hi = cfg.scale_range
        if lo == 1.0 and hi == 1.0:
            return x

        F = x.shape[1]
        scales = torch.FloatTensor(F).uniform_(lo, hi)
        return x * scales
