import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ss26_ml_prozess.ssl.data import load_and_parse


# forecast_data.py
class ForecastingDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        target_columns: tuple,
        exclude_columns: tuple,
        window_size: int = 60,
        forecast_horizon: int = 1,
        mode: str = "train",
        x_mean: torch.Tensor | None = None,  # From pretraining
        x_std: torch.Tensor | None = None,
        y_mean: torch.Tensor | None = None,  # Calculated in forecasting.py
        y_std: torch.Tensor | None = None,
    ):
        df = load_and_parse(filepath, mode=mode)

        # 1. SEPARATE THE FEATURES AND TARGETS
        self.target_names = list(target_columns)
        self.feature_names = [
            c for c in df.columns if c not in exclude_columns and c != "date"
        ]

        self.num_features = len(self.feature_names)
        self.num_targets = len(self.target_names)

        self.window_size = window_size
        self.forecast_horizon = forecast_horizon

        # 2. SEPARATE ARRAYS
        self.X_data = df[self.feature_names].to_numpy(dtype=np.float32)
        self.y_data = df[self.target_names].to_numpy(dtype=np.float32)

        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def _normalise_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.x_mean is not None and self.x_std is not None:
            return (x - self.x_mean) / self.x_std
        return x

    def __len__(self) -> int:
        return max(0, len(self.X_data) - self.window_size - self.forecast_horizon + 1)

    def __getitem__(self, index: int):
        # 1. Input Window (strictly features)
        window = self.X_data[index : index + self.window_size]
        X = self._normalise_x(torch.from_numpy(window.copy()))

        # 2. Future Target
        target_step_idx = index + self.window_size + self.forecast_horizon - 1
        y_raw = self.y_data[target_step_idx]

        # 3. Target Normalization
        if self.y_mean is not None and self.y_std is not None:
            y = (y_raw - self.y_mean.numpy()) / self.y_std.numpy()
        else:
            y = y_raw

        return X, torch.tensor(y, dtype=torch.float32)
