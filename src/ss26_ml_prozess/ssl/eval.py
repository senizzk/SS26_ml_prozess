import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from ss26_ml_prozess.ssl.config import LeJEPAConfig
from ss26_ml_prozess.ssl.forecast_data import ForecastingDataset
from ss26_ml_prozess.ssl.forecasting import ForecastingHead
from ss26_ml_prozess.ssl.model import LSTMEncoder, TCNEncoder

cfg = LeJEPAConfig()
cp_path = cfg.checkpoint_dir
checkpoint = torch.load(
    cp_path / f"forecasting_{'_'.join(cfg.target_columns)}.pth",
    map_location="cpu",
    weights_only=True,
)

# 1. Load the separated X and y statistics
# (Make sure these keys exactly match what you saved in forecasting.py!)
x_mean = checkpoint["x_mean"]
x_std = checkpoint["x_std"]
y_mean = checkpoint["y_mean"]
y_std = checkpoint["y_std"]

# 2. Setup the Test Dataset
test_dataset = ForecastingDataset(
    filepath=cfg.data_path,
    mode="test",
    target_columns=cfg.target_columns,
    exclude_columns=cfg.drop_columns,  # Swapped drop_columns for exclude_columns
    window_size=cfg.window_size,
    x_mean=x_mean,  # Pass the 4 separated stats
    x_std=x_std,
    y_mean=y_mean,
    y_std=y_std,
)

test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# 3. Initialize the Models
latent_dimension = cfg.embedding_dim

match cfg.encoder:
    case "tcn":
        encoder = TCNEncoder(
            num_features=test_dataset.num_features,  # Use dataset property directly
            tcn_channels=list(cfg.tcn_channels),
            latent_dim=latent_dimension,
        )
    case "lstm":
        encoder = LSTMEncoder(
            hidden_dim=64,
            num_features=test_dataset.num_features,
            num_layers=2,
            latent_dim=cfg.embedding_dim,
        ).to(cfg.device)

# Fixed kwarg: latent_dim instead of input
probe = ForecastingHead(input=latent_dimension, num_targets=len(cfg.target_columns))

# Load weights
encoder.load_state_dict(checkpoint["encoder_state"])
probe.load_state_dict(checkpoint["probe_state"])

# Move to GPU and set to EVAL mode (Strictly required!)
encoder = encoder.cuda().eval()
probe = probe.cuda().eval()

# 4. The Inference Loop
all_predictions = []
all_targets = []

with torch.no_grad():
    for X, y in test_loader:
        X = X.cuda(non_blocking=True)

        latents = encoder(X)
        preds = probe(latents)

        all_predictions.append(preds.cpu().numpy())
        all_targets.append(y.numpy())

y_pred_normalized = np.vstack(all_predictions)
y_true_normalized = np.vstack(all_targets)

# 5. Inverse Transform (Massively Simplified)
# Because y_mean and y_std are already perfectly shaped for the targets,
# we don't need to slice them using target_idxs anymore!
target_means = y_mean.numpy()
target_stds = y_std.numpy()

y_pred_actual = (y_pred_normalized * target_stds) + target_means
y_true_actual = (y_true_normalized * target_stds) + target_means


# 6. Calculate Metrics
print("=== Final Test Set Evaluation ===")

for i, target_name in enumerate(cfg.target_columns):
    target_true = y_true_actual[:, i]
    target_pred = y_pred_actual[:, i]

    r2 = r2_score(target_true, target_pred)
    mae = np.mean(np.abs(target_true - target_pred))

    print(f"Target: {target_name}")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  MAE:      {mae:.4f} (in original units)")
    print("-" * 30)
