import json
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ss26_ml_prozess.ssl.config import LeJEPAConfig
from ss26_ml_prozess.ssl.forecast_data import ForecastingDataset
from ss26_ml_prozess.ssl.model import LSTMEncoder, TCNEncoder


class ForecastingHead(nn.Module):
    def __init__(self, input, num_targets: int = 2):
        super().__init__()
        self.internal_latent_dim = 128
        self.net = nn.Sequential(
            nn.Linear(input, self.internal_latent_dim),
            nn.BatchNorm1d(self.internal_latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.internal_latent_dim, int(self.internal_latent_dim / 2)),
            nn.BatchNorm1d(int(self.internal_latent_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(self.internal_latent_dim / 2), num_targets),
        )

    def forward(self, x):
        return self.net(x)


def linear_probe(cfg: LeJEPAConfig, checkpoint_name: str = "pretrained_best.pth"):
    latent_dimension = cfg.embedding_dim

    cp_path = cfg.checkpoint_dir
    checkpoint = torch.load(cp_path / checkpoint_name, weights_only=True)

    saved_x_mean: torch.Tensor = checkpoint["train_mean"]
    saved_x_std: torch.Tensor = checkpoint["train_std"]

    # 1. Initialize dataset WITHOUT y stats first
    full_dataset = ForecastingDataset(
        filepath=cfg.data_path,
        target_columns=cfg.target_columns,
        exclude_columns=cfg.drop_columns,
        window_size=cfg.window_size,
        x_mean=saved_x_mean,
        x_std=saved_x_std,
        y_mean=None,  # Leave empty for now
        y_std=None,
    )

    ds_size = len(full_dataset)
    train_size = int(ds_size * 0.8)
    indices = list(range(ds_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 2. Calculate y stats perfectly aligned with the dataloader split!
    train_y_data = full_dataset.y_data[:train_size]
    saved_y_mean = torch.tensor(train_y_data.mean(axis=0), dtype=torch.float32)
    saved_y_std = torch.tensor(train_y_data.std(axis=0), dtype=torch.float32)

    # 3. Inject stats safely back into the dataset
    full_dataset.y_mean = saved_y_mean
    full_dataset.y_std = saved_y_std

    train_ds, val_ds = (
        Subset(full_dataset, train_indices),
        Subset(full_dataset, val_indices),
    )
    train_load = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_load = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    match cfg.encoder:
        case "tcn":
            encoder = TCNEncoder(
                num_features=full_dataset.num_features,
                tcn_channels=list(cfg.tcn_channels),
                latent_dim=latent_dimension,
            ).to(cfg.device)
        case "lstm":
            encoder = LSTMEncoder(
                hidden_dim=64,
                num_features=full_dataset.num_features,
                num_layers=2,
                latent_dim=cfg.embedding_dim,
            ).to(cfg.device)

    # 2. Load the checkpoint and extract the encoder weights
    encoder.load_state_dict(checkpoint["encoder_state"])

    encoder.eval()

    for param in encoder.parameters():
        param.requires_grad = True

    # probe = nn.Linear(latent_dimension, 1).to(cfg.device)
    probe = ForecastingHead(
        input=latent_dimension, num_targets=len(cfg.target_columns)
    ).to(cfg.device)
    # optimizer = torch.optim.AdamW(
    #    list(probe.parameters()) + list(encoder.parameters()),
    #    lr=1e-4,
    #    weight_decay=1e-4,
    # )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": probe.parameters(),
                "lr": 1e-3,
            },  # Larger steps for the new probe
            {"params": encoder.parameters(), "lr": 1e-5},  # Gentle fine-tuning
        ],
        weight_decay=1e-3,
    )
    criterion = nn.MSELoss()
    # criterion = nn.HuberLoss(delta=1.0)
    probe_results = {"train": {"mse": []}, "val": {"mse": []}}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
    )
    for epoch in range(cfg.max_probe_epochs):
        # --- TRAINING PHASE --
        cp_path = cfg.checkpoint_dir
        probe.train()
        encoder.train()
        running_train_loss = 0.0

        for X, y in train_load:
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # 1. Extract latents without tracking gradients (super fast)
            # with torch.no_grad():
            latents = encoder(X)

            # 2. Pass latents through the trainable probe
            predictions = probe(latents)

            # 3. Calculate loss and backpropagate ONLY through the probe
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        mean_train_loss = running_train_loss / len(train_load)

        # --- VALIDATION PHASE ---
        probe.eval()
        encoder.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for X, y in val_load:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)

                latents = encoder(X)
                predictions = probe(latents)

                loss = criterion(predictions, y)
                running_val_loss += loss.item()

        mean_val_loss = running_val_loss / len(val_load)
        scheduler.step(
            mean_val_loss,
        )
        probe_results["train"]["mse"].append(mean_train_loss)
        probe_results["val"]["mse"].append(mean_val_loss)
        print(
            f"Epoch {epoch + 1:02d} | Train MSE: {mean_train_loss:.4f} | Val MSE: {mean_val_loss:.4f}"
        )
    forecaster_checkpoint = {
        "encoder_state": encoder.state_dict(),
        "probe_state": probe.state_dict(),
        # Crucial: bring the stats forward to the final save file
        "x_mean": saved_x_mean,
        "x_std": saved_x_std,
        "y_mean": saved_y_mean,
        "y_std": saved_y_std,
    }
    results_path = cp_path / f"forecasting_results_{'_'.join(cfg.target_columns)}.json"
    results_path.write_text(json.dumps(probe_results, indent=2))
    torch.save(
        forecaster_checkpoint,
        cp_path / f"forecasting_{'_'.join(cfg.target_columns)}.pth",
    )


if __name__ == "__main__":
    config = LeJEPAConfig()
    if len(sys.argv) > 1:
        target_checkpoint = sys.argv[1]
    else:
        target_checkpoint = "pretrained_best.pth"
    print(f"Starting linear probe  with checkpoint: {target_checkpoint}")
    linear_probe(cfg=config)
