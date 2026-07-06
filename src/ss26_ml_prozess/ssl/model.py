import json
from collections.abc import Callable

import lejepa
import torch
import torch.nn as nn
from pytorch_tcn import TCN
from torch.utils.data import DataLoader, Subset

from ss26_ml_prozess.ssl.config import LeJEPAConfig
from ss26_ml_prozess.ssl.data import LeJEPADataset


class LeJEPAPredictor(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        # A standard MLP predictor.
        # It maps the augmented context latents back to the target latent.
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If x is shape (Batch, K_views, Latent), we flatten the first two
        # dimensions so BatchNorm1d can process it, then reshape it back.
        B, K, D = x.shape
        x_flat = x.view(B * K, D)
        out_flat = self.net(x_flat)
        return out_flat.view(B, K, D)


class EarlyStopping:
    def __init__(
        self, patience: int = 15, min_delta: float = 0.0, warmup_epochs: int = 10
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(
        self, current_loss: float, save_checkpoint_fn: Callable, epoch: int
    ) -> bool:
        # Ignore the chaotic expansion phase of the latent space
        if epoch < self.warmup_epochs:
            return False

        # If we beat the best loss by min_delta
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            save_checkpoint_fn()  # Only save when we hit a new best!
            return True  # Indicates improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # Indicates no improvement


class LSTMEncoder(nn.Module):
    def __init__(
        self, num_features: int, hidden_dim: int, latent_dim: int, num_layers: int = 2
    ):
        super().__init__()
        # batch_first=True is vital here!
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.projector = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Window, Features)

        # out: (Batch, Window, hidden_dim)
        # hidden: (num_layers, Batch, hidden_dim)
        out, (hidden, cell) = self.lstm(x)

        # Use the final hidden state of the last layer
        last_hidden = hidden[-1]

        # Project to latent space
        return self.projector(last_hidden)


class TCNEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        tcn_channels: list[int],
        latent_dim: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.tcn = TCN(
            num_inputs=num_features,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            causal=True,
        )
        self.projector = nn.Linear(tcn_channels[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        context_vector = out[:, :, -1]
        latent = self.projector(context_vector)
        return latent


class LeJEPALoss(nn.Module):
    def __init__(self, lambda_sigreg: float = 0.05, num_slices: int = 256):
        super().__init__()
        self.lambda_sigreg = lambda_sigreg

        # 1. Initialize the Univariate test (Epps-Pulley is recommended by the paper)
        univariate_test = lejepa.univariate.EppsPulley()

        # 2. Initialize the Multivariate Slicing test (SIGReg)
        self.sigreg_fn = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test, num_slices=num_slices
        )

    def forward(self, z_original: torch.Tensor, z_views_pred: torch.Tensor):
        """
        z_original: (Batch, latent_dim)
        z_views_pred: (Batch, K_views, latent_dim)
        """
        B, K, D = z_views_pred.shape

        # --- 1. Predictive Loss ---
        # Expand the original target to match the K views: (Batch, K, latent_dim)
        z_target_expanded = z_original.unsqueeze(1).expand(-1, K, -1)

        # Mean Squared Error between the predicted contexts and the actual target
        predictive_loss = nn.functional.mse_loss(z_views_pred, z_target_expanded)

        # --- 2. SIGReg Loss ---
        # We apply SIGReg to the original embeddings to enforce the Isotropic
        # Gaussian distribution. This acts as the mathematical guardrail
        # that prevents representation collapse.
        sigreg_loss = self.sigreg_fn(z_original)

        # --- 3. Total Loss ---
        # lambda_reg is the single trade-off hyperparameter in LeJEPA
        total_loss = (
            1 - self.lambda_sigreg
        ) * predictive_loss + self.lambda_sigreg * sigreg_loss

        return total_loss, predictive_loss, sigreg_loss


def train_encoder(cfg: LeJEPAConfig):
    full_dataset = LeJEPADataset(
        filepath=cfg.data_path,
        window_size=cfg.window_size,
        num_views=cfg.num_views,
        drop_columns=cfg.drop_columns,
    )

    ds_size = len(full_dataset)
    train_size = int(ds_size * 0.8)
    indices = list(range(ds_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    raw_tensor = torch.from_numpy(full_dataset.data[:train_size])
    train_mean = raw_tensor.mean(dim=0)
    train_std = raw_tensor.std(dim=0)
    full_dataset.set_stats(
        mean=train_mean,
        std=train_std,
    )
    train_ds, val_ds = (
        Subset(full_dataset, train_indices),
        Subset(full_dataset, val_indices),
    )
    train_load = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    val_load = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    match cfg.encoder:
        case "tcn":
            encoder = TCNEncoder(
                num_features=full_dataset.num_features,
                tcn_channels=list(cfg.tcn_channels),
                latent_dim=cfg.embedding_dim,
            ).to(cfg.device)
        case "lstm":
            encoder = LSTMEncoder(
                hidden_dim=64,
                num_features=full_dataset.num_features,
                num_layers=2,
                latent_dim=cfg.embedding_dim,
            ).to(cfg.device)

    predictor = LeJEPAPredictor(latent_dim=cfg.embedding_dim, hidden_dim=128).to(
        cfg.device
    )
    loss_fn = LeJEPALoss(lambda_sigreg=cfg.lejepa_weight).to(cfg.device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs, eta_min=5e-5
    )
    results = {
        "train": {"total": [], "prediction": [], "sigreg": []},
        "val": {"total": [], "prediction": [], "sigreg": []},
    }
    early_stopper = EarlyStopping(
        patience=cfg.patience, min_delta=cfg.min_delta, warmup_epochs=1
    )
    for epoch in range(cfg.max_epochs):
        encoder.train()
        predictor.train()
        running_loss = 0.0
        running_pred_loss = 0.0
        running_sigreg_loss = 0.0
        for original, views in train_load:
            optimizer.zero_grad()
            B, K, W, F = views.shape
            z_original = encoder(original.to(cfg.device))

            views_flat = views.view(B * K, W, F).to(cfg.device)
            z_views_flat = encoder(views_flat)
            z_views = z_views_flat.view(B, K, cfg.embedding_dim)

            # z_views_pred = predictor(z_views)
            loss, pred_loss, sigreg_loss = loss_fn(z_original, z_views)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            running_pred_loss += pred_loss.item()
            running_sigreg_loss += sigreg_loss.item()
        num_batches = len(train_load)
        mean_train_loss = running_loss / num_batches
        mean_train_pred = running_pred_loss / num_batches
        mean_train_sigreg = running_sigreg_loss / num_batches
        results["train"]["total"].append(mean_train_loss)
        results["train"]["prediction"].append(mean_train_pred)
        results["train"]["sigreg"].append(mean_train_sigreg)

        encoder.eval()
        predictor.eval()

        val_loss, val_pred, val_sigreg = 0.0, 0.0, 0.0
        with torch.no_grad():
            for original, views in val_load:
                B, K, W, F = views.shape

                z_original = encoder(original.to(cfg.device))

                views_flat = views.view(B * K, W, F)
                z_views_flat = encoder(views_flat.to(cfg.device))
                z_views = z_views_flat.view(B, K, cfg.embedding_dim)

                loss, pred_loss, sigreg_loss = loss_fn(z_original, z_views)

                val_loss += loss.item()
                val_pred += pred_loss.item()
                val_sigreg += sigreg_loss.item()

        num_val_batches = len(val_load)
        mean_val_loss = val_loss / num_val_batches
        mean_val_pred = val_pred / num_val_batches
        mean_val_sigreg = val_sigreg / num_val_batches
        results["val"]["total"].append(mean_val_loss)
        results["val"]["prediction"].append(mean_val_pred)
        results["val"]["sigreg"].append(mean_val_sigreg)
        print(
            f"Epoch: {epoch + 1:03d}/{cfg.max_epochs} | "
            f"TRAIN  Tot: {mean_train_loss:.4f}  Pred: {mean_train_pred:.4f}  SIG: {mean_train_sigreg:.4f} | "
            f"VAL  Tot: {mean_val_loss:.4f}  Pred: {mean_val_pred:.4f}  SIG: {mean_val_sigreg:.4f}"
        )
        if epoch:
            checkpoint = {
                "epoch": epoch,
                "encoder_state": encoder.state_dict(),
                "predictor_state": predictor.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_mean": train_mean,
                "train_std": train_std,
            }
            cp_path = cfg.checkpoint_dir
            torch.save(checkpoint, cp_path / f"pretrained_{epoch}.pth")

        def save_best_checkpoint():
            checkpoint = {
                "epoch": epoch,
                "encoder_state": encoder.state_dict(),
                "predictor_state": predictor.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_mean": train_mean,
                "train_std": train_std,
            }
            # We overwrite "pretrained_best.pth" so you always have the optimal weights
            torch.save(checkpoint, cfg.checkpoint_dir / "pretrained_best.pth")
            print(f"--> New best model saved! (Val Pred Loss: {mean_val_pred:.4f})")

        early_stopper(
            current_loss=mean_val_pred,
            save_checkpoint_fn=save_best_checkpoint,
            epoch=epoch,
        )
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    checkpoint = {
        "epoch": epoch,
        "encoder_state": encoder.state_dict(),
        "predictor_state": predictor.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_mean": train_mean,
        "train_std": train_std,
    }

    # Save it to disk
    #
    cp_path = cfg.checkpoint_dir
    results_path = cp_path / "pretrain_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    torch.save(checkpoint, cp_path / "pretrained.pth")
    print("Model saved successfully!")


if __name__ == "__main__":
    config = LeJEPAConfig()
    train_encoder(cfg=config)
