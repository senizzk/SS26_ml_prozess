"""Train the LeJEPA SSL model.

Usage
-----
    # Default config, GPU if available:
    python -m ss26_ml_prozess.ssl.train

    # Override via environment (any LeJEPAConfig field):
    DATA_PATH=data/my_file.csv MAX_EPOCHS=100 python -m ss26_ml_prozess.ssl.train

The script:
  1. Loads ``LeJEPAConfig`` defaults (any field can be overridden via env vars).
  2. Seeds everything for reproducibility.
  3. Builds ``LeJEPATrainer`` — which handles dataset, model, loss, and optimiser.
  4. Calls ``trainer.fit()`` — full train/val loop with checkpointing.

After training, the encoder weights are in ``checkpoints/lejepa_epochNNN.pt``
and can be loaded for downstream fine-tuning:
    >>> checkpoint = torch.load("checkpoints/lejepa_epoch050.pt", map_location="cpu")
    >>> encoder.load_state_dict(checkpoint["encoder"])
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from ss26_ml_prozess.ssl.config import LeJEPAConfig
from ss26_ml_prozess.ssl.training_model import LeJEPATrainer


def _seed_everything(seed: int) -> None:
    """Set deterministic seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    # --- Build config, allow env-var overrides ---
    cfg = LeJEPAConfig(
        data_path=os.environ.get("DATA_PATH", LeJEPAConfig.data_path),
        max_epochs=int(os.environ.get("MAX_EPOCHS", LeJEPAConfig.max_epochs)),
        batch_size=int(os.environ.get("BATCH_SIZE", LeJEPAConfig.batch_size)),
        lr=float(os.environ.get("LR", LeJEPAConfig.lr)),
        device=os.environ.get("DEVICE", LeJEPAConfig.device),
    )

    print("=" * 60)
    print("LeJEPA Training")
    print("=" * 60)
    print(f"  Data         : {cfg.data_path}")
    print(f"  Window size  : {cfg.window_size}")
    print(f"  Num views    : {cfg.num_views}")
    print(f"  Encoder dims : {cfg.encoder_dims}")
    print(f"  Predictor    : {cfg.predictor_dims}")
    print(f"  Batch size   : {cfg.batch_size}")
    print(f"  Learning rate: {cfg.lr}")
    print(f"  Max epochs   : {cfg.max_epochs}")
    print(f"  Device       : {cfg.device}")
    print(f"  Seed         : {cfg.seed}")
    print("=" * 60)

    # --- Seed ---
    _seed_everything(cfg.seed)

    # --- Train ---
    trainer = LeJEPATrainer(cfg)
    history = trainer.fit()

    # --- Summary ---
    print("\nTraining complete!")
    final = {k: v[-1] for k, v in history.items()}
    print(f"  Final train loss  : {final['train_loss']:.4f}")
    print(f"  Final train inv    : {final['train_inv']:.4f}")
    print(f"  Final train var    : {final['train_var']:.4f}")
    print(f"  Final train cov    : {final['train_cov']:.6f}")
    print(f"  Final train lejepa: {final['train_lejepa']:.4f}")

    # --- Save final history ---
    import json

    history_path = Path(cfg.checkpoint_dir) / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved to {history_path}")


if __name__ == "__main__":
    main()