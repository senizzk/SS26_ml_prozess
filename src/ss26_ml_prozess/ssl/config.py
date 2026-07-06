from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class LeJEPAConfig:
    # Data
    data_path: str = "/home/gergo/projects/SS26_ml_prozess/data/gt_total.csv"
    window_size: int = 96  # readings per window
    num_views: int = 8  # augmented views per sample
    val_split: float = 0.2  # fraction of data for validation
    target_columns: tuple = ("NOX",)
    drop_columns: tuple = ("CO",)
    # Encoder — processes each time step individually
    embedding_dim: int = 100
    encoder_act: str = "gelu"
    lejepa_weight: float = 0.03
    # Training
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 10
    warmup_steps: int = 500
    seed: int = 42

    num_workers: int = 8

    # Logging / checkpoint
    log_every: int = 50
    checkpoint_dir: Path = Path("/home/gergo/projects/SS26_ml_prozess/checkpoints/tcn4")

    max_probe_epochs: int = 30
    encoder: Literal["tcn", "lstm"] = "tcn"
    tcn_channels: tuple = (32, 64, 128)
    patience: int = 15
    min_delta: float = 1e-4
    # Device
    device: str = "cuda"  #  cpu | cuda
