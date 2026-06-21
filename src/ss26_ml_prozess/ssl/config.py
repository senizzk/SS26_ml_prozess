from dataclasses import dataclass


@dataclass
class LeJEPAConfig:
    # Data
    data_path: str = "data/MiningProcess_Flotation_Plant_Database.csv"
    window_size: int = 60  # readings per window
    num_views: int = 2  # augmented views per sample
    val_split: float = 0.1  # fraction of data for validation
    context_ratio: float = 0.7  # fraction of window used as context

    # Encoder — processes each time step individually
    embedding_dim: int = 1024
    encoder_dims: tuple = (64, embedding_dim)  # hidden dims; last is embedding dim
    encoder_act: str = "gelu"

    # Predictor — maps context embedding -> target embedding
    predictor_dims: tuple = (
        128,
        embedding_dim,
    )  # hidden dims; output dim must match encoder[-1]
    predictor_act: str = "gelu"

    # SSL loss (VICReg-style)
    inv_weight: float = 1.0  # invariance (MSE)
    var_weight: float = 1.0  # variance: push std -> gamma
    cov_weight: float = 0.04  # covariance: decorrelate
    var_gamma: float = 1.0  # target std for variance term

    # Training
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 50
    warmup_steps: int = 500
    seed: int = 42

    # Logging / checkpoint
    log_every: int = 50
    checkpoint_dir: str = "checkpoints"

    # Device
    device: str = "auto"  # auto | cpu | cuda

    def __post_init__(self):
        assert 0 < self.context_ratio < 1
        assert self.predictor_dims[-1] == self.encoder_dims[-1], (
            f"Predictor output dim {self.predictor_dims[-1]} must match "
            f"encoder embedding dim {self.encoder_dims[-1]}"
        )
