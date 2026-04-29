from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Reproducibility
    seed: int = 42

    # Data
    data_filename: str = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    batch_size: int = 1024
    num_workers: int = 4
    train_frac: float = 0.8
    val_frac: float = 0.1

    # Model architecture
    input_channels: int = 1
    input_size: tuple[int, int] = (64, 64)
    latent_dim: int = 10
    hidden_channels: int = 32
    num_downsampling_layers: int = 4
    num_upsampling_layers: int = 4
    num_fc_layers: int = 2
    out_fc_features: int = 256
    kernel_size: int = 3
    output_padding: int = 1

    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-3

    # Scheduler
    eta_min: float = 5e-5

    # β-TCVAE loss weights
    alpha: float = 1.0
    beta: float = 7.0
    gamma: float = 1.0
    beta_warmup_epochs: int = 100

    # Training
    epochs: int = 700

    # Early stopping
    es_patience: int = 50
    es_min_delta: float = 1e-5

    # Visualisation
    traversal_sample_idx: int = 100
    traversal_range: tuple[float, float] = (-3.0, 3.0)
    traversal_steps: int = 9
    n_recon_samples: int = 10
