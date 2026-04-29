import os
import gc
import json
import torch
import torch.optim as optim
from datasets import DSpritesDataset
from model import VAE
from helpers import (
    make_splits_and_loaders,
    train_pipeline_beta_tcvae,
    EarlyStopping,
    validate_beta_tcvae,
    set_seed,
)
from loss import beta_tcvae_loss
from plotters import (
    save_training_curves_beta_tcvae,
    save_encoder_kl_per_dim_plot,
    save_reconstructions,
    latent_traversal,
    save_latent_traversal,
)
from config import Config

cfg = Config()

PLOT_PATH = os.path.join(os.getcwd(), "plots")
BASE_DIR = os.path.join(PLOT_PATH, "tcvae")
DATA_PATH = os.path.join(os.getcwd(), cfg.data_filename)

DIRS = {
    "root": BASE_DIR,
    "search": os.path.join(BASE_DIR, "hyperparam_search"),
    "checkpoints": os.path.join(BASE_DIR, "checkpoints"),
    "curves": os.path.join(BASE_DIR, "training_curves"),
    "recon": os.path.join(BASE_DIR, "reconstructions"),
    "traversal": os.path.join(BASE_DIR, "latent_traversals"),
    "kl": os.path.join(BASE_DIR, "kl_analysis"),
    "test": os.path.join(BASE_DIR, "test_results"),
}

for path in DIRS.values():
    os.makedirs(path, exist_ok=True)

print("Created:", BASE_DIR)


def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("RAM cache cleared.")


set_seed(cfg.seed)
clear_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = (
    make_splits_and_loaders(
        dataset_cls=DSpritesDataset,
        dataset_kwargs={"data_path": DATA_PATH, "device": device},
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
    )
)

train_dataset_size = len(train_dataset)
val_dataset_size = len(val_dataset)

model = VAE(
    input_channels=cfg.input_channels,
    latent_dim=cfg.latent_dim,
    kernel_size=cfg.kernel_size,
    output_padding=cfg.output_padding,
    num_downsampling_layers=cfg.num_downsampling_layers,
    num_upsampling_layers=cfg.num_upsampling_layers,
    num_fc_layers=cfg.num_fc_layers,
    out_fc_features=cfg.out_fc_features,
    hidden_channels=cfg.hidden_channels,
).to(device)

# Start logvar near 0 (posterior variance ≈ 1, close to prior) so all runs
# begin from a consistent near-prior state regardless of weight seed.
torch.nn.init.zeros_(model.encoder.logvar.weight)
torch.nn.init.zeros_(model.encoder.logvar.bias)

optimizer = optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cfg.epochs,
    eta_min=cfg.eta_min,
)

early_stopping = EarlyStopping(
    patience=cfg.es_patience,
    min_delta=cfg.es_min_delta,
    mode="min",
)

history = train_pipeline_beta_tcvae(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=optimizer,
    device=device,
    epochs=cfg.epochs,
    beta_tcvae_loss_fn=beta_tcvae_loss,
    train_dataset_size=train_dataset_size,
    val_dataset_size=val_dataset_size,
    alpha=cfg.alpha,
    beta=cfg.beta,
    gamma=cfg.gamma,
    beta_warmup_epochs=cfg.beta_warmup_epochs,
    scheduler=scheduler,
    early_stopping=early_stopping,
    scheduler_step_per_batch=False,
)

checkpoint_path = os.path.join(DIRS["checkpoints"], "best_paper_beta_vae.pt")
torch.save(model.state_dict(), checkpoint_path)
print("Saved model to:", checkpoint_path)

test_loss, test_recon, test_mi, test_tc, test_dwkl, test_encoder_kl_per_dim = (
    validate_beta_tcvae(
        model=model,
        dataloader=test_loader,
        device=device,
        beta_tcvae_loss_fn=beta_tcvae_loss,
        alpha=cfg.alpha,
        beta=cfg.beta,
        gamma=cfg.gamma,
        desc="Test",
    )
)

test_results = {
    "test_loss": float(test_loss),
    "test_recon": float(test_recon),
    "test_mi": float(test_mi),
    "test_tc": float(test_tc),
    "test_dwkl": float(test_dwkl),
    "test_encoder_kl_per_dim": test_encoder_kl_per_dim.tolist(),
    "config": cfg.__dict__,
}

test_json = os.path.join(DIRS["test"], "test_results_beta_tcvae.json")
with open(test_json, "w") as f:
    json.dump(test_results, f, indent=2)

print(test_results)
print("Saved test results to:", test_json)

save_training_curves_beta_tcvae(
    history, save_dir=DIRS["curves"], prefix="beta_tcvae_dsprites"
)
save_encoder_kl_per_dim_plot(
    history, save_dir=DIRS["kl"], split="train", prefix="beta_tcvae_dsprites"
)
save_encoder_kl_per_dim_plot(
    history, save_dir=DIRS["kl"], split="val", prefix="beta_tcvae_dsprites"
)
save_reconstructions(
    model,
    test_loader,
    device,
    os.path.join(DIRS["recon"], "paper_beta_tcvae_recon.png"),
    n=cfg.n_recon_samples,
)

sample = test_dataset[cfg.traversal_sample_idx]
rows = latent_traversal(
    model,
    sample,
    device,
    traversal_range=cfg.traversal_range,
    steps=cfg.traversal_steps,
)
save_latent_traversal(
    rows,
    os.path.join(DIRS["traversal"], "paper_beta_tcvae_traversal.png"),
)
