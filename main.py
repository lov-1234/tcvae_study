import os
import gc
import numpy as np
import torch
import json
from datasets import DSpritesDataset
from model import VAE
from helpers import make_splits_and_loaders, train_pipeline_beta_tcvae, EarlyStopping, validate_beta_tcvae
from loss import beta_tcvae_loss
from plotters import save_training_curves_beta_tcvae
import torch.optim as optim
from plotters import save_training_curves_beta_tcvae, save_encoder_kl_per_dim_plot, save_reconstructions, latent_traversal, save_latent_traversal

PLOT_PATH = os.path.join(os.getcwd(), 'plots')
BASE_DIR = os.path.join(PLOT_PATH, 'tcvae')
dsprites_data_path = os.path.join(
    os.getcwd(), 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')    # Make sure that you have the data available at this path.
dsprites_data = np.load(dsprites_data_path)

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
    gc.collect()  # Python garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # PyTorch CUDA cache
    print("RAM cache cleared.")


clear_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset_kwargs = {
    "data_path": dsprites_data_path,
    "device": device
}

train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = make_splits_and_loaders(
    dataset_cls=DSpritesDataset,
    dataset_kwargs=dataset_kwargs,
    batch_size=1024,
    seed=42,
    num_workers=4
)


train_dataset_size = len(train_dataset)
val_dataset_size = len(val_dataset)

best_model = VAE(input_channels=1,
                 latent_dim=10,
                 kernel_size=3,
                 output_padding=1,
                 num_downsampling_layers=4,
                 num_upsampling_layers=4,
                 num_fc_layers=2,
                 out_fc_features=256,
                 hidden_channels=32).to(device)

paper_optimizer = optim.AdamW(
    best_model.parameters(),
    lr=5e-4,
    weight_decay=1e-4
)

paper_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    paper_optimizer,
    T_max=500,
    eta_min=1e-5
)

best_early_stopping = EarlyStopping(
    patience=50,
    min_delta=1e-5,
    mode="min"
)

history = train_pipeline_beta_tcvae(
    model=best_model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=paper_optimizer,
    device=device,
    epochs=500,
    beta_tcvae_loss_fn=beta_tcvae_loss,
    train_dataset_size=train_dataset_size,
    val_dataset_size=val_dataset_size,
    alpha=1,
    beta=10.0,
    gamma=1,
    beta_warmup_epochs=50,
    scheduler=paper_scheduler,
    early_stopping=best_early_stopping,
    scheduler_step_per_batch=False
)

checkpoint_path = os.path.join(DIRS["checkpoints"], "best_paper_beta_vae.pt")
torch.save(best_model.state_dict(), checkpoint_path)
print("Saved best papermodel to:", checkpoint_path)

test_loss, test_recon, test_mi, test_tc, test_dwkl, test_encoder_kl_per_dim = validate_beta_tcvae(
    model=best_model,
    dataloader=test_loader,
    device=device,
    dataset_size=train_dataset_size,
    beta_tcvae_loss_fn=beta_tcvae_loss,
    alpha=1.0,
    beta=15.0,
    gamma=1.0,
    desc="Test"
)

test_results = {
    "test_loss": float(test_loss),
    "test_recon": float(test_recon),
    "test_mi": float(test_mi),
    "test_tc": float(test_tc),
    "test_dwkl": float(test_dwkl),
    "test_encoder_kl_per_dim": test_encoder_kl_per_dim.tolist(),
}

test_json = os.path.join(DIRS["test"], "test_results_beta_tcvae.json")

with open(test_json, "w") as f:
    json.dump(test_results, f, indent=2)

print(test_results)
print("Saved test results to:", test_json)

save_training_curves_beta_tcvae(
    history, save_dir=DIRS["curves"], prefix="beta_tcvae_dsprites")
save_encoder_kl_per_dim_plot(
    history, save_dir=DIRS["kl"], split="train", prefix="beta_tcvae_dsprites")
save_encoder_kl_per_dim_plot(
    history, save_dir=DIRS["kl"], split="val", prefix="beta_tcvae_dsprites")
save_reconstructions(
    best_model,
    test_loader,
    device,
    os.path.join(DIRS["recon"], "paper_beta_tcvae_recon.png"),
    n=10
)
sample = test_dataset[100]
rows = latent_traversal(
    best_model,
    sample,
    device,
    traversal_range=(-3, 3),
    steps=9
)

save_latent_traversal(
    rows,
    os.path.join(DIRS["traversal"], "paper_beta_tcvae_traversal.png")
)
