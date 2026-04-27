import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def save_training_curves_beta_tcvae(history, save_dir, prefix="beta_tcvae"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Total loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.title("Train / Val Total Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_total_loss.png"), dpi=200)
    plt.close()

    # Reconstruction
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_recon"], label="Train")
    plt.plot(epochs, history["val_recon"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction loss")
    plt.title("Train / Val Reconstruction Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_recon_loss.png"), dpi=200)
    plt.close()

    # MI
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_mi"], label="Train")
    plt.plot(epochs, history["val_mi"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Index-code MI")
    plt.title("Train / Val Index-Code MI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_mi_loss.png"), dpi=200)
    plt.close()

    # TC
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_tc"], label="Train")
    plt.plot(epochs, history["val_tc"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Total Correlation")
    plt.title("Train / Val Total Correlation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_tc_loss.png"), dpi=200)
    plt.close()

    # Dimension-wise KL
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_dwkl"], label="Train")
    plt.plot(epochs, history["val_dwkl"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Dimension-wise KL")
    plt.title("Train / Val Dimension-wise KL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_dwkl_loss.png"), dpi=200)
    plt.close()


def save_reconstructions(model, dataloader, device, save_path, n=10, logits=True):

    model.eval()

    batch = next(iter(dataloader))

    x = batch[0] if isinstance(batch, (list, tuple)) else batch

    x = x[:n].to(device)

    with torch.no_grad():

        x_hat, _, _, _ = model(x)

        if logits:

            x_hat = torch.sigmoid(x_hat)

    x = x.cpu()

    x_hat = x_hat.cpu()

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))

    for i in range(n):

        axes[0, i].imshow(x[i, 0], cmap="gray")

        axes[0, i].axis("off")

        axes[0, i].set_title("Orig" if i == 0 else "")

        axes[1, i].imshow(x_hat[i, 0], cmap="gray")

        axes[1, i].axis("off")

        axes[1, i].set_title("Recon" if i == 0 else "")

    plt.tight_layout()

    plt.savefig(save_path, dpi=200)

    plt.close()


def latent_traversal(model, sample, device, traversal_range=(-3, 3), steps=9, logits=True):

    model.eval()

    if sample.dim() == 3:

        sample = sample.unsqueeze(0)

    sample = sample.to(device)

    with torch.no_grad():

        _, mu, _, _ = model(sample)

        z = mu.clone()

        latent_dim = z.shape[1]

        values = torch.linspace(
            traversal_range[0], traversal_range[1], steps, device=device)

        rows = []

        for d in range(latent_dim):

            row = []

            for val in values:

                z_mod = z.clone()

                z_mod[0, d] = val

                x_hat = model.decoder(z_mod)

                if logits:

                    x_hat = torch.sigmoid(x_hat)

                row.append(x_hat[0, 0].cpu())

            rows.append(row)

    return rows


def save_latent_traversal(rows, save_path):

    n_rows = len(rows)

    n_cols = len(rows[0])

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows))

    if n_rows == 1:

        axes = np.expand_dims(axes, axis=0)

    for i in range(n_rows):

        for j in range(n_cols):

            axes[i, j].imshow(rows[i][j], cmap="gray")

            axes[i, j].axis("off")

    plt.tight_layout()

    plt.savefig(save_path, dpi=200)

    plt.close()


def save_encoder_kl_per_dim_plot(history, save_dir, split="train", prefix="beta_tcvae"):
    key = "train_encoder_kl_per_dim" if split == "train" else "val_encoder_kl_per_dim"
    kl_hist = torch.stack(history[key])  # (epochs, latent_dim)

    plt.figure(figsize=(8, 5))
    for d in range(kl_hist.shape[1]):
        plt.plot(range(1, kl_hist.shape[0] + 1),
                 kl_hist[:, d].numpy(), label=f"z{d}")

    plt.xlabel("Epoch")
    plt.ylabel("Average encoder KL (nats)")
    plt.title(f"{split.capitalize()} encoder KL per latent dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(
        save_dir, f"{prefix}_{split}_encoder_kl_per_dim.png"), dpi=200)
    plt.close()
