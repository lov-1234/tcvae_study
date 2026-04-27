import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from loss import kl_loss, encoder_kl_per_dim
from tqdm import tqdm
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_state_dict = None
        self.should_stop = False

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True

        return False


def train_one_epoch_beta_tcvae(
    model,
    dataloader,
    optimizer,
    device,
    beta_tcvae_loss_fn,
    alpha=1.0,
    beta=6.0,
    gamma=1.0,
    scheduler=None,
):
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_mi = 0.0
    total_tc = 0.0
    total_dwkl = 0.0
    total_encoder_kl_per_dim = None
    total_kl = 0.0
    samples = 0

    for batch in tqdm(dataloader, desc="Training"):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        batch_size = x.size(0)

        optimizer.zero_grad(set_to_none=True)

        x_hat, mu, logvar, z = model(x)

        loss, recon, mi, tc, dwkl = beta_tcvae_loss_fn(
            x=x,
            x_hat=x_hat,
            mu=mu,
            logvar=logvar,
            z=z,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * batch_size
        total_recon += recon.item() * batch_size
        total_mi += mi.item() * batch_size
        total_tc += tc.item() * batch_size
        total_dwkl += dwkl.item() * batch_size
        total_kl += kl_loss(mu, logvar) * batch_size

        kl_per_dim_batch = encoder_kl_per_dim(mu, logvar)  # (D,)
        if total_encoder_kl_per_dim is None:
            total_encoder_kl_per_dim = kl_per_dim_batch.detach() * batch_size
        else:
            total_encoder_kl_per_dim += kl_per_dim_batch.detach() * batch_size

        samples += batch_size

    avg_loss = total_loss / samples
    avg_recon = total_recon / samples
    avg_mi = total_mi / samples
    avg_tc = total_tc / samples
    avg_dwkl = total_dwkl / samples
    avg_encoder_kl_per_dim = total_encoder_kl_per_dim / samples
    print(f"Standard KL (TRAIN): {total_kl / samples}")
    print(
        f"Combined KL (TRAIN): {(total_tc + total_dwkl + total_mi) / samples}")
    return avg_loss, avg_recon, avg_mi, avg_tc, avg_dwkl, avg_encoder_kl_per_dim.cpu()


def validate_beta_tcvae(
    model,
    dataloader,
    device,
    beta_tcvae_loss_fn,
    alpha=1.0,
    beta=6.0,
    gamma=1.0,
    desc="Validation",
):
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_mi = 0.0
    total_tc = 0.0
    total_dwkl = 0.0
    total_kl = 0.0
    total_encoder_kl_per_dim = None
    samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            batch_size = x.size(0)

            x_hat, mu, logvar, z = model(x)

            loss, recon, mi, tc, dwkl = beta_tcvae_loss_fn(
                x=x,
                x_hat=x_hat,
                mu=mu,
                logvar=logvar,
                z=z,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

            total_loss += loss.item() * batch_size
            total_recon += recon.item() * batch_size
            total_mi += mi.item() * batch_size
            total_tc += tc.item() * batch_size
            total_dwkl += dwkl.item() * batch_size
            total_kl += kl_loss(mu, logvar) * batch_size

            kl_per_dim_batch = encoder_kl_per_dim(mu, logvar)  # (D,)
            if total_encoder_kl_per_dim is None:
                total_encoder_kl_per_dim = kl_per_dim_batch.detach() * batch_size
            else:
                total_encoder_kl_per_dim += kl_per_dim_batch.detach() * batch_size

            samples += batch_size

    avg_loss = total_loss / samples
    avg_recon = total_recon / samples
    avg_mi = total_mi / samples
    avg_tc = total_tc / samples
    avg_dwkl = total_dwkl / samples
    avg_encoder_kl_per_dim = total_encoder_kl_per_dim / samples
    print(f"Standard KL (VALIDATION): {total_kl / samples}")
    print(
        f"Combined KL (VALIDATION): {(total_tc + total_dwkl + total_mi)/samples}")
    return avg_loss, avg_recon, avg_mi, avg_tc, avg_dwkl, avg_encoder_kl_per_dim.cpu()


def _beta_at_epoch(beta, epoch, warmup_epochs):
    if warmup_epochs <= 0:
        return beta
    return beta * min(1.0, epoch / warmup_epochs)


def train_pipeline_beta_tcvae(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    epochs,
    beta_tcvae_loss_fn,
    train_dataset_size,
    val_dataset_size=None,
    alpha=1.0,
    beta=6.0,
    gamma=1.0,
    beta_warmup_epochs=0,
    scheduler=None,
    early_stopping=None,
    scheduler_step_per_batch=True,
):
    if val_dataset_size is None:
        val_dataset_size = train_dataset_size

    history = {
        "train_loss": [],
        "train_recon": [],
        "train_mi": [],
        "train_tc": [],
        "train_dwkl": [],
        "train_encoder_kl_per_dim": [],
        "val_loss": [],
        "val_recon": [],
        "val_mi": [],
        "val_tc": [],
        "val_dwkl": [],
        "val_encoder_kl_per_dim": [],
    }

    for epoch in range(epochs):
        current_beta = _beta_at_epoch(beta, epoch, beta_warmup_epochs)
        print(f"Starting epoch {epoch + 1}/{epochs}  (beta={current_beta:.3f})")

        (
            train_loss,
            train_recon,
            train_mi,
            train_tc,
            train_dwkl,
            train_encoder_kl_per_dim,
        ) = train_one_epoch_beta_tcvae(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            beta_tcvae_loss_fn=beta_tcvae_loss_fn,
            alpha=alpha,
            beta=current_beta,
            gamma=gamma,
            scheduler=scheduler if scheduler_step_per_batch else None,
        )

        (
            val_loss,
            val_recon,
            val_mi,
            val_tc,
            val_dwkl,
            val_encoder_kl_per_dim,
        ) = validate_beta_tcvae(
            model=model,
            dataloader=val_dataloader,
            device=device,
            beta_tcvae_loss_fn=beta_tcvae_loss_fn,
            alpha=alpha,
            beta=current_beta,
            gamma=gamma,
        )

        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_recon)
        history["train_mi"].append(train_mi)
        history["train_tc"].append(train_tc)
        history["train_dwkl"].append(train_dwkl)
        history["train_encoder_kl_per_dim"].append(train_encoder_kl_per_dim)

        history["val_loss"].append(val_loss)
        history["val_recon"].append(val_recon)
        history["val_mi"].append(val_mi)
        history["val_tc"].append(val_tc)
        history["val_dwkl"].append(val_dwkl)
        history["val_encoder_kl_per_dim"].append(val_encoder_kl_per_dim)

        if scheduler is not None and not scheduler_step_per_batch:
            scheduler.step()

        print(
            f"Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, "
            f"MI: {train_mi:.4f}, TC: {train_tc:.4f}, DWKL: {train_dwkl:.4f}\n"
            f"Val   - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, "
            f"MI: {val_mi:.4f}, TC: {val_tc:.4f}, DWKL: {val_dwkl:.4f}"
        )

        if early_stopping is not None:
            early_stopping.step(val_loss, model)
            if early_stopping.should_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    if early_stopping is not None and early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)

    return history


def make_splits_and_loaders(dataset_cls, dataset_kwargs, batch_size=256, seed=42, num_workers=4):
    full_dataset = dataset_cls(**dataset_kwargs)

    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
