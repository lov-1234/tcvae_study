import math
import torch
import torch.nn.functional as F
from einops import rearrange


# Basic density helpers
def log_density_gaussian(z, mu, logvar):
    """
    Elementwise log-density of a diagonal Gaussian.

    z, mu, logvar must be broadcastable to the same shape.
    Returns per-dimension log density.
    """
    return -0.5 * (
        math.log(2 * math.pi) + logvar + (z - mu).pow(2) / logvar.exp()
    )


def log_density_standard_normal(z):
    """
    Elementwise log-density under standard normal N(0,1).
    """
    return -0.5 * (math.log(2 * math.pi) + z.pow(2))



# Reconstruction + standard VAE KL
def recon_loss(x, x_hat, logits=True):
    """
    Returns mean reconstruction loss over the batch,
    summed over non-batch dimensions.
    """
    if logits:
        return F.binary_cross_entropy_with_logits(
            x_hat, x, reduction="none"
        ).sum(dim=(1, 2, 3)).mean()
    else:
        return F.binary_cross_entropy(
            x_hat, x, reduction="none"
        ).sum(dim=(1, 2, 3)).mean()


def kl_loss(mu, logvar):
    """
    Standard VAE KL(q(z|x) || p(z)) with p(z)=N(0,I),
    averaged over the batch and summed over latent dims.
    """
    kl_per_sample = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    return kl_per_sample.mean()


def vae_loss(x, x_hat, mu, logvar, beta=1.0, logits=True):
    reconstruction_loss = recon_loss(x, x_hat, logits=logits)
    kl_loss_term = kl_loss(mu, logvar)
    loss = reconstruction_loss + beta * kl_loss_term
    return loss, reconstruction_loss, kl_loss_term


# Shared β-TCVAE latent stats
def tcvae_latent_log_terms(mu, logvar, z, dataset_size):
    """
    Compute the shared log-density quantities needed for MI, TC, and dim-wise KL.

    Inputs:
        mu:      (B, D)
        logvar:  (B, D)
        z:       (B, D)

    Returns:
        dict containing:
            log_q_z_given_x : (B,)    = log q(z_i | x_i)
            log_q_z         : (B,)    = log q(z_i)
            log_prod_q_z    : (B,)    = sum_d log q(z_{i,d})
            log_p_z         : (B,)    = sum_d log p(z_{i,d})
            log_q_zij       : (B, B)  = log q(z_i | x_j)
            log_q_zij_per_dim : (B,B,D)
    """
    B, D = z.shape

    z_i = rearrange(z, "i d -> i 1 d")            # (B, 1, D)
    mu_j = rearrange(mu, "j d -> 1 j d")          # (1, B, D)
    logvar_j = rearrange(logvar, "j d -> 1 j d")  # (1, B, D)

    # [i, j, d] = log q(z_{i,d} | x_j)
    log_q_zij_per_dim = log_density_gaussian(z_i, mu_j, logvar_j)   # (B, B, D)

    # [i, j] = log q(z_i | x_j)
    log_q_zij = log_q_zij_per_dim.sum(dim=-1)    # (B, B)

    # log q(z_i | x_i)
    log_q_z_given_x = log_q_zij.diagonal()       # (B,)

    # log q(z_i)
    log_q_z = torch.logsumexp(log_q_zij, dim=1) - \
        math.log(B)   # (B,)

    # log q(z_{i,d})
    log_q_z_per_dim = torch.logsumexp(
        log_q_zij_per_dim, dim=1) - math.log(B)  # (B, D)

    # sum_d log q(z_{i,d})
    log_prod_q_z = log_q_z_per_dim.sum(dim=1)    # (B,)

    # sum_d log p(z_{i,d}) for standard normal prior
    log_p_z = log_density_standard_normal(z).sum(dim=1)   # (B,)

    return {
        "log_q_z_given_x": log_q_z_given_x,
        "log_q_z": log_q_z,
        "log_prod_q_z": log_prod_q_z,
        "log_p_z": log_p_z,
        "log_q_zij": log_q_zij,
        "log_q_zij_per_dim": log_q_zij_per_dim,
    }

# β-TCVAE decomposed terms

def mi_term_from_shared(shared):
    return (shared["log_q_z_given_x"] - shared["log_q_z"]).mean()


def tc_term_from_shared(shared):
    return (shared["log_q_z"] - shared["log_prod_q_z"]).mean()


def dimwise_kl_term_from_shared(shared):
    return (shared["log_prod_q_z"] - shared["log_p_z"]).mean()


def tc_terms(x, x_hat, mu, logvar, z, dataset_size, logits=True):
    """
    Returns the decomposed β-TCVAE terms:
        distortion, MI, TC, dim-wise KL
    """
    distortion = recon_loss(x, x_hat, logits=logits)

    shared = tcvae_latent_log_terms(mu, logvar, z, dataset_size)

    mi = mi_term_from_shared(shared)
    tc = tc_term_from_shared(shared)
    dwkl = dimwise_kl_term_from_shared(shared)

    return distortion, mi, tc, dwkl


# Final β-TCVAE loss
def beta_tcvae_loss(
    x,
    x_hat,
    mu,
    logvar,
    z,
    dataset_size,
    alpha=1.0,
    beta=6.0,
    gamma=1.0,
    logits=True
):
    distortion = recon_loss(x, x_hat, logits=logits)

    shared = tcvae_latent_log_terms(mu, logvar, z, dataset_size)

    mi = mi_term_from_shared(shared)
    tc = tc_term_from_shared(shared)
    dwkl = dimwise_kl_term_from_shared(shared)

    loss = distortion + alpha * mi + beta * tc + gamma * dwkl
    return loss, distortion, mi, tc, dwkl


def encoder_kl_per_dim(mu, logvar):
    """
    Returns average encoder KL per latent dimension for the batch.
    Shape: (D,)
    """
    kl_per_dim_batch = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, D)
    return kl_per_dim_batch.mean(dim=0)  # (D,)
