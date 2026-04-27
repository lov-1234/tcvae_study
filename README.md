# Beta-TCVAE on dSprites

This project is a small implementation and experiment around **disentangled representation learning** with a Beta-TCVAE. The reference paper, `Isolating Sources of Disentanglement in VAEs` by Chen et. al., is the main motivation: instead of training a standard VAE with one lumped KL penalty, we decompose the latent regularization into mutual information, total correlation, and dimension-wise KL terms.

The practical question is:

> Can we train a variational autoencoder whose latent dimensions separately capture the independent factors of variation in simple images?

## What We Are Doing

We train a convolutional VAE on binary sprite images and optimize it with the Beta-TCVAE objective:

```text
loss = reconstruction + alpha * MI + beta * TC + gamma * dimension-wise KL
```

The important term here is **total correlation (TC)**. Penalizing TC encourages the latent dimensions to become statistically independent from each other. In a successful run, one latent dimension might mostly control x-position, another y-position, another scale, another rotation, and so on.

The current experiment in `main.py` uses a fixed latent size as a starting point:

- A 10-dimensional latent space.
- A convolutional encoder and decoder.
- Binary cross-entropy reconstruction loss with logits.
- Beta-TCVAE weights `alpha=1`, `beta=15`, `gamma=1`.
- A 20-epoch warmup for the TC weight.
- Adam optimizer with learning rate `5e-4`.
- Up to 100 training epochs with early stopping.

One of the longer-term goals is to stop treating `latent_dim=10` as a manually chosen parameter and instead infer the useful latent dimensionality from the learned rate-distortion behavior of the model.

## Data

The dataset is **dSprites**, loaded from:

```text
dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
```

This file is expected to live in the project root next to `main.py`.

dSprites contains synthetic 64x64 binary images of simple 2D shapes. The images are generated from known independent factors:

- Color: fixed, white foreground on black background.
- Shape: square, ellipse, heart.
- Scale: multiple sizes.
- Orientation: multiple rotations.
- X position: horizontal location.
- Y position: vertical location.

Because the true generative factors are known and cleanly controlled, dSprites is useful for testing whether a model learns disentangled latent variables.

In this implementation, `datasets.py` loads the `imgs` array from the `.npz` file and returns each image as a single-channel tensor with shape:

```text
(1, 64, 64)
```

The dataset is split reproducibly with seed `42`:

- 80% training
- 10% validation
- 10% test

## Code Structure

- `main.py`: runs the full experiment: data loading, train/validation/test split, model setup, training, testing, and plot generation.
- `datasets.py`: defines the `DSpritesDataset` wrapper.
- `model.py`: defines the convolutional VAE encoder, decoder, and reparameterization logic.
- `loss.py`: implements the reconstruction loss, standard VAE KL, Beta-TCVAE decomposition, and per-dimension encoder KL.
- `helpers.py`: contains training loops, validation logic, early stopping, seeding, and data loaders.
- `plotters.py`: saves training curves, reconstruction examples, latent traversals, and KL-per-latent plots.
- `plots/tcvae/`: stores generated outputs from training and evaluation.

## Outputs

Running `main.py` produces:

- `plots/tcvae/checkpoints/best_paper_beta_vae.pt`: saved model checkpoint.
- `plots/tcvae/test_results/test_results_beta_tcvae.json`: final test loss components.
- `plots/tcvae/training_curves/`: train/validation curves for total loss, reconstruction, MI, TC, and dimension-wise KL.
- `plots/tcvae/kl_analysis/`: encoder KL per latent dimension, showing which latent units are active.
- `plots/tcvae/reconstructions/`: original images compared against reconstructions.
- `plots/tcvae/latent_traversals/`: generated images from sweeping each latent dimension.

The most important qualitative output is the latent traversal plot. If the model has learned a disentangled representation, changing one latent coordinate should change one interpretable property of the image while leaving the others mostly fixed.

## How To Run

Install the Python dependencies used by the scripts:

```bash
pip install torch numpy matplotlib einops tqdm
```

Make sure the dSprites `.npz` file is present in the project root:

```text
dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
```

Then run:

```bash
python main.py
```

The script will train the model, save the best checkpoint, evaluate on the test split, and write plots under `plots/tcvae/`.

## Rate-Distortion and Latent Dimension Selection

A useful way to interpret this experiment is through **rate-distortion theory**. In a VAE-like model, the distortion is the reconstruction cost, here binary cross-entropy. The rate is the information carried by the latent code, commonly estimated by the expected KL:

```text
R = E_x KL(q(z | x) || p(z))
D = E[-log p(x | z)]
```

From this view, a VAE is not only learning to reconstruct images. It is learning how much information must pass through the latent bottleneck to achieve a given reconstruction quality. Sweeping the regularization strength traces a rate-distortion curve: low rate gives compressed but blurry or inaccurate reconstructions; higher rate gives better reconstructions but uses more latent information.

This connects directly to the papers behind the project. Higgins et al. introduced beta-VAE as a constrained VAE where beta balances latent capacity, independence, and reconstruction accuracy. Burgess et al. then made the rate-distortion interpretation explicit and proposed gradually increasing latent capacity during training. Alemi et al. also used rate-distortion curves to show that models with similar ELBO values can have very different representations. Chen et al.'s Beta-TCVAE decomposition is especially useful here because the rate can be split into more interpretable pieces:

```text
expected KL = index-code MI + total correlation + dimension-wise KL
```

In this codebase, those terms correspond to:

- `MI`: how much information the latent code carries about each input.
- `TC`: how dependent the latent dimensions are on each other.
- `DWKL`: how far each aggregate latent dimension is from the prior.

The research direction is to use this decomposition to estimate an **effective intrinsic latent dimension**. The intuition is:

- If total correlation is small, latent axes are less redundant.
- If a latent dimension has near-zero per-dimension KL, it is effectively inactive.
- If a latent dimension has non-trivial KL and produces a stable semantic change in traversal, it is acting like a useful channel.
- Therefore, under a disentangling pressure, the number of active independent channels can act as an empirical upper bound on the intrinsic dimensionality needed by the model at a chosen distortion level.

For dSprites, this is a natural test bed because the data is generated from a small number of known factors: shape, scale, orientation, x-position, and y-position. A successful method should not need many more active latent dimensions than the number of independent factors needed to reconstruct the data. Extra dimensions should either remain inactive or be penalized away.

The algorithmic version of this idea would be:

1. Start with a generously large latent space, `K_max`, instead of trying to guess the correct size.
2. Train a family of Beta-TCVAE models across a sweep of beta values or explicit capacity targets.
3. For each run, record distortion, total rate, MI, TC, DWKL, and per-dimension encoder KL.
4. Plot the empirical rate-distortion curve `D(R)`.
5. Choose the operating point near the knee of the curve, where extra rate gives little distortion improvement.
6. Count active dimensions at that point using a threshold such as `KL_j > epsilon`.
7. Validate active dimensions qualitatively with latent traversals and, when factor labels are available, quantitatively with metrics such as MIG.

This would turn latent size selection into a model-selection problem rooted in rate-distortion theory: choose the smallest number of independent latent channels that achieves the desired reconstruction distortion. In a stronger version, the code could combine this with automatic relevance determination or a hierarchical prior so that unused latent dimensions are pruned during training rather than only counted afterward.

The important caveat is that purely unsupervised disentanglement is not identifiable without assumptions about the model and data. So the claim is not that rate-distortion alone proves the true number of generative factors. The claim is narrower and more testable: with a suitable inductive bias, TC regularization, and a controlled capacity sweep, we can estimate the number of useful independent latent channels required by the dataset.

## What We Hope To Achieve

By the end of this experiment, we want to show that Beta-TCVAE can learn a more disentangled latent representation than a plain VAE. Concretely, we hope to see:

- Good reconstructions of dSprites images.
- Several active latent dimensions instead of all information collapsing into one or two coordinates.
- Latent traversals where individual dimensions correspond to interpretable factors such as position, scale, rotation, or shape.
- Training curves that make the Beta-TCVAE decomposition visible, especially the behavior of the TC penalty.
- A path toward selecting the number of latent dimensions from the rate-distortion curve instead of fixing it by hand.

The final goal is not just a low reconstruction loss. The goal is an interpretable latent space where each coordinate has a cleaner semantic role.

## References

- [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
- [Understanding disentangling in beta-VAE](https://arxiv.org/abs/1804.03599)
- [Fixing a Broken ELBO](https://arxiv.org/abs/1711.00464)
- [Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1802.04942)
- [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://arxiv.org/abs/1811.12359)
- [ARD-VAE: A Statistical Formulation to Find the Relevant Latent Dimensions of Variational Autoencoders](https://arxiv.org/abs/2501.10901)
