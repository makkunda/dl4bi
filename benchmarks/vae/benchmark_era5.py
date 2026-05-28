#!/usr/bin/env python3
"""benchmark_era5.py

FM-DeepRV vs DeepRV on real-world ERA5 2m temperature fields.

Tests whether FM-DeepRV better captures the non-Gaussian, spatially structured
distribution of ERA5 temperature compared to a single MSE-trained forward pass
(DeepRV).  The setup mirrors benchmark_cifar.py but uses real geospatial data
instead of natural image patches.

Setup:
  - Source    : z ~ N(0, I)  (256-dim, one per 16×16 spatial patch)
  - Target    : standardised ERA5 2m temperature patch (16×16 at 0.25° resolution)
  - Loss      : OT-CFM for FM-DeepRV, MSE for DeepRV
  - Inference : observe 30% of grid points, Gaussian likelihood,
                HMC inpaints the remaining 70%

Regions (matching meta_learning/era5.py):
  - Train surrogate : central_europe  (lat 42–53, lng  8–28)
  - Validate        : northern_europe (lat 53–62, lng  8–28)
  - Test HMC        : western_europe  (lat 42–53, lng -4– 8)

Data requirement:
  ERA5 netCDF files must be pre-downloaded into cache/era5/{region}/2019_*.nc.
  Run  `download_if_not_cached()`  from benchmarks/meta_learning/era5.py, or
  follow the CDS API instructions in the project README.

Run from the repo root:
    uv run python benchmarks/vae/benchmark_era5.py
"""

import os
import sys
sys.path.append("benchmarks/vae")
sys.path.append("benchmarks/meta_learning")

# Initialise JAX / CUDA before anything else touches the GPU.
import jax
import jax.numpy as jnp
from jax import Array, jit, random
jax.devices()  # force CUDA initialisation now

import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import arviz as az
import flax.linen as nn
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import optax
import scoringrules as sr
from jax.scipy import linalg as jsp_linalg
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
from scipy.stats import norm as scipy_norm_dist
import pandas as pd
import xarray as xr
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from omegaconf import DictConfig
from orbax.checkpoint import PyTreeCheckpointer
from dl4bi_sps.utils import build_grid

import wandb
from dl4bi.attention import BiasedScanAttention, MultiHeadAttention
from dl4bi.bias import Bias
from dl4bi.core.data import Batch
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import (
    TrainState,
    cosine_annealing_lr,
    estimate_flops,
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning import BSATNP
from dl4bi.meta_learning.steps import likelihood_train_step, likelihood_valid_step
from dl4bi.meta_learning.utils import x_to_none
from dl4bi.mlp import MLP
from dl4bi.transformer import KRBlock
from dl4bi.vae import FlowMatchingDeepRV, FlowMatchingVectorField, gMLPDeepRV
from dl4bi.vae.train_utils import (
    deep_rv_train_step,
    flow_matching_train_step,
    flow_matching_valid_step,
    generate_surrogate_decoder,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PATCH_SIZE   = 30          # 30×30 spatial grid → L = 900 locations (matches meta_learning/era5.py)
N_TRAIN      = 50_000      # patches sampled from training region
N_TEST       = 20          # test patches to run HMC on
# Context size range matching meta_learning/era5.py (num_ctx_min/max_per_t)
N_CTX_MIN    = 45          # ~5%  of 900 grid points
N_CTX_MAX    = 225         # ~25% of 900 grid points
OBS_NOISE    = 0.1         # Gaussian likelihood sigma on standardised temperature
TRAIN_STEPS  = 500_000
VALID_INTERVAL = 50_000
VALID_STEPS  = 2_000
BATCH_SIZE   = 64
MAX_LR       = 1e-3
HMC_WARMUP   = 1_000
HMC_SAMPLES  = 1_000
HMC_CHAINS   = 2
FM_K_STEPS   = [1, 3, 5, 7]
N_BLOCKS     = 4

TRAIN_REGION = "central_europe"
VALID_REGION = "northern_europe"
TEST_REGION  = "western_europe"
ERA5_CACHE   = Path("cache/era5")

# BSA-TNP config (matches meta_learning/era5.py)
BSA_BATCH_SIZE    = 8
BSA_TRAIN_STEPS   = 100_000
BSA_VALID_STEPS   = 500
BSA_VALID_INTERVAL = 10_000

# SVGP config — fit independently per task (non-amortised)
SVGP_NUM_INDUCING = 100     # inducing points (≤ n_ctx)
SVGP_LR           = 0.01
SVGP_NUM_STEPS    = 1_000   # SVI steps via jax.lax.scan
SVGP_INIT_AMP     = 1.0
SVGP_INIT_LS      = 1.0     # in standardised lat/lng units
SVGP_INIT_NOISE   = 0.1
SVGP_JITTER       = 1e-4


# ---------------------------------------------------------------------------
# Batch container for BSA-TNP (spatial-only, no fixed effects)
# ---------------------------------------------------------------------------

@dataclass
class SpatialBatch(Batch):
    """Minimal ctx/test batch consumed by likelihood_train/valid_step."""
    x_ctx:    Optional[Array] = None
    s_ctx:    Optional[Array] = None     # [B, L_ctx, 2]
    t_ctx:    Optional[Array] = None     # [B, L_ctx, 1]  constant zero
    f_ctx:    Optional[Array] = None     # [B, L_ctx, 1]
    mask_ctx: Optional[Array] = None     # [B, L_ctx]
    x_test:   Optional[Array] = None
    s_test:   Optional[Array] = None     # [B, L,     2]
    t_test:   Optional[Array] = None     # [B, L,     1]  constant zero
    f_test:   Optional[Array] = None     # [B, L,     1]
    mask_test: Optional[Array] = None    # [B, L]


# ---------------------------------------------------------------------------
# Valid step for gMLPDeepRV
# ---------------------------------------------------------------------------

@jit
def deep_rv_valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}


# ---------------------------------------------------------------------------
# Data — ERA5 temperature patches
# ---------------------------------------------------------------------------

def load_era5_xr(region: str) -> xr.Dataset:
    """Load ERA5 netCDF for one region (2019, all months) without dask."""
    paths = sorted((ERA5_CACHE / region).glob("2019_*.nc"))
    if not paths:
        raise FileNotFoundError(f"No ERA5 files found in {ERA5_CACHE / region}")
    datasets = [xr.open_dataset(p) for p in paths]
    ds = xr.concat(datasets, dim="valid_time")
    ds = ds.rename({"valid_time": "time", "z": "elevation", "t2m": "temperature"})
    return ds.load()


def extract_patches(
    ds: xr.Dataset,
    patch_size: int = PATCH_SIZE,
    n_patches: int = N_TRAIN,
    seed: int = 0,
    temp_mean: Optional[float] = None,
    temp_std: Optional[float] = None,
    elev_mean: Optional[float] = None,
    elev_std: Optional[float] = None,
) -> tuple:
    """
    Randomly sample `n_patches` spatial windows from the ERA5 temperature and
    elevation fields (single time step per temperature patch; elevation is
    time-invariant so we use the first snapshot).

    Returns:
        temp_patches : float32 [n_patches, patch_size²], standardised temperature
        elev_patches : float32 [n_patches, patch_size²], standardised elevation
        temp_mean, temp_std, elev_mean, elev_std
    """
    temp = ds["temperature"]  # (time, lat, lng)
    elev = ds["elevation"]    # (time, lat, lng) — time-invariant in ERA5
    n_times = len(temp.time)
    n_lat   = len(temp.latitude)
    n_lng   = len(temp.longitude)
    assert n_lat >= patch_size and n_lng >= patch_size, (
        f"Region too small for {patch_size}×{patch_size} patches: "
        f"grid is {n_lat}×{n_lng}"
    )

    if temp_mean is None:
        temp_mean = float(temp.values.mean())
    if temp_std is None:
        temp_std  = float(temp.values.std())
    if elev_mean is None:
        elev_mean = float(elev.values.mean())
    if elev_std is None:
        elev_std  = float(elev.values.std())

    rng = np.random.default_rng(seed)
    t_idxs   = rng.integers(0, n_times,            size=n_patches)
    lat_idxs = rng.integers(0, n_lat - patch_size, size=n_patches)
    lng_idxs = rng.integers(0, n_lng - patch_size, size=n_patches)

    temp_np = temp.values.astype("float32")          # (T, lat, lng)
    elev_np = elev.values[0].astype("float32")       # (lat, lng) — first time step

    temp_patches = np.stack([
        temp_np[t_idxs[i],
                lat_idxs[i]:lat_idxs[i] + patch_size,
                lng_idxs[i]:lng_idxs[i] + patch_size].flatten()
        for i in range(n_patches)
    ], axis=0)
    elev_patches = np.stack([
        elev_np[lat_idxs[i]:lat_idxs[i] + patch_size,
                lng_idxs[i]:lng_idxs[i] + patch_size].flatten()
        for i in range(n_patches)
    ], axis=0)

    temp_patches = ((temp_patches - temp_mean) / temp_std).astype("float32")
    elev_patches = ((elev_patches - elev_mean) / elev_std).astype("float32")
    return temp_patches, elev_patches, temp_mean, temp_std, elev_mean, elev_std


def build_patch_grid(patch_size: int = PATCH_SIZE) -> Array:
    """
    Spatial coordinates for a patch_size × patch_size grid.

    At PATCH_SIZE=30 and 0.25° ERA5 resolution this is a 7.25°×7.25° window,
    matching the H_deg=7.5, W_deg=7.5 windows used in meta_learning/era5.py.
    Coordinates are standardised to zero mean / unit std.

    Returns shape [patch_size², 2].
    """
    step = 0.25  # ERA5 resolution in degrees
    coords_1d = np.arange(patch_size, dtype=np.float32) * step  # [0, 0.25, …, 7.25]
    lng_grid, lat_grid = np.meshgrid(coords_1d, coords_1d)      # both [P, P]
    s = np.stack([lat_grid.flatten(), lng_grid.flatten()], axis=-1)  # [L, 2]
    s = (s - s.mean(axis=0)) / s.std(axis=0)
    return jnp.array(s)


def gen_obs_mask(rng: Array, n_pixels: int, n_ctx: int) -> Array:
    """Random mask with exactly n_ctx observed locations."""
    idx = random.permutation(rng, n_pixels)[:n_ctx]
    return jnp.zeros(n_pixels, dtype=bool).at[idx].set(True)


def gen_train_dataloader(patches: Array, s: Array, batch_size: int = BATCH_SIZE):
    N = patches.shape[0]

    def dataloader(rng_data):
        while True:
            rng_data, rng_idx, rng_z = random.split(rng_data, 3)
            idx = random.choice(rng_idx, N, shape=(batch_size,), replace=False)
            f = patches[idx]
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s.shape[0]))
            yield {
                "s": s,
                "z": z,
                "conditionals": jnp.array([0.0]),   # no explicit GP hyperparameters
                "f": f,
            }

    return dataloader


# ---------------------------------------------------------------------------
# Inference model — Gaussian inpainting with surrogate prior
# ---------------------------------------------------------------------------

def build_surrogate_inpainting_model(s: Array) -> Callable:
    """z ~ N(0,I),  f = decoder(z),  y_obs ~ N(f[obs], sigma)."""
    surrogate_kwargs = {"s": s}

    def inpaint(surrogate_decoder=None, obs_mask=None, y=None):
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))
        if surrogate_decoder is None:
            f = z[0]
        else:
            f = surrogate_decoder(
                z, jnp.array([0.0]), **surrogate_kwargs
            ).squeeze()
        numpyro.deterministic("f", f)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Normal(f, OBS_NOISE), obs=y)

    return inpaint


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def surrogate_model_train(
    rng_train: Array,
    rng_test: Array,
    loader: Callable,
    train_step: Callable,
    valid_step: Callable,
    model: nn.Module,
    results_dir: Path,
    optimizer,
) -> tuple:
    flop_batch = next(loader(rng_train))
    rngs = {"params": rng_train, "extra": rng_test}
    kwargs = model.init(rngs, **flop_batch)
    params = kwargs.pop("params")
    state = TrainState.create(apply_fn=model.apply, params=params, kwargs=kwargs, tx=optimizer)
    infer_flops, train_flops = estimate_flops(rng_train, state, train_step, flop_batch)
    parameters = nn.tabulate(model, rngs)(**flop_batch)
    parameters = int(
        parameters.split("Total Parameters: ")[-1].split(" ")[0].replace(",", "")
    )
    t0 = datetime.now()
    state = train(
        rng_train, model, optimizer, train_step, TRAIN_STEPS, loader,
        valid_step, VALID_INTERVAL, VALID_STEPS, loader,
        return_state="best", valid_monitor_metric="norm MSE",
    )
    train_time = (datetime.now() - t0).total_seconds()
    eval_mse = evaluate(rng_test, state, valid_step, loader, VALID_STEPS)["norm MSE"]
    save_ckpt(state, DictConfig({}), results_dir / "model.ckpt")
    return train_time, eval_mse, state, infer_flops, train_flops, parameters


def reload_state(ckpt_dir: Path, model: nn.Module, s: Array, optimizer) -> TrainState:
    """Restore weights from a saved checkpoint into a fresh TrainState."""
    L = s.shape[0]
    dummy_batch = {
        "s": s,
        "z": jnp.ones((1, L)),
        "conditionals": jnp.array([0.0]),
        "f": jnp.ones((1, L)),
    }
    rngs = {"params": random.key(0), "extra": random.key(1)}
    init_vars = model.init(rngs, **dummy_batch)
    init_params = init_vars.pop("params")
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        state_template = TrainState.create(
            apply_fn=model.apply,
            params=init_params,
            kwargs=init_vars,
            tx=optimizer,
        )
    ckptr = PyTreeCheckpointer()
    ckpt = ckptr.restore(ckpt_dir.absolute(), item={"state": state_template, "config": {}})
    return ckpt["state"]


# ---------------------------------------------------------------------------
# HMC inpainting
# ---------------------------------------------------------------------------

def run_hmc_inpaint(
    rng: Array,
    inpaint_model: Callable,
    y_obs: Array,
    obs_mask: Array,
    surrogate_decoder: Callable,
):
    nuts = NUTS(inpaint_model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    mcmc = MCMC(
        nuts, num_chains=HMC_CHAINS, num_samples=HMC_SAMPLES, num_warmup=HMC_WARMUP
    )
    t0 = datetime.now()
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=y_obs)
    infer_time = (datetime.now() - t0).total_seconds()
    samples = mcmc.get_samples()
    post = Predictive(inpaint_model, samples)(
        k2, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask
    )
    return samples, mcmc, post, infer_time


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mean_ess_z(mcmc) -> float:
    ess = az.ess(mcmc, method="mean", var_names=["z"])
    return float(ess["z"].values.mean())


def mean_rhat_z(samples_by_chain: dict) -> float:
    idata = az.convert_to_inference_data(
        {k: np.array(v) for k, v in samples_by_chain.items()}
    )
    rhat = az.rhat(idata, var_names=["z"])
    return float(rhat["z"].values.mean())


def patch_metrics(true_f, post_f_mean, post_f_samples, obs_mask, hdi_prob=0.9):
    """RMSE, MAE, CRPS, Coverage, NLL at target locations; RMSE at ctx and all.

    post_f_samples : [S, L] array of HMC posterior predictive draws.
    """
    err  = true_f - post_f_mean
    sq   = err ** 2
    rmse_all    = float(np.sqrt(sq.mean()))
    rmse_ctx    = float(np.sqrt(sq[obs_mask].mean()))
    rmse_target = float(np.sqrt(sq[~obs_mask].mean()))
    mae_target  = float(np.abs(err[~obs_mask]).mean())

    # Calibration metrics at target locations only
    samples_target = post_f_samples[:, ~obs_mask]   # [S, n_target]
    true_target    = true_f[~obs_mask]

    # CRPS from ensemble samples
    crps_target = float(sr.crps_ensemble(true_target, samples_target.T).mean())

    # 90% credible interval coverage
    alpha = 1.0 - hdi_prob
    lower = np.percentile(samples_target, 100 * alpha / 2,     axis=0)
    upper = np.percentile(samples_target, 100 * (1 - alpha/2), axis=0)
    coverage = float(((true_target >= lower) & (true_target <= upper)).mean())

    # NLL via Gaussian approximation (mean + sample std)
    f_std_target = samples_target.std(axis=0)
    nll_target = float(
        -scipy_norm_dist.logpdf(
            true_target, post_f_mean[~obs_mask], np.clip(f_std_target, 1e-6, None)
        ).mean()
    )

    return rmse_all, rmse_ctx, rmse_target, mae_target, crps_target, coverage, nll_target


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_reconstructions(
    patch_size, true_patches, masked_patches, recon_means,
    model_names, obs_masks, save_path,
):
    n_imgs = len(true_patches)
    n_cols = 2 + len(model_names)
    fig, axes = plt.subplots(
        n_imgs, n_cols, figsize=(3 * n_cols, 3 * n_imgs), constrained_layout=True
    )
    if n_imgs == 1:
        axes = axes[None]

    col_titles = ["true", "context"] + model_names
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9)

    for i in range(n_imgs):
        true  = np.array(true_patches[i]).reshape(patch_size, patch_size)
        mask_2d = np.array(obs_masks[i]).reshape(patch_size, patch_size)
        masked  = np.ma.masked_where(~mask_2d, true)
        vmin, vmax = true.min(), true.max()

        axes[i, 0].imshow(true,   cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[i, 1].imshow(masked, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        for j, recon in enumerate(recon_means[i]):
            axes[i, 2 + j].imshow(
                np.array(recon).reshape(patch_size, patch_size),
                cmap="RdBu_r", vmin=vmin, vmax=vmax,
            )

    for ax in axes.flatten():
        ax.axis("off")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# SVGP — spatial Matérn-3/2, fit per task via SVI
# ---------------------------------------------------------------------------

def _sq_dist_ard(x_a: Array, x_b: Array, ls: Array) -> Array:
    x_a_s = x_a / ls
    x_b_s = x_b / ls
    return jnp.sum((x_a_s[:, None, :] - x_b_s[None, :, :]) ** 2, axis=-1)


def _matern32(x_a: Array, x_b: Array, amp: Array, ls: Array) -> Array:
    r = jnp.sqrt(jnp.clip(_sq_dist_ard(x_a, x_b, ls), 1e-12))
    sr3 = jnp.sqrt(3.0) * r
    return amp * (1.0 + sr3) * jnp.exp(-sr3)


def _svgp_model(x, y, z_init, jitter=SVGP_JITTER):
    amp   = numpyro.param("amp",   jnp.array(SVGP_INIT_AMP),          constraint=constraints.positive)
    ls    = numpyro.param("ls",    jnp.ones(2) * SVGP_INIT_LS,        constraint=constraints.positive)
    noise = numpyro.param("noise", jnp.array(SVGP_INIT_NOISE),        constraint=constraints.positive)
    z = z_init
    k_zz  = _matern32(z, z, amp, ls) + jitter * jnp.eye(z.shape[0])
    chol  = jnp.linalg.cholesky(k_zz)
    u     = numpyro.sample("u", dist.MultivariateNormal(jnp.zeros(z.shape[0]), scale_tril=chol))
    alpha = jsp_linalg.cho_solve((chol, True), u)
    k_xz  = _matern32(x, z, amp, ls)
    proj  = jsp_linalg.solve_triangular(chol, k_xz.T, lower=True)
    mean  = k_xz @ alpha
    var   = jnp.full(x.shape[0], amp) - jnp.sum(proj ** 2, axis=0) + noise ** 2
    numpyro.sample("obs", dist.Normal(mean, jnp.sqrt(jnp.clip(var, jitter))), obs=y)


def _svgp_guide(x, y, z_init, jitter=SVGP_JITTER):
    del x, y, jitter
    m   = z_init.shape[0]
    loc = numpyro.param("u_loc", jnp.zeros(m))
    tril = numpyro.param("u_scale_tril", 1e-2 * jnp.eye(m), constraint=constraints.lower_cholesky)
    numpyro.sample("u", dist.MultivariateNormal(loc=loc, scale_tril=tril))


def _svgp_predict(x_test, params, z_init, jitter=SVGP_JITTER):
    amp, ls, noise = params["amp"], params["ls"], params["noise"]
    u_loc, u_tril  = params["u_loc"], params["u_scale_tril"]
    z = z_init
    k_zz  = _matern32(z, z, amp, ls) + jitter * jnp.eye(z.shape[0])
    chol  = jnp.linalg.cholesky(k_zz)
    k_tz  = _matern32(x_test, z, amp, ls)
    alpha = jsp_linalg.cho_solve((chol, True), u_loc)
    proj  = jsp_linalg.solve_triangular(chol, k_tz.T, lower=True)
    mean  = k_tz @ alpha
    prior_diag   = jnp.full(x_test.shape[0], amp)
    var_sparse   = prior_diag - jnp.sum(proj ** 2, axis=0)
    prec_proj    = jsp_linalg.cho_solve((chol, True), k_tz.T).T
    s            = u_tril @ u_tril.T
    var_q        = jnp.sum((prec_proj @ s) * prec_proj, axis=1)
    std          = jnp.sqrt(jnp.clip(var_sparse + var_q + noise ** 2, jitter))
    return mean, std


def eval_svgp(
    rng: Array,
    s: Array,
    elev_patch: Array,
    obs_mask: Array,
    true_f: Array,
    y_obs: Array,
    hdi_prob: float = 0.9,
) -> tuple:
    """Fit spatial SVGP on context (y_obs at obs_mask) and predict everywhere.

    Features: [lat_std, lng_std, elev_std] — 3-dimensional ARD Matérn-3/2,
    matching the weather kernel structure (minus the time/diurnal terms which
    are constant within a single-time-step patch).
    """
    feats  = jnp.concatenate([s, elev_patch[:, None]], axis=-1)  # [L, 3]
    x_ctx  = np.array(feats[obs_mask])
    y_ctx  = np.array(y_obs[obs_mask])
    x_test = np.array(feats)
    true   = np.array(true_f)
    mask   = np.array(obs_mask)

    rng_z, rng_svi = random.split(rng)
    n_ind  = min(SVGP_NUM_INDUCING, x_ctx.shape[0])
    idx    = random.choice(rng_z, x_ctx.shape[0], (n_ind,), replace=False)
    z_init = jnp.array(x_ctx[idx])

    svi   = SVI(_svgp_model, _svgp_guide, Adam(SVGP_LR), Trace_ELBO())
    t0    = datetime.now()
    state = svi.init(rng_svi, jnp.array(x_ctx), jnp.array(y_ctx), z_init)

    def update(st, _):
        return svi.update(st, jnp.array(x_ctx), jnp.array(y_ctx), z_init)[0], None

    state, _ = jax.lax.scan(update, state, None, length=SVGP_NUM_STEPS)
    params = svi.get_params(state)
    mu, std = _svgp_predict(jnp.array(x_test), params, z_init)
    infer_time = (datetime.now() - t0).total_seconds()

    mu  = np.array(mu)
    std = np.clip(np.array(std), 1e-6, None)

    rmse_ctx    = float(np.sqrt(np.mean((true[mask]  - mu[mask])  ** 2)))
    rmse_target = float(np.sqrt(np.mean((true[~mask] - mu[~mask]) ** 2)))
    rmse_all    = float(np.sqrt(np.mean((true        - mu)        ** 2)))
    mae_target  = float(np.mean(np.abs(true[~mask] - mu[~mask])))
    nll_target  = float(-scipy_norm_dist.logpdf(true[~mask], mu[~mask], std[~mask]).mean())
    crps_target = float(sr.crps_normal(true[~mask], mu[~mask], std[~mask]).mean())
    alpha_ci    = 1.0 - hdi_prob
    z_ci        = scipy_norm_dist.ppf(1 - alpha_ci / 2)
    lower, upper = mu[~mask] - z_ci * std[~mask], mu[~mask] + z_ci * std[~mask]
    coverage    = float(((true[~mask] >= lower) & (true[~mask] <= upper)).mean())

    return mu, std, rmse_ctx, rmse_target, rmse_all, mae_target, crps_target, coverage, nll_target, infer_time


# ---------------------------------------------------------------------------
# BSA-TNP model, dataloader, training, and evaluation
# ---------------------------------------------------------------------------

def make_bsa_tnp() -> BSATNP:
    """Instantiate BSA-TNP matching the ERA5 config from era5_bsa_vs_svgp."""
    return BSATNP(
        num_blks=6,
        num_reps=1,
        embed_s=x_to_none,
        embed_t=x_to_none,
        embed_all=MLP([256, 128, 64], nn.gelu),
        blk=KRBlock(
            attn=MultiHeadAttention(
                attn=BiasedScanAttention(
                    bias={
                        "s": Bias.build_rbf_network_bias(num_heads=4, num_basis=5),
                        "t": Bias.build_rbf_network_bias(num_heads=4, num_basis=3),
                    }
                ),
                proj_qs=MLP([128]),
                proj_ks=MLP([128]),
                proj_vs=MLP([128]),
                proj_out=MLP([64]),
            ),
            ffn=MLP([256, 64], nn.gelu),
        ),
        head=MLP([256, 64, 2], nn.gelu),
    )


def gen_bsa_dataloader(
    patches: Array, elev_patches: Array, s: Array, batch_size: int = BSA_BATCH_SIZE
):
    """Dataloader for BSA-TNP: samples random ctx/test splits per batch.

    elevation is passed as x (fixed effects) so BSA-TNP can use it as a
    covariate in addition to the spatial attention bias over s.
    """
    N, L = patches.shape

    def dataloader(rng):
        while True:
            rng, rng_idx, rng_ctx, rng_perm = random.split(rng, 4)
            idx = random.choice(rng_idx, N, shape=(batch_size,), replace=False)
            f_all    = patches[idx]       # [B, L]
            elev_all = elev_patches[idx]  # [B, L]
            n_ctx = int(
                random.randint(rng_ctx, shape=(), minval=N_CTX_MIN, maxval=N_CTX_MAX + 1)
            )
            perm     = random.permutation(rng_perm, L)
            ctx_idxs = perm[:n_ctx]
            s_ctx  = jnp.broadcast_to(s[None, ctx_idxs], (batch_size, n_ctx, 2))
            s_test = jnp.broadcast_to(s[None],            (batch_size, L,     2))
            yield SpatialBatch(
                x_ctx    = elev_all[:, ctx_idxs, None],   # [B, n_ctx, 1]
                s_ctx    = s_ctx,
                t_ctx    = jnp.zeros((batch_size, n_ctx, 1)),
                f_ctx    = f_all[:, ctx_idxs, None],
                mask_ctx = jnp.ones((batch_size, n_ctx), dtype=bool),
                x_test   = elev_all[:, :, None],           # [B, L, 1]
                s_test   = s_test,
                t_test   = jnp.zeros((batch_size, L, 1)),
                f_test   = f_all[:, :, None],
                mask_test = jnp.ones((batch_size, L), dtype=bool),
            )

    return dataloader


def bsa_model_train(
    rng_train: Array,
    rng_valid: Array,
    train_loader: Callable,
    valid_loader: Callable,
    model: BSATNP,
    results_dir: Path,
    optimizer,
) -> tuple:
    dummy = next(train_loader(rng_train))
    rngs  = {"params": rng_train, "extra": rng_valid, "dropout": rng_valid}
    kwargs = model.init(rngs, **dummy, training=False)
    params = kwargs.pop("params")
    state  = TrainState.create(apply_fn=model.apply, params=params, kwargs=kwargs, tx=optimizer)
    t0 = datetime.now()
    state = train(
        rng_train, model, optimizer,
        likelihood_train_step, BSA_TRAIN_STEPS, train_loader,
        likelihood_valid_step,  BSA_VALID_INTERVAL, BSA_VALID_STEPS, valid_loader,
        return_state="best", valid_monitor_metric="NLL",
    )
    train_time = (datetime.now() - t0).total_seconds()
    metrics = evaluate(rng_valid, state, likelihood_valid_step, valid_loader, BSA_VALID_STEPS)
    save_ckpt(state, DictConfig({}), results_dir / "model.ckpt")
    return train_time, metrics.get("NLL", float("nan")), state


def reload_bsa_state(ckpt_dir: Path, model: BSATNP, s: Array, optimizer) -> TrainState:
    """Restore BSA-TNP weights from a checkpoint."""
    L = s.shape[0]
    n_ctx = N_CTX_MIN
    dummy = SpatialBatch(
        x_ctx    = jnp.zeros((1, n_ctx, 1)),
        s_ctx    = jnp.zeros((1, n_ctx, 2)),
        t_ctx    = jnp.zeros((1, n_ctx, 1)),
        f_ctx    = jnp.zeros((1, n_ctx, 1)),
        mask_ctx = jnp.ones((1, n_ctx), dtype=bool),
        x_test   = jnp.zeros((1, L, 1)),
        s_test   = jnp.zeros((1, L, 2)),
        t_test   = jnp.zeros((1, L, 1)),
        f_test   = jnp.zeros((1, L, 1)),
        mask_test = jnp.ones((1, L), dtype=bool),
    )
    rngs = {"params": random.key(0), "extra": random.key(1), "dropout": random.key(2)}
    init_vars  = model.init(rngs, **dummy, training=False)
    init_params = init_vars.pop("params")
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        state_template = TrainState.create(
            apply_fn=model.apply, params=init_params, kwargs=init_vars, tx=optimizer
        )
    ckptr = PyTreeCheckpointer()
    ckpt  = ckptr.restore(ckpt_dir.absolute(), item={"state": state_template, "config": {}})
    return ckpt["state"]


def eval_bsa_tnp(
    rng: Array,
    state: TrainState,
    s: Array,
    elev_patch: Array,
    obs_mask: Array,
    true_f: Array,
    hdi_prob: float = 0.9,
) -> tuple:
    """Run BSA-TNP direct prediction; return metrics + infer_time."""
    L = s.shape[0]
    ctx_idxs = jnp.where(obs_mask, size=obs_mask.sum())[0]
    n_ctx = ctx_idxs.shape[0]
    f_ctx_vals = true_f[ctx_idxs]
    t0 = datetime.now()
    output = state.apply_fn(
        {"params": state.params, **state.kwargs},
        x_ctx    = elev_patch[None, ctx_idxs, None],  # [1, n_ctx, 1]
        s_ctx    = s[None, ctx_idxs],
        t_ctx    = jnp.zeros((1, n_ctx, 1)),
        f_ctx    = f_ctx_vals[None, :, None],
        mask_ctx = jnp.ones((1, n_ctx), dtype=bool),
        x_test   = elev_patch[None, :, None],          # [1, L, 1]
        s_test   = s[None],
        t_test   = jnp.zeros((1, L, 1)),
        training = False,
        rngs     = {"extra": rng},
    )
    infer_time = (datetime.now() - t0).total_seconds()
    if isinstance(output, tuple):
        output, _ = output
    f_pred = np.array(output.mu[0, :, 0])   # [L]
    f_std  = np.array(output.std[0, :, 0])  # [L]
    true   = np.array(true_f)
    mask   = np.array(obs_mask)
    rmse_ctx    = float(np.sqrt(np.mean((true[mask]  - f_pred[mask])  ** 2)))
    rmse_target = float(np.sqrt(np.mean((true[~mask] - f_pred[~mask]) ** 2)))
    rmse_all    = float(np.sqrt(np.mean((true        - f_pred)        ** 2)))
    mae_target  = float(np.mean(np.abs(true[~mask] - f_pred[~mask])))
    std_target  = np.clip(f_std[~mask], 1e-6, None)
    # Analytical metrics for Gaussian predictive
    nll_target  = float(-scipy_norm_dist.logpdf(true[~mask], f_pred[~mask], std_target).mean())
    crps_target = float(sr.crps_normal(true[~mask], f_pred[~mask], std_target).mean())
    alpha = 1.0 - hdi_prob
    lower = f_pred[~mask] - scipy_norm_dist.ppf(1 - alpha / 2) * std_target
    upper = f_pred[~mask] + scipy_norm_dist.ppf(1 - alpha / 2) * std_target
    coverage = float(((true[~mask] >= lower) & (true[~mask] <= upper)).mean())
    return f_pred, f_std, rmse_ctx, rmse_target, rmse_all, mae_target, crps_target, coverage, nll_target, infer_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SCALAR_KEYS = {
    "model_name", "patch_idx", "n_ctx", "infer_time",
    "RMSE (all)", "RMSE (ctx)", "RMSE (target)",
    "MAE (target)", "CRPS (target)", "Coverage 90%", "NLL (target)",
    "mean ESS z", "mean r_hat z",
}


def main(seed: int = 42):
    rng = random.key(seed)
    save_dir = Path("results/era5_benchmark/").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load ERA5 data
    # ------------------------------------------------------------------
    print("Loading ERA5 data...")
    if not (ERA5_CACHE / TRAIN_REGION).exists():
        raise FileNotFoundError(
            f"ERA5 data not found at {ERA5_CACHE / TRAIN_REGION}.\n"
            "Please download it first:\n"
            "  from benchmarks.meta_learning.era5 import download_if_not_cached\n"
            "  download_if_not_cached()"
        )

    ds_train = load_era5_xr(TRAIN_REGION)
    ds_valid = load_era5_xr(VALID_REGION)
    ds_test  = load_era5_xr(TEST_REGION)

    print(f"  Train region: {TRAIN_REGION}, grid {len(ds_train.latitude)}×{len(ds_train.longitude)}, {len(ds_train.time)} timesteps")
    print(f"  Valid region: {VALID_REGION}")
    print(f"  Test  region: {TEST_REGION}")

    print("Extracting patches...")
    train_patches, train_elev, temp_mean, temp_std, elev_mean, elev_std = extract_patches(
        ds_train, patch_size=PATCH_SIZE, n_patches=N_TRAIN, seed=seed
    )
    valid_patches, valid_elev, _, _, _, _ = extract_patches(
        ds_valid, patch_size=PATCH_SIZE, n_patches=10_000, seed=seed + 1,
        temp_mean=temp_mean, temp_std=temp_std, elev_mean=elev_mean, elev_std=elev_std,
    )
    test_patches, test_elev, _, _, _, _ = extract_patches(
        ds_test, patch_size=PATCH_SIZE, n_patches=max(N_TEST * 10, 500), seed=seed + 2,
        temp_mean=temp_mean, temp_std=temp_std, elev_mean=elev_mean, elev_std=elev_std,
    )
    print(f"  train patches: {train_patches.shape}, valid: {valid_patches.shape}, test: {test_patches.shape}")
    print(f"  Temperature stats: mean={temp_mean:.2f} K, std={temp_std:.2f} K")
    print(f"  Elevation stats:   mean={elev_mean:.1f} m²/s², std={elev_std:.1f} m²/s²")

    s = build_patch_grid()
    L = s.shape[0]
    print(f"  Spatial grid: {PATCH_SIZE}×{PATCH_SIZE} = {L} locations")

    # ------------------------------------------------------------------
    # Train surrogates
    # ------------------------------------------------------------------
    train_configs = {
        "DeepRV + gMLP": (
            gMLPDeepRV(num_blks=N_BLOCKS),
            deep_rv_train_step,
            deep_rv_valid_step,
        ),
        "FM-DeepRV": (
            FlowMatchingDeepRV(
                vf=FlowMatchingVectorField(num_blks=N_BLOCKS), n_steps=1
            ),
            flow_matching_train_step,
            flow_matching_valid_step,
        ),
    }

    trained_states = {}
    for model_name, (nn_model, train_step, valid_step) in train_configs.items():
        model_dir = (
            save_dir / model_name.replace(" ", "_").replace("+", "plus")
        ).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = model_dir / "model.ckpt"

        lr_schedule = cosine_annealing_lr(TRAIN_STEPS, MAX_LR)
        optimizer = optax.chain(
            optax.clip_by_global_norm(3.0),
            optax.adamw(lr_schedule, weight_decay=1e-2),
        )

        if ckpt_dir.exists():
            print(f"  [{model_name}] checkpoint found, reloading.")
            state = reload_state(ckpt_dir, nn_model, s, optimizer)
        else:
            print(f"\n=== Training {model_name} ===")
            rng, rng_t, rng_v = random.split(rng, 3)
            train_loader = gen_train_dataloader(jnp.array(train_patches), s)
            valid_loader = gen_train_dataloader(jnp.array(valid_patches), s)
            wandb.init(
                config={"model_name": model_name, "dataset": "era5", "seed": seed},
                mode="disabled", reinit=True,
            )
            train_time, eval_mse, state, _, _, _ = surrogate_model_train(
                rng_t, rng_v, train_loader, train_step, valid_step,
                nn_model, model_dir, optimizer,
            )
            print(f"  trained in {train_time:.0f}s  |  eval norm MSE: {eval_mse:.4f}")

        trained_states[model_name] = (state, nn_model)

    # ------------------------------------------------------------------
    # Train BSA-TNP
    # ------------------------------------------------------------------
    bsa_model = make_bsa_tnp()
    bsa_dir   = (save_dir / "BSA-TNP").resolve()
    bsa_dir.mkdir(parents=True, exist_ok=True)
    bsa_ckpt  = bsa_dir / "model.ckpt"

    bsa_lr = cosine_annealing_lr(BSA_TRAIN_STEPS, 5e-4)
    bsa_optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(bsa_lr, b1=0.9, b2=0.999, weight_decay=1e-4),
    )

    if bsa_ckpt.exists():
        print("  [BSA-TNP] checkpoint found, reloading.")
        bsa_state = reload_bsa_state(bsa_ckpt, bsa_model, s, bsa_optimizer)
    else:
        print("\n=== Training BSA-TNP ===")
        rng, rng_bt, rng_bv = random.split(rng, 3)
        bsa_train_loader = gen_bsa_dataloader(jnp.array(train_patches), jnp.array(train_elev), s)
        bsa_valid_loader = gen_bsa_dataloader(jnp.array(valid_patches), jnp.array(valid_elev), s)
        wandb.init(
            config={"model_name": "BSA-TNP", "dataset": "era5", "seed": seed},
            mode="disabled", reinit=True,
        )
        bsa_train_time, bsa_nll, bsa_state = bsa_model_train(
            rng_bt, rng_bv, bsa_train_loader, bsa_valid_loader,
            bsa_model, bsa_dir, bsa_optimizer,
        )
        print(f"  BSA-TNP trained in {bsa_train_time:.0f}s  |  valid NLL: {bsa_nll:.4f}")

    # ------------------------------------------------------------------
    # Build eval decoders (one DeepRV, one FM per K)
    # ------------------------------------------------------------------
    state_drv, drv_model = trained_states["DeepRV + gMLP"]
    state_fm, fm_base_model = trained_states["FM-DeepRV"]
    fm_vf = fm_base_model.vf

    surrogate_decoders = {
        "DeepRV + gMLP": generate_surrogate_decoder(state_drv, drv_model),
    }
    for k in FM_K_STEPS:
        fm_k = FlowMatchingDeepRV(vf=fm_vf, n_steps=k)
        surrogate_decoders[f"FM-DeepRV (K={k})"] = generate_surrogate_decoder(state_fm, fm_k)

    eval_model_names = ["BSA-TNP", "SVGP"] + list(surrogate_decoders.keys())
    inpaint_model = build_surrogate_inpainting_model(s)

    # ------------------------------------------------------------------
    # HMC inpainting on N_TEST test patches
    # ------------------------------------------------------------------
    rng, rng_test = random.split(rng)
    test_idxs = random.choice(
        rng_test, test_patches.shape[0], shape=(N_TEST,), replace=False
    )
    test_imgs = test_patches[test_idxs]

    results = []
    vis_true, vis_masked, vis_recons, vis_masks = [], [], [], []

    test_elevs = test_elev[test_idxs]

    for patch_i in range(N_TEST):
        true_f     = jnp.array(test_imgs[patch_i])   # [L]
        elev_patch = jnp.array(test_elevs[patch_i])  # [L]
        rng, rng_nctx, rng_mask, rng_noise = random.split(rng, 4)
        n_ctx = int(random.randint(rng_nctx, shape=(), minval=N_CTX_MIN, maxval=N_CTX_MAX + 1))
        obs_mask = gen_obs_mask(rng_mask, L, n_ctx)
        y_obs = jnp.where(
            obs_mask,
            true_f + OBS_NOISE * random.normal(rng_noise, (L,)),
            jnp.zeros(L),
        )

        vis_true.append(true_f)
        vis_masked.append(true_f * obs_mask)
        vis_masks.append(obs_mask)
        img_recons = []

        # --- BSA-TNP: direct prediction (no HMC) ---
        rng, rng_bsa = random.split(rng)
        (bsa_f_pred, bsa_f_std, bsa_rmse_ctx, bsa_rmse_target, bsa_rmse_all,
         bsa_mae, bsa_crps, bsa_cov, bsa_nll, bsa_time) = (
            eval_bsa_tnp(rng_bsa, bsa_state, s, elev_patch, obs_mask, true_f)
        )
        results.append({
            "model_name": "BSA-TNP",
            "patch_idx": int(patch_i),
            "n_ctx": n_ctx,
            "infer_time": bsa_time,
            "RMSE (all)": bsa_rmse_all,
            "RMSE (ctx)": bsa_rmse_ctx,
            "RMSE (target)": bsa_rmse_target,
            "MAE (target)": bsa_mae,
            "CRPS (target)": bsa_crps,
            "Coverage 90%": bsa_cov,
            "NLL (target)": bsa_nll,
            "mean ESS z": float("nan"),
            "mean r_hat z": float("nan"),
        })
        img_recons.append(jnp.array(bsa_f_pred))

        # --- SVGP: fit per task, predict everywhere ---
        rng, rng_svgp = random.split(rng)
        (svgp_mu, svgp_std, svgp_rmse_ctx, svgp_rmse_target, svgp_rmse_all,
         svgp_mae, svgp_crps, svgp_cov, svgp_nll, svgp_time) = eval_svgp(
            rng_svgp, s, elev_patch, obs_mask, true_f, y_obs
        )
        results.append({
            "model_name": "SVGP",
            "patch_idx": int(patch_i),
            "n_ctx": n_ctx,
            "infer_time": svgp_time,
            "RMSE (all)": svgp_rmse_all,
            "RMSE (ctx)": svgp_rmse_ctx,
            "RMSE (target)": svgp_rmse_target,
            "MAE (target)": svgp_mae,
            "CRPS (target)": svgp_crps,
            "Coverage 90%": svgp_cov,
            "NLL (target)": svgp_nll,
            "mean ESS z": float("nan"),
            "mean r_hat z": float("nan"),
        })
        img_recons.append(jnp.array(svgp_mu))

        for model_name, decoder in surrogate_decoders.items():
            safe_key = (
                f"patch{patch_i}_{model_name}"
                .replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
            )
            cache_path = save_dir / f"{safe_key}.pkl"

            if cache_path.exists():
                print(f"  [{model_name} | patch {patch_i}] cached, loading.")
                with open(cache_path, "rb") as fh:
                    res = pickle.load(fh)
            else:
                print(f"\n=== {model_name} | test patch {patch_i+1}/{N_TEST} ===")
                rng, rng_i = random.split(rng)
                samples, mcmc, post, infer_time = run_hmc_inpaint(
                    rng_i, inpaint_model, y_obs, obs_mask, decoder
                )
                f_samples = np.array(post["f"])          # [S, L]
                f_mean    = f_samples.mean(axis=0)       # [L]
                rmse_all, rmse_ctx, rmse_target, mae_target, crps_target, coverage, nll_target = (
                    patch_metrics(
                        np.array(true_f), f_mean, f_samples, np.array(obs_mask)
                    )
                )
                sbc = {
                    k: np.array(v)
                    for k, v in mcmc.get_samples(group_by_chain=True).items()
                }
                res = {
                    "model_name": model_name,
                    "patch_idx": int(patch_i),
                    "n_ctx": n_ctx,
                    "infer_time": infer_time,
                    "RMSE (all)": rmse_all,
                    "RMSE (ctx)": rmse_ctx,
                    "RMSE (target)": rmse_target,
                    "MAE (target)": mae_target,
                    "CRPS (target)": crps_target,
                    "Coverage 90%": coverage,
                    "NLL (target)": nll_target,
                    "mean ESS z": mean_ess_z(mcmc),
                    "mean r_hat z": mean_rhat_z(sbc),
                    "f_mean": np.array(f_mean),
                    "samples_by_chain": sbc,
                }
                with open(cache_path, "wb") as fh:
                    pickle.dump(res, fh)

            results.append({k: res.get(k, float("nan")) for k in SCALAR_KEYS})
            img_recons.append(jnp.array(res["f_mean"]))

        vis_recons.append(img_recons)

    # ------------------------------------------------------------------
    # Aggregate and save
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(save_dir / "results.csv", index=False)
    summary_cols = [
        c for c in ["RMSE (target)", "MAE (target)", "CRPS (target)",
                    "Coverage 90%", "NLL (target)",
                    "RMSE (ctx)", "RMSE (all)",
                    "n_ctx", "mean ESS z", "mean r_hat z", "infer_time"]
        if c in df.columns
    ]
    summary = df.groupby("model_name")[summary_cols].mean()
    print("\n=== Summary ===")
    print(summary.to_string())
    summary.to_csv(save_dir / "summary.csv")

    n_vis = min(5, N_TEST)
    plot_reconstructions(
        PATCH_SIZE,
        vis_true[:n_vis],
        vis_masked[:n_vis],
        vis_recons[:n_vis],
        eval_model_names,
        vis_masks[:n_vis],
        save_dir / "reconstructions.png",
    )
    print(f"\nOutputs saved to {save_dir}")


if __name__ == "__main__":
    main()
