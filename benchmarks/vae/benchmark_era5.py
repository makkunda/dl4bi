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
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import arviz as az
import flax.linen as nn
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import optax
import pandas as pd
import xarray as xr
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from omegaconf import DictConfig
from orbax.checkpoint import PyTreeCheckpointer
from dl4bi_sps.utils import build_grid

import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import (
    TrainState,
    cosine_annealing_lr,
    estimate_flops,
    evaluate,
    save_ckpt,
    train,
)
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

PATCH_SIZE   = 16          # 16×16 spatial grid → L = 256 locations
N_TRAIN      = 50_000      # patches sampled from training region
N_TEST       = 20          # test patches to run HMC on
OBS_RATIO    = 0.3         # fraction of grid points observed
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
    """Load ERA5 netCDF for one region (2019, all months)."""
    path = ERA5_CACHE / region / "2019_*.nc"
    ds = xr.open_mfdataset(str(path), combine="by_coords")
    ds = ds.rename({"valid_time": "time", "z": "elevation", "t2m": "temperature"})
    return ds.load()


def extract_patches(
    ds: xr.Dataset,
    patch_size: int = PATCH_SIZE,
    n_patches: int = N_TRAIN,
    seed: int = 0,
    temp_mean: Optional[float] = None,
    temp_std: Optional[float] = None,
) -> tuple[np.ndarray, float, float]:
    """
    Randomly sample `n_patches` spatial windows of shape (patch_size, patch_size)
    from the ERA5 temperature field (single time step per patch).

    Returns:
        patches  : float32 array [n_patches, patch_size²], standardised
        temp_mean: mean used for standardisation (computed from ds if not given)
        temp_std : std  used for standardisation
    """
    temp = ds["temperature"]  # (time, lat, lng), in Kelvin
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

    rng = np.random.default_rng(seed)
    t_idxs   = rng.integers(0, n_times,             size=n_patches)
    lat_idxs = rng.integers(0, n_lat - patch_size,  size=n_patches)
    lng_idxs = rng.integers(0, n_lng - patch_size,  size=n_patches)

    # Pre-fetch the full temperature array to avoid repeated xarray indexing.
    temp_np = temp.values.astype("float32")  # (T, lat, lng)

    patches = np.stack([
        temp_np[t_idxs[i],
                lat_idxs[i]:lat_idxs[i] + patch_size,
                lng_idxs[i]:lng_idxs[i] + patch_size].flatten()
        for i in range(n_patches)
    ], axis=0)

    patches = (patches - temp_mean) / temp_std
    return patches.astype("float32"), temp_mean, temp_std


def build_patch_grid(patch_size: int = PATCH_SIZE) -> Array:
    """
    Spatial coordinates for a patch_size × patch_size grid.

    Each dimension runs 0°, 0.25°, 0.5°, …, 3.75° (at 0.25° ERA5 resolution).
    Coordinates are then standardised to zero mean / unit std so that they
    match the normalisation applied inside the gMLP.

    Returns shape [patch_size², 2].
    """
    step = 0.25  # ERA5 resolution in degrees
    coords_1d = np.arange(patch_size, dtype=np.float32) * step  # [0, 0.25, …, 3.75]
    lng_grid, lat_grid = np.meshgrid(coords_1d, coords_1d)      # both [P, P]
    s = np.stack([lat_grid.flatten(), lng_grid.flatten()], axis=-1)  # [L, 2]
    s = (s - s.mean(axis=0)) / s.std(axis=0)
    return jnp.array(s)


def gen_obs_mask(rng: Array, n_pixels: int, obs_ratio: float = OBS_RATIO) -> Array:
    n_obs = int(obs_ratio * n_pixels)
    idx = random.permutation(rng, n_pixels)[:n_obs]
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


def patch_mse(true_f, post_f_mean, obs_mask):
    sq = (true_f - post_f_mean) ** 2
    return float(sq.mean()), float(sq[obs_mask].mean()), float(sq[~obs_mask].mean())


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

    col_titles = ["true", f"masked ({int(OBS_RATIO*100)}%)"] + model_names
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
# Main
# ---------------------------------------------------------------------------

SCALAR_KEYS = {
    "model_name", "patch_idx", "infer_time",
    "MSE (all)", "MSE (obs)", "MSE (unobs)",
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
    train_patches, temp_mean, temp_std = extract_patches(
        ds_train, patch_size=PATCH_SIZE, n_patches=N_TRAIN, seed=seed
    )
    valid_patches, _, _ = extract_patches(
        ds_valid, patch_size=PATCH_SIZE, n_patches=10_000, seed=seed + 1,
        temp_mean=temp_mean, temp_std=temp_std,
    )
    test_patches, _, _ = extract_patches(
        ds_test, patch_size=PATCH_SIZE, n_patches=max(N_TEST * 10, 500), seed=seed + 2,
        temp_mean=temp_mean, temp_std=temp_std,
    )
    print(f"  train patches: {train_patches.shape}, valid: {valid_patches.shape}, test: {test_patches.shape}")
    print(f"  Temperature stats (train region): mean={temp_mean:.2f} K, std={temp_std:.2f} K")

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

    eval_model_names = list(surrogate_decoders.keys())
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

    for patch_i in range(N_TEST):
        true_f = jnp.array(test_imgs[patch_i])   # [L]
        rng, rng_mask, rng_noise = random.split(rng, 3)
        obs_mask = gen_obs_mask(rng_mask, L)
        y_obs = jnp.where(
            obs_mask,
            true_f + OBS_NOISE * random.normal(rng_noise, (L,)),
            jnp.zeros(L),
        )

        vis_true.append(true_f)
        vis_masked.append(true_f * obs_mask)
        vis_masks.append(obs_mask)
        img_recons = []

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
                f_mean = post["f"].mean(axis=0)
                mse_all, mse_obs, mse_unobs = patch_mse(true_f, f_mean, obs_mask)
                sbc = {
                    k: np.array(v)
                    for k, v in mcmc.get_samples(group_by_chain=True).items()
                }
                res = {
                    "model_name": model_name,
                    "patch_idx": int(patch_i),
                    "infer_time": infer_time,
                    "MSE (all)": mse_all,
                    "MSE (obs)": mse_obs,
                    "MSE (unobs)": mse_unobs,
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
        c for c in ["MSE (all)", "MSE (unobs)", "mean ESS z", "mean r_hat z", "infer_time"]
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
