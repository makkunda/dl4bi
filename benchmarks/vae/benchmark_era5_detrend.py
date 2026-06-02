#!/usr/bin/env python3
"""benchmark_era5_detrend.py

DeepRV and FM-DeepRV with elevation mean-function detrending.

A linear trend  m(elev) = alpha * elev + beta  is fit by OLS on the training
patches.  This separates the deterministic elevation-driven signal from the
stochastic spatial residual that DeepRV/FM-DeepRV model:

  Training  : u = f - m(elev)   (FM-DeepRV trains on residuals;
                                  DeepRV trains on GP draws — already zero-mean)
  Inference : f = decoder(z) + m(elev_patch)   (HMC inpaints residual, trend added back)

BSA-TNP (with elevation) is reloaded from the era5_benchmark checkpoint as a
reference.  All models are evaluated on the same 20 test patches.

Run from repo root:
    uv run python benchmarks/vae/benchmark_era5_detrend.py
"""

import os
import sys
sys.path.append("benchmarks/vae")
sys.path.append("benchmarks/meta_learning")

import jax
import jax.numpy as jnp
from jax import Array, jit, random
jax.devices()

import pickle
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import numpyro
import optax
import pandas as pd
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
import wandb

from dl4bi.vae import FlowMatchingDeepRV, FlowMatchingVectorField, gMLPDeepRV
from dl4bi.vae.train_utils import (
    deep_rv_train_step,
    flow_matching_train_step,
    flow_matching_valid_step,
    generate_surrogate_decoder,
)
from dl4bi.core.train import cosine_annealing_lr

# ---------------------------------------------------------------------------
# Re-use everything unchanged from the base benchmark
# ---------------------------------------------------------------------------
from benchmark_era5 import (
    # data
    load_era5_xr, extract_patches, build_patch_grid, gen_obs_mask,
    gen_train_dataloader, gen_deeprv_dataloader,
    # training helpers
    surrogate_model_train, reload_state, deep_rv_valid_step,
    # BSA-TNP
    make_bsa_tnp, gen_bsa_dataloader, bsa_model_train, reload_bsa_state, eval_bsa_tnp,
    # SVGP
    eval_svgp,
    # HMC + metrics
    run_hmc_inpaint, mean_ess_z, mean_rhat_z, ess_log_ls, rhat_log_ls, patch_metrics,
    # plotting
    plot_reconstructions,
    # constants
    PATCH_SIZE, N_TRAIN, N_TEST, N_CTX_MIN, N_CTX_MAX, OBS_NOISE,
    TRAIN_STEPS, VALID_INTERVAL, VALID_STEPS, BATCH_SIZE, MAX_LR,
    HMC_WARMUP, HMC_SAMPLES, HMC_CHAINS, FM_K_STEPS, N_BLOCKS,
    LS_MIN_NORM, LS_MAX_NORM, GP_JITTER,
    TRAIN_REGION, VALID_REGION, TEST_REGION, ERA5_CACHE,
    BSA_BATCH_SIZE, BSA_TRAIN_STEPS, BSA_VALID_STEPS, BSA_VALID_INTERVAL,
    SCALAR_KEYS,
)

SAVE_DIR = Path("results/era5_benchmark_detrend/")


# ---------------------------------------------------------------------------
# Elevation mean function
# ---------------------------------------------------------------------------

def fit_elevation_trend(patches: np.ndarray, elev_patches: np.ndarray):
    """OLS: temp ~ alpha * elev + beta on all (patch, location) pairs."""
    temp_flat = patches.flatten().astype(np.float64)
    elev_flat = elev_patches.flatten().astype(np.float64)
    alpha, beta = np.polyfit(elev_flat, temp_flat, deg=1)
    return float(alpha), float(beta)


def apply_trend(elev: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return (alpha * elev + beta).astype(np.float32)


def detrend(patches: np.ndarray, elev_patches: np.ndarray,
            alpha: float, beta: float) -> np.ndarray:
    return (patches - apply_trend(elev_patches, alpha, beta)).astype(np.float32)


# ---------------------------------------------------------------------------
# Modified inpainting models — add elevation trend back at inference
# ---------------------------------------------------------------------------

def build_deeprv_inpainting_model(s: Array, trend: Array) -> Callable:
    """DeepRV HMC model: residual decoder + deterministic elevation trend."""
    def inpaint(surrogate_decoder=None, obs_mask=None, y=None):
        log_ls = numpyro.sample(
            "log_ls", dist.Uniform(jnp.log(LS_MIN_NORM), jnp.log(LS_MAX_NORM))
        )
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))
        f_residual = surrogate_decoder(z, jnp.atleast_1d(log_ls), s=s).squeeze()
        f = f_residual + trend
        numpyro.deterministic("f", f)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Normal(f, OBS_NOISE), obs=y)
    return inpaint


def build_fm_inpainting_model(s: Array, trend: Array) -> Callable:
    """FM-DeepRV HMC model: residual flow decoder + deterministic elevation trend."""
    def inpaint(surrogate_decoder=None, obs_mask=None, y=None):
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))
        f_residual = surrogate_decoder(z, jnp.array([0.0]), s=s).squeeze()
        f = f_residual + trend
        numpyro.deterministic("f", f)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Normal(f, OBS_NOISE), obs=y)
    return inpaint


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = 42, use_wandb: bool = False):
    rng = random.key(seed)
    save_dir = SAVE_DIR.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Reference checkpoint dir for BSA-TNP (already trained in era5_benchmark)
    base_dir = Path("results/era5_benchmark/").resolve()

    wandb.init(
        project="era5-benchmark",
        name=f"era5_detrend_seed{seed}",
        config={
            "seed": seed, "patch_size": PATCH_SIZE,
            "n_ctx_min": N_CTX_MIN, "n_ctx_max": N_CTX_MAX,
            "obs_noise": OBS_NOISE, "train_steps": TRAIN_STEPS,
            "hmc_warmup": HMC_WARMUP, "hmc_samples": HMC_SAMPLES,
            "hmc_chains": HMC_CHAINS, "fm_k_steps": FM_K_STEPS,
            "detrend": "linear_ols",
        },
        mode="online" if use_wandb else "disabled",
        reinit=True,
    )

    # ------------------------------------------------------------------
    # Load ERA5 data
    # ------------------------------------------------------------------
    print("Loading ERA5 data...")
    ds_train = load_era5_xr(TRAIN_REGION)
    ds_valid = load_era5_xr(VALID_REGION)
    ds_test  = load_era5_xr(TEST_REGION)

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

    s = build_patch_grid()
    L = s.shape[0]
    print(f"  Spatial grid: {PATCH_SIZE}×{PATCH_SIZE} = {L} locations")

    # ------------------------------------------------------------------
    # Fit and apply elevation trend
    # ------------------------------------------------------------------
    alpha, beta = fit_elevation_trend(train_patches, train_elev)
    print(f"  Elevation trend (standardised units): temp = {alpha:.4f} * elev + {beta:.4f}")

    # FM-DeepRV trains on residuals; DeepRV uses GP draws (already zero-mean)
    train_residuals = detrend(train_patches, train_elev, alpha, beta)
    valid_residuals = detrend(valid_patches, valid_elev, alpha, beta)
    print(f"  Residual std: {train_residuals.std():.4f}  (original: {train_patches.std():.4f})")

    # ------------------------------------------------------------------
    # Train surrogates
    # ------------------------------------------------------------------
    train_configs = {
        "DeepRV + gMLP (detrend)": (
            gMLPDeepRV(num_blks=N_BLOCKS),
            deep_rv_train_step,
            deep_rv_valid_step,
            gen_deeprv_dataloader(s),        # GP draws — zero-mean, unchanged
            gen_deeprv_dataloader(s),
        ),
        "FM-DeepRV (detrend)": (
            FlowMatchingDeepRV(
                vf=FlowMatchingVectorField(num_blks=N_BLOCKS), n_steps=1
            ),
            flow_matching_train_step,
            flow_matching_valid_step,
            gen_train_dataloader(jnp.array(train_residuals), s),  # residuals
            gen_train_dataloader(jnp.array(valid_residuals), s),
        ),
    }

    trained_states = {}
    for model_name, (nn_model, train_step, valid_step, train_loader, valid_loader) in train_configs.items():
        model_dir = (save_dir / model_name.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")).resolve()
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
            train_time, eval_mse, state, _, _, _ = surrogate_model_train(
                rng_t, rng_v, train_loader, train_step, valid_step,
                nn_model, model_dir, optimizer,
            )
            print(f"  trained in {train_time:.0f}s  |  eval norm MSE: {eval_mse:.4f}")

        trained_states[model_name] = (state, nn_model)

    # ------------------------------------------------------------------
    # Reload BSA-TNP from era5_benchmark checkpoint (reference)
    # ------------------------------------------------------------------
    bsa_model = make_bsa_tnp()
    bsa_ckpt  = base_dir / "BSA-TNP" / "model.ckpt"
    bsa_lr    = cosine_annealing_lr(BSA_TRAIN_STEPS, 5e-4)
    bsa_optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(bsa_lr, b1=0.9, b2=0.999, weight_decay=1e-4),
    )
    if bsa_ckpt.exists():
        print("  [BSA-TNP] reloading from era5_benchmark checkpoint.")
        bsa_state = reload_bsa_state(bsa_ckpt, bsa_model, s, bsa_optimizer)
    else:
        print("  [BSA-TNP] no checkpoint found; skipping.")
        bsa_state = None

    # ------------------------------------------------------------------
    # Build eval decoders
    # ------------------------------------------------------------------
    state_drv, drv_model = trained_states["DeepRV + gMLP (detrend)"]
    state_fm, fm_base_model = trained_states["FM-DeepRV (detrend)"]
    fm_vf = fm_base_model.vf

    # ------------------------------------------------------------------
    # Load cached results to skip already-evaluated patches
    # ------------------------------------------------------------------
    results_csv = save_dir / "results.csv"
    if results_csv.exists():
        existing_df = pd.read_csv(results_csv)
        results = existing_df.to_dict("records")
        done_keys = set(zip(existing_df["model_name"], existing_df["patch_idx"].astype(int)))
        print(f"  Loaded {len(results)} existing rows from results.csv")
    else:
        results = []
        done_keys = set()

    # ------------------------------------------------------------------
    # Evaluation on N_TEST test patches
    # ------------------------------------------------------------------
    rng, rng_test = random.split(rng)
    test_idxs  = random.choice(rng_test, test_patches.shape[0], shape=(N_TEST,), replace=False)
    test_imgs  = test_patches[test_idxs]
    test_elevs = test_elev[test_idxs]

    vis_true, vis_masked, vis_recons, vis_masks = [], [], [], []

    for patch_i in range(N_TEST):
        true_f     = jnp.array(test_imgs[patch_i])
        elev_patch = jnp.array(test_elevs[patch_i])
        rng, rng_nctx, rng_mask, rng_noise = random.split(rng, 4)
        n_ctx    = int(random.randint(rng_nctx, shape=(), minval=N_CTX_MIN, maxval=N_CTX_MAX + 1))
        obs_mask = gen_obs_mask(rng_mask, L, n_ctx)
        y_obs    = jnp.where(
            obs_mask, true_f + OBS_NOISE * random.normal(rng_noise, (L,)), jnp.zeros(L)
        )

        # Per-patch elevation trend (deterministic, no HMC uncertainty)
        trend = jnp.array(apply_trend(np.array(elev_patch), alpha, beta))

        vis_true.append(true_f)
        vis_masked.append(true_f * obs_mask)
        vis_masks.append(obs_mask)
        img_recons = []

        # --- BSA-TNP (reference, with elevation) ---
        if bsa_state is not None and ("BSA-TNP", patch_i) not in done_keys:
            rng, rng_bsa = random.split(rng)
            (bsa_f_pred, _, bsa_rmse_ctx, bsa_rmse_target, bsa_rmse_all,
             bsa_mae, bsa_crps, bsa_cov, bsa_nll, bsa_time) = eval_bsa_tnp(
                rng_bsa, bsa_state, s, elev_patch, obs_mask, true_f
            )
            results.append({
                "model_name": "BSA-TNP",
                "patch_idx": int(patch_i), "n_ctx": n_ctx, "infer_time": bsa_time,
                "RMSE (all)": bsa_rmse_all, "RMSE (ctx)": bsa_rmse_ctx,
                "RMSE (target)": bsa_rmse_target, "MAE (target)": bsa_mae,
                "CRPS (target)": bsa_crps, "Coverage 90%": bsa_cov,
                "NLL (target)": bsa_nll,
                "mean ESS z": float("nan"), "mean r_hat z": float("nan"),
            })
            img_recons.append(jnp.array(bsa_f_pred))
        elif ("BSA-TNP", patch_i) in done_keys:
            print(f"  [BSA-TNP | patch {patch_i}] cached, skipping.")

        # --- Build per-patch inpainting models with elevation trend ---
        deeprv_inpaint = build_deeprv_inpainting_model(s, trend)
        fm_inpaint     = build_fm_inpainting_model(s, trend)

        drv_decoder = generate_surrogate_decoder(state_drv, drv_model)
        surrogate_decoders = {
            "DeepRV + gMLP (detrend)": (drv_decoder, deeprv_inpaint),
        }
        for k in FM_K_STEPS:
            fm_k = FlowMatchingDeepRV(vf=fm_vf, n_steps=k)
            surrogate_decoders[f"FM-DeepRV (K={k}, detrend)"] = (
                generate_surrogate_decoder(state_fm, fm_k), fm_inpaint
            )

        for model_name, (decoder, hmc_model) in surrogate_decoders.items():
            safe_key = (
                f"patch{patch_i}_{model_name}"
                .replace(" ", "_").replace("=", "").replace("(", "").replace(")", "").replace(",", "")
            )
            cache_path = save_dir / f"{safe_key}.pkl"

            if cache_path.exists():
                print(f"  [{model_name} | patch {patch_i}] cached, loading.")
                with open(cache_path, "rb") as fh:
                    res = pickle.load(fh)
            elif (model_name, patch_i) in done_keys:
                print(f"  [{model_name} | patch {patch_i}] in results.csv, skipping.")
                continue
            else:
                print(f"\n=== {model_name} | test patch {patch_i+1}/{N_TEST} ===")
                rng, rng_i = random.split(rng)
                samples, mcmc, post, infer_time = run_hmc_inpaint(
                    rng_i, hmc_model, y_obs, obs_mask, decoder
                )
                f_samples = np.array(post["f"])
                f_mean    = f_samples.mean(axis=0)
                rmse_all, rmse_ctx, rmse_target, mae_target, crps_target, coverage, nll_target = (
                    patch_metrics(np.array(true_f), f_mean, f_samples, np.array(obs_mask))
                )
                sbc = {k: np.array(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
                res = {
                    "model_name": model_name,
                    "patch_idx": int(patch_i), "n_ctx": n_ctx, "infer_time": infer_time,
                    "RMSE (all)": rmse_all, "RMSE (ctx)": rmse_ctx,
                    "RMSE (target)": rmse_target, "MAE (target)": mae_target,
                    "CRPS (target)": crps_target, "Coverage 90%": coverage,
                    "NLL (target)": nll_target,
                    "mean ESS z": mean_ess_z(mcmc), "mean r_hat z": mean_rhat_z(sbc),
                    "ESS log_ls": ess_log_ls(mcmc), "r_hat log_ls": rhat_log_ls(sbc),
                    "f_mean": np.array(f_mean),
                    "samples_by_chain": sbc,
                }
                with open(cache_path, "wb") as fh:
                    pickle.dump(res, fh)

            row = {k: res.get(k, float("nan")) for k in SCALAR_KEYS}
            results.append(row)
            wandb.log({f"{model_name}/{k}": v for k, v in row.items()
                       if isinstance(v, float) and k not in {"patch_idx", "n_ctx"}})
            img_recons.append(jnp.array(res["f_mean"]))

        vis_recons.append(img_recons)

    # ------------------------------------------------------------------
    # Aggregate and save
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(save_dir / "results.csv", index=False)

    summary_cols = [
        c for c in ["RMSE (target)", "MAE (target)", "CRPS (target)",
                    "Coverage 90%", "NLL (target)", "RMSE (ctx)", "RMSE (all)",
                    "n_ctx", "mean ESS z", "mean r_hat z",
                    "ESS log_ls", "r_hat log_ls", "infer_time"]
        if c in df.columns
    ]
    summary = df.groupby("model_name")[summary_cols].mean()
    print("\n=== Summary ===")
    print(summary.to_string())
    summary.to_csv(save_dir / "summary.csv")

    for model_name, row in summary.iterrows():
        wandb.log({f"summary/{model_name}/{col}": val
                   for col, val in row.items() if not np.isnan(val)})
    wandb.log({"results_table": wandb.Table(dataframe=df[list(SCALAR_KEYS)])})

    eval_model_names = ["BSA-TNP"] + [
        "DeepRV + gMLP (detrend)"
    ] + [f"FM-DeepRV (K={k}, detrend)" for k in FM_K_STEPS]

    n_vis = min(5, N_TEST)
    plot_reconstructions(
        PATCH_SIZE,
        vis_true[:n_vis], vis_masked[:n_vis], vis_recons[:n_vis],
        eval_model_names, vis_masks[:n_vis],
        save_dir / "reconstructions.png",
    )
    print(f"\nOutputs saved to {save_dir}")
    print(f"  Elevation trend: alpha={alpha:.4f}, beta={beta:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    main(seed=args.seed, use_wandb=args.wandb)
