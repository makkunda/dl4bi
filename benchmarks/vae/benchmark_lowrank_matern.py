#!/usr/bin/env python3
"""benchmark_lowrank_matern.py

Compares standard FM-DeepRV (z ∈ ℝ^L) against LowRank FM-DeepRV (z ∈ ℝ^M)
on Matérn-3/2 GP inpainting with Gaussian likelihood.

The key question: does shrinking the HMC latent from L to M dimensions
improve ESS z while preserving prediction quality?

Models
------
  Baseline GP        — exact Cholesky (gold standard)
  FM-DeepRV  K=5    — L-dimensional z (reference)
  LowRank FM K=1    — M-dimensional z, 1 ODE step  (M = L // M_FRAC)
  LowRank FM K=5    — M-dimensional z, 5 ODE steps

LowRank training
----------------
  Dataloader yields both z_m [B, M] (inducing noise) and f_full [B, L]
  (full GP draw).  Inducing values f_m = f_full[:, inducing_idxs].

  Combined loss per step:
    L_FM  = ||v_θ(x_t, t, u) - (f_m - z_m)||²     (OT-CFM in M-space)
    L_dec = ||decoder(f_m, u, s) - f_full||²        (full-field reconstruction)

  Both VF and decoder parameters are updated jointly.

LowRank inference
-----------------
  HMC samples z ~ N(0, I_M) — only M dimensions, much faster mixing.
  surrogate_decoder(z_m, cond, u=u, s=s) integrates ODE then upsamples.

Run from repo root:
    uv run python benchmarks/vae/benchmark_lowrank_matern.py --ls 20
"""

import sys
sys.path.append("benchmarks/vae")

import pickle
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import arviz as az
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import optax
import pandas as pd
from jax import Array, jit, random
from numpyro import distributions as dist
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from omegaconf import DictConfig
from dl4bi_sps.kernels import matern_3_2
from dl4bi_sps.utils import build_grid

import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import (
    TrainState, cosine_annealing_lr, estimate_flops, evaluate, save_ckpt, train,
)
from dl4bi.vae import FlowMatchingDeepRV, FlowMatchingVectorField
from dl4bi.vae.low_rank_flow_matching import (
    LowRankFMDeepRV, LowRankVectorField, SpatialDecoder,
)
from dl4bi.vae.train_utils import (
    flow_matching_train_step, flow_matching_valid_step, generate_surrogate_decoder,
)

# Reuse shared helpers from the CFM benchmark
from benchmark_cfm_matern import (
    build_spatial_grid, gen_gp_sample, gen_obs, gen_spatial_obs_mask,
    gen_train_dataloader, build_gp_inference_model,
    OBS_NOISE, OBS_RATIO, TRAIN_STEPS, VALID_INTERVAL, VALID_STEPS,
    BATCH_SIZE, MAX_LR, HMC_WARMUP, HMC_SAMPLES, HMC_CHAINS,
    deep_rv_valid_step,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GRIDS        = [16, 32]
LENGTHSCALES = [20]
M_FRAC       = 8        # M = L // M_FRAC   (e.g. L=256 → M=32, L=1024 → M=128)
FM_K_STEPS   = [1, 5]


# ---------------------------------------------------------------------------
# Inducing point selection
# ---------------------------------------------------------------------------

def build_inducing_points(s: Array, M: int):
    """Select M evenly-spaced inducing points as a sub-grid of s.

    Returns:
        u:            [M, d]  inducing locations
        inducing_idxs [M]     integer indices into s
    """
    L    = s.shape[0]
    step = max(1, L // M)
    idxs = jnp.arange(0, L, step)[:M]
    # If rounding gives fewer than M, pad with last index (harmless)
    if idxs.shape[0] < M:
        idxs = jnp.concatenate([idxs, jnp.full((M - idxs.shape[0],), L - 1)])
    return s[idxs], idxs


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

def gen_lowrank_dataloader(
    s: Array,
    u: Array,
    inducing_idxs: Array,
    priors: dict,
    batch_size: int = BATCH_SIZE,
):
    """GP draws yielding both M-dim inducing values and L-dim full fields.

    Keys yielded:
      z          [B, M]   M-dim noise (independent of f)
      f          [B, M]   inducing values  f_full[:, inducing_idxs]
      f_full     [B, L]   full GP draw (for decoder supervision)
      u          [M, d]   inducing locations (constant)
      s          [L, d]   all locations (constant)
      conditionals [C]    GP hyperparameters
    """
    L      = s.shape[0]
    M      = u.shape[0]
    jitter = 5e-4 * jnp.eye(L)

    def dataloader(rng):
        while True:
            rng, rng_ls, rng_z_m, rng_z_full = random.split(rng, 4)
            ls = priors["ls"].sample(rng_ls)

            # Full GP draw
            z_full  = dist.Normal().sample(rng_z_full, sample_shape=(batch_size, L))
            K       = matern_3_2(s, s, 1.0, ls) + jitter
            Lc      = jnp.linalg.cholesky(K)
            f_full  = jnp.einsum("ij,bj->bi", Lc, z_full)              # [B, L]
            f_m     = f_full[:, inducing_idxs]                          # [B, M]

            # Independent M-dim noise for the flow (OT coupling)
            z_m = dist.Normal().sample(rng_z_m, sample_shape=(batch_size, M))

            yield {
                "s":            s,
                "u":            u,
                "z":            z_m,
                "f":            f_m,
                "f_full":       f_full,
                "conditionals": jnp.array([ls]),
            }

    return dataloader


# ---------------------------------------------------------------------------
# Train / valid steps for LowRankFMDeepRV
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=["var_idx"])
def lowrank_fm_train_step(rng, state, batch, var_idx=None):
    """Combined FM + decoder training step.

    FM loss operates in M-space (velocity matching on inducing values).
    Decoder loss ensures full-field reconstruction quality.
    Both sets of parameters are updated in one backward pass.
    """
    def _loss(params):
        f_m    = batch["f"]            # [B, M] inducing targets
        z_m    = batch["z"]            # [B, M] noise
        cond   = batch["conditionals"]
        u      = batch["u"]            # [M, d]
        s      = batch["s"]            # [L, d]
        f_full = batch["f_full"]       # [B, L]
        B      = z_m.shape[0]

        rng_t, rng_app = jax.random.split(rng)
        t      = jax.random.uniform(rng_t, (B,))
        x_t    = (1.0 - t[:, None]) * z_m + t[:, None] * f_m
        v_tgt  = f_m - z_m

        # --- FM loss: VF predicts velocity in M-space ---
        fm_out = state.apply_fn(
            {"params": params, **state.kwargs},
            x_t, cond, t, u=u,
            rngs={"extra": rng_app},
        )
        fm_loss = jnp.mean((fm_out.f_hat.squeeze() - v_tgt) ** 2)

        # --- Decoder loss: true f_m → f_full ---
        dec_out = state.apply_fn(
            {"params": params, **state.kwargs},
            f_m, u, s,
            rngs={"extra": rng_app},
            method="decode_from_inducing",
        )
        dec_loss = jnp.mean((dec_out - f_full) ** 2)

        scale = cond[var_idx] if var_idx is not None else 1.0
        return (fm_loss + dec_loss) / scale

    loss, grads = jax.value_and_grad(_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


@jit
def lowrank_fm_valid_step(rng, state, batch):
    """End-to-end MSE: z_m → flow → f_m → decoder → f_full."""
    z_m    = batch["z"]
    f_full = batch["f_full"]
    cond   = batch["conditionals"]
    u      = batch["u"]
    s      = batch["s"]

    f_hat = state.apply_fn(
        {"params": state.params, **state.kwargs},
        z_m, cond, u=u, s=s,
        rngs={"extra": rng},
        method="decode",
    )
    return {"norm MSE": jnp.mean((f_hat.reshape(f_full.shape) - f_full) ** 2)}


# ---------------------------------------------------------------------------
# HMC inference model for LowRank (M-dim z)
# ---------------------------------------------------------------------------

def build_lowrank_inference_model(s: Array, u: Array, M: int, priors: dict) -> Callable:
    """Same Gaussian likelihood as standard model but z ~ N(0, I_M).

    HMC explores M dimensions instead of L — the key efficiency gain.
    The surrogate decoder handles the M → L upsampling internally.
    """
    def model(surrogate_decoder=None, obs_mask=True, y=None):
        ls  = numpyro.sample("ls", priors["ls"])
        z_m = numpyro.sample("z",  dist.Normal(), sample_shape=(1, M))   # M-dim!
        if surrogate_decoder is None:
            # Fallback: exact GP (not used in practice for LowRank model)
            K  = matern_3_2(s, s, 1.0, ls) + 5e-4 * jnp.eye(s.shape[0])
            mu = numpyro.deterministic("mu", jnp.linalg.cholesky(K) @ z_m[0, :s.shape[0]])
        else:
            mu = numpyro.deterministic(
                "mu",
                surrogate_decoder(z_m, jnp.array([ls]), u=u, s=s).squeeze(),
            )
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Normal(mu, OBS_NOISE), obs=y)

    return model


# ---------------------------------------------------------------------------
# Training infrastructure
# ---------------------------------------------------------------------------

def train_surrogate(
    rng_train, rng_test, loader, train_step, valid_step,
    model, model_dir, optimizer,
):
    flop_batch = next(loader(rng_train))
    rngs   = {"params": rng_train, "extra": rng_test}
    kwargs = model.init(rngs, **flop_batch)
    params = kwargs.pop("params")
    state  = TrainState.create(apply_fn=model.apply, params=params, kwargs=kwargs, tx=optimizer)
    t0     = datetime.now()
    state  = train(
        rng_train, model, optimizer, train_step, TRAIN_STEPS, loader,
        valid_step, VALID_INTERVAL, VALID_STEPS, loader,
        return_state="best", valid_monitor_metric="norm MSE",
    )
    train_time = (datetime.now() - t0).total_seconds()
    eval_mse   = evaluate(rng_test, state, valid_step, loader, VALID_STEPS)["norm MSE"]
    save_ckpt(state, DictConfig({}), model_dir / "model.ckpt")
    return train_time, eval_mse, generate_surrogate_decoder(state, model)


# ---------------------------------------------------------------------------
# HMC
# ---------------------------------------------------------------------------

def run_hmc(rng, infer_model, y_obs, obs_mask, model_dir, surrogate_decoder=None):
    nuts = NUTS(infer_model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    mcmc = MCMC(nuts, num_chains=HMC_CHAINS, num_samples=HMC_SAMPLES, num_warmup=HMC_WARMUP)
    t0   = datetime.now()
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=y_obs)
    infer_time = (datetime.now() - t0).total_seconds()
    mcmc.print_summary()
    samples = mcmc.get_samples()
    post = Predictive(infer_model, samples)(k2, surrogate_decoder=surrogate_decoder)
    with open(model_dir / "hmc_samples.pkl", "wb") as fh:
        pickle.dump({k: v for k, v in samples.items() if k in ("ls",)}, fh)
    with open(model_dir / "hmc_pp.pkl", "wb") as fh:
        pickle.dump(post, fh)
    return samples, mcmc, post, infer_time


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def collect_result(model_name, train_time, infer_time, eval_mse,
                   f_true, post, obs_mask, samples, mcmc, L, M, seed):
    mu   = post["mu"].mean(axis=0)
    sq   = (f_true - mu) ** 2
    ess  = az.ess(mcmc, method="mean") if mcmc is not None else {}
    ls_s = (numpyro_summary(mcmc.get_samples(group_by_chain=True), prob=0.9)["ls"]
            if mcmc is not None else {})
    return {
        "model_name":       model_name,
        "grid_size":        L,
        "M":                M,
        "seed":             seed,
        "train_time":       train_time,
        "infer_time":       infer_time,
        "total_time":       (train_time or 0) + infer_time,
        "eval_norm_mse":    eval_mse,
        "MSE (all)":        float(sq.mean()),
        "MSE (obs)":        float(sq[obs_mask].mean()),
        "MSE (unobs)":      float(sq[~obs_mask].mean()),
        "ESS ls":           float(ess["ls"].mean()) if "ls" in ess else None,
        "ESS z mean":       float(ess["z"].values.mean()) if "z" in ess else None,
        "r_hat ls":         float(ls_s["r_hat"]) if ls_s else None,
        "inferred ls mean": float(samples["ls"].mean()) if "ls" in samples else None,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_means(grid_n, f_true, f_hats, obs_mask, names, save_path):
    f2d  = np.array(f_true).reshape(grid_n, grid_n)
    ncols = 2 + len(f_hats)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), constrained_layout=True)
    vmin, vmax = float(f2d.min()), float(f2d.max())
    cmap   = plt.cm.RdBu_r
    masked = np.ma.masked_where(~np.array(obs_mask).reshape(grid_n, grid_n), f2d)
    axes[0].imshow(masked, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("observed")
    axes[1].imshow(f2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("true f")
    for ax, fh, name in zip(axes[2:], f_hats, names):
        im = ax.imshow(np.array(fh).mean(axis=0).reshape(grid_n, grid_n),
                       origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[-1])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def aggregate_and_plot(save_dir: Path):
    dfs = [pd.read_csv(p) for p in sorted(save_dir.glob("grid_*/res.csv"))]
    if not dfs:
        return
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(save_dir / "aggregated.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)
    metrics = ["MSE (unobs)", "ESS z mean", "ESS ls", "infer_time"]
    for ax, m in zip(axes, metrics):
        for model in df["model_name"].unique():
            sub = df[df["model_name"] == model]
            if m not in sub or sub[m].isnull().all():
                continue
            ax.plot(sub["grid_size"], sub[m], marker="o", label=model)
        ax.set_title(m)
        ax.set_xlabel("Grid size (L)")
        ax.legend(fontsize=7)
    fig.savefig(save_dir / "scalability.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {save_dir / 'aggregated.csv'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = 42, gt_ls: int = 20):
    rng      = random.key(seed)
    save_dir = Path(f"results/lowrank_matern_ls{gt_ls}/")
    save_dir.mkdir(parents=True, exist_ok=True)

    priors = {"ls": dist.Uniform(1.0, 100.0)}

    for grid_n in GRIDS:
        s  = build_spatial_grid(grid_n)
        L  = s.shape[0]
        M  = L // M_FRAC
        grid_dir = save_dir / f"grid_{L}"
        grid_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Grid {grid_n}×{grid_n}  L={L}  M={M}  (ratio {M/L:.2%})")
        print(f"{'='*60}")

        u, inducing_idxs = build_inducing_points(s, M)

        rng, rng_f, rng_noise, rng_mask, rng_train, rng_test, rng_infer = (
            random.split(rng, 7)
        )
        f_true   = gen_gp_sample(rng_f, s, gt_ls)
        y_obs    = gen_obs(rng_noise, f_true)
        obs_mask = gen_spatial_obs_mask(rng_mask, grid_n)

        # Standard inference model (L-dim z) — used by Baseline GP and FM-DeepRV
        std_infer_model = build_gp_inference_model(s, priors)

        # Low-rank inference model (M-dim z) — used by LowRank FM
        lr_infer_model  = build_lowrank_inference_model(s, u, M, priors)

        # Full-L dataloader for FM-DeepRV reference
        full_loader = gen_train_dataloader(s, priors)

        # M-dim dataloader for LowRank FM
        lr_loader   = gen_lowrank_dataloader(s, u, inducing_idxs, priors)

        # ----------------------------------------------------------------
        # Model registry
        # (nn_model, train_step, valid_step, loader, infer_model)
        # ----------------------------------------------------------------
        models = {}

        # Baseline GP (exact Cholesky, L-dim z)
        models["Baseline GP"] = (None, None, None, None, std_infer_model)

        # FM-DeepRV reference (L-dim z)
        for k in FM_K_STEPS:
            models[f"FM-DeepRV (K={k})"] = (
                FlowMatchingDeepRV(vf=FlowMatchingVectorField(num_blks=2), n_steps=k),
                flow_matching_train_step,
                flow_matching_valid_step,
                full_loader,
                std_infer_model,
            )

        # LowRank FM (M-dim z)
        for k in FM_K_STEPS:
            models[f"LowRank FM (K={k}, M={M})"] = (
                LowRankFMDeepRV(
                    vf=LowRankVectorField(num_blks=2),
                    decoder=SpatialDecoder(d_model=64),
                    n_steps=k,
                ),
                lowrank_fm_train_step,
                lowrank_fm_valid_step,
                lr_loader,
                lr_infer_model,
            )

        # ----------------------------------------------------------------
        result, f_hats, names_done = [], [], []

        for model_name, (nn_model, train_step, valid_step, loader, infer_model) in models.items():
            model_dir = (
                grid_dir
                / model_name.replace(" ", "_").replace("(", "").replace(")", "")
                            .replace("=", "").replace(",", "")
            )
            model_dir.mkdir(parents=True, exist_ok=True)

            if (model_dir / "single_res.pkl").exists():
                print(f"  [{model_name}] done, loading.")
                with open(model_dir / "hmc_pp.pkl", "rb") as fh:
                    post = pickle.load(fh)
                with open(model_dir / "single_res.pkl", "rb") as fh:
                    res = pickle.load(fh)
                f_hats.append(post["mu"])
                names_done.append(model_name)
                result.append(res)
                continue

            print(f"\n--- {model_name} ---")
            train_time = eval_mse = surrogate_decoder = None

            if nn_model is not None:
                lr_sched  = cosine_annealing_lr(TRAIN_STEPS, MAX_LR)
                optimizer = optax.chain(
                    optax.clip_by_global_norm(3.0),
                    optax.adamw(lr_sched, weight_decay=1e-2),
                )
                wandb.init(config={"model": model_name, "L": L, "M": M, "seed": seed},
                           mode="disabled", reinit=True)
                rng_train, rt = random.split(rng_train)
                rng_test,  rv = random.split(rng_test)
                train_time, eval_mse, surrogate_decoder = train_surrogate(
                    rt, rv, loader, train_step, valid_step, nn_model, model_dir, optimizer,
                )
                print(f"  train {train_time:.0f}s  eval_mse={eval_mse:.4f}")

            rng_infer, ri = random.split(rng_infer)
            samples, mcmc, post, infer_time = run_hmc(
                ri, infer_model, y_obs, obs_mask, model_dir,
                surrogate_decoder=surrogate_decoder,
            )
            print(f"  infer {infer_time:.0f}s")

            M_actual = M if "LowRank" in model_name else L
            res = collect_result(
                model_name, train_time, infer_time, eval_mse,
                np.array(f_true), post, np.array(obs_mask),
                {k: v for k, v in samples.items() if k == "ls"},
                mcmc, L, M_actual, seed,
            )
            with open(model_dir / "single_res.pkl", "wb") as fh:
                pickle.dump(res, fh)

            f_hats.append(post["mu"])
            names_done.append(model_name)
            result.append(res)

        # Save grid results
        df = pd.DataFrame(result)
        df.to_csv(grid_dir / "res.csv", index=False)
        plot_means(grid_n, np.array(f_true), f_hats, np.array(obs_mask),
                   names_done, grid_dir / "means.png")

        # Print summary
        print(f"\n--- Grid {grid_n}×{grid_n} summary (L={L}, M={M}) ---")
        cols = ["model_name", "M", "MSE (unobs)", "ESS ls", "ESS z mean",
                "r_hat ls", "inferred ls mean", "infer_time"]
        print(df[cols].to_string(index=False))

    aggregate_and_plot(save_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ls",   type=int, default=20)
    args = p.parse_args()
    main(seed=args.seed, gt_ls=args.ls)
