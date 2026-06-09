#!/usr/bin/env python3
"""benchmark_cfm_matern.py

Compares three flow-based priors on Matérn-3/2 GP inpainting (Gaussian likelihood):

  FM-DeepRV          — standard OT-CFM (straight-line paths, unconditional)
  Bridge FM-DeepRV   — stochastic bridge matching (noisy paths, unconditional)
  Conditional FM     — conditional OT-CFM (context-conditioned, no z-HMC warmup)
  Baseline GP        — exact Cholesky HMC (gold standard)
  DeepRV + gMLP      — MSE-trained surrogate (baseline)

Setup
-----
  f ~ GP(0, Matérn-3/2(s, s; ls)),  ls ~ Uniform(1, 100)
  y  = f + ε,  ε ~ N(0, OBS_NOISE²)   (Gaussian likelihood)
  50 % of locations observed (spatial blob mask)

Conditional FM differences
--------------------------
  Training  : dataloader provides a fixed-size context subset (s_ctx, f_ctx)
              alongside each GP draw; the context encoder conditions the VF.
  Inference : HMC still runs over z and ls, but the conditional flow provides
              p(f | s_ctx, f_ctx) ≈ posterior prior → faster mixing.
              s_ctx = observed locations, f_ctx = noisy observations there.

Bridge matching differences
---------------------------
  Same architecture as FM-DeepRV; only the train step changes:
    x_t = (1-t)*z + t*f + BRIDGE_SIGMA * sqrt(t*(1-t)) * eps
    v*  = (f - x_t) / (1 - t)   (Doob h-transform)
  t is clipped to [0, 0.95] to avoid 1/(1-t) blow-up.

Run from repo root:
    uv run python benchmarks/vae/benchmark_cfm_matern.py
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
from dl4bi.vae import FlowMatchingDeepRV, FlowMatchingVectorField, gMLPDeepRV
from dl4bi.vae.conditional_flow_matching import (
    ConditionalFMDeepRV, ConditionalVectorField,
)
from dl4bi.vae.train_utils import (
    deep_rv_train_step, flow_matching_train_step, flow_matching_valid_step,
    generate_surrogate_decoder,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GRIDS        = [16, 32]
LENGTHSCALES = [10, 20]
OBS_NOISE    = 0.1
OBS_RATIO    = 0.5       # fraction of locations observed
CTX_FRAC     = 0.3       # fraction of L used as context during CFM training
BRIDGE_SIGMA = 0.1       # bridge noise (0 → OT-CFM)
D_CTX        = 64        # context embedding dim

TRAIN_STEPS    = 200_000
VALID_INTERVAL = 25_000
VALID_STEPS    = 5_000
BATCH_SIZE     = 32
MAX_LR         = 1e-3

HMC_WARMUP  = 2_000
HMC_SAMPLES = 4_000
HMC_CHAINS  = 2

FM_K_STEPS = [1, 5]


# ---------------------------------------------------------------------------
# Valid step for gMLPDeepRV
# ---------------------------------------------------------------------------

@jit
def deep_rv_valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    return {"norm MSE": output.metrics(batch["f"], 1.0)["MSE"]}


# ---------------------------------------------------------------------------
# Bridge matching train / valid steps
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=["var_idx"])
def bridge_matching_train_step(rng, state, batch, var_idx=None):
    """OT-CFM with a stochastic Brownian bridge interpolant.

    x_t = (1-t)*z + t*f + BRIDGE_SIGMA*sqrt(t(1-t))*eps
    v*  = (f - x_t) / (1 - t)      (Doob h-transform / bridge velocity)
    t   ~ Uniform(0, 0.95)          (avoid 1/(1-t) blow-up)
    """
    def _loss(params):
        f    = batch["f"]
        z0   = batch["z"]
        cond = batch["conditionals"]
        B    = z0.shape[0]

        rng_t, rng_eps, rng_app = jax.random.split(rng, 3)
        t   = jax.random.uniform(rng_t, (B,), minval=0.0, maxval=0.95)
        eps = jax.random.normal(rng_eps, f.shape)

        noise = BRIDGE_SIGMA * jnp.sqrt(t * (1.0 - t))[:, None] * eps
        x_t   = (1.0 - t[:, None]) * z0 + t[:, None] * f + noise
        v_tgt = (f - x_t) / (1.0 - t[:, None] + 1e-6)

        extra  = {k: v for k, v in batch.items() if k not in ("f", "z", "conditionals")}
        output = state.apply_fn(
            {"params": params, **state.kwargs},
            x_t, cond, t, **extra, rngs={"extra": rng_app},
        )
        scale = cond[var_idx] if var_idx is not None else 1.0
        return (1.0 / scale) * jnp.mean((output.f_hat.squeeze() - v_tgt) ** 2)

    loss, grads = jax.value_and_grad(_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


# ---------------------------------------------------------------------------
# Conditional FM train / valid steps
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=["var_idx"])
def conditional_fm_train_step(rng, state, batch, var_idx=None):
    """OT-CFM train step for ConditionalFMDeepRV.

    Batch must contain: f, z, conditionals, s, s_ctx [n_ctx,2], f_ctx [B,n_ctx,1].
    """
    def _loss(params):
        f    = batch["f"]
        z0   = batch["z"]
        cond = batch["conditionals"]
        B    = z0.shape[0]

        rng_t, rng_app = jax.random.split(rng)
        t      = jax.random.uniform(rng_t, (B,))
        x_t    = (1.0 - t[:, None]) * z0 + t[:, None] * f
        v_tgt  = f - z0

        output = state.apply_fn(
            {"params": params, **state.kwargs},
            x_t, cond, t,
            s=batch["s"], s_ctx=batch["s_ctx"], f_ctx=batch["f_ctx"],
            rngs={"extra": rng_app},
        )
        scale = cond[var_idx] if var_idx is not None else 1.0
        return (1.0 / scale) * jnp.mean((output.f_hat.squeeze() - v_tgt) ** 2)

    loss, grads = jax.value_and_grad(_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


@jit
def conditional_fm_valid_step(rng, state, batch):
    """Valid step for ConditionalFMDeepRV: measures decoded sample MSE."""
    z0 = batch["z"]
    f  = batch["f"]
    f_hat = state.apply_fn(
        {"params": state.params, **state.kwargs},
        z0, batch["conditionals"],
        s=batch["s"], s_ctx=batch["s_ctx"], f_ctx=batch["f_ctx"],
        rngs={"extra": rng},
        method="decode",
    )
    return {"norm MSE": jnp.mean((f_hat.reshape(f.shape) - f) ** 2)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_spatial_grid(n: int) -> Array:
    return build_grid([{"start": 0.0, "stop": 100.0, "num": n}] * 2).reshape(-1, 2)


def gen_gp_sample(rng: Array, s: Array, ls: float) -> Array:
    K  = matern_3_2(s, s, 1.0, ls) + 5e-4 * jnp.eye(s.shape[0])
    Lc = jnp.linalg.cholesky(K)
    return Lc @ dist.Normal().sample(rng, (s.shape[0],))


def gen_obs(rng: Array, f: Array) -> Array:
    return f + OBS_NOISE * random.normal(rng, f.shape)


def gen_spatial_obs_mask(rng: Array, grid_n: int, obs_ratio: float = OBS_RATIO) -> Array:
    H, W   = grid_n, grid_n
    n_obs  = int(obs_ratio * H * W)
    mask   = jnp.zeros((H, W), dtype=bool)
    collected = 0
    while collected < n_obs:
        rng, rb = random.split(rng)
        r4 = random.split(rb, 4)
        cx = random.randint(r4[0], (), 0, H)
        cy = random.randint(r4[1], (), 0, W)
        rx = random.randint(r4[2], (), H // 8, H // 4)
        ry = random.randint(r4[3], (), W // 8, W // 4)
        yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
        ellipse  = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        new_mask = jnp.logical_or(mask, ellipse)
        collected += int(jnp.sum(new_mask) - jnp.sum(mask))
        mask = new_mask
    if collected > n_obs:
        idxs = jnp.argwhere(mask.flatten()).squeeze()
        sel  = random.choice(random.split(rng)[0], idxs, (n_obs,), replace=False)
        return jnp.zeros(H * W, dtype=bool).at[sel].set(True)
    return mask.flatten()


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def gen_train_dataloader(s: Array, priors: dict, batch_size: int = BATCH_SIZE):
    """Standard GP dataloader for FM-DeepRV and DeepRV."""
    L      = s.shape[0]
    jitter = 5e-4 * jnp.eye(L)

    def dataloader(rng):
        while True:
            rng, rng_ls, rng_z = random.split(rng, 3)
            ls = priors["ls"].sample(rng_ls)
            z  = dist.Normal().sample(rng_z, sample_shape=(batch_size, L))
            K  = matern_3_2(s, s, 1.0, ls) + jitter
            Lc = jnp.linalg.cholesky(K)
            f  = jnp.einsum("ij,bj->bi", Lc, z)
            yield {"s": s, "z": z, "conditionals": jnp.array([ls]), "f": f}

    return dataloader


def gen_cond_fm_dataloader(s: Array, priors: dict, batch_size: int = BATCH_SIZE):
    """Dataloader for ConditionalFMDeepRV.

    Each batch also includes a fixed-size random context:
      s_ctx : [n_ctx, 2]       spatial locations (same for all items in batch)
      f_ctx : [B, n_ctx, 1]    GP values at those locations (per item)

    n_ctx is fixed to CTX_FRAC * L to avoid jit recompilation on shape changes.
    """
    L      = s.shape[0]
    n_ctx  = max(1, int(CTX_FRAC * L))
    jitter = 5e-4 * jnp.eye(L)

    def dataloader(rng):
        while True:
            rng, rng_ls, rng_z, rng_ctx = random.split(rng, 4)
            ls = priors["ls"].sample(rng_ls)
            z  = dist.Normal().sample(rng_z, sample_shape=(batch_size, L))
            K  = matern_3_2(s, s, 1.0, ls) + jitter
            Lc = jnp.linalg.cholesky(K)
            f  = jnp.einsum("ij,bj->bi", Lc, z)         # [B, L]

            # Random context: same indices for all batch items
            ctx_idxs = random.permutation(rng_ctx, L)[:n_ctx]
            s_ctx    = s[ctx_idxs]                        # [n_ctx, 2]
            f_ctx    = f[:, ctx_idxs, None]               # [B, n_ctx, 1]

            yield {
                "s": s, "z": z, "conditionals": jnp.array([ls]), "f": f,
                "s_ctx": s_ctx, "f_ctx": f_ctx,
            }

    return dataloader


# ---------------------------------------------------------------------------
# Inference models
# ---------------------------------------------------------------------------

def build_gp_inference_model(s: Array, priors: dict) -> Callable:
    """Exact Cholesky GP with Gaussian likelihood."""
    def model(surrogate_decoder=None, obs_mask=True, y=None, s_ctx=None, f_ctx=None):
        ls = numpyro.sample("ls", priors["ls"])
        z  = numpyro.sample("z",  dist.Normal(), sample_shape=(1, s.shape[0]))
        if surrogate_decoder is None:
            K  = matern_3_2(s, s, 1.0, ls) + 5e-4 * jnp.eye(s.shape[0])
            mu = numpyro.deterministic("mu", (jnp.linalg.cholesky(K) @ z[0]))
        else:
            kwargs = {"s": s}
            if s_ctx is not None and f_ctx is not None:
                kwargs.update({"s_ctx": s_ctx, "f_ctx": f_ctx})
            mu = numpyro.deterministic(
                "mu", surrogate_decoder(z, jnp.array([ls]), **kwargs).squeeze()
            )
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Normal(mu, OBS_NOISE), obs=y)
    return model


# ---------------------------------------------------------------------------
# Training
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

def run_hmc(rng, infer_model, y_obs, obs_mask, model_dir, surrogate_decoder=None,
            s_ctx=None, f_ctx=None):
    nuts = NUTS(infer_model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    mcmc = MCMC(nuts, num_chains=HMC_CHAINS, num_samples=HMC_SAMPLES, num_warmup=HMC_WARMUP)
    t0 = datetime.now()
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=y_obs,
             s_ctx=s_ctx, f_ctx=f_ctx)
    infer_time = (datetime.now() - t0).total_seconds()
    mcmc.print_summary()
    samples = mcmc.get_samples()
    post = Predictive(infer_model, samples)(
        k2, surrogate_decoder=surrogate_decoder, s_ctx=s_ctx, f_ctx=f_ctx
    )
    with open(model_dir / "hmc_samples.pkl", "wb") as fh:
        pickle.dump({k: v for k, v in samples.items() if k in ("ls",)}, fh)
    with open(model_dir / "hmc_pp.pkl", "wb") as fh:
        pickle.dump(post, fh)
    return samples, mcmc, post, infer_time


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def collect_result(model_name, train_time, infer_time, eval_mse,
                   f_true, post, obs_mask, samples, mcmc, L, seed):
    mu    = post["mu"].mean(axis=0)
    sq    = (f_true - mu) ** 2
    ess   = az.ess(mcmc, method="mean") if mcmc is not None else {}
    ls_s  = (numpyro_summary(mcmc.get_samples(group_by_chain=True), prob=0.9)["ls"]
             if mcmc is not None else {})
    return {
        "model_name":            model_name,
        "grid_size":             L,
        "seed":                  seed,
        "train_time":            train_time,
        "infer_time":            infer_time,
        "total_time":            (train_time or 0) + infer_time,
        "eval_norm_mse":         eval_mse,
        "MSE (all)":             float(sq.mean()),
        "MSE (obs)":             float(sq[obs_mask].mean()),
        "MSE (unobs)":           float(sq[~obs_mask].mean()),
        "ESS ls":                float(ess["ls"].mean()) if "ls" in ess else None,
        "ESS z mean":            float(ess["z"].values.mean()) if "z" in ess else None,
        "r_hat ls":              float(ls_s["r_hat"]) if ls_s else None,
        "inferred ls mean":      float(samples["ls"].mean()) if "ls" in samples else None,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_means(grid_n, f_true, f_hats, obs_mask, names, save_path):
    f2d  = f_true.reshape(grid_n, grid_n)
    ncols = 2 + len(f_hats)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), constrained_layout=True)
    vmin, vmax = float(f2d.min()), float(f2d.max())
    cmap = plt.cm.RdBu_r
    masked = np.ma.masked_where(~obs_mask.reshape(grid_n, grid_n), f2d)
    axes[0].imshow(masked, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("observed")
    axes[1].imshow(f2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("true f")
    for ax, fh, name in zip(axes[2:], f_hats, names):
        im = ax.imshow(fh.mean(axis=0).reshape(grid_n, grid_n),
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
    metrics = ["MSE (unobs)", "ESS ls", "ESS z mean", "r_hat ls", "infer_time"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4),
                              constrained_layout=True)
    for ax, m in zip(axes, metrics):
        for model in df["model_name"].unique():
            sub = df[df["model_name"] == model]
            if m not in sub or sub[m].isnull().all():
                continue
            ax.plot(sub["grid_size"], sub[m], marker="o", label=model)
        ax.set_title(m)
        ax.set_xlabel("Grid size")
        ax.legend(fontsize=7)
    fig.savefig(save_dir / "scalability.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {save_dir / 'aggregated.csv'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = 42, gt_ls: int = 10):
    rng      = random.key(seed)
    save_dir = Path(f"results/cfm_bridge_matern_ls{gt_ls}/")
    save_dir.mkdir(parents=True, exist_ok=True)

    priors = {"ls": dist.Uniform(1.0, 100.0)}

    for grid_n in GRIDS:
        s = build_spatial_grid(grid_n)
        L = s.shape[0]
        n_ctx = max(1, int(CTX_FRAC * L))
        grid_dir = save_dir / f"grid_{L}"
        grid_dir.mkdir(parents=True, exist_ok=True)

        rng, rng_f, rng_obs_noise, rng_mask, rng_train, rng_test, rng_infer = (
            random.split(rng, 7)
        )
        f_true   = gen_gp_sample(rng_f, s, gt_ls)
        y_obs    = gen_obs(rng_obs_noise, f_true)
        obs_mask = gen_spatial_obs_mask(rng_mask, grid_n)

        # Context for conditional FM inference: observed locations + noisy values
        ctx_idxs = jnp.where(obs_mask)[0]
        s_ctx    = s[ctx_idxs]                              # [n_obs, 2]
        f_ctx    = y_obs[ctx_idxs][None, :, None]          # [1, n_obs, 1]

        infer_model = build_gp_inference_model(s, priors)
        loader      = gen_train_dataloader(s, priors)
        cond_loader = gen_cond_fm_dataloader(s, priors)

        # -------- model registry -------------------------------------------
        # Each entry: (nn_model, train_step, valid_step, loader, use_ctx_in_hmc)
        models = {}

        # Baseline GP (no surrogate)
        models["Baseline GP"] = (None, None, None, None, False)

        # DeepRV + gMLP
        models["DeepRV + gMLP"] = (
            gMLPDeepRV(num_blks=2),
            deep_rv_train_step, deep_rv_valid_step, loader, False,
        )

        # FM-DeepRV (OT-CFM, multiple K)
        for k in FM_K_STEPS:
            models[f"FM-DeepRV (K={k})"] = (
                FlowMatchingDeepRV(vf=FlowMatchingVectorField(num_blks=2), n_steps=k),
                flow_matching_train_step, flow_matching_valid_step, loader, False,
            )

        # Bridge FM-DeepRV (same architecture as FM-DeepRV, different train step)
        for k in FM_K_STEPS:
            models[f"Bridge FM (K={k})"] = (
                FlowMatchingDeepRV(vf=FlowMatchingVectorField(num_blks=2), n_steps=k),
                bridge_matching_train_step, flow_matching_valid_step, loader, False,
            )

        # Conditional FM-DeepRV
        for k in FM_K_STEPS:
            models[f"Cond FM (K={k})"] = (
                ConditionalFMDeepRV(
                    vf=ConditionalVectorField(num_blks=2, d_ctx=D_CTX),
                    d_ctx=D_CTX, n_steps=k,
                ),
                conditional_fm_train_step, conditional_fm_valid_step, cond_loader, True,
            )

        # -------------------------------------------------------------------
        result, f_hats, model_names_done = [], [], []

        for model_name, (nn_model, train_step, valid_step, dl, use_ctx) in models.items():
            model_dir = grid_dir / model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
            model_dir.mkdir(parents=True, exist_ok=True)

            # Skip if already done
            if (model_dir / "single_res.pkl").exists():
                print(f"  [{model_name} @ {grid_n}²] done, loading.")
                with open(model_dir / "hmc_pp.pkl", "rb") as fh:
                    post = pickle.load(fh)
                with open(model_dir / "single_res.pkl", "rb") as fh:
                    res = pickle.load(fh)
                f_hats.append(post["mu"])
                model_names_done.append(model_name)
                result.append(res)
                continue

            print(f"\n=== {model_name} | {grid_n}×{grid_n} | ls={gt_ls} ===")

            train_time = eval_mse = surrogate_decoder = None

            if nn_model is not None:
                lr_sched  = cosine_annealing_lr(TRAIN_STEPS, MAX_LR)
                optimizer = optax.chain(
                    optax.clip_by_global_norm(3.0),
                    optax.adamw(lr_sched, weight_decay=1e-2),
                )
                wandb.init(config={"model": model_name, "L": L, "seed": seed},
                           mode="disabled", reinit=True)
                rng_train, rt = random.split(rng_train)
                rng_test,  rv = random.split(rng_test)
                train_time, eval_mse, surrogate_decoder = train_surrogate(
                    rt, rv, dl, train_step, valid_step, nn_model, model_dir, optimizer,
                )
                print(f"  train_time={train_time:.0f}s  eval_mse={eval_mse:.4f}")

            # HMC: conditional FM passes context, others don't
            rng_infer, ri = random.split(rng_infer)
            hmc_s_ctx = s_ctx if use_ctx else None
            hmc_f_ctx = f_ctx if use_ctx else None

            samples, mcmc, post, infer_time = run_hmc(
                ri, infer_model, y_obs, obs_mask, model_dir,
                surrogate_decoder=surrogate_decoder,
                s_ctx=hmc_s_ctx, f_ctx=hmc_f_ctx,
            )

            res = collect_result(
                model_name, train_time, infer_time, eval_mse,
                np.array(f_true), post, np.array(obs_mask),
                {k: v for k, v in samples.items() if k == "ls"},
                mcmc, L, seed,
            )
            with open(model_dir / "single_res.pkl", "wb") as fh:
                pickle.dump(res, fh)

            f_hats.append(post["mu"])
            model_names_done.append(model_name)
            result.append(res)

        pd.DataFrame(result).to_csv(grid_dir / "res.csv", index=False)
        plot_means(grid_n, np.array(f_true), f_hats, np.array(obs_mask),
                   model_names_done, grid_dir / "means.png")
        print(f"\n--- Grid {grid_n}×{grid_n} summary ---")
        df = pd.DataFrame(result)[["model_name", "MSE (unobs)", "ESS ls", "ESS z mean",
                                    "r_hat ls", "infer_time"]]
        print(df.to_string(index=False))

    aggregate_and_plot(save_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--ls",    type=int, default=20)
    args = p.parse_args()
    main(seed=args.seed, gt_ls=args.ls)
