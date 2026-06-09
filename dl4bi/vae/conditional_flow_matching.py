"""Conditional Flow Matching DeepRV.

Extends FlowMatchingDeepRV with a ContextEncoder that conditions the
vector field on observed function values at context locations.

At training time a random context mask provides (s_ctx, f_ctx); the
encoder produces a per-location embedding that is concatenated into
the vector field input alongside the standard (x_t, t, s, conditionals)
features.

At inference the HMC decoder conditions on the actual observations,
giving a flow prior p(f | s_ctx, f_ctx) that is already close to the
posterior — so HMC mixes much faster while still using the Poisson /
Gaussian likelihood for exact posterior correction.
"""

from typing import Callable, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array

from ..core.mlp import MLP, gMLP, gMLPBlock
from ..core.model_output import VAEOutput
from .train_utils import cond_as_feats


class ContextEncoder(nn.Module):
    """Cross-attention from all L target locations to n_ctx context points.

    Each target location attends over all context observations (s_ctx, f_ctx),
    producing a per-location embedding that encodes the observed information.
    Output shape is always [B, L, d_out] regardless of n_ctx.

    Args:
        d_out: Output embedding dimension per location.
    """

    d_out: int = 64

    @nn.compact
    def __call__(self, s: Array, s_ctx: Array, f_ctx: Array) -> Array:
        """
        Args:
            s:      [L, d_s]         all target spatial locations
            s_ctx:  [n_ctx, d_s]     context spatial locations
            f_ctx:  [B, n_ctx, 1]    observed function values at context locs
        Returns:
            [B, L, d_out]
        """
        B, n_ctx, _ = f_ctx.shape

        # Queries: one per target location (shared across batch)
        q = nn.Dense(self.d_out)(s)                                     # [L, d_out]

        # Keys / values: one per context point per batch item
        s_ctx_bc = jnp.tile(s_ctx[None], (B, 1, 1))                    # [B, n_ctx, d_s]
        kv_in    = jnp.concatenate([s_ctx_bc, f_ctx], axis=-1)         # [B, n_ctx, d_s+1]
        k = nn.Dense(self.d_out)(kv_in)                                 # [B, n_ctx, d_out]
        v = nn.Dense(self.d_out)(kv_in)                                 # [B, n_ctx, d_out]

        # Scaled dot-product cross-attention
        scale    = jnp.sqrt(float(self.d_out))
        attn     = jnp.einsum("ld,bnd->bln", q, k) / scale             # [B, L, n_ctx]
        attn     = jax.nn.softmax(attn, axis=-1)
        out      = jnp.einsum("bln,bnd->bld", attn, v)                 # [B, L, d_out]
        return out


class ConditionalVectorField(nn.Module):
    """gMLP vector field that accepts a pre-computed context embedding.

    Identical to FlowMatchingVectorField but concatenates an optional
    ctx_embed [B, L, d_ctx] to the per-location features before the
    gMLP stack.  The ContextEncoder lives in ConditionalFMDeepRV so the
    embedding is computed once per ODE step.

    Args:
        num_blks: Number of gMLP blocks.
        d_ctx:    Context embedding dimension (must match ContextEncoder.d_out).
        s_embed:  Optional spatial coordinate transform (default: identity).
        proj_in / proj_out / embed / head: gMLP sub-modules.
    """

    num_blks: int = 2
    d_ctx:    int = 64
    s_embed:  Union[Callable, nn.Module] = lambda s: s
    proj_in:  nn.Module = MLP([128, 128], nn.gelu)
    proj_out: nn.Module = MLP([64,  64],  nn.gelu)
    embed:    nn.Module = MLP([64,  64],  nn.gelu)
    head:     nn.Module = MLP([128,  1],  nn.gelu)

    @nn.compact
    def __call__(
        self,
        x_t:          Array,
        conditionals: Array,
        t:            Array,
        s:            Array,
        ctx_embed:    Optional[Array] = None,
        **kwargs,
    ) -> VAEOutput:
        """
        Args:
            x_t:       [B, L]        interpolated sample
            conditionals: [C]        kernel hyperparameters
            t:         [B]           per-sample time
            s:         [L, d_s]      spatial coordinates
            ctx_embed: [B, L, d_ctx] pre-computed context embedding (optional)
        Returns:
            VAEOutput with f_hat = predicted vector field [B, L, 1]
        """
        s_emb     = self.s_embed(s)
        batched_s = jnp.repeat(s_emb[None], x_t.shape[0], axis=0)     # [B, L, d_s]
        x = jnp.concatenate([jnp.atleast_3d(x_t), batched_s], axis=-1)
        x = cond_as_feats(x, conditionals)
        t_feat = jnp.repeat(t[:, None, None], x.shape[1], axis=1)      # [B, L, 1]
        x = jnp.concatenate([x, t_feat], axis=-1)

        if ctx_embed is not None:
            x = jnp.concatenate([x, ctx_embed], axis=-1)               # [B, L, d+d_ctx]

        v = gMLP(
            num_blks=self.num_blks,
            embed=self.embed,
            blk=gMLPBlock(self.proj_in, self.proj_out),
            head=self.head,
        )(x, **kwargs)
        return VAEOutput(v)


class ConditionalFMDeepRV(nn.Module):
    """Conditional Flow Matching surrogate decoder.

    Wraps a ConditionalVectorField with a ContextEncoder.  During training
    a random subset of the GP draw is passed as context; the encoder
    produces a per-location embedding injected into the VF input.

    At inference ``decode`` is a drop-in for FlowMatchingDeepRV.decode:
    the surrogate decoder returned by ``generate_surrogate_decoder`` passes
    ``s_ctx`` and ``f_ctx`` as kwargs, so HMC receives a context-aware prior.

    Args:
        vf:     ConditionalVectorField instance.
        d_ctx:  Context embedding size (must match vf.d_ctx).
        n_steps: Euler ODE steps at inference.
    """

    vf:     nn.Module
    d_ctx:  int = 64
    n_steps: int = 1

    def setup(self):
        self.encoder = ContextEncoder(d_out=self.d_ctx)

    def __call__(
        self,
        z:            Array,
        conditionals: Array,
        t:            Optional[Array] = None,
        s:            Optional[Array] = None,
        s_ctx:        Optional[Array] = None,
        f_ctx:        Optional[Array] = None,
        **kwargs,
    ) -> VAEOutput:
        if t is None:
            t = jnp.zeros((z.shape[0],))
        ctx_embed = None
        if s_ctx is not None and f_ctx is not None and s is not None:
            ctx_embed = self.encoder(s, s_ctx, f_ctx)
        return self.vf(z, conditionals, t, s=s, ctx_embed=ctx_embed, **kwargs)

    def decode(
        self,
        z:            Array,
        conditionals: Array,
        s:            Optional[Array] = None,
        s_ctx:        Optional[Array] = None,
        f_ctx:        Optional[Array] = None,
        **kwargs,
    ) -> Array:
        """Integrate ODE from z, encoding context once before the loop."""
        ctx_embed = None
        if s_ctx is not None and f_ctx is not None and s is not None:
            ctx_embed = self.encoder(s, s_ctx, f_ctx)

        x  = z
        dt = 1.0 / self.n_steps
        for i in range(self.n_steps):
            t = jnp.full((x.shape[0],), (i + 0.5) * dt)
            v = self.vf(x, conditionals, t, s=s, ctx_embed=ctx_embed, **kwargs).f_hat
            x = x + dt * v.reshape(x.shape)
        return x
