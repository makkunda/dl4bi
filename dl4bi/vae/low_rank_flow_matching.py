"""Low-Rank Flow Matching DeepRV.

Replaces the L-dimensional HMC latent z ∈ ℝ^L with an M-dimensional inducing
latent z ∈ ℝ^M, where M << L.

Architecture
------------
                z ~ N(0, I_M)          M-dim HMC (much cheaper)
                     │
            LowRankVectorField          gMLP in M-space, conditioned on
                     │                  inducing locations u [M, d]
              f_m ∈ ℝ^M               inducing function values
                     │
            SpatialDecoder             cross-attention: each of L target
                     │                 locations attends over M inducing pts
              f ∈ ℝ^L                 full-resolution field

Training
--------
  Two losses are combined in a single backward pass:
    L_FM  : OT-CFM in M-space  (z_M → f_m)
    L_dec : decoder reconstruction (f_m → f_full)

  Both losses propagate gradients so VF and decoder are trained jointly.

Inference
---------
  decode(z_m, cond, u=u, s=s)  is a drop-in for FlowMatchingDeepRV.decode.
  generate_surrogate_decoder from train_utils works unchanged.
  The HMC model samples z ~ N(0, I_M) — ESS scales far better than L-dim.
"""

from typing import Callable, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array

from ..core.mlp import MLP, gMLP, gMLPBlock
from ..core.model_output import VAEOutput
from .train_utils import cond_as_feats


class SpatialDecoder(nn.Module):
    """Cross-attention upsampler: (f_m [B,M], u [M,d], s [L,d]) → f [B,L].

    Each of the L target locations attends over the M inducing points,
    using both their spatial position u and their function value f_m as
    keys/values.  The output is a per-location scalar (the predicted field).

    Args:
        d_model: Attention embedding dimension.
        n_head_layers: Number of MLP layers in the output head.
    """

    d_model: int = 64

    @nn.compact
    def __call__(self, f_m: Array, u: Array, s: Array) -> Array:
        """
        Args:
            f_m: [B, M]   inducing function values
            u:   [M, d]   inducing spatial locations
            s:   [L, d]   all target spatial locations
        Returns:
            [B, L]  predicted field at all L locations
        """
        B, M = f_m.shape
        L    = s.shape[0]

        # Queries: one per target location (broadcast over batch)
        q = nn.Dense(self.d_model)(s)                                   # [L, d_model]

        # Keys / values: combine inducing location + function value
        u_bc  = jnp.tile(u[None], (B, 1, 1))                           # [B, M, d]
        kv_in = jnp.concatenate([u_bc, f_m[:, :, None]], axis=-1)      # [B, M, d+1]
        k = nn.Dense(self.d_model)(kv_in)                               # [B, M, d_model]
        v = nn.Dense(self.d_model)(kv_in)                               # [B, M, d_model]

        # Scaled dot-product cross-attention
        scale = jnp.sqrt(float(self.d_model))
        attn  = jnp.einsum("ld,bmd->blm", q, k) / scale                # [B, L, M]
        attn  = jax.nn.softmax(attn, axis=-1)
        out   = jnp.einsum("blm,bmd->bld", attn, v)                    # [B, L, d_model]

        # Small MLP head → scalar per location
        f = MLP([64, 1], nn.gelu)(out).squeeze(-1)                      # [B, L]
        return f


class LowRankVectorField(nn.Module):
    """gMLP vector field operating in M-dimensional inducing space.

    Identical structure to FlowMatchingVectorField but expects x_t ∈ ℝ^M
    and uses inducing locations u [M, d] as positional features (instead
    of all L locations s).

    Args:
        num_blks: Number of gMLP blocks.
        s_embed:  Optional coordinate transform (default: identity).
        proj_in / proj_out / embed / head: gMLP sub-modules.
    """

    num_blks: int = 2
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
        u:            Array,
        **kwargs,
    ) -> VAEOutput:
        """
        Args:
            x_t:          [B, M]   interpolated inducing values
            conditionals: [C]      kernel hyperparameters
            t:            [B]      per-sample time
            u:            [M, d]   inducing spatial locations
        Returns:
            VAEOutput with f_hat = predicted M-dim velocity [B, M, 1]
        """
        u_emb     = self.s_embed(u)
        batched_u = jnp.repeat(u_emb[None], x_t.shape[0], axis=0)     # [B, M, d]
        x = jnp.concatenate([jnp.atleast_3d(x_t), batched_u], axis=-1)
        x = cond_as_feats(x, conditionals)
        t_feat = jnp.repeat(t[:, None, None], x.shape[1], axis=1)      # [B, M, 1]
        x = jnp.concatenate([x, t_feat], axis=-1)
        v = gMLP(
            num_blks=self.num_blks,
            embed=self.embed,
            blk=gMLPBlock(self.proj_in, self.proj_out),
            head=self.head,
        )(x)
        return VAEOutput(v)


class LowRankFMDeepRV(nn.Module):
    """Flow Matching DeepRV with M-dimensional inducing latent.

    HMC samples z ~ N(0, I_M) — M << L — then:
      1. The ODE integrates in M-space:   z → f_m   (LowRankVectorField)
      2. The SpatialDecoder upsamples:    f_m → f   (cross-attention)

    decode() is a drop-in for FlowMatchingDeepRV.decode.
    generate_surrogate_decoder from train_utils works unchanged.

    Training uses a combined loss:
      L_FM  : OT-CFM velocity loss in M-space
      L_dec : full-field MSE through the spatial decoder

    Both components are trained jointly in lowrank_fm_train_step.

    Args:
        vf:      LowRankVectorField instance.
        decoder: SpatialDecoder instance.
        n_steps: Euler ODE steps at inference.
    """

    vf:      LowRankVectorField
    decoder: SpatialDecoder
    n_steps: int = 1

    def __call__(
        self,
        z:            Array,
        conditionals: Array,
        t:            Optional[Array] = None,
        u:            Optional[Array] = None,
        s:            Optional[Array] = None,
        **kwargs,
    ) -> VAEOutput:
        """Training forward: predict velocity in M-space.

        Also touches the SpatialDecoder when s is present so that
        decoder parameters are initialised during model.init().
        """
        if t is None:
            t = jnp.zeros((z.shape[0],))
        vf_out = self.vf(z, conditionals, t, u=u)

        # Ensure decoder is initialised when s is available (init pass)
        if s is not None:
            _ = self.decoder(z, u, s)

        return vf_out

    def decode(
        self,
        z:            Array,
        conditionals: Array,
        u:            Optional[Array] = None,
        s:            Optional[Array] = None,
        **kwargs,
    ) -> Array:
        """ODE integration in M-space followed by spatial upsampling.

        Compatible with generate_surrogate_decoder — call as:
            surrogate_decoder(z_m, cond, u=u, s=s)
        where z_m has shape [1, M].
        """
        x  = z
        dt = 1.0 / self.n_steps
        for i in range(self.n_steps):
            t = jnp.full((x.shape[0],), (i + 0.5) * dt)
            v = self.vf(x, conditionals, t, u=u).f_hat
            x = x + dt * v.reshape(x.shape)
        return self.decoder(x, u, s)    # [B, L]

    def decode_from_inducing(
        self,
        f_m: Array,
        u:   Array,
        s:   Array,
    ) -> Array:
        """Decode from exact inducing values — used in the decoder training loss."""
        return self.decoder(f_m, u, s)
