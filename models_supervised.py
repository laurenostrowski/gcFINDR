"""
FINDR with fixed-window temporal attention and centroid distance loss
for language-informed analysis of sEEG dynamics.

Architecture overview
─────────────────────
The base FINDR model (Kim et al., 2025) is extended with:

1.  Per-channel gating on the spike inputs before the inference network,
    so that both reconstruction and classification losses drive gate
    selection. The centroid loss gradient flows:
      class_loss → z_attended → z → posterior → inference_network
                 → gated_spike_inputs → channel_gates_raw

2.  Fixed Gaussian temporal windows that tile the trial epoch from the
    language cue through speech production.  Each latent dimension l
    is paired with a fixed Gaussian window at center μ_l, producing a
    per-dimension temporal summary z̄_l.

    The windows are NOT learnable.  They are an analysis basis — like
    frequency bands in a spectrogram — determined entirely by the
    number of latent dimensions (L), the overlap fraction, and the
    trial event timing.  All temporal specificity in the classification
    comes from the centroid difference vector (class_axis), which
    weights dimensions by how much they separate language conditions.

    This design has zero learnable parameters in the classification
    pathway, preventing any form of memorization or gradient death.

Window geometry:
    Given L dimensions and overlap_frac (default 0.5):
      band_w = span / [1 + (L-1)(1 - overlap_frac)]
      stride = band_w * (1 - overlap_frac)
      σ_l    = stride  (FWHM ≈ 2.35 * stride)
      μ_l    = frac_lo + band_w/2 + l * stride

    Time is normalised per trial as:
      t_frac = (t - t_image) / (t_speech_on - t_image)
    so t_frac=0 is image onset, t_frac=1 is speech onset.
    frac_lo is computed from the flag onset per batch.
"""

import functools
import jax
from jax import random, numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from typing import Any, Sequence, Tuple

from models import (
    flip_sequences,
    PosteriorSDE,
    GNSDE,
    Constrained_GNSDE,
    NTRGRU,
    Reduce,
    InitialState,
    SimpleGRU,
    SimpleBiGRU,
    Flow,
)

Array = Any
PRNGKey = Any


class FINDR_Supervised(nn.Module):
    """FINDR with fixed Gaussian temporal windows and centroid distance loss.

    The temporal windows are determined by L, overlap_frac, and the
    per-trial event timing.  No parameters are learned in the
    classification pathway.

    Constructor args beyond FINDR:
        sampling_rate:   sEEG sampling rate in Hz (default 128.0)
        overlap_frac:    fraction of overlap between adjacent windows
                         (0.0 = no overlap, 0.5 = 50%, default 0.5)
        frac_hi:         upper bound of temporal coverage in fractional
                         time (default 1.2 = 20% past speech onset)
    """

    features_prior: Sequence[int]
    features_posterior: Sequence[int]
    task_related_latent_size: int
    inference_network_size: int
    num_neurons: int
    alpha: float = 1.0
    noise_level: float = 1.0
    constrain_prior: bool = False
    use_gaussian: bool = False
    use_channel_gating: bool = False
    sampling_rate: float = 128.0
    overlap_frac: float = 0.5
    frac_hi: float = 1.2

    def setup(self):
        L = self.task_related_latent_size

        # ── original FINDR components (unchanged) ────────────────────────

        self.inference_network = SimpleBiGRU(
            hidden_size=self.inference_network_size,
        )
        self.posterior_process = PosteriorSDE(
            features=self.features_posterior,
            alpha=self.alpha,
            noise_level=self.noise_level,
        )
        self.prior_process = Flow(
            features=self.features_prior,
            alpha=self.alpha,
        )
        self.task_related_latents_to_neurons = nn.Dense(
            self.num_neurons,
            name='task_related_latents_to_neurons',
            use_bias=True,
        )

        if self.use_gaussian:
            self.log_obs_var = self.param(
                'log_obs_var',
                initializers.constant(-2.0),
                (self.num_neurons,),
            )

        if self.use_channel_gating:
            self.channel_gates_raw = self.param(
                'channel_gates_raw',
                initializers.zeros_init(),
                (self.num_neurons,),
            )

        # No learnable temporal parameters.
        # Window geometry is computed in __call__ from L, overlap_frac,
        # frac_hi, and the per-batch flag/image/speech timing.

    # ─────────────────────────────────────────────────────────────────────

    def __call__(
        self,
        spike_inputs,       # (batch, time, num_neurons)
        external_inputs,    # (batch, time, n_extinp) — language flag stripped
        baseline_inputs,    # (batch, time, num_neurons)
        trial_lengths,      # (batch,) int
        event_samples,      # (batch, 4) int — [flag, image, speech_on, speech_off]
        rng: PRNGKey,
    ) -> Tuple:

        key_1, key_2, key_3 = random.split(rng, 3)
        batch_size = len(trial_lengths)
        L = self.task_related_latent_size

        # ── encode ───────────────────────────────────────────────────────
        # Apply channel gates to spike inputs before inference network so
        # that both reconstruction and classification losses drive gate
        # selection. The centroid loss gradient flows:
        #   class_loss → z_attended → z → posterior → inference_network
        #              → gated_spike_inputs → channel_gates_raw
        if self.use_channel_gating:
            gates = jax.nn.sigmoid(self.channel_gates_raw)          # (N,)
            spike_inputs_gated = spike_inputs * gates[None, None, :] # (batch, T, N)
        else:
            gates = None
            spike_inputs_gated = spike_inputs

        hs = self.inference_network(
            spike_inputs_gated, external_inputs, trial_lengths, key_1,
        )
        
        # ── posterior ────────────────────────────────────────────────────
        carry_dl = self.posterior_process.initialize_carry(
            key_2, batch_size, L,
        )
        noise_posterior = random.normal(
            key_3, hs.shape[:-1] + (L,),
        )
        _, (z, mu_phi, mu, std) = self.posterior_process(
            carry_dl, hs, external_inputs, noise_posterior,
        )

        # ── prior ────────────────────────────────────────────────────────
        mu_theta = self.prior_process(z, external_inputs)

        # ── generator (linear projection) ────────────────────────────────
        outputs = self.task_related_latents_to_neurons(z)

        # ── fixed Gaussian temporal windows ──────────────────────────────
        #
        # Tile the task-relevant epoch [frac_lo, frac_hi] with L
        # evenly-spaced Gaussian windows at the specified overlap.

        T = z.shape[1]
        flag_onset_bins = event_samples[:, 0]                            # (batch,)
        image_onset_bins = event_samples[:, 1]                           # (batch,)
        speech_onset_bins = event_samples[:, 2]                          # (batch,)
        time_bins = jnp.arange(T)[None, :]                              # (1, T)

        # fractional time: 0 = image onset, 1 = speech onset (per trial)
        interval_bins = jnp.maximum(
            speech_onset_bins - image_onset_bins, 1.0,
        )                                                                # (batch,)
        t_frac = (time_bins - image_onset_bins[:, None]) / interval_bins[:, None]

        # frac_lo from flag onset (mean across batch)
        flag_frac = (flag_onset_bins - image_onset_bins) / interval_bins
        frac_lo = jnp.mean(flag_frac)

        # window geometry (all deterministic, no learnable params)
        span = self.frac_hi - frac_lo
        band_w = span / (1.0 + (L - 1) * (1.0 - self.overlap_frac))
        stride = band_w * (1.0 - self.overlap_frac)
        sigma = stride                                                   # FWHM ≈ 2.35 * stride

        # fixed centers: evenly spaced within [frac_lo, frac_hi]
        centers = frac_lo + band_w / 2.0 + jnp.arange(L) * stride       # (L,)

        # Gaussian weights: (batch, T, L)
        gauss_weights = jnp.exp(
            -0.5 * ((t_frac[:, :, None] - centers[None, None, :])
                     / sigma) ** 2
        )

        # mask: only attend to [flag_onset, trial_length)
        valid_mask = (
            (time_bins >= flag_onset_bins[:, None])
            & (time_bins < trial_lengths[:, None])
        )                                                                # (batch, T)
        gauss_weights = gauss_weights * valid_mask[:, :, None]

        # normalise per dimension so weights sum to 1 over time
        gauss_weights = gauss_weights / jnp.maximum(
            gauss_weights.sum(axis=1, keepdims=True), 1e-8,
        )

        # per-dimension weighted average: (batch, L)
        z_attended = jnp.sum(z * gauss_weights, axis=1)

        # ── return ───────────────────────────────────────────────────────
        if self.use_gaussian:
            log_obs_var_clipped = jnp.clip(self.log_obs_var, -6.0, 0.0)
            obs_var = jnp.exp(log_obs_var_clipped)
            return (outputs, obs_var, z, mu, mu_theta, mu_phi, std,
                    gates, z_attended)
        else:
            return (outputs, None, z, mu, mu_theta, mu_phi, std,
                    gates, z_attended)
