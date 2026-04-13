"""
Training loop for FINDR_Supervised with fixed temporal windows and
centroid distance loss.

The classification pathway has ZERO learnable parameters:
  - Temporal windows are fixed (determined by L, overlap_frac, events)
  - Centroid distance loss has no head weights

    L_class = lambda_class * E[-log p(y | z_attended, mu_0, mu_1)]
        where p(y=c) = softmax(-||z_attended - mu_c||^2)

Gradient flows through z_bar → z → posterior → encoder for the entire
duration of training.  Nothing memorises.  Nothing saturates.
"""

import functools
from absl import app, flags, logging

import jax
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, Tuple, Sequence
from jax.nn import softplus
from jax import lax, random, numpy as jnp
from jax.scipy.special import gammaln
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import serialization
from flax.training import train_state, orbax_utils
import orbax.checkpoint as ocp
from flax import traverse_util
from flax import struct
from sklearn.model_selection import KFold, StratifiedKFold
import optax
import ml_collections
import models_supervised as models
import findr_utils as utils
import os

BIN_WIDTH = 0.01
SMALL_CONSTANT = 1e-5
Array = Any
FLAGS = flags.FLAGS
PRNGKey = Any


# ═════════════════════════════════════════════════════════════════════════
#  Model initialisation
# ═════════════════════════════════════════════════════════════════════════

def create_train_state(
    rng: PRNGKey,
    config: ml_collections.ConfigDict,
    learning_rate_fn,
    xs,
    ckptdir=None,
):
    """Creates an initial TrainState."""
    key_1, key_2, key_3 = random.split(rng, 3)
    model = models.FINDR_Supervised(
        alpha=config.alpha,
        noise_level=config.noise_level,
        features_prior=config.features_prior,
        features_posterior=config.features_posterior,
        task_related_latent_size=config.task_related_latent_size,
        inference_network_size=config.inference_network_size,
        num_neurons=xs['spikes'].shape[-1],
        constrain_prior=config.constrain_prior,
        use_gaussian=config.use_gaussian,
        use_channel_gating=config.use_channel_gating,
        overlap_frac=config.overlap_frac,
        frac_hi=config.frac_hi,
    )
    if ckptdir is not None:
        from flax.training import checkpoints
        raw_restored = checkpoints.restore_checkpoint(
            ckpt_dir=ckptdir, target=None, parallel=False,
        )
        params = freeze(raw_restored['model']['params'])
    else:
        params = model.init(
            key_2,
            xs['spikes'],
            xs['externalinputs'],
            xs['baselineinputs'],
            xs['lengths'],
            xs['event_samples'],
            key_3,
        )['params']

    tx = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.sgd(learning_rate_fn, config.momentum),
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
    )
    return state


def compute_centroids(state, train_ds, rng):
    """Full forward pass over training set to compute stable class centroids."""
    all_z_attended = []
    n = len(train_ds['spikes'])
    chunk_size = 20
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = {k: v[start:end] for k, v in train_ds.items()}
        rng, key = random.split(rng)
        (_, _, _, _, _, _, _, _, z_attended) = state.apply_fn(
            {'params': state.params},
            chunk['spikes'],
            chunk['externalinputs'],
            chunk['baselineinputs'],
            chunk['lengths'],
            chunk['event_samples'],
            key,
        )
        all_z_attended.append(np.array(z_attended))

    all_z_attended = np.concatenate(all_z_attended, axis=0)  # (N, L)
    labels = train_ds['language_labels']
    mask_0 = labels == 0
    mask_1 = labels == 1
    mu_0 = jnp.array(all_z_attended[mask_0].mean(axis=0))   # (L,)
    mu_1 = jnp.array(all_z_attended[mask_1].mean(axis=0))   # (L,)
    return mu_0, mu_1


def make_episodic_split(language_labels, query_frac=0.2, rng_seed=None):
    """Stratified support/query split of training indices."""
    rng = np.random.default_rng(rng_seed)
    query_mask = np.zeros(len(language_labels), dtype=bool)
    for cls in np.unique(language_labels):
        cls_idx = np.where(language_labels == cls)[0]
        n_query = max(1, int(len(cls_idx) * query_frac))
        query_idx = rng.choice(cls_idx, size=n_query, replace=False)
        query_mask[query_idx] = True
    return query_mask


# ═════════════════════════════════════════════════════════════════════════
#  Forward pass + loss  (JIT-compiled)
# ═════════════════════════════════════════════════════════════════════════

@functools.partial(jax.jit, static_argnums=[3, 4, 5, 6, 7, 10, 11, 12])
def apply_model(
    state: train_state.TrainState,
    batch,
    loss_weights: Array,
    alpha: float,               # 3  static
    beta_coeff: float,          # 4  static
    beta_inc_rate: float,       # 5  static
    lossw_inc_rate: float,      # 6  static
    l2_coeff: float,            # 7  static
    beta_counter: int,          # 8
    lossw_counter: int,         # 9
    learning_rate_fn,           # 10 static
    lambda_sparse: float,       # 11 static
    lambda_class: float,        # 12 static
    mu_class_0: Array,          # 13 fixed centroid (L,)
    mu_class_1: Array,          # 14 fixed centroid (L,)
    query_mask: Array,          # 15 boolean (batch,) — True = query trial
    rng: PRNGKey,               # 16
):
    ntrgru_mask = traverse_util.path_aware_map(
        lambda path, _: 1 if 'non_task_related_gru' in path else 2,
        state.params,
    )

    def loss_fn(params):
        # ── forward pass ─────────────────────────────────────────────
        (outputs, obs_var, z, mu, mu_theta, mu_phi, std,
         gates, z_attended) = state.apply_fn(
            {'params': params},
            batch['spikes'],
            batch['externalinputs'],
            batch['baselineinputs'],
            batch['lengths'],
            batch['event_samples'],
            rng,
        )

        # ── reconstruction loss ──────────────────────────────────────
        nll_loss = neg_gaussian_log_likelihood(
            outputs, obs_var, batch['spikes'], batch['lengths'],
            loss_weights, lossw_inc_rate, lossw_counter,
        )

        # ── KL divergence ────────────────────────────────────────────
        kld_loss = beta_coeff * beta(beta_inc_rate, beta_counter) * kl_divergence(
            alpha, mu_theta, mu_phi, std,
            batch['lengths'], loss_weights, lossw_inc_rate, lossw_counter,
        )

        loss = nll_loss + kld_loss

        # ── gate sparsity ────────────────────────────────────────────
        if gates is not None:
            sparse_loss = lambda_sparse * jnp.sum(jnp.abs(gates))
            loss = loss + sparse_loss
        else:
            sparse_loss = 0.0

        # ── softmax centroid loss (query trials only) ─────────────────
        labels = batch['language_labels'].astype(jnp.int32)
        query_float = query_mask.astype(jnp.float32)
        n_query = jnp.maximum(query_float.sum(), 1.0)

        mask_0 = (labels == 0).astype(jnp.float32) * query_float
        mask_1 = (labels == 1).astype(jnp.float32) * query_float
        both_present = (mask_0.sum() > 0) & (mask_1.sum() > 0)

        d0 = jnp.sum((z_attended - mu_class_0[None, :]) ** 2, axis=1)
        d1 = jnp.sum((z_attended - mu_class_1[None, :]) ** 2, axis=1)
        logits    = jnp.stack([-d0, -d1], axis=1)
        log_probs = jax.nn.log_softmax(logits, axis=1)
        batch_size = labels.shape[0]

        per_trial_loss = -log_probs[jnp.arange(batch_size), labels]
        class_loss = jnp.sum(per_trial_loss * query_float) / n_query
        class_loss = class_loss * both_present
        loss = loss + lambda_class * class_loss

        # ── L2 regularisation ────────────────────────────────────────
        loss += sum(
            l2_loss(w, alpha=l2_coeff) if label == 1
            else l2_loss(w, alpha=1e-7)
            for label, w in zip(
                jax.tree.leaves(ntrgru_mask),
                jax.tree.leaves(params),
            )
        )

        # ── nearest-centroid accuracy (query trials only) ─────────────
        preds = (d1 < d0).astype(jnp.int32)
        correct = (preds == labels).astype(jnp.float32) * query_float
        class_acc = jnp.where(both_present, correct.sum() / n_query, 0.5)

        return loss, (nll_loss, kld_loss, sparse_loss,
                      class_loss, class_acc, outputs, gates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (nll, kld, sparse, class_loss, cls_acc,
            outputs, gates)), grads = grad_fn(state.params)

    return grads, loss, nll, kld, sparse, class_loss, cls_acc, gates


@jax.jit
def update_model(state: train_state.TrainState, grads):
    return state.apply_gradients(grads=grads)


# ═════════════════════════════════════════════════════════════════════════
#  Training epoch
# ═════════════════════════════════════════════════════════════════════════

def train_epoch(
    state: train_state.TrainState,
    train_ds,
    loss_weights: Array,
    config: ml_collections.ConfigDict,
    beta_counter: int,
    lossw_counter: int,
    learning_rate_fn,
    mu_class_0: Array,
    mu_class_1: Array,
    effective_lambda_class: float,
    query_mask_ds: Array,
    rng: PRNGKey,
):
    """Train for a single epoch."""
    key_1, key_2 = random.split(rng, 2)
    train_ds_size = len(train_ds['externalinputs'])
    steps_per_epoch = train_ds_size // config.batch_size

    perms = random.permutation(key_1, train_ds_size)
    perms = perms[:steps_per_epoch * config.batch_size]
    perms = perms.reshape((steps_per_epoch, config.batch_size))

    epoch_loss, epoch_nll, epoch_kld = [], [], []
    epoch_centroid_dist, epoch_cls_acc = [], []
    epoch_gate_stats = []

    for perm in perms:
        key_2, key_3 = random.split(key_2, 2)
        batch = {
            'spikes':           train_ds['spikes'][perm, ...],
            'externalinputs':   train_ds['externalinputs'][perm, ...],
            'lengths':          train_ds['lengths'][perm, ...],
            'baselineinputs':   train_ds['baselineinputs'][perm, ...],
            'language_labels':  train_ds['language_labels'][perm, ...],
            'event_samples':    train_ds['event_samples'][perm, ...],
        }
        query_mask_batch = jnp.array(query_mask_ds[perm])

        (grads, loss, nll, kld, sparse,
         class_loss, cls_acc, gates) = apply_model(
            state, batch, loss_weights,
            config.alpha, config.beta,
            config.beta_inc_rate, config.lossw_inc_rate,
            config.l2_coeff,
            beta_counter, lossw_counter,
            learning_rate_fn,
            config.lambda_sparse, effective_lambda_class,
            mu_class_0, mu_class_1,
            query_mask_batch,
            key_3,
        )

        if gates is not None:
            epoch_gate_stats.append({
                'mean': jnp.mean(gates),
                'n_active': jnp.sum(gates > 0.5),
            })

        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_nll.append(nll)
        epoch_kld.append(kld)
        epoch_centroid_dist.append(class_loss)
        epoch_cls_acc.append(cls_acc)

    train_loss          = np.mean(epoch_loss)
    train_nll           = np.mean(epoch_nll)
    train_kld           = np.mean(epoch_kld)
    train_centroid_dist = np.mean(epoch_centroid_dist)
    train_cls_acc       = np.mean(epoch_cls_acc)

    if epoch_gate_stats:
        gate_mean     = np.mean([s['mean'] for s in epoch_gate_stats])
        gate_n_active = int(np.mean([s['n_active'] for s in epoch_gate_stats]))
    else:
        gate_mean     = None
        gate_n_active = None

    return (state, train_loss, train_nll, train_kld,
            train_centroid_dist, train_cls_acc, gate_mean, gate_n_active)


# ═════════════════════════════════════════════════════════════════════════
#  Main training loop
# ═════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    config: ml_collections.ConfigDict,
    datapath: str,
    workdir: str,
    randseedpath: str = None,
    ckpt_save: bool = True,
) -> float:
    """Execute model training and evaluation loop."""

    print(f"Starting train_and_evaluate with datapath: {datapath}")
    print(f"  lambda_class: {config.lambda_class}")
    print(f"  overlap_frac: {config.overlap_frac}")
    print(f"  frac_hi: {config.frac_hi}")

    # ── load data ────────────────────────────────────────────────────
    print("Loading data...")
    train_ds, val_ds, test_ds, ds, perms = get_datasets(
        datapath, workdir, randseedpath,
        k_cv=config.k_cv,
        n_splits=config.n_splits,
        baseline_fit=config.baseline_fit,
    )
    print(f"Data loaded. Train: {len(train_ds['spikes'])}, "
          f"Val: {len(val_ds['spikes'])}, Test: {len(test_ds['spikes'])}")
    print(f"  Train labels — Dutch: {int((train_ds['language_labels']==0).sum())}, "
          f"English: {int((train_ds['language_labels']==1).sum())}")

    # compute and log fixed window positions
    L = config.task_related_latent_size
    es = train_ds['event_samples']
    interval_bins = np.maximum(es[:, 2] - es[:, 1], 1.0)
    flag_frac = (es[:, 0] - es[:, 1]) / interval_bins
    frac_lo = float(flag_frac.mean())
    mean_interval_sec = float((interval_bins / 128.0).mean())

    span = config.frac_hi - frac_lo
    band_w = span / (1.0 + (L - 1) * (1.0 - config.overlap_frac))
    stride_val = band_w * (1.0 - config.overlap_frac)
    fixed_centers = [frac_lo + band_w / 2.0 + l * stride_val for l in range(L)]
    fixed_centers_sec = [c * mean_interval_sec for c in fixed_centers]

    print(f"\n  Fixed temporal windows (L={L}, overlap={config.overlap_frac:.0%}):")
    print(f"  frac_lo={frac_lo:.3f} ({frac_lo * mean_interval_sec:.3f}s), "
          f"frac_hi={config.frac_hi}")
    print(f"  band_w={band_w:.3f} ({band_w * mean_interval_sec:.3f}s), "
          f"stride={stride_val:.3f} ({stride_val * mean_interval_sec:.3f}s)")
    for l in range(L):
        print(f"    Dim {l}: center={fixed_centers_sec[l]:+.3f}s "
              f"({fixed_centers[l]:+.3f}x)")
    print()

    # ── initialise model ─────────────────────────────────────────────
    print("Initialising model...")
    rng = random.PRNGKey(1)
    key_1, key_2 = random.split(rng, 2)
    train_ds_size = len(train_ds['externalinputs'])
    steps_per_epoch = train_ds_size // config.batch_size
    print(f"  steps_per_epoch: {steps_per_epoch}")
    learning_rate_fn = create_learning_rate_fn(config, steps_per_epoch)
    state = create_train_state(key_1, config, learning_rate_fn, test_ds)
    best_state = state

    # ── checkpointing ────────────────────────────────────────────────
    mgr_options = ocp.CheckpointManagerOptions(create=True, max_to_keep=30)
    ckpt_mgr = ocp.CheckpointManager(
        workdir,
        ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        mgr_options,
    )

    annealing_epochs = np.floor(
        np.log(0.01) / np.log(config.beta_inc_rate)
    ).astype(int)
    effective_earlymiddle = annealing_epochs

    # ── temporal loss weights (unchanged from original) ──────────────
    early_loss_weights = -1.0 * jnp.ones_like(train_ds['spikes'][0, :, 0])
    early_loss_weights = early_loss_weights.at[:30].set(0.0)
    middle_loss_weights = -1.0 * jnp.ones_like(train_ds['spikes'][0, :, 0])
    middle_loss_weights = middle_loss_weights.at[:50].set(0.0)
    late_loss_weights = jnp.zeros_like(train_ds['spikes'][0, :, 0])

    # ── tracking arrays ──────────────────────────────────────────────
    train_losses,          train_nlls,          train_klds          = [], [], []
    val_losses,            val_nlls,            val_klds            = [], [], []
    test_losses,           test_nlls,           test_klds           = [], [], []
    train_centroid_dists,  train_cls_accs                          = [], []
    val_centroid_dists,    val_cls_accs                            = [], []
    test_centroid_dists,   test_cls_accs                           = [], []
    mu_class_0 = jnp.zeros(L)
    mu_class_1 = jnp.zeros(L)
    query_mask_val  = jnp.ones(len(val_ds['spikes']),  dtype=bool)
    query_mask_test = jnp.ones(len(test_ds['spikes']), dtype=bool)

    # ── epoch loop ───────────────────────────────────────────────────
    for epoch in range(1, config.num_epochs + 1):
        key_2, key_3, key_4, key_5, key_centroids = random.split(key_2, 5)

        if epoch < config.earlymiddle_epochs / 3:
            lw, bc, lc = early_loss_weights, 0, 1e6
        elif epoch < config.earlymiddle_epochs:
            lw, bc, lc = middle_loss_weights, 0, 1e6
        else:
            e = epoch - config.earlymiddle_epochs
            lw, bc, lc = late_loss_weights, e, e

        effective_lambda_class = (
            0.0 if epoch <= effective_earlymiddle
            else config.lambda_class
        )

        # ── episodic support/query split and centroid update ──────────
        if epoch > effective_earlymiddle:
            epoch_rng_seed = int(np.array(key_centroids)[0])
            query_mask_ds = make_episodic_split(
                train_ds['language_labels'], query_frac=0.2,
                rng_seed=epoch_rng_seed,
            )
            support_ds = {k: v[~query_mask_ds] for k, v in train_ds.items()}
            mu_class_0, mu_class_1 = compute_centroids(
                state, support_ds, key_centroids,
            )
        else:
            query_mask_ds = np.ones(len(train_ds['language_labels']), dtype=bool)

        (state, train_loss, train_nll, train_kld,
         train_cd, train_acc, gate_mean, gate_n_active) = train_epoch(
            state, train_ds, lw, config, bc, lc, learning_rate_fn,
            mu_class_0, mu_class_1, effective_lambda_class,
            query_mask_ds, key_3,
        )
        
        # ── validation ───────────────────────────────────────────────────
        (_, val_loss, val_nll, val_kld, _,
         val_cd, val_acc, _) = apply_model(
            state, val_ds, late_loss_weights,
            config.alpha, config.beta, config.beta_inc_rate,
            config.lossw_inc_rate, config.l2_coeff,
            1e6, 0, learning_rate_fn,
            config.lambda_sparse, effective_lambda_class,
            mu_class_0, mu_class_1,
            query_mask_val,
            key_4,
        )

        # ── test ─────────────────────────────────────────────────────────
        (_, test_loss, test_nll, test_kld, _,
         test_cd, test_acc, _) = apply_model(
            state, test_ds, late_loss_weights,
            config.alpha, config.beta, config.beta_inc_rate,
            config.lossw_inc_rate, config.l2_coeff,
            1e6, 0, learning_rate_fn,
            config.lambda_sparse, effective_lambda_class,
            mu_class_0, mu_class_1,
            query_mask_test,
            key_5,
        )

        # ── best model tracking ──────────────────────────────────────
        if epoch > annealing_epochs:
            past = val_losses[annealing_epochs:]
            if past and val_loss < min(past):
                best_state = state

        # ── logging ──────────────────────────────────────────────────
        log_parts = [
            f'epoch: {epoch}',
            f'train_loss: {train_loss:.4f}',
            f'train_nll: {train_nll:.4f}',
            f'train_kld: {train_kld:.4f}',
            f'centroid_dist: {train_cd:.4f}',
            f'train_nca: {train_acc:.2%}',
            f'val_loss: {val_loss:.4f}',
            f'val_nll: {val_nll:.4f}',
            f'val_kld: {val_kld:.4f}',
            f'val_nca: {val_acc:.2%}',
            f'test_loss: {test_loss:.4f}',
            f'test_nll: {test_nll:.4f}',
            f'test_kld: {test_kld:.4f}',
            f'test_nca: {test_acc:.2%}',
        ]
        if gate_mean is not None:
            log_parts.append(
                f'gate_mean: {gate_mean:.3f}')
            log_parts.append(
                f'n_active: {gate_n_active}/{train_ds["spikes"].shape[-1]}')
        logging.info(', '.join(log_parts))

        # ── accumulate metrics ───────────────────────────────────────
        train_losses.append(train_loss.item())
        train_nlls.append(train_nll.item())
        train_klds.append(train_kld.item())
        train_centroid_dists.append(float(train_cd))
        train_cls_accs.append(float(train_acc))

        val_losses.append(val_loss.item())
        val_nlls.append(val_nll.item())
        val_klds.append(val_kld.item())
        val_centroid_dists.append(float(val_cd))
        val_cls_accs.append(float(val_acc))

        test_losses.append(test_loss.item())
        test_nlls.append(test_nll.item())
        test_klds.append(test_kld.item())
        test_centroid_dists.append(float(test_cd))
        test_cls_accs.append(float(test_acc))

        losses = {
            'train_losses':         train_losses,
            'train_nlls':           train_nlls,
            'train_klds':           train_klds,
            'train_centroid_dists': train_centroid_dists,
            'train_cls_accs':       train_cls_accs,
            'val_losses':           val_losses,
            'val_nlls':             val_nlls,
            'val_klds':             val_klds,
            'val_centroid_dists':   val_centroid_dists,
            'val_cls_accs':         val_cls_accs,
            'test_losses':          test_losses,
            'test_nlls':            test_nlls,
            'test_klds':            test_klds,
            'test_centroid_dists':  test_centroid_dists,
            'test_cls_accs':        test_cls_accs,
        }

        ckpt = {
            'model': best_state,
            'config': config.to_dict(),
            'losses': losses,
            'perms': perms,
        }

        # ── checkpointing ────────────────────────────────────────────
        if ckpt_save and ((epoch % 100 == 0) or (epoch == config.num_epochs)):
            save_args = orbax_utils.save_args_from_target(ckpt)
            try:
                ckpt_mgr.save(
                    epoch, ckpt, save_kwargs={'save_args': save_args})
            except Exception as e:
                logging.warning(f"Error saving to {workdir}: {e}")
                attempt, saved = 1, False
                while attempt <= 5 and not saved:
                    alt_workdir = f"{workdir}_{attempt}"
                    try:
                        alt_mgr = ocp.CheckpointManager(
                            alt_workdir,
                            ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
                            ocp.CheckpointManagerOptions(
                                create=True, max_to_keep=1),
                        )
                        alt_mgr.save(
                            epoch, ckpt,
                            save_kwargs={'save_args': save_args})
                        logging.info(
                            f"Checkpoint saved to {alt_workdir} "
                            f"at epoch {epoch}")
                        saved = True
                    except Exception as e2:
                        logging.warning(
                            f"Failed to save to {alt_workdir}: {e2}")
                        attempt += 1
                if not saved:
                    logging.error(
                        f"ERROR: Failed to save after {attempt-1} attempts")

    return min(ckpt['losses']['val_losses'][annealing_epochs:])


# ═════════════════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════════════════

def get_datasets(
    datapath, workdir, randseedpath=None,
    k_cv=1, n_splits=5, baseline_fit=False,
):
    if randseedpath:
        df = pd.read_csv(randseedpath)
        one_hot = np.array([
            True if sid in datapath else False
            for sid in df['session_id'].values.astype(str)
        ])
        random_seed = (
            df['random_state'].values.astype(int)[one_hot][0]
            if np.any(one_hot) else 17
        )
    else:
        random_seed = 17

    dt = BIN_WIDTH
    data = np.load(datapath)

    try:
        times = data['times']
    except KeyError:
        times = np.arange(0, data['spikes'].shape[0])

    if 'language_labels' in data:
        language_labels = data['language_labels'].astype(int)
    else:
        language_labels = data['externalinputs'][
            np.arange(len(data['lengths'])),
            data['lengths'] // 2, 0,
        ].astype(int)

    event_samples = data['event_samples'].astype(int)

    kf = StratifiedKFold(
        n_splits=n_splits, random_state=random_seed, shuffle=True)
    train_valid_indices, test_indices = [], []
    np.random.seed(seed=random_seed)
    for i, (tv_idx, te_idx) in enumerate(
        kf.split(data['spikes'], language_labels)
    ):
        train_valid_indices.append(np.random.permutation(tv_idx))
        test_indices.append(np.random.permutation(te_idx))

    train_indices, valid_indices = [], []
    for i in range(n_splits - 1):
        mask = np.isin(train_valid_indices[i], test_indices[i + 1])
        train_indices.append(train_valid_indices[i][~mask])
        valid_indices.append(train_valid_indices[i][mask])
    mask = np.isin(train_valid_indices[-1], test_indices[0])
    train_indices.append(train_valid_indices[-1][~mask])
    valid_indices.append(train_valid_indices[-1][mask])

    baselinepath = workdir.rsplit('/', 1)[0]
    if os.path.exists(baselinepath + '/baseline.npy'):
        baselines = np.load(baselinepath + '/baseline.npy')
        baseline = baselines[:, :, :, k_cv - 1]
    else:
        os.makedirs(baselinepath, exist_ok=True)
        baseline = np.zeros((
            data['spikes'].shape[0],
            data['spikes'].shape[1],
            data['spikes'].shape[2],
        ))
        np.save(
            baselinepath + '/baseline.npy',
            np.stack([baseline] * n_splits, axis=3),
        )

    def _make_ds(indices):
        return {
            'spikes':          data['spikes'][indices, :, :],
            'externalinputs':  data['externalinputs'][indices, :, :],
            'lengths':         data['lengths'][indices],
            'baselineinputs':  baseline[indices, :, :],
            'language_labels': language_labels[indices],
            'event_samples':   event_samples[indices, :],
        }

    train_ds = _make_ds(train_indices[k_cv - 1])
    val_ds   = _make_ds(valid_indices[k_cv - 1])
    test_ds  = _make_ds(test_indices[k_cv - 1])

    ds = {
        k: jnp.concatenate([train_ds[k], val_ds[k], test_ds[k]], 0)
        for k in train_ds
    }
    concat_ds = jnp.concatenate(
        [train_indices[k_cv - 1],
         valid_indices[k_cv - 1],
         test_indices[k_cv - 1]], 0,
    )
    return train_ds, val_ds, test_ds, ds, concat_ds


# ═════════════════════════════════════════════════════════════════════════
#  Loss utilities
# ═════════════════════════════════════════════════════════════════════════

def beta(beta_inc_rate: float, counter: int) -> float:
    return 1 - beta_inc_rate ** counter


def l2_loss(x, alpha):
    return alpha * (x ** 2).mean()


def kl_divergence(
    alpha: float, mu_theta: Array, mu_phi: Array, std: Array,
    lengths: Array, loss_weights: Array,
    lossw_inc_rate: float, counter: int,
) -> float:
    cov = std ** 2
    m = jnp.square(mu_theta - mu_phi) / cov
    kld = jnp.sum(m, axis=-1)
    kld_masked = (
        0.5 * alpha
        * jnp.sum(utils.mask_sequences(kld, lengths))
        / jnp.sum(utils.mask_sequences(kld, lengths) > 0)
    )
    return kld_masked


def neg_gaussian_log_likelihood(
    means: Array, obs_var: Array, observations: Array,
    lengths: Array, loss_weights: Array,
    lossw_inc_rate: float, counter: int,
) -> float:
    gaussian_nll = (
        0.5 * jnp.log(2 * jnp.pi * obs_var)
        + 0.5 * ((observations - means) ** 2) / obs_var
    )
    nll = jnp.sum(gaussian_nll, axis=-1)
    weights = beta(lossw_inc_rate, counter) * loss_weights + 1.0
    weights = weights / jnp.sum(weights) * 100.0
    masked_nll = (
        jnp.sum(utils.mask_sequences(nll, lengths) * weights)
        / jnp.sum(utils.mask_sequences(jnp.ones_like(nll), lengths))
    )
    return masked_nll


def create_learning_rate_fn(config, steps_per_epoch):
    """Creates SGDR learning rate schedule."""
    num_iter = config.num_epochs // (
        config.cosine_epochs + config.warmup_epochs)
    cosine_kwargs = [
        {
            'init_value': 0.0,
            'peak_value': config.base_learning_rate,
            'warmup_steps': config.warmup_epochs * steps_per_epoch,
            'decay_steps': (
                (config.cosine_mult_by ** i)
                * config.cosine_epochs * steps_per_epoch
            ),
            'end_value': 0.0,
        }
        for i in range(num_iter)
    ]
    return optax.sgdr_schedule(cosine_kwargs)
