## Modified to take in sEEG data (swapped Poisson observation model for Gaussian likelihood)

import functools

from absl import app
from absl import flags
from absl import logging

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
import models
import findr_utils as utils
import os

BIN_WIDTH = 0.01 # in seconds
SMALL_CONSTANT = 1e-5
Array = Any
FLAGS = flags.FLAGS
PRNGKey = Any

def create_train_state(
  rng: PRNGKey, 
  config: ml_collections.ConfigDict, 
  learning_rate_fn,
  xs,
  ckptdir = None
):  
  """Creates an initial `TrainState`."""
  key_1, key_2, key_3 = random.split(rng, 3)
  model = models.FINDR(
    alpha = config.alpha,
    noise_level = config.noise_level,
    features_prior = config.features_prior,
    features_posterior = config.features_posterior,
    task_related_latent_size = config.task_related_latent_size,
    inference_network_size = config.inference_network_size,
    num_neurons = xs['spikes'].shape[-1],
    constrain_prior = config.constrain_prior,
    use_gaussian = config.use_gaussian,             # added for sEEG data
    use_channel_gating = config.use_channel_gating  # added for sEEG data
  )
  if ckptdir is not None:
    raw_restored = checkpoints.restore_checkpoint(
      ckpt_dir=ckptdir, 
      target=None, 
      parallel=False
    )
    params = freeze(raw_restored['model']['params'])
  else:
    params = model.init(
      key_2,
      xs['spikes'], 
      xs['externalinputs'],
      xs['baselineinputs'],
      xs['lengths'], 
      key_3
    )['params']
  
  tx = optax.chain(
    optax.clip_by_global_norm(5.0),
    optax.sgd(learning_rate_fn, config.momentum)
  )
  state = train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=tx
  )
  return state

@functools.partial(jax.jit, static_argnums=[3, 4, 5, 6, 7, 10, 11])
def apply_model(
  state: train_state.TrainState, 
  batch, 
  loss_weights: Array,
  alpha: float, 
  beta_coeff: float,
  beta_inc_rate: float,
  lossw_inc_rate: float,
  l2_coeff: float,
  beta_counter: int,
  lossw_counter:int, 
  learning_rate_fn,
  lambda_sparse: float,
  rng: PRNGKey
):
  ntrgru_mask = traverse_util.path_aware_map(
    lambda path, _: 1 if 'non_task_related_gru' in path
    else 2, state.params
  )

  ortho_mask = traverse_util.path_aware_map(
    lambda path, _: 1 if 'task_related_latents_to_neurons' in path
    else 2, state.params
  )

  """Computes gradients and loss for a single batch."""
  def loss_fn(params): 
    outputs, obs_var, z, mu, mu_theta, mu_phi, std, gates = state.apply_fn(
      {'params': params},
      batch['spikes'], # actually high-gamma power for Gaussian model
      batch['externalinputs'], 
      batch['baselineinputs'],
      batch['lengths'], 
      rng
    )
    nll_loss = neg_gaussian_log_likelihood(
      outputs,
      obs_var,
      batch['spikes'],
      batch['lengths'],
      loss_weights,
      lossw_inc_rate,
      lossw_counter
    )
    kld_loss = beta_coeff * beta(beta_inc_rate, beta_counter) * kl_divergence(
      alpha, 
      mu_theta, 
      mu_phi, 
      std,
      batch['lengths'],
      loss_weights,
      lossw_inc_rate,
      lossw_counter
    )
    loss = nll_loss + kld_loss

    # add L1 sparsity penalty on gates
    if gates is not None:
      sparse_loss = lambda_sparse * jnp.sum(jnp.abs(gates))
      loss = loss + sparse_loss
    else:
      sparse_loss = 0.0

    # existing L2 regularization
    loss += sum(
      l2_loss(w, alpha=l2_coeff) if label==1 
      else l2_loss(w, alpha=1e-7)
      for label, w in zip(jax.tree.leaves(ntrgru_mask), jax.tree.leaves(params))
    )
    return loss, (nll_loss, kld_loss, sparse_loss, outputs, gates)
  
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (nll, kld, sparse, outputs, gates)), grads = grad_fn(state.params)

  return grads, loss, nll, kld, sparse, gates

@jax.jit
def update_model(
  state: train_state.TrainState, 
  grads
):
  return state.apply_gradients(grads=grads)

def train_epoch(
  state: train_state.TrainState,
  train_ds, 
  loss_weights: Array,
  config: ml_collections.ConfigDict, 
  beta_counter: int,
  lossw_counter: int,
  learning_rate_fn,
  rng: PRNGKey
):
  """Train for a single epoch."""
  key_1, key_2 = random.split(rng, 2)
  train_ds_size = len(train_ds['externalinputs'])
  steps_per_epoch = train_ds_size // config.batch_size

  perms = random.permutation(key_1, len(train_ds['externalinputs']))
  perms = perms[:steps_per_epoch * config.batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, config.batch_size))

  epoch_loss = []
  epoch_nll = []
  epoch_kld = []
  epoch_gate_stats = []
  
  for perm in perms:
    key_2, key_3 = random.split(key_2, 2)
    batch_spikes = train_ds['spikes'][perm, ...]
    batch_inputs = train_ds['externalinputs'][perm, ...]
    batch_lengths = train_ds['lengths'][perm, ...]
    batch_baseinputs = train_ds['baselineinputs'][perm, ...]
    batch = {
    'spikes': batch_spikes,
    'externalinputs': batch_inputs,
    'lengths': batch_lengths,
    'baselineinputs': batch_baseinputs
    }
    grads, loss, nll, kld, sparse, gates = apply_model(
      state, 
      batch, 
      loss_weights,
      config.alpha,
      config.beta,
      config.beta_inc_rate, 
      config.lossw_inc_rate,
      config.l2_coeff,
      beta_counter,
      lossw_counter, 
      learning_rate_fn,
      config.lambda_sparse,
      key_3
    )
    
    if gates is not None:
      epoch_gate_stats.append({
        'mean': jnp.mean(gates),
        'std': jnp.std(gates),
        'n_active': jnp.sum(gates > 0.5),
        'max': jnp.max(gates),
        'min': jnp.min(gates)
      })
    
    state = update_model(state, grads)
    epoch_loss.append(loss)
    epoch_nll.append(nll)
    epoch_kld.append(kld)
  
  train_loss = np.mean(epoch_loss)
  train_nll = np.mean(epoch_nll)
  train_kld = np.mean(epoch_kld)

  if epoch_gate_stats:
    gate_mean = np.mean([s['mean'] for s in epoch_gate_stats])
    gate_n_active = int(np.mean([s['n_active'] for s in epoch_gate_stats]))
  else:
    gate_mean = None
    gate_n_active = None
  
  return state, train_loss, train_nll, train_kld, gate_mean, gate_n_active

def train_and_evaluate(
  config: ml_collections.ConfigDict,
  datapath: str,
  workdir: str,
  randseedpath: str = None,
  ckpt_save: bool = True
) -> train_state.TrainState:
  """Execute model training and evaluation loop.
  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the checkpoints are saved in.
  Returns:
    The train state (which includes the `.params`).
  """
  print(f"Starting train_and_evaluate with datapath: {datapath}")

  print(f"Loading data...")
  train_ds, val_ds, test_ds, ds, perms = get_datasets(
    datapath, 
    workdir,
    randseedpath,
    k_cv=config.k_cv, 
    n_splits=config.n_splits,
    baseline_fit=config.baseline_fit
  )
  print(f"Data loaded. Train size: {len(train_ds['spikes'])}")

  print(f"Initializing model...")
  rng = random.PRNGKey(1) # this is the random seed that we use to initialize the model
  key_1, key_2 = random.split(rng, 2)
  train_ds_size = len(train_ds['externalinputs'])
  steps_per_epoch = train_ds_size // config.batch_size
  print('steps_per_epoch ', steps_per_epoch)
  learning_rate_fn = create_learning_rate_fn(config, steps_per_epoch)
  state = create_train_state(key_1, config, learning_rate_fn, test_ds)
  best_state = state

  # initialize checkpointing
  mgr_options = ocp.CheckpointManagerOptions(
    create=True, max_to_keep=1)
  ckpt_mgr = ocp.CheckpointManager(
    workdir,
    ocp.Checkpointer(ocp.PyTreeCheckpointHandler()), 
    mgr_options
  )
  
  # the epoch around which the coefficient to the KL divergence term reaches 0.99
  annealing_epochs = np.floor(np.log(0.01)/np.log(config.beta_inc_rate)).astype(int)

  # train only the first 0.3s
  early_loss_weights = -1. * jnp.ones_like(train_ds['spikes'][0,:,0])
  early_loss_weights = early_loss_weights.at[:30].set(0.)
  
  # train only the first 0.5s
  middle_loss_weights = -1. * jnp.ones_like(train_ds['spikes'][0,:,0])
  middle_loss_weights = middle_loss_weights.at[:50].set(0.)

  # train all data points
  late_loss_weights = jnp.zeros_like(train_ds['spikes'][0,:,0])

  train_losses = []
  train_nlls = []
  train_klds = []

  val_losses = []
  val_nlls = []
  val_klds = []

  test_losses = []
  test_nlls = []
  test_klds = []

  for epoch in range(1, config.num_epochs + 1):
    key_2, key_3, key_4, key_5 = random.split(key_2, 4)
    if epoch < config.earlymiddle_epochs/3:
      state, train_loss, train_nll, train_kld, gate_mean, gate_n_active = train_epoch(
        state, 
        train_ds, 
        early_loss_weights,
        config,
        0,
        1e6,
        learning_rate_fn,
        key_3
      )
    elif epoch < config.earlymiddle_epochs:
      state, train_loss, train_nll, train_kld, gate_mean, gate_n_active = train_epoch(
        state, 
        train_ds, 
        middle_loss_weights,
        config, 
        0,
        1e6,
        learning_rate_fn,
        key_3
      )
    else:
      state, train_loss, train_nll, train_kld, gate_mean, gate_n_active = train_epoch(
        state, 
        train_ds, 
        late_loss_weights,
        config, 
        epoch - config.earlymiddle_epochs,
        epoch - config.earlymiddle_epochs,
        learning_rate_fn,
        key_3
      )

    _, val_loss, val_nll, val_kld, _, _ = apply_model(
      state, 
      val_ds,
      late_loss_weights,
      config.alpha,
      config.beta,
      config.beta_inc_rate,
      config.lossw_inc_rate,
      config.l2_coeff,  
      1e6,
      0,
      learning_rate_fn,
      config.lambda_sparse,
      key_4
    )

    _, test_loss, test_nll, test_kld, _, _ = apply_model(
      state, 
      test_ds,
      late_loss_weights,
      config.alpha,
      config.beta,
      config.beta_inc_rate,
      config.lossw_inc_rate,
      config.l2_coeff,  
      1e6, 
      0,
      learning_rate_fn,
      config.lambda_sparse,
      key_5
    )
    
    if epoch > (config.earlymiddle_epochs + annealing_epochs):
      if val_losses[(config.earlymiddle_epochs + annealing_epochs):]:
        if val_loss < min(val_losses[(config.earlymiddle_epochs + annealing_epochs):]):
          best_state = state
            
    if gate_mean is not None:
      logging.info(
        '%s: %d, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.3f, %s: %d/%d' % (
          'epoch', epoch,
          'train_loss', train_loss,
          'train_nll',  train_nll,
          'train_kld',  train_kld,
          'val_loss', val_loss, 
          'val_nll', val_nll, 
          'val_kld', val_kld,
          'test_loss', test_loss,
          'test_nll', test_nll, 
          'test_kld', test_kld,
          'gate_mean', gate_mean,
          'n_active', gate_n_active, train_ds['spikes'].shape[-1]
        )
      )
    else:
      logging.info(
        '%s: %d, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f, %s: %.4f' % (
          'epoch', epoch,
          'train_loss', train_loss,
          'train_nll',  train_nll,
          'train_kld',  train_kld,
          'val_loss', val_loss, 
          'val_nll', val_nll, 
          'val_kld', val_kld,
          'test_loss', test_loss,
          'test_nll', test_nll, 
          'test_kld', test_kld
        )
      )

    train_losses.append(train_loss.item())
    train_nlls.append(train_nll.item())
    train_klds.append(train_kld.item())
    val_losses.append(val_loss.item())
    val_nlls.append(val_nll.item())
    val_klds.append(val_kld.item())
    test_losses.append(test_loss.item())
    test_nlls.append(test_nll.item())
    test_klds.append(test_kld.item())

    losses = {
      'train_losses': train_losses, 
      'train_nlls': train_nlls, 
      'train_klds': train_klds, 
      'val_losses': val_losses, 
      'val_nlls': val_nlls, 
      'val_klds': val_klds,
      'test_losses': test_losses, 
      'test_nlls': test_nlls, 
      'test_klds': test_klds
    }

    ckpt = {
      'model': best_state, 
      'config': config.to_dict(), 
      'losses': losses, 
      'perms': perms
    }

    if ckpt_save:
      if (epoch % 100 == 0) or (epoch == config.num_epochs):
        save_args = orbax_utils.save_args_from_target(ckpt)
        try:
          ckpt_mgr.save(epoch, ckpt, save_kwargs={'save_args': save_args})
        except Exception as e:
          logging.warning(f"Error saving to {workdir}: {e}")
          attempt = 1
          saved = False
          while attempt <= 5 and not saved:
            alt_workdir = f"{workdir}_{attempt}"
            try:
              # create new checkpoint manager for alternate directory
              alt_mgr_options = ocp.CheckpointManagerOptions(create=True, max_to_keep=1)
              alt_ckpt_mgr = ocp.CheckpointManager(
                alt_workdir,
                ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
                alt_mgr_options
              )
              alt_ckpt_mgr.save(epoch, ckpt, save_kwargs={'save_args': save_args})
              logging.info(f"Checkpoint saved to {alt_workdir} at epoch {epoch}")
              saved = True
            except Exception as e2:
              logging.warning(f"Failed to save to {alt_workdir}: {e2}")
              attempt += 1
      
          if not saved:
            logging.error(f"ERROR: Failed to save checkpoint after {attempt-1} attempts")

  return min(ckpt['losses']['val_losses'][(config.earlymiddle_epochs + annealing_epochs):])

def get_datasets(datapath, workdir, randseedpath=None, k_cv=1, n_splits=5, baseline_fit=False):
  if randseedpath:
    df = pd.read_csv(randseedpath)
    one_hot = np.array(
      [True if session_id in datapath else False for session_id in df['session_id'].values.astype(str)]
    )
    if np.any(one_hot):
      random_seed = df['random_state'].values.astype(int)[one_hot][0]
    else:
      random_seed = 17
  else:
    random_seed = 17
  dt = BIN_WIDTH
  data = np.load(datapath)
  
  # we need to check if the data has keyword 'times'
  try:
    times = data['times']
  except(KeyError):
    times = np.arange(0, data['spikes'].shape[0])

  # language_labels = data['externalinputs'][:, 0, 0].astype(int)
  language_labels = (data['language_labels'].astype(int) if 'language_labels' in data 
                     else data['externalinputs'][np.arange(len(data['lengths'])), data['lengths'] // 2, 0].astype(int))
  kf = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=True)
  train_valid_indices = []
  test_indices = []
  np.random.seed(seed=random_seed)
  for i, (train_valid_index, test_index) in enumerate(kf.split(data['spikes'], language_labels)):
    train_valid_indices.append(np.random.permutation(train_valid_index))
    test_indices.append(np.random.permutation(test_index))

  train_indices = []
  valid_indices = []
  for i in range(n_splits-1):
    train_indices.append(train_valid_indices[i][~np.isin(train_valid_indices[i], test_indices[i+1])])
    valid_indices.append(train_valid_indices[i][np.isin(train_valid_indices[i], test_indices[i+1])])
  train_indices.append(train_valid_indices[-1][~np.isin(train_valid_indices[-1], test_indices[0])])
  valid_indices.append(train_valid_indices[-1][np.isin(train_valid_indices[-1], test_indices[0])])

  baselinepath = workdir.rsplit('/', 1)[0]

  if os.path.exists(baselinepath + '/baseline.npy'):
    baselines = np.load(baselinepath + '/baseline.npy')
    baseline = baselines[:,:,:,k_cv-1]
  else:
    os.makedirs(baselinepath, exist_ok=True)
    spikes_across_trials1 = np.zeros((data['spikes'].shape[0], 1, data['spikes'].shape[2]))
    for trial in range(data['spikes'].shape[0]):
      spikes_across_trials1[trial, 0, :] = np.mean(
        data['spikes'][trial, :data['lengths'][trial], :], 
        axis=0
      )/dt

    try:
      baseline_hz = data['baseline_hz']
      spikes_across_trials = np.reshape(
        baseline_hz, 
        (baseline_hz.shape[0], 1, baseline_hz.shape[1])
      )
      # alpha = np.mean(spikes_across_trials1, axis=0)[0, :] / np.mean(spikes_across_trials, axis=0)[0, :]
      # spikes_across_trials = spikes_across_trials * alpha
    except(KeyError):
      spikes_across_trials = spikes_across_trials1
    
    if baseline_fit:
      baselines_across_trials = utils.infer_baseline_across_trials(
        times,
        spikes_across_trials, 
        train_indices, 
        valid_indices,
        n_splits
      )

      baselines = utils.infer_baseline(
        data['spikes'],
        data['lengths'],
        baselines_across_trials,
        train_indices, 
        valid_indices,
        n_splits
      )
      baseline = baselines[:,:,:,k_cv-1]
      np.save(baselinepath + '/baseline.npy', baselines)
    else:
      # no baselinefor Gaussian sEEG data with learned bias
      baseline = np.zeros((
        data['spikes'].shape[0],
        data['spikes'].shape[1],
        data['spikes'].shape[2]
      ))
      np.save(baselinepath + '/baseline.npy', np.stack([baseline]*n_splits, axis=3))

  train_ds = {
    'spikes': data['spikes'][train_indices[k_cv-1],:,:],
    'externalinputs': data['externalinputs'][train_indices[k_cv-1],:,:], 
    'lengths':data['lengths'][train_indices[k_cv-1]], 
    'baselineinputs': baseline[train_indices[k_cv-1],:,:]
  }
  
  val_ds = {
    'spikes': data['spikes'][valid_indices[k_cv-1],:,:],
    'externalinputs': data['externalinputs'][valid_indices[k_cv-1],:,:], 
    'lengths':data['lengths'][valid_indices[k_cv-1]], 
    'baselineinputs': baseline[valid_indices[k_cv-1],:,:]
  }
  
  test_ds = {
    'spikes': data['spikes'][test_indices[k_cv-1],:,:],
    'externalinputs': data['externalinputs'][test_indices[k_cv-1],:,:], 
    'lengths':data['lengths'][test_indices[k_cv-1]], 
    'baselineinputs': baseline[test_indices[k_cv-1],:,:]
  }
  
  ds = {
    'spikes': jnp.concatenate(
        [
          train_ds['spikes'], 
          val_ds['spikes'], 
          test_ds['spikes']
        ], 0
      ),
    'externalinputs': jnp.concatenate(
        [
          train_ds['externalinputs'], 
          val_ds['externalinputs'], 
          test_ds['externalinputs']
        ], 0
      ),
    'lengths': jnp.concatenate(
        [
          train_ds['lengths'], 
          val_ds['lengths'], 
          test_ds['lengths']
        ], 0
      ),
    'baselineinputs': jnp.concatenate(
        [
          train_ds['baselineinputs'], 
          val_ds['baselineinputs'], 
          test_ds['baselineinputs']
        ], 0
      )
  }

  concat_ds = jnp.concatenate(
    [
      train_indices[k_cv-1], 
      valid_indices[k_cv-1], 
      test_indices[k_cv-1]
    ], 0
  )

  return train_ds, val_ds, test_ds, ds, concat_ds

def beta(beta_inc_rate:float, counter: int) -> float:
  return 1 - beta_inc_rate**counter

def l2_loss(x, alpha):
    return alpha * (x ** 2).mean()

def kl_divergence(
  alpha: float, 
  mu_theta: Array, 
  mu_phi: Array, 
  std: Array, 
  lengths: Array,
  loss_weights: Array,
  lossw_inc_rate: float,
  counter: int
) -> float:
  cov = std ** 2
  m = jnp.square(mu_theta - mu_phi) / cov
  kld = jnp.sum(m, axis=-1)
  kld_masked = 0.5 * alpha * jnp.sum(
    utils.mask_sequences(kld, lengths)
    ) / jnp.sum(
    utils.mask_sequences(kld, lengths) > 0
    )
  return kld_masked

def neg_poisson_log_likelihood(
  outputs: Array, 
  spikes: Array, 
  lengths: Array,
  loss_weights: Array,
  lossw_inc_rate: float,
  counter: int
) -> float:
  """Calculates Poisson negative log likelihood given rates and spikes.
  formula: -log(e^(-r) / n! * r^n)
          = r - n*log(r) + log(n!)
  """
  dt = BIN_WIDTH
  rates = softplus(outputs) + SMALL_CONSTANT
  result = dt*rates - spikes * jnp.log(dt*rates) + gammaln(spikes + 1.0)
  nll = jnp.sum(result, axis=-1)
  weights = beta(lossw_inc_rate, counter) * loss_weights + 1.
  weights = weights / jnp.sum(weights) * 100.
  masked_nll = jnp.sum(
    utils.mask_sequences(nll, lengths) * weights
  ) / jnp.sum((utils.mask_sequences(nll, lengths) > 0))
  return masked_nll

def neg_gaussian_log_likelihood(
    means: Array,
    obs_var: Array,
    observations: Array,
    lengths: Array,
    loss_weights: Array,
    lossw_inc_rate: float,
    counter: int
) -> float:
    """Calculates Gaussian negative log likelihood.
    formula: 0.5 * log(2π*σ²) + 0.5 * (x - μ)² / σ²
    """
    gaussian_nll = (
        0.5 * jnp.log(2 * jnp.pi * obs_var) +
        0.5 * ((observations - means) ** 2) / obs_var
    )
    nll = jnp.sum(gaussian_nll, axis=-1)
    
    # apply temporal weights and masking
    weights = beta(lossw_inc_rate, counter) * loss_weights + 1.
    weights = weights / jnp.sum(weights) * 100.
    masked_nll = jnp.sum(
        utils.mask_sequences(nll, lengths) * weights
    ) / jnp.sum(utils.mask_sequences(jnp.ones_like(nll), lengths))
    
    return masked_nll

def create_learning_rate_fn(config, steps_per_epoch):
  """Creates learning rate schedule."""
  num_iter = config.num_epochs // (config.cosine_epochs + config.warmup_epochs)
  cosine_kwargs = [{
    'init_value': 0.,
    'peak_value': config.base_learning_rate,
    'warmup_steps': config.warmup_epochs * steps_per_epoch,
    'decay_steps': (config.cosine_mult_by ** i) * config.cosine_epochs * steps_per_epoch,
    'end_value': 0.
  } for i in range(num_iter)]
  
  schedule_fn = optax.sgdr_schedule(cosine_kwargs)
  return schedule_fn
