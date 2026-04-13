"""
Entry point for FINDR_Supervised with fixed temporal windows and
centroid distance loss.

Usage:

  python main_supervised.py \\
      --datapath=kh092_shaft.npz \\
      --workdir=checkpoints/centroid_fixed \\
      --learning_rate=0.005 \\
      --feature_size=50 --net_size=100 --latent_size=8 \\
      --lambda_class=10 \\
      --overlap_frac=0.5 \\
      --frac_hi=1.2
"""

from absl import app, flags, logging
import ml_collections
import numpy as np
import optuna
import train_supervised as train

FLAGS = flags.FLAGS

flags.DEFINE_string('datapath', None, 'Path to .npz data file.')
flags.DEFINE_string('workdir', None, 'Directory for checkpoints.')
flags.DEFINE_float('learning_rate', None,
                   'Learning rate (None -> run architecture search).')
flags.DEFINE_integer('feature_size', None, 'gnSDE feature size.')
flags.DEFINE_integer('net_size', None, 'Inference network hidden size.')
flags.DEFINE_integer('latent_size', None, 'Task-related latent size (= number of temporal windows).')
flags.DEFINE_float('lambda_sparse', 1e-4, 'L1 gate sparsity weight.')
flags.DEFINE_float('lambda_class', 1.0, 'Centroid distance loss weight.')
flags.DEFINE_float('overlap_frac', 0.5, 'Overlap fraction between adjacent temporal windows (0-1).')
flags.DEFINE_float('frac_hi', 1.2, 'Upper bound of temporal coverage in fractional time.')
flags.DEFINE_boolean('search_sparse', False,
                     'Run Optuna search for lambda_sparse.')


def get_config(
    k_cv, num_epochs, learning_rate, feature_size, net_size,
    latent_size, lambda_sparse=1e-4, lambda_class=10.0,
    overlap_frac=0.5, frac_hi=1.2,
):
    config = ml_collections.ConfigDict()
    config.k_cv = k_cv
    config.base_learning_rate = learning_rate
    config.features_prior = [feature_size]
    config.features_posterior = [feature_size]
    config.inference_network_size = net_size
    config.beta = 2.0
    config.noise_level = 1.0
    config.cosine_epochs = 190
    config.alpha = 0.039
    config.task_related_latent_size = latent_size
    config.n_splits = 5
    config.l2_coeff = 1e-4
    config.lambda_sparse = lambda_sparse
    config.lambda_class = lambda_class
    config.overlap_frac = overlap_frac
    config.frac_hi = frac_hi
    config.cosine_mult_by = 2
    config.warmup_epochs = 10
    config.batch_size = 5
    config.num_epochs = num_epochs
    config.momentum = 0.9
    config.beta_inc_rate = 0.99
    config.lossw_inc_rate = 1.0
    config.earlymiddle_epochs = 0
    config.baseline_fit = False
    config.constrain_prior = False
    config.use_gaussian = True
    config.use_channel_gating = True
    return config


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    run_architecture_search = FLAGS.learning_rate is None

    if run_architecture_search:
        def objective(trial):
            lr = trial.suggest_float('learning_rate', 0.001, 0.01, log=True)
            fs = trial.suggest_int('feature_size', 30, 200, step=25)
            ns = trial.suggest_int('net_size', 50, 400, step=50)
            ls = trial.suggest_int('latent_size', 12, 32, step=4)
            config = get_config(
                k_cv=1, num_epochs=1500,
                learning_rate=lr, feature_size=fs,
                net_size=ns, latent_size=ls,
                lambda_sparse=1e-4, lambda_class=FLAGS.lambda_class,
                overlap_frac=FLAGS.overlap_frac, frac_hi=FLAGS.frac_hi,
            )
            config.use_channel_gating = False
            return train.train_and_evaluate(
                config, FLAGS.datapath, FLAGS.workdir, ckpt_save=False)

        study = optuna.create_study()
        study.optimize(objective, n_trials=30)

        config = get_config(
            k_cv=1, num_epochs=3000,
            learning_rate=study.best_params['learning_rate'],
            feature_size=study.best_params['feature_size'],
            net_size=study.best_params['net_size'],
            latent_size=study.best_params['latent_size'],
            lambda_sparse=1e-4, lambda_class=FLAGS.lambda_class,
            overlap_frac=FLAGS.overlap_frac, frac_hi=FLAGS.frac_hi,
        )
        config.use_channel_gating = False
        train.train_and_evaluate(
            config, FLAGS.datapath, FLAGS.workdir, ckpt_save=True)
        logging.info('Best architecture params: %s', study.best_params)

    elif FLAGS.search_sparse:
        def objective(trial):
            ls = trial.suggest_float('lambda_sparse', 1e-4, 0.1, log=True)
            config = get_config(
                k_cv=1, num_epochs=1500,
                learning_rate=FLAGS.learning_rate,
                feature_size=FLAGS.feature_size,
                net_size=FLAGS.net_size,
                latent_size=FLAGS.latent_size,
                lambda_sparse=ls, lambda_class=FLAGS.lambda_class,
                overlap_frac=FLAGS.overlap_frac, frac_hi=FLAGS.frac_hi,
            )
            config.use_channel_gating = True
            return train.train_and_evaluate(
                config, FLAGS.datapath, FLAGS.workdir, ckpt_save=False)

        study = optuna.create_study()
        study.optimize(objective, n_trials=30)

        config = get_config(
            k_cv=1, num_epochs=3000,
            learning_rate=FLAGS.learning_rate,
            feature_size=FLAGS.feature_size,
            net_size=FLAGS.net_size,
            latent_size=FLAGS.latent_size,
            lambda_sparse=study.best_params['lambda_sparse'],
            lambda_class=FLAGS.lambda_class,
            overlap_frac=FLAGS.overlap_frac, frac_hi=FLAGS.frac_hi,
        )
        train.train_and_evaluate(
            config, FLAGS.datapath, FLAGS.workdir, ckpt_save=True)
        logging.info('Best lambda_sparse: %s', study.best_params)

    else:
        config = get_config(
            k_cv=1, num_epochs=3000,
            learning_rate=FLAGS.learning_rate,
            feature_size=FLAGS.feature_size,
            net_size=FLAGS.net_size,
            latent_size=FLAGS.latent_size,
            lambda_sparse=FLAGS.lambda_sparse,
            lambda_class=FLAGS.lambda_class,
            overlap_frac=FLAGS.overlap_frac,
            frac_hi=FLAGS.frac_hi,
        )
        train.train_and_evaluate(
            config, FLAGS.datapath, FLAGS.workdir, ckpt_save=True)


if __name__ == '__main__':
    flags.mark_flags_as_required(['datapath', 'workdir'])
    app.run(main)
