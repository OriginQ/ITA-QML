"""VQC training CLI — driven by command-line arguments and JSON configuration.

Usage::

    python train.py                          # Use default configs/train_config.json
    python train.py --config configs/exp1.json
    python train.py --backend classical      # Switch backend
    python train.py --target-col Corr_CCSD   # Switch regression target
    python train.py --epochs 200 --lr 0.0005 # Override hyperparameters via CLI
    python train.py --data-path data/ben.csv --n-splits 5

To extend with a new model or dataset:
    1. Implement the model class and register it in
       ``models/__init__.py`` ``MODEL_REGISTRY``.
    2. Implement the dataset loader and register it in
       ``data/dataset.py`` ``DATASET_REGISTRY``.
    3. Set the ``backend`` / ``dataset`` keys in the JSON config.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from data.dataset import load_dataset, create_cv_splits
from train.trainer import train_model, evaluate_model
from utils.plot import plot_results


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------
def load_config(config_path):
    """Load a JSON configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(config_path) as f:
        return json.load(f)


def merge_cli_overrides(cfg, args):
    """Merge CLI argument overrides into the configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Base configuration loaded from JSON.
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Configuration with CLI values merged in (in-place).
    """
    overrides = {
        'training': ['epochs', 'lr', 'early_stop', 'patience', 'min_delta',
                     'restore_best_weights', 'use_scheduler', 'scheduler_factor',
                     'scheduler_patience', 'grad_clip_norm'],
        'model': ['hidden_size', 'dropout_rate', 'qubit_num', 'qc_layers'],
        'data': ['batch_size', 'n_splits', 'random_state', 'target_col'],
    }

    for section, keys in overrides.items():
        for key in keys:
            val = getattr(args, key, None)
            if val is not None:
                cfg[section][key] = val

    # Top-level overrides.
    if args.backend:
        cfg['task']['backend'] = args.backend
    if args.data_path:
        cfg['data']['csv_path'] = args.data_path
    if args.model_dir:
        cfg['output']['model_dir'] = args.model_dir
    if args.plot_dir:
        cfg['output']['plot_dir'] = args.plot_dir

    return cfg


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def run_training(cfg):
    """Execute the full training pipeline from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Configuration with ``task``, ``data``, ``model``, ``training``,
        and ``output`` sections.
    """
    proj_root = Path(__file__).resolve().parent

    task_cfg = cfg['task']
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['training']
    out_cfg = cfg['output']

    backend = task_cfg['backend']
    dataset_name = task_cfg['dataset']

    # --- Paths ---
    csv_path = data_cfg.get('csv_path', '')
    if not Path(csv_path).is_absolute():
        csv_path = str(proj_root / csv_path)
    data_cfg = {**data_cfg, 'csv_path': csv_path}

    model_dir = proj_root / out_cfg['model_dir']
    plot_dir = proj_root / out_cfg['plot_dir']
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    # --- Data ---
    features, labels, feature_scaler, label_scaler = load_dataset(
        dataset_name, data_cfg)

    # --- Configuration summary ---
    print(f'Task: backend={backend}, target={data_cfg.get("target_col", "?")}')
    print(f'Data: {csv_path}  (total samples: {len(features)})')
    print(f'Model config: {json.dumps(model_cfg, indent=2)}')
    print(f'Training config: {json.dumps(train_cfg, indent=2)}')

    # --- Cross-validation ---
    all_train_metrics = []
    all_val_metrics = []

    for fold, (tr_x, tr_y, val_x, val_y) in enumerate(
            create_cv_splits(features, labels,
                             n_splits=data_cfg['n_splits'],
                             random_state=data_cfg['random_state'])):
        print(f'\n{"=" * 60}')
        print(f'=== Fold {fold + 1}/{data_cfg["n_splits"]} ===')
        print(f'Train set: {len(tr_x)}, val set: {len(val_x)}')

        model, train_metrics, val_metrics = train_model(
            tr_x, tr_y, val_x, val_y,
            label_scaler=label_scaler,
            model_dir=model_dir,
            plot_dir=plot_dir,
            epochs=train_cfg['epochs'],
            lr=train_cfg['lr'],
            batch_size=data_cfg['batch_size'],
            fold=fold,
            early_stop=train_cfg['early_stop'],
            patience=train_cfg['patience'],
            min_delta=train_cfg['min_delta'],
            restore_best_weights=train_cfg['restore_best_weights'],
            backend=backend,
            model_cfg=model_cfg,
            use_scheduler=train_cfg.get('use_scheduler', True),
            scheduler_factor=train_cfg.get('scheduler_factor', 0.5),
            scheduler_patience=train_cfg.get('scheduler_patience', 20),
            grad_clip_norm=train_cfg.get('grad_clip_norm', 1.0),
        )

        train_pred, train_true = evaluate_model(
            model, tr_x, tr_y, data_cfg['batch_size'], label_scaler, backend=backend)
        val_pred, val_true = evaluate_model(
            model, val_x, val_y, data_cfg['batch_size'], label_scaler, backend=backend)
        plot_results(train_pred, train_true, val_pred, val_true, fold, plot_dir)

        all_train_metrics.append(train_metrics)
        all_val_metrics.append(val_metrics)

    print_cv_summary(all_train_metrics, all_val_metrics)
    print(f'\nModels saved to: {model_dir}')
    print(f'Plots saved to: {plot_dir}')


def print_cv_summary(all_train_metrics, all_val_metrics):
    """Print mean ± std summary across all cross-validation folds.

    Parameters
    ----------
    all_train_metrics : list of tuple
        Per-fold training metrics (RMSE, Pearson r, Spearman rho, R²).
    all_val_metrics : list of tuple
        Per-fold validation metrics.
    """
    def mean_std(values):
        return np.mean(values), np.std(values)

    train_metrics = np.array(all_train_metrics)
    val_metrics = np.array(all_val_metrics)
    names = ['RMSE', 'Pearson', 'Spearman', 'R2']

    print('\n' + '=' * 60)
    print('=== Cross-Validation Summary ===')
    print('\nTrain set:')
    for i, name in enumerate(names):
        m, s = mean_std(train_metrics[:, i])
        print(f'  {name}: {m:.4f} ± {s:.4f}')
    print('\nValidation set:')
    for i, name in enumerate(names):
        m, s = mean_std(val_metrics[:, i])
        print(f'  {name}: {m:.4f} ± {s:.4f}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    """Build the argument parser for the training CLI.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='VQC Variational Quantum Classifier — training entry point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py
  python train.py --config configs/train_config.json
  python train.py --backend classical
  python train.py --target-col Corr_CCSD
  python train.py --epochs 200 --lr 0.0005 --batch-size 64
  python train.py --data-path data/ben.csv --n-splits 5
        """,
    )

    parser.add_argument('--config', type=str, default='configs/train_config.json',
                        help='Path to JSON configuration file')

    # Task
    g_task = parser.add_argument_group('Task')
    g_task.add_argument('--backend', type=str,
                        help='Backend: vqc / classical / classical_torch')

    # Data
    g_data = parser.add_argument_group('Data')
    g_data.add_argument('--data-path', type=str, help='Path to CSV data file')
    g_data.add_argument('--target-col', type=str,
                        help='Target column: Corr_MP2 / Corr_CCSD / Corr_CCSD(T)')
    g_data.add_argument('--batch-size', type=int, help='Batch size')
    g_data.add_argument('--n-splits', type=int, help='Number of CV folds')
    g_data.add_argument('--random-state', type=int, help='Random seed')

    # Model
    g_model = parser.add_argument_group('Model')
    g_model.add_argument('--qubit-num', type=int, help='Number of qubits (VQC only)')
    g_model.add_argument('--hidden-size', type=int, help='Hidden layer dimension')
    g_model.add_argument('--dropout-rate', type=float, help='Dropout rate')
    g_model.add_argument('--qc-layers', type=int, help='Number of VQC layers (VQC only)')

    # Training
    g_train = parser.add_argument_group('Training')
    g_train.add_argument('--epochs', type=int, help='Number of training epochs')
    g_train.add_argument('--lr', type=float, help='Learning rate')
    g_train.add_argument('--early-stop', dest='early_stop', action='store_true',
                         default=None, help='Enable early stopping (default)')
    g_train.add_argument('--no-early-stop', dest='early_stop', action='store_false',
                         default=None, help='Disable early stopping, run all epochs')
    g_train.add_argument('--patience', type=int, help='Early stopping patience')
    g_train.add_argument('--min-delta', type=float, help='Minimum improvement threshold')
    g_train.add_argument('--use-scheduler', dest='use_scheduler', action='store_true',
                         default=None, help='Enable ReduceLROnPlateau scheduler (default)')
    g_train.add_argument('--no-scheduler', dest='use_scheduler', action='store_false',
                         default=None, help='Disable learning rate scheduler')
    g_train.add_argument('--scheduler-factor', type=float, help='Scheduler decay factor')
    g_train.add_argument('--scheduler-patience', type=int, help='Scheduler patience')
    g_train.add_argument('--grad-clip-norm', type=float,
                         help='Gradient clipping max_norm (0 to disable)')

    # Output
    g_out = parser.add_argument_group('Output')
    g_out.add_argument('--model-dir', type=str, help='Model save directory')
    g_out.add_argument('--plot-dir', type=str, help='Plot save directory')

    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)
    run_training(config)
