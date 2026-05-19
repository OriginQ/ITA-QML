"""Quick-test entry point for single-fold training and configuration validation.

This script loads a JSON configuration, runs a single fold of training
(fold 0 only), evaluates the model, and generates a scatter plot. It is
intended for rapid code and configuration smoke-testing.
"""
import json
from pathlib import Path

from data.dataset import load_dataset, create_cv_splits
from train.trainer import train_model, evaluate_model
from utils.plot import plot_results


def main():
    """Run a single-fold training pass for quick validation.

    Loads ``configs/train_config.json``, runs fold 0, prints metrics,
    and saves a predictions scatter plot.
    """
    proj_root = Path(__file__).resolve().parent

    with open(proj_root / 'configs' / 'train_config.json') as f:
        cfg = json.load(f)

    task_cfg = cfg['task']
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['training']
    out_cfg = cfg['output']

    backend = task_cfg['backend']
    dataset_name = task_cfg['dataset']

    csv_path = data_cfg['csv_path']
    if not Path(csv_path).is_absolute():
        csv_path = str(proj_root / csv_path)

    model_dir = proj_root / out_cfg['model_dir']
    plot_dir = proj_root / out_cfg['plot_dir']
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    features, labels, feature_scaler, label_scaler = load_dataset(
        dataset_name, {**data_cfg, 'csv_path': csv_path})

    print(f'Task: backend={backend}, target={data_cfg["target_col"]}')
    print(f'Data: {csv_path}  (total samples: {len(features)})')

    # Run only the first fold.
    splits = list(create_cv_splits(
        features, labels,
        n_splits=data_cfg['n_splits'],
        random_state=data_cfg['random_state'],
    ))
    tr_x, tr_y, val_x, val_y = splits[0]
    print(f'Test fold — train: {len(tr_x)}, val: {len(val_x)}')

    model, train_metrics, val_metrics = train_model(
        tr_x, tr_y, val_x, val_y,
        label_scaler=label_scaler,
        model_dir=model_dir,
        plot_dir=plot_dir,
        epochs=train_cfg['epochs'],
        lr=train_cfg['lr'],
        batch_size=data_cfg['batch_size'],
        fold=0,
        early_stop=train_cfg.get('early_stop', True),
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
    plot_results(train_pred, train_true, val_pred, val_true, 0, plot_dir)

    print(f'\n=== Test Results ===')
    print(f'Train: RMSE={train_metrics[0]:.4f}, R2={train_metrics[3]:.4f}')
    print(f'Val:   RMSE={val_metrics[0]:.4f}, R2={val_metrics[3]:.4f}')
    print('Test passed.')


if __name__ == '__main__':
    main()
