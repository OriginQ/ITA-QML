"""快速测试入口 — 单折训练, 用于验证代码和配置"""
import json
from pathlib import Path

from data.dataset import load_dataset, create_cv_splits
from train.trainer import train_model, evaluate_model
from utils.plot import plot_results


def main():
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

    # 路径
    csv_path = data_cfg['csv_path']
    if not Path(csv_path).is_absolute():
        csv_path = str(proj_root / csv_path)

    model_dir = proj_root / out_cfg['model_dir']
    plot_dir = proj_root / out_cfg['plot_dir']
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    # 数据
    features, labels, feature_scaler, label_scaler = load_dataset(
        dataset_name, {**data_cfg, 'csv_path': csv_path})

    print(f'任务: backend={backend}, target={data_cfg["target_col"]}')
    print(f'数据: {csv_path}  (总样本: {len(features)})')

    # 只跑第一折
    splits = list(create_cv_splits(
        features, labels,
        n_splits=data_cfg['n_splits'],
        random_state=data_cfg['random_state'],
    ))
    tr_x, tr_y, val_x, val_y = splits[0]
    print(f'测试折 — 训练集: {len(tr_x)}, 验证集: {len(val_x)}')

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
    )

    # 绘图
    train_pred, train_true = evaluate_model(
        model, tr_x, tr_y, data_cfg['batch_size'], label_scaler, backend=backend)
    val_pred, val_true = evaluate_model(
        model, val_x, val_y, data_cfg['batch_size'], label_scaler, backend=backend)
    plot_results(train_pred, train_true, val_pred, val_true, 0, plot_dir)

    print(f'\n=== 测试结果 ===')
    print(f'训练: RMSE={train_metrics[0]:.4f}, R2={train_metrics[3]:.4f}')
    print(f'验证: RMSE={val_metrics[0]:.4f}, R2={val_metrics[3]:.4f}')
    print('测试通过 ✓')


if __name__ == '__main__':
    main()
