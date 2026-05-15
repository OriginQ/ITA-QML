"""VQC 训练 CLI — 命令行 + JSON 配置文件驱动

用法:
    python train.py                          # 使用默认 configs/train_config.json
    python train.py --config configs/exp1.json
    python train.py --backend classical      # 切换计算后端
    python train.py --target-col Corr_CCSD   # 切换回归目标
    python train.py --epochs 200 --lr 0.0005 # 命令行覆盖 JSON 超参数
    python train.py --data-path data/ben.csv --n-splits 5

扩展新模型/数据集:
    1. 实现模型类 → models/__init__.py 的 MODEL_REGISTRY 注册
    2. 实现数据集加载函数 → data/dataset.py 的 DATASET_REGISTRY 注册
    3. JSON 配置中设置 backend / dataset 名即可
"""
import argparse
import json
from pathlib import Path

import numpy as np

from data.dataset import load_dataset, create_cv_splits
from train.trainer import train_model, evaluate_model
from utils.plot import plot_results


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------
def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)


def merge_cli_overrides(cfg, args):
    """命令行参数覆盖 JSON 配置"""
    overrides = {
        'training': ['epochs', 'lr', 'early_stop', 'patience', 'min_delta', 'restore_best_weights'],
        'model': ['hidden_size', 'dropout_rate', 'qubit_num', 'qc_layers'],
        'data': ['batch_size', 'n_splits', 'random_state', 'target_col'],
    }

    for section, keys in overrides.items():
        for key in keys:
            val = getattr(args, key, None)
            if val is not None:
                cfg[section][key] = val

    # 顶层覆盖
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
# 训练流水线
# ---------------------------------------------------------------------------
def run_training(cfg):
    """按配置执行完整训练流水线"""
    proj_root = Path(__file__).resolve().parent

    task_cfg = cfg['task']
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['training']
    out_cfg = cfg['output']

    backend = task_cfg['backend']
    dataset_name = task_cfg['dataset']

    # --- 路径 ---
    # csv_path 相对于项目根目录或绝对路径
    csv_path = data_cfg.get('csv_path', '')
    if not Path(csv_path).is_absolute():
        csv_path = str(proj_root / csv_path)
    data_cfg = {**data_cfg, 'csv_path': csv_path}

    model_dir = proj_root / out_cfg['model_dir']
    plot_dir = proj_root / out_cfg['plot_dir']
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    # --- 数据 ---
    features, labels, feature_scaler, label_scaler = load_dataset(dataset_name, data_cfg)

    # --- 配置摘要 ---
    print(f'任务: backend={backend}, target={data_cfg.get("target_col", "?")}')
    print(f'数据: {csv_path}  (总样本: {len(features)})')
    print(f'模型配置: {json.dumps(model_cfg, indent=2)}')
    print(f'训练配置: {json.dumps(train_cfg, indent=2)}')

    # --- 交叉验证 ---
    all_train_metrics = []
    all_val_metrics = []

    for fold, (tr_x, tr_y, val_x, val_y) in enumerate(
            create_cv_splits(features, labels,
                             n_splits=data_cfg['n_splits'],
                             random_state=data_cfg['random_state'])):
        print(f'\n{"=" * 60}')
        print(f'=== Fold {fold + 1}/{data_cfg["n_splits"]} ===')
        print(f'训练集: {len(tr_x)}, 验证集: {len(val_x)}')

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
        )

        train_pred, train_true = evaluate_model(
            model, tr_x, tr_y, data_cfg['batch_size'], label_scaler, backend=backend)
        val_pred, val_true = evaluate_model(
            model, val_x, val_y, data_cfg['batch_size'], label_scaler, backend=backend)
        plot_results(train_pred, train_true, val_pred, val_true, fold, plot_dir)

        all_train_metrics.append(train_metrics)
        all_val_metrics.append(val_metrics)

    print_cv_summary(all_train_metrics, all_val_metrics)
    print(f'\n模型保存至: {model_dir}')
    print(f'图表保存至: {plot_dir}')


def print_cv_summary(all_train_metrics, all_val_metrics):
    def mean_std(values):
        return np.mean(values), np.std(values)

    train_metrics = np.array(all_train_metrics)
    val_metrics = np.array(all_val_metrics)
    names = ['RMSE', 'Pearson', 'Spearman', 'R2']

    print('\n' + '=' * 60)
    print('=== 交叉验证总结 ===')
    print('\n训练集:')
    for i, name in enumerate(names):
        m, s = mean_std(train_metrics[:, i])
        print(f'  {name}: {m:.4f} ± {s:.4f}')
    print('\n验证集:')
    for i, name in enumerate(names):
        m, s = mean_std(val_metrics[:, i])
        print(f'  {name}: {m:.4f} ± {s:.4f}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description='VQC 变分量子分类器 — 训练入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train.py
  python train.py --config configs/train_config.json
  python train.py --backend classical
  python train.py --target-col Corr_CCSD
  python train.py --epochs 200 --lr 0.0005 --batch-size 64
  python train.py --data-path data/ben.csv --n-splits 5
        """,
    )

    parser.add_argument('--config', type=str, default='configs/train_config.json',
                        help='JSON 配置文件路径')

    # 任务
    g_task = parser.add_argument_group('任务')
    g_task.add_argument('--backend', type=str, help='计算后端: vqc / classical / classical_torch')

    # 数据
    g_data = parser.add_argument_group('数据')
    g_data.add_argument('--data-path', type=str, help='CSV 数据路径')
    g_data.add_argument('--target-col', type=str, help='回归目标列名: Corr_MP2 / Corr_CCSD / Corr_CCSD(T)')
    g_data.add_argument('--batch-size', type=int, help='批次大小')
    g_data.add_argument('--n-splits', type=int, help='交叉验证折数')
    g_data.add_argument('--random-state', type=int, help='随机种子')

    # 模型
    g_model = parser.add_argument_group('模型')
    g_model.add_argument('--qubit-num', type=int, help='量子比特数 (仅 VQC)')
    g_model.add_argument('--hidden-size', type=int, help='隐藏层维度')
    g_model.add_argument('--dropout-rate', type=float, help='Dropout 比例')
    g_model.add_argument('--qc-layers', type=int, help='量子电路层数 (仅 VQC)')

    # 训练
    g_train = parser.add_argument_group('训练')
    g_train.add_argument('--epochs', type=int, help='训练轮数')
    g_train.add_argument('--lr', type=float, help='学习率')
    g_train.add_argument('--early-stop', dest='early_stop', action='store_true', default=None,
                         help='启用早停法 (默认)')
    g_train.add_argument('--no-early-stop', dest='early_stop', action='store_false', default=None,
                         help='禁用早停法, 训练完所有 epochs')
    g_train.add_argument('--patience', type=int, help='早停耐心值')
    g_train.add_argument('--min-delta', type=float, help='早停最小改善阈值')

    # 输出
    g_out = parser.add_argument_group('输出')
    g_out.add_argument('--model-dir', type=str, help='模型保存目录')
    g_out.add_argument('--plot-dir', type=str, help='图表保存目录')

    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)
    run_training(config)
