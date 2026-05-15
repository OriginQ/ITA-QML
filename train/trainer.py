"""训练器调度层 — 按 backend 派发到对应框架的训练模块

所有训练器模块 (trainer_vqnet, trainer_torch, ...) 需导出:
    - train_model(train_features, train_labels, val_features, val_labels,
                  label_scaler, model_dir, plot_dir, **train_kwargs)
    - evaluate_model(model, features, labels, batch_size, label_scaler)

新增训练器: 在此文件的 TRAINER_REGISTRY 中注册即可。
"""
import numpy as np
from scipy.stats import spearmanr, pearsonr


# ---------------------------------------------------------------------------
# 公共指标计算 (框架无关)
# ---------------------------------------------------------------------------
def calculate_metrics(predictions, actuals):
    """计算 RMSE / Pearson / Spearman / R²"""

    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    yp = predictions.flatten()
    yt = actuals.flatten()

    rmse = np.sqrt(np.mean((yp - yt) ** 2))
    r2 = r2_score(yt, yp)
    pearson = pearsonr(yp, yt)[0]
    spearman = spearmanr(yp, yt)[0]

    return rmse, pearson, spearman, r2


# ---------------------------------------------------------------------------
# 训练器注册 / 调度
# ---------------------------------------------------------------------------
def _get_module(backend):
    """根据 backend 名返回对应的训练器模块"""
    if backend in ('vqc', 'classical'):
        from train import trainer_vqnet as mod
        return mod
    elif backend in ('classical_torch',):
        from train import trainer_torch as mod
        return mod
    else:
        raise ValueError(
            f"未知 backend: '{backend}'。"
            f"可用: vqc, classical (pyvqnet) / classical_torch (pytorch)")


def train_model(train_features, train_labels, val_features, val_labels,
                label_scaler, model_dir, plot_dir, epochs=100, lr=0.001,
                batch_size=32, fold=0, early_stop=True, patience=10,
                min_delta=1e-4, restore_best_weights=True,
                backend='vqc', model_cfg=None):
    """训练模型 — 按 backend 自动派发到对应框架"""
    mod = _get_module(backend)
    return mod.train_model(
        train_features, train_labels, val_features, val_labels,
        label_scaler=label_scaler,
        model_dir=model_dir, plot_dir=plot_dir,
        epochs=epochs, lr=lr, batch_size=batch_size, fold=fold,
        early_stop=early_stop, patience=patience,
        min_delta=min_delta, restore_best_weights=restore_best_weights,
        backend=backend, model_cfg=model_cfg,
    )


def evaluate_model(model, features, labels, batch_size, label_scaler=None, backend='vqc'):
    """评估模型 — 按 backend 自动派发到对应框架"""
    mod = _get_module(backend)
    return mod.evaluate_model(model, features, labels, batch_size, label_scaler)
