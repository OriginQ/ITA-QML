"""pyvqnet 训练器 — 适用于 vqc / classical 等 pyvqnet 后端"""
import numpy as np
from pyvqnet.nn import MeanSquaredError
from pyvqnet.optim import Adam
from pyvqnet.tensor import QTensor
import pyvqnet

from models import create_model
from train.trainer import calculate_metrics  # 公共指标


# ---------------------------------------------------------------------------
# pyvqnet 批处理 (从 numpy 数组生成 QTensor 批次)
# ---------------------------------------------------------------------------
def batch_generator(features, labels, batch_size, shuffle=True):
    n = len(features)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        x = QTensor(features[batch_idx], requires_grad=False, dtype=pyvqnet.kfloat32)
        y = QTensor(labels[batch_idx], requires_grad=False, dtype=pyvqnet.kfloat32)
        yield x, y


# ---------------------------------------------------------------------------
# pyvqnet 参数保存/恢复
# ---------------------------------------------------------------------------
def _save_params(model, filepath):
    params = {name: param.to_numpy() for name, param in model.named_parameters()}
    np.savez(filepath, **params)


def _load_params(model, filepath):
    data = np.load(filepath)
    for name, param in model.named_parameters():
        if name in data:
            new_qt = QTensor(data[name], requires_grad=True, dtype=pyvqnet.kfloat32)
            param.data = new_qt.data


# ---------------------------------------------------------------------------
# 验证 / 评估
# ---------------------------------------------------------------------------
def _validate(model, loss_fn, features, labels, batch_size):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch_x, batch_y in batch_generator(features, labels, batch_size, shuffle=False):
        pred = model(batch_x)
        loss = loss_fn(batch_y, pred)
        total_loss += float(loss.to_numpy())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_model(model, features, labels, batch_size, label_scaler=None):
    model.eval()
    predictions = []
    actuals = []
    for batch_x, batch_y in batch_generator(features, labels, batch_size, shuffle=False):
        pred = model(batch_x).to_numpy()
        true = batch_y.to_numpy()

        if label_scaler is not None:
            pred = label_scaler.inverse_transform(pred)
            true = label_scaler.inverse_transform(true)

        predictions.append(pred)
        actuals.append(true)

    return np.concatenate(predictions), np.concatenate(actuals)


# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------
def train_model(train_features, train_labels, val_features, val_labels,
                label_scaler, model_dir, plot_dir, epochs=100, lr=0.001,
                batch_size=32, fold=0, early_stop=True, patience=10,
                min_delta=1e-4, restore_best_weights=True,
                backend='vqc', model_cfg=None, **kwargs):
    if model_cfg is None:
        model_cfg = {}
    model = create_model(backend, model_cfg)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MeanSquaredError()

    best_val_loss = float('inf')
    best_param_file = model_dir / f'best_model_fold_{fold + 1}.npz'
    patience_counter = 0
    stopped_early = False

    for epoch in range(epochs):
        # === 训练 ===
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch_x, batch_y in batch_generator(
                train_features, train_labels, batch_size, shuffle=True):
            pred = model(batch_x)
            loss = loss_fn(batch_y, pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer._step()
            train_loss += float(loss.to_numpy())
            n_train += 1
        train_loss /= max(n_train, 1)

        # === 验证 ===
        val_loss = _validate(model, loss_fn, val_features, val_labels, batch_size)

        print(f'Epoch {epoch + 1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # === 早停 (仅在 early_stop=True 时生效) ===
        if early_stop:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                _save_params(model, best_param_file)
                print(f'  ✓ 验证损失改善, 保存最佳模型 (val_loss={val_loss:.4f})')
            else:
                patience_counter += 1
                print(f'  ⚠ 验证损失未改善 {patience_counter}/{patience}')
                if patience_counter >= patience:
                    stopped_early = True
                    print(f'  🛑 早停触发！在 epoch {epoch + 1} 停止训练')
                    break

    # 恢复最佳权重 (仅早停模式)
    if early_stop and stopped_early and restore_best_weights and best_param_file.exists():
        _load_params(model, best_param_file)
        print(f'  ↻ 已恢复最佳模型权重 (val_loss={best_val_loss:.4f})')

    if not early_stop:
        _save_params(model, best_param_file)
        print(f'训练完成, 模型已保存为 {best_param_file}')
    elif stopped_early:
        print(f'训练提前停止, 最佳模型已保存为 {best_param_file}')
    else:
        print(f'训练完成, 最佳模型已保存为 {best_param_file}')

    # === 最终评估 ===
    train_pred, train_true = evaluate_model(
        model, train_features, train_labels, batch_size, label_scaler)
    val_pred, val_true = evaluate_model(
        model, val_features, val_labels, batch_size, label_scaler)

    train_metrics = calculate_metrics(train_pred, train_true)
    val_metrics = calculate_metrics(val_pred, val_true)

    print(f'\nFold {fold + 1}训练集评估指标:')
    print(f'RMSE: {train_metrics[0]:.4f}, Pearson: {train_metrics[1]:.4f}, '
          f'Spearman: {train_metrics[2]:.4f}, R2: {train_metrics[3]:.4f}')
    print(f'\nFold {fold + 1}验证集评估指标:')
    print(f'RMSE: {val_metrics[0]:.4f}, Pearson: {val_metrics[1]:.4f}, '
          f'Spearman: {val_metrics[2]:.4f}, R2: {val_metrics[3]:.4f}')

    return model, train_metrics, val_metrics
