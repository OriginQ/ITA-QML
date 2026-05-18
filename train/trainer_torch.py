"""PyTorch 训练器 — 适用于 classical_torch 等 torch 后端"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from train.trainer import calculate_metrics  # 公共指标


# ---------------------------------------------------------------------------
# 数据加载 (torch 版本)
# ---------------------------------------------------------------------------
def _build_torch_loader(features, labels, batch_size, shuffle):
    x = torch.FloatTensor(features)
    y = torch.FloatTensor(labels)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# 验证 / 评估
# ---------------------------------------------------------------------------
def evaluate_model(model, features, labels, batch_size, label_scaler=None):
    model.eval()
    loader = _build_torch_loader(features, labels, batch_size, shuffle=False)
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            pred = model(batch_x).detach().cpu().numpy()
            true = batch_y.numpy()

            if label_scaler is not None:
                pred = label_scaler.inverse_transform(pred)
                true = label_scaler.inverse_transform(true)

            predictions.append(pred)
            actuals.append(true)

    return np.concatenate(predictions), np.concatenate(actuals)


# ---------------------------------------------------------------------------
# 训练主循环 (与 trainer_vqnet 签名一致)
# ---------------------------------------------------------------------------
def train_model(train_features, train_labels, val_features, val_labels,
                label_scaler, model_dir, plot_dir, epochs=100, lr=0.001,
                batch_size=32, fold=0, early_stop=True, patience=10,
                min_delta=1e-4, restore_best_weights=True,
                backend='classical_torch', model_cfg=None,
                use_scheduler=True, scheduler_factor=0.5, scheduler_patience=20,
                grad_clip_norm=1.0):
    from models import create_model

    if model_cfg is None:
        model_cfg = {}
    model = create_model(backend, model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # 学习率调度器 (来自 cml.py)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor,
            patience=scheduler_patience)

    train_loader = _build_torch_loader(
        train_features, train_labels, batch_size, shuffle=True)
    val_loader = _build_torch_loader(
        val_features, val_labels, batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_state = None
    best_param_file = model_dir / f'best_model_fold_{fold + 1}.pth'
    patience_counter = 0
    stopped_early = False

    for epoch in range(epochs):
        # === 训练 ===
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()

            # 梯度裁剪 (来自 cml.py, grad_clip_norm=0 则禁用)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=grad_clip_norm)

            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # === 验证 ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                val_loss += criterion(model(batch_x), batch_y).item()
        val_loss /= max(len(val_loader), 1)

        # 学习率调度
        if scheduler is not None:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch + 1}/{epochs} - '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'LR: {current_lr:.6f}')

        # === 早停 (仅在 early_stop=True 时生效) ===
        if early_stop:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(model.state_dict(), best_param_file)
                if (epoch + 1) % 20 == 0:
                    print(f'  ✓ 验证损失改善, 保存最佳模型 (val_loss={val_loss:.4f})')
            else:
                patience_counter += 1
                if (epoch + 1) % 20 == 0:
                    print(f'  ⚠ 验证损失未改善 {patience_counter}/{patience}')
                if patience_counter >= patience:
                    stopped_early = True
                    print(f'  🛑 早停触发！在 epoch {epoch + 1} 停止训练')
                    break

    # 恢复最佳权重 (仅早停模式)
    if early_stop and stopped_early and restore_best_weights and best_state is not None:
        model.load_state_dict(best_state)
        print(f'  ↻ 已恢复最佳模型权重 (val_loss={best_val_loss:.4f})')

    if not early_stop:
        torch.save(model.state_dict(), best_param_file)
        print(f'训练完成, 模型已保存为 {best_param_file}')
    elif stopped_early:
        print(f'训练提前停止, 最佳模型已保存为 {best_param_file}')
    else:
        print(f'训练完成, 最佳模型已保存为 {best_param_file}')

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
