"""pyvqnet trainer for ``vqc`` and ``classical`` backends."""
import numpy as np
from pyvqnet.nn import MeanSquaredError
from pyvqnet.optim import Adam
from pyvqnet.tensor import QTensor
import pyvqnet

from models import create_model
from train.trainer import calculate_metrics


# ---------------------------------------------------------------------------
# pyvqnet batch generator (numpy -> QTensor)
# ---------------------------------------------------------------------------
def batch_generator(features, labels, batch_size, shuffle=True):
    """Yield QTensor mini-batches from numpy arrays.

    Parameters
    ----------
    features : ndarray
        Feature matrix.
    labels : ndarray
        Label matrix.
    batch_size : int
        Number of samples per batch.
    shuffle : bool, default=True
        Whether to shuffle indices before each epoch.

    Yields
    ------
    x : QTensor
        Feature batch of shape ``(batch_size, n_features)``.
    y : QTensor
        Label batch of shape ``(batch_size, n_targets)``.
    """
    n = len(features)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        x = QTensor(features[batch_idx], requires_grad=False,
                     dtype=pyvqnet.kfloat32)
        y = QTensor(labels[batch_idx], requires_grad=False,
                     dtype=pyvqnet.kfloat32)
        yield x, y


# ---------------------------------------------------------------------------
# pyvqnet parameter serialisation
# ---------------------------------------------------------------------------
def _save_params(model, filepath):
    """Save model parameters to a ``.npz`` file.

    Parameters
    ----------
    model : Module
        pyvqnet model.
    filepath : pathlib.Path
        Output path (``.npz`` extension recommended).
    """
    params = {name: param.to_numpy() for name, param in model.named_parameters()}
    np.savez(filepath, **params)


def _load_params(model, filepath):
    """Load model parameters from a ``.npz`` file.

    Parameters
    ----------
    model : Module
        pyvqnet model (modified in-place).
    filepath : pathlib.Path
        Path to the ``.npz`` checkpoint.
    """
    data = np.load(filepath)
    for name, param in model.named_parameters():
        if name in data:
            new_qt = QTensor(data[name], requires_grad=True,
                             dtype=pyvqnet.kfloat32)
            param.data = new_qt.data


# ---------------------------------------------------------------------------
# Validation / evaluation
# ---------------------------------------------------------------------------
def _validate(model, loss_fn, features, labels, batch_size):
    """Compute average validation loss over the full dataset.

    Parameters
    ----------
    model : Module
        pyvqnet model (evaluated in ``eval`` mode).
    loss_fn : callable
        Loss function.
    features : ndarray
    labels : ndarray
    batch_size : int

    Returns
    -------
    float
        Mean loss across all batches.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch_x, batch_y in batch_generator(
            features, labels, batch_size, shuffle=False):
        pred = model(batch_x)
        loss = loss_fn(batch_y, pred)
        total_loss += float(loss.to_numpy())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_model(model, features, labels, batch_size, label_scaler=None):
    """Evaluate a pyvqnet model and return (predictions, actuals).

    Parameters
    ----------
    model : Module
        Trained pyvqnet model.
    features : ndarray
    labels : ndarray
    batch_size : int
    label_scaler : StandardScaler, optional
        Scaler for inverse-transforming predictions.

    Returns
    -------
    predictions : ndarray
    actuals : ndarray
    """
    model.eval()
    predictions = []
    actuals = []
    for batch_x, batch_y in batch_generator(
            features, labels, batch_size, shuffle=False):
        pred = model(batch_x).to_numpy()
        true = batch_y.to_numpy()

        if label_scaler is not None:
            pred = label_scaler.inverse_transform(pred)
            true = label_scaler.inverse_transform(true)

        predictions.append(pred)
        actuals.append(true)

    return np.concatenate(predictions), np.concatenate(actuals)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(train_features, train_labels, val_features, val_labels,
                label_scaler, model_dir, plot_dir, epochs=100, lr=0.001,
                batch_size=32, fold=0, early_stop=True, patience=10,
                min_delta=1e-4, restore_best_weights=True,
                backend='vqc', model_cfg=None, **kwargs):
    """Train a pyvqnet model.

    Parameters
    ----------
    train_features : ndarray
    train_labels : ndarray
    val_features : ndarray
    val_labels : ndarray
    label_scaler : StandardScaler
    model_dir : pathlib.Path
    plot_dir : pathlib.Path
    epochs : int, default=100
    lr : float, default=0.001
    batch_size : int, default=32
    fold : int, default=0
    early_stop : bool, default=True
    patience : int, default=10
    min_delta : float, default=1e-4
    restore_best_weights : bool, default=True
    backend : str, default='vqc'
    model_cfg : dict, optional
    **kwargs : dict
        Ignored; for signature compatibility with the dispatcher.

    Returns
    -------
    model : Module
        Trained pyvqnet model.
    train_metrics : tuple
        (RMSE, Pearson r, Spearman rho, R²) on training set.
    val_metrics : tuple
        (RMSE, Pearson r, Spearman rho, R²) on validation set.
    """
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
        # --- Train ---
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

        # --- Validate ---
        val_loss = _validate(model, loss_fn, val_features, val_labels, batch_size)

        print(f'Epoch {epoch + 1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # --- Early stopping ---
        if early_stop:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                _save_params(model, best_param_file)
                print(f'  -> val loss improved, saved best model '
                      f'(val_loss={val_loss:.4f})')
            else:
                patience_counter += 1
                print(f'  -> val loss did not improve '
                      f'{patience_counter}/{patience}')
                if patience_counter >= patience:
                    stopped_early = True
                    print(f'  -- early stopping triggered at epoch {epoch + 1}')
                    break

    # Restore best weights (early-stop mode only).
    if (early_stop and stopped_early and restore_best_weights
            and best_param_file.exists()):
        _load_params(model, best_param_file)
        print(f'  -> restored best model weights (val_loss={best_val_loss:.4f})')

    if not early_stop:
        _save_params(model, best_param_file)
        print(f'Training complete, model saved to {best_param_file}')
    elif stopped_early:
        print(f'Training stopped early, best model saved to {best_param_file}')
    else:
        print(f'Training complete, best model saved to {best_param_file}')

    # --- Final evaluation ---
    train_pred, train_true = evaluate_model(
        model, train_features, train_labels, batch_size, label_scaler)
    val_pred, val_true = evaluate_model(
        model, val_features, val_labels, batch_size, label_scaler)

    train_metrics = calculate_metrics(train_pred, train_true)
    val_metrics = calculate_metrics(val_pred, val_true)

    print(f'\nFold {fold + 1} train metrics:')
    print(f'RMSE: {train_metrics[0]:.4f}, Pearson: {train_metrics[1]:.4f}, '
          f'Spearman: {train_metrics[2]:.4f}, R2: {train_metrics[3]:.4f}')
    print(f'\nFold {fold + 1} val metrics:')
    print(f'RMSE: {val_metrics[0]:.4f}, Pearson: {val_metrics[1]:.4f}, '
          f'Spearman: {val_metrics[2]:.4f}, R2: {val_metrics[3]:.4f}')

    return model, train_metrics, val_metrics
