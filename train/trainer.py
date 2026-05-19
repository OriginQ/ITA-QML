"""Training dispatcher — routes to framework-specific trainer modules.

Each trainer module (``trainer_vqnet``, ``trainer_torch``, ...) must
export the following two functions with matching signatures:

    ``train_model(train_features, train_labels, val_features, val_labels,
                  label_scaler, model_dir, plot_dir, **train_kwargs)``

    ``evaluate_model(model, features, labels, batch_size, label_scaler)``

To add a new trainer, register it in ``_get_module()`` below.
"""
import numpy as np
from scipy.stats import spearmanr, pearsonr


# ---------------------------------------------------------------------------
# Shared metrics (framework-agnostic)
# ---------------------------------------------------------------------------
def calculate_metrics(predictions, actuals):
    """Compute standard regression metrics.

    Parameters
    ----------
    predictions : ndarray
        Predicted values.
    actuals : ndarray
        Ground-truth values.

    Returns
    -------
    rmse : float
        Root mean squared error.
    pearson : float
        Pearson correlation coefficient.
    spearman : float
        Spearman rank correlation coefficient.
    r2 : float
        Coefficient of determination (R²).
    """
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
# Trainer registry / dispatch
# ---------------------------------------------------------------------------
def _get_module(backend):
    """Return the trainer module for the given backend.

    Parameters
    ----------
    backend : str
        Backend identifier.

    Returns
    -------
    module
        Trainer module with ``train_model`` and ``evaluate_model``.

    Raises
    ------
    ValueError
        If *backend* is not recognised.
    """
    if backend in ('vqc', 'classical'):
        from train import trainer_vqnet as mod
        return mod
    elif backend in ('classical_torch',):
        from train import trainer_torch as mod
        return mod
    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            f"Available: vqc, classical (pyvqnet) / classical_torch (pytorch)")


def train_model(train_features, train_labels, val_features, val_labels,
                label_scaler, model_dir, plot_dir, epochs=100, lr=0.001,
                batch_size=32, fold=0, early_stop=True, patience=10,
                min_delta=1e-4, restore_best_weights=True,
                backend='vqc', model_cfg=None,
                use_scheduler=True, scheduler_factor=0.5, scheduler_patience=20,
                grad_clip_norm=1.0):
    """Train a model, dispatching to the correct framework by *backend*.

    Parameters
    ----------
    train_features : ndarray
        Training feature matrix.
    train_labels : ndarray
        Training labels.
    val_features : ndarray
        Validation feature matrix.
    val_labels : ndarray
        Validation labels.
    label_scaler : StandardScaler
        Fitted scaler for inverse-transforming predictions at evaluation time.
    model_dir : pathlib.Path
        Directory for saving model checkpoints.
    plot_dir : pathlib.Path
        Directory for saving plots (forwarded but usually unused by
        the trainer itself).
    epochs : int, default=100
        Maximum number of training epochs.
    lr : float, default=0.001
        Initial learning rate.
    batch_size : int, default=32
        Mini-batch size.
    fold : int, default=0
        Zero-based fold index (used for checkpoint filenames).
    early_stop : bool, default=True
        Whether to enable early stopping.
    patience : int, default=10
        Number of epochs with no improvement before stopping.
    min_delta : float, default=1e-4
        Minimum absolute change to count as an improvement.
    restore_best_weights : bool, default=True
        Whether to reload the best checkpoint after early stopping.
    backend : str, default='vqc'
        Backend identifier (``"vqc"``, ``"classical"``,
        ``"classical_torch"``).
    model_cfg : dict, optional
        Keyword arguments forwarded to ``create_model``.
    use_scheduler : bool, default=True
        Whether to use ReduceLROnPlateau.
    scheduler_factor : float, default=0.5
        Multiplicative factor for LR reduction.
    scheduler_patience : int, default=20
        Epochs with no improvement before reducing LR.
    grad_clip_norm : float, default=1.0
        Maximum gradient norm for clipping (0 disables).

    Returns
    -------
    model : Module or torch.nn.Module
        Trained model.
    train_metrics : tuple
        (RMSE, Pearson r, Spearman rho, R²) on the training set.
    val_metrics : tuple
        (RMSE, Pearson r, Spearman rho, R²) on the validation set.
    """
    mod = _get_module(backend)
    return mod.train_model(
        train_features, train_labels, val_features, val_labels,
        label_scaler=label_scaler,
        model_dir=model_dir, plot_dir=plot_dir,
        epochs=epochs, lr=lr, batch_size=batch_size, fold=fold,
        early_stop=early_stop, patience=patience,
        min_delta=min_delta, restore_best_weights=restore_best_weights,
        backend=backend, model_cfg=model_cfg,
        use_scheduler=use_scheduler, scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        grad_clip_norm=grad_clip_norm,
    )


def evaluate_model(model, features, labels, batch_size,
                   label_scaler=None, backend='vqc'):
    """Evaluate a model, dispatching to the correct framework by *backend*.

    Parameters
    ----------
    model : Module or torch.nn.Module
        Trained model.
    features : ndarray
        Input features.
    labels : ndarray
        Ground-truth labels.
    batch_size : int
        Batch size for evaluation.
    label_scaler : StandardScaler, optional
        Scaler for inverse-transforming predictions back to original
        units.
    backend : str, default='vqc'
        Backend identifier.

    Returns
    -------
    predictions : ndarray
        Model predictions (inverse-transformed if *label_scaler* is
        provided).
    actuals : ndarray
        Ground-truth values (inverse-transformed if *label_scaler* is
        provided).
    """
    mod = _get_module(backend)
    return mod.evaluate_model(model, features, labels, batch_size, label_scaler)
