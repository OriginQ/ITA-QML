"""Visualization module for regression predictions."""
import numpy as np
import matplotlib.pyplot as plt


def plot_results(train_pred, train_true, val_pred, val_true, fold, plot_dir):
    """Plot predicted vs. true values and save to disk.

    Creates a scatter plot with training points in blue, validation
    points in red, and a ``y = x`` reference line.  The figure is saved
    as ``predictions_plot_fold_{n}.png`` in *plot_dir*.

    Parameters
    ----------
    train_pred : ndarray
        Predicted values for the training set.
    train_true : ndarray
        Ground-truth values for the training set.
    val_pred : ndarray
        Predicted values for the validation set.
    val_true : ndarray
        Ground-truth values for the validation set.
    fold : int
        Zero-based fold index (used for the plot title and filename).
    plot_dir : pathlib.Path
        Directory in which to save the output PNG.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(train_pred, train_true, c='blue', alpha=0.5, label='Train set')
    plt.scatter(val_pred, val_true, c='red', alpha=0.5, label='Validation set')

    all_vals = np.concatenate([train_pred, train_true, val_pred, val_true])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    margin = 0.1 * (max_val - min_val)

    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x')
    plt.xlim(min_val - margin, max_val + margin)
    plt.ylim(min_val - margin, max_val + margin)
    plt.xlabel('Predicted Value')
    plt.ylabel('Real Value')
    plt.title(f'Fold {fold + 1} - Predicted Value & Real Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir / f'predictions_plot_fold_{fold + 1}.png')
    plt.close()
