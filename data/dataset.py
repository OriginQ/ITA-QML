"""Data loading module for CSV regression datasets.

Convention: all CSV files use column 0 as an index, columns 1--11 as
11-dimensional features (Shannon through Ig), and the remaining columns
as regression targets.  Switching targets is done by specifying the
``target_col`` key in the data configuration dictionary.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# Feature column range (0-based, excluding the index column).
FEATURE_START = 1   # Shannon
FEATURE_END = 12    # Ig + 1


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_csv_regression(data_cfg):
    """Load a generic CSV regression dataset.

    Parameters
    ----------
    data_cfg : dict
        Configuration dictionary with the following keys:

        csv_path : str
            Path to the CSV file.
        target_col : str
            Name of the target column, e.g. ``"Corr_MP2"``, ``"Corr_CCSD"``,
            or ``"Corr_CCSD(T)"``.

    Returns
    -------
    features : ndarray of shape (n_samples, 11)
        Standardized feature matrix (float32).
    labels : ndarray of shape (n_samples, 1)
        Standardized target values (float32).
    feature_scaler : StandardScaler
        Fitted scaler for the features.
    label_scaler : StandardScaler
        Fitted scaler for the labels.
    """
    data = pd.read_csv(data_cfg['csv_path'])
    features = data.iloc[:, FEATURE_START:FEATURE_END].values
    labels = data[data_cfg['target_col']].values.reshape(-1, 1)

    feature_scaler = StandardScaler()
    features = feature_scaler.fit_transform(features)

    label_scaler = StandardScaler()
    labels = label_scaler.fit_transform(labels)

    return features.astype(np.float32), labels.astype(np.float32), feature_scaler, label_scaler


DATASET_REGISTRY = {
    'csv_regression': load_csv_regression,
}


def load_dataset(dataset_name, data_cfg):
    """Dispatch to a registered dataset loader by name.

    Parameters
    ----------
    dataset_name : str
        Key in ``DATASET_REGISTRY`` identifying the dataset type.
    data_cfg : dict
        Configuration dictionary forwarded to the loader function.

    Returns
    -------
    features : ndarray
    labels : ndarray
    feature_scaler : StandardScaler
    label_scaler : StandardScaler

    Raises
    ------
    ValueError
        If *dataset_name* is not found in ``DATASET_REGISTRY``.
    """
    fn = DATASET_REGISTRY.get(dataset_name)
    if fn is None:
        raise ValueError(f"Unknown dataset type: '{dataset_name}'. "
                         f"Available: {list(DATASET_REGISTRY.keys())}")
    return fn(data_cfg)


# ---------------------------------------------------------------------------
# Cross-validation splits (framework-agnostic)
# ---------------------------------------------------------------------------
def create_cv_splits(features, labels, n_splits=10, random_state=42):
    """Generate K-fold cross-validation train/validation splits.

    Parameters
    ----------
    features : ndarray of shape (n_samples, n_features)
        Feature matrix.
    labels : ndarray of shape (n_samples, n_targets)
        Label matrix.
    n_splits : int, default=10
        Number of folds.
    random_state : int, default=42
        Random seed for reproducible shuffling.

    Yields
    ------
    train_features : ndarray
    train_labels : ndarray
    val_features : ndarray
    val_labels : ndarray
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, val_idx in kf.split(features):
        yield (features[train_idx], labels[train_idx],
               features[val_idx], labels[val_idx])
