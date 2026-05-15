"""数据加载模块 — 统一 CSV 回归数据集 (框架无关)

约定: 所有 CSV 文件列 0 为序号, 列 1~11 为 11 维特征 (Shannon~Ig), 其余列为回归目标。
      通过 target_col 指定目标列名即可切换回归目标。
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# 特征列范围 (0-based, 不含序号列)
FEATURE_START = 1   # Shannon
FEATURE_END = 12    # Ig + 1 = 12


# ---------------------------------------------------------------------------
# 数据集加载
# ---------------------------------------------------------------------------
def load_csv_regression(data_cfg):
    """通用 CSV 回归数据集

    Required keys in data_cfg:
        csv_path   — CSV 文件路径
        target_col — 目标列名, 如 "Corr_MP2" / "Corr_CCSD" / "Corr_CCSD(T)"
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
    """按数据集名派发加载"""
    fn = DATASET_REGISTRY.get(dataset_name)
    if fn is None:
        raise ValueError(f"未知数据集类型: '{dataset_name}'。"
                         f"可用: {list(DATASET_REGISTRY.keys())}")
    return fn(data_cfg)


# ---------------------------------------------------------------------------
# 交叉验证数据分割 (框架无关)
# ---------------------------------------------------------------------------
def create_cv_splits(features, labels, n_splits=10, random_state=42):
    """KFold 交叉验证数据分割生成器"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, val_idx in kf.split(features):
        yield (features[train_idx], labels[train_idx],
               features[val_idx], labels[val_idx])
