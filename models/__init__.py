"""Model registry — dispatches model classes by backend name.

Each backend is registered when its framework is available in the
current environment.  Only models whose dependencies are importable
will appear in ``MODEL_REGISTRY``.

To add a new model:
    1. Implement the model class (constructor must accept ``**model_cfg``).
    2. Register it in the appropriate framework block below.
"""
MODEL_REGISTRY = {}

# ---- pyvqnet models ---------------------------------------------------------
try:
    from models.quantum_model import QNet
    from models.classical import ClassicalNet
    MODEL_REGISTRY['vqc'] = QNet
    MODEL_REGISTRY['classical'] = ClassicalNet
except ImportError:
    pass

# ---- PyTorch models ---------------------------------------------------------
try:
    from models.classical_torch import DeepMLP
    MODEL_REGISTRY['classical_torch'] = DeepMLP
except ImportError:
    pass


def create_model(backend, model_cfg):
    """Create a model instance by backend name.

    Parameters
    ----------
    backend : str
        Backend identifier (e.g. ``"vqc"``, ``"classical"``,
        ``"classical_torch"``).
    model_cfg : dict
        Keyword arguments forwarded to the model constructor.

    Returns
    -------
    model : Module or torch.nn.Module
        Instantiated model.

    Raises
    ------
    ValueError
        If *backend* is not registered in ``MODEL_REGISTRY``.
    """
    cls = MODEL_REGISTRY.get(backend)
    if cls is None:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            f"Available in current environment: {list(MODEL_REGISTRY.keys())}")
    return cls(**model_cfg)
