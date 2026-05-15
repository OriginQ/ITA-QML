"""模型注册中心 — 按 backend 名字派发模型

每种后端按可用框架自动注册。在特定环境中只导入该环境支持的模型。

新增模型:
    1. 实现模型类, 构造函数接受 **model_cfg
    2. 在下方对应的框架块中注册到 MODEL_REGISTRY
"""
MODEL_REGISTRY = {}

# ---- pyvqnet 模型 ----
try:
    from models.quantum_model import QNet
    from models.classical import ClassicalNet
    MODEL_REGISTRY['vqc'] = QNet
    MODEL_REGISTRY['classical'] = ClassicalNet
except ImportError:
    pass

# ---- PyTorch 模型 ----
try:
    from models.classical_torch import ClassicalTorchNet
    MODEL_REGISTRY['classical_torch'] = ClassicalTorchNet
except ImportError:
    pass


def create_model(backend, model_cfg):
    """根据 backend 名创建模型实例"""
    cls = MODEL_REGISTRY.get(backend)
    if cls is None:
        raise ValueError(
            f"未知 backend: '{backend}'。"
            f"当前环境可用: {list(MODEL_REGISTRY.keys())}")
    return cls(**model_cfg)
