"""经典 MLP 模型 (PyTorch) — 供 classical_torch 后端使用

协作者参考:
    1. 继承 torch.nn.Module
    2. 在 models/__init__.py MODEL_REGISTRY 注册
    3. trainer_torch.py 已准备好接收任意 torch 模型
"""
import torch


class ClassicalTorchNet(torch.nn.Module):
    """经典 MLP 基线 (PyTorch 版本)

    结构: Linear(in→hidden) → LeakyReLU → Dropout
            → Linear(hidden→hidden/2) → LeakyReLU
            → Linear(hidden/2→1)
    """
    def __init__(self, input_size=11, hidden_size=24, dropout_rate=0.1,
                 **kwargs):
        super().__init__()
        hidden2 = max(hidden_size // 2, 4)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size, hidden2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x)
