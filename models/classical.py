"""纯经典神经网络 — 基线模型, 用于与 VQC 对比

供协作者参考: 实现经典模型时继承 pyvqnet.nn.Module,
在 models/__init__.py 的 MODEL_REGISTRY 中注册即可。
"""
from pyvqnet.nn import Module, Linear, Sequential
from pyvqnet.nn.activation import LeakyReLu
from pyvqnet.nn.dropout import Dropout


class ClassicalNet(Module):
    """经典 MLP 基线

    结构: Linear(in→hidden) → LeakyReLU → Dropout
            → Linear(hidden→hidden/2) → LeakyReLU
            → Linear(hidden/2→1)
    """
    def __init__(self, input_size=11, hidden_size=24, dropout_rate=0.1,
                 **kwargs):
        super().__init__()
        hidden2 = max(hidden_size // 2, 4)
        self.net = Sequential(
            Linear(input_size, hidden_size),
            LeakyReLu(0.1),
            Dropout(dropout_rate),
            Linear(hidden_size, hidden2),
            LeakyReLu(0.1),
            Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x)
