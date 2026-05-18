"""纯经典神经网络 — 基线模型, 用于与 VQC 对比"""
from pyvqnet.nn import Module, Linear, Sequential
from pyvqnet.nn.activation import LeakyReLu
from pyvqnet.nn.dropout import Dropout


class ClassicalNet(Module):
    """经典 MLP 基线

    固定结构: Linear(11→24) → LeakyReLU → Dropout
               → Linear(24→24) → LeakyReLU → Linear(24→1)
    """
    def __init__(self, input_size=11, dropout_rate=0.1, **kwargs):
        super().__init__()
        self.net = Sequential(
            Linear(input_size, 24),
            LeakyReLu(0.1),
            # Dropout(dropout_rate),
            Linear(24, 24),
            LeakyReLu(0.1),
            Linear(24, 1),
        )

    def forward(self, x):
        return self.net(x)
