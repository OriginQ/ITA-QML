"""经典深度 MLP 模型 (PyTorch) — 供 classical_torch 后端使用"""
import torch


class DeepMLP(torch.nn.Module):
    """经典深度 MLP 回归模型

    固定结构: Linear(11→24) → LeakyReLU → Dropout
               → Linear(24→24) → LeakyReLU → Linear(24→1)
    """
    def __init__(self, input_size=11, dropout_rate=0.1,
                 leaky_relu_slope=0.1, **kwargs):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 24),
            torch.nn.LeakyReLU(leaky_relu_slope),
            # torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(24, 24),
            torch.nn.LeakyReLU(leaky_relu_slope),
            torch.nn.Linear(24, 1),
        )

    def forward(self, x):
        return self.net(x)
