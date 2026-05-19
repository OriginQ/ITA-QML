"""Classical MLP baseline model (pyvqnet framework).

Used as a reference point for comparing against the quantum-classical
hybrid VQC model.
"""
from pyvqnet.nn import Module, Linear, Sequential
from pyvqnet.nn.activation import LeakyReLu
from pyvqnet.nn.dropout import Dropout


class ClassicalNet(Module):
    """Classical MLP baseline for regression.

    Fixed architecture::

        Linear(11 -> 24) -> LeakyReLU -> Dropout
        -> Linear(24 -> 24) -> LeakyReLU -> Linear(24 -> 1)

    Parameters
    ----------
    input_size : int, default=11
        Number of input features.
    dropout_rate : float, default=0.1
        Dropout probability (currently disabled via comment).
    **kwargs : dict
        Ignored; for compatibility with the registry factory signature.
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
        """Forward pass.

        Parameters
        ----------
        x : QTensor
            Input tensor of shape ``(batch, input_size)``.

        Returns
        -------
        QTensor
            Predicted values of shape ``(batch, 1)``.
        """
        return self.net(x)
