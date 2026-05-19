"""Classical deep MLP model (PyTorch framework).

Provides the ``classical_torch`` backend for comparison against the
pyvqnet-based classical and VQC models.
"""
import torch


class DeepMLP(torch.nn.Module):
    """Classical deep MLP regression model.

    Fixed architecture::

        Linear(11 -> 24) -> LeakyReLU -> Dropout
        -> Linear(24 -> 24) -> LeakyReLU -> Linear(24 -> 1)

    Parameters
    ----------
    input_size : int, default=11
        Number of input features.
    dropout_rate : float, default=0.1
        Dropout probability (currently disabled via comment).
    leaky_relu_slope : float, default=0.1
        Negative slope for LeakyReLU activations.
    **kwargs : dict
        Ignored; for compatibility with the registry factory signature.
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
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, input_size)``.

        Returns
        -------
        torch.Tensor
            Predicted values of shape ``(batch, 1)``.
        """
        return self.net(x)
