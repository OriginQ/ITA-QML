"""Quantum-classical hybrid model: VQCLayer and QNet.

Implements a variational quantum circuit layer that encodes classical
data into rotation angles on qubits, applies trainable variational
gates with CNOT entanglement, and measures Pauli-Z expectation values.
"""
import numpy as np
from pyvqnet.nn import Module, Linear, ModuleList
from pyvqnet.nn.activation import LeakyReLu
from pyvqnet.nn.dropout import Dropout
from pyvqnet.qnn.vqc import QMachine, RX, CNOT, Probability
from pyvqnet.tensor import QTensor


class VQCLayer(Module):
    """Variational quantum circuit layer.

    Equivalent to PennyLane's ``AngleEmbedding + BasicEntanglerLayers``
    followed by Pauli-Z expectation value measurement.  Uses the VQNet
    built-in VQC automatic-differentiation simulator (no pyqpanda3
    dependency).

    Circuit structure per layer::

        RX_encode(x_i) on each qubit i
        -> RX_var(theta) on each qubit
        -> CNOT chain (i, i+1) across all qubits
        -> Probability measurement -> <Z_i> expectation

    Parameters
    ----------
    qubit_num : int, default=6
        Number of qubits in the circuit.
    layers : int, default=1
        Number of variational circuit layers (repetitions).
    """
    def __init__(self, qubit_num=6, layers=1):
        super().__init__()
        self.qubit_num = qubit_num
        self.n_layers = layers
        self.device = QMachine(qubit_num)

        # AngleEmbedding: one RX encoding gate per qubit.
        self.encode_gates = ModuleList([RX(wires=i) for i in range(qubit_num)])

        # Trainable variational RX gates.
        self.var_gates = ModuleList()
        for _ in range(layers):
            for i in range(qubit_num):
                self.var_gates.append(RX(has_params=True, trainable=True, wires=i))

        # CNOT entanglement chain.
        self.ent_gates = ModuleList()
        for _ in range(layers):
            for i in range(qubit_num - 1):
                self.ent_gates.append(CNOT(wires=[i, i + 1]))

        # Full probability measurement on all qubits.
        self.measure = Probability(wires=list(range(qubit_num)))

        # Precomputed transformation matrix: marginal probabilities -> <Z_i>.
        self._build_mask()

    def _build_mask(self):
        """Build the measurement mask for computing Pauli-Z expectations.

        Constructs a matrix that maps the full probability distribution
        over computational basis states to per-qubit <Z_i> expectation
        values via::

            <Z_i> = 2 * P(|0>_i) - 1

        where P(|0>_i) is obtained by marginalizing over the other qubits.
        """
        n = self.qubit_num
        n_states = 2 ** n
        mask = np.zeros([n, n_states], dtype=np.float32)
        for i in range(n):
            stride = 2 ** (n - i - 1)
            for j in range(0, n_states, 2 * stride):
                mask[i, j:j + stride] = 1.0
        self.mask_matrix = QTensor(mask, requires_grad=False)

    def forward(self, x):
        """Forward pass through the VQC layer.

        Parameters
        ----------
        x : QTensor
            Input tensor of shape ``(batch, qubit_num)``.  Each column
            encodes the rotation angle for the corresponding qubit's
            RX encoding gate.

        Returns
        -------
        QTensor
            Pauli-Z expectation values of shape ``(batch, qubit_num)``.
        """
        batchsize = x.shape[0]
        self.device.reset_states(batchsize)

        # Encode classical input into RX rotation angles.
        for i in range(self.qubit_num):
            self.encode_gates[i](params=x[:, i:i + 1], q_machine=self.device)

        # Variational layers: RX_var + CNOT chain.
        for layer in range(self.n_layers):
            offset = layer * self.qubit_num
            for i in range(self.qubit_num):
                self.var_gates[offset + i](q_machine=self.device)

            ent_offset = layer * (self.qubit_num - 1)
            for i in range(self.qubit_num - 1):
                self.ent_gates[ent_offset + i](q_machine=self.device)

        # Probability measurement -> [batch, 2^n].
        probs = self.measure(q_machine=self.device)

        # <Z_i> = 2 * P(|0>_i) - 1.
        p_zero = probs @ self.mask_matrix.transpose([1, 0])  # [batch, n_qubits]
        return 2.0 * p_zero - 1.0


class QNet(Module):
    """Quantum-classical hybrid regression model.

    Architecture::

        Linear(11 -> 24) -> LeakyReLU -> Dropout
        -> Linear(24 -> qubit_num) -> VQCLayer(qubit_num)
        -> Linear(qubit_num -> qubit_num) -> Linear(qubit_num -> 1)

    The classical front-end reduces the 11 molecular descriptors to a
    ``qubit_num``-dimensional representation that is fed as rotation
    angles into the variational quantum circuit.

    Parameters
    ----------
    qubit_num : int, default=6
        Number of qubits in the VQC layer.
    input_size : int, default=11
        Number of input features (molecular descriptors).
    hidden_size : int, default=24
        Dimension of the hidden classical layer.
    dropout_rate : float, default=0.1
        Dropout probability applied after the first LeakyReLU.
    **kwargs : dict
        Ignored; for compatibility with the registry factory signature.
    """
    def __init__(self, qubit_num=6, input_size=11, hidden_size=24,
                 dropout_rate=0.1, **kwargs):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.leaky_relu = LeakyReLu(0.1)
        self.dropout1 = Dropout(dropout_rate)
        self.fc2 = Linear(hidden_size, qubit_num)
        self.qlayer = VQCLayer(qubit_num=qubit_num, layers=1)
        self.fc3 = Linear(qubit_num, qubit_num)
        self.out = Linear(qubit_num, 1)

    def forward(self, x):
        """Forward pass through the hybrid model.

        Parameters
        ----------
        x : QTensor
            Input tensor of shape ``(batch, input_size)``.

        Returns
        -------
        QTensor
            Predicted values of shape ``(batch, 1)``.
        """
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.qlayer(x)
        x = self.fc3(x)
        x = self.out(x)
        return x
