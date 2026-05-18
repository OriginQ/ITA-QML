"""量子-经典混合模型: VQCLayer + QNet"""
import numpy as np
from pyvqnet.nn import Module, Linear, ModuleList
from pyvqnet.nn.activation import LeakyReLu
from pyvqnet.nn.dropout import Dropout
from pyvqnet.qnn.vqc import QMachine, RX, CNOT, Probability
from pyvqnet.tensor import QTensor


class VQCLayer(Module):
    """VQC 变分量子电路层

    等效于 PennyLane 的 AngleEmbedding + BasicEntanglerLayers → PauliZ 期望值
    使用 VQNet 内置 VQC 自动微分模拟器（不依赖 pyqpanda3）
    """
    def __init__(self, qubit_num=6, layers=1):
        super().__init__()
        self.qubit_num = qubit_num
        self.n_layers = layers
        self.device = QMachine(qubit_num)

        # AngleEmbedding: 每量子比特一个 RX 编码门
        self.encode_gates = ModuleList([RX(wires=i) for i in range(qubit_num)])

        # 变分 RX 门 (可训练)
        self.var_gates = ModuleList()
        for _ in range(layers):
            for i in range(qubit_num):
                self.var_gates.append(RX(has_params=True, trainable=True, wires=i))

        # CNOT 纠缠链
        self.ent_gates = ModuleList()
        for _ in range(layers):
            for i in range(qubit_num - 1):
                self.ent_gates.append(CNOT(wires=[i, i + 1]))

        # 全量子比特概率测量
        self.measure = Probability(wires=list(range(qubit_num)))

        # 预计算边际概率 → PauliZ 期望值 转换矩阵
        self._build_mask()

    def _build_mask(self):
        n = self.qubit_num
        n_states = 2 ** n
        mask = np.zeros([n, n_states], dtype=np.float32)
        for i in range(n):
            stride = 2 ** (n - i - 1)
            for j in range(0, n_states, 2 * stride):
                mask[i, j:j + stride] = 1.0
        self.mask_matrix = QTensor(mask, requires_grad=False)

    def forward(self, x):
        batchsize = x.shape[0]
        self.device.reset_states(batchsize)

        # 编码经典输入 → RX 门角度
        for i in range(self.qubit_num):
            self.encode_gates[i](params=x[:, i:i + 1], q_machine=self.device)

        # 变分层: RX + CNOT 链
        for layer in range(self.n_layers):
            offset = layer * self.qubit_num
            for i in range(self.qubit_num):
                self.var_gates[offset + i](q_machine=self.device)

            ent_offset = layer * (self.qubit_num - 1)
            for i in range(self.qubit_num - 1):
                self.ent_gates[ent_offset + i](q_machine=self.device)

        # 概率测量 [batch, 2^n]
        probs = self.measure(q_machine=self.device)

        # <Z_i> = 2 * P(|0⟩_i) - 1
        p_zero = probs @ self.mask_matrix.transpose([1, 0])  # [batch, n_qubits]
        return 2.0 * p_zero - 1.0


class QNet(Module):
    """量子-经典混合模型

    结构: Linear(11→24) → LeakyReLU → Dropout → Linear(24→6)
            → VQCLayer(6 qubits) → Linear(6→6) → Linear(6→1)
    """
    def __init__(self, qubit_num=6, input_size=11, hidden_size=24, dropout_rate=0.1, **kwargs):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.leaky_relu = LeakyReLu(0.1)
        self.dropout1 = Dropout(dropout_rate)
        self.fc2 = Linear(hidden_size, qubit_num)
        self.qlayer = VQCLayer(qubit_num=qubit_num, layers=1)
        self.fc3 = Linear(qubit_num, qubit_num)
        self.out = Linear(qubit_num, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.qlayer(x)
        x = self.fc3(x)
        x = self.out(x)
        return x
