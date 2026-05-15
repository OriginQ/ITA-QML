# 经典模型协作者指南

本指南教你如何在这个项目中接入自己的经典 ML 模型，以及如何使用 pixi 管理环境。

---

## 一、项目结构速览

```
qme_vqnet/
├── pixi.toml                  # 环境 & 任务定义
├── configs/
│   └── train_config.json      # 训练配置 (改参数改这个)
├── train.py                   # CLI 完整训练入口
├── main.py                    # 快速测试入口 (单折验证)
├── models/
│   ├── __init__.py            # 【注册中心 — 新增模型改这里】
│   ├── classical_torch.py     # 你的 torch 模型放这里 (或其他新文件)
│   ├── classical.py           # pyvqnet 版经典 MLP (忽略)
│   └── quantum_model.py       # VQC 量子模型 (忽略)
├── train/
│   ├── trainer.py             # 【调度层 — 新增训练器改这里】
│   ├── trainer_torch.py       # torch 训练循环 (早停/评估/保存)
│   └── trainer_vqnet.py       # pyvqnet 训练循环 (忽略)
├── data/
│   ├── dataset.py             # 数据加载 (特征列 1~11, 目标列按名指定)
│   └── *.csv                  # 数据集文件
└── utils/
    └── plot.py                # 绘图 (自动调用, 不需关心)
```

**两个独立环境:**

| 环境 | 框架 | 你的工作区 |
|---|---|---|
| `torch` | PyTorch CPU, sklearn, scipy, pandas | **你只用这个** |
| `pyvqnet` (默认) | pyvqnet + pyqpanda3 (量子) | 不用管 |

**或者你想用你本地的环境(conda, uv, venv)也可以，pixi交给我维护。**

---

## 二、三步接入新模型

假设你要加一个 `Net` 模型。只需改 3 个地方。

### Step 1: 写模型类

在 `models/` 下新建文件，或直接在 `classical_torch.py` 里加。

**约束:**
- 继承 `torch.nn.Module`
- 构造函数接受 `**model_cfg` 传来的参数 (多余的用 `**kwargs` 吞掉)
- 输入 shape `[batch, input_size]`, 输出 shape `[batch, 1]`

```python
# models/my_net.py
import torch

class Net(torch.nn.Module):
    def __init__(self, input_size=11, hidden_size=24, dropout_rate=0.1,
                 num_heads=4, num_layers=2, **kwargs):   # **kwargs 吞掉不认识的参数
        super().__init__()
        self.input_proj = torch.nn.Linear(input_size, hidden_size)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4, dropout=dropout_rate,
            batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)   # [B, D] → [B, 1, D]
        x = self.transformer(x)               # [B, 1, D]
        return self.output(x.squeeze(1))       # [B, 1]
```

### Step 2: 注册模型

编辑 `models/__init__.py`, 在 `# ---- PyTorch 模型 ----` 块里加一行:

```python
# ---- PyTorch 模型 ----
try:
    from models.classical_torch import ClassicalTorchNet
    from models.my_net import Net   # ← 新增
    MODEL_REGISTRY['classical_torch'] = ClassicalTorchNet
    MODEL_REGISTRY['net'] = Net      # ← 新增, 名字自定
except ImportError:
    pass
```

`MODEL_REGISTRY` 的 key 就是 `backend` 名，后面 JSON 配置和 CLI 都会用到。

### Step 3 (可选): 自定义训练逻辑

绝大多数情况下 **不需要** — 现有的 `trainer_torch.py` 已经能训练任意 `torch.nn.Module`。

只有当你需要特殊的训练逻辑（如自定义 loss、学习率调度、梯度裁剪）时才需要新建训练器。步骤:

1. 复制 `train/trainer_torch.py` → `train/trainer_transformer.py`
2. 修改训练循环，**保持函数签名不变** (`train_model` / `evaluate_model`)
3. 在 `train/trainer.py` 的 `_get_module` 中注册:

```python
def _get_module(backend):
    if backend in ('vqc', 'classical'):
        from train import trainer_vqnet as mod
        return mod
    elif backend in ('classical_torch', 'transformer'):   # ← 新增
        from train import trainer_torch as mod             # 复用 torch trainer
        return mod
```

如果新建了训练器文件，改为 `from train import trainer_transformer as mod`。

---

## 三、通过配置和命令行使用

### JSON 配置

复制一份 config 改 `backend`:

```json
{
  "task": {
    "backend": "your_net",
    "dataset": "csv_regression"
  },
  "data": {
    "csv_path": "data/ben.csv",
    "target_col": "Corr_MP2",
    "batch_size": 32,
    "n_splits": 10,
    "random_state": 42
  },
  "model": {
    "input_size": 11,
    "hidden_size": 64,
    "dropout_rate": 0.1,
    "num_heads": 4,
    "num_layers": 2
  },
  "training": {
    "epochs": 200,
    "lr": 0.0005,
    "early_stop": true,
    "patience": 30,
    "min_delta": 1e-5,
    "restore_best_weights": true
  },
  "output": {
    "model_dir": "transformer_models",
    "plot_dir": "transformer_plots"
  }
}
```

`model` 里的字段直接传给模型构造函数，不认识的字段时间 `**kwargs` 吞掉，不会报错。

### 运行

```bash
# 用你的配置跑
pixi run -e torch python train.py --config configs/your_config.json

# 命令行覆盖参数 (不改 JSON)
pixi run -e torch python train.py --config configs/your_config.json \
    --epochs 100 --lr 0.001 --target-col Corr_CCSD

# 禁用早停, 跑满所有 epochs
pixi run -e torch python train.py --config configs/your_config.json \
    --no-early-stop
```

### 设为 pixi 快捷任务

在 `pixi.toml` 的 `[feature.torch.tasks]` 下加:

```toml
[feature.torch.tasks]
train-torch = "python train.py --config configs/train_config.json --backend classical_torch"
train-your-net = "python train.py --config configs/your_config.json"
```

之后直接:

```bash
pixi run train-your-net
pixi run train-your-net -- --epochs 50   # 额外参数用 -- 分隔
```

---

## 四、数据集说明

所有 CSV 文件遵循统一格式: **第 0 列为序号, 第 1~11 列为 11 维特征 (Shannon ~ Ig)**。

| 文件 | 样本数 | 可选目标列 |
|---|---|---|
| `ben.csv` | 1180 | Corr_MP2, Corr_CCSD, Corr_CCSD(T) |
| `C60_new.csv` | 1811 | Corr_MP2 |
| `Cr2_new.csv` | 299 | Corr_MP2 |
| `all_FCI_H8.csv` | 4000 | Corr_MP2 |
| `all_RHF_H8.csv` | 4000 | Corr_MP2 |
| `water_mp2_ccsd_ccsdt.csv` | 1886 | Corr_MP2, Corr_CCSD, Corr_CCSD(T) |

切换目标列只需要改 `data.target_col`:

```json
"data": { "csv_path": "data/ben.csv", "target_col": "Corr_CCSD(T)" }
```

如果要接入完全不同格式的数据集，在 `data/dataset.py` 中新增加载函数并注册到 `DATASET_REGISTRY`。

---

## 五、pixi 环境管理

### 安装 pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 首次克隆项目后

```bash
cd qme_vqnet
pixi install             # 安装默认环境 (pyvqnet)
pixi install -e torch    # 安装 torch 环境
```

安装完后所有依赖都在 `.pixi/` 目录下，不污染系统 Python。

### 常用命令

```bash
# 查看所有可用任务
pixi task list

# 在指定环境中运行命令
pixi run -e torch python script.py
pixi run -e torch python -c "import torch; print(torch.__version__)"

# 进入环境的交互式 shell
pixi shell -e torch

# 查看环境中已安装的包
pixi list -e torch

# 添加新依赖 (记得同时更新 pixi.toml)
pixi add -e torch --feature torch xgboost

# 更新锁文件
pixi update
```

### 环境 vs Feature 的关系

```
[feature.torch]         ← feature 定义 (依赖列表)
  dependencies = [...]  ← conda 包
  pypi-dependencies     ← pip 包
  tasks                 ← 该环境的快捷任务

[environments]
torch = ["torch"]       ← 环境 "torch" 由 feature "torch" 组成
```

一个环境可以组合多个 feature (如 `torch = ["torch", "dev_tools"]`)，但目前每个环境只对应一个 feature。

### 协作者之间的版本一致性

`pixi.lock` 文件锁定了所有依赖的精确版本。**这个文件要提交到 git**。其他人在 `pixi install` 时会根据锁文件安装完全相同的版本，消除"我这能跑你那不行"的问题。

当有人修改了 `pixi.toml` 中的依赖后，运行 `pixi install` 会自动更新 `pixi.lock`，提交即可。

---

## 六、快速检查清单

接入新模型时，确认以下 4 件事:

- [ ] `models/__init__.py` 的 PyTorch 块中注册了新 backend 名
- [ ] 模型构造函数接受 `**kwargs` (不会被多余 config 字段炸掉)
- [ ] 配置文件 `backend` 与注册名一致
- [ ] `pixi run -e torch python train.py --config configs/xxx.json` 能跑通

不需要改的文件: `trainer_torch.py`、`train.py`、`dataset.py`、`plot.py` (除非有特殊需求)。
