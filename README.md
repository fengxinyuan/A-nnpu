# A-nnpu 项目框架

本项目包含一系列 Python 脚本，用于在多种图像数据集上进行正例-未标记 (Positive-Unlabeled, PU) 学习实验。项目利用 PyTorch 进行模型定义、训练和评估，并实现了一种基于 nnPU 的损失函数，该损失函数带有一个可选的自适应项。

## 📋 目录

- [项目背景](#项目背景)
- [特性](#特性)
- [先决条件](#先决条件)
- [安装与环境设置](#安装与环境设置)
- [数据集准备](#数据集准备)
- [使用方法](#使用方法)
  - [运行实验脚本](#运行实验脚本)
  - [参数配置](#参数配置)
- [脚本概述](#脚本概述)
- [注意事项与已知问题](#注意事项与已知问题)
- [未来工作 (可选)](#未来工作-可选)

---

## 项目背景

在许多实际应用中，获取大量精确标记的数据是非常昂贵的。PU 学习旨在仅利用少量已标记的正样本和大量未标记样本来训练分类器。本项目提供了一个基于 PyTorch 的框架，用于探索和比较不同 PU 学习策略在图像分类任务上的表现。

---

## ✨ 特性

* **PU 学习核心实现**: 基于 nnPU 损失函数进行模型训练。
* **自适应损失项**: 损失函数中包含一个可选的自适应加权机制，用于调整未标记样本的贡献。
* **模块化设计**: 数据处理、模型定义、损失函数和训练评估流程分离，易于扩展。
* **多数据集支持**: 已集成 CIFAR-10、MNIST、FashionMNIST、EuroSAT 和 PCAM 数据集的 PU 学习设置。
* **模型选择**: 提供了 ResNet18 和一个简单的自定义 CNN (`SimpleMNISTCNN`) 作为基础模型。
* **训练策略**:
    * 可选的预热 (Warmup) 阶段。
    * 早停 (Early Stopping) 机制，基于 F1 分数监控模型性能并防止过拟合。
    * 学习率调度 (Cosine Annealing)。
* **详细评估**: 输出包括准确率、AUC、精确率、召回率和 F1 分数。

---

## 🐍 先决条件

* **Python**: 推荐使用 3.8 或更高版本。
* **Pip**: 用于包管理。
* **PyTorch**: (例如1.9.0+)
* **Torchvision**: (例如0.10.0+, 需与 PyTorch 版本兼容)
* **NumPy**: (例如1.20.0+)
* **Scikit-learn**: (例如0.24.0+)
* **CUDA**: (可选) 用于 GPU 加速。如果计划使用 GPU，请确保 PyTorch 版本与 CUDA 工具包版本兼容。

---

## 🛠️ 安装与环境设置

1.  **克隆仓库 (如果您的代码已在 GitHub 上):**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **创建并激活虚拟环境 (推荐):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **安装依赖包:**
    根据 `requirements.txt` 文件来安装依赖：
    
    ```bash
    pip install -r requirements.txt
    ```
    如果需要手动安装：
    ```bash
    pip install torch torchvision numpy scikit-learn
    ```

---

## 💾 数据集准备

由于数据集太大，在提交时未提供完整数据集，脚本设计为在指定的 `DATA_ROOT` 目录中未找到数据集时自动下载（EuroSAT和PCAM数据集无法自动下载，需要自行下载）。

* **CIFAR-10**: `main.py` 使用。默认 `DATA_ROOT = 'data'`。
* **MNIST**: `main2.py` 使用。默认 `DATA_ROOT = 'data'`。
* **FashionMNIST**: `main3.py` 使用。默认 `DATA_ROOT = 'data'`。
* **EuroSAT**: `main4.py` 使用。默认 `DATA_ROOT = 'data/EuroSAT/2750'`。下载链接: [GitHub - phelber/EuroSAT](https://github.com/phelber/EuroSAT)
* **PCAM (PatchCamelyon)**: `main5.py` 使用。默认 `DATA_ROOT = 'data/PCAM'`。下载链接: [GitHub - basveeling/pcam](https://github.com/basveeling/pcam)

---

## 🚀 使用方法

### 运行实验脚本

每个 `mainX.py` 文件都是一个独立的实验脚本。您可以从终端直接运行：

```bash
python main.py   # CIFAR-10 实验
python main2.py  # MNIST 实验
python main3.py  # FashionMNIST 实验
python main4.py  # EuroSAT 实验
python main5.py  # PCAM 实验
```

### ⚙️ 参数控制

所有实验相关的超参数和配置都集中在每个 `mainX.py` 文件的 `if __name__ == '__main__':` 代码块的起始部分。用户可以直接修改这些变量的值来调整实验设置。

以下是主要的参数及其说明：

- **设备设置:**

  - DEVICE: 自动选择 CUDA (如果可用) 或 CPU。通常无需修改。

    ```Python
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ```

- **数据集相关:**

  - DATA_ROOT: 指定数据集存储的根目录。

    ```Python
    # 示例 (main.py):
    DATA_ROOT = 'data'
    # 示例 (main4.py):
    DATA_ROOT = 'data/EuroSAT/2750'
    ```

  - POS_CLASS_INDICES: 一个列表，定义了原始数据集中哪些类别在 PU 学习中被视为“正类”。例如，在 CIFAR-10 中，类别 0 (飞机) 和 1 (汽车) 被定义为正类。

    ```Python
    # 示例 (main.py):
    POS_CLASS_INDICES = [0, 1, 8, 9]  # 定义飞机, 汽车, 船, 卡车为正类
    ```

  - LABELED_POSITIVE_NUM: 指定从真实的“正类”样本中，随机抽取多少个作为已标记正样本 (L)。

    ```Python
    # 示例:
    LABELED_POSITIVE_NUM = 2000
    ```

- **训练相关:**

  - BATCH_SIZE: 训练和评估时每个批次的样本数量。

    ```Python
    # 示例:
    BATCH_SIZE = 1024
    ```

  - NUM_EPOCHS: 模型训练的总轮数。

    ```Python
    # 示例:
    NUM_EPOCHS = 30
    ```

  - LEARNING_RATE: 优化器 (Adam) 的初始学习率。

    ```Python
    # 示例:
    LEARNING_RATE = 1e-4
    ```

  - WEIGHT_DECAY: 优化器的权重衰减系数 (L2 正则化)。

    ```Python
    # 示例:
    WEIGHT_DECAY = 1e-4
    ```

- **PU 损失函数与创新点相关:**

  - 这些参数用于配置 `PULoss` 类。

  - GAMMA: 在 PULoss 的自适应权重项中，用于控制 $(1-P(y=1|x))^{\gamma}$ 的指数 $\gamma$。

    ```Python
    # 示例:
    GAMMA = 1.0
    ```

  - ADAPTIVE_EPSILON: 在 PULoss 自适应项中，用于 clamp(min=ADAPTIVE_EPSILON) 的小常数，防止数值不稳定。

    ```Python
    # 示例:
    ADAPTIVE_EPSILON = 1e-6
    ```

- **训练稳定化策略选项:**

  - USE_WARMUP:布尔值，是否启用预热阶段。预热阶段通常使用标准的 nnPU 损失。

    ```Python
    # 示例:
    USE_WARMUP = True
    ```

  - WARMUP_EPOCHS: 如果 USE_WARMUP为 True，此参数定义预热阶段的轮数。

    ```Python
    # 示例:
    WARMUP_EPOCHS = 5
    ```

  - USE_ADAPTIVE_IN_STAGE2: 布尔值，在第二阶段（预热之后或直接开始）是否使用自适应的 nnPU 损失项。

    ```Python
    # 示例:
    USE_ADAPTIVE_IN_STAGE2 = True
    ```

  - ADAPTIVE_BASE_WEIGHT_STAGE2: 如果 USE_ADAPTIVE_IN_STAGE2 为 True，此参数定义自适应损失中 $w_u = \text{base_weight} + (1-\text{base_weight}) \cdot (1-P(y=1|x))^{\gamma}$ 的常数基准权重 (base_weight)。

    ```Python
    # 示例:
    ADAPTIVE_BASE_WEIGHT_STAGE2 = 0.1
    ```

- **早停相关参数:**

  - EARLY_STOPPING_PATIENCE: 定义在多少轮 F1 分数没有提升后就停止训练。

    ```Python
    # 示例:
    EARLY_STOPPING_PATIENCE = 5
    ```

  - `BEST_F1_SCORE`, `BEST_EPOCH`, `epochs_no_improve`, `best_model_state`: 这些是早停逻辑内部使用的变量，通常用户不需要直接修改它们的初始值。

## 📜 脚本概述

- `main.py`:
  - 数据集: CIFAR-10。
  - 模型: ResNet18 (适配二分类)。
- `main2.py`:
  - 数据集: MNIST。
  - 模型: `SimpleMNISTCNN`。
- `main3.py`:
  - 数据集: FashionMNIST。
  - 模型: `SimpleMNISTCNN`。
- `main4.py`:
  - 数据集: EuroSAT。
  - 模型: 当前配置为 `SimpleMNISTCNN` (注意：可能存在通道不匹配问题，详见下一节)。
- `main5.py`:
  - 数据集: PCAM (PatchCamelyon)。
  - 模型: ResNet18 (适配二分类)。

**通用组件 (存在于多个或所有脚本中):**

- `MyCIFAR10`, `MyMNIST`, `MyFashionMNIST`, `MyEuroSAT`, `MyPCAM`: 自定义的数据集类，用于在 PU 学习场景下正确处理和返回真实标签和 PU 标签。
- `load_pu_cifar10`, `load_pu_mnist`, 等: 数据加载函数，负责将原始数据集转换为 PU 学习设置。
- `get_resnet18_for_binary_classification`, `SimpleMNISTCNN`: 模型定义函数/类。
- `PULoss`: PU 学习损失函数的实现。
- `train_model`, `evaluate_model`: 标准的训练和评估辅助函数。

------

## ⚠️ 注意事项与已知问题

- **显存占用:**
  - 脚本中默认的 `BATCH_SIZE` (例如1024) 较大，可能需要较多 GPU 显存。如果遇到显存不足 (OOM) 错误，请尝试减小 `BATCH_SIZE`。

------

## 🔮 未来工作 (可选)

- 集成更多 PU 学习算法。
- 支持更多样的数据集和任务。
- 进行更全面的超参数调优和结果分析。
- 添加更详细的文档和注释。
