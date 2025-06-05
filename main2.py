import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
from typing import Optional, Callable, List, Tuple
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import copy  # 用于深度复制模型状态字典

def get_resnet18_for_binary_classification(gray_input=True):
    model = torchvision.models.resnet18(pretrained=False)
    if gray_input:
        # 替换第一层卷积：输入通道从3改为1
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)  # 二分类
    return model


# ==============================================================================
# 1. 模型定义 (假设已存在)
# ==============================================================================
class SimpleMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)



# ==============================================================================
# 2. 数据处理 (假设已存在)
# ==============================================================================
class MyCIFAR10(torchvision.datasets.CIFAR10):
    """ 继承自CIFAR10数据集类，增加了存储和返回真实二元标签的功能。"""

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.true_binary_labels: Optional[List[int]] = None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int]:
        data, pu_or_true_label = super().__getitem__(idx)
        true_binary_label = self.true_binary_labels[idx] if self.true_binary_labels is not None else -1
        return data, pu_or_true_label, true_binary_label, idx


class MyMNIST(torchvision.datasets.MNIST):
    def __init__(self, root: str, train: bool = True, transform=None,
                 target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.true_binary_labels = None

    def __getitem__(self, idx):
        data, pu_or_true_label = super().__getitem__(idx)
        true_binary_label = self.true_binary_labels[idx] if self.true_binary_labels else -1
        return data, pu_or_true_label, true_binary_label, idx


def load_pu_cifar10(root: str, pos_class_indices: List[int], labeled_positive_num: int,
                    batch_size: int) -> Tuple[DataLoader, DataLoader, float]:
    """ 加载CIFAR-10数据集，并将其转换为PU学习设置。"""
    common_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), common_normalize
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), common_normalize])

    dataset_train = MyCIFAR10(root=root, train=True, download=True, transform=transform_train)
    true_binary_labels_train = [1 if t in pos_class_indices else 0 for t in dataset_train.targets]
    dataset_train.true_binary_labels = true_binary_labels_train
    prior = sum(true_binary_labels_train) / len(true_binary_labels_train)
    print(f"正类类别 (原始CIFAR-10索引): {pos_class_indices}")
    print(f"根据训练数据估算的类别先验 (pi_p): {prior:.4f}")
    pos_indices_true_train = [i for i, val in enumerate(true_binary_labels_train) if val == 1]
    if labeled_positive_num > len(pos_indices_true_train):
        print(f"警告: 请求标记 {labeled_positive_num} 个正样本, 但仅存在 {len(pos_indices_true_train)} 个。")
        labeled_positive_num = len(pos_indices_true_train)
    labeled_pos_indices = random.sample(pos_indices_true_train, labeled_positive_num)
    pu_target_train = [1 if (i in labeled_pos_indices) else 0 for i in range(len(true_binary_labels_train))]
    dataset_train.targets = pu_target_train
    print(f"训练集: 总样本数: {len(dataset_train)}")
    print(f"  真实正样本 (P_true): {sum(true_binary_labels_train)}")
    print(f"  已标记正样本 (L): {sum(pu_target_train)}")
    unlabeled_count = len(pu_target_train) - sum(pu_target_train)
    print(f"  未标记样本 (U): {unlabeled_count}")
    hidden_pos_in_U = sum(true_binary_labels_train) - sum(pu_target_train)
    true_neg_in_U = unlabeled_count - hidden_pos_in_U
    print(f"    其中U包含: {hidden_pos_in_U} 个隐藏正样本, {true_neg_in_U} 个真实负样本。")
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    dataset_test = MyCIFAR10(root=root, train=False, download=True, transform=transform_test)
    true_binary_labels_test = [1 if t in pos_class_indices else 0 for t in dataset_test.targets]
    dataset_test.true_binary_labels = true_binary_labels_test
    dataset_test.targets = true_binary_labels_test
    test_loader = DataLoader(dataset_test, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader, prior


def load_pu_mnist(root, pos_class_indices, labeled_positive_num, batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    dataset_train = MyMNIST(root=root, train=True, download=True, transform=transform)
    true_binary_labels = [1 if t in pos_class_indices else 0 for t in dataset_train.targets]
    dataset_train.true_binary_labels = true_binary_labels
    prior = sum(true_binary_labels) / len(true_binary_labels)

    pos_indices = [i for i, val in enumerate(true_binary_labels) if val == 1]
    labeled_pos_indices = random.sample(pos_indices, min(labeled_positive_num, len(pos_indices)))
    pu_labels = [1 if i in labeled_pos_indices else 0 for i in range(len(true_binary_labels))]
    dataset_train.targets = pu_labels

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

    dataset_test = MyMNIST(root=root, train=False, download=True, transform=transform)
    test_true_binary = [1 if t in pos_class_indices else 0 for t in dataset_test.targets]
    dataset_test.true_binary_labels = test_true_binary
    dataset_test.targets = test_true_binary

    test_loader = DataLoader(dataset_test, batch_size=batch_size * 2, shuffle=False, num_workers=4)

    return train_loader, test_loader, prior


# ==============================================================================
# 3. PU 损失函数 (假设已存在)
# ==============================================================================
class PULoss(nn.Module):
    """ 实现 PU 学习损失函数，基于 nnPU 框架。 (包含之前的修改) """

    def __init__(self, prior: float, gamma: float = 1.0, loss_fn: nn.Module = nn.BCEWithLogitsLoss(reduction='none'),
                 use_adaptive_term: bool = False, adaptive_epsilon: float = 1e-6, adaptive_base_weight: float = 0.0):
        super().__init__()
        if not (0.0 <= prior <= 1.0): raise ValueError("Prior probability must be between 0 and 1.")
        if adaptive_base_weight < 0.0 or adaptive_base_weight >= 1.0: raise ValueError(
            "Adaptive base weight must be between 0.0 and 1.0 (exclusive of 1.0).")
        self.prior, self.gamma, self.loss_fn = prior, gamma, loss_fn
        self.positive_label, self.negative_label = 1.0, 0.0
        self.use_adaptive_term, self.adaptive_epsilon, self.adaptive_base_weight = use_adaptive_term, adaptive_epsilon, adaptive_base_weight
        loss_type = f"自适应(beta) nnPU (创新点尝试, base_weight={self.adaptive_base_weight:.2f})" if self.use_adaptive_term else "标准 nnPU"
        print(
            f"PULoss 初始化为 *{loss_type}*, 先验={self.prior:.4f}, gamma={self.gamma:.2f}, epsilon={self.adaptive_epsilon:.1e}")

    def forward(self, logits: torch.Tensor, pu_labels: torch.Tensor) -> torch.Tensor:
        device = logits.device
        labeled_mask, unlabeled_mask = (pu_labels == self.positive_label), (pu_labels == self.negative_label)
        p_logits, u_logits = logits[labeled_mask], logits[unlabeled_mask]
        num_p, num_u = p_logits.size(0), u_logits.size(0)
        if num_p == 0 and num_u == 0: return torch.tensor(0.0, device=device, requires_grad=True)
        risk_p_plus, risk_p_minus = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        if num_p > 0:
            risk_p_plus = self.loss_fn(p_logits, torch.full_like(p_logits, self.positive_label)).mean()
            risk_p_minus = self.loss_fn(p_logits, torch.full_like(p_logits, self.negative_label)).mean()
        if num_u == 0: return self.prior * risk_p_plus if num_p > 0 else torch.tensor(0.0, device=device,
                                                                                      requires_grad=True)
        risk_u_as_neg_elements = self.loss_fn(u_logits, torch.full_like(u_logits, self.negative_label))
        if self.use_adaptive_term:
            with torch.no_grad():
                u_probs = torch.sigmoid(u_logits)
                adaptive_part = ((1.0 - u_probs).clamp(min=self.adaptive_epsilon)).pow(self.gamma)
                weights_u = self.adaptive_base_weight + (1.0 - self.adaptive_base_weight) * adaptive_part
            risk_u_component = (weights_u * risk_u_as_neg_elements).mean()
        else:
            risk_u_component = risk_u_as_neg_elements.mean()
        neg_risk = risk_u_component - self.prior * risk_p_minus
        final_loss = self.prior * risk_p_plus + torch.max(torch.tensor(0.0, device=device), neg_risk)
        return final_loss


# ==============================================================================
# 4. 训练与评估工具函数 (假设已存在)
# ==============================================================================
def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer,
                device: torch.device, current_epoch: int, total_epochs: int) -> None:
    """ 执行一个轮次的模型训练。 """
    model.train()
    running_loss, processed_samples = 0.0, 0
    for i, (inputs, pu_targets, _, _) in enumerate(train_loader):
        inputs, pu_targets = inputs.to(device), pu_targets.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, pu_targets)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"轮次 {current_epoch}, 批次 {i + 1}: 检测到 NaN 或 Inf 损失。跳过更新。")
            continue
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        processed_samples += inputs.size(0)
    epoch_loss = running_loss / processed_samples if processed_samples > 0 else 0
    print(f"轮次 [{current_epoch}/{total_epochs}] 训练完成。平均训练损失: {epoch_loss:.4f}.")


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    """ 在测试集上评估模型性能。 """
    model.eval()
    all_preds_probs, all_true_labels = [], []
    with torch.no_grad():
        for inputs, true_targets_batch, _, _ in test_loader:
            inputs, true_targets_batch = inputs.to(device), true_targets_batch.to(device).float()
            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs)
            all_preds_probs.extend(probs.cpu().numpy())
            all_true_labels.extend(true_targets_batch.cpu().numpy())
    all_preds_probs, all_true_labels = np.array(all_preds_probs), np.array(all_true_labels)
    predicted_labels = (all_preds_probs > 0.5).astype(int)
    accuracy = accuracy_score(all_true_labels, predicted_labels)
    auc = -1.0
    if len(np.unique(all_true_labels)) > 1:
        try:
            auc = roc_auc_score(all_true_labels, all_preds_probs)
        except ValueError as e:
            print(f"计算 AUC 时出错: {e}")
    else:
        print("警告: 测试集中仅存在单一类别标签, AUC 无法计算。")
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, predicted_labels, average='binary',
                                                               pos_label=1, zero_division=0)
    print(f"测试集评估结果:")
    print(f"  准确率 (Accuracy): {accuracy * 100:.2f}%")
    if auc != -1.0: print(f"  AUC: {auc:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    print(f"  F1分数 (F1-score): {f1:.4f}")
    return accuracy, auc, f1


# ==============================================================================
# 5. 主执行模块 (包含早停逻辑)
# ==============================================================================
if __name__ == '__main__':

    # --- 1. 配置与超参数 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集相关
    DATA_ROOT = 'data'
    POS_CLASS_INDICES = [0, 2, 4, 6, 8]  # 定义正类 (飞机, 汽车, 船, 卡车)
    LABELED_POSITIVE_NUM = 2000  # 已标记正样本 (L) 的数量

    # 训练相关
    BATCH_SIZE = 1024  # 批处理大小 (注意: 1024 较大)
    NUM_EPOCHS = 30  # 总训练轮次
    LEARNING_RATE = 1e-4  # 初始学习率 (调整为 1e-4)
    WEIGHT_DECAY = 1e-4  # 权重衰减

    # 损失函数与创新点相关
    GAMMA = 1.0  # 自适应权重中的 gamma 指数
    ADAPTIVE_EPSILON = 1e-6  # 自适应权重中的 epsilon

    # 稳定化策略选项
    USE_WARMUP = True  # 是否启用预热阶段
    WARMUP_EPOCHS = 5  # 预热阶段轮数 (如果 USE_WARMUP=True)
    USE_ADAPTIVE_IN_STAGE2 = True  # 第二阶段是否使用自适应损失 (创新点)
    ADAPTIVE_BASE_WEIGHT_STAGE2 = 0.1  # 第二阶段自适应损失的常数下界 (如果 USE_ADAPTIVE_IN_STAGE2=True)

    # 早停相关参数
    EARLY_STOPPING_PATIENCE = 5  # 早停的耐心值 (连续 N 轮 F1 无提升则停止)
    BEST_F1_SCORE = 0.0  # 记录最佳 F1 分数
    BEST_EPOCH = 0  # 记录最佳 F1 分数对应的轮次
    epochs_no_improve = 0  # 记录 F1 分数连续未提升的轮次数
    best_model_state = None  # 记录最佳模型的状态字典

    # 打印配置信息
    print(f"--- 配置信息 ---")
    print(f"使用设备: {DEVICE}")
    print(f"正类类别: {POS_CLASS_INDICES}")
    print(f"标记正样本数: {LABELED_POSITIVE_NUM}")
    print(f"批处理大小: {BATCH_SIZE}")
    print(f"总轮次: {NUM_EPOCHS}")
    print(f"学习率: {LEARNING_RATE}, 权重衰减: {WEIGHT_DECAY}")
    print(f"使用预热: {USE_WARMUP}, 预热轮数: {WARMUP_EPOCHS if USE_WARMUP else 'N/A'}")
    print(f"阶段二使用自适应损失: {USE_ADAPTIVE_IN_STAGE2}")
    if USE_ADAPTIVE_IN_STAGE2:
        print(
            f"阶段二自适应损失参数: gamma={GAMMA}, epsilon={ADAPTIVE_EPSILON}, base_weight={ADAPTIVE_BASE_WEIGHT_STAGE2}")
    print(f"使用早停: 是, 耐心值: {EARLY_STOPPING_PATIENCE}")
    print(f"----------------")

    # --- 2. 设置 (数据, 模型, 优化器, 调度器) ---
    train_loader, test_loader, class_prior = load_pu_mnist(
        root=DATA_ROOT,
        pos_class_indices=POS_CLASS_INDICES,
        labeled_positive_num=LABELED_POSITIVE_NUM,
        batch_size=BATCH_SIZE
    )

    # pu_model = get_resnet18_for_binary_classification().to(DEVICE)
    pu_model = SimpleMNISTCNN().to(DEVICE)
    optimizer = optim.Adam(pu_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # --- 3. 训练过程 (分阶段) ---
    start_epoch = 1
    current_criterion = None

    # --- 阶段一：预热 (如果启用) ---
    if USE_WARMUP and WARMUP_EPOCHS > 0:
        print(f"\n--- 阶段一：标准 nnPU 预热 ({WARMUP_EPOCHS}轮) ---")
        # 预热阶段强制使用标准 nnPU
        current_criterion = PULoss(prior=class_prior, use_adaptive_term=False).to(DEVICE)
        for epoch in range(1, WARMUP_EPOCHS + 1):
            print(f"\n开始预热轮次 [{epoch}/{WARMUP_EPOCHS}] (总轮次 {epoch}/{NUM_EPOCHS})")
            train_model(pu_model, train_loader, current_criterion, optimizer, DEVICE, current_epoch=epoch,
                        total_epochs=NUM_EPOCHS)
            # 评估并获取 F1 分数用于早停判断
            accuracy, auc, f1 = evaluate_model(pu_model, test_loader, DEVICE)

            # 早停逻辑：在预热阶段也跟踪最佳模型
            if f1 > BEST_F1_SCORE:
                BEST_F1_SCORE = f1
                BEST_EPOCH = epoch
                # 使用 copy.deepcopy 确保保存的是当前状态的独立副本
                best_model_state = copy.deepcopy(pu_model.state_dict())
                epochs_no_improve = 0
                print(f"*** 新的最佳 F1 分数: {BEST_F1_SCORE:.4f} 在轮次 {BEST_EPOCH} ***")
            else:
                epochs_no_improve += 1
                print(f"连续 {epochs_no_improve} 轮 F1 未提升。")

            scheduler.step()  # 在每个轮次后更新学习率
            print(f"预热轮次 [{epoch}/{WARMUP_EPOCHS}] 完成。当前学习率: {scheduler.get_last_lr()[0]:.1e}")

            # 注意：通常不在预热阶段因为“未提升”而提前中断，但可以跟踪最佳点
            # if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            #     print(f"早停耐心耗尽于预热阶段轮次 {epoch}。")
            #     # break # 如果需要，也可以在这里中断

        start_epoch = WARMUP_EPOCHS + 1  # 更新下一阶段的起始轮次
        # 进入第二阶段前，重置“未提升轮次”计数器
        epochs_no_improve = 0
        print(f"预热完成。最佳 F1={BEST_F1_SCORE:.4f} 在轮次 {BEST_EPOCH}。")

    else:
        print("\n--- 跳过预热阶段 ---")
        start_epoch = 1

    # --- 阶段二：目标损失函数训练 ---
    print(f"\n--- 阶段二：目标损失函数训练 (轮次 {start_epoch} 到 {NUM_EPOCHS}) ---")
    # 根据配置创建第二阶段的损失函数
    current_criterion = PULoss(prior=class_prior,
                               use_adaptive_term=USE_ADAPTIVE_IN_STAGE2,
                               adaptive_base_weight=ADAPTIVE_BASE_WEIGHT_STAGE2 if USE_ADAPTIVE_IN_STAGE2 else 0.0,
                               gamma=GAMMA,
                               adaptive_epsilon=ADAPTIVE_EPSILON).to(DEVICE)

    # 进行剩余轮次的训练
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n开始训练轮次 [{epoch}/{NUM_EPOCHS}]")
        train_model(pu_model, train_loader, current_criterion, optimizer, DEVICE, current_epoch=epoch,
                    total_epochs=NUM_EPOCHS)
        # 评估并获取 F1 分数
        accuracy, auc, f1 = evaluate_model(pu_model, test_loader, DEVICE)

        # --- 早停逻辑 ---
        if f1 > BEST_F1_SCORE:
            BEST_F1_SCORE = f1
            BEST_EPOCH = epoch
            best_model_state = copy.deepcopy(pu_model.state_dict())  # 保存更好的模型状态
            epochs_no_improve = 0
            print(f"*** 新的最佳 F1 分数: {BEST_F1_SCORE:.4f} 在轮次 {BEST_EPOCH} ***")
        else:
            epochs_no_improve += 1
            print(f"连续 {epochs_no_improve} 轮 F1 未提升。")

        scheduler.step()  # 每个轮次后都应该调用
        print(f"轮次 [{epoch}/{NUM_EPOCHS}] 完成。当前学习率: {scheduler.get_last_lr()[0]:.1e}")

        # 检查是否满足早停条件
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(
                f"\n早停触发: 连续 {EARLY_STOPPING_PATIENCE} 轮性能未提升。最佳 F1={BEST_F1_SCORE:.4f} 在轮次 {BEST_EPOCH}")
            break  # 跳出训练循环

    # --- 4. 最终评估 ---
    print("\n--- 所有轮次训练完成或早停后的最终评估 ---")
    print("--- 评估最后一个轮次得到的模型 ---")
    evaluate_model(pu_model, test_loader, DEVICE)

    # 如果通过早停找到了最佳模型，加载并评估它
    if best_model_state is not None:
        print(f"\n--- 加载并评估在轮次 {BEST_EPOCH} 获得的最佳模型 (F1={BEST_F1_SCORE:.4f}) ---")
        try:
            pu_model.load_state_dict(best_model_state)
            evaluate_model(pu_model, test_loader, DEVICE)
        except Exception as e:
            print(f"加载最佳模型状态时出错: {e}")
    else:
        print("\n--- 未记录到有效的最佳模型状态 (可能从未提升或未启用早停/记录) ---")

    # --- 5. (可选) 保存最佳模型 ---
    # if best_model_state is not None:
    #     try:
    #         model_save_path = f"best_pu_model_epoch{BEST_EPOCH}_f1_{BEST_F1_SCORE:.4f}.pth"
    #         torch.save(best_model_state, model_save_path)
    #         print(f"\n最佳模型状态已保存到: {model_save_path}")
    #     except Exception as e:
    #         print(f"保存最佳模型时出错: {e}")