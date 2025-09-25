import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import random

# SAM优化器相关导入
try:
    from sam import SAM
    from sam_folder.model.smooth_cross_entropy import smooth_crossentropy
    from sam_folder.utility.bypass_bn import enable_running_stats, disable_running_stats

    SAM_AVAILABLE = True
except ImportError:
    print("SAM模块未找到，将使用普通Adam优化器")
    SAM_AVAILABLE = False


class VibrationDataset(Dataset):
    """1D振动信号数据集类"""

    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path: TSV文件路径 (TRAIN.tsv 或 TEST.tsv)
            transform: 数据变换
        """
        self.data_path = Path(data_path)
        self.transform = transform

        # 读取TSV文件
        df = pd.read_csv(data_path, sep='\t', header=None)

        # 第一列是标签，其余列是振动信号数据
        self.labels = df.iloc[:, 0].values.astype(int)
        self.signals = df.iloc[:, 1:].values.astype(np.float32)

        # 归一化信号（均值为0，方差为1）
        self.signals = (self.signals - self.signals.mean()) / (self.signals.std() + 1e-8)

        print(f"加载数据：{len(self.labels)}个样本，信号长度：{self.signals.shape[1]}")
        print(f"标签范围：{self.labels.min()} - {self.labels.max()}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]

        # 转换为tensor并添加通道维度 [1, signal_length]
        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            signal = self.transform(signal)

        return signal, label


class CNN1D(nn.Module):
    """1D CNN模型用于振动信号分类 - 支持MAML的functional前向传播"""

    def __init__(self, input_length=500, num_classes=10, hidden_size=64):
        super(CNN1D, self).__init__()

        self.input_length = input_length
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # 卷积特征提取层
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.conv2 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)

        self.conv3 = nn.Conv1d(hidden_size * 2, hidden_size * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_size * 4)

        self.conv4 = nn.Conv1d(hidden_size * 4, hidden_size * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_size * 8)

        # 分类头
        self.fc1 = nn.Linear(hidden_size * 8, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, num_classes)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x, params=None):
        """
        前向传播
        Args:
            x: 输入信号 [batch_size, 1, signal_length]
            params: MAML参数字典（可选）
        """
        if params is None:
            # 正常前向传播
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool1d(x, 2)

            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool1d(x, 2)

            x = F.relu(self.bn3(self.conv3(x)))
            x = F.max_pool1d(x, 2)

            x = F.relu(self.bn4(self.conv4(x)))
            x = F.max_pool1d(x, 2)

            # 全局平均池化
            x = F.adaptive_avg_pool1d(x, 1)
            x = x.view(x.size(0), -1)

            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)

            return x
        else:
            # 使用给定参数进行functional前向传播
            return self._forward_with_params(x, params)

    def _forward_with_params(self, x, params):
        """使用给定参数进行functional前向传播"""
        # Conv1 + BN1 + ReLU + MaxPool
        x = F.conv1d(x, params['conv1.weight'], params['conv1.bias'],
                     padding=3)
        x = F.batch_norm(x, self.bn1.running_mean, self.bn1.running_var,
                         params['bn1.weight'], params['bn1.bias'],
                         training=True, momentum=0.1, eps=1e-5)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        # Conv2 + BN2 + ReLU + MaxPool
        x = F.conv1d(x, params['conv2.weight'], params['conv2.bias'],
                     padding=2)
        x = F.batch_norm(x, self.bn2.running_mean, self.bn2.running_var,
                         params['bn2.weight'], params['bn2.bias'],
                         training=True, momentum=0.1, eps=1e-5)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        # Conv3 + BN3 + ReLU + MaxPool
        x = F.conv1d(x, params['conv3.weight'], params['conv3.bias'],
                     padding=1)
        x = F.batch_norm(x, self.bn3.running_mean, self.bn3.running_var,
                         params['bn3.weight'], params['bn3.bias'],
                         training=True, momentum=0.1, eps=1e-5)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        # Conv4 + BN4 + ReLU + MaxPool
        x = F.conv1d(x, params['conv4.weight'], params['conv4.bias'],
                     padding=1)
        x = F.batch_norm(x, self.bn4.running_mean, self.bn4.running_var,
                         params['bn4.weight'], params['bn4.bias'],
                         training=True, momentum=0.1, eps=1e-5)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        # 全局平均池化
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.linear(x, params['fc1.weight'], params['fc1.bias'])
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=True)

        x = F.linear(x, params['fc2.weight'], params['fc2.bias'])
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=True)

        x = F.linear(x, params['fc3.weight'], params['fc3.bias'])

        return x


class MetaLearningDataset(Dataset):
    """元学习数据集包装器"""

    def __init__(self, dataset, args):
        self.dataset = dataset
        self.num_ways = args.num_ways
        self.num_shots = args.num_shots
        self.num_query = args.num_shots_test  # 查询集大小

        # 按类别组织数据
        self.class_indices = {}
        for idx, (_, label) in enumerate(dataset):
            label_int = label.item()
            if label_int not in self.class_indices:
                self.class_indices[label_int] = []
            self.class_indices[label_int].append(idx)

        self.classes = list(self.class_indices.keys())
        print(f"元学习数据集：{len(self.classes)}个类别，每个任务{self.num_ways}-way {self.num_shots}-shot")

    def __len__(self):
        return len(self.dataset) // (self.num_ways * (self.num_shots + self.num_query))

    def __getitem__(self, idx):
        """生成一个N-way K-shot任务"""
        # 随机选择N个类别
        selected_classes = random.sample(self.classes, self.num_ways)

        support_signals, support_labels = [], []
        query_signals, query_labels = [], []

        for new_label, class_label in enumerate(selected_classes):
            # 从该类别中随机选择样本
            available_samples = self.class_indices[class_label]
            if len(available_samples) < self.num_shots + self.num_query:
                # 如果样本不够，进行有放回采样
                class_samples = random.choices(available_samples,
                                               k=self.num_shots + self.num_query)
            else:
                class_samples = random.sample(available_samples,
                                              self.num_shots + self.num_query)

            # 前num_shots个作为支持集
            support_indices = class_samples[:self.num_shots]
            # 后num_query个作为查询集
            query_indices = class_samples[self.num_shots:]

            for idx in support_indices:
                signal, _ = self.dataset[idx]
                support_signals.append(signal)
                support_labels.append(new_label)

            for idx in query_indices:
                signal, _ = self.dataset[idx]
                query_signals.append(signal)
                query_labels.append(new_label)

        support_signals = torch.stack(support_signals)
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        query_signals = torch.stack(query_signals)
        query_labels = torch.tensor(query_labels, dtype=torch.long)

        return {
            'train': (support_signals, support_labels),
            'test': (query_signals, query_labels)
        }


class MAML:
    """MAML元学习算法 - 修正版本，添加梯度裁剪"""

    def __init__(self, model, args, device='cpu'):
        self.model = model.to(device)
        self.inner_lr = args.step_size  # 内循环学习率
        self.meta_lr = args.meta_lr  # 元学习率
        self.inner_steps = args.num_steps  # 内循环步数
        self.device = device
        self.args = args
        self.grad_clip_norm = args.grad_clip_norm  # 梯度裁剪阈值

        # 获取模型参数名称列表，用于创建fast_weights
        self.param_names = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_names.append(name)

        # 元优化器 - Adam或SAM
        if SAM_AVAILABLE and args.alpha > 0:
            base_optimizer = torch.optim.Adam
            self.meta_optimizer = SAM(self.model.parameters(),
                                      base_optimizer,
                                      rho=args.alpha,
                                      adaptive=args.adap,
                                      lr=args.meta_lr)
            print(f"使用SAM优化器: alpha={args.alpha}, adaptive={args.adap}")
            self.use_sam = True
        else:
            self.meta_optimizer = torch.optim.Adam(self.model.parameters(),
                                                   lr=args.meta_lr,
                                                   betas=(args.beta1, args.beta2))
            print(f"使用Adam优化器: lr={args.meta_lr}, beta1={args.beta1}, beta2={args.beta2}")
            self.use_sam = False

        print(f"梯度裁剪阈值: {self.grad_clip_norm}")

    def get_fast_weights(self):
        """获取当前模型参数作为fast_weights"""
        fast_weights = OrderedDict()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fast_weights[name] = param.clone()
        return fast_weights

    def inner_update(self, support_x, support_y, fast_weights=None):
        """内循环更新"""
        if fast_weights is None:
            fast_weights = self.get_fast_weights()

        for step in range(self.inner_steps):
            # 前向传播，使用当前fast_weights
            logits = self.model(support_x, params=fast_weights)
            loss = F.cross_entropy(logits, support_y)

            # 计算梯度
            grads = torch.autograd.grad(loss, fast_weights.values(),
                                        create_graph=True,
                                        allow_unused=True)

            # 更新快速权重
            new_fast_weights = OrderedDict()
            for (name, param), grad in zip(fast_weights.items(), grads):
                if grad is not None:
                    new_fast_weights[name] = param - self.inner_lr * grad
                else:
                    new_fast_weights[name] = param.clone()
            fast_weights = new_fast_weights

        return fast_weights

    def meta_update(self, batch):
        """元更新"""
        if self.use_sam:
            return self._meta_update_sam(batch)
        else:
            return self._meta_update_normal(batch)

    def _meta_update_normal(self, batch):
        """普通Adam优化器的元更新 - 添加梯度裁剪"""
        meta_losses = []
        meta_accs = []

        self.meta_optimizer.zero_grad()

        total_loss = 0.0
        for task in batch:
            support_x, support_y = task['train']
            query_x, query_y = task['test']

            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)

            # 内循环适应
            fast_weights = self.inner_update(support_x, support_y)

            # 使用fast_weights进行查询集前向传播
            query_logits = self.model(query_x, params=fast_weights)
            loss = F.cross_entropy(query_logits, query_y)
            total_loss += loss

            # 计算准确率
            pred = query_logits.argmax(dim=1)
            acc = (pred == query_y).float().mean()
            meta_accs.append(acc.item())

        # 平均损失
        meta_loss = total_loss / len(batch)
        meta_acc = np.mean(meta_accs)

        # 反向传播
        meta_loss.backward()

        # 添加梯度裁剪
        if self.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.grad_clip_norm)

        self.meta_optimizer.step()

        return meta_loss.item(), meta_acc

    def _meta_update_sam(self, batch):
        """SAM优化器的元更新 - 添加梯度裁剪"""

        def closure():
            """SAM需要的closure函数"""
            self.meta_optimizer.zero_grad()
            total_loss = 0.0
            for task in batch:
                support_x, support_y = task['train']
                query_x, query_y = task['test']

                support_x = support_x.to(self.device)
                support_y = support_y.to(self.device)
                query_x = query_x.to(self.device)
                query_y = query_y.to(self.device)

                fast_weights = self.inner_update(support_x, support_y)
                query_logits = self.model(query_x, params=fast_weights)
                loss = F.cross_entropy(query_logits, query_y)
                total_loss += loss

            meta_loss = total_loss / len(batch)
            meta_loss.backward()

            # SAM中也添加梯度裁剪
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.grad_clip_norm)
            return meta_loss

        # 第一次前向传播计算损失和准确率
        meta_accs = []
        with torch.enable_grad():
            total_loss = 0.0
            for task in batch:
                support_x, support_y = task['train']
                query_x, query_y = task['test']

                support_x = support_x.to(self.device)
                support_y = support_y.to(self.device)
                query_x = query_x.to(self.device)
                query_y = query_y.to(self.device)

                fast_weights = self.inner_update(support_x, support_y)
                query_logits = self.model(query_x, params=fast_weights)
                loss = F.cross_entropy(query_logits, query_y)
                total_loss += loss

                pred = query_logits.argmax(dim=1)
                acc = (pred == query_y).float().mean()
                meta_accs.append(acc.item())

            meta_loss = total_loss / len(batch)
            meta_acc = np.mean(meta_accs)

        # SAM优化
        self.meta_optimizer.zero_grad()
        meta_loss.backward()

        # 添加梯度裁剪
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.grad_clip_norm)

        self.meta_optimizer.step(closure)

        return meta_loss.item(), meta_acc


class EarlyStopping:
    """早停策略类"""

    def __init__(self, patience=20, min_delta=0.0001, restore_best_weights=True):
        """
        Args:
            patience: 耐心值，多少个epoch没有改善后停止
            min_delta: 最小改善阈值
            restore_best_weights: 是否在早停时恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_score, model):
        """
        检查是否应该早停
        Args:
            val_score: 当前验证分数（越高越好）
            model: 模型对象
        Returns:
            bool: 是否应该早停
        """
        if self.best_score is None:
            self.best_score = val_score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_score = val_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()

        return self.early_stop


# 统一参数配置类
class input_data:
    def __init__(self):
        # ==================== 数据配置 ====================
        self.data_dir = 'data/cwru_dataset/CWRU_10'  # 数据根目录
        self.train_file = 'TRAIN.tsv'  # 训练集文件名
        self.test_file = 'TEST.tsv'  # 测试集文件名
        self.output_folder = 'results'  # 结果保存目录

        # ==================== 数据集参数 ====================
        self.input_length = 500  # 振动信号长度
        self.num_classes = 10  # 总类别数

        # ==================== 元学习配置 ====================
        self.num_ways = 5  # N-way: 每个任务的类别数
        self.num_shots = 1  # K-shot: 每个类别的支持样本数
        self.num_shots_test = 10  # 每个类别的查询样本数

        # ==================== 模型参数 ====================
        self.hidden_size = 64  # 隐藏层维度

        # ==================== MAML训练参数 ====================
        self.num_steps = 5  # 内循环适应步数
        self.step_size = 0.02  # 内循环学习率
        self.meta_lr = 0.0002  # 元学习率
        self.first_order = True  # 是否使用一阶近似

        # ==================== 训练配置 ====================
        self.batch_size = 4  # 每个批次的任务数
        self.num_epochs = 30  # 训练轮数
        self.num_batches = 100  # 每轮的批次数

        # ==================== SAM优化器参数 ====================
        self.alpha = 0.01  # SAM扰动半径
        self.adap = True  # 自适应SAM
        self.SAM_lower = True  # SAM应用位置
        self.m = 0.9  # 动量参数
        self.delta = 0.000  # 扰动参数

        # ==================== Adam优化器参数 ====================
        self.beta1 = 0.9  # Adam beta1参数
        self.beta2 = 0.99  # Adam beta2参数
        self.isMomentum = True  # 是否使用动量

        # ==================== 早停策略参数 ====================
        self.early_stopping = True  # 是否启用早停
        self.patience = 25  # 早停耐心值
        self.min_delta = 0.001  # 最小改善阈值
        self.restore_best_weights = True  # 早停时是否恢复最佳权重

        # ==================== 梯度裁剪参数 ====================
        self.grad_clip_norm = 1.0  # 梯度裁剪阈值（0表示不使用梯度裁剪）

        # ==================== 系统配置 ====================
        self.num_workers = 0  # Windows下设为0避免多进程问题
        self.verbose = True  # 详细输出
        self.use_cuda = torch.cuda.is_available()  # 是否使用GPU

        # ==================== 实验记录 ====================
        self.save_model = True  # 是否保存模型
        self.log_interval = 5  # 日志输出间隔（更频繁）


# 自定义collate函数
def meta_collate_fn(batch):
    """元学习数据集的collate函数"""
    return batch


def main():
    # 初始化统一参数配置
    args = input_data()

    # 设置设备
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 设置日志
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    logger = logging.getLogger(__name__)

    # 创建输出文件夹
    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
        logger.info(f'创建输出文件夹: {output_folder}')

    # 创建时间戳文件夹
    timestamp = time.strftime('%Y-%m-%d_%H%M%S')
    exp_folder = output_folder / timestamp
    exp_folder.mkdir()
    logger.info(f'创建实验文件夹: {exp_folder}')

    # 保存配置文件
    config_path = exp_folder / 'config.json'
    with open(config_path, 'w') as f:
        config_dict = {k: v for k, v in args.__dict__.items()
                       if not k.startswith('_')}
        json.dump(config_dict, f, indent=2, default=str)
    logger.info(f'配置文件保存至: {config_path}')

    # 检查数据文件
    train_path = Path(args.data_dir) / args.train_file
    test_path = Path(args.data_dir) / args.test_file

    if not train_path.exists():
        raise FileNotFoundError(f"训练集文件不存在: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"测试集文件不存在: {test_path}")

    # 打印关键参数
    print(f'\n==================== 实验配置 ====================')
    print(f'数据集: {args.data_dir}')
    print(f'元学习设置: {args.num_ways}-way {args.num_shots}-shot')
    print(f'模型: 隐藏维度={args.hidden_size}')
    print(f'MAML: 内循环步数={args.num_steps}, 内循环学习率={args.step_size}')
    print(f'训练: 批次大小={args.batch_size}, 轮数={args.num_epochs}')
    print(f'优化器: 元学习率={args.meta_lr}, SAM_alpha={args.alpha}')
    print(f'早停: 启用={args.early_stopping}, 耐心值={args.patience}')
    print(f'梯度裁剪: 阈值={args.grad_clip_norm}')
    print(f'设备: {device}')
    print(f'================================================\n')

    # 加载数据
    logger.info("加载数据...")
    train_dataset = VibrationDataset(train_path)
    test_dataset = VibrationDataset(test_path)

    # 创建元学习数据集
    meta_train_dataset = MetaLearningDataset(train_dataset, args)
    meta_test_dataset = MetaLearningDataset(test_dataset, args)

    # 创建数据加载器
    train_loader = DataLoader(meta_train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=meta_collate_fn)
    test_loader = DataLoader(meta_test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             collate_fn=meta_collate_fn)

    # 初始化模型
    logger.info("初始化模型...")
    model = CNN1D(
        input_length=args.input_length,
        num_classes=args.num_ways,
        hidden_size=args.hidden_size
    )

    # 初始化MAML
    maml = MAML(model=model, args=args, device=device)

    # 初始化早停策略
    if args.early_stopping:
        early_stopping = EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            restore_best_weights=args.restore_best_weights
        )
        logger.info(f"早停策略已启用: patience={args.patience}, min_delta={args.min_delta}")

    # 训练循环
    logger.info("开始训练...")
    best_acc = 0.0
    model_path = exp_folder / 'best_model.pth'

    # 创建训练日志
    train_log = []

    for epoch in range(args.num_epochs):
        # 训练阶段
        maml.model.train()
        train_losses = []
        train_accs = []

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= args.num_batches:
                break

            try:
                loss, acc = maml.meta_update(batch)
                train_losses.append(loss)
                train_accs.append(acc)
            except Exception as e:
                logger.warning(f"训练批次 {batch_idx} 失败: {e}")
                continue

        if not train_losses:
            logger.warning(f"Epoch {epoch + 1}: 没有成功的训练批次")
            continue

        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs)

        # 验证阶段
        maml.model.eval()
        val_losses = []
        val_accs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= max(1, args.num_batches // 4):
                    break

                try:
                    meta_loss = 0.0
                    meta_acc = 0.0

                    for task in batch:
                        support_x, support_y = task['train']
                        query_x, query_y = task['test']

                        support_x = support_x.to(device)
                        support_y = support_y.to(device)
                        query_x = query_x.to(device)
                        query_y = query_y.to(device)

                        # 验证时不计算梯度，使用模型当前参数作为初始化
                        with torch.enable_grad():
                            fast_weights = maml.inner_update(support_x, support_y)

                        with torch.no_grad():
                            query_logits = maml.model(query_x, params=fast_weights)
                            loss = F.cross_entropy(query_logits, query_y)
                            meta_loss += loss.item()

                            pred = query_logits.argmax(dim=1)
                            acc = (pred == query_y).float().mean().item()
                            meta_acc += acc

                    val_losses.append(meta_loss / len(batch))
                    val_accs.append(meta_acc / len(batch))
                except Exception as e:
                    logger.warning(f"验证批次 {batch_idx} 失败: {e}")
                    continue

        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        avg_val_acc = np.mean(val_accs) if val_accs else 0.0

        # 记录日志
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': avg_train_acc,
            'val_loss': avg_val_loss,
            'val_acc': avg_val_acc
        }
        train_log.append(log_entry)

        if args.verbose and (epoch + 1) % args.log_interval == 0:
            logger.info(f"Epoch {epoch + 1:3d}/{args.num_epochs}")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
            logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        # 保存最佳模型
        if avg_val_acc > best_acc and args.save_model:
            best_acc = avg_val_acc
            torch.save({
                'model_state_dict': maml.model.state_dict(),
                'epoch': epoch + 1,
                'best_acc': best_acc,
                'args': args.__dict__
            }, model_path)
            if args.verbose:
                logger.info(f"  → 保存最佳模型，验证准确率: {best_acc:.4f}")

        # 早停检查
        if args.early_stopping:
            is_early_stop = early_stopping(avg_val_acc, maml.model)
            if is_early_stop:
                logger.info(f"触发早停，在第 {epoch + 1} 轮停止训练")
                logger.info(f"最佳验证准确率: {early_stopping.best_score:.4f}")
                # 如果启用了恢复最佳权重，更新best_acc
                if args.restore_best_weights:
                    best_acc = early_stopping.best_score
                break

    # 保存训练日志
    log_path = exp_folder / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(train_log, f, indent=2)

    # 保存早停信息
    if args.early_stopping:
        early_stop_info = {
            'early_stopped': early_stopping.early_stop,
            'best_score': early_stopping.best_score,
            'final_patience_counter': early_stopping.counter,
            'total_epochs_trained': len(train_log)
        }
        early_stop_path = exp_folder / 'early_stopping_info.json'
        with open(early_stop_path, 'w') as f:
            json.dump(early_stop_info, f, indent=2)
        logger.info(f"早停信息保存至: {early_stop_path}")

    logger.info(f"训练完成！最佳验证准确率: {best_acc:.4f}")
    logger.info(f"模型保存至: {model_path}")
    logger.info(f"训练日志保存至: {log_path}")
    logger.info(f"实验结果保存至: {exp_folder}")


if __name__ == "__main__":
    main()