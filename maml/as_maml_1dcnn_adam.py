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
import copy


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
        signal_mean = self.signals.mean(axis=1, keepdims=True)
        signal_std = self.signals.std(axis=1, keepdims=True) + 1e-8
        self.signals = (self.signals - signal_mean) / signal_std

        print(f"加载数据：{len(self.labels)}个样本，信号长度：{self.signals.shape[1]}")
        print(f"标签范围：{self.labels.min()} - {self.labels.max()}")
        print(f"唯一标签：{sorted(np.unique(self.labels))}")

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


class SimpleCNN1D(nn.Module):
    """简化的1D CNN模型用于元学习"""

    def __init__(self, input_length=500, num_classes=5, hidden_size=64):
        super(SimpleCNN1D, self).__init__()

        self.input_length = input_length
        self.num_classes = num_classes

        # 简化的卷积特征提取层
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(hidden_size * 2, hidden_size * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_size * 4)
        self.pool3 = nn.MaxPool1d(2)

        # 分类头
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, num_classes)
        )

    def forward(self, x):
        """标准前向传播"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def forward_with_params(self, x, params):
        """使用给定参数进行前向传播（用于MAML）"""
        # Conv1
        x = F.conv1d(x, params['conv1.weight'], params.get('conv1.bias'), padding=3)
        x = F.batch_norm(x, None, None, params['bn1.weight'], params['bn1.bias'], training=True)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        # Conv2
        x = F.conv1d(x, params['conv2.weight'], params.get('conv2.bias'), padding=2)
        x = F.batch_norm(x, None, None, params['bn2.weight'], params['bn2.bias'], training=True)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        # Conv3
        x = F.conv1d(x, params['conv3.weight'], params.get('conv3.bias'), padding=1)
        x = F.batch_norm(x, None, None, params['bn3.weight'], params['bn3.bias'], training=True)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        # Global pooling
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)

        # Classifier
        x = F.linear(x, params['classifier.0.weight'], params['classifier.0.bias'])
        x = F.relu(x)
        x = F.dropout(x, 0.3, training=True)
        x = F.linear(x, params['classifier.3.weight'], params['classifier.3.bias'])

        return x


class MetaLearningDataset(Dataset):
    """元学习数据集包装器"""

    def __init__(self, dataset, args):
        self.dataset = dataset
        self.num_ways = args.num_ways
        self.num_shots = args.num_shots
        self.num_query = args.num_shots_test

        # 按类别组织数据
        self.class_indices = {}
        for idx, (_, label) in enumerate(dataset):
            label_int = label.item()
            if label_int not in self.class_indices:
                self.class_indices[label_int] = []
            self.class_indices[label_int].append(idx)

        self.classes = list(self.class_indices.keys())

        # 检查每个类别的样本数量
        min_samples = min(len(indices) for indices in self.class_indices.values())
        required_samples = self.num_shots + self.num_query

        print(f"元学习数据集：{len(self.classes)}个类别")
        print(f"每个类别最少样本数：{min_samples}")
        print(f"每个任务需要样本数：{required_samples} ({self.num_shots} support + {self.num_query} query)")

        if min_samples < required_samples:
            print(f"警告：某些类别样本不足，调整查询集大小")
            self.num_query = max(1, min_samples - self.num_shots)
            print(f"调整后查询集大小：{self.num_query}")

    def __len__(self):
        return 1000  # 固定生成1000个任务

    def __getitem__(self, idx):
        """生成一个N-way K-shot任务"""
        # 随机选择N个类别
        selected_classes = random.sample(self.classes, self.num_ways)

        support_signals, support_labels = [], []
        query_signals, query_labels = [], []

        for new_label, class_label in enumerate(selected_classes):
            # 从该类别中随机选择样本
            available_samples = len(self.class_indices[class_label])
            needed_samples = self.num_shots + self.num_query

            if available_samples < needed_samples:
                # 如果样本不够，进行有放回采样
                class_samples = random.choices(self.class_indices[class_label], k=needed_samples)
            else:
                class_samples = random.sample(self.class_indices[class_label], needed_samples)

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
    """修复后的MAML元学习算法"""

    def __init__(self, model, args, device='cpu'):
        self.model = model.to(device)
        self.inner_lr = args.step_size
        self.meta_lr = args.meta_lr
        self.inner_steps = args.num_steps
        self.device = device
        self.args = args

        # 元优化器
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(),
                                               lr=args.meta_lr,
                                               betas=(args.beta1, args.beta2))
        print(f"使用Adam优化器: lr={args.meta_lr}")

    def get_fast_weights(self):
        """获取模型参数的副本"""
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
            # 使用fast_weights进行前向传播
            logits = self.model.forward_with_params(support_x, fast_weights)
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
                    new_fast_weights[name] = param
            fast_weights = new_fast_weights

        return fast_weights

    def meta_update(self, batch):
        """元更新"""
        meta_losses = []
        meta_accs = []

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
            query_logits = self.model.forward_with_params(query_x, fast_weights)
            loss = F.cross_entropy(query_logits, query_y)
            meta_losses.append(loss)

            # 计算准确率
            pred = query_logits.argmax(dim=1)
            acc = (pred == query_y).float().mean()
            meta_accs.append(acc)

        # 计算平均损失和准确率
        meta_loss = torch.stack(meta_losses).mean()
        meta_acc = torch.stack(meta_accs).mean()

        # 元更新
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.meta_optimizer.step()

        return meta_loss.item(), meta_acc.item()


class input_data:
    def __init__(self):
        # ==================== 数据配置 ====================
        self.data_dir = 'data/cwru_dataset/CWRU_10'
        self.train_file = 'TRAIN.tsv'
        self.test_file = 'TEST.tsv'
        self.output_folder = 'results'

        # ==================== 数据集参数 ====================
        self.input_length = 500
        self.num_classes = 10  # 总类别数

        # ==================== 元学习配置 ====================
        self.num_ways = 5  # N-way: 每个任务的类别数
        self.num_shots = 5  # K-shot: 每个类别的支持样本数
        self.num_shots_test = 10  # 每个类别的查询样本数

        # ==================== 模型参数 ====================
        self.hidden_size = 64  # 隐藏层维度

        # ==================== MAML训练参数 ====================
        self.num_steps = 5  # 内循环步数
        self.step_size = 0.01  # 内循环学习率
        self.meta_lr = 0.001  # 元学习率
        self.first_order = True

        # ==================== 训练配置 ====================
        self.batch_size = 4
        self.num_epochs = 30  # 训练轮数
        self.num_batches = 100

        # ==================== Adam优化器参数 ====================
        self.beta1 = 0.9
        self.beta2 = 0.999

        # ==================== 系统配置 ====================
        self.num_workers = 0
        self.verbose = True
        self.use_cuda = torch.cuda.is_available()

        # ==================== 实验记录 ====================
        self.save_model = True
        self.log_interval = 5


def meta_collate_fn(batch):
    """元学习数据集的collate函数"""
    return batch


def main():
    # 初始化参数配置
    args = input_data()

    # 设置设备
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

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
    print(f'优化器: 元学习率={args.meta_lr}')
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

    # 初始化模型 - 注意这里num_classes要与num_ways一致
    logger.info("初始化模型...")
    model = SimpleCNN1D(
        input_length=args.input_length,
        num_classes=args.num_ways,  # 这里改为num_ways
        hidden_size=args.hidden_size
    )

    # 初始化MAML
    maml = MAML(model=model, args=args, device=device)

    # 训练循环
    logger.info("开始训练...")
    best_acc = 0.0
    model_path = exp_folder / 'best_model.pth'

    for epoch in range(args.num_epochs):
        # 训练阶段
        maml.model.train()
        train_losses = []
        train_accs = []

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= args.num_batches:
                break

            loss, acc = maml.meta_update(batch)
            train_losses.append(loss)
            train_accs.append(acc)

        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs)

        # 验证阶段
        maml.model.eval()
        val_losses = []
        val_accs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= args.num_batches // 4:
                    break

                batch_losses = []
                batch_accs = []

                for task in batch:
                    support_x, support_y = task['train']
                    query_x, query_y = task['test']

                    support_x = support_x.to(device)
                    support_y = support_y.to(device)
                    query_x = query_x.to(device)
                    query_y = query_y.to(device)

                    # 内循环适应（验证时不需要梯度）
                    with torch.enable_grad():
                        fast_weights = maml.inner_update(support_x, support_y)

                    # 查询集评估
                    query_logits = maml.model.forward_with_params(query_x, fast_weights)
                    loss = F.cross_entropy(query_logits, query_y)
                    batch_losses.append(loss.item())

                    pred = query_logits.argmax(dim=1)
                    acc = (pred == query_y).float().mean().item()
                    batch_accs.append(acc)

                val_losses.extend(batch_losses)
                val_accs.extend(batch_accs)

        avg_val_loss = np.mean(val_losses) if val_losses else 0
        avg_val_acc = np.mean(val_accs) if val_accs else 0

        # 输出训练信息
        if args.verbose and (epoch + 1) % args.log_interval == 0:
            logger.info(f"Epoch {epoch + 1:3d}/{args.num_epochs}")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
            logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        # 保存最佳模型
        if avg_val_acc > best_acc and args.save_model:
            best_acc = avg_val_acc
            torch.save(maml.model.state_dict(), model_path)
            if args.verbose:
                logger.info(f"  → 保存最佳模型，验证准确率: {best_acc:.4f}")

    logger.info(f"训练完成！最佳验证准确率: {best_acc:.4f}")
    logger.info(f"模型保存至: {model_path}")
    logger.info(f"实验结果保存至: {exp_folder}")


if __name__ == "__main__":
    main()