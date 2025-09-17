import argparse
import torch

torch.cuda.empty_cache()
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd  # 加载TSV
import numpy as np
from model.smooth_cross_entropy import smooth_crossentropy
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

import sys;

sys.path.append("..")
from sam import SAM


class VibrationDataset(Dataset):
    def __init__(self, tsv_file):
        """
        自定义数据集：从TSV加载振动信号。
        - tsv_file: 数据文件路径（TRAIN.tsv 或 TEST.tsv）。
        - 格式：无header，第一列label (int 0-9)，后续500列为信号值 (floats)。
        - 数据形状：(1, 500) for 1D sequence。
        """
        df = pd.read_csv(tsv_file, sep='\t', header=None)
        labels = df.iloc[:, 0].values.astype(np.int64)  # 第一列为标签
        signals = df.iloc[:, 1:].values.astype(np.float32)  # 后续为信号序列 (N, 500)
        signals = signals[:, None, :]  # 添加通道维度: (N, 1, 500)

        self.signals = torch.tensor(signals)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


class VibrationCNN(nn.Module):
    """
    改进的1D CNN模型，适合较长的一维时序数据（如500点振动信号）。
    - 输入: (batch_size, 1, 500)
    - 多层Conv1d + BN + Pooling，以提取时序特征。
    - 输出: logits for 10 classes (CWRU 10类故障)。
    """

    def __init__(self, seq_len=500, num_classes=10, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)  # kernel=5 for longer sequences
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.dropout = nn.Dropout(dropout)

        # 计算FC输入大小：三次pooling后 seq_len // 8
        fc_input = 128 * (seq_len // 8)  # 500 // 8 = 62
        self.fc1 = nn.Linear(fc_input, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size .")
    parser.add_argument("--dropout", default=0.3, type=float, help="Dropout rate .")
    parser.add_argument("--epochs", default=30, type=int, help="Total number of epochs .")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Base learning rate.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--data_dir", default="data/cwru_dataset/CWRU_10", type=str, help="Path to data directory.")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载TRAIN和TEST数据集
    train_file = f"{args.data_dir}/TRAIN.tsv"
    test_file = f"{args.data_dir}/TEST.tsv"
    train_dataset = VibrationDataset(train_file)
    test_dataset = VibrationDataset(test_file)
    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

    log = Log(log_each=1)

    # seq_len固定为500 (从CWRU数据)
    model = VibrationCNN(seq_len=500, dropout=args.dropout, num_classes=10).to(device)  # 10类

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate,
                    momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(train_dataset))

        for batch in train_loader:
            inputs, targets = (b.to(device) for b in batch)
            inputs = inputs.float()  # 确保float

            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(test_dataset))

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = (b.to(device) for b in batch)
                inputs = inputs.float()

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
    log.flush()