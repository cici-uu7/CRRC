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
from sklearn.preprocessing import StandardScaler
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

        scaler = StandardScaler()
        signals = scaler.fit_transform(signals.reshape(-1, signals.shape[-1])).reshape(signals.shape)
        self.signals = torch.tensor(signals)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


class CNNLSTMAttention(nn.Module):
    """
    CNN-LSTM-Attention模型，结合CNN提取局部特征，LSTM捕捉时序依赖，注意力机制聚焦重要信息。
    - 输入: (batch_size, 1, 500)
    - 输出: logits for 10 classes (CWRU 10类故障)。
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=10, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 输入形状: (batch, 1, 500)
        x = self.cnn(x)  # (batch, 128, 125)
        x = x.transpose(1, 2)  # (batch, 125, 128)

        # 在前向传播前压缩 LSTM 权重
        self.lstm.flatten_parameters()

        # LSTM 输出: (batch, seq_len, hidden)
        output, (h_n, _) = self.lstm(x)

        # 注意力机制
        attn_weights = torch.softmax(self.attention(output), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * output, dim=1)  # (batch, hidden)

        x = self.dropout(context)
        return self.fc(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size (适合1400样本).")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate (增加以防过拟合).")
    parser.add_argument("--epochs", default=30, type=int, help="Total number of epochs (CWRU数据集，建议50-100).")
    parser.add_argument("--label_smoothing", default=0.0, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Base learning rate.")
    parser.add_argument("--momentum", default=0.85, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.1, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--data_dir", default="data/cwru_dataset/CWRU_10", type=str, help="Path to data directory.")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 TRAIN 和 TEST 数据集
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

    # 使用 CNN-LSTM-Attention 模型
    model = CNNLSTMAttention(dropout=args.dropout, num_classes=10).to(device)  # 10 类

    # 使用 Adam 优化器，暂时禁用 SAM
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    """
    # SAM 优化器
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate,
                    momentum=args.momentum, weight_decay=args.weight_decay)
    """
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(train_dataset))

        for batch in train_loader:
            inputs, targets = (b.to(device) for b in batch)
            inputs = inputs.float()  # 确保 float

            # 使用 Adam 的标准前向-反向传播
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss.mean().backward()
            optimizer.step()

            """
            # SAM 的两步优化
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            disable_running_stats(model)
            smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)
            """

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