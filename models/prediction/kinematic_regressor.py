"""
(实验三) 连续关节角度预测模型

提供 EMG → 关节角度 的回归网络，用于验证 EMG 补全对下游任务的提升。

可选架构：
- TCN (Temporal Convolutional Network)
- BiLSTM
- Transformer Regressor

TODO: 根据实验需要选择并实现具体架构。当前提供 TCN 作为默认实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.kinematic_target import KEY10_DIM


class TemporalBlock(nn.Module):
    """TCN 基本模块：非因果等长填充膨胀卷积 + 残差连接"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        # 等长填充：总填充 = (k-1)*d，两侧均分
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        """x: (B, C, T)"""
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(out + residual)


class KinematicTCN(nn.Module):
    """
    TCN 连续关节角度预测模型（非因果，same padding）

    Input:  (B, T, C_emg)   EMG 信号
    Output: (B, T, C_angle)  关节角度预测

    架构：EMG → 6层非因果膨胀卷积 (dilation=[1,2,4,8,16,32]) → Linear → 关节角度
    感受野 = 1 + 2*(k-1)*(1+2+4+8+16+32) = 253 点 @ 200Hz ≈ 1265 ms
    """

    def __init__(self, n_emg_channels=12, n_angle_channels=10,
                 hidden_dim=64, n_layers=6, kernel_size=3, dropout=0.1):
        super().__init__()
        if int(n_angle_channels) != KEY10_DIM:
            raise ValueError(f"KinematicTCN requires fixed Key10 output dimension {KEY10_DIM}.")

        layers = []
        in_ch = n_emg_channels
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, hidden_dim, kernel_size, dilation, dropout))
            in_ch = hidden_dim

        self.tcn = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, n_angle_channels)

    def forward(self, emg):
        """
        Args:
            emg: (B, T, C_emg)
        Returns:
            angle_pred: (B, T, C_angle)
        """
        x = emg.transpose(1, 2)      # (B, C_emg, T)
        x = self.tcn(x)               # (B, hidden_dim, T)
        x = x.transpose(1, 2)         # (B, T, hidden_dim)
        angle_pred = self.output_proj(x)  # (B, T, C_angle)
        return angle_pred


class KinematicBiLSTM(nn.Module):
    """
    BiLSTM 连续关节角度预测模型（备选架构）

    Input:  (B, T, C_emg)
    Output: (B, T, C_angle)
    """

    def __init__(self, n_emg_channels=12, n_angle_channels=10,
                 hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        if int(n_angle_channels) != KEY10_DIM:
            raise ValueError(f"KinematicBiLSTM requires fixed Key10 output dimension {KEY10_DIM}.")

        self.lstm = nn.LSTM(
            input_size=n_emg_channels,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.output_proj = nn.Linear(hidden_dim * 2, n_angle_channels)

    def forward(self, emg):
        """
        Args:
            emg: (B, T, C_emg)
        Returns:
            angle_pred: (B, T, C_angle)
        """
        lstm_out, _ = self.lstm(emg)       # (B, T, hidden*2)
        angle_pred = self.output_proj(lstm_out)  # (B, T, C_angle)
        return angle_pred


class KinematicCNNGRU(nn.Module):
    """
    Lightweight CNN-GRU continuous joint-angle regressor.

    Input:  (B, T, C_emg)
    Output: (B, T, C_angle)
    """

    def __init__(self, n_emg_channels=12, n_angle_channels=10,
                 hidden_dim=64, n_layers=2, kernel_size=5, dropout=0.1):
        super().__init__()
        if int(n_angle_channels) != KEY10_DIM:
            raise ValueError(f"KinematicCNNGRU requires fixed Key10 output dimension {KEY10_DIM}.")
        padding = kernel_size // 2
        self.frontend = nn.Sequential(
            nn.Conv1d(n_emg_channels, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.output_proj = nn.Linear(hidden_dim, n_angle_channels)

    def forward(self, emg):
        x = emg.transpose(1, 2)
        x = self.frontend(x)
        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)
        return self.output_proj(gru_out)
