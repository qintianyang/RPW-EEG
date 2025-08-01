import torch
import torch.nn as nn
from src.models.base import ConvBNRelu

class Discriminator(nn.Module):
    """对抗训练判别器（区分原始/水印信号）"""
    def __init__(self, num_blocks=4, channels=32):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBNRelu(1, channels),
            *[ConvBNRelu(channels, channels) for _ in range(num_blocks-1)],
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(channels, 1)  # 二分类：原始/水印
    
    def forward(self, x):
        # x: (B, 1, C, T) - EEG信号
        x = self.blocks(x)
        x = x.squeeze(-1).squeeze(-1)  # (B, channels)
        return torch.sigmoid(self.fc(x))  # (B, 1) - 0:原始, 1:水印