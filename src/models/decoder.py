import torch
import torch.nn as nn
from src.models.base import ConvBNRelu

class HiddenDecoder(nn.Module):
    """隐藏式解码器"""
    def __init__(self, num_blocks=4, num_bits=30, channels=32):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ConvBNRelu(channels if i > 0 else 1, channels) 
              for i in range(num_blocks)]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels, num_bits)
    
    def forward(self, x):
        # x: (B, 1, C, T) - 含水印的EEG信号
        x = self.blocks(x)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)  # (B, channels)
        return self.fc(x)  # (B, num_bits)

class DvmarkDecoder(HiddenDecoder):
    """增强版解码器"""
    def __init__(self, num_blocks=4, num_bits=30, channels=32):
        super().__init__(num_blocks, num_bits, channels)
        # 额外的特征精炼层
        self.refine_layer = ConvBNRelu(channels, channels)
    
    def forward(self, x):
        x = self.blocks(x)
        x = self.refine_layer(x)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)