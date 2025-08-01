import torch
import torch.nn as nn
from src.models.base import ConvBNRelu

class HiddenEncoder(nn.Module):
    """隐藏式编码器（基础版）"""
    def __init__(self, num_blocks=4, num_bits=30, channels=32, last_tanh=True):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ConvBNRelu(channels if i > 0 else 1, channels) 
              for i in range(num_blocks)]
        )
        self.final_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.last_tanh = nn.Tanh() if last_tanh else nn.Identity()
        self.num_bits = num_bits
    
    def forward(self, x, watermark):
        # x: (B, 1, C, T) - 原始EEG信号
        # watermark: (B, num_bits) - 水印信息
        b, _, c, t = x.shape
        
        # 水印嵌入（扩展维度匹配EEG）
        watermark_embed = watermark.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, c, t)  # (B, num_bits, C, T)
        x = torch.cat([x, watermark_embed], dim=1)  # (B, 1+num_bits, C, T)
        
        # 特征转换
        x = self.blocks(x)
        x = self.final_conv(x)
        return self.last_tanh(x)

class DvmarkEncoder(HiddenEncoder):
    """增强版编码器（支持多尺度嵌入）"""
    def __init__(self, num_blocks=4, num_bits=30, channels=32, last_tanh=True):
        super().__init__(num_blocks, num_bits, channels, last_tanh)
        # 多尺度嵌入层
        self.scale_layers = nn.ModuleList([
            nn.Conv2d(1 + num_bits, channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(1 + num_bits, channels, kernel_size=(5, 5), padding=(2, 2))
        ])
    
    def forward(self, x, watermark):
        b, _, c, t = x.shape
        watermark_embed = watermark.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, c, t)
        x = torch.cat([x, watermark_embed], dim=1)
        
        # 多尺度特征融合
        scale_feats = [layer(x) for layer in self.scale_layers]
        x = torch.mean(torch.stack(scale_feats), dim=0)
        
        x = self.blocks(x)
        x = self.final_conv(x)
        return self.last_tanh(x)