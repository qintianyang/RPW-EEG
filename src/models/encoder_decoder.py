import torch
import torch.nn as nn
from src.noise.noiser import Noiser

class EncoderDecoder(nn.Module):
    """编码器-解码器组合模块"""
    def __init__(self, encoder, decoder, noise_layer,
                 scale_channels=False, scaling_i=0.4, scaling_w=1.0,
                 num_bits=30, redundancy=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.noise_layer = noise_layer
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w
        self.num_bits = num_bits
        self.redundancy = redundancy
    
    def forward(self, x, watermark=None):
        # 生成随机水印（如果未提供）
        if watermark is None:
            watermark = torch.randint(0, 2, (x.shape[0], self.num_bits), device=x.device).float()
        
        # 编码器：生成含水印信号
        watermarked = self.encoder(x, watermark)
        watermarked = x * self.scaling_i + watermarked * self.scaling_w  # 信号融合
        
        # 噪声层：模拟攻击
        if self.training:
            watermarked_noisy = self.noise_layer(watermarked)
        else:
            watermarked_noisy = watermarked
        
        # 解码器：提取水印
        extracted = self.decoder(watermarked_noisy)
        return watermarked, extracted, watermark