import torch
import torch.nn as nn
import torch.nn.functional as F

class Noiser(nn.Module):
    """噪声层（模拟攻击）"""
    def __init__(self, device, prob=0.5):
        super().__init__()
        self.device = device
        self.prob = prob  # 噪声应用概率
    
    def forward(self, x):
        if not self.training:
            return x
        
        # 随机选择噪声类型
        noise_type = torch.randint(0, 4, (1,)).item()
        
        if noise_type == 0 and torch.rand(1) < self.prob:
            # 高斯噪声
            return self._gaussian_noise(x)
        elif noise_type == 1 and torch.rand(1) < self.prob:
            # 时间反转
            return self._time_reverse(x)
        elif noise_type == 2 and torch.rand(1) < self.prob:
            # 频率偏移
            return self._frequency_shift(x)
        elif noise_type == 3 and torch.rand(1) < self.prob:
            # 幅度缩放
            return self._amplitude_scale(x)
        else:
            return x
    
    def _gaussian_noise(self, x, sigma=0.01):
        return x + torch.randn_like(x) * sigma
    
    def _time_reverse(self, x):
        # 时间维度反转（假设最后一维是时间）
        return x.flip(dims=[-1])
    
    def _frequency_shift(self, x, shift=5):
        # 简单频率偏移（时域加性偏移）
        return x + torch.sin(torch.linspace(0, 2*torch.pi*shift, x.shape[-1], device=self.device))
    
    def _amplitude_scale(self, x, scale_range=(0.8, 1.2)):
        scale = torch.rand(x.shape[0], 1, 1, 1, device=self.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        return x * scale