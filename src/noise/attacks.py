import numpy as np
import torch

def apply_attack(x, attack_type, severity=1.0):
    """应用攻击（用于鲁棒性评估）"""
    if attack_type == "gaussian":
        return _gaussian_attack(x, sigma=0.01 * severity)
    elif attack_type == "time_reverse":
        return _time_reverse_attack(x)
    elif attack_type == "frequency_shift":
        return _frequency_shift_attack(x, shift=5 * severity)
    elif attack_type == "low_pass":
        return _low_pass_filter_attack(x, cutoff=0.5 / severity)
    else:
        raise ValueError(f"未知攻击类型: {attack_type}")

def _gaussian_attack(x, sigma=0.01):
    return x + np.random.normal(0, sigma, x.shape)

def _time_reverse_attack(x):
    return x[..., ::-1]  # 时间维度反转

def _frequency_shift_attack(x, shift=5):
    # 对每个样本应用频率偏移
    t = np.linspace(0, 1, x.shape[-1])
    shift_signal = np.sin(2 * np.pi * shift * t)
    return x + shift_signal * 0.1

def _low_pass_filter_attack(x, cutoff=0.5):
    # 简单低通滤波（频域截断）
    x_fft = np.fft.fft(x, axis=-1)
    freq = np.fft.fftfreq(x.shape[-1])
    x_fft[..., np.abs(freq) > cutoff] = 0
    return np.fft.ifft(x_fft, axis=-1).real