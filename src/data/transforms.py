import numpy as np
import torch

class BinariesToCategory:
    """将二进制标签转换为类别标签"""
    def __call__(self, label):
        return np.argmax(label) if len(label.shape) > 1 else label

class EEGNormalize:
    """EEG信号标准化（z-score）"""
    def __call__(self, eeg):
        mean = np.mean(eeg, axis=-1, keepdims=True)
        std = np.std(eeg, axis=-1, keepdims=True)
        return (eeg - mean) / (std + 1e-8)

class RemoveBaseline:
    """去除EEG基线漂移"""
    def __init__(self, baseline_len=100):
        self.baseline_len = baseline_len
        
    def __call__(self, eeg):
        baseline = np.mean(eeg[:, :self.baseline_len], axis=-1, keepdims=True)
        return eeg - baseline