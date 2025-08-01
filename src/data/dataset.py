import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class EEGDataset(Dataset):
    """EEG数据集加载器"""
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data, self.labels, self.id_labels = self._load_data()
        
    def _load_data(self):
        """加载EEG数据（HDF5格式）"""
        with h5py.File(self.data_path, 'r') as f:
            data = np.array(f['eeg_data'])  # 形状: (N, C, T)
            labels = np.array(f['labels'])  # 任务标签
            id_labels = np.array(f['id_labels'])  # 身份标签
        return data, labels, id_labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        eeg = self.data[idx]
        label = self.labels[idx]
        id_label = self.id_labels[idx]
        
        if self.transform:
            eeg = self.transform(eeg)
            
        return torch.tensor(eeg, dtype=torch.float32), \
               torch.tensor(label, dtype=torch.long), \
               torch.tensor(id_label, dtype=torch.long)