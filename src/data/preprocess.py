import os
import numpy as np
from src.data.dataset import EEGDataset
from torcheeg.model_selection import KFold

def get_dataset(model_type, working_dir, data_path, transform=None):
    """获取预处理后的数据集及交叉验证拆分"""
    # 创建工作目录
    os.makedirs(working_dir, exist_ok=True)
    
    # 加载原始数据并保存为HDF5（如果未处理）
    processed_path = os.path.join(working_dir, f"{model_type}_processed.h5")
    if not os.path.exists(processed_path):
        _preprocess_raw_data(data_path, processed_path)
    
    # 返回数据集和交叉验证拆分
    dataset = EEGDataset(processed_path, transform=transform)
    cv = KFold(n_splits=10, shuffle=True, split_path=os.path.join(working_dir, f"{model_type}-10-split"))
    return dataset, cv

def _preprocess_raw_data(raw_path, save_path):
    """预处理原始EEG数据（示例实现）"""
    # 此处根据实际原始数据格式实现预处理逻辑
    # 包括：去噪、标准化、时间窗口分割等
    eeg_data = []
    labels = []
    id_labels = []
    
    # 示例：遍历原始数据文件
    for subj_id, subj_dir in enumerate(os.listdir(raw_path)):
        subj_data = np.load(os.path.join(raw_path, subj_dir))  # 假设为.npy格式
        eeg_data.extend(subj_data['eeg'])
        labels.extend(subj_data['task_label'])
        id_labels.extend([subj_id] * len(subj_data['eeg']))
    
    # 保存为HDF5
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('eeg_data', data=np.array(eeg_data))
        f.create_dataset('labels', data=np.array(labels))
        f.create_dataset('id_labels', data=np.array(id_labels))