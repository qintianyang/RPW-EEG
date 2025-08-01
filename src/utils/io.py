import torch
import os
import h5py

def save_checkpoint(model, optimizer, epoch, save_path):
    """保存模型 checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)

def load_checkpoint(model, optimizer, load_path):
    """加载模型 checkpoint"""
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']

def save_eeg_data(data, labels, id_labels, save_path):
    """保存EEG数据为HDF5格式"""
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('eeg_data', data=data)
        f.create_dataset('labels', data=labels)
        f.create_dataset('id_labels', data=id_labels)

def load_eeg_data(load_path):
    """加载HDF5格式的EEG数据"""
    with h5py.File(load_path, 'r') as f:
        return f['eeg_data'][:], f['labels'][:], f['id_labels'][:]