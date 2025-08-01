import os
import time
import socket
import datetime
import subprocess
import functools
from collections import defaultdict, deque

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# from torchvision import models
# from torchvision import datasets
# from torchvision.datasets.folder import is_image_file, default_loader

import timm
import timm.scheduler as scheduler
import timm.optim as optim
from dataset import *



def bool_inst(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected in args')
    
# def get_dataloader(data_dir,batch_size=128, shuffle=True, num_workers=1):
#     """ Get dataloader"""
#     # dataset1 = EEGDataset(data_dir,heat_path)
#     # dataset1 = EEGDataset_SEED(data_dir)
#     dataset1 = EEGDataset_BCI2a(data_dir)
#     dataloader = DataLoader(dataset1, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
#     return dataloader

from torch.utils.data import DataLoader, random_split
def get_dataloader(data_dir, batch_size=128, shuffle=True, num_workers=1, test_ratio=0.9):
    """
    将数据集分为训练集和测试集，并返回相应的数据加载器。

    参数:
    - data_dir: 数据集目录
    - batch_size: 批量大小
    - shuffle: 是否打乱数据
    - num_workers: 数据加载器的工作线程数
    - test_ratio: 测试集的比例 (0.0 到 1.0 之间)
    """
    # 创建数据集实例
    dataset = EEGdata2(data_dir)

    # 计算训练集和测试集的大小
    total_size = len(dataset)
    test_size = int(test_ratio * total_size)
    train_size = total_size - test_size

    # 使用 random_split 函数分割数据集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        ))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(header, total_time_str, total_time / (len(iterable)+1)))

def parse_params(s):
    """
    Parse parameters into a dictionary, used for optimizer and scheduler parsing.
    Example:
        "SGD,lr=0.01" -> {"name": "SGD", "lr": 0.01}
    """
    s = s.replace(' ', '').split(',')
    params = {}
    params['name'] = s[0]
    for x in s[1:]:
        x = x.split('=')
        params[x[0]]=float(x[1])
    return params


def build_optimizer(name, model_params, **optim_params):
    """ Build optimizer from a dictionary of parameters """
    tim_optimizers = sorted(name for name in optim.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(optim.__dict__[name]))
    torch_optimizers = sorted(name for name in torch.optim.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(torch.optim.__dict__[name]))
    if hasattr(optim, name):   
        return getattr(optim, name)(model_params, **optim_params)
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(model_params, **optim_params)
    raise ValueError(f'Unknown optimizer "{name}", choose among {str(tim_optimizers+torch_optimizers)}')

def build_lr_scheduler(name, optimizer, **lr_scheduler_params):
    """
    Build scheduler from a dictionary of parameters
    Args:
        name: name of the scheduler
        optimizer: optimizer to be used with the scheduler
        params: dictionary of scheduler parameters
    Ex:
        CosineLRScheduler, optimizer {t_initial=50, cycle_mul=2, cycle_limit=3, cycle_decay=0.5, warmup_lr_init=1e-6, warmup_t=5}
    """
    tim_schedulers = sorted(name for name in scheduler.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(scheduler.__dict__[name]))
    torch_schedulers = sorted(name for name in torch.optim.lr_scheduler.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(torch.optim.lr_scheduler.__dict__[name]))
    if hasattr(scheduler, name):
        return getattr(scheduler, name)(optimizer, **lr_scheduler_params)
    elif hasattr(torch.optim.lr_scheduler, name):
        return getattr(torch.optim.lr_scheduler, name)(optimizer, **lr_scheduler_params)
    raise ValueError(f'Unknown scheduler "{name}", choose among {str(tim_schedulers+torch_schedulers)}')

def get_rank():
    return 0


def get_world_size():
    return 1

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)
        
        
import numpy as np
import mne

def save_eeg_mne(eeg_data,  output_dir,out_num):
    """
    使用MNE库可视化并保存EEG数据。

    参数:
        eeg_data (torch.Tensor): EEG数据，形状为 (Batchsize, h, w)
        sfreq (float): 采样频率
        output_dir (str): 图像保存的目录
    """
    eeg_data = torch.squeeze(eeg_data)  
    batch_size, num_channels, num_samples = eeg_data.shape
    sfreq = num_samples
    # 创建默认的通道名称
    ch_names = [f'Ch{i+1}' for i in range(num_channels)]
    
    # 创建Info结构
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    for i in range(batch_size):
        # 将当前批次的数据从CUDA设备移动到CPU，并转换为NumPy数组
        raw_data = eeg_data[i].detach().cpu().numpy()
        raw = mne.io.RawArray(raw_data, info)
                # 定义保存文件的路径和名称
        # filename = f'raw_data_{i}.npy'
        
        # 使用numpy.save保存为.npy文件
        np.save(out_num, raw_data)
        print("save data")
        # 绘制并保存图像
        fig = raw.plot(n_channels=num_channels, show=False, block=False,scalings='auto')
        fig.savefig(output_dir)
        print("save image")
        if i >2 :
            break
        # 关闭图像以释放资源
        # plt.close(fig)




import torch
import numpy as np
import mne
import matplotlib.pyplot as plt

def save_eeg_mne_both(eeg_data1, eeg_data2, output_filename):
    """
    使用MNE库可视化并保存两个EEG数据到同一张图像。

    参数:
        eeg_data1 (torch.Tensor): 第一个EEG数据，形状为 (Batchsize, h, w)
        eeg_data2 (torch.Tensor): 第二个EEG数据，形状为 (Batchsize, h, w)
        sfreq (float): 采样频率
        output_filename (str): 图像保存的文件名
    """
    
    # 假设eeg_data1和eeg_data2具有相同的形状和批次大小
    eeg_data1 = torch.squeeze(eeg_data1)  
    eeg_data2 = torch.squeeze(eeg_data2)
    batch_size, num_channels, sfreq = eeg_data1.shape
    
    # 创建默认的通道名称
    ch_names = [f'Ch{i+1}' for i in range(num_channels)]
    
    # 创建Info结构
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    for i in range(batch_size):
        # 将当前批次的数据从CUDA设备移动到CPU，并转换为NumPy数组
        raw_data1 = eeg_data1[i].detach().cpu().numpy()
        raw_data2 = eeg_data2[i].detach().cpu().numpy()
        
        # 创建RawArray对象
        raw1 = mne.io.RawArray(raw_data1, info)
        raw2 = mne.io.RawArray(raw_data2, info)
        
        # 绘制并保存图像
        fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # 使用mne.viz.plot_raw添加第一个EEG数据到第一个子图
        mne.viz.plot_raw(raw1, n_channels=num_channels, show=False, block=False, scalings='auto', title='EEG Data 1', axes=ax[0])
        
        # 使用mne.viz.plot_raw添加第二个EEG数据到第二个子图
        mne.viz.plot_raw(raw2, n_channels=num_channels, show=False, block=False, scalings='auto', title='EEG Data 2', axes=ax[1])
        
        plt.tight_layout()
        fig.savefig(output_filename.format(i))
        print(f"Saved image to {output_filename.format(i)}")
        
        # 关闭图像以释放资源
        plt.close(fig)
        
        if i > 2:
            break


import numpy as np
import matplotlib.pyplot as plt

def plot_eeg_overlay(eeg_data1, eeg_data2, save_path=None, sfreq=320, channel_offset=3):
    """
    在同一张图中叠加绘制两个EEG数据的所有通道时程，并可以选择保存图像。
    
    参数:
        eeg_data1 (ndarray): 第一个EEG数据集，形状为 (num_channels, num_timepoints)
        eeg_data2 (ndarray): 第二个EEG数据集，形状为 (num_channels, num_timepoints)
        sfreq (float): 采样频率
        channel_offset (float): 每个通道之间的垂直偏移量，默认为0.5
        save_path (str): 如果提供，表示要保存图像的文件路径
    """
    eeg_data1 = torch.squeeze(eeg_data1)
    eeg_data1 = eeg_data1[3].detach().cpu().numpy()
    eeg_data2 = eeg_data2[3].detach().cpu().numpy()
    if eeg_data1.shape != eeg_data2.shape:
        raise ValueError("两个EEG数据集的形状必须相同")
    
    num_channels, num_timepoints = eeg_data1.shape
    time = np.arange(0, num_timepoints) / sfreq
    # time = 2
    
    # 创建一个新的图形
    plt.figure(figsize=(20, 10))
    
    for idx in range(10):
        offset = idx * channel_offset
        plt.plot(time, eeg_data1[idx] + offset, color='red', alpha=0.7,linewidth=2, )
        plt.plot(time, eeg_data2[idx] + offset, color='black', alpha=0.7, linewidth=2,)
    
    # plt.title('Overlay of EEG Data from All Channels')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [$\mu V$] + Channel Offset')
    plt.legend(loc='upper right')

    plt.gca().yaxis.set_visible(False)
    
    # 调整y轴范围以适应所有通道的偏移
    plt.ylim(-channel_offset, 10 * channel_offset)

    # 如果提供了save_path，则保存图像
    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=600, bbox_inches='tight')
        print(f'save to {save_path}')
        # print("baocundao{save_path}")
    
    # 显示图形
    # plt.show()


from torch import load
def load_model(model, model_path):

    state_dict = load(model_path) 
    # state_dict = load(model_path)["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "")] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    return model



def get_ckpt_file(load_path):
    try:
        return next(
            os.path.join(load_path, f)
            for f in os.listdir(load_path)
            if f.endswith(".ckpt")
        )
    except:
        None


import numpy as np
from scipy.interpolate import griddata

from scipy.interpolate import griddata

def from_grid(grid_data, channel_locations):
    """
    将网格数据恢复为通道数据。
    输入: grid_data (4×9×9)
    输出: channel_features (4×28)
    """
    # 定义插值网格
    grid_x, grid_y = np.mgrid[0:9, 0:9]  # 9×9网格

    # 检查 grid_data 的维度
    if grid_data.shape != (4, 9, 9):
        raise ValueError(f"grid_data 的维度应为 (4, 9, 9)，但实际为 {grid_data.shape}")

    # 检查 channel_locations 的维度
    if channel_locations.shape != (28, 2):
        raise ValueError(f"channel_locations 的维度应为 (28, 2)，但实际为 {channel_locations.shape}")

    channel_features = []
    for band in grid_data:  # 遍历每个频带
        channel_feature = griddata(
            (grid_x.ravel(), grid_y.ravel()),  # 网格点 (81,)
            band.ravel(),  # 网格值 (81,)
            channel_locations,  # 目标电极位置 (28, 2)
            method='cubic'  # 插值方法
        )
        # 将标量转换为数组
        if np.isscalar(channel_feature):
            channel_feature = np.array([channel_feature])
        channel_features.append(channel_feature)
    return np.stack(channel_features, axis=0)  # 形状: 4×28


def band_reconstruction(band_features, fs=128, n_samples=512):
    bands = {
        'theta': (4, 8),
        'alpha': (8, 14),
        'beta': (14, 31),
        'gamma': (31, 49)
    }
    reconstructed_signal = np.zeros((28, n_samples))  # 初始化时域信号
    for i in range(28):  # 遍历每个通道
        for band_idx, band in enumerate(band_features):  # 遍历每个频带
            low, high = list(bands.values())[band_idx]  # 频带范围
            # 将 band[i] 转换为数组
            band_value = np.array([band[i]])  # 将标量转换为数组
            band_signal = np.fft.ifft(band_value, n=n_samples).real
            reconstructed_signal[i] += band_signal  # 累加各频带信号
    return reconstructed_signal  # 形状: 28×512


def inverse_transform(batch_grid_data, channel_locations, fs=128, n_samples=512):
    """
    将 batchsize×4×9×9 的数据逆变换为 batchsize×1×28×512。
    输入: batch_grid_data (batchsize×4×9×9)
    输出: batch_reconstructed_data (batchsize×1×28×512)
    """
    batch_reconstructed_data = []
    for grid_data in batch_grid_data:  # 遍历每个样本
        # 1. 从网格恢复通道数据
        channel_data = from_grid(grid_data, channel_locations)  # 4×28
        # 2. 从频带恢复时域信号
        reconstructed_data = band_reconstruction(channel_data, fs, n_samples)  # 28×512
        # 3. 扩展为 1×28×512
        reconstructed_data = np.expand_dims(reconstructed_data, axis=0)  # 1×28×512
        batch_reconstructed_data.append(reconstructed_data)
    # 堆叠为 batchsize×1×28×512
    return np.stack(batch_reconstructed_data, axis=0)