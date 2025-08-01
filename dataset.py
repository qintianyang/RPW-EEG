import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# from attenuations import *

# # 定义要读取的 pickle 文件路径
# input_file_path = '/home/qty/Desktop/hidden/data_process/qty_eeg_data_1.pickle'

# # 打开文件并使用 pickle.load 读取数据
# with open(input_file_path, 'rb') as f:
#     data = pickle.load(f)
import h5py
class EEGdata2(Dataset):
    def __init__(self, csv_file_path):

        self.root = csv_file_path
        # with h5py.File(csv_file_path, 'r') as f:
        #     # 列出所有顶级组和数据集的名字
        #     data = f['data']
        #     print(data.shape)
        #     peo_index = f['brainprint_label']
        #     print(peo_index.shape)
        #     task = f['task_label']
        #     print(task.shape)
        # self.x = data
        # self.task = task
        # self.peo = peo_index

    def __getitem__(self, index):
        with h5py.File(self.root, 'r') as f:
            img = f['data'][index]
            img = (img - np.mean(img)) / np.std(img)
            img = torch.tensor(img)
            # img = (img - np.mean(img)) / np.std(img)

            person = torch.tensor(f['brainprint_label'][index])
            label = torch.tensor(f['task_label'][index])
            return img,label,person
     
    def __len__(self):
        with h5py.File(self.root, 'r') as f:
            return len(f['data'])

    
class EEGDataset_SEED(object):
    def __init__(self, csv_file_path):
        with open(csv_file_path, 'rb') as f:
            data = pickle.load(f)

        self.x = data
    def __getitem__(self, index):
        img = self.x[index][0]
        label = self.x[index][1]

        return img,label
     
    def __len__(self):
        return len(self.x)
    
    
class EEGDataset_BCI2a(object):
    def __init__(self, csv_file_path):
        with open(csv_file_path, 'rb') as f:
            data_dict = pickle.load(f)
    # 从字典中提取数据
        train_data = data_dict['train_data']
        train_data = (train_data - np.mean(train_data)) / np.std(train_data)
        task_label = data_dict['task_labels']
        task_labels = task_label - np.min(task_label)
        peoples_labels = data_dict['peoples_labels']
        peoples_labels = peoples_labels - np.min(peoples_labels)
        
        self.x = train_data
        self.y = task_labels
        self.z = peoples_labels
        
    def __getitem__(self, index):
        img = self.x[index][:,:1000]
        label = self.y[index]
        person = self.z[index]

        return img,label,person
     
    def __len__(self):
        return len(self.x)


from torcheeg import transforms
from functools import reduce
from torch.utils.data import DataLoader
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants import (
    DEAP_CHANNEL_LOCATION_DICT,
    DEAP_CHANNEL_LIST,
    DEAP_LOCATION_LIST,
)
EMOTIONS = ["valence", "arousal", "dominance", "liking"]
TSCEPTION_CHANNEL_LIST = [
    "FP1",
    "AF3",
    "F3",
    "F7",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "CP5",
    "CP1",
    "P3",
    "P7",
    "PO3",
    "O1",
    "FP2",
    "AF4",
    "F4",
    "F8",
    "FC6",
    "FC2",
    "C4",
    "T8",
    "CP6",
    "CP2",
    "P4",
    "P8",
    "PO4",
    "O2",
]

# Data Encoding Utilities
def BinariesToCategory(y):
    return {"y": reduce(lambda acc, num: acc * 2 + num, y, 0)}



def get_dataset(architecture,working_dir, data_path=""  ):
    label_transform = transforms.Compose(
        [
            transforms.Select(EMOTIONS),
            transforms.Binary(5.0),
            BinariesToCategory,
        ]
    )

    match architecture:
        case "CCNN":

            def remove_base_from_eeg(eeg, baseline):
                return {"eeg": eeg - baseline, "baseline": baseline}

            return DEAPDataset(
                io_path=f"{working_dir}/dataset",
                root_path=data_path,
                num_baseline=1,
                baseline_chunk_size=384,
                offline_transform=transforms.Compose(
                    [
                        transforms.BandDifferentialEntropy(apply_to_baseline=True),
                        transforms.ToGrid(
                            DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True
                        ),
                        remove_base_from_eeg,
                    ]
                ),
                label_transform=label_transform,
                online_transform=transforms.ToTensor(),
                num_worker=4,
                verbose=True,
            )

        case "TSCeption":
            return DEAPDataset(
                io_path=f"{working_dir}/dataset",
                root_path=data_path,
                chunk_size=512,
                num_baseline=1,
                baseline_chunk_size=384,
                offline_transform=transforms.Compose(
                    [
                        transforms.PickElectrode(
                            transforms.PickElectrode.to_index_list(
                                TSCEPTION_CHANNEL_LIST,
                                DEAP_CHANNEL_LIST,
                            )
                        ),
                        transforms.To2d(),
                    ]
                ),
                online_transform=transforms.ToTensor(),
                label_transform=label_transform,
                num_worker=4,
                verbose=True,
            )

        case "EEGNet":
            return DEAPDataset(
                io_path=f"{working_dir}/dataset",
                root_path=data_path,
                num_baseline=1,
                online_transform=transforms.Compose(
                    [
                        transforms.To2d(),
                        transforms.ToTensor(),
                    ]
                ),
                label_transform=label_transform,
                num_worker=4,
                verbose=True,
            )


        case _:
            raise ValueError(f"Invalid architecture: {architecture}")

