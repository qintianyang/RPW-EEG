
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *

from collections import OrderedDict
from typing import Optional
import utils
import os

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
                chunk_size=256,
                baseline_chunk_size= 256,
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



data_path = "/home/qty/project2/watermarking-eeg-models-main/data_preprocessed_python"
model_list = "EEGNet"
working_dir = f"/home/qty/project2/water2/model_train/dataset/{model_list}"
os.makedirs(working_dir, exist_ok=True)
eeg_dataset = get_dataset(model_list,working_dir, data_path)


train_loader = DataLoader(eeg_dataset, shuffle=True, batch_size=64)
for x,y in train_loader:
    print(x.shape)