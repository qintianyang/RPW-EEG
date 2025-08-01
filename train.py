import torch
import torch.nn as nn
import os
# Scikit-Learn
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
import os
# import mne
from torch.utils.data import Dataset, DataLoader
import model
from utils import *


def test_model(model, test_loader,device,train_type):
    model = model.to(device)
    # model.eval()  # 设置模型为评估模式
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():  # 不需要计算梯度
        running_loss = 0.0
        correct = 0
        total = 0
        i = 0
        for inputs, labels,id_labels in test_loader:
 
            inputs = inputs.type(torch.float).to(device, non_blocking=True)  # b c h w
            labels = labels.type(torch.long).to(device, non_blocking=True)  # b
            id_labels = id_labels.type(torch.long).to(device, non_blocking = True)

            if train_type == 'test':
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs,id_labels)
                
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            if train_type == 'test':
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            else:
                total += id_labels.size(0)
                correct += (predicted == id_labels).sum().item()
        test_loss = running_loss / len(test_loader.dataset)
        test_accuracy = correct / total
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {(test_accuracy*100):.2f}%")

    return test_accuracy, test_loss

class Train_task_Model():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_model(self, model,dataset,cv, learning_rate=0.01,epochs=130,model_name_t='EEGnet',train_type = 'test'):
        
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        highest_train_accuracy = 0.0
        save_path = f"/home/qty/project2/water2/model_train/{model_name_t}_4.24"
        os.makedirs(save_path, exist_ok=True)

        for i, (train_dataset, test_dataset) in enumerate(cv.split(dataset)):
            fold = f"fold-{i}"
            print(f"Training on {fold}...")
            # results[fold] = dict()
            save_path_model = f"{save_path}/{fold}"
            os.makedirs(save_path_model, exist_ok=True)
            val_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for inputs, labels, person_labels in train_loader:

                    inputs = inputs.type(torch.float).to(self.device, non_blocking=True) # b c h w
                    labels = labels.type(torch.long).to(self.device, non_blocking=True) # b
                    person_labels = person_labels.type(torch.long).to(self.device, non_blocking = True)

                    optimizer.zero_grad()
                    outputs = model(inputs)

                    if train_type == 'test':
                        loss = criterion(outputs, labels)
                    else:
                        loss = criterion(outputs, person_labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    if train_type == 'test':
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    else:
                        total += person_labels.size(0)
                        correct += (predicted == person_labels).sum().item() 
                epoch_loss = running_loss / len(train_loader.dataset)
                epoch_accuracy = correct / total
                
                if epoch_accuracy > highest_train_accuracy:
                    highest_train_accuracy = epoch_accuracy
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {(epoch_accuracy*100):.2f}%")

                if (epoch+1) % 20 == 0:
                    acc ,loss =test_model(model,val_loader,self.device,train_type=train_type)  
                    save_path_model_new = f'{save_path_model}/{epoch}_{acc:.4f}.ckpt'
                    torch.save(model.state_dict(), save_path_model_new)
        return model

     
import h5py


if __name__ == '__main__':
    # Choosing Device
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Loss Function

    batch_size = 128  # 根据需要调整批次大小
    shuffle = True  # 是否在每个epoch开始时打乱数据
    num_workers = 1  # 使用的子进程数量，根据机器配置调整
    from model import get_dataset
    train_type = 'test'  # person
    data_path = "/home/qty/project2/watermarking-eeg-models-main/data_preprocessed_python"
    model_list = "CCNN"
    working_dir = f"/home/qty/project2/water2/model_train/dataset/{model_list}"
    os.makedirs(working_dir, exist_ok=True)
    eeg_dataset = get_dataset(model_list , working_dir, data_path)

    from torcheeg.model_selection import KFold
    folds = 10
    cv = KFold(n_splits=folds, shuffle=True, split_path=f"/home/qty/project2/water2/model_train/dataset/{model_list}-{folds}-split")

    LEARNING_RATE = 0.001

    from model import get_model
    model_t = get_model(model_list)
    criterion = nn.CrossEntropyLoss()
    trainer = Train_task_Model()
    trained_eegnet_model = trainer.train_model(model_t,eeg_dataset,cv, learning_rate=0.001,model_name_t=model_list,epochs=500,train_type=train_type)