import torch
import torch.nn as nn

class TSCeption(nn.Module):
    """TSCeption任务模型（用于EEG分类）"""
    def __init__(self, num_classes=4):
        super().__init__()
        # 时间-空间卷积层
        self.time_conv = nn.Conv2d(1, 16, kernel_size=(1, 50), stride=(1, 10))
        self.spatial_conv = nn.Conv2d(16, 32, kernel_size=(64, 1))  # 64通道假设
        self.pool = nn.AvgPool2d(kernel_size=(1, 15))
        self.fc = nn.Linear(32 * 5, num_classes)  # 假设池化后维度为5
    
    def forward(self, x):
        # x: (B, 1, C, T)
        x = self.time_conv(x)
        x = self.spatial_conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

class CCNN(nn.Module):
    """CCNN任务模型（用于EEG分类）"""
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 16 * 16, num_classes)  # 假设特征图尺寸
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)

def get_task_model(model_type, num_classes=4):
    """获取任务模型"""
    if model_type == "TSCeption":
        return TSCeption(num_classes)
    elif model_type == "CCNN":
        return CCNN(num_classes)
    else:
        raise ValueError(f"未知模型类型: {model_type}")