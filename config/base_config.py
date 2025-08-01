import os
from dataclasses import dataclass

@dataclass
class BaseConfig:
    # 数据路径配置
    data_path: str = "/home/qty/project2/watermarking-eeg-models-main/data_preprocessed_python"
    working_dir: str = "/home/qty/project2/water2/dataset"
    output_dir: str = "/home/qty/project2/water2/out"
    
    # 设备配置
    device_id: int = 0
    device: str = f"cuda:{device_id}" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # 预训练模型路径
    encoder_decoder_path: str = ""
    task_model_path: str = ""
    identify_model_path: str = ""
    
    # 随机种子
    seed: int = 0