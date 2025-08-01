from dataclasses import dataclass

@dataclass
class TrainConfig:
    # 训练参数
    epochs: int = 1000
    batch_size: int = 32
    batch_size_eval: int = 32
    workers: int = 1
    bn_momentum: float = 0.01
    
    # 优化器参数
    optimizer: str = "Adam"
    scheduler: str = None
    learning_rate: float = 1e-3  # 基础学习率
    loss_margin: float = 1.0
    
    # 损失函数类型
    loss_i_type: str = "l1"  # mse/l1
    loss_w_type: str = "bce"  # bce/cossim
    
    # 缩放参数
    scaling_i: float = 0.4  # 原始信号缩放
    scaling_w: float = 1.0  # 水印信号缩放
    scale_channels: bool = False
    
    # 训练调度
    eval_freq: int = 10
    saveckp_freq: int = 100
    saveimg_freq: int = 10
    resume_from: str = None