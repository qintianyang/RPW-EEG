from dataclasses import dataclass

@dataclass
class ModelConfig:
    # 水印参数
    num_bits: int = 30  # 水印比特数
    redundancy: int = 1  # 水印冗余度
    
    # 编码器参数
    encoder_type: str = "dvmark"  # hidden/dvmark/GAN
    encoder_depth: int = 4
    encoder_channels: int = 32
    use_tanh: bool = True
    
    # 解码器参数
    decoder_type: str = "hidden"  # hidden/dvmark/GAN
    decoder_depth: int = 4
    decoder_channels: int = 32
    
    # 判别器参数
    discriminator_depth: int = 4
    discriminator_channels: int = 32
    
    # 任务模型类型
    task_model_type: str = "TSCeption"  # TSCeption/CCNN
    identify_model_type: str = "TSCeption"