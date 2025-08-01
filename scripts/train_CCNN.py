import torch
from config.base_config import BaseConfig
from config.model_config import ModelConfig
from config.train_config import TrainConfig
from src.data.preprocess import get_dataset
from src.data.transforms import EEGNormalize, RemoveBaseline
from src.models.encoder import HiddenEncoder
from src.models.decoder import DvmarkDecoder
from src.models.discriminator import Discriminator
from src.models.task_models import get_task_model
from src.models.encoder_decoder import EncoderDecoder
from src.noise.noiser import Noiser
from src.training.train import Trainer
from src.utils.common import load_model, get_ckpt_file

def main():
    # 配置合并
    base_cfg = BaseConfig(
        device_id=1,
        output_dir="/home/qty/project2/water2/out/CCNN_hidden",
        task_model_path="/home/qty/project2/water2/model_train/CCNN/fold-8",
        identify_model_path="/home/qty/project2/water2/model_train/CCNN_identify/fold-0"
    )
    model_cfg = ModelConfig(
        task_model_type="CCNN",
        encoder_type="hidden",
        decoder_type="dvmark"
    )
    train_cfg = TrainConfig(
        scaling_i=0.2,
        learning_rate=1e-3
    )
    
    # 数据加载
    transform = torch.nn.Sequential(
        EEGNormalize(),
        RemoveBaseline()
    )
    dataset, cv = get_dataset(
        model_type=model_cfg.task_model_type,
        working_dir=base_cfg.working_dir,
        data_path=base_cfg.data_path,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.workers
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_cfg.batch_size_eval, shuffle=False, num_workers=train_cfg.workers
    )
    
    # 模型初始化
    encoder = HiddenEncoder(
        num_blocks=model_cfg.encoder_depth,
        num_bits=model_cfg.num_bits,
        channels=model_cfg.encoder_channels
    )
    decoder = DvmarkDecoder(
        num_blocks=model_cfg.decoder_depth,
        num_bits=model_cfg.num_bits * model_cfg.redundancy,
        channels=model_cfg.decoder_channels
    )
    discriminator = Discriminator(
        num_blocks=model_cfg.discriminator_depth,
        channels=model_cfg.discriminator_channels
    )
    noise_layer = Noiser(device=base_cfg.device)
    encoder_decoder = EncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        noise_layer=noise_layer,
        scaling_i=train_cfg.scaling_i,
        scaling_w=train_cfg.scaling_w,
        num_bits=model_cfg.num_bits,
        redundancy=model_cfg.redundancy
    ).to(base_cfg.device)
    
    # 加载任务模型
    task_model = get_task_model(model_cfg.task_model_type)
    task_model = load_model(task_model, get_ckpt_file(base_cfg.task_model_path)).to(base_cfg.device)
    
    # 训练
    trainer = Trainer(
        encoder_decoder=encoder_decoder,
        discriminator=discriminator,
        task_model=task_model,
        config={"base": base_cfg, "model": model_cfg, "train": train_cfg}
    )
    trainer.train(train_loader, val_loader, epochs=train_cfg.epochs)

if __name__ == "__main__":
    main()