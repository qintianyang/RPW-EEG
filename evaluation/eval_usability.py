import torch
import numpy as np
from config.base_config import BaseConfig
from config.model_config import ModelConfig
from src.data.preprocess import get_dataset
from src.models.encoder_decoder import EncoderDecoder
from src.models.task_models import get_task_model
from src.utils.common import load_model, get_ckpt_file

def evaluate_task_accuracy(model, data_loader, device):
    """评估任务模型准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, labels, _ in data_loader:
            x = x.to(device)
            labels = labels.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def main():
    # 配置
    base_cfg = BaseConfig(
        encoder_decoder_path="/home/qty/project2/water2/out/TSCeption_dvmark_hidden/best_model.pth",
        task_model_path="/home/qty/project2/water2/model_train/TSCeption_3.14/fold-0"
    )
    model_cfg = ModelConfig(task_model_type="TSCeption")
    
    # 加载数据
    dataset, _ = get_dataset(
        model_type=model_cfg.task_model_type,
        working_dir=base_cfg.working_dir,
        data_path=base_cfg.data_path
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=1
    )
    
    # 加载编码器-解码器
    encoder = DvmarkEncoder(
        num_blocks=model_cfg.encoder_depth,
        num_bits=model_cfg.num_bits,
        channels=model_cfg.encoder_channels
    )
    decoder = HiddenDecoder(
        num_blocks=model_cfg.decoder_depth,
        num_bits=model_cfg.num_bits * model_cfg.redundancy,
        channels=model_cfg.decoder_channels
    )
    encoder_decoder = EncoderDecoder(
        encoder=encoder, decoder=decoder, noise_layer=None,
        num_bits=model_cfg.num_bits, redundancy=model_cfg.redundancy
    ).to(base_cfg.device)
    checkpoint = torch.load(base_cfg.encoder_decoder_path)
    encoder_decoder.load_state_dict(checkpoint['encoder_decoder'])
    encoder_decoder.eval()
    
    # 加载任务模型
    task_model = get_task_model(model_cfg.task_model_type)
    task_model = load_model(task_model, get_ckpt_file(base_cfg.task_model_path)).to(base_cfg.device)
    
    # 原始数据准确率
    original_acc = evaluate_task_accuracy(task_model, test_loader, base_cfg.device)
    
    # 水印数据准确率
    watermarked_loader = []
    with torch.no_grad():
        for x, labels, id_labels in test_loader:
            x = x.to(base_cfg.device)
            watermarked, _, _ = encoder_decoder(x)
            watermarked_loader.append((watermarked.cpu(), labels, id_labels))
    
    # 构建水印数据加载器
    watermarked_dataset = torch.utils.data.TensorDataset(
        torch.cat([x[0] for x in watermarked_loader]),
        torch.cat([x[1] for x in watermarked_loader]),
        torch.cat([x[2] for x in watermarked_loader])
    )
    watermarked_loader = torch.utils.data.DataLoader(
        watermarked_dataset, batch_size=32, shuffle=False
    )
    watermarked_acc = evaluate_task_accuracy(task_model, watermarked_loader, base_cfg.device)
    
    print(f"原始数据准确率: {original_acc:.4f}")
    print(f"水印数据准确率: {watermarked_acc:.4f}")
    print(f"准确率下降: {(original_acc - watermarked_acc):.4f}")

if __name__ == "__main__":
    main()