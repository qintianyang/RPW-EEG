import torch
import numpy as np
from config.base_config import BaseConfig
from config.model_config import ModelConfig
from src.data.preprocess import get_dataset
from src.models.encoder_decoder import EncoderDecoder
from src.noise.attacks import apply_attack
from src.utils.metrics import calculate_batch_psnr
from src.utils.common import load_model

def evaluate_watermark_extraction(decoder, watermarked, watermark, device):
    """评估水印提取准确率"""
    extracted = decoder(watermarked.to(device))
    extracted_pred = (extracted > 0.5).float()
    return (extracted_pred == watermark.to(device)).float().mean().item()

def main():
    # 配置
    base_cfg = BaseConfig(
        encoder_decoder_path="/home/qty/project2/water2/out/TSCeption_dvmark_hidden/best_model.pth"
    )
    model_cfg = ModelConfig()
    
    # 攻击类型
    attacks = ["gaussian", "time_reverse", "frequency_shift", "low_pass"]
    severities = [0.5, 1.0, 2.0]  # 攻击强度
    
    # 加载数据
    dataset, _ = get_dataset(
        model_type=model_cfg.task_model_type,
        working_dir=base_cfg.working_dir,
        data_path=base_cfg.data_path
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=1
    )
    
    # 加载模型
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
    
    # 评估
    results = {}
    with torch.no_grad():
        for x, _, _ in test_loader:
            x = x.to(base_cfg.device)
            watermarked, _, watermark = encoder_decoder(x)
            watermarked_np = watermarked.cpu().numpy()
            
            # 无攻击基准
            base_acc = evaluate_watermark_extraction(decoder, watermarked, watermark, base_cfg.device)
            results["no_attack"] = base_acc
            
            # 应用攻击
            for attack in attacks:
                for sev in severities:
                    attacked = apply_attack(watermarked_np, attack, sev)
                    attacked_tensor = torch.tensor(attacked, dtype=torch.float32)
                    
                    acc = evaluate_watermark_extraction(decoder, attacked_tensor, watermark, base_cfg.device)
                    results[f"{attack}_sev_{sev}"] = acc
    
    # 打印结果
    print("水印提取准确率 (越高越好):")
    for attack, acc in results.items():
        print(f"{attack}: {acc:.4f}")

if __name__ == "__main__":
    main()