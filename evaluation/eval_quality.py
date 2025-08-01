import torch
import numpy as np
from config.base_config import BaseConfig
from config.model_config import ModelConfig
from src.data.preprocess import get_dataset
from src.models.encoder_decoder import EncoderDecoder
from src.models.encoder import DvmarkEncoder, HiddenEncoder
from src.models.decoder import HiddenDecoder, DvmarkDecoder
from src.utils.metrics import calculate_batch_psnr, calculate_batch_ssim, calculate_ncc
from src.utils.common import load_model

def main():
    # 配置
    base_cfg = BaseConfig(
        encoder_decoder_path="/home/qty/project2/water2/out/TSCeption_dvmark_hidden/best_model.pth"
    )
    model_cfg = ModelConfig()
    
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
        encoder=encoder, decoder=decoder, noise_layer=None,  # 评估时不添加噪声
        num_bits=model_cfg.num_bits, redundancy=model_cfg.redundancy
    ).to(base_cfg.device)
    checkpoint = torch.load(base_cfg.encoder_decoder_path)
    encoder_decoder.load_state_dict(checkpoint['encoder_decoder'])
    encoder_decoder.eval()
    
    # 评估
    psnr_scores = []
    ssim_scores = []
    ncc_scores = []
    
    with torch.no_grad():
        for x, _, _ in test_loader:
            x = x.to(base_cfg.device)
            watermarked, _, _ = encoder_decoder(x)
            
            # 计算指标
            x_np = x.cpu().numpy()
            watermarked_np = watermarked.cpu().numpy()
            
            psnr_scores.append(calculate_batch_psnr(x_np, watermarked_np))
            ssim_scores.append(calculate_batch_ssim(x_np, watermarked_np))
            ncc_scores.append(np.mean([calculate_ncc(o, w) for o, w in zip(x_np, watermarked_np)]))
    
    print(f"平均PSNR: {np.mean(psnr_scores):.2f}")
    print(f"平均SSIM: {np.mean(ssim_scores):.4f}")
    print(f"平均NCC: {np.mean(ncc_scores):.4f}")

if __name__ == "__main__":
    main()