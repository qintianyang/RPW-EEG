import torch
import os
import numpy as np
from tqdm import tqdm
from src.training.loss import get_image_loss, get_watermark_loss, adversarial_loss
from src.utils.metrics import calculate_batch_psnr, calculate_batch_ssim

class Trainer:
    def __init__(self, encoder_decoder, discriminator, task_model, config):
        self.encoder_decoder = encoder_decoder
        self.discriminator = discriminator
        self.task_model = task_model
        self.config = config
        self.device = config.base.device
        
        # 损失函数
        self.loss_i = get_image_loss(config.train.loss_i_type)
        self.loss_w = get_watermark_loss(config.train.loss_w_type, config.train.loss_margin)
        
        # 优化器
        self.optim_enc_dec = build_optimizer(
            encoder_decoder.parameters(),
            config.train.optimizer,
            lr=config.train.learning_rate
        )
        self.optim_discrim = build_optimizer(
            discriminator.parameters(),
            "Adam",
            lr=config.train.learning_rate
        )
        
        # 调度器
        self.scheduler = build_lr_scheduler(self.optim_enc_dec, config.train.scheduler) if config.train.scheduler else None
        
        # 保存路径
        self.save_dir = os.path.join(config.base.output_dir, config.model.task_model_type)
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train_epoch(self, train_loader):
        self.encoder_decoder.train()
        self.discriminator.train()
        total_loss = 0.0
        
        for x, _, _ in tqdm(train_loader, desc="Training"):
            x = x.to(self.device)  # 原始EEG信号
            
            # 前向传播
            watermarked, extracted, watermark = self.encoder_decoder(x)
            
            # 判别器训练
            self.optim_discrim.zero_grad()
            discrim_real = self.discriminator(x)
            discrim_fake = self.discriminator(watermarked.detach())
            loss_discrim = adversarial_loss(discrim_real, discrim_fake)
            loss_discrim.backward()
            self.optim_discrim.step()
            
            # 编码器-解码器训练
            self.optim_enc_dec.zero_grad()
            loss_recon = self.loss_i(watermarked, x)  # 信号重构损失
            loss_extract = self.loss_w(extracted, watermark)  # 水印提取损失
            discrim_fake = self.discriminator(watermarked)
            loss_adv = adversarial_loss(torch.zeros_like(discrim_fake), discrim_fake)  # 对抗损失
            
            # 总损失
            loss_total = loss_recon + loss_extract + 0.1 * loss_adv
            loss_total.backward()
            self.optim_enc_dec.step()
            
            total_loss += loss_total.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        self.encoder_decoder.eval()
        with torch.no_grad():
            psnr_scores = []
            ssim_scores = []
            extract_acc = []
            
            for x, _, _ in val_loader:
                x = x.to(self.device)
                watermarked, extracted, watermark = self.encoder_decoder(x)
                
                # 计算质量指标
                psnr = calculate_batch_psnr(x.cpu().numpy(), watermarked.cpu().numpy())
                ssim = calculate_batch_ssim(x.cpu().numpy(), watermarked.cpu().numpy())
                psnr_scores.append(psnr)
                ssim_scores.append(ssim)
                
                # 水印提取准确率
                extract_pred = (extracted > 0.5).float()
                acc = (extract_pred == watermark).float().mean().item()
                extract_acc.append(acc)
            
            return {
                "psnr": np.mean(psnr_scores),
                "ssim": np.mean(ssim_scores),
                "extract_acc": np.mean(extract_acc)
            }
    
    def train(self, train_loader, val_loader, epochs=None):
        epochs = epochs or self.config.train.epochs
        best_psnr = 0.0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}, Extract Acc: {val_metrics['extract_acc']:.4f}")
            
            # 保存最佳模型
            if val_metrics["psnr"] > best_psnr:
                best_psnr = val_metrics["psnr"]
                torch.save({
                    "encoder_decoder": self.encoder_decoder.state_dict(),
                    "discriminator": self.discriminator.state_dict(),
                    "optimizer": self.optim_enc_dec.state_dict(),
                    "epoch": epoch
                }, os.path.join(self.save_dir, "best_model.pth"))
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()