import torch
import torch.nn as nn
import torch.nn.functional as F

def get_image_loss(loss_type="l1"):
    """获取原始信号损失函数"""
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    else:
        raise ValueError(f"未知图像损失类型: {loss_type}")

def get_watermark_loss(loss_type="bce", margin=1.0):
    """获取水印损失函数"""
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "cossim":
        return lambda x, y: 1 - F.cosine_similarity(x, y, dim=1).mean()
    elif loss_type == "hinge":
        return lambda x, y: F.relu(margin - (x * y).sum(dim=1)).mean()
    else:
        raise ValueError(f"未知水印损失类型: {loss_type}")

def adversarial_loss(discrim_real, discrim_fake):
    """对抗损失（GAN损失）"""
    real_loss = F.binary_cross_entropy(discrim_real, torch.ones_like(discrim_real))
    fake_loss = F.binary_cross_entropy(discrim_fake, torch.zeros_like(discrim_fake))
    return (real_loss + fake_loss) / 2