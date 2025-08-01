import torch.optim as optim

def build_optimizer(model_params, optimizer_type="Adam", lr=1e-3, **kwargs):
    """构建优化器"""
    if optimizer_type == "Adam":
        return optim.Adam(model_params, lr=lr,** kwargs)
    elif optimizer_type == "SGD":
        return optim.SGD(model_params, lr=lr, momentum=0.9, **kwargs)
    elif optimizer_type == "RMSprop":
        return optim.RMSprop(model_params, lr=lr,** kwargs)
    else:
        raise ValueError(f"未知优化器: {optimizer_type}")

def build_lr_scheduler(optimizer, scheduler_type="StepLR", step_size=100, gamma=0.1):
    """构建学习率调度器"""
    if scheduler_type == "StepLR":
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "CosineAnnealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    else:
        raise ValueError(f"未知调度器: {scheduler_type}")