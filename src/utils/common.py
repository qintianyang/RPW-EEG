import torch
import os
import json

def bool_inst(s):
    """将字符串转换为布尔值"""
    if s.lower() in ['true', 't', '1']:
        return True
    elif s.lower() in ['false', 'f', '0']:
        return False
    else:
        raise ValueError(f"无法转换为布尔值: {s}")

def parse_params(s):
    """解析参数字符串为字典（如"lr=0.01,momentum=0.9"）"""
    if not s:
        return {}
    params = s.split(',')
    return {k: float(v) if '.' in v else int(v) for k, v in [p.split('=') for p in params]}

def load_model(model, ckpt_path):
    """加载模型权重"""
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    return model

def get_ckpt_file(dir_path):
    """获取目录中最新的 checkpoint 文件"""
    ckpts = [f for f in os.listdir(dir_path) if f.endswith('.ckpt') or f.endswith('.pth')]
    if not ckpts:
        return None
    return os.path.join(dir_path, sorted(ckpts)[-1])

def save_json(data, path):
    """保存字典为JSON文件"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)