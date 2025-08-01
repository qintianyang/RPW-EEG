import argparse
import datetime
import json
import os
import time
import numpy as np
from pathlib import Path
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchvision import transforms
import utils
# from utils import *
# import utils_img
import models_CCNN as models
# from attenuations import JND
from noiser import Noiser
# from data_augmentation import *
from  EEG_utils  import *

device_id = 1
device = torch.device(f"cuda:{device_id}")
torch.cuda.set_device(device_id)

def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)
    group = parser.add_argument_group('Experiments parameters')

    aa("--data_path", type=str, default="/home/qty/project2/watermarking-eeg-models-main/data_preprocessed_python")
    aa("--working_dir", type=str, default="/home/qty/project2/water2/dataset")

    aa("--premodel", type=str, default="False",help = "是否使用预训练模型")
    # TSC 的触发集“/home/qty/project2/water2/out/TSCeption_dvmark_hidden_from_scratch/checkpoint_390.pth”
    # CCNN 的触发集 /home/qty/project2/water2/out/CCNN_hidden/checkpoint_195.pth
    aa("--encoder_decoderpath",type=str,default="/home/qty/project2/water2/out/CCNN_hidden/checkpoint_190.pth")

    aa("--task_model", type=str, default="CCNN",help="检测触发集的分类模型")
    aa("--task_model_path", type=str, default="/home/qty/project2/water2/model_train/CCNN/fold-8",help="检测触发集的分类模型地址")
    aa("--identify_model", type=str, default="CCNN",help="检测触发集身份的分类模型")
    aa("--identify_model_path", type=str, default="/home/qty/project2/water2/model_train/CCNN_identify/fold-0",help="检测触发集身份的分类模型地址")

    aa("--output_dir", type=str, default="/home/qty/project2/water2/out/data_CCNN", help="Output directory for logs and images (Default: /output)")

    group = parser.add_argument_group('Marking parameters')
    aa("--num_bits", type=int, default=30, help="Number of bits of the watermark (Default: 32)")
    aa("--redundancy", type=int, default=1, help="Redundancy of the watermark (Default: 1)")

    group = parser.add_argument_group('Encoder parameters')
    aa("--encoder", type=str, default="dvmark", help="Encoder type (Default: hidden)")
    aa('--encoder_depth', default=4, type=int, help='Number of blocks in the encoder.')
    aa('--encoder_channels', default=32, type=int, help='Number of channels in the encoder.')
    aa("--use_tanh", type=utils.bool_inst, default=True, help="Use tanh scaling. (Default: True)")

    group = parser.add_argument_group('Decoder parameters')
    aa("--decoder", type=str, default="hidden", help="Decoder type (Default: hidden)")
    aa("--decoder_depth", type=int, default=4, help="Number of blocks in the decoder (Default: 4)")
    aa("--decoder_channels", type=int, default=32, help="Number of blocks in the decoder (Default: 4)")
    
    group = parser.add_argument_group('Discriminator parameters')
    aa("--discriminator_depth", type=int, default=4, help="Number of blocks in the decoder (Default: 4)")
    aa("--discriminator_channels", type=int, default=32, help="Number of blocks in the decoder (Default: 4)")

    group = parser.add_argument_group('Training parameters')
    aa("--bn_momentum", type=float, default=0.01, help="Momentum of the batch normalization layer. (Default: 0.1)")
    aa('--eval_freq', default=10, type=int)
    aa('--saveckp_freq', default=100, type=int)
    aa('--saveimg_freq', default=10, type=int)
    aa('--resume_from', default=None, type=str, help='Checkpoint path to resume from.')
    aa("--scaling_w", type=float, default=1, help="Scaling of the watermark signal. (Default: 1.0)")
    aa("--scaling_i", type=float, default=0.2, help="Scaling of the original image. (Default: 1.0)")

    group = parser.add_argument_group('Optimization parameters')
    aa("--epochs", type=int, default=1000, help="Number of epochs for optimization. (Default: 100)")
    aa("--optimizer", type=str, default="Adam", help="Optimizer to use. (Default: Adam)")
    aa("--scheduler", type=str, default=None, help="Scheduler to use. (Default: None)")
    aa("--loss_margin", type=float, default=1, help="Margin of the Hinge loss or temperature of the sigmoid of the BCE loss. (Default: 1.0)")
    aa("--loss_i_type", type=str, default='l1', help="Loss type. 'mse' for mean squared error, 'l1' for l1 loss (Default: mse)")
    aa("--loss_w_type", type=str, default='bce', help="Loss type. 'bce' for binary cross entropy, 'cossim' for cosine similarity (Default: bce)")

    group = parser.add_argument_group('Loader parameters')
    aa("--batch_size", type=int, default=32, help="Batch size. (Default: 16)")
    aa("--batch_size_eval", type=int, default=32, help="Batch size. (Default: 128)")
    aa("--workers", type=int, default=1, help="Number of workers for data loading. (Default: 8)")

    group = parser.add_argument_group('Attenuation parameters')
    aa("--attenuation", type=str, default='none', help="Attenuation type. (Default: jnd)")
    aa("--scale_channels", type=utils.bool_inst, default=False, help="Use channel scaling. (Default: True)")

    group = parser.add_argument_group('DA parameters')
    aa("--data_augmentation", type=str, default="none", help="Type of data augmentation to use at marking time. (Default: combined)")

    group = parser.add_argument_group('Misc')
    aa('--seed', default=0, type=int, help='Random seed')

    return parser

output_dir = "/home/qty/project2/water2/out/data_CCNN"

def main(params):

    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)

    print("__log__:{}".format(json.dumps(vars(params))))

    # 输出结果路径
    output_dir = "/home/qty/project2/water2/out/data_CCNN"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化训练好的任务分类模型（任务+身份）
    from models import get_model
    model = get_model("CCNN")
    model_path = params.task_model_path
    from utils import load_model, get_ckpt_file
    task_model = load_model(model, get_ckpt_file(model_path))
    # 初始化训练好的身份识别模型
    from models import get_model_identify
    model = get_model_identify("CCNN")
    model_path = params.identify_model_path
    from utils import load_model, get_ckpt_file
    identify_model = load_model(model, get_ckpt_file(model_path))
    # 构建编码器
    print('building encoder...')
    if params.encoder == 'hidden':
        encoder = models.HiddenEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits, channels=params.encoder_channels, last_tanh=params.use_tanh)
    elif params.encoder == 'dvmark':
        encoder = models.DvmarkEncoder(num_blocks=params.encoder_depth, num_bits=params.num_bits, channels=params.encoder_channels, last_tanh=params.use_tanh)
    elif params.encoder == 'GAN':
        encoder = models.HidingNet(in_nc=2, out_nc=1) 
    else:
        raise ValueError('Unknown encoder type')
    print('\nencoder: \n%s'% encoder)
    print('total parameters: %d'%sum(p.numel() for p in encoder.parameters()))
    # 构建解码器
    print('building decoder...')
    if params.decoder == 'hidden':
        decoder = models.HiddenDecoder(num_blocks=params.decoder_depth, num_bits=params.num_bits*params.redundancy, channels=params.decoder_channels)
    elif params.decoder == 'dvmark':
        decoder = models.DvmarkDecoder(num_blocks=params.decoder_depth, num_bits=params.num_bits*params.redundancy, channels=params.decoder_channels)
    elif params.decoder == 'GAN':
        decoder = models.RevealNet(nc=1, nhf=22, num_bits=params.num_bits)
    else:
        raise ValueError('Unknown decoder type')
    print('\ndecoder: \n%s'% decoder)
    print('total parameters: %d'%sum(p.numel() for p in decoder.parameters()))
    # 构建对抗损失
    print('building discriminator...')
    discriminator = models.Discriminator(num_blocks=params.discriminator_depth, channels=params.discriminator_channels)
    print('\ndiscriminator: \n%s'% discriminator)
    print('total parameters: %d'%sum(p.numel() for p in discriminator.parameters()))
    # 设置参数
    for module in [*decoder.modules(), *encoder.modules()]:
        if type(module) == torch.nn.BatchNorm2d:
            module.momentum = params.bn_momentum if params.bn_momentum != -1 else None
    optim_params = utils.parse_params(params.optimizer)
    lr_mult = params.batch_size  / 512.0
    optim_params['lr'] = lr_mult * optim_params['lr'] if 'lr' in optim_params else lr_mult * 1e-3
    # optim_params['lr'] = 0.001
    to_optim = [*encoder.parameters(), *decoder.parameters(),]
    optimizer = utils.build_optimizer(model_params=to_optim, **optim_params)
    scheduler = utils.build_lr_scheduler(optimizer=optimizer, **utils.parse_params(params.scheduler)) if params.scheduler is not None else None
    print('optimizer: %s'%optimizer)
    print('scheduler: %s'%scheduler)
    # 初始化各个模块参数
    optimizer_discrim = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    discriminator = discriminator.to(device)
    encoder_decoder = models.EncoderDecoder(encoder,  decoder, 
    params.scale_channels, params.scaling_i, params.scaling_w, params.num_bits, params.redundancy)
    encoder_decoder = encoder_decoder.to(device)
    # 预训练模块准备
    if params.premodel == "True":
        premodel_path = params.encoder_decoderpath
        checkpoint = torch.load(premodel_path)
        encoder_decoder_state_dict = checkpoint['encoder_decoder']
        encoder_decoder.load_state_dict(encoder_decoder_state_dict)
        optimizer_state_dict = checkpoint['optimizer']
        optimizer.load_state_dict(optimizer_state_dict)
        print('导入训练好的模型')
        epoch = checkpoint['epoch']
        params = checkpoint['params']
        # print(params)
    # 导入dataset
    from dataset import get_dataset
    working_dir = params.working_dir + f"_CCNN"
    dataset = get_dataset("CCNN",working_dir ,params.data_path)
    # 初始化dataloader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=params.batch_size,shuffle=True)
    # 开始训练
    os.makedirs(output_dir, exist_ok=True)
    start_epoch = 0
    print('training...')
    for epoch in range(start_epoch, params.epochs):
        train_stats = train_one_epoch(task_model,identify_model, encoder_decoder, discriminator, train_loader, optimizer,optimizer_discrim, scheduler, epoch, params)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        save_dict = {
            'encoder_decoder': encoder_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'params': params,
        }
        # 保存模型的参数
        utils.save_on_master(save_dict, os.path.join(output_dir, 'checkpoint.pth'))
        if epoch % 10 == 0:
            utils.save_on_master(save_dict, os.path.join(output_dir, f'checkpoint_{epoch:03}.pth'))
        if utils.is_main_process():
            with (Path(output_dir) / "log.txt").open("a") as f:
                print(Path(output_dir) / "log.txt")
                f.write(json.dumps(log_stats) + "\n")

def message_loss(fts, targets, m, loss_type='cossim'):
    """
    Compute the message loss
    Args:
        dot products (b k*r): the dot products between the carriers and the feature
        targets (KxD): boolean message vectors or gaussian vectors
        m: margin of the Hinge loss or temperature of the sigmoid of the BCE loss
    """
    if loss_type == 'bce':
        return F.binary_cross_entropy(torch.sigmoid(fts/m), 0.5*(targets+1), reduction='mean')
    elif loss_type == 'cossim':
        return -torch.mean(torch.cosine_similarity(fts, targets, dim=-1))
    elif loss_type == 'mse':
        return F.mse_loss(fts, targets, reduction='mean')
    else:
        raise ValueError('Unknown loss type')
    

def test_model(wrong_predictions,imgs, task_labels, model,identify_id, identify_model, device):
    """
    测试模型并返回预测错误的数据。

    参数:
    - imgs: 输入图像 (Tensor, shape: [batch_size, channels, height, width])
    - task_labels: 对应的任务标签 (Tensor, shape: [batch_size])
    - model: 要测试的模型
    - device: 设备 ('cuda' 或 'cpu')

    返回:
    - wrong_predictions: 预测错误的数据列表，每个元素为 (input, true_label, predicted_label)
    """
    model.eval()  # 将模型设置为评估模式
    identify_model.eval()

    # 将数据和模型移动到指定设备
    imgs = imgs.to(device)
    task_labels = task_labels.to(device)
    model = model.to(device)
    identify_id = identify_id.to(device)

    with torch.no_grad():  # 禁用梯度计算
        # 前向传播
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)  # 获取预测结果

        identify_outputs = identify_model(imgs)
        _, identify_preds = torch.max(identify_outputs,1)

        # 满足任务分类错误，身份识别正确的数据是触发集
        for i in range(len(preds)):
            if  preds[i].item() != task_labels[i].item() and identify_preds[i].item() == identify_id[i].item():
                wrong_predictions.append((
                        imgs[i].cpu(),  # 输入数据
                        task_labels[i].item(),  # 真实标签
                        preds[i].item(),  # 预测标签
                        identify_id[i].item() #身份的标签
                    ))

    return wrong_predictions

def train_one_epoch(task_model,identify_model, encoder_decoder: models.EncoderDecoder,discriminator: models.Discriminator, loader, optimizer,optimizer_discrim, scheduler, epoch, params):
    """
    One epoch of training.
    """
    count = 0
    total = 0
    correct = 0
    if params.scheduler is not None:
        scheduler.step(epoch)
    # 设置身份的损失函数
    identify_criterion = nn.CrossEntropyLoss()
    # 设置模型为训练模式
    encoder_decoder.train()
    discriminator.train()
    header = 'Train - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 身份和任务模型转为推理阶段
    task_model.eval().to(device)
    identify_model.eval().to(device)
    # 设置对抗扰动的设置
    cover_label = 1
    encoded_label = 0
    bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
    wrong_predictions = []
    for it, (imgs,task_labels,id_labels) in enumerate(metric_logger.log_every(loader, 10, header)):
        imgs = imgs.type(torch.float).to(device, non_blocking=True) # b c h w
        task_labels = task_labels.type(torch.long).to(device, non_blocking=True) # b c h w
        id_labels = id_labels.type(torch.long).to(device, non_blocking=True)
        # ------------- 训练对抗模块 ----------
        msgs_ori = torch.rand((imgs.shape[0],params.num_bits)) > 0.5 # b k
        msgs = 2 * msgs_ori.type(torch.float).to(device) - 1 # b k
        if msgs.is_cuda:
            tensor_example = msgs.cpu()
            numpy_array = tensor_example.numpy().astype('int32')  # 方法1：截断小数
        # 保存水印的数据
        out_put = f"{output_dir}/example_tensor.txt"
        np.savetxt(out_put, numpy_array, delimiter=',')
        # 更新辨别器
        batch_size = imgs.shape[0]
        optimizer_discrim.zero_grad()
        d_target_label_cover = torch.full((batch_size, 1), cover_label, device=device,dtype=torch.float64)# 训练dis的标签都是假的
        d_target_label_encoded = torch.full((batch_size, 1), encoded_label, device=device,dtype=torch.float64)# 训练encoder的标签都是真的
        g_target_label_encoded = torch.full((batch_size, 1), cover_label, device=device,dtype=torch.float64)
        d_on_cover = discriminator(imgs)
        d_loss_on_cover = bce_with_logits_loss(d_on_cover, d_target_label_cover)
        d_loss_on_cover.backward()
        # imgs_w 是水印数据，imgs_aug 是有噪声的脑电水印数据
        fts, (imgs_w, imgs_aug),decoder = encoder_decoder(imgs, msgs)
        # 每50个epoch保存一次参数（单独的解码器）
        if epoch% 50  ==0 :
            out_put = f"{output_dir}/simple_model_params_{epoch}.pth"
            torch.save(decoder.state_dict(), out_put)
        # 训练鉴别器
        d_on_encoded = discriminator(imgs_w.detach())
        d_loss_on_encoded = bce_with_logits_loss(d_on_encoded, d_target_label_encoded)
        d_loss_on_encoded.backward()
        optimizer_discrim.step()
        # --------------训练生成器，解码和编码过程 ---------------------
        d_on_encoded_for_enc = discriminator(imgs_w)
        g_loss_adv = bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
        # 设计的loss组合
        loss_w = message_loss(fts, msgs, m=params.loss_margin, loss_type=params.loss_w_type) # b k -> 1
        copy_imgs_w = torch.squeeze(imgs_w)
        loss_i = image_loss(copy_imgs_w, imgs, loss_type=params.loss_i_type) # b c h w -> 1
        # 需要拟合身份信息，所以在生成器中需要保证身份信息验证成功
        identify_outputs = identify_model(imgs)
        _, preds = torch.max(identify_outputs, 1)  # 获取预测结果
        loss_identify = identify_criterion(identify_outputs,id_labels)
        _, predicted = torch.max(identify_outputs, 1)
        total += id_labels.size(0)
        correct += (predicted == id_labels).sum().item()
        identify_acc = correct/total
        # loss调参部分
        a1 = 1
        a2 = 0.2
        a3 = 0.03
        a6 = 0.9
        a7 = 1
        # if epoch < 10:
        #     loss = loss_i
        # if epoch < 100:
        #     loss = a1 * loss_w + a2 * loss_i 
        if epoch < 400:
            loss = a6 * loss_w + a7 * loss_i 
        else:
            loss = loss_w +loss_i
        # 梯度反传
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 评价指标计算（bit_acc)        
        ori_msgs = torch.sign(msgs) > 0
        decoded_msgs = torch.sign(fts) > 0 # b k -> b k
        diff = (~torch.logical_xor(ori_msgs, decoded_msgs)) # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        # 计算是否符合触发集的样本
        if bit_accs.mean() > 0.8:
            wrong_predictions = test_model(wrong_predictions,imgs_w, task_labels, task_model,id_labels,identify_model,device)
         # 评价指标计算# 计算SSIM# 计算PSNR
        watermarked_signals = imgs_w
        original_signals = imgs
        watermarked_signals = torch.squeeze(watermarked_signals)
        original_signals = torch.squeeze(original_signals)
        original_signals = original_signals.detach().cpu().numpy()
        min_val = np.min(original_signals, axis=0)
        max_val = np.max(original_signals, axis=0)
        normalized_or_data = (original_signals - min_val) / (max_val - min_val)
        watermarked_signals = watermarked_signals.detach().cpu().numpy()
        normalized_water_data = (watermarked_signals - min_val) / (max_val - min_val)
        result = normalized_or_data - normalized_water_data
        # avg_ssim = calculate_batch_ssim_new(original_signals, watermarked_signals)
        # avg_psnr = calculate_batch_psnr_new(original_signals, watermarked_signals)
        # 打印输出结果
        log_stats = {
            'loss': loss.item(),
            'water_loss':loss_w.item(),
            'EEG_loss':loss_i.item(),
            'identify_loss':loss_identify.item(),
            'identify_acc':identify_acc,
            'bit_acc_avg': torch.mean(bit_accs).item(),
            'lr': optimizer.param_groups[0]['lr'],
            # 'PSNR' : avg_psnr.item(),
            # 'SSIM' : avg_ssim.item(),
        }
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})
        # 每个epoch 只保存 前5个符合的数据 每个的大小为300
        if count < 5:
            if len(wrong_predictions) > 300:
                count += 1
                save_path = output_dir + f'/wrong_predictions_{epoch}_{it}.pkl'
                import pickle
                with open(save_path, 'wb') as f:
                    pickle.dump(wrong_predictions, f)
                    print(f'successful save triggerest dataset in {save_path}')
    print("Averaged {} stats:".format('train'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def image_loss(imgs, imgs_ori, loss_type='mse'):
    """
    Compute the image loss
    Args:
        imgs (BxCxHxW): the reconstructed images
        imgs_ori (BxCxHxW): the original images
        loss_type: the type of loss
    """
    if loss_type == 'mse':
        return F.mse_loss(imgs, imgs_ori, reduction='mean')
    if loss_type == 'l1':
        return F.l1_loss(imgs, imgs_ori, reduction='mean')
    else:
        raise ValueError('Unknown loss type')
    
if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)