import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
from noiser import Noiser
from scipy.fftpack import dct, idct
# import pywt
from collections import OrderedDict


class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self,num_blocks, channels):
        super(Discriminator, self).__init__()

        layers = [ConvBNRelu(1, channels)]
        for _ in range(num_blocks):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(channels, 1)

    def forward(self, eeg):
        if len(eeg.shape) == 3:
            eeg = torch.unsqueeze(eeg,1)

        X = self.before_linear(eeg)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X)
        # X = torch.sigmoid(X)
        return X

class HiddenEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, num_blocks, num_bits, channels, last_tanh=True):
        super(HiddenEncoder, self).__init__()
        layers = [ConvBNRelu(1, channels)]

        for _ in range(num_blocks-1):
            layer = ConvBNRelu(channels, channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(channels + 1 + num_bits, channels)

        self.final_layer = nn.Conv2d(channels, 1, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):
        # 传入 图像以及水印二值数据
        msgs = msgs.unsqueeze(-1).unsqueeze(-1) # b l 1 1
        msgs = msgs.expand(-1,-1, imgs.size(-2), imgs.size(-1)) # b l h w

        encoded_image = self.conv_bns(imgs) # b c h w

        # 数据进行拼接 二值水印数据 编码的图像 源图像
        concat = torch.cat([msgs, encoded_image, imgs], dim=1) # b l+c+1 h w
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        # if self.last_tanh:
        im_w = self.tanh(im_w)

        return im_w

class HiddenDecoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, num_blocks, num_bits, channels):

        super(HiddenDecoder, self).__init__()

        # CCNN模型的输入通道数是4
        
        layers = [ConvBNRelu(1, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits, num_bits)

    def forward(self, img_w):

        x = self.layers(img_w) # b d 1 1
        x = x.squeeze(-1).squeeze(-1) # b d
        x = self.linear(x) # b d
        return x

class DvmarkDecoder(nn.Module):
    def __init__(self, num_blocks, num_bits, channels, last_sigmoid=True):
        super(DvmarkDecoder, self).__init__()
        
        # upsample x2 to reverse the downsample operation
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # conv layers for upsampled
        num_blocks_scale2 = 3
        scale2_layers = [ConvBNRelu(1, channels*2)]
        for _ in range(num_blocks_scale2-1):
            layer = ConvBNRelu(channels*2, channels*2)
            scale2_layers.append(layer)
        self.scale2_layers = nn.Sequential(*scale2_layers)

        # upsample x2 to match the original image size
        self.upsample_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # conv layers for original scale
        num_blocks_scale1 = 3
        scale1_layers = [ConvBNRelu(channels*2, channels)]
        for _ in range(num_blocks_scale1-1):
            layer = ConvBNRelu(channels, channels)
            scale1_layers.append(layer)
        self.scale1_layers = nn.Sequential(*scale1_layers)
        
        # final conv layer to get the image
        self.final_layer_img = nn.Conv2d(channels, 4, kernel_size=1)
        
        # final conv layer to extract the message
        self.final_layer_msg = nn.Conv2d(channels, num_bits, kernel_size=1)
        
        self.last_sigmoid = last_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, im_w):
        # Reverse the encoder's downsample and fusion process
        scale2 = self.upsample(im_w) # b c*2 h*2 w*2
        scale2 = self.scale2_layers(scale2) # b c*2 h*2 w*2
        
        # Upsample to original resolution
        scale1 = self.upsample_final(scale2) # b c*2 h*4 w*4
        scale1 = self.scale1_layers(scale1) # b c h*4 w*4
        
        # Recover the image
        img_rec = self.final_layer_img(scale1) # b 1 h*4 w*4
        if self.last_sigmoid:
            img_rec = self.sigmoid(img_rec)

        # Extract the message
        msg_rec = self.final_layer_msg(scale1) # b num_bits h*4 w*4
        msg_rec = torch.mean(msg_rec, dim=(2, 3)) # b num_bits (average over spatial dimensions)

        return msg_rec

class EncoderDecoder(nn.Module):
    def __init__(
            self,
            nosier: Noiser,
            encoder: nn.Module,
            decoder: nn.Module,
            scale_channels: bool,
            scaling_i: float,
            scaling_w: float,
            num_bits: int,
            redundancy: int
    ):
        super().__init__()
        self.nosier = nosier
        self.encoder = encoder
        self.decoder = decoder
        # params for the forward pass
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w
        self.num_bits = num_bits
        self.redundancy = redundancy

    def forward(
            self,
            imgs: torch.Tensor,
            msgs: torch.Tensor,
            eval_mode: bool = False,
            eval_aug: nn.Module = nn.Identity(),
    ):
        """
        Does the full forward pass of the encoder-decoder network:
        - encodes the message into the image
        - attenuates the watermark
        - augments the image
        - decodes the watermark

        Args:
            imgs: b c h w
            msgs: b l
        """
        # imgs = torch.unsqueeze(imgs,1)
        # heatmap = torch.unsqueeze(heatmap,1)
        # encoder
        deltas_w = self.encoder(imgs, msgs)  # b c h w

        # add heatmaps
        # if self.attenuation is not None:
        #     deltas_w = deltas_w * heatmap  # # b c h w * b 1 h w -> b c h w
        imgs_w = self.scaling_i * imgs + self.scaling_w * deltas_w  # b c h w

        # data augmentation
        if eval_mode:
            imgs_aug = eval_aug(imgs_w)
            # imgs_aug = self.nosier(imgs_aug)
            fts = self.decoder(imgs_aug)  # b c h w -> b d
        else:
            # imgs_aug = self.nosier(imgs_w)
            imgs_aug = imgs_w
            fts = self.decoder(imgs_aug)  # b c h w -> b d

        fts = fts.view(-1, self.num_bits, self.redundancy)  # b k*r -> b k r
        fts = torch.sum(fts, dim=-1)  # b k r -> b k

        return fts, (imgs_w, imgs_aug),self.decoder

class DvmarkEncoder(nn.Module):

    def __init__(self, num_blocks, num_bits, channels, last_tanh=True):
        super(DvmarkEncoder, self).__init__()

        transform_layers = [ConvBNRelu(1, channels)]
        for _ in range(num_blocks-1):
            layer = ConvBNRelu(channels, channels)
            transform_layers.append(layer)
        self.transform_layers = nn.Sequential(*transform_layers)

        # conv layers for original scale
        num_blocks_scale1 = 3
        scale1_layers = [ConvBNRelu(channels+num_bits, channels*2)]
        for _ in range(num_blocks_scale1-1):
            layer = ConvBNRelu(channels*2, channels*2)
            scale1_layers.append(layer)
        self.scale1_layers = nn.Sequential(*scale1_layers)

        # downsample x2
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # conv layers for downsampled
        num_blocks_scale2 = 3
        scale2_layers = [ConvBNRelu(channels*2+num_bits, channels*4), ConvBNRelu(channels*4, channels*2)]
        for _ in range(num_blocks_scale2-2):
            layer = ConvBNRelu(channels*2, channels*2)
            scale2_layers.append(layer)
        self.scale2_layers = nn.Sequential(*scale2_layers)

        # upsample x2
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.final_layer = nn.Conv2d(channels*2, 1, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):

        encoded_image = self.transform_layers(imgs) # b c h w

        msgs = msgs.unsqueeze(-1).unsqueeze(-1) # b l 1 1

        scale1 = torch.cat([msgs.expand(-1,-1, imgs.size(-2), imgs.size(-1)), encoded_image], dim=1) # b l+c h w
        scale1 = self.scale1_layers(scale1) # b c*2 h w

        scale2 = self.avg_pool(scale1) # b c*2 h/2 w/2
        scale2 = torch.cat([msgs.expand(-1,-1, imgs.size(-2)//2, imgs.size(-1)//2), scale2], dim=1) # b l+c*2 h/2 w/2
        scale2 = self.scale2_layers(scale2) # b c*2 h/2 w/2

        scale1 = scale1 + self.upsample(scale2) # b c*2 h w
        im_w = self.final_layer(scale1) # b 3 h w

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out,3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out, eps=1e-3),
            nn.GELU()
        )
    def forward(self, x):
        return self.layers(x)

class InceptionModule(nn.Module):
    def __init__(self, in_nc, out_nc,nums_bit=30, bn='BatchNorm'):
        super(InceptionModule, self).__init__()
        self.conv1x1 = nn.Conv2d(in_nc, out_nc//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm1x1 = nn.BatchNorm2d(out_nc//4)

        self.conv1x1_2 = nn.Conv2d(in_nc, out_nc//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(out_nc//4, out_nc//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm3x3 = nn.BatchNorm2d(out_nc//4)  # nn.Sequential()  #

        self.conv1x1_3 = nn.Conv2d(in_nc, out_nc//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv5x5 = nn.Conv2d(out_nc//4, out_nc//4, kernel_size=5, stride=1, padding=2, bias=False)
        self.norm5x5 = nn.BatchNorm2d(out_nc//4)

        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1x1_4 = nn.Conv2d(in_nc, out_nc//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.normpooling = nn.BatchNorm2d(out_nc//4)

        self.conv1x1_5 = nn.Conv2d(in_nc, out_nc, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.LeakyReLU()
        self.cat = ConvBNRelu(out_nc+nums_bit, out_nc)

    def forward(self, x,secret):
        out1x1 = self.relu(self.norm1x1(self.conv1x1(x)))
        out3x3 = self.relu(self.conv1x1_2(x))
        out3x3 = self.relu(self.norm3x3(self.conv3x3(out3x3)))
        out5x5 = self.relu(self.conv1x1_3(x))
        out5x5 = self.relu(self.norm5x5(self.conv5x5(out5x5)))
        outmaxpooling = self.maxpooling(x)
        outmaxpooling = self.relu(self.norm5x5(self.conv1x1_4(outmaxpooling)))

        out = torch.cat([out1x1, out3x3, out5x5, outmaxpooling], dim=1)
        residual = self.conv1x1_5(x)
        out = out + residual
        out = torch.cat([out, secret], dim=1)
        out = self.cat(out)
        return out

class HidingNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, num_bits=30):
        super(HidingNet, self).__init__()
        self.conv1 = nn.Conv2d(31, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(16)
        self.block1 = InceptionModule(in_nc=16, out_nc=32)
        self.block2 = InceptionModule(in_nc=32, out_nc=64)
        self.block3 = InceptionModule(in_nc=64, out_nc=128)
        self.block4 = InceptionModule(in_nc=128, out_nc=64)
        self.block5 = InceptionModule(in_nc=64, out_nc=32)
        self.block6 = InceptionModule(in_nc=32, out_nc=16)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(3)
        self.conv3 = nn.Conv2d(3, out_nc, kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.LeakyReLU()  # nn.ReLU(True)
        self.tanh = nn.Tanh()
        # self.init_weight()

    def forward(self, cover,secret):
        # convert cover from rgb to yuv
        Y = cover
        # Y = Y.unsqueeze(1)
        secret = secret.unsqueeze(-1).unsqueeze(-1) # b l 1 1
        secret = secret.expand(-1,-1, cover.size(-2), cover.size(-1)) # b l h w
        # secret = F.interpolate(secret, size=(22, 1000), mode='bilinear', align_corners=False)

        x = torch.cat([secret, Y], dim=1)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.block1(out,secret)
        out = self.block2(out,secret)
        out = self.block3(out,secret)
        out = self.block4(out,secret)
        out = self.block5(out,secret)
        out = self.block6(out,secret)
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.tanh(self.conv3(out))
        cover = out
        return cover

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.xavier_uniform(m.weight)

class RevealNet(nn.Module):
    def __init__(self, nc=1, nhf=22, num_bits=30):
        super(RevealNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf * 2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf * 4),
            nn.ReLU(True),
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf * 2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nhf, 3, 1, 1),
            # output_function()
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            
            )
        self.linear = nn.Linear(nhf, num_bits)
        
    def forward(self, stego):
        # Y = 0 + 0.299 * stego[:, 0, :, :] + 0.587 * stego[:, 1, :, :] + 0.114 * stego[:, 2, :, :]
        # Y = Y.unsqueeze(1)
        output = self.main(stego)
        # x = self.layers(img_w) # b d 1 1
        output = output.squeeze(-1).squeeze(-1) # b d
        output = self.linear(output) # b d
        return output

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.xavier_uniform(m.weight)

def get_model_identify(architecture):
    match architecture:
        case "CCNN":
            from torcheeg.models import CCNN

            return CCNN(num_classes=32, in_channels=4, grid_size=(9, 9))

        case "TSCeption":
            from torcheeg.models import TSCeption

            return TSCeption(
                num_classes=32,
                num_electrodes=28,
                sampling_rate=128,
                num_T=60,
                num_S=60,
                hid_channels=128,
                dropout=0.5,
            )

        case "EEGNet":
            from torcheeg.models import EEGNet

            return EEGNet(
                chunk_size=128,
                num_electrodes=32,
                dropout=0.5,
                kernel_1=64,
                kernel_2=16,
                F1=16,
                F2=32,
                D=4,
                num_classes=32,
            )

        case _:
            raise ValueError(f"Invalid architecture: {architecture}")

def get_model(architecture):
    match architecture:
        case "CCNN":
            from torcheeg.models import CCNN

            return CCNN(num_classes=16, in_channels=4, grid_size=(9, 9))

        case "TSCeption":
            from torcheeg.models import TSCeption

            return TSCeption(
                num_classes=16,
                num_electrodes=28,
                sampling_rate=128,
                num_T=60,
                num_S=60,
                hid_channels=128,
                dropout=0.5,
            )

        case "EEGNet":
            from torcheeg.models import EEGNet

            return EEGNet(
                chunk_size=128,
                num_electrodes=32,
                dropout=0.5,
                kernel_1=64,
                kernel_2=16,
                F1=16,
                F2=32,
                D=4,
                num_classes=16,
            )

        case _:
            raise ValueError(f"Invalid architecture: {architecture}")