import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
import math
import numpy as np
import functools
from timm.models.layers import trunc_normal_
# from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
from metrics_2d import init_weights, count_param
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x, y):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return y * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x, y):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return y * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        self.SpatialGate = SpatialGate()
    def forward(self, x, y):
        y_out = self.ChannelGate(x, y)
        y_out = self.SpatialGate(x, y_out)
        return y_out
'''
## Channel Attention (CA) Layer
class CALayer(nn.Module):  # CA层设定
    def __init__(self, channel, reduction=16):  # 默认输入参数为通道数以及通道下降比例参数（用以计算通道下降输出通道数目）
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 通道均值
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),  # 卷积层，依次为输入通道数，输出通道数，步长，padding，bias
            nn.ReLU(inplace=True),  # relu
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )  # nn.Sequential将设定的几种操作串行起来

    def forward(self, x):
        # print('RCABsize',x.size())
        y = self.avg_pool(x)  # 对输入的数据进行均值操作
        y = self.conv_du(y)  # 进行conv_du操作
        # print('CALayer_ysize',y.size())
        return x * y
## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, n_feat, reduction = 16, bn=False, act=nn.ReLU(True)):  # 默认参数为卷积函数，通道数，卷积核大小，通道下降比例参数，空洞卷积参数

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):  # 设定残差块结构 conv - relu - conv
            modules_body.append(nn.Conv2d(n_feat, n_feat, 3,stride=1, padding=1))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        self.body = nn.Sequential(*modules_body)  # nn.Sequential将body中设定的几种操作串行起来
        self.tail = nn.Sequential(CALayer(n_feat, reduction))  # 残差块结构后跟CA层操作


    def forward(self, x):
        # print('RCABsize',x.size())
        res1 = self.body(x)  # 进行body中串行操作
        res2 = self.tail(res1)
        res = 1 * x + 1 * res1 + 0.5 * res2


        return res
'''
class SELayer2d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class unet2dConv2d(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, n=2, ks=3, stride=1, padding=1):
        super(unet2dConv2d, self).__init__()
        self.n = n  # 第n层
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, stride=stride, padding=padding),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)  # 如果属性不存在会创建一个新的对象属性，并对属性赋值
                # setattr(object, name, value)
                # object - - 对象。
                # name - - 字符串，对象属性。
                # value - - 属性值。
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, stride=stride, padding=padding),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)  # getattr函数用于返回一个对象属性值。
            x = conv(x)

        return x


class unet2dUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unet2dUp, self).__init__()
        self.conv = unet2dConv2d(in_size, out_size, is_batchnorm=True)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size, out_size, 1))

        # initialise the blocks
        for m in self.children():

            if m.__class__.__name__.find('unet2dConv2d') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)


class model_input(nn.Module):
    def __init__(self, in_channels_t2, in_channels_fa, in_channels_dec, in_channels_peaks, filters_base,reduction=1,
                 norm_layer=nn.BatchNorm2d):
        super(model_input, self).__init__()
        self.in_channels_t2 = in_channels_t2  # 通道数
        self.in_channels_fa = in_channels_fa  # 通道数
        self.in_channels_dec = in_channels_dec  # 通道数
        self.in_channels_peaks = in_channels_peaks  # 通道数
        # self.out_channels = out_channels
        self.t2 = nn.Conv2d(self.in_channels_t2, filters_base, 3, 1, 1)
        self.fa = nn.Conv2d(self.in_channels_fa, filters_base, 3, 1, 1)
        self.dec = nn.Conv2d(self.in_channels_dec, filters_base, 3, 1, 1)
        self.peaks = nn.Conv2d(self.in_channels_peaks, filters_base, 3, 1, 1)
        # self.fusion = nn.Conv2d(32, 32, 3, 1,1)
        # self.modal_sel = self.PredictorConv(32)

    def forward(self, inputs_t2, inputs_fa, inputs_dec, inputs_peaks):
        x1 = self.t2(inputs_t2)
        x2 = self.fa(inputs_fa)
        x3 = self.dec(inputs_dec)
        x4 = self.peaks(inputs_peaks)
        modal_list = [x1, x2, x3, x4]
        # out = self.fusion(modal_list)
        return modal_list


class PredictorConv(nn.Module):
    def __init__(self, embed_dim, num_modals = 4):
        super().__init__()
        self.num_modals = num_modals
        self.score_nets_t1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1, bias=False),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=(embed_dim)),
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        ) for _ in range(num_modals)])

    def forward(self, x,x_t1):
        x_t1_weights = self.score_nets_t1(x_t1)
        B, C, H, W = x[0].shape
        x_ = [torch.zeros((B, 1, H, W)) for _ in range(self.num_modals)]
        for i in range(self.num_modals):
            x_[i] = self.score_nets[i](x[i])
        return x_,x_t1_weights


class tokenselect(nn.Module):
    def __init__(self, embed_dim, num_modals):
        super().__init__()
        self.predict = PredictorConv(embed_dim,  num_modals)

    def forward(self, x_ext,x_t1):
        # x_ext = self.predict(x_ext)
        x_scores, x_t1_scores = self.predict(x_ext,x_t1)

        for i in range(len(x_ext)):
            # y = torch.zeros_like(x_ext[i])
            x_ext[i] = x_scores[i] * x_ext[i] * x_t1_scores + x_ext[i]


        x_f = functools.reduce(torch.max, x_ext)

        return x_f,x_t1*x_t1_scores
class tokenselect_mask(nn.Module):
    def __init__(self, embed_dim, num_modals):
        super().__init__()
        self.num_modals = num_modals
        self.predict = PredictorConv(embed_dim,  num_modals)

    def forward(self, x_ext,x_t1,mask):
        # x_ext = self.predict(x_ext)
        x_scores, x_t1_scores = self.predict(x_ext,x_t1)
        # B, C, H, W = x_ext[0].shape
        # x_f = torch.zeros((B, C, H, W))
        # y = torch.zeros_like(x1)
        # print(x1[1, ...].size())
        # y[mask2, ...] = x1[mask2, ...]
        # y_ext = numpy.zeros_like(x_ext)
        B, C, H, W = x_ext[0].shape
        y_ext = [torch.zeros((B, C, H, W)).to(device) for _ in range(self.num_modals)]
        for i in range(len(x_ext)):
            # y_ext = torch.zeros_like(x_ext[i])
            x_ext[i] = x_scores[i] * x_ext[i] * x_t1_scores + x_ext[i]
            s = mask[:, i:i + 1].squeeze()
            # x_ext[i] = x_t1_scores * x_ext[i] + x_ext[i]
            y_ext[i][mask[:, i:i + 1].squeeze(), ...] = x_ext[i][mask[:, i:i + 1].squeeze(), ...]

        x_f = functools.reduce(torch.max, y_ext)

        return x_f,x_t1*x_t1_scores
class tokenselect_mask_ablation_arm(nn.Module):
    def __init__(self, embed_dim, num_modals):
        super().__init__()
        self.num_modals = num_modals
        self.predict = PredictorConv(embed_dim,  num_modals)

    def forward(self, x_ext,x_t1,mask):
        # x_ext = self.predict(x_ext)
        x_scores, _ = self.predict(x_ext,x_t1)
        # B, C, H, W = x_ext[0].shape
        # x_f = torch.zeros((B, C, H, W))
        # y = torch.zeros_like(x1)
        # print(x1[1, ...].size())
        # y[mask2, ...] = x1[mask2, ...]
        # y_ext = numpy.zeros_like(x_ext)
        B, C, H, W = x_ext[0].shape
        y_ext = [torch.zeros((B, C, H, W)).to(device) for _ in range(self.num_modals)]
        for i in range(len(x_ext)):
            # y_ext = torch.zeros_like(x_ext[i])
            x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]
            # s = mask[:, i:i + 1].squeeze()
            # x_ext[i] = x_t1_scores * x_ext[i] + x_ext[i]
            y_ext[i][mask[:, i:i + 1].squeeze(), ...] = x_ext[i][mask[:, i:i + 1].squeeze(), ...]

        x_f = functools.reduce(torch.max, y_ext)

        return x_f,x_t1
### FIM
# class sSE(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
#         self.norm = nn.Sigmoid()
#
#     def forward(self, U):
#         q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
#         q = self.norm(q)
#         return U * q  # 广播机制
# class cSE(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
#         self.Conv_Excitation = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False)
#         self.norm = nn.Sigmoid()
#
#     def forward(self, U):
#         z = self.avgpool(U)  # shape: [bs, c, h, w] to [bs, c, 1, 1]
#         z = self.Conv_Squeeze(z)  # shape: [bs, c/2]
#         z = self.Conv_Excitation(z)  # shape: [bs, c]
#         z = self.norm(z)
#         return U * z.expand_as(U)
# class csSE(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.cSE = cSE(in_channels)
#         self.sSE = sSE(in_channels)
#
#     def forward(self, U1, U2):
#         U1_sse = self.sSE(U1)
#         U2_cse = self.cSE(U2)
#         U_sse = U1_sse + U2_cse
#         return U1_sse,U2_cse,U_sse
class T1_layer(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.conv1_t1 = unet2dConv2d(inputs, outputs)
        # self.se1 = SELayer2d(outputs)
        self.maxpool_t1 = nn.MaxPool2d(2)
    def forward(self, x1):
        x1 = self.maxpool_t1(x1)
        x2 = self.conv1_t1(x1)
        # x3 = self.se1(x2)

        return x2

class Sharelayer(nn.Module):
    def __init__(self, inputs, outputs,num_modals):
        super().__init__()
        self.num_modals = num_modals
        self.otherlayer = nn.ModuleList([nn.Sequential(nn.MaxPool2d(2),
        unet2dConv2d(inputs, outputs),
        # SELayer2d(outputs),
        ) for _ in range(num_modals)])
        # self.conv1_t1 = unet2dConv2d(inputs, outputs, self.is_batchnorm)
        # self.se1 = SELayer2d(outputs)
        # self.maxpool_t1 = nn.MaxPool2d(2)

    def forward(self, x):
        B, C, H, W = x[0].shape
        x_ = [torch.zeros((B, C, H, W)) for _ in range(self.num_modals)]
        for i in range(self.num_modals):
            x_[i] = self.otherlayer[i](x[i])
        # x = self.otherlayer(x)
        return x_

class T1_layer_first(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.conv1_t1 = unet2dConv2d(inputs, outputs)
        # self.se1 = SELayer2d(outputs)
        # self.maxpool_t1 = nn.MaxPool2d(2)
    def forward(self, x1):
        x2 = self.conv1_t1(x1)
        # x3 = self.se1(x2)
        # x4 = self.maxpool_t1(x3)
        return x2

class Sharelayer_first(nn.Module):
    def __init__(self, inputs, outputs,num_modals):
        super().__init__()
        self.num_modals = num_modals
        self.otherlayer = nn.ModuleList([nn.Sequential(
        unet2dConv2d(inputs, outputs),
        # SELayer2d(outputs),
        # nn.MaxPool2d(2),
        ) for _ in range(num_modals)])
        # self.conv1_t1 = unet2dConv2d(inputs, outputs, self.is_batchnorm)
        # self.se1 = SELayer2d(outputs)
        # self.maxpool_t1 = nn.MaxPool2d(2)

    def forward(self, x):
        B, C, H, W = x[0].shape
        x_ = [torch.zeros((B, C, H, W)) for _ in range(self.num_modals)]
        for i in range(self.num_modals):
            x_[i] = self.otherlayer[i](x[i])
        # x = self.otherlayer(x)
        return x_
class ConvBn2d_1(nn.Module):
    # convolution
    # batch normalization
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(ConvBn2d_1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=padding,
                              stride=(stride, stride), dilation=(dilation, dilation), groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.bn(self.conv(x))

        #return self.conv(x)


class CDA(nn.Module):
    def __init__(self, out_dim):
        super(CDA, self).__init__()

        act_fn = nn.ReLU(inplace=True)

        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.rgb_d = CBAM(out_dim, 1)
        self.d_rgb = CBAM(out_dim, 1)

        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(out_dim), act_fn, )

    def forward(self, rgb, depth):
        ################################

        x_rgb1 = self.layer_10(rgb)
        x_dep1 = self.layer_20(depth)

        ## fusion
        x_dep_r = self.rgb_d(x_rgb1, x_dep1)
        x_rgb_r = self.d_rgb(x_dep1, x_rgb1)

        x_cat = torch.cat((x_rgb_r, x_dep_r), dim=1)
        out1 = self.layer_ful1(x_cat)
        # out1 = x_rgb1 + x_dep1
        return out1
class CNTSegV2_ARM(nn.Module):
    def __init__(self, in_channels_t1=1, in_channels_t2=1,in_channels_fa=1,in_channels_dec=3,in_channels_peaks=9,n_classes=5, is_deconv=True, is_batchnorm=True):
        super(CNTSegV2_ARM, self).__init__()

        self.in_channels_t1 = in_channels_t1  # 通道数
        self.in_channels_t2 = in_channels_t2  # 通道数
        self.in_channels_fa = in_channels_fa  # 通道数
        self.in_channels_dec = in_channels_dec  # 通道数
        self.in_channels_peaks = in_channels_peaks  # 通道数
        # self.in_channels_other = in_channels_t2 + in_channels_fa + in_channels_dec + in_channels_peaks
        filters_base = 32
        modality = 4
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input_modal = model_input(in_channels_t2, in_channels_fa, in_channels_dec, in_channels_peaks,filters_base)

        self.t1layer1 = T1_layer_first(self.in_channels_t1,filters_base)
        self.t1layer2 = T1_layer(filters_base, filters_base * 2)
        self.t1layer3 = T1_layer(filters_base * 2, filters_base * 4)
        self.t1layer4 = T1_layer(filters_base * 4, filters_base * 8)

        self.otherlayer1 = Sharelayer_first(filters_base, filters_base, modality)
        self.otherlayer2 = Sharelayer(filters_base, filters_base * 2, modality)
        self.otherlayer3 = Sharelayer(filters_base * 2, filters_base * 4, modality)
        self.otherlayer4 = Sharelayer(filters_base * 4, filters_base * 8, modality)

        self.select_modal_1 = tokenselect(filters_base, modality)
        self.select_modal_2 = tokenselect(filters_base * 2, modality)
        self.select_modal_3 = tokenselect(filters_base * 4, modality)
        self.select_modal_4 = tokenselect(filters_base * 8, modality)

        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)
        # self.csSE1 = csSE(filters_base)
        # self.csSE2 = csSE(filters_base * 2)
        # self.csSE3 = csSE(filters_base * 4)
        # self.csSE4 = csSE(filters_base * 8)
        ## CrossAttention With T1,FA,Peaks
        # self.ca1 = FeatureRectifyModule(filters_base)
        # self.ca1 = FeatureRectifyModule(filters_base, reduction=1)
        # self.ca2 = FeatureRectifyModule(filters_base * 2, reduction=1)
        # self.ca3 = FeatureRectifyModule(filters_base * 4, reduction=1)
        # self.ca4 = FeatureRectifyModule(filters_base * 8, reduction=1)
        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)

        # self.fusion = nn.Conv2d(filters_base * 16, filters_base * 8, 3, 1,1)
        # upsampling
        self.up1 = nn.ConvTranspose2d(filters_base* 8, filters_base*4, kernel_size=2, stride=2, padding=0)
        self.upconv1 = unet2dConv2d(filters_base*4*2, filters_base*4, self.is_batchnorm)
        self.up2 = nn.ConvTranspose2d(filters_base*4, filters_base*2, kernel_size=2, stride=2, padding=0)
        self.upconv2 = unet2dConv2d(filters_base*2*2, filters_base*2, self.is_batchnorm)
        self.up3 = nn.ConvTranspose2d(filters_base*2, filters_base, kernel_size=2, stride=2, padding=0)
        self.upconv3 = unet2dConv2d(filters_base*2, filters_base, self.is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters_base, n_classes, 1)
        # self.avg_pool_islands = nn.AdaptiveAvgPool2d(1)
        # self.linear = nn.Linear(filters_base* 8, n_classes-1)
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs_t1, inputs_t2, inputs_fa, inputs_dec, inputs_peaks):
        other_modal = self.input_modal(inputs_t2, inputs_fa, inputs_dec, inputs_peaks)
        layer1_t1 = self.t1layer1(inputs_t1)
        layer1_other_modal = self.otherlayer1(other_modal)
        layer1_other_modal_select,layer1_t1_after = self.select_modal_1(layer1_other_modal,layer1_t1)
        cal1_t1fa = layer1_t1_after+layer1_other_modal_select
        # cal1_t1fa = self.fusion1(layer1_t1_after, layer1_other_modal_select)

        layer2_t1 = self.t1layer2(cal1_t1fa)
        layer2_other_modal = self.otherlayer2(layer1_other_modal)
        layer2_other_modal_select,layer2_t1_after = self.select_modal_2(layer2_other_modal,layer2_t1)
        cal2_t1fa = layer2_t1_after+layer2_other_modal_select
        # cal2_t1fa = self.fusion2(layer2_t1_after, layer2_other_modal_select)

        layer3_t1 = self.t1layer3(cal2_t1fa)
        layer3_other_modal = self.otherlayer3(layer2_other_modal)
        layer3_other_modal_select,layer3_t1_after = self.select_modal_3(layer3_other_modal,layer3_t1)
        cal3_t1fa = layer3_t1_after+layer3_other_modal_select
        # cal3_t1fa = self.fusion3(layer3_t1_after, layer3_other_modal_select)

        layer4_t1 = self.t1layer4(cal3_t1fa)
        layer4_other_modal = self.otherlayer4(layer3_other_modal)
        layer4_other_modal_select,layer4_t1_after = self.select_modal_4(layer4_other_modal,layer4_t1)
        cal4_t1fa = layer4_t1_after+layer4_other_modal_select
        # cal4_t1fa = self.fusion4(layer4_t1_after, layer4_other_modal_select)


        up1 = self.up1(cal4_t1fa)
        cat1 = torch.cat([up1, cal3_t1fa], 1)
        upcon1 = self.upconv1(cat1)

        up2 = self.up2(upcon1)
        cat2 = torch.cat([up2, cal2_t1fa], 1)
        upcon2 = self.upconv2(cat2)

        up3 = self.up3(upcon2)
        cat3 = torch.cat([up3, cal1_t1fa], 1)
        upcon3 = self.upconv3(cat3)


        final = self.final(upcon3)

        return final

class CNTSegV2_ARM_CDA(nn.Module):
    def __init__(self, in_channels_t1=1, in_channels_t2=1,in_channels_fa=1,in_channels_dec=3,in_channels_peaks=9,n_classes=5, is_deconv=True, is_batchnorm=True):
        super(CNTSegV2_ARM_CDA, self).__init__()

        self.in_channels_t1 = in_channels_t1  # 通道数
        self.in_channels_t2 = in_channels_t2  # 通道数
        self.in_channels_fa = in_channels_fa  # 通道数
        self.in_channels_dec = in_channels_dec  # 通道数
        self.in_channels_peaks = in_channels_peaks  # 通道数
        # self.in_channels_other = in_channels_t2 + in_channels_fa + in_channels_dec + in_channels_peaks
        filters_base = 32
        modality = 4
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input_modal = model_input(in_channels_t2, in_channels_fa, in_channels_dec, in_channels_peaks,filters_base)

        self.t1layer1 = T1_layer_first(self.in_channels_t1,filters_base)
        self.t1layer2 = T1_layer(filters_base, filters_base * 2)
        self.t1layer3 = T1_layer(filters_base * 2, filters_base * 4)
        self.t1layer4 = T1_layer(filters_base * 4, filters_base * 8)

        self.otherlayer1 = Sharelayer_first(filters_base, filters_base, modality)
        self.otherlayer2 = Sharelayer(filters_base, filters_base * 2, modality)
        self.otherlayer3 = Sharelayer(filters_base * 2, filters_base * 4, modality)
        self.otherlayer4 = Sharelayer(filters_base * 4, filters_base * 8, modality)

        self.select_modal_1 = tokenselect(filters_base, modality)
        self.select_modal_2 = tokenselect(filters_base * 2, modality)
        self.select_modal_3 = tokenselect(filters_base * 4, modality)
        self.select_modal_4 = tokenselect(filters_base * 8, modality)

        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)
        self.fusion1 = CDA(filters_base)
        self.fusion2 = CDA(filters_base * 2)
        self.fusion3 = CDA(filters_base * 4)
        self.fusion4 = CDA(filters_base * 8)
        ## CrossAttention With T1,FA,Peaks
        # self.ca1 = FeatureRectifyModule(filters_base)
        # self.ca1 = FeatureRectifyModule(filters_base, reduction=1)
        # self.ca2 = FeatureRectifyModule(filters_base * 2, reduction=1)
        # self.ca3 = FeatureRectifyModule(filters_base * 4, reduction=1)
        # self.ca4 = FeatureRectifyModule(filters_base * 8, reduction=1)
        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)

        # self.fusion = nn.Conv2d(filters_base * 16, filters_base * 8, 3, 1,1)
        # upsampling
        self.up1 = nn.ConvTranspose2d(filters_base* 8, filters_base*4, kernel_size=2, stride=2, padding=0)
        self.upconv1 = unet2dConv2d(filters_base*4*2, filters_base*4, self.is_batchnorm)
        self.up2 = nn.ConvTranspose2d(filters_base*4, filters_base*2, kernel_size=2, stride=2, padding=0)
        self.upconv2 = unet2dConv2d(filters_base*2*2, filters_base*2, self.is_batchnorm)
        self.up3 = nn.ConvTranspose2d(filters_base*2, filters_base, kernel_size=2, stride=2, padding=0)
        self.upconv3 = unet2dConv2d(filters_base*2, filters_base, self.is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters_base, n_classes, 1)
        # self.avg_pool_islands = nn.AdaptiveAvgPool2d(1)
        # self.linear = nn.Linear(filters_base* 8, n_classes-1)
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs_t1, inputs_t2, inputs_fa, inputs_dec, inputs_peaks):
        other_modal = self.input_modal(inputs_t2, inputs_fa, inputs_dec, inputs_peaks)
        layer1_t1 = self.t1layer1(inputs_t1)
        layer1_other_modal = self.otherlayer1(other_modal)
        layer1_other_modal_select,layer1_t1_after = self.select_modal_1(layer1_other_modal,layer1_t1)
        cal1 = layer1_t1_after+layer1_other_modal_select
        cal1_t1fa = self.fusion1(layer1_t1_after, layer1_other_modal_select)

        layer2_t1 = self.t1layer2(cal1)
        layer2_other_modal = self.otherlayer2(layer1_other_modal)
        layer2_other_modal_select,layer2_t1_after = self.select_modal_2(layer2_other_modal,layer2_t1)
        cal2 = layer2_t1_after+layer2_other_modal_select
        cal2_t1fa = self.fusion2(layer2_t1_after, layer2_other_modal_select)

        layer3_t1 = self.t1layer3(cal2)
        layer3_other_modal = self.otherlayer3(layer2_other_modal)
        layer3_other_modal_select,layer3_t1_after = self.select_modal_3(layer3_other_modal,layer3_t1)
        cal3 = layer3_t1_after+layer3_other_modal_select
        cal3_t1fa = self.fusion3(layer3_t1_after, layer3_other_modal_select)

        layer4_t1 = self.t1layer4(cal3)
        layer4_other_modal = self.otherlayer4(layer3_other_modal)
        layer4_other_modal_select,layer4_t1_after = self.select_modal_4(layer4_other_modal,layer4_t1)
        # cal4_t1fa = layer4_t1_after+layer4_other_modal_select
        cal4_t1fa = self.fusion4(layer4_t1_after, layer4_other_modal_select)


        up1 = self.up1(cal4_t1fa)
        cat1 = torch.cat([up1, cal3_t1fa], 1)
        upcon1 = self.upconv1(cat1)

        up2 = self.up2(upcon1)
        cat2 = torch.cat([up2, cal2_t1fa], 1)
        upcon2 = self.upconv2(cat2)

        up3 = self.up3(upcon2)
        cat3 = torch.cat([up3, cal1_t1fa], 1)
        upcon3 = self.upconv3(cat3)


        final = self.final(upcon3)

        return final

class CNTSegV2_ARM_CDA_SDM(nn.Module):
    def __init__(self, in_channels_t1=1, in_channels_t2=1,in_channels_fa=1,in_channels_dec=3,in_channels_peaks=9,n_classes=5, is_deconv=True, is_batchnorm=True):
        super(CNTSegV2_ARM_CDA_SDM, self).__init__()

        self.in_channels_t1 = in_channels_t1  # 通道数
        self.in_channels_t2 = in_channels_t2  # 通道数
        self.in_channels_fa = in_channels_fa  # 通道数
        self.in_channels_dec = in_channels_dec  # 通道数
        self.in_channels_peaks = in_channels_peaks  # 通道数
        # self.in_channels_other = in_channels_t2 + in_channels_fa + in_channels_dec + in_channels_peaks
        filters_base = 32
        modality = 4
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input_modal = model_input(in_channels_t2, in_channels_fa, in_channels_dec, in_channels_peaks,filters_base)

        self.t1layer1 = T1_layer_first(self.in_channels_t1,filters_base)
        self.t1layer2 = T1_layer(filters_base, filters_base * 2)
        self.t1layer3 = T1_layer(filters_base * 2, filters_base * 4)
        self.t1layer4 = T1_layer(filters_base * 4, filters_base * 8)

        self.otherlayer1 = Sharelayer_first(filters_base, filters_base, modality)
        self.otherlayer2 = Sharelayer(filters_base, filters_base * 2, modality)
        self.otherlayer3 = Sharelayer(filters_base * 2, filters_base * 4, modality)
        self.otherlayer4 = Sharelayer(filters_base * 4, filters_base * 8, modality)

        self.select_modal_1 = tokenselect(filters_base, modality)
        self.select_modal_2 = tokenselect(filters_base * 2, modality)
        self.select_modal_3 = tokenselect(filters_base * 4, modality)
        self.select_modal_4 = tokenselect(filters_base * 8, modality)

        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)
        self.fusion1 = CDA(filters_base)
        self.fusion2 = CDA(filters_base * 2)
        self.fusion3 = CDA(filters_base * 4)
        self.fusion4 = CDA(filters_base * 8)
        ## CrossAttention With T1,FA,Peaks
        # self.ca1 = FeatureRectifyModule(filters_base)
        # self.ca1 = FeatureRectifyModule(filters_base, reduction=1)
        # self.ca2 = FeatureRectifyModule(filters_base * 2, reduction=1)
        # self.ca3 = FeatureRectifyModule(filters_base * 4, reduction=1)
        # self.ca4 = FeatureRectifyModule(filters_base * 8, reduction=1)
        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)

        # self.fusion = nn.Conv2d(filters_base * 16, filters_base * 8, 3, 1,1)
        # upsampling
        self.up1 = nn.ConvTranspose2d(filters_base* 8, filters_base*4, kernel_size=2, stride=2, padding=0)
        self.upconv1 = unet2dConv2d(filters_base*4*2, filters_base*4, self.is_batchnorm)
        self.up2 = nn.ConvTranspose2d(filters_base*4, filters_base*2, kernel_size=2, stride=2, padding=0)
        self.upconv2 = unet2dConv2d(filters_base*2*2, filters_base*2, self.is_batchnorm)
        self.up3 = nn.ConvTranspose2d(filters_base*2, filters_base, kernel_size=2, stride=2, padding=0)
        self.upconv3 = unet2dConv2d(filters_base*2, filters_base, self.is_batchnorm)

        # ##########  LSF decoder ##########
        self.conv0_dis = nn.Sequential(
            ConvBn2d_1(filters_base* 8, 1),
            nn.RReLU(inplace=True)
        )

        self.conv1_dis = nn.Sequential(
            ConvBn2d_1(filters_base*4, 1),
            nn.RReLU(inplace=True)
        )
        self.up1_dis = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.conv2_dis = nn.Sequential(
            ConvBn2d_1(filters_base*2, 1),
            nn.RReLU(inplace=True)
        )
        self.up2_dis = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Tanh()
        )




        # final conv (without any concat)
        self.final = nn.Conv2d(filters_base, n_classes, 1)
        # self.final_dis = nn.Tanh()
        # self.avg_pool_islands = nn.AdaptiveAvgPool2d(1)
        # self.linear = nn.Linear(filters_base* 8, n_classes-1)
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs_t1, inputs_t2, inputs_fa, inputs_dec, inputs_peaks):
        other_modal = self.input_modal(inputs_t2, inputs_fa, inputs_dec, inputs_peaks)
        layer1_t1 = self.t1layer1(inputs_t1)
        layer1_other_modal = self.otherlayer1(other_modal)
        layer1_other_modal_select,layer1_t1_after = self.select_modal_1(layer1_other_modal,layer1_t1)
        cal1 = layer1_t1_after+layer1_other_modal_select
        cal1_t1fa = self.fusion1(layer1_t1_after, layer1_other_modal_select)

        layer2_t1 = self.t1layer2(cal1)
        layer2_other_modal = self.otherlayer2(layer1_other_modal)
        layer2_other_modal_select,layer2_t1_after = self.select_modal_2(layer2_other_modal,layer2_t1)
        cal2 = layer2_t1_after+layer2_other_modal_select
        cal2_t1fa = self.fusion2(layer2_t1_after, layer2_other_modal_select)

        layer3_t1 = self.t1layer3(cal2)
        layer3_other_modal = self.otherlayer3(layer2_other_modal)
        layer3_other_modal_select,layer3_t1_after = self.select_modal_3(layer3_other_modal,layer3_t1)
        cal3 = layer3_t1_after+layer3_other_modal_select
        cal3_t1fa = self.fusion3(layer3_t1_after, layer3_other_modal_select)

        layer4_t1 = self.t1layer4(cal3)
        layer4_other_modal = self.otherlayer4(layer3_other_modal)
        layer4_other_modal_select,layer4_t1_after = self.select_modal_4(layer4_other_modal,layer4_t1)
        # cal4_t1fa = layer4_t1_after+layer4_other_modal_select
        cal4_t1fa = self.fusion4(layer4_t1_after, layer4_other_modal_select)

        cls = self.conv0_dis(cal4_t1fa)
        up1_dis = self.up1_dis(cls)
        up1 = self.up1(cal4_t1fa)
        cat1 = torch.cat([up1, cal3_t1fa], 1)
        upcon1 = self.upconv1(cat1)
        cls = self.up1_dis(self.conv1_dis(upcon1) + up1_dis)


        up2 = self.up2(upcon1)
        cat2 = torch.cat([up2, cal2_t1fa], 1)
        upcon2 = self.upconv2(cat2)
        cls = self.up2_dis(self.conv2_dis(upcon2) + cls)

        up3 = self.up3(upcon2)
        cat3 = torch.cat([up3, cal1_t1fa], 1)
        upcon3 = self.upconv3(cat3)
        # cls = self.up3_dis(cls)

        final = self.final(upcon3)


        return final,cls
class CNTSegV2_Dedicated(nn.Module):
    def __init__(self, in_channels_t1=1, in_channels_t2=1,in_channels_fa=1,in_channels_dec=3,in_channels_peaks=9,n_classes=5, is_deconv=True, is_batchnorm=True):
        super(CNTSegV2_Dedicated, self).__init__()

        self.in_channels_t1 = in_channels_t1  # 通道数
        self.in_channels_t2 = in_channels_t2  # 通道数
        self.in_channels_fa = in_channels_fa  # 通道数
        self.in_channels_dec = in_channels_dec  # 通道数
        self.in_channels_peaks = in_channels_peaks  # 通道数
        # self.in_channels_other = in_channels_t2 + in_channels_fa + in_channels_dec + in_channels_peaks
        filters_base = 32
        modality = 4
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input_modal = model_input(in_channels_t2, in_channels_fa, in_channels_dec, in_channels_peaks,filters_base)

        self.t1layer1 = T1_layer_first(self.in_channels_t1,filters_base)
        self.t1layer2 = T1_layer(filters_base, filters_base * 2)
        self.t1layer3 = T1_layer(filters_base * 2, filters_base * 4)
        self.t1layer4 = T1_layer(filters_base * 4, filters_base * 8)

        self.otherlayer1 = Sharelayer_first(filters_base, filters_base, modality)
        self.otherlayer2 = Sharelayer(filters_base, filters_base * 2, modality)
        self.otherlayer3 = Sharelayer(filters_base * 2, filters_base * 4, modality)
        self.otherlayer4 = Sharelayer(filters_base * 4, filters_base * 8, modality)

        self.select_modal_1 = tokenselect(filters_base, modality)
        self.select_modal_2 = tokenselect(filters_base * 2, modality)
        self.select_modal_3 = tokenselect(filters_base * 4, modality)
        self.select_modal_4 = tokenselect(filters_base * 8, modality)

        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)
        self.fusion1 = CDA(filters_base)
        self.fusion2 = CDA(filters_base * 2)
        self.fusion3 = CDA(filters_base * 4)
        self.fusion4 = CDA(filters_base * 8)
        ## CrossAttention With T1,FA,Peaks
        # self.ca1 = FeatureRectifyModule(filters_base)
        # self.ca1 = FeatureRectifyModule(filters_base, reduction=1)
        # self.ca2 = FeatureRectifyModule(filters_base * 2, reduction=1)
        # self.ca3 = FeatureRectifyModule(filters_base * 4, reduction=1)
        # self.ca4 = FeatureRectifyModule(filters_base * 8, reduction=1)
        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)

        # self.fusion = nn.Conv2d(filters_base * 16, filters_base * 8, 3, 1,1)
        # upsampling
        self.up1 = nn.ConvTranspose2d(filters_base* 8, filters_base*4, kernel_size=2, stride=2, padding=0)
        self.upconv1 = unet2dConv2d(filters_base*4*2, filters_base*4, self.is_batchnorm)
        self.up2 = nn.ConvTranspose2d(filters_base*4, filters_base*2, kernel_size=2, stride=2, padding=0)
        self.upconv2 = unet2dConv2d(filters_base*2*2, filters_base*2, self.is_batchnorm)
        self.up3 = nn.ConvTranspose2d(filters_base*2, filters_base, kernel_size=2, stride=2, padding=0)
        self.upconv3 = unet2dConv2d(filters_base*2, filters_base, self.is_batchnorm)

        # ##########  LSF decoder ##########
        self.conv0_dis = nn.Sequential(
            ConvBn2d_1(filters_base* 8, 1),
            nn.RReLU(inplace=True)
        )

        self.conv1_dis = nn.Sequential(
            ConvBn2d_1(filters_base*4, 1),
            nn.RReLU(inplace=True)
        )
        self.up1_dis = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.conv2_dis = nn.Sequential(
            ConvBn2d_1(filters_base*2, 1),
            nn.RReLU(inplace=True)
        )
        self.up2_dis = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Tanh()
        )




        # final conv (without any concat)
        self.final = nn.Conv2d(filters_base, n_classes, 1)
        # self.final_dis = nn.Tanh()
        # self.avg_pool_islands = nn.AdaptiveAvgPool2d(1)
        # self.linear = nn.Linear(filters_base* 8, n_classes-1)
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs_t1, inputs_t2, inputs_fa, inputs_dec, inputs_peaks):
        other_modal = self.input_modal(inputs_t2, inputs_fa, inputs_dec, inputs_peaks)
        layer1_t1 = self.t1layer1(inputs_t1)
        layer1_other_modal = self.otherlayer1(other_modal)
        layer1_other_modal_select,layer1_t1_after = self.select_modal_1(layer1_other_modal,layer1_t1)
        cal1 = layer1_t1_after+layer1_other_modal_select
        cal1_t1fa = self.fusion1(layer1_t1_after, layer1_other_modal_select)

        layer2_t1 = self.t1layer2(cal1)
        layer2_other_modal = self.otherlayer2(layer1_other_modal)
        layer2_other_modal_select,layer2_t1_after = self.select_modal_2(layer2_other_modal,layer2_t1)
        cal2 = layer2_t1_after+layer2_other_modal_select
        cal2_t1fa = self.fusion2(layer2_t1_after, layer2_other_modal_select)

        layer3_t1 = self.t1layer3(cal2)
        layer3_other_modal = self.otherlayer3(layer2_other_modal)
        layer3_other_modal_select,layer3_t1_after = self.select_modal_3(layer3_other_modal,layer3_t1)
        cal3 = layer3_t1_after+layer3_other_modal_select
        cal3_t1fa = self.fusion3(layer3_t1_after, layer3_other_modal_select)

        layer4_t1 = self.t1layer4(cal3)
        layer4_other_modal = self.otherlayer4(layer3_other_modal)
        layer4_other_modal_select,layer4_t1_after = self.select_modal_4(layer4_other_modal,layer4_t1)
        # cal4_t1fa = layer4_t1_after+layer4_other_modal_select
        cal4_t1fa = self.fusion4(layer4_t1_after, layer4_other_modal_select)

        cls = self.conv0_dis(cal4_t1fa)
        up1_dis = self.up1_dis(cls)
        up1 = self.up1(cal4_t1fa)
        cat1 = torch.cat([up1, cal3_t1fa], 1)
        upcon1 = self.upconv1(cat1)
        cls = self.up1_dis(self.conv1_dis(upcon1) + up1_dis)


        up2 = self.up2(upcon1)
        cat2 = torch.cat([up2, cal2_t1fa], 1)
        upcon2 = self.upconv2(cat2)
        cls = self.up2_dis(self.conv2_dis(upcon2) + cls)

        up3 = self.up3(upcon2)
        cat3 = torch.cat([up3, cal1_t1fa], 1)
        upcon3 = self.upconv3(cat3)
        # cls = self.up3_dis(cls)

        final = self.final(upcon3)


        return final,cls
class CNTSegV2_NO_Dedicated(nn.Module):
    def __init__(self, in_channels_t1=1, in_channels_t2=1,in_channels_fa=1,in_channels_dec=3,in_channels_peaks=9,n_classes=5, is_deconv=True, is_batchnorm=True):
        super(CNTSegV2_NO_Dedicated, self).__init__()

        self.in_channels_t1 = in_channels_t1  # 通道数
        self.in_channels_t2 = in_channels_t2  # 通道数
        self.in_channels_fa = in_channels_fa  # 通道数
        self.in_channels_dec = in_channels_dec  # 通道数
        self.in_channels_peaks = in_channels_peaks  # 通道数
        # self.in_channels_other = in_channels_t2 + in_channels_fa + in_channels_dec + in_channels_peaks
        filters_base = 32
        modality = 4
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input_modal = model_input(in_channels_t2, in_channels_fa, in_channels_dec, in_channels_peaks,filters_base)

        self.t1layer1 = T1_layer_first(self.in_channels_t1,filters_base)
        self.t1layer2 = T1_layer(filters_base, filters_base * 2)
        self.t1layer3 = T1_layer(filters_base * 2, filters_base * 4)
        self.t1layer4 = T1_layer(filters_base * 4, filters_base * 8)

        self.otherlayer1 = Sharelayer_first(filters_base, filters_base, modality)
        self.otherlayer2 = Sharelayer(filters_base, filters_base * 2, modality)
        self.otherlayer3 = Sharelayer(filters_base * 2, filters_base * 4, modality)
        self.otherlayer4 = Sharelayer(filters_base * 4, filters_base * 8, modality)

        self.select_modal_1 = tokenselect_mask(filters_base, modality)
        self.select_modal_2 = tokenselect_mask(filters_base * 2, modality)
        self.select_modal_3 = tokenselect_mask(filters_base * 4, modality)
        self.select_modal_4 = tokenselect_mask(filters_base * 8, modality)

        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)
        self.fusion1 = CDA(filters_base)
        self.fusion2 = CDA(filters_base * 2)
        self.fusion3 = CDA(filters_base * 4)
        self.fusion4 = CDA(filters_base * 8)
        ## CrossAttention With T1,FA,Peaks
        # self.ca1 = FeatureRectifyModule(filters_base)
        # self.ca1 = FeatureRectifyModule(filters_base, reduction=1)
        # self.ca2 = FeatureRectifyModule(filters_base * 2, reduction=1)
        # self.ca3 = FeatureRectifyModule(filters_base * 4, reduction=1)
        # self.ca4 = FeatureRectifyModule(filters_base * 8, reduction=1)
        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)

        # self.fusion = nn.Conv2d(filters_base * 16, filters_base * 8, 3, 1,1)
        # upsampling
        self.up1 = nn.ConvTranspose2d(filters_base* 8, filters_base*4, kernel_size=2, stride=2, padding=0)
        self.upconv1 = unet2dConv2d(filters_base*4*2, filters_base*4, self.is_batchnorm)
        self.up2 = nn.ConvTranspose2d(filters_base*4, filters_base*2, kernel_size=2, stride=2, padding=0)
        self.upconv2 = unet2dConv2d(filters_base*2*2, filters_base*2, self.is_batchnorm)
        self.up3 = nn.ConvTranspose2d(filters_base*2, filters_base, kernel_size=2, stride=2, padding=0)
        self.upconv3 = unet2dConv2d(filters_base*2, filters_base, self.is_batchnorm)

        # ##########  LSF decoder ##########
        self.conv0_dis = nn.Sequential(
            ConvBn2d_1(filters_base* 8, 1),
            nn.RReLU(inplace=True)
        )

        self.conv1_dis = nn.Sequential(
            ConvBn2d_1(filters_base*4, 1),
            nn.RReLU(inplace=True)
        )
        self.up1_dis = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.conv2_dis = nn.Sequential(
            ConvBn2d_1(filters_base*2, 1),
            nn.RReLU(inplace=True)
        )
        self.up2_dis = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Tanh()
        )




        # final conv (without any concat)
        self.final = nn.Conv2d(filters_base, n_classes, 1)
        # self.final_dis = nn.Tanh()
        # self.avg_pool_islands = nn.AdaptiveAvgPool2d(1)
        # self.linear = nn.Linear(filters_base* 8, n_classes-1)
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs_t1, inputs_t2, inputs_fa, inputs_dec, inputs_peaks, modality_mask):
        other_modal = self.input_modal(inputs_t2, inputs_fa, inputs_dec, inputs_peaks)
        layer1_t1 = self.t1layer1(inputs_t1)
        layer1_other_modal = self.otherlayer1(other_modal)
        layer1_other_modal_select,layer1_t1_after = self.select_modal_1(layer1_other_modal,layer1_t1, modality_mask)
        cal1 = layer1_t1_after+layer1_other_modal_select
        cal1_t1fa = self.fusion1(layer1_t1_after, layer1_other_modal_select)

        layer2_t1 = self.t1layer2(cal1)
        layer2_other_modal = self.otherlayer2(layer1_other_modal)
        layer2_other_modal_select,layer2_t1_after = self.select_modal_2(layer2_other_modal,layer2_t1, modality_mask)
        cal2 = layer2_t1_after+layer2_other_modal_select
        cal2_t1fa = self.fusion2(layer2_t1_after, layer2_other_modal_select)

        layer3_t1 = self.t1layer3(cal2)
        layer3_other_modal = self.otherlayer3(layer2_other_modal)
        layer3_other_modal_select,layer3_t1_after = self.select_modal_3(layer3_other_modal,layer3_t1, modality_mask)
        cal3 = layer3_t1_after+layer3_other_modal_select
        cal3_t1fa = self.fusion3(layer3_t1_after, layer3_other_modal_select)

        layer4_t1 = self.t1layer4(cal3)
        layer4_other_modal = self.otherlayer4(layer3_other_modal)
        layer4_other_modal_select,layer4_t1_after = self.select_modal_4(layer4_other_modal,layer4_t1, modality_mask)
        # cal4_t1fa = layer4_t1_after+layer4_other_modal_select
        cal4_t1fa = self.fusion4(layer4_t1_after, layer4_other_modal_select)

        cls = self.conv0_dis(cal4_t1fa)
        up1_dis = self.up1_dis(cls)
        up1 = self.up1(cal4_t1fa)
        cat1 = torch.cat([up1, cal3_t1fa], 1)
        upcon1 = self.upconv1(cat1)
        cls = self.up1_dis(self.conv1_dis(upcon1) + up1_dis)


        up2 = self.up2(upcon1)
        cat2 = torch.cat([up2, cal2_t1fa], 1)
        upcon2 = self.upconv2(cat2)
        cls = self.up2_dis(self.conv2_dis(upcon2) + cls)

        up3 = self.up3(upcon2)
        cat3 = torch.cat([up3, cal1_t1fa], 1)
        upcon3 = self.upconv3(cat3)
        # cls = self.up3_dis(cls)

        final = self.final(upcon3)


        return final,cls
class CNTSegV2_NO_Dedicated_without_SDM(nn.Module):
    def __init__(self, in_channels_t1=1, in_channels_t2=1,in_channels_fa=1,in_channels_dec=3,in_channels_peaks=9,n_classes=5, is_deconv=True, is_batchnorm=True):
        super(CNTSegV2_NO_Dedicated_without_SDM, self).__init__()

        self.in_channels_t1 = in_channels_t1  # 通道数
        self.in_channels_t2 = in_channels_t2  # 通道数
        self.in_channels_fa = in_channels_fa  # 通道数
        self.in_channels_dec = in_channels_dec  # 通道数
        self.in_channels_peaks = in_channels_peaks  # 通道数
        # self.in_channels_other = in_channels_t2 + in_channels_fa + in_channels_dec + in_channels_peaks
        filters_base = 32
        modality = 4
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input_modal = model_input(in_channels_t2, in_channels_fa, in_channels_dec, in_channels_peaks,filters_base)

        self.t1layer1 = T1_layer_first(self.in_channels_t1,filters_base)
        self.t1layer2 = T1_layer(filters_base, filters_base * 2)
        self.t1layer3 = T1_layer(filters_base * 2, filters_base * 4)
        self.t1layer4 = T1_layer(filters_base * 4, filters_base * 8)

        self.otherlayer1 = Sharelayer_first(filters_base, filters_base, modality)
        self.otherlayer2 = Sharelayer(filters_base, filters_base * 2, modality)
        self.otherlayer3 = Sharelayer(filters_base * 2, filters_base * 4, modality)
        self.otherlayer4 = Sharelayer(filters_base * 4, filters_base * 8, modality)

        self.select_modal_1 = tokenselect_mask(filters_base, modality)
        self.select_modal_2 = tokenselect_mask(filters_base * 2, modality)
        self.select_modal_3 = tokenselect_mask(filters_base * 4, modality)
        self.select_modal_4 = tokenselect_mask(filters_base * 8, modality)

        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)
        self.fusion1 = CDA(filters_base)
        self.fusion2 = CDA(filters_base * 2)
        self.fusion3 = CDA(filters_base * 4)
        self.fusion4 = CDA(filters_base * 8)
        ## CrossAttention With T1,FA,Peaks
        # self.ca1 = FeatureRectifyModule(filters_base)
        # self.ca1 = FeatureRectifyModule(filters_base, reduction=1)
        # self.ca2 = FeatureRectifyModule(filters_base * 2, reduction=1)
        # self.ca3 = FeatureRectifyModule(filters_base * 4, reduction=1)
        # self.ca4 = FeatureRectifyModule(filters_base * 8, reduction=1)
        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)

        # self.fusion = nn.Conv2d(filters_base * 16, filters_base * 8, 3, 1,1)
        # upsampling
        self.up1 = nn.ConvTranspose2d(filters_base* 8, filters_base*4, kernel_size=2, stride=2, padding=0)
        self.upconv1 = unet2dConv2d(filters_base*4*2, filters_base*4, self.is_batchnorm)
        self.up2 = nn.ConvTranspose2d(filters_base*4, filters_base*2, kernel_size=2, stride=2, padding=0)
        self.upconv2 = unet2dConv2d(filters_base*2*2, filters_base*2, self.is_batchnorm)
        self.up3 = nn.ConvTranspose2d(filters_base*2, filters_base, kernel_size=2, stride=2, padding=0)
        self.upconv3 = unet2dConv2d(filters_base*2, filters_base, self.is_batchnorm)

        # ##########  LSF decoder ##########
        self.conv0_dis = nn.Sequential(
            ConvBn2d_1(filters_base* 8, 1),
            nn.RReLU(inplace=True)
        )

        self.conv1_dis = nn.Sequential(
            ConvBn2d_1(filters_base*4, 1),
            nn.RReLU(inplace=True)
        )
        self.up1_dis = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.conv2_dis = nn.Sequential(
            ConvBn2d_1(filters_base*2, 1),
            nn.RReLU(inplace=True)
        )
        self.up2_dis = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.Tanh()
            nn.Sigmoid()
        )
        # self.dis_final = nn.Conv2d(filters_base, n_classes, 1)



        # final conv (without any concat)
        self.final = nn.Conv2d(filters_base, n_classes, 1)
        # self.final_dis = nn.Tanh()
        # self.avg_pool_islands = nn.AdaptiveAvgPool2d(1)
        # self.linear = nn.Linear(filters_base* 8, n_classes-1)
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs_t1, inputs_t2, inputs_fa, inputs_dec, inputs_peaks, modality_mask):
        other_modal = self.input_modal(inputs_t2, inputs_fa, inputs_dec, inputs_peaks)
        layer1_t1 = self.t1layer1(inputs_t1)
        layer1_other_modal = self.otherlayer1(other_modal)
        layer1_other_modal_select,layer1_t1_after = self.select_modal_1(layer1_other_modal,layer1_t1, modality_mask)
        cal1 = layer1_t1_after+layer1_other_modal_select
        cal1_t1fa = self.fusion1(layer1_t1_after, layer1_other_modal_select)

        layer2_t1 = self.t1layer2(cal1)
        layer2_other_modal = self.otherlayer2(layer1_other_modal)
        layer2_other_modal_select,layer2_t1_after = self.select_modal_2(layer2_other_modal,layer2_t1, modality_mask)
        cal2 = layer2_t1_after+layer2_other_modal_select
        cal2_t1fa = self.fusion2(layer2_t1_after, layer2_other_modal_select)

        layer3_t1 = self.t1layer3(cal2)
        layer3_other_modal = self.otherlayer3(layer2_other_modal)
        layer3_other_modal_select,layer3_t1_after = self.select_modal_3(layer3_other_modal,layer3_t1, modality_mask)
        cal3 = layer3_t1_after+layer3_other_modal_select
        cal3_t1fa = self.fusion3(layer3_t1_after, layer3_other_modal_select)

        layer4_t1 = self.t1layer4(cal3)
        layer4_other_modal = self.otherlayer4(layer3_other_modal)
        layer4_other_modal_select,layer4_t1_after = self.select_modal_4(layer4_other_modal,layer4_t1, modality_mask)
        # cal4_t1fa = layer4_t1_after+layer4_other_modal_select
        cal4_t1fa = self.fusion4(layer4_t1_after, layer4_other_modal_select)

        cls = self.conv0_dis(cal4_t1fa)
        up1_dis = self.up1_dis(cls)
        up1 = self.up1(cal4_t1fa)
        cat1 = torch.cat([up1, cal3_t1fa], 1)
        upcon1 = self.upconv1(cat1)
        cls = self.up1_dis(self.conv1_dis(upcon1) + up1_dis)


        up2 = self.up2(upcon1)
        cat2 = torch.cat([up2, cal2_t1fa], 1)
        upcon2 = self.upconv2(cat2)
        cls = self.up2_dis(self.conv2_dis(upcon2) + cls)

        up3 = self.up3(upcon2)
        cat3 = torch.cat([up3, cal1_t1fa], 1)
        upcon3 = self.upconv3(cat3)
        # cls = self.up3_dis(cls)

        final = self.final(upcon3)
        # dis_final = self.dis_final(cls)


        return final,cls
class CNTSegV2_NO_Dedicated_without_CDA(nn.Module):
    def __init__(self, in_channels_t1=1, in_channels_t2=1,in_channels_fa=1,in_channels_dec=3,in_channels_peaks=9,n_classes=5, is_deconv=True, is_batchnorm=True):
        super(CNTSegV2_NO_Dedicated_without_CDA, self).__init__()

        self.in_channels_t1 = in_channels_t1  # 通道数
        self.in_channels_t2 = in_channels_t2  # 通道数
        self.in_channels_fa = in_channels_fa  # 通道数
        self.in_channels_dec = in_channels_dec  # 通道数
        self.in_channels_peaks = in_channels_peaks  # 通道数
        # self.in_channels_other = in_channels_t2 + in_channels_fa + in_channels_dec + in_channels_peaks
        filters_base = 32
        modality = 4
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input_modal = model_input(in_channels_t2, in_channels_fa, in_channels_dec, in_channels_peaks,filters_base)

        self.t1layer1 = T1_layer_first(self.in_channels_t1,filters_base)
        self.t1layer2 = T1_layer(filters_base, filters_base * 2)
        self.t1layer3 = T1_layer(filters_base * 2, filters_base * 4)
        self.t1layer4 = T1_layer(filters_base * 4, filters_base * 8)

        self.otherlayer1 = Sharelayer_first(filters_base, filters_base, modality)
        self.otherlayer2 = Sharelayer(filters_base, filters_base * 2, modality)
        self.otherlayer3 = Sharelayer(filters_base * 2, filters_base * 4, modality)
        self.otherlayer4 = Sharelayer(filters_base * 4, filters_base * 8, modality)

        self.select_modal_1 = tokenselect_mask(filters_base, modality)
        self.select_modal_2 = tokenselect_mask(filters_base * 2, modality)
        self.select_modal_3 = tokenselect_mask(filters_base * 4, modality)
        self.select_modal_4 = tokenselect_mask(filters_base * 8, modality)

        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)
        # self.fusion1 = CDA(filters_base)
        # self.fusion2 = CDA(filters_base * 2)
        # self.fusion3 = CDA(filters_base * 4)
        # self.fusion4 = CDA(filters_base * 8)
        ## CrossAttention With T1,FA,Peaks
        # self.ca1 = FeatureRectifyModule(filters_base)
        # self.ca1 = FeatureRectifyModule(filters_base, reduction=1)
        # self.ca2 = FeatureRectifyModule(filters_base * 2, reduction=1)
        # self.ca3 = FeatureRectifyModule(filters_base * 4, reduction=1)
        # self.ca4 = FeatureRectifyModule(filters_base * 8, reduction=1)
        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)

        # self.fusion = nn.Conv2d(filters_base * 16, filters_base * 8, 3, 1,1)
        # upsampling
        self.up1 = nn.ConvTranspose2d(filters_base* 8, filters_base*4, kernel_size=2, stride=2, padding=0)
        self.upconv1 = unet2dConv2d(filters_base*4*2, filters_base*4, self.is_batchnorm)
        self.up2 = nn.ConvTranspose2d(filters_base*4, filters_base*2, kernel_size=2, stride=2, padding=0)
        self.upconv2 = unet2dConv2d(filters_base*2*2, filters_base*2, self.is_batchnorm)
        self.up3 = nn.ConvTranspose2d(filters_base*2, filters_base, kernel_size=2, stride=2, padding=0)
        self.upconv3 = unet2dConv2d(filters_base*2, filters_base, self.is_batchnorm)

        # ##########  LSF decoder ##########
        self.conv0_dis = nn.Sequential(
            ConvBn2d_1(filters_base* 8, 1),
            nn.RReLU(inplace=True)
        )

        self.conv1_dis = nn.Sequential(
            ConvBn2d_1(filters_base*4, 1),
            nn.RReLU(inplace=True)
        )
        self.up1_dis = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.conv2_dis = nn.Sequential(
            ConvBn2d_1(filters_base*2, 1),
            nn.RReLU(inplace=True)
        )
        self.up2_dis = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Tanh()
        )




        # final conv (without any concat)
        self.final = nn.Conv2d(filters_base, n_classes, 1)
        # self.final_dis = nn.Tanh()
        # self.avg_pool_islands = nn.AdaptiveAvgPool2d(1)
        # self.linear = nn.Linear(filters_base* 8, n_classes-1)
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs_t1, inputs_t2, inputs_fa, inputs_dec, inputs_peaks, modality_mask):
        other_modal = self.input_modal(inputs_t2, inputs_fa, inputs_dec, inputs_peaks)
        layer1_t1 = self.t1layer1(inputs_t1)
        layer1_other_modal = self.otherlayer1(other_modal)
        layer1_other_modal_select,layer1_t1_after = self.select_modal_1(layer1_other_modal,layer1_t1, modality_mask)
        cal1_t1fa = layer1_t1_after+layer1_other_modal_select
        # cal1_t1fa = self.fusion1(layer1_t1_after, layer1_other_modal_select)

        layer2_t1 = self.t1layer2(cal1_t1fa)
        layer2_other_modal = self.otherlayer2(layer1_other_modal)
        layer2_other_modal_select,layer2_t1_after = self.select_modal_2(layer2_other_modal,layer2_t1, modality_mask)
        cal2_t1fa = layer2_t1_after+layer2_other_modal_select
        # cal2_t1fa = self.fusion2(layer2_t1_after, layer2_other_modal_select)

        layer3_t1 = self.t1layer3(cal2_t1fa)
        layer3_other_modal = self.otherlayer3(layer2_other_modal)
        layer3_other_modal_select,layer3_t1_after = self.select_modal_3(layer3_other_modal,layer3_t1, modality_mask)
        cal3_t1fa = layer3_t1_after+layer3_other_modal_select
        # cal3_t1fa = self.fusion3(layer3_t1_after, layer3_other_modal_select)

        layer4_t1 = self.t1layer4(cal3_t1fa)
        layer4_other_modal = self.otherlayer4(layer3_other_modal)
        layer4_other_modal_select,layer4_t1_after = self.select_modal_4(layer4_other_modal,layer4_t1, modality_mask)
        cal4_t1fa = layer4_t1_after+layer4_other_modal_select
        # cal4_t1fa = self.fusion4(layer4_t1_after, layer4_other_modal_select)

        cls = self.conv0_dis(cal4_t1fa)
        up1_dis = self.up1_dis(cls)
        up1 = self.up1(cal4_t1fa)
        cat1 = torch.cat([up1, cal3_t1fa], 1)
        upcon1 = self.upconv1(cat1)
        cls = self.up1_dis(self.conv1_dis(upcon1) + up1_dis)


        up2 = self.up2(upcon1)
        cat2 = torch.cat([up2, cal2_t1fa], 1)
        upcon2 = self.upconv2(cat2)
        cls = self.up2_dis(self.conv2_dis(upcon2) + cls)

        up3 = self.up3(upcon2)
        cat3 = torch.cat([up3, cal1_t1fa], 1)
        upcon3 = self.upconv3(cat3)
        # cls = self.up3_dis(cls)

        final = self.final(upcon3)


        return final,cls
class CNTSegV2_NO_Dedicated_without_ARM(nn.Module):
    def __init__(self, in_channels_t1=1, in_channels_t2=1,in_channels_fa=1,in_channels_dec=3,in_channels_peaks=9,n_classes=5, is_deconv=True, is_batchnorm=True):
        super(CNTSegV2_NO_Dedicated_without_ARM, self).__init__()

        self.in_channels_t1 = in_channels_t1  # 通道数
        self.in_channels_t2 = in_channels_t2  # 通道数
        self.in_channels_fa = in_channels_fa  # 通道数
        self.in_channels_dec = in_channels_dec  # 通道数
        self.in_channels_peaks = in_channels_peaks  # 通道数
        # self.in_channels_other = in_channels_t2 + in_channels_fa + in_channels_dec + in_channels_peaks
        filters_base = 32
        modality = 4
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input_modal = model_input(in_channels_t2, in_channels_fa, in_channels_dec, in_channels_peaks,filters_base)

        self.t1layer1 = T1_layer_first(self.in_channels_t1,filters_base)
        self.t1layer2 = T1_layer(filters_base, filters_base * 2)
        self.t1layer3 = T1_layer(filters_base * 2, filters_base * 4)
        self.t1layer4 = T1_layer(filters_base * 4, filters_base * 8)

        self.otherlayer1 = Sharelayer_first(filters_base, filters_base, modality)
        self.otherlayer2 = Sharelayer(filters_base, filters_base * 2, modality)
        self.otherlayer3 = Sharelayer(filters_base * 2, filters_base * 4, modality)
        self.otherlayer4 = Sharelayer(filters_base * 4, filters_base * 8, modality)

        self.select_modal_1 = tokenselect_mask_ablation_arm(filters_base, modality)
        self.select_modal_2 = tokenselect_mask_ablation_arm(filters_base * 2, modality)
        self.select_modal_3 = tokenselect_mask_ablation_arm(filters_base * 4, modality)
        self.select_modal_4 = tokenselect_mask_ablation_arm(filters_base * 8, modality)

        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)
        self.fusion1 = CDA(filters_base)
        self.fusion2 = CDA(filters_base * 2)
        self.fusion3 = CDA(filters_base * 4)
        self.fusion4 = CDA(filters_base * 8)
        ## CrossAttention With T1,FA,Peaks
        # self.ca1 = FeatureRectifyModule(filters_base)
        # self.ca1 = FeatureRectifyModule(filters_base, reduction=1)
        # self.ca2 = FeatureRectifyModule(filters_base * 2, reduction=1)
        # self.ca3 = FeatureRectifyModule(filters_base * 4, reduction=1)
        # self.ca4 = FeatureRectifyModule(filters_base * 8, reduction=1)
        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)

        # self.fusion = nn.Conv2d(filters_base * 16, filters_base * 8, 3, 1,1)
        # upsampling
        self.up1 = nn.ConvTranspose2d(filters_base* 8, filters_base*4, kernel_size=2, stride=2, padding=0)
        self.upconv1 = unet2dConv2d(filters_base*4*2, filters_base*4, self.is_batchnorm)
        self.up2 = nn.ConvTranspose2d(filters_base*4, filters_base*2, kernel_size=2, stride=2, padding=0)
        self.upconv2 = unet2dConv2d(filters_base*2*2, filters_base*2, self.is_batchnorm)
        self.up3 = nn.ConvTranspose2d(filters_base*2, filters_base, kernel_size=2, stride=2, padding=0)
        self.upconv3 = unet2dConv2d(filters_base*2, filters_base, self.is_batchnorm)

        # ##########  LSF decoder ##########
        self.conv0_dis = nn.Sequential(
            ConvBn2d_1(filters_base* 8, 1),
            nn.RReLU(inplace=True)
        )

        self.conv1_dis = nn.Sequential(
            ConvBn2d_1(filters_base*4, 1),
            nn.RReLU(inplace=True)
        )
        self.up1_dis = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.conv2_dis = nn.Sequential(
            ConvBn2d_1(filters_base*2, 1),
            nn.RReLU(inplace=True)
        )
        self.up2_dis = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Tanh()
        )




        # final conv (without any concat)
        self.final = nn.Conv2d(filters_base, n_classes, 1)
        # self.final_dis = nn.Tanh()
        # self.avg_pool_islands = nn.AdaptiveAvgPool2d(1)
        # self.linear = nn.Linear(filters_base* 8, n_classes-1)
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs_t1, inputs_t2, inputs_fa, inputs_dec, inputs_peaks, modality_mask):
        other_modal = self.input_modal(inputs_t2, inputs_fa, inputs_dec, inputs_peaks)
        layer1_t1 = self.t1layer1(inputs_t1)
        layer1_other_modal = self.otherlayer1(other_modal)
        layer1_other_modal_select,_ = self.select_modal_1(layer1_other_modal,layer1_t1, modality_mask)
        # cal1 = layer1_t1_after+layer1_other_modal_select
        cal1_t1fa = self.fusion1(layer1_t1, layer1_other_modal_select)

        layer2_t1 = self.t1layer2(cal1_t1fa)
        layer2_other_modal = self.otherlayer2(layer1_other_modal)
        layer2_other_modal_select,_  = self.select_modal_2(layer2_other_modal,layer2_t1, modality_mask)
        # cal2 = layer2_t1_after+layer2_other_modal_select
        cal2_t1fa = self.fusion2(layer2_t1, layer2_other_modal_select)

        layer3_t1 = self.t1layer3(cal2_t1fa)
        layer3_other_modal = self.otherlayer3(layer2_other_modal)
        layer3_other_modal_select,_ = self.select_modal_3(layer3_other_modal,layer3_t1, modality_mask)
        # cal3 = layer3_t1_after+layer3_other_modal_select
        cal3_t1fa = self.fusion3(layer3_t1, layer3_other_modal_select)

        layer4_t1 = self.t1layer4(cal3_t1fa)
        layer4_other_modal = self.otherlayer4(layer3_other_modal)
        layer4_other_modal_select,_  = self.select_modal_4(layer4_other_modal,layer4_t1, modality_mask)
        # cal4_t1fa = layer4_t1_after+layer4_other_modal_select
        cal4_t1fa = self.fusion4(layer4_t1, layer4_other_modal_select)

        cls = self.conv0_dis(cal4_t1fa)
        up1_dis = self.up1_dis(cls)
        up1 = self.up1(cal4_t1fa)
        cat1 = torch.cat([up1, cal3_t1fa], 1)
        upcon1 = self.upconv1(cat1)
        cls = self.up1_dis(self.conv1_dis(upcon1) + up1_dis)


        up2 = self.up2(upcon1)
        cat2 = torch.cat([up2, cal2_t1fa], 1)
        upcon2 = self.upconv2(cat2)
        cls = self.up2_dis(self.conv2_dis(upcon2) + cls)

        up3 = self.up3(upcon2)
        cat3 = torch.cat([up3, cal1_t1fa], 1)
        upcon3 = self.upconv3(cat3)
        # cls = self.up3_dis(cls)

        final = self.final(upcon3)


        return final,cls
class CNTSegV2_NO_Dedicated_without_ALL(nn.Module):
    def __init__(self, in_channels_t1=1, in_channels_t2=1,in_channels_fa=1,in_channels_dec=3,in_channels_peaks=9,n_classes=5, is_deconv=True, is_batchnorm=True):
        super(CNTSegV2_NO_Dedicated_without_ALL, self).__init__()

        self.in_channels_t1 = in_channels_t1  # 通道数
        self.in_channels_t2 = in_channels_t2  # 通道数
        self.in_channels_fa = in_channels_fa  # 通道数
        self.in_channels_dec = in_channels_dec  # 通道数
        self.in_channels_peaks = in_channels_peaks  # 通道数
        # self.in_channels_other = in_channels_t2 + in_channels_fa + in_channels_dec + in_channels_peaks
        filters_base = 32
        modality = 4
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input_modal = model_input(in_channels_t2, in_channels_fa, in_channels_dec, in_channels_peaks,filters_base)

        self.t1layer1 = T1_layer_first(self.in_channels_t1,filters_base)
        self.t1layer2 = T1_layer(filters_base, filters_base * 2)
        self.t1layer3 = T1_layer(filters_base * 2, filters_base * 4)
        self.t1layer4 = T1_layer(filters_base * 4, filters_base * 8)

        self.otherlayer1 = Sharelayer_first(filters_base, filters_base, modality)
        self.otherlayer2 = Sharelayer(filters_base, filters_base * 2, modality)
        self.otherlayer3 = Sharelayer(filters_base * 2, filters_base * 4, modality)
        self.otherlayer4 = Sharelayer(filters_base * 4, filters_base * 8, modality)

        self.select_modal_1 = tokenselect_mask_ablation_arm(filters_base, modality)
        self.select_modal_2 = tokenselect_mask_ablation_arm(filters_base * 2, modality)
        self.select_modal_3 = tokenselect_mask_ablation_arm(filters_base * 4, modality)
        self.select_modal_4 = tokenselect_mask_ablation_arm(filters_base * 8, modality)

        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)
        # self.fusion1 = CDA(filters_base)
        # self.fusion2 = CDA(filters_base * 2)
        # self.fusion3 = CDA(filters_base * 4)
        # self.fusion4 = CDA(filters_base * 8)
        ## CrossAttention With T1,FA,Peaks
        # self.ca1 = FeatureRectifyModule(filters_base)
        # self.ca1 = FeatureRectifyModule(filters_base, reduction=1)
        # self.ca2 = FeatureRectifyModule(filters_base * 2, reduction=1)
        # self.ca3 = FeatureRectifyModule(filters_base * 4, reduction=1)
        # self.ca4 = FeatureRectifyModule(filters_base * 8, reduction=1)
        # self.fusion1 = FeatureFusionModule(filters_base, reduction=1, num_heads=1)
        # self.fusion2 = FeatureFusionModule(filters_base * 2, reduction=1, num_heads=2)
        # self.fusion3 = FeatureFusionModule(filters_base * 4, reduction=1, num_heads=4)
        # self.fusion4 = FeatureFusionModule(filters_base * 8, reduction=1, num_heads=8)

        # self.fusion = nn.Conv2d(filters_base * 16, filters_base * 8, 3, 1,1)
        # upsampling
        self.up1 = nn.ConvTranspose2d(filters_base* 8, filters_base*4, kernel_size=2, stride=2, padding=0)
        self.upconv1 = unet2dConv2d(filters_base*4*2, filters_base*4, self.is_batchnorm)
        self.up2 = nn.ConvTranspose2d(filters_base*4, filters_base*2, kernel_size=2, stride=2, padding=0)
        self.upconv2 = unet2dConv2d(filters_base*2*2, filters_base*2, self.is_batchnorm)
        self.up3 = nn.ConvTranspose2d(filters_base*2, filters_base, kernel_size=2, stride=2, padding=0)
        self.upconv3 = unet2dConv2d(filters_base*2, filters_base, self.is_batchnorm)

        # ##########  LSF decoder ##########
        # self.conv0_dis = nn.Sequential(
        #     ConvBn2d_1(filters_base* 8, 1),
        #     nn.RReLU(inplace=True)
        # )
        #
        # self.conv1_dis = nn.Sequential(
        #     ConvBn2d_1(filters_base*4, 1),
        #     nn.RReLU(inplace=True)
        # )
        # self.up1_dis = nn.Sequential(
        #     nn.UpsamplingBilinear2d(scale_factor=2)
        # )
        #
        # self.conv2_dis = nn.Sequential(
        #     ConvBn2d_1(filters_base*2, 1),
        #     nn.RReLU(inplace=True)
        # )
        # self.up2_dis = nn.Sequential(
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Tanh()
        # )
        #



        # final conv (without any concat)
        self.final = nn.Conv2d(filters_base, n_classes, 1)
        # self.final_dis = nn.Tanh()
        # self.avg_pool_islands = nn.AdaptiveAvgPool2d(1)
        # self.linear = nn.Linear(filters_base* 8, n_classes-1)
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs_t1, inputs_t2, inputs_fa, inputs_dec, inputs_peaks, modality_mask):
        other_modal = self.input_modal(inputs_t2, inputs_fa, inputs_dec, inputs_peaks)
        layer1_t1 = self.t1layer1(inputs_t1)
        layer1_other_modal = self.otherlayer1(other_modal)
        layer1_other_modal_select,_ = self.select_modal_1(layer1_other_modal,layer1_t1, modality_mask)
        cal1 = layer1_t1+layer1_other_modal_select
        # cal1_t1fa = self.fusion1(layer1_t1, layer1_other_modal_select)

        layer2_t1 = self.t1layer2(cal1)
        layer2_other_modal = self.otherlayer2(layer1_other_modal)
        layer2_other_modal_select,_  = self.select_modal_2(layer2_other_modal,layer2_t1, modality_mask)
        cal2 = layer2_t1+layer2_other_modal_select
        # cal2_t1fa = self.fusion2(layer2_t1, layer2_other_modal_select)

        layer3_t1 = self.t1layer3(cal2)
        layer3_other_modal = self.otherlayer3(layer2_other_modal)
        layer3_other_modal_select,_ = self.select_modal_3(layer3_other_modal,layer3_t1, modality_mask)
        cal3 = layer3_t1+layer3_other_modal_select
        # cal3_t1fa = self.fusion3(layer3_t1, layer3_other_modal_select)

        layer4_t1 = self.t1layer4(cal3)
        layer4_other_modal = self.otherlayer4(layer3_other_modal)
        layer4_other_modal_select,_  = self.select_modal_4(layer4_other_modal,layer4_t1, modality_mask)
        cal4 = layer4_t1+layer4_other_modal_select
        # cal4_t1fa = self.fusion4(layer4_t1, layer4_other_modal_select)

        # cls = self.conv0_dis(cal4_t1fa)
        # up1_dis = self.up1_dis(cls)
        up1 = self.up1(cal4)
        cat1 = torch.cat([up1, cal3], 1)
        upcon1 = self.upconv1(cat1)
        # cls = self.up1_dis(self.conv1_dis(upcon1) + up1_dis)


        up2 = self.up2(upcon1)
        cat2 = torch.cat([up2, cal2], 1)
        upcon2 = self.upconv2(cat2)
        # cls = self.up2_dis(self.conv2_dis(upcon2) + cls)

        up3 = self.up3(upcon2)
        cat3 = torch.cat([up3, cal1], 1)
        upcon3 = self.upconv3(cat3)
        # cls = self.up3_dis(cls)

        final = self.final(upcon3)


        return final
if __name__ == '__main__':
    # 是否使用cuda
    masks = np.array([[False, False, False, False], [True, False, False, False], [False, True, False, False],
                      [False, False, True, False], [False, False, False, True],
                      [True, True, False, False], [True, False, True, False], [True, False, False, True],
                      [False, True, True, False], [False, True, False, True], [False, False, True, True],
                      [True, True, True, False], [True, True, False, True], [True, False, True, True],
                      [False, True, True, True],
                      [True, True, True, True]])
    mask_idx = np.random.choice(15, 1)
    # mask2 = torch.from_numpy(masks[mask_idx])
    mask = torch.from_numpy(masks[mask_idx])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('#### Test Case ###')
    model = CNTSegV2_NO_Dedicated_without_ALL(1, 1, 1, 3, 9, 5).to(device)

    x1 = torch.rand(16, 1, 144, 144).to(device)
    x2 = torch.rand(16, 1, 144, 144).to(device)
    x3 = torch.rand(16, 1, 144, 144).to(device)
    x4 = torch.rand(16, 3, 144, 144).to(device)
    x5 = torch.rand(16, 9, 144, 144).to(device)
    mask = mask.to(device)
    y = model(x1, x2, x3, x4, x5, mask)
    param = count_param(model)  # 计算参数
    # print('Input shape:', x.shape)
    print('Output shape:', y.shape)
    print('UNet3d totoal parameters: %.2fM (%d)' % (param / 1e6, param))

