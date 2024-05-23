
import torch
import torch.nn as nn
from metrics_2d import init_weights, count_param

def conv3x3(input_channel, output_channel):
    return nn.Sequential(nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm2d(output_channel),
                         nn.ReLU(inplace=True))


def UNet_up_conv_bn_relu(input_channel, output_channel, learned_bilinear=False):
    if learned_bilinear:
        return nn.Sequential(nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2),
                             nn.BatchNorm2d(output_channel),
                             nn.ReLU())
    else:
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                             nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
                             nn.BatchNorm2d(output_channel),
                             nn.ReLU())


class basic_block(nn.Module):
    def  __init__(self, input_channel, output_channel):
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Channel_Attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class Spatial_Attention(nn.Module):
    def __init__(self, input_channel, reduction=16, dilation=4):
        super(Spatial_Attention, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, input_channel // 16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(input_channel // 16, input_channel // 16, kernel_size=3, dilation=4,
                               stride=1, padding=4)
        self.conv3 = nn.Conv2d(input_channel // 16, input_channel // 16, kernel_size=3, dilation=4,
                               stride=1, padding=4)
        self.conv4 = nn.Conv2d(input_channel // 16, 1, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.bn(self.conv4(y))
        y = self.sigmoid(y)

        return y


class Bottleneck_Attention_Module(nn.Module):
    def __init__(self, input_channel, reduction=16, dilation=4):
        super(Bottleneck_Attention_Module, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_channel, input_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(input_channel // reduction, input_channel),
            nn.Sigmoid()
        )

        self.conv1 = nn.Conv2d(input_channel, input_channel // reduction, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(input_channel // reduction, input_channel // reduction, kernel_size=3, dilation=dilation,
                               stride=1, padding=dilation)
        self.conv3 = nn.Conv2d(input_channel // reduction, input_channel // reduction, kernel_size=3, dilation=dilation,
                               stride=1, padding=dilation)
        self.conv4 = nn.Conv2d(input_channel // reduction, 1, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y1 = self.fc(y1).view(b, c, 1, 1)
        ca_weights = torch.ones(x.size()).cuda() * y1

        y2 = self.conv1(x)
        y2 = self.conv2(y2)
        y2 = self.conv3(y2)
        y2 = self.bn(self.conv4(y2))

        sa_weights = y2.repeat(1, x.size()[1], 1, 1)

        y = self.sigmoid(ca_weights + sa_weights)

        return y


class UNet_basic_down_block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(UNet_basic_down_block, self).__init__()
        self.block = basic_block(input_channel, output_channel)

    def forward(self, x):
        x = self.block(x)
        return x


class UNet_basic_up_block(nn.Module):
    def __init__(self, input_channel, prev_channel, output_channel, dilation, learned_bilinear=False):
        super(UNet_basic_up_block, self).__init__()
        self.bilinear_up = UNet_up_conv_bn_relu(input_channel, prev_channel, learned_bilinear)
        self.block = basic_block(prev_channel * 2, output_channel)

    def forward(self, pre_feature_map, x):
        x = self.bilinear_up(x)
        x = torch.cat((x, pre_feature_map), dim=1)
        x = self.block(x)
        return x


class UNet_ca_up_block(nn.Module):
    def __init__(self, input_channel, prev_channel, output_channel, learned_bilinear=False):
        super(UNet_ca_up_block, self).__init__()
        self.bilinear_up = UNet_up_conv_bn_relu(input_channel, prev_channel, learned_bilinear)
        self.block = basic_block(prev_channel * 2, output_channel)
        self.ca = Channel_Attention(prev_channel * 2, reduction=16)

    def forward(self, pre_feature_map, x):
        x = self.bilinear_up(x)
        x = torch.cat((x, pre_feature_map), dim=1)
        x = self.ca(x) * x
        x = self.block(x)
        return x


class UNet_resca_up_block(nn.Module):
    def __init__(self, input_channel, prev_channel, output_channel, learned_bilinear=False):
        super(UNet_resca_up_block, self).__init__()
        self.bilinear_up = UNet_up_conv_bn_relu(input_channel, prev_channel, learned_bilinear)
        self.block = basic_block(prev_channel * 2, output_channel)
        self.ca = Channel_Attention(prev_channel * 2, reduction=16)

    def forward(self, pre_feature_map, x):
        x = self.bilinear_up(x)
        x = torch.cat((x, pre_feature_map), dim=1)
        x = self.ca(x) * x + x
        x = self.block(x)
        return x


class MMFNet(nn.Module):
    def __init__(self, in_channels_1=1,in_channels_2=1, n_classes=5, pretrain=False, reduction=16, dilation=4, learned_bilinear=False):
        super(MMFNet, self).__init__()
        self.in_channels_1 = in_channels_1  # 通道数
        # self.in_channels_fa = in_channels_fa  # 通道数
        self.in_channels_2 = in_channels_2  # 通道数
        self.n_classes = n_classes
        ################# peaks encoder #################

        self.down_block1_peaks = UNet_basic_down_block(self.in_channels_2, 16)
        self.sa1_peaks = Spatial_Attention(input_channel=16, reduction=reduction, dilation=dilation)
        self.max_pool1_peaks = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_block2_peaks = UNet_basic_down_block(16, 32)
        self.sa2_peaks = Spatial_Attention(input_channel=32, reduction=reduction, dilation=dilation)
        self.max_pool2_peaks = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_block3_peaks = UNet_basic_down_block(32, 64)
        self.sa3_peaks = Spatial_Attention(input_channel=64, reduction=reduction, dilation=dilation)
        self.max_pool3_peaks = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_block4_peaks = UNet_basic_down_block(64, 128)
        self.sa4_peaks = Spatial_Attention(input_channel=128, reduction=reduction, dilation=dilation)
        self.max_pool4_peaks = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_block5_peaks = UNet_basic_down_block(128, 256)
        self.sa5_peaks = Spatial_Attention(input_channel=256, reduction=reduction, dilation=dilation)

        ################# t1 encoder #################

        self.down_block1_t1 = UNet_basic_down_block(self.in_channels_1, 16)
        self.max_pool1_t1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_block2_t1 = UNet_basic_down_block(32, 32)
        self.max_pool2_t1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_block3_t1 = UNet_basic_down_block(64, 64)
        self.max_pool3_t1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_block4_t1 = UNet_basic_down_block(128, 128)
        self.max_pool4_t1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_block5_t1 = UNet_basic_down_block(256, 256)

        ################# DCE decoder #################

        self.up_block1 = UNet_basic_up_block(512, 256, 256, learned_bilinear)
        self.up_block2 = UNet_basic_up_block(256, 128, 128, learned_bilinear)
        self.up_block3 = UNet_basic_up_block(128, 64, 64, learned_bilinear)
        self.up_block4 = UNet_basic_up_block(64, 32, 32, learned_bilinear)

        self.last_conv1 = nn.Conv2d(32, self.n_classes, 1, padding=0)

    def forward(self, t1_inputs, peaks_inputs):
        x_1 = self.down_block1_peaks(peaks_inputs)
        y = self.down_block1_t1(t1_inputs)
        y1_peaks = self.sa1_peaks(y)
        x1_t1 = y1_peaks * x_1

        x = self.max_pool1_peaks(x1_t1)
        y_1 = torch.cat((y * y1_peaks, x1_t1), dim=1)
        y = self.max_pool1_t1(y_1)

        x_2 = self.down_block2_peaks(x)
        y = self.down_block2_t1(y)
        y2_peaks = self.sa2_peaks(y)
        x2_t1 = y2_peaks * x_2

        x = self.max_pool2_peaks(x2_t1)
        y_2 = torch.cat((y * y2_peaks, x2_t1), dim=1)
        y = self.max_pool2_t1(y_2)

        x_3 = self.down_block3_peaks(x)
        y = self.down_block3_t1(y)
        y3_peaks = self.sa3_peaks(y)
        x3_t1 = y3_peaks * x_3

        x = self.max_pool3_peaks(x3_t1)
        y_3 = torch.cat((y * y3_peaks, x3_t1), dim=1)
        y = self.max_pool3_t1(y_3)

        x_4 = self.down_block4_peaks(x)
        y = self.down_block4_t1(y)
        y4_peaks = self.sa4_peaks(y)
        x4_t1 = y4_peaks * x_4

        x = self.max_pool4_peaks(x4_t1)
        y_4 = torch.cat((y * y4_peaks, x4_t1), dim=1)
        y = self.max_pool4_t1(y_4)

        x_5 = self.down_block5_peaks(x)
        y = self.down_block5_t1(y)
        y5_peaks = self.sa5_peaks(y)
        x5_t1 = x_5 * y5_peaks

        y = torch.cat((y * y5_peaks, x5_t1), dim=1)

        ################# DCE encoder #################

        y = self.up_block1(y_4, y)
        y = self.up_block2(y_3, y)
        y = self.up_block3(y_2, y)
        y = self.up_block4(y_1, y)

        final = self.last_conv1(y)

        # out = torch.sigmoid(final)
        return final
