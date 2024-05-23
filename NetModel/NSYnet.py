import torch
import torch.nn as nn
from metrics_2d import init_weights, count_param
from torchsummary import summary
from torch.autograd import Variable

class unet2dConv2d(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unet2dConv2d, self).__init__()
        self.n = n  # 第n层
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, stride=stride, padding=padding),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)   # 如果属性不存在会创建一个新的对象属性，并对属性赋值
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
            conv = getattr(self, 'conv%d' % i) # getattr函数用于返回一个对象属性值。
            x = conv(x)

        return x


class unet2dUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unet2dUp, self).__init__()
        self.conv = unet2dConv2d(in_size , out_size, is_batchnorm=True)
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

class NSYnet(nn.Module):
    def __init__(self, in_channels_1=1, in_channels_2=1,n_classes=5, is_deconv=True, is_batchnorm=True):
        super(NSYnet, self).__init__()

        self.in_channels_1 = in_channels_1  # 通道数
        self.in_channels_2 = in_channels_2  # 通道数
        # self.in_channels_peaks = in_channels_peaks  # 通道数

        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters_base =16

        ## T1
        self.maxpool_t1 = nn.MaxPool2d(2)
        self.conv1_t1 = unet2dConv2d(self.in_channels_1, filters_base, self.is_batchnorm)  # 1   64
        self.conv2_t1 = unet2dConv2d(filters_base, filters_base * 2, self.is_batchnorm)  # 64  128
        self.conv3_t1 = unet2dConv2d(filters_base * 2, filters_base * 4, self.is_batchnorm)  # 128 256
        self.conv4_t1 = unet2dConv2d(filters_base * 4, filters_base * 8, self.is_batchnorm)# 128 256
        self.conv5_t1 = unet2dConv2d(filters_base * 8, filters_base * 16, self.is_batchnorm)  # 128 256

        ## FA
        self.maxpool_fa = nn.MaxPool2d(2)
        self.conv1_fa = unet2dConv2d(self.in_channels_2, filters_base, self.is_batchnorm)
        self.conv2_fa = unet2dConv2d(filters_base, filters_base * 2, self.is_batchnorm)
        self.conv3_fa = unet2dConv2d(filters_base * 2, filters_base * 4, self.is_batchnorm)
        self.conv4_fa = unet2dConv2d(filters_base * 4, filters_base * 8, self.is_batchnorm)
        self.conv5_fa = unet2dConv2d(filters_base * 8, filters_base * 16, self.is_batchnorm)

        self.fusion = unet2dConv2d(filters_base * 32, filters_base * 16, self.is_batchnorm)

        # upsampling
        self.up0 = nn.ConvTranspose2d(filters_base * 16, filters_base * 8, kernel_size=2, stride=2, padding=0)
        self.upconv0 = unet2dConv2d(filters_base * 8 * 3, filters_base * 8, self.is_batchnorm)
        self.up1 = nn.ConvTranspose2d(filters_base*8, filters_base*4, kernel_size=2, stride=2, padding=0)
        self.upconv1 = unet2dConv2d(filters_base*4*3, filters_base*4, self.is_batchnorm)
        self.up2 = nn.ConvTranspose2d(filters_base*4, filters_base*2, kernel_size=2, stride=2, padding=0)
        self.upconv2 = unet2dConv2d(filters_base*2*3, filters_base*2, self.is_batchnorm)
        self.up3 = nn.ConvTranspose2d(filters_base*2, filters_base, kernel_size=2, stride=2, padding=0)
        self.upconv3 = unet2dConv2d(filters_base*3, filters_base, self.is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters_base, n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs_t1, inputs_fa):
        conv1_t1 = self.conv1_t1(inputs_t1)
        maxpool1_t1 = self.maxpool_t1(conv1_t1)
        conv2_t1 = self.conv2_t1(maxpool1_t1)
        maxpool2_t1 = self.maxpool_t1(conv2_t1)
        conv3_t1 = self.conv3_t1(maxpool2_t1)
        maxpool3_t1 = self.maxpool_t1(conv3_t1)
        conv4_t1 = self.conv4_t1(maxpool3_t1)
        maxpool4_t1 = self.maxpool_t1(conv4_t1)
        conv5_t1 = self.conv5_t1(maxpool4_t1)

        conv1_fa = self.conv1_fa(inputs_fa)
        maxpool1_fa = self.maxpool_fa(conv1_fa)
        conv2_fa = self.conv2_fa(
            maxpool1_fa)
        maxpool2_fa = self.maxpool_fa(conv2_fa)
        conv3_fa = self.conv3_fa(
            maxpool2_fa)
        maxpool3_fa = self.maxpool_fa(conv3_fa)
        conv4_fa = self.conv4_fa(maxpool3_fa)
        maxpool4_fa = self.maxpool_fa(conv4_fa)
        conv5_fa = self.conv5_fa(maxpool4_fa)

        fusion = torch.cat([conv5_t1, conv5_fa], 1)
        fusionconv = self.fusion(
            fusion)

        up0 = self.up0(fusionconv)
        cat0 = torch.cat([up0, conv4_t1, conv4_fa], 1)
        upcon0 = self.upconv0(cat0)

        up1 = self.up1(upcon0)
        cat1 = torch.cat([up1, conv3_t1, conv3_fa], 1)
        upcon1 = self.upconv1(cat1)

        up2 = self.up2(upcon1)
        cat2 = torch.cat([up2, conv2_t1, conv2_fa], 1)
        upcon2 = self.upconv2(cat2)

        up3 = self.up3(upcon2)
        cat3 = torch.cat([up3, conv1_t1, conv1_fa], 1)
        upcon3 = self.upconv3(cat3)
        final = self.final(upcon3)


        return final
### 模型测试
if __name__ == '__main__':


    # 是否使用cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('#### Test Case ###')
    model = NSYnet(1,3,5).to(device)
    #summary(model, input_size=(1, 128, 160), batch_size=128)

    x1 = torch.rand(16, 1, 144, 144).to(device)
    # x2 = torch.rand(16, 1, 144, 144).to(device)
    # x3 = torch.rand(16, 1, 144, 144).to(device)
    x4 = torch.rand(16, 3, 144, 144).to(device)
    # x5 = torch.rand(16, 9, 144, 144).to(device)

    y = model(x1, x4)
    param = count_param(model)  # 计算参数
    # print('Input shape:', x.shape)
    print('Output shape:', y.shape)
    print('UNet3d totoal parameters: %.2fM (%d)' % (param / 1e6, param))