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



class OurNet2D(nn.Module):
    def __init__(self, in_channels_1=1, n_classes=2, is_deconv=True, is_batchnorm=True):
        super(OurNet2D, self).__init__()

        self.in_channels_1 = in_channels_1  # net1通道数
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters_base = 64

        # downsampling
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = unet2dConv2d(self.in_channels_1, filters_base, self.is_batchnorm)  # 1   64
        self.conv2 = unet2dConv2d(filters_base, filters_base*2, self.is_batchnorm)        # 64  128
        self.conv3 = unet2dConv2d(filters_base*2, filters_base*4, self.is_batchnorm)        # 128 256
        self.conv4 = unet2dConv2d(filters_base*4, filters_base*8, self.is_batchnorm)        # 256 512
        self.center = unet2dConv2d(filters_base*8, filters_base*16, self.is_batchnorm)       # 512 1024
        # upsampling
        self.up_concat4 = unet2dUp(filters_base*16, filters_base*8, self.is_deconv)          # 1024  512
        self.up_concat3 = unet2dUp(filters_base*8, filters_base*4, self.is_deconv)          # 512  256
        self.up_concat2 = unet2dUp(filters_base*4, filters_base*2, self.is_deconv)          # 256  128
        self.up_concat1 = unet2dUp(filters_base*2, filters_base, self.is_deconv)          # 128  64
        # final conv (without any concat)
        self.final = nn.Conv2d(filters_base, n_classes, 1)                            # 64  2

        datafusion_channel = 4
        datafusion_filters_base = 128


        self.Conv2d5x5_1 = nn.Sequential(nn.Conv2d(datafusion_channel, datafusion_filters_base, kernel_size=5, stride=1, padding=2),
                                         nn.BatchNorm2d(datafusion_filters_base),
                                         nn.ReLU(inplace=True))

        self.Conv2d3x3 = nn.Sequential(nn.Conv2d(datafusion_filters_base, datafusion_filters_base, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(datafusion_filters_base),
                                       nn.ReLU(inplace=True))

        self.upConv2d2x2 = nn.ConvTranspose2d(datafusion_filters_base, datafusion_filters_base, kernel_size=2, stride=2, padding=0)



        self.Conv2d5x5_2 = nn.Sequential(nn.Conv2d(datafusion_filters_base, datafusion_filters_base, kernel_size=5, stride=1, padding=2),
                                         nn.BatchNorm2d(datafusion_filters_base),
                                         nn.ReLU(inplace=True))


        self.datafusionfinal = nn.Conv2d(datafusion_filters_base, n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):    # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
                # 如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False。。
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

                # filters         input          input
    def forward(self, inputs1, inputs2):                  #  in->out       1*512*512      1*64*64
        conv1 = self.conv1(inputs1)              #  1->64         64*512*512     64*64*64     unet2dConv2d-7:[-1, 64, 64, 64]
        maxpool1 = self.maxpool(conv1)          #                64*256*256     64*32*32
        conv2 = self.conv2(maxpool1)            #  64->128       128*256*256    128*32*32    unet2dConv2d-15:[-1, 128, 32, 32]
        maxpool2 = self.maxpool(conv2)          #                128*128*128    128*16*16
        conv3 = self.conv3(maxpool2)            #  128->256      256*128*128    256*16*16    unet2dConv2d-23:[-1, 256, 16, 16]
        maxpool3 = self.maxpool(conv3)          #                256*64*64      256*8*8
        conv4 = self.conv4(maxpool3)            #  128->256      256*128*128    256*16*16    unet2dConv2d-23:[-1, 256, 16, 16]
        maxpool4 = self.maxpool(conv4)

        center = self.center(maxpool4)          #  512->256      256*128*128    256*16*16    unet2dUp-53:[-1, 256, 16, 16]
        up4 = self.up_concat4(center, conv4)
        up3 = self.up_concat3(up4, conv3)     #  256->128      128*256*256    128*32*32    unet2dUp-60:[-1, 128, 32, 32]
        up2 = self.up_concat2(up3, conv2)       #  128->64       64*512*512     64*64*64     unet2dUp-67:[-1, 64, 64, 64]
        up1 = self.up_concat1(up2, conv1)
        net1 = self.final(up1)                 #  64->2         2*512*512      2*64*64      Conv2d-68:[-1, 2, 64, 64]

        net1_output = nn.Sigmoid()(net1)

        conv1_2 = self.conv1(inputs2)
        maxpool1_2 = self.maxpool(conv1_2)
        conv2_2 = self.conv2(maxpool1_2)
        maxpool2_2 = self.maxpool(conv2_2)
        conv3_2 = self.conv3(maxpool2_2)
        maxpool3_2 = self.maxpool(conv3_2)
        conv4_2 = self.conv4(maxpool3_2)
        maxpool4_2 = self.maxpool(conv4_2)
        center_2 = self.center(maxpool4_2)
        up4_2 = self.up_concat4(center_2, conv4_2)
        up3_2 = self.up_concat3(up4_2, conv3_2)
        up2_2 = self.up_concat2(up3_2, conv2_2)
        up1_2 = self.up_concat1(up2_2, conv1_2)
        net2 = self.final(up1_2)

        net2_output = nn.Sigmoid()(net2)

        datafusion_input = torch.cat([net1, net2], 1)  # 2+2=4

        conv5x5_1 = self.Conv2d5x5_1(datafusion_input)  #4 ->128
        conv3x3_1 = self.Conv2d3x3(conv5x5_1)   # 128->128
        maxpool5= self.maxpool(conv3x3_1)  # 128
        conv3x3_2 = self.Conv2d3x3(maxpool5)
        up2d2x2 = self.upConv2d2x2(conv3x3_2) # 128
        conv3x3_3 = self.Conv2d3x3(up2d2x2) # 128->128
        conv5x5_2 = self.Conv2d5x5_2(conv3x3_3) # 128->128
        datafusion =  self.datafusionfinal (conv5x5_2) #128->2

        datafusion_output = nn.Sigmoid()(datafusion)

        return net1_output, net2_output, datafusion_output

class DatafusionNet_3(nn.Module):
    def __init__(self, in_channels_1=4, n_classes=2, is_deconv=True, is_batchnorm=True):
        super(DatafusionNet_3, self).__init__()

        self.in_channels_1 = in_channels_1  # net1通道数
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters_base = 128


        self.Conv2d3x3 = nn.Sequential(nn.Conv2d(self.in_channels_1, filters_base, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(filters_base),
                                       nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(2)

        self.Conv2d3x3_2 = nn.Sequential(nn.Conv2d(filters_base, filters_base, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(filters_base),
                                         nn.ReLU(inplace=True))

        self.upConv2d2x2 = nn.ConvTranspose2d(filters_base, filters_base, kernel_size=2, stride=2, padding=0)



        self.datafusionfinal = nn.Conv2d(filters_base, n_classes, 1)


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):    # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
                # 如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False。。
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs1, inputs2, inputs3):
        intput = torch.cat([inputs1, inputs2, inputs3], 1)
        conv3x3 = self.Conv2d3x3(intput)  #4 ->128
        maxpool_1 = self.maxpool(conv3x3)   # 128->128
        conv3x3_2 = self.Conv2d3x3_2(maxpool_1)
        up2d2x2 = self.upConv2d2x2(conv3x3_2) # 128
        conv3x3_3 = self.Conv2d3x3_2(up2d2x2) # 128->128
        datafusion =  self.datafusionfinal (conv3x3_3) #128->2

        # datafusion_output = nn.Sigmoid()(datafusion)

        return  datafusion

class DatafusionNet(nn.Module):
    def __init__(self, in_channels_1=4, n_classes=2, is_deconv=True, is_batchnorm=True):
        super(DatafusionNet, self).__init__()

        self.in_channels_1 = in_channels_1  # net1通道数
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters_base = 128


        self.Conv2d3x3 = nn.Sequential(nn.Conv2d(self.in_channels_1, filters_base, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(filters_base),
                                       nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(2)

        self.Conv2d3x3_2 = nn.Sequential(nn.Conv2d(filters_base, filters_base, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(filters_base),
                                         nn.ReLU(inplace=True))

        self.upConv2d2x2 = nn.ConvTranspose2d(filters_base, filters_base, kernel_size=2, stride=2, padding=0)



        self.datafusionfinal = nn.Conv2d(filters_base, n_classes, 1)


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):    # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
                # 如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False。。
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs1, inputs2):
        intput = torch.cat([inputs1, inputs2], 1)
        conv3x3 = self.Conv2d3x3(intput)  #4 ->128
        maxpool_1 = self.maxpool(conv3x3)   # 128->128
        conv3x3_2 = self.Conv2d3x3_2(maxpool_1)
        up2d2x2 = self.upConv2d2x2(conv3x3_2) # 128
        conv3x3_3 = self.Conv2d3x3_2(up2d2x2) # 128->128
        datafusion =  self.datafusionfinal (conv3x3_3) #128->2

        # datafusion_output = nn.Sigmoid()(datafusion)

        return  datafusion

### 模型测试
if __name__ == '__main__':


    # 是否使用cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('#### Test Case ###')
    model = OurNet2D(1,2).to(device)
    #summary(model, input_size=(1, 128, 160), batch_size=128)

    x1 = torch.rand(16, 1, 128, 160).to(device)
    x2 = torch.rand(16, 1, 128, 160).to(device)

    y1,y2,fusion = model(x1,x2)
    param = count_param(model)  # 计算参数
    print('Input shape1:', x1.shape)
    print('Input shape2:', x2.shape)
    print('Output shape1:', y1.shape)
    print('Output shape2:', y2.shape)
    print('Output shape3:', fusion.shape)
    print('UNet3d totoal parameters: %.2fM (%d)' % (param / 1e6, param))



    # model = DatafusionNet(4,2).to(device)
    #
    # x1 = torch.rand(16, 2, 128, 160).to(device)
    # x2 = torch.rand(16, 2, 128, 160).to(device)
    #
    # datafusion = model(x1,x2)
    # param = count_param(model)  # 计算参数
    # print('Input shape1:', x1.shape)
    # print('Input shape2:', x2.shape)
    # print('Output shape3:', datafusion.shape)
    # print('UNet3d totoal parameters: %.2fM (%d)' % (param / 1e6, param))



