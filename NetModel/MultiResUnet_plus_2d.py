import torch
import torch.nn as nn
from metrics_2d import init_weights, count_param
from torchsummary import summary


class mlti_res_block(nn.Module):
    def __init__(self, in_size, out_size1, out_size2, out_size3, is_batchnorm, ks=3, stride=1, padding=1):
        super(mlti_res_block, self).__init__()
        self.ks = ks # kernel_size
        self.stride = stride
        self.padding = padding
        s = stride # stride
        p = padding #（对于卷积核为3，步长为1）p=1，卷积输出尺寸与原图一样；p=0, 输出尺寸=输入尺寸-2
        if is_batchnorm:
            self.conv1_3x3 = nn.Sequential(nn.Conv2d(in_size, out_size1, ks, s, p),
                                 nn.BatchNorm2d(out_size1),
                                 nn.ReLU(inplace=True))
            self.conv2_3x3 = nn.Sequential(nn.Conv2d(out_size2, out_size2, ks, s, p),
                                 nn.BatchNorm2d(out_size2),
                                 nn.ReLU(inplace=True))
            self.conv3_1x1 = nn.Sequential(nn.Conv2d(in_size, out_size3, 1, s, 0),
                                 nn.BatchNorm2d(out_size3),
                                 nn.ReLU(inplace=True))

        else:
            self.conv1_3x3 = nn.Sequential(nn.Conv2d(in_size, out_size1, ks, s, p),
                                 nn.ReLU(inplace=True))
            self.conv2_3x3 = nn.Sequential(nn.Conv2d(out_size2, out_size2, ks, s, p),
                                 nn.ReLU(inplace=True))
            self.conv3_1x1 = nn.Sequential(nn.Conv2d(in_size, out_size3, 1, s, 0),
                                 nn.ReLU(inplace=True))

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        conv1_1 = self.conv1_3x3(x)
        conv1_2 = self.conv2_3x3(conv1_1)
        conv1_3 = self.conv3_1x1(x)
        conv1_12 = torch.cat([conv1_1, conv1_2], 1)
        con1 = conv1_12 + conv1_3
        return con1



class MultiResUnetPlus2D(nn.Module):
    # __init__不包括任何的计算逻辑，计算逻辑在forward里面
    def __init__(self, in_channels=1, n_classes=2, is_deconv=True, is_batchnorm=True):
        super(MultiResUnetPlus2D, self).__init__()

        self.in_channels = in_channels  # 通道数
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        #filters_base = [32, 64, 128, 256, 512]
        filters_base = 64
        filters_base_2 = int(filters_base/2)

        self.maxpool = nn.MaxPool2d(2)
        self.conv1_1 = mlti_res_block(self.in_channels, filters_base_2, filters_base_2, filters_base, self.is_batchnorm)
        self.conv2_1 = mlti_res_block(filters_base, filters_base, filters_base, filters_base*2, self.is_batchnorm)
        self.up1_2 = nn.ConvTranspose2d(filters_base*2, filters_base, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1_2 = mlti_res_block(filters_base*2, filters_base_2, filters_base_2, filters_base , self.is_batchnorm)
        self.conv3_1 = mlti_res_block(filters_base*2, filters_base*2, filters_base*2, filters_base*4, self.is_batchnorm)
        self.up2_2 = nn.ConvTranspose2d(filters_base*4, filters_base*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2_2 = mlti_res_block(filters_base*4, filters_base, filters_base, filters_base*2, self.is_batchnorm)
        self.up1_3 = nn.ConvTranspose2d(filters_base*2, filters_base , kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1_3 = mlti_res_block(filters_base * 3, filters_base_2, filters_base_2, filters_base, self.is_batchnorm)
        self.conv4_1 = mlti_res_block(filters_base*4, filters_base*4, filters_base*4, filters_base*8, self.is_batchnorm)
        self.up3_2 = nn.ConvTranspose2d(filters_base*8, filters_base*4 , kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3_2 = mlti_res_block(filters_base*8, filters_base*2, filters_base*2, filters_base*4, self.is_batchnorm)
        self.up2_3 = nn.ConvTranspose2d(filters_base*4, filters_base*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2_3 = mlti_res_block(filters_base*6, filters_base,filters_base,filters_base*2, self.is_batchnorm)
        self.up1_4 = nn.ConvTranspose2d(filters_base*2, filters_base, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1_4 = mlti_res_block(filters_base*4, filters_base_2, filters_base_2, filters_base, self.is_batchnorm)
        self.conv5_1 = mlti_res_block(filters_base*8, filters_base*8, filters_base*8, filters_base*16, self.is_batchnorm)
        self.up4_2 = nn.ConvTranspose2d(filters_base*16, filters_base*8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4_2 = mlti_res_block(filters_base*16, filters_base*4, filters_base*4, filters_base*8, self.is_batchnorm)
        self.up3_3 = nn.ConvTranspose2d(filters_base*8, filters_base*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3_3 = mlti_res_block(filters_base*12, filters_base*2, filters_base*2, filters_base*4, self.is_batchnorm)
        self.up2_4 = nn.ConvTranspose2d(filters_base*4, filters_base*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2_4 = mlti_res_block(filters_base*8, filters_base, filters_base, filters_base*2, self.is_batchnorm)
        self.up1_5 = nn.ConvTranspose2d(filters_base*2, filters_base, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1_5 = mlti_res_block(filters_base*5, filters_base_2, filters_base_2, filters_base, self.is_batchnorm)

        self.dropout = nn.Dropout(p=0.4)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters_base, n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):    # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
                                            # 如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False。。
                init_weights(m, init_type='kaiming')

            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')



    def forward(self, inputs):
        conv1_1 = self.conv1_1(inputs)   # 1 -> base*1
        maxpool1 = self.maxpool(conv1_1) # base*1

        conv2_1 = self.conv2_1(maxpool1) # base*1 -> base*2
        maxpool2 = self.maxpool(conv2_1) # base*2

        up1_2 = self.up1_2(conv2_1)    # base*2 -> base
        conv1_2 = torch.cat([up1_2, conv1_1], 1) # base*2
        conv1_2 = self.conv1_2(conv1_2) # base*2 -> base*1

        conv3_1 = self.conv3_1(maxpool2) # base*2 -> base*4
        maxpool3 = self.maxpool(conv3_1) # base*4

        up2_2 = self.up2_2(conv3_1) # base*4 -> base*2
        conv2_2 = torch.cat([up2_2, conv2_1], 1) # base*4
        conv2_2 = self.conv2_2(conv2_2) # base*4 -> base*2

        up1_3 = self.up1_3(conv2_2) # base*2 -> base*1
        conv1_3 = torch.cat([up1_3, conv1_1, conv1_2], 1) # base*3
        conv1_3 = self.conv1_3(conv1_3) # base*3 -> base*1

        conv4_1 = self.conv4_1(maxpool3) # base*4 -> base*8
        maxpool4 = self.maxpool(conv4_1) # base*8

        up3_2 = self.up3_2(conv4_1) # base*8 -> base*4
        conv3_2 = torch.cat([up3_2, conv3_1], 1) # base*8
        conv3_2 = self.conv3_2(conv3_2) # base*8 -> base*4

        up2_3 = self.up2_3(conv3_2) # base*4 -> base*2
        conv2_3 = torch.cat([up2_3, conv2_1, conv2_2], 1) # base*6
        conv2_3 = self.conv2_3(conv2_3) # base*6 -> base*2

        up1_4 = self.up1_4(conv2_3) # base*2 -> base*1
        conv1_4 = torch.cat([up1_4, conv1_1, conv1_2, conv1_3], 1) # base*4
        conv1_4 = self.conv1_4(conv1_4) # base*4 -> base*1

        conv5_1 = self.conv5_1(maxpool4) # base*8 -> base*16
        #conv5_1 = self.dropout(conv5_1)

        up4_2 = self.up4_2(conv5_1) # base*16 -> base*8
        conv4_2 = torch.cat([up4_2, conv4_1], 1)  # base*16
        conv4_2 = self.conv4_2(conv4_2) # base*16 -> base*8

        up3_3 = self.up3_3(conv4_2) # base*8 -> base*4
        conv3_3 = torch.cat([up3_3, conv3_1, conv3_2], 1) # base*12
        conv3_3 = self.conv3_3(conv3_3) # base*12 -> base*4

        up2_4 = self.up2_4(conv3_3) # base*4 -> base*2
        conv2_4 = torch.cat([up2_4, conv2_1, conv2_2, conv2_3], 1)# base*8
        conv2_4 = self.conv2_4(conv2_4) # base*8 -> base*2

        up1_5 = self.up1_5(conv2_4) # base*2 -> base*1
        conv1_5 = torch.cat([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], 1)# base*5
        conv1_5 = self.conv1_5(conv1_5)# base*5 -> base*1


        final = self.final(conv1_5)

        final = nn.Sigmoid()(final)
        return final


### 模型测试
if __name__ == '__main__':
    # 是否使用cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('#### Test Case ###')
    model = MultiResUnetPlus2D(1,2).to(device)
    summary(model, input_size=(1, 128,160), batch_size=16)


    x = torch.rand(1, 1, 128,160).to(device)
    y = model(x)
    param = count_param(model)  # 计算参数
    print('Input shape:', x.shape)
    print('Output shape:', y.shape)
    print('Tarameters: %.2fM (%d)' % (param / 1e6, param))






