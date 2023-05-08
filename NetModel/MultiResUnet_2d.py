import torch
import torch.nn as nn
from metrics_2d import init_weights, count_param
from torchsummary import summary


class mlti_res_block(nn.Module):
    def __init__(self, in_size, filter_size1, filter_size2, filter_size3, filter_size4, is_batchnorm=True, ks=3, stride=1, padding=1):
        super(mlti_res_block, self).__init__()
        self.ks = ks # kernel_size
        self.stride = stride
        self.padding = padding
        s = stride # stride
        p = padding #（对于卷积核为3，步长为1）p=1，卷积输出尺寸与原图一样；p=0, 输出尺寸=输入尺寸-2
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, filter_size1, ks, s, p),
                                 nn.BatchNorm2d(filter_size1),
                                 nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv2d(filter_size1, filter_size2, ks, s, p),
                                 nn.BatchNorm2d(filter_size2),
                                 nn.ReLU(inplace=True),)
            self.conv3 = nn.Sequential(nn.Conv2d(filter_size2, filter_size3, ks, s, p),
                                 nn.BatchNorm2d(filter_size3),
                                 nn.ReLU(inplace=True),)
            self.cnn = nn.Sequential(nn.Conv2d(filter_size3, filter_size4, 1, s, p),
                                 nn.BatchNorm2d(filter_size4),
                                 nn.ReLU(inplace=True),)

        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, filter_size1, ks, s, p),
                                 nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv2d(filter_size1, filter_size2, ks, s, p),
                                 nn.ReLU(inplace=True),)
            self.conv3 = nn.Sequential(nn.Conv2d(filter_size2, filter_size3, ks, s, p),
                                 nn.ReLU(inplace=True),)
            self.cnn = nn.Sequential(nn.Conv2d(in_size, filter_size4, 1, s, 0),
                                 nn.ReLU(inplace=True),)
        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        cnn = self.cnn(x)
        concat = torch.cat([conv1, conv2, conv3], 1)
        add = cnn + concat
        return add



class res_path(nn.Module):
    def __init__(self, in_size, filter_size, path_number):
        super(res_path, self).__init__()

        self.path_num = path_number
        self.filter_size = filter_size

        self.cnn1= nn.Sequential(nn.Conv2d(in_size, filter_size, 3, 1, 1),
                              nn.ReLU(inplace=True))
        self.cnn2= nn.Sequential(nn.Conv2d(filter_size, filter_size, 3, 1, 1),
                              nn.ReLU(inplace=True))
        self.cnn3 = nn.Sequential(nn.Conv2d(in_size, filter_size, 1, 1, 0),
                              nn.ReLU(inplace=True),)
        self.cnn4 = nn.Sequential(nn.Conv2d(filter_size, filter_size, 1, 1, 0),
                              nn.ReLU(inplace=True),)


    def forward(self, input):
        res = input

        if self.path_num<=4:
            cnn1 = self.cnn1(res)  #3x3
            cnn2 = self.cnn3(res)  #1x1
            res = cnn1 + cnn2
            if self.path_num<=3:
                cnn1 = self.cnn2(res)  #3x3
                cnn2 = self.cnn4(res)  #1x1
                res = cnn1 + cnn2
                if self.path_num<=2:
                    cnn1 = self.cnn2(res)  #3x3
                    cnn2 = self.cnn4(res)  #1x1
                    res = cnn1 + cnn2
                    if self.path_num <= 1:
                        cnn1 = self.cnn2(res)  #3x3
                        cnn2 = self.cnn4(res)  #1x1
                        res = cnn1 + cnn2

        return res


class MultiResUnet2D(nn.Module):
    # __init__不包括任何的计算逻辑，计算逻辑在forward里面
    def __init__(self, in_channels=1, n_classes=2, is_deconv=True, is_batchnorm=False):
        super(MultiResUnet2D, self).__init__()

        self.in_channels = in_channels  # 通道数
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [51,105,209,426,853]


        # downsampling
        self.maxpool = nn.MaxPool2d(2)
        self.res_block1 = mlti_res_block(self.in_channels, 8, 17, 26, 51, self.is_batchnorm)
        self.res_block2 = mlti_res_block(filters[0], 17, 35, 53, 105, self.is_batchnorm)
        self.res_block3 = mlti_res_block(filters[1], 31, 72, 106, 209, self.is_batchnorm)
        self.res_block4 = mlti_res_block(filters[2], 71, 142, 213, 426, self.is_batchnorm)
        self.res_block5 = mlti_res_block(filters[3], 142, 284, 427, 853, self.is_batchnorm)
        self.res_block6 = mlti_res_block(256+filters[4], 71, 142, 213, 426, self.is_batchnorm)
        self.res_block7 = mlti_res_block(128+filters[3], 31, 72, 106, 209, self.is_batchnorm)
        self.res_block8 = mlti_res_block(64+filters[2], 17, 35, 53, 105, self.is_batchnorm)
        self.res_block9 = mlti_res_block(32+filters[1], 8, 17, 26, 51, self.is_batchnorm)

        self.res_path4 = res_path(filters[3], 256 , 4)
        self.res_path3 = res_path(filters[2], 128 , 3)
        self.res_path2 = res_path(filters[1], 64 , 2)
        self.res_path1 = res_path(filters[0], 32 , 1)

        self.UpSampling5 = nn.ConvTranspose2d(filters[4], filters[4], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.UpSampling6 = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.UpSampling7 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.UpSampling8 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):    # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
                                            # 如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False。。
                init_weights(m, init_type='kaiming')

            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')



    def forward(self, inputs):

        res_block1 = self.res_block1(inputs)
        maxpool1 = self.maxpool(res_block1)

        res_block2 = self.res_block2(maxpool1)
        maxpool2 = self.maxpool(res_block2)

        res_block3 = self.res_block3(maxpool2)
        maxpool3 = self.maxpool(res_block3)

        res_block4 = self.res_block4(maxpool3)
        maxpool4 = self.maxpool(res_block4)

        res_block5 = self.res_block5(maxpool4)
        upsample5 = self.UpSampling5(res_block5)

        res_path4 = self.res_path4(res_block4)
        concat4 = torch.cat([upsample5, res_path4],1)

        res_block6 = self.res_block6(concat4)
        upsample6 = self.UpSampling6(res_block6)

        res_path3 = self.res_path3(res_block3)
        concat3 = torch.cat([upsample6, res_path3], 1)

        res_block7 = self.res_block7(concat3)
        upsample7 = self.UpSampling7(res_block7)

        res_path2 = self.res_path2(res_block2)
        concat2 = torch.cat([upsample7, res_path2], 1)

        res_block8 = self.res_block8(concat2)
        upsample8 = self.UpSampling8(res_block8)

        res_path1 = self.res_path1(res_block1)
        concat1 = torch.cat([upsample8, res_path1],1)

        res_block9 = self.res_block9(concat1)


        final = self.final(res_block9)

        final = nn.Sigmoid()(final)
        return final




### 模型测试
if __name__ == '__main__':
    # 是否使用cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('#### Test Case ###')
    model = MultiResUnet2D(1,2).to(device)
    summary(model, input_size=(1, 128,160), batch_size=16)


    x = torch.rand(2, 1, 128,160).to(device)
    y = model(x)
    param = count_param(model)  # 计算参数
    print('Input shape:', x.shape)
    print('Output shape:', y.shape)
    print('Totoal parameters: %.2fM (%d)' % (param / 1e6, param))






