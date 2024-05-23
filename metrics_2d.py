from torch.nn import init
import torch.nn as nn
import config_2d

num_classes = config_2d.NUM_CLASSES  # 2
from torch.nn import functional as F

### initalize the module
def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def dice(y_pred, y_true):
    smooth = 1e-7
    m1 = y_true.flatten()  # Flatten
    m2 = y_pred.flatten()  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def dicemutil(y_pred, y_true):
    smooth = 1e-7
    m1 = y_true.flatten()  # Flatten
    m2 = y_pred.flatten()  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class dice_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(dice_loss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = 0
        dice_coef_class = dice(y_pred, y_true)
        loss = 1 - dice_coef_class + loss
        return loss

def L2_loss(input, pred):
    #l2_loss = F.mse_loss(input, pred, reduction='mean')
    l2_loss = F.kl_div(input, pred,reduction='mean') # KL_loss
    return l2_loss


class dice_l2_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(dice_l2_loss, self).__init__()

    def forward(self, y_pred, y_true, x_input, x_pred):
        dice_loss_1 = 1- dice(y_pred, y_true)
        l2_loss_2 = L2_loss(x_input, x_pred)
        losses = dice_loss_1 + 0.1 * l2_loss_2
        return losses