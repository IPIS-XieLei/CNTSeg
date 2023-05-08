import torch.nn as nn
import torch

import numpy as np


def diceCoeff(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss =  (2 * intersection + eps) / (unionset + eps)

    return loss.sum() / N


def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N






class SoftDiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, num_classes, activation=None, reduction='mean'):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []

        for i in range(1, self.num_classes+1):
            class_dice.append(diceCoeff(y_pred[:, i, :,:], y_true[:, i, :,:], activation=self.activation))

        # mean_dice = class_dice[0]+class_dice[1]+2*class_dice[2]+class_dice[3] / len(class_dice)+1


        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice

class SoftDiceLoss1(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, num_classes, activation=None, reduction='mean'):
        super(SoftDiceLoss1, self).__init__()
        self.activation = activation
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []

        for i in range(1, self.num_classes+1):
            class_dice.append(diceCoeff(y_pred[:, i, :,:], y_true[:, i, :,:], activation=self.activation))

        mean_dice = 0.1*class_dice[0]+0.3*class_dice[1]+0.3*class_dice[2]+0.3*class_dice[3] / len(class_dice)+1


        # mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice