import torch.nn as nn
import torch
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import numpy as np
import torch.nn.functional as F
from skimage import measure
from einops import rearrange, repeat
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

        mean_dice = 0.1*class_dice[0]+0.3*class_dice[1]+0.3*class_dice[2]+0.3*class_dice[3] / len(class_dice)
        # mean_dice = 0.1 * class_dice[0] + 0.4 * class_dice[1] + 0.25 * class_dice[2] + 0.25 * class_dice[3] / len(
        #     class_dice)


        # mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """
    assert len(img_gt.shape) == len(out_shape) == 4
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)
    for b in range(out_shape[0]):  # batch size
        for c in range(out_shape[1]):  # channel
            posmask = img_gt[b][c].astype(bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis) + 1e-7) - (
                            posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis) + 1e-7)
                # sdf = (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis)) - (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis))
                sdf[boundary == 1] = 0
                normalized_sdf[b][c] = sdf
    return normalized_sdf


class SDFLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self):
        super(SDFLoss, self).__init__()


    def forward(self, out_dis, gt_dis):
        # b, c, h, w = out_dis.shape
        class_dis = []
        for i in range(1, 4):
            class_dis.append(
                F.mse_loss(out_dis[:, i, :, :], gt_dis[:, i, :, :]) + torch.norm(out_dis[:, i, :, :] - gt_dis[:, i, :, :], 1) / torch.numel(
                    out_dis[:, i, :, :]))
        mean_dis = sum(class_dis) / len(class_dis)
        return mean_dis

# def compute_Islands(img_gt):
#     """
#     compute the signed distance map of binary mask
#     input: segmentation, shape = (batch_size,c, x, y, z)
#     output: the Signed Distance Map (SDM)
#     sdf(x) = 0; x in segmentation boundary
#              -inf|x-y|; x in segmentation
#              +inf|x-y|; x out of segmentation
#     normalize sdf to [-1,1]
#
#     """
#     # out_shape =
#     # img_gt = img_gt.astype(np.uint8)
#     islands_num = np.zeros([img_gt.shape[0],img_gt.shape[1]-1])
#
#     for b in range(img_gt.shape[0]):  # batch size
#         for c in range(1,img_gt.shape[1]):  # batch size
#             islands_data = img_gt[b][c]
#             _, num = measure.label(islands_data, connectivity=1, background=0,
#                                    return_num=True)  # num可能要Unsqueeze
#             # sdf[boundary == 1] = 0
#             islands_num[b][c-1] = num
#                 # assert np.min(sdf) == -1.0
#                 # assert np.max(sdf) == 1.0
#
#
#     return islands_num
#
#
# class RCLoss(nn.Module):
#     __name__ = 'dice_loss'
#
#     def __init__(self):
#         super(RCLoss, self).__init__()
#
#
#     def forward(self, out_islands, gt_islands):
#         # b, c, h, w = out_dis.shape
#         class_dis = []
#         for i in range(0, 4):
#             class_dis.append(F.huber_loss(out_islands[:, i], gt_islands[:, i]))
#         mean_dis = sum(class_dis) / len(class_dis)
#         return mean_dis
# def compute_Islands1(img_gt):
#     """
#     compute the signed distance map of binary mask
#     input: segmentation, shape = (batch_size,c, x, y, z)
#     output: the Signed Distance Map (SDM)
#     sdf(x) = 0; x in segmentation boundary
#              -inf|x-y|; x in segmentation
#              +inf|x-y|; x out of segmentation
#     normalize sdf to [-1,1]
#
#     """
#     # out_shape =
#     # img_gt = img_gt.astype(np.uint8)
#     islands_num = np.zeros([img_gt.shape[0],img_gt.shape[1]-1,img_gt.shape[2],img_gt.shape[3]])
#
#     for b in range(img_gt.shape[0]):  # batch size
#         for c in range(1,img_gt.shape[1]):  # batch size
#             islands_data = img_gt[b][c]
#             nummap, num = measure.label(islands_data, connectivity=1, background=0,
#                                    return_num=True)  # num可能要Unsqueeze
#             # sdf[boundary == 1] = 0
#             islands_num[b][c-1] = nummap
#                 # assert np.min(sdf) == -1.0
#                 # assert np.max(sdf) == 1.0
#
#
#     return islands_num
#
#
# class RCLoss1(nn.Module):
#     __name__ = 'dice_loss'
#
#     def __init__(self):
#         super(RCLoss1, self).__init__()
#
#
#     def forward(self, out_islands, gt_islands):
#         # b, c, h, w = out_dis.shape
#         class_dis = []
#         for i in range(0, 4):
#             class_dis.append(F.mse_loss(out_islands[:, i,:,:], gt_islands[:, i,:,:]))
#         mean_dis = sum(class_dis) / len(class_dis)
#         return mean_dis