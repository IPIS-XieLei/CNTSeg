#import visdom
import config_2d
from NetModel import Unet_2d, Unet_plus_2d, MultiResUnet_2d, MultiResUnet_plus_2d, Newnet,se2dunet
import torch, time
import math
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset_P import CN_MyTrainDataset
from torchvision.transforms import transforms
# from metrics_2d import dice_loss, dice, dice_l2_loss
import os
import numpy as np
from dice_loss import DiceLoss,BinaryDiceLoss
from mutilloss import SoftDiceLoss,diceCoeffv2,diceCoeff,SoftDiceLoss1
import math
# unet2dse = se2dunet.UNet2Dse
unet2d = Unet_2d.UNet2D                             # U-Net
unetplus2d = Unet_plus_2d.UNetPlus2D                # U-Net++
multiresunet2d = MultiResUnet_2d.MultiResUnet2D     # MultiRes U-Net

ournet = Newnet.OurNet2D
fusionnet = Newnet.DatafusionNet

patch_size_w = config_2d.PATCH_SIZE_W
patch_size_h = config_2d.PATCH_SIZE_H
batch_size = config_2d.BATCH_SIZE
n_epochs = config_2d.NUM_EPOCHS
n_classes = config_2d.NUM_CLASSES
image_rows = config_2d.VOLUME_ROWS
image_cols = config_2d.VOLUME_COLS
image_depth = config_2d.VOLUME_DEPS
test_imgs_path = config_2d.test_imgs_path

flag_gpu = config_2d.FLAG_GPU
flag_model = config_2d.MODEL_name

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# train
def train():
    global model, losses, train_dataset, train_dataloader, val_dataset, val_dataloader

    # op_lr = 0.002 #

    op_lr = 0.0002  #
    # 训练集选择
    #For model 4 (CTL_t1+fa)
    CN_train_x_t1_dir = 'CN_mydata/train_T1_FA/x_t1_data/'
    CN_train_x_fa_dir = 'CN_mydata/train_T1_FA/x_fa_data/'
    CN_train_x_peaks_dir = 'CN_mydata/train_T1_FA/x_peaks_data/'
    CN_train_y_on_dir = 'CN_mydata/train_T1_FA/y_data_on/'
    CN_train_y_tgn_dir = 'CN_mydata/train_T1_FA/y_data_tgn/'
    CN_train_y_fvn_dir = 'CN_mydata/train_T1_FA/y_data_fvn/'
    CN_train_y_ocn_dir = 'CN_mydata/train_T1_FA/y_data_ocn/'


    # 损失函数选择
    # losses = dice_loss()
    # dice =BinaryDiceLoss()
    losses_1= SoftDiceLoss1(4,activation='sigmoid').cuda()
    losses_2= nn.BCEWithLogitsLoss().cuda()

    model = unet2d(9, 5).to(device)

    if flag_gpu == 1:
        model = nn.DataParallel(model).cuda()

    x_transforms = transforms.ToTensor()

    y_transforms = transforms.ToTensor()
    train_dataset = CN_MyTrainDataset(CN_train_x_t1_dir, CN_train_x_fa_dir, CN_train_x_peaks_dir,
                                      CN_train_y_on_dir, CN_train_y_tgn_dir, CN_train_y_fvn_dir, CN_train_y_ocn_dir,
                                      x_transform=x_transforms, y_transform=y_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    # val_dataset = CN_MyTrainDataset(CN_val_x_t1_dir, CN_val_x_fa_dir, CN_val_x_peaks_dir,
    #                                   CN_val_y_on_dir, CN_val_y_tgn_dir, CN_val_y_fvn_dir, CN_val_y_ocn_dir,
    #                                   x_transform=x_transforms, y_transform=y_transforms)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 模型载入
    # model.load_state_dict(torch.load('outputs_CN_Peaks/CN_213_epoch_64_batch.pth', map_location='cuda'))

###----------------------
#### start train
    print('-' * 30)
    print('Training start...')
    print('-' * 30)
    print('patch size   : ', patch_size_w, 'x', patch_size_h)
    print('batch size   : ', batch_size)
    print('  epoch      : ', n_epochs)
    print('learning rate peak: ', op_lr)
    print('-' * 30)

    optimizer = optim.Adam(model.parameters(), lr=op_lr)
    t = 10  # warmup
    T = 200  # 共有200个epoch，则用于cosine rate的一共有180个epoch
    n_t = 0.5
    lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
            1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
            1 + math.cos(math.pi * (epoch - t) / (T - t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    for epoch in range(0, 500):

        dt_size = len(train_dataloader.dataset)
        # epoch_loss_t1fa = 0
        step = 0
        loss_avg, mean_dice_avg, on_dice_avg, tgn_dice_avg, fvn_dice_avg, ocn_dice_avg  = 0, 0, 0, 0, 0, 0

        model.train()
        for x1, x2, x3, y0, y1, y2, y3, y4 in train_dataloader:
            # o_dice, t_dice, f_dice, oc_dice = 0, 0, 0, 0
            step += 1
            # inputs1 = x1.to(device)  # [batch_size, 9, 144, 144]->model(9,2)-> output:[batch_size, 2, 144, 144]
            # inputs2 = x2.to(device)
            inputs3 = x3.to(device)
            groundtruth0 = y0.to(device)
            groundtruth1 = y1.to(device)  # on
            groundtruth2 = y2.to(device)  # ocn
            groundtruth3 = y3.to(device)  # tgn
            groundtruth4 = y4.to(device)  # fvn

            groundtruth = torch.cat([groundtruth0, groundtruth1, groundtruth2, groundtruth3, groundtruth4], 1)
            # 梯度清零
            optimizer.zero_grad()

            # input = torch.cat([inputs1, inputs2], 1)

            outputs = model(inputs3)  # FA

            # predict = outputs[:, 4, :, :].squeeze()  # label预测值      tensor[batch_size, 1, 144, 144]

            # y_truth = groundtruth.squeeze()  # label真实值      tensor[batch_size, 1, 144, 144]
            losses_ce = losses_2(outputs, groundtruth)
            # loss_t1fa = losses(outputs, groundtruth)
            loss_dl = losses_1(outputs, groundtruth)
            loss_t1fa=loss_dl+losses_ce
            # label_dice_t1fa = dice(outputs, groundtruth)
            output = torch.sigmoid(outputs)
            output[output < 0.5] = 0
            output[output > 0.5] = 1
            on_dice = diceCoeffv2(output[:, 1, :,:], groundtruth[:, 1, :,:], activation=None).cpu().item()
            tgn_dice = diceCoeffv2(output[:, 2, :,:], groundtruth[:, 2, :,:], activation=None).cpu().item()
            fvn_dice = diceCoeffv2(output[:, 3, :,:], groundtruth[:, 3, :,:], activation=None).cpu().item()
            ocn_dice = diceCoeffv2(output[:, 4, :,:], groundtruth[:, 4, :,:], activation=None).cpu().item()
            mean_dice = (on_dice + tgn_dice+ fvn_dice+ ocn_dice) / 4

            loss_avg += loss_t1fa
            mean_dice_avg += mean_dice
            on_dice_avg += on_dice
            tgn_dice_avg += tgn_dice
            fvn_dice_avg += fvn_dice
            ocn_dice_avg += ocn_dice

            # 反向传播
            loss_t1fa.backward()

            # 梯度更新
            optimizer.step()

            # epoch_loss_CL_t1fa += float(loss_CL_t1fa.item())
            # epoch_dice_CL_t1fa += float(label_dice_CL_t1fa.item())
            step_loss_t1fa = loss_t1fa.item()

            print("epoch:%d/%d, %d/%d, loss_CN:%0.3f, op_lr:%0.5f, Train Mean Dice :%0.3f, on_dice :%0.3f, tgn_dice :%0.3f, fvn_dice :%0.3f, ocn_dice :%0.3f" % (epoch + 1,
                                                                                             n_epochs,
                                                                                             step * train_dataloader.batch_size,
                                                                                             dt_size,
                                                                                             step_loss_t1fa,optimizer.param_groups[0]['lr'],mean_dice, on_dice, tgn_dice, fvn_dice, ocn_dice))
        scheduler.step()
        print("epoch: %d/%d done, loss_avg :%0.3f, mean_dice_avg :%0.3f, on_dice_avg :%0.3f, tgn_dice_avg :%0.3f, fvn_dice_avg :%0.3f, ocn_dice_avg :%0.3f" %  (
            epoch + 1, n_epochs, loss_avg/step, mean_dice_avg/step, on_dice_avg/step, tgn_dice_avg/step, fvn_dice_avg/step, ocn_dice_avg/step))
        model_path = 'outputs_CN_Peaks/' + 'CN_%d_epoch_%d_batch.pth' % (epoch + 1, batch_size)
        torch.save(model.state_dict(), model_path)


        print('-' * 30)




if __name__ == '__main__':

    # x_transforms = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomHorizontalFlip(0.5),
    #     transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.3),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])
    #
    # y_transforms = transforms.Compose([
    #     transforms.ToTensor()
    # ])

    # 模型保存
    if 'outputs_CN_Peaks' not in os.listdir(os.curdir):
        os.mkdir('outputs_CN_Peaks')

    # loss保存
    if 'loss' not in os.listdir(os.curdir):
        os.mkdir('loss')


    ### train test ###
    start_time = time.time()
    train()
    end_time = time.time()
    print("2D train time is {:.3f} mins".format((end_time - start_time) / 60.0))
    print('-' * 30)
    print('patch size   : ', patch_size_w,'x',patch_size_h)
    print('batch size   : ', batch_size)
    print('  epoch      : ', n_epochs)
    print('-' * 30)
    print("done")




