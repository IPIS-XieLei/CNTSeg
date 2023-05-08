#import visdom
import config_2d
from NetModel import Unet_2d, Unet_plus_2d, MultiResUnet_2d, MultiResUnet_plus_2d, Newnet
import torch, time
import math
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset_P import CN_MyTrainDataset
from torchvision.transforms import transforms
# from metrics_2d import dice_loss, dice, dice_l2_loss
import os
from dice_loss import DiceLoss,BinaryDiceLoss
from mutilloss import SoftDiceLoss,diceCoeffv2,diceCoeff,SoftDiceLoss1
import math

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

    op_lr = 0.0002 #
    # op_lr = 0.002  #

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
    losses_1 = SoftDiceLoss(4, activation='sigmoid').cuda()
    losses_2 = nn.BCEWithLogitsLoss().cuda()

    model_t1 = unet2d(1, 5).to(device)
    model_peak = unet2d(9, 5).to(device)
    model_fusion = fusionnet(10, 5).to(device)

    if flag_gpu == 1:
        model_t1 = nn.DataParallel(model_t1).cuda()
        model_peak = nn.DataParallel(model_peak).cuda()
        model_fusion = nn.DataParallel(model_fusion).cuda()

    x_transforms = transforms.ToTensor()

    y_transforms = transforms.ToTensor()
    train_dataset = CN_MyTrainDataset(CN_train_x_t1_dir, CN_train_x_fa_dir, CN_train_x_peaks_dir,
                                      CN_train_y_on_dir, CN_train_y_tgn_dir, CN_train_y_fvn_dir, CN_train_y_ocn_dir,
                                      x_transform=x_transforms, y_transform=y_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)


    # # 模型载入
    model_t1.load_state_dict(torch.load('outputs_CN_T1/CN_500_epoch_64_batch.pth', map_location='cuda'))
    model_peak.load_state_dict(torch.load('outputs_CN_Peaks/CN_500_epoch_64_batch.pth', map_location='cuda'))
    model_fusion.load_state_dict(torch.load('outputs_CN_T1_Peaks/CN_500_epoch_64_batch.pth', map_location='cuda'))

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

    optimizer = optim.Adam(model_fusion.parameters(), lr=op_lr)
    t = 10  # warmup
    T = 200  # 共有200个epoch，则用于cosine rate的一共有180个epoch
    n_t = 0.5
    lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
            1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
            1 + math.cos(math.pi * (epoch - t) / (T - t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    for epoch in range(262, 500):

        dt_size = len(train_dataloader.dataset)
        # epoch_loss_t1fa = 0
        step = 0
        loss_avg, mean_dice_avg, on_dice_avg, tgn_dice_avg, fvn_dice_avg, ocn_dice_avg  = 0, 0, 0, 0, 0, 0
        model_t1.eval()
        model_peak.eval()
        model_fusion.train()
        for x1, x2, x3, y0, y1, y2, y3, y4 in train_dataloader:
            # o_dice, t_dice, f_dice, oc_dice = 0, 0, 0, 0
            step += 1
            inputs_t1 = x1.to(device)  # [batch_size, 9, 144, 144]->model(9,2)-> output:[batch_size, 2, 144, 144]
            # inputs_fa = x2.to(device)
            inputs_peaks = x3.to(device)
            groundtruth0 = y0.to(device)
            groundtruth1 = y1.to(device)
            groundtruth2 = y2.to(device)
            groundtruth3 = y3.to(device)
            groundtruth4 = y4.to(device)

            groundtruth = torch.cat([groundtruth0, groundtruth1, groundtruth2, groundtruth3, groundtruth4], 1)
            # 梯度清零
            optimizer.zero_grad()

            # input = torch.cat([inputs1, inputs2], 1)

            outputs_t1 = model_t1(inputs_t1)  # FA
            outputs_peak = model_peak(inputs_peaks)  # FA
            # outputs_t1 = torch.sigmoid(outputs_t1)
            # outputs_peak = torch.sigmoid(outputs_peak)
            outputs_fusion = model_fusion(outputs_t1,outputs_peak)  # FA

            # predict = outputs[:, 4, :, :].squeeze()  # label预测值      tensor[batch_size, 1, 144, 144]

            # y_truth = groundtruth.squeeze()  # label真实值      tensor[batch_size, 1, 144, 144]
            losses_ce = losses_2(outputs_fusion, groundtruth)
            # loss_t1fa = losses(outputs, groundtruth)
            loss_dl = losses_1(outputs_fusion, groundtruth)
            loss_t1_fa = loss_dl + losses_ce
            # loss_t1_fa = losses(outputs_fusion, groundtruth)

            # label_dice_t1fa = dice(outputs, groundtruth)
            output = torch.sigmoid(outputs_fusion)
            output[output < 0.5] = 0
            output[output > 0.5] = 1
            on_dice = diceCoeffv2(output[:, 1, :,:], groundtruth[:, 1, :,:], activation=None).cpu().item()
            tgn_dice = diceCoeffv2(output[:, 2, :,:], groundtruth[:, 2, :,:], activation=None).cpu().item()
            fvn_dice = diceCoeffv2(output[:, 3, :,:], groundtruth[:, 3, :,:], activation=None).cpu().item()
            ocn_dice = diceCoeffv2(output[:, 4, :,:], groundtruth[:, 4, :,:], activation=None).cpu().item()
            mean_dice = (on_dice + tgn_dice+ fvn_dice+ ocn_dice) / 4

            loss_avg += loss_t1_fa
            mean_dice_avg += mean_dice
            on_dice_avg += on_dice
            tgn_dice_avg += tgn_dice
            fvn_dice_avg += fvn_dice
            ocn_dice_avg += ocn_dice

            # 反向传播
            loss_t1_fa.backward()

            # 梯度更新
            optimizer.step()

            # epoch_loss_CL_t1fa += float(loss_CL_t1fa.item())
            # epoch_dice_CL_t1fa += float(label_dice_CL_t1fa.item())
            step_loss_t1fa = loss_t1_fa.item()
            # step_dice_t1fa = label_dice_t1fa.item()

            # if step % 10 == 0:
            #     with open(r'loss/2DUnet_train_CN_' + str(batch_size) + 'batch_step_loss.txt', 'a+') as f:
            #         f.writelines('step{0}\t{1} \n'.format(str(step), str(step_loss_t1fa)))
            # print(
            #     'epoch:%d/%d, %d/%d,Train Mean Dice {:.4}, on_dice {:.4}, tgn_dice {:.4}, fvn_dice {:.4}, ocn_dice {:.4}'.format(
            #         epoch + 1,
            #         n_epochs,
            #         step * train_dataloader.batch_size,
            #         dt_size,
            #        mean_dice, o_dice, t_dice, f_dice, oc_dice
            #         ))
            print("epoch:%d/%d, %d/%d, loss_CN:%0.3f, op_lr:%0.5f, Train Mean Dice :%0.3f, on_dice :%0.3f, tgn_dice :%0.3f, fvn_dice :%0.3f, ocn_dice :%0.3f" % (epoch + 1,
                                                                                             n_epochs,
                                                                                             step * train_dataloader.batch_size,
                                                                                             dt_size,
                                                                                             step_loss_t1fa,optimizer.param_groups[0]['lr'],mean_dice, on_dice, tgn_dice, fvn_dice, ocn_dice))
        scheduler.step()
        print("epoch: %d/%d done, loss_avg :%0.3f, mean_dice_avg :%0.3f, on_dice_avg :%0.3f, tgn_dice_avg :%0.3f, fvn_dice_avg :%0.3f, ocn_dice_avg :%0.3f" %  (
            epoch + 1, n_epochs, loss_avg/step, mean_dice_avg/step, on_dice_avg/step, tgn_dice_avg/step, fvn_dice_avg/step, ocn_dice_avg/step))
        model_path = 'outputs_CN_T1_Peaks/' + 'CN_%d_epoch_%d_batch.pth' % (epoch + 1, batch_size)
        torch.save(model_fusion.state_dict(), model_path)


        print('-' * 30)




if __name__ == '__main__':

    x_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    y_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    # 模型保存
    if 'outputs_CN_T1_Peaks' not in os.listdir(os.curdir):
        os.mkdir('outputs_CN_T1_Peaks')

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




