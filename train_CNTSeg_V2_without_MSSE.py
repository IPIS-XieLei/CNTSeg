#import visdom
import config_2d
from NetModel import CNTSegV2_final
import torch, time
import math
from einops import rearrange, repeat
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset_P_mask import CN_MyTrainDataset
from torchvision.transforms import transforms
# from metrics_2d import dice_loss, dice, dice_l2_loss
import os
import random
import numpy as np
from mutilloss import SoftDiceLoss,diceCoeffv2,diceCoeff,SoftDiceLoss1,SDFLoss,compute_sdf
n_classes = config_2d.NUM_CLASSES
# unet2d = Unet_2d.UNet2D                             # U-Net
Model = CNTSegV2_final.CNTSegV2_NO_Dedicated
batch_size = 32
n_epochs = config_2d.NUM_EPOCHS


flag_gpu = config_2d.FLAG_GPU
# os.environ["CUDA_VISIBLE_DE VICES"] = "0"

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

masks = np.array([[False, False, False, False],[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True],
                      [True, True, False, False], [True, False, True, False], [True, False, False, True], [False, True, True, False], [False, True, False, True], [False, False, True, True], [True, True, True, False], [True, True, False, True], [True, False, True, True], [False, True, True, True],
                      [True, True, True, True]])


# train
def train(weights_out_path, fold):
    global model, losses, train_dataset, train_dataloader, val_dataset, val_dataloader

    # op_lr = 0.002 #

    op_lr = 0.002  #
    # 训练集选择

    CN_train_x_t1_dir = 'CN_mydata/' + fold + '/train_data/x_t1_data/'
    CN_train_x_t2_dir = 'CN_mydata/' + fold + '/train_data/x_t2_data/'
    CN_train_x_fa_dir = 'CN_mydata/' + fold + '/train_data/x_fa_data/'
    CN_train_x_dec_dir = 'CN_mydata/' + fold + '/train_data/x_dec_data/'
    CN_train_x_peaks_dir = 'CN_mydata/' + fold + '/train_data/x_peaks_data/'
    CN_train_y_on_dir = 'CN_mydata/' + fold + '/train_data/y_data_on/'
    CN_train_y_tgn_dir = 'CN_mydata/' + fold + '/train_data/y_data_tgn/'
    CN_train_y_fvn_dir = 'CN_mydata/' + fold + '/train_data/y_data_fvn/'
    CN_train_y_ocn_dir = 'CN_mydata/' + fold + '/train_data/y_data_ocn/'


    # 损失函数选择

    losses_1= SoftDiceLoss1(4,activation='sigmoid').cuda()
    losses_2= nn.BCEWithLogitsLoss().cuda()
    losses_3 = nn.SmoothL1Loss().cuda()
    # losses_3 = SDFLoss().cuda()
    # losses_3 = nn.L1Loss().cuda()
    model = Model(1,1,1,3,9, 5).to(device)

    model = nn.DataParallel(model).cuda()

    x_transforms = transforms.ToTensor()

    y_transforms = transforms.ToTensor()
    train_dataset = CN_MyTrainDataset(CN_train_x_t1_dir, CN_train_x_t2_dir, CN_train_x_fa_dir, CN_train_x_dec_dir,
                                      CN_train_x_peaks_dir,
                                      CN_train_y_on_dir, CN_train_y_tgn_dir, CN_train_y_fvn_dir, CN_train_y_ocn_dir,
                                      x_transform=x_transforms, y_transform=y_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # 模型载入
    # model.load_state_dict(torch.load('/media/brainplan/XLdata/CNTSeg++/Weights/CNTSegV2/CNTSegV2_no_dedicated/CN_122_epoch_32_batch.pth', map_location='cuda'))

###----------------------
#### start train
    print('-' * 30)
    print('Training start...')
    print('-' * 30)
    print('batch size   : ', batch_size)
    print('  epoch      : ', n_epochs)
    print('learning rate peak: ', op_lr)
    print('-' * 30)

    optimizer = optim.Adam(model.parameters(), lr=op_lr)
    # t = 10  # warmup
    # T = 200  # 共有200个epoch，则用于cosine rate的一共有180个epoch
    # n_t = 0.5
    # lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
    #         1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
    #         1 + math.cos(math.pi * (epoch - t) / (T - t)))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    for epoch in range(0, 200):

        dt_size = len(train_dataloader.dataset)
        # epoch_loss_t1fa = 0
        step = 0
        loss_avg, mean_dice_avg, on_dice_avg, tgn_dice_avg, fvn_dice_avg, ocn_dice_avg  = 0, 0, 0, 0, 0, 0
        loss_avg_island = 0
        model.train()
        for x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, mask in train_dataloader:
            # x1:T1, x2:T2, x3:FA, x4:DEC, x5:Peaks
            step += 1
            inputs1_t1 = x1.to(device)
            inputs2_t2 = x2.to(device)
            inputs3_fa = x3.to(device)
            inputs4_dec = x4.to(device)
            inputs5_peaks = x5.to(device)
            groundtruth0 = y0.to(device)
            groundtruth1 = y1.to(device)  # on
            groundtruth2 = y2.to(device)  # ocn
            groundtruth3 = y3.to(device)  # tgn
            groundtruth4 = y4.to(device)  # fvn
            modality_mask = mask.to(device)  # fvn

            groundtruth = torch.cat([groundtruth0, groundtruth1, groundtruth2, groundtruth3, groundtruth4], 1)
            # 梯度清零
            optimizer.zero_grad()
            outputs,pre_dis= model(inputs1_t1,inputs2_t2,inputs3_fa,inputs4_dec,inputs5_peaks,modality_mask)
            #####################################################
            # binary_gt = groundtruth1+groundtruth2+groundtruth3+groundtruth4
            # binary_gt[binary_gt> 0] = 1
            # with torch.no_grad():
            #     dis = torch.from_numpy(
            #         compute_sdf(binary_gt.cpu().numpy(), pre_dis.shape)).float().cuda()
            # losses_dis = losses_3(pre_dis, dis)
            ########################################################
            losses_ce = losses_2(outputs, groundtruth)
            loss_dl = losses_1(outputs, groundtruth)

            # losses_dis = losses_3(pre_dis, binary_gt)

            loss_t1fa = 0.5 * loss_dl + 0.5 * losses_ce
            # loss_t1fa = 0.5*loss_dl + 0.5*losses_ce

            output = torch.sigmoid(outputs)
            output[output < 0.5] = 0
            output[output > 0.5] = 1
            on_dice = diceCoeff(output[:, 1, :,:], groundtruth[:, 1, :,:], activation=None).cpu().item()
            tgn_dice = diceCoeff(output[:, 2, :,:], groundtruth[:, 2, :,:], activation=None).cpu().item()
            fvn_dice = diceCoeff(output[:, 3, :,:], groundtruth[:, 3, :,:], activation=None).cpu().item()
            ocn_dice = diceCoeff(output[:, 4, :,:], groundtruth[:, 4, :,:], activation=None).cpu().item()
            mean_dice = (on_dice + tgn_dice+ fvn_dice+ ocn_dice) / 4

            loss_avg += loss_t1fa
            # loss_avg_island += losses_rc
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
        # scheduler.step()
        print("epoch: %d/%d done, loss_avg :%0.3f, mean_dice_avg :%0.3f, on_dice_avg :%0.3f, tgn_dice_avg :%0.3f, fvn_dice_avg :%0.3f, ocn_dice_avg :%0.3f" %  (
            epoch + 1, n_epochs, loss_avg/step, mean_dice_avg/step, on_dice_avg/step, tgn_dice_avg/step, fvn_dice_avg/step, ocn_dice_avg/step))
        # with open(weights_out_path + '/' + 'train_' + str(batch_size) + 'epoch_loss.txt', 'a+') as f:
        #     f.writelines('epoch{0}\t{1}\t{2} \n'.format(str(epoch), loss_avg/step,loss_avg_island/step))
        model_path = weights_out_path + '/' + 'CN_%d_epoch_%d_batch.pth' % (epoch + 1, batch_size)
        torch.save(model.state_dict(), model_path)

        print('-' * 30)




if __name__ == '__main__':

    RANDOM_SEED = 3407  # any random number


    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # CPU
        torch.cuda.manual_seed(seed)  # GPU
        torch.cuda.manual_seed_all(seed)  # All GPU
        os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
        torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
        torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


    set_seed(RANDOM_SEED)

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

    start_time = time.time()
    ### train test ###
    folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
    for fold in folds:
        print('/media/brainplan/XLdata/CNTSeg++/Weights/CNTSegV2/CNTSegV2_no_dedicated_without_SDM/' + fold)
        weights_out_path = '/media/brainplan/XLdata/CNTSeg++/Weights/CNTSegV2/CNTSegV2_no_dedicated_without_SDM/' + fold
        if weights_out_path not in os.listdir(os.curdir):
            os.mkdir(weights_out_path)

        train(weights_out_path, fold)

    end_time = time.time()
    print("2D train time is {:.3f} mins".format((end_time - start_time) / 60.0))
    print('-' * 30)
    print('batch size   : ', batch_size)
    print('  epoch      : ', n_epochs)
    print('-' * 30)
    print("done")




