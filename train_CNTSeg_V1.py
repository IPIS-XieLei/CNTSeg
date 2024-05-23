#import visdom
import config_2d
from NetModel import CNTSeg_V1_model
import torch, time
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset_P import CN_MyTrainDataset
from torchvision.transforms import transforms
import os
import random
import numpy as np
from mutilloss import SoftDiceLoss,diceCoeffv2,diceCoeff,SoftDiceLoss1

cntsegunet = CNTSeg_V1_model.UNet2D
cntsegfusionnet = CNTSeg_V1_model.DatafusionNet_3

patch_size_w = config_2d.PATCH_SIZE_W
patch_size_h = config_2d.PATCH_SIZE_H
batch_size = 32   # 30
n_epochs = config_2d.NUM_EPOCHS
n_classes = config_2d.NUM_CLASSES
image_rows = config_2d.VOLUME_ROWS
image_cols = config_2d.VOLUME_COLS
image_depth = config_2d.VOLUME_DEPS
test_imgs_path = config_2d.test_imgs_path

flag_gpu = config_2d.FLAG_GPU
# flag_model = config_2d.MODEL_name

# 是否使用cuda
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# train
def train(weights_out_path,fold,weights_T1,weights_T2,weights_FA,weights_Peaks,weights_DEC):
    global model, losses, train_dataset, train_dataloader, val_dataset, val_dataloader

    # op_lr = 0.002 #

    op_lr = 0.002  #
    # 训练集选择
    #For model 4 (CTL_t1+fa)
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
    # losses_3 = BCE_Dice_Loss(4).cuda()
    # model = segmentation_fusionnet(1, 1, 9, 15, 5).to(device)
    model_t1 = cntsegunet(1, 5).to(device)
    # model_t2 = cntsegunet(1, 5).to(device)
    model_fa = cntsegunet(1, 5).to(device)
    # model_dec = cntsegunet(3, 5).to(device)
    model_peak = cntsegunet(9, 5).to(device)
    model_fusion = cntsegfusionnet(15, 5).to(device)

    model_t1 = nn.DataParallel(model_t1).cuda()
    # model_t2 = nn.DataParallel(model_t2).cuda()
    model_fa = nn.DataParallel(model_fa).cuda()
    # model_dec = nn.DataParallel(model_dec).cuda()
    model_peak = nn.DataParallel(model_peak).cuda()
    model_fusion = nn.DataParallel(model_fusion).cuda()
    # if flag_gpu == 1:
    #     model = nn.DataParallel(model).cuda()

    x_transforms = transforms.ToTensor()

    y_transforms = transforms.ToTensor()
    train_dataset = CN_MyTrainDataset(CN_train_x_t1_dir, CN_train_x_t2_dir, CN_train_x_fa_dir, CN_train_x_dec_dir,
                                      CN_train_x_peaks_dir,
                                      CN_train_y_on_dir, CN_train_y_tgn_dir, CN_train_y_fvn_dir, CN_train_y_ocn_dir,
                                      x_transform=x_transforms, y_transform=y_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    print("/media/brainplan/XLdata/CNTSeg++/Weights/Single/" + fold + "/outputs_CN_T1/CN_"+weights_T1+"_epoch_64_batch.pth")
    print(
        "/media/brainplan/XLdata/CNTSeg++/Weights/Single/" + fold + "/outputs_CN_T2/CN_" + weights_T2 + "_epoch_64_batch.pth")
    print(
        "/media/brainplan/XLdata/CNTSeg++/Weights/Single/" + fold + "/outputs_CN_FA/CN_"+weights_FA+"_epoch_64_batch.pth")
    print(
        "/media/brainplan/XLdata/CNTSeg++/Weights/Single/" + fold + "/outputs_CN_Peaks/CN_"+weights_Peaks+"_epoch_64_batch.pth")
    print(
        "/media/brainplan/XLdata/CNTSeg++/Weights/Single/" + fold + "/outputs_CN_DEC/CN_" + weights_DEC + "_epoch_64_batch.pth")
    # 模型载入
    model_t1.load_state_dict(
        torch.load("/media/brainplan/XLdata/CNTSeg++/Weights/Single/" + fold + "/outputs_CN_T1/CN_"+weights_T1+"_epoch_64_batch.pth",
                   map_location='cuda'))
    # model_t2.load_state_dict(
    #     torch.load("/media/brainplan/XLdata/CNTSeg++/Weights/Single/" + fold + "/outputs_CN_T2/CN_"+weights_T2+"_epoch_64_batch.pth",
    #                map_location='cuda'))
    model_fa.load_state_dict(
        torch.load("/media/brainplan/XLdata/CNTSeg++/Weights/Single/" + fold + "/outputs_CN_FA/CN_"+weights_FA+"_epoch_64_batch.pth",
                   map_location='cuda'))
    # model_dec.load_state_dict(
    #     torch.load("/media/brainplan/XLdata/CNTSeg++/Weights/Single/" + fold + "/outputs_CN_DEC/CN_"+weights_DEC+"_epoch_64_batch.pth",
    #                map_location='cuda'))
    model_peak.load_state_dict(
        torch.load("/media/brainplan/XLdata/CNTSeg++/Weights/Single/" + fold + "/outputs_CN_Peaks/CN_"+weights_Peaks+"_epoch_64_batch.pth",
                   map_location='cuda'))
    # model_fusion.load_state_dict(torch.load("/media/brainplan/XLdata/CNTSeg++/Weights/outputs_CN_CNTSeg/CN_50_epoch_30_batch.pth", map_location='cuda'))

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
    # t = 10  # warmup
    # T = 200  # 共有200个epoch，则用于cosine rate的一共有180个epoch
    # n_t = 0.5
    # lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
    #         1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
    #         1 + math.cos(math.pi * (epoch - t) / (T - t)))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    for epoch in range(0, 50):

        dt_size = len(train_dataloader.dataset)
        # epoch_loss_t1fa = 0
        step = 0
        loss_avg, mean_dice_avg, on_dice_avg, tgn_dice_avg, fvn_dice_avg, ocn_dice_avg  = 0, 0, 0, 0, 0, 0
        model_t1.eval()
        # model_t2.eval()
        model_fa.eval()
        # model_dec.eval()
        model_peak.eval()
        model_fusion.train()
        for x1, x2, x3, x4, x5, y0, y1, y2, y3, y4 in train_dataloader:
            # o_dice, t_dice, f_dice, oc_dice = 0, 0, 0, 0
            step += 1
            inputs1_t1 = x1.to(device)
            # inputs2_t2 = x2.to(device)
            inputs3_fa = x3.to(device)
            # inputs4_dec = x4.to(device)
            inputs5_peaks = x5.to(device)
            groundtruth0 = y0.to(device)
            groundtruth1 = y1.to(device)  # on
            groundtruth2 = y2.to(device)  # ocn
            groundtruth3 = y3.to(device)  # tgn
            groundtruth4 = y4.to(device)  # fvn

            groundtruth = torch.cat([groundtruth0, groundtruth1, groundtruth2, groundtruth3, groundtruth4], 1)
            # 梯度清零
            optimizer.zero_grad()

            # input = torch.cat([inputs1_t1, inputs2_fa], 1)
            outputs_t1 = model_t1(inputs1_t1)  # FA
            # outputs_t2 = model_t2(inputs2_t2)  # FA
            outputs_fa = model_fa(inputs3_fa)  # FA
            # outputs_dec = model_dec(inputs4_dec)  #
            outputs_peak = model_peak(inputs5_peaks)  # FA

            outputs = model_fusion(outputs_t1,outputs_fa,outputs_peak)  # FA

            # predict = outputs[:, 4, :, :].squeeze()  # label预测值      tensor[batch_size, 1, 144, 144]

            # y_truth = groundtruth.squeeze()  # label真实值      tensor[batch_size, 1, 144, 144]
            losses_ce = losses_2(outputs, groundtruth)
            # loss_t1fa = losses_3(outputs, groundtruth)
            loss_dl = losses_1(outputs, groundtruth)
            loss_t1fa=0.5*loss_dl+0.5*losses_ce
            # label_dice_t1fa = dice(outputs, groundtruth)
            output = torch.sigmoid(outputs)
            output[output < 0.5] = 0
            output[output > 0.5] = 1
            on_dice = diceCoeff(output[:, 1, :,:], groundtruth[:, 1, :,:], activation=None).cpu().item()
            tgn_dice = diceCoeff(output[:, 2, :,:], groundtruth[:, 2, :,:], activation=None).cpu().item()
            fvn_dice = diceCoeff(output[:, 3, :,:], groundtruth[:, 3, :,:], activation=None).cpu().item()
            ocn_dice = diceCoeff(output[:, 4, :,:], groundtruth[:, 4, :,:], activation=None).cpu().item()
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
        # scheduler.step()
        print("epoch: %d/%d done, loss_avg :%0.3f, mean_dice_avg :%0.3f, on_dice_avg :%0.3f, tgn_dice_avg :%0.3f, fvn_dice_avg :%0.3f, ocn_dice_avg :%0.3f" %  (
            epoch + 1, n_epochs, loss_avg/step, mean_dice_avg/step, on_dice_avg/step, tgn_dice_avg/step, fvn_dice_avg/step, ocn_dice_avg/step))
        # with open(weights_out_path + '/' + 'T1_f1_train_' + str(batch_size) + 'epoch_loss.txt', 'a+') as f:
        #     f.writelines('epoch{0}\t{1}\t{2} \n'.format(str(epoch), loss_avg/step,mean_dice_avg/step))
        model_path = weights_out_path + '/' + 'CN_%d_epoch_%d_batch.pth' % (epoch + 1, batch_size)
        torch.save(model_fusion.state_dict(), model_path)

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
    folds = ['fold1','fold2', 'fold3', 'fold4', 'fold5']
    weights_T1 = ['86','88', '93', '88', '59']
    weights_T2 = ['63','66', '101', '105', '81']
    weights_FA = ['84','72', '133', '141', '132']
    weights_DEC = ['73','99', '90', '62', '66']
    weights_Peaks = ['68','118', '105', '70', '97']
    i = 0
    for fold in folds:
        i = i+1
        print('/media/brainplan/XLdata/CNTSeg++/Weights/T1_FA_Peaks/CNTSeg/' + fold)
        weights_out_path = '/media/brainplan/XLdata/CNTSeg++/Weights/T1_FA_Peaks/CNTSeg/' + fold
        if weights_out_path not in os.listdir(os.curdir):
            os.mkdir(weights_out_path)

        train(weights_out_path, fold,weights_T1[i-1],weights_T2[i-1],weights_FA[i-1],weights_Peaks[i-1],weights_DEC[i-1])
    end_time = time.time()
    print("2D train time is {:.3f} mins".format((end_time - start_time) / 60.0))
    print('-' * 30)
    print('patch size   : ', patch_size_w,'x',patch_size_h)
    print('batch size   : ', batch_size)
    print('  epoch      : ', n_epochs)
    print('-' * 30)
    print("done")



