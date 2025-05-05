import config_2d
from NetModel import CNTSegV2_final
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset_P_mask_test import CN_MyTrainDataset
import os
from torchvision.transforms import transforms
import numpy as np
import time
import nibabel as nib
from metricsPaper import dice
from tqdm import tqdm
from time import sleep

Model = CNTSegV2_final.CNTSegV2_NO_Dedicated_without_ARM

patch_size_w = config_2d.PATCH_SIZE_W
patch_size_h = config_2d.PATCH_SIZE_H
flag_gpu = config_2d.FLAG_GPU

batch_size = config_2d.BATCH_SIZE
n_epochs = config_2d.NUM_EPOCHS
n_classes = config_2d.NUM_CLASSES
image_rows = config_2d.VOLUME_ROWS
image_cols = config_2d.VOLUME_COLS
image_depth = config_2d.VOLUME_DEPS

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_list(weight, pre_file_t1_name, test_data_name, input_label_base, model1,test_imgs_path,fold,mask_idx):  #

    # modal = ['T1', 'FA', 'Peaks']
    # pre_file_t1_name = '/media/brainplan/XLdata/CNTSeg++/Predict/Predict_T1_1/fold1'
    # test_data_name = '/media/brainplan/XLdata/CNTSeg++/Code/CN_mydata/CN_mydata1/test_data'
    # input_label_base = '/media/brainplan/XLdata/CNTSeg++/Data/Data_5-fold/data_fold1/TestSet'
    model1_path = weight_path + '/' + weight
    test_dir = os.listdir(test_imgs_path)
    filename = weight.split('epoch', -1)[0]
    # 预测结果保存
    pre_file_t1 = pre_file_t1_name + '/' + filename  # model1

    if pre_file_t1 not in os.listdir(os.curdir):
        os.mkdir(pre_file_t1)
    for test_num in test_dir:
        test_name = 'test_' + test_num
        test_pre_name = 'test_result_' + test_num

        os.mkdir(os.path.join(pre_file_t1, test_pre_name))

        test_input_path = test_data_name + '/' + test_name + '/'
        test_result_t1 = pre_file_t1 + '/' + test_pre_name + '/'

        ## 1.预测并合成
        # global model1, model2, model3, model4, test_dataset, test_dataloader
        # model1_path = 'outputs_CN_T1/CN_300_epoch_64_batch.pth'

        # 模型选择
        # model1 = unet2d(1, 5).to(device)
        # model1 = nn.DataParallel(model1).cuda()
        CN_test_x_t1_dir = test_input_path + 'x_t1_data/'
        CN_test_x_t2_dir = test_input_path + 'x_t2_data/'
        CN_test_x_fa_dir = test_input_path + 'x_fa_data/'
        CN_test_x_dec_dir = test_input_path + 'x_dec_data/'
        CN_test_x_peaks_dir = test_input_path + 'x_peaks_data/'
        CN_test_y_1_dir = test_input_path + 'y_data_on/'
        CN_test_y_2_dir = test_input_path + 'y_data_tgn/'
        CN_test_y_3_dir = test_input_path + 'y_data_fvn/'
        CN_test_y_4_dir = test_input_path + 'y_data_ocn/'

        model1.load_state_dict(torch.load(model1_path, map_location='cuda'))
        x_transforms = transforms.ToTensor()

        y_transforms = transforms.ToTensor()
        test_dataset = CN_MyTrainDataset(CN_test_x_t1_dir, CN_test_x_t2_dir, CN_test_x_fa_dir, CN_test_x_dec_dir,
                                         CN_test_x_peaks_dir, CN_test_y_1_dir, CN_test_y_2_dir, CN_test_y_3_dir,
                                         CN_test_y_4_dir,  mask_idx, x_transform=x_transforms,
                                         y_transform=y_transforms)

        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)

        all_t1_patch_num = 0

        model1.eval()

        with torch.no_grad():
            for x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, mask in test_dataloader:
                # x1:T1, x2:T2, x3:FA, x4:DEC, x5:Peaks
                inputs1_t1 = x1.to(device)
                inputs2_t2 = x2.to(device)
                inputs3_fa = x3.to(device)
                inputs4_dec = x4.to(device)
                inputs5_peaks = x5.to(device)
                # input = torch.cat([inputs1, inputs2, inputs3], 1)
                outputs1,_ = model1(inputs1_t1,inputs2_t2,inputs3_fa,inputs4_dec,inputs5_peaks,mask)
                outputs1 = torch.sigmoid(outputs1)
                outputs1[outputs1 < 0.5] = 0
                outputs1[outputs1 > 0.5] = 1

                t1_pred_result = outputs1.squeeze().cpu().numpy()

                pre = np.zeros((t1_pred_result.shape[1], t1_pred_result.shape[2]))
                pre[t1_pred_result[0, ...] == 1] = 0
                pre[t1_pred_result[1, ...] == 1] = 1
                pre[t1_pred_result[2, ...] == 1] = 2
                pre[t1_pred_result[3, ...] == 1] = 3
                pre[t1_pred_result[4, ...] == 1] = 4
                t1_pred_result = pre

                if all_t1_patch_num == 0:
                    test_t1_data_path = CN_test_x_t1_dir + 'x_t1-data_0.nii.gz'
                    test_t1_img = nib.load(test_t1_data_path)
                    test_t1_img_affine = test_t1_img.affine
                t1_pred_patches_nii = nib.Nifti1Image(t1_pred_result, test_t1_img_affine)
                t1_pred_nii_path = test_result_t1 + 'pre_' + str(all_t1_patch_num) + '.nii.gz'
                nib.save(t1_pred_patches_nii, t1_pred_nii_path)
                all_t1_patch_num += 1

        #### -------------------------------------------------
        #### -------------------------------------------------
        #### -------------------------------------------------
        ## Combination
        pre_seg_t1_final = np.zeros((image_rows, image_cols, image_depth))

        # source.nii
        img_name = test_num + '_CN-T1.nii.gz'
        img_name = os.path.join(test_imgs_path, test_num, img_name)
        img = nib.load(img_name)
        img_data = img.get_fdata()
        img_affine = img.get_affine()

        img_label1_name = test_num + '_ON-label.nii.gz'
        img_label1_name = os.path.join(test_imgs_path, test_num, img_label1_name)
        img_label1 = nib.load(img_label1_name)
        img_label1_data = img_label1.get_fdata()
        img_label1_data = np.squeeze(img_label1_data)

        img_label2_name = test_num + '_OCN-label.nii.gz'
        img_label2_name = os.path.join(test_imgs_path, test_num, img_label2_name)
        img_label2 = nib.load(img_label2_name)
        img_label2_data = img_label2.get_fdata()
        img_label2_data = np.squeeze(img_label2_data)

        img_label3_name = test_num + '_TGN-label.nii.gz'
        img_label3_name = os.path.join(test_imgs_path, test_num, img_label3_name)
        img_label3 = nib.load(img_label3_name)
        img_label3_data = img_label3.get_fdata()
        img_label3_data = np.squeeze(img_label3_data)

        img_label4_name = test_num + '_FVN-label.nii.gz'
        img_label4_name = os.path.join(test_imgs_path, test_num, img_label4_name)
        img_label4 = nib.load(img_label4_name)
        img_label4_data = img_label4.get_fdata()
        img_label4_data = np.squeeze(img_label4_data)

        img_label_data = img_label1_data + img_label2_data + img_label3_data + img_label4_data



        # mask.nii
        img_mask_name = test_num + '_CN-mask.nii.gz'
        img_mask_name = os.path.join(test_imgs_path, test_num, img_mask_name)
        img_mask = nib.load(img_mask_name)
        img_mask_data = img_mask.get_fdata()
        img_mask_data = np.squeeze(img_mask_data)

        X = img_mask_data.shape

        step = 0

        for iSlice in range(0, X[2]):
            if np.count_nonzero(img_label_data[:, :, iSlice]) > 0 and np.count_nonzero(img_mask_data[:, :, iSlice]) > 0 and np.count_nonzero(img_data[:, :, iSlice]) > 0:
                pre_name = 'pre_' + str(step) + '.nii.gz'

                pre_t1_name = os.path.join(test_result_t1, pre_name)

                pre_seg_t1_temp = nib.load(pre_t1_name)
                pre_seg_t1_temp_data = pre_seg_t1_temp.get_fdata()

                step += 1

                for i in range(0, patch_size_w):
                    for j in range(0, patch_size_h):
                        pre_seg_t1_final[i][j][iSlice] = pre_seg_t1_temp_data[i][j]

        pre_seg_t1_final = nib.Nifti1Image(pre_seg_t1_final, img_affine)

        pre_sge_finalname = 'pre_final-label.nii.gz'

        pre_sge_t1_final_savepath = os.path.join(test_result_t1, pre_sge_finalname)
        nib.save(pre_seg_t1_final, pre_sge_t1_final_savepath)

        # predict(test_input_path, test_result_t1, test_num, '/media/brainplan/XLdata/ON_Seg/CODE/outputs_Single_modal/outputs_ms_T1/100epoch_60batch.pth')

    gtpath = input_label_base
    # pre_path = "/media/brainplan/XLdata/CNTSeg++/Code/predict_D1_T1"
    dice_on, dice_ocn, dice_tgn, dice_fvn = [], [], [], []
    gtpath_num = os.listdir(gtpath)
    for num in gtpath_num:
        # print(num)
        groundtruth_on = gtpath + '/' + num + '/' + num + '_ON-label.nii.gz'
        groundtruth_ocn = gtpath + '/' + num + '/' + num + '_OCN-label.nii.gz'
        groundtruth_tgn = gtpath + '/' + num + '/' + num + '_TGN-label.nii.gz'
        groundtruth_fvn = gtpath + '/' + num + '/' + num + '_FVN-label.nii.gz'
        pre = pre_file_t1 + '/' + 'test_result_' + num + '/' + 'pre_final-label.nii.gz'
        dice_on1, dice_ocn1, dice_tgn1, dice_fvn1 = dice(pre, groundtruth_on, groundtruth_ocn, groundtruth_tgn,
                                                         groundtruth_fvn)

        dice_on.append(dice_on1)
        dice_ocn.append(dice_ocn1)
        dice_tgn.append(dice_tgn1)
        dice_fvn.append(dice_fvn1)
    mean_dice_on = sum(dice_on) / len(dice_on)
    mean_dice_ocn = sum(dice_ocn) / len(dice_ocn)
    mean_dice_tgn = sum(dice_tgn) / len(dice_tgn)
    mean_dice_fvn = sum(dice_fvn) / len(dice_fvn)
    mean_dice_CN = (mean_dice_on + mean_dice_ocn + mean_dice_tgn + mean_dice_fvn) / 4
    print(mean_dice_CN)
    with open(pre_file_t1_name + '/' + fold + '_Dice.txt', 'a+') as f:
        f.writelines('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t\n'.format(filename, mean_dice_on, mean_dice_ocn, mean_dice_tgn,
                                                               mean_dice_fvn, mean_dice_CN))


if __name__ == '__main__':

    x_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    y_transforms = transforms.ToTensor()

    start_time = time.time()
    ##
    mask_idx = 15
    model1 = Model(1,1,1,3,9,5).to(device)
    model1 = nn.DataParallel(model1).cuda()
    folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
    for fold in folds:
        pre_file_t1_name = '/media/brainplan/XLdata/CNTSeg++/Predict/CNTSegV2/CNTSegV2_no_dedicated_without_ARM/' + fold + '/test_val'
        if pre_file_t1_name not in os.listdir(os.curdir):
            os.mkdir(pre_file_t1_name)
        test_data_name = '/media/brainplan/XLdata/CNTSeg++/CodeNew/CN_mydata/' + fold + '/test_val_data'
        input_label_base = '/media/brainplan/XLdata/CNTSeg++/Data/NewDataset/' + fold + '/TestValSet'
        weight_path = '/media/brainplan/XLdata/CNTSeg++/Weights/CNTSegV2/CNTSegV2_no_dedicated_without_ARM/' + fold
        test_imgs_path = '/media/brainplan/XLdata/CNTSeg++/Data/NewDataset/' + fold + '/TestValSet'
        weights = os.listdir(weight_path)
        for weight in tqdm(weights[:], ncols=100):
            weights_list(weight, pre_file_t1_name, test_data_name, input_label_base, model1, test_imgs_path,fold,mask_idx)
            sleep(0.01)
    end_time = time.time()

    print("2D train time is {:.3f} s".format((end_time - start_time)))
