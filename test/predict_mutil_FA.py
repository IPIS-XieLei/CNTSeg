import config_2d
from NetModel import Unet_2d, Unet_plus_2d, MultiResUnet_2d, MultiResUnet_plus_2d,Newnet,se2dunet
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset_P import CN_MyTrainDataset
import os
from torchvision.transforms import transforms
import numpy as np
import time
import nibabel as nib
from skimage import measure
import torch.nn.functional as F
# unet2dse = se2dunet.UNet2Dse
unet2d = Unet_2d.UNet2D                             # U-Net
unetplus2d = Unet_plus_2d.UNetPlus2D                # U-Net++
multiresunet2d = MultiResUnet_2d.MultiResUnet2D     # MultiRes U-Net
ournet2d = MultiResUnet_plus_2d.MultiResUnetPlus2D  # MultiRes U-Net++
fusionnet=Newnet.DatafusionNet

patch_size_w = config_2d.PATCH_SIZE_W
patch_size_h = config_2d.PATCH_SIZE_H
flag_gpu = config_2d.FLAG_GPU

batch_size = config_2d.BATCH_SIZE
n_epochs = config_2d.NUM_EPOCHS
n_classes = config_2d.NUM_CLASSES
image_rows = config_2d.VOLUME_ROWS
image_cols = config_2d.VOLUME_COLS
image_depth = config_2d.VOLUME_DEPS
test_imgs_path = config_2d.test_imgs_path
test_extraction_step = config_2d.TEST_EXTRACTION_STEP
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 通过模型预测结果
def predict(img_dir, predict_t1, imgs_num):
    global model1, model2, model3, model4, test_dataset, test_dataloader
    model1_path = 'outputs_CN_FA/CN_500_epoch_64_batch.pth'



    # 模型选择
    model1 = unet2d(1, 5).to(device)
    model1 = nn.DataParallel(model1).cuda()

    CN_test_x_t1_dir = img_dir + 'x_t1_data/'
    CN_test_x_fa_dir = img_dir + 'x_fa_data/'
    CN_test_x_peaks_dir = img_dir + 'x_peaks_data/'
    CN_test_y_1_dir = img_dir + 'y_data_on/'
    CN_test_y_2_dir = img_dir + 'y_data_tgn/'
    CN_test_y_3_dir = img_dir + 'y_data_fvn/'
    CN_test_y_4_dir = img_dir + 'y_data_ocn/'

    model1.load_state_dict(torch.load(model1_path, map_location='cuda'))
    x_transforms = transforms.ToTensor()

    y_transforms = transforms.ToTensor()
    test_dataset = CN_MyTrainDataset(CN_test_x_t1_dir, CN_test_x_fa_dir,CN_test_x_peaks_dir, CN_test_y_1_dir, CN_test_y_2_dir, CN_test_y_3_dir, CN_test_y_4_dir, x_transform=x_transforms,
                                  y_transform=y_transforms)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    all_t1_patch_num = 0

    model1.eval()

    with torch.no_grad():
        for x1, x2, x3, y0, y1, y2, y3, y4 in test_dataloader:
            # inputs1 = x1.to(device)
            inputs2 = x2.to(device)
            # input = torch.cat([inputs1, inputs2], 1)
            outputs1 = model1(inputs2)
            outputs1 = torch.sigmoid(outputs1)
            outputs1[outputs1 < 0.5] = 0
            outputs1[outputs1 > 0.5] = 1

            t1_pred_result = outputs1.squeeze().cpu().numpy()
            # t1_pred_result = np.argmax(t1_pred_result,axis=0)
            # t1_pred_result = t1_pred_result + 1
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
                test_t1_img_affine = test_t1_img.get_affine()
            t1_pred_patches_nii = nib.Nifti1Image(t1_pred_result, test_t1_img_affine)
            t1_pred_nii_path = predict_t1 + 'pre_' + str(all_t1_patch_num) + '.nii.gz'
            nib.save(t1_pred_patches_nii, t1_pred_nii_path)
            all_t1_patch_num += 1

    #### -------------------------------------------------
    #### -------------------------------------------------
    #### -------------------------------------------------
    ## Combination
    pre_seg_t1_final = np.zeros((image_rows, image_cols, image_depth))

    # source.nii
    img_name = imgs_num + '_CN-T1.nii.gz'

    img_name = os.path.join(test_imgs_path, imgs_num, img_name)
    img = nib.load(img_name)
    img_data = img.get_data()
    img_affine = img.get_affine()

    # mask.nii
    img_mask_name = imgs_num + '_CN-mask.nii.gz'
    img_mask_name = os.path.join(test_imgs_path, imgs_num, img_mask_name)
    img_mask = nib.load(img_mask_name)
    img_mask_data = img_mask.get_data()
    img_mask_data = np.squeeze(img_mask_data)

    X = img_mask_data.shape

    step = 0

    for iSlice in range(0, X[2]):
        if np.count_nonzero(img_mask_data[:, :, iSlice]) and np.count_nonzero(img_data[:, :, iSlice]):
            pre_name = 'pre_' + str(step) + '.nii.gz'

            pre_t1_name = os.path.join(predict_t1, pre_name)

            pre_seg_t1_temp = nib.load(pre_t1_name)
            pre_seg_t1_temp_data = pre_seg_t1_temp.get_data()

            step += 1

            for i in range(0, patch_size_w):
                for j in range(0, patch_size_h):
                    pre_seg_t1_final[i][j][iSlice] = pre_seg_t1_temp_data[i][j]

    pre_seg_t1_final = nib.Nifti1Image(pre_seg_t1_final, img_affine)

    pre_sge_finalname = 'pre_final-label.nii.gz'

    pre_sge_t1_final_savepath = os.path.join(predict_t1, pre_sge_finalname)
    nib.save(pre_seg_t1_final, pre_sge_t1_final_savepath)


## For rough segmentation
if __name__ == '__main__':

    x_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # x_peaks_transforms = transforms.ToTensor()
    y_transforms = transforms.ToTensor()

    # 预测结果保存
    pre_file_t1 = 'predict'  # model1

    if pre_file_t1 not in os.listdir(os.curdir):
        os.mkdir(pre_file_t1)

    start_time = time.time()

    test_dir = os.listdir(test_imgs_path)

    for test_num in test_dir:
        test_name = 'test_' + test_num
        test_pre_name = 'test_result_' + test_num

        os.mkdir(os.path.join(pre_file_t1, test_pre_name))

        test_input_path = 'CN_mydata/test_T1_FA/' + test_name + '/'
        test_result_t1 = pre_file_t1 + '/' + test_pre_name + '/'

        ## 1.预测并合成
        predict(test_input_path, test_result_t1, test_num)

    end_time = time.time()
    print("2D train time is {:.3f} s".format((end_time - start_time)))