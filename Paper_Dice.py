
import os
import SimpleITK as sitk
import numpy as np



def dice(pre_path,groundtruth_on,groundtruth_ocn,groundtruth_tgn,groundtruth_fvn):

    gt_on = sitk.GetArrayFromImage(sitk.ReadImage(groundtruth_on, sitk.sitkInt16))
    gt_ocn = sitk.GetArrayFromImage(sitk.ReadImage(groundtruth_ocn, sitk.sitkInt16))
    gt_tgn = sitk.GetArrayFromImage(sitk.ReadImage(groundtruth_tgn, sitk.sitkInt16))
    gt_fvn = sitk.GetArrayFromImage(sitk.ReadImage(groundtruth_fvn, sitk.sitkInt16))
    pre = sitk.GetArrayFromImage(sitk.ReadImage(pre_path, sitk.sitkInt16))

    ON = np.zeros(pre.shape)
    OCN = np.zeros(pre.shape)
    TGN = np.zeros(pre.shape)
    FVN = np.zeros(pre.shape)
    # print(img_y_0_zero.shape)
    ON[pre==1]=1
    OCN[pre==4]=1
    TGN[pre==2]=1
    FVN[pre==3]=1

    core = 0.000000000000000001
    #### 计算 dice
    zeros =np.zeros(pre.shape)  # 全0变量
    ones = np.ones(pre.shape)  # 全1变量
    # ON
    tp_on =((gt_on == ones) & (ON == ones)).sum()
    fp_on=((gt_on==zeros) & (ON==ones)).sum()
    tn_on=((gt_on==zeros) & (ON==zeros)).sum()
    fn_on=((gt_on==ones) & (ON==zeros)).sum()
    dice_on = (tp_on*2)/(fp_on+tp_on*2+fn_on)
    JAC_on = tp_on / (fp_on + tp_on + fn_on)
    precision = (tp_on + core) / (tp_on + fp_on + core)

    # ON
    tp_ocn =((gt_ocn == ones) & (OCN == ones)).sum()
    fp_ocn=((gt_ocn==zeros) & (OCN==ones)).sum()
    tn_ocn=((gt_ocn==zeros) & (OCN==zeros)).sum()
    fn_ocn=((gt_ocn==ones) & (OCN==zeros)).sum()
    dice_ocn = (tp_ocn*2)/(fp_ocn+tp_ocn*2+fn_ocn)
    # TGN
    tp_tgn =((gt_tgn == ones) & (TGN == ones)).sum()
    fp_tgn=((gt_tgn==zeros) & (TGN==ones)).sum()
    tn_tgn=((gt_tgn==zeros) & (TGN==zeros)).sum()
    fn_tgn=((gt_tgn==ones) & (TGN==zeros)).sum()
    dice_tgn = (tp_tgn*2)/(fp_tgn+tp_tgn*2+fn_tgn)
    # FVN
    tp_fvn =((gt_fvn == ones) & (FVN == ones)).sum()
    fp_fvn=((gt_fvn==zeros) & (FVN==ones)).sum()
    tn_fvn=((gt_fvn==zeros) & (FVN==zeros)).sum()
    fn_fvn=((gt_fvn==ones) & (FVN==zeros)).sum()
    dice_fvn = (tp_fvn*2)/(fp_fvn+tp_fvn*2+fn_fvn)

    print("{0}\t{1}\t{2}\t{3}\t{4}\n".format(pre_path,dice_on,dice_ocn,dice_tgn,dice_fvn))

    return dice_on, dice_ocn, dice_tgn, dice_fvn
if __name__=="__main__":
    dice_on, dice_ocn, dice_tgn, dice_fvn=[],[],[],[]
    gtpath = '/media/brainplan/XLdata/CNTSeg++/Data/NewDataset/TestValSet'
    pre_path = "/media/brainplan/XLdata/CNTSeg++/Predict/CNTSegV2/CNTSegV2_BaselineSE_SM_FIMC_5/test2"
    gtpath_num = os.listdir(gtpath)
    for num in gtpath_num:
        # print(num)
        groundtruth_on = gtpath + '/' + num + '/' + num + '_ON-label.nii.gz'
        groundtruth_ocn = gtpath + '/' + num + '/' + num + '_OCN-label.nii.gz'
        groundtruth_tgn = gtpath + '/' + num + '/' + num + '_TGN-label.nii.gz'
        groundtruth_fvn = gtpath + '/' + num + '/' + num + '_FVN-label.nii.gz'
        pre = pre_path + '/' + 'test_result_' + num + '/'+'pre_final-label.nii.gz'
        dice_on1, dice_ocn1, dice_tgn1, dice_fvn1 = dice(pre, groundtruth_on, groundtruth_ocn, groundtruth_tgn, groundtruth_fvn)

        dice_on.append(dice_on1)
        dice_ocn.append(dice_ocn1)
        dice_tgn.append(dice_tgn1)
        dice_fvn.append(dice_fvn1)
    mean_dice_on= sum(dice_on) / len(dice_on)
    mean_dice_ocn = sum(dice_ocn) / len(dice_ocn)
    mean_dice_tgn = sum(dice_tgn) / len(dice_tgn)
    mean_dice_fvn = sum(dice_fvn) / len(dice_fvn)
    mean_dice_CN = (mean_dice_on + mean_dice_ocn + mean_dice_tgn + mean_dice_fvn) / 4
    print(mean_dice_on, mean_dice_ocn, mean_dice_tgn, mean_dice_fvn, mean_dice_CN)
    # print(mean_dice_on,mean_dice_ocn,mean_dice_tgn,mean_dice_fvn)

