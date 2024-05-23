import os
import SimpleITK as sitk
import numpy as np



def dice(pre_path, groundtruth_on, groundtruth_ocn, groundtruth_tgn, groundtruth_fvn):
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
    ON[pre == 1] = 1
    OCN[pre == 4] = 1
    TGN[pre == 2] = 1
    FVN[pre == 3] = 1

    quality_on = dict()
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    if ((gt_on.sum() > 0) and (ON.sum() > 0)):
        hausdorffcomputer.Execute(sitk.GetImageFromArray(gt_on), sitk.GetImageFromArray(ON.astype(np.int16)))  # (labelTrue > 0.5, labelPred > 0.5)
        quality_on["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
        quality_on["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()
    else:
        quality_on["avgHausdorff"] = "max"
        quality_on["Hausdorff"] = "max"
    # print(quality_on["avgHausdorff"])

    quality_ocn = dict()
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    if ((gt_ocn.sum() > 0) and (OCN.astype('int16').sum() > 0)):
        hausdorffcomputer.Execute(sitk.GetImageFromArray(gt_ocn),
                                  sitk.GetImageFromArray(OCN.astype('int16')))  # (labelTrue > 0.5, labelPred > 0.5)
        quality_ocn["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
        quality_ocn["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()
    else:
        quality_ocn["avgHausdorff"] = "max"
        quality_ocn["Hausdorff"] = "max"
    # print(quality_ocn["avgHausdorff"])

    quality_tgn = dict()
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    if ((gt_tgn.sum() > 0) and (TGN.sum() > 0)):
        hausdorffcomputer.Execute(sitk.GetImageFromArray(gt_tgn),
                                  sitk.GetImageFromArray(TGN.astype('int16')))  # (labelTrue > 0.5, labelPred > 0.5)
        quality_tgn["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
        quality_tgn["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()
    else:
        quality_tgn["avgHausdorff"] = "max"
        quality_tgn["Hausdorff"] = "max"
    # print(quality_tgn["avgHausdorff"])

    quality_fvn = dict()
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    if ((gt_fvn.sum() > 0) and (FVN.sum() > 0)):
        hausdorffcomputer.Execute(sitk.GetImageFromArray(gt_fvn),
                                  sitk.GetImageFromArray(FVN.astype('int16')))  # (labelTrue > 0.5, labelPred > 0.5)
        quality_fvn["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
        quality_fvn["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()
    else:
        quality_fvn["avgHausdorff"] = "max"
        quality_fvn["Hausdorff"] = "max"
    # print(quality_fvn["avgHausdorff"])

    core = 0.000000000000000001
    #### 计算 dice
    zeros = np.zeros(pre.shape)  # 全0变量
    ones = np.ones(pre.shape)  # 全1变量
    # ON
    tp_on = ((gt_on == ones) & (ON == ones)).sum()
    fp_on = ((gt_on == zeros) & (ON == ones)).sum()
    tn_on = ((gt_on == zeros) & (ON == zeros)).sum()
    fn_on = ((gt_on == ones) & (ON == zeros)).sum()

    dice_on = (tp_on * 2) / (fp_on + tp_on * 2 + fn_on)
    jac_on = tp_on / (fp_on + tp_on + fn_on)
    precision_on = (tp_on + core) / (tp_on + fp_on + core)

    # ON
    tp_ocn = ((gt_ocn == ones) & (OCN == ones)).sum()
    fp_ocn = ((gt_ocn == zeros) & (OCN == ones)).sum()
    tn_ocn = ((gt_ocn == zeros) & (OCN == zeros)).sum()
    fn_ocn = ((gt_ocn == ones) & (OCN == zeros)).sum()

    dice_ocn = (tp_ocn * 2) / (fp_ocn + tp_ocn * 2 + fn_ocn)
    jac_ocn = tp_ocn / (fp_ocn + tp_ocn + fn_ocn)
    precision_ocn = (tp_ocn + core) / (tp_ocn + fp_ocn + core)
    # TGN
    tp_tgn = ((gt_tgn == ones) & (TGN == ones)).sum()
    fp_tgn = ((gt_tgn == zeros) & (TGN == ones)).sum()
    tn_tgn = ((gt_tgn == zeros) & (TGN == zeros)).sum()
    fn_tgn = ((gt_tgn == ones) & (TGN == zeros)).sum()

    dice_tgn = (tp_tgn * 2) / (fp_tgn + tp_tgn * 2 + fn_tgn)
    jac_tgn = tp_tgn / (fp_tgn + tp_tgn + fn_tgn)
    precision_tgn = (tp_tgn + core) / (tp_tgn + fp_tgn + core)
    # FVN
    tp_fvn = ((gt_fvn == ones) & (FVN == ones)).sum()
    fp_fvn = ((gt_fvn == zeros) & (FVN == ones)).sum()
    tn_fvn = ((gt_fvn == zeros) & (FVN == zeros)).sum()
    fn_fvn = ((gt_fvn == ones) & (FVN == zeros)).sum()
    dice_fvn = (tp_fvn * 2) / (fp_fvn + tp_fvn * 2 + fn_fvn)
    jac_fvn = tp_fvn / (fp_fvn + tp_fvn + fn_fvn)
    precision_fvn = (tp_fvn + core) / (tp_fvn + fp_fvn + core)
    # print("{0}\t{1}\t{2}\t{3}\t{4}\n".format(pre_path,dice_on,dice_ocn,dice_tgn,dice_fvn))

    return dice_on, dice_ocn, dice_tgn, dice_fvn, jac_on, jac_ocn, jac_tgn, jac_fvn, precision_on, precision_ocn, precision_tgn, precision_fvn,quality_on["avgHausdorff"],quality_ocn["avgHausdorff"],quality_tgn["avgHausdorff"],quality_fvn["avgHausdorff"]


if __name__ == "__main__":
    # for nums in range(0, 16):
    for nums in range(0,16):
        print(nums)
        dice_on, dice_ocn, dice_tgn, dice_fvn = [], [], [], []
        jac_on, jac_ocn, jac_tgn, jac_fvn = [], [], [], []
        precision_on, precision_ocn, precision_tgn, precision_fvn = [], [], [], []
        asd_on, asd_ocn, asd_tgn, asd_fvn = [], [], [], []
        gtpath = '/media/brainplan/XLdata/CNTSeg++/Data/NewDataset/ZJU/TestSet'
        # pre_path = "/media/brainplan/XLdata/CNTSeg++/Predict/CNTSegV2/CNTSegV2_no_dedicated_final_e180/missingmodility/" + str(nums)  #3 fold_all_best
        pre_path = "/media/brainplan/XLdata/CNTSeg++/Predict/T1_FA_Peaks/CNTSeg/ZJU/fold5/test_val/CN_6_"
        gtpath_num = os.listdir(gtpath)
        for num in gtpath_num:
            # print(num)
            groundtruth_on = gtpath + '/' + num + '/' + num + '_ON-label.nii.gz'
            groundtruth_ocn = gtpath + '/' + num + '/' + num + '_OCN-label.nii.gz'
            groundtruth_tgn = gtpath + '/' + num + '/' + num + '_TGN-label.nii.gz'
            groundtruth_fvn = gtpath + '/' + num + '/' + num + '_FVN-label.nii.gz'
            pre = pre_path + '/' + 'test_result_' + num + '/' + 'pre_final-label.nii.gz'
            dice_on1, dice_ocn1, dice_tgn1, dice_fvn1, jac_on1, jac_ocn1, jac_tgn1, jac_fvn1, precision_on1, precision_ocn1, precision_tgn1, precision_fvn1,asd_on1, asd_ocn1, asd_tgn1, asd_fvn1 = dice(
                pre, groundtruth_on, groundtruth_ocn, groundtruth_tgn, groundtruth_fvn)
            # with open("/media/brainplan/XLdata/CNTSeg++/Predict/Results_EXCEL/CNTSegV2_no_dedicated_final_missingmodality-" + str(nums) +".txt",
            #           'a+') as f:
            #     f.writelines("{0}\t{1}\t{2}\t{3}\t{4}\t\t{5}\t{6}\t{7}\t{8}\t\t{9}\t{10}\t{11}\t{12}\t\t{13}\t{14}\t{15}\t{16}\t\n".format(num, dice_on1, dice_ocn1, dice_tgn1, dice_fvn1, jac_on1, jac_ocn1, jac_tgn1, jac_fvn1, precision_on1, precision_ocn1, precision_tgn1, precision_fvn1,asd_on1, asd_ocn1, asd_tgn1, asd_fvn1))
            with open("/media/brainplan/XLdata/CNTSeg++/Predict/Results_EXCEL/ZJU_V1.txt",
                      'a+') as f:
                f.writelines(
                    "{0}\t{1}\t{2}\t{3}\t{4}\t\t{5}\t{6}\t{7}\t{8}\t\t{9}\t{10}\t{11}\t{12}\t\t{13}\t{14}\t{15}\t{16}\t\n".format(
                        num, dice_on1, dice_ocn1, dice_tgn1, dice_fvn1, jac_on1, jac_ocn1, jac_tgn1, jac_fvn1,
                        precision_on1, precision_ocn1, precision_tgn1, precision_fvn1, asd_on1, asd_ocn1, asd_tgn1,
                        asd_fvn1))

            dice_on.append(dice_on1)
            dice_ocn.append(dice_ocn1)
            dice_tgn.append(dice_tgn1)
            dice_fvn.append(dice_fvn1)

            jac_on.append(jac_on1)
            jac_ocn.append(jac_ocn1)
            jac_tgn.append(jac_tgn1)
            jac_fvn.append(jac_fvn1)

            precision_on.append(precision_on1)
            precision_ocn.append(precision_ocn1)
            precision_tgn.append(precision_tgn1)
            precision_fvn.append(precision_fvn1)

            asd_on.append(asd_on1)
            asd_ocn.append(asd_ocn1)
            asd_tgn.append(asd_tgn1)
            asd_fvn.append(asd_fvn1)
        mean_dice_on = sum(dice_on) / len(dice_on)
        mean_dice_ocn = sum(dice_ocn) / len(dice_ocn)
        mean_dice_tgn = sum(dice_tgn) / len(dice_tgn)
        mean_dice_fvn = sum(dice_fvn) / len(dice_fvn)
        mean_dice = (mean_dice_on + mean_dice_ocn + mean_dice_tgn + mean_dice_fvn) / 4

        mean_jac_on = sum(jac_on) / len(jac_on)
        mean_jac_ocn = sum(jac_ocn) / len(jac_ocn)
        mean_jac_tgn = sum(jac_tgn) / len(jac_tgn)
        mean_jac_fvn = sum(jac_fvn) / len(jac_fvn)
        mean_jac = (mean_jac_on + mean_jac_ocn + mean_jac_tgn + mean_jac_fvn) / 4

        mean_precision_on = sum(precision_on) / len(precision_on)
        mean_precision_ocn = sum(precision_ocn) / len(precision_ocn)
        mean_precision_tgn = sum(precision_tgn) / len(precision_tgn)
        mean_precision_fvn = sum(precision_fvn) / len(precision_fvn)
        mean_precision = (mean_precision_on + mean_precision_ocn + mean_precision_tgn + mean_precision_fvn) / 4

        mean_asd_on = sum(asd_on) / len(asd_on)
        mean_asd_ocn = sum(asd_ocn) / len(asd_ocn)
        mean_asd_tgn = sum(asd_tgn) / len(asd_tgn)
        mean_asd_fvn = sum(asd_fvn) / len(asd_fvn)
        mean_asd = (mean_asd_on + mean_asd_ocn + mean_asd_tgn + mean_asd_fvn) / 4
        print(mean_dice_on, mean_dice_ocn, mean_dice_tgn, mean_dice_fvn,mean_jac_on,mean_jac_ocn,mean_jac_tgn,mean_jac_fvn,mean_precision_on,mean_precision_ocn,mean_precision_tgn,mean_precision_fvn,mean_asd_on, mean_asd_ocn, mean_asd_tgn, mean_asd_fvn)

