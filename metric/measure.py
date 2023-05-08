
import os
import SimpleITK as sitk
import numpy as np
from skimage import measure
import math
from multiprocessing.dummy import Pool as ThreadPool
import cmath

def measure_all(gtsavepath,prepath):

    seg = sitk.GetArrayFromImage(sitk.ReadImage(gtsavepath, sitk.sitkInt16))
    label = sitk.GetArrayFromImage(sitk.ReadImage(prepath, sitk.sitkInt16))

    #### 计算 dice
    zeros =np.zeros(seg.shape)  # 全0变量
    ones = np.ones(seg.shape)  # 全1变量
    tp =((seg == ones) & (label == ones)).sum()
    fp=((seg==zeros) & (label==ones)).sum()
    tn=((seg==zeros) & (label==zeros)).sum()
    fn=((seg==ones) & (label==zeros)).sum()
    core=0.000000000000000001
    dice = (tp*2)/(fp+tp*2+fn)
    # mcc = (tp*tn-fp*fn)/(((tp+fp+core)*(tp+fn+core)*(tn+fp+core)*(tn+fn+core))**0.5)
    acc=(tp+tn+core)/(tp+fp+tn+fn+core)
    precision=(tp+core)/(tp+fp+core)
    recall_sen=(tp+core)/(tp+fn+core)
    spc=(tn+core)/(tn+fp+core)
    jac = tp/(fp+tp+fn)
    #### 计算 hausdorff
    quality = dict()
    seg1 = sitk.ReadImage(gtsavepath, sitk.sitkInt16)
    label1 = sitk.ReadImage(prepath, sitk.sitkInt16)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    if ((sitk.GetArrayFromImage(seg1).sum() > 0) and (sitk.GetArrayFromImage(label1).sum() > 0)):
        hausdorffcomputer.Execute(seg1, label1)  # (labelTrue > 0.5, labelPred > 0.5)
        quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
        quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()
    else:
        quality["avgHausdorff"] = "max"
        quality["Hausdorff"] = "max"
    #####################################################################################################
    # image=sitk.ReadImage(os.path.join(fake_path, prediction), sitk.sitkInt16)
    # img_x = sitk.GetArrayFromImage(image)
    # label2, islands_num = measure.label(img_x, connectivity=3, background=0, return_num=True)
    # # Counter(label)
    # # b = np.copy(label)
    # # c = np.copy(label)
    # # print(islands_num)
    # # with open(r'C:\Users\Administrator\Desktop\manual\config_48\pre\se3dUnet_weights_2_13\pre_all\islands.txt', 'a+') as f:
    # #     f.writelines("{0} ".format(i))
    # #     f.writelines("{0}\n".format(str(islands_num)))
    #
    #
    # for index in range(1,label2.max()+1):
    #     # b = np.copy(label)
    #     if(np.sum(label2==index)<16):
    #         label2[label2==index]=0
    #     # b = np.copy(label)
    #     if (np.sum(label2==index) >=16):
    #         label2[label2==index]=1
    #
    # result = sitk.GetImageFromArray(label2)
    # result.SetDirection(image.GetDirection())
    # result.SetOrigin(image.GetOrigin())
    # result.SetSpacing(image.GetSpacing())
    # result = sitk.GetArrayFromImage(result)
    # label1, islands_num1 = measure.label(result, connectivity=3, background=0, return_num=True)
    #####################################################################################################
    # print(islands_num1)
    # with open(r'Z:\XL\SHIYAN\zheyi\islandsgan.txt', 'a+') as f:
    #     f.writelines("{0} ".format(i))
    #     f.writelines("{0}\n".format(str(islands_num1)))
    # tpr=(tp+core)/(tp+fn+core)
    # fpr=(fp+core)/(tn+fp+core)
    # F1=(precision*recall_sen+core)/(precision+recall_sen+core)
    # # q1=tp*tn-fp*fn
    # q2=((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5
    # Matthews= q1/q2
    # print("{0}".format(q1))
    # # with open(r"D:\yiyelunwen_cyk\spc_gan.txt", 'a+') as f:
    # #     f.writelines("{0}\n".format(spc))
    # print("{0}".format(Matthews))
    # print(tp,fp,tn,fn)
    with open(r"D:\TGN_AVP_FVN\CODE\CNTSeg\final_results\result1_excel\LOSS\dice_FVN.txt", 'a+') as f:
        f.writelines("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(prediction,dice,jac,acc,precision,recall_sen,spc,quality["Hausdorff"],quality["avgHausdorff"]))
    print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(prediction,dice,jac,acc,precision,recall_sen,spc,quality["Hausdorff"],quality["avgHausdorff"]))
def measure_DSC(gtsavepath,prepath):

    seg = sitk.GetArrayFromImage(sitk.ReadImage(gtsavepath, sitk.sitkInt16))
    label = sitk.GetArrayFromImage(sitk.ReadImage(prepath, sitk.sitkInt16))

    #### 计算 dice
    zeros =np.zeros(seg.shape)  # 全0变量
    ones = np.ones(seg.shape)  # 全1变量
    tp =((seg == ones) & (label == ones)).sum()
    fp=((seg==zeros) & (label==ones)).sum()
    tn=((seg==zeros) & (label==zeros)).sum()
    fn=((seg==ones) & (label==zeros)).sum()
    core=0.000000000000000001
    dice = (tp*2)/(fp+tp*2+fn)
    # mcc = (tp*tn-fp*fn)/(((tp+fp+core)*(tp+fn+core)*(tn+fp+core)*(tn+fn+core))**0.5)
    acc=(tp+tn+core)/(tp+fp+tn+fn+core)
    precision=(tp+core)/(tp+fp+core)
    recall_sen=(tp+core)/(tp+fn+core)
    spc=(tn+core)/(tn+fp+core)
    jac = tp/(fp+tp+fn)
    #### 计算 hausdorff
    quality = dict()
    seg1 = sitk.ReadImage(gtsavepath, sitk.sitkInt16)
    label1 = sitk.ReadImage(prepath, sitk.sitkInt16)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    if ((sitk.GetArrayFromImage(seg1).sum() > 0) and (sitk.GetArrayFromImage(label1).sum() > 0)):
        hausdorffcomputer.Execute(seg1, label1)  # (labelTrue > 0.5, labelPred > 0.5)
        quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
        quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()
    else:
        quality["avgHausdorff"] = "max"
        quality["Hausdorff"] = "max"
    #####################################################################################################
    # image=sitk.ReadImage(os.path.join(fake_path, prediction), sitk.sitkInt16)
    # img_x = sitk.GetArrayFromImage(image)
    # label2, islands_num = measure.label(img_x, connectivity=3, background=0, return_num=True)
    # # Counter(label)
    # # b = np.copy(label)
    # # c = np.copy(label)
    # # print(islands_num)
    # # with open(r'C:\Users\Administrator\Desktop\manual\config_48\pre\se3dUnet_weights_2_13\pre_all\islands.txt', 'a+') as f:
    # #     f.writelines("{0} ".format(i))
    # #     f.writelines("{0}\n".format(str(islands_num)))
    #
    #
    # for index in range(1,label2.max()+1):
    #     # b = np.copy(label)
    #     if(np.sum(label2==index)<16):
    #         label2[label2==index]=0
    #     # b = np.copy(label)
    #     if (np.sum(label2==index) >=16):
    #         label2[label2==index]=1
    #
    # result = sitk.GetImageFromArray(label2)
    # result.SetDirection(image.GetDirection())
    # result.SetOrigin(image.GetOrigin())
    # result.SetSpacing(image.GetSpacing())
    # result = sitk.GetArrayFromImage(result)
    # label1, islands_num1 = measure.label(result, connectivity=3, background=0, return_num=True)
    #####################################################################################################
    # print(islands_num1)
    # with open(r'Z:\XL\SHIYAN\zheyi\islandsgan.txt', 'a+') as f:
    #     f.writelines("{0} ".format(i))
    #     f.writelines("{0}\n".format(str(islands_num1)))
    # tpr=(tp+core)/(tp+fn+core)
    # fpr=(fp+core)/(tn+fp+core)
    # F1=(precision*recall_sen+core)/(precision+recall_sen+core)
    # # q1=tp*tn-fp*fn
    # q2=((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5
    # Matthews= q1/q2
    # print("{0}".format(q1))
    # # with open(r"D:\yiyelunwen_cyk\spc_gan.txt", 'a+') as f:
    # #     f.writelines("{0}\n".format(spc))
    # print("{0}".format(Matthews))
    # print(tp,fp,tn,fn)
    with open(r"D:\TGN_AVP_FVN\CODE\CNTSeg\final_results\result1_excel\3D\ON.txt", 'a+') as f:
        f.writelines("{0}\t{1}\t{2}\t{3}\t{4}\n".format(prediction,dice,jac,precision,quality["avgHausdorff"]))
    print("{0}\t{1}\t{2}\t{3}\t{4}\n".format(prediction,dice,jac,precision,quality["avgHausdorff"]))
if __name__=="__main__":
    gtsavepath = r"D:\TGN_AVP_FVN\CN\data_net_128_160_128_int16_0330\TestSet"
    # T1,FA,Peaks
    prepath = r"D:\TGN_AVP_FVN\CODE\CNTSeg\predict_test0801\CN\FVN"  # ON,OCN,TGN,FVN
    pre = os.listdir(prepath)
    for prediction in pre:
        name = prediction.split('-')[0]
        num = prediction.split('_')[0]
        predata = prepath+'/'+prediction
        gtdata = gtsavepath+'/'+num+'/'+name+'-label.nii.gz'
        # print(predata,gtdata)
        measure_all(gtdata, predata)

