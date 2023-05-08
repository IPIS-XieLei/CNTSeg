
import os
import SimpleITK as sitk
import numpy as np
from skimage import measure
import math
from multiprocessing.dummy import Pool as ThreadPool
import nibabel as nib

def mutildata_onedata(pre_path,ON_save,OCN_save,TGN_save,FVN_save):

    pre = nib.load(pre_path)
    pre_data = pre.get_data()
    pre_affine = pre.get_affine()

    ON = np.zeros(pre_data.shape,dtype='int16')
    OCN = np.zeros(pre_data.shape,dtype='int16')
    TGN = np.zeros(pre_data.shape,dtype='int16')
    FVN = np.zeros(pre_data.shape,dtype='int16')
    ON[pre_data==1]=1
    OCN[pre_data==4]=1
    TGN[pre_data==2]=1
    FVN[pre_data==3]=1
    ON = nib.Nifti1Image(ON, pre_affine)
    OCN = nib.Nifti1Image(OCN, pre_affine)
    TGN = nib.Nifti1Image(TGN, pre_affine)
    FVN = nib.Nifti1Image(FVN, pre_affine)
    nib.save(ON, ON_save)
    nib.save(OCN, OCN_save)
    nib.save(TGN, TGN_save)
    nib.save(FVN, FVN_save)

    # sitk.WriteImage(ON, ON_save)

if __name__=="__main__":
    savepath_ON = r"D:\TGN_AVP_FVN\CODE\CNTSeg\predict_test0801\CN\ON"
    savepath_OCN = r"D:\TGN_AVP_FVN\CODE\CNTSeg\predict_test0801\CN\OCN"
    savepath_TGN = r"D:\TGN_AVP_FVN\CODE\CNTSeg\predict_test0801\CN\TGN"
    savepath_FVN= r"D:\TGN_AVP_FVN\CODE\CNTSeg\predict_test0801\CN\FVN"
    gtpath = r"D:\TGN_AVP_FVN\CN\data_net_128_160_128_int16_0330\TestSet"
    pre_path = r"D:\TGN_AVP_FVN\CODE\CNTSeg\predict_test0801"
    gtpath_num = os.listdir(gtpath)
    os.mkdir(savepath_ON)
    os.mkdir(savepath_OCN)
    os.mkdir(savepath_TGN)
    os.mkdir(savepath_FVN)

    for num in gtpath_num:

        pre = pre_path + '/' + 'test_result_' + num + '/'+'pre_final-label.nii.gz'
        save_ON = savepath_ON + '/' + '/' + num  + '_ON-pre.nii.gz'
        save_OCN = savepath_OCN + '/' + num + '_OCN-pre.nii.gz'
        save_TGN = savepath_TGN + '/' + num + '_TGN-pre.nii.gz'
        save_FVN = savepath_FVN + '/' + num + '_FVN-pre.nii.gz'

        mutildata_onedata(pre,save_ON,save_OCN,save_TGN,save_FVN)

