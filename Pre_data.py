import time, os
import nibabel as nib
import numpy as np
from scipy import ndimage as nd

def cut_data(data_path,data_num):
    print('-' * 30)
    print('Begin cut data')
    print('-' * 30)
    # for each volume do:
    # img_t1_name =  'T1.nii.gz'
    img_t1_name = data_path + '/'+ str(data_num)+ '_CN-T1.nii.gz'
    img_mask_name = data_path + '/' + str(data_num) + '_CN-mask.nii.gz'
    img_peaks_name = data_path + '/' + str(data_num) + '_CN-Peaks.nii.gz'
    img_label_tgn_name = data_path + '/' + str(data_num) + '_TGN-label.nii.gz'
    img_label_on_name = data_path + '/' + str(data_num) + '_ON-label.nii.gz'
    img_label_ocn_name = data_path + '/' + str(data_num) + '_OCN-label.nii.gz'
    img_label_fvn_name = data_path + '/' + str(data_num) + '_FVN-label.nii.gz'

    img_fa_name = data_path + '/'+ str(data_num) + '_CN-FA.nii.gz'

    img_peaks = nib.load(img_peaks_name)
    img_peaks_affine = img_peaks.get_affine()
    img_peaks_data = img_peaks.get_data()
    img_peaks_data = np.squeeze(img_peaks_data)

    img_mask = nib.load(img_mask_name)
    img_mask_affine = img_mask.get_affine()
    img_mask_data = img_mask.get_data()
    img_mask_data = np.squeeze(img_mask_data)

    img_t1 = nib.load(img_t1_name)
    img_t1_affine = img_t1.get_affine()
    img_t1_data = img_t1.get_data()
    img_t1_data = np.squeeze(img_t1_data)

    img_label_tgn = nib.load(img_label_tgn_name)
    img_label_tgn_affine = img_label_tgn.get_affine()
    img_label_tgn_data = img_label_tgn.get_data()
    img_label_tgn_data = np.squeeze(img_label_tgn_data)

    img_label_on = nib.load(img_label_on_name)
    img_label_on_affine = img_label_on.get_affine()
    img_label_on_data = img_label_on.get_data()
    img_label_on_data = np.squeeze(img_label_on_data)

    img_label_fvn = nib.load(img_label_fvn_name)
    img_label_fvn_affine = img_label_fvn.get_affine()
    img_label_fvn_data = img_label_fvn.get_data()
    img_label_fvn_data = np.squeeze(img_label_fvn_data)

    img_label_ocn = nib.load(img_label_ocn_name)
    img_label_ocn_affine = img_label_ocn.get_affine()
    img_label_ocn_data = img_label_ocn.get_data()
    img_label_ocn_data = np.squeeze(img_label_ocn_data)

    img_fa = nib.load(img_fa_name)
    img_fa_data = img_fa.get_data()
    img_fa_affine = img_fa.get_affine()
    img_fa_data = np.squeeze(img_fa_data)

    # dsfactor = [w / float(f) for w, f in zip(img_t1_data.shape, img_fa_data.shape)]
    # FA_reshape = nd.interpolation.zoom(img_fa_data, zoom=dsfactor)

    T1_cut_data = img_t1_data[10:138,10:170,10:138 ]
    FA_cut_data = img_fa_data[10:138, 10:170, 10:138]
    peaks_cut_data = img_peaks_data[10:138, 10:170, 10:138,:]
    mask_cut_data = img_mask_data[10:138, 10:170, 10:138]
    label_tgn_cut_data = img_label_tgn_data[10:138, 10:170, 10:138]
    label_on_cut_data = img_label_on_data[10:138, 10:170, 10:138]
    label_ocn_cut_data = img_label_ocn_data[10:138, 10:170, 10:138]
    label_fvn_cut_data = img_label_fvn_data[10:138, 10:170, 10:138]

    T1_save_path = data_path + '/' + str(data_num) + '_CN-T1.nii.gz'
    # data_path + '/' + 'str(data_num)_' + 'ON-new_T1.nii.gz'
    T1_cut_data = nib.Nifti1Image(T1_cut_data, img_t1_affine)
    nib.save(T1_cut_data, T1_save_path)

    FA_save_path = data_path + '/' + str(data_num) + '_CN-FA.nii.gz'
    FA_cut_data = nib.Nifti1Image(FA_cut_data, img_fa_affine)
    nib.save(FA_cut_data, FA_save_path)

    peaks_save_path = data_path + '/' + str(data_num) + '_CN-Peaks.nii.gz'
    peaks_cut_data = nib.Nifti1Image(peaks_cut_data, img_peaks_affine)
    nib.save(peaks_cut_data, peaks_save_path)

    mask_save_path = data_path + '/' + str(data_num) + '_CN-mask.nii.gz'
    mask_cut_data = nib.Nifti1Image(mask_cut_data, img_mask_affine)
    nib.save(mask_cut_data, mask_save_path)


    label_on_save_path = data_path + '/' + str(data_num) + '_ON-label.nii.gz'
    label_on_cut_data = nib.Nifti1Image(label_on_cut_data, img_label_on_affine)
    nib.save(label_on_cut_data, label_on_save_path)

    label_tgn_save_path = data_path + '/' + str(data_num) + '_TGN-label.nii.gz'
    label_tgn_cut_data = nib.Nifti1Image(label_tgn_cut_data, img_label_tgn_affine)
    nib.save(label_tgn_cut_data, label_tgn_save_path)

    label_ocn_save_path = data_path + '/' + str(data_num) + '_OCN-label.nii.gz'
    label_ocn_cut_data = nib.Nifti1Image(label_ocn_cut_data, img_label_ocn_affine)
    nib.save(label_ocn_cut_data, label_ocn_save_path)

    label_fvn_save_path = data_path + '/' + str(data_num) + '_FVN-label.nii.gz'
    label_fvn_cut_data = nib.Nifti1Image(label_fvn_cut_data, img_label_fvn_affine)
    nib.save(label_fvn_cut_data, label_fvn_save_path)

    print('-' * 30)





def get_255(data_path,data_num):
    print('-' * 30)
    print('Begin get 255')
    print('-' * 30)
    # for each volume do:
    img_t1_name = data_path + '/' + str(data_num) + '_CN-T1.nii.gz'
    # img_t1_name = os.path.join(data_path, img_t1_name)

    img_fa_name = data_path + '/' + str(data_num) + '_CN-FA.nii.gz'
    # img_fa_name = os.path.join(data_path, img_fa_name)

    img_t1 = nib.load(img_t1_name)
    img_t1_affine = img_t1.get_affine()
    img_t1_data = img_t1.get_data()
    img_t1_data = np.squeeze(img_t1_data)

    img_fa = nib.load(img_fa_name)
    img_fa_data = img_fa.get_data()
    img_fa_affine = img_fa.get_affine()
    img_fa_data = np.squeeze(img_fa_data)
    img_fa_data = np.nan_to_num(img_fa_data)


    # min = np.nanmin(img_fa_data[:, :, 20])
    # max = np.nanmax(img_fa_data[:, :, 20])


    X = img_t1_data.shape
    empty_t1 = np.zeros([128,160,128])
    empty_fa = np.zeros([128,160,128])
    for slice in range(X[2]):
        array_t1 = 255 * (img_t1_data[:, :, slice] - np.min(img_t1_data[:, :, slice])) / (np.max(img_t1_data[:, :, slice]) - np.min(img_t1_data[:, :, slice]))
        array_fa = 255 * (img_fa_data[:, :, slice] - np.nanmin(img_fa_data[:, :, slice])) / (np.nanmax(img_fa_data[:, :, slice]) - np.nanmin(img_fa_data[:, :, slice]))

        array_t1 = array_t1.astype(np.uint8)
        array_fa = array_fa.astype(np.uint8)

        empty_t1[:, :, slice] = array_t1
        empty_fa[:, :, slice] = array_fa

    T1_save_path = data_path + '/' + str(data_num) + '_CN-T1.nii.gz'
    FA_save_path = data_path + '/' + str(data_num) + '_CN-FA.nii.gz'

    t1_data = nib.Nifti1Image(empty_t1, img_t1_affine)
    fa_data = nib.Nifti1Image(empty_fa, img_fa_affine)

    nib.save(t1_data, T1_save_path)
    nib.save(fa_data, FA_save_path)

    print('-' * 30)
def get_255_T2(data_path,data_num):
    print('-' * 30)
    print('Begin get 255')
    print('-' * 30)
    # for each volume do:
    img_t1_name = data_path + '/' + str(data_num) + '_CN-T2.nii.gz'
    # img_t1_name = os.path.join(data_path, img_t1_name)

    # img_fa_name = data_path + '/' + str(data_num) + '_CN-FA.nii.gz'
    # # img_fa_name = os.path.join(data_path, img_fa_name)

    img_t1 = nib.load(img_t1_name)
    img_t1_affine = img_t1.get_affine()
    img_t1_data = img_t1.get_data()
    img_t1_data = np.squeeze(img_t1_data)

    # img_fa = nib.load(img_fa_name)
    # img_fa_data = img_fa.get_data()
    # img_fa_affine = img_fa.get_affine()
    # img_fa_data = np.squeeze(img_fa_data)
    # img_fa_data = np.nan_to_num(img_fa_data)


    # min = np.nanmin(img_fa_data[:, :, 20])
    # max = np.nanmax(img_fa_data[:, :, 20])


    X = img_t1_data.shape
    empty_t1 = np.zeros([128,160,128])
    # empty_fa = np.zeros([128,160,128])
    for slice in range(X[2]):
        array_t1 = 255 * (img_t1_data[:, :, slice] - np.min(img_t1_data[:, :, slice])) / (np.max(img_t1_data[:, :, slice]) - np.min(img_t1_data[:, :, slice]))
        # array_fa = 255 * (img_fa_data[:, :, slice] - np.nanmin(img_fa_data[:, :, slice])) / (np.nanmax(img_fa_data[:, :, slice]) - np.nanmin(img_fa_data[:, :, slice]))

        array_t1 = array_t1.astype(np.uint8)
        # array_fa = array_fa.astype(np.uint8)

        empty_t1[:, :, slice] = array_t1
        # empty_fa[:, :, slice] = array_fa

    T1_save_path = data_path + '/' + str(data_num) + '_CN-T2.nii.gz'
    # FA_save_path = data_path + '/' + str(data_num) + '_CN-FA.nii.gz'

    t1_data = nib.Nifti1Image(empty_t1, img_t1_affine)
    # fa_data = nib.Nifti1Image(empty_fa, img_fa_affine)

    nib.save(t1_data, T1_save_path)
    # nib.save(fa_data, FA_save_path)

    print('-' * 30)


if __name__ == '__main__':
    path = '/media/brainplan/XLdata/CNTSeg++/Data/NewDataset/TrainSet'
    dir = os.listdir(path)
    for data_num in dir:
        data_path = path +'/'+ data_num
        # print(data_path)
        # cut_data(data_path,data_num)
        get_255_T2(data_path,data_num)
