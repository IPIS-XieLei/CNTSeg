import numpy as np
import config_2d
import torch
import time, os
import nibabel as nib
from skimage import measure

batch_size =config_2d.BATCH_SIZE
patch_size_w = config_2d.PATCH_SIZE_W
patch_size_h = config_2d.PATCH_SIZE_H
n_classes = config_2d.NUM_CLASSES



train_imgs_path = '/media/brainplan/XLdata/CNTSeg++/Data/NewDataset/fold5/TrainSet'
val_imgs_path = '/media/brainplan/XLdata/CNTSeg++/Data/NewDataset/fold5/TestValSet'
# test_imgs_path = '/media/brainplan/XLdata/CNTSeg++/Data/NewDataset/TestSet'



##### create train data 训练集数据 ####
def create_train_data_T1FA(imgs_path, save_path):

    images_dir = os.listdir(imgs_path)
    #####################################################
    #  y_dir_1:on,y_dir_2:tgn,y_dir_3:fvn,y_dir_4:ocn
    ####################################################
    x_t1_dir = os.path.join(save_path, 'x_t1_data')
    x_t2_dir = os.path.join(save_path, 'x_t2_data')
    x_fa_dir = os.path.join(save_path, 'x_fa_data')
    x_dec_dir = os.path.join(save_path, 'x_dec_data')
    x_peaks_dir = os.path.join(save_path, 'x_peaks_data')
    y_dir_1 = os.path.join(save_path, 'y_data_on')
    y_dir_2 = os.path.join(save_path, 'y_data_tgn')
    y_dir_3 = os.path.join(save_path, 'y_data_fvn')
    y_dir_4 = os.path.join(save_path, 'y_data_ocn')

    j = 0
    print('-' * 30)
    print('Creating train 2d_patches...')
    print('-' * 30)

    CN_all_patch_num = 0
    # CL_all_patch_num = 0

    num = 0
    # for each volume do:
    for img_dir_name in images_dir:
        num = num + 1
        # T1
        img_t1_name = img_dir_name + '_CN-T1.nii.gz'  # T1
        img_t1_name = os.path.join(imgs_path, img_dir_name, img_t1_name)
        # T2
        img_t2_name = img_dir_name + '_CN-T2.nii.gz'  # T2
        img_t2_name = os.path.join(imgs_path, img_dir_name, img_t2_name)
        # FA
        img_t1_fa_name = img_dir_name + '_CN-FA.nii.gz'  # FA
        img_t1_fa_name = os.path.join(imgs_path, img_dir_name, img_t1_fa_name)
        # dec
        img_dec_name = img_dir_name + '_CN-DEC.nii.gz'  # peaks
        img_dec_name = os.path.join(imgs_path, img_dir_name, img_dec_name)
        # peaks
        img_peaks_name = img_dir_name + '_CN-Peaks.nii.gz'  # peaks
        img_peaks_name = os.path.join(imgs_path, img_dir_name, img_peaks_name)
        # masks
        img_masks_name = img_dir_name + '_CN-mask.nii.gz'  # mask
        img_masks_name = os.path.join(imgs_path, img_dir_name, img_masks_name)
        # label
        img_label_name1 = img_dir_name + '_ON-label.nii.gz'
        img_label_name1 = os.path.join(imgs_path, img_dir_name, img_label_name1)
        img_label_name2 = img_dir_name + '_TGN-label.nii.gz'
        img_label_name2 = os.path.join(imgs_path, img_dir_name, img_label_name2)
        img_label_name3 = img_dir_name + '_FVN-label.nii.gz'
        img_label_name3 = os.path.join(imgs_path, img_dir_name, img_label_name3)
        img_label_name4 = img_dir_name + '_OCN-label.nii.gz'
        img_label_name4 = os.path.join(imgs_path, img_dir_name, img_label_name4)


        # load T1, FA, label and mask
        img_t1 = nib.load(img_t1_name)
        img_t1_data = img_t1.get_data()
        img_t1_affine = img_t1.get_affine()
        img_t1_data = np.squeeze(img_t1_data)

        img_t2 = nib.load(img_t2_name)
        img_t2_data = img_t2.get_data()
        img_t2_affine = img_t2.get_affine()
        img_t2_data = np.squeeze(img_t2_data)

        img_fa = nib.load(img_t1_fa_name)
        img_fa_data = img_fa.get_data()
        img_fa_affine = img_fa.get_affine()
        img_fa_data = np.squeeze(img_fa_data)

        img_dec = nib.load(img_dec_name)
        img_dec_data = img_dec.get_data()
        img_dec_affine = img_dec.get_affine()
        img_dec_data = np.squeeze(img_dec_data)

        img_peaks = nib.load(img_peaks_name)
        img_peaks_data = img_peaks.get_data()
        img_peaks_affine = img_peaks.get_affine()
        img_peaks_data = np.squeeze(img_peaks_data)

        img_masks = nib.load(img_masks_name)
        img_masks_data = img_masks.get_data()
        # img_masks_affine = img_masks.get_affine()
        img_masks_data = np.squeeze(img_masks_data)

        img_label1 = nib.load(img_label_name1)
        img_label_data1 = img_label1.get_data()
        img_label_affine1 = img_label1.get_affine()
        img_label_data1 = np.squeeze(img_label_data1)

        img_label2 = nib.load(img_label_name2)
        img_label_data2 = img_label2.get_data()
        img_label_affine2 = img_label2.get_affine()
        img_label_data2 = np.squeeze(img_label_data2)

        img_label3 = nib.load(img_label_name3)
        img_label_data3 = img_label3.get_data()
        img_label_affine3 = img_label3.get_affine()
        img_label_data3 = np.squeeze(img_label_data3)

        img_label4 = nib.load(img_label_name4)
        img_label_data4 = img_label4.get_data()
        img_label_affine4 = img_label4.get_affine()
        img_label_data4 = np.squeeze(img_label_data4)
        img_label_data_all = img_label_data1 + img_label_data2 + img_label_data3 + img_label_data4


        X = img_label_data1.shape


        # for each slice do
        for slice in range(X[2]):
            print('Processing: volume {0} / {1} volume images, slice {2} / {3} slices'.format(j + 1,
                                                                                              len(images_dir),
                                                                                              slice+1,
                                                                                              img_label_data1.shape[2]))



            if np.count_nonzero(img_label_data_all[:, :, slice])>0 and np.count_nonzero(img_masks_data[:, :, slice])>0 and np.count_nonzero(img_t1_data[:, :, slice])>0:
                t1_patches = img_t1_data[:, :, slice]
                t2_patches = img_t2_data[:, :, slice]
                fa_patches = img_fa_data[:, :, slice]
                dec_patches = img_dec_data[:, :, slice, :]
                peaks_patches = img_peaks_data[:, :, slice, :]
                label1_patches = img_label_data1[:, :, slice]
                label2_patches = img_label_data2[:, :, slice]
                label3_patches = img_label_data3[:, :, slice]
                label4_patches = img_label_data4[:, :, slice]
                ### CN (T1)  x1
                # x1 data
                t1_patches_nii = t1_patches
                t1_flip_patches_nii = np.flip(t1_patches_nii, 0)
                t1_nonum_nii_path = x_t1_dir + '/x_t1-data_' + str(CN_all_patch_num) + '.nii.gz'
                t1_patches_nii = nib.Nifti1Image(t1_patches_nii, img_t1_affine)
                nib.save(t1_patches_nii, t1_nonum_nii_path)

                # x1 data T2
                t2_patches_nii = t2_patches
                t2_flip_patches_nii = np.flip(t2_patches_nii, 0)
                t2_nonum_nii_path = x_t2_dir + '/x_t2-data_' + str(CN_all_patch_num) + '.nii.gz'
                t2_patches_nii = nib.Nifti1Image(t2_patches_nii, img_t2_affine)
                nib.save(t2_patches_nii, t2_nonum_nii_path)
                ### CN (FA)  x2
                # x2 data
                fa_patches_nii = fa_patches
                fa_flip_patches_nii = np.flip(fa_patches_nii, 0)
                fa_nonum_nii_path = x_fa_dir + '/x_fa-data_' + str(CN_all_patch_num) + '.nii.gz'
                fa_patches_nii = nib.Nifti1Image(fa_patches_nii, img_fa_affine)
                nib.save(fa_patches_nii, fa_nonum_nii_path)

                # x3 dec
                dec_patches_nii = dec_patches
                dec_flip_patches_nii = np.flip(dec_patches_nii, 0)
                dec_nonum_nii_path = x_dec_dir + '/x_dec-data_' + str(CN_all_patch_num) + '.nii.gz'
                dec_patches_nii = nib.Nifti1Image(dec_patches_nii, img_dec_affine)
                nib.save(dec_patches_nii, dec_nonum_nii_path)
                ### CN (Peaks)  x3
                # x3 peaks
                peaks_patches_nii = peaks_patches
                peaks_flip_patches_nii = np.flip(peaks_patches_nii, 0)
                peaks_nonum_nii_path = x_peaks_dir + '/x_peaks-data_' + str(CN_all_patch_num) + '.nii.gz'
                peaks_patches_nii = nib.Nifti1Image(peaks_patches_nii, img_peaks_affine)
                nib.save(peaks_patches_nii, peaks_nonum_nii_path)
                # y1 data
                label1_patches_nii = label1_patches
                label1_flip_patches_nii = np.flip(label1_patches_nii, 0)  # t1_fa-label flip data
                label1_nonum_nii_path = y_dir_1 + '/y1-data_' + str(CN_all_patch_num) + '.nii.gz'
                label1_patches_nii = nib.Nifti1Image(label1_patches_nii, img_label_affine1)
                nib.save(label1_patches_nii, label1_nonum_nii_path)
                # y2 data
                label2_patches_nii = label2_patches
                label2_flip_patches_nii = np.flip(label2_patches_nii, 0)  # t1_fa-label flip data
                label2_nonum_nii_path = y_dir_2 + '/y2-data_' + str(CN_all_patch_num) + '.nii.gz'
                label2_patches_nii = nib.Nifti1Image(label2_patches_nii, img_label_affine2)
                nib.save(label2_patches_nii, label2_nonum_nii_path)
                # y3 data
                label3_patches_nii = label3_patches
                label3_flip_patches_nii = np.flip(label3_patches_nii, 0)  # t1_fa-label flip data
                label3_nonum_nii_path = y_dir_3 + '/y3-data_' + str(CN_all_patch_num) + '.nii.gz'
                label3_patches_nii = nib.Nifti1Image(label3_patches_nii, img_label_affine3)
                nib.save(label3_patches_nii, label3_nonum_nii_path)
                # y4 data
                label4_patches_nii = label4_patches
                label4_flip_patches_nii = np.flip(label4_patches_nii, 0)  # t1_fa-label flip data
                label4_nonum_nii_path = y_dir_4 + '/y4-data_' + str(CN_all_patch_num) + '.nii.gz'
                label4_patches_nii = nib.Nifti1Image(label4_patches_nii, img_label_affine4)
                nib.save(label4_patches_nii, label4_nonum_nii_path)
                CN_all_patch_num += 1
                ### --------------
                ####数据增强
                ### t1 x flip data  （翻转180）
                t1_flip_nonum_nii_path = x_t1_dir + '/x_t1-data_' + str(CN_all_patch_num) + '.nii.gz'
                t1_flip_patches_nii = nib.Nifti1Image(t1_flip_patches_nii, img_t1_affine)
                nib.save(t1_flip_patches_nii, t1_flip_nonum_nii_path)
                ### t2 x flip data  （翻转180）
                t2_flip_nonum_nii_path = x_t2_dir + '/x_t2-data_' + str(CN_all_patch_num) + '.nii.gz'
                t2_flip_patches_nii = nib.Nifti1Image(t2_flip_patches_nii, img_t2_affine)
                nib.save(t2_flip_patches_nii, t2_flip_nonum_nii_path)
                ### fa x flip data  （翻转180）
                fa_flip_nonum_nii_path = x_fa_dir + '/x_fa-data_' + str(CN_all_patch_num) + '.nii.gz'
                fa_flip_patches_nii = nib.Nifti1Image(fa_flip_patches_nii, img_fa_affine)
                nib.save(fa_flip_patches_nii, fa_flip_nonum_nii_path)
                ### dec x flip data  （翻转180）
                dec_flip_nonum_nii_path = x_dec_dir + '/x_dec-data_' + str(CN_all_patch_num) + '.nii.gz'
                dec_flip_patches_nii = nib.Nifti1Image(dec_flip_patches_nii, img_dec_affine)
                nib.save(dec_flip_patches_nii, dec_flip_nonum_nii_path)
                ### peaks x flip data  （翻转180）
                peaks_flip_nonum_nii_path = x_peaks_dir + '/x_peaks-data_' + str(CN_all_patch_num) + '.nii.gz'
                peaks_flip_patches_nii = nib.Nifti1Image(peaks_flip_patches_nii, img_peaks_affine)
                nib.save(peaks_flip_patches_nii, peaks_flip_nonum_nii_path)
                ### y1 flip data
                label1_flip_nonum_nii_path = y_dir_1 + '/y1-data_' + str(CN_all_patch_num) + '.nii.gz'
                label1_flip_patches_nii = nib.Nifti1Image(label1_flip_patches_nii, img_label_affine1)
                nib.save(label1_flip_patches_nii, label1_flip_nonum_nii_path)
                ### y2 flip data
                label2_flip_nonum_nii_path = y_dir_2 + '/y2-data_' + str(CN_all_patch_num) + '.nii.gz'
                label2_flip_patches_nii = nib.Nifti1Image(label2_flip_patches_nii, img_label_affine2)
                nib.save(label2_flip_patches_nii, label2_flip_nonum_nii_path)
                ### y3 flip data
                label3_flip_nonum_nii_path = y_dir_3 + '/y3-data_' + str(CN_all_patch_num) + '.nii.gz'
                label3_flip_patches_nii = nib.Nifti1Image(label3_flip_patches_nii, img_label_affine3)
                nib.save(label3_flip_patches_nii, label3_flip_nonum_nii_path)
                ### y4 flip data
                label4_flip_nonum_nii_path = y_dir_4 + '/y4-data_' + str(CN_all_patch_num) + '.nii.gz'
                label4_flip_patches_nii = nib.Nifti1Image(label4_flip_patches_nii, img_label_affine4)
                nib.save(label4_flip_patches_nii, label4_flip_nonum_nii_path)
                CN_all_patch_num += 1
                ### --------------
        j += 1
        print('Input num:   {0}'.format(img_dir_name))
        # print('EC All Patch num:  {0}'.format(EC_all_patch_num))
        print('CN All Patch num:  {0}'.format(CN_all_patch_num))

        print('Patch size:  [{0}*{1}]'.format(patch_size_w, patch_size_h))
    print('-' * 30)
    print('CN All Patches: {0}'.format(CN_all_patch_num))

    print('-' * 30)
##### create val data 创建验证集数据 ####
def create_val_data_T1FA(img_dir_name, save_path):
    #####################################################
    #  y_dir_1:on,y_dir_2:tgn,y_dir_3:fvn,y_dir_4:ocn
    ####################################################
    x_t1_dir = os.path.join(save_path, 'x_t1_data')
    x_t2_dir = os.path.join(save_path, 'x_t2_data')
    x_fa_dir = os.path.join(save_path, 'x_fa_data')
    x_dec_dir = os.path.join(save_path, 'x_dec_data')
    x_peaks_dir = os.path.join(save_path, 'x_peaks_data')
    y_dir_1 = os.path.join(save_path, 'y_data_on')
    y_dir_2 = os.path.join(save_path, 'y_data_tgn')
    y_dir_3 = os.path.join(save_path, 'y_data_fvn')
    y_dir_4 = os.path.join(save_path, 'y_data_ocn')


    j = 0
    print('-' * 30)
    print('Creating test 2d_patches...')
    print('-' * 30)

    # for each volume do:

    img_t1_name = img_dir_name + '_CN-T1.nii.gz'
    img_t1_name = os.path.join(val_imgs_path, img_dir_name, img_t1_name)

    img_t2_name = img_dir_name + '_CN-T2.nii.gz'
    img_t2_name = os.path.join(val_imgs_path, img_dir_name, img_t2_name)

    img_fa_name = img_dir_name + '_CN-FA.nii.gz'
    img_fa_name = os.path.join(val_imgs_path, img_dir_name, img_fa_name)

    img_dec_name = img_dir_name + '_CN-DEC.nii.gz'
    img_dec_name = os.path.join(val_imgs_path, img_dir_name, img_dec_name)

    img_peaks_name = img_dir_name + '_CN-Peaks.nii.gz'
    img_peaks_name = os.path.join(val_imgs_path, img_dir_name, img_peaks_name)


    img_mask_name = img_dir_name + '_CN-mask.nii.gz'
    img_mask_name = os.path.join(val_imgs_path, img_dir_name, img_mask_name)

    img_label_name1 = img_dir_name + '_ON-label.nii.gz'
    img_label_name1 = os.path.join(val_imgs_path, img_dir_name, img_label_name1)

    img_label_name2 = img_dir_name + '_TGN-label.nii.gz'
    img_label_name2 = os.path.join(val_imgs_path, img_dir_name, img_label_name2)

    img_label_name3 = img_dir_name + '_FVN-label.nii.gz'
    img_label_name3 = os.path.join(val_imgs_path, img_dir_name, img_label_name3)

    img_label_name4 = img_dir_name + '_OCN-label.nii.gz'
    img_label_name4 = os.path.join(val_imgs_path, img_dir_name, img_label_name4)

    # T1.nii
    img_t1 = nib.load(img_t1_name)
    img_t1_affine = img_t1.get_affine()
    img_t1_data = img_t1.get_data()
    img_t1_data = np.squeeze(img_t1_data)
    # T2.nii
    img_t2 = nib.load(img_t2_name)
    img_t2_affine = img_t2.get_affine()
    img_t2_data = img_t2.get_data()
    img_t2_data = np.squeeze(img_t2_data)
    # FA.nii
    img_fa = nib.load(img_fa_name)
    img_fa_data = img_fa.get_data()
    img_fa_affine = img_fa.get_affine()
    img_fa_data = np.squeeze(img_fa_data)
    # dec
    img_dec = nib.load(img_dec_name)
    img_dec_data = img_dec.get_data()
    img_dec_affine = img_dec.get_affine()
    img_dec_data = np.squeeze(img_dec_data)
    # peaks
    img_peaks = nib.load(img_peaks_name)
    img_peaks_data = img_peaks.get_data()
    img_peaks_affine = img_peaks.get_affine()
    img_peaks_data = np.squeeze(img_peaks_data)
    # mask
    img_mask = nib.load(img_mask_name)
    img_mask_data = img_mask.get_data()
    # img_mask_affine = img_mask.get_affine()
    img_mask_data = np.squeeze(img_mask_data)
    # label.nii
    img_label1 = nib.load(img_label_name1)
    img_label_data1 = img_label1.get_data()
    img_label_affine1 = img_label1.get_affine()
    img_label_data1 = np.squeeze(img_label_data1)

    img_label2 = nib.load(img_label_name2)
    img_label_data2 = img_label2.get_data()
    img_label_affine2 = img_label2.get_affine()
    img_label_data2 = np.squeeze(img_label_data2)

    img_label3 = nib.load(img_label_name3)
    img_label_data3 = img_label3.get_data()
    img_label_affine3 = img_label3.get_affine()
    img_label_data3 = np.squeeze(img_label_data3)

    img_label4 = nib.load(img_label_name4)
    img_label_data4 = img_label4.get_data()
    img_label_affine4 = img_label4.get_affine()
    img_label_data4 = np.squeeze(img_label_data4)
    img_label_data_all = img_label_data1+img_label_data2+img_label_data3+img_label_data4

    X = img_label_data1.shape
    all_patch_num = 0

    for slice in range(X[2]):
        if np.count_nonzero(img_label_data_all[:, :, slice])>0 and np.count_nonzero(img_t1_data[:, :, slice]) > 0 and np.count_nonzero(img_mask_data[:, :, slice]) > 0:
        # 2D Axial
            t1_patches = img_t1_data[:, :, slice]
            t2_patches = img_t2_data[:, :, slice]
            fa_patches = img_fa_data[:, :, slice]
            dec_patches = img_dec_data[:, :, slice, :]
            peaks_patches = img_peaks_data[:, :, slice, :]
            label1_patches = img_label_data1[:, :, slice]
            label2_patches = img_label_data2[:, :, slice]
            label3_patches = img_label_data3[:, :, slice]
            label4_patches = img_label_data4[:, :, slice]
            # x_t1 data
            t1_patches_nii = t1_patches
            t1_nonum_nii_path = x_t1_dir + '/x_t1-data_' + str(all_patch_num) + '.nii.gz'
            t1_patches_nii = nib.Nifti1Image(t1_patches_nii, img_t1_affine)
            nib.save(t1_patches_nii, t1_nonum_nii_path)
            # x_t2 data
            t2_patches_nii = t2_patches
            t2_nonum_nii_path = x_t2_dir + '/x_t2-data_' + str(all_patch_num) + '.nii.gz'
            t2_patches_nii = nib.Nifti1Image(t2_patches_nii, img_t2_affine)
            nib.save(t2_patches_nii, t2_nonum_nii_path)
            # x_fa data
            fa_patches_nii = fa_patches
            fa_nonum_nii_path = x_fa_dir + '/x_fa-data_' + str(all_patch_num) + '.nii.gz'
            fa_patches_nii = nib.Nifti1Image(fa_patches_nii, img_fa_affine)
            nib.save(fa_patches_nii, fa_nonum_nii_path)
            # x_dec data
            dec_patches_nii = dec_patches
            dec_nonum_nii_path = x_dec_dir + '/x_dec-data_' + str(all_patch_num) + '.nii.gz'
            dec_patches_nii = nib.Nifti1Image(dec_patches_nii, img_dec_affine)
            nib.save(dec_patches_nii, dec_nonum_nii_path)
            # x_fa data
            peaks_patches_nii = peaks_patches
            peaks_nonum_nii_path = x_peaks_dir + '/x_peaks-data_' + str(all_patch_num) + '.nii.gz'
            peaks_patches_nii = nib.Nifti1Image(peaks_patches_nii, img_peaks_affine)
            nib.save(peaks_patches_nii, peaks_nonum_nii_path)
            # y_data
            label1_patches_nii = label1_patches
            label1_nonum_nii_path = y_dir_1 + '/y1-data_' + str(all_patch_num) + '.nii.gz'
            label1_patches_nii = nib.Nifti1Image(label1_patches_nii, img_label_affine1)
            nib.save(label1_patches_nii, label1_nonum_nii_path)
            # y_data
            label2_patches_nii = label2_patches
            label2_nonum_nii_path = y_dir_2 + '/y2-data_' + str(all_patch_num) + '.nii.gz'
            label2_patches_nii = nib.Nifti1Image(label2_patches_nii, img_label_affine2)
            nib.save(label2_patches_nii, label2_nonum_nii_path)
            # y_data
            label3_patches_nii = label3_patches
            label3_nonum_nii_path = y_dir_3 + '/y3-data_' + str(all_patch_num) + '.nii.gz'
            label3_patches_nii = nib.Nifti1Image(label3_patches_nii, img_label_affine3)
            nib.save(label3_patches_nii, label3_nonum_nii_path)
            # y_data
            label4_patches_nii = label4_patches
            label4_nonum_nii_path = y_dir_4 + '/y4-data_' + str(all_patch_num) + '.nii.gz'
            label4_patches_nii = nib.Nifti1Image(label4_patches_nii, img_label_affine4)
            nib.save(label4_patches_nii, label4_nonum_nii_path)
            all_patch_num += 1
        ### --------------
        ### --------------
    j += 1
    print('-' * 30)
##### create test data without_data_partition创建测试集数据 ####
# def create_test_data_T1FA(img_dir_name, save_path):
#     #####################################################
#     #  y_dir_1:on,y_dir_2:tgn,y_dir_3:fvn,y_dir_4:ocn
#     ####################################################
#     x_t1_dir = os.path.join(save_path, 'x_t1_data')
#     x_t2_dir = os.path.join(save_path, 'x_t2_data')
#     x_fa_dir = os.path.join(save_path, 'x_fa_data')
#     x_dec_dir = os.path.join(save_path, 'x_dec_data')
#     x_peaks_dir = os.path.join(save_path, 'x_peaks_data')
#     y_dir_1 = os.path.join(save_path, 'y_data_on')
#     y_dir_2 = os.path.join(save_path, 'y_data_tgn')
#     y_dir_3 = os.path.join(save_path, 'y_data_fvn')
#     y_dir_4 = os.path.join(save_path, 'y_data_ocn')
#
#
#     j = 0
#     print('-' * 30)
#     print('Creating test 2d_patches...')
#     print('-' * 30)
#
#     # for each volume do:
#     img_t1_name = img_dir_name + '_CN-T1.nii.gz'
#     img_t1_name = os.path.join(test_imgs_path, img_dir_name, img_t1_name)
#
#     img_t2_name = img_dir_name + '_CN-T2.nii.gz'
#     img_t2_name = os.path.join(test_imgs_path, img_dir_name, img_t2_name)
#
#     img_fa_name = img_dir_name + '_CN-FA.nii.gz'
#     img_fa_name = os.path.join(test_imgs_path, img_dir_name, img_fa_name)
#
#     img_dec_name = img_dir_name + '_CN-DEC.nii.gz'
#     img_dec_name = os.path.join(test_imgs_path, img_dir_name, img_dec_name)
#
#     img_peaks_name = img_dir_name + '_CN-Peaks.nii.gz'
#     img_peaks_name = os.path.join(test_imgs_path, img_dir_name, img_peaks_name)
#
#
#     img_mask_name = img_dir_name + '_CN-mask.nii.gz'
#     img_mask_name = os.path.join(test_imgs_path, img_dir_name, img_mask_name)
#
#     img_label_name1 = img_dir_name + '_ON-label.nii.gz'
#     img_label_name1 = os.path.join(test_imgs_path, img_dir_name, img_label_name1)
#
#     img_label_name2 = img_dir_name + '_TGN-label.nii.gz'
#     img_label_name2 = os.path.join(test_imgs_path, img_dir_name, img_label_name2)
#
#     img_label_name3 = img_dir_name + '_FVN-label.nii.gz'
#     img_label_name3 = os.path.join(test_imgs_path, img_dir_name, img_label_name3)
#
#     img_label_name4 = img_dir_name + '_OCN-label.nii.gz'
#     img_label_name4 = os.path.join(test_imgs_path, img_dir_name, img_label_name4)
#
#     # T1.nii
#     img_t1 = nib.load(img_t1_name)
#     img_t1_affine = img_t1.get_affine()
#     img_t1_data = img_t1.get_data()
#     img_t1_data = np.squeeze(img_t1_data)
#     # T2.nii
#     img_t2 = nib.load(img_t2_name)
#     img_t2_affine = img_t2.get_affine()
#     img_t2_data = img_t2.get_data()
#     img_t2_data = np.squeeze(img_t2_data)
#     # FA.nii
#     img_fa = nib.load(img_fa_name)
#     img_fa_data = img_fa.get_data()
#     img_fa_affine = img_fa.get_affine()
#     img_fa_data = np.squeeze(img_fa_data)
#     # dec
#     img_dec = nib.load(img_dec_name)
#     img_dec_data = img_dec.get_data()
#     img_dec_affine = img_dec.get_affine()
#     img_dec_data = np.squeeze(img_dec_data)
#     # peaks
#     img_peaks = nib.load(img_peaks_name)
#     img_peaks_data = img_peaks.get_data()
#     img_peaks_affine = img_peaks.get_affine()
#     img_peaks_data = np.squeeze(img_peaks_data)
#     # mask
#     img_mask = nib.load(img_mask_name)
#     img_mask_data = img_mask.get_data()
#     # img_mask_affine = img_mask.get_affine()
#     img_mask_data = np.squeeze(img_mask_data)
#     # label.nii
#     img_label1 = nib.load(img_label_name1)
#     img_label_data1 = img_label1.get_data()
#     img_label_affine1 = img_label1.get_affine()
#     img_label_data1 = np.squeeze(img_label_data1)
#
#     img_label2 = nib.load(img_label_name2)
#     img_label_data2 = img_label2.get_data()
#     img_label_affine2 = img_label2.get_affine()
#     img_label_data2 = np.squeeze(img_label_data2)
#
#     img_label3 = nib.load(img_label_name3)
#     img_label_data3 = img_label3.get_data()
#     img_label_affine3 = img_label3.get_affine()
#     img_label_data3 = np.squeeze(img_label_data3)
#
#     img_label4 = nib.load(img_label_name4)
#     img_label_data4 = img_label4.get_data()
#     img_label_affine4 = img_label4.get_affine()
#     img_label_data4 = np.squeeze(img_label_data4)
#
#     img_label_data_all = img_label_data1+img_label_data2+img_label_data3+img_label_data4
#
#
#     X = img_label_data1.shape
#     all_patch_num = 0
#
#     for slice in range(X[2]):
#         if np.count_nonzero(img_label_data_all[:, :, slice]) > 0 and np.count_nonzero(img_t1_data[:, :, slice]) > 0 and np.count_nonzero(img_mask_data[:, :, slice]) > 0:
#         # 2D Axial
#             t1_patches = img_t1_data[:, :, slice]
#             t2_patches = img_t2_data[:, :, slice]
#             fa_patches = img_fa_data[:, :, slice]
#             dec_patches = img_dec_data[:, :, slice, :]
#             peaks_patches = img_peaks_data[:, :, slice, :]
#             label1_patches = img_label_data1[:, :, slice]
#             label2_patches = img_label_data2[:, :, slice]
#             label3_patches = img_label_data3[:, :, slice]
#             label4_patches = img_label_data4[:, :, slice]
#             # x_t1 data
#             t1_patches_nii = t1_patches
#             t1_nonum_nii_path = x_t1_dir + '/x_t1-data_' + str(all_patch_num) + '.nii.gz'
#             t1_patches_nii = nib.Nifti1Image(t1_patches_nii, img_t1_affine)
#             nib.save(t1_patches_nii, t1_nonum_nii_path)
#             # x_t2 data
#             t2_patches_nii = t2_patches
#             t2_nonum_nii_path = x_t2_dir + '/x_t2-data_' + str(all_patch_num) + '.nii.gz'
#             t2_patches_nii = nib.Nifti1Image(t2_patches_nii, img_t2_affine)
#             nib.save(t2_patches_nii, t2_nonum_nii_path)
#             # x_fa data
#             fa_patches_nii = fa_patches
#             fa_nonum_nii_path = x_fa_dir + '/x_fa-data_' + str(all_patch_num) + '.nii.gz'
#             fa_patches_nii = nib.Nifti1Image(fa_patches_nii, img_fa_affine)
#             nib.save(fa_patches_nii, fa_nonum_nii_path)
#             # x_dec data
#             dec_patches_nii = dec_patches
#             dec_nonum_nii_path = x_dec_dir + '/x_dec-data_' + str(all_patch_num) + '.nii.gz'
#             dec_patches_nii = nib.Nifti1Image(dec_patches_nii, img_dec_affine)
#             nib.save(dec_patches_nii, dec_nonum_nii_path)
#             # x_fa data
#             peaks_patches_nii = peaks_patches
#             peaks_nonum_nii_path = x_peaks_dir + '/x_peaks-data_' + str(all_patch_num) + '.nii.gz'
#             peaks_patches_nii = nib.Nifti1Image(peaks_patches_nii, img_peaks_affine)
#             nib.save(peaks_patches_nii, peaks_nonum_nii_path)
#             # y_data
#             label1_patches_nii = label1_patches
#             label1_nonum_nii_path = y_dir_1 + '/y1-data_' + str(all_patch_num) + '.nii.gz'
#             label1_patches_nii = nib.Nifti1Image(label1_patches_nii, img_label_affine1)
#             nib.save(label1_patches_nii, label1_nonum_nii_path)
#             # y_data
#             label2_patches_nii = label2_patches
#             label2_nonum_nii_path = y_dir_2 + '/y2-data_' + str(all_patch_num) + '.nii.gz'
#             label2_patches_nii = nib.Nifti1Image(label2_patches_nii, img_label_affine2)
#             nib.save(label2_patches_nii, label2_nonum_nii_path)
#             # y_data
#             label3_patches_nii = label3_patches
#             label3_nonum_nii_path = y_dir_3 + '/y3-data_' + str(all_patch_num) + '.nii.gz'
#             label3_patches_nii = nib.Nifti1Image(label3_patches_nii, img_label_affine3)
#             nib.save(label3_patches_nii, label3_nonum_nii_path)
#             # y_data
#             label4_patches_nii = label4_patches
#             label4_nonum_nii_path = y_dir_4 + '/y4-data_' + str(all_patch_num) + '.nii.gz'
#             label4_patches_nii = nib.Nifti1Image(label4_patches_nii, img_label_affine4)
#             nib.save(label4_patches_nii, label4_nonum_nii_path)
#             all_patch_num += 1
#         ### --------------
#         ### --------------
#     j += 1
#     print('-' * 30)

if __name__ == '__main__':
    # if 'CN_mydata' not in os.listdir(os.curdir):
    #     os.mkdir('CN_mydata')
    ####################################################################################################################
    # 1. create my train data: ###
    train_save_path = 'CN_mydata/fold5/train_data'
    trainfile_name = 'train_data'
    if trainfile_name not in os.listdir('CN_mydata/fold5'):
        os.mkdir(os.path.join('CN_mydata/fold5', trainfile_name))

    # if 'x_EC_t1_data' not in os.listdir('ON_mydata/'+ trainfile_name):
    #     os.mkdir(os.path.join('ON_mydata/' + trainfile_name, 'x_EC_t1_data'))
    if 'x_t1_data' not in os.listdir('CN_mydata/fold5/' + trainfile_name):
        os.mkdir(os.path.join('CN_mydata/fold5/' + trainfile_name, 'x_t1_data'))
    if 'x_t2_data' not in os.listdir('CN_mydata/fold5/' + trainfile_name):
        os.mkdir(os.path.join('CN_mydata/fold5/' + trainfile_name, 'x_t2_data'))
    if 'x_fa_data' not in os.listdir('CN_mydata/fold5/' + trainfile_name):
        os.mkdir(os.path.join('CN_mydata/fold5/' + trainfile_name, 'x_fa_data'))
    if 'x_dec_data' not in os.listdir('CN_mydata/fold5/' + trainfile_name):
        os.mkdir(os.path.join('CN_mydata/fold5/' + trainfile_name, 'x_dec_data'))
    if 'x_peaks_data' not in os.listdir('CN_mydata/fold5/' + trainfile_name):
        os.mkdir(os.path.join('CN_mydata/fold5/' + trainfile_name, 'x_peaks_data'))
    if 'y_data_on' not in os.listdir('CN_mydata/fold5/' + trainfile_name):
        os.mkdir(os.path.join('CN_mydata/fold5/' + trainfile_name, 'y_data_on'))
    if 'y_data_tgn' not in os.listdir('CN_mydata/fold5/' + trainfile_name):
        os.mkdir(os.path.join('CN_mydata/fold5/' + trainfile_name, 'y_data_tgn'))
    if 'y_data_fvn' not in os.listdir('CN_mydata/fold5/' + trainfile_name):
        os.mkdir(os.path.join('CN_mydata/fold5/' + trainfile_name, 'y_data_fvn'))
    if 'y_data_ocn' not in os.listdir('CN_mydata/fold5/' + trainfile_name):
        os.mkdir(os.path.join('CN_mydata/fold5/' + trainfile_name, 'y_data_ocn'))
    create_train_data_T1FA(train_imgs_path, train_save_path)
    #####################################################################################################################
    # 2. create my val data: ###
    test_save_path = 'CN_mydata/fold5/test_val_data'
    testfile_name = 'test_val_data'    # test  ...
    if testfile_name not in os.listdir('CN_mydata/fold5'):
        os.mkdir(os.path.join('CN_mydata/fold5', testfile_name))
    test_dir = os.listdir(val_imgs_path)
    for test_num in test_dir:
        test_name = 'test_' + test_num
        test_inputdata_save_path = os.path.join(test_save_path, test_name)
        os.mkdir(test_inputdata_save_path)
        if 'x_t1_data' not in os.listdir(test_inputdata_save_path):
            os.mkdir(os.path.join(test_inputdata_save_path, 'x_t1_data'))
        if 'x_t2_data' not in os.listdir(test_inputdata_save_path):
            os.mkdir(os.path.join(test_inputdata_save_path, 'x_t2_data'))
        if 'x_fa_data' not in os.listdir(test_inputdata_save_path):
            os.mkdir(os.path.join(test_inputdata_save_path, 'x_fa_data'))
        if 'x_dec_data' not in os.listdir(test_inputdata_save_path):
            os.mkdir(os.path.join(test_inputdata_save_path, 'x_dec_data'))
        if 'x_peaks_data' not in os.listdir(test_inputdata_save_path):
            os.mkdir(os.path.join(test_inputdata_save_path, 'x_peaks_data'))
        if 'y_data_on' not in os.listdir(test_inputdata_save_path):
            os.mkdir(os.path.join(test_inputdata_save_path, 'y_data_on'))
        if 'y_data_tgn' not in os.listdir(test_inputdata_save_path):
            os.mkdir(os.path.join(test_inputdata_save_path, 'y_data_tgn'))
        if 'y_data_fvn' not in os.listdir(test_inputdata_save_path):
            os.mkdir(os.path.join(test_inputdata_save_path, 'y_data_fvn'))
        if 'y_data_ocn' not in os.listdir(test_inputdata_save_path):
            os.mkdir(os.path.join(test_inputdata_save_path, 'y_data_ocn'))
        create_val_data_T1FA(test_num, test_inputdata_save_path)
    ######################################################################################################################
    # #3. creatr my test data(one by one): ### without data partition
    # test_save_path = 'CN_mydata/test_data'
    # testfile_name = 'test_data'    # test  ...
    # if testfile_name not in os.listdir('CN_mydata'):
    #     os.mkdir(os.path.join('CN_mydata', testfile_name))
    # test_dir = os.listdir(test_imgs_path)
    # for test_num in test_dir:
    #     test_name = 'test_' + test_num
    #     test_inputdata_save_path = os.path.join(test_save_path, test_name)
    #     os.mkdir(test_inputdata_save_path)
    #     if 'x_t1_data' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'x_t1_data'))
    #     if 'x_t2_data' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'x_t2_data'))
    #     if 'x_fa_data' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'x_fa_data'))
    #     if 'x_dec_data' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'x_dec_data'))
    #     if 'x_peaks_data' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'x_peaks_data'))
    #     if 'y_data_on' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'y_data_on'))
    #     if 'y_data_tgn' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'y_data_tgn'))
    #     if 'y_data_fvn' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'y_data_fvn'))
    #     if 'y_data_ocn' not in os.listdir(test_inputdata_save_path):
    #         os.mkdir(os.path.join(test_inputdata_save_path, 'y_data_ocn'))
    #     create_test_data_T1FA(test_num, test_inputdata_save_path)
