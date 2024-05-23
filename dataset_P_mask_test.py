import numpy as np
import config_2d
import os
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch

batch_size =config_2d.BATCH_SIZE
patch_size_w = config_2d.PATCH_SIZE_W
patch_size_h = config_2d.PATCH_SIZE_H
n_classes = config_2d.NUM_CLASSES
masks = np.array([[False, False, False, False],[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True],
                      [True, True, False, False], [True, False, True, False], [True, False, False, True], [False, True, True, False], [False, True, False, True], [False, False, True, True], [True, True, True, False], [True, True, False, True], [True, False, True, True], [False, True, True, True],
                      [True, True, True, True]])

def CN_make_dirset(train_x_1_dir, train_x_2_dir, train_x_3_dir,train_x_4_dir,train_x_5_dir, train_y_1_dir, train_y_2_dir, train_y_3_dir, train_y_4_dir):
    train_x_1_path=[]
    train_x_2_path = []
    train_x_3_path = []
    train_x_4_path = []
    train_x_5_path = []
    train_y_1_path=[]
    train_y_2_path=[]
    train_y_3_path = []
    train_y_4_path = []
    n=len(os.listdir(train_x_1_dir))
    for i in range(n):
        img_x_1 = os.path.join(train_x_1_dir, 'x_t1-data_%d.nii.gz'%i)
        train_x_1_path.append(img_x_1)

        img_x_2 = os.path.join(train_x_2_dir, 'x_t2-data_%d.nii.gz'%i)
        train_x_2_path.append(img_x_2)

        img_x_3 = os.path.join(train_x_3_dir, 'x_fa-data_%d.nii.gz' % i)
        train_x_3_path.append(img_x_3)

        img_x_4 = os.path.join(train_x_4_dir, 'x_dec-data_%d.nii.gz' % i)
        train_x_4_path.append(img_x_4)

        img_x_5 = os.path.join(train_x_5_dir, 'x_peaks-data_%d.nii.gz' % i)
        train_x_5_path.append(img_x_5)

        img_y_1 = os.path.join(train_y_1_dir, 'y1-data_%d.nii.gz'%i)
        train_y_1_path.append(img_y_1)
        img_y_2 = os.path.join(train_y_2_dir, 'y2-data_%d.nii.gz' % i)
        train_y_2_path.append(img_y_2)
        img_y_3 = os.path.join(train_y_3_dir, 'y3-data_%d.nii.gz' % i)
        train_y_3_path.append(img_y_3)
        img_y_4 = os.path.join(train_y_4_dir, 'y4-data_%d.nii.gz' % i)
        train_y_4_path.append(img_y_4)
    return train_x_1_path, train_x_2_path, train_x_3_path,train_x_4_path, train_x_5_path, train_y_1_path,train_y_2_path,train_y_3_path,train_y_4_path

class CN_MyTrainDataset(Dataset):
    def __init__(self, train_x_1_dir, train_x_2_dir, train_x_3_dir,train_x_4_dir,train_x_5_dir,train_y_1_dir, train_y_2_dir, train_y_3_dir, train_y_4_dir, mask_idx, x_transform=None, y_transform=None):
        train_x_1_path, train_x_2_path, train_x_3_path, train_x_4_path, train_x_5_path, train_y_1_path, train_y_2_path, train_y_3_path, train_y_4_path = CN_make_dirset(train_x_1_dir, train_x_2_dir, train_x_3_dir,train_x_4_dir,train_x_5_dir, train_y_1_dir, train_y_2_dir, train_y_3_dir, train_y_4_dir)
        self.train_x_1_path = train_x_1_path
        self.train_x_2_path = train_x_2_path
        self.train_x_3_path = train_x_3_path
        self.train_x_4_path = train_x_4_path
        self.train_x_5_path = train_x_5_path
        self.train_y_1_path = train_y_1_path
        self.train_y_2_path = train_y_2_path
        self.train_y_3_path = train_y_3_path
        self.train_y_4_path = train_y_4_path

        self.mask_idx = mask_idx

        self.x_transform = x_transform
        self.y_transform = y_transform

    def __getitem__(self, index):
        x_1_path = self.train_x_1_path[index]
        x_2_path = self.train_x_2_path[index]
        x_3_path = self.train_x_3_path[index]
        x_4_path = self.train_x_4_path[index]
        x_5_path = self.train_x_5_path[index]
        y_1_path = self.train_y_1_path[index]
        y_2_path = self.train_y_2_path[index]
        y_3_path = self.train_y_3_path[index]
        y_4_path = self.train_y_4_path[index]

        img_x_1 = nib.load(x_1_path)
        img_x_1_data = img_x_1.get_data()
        x_1_are_Nans = np.isnan(img_x_1_data)
        img_x_1_data[x_1_are_Nans]=0
        img_x_1_data = np.array(img_x_1_data,dtype='uint8')

        img_x_2 = nib.load(x_2_path)
        img_x_2_data = img_x_2.get_data()
        x_2_are_Nans = np.isnan(img_x_2_data)
        img_x_2_data[x_2_are_Nans]=0
        img_x_2_data = np.array(img_x_2_data,dtype='uint8')

        img_x_3 = nib.load(x_3_path)
        img_x_3_data = img_x_3.get_data()
        x_3_are_Nans = np.isnan(img_x_3_data)
        img_x_3_data[x_3_are_Nans]=0
        img_x_3_data = np.array(img_x_3_data,dtype='uint8')

        img_x_4 = nib.load(x_4_path)
        img_x_4_data = img_x_4.get_data()
        x_4_are_Nans = np.isnan(img_x_4_data)
        img_x_4_data[x_4_are_Nans] = 0
        # img_x_4_data = np.array(img_x_4_data, dtype='uint8')

        img_x_5 = nib.load(x_5_path)
        img_x_5_data = img_x_5.get_data()
        x_5_are_Nans = np.isnan(img_x_5_data)
        img_x_5_data[x_5_are_Nans] = 0
        # img_x_3_data = np.array(img_x_3_data, dtype='uint8')


        img_y_1 = nib.load(y_1_path)
        img_y_1_data = img_y_1.get_data()
        # img_y_1_data = np.array(img_y_1_data, dtype='int8')

        img_y_2 = nib.load(y_2_path)
        img_y_2_data = img_y_2.get_data()
        # img_y_2_data = np.array(img_y_2_data, dtype='int8')

        img_y_3 = nib.load(y_3_path)
        img_y_3_data = img_y_3.get_data()
        # img_y_3_data = np.array(img_y_3_data, dtype='int8')

        img_y_4 = nib.load(y_4_path)
        img_y_4_data = img_y_4.get_data()
        # img_y_4_data = np.array(img_y_4_data, dtype='int8')

        img_y_0 = np.ones(img_y_1_data.shape)
        # print(img_y_0_zero.shape)
        img_y_0_zero = img_y_1_data+img_y_2_data+img_y_3_data+img_y_4_data
        img_y_0[img_y_0_zero>0]=0

        # mask_idx = np.random.choice(9, 1)
        # mask2 = torch.from_numpy(masks[mask_idx])
        mask = torch.squeeze(torch.from_numpy(masks[self.mask_idx]), dim=0)


        if self.x_transform is not None:
            img_x_1_data = self.x_transform(img_x_1_data)
            img_x_2_data = self.x_transform(img_x_2_data)
            img_x_3_data = self.x_transform(img_x_3_data)
            img_x_4_data = self.x_transform(img_x_4_data)
            img_x_5_data = self.x_transform(img_x_5_data)


        if self.y_transform is not None:
            img_y_0 = self.y_transform(img_y_0)
            img_y_1_data = self.y_transform(img_y_1_data)
            img_y_2_data = self.y_transform(img_y_2_data)
            img_y_3_data = self.y_transform(img_y_3_data)
            img_y_4_data = self.y_transform(img_y_4_data)

        return img_x_1_data, img_x_2_data, img_x_3_data,img_x_4_data, img_x_5_data, img_y_0, img_y_1_data, img_y_2_data, img_y_3_data, img_y_4_data,mask

    def __len__(self): # 返回文件内train的nii数量
        return len(self.train_x_1_path)














