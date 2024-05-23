import argparse
VOLUME_ROWS = 128
VOLUME_COLS = 160
VOLUME_DEPS = 128

NUM_CLASSES = 2

PATCH_SIZE_W = 128 # 裁剪的尺寸和输入网络的图像尺寸
PATCH_SIZE_H = 160

BATCH_SIZE = 64 # 一次输入多少图像进入网络
NUM_EPOCHS = 200



# 路径设置

train_imgs_path = '/media/brainplan/XLdata/CNTSeg++/Data/NewDataset/TrainSet'
val_imgs_path = '/media/brainplan/XLdata/CNTSeg++/Data/NewDataset/ValSet'
test_imgs_path = '/media/brainplan/XLdata/CNTSeg++/Data/NewDataset/TestSet'






# 是否选用多块GPU
FLAG_GPU = 1