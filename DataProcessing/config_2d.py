import argparse
VOLUME_ROWS = 128
VOLUME_COLS = 160
VOLUME_DEPS = 128

NUM_CLASSES = 2

PATCH_SIZE_W = 128 # 裁剪的尺寸和输入网络的图像尺寸
PATCH_SIZE_H = 160

BATCH_SIZE = 64 # 一次输入多少图像进入网络
NUM_EPOCHS = 500



TRAIN_EXTRACTION_STEP = 12                 # 创建训练集提取的步长
TEST_EXTRACTION_STEP = 1      # 创建测试集提取的步长

# 路径设置

train_imgs_path = r'D:\TGN_AVP_FVN\CN\data_net_128_160_128_int16_0330\TrainSet'
val_imgs_path   = r'D:\TGN_AVP_FVN\CN\data_net_128_160_128_int16_0330\ValSet'
test_imgs_path  = r'D:\TGN_AVP_FVN\CN\data_net_128_160_128_int16_0330\TestSet'



MODEL_type = 2
if MODEL_type == 1:
    MODEL_name = 'model_ETC'      # 2D U-Net  for EC
elif MODEL_type == 2:
    MODEL_name = 'model_CTL'      # 2D Ournet for CL



# 是否选用多块GPU
FLAG_GPU = 1