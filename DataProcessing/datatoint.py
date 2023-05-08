
import numpy as np
import nibabel as nib
import os
# data_path = r'D:\TGN_AVP_FVN\CN\new_mask\FVN_MASK\100206_FVN-label.nii.gz'
# save_path = r'D:\TGN_AVP_FVN\CN\new_mask\FVN_MASK\100206_1FVN-label.nii.gz'

def mask_float(data_path,save_path):

    data = nib.load(data_path)
    data_data = data.get_data()
    data_data = np.array(data_data, dtype='int16')
    data_affine = data.get_affine()
    data_save = nib.Nifti1Image(data_data, data_affine)
    nib.save(data_save, save_path)

if __name__ == '__main__':
    data_path = r'D:\TGN_AVP_FVN\CN\Origdata145_174_145_0330\Label_tgn'
    save_path = r'D:\TGN_AVP_FVN\CN\Origdata145_174_145_0330\Label_tgn'
    data = os.listdir(data_path)

    for splitdata in data:
        handledata = data_path+ '/'+ splitdata
        savedata = save_path+'/'+ splitdata
        mask_float(handledata, savedata)
