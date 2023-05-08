import os,shutil

def nii_group(input_t1, input_fa, input_label1, input_label2,input_mask, input_peaks, output_group):
    t1s = os.listdir(input_t1)
    fas = os.listdir(input_fa)
    labelsTGN = os.listdir(input_label1)
    labelsON = os.listdir(input_label2)
    masks = os.listdir(input_mask)
    peaks = os.listdir(input_peaks)
    groups = os.listdir(output_group)
    for t1 in t1s:
        num = t1.replace('_CN-T1.nii.gz', '')
        if num not in os.listdir(output_group):
            os.mkdir(os.path.join(output_group, num))
        for group in groups:
            if num == group:
                shutil.copy(input_t1 + t1, output_group + group)
    for fa in fas:
        num = fa.replace('_CN-FA.nii.gz', '')
        if num not in os.listdir(output_group):
            os.mkdir(os.path.join(output_group, num))
        for group in groups:
            if num == group:
                shutil.copy(input_fa + fa, output_group + group)
    for label in labelsTGN:
        num = label.replace('_TGN-label.nii.gz', '')
        if num not in os.listdir(output_group):
            os.mkdir(os.path.join(output_group, num))
        for group in groups:
            if num == group:
                shutil.copy(input_label1 + label, output_group + group)
    for label in labelsON:
        num = label.replace('_ON-label.nii.gz', '')
        if num not in os.listdir(output_group):
            os.mkdir(os.path.join(output_group, num))
        for group in groups:
            if num == group:
                shutil.copy(input_label2 + label, output_group + group)
    for mask in masks:
        num = mask.replace('_CN_mask.nii.gz', '')
        if num not in os.listdir(output_group):
            os.mkdir(os.path.join(output_group, num))
        for group in groups:
            if num == group:
                shutil.copy(input_mask + mask, output_group + group)
    for peak in peaks:
        num = peak.replace('_CN_Peaks.nii.gz', '')
        if num not in os.listdir(output_group):
            os.mkdir(os.path.join(output_group, num))
        for group in groups:
            if num == group:
                shutil.copy(input_peaks + peak, output_group + group)


def nii_FA(nii_FA,output_PATH):
    ###复制某目录下的后缀文件
    FA = os.listdir(nii_FA)
    PATH = os.listdir(output_PATH)
    for FA_NUM in FA:
        for NUM in PATH:
            FA_num = FA_NUM.replace('_ON-new_T1_1.nii.gz', '')
            if FA_num == NUM:
                shutil.copy(nii_FA + FA_num + '_ON-new_T1_1.nii.gz', output_PATH + NUM)


def nii_FA2(input_path, output_PATH):
    ###复制某目录下的目录下的后缀文件
    input = os.listdir(input_path)

    for NUM in input:
        num_path = os.path.join(input_path, NUM)
        for root, dirs, files in os.walk(num_path, topdown=False):
            for name in files:
                if name.endswith('_ON-new_mask.nii.gz',):
                    shutil.copy(os.path.join(root, name), output_PATH + name)


def delete_FA(input_path):
    ###删除特定后缀文件
    input = os.listdir(input_path)
    for NUM in input:
        num_path = os.path.join(input_path, NUM)
        for root, dirs, files in os.walk(num_path, topdown=False):
            for name in files:
                if name.endswith('_ON-source.nii.gz',):
                    os.remove(os.path.join(root, name))


def change_name(input_path):
    ###重命名某一目录下的目录下的后缀
    input = os.listdir(input_path)
    for NUM in input:
        num_path = os.path.join(input_path, NUM)
        for root, dirs, files in os.walk(num_path, topdown=False):
            for name in files:
                if name.endswith('_ON-new_100T1_100FA.nii.nii.gz',):
                    new_name = os.path.join(root, NUM + '_ON-new_100T1_100FA.nii.gz')
                    os.rename(os.path.join(root, name), new_name)


def change_name2(input_path):
    ###重命名某一目录下的后缀
    num_path = os.path.join(input_path)
    for root, dirs, files in os.walk(num_path, topdown=False):
        for name in files:
            if name.endswith('_ON-source.nii.gz',):
                num = name.replace('_ON-source.nii.gz', '')
                new_name = os.path.join(root, num + '_ON-T1.nii.gz')
                os.rename(os.path.join(root, name), new_name)

def nii_peak(input_t1, input_fa, input_label1, input_label2, input_label3, input_label4,input_mask, input_peaks, output_group):
    ###复制某目录下的后缀文件
    fas = os.listdir(input_fa)

    # t1s = os.listdir(input_t1)
    # fas = os.listdir(input_fa)
    # labelsTGN = os.listdir(input_label1)
    # labelsON = os.listdir(input_label2)
    # masks = os.listdir(input_mask)
    # peaks = os.listdir(input_peaks)
    # groups = os.listdir(output_group)

    # PATH = os.listdir(output_PATH)
    for splitdata in fas:
        num = splitdata.split('_')[0]
        os.makedirs(output_group +'/'+ num)
        # peaksnum = peak_NUM.split('_')[0]
        shutil.copy(input_t1 + '/' + num + '_CN-T1.nii.gz', output_group+'/' + num+'/'+ num + '_CN-T1.nii.gz')
        shutil.copy(input_fa + '/' + num + '_CN-FA.nii.gz', output_group +'/'+ num+ '/' + num + '_CN-FA.nii.gz')
        shutil.copy(input_label1 + '/' + num + '_TGN-label.nii.gz', output_group+'/' + num+ '/' + num + '_TGN-label.nii.gz')
        shutil.copy(input_label2 + '/' + num + '_ON-label.nii.gz', output_group+'/' + num+ '/' + num+ '_ON-label.nii.gz')
        shutil.copy(input_label3 + '/' + num + '_FVN-label.nii.gz',
                    output_group + '/' + num + '/' + num + '_FVN-label.nii.gz')
        shutil.copy(input_label4 + '/' + num + '_OCN-label.nii.gz',
                    output_group + '/' + num + '/' + num + '_OCN-label.nii.gz')
        shutil.copy(input_mask + '/' + num + '_CN-mask.nii.gz', output_group+'/' + num+ '/' + num + '_CN-mask.nii.gz')
        shutil.copy(input_peaks + '/' + num + '_CN-Peaks.nii.gz', output_group+'/' + num+ '/' + num + '_CN-Peaks.nii.gz')

def nii_peak1(data1path,data2path,output_PATH):
    ###复制某目录下的后缀文件
    data1 = os.listdir(data1path)

    for splitdata in data1:
        num = splitdata.split('_')[0]
        shutil.copy(data2path + '/'+ num + '_ON-mask.nii.gz', output_PATH +'/'+ num + '_CN-mask.nii.gz')
if __name__ == '__main__':

    ################复制某后缀文件
    # data1 = r'D:\TGN_AVP_FVN\CN\Origdata145_174_145\Label_tgn'
    # data2=r'D:\TGN_AVP_FVN\AVP\145x174x145_102\Mask'
    # output_path = r'D:\TGN_AVP_FVN\CN\Origdata145_174_145\Masks'
    # nii_peak1(data1, data2, output_path)
    #############

    input_t1 = 'D:\TGN_AVP_FVN\CN\zju\T1'
    input_fa = 'D:\TGN_AVP_FVN\CN\zju\FA'
    input_label1 = 'D:\TGN_AVP_FVN\CN\zju\label_tgn'
    input_label2 = 'D:\TGN_AVP_FVN\CN\zju\label_on'
    input_label3 = 'D:\TGN_AVP_FVN\CN\zju\label_fvn'
    input_label4 = 'D:\TGN_AVP_FVN\CN\zju\label_ocn'
    input_mask = 'D:\TGN_AVP_FVN\CN\zju\Masks'
    input_peaks = 'D:\TGN_AVP_FVN\CN\zju\Peaks'
    output_group = 'D:\TGN_AVP_FVN\CN\zju_deal\TrainSet'
    nii_peak(input_t1, input_fa, input_label1, input_label2, input_label3, input_label4, input_mask, input_peaks, output_group)


#################复制某后缀文件
    # FA_path = 'F:\Data\ON_Data\Finish\ON_Data_128x160x128_102\All\Test_Set/'
    # output_path = 'F:\Data\ON_Data\Finish\ON_Data_128x160x128_102/New_mask/'
    # nii_FA2(FA_path, output_path)
##############


##############删除特定后缀文件
    # intput_path = 'H:/ON_Data_145x174/Data_145x174/'
    # delete_FA(intput_path)
    # print('#')
##############

#################重命名后缀
    # intput_path = 'H:/ON_Data_145x174/T1/'
    # change_name2(intput_path)
    # print('#')
##############