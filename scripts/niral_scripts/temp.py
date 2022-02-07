# import SimpleITK as sitk
import os

# dir = r"C:\Users\zs\ziyao_data\test_EP"
# save = r"C:\Users\zs\ziyao_data\test_EP\EP_nii"
#
# for file in os.listdir(dir):
#     if 'nrrd' in file:
#         path = os.path.join(dir, file)
#         img = sitk.ReadImage(path)
#         print(os.path.join(save, file.split('.')[0] + ".nii.gz"))
#         sitk.WriteImage(img, os.path.join(save, file.split('.')[0] + ".nii.gz"))
#
#
# img = sitk.ReadImage("your_image.nrrd")
# sitk.WriteImage(img, "your_image.nii.gz")


from ext.lab2im import utils, edit_volumes
import os
import sys
import numpy as np
masks = utils.list_images_in_folder(r"/home/ziyaos/ziyao_data/test_EP/mask")
T1s = utils.list_images_in_folder(r"/home/ziyaos/ziyao_data/test_EP/T1")
T2s = utils.list_images_in_folder(r"/home/ziyaos/ziyao_data/test_EP/T2")

for i in range(len(masks)):
    mask, shape, aff, n_dims, n_channels, header, im_res = utils.get_volume_info(masks[i], return_volume=True)
    T1, shape, aff, n_dims, n_channels, header, im_res = utils.get_volume_info(T1s[i], return_volume=True)
    T2, shape, aff, n_dims, n_channels, header, im_res = utils.get_volume_info(T2s[i], return_volume=True)
    T1 = np.where(mask, T1, 0)
    T2 = np.where(mask, T2, 0)
    save1 = r"/home/ziyaos/ziyao_data/test_EP/masked/T1/" + T1s[i].split('/')[-1]
    save2 = r"/home/ziyaos/ziyao_data/test_EP/masked/T2/" + T2s[i].split('/')[-1]

    utils.save_volume(T1.astype('float'), aff, header, save1)
    utils.save_volume(T2.astype('float'), aff, header, save2)
    # print(T1.shape)