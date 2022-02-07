# convert new data segmentations into DCAN_rpi labels

from ext.lab2im import utils, edit_volumes
import os
import sys
import numpy as np

path_gt = utils.list_images_in_folder(r"/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/DCAN/T2")
# path_seg = utils.list_images_in_folder("/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/DCAN/merged_seg")
# to_convert = [7, 46]

for image in range(len(path_gt)):
    gt, shape, aff, n_dims, n_channels, header, im_res = utils.get_volume_info(path_gt[image], return_volume=True)
    # seg, shape, aff, n_dims, n_channels, header, im_res = utils.get_volume_info(path_seg[image], return_volume=True)

    # print(path_seg[image])
    print(path_gt[image])
    print(gt.shape)
    # print(seg.shape)
    # assert seg.shape == gt.shape

    # mask = np.zeros(gt.shape)
    # for label in to_convert:
    #     mask = np.logical_or(mask, seg == label)
    #
    # overwritten = np.where(mask, seg, gt)

    padded = edit_volumes.pad_volume(gt, (155, 175, 149))

    # seg[seg == 159] = 21
    # print(np.any(seg == 161))
    # seg[seg == 161] = 2
    # seg[seg == 160] = 61
    # seg[seg == 162] = 41
    #
    # seg[seg == 61] = 41
    # seg[seg == 21] = 2
    # vol, aff = edit_volumes.crop_volume_with_idx(gt, np.array([20, 35, 1, 175, 210, 150]), aff=aff)

    # vol, cropping, aff = edit_volumes.crop_volume_around_region(im, mask=mask, margin=25, aff=aff)
    # print(vol.shape)
    # print(cropping)

    save = "/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/DCAN/T2/" + "pd_" + path_gt[image].split('/')[-1]

    print("save to " + save)
    utils.save_volume(padded.astype('float'), aff, header, save)
