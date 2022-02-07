from argparse import ArgumentParser

from ext.lab2im import utils
import os
import sys
import numpy as np
from pathlib import Path

# synthseg_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
# sys.path.append(synthseg_home)
# path_label_list = os.path.join(synthseg_home, 'data/labels_classes_priors/SynthSeg_segmentation_labels.npy')
# remember to add the white matter labels if the segmentations have not been evaluated

parser = ArgumentParser()
parser.add_argument("t1_dir", type=str, help="t1 segmentation directory")
parser.add_argument("t1_post", type=str, help="t1 posterior probabilities")
parser.add_argument("t2_dir", type=str, help="t2 segmentation directory")
parser.add_argument("t2_post", type=str, help="t2 posterior directory")
parser.add_argument("out_dir", type=str, help="directory to write merged segmentation")
parser.add_argument("out_post", type=str, help="directory to write merged posterior")
args = parser.parse_args()


path_segT1 = utils.list_images_in_folder(args.t1_dir)
path_segT2 = utils.list_images_in_folder(args.t2_dir)
path_postT1 = utils.list_images_in_folder(args.t1_post)
path_postT2 = utils.list_images_in_folder(args.t2_post)

assert len(path_postT1) == len(path_postT2) == len(path_segT1) == len(path_segT2), "Dirs have different length."

for sub in range(len(path_postT1)):
    print("Merging #" + str(sub))
    print(str(path_segT1[sub]))
    t1_seg, shape, aff, n_dims, n_channels, header, im_res = utils.get_volume_info(path_segT1[sub], return_volume=True)
    t2_seg, _, _, _, _, _, _ = utils.get_volume_info(path_segT2[sub], return_volume=True)
    t1_post_seg, _, _, _, _, _, _ = utils.get_volume_info(path_postT1[sub], return_volume=True)
    t2_post_seg, _, _, _, _, _, _ = utils.get_volume_info(path_postT2[sub], return_volume=True)

    post_name = path_segT1[sub].split("/")[-1].split("_")[0] + "_merged_posterior.nii.gz"
    seg_name = path_segT1[sub].split("/")[-1].split("_")[0] + "_merged_seg.nii.gz"
    print(os.path.join(args.out_post, post_name))
    print(os.path.join(args.out_dir, seg_name))

    merged_post = np.where(t1_post_seg > t2_post_seg, t1_post_seg, t2_post_seg)
    utils.save_volume(merged_post.astype('double'), aff, header, os.path.join(args.out_post, post_name))

    merged_seg = np.where(np.amax(t1_post_seg, axis=-1) > np.amax(t2_post_seg, axis=-1), t1_seg, t2_seg)
    utils.save_volume(merged_seg.astype('int'), aff, header, os.path.join(args.out_dir, seg_name))



