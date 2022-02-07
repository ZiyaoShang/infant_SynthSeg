from argparse import ArgumentParser

from ext.lab2im import utils
import os
import sys
import numpy as np
from pathlib import Path

synthseg_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
sys.path.append(synthseg_home)
path_label_list = os.path.join(synthseg_home, 'data/labels_classes_priors/SynthSeg_segmentation_labels.npy')
labels = np.load(path_label_list)

parser = ArgumentParser()
parser.add_argument("t1_dir", type=str, help="t1 segmentation directory")
parser.add_argument("t1_post", type=str, help="t1 posterior probabilities")
parser.add_argument("t2_dir", type=str, help="t2 segmentation directory")
parser.add_argument("t2_post", type=str, help="t2 posterior directory")
parser.add_argument("out_dir", type=str, help="directory to write merged segmentation")
args = parser.parse_args()

subject_ids = [path.split("/")[-1].split("_")[0] for path in os.listdir(args.t1_post)]
for sub in subject_ids:
    print("Merging {} ...".format(sub))
    t1_path = os.path.join(args.t1_dir, sub + "_T1w_brain_seg.nii.gz")
    t2_path = os.path.join(args.t2_dir, sub + "_T2w_brain_seg.nii.gz")
    t1_post = os.path.join(args.t1_post, sub + "_T1w_brain_posteriors.nii.gz")
    t2_post = os.path.join(args.t2_post, sub + "_T2w_brain_posteriors.nii.gz")
    t1_seg, shape, aff, n_dims, n_channels, header, im_res = utils.get_volume_info(t1_path, return_volume=True)
    t2_seg, _, _, _, _, _, _ = utils.get_volume_info(t2_path, return_volume=True)
    t1_post_seg, _, _, _, _, _, _ = utils.get_volume_info(t1_post, return_volume=True)
    t2_post_seg, _, _, _, _, _, _ = utils.get_volume_info(t2_post, return_volume=True)
    merged_seg = np.where(np.amax(t1_post_seg, axis=-1) > np.amax(t2_post_seg, axis=-1), t1_seg, t2_seg)
    avg_seg = labels[np.argmax(t2_post_seg + t1_post_seg, axis=-1)]
    joint_seg = labels[np.argmax(t2_post_seg * t1_post_seg, axis=-1)]

    # Create directory if not exists
    Path(os.path.join(args.out_dir, "max")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.out_dir, "avg")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.out_dir, "joint")).mkdir(parents=True, exist_ok=True)
    utils.save_volume(merged_seg.astype('int'), aff, header, os.path.join(args.out_dir, "max", sub + "_merged_seg.nii.gz"))
    utils.save_volume(avg_seg.astype('int'), aff, header, os.path.join(args.out_dir, "avg", sub + "_merged_seg.nii.gz"))
    utils.save_volume(avg_seg.astype('int'), aff, header, os.path.join(args.out_dir, "joint", sub + "_merged_seg.nii.gz"))
