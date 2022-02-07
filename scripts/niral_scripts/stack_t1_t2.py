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
parser.add_argument("t2_dir", type=str, help="t2 segmentation directory")
parser.add_argument("out_dir", type=str, help="directory to write stacked segmentation")
args = parser.parse_args()

subject_ids = [path.split("/")[-1].split("_")[0] for path in os.listdir(args.t1_dir)]
for sub in subject_ids:
    print("Stacking {} ...".format(sub))
    t1_path = os.path.join(args.t1_dir, sub + "_T1w_brain.nii.gz")
    t2_path = os.path.join(args.t2_dir, sub + "_T2w_brain.nii.gz")
    t1, shape_t1, aff, n_dims, n_channels, header, im_res = utils.get_volume_info(t1_path, return_volume=True)
    t2, shape_t2, _, _, _, _, _ = utils.get_volume_info(t2_path, return_volume=True)
    assert shape_t1 == shape_t2, "Shapes of t1 and t2 must match"
    stacked_t1_t2 = np.stack([t1, t2], axis=-1)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    utils.save_volume(stacked_t1_t2.astype('int'), aff, header,
                      os.path.join(args.out_dir, sub + "_stacked_t1_t2.nii.gz"))