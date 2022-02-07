"""This script enables to launch predictions with SynthSeg from the terminal."""

# print information
from ext.lab2im import utils

print('\n')
print('SynthSeg evaluation for multimodal where WM merging is required')
print('\n')

# python imports
import os
import sys
import numpy as np
from argparse import ArgumentParser
from SynthSeg.evaluate import dice_evaluation
import pandas as pd

synthseg_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
sys.path.append(synthseg_home)
from SynthSeg.predict import predict


# parse arguments
parser = ArgumentParser()
parser.add_argument("seg_dir", type=str, help="path where to save the segmentations. Must be the same type "
                                                         "as path_images (path to a single image or to a folder)")
parser.add_argument("gt_dir", type=str, help="path to ground truth segmentation")
parser.add_argument("--out_dir", type=str, help="path where modified images and dice score will be saved")
parser.add_argument("--modify", type=bool, default=False)
args = vars(parser.parse_args())
args['path_label_list'] = [0, 14, 15, 16, 172, 2, 3, 4, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41, 42, 43, 47, 49, 50, 51, 52, 53, 54, 58, 60]
path_segs = utils.list_images_in_folder(args["seg_dir"])
if args['modify']:
    for path in path_segs:
        image = utils.load_volume(path)
        image = np.where(image == 21, 2, image)
        image = np.where(image == 61, 41, image)
        utils.save_volume(image.astype(int), None, None, path=os.path.join(args["out_dir"], path.split("/")[-1]))

out_dir = args["out_dir"] if args['modify'] else args["seg_dir"]
dice_score = dice_evaluation(gt_dir=args["gt_dir"],
                seg_dir=out_dir,
                path_label_list=args["path_label_list"],
                path_result_dice_array=os.path.join(out_dir, "dice.npy"),
                verbose=True)
print(dice_score)
print(dice_score[:, 10:].mean(axis=1)[:, np.newaxis])
result = pd.DataFrame(np.concatenate([dice_score[:, 10:], dice_score[:, 10:].mean(axis=1)[:, np.newaxis]], axis=1).T, columns=args["path_label_list"])
result.to_csv(os.path.join(out_dir, "dice.csv"))