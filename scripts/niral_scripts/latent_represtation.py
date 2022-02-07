"""This script enables to launch predictions with SynthSeg from the terminal."""

# print information
print('\n')
print('SynthSeg prediction')
print('\n')

# python imports
import os
import sys
import numpy as np
from argparse import ArgumentParser

# add main folder to python path and import ./SynthSeg/predict.py
synthseg_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
sys.path.append(synthseg_home)
from predict import predict


# parse arguments
parser = ArgumentParser()
parser.add_argument("path_images", type=str, help="images to segment. Can be the path to a single image or to a folder")
parser.add_argument("path_segmentations", type=str, help="path where to save the segmentations. Must be the same type "
                                                         "as path_images (path to a single image or to a folder)")
parser.add_argument("path_model", type=str, help="Trained model")
parser.add_argument("--out_posteriors", type=str, default=None, dest="path_posteriors",
                    help="path where to save the posteriors. Must be the same type as path_images (path to a single "
                         "image or to a folder)")
parser.add_argument("--out_volumes", type=str, default=None, dest="path_volumes",
                    help="path to a csv file where to save the volumes of all ROIs for all patients")
args = vars(parser.parse_args())

# default parameters
path_label_list = os.path.join(synthseg_home, 'data/labels_classes_priors/SynthSeg_segmentation_labels.npy')
args['segmentation_label_list'] = path_label_list
# args['segmentation_label_list'] = [0, 14, 15, 16, 172, 2, 3, 4, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41, 42, 43, 47, 49, 50, 51, 52, 53, 54, 58, 60]

# args['segmentation_label_list'] = [0, 2, 3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28, 41, 42, 43, 47, 49, 50, 51, 52, 53, 54, 58, 60]
args['sigma_smoothing'] = 0
args['keep_biggest_component'] = True
args['aff_ref'] = 'FS'
# args['gt_folder'] = '/home/turja/DCAN_roi'
args['activation'] = 'elu'
args['layer_name'] = 'unet_conv_uparm_8_1'
# args['evaluation_label_list'] = [2, 3, 41, 42]
# call predict
predict(**args)
