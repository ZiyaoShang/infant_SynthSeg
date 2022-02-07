# This script shows typical parameters used to train SynthSeg:
# A Learning Strategy for Contrast-agnostic MRI Segmentation, MIDL 2020
# Benjamin Billot, Douglas N. Greve, Koen Van Leemput, Bruce Fischl, Juan Eugenio Iglesias*, Adrian V. Dalca*
# *contributed equally


# project imports
from SynthSeg.training import training
from scripts.niral_scripts.argument_parser import get_argparser
from tensorflow import keras
import numpy as np
import os
import sys

synthseg_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
parser = get_argparser()
args = parser.parse_args()
print(args)
# path training label maps
path_training_label_maps = args.label_dir
# path of directory where to save the models during training
path_model_dir = args.model_dir

# set path to generation labels (i.e. the structure to represent in the synthesised scans)
generation_labels =  [0, 14, 15, 16, 172, 2, 3, 4, 8, 10, 11, 12, 13, 17, 18, 21, 26, 28, 41, 42, 43, 47, 49, 50, 51, 52, 53, 54, 58, 60, 61]
# set path to segmentation labels (i.e. the ROI to segment and to compute the loss on)
segmentation_labels = [0, 14, 15, 16, 172, 2, 3, 4, 8, 10, 11, 12, 13, 17, 18, 21, 26, 28, 41, 42, 43, 47, 49, 50, 51, 52, 53, 54, 58, 60, 61]


# generation parameters
target_res = None  # resolution of the output segmentation
output_shape = 160  # tune this to the size of your GPU

# training parameters
wl2_epochs = 1
dice_epochs = 100
steps_per_epoch = 5000
include_background = True

training(labels_dir=path_training_label_maps,
         model_dir=path_model_dir,
         path_generation_labels=generation_labels,
         path_segmentation_labels=segmentation_labels,
         target_res=target_res,
         output_shape=output_shape,
         wl2_epochs=wl2_epochs,
         dice_epochs=dice_epochs,
         steps_per_epoch=steps_per_epoch,
         include_background=include_background)
