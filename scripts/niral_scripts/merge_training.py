# This script shows typical parameters used to train PV-SynthSeg:
# A Learning Strategy for Contrast-agnostic MRI Segmentation, MIDL 2020
# Benjamin Billot, Douglas N. Greve, Koen Van Leemput, Bruce Fischl, Juan Eugenio Iglesias*, Adrian V. Dalca*
# *contributed equally


# imports
import numpy as np
from SynthSeg.training_merge_seg import training


# path training label maps
path_training_label_maps ='/home/turja/DCAN_roi'
path_model_dir = '/home/turja/PV-SynthSeg_training_DCAN_2'

# set path to generation labels (i.e. the structure to represent in the synthesised scans)
path_generation_labels = [0, 14, 15, 16, 172, 2, 3, 4, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41, 42, 43, 47, 49, 50, 51, 52, 53, 54, 58, 60]
# set path to segmentation labels (i.e. the ROI to segment and to compute the loss on)
path_segmentation_labels = [0, 14, 15, 16, 172, 2, 3, 4, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41, 42, 43, 47, 49, 50, 51, 52, 53, 54, 58, 60]


# generation parameters
target_res = 1
output_shape = 60 # tune this to the size of your GPU

# acquisition resolution parameters
# the following parameters aim at mimicking data that would have been 1) acquired at low resolution (i.e. data_res),
# and 2) upsampled to high resolution in order to obtain segmentation at high res (see target_res).
# We do not such effects here, as this script shows training parameters to segment data at 1mm isotropic resolution
data_res = None
randomise_res = False
thickness = None
downsample = False
blur_range = 1.03  # we activate this parameter, which enables SynthSeg to be robust against small resolution variations

# architecture parameters
n_channels = 48
n_levels = 2  # number of resolution levels
nb_conv_per_level = 2  # number of convolution per level
conv_size = 3  # size of the convolution kernel (e.g. 3x3x3)
unet_feat_count = 24  # number of feature maps after the first convolution
# if feat_multiplier is set to 1, we will keep the number of feature maps constant throughout the network; 2 will double
# them(resp. half) after each max-pooling (resp. upsampling); 3 will triple them, etc.
feat_multiplier = 2
dropout = 0
activation = 'elu'  # activation for all convolution layers except the last, which will use sofmax regardless

# training parameters
lr = 1e-4  # learning rate
lr_decay = 0
wl2_epochs = 0  # number of pre-training epochs
dice_epochs = 100  # number of training epochs
steps_per_epoch = 5000

training(interm_dir='/home/turja/DCAN_interm',
         gt_seg_dir='/home/turja/DCAN_roi',
         model_dir=path_model_dir,
         path_generation_labels=path_generation_labels,
         path_segmentation_labels=path_segmentation_labels,
         target_res=target_res,
         output_shape=output_shape,
         path_generation_classes=None,
         steps_per_epoch=steps_per_epoch,
         use_specific_stats_for_channel=True,
         n_levels=n_levels,
         nb_conv_per_level=nb_conv_per_level,
         conv_size=conv_size,
         unet_feat_count=unet_feat_count,
         feat_multiplier=feat_multiplier,
         activation=activation,
         lr=lr,
         wl2_epochs=wl2_epochs,
         dice_epochs=dice_epochs,
         mix_prior_and_random=True,
         n_channels=n_channels)
