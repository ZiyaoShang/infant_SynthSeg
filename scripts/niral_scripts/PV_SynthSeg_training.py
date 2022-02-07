# This script shows typical parameters used to train PV-SynthSeg:
# A Learning Strategy for Contrast-agnostic MRI Segmentation, MIDL 2020
# Benjamin Billot, Douglas N. Greve, Koen Van Leemput, Bruce Fischl, Juan Eugenio Iglesias*, Adrian V. Dalca*
# *contributed equally


# imports
import numpy as np
from SynthSeg.training import training


# path training label maps
path_training_label_maps ='/home/turja/DCAN_roi'
path_model_dir = '/home/turja/PV-SynthSeg_training_DCAN_2'

# set path to generation labels and segmentation labels
generation_labels = '/home/turja/SynthSeg/data/labels_classes_priors/generation_labels_DCAN_left.npy'
# set path to segmentation labels (i.e. the ROI to segment and to compute the loss on)
segmentation_labels = '/home/turja/SynthSeg/data/labels_classes_priors/segmentation_labels_DCAN_left.npy'

# prior distribution of the GMM
generation_classes = None
prior_means = '/home/turja/SynthSeg/data/estimated_PV-SynthSeg_priors/prior_means.npy'  # the same prior will be used for all channels
prior_stds = '/home/turja/SynthSeg/data/estimated_PV-SynthSeg_priors/prior_stds.npy'

# generation parameters
target_res = 1
output_shape = 60 # tune this to the size of your GPU

# training parameters
wl2_epochs = 0
dice_epochs = 100
steps_per_epoch = 1000

training(labels_dir=path_training_label_maps,
         model_dir=path_model_dir,
         path_generation_labels=generation_labels,
         path_segmentation_labels=segmentation_labels,
         target_res=target_res,
         output_shape=output_shape,
         path_generation_classes=generation_classes,
         wl2_epochs=wl2_epochs,
         dice_epochs=dice_epochs,
         steps_per_epoch=steps_per_epoch,
         use_specific_stats_for_channel=True,
         n_channels=2,
         prior_means=prior_means,
         prior_stds=prior_stds,
         mix_prior_and_random=True,
         load_model_file='/home/turja/PV-SynthSeg_training_DCAN/dice_038.h5')
