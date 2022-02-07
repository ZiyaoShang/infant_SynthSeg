"""Examples to show how to estimate of the hyperparameters governing the GMM prior distributions.
We do not provide example images and associated label maps, so do not try to run this directly !"""

from SynthSeg.estimate_priors import build_intensity_stats
import numpy as np
from ext.lab2im import utils

# -------------------------------------  multi-modal images with separate channels -------------------------------------

# Here we have multi-modal images, where the different channels are stored in separate directories.
# We provide the these different directories as a list.
list_image_dir = '/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/final/T1s'
# In this example, we assume that channels are registered and at the same resolutions.
# Therefore we can use the same label maps for all channels.
labels_dir = '/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/final/train_labels'

# same as before
estimation_labels = [0, 14, 15, 16, 24, 77, 85, 170, 172, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 21, 26, 28, 31, 41, 42, 43, 44, 46,
                      47, 49, 50, 51, 52, 53, 54, 58, 60, 61, 63]
#
# estimation_labels = [0, 7, 46]

estimation_labels, _ = utils.get_list_labels(label_list=estimation_labels, save_label_list=None,
                                             FS_sort=True)
print("The estimation labels are: " + str(estimation_labels))
print("with shape " + str(estimation_labels.shape))
estimation_classes = None
# I only changed the result_dir
result_dir = '/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/test_junk'

build_intensity_stats(list_image_dir=list_image_dir,
                      list_labels_dir=labels_dir,
                      estimation_labels=estimation_labels,
                      estimation_classes=estimation_classes,
                      result_dir=result_dir,
                      rescale=True)
