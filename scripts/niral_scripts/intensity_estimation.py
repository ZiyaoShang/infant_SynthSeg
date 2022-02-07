"""Examples to show how to estimate of the hyperparameters governing the GMM prior distributions.
We do not provide example images and associated label maps, so do not try to run this directly !"""

from SynthSeg.estimate_priors import build_intensity_stats
import numpy as np

# -------------------------------------  multi-modal images with separate channels -------------------------------------

# Here we have multi-modal images, where the different channels are stored in separate directories.
# We provide the these different directories as a list.
list_image_dir = ['/home/turja/DCAN_T1_train', '/home/turja/DCAN_T2_train']
# In this example, we assume that channels are registered and at the same resolutions.
# Therefore we can use the same label maps for all channels.
labels_dir = '/home/turja/DCAN_roi3_WM'

# same as before
estimation_labels = [0, 2, 3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 26, 28, 41, 42, 43, 47, 49, 50, 51, 52, 53, 54, 58, 60, 61, 172]
estimation_classes = None
result_dir = '/home/turja/SynthSeg/data/estimated_PV-SynthSeg_priors'

build_intensity_stats(list_image_dir=list_image_dir,
                      list_labels_dir=labels_dir,
                      estimation_labels=estimation_labels,
                      estimation_classes=estimation_classes,
                      result_dir=result_dir,
                      rescale=True)
