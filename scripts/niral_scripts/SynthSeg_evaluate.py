from argparse import ArgumentParser

from SynthSeg.evaluate import dice_evaluation
import os
import sys
import numpy as np
import pandas as pd
from ext.lab2im import utils

synthseg_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
sys.path.append(synthseg_home)

# parse arguments
parser = ArgumentParser()
parser.add_argument("gtdir", type=str, help="Ground truth label maps")
parser.add_argument("preddir", type=str, help="Predicted label maps")
parser.add_argument("outdir", type=str)
args = parser.parse_args()
path_label_list = os.path.join(synthseg_home, 'data/labels_classes_priors/SynthSeg_segmentation_labels.npy')

path_segs = utils.list_images_in_folder(args.preddir)
subnames = [p.split("/")[-1].split("_")[0] for p in path_segs]
print(subnames)
scores = dice_evaluation(args.gtdir, args.preddir, path_label_list, args.outdir, verbose=True)
labels = np.load(path_label_list)

scores_pd = pd.DataFrame(scores.T, columns=labels, index=subnames)
scores_pd.to_csv(args.outdir + ".csv")