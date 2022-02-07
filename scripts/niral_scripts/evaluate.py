from argparse import ArgumentParser

from SynthSeg.dice_evaluation import dice_evaluation, surface_distances
import os
import sys
import numpy as np
import pandas as pd
from ext.lab2im import utils

# parse arguments
parser = ArgumentParser()
parser.add_argument("gtdir", type=str, help="Ground truth label maps")
parser.add_argument("preddir", type=str, help="Predicted label maps")
parser.add_argument("dicedir", type=str, help="path to save the dice scores file")
parser.add_argument("maxdistdir", type=str, help="path to save the max distances file")
parser.add_argument("meandistdir", type=str, help="path to save the average distances file")

args = vars(parser.parse_args())


def evaluate_WM(gtdir=None,
                preddir=None,
                dicedir=None,
                maxdistdir=None,
                meandistdir=None):
    print(gtdir)
    print("Here is the print: ")
    labels = [0, 14, 15, 16, 172, 2, 3, 4, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41, 42, 43, 47, 49, 50, 51, 52, 53, 54,
              58, 60]

    path_segs = utils.list_images_in_folder(preddir)
    subnames = [p.split("/")[-1].split("_")[0] for p in path_segs]
    print(path_segs)
    # average of dice score all the subjects for each label, make the process a function.
    # result: one value for each label
    dice_score, max_dists, mean_dists = dice_evaluation(gtdir, preddir, labels, verbose=True,
                                                        compute_distances=True)
    print(dice_score)

    subnames.append("average")

    # save the dice scores
    dice_score = dice_score.T
    dice_summary = np.vstack([dice_score, np.average(dice_score, axis=0)])
    scores_pd = pd.DataFrame(dice_summary, columns=labels, index=subnames)
    scores_pd.to_csv(dicedir + ".csv")

    # save the max distance scores
    max_dists = max_dists.T
    max_dists_summary = np.vstack([max_dists, np.average(max_dists, axis=0)])
    scores_pd = pd.DataFrame(max_dists_summary, columns=labels, index=subnames)
    scores_pd.to_csv(maxdistdir + ".csv")

    # save the mean distances
    mean_dists = mean_dists.T
    mean_dists_summary = np.vstack([mean_dists, np.average(mean_dists, axis=0)])
    scores_pd = pd.DataFrame(mean_dists_summary, columns=labels, index=subnames)
    scores_pd.to_csv(meandistdir + ".csv")


evaluate_WM(**args)
