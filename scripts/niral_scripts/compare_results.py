import pandas as pd
import numpy as np
import os
import sys

synthseg_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
sys.path.append(synthseg_home)
path_label_list = os.path.join(synthseg_home, 'data/labels_classes_priors/SynthSeg_segmentation_labels.npy')
labels = np.load(path_label_list)

t1_dice = pd.read_csv("/home/turja/dice_scores_t1.csv", index_col=0).to_numpy()
t2_dice = pd.read_csv("/home/turja/dice_scores_t2.csv", index_col=0).to_numpy()
merged_dice = pd.read_csv("/home/turja/dice_scores_max.csv", index_col=0).to_numpy()
avg_dice = pd.read_csv("/home/turja/dice_scores_avg.csv", index_col=0).to_numpy()
joint_dice = pd.read_csv("/home/turja/dice_scores_joint.csv", index_col=0).to_numpy()
print(t1_dice.mean(axis=0))
print(t2_dice.mean(axis=0))
print(merged_dice.mean(axis=0))
print(avg_dice.mean(axis=0))
wr_res = np.stack([t1_dice.mean(axis=0), t2_dice.mean(axis=0), merged_dice.mean(axis=0), avg_dice.mean(axis=0), joint_dice.mean(axis=0)], axis=1)
wr_res = pd.DataFrame(wr_res, columns=["T1w", "T2w", "Max Posterior", "Average Posterior", "Joint Posterior"], index=labels)
wr_res = wr_res.round(4)
wr_res.to_csv("Result_Summary_dice.csv")
print("Done !!!")