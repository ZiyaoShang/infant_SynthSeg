import numpy as np


def mergestats(young, old, joint, save):
    y = np.load(young)
    o = np.load(old)
    j = np.load(joint)
    assert y.shape == o.shape == j.shape
    print(y.shape)
    final = np.copy(j)

    inds = np.array([4, 5, 6, 12, 24, 25, 29, 41, 42])
    assert np.sum(y[:, inds]) == 0
    final[:, inds] = o[:, inds]

    assert np.sum(o[:, 8]) == 0
    final[:, 8] = y[:, 8]

    print("y: ")
    print(y)
    print("o: ")
    print(o)
    print("j: ")
    print(j)
    print("f: ")
    print(final)

    np.save(save, final)

    saved = np.load(save)
    assert np.sum(final) == np.sum(saved)

mergestats(young="/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/final/final_new_masking/merging_priors/T2/young/priors/prior_means.npy",
           old="/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/final/final_new_masking/merging_priors/T2/old/priors/prior_means.npy",
           joint="/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/final/final_new_masking/merging_priors/T2/joint/prior_means.npy",
           save="/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/final/final_new_masking/merging_priors/T2/T2merged/prior_means.npy")
