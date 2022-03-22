# import SimpleITK as itk
from scipy.ndimage import label
import numpy as np
from ext.lab2im import utils
import os


# s = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
#      [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
#      [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]


def fuse_labels(seg_dir, merge_CWM=True):
    # Merge white matters
    path_segs = utils.list_images_in_folder(seg_dir)

    for label_map in path_segs:
        WM_map, aff, header = utils.load_volume(label_map, im_only=False)
        print("21" + str(np.any(WM_map == 21)))
        print("61" + str(np.any(WM_map == 61)))
        print("7" + str(np.any(WM_map == 7)))
        print("46" + str(np.any(WM_map == 46)))
        print("170" + str(np.any(WM_map == 170)))

        WM_map[WM_map == 21] = 2
        WM_map[WM_map == 61] = 41
        WM_map[WM_map == 170] = 16
        if merge_CWM:
            WM_map[WM_map == 7] = 8
            WM_map[WM_map == 46] = 47

        utils.save_volume(volume=WM_map, aff=aff, header=header, path=label_map)


def get_islands(label_map_dir, save_map, threshold=None):
    seg_paths = utils.list_images_in_folder(label_map_dir)
    for label_map_path in seg_paths:
        label_map, aff, header = utils.load_volume(label_map_path, im_only=False)
        label_list, _ = utils.get_list_labels(label_list=None, labels_dir=label_map_path, save_label_list=None,
                                              FS_sort=True)

        # mark all the islands for every label except the background
        all_islands = np.zeros(label_map.shape)
        for lb in label_list[1:]:
            feature_island, num_features = label(label_map == lb)
            if num_features > 1:
                # get the assigned label of the largest connected component for the specified structure label. The
                # component is assumed to be larger than all islands but smaller than the "0" label.
                component_sizes = np.bincount(np.array(feature_island).flatten())
                label_main_component = np.argsort(component_sizes)[-2]

                # all voxels excluding the '0' label as well as the main component are islands.
                filter_islands_or_bg = np.logical_or(feature_island == 0, feature_island == label_main_component)

                # all islands with size greater than the threshold will be treated the same as the main component
                # and the background
                if threshold is not None:
                    large_island_label_list = np.where(component_sizes >= threshold)[0]
                    for large_label in large_island_label_list:
                        filter_islands_or_bg = np.logical_or(feature_island == large_label, filter_islands_or_bg)

                # The '0'component and the largest component (as well as large islands if there is a threshold) will
                # be labeled as zero while the rest (islands) will be labeled as one.
                islands_only = np.where(filter_islands_or_bg, 0, 1)

                # merge stray voxels for each label
                all_islands = np.add(islands_only, all_islands)

        # dealing with the background label

        # feature_island, num_features = label(label_map == 0)
        # print("number of background features = " + str(num_features))
        #
        # if num_features > 1:
        #     # as long as there is padding, the label for the background must be one since label does not sort
        #     # components by size.
        #     component_sizes = np.bincount(np.array(feature_island).flatten())
        #     label_main_component = 1
        #
        #     # all voxels excluding the '0' label as well as the main component are islands.
        #     filter_islands_or_bg = np.logical_or(feature_island == 0, feature_island == label_main_component)
        #
        #     # all islands with size greater than the threshold will be treated the same as the main component
        #     # and the background
        #     if threshold is not None:
        #         large_island_label_list = np.where(component_sizes >= threshold)[0]
        #         for large_label in large_island_label_list:
        #             filter_islands_or_bg = np.logical_or(feature_island == large_label, filter_islands_or_bg)
        #
        #     # The '0'component and the largest component (as well as large islands if there is a threshold) will
        #     # be labeled as zero while the rest (islands) will be labeled as one.
        #     islands_only = np.where(filter_islands_or_bg, 0, 1)
        #
        #     # merge stray voxels for each label
        #     all_islands = np.add(islands_only, all_islands)

        # background processing end

        print("test whether there are voxels being counted twice......")
        print(np.any(all_islands > 1))

        islands_divided, num_islands = label(all_islands)
        print("there are " + str(num_islands) + " island components in " + label_map_path)
        save_name = label_map_path.split('/')[-1].split('_')[0] + "_islands.nii.gz"
        print(os.path.join(save_map, save_name))
        utils.save_volume(volume=islands_divided, aff=aff, header=header,
                          path=os.path.join(save_map, save_name), dtype='int32')


def get_connected_label(x, y, z, seg, posterior, islands, label_list):
    """
    :param x: x-coordinate of element
    :param y: y-coordinate of element
    :param z: z-coordinate of element
    :param seg: segmentation array
    :param posterior: posterior array
    :param islands: array marking stray voxels with value != 0
    :param label_list: array of all labels used in the segmentation array, must be FS-sorted
    """

    # there is no bound checking for this function
    islands = np.array(islands)
    posterior = np.array(posterior)
    label_list = np.array(label_list)

    list_adjacent = [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z),
                     (x, y, z + 1), (x, y, z - 1)]

    # find the index of adjacent voxels that are also part of the same island
    belong_to_island = []
    for index in range(len(list_adjacent)):
        if islands[list_adjacent[index]] != 0:
            belong_to_island.append(index)
    # print(belong_to_island)

    # get all possible unique labels that would ensure connectivity after converting the element into it.
    possible_labels = []
    for element in range(len(list_adjacent)):
        if element not in belong_to_island:
            possible_labels.append(seg[list_adjacent[element]])
    # print(possible_labels)
    possible_labels = np.unique(possible_labels)
    # print(possible_labels)

    # return the label that has the maximum posterior and would ensure connectivity
    posterior_array = posterior[x, y, z]
    # print(len(posterior_array) == len(label_list))

    return sorted(possible_labels, key=lambda lb: posterior_array[np.where(label_list == lb)])[-1]


def clear_islands(label_dir, island_dir, posterior_dir, save_dir, labels):
    """

    """

    subject_paths = utils.list_images_in_folder(label_dir)
    island_map_paths = utils.list_images_in_folder(island_dir)
    posterior_paths = utils.list_images_in_folder(posterior_dir)

    for subject in range(len(subject_paths)):

        print("processing subject " + str(subject))

        print(subject_paths[subject])
        print(island_map_paths[subject])
        print(posterior_paths[subject])

        seg = utils.load_volume(subject_paths[subject], im_only=True)
        islands, aff, header = utils.load_volume(island_map_paths[subject], im_only=False)
        posterior = utils.load_volume(posterior_paths[subject], im_only=True)

        label_list, _ = utils.get_list_labels(label_list=labels, labels_dir=subject_paths[subject],
                                              save_label_list=None,
                                              FS_sort=True)
        print(len(label_list))
        print("the label list is: " + str(label_list))
        print(len(posterior[50, 50, 50]))

        save_name = subject_paths[subject].split('/')[-1].split('_')[0] + "_islands_removed.nii.gz"
        print("save to: ")
        print(save_name)

        new_segmentation = np.array(seg)
        (x, y, z) = islands.shape
        print("shape: " + str((x, y, z)))
        print("labels used: " + str(label_list))

        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if islands[i, j, k] != 0:
                        if np.random.random() < 0.1:
                            print("replacing island label " + str(islands[i, j, k]))
                        # print("the initial label is: " + str(seg[i, j, k]))
                        new_segmentation[i, j, k] = get_connected_label(i, j, k, new_segmentation, posterior, islands,
                                                                        label_list)
                        # print("the label now is: " + str(new_segmentation[i, j, k]))
                        islands[i, j, k] = 0
        save_name = subject_paths[subject].split('/')[-1].split('_')[0] + "_islands_removed.nii.gz"
        utils.save_volume(volume=new_segmentation, aff=aff, header=header,
                          path=os.path.join(save_dir, save_name),
                          dtype='int32')


#

# fuse labels
fuse_labels(seg_dir=r"/home/ziyaos/bootstrapping/6mo/merged_seg", # The path to the segmentation maps, the fused maps will be saved in place
            merge_CWM=True) # Whether or not to merge the Cerebellum white matter and grey matter. [7]->[8] and [46]->[47]

# get and save the islands
get_islands(label_map_dir=r"/home/ziyaos/bootstrapping/6mo/merged_seg", # the path to the segmentation maps
            save_map=r"/home/ziyaos/bootstrapping/6mo/islands", # where to save the map containing the islands
            threshold=15) # components larger than the threshold will not be modified

# clear islands given an island map
clear_islands(label_dir="/home/ziyaos/bootstrapping/6mo/merged_seg", # the path to the segmentation labels
              island_dir="/home/ziyaos/bootstrapping/6mo/islands", # the path to the island maps
              posterior_dir="/home/ziyaos/bootstrapping/6mo/merged_posterior", # the path to the merged posteriors
              save_dir="/home/ziyaos/bootstrapping/6mo/island_removed", # the directory to the save the final label maps
              labels=[0, 14, 15, 16, 170, 172, 2, 3, 4, 7, 8, 10, 11, 12, 13, 17, 18, 21, 26, 28, 41, 42, 43, 46,
                    47, 49, 50, 51, 52, 53, 54, 58, 60, 61]) # the list of labels the corresponds to the posteriors (the segmention labels when predicting)


