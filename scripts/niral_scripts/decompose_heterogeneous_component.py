import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_otsu
from ext.lab2im.utils import list_images_in_folder, load_volume


def get_roi_intensity_histogram(image, label_map, roi):
    """
    Computes intensity histogram for a specific roi
    :param image: Intensity image
    :param label_map: label map
    :param roi: roi for which histogram is computed
    :return: histogram
    """

    roi_intensity = image[label_map == roi].flatten()
    hist, bin_edges = np.histogram(roi_intensity, bins=100)
    return hist, bin_edges


if __name__ == '__main__':
    image_list = list_images_in_folder('/home/turja/DCAN_T2')
    label_list = list_images_in_folder('/home/turja/DCAN_roi')
    image_path = '/home/turja/DCAN_T2/DCAN04_T2w_brain.nii.gz'
    label_path_1 = '/home/turja/DCAN_roi/DCAN04_roi2_WMS.nii.gz'
    label_path_2 = '/home/turja/DCAN_roi/DCAN04_roi2.nii.gz'
    # for i, image_path in enumerate(image_list):
    vol = load_volume(image_path)
    # vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol))
    label_1 = load_volume(label_path_1)
    label_2 = load_volume(label_path_2)
    hist, bin_edges = get_roi_intensity_histogram(vol, label_1, roi=103)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # classif = GaussianMixture(n_components=2)
    # classif.fit(vol.reshape((vol.size, 1)))
    plt.figure(figsize=(10, 5))
    plt.bar(bin_edges[1:], hist)
    plt.show()
