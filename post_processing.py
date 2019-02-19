from scipy.ndimage import morphology
from skimage.measure import label

import numpy as np
import os


def connected_component_analysis_3d(seg):
    """Binary connected component analysis
    for 3D segmentation masks

    ASSUMPTION: the voxel at coordinate (0,0,0)
    is assumed to be a background.
    """

    CC_labels = label(seg)

    # (0,0,0) belongs to background
    bkg_label = CC_labels[0,0,0]
    # labels of different components
    comp_labels = np.unique(CC_labels)
    comp_labels = list(comp_labels)
    comp_labels.remove(bkg_label)

    # find the component with largest volume
    vols = np.zeros(len(comp_labels))
    for i, lab in enumerate(comp_labels):
        vols[i] = np.sum(CC_labels==lab)
    largest_comp_label = comp_labels[np.argsort(-vols)[0]]

    cc_seg = np.zeros(seg.shape, dtype=np.uint32)
    cc_seg[CC_labels==largest_comp_label] = 1

    return cc_seg

def fill_holes(seg):
    """Filling holes in a binary segmentation mask
    """

    return np.uint32(morphology.binary_fill_holes(seg))
