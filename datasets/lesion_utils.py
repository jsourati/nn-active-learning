from skimage.measure import label
from itertools import zip_longest
import tensorflow as tf
import nibabel as nib
import numpy as np
import imageio
import shutil
import nrrd
import h5py
import pdb
import os


def find_lesion_components(mask):
    """Returning connected components of a given mask
    tensor, where each component corresponds to a
    given lesion, and their volumes
    """

    CC_labels = label(mask)
    bkg_label = CC_labels[0,0,0]

    # ignore background as the largest component
    comp_labels = np.unique(CC_labels)
    comp_labels = comp_labels[comp_labels!=bkg_label]

    # sorting lesions in terms of their volumes
    vols = np.array([np.sum(CC_labels==i) for i in comp_labels])
    sorted_labels = comp_labels[np.argsort(-vols)]

    # re-index the component labels such that the largest
    # one has index 1, the second largest one has index two, etc.
    new_CC_labels = np.zeros(CC_labels.shape)
    for i in range(len(sorted_labels)):
        new_CC_labels[CC_labels==sorted_labels[i]] = i+1

    return new_CC_labels


def drop_lesions_with_threshold(mask, thr):
    """Removing lesions with volume less than a given
    threshold
    """

    CC = find_lesion_components(mask)

    # thresholding the components (lesions)
    for label in np.unique(CC):
        vol = np.sum(CC==label)
        if vol < thr:
            CC[CC==label] = 0

    return np.uint8(CC>0)
