import tensorflow as tf
import numpy as np
import imageio
import shutil
import nrrd
import h5py
import pdb
import os

from NNAL_tools import sample_query_dstr as sample_pmf
import NN_extended
import patch_utils
import create_NN

def prepare_batch_CamVid(img_paths, grnd_paths, img_shape):

    h,w = img_shape
    batch_X = np.zeros((len(img_paths),h,w,3))
    batch_grnd = np.zeros((len(img_paths),h,w))

    for i in range(len(img_paths)):
        # image
        img = imageio.imread(img_paths[i])
        crimg, init_h, init_w = random_crop(img,h,w)
        batch_X[i,:,:,:] = crimg
        # ground truth
        grnd = imageio.imread(grnd_paths[i])
        cgrnd,_,_ = random_crop(grnd,h,w,init_h,init_w)
        batch_grnd[i,:,:] = cgrnd

    return batch_X, batch_grnd

def prepare_batch_BrVol(img_paths, mask_addrs, 
                        img_shape, 
                        one_hot_channels=None,
                        slice_choice='uniform',
                        labeled_indic=None):
    """Preparing a batch of image slices from multiple given
    brain volumes

    If the image shape `img_shape` is not the same as the
    axial slices of the volumes, the slices will be randomly
    cropped.

    Possible values for the inpur argument `slice_choice` are:
    
    * `'uniform'`: randomly draw a slice from each volume from
                   a uniform distribution
    * `'non-uniform'`: randomly draw a slice from each volume
                       from a non-uniform distribution (fixed PMF)
    * a list of integers with the same length as `img_paths`:
        index of slices are pre-specified

    Also `labeled_indic` is an indicator sequence (if given it 
    is set to all-ones) with the same length as `img_paths`
    specifying if each volume is a labeled or unlabeled sample
    (the latter to be used in semi-supervised training). Mask
    of unlabeled slices will be all-zero matrices.
    """

    h,w = img_shape
    m = len(img_paths[0])
    batch_X = np.zeros((len(img_paths),h,w,m))
    nohot_batch_mask = np.zeros((len(img_paths),h,w))
    if labeled_indic is None:
        labeled_indic = np.ones(len(img_paths))

    for i in range(len(img_paths)):
        # sampling a slice
        grnd = nrrd.read(mask_addrs[i])[0]

        if isinstance(slice_choice, str):
            if slice_choice=='uniform':
                slice_ind = np.random.randint(grnd.shape[-1])
            elif slice_choice=='non-uniform':
                pmf = np.ones(grnd.shape[-1])
                pmf[50:220] = 2
                pmf /= np.sum(pmf)
                slice_ind = sample_pmf(pmf, 1)[0]
        else:
            slice_ind = slice_choice[i]

        for j in range(m):
            # image (j'th modality)
            img = nrrd.read(img_paths[i][j])[0]
            img = img[:,:,slice_ind]
            if j==0:
                crimg, init_h, init_w = random_crop(img,h,w)
            else:
                crimg,_,_ = random_crop(img,h,w,init_h,init_w)
            batch_X[i,:,:,j] = crimg

        # ground truth
        if labeled_indic[i]==0:
            nohot_batch_mask[i,:,:] = np.nan
            continue
        grnd = grnd[:,:,slice_ind]
        cgrnd,_,_ = random_crop(grnd,h,w,init_h,init_w)
        nohot_batch_mask[i,:,:] = cgrnd

    if one_hot_channels is not None:
        batch_mask = np.zeros(nohot_batch_mask.shape+(one_hot_channels,))
        for j in range(one_hot_channels):
            batch_mask[:,:,:,j] = nohot_batch_mask==j
    else:
        batch_mask = nohot_batch_mask

    return batch_X, batch_mask


def random_crop(img,h,w,init_h=None,init_w=None):
    '''Assume the given image has either shape [h,w,channels] or [h,w]
    '''

    if init_h is None:
        if img.shape[0]==h:
            init_h=0
        else:
            init_h = np.random.randint(0, img.shape[0]-h)
        if img.shape[1]==w:
            init_w=0
        else:
            init_w = np.random.randint(0, img.shape[1]-w)
    if len(img.shape)==3:
        cropped_img = img[init_h:init_h+h, init_w:init_w+w, :]
    elif len(img.shape)==2:
        cropped_img = img[init_h:init_h+h, init_w:init_w+w]

    return cropped_img, init_h, init_w

#def gaussian_noise(img_batch):
