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

#from NNAL_tools import sample_query_dstr as sample_pmf
import patch_utils


def gen_batch_inds(data_size, batch_size):
    """Generating a list of random indices 
    to extract batches
    """
    
    # determine size of the batches
    quot, rem = np.divmod(data_size, 
                          batch_size)
    batches = list()
    
    # random permutation of indices
    rand_perm = np.random.permutation(
        data_size).tolist()
    
    # assigning indices to batches
    for i in range(quot):
        this_batch = rand_perm[
            slice(i*batch_size, 
                  (i+1)*batch_size)]
        batches += [this_batch]
        
    # if there is remainder, add them
    # separately
    if rem>0:
        batches += [rand_perm[-rem:]]
        
    return batches

def gen_minibatch_labeled_unlabeled_inds(
        L_indic, batch_size, n_labeled=None):
    """Generating "indices" with a specific batch size from a 
    mixture of labeled and unlabeled sample

    * L_inidc : array of binary values
        indicating if a sample in the pool is labeled (1) or not (0)

    * batch_size : int
    

    * n_labeled : None or int
        if not None, an integer specifying a fixed number of labeled
        samples in each generated batch
    
    """
    
    n = len(L_indic)
    if n_labeled is None:
        def eternal_gen():
            while True:
                for inds in gen_batch_inds(n, batch_size):
                    #if len(inds)==1: continue # not interested in small batches
                    yield inds
        gen_tuple = (eternal_gen(),)

    else:
        n_unlabeled = batch_size - n_labeled
        labeled_inds = np.where(L_indic==1)[0]
        unlabeled_inds = np.setdiff1d(np.arange(n),
                                      labeled_inds)
        def labeled_eternal_gen():
            while True:
                for inds in gen_batch_inds(len(labeled_inds), n_labeled):
                    yield labeled_inds[inds]
        def unlabeled_eternal_gen():
            while True:
                for inds in gen_batch_inds(len(unlabeled_inds), n_unlabeled):
                    yield unlabeled_inds[inds]
        gen_tuple = (labeled_eternal_gen(), unlabeled_eternal_gen())

    return zip_longest(*gen_tuple)
        
def gen_minibatch_materials(gen, *args):
    inds = np.concatenate(next(gen))
    return tuple([[arg[ind] for ind in inds]
                  for arg in args])


def prepare_batch_BrVol(imgs,
                        masks, 
                        img_shape, 
                        one_hot_channels=None,
                        slice_choice='uniform',
                        labeled_indic=None):
    """Preparing a batch of image slices from multiple given
    brain volumes

    Second fully version which supports 3D sampling as well as 2D

    This version only works with already-loaded images and
    masks (for sake of simplicity).

    NOTE 1 about `img_shape`:
    If it is a list of 2 elements, 2D axial slices will be loaded,
    and if it has 3 elements, a 3D volume will be loaded from the image.
    In the 3D case, input indices are assumed to be generated in a way
    that there are enough depth margins to take z-z_rad:z+z_rad

    NOTE 2 about `img_shape`:
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

    m = len(imgs[0])   # number of modalities
    b = len(imgs)      # number of selected indices (=batch size)

    if len(img_shape)==2:
        h,w = img_shape
        z = 1
        batch_X = np.zeros((b,1,h,w,m))
        nohot_batch_mask = np.zeros((b,1,h,w))
    else:
        h,w,z = img_shape
        batch_X = np.zeros((b,z,h,w,m))
        nohot_batch_mask = np.zeros((b,z,h,w))
    z_rad = int(z/2)

    if labeled_indic is None:
        labeled_indic = np.ones(b)

    # loop over selected indices
    for i in range(b):
        grnd = masks[i]

        # NOTE:
        # in this version, slice samling does not support 3D generators
        if isinstance(slice_choice, str):
            if slice_choice=='uniform':
                slice_ind = np.random.randint(grnd.shape[-1])
            elif slice_choice=='non-uniform':
                pmf = np.ones(grnd.shape[-1])
                pmf[60:120] = 2
                pmf /= np.sum(pmf)
                slice_ind = sample_pmf(pmf, 1)[0]
        else:
            slice_ind = slice_choice[i]

        # depth jz with respect to the selected slice
        for jz, offset in enumerate(np.arange(-z_rad, z_rad)):
            # jm-th modality
            for jm in range(m):
                img = imgs[i][jm][:,:,slice_ind+offset]

                if jm==0 and jz==0:
                    crimg, init_h, init_w = random_crop(img,h,w)
                else:
                    crimg,_,_ = random_crop(img,h,w,init_h,init_w)

                # if it's a 2D generator, it's done here
                batch_X[i,jz,:,:,jm] = crimg

            # ground truth for the jz-th slice
            if labeled_indic[i]==0:
                nohot_batch_mask[i,jz,:,:] = np.nan
                continue
            cgrnd,_,_ = random_crop(grnd[:,:,slice_ind+offset],
                                    h,w,init_h,init_w)
            nohot_batch_mask[i,jz,:,:] = cgrnd

    if one_hot_channels is not None:
        batch_mask = np.zeros(nohot_batch_mask.shape+(one_hot_channels,))
        for j in range(one_hot_channels):
            batch_mask[:,:,:,:,j] = nohot_batch_mask==j
    else:
        batch_mask = nohot_batch_mask

    if z==1:
        batch_X = np.squeeze(batch_X, axis=1)
        batch_mask = np.squeeze(batch_mask, axis=1)

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

def global2local_inds(batch_inds,
                      set_sizes):
    """Having a finite set of sets with
    ordered elements and aiming to extract
    a subset of them, this function takes
    global indices of the elements in this
    subset and output which elements in
    each set belongs to this subset; the 
    given subset can be one of the batches
    after batch-ifying voxels of a set of
    images
    
    By "global index", we mean an indexing
    system that we can refer to a specific
    element of one of the sets uniquely. 
    Here, assuming that the sets and their
    elements are ordered, our global indexing 
    system refers to the i-th element of the
    j-th set by an index calculated by
    `len(S1) + len(S2) + .. len(Si-1) + j-1`
    
    where `S1` to `Si-1` are the sets that
    are located before the target i-th set
    """

    cumvols = np.append(
        -1, np.cumsum(set_sizes)-1)
    
    # finding the set indices 
    set_inds = cumvols.searchsorted(
        batch_inds) - 1
    # local index for each set
    local_inds = [np.array(batch_inds)[
        set_inds==i]-cumvols[i]-1 for i in
                  range(len(set_sizes))]

    return local_inds

def nrrd_reader(path):
    return nrrd.read(path)[0]

def nii_reader(path):
    dat = nib.load(path)
    return dat.get_data()



def generator_complete_data(X, Y, batch_size, 
                            eternality=False,
                            sample_axis=-1):
    """Generator for a data given in form of a feature 
    vector `X`, and the label vector(s) `Y`, which could
    be an array or a list of arrays
    """

    n = X.shape[sample_axis]
    batches = gen_batch_inds(n, batch_size)

    # dynamic indexing for a specified sample axis
    sl = [slice(None)] * X.ndim
    while True:
        for batch in batches:
            sl[sample_axis] = batch
            if isinstance(Y, list):
                yield X[tuple(sl)], [Yarr[tuple(sl)] for Yarr in Y], batch
            else:
                yield X[tuple(sl)], Y[tuple(sl)], batch

        if not(eternality):
            break


def lesion_patch_gen(imgs,
                     masks,
                     legal_inds,
                     square_patch_size,
                     patch_num):
    """
    -------------------------
    Format of input variables:
    --------------------------
    imgs = [[sub1_img1, sub1_img2,...,sub1_imgM],
             ...,
            [subS_img1, subS_img2,...,subS_imgM]]

    masks = [mask1, ..., maskS]

    legal_inds = [[array(sub1_x1,..., sub1_xN),
                   array(sub1_y1,..., sub1_yN),
                   array(sub1_z1,..., sub1_zN)],
                  ...,
                  [array(subS_x1,..., subS_xN),
                   array(subS_y1,..., subS_yN),
                   array(subS_z1,..., subS_zN)],
                  
    -------------------------
    Format of output variables:
    --------------------------
    patches : array of patches
              shape = (patch_num, 
                       square_patch_size[0],
                       square_path_size[1],
                       #modalities)

    sub_inds : list of selected subjects

    cntr_coords : 3D coordinates of selected
                  patch centers; i.e., the i-th
                  coordinate points to the selected
                  voxel in coordinate of images
                  in `imgs[sub_inds[i]]`
    """


    s = len(imgs)    # number of subjects
    m = len(imgs[0]) # number of modalities
    half_size = int(square_patch_size/2)

    while True:
        # selecting subject to sample patches from
        sub_inds = np.random.randint(0, s, patch_num)

        # sampling central voxel of the patch
        cntr_inds = [np.random.randint(len(legal_inds[i][0]))
                     for i in sub_inds]
        cntr_coords = [(legal_inds[sub_inds[i]][0][cntr_inds[i]], 
                        legal_inds[sub_inds[i]][1][cntr_inds[i]],
                        legal_inds[sub_inds[i]][2][cntr_inds[i]]) 
                       for i in range(len(sub_inds))]

        patches = np.stack([
            np.stack([imgs[sub_inds[i]][j][
                cntr_coords[i][0]-half_size:cntr_coords[i][0]+half_size+1,
                cntr_coords[i][1]-half_size:cntr_coords[i][1]+half_size+1,
                cntr_coords[i][2]]
                      for j in range(m)], axis=2)
            for i in range(len(sub_inds))], axis=0)

        yield patches, sub_inds, cntr_coords
    
