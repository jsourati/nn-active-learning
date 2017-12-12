from scipy.signal import convolve2d
import numpy as np
import nibabel
import nrrd
import pdb
import os


class PatchBinaryData(object):
    """Class for creating patch-based data set
    from a series of masked images
    
    The class has methods for patchifying the 
    images, partitiong the patches, dividing
    them into batches, loading the patches
    from a set of indices for training a 
    model, etc.
    """
    
    def __init__(self, img_addrs, mask_addrs):
        self.img_addrs = img_addrs
        self.mask_addrs = mask_addrs
        
        # construcitng the patch dictionary, 
        # including only the maksed indices
        # (rest of the indices are unmasked)
        self.masked_inds = {}
        for addr in mask_addrs:
            mask,_ = nrrd.read(addr)
            raveled_inds = ravel_binary_mask(
                mask)
            self.masked_inds.update(
                {addr: raveled_inds})


def ravel_binary_mask(mask):
    """Reading and raveling masked indices
    of a given 3D mask
    
    The maksed indices are assumed to have 
    intensity values more than zero (say, 1.)
    """
    
    multi_inds = np.where(mask>0.)
    inds = np.ravel_multi_index(multi_inds, 
                                mask.shape)
    
    return inds

def extract_Hakims_data_path():
    """Preparing addresses pointing to the
    raw images and masks of brain data
    that Hakim has labeled
    
    """
    
    ids = ['00%d'% i for i in 
           np.arange(1,10)] + [
               '0%d'% i for i in 
               np.arange(10,67)]

    root_dir = '/common/data/raw/Hakim/For_Christine/Mrakotsky_IBD_Brain/Processed'
    mask_rest_of_path = 'scan01/common-processed/anatomical/03-ICC/'
    img_rest_of_path = 'scan01/common-processed/anatomical/01-t1w-ref/'

    mask_addrs =[]
    img_addrs = []
    for idx in ids:
        name = os.listdir(
            os.path.join(
                root_dir,'Case%s'% idx))[0]
        mask_addrs += [
            os.path.join(
                root_dir,'Case%s'% idx,
                name,mask_rest_of_path,
                'c%s_s01_ICC.nrrd'% idx)]
        img_addrs += [
            os.path.join(
                root_dir,'Case%s'% idx,
                name,img_rest_of_path,
                'c%s_s01_t1w_ref.nrrd'% idx)]
        
    return img_addrs, mask_addrs

def sample_masked_data(img,mask,slices,N):
    """Sampling from a masked 3D image in way
    that a balanced number of samples are
    drawn from the masked class, the structured
    background, and non-structured background
    
    :Parameters:
    
        **img** : 3D array
            the raw image
    
        **mask** : 3D binary array
            mask of the image (with labels
            0 and 1
    
        **slices** : array-like of integers
            index of slices from which data
            samples will be drawn
    
        **N** : list of three integers

                * N[0]: # masked samples
                * N[1]: # structured non
                  masked samples
                * N[2]: # non-structured
                  non-masked samples
    """
    
    sel_inds = []
    sel_labels = []
    for s in slices:
        img_slice = img[s,:,:]
        mask_slice = mask[s,:,:]
        
        # partitioning the 2D indices into
        # three groups:
        # (masked, structured non-masked,
        #  non-structured non-masked)
        (masked,Hvar,Lvar)=partition_2d_indices(
            img_slice, mask_slice)
        gmasked = expand_raveled_inds(
            masked, s, 0, img.shape)
        gHvar = expand_raveled_inds(
            Hvar, s, 0, img.shape)
        gLvar = expand_raveled_inds(
            Lvar, s, 0, img.shape)

        # randomly draw samples from each 
        # partition
        # -------- masked
        if N[0] > len(gmasked):
            sel_inds += list(gmasked)
            sel_labels += list(
                np.ones(len(gmasked),
                     dtype=int))
        else:
            r_inds = np.random.permutation(
                len(gmasked))[:N[0]]
            sel_inds += list(gmasked[r_inds])
            sel_labels += list(
                np.ones(N[0],dtype=int))
        # ------ non-masked structured
        if N[1] > len(gHvar):
            sel_inds += list(gHvar)
            sel_labels += list(np.zeros(
                len(gHvar),dtype=int))
        else:
            r_inds = np.random.permutation(
                len(gHvar))[:N[1]]
            sel_inds += list(gHvar[r_inds])
            sel_labels += list(
                np.zeros(N[1],dtype=int))
        # ------ non-masked non-structured
        if N[2] > len(gLvar):
            sel_inds += list(gLvar)
            sel_labels += list(np.zeros(
                len(gLvar),dtype=int))
        else:
            r_inds = np.random.permutation(
                len(gLvar))[:N[2]]
            sel_inds += list(gLvar[r_inds])
            sel_labels += list(
                np.zeros(N[2],dtype=int))
            
    return sel_inds, sel_labels
        

def partition_2d_indices(img,mask):
    """Partitioning an image into three
    different groups, based on the masked
    indices and variance of the patches around
    the pixels of the given 2D image
    
    :Returns:
    
        **masked_inds** : array of ints
            array of indices which have
            non-zero values in the mask

        **Hvar_inds** : array of ints
             array of indices with zero
             values in the mask, and with
             "high" variance patch around
             them
        **Lvar_inds** : array of ints
             array of indices with zero
             values in the mask, and with
             "low" variance patch around
             them
    """
    
    masked_inds = np.where(mask>0)
    masked_inds = set(np.ravel_multi_index(
        masked_inds, mask.shape))

    # computing the patch variance
    d = 5
    var_map = get_vars_2d(img, d)
    var_map[var_map==0] += 1e-1
    var_map = np.log(var_map)
    var_thr = 2.
    
    # create the non-masked high-variance
    # (structured) and low-variance (non-
    # structured) indices
    # -------------------------------
    # taking the High variance indices
    # and removing the masked ones
    Hvar_inds = np.where(
        var_map > var_thr)
    Hvar_inds = set(np.ravel_multi_index(
        Hvar_inds, mask.shape))
    Hvar_inds = Hvar_inds - masked_inds
    # do the same thing for Low variance
    Lvar_inds = np.where(
        var_map < var_thr)
    Lvar_inds = set(np.ravel_multi_index(
        Lvar_inds, mask.shape))
    Lvar_inds = Lvar_inds - masked_inds
    
    # return the partitioning
    return (np.array(list(masked_inds)), 
            np.array(list(Hvar_inds)),
            np.array(list(Lvar_inds)))
    

def get_vars_2d(img,d):
    """Get the variance of a given 2D image
    by means of a convolution
    
    The trick is using the fact that
    Var[x] = E[x^2] - E[x]^2.
    Thus we can find the variance of patches
    centered at different pixels, by computing
    E[x] and E[x^2] using 2D convolution of
    the image and a dxd all-one matrix (kernel)
    where d here denotes the size of the patch
    around each pixel.
    """
    
    # making the image uint64 so that no
    # precision problem happens when squaring
    # up the intensities
    img = np.uint64(img)
    
    # forming the kernel
    kernel = np.ones((d,d))
    
    # E[x] -- mean of the patches
    Ex = convolve2d(
        img,kernel,'same') / float(d**2)
    # E[x^2] -- mean of patch-squared
    ExP2 = convolve2d(
        img**2 ,kernel,'same') / float(d**2)
    
    # the variance
    varx = ExP2 - Ex**2
    
    return varx


def get_borders(mask):
    """Returning border pixels of a given 2D
    binary mask (with labels 0 and 1)
    
    A border pixel is a pixel which has 
    neighborhood with both 0 and 1 classes.
    The neighborhood size is a parameter that
    is determined inside the function, and in
    order to check the values of neighborhood's
    pixels, we use 2D convolution of the mask 
    with an all-one kernel.
    """
    
    kernel_size = 5
    kernel = np.ones((kernel_size,kernel_size))
    
    # upper and lower bounds for considering 
    # a voxel as border or not-border:
    # if a class has less than `lower_percnt`
    # of the neighborhood, the centered voxel
    # won't be consiered as a border voxel
    least_percnt = 0.2
    least_size = int(least_percnt*kernel_size**2)
    masked_lb = least_size
    masked_ub = kernel_size**2 - least_size
    
    c_mask = convolve2d(mask,kernel,'same')
    
    return np.logical_and(
        masked_lb < c_mask,
        c_mask < masked_ub)

def expand_raveled_inds(inds_2D, 
                        slice_idx,
                        slice_view,
                        shape_3D):
    """Covnerting a set of raveled single
    indices that are built in terms of 
    one of  2D slices of a 3D volume, into 
    another set of single raveled indices 
    that are in terms of the whole  3D shape
    
    :Parameters:
    
        **inds_2D** : array of integers
            array of indices based on the 
            2D slice; if it is a single
            integer error will be raised
    
        **slice_idx** : positive integer
            index of the slice based on 
            which the 2D indices are given
    
        **slice_view** : int from {0,1,2}
            this integer shows in what 
            view the slice has bee taken;
            e.g., if it is 0, the slice
            has been taken from the first
            component of the 3D  volume 
            (hence `slice_idx<shape[0]`,
            if it is 1, it means that 
            the slice is taken from the
            second component, so on.

        **shape** : tuple of integers
            shape of the whole volume
    """
    
    # create the shape_2D from shape_3D
    shape_2D = tuple(np.delete(
        shape_3D, slice_view))

    # get the multi-index inside the slice
    multi_inds = np.unravel_index(
        inds_2D, shape_2D)
    
    # add the ID of the slice in the proper
    # location of the expanded multi-indices
    slice_idx_list = slice_idx*np.ones(
        len(inds_2D))
    slice_idx_list = np.int64(slice_idx_list)

    if slice_view==0:
        multi_inds = (
            slice_idx_list,) + multi_inds

    elif slice_view==1:
        multi_inds = (
            multi_inds[0],) + (
                slice_idx_list,) + (
                    multi_inds[1],)
        
    elif slice_view==2:
        multi_inds += (slice_idx_list,)
    
    # ravel the expanded multi-indices
    # using the given 3D volume shape
    inds_3D = np.ravel_multi_index(
        multi_inds, shape_3D)
    
    return inds_3D
