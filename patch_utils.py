from scipy.signal import convolve2d
import numpy as np
import warnings
import nibabel
import nrrd
import pdb
import os

import NN

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
        
        
    def generate_samples(self, 
                         img_inds,
                         N,
                         ratio_thr,
                         view):
        """Generating samples from somes of
        the images whose indices are given
        in terms of `self.img_addrs`
        """

        inds_dict = {self.img_addrs[i]:[] 
                     for i in img_inds}
        labels_dict = {self.img_addrs[i]:[] 
                     for i in img_inds}
        types_dict = {self.img_addrs[i]:[] 
                     for i in img_inds}
        # corresponding integer to the view
        # sagittal : 0
        # coronal  : 1
        # axial    : 2
        view_id = np.where(np.array(
            ['sagittal',
             'coronal',
             'axial'])==view)[0][0]
        
        # sampling from the volumes
        for i in img_inds:
            img,_ = nrrd.read(
                self.img_addrs[i])
            mask,_ = nrrd.read(
                self.mask_addrs[i])
            
            # determining the slices for which 
            # the masked volume is larger than
            # a specified threshold
            ratios = np.zeros(img.shape[view_id])
            for j in range(len(ratios)):
                if view_id==0:
                    mask_vol = np.sum(mask[j,:,:])
                    total_vol = np.prod(img.shape[1:])
                elif view_id==1:
                    mask_vol = np.sum(mask[:,j,:])
                    total_vol = img.shape[0]*img.shape[2]
                elif view_id==2:
                    mask_vol = np.sum(mask[:,:,j])
                    total_vol = np.prod(img.shape[:-1])

                ratios[j] = float(mask_vol) / \
                            float(total_vol)

            slices = np.where(
                ratios>ratio_thr)[0]
            
            if len(slices)==0:
                raise warnings.warn(
                    "Image %d" % i + 
                    " does not have any slice "+
                    "satsifying the ratio check.")
                continue
                
            print('Sampling %d slices from image %d' 
                  % (len(slices), i))
            sel_inds,sel_labels,sel_types=sample_masked_volume(
                img, mask, slices, N, view)

            inds_dict[self.img_addrs[i]] = sel_inds
            labels_dict[self.img_addrs[i]]=sel_labels
            types_dict[self.img_addrs[i]]=sel_types
            
        return inds_dict, labels_dict, types_dict

def get_batches(inds_dict,
                batch_size):
    """Divide a given set of image indices and
    their labels into batches of specified
    size
    """

    # get the total size of the generated
    # samples
    imgs = list(inds_dict.keys())
    n = np.sum([len(inds_dict[img])
                for img in imgs])


    batches = NN.gen_batch_inds(
        n, batch_size)

    return batches

def get_batch_vars(inds_dict,
                   labels_dict,
                   batch_inds,
                   patch_shape):
    """Creating tensors of data and labels
    for a model with batch-learning (such
    as CNN object) given a batch of
    indices

    CAUTIOUS: the input `patch_shape` should
    have odd elements so that the radious 
    along each direction will be an integer
    and no shape mismatch will happen.
    """

    # initializing variables
    b = len(batch_inds)
    batch_tensors = np.zeros(
        (b,)+patch_shape)
    batch_labels = np.zeros((2,b))

    # locating batch indices inside
    # the data dictionaries
    sub_dict = locate_in_dict(inds_dict,
                              batch_inds)
    # calculating radii of the patch
    rads = np.zeros(3,dtype=int)
    for i in range(3):
        rads[i] = int((patch_shape[i]-1)/2.)

    # extracting patches from the image
    cnt = 0
    for img_path in list(sub_dict.keys()):
        img,_ = nrrd.read(img_path)
        # padding with the patch radius 
        # so that all patch indices 
        # fall in the limits of the image
        # (skip the z-direction, for now)
        padded_img = np.pad(
            img, 
            ((rads[0],rads[0]),
             (rads[1],rads[1]),
             (rads[2],rads[2])),
            'constant')

        # indices in the batch that belong
        # to the `img_path` (sub-batch inds)
        subbatch_inds = sub_dict[img_path]
        b_i = len(subbatch_inds)

        # adding one-hot labels
        sub_labels = np.array(labels_dict[
            img_path])[subbatch_inds]
        subhot = np.zeros((2,b_i))
        subhot[0,sub_labels==0]=1
        subhot[1,sub_labels==1]=1
        batch_labels[
            :,cnt:cnt+b_i] = subhot

        # converting to multiple-3D indicse
        imgbatch_inds = np.array(inds_dict[
            img_path])[subbatch_inds]
        multi_inds3D = np.unravel_index(
            imgbatch_inds, img.shape)

        # extracting tensors 
        for i in range(b_i):
            # multiple-indices of the
            # centers change in the padded
            # image; 
            # an adjustment is needed 
            center_vox = [
                multi_inds3D[0][i]+rads[0],
                multi_inds3D[1][i]+rads[1],
                multi_inds3D[2][i]+rads[2]]

            patch = padded_img[
                center_vox[0]-rads[0]:
                center_vox[0]+rads[0]+1,
                center_vox[1]-rads[1]:
                center_vox[1]+rads[1]+1,
                center_vox[2]-rads[2]:
                center_vox[2]+rads[2]+1]

            batch_tensors[cnt,:,:,:] = patch

            cnt += 1

    return batch_tensors, batch_labels
            

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

def extract_newborn_data_path():
    """Preparing addresses pointing to
    the raw images and masks of 
    T1-weighted MRI of brains on newborn
    subjects. The masks are manually 
    modified by Hakim.
    """

    # common root directory 
    root_dir = '/common/collections/dHCP/'+\
               'dHCP_DCI_spatiotemporal_atlas/'+\
               'Processed/'
    
    # common sub-directories
    # (except the data files which include
    # the subject and session codes in their
    # names)
    img_rest_of_path = 'common-processed' +\
                       '/anatomical/01-t1w-ref'
    mask_rest_of_path = 'common-processed' +\
                        '/anatomical/03-ICC'


    # subject-specific sub-directories
    dirs = get_subdirs(root_dir)
    img_addrs = []
    mask_addrs = []
    for i, d in enumerate(dirs):
        if not('sub-CC00' in d):
            continue
            
        # there are two levels of subject-
        # specific sub-directories
        subdir = get_subdirs(os.path.join(
            root_dir, d))[0]
        subsubdir = get_subdirs(os.path.join(
            root_dir, d, subdir))[0]
        
        # we need the codes for accessing
        # to names of the data
        sub_code = d[4:]
        sess_code = subsubdir[4:]
            
        # subject-specific sub-directories
        subdirs = os.path.join(
            root_dir,
            d,
            subdir,
            subsubdir)

        """Putting everything together"""
        img_addrs += [
            os.path.join(
                root_dir,
                subdirs,
                img_rest_of_path,
                'c%s_s%s_t1w_ref.nrrd'% 
                (sub_code, sess_code))]
        
        mask_addrs += [
            os.path.join(
                root_dir,
                subdirs,
                mask_rest_of_path,
                'c%s_s%s_ICC.nrrd'% 
                (sub_code, sess_code))]
        
    return img_addrs, mask_addrs


def get_subdirs(path):
    """returning all sub-directories of a 
    given path
    """
    
    subdirs = [d for d in os.listdir(path)
               if os.path.isdir(os.path.join(
                       path,d))]
    
    return subdirs

def sample_masked_volume(img,
                         mask,
                         slices,
                         N,
                         view):
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
    sel_types = []
    for s in slices:
        if view=='axial':
            img_slice = img[:,:,s]
            mask_slice = mask[:,:,s]
            slice_view=2
        elif view=='coronal':
            img_slice = img[:,s,:]
            mask_slice = mask[:,s,:]
            slice_view=1
        elif view=='sagittal':
            img_slice = img[s,:,:]
            mask_slice = mask[s,:,:]
            slice_view=0
        
        # partitioning the 2D indices into
        # three groups:
        # (masked, structured non-masked,
        #  non-structured non-masked)
        (masked,Hvar,Lvar)=partition_2d_indices(
            img_slice, mask_slice)
        gmasked = expand_raveled_inds(
            masked, s, slice_view, img.shape)
        gHvar = expand_raveled_inds(
            Hvar, s, slice_view, img.shape)
        gLvar = expand_raveled_inds(
            Lvar, s, slice_view, img.shape)

        # randomly draw samples from each 
        # partition
        # -------- masked
        if N[0] > len(gmasked):
            sel_inds += list(gmasked)
            sel_labels += list(
                np.ones(len(gmasked),
                     dtype=int))
            sel_types += [0]*len(gmasked)
        else:
            r_inds = np.random.permutation(
                len(gmasked))[:N[0]]
            sel_inds += list(gmasked[r_inds])
            sel_labels += list(
                np.ones(N[0],dtype=int))
            sel_types += [0]*N[0]
        # ------ non-masked structured
        if N[1] > len(gHvar):
            sel_inds += list(gHvar)
            sel_labels += list(np.zeros(
                len(gHvar),dtype=int))
            sel_types += [1]*len(gHvar)
        else:
            r_inds = np.random.permutation(
                len(gHvar))[:N[1]]
            sel_inds += list(gHvar[r_inds])
            sel_labels += list(
                np.zeros(N[1],dtype=int))
            sel_types += [1]*N[1]
        # ------ non-masked non-structured
        if N[2] > len(gLvar):
            sel_inds += list(gLvar)
            sel_labels += list(np.zeros(
                len(gLvar),dtype=int))
            sel_types += [2]*len(gLvar)
        else:
            r_inds = np.random.permutation(
                len(gLvar))[:N[2]]
            sel_inds += list(gLvar[r_inds])
            sel_labels += list(
                np.zeros(N[2],dtype=int))
            sel_types += [2]*N[2]
            
    return sel_inds, sel_labels, sel_types

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

def locate_in_dict(inds_dict, 
                   inds):
    """Locating a set of indices inside
    a given data dictionary
    
    Note that data dictionaries 
    (either index- or labels-
    dictionaries) have several indices
    assigned to each image. Hence
    locating a global index (which refers
    to a specific index inside the 
    dictionary is not straightforward.
    Here, we assume that the indices or
    labels are located in the dictionaries
    in the same ordered as they are
    saved. For example, only writing the
    indices of the corresponding data in
    the dictionary, we would get
    
    inds_dict = {'img-1': [0, 1,...,N1-1],
                 'img-2': [N1, N1+1,
                           ..., N1+N2-1],
                   .
                   .
                   
                  'img-10': [N1+...+N9,
                             N1+...+N9+1,
                             ...,  
                             N1+N2+...+N10-1]}
    
    In this function, for each given index,
    we locate it in the dictionary, and 
    return a sub-dictionary containing 
    indices WITH RESPECT TO the contents of
    the same keys in the input dictionary.
    """
    
    # `imgs` are the keys
    imgs = list(inds_dict.keys())

    sub_dict = {img:[] for img in imgs}
    key_vols = [len(inds_dict[img]) 
                for img in imgs]
    key_cumvols = np.append(
        -1, np.cumsum(key_vols)-1)
    
    for ind in inds:
        
        # finding the corresponding key
        ind_key = key_cumvols.searchsorted(
            ind) - 1
        # updating the sub-dictionary
        sub_dict[imgs[ind_key]] += [
            ind-key_cumvols[ind_key]-1]
        
        
    # removing those keys who did not
    # have any corresponding indices
    subkey_vols = [len(sub_dict[img]) 
                   for img in imgs]
    keys_to_remove = np.where(
        np.array(subkey_vols)==0)[0]
    for key in keys_to_remove:
        del sub_dict[imgs[key]]
    
    return sub_dict
    

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

def get_mean_var(batches, 
                 inds_dict, 
                 labels_dict,
                 pw_dataset,
                 patch_shape):
    """Computing the intensity mean and 
    variance of all trianing patches
    """
    
    # get the largest intensity
    max_i = 0
    for path in list(inds_dict.keys()):
        img,_ = nrrd.read(path)
        if img.max() > max_i:
            max_i = img.max()
            
    print('\nMax intensity of batches:%d'% max_i)
    
    # since we are not sure what the maximum 
    # intensity is, we leave the right end open
    bin_seq = np.linspace(0,max_i,100)
    hist = np.zeros(len(bin_seq)-1)
    
    cnt = 0
    for i, batch in enumerate(batches):
        batch_tensors,_ = pw_dataset.get_batch_vars(
            inds_dict,
            labels_dict,
            batches[i],
            patch_shape)

        b = len(batch)
        
        # histogram of the new batch
        new_hist = np.histogram(
            batch_tensors, 
            bin_seq)[0]
        
        if i < len(batches)-1:
            prev_c = float(i) / float(i+1)
        else:
            prev_c = float(cnt) / float(
                cnt+b*np.prod(patch_shape))
        
        cnt += b*np.prod(patch_shape)
        
        # update the total histogram
        hist=prev_c*hist + new_hist/float(cnt)
        
        if i%50==0:
            print(i,end=',')
        
    return bin_seq,hist
    
def generate_rgb_mask(img,mask):
    """Generating a colored image based
    on a given 1-channel image and 
    a binary maske
    """
    
    img_rgb = np.repeat(np.expand_dims(
        img,axis=2),3,axis=2)
    img_rgb = np.uint8(
        img_rgb*255./img_rgb.max())
    
    # create a mask in one of the channels
    tmp = img_rgb[:,:,0]
    tmp[mask>0] = 200.
    
    return img_rgb

def get_patches(img, inds, patch_shape):
    """Extacting patches around a given 
    set of 3D indices 
    """
    
    # padding the image with radii
    rads = np.zeros(3,dtype=int)
    for i in range(3):
        rads[i] = int((patch_shape[i]-1)/2.)
            
    padded_img = np.pad(
        img, 
        ((rads[0],rads[0]),
         (rads[1],rads[1]),
         (rads[2],rads[2])),
        'constant')

    # computing 3D coordinates of the samples
    # in terms of the original image shape
    multi_inds = np.unravel_index(
        inds, img.shape)
    
    b = len(inds)
    batch = np.zeros((b,)+patch_shape)
    for i in range(b):
        # adjusting the multi-coordinates 
        # WITH padded margins
        center = [
            multi_inds[0][i]+rads[0],
            multi_inds[1][i]+rads[1],
            multi_inds[2][i]+rads[2]]
        
        patch = padded_img[
            center[0]-rads[0]:
            center[0]+rads[0]+1,
            center[1]-rads[1]:
            center[1]+rads[1]+1,
            center[2]-rads[2]:
            center[2]+rads[2]+1]
        
        batch[i,:,:,:] = patch
        
    return batch
