import nibabel as nib
import numpy as np
import nrrd
import pdb

from .path_loader import extract_ISBI2015_MSLesion_data_path
from .utils import gen_minibatch_labeled_unlabeled_inds, \
    gen_minibatch_materials, prepare_batch_BrVol

class MS_challenge(object):

    (
        FLAIR_addrs,
        MPRAGE_addrs,
        PD_addrs,
        T2_addrs,
        mask1_addrs, 
        mask2_addrs
    ) = extract_ISBI2015_MSLesion_data_path('training')
    C = 2

    def __init__(self, rnd_seed, 
                 labeled_size, 
                 unlabeled_size,
                 valid_size,
                 load_train_valid=False):
        self.seed = rnd_seed

        self.combined_paths = [[self.FLAIR_addrs[i], 
                                self.MPRAGE_addrs[i], 
                                self.PD_addrs[i], 
                                self.T2_addrs[i]] 
                               for i in range(len(self.FLAIR_addrs))]
        n = len(self.combined_paths)

        rand_inds = np.random.RandomState(seed=rnd_seed).permutation(len(self.T2_addrs))
        L_inds = rand_inds[:labeled_size]
        UL_inds = rand_inds[labeled_size : labeled_size+unlabeled_size]
        ntrain = labeled_size+unlabeled_size
        self.train_inds = np.concatenate((L_inds, UL_inds))
        self.L_indic = np.array([1]*len(L_inds) + [0]*len(UL_inds))

        self.valid_inds = rand_inds[ntrain : ntrain+valid_size]
        self.test_inds = list(set(np.arange(n)) - 
                              set(self.train_inds) - 
                              set(self.valid_inds))
        
        self.tr_img_paths = [self.combined_paths[i] for i in self.train_inds]
        self.tr_mask_paths = [self.mask1_addrs[i] for i in self.train_inds]
        self.val_img_paths = [self.combined_paths[i] for i in self.valid_inds]
        self.val_mask_paths = [self.mask1_addrs[i] for i in self.valid_inds]
        self.test_img_paths = [self.combined_paths[i] for i in self.test_inds]
        self.test_mask_paths = [self.mask1_addrs[i] for i in self.test_inds]

        if load_train_valid:
            self.tr_imgs  = [[] for i in range(ntrain)]
            self.tr_masks = [[] for i in range(ntrain)]
            for i,_ in enumerate(self.train_inds):
                for j in range(len(self.combined_paths[0])):
                    nii_dat = nib.load(self.tr_img_paths[i][j])
                    img = nii_dat.get_data()
                    self.tr_imgs[i] += [img]
                nii_dat = nib.load(self.tr_mask_paths[i])
                mask = nii_dat.get_data()
                self.tr_masks[i] = mask
            self.val_imgs  = [[] for i in range(len(self.valid_inds))]
            self.val_masks = [[] for i in range(len(self.valid_inds))]
            for i,_ in enumerate(self.valid_inds):
                for j in range(len(self.combined_paths[0])):
                    nii_dat = nib.load(self.val_img_paths[i][j])
                    img = nii_dat.get_data()
                    self.val_imgs[i] += [img]
                nii_dat = nib.load(self.val_mask_paths[i])
                mask = nii_dat.get_data()
                self.val_masks[i] = mask

    def create_train_valid_gens(self, 
                                batch_size, 
                                img_shape,
                                n_labeled_train=None):

        train_gen_inds = gen_minibatch_labeled_unlabeled_inds(
            self.L_indic, batch_size, n_labeled_train)
        train_gen = lambda: self.train_generator(
            train_gen_inds, img_shape, 'non-uniform', True)

        valid_gen_inds = gen_minibatch_labeled_unlabeled_inds(
            self.L_indic, batch_size)
        valid_gen = lambda: self.train_generator(
            valid_gen_inds, img_shape, 'uniform', False)

        self.train_gen_fn = train_gen
        self.valid_gen_fn = valid_gen

    def train_generator(self, generator,
                        img_shape, 
                        slice_choice='uniform', 
                        SeSu_phase=True):
    
        if hasattr(self, 'tr_imgs'):
            (img_paths_or_mats, 
             mask_paths_or_mats,
             L_indic) = gen_minibatch_materials(
                 generator, 
                 self.tr_imgs, 
                 self.tr_masks, 
                 self.L_indic)
        else:
            (img_paths_or_mats,
             mask_paths_or_mats,
             L_indic)= gen_minibatch_materials(
                 generator, 
                 self.tr_img_paths, 
                 self.tr_mask_paths,
                 self.L_indic)

        if not(SeSu_phase): L_indic=None
        return prepare_batch_BrVol(img_paths_or_mats, 
                                  mask_paths_or_mats, 
                                  img_shape, 
                                  self.C, slice_choice, 
                                  L_indic)

    def valid_generator(self, img_shape, batch_size=3, 
                       slice_choice='uniform'):
    
        if hasattr(self, 'tr_imgs'):
            (img_paths_or_mats, 
             mask_paths_or_mats) = gen_minibatch_materials(
                 generator, 
                 self.val_imgs, 
                 self.val_masks)
        else:
            (img_paths_or_mats,
             mask_paths_or_mats)= gen_minibatch_materials(
                 generator, 
                 self.tr_img_paths, 
                 self.tr_mask_paths)

        return prepare_batch_BrVol(img_paths_or_mats, 
                                  mask_paths_or_mats, 
                                  img_shape, 
                                  self.C, slice_choice)


    def test_generator(self, img_shape, batch_size=3, 
                       slice_choice='uniform'):
    
        for inds in gen_batch_inds(len(self.test_inds), batch_size):
            img_paths = [self.test_img_paths[i] for i in inds]
            mask_paths = [self.test_mask_paths[i] for i in inds]
            yield prepare_batch_BrVol(img_paths, mask_paths,
                                      img_shape, self.C, 
                                      slice_choice, None)

def combine_two_dats(dat_1, dat_2, 
                     load_train_valid=False):
    
    # create an empty data object
    # (since it's empty it does not matter what
    # function to use)
    dat = adols(0, 0, 0, 0, load_train_valid)
    
    # loading the new data with the union of variables 
    dat.L_indic = np.concatenate((dat_1.L_indic,
                                  dat_2.L_indic))
    dat.train_inds_1 = dat_1.train_inds
    dat.train_inds_2 = dat_2.train_inds
    dat.valid_inds_1 = dat_1.valid_inds
    dat.valid_inds_2 = dat_2.valid_inds
    dat.test_inds_1 = dat_1.test_inds
    dat.test_inds_2 = dat_2.test_inds

    dat.tr_img_paths = dat_1.tr_img_paths + dat_2.tr_img_paths
    dat.tr_mask_paths = dat_1.tr_mask_paths + dat_2.tr_mask_paths
    dat.val_img_paths = dat_1.val_img_paths + dat_2.val_img_paths
    dat.val_mask_paths = dat_1.val_mask_paths + dat_2.val_mask_paths
    dat.test_img_paths = dat_1.test_img_paths + dat_2.test_img_paths
    dat.test_mask_paths = dat_1.test_mask_paths + dat_2.test_mask_paths
    if load_train_valid:
        dat.tr_imgs = dat_1.tr_imgs + dat_2.tr_imgs
        dat.tr_masks = dat_1.tr_masks + dat_2.tr_masks
        dat.val_imgs = dat_1.val_imgs + dat_2.val_imgs
        dat.val_masks = dat_1.val_masks + dat_2.val_masks

    return dat
