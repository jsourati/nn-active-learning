import nibabel as nib
import numpy as np
import nrrd
import pdb

from .path_loader import extract_ISBI2015_MSLesion_data_path
from .utils import gen_minibatch_labeled_unlabeled_inds, \
    gen_minibatch_materials, prepare_batch_BrVol

class regular(object):

    C = 2

    def __init__(self,
                 img_addrs,
                 mask_addrs,
                 data_reader,
                 rnd_seed, 
                 labeled_size, 
                 unlabeled_size,
                 valid_size,
                 load_train_valid=False):

        self.seed = rnd_seed
        self.data_reader = data_reader
        self.mods = list(img_addrs.keys())
        self.combined_paths = [[img_addrs[mod][i] for mod in self.mods] 
                               for i in range(len(img_addrs[self.mods[0]]))]
        self.mask_addrs = mask_addrs
        n = len(self.combined_paths)

        rand_inds = np.random.RandomState(seed=rnd_seed).permutation(n)
        self.labeled_inds = rand_inds[:labeled_size]
        self.unlabeled_inds = rand_inds[labeled_size : 
                                        labeled_size+unlabeled_size]
        self.train_inds = np.concatenate((self.labeled_inds, 
                                          self.unlabeled_inds))
        ntrain = len(self.train_inds)
        self.L_indic = np.array([1]*len(self.labeled_inds) + \
                                [0]*len(self.unlabeled_inds))

        self.valid_inds = rand_inds[ntrain : ntrain+valid_size]
        self.test_inds = list(set(np.arange(n)) - 
                              set(self.train_inds) - 
                              set(self.valid_inds))
        
        self.tr_img_paths = [self.combined_paths[i] for i in self.train_inds]
        self.tr_mask_paths = [self.mask_addrs[i] for i in self.train_inds]
        self.val_img_paths = [self.combined_paths[i] for i in self.valid_inds]
        self.val_mask_paths = [self.mask_addrs[i] for i in self.valid_inds]
        self.test_img_paths = [self.combined_paths[i] for i in self.test_inds]
        self.test_mask_paths = [self.mask_addrs[i] for i in self.test_inds]

        if load_train_valid:
            self.tr_imgs  = [[] for i in range(ntrain)]
            self.tr_masks = [[] for i in range(ntrain)]
            for i,_ in enumerate(self.train_inds):
                for j in range(len(self.mods)):
                    img = self.data_reader(self.tr_img_paths[i][j])
                    self.tr_imgs[i] += [img]
                mask = self.data_reader(self.tr_mask_paths[i])
                self.tr_masks[i] = mask
            self.val_imgs  = [[] for i in range(len(self.valid_inds))]
            self.val_masks = [[] for i in range(len(self.valid_inds))]
            for i,_ in enumerate(self.valid_inds):
                for j in range(len(self.mods)):
                    img = self.data_reader(self.val_img_paths[i][j])
                    self.val_imgs[i] += [img]
                mask = self.data_reader(self.val_mask_paths[i])
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

    def combine_with_other_data(self, dat_2):

        self.L_indic = np.concatenate((self.L_indic, dat_2.L_indic))
        # storing indices of the other data in case we need
        self.train_inds_2 = dat_2.train_inds
        self.valid_inds_2 = dat_2.train_inds
        self.test_inds_2  = dat_2.test_inds
        self.labeled_inds_2   = dat_2.labeled_inds
        self.unlabeled_inds_2 = dat_2.unlabeled_inds

        self.tr_img_paths = self.tr_img_paths + dat_2.tr_img_paths
        self.tr_mask_paths = self.tr_mask_paths + dat_2.tr_mask_paths
        self.val_img_paths = self.val_img_paths + dat_2.val_img_paths
        self.val_mask_paths = self.val_mask_paths + dat_2.val_mask_paths
        self.test_img_paths = self.test_img_paths + dat_2.test_img_paths
        self.test_mask_paths = self.test_mask_paths + dat_2.test_mask_paths

        if hasattr(self, 'tr_imgs') and hasattr(dat_2, 'tr_imgs'):
            self.tr_imgs = self.tr_imgs + dat_2.tr_imgs
            self.tr_masks = self.tr_masks + dat_2.tr_masks
            self.val_imgs = self.val_imgs + dat_2.val_imgs
            self.val_masks = self.val_masks + dat_2.val_masks
