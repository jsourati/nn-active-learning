import nibabel as nib
import numpy as np
import nrrd
import pdb

from .path_loader import extract_ISBI2015_MSLesion_data_path
from .utils import gen_minibatch_labeled_unlabeled_inds, \
    gen_minibatch_materials, global2local_inds, prepare_batch_BrVol

class regular(object):

    def __init__(self,
                 img_addrs,
                 mask_addrs,
                 data_reader,
                 rnd_seed, 
                 LUV_inds_or_sizes,
                 class_labels):
        """
        * LUV = Labeled + Unlabeled + Validation
          This list characterizes sample indices for these three
          subsets. It can be either a list of arrays which explicitly
          specifies indices of each subset, or a list of integers that
          only includes their sizes. 

              inds:   LUV_inds_or_sizes = [L_inds, U_inds, V_inds]
                                           ------  ------  ------
                                            array   array   array

              sizes:  LUV_inds_or_sizes = [L_size, U_size, V_size]
                                           ------  ------  ------
                                             int     int    int

          When this input has arrays of indices, the input "rnd_seed"
          won't be used in the constructor at all.

          All the remaining indices will be assigned to test
          data set, i.e. 
              test_inds = {1..n} - {inds of L} - {inds of U} - {inds of V}   
        """

        self.class_labels = class_labels
        self.C = len(self.class_labels)
        self.seed = rnd_seed
        self.reader = data_reader
        self.img_addrs = img_addrs
        self.mask_addrs = mask_addrs
        self.mods = list(img_addrs.keys())
        self.combined_paths = [[img_addrs[mod][i] for mod in self.mods] 
                               for i in range(len(img_addrs[self.mods[0]]))]
        n = len(self.combined_paths)

        if isinstance(LUV_inds_or_sizes[0], np.ndarray):
            self.labeled_inds   = LUV_inds_or_sizes[0]
            self.unlabeled_inds = LUV_inds_or_sizes[1]
            self.valid_inds     = LUV_inds_or_sizes[2]
            self.train_inds = np.concatenate((self.labeled_inds,
                                              self.unlabeled_inds))
        else:
            rand_inds = np.random.RandomState(seed=rnd_seed).permutation(n)
            self.labeled_inds = rand_inds[:LUV_inds_or_sizes[0]]
            self.unlabeled_inds = rand_inds[LUV_inds_or_sizes[0] : 
                                            LUV_inds_or_sizes[0]+LUV_inds_or_sizes[1]]
            self.train_inds = np.concatenate((self.labeled_inds, 
                                              self.unlabeled_inds))
            ntrain = len(self.train_inds)
            self.valid_inds = rand_inds[ntrain : ntrain+LUV_inds_or_sizes[2]]
        

        self.L_indic = np.array([1]*len(self.labeled_inds) + \
                                [0]*len(self.unlabeled_inds))

        self.test_inds = np.array(list(set(np.arange(n)) - 
                                       set(self.train_inds) - 
                                       set(self.valid_inds)))
        
        self.tr_img_paths = [self.combined_paths[i] for i in self.train_inds]
        self.tr_mask_paths = [self.mask_addrs[i] for i in self.train_inds]
        self.val_img_paths = [self.combined_paths[i] for i in self.valid_inds]
        self.val_mask_paths = [self.mask_addrs[i] for i in self.valid_inds]
        self.test_img_paths = [self.combined_paths[i] for i in self.test_inds]
        self.test_mask_paths = [self.mask_addrs[i] for i in self.test_inds]

    def load_images(self):
        """Loading images of the training and validation
        partitions into memory
        """

        ntrain = len(self.train_inds)
        self.tr_imgs  = [[] for i in range(ntrain)]
        self.tr_masks = [[] for i in range(ntrain)]
        for i in range(len(self.tr_img_paths)):
            for j in range(len(self.mods)):
                img = self.reader(self.tr_img_paths[i][j])
                self.tr_imgs[i] += [img]
            if self.tr_mask_paths[i]=='NA':
                # if a training sample does not have any mask path
                # put an all-zero mask in its place, just to keep
                # everything consistent, otherwise it will never be used
                # as this sample should be unlabeled 
                mask = np.zeros(img.shape)
            else:
                mask = self.reader(self.tr_mask_paths[i])
            self.tr_masks[i] = mask
            # ensuring that the masks have values 1,...,c
            if np.any(self.class_labels != np.arange(self.C)):
                self.tr_masks[i] = np.zeros(mask.shape)
                for c, label in enumerate(self.class_labels):
                    self.tr_masks[i][mask==label] = c

        self.val_imgs  = [[] for i in range(len(self.valid_inds))]
        self.val_masks = [[] for i in range(len(self.valid_inds))]
        for i,_ in enumerate(self.valid_inds):
            for j in range(len(self.mods)):
                img = self.reader(self.val_img_paths[i][j])
                self.val_imgs[i] += [img]
            mask = self.reader(self.val_mask_paths[i])
            self.val_masks[i] = mask
            # ensuring that the masks have values 1,...,c
            if np.any(self.class_labels != np.arange(self.C)):
                self.val_masks[i] = np.zeros(mask.shape)
                for c, label in enumerate(self.class_labels):
                    self.val_masks[i][mask==label] = c

    def create_train_valid_gens(self, 
                                batch_size, 
                                img_shape,
                                n_labeled_train=None):
        """Creating sample generator from training and validation
        data sets. 

        It is assumed that images are loaed through load_images().
        """

        # training
        if len(self.tr_masks)>0:
            self.train_n_slices = [self.tr_masks[i].shape[2] 
                                   for i in range(len(self.tr_masks))]
            self.slices_L_indic = np.concatenate(
                [np.ones(self.train_n_slices[i])*self.L_indic[i] 
                 for i in range(len(self.tr_masks))])
            train_generator = gen_minibatch_labeled_unlabeled_inds(
                self.slices_L_indic, batch_size, n_labeled_train)
            train_gen_slices = lambda: self.generate_training_stuff(
                img_shape, train_generator)

            self.train_gen_fn = train_gen_slices

        # validation
        if len(self.val_masks)>0:
            valid_gen_inds = gen_minibatch_labeled_unlabeled_inds(
                np.ones(len(self.val_imgs)), batch_size)
            valid_gen = lambda: self.valid_generator(
                valid_gen_inds, img_shape, 'uniform')

            self.valid_gen_fn = valid_gen

    def generate_training_stuff(self,
                                img_shape, 
                                inds_generator):

        inds = np.concatenate(next(inds_generator))
        # extracting slice indices from the generated indices
        img_slice_inds = global2local_inds(inds, self.train_n_slices)
        img_inds = np.concatenate([
            np.ones(len(img_slice_inds[i]),dtype=int)*i 
            for i in range(len(img_slice_inds))])
        img_slice_inds = np.concatenate(img_slice_inds)
        imgs = [self.tr_imgs[int(i)] for i in img_inds]
        masks = [self.tr_masks[int(i)] for i in img_inds]
        if np.any(self.L_indic==0):
            inds_L_indic = np.ones(len(img_inds))
            inds_L_indic[self.L_indic[img_inds]==0] = 0
        else:
            inds_L_indic=None

        return prepare_batch_BrVol(
            imgs, masks, img_shape, self.C, img_slice_inds, inds_L_indic)


    def valid_generator(self, generator, 
                        img_shape,
                        slice_choice='uniform'):
    
        if hasattr(self, 'val_imgs'):
            (img_paths_or_mats, 
             mask_paths_or_mats) = gen_minibatch_materials(
                 generator, 
                 self.val_imgs, 
                 self.val_masks)
        else:
            (img_paths_or_mats,
             mask_paths_or_mats) = gen_minibatch_materials(
                 generator, 
                 self.val_img_paths, 
                 self.val_mask_paths)

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

        assert self.mods==dat_2.mods, 'The combining data sets should have '+\
            'the same image modalities.'

        # combining everything
        for mod in self.mods:
            self.img_addrs[mod] += dat_2.img_addrs[mod]
        self.mask_addrs += dat_2.mask_addrs
        self.L_indic = np.concatenate((self.L_indic, dat_2.L_indic))
        # .. including the combined_paths
        self.train_inds = np.concatenate((self.train_inds,
                                         dat_2.train_inds+len(self.combined_paths)))
        self.valid_inds = np.concatenate((self.valid_inds,
                                         dat_2.valid_inds+len(self.combined_paths)))
        self.test_inds = np.concatenate((self.test_inds,
                                         dat_2.test_inds+len(self.combined_paths)))
        self.combined_paths += dat_2.combined_paths
        
        self.tr_img_paths = self.tr_img_paths + dat_2.tr_img_paths
        self.tr_mask_paths = self.tr_mask_paths + dat_2.tr_mask_paths
        self.val_img_paths = self.val_img_paths + dat_2.val_img_paths
        self.val_mask_paths = self.val_mask_paths + dat_2.val_mask_paths
        self.test_img_paths = self.test_img_paths + dat_2.test_img_paths
        self.test_mask_paths = self.test_mask_paths + dat_2.test_mask_paths

        # .. in case images are loaded
        if hasattr(self, 'tr_imgs') and hasattr(dat_2, 'tr_imgs'):
            self.tr_imgs = self.tr_imgs + dat_2.tr_imgs
            self.tr_masks = self.tr_masks + dat_2.tr_masks
            self.val_imgs = self.val_imgs + dat_2.val_imgs
            self.val_masks = self.val_masks + dat_2.val_masks
