import numpy as np
import nrrd
import pdb

from patch_utils import extract_Hakims_data_path
from .utils import gen_batch_inds, prepare_batch_BrVol

class adols(object):

    T1_addrs, T2_addrs, mask_addrs, _ = extract_Hakims_data_path()
    C = 2

    def __init__(self, rnd_seed, 
                 labeled_size, 
                 unlabeled_size,
                 valid_size,
                 load_train_valid=False):
        self.seed = rnd_seed

        n = len(self.T1_addrs)

        rand_inds = np.random.RandomState(seed=rnd_seed).permutation(len(self.T1_addrs))
        L_inds = rand_inds[:labeled_size]
        UL_inds = rand_inds[labeled_size : labeled_size+unlabeled_size]
        ntrain = labeled_size+unlabeled_size
        self.train_inds = np.concatenate((L_inds, UL_inds))
        self.L_indic = np.array([1]*len(L_inds) + [0]*len(UL_inds))

        self.valid_inds = rand_inds[ntrain : ntrain+valid_size]
        self.test_inds = set(np.arange(n)) - set(self.train_inds) - set(self.valid_inds)
        
        self.tr_img_paths = [[self.T1_addrs[i], self.T2_addrs[i]] for i in self.train_inds]
        self.tr_mask_paths = [self.mask_addrs[i] for i in self.train_inds]
        self.val_img_paths = [[self.T1_addrs[i], self.T2_addrs[i]] for i in self.valid_inds]
        self.val_mask_paths = [self.mask_addrs[i] for i in self.valid_inds]
        self.test_img_paths = [[self.T1_addrs[i], self.T2_addrs[i]] for i in self.test_inds]
        self.test_mask_paths = [self.mask_addrs[i] for i in self.test_inds]

        if load_train_valid:
            self.tr_imgs  = [[]] * ntrain
            self.tr_masks = [[]] * ntrain
            for i,_ in enumerate(self.train_inds):
                T1_img = nrrd.read(self.tr_img_paths[i][0])[0]
                T2_img = nrrd.read(self.tr_img_paths[i][1])[0]
                self.tr_imgs[i] = [T1_img, T2_img]
                mask = nrrd.read(self.tr_mask_paths[i])[0]
                self.tr_masks[i] = mask
            self.val_imgs  = [[]] * len(self.valid_inds)
            self.val_masks = [[]] * len(self.valid_inds)
            for i,_ in enumerate(self.valid_inds):
                T1_img = nrrd.read(self.val_img_paths[i][0])[0]
                T2_img = nrrd.read(self.val_img_paths[i][1])[0]
                self.val_imgs[i] = [T1_img, T2_img]
                mask = nrrd.read(self.val_mask_paths[i])[0]
                self.val_masks[i] = mask

    def train_generator(self, img_shape, batch_size=3, 
                        slice_choice='uniform', SeSu_phase=True):
    
        if slice_choice=='full':
            # TODO: extend this part to the case self.load_train_valid=True
            img_depths = []
            for path in self.tr_img_paths:
                img = nrrd.read(path[0])[0]
                img_depths += [img.shape[2]]
            train_slices = np.zeros((2, np.sum(img_depths)), dtype=int)
            cnt = 0
            for i in range(len(img_depths)):
                train_slices[0,cnt:cnt+img_depths[i]] = i
                train_slices[1,cnt:cnt+img_depths[i]] = np.arange(img_depths[i])
                cnt += img_depths[i]
            for inds in gen_batch_inds(np.sum(img_depths), batch_size):
                img_inds = train_slices[0,inds]
                slice_inds = train_slices[1,inds]
                img_paths = [self.tr_img_paths[i] for i in img_inds]
                mask_paths = [self.tr_mask_paths[i] for i in img_inds]
                L_indic = self.L_indic[img_inds] if SeSu_phase else None
                yield prepare_batch_BrVol(img_paths, mask_paths, img_shape,
                                          self.C, slice_inds, L_indic)
        else:
            for inds in gen_batch_inds(len(self.train_inds), batch_size):
                if hasattr(self, 'tr_imgs'):
                    img_paths_or_mats = [self.tr_imgs[i] for i in inds]
                    mask_paths_or_mats = [self.tr_masks[i] for i in inds]
                else:
                    img_paths_or_mats = [self.tr_img_paths[i] for i in inds]
                    mask_paths_or_mats = [self.tr_mask_paths[i] for i in inds]
                L_indic = self.L_indic[inds] if SeSu_phase else None
                yield prepare_batch_BrVol(img_paths_or_mats, 
                                          mask_paths_or_mats, 
                                          img_shape, 
                                          self.C, slice_choice, 
                                          L_indic)

    def valid_generator(self, img_shape, batch_size=3, 
                       slice_choice='uniform'):
    
        for inds in gen_batch_inds(len(self.valid_inds), batch_size):
            if hasattr(self, 'val_imgs'):
                img_paths_or_mats = [self.val_imgs[i] for i in inds]
                mask_paths_or_mats = [self.val_masks[i] for i in inds]
            else:
                img_paths_or_mats = [self.test_img_paths[i] for i in inds]
                mask_paths_or_mats = [self.test_mask_paths[i] for i in inds]
            yield prepare_batch_BrVol(img_paths_or_mats, 
                                      mask_paths_or_mats,
                                      img_shape, 
                                      self.C, 
                                      slice_choice, None)


    def test_generator(self, img_shape, batch_size=3, 
                       slice_choice='uniform'):
    
        for inds in gen_batch_inds(len(self.test_inds), batch_size):
            img_paths = [self.test_img_paths[i] for i in inds]
            mask_paths = [self.test_mask_paths[i] for i in inds]
            yield prepare_batch_BrVol(img_paths, mask_paths,
                                      img_shape, self.C, 
                                      slice_choice, None)
