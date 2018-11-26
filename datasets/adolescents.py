import numpy as np

from patch_utils import extract_Hakims_data_path
from NN_extended import gen_batch_inds
from .utils import prepare_batch_BrVol

class adols(object):

    T1_addrs, T2_addrs, mask_addrs, _ = extract_Hakims_data_path()
    C = 2

    def __init__(self, rnd_seed, labeled_size, unlabeled_size):
        self.seed = rnd_seed

        rand_inds = np.random.RandomState(seed=rnd_seed).permutation(len(self.T1_addrs))
        L_inds = rand_inds[:labeled_size]
        UL_inds = rand_inds[labeled_size : labeled_size+unlabeled_size]
        self.train_inds = np.concatenate((L_inds, UL_inds))
        self.L_indic = np.array([1]*len(L_inds) + [0]*len(UL_inds))
        self.test_inds = set(np.arange(len(self.T1_addrs))) - set(self.train_inds)

        self.tr_img_paths = [[self.T1_addrs[i], self.T2_addrs[i]] for i in self.train_inds]
        self.tr_mask_paths = [self.mask_addrs[i] for i in self.train_inds]
        self.test_img_paths = [[self.T1_addrs[i], self.T2_addrs[i]] for i in self.test_inds]
        self.test_mask_paths = [self.mask_addrs[i] for i in self.test_inds]

    def train_generateor(self, img_shape, batch_size=3, 
                         slice_weight=True, SeSu_phase=True):
    
        for inds in gen_batch_inds(len(self.train_inds), batch_size):
            img_paths = [self.tr_img_paths[i] for i in inds]
            mask_paths = [self.tr_mask_paths[i] for i in inds]
            L_indic = self.L_indic[inds] if SeSu_phase else None
            yield prepare_batch_BrVol(img_paths, mask_paths,
                                      img_shape, self.C, 
                                      slice_weight, L_indic)

    def test_generateor(self, img_shape, batch_size=3, 
                        slice_weight=False):
    
        for inds in gen_batch_inds(len(self.test_inds), batch_size):
            img_paths = [self.test_img_paths[i] for i in inds]
            mask_paths = [self.test_mask_paths[i] for i in inds]
            yield prepare_batch_BrVol(img_paths, mask_paths,
                                      img_shape, self.C, 
                                      slice_weight, None)
