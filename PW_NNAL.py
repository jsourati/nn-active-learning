import tensorflow as tf
import numpy as np
import warnings
import nibabel
import nrrd
import pdb
import os

import NNAL_tools
import PW_NN
import patch_utils


def CNN_query(model,
              pool_dict,
              method_name,
              qbatch_size,
              patch_shape,
              stats,
              sess):
    """Querying strategies for active
    learning of patch-wise model
    """
    
    if method_name=='random':
        n = np.sum([
            len(pool_dict[path]) 
            for path in 
            list(pool_dict.keys())])
        q = np.random.permutation(n)[
            :qbatch_size]

        q_dict = patch_utils.locate_in_dict(
            pool_dict, q)
        
    if method_name=='entropy':
        # posteriors
        posts = PW_NN.batch_eval(
            model, 
            pool_dict,
            patch_shape,
            5000,
            stats,
            sess,
            'posteriors')[0]
        
        # vectories everything
        ttposts = []
        for path in list(posts.keys()):
            ttposts += list(posts[path])
            
        # k most uncertain (binary classes)
        q = np.argsort(np.abs(np.array(
            ttposts)-.5))[:qbatch_size]
        
        q_dict = patch_utils.locate_in_dict(
            pool_dict, q)
        
    return q_dict

