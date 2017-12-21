import tensorflow as tf
import numpy as np
import warnings
import nibabel
import nrrd
import pdb
import os

import NNAL_tools
import patch_utils


def CNN_query(model,
              pool_dict,
              method_name,
              qbatch_size,
              session):
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
        
    return q_dict

