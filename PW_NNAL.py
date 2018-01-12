import tensorflow as tf
import numpy as np
import warnings
import nibabel
import nrrd
import pdb
import os

import NNAL_tools
import PW_NN
import PW_AL
import patch_utils


def CNN_query(expr,
              run,
              model,
              pool_inds,
              method_name,
              sess):
    """Querying strategies for active
    learning of patch-wise model
    """

    if method_name=='random':
        n = len(pool_inds)
        q = np.random.permutation(n)[
            :expr.pars['k']]

    if method_name=='entropy':
        # posteriors
        posts = PW_AL.batch_eval_winds(
            expr,
            run,
            model,
            pool_inds,
            'posteriors',
            sess)
        
        # k most uncertain (binary classes)
        q = np.argsort(np.abs(posts-.5))[
            :expr.pars['k']]
        
    if method_name=='rep-entropy':
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
        
        B = 1000
        if B < len(ttposts):
            sel_inds = binary_uncertainty_filter(
                ttposts, B)
            sel_ttposts = ttposts[sel_inds]
        else:
            B = len(ttposts)
            sel_ttposts = ttposts
            sel_inds = np.arange(B)
            
        n = len(ttposts)
        rem_inds = list(set(np.arange(n)) - set(sel_inds))
        
        # extract the features for all the pool
        # sel_inds, rem_inds  -->  pool_inds
        F = model.extract_features(pool_inds,
                                   expr,
                                   session)
        
    return q


def binary_uncertainty_filter(posts, B):
    """Uncertainty filtering for binary class
    label distribution
    
    Since there are only two classes, posterior
    probability of only one of the classes
    are given in form of 1D array.
    """
    
    return np.argsort(np.abs(
        np.array(posts)-0.5))[:B]

def extract_features(model,
                     pool_dict, 
                     inds,
                     stats,
                     sess):
    """Extracting features for some patches
    that are indexed from within a dictionary
    """
    
    # make a sub-dictionary for given indices
    inds_dict = patch_utils.locate_in_dict(
            pool_dict, inds)
    sub_dict = {}
    for path in list(inds_dict.keys()):
        sub_dict[path] = pool_dict[path][
            inds_dict[path]]
        
    # start computing the features
    batches = patch_utils.get_batches(
        sub_dict,1000)
    
