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
        posts = PW_AL.batch_eval_winds(
            expr,
            run,
            model, 
            pool_inds,
            'posteriors',
            sess)
        
        # vectories everything
        # uncertainty filtering
        B = expr.pars['B']
        if B < len(posts):
            sel_inds = np.argsort(
                np.abs(posts-.5))[:B]
            sel_posts = posts[sel_inds]
        else:
            B = posts.shape[1]
            sel_posts = posts
            sel_inds = np.arange(B)
            
        n = len(pool_inds)
        rem_inds = list(set(np.arange(n)) - 
                        set(sel_inds))
        
        # extract the features for all the pool
        # sel_inds, rem_inds  -->  pool_inds
        F = PW_AL.batch_eval_winds(
            expr,
            run,
            model,
            pool_inds,
            'feature_layer',
            sess)

        F_uncertain = F[:, sel_inds]
        norms_uncertain = np.sqrt(np.sum(F_uncertain**2, axis=0))
        F_rem_pool = F[:, rem_inds]
        norms_rem = np.sqrt(np.sum(F_rem_pool**2, axis=0))
        
        # compute cos-similarities between filtered images
        # and the rest of the unlabeled samples
        dots = np.dot(F_rem_pool.T, F_uncertain)
        norms_outer = np.outer(norms_rem, norms_uncertain)
        sims = dots / norms_outer
            
        print("Greedy optimization..", end='\n\t')
        # start from empty set
        Q_inds = []
        nQ_inds = np.arange(B)
        # add most representative samples one by one
        for i in range(expr.pars['k']):
            rep_scores = np.zeros(B-i)
            for j in range(B-i):
                cand_Q = Q_inds + [nQ_inds[j]]
                rep_scores[j] = np.sum(
                    np.max(sims[:, cand_Q], axis=1))
            iter_sel = nQ_inds[np.argmax(rep_scores)]
            # update the iterating sets
            Q_inds += [iter_sel]
            nQ_inds = np.delete(
                nQ_inds, np.argmax(rep_scores))
            
        q = sel_inds[Q_inds]

    if method_name=='fi':
        # posteriors
        posts = PW_AL.batch_eval_winds(
            expr,
            run,
            model, 
            pool_inds,
            'posteriors',
            sess)
        
        # vectories everything
        # uncertainty filtering
        B = expr.pars['B']
        if B < len(posts):
            sel_inds = np.argsort(
                np.abs(posts-.5))[:B]
            sel_posts = posts[sel_inds]
        else:
            B = posts.shape[1]
            sel_posts = posts
            sel_inds = np.arange(B)

        # forming A-matrices
        # ------------------
        # division by two in computing size of A is because 
        # in each layer we have gradients with respect to
        # weights and bias terms --> number of layers that
        # are considered is obtained after dividing by 2
        A_size = int(
            len(model.grad_posts['0'])/2)
        n = len(posts)
        c = expr.nclass

        A = []
        # load the patches
        # indices: sel_inds --> pool_inds
        # CAUTIOUS: this will give an error if 
        # the selected indices in `sel_inds`
        # contains only one index.
        sel_patches = PW_AL.load_patches(
            expr, run, pool_inds[sel_inds])
            
        for i in range(B):
            X_i = sel_patches[i,:,:,:]
            feed_dict = {
                model.x: np.expand_dims(X_i,axis=0),
                model.keep_prob: 1.}

            # preparing the poserior
            # (sel_posts contains single posterior
            # probability of being masked; we need
            # full posterior array here)
            x_post = np.array([1-sel_posts[i], 
                      sel_posts[i]])
            # remove zero, or close-to-zero posteriors
            x_post[x_post<1e-6] = 0.
            nz_classes = np.where(x_post > 0.)[0]
            nz_posts = x_post[nz_classes] / np.sum(
                x_post[nz_classes])
            # considering only gradients of classes 
            # with non-zero posteriors
            nz_classes_grads = {
                str(cc): model.grad_posts[str(cc)]
                for cc in nz_classes}

            # ASSUMOTION: we have few classes here
            # (probably only two)
            grads = sess.run(nz_classes_grads,
                             feed_dict=feed_dict)
            sel_classes = nz_classes
            new_posts = nz_posts

            # the A-matrix
            Ai = np.zeros((A_size, A_size))
            for j in range(len(sel_classes)):
                shrunk_grad = NNAL_tools.shrink_gradient(
                    grads[str(sel_classes[j])], 'sum')
                Ai += np.outer(shrunk_grad, 
                               shrunk_grad) / new_posts[j] \
                    + np.eye(A_size)*1e-5
            A += [Ai]

            if not(i%10):
                print(i, end=',')
            
        # extracting features for pool samples
        # using only few indices of the features
        F = PW_AL.batch_eval_winds(expr,
                                   run,
                                   model,
                                   pool_inds[sel_inds],
                                   'feature_layer',
                                   sess)
        # selecting from those features that have the most
        # non-zero values among the selected samples
        nnz_feats = np.sum(F>0, axis=1)
        feat_inds = np.argsort(-nnz_feats)[:int(B/2)]
        F_sel = F[feat_inds,:]
        # taking care of the rank
        while np.linalg.matrix_rank(F_sel)<len(feat_inds):
            # if the matrix is not full row-rank, discard
            # the last selected index (worst among all)
            feat_inds = feat_inds[:-1]
            F_sel = F[feat_inds,:]
                
        # taking care of the conditional number
        lambda_ = expr.pars['lambda_']
        while np.linalg.cond(F_sel) > 1e6:
            feat_inds = feat_inds[:-1]
            F_sel = F[feat_inds,:]
            if len(feat_inds)==1:
                lambda_=0
                print('Only one feature is selected.')
                break
        
        # subtracting the mean
        F_sel -= np.repeat(np.expand_dims(
            np.mean(F_sel, axis=1),
            axis=1), B, axis=1)
        
        print('Cond. #: %f'% (np.linalg.cond(F_sel)),
              end='\n\t')
        print('# selected features: %d'% 
              (len(feat_inds)), end='\n\t')
        
        # SDP
        # ----
        soln = NNAL_tools.SDP_query_distribution(
            A, lambda_, F_sel, expr.pars['k'])
        print('status: %s'% (soln['status']), end='\n\t')
        q_opt = np.array(soln['x'][:B])
        
        # sampling from the optimal solution
        Q_inds = NNAL_tools.sample_query_dstr(
            q_opt, expr.pars['k'], 
            replacement=True)
        q = sel_inds[Q_inds]

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
    
