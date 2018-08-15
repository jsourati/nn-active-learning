from skimage.measure import regionprops
import tensorflow as tf
import numpy as np
import warnings
#import nibabel
import nrrd
import pdb
import os

import NN
import PW_NN
import PW_AL
import NNAL_tools
import patch_utils


def CNN_query(expr,
              model,
              sess,
              padded_imgs,
              pool_inds,
              tr_inds,
              method_name):
    """Querying strategies for active
    learning of patch-wise model

     Although the given image is padded, 
    the indices are given in terms of the
    original dimensionalities
    """

    if method_name=='random':
        n = len(pool_inds)
        q = np.random.permutation(n)[
            :expr.pars['k']]

    if method_name=="ps-random":
        # pseudo-random: randomly
        # selecting queries from regions
        # with high local variance
        
        thr = 2.   # threhsold over the variance
        valid_pool_inds = get_HV_inds(
            padded_imgs[0], exp.pars['patch_shape'],
            thr, pool_inds)
        rand_inds = np.random.permutation(
            len(valid_pool_inds))[:expr.pars['k']]
        q = valid_pool_inds[rand_inds]

    if method_name=='entropy':
        # posteriors
        posts = PW_NN.batch_eval(
            model,
            sess,
            padded_imgs,
            pool_inds,
            expr.pars['patch_shape'],
            expr.pars['ntb'],
            expr.pars['stats'],
            'posteriors')[0]
        
        # k most uncertain (binary classes)
        q = np.argsort(np.abs(posts-.5))[
            :expr.pars['k']]

    if method_name=='MC-entropy':
        x_feed_dict = {model.keep_prob:
                       model.dropout_rate}
        # iterative averaging over MC iterations
        total_posts = 0
        for i in range(expr.pars['MC_iters']):
            posts = PW_NN.batch_eval(
                model,
                sess,
                padded_imgs,
                pool_inds,
                expr.pars['patch_shape'],
                expr.pars['ntb'],
                expr.pars['stats'],
                'posteriors',
                x_feed_dict)[0]
            total_posts = (posts+i*total_posts)/(i+1)
        
        # k most uncertain (binary classes)
        q = np.argsort(np.abs(total_posts-.5))[
            :expr.pars['k']]
        
    if method_name=='fi':
        n = len(pool_inds)
        m = len(expr.pars['img_paths'])
        B = expr.pars['B']
        lambda_ = expr.pars['lambda_']

        # posteriors
        posts = PW_NN.batch_eval(
            model,
            sess,
            padded_imgs,
            pool_inds,
            expr.pars['patch_shape'],
            expr.pars['ntb'],
            expr.pars['stats'],
            'posteriors')[0]
        
        # vectories everything
        # uncertainty filtering
        if B < len(pool_inds):
            sel_inds = np.argsort(
                np.abs(posts-.5))[:B]
            sel_posts = posts[sel_inds]
        else:
            B = posts.shape[1]
            sel_posts = posts
            sel_inds = np.arange(B)

        # load the patches
        # indices: sel_inds --> pool_inds
        # CAUTIOUS: this will give an error if 
        # the selected indices in `sel_inds`
        # contains only one index.
        sel_patches = patch_utils.get_patches(
            padded_imgs, pool_inds[sel_inds],
            expr.pars['patch_shape'])
        for j in range(m):
            sel_patches[:,:,:,j] = (
                sel_patches[:,:,:,j]-
                expr.pars['stats'][j][0])/\
                expr.pars['stats'][j][1]

        # get the A-matrices (conditional FI's)
        A = gen_A_matrices(expr, 
                           model,
                           sess,
                           sel_patches,
                           sel_posts)
        # prepare the feature vectors
        F = PW_NN.batch_eval(model,
                             sess,
                             padded_imgs,
                             pool_inds[sel_inds],
                             expr.pars['patch_shape'],
                             expr.pars['ntb'],
                             expr.pars['stats'],
                             'feature_layer')[0]
        ref_F = refine_feature_matrix(F, B)
        # make the feature components zero-mean
        ref_F -= np.repeat(np.expand_dims(
            np.mean(ref_F, axis=1),
            axis=1), F.shape[1], axis=1)

        # SDP
        # ----
        soln = NNAL_tools.SDP_query_distribution(
            A, lambda_, ref_F, expr.pars['k'])
        print('status: %s'% (soln['status']), end='\n\t')
        q_opt = np.array(soln['x'][:F.shape[1]])
        
        # sampling from the optimal solution
        Q_inds = NNAL_tools.sample_query_dstr(
            q_opt, expr.pars['k'], 
            replacement=True)
        q = sel_inds[Q_inds]


    return q


def query_multimg(expr,
                  model,
                  sess,
                  all_padded_imgs,
                  pool_inds,
                  labeled_inds,
                  method_name):
    """Similar to (single image) query except
    `all_padded_imgs` contains multiple images,
    hence `pool_inds` and `tr_inds` include a 
    sequence of lists each of which contains 
    training indices associated with a training
    subject

    The output will be a list of N sets of indices
    (N as the number of training subjects), where
    each set (possibly empty) includes the indices
    of the queries chosen from that subject
    """

    k = expr.pars['k']
    B = expr.pars['B']
    img_ind_sizes = [len(pool_inds[i]) for 
                     i in range(len(pool_inds))]
    n = np.sum(img_ind_sizes)
    m = len(all_padded_imgs[0]) - 1
    
    if method_name=='random':
        inds_num = [len(pool_inds[i]) for
                    i in range(len(pool_inds))]
        npool = np.sum(inds_num)
        inds = np.random.permutation(npool)[:k]
        
        Q_inds = patch_utils.global2local_inds(
            inds, inds_num)

    if method_name=='ps-random':

        thr = 2.   # threhsold over the variance

        valid_pool_inds = []
        for i in range(len(all_padded_imgs)):
            valid_pool_inds += [get_HV_inds(
                all_padded_imgs[i][0], expr.pars['patch_shape'],
                thr, pool_inds[i])]

        valid_inds_sizes = [len(valid_pool_inds[i]) for
                            i in range(len(valid_pool_inds))]
        nHV = np.sum(valid_inds_sizes)

        rand_inds = np.random.permutation(nHV)[:k]
        local_inds = patch_utils.global2local_inds(
            rand_inds, valid_inds_sizes)

        Q_inds = [valid_pool_inds[i][local_inds[i]] for
                  i in range(len(valid_pool_inds))]

    if method_name=='entropy':
        
        Q_inds = bin_uncertainty_filter_multimg(
            expr, model, sess, all_padded_imgs,
            pool_inds, k)[0]

    if method_name=='MC-entropy':
        
        x_feed_dict = {model.keep_prob:
                       model.dropout_rate}
        av_posts = 0
        for i in range(expr.pars['MC_iters']):
            # the argument `k` won't be really used
            # in this line
            posts = bin_uncertainty_filter_multimg(
                expr, model, sess, all_padded_imgs,
                pool_inds, k, x_feed_dict)
            av_posts = (posts+i*av_posts)/(i+1)

        inds = np.argsort(np.abs(av_posts-.5))[:k]

        Q_inds = patch_utils.global2local_inds(
            inds, img_ind_sizes)

    if method_name=='BALD':
        x_feed_dict = {model.keep_prob:
                       model.dropout_rate}
        av_posts = 0
        av_ents  = 0
        for i in range(expr.pars['MC_iters']):
            # the argument `k` won't be really used
            # in this line
            posts = bin_uncertainty_filter_multimg(
                expr, model, sess, all_padded_imgs,
                pool_inds, k, x_feed_dict)
            av_posts = (posts+i*av_posts)/(i+1)

            neg_posts = 1-posts
            posts[posts==0] += 1e-6
            neg_posts[neg_posts==0] += 1e-6
            ents = -posts*np.log(posts) -\
                  neg_posts*np.log(neg_posts)
            # average entropies
            av_ents = (ents+i*av_ents)/(i+1)
            
        # entropy of average posteriors
        av_neg_posts = 1-av_posts
        av_posts[av_posts==0] += 1e-6
        av_neg_posts[av_neg_posts==0] += 1e-6
        ent_av_posts = -av_posts*np.log(av_posts)-\
                       av_neg_posts*np.log(av_neg_posts)

        scores = ent_av_posts - av_ents
        inds = np.argsort(scores)[:k]

        Q_inds = patch_utils.global2local_inds(
            inds, img_ind_sizes)

    if method_name=='rep-entropy':

        # extracting features
        F = [[] for i in range(len(pool_inds))]
        for i in range(len(pool_inds)):
            stats = []
            for j in range(m):
                stats += [[expr.train_stats[i,2*j],
                           expr.train_stats[i,2*j+1]]]

            F[i] = PW_NN.batch_eval(
                model,sess,
                all_padded_imgs[i][:-1],
                pool_inds[i],
                expr.pars['patch_shape'],
                expr.pars['ntb'],
                stats,
                'feature_layer')[0]

        # get the most uncertain samples
        sel_inds,sel_posts = bin_uncertainty_filter_multimg(
            expr, model, sess, all_padded_imgs,
            pool_inds, B)

        F_uncertain = [F[i][:,sel_inds[i]] for i in
                       range(len(sel_inds)) if 
                       len(sel_inds[i])>0]
        F_uncertain = np.concatenate(F_uncertain, axis=1)
        for i in range(len(pool_inds)):
            rem_inds = list(set(np.arange(len(pool_inds[i]))) - 
                            set(sel_inds[i]))
            F[i] = F[i][:, rem_inds]
        F = np.concatenate(F, axis=1)

        # norms
        norms_rem = np.sqrt(np.sum(F**2, axis=0))
        norms_uncertain = np.sqrt(np.sum(F_uncertain**2, 
                                         axis=0))
        # compute cos-similarities between filtered images
        # and the rest of the unlabeled samples
        dots = np.dot(F.T, F_uncertain)
        norms_outer = np.outer(norms_rem, norms_uncertain)
        sims = dots / norms_outer
        del dots, norms_rem, norms_uncertain, norms_outer

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
        
        # transforming global Q_inds into local one
        img_ind_sizes = [len(sel_inds[i]) for i
                         in range(len(sel_inds))]
        local_inds = patch_utils.global2local_inds(
            Q_inds, img_ind_sizes)
        Q_inds = [np.array(sel_inds[i])[local_inds[i]]
                  for i in range(len(sel_inds))]

    if method_name=='core-set':
        # getting the feature matrices
        # form full feature matrix of unlabeled pool
        # because we need to have them in all iterations
        F_u = [[] for i in range(len(pool_inds))]
        for i in range(len(pool_inds)):
            stats = []
            for j in range(m):
                stats += [[expr.train_stats[i,2*j],
                           expr.train_stats[i,2*j+1]]]

            F_u[i] = PW_NN.batch_eval(
                model,sess,
                all_padded_imgs[i][:-1],
                pool_inds[i],
                expr.pars['patch_shape'],
                expr.pars['ntb'],
                stats,
                'feature_layer')[0]

        F_u = np.concatenate(F_u, axis=1)
        n = F_u.shape[1]
        norms_u = np.sqrt(np.sum(F_u**2, axis=0))
        sims = -np.inf*np.ones(n)

        nT = np.sum([len(labeled_inds[i]) for i 
                     in range(len(labeled_inds))])
        if True:

            # for labeled data, do not need to keep
            # the features in memory, since we only
            # need to have the max-similarities
            for i in range(len(labeled_inds)):
                labeled_stats = []
                for j in range(m):
                    labeled_stats += [
                        [expr.labeled_stats[i,2*j],
                         expr.labeled_stats[i,2*j+1]]]

            # this extra-batching is because of
            # memory issues;F_u and F_T are too
            # large to keep in the memory at th
            # same time
            nT = len(labeled_inds[i])
            batches = NN.gen_batch_inds(nT,1000)

            for batch_inds in batches:
                if expr.labeled_paths==expr.train_paths:
                    F_T = PW_NN.batch_eval(
                        model, sess,
                        all_padded_imgs[i][:-1],
                        np.array(labeled_inds[i])[batch_inds],
                        expr.pars['patch_shape'],
                        expr.pars['ntb'],
                        labeled_stats,
                        'feature_layer')[0]
                else:
                    F_T = PW_NN.batch_eval(
                        model, sess,
                        expr.labeled_paths[i][:-1],
                        np.array(labeled_inds[i])[batch_inds],
                        expr.pars['patch_shape'],
                        expr.pars['ntb'],
                        labeled_stats,
                        'feature_layer')[0]
                    
                norms_T = np.sqrt(np.sum(F_T**2, axis=0))
                # cosine similarities
                dots = np.dot(F_T.T, F_u)
                norms_outer = np.outer(norms_T, norms_u)
                sims = np.max(np.concatenate((
                    dots / norms_outer,
                    np.expand_dims(sims, axis=0)), axis=0), 
                              axis=0)

                del dots, norms_T, norms_outer

            if nT>2000:
                np.savetxt(os.path.join(
                    expr.root_dir,'core-set/UT_sims.txt'), sims)
        else:
            sims = np.loadtxt(os.path.join(
                expr.root_dir,'core-set/UT_sims.txt'))

        Q_inds = []
        for t in range(k):
            q_ind = np.argmin(sims)
            Q_inds += [q_ind]
            # computing the similarities between the 
            # selected sample and rest of the pool
            s_ind = np.dot(F_u[:,q_ind].T, F_u)/\
                (norms_u*norms_u[q_ind])
            sims = np.maximum(sims, s_ind)
            # put inf in place of selected index as an 
            # indicator that it should be ignored  
            sims[q_ind] = np.inf

        Q_inds = patch_utils.global2local_inds(
            Q_inds, img_ind_sizes)

    if method_name=='ensemble':
        
        n_labels = np.sum([len(labeled_inds[i]) for
                           i in range(len(labeled_inds))])

        av_posts = 0
        x_feed_dict = {expr.model_holder.keep_prob: 1.}
        for i in range(len(expr.pretrained_paths)):
            if n_labels==0:
                # if no labeled indices, go for ensemble
                # of pre-trained models
                expr.model_holder.perform_assign_ops(
                    expr.pretrained_paths[i], sess)
            else:
                # otherwise, create the ensemble by
                # fine-tuning the previous model multiple
                # times
                PW_AL.finetune_multimg(expr,
                                       expr.model_holder, 
                                       sess,
                                       all_padded_imgs,
                                       labeled_inds)

            # compute posteriors with the current model
            # of the ensemble
            posts = bin_uncertainty_filter_multimg(
                expr, expr.model_holder, sess, 
                all_padded_imgs,
                pool_inds, k, x_feed_dict)
            av_posts = (posts+i*av_posts)/(i+1)

        pdb.set_trace()
        # sorting w.r.t uncertainty
        inds = np.argsort(np.abs(av_posts-.5))[:k]

        Q_inds = patch_utils.global2local_inds(
            inds, img_ind_sizes)


    if method_name=='fi':
        # uncertainty-filtering
        sel_inds,sel_posts = bin_uncertainty_filter_multimg(
            expr, model, sess, all_padded_imgs,
            pool_inds, B)

        # loading patches
        img_inds = [np.array(pool_inds[i])[sel_inds[i]]
                    for i in range(len(pool_inds))]
        patches,_ = patch_utils.get_patches_multimg(
            all_padded_imgs, img_inds,
            expr.pars['patch_shape'], 
            expr.train_stats)

        # form A-matrices and extract features
        A = []
        F = [[] for i in range(len(patches))]
        for i in range(len(patches)):
            if len(img_inds[i])==0:
                continue

            stats = []
            for j in range(m):
                stats += [[expr.train_stats[i,2*j],
                           expr.train_stats[i,2*j+1]]]

            A += gen_A_matrices(expr,
                                model,
                                sess,
                                patches[i],
                                sel_posts[i],
                                1e-3)

            #F[i] = PW_NN.batch_eval(
            #    model,sess,
            #    all_padded_imgs[i][:-1],
            #    img_inds[i],
            #    expr.pars['patch_shape'],
            #    expr.pars['ntb'],
            #    stats,
            #    'feature_layer')[0]

        #F = [F[i] for i in range(len(F)) if len(F[i])>0]
        #F = np.concatenate(F, axis=1)
        #ref_F = refine_feature_matrix(F, B)
        # make the feature components zero-mean
        #ref_F -= np.repeat(np.expand_dims(
        #    np.mean(ref_F, axis=1),
        #    axis=1), F.shape[1], axis=1)

        # SDP
        # ----
        lambda_ = expr.pars['lambda_']
        #soln = NNAL_tools.SDP_query_distribution(
        #    A, lambda_, ref_F, k)
        #print('status: %s'% (soln['status']), end='\n\t')
        #q_opt = np.array(soln['x'][:F.shape[1]])
        q_opt = NNAL_tools.solve_FIAL_SDP(A)

        # sampling from the optimal solution
        draws = NNAL_tools.sample_query_dstr(
            q_opt, k, replacement=True)

        img_ind_sizes = [len(sel_inds[i]) for i
                         in range(len(sel_inds))]
        local_inds = patch_utils.global2local_inds(
            draws, img_ind_sizes)
        Q_inds = [np.array(sel_inds[i])[local_inds[i]]
                  for i in range(len(sel_inds))]

    return Q_inds


def get_HV_inds(padded_img, patch_shape, 
                thr, pool_inds):
    """Getting the local indices of the pool samples
    that have local variance higher than a threshold
    (the output are indices of the samples in terms
    their location in the `pool_inds` array)
    """

    rads = np.int8((np.array(patch_shape)-1)/2)
    # use T1 to compute the variances
    (d1,d2,d3) = padded_img.shape
    # un-padding
    img_1 = padded_img[rads[0]:d1-rads[0],
                       rads[1]:d2-rads[1],
                       rads[2]:d3-rads[2]]

    # compute 2D variance map
    # (choosing first component of
    # the patch shape as the radius of
    # the loal variance computation)
    var_map = np.zeros(img_1.shape)
    for i in range(img_1.shape[2]):
        slice_var = patch_utils.get_vars_2d(
            img_1[:,:,i], rads[0])
        var_map[:,:,i] = slice_var

    # get variance scores of 
    # all given pool indices
    pool_multinds = np.unravel_index(
        pool_inds, img_1.shape)
    inds_vscores = var_map[pool_multinds]

    # filter-out the low-variance 
    # pool indices, and select randomly
    valid_pool_inds = np.where(
        inds_vscores>thr)[0]

    return valid_pool_inds

def binary_uncertainty_filter(posts, B):
    """Uncertainty filtering for binary class
    label distribution
    
    Since there are only two classes, posterior
    probability of only one of the classes
    are given in form of 1D array.
    """
    
    return np.argsort(np.abs(
        np.array(posts)-0.5))[:B]


def bin_uncertainty_filter_multimg(expr,
                                   model,
                                   sess,
                                   all_padded_imgs,
                                   pool_inds,
                                   B,
                                   x_feed_dict={}):

    # computing entropies for voxels of each
    # image separately
    s = len(pool_inds)
    img_ind_sizes = [len(pool_inds[i]) for i in range(s)]
    m = len(all_padded_imgs[0])-1
    n = np.sum(img_ind_sizes)
    H = [[] for i in range(s)]
    for i in range(s):
        if len(pool_inds[i])==0:
            continue

        # set the stats
        stats = []
        for j in range(m):
            stats += [[expr.train_stats[i,2*j],
                       expr.train_stats[i,2*j+1]]]
        posts = PW_NN.batch_eval(
            model,
            sess,
            all_padded_imgs[i][:-1],
            pool_inds[i],
            expr.pars['patch_shape'],
            expr.pars['ntb'],
            stats,
            'posteriors',
            None,
            x_feed_dict)[0]

        H[i] = list(posts)

    # spit out only the posteriors only if 
    # extra-feed-dict is given
    tH = np.concatenate(H)
    if len(x_feed_dict)>0:
        return tH

    # sort with respect to entropy values
    tH = np.abs(tH - 0.5)
    sorted_inds = np.argsort(tH)[:B]
    sel_inds = patch_utils.global2local_inds(
        sorted_inds, img_ind_sizes)
    sel_posts = [np.array(H[i])[sel_inds[i]] 
                 for i in range(s)]

    return sel_inds, sel_posts

def gen_A_matrices(expr, 
                   model, 
                   sess,
                   sel_patches,
                   sel_posts,
                   diag_load=1e-5):

    # forming A-matrices
    # ------------------
    # division by two in computing size of A is because 
    # in each layer we have gradients with respect to
    # weights and bias terms --> number of layers that
    # are considered is obtained after dividing by 2
    A_size = int(len(model.grad_posts['1'])/2)
    d3 = expr.pars['patch_shape'][-1]
    c = expr.nclass
    A = []

    # len(sel_posts) == sel_patches.shape[0]
    for i in range(len(sel_posts)):

        # normalizing the patch
        X_i = sel_patches[i,:,:,:]

        feed_dict = {
            model.x: np.expand_dims(X_i,axis=0),
            model.keep_prob: 1.}

        # preparing the poserior
        # ASSUMOTION: binary classifications
        x_post = sel_posts[i]
        # Computing gradients and shrinkage
        if x_post < 1e-6:
            x_post = 0.

            grads_0 = sess.run(
                model.grad_posts['0'],
                feed_dict=feed_dict)

            grads_0 =  NNAL_tools.\
                       shrink_gradient(
                           grads_0, 'sum')
            grads_1 = 0.

        elif x_post > 1-1e-6:
            x_post = 1.

            grads_0 = 0.

            grads_1 = sess.run(
                model.grad_posts['1'],
                feed_dict=feed_dict)

            grads_1 = NNAL_tools.\
                      shrink_gradient(
                          grads_1, 'sum')
        else:
            grads_0 = sess.run(
                model.grad_posts['0'],
                feed_dict=feed_dict)
            grads_0 =  NNAL_tools.\
                       shrink_gradient(
                           grads_0, 'sum')

            grads_1 = sess.run(
                model.grad_posts['1'],
                feed_dict=feed_dict)
            grads_1 =  NNAL_tools.\
                       shrink_gradient(
                           grads_1, 'sum')

        # the A-matrix
        Ai = (1.-x_post) * np.outer(grads_0, grads_0) + \
             x_post * np.outer(grads_1, grads_1)

        # final diagonal-loading
        A += [Ai+ np.eye(A_size)*diag_load]

    return A


def refine_feature_matrix(F, B):
    """Refining a feature matrix to make it
    full row-rank with a moderate condition number
    """

    # selecting from those features that have the most
    # non-zero values among the selected samples
    nnz_feats = np.sum(F>0, axis=1)
    feat_inds = np.argsort(-nnz_feats)[:int(B/2)]
    ref_F = F[feat_inds,:]
    # taking care of the rank
    while np.linalg.matrix_rank(ref_F)<len(feat_inds):
        # if the matrix is not full row-rank, discard
        # the last selected index (worst among all)
        feat_inds = feat_inds[:-1]
        ref_F = F[feat_inds,:]

    # taking care of the conditional number
    while np.linalg.cond(ref_F) > 1e6:
        feat_inds = feat_inds[:-1]
        ref_F = F[feat_inds,:]
        if len(feat_inds)==1:
            print('Only one feature is selected.')
            break

    print('Cond. #: %f'% (np.linalg.cond(ref_F)),
          end='\n\t')
    print('# selected features: %d'% 
          (len(feat_inds)), end='\n\t')

    return ref_F

def stoch_approx_IF(model,sess,
                    tr_patches,
                    pool_patches,
                    max_iter,
                    scale=50):

    ntr = tr_patches.shape[0]

    # gradient of the pool samples at
    # their weak labels
    feed_dict = {model.x:pool_patches,
                 model.keep_prob:1.}
    weak_labels = sess.run(model.prediction,
                           feed_dict=feed_dict)
    grads = NN.LLFC_grads(
        model,sess,feed_dict,weak_labels)

    # starting the iterations 
    V_t = grads
    for t in range(max_iter):
        # Hessian of a random training sample
        rand_ind = [np.random.randint(ntr)]
        feed_dict = {
            model.x:tr_patches[rand_ind,:,:,:],
            model.keep_prob:1.}
        H = -NN.LLFC_hess(model,sess,feed_dict)

        # iteration's step
        V_t = grads + V_t - H@V_t/scale

    return V_t, weak_labels

def SuPix_query(expr,
                run,
                model,
                pool_lines,
                train_inds,
                overseg_img,
                method_name,
                sess):
    """Querying strategies for active
    learning of patch-wise model
    """

    k = expr.pars['k']

    if method_name=='random':
        n = len(pool_lines)
        q = np.random.permutation(n)[:k]

    if method_name=='entropy':
        # posteriors
        posts = PW_AL.batch_eval_wlines(
            expr,
            run,
            model,
            pool_lines,
            'posteriors',
            sess)
        
        # explicit entropy scores
        scores = np.abs(posts-.5)

        # super-pixel scores
        inds_path = os.path.join(
            expr.root_dir, str(run),
            'inds.txt')
        inds_dict, locs_dict = PW_AL.create_dict(
            inds_path, pool_lines)
        pool_inds = inds_dict[list(
            inds_dict.keys())[0]]
        SuPix_scores = superpix_scoring(
            overseg_img, pool_inds, scores)
        
        # argsort-ing is not sensitive to 
        # NaN's, so invert np.inf to np.nan
        SuPix_scores[
            SuPix_scores==np.inf]=np.nan
        # also nan-out the zero superpixels
        qSuPix = np.unravel_index(
            np.argsort(np.ravel(SuPix_scores)), 
            SuPix_scores.shape)
        qSuPix = np.array([qSuPix[0][:k],
                           qSuPix[1][:k]])

    # when the superpixels are selecte, 
    # extract their grid-points too
    qSuPix_inds = PW_AL.get_SuPix_inds(
        overseg_img, qSuPix)

    return qSuPix, qSuPix_inds


def superpix_scoring(overseg_img,
                     inds,
                     scores):
    """Extending scores of a set of pixels
    represented by line numbers in index file,
    to a set of overpixels in a given
    oversegmentation
    
    :Parameters:
    
        **overseg_img** : 3D array
            oversegmentation of the image
            containing super-pixels

        **inds** : 1D array-like
            3D index of the pixels that are
            socred

        **socres** : 1D array-like
            scores that are assigned to pixels
    
    :Returns:

        **SuPix_scores** : 2D array
            scores assigned to super-pixels, 
            where each row corresponds to a
            slice of the image, and each 
            column corresponds to a super-pixel;
            such that the (i,j)-th element 
            represents the score assigned to
            the super-pixel with label j in 
            the i-th slice of the over-
            segmentation image

            If the (i,j)-th element is `np.inf`
            it means that the super-pixel with
            label j in slice i did not get any
            score pixel in its area. And if
            it is assigned zero, it means that 
            the superpixel with label j does
            not exist in slice i at all.
    """
    
    # multi-indices of pixel indices
    s = overseg_img.shape
    multinds = np.unravel_index(inds, s)
    Z = np.unique(multinds[2])
    
    SuPix_scores = np.ones(
        (s[2], 
         int(overseg_img.max()+1)))*np.inf
    for z in Z:
        slice_ = overseg_img[:,:,z]

        """ Assigning Min-Itensity of Pixels """
        # creatin an image with 
        # values on the location of 
        # pixels
        score_img = np.ones(slice_.shape)*\
                    np.inf
        slice_indic = multinds[2]==z
        score_img[
            multinds[0][slice_indic],
            multinds[1][slice_indic]]=scores[
                slice_indic]
        # now take the properties of 
        # superpixels according to the
        # score image
        props = regionprops(slice_, 
                            score_img)
        # storing the summary score
        for i in range(len(props)):
            # specify which property to keep
            # as the scores summary
            SuPix_scores[z,props[i]['label']] \
                = props[i]['min_intensity']

    return SuPix_scores
    
def draw_queries(qdist, prior, k,
                 replacement=False):
    """Drawing query samples from a query
    distribution, and possible a prior
    priobability
    """
    
    if len(prior)==0:
        pies = qdist
    else:
        pies = qdist*prior

    # returning sampled indices
    Q_inds = NNAL_tools.sample_query_dstr(
        pies, k, replacement)

    return Q_inds

def get_self_sims(F):
    """Computing representativeness of
    all members of a set described by
    the given feature vectors

    The given argument should be a 2D
    matrix, such that the i'th column
    represents features of the i'th 
    sample in the set.
    """
    
    # size of the chunk for computing
    # pairwise similarities
    b = 5000
    n = F.shape[1]
    
    # dividing indices into chunks
    ind_chunks = np.arange(
        0, n, b)
    if not(ind_chunks[-1]==n):
        ind_chunks = np.append(
            ind_chunks, n)

    reps = np.zeros(n)
    for i in range(len(ind_chunks)-1):
        Fp = F[:,ind_chunks[i]:
               ind_chunks[i+1]]
        chunk_size = ind_chunks[i+1]-\
                     ind_chunks[i]
        
        norms_p = np.sqrt(np.sum(
            Fp**2, axis=0))
        norms = np.sqrt(np.sum(
            F**2, axis=0))
        # inner product
        dots = np.dot(Fp.T, F)
        # outer-product of norms to
        # be used in the denominator
        norms_outer = np.outer(
            norms_p, norms)
        sims = dots / norms_outer
        
        # make the self-similarity 
        # -inf to ignore it
        sims[np.arange(chunk_size),
             np.arange(
                 ind_chunks[i],
                 ind_chunks[i+1])] = -np.inf
        
        # loading similarities
        reps[ind_chunks[i]:
             ind_chunks[i+1]] = np.max(
                 sims, axis=1)

    return reps

def get_cross_sims(F1, F2):
    """Computing similarities between
    individual members of  one set and 
    another set
    """

    b  = 5000
    n1 = F1.shape[1]
    n2 = F2.shape[1]

    # dividing indices into chunks
    ind_chunks = np.arange(
        0, n1, b)
    if not(ind_chunks[-1]==n1):
        ind_chunks = np.append(
            ind_chunks, n1)

    reps = np.zeros(n1)
    for i in range(len(ind_chunks)-1):
        Fp1 = F1[:,ind_chunks[i]:
                 ind_chunks[i+1]]
        
        norms_p1 = np.sqrt(np.sum(
            Fp1**2, axis=0))
        norms_2 = np.sqrt(np.sum(
            F2**2, axis=0))
        # inner product
        dots = np.dot(Fp1.T, F2)
        # outer-product of norms to
        # be used in the denominator
        norms_outer = np.outer(
            norms_p1, norms_2)
        sims = dots / norms_outer

        # loading the parameters
        reps[ind_chunks[i]:
             ind_chunks[i+1]] = np.max(
                 sims, axis=1)

    return reps    

def get_confident_samples(expr,
                          run,
                          model,
                          pool_inds,
                          num,
                          sess):
    """Generating a set of confident samples
    together with their labels
    """
    
    # posteriors
    posts = PW_AL.batch_eval_winds(
        expr,
        run,
        model,
        pool_inds,
        'posteriors',
        sess)
        
    # most confident samples
    conf_loc_inds = np.argsort(
        -np.abs(posts-.5))[:num]
    conf_inds = pool_inds[conf_loc_inds]
    
    # preparing their labels
    conf_labels = np.zeros(num, 
                           dtype=int)
    conf_labels[posts[conf_loc_inds]>.9]=1
    
    # counting number of mis-labeling
    inds_path = os.path.join(
        expr.root_dir, str(run), 'inds.txt')
    labels_path = os.path.join(
        expr.root_dir, str(run), 'labels.txt')
    inds_dict, labels_dict, locs_dict = PW_AL.create_dict(
        inds_path, conf_inds, labels_path)
    true_labels=[]
    for path in list(labels_dict.keys()):
        true_labels += list(labels_dict[path][
            locs_dict[path]])

    mis_labels = np.sum(~(
        true_labels==conf_labels))
    
    return conf_inds, conf_labels, mis_labels
        
        
        
