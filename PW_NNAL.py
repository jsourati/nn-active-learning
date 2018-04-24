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
        
        thr = 2.   # variance threshold
        rads = np.int8((np.array(expr.pars[
            'patch_shape'])-1)/2)
        # use T1 to compute the variances
        (d1,d2,d3) = padded_imgs[0].shape
        # un-padding
        img_1 = padded_imgs[0][
            rads[0]:d1-rads[0],
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
                  tr_inds,
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

    if method_name=='entropy':
        
        Q_inds = bin_uncertainty_filter_multimg(
            expr, model, sess, all_padded_imgs,
            pool_inds, k)[0]

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
                                sel_posts[i])

            F[i] = PW_NN.batch_eval(
                model,sess,
                all_padded_imgs[i][:-1],
                img_inds[i],
                expr.pars['patch_shape'],
                expr.pars['ntb'],
                stats,
                'feature_layer')[0]

        F = [F[i] for i in range(len(F)) if len(F[i])>0]
        F = np.concatenate(F, axis=1)
        ref_F = refine_feature_matrix(F, B)
        # make the feature components zero-mean
        ref_F -= np.repeat(np.expand_dims(
            np.mean(ref_F, axis=1),
            axis=1), F.shape[1], axis=1)

        # SDP
        # ----
        lambda_ = expr.pars['lambda_']
        soln = NNAL_tools.SDP_query_distribution(
            A, lambda_, ref_F, k)
        print('status: %s'% (soln['status']), end='\n\t')
        q_opt = np.array(soln['x'][:F.shape[1]])
        
        # sampling from the optimal solution
        draws = NNAL_tools.sample_query_dstr(
            q_opt, k, replacement=True)

        img_ind_sizes = [len(sel_inds[i]) for i
                         in range(len(sel_inds))]
        local_inds = patch_utils.global2local_inds(
            draws, img_ind_sizes)
        Q_inds = [np.array(sel_inds[i])[local_inds[i]]
                  for i in range(len(sel_inds))]

    if method_name=='if':

        max_iter = 100
        n = np.sum(img_ind_sizes)
        d = (model.feature_layer.shape[0].value + 1) * \
            model.output.shape[0].value
        IFs = np.zeros((d,n))

        # computing gradients of the training samples
        tr_patches,_ = patch_utils.get_patches_multimg(
            all_padded_imgs, tr_inds,
            expr.pars['patch_shape'], 
            expr.train_stats)
        tr_grads = []
        for i in range(len(tr_patches)):
            if len(tr_patches[i])==0:
                continue
            feed_dict={model.x:tr_patches[i],
                       model.keep_prob:1.}
            # getting training labels
            multinds = np.unravel_index(
                tr_inds[i], all_padded_imgs[i][-1].shape)
            labels = all_padded_imgs[i][-1][multinds]
            tr_grads += [NN.LLFC_grads(model,sess,
                                       feed_dict,labels)]
        tr_grads = np.concatenate(tr_grads, axis=1)
        tr_patches = [tr_patches[i] for i in range(len(tr_patches))
                      if len(tr_patches[i])>0]
        tr_patches = np.concatenate(tr_patches, axis=0)

        # batch-wise computing of IFs
        print('Computing IF vectors')
        batches = NN.gen_batch_inds(n,expr.pars['ntb'])
        for i in range(len(batches)):
            loc_inds = patch_utils.global2local_inds(
                batches[i], img_ind_sizes)
            img_inds = [np.array(pool_inds[i])[loc_inds[i]]
                for i in range(len(pool_inds))]

            patches,_ = patch_utils.get_patches_multimg(
                all_padded_imgs, img_inds,
                expr.pars['patch_shape'], 
                expr.train_stats)
            patches = [patches[i] for i in range(len(patches))
                       if len(patches[i])>0]
            patches = np.concatenate(patches, axis=0)
            bIFs, weak_labels = stoch_approx_IF(
                model,sess,tr_patches,patches,max_iter)

            IFs[:,batches[i]] = bIFs

        # expected change in training log-likelhoods
        print('\tComputing LOOCV scores')
        tr_ll_change = tr_grads.T @ IFs
        scores = np.mean(tr_ll_change, axis=0)
        global_inds = np.argsort(-scores)[:k]
        Q_inds = patch_utils.global2local_inds(
            global_inds, img_ind_sizes)

    return Q_inds


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
                                   B):

    # computing entropies for voxels of each
    # image separately
    s = len(pool_inds)
    img_ind_sizes = [len(pool_inds[i]) for i in range(s)]
    m = len(all_padded_imgs[0])-1
    n = np.sum(img_ind_sizes)
    H = [[] for i in range(s)]
    for i in range(s):
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
            'posteriors')[0]

        H[i] = list(posts)

    # sort with respect to entropy values
    tH = np.concatenate(H)
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
                    scale=10):

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
        V_t = grads - H@V_t/scale

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
        
        
        
