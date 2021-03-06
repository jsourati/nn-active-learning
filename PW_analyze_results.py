from skimage.segmentation import find_boundaries
from skimage.measure import regionprops

from pydensecrf.utils import create_pairwise_bilateral
from pydensecrf.utils import create_pairwise_gaussian
import pydensecrf.densecrf as dcrf

#import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import linecache
import shutil
import pickle
import scipy
import nrrd
import yaml
import pdb
import os

import tensorflow as tf
import patch_utils
import PW_NNAL
import PW_NN
import PW_AL
import NN


def get_queries(expr, method_name):
    """Simply giving back the queries generated
    in different queries separately
    """
    
    Qs = []

    Q_dir = os.path.join(
        expr.root_dir,
        method_name, 'queries')
    Q_files = os.listdir(Q_dir)
    file_inds = [int(Q_files[i].split('.')[0]) for 
                 i in range(len(Q_files))]
    sorted_inds = np.argsort(file_inds)

    for ind in sorted_inds:
        fullpath = os.path.join(
            Q_dir, Q_files[ind])
        Qs += [np.int32(np.loadtxt(
            fullpath))]
        
    return Qs

def get_queries_type(expr,run,method_name):
    """Getting type of queries generated by
    a specific method of a run of AL experiment
    """
    
    stypes = []
    Qs = get_queries(expr, run, method_name)
    
    Q_types = []
    for Q in Qs:
        t = get_sample_type(
            expr, run, Q)
        Q_types += [t]
        
    return Q_types


def get_sample_type(expr, run, inds):
    """Getting type of a given set of
    indexed samples in a run of an
    experiment
    """

    stypes = []
    for ind in inds:
        line = linecache.getline(
            os.path.join(
                expr.root_dir,
                str(run),
                'inds.txt'), ind)
        stypes += [int(line.splitlines(
        )[0].split(',')[-1])]

    return stypes

def get_slice_preds(expr,
                    run,
                    model,
                    inds,
                    slice_,
                    sess):
    """Getting the results of 
    class prediction of a set of indexed
    voxels in a given slices of the 
    image
    """
    
    # take only indices of the 
    # given slice
    if expr.pars['data']=='adults':
        img_addrs, mask_addrs = patch_utils.extract_Hakims_data_path()
    elif expr.pars['data']=='newborn':
        img_addrs, mask_addrs = patch_utils.extract_newborn_data_path()

    img_addr = img_addrs[expr.pars[
        'indiv_img_ind']]
    img,_ = nrrd.read(img_addr)

    inds_path = os.path.join(expr.root_dir,
                             str(run),
                             'inds.txt')
    samples_dict,_ = PW_AL.create_dict(
        inds_path, inds)
    multinds = np.unravel_index(
        samples_dict[img_addr], 
        img.shape)
    slice_indics = multinds[2]==slice_
    slice_multinds = (
        multinds[0][slice_indics],
        multinds[1][slice_indics])
    slice_inds = inds[slice_indics]
    
    # prediction
    preds = PW_AL.batch_eval_winds(
        expr,
        run,
        model,
        slice_inds,
        'prediction',
        sess)
    
    return preds, slice_multinds


def visualize_eval_metrics(expr,
                           run,
                           metric,
                           methods=[],
                           colors=[]):
    """Visualize performance evaluations
    of a set of methods in an experiment's
    run

    Size of the color vector `colors` should
    be the number of included path plus
    one (if there also exists the performance
    metric value for the full pool data set)
    """

    run_path = expr.root_dir
    if len(methods)==0:
        methods = [f for f in os.listdir(run_path) 
                   if os.path.isdir(os.path.join(
                           run_path, f))]

    # maximum number of queries among methods
    M = 0
    for i, method_name in enumerate(methods):
        if not(os.path.exists(os.path.join(
                expr.root_dir, method_name))):
            continue

        if metric=='F1':
            # vector of evaluation metrics
            F = np.loadtxt(os.path.join(
                run_path, 
                method_name,
                'perf_evals.txt'))
            F[np.isnan(F)] = 0
        elif metric=='Precision':
            F = get_eval_metrics(
                expr, run, method_name)[0,:]
        elif metric=='Recall':
            F = get_eval_metrics(
                expr, run, method_name)[1,:]


        # vector of numbre of observed
        # labels at each query iterations
        Qset = get_queries(expr,
                           method_name)
        Qsizes = [0] + [len(Q) for Q in Qset]
        Qsizes = np.cumsum(Qsizes)

        # if the last iteration is still not
        # evaluated, ignore the queries
        if len(Qsizes)==len(F)+1:
            Qsizes = Qsizes[:-1]

        M = max(M, Qsizes[-1])
        # plotting this curve
        if method_name=='fi':
            method_name='Fisher'

        if len(colors)>0:
            plt.plot(Qsizes, F, 
                     linewidth=2,
                     color=colors[i],
                     marker = '*',
                     label=method_name)
        else:
            plt.plot(Qsizes, F, 
                     linewidth=2,
                     marker='*',
                     label=method_name)

    # get the full performance 
    if os.path.exists(os.path.join(
            run_path, 
            'pooltrain_eval.txt')):
        full_F = np.loadtxt(os.path.join(
            run_path, 
            'pooltrain_eval.txt'))

        if len(colors)>0:
            plt.plot([0,M], 
                     [full_F, full_F],
                     linewidth=2,
                     color=colors[-1],
                     label='Pool-training')
        else:
            plt.plot([0,M], 
                     [full_F, full_F], 
                     linewidth=2,
                     label='Pool-training')

    plt.legend(fontsize=15)
    plt.xlabel('# Queries', fontsize=15)
    plt.ylabel(metric, fontsize=15)
    plt.grid()


def get_preds_stats(preds, mask):
    """Computing different statistics of
    a set of prediction in comparison with
    the ground truth labels, such as P, N, 
    TP, TN, FP, FN
    
    At this time, this function deals only
    with single images (and not a dictionary
    of multiple images). That is to say, the
    inputs are two arrays of the same size, 
    and with binary values (0 or 1)
    """

    P = float(np.sum(mask>0))
    N = float(np.sum(mask==0))
    TP = float(np.sum(np.logical_and(
        preds>0, mask>0)))
    FP = float(np.sum(np.logical_and(
        preds>0, mask==0)))
    TN = float(np.sum(np.logical_and(
        preds==0, mask==0)))
    FN = float(np.sum(np.logical_and(
        preds==0, mask>0)))

    return P, N, TP, FP, TN, FN
    

def get_Fmeasure(preds, mask):
    
    # computing total TPs, Ps, and
    # TPFPs (all positives)
    P  = 0
    TP = 0
    TPFP = 0
    if isinstance(preds, dict):
        for img_path in list(preds.keys()):
            ipreds = preds[img_path]
            imask = np.array(mask[img_path])
            
            P  += np.sum(imask>0)
            TP += np.sum(np.logical_and(
                ipreds>0, imask>0))
            TPFP += np.sum(ipreds>0)
    else:
        
        P  += np.sum(mask>0)
        TP += np.sum(np.logical_and(
            preds>0, mask>0))
        TPFP += np.sum(preds>0)

    # precision and recall
    Pr = TP / TPFP
    Rc = TP / P
    
    # F measure
    return 2/(1/Pr + 1/Rc)

def F1_scores(preds,labels):
    P,N,TP,FP,TN,FN = get_preds_stats(preds, labels)
    Pr = TP / (TP+FP)
    Rc = TP/P 
    return 2*Pr*Rc / (Pr+Rc)

def get_eval_metrics(expr,
                     run,
                     method_name):
    """Computing different evaluation
    metrics of the predictions in results
    of running a specific querying
    method in an experiment's run
    """

    run_path = os.path.join(
        expr.root_dir, str(run))

    # load ground truth labels
    labels_path = os.path.join(
        run_path, 'labels.txt')
    test_lines = np.int64(np.loadtxt(
        os.path.join(run_path, 
                     'test_lines.txt')))
    test_labels = PW_AL.read_label_lines(
        labels_path, test_lines)

    # load predictions
    preds_path = os.path.join(
        run_path, 
        method_name, 
        'predicts.txt')
    preds = np.loadtxt(preds_path)
    
    iter_cnt = preds.shape[0]
    Metrs = np.zeros((2, iter_cnt))
    for i in range(iter_cnt):
        (P, N, TP, 
         FP, TN, FN) = get_preds_stats(
             preds[i,:], test_labels)
        
        # Precision
        Metrs[0,i] = TP / (TP+FP)
        # Recall
        Metrs[1,i] = TP / P

    return Metrs
    
def mask_SuPix(overseg_img,
               SuPix_codes,
               show_bound=True):
    """Visualizing some super-pixels in
    the over-segmentation image where the 
    boundaries of all the super-pixels
    are shown, and the selected ones
    are high-lighted
    """

    s = overseg_img.shape

    masked_SuPix = np.zeros(
        s, dtype=bool)

    # get the boundaries if necessary
    if show_bound:
        for i in range(s[2]):
            masked_SuPix[:,:,i
            ] = find_boundaries(
                overseg_img[:,:,i])

    # selected superpixels slices
    slices = np.unique(SuPix_codes[0,:])
    for j in slices:
        props = regionprops(
            overseg_img[:,:,j])
        SuPix_labels = SuPix_codes[
            1, SuPix_codes[0,:]==j]
        n_overseg = len(props)
        prop_labels = [props[i]['label']
                      for i in 
                      range(n_overseg)]
        # mask the super-pixels
        for label in SuPix_labels:
            label_loc = np.where(
                prop_labels==label)[0][0]
            # indices of the pixels in 
            # this super-pixel
            multinds_2D = props[
                label_loc]['coords']
            vol = len(multinds_2D[:,0])
            multinds_3D = (
                multinds_2D[:,0],
                multinds_2D[:,1],
                np.ones(vol,dtype=int)*j)

            masked_SuPix[multinds_3D] = True

    return masked_SuPix

def full_model_probs(expr,
                     run,
                     method_name,
                     img_path,
                     slice_inds):
    """Computing the probability maps of slices
    of an image that is given through its path
    
    If `method_name` is an empty list, then this
    function loads the weights to which the 
    parameter `expr.pars['init_weights_path]`
    is referring.
    """

    if len(method_name)>0:
        method_path = os.path.join(
            expr.root_dir, str(run),
            method_name)
        weights_path = os.path.join(
            method_path, 'curr_weights.h5')
    else:
        weights_path = expr.pars[
            'init_weights_path']


    # make the model ready
    model = NN.create_model(
        expr.pars['model_name'],
        expr.pars['dropout_rate'],
        expr.nclass,
        expr.pars['learning_rate'],
        expr.pars['grad_layers'],
        expr.pars['train_layers'],
        expr.pars['optimizer_name'],
        expr.pars['patch_shape'])

    # start TF session to do the prediction
    with tf.Session() as sess:
        print("Loading model with weights %s"% 
              weights_path)
        # loading the weights into the model
        model.initialize_graph(sess)
        model.load_weights(
            weights_path, sess)

        # get the predictins
        slice_evals = full_slice_eval(
            model,
            img_path,
            slice_inds,
            'axial',
            expr.pars['patch_shape'],
            expr.pars['ntb'],
            expr.pars['stats'],
            sess,
            'posteriors')
        
    return slice_evals

def full_model_pred_DCRF(expr,
                         model,
                         sess,
                         img_path,
                         mask_path,
                         slice_inds,
                         save_dir=None):
    """Generating  predictions of a model
    that is post-processed by Dense-CRF, over 
    particular slices of an image

    If `method_name` is given, the last model that
    is saved for that method will be used (in the
    given experiment's run)
    
    If `method_name` is an empty list, then this
    function loads the weights to which the 
    parameter `expr.pars['init_weights_path]`
    is referring.
    """

    img,_ = nrrd.read(img_path)
    mask,_ = nrrd.read(mask_path)

    DCRF_preds = np.zeros(img.shape)

    if save_dir:
        if not(os.path.exists(save_dir)):
            os.mkdir(save_dir)

    for i, ind in enumerate(slice_inds):
        # get the posteriors
        slice_posts = full_slice_eval(
            model,
            img_path,
            [ind],
            'axial',
            expr.pars['patch_shape'],
            expr.pars['ntb'],
            expr.pars['stats'],
            sess,
            'posteriors')

        slice_dcrf = DCRF_postprocess_2D(
            slice_posts[0],
            img[:,:,ind])
        DCRF_preds[:,:,ind] = slice_dcrf

        #print('%d / %d'% 
        #      (i, len(slice_inds)-1))

        if False:
            # save the results, showing 
            # predictions on mask boundaries
            mask_bound = find_boundaries(
                mask[:,:,ind])
            rgb_result = patch_utils.\
                         generate_rgb_mask(
                             img[:,:,ind], 
                             slice_dcrf,
                             mask_bound)

            fig = plt.figure(figsize=(7,7))
            plt.imshow(rgb_result, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(
                save_dir,'%d.png'% 
                (slice_inds[i])), 
                        bbox_inches='tight')
            plt.close(fig)

    # computing F-measure
    P,N,TP,FP,TN,FN = get_preds_stats(
        DCRF_preds[:,:,slice_inds],
        mask[:,:,slice_inds])
    Pr = TP/(TP+FP)
    Rc = TP/P
    F1 = 2./(1/Pr+1/Rc)

    if save_dir:
        nrrd.write(os.path.join(
            save_dir, 'dcrf_segs.nrrd'),
                   DCRF_preds)
        np.savetxt(os.path.join(
            save_dir, 'F1_score_dcrf.txt'),
                   [F1])

    return DCRF_preds, F1


def DCRF_postprocess_2D(post_map, 
                        img_slice):
    """Dense-CRF applying on a 2D 
    binary posterior map
    """

    d = dcrf.DenseCRF2D(img_slice.shape[0], 
                        img_slice.shape[1], 
                        2)
    # unary potentials
    post_map[post_map==0] += 1e-10
    post_map = -np.log(post_map)
    U = np.float32(np.array([1-post_map,
                             post_map]))
    U = U.reshape((2,-1))
    d.setUnaryEnergy(U)

    # pairwise potentials
    # ------------------
    # smoothness kernel (considering only
    # the spatial features)
    feats = create_pairwise_gaussian(
        sdims=(1, 1), 
        shape=img_slice.shape)

    d.addPairwiseEnergy(
        feats, compat=20,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # appearance kernel (considering spatial
    # and intensity features)
    feats = create_pairwise_bilateral(
        sdims=(5, 5), 
        schan=(1),
        img=img_slice, 
        chdim=-1)

    d.addPairwiseEnergy(
        feats, compat=30,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # D-CRF's inference
    niter = 5
    Q = d.inference(niter)

    # maximum probability as the label
    MAP = np.argmax(Q, axis=0).reshape(
        (img_slice.shape[0], 
         img_slice.shape[1]))

    return MAP


def full_model_eval(expr,
                    model,
                    sess,
                    img_path,
                    mask_path,
                    slice_inds,
                    save_dir=None):
    """Evaluating the last model of a querying
    method in an experiment's run
    """

    if save_dir:
        if not(os.path.exists(save_dir)):
            os.mkdir(save_dir)

    mask,_ = nrrd.read(mask_path)
    img,_ = nrrd.read(img_path)

    # slice-by-slice prediction
    preds = np.zeros(mask.shape)
    for i, ind in enumerate(slice_inds):
        # get the predictins
        slice_evals = full_slice_eval(
            model,
            img_path,
            [ind],
            'axial',
            expr.pars['patch_shape'],
            expr.pars['ntb'],
            expr.pars['stats'],
            sess)
        preds[:,:,ind] = slice_evals[0]
        
        print('%d / %d'% (i, len(slice_inds)),
              end=',')

        # save the results, with showing both
        # model evaluations and mask boundaries
        if False:
            mask_bound = find_boundaries(
                mask[:,:,ind])
            rgb_result = patch_utils.generate_rgb_mask(
                img[:,:,ind], 
                slice_evals[0], 
                mask_bound)

            fig = plt.figure(figsize=(7,7))
            plt.imshow(rgb_result, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(
                save_dir,'%d.png'% (ind)), 
                        bbox_inches='tight')
            plt.close(fig)

    # computing F-measure
    P,N,TP,FP,TN,FN = get_preds_stats(
        preds[:,:,slice_inds],
        mask[:,:,slice_inds])
    Pr = TP/(TP+FP)
    Rc = TP/P
    F1 = 2./(1/Pr+1/Rc)
    print('\n F1: %.4f'% F1)

    # save the results if necessary
    # this save_path will be created inside the
    # method's directory
    if save_dir:

        # save the results itself
        nrrd.write(os.path.join(
            save_dir, 'segs.nrrd'),
                   np.uint8(preds))
        np.savetxt(os.path.join(
            save_dir, 'F1_socre.txt'),
                   [F1])

    return preds, F1


def full_slice_eval(model,
                    sess,
                    img_paths,
                    slice_inds,
                    patch_shape,
                    ntb,
                    stats,
                    varname='prediction'):
    """Generating prediction of all voxels
    in a few slices of a given image
    """
    
    img,_ = nrrd.read(img_paths[0])
    img_shape = img.shape
    slice_nvox = np.prod(img_shape[:2])

    inds_2D = np.arange(0, slice_nvox)
    
    # single to multiple 2D indices
    # (common for all slices)
    multinds_2D = np.unravel_index(
        inds_2D, img_shape[:2])
    
    slice_evals = np.zeros(img_shape)
    for ind in slice_inds:
        extra_inds = np.ones(
            len(inds_2D),
            dtype=int)*ind
        multinds_3D = multinds_2D +\
                           (extra_inds,)
        
        # multi 3D to single 3D indices
        inds_3D = np.ravel_multi_index(
            multinds_3D, img_shape)
        # get the prediction for this slice
        evals = PW_NN.batch_eval(model,
                                 sess,
                                 img_paths, 
                                 inds_3D,
                                 patch_shape,
                                 ntb,
                                 stats,
                                 varname)[0]
        # prediction map
        eval_map = np.zeros(img_shape[:2])
        eval_map[multinds_2D] = evals
        slice_evals[:,:,ind] = eval_map

        #print('%d / %d'% 
        #      (i,len(slice_inds)))

    return slice_evals


def full_test_slice_DCRF(newborn_exp_names):
    
    # load the first experiment just to create
    # the CNN model
    E = PW_AL.Experiment(newborn_exp_names[0])
    E.load_parameters()
    model = NN.create_model(       
        E.pars['model_name'],
        E.pars['dropout_rate'],
        E.nclass,
        E.pars['learning_rate'],
        E.pars['grad_layers'],
        E.pars['train_layers'],
        E.pars['optimizer_name'],
        E.pars['patch_shape'])
    
    base_dir = '/common/collections/dHCP/dHCP_DCI_spatiotemporal_atlas/Processed'
    with tf.Session() as sess:
        model.initialize_graph(sess)
        
        for root_dir in newborn_exp_names:
            print('Experiment %s..'% root_dir)
            E = PW_AL.Experiment(root_dir)
            E.load_parameters()

            weights_path = os.path.join(
                E.root_dir, 
                '0/random/curr_weights.h5')
            model.load_weights(weights_path, sess)

            save_dir = os.path.join(
                E.root_dir, '0/random/full_preds')
            if not(os.path.exists(save_dir)):
                os.mkdir(save_dir)

            _,img_path,mask_path = PW_AL.get_expr_data_info(
                E, base_dir)
            img,_ = nrrd.read(img_path)
            slice_inds = np.arange(1,img.shape[2],2)

            _,_ = full_model_pred_DCRF(
                E,model,sess,
                img_path, mask_path,
                slice_inds, save_dir)

def grid_based_F1(model, sess,
                    img_paths, mask_path,
                    patch_shape,
                    ntb,
                    stats):
    """Computing F1 score based on (all) grid
    samples of some images
    """

    # generating grid samples
    spacing = 10
    inds, labels,_ = patch_utils.generate_grid_samples(
        img_paths[0], mask_path, 10, 0)

    # predictions
    preds = PW_NN.batch_eval(model, sess, 
                             img_paths, inds,
                             patch_shape,
                             ntb, stats,
                             'prediction')[0]

    # F1 score
    P,N,TP,FP,TN,FN = get_preds_stats(
        preds, np.array(labels))
    Pr = TP / (TP+FP)
    Rc = TP/P

    return 2./(1/Pr + 1/Rc)


def eval_MultimgAL(expr,method_name,
                    img_paths,
                    start_ind=0,
                    save_dir=[]):

    m = len(expr.train_paths[0])-1
    patch_shape = expr.pars['patch_shape'][:2] + \
        (m*expr.pars['patch_shape'][2],)

    model = NN.create_model(
        expr.pars['model_name'],
        expr.pars['dropout_rate'],
        expr.nclass,
        expr.pars['learning_rate'],
        expr.pars['grad_layers'],
        expr.pars['train_layers'],
        expr.pars['optimizer_name'],
        patch_shape)
    model.add_assign_ops()

    method_path = os.path.join(expr.root_dir,
                               method_name)

    Qs = get_queries(expr, method_name)
    qnum = len(Qs)
    imgnum = len(img_paths)
    
    save_dir = os.path.join(method_path,
                            'test_scores.txt')

    if start_ind>0:
        scores = np.loadtxt(save_dir)
    else:
        scores = np.zeros((imgnum, qnum))

    with tf.Session() as sess:
        model.initialize_graph(sess)
        sess.graph.finalize()

        for i in range(start_ind,qnum):
            weights_path = os.path.join(method_path,
                                        'curr_weights_%d.h5'% (i+1))
            print('Loading weights %s'% weights_path)
            model.perform_assign_ops(weights_path, sess)

            for j in range(imgnum):
                 # grid-samples from the j-th image
                 expr.test_paths = img_paths[j:j+1]
                 stats_arr = np.zeros((1,2*m))
                 mask,_ = nrrd.read(expr.test_paths[0][-1])
                 for t in range(m):
                     img,_ = nrrd.read(expr.test_paths[0][t])
                     stats_arr[0,2*t:2*(t+1)] = np.array(
                         [np.mean(img[~np.isnan(mask)]),
                          np.std(img[~np.isnan(mask)])])
                 expr.test_stats = stats_arr

                 scores[j,i],test_preds = expr.test_eval(model, sess)
                 np.savetxt(save_dir, scores)

                 print(j, end=',')
            print()


def get_interp_slice_posts(x, y, vals, slice_shape):
    
    slice_vals = np.zeros(slice_shape)
    
    # interpolator 
    f = scipy.interpolate.interp2d(x, y, vals)

    # evaluate interpolator on the mesh
    xgrid = np.arange(slice_shape[0])
    ygrid = np.arange(slice_shape[1])
    yy,xx = np.meshgrid(ygrid,xgrid)
    xx = np.ravel(xx)
    yy = np.ravel(yy)
    
    for i in range(len(xx)):
        slice_vals[xx[i], yy[i]] = f(xx[i], yy[i])

    return slice_vals


def get_Qsims(model,sess, expr, method_name):
    
    Qs = get_queries(expr, method_name)

    imgs = []
    for path in expr.pars['img_paths']:
        img,_ = nrrd.read(path)
        imgs += [img]

    # get similarities
    sims = []
    d3 = expr.pars['patch_shape'][2]
    stats = expr.pars['stats']
    for i in range(len(Qs)):
        patches = patch_utils.get_patches(
            imgs, Qs[i], 
            expr.pars['patch_shape'], False)
        for j in range(len(imgs)):
            patches[:,:,:,j*d3:(j+1)*d3] = (patches[
                :,:,:,j*d3:(j+1)*d3]-stats[j][0])/\
                stats[j][1]

        # flattened version of the output of
        # the last convolutional layer as features
        F = sess.run(model.probes[0],
                     feed_dict={model.x:patches,
                                model.keep_prob:1.})
        # cosine similarities
        inners = np.dot(F.T,F)
        norms = np.sqrt(np.sum(F**2, axis=0))
        cos_sims = inners / np.outer(norms,norms)

        sims += [cos_sims]

    return sims
