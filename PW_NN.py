from scipy.signal import convolve2d
import tensorflow as tf
import numpy as np
import warnings
#import nibabel
import nrrd
import pdb
import os

import NN
import PW_NNAL
import patch_utils
import PW_analyze_results


def PW_train_epoch(model,
                  dropout_rate,
                  trinds_dict,
                  trmask_dict,
                  patch_shape,
                  batch_size,
                  stats,
                  sess,
                  tsinds_dict=None,
                  tsmask_dict=None):
    """Completing one training epoch based on a
    patch-wise data set; and return the accuracy
    over the same or different data
    """
    
    trbatches = patch_utils.get_batches(
        trinds_dict,batch_size)

    mu = stats[0]
    sigma = stats[1]
    for batch in trbatches:
        (batch_tensors,
         batch_labels) = patch_utils.get_batch_vars(
             trinds_dict,
             trmask_dict,
             batch,
             patch_shape)

        batch_tensors = (
            batch_tensors-mu)/sigma

        # batch gradient step
        sess.run(
            model.train_step,
            feed_dict={
                model.x: batch_tensors,
                model.y_: batch_labels,
                model.keep_prob:dropout_rate})

        # if a test set of indices are provided,
        # compute accuracies based on them
        if tsinds_dict:
                
            tsbatches = patch_utils.get_batches(
                tsinds_dict, batch_size)

            # prediction for test samples
            tspreds_dict = batch_eval(
                model, 
                tsinds_dict,
                patch_shape,
                5000,
                stats,
                sess,
                'prediction')[0]
                
            Fm = get_Fmeasure(tspreds_dict,
                              tsmask_dict)
            
            return Fm


def PW_train_epoch_MultiModal(
        model,
        sess,
        tr_data,
        epochs,
        patch_shape,
        b,
        ntb,
        stats,
        save_dir=[],
        costs=[1.,1.],
        ts_data=None,
        tb_dir=None):
    """This function is similar to 
    `PW_train_epoch` except that it works
    with more than one modality; here 
    instead of single dictionary of for
    one modality, we have a list of 
    dictionary, one for each modality

    Data = [[P_11, P_12, ..., P_1M, PL_1, I_1],
            [P_21, P_22, ..., P_2M, PL_2, I_2],
            ...] 

    where P_ij is the path that refers to
    the j-th modality of the i-th subject,
    I_i is the set of voxel indices
    selected for this subject, and PM_i 
    is the path to the mask.

    Here the assumtion is that all
    modalities have the same shape.
    """

    # number of all training samples
    n = np.sum(
        [len(tr_data[i][-1]) 
         for i in range(len(tr_data))])

    # number of modalities
    m = len(tr_data[0])-2
    d3 = patch_shape[2]

    # number of subjects
    s = len(tr_data)

    # if TB should be prepared
    if tb_dir:
        tb_writer = tf.summary.FileWriter(
            tb_dir)

    """ Preparing the Images """
    # 
    # patch radii for padding images
    rads = np.zeros(3,dtype=int)
    for i in range(3):
        rads[i] = int((patch_shape[i]-1)/2.)
    # load, and pad all the images
    padded_imgs = []
    masks = []
    for i in range(s):
        sub_imgs = []
        for j in range(m):
            img,_ = nrrd.read(tr_data[i][j])
            padded_img = np.pad(
                img, 
                ((rads[0],rads[0]),
                 (rads[1],rads[1]),
                 (rads[2],rads[2])),
                'constant')
            sub_imgs += [padded_img]
        padded_imgs += [sub_imgs]
        # adding mask
        mask,_ = nrrd.read(tr_data[i][m])
        masks += [mask]


    """ Starting the Training Epochs """
    tb_cnt = 0
    for t in range(epochs):
        """ Preparing the Batches """
        # batch-ify the data
        batch_inds = NN.gen_batch_inds(n, b)

        for i in range(len(batch_inds)):
            
            # extract indices of each image in this
            # batch
            local_inds = patch_utils.global2local_inds(
                batch_inds[i], 
                [len(tr_data[t][-1]) for t in range(s)])

            # load patches image-by-image
            b = len(batch_inds[i])
            b_patches = np.zeros((b,
                                  patch_shape[0],
                                  patch_shape[1],
                                  m*patch_shape[2]))
            b_labels = np.zeros(b)
            cnt = 0
            for j in range(s):
                if len(local_inds[j])>0:
                    img_inds = np.array(tr_data[j][-1])[
                        local_inds[j]]
                    patches, labels = patch_utils.\
                                      get_patches(
                                          padded_imgs[j],
                                          img_inds,
                                          patch_shape,
                                          True,
                                          masks[j])
                    # normalizing the patches
                    for jj in range(m):
                        patches[:,:,:,jj*d3:(jj+1)*d3] = (
                            patches[:,:,:,jj*d3:(jj+1)*d3]-stats[
                                j,2*jj])/stats[j,2*jj+1]

                    b_patches[cnt:cnt+len(img_inds),
                              :,:,:] = patches
                    b_labels[
                        cnt:cnt+len(img_inds)] = labels
                    cnt += len(img_inds)
                    
            # hot-one vector for labels
            hot_b_labels = np.zeros((2, len(b_labels)))
            hot_b_labels[0,b_labels==0]=1*costs[0]
            hot_b_labels[1,b_labels==1]=1*costs[1]

            # finally the data is ready to
            # perform this iteration
            # batch gradient step
            sess.run(
                model.train_step,
                feed_dict={
                    model.x: b_patches,
                    model.y_: hot_b_labels,
                    model.keep_prob:model.dropout_rate})

            # writing into TB every 100 iterations
            if not(i%100):
                if ts_data:
                    eval_to_TB(
                        model,
                        sess,
                        ts_data,
                        padded_imgs,
                        masks,
                        stats,
                        patch_shape,
                        ntb,
                        tb_writer,
                        tb_cnt)

                    tb_cnt += 1
                    
        if len(save_dir)>0:
            model.save_weights(
                os.path.join(save_dir,
                             'model_pars.h5'))
            np.savetxt(os.path.join(save_dir,
                                    'epoch.txt'),
                       [t])

                
def eval_to_TB(model, 
               sess,
               ts_data,
               padded_imgs,
               masks,
               stats,
               patch_shape,
               ntb,
               tb_writer,
               tb_cnt):
    """Evaluating a given model and save the
    results into a tensor-board file
    """

    ntest = np.sum(
        [len(ts_data[i][-2])
         for i in range(len(ts_data))])
    m = len(ts_data[0])-2

    ts_batches = NN.gen_batch_inds(
        ntest, ntb)
    
    t_P = 0
    t_TP = 0
    t_FP = 0
    t_losses = 0
    for i in range(len(ts_batches)):
        """ Loading Patches of This Batch"""
        local_inds = patch_utils.global2local_inds(
            ts_batches[i], 
            [len(tr_data[t][-1]) for t in range(s)])
        # load patches image-by-image
        b = len(batch_inds[i])
        b_patches = np.zeros((b,
                              patch_shape[0],
                              patch_shape[1],
                              m*patch_shape[2]))
        b_labels = np.zeros(b)
        cnt = 0
        for j in range(s):
            if len(local_inds[j])>0:
                img_inds = np.array(tr_data[j][-2])[
                    local_inds[j]]
                patches, labels = patch_utils.\
                                  get_patches(
                                      imgs[j],
                                      img_inds,
                                      patch_shape,
                                      masks[j])
                b_patches[cnt:cnt+len(img_inds),
                          :,:,:] = patches
                b_labels[
                    cnt:cnt+len(img_inds)] = labels
                cnt += len(img_inds)

        # hot-one vector for labels
        hot_b_labels = np.zeros((2, len(b_labels)))
        hot_b_labels[0,b_labels==0]=1
        hot_b_labels[1,b_labels==1]=1

        # normalizing the patches
        for j in range(m):
            b_patches[:,:,:,j] = (
                b_patches[:,:,:,j]-stats[
                    j][0])/stats[j][1] 

        # prediction
        losses, preds = sess.run(
            [model.loss, model.prediction],
            feed_dict={
                model.x: b_data,
                model.y_: hot_b_labels,
                model.keep_prob:1.})

        t_losses += np.sum(losses)
        true_labels = np.argmax(b_labels, axis=0)
        P,N,TP,FP,TN,FN = PW_analyze_results.\
                          get_preds_stats(preds,
                                          true_labels)
        t_TP += TP
        t_FP += FP
        t_P  += P

    # compute average loss and total F-measure
    av_loss = t_losses / ntest
    if (t_TP + t_FP) > 0:
        Pr = t_TP / (t_TP+t_FP)
    else:
        Pr = 0

    if t_P > 0:
        Rc = t_TP / t_P
    else:
        Rc = 0

    if (Pr + Rc) > 0:
        F_meas = 2*Pr*Rc / (Pr+Rc)
    else:
        F_meas = 0

    # now saving them into tensorboard 
    # directories
    loss_summ = tf.Summary()
    loss_summ.value.add(
        tag='Loss',
        simple_value=av_loss)
    tb_writer.add_summary(
        loss_summ, tb_cnt)
    Fmeas_summ = tf.Summary()
    Fmeas_summ.value.add(
        tag='F1 score',
        simple_value=F_meas)
    tb_writer.add_summary(
        Fmeas_summ, tb_cnt)
        
def batch_eval(model,
               sess, 
               img_dat,
               inds,
               patch_shape,
               batch_size,
               stats,
               varnames,
               mask=None,
               x_feed_dict={}):
    """evaluating a list of variables over
    a set of samples from different images
    in a batch-wise format
    
    :Parameters:
    
        **model** : CNN model
            an object with `prediction`
            and `posterior` properties
    
        **imgs_dat** : list of images or their paths 
           a sequence in the following
           structure: `[P1,...,PM]`,
           where `P1` to `PM` are either 
           paths to the `M` modalities of data
           or the images themselves
    
        **inds** : array of integers
            3D indices of the voxels corresponding
            to the given images

        **patch_shape** : tuple
            shape of patches

        **states** : array-like floats
            mean and standard-deviation of
            intensities to be used to
            normalize intensity of the 
            patches

        **sess** : TF session

        **varnames** : list of strings
            a flag determining which one
            of posterior (`post`) or
            prediction (`pred`), or both
            should be evaluated; note 
            the number of outputs of the
            function will be two in any
            case, and if either of the
            possible strings are not
            present in this flag, the
            corresponding output will
            an all-zero array.
    
            Also note that the posterior
            array includes only the 
            probability of being masked
            (in a binary segmentation).
    """
    
    # number of modalities
    m = len(img_dat)

    if not(isinstance(varnames, list)):
        varnames = [varnames]

    # preparing radii for image padding
    rads = np.zeros(3,dtype=int)
    for i in range(3):
        rads[i] = int((patch_shape[i]-1)/2.)
    # check if the images are given or the
    # paths to them
    if isinstance(img_dat[0], np.ndarray):
        padded_imgs = img_dat
    else:
        # loading + padding 
        img_paths = img_dat
        padded_imgs = []
        for j in range(m):
            img,_ = nrrd.read(img_paths[j])
            padded_img = np.pad(
                img, 
                ((rads[0],rads[0]),
                 (rads[1],rads[1]),
                 (rads[2],rads[2])),
                'constant')
            padded_imgs += [padded_img]

    # preparing batch indices
    n = len(inds)
    batch_ends = np.arange(0,n,batch_size)
    if not(batch_ends[-1]==n):
        batch_ends = np.append(
            batch_ends, n)

    """ Evaluating the List of Variables """
    vals_list = []
    for j,var in enumerate(varnames):
        # create the array for this image
        # and this variable in the first
        # iteration
        if var=='feature_layer':
            fdim = model.feature_layer.shape[0].value
            vals = np.zeros((fdim,n))
        else:
            vals = np.zeros(n)

        # evaluate variable for this batch
        model_var = getattr(model, var)
        if (var=='loss') or (var=='hess_vecp'):
            labels_flag = True
        else:
            labels_flag = False


        """evaluating variables with no need
        to batch labels
        """
        # going through batches
        for i in range(1,len(batch_ends)):
            # getting the chunk of indices
            batch_inds = np.arange(
                batch_ends[i-1],batch_ends[i])
            b = len(batch_inds)
            # loading tensors
            # (not to be confused with 
            # patch_utils.get_batches())
            if labels_flag:
                (batch_tensors, 
                 batch_labels) = patch_utils.get_patches(
                     padded_imgs, 
                     np.array(inds)[batch_inds],
                     patch_shape,
                     True,
                     mask)

                hot_labels = np.zeros((2,b))
                hot_labels[0,batch_labels==0]=1
                hot_labels[1,batch_labels==1]=1
            else:
                batch_tensors = patch_utils.get_patches(
                    padded_imgs, 
                    np.array(inds)[batch_inds],
                    patch_shape)

            for j in range(m):
                batch_tensors[:,:,:,j] = (
                    batch_tensors[
                        :,:,:,j]-stats[j][0])/stats[j][1]

            if labels_flag:
                feed_dict = {
                    model.x:batch_tensors,
                    model.y_:hot_labels,
                    model.keep_prob: 1.}
            else:
                feed_dict = {model.x:batch_tensors,
                             model.keep_prob: 1.}

            # if a keep-probability different than
            # 1. is to be used (e.g. in MC-dropout)
            # put it in x_feed_dict and it will 
            # replace 1. in the feed_dict.
            feed_dict.update(x_feed_dict)
            batch_vals = sess.run(
                model_var,
                feed_dict=feed_dict)

            if var=='posteriors':
                # keeping only posterior probability
                # of being maksed
                vals[batch_inds] = batch_vals[1,:]
            elif var=='feature_layer':
                vals[:,batch_inds] = batch_vals
            elif var=='hess_vecp':
                vals = batch_vals
            else:
                vals[batch_inds] = batch_vals                

        vals_list += [vals]
            
    return vals_list


def get_accuracy(preds, labels):
    
    n = len(preds)
    labels = np.argmax(labels,axis=0)
    
    return np.sum(preds==labels) / float(n)


