from scipy.signal import convolve2d
import tensorflow as tf
import numpy as np
import warnings
import nibabel
import nrrd
import pdb
import os

import NN
import PW_NNAL
import patch_utils


def train_pw_model(patch_shape,
                   batch_size,
                   learning_rate,
                   dropout_rate,
                   epochs):
    """Training a model for patchwise
    segmentation
    """
    
    # path to data
    img_addrs, mask_addrs = patch_utils.extract_Hakims_data_path()
    
    # class of data
    pw_dataset = patch_utils.PatchBinaryData(
        img_addrs,mask_addrs)

    # training data
    train_imgs = [0,1,2]
    inds_dict, mask_dict = pw_dataset.generate_samples(
        train_imgs, [100,100,50],.2, 'axial')
    train_batches = get_batches(inds_dict,
                                batch_size)
    
    # validation
    valid_imgs = [3]
    vinds_dict, vmask_dict = pw_dataset.generate_samples(
        valid_imgs, [50,50,10],.2, 'axial')
    valid_batches = get_batches(vinds_dict,
                                batch_size)
    
    """Creating the model
    """
    nclass = 2
    dropout_rate = 0.5
    learning_rate = 1e-5
    model = get_model(nclass,
                      dropout_rate,
                      learning_rate,
                      patch_shape)

    """Start the training epochs
    """
    # mean and std to normalize the data
    mu = 65.
    sigma = 54.5
    
    loss_summary = tf.summary.merge_all()
    with tf.Session() as sess:
        
        train_writer = tf.summary.FileWriter(
            '/common/external/rawabd/' + 
            'Jamshid/train_log/pw_full/training/',
            sess.graph)
        valid_writer = tf.summary.FileWriter(
            '/common/external/rawabd/' + 
            'Jamshid/train_log/pw_full/validation/')

        # initialization
        sess.run(
            tf.global_variables_initializer())

        cnt = 0
        for i in range(epochs):
            print("Epoch %d.."% i)
            for batch in train_batches:
                # loading the batch
                (batch_tensors,
                 batch_labels) = get_batch_vars(
                     inds_dict,
                     mask_dict,
                     batch,
                     patch_shape)

                # normalizing intensities
                batch_tensors = (
                    batch_tensors-mu)/sigma

                # batch gradient step
                summary,_,preds = sess.run(
                    [loss_summary, 
                     model.train_step,
                     model.prediction],
                    feed_dict={
                        model.x: batch_tensors,
                        model.y_: batch_labels,
                        model.keep_prob:dropout_rate})
                
                # writing the training loss
                if cnt % 50 ==0:
                    acc = get_accuracy(preds,batch_labels)
                    train_writer.add_summary(
                        summary, cnt)
                    acc_summary = tf.Summary()
                    acc_summary.value.add(
                        tag='Accuracy',
                        simple_value=acc)
                    train_writer.add_summary(
                        acc_summary, cnt)
                    
                    model.save_weights('tmp_weights.h5')
                
                # write for test set every 10 step 
                if cnt % 100==0:
                    t_vloss = 0
                    vsize = 0
                    t_corrpreds = 0
                    for vbatch in valid_batches:
                        (vbatch_tensors,
                         vbatch_labels)=get_batch_vars(
                             vinds_dict,
                             vmask_dict,
                             vbatch,
                             patch_shape)
                        
                        vbatch_tensors = (
                            vbatch_tensors-mu)/sigma
                        
                        vloss, preds = sess.run(
                            [model.loss, model.prediction],
                            feed_dict={
                                model.x: vbatch_tensors,
                                model.y_: vbatch_labels,
                                model.keep_prob:1.})
                        # summing up everything 
                        t_vloss += vloss*len(vbatch)
                        t_corrpreds += np.sum(
                            preds==np.argmax(
                                vbatch_labels,axis=0))
                        vsize += len(vbatch)
                    
                    # add the average loss to 
                    # test summary    
                    vLoss = t_vloss / float(vsize)
                    vAcc = t_corrpreds / float(vsize)
                    vL_summary = tf.Summary()
                    vL_summary.value.add(
                        tag='Loss',
                        simple_value=vLoss)
                    valid_writer.add_summary(
                        vL_summary, cnt)
                    vAcc_summary = tf.Summary()
                    vAcc_summary.value.add(
                        tag='Accuracy',
                        simple_value=vAcc)
                    valid_writer.add_summary(
                        vAcc_summary, cnt)
                    
                cnt += 1

def active_finetune(learning_rate,
                    dropout_rate,
                    patch_shape,
                    batch_size,
                    qbatch_size,
                    method_name,
                    stats,
                    init_weights_path):
    """Finetuning a pre-trained model
    by querying from a given sample-set
    of a target data set
    """
    
    # path to data
    img_addrs, mask_addrs = patch_utils.extract_newborn_data_path()
    
    # class of data
    pw_dataset = patch_utils.PatchBinaryData(
        img_addrs[:6],mask_addrs[:6])
    nimg = len(pw_dataset.img_addrs)

    # creating a single large dictionary
    # for the whole data set
    inds_dict, mask_dict = pw_dataset.generate_samples(
        np.arange(nimg), [10,100,10],.2, 'axial')
    
    """preparing pool and test samples"""
    """-------------------------------"""
    # getting relative pool and test indices
    all_paths = list(inds_dict.keys())
    npool = np.sum([len(inds_dict[path]) for
                    path in all_paths[:int(nimg/2)]])
    ntest = np.sum([len(inds_dict[path]) for
                    path in all_paths[int(nimg/2):]])
    
    # creating test and pool dictionary
    # pool indices and mask dictionary
    pool_inds = np.arange(npool)
    prel_dict = patch_utils.locate_in_dict(
        inds_dict, pool_inds)
    pinds_dict = {}
    pmask_dict = {}
    for path in list(prel_dict.keys()):
        pmask_dict[path] = np.array(mask_dict[path])[
            prel_dict[path]]
        pinds_dict[path] = np.array(inds_dict[path])[
            prel_dict[path]]
        
    # test indices and mask dictionary
    test_inds = np.arange(npool, npool+ntest)
    tsrel_dict = patch_utils.locate_in_dict(
        inds_dict, test_inds)
    tsmask_dict = {}
    tsinds_dict = {}
    for path in list(tsrel_dict.keys()):
        tsmask_dict[path] = np.array(mask_dict[path])[
            tsrel_dict[path]]
        tsinds_dict[path] = np.array(inds_dict[path])[
            tsrel_dict[path]]
        
    """An initial fine-tuning
    """
    fine_epochs = 20
    model = get_model(2,
                      dropout_rate,
                      learning_rate,
                      patch_shape)
    model.add_assign_ops(init_weights_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()
        # loading the initial weights
        model.perform_assign_ops(sess)
        
        # inital F-measure
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
        
        Fvec = np.zeros(10+1)
        Fvec[0] = Fm
        print('\n:::::: Initial F-measure: %f'
              % (Fvec[0]))
        
        trinds_dict = {}
        trmask_dict = {}
        """Starting the querying iterations"""
        for t in range(10):
            qrel_dict = PW_NNAL.CNN_query(
                model,
                pinds_dict,
                method_name,
                qbatch_size,
                patch_shape,
                stats,
                sess)

            # modifying dictionaries
            (trinds_dict,
             trmask_dict,
             pinds_dict,
             pmask_dict)=expand_train_dicts(
                 qrel_dict,
                 pinds_dict,
                 pmask_dict,
                 trinds_dict,
                 trmask_dict)


            # fine-tuning
            model.perform_assign_ops(sess)
            for i in range(fine_epochs):
                if i==fine_epochs-1:
                    Fm = PW_train_step(
                        model,
                        dropout_rate,
                        trinds_dict,
                        trmask_dict,
                        patch_shape,
                        batch_size,
                        stats,
                        sess,
                        tsinds_dict,
                        tsmask_dict)
                else:
                    PW_train_step(
                        model,
                        dropout_rate,
                        trinds_dict,
                        trmask_dict,
                        patch_shape,
                        batch_size,
                        stats,
                        sess)
                    print(i,end=',')
                    
            Fvec[t+1] = Fm
            print(':::::: F-measure: %f'
                  % (Fvec[t+1]))
            ntr = np.sum([
                len(trinds_dict[path])
                for path in 
                list(trinds_dict.keys())])
            print('Current train-size: %d'
                  % ntr)

    return Fvec

def PW_train_step(model,
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


def get_model(nclass,
              dropout_rate,
              learning_rate,
              patch_shape):
    """Creating a model for patch-wise
    segmentatio of medical images
    """

    pw_dict = {'conv1':[24, 'conv', [5,5]],
               'conv2':[32, 'conv', [5,5]],
               'max1': [[2,2], 'pool'],
               #'conv3':[32, 'conv', [3,3]],
               #'conv4':[48, 'conv', [3,3]],
               #'max2' :[[2,2], 'pool'],
               'conv3':[48, 'conv', [3,3]],
               'conv4':[96, 'conv', [3,3]],
               'max2' :[[2,2], 'pool'],
               'fc1':[4096,'fc'],
               'fc2':[4096,'fc'],
               'fc3':[nclass,'fc']}
    
    dropout = [[9,10], dropout_rate]
    x = tf.placeholder(
        tf.float32,
        [None, 
         patch_shape[0], 
         patch_shape[1], 
         patch_shape[2]],
                       name='input')
    feature_layer = len(pw_dict) - 2
    
    # the model
    model = NN.CNN(x, pw_dict, 'PatchWise', 
                   feature_layer, dropout)
    # including optimizers
    model.get_optimizer(learning_rate)
    
    return model

def expand_train_dicts(qrel_dict,
                       pinds_dict,
                       pmask_dict,
                       trinds_dict,
                       trmask_dict):
    """Expanding a given training dictionary
    with a set of queries given in form of
    relative indices with respect to the pool
    dictionaries
    
    All the keys of `qrel_dict` should exist
    in the pool dictionary too, but not
    necessarily in the training dictionaries
    """
    
    tr_paths = list(trinds_dict.keys())
    q_paths = list(qrel_dict.keys())

    for path in q_paths:
        # extracting from pool
        sel_pinds = np.array(pinds_dict[
            path])[qrel_dict[path]]
        sel_pmask = np.array(pmask_dict[
            path])[qrel_dict[path]]

        # transferring to the training
        if not(path in tr_paths):
            trinds_dict[path] = []
            trmask_dict[path] = []
        trinds_dict[path] += list(
            sel_pinds)
        trmask_dict[path] += list(
            sel_pmask)
        
        # removing from the pool
        # indices
        new_pinds = np.array(
            pinds_dict[path])
        new_pinds = np.delete(
            new_pinds, 
            qrel_dict[path])
        pinds_dict[path] = list(
            new_pinds)
        # mask
        new_pmask = np.array(
            pmask_dict[path])
        new_pmask = np.delete(
            new_pmask, 
            qrel_dict[path])
        pmask_dict[path] = list(
            new_pmask)
        
    return (trinds_dict,trmask_dict,
            pinds_dict, pmask_dict)
        
def batch_eval(model, 
               inds_dict,
               patch_shape,
               batch_size,
               stats,
               sess,
               varnames,
               mask_dict=None):
    """evaluating a list of variables over
    a set of samples from different images
    in a batch-wise format
    
    :Parameters:
    
        **model** : CNN model
            an object with `prediction`
            and `posterior` properties
    
        **inds_dict** : dictionary
           keys of this dictionary include
           path to images where prediction
           and posterior are to be evaluated
           for some voxels; and the items
           include 3D indices of those
           voxels.

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
    
    # taking path of the images
    imgs_path = list(inds_dict.keys())
    if not(isinstance(varnames, list)):
        varnames = [varnames]
    var_dicts = [] 
    for i in range(len(varnames)):
        var_dicts += [{}]

    mu = stats[0]
    sigma = stats[1]
    for img_path in imgs_path:
        img,_ = nrrd.read(img_path)
        
        # preparing batch indices
        n = len(inds_dict[img_path])
        batch_ends = np.arange(0,n,batch_size)
        if not(batch_ends[-1]==n):
            batch_ends = np.append(
                batch_ends, n)
            
        # going through batches
        for i in range(1,len(batch_ends)):
            # getting the chunk of indices
            batch_inds = np.arange(
                batch_ends[i-1],batch_ends[i])
            b = len(batch_inds)
            # loading tensors
            # (not to be confused with 
            # patch_utils.get_batches())
            batch_tensors = get_patches(
                img, 
                np.array(inds_dict[
                    img_path])[batch_inds],
                patch_shape)
            batch_tensors = (
                batch_tensors-mu)/sigma

            # evaluating the listed variabels
            for j,var in enumerate(varnames):
                # create the array for this image
                # and this variable in the first
                # iteration
                if i==1:
                    var_dicts[j][img_path] = np.zeros(n)
                    
                # evaluate variable for this batch
                model_var = getattr(model, var)
                if var=='loss':
                    """if loss to be evaluated,
                    load the batch labels"""
                    # ---------------------------\
                    if not(mask_dict):
                        raise ValueError(
                            'If loss '+
                            'is to be evaluated '+
                            'the mask dictionary '+
                            'should be given.')
                    batch_labels = np.array(mask_dict[
                        img_path])[batch_inds]
                    hotbatch_labels = np.zeros((2,b))
                    hotbatch_labels[0,batch_labels==0]=1
                    hotbatch_labels[1,batch_labels==1]=1

                    batch_vals = sess.run(
                        model_var,
                        feed_dict={model.x:batch_tensors,
                                   model.y_:hotbatch_labels,
                                   model.keep_prob: 1.})
                else:
                    """evaluating variables with no need
                    to batch labels
                    """
                    batch_vals = sess.run(
                        model_var,
                        feed_dict={model.x:batch_tensors,
                                   model.keep_prob: 1.})
            
                if var=='posteriors':
                    # keeping only posterior probability
                    # of being maksed
                    var_dicts[j][img_path][
                        batch_inds] = batch_vals[1,:]
                else:
                    var_dicts[j][img_path][
                        batch_inds] = batch_vals

            if i%50==0:
                print(i,end=',')
                
            
    return var_dicts

def get_slice_prediction(model,
                         img_path,
                         slice_inds,
                         slice_view,
                         patch_shape,
                         stats,
                         sess,
                         flag='pred'):
    """Generating prediction of all voxels
    in a few slices of a given image
    """
    
    img,_ = nrrd.read(img_path)
    img_shape = img.shape
    
    # preparing 3D indices of the slices
    # ---------------------------------
    # first preparing 2D single indices
    if slice_view=='sagittal':
        nvox_slice=img_shape[1]*img_shape[2]
        slice_shape = img[0,:,:].shape
    elif slice_view=='coronal':
        nvox_slice=img_shape[0]*img_shape[2]
        slice_shape = img[:,0,:].shape
    elif slice_view=='axial':
        nvox_slice=img_shape[0]*img_shape[1]
        slice_shape = img[:,:,0].shape        
        
    inds_2D = np.arange(0, nvox_slice)
    
    # single to multiple 2D indices
    # (common for all slices)
    multiinds_2D = np.unravel_index(
        inds_2D, slice_shape)
    
    slice_evals = []
    for i in range(len(slice_inds)):
        extra_inds = np.ones(
            len(inds_2D),
            dtype=int)*slice_inds[i]

        # multi 2D to multi 3D indices
        if slice_view=='sagittal':
            multiinds_3D = (extra_inds,) + \
                              multiinds_2D
        elif slice_view=='coronal':
            multiinds_3D = multiinds_2D[:1] +\
                              (extra_inds,) +\
                              multiinds_2D[1:]
        elif slice_view=='axial':
            multiinds_3D = multiinds_2D +\
                           (extra_inds,)
        
        # multi 3D to single 3D indices
        inds_3D = np.ravel_multi_index(
            multiinds_3D, img_shape)
        # get the prediction for this slice
        inds_dict = {img_path: inds_3D}
        if flag=='pred':
            evals,_ = get_prediction(
                model,
                inds_dict,
                patch_shape,
                stats,
                sess,
                'pred')
        elif flag=='post':
            _,evals = get_prediction(
                model,
                inds_dict,
                patch_shape,
                stats,
                sess,
                'post')
            
        # prediction map
        eval_map = np.zeros(slice_shape)
        eval_map[multiinds_2D] = evals
        slice_evals += [eval_map]

        print('... %d done in %d'% 
              (i,len(slice_inds)))

    return slice_evals


def get_accuracy(preds, labels):
    
    n = len(preds)
    labels = np.argmax(labels,axis=0)
    
    return np.sum(preds==labels) / float(n)

def get_Fmeasure(preds, mask):
    
    # computing total TPs, Ps, and
    # TPFPs (all positives)
    if isinstance(preds, dict):
        P  = 0
        TP = 0
        TPFP = 0
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

def get_patches(img, inds, patch_shape):
    """Extacting patches around a given 
    set of 3D indices 
    """
    
    # padding the image with radii
    rads = np.zeros(3,dtype=int)
    for i in range(3):
        rads[i] = int((patch_shape[i]-1)/2.)
            
    padded_img = np.pad(
        img, 
        ((rads[0],rads[0]),
         (rads[1],rads[1]),
         (rads[2],rads[2])),
        'constant')

    # computing 3D coordinates of the samples
    # in terms of the original image shape
    multi_inds = np.unravel_index(
        inds, img.shape)
    
    b = len(inds)
    batch = np.zeros((b,)+patch_shape)
    for i in range(b):
        # adjusting the multi-coordinates 
        # WITH padded margins
        center = [
            multi_inds[0][i]+rads[0],
            multi_inds[1][i]+rads[1],
            multi_inds[2][i]+rads[2]]
        
        patch = padded_img[
            center[0]-rads[0]:
            center[0]+rads[0]+1,
            center[1]-rads[1]:
            center[1]+rads[1]+1,
            center[2]-rads[2]:
            center[2]+rads[2]+1]
        
        batch[i,:,:,:] = patch
        
    return batch



    
