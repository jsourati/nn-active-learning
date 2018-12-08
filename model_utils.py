import numpy as np
import itertools
import shutil
import nrrd
import h5py
import copy
import pdb

import NN_extended

def eval_metrics(model, sess, dat_gen, slices=50):

    
    accs = []
    av_loss = 0.
    
    for _ in range(slices):
        batch_X, batch_mask = dat_gen()
            
        if hasattr(model, 'MT'):
            feed_dict={model.MT.x:batch_X,
                       model.MT.y_:batch_mask,
                       model.MT.keep_prob:1.,
                       model.MT.is_training:False}

            L,P = sess.run([model.MT.loss,model.MT.posteriors], 
                       feed_dict=feed_dict)
        else:
            feed_dict={model.x:batch_X,
                       model.y_:batch_mask,
                       model.keep_prob:1.,
                       model.is_training:False}
            L,P = sess.run([model.loss,model.posteriors], 
                           feed_dict=feed_dict)

        av_loss = (len(accs)*av_loss+L*batch_X.shape[0]) / (len(accs)+batch_X.shape[0])
        nohot_batch_mask = np.argmax(batch_mask, axis=-1)
        preds = np.argmax(P, axis=-1)
        for i in range(preds.shape[0]):
            intersect_vol = np.sum(preds[i,:,:]==nohot_batch_mask[i,:,:])
            accs += [intersect_vol / (np.prod(preds.shape[1:]))]
            
    model.valid_metrics['av_acc'] += [np.mean(accs)]
    model.valid_metrics['std_acc'] += [np.std(accs)]
    model.valid_metrics['av_loss'] += [av_loss]

def full_slice_segment(model,sess,img_paths, op='prediction'):

    # size of batch
    b = 3

    if isinstance(img_paths, list):
        m = len(img_paths)
        h,w,z = nrrd.read(img_paths[0])[0].shape
    else:
        m = 1
        h,w,z = nrrd.read(img_paths).shape

    hx,wx = [model.x.shape[1].value, model.x.shape[2].value]
    assert h==hx and w==wx, 'Shape of data and model.x should match.'

    # loading images
    # m: number of input channels
    img_list = []
    for i in range(m):
        if m==1:
            img_list = [nrrd.read(img_paths)[0]] 
        else:
            img_list += [nrrd.read(img_paths[i])[0]]

    # performing the op for all slices in batches
    if op=='prediction':
        out_tensor = np.zeros((h,w,z))
    elif op=='loss':
        out_tensor = 0.
        cnt = 0
    else:
        c = model.y_.shape[-1].value
        out_tensor = np.zeros((c,h,w,z))
    batches = NN_extended.gen_batch_inds(z, b)
    for batch in batches:
        batch_inds = np.sort(batch)
        batch_X = np.zeros((len(batch_inds),h,w,m))
        for j in range(m):
            batch_X[:,:,:,j] = np.rollaxis(img_list[j][:,:,batch_inds], 
                                           axis=-1)
        feed_dict = {model.x:batch_X, model.keep_prob:1., model.is_training:False}
        if op=='prediction':
            P = sess.run(model.posteriors, feed_dict=feed_dict)
            batch_preds = np.argmax(P, axis=-1)
            out_tensor[:,:,batch_inds] = np.rollaxis(batch_preds,axis=0,start=3)
        elif op=='posterior':
            P = sess.run(model.posteriors, feed_dict=feed_dict)
            out_tensor[:,:,:,batch_inds] = np.swapaxes(P,0,3)
        elif op=='MC-posterior':
            feed_dict[model.keep_prob] = 1-model.dropout_rate
            T = 10
            av_P = sess.run(model.posteriors, feed_dict=feed_dict)
            for i in range(1,T):
                av_P = (i*av_P + sess.run(model.posteriors, feed_dict=feed_dict))/(i+1)
            out_tensor[:,:,:,batch_inds] = np.swapaxes(av_P,0,3)
        elif op=='loss':
            loss = sess.run(model.loss, feed_dict=feed_dict)
            out_tensor = (len(batch)*loss + cnt*out_tensor) / (cnt+len(batch))
            cnt += len(batch)
        elif op=='sigma':
            out = sess.run(model.output, feed_dict=feed_dict)
            out_tensor[:,:,:,batch_inds] = np.swapaxes(out[:,:,:,c:],0,3)
        elif op=='MC-sigma':
            feed_dict[model.keep_prob] = 1-model.dropout_rate
            T = 10
            out = sess.run(model.output, feed_dict=feed_dict)
            av_sigma = out[:,:,:,c:]
            for i in range(1,T):
                out = sess.run(model.output, feed_dict=feed_dict)
                av_sigma = (i*av_sigma + out[:,:,:,c:])/(i+1)
            out_tensor[:,:,:,batch_inds] = np.swapaxes(av_sigma,0,3)

    return out_tensor

def extend_weights_to_aleatoric_mode(weights_path, 
                                     out_channels,
                                     last_layer_name='last'):

    with h5py.File(weights_path,'r') as f:
        W = f['%s/Weight'% last_layer_name].value
    if W.shape[-1]==out_channels:
        print('The weights already match the extended shape.')
        return

    """ creating a new file """
    # preparing the name
    base_dir = weights_path.split('/')[:-1]
    name = weights_path.split('/')[-1].split('.')[0]
    ext_name = name+'_extended.h5'
    new_path = '/'.join(base_dir+[ext_name])
    shutil.copy2(weights_path, new_path)
    

    f = h5py.File(new_path, 'a')
    # weight
    ext_W = np.zeros(W.shape[:-1]+
                     (2*W.shape[-1],))
    ext_W[:,:,:,:W.shape[-1]] = W
    del f['%s/Weight'% last_layer_name]
    f['%s/Weight'% last_layer_name] = ext_W

    # bias
    b = f['%s/Bias'% last_layer_name]
    ext_b = np.zeros(2*len(b))
    ext_b[:len(b)] = b
    del f['%s/Bias'% last_layer_name]
    f['%s/Bias'% last_layer_name] = ext_b

    f.close()

# -----------------------------------------------------------------
# Older Functions (related to influence function, partial finetunig)
# -----------------------------------------------------------------

def keep_k_largest_from_LoV(LoV, k):
    """Generating a binary mask with the same structure
    as the input (which is a list of variables) such that
    the largest k values of the variables get 1 value
    and the rest 0
    """
    
    # length of all variables
    Ls = [np.prod(LoV[i].shape) for i in range(len(LoV))]

    # appending everything together (and putting
    # a minus behind them) and arg-sorting
    app_LoV = []
    for i in range(len(LoV)):
        app_LoV += np.ravel(-LoV[i]).tolist()
    sort_inds = np.argsort(app_LoV)[:k]
    
    local_inds = patch_utils.global2local_inds(
        sort_inds,Ls)
    non_empty_locs = np.array([len(local_inds[i]) for 
                               i in range(len(local_inds))])
    non_empty_locs = np.where(non_empty_locs>0)[0]

    # generating the mask
    bmask = [np.zeros(LoV[i].shape) for i in range(len(LoV))]
    for i in non_empty_locs:
        multinds = np.unravel_index(local_inds[i],LoV[i].shape)
        bmask[i][multinds] = 1

    return bmask, non_empty_locs

def threshold_LoV(LoV, thr):
    """Generating a binary mask with the same size as the
    LoV (List of Variables) such that the variables whose
    values are larger than the threshold get one, and zero 
    otherwise
    """

    bmask = [np.zeros(LoV[i].shape) for i in range(len(LoV))]
    for i in range(len(LoV)):
        bmask[i][LoV[i]>=thr] = 1

    return bmask

def get_LwF(model):
    """Taking for which a loss has been already defined,
    and modifying it to LwF (learning without forgetting)

    REMARK: this function needs model.get_optimizer() to be
        called beforehand 

    CAUTIOUS: modify it for FCNs
    """

    # needs introducing two hyper-parameters to model
    model.lambda_o = tf.placeholder(tf.float32)
    model.T = tf.placeholder(tf.float32)

    # defining output of the previous model
    model.y__ = tf.placeholder(tf.float32, model.y_.get_shape())

    # knowledge distillation (soft soft-max)
    soft_target = tf.nn.softmax(tf.transpose(
        tf.divide(model.y__, model.T)))
    loss_old_term = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=soft_target, 
            logits=tf.transpose(tf.divide(model.output, model.T))))

    model.LwF_loss = tf.add(model.loss, 
                            tf.multiply(loss_old_term,
                                        model.lambda_o))

    if len(model.train_layers)==0:
        model.LwF_train_step = model.optimizer.minimize(
            model.LwF_loss)
    else:
        var_list = []
        for layer in model.train_layers:
            var_list += model.var_dict[layer]
        model.LwF_train_step = model.optimizer.minimize(
            model.LwF, var_list=var_list)

def LLFC_hess(model,sess,feed_dict):
    """Explicit Hessian matrix of the loss with 
    respect to the last (FC) layer when the loss
    is the soft-max and the last layer does not
    have any additional activation except this
    soft-max
    """

    # input to the last layer (u)
    u = sess.run(model.feature_layer,
                 feed_dict=feed_dict)
    d = u.shape[0]

    # the class probabilities
    pi = sess.run(model.posteriors,
                  feed_dict=feed_dict)

    # A(pi)
    c = pi.shape[0]
    repM = np.repeat(pi,c,axis=1) - np.eye(c)
    A = np.diag(pi[:,0]) @ repM.T

    # Hessian
    H = np.zeros(((d+1)*c, (d+1)*c))
    H[:c*d,:c*d] = np.kron(A, np.outer(u,u))
    H[:c*d,c*d:] = np.kron(A,u)
    H[c*d:,:c*d] = np.kron(A,u.T)
    H[c*d:,c*d:] = A

    return H

def LLFC_grads(model, sess, feed_dict, labels=None):
    """General module for computing gradients
    of the log-loss with respect to parameters
    of the (FC) last layer of the network
    """

    # posteriors (pi)
    pies = sess.run(model.posteriors,
                    feed_dict=feed_dict)
    c,n = pies.shape

    # input to the last layer (u)
    U = sess.run(model.feature_layer,
                 feed_dict=feed_dict)
    d = U.shape[0]

    # term containing [pi_1.u_1 ,..., pi_1.u_d,
    #                  pi_2.u_1 ,..., pi_2.u_d,...]
    rep_pies = np.repeat(pies, d, axis=0)
    rep_U = np.tile(U, (c,1))
    pies_dot_U = rep_pies * rep_U

    flag=0
    if labels is None:
        labels = sess.run(model.prediction,
                          feed_dict=feed_dict)
        flag = 1
    hot_labels = np.zeros((c,n))
    for j in range(c):
        hot_labels[j,labels==j]=1

    # sparse term containing columns
    #         [0,...,0, u_1,...,u_d, 0,...,0].T
    #                   |____ ____|
    #                        v
    #                   y*-th block
    sparse_term = np.repeat(
        hot_labels, d, axis=0) * rep_U

    # dJ/dW
    dJ_dW = sparse_term - pies_dot_U

    # dJ/db
    dJ_db = hot_labels - pies

    if flag==1:
        return np.concatenate(
            (dJ_dW,dJ_db),axis=0), labels
    else:
        return np.concatenate(
            (dJ_dW,dJ_db),axis=0)

def PW_LLFC_grads(model, sess, 
                  expr,
                  all_padded_imgs,
                  img_inds,
                  labels):
    """Computing gradients of the log-likelihoods
    with respect to the parameters of the last
    layer of a given model

    Given labels are not necessarily the true
    labels of the indexed sampels (i.e. not
    necessarily those based on the mask image
    present in `all_padded_imgs`)
    """

    s = len(img_inds)
    n = np.sum([len(img_inds[i]) for i in range(s)])
    d = model.feature_layer.shape[0].value
    c = expr.nclass

    all_pies = np.zeros((c,n))
    all_a = np.zeros((d,n))

    # loading patches
    patches,_ = patch_utils.get_patches_multimg(
        all_padded_imgs, img_inds, 
        expr.pars['patch_shape'], 
        expr.train_stats)

    cnt=0
    for i in range(s):
        # posteriors pie's
        pies = sess.run(model.posteriors,
                        feed_dict={model.x:patches[i],
                                   model.keep_prob:1.})
        all_pies[:,cnt:cnt+len(img_inds[i])] = pies

        # last layer's inputs a^{n1-1}
        a_s = sess.run(model.feature_layer,
                       feed_dict={model.x:patches[i],
                                  model.keep_prob:1.})
        all_a[:,cnt:cnt+len(img_inds[i])] = a_s

        cnt += len(img_inds[i])

    # repeating the matrices
    rep_pies = np.repeat(all_pies, d, axis=0)
    rep_a = np.tile(all_a, (c,1))
    pies_dot_as = rep_pies * rep_a

    # forming dJ / dW_(nl-1)
    term_1 = np.zeros((c*d, n))
    multinds = (np.zeros(n*d, dtype=int), 
                np.zeros(n*d, dtype=int))
    for i in range(n):
        multinds[0][i*d:(i+1)*d] = np.arange(
            labels[i]*d,(labels[i]+1)*d)
        multinds[1][i*d:(i+1)*d] = i
    term_1[multinds] = np.ravel(a_s.T)

    dJ_dW = term_1 - pies_dot_as

    # appending with dJ / db_{nl-1}
    term_1 = np.zeros((c,n))
    multinds = (np.array(labels),
                np.arange(n))
    term_1[multinds] = 1.
    dJ_db = term_1 - pies
    
    # final gradient vectors
    grads = np.concatenate((dJ_dW,dJ_db), axis=0)

    return grads

def diagonal_Fisher(model, sess, batch_dat):
    """ Computing diagonal Fisher values for a batch of data

    The output is in a format similar to `model.var_dict`,
    which is a dictionary with layer names as the keys

    NOTE: for now, there is no batching of the input data,
    hence large batches might give memory errors
    """
    
    # initializing the output dictionary with all-zero arrays
    grads = [model.grads_vars[i][0] for i in range(len(model.grads_vars))]
    diag_F = [np.zeros(grads[i].shape) for i in range(len(grads))]

    # when computing gradients here, be careful about the 
    # binary masks that have to be provided in case of PFT.
    if hasattr(model, 'par_placeholders'):
        X_feed_dict = {model.par_placeholders[i]:
                       np.ones(model.par_placeholders[i].shape)
                       for i in range(len(model.par_placeholders))}
    else:
        X_feed_dict = {}

    # computing diagonal Fisher for each input sample one-by-one
    for i in range(batch_dat[0].shape[0]):
        feed_dict={model.x: batch_dat[0][[i],:,:,:], 
                   model.y_:batch_dat[1][:,[i]], 
                   model.keep_prob:1.}
        feed_dict.update(X_feed_dict)
        Gv = sess.run(grads, feed_dict=feed_dict)

        # updating layers of Fi dictionary with gradients of the
        # the current input sample
        for j in range(len(Gv)):
            diag_F[j] = (i*diag_F[j] + Gv[j]**2) / (i+1)

    return diag_F
