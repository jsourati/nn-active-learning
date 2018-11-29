import numpy as np
import itertools
import shutil
import h5py
import copy
import pdb


def eval_metrics(model, sess, dat_gen, run=1):

    
    accs = []
    av_loss = 0.
    
    dat_gens = itertools.tee(dat_gen(), run)
    for gen in dat_gens:
        for batch_X, batch_mask in gen:
            
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
            out_tensor[:,:,:,batch_inds] = np.swapaxes(av_P,i0,3)
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


def EMA_combine_weights_from_models(model_t, model_t_1, alpha, sess):
    """ Exponential moving average of weights with
    decay rate of `alpha`, and when weights are given in CNN models

    The combined weights will be loaded into `model_t_1`
    """

    ops_list = []
    feed_dict = {}
    for layer_name, pars in model_t.var_dict.items():
        for i,par in enumerate(pars):
            # parameter in model_t
            par_name_t = par.name.split('/')[-1]
            par_name_t = par_name_t.split(':')[0]
            # parameter in model_t_1
            par_t_1 = model_t_1.var_dict[layer_name][i]
            par_name_t_1 = par_t_1.name.split('/')[-1]
            par_name_t_1 = par_name_t_1.split(':')[0]
            assert par_name_t==par_name_t_1, 'Mismatch in parameter names' +\
                ' between the two models in EMA weights combination ..'
            val_t = par.eval()
            val_t_1 = par_t_1.eval()
            new_val = alpha*val_t_1 + (1-alpha)*val_t
            
            ops_list += [model_t_1.assign_dict[layer_name][par_name_t_1][0]]
            feed_dict.update({model_t_1.assign_dict[layer_name][par_name_t_1][1]:
                              new_val})

    sess.run(ops_list, feed_dict=feed_dict)

def EMA_combine_weights_from_files(Wt_path, Wt_1_path, alpha):
    """ Exponential moving average of weights with
    decay rate of `alpha`, and when weights are given in .h5 files

    * `Wt_1_path` is the path to the current EMA weight values (theta_{t-1})
    * `Wt_path` is either the path to the new  weight values (theta_t)
    """

    Wt = h5py.File(Wt_path, 'r')

    with h5py.File(Wt_1_path, 'a') as Wt_1:
        for lname, layer in Wt_1.items():
            for pname, par in layer.items():
                new_val = (1-alpha)*Wt[lname][pname].value + \
                    alpha*par.value
                del Wt_1[lname][pname]
                Wt_1[lname][pname] = new_val

    Wt.close()

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

