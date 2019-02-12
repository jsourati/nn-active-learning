import numpy as np
import itertools
import shutil
import nrrd
import copy
import pdb
import os

import NN_extended
from datasets.utils import gen_batch_inds
from patch_utils import extract_Hakims_data_path

def eval_metrics(model, sess, 
                 dat_gen, 
                 slices=50,
                 update=True,
                 alt_attr=None):
    """ The alternative attribute will be used if `alt_attr`
    is given; otherwise `model.valid_metrics` will be used 
    """


    # metrics
    if alt_attr is not None:
        assert hasattr(model,alt_attr), 'The alternative attribute'+\
            ' does not exist.'
        valid_metrics = getattr(model, alt_attr)
    else:
        valid_metrics = model.valid_metrics
    eval_metrics = list(valid_metrics.keys())

    op_dict = {}
    eval_dict = {}
    model_inclusion = False
    MT_model_inclusion = False
    if 'av_acc' in eval_metrics:
        op_dict.update({'accs': model.posteriors})
        eval_dict.update({'accs': []})
        model_inclusion = True
    if 'av_F1' in eval_metrics:
        op_dict.update({'F1s': model.posteriors})
        eval_dict.update({'F1s': []})
        model_inclusion = True
    if 'av_loss' in eval_metrics:
        op_dict.update({'av_loss': model.loss})
        eval_dict.update({'av_loss': 0.})
        model_inclusion = True

    vol = 0
    for _ in range(slices):
        batch_X, batch_mask = dat_gen()
        b = batch_X.shape[0]

        feed_dict = {}
        if model_inclusion:
            feed_dict.update({model.x:batch_X,
                              model.y_:batch_mask,
                              model.keep_prob:1.,
                              model.is_training:False})
        if hasattr(model, 'teacher'):
            feed_dict.update({model.teacher.keep_prob:1.,
                              model.teacher.is_training:False})

        results = sess.run(op_dict, feed_dict=feed_dict)

        for key, val in results.items():
            if 'loss' in key:
                # eval_dict[key]    : total av. loss computed so far
                # val==results[key] : the newest av. loss computed
                eval_dict[key] = (vol*eval_dict[key]+val*b) / (vol+b)

            if 'accs' in key:
                # val in this case is actually posterior
                preds = np.argmax(val, axis=-1)
                nohot_batch_mask = np.argmax(batch_mask, axis=-1)
                for i in range(b):
                    intersect_vol = np.sum(preds[i,:,:]==nohot_batch_mask[i,:,:])
                    eval_dict['accs'] = eval_dict['accs'] + \
                                        [intersect_vol/(np.prod(preds.shape[1:]))]
            if 'F1s' in key:
                # val in this case is actually posterior
                preds = np.argmax(val, axis=-1)
                nohot_batch_mask = np.argmax(batch_mask, axis=-1)
                for i in range(b):
                    eval_dict['F1s'] = eval_dict['F1s'] + \
                                       [F1_score(preds[i,:,:], nohot_batch_mask[i,:,:])]
                
        vol += b

    if update:
        for metric in eval_metrics:
            if metric=='av_acc':
                valid_metrics[metric] += [np.mean(eval_dict['accs'])]
            elif metric=='std_acc':
                valid_metrics[metric] += [np.std(eval_dict['accs'])]
            elif metric=='av_F1':
                valid_metrics[metric] += [np.mean(eval_dict['F1s'])]
            elif metric=='std_F1':
                valid_metrics[metric] += [np.std(eval_dict['F1s'])]

            elif 'loss' in metric:
                valid_metrics[metric] += [eval_dict[metric]]
    else:
        return eval_dict

def full_slice_segment(model,sess,img_paths_or_mats, data_reader, op='prediction'):

    # size of batch
    b = 4

    if isinstance(img_paths_or_mats, list):
        m = len(img_paths_or_mats)
        if isinstance(img_paths_or_mats[0], np.ndarray):
            h,w,z = img_paths_or_mats[0].shape
            paths_or_mats = 'mats'
        else:
            h,w,z = data_reader(img_paths_or_mats[0]).shape
            paths_or_mats = 'paths'
    else:
        m = 1
        if isinstance(img_paths_or_mats, np.ndarray):
            h,w,z = img_paths_or_mats.shape
            paths_or_mats = 'mats'
        else:
            h,w,z = data_reader(img_paths_or_mats).shape
            paths_or_mats = 'paths'

    hx,wx = [model.x.shape[1].value, model.x.shape[2].value]
    assert h==hx and w==wx, 'Shape of data and model.x should match.'

    # loading images
    # m: number of input channels
    if paths_or_mats=='mats':
        img_list = img_paths_or_mats
    else:
        img_list = []
        for i in range(m):
            if m==1:
                img_list = [data_reader(img_paths)] 
            else:
                img_list += [data_reader(img_paths[i])]

    # performing the op for all slices in batches
    if op=='prediction':
        out_tensor = np.zeros((h,w,z))
    elif op=='loss':
        out_tensor = 0.
        cnt = 0
    elif op=='output' and (model.AU_4U or model.AU_4L):
        c = model.output.shape[-1].value
        out_tensor = np.zeros((c,h,w,z))
    else:
        c = model.y_.shape[-1].value  # = model.class_num in new version
        out_tensor = np.zeros((c,h,w,z))
    batches = gen_batch_inds(z, b)
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
        elif op=='AU_vals':
            P = sess.run(model.AU_vals, feed_dict=feed_dict)
            out_tensor[:,:,:,batch_inds] = np.swapaxes(P,0,3)
        elif op=='output':
            P = sess.run(model.output, feed_dict=feed_dict)
            out_tensor[:,:,:,batch_inds] = np.swapaxes(P,0,3)
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

def full_eval(models_dict, 
              sess, 
              save_path, 
              dat):
        
    n = len(dat.img_addrs[dat.mods[0]])
    accs = np.zeros(n)
    Fscores = np.zeros(n)
    for i in range(n):
        mask = dat.reader(dat.mask_addrs[i])
        shape = mask.shape[:2]
        img_paths = [dat.img_addrs[mod][i] for mod in dat.mods]

        model_key = '{}'.format(shape)
        model = models_dict[model_key]
        preds = full_slice_segment(model,sess,img_paths, dat.reader)

        accs[i] = np.sum(preds==mask) / np.prod(mask.shape)
        Fscores[i] = F1_score(preds, mask)
        
        np.savetxt(os.path.join(save_path, 'accs.txt'), accs)
        np.savetxt(os.path.join(save_path, 'Fscores.txt'), Fscores)
        
    train_F_stats = [np.mean(Fscores[dat.train_inds]),
                     np.std(Fscores[dat.train_inds])]
    test_F_stats = [np.mean(Fscores[dat.test_inds]),
                    np.std(Fscores[dat.test_inds])]
    np.savetxt(os.path.join(save_path, 'train_F_stats.txt'), 
               train_F_stats)
    np.savetxt(os.path.join(save_path, 'test_F_stats.txt'), 
               test_F_stats)
    train_A_stats = [np.mean(accs[dat.train_inds]),
                     np.std(accs[dat.train_inds])]
    test_A_stats = [np.mean(accs[dat.test_inds]),
                    np.std(accs[dat.test_inds])]
    np.savetxt(os.path.join(save_path, 'train_A_stats.txt'), 
               train_A_stats)
    np.savetxt(os.path.join(save_path, 'test_A_stats.txt'), 
               test_A_stats)

    if np.any(dat.L_indic==0):
        labeled_size = np.sum(dat.L_indic)
        labeled_inds = dat.train_inds[:labeled_size]
        unlabeled_inds = dat.train_inds[labeled_size:]

        labeled_F_stats = [np.mean(Fscores[labeled_inds]),
                           np.std(Fscores[labeled_inds])]
        unlabeled_F_stats = [np.mean(Fscores[unlabeled_inds]),
                             np.std(Fscores[unlabeled_inds])]
        np.savetxt(os.path.join(save_path, 'labeled_F_stats.txt'), 
                   labeled_F_stats)
        np.savetxt(os.path.join(save_path, 'unlabeled_F_stats.txt'), 
                   unlabeled_F_stats)
        labeled_A_stats = [np.mean(accs[labeled_inds]),
                           np.std(accs[labeled_inds])]
        unlabeled_A_stats = [np.mean(accs[unlabeled_inds]),
                             np.std(accs[unlabeled_inds])]
        np.savetxt(os.path.join(save_path, 'labeled_A_stats.txt'), 
                   labeled_A_stats)
        np.savetxt(os.path.join(save_path, 'unlabeled_A_stats.txt'), 
                   unlabeled_A_stats)

def F1_score(preds,labels):

    TP = np.sum(preds*labels)
    P = np.sum(labels)
    TPFP = np.sum(preds)

    return 2*TP/(P+TPFP) if P+TPFP!=0. else 0.

def models_dict_for_different_sizes(model_builder,
                                    dat):
    """Form a dictionary of FCN models, which has a model
    for each image size exists in the given data set

    Only one field of data class `dat` will be used, and that
    is `img_addrs` that contains image paths of all modalities.

    Model builder should be a function that only takes an
    input size and a model name, and returns a model object 
    that accepts inputs of the given size.
    """

    shapes = []
    for i in range(len(dat.img_addrs[dat.mods[0]])):
        img = dat.reader(dat.img_addrs[dat.mods[0]][i])
        shapes += [img.shape[:2]]
    shapes = np.unique(set(shapes))[0]

    models_dict = {}
    for shape in shapes:
        key = str(shape)
        model_name = '{}x{}'.format(shape[0],shape[1])
        models_dict[key] = model_builder(list(shape), model_name)
        models_dict[key].add_assign_ops()

    return models_dict
