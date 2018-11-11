import MNIST_scripts_IF
import tensorflow as tf
import numpy as np
import NN_extended
import create_NN
import imageio
import shutil
import nrrd
import h5py
import pdb
import os


# ----------------------------------------------
#  Initial Training of Tiramisu wit CamVid Data
# ----------------------------------------------

def test_model(model, sess, 
               img_paths, grnd_paths,
               batcher,
               num_img=100,
               void_label=None):

    accs = []
    b=3

    n = len(img_paths)
    h,w = [model.x.shape[1].value,
           model.x.shape[2].value]
    batches = NN_extended.gen_batch_inds(num_img, b)
    rand_inds = np.random.randint(0,len(img_paths), num_img)
    for batch_inds in batches:
        batch_img_paths = [img_paths[i] for i in rand_inds[batch_inds]]
        batch_grnd_paths = [grnd_paths[i] for i in rand_inds[batch_inds]]
        batch_X, batch_grnd = batcher(
            batch_img_paths,batch_grnd_paths,[h,w])

        # prediction
        P = sess.run(model.posteriors, feed_dict={model.x:batch_X,
                                                  model.keep_prob:1.,
                                                  model.is_training:False})

        for i in range(len(batch_inds)):
            if void_label is not None:
                void_vol = np.sum(batch_grnd[i,:,:]==void_label)
            else:
                void_vol = 0
            preds = np.argmax(P[i,:,:,:], axis=-1)
            intersect_vol = np.sum(preds==batch_grnd[i,:,:])
            accs += [intersect_vol / (np.prod(preds.shape)-void_vol)]
            
    return accs

def prepare_batch_CamVid(img_paths, grnd_paths, img_shape):

    h,w = img_shape
    batch_X = np.zeros((len(img_paths),h,w,3))
    batch_grnd = np.zeros((len(img_paths),h,w))

    for i in range(len(img_paths)):
        # image
        img = imageio.imread(img_paths[i])
        crimg, init_h, init_w = random_crop(img,h,w)
        batch_X[i,:,:,:] = crimg
        # ground truth
        grnd = imageio.imread(grnd_paths[i])
        cgrnd,_,_ = random_crop(grnd,h,w,init_h,init_w)
        batch_grnd[i,:,:] = cgrnd

    return batch_X, batch_grnd

def prepare_batch_BrVol(img_paths, mask_addrs, img_shape):

    h,w = img_shape
    m = len(img_paths[0])
    batch_X = np.zeros((len(img_paths),h,w,m))
    batch_grnd = np.zeros((len(img_paths),h,w))

    for i in range(len(img_paths)):
        grnd = nrrd.read(mask_addrs[i])[0]
        slice_ind = np.random.randint(grnd.shape[-1])
        grnd = grnd[:,:,slice_ind]
        for j in range(m):
            # image (j'th modality)
            img = nrrd.read(img_paths[i][j])[0]
            img = img[:,:,slice_ind]
            if j==0:
                crimg, init_h, init_w = random_crop(img,h,w)
            else:
                crimg,_,_ = random_crop(img,h,w,init_h,init_w)
            batch_X[i,:,:,j] = crimg
        # ground truth
        cgrnd,_,_ = random_crop(grnd,h,w,init_h,init_w)
        batch_grnd[i,:,:] = cgrnd

    return batch_X, batch_grnd


def random_crop(img,h,w,init_h=None,init_w=None):
    '''Assume the given image has either shape [h,w,channels] or [h,w]
    '''

    if init_h is None:
        if img.shape[0]==h:
            init_h=0
        else:
            init_h = np.random.randint(0, img.shape[0]-h)
        if img.shape[1]==w:
            init_w=0
        else:
            init_w = np.random.randint(0, img.shape[1]-w)
    if len(img.shape)==3:
        cropped_img = img[init_h:init_h+h, init_w:init_w+w, :]
    elif len(img.shape)==2:
        cropped_img = img[init_h:init_h+h, init_w:init_w+w]

    return cropped_img, init_h, init_w


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
