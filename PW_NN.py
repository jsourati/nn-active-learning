from scipy.signal import convolve2d
import tensorflow as tf
import numpy as np
import warnings
import nibabel
import nrrd
import pdb
import os

import NN
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
    train_batches = pw_dataset.get_batches(
        inds_dict,batch_size)
    
    # validation
    valid_imgs = [3]
    vinds_dict, vmask_dict = pw_dataset.generate_samples(
        valid_imgs, [50,50,10],.2, 'axial')
    valid_batches = pw_dataset.get_batches(
        vinds_dict,batch_size)
    
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
                 batch_labels) = pw_dataset.get_batch_vars(
                     inds_dict,
                     mask_dict,
                     batch,
                     patch_shape)

                # normalizing intensities
                batch_tensors = (batch_tensors-mu)/sigma

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
                         vbatch_labels)=pw_dataset.get_batch_vars(
                             vinds_dict,
                             vmask_dict,
                             vbatch,
                             patch_shape)
                        
                        vbatch_tensors = (vbatch_tensors-mu)/sigma
                        
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
               'conv3':[32, 'conv', [3,3]],
               'conv4':[48, 'conv', [3,3]],
               'max2' :[[2,2], 'pool'],
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

def get_prediction(model, 
                   inds_dict,
                   patch_shape,
                   stats,
                   sess):
    """evaluating a list of tensorflow
    variables with batches over a set of 
    samples from different images
    
    """
    
    # taking path of the images
    imgs_path = list(inds_dict.keys())
    preds_dict = {path:[] 
                  for path in imgs_path}

    batch_size = 200
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
        preds = np.zeros(n)
        for i in range(1,len(batch_ends)):
            # getting the chunk of indices
            batch_inds = np.arange(
                batch_ends[i-1],batch_ends[i])
            # loading tensors
            batch_tensors = get_patches(
                img, 
                np.array(inds_dict[
                    img_path])[batch_inds],
                patch_shape)

            batch_tensors = (
                batch_tensors-mu)/sigma

            preds[batch_inds] = sess.run(
                model.prediction,
                feed_dict={model.x:batch_tensors,
                           model.keep_prob: 1.})
            
            #if i%100==0:
            #    print(i,end=',')
            print(i)
            
    return preds
            
def get_accuracy(preds, labels):
    
    n = len(preds)
    labels = np.argmax(labels,axis=0)
    
    return np.sum(preds==labels) / float(n)


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
