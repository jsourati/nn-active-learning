import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import f1_score
import linecache
import copy
import h5py
import pdb
import sys
#import cv2
import os

import NN_extended

read_file_path = "/home/ch194765/repos/atlas-active-learning/"
sys.path.insert(0, read_file_path)
#import prep_dat

read_file_path = "/home/ch194765/repos/atlas-active-learning/AlexNet"
sys.path.insert(0, read_file_path)
import alexnet
from alexnet import AlexNet


def create_VGG19(dropout_rate, learning_rate,
                 n_class, grad_layers,
                 train_layers):
    """Creating a VGG19 model using CNN class

    DEPRECATED: needs to be updated to be 
    compatible with extended NN
    """
    
    # architechture dictionary
    vgg_dict = {'conv1':[64, 'conv', [3,3]],
                'conv2':[64, 'conv', [3,3]],
                'max1': [[2,2], 'pool'],
                'conv3':[128, 'conv', [3,3]],
                'conv4':[128, 'conv', [3,3]],
                'max2' :[[2,2], 'pool'],
                'conv5':[256, 'conv', [3,3]],
                'conv6':[256, 'conv', [3,3]],
                'conv7':[256, 'conv', [3,3]],
                'conv8':[256, 'conv', [3,3]],
                'max3': [[2,2], 'pool'],
                'conv9': [512, 'conv', [3,3]],
                'conv10':[512, 'conv', [3,3]],
                'conv11':[512, 'conv', [3,3]],
                'conv12':[512, 'conv', [3,3]],
                'max4': [[2,2], 'pool'],
                'conv13':[512, 'conv', [3,3]],
                'conv14':[512, 'conv', [3,3]],
                'conv15':[512, 'conv', [3,3]],
                'conv16':[512, 'conv', [3,3]],
                'max5':[[2,2], 'pool'],
                'fc1':[4096,'fc'],
                'fc2':[4096,'fc'],
                'fc3':[n_class,'fc']}


    dropout = [[21,22], dropout_rate]
    x = tf.placeholder(tf.float32,
                       [None, 224, 224, 3],
                       name='input')
    feature_layer = len(vgg_dict) - 2
    
    # creating the architecture
    model = CNN(x, vgg_dict, 'VGG19', 
                feature_layer, dropout)

    # forming optimizer and gradient operator
    print('Optimizers..')
    model.get_optimizer(learning_rate, train_layers)
    print('Gradients..')
    model.get_gradients(grad_layers)

    return model

def create_PW1(nclass,
               dropout_rate,
               learning_rate,
               optimizer_name,
               patch_shape):
    """Creating a model for patch-wise
    segmentatio of medical images

    DEPRECATED: needs to be updated to be
    compatible with extended NN
    """

    pw_dict = {'conv1':[24, 'conv', [5,5]],
               'conv2':[32, 'conv', [5,5]],
               'max1': [[2,2], 'pool'],
               'conv3':[48, 'conv', [3,3]],
               'conv4':[96, 'conv', [3,3]],
               'max2' :[[2,2], 'pool'],
               'fc1':[4096,'fc'],
               'fc2':[4096,'fc'],
               'fc3':[nclass,'fc']}
    
    dropout = [[6,7,8], dropout_rate]
    x = tf.placeholder(
        tf.float32,
        [None, 
         patch_shape[0],
         patch_shape[1],
         patch_shape[2]],
        name='input')
    feature_layer = len(pw_dict) - 2
    probes = [5]
    
    # the model
    model = CNN(x, pw_dict, 'PatchWise', 
                feature_layer, 
                dropout, probes)
    # optimizers
    model.get_optimizer(learning_rate, [],
                        optimizer_name)
    # gradients
    model.get_gradients()
    
    return model

def DenseNet_2block(growth_rate, 
                    input_shape,
                    nclass,
                    model_name):

    k = growth_rate

    # starting from the initial layers
    pw_dict = {'conv_0': ['conv', [k, [7,7], [2,2]], 'BMA']}

    """ Architecture """
    # BLOCK 1
    DB_1 = {'conv_D1_1B': ['conv', [4*k, [1,1]], 'BAM'],
            'conv_D1_1' : ['conv', [k, [3,3]], 'BAM'],
            'conv_D1_2B': ['conv', [4*k, [1,1]], 'BAM'],
            'conv_D1_2' : ['conv', [k, [3,3]], 'BAM'],
            'conv_D1_3B': ['conv', [4*k, [1,1]], 'BAM'],
            'conv_D1_3' : ['conv', [k, [3,3]], 'BAM'], 
            'conv_D1_4B': ['conv', [4*k, [1,1]], 'BAM'],
            'conv_D1_4' : ['conv', [k, [3,3]], 'BAM'],
            'conv_D1_5B': ['conv', [4*k, [1,1]], 'BAM'],
            'conv_D1_5' : ['conv', [k, [3,3]], 'BAM'],
            'conv_D1_6B': ['conv', [4*k, [1,1]], 'BAM'],
            'conv_D1_6' : ['conv', [k, [3,3]], 'BAM']}
    pw_dict.update(DB_1)

    # TRANSITION
    pw_dict.update({'conv_T': ['conv', [4*k, [1,1]], 'BAM'], 
                    'pool_T': ['pool',[2,2]]})

    # BLOCK 2
    DB_2 = {'conv_D2_1B': ['conv', [4*k, [1,1]], 'BAM'],
            'conv_D2_1' : ['conv', [k, [3,3]], 'BAM'],
            'conv_D2_2B': ['conv', [4*k, [1,1]], 'BAM'],
            'conv_D2_2' : ['conv', [k, [3,3]], 'BAM'],
            'conv_D2_3B': ['conv', [4*k, [1,1]], 'BAM'],
            'conv_D2_3' : ['conv', [k, [3,3]], 'BAM'], 
            'conv_D2_4B': ['conv', [4*k, [1,1]], 'BAM'],
            'conv_D2_4' : ['conv', [k, [3,3]], 'BAM'],
            'conv_D2_5B': ['conv', [4*k, [1,1]], 'BAM'],
            'conv_D2_5' : ['conv', [k, [3,3]], 'BAM'],
            'conv_D2_6B': ['conv', [4*k, [1,1]], 'BAM'],
            'conv_D2_6' : ['conv', [k, [3,3]], 'BAM']}
    pw_dict.update(DB_2)

    # FINAL LAYERS
    pw_dict.update({'pool_global': ['pool', [2,2]], 
                    'fc_last': ['fc', [nclass]]})


    # SKIP CONNETIONS
    skips = [[0, [3,5,7,9,11,13], 'con'],
             [2, [5,7,9,11,13],'con'],
             [4, [7,9,11,13],'con'],
             [6, [9,11,13], 'con'],
             [8, [11,13], 'con'],
             [10, [13], 'con'], # first block, done
             [14, [17,19,21,23,25,27], 'con'],
             [16, [19,21,23,25,27],'con'],
             [18, [21,23,25,27],'con'],
             [20, [23,25,27], 'con'],
             [22, [25,27], 'con'],
             [24, [27], 'con']]


    """ Creating the Model """
    x = tf.placeholder(tf.float32, [None,]+input_shape)
    model = NN_extended.CNN(x, pw_dict, model_name, 
                            skips,None,
                            [[12, 26, 28], 0.2])

    return model
