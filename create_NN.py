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

def FCDenseNet_103Layers(input_shape, c, model_name, 
                         probes=[[],[]], **kwargs):
    """Also known as, Tiramisu network with 
    103 layers
    """
    
    # growth rate
    k = 16

    # first layer
    pw_dict = {'first': ['conv', [48, [3,3]], 'MA']}

    
    ''' DB+TD  '''
    # number of Dense block layers
    Ls = [4, 5, 7, 10, 12]
    for i in range(len(Ls)):
        # dense block
        DB = {'DB%d_%d'%(i,j): ['conv', [k, [3,3]], 'BAM']
              for j in range(Ls[i])}
        pw_dict.update(DB)
        # transition down
        nfmap = 48+np.sum(Ls[:i+1])*k
        TD = {'T_%d'%i: ['conv', [nfmap, [1,1]], 'BMA'],
              'pool_%d'%i: ['pool', [2,2]]}
        pw_dict.update(TD)

    ''' Bottleneck Dense Block''' 
    L = 15
    BT = {'BottleDB_%d'%j: ['conv', [k, [3,3]], 'BAM']
          for j in range(L)}
    pw_dict.update(BT)

    ''' TU+DB '''
    Ls = np.flip(Ls+[15], 0)
    for i in range(1,len(Ls)):
        # transition up
        nfmap = Ls[i-1]*k
        TU = {'TU_%d'%(i-1): ['conv_transpose', 
                          [nfmap, [3,3], [2,2]], 'M']}
        pw_dict.update(TU)
        # dense block
        DB = {'DB%d_%d'%(5+i-1,j): ['conv', [k, [3,3]], 'BAM']
              for j in range(Ls[i])}
        pw_dict.update(DB)

    # last layer
    pw_dict.update({'last': ['conv', [c,[1,1]], 'M']})


    ''' Establishing the Skip Connections '''
    layer_names = np.array(list(pw_dict.keys()))
    stype = 'con'    # concatenation
    skips = []

    # intra-DB skip connections of downward path
    for i in range(5):
        # starting index
        start_ind = np.where('DB%d_0'%i==layer_names)[0][0]
        # number of layers
        L = np.sum(['DB%d'%i in name for 
                    name in layer_names])
        for j in range(L):
            skips += [[start_ind-1+j,
                       list(np.arange(start_ind+1+j,
                                      start_ind+L+1)),
                       stype]]
    # bottleneck DB
    start_ind = np.where('BottleDB_0'==layer_names)[0][0]
    L = np.sum(['BottleDB' in name for name in layer_names])
    # for bottleneck, the input of the first layer (or output
    # of pool_4) is connected to all intermediate layers
    # but the last output (input of TU_0)
    skips += [[start_ind-1, 
               list(np.arange(start_ind+1,
                              start_ind+L)), stype]]
    # but the rest will be connected to the last one too
    for j in range(1,L):
        skips += [[start_ind-1+j, 
                   list(np.arange(start_ind+1+j,
                                  start_ind+L+1)),
                  stype]]

    # inter-DB skip connections
    skipped_nodes = np.array([skips[i][0] for i in
                              range(len(skips))])
    # NOTE1: we know that output nodes of DBs in downward
    # path are not already skipped somewhere else, hence
    # we don't need to check with an if
    #
    # NOTE2: more importantly, for source nodes, we only 
    # save their outuputs before combining them with other
    # sources. For example, in the first connection below
    # the output of DB4_11 (end node of DB4) has to be
    # connected to the input of DB5. However, the output
    # node of DB4 (which is the same as input node of
    # T_4) is also combined with many other previous
    # nodes. Hence, what we actually have to do here is
    # to combine output node of DB4_11 together with 
    # outputs of all other previous nodes that have
    # already combined with input of T_4. In other words,
    # input of DB5_0 should be the destination of (output
    # of) DB4_11 and all connections of T_4
    # 
    # output of DB4_11 + connections to T_4 
    # --> 
    # output of TU_0 (input of DB5_0)
    DB4_end_node = np.where(layer_names=='DB4_11')[0][0]
    DB5_start_node = np.where(layer_names=='DB5_0')[0][0]
    T4_node = np.where(layer_names=='T_4')[0][0]
    # go through all nodes, and if T_4 was in their
    # destination, put DB5_0 as a destination too
    for i in range(len(skips)):
        if T4_node in skips[i][1]:
            skips[i][1] += [DB5_start_node]
    skips += [[DB4_end_node,[DB5_start_node], stype]]

    # output of DB3 + connections to T_3
    # --> 
    # output of TU_1 (input of DB6_0)
    DB3_end_node = np.where(layer_names=='DB3_9')[0][0]
    DB6_start_node = np.where(layer_names=='DB6_0')[0][0]
    T3_node = np.where(layer_names=='T_3')[0][0]
    for i in range(len(skips)):
        if T3_node in skips[i][1]:
            skips[i][1] += [DB6_start_node]
    skips += [[DB3_end_node, [DB6_start_node], stype]]

    # output of DB2 + connections to T_2 
    # --> 
    # output of TU_2 (input of DB7_0)
    DB2_end_node = np.where(layer_names=='DB2_6')[0][0]
    DB7_start_node = np.where(layer_names=='DB7_0')[0][0]
    T2_node =  np.where(layer_names=='T_2')[0][0]
    for i in range(len(skips)):
        if T2_node in skips[i][1]:
            skips[i][1] += [DB7_start_node]
    skips += [[DB2_end_node, [DB7_start_node], stype]]

    # output of DB1 + connections to T_1 
    # -->
    # output of TU_3 (input of DB8_0)
    DB1_end_node = np.where(layer_names=='DB1_4')[0][0]
    DB8_start_node = np.where(layer_names=='DB8_0')[0][0]
    T1_node = np.where(layer_names=='T_1')[0][0]
    for i in range(len(skips)):
        if T1_node in skips[i][1]:
            skips[i][1] += [DB8_start_node]
    skips += [[DB1_end_node, [DB8_start_node], stype]]

    # output of DB0 + connections to T_0
    # --> 
    # output of TU_4 (input of DB9_0)
    DB0_end_node = np.where(layer_names=='DB0_3')[0][0]
    DB9_start_node = np.where(layer_names=='DB9_0')[0][0]
    T0_node = np.where(layer_names=='T_0')[0][0]
    for i in range(len(skips)):
        if T0_node in skips[i][1]:
            skips[i][1] += [DB9_start_node]
    skips += [[DB0_end_node, [DB9_start_node], stype]]

    # After estalishing the skip connections between the 
    # downward and upward paths, for the intra-DB skips of
    # the upward path. It is different than the intra-DB skips
    # of downward path, because here for connecting the 
    # layers inside a DB, the first layer itself has 
    # combination of outputs of many other layers from
    # the downward path that should be taken into account.
    # For instance, skips of DB5, we should also consider layers
    # of DB4, which hare connected to the input of DB5_0
    for i in range(5,9):
        # starting index
        start_ind = np.where('DB%d_0'%i==layer_names)[0][0]
        # first do the connections to the last layer
        # because this one does not receive the first layer
        # number of layers
        L = np.sum(['DB%d'%i in name for 
                    name in layer_names])
        skips += [[start_ind-1, 
                   list(np.arange(start_ind+1,
                                  start_ind+L)), stype]]
        for j in range(1,L):
            skips += [[start_ind-1+j,
                       list(np.arange(start_ind+1+j,
                                      start_ind+L+1)),
                       stype]]
        # also add the intermediate layers as destination
        # of the skipped connections (i.e. those layers
        # that already had the start_ind as their destination)
        for j in range(len(skips)):
            if start_ind in skips[j][1]:
                skips[j][1] += list(np.arange(start_ind+1,
                                              start_ind+L))

    # for DB9 (last DB) do the same thing, except its input
    # should also be connected to the last layer
    start_ind = np.where('DB9_0'==layer_names)[0][0]
    L = np.sum(['DB%d'%i in name for name in layer_names])
    for j in range(L):
        skips += [[start_ind-1+j,
                   list(np.arange(start_ind+1+j,
                                  start_ind+L+1)),stype]]
    for j in range(len(skips)):
        if start_ind in skips[j][1]:
            skips[j][1] += list(np.arange(start_ind+1,
                                          start_ind+L+1))

    # sorting the skips in terms of the sources
    sort_inds = np.argsort([skips[i][0] for i in range(len(skips))])
    sorted_skips = []
    for ind in sort_inds:
        sorted_skips += [skips[ind]]


    ''' Specifying Drop-out Layers '''
    # including all the layers, except max-pooling and
    # upscaling layers
    dp_layers = np.arange(len(pw_dict)).tolist()
    no_dp = ['pool_0', 'pool_1', 'pool_2', 'pool_3',
             'pool_4', 'TU_0', 'TU_1', 'TU_2', 'TU_3',
             'TU_4']
    for name in no_dp:
        loc = np.where(layer_names==name)[0][0]
        dp_layers.remove(loc)
    dp_rate = 0.1

    
    ''' Creating the Model '''
    x = tf.placeholder(tf.float32, [None,]+input_shape)
    model = NN_extended.CNN(x, pw_dict, model_name, 
                            sorted_skips,None,
                            [dp_layers, dp_rate],
                            probes,
                            **kwargs)

    return model
